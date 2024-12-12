#include <common.h>
#include "kernel_preprocessor.hpp"

#define MSTRINGIFY(...) #__VA_ARGS__
#define FETCH_PER_WI 16
#define BUILD_OPTIONS " -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math"
// #define BUILD_OPTIONS " -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -Dgroup_size="

#define PRINT(...) print(__VA_ARGS__)

template<typename T, typename... Args>
void print(T arg, Args... args) {
    std::cout << arg << " ";
}
static const std::string stringifiedKernels =
#include "gemm.cl"
    ;


double run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel, cl::NDRange &globalSize, cl::NDRange &localSize, uint iters)
{
  bool useEventTimer = true;
  double timed = 0;

  // Dummy calls
  // queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
  for (int i = 0; i < 10; i++)
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
  queue.finish();
  for (uint i = 0; i < iters; i++)
  {
    cl::Event timeEvent;

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &timeEvent);
    queue.finish();
    timed += timeInUS(timeEvent);
  }
  return (timed / static_cast<float>(iters));
}

void gemv_int4_fp32(const uint8_t* A, const float* B, const float* scale, const float* bias, float* C, const int m, const int n, const int group_size) {
  const int scale_per_row = n / group_size;
  for (int row = 0; row < m; row++) {
    float acc = 0;
    for (int s = 0; s < scale_per_row; s++) {
      float sc = scale[row * scale_per_row + s];
      for (int k = s * group_size; k < (s + 1) * group_size; k += 2) {
        char a = A[(row * n + k) / 2];
        float a0 = ((a & 0x0f) - 8) * sc;
        float a1 = (((a >> 4) & 0x0f) - 8) * sc; 
        float b0 = B[k];
        float b1 = B[k + 1];
        acc += a0 * b0 + a1 * b1;
      }
    }
    C[row] = acc + bias[row];
  }
}


// if flag = true then transpose else jusr copy ori to after
void matrix_transpose(const float * ori, float * after, int m, int n, bool flag = true) {
  if (flag) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        after[j * m + i] = ori[i * n + j];
      }
    }
  } 
  else {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        after[i * n + j] = ori[i * n + j];
      }
    }
  }
}


void initializeData(uint8_t* A, float* B, float* scale, float* C, float* bias, const int m, const int n, const int group_size) {
  for (int i = 0; i < m * n / 2; i++) {
    A[i] = 1;//rand() % 256;
  }
  for (int i = 0; i < n; i++) {
    B[i] = 1;//rand() % 1024;

  }
  for (int i = 0; i < n; i++) {
    C[i] = 1;
  }
  for (int i = 0; i < n; i++) {
    bias[i] = 1;//rand() % 1024;
  }
  for (int i = 0; i < m * n / group_size; i++) {
    scale[i] = 1;//rand() % 10;
  }
}

template <typename T>
void row_major2column_major(T* A, const int m, const int k) {
  T* Ac = (T*)malloc(m * k * sizeof(T));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      int src_idx = i * k + j;
      int dst_idx = j * m + k;
      Ac[dst_idx] = A[src_idx];
    }
  }
  memcpy(A, Ac, m * k * sizeof(T));
  free(Ac);
}

int main(int argc, char **argv)
{
    // gemv, A[m, n], B[n,1], C[m,1]
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    const int m = 4096;
    // const int k = 4096;
    const int n = 4096;
    const int group_size = 128;
    const int iters = 1;
    uint8_t* A = (uint8_t*)malloc(m * n * sizeof(uint8_t) / 2);
    float* scale = (float*)malloc(m * n / group_size * sizeof(float));

    float* B = (float*)malloc(n * sizeof(float));
    float* bias = (float*)malloc(m * sizeof(float));
    float* C = (float*)malloc(m * sizeof(float));
    float* C_expect = (float*)malloc(m * sizeof(float));

    initializeData(A, B, scale, C, bias, m, n, group_size);


    // row_major2column_major(A, m, k / 2);
    // row_major2column_major(B, k, n);
    for (size_t p = 0; p < 1/*platforms.size()*/; p++)
    {
        cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[p])(), 0};

        cl::Context ctx(CL_DEVICE_TYPE_ALL, cps);
        vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
        std::string kernel_string = clblast::PreprocessKernelSource(stringifiedKernels);
        std::cout << kernel_string;
        cl::Program::Sources source(1, kernel_string);
        cl::Program prog = cl::Program(ctx, source);
        // std::string compile_opts = BUILD_OPTIONS + std::to_string(group_size);
        std::string compile_opts = BUILD_OPTIONS;
        for (size_t d = 0; d < 1/*devices.size()*/; d++)
        {
            PRINT("*****************THIS IS A NEW DEVICE***********************\n");
            device_info_t devInfo = getDeviceInfo(devices[d]);

            try
            {
                vector<cl::Device> dev = {devices[d]};
                prog.build(dev, compile_opts.c_str());
            }
            catch (cl::Error &error)
            {
                PRINT("Build Log: " + prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[d]) + NEWLINE NEWLINE);
                UNUSED(error);
                PRINT("Error");
                continue;
            }

            PRINT("Device: " + devInfo.deviceName + NEWLINE);
            PRINT("Driver version  : ");
            PRINT(devInfo.driverVersion);
            PRINT(" (" OS_NAME ")" NEWLINE);
            PRINT("Compute units   : ");
            PRINT(devInfo.numCUs);
            PRINT(NEWLINE);
            PRINT("Clock frequency : ");
            PRINT(devInfo.maxClockFreq);
            PRINT(" MHz" NEWLINE);
            PRINT("device name:");
            PRINT(devInfo.deviceName);
            PRINT(NEWLINE);
            PRINT("driver_version: ");
            PRINT(devInfo.driverVersion);
            PRINT(NEWLINE);
            PRINT("compute_units: ");
            PRINT(devInfo.numCUs);
            PRINT(NEWLINE);
            PRINT("clock_frequency: ");
            PRINT(devInfo.maxClockFreq);
            PRINT(NEWLINE);
            PRINT("clock_frequency_unit: MHz");
            // PRINT(TAB TAB "Build Log: ", prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[d]), NEWLINE NEWLINE);

            cl::CommandQueue queue = cl::CommandQueue(ctx, devices[d], CL_QUEUE_PROFILING_ENABLE);

            cl::Buffer ABuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (m * n / 2));
            cl::Buffer scaleBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, m * n / group_size * sizeof(float));
            cl::Buffer BBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, n * sizeof(float));
            cl::Buffer CBuf = cl::Buffer(ctx, CL_MEM_READ_WRITE, m * sizeof(float));
            cl::Buffer biasBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, m * sizeof(float));

            queue.enqueueWriteBuffer(ABuf, CL_TRUE, 0, m * n / 2, A);
            queue.enqueueWriteBuffer(scaleBuf, CL_TRUE, 0, m * n / group_size * sizeof(float), scale);
            queue.enqueueWriteBuffer(BBuf, CL_TRUE, 0, n * sizeof(float), B);
            queue.enqueueWriteBuffer(biasBuf, CL_TRUE, 0, m * sizeof(float), bias);            
            double timed = 0.0;

            cl::NDRange globalSize, localSize;

            cl::Kernel kernel_gemm(prog, "gemv_int4_fp32");

            // gemv_int4_fp32 param
            {
              gemv_int4_fp32(A, B, scale, bias, C_expect, m, n, group_size);



              globalSize = {m}, localSize = {64};
              kernel_gemm.setArg(0, ABuf), kernel_gemm.setArg(1, BBuf);
              kernel_gemm.setArg(2, scaleBuf), kernel_gemm.setArg(3, biasBuf);
              kernel_gemm.setArg(4, CBuf);
              kernel_gemm.setArg(5, m), kernel_gemm.setArg(6, n);
              kernel_gemm.setArg(7, group_size);
              timed = run_kernel(queue, kernel_gemm, globalSize, localSize, iters);

              queue.finish();
              queue.enqueueReadBuffer(CBuf, CL_TRUE, 0, m * sizeof(float), C);
              queue.finish();

              // matrix_transpose(C, C_T, n, m, false);
            }

            double L2norm = 0.0;

            for (int i = 0; i < n; i++) {
                double val = C[i] - C_expect[i];
                L2norm += val*val;
                if (L2norm > 1) {
                  printf("c[%d] is %f, %f", i, C[i], C_expect[i]);
                  break;
                }
            }

            PRINT(NEWLINE);
            double gflops = 2.0 * m * n / 1e9;
            double gB= (2.0 * (m * n) + n) / 1e9;
            printf("(%d, %d) GEMM GFLOP: %f, GB: %f, compute intensity: %f\n", m, n, gflops, gB, gflops/gB);
            printf("L2NORM is %.2f\n", L2norm);
            PRINT("time is : ");
            PRINT(timed);
            PRINT("us, compute is : ");
            PRINT(gflops);
            PRINT(" GFlops, bandwidth is : ");
            PRINT(gB);
            PRINT(" GB/s");
            PRINT(NEWLINE);
        }
    }
    free(A);
    free(B);
    free(C);
    free(C_expect);
    free(scale);
    free(bias);
    return 0;
}