#include <common.h>
#include "kernel_preprocessor.hpp"

#define MSTRINGIFY(...) #__VA_ARGS__
#define FETCH_PER_WI 16
#define BUILD_OPTIONS " -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -Dgroup_size="

#define PRINT(...) print(__VA_ARGS__)

template<typename T, typename... Args>
void print(T arg, Args... args) {
    std::cout << arg << " ";
}
static const std::string stringifiedKernels =
#include "gemm.cl"
#include "gemmv1.cl"
#include "gemmv2.cl"
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

void gemm_int4_fp32(const uint8_t* A, const float* B, float* scale, float* C, const int m, const int k, const int n, const int group_size) {
  const int scale_per_row = k / group_size;
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      float acc = 0;
      for (int t = 0; t < k; t += 2) {
        float s = scale[row * scale_per_row + t / group_size];
        uint8_t a = A[(row * k + t) / 2];
        float a0 = ((a & 0x0f) - 8) * s;
        float a1 = (((a >> 4) & 0x0f) - 8) * s; 
        float b0 = B[t * n + col];
        float b1 = B[(t + 1) * n + col];
        acc += a0 * b0 + a1 * b1;
      }
      C[row * n + col] = acc;
      // printf("%.3f, ", acc);
    }
  }
}

void gemm_add_q4_gs32(const uint8_t * A, const float * B, float* scale, float* C, float* Bias, const int m, const int k, const int n) {
  const int scale_per_row = k / 32;
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      float acc = Bias[row * n + col];
      for (int t = 0; t < k; t += 32) {
        float s = scale[row * scale_per_row + t / 32];

        float b[32];
        for (int i = 0; i < 32; i++) {
          b[i] = B[(t + i) * n + col];
        }
          
        
        float a_v[32];
        for (int i = 0; i < 16; i++) {
          uint8_t a = A[(row * k / 2 + t / 2) + i];
          a_v[i] = ((a & 0x0f) - 8);
          a_v[16 + i] = ((a >> 4  & 0x0f) - 8);
        }

        for (int i = 0; i < 32; i++) {
          acc += s * b[i] * a_v[i];
        }
      
      }

      C[row * n + col] = acc;
    }
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


void initializeData(uint8_t* A, float* B, float* scale, float* C, float* Bias, const int m, const int k, const int n, const int group_size) {
  for (int i = 0; i < m * k / 2; i++) {
    A[i] = rand() % 256;
  }
  for (int i = 0; i < n * k; i++) {
    B[i] = rand() % 1024;
    
  }
  for (int i = 0; i < m * n; i++) {
    C[i] = 1;
  }
  for (int i = 0; i < m * n; i++) {
    Bias[i] = rand() % 1024;
  }
  for (int i = 0; i < m * k / group_size; i++) {
    scale[i] = rand() % 10;
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
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    const int m = 4096;
    const int k = 4096;
    const int n = 512;
    const int group_size = 128;
    const int iters = 10;
    uint8_t* A = (uint8_t*)malloc(m * k * sizeof(uint8_t) / 2);
    float* scale = (float*)malloc(m * k / group_size * sizeof(float));

    float* B = (float*)malloc(k * n * sizeof(float));
    float* Bias = (float*)malloc(m * n * sizeof(float));
    float* C = (float*)malloc(m * n * sizeof(float));
    float* C_expect = (float*)malloc(m * n * sizeof(float));

    float* B_T = (float*)malloc(k * n * sizeof(float));
    float* Bias_T = (float*)malloc(m * n * sizeof(float));
    float* C_T = (float*)malloc(m * n * sizeof(float));

    initializeData(A, B, scale, C, Bias, m, k, n, group_size);
    

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
        std::string compile_opts = BUILD_OPTIONS + std::to_string(group_size);
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

            cl::Buffer ABuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (m * k / 2));
            cl::Buffer scaleBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, m * k / group_size * sizeof(float));
            cl::Buffer BBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, k * n * sizeof(float));
            cl::Buffer CBuf = cl::Buffer(ctx, CL_MEM_READ_WRITE, m * n * sizeof(float));
            cl::Buffer BiasBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, m * n * sizeof(float));

            queue.enqueueWriteBuffer(ABuf, CL_TRUE, 0, m * k / 2, A);
            queue.enqueueWriteBuffer(scaleBuf, CL_TRUE, 0, m * k / group_size * sizeof(float), scale);
            
            double timed = 0.0;

            cl::NDRange globalSize, localSize;

            cl::Kernel kernel_gemm(prog, "gemm_int4_fp32");
            cl::Kernel kernel_gemm_v1(prog, "fused_gemm_add_v1");
            cl::Kernel kernel_gemm_v2(prog, "fused_gemm_add_v2");

            // gemm_int4_fp32 param
            {
              gemm_int4_fp32(A, B, scale, C_expect, m, k, n, group_size);

              queue.enqueueWriteBuffer(BBuf, CL_TRUE, 0, k * n * sizeof(float), B);
              queue.enqueueWriteBuffer(BiasBuf, CL_TRUE, 0, m * n * sizeof(float), Bias);

              globalSize = {n, m}, localSize = {32, 16};
              kernel_gemm.setArg(0, ABuf), kernel_gemm.setArg(1, BBuf);
              kernel_gemm.setArg(2, scaleBuf), kernel_gemm.setArg(3, CBuf);
              kernel_gemm.setArg(4, m), kernel_gemm.setArg(5, k);
              kernel_gemm.setArg(6, n);// kernel_gemm.setArg(7, group_size);
              timed = run_kernel(queue, kernel_gemm, globalSize, localSize, iters);

              queue.finish();
              queue.enqueueReadBuffer(CBuf, CL_TRUE, 0, m * n * sizeof(float), C);
              queue.finish();

              matrix_transpose(C, C_T, n, m, false);
            }

            // // fused_gemm_v1 param
            // {
            //   _ASSERT(group_size == 32);

            //   gemm_add_q4_gs32(A, B, scale, C_expect, Bias, m, k, n);

            //   matrix_transpose(B, B_T, k, n);
            //   matrix_transpose(Bias, Bias_T, m, n);

            //   queue.enqueueWriteBuffer(BBuf, CL_TRUE, 0, k * n * sizeof(float), B_T);
            //   queue.enqueueWriteBuffer(BiasBuf, CL_TRUE, 0, m * n * sizeof(float), Bias_T);

            //   globalSize = {m, n}, localSize = {16, 16};
            //   kernel_gemm_v1.setArg(0, m), kernel_gemm_v1.setArg(1, n);
            //   kernel_gemm_v1.setArg(2, k), kernel_gemm_v1.setArg(3, ABuf);
            //   kernel_gemm_v1.setArg(4, BBuf), kernel_gemm_v1.setArg(5, scaleBuf);
            //   kernel_gemm_v1.setArg(6, BiasBuf), kernel_gemm_v1.setArg(7, CBuf);
            //   timed = run_kernel(queue, kernel_gemm_v1, globalSize, localSize, iters);

            //   queue.finish();
            //   queue.enqueueReadBuffer(CBuf, CL_TRUE, 0, m * n * sizeof(float), C);
            //   queue.finish();

            //   matrix_transpose(C, C_T, n, m);
            // }
            
            // fused_gemm_v2 param
            // {
            //   _ASSERT(group_size == 32);

            //   gemm_add_q4_gs32(A, B, scale, C_expect, Bias, m, k, n);

            //   matrix_transpose(B, B_T, k, n);
            //   matrix_transpose(Bias, Bias_T, m, n);

            //   queue.enqueueWriteBuffer(BBuf, CL_TRUE, 0, k * n * sizeof(float), B_T);
            //   queue.enqueueWriteBuffer(BiasBuf, CL_TRUE, 0, m * n * sizeof(float), Bias_T);

            //   globalSize = {m / 2, n / 2}, localSize = {16, 16};
            //   kernel_gemm_v2.setArg(0, m), kernel_gemm_v2.setArg(1, n);
            //   kernel_gemm_v2.setArg(2, k), kernel_gemm_v2.setArg(3, ABuf);
            //   kernel_gemm_v2.setArg(4, BBuf), kernel_gemm_v2.setArg(5, scaleBuf);
            //   kernel_gemm_v2.setArg(6, BiasBuf), kernel_gemm_v2.setArg(7, CBuf);
            //   timed = run_kernel(queue, kernel_gemm_v2, globalSize, localSize, iters);

            //   queue.finish();
            //   queue.enqueueReadBuffer(CBuf, CL_TRUE, 0, m * n * sizeof(float), C);
            //   queue.finish();

            //   matrix_transpose(C, C_T, n, m);
            // }
            
            
            double gflops = 1.0 * m * k / timed  * n * 2 / 1e3;
            double gbps = ((m * k * n) * 2 + m * n) / timed / 1e3;

            double L2norm = 0.0;
            
            for (int i = 0; i < m * n; i++) {
                double val = C_T[i] - C_expect[i];
                L2norm += val*val;
            }

            PRINT(NEWLINE);
            double gflop = 2.0 * m * k * n / 1e9;
            double gB= (2.0 * (m * k * n) + m * n) / 1e9;
            printf("(%d, %d, %d) GEMM GFLOP: %f, GB: %f, compute intensity: %f\n", m, k, n, gflop, gB, gflop/gB);
            printf("L2NORM is %.2f\n", L2norm);
            // printf("c0 is %f, %f", C[0], C_expect[0]);
            PRINT("time is : ");
            PRINT(timed);
            PRINT("us, compute is : ");
            PRINT(gflops);
            PRINT(" GFlops, bandwidth is : ");
            PRINT(gbps);
            PRINT(" GB/s");
            PRINT(NEWLINE);
        }
    }
    free(A);
    free(B);
    free(C);
    free(C_expect);
    free(scale);
    free(C_T);
    free(Bias);
    free(Bias_T);
    return 0;
}
