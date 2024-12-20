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

#define CL_CHECK(err)                                               \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            fprintf(stderr, "ggml_opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            exit(1);                                                \
        }                                                           \
    } while (0)

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

// layout: |1|0| |3|2| ...
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

// change |1|0|  |3|2|  to |16|0| |17|1|
void matrix_mix(uint8_t * ori, uint8_t * after, int m, int n, int group_size) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j += group_size) {
        uint8_t* after_off = after + i * n / 2 + j / 2;
        uint8_t* ori_off = ori + i * n / 2 + j / 2;
        for (int k = 0; k < group_size / 4; k += 1) {
          after_off[2 * k] = (0xf0 & (ori_off[k + group_size / 4] << 4)) | (0xf & ori_off[k]);
          after_off[2 * k + 1] = (0xf0 & (ori_off[k + group_size / 4])) | (0xf & (ori_off[k] >> 4));
        }
      }
    }
    // printf("\n");
    // for (int i = 0; i < 16;i++) {
    //   printf(" %x(%d), ", ori[i], i);
    // }
    // printf("\n");
    // for (int i = 0; i < 16;i++) {
    //   printf(" %x(%d), ", after[i],i);
    // }    
}

void initializeData(uint8_t* A, float* B, float* scale, float* C, float* bias, const int m, const int n, const int group_size) {
  for (int i = 0; i < m * n / 2; i++) {
    A[i] = rand() % 256;
  }
  for (int i = 0; i < n; i++) {
    B[i] = rand() % 1024;

  }
  for (int i = 0; i < n; i++) {
    C[i] = 1;
  }
  for (int i = 0; i < n; i++) {
    bias[i] = rand() % 1024;
  }
  for (int i = 0; i < m * n / group_size; i++) {
    scale[i] = rand() % 10;
  }
}

template <typename T>
void row_major2column_major(T* A, const int m, const int k) {
  T* Ac = (T*)malloc(m * k * sizeof(T));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      int src_idx = i * k + j;
      int dst_idx = j * m + i;
      Ac[dst_idx] = A[src_idx];
    }
  }
  memcpy(A, Ac, m * k * sizeof(T));
  free(Ac);
}

void row_major2column_major_block(uint8_t* A, const int m, const int k, const int block_byte) {
  uint8_t* Ac = (uint8_t*)malloc(m * k * block_byte);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      int src_idx = i * k + j;
      int dst_idx = j * m + i;
      memcpy(Ac + dst_idx * block_byte, A + src_idx * block_byte, block_byte);
    }
  }
  memcpy(A, Ac, m * k * block_byte);
  free(Ac);
}

void print_log(const int m, const int n, const int group_size, double timed, const float* C, const float* C_expect) {
    double L2norm = 0.0;

    for (int i = 0; i < n; i++) {
        double val = C[i] - C_expect[i];
        L2norm += val*val;
        if (L2norm > 1) {
          printf("c[%d] is %f, %f", i, C[i], C_expect[i]);
          break;
        }
    }
    bool passed = false;
    if (L2norm < 1e3) passed = true;
    PRINT(NEWLINE);
    double gflops = 2.0 * m * n / 1e9;
    double gB= ((m * n) / 2 + (m * n) / group_size * 4 + n * 4) / 1e9;
    // printf("(%d, %d) GEMM GFLOP: %f, GB: %f, compute intensity: %f\n", m, n, gflops, gB, gflops/gB);
    printf("L2NORM is %.2f, ", L2norm);
    if (passed) printf("PASSED\n");
    else printf("FAILED\n");
    PRINT("time is : ");
    PRINT(timed);
    PRINT("us, compute is : ");
    PRINT(gflops);
    PRINT(" GFlops, bandwidth is : ");
    PRINT(gB / timed * 1e6);
    PRINT(" GB/s");
    PRINT(NEWLINE);  
}

void combine_scale(uint8_t* A, float* scale, uint8_t* As, const int m, const int n, const int group_size) {
  const int bpr = n / group_size;
  const int dbpb = group_size / 2;
  const int bpb = dbpb + sizeof(float);
  for(int i = 0; i < m; i++) {
    for (int j = 0; j < bpr; j++) {
      int block_index = i * bpr + j;
      memcpy(As + block_index * bpb, scale + block_index, sizeof(float));
      memcpy(As + block_index * bpb + sizeof(float), A + block_index * dbpb, dbpb);
    }
  }
}
void gemv_int4_fp32_v1_call(cl::Program prog, cl::CommandQueue queue, cl::Buffer ABuf, cl::Buffer BBuf, cl::Buffer scaleBuf, cl::Buffer biasBuf, cl::Buffer CBuf, float* C, float* C_expect, const int m, const int n, const int group_size, const int iters) {
    cl::Kernel kernel_gemm(prog, "gemv_int4_fp32");
    cl::NDRange globalSize = {(cl::size_type)m};
    cl::NDRange localSize = {64};
    kernel_gemm.setArg(0, ABuf), kernel_gemm.setArg(1, BBuf);
    kernel_gemm.setArg(2, scaleBuf), kernel_gemm.setArg(3, biasBuf);
    kernel_gemm.setArg(4, CBuf);
    kernel_gemm.setArg(5, m), kernel_gemm.setArg(6, n);
    kernel_gemm.setArg(7, group_size);
    double timed = run_kernel(queue, kernel_gemm, globalSize, localSize, iters);

    queue.finish();
    queue.enqueueReadBuffer(CBuf, CL_TRUE, 0, m * sizeof(float), C);
    queue.finish();
    printf("*******gemv_int4_fp32_v1********");
    print_log(m, n, group_size, timed, C, C_expect);
}
void gemv_int4_fp32_v2_call(cl::Program prog, cl::CommandQueue queue, cl::Context ctx, cl::Buffer ABuf, cl::Buffer BBuf, cl::Buffer scaleBuf, cl::Buffer biasBuf, cl::Buffer CBuf, float* C, float* C_expect, uint8_t* A, float* scale, const int m, const int n, const int group_size, const int iters) {
    cl::Kernel kernel_gemm(prog, "gemv_int4_fp32_v2");
    const int local_size = 64;
    cl::NDRange globalSize = {(cl::size_type)m * 64};
    cl::NDRange localSize = {local_size};
    uint8_t* A_mix = (uint8_t*)malloc(m * n * sizeof(uint8_t) / 2);
    uint8_t* As = (uint8_t*)malloc(m * n * sizeof(uint8_t) / 2 + m * n / group_size * sizeof(float));
    matrix_mix(A, A_mix, m, n, group_size);
    combine_scale(A_mix, scale, As, m, n, group_size);
    cl::Buffer AsBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (sizeof(float) + group_size / 2) * m * n / group_size);
    queue.enqueueWriteBuffer(ABuf, CL_TRUE, 0, m * n / 2, A_mix);
    queue.enqueueWriteBuffer(AsBuf, CL_TRUE, 0, (sizeof(float) + group_size / 2) * m * n / group_size, As);
    free(A_mix);
    free(As);
    cl::Buffer tmp = cl::Buffer(ctx, CL_MEM_READ_WRITE, local_size);

    kernel_gemm.setArg(0, AsBuf), kernel_gemm.setArg(1, BBuf);
    kernel_gemm.setArg(2, biasBuf);
    kernel_gemm.setArg(3, CBuf);
    kernel_gemm.setArg(4, m), kernel_gemm.setArg(5, n);
    kernel_gemm.setArg(6, group_size);
    kernel_gemm.setArg(7, tmp);
    kernel_gemm.setArg(8, scaleBuf);
    kernel_gemm.setArg(9, ABuf);
    double timed = run_kernel(queue, kernel_gemm, globalSize, localSize, iters);

    queue.finish();
    queue.enqueueReadBuffer(CBuf, CL_TRUE, 0, m * sizeof(float), C);
    queue.finish();

    printf("*******gemv_int4_fp32_v2********");
    print_log(m, n, group_size, timed, C, C_expect);
}

void gemv_int4_fp32_v3_call(cl::Program prog, cl::CommandQueue queue, cl::Context ctx, cl::Buffer ABuf, cl::Buffer BBuf, cl::Buffer scaleBuf, cl::Buffer biasBuf, cl::Buffer CBuf, float* C, float* C_expect, uint8_t* A, float* scale, const int m, const int n, const int group_size, const int iters) {
    cl::Kernel kernel_gemm(prog, "gemv_int4_fp32_v3");
    const int local_size = 64;
    cl::NDRange globalSize = {(cl::size_type)m * 64};
    cl::NDRange localSize = {local_size};
    uint8_t* A_mix = (uint8_t*)malloc(m * n * sizeof(uint8_t) / 2);
    matrix_mix(A, A_mix, m, n, group_size);
    queue.enqueueWriteBuffer(ABuf, CL_TRUE, 0, m * n / 2, A_mix);
    free(A_mix);
    cl::Buffer tmp = cl::Buffer(ctx, CL_MEM_READ_WRITE, local_size);

    kernel_gemm.setArg(0, ABuf), kernel_gemm.setArg(1, BBuf);
    kernel_gemm.setArg(2, scaleBuf), kernel_gemm.setArg(3, biasBuf);
    kernel_gemm.setArg(4, CBuf);
    kernel_gemm.setArg(5, m), kernel_gemm.setArg(6, n);
    kernel_gemm.setArg(7, group_size);
    kernel_gemm.setArg(8, tmp);
    double timed = run_kernel(queue, kernel_gemm, globalSize, localSize, iters);

    queue.finish();
    queue.enqueueReadBuffer(CBuf, CL_TRUE, 0, m * sizeof(float), C);
    queue.finish();

    printf("*******gemv_int4_fp32_v3********");
    print_log(m, n, group_size, timed, C, C_expect);
}

void gemv_int4_fp32_v4_call(cl::Program prog, cl::CommandQueue queue, cl::Context ctx, cl::Buffer ABuf, cl::Buffer BBuf, cl::Buffer scaleBuf, cl::Buffer biasBuf, cl::Buffer CBuf, float* C, float* C_expect, uint8_t* A, const int m, const int n, const int group_size, const int iters) {
    uint8_t* A_mix = (uint8_t*)malloc(m * n * sizeof(uint8_t) / 2);
    float* scale_mix = (float*)malloc(m * n / group_size * sizeof(float));
    matrix_mix(A, A_mix, m, n, group_size);
    queue.enqueueWriteBuffer(ABuf, CL_TRUE, 0, m * n / 2, A_mix);
    free(A_mix);
    cl::Kernel kernel_gemm(prog, "gemv_int4_fp32_v4");
    const int local_size = 64;
    cl::NDRange globalSize = {(cl::size_type)m};
    cl::NDRange localSize = {local_size};

    kernel_gemm.setArg(0, ABuf), kernel_gemm.setArg(1, BBuf);
    kernel_gemm.setArg(2, scaleBuf), kernel_gemm.setArg(3, biasBuf);
    kernel_gemm.setArg(4, CBuf);
    kernel_gemm.setArg(5, m), kernel_gemm.setArg(6, n);
    kernel_gemm.setArg(7, group_size);
    double timed = run_kernel(queue, kernel_gemm, globalSize, localSize, iters);

    queue.finish();
    queue.enqueueReadBuffer(CBuf, CL_TRUE, 0, m * sizeof(float), C);
    queue.finish();

    printf("*******gemv_int4_fp32_v4********");
    print_log(m, n, group_size, timed, C, C_expect);
}


void gemv_int4_fp32_v5_call(cl::Program prog, cl::CommandQueue queue, cl::Context ctx, cl::Buffer ABuf, cl::Buffer BBuf, cl::Buffer scaleBuf, cl::Buffer biasBuf, cl::Buffer CBuf, float* C, float* C_expect, uint8_t* A, float* scale, const int m, const int n, const int group_size, const int iters) {
    uint8_t* A_mix = (uint8_t*)malloc(m * n * sizeof(uint8_t) / 2);
    float* scale_mix = (float*)malloc(m * n / group_size * sizeof(float));
    matrix_mix(A, A_mix, m, n, group_size);
    memcpy(scale_mix, scale, m * n / group_size * sizeof(float));
    row_major2column_major_block(A_mix, m, n / group_size, group_size / 2);
    row_major2column_major(scale_mix, m, n / group_size);
    queue.enqueueWriteBuffer(ABuf, CL_TRUE, 0, m * n / 2, A_mix);
    queue.enqueueWriteBuffer(scaleBuf, CL_TRUE, 0, m * n / group_size * sizeof(float), scale_mix);
    free(A_mix);
    free(scale_mix);
    cl::Kernel kernel_gemm(prog, "gemv_int4_fp32_v5");
    const int local_size = 64;
    cl::NDRange globalSize = {(cl::size_type)m};
    cl::NDRange localSize = {local_size};

    kernel_gemm.setArg(0, ABuf), kernel_gemm.setArg(1, BBuf);
    kernel_gemm.setArg(2, scaleBuf), kernel_gemm.setArg(3, biasBuf);
    kernel_gemm.setArg(4, CBuf);
    kernel_gemm.setArg(5, m), kernel_gemm.setArg(6, n);
    kernel_gemm.setArg(7, group_size);
    double timed = run_kernel(queue, kernel_gemm, globalSize, localSize, iters);

    queue.finish();
    queue.enqueueReadBuffer(CBuf, CL_TRUE, 0, m * sizeof(float), C);
    queue.finish();
    printf("*******gemv_int4_fp32_v5********");
    print_log(m, n, group_size, timed, C, C_expect);
}

int main(int argc, char **argv)
{
    // gemv, A[m, n], B[n,1], C[m,1]
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    const int m = 4096;
    // const int k = 4096;
    const int n = 4096;
    const int group_size = 32;
    const int iters = 100;
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
            PRINT(NEWLINE);
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

            gemv_int4_fp32(A, B, scale, bias, C_expect, m, n, group_size);
            // naive opencl uitlization. one work-item process one row.
            gemv_int4_fp32_v1_call(prog, queue, ABuf, BBuf, scaleBuf, biasBuf, CBuf, C, C_expect, m, n, group_size, iters);
            // llama.cpp uitlization. one work-group process one row.
            gemv_int4_fp32_v2_call(prog, queue, ctx, ABuf, BBuf, scaleBuf, biasBuf, CBuf, C, C_expect, A, scale, m, n, group_size, iters);
            // llama.cpp uitlization + optimization to separate scale and data. one work-group process one row.
            gemv_int4_fp32_v3_call(prog, queue, ctx, ABuf, BBuf, scaleBuf, biasBuf, CBuf, C, C_expect, A, scale, m, n, group_size, iters);
            // optimization 1, layout between block is not transposed.
            gemv_int4_fp32_v4_call(prog, queue, ctx, ABuf, BBuf, scaleBuf, biasBuf, CBuf, C, C_expect, A, m, n, group_size, iters);
            // optimization 2, layout between block is transposed.
            gemv_int4_fp32_v5_call(prog, queue, ctx, ABuf, BBuf, scaleBuf, biasBuf, CBuf, C, C_expect, A, scale, m, n, group_size, iters);

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