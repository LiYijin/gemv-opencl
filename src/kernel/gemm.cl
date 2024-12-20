R"(
typedef char int8_t;
typedef uchar uint8_t;

// A layout: |1|0|   |3|2|   |..|
// every thread process one row.
__kernel void gemv_int4_fp32(__global char *A, __global float *B, __global float *scale, __global float* bias, __global float* C, const int M, const int N, const int group_size)
{
    // Thread identifiers
    // const int idx = get_global_id(0);
    const int row = get_global_id(0);
    const int scale_per_row = N / group_size;

    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int s = 0; s < scale_per_row; s++) {
        float sc = scale[row * scale_per_row + s];
        for (int k = s * group_size; k < (s + 1) * group_size; k += 2) {
          char a = A[(row * N + k) / 2];
          float a0 = ((a & 0x0f) - 8) * sc;
          float a1 = (((a >> 4) & 0x0f) - 8) * sc; 
          float b0 = B[k];
          float b1 = B[k + 1];
          acc += a0 * b0 + a1 * b1;
      }
    }
    // Store the result
    C[row] = acc + bias[row]; 
}
// every group process one row.
// work-item in on work-group process adjacent 
// weight layout: |16|0|  |17|1| .. 
// As is organized as [scale, 32 data], [scale, 32 data]
__kernel void gemv_int4_fp32_v2(__global char *As, __global float *B, __global float* bias, __global float* C, const int M, const int N, const int group_size, __local float* tmp, __global float* scale,  __global char* A)
{
  const int local_size = get_local_size(0);
  const int row = get_group_id(0);
  const int tid = get_local_id(0);
  const int col_step = local_size * 2;
  const int y_offset = group_size / 2;

  tmp[tid] = 0;
  for (int col = tid * 2; col < N; col += col_step) {
      const int ib = (row * N + col) / group_size; // block index
      const int iybs = col - col % group_size; // y block start index
      const int iqs = (col % group_size) / 2; // quant index

      // dequantize
      float v0, v1;

      // load scale value.
      const float sc = *(__global float*)(As + ib * 20);

      const uint8_t vui = As[(row * N + col) / 2 + (ib + 1) * 4];

      v0 = ((vui & 0xF) - 8) * sc;
      v1 = (((vui >> 4) & 0xF) - 8) * sc;

      // matrix multiplication
      tmp[tid] += v0 * B[iybs + iqs];
      tmp[tid] += v1 * B[iybs + iqs + y_offset];
  }

  // sum up partial sums and write back result
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = local_size / 2; s > 0; s >>= 1) {
      if (tid < s) {
          tmp[tid] += tmp[tid + s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
      C[row] = tmp[0] + bias[row];
  }
}

// every group process one row.
// work-item in on work-group process adjacent 
// weight layout: |16|0|  |17|1| .. 
// As is organized as [scale, 32 data], [scale, 32 data]
__kernel void gemv_int4_fp32_v3(__global char *A, __global float *B, __global float *scale,__global float* bias, __global float* C, const int M, const int N, const int group_size, __local float* tmp)
{
  const int local_size = get_local_size(0);
  const int row = get_group_id(0);
  const int tid = get_local_id(0);
  const int col_step = local_size * 2;
  const int y_offset = group_size / 2;

  tmp[tid] = 0;
  for (int col = tid * 2; col < N; col += col_step) {
      const int ib = (row * N + col) / group_size; // block index
      const int iybs = col - col % group_size; // y block start index
      const int iqs = (col % group_size) / 2; // quant index

      // dequantize
      float v0, v1;

      // load scale value.
      const float sc = scale[ib];

      const uint8_t vui = A[(row * N + col) / 2];

      v0 = ((vui & 0xF) - 8) * sc;
      v1 = (((vui >> 4) & 0xF) - 8) * sc;

      // matrix multiplication
      tmp[tid] += v0 * B[iybs + iqs];
      tmp[tid] += v1 * B[iybs + iqs + y_offset];
  }

  // sum up partial sums and write back result
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = local_size / 2; s > 0; s >>= 1) {
      if (tid < s) {
          tmp[tid] += tmp[tid + s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
      C[row] = tmp[0] + bias[row];
  }
}

// 
// every work-item process one row.
// |16|0|  |17|1| .. 
// block layout [0,0] [0, 1]
__kernel void gemv_int4_fp32_v4(__global char *A, __global float4 *B, __global float *scale, __global float* bias, __global float* C, const int M, const int N, const int group_size)
{
    const int row = get_global_id(0);
    const int col_step = group_size;
    float acc = 0.0f;
    for (int col = 0; col < N; col += col_step) {
        const int iybs = col / 4;

        float4 y00 = B[iybs + 0];
        float4 y10 = B[iybs + 4];
        float4 y01 = B[iybs + 1];
        float4 y11 = B[iybs + 5];
        float4 y02 = B[iybs + 2];
        float4 y12 = B[iybs + 6];
        float4 y03 = B[iybs + 3];
        float4 y13 = B[iybs + 7];

        const int ib = row * N / group_size +  col / group_size;
        const float sc = scale[ib];
        const int4 vui = *((__global int4*)(A) + ib);
        const float4 v00 = convert_float4((char)-8 + as_char4(vui.x & 0x0f0f0f0f));
        const float4 v10 = convert_float4((char)-8 + as_char4((vui.x >> 4) & 0x0f0f0f0f));
        const float4 v01 = convert_float4((char)-8 + as_char4(vui.y & 0x0f0f0f0f));
        const float4 v11 = convert_float4((char)-8 + as_char4((vui.y >> 4) & 0x0f0f0f0f));
        const float4 v02 = convert_float4((char)-8 + as_char4(vui.z & 0x0f0f0f0f));
        const float4 v12 = convert_float4((char)-8 + as_char4((vui.z >> 4) & 0x0f0f0f0f));                
        const float4 v03 = convert_float4((char)-8 + as_char4(vui.w & 0x0f0f0f0f));
        const float4 v13 = convert_float4((char)-8 + as_char4((vui.w >> 4) & 0x0f0f0f0f));
        acc += sc * dot(v00, y00);
        acc += sc * dot(v10, y10);
        acc += sc * dot(v01, y01);
        acc += sc * dot(v11, y11);
        acc += sc * dot(v02, y02);
        acc += sc * dot(v12, y12);
        acc += sc * dot(v03, y03);
        acc += sc * dot(v13, y13);
    }
    C[row] = acc + bias[row]; 
}

// 
// every work-item process one row.
// |16|0|  |17|1| .. 
// block layout [0,0] [1,0]
__kernel void gemv_int4_fp32_v5(__global char *A, __global float4 *B, __global float *scale, __global float* bias, __global float* C, const int M, const int N, const int group_size)
{
    const int local_size = get_local_size(0);
    const int row = get_global_id(0);
    const int col_step = group_size;
    float acc = 0.0f;
    for (int col = 0; col < N; col += col_step) {
        const int iybs = col / 4;

        float4 y00 = B[iybs + 0];
        float4 y10 = B[iybs + 4];
        float4 y01 = B[iybs + 1];
        float4 y11 = B[iybs + 5];
        float4 y02 = B[iybs + 2];
        float4 y12 = B[iybs + 6];
        float4 y03 = B[iybs + 3];
        float4 y13 = B[iybs + 7];

        const int ib = row +  col / group_size * M;
        const float sc = scale[ib];
        const int4 vui = *((__global int4*)(A) + ib);
        const float4 v00 = convert_float4((char)-8 + as_char4(vui.x & 0x0f0f0f0f));
        const float4 v10 = convert_float4((char)-8 + as_char4((vui.x >> 4) & 0x0f0f0f0f));
        const float4 v01 = convert_float4((char)-8 + as_char4(vui.y & 0x0f0f0f0f));
        const float4 v11 = convert_float4((char)-8 + as_char4((vui.y >> 4) & 0x0f0f0f0f));
        const float4 v02 = convert_float4((char)-8 + as_char4(vui.z & 0x0f0f0f0f));
        const float4 v12 = convert_float4((char)-8 + as_char4((vui.z >> 4) & 0x0f0f0f0f));                
        const float4 v03 = convert_float4((char)-8 + as_char4(vui.w & 0x0f0f0f0f));
        const float4 v13 = convert_float4((char)-8 + as_char4((vui.w >> 4) & 0x0f0f0f0f));
        acc += sc * dot(v00, y00);
        acc += sc * dot(v10, y10);
        acc += sc * dot(v01, y01);
        acc += sc * dot(v11, y11);
        acc += sc * dot(v02, y02);
        acc += sc * dot(v12, y12);
        acc += sc * dot(v03, y03);
        acc += sc * dot(v13, y13);
    }
    C[row] = acc + bias[row]; 
}

)"