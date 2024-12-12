R"(

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

)"