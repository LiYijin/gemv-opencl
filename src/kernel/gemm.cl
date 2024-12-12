R"(

#define VWM 2
#define VWN 4
#define MWI 4
#define NWI 16
#define KREG 1
#define KWG 32
#define MDIMA 16
#define MDIMC 16
#define MWG 64
#define NDIMB 8
#define NDIMC 8
#define NWG 128
#define KWI 2

__kernel void gemm_int4_fp32(__global char *A, __global float *B, __global float *scale, __global float* C, const int M, const int K, const int N)
{
    // Thread identifiers
    // const int idx = get_global_id(0);
    const int globalRow = get_global_id(1);//idx / N; // Row ID of C (0..M)
    const int globalCol = get_global_id(0);//idx % N; // Col ID of C (0..N)
    const int scale_per_row = K / group_size;
 
    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int s = 0; s < scale_per_row; s++) {
        float sc = scale[globalRow * scale_per_row + s];
        for (int k = s * group_size; k < (s + 1) * group_size; k += 2) {
          char a = A[(globalRow * K + k) / 2];
          float a0 = ((a & 0x0f) - 8) * sc;
          float a1 = (((a >> 4) & 0x0f) - 8) * sc; 
          float b0 = B[k * N + globalCol];
          float b1 = B[(k + 1) * N + globalCol];
          acc += a0 * b0 + a1 * b1;
      }
    }
    // Store the result
    C[globalRow * N + globalCol] = acc; 
}



#if USE_CL_MAD == 1
    #define MultiplyAdd(c,a,b) c = mad(a, b, c)
#else
    #define MultiplyAdd(c,a,b) c += a * b
#endif

inline float4 MultiplyAddVector(float4 cvec, const float aval, const float4 bvec) {
    MultiplyAdd(cvec.x, aval, bvec.x);
    MultiplyAdd(cvec.y, aval, bvec.y);
    MultiplyAdd(cvec.z, aval, bvec.z);
    MultiplyAdd(cvec.w, aval, bvec.w);

  return cvec;
}

inline float2 GlobalToPrivateA(const __global float2* restrict agm, const int _mi,
                                   const int kSizeM, const int idk, const int kwg) {
  // Computes the indices based on strided/non-strided access
    int mg = get_local_id(0) + _mi*MDIMC;

  // Computes the indices for the global memory
  int idm = mg + get_group_id(0) * (MWG/VWM);

  // Loads the data from global memory (not transposed) and stores into registers
  return agm[idk*(kSizeM/VWM) + idm];
}

inline float4 GlobalToPrivateB(const __global float4* restrict bgm, const int _ni,
                                   const int kSizeN, const int idk) {
  // Computes the indices based on strided/non-strided access
  int ng = _ni + get_local_id(1)*(NWI/VWN);
  // Computes the indices for the global memory
  int idn = ng + get_group_id(0) * (NWG/VWN);
  // Loads the data from global memory (transposed) and stores into registers
  return bgm[idk*(kSizeN/VWN) + idn];
}

__kernel void gemm_int4_fp32_clblast(__global float *A, __global float *B, __global float *scale, __global float* C, const int M, const int K, const int N)
{
    #pragma promote_to_registers
    float2 apm[MWI/VWM]; // MWI * 1
    #pragma promote_to_registers
    float4 bpm[NWI/VWN]; // 1 * NWI
}
)"