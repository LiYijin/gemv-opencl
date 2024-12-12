R"(

typedef uchar uint8_t;

struct __attribute__((packed)) block_q4_0_new
{
    union {
        uint8_t qs[16];
        uint qs32[4];
        uint4 qs128;
    };
};

__kernel void fused_gemm_add_v1(const int M, const int N, const int K, __global struct block_q4_0_new* x, __global float4* y, __global float * scale, __global float* y_bias, __global float* dst) {

    const int BM = 16;
    const int BN = 16;
    const int BK = 128;
    const int TM = 1;
    const int TN = 1;

    const int bx = get_group_id(0);

    const int by = get_group_id(1);
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tid = ty * get_local_size(0) + tx;
    // const int scalarOffset = 4096 * 128;

    const int g0 = get_global_id(0);
    const int g1 = by * BN + ty;

    __local float4 s_y[BN][BK / 4];

    __local int4 s_x[BM][BK / 32];
    __local float s_s[BM][BK / 32];

    float acc = y_bias[g1 * M + g0];
    

    int load_x_smem_m = (tid >> 2) >> 2;
    int load_x_smem_k = (tid >> 2) & 3;
    int load_s_smem_m = (tid >> 2) >> 2;
    int load_s_smem_k = (tid >> 2) & 3;

    int load_y_smem_n = tid >> 4;
    int load_y_smem_k = (tid & 15) << 1;

    int load_x_gmem_m = bx * BM + load_x_smem_m;
    int load_s_gmem_m = bx * BM + load_s_smem_m;

    int load_y_gmem_n = by * BN + load_y_smem_n;

    
    for (int bk = 0; bk < K / BK; bk++) {

        int load_x_gmem_k = bk * BK / 32 + load_x_smem_k;
        int load_x_gmem_addr = load_x_gmem_m * 128 + load_x_gmem_k;
        s_x[load_x_smem_m][load_x_smem_k] = *((__global int4*)(x + load_x_gmem_addr) + 0);

        // s_s[load_s_smem_m][load_s_smem_k] = vload_half(0, (__global half*)(&x[scalarOffset]) + load_x_gmem_addr);
        s_s[load_s_smem_m][load_s_smem_k] = scale[load_x_gmem_addr];

        int load_y_gmem_k = bk * BK / 4 + load_y_smem_k;
        int load_y_gmem_addr = load_y_gmem_n * K / 4 + load_y_gmem_k;
        s_y[load_y_smem_n][load_y_smem_k] = y[load_y_gmem_addr];
        s_y[load_y_smem_n][load_y_smem_k + 1] = y[load_y_gmem_addr + 1];


        // Sync
        barrier(CLK_LOCAL_MEM_FENCE);


        // Compute
        // 0 - 3
        for (int k = 0; k < BK / 32; k++) {

            // 4 = BK / 32
            float4 y00 = s_y[ty][k * 8 + 0];
            float4 y01 = s_y[ty][k * 8 + 1];
            float4 y02 = s_y[ty][k * 8 + 2];
            float4 y03 = s_y[ty][k * 8 + 3];
            float4 y10 = s_y[ty][k * 8 + 4];
            float4 y11 = s_y[ty][k * 8 + 5];
            float4 y12 = s_y[ty][k * 8 + 6];
            float4 y13 = s_y[ty][k * 8 + 7];

            const float d = s_s[tx][k];
            const int4 vui = s_x[tx][k];
            const float4 v00 = convert_float4((char)-8 + as_char4(vui.x & 0x0f0f0f0f));
            const float4 v10 = convert_float4((char)-8 + as_char4((vui.x >> 4) & 0x0f0f0f0f));
            const float4 v01 = convert_float4((char)-8 + as_char4(vui.y & 0x0f0f0f0f));
            const float4 v11 = convert_float4((char)-8 + as_char4((vui.y >> 4) & 0x0f0f0f0f));
            const float4 v02 = convert_float4((char)-8 + as_char4(vui.z & 0x0f0f0f0f));
            const float4 v12 = convert_float4((char)-8 + as_char4((vui.z >> 4) & 0x0f0f0f0f));
            const float4 v03 = convert_float4((char)-8 + as_char4(vui.w & 0x0f0f0f0f));
            const float4 v13 = convert_float4((char)-8 + as_char4((vui.w >> 4) & 0x0f0f0f0f));
            
            acc += d * dot(v00, y00);
            acc += d * dot(v10, y10);
            acc += d * dot(v01, y01);
            acc += d * dot(v11, y11);
            acc += d * dot(v02, y02);
            acc += d * dot(v12, y12);
            acc += d * dot(v03, y03);
            acc += d * dot(v13, y13);

        }
        

        // Sync
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    dst[g1 * M + g0] = acc;
    
}

)"