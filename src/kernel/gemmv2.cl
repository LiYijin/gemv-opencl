R"(


__kernel void fused_gemm_add_v2(const int M, const int N, const int K, __global struct block_q4_0_new* x, __global float4* y, __global float * scale, __global float* y_bias, __global float* dst) {

    const int BM = 32;
    const int BN = 32;
    const int BK = 128;
    const int TM = 2;
    const int TN = 2;

    const int bx = get_group_id(0);     // 0 - 127

    const int by = get_group_id(1);     // 0 - N/2/16-1( 7 )
    const int tx = get_local_id(0);     // 0 - 15
    const int ty = get_local_id(1);     // 0 - 15
    const int tid = ty * get_local_size(0) + tx;    // ty * 16 + tx : 0 - 255
    // printf("bx: %d, by: %d, tx: %d, ty: %d\n", bx, by, tx, ty);
    const int scalarOffset = 4096 * 128;

    __local float4 s_y[BN][BK / 4];     // 32 x 32   16

    __local int4 s_x[BM][BK / 32];      // 32 x 4     2
    __local float s_s[BM][BK / 32];     // 32 x 4     0.5

    float acc[TM][TN];

    for (int i = 0; i < TN; i++) {
        int load_d_gmem_n = by * BN + ty * TN + i;
        for (int j = 0; j < TM; j++) {
            int load_d_gmem_m = bx * BM + tx * TM + j;
            acc[j][i] = y_bias[load_d_gmem_n * M + load_d_gmem_m];
        }
    }

    
    int load_x_smem_m = (tid >> 1) >> 2;    // 0 - 31
    int load_x_smem_k = (tid >> 1) & 3;     // 0 - 3
    // int load_s_smem_m = (tid >> 1) >> 2;    // 0 - 31
    // int load_s_smem_k = (tid >> 1) & 3;     // 0 - 3

    int load_y_smem_n = tid >> 3;       // 0 - 31
    int load_y_smem_k = (tid & 7) << 2;     // 0 4 8 12 16  ---- 28

    int load_x_gmem_m = bx * BM + load_x_smem_m;     // 127 * 16  * 2 + 31 = 4095
    // int load_s_gmem_m = bx * BM + load_s_smem_m;     // 4095

    int load_y_gmem_n = by * BN + load_y_smem_n;     // 7 * 16 * 2 + 31 = 255

    // Loop over all tiles
    // bk 0 - 31
    for (int bk = 0; bk < K / BK; bk++) {

        int load_x_gmem_k = bk * BK / 32 + load_x_smem_k;           //  0 - 127
        int load_x_gmem_addr = load_x_gmem_m * 128 + load_x_gmem_k;   // 
        s_x[load_x_smem_m][load_x_smem_k] = *((__global int4*)(x + load_x_gmem_addr) + 0);

        // int load_s_gmem_k = bk * BK / 32 + load_x_smem_k;
        // int load_s_gmem_addr = load_s_gmem_m * 128 + load_s_gmem_k;
        // s_s[load_x_smem_m][load_x_smem_k] = vload_half(0, (__global half*)(&x[scalarOffset]) + load_x_gmem_addr);
        s_s[load_x_smem_m][load_x_smem_k] = scale[load_x_gmem_addr];

        int load_y_gmem_k = bk * BK / 4 + load_y_smem_k;
        int load_y_gmem_addr = load_y_gmem_n * K / 4 + load_y_gmem_k;
        s_y[load_y_smem_n][load_y_smem_k + 0] = y[load_y_gmem_addr + 0];
        s_y[load_y_smem_n][load_y_smem_k + 1] = y[load_y_gmem_addr + 1];
        s_y[load_y_smem_n][load_y_smem_k + 2] = y[load_y_gmem_addr + 2];
        s_y[load_y_smem_n][load_y_smem_k + 3] = y[load_y_gmem_addr + 3];

        // Sync
        barrier(CLK_LOCAL_MEM_FENCE);


        // Compute
        for (int m = 0; m < TM; m++) {
            int comp_x_smem_m = tx * TM + m;
            for (int n = 0; n < TN; n++) {
                int comp_y_smem_n = ty * TN + n;
                for (int k = 0; k < 4; k++) {
                    float4 y00 = s_y[comp_y_smem_n][k * 8 + 0];
                    float4 y10 = s_y[comp_y_smem_n][k * 8 + 4];
                    float4 y01 = s_y[comp_y_smem_n][k * 8 + 1];
                    float4 y11 = s_y[comp_y_smem_n][k * 8 + 5];
                    float4 y02 = s_y[comp_y_smem_n][k * 8 + 2];
                    float4 y12 = s_y[comp_y_smem_n][k * 8 + 6];
                    float4 y03 = s_y[comp_y_smem_n][k * 8 + 3];
                    float4 y13 = s_y[comp_y_smem_n][k * 8 + 7];

                    const float d = s_s[comp_x_smem_m][k];
                    const int4 vui = s_x[comp_x_smem_m][k];
                    const float4 v00 = convert_float4((char)-8 + as_char4(vui.x & 0x0f0f0f0f));
                    const float4 v10 = convert_float4((char)-8 + as_char4((vui.x >> 4) & 0x0f0f0f0f));
                    const float4 v01 = convert_float4((char)-8 + as_char4(vui.y & 0x0f0f0f0f));
                    const float4 v11 = convert_float4((char)-8 + as_char4((vui.y >> 4) & 0x0f0f0f0f));
                    const float4 v02 = convert_float4((char)-8 + as_char4(vui.z & 0x0f0f0f0f));
                    const float4 v12 = convert_float4((char)-8 + as_char4((vui.z >> 4) & 0x0f0f0f0f));
                    const float4 v03 = convert_float4((char)-8 + as_char4(vui.w & 0x0f0f0f0f));
                    const float4 v13 = convert_float4((char)-8 + as_char4((vui.w >> 4) & 0x0f0f0f0f));

                    acc[m][n] += d * dot(v00, y00);
                    acc[m][n] += d * dot(v10, y10);
                    acc[m][n] += d * dot(v01, y01);
                    acc[m][n] += d * dot(v11, y11);
                    acc[m][n] += d * dot(v02, y02);
                    acc[m][n] += d * dot(v12, y12);
                    acc[m][n] += d * dot(v03, y03);
                    acc[m][n] += d * dot(v13, y13);
                }
            }
        }
        // Sync
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Return

    for (int i = 0; i < TN; i++) {
        int load_d_gmem_n = by * BN + ty * TN + i;
        for (int j = 0; j < TM; j++) {
            int load_d_gmem_m = bx * BM + tx * TM + j;
            dst[load_d_gmem_n * M + load_d_gmem_m] = acc[j][i];
        }
    }

}
)"