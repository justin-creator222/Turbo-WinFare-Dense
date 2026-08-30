// Affine INT4 matrix-vector product: out[row] = sum_c (q[row][c]*scale + bias) * x[c]
//
// The workhorse kernel: q/k/v/o projections, the shared expert's gate/up/down, every routed
// expert, and the tied LM head all run through here.
//
//   t0 weights   t1 scales   t2 biases   t3 x
//   u0 out (FP32)
//
//   gp0 = (rows, in_dim, w_byte_off, s_byte_off)
//   gp1 = (b_byte_off, x_byte_off, out_byte_off, row_base)
//
// One wave (32 lanes) per output row. Lanes stride over the quantization groups and reduce
// with WaveActiveSum -- SM 6.6 is required, and there is deliberately no cs_5_0 fallback
// where that intrinsic degrades to identity.
//
// GEMV_ROWS_PER_GROUP waves share a threadgroup. This kernel used one wave per group, and RDNA
// caps the number of workgroups resident per CU well below the number of waves a CU can hold,
// so single-wave groups left most of the SIMD capacity idle and no latency hiding for a kernel
// that is purely memory-bound. Packing several rows into one group raises occupancy without
// changing the arithmetic: each row is still reduced by exactly one wave, so results are
// bit-identical.
//
// The dispatch is (rows + GEMV_ROWS_PER_GROUP - 1) / GEMV_ROWS_PER_GROUP groups. Callers that
// still dispatch one group per row will compute only the first 1/GEMV_ROWS_PER_GROUP of the
// output.

#include "Common.hlsli"

#define GEMV_ROWS_PER_GROUP 8

[numthreads(WAVE_SIZE * GEMV_ROWS_PER_GROUP, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint rows       = gp0.x;
    const uint in_dim     = gp0.y;
    const uint w_off      = gp0.z;
    const uint s_off      = gp0.w;
    const uint b_off      = gp1.x;
    const uint x_off      = gp1.y;
    const uint out_off    = gp1.z;
    const uint row_base   = gp1.w;

    // tid is linear across the group, so tid / WAVE_SIZE selects the wave and tid % WAVE_SIZE
    // is the lane within it -- exactly the mapping WaveActiveSum below reduces over.
    const uint lane = tid % WAVE_SIZE;
    const uint row  = gid.x * GEMV_ROWS_PER_GROUP + (tid / WAVE_SIZE) + row_base;
    if (row >= rows) return;

    const float partial = gemv_int4_row_lane(g_in0, w_off, g_in1, s_off, g_in2, b_off,
                                             g_in3, x_off, row, in_dim, lane);
    const float total = WaveActiveSum(partial);
    if (lane == 0) {
        f32_store(g_out0, out_off + row * 4, total);
    }
}
