// Affine INT4 matrix times a BATCH of vectors: out[m][row] = sum_c (q[row][c]*scale + bias) * x[m][c]
//
// This is the prefill kernel. Decoding one token at a time reads all 16 GB of weights per
// token, so a 15-token prompt reads the model 15 times -- which is most of the time it takes
// to answer a short question. Processing M positions against one pass of the weights is the
// fix.
//
// WHY THIS IS NOT JUST GemvInt4 IN A LOOP
//
// Weight traffic is the obvious cost, but it is not the only one: in GemvInt4 every output row
// re-reads the whole activation vector, which for gate_proj is 21504 rows x 5376 floats =
// 462 MB of (cached) activation reads against 65 MB of weights. Running M positions naively
// multiplies that 462 MB by M and gains nothing -- measured on this machine, activation reads
// already sustain ~210 GB/s while weights sustain ~30 GB/s, so the activations would become
// the wall.
//
// So this kernel blocks properly:
//   * GEMM_ROWS_PER_GROUP rows share one threadgroup, and
//   * a tile of the activations is staged in groupshared ONCE per tile and read by every row.
// That cuts activation traffic by GEMM_ROWS_PER_GROUP and weight traffic by M.
//
// Tiling the input dimension is not optional: down_proj has in_dim 21504, which is 86 KB of
// activations per position against a 32 KB groupshared budget.
//
//   t0 weights   t1 scales   t2 biases   t3 x
//   u0 out (FP32)
//
//   gp0 = (rows, in_dim, w_byte_off, s_byte_off)
//   gp1 = (b_byte_off, x_byte_off, out_byte_off, row_base)
//   gp2 = (batch M, x_stride_bytes, out_stride_bytes, 0)
//
// Activations are laid out [position][dim] and outputs [position][row], both with an explicit
// stride, so every elementwise kernel downstream keeps working on a contiguous per-position
// slice with no layout change.

#include "Common.hlsli"

// Rows per threadgroup, one wave each. Matches kGemvRowsPerGroup in vk_pipeline.hpp.
#define GEMM_ROWS_PER_GROUP 8
// Maximum positions per dispatch. The accumulator array must stay in registers, so it is
// indexed only by fully unrolled loops -- a dynamically indexed local array spills to scratch
// and would cost far more than the batching saves.
#define GEMM_MAX_BATCH 8
// Activation columns staged per tile. GEMM_MAX_BATCH * GEMM_TILE * 4 bytes of groupshared;
// 8 * 512 * 4 = 16 KB, half the 32 KB budget, which keeps two groups resident per CU.
#define GEMM_TILE 512

#define GEMM_THREADS (WAVE_SIZE * GEMM_ROWS_PER_GROUP)

// LANE -> COLUMN MAPPING IS PERFORMANCE-CRITICAL.
//
// The first version of this kernel gave each lane a contiguous 16-column block, so lane L read
// groupshared at float index L*16. LDS has 32 banks of 4 bytes, and L*16 mod 32 takes only two
// values -- a 16-way bank conflict on every read. Measured, that made the batched kernel
// SLOWER than running GemvInt4 M times: 0.58x on gate_proj.
//
// Consecutive lanes must read consecutive floats. So the wave walks one 64-wide quantization
// group at a time in two half-wave steps, lane L taking column L of each half. Scale and bias
// are then wave-uniform (one group per iteration), and the eight lanes sharing a packed word
// read the same address, which coalesces.

groupshared float s_x[GEMM_MAX_BATCH * GEMM_TILE];

[numthreads(GEMM_THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint rows      = gp0.x;
    const uint in_dim    = gp0.y;
    const uint w_off     = gp0.z;
    const uint s_off     = gp0.w;
    const uint b_off     = gp1.x;
    const uint x_off     = gp1.y;
    const uint out_off   = gp1.z;
    const uint row_base  = gp1.w;
    const uint batch     = min(gp2.x, (uint)GEMM_MAX_BATCH);
    const uint x_stride  = gp2.y;
    const uint out_stride = gp2.z;

    const uint lane = tid % WAVE_SIZE;
    const uint wave = tid / WAVE_SIZE;
    const uint row  = gid.x * GEMM_ROWS_PER_GROUP + wave + row_base;

    const uint num_groups   = in_dim / GROUP_SIZE;
    const uint row_w_bytes  = (in_dim / 8u) * 4u;
    const uint row_sb_bytes = num_groups * 2u;
    const uint row_w_base   = w_off + row * row_w_bytes;
    const uint row_s_base   = s_off + row * row_sb_bytes;
    const uint row_b_base   = b_off + row * row_sb_bytes;

    // One accumulator per position. Unrolled everywhere it is touched, so these stay in VGPRs.
    float acc[GEMM_MAX_BATCH];
    [unroll] for (uint i = 0; i < GEMM_MAX_BATCH; ++i) acc[i] = 0.0f;

    for (uint tile = 0; tile < in_dim; tile += GEMM_TILE) {
        const uint tile_cols = min((uint)GEMM_TILE, in_dim - tile);

        // Stage this tile of every position's activations. Cooperative across the whole group:
        // all GEMM_ROWS_PER_GROUP waves read what one of them loads.
        GroupMemoryBarrierWithGroupSync();
        for (uint idx = tid; idx < batch * tile_cols; idx += GEMM_THREADS) {
            const uint m = idx / tile_cols;
            const uint c = idx - m * tile_cols;
            s_x[m * GEMM_TILE + c] = f32_load(g_in3, x_off + m * x_stride + (tile + c) * 4u);
        }
        GroupMemoryBarrierWithGroupSync();

        // Rows past the end still have to reach the barriers above, so the bail-out is below
        // rather than at the top of the function.
        if (row < rows) {
            for (uint gs = 0; gs < tile_cols; gs += GROUP_SIZE) {
                const uint g = (tile + gs) / GROUP_SIZE;          // wave-uniform
                const float scale = bf16_load(g_in1, row_s_base + g * 2u);
                const float bias  = bf16_load(g_in2, row_b_base + g * 2u);

                // Two half-wave steps cover the 64 columns of this group, lane L taking
                // column L of each -- so lanes touch consecutive groupshared floats.
                [unroll] for (uint half = 0; half < GROUP_SIZE / WAVE_SIZE; ++half) {
                    const uint lds_c = gs + half * WAVE_SIZE + lane;
                    if (lds_c >= tile_cols) continue;
                    const uint c = tile + lds_c;

                    const uint packed = u32_load(g_in0, row_w_base + (c >> 3) * 4u);
                    const float wv = float((packed >> ((c & 7u) * 4u)) & 0xFu) * scale + bias;

                    [unroll] for (uint m = 0; m < GEMM_MAX_BATCH; ++m) {
                        if (m < batch) acc[m] += wv * s_x[m * GEMM_TILE + lds_c];
                    }
                }
            }
        }
    }

    if (row >= rows) return;

    // Each lane owned a disjoint column range of the same row, so the row's dot product is the
    // sum across the wave -- one reduction per position.
    [unroll] for (uint m = 0; m < GEMM_MAX_BATCH; ++m) {
        if (m < batch) {
            const float total = WaveActiveSum(acc[m]);
            if (lane == 0) {
                f32_store(g_out0, out_off + m * out_stride + row * 4u, total);
            }
        }
    }
}
