// Grouped-query attention with sliding-window masking, as a single-pass online softmax.
//
// The attention scale comes in as a push constant and is 1.0 for this model:
// Gemma4TextAttention sets `self.scaling = 1.0` outright, and the config defines no
// query_pre_attn_scalar. A round-4 change made it head_dim^-0.5 on the reasoning that q_norm
// (a constant 1.8779) could not be absorbing the factor; the premise was right and the
// conclusion wrong -- upstream simply does not scale here.
//
// It is passed in rather than baked into the kernel so it stays a property of the model, and
// so the parity test can prove the kernel honours a non-unit value.
//
// One threadgroup per (batch position, query head).
//
//   t0 q   t1 k_cache   t2 v_cache
//   u0 ctx (FP32, [q_heads * head_dim])
//
// The KV cache is a ring: slot = t % capacity. Sliding-window layers pass
// capacity == sliding_window, so they use a fixed number of slots no matter how long the
// context is. Full-attention layers pass capacity == max_context, where t < capacity makes
// the modulo an identity -- one code path, no branch.
//
//   gp0 = (q_heads, kv_heads, head_dim, n_pos)
//   gp1 = (first_pos, q_off, k_off, v_off)
//   gp2 = (out_off, kv_capacity, scale_bits, 0)
//   gp3 = (batch M, q_stride_bytes, out_stride_bytes, sliding_window)
//
// Batched: dispatch q_heads * M groups. Group g covers position g / q_heads, which
// attends over keys [first_b, n_pos + slot] -- each query in the batch sees one more
// key than the last. Getting that span wrong is invisible at M = 1 and wrong
// everywhere else, so the multi-position oracle diff is what checks it.
//
// ---------------------------------------------------------------------------------------
// Why online softmax
//
// The previous form staged every score in groupshared and made three passes over them: find
// the max, exponentiate, then weight V. That needed one float of LDS per attended key, which
// put a hard ATTN_MAX_SPAN = 4096 ceiling on the attended span and so on max_context.
//
// This form walks the keys in tiles, carrying a running max `m`, a running denominator `l`,
// and the output accumulator itself. When a tile's max exceeds the running max, everything
// accumulated so far is rescaled by exp(m_old - m_new) -- algebraically identical to the
// two-pass form, because both numerator and denominator carry the same factor.
//
// Groupshared drops from 16 KB of scores to one tile of scores plus head_dim accumulators
// (~3 KB), and the span ceiling disappears: the only remaining cap is on head_dim, which is a
// property of the architecture (512 on the 31B's global layers) rather than of the context.
//
// The arithmetic is NOT bit-identical to the two-pass form -- rescaling reorders the sum -- so
// this is gated on the argmax and top-5 oracle checks rather than on an exact logit diff.

#include "Common.hlsli"

#define ATTN_THREADS 256
// One key per thread per tile, so the tile size is the group size.
#define ATTN_TILE ATTN_THREADS
// Bounds head_dim, not context. 512 is the largest in the architecture (global layers);
// sliding layers use 256.
#define ATTN_MAX_HEAD_DIM 512

groupshared float s_e[ATTN_TILE];              // this tile's exponentiated scores
groupshared float s_acc[ATTN_MAX_HEAD_DIM];    // running unnormalized output
groupshared float s_partial[ATTN_THREADS / 4];
groupshared float s_reduced;

[numthreads(ATTN_THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint q_heads  = gp0.x;
    const uint kv_heads = gp0.y;
    const uint head_dim = gp0.z;
    const uint n_pos    = gp0.w;
    const uint first    = gp1.x;
    const uint q_off    = gp1.y;
    const uint k_off    = gp1.z;
    const uint v_off    = gp1.w;
    const uint out_off  = gp2.x;
    const uint capacity = gp2.y;
    const float scale   = asfloat(gp2.z);

    const uint h    = gid.x % q_heads;
    const uint slot = gid.x / q_heads;            // which position in the batch

    // This query's own causal span. n_pos is the context length of the FIRST position in the
    // batch, so position `slot` sees `slot` more keys.
    const uint n_pos_b = n_pos + slot;
    const uint window  = gp3.w;
    const uint first_b = (window != 0u && n_pos_b > window) ? (n_pos_b - window) : first;

    const uint gqa  = q_heads / kv_heads;
    const uint kvh  = h / gqa;
    const uint qbase = q_off + slot * gp3.y + h * head_dim * 4;
    // The KV cache is FP16: two bytes per element, so the slot stride halves.
    const uint kv_stride = kv_heads * head_dim * 2;

    const uint lane_count = WaveGetLaneCount();
    const uint num_waves  = ATTN_THREADS / lane_count;

    for (uint d0 = tid; d0 < head_dim; d0 += ATTN_THREADS) s_acc[d0] = 0.0f;

    float m_run = -1e30f;   // running max of the scaled scores
    float l_run = 0.0f;     // running sum of exp(score - m_run)

    for (uint t0 = first_b; t0 < n_pos_b; t0 += ATTN_TILE) {
        const uint tile_n = min(ATTN_TILE, n_pos_b - t0);
        const uint t = t0 + tid;

        // --- this thread's score, or -inf if it is past the end of the tile --------------
        float sc = -1e30f;
        if (tid < tile_n) {
            const uint ring  = t % capacity;
            const uint kbase = k_off + ring * kv_stride + kvh * head_dim * 2;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += f32_load(g_in0, qbase + d * 4) * f16_load(g_in1, kbase + d * 2);
            }
            // Track the max of the SCALED score: the exponent below subtracts from this same
            // value, so a max taken before scaling would shift every exponent.
            sc = dot * scale;
        }

        // --- tile max -------------------------------------------------------------------
        // s_partial is reused for the sum reduction below, so every read of the reduced value
        // is fenced on both sides: the group sync guarantees arrival, not that every thread
        // has finished the read that follows it.
        GroupMemoryBarrierWithGroupSync();
        {
            const float wm = WaveActiveMax(sc);
            if (WaveIsFirstLane()) s_partial[tid / lane_count] = wm;
            GroupMemoryBarrierWithGroupSync();
            if (tid == 0) {
                float m = -1e30f;
                for (uint w = 0; w < num_waves; ++w) m = max(m, s_partial[w]);
                s_reduced = m;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        const float m_tile = s_reduced;

        // exp(m_run - m_new) is 0 on the first tile (m_run is -1e30), which zeroes the
        // untouched accumulator, and 1 whenever this tile brings nothing larger.
        const float m_new   = max(m_run, m_tile);
        const float rescale = exp(m_run - m_new);

        // --- exponentiate, and this tile's denominator contribution ----------------------
        const float e = (tid < tile_n) ? exp(sc - m_new) : 0.0f;
        GroupMemoryBarrierWithGroupSync();
        s_e[tid] = e;
        {
            const float ws = WaveActiveSum(e);
            if (WaveIsFirstLane()) s_partial[tid / lane_count] = ws;
            GroupMemoryBarrierWithGroupSync();
            if (tid == 0) {
                float s = 0.0f;
                for (uint w = 0; w < num_waves; ++w) s += s_partial[w];
                s_reduced = s;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        l_run = l_run * rescale + s_reduced;
        m_run = m_new;

        // --- fold this tile's V into the accumulator, parallel over head_dim -------------
        // s_e is complete for the whole group at this point (the sum reduction synced), and
        // the barrier at the top of the next tile keeps the next write from racing these
        // reads.
        for (uint d = tid; d < head_dim; d += ATTN_THREADS) {
            float acc = s_acc[d] * rescale;
            for (uint j = 0; j < tile_n; ++j) {
                const uint vbase = v_off + ((t0 + j) % capacity) * kv_stride + kvh * head_dim * 2;
                acc += s_e[j] * f16_load(g_in2, vbase + d * 2);
            }
            s_acc[d] = acc;
        }
    }

    const float inv_sum = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    for (uint d = tid; d < head_dim; d += ATTN_THREADS) {
        f32_store(g_out0, out_off + slot * gp3.z + (h * head_dim + d) * 4, s_acc[d] * inv_sum);
    }
}
