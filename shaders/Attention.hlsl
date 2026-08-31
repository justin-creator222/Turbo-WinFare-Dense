// Grouped-query attention with sliding-window masking.
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
// One threadgroup per query head. Scores are staged in groupshared, so max context is
// ATTN_MAX_POS.
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

#include "Common.hlsli"

#define ATTN_THREADS 256
// Scores are staged relative to `first`, so this bounds the ATTENDED SPAN, not the total
// context: a sliding-window layer never needs more than `sliding_window` entries however
// long the conversation gets. 4096 floats = 16 KB of the 32 KB groupshared guarantee, which
// keeps ~4 threadgroups resident per WGP on RDNA 3.
#define ATTN_MAX_SPAN 4096

groupshared float s_scores[ATTN_MAX_SPAN];
groupshared float s_partial[ATTN_THREADS / 4];

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
    if (h >= q_heads) return;

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
    const uint num_waves = ATTN_THREADS / lane_count;

    // --- 1. Scores, and their maximum -------------------------------------
    float local_max = -1e30f;
    for (uint t = first_b + tid; t < n_pos_b; t += ATTN_THREADS) {
        const uint slot = t % capacity;
        const uint kbase = k_off + slot * kv_stride + kvh * head_dim * 2;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot += f32_load(g_in0, qbase + d * 4) * f16_load(g_in1, kbase + d * 2);
        }
        const float score = dot * scale;
        s_scores[t - first_b] = score;   // staged relative to `first`
        // Track the max of the SCALED score: step 2 subtracts `best` from these same values,
        // so a max taken before scaling would shift every exponent.
        local_max = max(local_max, score);
    }
    local_max = WaveActiveMax(local_max);
    if (WaveIsFirstLane()) s_partial[tid / lane_count] = local_max;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
        float m = -1e30f;
        for (uint w = 0; w < num_waves; ++w) m = max(m, s_partial[w]);
        s_partial[0] = m;
    }
    GroupMemoryBarrierWithGroupSync();
    const float best = s_partial[0];
    // s_partial is reused below for the sum reduction. Without this barrier a fast wave can
    // overwrite s_partial[0] with its partial sum while a slower wave is still reading `best`
    // from it -- the group-sync above only guarantees arrival at that point, not that every
    // thread has completed the read that follows.
    //
    // The race window scales with the length of the loop below, so a single-token pass
    // (n_pos == 1) almost never hits it while multi-token generation does: the oracle diff at
    // position 0 stayed argmax-exact while real generation degenerated into repeated tokens.
    GroupMemoryBarrierWithGroupSync();

    // --- 2. exp and its sum ------------------------------------------------
    float local_sum = 0.0f;
    for (uint t = first_b + tid; t < n_pos_b; t += ATTN_THREADS) {
        const float e = exp(s_scores[t - first_b] - best);
        s_scores[t - first_b] = e;
        local_sum += e;
    }
    local_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) s_partial[tid / lane_count] = local_sum;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
        float s = 0.0f;
        for (uint w = 0; w < num_waves; ++w) s += s_partial[w];
        s_partial[0] = s;
    }
    GroupMemoryBarrierWithGroupSync();
    const float inv_sum = 1.0f / s_partial[0];
    // Same hazard in the other direction: step 3 below reads s_scores across the whole group,
    // so every thread must have taken inv_sum before any of them moves on.
    GroupMemoryBarrierWithGroupSync();

    // --- 3. Weighted sum of V, parallel over the head dimension -----------
    for (uint d = tid; d < head_dim; d += ATTN_THREADS) {
        float acc = 0.0f;
        for (uint t = first_b; t < n_pos_b; ++t) {
            const uint vbase = v_off + (t % capacity) * kv_stride + kvh * head_dim * 2;
            acc += s_scores[t - first_b] * f16_load(g_in2, vbase + d * 2);
        }
        f32_store(g_out0, out_off + slot * gp3.z + (h * head_dim + d) * 4, acc * inv_sum);
    }
}
