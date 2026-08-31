// Packs this pass's K and V into the KV cache ring as FP16.
//
// Replaces a pair of vkCmdCopyBuffer calls per layer. The cache used to hold FP32 -- a raw byte
// copy of what QKVEpilogue produced -- which cost 2.19 GiB for a 4096-position context. At FP16
// it costs 1.10 GiB, and that 1.09 GiB buys roughly four more permanently resident layers,
// which is ~180 ms of streaming per token.
//
// Each thread writes one uint holding two adjacent halves, so nothing does a read-modify-write
// on a 2-byte store. kv_heads * head_dim is always even (4096 sliding, 2048 full-attention), so
// the pairing never straddles a position.
//
// The ring wraps at kv_capacity, so a batch is not contiguous in the destination and each
// position resolves its own slot.
//
//   t0 k (FP32)   t1 v (FP32)
//   u0 k_cache (FP16)   u1 v_cache (FP16)
//
//   gp0 = (n_per_position, batch, base_position, kv_capacity)
//   gp1 = (src_stride_bytes, dst_slot_stride_bytes, 0, 0)

#include "Common.hlsli"

#define KVW_THREADS 256

[numthreads(KVW_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint n          = gp0.x;
    const uint batch      = gp0.y;
    const uint base_pos   = gp0.z;
    const uint capacity   = gp0.w;
    const uint src_stride = gp1.x;
    const uint dst_stride = gp1.y;

    const uint pairs_per_pos = n / 2u;
    const uint total = pairs_per_pos * batch;
    const uint i = gid.x;
    if (i >= total) return;

    const uint m    = i / pairs_per_pos;          // position within the batch
    const uint pair = i - m * pairs_per_pos;
    const uint slot = (base_pos + m) % capacity;

    const uint src = m * src_stride + pair * 8u;  // two FP32s
    const uint dst = slot * dst_stride + pair * 4u;

    const float k0 = f32_load(g_in0, src + 0u);
    const float k1 = f32_load(g_in0, src + 4u);
    g_out0.Store(dst, f32tof16(k0) | (f32tof16(k1) << 16));

    const float v0 = f32_load(g_in1, src + 0u);
    const float v1 = f32_load(g_in1, src + 4u);
    g_out1.Store(dst, f32tof16(v0) | (f32tof16(v1) << 16));
}
