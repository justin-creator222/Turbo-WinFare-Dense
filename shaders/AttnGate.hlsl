// out[i] = context[i] * sigmoid(gate[i])
//
// Gated attention output elementwise fusion for Muse-Glimmer.
//
//   t0 context   t1 gate
//   u0 out (FP32)
//
//   gp0 = (n, ctx_byte_off, gate_byte_off, out_byte_off)
//   gp1 = (batch, ctx_stride, gate_stride, out_stride)   -- strides in bytes, per position

#include "Common.hlsli"

#define ATTN_GATE_THREADS 256

[numthreads(ATTN_GATE_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint n       = gp0.x;
    const uint c_off   = gp0.y;
    const uint g_off   = gp0.z;
    const uint out_off = gp0.w;

    const uint batch = max(gp1.x, 1u);
    const uint i = gid.x;
    if (i >= n * batch) return;

    const uint m = i / n;              // position
    const uint c = i - m * n;

    const float ctx = f32_load(g_in0, c_off + m * gp1.y + c * 4);
    const float g   = f32_load(g_in1, g_off + m * gp1.z + c * 4);
    f32_store(g_out0,                 out_off + m * gp1.w + c * 4, ctx * sigmoid_act(g));
}
