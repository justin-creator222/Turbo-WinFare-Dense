// act[i] = gelu_pytorch_tanh(gate[i]) * up[i]
//
// Shared by the dense MLP and every routed expert -- both are GeGLU with the same
// activation, differing only in width (2112 vs 704).
//
//   t0 gate   t1 up
//   u0 act (FP32)
//
//   gp0 = (n, gate_byte_off, up_byte_off, out_byte_off)
//   gp1 = (batch, gate_stride, up_stride, out_stride)   -- strides in bytes, per position
//
// The strides exist for the per-layer-embedding block, where `up` is a slice of a tensor laid
// out [position][layer][ple_dim]: the gate and output are contiguous per position but `up` is
// not. batch = 0 or 1 with zero strides is the plain elementwise form, where callers pass
// n = width * positions and everything is contiguous.

#include "Common.hlsli"

#define GEGLU_THREADS 256

[numthreads(GEGLU_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint n       = gp0.x;
    const uint g_off   = gp0.y;
    const uint u_off   = gp0.z;
    const uint out_off = gp0.w;

    const uint batch = max(gp1.x, 1u);
    const uint i = gid.x;
    if (i >= n * batch) return;

    const uint m = i / n;              // position
    const uint c = i - m * n;

    const float g = f32_load(g_in0, g_off   + m * gp1.y + c * 4);
    const float u = f32_load(g_in1, u_off   + m * gp1.z + c * 4);
    f32_store(g_out0,               out_off + m * gp1.w + c * 4, gelu_tanh(g) * u);
}
