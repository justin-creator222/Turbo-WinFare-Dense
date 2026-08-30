// out[i] = (hidden[i] + res[i] * res_scale) * out_scale
//
// `out_scale` carries Gemma 4's per-layer `layer_scalar`. Upstream applies it as a single
// multiply of the WHOLE hidden state at the end of the decoder layer:
//
//     hidden_states *= self.layer_scalar        (modular_gemma4.py, after both residual adds)
//
// Folding it into the second (FFN) residual add reproduces that exactly for this model, since
// the per-layer-input block that would otherwise sit between them is absent when
// hidden_size_per_layer_input == 0, as it is on the 31B.
//
// Two earlier attempts got this wrong, and both produced fluent-looking but meaningless output:
// applying it to each residual BRANCH (hidden += branch * scalar, twice per layer) attenuated
// every layer roughly 11x twice over; removing it entirely let the residual stream grow
// unbounded to rms ~50 by layer 59. The correct form leaves the stream bounded at roughly
// s/(1-s) times the branch magnitude.
//
//   t0 hidden   t1 res
//   u0 out
//
//   gp0 = (n, res_scale_bits, out_scale_bits, 0)

#include "Common.hlsli"

#define RES_THREADS 256

[numthreads(RES_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint n = gp0.x;
    const float res_scale = asfloat(gp0.y);
    const float out_scale = asfloat(gp0.z);
    const uint i = gid.x;
    if (i >= n) return;

    const float h = f32_load(g_in0, i * 4);
    const float r = f32_load(g_in1, i * 4);
    f32_store(g_out0, i * 4, (h + r * res_scale) * out_scale);
}
