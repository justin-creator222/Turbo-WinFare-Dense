// Copyright 2026 justin-creator222. Licensed under the Apache License, Version 2.0.
//
// These kernels are original HLSL, but the numerics they implement -- the forward-pass
// constants, normalization conventions, and dispatch order -- are derived from the Metal
// implementation in TurboFieldfare (https://github.com/drumih/turbo-fieldfare), copyright
// Andrey Mikhaylov, Apache-2.0. See NOTICE in the repository root.
//
// Shared helpers and binding convention for the Turbo-WinFare compute kernels.

#ifndef GTURBO_COMMON_HLSLI
#define GTURBO_COMMON_HLSLI

static const uint GROUP_SIZE = 64;
static const uint GEMV_THREADS = 32;
static const uint WAVE_SIZE = GEMV_THREADS;

[[vk::binding(0, 0)]] ByteAddressBuffer   g_in0 : register(t0);
[[vk::binding(1, 0)]] ByteAddressBuffer   g_in1 : register(t1);
[[vk::binding(2, 0)]] ByteAddressBuffer   g_in2 : register(t2);
[[vk::binding(3, 0)]] ByteAddressBuffer   g_in3 : register(t3);
[[vk::binding(4, 0)]] ByteAddressBuffer   g_in4 : register(t4);
[[vk::binding(5, 0)]] ByteAddressBuffer   g_in5 : register(t5);

[[vk::binding(6, 0)]] RWByteAddressBuffer g_out0 : register(u0);
[[vk::binding(7, 0)]] RWByteAddressBuffer g_out1 : register(u1);
[[vk::binding(8, 0)]] RWByteAddressBuffer g_out2 : register(u2);
[[vk::binding(9, 0)]] RWByteAddressBuffer g_out3 : register(u3);

struct PushParams {
    uint4 gp0;
    uint4 gp1;
    uint4 gp2;
    uint4 gp3;
};

[[vk::push_constant]]
PushParams g_params;

#define gp0 g_params.gp0
#define gp1 g_params.gp1
#define gp2 g_params.gp2
#define gp3 g_params.gp3

// Reads a BF16 value at an arbitrary 2-byte-aligned offset.
//
// Use this for EVERY non-quantized tensor: the LayerNorm family (input/post_attn/pre_ffn/
// post_ffn layernorms, q_norm, k_norm, the final model norm), the per-layer scalar, and the
// quantization scales and biases. The MLX export tags them all BF16 and that tag is correct.
//
// A previous round decided the LayerNorm family was secretly IEEE FP16, on the grounds that
// the BF16 decode "looks wrong" (model.norm.weight median 6.28, max 510) while the FP16
// decode looks like a textbook norm weight (median 2.393). That reasoning does not work: on
// this value range the two decodings are a BIJECTION, so each is self-consistent and neither
// can be picked by how it looks.
//
// What settles it is behaviour, measured on the checkpoint itself:
//
//                        pre-softmax |score|      hidden rms, layers 0..7
//     read as BF16       mean 5-16, max 24        0.81 1.69 1.68 1.66 1.66 1.76 2.08  (stable)
//     read as FP16       mean 192-327, max 549    4.3 8.2 15.6 30 57 106 197  (x2 per layer)
//
// Gemma4TextAttention sets self.scaling = 1.0, so nothing downstream rescales those scores.
// FP16 makes softmax a hard argmax and makes the residual stream diverge as 2^60; BF16 gives
// an ordinary transformer. The FP16 reading was the defect, and the NumPy reference agreed
// with it only because that reference was written from the same assumption.
float bf16_load(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    uint half_bits = ((byte_off & 2u) != 0u) ? (word >> 16) : (word & 0xFFFFu);
    return asfloat(half_bits << 16);
}

// Reads an IEEE FP16 at an arbitrary 2-byte-aligned offset.
//
// Used for the KV cache, which is stored at half precision: a 4096-position context costs
// 1.10 GiB instead of 2.19 GiB, and the difference is worth about four more resident layers.
// f32tof16/f16tof32 are hardware conversions, so this is a load plus a shift.
float f16_load(ByteAddressBuffer buf, uint byte_off) {
    const uint word = buf.Load(byte_off & ~3u);
    const uint h = ((byte_off & 2u) != 0u) ? (word >> 16) : (word & 0xFFFFu);
    return f16tof32(h);
}

// Reads a 32-bit uint at an arbitrary 2-byte-aligned offset.
uint u32_load(ByteAddressBuffer buf, uint byte_off) {
    uint base = byte_off & ~3u;
    if ((byte_off & 2u) == 0u) {
        return buf.Load(base);
    } else {
        uint w0 = buf.Load(base);
        uint w1 = buf.Load(base + 4u);
        return (w0 >> 16) | (w1 << 16);
    }
}

float f32_load(ByteAddressBuffer buf, uint byte_off) {
    return asfloat(buf.Load(byte_off));
}

// Overload for reading back a value this threadgroup just wrote. Callers must place an
// AllMemoryBarrierWithGroupSync() between the store and the load.
float f32_load(RWByteAddressBuffer buf, uint byte_off) {
    return asfloat(buf.Load(byte_off));
}

// Four consecutive 32-bit words in one instruction. `byte_off` must be 16-byte aligned.
//
// The gemv's WEIGHT reads were eight separate 4-byte loads per quantization group -- 256
// requests per wave for 1024 contiguous bytes. That is the one access pattern in the kernel
// that does not match what the streaming-bandwidth probe does, and the probe reaches 65-74 GB/s
// where the kernel reached ~33.
//
// What this bought, measured three runs each way on the 31B ("Hi", 14 tokens, greedy):
//
//                     GPU phase        throughput
//   8 x u32_load      848-856 ms    0.746-0.751 tok/s
//   2 x Load4         746-748 ms    0.750-0.760 tok/s
//
// So the kernel really is ~12% faster, and the third hypothesis for this gemv was the right
// one after two dead ends. But it does NOT show up as throughput, because at 45 of 60 layers
// resident the pass is bound by streaming I/O: time saved on the GPU is returned as time
// waiting on a layer read. E2B, which is fully resident, is unchanged too (15.9 against 15.8
// tok/s) -- at 2.5 GB it is not bandwidth-bound in the first place.
//
// It is kept because it is strictly less work for identical arithmetic, and it converts to
// real throughput the moment streaming stops binding. It is not a throughput win today, and
// an earlier single-run measurement that suggested otherwise was an outlier.
uint4 u32x4_load(ByteAddressBuffer buf, uint byte_off) {
    return buf.Load4(byte_off);
}

// Four consecutive FP32s in one instruction. `byte_off` must be 16-byte aligned.
//
// The gemv inner loop is dominated by ACTIVATION loads, not weight loads: every output row
// re-reads the whole activation vector, so a group of 64 weights costs 8 weight loads and 64
// activation loads. Fetching four at a time cuts that 64 to 16.
float4 f32x4_load(ByteAddressBuffer buf, uint byte_off) {
    return asfloat(buf.Load4(byte_off));
}

void f32_store(RWByteAddressBuffer buf, uint byte_off, float v) {
    buf.Store(byte_off, asuint(v));
}

// gelu_pytorch_tanh -- the activation named in the checkpoint's config.json.
float gelu_tanh(float x) {
    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCubic = 0.044715f;
    return 0.5f * x * (1.0f + tanh(kSqrt2OverPi * (x + kCubic * x * x * x)));
}

// One wave's dot product of an affine-quantized 4-bit row against an FP32 vector.
// Each lane walks a strided subset of the groups; the caller reduces with WaveActiveSum.
float gemv_int4_row_lane(ByteAddressBuffer W, uint w_base,
                         ByteAddressBuffer S, uint s_base,
                         ByteAddressBuffer B, uint b_base,
                         ByteAddressBuffer X, uint x_base,
                         uint row, uint in_dim, uint lane) {
    const uint num_groups       = in_dim / GROUP_SIZE;
    const uint row_w_bytes      = (in_dim / 8u) * 4u;
    const uint row_sb_bytes     = num_groups * 2u;
    const uint row_w_base       = w_base + row * row_w_bytes;
    const uint row_s_base       = s_base + row * row_sb_bytes;
    const uint row_b_base       = b_base + row * row_sb_bytes;

    float acc = 0.0f;

    for (uint g = lane; g < num_groups; g += GEMV_THREADS) {
        const float scale = bf16_load(S, row_s_base + g * 2u);
        const float bias  = bf16_load(B, row_b_base + g * 2u);

        const uint g_w_base = row_w_base + g * 32u;
        const uint g_x_base = x_base     + g * (GROUP_SIZE * 4u);

        // g_w_base is a multiple of 32, so both halves are 16-byte aligned.
        const uint4 wa = u32x4_load(W, g_w_base + 0u);
        const uint4 wb = u32x4_load(W, g_w_base + 16u);
        const uint packed_words[8] = { wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w };

        [unroll] for (uint word_idx = 0; word_idx < 8u; ++word_idx) {
            const uint packed = packed_words[word_idx];
            const uint x_off  = g_x_base + word_idx * 32u;

            // x_off is a multiple of 32, so both halves are 16-byte aligned.
            const float4 xa = f32x4_load(X, x_off + 0u);
            const float4 xb = f32x4_load(X, x_off + 16u);
            const float x0 = xa.x, x1 = xa.y, x2 = xa.z, x3 = xa.w;
            const float x4 = xb.x, x5 = xb.y, x6 = xb.z, x7 = xb.w;

            const uint q0 = (packed >>  0) & 0xFu;
            const uint q1 = (packed >>  4) & 0xFu;
            const uint q2 = (packed >>  8) & 0xFu;
            const uint q3 = (packed >> 12) & 0xFu;
            const uint q4 = (packed >> 16) & 0xFu;
            const uint q5 = (packed >> 20) & 0xFu;
            const uint q6 = (packed >> 24) & 0xFu;
            const uint q7 = (packed >> 28) & 0xFu;

            const float w0 = float(q0) * scale + bias;
            const float w1 = float(q1) * scale + bias;
            const float w2 = float(q2) * scale + bias;
            const float w3 = float(q3) * scale + bias;
            const float w4 = float(q4) * scale + bias;
            const float w5 = float(q5) * scale + bias;
            const float w6 = float(q6) * scale + bias;
            const float w7 = float(q7) * scale + bias;

            acc += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3
                 + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;
        }
    }
    return acc;
}

// One wave's dot product of an affine-quantized 8-bit row against an FP32 vector.
float gemv_int8_row_lane(ByteAddressBuffer W, uint w_base,
                         ByteAddressBuffer S, uint s_base,
                         ByteAddressBuffer B, uint b_base,
                         ByteAddressBuffer X, uint x_base,
                         uint row, uint in_dim, uint lane) {
    const uint num_groups       = in_dim / GROUP_SIZE;
    const uint row_w_bytes      = in_dim;
    const uint row_sb_bytes     = num_groups * 2u;
    const uint row_w_base       = w_base + row * row_w_bytes;
    const uint row_s_base       = s_base + row * row_sb_bytes;
    const uint row_b_base       = b_base + row * row_sb_bytes;

    float acc = 0.0f;

    for (uint g = lane; g < num_groups; g += GEMV_THREADS) {
        const float scale = bf16_load(S, row_s_base + g * 2u);
        const float bias  = bf16_load(B, row_b_base + g * 2u);

        const uint g_w_base = row_w_base + g * 64u;
        const uint g_x_base = x_base     + g * (GROUP_SIZE * 4u);

        for (uint word_idx = 0; word_idx < 16u; ++word_idx) {
            const uint packed = u32_load(W, g_w_base + word_idx * 4u);
            const uint x_off  = g_x_base + word_idx * 16u;

            const float x0 = f32_load(X, x_off + 0u);
            const float x1 = f32_load(X, x_off + 4u);
            const float x2 = f32_load(X, x_off + 8u);
            const float x3 = f32_load(X, x_off + 12u);

            const uint q0 = (packed >>  0) & 0xFFu;
            const uint q1 = (packed >>  8) & 0xFFu;
            const uint q2 = (packed >> 16) & 0xFFu;
            const uint q3 = (packed >> 24) & 0xFFu;

            const float w0 = float(q0) * scale + bias;
            const float w1 = float(q1) * scale + bias;
            const float w2 = float(q2) * scale + bias;
            const float w3 = float(q3) * scale + bias;

            acc += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
        }
    }
    return acc;
}

#endif // GTURBO_COMMON_HLSLI
