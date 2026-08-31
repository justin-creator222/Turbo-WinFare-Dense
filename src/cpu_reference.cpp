#include "g4dense/cpu_reference.hpp"
#include "g4dense/format.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/sampling.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

namespace g4dense {

namespace {

// Every non-quantized tensor in the MLX export is BF16, exactly as the safetensors header
// says: the LayerNorm family (input/post_attn/pre_ffn/post_ffn layernorms, q_norm, k_norm,
// the final model norm), the per-layer scalar, and the quantization scales and biases.
//
// A previous round decided the LayerNorm family was secretly IEEE FP16, because the BF16
// decode "looks wrong" (model.norm.weight median 6.28, max 510) while the FP16 decode looks
// like a textbook norm weight (median 2.393). That reasoning does not work: on this value
// range the two decodings are a BIJECTION, so both are self-consistent and neither can be
// chosen by appearance. Behaviour on the checkpoint settles it:
//
//                    pre-softmax |score|      hidden rms, layers 0..7
//   read as BF16     mean 5-16, max 24        0.81 1.69 1.68 1.66 1.66 1.76 2.08  (stable)
//   read as FP16     mean 192-327, max 549    4.3 8.2 15.6 30 57 106 197  (x2 per layer)
//
// Gemma4TextAttention sets self.scaling = 1.0, so nothing downstream rescales those scores:
// FP16 turns softmax into a hard argmax and makes the residual stream diverge as 2^60.

inline float bf16_to_f32(uint16_t val) {
    uint32_t u32 = static_cast<uint32_t>(val) << 16;
    float f;
    std::memcpy(&f, &u32, sizeof(float));
    return f;
}

// Round-trip through IEEE FP16.
//
// The GPU stores its KV cache at half precision, so this reference must too -- otherwise the
// oracle diff reports the cache's quantization error as an engine defect, and the tolerance
// would have to be loosened to ~0.5, which is wide enough to hide a real bug.
inline float round_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    uint16_t h;
    if (exp <= 0)        h = static_cast<uint16_t>(sign);
    else if (exp >= 31)  h = static_cast<uint16_t>(sign | 0x7C00u);
    else {
        const uint32_t round_bit = (man >> 12) & 1u;
        const uint32_t sticky = (man & 0xFFFu) != 0u;
        uint32_t v = sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13);
        if (round_bit && (sticky || (v & 1u))) ++v;
        h = static_cast<uint16_t>(v);
    }
    const uint32_t s2 = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t e2 = (h >> 10) & 0x1Fu;
    uint32_t m2 = h & 0x3FFu;
    uint32_t bits;
    if (e2 == 0) {
        if (m2 == 0) bits = s2;
        else {
            e2 = 1;
            while ((m2 & 0x400u) == 0) { m2 <<= 1; --e2; }
            m2 &= 0x3FFu;
            bits = s2 | ((e2 + 127 - 15) << 23) | (m2 << 13);
        }
    } else if (e2 == 0x1Fu) bits = s2 | 0x7F800000u | (m2 << 13);
    else bits = s2 | ((e2 + 127 - 15) << 23) | (m2 << 13);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Gemma GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline float gelu_tanh(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    float inner = SQRT_2_OVER_PI * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

// Dequantize affine INT4 group-64 weights to float32 matrix
void dequantize_affine_int4_matrix(
    const uint32_t* packed_w,
    const uint16_t* scales_bf16,
    const uint16_t* biases_bf16,
    uint32_t rows,
    uint32_t cols,
    uint32_t group_size,
    std::vector<float>& out_f32
) {
    out_f32.resize(rows * cols);
    uint32_t groups_per_row = cols / group_size;
    uint32_t packed_cols = cols / 8;

    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t* row_packed = packed_w + r * packed_cols;
        const uint16_t* row_scales = scales_bf16 + r * groups_per_row;
        const uint16_t* row_biases = biases_bf16 + r * groups_per_row;
        float* row_out = out_f32.data() + r * cols;

        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t word_idx = c / 8;
            uint32_t nibble_idx = c % 8;
            uint32_t word = row_packed[word_idx];
            uint8_t q_val = (word >> (nibble_idx * 4)) & 0x0F;

            uint32_t g = c / group_size;
            float scale = bf16_to_f32(row_scales[g]);
            float bias = bf16_to_f32(row_biases[g]);

            row_out[c] = static_cast<float>(q_val) * scale + bias;
        }
    }
}

// Matrix-Vector multiply: out [rows] = W [rows, cols] * in [cols]
void gemv(const float* W, const float* in, uint32_t rows, uint32_t cols, float* out) {
    for (uint32_t r = 0; r < rows; ++r) {
        const float* row = W + r * cols;
        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            sum += row[c] * in[c];
        }
        out[r] = sum;
    }
}

// RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight
void rms_norm(const float* x, const float* weight, uint32_t dim, float eps, float* out) {
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + eps);
    for (uint32_t i = 0; i < dim; ++i) {
        out[i] = x[i] * rms * weight[i];
    }
}

// RMSNorm without scale (unit weight): y = x / sqrt(mean(x^2) + eps)
void rms_norm_no_scale(const float* x, uint32_t dim, float eps, float* out) {
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + eps);
    for (uint32_t i = 0; i < dim; ++i) {
        out[i] = x[i] * rms;
    }
}

void dump_tensor_f32(const std::string& dir, const std::string& name, const float* data, size_t count) {
    namespace fs = std::filesystem;
    if (dir.empty()) return;
    fs::path p = fs::path(dir) / (name + ".bin");
    std::ofstream f(p, std::ios::binary);
    if (f.is_open()) {
        f.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    }
}

} // namespace

CpuReferenceRunner::CpuReferenceRunner(const CpuReferenceConfig& config,
                                       std::shared_ptr<Tokenizer> tokenizer)
    : config_(config), tokenizer_(tokenizer) {}

CpuReferenceRunner::~CpuReferenceRunner() {
#ifdef _WIN32
    if (mapped_data_) {
        UnmapViewOfFile(mapped_data_);
        mapped_data_ = nullptr;
    }
    if (mapping_handle_) {
        CloseHandle(static_cast<HANDLE>(mapping_handle_));
        mapping_handle_ = nullptr;
    }
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(static_cast<HANDLE>(file_handle_));
        file_handle_ = nullptr;
    }
#endif
}

void CpuReferenceRunner::initialize() {
#ifdef _WIN32
    HANDLE hFile = CreateFileA(config_.container_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        throw G4DenseFormatError("CpuReferenceRunner: cannot open container " + config_.container_path);
    }
    file_handle_ = hFile;

    LARGE_INTEGER liSize;
    if (!GetFileSizeEx(hFile, &liSize)) {
        throw G4DenseFormatError("CpuReferenceRunner: GetFileSizeEx failed");
    }
    mapped_size_ = static_cast<size_t>(liSize.QuadPart);

    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) {
        throw G4DenseFormatError("CpuReferenceRunner: CreateFileMapping failed");
    }
    mapping_handle_ = hMap;

    mapped_data_ = static_cast<const uint8_t*>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
    if (!mapped_data_) {
        throw G4DenseFormatError("CpuReferenceRunner: MapViewOfFile failed");
    }

    if (mapped_size_ < sizeof(G4DenseHeader)) {
        throw G4DenseFormatError("CpuReferenceRunner: container file too small for header");
    }
    std::memcpy(&header_, mapped_data_, sizeof(G4DenseHeader));
    validate_header(header_, mapped_size_);
#else
    std::ifstream f(config_.container_path, std::ios::binary);
    if (!f.is_open()) {
        throw G4DenseFormatError("CpuReferenceRunner: cannot open container " + config_.container_path);
    }
    f.read(reinterpret_cast<char*>(&header_), sizeof(G4DenseHeader));
    validate_header(header_);
#endif

    k_cache_.resize(header_.num_layers);
    v_cache_.resize(header_.num_layers);

    if (!config_.dump_tensors_dir.empty()) {
        std::filesystem::create_directories(config_.dump_tensors_dir);
    }
}

std::vector<float> CpuReferenceRunner::forward_single_token(uint32_t token, uint32_t position, bool dump_tensors) {
    if (token >= header_.vocab_size) {
        if (header_.vocab_size < 262144) {
            token = token % header_.vocab_size;
        } else {
            throw G4DenseFormatError("Token ID exceeds vocab size");
        }
    }

    uint32_t d_model = header_.d_model;
    uint32_t d_ff = header_.d_ff;
    uint32_t g_size = header_.quant_group_size;

    const uint8_t* base_ptr = mapped_data_;

    // Embedding Dequantization for token
    const uint8_t* embed_ptr = base_ptr + header_.embed_offset;
    uint32_t packed_cols = d_model / 8;
    uint32_t groups_per_row = d_model / g_size;

    const uint32_t* embed_w = reinterpret_cast<const uint32_t*>(embed_ptr);
    const uint16_t* embed_s = reinterpret_cast<const uint16_t*>(embed_ptr + header_.vocab_size * packed_cols * sizeof(uint32_t));
    const uint16_t* embed_b = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(embed_s) + header_.vocab_size * groups_per_row * sizeof(uint16_t));

    // Clamp/modulo token to vocab_size
    uint32_t safe_token = token % header_.vocab_size;

    std::vector<float> hidden(d_model);
    const uint32_t* token_packed = embed_w + safe_token * packed_cols;
    const uint16_t* token_scales = embed_s + safe_token * groups_per_row;
    const uint16_t* token_biases = embed_b + safe_token * groups_per_row;

    for (uint32_t c = 0; c < d_model; ++c) {
        uint32_t w_idx = c / 8;
        uint32_t n_idx = c % 8;
        uint8_t q_val = (token_packed[w_idx] >> (n_idx * 4)) & 0x0F;
        uint32_t g = c / g_size;
        float s = bf16_to_f32(token_scales[g]);
        float b = bf16_to_f32(token_biases[g]);
        hidden[c] = static_cast<float>(q_val) * s + b;
    }

    // Embed scaling by sqrt(d_model)
    float embed_scale = std::sqrt(static_cast<float>(d_model));
    for (float& val : hidden) {
        val *= embed_scale;
    }

    // Dumps are keyed by POSITION, not by a literal 0. Position 0 still writes the same names
    // every existing fixture uses, but prefilling a sequence now produces one set per position
    // -- which is what lets the NumPy reference bisect attention over history rather than only
    // position 0.
    const std::string dump_prefix = "token" + std::to_string(position) + "_";

    if (dump_tensors) {
        dump_tensor_f32(config_.dump_tensors_dir, dump_prefix + "embed", hidden.data(), d_model);
    }

    std::vector<float> norm_buf(d_model);
    std::vector<float> attn_out(d_model);
    std::vector<float> gate_buf(d_ff);
    std::vector<float> up_buf(d_ff);
    std::vector<float> ffn_out(d_model);

    // Iterate through Layers 0..num_layers-1
    for (uint32_t l = 0; l < header_.num_layers; ++l) {
        const uint8_t* layer_ptr = base_ptr + header_.layer_offsets[l];
        bool is_global = (header_.global_layer_mask & (1ULL << l)) != 0;

        // Full-attention layers carry their own head_dim and kv-head count (512 / 4 on the
        // 31B vs 256 / 16 sliding). These were 31B literals here and in runner.cpp; both now
        // read the container header through the same resolver so they cannot drift apart.
        const LayerGeometry geo = resolve_layer_geometry(header_, l);
        uint32_t layer_head_dim = geo.head_dim;
        uint32_t layer_q_heads = geo.q_heads;
        uint32_t layer_kv_heads = geo.kv_heads;

        // Parse Norms + Layer Scalar (BF16)
        const uint16_t* raw_in_norm = reinterpret_cast<const uint16_t*>(layer_ptr);
        const uint16_t* raw_post_attn_norm = raw_in_norm + d_model;
        const uint16_t* raw_pre_ffn_norm = raw_post_attn_norm + d_model;
        const uint16_t* raw_post_ffn_norm = raw_pre_ffn_norm + d_model;
        const uint16_t* raw_q_norm = raw_post_ffn_norm + d_model;
        const uint16_t* raw_k_norm = raw_q_norm + layer_head_dim;
        const uint16_t* raw_layer_scalar = raw_k_norm + layer_head_dim;
        // BF16, not FP16: see the note in runner.cpp. A depth-scaled residual factor
        // (~0.0894 = 1/sqrt(2*60)), not a norm weight.
        // NOTE: layer_scalar is present in the checkpoint (BF16, ~0.0894 at layer 0, falling
        // with depth) but is NOT applied to the residual stream. Applying it -- as the sibling
        // project's 26B MoE decoder tail did -- attenuates every layer's contribution ~11x,
        // twice per layer, over 60 layers: an independent NumPy reference built from the
        // checkpoint gives layer-0 output rms 3.00 where the scaled engine gave 1.48 against an
        // embedding rms of 1.40, i.e. the layers were near no-ops and the logits were dominated
        // by the embedding. That is exactly the observed failure (generic subword fragments).
        //
        // Kept read for diagnostics; its correct use in this architecture is not established.
        float layer_scalar_val = bf16_to_f32(*raw_layer_scalar);
        const uint8_t* weights_start = reinterpret_cast<const uint8_t*>(raw_layer_scalar + 1);

        std::vector<float> in_norm_f32(d_model), post_attn_norm_f32(d_model);
        std::vector<float> pre_ffn_norm_f32(d_model), post_ffn_norm_f32(d_model);
        std::vector<float> q_norm_f32(layer_head_dim), k_norm_f32(layer_head_dim);

        for (uint32_t i = 0; i < d_model; ++i) {
            in_norm_f32[i] = bf16_to_f32(raw_in_norm[i]);
            post_attn_norm_f32[i] = bf16_to_f32(raw_post_attn_norm[i]);
            pre_ffn_norm_f32[i] = bf16_to_f32(raw_pre_ffn_norm[i]);
            post_ffn_norm_f32[i] = bf16_to_f32(raw_post_ffn_norm[i]);
        }
        for (uint32_t i = 0; i < layer_head_dim; ++i) {
            q_norm_f32[i] = bf16_to_f32(raw_q_norm[i]);
            k_norm_f32[i] = bf16_to_f32(raw_k_norm[i]);
        }

        // Dequantize Attention & FFN weights
        uint32_t q_rows = layer_q_heads * layer_head_dim;
        uint32_t kv_rows = layer_kv_heads * layer_head_dim;
        uint32_t o_rows = d_model;
        uint32_t o_cols = layer_q_heads * layer_head_dim;

        const uint8_t* p = weights_start;

        auto parse_quant_block = [&](uint32_t rows, uint32_t cols, std::vector<float>& mat_out) {
            uint32_t w_bytes = rows * (cols / 8) * sizeof(uint32_t);
            uint32_t s_bytes = rows * (cols / g_size) * sizeof(uint16_t);
            uint32_t b_bytes = rows * (cols / g_size) * sizeof(uint16_t);

            const uint32_t* pw = reinterpret_cast<const uint32_t*>(p);
            const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + w_bytes);
            const uint16_t* pb = reinterpret_cast<const uint16_t*>(p + w_bytes + s_bytes);

            dequantize_affine_int4_matrix(pw, ps, pb, rows, cols, g_size, mat_out);
            p += w_bytes + s_bytes + b_bytes;
        };

        std::vector<float> W_q, W_k, W_v, W_o, W_gate, W_up, W_down;
        parse_quant_block(q_rows, d_model, W_q);
        parse_quant_block(kv_rows, d_model, W_k);
        if (!is_global) {
            parse_quant_block(kv_rows, d_model, W_v);
        }
        parse_quant_block(o_rows, o_cols, W_o);
        parse_quant_block(d_ff, d_model, W_gate);
        parse_quant_block(d_ff, d_model, W_up);
        parse_quant_block(d_model, d_ff, W_down);

        // 1. Input RMSNorm
        rms_norm(hidden.data(), in_norm_f32.data(), d_model, 1e-6f, norm_buf.data());

        // 2. Q, K, V Projections
        std::vector<float> q_buf(q_rows);
        std::vector<float> k_buf(kv_rows);
        std::vector<float> v_buf(kv_rows);

        gemv(W_q.data(), norm_buf.data(), q_rows, d_model, q_buf.data());
        gemv(W_k.data(), norm_buf.data(), kv_rows, d_model, k_buf.data());
        if (!is_global) {
            gemv(W_v.data(), norm_buf.data(), kv_rows, d_model, v_buf.data());
        }

        // 3. Q/K Norm & RoPE (NeoX rotate_half)
        float rope_theta = is_global ? header_.rope_theta_global : header_.rope_theta_local;
        uint32_t half_dim = layer_head_dim / 2;
        uint32_t rotated_pairs = is_global ? 64 : half_dim;

        for (uint32_t h = 0; h < layer_q_heads; ++h) {
            float* q_head = q_buf.data() + h * layer_head_dim;
            rms_norm(q_head, q_norm_f32.data(), layer_head_dim, 1e-6f, q_head);

            for (uint32_t p = 0; p < rotated_pairs; ++p) {
                float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * p) / static_cast<float>(layer_head_dim));
                float angle = position * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                float x0 = q_head[p];
                float x1 = q_head[p + half_dim];
                q_head[p] = x0 * cos_a - x1 * sin_a;
                q_head[p + half_dim] = x0 * sin_a + x1 * cos_a;
            }
        }

        for (uint32_t h = 0; h < layer_kv_heads; ++h) {
            float* k_head = k_buf.data() + h * layer_head_dim;
            float* v_head = v_buf.data() + h * layer_head_dim;

            // Upstream (Gemma4TextAttention.forward, modular_gemma4.py):
            //
            //   key_states   = k_proj(x)
            //   value_states = v_proj(x) if v_proj is not None else key_states   <-- BEFORE k_norm
            //   key_states   = k_norm(key_states)
            //   key_states   = apply_rotary_pos_emb(key_states, ...)
            //   value_states = v_norm(value_states)                             <-- EVERY layer
            //
            // Two things follow, and the engine originally got both wrong.
            //
            // 1. On full-attention layers there is no v_proj (attention_k_eq_v), so V aliases
            //    the RAW k_proj output. Python rebinds `key_states` when k_norm returns a new
            //    tensor, so V never sees k_norm. (A round-4 change applied k_norm here; wrong.)
            //
            // 2. v_norm is `Gemma4RMSNorm(head_dim, with_scale=False)` -- an unweighted
            //    RMSNorm -- and it is applied unconditionally, on the 50 sliding layers as
            //    well. `is_kv_shared_layer` is false for every layer here because this config
            //    sets num_kv_shared_layers = 0, so there is no branch that skips it. The
            //    engine normalized V only on the 10 global layers and fed the 50 sliding
            //    layers a raw, unnormalized v_proj output.
            if (is_global) {
                rms_norm_no_scale(k_head, layer_head_dim, 1e-6f, v_head);
            } else {
                rms_norm_no_scale(v_head, layer_head_dim, 1e-6f, v_head);
            }
            rms_norm(k_head, k_norm_f32.data(), layer_head_dim, 1e-6f, k_head);

            for (uint32_t p = 0; p < rotated_pairs; ++p) {
                float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * p) / static_cast<float>(layer_head_dim));
                float angle = position * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                float x0 = k_head[p];
                float x1 = k_head[p + half_dim];
                k_head[p] = x0 * cos_a - x1 * sin_a;
                k_head[p + half_dim] = x0 * sin_a + x1 * cos_a;
            }
        }

        // 4. Update KV Cache
        // Stored at half precision, matching the GPU's KV cache.
        for (float& kv : k_buf) kv = round_to_f16(kv);
        for (float& kv : v_buf) kv = round_to_f16(kv);
        k_cache_[l].insert(k_cache_[l].end(), k_buf.begin(), k_buf.end());
        v_cache_[l].insert(v_cache_[l].end(), v_buf.begin(), v_buf.end());

        uint32_t total_ctx = position + 1;
        uint32_t window = is_global ? total_ctx : std::min(total_ctx, header_.sliding_window);
        uint32_t start_pos = total_ctx - window;

        // 5. Multi-Head / Grouped Query Attention
        std::vector<float> head_out(layer_q_heads * layer_head_dim, 0.0f);
        uint32_t q_per_kv = layer_q_heads / layer_kv_heads;

        // Gemma 4 sets self.scaling = 1.0 (modular_gemma4.py), not head_dim^-0.5. An earlier
        // round changed this to head_dim^-0.5 on the reasoning that q_norm (a constant 1.8779)
        // could not be absorbing the factor. That reasoning was wrong: upstream simply does not
        // scale here.
        const float attn_scale = 1.0f;

        for (uint32_t h = 0; h < layer_q_heads; ++h) {
            uint32_t kv_h = h / q_per_kv;
            const float* q_h = q_buf.data() + h * layer_head_dim;

            std::vector<float> scores(window);
            for (uint32_t t = 0; t < window; ++t) {
                uint32_t pos_idx = start_pos + t;
                const float* k_t = k_cache_[l].data() + (pos_idx * layer_kv_heads + kv_h) * layer_head_dim;

                float score = 0.0f;
                for (uint32_t d = 0; d < layer_head_dim; ++d) {
                    score += q_h[d] * k_t[d];
                }
                scores[t] = score * attn_scale;
            }

            // Softmax
            float max_s = -1e9f;
            for (float s : scores) max_s = std::max(max_s, s);
            float sum_exp = 0.0f;
            for (float& s : scores) {
                s = std::exp(s - max_s);
                sum_exp += s;
            }
            for (float& s : scores) s /= sum_exp;

            // Attention * V
            float* out_h = head_out.data() + h * layer_head_dim;
            for (uint32_t t = 0; t < window; ++t) {
                uint32_t pos_idx = start_pos + t;
                const float* v_t = v_cache_[l].data() + (pos_idx * layer_kv_heads + kv_h) * layer_head_dim;
                float a = scores[t];
                for (uint32_t d = 0; d < layer_head_dim; ++d) {
                    out_h[d] += a * v_t[d];
                }
            }
        }

        // 6. Out Projection & Post-Attention Residual
        gemv(W_o.data(), head_out.data(), o_rows, o_cols, attn_out.data());
        rms_norm(attn_out.data(), post_attn_norm_f32.data(), d_model, 1e-6f, attn_out.data());
        for (uint32_t i = 0; i < d_model; ++i) {
            hidden[i] += attn_out[i];
        }

        // 7. FFN: Pre-FFN Norm + Gate/Up GEMVs + GeGLU + Down GEMV + Post-FFN Norm + Residual
        rms_norm(hidden.data(), pre_ffn_norm_f32.data(), d_model, 1e-6f, norm_buf.data());
        gemv(W_gate.data(), norm_buf.data(), d_ff, d_model, gate_buf.data());
        gemv(W_up.data(), norm_buf.data(), d_ff, d_model, up_buf.data());

        for (uint32_t i = 0; i < d_ff; ++i) {
            gate_buf[i] = gelu_tanh(gate_buf[i]) * up_buf[i];
        }

        gemv(W_down.data(), gate_buf.data(), d_model, d_ff, ffn_out.data());
        rms_norm(ffn_out.data(), post_ffn_norm_f32.data(), d_model, 1e-6f, ffn_out.data());
        for (uint32_t i = 0; i < d_model; ++i) {
            // Layer end: hidden_states *= layer_scalar (modular_gemma4.py). The per-layer-input
            // block that would sit between this add and that multiply is absent when
            // hidden_size_per_layer_input == 0, as on the 31B, so folding it here is exact.
            hidden[i] = (hidden[i] + ffn_out[i]) * layer_scalar_val;
        }

        if (dump_tensors) {
            // Every layer, not just layer 0. Layer 0 matching an independent reference proved
            // the sliding-layer path; bisecting the rest needs each layer's output, and the
            // first divergence is what localizes a defect in the 10 global-attention layers.
            // 60 files x 21 KB is negligible.
            dump_tensor_f32(config_.dump_tensors_dir,
                            dump_prefix + "layer" + std::to_string(l) + "_hidden",
                            hidden.data(), d_model);
            if (l == 0) {
                dump_tensor_f32(config_.dump_tensors_dir, dump_prefix + "layer0_hidden",
                                hidden.data(), d_model);
            }
        }
    }

    // Final RMSNorm (stored immediately after embedding table)
    size_t embed_w_bytes = header_.vocab_size * packed_cols * sizeof(uint32_t);
    size_t embed_s_bytes = header_.vocab_size * groups_per_row * sizeof(uint16_t);
    size_t embed_b_bytes = header_.vocab_size * groups_per_row * sizeof(uint16_t);
    const uint16_t* final_norm_raw = reinterpret_cast<const uint16_t*>(embed_ptr + embed_w_bytes + embed_s_bytes + embed_b_bytes);

    std::vector<float> final_norm_f32(d_model);
    for (uint32_t i = 0; i < d_model; ++i) {
        final_norm_f32[i] = bf16_to_f32(final_norm_raw[i]);
    }

    // Dumped so the LM head can be reasoned about separately from the layer stack: the layers
    // produce a healthy hidden state (rms ~1.7) while the logits come out with a +9.7 mean, so
    // the transition between them is where the defect lives.
    if (dump_tensors) {
        dump_tensor_f32(config_.dump_tensors_dir, dump_prefix + "final_hidden_pre_norm", hidden.data(), d_model);
        dump_tensor_f32(config_.dump_tensors_dir, dump_prefix + "final_norm_weight", final_norm_f32.data(), d_model);
    }

    rms_norm(hidden.data(), final_norm_f32.data(), d_model, 1e-6f, hidden.data());

    if (dump_tensors) {
        dump_tensor_f32(config_.dump_tensors_dir, dump_prefix + "final_normed", hidden.data(), d_model);
    }

    // LM Head (Tied to Embedding Table)
    std::vector<float> logits(header_.vocab_size);
    const uint32_t* lm_w = reinterpret_cast<const uint32_t*>(base_ptr + header_.lm_head_offset);
    const uint16_t* lm_s = reinterpret_cast<const uint16_t*>(base_ptr + header_.lm_head_offset + header_.vocab_size * packed_cols * sizeof(uint32_t));
    const uint16_t* lm_b = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(lm_s) + header_.vocab_size * groups_per_row * sizeof(uint16_t));

    float softcap = header_.final_logit_softcapping;

    for (uint32_t v = 0; v < header_.vocab_size; ++v) {
        const uint32_t* row_packed = lm_w + v * packed_cols;
        const uint16_t* row_s = lm_s + v * groups_per_row;
        const uint16_t* row_b = lm_b + v * groups_per_row;

        float sum = 0.0f;
        for (uint32_t c = 0; c < d_model; ++c) {
            uint32_t word_idx = c / 8;
            uint32_t nibble_idx = c % 8;
            uint8_t q_val = (row_packed[word_idx] >> (nibble_idx * 4)) & 0x0F;
            uint32_t g = c / g_size;
            float scale = bf16_to_f32(row_s[g]);
            float bias = bf16_to_f32(row_b[g]);
            float w_val = static_cast<float>(q_val) * scale + bias;
            sum += w_val * hidden[c];
        }

        // Softcapping
        if (softcap > 0.0f) {
            sum = softcap * std::tanh(sum / softcap);
        }
        logits[v] = sum;
    }

    if (dump_tensors) {
        dump_tensor_f32(config_.dump_tensors_dir, dump_prefix + "logits", logits.data(), header_.vocab_size);
    }

    return logits;
}

void CpuReferenceRunner::generate(const std::string& prompt,
                                  int max_tokens,
                                  const SamplingParams& sampling,
                                  std::function<bool(uint32_t token, const std::string& piece)> on_token,
                                  std::atomic<bool>* cancel_flag) {
    if (!tokenizer_ || !tokenizer_->is_loaded()) {
        throw G4DenseFormatError("CpuReferenceRunner: tokenizer vocabulary not loaded");
    }

    std::vector<uint32_t> prompt_tokens = tokenizer_->encode(prompt, true);
    if (prompt_tokens.empty()) return;

    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        bool dump = (!config_.dump_tensors_dir.empty() && i == 0);
        forward_single_token(prompt_tokens[i], static_cast<uint32_t>(i), dump);
    }

    uint32_t next_token = prompt_tokens.back();
    std::vector<uint32_t> history = prompt_tokens;

    for (int step = 0; step < max_tokens; ++step) {
        if (cancel_flag && cancel_flag->load()) break;

        uint32_t pos = static_cast<uint32_t>(history.size() - 1);
        std::vector<float> logits = forward_single_token(next_token, pos);

        if (sampling.repetition_penalty != 1.0f) {
            apply_repetition_penalty(logits.data(), header_.vocab_size, history,
                                     sampling.repetition_penalty, header_.final_logit_softcapping);
        }

        uint64_t seed = seed_for(sampling, static_cast<int>(pos));
        uint32_t sampled = sample_token(logits.data(), header_.vocab_size, sampling, seed);

        if (sampled == tokenizer_->eos_id() || sampled == tokenizer_->end_of_turn_id()) {
            break;
        }

        std::string piece = tokenizer_->decode_single(sampled, true);
        if (on_token && !on_token(sampled, piece)) {
            break;
        }

        history.push_back(sampled);
        next_token = sampled;
    }
}

} // namespace g4dense
