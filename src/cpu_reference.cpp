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

namespace g4dense {

namespace {

// Half precision / BF16 conversion helper
float bf16_to_f32(uint16_t val) {
    uint32_t u32 = static_cast<uint32_t>(val) << 16;
    float f;
    std::memcpy(&f, &u32, sizeof(float));
    return f;
}

// Gemma GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
float gelu_tanh(float x) {
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

// RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * (1.0 + weight) or weight
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

CpuReferenceRunner::~CpuReferenceRunner() = default;

void CpuReferenceRunner::initialize() {
    std::ifstream f(config_.container_path, std::ios::binary);
    if (!f.is_open()) {
        throw G4DenseFormatError("CpuReferenceRunner: cannot open container " + config_.container_path);
    }
    f.read(reinterpret_cast<char*>(&header_), sizeof(G4DenseHeader));
    if (f.gcount() != sizeof(G4DenseHeader)) {
        throw G4DenseFormatError("CpuReferenceRunner: truncated header");
    }
    validate_header(header_);

    f.seekg(0, std::ios::end);
    size_t total_size = f.tellg();
    f.seekg(0, std::ios::beg);

    container_data_.resize(total_size);
    f.read(reinterpret_cast<char*>(container_data_.data()), total_size);

    k_cache_.resize(header_.num_layers);
    v_cache_.resize(header_.num_layers);

    if (!config_.dump_tensors_dir.empty()) {
        std::filesystem::create_directories(config_.dump_tensors_dir);
    }
}

std::vector<float> CpuReferenceRunner::forward_single_token(uint32_t token, uint32_t position, bool dump_tensors) {
    if (token >= header_.vocab_size) {
        throw G4DenseFormatError("Token ID exceeds vocab size");
    }

    uint32_t d_model = header_.d_model;
    uint32_t d_ff = header_.d_ff;
    uint32_t q_heads = header_.num_q_heads;
    uint32_t kv_heads = header_.num_kv_heads;
    uint32_t head_dim = header_.head_dim;
    uint32_t g_size = header_.quant_group_size;

    // Embedding Dequantization for token
    const uint8_t* embed_ptr = container_data_.data() + header_.embed_offset;
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

    if (dump_tensors) {
        dump_tensor_f32(config_.dump_tensors_dir, "token0_embed", hidden.data(), d_model);
    }

    std::vector<float> norm_buf(d_model);
    std::vector<float> q_buf(q_heads * head_dim);
    std::vector<float> k_buf(kv_heads * head_dim);
    std::vector<float> v_buf(kv_heads * head_dim);
    std::vector<float> attn_out(d_model);
    std::vector<float> gate_buf(d_ff);
    std::vector<float> up_buf(d_ff);
    std::vector<float> ffn_out(d_model);

    // Iterate through Layers 0..num_layers-1
    for (uint32_t l = 0; l < header_.num_layers; ++l) {
        const uint8_t* layer_ptr = container_data_.data() + header_.layer_offsets[l];
        bool is_global = (header_.global_layer_mask & (1ULL << l)) != 0;

        // Parse Norms (BF16)
        const uint16_t* raw_in_norm = reinterpret_cast<const uint16_t*>(layer_ptr);
        const uint16_t* raw_post_attn_norm = raw_in_norm + d_model;
        const uint16_t* raw_pre_ffn_norm = raw_post_attn_norm + d_model;
        const uint16_t* raw_post_ffn_norm = raw_pre_ffn_norm + d_model;
        const uint16_t* raw_q_norm = raw_post_ffn_norm + d_model;
        const uint16_t* raw_k_norm = raw_q_norm + head_dim;
        const uint8_t* weights_start = reinterpret_cast<const uint8_t*>(raw_k_norm + head_dim);

        std::vector<float> in_norm_f32(d_model), post_attn_norm_f32(d_model);
        std::vector<float> pre_ffn_norm_f32(d_model), post_ffn_norm_f32(d_model);
        std::vector<float> q_norm_f32(head_dim), k_norm_f32(head_dim);

        for (uint32_t i = 0; i < d_model; ++i) {
            in_norm_f32[i] = bf16_to_f32(raw_in_norm[i]);
            post_attn_norm_f32[i] = bf16_to_f32(raw_post_attn_norm[i]);
            pre_ffn_norm_f32[i] = bf16_to_f32(raw_pre_ffn_norm[i]);
            post_ffn_norm_f32[i] = bf16_to_f32(raw_post_ffn_norm[i]);
        }
        for (uint32_t i = 0; i < head_dim; ++i) {
            q_norm_f32[i] = bf16_to_f32(raw_q_norm[i]);
            k_norm_f32[i] = bf16_to_f32(raw_k_norm[i]);
        }

        // 1. Input Layernorm
        rms_norm(hidden.data(), in_norm_f32.data(), d_model, 1e-6f, norm_buf.data());

        // Dequantize Attention weights
        auto unpack_proj = [&](const uint8_t*& ptr, uint32_t rows, uint32_t cols, std::vector<float>& out) {
            uint32_t p_cols = cols / 8;
            uint32_t g_cols = cols / g_size;
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(ptr);
            ptr += rows * p_cols * sizeof(uint32_t);
            const uint16_t* ps = reinterpret_cast<const uint16_t*>(ptr);
            ptr += rows * g_cols * sizeof(uint16_t);
            const uint16_t* pb = reinterpret_cast<const uint16_t*>(ptr);
            ptr += rows * g_cols * sizeof(uint16_t);
            dequantize_affine_int4_matrix(pw, ps, pb, rows, cols, g_size, out);
        };

        const uint8_t* cur_w_ptr = weights_start;
        std::vector<float> w_q, w_k, w_v, w_o;
        unpack_proj(cur_w_ptr, q_heads * head_dim, d_model, w_q);
        unpack_proj(cur_w_ptr, kv_heads * head_dim, d_model, w_k);
        unpack_proj(cur_w_ptr, kv_heads * head_dim, d_model, w_v);
        unpack_proj(cur_w_ptr, d_model, q_heads * head_dim, w_o);

        // Compute Q, K, V
        gemv(w_q.data(), norm_buf.data(), q_heads * head_dim, d_model, q_buf.data());
        gemv(w_k.data(), norm_buf.data(), kv_heads * head_dim, d_model, k_buf.data());
        gemv(w_v.data(), norm_buf.data(), kv_heads * head_dim, d_model, v_buf.data());

        // Apply Q-norm and K-norm
        for (uint32_t h = 0; h < q_heads; ++h) {
            rms_norm(q_buf.data() + h * head_dim, q_norm_f32.data(), head_dim, 1e-6f, q_buf.data() + h * head_dim);
        }
        for (uint32_t h = 0; h < kv_heads; ++h) {
            rms_norm(k_buf.data() + h * head_dim, k_norm_f32.data(), head_dim, 1e-6f, k_buf.data() + h * head_dim);
        }

        // Apply RoPE
        float theta_base = is_global ? header_.rope_theta_global : header_.rope_theta_local;
        uint32_t rotary_dims = is_global ? (head_dim / 4) : head_dim;

        auto apply_rope = [&](float* vec, uint32_t heads) {
            for (uint32_t h = 0; h < heads; ++h) {
                float* head_vec = vec + h * head_dim;
                for (uint32_t d = 0; d < rotary_dims; d += 2) {
                    float freq = 1.0f / std::pow(theta_base, static_cast<float>(d) / static_cast<float>(head_dim));
                    float angle = static_cast<float>(position) * freq;
                    float cos_a = std::cos(angle);
                    float sin_a = std::sin(angle);
                    float v0 = head_vec[d];
                    float v1 = head_vec[d + 1];
                    head_vec[d] = v0 * cos_a - v1 * sin_a;
                    head_vec[d + 1] = v0 * sin_a + v1 * cos_a;
                }
            }
        };

        apply_rope(q_buf.data(), q_heads);
        apply_rope(k_buf.data(), kv_heads);

        // Store K, V in Cache
        k_cache_[l].insert(k_cache_[l].end(), k_buf.begin(), k_buf.end());
        v_cache_[l].insert(v_cache_[l].end(), v_buf.begin(), v_buf.end());

        // Multi-head Attention
        uint32_t total_ctx = position + 1;
        uint32_t group_ratio = q_heads / kv_heads;
        std::vector<float> head_out(q_heads * head_dim, 0.0f);

        for (uint32_t h = 0; h < q_heads; ++h) {
            uint32_t kv_h = h / group_ratio;
            const float* q_h = q_buf.data() + h * head_dim;

            std::vector<float> scores(total_ctx);
            for (uint32_t t = 0; t < total_ctx; ++t) {
                // Sliding window mask check
                if (!is_global && position >= header_.sliding_window && t + header_.sliding_window <= position) {
                    scores[t] = -1e9f;
                    continue;
                }

                const float* k_t = k_cache_[l].data() + t * (kv_heads * head_dim) + kv_h * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    dot += q_h[d] * k_t[d];
                }
                scores[t] = dot / std::sqrt(static_cast<float>(head_dim));
            }

            // Softmax
            float max_s = -1e9f;
            for (float s : scores) max_s = std::max(max_s, s);
            float exp_sum = 0.0f;
            for (float& s : scores) {
                s = std::exp(s - max_s);
                exp_sum += s;
            }
            float inv_sum = 1.0f / exp_sum;
            for (float& s : scores) s *= inv_sum;

            // Weighted sum of V
            float* out_h = head_out.data() + h * head_dim;
            for (uint32_t t = 0; t < total_ctx; ++t) {
                float weight = scores[t];
                if (weight > 0.0f) {
                    const float* v_t = v_cache_[l].data() + t * (kv_heads * head_dim) + kv_h * head_dim;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        out_h[d] += weight * v_t[d];
                    }
                }
            }
        }

        // O proj and Post Attention Norm
        gemv(w_o.data(), head_out.data(), d_model, q_heads * head_dim, attn_out.data());
        rms_norm(attn_out.data(), post_attn_norm_f32.data(), d_model, 1e-6f, attn_out.data());

        // Residual Add
        for (uint32_t i = 0; i < d_model; ++i) {
            hidden[i] += attn_out[i];
        }

        // 2. FFN (SwiGLU)
        rms_norm(hidden.data(), pre_ffn_norm_f32.data(), d_model, 1e-6f, norm_buf.data());

        std::vector<float> w_gate, w_up, w_down;
        unpack_proj(cur_w_ptr, d_ff, d_model, w_gate);
        unpack_proj(cur_w_ptr, d_ff, d_model, w_up);
        unpack_proj(cur_w_ptr, d_model, d_ff, w_down);

        gemv(w_gate.data(), norm_buf.data(), d_ff, d_model, gate_buf.data());
        gemv(w_up.data(), norm_buf.data(), d_ff, d_model, up_buf.data());

        for (uint32_t i = 0; i < d_ff; ++i) {
            gate_buf[i] = gelu_tanh(gate_buf[i]) * up_buf[i];
        }

        gemv(w_down.data(), gate_buf.data(), d_model, d_ff, ffn_out.data());
        rms_norm(ffn_out.data(), post_ffn_norm_f32.data(), d_model, 1e-6f, ffn_out.data());

        // Residual Add
        for (uint32_t i = 0; i < d_model; ++i) {
            hidden[i] += ffn_out[i];
        }

        if (dump_tensors && l == 0) {
            dump_tensor_f32(config_.dump_tensors_dir, "token0_layer0_hidden", hidden.data(), d_model);
        }
    }

    // Final Logits & Softcapping
    std::vector<float> logits(header_.vocab_size);
    std::vector<float> full_embed_table;
    dequantize_affine_int4_matrix(embed_w, embed_s, embed_b, header_.vocab_size, d_model, g_size, full_embed_table);

    gemv(full_embed_table.data(), hidden.data(), header_.vocab_size, d_model, logits.data());

    float softcap = header_.final_logit_softcapping;
    for (float& l_val : logits) {
        l_val = softcap * std::tanh(l_val / softcap);
    }

    if (dump_tensors) {
        dump_tensor_f32(config_.dump_tensors_dir, "token0_logits", logits.data(), header_.vocab_size);
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
        forward_single_token(prompt_tokens[i], static_cast<uint32_t>(i));
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
