#include "g4dense/speculator.hpp"
#include "g4dense/sampling.hpp"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <random>

namespace g4dense {

namespace {

void compute_softmax(float* logits, uint32_t n, float temp) {
    if (temp <= 0.0f) temp = 1.0f;
    float max_l = -1e9f;
    for (uint32_t i = 0; i < n; ++i) {
        logits[i] /= temp;
        if (logits[i] > max_l) max_l = logits[i];
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        logits[i] = std::exp(logits[i] - max_l);
        sum += logits[i];
    }
    float inv = 1.0f / (sum > 1e-12f ? sum : 1.0f);
    for (uint32_t i = 0; i < n; ++i) {
        logits[i] *= inv;
    }
}

} // namespace

SpeculativeEvaluation SpeculativeCoordinator::evaluate_verification(
    const std::vector<uint32_t>& draft_tokens,
    const std::vector<const float*>& target_logits_per_pos,
    uint32_t vocab_size,
    const SamplingParams& sampling,
    uint64_t base_seed,
    const std::vector<std::vector<float>>& draft_logits
) {
    SpeculativeEvaluation eval{};
    eval.total_drafted = static_cast<uint32_t>(draft_tokens.size());
    total_draft_tokens_ += eval.total_drafted;

    if (draft_tokens.empty() || target_logits_per_pos.empty()) {
        return eval;
    }

    size_t k = std::min(draft_tokens.size(), target_logits_per_pos.size());
    size_t accepted_count = 0;

    for (size_t i = 0; i < k; ++i) {
        uint32_t draft_tok = draft_tokens[i];
        const float* logits = target_logits_per_pos[i];
        uint64_t step_seed = splitmix64(base_seed + i);

        if (sampling.is_greedy()) {
            uint32_t target_tok = sample_token(logits, vocab_size, sampling, step_seed);
            if (draft_tok == target_tok) {
                eval.accepted_tokens.push_back(draft_tok);
                accepted_count++;
            } else {
                // First mismatch: reject draft_tok, emit target_tok as bonus/correction token
                eval.bonus_token = target_tok;
                eval.bonus_token_sampled = true;
                eval.accepted_tokens.push_back(target_tok);
                break;
            }
        } else {
            // Rejection sampling using probability ratio
            bool has_draft_probs = (i < draft_logits.size() && !draft_logits[i].empty());
            if (has_draft_probs) {
                // Compute softmax probabilities
                std::vector<float> p_target(vocab_size);
                std::memcpy(p_target.data(), logits, vocab_size * sizeof(float));
                compute_softmax(p_target.data(), vocab_size, sampling.temperature);

                std::vector<float> q_draft(vocab_size);
                std::memcpy(q_draft.data(), draft_logits[i].data(), vocab_size * sizeof(float));
                compute_softmax(q_draft.data(), vocab_size, sampling.temperature);

                float p_val = (draft_tok < vocab_size) ? p_target[draft_tok] : 0.0f;
                float q_val = (draft_tok < vocab_size) ? q_draft[draft_tok] : 0.0f;
                float r = (q_val > 1e-12f) ? (p_val / q_val) : 1.0f;

                std::mt19937_64 rng(step_seed);
                std::uniform_real_distribution<float> udist(0.0f, 1.0f);
                float u = udist(rng);

                if (u <= r) {
                    eval.accepted_tokens.push_back(draft_tok);
                    accepted_count++;
                } else {
                    // Rejection: sample from (p - q)^+
                    std::vector<float> diff(vocab_size, 0.0f);
                    float sum_diff = 0.0f;
                    for (uint32_t v = 0; v < vocab_size; ++v) {
                        float d = std::max(0.0f, p_target[v] - q_draft[v]);
                        diff[v] = d;
                        sum_diff += d;
                    }
                    uint32_t corr_tok = 0;
                    if (sum_diff > 1e-9f) {
                        float r_val = udist(rng) * sum_diff;
                        float acc = 0.0f;
                        for (uint32_t v = 0; v < vocab_size; ++v) {
                            acc += diff[v];
                            if (acc >= r_val) {
                                corr_tok = v;
                                break;
                            }
                        }
                    } else {
                        corr_tok = sample_token(logits, vocab_size, sampling, step_seed);
                    }

                    eval.bonus_token = corr_tok;
                    eval.bonus_token_sampled = true;
                    eval.accepted_tokens.push_back(corr_tok);
                    break;
                }
            } else {
                uint32_t target_tok = sample_token(logits, vocab_size, sampling, step_seed);
                if (draft_tok == target_tok) {
                    eval.accepted_tokens.push_back(draft_tok);
                    accepted_count++;
                } else {
                    eval.bonus_token = target_tok;
                    eval.bonus_token_sampled = true;
                    eval.accepted_tokens.push_back(target_tok);
                    break;
                }
            }
        }
    }

    // If all K tokens accepted, sample K+1 bonus token if available
    if (accepted_count == k && target_logits_per_pos.size() > k) {
        const float* bonus_logits = target_logits_per_pos[k];
        uint64_t bonus_seed = splitmix64(base_seed + k);
        uint32_t bonus_tok = sample_token(bonus_logits, vocab_size, sampling, bonus_seed);
        eval.bonus_token = bonus_tok;
        eval.bonus_token_sampled = true;
        eval.accepted_tokens.push_back(bonus_tok);
    }

    eval.num_accepted = static_cast<uint32_t>(accepted_count);
    total_accepted_tokens_ += eval.num_accepted;
    eval.acceptance_rate = eval.total_drafted == 0 ? 0.0 : (double)eval.num_accepted / eval.total_drafted;

    return eval;
}

} // namespace g4dense
