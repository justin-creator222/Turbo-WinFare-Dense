#include "g4dense/speculator.hpp"
#include "g4dense/sampling.hpp"

#include <cmath>
#include <algorithm>
#include <iostream>

namespace g4dense {

SpeculativeEvaluation SpeculativeCoordinator::evaluate_verification(
    const std::vector<uint32_t>& draft_tokens,
    const std::vector<const float*>& target_logits_per_pos,
    uint32_t vocab_size,
    const SamplingParams& sampling,
    uint64_t base_seed
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

        // Sample target token from verification logits
        uint32_t target_tok = sample_token(logits, vocab_size, sampling, step_seed);

        if (sampling.is_greedy()) {
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
            // Sampling mode
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
