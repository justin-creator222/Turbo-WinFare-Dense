#pragma once

#include "g4dense/sampling.hpp"
#include "g4dense/draft_runtime.hpp"
#include <vector>
#include <cstdint>

namespace g4dense {

struct SpeculativeEvaluation {
    std::vector<uint32_t> accepted_tokens; // N accepted + 1 bonus token
    uint32_t num_accepted{0};
    uint32_t total_drafted{0};
    bool bonus_token_sampled{false};
    uint32_t bonus_token{0};
    double acceptance_rate{0.0};
};

class SpeculativeCoordinator {
public:
    SpeculativeCoordinator() = default;

    // Evaluates draft candidate tokens against 31B verification logits (batch M=K)
    SpeculativeEvaluation evaluate_verification(
        const std::vector<uint32_t>& draft_tokens,
        const std::vector<const float*>& target_logits_per_pos,
        uint32_t vocab_size,
        const SamplingParams& sampling,
        uint64_t base_seed
    );

    // Cumulative telemetry metrics
    uint64_t total_draft_tokens() const { return total_draft_tokens_; }
    uint64_t total_accepted_tokens() const { return total_accepted_tokens_; }
    double cumulative_acceptance_rate() const {
        return total_draft_tokens_ == 0 ? 0.0 : (double)total_accepted_tokens_ / total_draft_tokens_;
    }

private:
    uint64_t total_draft_tokens_{0};
    uint64_t total_accepted_tokens_{0};
};

} // namespace g4dense
