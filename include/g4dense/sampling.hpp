#pragma once

#include "g4dense/format.hpp"
#include <cstdint>
#include <vector>

namespace g4dense {

struct SamplingParams {
    float temperature{0.0f};        // 0 = greedy
    float top_p{1.0f};              // 1 = no nucleus truncation
    int   top_k{0};                 // 0 = no top-k truncation
    float repetition_penalty{1.0f}; // 1 = disabled
    bool  has_seed{false};
    uint64_t seed{0};

    bool is_greedy() const { return temperature <= 0.0f && repetition_penalty == 1.0f; }
};

void validate_sampling(const SamplingParams& p);
uint64_t splitmix64(uint64_t x);
uint64_t seed_for(const SamplingParams& p, int position);
void apply_repetition_penalty(float* logits, uint32_t vocab,
                              const std::vector<uint32_t>& history,
                              float penalty, float softcap);
uint32_t sample_token(const float* logits, uint32_t vocab,
                      const SamplingParams& p, uint64_t seed);

} // namespace g4dense
