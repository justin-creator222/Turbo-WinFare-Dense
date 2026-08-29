#pragma once

#include "g4dense/sampling.hpp"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace g4dense {

struct DraftResult {
    std::vector<uint32_t> draft_tokens;
    std::vector<std::vector<float>> draft_logits; // per draft position logits
    double draft_time_ms{0.0};
};

class DraftRuntime {
public:
    DraftRuntime();
    ~DraftRuntime();

    // Loads E2B draft model checkpoint
    bool load_model(const std::string& model_path);

    // Generates K candidate draft tokens starting from current context
    DraftResult generate_draft_tokens(const std::vector<uint32_t>& prompt_tokens,
                                      uint32_t k,
                                      const SamplingParams& sampling);

    bool is_loaded() const { return loaded_; }
    uint32_t draft_k() const { return default_k_; }
    void set_draft_k(uint32_t k) { default_k_ = k; }

private:
    bool loaded_{false};
    uint32_t default_k_{6};
    int num_threads_{8}; // Zen 4 physical cores
};

} // namespace g4dense
