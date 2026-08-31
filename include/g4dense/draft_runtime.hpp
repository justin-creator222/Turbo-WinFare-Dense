#pragma once

#include "g4dense/sampling.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/vk_context.hpp"
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

    // Loads the draft checkpoint onto the GPU, sharing the caller's Vulkan context so the
    // two models are not competing for separate devices or duplicating pipelines.
    bool load_model(const std::string& model_path,
                    std::shared_ptr<VulkanContext> ctx,
                    std::shared_ptr<Tokenizer> tokenizer);

    // Drafts K candidate tokens continuing `context_tokens`. Only the tokens not already in
    // this runtime's KV cache are re-run, so repeated calls are incremental rather than
    // quadratic in the generated length.
    DraftResult generate_draft_tokens(const std::vector<uint32_t>& context_tokens,
                                      uint32_t k,
                                      const SamplingParams& sampling);

    // Tell the draft how much of the context actually survived verification.
    void accept(size_t accepted_context_length);

    // Drop all cached state, for an independent sequence.
    void reset();

    bool is_loaded() const { return loaded_; }
    uint32_t draft_k() const { return default_k_; }
    void set_draft_k(uint32_t k) { default_k_ = k; }

private:
    bool loaded_{false};
    uint32_t default_k_{6};
    int num_threads_{8}; // Zen 4 physical cores
    std::shared_ptr<Tokenizer> tokenizer_;
    size_t consumed_{0};   // context tokens already in the draft's KV cache
    std::unique_ptr<class ForwardRunner> runner_;
};

} // namespace g4dense
