#include "g4dense/draft_runtime.hpp"
#include "g4dense/runner.hpp"
#include "g4dense/format.hpp"
#include "g4dense/sampling.hpp"

#include <chrono>
#include <filesystem>
#include <stdexcept>

namespace g4dense {

DraftRuntime::DraftRuntime() = default;
DraftRuntime::~DraftRuntime() = default;

// The draft model runs on the GPU, like the target.
//
// It used to wrap CpuReferenceRunner -- the oracle -- which is minutes per token and made
// speculative decoding strictly slower than not speculating. E2B on the GPU measures ~13 tok/s
// against the 31B's ~0.8, which is the 15-20x ratio speculation needs to pay.
bool DraftRuntime::load_model(const std::string& model_path,
                              std::shared_ptr<VulkanContext> ctx,
                              std::shared_ptr<Tokenizer> tokenizer) {
    if (!std::filesystem::exists(model_path)) {
        loaded_ = false;
        runner_.reset();
        throw G4DenseFormatError("DraftRuntime: draft model checkpoint not found: " + model_path);
    }
    if (!ctx) {
        throw G4DenseFormatError("DraftRuntime: a VulkanContext is required");
    }

    tokenizer_ = tokenizer;
    runner_ = std::make_unique<ForwardRunner>(ctx, tokenizer, model_path);
    runner_->initialize();
    consumed_ = 0;
    loaded_ = true;
    return true;
}

void DraftRuntime::reset() {
    if (runner_) runner_->reset_kv_cache();
    consumed_ = 0;
}

DraftResult DraftRuntime::generate_draft_tokens(const std::vector<uint32_t>& context_tokens,
                                                uint32_t k,
                                                const SamplingParams& sampling) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    DraftResult result;
    if (!loaded_ || !runner_ || k == 0 || context_tokens.empty()) {
        return result;
    }

    const uint32_t vocab = runner_->header().vocab_size;
    result.draft_tokens.reserve(k);
    result.draft_logits.reserve(k);

    // Only the tokens this runtime has not already seen are fed in. The previous version
    // re-ran the entire context on every call, which is quadratic in the generated length and
    // would have swamped any speedup speculation produced.
    if (consumed_ > context_tokens.size()) {
        runner_->reset_kv_cache();
        consumed_ = 0;
    }
    std::vector<float> logits(vocab);
    while (consumed_ + 1 < context_tokens.size()) {
        runner_->forward_single_token(context_tokens[consumed_],
                                      static_cast<uint32_t>(consumed_), logits.data());
        ++consumed_;
    }

    uint32_t curr_token = context_tokens.back();
    uint32_t pos = static_cast<uint32_t>(consumed_);
    for (uint32_t step = 0; step < k; ++step) {
        runner_->forward_single_token(curr_token, pos, logits.data());
        ++pos;

        const uint64_t step_seed =
            splitmix64(sampling.has_seed ? (sampling.seed + step) : (1337 + step));
        const uint32_t cand = sample_token(logits.data(), vocab, sampling, step_seed);

        result.draft_tokens.push_back(cand);
        result.draft_logits.push_back(logits);
        curr_token = cand;
    }

    // The context token itself is now in the draft's KV cache; the drafted tokens are too, but
    // whether they survive depends on verification, so the caller re-syncs via accept().
    consumed_ = context_tokens.size();

    const auto t1 = std::chrono::high_resolution_clock::now();
    result.draft_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

void DraftRuntime::accept(size_t accepted_context_length) {
    // Drafted tokens that were rejected left entries in the draft's KV cache beyond this
    // point. They are never read again -- attention only looks at [first, current_position) --
    // and the next draft overwrites those ring slots, so no rollback is needed. Only the
    // bookkeeping has to agree with what the caller kept.
    consumed_ = accepted_context_length;
}

} // namespace g4dense
