#include "g4dense/draft_runtime.hpp"
#include "g4dense/cpu_reference.hpp"
#include "g4dense/format.hpp"
#include "g4dense/sampling.hpp"

#include <chrono>
#include <filesystem>
#include <stdexcept>

namespace g4dense {

DraftRuntime::DraftRuntime() = default;
DraftRuntime::~DraftRuntime() = default;

bool DraftRuntime::load_model(const std::string& model_path) {
    if (!std::filesystem::exists(model_path)) {
        loaded_ = false;
        runner_.reset();
        throw G4DenseFormatError("DraftRuntime: draft model checkpoint not found: " + model_path);
    }

    CpuReferenceConfig config;
    config.container_path = model_path;
    config.verbose = false;

    runner_ = std::make_unique<CpuReferenceRunner>(config, tokenizer_);
    runner_->initialize();
    loaded_ = true;
    return true;
}

DraftResult DraftRuntime::generate_draft_tokens(const std::vector<uint32_t>& prompt_tokens,
                                                uint32_t k,
                                                const SamplingParams& sampling) {
    auto t0 = std::chrono::high_resolution_clock::now();
    DraftResult result;
    if (!loaded_ || !runner_ || k == 0 || prompt_tokens.empty()) {
        return result;
    }

    result.draft_tokens.reserve(k);
    result.draft_logits.reserve(k);

    // Run context tokens through draft runner KV cache
    uint32_t pos = 0;
    for (size_t i = 0; i + 1 < prompt_tokens.size(); ++i) {
        runner_->forward_single_token(prompt_tokens[i], pos++);
    }

    uint32_t curr_token = prompt_tokens.back();
    for (uint32_t step = 0; step < k; ++step) {
        std::vector<float> logits = runner_->forward_single_token(curr_token, pos++);
        uint64_t step_seed = splitmix64(sampling.has_seed ? (sampling.seed + step) : (1337 + step));
        uint32_t draft_cand = sample_token(logits.data(), static_cast<uint32_t>(logits.size()), sampling, step_seed);

        result.draft_tokens.push_back(draft_cand);
        result.draft_logits.push_back(std::move(logits));
        curr_token = draft_cand;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.draft_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

} // namespace g4dense
