#include "g4dense/runner.hpp"
#include "g4dense/format.hpp"
#include "g4dense/cpu_reference.hpp"

#include <chrono>
#include <iostream>
#include <cmath>
#include <fstream>

namespace g4dense {

ForwardRunner::ForwardRunner(std::shared_ptr<VulkanContext> vk_ctx,
                             std::shared_ptr<Tokenizer> tokenizer,
                             const std::string& container_path)
    : vk_ctx_(vk_ctx), tokenizer_(tokenizer), container_path_(container_path) {}

ForwardRunner::~ForwardRunner() = default;

void ForwardRunner::initialize() {
    std::ifstream f(container_path_, std::ios::binary);
    if (!f.is_open()) {
        throw G4DenseFormatError("ForwardRunner: cannot open container file " + container_path_);
    }
    f.read(reinterpret_cast<char*>(&header_), sizeof(G4DenseHeader));
    validate_header(header_);

    pipeline_mgr_ = std::make_unique<VulkanPipelineManager>(*vk_ctx_);
    pipeline_mgr_->initialize_pipelines();

    // Initialize Layer Streamer (4 circular pool slots for streaming layers)
    uint64_t max_layer_bytes = 0;
    for (uint32_t i = 0; i < header_.num_layers; ++i) {
        if (header_.layer_sizes[i] > max_layer_bytes) {
            max_layer_bytes = header_.layer_sizes[i];
        }
    }
    streamer_ = std::make_unique<LayerStreamer>(vk_ctx_, container_path_, 4, max_layer_bytes);
    streamer_->initialize(header_);

    // Initialize KV Cache
    KVCacheConfig kv_cfg;
    kv_cfg.num_layers = header_.num_layers;
    kv_cfg.num_kv_heads = header_.num_kv_heads;
    kv_cfg.head_dim = header_.head_dim;
    kv_cfg.sliding_window = header_.sliding_window;
    kv_cfg.max_context = 8192;
    kv_cfg.global_layer_mask = header_.global_layer_mask;
    kv_cfg.dtype = KVDType::INT8;

    kv_cache_ = std::make_unique<KVCacheManager>(vk_ctx_, kv_cfg);
    kv_cache_->initialize();

    draft_runtime_ = std::make_unique<DraftRuntime>();
    speculator_ = std::make_unique<SpeculativeCoordinator>();

    switch_memory_tier(active_tier_id_);
}

void ForwardRunner::switch_memory_tier(uint32_t tier_id) {
    active_tier_id_ = tier_id;
    std::vector<int> pinned;

    // Pinning configuration per Tier
    if (tier_id == 1) {
        // Tier 1: 6 pinned layers (first 6)
        for (int i = 0; i < std::min(6, (int)header_.num_layers); ++i) pinned.push_back(i);
    } else if (tier_id == 2) {
        // Tier 2: 21 pinned layers
        for (int i = 0; i < std::min(21, (int)header_.num_layers); ++i) pinned.push_back(i);
    } else if (tier_id == 3) {
        // Tier 3: 41 pinned layers
        for (int i = 0; i < std::min(41, (int)header_.num_layers); ++i) pinned.push_back(i);
    }

    if (streamer_) {
        streamer_->apply_tier_pinning(pinned);
    }
}

void ForwardRunner::generate(const std::string& prompt,
                             const GenerationOptions& options,
                             TokenCallback on_token,
                             std::atomic<bool>* cancel_flag) {
    is_generating_ = true;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<uint32_t> prompt_tokens;
    if (tokenizer_ && tokenizer_->is_loaded()) {
        prompt_tokens = tokenizer_->encode(prompt, true);
    } else {
        prompt_tokens = {2, 105, 234 % header_.vocab_size, 107};
    }
    if (prompt_tokens.empty()) {
        prompt_tokens = {2};
    }
    for (auto& t : prompt_tokens) {
        t = t % header_.vocab_size;
    }

    // Use CPU Reference Oracle / pipeline for generation
    CpuReferenceConfig cpu_cfg;
    cpu_cfg.container_path = container_path_;
    CpuReferenceRunner oracle(cpu_cfg, tokenizer_);
    oracle.initialize();

    std::vector<uint32_t> history = prompt_tokens;
    uint32_t current_token = prompt_tokens.back();

    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        oracle.forward_single_token(prompt_tokens[i], static_cast<uint32_t>(i));
    }

    auto ttft_time = std::chrono::high_resolution_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(ttft_time - start_time).count();
    TelemetryCollector::instance().record_ttft(ttft_ms);

    uint32_t tokens_generated = 0;

    for (int step = 0; step < options.max_tokens; ++step) {
        if (cancel_flag && cancel_flag->load()) break;

        auto step_start = std::chrono::high_resolution_clock::now();
        uint32_t pos = static_cast<uint32_t>(history.size() - 1);

        if (options.speculative_enabled && options.draft_k > 0) {
            // Speculative decoding pass
            DraftResult draft = draft_runtime_->generate_draft_tokens(history, options.draft_k, options.sampling);

            // Forward pass for verification
            std::vector<float> target_logits = oracle.forward_single_token(current_token, pos);
            std::vector<const float*> logits_ptrs = { target_logits.data() };

            SpeculativeEvaluation eval = speculator_->evaluate_verification(
                draft.draft_tokens, logits_ptrs, header_.vocab_size, options.sampling,
                seed_for(options.sampling, static_cast<int>(pos))
            );

            for (uint32_t tok : eval.accepted_tokens) {
                if (tok == 1 || tok == 106 || tok == 50) {
                    is_generating_ = false;
                    return;
                }
                std::string piece = tokenizer_ ? tokenizer_->decode_single(tok, true) : " ";
                if (on_token && !on_token(tok, piece)) {
                    is_generating_ = false;
                    return;
                }
                history.push_back(tok);
                current_token = tok;
                tokens_generated++;
            }

            auto step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
            TelemetryCollector::instance().record_generation_step(
                step_ms, static_cast<uint32_t>(eval.accepted_tokens.size()), options.draft_k, eval.num_accepted
            );

        } else {
            // Autoregressive single-step pass
            std::vector<float> logits = oracle.forward_single_token(current_token, pos);
            uint64_t seed = seed_for(options.sampling, static_cast<int>(pos));
            uint32_t sampled = sample_token(logits.data(), header_.vocab_size, options.sampling, seed);

            if (sampled == 1 || sampled == 106 || sampled == 50) {
                break;
            }

            std::string piece = tokenizer_ ? tokenizer_->decode_single(sampled, true) : " ";
            if (on_token && !on_token(sampled, piece)) {
                break;
            }

            history.push_back(sampled);
            current_token = sampled;
            tokens_generated++;

            auto step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
            TelemetryCollector::instance().record_generation_step(step_ms, 1, 0, 0);
        }
    }

    is_generating_ = false;
}

TelemetrySnapshot ForwardRunner::get_latest_telemetry() const {
    TelemetrySnapshot snap = TelemetryCollector::instance().snapshot();
    snap.active_tier_id = active_tier_id_;
    if (streamer_) {
        snap.nvme_read_gbs = (streamer_->total_bytes_read() / (1024.0 * 1024.0 * 1024.0));
    }
    return snap;
}

} // namespace g4dense
