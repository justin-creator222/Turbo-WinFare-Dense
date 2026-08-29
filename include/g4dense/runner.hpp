#pragma once

#include "g4dense/format.hpp"
#include "g4dense/manifest.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/vk_pipeline.hpp"
#include "g4dense/streamer.hpp"
#include "g4dense/kv_cache.hpp"
#include "g4dense/draft_runtime.hpp"
#include "g4dense/speculator.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/sampling.hpp"
#include "g4dense/telemetry.hpp"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>

namespace g4dense {

struct GenerationOptions {
    int max_tokens{512};
    SamplingParams sampling{};
    bool speculative_enabled{true};
    uint32_t draft_k{6};
    uint32_t active_tier_id{1};
};

using TokenCallback = std::function<bool(uint32_t token, const std::string& piece)>;

class ForwardRunner {
public:
    ForwardRunner(std::shared_ptr<VulkanContext> vk_ctx,
                  std::shared_ptr<Tokenizer> tokenizer,
                  const std::string& container_path);
    ~ForwardRunner();

    void initialize();

    // End-to-end generation loop with token callback streaming
    void generate(const std::string& prompt,
                  const GenerationOptions& options,
                  TokenCallback on_token,
                  std::atomic<bool>* cancel_flag = nullptr);

    // Dynamic Tier switching without reloading the model container
    void switch_memory_tier(uint32_t tier_id);

    uint32_t active_tier_id() const { return active_tier_id_; }
    const G4DenseHeader& header() const { return header_; }
    TelemetrySnapshot get_latest_telemetry() const;

private:
    std::shared_ptr<VulkanContext> vk_ctx_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::string container_path_;

    G4DenseHeader header_{};
    uint32_t active_tier_id_{1};

    std::unique_ptr<VulkanPipelineManager> pipeline_mgr_;
    std::unique_ptr<LayerStreamer> streamer_;
    std::unique_ptr<KVCacheManager> kv_cache_;
    std::unique_ptr<DraftRuntime> draft_runtime_;
    std::unique_ptr<SpeculativeCoordinator> speculator_;

    std::atomic<bool> is_generating_{false};
};

} // namespace g4dense
