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
#include <windows.h>

namespace g4dense {

struct GenerationOptions {
    int max_tokens{512};
    SamplingParams sampling{};
    bool speculative_enabled{true};
    uint32_t draft_k{6};
    uint32_t active_tier_id{1};

    // Wrap the prompt in the Gemma 4 turn structure before encoding.
    //
    // These are instruction-tuned checkpoints: fed a bare prompt with no <|turn>user ...
    // <turn|> framing, they do not answer, they drift. "What is the capital of France?"
    // produced " la s s s" until this was applied -- degenerate output that GPU-vs-CPU
    // parity could never catch, because both paths were fed the same unframed tokens.
    //
    // Set false only to feed raw token text deliberately, e.g. base-model continuation or
    // an oracle diff that must control the exact input tokens.
    bool use_chat_template{true};
};

using TokenCallback = std::function<bool(uint32_t token, const std::string& piece)>;

struct LayerOffsetsGPU {
    uint32_t in_norm_off{0};
    uint32_t post_attn_norm_off{0};
    uint32_t pre_ffn_norm_off{0};
    uint32_t post_ffn_norm_off{0};
    uint32_t q_norm_off{0};
    uint32_t k_norm_off{0};
    uint32_t layer_scalar_off{0};
    float layer_scalar{0.089355f};

    // Per-layer embeddings (E2B/E4B). Zero-width on models without them.
    uint32_t post_ple_norm_off{0};
    // True when this layer has no k_proj/v_proj/k_norm of its own and reads another layer's
    // K/V cache instead.
    bool kv_shared{false};
    uint32_t kv_donor{0};

    struct ProjOffsets {
        uint32_t w_off{0};
        uint32_t s_off{0};
        uint32_t b_off{0};
        uint32_t rows{0};
        uint32_t in_dim{0};
    };

    ProjOffsets q_proj;
    ProjOffsets k_proj;
    ProjOffsets v_proj;
    ProjOffsets o_proj;
    ProjOffsets ple_gate;      // hidden -> ple_dim
    ProjOffsets ple_proj;      // ple_dim -> hidden

    ProjOffsets gate_proj;
    ProjOffsets up_proj;
    ProjOffsets down_proj;
};

class ForwardRunner {
public:
    ForwardRunner(std::shared_ptr<VulkanContext> vk_ctx,
                  std::shared_ptr<Tokenizer> tokenizer,
                  const std::string& container_path);
    ~ForwardRunner();

    void initialize();

    // GPU forward pass for a single token, producing logits on host/GPU
    // One weight pass for `batch` consecutive positions from `base_position`. `out_logits`
    // receives the LAST position's logits. batch == 1 is the decode path and is bit-identical
    // to the pre-batching implementation.
    // `all_positions` writes batch * vocab_size logits instead of just the last position's.
    // Speculative verification needs every position's distribution to decide how many drafted
    // tokens to accept, and the LM head is batched for it -- one 756 MB weight pass for all of
    // them rather than one per position.
    void forward_batch(const uint32_t* token_ids, uint32_t batch, uint32_t base_position,
                       float* out_logits, bool all_positions = false);

    void forward_single_token(uint32_t token_id, uint32_t position, float* out_logits);

    // End-to-end generation loop with token callback streaming
    void generate(const std::string& prompt,
                  const GenerationOptions& options,
                  TokenCallback on_token,
                  std::atomic<bool>* cancel_flag = nullptr);

    // Dynamic Tier switching without reloading the model container
    void switch_memory_tier(uint32_t tier_id);

    // Must match ATTN_MAX_SPAN in shaders/Attention.hlsl. Full-attention layers stage their
    // whole score span in groupshared, so any context above this overflows that array.
    static constexpr uint32_t kAttentionMaxSpan = 4096;

    uint32_t active_tier_id() const { return active_tier_id_; }
    std::shared_ptr<Tokenizer> tokenizer() const { return tokenizer_; }
    const std::string& container_path() const { return container_path_; }

    // Drops non-resident streamer slots; layers pinned by the active tier are kept.
    void clear_layer_cache() { if (streamer_) streamer_->clear_cache(); }

    // Drops accumulated K/V so an independent sequence can be run from position 0. generate()
    // does this per call; tests need it to compare a batched prefill against a sequential one.
    void reset_kv_cache() { if (kv_cache_) kv_cache_->reset(); }

    // Loads a draft model for speculative decoding, sharing this runner's Vulkan context.
    //
    // Deliberately explicit rather than automatic: the draft is resident alongside the target
    // and competes for the same import budget, so whether it is worth loading is a decision
    // with a measurable cost, not a default.
    bool load_draft_model(const std::string& path);

    // Hold back device memory from this model's layer import, so a draft model can be loaded
    // afterwards. Must be called BEFORE initialize(): the import is greedy, and once it has
    // taken the budget there is nothing left for a second model.
    void set_import_reserve(uint64_t bytes) { import_reserve_bytes_ = bytes; }
    bool has_draft() const { return draft_runtime_ && draft_runtime_->is_loaded(); }
    const G4DenseHeader& header() const { return header_; }
    TelemetrySnapshot get_latest_telemetry() const;

private:
    void allocate_gpu_resources();
    void compute_layer_offsets();

    std::shared_ptr<VulkanContext> vk_ctx_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::string container_path_;

    G4DenseHeader header_{};
    uint32_t active_tier_id_{1};
    uint64_t import_reserve_bytes_{0};

    // Layers held permanently in GPU-readable memory, imported once at load
    // (VK_EXT_external_memory_host) so the forward pass never copies them again. An entry with
    // a null buffer means that layer is still streamed through LayerStreamer.
    //
    // Import stops when the driver refuses, which it does at a total-memory ceiling rather than
    // a per-allocation one, so how many layers land here depends on the machine's BIOS UMA
    // split and on what else is allocated. That is why this is greedy rather than a fixed tier.
    std::vector<VkMemoryAllocation> resident_layer_bufs_;
    std::vector<void*> resident_regions_;      // VirtualAlloc bases, freed in the destructor
    std::vector<uint32_t> streamed_layers_;    // layers that did NOT fit, in ascending order

    void load_resident_layers();
    bool can_submit_work();
    std::vector<int> pinned_layers_;

    std::unique_ptr<VulkanPipelineManager> pipeline_mgr_;
    std::unique_ptr<LayerStreamer> streamer_;
    std::unique_ptr<KVCacheManager> kv_cache_;
    std::unique_ptr<DraftRuntime> draft_runtime_;
    std::unique_ptr<SpeculativeCoordinator> speculator_;

    std::atomic<bool> is_generating_{false};

    // Memory mapping for zero-copy container access
    HANDLE file_handle_{INVALID_HANDLE_VALUE};
    HANDLE mapping_handle_{NULL};
    const uint8_t* mapped_data_{nullptr};
    uint64_t file_size_{0};

    // Precomputed per-layer GPU layout
    std::vector<LayerOffsetsGPU> layer_gpu_offsets_;

    // GPU Command Pool & Buffer
    VkCommandPool cmd_pool_{VK_NULL_HANDLE};
    VkCommandBuffer cmd_{VK_NULL_HANDLE};
    VkFence fence_{VK_NULL_HANDLE};

    // GPU Activation Buffers
    VkMemoryAllocation buf_hidden_{};
    VkMemoryAllocation buf_norm_{};
    VkMemoryAllocation buf_q_{};
    VkMemoryAllocation buf_k_{};
    VkMemoryAllocation buf_v_{};
    VkMemoryAllocation buf_attn_out_{};
    VkMemoryAllocation buf_proj_out_{};
    VkMemoryAllocation buf_gate_{};
    VkMemoryAllocation buf_up_{};
    VkMemoryAllocation buf_ffn_out_{};
    VkMemoryAllocation buf_final_norm_w_{};
    VkMemoryAllocation buf_logits_{};

    // Tied LM head / embedding table, resident on the GPU: packed INT4 weights followed by
    // BF16 scales and biases, laid out exactly as in the container (~756 MB at vocab 262,144
    // x d_model 5,376).
    //
    // Without this the head ran as a scalar CPU loop over 1.41 billion multiply-adds per
    // token -- measured at roughly five cores pegged and the dominant term in a 23.6 s
    // forward pass, while LMHeadGreedy.hlsl sat compiled and unused. Uploading once at load
    // trades a one-time copy for that cost on every token.
    VkMemoryAllocation buf_lm_head_{};

    // Per-layer embeddings (E2B/E4B). Unallocated when header_.ple_dim == 0.
    //
    // embed_tokens_per_layer is NOT held on the device: at 262144 x 8960 it is 1.17 GB, and
    // only one row per token is ever needed, so it is dequantized from the mapping on the CPU
    // exactly like the main embedding. The projection matrix is small enough to upload once.
    VkMemoryAllocation buf_ple_w_{};       // per_layer_model_projection (weights+scales+biases)
    VkMemoryAllocation buf_ple_norm_w_{};  // per_layer_projection_norm
    VkMemoryAllocation buf_ple_emb_{};     // token-identity part, CPU-filled
    VkMemoryAllocation buf_ple_{};         // combined per-layer inputs
    VkMemoryAllocation buf_ple_g_{};       // in-layer gate scratch
    LayerOffsetsGPU::ProjOffsets ple_model_proj_{};
    uint64_t ple_emb_w_bytes_{0};
    uint64_t ple_emb_s_bytes_{0};
    uint64_t lm_head_w_bytes_{0};
    uint64_t lm_head_s_bytes_{0};
};

} // namespace g4dense
