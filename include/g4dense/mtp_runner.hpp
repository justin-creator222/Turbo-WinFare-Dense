#pragma once

#include "g4dense/format.hpp"
#include "g4dense/sampling.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/vk_pipeline.hpp"
#include "g4dense/kv_cache.hpp"
#include "g4dense/draft_runtime.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <functional>

namespace g4dense {

struct MtpLayerOffsets {
    uint32_t in_norm_off{0};
    uint32_t post_attn_norm_off{0};
    uint32_t pre_ffn_norm_off{0};
    uint32_t post_ffn_norm_off{0};
    uint32_t q_norm_off{0};
    uint32_t layer_scalar_off{0};
    float layer_scalar{1.0f};

    uint32_t q_proj_w_off{0};
    uint32_t q_proj_s_off{0};
    uint32_t q_proj_b_off{0};
    uint32_t q_proj_rows{0};
    uint32_t q_proj_in_dim{0};

    uint32_t o_proj_w_off{0};
    uint32_t o_proj_s_off{0};
    uint32_t o_proj_b_off{0};
    uint32_t o_proj_rows{0};
    uint32_t o_proj_in_dim{0};

    uint32_t gate_proj_w_off{0};
    uint32_t gate_proj_s_off{0};
    uint32_t gate_proj_b_off{0};
    uint32_t gate_proj_rows{0};
    uint32_t gate_proj_in_dim{0};

    uint32_t up_proj_w_off{0};
    uint32_t up_proj_s_off{0};
    uint32_t up_proj_b_off{0};
    uint32_t up_proj_rows{0};
    uint32_t up_proj_in_dim{0};

    uint32_t down_proj_w_off{0};
    uint32_t down_proj_s_off{0};
    uint32_t down_proj_b_off{0};
    uint32_t down_proj_rows{0};
    uint32_t down_proj_in_dim{0};
};

class MtpRunner {
public:
    MtpRunner();
    ~MtpRunner();

    // Loads the .g4mtp assistant model checkpoint onto the GPU, sharing the target model's
    // Vulkan context so both models run on the same device.
    bool load_model(const std::string& model_path, std::shared_ptr<VulkanContext> ctx);

    // Generates K candidate draft tokens using the MTP 4-layer drafter.
    // initial_target_hidden_state: pointer to target model's final post-RMSNorm hidden state (d=5376)
    // seed_token: the token to start drafting from
    // context_length: current context length in target KV cache
    // k: number of draft tokens to produce
    // sampling: sampling parameters
    // get_embedding_fn: callback to look up target model's token embedding vector (d=5376)
    // kv_cache: target model's KV cache manager (shared KV donor layers 58 & 59)
    DraftResult generate_draft_tokens(
        const float* initial_target_hidden_state,
        uint32_t seed_token,
        uint32_t context_length,
        uint32_t k,
        const SamplingParams& sampling,
        const std::function<void(uint32_t, float*)>& get_embedding_fn,
        KVCacheManager* kv_cache);

    bool is_loaded() const { return loaded_; }
    const G4MtpHeader& header() const { return header_; }

    // Memory footprint in bytes on GPU
    uint64_t total_gpu_bytes() const { return total_gpu_bytes_; }

private:
    void compute_layer_offsets();
    void allocate_gpu_resources();
    void cleanup();

    bool loaded_{false};
    std::shared_ptr<VulkanContext> ctx_;
    std::unique_ptr<VulkanPipelineManager> pipeline_mgr_;

    G4MtpHeader header_{};
    std::string model_path_;

    // Win32 memory-mapping
    HANDLE file_handle_{INVALID_HANDLE_VALUE};
    HANDLE mapping_handle_{NULL};
    const uint8_t* mapped_data_{nullptr};
    uint64_t file_size_{0};
    uint64_t total_gpu_bytes_{0};

    // Precomputed per-layer offsets
    std::vector<MtpLayerOffsets> layer_offsets_;

    // GPU Command Pool & Buffer
    VkCommandPool cmd_pool_{VK_NULL_HANDLE};
    VkCommandBuffer cmd_{VK_NULL_HANDLE};
    VkFence fence_{VK_NULL_HANDLE};

    // Weight allocations on GPU
    VkMemoryAllocation buf_embed_{};      // Tied LM head (262144 x 1024 INT4)
    VkMemoryAllocation buf_pre_proj_{};   // Pre-projection (1024 x 10752 INT4)
    VkMemoryAllocation buf_post_proj_{};  // Post-projection (5376 x 1024 INT4)
    VkMemoryAllocation buf_norm_{};       // Final RMSNorm weight (1024 BF16)
    std::vector<VkMemoryAllocation> buf_layers_; // 4 layers

    // Activation scratch buffers on GPU
    VkMemoryAllocation buf_in_{};            // [embed, hidden] (10752 floats)
    VkMemoryAllocation buf_hidden_{};        // hidden state (1024 floats)
    VkMemoryAllocation buf_norm_act_{};      // normalized hidden (1024 floats)
    VkMemoryAllocation buf_q_{};             // query (max 16384 floats)
    VkMemoryAllocation buf_attn_out_{};      // attention out (max 16384 floats)
    VkMemoryAllocation buf_proj_out_{};      // projection out (1024 floats)
    VkMemoryAllocation buf_gate_{};          // MLP gate (8192 floats)
    VkMemoryAllocation buf_up_{};            // MLP up (8192 floats)
    VkMemoryAllocation buf_post_proj_act_{}; // Post-projected hidden state (5376 floats)
    VkMemoryAllocation buf_logits_{};        // Logits (262144 floats)
};

} // namespace g4dense
