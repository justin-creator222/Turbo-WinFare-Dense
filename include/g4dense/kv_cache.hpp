#pragma once

#include "g4dense/vk_context.hpp"
#include <vector>
#include <memory>
#include <cstdint>

namespace g4dense {

enum class KVDType {
    FP16 = 0,
    INT8 = 1
};

struct KVCacheConfig {
    uint32_t num_layers{60};
    uint32_t num_kv_heads{16};
    uint32_t head_dim{256};
    uint32_t sliding_window{1024};
    uint32_t max_context{8192};
    uint64_t global_layer_mask{0};
    KVDType  dtype{KVDType::INT8};
};

class KVCacheManager {
public:
    explicit KVCacheManager(std::shared_ptr<VulkanContext> ctx, const KVCacheConfig& config);
    ~KVCacheManager();

    void initialize();

    // Ring-buffer physical slot computation
    uint32_t physical_slot(uint32_t layer_idx, uint32_t logical_pos) const;
    uint32_t layer_capacity(uint32_t layer_idx) const;
    bool is_global_layer(uint32_t layer_idx) const;

    // Speculative draft slot management & rollback
    void begin_speculative_draft(uint32_t current_pos, uint32_t draft_k);
    void commit_speculative_tokens(uint32_t num_accepted);
    void rollback_speculative_tokens();

    VkBuffer k_buffer(uint32_t layer_idx) const;
    VkBuffer v_buffer(uint32_t layer_idx) const;
    VkBuffer scale_buffer(uint32_t layer_idx) const; // For INT8 KV mode

    uint64_t total_memory_bytes() const { return total_bytes_; }
    KVDType dtype() const { return config_.dtype; }

private:
    std::shared_ptr<VulkanContext> ctx_;
    KVCacheConfig config_;

    uint32_t current_pos_{0};
    uint32_t speculative_draft_k_{0};

    uint64_t total_bytes_{0};
    std::vector<VkMemoryAllocation> k_allocations_;
    std::vector<VkMemoryAllocation> v_allocations_;
    std::vector<VkMemoryAllocation> scale_allocations_;
};

} // namespace g4dense
