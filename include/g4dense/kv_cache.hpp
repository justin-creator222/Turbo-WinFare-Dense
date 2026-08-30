#pragma once

#include "g4dense/vk_context.hpp"
#include <vector>
#include <memory>
#include <cstdint>

namespace g4dense {

enum class KVDType {
    FP16 = 0,
    INT8 = 1,
    FP32 = 2
};

struct KVCacheConfig {
    uint32_t num_layers{60};
    uint32_t num_kv_heads{16};
    uint32_t head_dim{256};
    // Full-attention layers carry their own geometry (512 / 4 on the 31B vs 256 / 16
    // sliding), so a slot sized from the sliding values alone is the wrong size for them.
    // It happens to be larger on this model, which is why the mismatch was harmless; it
    // would silently overflow on a model where the full-attention slot is bigger.
    uint32_t global_kv_heads{4};
    uint32_t global_head_dim{512};
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

    // Zeroes every layer's K/V (and INT8 scales) and rewinds the write position.
    //
    // There was no way to do this at all, and ForwardRunner::generate() never tried: a second
    // generate() on the same runner restarted positions at 0 while the buffers still held the
    // previous conversation's keys and values, so attention over `position + 1` slots read a
    // mixture of both. Call this at the start of every independent generation.
    void reset();

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
