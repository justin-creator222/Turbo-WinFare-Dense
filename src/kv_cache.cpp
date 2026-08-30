#include "g4dense/kv_cache.hpp"
#include "g4dense/format.hpp"

#include <cstring>
#include <iostream>

namespace g4dense {

KVCacheManager::KVCacheManager(std::shared_ptr<VulkanContext> ctx, const KVCacheConfig& config)
    : ctx_(ctx), config_(config) {}

KVCacheManager::~KVCacheManager() {
    for (auto& a : k_allocations_) ctx_->free_buffer(a);
    for (auto& a : v_allocations_) ctx_->free_buffer(a);
    for (auto& a : scale_allocations_) ctx_->free_buffer(a);
    k_allocations_.clear();
    v_allocations_.clear();
    scale_allocations_.clear();
}

bool KVCacheManager::is_global_layer(uint32_t layer_idx) const {
    return (config_.global_layer_mask & (1ULL << layer_idx)) != 0;
}

uint32_t KVCacheManager::layer_capacity(uint32_t layer_idx) const {
    return is_global_layer(layer_idx) ? config_.max_context : config_.sliding_window;
}

uint32_t KVCacheManager::physical_slot(uint32_t layer_idx, uint32_t logical_pos) const {
    uint32_t cap = layer_capacity(layer_idx);
    return logical_pos % cap;
}

void KVCacheManager::reset() {
    // All KV allocations are host-visible and persistently mapped, so this is a plain memset
    // -- no command buffer, no queue submission, no synchronization with in-flight work
    // (callers reset between generations, never mid-pass).
    auto zero = [](std::vector<VkMemoryAllocation>& allocs) {
        for (auto& a : allocs) {
            if (a.mapped_ptr && a.size_bytes > 0) {
                std::memset(a.mapped_ptr, 0, static_cast<size_t>(a.size_bytes));
            }
        }
    };
    zero(k_allocations_);
    zero(v_allocations_);
    zero(scale_allocations_);

    current_pos_ = 0;
    speculative_draft_k_ = 0;
}

void KVCacheManager::initialize() {
    uint32_t elem_size = (config_.dtype == KVDType::FP32) ? 4 : ((config_.dtype == KVDType::FP16) ? 2 : 1);
    total_bytes_ = 0;

    k_allocations_.reserve(config_.num_layers);
    v_allocations_.reserve(config_.num_layers);
    scale_allocations_.reserve(config_.num_layers);

    for (uint32_t l = 0; l < config_.num_layers; ++l) {
        uint32_t cap = layer_capacity(l);
        // Size each slot from THIS layer's geometry. Using the sliding values throughout
        // over-allocated the full-attention layers on the 31B (16x256 for a 4x512 slot) and
        // would under-allocate a model whose full-attention slot is the larger of the two.
        const bool global = is_global_layer(l);
        const uint32_t slot_kv_heads = global ? config_.global_kv_heads : config_.num_kv_heads;
        const uint32_t slot_head_dim = global ? config_.global_head_dim : config_.head_dim;
        uint64_t tensor_bytes = static_cast<uint64_t>(cap) * slot_kv_heads * slot_head_dim * elem_size;

        VkMemoryAllocation k_alloc = ctx_->allocate_buffer(tensor_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation v_alloc = ctx_->allocate_buffer(tensor_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        k_allocations_.push_back(k_alloc);
        v_allocations_.push_back(v_alloc);
        total_bytes_ += tensor_bytes * 2;

        if (config_.dtype == KVDType::INT8) {
            uint64_t scale_bytes = static_cast<uint64_t>(cap) * slot_kv_heads * sizeof(uint16_t);
            VkMemoryAllocation s_alloc = ctx_->allocate_buffer(scale_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
            scale_allocations_.push_back(s_alloc);
            total_bytes_ += scale_bytes;
        }
    }
}

void KVCacheManager::begin_speculative_draft(uint32_t current_pos, uint32_t draft_k) {
    current_pos_ = current_pos;
    speculative_draft_k_ = draft_k;
}

void KVCacheManager::commit_speculative_tokens(uint32_t num_accepted) {
    current_pos_ += num_accepted;
    speculative_draft_k_ = 0;
}

void KVCacheManager::rollback_speculative_tokens() {
    speculative_draft_k_ = 0;
}

VkBuffer KVCacheManager::k_buffer(uint32_t layer_idx) const {
    return k_allocations_[layer_idx].buffer;
}

VkBuffer KVCacheManager::v_buffer(uint32_t layer_idx) const {
    return v_allocations_[layer_idx].buffer;
}

VkBuffer KVCacheManager::scale_buffer(uint32_t layer_idx) const {
    if (config_.dtype == KVDType::INT8 && layer_idx < scale_allocations_.size()) {
        return scale_allocations_[layer_idx].buffer;
    }
    return VK_NULL_HANDLE;
}

} // namespace g4dense
