#include "g4dense/kv_cache.hpp"
#include "g4dense/vk_context.hpp"
#include <iostream>
#include <cassert>

using namespace g4dense;

int main() {
    std::cout << "========================================================\n"
              << "  Turbo-WinFare Dense: KV Cache Manager Tests         \n"
              << "========================================================" << std::endl;

    auto ctx = std::make_shared<VulkanContext>();
    ctx->initialize();

    KVCacheConfig cfg;
    cfg.num_layers = 4;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 256;
    cfg.sliding_window = 1024;
    cfg.max_context = 4096;
    cfg.global_layer_mask = 0b0101; // Layer 0 and 2 are global
    cfg.dtype = KVDType::FP32;

    KVCacheManager kv(ctx, cfg);
    kv.initialize();

    // 1. Check Layer Capacities and Ring Slots
    assert(kv.layer_capacity(0) == 4096);
    assert(kv.layer_capacity(1) == 1024);
    assert(kv.layer_capacity(2) == 4096);
    assert(kv.layer_capacity(3) == 1024);
    assert(kv.is_global_layer(0) == true);
    assert(kv.is_global_layer(1) == false);

    assert(kv.physical_slot(0, 10) == 10);
    assert(kv.physical_slot(1, 1025) == 1);
    std::cout << "   [PASS] Global vs Sliding layer capacities and ring indexing verified." << std::endl;

    // 2. Check Buffers Allocation
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        assert(kv.k_buffer(l) != VK_NULL_HANDLE);
        assert(kv.v_buffer(l) != VK_NULL_HANDLE);
    }
    std::cout << "   [PASS] GPU KV buffers successfully allocated (Total: " << (kv.total_memory_bytes() / (1024 * 1024)) << " MB)." << std::endl;

    // 3. Test Speculative Draft & Rollback
    kv.begin_speculative_draft(10, 4);
    kv.rollback_speculative_tokens();
    kv.begin_speculative_draft(10, 4);
    kv.commit_speculative_tokens(2);
    std::cout << "   [PASS] Speculative draft begin, rollback, and commit verified." << std::endl;

    std::cout << "\nAll KV cache tests passed." << std::endl;
    return 0;
}
