#include <g4dense/format.hpp>
#include <g4dense/json.hpp>
#include <g4dense/manifest.hpp>
#include <g4dense/tokenizer.hpp>
#include <g4dense/sampling.hpp>
#include <g4dense/detokenizer.hpp>
#include <g4dense/http.hpp>
#include <g4dense/telemetry.hpp>
#include <g4dense/openai_api.hpp>
#include <g4dense/vk_context.hpp>
#include <g4dense/vk_pipeline.hpp>
#include <g4dense/streamer.hpp>
#include <g4dense/layer_cache.hpp>
#include <g4dense/kv_cache.hpp>
#include <g4dense/draft_runtime.hpp>
#include <g4dense/speculator.hpp>
#include <g4dense/runner.hpp>
#include <g4dense/cpu_reference.hpp>
#include <g4dense/server.hpp>
#include <g4dense/hf_client.hpp>
#include <g4dense/c_api.h>

#include <iostream>
#include <cassert>

using namespace g4dense;

int main() {
    std::cout << "[test_contracts] Testing contract headers..." << std::endl;
    assert(sizeof(G4DenseHeader) == 4096);
    
    G4DenseHeader h{};
    h.magic = G4DenseHeader::EXPECTED_MAGIC;
    h.version = G4DenseHeader::EXPECTED_VERSION;
    h.num_layers = 1;
    h.d_model = 256;
    h.d_ff = 512;
    h.num_q_heads = 4;
    h.num_kv_heads = 2;
    h.head_dim = 64;
    h.vocab_size = 1024;
    h.sliding_window = 512;
    h.quant_group_size = 64;
    h.scale_dtype = 1;
    h.tied_embeddings = 1;
    h.embed_offset = 4096;
    h.embed_size = 4096;
    h.lm_head_offset = 4096;
    h.lm_head_size = 4096;
    h.layer_offsets[0] = 8192;
    h.layer_sizes[0] = 4096;
    validate_header(h, 12288);

    // Verify KVCacheManager runtime functionality
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

    assert(kv.layer_capacity(0) == 4096);
    assert(kv.layer_capacity(1) == 1024);
    assert(kv.layer_capacity(2) == 4096);
    assert(kv.layer_capacity(3) == 1024);
    assert(kv.is_global_layer(0) == true);
    assert(kv.is_global_layer(1) == false);

    assert(kv.physical_slot(0, 10) == 10);
    assert(kv.physical_slot(1, 1025) == 1);

    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        assert(kv.k_buffer(l) != VK_NULL_HANDLE);
        assert(kv.v_buffer(l) != VK_NULL_HANDLE);
    }

    kv.begin_speculative_draft(10, 4);
    kv.rollback_speculative_tokens();
    kv.begin_speculative_draft(10, 4);
    kv.commit_speculative_tokens(2);

    std::cout << "All G4Dense contract headers and KV cache manager validated successfully!" << std::endl;
    return 0;
}
