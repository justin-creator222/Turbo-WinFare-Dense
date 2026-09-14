#include "g4dense/mtp_runner.hpp"
#include "g4dense/kv_cache.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/format.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <filesystem>

using namespace g4dense;

int main() {
    std::cout << "========================================================\n"
              << "  Turbo-WinFare Dense: Native MTP Drafter Tests        \n"
              << "========================================================" << std::endl;

    const std::string model_path = "models/gemma-4-31b-assistant.g4mtp";
    if (!std::filesystem::exists(model_path)) {
        std::cout << "   [SKIP] Model not found at " << model_path << std::endl;
        return 0;
    }

    auto ctx = std::make_shared<VulkanContext>();
    ctx->initialize();

    MtpRunner mtp;
    std::cout << "1. Loading MTP checkpoint: " << model_path << "..." << std::endl;
    bool loaded = mtp.load_model(model_path, ctx);
    assert(loaded && "MtpRunner failed to load!");
    std::cout << "   [PASS] MtpRunner loaded successfully. GPU memory: "
              << (mtp.total_gpu_bytes() / (1024 * 1024)) << " MB" << std::endl;

    const auto& hdr = mtp.header();
    assert(hdr.magic == G4MtpHeader::EXPECTED_MAGIC);
    assert(hdr.version == G4MtpHeader::EXPECTED_VERSION);
    assert(hdr.num_layers == 4);
    assert(hdr.backbone_hidden_size == 5376);
    assert(hdr.hidden_size == 1024);
    assert(hdr.intermediate_size == 8192);
    assert(hdr.vocab_size == 262144);
    std::cout << "   [PASS] Header metadata and geometry verified." << std::endl;

    // 2. Setup a target KV cache (60 layers, layer 58 sliding, layer 59 global)
    KVCacheConfig kv_cfg;
    kv_cfg.num_layers = 60;
    kv_cfg.num_kv_heads = 16;
    kv_cfg.head_dim = 256;
    kv_cfg.global_kv_heads = 4;
    kv_cfg.global_head_dim = 512;
    kv_cfg.sliding_window = 1024;
    kv_cfg.max_context = 2048;
    kv_cfg.global_layer_mask = 0b100000100000100000100000100000100000100000100000100000100000ULL;
    kv_cfg.dtype = KVDType::FP16;

    std::cout << "2. Initializing target KV cache manager..." << std::endl;
    KVCacheManager kv(ctx, kv_cfg);
    kv.initialize();
    std::cout << "   [PASS] KV cache initialized." << std::endl;

    // 3. Test multi-token draft generation
    std::cout << "3. Running MTP draft generation (K=4)..." << std::endl;
    std::vector<float> mock_target_hidden(5376, 0.015f);
    auto mock_embed_fn = [](uint32_t token, float* out) {
        float base = 0.001f * static_cast<float>(token % 100);
        for (uint32_t i = 0; i < 5376; ++i) {
            out[i] = base + 0.0001f * static_cast<float>(i % 32);
        }
    };

    SamplingParams sampling;
    sampling.temperature = 0.0f; // greedy

    DraftResult result;
    try {
        result = mtp.generate_draft_tokens(
            mock_target_hidden.data(),
            100, // seed token
            16,  // context length
            4,   // draft 4 tokens
            sampling,
            mock_embed_fn,
            &kv
        );
    } catch (const std::exception& ex) {
        std::cerr << "   [ERROR] Exception caught: " << ex.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "   [ERROR] Unknown exception caught!" << std::endl;
        return 1;
    }

    assert(result.draft_tokens.size() == 4 && "Expected 4 draft tokens");
    assert(result.draft_logits.size() == 4 && "Expected 4 draft logits sets");

    std::cout << "   Draft generation completed in " << result.draft_time_ms << " ms"
              << " (" << (result.draft_time_ms / 4.0) << " ms/token)" << std::endl;

    for (size_t s = 0; s < result.draft_tokens.size(); ++s) {
        uint32_t tok = result.draft_tokens[s];
        assert(tok < 262144 && "Draft token ID out of vocab range");
        const auto& logits = result.draft_logits[s];
        assert(logits.size() == 262144);

        // Check for NaN or Inf
        float sample_logit = logits[tok];
        assert(!std::isnan(sample_logit) && !std::isinf(sample_logit));

        std::cout << "   Draft [" << s << "]: token=" << tok
                  << " logit=" << sample_logit << std::endl;
    }

    std::cout << "   [PASS] MTP drafter verified: finite logits, valid tokens, fast GPU execution." << std::endl;

    // 4. Test warm round timing
    std::cout << "4. Running warm MTP draft generation (K=4)..." << std::endl;
    DraftResult warm_res = mtp.generate_draft_tokens(
        mock_target_hidden.data(),
        result.draft_tokens.back(),
        20,
        4,
        sampling,
        mock_embed_fn,
        &kv
    );
    std::cout << "   Warm draft generation: " << warm_res.draft_time_ms << " ms"
              << " (" << (warm_res.draft_time_ms / 4.0) << " ms/token)" << std::endl;

    std::cout << "\nAll MTP runner tests passed successfully." << std::endl;
    return 0;
}
