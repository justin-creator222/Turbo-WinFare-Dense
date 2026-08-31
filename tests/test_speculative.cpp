#include "g4dense/speculator.hpp"
#include "g4dense/draft_runtime.hpp"
#include "g4dense/runner.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/manifest.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <filesystem>
#include <iomanip>

using namespace g4dense;
namespace fs = std::filesystem;

int main() {
    // Shared with the draft runtime: one device, one set of pipelines.
    auto vk_ctx = std::make_shared<VulkanContext>();
    vk_ctx->initialize();
    auto tok = std::make_shared<Tokenizer>();
    if (fs::exists(resolve_resource_path("tokenizer.json"))) {
        tok->load_vocabulary(resolve_resource_path("tokenizer.json"));
    }

    std::cout << "========================================================\n"
              << "  Turbo-WinFare Dense: Speculative Decoding Tests     \n"
              << "========================================================" << std::endl;

    // 1. Test SpeculativeCoordinator Exact Parity Under Greedy
    std::cout << "1. Testing SpeculativeCoordinator Greedy Invariant..." << std::endl;
    {
        SpeculativeCoordinator spec;
        uint32_t vocab_size = 1024;
        SamplingParams greedy_params;
        greedy_params.temperature = 0.0f; // greedy

        // Setup mock target logits
        std::vector<std::vector<float>> target_logits(5, std::vector<float>(vocab_size, 0.0f));
        target_logits[0][42] = 10.0f;  // argmax = 42
        target_logits[1][100] = 10.0f; // argmax = 100
        target_logits[2][200] = 10.0f; // argmax = 200
        target_logits[3][300] = 10.0f; // argmax = 300
        target_logits[4][400] = 10.0f; // bonus token argmax = 400

        std::vector<const float*> target_ptrs;
        for (const auto& l : target_logits) target_ptrs.push_back(l.data());

        // Case A: Full match (draft = 42, 100, 200, 300)
        std::vector<uint32_t> draft_perfect = {42, 100, 200, 300};
        auto eval_perfect = spec.evaluate_verification(draft_perfect, target_ptrs, vocab_size, greedy_params, 12345);
        assert(eval_perfect.num_accepted == 4);
        assert(eval_perfect.bonus_token_sampled);
        assert(eval_perfect.bonus_token == 400);
        assert(eval_perfect.accepted_tokens.size() == 5);
        assert((eval_perfect.accepted_tokens == std::vector<uint32_t>{42, 100, 200, 300, 400}));
        std::cout << "   [PASS] Perfect draft match: accepted 4/4 + 1 bonus token = 5 tokens." << std::endl;

        // Case B: Partial match (draft = 42, 100, 999, 300) -> should accept 42, 100, reject 999 and emit 200
        std::vector<uint32_t> draft_partial = {42, 100, 999, 300};
        auto eval_partial = spec.evaluate_verification(draft_partial, target_ptrs, vocab_size, greedy_params, 12345);
        assert(eval_partial.num_accepted == 2);
        assert(eval_partial.bonus_token_sampled);
        assert(eval_partial.bonus_token == 200);
        assert(eval_partial.accepted_tokens.size() == 3);
        assert((eval_partial.accepted_tokens == std::vector<uint32_t>{42, 100, 200}));
        std::cout << "   [PASS] Partial draft match: accepted 2/4 + correction token = 3 tokens." << std::endl;

        // Case C: Immediate mismatch (draft = 999, 100) -> should accept 0, emit 42
        std::vector<uint32_t> draft_mismatch = {999, 100};
        auto eval_mismatch = spec.evaluate_verification(draft_mismatch, target_ptrs, vocab_size, greedy_params, 12345);
        assert(eval_mismatch.num_accepted == 0);
        assert(eval_mismatch.bonus_token_sampled);
        assert(eval_mismatch.bonus_token == 42);
        assert(eval_mismatch.accepted_tokens.size() == 1);
        assert(eval_mismatch.accepted_tokens[0] == 42);
        std::cout << "   [PASS] Immediate mismatch: accepted 0/2 + correction token = 1 token." << std::endl;
    }

    // 2. Test DraftRuntime Loading and Rejection on Missing Checkpoint
    std::cout << "\n2. Testing DraftRuntime Checkpoint Contracts..." << std::endl;
    {
        DraftRuntime draft;
        bool caught = false;
        try {
            draft.load_model("non_existent_model_checkpoint.g4dense", vk_ctx, tok);
        } catch (const std::exception& e) {
            caught = true;
        }
        assert(caught && "DraftRuntime must fail loudly on missing checkpoint!");
        std::cout << "   [PASS] DraftRuntime correctly throws on missing checkpoint." << std::endl;

        // E2B is a real draft model again: round 7 implemented the per-layer embeddings and
        // KV sharing it needs, and it is validated against tools/numpy_reference.py.
        std::string e2b_path = resolve_bundle_path("models/gemma-4-e2b-dense.g4dense");
        if (fs::exists(e2b_path)) {
            bool ok = draft.load_model(e2b_path, vk_ctx, tok);
            assert(ok && draft.is_loaded());
            std::cout << "   [PASS] DraftRuntime loaded real E2B model: " << e2b_path << std::endl;

            SamplingParams sp;
            sp.temperature = 0.0f;
            std::vector<uint32_t> prompt = {2, 100, 200};

            for (uint32_t k : {2, 4, 6}) {
                auto res = draft.generate_draft_tokens(prompt, k, sp);
                assert(res.draft_tokens.size() == k);
                assert(res.draft_logits.size() == k);
                std::cout << "   [PASS] Draft generation K=" << k << " produced " << res.draft_tokens.size()
                          << " tokens in " << res.draft_time_ms << " ms." << std::endl;
            }
        }
    }

    // 3. Test End-to-End Speculative vs Autoregressive Equivalence on Engine
    std::cout << "\n3. Testing End-to-End Engine Speculative vs Non-Speculative Equivalence..." << std::endl;
    {
        auto ctx = std::make_shared<VulkanContext>();
        ctx->initialize();

        auto tok = std::make_shared<Tokenizer>();
        std::string tok_path = resolve_resource_path("tests/fixtures/tokenizer.json");
        tok->load_vocabulary(tok_path);

        std::string model_path = resolve_bundle_path("tests/fixtures/tiny.g4dense");

        ForwardRunner runner_nonspec(ctx, tok, model_path);
        runner_nonspec.initialize();

        ForwardRunner runner_spec(ctx, tok, model_path);
        runner_spec.initialize();

        std::string prompt = "What is the capital of France?";
        int max_tokens = 3;

        GenerationOptions opts_nonspec;
        opts_nonspec.max_tokens = max_tokens;
        opts_nonspec.sampling.temperature = 0.0f;
        opts_nonspec.speculative_enabled = false;

        GenerationOptions opts_spec;
        opts_spec.max_tokens = max_tokens;
        opts_spec.sampling.temperature = 0.0f;
        opts_spec.speculative_enabled = true;
        opts_spec.draft_k = 4;

        std::vector<uint32_t> nonspec_tokens;
        runner_nonspec.generate(prompt, opts_nonspec, [&](uint32_t token, const std::string&) {
            nonspec_tokens.push_back(token);
            return true;
        });

        std::vector<uint32_t> spec_tokens;
        runner_spec.generate(prompt, opts_spec, [&](uint32_t token, const std::string&) {
            spec_tokens.push_back(token);
            return true;
        });

        std::cout << "   Non-speculative generated " << nonspec_tokens.size() << " tokens." << std::endl;
        std::cout << "   Speculative generated     " << spec_tokens.size() << " tokens." << std::endl;

        assert(nonspec_tokens.size() == spec_tokens.size());
        for (size_t i = 0; i < nonspec_tokens.size(); ++i) {
            assert(nonspec_tokens[i] == spec_tokens[i] && "Speculative token must equal non-speculative token under greedy!");
        }

        std::cout << "   [PASS] Gate R5: Speculative output equals Non-Speculative output token-for-token under greedy!" << std::endl;
    }

    std::cout << "\nAll speculative decoding tests passed." << std::endl;
    return 0;
}
