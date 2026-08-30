// P0.1 — Prove the real Gemma 4 31B forward pass produces coherent text.
//
// GPU==CPU parity proves the two paths agree; it cannot prove either is right, because both
// share the same conventions (global-layer head_dim, the absent v_proj, RoPE pair counts).
// A forward pass that is wrong in a shared way produces fluent-but-drifting text -- the
// hardest failure mode to spot, and the one this test exists to catch.
//
// Read the output. Automation cannot judge coherence; a human can.
//
//   run_real_generation_test [max_tokens] [model_path]

#include "g4dense/runner.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/manifest.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace g4dense;

int main(int argc, char** argv) {
    int max_tokens = (argc > 1) ? std::atoi(argv[1]) : 24;
    std::string model_path = (argc > 2) ? argv[2] : "models/gemma-4-31b-dense.g4dense";

    std::cout << "========================================================\n"
              << "  Real 31B Generation -- Coherence Check (P0.1)         \n"
              << "========================================================\n" << std::flush;

    // A missing model is a hard failure, not a silent skip. The round-2 tests returned 0 here
    // and so never exercised the real model in CI at all.
    if (!fs::exists(model_path)) {
        std::cerr << "FAIL: model container not found: " << model_path << "\n"
                  << "      This test exists to exercise the REAL model. Not skipping.\n";
        return 1;
    }

    try {
        auto ctx = std::make_shared<VulkanContext>();
        ctx->initialize();
        // subgroup_size() is the DEVICE DEFAULT (64 on RDNA 3). Compute pipelines are created
        // with requiredSubgroupSize = 32 in vk_pipeline.cpp, which is what the kernels
        // actually execute at -- printing this without the qualifier reads like a Wave32
        // negotiation failure when there is none.
        std::cout << "Device: " << ctx->device_name()
                  << " (device-default subgroup " << ctx->subgroup_size()
                  << "; compute pipelines pinned to 32)\n" << std::flush;

        auto tok = std::make_shared<Tokenizer>();
        tok->load_vocabulary(resolve_resource_path("tokenizer.json"));
        if (!tok->is_loaded()) {
            std::cerr << "FAIL: tokenizer did not load\n";
            return 1;
        }

        std::cout << "Loading " << model_path << " ...\n" << std::flush;
        ForwardRunner runner(ctx, tok, model_path);
        runner.initialize();
        std::cout << "Loaded. Layers=" << runner.header().num_layers
                  << " vocab=" << runner.header().vocab_size
                  << " max_tokens=" << max_tokens << "\n" << std::flush;

        const std::vector<std::string> prompts = {
            "What is the capital of France?",
            "Write one sentence explaining what a ring buffer is.",
            "List three primary colors.",
        };

        bool all_nonempty = true;

        for (size_t i = 0; i < prompts.size(); ++i) {
            std::cout << "\n--------------------------------------------------------\n"
                      << "PROMPT " << (i + 1) << ": " << prompts[i] << "\n"
                      << "RESPONSE: " << std::flush;

            GenerationOptions opts;
            opts.max_tokens = max_tokens;
            opts.sampling.temperature = 0.0f;   // greedy: deterministic and reproducible
            opts.speculative_enabled = false;   // isolate the target model's own output
            opts.active_tier_id = 1;

            std::string text;
            int count = 0;
            auto t0 = std::chrono::high_resolution_clock::now();

            runner.generate(prompts[i], opts, [&](uint32_t, const std::string& piece) {
                text += piece;
                ++count;
                std::cout << piece << std::flush;
                return true;
            });

            auto t1 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();

            std::cout << "\n[" << count << " tokens in " << std::fixed << std::setprecision(1)
                      << sec << " s = " << std::setprecision(4)
                      << (sec > 0 ? count / sec : 0.0) << " TPS]\n" << std::flush;

            if (text.empty()) {
                all_nonempty = false;
                std::cerr << "FAIL: prompt " << (i + 1) << " produced no text\n";
            }
        }

        std::cout << "\n========================================================\n";
        if (!all_nonempty) {
            std::cerr << "FAIL: at least one prompt produced no output.\n";
            return 1;
        }
        std::cout << "Generation completed. COHERENCE MUST BE JUDGED BY READING THE TEXT ABOVE.\n"
                  << "This test asserts only that generation ran and produced tokens.\n";
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "\nEXCEPTION: " << ex.what() << "\n";
        return 1;
    }
}
