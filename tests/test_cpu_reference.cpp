#include "g4dense/cpu_reference.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/manifest.hpp"   // resolve_resource_path
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <algorithm>
#include <string>

using namespace g4dense;

// Usage: run_cpu_reference_test [model_path] [dump_dir] [token_id]
//
// Defaults exercise the tiny fixture. Passing the real container makes this the oracle-dump
// tool: it is how tests/fixtures/oracle_tensors/ is regenerated, and how the per-stage tensors
// used to localize a forward-pass defect are produced.
int main(int argc, char** argv) {
    std::cout << "[test_cpu_reference] Starting CPU Reference Oracle verification..." << std::endl;

    CpuReferenceConfig config;
    if (argc > 1) {
        config.container_path = argv[1];
        if (!std::filesystem::exists(config.container_path)) {
            std::cerr << "FAIL: model container not found: " << config.container_path << "\n";
            return 1;
        }
    } else {
        std::vector<std::string> candidates = {
            "tests/fixtures/tiny.g4dense",
            "../tests/fixtures/tiny.g4dense",
            "../../tests/fixtures/tiny.g4dense"
        };
        for (const auto& c : candidates) {
            if (std::filesystem::exists(c)) {
                config.container_path = c;
                break;
            }
        }
    }
    config.dump_tensors_dir = (argc > 2) ? argv[2] : "tensors_dump";
    const uint32_t probe_token = (argc > 3) ? static_cast<uint32_t>(std::atoi(argv[3])) : 10u;
    config.verbose = true;

    // resolve_resource_path searches the known roots, so the nested catch(...) that silently
    // left the tokenizer unloaded is unnecessary -- and it was harmful: an unloaded tokenizer
    // throws on encode(), so swallowing the failure here just moved the error somewhere less
    // informative.
    auto tok = std::make_shared<Tokenizer>();
    tok->load_vocabulary(resolve_resource_path("tokenizer.json"));
    if (!tok->is_loaded()) {
        std::cerr << "FAIL: tokenizer vocabulary did not load\n";
        return 1;
    }

    CpuReferenceRunner oracle(config, tok);
    oracle.initialize();

    std::cout << "  Running forward pass on " << config.container_path
              << " (token " << probe_token << ", position 0)..." << std::endl;
    // Optional 4th argument: a comma-separated token sequence, fed at positions 0..N-1.
    //
    // The CPU reference has only ever been exercised at position 0, so its multi-position
    // behaviour -- whether attention actually reads the KV written for earlier positions -- has
    // never been checked. The GPU path demonstrably fails that (test_context_dependence check
    // D). Running the same sequence here separates "the shared forward pass mishandles
    // history" from "only the GPU path does", which position-0 parity cannot distinguish.
    std::vector<float> logits;
    if (argc > 4) {
        std::vector<uint32_t> seq;
        std::string spec = argv[4];
        size_t pos = 0;
        while (pos <= spec.size()) {
            size_t comma = spec.find(',', pos);
            if (comma == std::string::npos) comma = spec.size();
            if (comma > pos) seq.push_back(static_cast<uint32_t>(std::stoul(spec.substr(pos, comma - pos))));
            pos = comma + 1;
        }
        std::cout << "  Feeding " << seq.size() << " tokens at positions 0.."
                  << (seq.size() - 1) << " ..." << std::endl;
        for (size_t i = 0; i < seq.size(); ++i) {
            // Dump at EVERY position. The dumps are keyed by position, so this writes
            // token0_*, token1_*, ... which is what tools/numpy_reference.py --tokens
            // diffs against to localize a defect in attention over history.
            logits = oracle.forward_single_token(seq[i], static_cast<uint32_t>(i), true);
            std::cout << "    pos " << i << " token " << seq[i] << " -> argmax "
                      << static_cast<uint32_t>(
                             std::max_element(logits.begin(), logits.end()) - logits.begin())
                      << std::endl;
        }
    } else {
        logits = oracle.forward_single_token(probe_token, 0, true);
    }

    // Vocabulary size is a property of the container, not a constant: the tiny fixture has
    // 1,024 entries and the real 31B container has 262,144.
    assert(!logits.empty());
    std::cout << "  Logits size: " << logits.size() << std::endl;

    // Check finite numbers
    float max_logit = -1e9f;
    float min_logit = 1e9f;
    uint32_t argmax = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        float l = logits[i];
        assert(!std::isnan(l));
        assert(!std::isinf(l));
        if (l > max_logit) {
            max_logit = l;
            argmax = static_cast<uint32_t>(i);
        }
        if (l < min_logit) {
            min_logit = l;
        }
    }

    std::cout << "  Forward pass output: argmax=" << argmax
              << ", max_logit=" << max_logit
              << ", min_logit=" << min_logit << std::endl;

    assert(max_logit <= 30.0f); // Softcapped at 30.0
    assert(min_logit >= -30.0f);

    std::cout << "[test_cpu_reference] ALL CPU REFERENCE TESTS PASSED!" << std::endl;
    return 0;
}
