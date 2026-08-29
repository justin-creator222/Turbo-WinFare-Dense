#include "g4dense/cpu_reference.hpp"
#include "g4dense/tokenizer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>

using namespace g4dense;

int main() {
    std::cout << "[test_cpu_reference] Starting CPU Reference Oracle verification..." << std::endl;

    CpuReferenceConfig config;
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
    config.dump_tensors_dir = "tensors_dump";
    config.verbose = true;

    auto tok = std::make_shared<Tokenizer>();
    // Try loading tokenizer vocabulary
    try {
        tok->load_vocabulary("tests/fixtures/tokenizer.json");
    } catch (...) {
        try {
            tok->load_vocabulary("../tests/fixtures/tokenizer.json");
        } catch (...) {}
    }

    CpuReferenceRunner oracle(config, tok);
    oracle.initialize();

    std::cout << "  Running token 0 forward pass on tiny.g4dense..." << std::endl;
    std::vector<float> logits = oracle.forward_single_token(10, 0, true);

    assert(logits.size() == 1024);
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
