#include "g4dense/runner.hpp"
#include "g4dense/cpu_reference.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/vk_context.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <iomanip>
#include <string>
#include <chrono>

namespace fs = std::filesystem;
using namespace g4dense;

namespace {

std::vector<float> load_binary_tensor(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open binary oracle tensor: " + path);
    }
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

} // namespace

int main(int argc, char** argv) {
    std::cout << "========================================================\n"
              << "  Turbo-WinFare Dense: GPU Forward Pass Oracle Diff     \n"
              << "========================================================\n" << std::flush;

    try {
        std::string model_path = "models/gemma-4-31b-dense.g4dense";
        if (argc > 1) {
            model_path = argv[1];
        }
        // There is deliberately no fallback to the E2B container. It was built from a
        // checkpoint using per-layer embeddings and KV sharing, neither of which this engine
        // implements, so it loads and computes noise. See models/quarantine/README.md.

        // A missing model is a FAILURE, not a skip.
        //
        // This test backs the numerical-parity gate. Returning 0 when the container is absent
        // meant `ctest` could go green with nothing having touched the real model at all --
        // which is exactly what happened for two validation rounds, compounded by the test
        // not being registered in CMakeLists in the first place.
        if (!fs::exists(model_path)) {
            std::cerr << "FAIL: no model container at " << model_path << "\n"
                      << "      This test exists to diff the REAL model against the CPU "
                         "oracle. Not skipping.\n";
            return 1;
        }

        std::cout << "Loading Model: " << model_path << "\n" << std::flush;
        auto vk_ctx = std::make_shared<VulkanContext>();
        vk_ctx->initialize();

        auto tokenizer = std::make_shared<Tokenizer>();
        if (fs::exists("tokenizer.json")) {
            tokenizer->load_vocabulary("tokenizer.json");
        }

        ForwardRunner gpu_runner(vk_ctx, tokenizer, model_path);
        gpu_runner.initialize();

        uint32_t vocab_size = gpu_runner.header().vocab_size;
        std::vector<float> gpu_logits(vocab_size);

        // Optional 2nd/3rd arguments: a dump directory and a comma-separated token sequence,
        // prefilled at positions 0..N-1 and diffed against token{p}_logits.bin at EVERY
        // position.
        //
        // Position 0 alone cannot see a defect in attention over history, and that is the one
        // property still unverified on the GPU: the whole stack matches the oracle at
        // position 0 while real generation produces incoherent text. Generate the dumps with
        //   run_cpu_reference_test.exe <model> <dump-dir> 2 "2,3689,563,506"
        std::string dump_dir = (argc > 2) ? argv[2] : "tests/fixtures/oracle_tensors";
        std::vector<uint32_t> seq;
        if (argc > 3) {
            std::string spec = argv[3];
            size_t cur = 0;
            while (cur <= spec.size()) {
                size_t comma = spec.find(',', cur);
                if (comma == std::string::npos) comma = spec.size();
                if (comma > cur) {
                    seq.push_back(static_cast<uint32_t>(std::stoul(spec.substr(cur, comma - cur))));
                }
                cur = comma + 1;
            }
        }
        if (seq.empty()) seq.push_back(2);   // bare <bos> at position 0

        const bool is_real_model = (gpu_runner.header().num_layers == 60);
        int failures = 0;

        std::cout << "Executing GPU forward pass over " << seq.size()
                  << " position(s) 0.." << (seq.size() - 1) << " ...\n" << std::flush;

        for (size_t pos = 0; pos < seq.size(); ++pos) {
            auto t0 = std::chrono::high_resolution_clock::now();
            gpu_runner.forward_single_token(seq[pos], static_cast<uint32_t>(pos), gpu_logits.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::string oracle_path = dump_dir + "/token" + std::to_string(pos) + "_logits.bin";
            std::cout << "\npos " << pos << " token " << seq[pos]
                      << "  (" << std::fixed << std::setprecision(1) << ms << " ms)\n" << std::flush;

            if (!is_real_model) continue;

            // The oracle diff is the entire point of this test, so a missing oracle fails
            // rather than quietly reducing it to "the forward pass did not crash".
            if (!fs::exists(oracle_path)) {
                std::cerr << "  FAIL: missing " << oracle_path << " -- nothing to diff against.\n"
                          << "        Regenerate with run_cpu_reference_test.exe <model> "
                          << dump_dir << " 2 \"<comma-separated tokens>\"\n";
                ++failures;
                continue;
            }

            std::vector<float> cpu_logits = load_binary_tensor(oracle_path);
            if (cpu_logits.size() != vocab_size) {
                std::cerr << "  FAIL: oracle has " << cpu_logits.size() << " logits, model vocab is "
                          << vocab_size << "\n";
                ++failures;
                continue;
            }

            float max_diff = 0.0f, sum_diff = 0.0f;
            uint32_t cpu_argmax = 0, gpu_argmax = 0;
            float cpu_max = -1e9f, gpu_max = -1e9f;
            for (uint32_t v = 0; v < vocab_size; ++v) {
                float diff = std::abs(gpu_logits[v] - cpu_logits[v]);
                if (diff > max_diff) max_diff = diff;
                sum_diff += diff;
                if (cpu_logits[v] > cpu_max) { cpu_max = cpu_logits[v]; cpu_argmax = v; }
                if (gpu_logits[v] > gpu_max) { gpu_max = gpu_logits[v]; gpu_argmax = v; }
            }

            std::cout << "  mean|diff| " << std::scientific << std::setprecision(6)
                      << (sum_diff / static_cast<float>(vocab_size))
                      << "   max|diff| " << max_diff << "\n"
                      << "  CPU argmax " << cpu_argmax << " (" << cpu_max << ")"
                      << "   GPU argmax " << gpu_argmax << " (" << gpu_max << ")\n" << std::flush;

            if (cpu_argmax != gpu_argmax) {
                std::cerr << "  [FAIL] argmax differs at position " << pos << "\n";
                ++failures;
            } else if (max_diff >= 0.05f) {
                std::cerr << "  [FAIL] max|diff| " << max_diff << " exceeds FP32/BF16 tolerance\n";
                ++failures;
            } else {
                std::cout << "  [PASS] matches the CPU oracle, argmax exact\n" << std::flush;
            }
        }

        if (failures != 0) {
            std::cerr << "\nFAIL: " << failures << " position(s) diverged from the CPU oracle.\n"
                      << "      The FIRST diverging position localizes the defect: position 0 "
                         "exercises the layer stack,\n"
                      << "      every later position additionally exercises attention over KV "
                         "history.\n";
            return 1;
        }

        std::cout << "\n[test_gpu_forward] ALL GPU FORWARD PASS CHECKS PASSED!\n" << std::flush;
    } catch (const std::exception& ex) {
        std::cerr << "EXCEPTION in test_gpu_forward: " << ex.what() << "\n" << std::flush;
        return 1;
    }
    return 0;
}
