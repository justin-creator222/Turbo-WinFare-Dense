#include <filesystem>
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "g4dense/runner.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/telemetry.hpp"
#include "g4dense/manifest.hpp"

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

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Turbo-WinFare Dense Smoke Engine Tests        " << std::endl;
    std::cout << "================================================" << std::endl;

    // 1. Initialize VulkanContext & Tokenizer
    std::cout << "1. Initializing Vulkan 1.3 Context..." << std::endl;
    auto ctx = std::make_shared<VulkanContext>();
    ctx->initialize();
    std::cout << "   GPU Device: " << ctx->device_name() << std::endl;
    std::cout << "   Subgroup Size: " << ctx->subgroup_size() << std::endl;

    auto tok = std::make_shared<Tokenizer>();
    std::string tok_path = "tokenizer.json";
    if (!fs::exists(tok_path)) {
        tok_path = resolve_resource_path("tests/fixtures/tokenizer.json");
    }
    if (fs::exists(tok_path)) {
        tok->load_vocabulary(tok_path);
    }

    // 2. Initialize ForwardRunner on tiny fixture
    std::cout << "2. Initializing ForwardRunner on tiny fixture..." << std::endl;
    std::string model_path = resolve_bundle_path("tests/fixtures/tiny.g4dense");
    if (!fs::exists(model_path)) {
        model_path = resolve_bundle_path("tiny.g4dense");
    }

    auto runner = std::make_shared<ForwardRunner>(ctx, tok, model_path);
    runner->initialize();
    std::cout << "   Model loaded successfully: " << model_path << std::endl;

    // 3. Multi-Tier Dynamic Live Switching Smoke Check
    std::cout << "3. Testing Memory Tier Switching API..." << std::endl;
    runner->switch_memory_tier(1);
    assert(runner->active_tier_id() == 1);
    std::cout << "   Tier 1 switch confirmed." << std::endl;

    runner->switch_memory_tier(2);
    assert(runner->active_tier_id() == 2);
    std::cout << "   Tier 2 switch confirmed." << std::endl;

    runner->switch_memory_tier(3);
    assert(runner->active_tier_id() == 3);
    std::cout << "   Tier 3 switch confirmed." << std::endl;

    // 4. Generation Loop Smoke Check
    std::cout << "4. Testing Generation Loop..." << std::endl;
    GenerationOptions opts;
    opts.max_tokens = 4;
    opts.speculative_enabled = false;
    opts.sampling.temperature = 0.0f; // Greedy

    std::string generated_text;
    int token_count = 0;

    runner->generate("Hi", opts, [&](uint32_t token, const std::string& piece) {
        generated_text += piece;
        token_count++;
        std::cout << "   Token " << token_count << " (id=" << token << "): \"" << piece << "\"" << std::endl;
        return true;
    });

    assert(token_count > 0);
    std::cout << "   Generated " << token_count << " tokens." << std::endl;

    TelemetrySnapshot tele = runner->get_latest_telemetry();
    std::cout << "   Telemetry: RAM footprint=" << tele.ram_footprint_mb << " MB, total RAM="
              << tele.ram_total_mb << " MB, TPS=" << tele.tps << std::endl;
    assert(tele.ram_footprint_mb > 0.0);

    // 5. Real Model Container & Oracle Parity Check
    std::string real_31b_path = "models/gemma-4-31b-dense.g4dense";
    if (fs::exists(real_31b_path)) {
        std::cout << "\n5. Testing GPU Forward Pass on Real 31B Model (" << real_31b_path << ")..." << std::endl;
        ForwardRunner real_runner(ctx, tok, real_31b_path);
        real_runner.initialize();

        uint32_t vocab_size = real_runner.header().vocab_size;
        std::vector<float> gpu_logits(vocab_size);

        auto t0 = std::chrono::high_resolution_clock::now();
        real_runner.forward_single_token(2, 0, gpu_logits.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "   GPU Forward Pass (Token 2, Pos 0) completed in " << ms << " ms." << std::endl;

        std::string oracle_path = "tests/fixtures/oracle_tensors/token0_logits.bin";
        if (fs::exists(oracle_path)) {
            std::vector<float> cpu_logits = load_binary_tensor(oracle_path);
            assert(cpu_logits.size() == vocab_size);

            float max_diff = 0.0f;
            float sum_diff = 0.0f;
            uint32_t cpu_argmax = 0, gpu_argmax = 0;
            float cpu_max = -1e9f, gpu_max = -1e9f;

            for (uint32_t v = 0; v < vocab_size; ++v) {
                float diff = std::abs(gpu_logits[v] - cpu_logits[v]);
                if (diff > max_diff) max_diff = diff;
                sum_diff += diff;

                if (cpu_logits[v] > cpu_max) {
                    cpu_max = cpu_logits[v];
                    cpu_argmax = v;
                }
                if (gpu_logits[v] > gpu_max) {
                    gpu_max = gpu_logits[v];
                    gpu_argmax = v;
                }
            }

            float mean_diff = sum_diff / static_cast<float>(vocab_size);
            std::cout << "   Mean abs diff vs CPU Oracle: " << std::scientific << std::setprecision(6) << mean_diff << std::endl;
            std::cout << "   Max abs diff vs CPU Oracle:  " << max_diff << std::endl;
            std::cout << "   CPU Argmax: token " << cpu_argmax << " (logit=" << cpu_max << ")" << std::endl;
            std::cout << "   GPU Argmax: token " << gpu_argmax << " (logit=" << gpu_max << ")" << std::endl;

            assert(cpu_argmax == gpu_argmax && "GPU argmax token must match CPU reference token!");
            assert(max_diff < 0.05f && "Max logit difference must be within FP32/BF16 tolerance");
            std::cout << "   [PASS] Real 31B GPU Forward Pass matches CPU Oracle with exact token equality!" << std::endl;
        }
    }

    std::cout << "\nAll smoke engine tests passed." << std::endl;
    return 0;
}
