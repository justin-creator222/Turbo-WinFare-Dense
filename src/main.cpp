#include "g4dense/runner.hpp"
#include "g4dense/server.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/telemetry.hpp"
#include "g4dense/manifest.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>
#include <csignal>
#include <windows.h>

namespace fs = std::filesystem;

static std::atomic<bool> g_shutdown{false};

void signal_handler(int) {
    g_shutdown = true;
}

void print_usage() {
    std::cout << "Turbo-WinFare Dense: Gemma 4 31B Streaming Inference Engine\n\n"
              << "Usage: turbo-dense.exe [options]\n\n"
              << "Options:\n"
              << "  --model <path>      Path to .g4dense model file or bundle directory\n"
              << "  --tier <1|2|3|4>    Target memory tier (default: 1)\n"
              << "  --prompt <text>     Prompt to run generation on\n"
              << "  --max-tokens <N>    Maximum tokens to generate (default: 512)\n"
              << "  --temp <float>      Sampling temperature (0.0 = greedy, default: 0.2)\n"
              << "  --top-p <float>     Nucleus sampling top-p (default: 0.95)\n"
              << "  --top-k <int>       Top-k truncation (default: 64)\n"
              << "  --draft-k <int>     Speculative draft tokens per step (default: 4)\n"
              << "  --no-spec           Disable speculative decoding\n"
              << "  --server            Start OpenAI-compatible HTTP server and Web GUI\n"
              << "  --port <port>       Server port (default: 8080)\n"
              << "  --gui               Auto-open Web GUI in default browser\n"
              << "  --help              Display this help message\n"
              << std::endl;
}

int main(int argc, char** argv) {
    std::string model_path = "gemma-4-31b-dense.g4dense";
    std::string prompt = "";
    int tier_id = 1;
    int max_tokens = 512;
    float temp = 0.2f;
    float top_p = 0.95f;
    int top_k = 64;
    int draft_k = 4;
    bool speculative = true;
    bool run_server = false;
    uint16_t port = 8080;
    bool open_gui = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--tier" && i + 1 < argc) {
            tier_id = std::stoi(argv[++i]);
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--draft-k" && i + 1 < argc) {
            draft_k = std::stoi(argv[++i]);
        } else if (arg == "--no-spec") {
            speculative = false;
        } else if (arg == "--server") {
            run_server = true;
        } else if (arg == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--gui") {
            open_gui = true;
            run_server = true;
        }
    }

    std::cout << "========================================================\n"
              << "  Turbo-WinFare Dense: Gemma 4 31B Streaming Engine   \n"
              << "========================================================" << std::endl;

    std::string resolved_path = g4dense::resolve_bundle_path(model_path);
    if (!fs::exists(resolved_path)) {
        // Check relative fallback
        if (fs::exists("tests/fixtures/tiny.g4dense")) {
            resolved_path = "tests/fixtures/tiny.g4dense";
        }
    }

    if (!fs::exists(resolved_path)) {
        std::cerr << "Error: Model file not found at " << resolved_path << std::endl;
        return 1;
    }

    // Initialize Vulkan Context
    std::cout << "Initializing Vulkan 1.3 Compute Device..." << std::endl;
    auto ctx = std::make_shared<g4dense::VulkanContext>();
    ctx->initialize();
    std::cout << "Device: " << ctx->device_name() << " (Wave32 Subgroups)" << std::endl;

    // Initialize Tokenizer
    auto tok = std::make_shared<g4dense::Tokenizer>();
    if (fs::exists("tests/fixtures/tokenizer.json")) {
        tok->load_vocabulary("tests/fixtures/tokenizer.json");
    } else {
        tok->load_vocabulary();
    }

    // Initialize Forward Runner
    std::cout << "Loading model container: " << resolved_path << std::endl;
    auto runner = std::make_shared<g4dense::ForwardRunner>(ctx, tok, resolved_path);
    runner->initialize();
    runner->switch_memory_tier(tier_id);
    std::cout << "Activated Memory Tier " << tier_id << " (Pinned layers active)." << std::endl;

    if (run_server) {
        g4dense::HTTPServer server;
        server.start(port, runner, ctx);
        std::cout << "\nServer listening on http://127.0.0.1:" << port << std::endl;
        std::cout << "Web GUI available at http://127.0.0.1:" << port << "/\n" << std::endl;

        if (open_gui) {
            std::string open_cmd = "start http://127.0.0.1:" + std::to_string(port);
            system(open_cmd.c_str());
        }

        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        while (!g_shutdown) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "Shutting down server..." << std::endl;
        server.stop();
        return 0;
    }

    // CLI Generation Mode
    std::string gen_prompt = prompt.empty() ? "Hello! Introduce yourself." : prompt;
    std::cout << "\nPrompt: " << gen_prompt << "\n\nResponse:\n";

    g4dense::GenerationOptions opts;
    opts.max_tokens = max_tokens;
    opts.sampling.temperature = temp;
    opts.sampling.top_p = top_p;
    opts.sampling.top_k = top_k;
    opts.speculative_enabled = speculative;
    opts.draft_k = draft_k;

    auto t0 = std::chrono::high_resolution_clock::now();
    int tok_count = 0;

    runner->generate(gen_prompt, opts, [&](uint32_t, const std::string& piece) {
        std::cout << piece << std::flush;
        tok_count++;
        return true;
    });

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t1 - t0).count();

    g4dense::TelemetrySnapshot tele = runner->get_latest_telemetry();
    std::cout << "\n\n========================================================"
              << "\nGeneration Summary:"
              << "\n  Tokens Generated: " << tok_count
              << "\n  Elapsed Time:     " << total_sec << " s"
              << "\n  Throughput (TPS): " << (tok_count / (total_sec > 0.0 ? total_sec : 1.0)) << " tokens/s"
              << "\n  RAM Footprint:    " << tele.ram_footprint_mb << " MB / " << tele.ram_total_mb << " MB"
              << "\n  Memory Tier:      Tier " << tele.active_tier_id
              << "\n========================================================"
              << std::endl;

    return 0;
}
