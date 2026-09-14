#include "g4dense/runner.hpp"
#include "g4dense/server.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/telemetry.hpp"
#include "g4dense/manifest.hpp"
#include "g4dense/cpu_reference.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>
#include <csignal>
#include <windows.h>
#include <shellapi.h>
#include <fstream>

namespace fs = std::filesystem;

static std::atomic<bool> g_shutdown{false};

void signal_handler(int) {
    g_shutdown = true;
}

void print_usage() {
    std::cout << "Turbo-WinFare Dense: Gemma 4 31B Streaming Inference Engine\n\n"
              << "Usage: turbo-dense.exe [options]\n\n"
              << "Options:\n"
              << "  --model <path>       Path to .g4dense model file or bundle directory\n"
              << "  --tier <1|2|3|4>     Target memory tier (default: 1)\n"
              << "  --prompt <text>      Prompt to run generation on\n"
              << "  --cpu                Run pure scalar FP32 CPU reference path\n"
              << "  --dump-tensors <dir> Dump per-stage FP32 reference tensors (--cpu only)\n"
              << "  --max-tokens <N>     Maximum tokens to generate (default: 512)\n"
              << "  --temp <float>       Sampling temperature (0.0 = greedy, default: 0.2)\n"
              << "  --top-p <float>      Nucleus sampling top-p (default: 0.95)\n"
              << "  --top-k <int>        Top-k truncation (default: 64)\n"
              << "  --spec               Enable speculative decoding. Auto-loads MTP drafter\n"
              << "                       (models/gemma-4-31b-assistant.g4mtp) if present, or E2B.\n"
              << "  --draft-model <path> Path to draft model (.g4mtp or .g4dense)\n"
              << "  --draft-k <int>      Speculative verify-batch width, 2-8 (default: 5 for MTP, 6 for E2B)\n"
              << "  --no-spec            Deprecated no-op: speculation is off unless --spec\n"
              << "  --server             Start OpenAI-compatible HTTP server and Web GUI\n"
              << "  --port <port>        Server port (default: 8080)\n"
              << "  --gui                Auto-open Web GUI in default browser\n"
              << "  --help               Display this help message\n"
              << std::endl;
}

// The draft model for speculative decoding, and what to hold back from the target's layer
// import so it has room.
// MTP assistant drafter is ~252 MB weights, requiring only ~384 MB reserve,
// freeing 5-6 resident layers for the 31B target model.
// E2B drafter imports 0.977 GiB of layer blocks plus its own activations, KV
// cache and LM head, requiring 1536 MB reserve.
static const char* kMtpDraftModelPath = "models/gemma-4-31b-assistant.g4mtp";
static constexpr unsigned long long kMtpDraftImportReserveBytes = 384ull * 1024 * 1024;

static const char* kE2bDraftModelPath = "models/gemma-4-e2b-dense.g4dense";
static constexpr unsigned long long kE2bDraftImportReserveBytes = 1536ull * 1024 * 1024;

int main(int argc, char** argv) {
    std::string model_path = "gemma-4-31b-dense.g4dense";
    std::string prompt = "";
    int tier_id = 1;
    int max_tokens = 512;
    // 0 = the engine default (4096). Attention no longer caps this -- a larger
    // context just costs KV cache, ~336 MB more at 8192.
    uint32_t max_context = 0;
    float temp = 0.2f;
    float top_p = 0.95f;
    int top_k = 64;
    // 6, and capped by kGemmMaxBatch (the verify batch is K wide). Minimum 2: K is the batch
    // width, one pending token plus K-1 drafts, so K=1 asks for zero drafts.
    //
    // Round 9 set this to 8 and recorded "K=8 is 21% faster, drafting more per round is nearly
    // free". That was measured against a ~1,330 ms target pass -- a round-8 number -- and it
    // never measured --no-spec on the same prompt at all. A 132-run sweep over 10 prompts x
    // {no-spec, K=2,4,6,8} retired it. Best K is prompt-dependent and NOT monotonic, because K
    // decides where round boundaries fall: an unpredictable token costs one wasted draft or
    // seven depending on where it lands. The JSON prompt accepts 76% at K=6 and 46% at K=8.
    //
    // 6 is the best value across the prompts where speculation is worth enabling at all
    // (33.08 / 29.62 / 30.88 s on the three net winners, against K=8's 23.01 / 31.51 / 40.56).
    // K=8 wins only on near-deterministic sequences and collapses everywhere else.
    int draft_k = 5;
    bool draft_k_specified = false;
    std::string draft_model_path = "";

    // Speculation is OFF by default, and --spec opts in.
    //
    // Loading the draft model reserves memory (384 MiB for MTP, 1536 MiB for E2B).
    // The MTP assistant drafter is ~252 MB, allowing the target model to retain ~44 resident layers.
    bool speculative = false;
    bool run_server = false;
    bool cpu_mode = false;
    std::string dump_tensors_dir = "";
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
        } else if (arg == "--cpu") {
            cpu_mode = true;
        } else if (arg == "--dump-tensors" && i + 1 < argc) {
            dump_tensors_dir = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--max-context" && i + 1 < argc) {
            max_context = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--draft-model" && i + 1 < argc) {
            draft_model_path = argv[++i];
        } else if (arg == "--draft-k" && i + 1 < argc) {
            draft_k_specified = true;
            draft_k = std::stoi(argv[++i]);
            if (draft_k < 2) draft_k = 2;
            if (draft_k > static_cast<int>(g4dense::kGemmMaxBatch)) {
                draft_k = static_cast<int>(g4dense::kGemmMaxBatch);
            }
        } else if (arg == "--spec") {
            speculative = true;
        } else if (arg == "--no-spec") {
            // Retained as a no-op: speculation is off unless --spec, and scripts and benchmark
            // harnesses written against the old default still pass this.
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
    if (!g4dense::bundle_loads(resolved_path)) {
        std::string fallback = g4dense::resolve_bundle_path("tests/fixtures/tiny.g4dense");
        if (g4dense::bundle_loads(fallback)) {
            resolved_path = fallback;
        }
    }

    if (!g4dense::bundle_loads(resolved_path)) {
        std::cerr << "Error: Model file or valid container not found for: " << model_path << std::endl;
        return 1;
    }

    // Initialize Tokenizer
    auto tok = std::make_shared<g4dense::Tokenizer>();
    std::string tok_file = g4dense::resolve_resource_path("tokenizer.json");
    if (!fs::exists(tok_file)) {
        tok_file = g4dense::resolve_resource_path("tests/fixtures/tokenizer.json");
    }
    if (fs::exists(tok_file)) {
        if (!tok->load_vocabulary(tok_file)) {
            std::cerr << "Warning: Failed to parse tokenizer vocabulary at " << tok_file << std::endl;
        }
    } else {
        std::cerr << "Warning: tokenizer.json not found in search paths." << std::endl;
    }

    // CPU Reference Path
    if (cpu_mode) {
        std::cout << "Running pure scalar CPU Reference Oracle..." << std::endl;
        std::cout << "Loading model container: " << resolved_path << std::endl;
        g4dense::CpuReferenceConfig cpu_cfg;
        cpu_cfg.container_path = resolved_path;
        cpu_cfg.dump_tensors_dir = dump_tensors_dir;
        cpu_cfg.verbose = true;

        try {
            g4dense::CpuReferenceRunner oracle(cpu_cfg, tok);
            oracle.initialize();

            std::string gen_prompt = prompt.empty() ? "Hello! Introduce yourself." : prompt;
            std::cout << "\nPrompt: " << gen_prompt << "\n\nResponse:\n";

            g4dense::SamplingParams samp;
            samp.temperature = temp;
            samp.top_p = top_p;
            samp.top_k = top_k;

            auto t0 = std::chrono::high_resolution_clock::now();
            int tok_count = 0;

            oracle.generate(gen_prompt, max_tokens, samp, [&](uint32_t, const std::string& piece) {
                std::cout << piece << std::flush;
                tok_count++;
                return true;
            });

            auto t1 = std::chrono::high_resolution_clock::now();
            double total_sec = std::chrono::duration<double>(t1 - t0).count();

            std::cout << "\n\n========================================================"
                      << "\nCPU Reference Summary:"
                      << "\n  Tokens Generated: " << tok_count
                      << "\n  Elapsed Time:     " << total_sec << " s"
                      << "\n  Throughput (TPS): " << (tok_count / (total_sec > 0.0 ? total_sec : 1.0)) << " tokens/s"
                      << "\n========================================================"
                      << std::endl;
            return 0;
        } catch (const std::exception& ex) {
            std::cerr << "CPU Reference error: " << ex.what() << std::endl;
            return 1;
        }
    }

    // GPU Vulkan Path
    std::cout << "Initializing Vulkan 1.3 Compute Device..." << std::endl;
    auto ctx = std::make_shared<g4dense::VulkanContext>();
    ctx->initialize();
    std::cout << "Device: " << ctx->device_name() << " (Wave" << ctx->subgroup_size() << " Subgroups)" << std::endl;

    // Initialize Forward Runner
    std::cout << "Loading model container: " << resolved_path << std::endl;
    auto runner = std::make_shared<g4dense::ForwardRunner>(ctx, tok, resolved_path);
    // Speculative decoding keeps a second model resident. Reserve for it before the target's
    // layer import runs, because that import takes everything it is allowed to.
    //
    // The reserve and the load must agree: reserving without loading costs resident
    // layers for nothing, and loading without reserving fails outright once the target's
    // greedy import has taken the budget.
    std::string active_draft_path = "";
    uint64_t draft_reserve_bytes = 0;
    if (speculative) {
        if (!draft_model_path.empty()) {
            if (fs::exists(draft_model_path)) {
                active_draft_path = draft_model_path;
            } else {
                std::cerr << "Warning: Specified draft model not found: " << draft_model_path << std::endl;
            }
        } else if (fs::exists(kMtpDraftModelPath)) {
            active_draft_path = kMtpDraftModelPath;
        } else if (fs::exists(kE2bDraftModelPath)) {
            active_draft_path = kE2bDraftModelPath;
        }

        if (!active_draft_path.empty()) {
            bool is_mtp = false;
            std::ifstream f(active_draft_path, std::ios::binary);
            if (f.is_open()) {
                uint32_t magic = 0;
                f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                if (magic == g4dense::G4MtpHeader::EXPECTED_MAGIC) {
                    is_mtp = true;
                }
            }
            draft_reserve_bytes = is_mtp ? kMtpDraftImportReserveBytes : kE2bDraftImportReserveBytes;
            if (!draft_k_specified) {
                draft_k = is_mtp ? 5 : 6;
            }
        }
    }

    const bool draft_wanted = speculative && !active_draft_path.empty();
    if (draft_wanted) {
        runner->set_import_reserve(draft_reserve_bytes);
    }
    if (max_context != 0) runner->set_max_context(max_context);
    runner->initialize();
    runner->switch_memory_tier(tier_id);
    // Not "pinned layers active": pinning was removed in round 6, and residency is decided by
    // how much host memory the driver accepts, identically for every tier.
    std::cout << "Memory tier " << tier_id << " selected (residency is driver-determined)." << std::endl;

    // Load the draft HERE, not after the server branch.
    //
    // --server / --gui start the server below and never return, so a draft loaded further
    // down was only ever loaded in one-shot CLI mode. Server mode therefore paid the
    // import reserve and got no speculation for it.
    if (draft_wanted) {
        runner->load_draft_model(active_draft_path);
    }

    if (run_server) {
        g4dense::HTTPServer server;
        server.start(port, runner, ctx);
        std::cout << "\nServer listening on http://127.0.0.1:" << port << std::endl;
        std::cout << "Web GUI available at http://127.0.0.1:" << port << "/\n" << std::endl;

        if (open_gui) {
            std::string url = "http://127.0.0.1:" + std::to_string(port);
            ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
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
              << "\n  Draft acceptance: " << (tele.speculative_acceptance_rate * 100.0) << " %"
              << " (" << tele.speculative_accepted << "/" << tele.speculative_drafted << ")"
              // Printed here rather than when it happens: the gate trips mid-stream, and
              // anything written there lands inside the generated text.
              << "\n  Drafting:         "
              << (!tele.has_draft ? "no drafter loaded (--spec to enable)"
                                  : (tele.drafting_gated ? "GATED OFF -- acceptance below threshold"
                                                         : "active"))
              // Per-forward-pass attribution. Printed so an optimization can be credited to a
              // phase instead of guessed at; the numbers are an EMA over the pass, so a single
              // cold pass does not dominate them.
              << "\n  --- per forward pass (ms) ---"
              << "\n  Layer stream I/O: " << tele.stream_io_ms
              << "\n  GPU queue wait:   " << tele.gpu_wait_ms
              << "\n  LM head:          " << tele.lm_head_ms
              << "\n  CPU other:        " << tele.cpu_other_ms
              << "\n========================================================"
              << std::endl;

    return 0;
}
