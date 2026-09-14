#include "g4dense/runner.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/manifest.hpp"
#include "g4dense/telemetry.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <windows.h>
#include <psapi.h>

namespace fs = std::filesystem;
using namespace g4dense;

struct PromptTestCase {
    std::string category;
    std::string prompt;
    int max_tokens;
    float temperature;
    bool verify_determinism;
    std::string expected_keyword;
};

static size_t get_ram_working_set_bytes() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
}

static std::string truncate_str(const std::string& str, size_t max_len) {
    if (str.length() <= max_len) return str;
    return str.substr(0, max_len - 3) + "...";
}

int main(int argc, char** argv) {
    std::cout << "========================================================\n"
              << "  Turbo-WinFare Dense: Deep Random Prompts Test Suite   \n"
              << "========================================================\n" << std::flush;

    std::string model_path = (argc > 1) ? argv[1] : "tests/fixtures/tiny_muse.g4dense";
    size_t limit_cases = (argc > 2) ? static_cast<size_t>(std::atoi(argv[2])) : 0;
    int override_max_tokens = (argc > 3) ? std::atoi(argv[3]) : 0;
    float override_temp = (argc > 4) ? static_cast<float>(std::atof(argv[4])) : -1.0f;

    if (!fs::exists(model_path)) {
        std::cerr << "FAIL: model container not found at " << model_path << "\n";
        return 1;
    }

    try {
        auto ctx = std::make_shared<VulkanContext>();
        ctx->initialize();

        std::cout << "Vulkan Device: " << ctx->device_name()
                  << " (Subgroups: " << ctx->subgroup_size() << ")\n" << std::flush;

        auto tok = std::make_shared<Tokenizer>();
        tok->load_vocabulary(resolve_resource_path("tokenizer.json"));
        if (!tok->is_loaded()) {
            std::cerr << "FAIL: tokenizer could not be loaded from tokenizer.json\n";
            return 1;
        }

        std::cout << "Loading model: " << model_path << " ...\n" << std::flush;
        auto t_load_start = std::chrono::high_resolution_clock::now();
        ForwardRunner runner(ctx, tok, model_path);
        runner.initialize();
        auto t_load_end = std::chrono::high_resolution_clock::now();
        double load_sec = std::chrono::duration<double>(t_load_end - t_load_start).count();

        const auto& hdr = runner.header();
        const bool is_muse = is_muse_glimmer(hdr);
        const bool is_real_31b = (hdr.num_layers >= 50);

        std::cout << "Model Loaded in " << std::fixed << std::setprecision(2) << load_sec << " s\n"
                  << "  Architecture: " << (is_muse ? "Muse-Glimmer-30B" : "Gemma 4 31B Dense") << "\n"
                  << "  Layers:       " << hdr.num_layers << " (resident: " << (hdr.num_layers - runner.streamed_layers().size()) << ")\n"
                  << "  Model Dim:    " << hdr.d_model << ", FFN Dim: " << hdr.d_ff << "\n"
                  << "  Vocab Size:   " << hdr.vocab_size << "\n\n" << std::flush;

        // Comprehensive taxonomy of test prompts
        std::vector<PromptTestCase> test_cases = {
            // Category: Factual Knowledge, Science & Arithmetic
            {"Arithmetic", "The product of 7 and 8 is ", 6, 0.0f, true, "56"},
            {"Geography", "What is the capital of France? The capital is ", 6, 0.0f, true, "Paris"},
            {"Chemistry", "Question: What is the chemical formula for water? Answer: ", 6, 0.0f, true, "H2O"},
            {"Astronomy", "Which planet is known as the Red Planet? The planet is ", 6, 0.0f, true, "Mars"},
            {"Physics", "The force on an object equals mass times ", 6, 0.0f, true, "acceleration"},
            {"Biology", "Question: What process do plants use to convert sunlight into energy? Answer: ", 6, 0.0f, true, "photosynthesis"},
            {"Proverb", "A journey of a thousand miles begins with a single ", 6, 0.0f, true, "step"},
            {"Code-Python", "def square(x):\n    return ", 8, 0.0f, true, ""},
            {"Poetry", "The silver moon illuminates the night,\n", 12, 0.0f, true, ""},
            {"Sampling-Creative", "Once upon a time in a celestial kingdom, ", 12, 0.7f, false, ""},

            // Category: Math & Quantitative Reasoning
            {"Algebra", "If 2x = 10, then x = ", 6, 0.0f, true, "5"},
            {"Arithmetic-15", "Compute 15 * 6 = ", 6, 0.0f, true, "90"},
            {"Number Theory", "The smallest prime number is ", 6, 0.0f, true, "2"},

            // Category: Coding & Technical Algorithms
            {"Code-C++", "#include <iostream>\nint main() {\n    std::cout << ", 10, 0.0f, true, ""},
            {"Code-SQL", "SELECT name, email FROM customers WHERE active = ", 8, 0.0f, true, ""},
            {"Data-Structures", "A queue follows First-In, First-", 6, 0.0f, true, "Out"},

            // Category: Linguistics & Language Understanding
            {"Language-FR", "Translate 'cat' into French: ", 6, 0.0f, true, "chat"},
            {"Language-ES", "Translate 'sun' into Spanish: ", 6, 0.0f, true, "sol"},
            {"Grammar", "The plural of 'mouse' is ", 6, 0.0f, true, "mice"},
            {"Philosophy", "I think, therefore I ", 6, 0.0f, true, "am"},

            // Category: Creative & Dialogue
            {"Sci-Fi", "The quantum engine engaged, warping space into ", 12, 0.0f, true, ""},

            // Category: Edge Cases & Syntax Stress
            {"Short-Prompt", "The ", 6, 0.0f, true, ""},
            {"Numeric-Seq", "1, 2, 4, 8, 16, ", 6, 0.0f, true, ""},
            {"Boolean-Logic", "True and False evaluates to ", 6, 0.0f, true, "False"},
            {"Punctuation", "Math symbols: x + y = z; a * b = ", 8, 0.0f, true, ""},
            {"JSON-Syntax", "{\"name\": \"Alice\", \"age\": 30, \"role\": \"", 8, 0.0f, true, ""},
            {"Long-Context", "In a historic city surrounded by ancient stone walls, travelers gathered at the central marketplace where merchants traded rare spices, silks, and books. An old scholar approached the fountain and said,", 16, 0.0f, true, ""},
            {"Sampling-Code", "def fibonacci(n): ", 12, 0.6f, false, ""}
        };

        if (limit_cases > 0 && limit_cases < test_cases.size()) {
            test_cases.resize(limit_cases);
        }

        std::cout << "Running " << test_cases.size() << " Deep Random Prompt Test Cases...\n"
                  << "----------------------------------------------------------------------------------------------------\n"
                  << std::left << std::setw(4)  << "#"
                  << std::left << std::setw(18) << "Category"
                  << std::left << std::setw(8)  << "Temp"
                  << std::left << std::setw(6)  << "Toks"
                  << std::left << std::setw(10) << "Time (s)"
                  << std::left << std::setw(10) << "TPS"
                  << std::left << std::setw(12) << "Determ."
                  << std::left << std::setw(10) << "Status"
                  << "Preview Response\n"
                  << "----------------------------------------------------------------------------------------------------\n"
                  << std::flush;

        size_t initial_ram = get_ram_working_set_bytes();
        int passed = 0;
        int failed = 0;
        int determinism_tested = 0;
        int determinism_passed = 0;
        double total_gen_sec = 0.0;
        uint64_t total_tokens = 0;
        std::vector<double> tps_list;

        for (size_t i = 0; i < test_cases.size(); ++i) {
            const auto& tc = test_cases[i];
            int max_tokens = (override_max_tokens > 0) ? override_max_tokens : tc.max_tokens;
            float temperature = (override_temp >= 0.0f) ? override_temp : tc.temperature;

            GenerationOptions opts;
            opts.max_tokens = max_tokens;
            opts.sampling.temperature = temperature;
            opts.sampling.top_p = 0.90f;
            opts.sampling.top_k = 40;
            opts.speculative_enabled = false;
            opts.active_tier_id = 1;

            // Run 1
            std::vector<uint32_t> tokens_run1;
            std::string text_run1;
            auto t0 = std::chrono::high_resolution_clock::now();

            runner.generate(tc.prompt, opts, [&](uint32_t token, const std::string& piece) {
                tokens_run1.push_back(token);
                text_run1 += piece;
                return true;
            });

            auto t1 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            double tps = (sec > 0.0) ? (tokens_run1.size() / sec) : 0.0;

            total_gen_sec += sec;
            total_tokens += tokens_run1.size();
            tps_list.push_back(tps);

            bool case_passed = true;
            std::string determ_status = "N/A";

            // If greedy mode and requested, verify 100% determinism with Run 2
            if (tc.verify_determinism && temperature == 0.0f) {
                ++determinism_tested;
                std::vector<uint32_t> tokens_run2;
                std::string text_run2;

                runner.generate(tc.prompt, opts, [&](uint32_t token, const std::string& piece) {
                    tokens_run2.push_back(token);
                    text_run2 += piece;
                    return true;
                });

                if (tokens_run1 == tokens_run2) {
                    ++determinism_passed;
                    determ_status = "100% Match";
                } else {
                    determ_status = "MISMATCH";
                    case_passed = false;
                }
            }

            // Accuracy check for real 31B model on known keywords
            if (is_real_31b && !tc.expected_keyword.empty()) {
                std::string lower_text = text_run1;
                std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
                std::string lower_exp = tc.expected_keyword;
                std::transform(lower_exp.begin(), lower_exp.end(), lower_exp.begin(), ::tolower);

                if (lower_text.find(lower_exp) == std::string::npos) {
                    if (temperature == 0.0f) {
                        case_passed = false;
                    }
                }
            }

            if (tokens_run1.empty()) {
                case_passed = false;
            }

            if (case_passed) ++passed;
            else ++failed;

            std::string clean_text = "";
            for (char c : text_run1) {
                if (c == '\n') clean_text += "\\n";
                else if (c == '\r') continue;
                else clean_text += c;
            }

            std::cout << std::left << std::setw(4)  << (i + 1)
                      << std::left << std::setw(18) << truncate_str(tc.category, 17)
                      << std::left << std::setw(8)  << std::fixed << std::setprecision(1) << temperature
                      << std::left << std::setw(6)  << tokens_run1.size()
                      << std::left << std::setw(10) << std::fixed << std::setprecision(2) << sec
                      << std::left << std::setw(10) << std::fixed << std::setprecision(2) << tps
                      << std::left << std::setw(12) << determ_status
                      << std::left << std::setw(10) << (case_passed ? "[PASS]" : "[FAIL]")
                      << "\"" << truncate_str(clean_text, 35) << "\"\n"
                      << std::flush;
        }

        size_t final_ram = get_ram_working_set_bytes();
        double ram_delta_mb = static_cast<double>(static_cast<int64_t>(final_ram) - static_cast<int64_t>(initial_ram)) / (1024.0 * 1024.0);

        std::sort(tps_list.begin(), tps_list.end());
        double min_tps = tps_list.empty() ? 0.0 : tps_list.front();
        double max_tps = tps_list.empty() ? 0.0 : tps_list.back();
        double mean_tps = (total_gen_sec > 0.0) ? (total_tokens / total_gen_sec) : 0.0;

        std::cout << "----------------------------------------------------------------------------------------------------\n"
                  << "\n========================================================\n"
                  << "  DEEP RANDOM PROMPTS SUITE SUMMARY                     \n"
                  << "========================================================\n"
                  << "  Total Test Cases:        " << test_cases.size() << "\n"
                  << "  Passed:                  " << passed << " / " << test_cases.size()
                  << " (" << std::fixed << std::setprecision(1) << (100.0 * passed / test_cases.size()) << "%)\n"
                  << "  Failed:                  " << failed << "\n"
                  << "  Determinism Checks:      " << determinism_passed << " / " << determinism_tested
                  << " (" << (determinism_tested > 0 ? (100.0 * determinism_passed / determinism_tested) : 100.0) << "%)\n"
                  << "  Total Tokens Generated:  " << total_tokens << "\n"
                  << "  Total Generation Time:   " << std::fixed << std::setprecision(2) << total_gen_sec << " s\n"
                  << "  Throughput (TPS):        Mean=" << std::setprecision(2) << mean_tps
                  << ", Min=" << min_tps << ", Max=" << max_tps << " tok/s\n"
                  << "  Working Set RAM Drift:   " << std::showpos << std::setprecision(2) << ram_delta_mb << " MB"
                  << std::noshowpos << " (initial: " << (initial_ram / (1024 * 1024)) << " MB, final: " << (final_ram / (1024 * 1024)) << " MB)\n"
                  << "========================================================\n\n";

        if (failed > 0) {
            std::cerr << "FAILED: " << failed << " test cases failed.\n";
            return 1;
        }

        std::cout << "[SUCCESS] ALL DEEP RANDOM PROMPT TESTS PASSED 100%!\n";
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "\nFATAL EXCEPTION: " << ex.what() << "\n";
        return 1;
    }
}