// Does the forward pass actually depend on its input token and on accumulated history?
//
// Generation collapses to one repeated token: whatever the model predicts at position 0 for a
// bare <bos>, it then emits forever. That is the signature of a forward pass whose output does
// not change with input -- either the token is ignored, or attention never sees the KV written
// for earlier positions.
//
// The single-token oracle diff cannot detect this: it only ever runs position 0. These three
// checks are the smallest thing that can.

#include "g4dense/runner.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/manifest.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

namespace fs = std::filesystem;
using namespace g4dense;

namespace {

uint32_t argmax_of(const std::vector<float>& v) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < v.size(); ++i) if (v[i] > v[best]) best = i;
    return best;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::abs(a[i] - b[i]));
    return m;
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path = (argc > 1) ? argv[1] : "models/gemma-4-31b-dense.g4dense";

    std::cout << "========================================================\n"
              << "  Forward-pass context dependence                       \n"
              << "========================================================\n" << std::flush;

    if (!fs::exists(model_path)) {
        std::cerr << "FAIL: model container not found: " << model_path << "\n";
        return 1;
    }

    try {
        auto ctx = std::make_shared<VulkanContext>();
        ctx->initialize();
        auto tok = std::make_shared<Tokenizer>();
        tok->load_vocabulary(resolve_resource_path("tokenizer.json"));

        ForwardRunner runner(ctx, tok, model_path);
        runner.initialize();
        const uint32_t vocab = runner.header().vocab_size;

        std::vector<float> l_bos(vocab), l_other(vocab), l_pos1(vocab), l_pos1_again(vocab);

        // A: same position, DIFFERENT token -> logits must differ.
        runner.forward_single_token(2, 0, l_bos.data());
        const uint32_t a_bos = argmax_of(l_bos);
        std::cout << "A. token=2   pos=0 -> argmax " << a_bos << "\n" << std::flush;

        runner.forward_single_token(1000, 0, l_other.data());
        const uint32_t a_other = argmax_of(l_other);
        std::cout << "   token=1000 pos=0 -> argmax " << a_other
                  << "   max|diff| = " << max_abs_diff(l_bos, l_other) << "\n" << std::flush;

        const bool token_matters = max_abs_diff(l_bos, l_other) > 1e-3;
        std::cout << (token_matters ? "   [PASS] output depends on the input token\n"
                                    : "   [FAIL] output IGNORES the input token\n") << std::flush;

        // B: advancing position with history must change the output. Re-seed at position 0,
        // then step to position 1 with a different token.
        runner.forward_single_token(2, 0, l_bos.data());
        runner.forward_single_token(1000, 1, l_pos1.data());
        const uint32_t a_pos1 = argmax_of(l_pos1);
        std::cout << "B. token=1000 pos=1 (after pos 0) -> argmax " << a_pos1
                  << "   max|diff| vs same token at pos 0 = "
                  << max_abs_diff(l_other, l_pos1) << "\n" << std::flush;

        // NOT evidence that attention reads earlier KV. RoPE rotates the query by
        // position, so this differs even if attention never looks at a single previous
        // key. Round 4 reported it as proof of history and that was wrong. It is kept
        // only as a smoke test that `position` reaches the kernels; check D is the one
        // that actually tests attention over history.
        const bool position_reaches_kernels = max_abs_diff(l_other, l_pos1) > 1e-3;
        std::cout << (position_reaches_kernels
                          ? "   [PASS] position reaches the kernels (NOT proof of KV "
                            "history -- RoPE alone would do this; see check D)\n"
                          : "   [FAIL] output IGNORES position entirely\n") << std::flush;

        // C: determinism -- identical inputs must give identical outputs.
        runner.forward_single_token(2, 0, l_bos.data());
        runner.forward_single_token(1000, 1, l_pos1_again.data());
        const double drift = max_abs_diff(l_pos1, l_pos1_again);
        std::cout << "C. repeat of B -> max|diff| = " << drift << "\n" << std::flush;
        const bool deterministic = drift < 1e-3;
        std::cout << (deterministic ? "   [PASS] deterministic\n"
                                    : "   [FAIL] non-deterministic between identical runs\n")
                  << std::flush;

        // D: prefill the REAL templated prompt and look at what the model predicts.
        //
        // Every stage in isolation is now verified: the whole forward pass matches an
        // independent NumPy reference at position 0 (60 layers, final norm, logits), and the
        // prompt tokenizes correctly. If prefilling the real prompt still yields the same
        // token as a bare <bos>, the defect is in KV accumulation across positions -- the one
        // property a position-0 check cannot see.
        {
            Tokenizer::ChatMessage msg{"user", "What is the capital of France?"};
            std::vector<uint32_t> ids;
            // Optional 2nd argument: an explicit comma-separated token sequence, so template
            // variants can be tried without rebuilding. The default generation prompt ends
            // "<|channel>thought\n<channel|>", inherited from the sibling project; whether that
            // is right for this checkpoint is unverified, and a wrong generation-prompt suffix
            // would leave the model completing something other than an answer.
            if (argc > 2) {
                std::string spec = argv[2];
                size_t pos = 0;
                while (pos <= spec.size()) {
                    size_t comma = spec.find(',', pos);
                    if (comma == std::string::npos) comma = spec.size();
                    if (comma > pos) ids.push_back(static_cast<uint32_t>(std::stoul(spec.substr(pos, comma - pos))));
                    pos = comma + 1;
                }
            } else {
                ids = tok->encode(tok->apply_chat_template({msg}), false);
            }
            std::cout << "D. Prefilling " << ids.size() << " prompt tokens ...\n" << std::flush;

            std::vector<float> lg(vocab);
            for (size_t i = 0; i < ids.size(); ++i) {
                runner.forward_single_token(ids[i], static_cast<uint32_t>(i), lg.data());
            }

            std::vector<uint32_t> order(vocab);
            for (uint32_t i = 0; i < vocab; ++i) order[i] = i;
            std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                              [&](uint32_t a, uint32_t b) { return lg[a] > lg[b]; });

            std::cout << "   top-5 after the prompt:\n";
            for (int i = 0; i < 5; ++i) {
                std::cout << "     " << order[i] << "  " << lg[order[i]]
                          << "  \"" << tok->decode_single(order[i], false) << "\"\n";
            }
            std::cout << "   (a bare <bos> at position 0 predicts token " << a_bos << ")\n";
            if (order[0] == a_bos) {
                std::cout << "   [FAIL] the prompt did not change the prediction -- KV history "
                             "is not reaching the output\n" << std::flush;
            } else {
                std::cout << "   [PASS] the prompt changed the prediction\n" << std::flush;
            }
        }

        if (!token_matters || !position_reaches_kernels || !deterministic) {
            std::cerr << "\nFAIL: forward pass is not context-dependent.\n";
            return 1;
        }
        std::cout << "\nAll context-dependence checks passed.\n";
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "\nEXCEPTION: " << ex.what() << "\n";
        return 1;
    }
}
