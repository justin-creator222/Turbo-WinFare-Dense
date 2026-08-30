// Incremental detokenization and streaming stop matching.
//
// The invariant that matters most is at the bottom: streaming a sequence token by token must
// produce exactly the same bytes as decoding it in one batch. If it does not, streamed output
// differs from non-streamed output for the same generation.

#include "g4dense/detokenizer.hpp"
#include "g4dense/tokenizer.hpp"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace g4dense;

namespace {

std::string write_mini_vocab() {
    std::string json = R"({"added_tokens":[{"id":0,"content":"<pad>","special":true},)"
                       R"({"id":1,"content":"<eos>","special":true}],)"
                       R"("model":{"vocab":{"<pad>":0,"<eos>":1,)"
                       R"("Hi":2," there":3,"!":4)";
    for (int b = 0; b < 256; ++b) {
        char piece[16];
        std::snprintf(piece, sizeof(piece), "<0x%02X>", b);
        json += ",\"";
        json += piece;
        json += "\":" + std::to_string(100 + b);
    }
    json += R"(},"merges":[]}})";

    fs::path p = "test_detok_vocab.json";
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out << json;
    out.close();
    return p.string();
}

std::string stream_all(const Tokenizer& tok, const std::vector<uint32_t>& ids) {
    IncrementalDetokenizer d(tok);
    std::string out;
    for (uint32_t id : ids) out += d.push(id);
    out += d.finish();
    return out;
}

} // namespace

int main() {
    std::cout << "[TEST] Running incremental detokenizer / stop matcher tests...\n";

    const std::string vocab_path = write_mini_vocab();
    Tokenizer tok;
    if (!tok.load_vocabulary(vocab_path)) {
        std::cout << "  [FAIL] could not load the synthetic vocabulary\n";
        return 1;
    }

    // ---- A multi-byte character split across byte-fallback tokens -------------------
    {
        // U+20AC EURO SIGN is E2 82 AC: three tokens that mean nothing individually.
        IncrementalDetokenizer d(tok);
        assert(d.push(100u + 0xE2).empty() && "emitted a partial UTF-8 sequence");
        assert(d.push(100u + 0x82).empty() && "emitted a partial UTF-8 sequence");
        const std::string third = d.push(100u + 0xAC);
        assert(third == "\xE2\x82\xAC");
        assert(d.finish().empty());
        std::cout << "  [PASS] Multi-byte character emitted only once complete.\n";
    }

    // ---- An incomplete trailing sequence becomes U+FFFD, and does not throw ----------
    {
        IncrementalDetokenizer d(tok);
        assert(d.push(100u + 0xE2).empty());
        const std::string tail = d.finish();
        assert(tail == "\xEF\xBF\xBD" && "dangling sequence should become U+FFFD");
        std::cout << "  [PASS] Dangling byte run flushes as U+FFFD.\n";
    }

    // ---- Byte run followed by an ordinary piece -------------------------------------
    {
        IncrementalDetokenizer d(tok);
        std::string out;
        out += d.push(100u + 0xE2);
        out += d.push(100u + 0x82);
        out += d.push(100u + 0xAC);
        out += d.push(2);            // "Hi"
        out += d.finish();
        assert(out == "\xE2\x82\xAC" "Hi");
        std::cout << "  [PASS] Byte run then ordinary piece concatenates correctly.\n";
    }

    // ---- Special tokens are skipped --------------------------------------------------
    {
        IncrementalDetokenizer d(tok);
        std::string out;
        out += d.push(2);            // "Hi"
        out += d.push(1);            // <eos>, special
        out += d.push(4);            // "!"
        out += d.finish();
        assert(out == "Hi!");
        std::cout << "  [PASS] Special tokens skipped while streaming.\n";
    }

    // The matcher's contract is push()/finish(), not feed(). It returns the text that is
    // SAFE TO EMIT, withholding any tail that could still become a stop string, so a stop is
    // never partially shown to the caller. These three cases were written against an API that
    // never existed, which is why this file stopped compiling and was dropped from the build
    // rather than fixed.

    // ---- Streaming Stop Matcher: exact single token ----------------------------------
    {
        StreamingStopMatcher m({"<eos>"});
        std::string emitted = m.push("<eos>");
        assert(emitted.empty());          // the stop itself is never emitted
        assert(m.stopped());
        assert(m.matched() == "<eos>");
        std::cout << "  [PASS] Exact stop matched and withheld from output.\n";
    }

    // ---- Streaming Stop Matcher: stop split across two chunks ------------------------
    {
        StreamingStopMatcher m({"<start_of_turn>user"});
        std::string emitted = m.push("<start_of_turn>");
        assert(emitted.empty());          // withheld: still a viable prefix
        assert(!m.stopped());
        emitted += m.push("user");
        assert(emitted.empty());          // completes the stop; nothing emitted
        assert(m.stopped());
        assert(m.matched() == "<start_of_turn>user");
        std::cout << "  [PASS] Multi-chunk stop phrase matched across chunks.\n";
    }

    // ---- Streaming Stop Matcher: viable prefix that then diverges --------------------
    {
        StreamingStopMatcher m({"<start_of_turn>model"});
        std::string emitted = m.push("<start_of_turn>");
        assert(emitted.empty());          // withheld while it could still match
        emitted += m.push("user");        // diverges -> the withheld text must be released
        emitted += m.finish();
        assert(!m.stopped());
        assert(emitted == "<start_of_turn>user");
        std::cout << "  [PASS] Divergent prefix releases withheld text and does not stop.\n";
    }

    // ---- Property: streaming matches batch decoding on random byte sequences --------
    {
        std::mt19937_64 rng(0xD370C0D3ULL);
        std::uniform_int_distribution<int> len_dist(1, 64);
        std::uniform_int_distribution<int> byte_dist(0, 255);

        for (int trial = 0; trial < 200; ++trial) {
            const int len = len_dist(rng);
            std::vector<uint32_t> ids;
            ids.reserve(len);
            for (int i = 0; i < len; ++i) {
                ids.push_back(100u + static_cast<uint32_t>(byte_dist(rng)));
            }

            const std::string streamed = stream_all(tok, ids);
            const std::string batched  = tok.decode(ids, false);
            assert(streamed == batched && "streamed decode diverged from batch decode");
        }
        std::cout << "  [PASS] Streamed decode == batch decode across 200 random byte sequences.\n";
    }

    std::remove(vocab_path.c_str());
    std::cout << "[test_detokenizer] ALL INCREMENTAL DETOKENIZER CHECKS PASSED!\n";
    return 0;
}
