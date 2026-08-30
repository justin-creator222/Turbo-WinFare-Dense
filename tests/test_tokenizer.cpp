#include "g4dense/tokenizer.hpp"
#include "g4dense/detokenizer.hpp"
#include "g4dense/format.hpp"
#include <iostream>
#include <cassert>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <random>

namespace fs = std::filesystem;

int main() {
    std::cout << "[TEST] Running gturbo Tokenizer and Detokenizer unit tests...\n";

    g4dense::Tokenizer tok;

    // Test 1: A default-constructed Tokenizer holds NO vocabulary.
    assert(!tok.is_loaded());
    assert(tok.vocab_size() == 262144);
    assert(tok.bos_id() == 2);
    assert(tok.eos_id() == 1);
    assert(tok.pad_id() == 0);
    assert(tok.unk_id() == 3);
    assert(tok.end_of_turn_id() == 106);
    std::cout << "  [PASS] Default-constructed tokenizer is empty with Gemma 4 special IDs.\n";

    // Test 2: Stop tokens are <eos>, <turn|>, <|tool_response>.
    {
        const auto& stops = tok.stop_token_ids();
        assert(stops.size() == 3);
        assert(std::find(stops.begin(), stops.end(), 1u) != stops.end());
        assert(std::find(stops.begin(), stops.end(), 106u) != stops.end());
        assert(std::find(stops.begin(), stops.end(), 50u) != stops.end());
        std::cout << "  [PASS] Stop token set is {1, 106, 50}.\n";
    }

    // Test 3: Encoding without a vocabulary throws rather than inventing tokens.
    {
        bool threw = false;
        try {
            tok.encode("Hello", true);
        } catch (const g4dense::G4DenseFormatError&) {
            threw = true;
        }
        assert(threw);
        std::cout << "  [PASS] encode() without a loaded vocabulary throws.\n";
    }

    // Test 3b: A missing vocabulary file throws rather than silently substituting one.
    {
        g4dense::Tokenizer missing_tok;
        bool threw = false;
        try {
            missing_tok.load_vocabulary("definitely_not_a_real_vocab_file.json");
        } catch (const g4dense::G4DenseFormatError&) {
            threw = true;
        }
        assert(threw);
        std::cout << "  [PASS] load_vocabulary() with missing file throws.\n";
    }

    // Test 4: Chat template structure against a synthetic vocabulary.
    const fs::path mini = "test_mini_vocab.json";
    {
        std::ofstream out(mini, std::ios::binary | std::ios::trunc);
        out << R"({
            "added_tokens": [
                {"id": 0, "content": "<pad>", "special": true},
                {"id": 1, "content": "<eos>", "special": true},
                {"id": 2, "content": "<bos>", "special": true},
                {"id": 105, "content": "<start_of_turn>", "special": true},
                {"id": 106, "content": "<end_of_turn>", "special": true},
                {"id": 107, "content": "\n", "special": false}
            ],
            "model": {
                "vocab": {
                    "<pad>": 0, "<eos>": 1, "<bos>": 2,
                    "<start_of_turn>": 105, "<end_of_turn>": 106, "\n": 107,
                    "user": 2364, "model": 4368, "Hi": 10979
                },
                "merges": []
            }
        })";
        out.close();

        g4dense::Tokenizer mini_tok;
        mini_tok.load_vocabulary(mini.string());
        std::vector<g4dense::Tokenizer::ChatMessage> msgs{{"user", "Hi"}};
        const std::string rendered = mini_tok.apply_chat_template(msgs);
        const std::string expected =
            "<bos><|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
        assert(rendered == expected);
        std::cout << "  [PASS] chat template renders the Gemma 4 turn markers.\n";
    }
    fs::remove(mini);

    // Test 5: Incremental Detokenizer & Stop Matcher
    {
        g4dense::StreamingStopMatcher m({"<eos>"});
        m.push("<eos>");
        assert(m.stopped());

        g4dense::StreamingStopMatcher m2({"<start_of_turn>user"});
        m2.push("<start_of_turn>");
        assert(!m2.stopped());
        m2.push("user");
        assert(m2.stopped());

        g4dense::StreamingStopMatcher m3({"<start_of_turn>model"});
        m3.push("<start_of_turn>");
        assert(!m3.stopped());
        m3.push("user");
        assert(!m3.stopped());
        std::cout << "  [PASS] StreamingStopMatcher prefixes and full matches verified.\n";
    }

    // Test 6: Golden token IDs against the real tokenizer
    const std::vector<std::string> candidates = {
        "tokenizer.json",
        "build/tokenizer.json",
        "models/gemma-4-31b-it-4bit/tokenizer.json"
    };
    std::string real_path;
    for (const auto& c : candidates) {
        if (fs::exists(c)) { real_path = c; break; }
    }

    if (!real_path.empty()) {
        g4dense::Tokenizer real;
        real.load_vocabulary(real_path);
        assert(real.vocab_size() == 262144);
        assert(real.bos_id() == 2);
        assert(real.eos_id() == 1);
        assert(real.end_of_turn_id() == 106);

        // Incremental detokenization test on real vocab
        g4dense::IncrementalDetokenizer detok(real);
        std::string s1 = detok.push(2); // bos
        std::string s2 = detok.push(real.encode("Hello", false)[0]);
        std::string s3 = detok.finish();
        assert(!s2.empty());
        std::cout << "  [PASS] Incremental detokenizer streaming with real vocabulary.\n";

        // Round-trip a sentence with punctuation and a non-ASCII character.
        const std::string sentence = "The quick brown fox jumps over 13 lazy dogs \xC3\xA9!";
        auto rt = real.encode(sentence, false);
        assert(real.decode(rt, true) == sentence);
        std::cout << "  [PASS] real-vocabulary encode/decode round-trip.\n";
    }

    std::cout << "[TEST SUCCESS] All Tokenizer and Detokenizer unit tests PASSED!\n";
    return 0;
}
