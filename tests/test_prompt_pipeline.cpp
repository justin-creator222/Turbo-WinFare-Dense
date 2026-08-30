// What does a prompt actually become before it reaches the model?
//
// The full forward pass is verified correct: all 60 layers, the final norm and the logits
// match an independent NumPy reference built from the checkpoint, and that reference
// independently predicts the same top-5 tokens. Yet generation emits the position-0 argmax
// forever, which means the prompt is not influencing the output.
//
// This checks the only remaining stage between a user's string and the model: chat templating
// and tokenization. It needs no GPU and no model container -- just the tokenizer.

#include "g4dense/tokenizer.hpp"
#include "g4dense/manifest.hpp"

#include <iostream>
#include <string>
#include <vector>

using namespace g4dense;

int main() {
    std::cout << "========================================================\n"
              << "  Prompt pipeline: string -> template -> tokens          \n"
              << "========================================================\n";

    Tokenizer tok;
    tok.load_vocabulary(resolve_resource_path("tokenizer.json"));
    if (!tok.is_loaded()) {
        std::cerr << "FAIL: tokenizer did not load\n";
        return 1;
    }
    std::cout << "vocab_size=" << tok.vocab_size()
              << " bos=" << tok.bos_id() << " eos=" << tok.eos_id()
              << " end_of_turn=" << tok.end_of_turn_id() << "\n\n";

    const std::string prompt = "What is the capital of France?";

    // 1. Raw encode, for comparison.
    std::vector<uint32_t> raw = tok.encode(prompt, true);
    std::cout << "RAW encode (" << raw.size() << " tokens):\n  ";
    for (uint32_t t : raw) std::cout << t << " ";
    std::cout << "\n  round-trip: \"" << tok.decode(raw, false) << "\"\n\n";

    // 2. Through the chat template, which is what generate() now does.
    Tokenizer::ChatMessage msg{"user", prompt};
    std::string templated = tok.apply_chat_template({msg});
    std::cout << "TEMPLATE text (" << templated.size() << " chars):\n  \"";
    for (char c : templated) {
        if (c == '\n') std::cout << "\\n";
        else std::cout << c;
    }
    std::cout << "\"\n\n";

    std::vector<uint32_t> ids = tok.encode(templated, false);
    std::cout << "TEMPLATED encode (" << ids.size() << " tokens):\n  ";
    for (uint32_t t : ids) std::cout << t << " ";
    std::cout << "\n\n  per-token:\n";
    for (size_t i = 0; i < ids.size(); ++i) {
        std::cout << "    [" << i << "] " << ids[i]
                  << "  \"" << tok.decode_single(ids[i], false) << "\"\n";
    }
    std::cout << "\n  round-trip: \"" << tok.decode(ids, false) << "\"\n";

    bool ok = true;
    if (ids.empty()) {
        std::cerr << "\nFAIL: templated prompt encoded to nothing\n";
        ok = false;
    }
    if (!ids.empty() && ids[0] != tok.bos_id()) {
        std::cerr << "\nFAIL: first token is " << ids[0] << ", expected BOS "
                  << tok.bos_id() << "\n";
        ok = false;
    }
    // The prompt's own words must survive templating. If the encode collapses to a handful of
    // special tokens, the model never sees the question -- which is exactly the failure mode
    // under investigation.
    if (ids.size() < 8) {
        std::cerr << "\nFAIL: only " << ids.size() << " tokens for a templated 30-character "
                     "question; the prompt text is not surviving encode\n";
        ok = false;
    }

    std::cout << (ok ? "\nPrompt pipeline looks sane.\n"
                     : "\nPrompt pipeline is BROKEN.\n");
    return ok ? 0 : 1;
}
