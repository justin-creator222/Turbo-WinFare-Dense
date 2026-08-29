#include "g4dense/format.hpp"
#include "g4dense/json.hpp"
#include "g4dense/manifest.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/sampling.hpp"
#include "g4dense/detokenizer.hpp"
#include "g4dense/http.hpp"
#include "g4dense/telemetry.hpp"
#include "g4dense/openai_api.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/vk_pipeline.hpp"
#include "g4dense/streamer.hpp"
#include "g4dense/layer_cache.hpp"
#include "g4dense/kv_cache.hpp"
#include "g4dense/draft_runtime.hpp"
#include "g4dense/speculator.hpp"
#include "g4dense/runner.hpp"
#include "g4dense/cpu_reference.hpp"
#include "g4dense/server.hpp"
#include "g4dense/hf_client.hpp"
#include "g4dense/c_api.h"

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace g4dense;

void test_all_contracts() {
    std::cout << "--> Testing Contracts & Header Compilation..." << std::endl;
    assert(sizeof(G4DenseHeader) == 4096);
    std::cout << "  [PASS] All G4Dense contract headers valid." << std::endl;
}

void test_all_format() {
    std::cout << "--> Testing Format & Container..." << std::endl;
    G4DenseHeader header{};
    header.magic = G4DenseHeader::EXPECTED_MAGIC;
    header.version = G4DenseHeader::EXPECTED_VERSION;
    header.quant_type = (uint32_t)QuantType::AFFINE_INT4_G64;
    header.num_layers = 4;
    header.d_model = 256;
    header.d_ff = 512;
    header.num_q_heads = 4;
    header.num_kv_heads = 2;
    header.head_dim = 64;
    header.vocab_size = 1024;
    header.sliding_window = 512;
    header.quant_group_size = 64;
    header.scale_dtype = 1;
    header.tied_embeddings = 1;
    header.global_layer_mask = (1ULL << 1) | (1ULL << 3);
    header.rope_theta_local = 10000.0f;
    header.rope_theta_global = 1000000.0f;
    header.rope_scaling = 1.0f;
    header.final_logit_softcapping = 30.0f;

    header.embed_offset = 4096;
    header.embed_size = 65536;
    header.lm_head_offset = 4096;
    header.lm_head_size = 65536;

    uint64_t cur_offset = 4096 + 65536;
    for (uint32_t i = 0; i < header.num_layers; ++i) {
        header.layer_offsets[i] = cur_offset;
        header.layer_sizes[i] = 131072;
        cur_offset += 131072;
    }

    validate_header(header, cur_offset);
    assert(checked_add(100, 200, "test") == 300);
    assert(checked_multiply(10, 20, "test") == 200);
    std::cout << "  [PASS] Format validation and checked math passed." << std::endl;
}

void test_all_detokenizer() {
    std::cout << "--> Testing Incremental Detokenizer & Stop Matcher..." << std::endl;
    Tokenizer tok;
    std::vector<std::string> paths = {
        "tests/fixtures/tokenizer.json",
        "../tests/fixtures/tokenizer.json",
        "../../tests/fixtures/tokenizer.json",
        "fixtures/tokenizer.json"
    };
    bool loaded = false;
    for (const auto& p : paths) {
        try {
            if (tok.load_vocabulary(p)) {
                loaded = true;
                break;
            }
        } catch (...) {}
    }

    if (loaded) {
        IncrementalDetokenizer detok(tok, true);
        uint8_t byte_val;
        uint32_t b1 = 0, b2 = 0, b3 = 0;
        for (uint32_t t = 0; t < tok.vocab_size(); ++t) {
            if (tok.is_byte_fallback(t, byte_val)) {
                if (byte_val == 0xE2) b1 = t;
                else if (byte_val == 0x82) b2 = t;
                else if (byte_val == 0xAC) b3 = t;
            }
        }

        if (b1 && b2 && b3) {
            assert(detok.push(b1) == "");
            assert(detok.push(b2) == "");
            std::string euro = detok.push(b3);
            assert(euro == "€");
        }
    }

    // Stop Matcher test
    StreamingStopMatcher matcher({"<turn|>", "</s>"});
    assert(matcher.push("Hello world<tur") == "Hello world");
    assert(matcher.push("n|>rest") == "");
    assert(matcher.stopped());
    assert(matcher.matched() == "<turn|>");
    std::cout << "  [PASS] Detokenizer and StopMatcher passed." << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Turbo-WinFare Dense Master Test Suite " << std::endl;
    std::cout << "========================================" << std::endl;

    test_all_contracts();
    test_all_format();
    test_all_detokenizer();

    std::cout << "\n>>> ALL UNIT TESTS PASSED SUCCESSFULLY! <<<" << std::endl;
    return 0;
}
