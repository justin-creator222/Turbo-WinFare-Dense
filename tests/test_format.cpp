#include "g4dense/format.hpp"
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
#include <fstream>
#include <cassert>
#include <cstring>

using namespace g4dense;

int main() {
    std::cout << "[test_format] starting format unit tests..." << std::endl;

    // Test 1: Header size must be exactly 4096 bytes
    assert(sizeof(G4DenseHeader) == 4096);

    // Test 2: Valid synthetic header round-trip
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
    assert(model_architecture(header) == ModelArch::GEMMA4);
    assert(!is_muse_glimmer(header));

    // Test 2b: Muse-Glimmer architecture flags
    {
        G4DenseHeader muse_hdr = header;
        muse_hdr.arch_type = static_cast<uint32_t>(ModelArch::MUSE_GLIMMER);
        muse_hdr.attn_out_dim = 4096;
        muse_hdr.qk_scale_factor = 3.87f;
        muse_hdr.output_multiplier = 0.196116f;
        validate_header(muse_hdr, cur_offset);
        assert(is_muse_glimmer(muse_hdr));
        assert(model_architecture(muse_hdr) == ModelArch::MUSE_GLIMMER);
        assert(model_attn_out_dim(muse_hdr) == 4096);
    }

    std::cout << "[test_format] valid header validation passed." << std::endl;

    // Test 3: Bad magic rejection
    {
        G4DenseHeader bad_magic = header;
        bad_magic.magic = 0xDEADBEEF;
        bool caught = false;
        try {
            validate_header(bad_magic);
        } catch (const G4DenseFormatError& e) {
            caught = true;
        }
        assert(caught);
    }

    // Test 4: Bad alignment rejection
    {
        G4DenseHeader bad_align = header;
        bad_align.layer_offsets[0] = 4097; // not 4096 aligned
        bool caught = false;
        try {
            validate_header(bad_align);
        } catch (const G4DenseFormatError& e) {
            caught = true;
        }
        assert(caught);
    }

    // Test 5: Out of bounds file size rejection
    {
        bool caught = false;
        try {
            validate_header(header, cur_offset - 100);
        } catch (const G4DenseFormatError& e) {
            caught = true;
        }
        assert(caught);
    }

    // Test 6: Checked arithmetic
    assert(checked_add(100, 200, "test") == 300);
    assert(checked_multiply(10, 20, "test") == 200);
    bool caught_overflow = false;
    try {
        checked_add(UINT64_MAX, 1, "overflow");
    } catch (const G4DenseFormatError&) {
        caught_overflow = true;
    }
    assert(caught_overflow);

    // Test 7: Verify G4MtpHeader size and validation
    assert(sizeof(G4MtpHeader) == 4096);
    {
        G4MtpHeader mtp{};
        mtp.magic = G4MtpHeader::EXPECTED_MAGIC;
        mtp.version = G4MtpHeader::EXPECTED_VERSION;
        mtp.quant_type = 1;
        mtp.num_layers = 4;
        mtp.backbone_hidden_size = 5376;
        mtp.hidden_size = 1024;
        mtp.intermediate_size = 8192;
        mtp.vocab_size = 262144;
        mtp.embed_offset = 4096;
        mtp.pre_proj_offset = 8192;
        mtp.post_proj_offset = 12288;
        mtp.norm_offset = 16384;
        for (uint32_t i = 0; i < 4; ++i) {
            mtp.layer_offsets[i] = 20480 + i * 4096;
            mtp.layer_sizes[i] = 4096;
        }
        validate_mtp_header(mtp, 20480 + 4 * 4096);
        std::cout << "[test_format] synthetic G4MtpHeader validation passed." << std::endl;

        // Verify real file if present
        std::ifstream real_mtp("models/gemma-4-31b-assistant.g4mtp", std::ios::binary | std::ios::ate);
        if (real_mtp.is_open()) {
            uint64_t fsize = static_cast<uint64_t>(real_mtp.tellg());
            real_mtp.seekg(0);
            G4MtpHeader real_hdr{};
            real_mtp.read(reinterpret_cast<char*>(&real_hdr), sizeof(real_hdr));
            validate_mtp_header(real_hdr, fsize);
            std::cout << "[test_format] real models/gemma-4-31b-assistant.g4mtp validated ("
                      << (fsize / (1024*1024)) << " MiB)." << std::endl;
        }
    }

    // Test 8: Verify all contract headers compile and link
    std::cout << "[test_format] contract header compilation verified." << std::endl;

    std::cout << "[test_format] ALL FORMAT & CONTRACT TESTS PASSED." << std::endl;
    return 0;
}

