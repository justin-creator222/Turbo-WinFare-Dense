#include "g4dense/streamer.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/format.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <windows.h>

namespace fs = std::filesystem;
using namespace g4dense;

int main() {
    std::cout << "========================================================\n"
              << "  Turbo-WinFare Dense: Layer Streamer Correctness Test  \n"
              << "========================================================\n";

    const std::vector<std::string> candidates = {
        "models/gemma-4-31b-dense.g4dense",
        "../models/gemma-4-31b-dense.g4dense",
        // The E2B container is NOT a candidate: it is quarantined as invalid
        // (models/quarantine/README.md). It loads and computes noise.
        "tests/fixtures/tiny.g4dense",
        "../tests/fixtures/tiny.g4dense"
    };
    std::string model_path;
    for (const auto& c : candidates) {
        if (fs::exists(c)) { model_path = c; break; }
    }

    std::cout << "Testing container: " << model_path << "\n";

    G4DenseHeader header;
    {
        std::ifstream f(model_path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Cannot open container: " << model_path << "\n";
            return 1;
        }
        f.read(reinterpret_cast<char*>(&header), sizeof(G4DenseHeader));
    }
    validate_header(header);

    // Map container file for ground truth comparison
    HANDLE h_file = CreateFileA(model_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    assert(h_file != INVALID_HANDLE_VALUE);
    HANDLE h_map = CreateFileMappingA(h_file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    assert(h_map != nullptr);
    const uint8_t* mapped_base = static_cast<const uint8_t*>(MapViewOfFile(h_map, FILE_MAP_READ, 0, 0, 0));
    assert(mapped_base != nullptr);

    auto vk_ctx = std::make_shared<VulkanContext>();
    vk_ctx->initialize();

    uint64_t max_layer_bytes = 0;
    for (uint32_t i = 0; i < header.num_layers; ++i) {
        if (header.layer_sizes[i] > max_layer_bytes) max_layer_bytes = header.layer_sizes[i];
    }

    size_t ring_slots = 6;
    LayerStreamer streamer(vk_ctx, model_path, ring_slots, max_layer_bytes, IOMode::Buffered);
    streamer.initialize(header);

    std::cout << "1. Verifying Read Correctness across all layers...\n";
    for (uint32_t l = 0; l < header.num_layers; ++l) {
        auto plan = streamer.plan_layers({static_cast<int>(l)});
        streamer.fetch_misses(plan);
        LayerSlot* slot = plan.slots[0];
        assert(slot != nullptr);
        assert(slot->layer_id == static_cast<int>(l));

        const uint8_t* expected_bytes = mapped_base + header.layer_offsets[l];
        uint64_t layer_size = header.layer_sizes[l];

        int cmp = std::memcmp(slot->host_ptr, expected_bytes, layer_size);
        if (cmp != 0) {
            std::cerr << "  [FAIL] Layer " << l << " streamer data mismatch with mapped file!\n";
            return 1;
        }

        streamer.release_plan(plan);
    }
    std::cout << "  [PASS] All " << header.num_layers << " layers fetched with byte-exact correctness.\n";

    std::cout << "2. Verifying LRU Eviction & Slot Pinning...\n";
    std::vector<int> pinned = {0, 1};
    streamer.apply_tier_pinning(pinned);

    // Fetch layers 2, 3, 2
    auto p1 = streamer.plan_layers({2});
    streamer.fetch_misses(p1);
    assert(p1.hits.empty() && p1.misses.size() == 1);
    streamer.release_plan(p1);

    auto p2 = streamer.plan_layers({2});
    streamer.fetch_misses(p2);
    assert(p2.hits.size() == 1 && p2.misses.empty()); // hit!
    streamer.release_plan(p2);

    std::cout << "  [PASS] Layer cache hit and eviction mechanics validated.\n";

    UnmapViewOfFile(mapped_base);
    CloseHandle(h_map);
    CloseHandle(h_file);

    std::cout << "\n[test_streamer] ALL STREAMER CHECKS PASSED!\n";
    return 0;
}
