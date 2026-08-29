#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>
#include <array>

namespace g4dense {

enum class QuantType : uint32_t {
    AFFINE_INT4_G64 = 1,
    QAT_W4A16       = 2,
    FP16            = 3
};

enum class DType : uint8_t {
    U32  = 0,
    BF16 = 1,
    FP16 = 2,
    FP32 = 3
};

#pragma pack(push, 1)
struct G4DenseHeader {
    static constexpr uint32_t EXPECTED_MAGIC = 0x4734444E; // 'G4DN'
    static constexpr uint32_t EXPECTED_VERSION = 2;
    static constexpr uint64_t ALIGNMENT_BYTES = 4096;
    static constexpr size_t MAX_LAYERS = 60;

    uint32_t magic;                 // 0x4734444E ('G4DN')
    uint32_t version;               // 2
    uint32_t quant_type;            // QuantType enum
    uint32_t num_layers;            // 60
    uint32_t d_model;               // 5376
    uint32_t d_ff;                  // 21504
    uint32_t num_q_heads;           // 32
    uint32_t num_kv_heads;          // 16
    uint32_t head_dim;              // 256
    uint32_t vocab_size;            // 262144
    uint32_t sliding_window;        // 1024 (or 512)
    uint32_t quant_group_size;      // 64
    uint32_t scale_dtype;           // 1 = BF16, 2 = FP16
    uint32_t tied_embeddings;       // 1 = true
    uint64_t global_layer_mask;     // Bitmask of 10 global attention layers
    float    rope_theta_local;      // 10000.0f
    float    rope_theta_global;     // 1000000.0f
    float    rope_scaling;          // 1.0f
    float    final_logit_softcapping; // 30.0f

    uint64_t embed_offset;          // Offset in bytes (4096-aligned)
    uint64_t embed_size;            // Size in bytes
    uint64_t lm_head_offset;        // Offset in bytes
    uint64_t lm_head_size;          // Size in bytes

    uint64_t layer_offsets[MAX_LAYERS]; // Offsets for blocks 0..59
    uint64_t layer_sizes[MAX_LAYERS];   // Sizes for blocks 0..59

    uint8_t  payload_sha256[32];    // SHA-256 hash of entire payload post-header

    uint8_t  reserved[2992];        // Padding to exactly 4096 bytes
};
#pragma pack(pop)

static_assert(sizeof(G4DenseHeader) == 4096, "G4DenseHeader must be exactly 4096 bytes");

class G4DenseFormatError : public std::runtime_error {
public:
    explicit G4DenseFormatError(const std::string& message)
        : std::runtime_error(message) {}
};

inline uint64_t checked_add(uint64_t a, uint64_t b, std::string_view field) {
    uint64_t res = a + b;
    if (res < a) {
        throw G4DenseFormatError(std::string(field) + ": arithmetic overflow");
    }
    return res;
}

inline uint64_t checked_multiply(uint64_t a, uint64_t b, std::string_view field) {
    if (a == 0 || b == 0) return 0;
    uint64_t res = a * b;
    if (res / a != b) {
        throw G4DenseFormatError(std::string(field) + ": arithmetic overflow");
    }
    return res;
}

class PathValidator {
public:
    static bool validate_relative_path(std::string_view path, std::string_view field) {
        if (path.empty() || path.front() == '/' || path.front() == '\\') {
            throw G4DenseFormatError(std::string(field) + ": unsafe relative path");
        }
        if (path.find('\0') != std::string_view::npos) {
            throw G4DenseFormatError(std::string(field) + ": null character in path");
        }
        if (path.find("..") != std::string_view::npos) {
            throw G4DenseFormatError(std::string(field) + ": non-canonical path containing '..'");
        }
        return true;
    }

    static bool validate_basename(std::string_view name, std::string_view field) {
        validate_relative_path(name, field);
        if (name.find('/') != std::string_view::npos || name.find('\\') != std::string_view::npos) {
            throw G4DenseFormatError(std::string(field) + ": expected basename without directory separators");
        }
        return true;
    }
};

// Validates header structure, bounds, alignments, and internal consistency
void validate_header(const G4DenseHeader& header, uint64_t file_size = 0);

} // namespace g4dense
