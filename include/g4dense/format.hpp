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
    // 3: packed-weight blocks are 16-byte aligned within each layer (see runner's setup_proj).
    //
    // This one MUST be a version bump. Every previous field addition was backward compatible --
    // zeroed fields read as "unspecified" -- but a layout change is not: a version-2 container
    // read by this engine decodes at the wrong offsets and produces plausible-looking garbage
    // rather than failing. Rejecting it outright is the only safe behaviour.
    static constexpr uint32_t EXPECTED_VERSION = 3;
    static constexpr uint64_t ALIGNMENT_BYTES = 4096;
    static constexpr size_t MAX_LAYERS = 60;

    uint32_t magic;                 // 0x4734444E ('G4DN')
    uint32_t version;               // EXPECTED_VERSION
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

    // Full-attention geometry. Gemma 4 gives its full-attention layers a different head_dim
    // and kv-head count from its sliding layers (512 / 4 vs 256 / 16 on the 31B), so the three
    // fields above describe only the sliding layers. These were hardcoded as 31B literals in
    // the runner and the CPU reference, which meant the engine could not run any other model.
    //
    // Placed at the head of `reserved` rather than beside `head_dim` so that every offset in
    // this struct is unchanged: containers written before these existed have them zeroed, and
    // zero is read as "not specified" (see resolve_layer_geometry). The struct stays 4096
    // bytes and the version was not bumped for them.
    uint32_t global_head_dim;       // 512 on the 31B; 0 = unspecified (legacy container)
    uint32_t global_kv_heads;       // 4 on the 31B;   0 = unspecified (legacy container)

    // Per-layer embeddings (PLE) and KV sharing -- the E2B/E4B architecture.
    //
    // Zero on every model that does not use them, and the per-layer block layout is then
    // byte-for-byte what it was, so those additions needed no version bump.
    // `ple_dim != 0` is the discriminator.
    uint32_t ple_dim;               // hidden_size_per_layer_input (256 on E2B), 0 = no PLE
    uint32_t ple_vocab;             // vocab_size_per_layer_input
    uint32_t num_kv_shared_layers;  // trailing layers that reuse an earlier layer's K/V
    uint32_t ple_reserved;          // padding, keeps the block 8-byte aligned
    uint64_t ple_offset;            // model-level PLE block: embed_tokens_per_layer,
    uint64_t ple_size;              // per_layer_model_projection, per_layer_projection_norm

    // Feed-forward width per layer.
    //
    // Not derivable from `d_ff`: E2B's config says intermediate_size = 6144, but its last 20
    // layers -- exactly the KV-sharing ones -- have 12288, and nothing in config.json states
    // that. Upstream carries it in per_layer_config. Recording the measured width per layer is
    // the only way to read such a container back correctly. Zero means "use d_ff", so existing
    // containers keep working.
    uint32_t layer_d_ff[MAX_LAYERS];

    uint8_t  reserved[2712];        // Padding to exactly 4096 bytes
};
#pragma pack(pop)

static_assert(sizeof(G4DenseHeader) == 4096, "G4DenseHeader must be exactly 4096 bytes");

// This layer's feed-forward width. Falls back to the model-wide d_ff for containers written
// before the per-layer array existed.
inline uint32_t layer_ffn_width(const G4DenseHeader& h, uint32_t layer_idx) {
    if (layer_idx < G4DenseHeader::MAX_LAYERS && h.layer_d_ff[layer_idx] != 0) {
        return h.layer_d_ff[layer_idx];
    }
    return h.d_ff;
}

// Per-layer attention geometry. `head_dim`/`num_kv_heads` in the header describe the sliding
// layers; full-attention layers carry their own. Both the GPU runner and the CPU reference go
// through here so they cannot drift apart.
struct LayerGeometry {
    uint32_t head_dim;
    uint32_t q_heads;
    uint32_t kv_heads;
};

// True when this layer reuses an earlier layer's K/V instead of projecting its own. Upstream:
// `layer_idx >= num_hidden_layers - num_kv_shared_layers`. Such a layer carries no k_proj,
// v_proj or k_norm at all -- E2B has 15 of each for 35 layers.
inline bool is_kv_shared_layer(const G4DenseHeader& h, uint32_t layer_idx) {
    if (h.num_kv_shared_layers == 0 || h.num_kv_shared_layers > h.num_layers) return false;
    return layer_idx >= (h.num_layers - h.num_kv_shared_layers);
}

// The layer a shared layer takes its K/V from: the LAST non-shared layer of the SAME attention
// type. Donor type always matches, so a sliding layer never reads a full-attention cache and
// the cache geometry lines up without any copying. Returns layer_idx itself when not shared.
inline uint32_t kv_donor_layer(const G4DenseHeader& h, uint32_t layer_idx) {
    if (!is_kv_shared_layer(h, layer_idx)) return layer_idx;
    const bool want_global = ((h.global_layer_mask >> layer_idx) & 1ull) != 0;
    const uint32_t first_shared = h.num_layers - h.num_kv_shared_layers;
    for (uint32_t l = first_shared; l-- > 0;) {
        const bool is_global = ((h.global_layer_mask >> l) & 1ull) != 0;
        if (is_global == want_global) return l;
    }
    return 0;
}

inline LayerGeometry resolve_layer_geometry(const G4DenseHeader& h, uint32_t layer_idx) {
    const bool is_global = (h.global_layer_mask >> layer_idx) & 1ull;
    LayerGeometry g{h.head_dim, h.num_q_heads, h.num_kv_heads};
    if (is_global) {
        // A container written before these fields existed leaves them zero. Falling back to
        // the sliding geometry is wrong for the 31B, so fall back to the 31B's actual values
        // and let the caller notice via validate_header's warning path rather than silently
        // computing nonsense.
        g.head_dim = h.global_head_dim ? h.global_head_dim : 512u;
        g.kv_heads = h.global_kv_heads ? h.global_kv_heads : 4u;
    }
    return g;
}

// True when the container predates the full-attention geometry fields, so the values above are
// assumed rather than read. Callers should warn once.
inline bool has_legacy_global_geometry(const G4DenseHeader& h) {
    return h.global_layer_mask != 0 && (h.global_head_dim == 0 || h.global_kv_heads == 0);
}

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
