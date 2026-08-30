#pragma once

#include "g4dense/format.hpp"
#include <string>
#include <vector>
#include <map>
#include <optional>

namespace g4dense {

struct ManifestArch {
    int hidden_size{5376};
    int ffn_intermediate{21504};
    int num_heads{32};
    int num_kv_heads{16};
    int head_dim{256};
    int global_head_dim{512};
    int vocab_size{262144};
    int sliding_window{1024};
    double final_logit_softcap{30.0};
    double rope_theta{10000.0};
    double full_rope_theta{1000000.0};
    double partial_rotary_factor{0.25};
    double rms_norm_eps{1e-6};
    int num_layers{60};
    bool tie_word_embeddings{true};
    bool attention_k_eq_v{true};
    std::string hidden_activation{"gelu_pytorch_tanh"};
    std::vector<int> full_attention_layer_mask; // 10 entries: layers 5, 11, 17, 23, 29, 35, 41, 47, 53, 59
};

struct ManifestQuantSlot {
    int weight_bits{4};
    std::string scheme{"mlx_affine"};
    std::string scale_type{"BF16"};
    std::string bias_type{"BF16"};
    int group_size{64};
};

struct ManifestQuant {
    ManifestQuantSlot embedding;
    ManifestQuantSlot attention;
    ManifestQuantSlot ffn;
};

struct G4DenseManifest {
    std::string magic{"G4DN"};
    int version{2};
    std::map<std::string, bool> flags;
    std::string model_id;
    std::optional<std::string> source_snapshot_hash;
    ManifestArch arch;
    std::optional<ManifestQuant> quant;
    std::string container_file;
    uint64_t container_size{0};
    std::string container_sha256;

    void validate() const;
    std::string to_json_string() const;
    static G4DenseManifest from_json_string(const std::string& json_str);
    static G4DenseManifest from_header(const G4DenseHeader& header, const std::string& model_id = "gemma-4-31b-dense");
};

// Bundle and resource path resolution utilities
std::string read_text_file(const std::string& path);
bool bundle_loads(const std::string& path);
std::string resolve_bundle_path(const std::string& name_or_path);
std::string resolve_resource_path(const std::string& name_or_path);
std::vector<std::string> bundle_search_roots();

} // namespace g4dense
