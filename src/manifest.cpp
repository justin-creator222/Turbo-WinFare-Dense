#include "g4dense/manifest.hpp"
#include "g4dense/json.hpp"
#include "g4dense/format.hpp"

#include <sstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#endif

namespace g4dense {

std::string read_text_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw G4DenseFormatError("Cannot open file: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void G4DenseManifest::validate() const {
    if (magic != "G4DN") {
        throw G4DenseFormatError("manifest.magic: expected 'G4DN', got '" + magic + "'");
    }
    if (version != 2) {
        throw G4DenseFormatError("manifest.version: unsupported version: " + std::to_string(version));
    }
    if (arch.num_layers <= 0 || arch.num_layers > (int)G4DenseHeader::MAX_LAYERS) {
        throw G4DenseFormatError("manifest.arch.num_layers: invalid layer count " + std::to_string(arch.num_layers));
    }
    if (arch.hidden_size <= 0 || arch.ffn_intermediate <= 0 || arch.num_heads <= 0 ||
        arch.num_kv_heads <= 0 || arch.vocab_size <= 0) {
        throw G4DenseFormatError("manifest.arch: invalid dimensions");
    }
}

std::string G4DenseManifest::to_json_string() const {
    validate();
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"magic\": \"" << magic << "\",\n";
    ss << "  \"version\": " << version << ",\n";
    ss << "  \"model_id\": \"" << model_id << "\",\n";
    ss << "  \"container_file\": \"" << container_file << "\",\n";
    ss << "  \"container_size\": " << container_size << ",\n";
    ss << "  \"container_sha256\": \"" << container_sha256 << "\",\n";
    ss << "  \"arch\": {\n";
    ss << "    \"hidden_size\": " << arch.hidden_size << ",\n";
    ss << "    \"ffn_intermediate\": " << arch.ffn_intermediate << ",\n";
    ss << "    \"num_layers\": " << arch.num_layers << ",\n";
    ss << "    \"num_heads\": " << arch.num_heads << ",\n";
    ss << "    \"num_kv_heads\": " << arch.num_kv_heads << ",\n";
    ss << "    \"head_dim\": " << arch.head_dim << ",\n";
    ss << "    \"vocab_size\": " << arch.vocab_size << ",\n";
    ss << "    \"sliding_window\": " << arch.sliding_window << ",\n";
    ss << "    \"hidden_activation\": \"" << arch.hidden_activation << "\",\n";
    ss << "    \"final_logit_softcap\": " << arch.final_logit_softcap << "\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

G4DenseManifest G4DenseManifest::from_json_string(const std::string& json_str) {
    JsonValue root = JsonValue::parse(json_str, "manifest.json");
    G4DenseManifest m;
    m.magic = root.at("magic", "manifest").as_string("manifest.magic");
    m.version = root.at("version", "manifest").as_uint32("manifest.version");
    m.model_id = root.at("model_id", "manifest").as_string("manifest.model_id");

    if (root.has("container_file")) {
        m.container_file = root.at("container_file", "manifest").as_string("manifest.container_file");
    }
    if (root.has("container_size")) {
        m.container_size = root.at("container_size", "manifest").as_uint64("manifest.container_size");
    }
    if (root.has("container_sha256")) {
        m.container_sha256 = root.at("container_sha256", "manifest").as_string("manifest.container_sha256");
    }

    const auto& arch = root.at("arch", "manifest");
    m.arch.hidden_size = arch.at("hidden_size", "arch").as_uint32("arch.hidden_size");
    m.arch.ffn_intermediate = arch.at("ffn_intermediate", "arch").as_uint32("arch.ffn_intermediate");
    m.arch.num_layers = arch.at("num_layers", "arch").as_uint32("arch.num_layers");
    m.arch.num_heads = arch.at("num_heads", "arch").as_uint32("arch.num_heads");
    m.arch.num_kv_heads = arch.at("num_kv_heads", "arch").as_uint32("arch.num_kv_heads");
    m.arch.head_dim = arch.at("head_dim", "arch").as_uint32("arch.head_dim");
    m.arch.vocab_size = arch.at("vocab_size", "arch").as_uint32("arch.vocab_size");
    m.arch.sliding_window = arch.at("sliding_window", "arch").as_uint32("arch.sliding_window");

    if (arch.has("hidden_activation")) {
        m.arch.hidden_activation = arch.at("hidden_activation", "arch").as_string("arch.hidden_activation");
    }
    if (arch.has("final_logit_softcap")) {
        m.arch.final_logit_softcap = arch.at("final_logit_softcap", "arch").as_double("arch.final_logit_softcap");
    }

    m.validate();
    return m;
}

G4DenseManifest G4DenseManifest::from_header(const G4DenseHeader& header, const std::string& model_id) {
    validate_header(header);
    G4DenseManifest m;
    m.magic = "G4DN";
    m.version = header.version;
    m.model_id = model_id;
    m.arch.hidden_size = header.d_model;
    m.arch.ffn_intermediate = header.d_ff;
    m.arch.num_layers = header.num_layers;
    m.arch.num_heads = header.num_q_heads;
    m.arch.num_kv_heads = header.num_kv_heads;
    m.arch.head_dim = header.head_dim;
    m.arch.vocab_size = header.vocab_size;
    m.arch.sliding_window = header.sliding_window;
    m.arch.final_logit_softcap = header.final_logit_softcapping;
    m.arch.tie_word_embeddings = (header.tied_embeddings != 0);

    for (int i = 0; i < 60; ++i) {
        if (header.global_layer_mask & (1ULL << i)) {
            m.arch.full_attention_layer_mask.push_back(i);
        }
    }
    return m;
}

bool bundle_loads(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;

    // Check if it's a direct .g4dense file
    if (fs::is_regular_file(path, ec)) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        G4DenseHeader h{};
        f.read((char*)&h, sizeof(h));
        if (f.gcount() != sizeof(h)) return false;
        try {
            validate_header(h, fs::file_size(path, ec));
            return true;
        } catch (...) {
            return false;
        }
    }

    // Check if it's a directory containing manifest.json or a .g4dense file
    if (fs::is_directory(path, ec)) {
        fs::path dir(path);
        fs::path manifest_path = dir / "manifest.json";
        if (fs::exists(manifest_path, ec)) {
            try {
                std::string txt = read_text_file(manifest_path.string());
                auto m = G4DenseManifest::from_json_string(txt);
                return true;
            } catch (...) {
                return false;
            }
        }

        // Search directory for any .g4dense file
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (entry.path().extension() == ".g4dense") {
                if (bundle_loads(entry.path().string())) return true;
            }
        }
    }
    return false;
}

std::vector<std::string> bundle_search_roots() {
    std::vector<std::string> roots;
    roots.push_back(".");
    roots.push_back("..");
    roots.push_back("models");
    roots.push_back("../models");
    roots.push_back("build");
    roots.push_back("tests/fixtures");
    roots.push_back("../tests/fixtures");
    roots.push_back("../../tests/fixtures");

#ifdef _WIN32
    char exe_buf[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, exe_buf, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        std::filesystem::path exe_dir = std::filesystem::path(exe_buf).parent_path();
        roots.push_back(exe_dir.string());
        roots.push_back((exe_dir / "..").string());
        roots.push_back((exe_dir / ".." / "tests" / "fixtures").string());
        roots.push_back((exe_dir / ".." / "models").string());
    }
#endif
    return roots;
}

std::string resolve_resource_path(const std::string& name_or_path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(name_or_path, ec)) {
        return name_or_path;
    }
    for (const auto& root : bundle_search_roots()) {
        fs::path candidate = fs::path(root) / name_or_path;
        if (fs::exists(candidate, ec)) {
            return candidate.string();
        }
    }
    return name_or_path;
}

std::string resolve_bundle_path(const std::string& name_or_path) {
    namespace fs = std::filesystem;
    std::error_code ec;

    // Check directly first
    if (bundle_loads(name_or_path)) {
        return name_or_path;
    }

    // Search search roots
    for (const auto& root : bundle_search_roots()) {
        fs::path candidate = fs::path(root) / name_or_path;
        if (bundle_loads(candidate.string())) {
            return candidate.string();
        }
        // Try appending .g4dense
        fs::path candidate_ext = fs::path(root) / (name_or_path + ".g4dense");
        if (bundle_loads(candidate_ext.string())) {
            return candidate_ext.string();
        }
    }
    return name_or_path;
}

} // namespace g4dense
