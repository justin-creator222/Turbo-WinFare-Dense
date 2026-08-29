#pragma once

#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <cstdint>

namespace g4dense {

struct DownloadProgress {
    std::string filename;
    uint64_t bytes_downloaded{0};
    uint64_t total_bytes{0};
    double download_speed_mbps{0.0};
    double eta_seconds{0.0};
    double percent_complete{0.0};
    std::string phase; // "downloading", "verifying_sha256", "converting", "complete", "error"
    std::string error_message;
};

using ProgressCallback = std::function<void(const DownloadProgress& progress)>;

class HFClient {
public:
    explicit HFClient(const std::string& auth_token = "");
    ~HFClient() = default;

    // Resolves repo file tree and total byte sizes from Hugging Face API
    bool resolve_repo(const std::string& repo_id, std::vector<std::string>& out_files, uint64_t& out_total_bytes);

    // Downloads file with chunked resume support and streaming SHA-256 check
    bool download_file(const std::string& repo_id,
                       const std::string& rfilename,
                       const std::string& destination_path,
                       const std::string& expected_sha256 = "",
                       ProgressCallback on_progress = nullptr,
                       std::atomic<bool>* cancel_flag = nullptr);

    // Download entire model checkpoint directory
    bool download_model(const std::string& repo_id,
                        const std::string& output_dir,
                        ProgressCallback on_progress = nullptr,
                        std::atomic<bool>* cancel_flag = nullptr);

private:
    std::string auth_token_;
};

} // namespace g4dense
