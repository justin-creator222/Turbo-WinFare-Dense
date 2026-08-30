#pragma once

#include "g4dense/format.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/sampling.hpp"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>

namespace g4dense {

struct CpuReferenceConfig {
    std::string container_path;
    std::string dump_tensors_dir; // Optional directory to dump per-stage FP32 tensors
    bool verbose{false};
};

class CpuReferenceRunner {
public:
    explicit CpuReferenceRunner(const CpuReferenceConfig& config,
                                std::shared_ptr<Tokenizer> tokenizer);
    ~CpuReferenceRunner();

    void initialize();

    // Runs pure CPU scalar FP32 reference inference
    void generate(const std::string& prompt,
                  int max_tokens,
                  const SamplingParams& sampling,
                  std::function<bool(uint32_t token, const std::string& piece)> on_token,
                  std::atomic<bool>* cancel_flag = nullptr);

    // Single-step forward pass returning full vocabulary logits
    std::vector<float> forward_single_token(uint32_t token, uint32_t position, bool dump_tensors = false);

private:
    CpuReferenceConfig config_;
    std::shared_ptr<Tokenizer> tokenizer_;
    G4DenseHeader header_{};

    // Memory-mapped container buffer for zero-copy 64-bit safe reads
    const uint8_t* mapped_data_{nullptr};
    size_t mapped_size_{0};
    void* mapping_handle_{nullptr};
    void* file_handle_{nullptr};

    // CPU KV cache
    std::vector<std::vector<float>> k_cache_; // per layer
    std::vector<std::vector<float>> v_cache_; // per layer
};

} // namespace g4dense
