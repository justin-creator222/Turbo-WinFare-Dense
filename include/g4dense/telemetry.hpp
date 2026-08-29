#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace g4dense {

struct TelemetrySnapshot {
    double tps{0.0};
    double ttft_ms{0.0};
    double speculative_acceptance_rate{0.0};
    double nvme_read_gbs{0.0};
    double ram_footprint_mb{0.0};
    double ram_total_mb{0.0};
    double apu_power_watts{0.0}; // null / 0 if not exposed by platform

    uint32_t active_tier_id{1};
    uint32_t pinned_layers{0};
    uint32_t streamed_layers{0};
    uint32_t context_tokens{0};
    uint32_t max_context{8192};

    uint64_t total_tokens_generated{0};
    uint64_t total_speculative_passes{0};

    // Per-phase breakdown in ms
    double stream_io_ms{0.0};
    double gpu_wait_ms{0.0};
    double draft_ms{0.0};
    double verify_ms{0.0};
    double lm_head_ms{0.0};
    double cpu_other_ms{0.0};

    std::string to_json_string() const;
};

class TelemetryCollector {
public:
    static TelemetryCollector& instance();

    void record_generation_step(double step_time_ms, uint32_t tokens_produced,
                                uint32_t draft_tokens, uint32_t accepted_tokens);
    void record_io_throughput(double bytes_read, double io_time_sec);
    void record_ttft(double ttft_ms);

    TelemetrySnapshot snapshot() const;
    void reset();

private:
    TelemetryCollector() = default;
};

} // namespace g4dense
