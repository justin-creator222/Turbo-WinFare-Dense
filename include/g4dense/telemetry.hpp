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

    std::string gpu_name{"Radeon 780M"};
    double heap0_usage_mb{0.0};
    double heap1_usage_mb{0.0};
    double heap0_budget_mb{13417.62};
    double heap1_budget_mb{6708.75};
    bool has_draft{false};
    uint32_t draft_k{6};

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

    // Per-forward-pass phase attribution, as an exponential moving average so a single slow
    // pass does not dominate. Without this, "the pass takes 9.6 s" is all we know and every
    // optimization is a guess about which part of it moved.
    void record_phase_breakdown(double stream_io_ms, double gpu_wait_ms,
                                double lm_head_ms, double cpu_other_ms);

    TelemetrySnapshot snapshot() const;
    void reset();

private:
    TelemetryCollector() = default;
};

} // namespace g4dense
