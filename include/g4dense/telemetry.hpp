#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace g4dense {

struct TelemetrySnapshot {
    double tps{0.0};
    double ttft_ms{0.0};
    double speculative_acceptance_rate{0.0};
    // The rate's denominator, published because it is SMALL. A 24-token generation runs a
    // handful of verify rounds, so this is typically 15-35 drafts: "46.7%" is 7 of 15, not a
    // stable percentage, and reading it as one is how a 76.2% single sample became a phantom
    // regression to chase.
    uint64_t speculative_drafted{0};
    uint64_t speculative_accepted{0};
    // True once the adaptive gate has stopped drafting for this generation. Distinguishes "the
    // drafter is loaded and idle" from "the drafter is loaded and losing", which the GUI
    // otherwise shows identically.
    bool drafting_gated{false};

    // Sizes the GUI would otherwise have to hardcode. It did: a 24 GB RAM constant from a
    // different BIOS setting, and a per-layer size that no longer matched the container.
    double kv_cache_mb{0.0};
    double lm_head_mb{0.0};
    double layer_mb{0.0};
    double ram_available_mb{0.0};
    uint32_t total_layers{0};
    uint32_t stream_slots{0};
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
    uint32_t draft_k{8};

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
    // Static model/device facts the web UI displays. Without these the snapshot reported
    // struct defaults -- 0 resident layers, 0 streamed, and a max_context of 8192 when the
    // engine actually caps at 4096, which is worse than reporting nothing.
    void record_model_state(uint32_t resident_layers, uint32_t streamed_layers,
                            uint32_t max_context, const std::string& gpu_name,
                            bool has_draft, uint32_t draft_k);

    // Static geometry, published once at load so the GUI stops guessing it.
    void record_model_geometry(uint32_t total_layers, double layer_mb, double lm_head_mb,
                               double kv_cache_mb, uint32_t stream_slots);

    // Reported separately from record_model_state: at load time no request has run, so the
    // K the verify loop will use is not known yet. Without this the endpoint showed
    // draft_k 0 beside has_draft true and a non-zero acceptance rate.
    void record_draft_k(uint32_t draft_k);
    void record_drafting_gated(bool gated);
    void record_heap_usage(double heap0_used_mb, double heap0_budget_mb,
                           double heap1_used_mb, double heap1_budget_mb);

    void record_phase_breakdown(double stream_io_ms, double gpu_wait_ms,
                                double lm_head_ms, double cpu_other_ms);

    TelemetrySnapshot snapshot() const;
    void reset();

    // Clears ONLY the speculative counters. reset() wipes the whole snapshot, including the
    // model state published once at load time (resident layers, gpu_name, max_context), so it
    // cannot be used per generation without blanking the fields the UI reads.
    void reset_speculative_stats();

private:
    TelemetryCollector() = default;
};

} // namespace g4dense
