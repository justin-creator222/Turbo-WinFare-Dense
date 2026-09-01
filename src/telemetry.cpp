#include "g4dense/telemetry.hpp"
#include <string>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <windows.h>
#include <psapi.h>

namespace g4dense {

std::string TelemetrySnapshot::to_json_string() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "{\n";
    ss << "  \"tps\": " << tps << ",\n";
    ss << "  \"ttft_ms\": " << ttft_ms << ",\n";
    ss << "  \"speculative_acceptance_rate\": " << (speculative_acceptance_rate * 100.0) << ",\n";
    ss << "  \"nvme_read_gbs\": " << nvme_read_gbs << ",\n";
    ss << "  \"ram_footprint_mb\": " << ram_footprint_mb << ",\n";
    ss << "  \"ram_total_mb\": " << ram_total_mb << ",\n";
    ss << "  \"apu_power_watts\": " << apu_power_watts << ",\n";
    ss << "  \"active_tier_id\": " << active_tier_id << ",\n";
    ss << "  \"pinned_layers\": " << pinned_layers << ",\n";
    ss << "  \"streamed_layers\": " << streamed_layers << ",\n";
    ss << "  \"context_tokens\": " << context_tokens << ",\n";
    ss << "  \"max_context\": " << max_context << ",\n";
    ss << "  \"gpu_name\": \"" << (gpu_name.empty() ? "Radeon 780M" : gpu_name) << "\",\n";
    ss << "  \"heap0_usage_mb\": " << heap0_usage_mb << ",\n";
    ss << "  \"heap1_usage_mb\": " << heap1_usage_mb << ",\n";
    ss << "  \"heap0_budget_mb\": " << heap0_budget_mb << ",\n";
    ss << "  \"heap1_budget_mb\": " << heap1_budget_mb << ",\n";
    ss << "  \"has_draft\": " << (has_draft ? "true" : "false") << ",\n";
    ss << "  \"draft_k\": " << draft_k << ",\n";
    ss << "  \"speculative_drafted\": " << speculative_drafted << ",\n";
    ss << "  \"speculative_accepted\": " << speculative_accepted << ",\n";
    ss << "  \"drafting_gated\": " << (drafting_gated ? "true" : "false") << ",\n";
    ss << "  \"total_tokens_generated\": " << total_tokens_generated << ",\n";
    ss << "  \"ram_available_mb\": " << ram_available_mb << ",\n";
    ss << "  \"kv_cache_mb\": " << kv_cache_mb << ",\n";
    ss << "  \"lm_head_mb\": " << lm_head_mb << ",\n";
    ss << "  \"layer_mb\": " << layer_mb << ",\n";
    ss << "  \"total_layers\": " << total_layers << ",\n";
    ss << "  \"stream_slots\": " << stream_slots << ",\n";
    ss << "  \"breakdown\": {\n";
    ss << "    \"stream_io_ms\": " << stream_io_ms << ",\n";
    ss << "    \"gpu_wait_ms\": " << gpu_wait_ms << ",\n";
    ss << "    \"draft_ms\": " << draft_ms << ",\n";
    ss << "    \"verify_ms\": " << verify_ms << ",\n";
    ss << "    \"lm_head_ms\": " << lm_head_ms << "\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

namespace {
std::mutex g_telemetry_mutex;
TelemetrySnapshot g_snapshot;
uint64_t g_total_draft = 0;
uint64_t g_total_accepted = 0;
}

TelemetryCollector& TelemetryCollector::instance() {
    static TelemetryCollector inst;
    return inst;
}

void TelemetryCollector::record_ttft(double ttft_ms) {
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    g_snapshot.ttft_ms = ttft_ms;
}

void TelemetryCollector::record_generation_step(double step_time_ms, uint32_t tokens_produced,
                                               uint32_t draft_tokens, uint32_t accepted_tokens) {
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    if (step_time_ms > 0.0 && tokens_produced > 0) {
        double current_tps = (tokens_produced / (step_time_ms / 1000.0));
        // Exponential moving average
        g_snapshot.tps = (g_snapshot.tps == 0.0) ? current_tps : (0.8 * g_snapshot.tps + 0.2 * current_tps);
    }
    g_snapshot.total_tokens_generated += tokens_produced;
    g_total_draft += draft_tokens;
    g_total_accepted += accepted_tokens;
    g_snapshot.speculative_drafted = g_total_draft;
    g_snapshot.speculative_accepted = g_total_accepted;
    if (g_total_draft > 0) {
        g_snapshot.speculative_acceptance_rate = static_cast<double>(g_total_accepted) / g_total_draft;
    }
}

void TelemetryCollector::record_model_state(uint32_t resident_layers, uint32_t streamed_layers,
                                            uint32_t max_context, const std::string& gpu_name,
                                            bool has_draft, uint32_t draft_k) {
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    g_snapshot.pinned_layers = resident_layers;
    g_snapshot.streamed_layers = streamed_layers;
    g_snapshot.max_context = max_context;
    if (!gpu_name.empty()) g_snapshot.gpu_name = gpu_name;
    g_snapshot.has_draft = has_draft;
    g_snapshot.draft_k = draft_k;
}

void TelemetryCollector::record_model_geometry(uint32_t total_layers, double layer_mb,
                                              double lm_head_mb, double kv_cache_mb,
                                              uint32_t stream_slots) {
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    g_snapshot.total_layers = total_layers;
    g_snapshot.layer_mb = layer_mb;
    g_snapshot.lm_head_mb = lm_head_mb;
    g_snapshot.kv_cache_mb = kv_cache_mb;
    g_snapshot.stream_slots = stream_slots;
}


void TelemetryCollector::record_drafting_gated(bool gated) {
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    g_snapshot.drafting_gated = gated;
}
 
void TelemetryCollector::record_draft_k(uint32_t draft_k) {
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    g_snapshot.draft_k = draft_k;
}

void TelemetryCollector::record_heap_usage(double heap0_used_mb, double heap0_budget_mb,
                                           double heap1_used_mb, double heap1_budget_mb) {
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    g_snapshot.heap0_usage_mb = heap0_used_mb;
    g_snapshot.heap0_budget_mb = heap0_budget_mb;
    g_snapshot.heap1_usage_mb = heap1_used_mb;
    g_snapshot.heap1_budget_mb = heap1_budget_mb;
}

void TelemetryCollector::record_phase_breakdown(double stream_io_ms, double gpu_wait_ms,
                                                double lm_head_ms, double cpu_other_ms) {
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    auto ema = [](double prev, double next) {
        return (prev == 0.0) ? next : (0.8 * prev + 0.2 * next);
    };
    g_snapshot.stream_io_ms = ema(g_snapshot.stream_io_ms, stream_io_ms);
    g_snapshot.gpu_wait_ms  = ema(g_snapshot.gpu_wait_ms, gpu_wait_ms);
    g_snapshot.lm_head_ms   = ema(g_snapshot.lm_head_ms, lm_head_ms);
    g_snapshot.cpu_other_ms = ema(g_snapshot.cpu_other_ms, cpu_other_ms);
}

void TelemetryCollector::record_io_throughput(double bytes_read, double io_time_sec) {
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    if (io_time_sec > 0.0) {
        g_snapshot.nvme_read_gbs = (bytes_read / (1024.0 * 1024.0 * 1024.0)) / io_time_sec;
    }
}

TelemetrySnapshot TelemetryCollector::snapshot() const {
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    TelemetrySnapshot s = g_snapshot;

    // Get current process RAM usage
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        s.ram_footprint_mb = static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }

    MEMORYSTATUSEX mem_status{};
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        s.ram_total_mb = static_cast<double>(mem_status.ullTotalPhys) / (1024.0 * 1024.0);
        s.ram_available_mb = static_cast<double>(mem_status.ullAvailPhys) / (1024.0 * 1024.0);
    }

    return s;
}

void TelemetryCollector::reset_speculative_stats() {
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    g_total_draft = 0;
    g_total_accepted = 0;
    g_snapshot.speculative_drafted = 0;
    g_snapshot.speculative_accepted = 0;
    g_snapshot.speculative_acceptance_rate = 0.0;
    g_snapshot.drafting_gated = false;
}

void TelemetryCollector::reset() {
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    g_snapshot = TelemetrySnapshot{};
    g_total_draft = 0;
    g_total_accepted = 0;
}

} // namespace g4dense
