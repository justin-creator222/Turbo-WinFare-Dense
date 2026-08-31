#pragma once

#include "g4dense/format.hpp"
#include "g4dense/vk_context.hpp"
#include <windows.h>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <deque>
#include <condition_variable>

namespace g4dense {

enum class IOMode {
    Auto,
    Buffered,
    Unbuffered,
    DirectStorage
};

enum class EvictionPolicy {
    LFU,
    LRU
};

struct LayerSlot {
    int layer_id{-1};
    VkMemoryAllocation buffer; // UMA host-coherent Vulkan memory
    void* host_ptr{nullptr};
    uint64_t last_used_timestamp{0};
    uint32_t frequency{0};

    // Persistent per-slot Win32 event & OVERLAPPED structure. The OVERLAPPED carries the file
    // offset, so slots never share a file pointer and reads on different slots are independent.
    HANDLE event{nullptr};
    OVERLAPPED ov{};
    bool read_pending{false};

    // Set while an I/O worker owns this slot; cleared by the worker on completion. Guarded by
    // LayerStreamer::io_mutex_, not by lock_.
    bool io_pending{false};
    DWORD io_error{0};
    bool io_truncated{false};

    // Slot pinning to prevent self-eviction within an active forward pass
    bool pinned{false};
    bool permanently_pinned{false}; // For resident layers in active Tier
};

class LayerStreamer {
public:
    LayerStreamer(std::shared_ptr<VulkanContext> ctx,
                  const std::string& container_path,
                  size_t slot_count,
                  uint64_t layer_bytes,
                  IOMode io_mode = IOMode::Auto,
                  EvictionPolicy policy = EvictionPolicy::LRU,
                  size_t io_threads = 2);
    ~LayerStreamer();

    void initialize(const G4DenseHeader& header);

    struct LayerPlan {
        std::vector<LayerSlot*> slots;   // one per layer requested
        std::vector<size_t> hits;         // indices in slots requiring no disk I/O
        std::vector<size_t> misses;       // indices in slots awaiting fetch
        LayerStreamer* owner{nullptr};

        bool valid() const { return owner != nullptr; }
    };

    // Resolves cache hits and pins slots without issuing disk reads
    LayerPlan plan_layers(const std::vector<int>& layer_ids);

    // Issues asynchronous DMA reads for cache misses and waits for completion
    // Issue the reads for a plan's misses and return immediately. Pair with await_reads()
    // when the layer is actually needed -- that is what lets I/O overlap GPU work.
    void issue_reads(LayerPlan& plan);
    void await_reads(LayerPlan& plan);

    // issue_reads() followed immediately by await_reads(). Convenient, but it overlaps
    // nothing: callers on the hot path should use the two halves.
    void fetch_misses(LayerPlan& plan);

    // Releases and unpins slots after computation
    void release_plan(LayerPlan& plan);

    // Live re-tiering: permanently pin priority layers
    void apply_tier_pinning(const std::vector<int>& pinned_layers);

    // Invalidates every slot that is not permanently pinned by the active tier, so the next
    // pass re-reads those layers from disk. Permanently pinned (resident) layers are kept --
    // dropping them would silently change the tier the user selected.
    void clear_cache();

    size_t slot_count() const { return slots_.size(); }
    uint64_t layer_bytes() const { return layer_bytes_; }
    uint64_t total_bytes_read() const { return total_bytes_read_.load(); }
    uint64_t total_io_calls() const { return total_io_calls_.load(); }
    uint64_t total_hits() const { return total_hits_.load(); }
    uint64_t total_misses() const { return total_misses_.load(); }
    double hit_rate_pct() const {
        uint64_t h = total_hits_.load();
        uint64_t m = total_misses_.load();
        return (h + m == 0) ? 0.0 : (double)h / (double)(h + m) * 100.0;
    }

    IOMode effective_io_mode() const { return effective_io_mode_; }

private:
    std::shared_ptr<VulkanContext> ctx_;
    std::string container_path_;
    size_t slot_count_;
    uint64_t layer_bytes_;
    IOMode requested_io_mode_;
    IOMode effective_io_mode_{IOMode::Buffered};
    EvictionPolicy policy_;

    G4DenseHeader header_{};
    HANDLE file_handle_{INVALID_HANDLE_VALUE};

    std::vector<std::unique_ptr<LayerSlot>> slots_;
    std::mutex lock_;
    uint64_t global_clock_{0};

    std::atomic<uint64_t> total_bytes_read_{0};
    std::atomic<uint64_t> total_io_calls_{0};
    std::atomic<uint64_t> total_hits_{0};
    std::atomic<uint64_t> total_misses_{0};

    // The read is performed on a worker thread, and the reason is worth stating.
    //
    // Round 6 measured that BUFFERED overlapped reads complete INLINE when the data is already
    // in the page cache, so ReadFile blocked the caller and nothing overlapped -- 956 ms moved
    // out of the I/O phase into "CPU other" without being saved. The conclusion drawn was to
    // use FILE_FLAG_NO_BUFFERING, which does make the read genuinely asynchronous.
    //
    // But that trades away bandwidth to fix a threading problem. Ground truth measures the
    // unbuffered path at 3.08-3.09 GB/s against 7.14 GB/s buffered-warm. Inline completion is
    // only fatal because the read is issued on the thread that submits GPU work; performing it
    // on a worker makes it irrelevant, and keeps the faster path.
    struct IoJob {
        LayerSlot* slot;
        uint64_t offset;
        size_t count;
    };

    std::vector<std::thread> io_threads_;
    std::deque<IoJob> io_queue_;
    std::mutex io_mutex_;
    std::condition_variable io_cv_;       // wakes workers when a job arrives
    std::condition_variable io_done_cv_;  // wakes await_read() on completion
    bool io_stop_{false};
    size_t io_thread_count_{2};

    void io_worker();
    void stop_io_threads();

    LayerSlot* find_or_evict_slot(int layer_id, bool& was_hit);
    void issue_read(LayerSlot* slot, uint64_t file_offset, size_t count);
    void await_read(LayerSlot* slot, size_t count);
};

} // namespace g4dense
