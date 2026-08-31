#include "g4dense/streamer.hpp"
#include "g4dense/format.hpp"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace g4dense {

LayerStreamer::LayerStreamer(std::shared_ptr<VulkanContext> ctx,
                             const std::string& container_path,
                             size_t slot_count,
                             uint64_t layer_bytes,
                             IOMode io_mode,
                             EvictionPolicy policy, size_t io_threads)
    : ctx_(ctx), container_path_(container_path), slot_count_(slot_count),
      layer_bytes_(layer_bytes), requested_io_mode_(io_mode), policy_(policy),
      io_thread_count_(io_threads == 0 ? 1 : io_threads) {}

LayerStreamer::~LayerStreamer() {
    // Workers first: they hold file_handle_ and write into slot->ov, so closing the handle or
    // freeing slots underneath a running read is a use-after-free.
    stop_io_threads();
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
    for (auto& slot : slots_) {
        if (slot->event != nullptr) {
            CloseHandle(slot->event);
            slot->event = nullptr;
        }
        if (ctx_ && slot->buffer.buffer != VK_NULL_HANDLE) {
            ctx_->free_buffer(slot->buffer);
        }
    }
    slots_.clear();
}

void LayerStreamer::initialize(const G4DenseHeader& header) {
    header_ = header;

    // Open file for asynchronous DMA
    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
    if (requested_io_mode_ == IOMode::Unbuffered) {
        flags |= FILE_FLAG_NO_BUFFERING;
        effective_io_mode_ = IOMode::Unbuffered;
    } else {
        effective_io_mode_ = IOMode::Buffered;
    }

    file_handle_ = CreateFileA(
        container_path_.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        flags,
        nullptr
    );

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        // Fallback to buffered if unbuffered failed
        if (effective_io_mode_ == IOMode::Unbuffered) {
            flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
            file_handle_ = CreateFileA(container_path_.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                       nullptr, OPEN_EXISTING, flags, nullptr);
            effective_io_mode_ = IOMode::Buffered;
        }
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            throw G4DenseFormatError("LayerStreamer: cannot open container file " + container_path_);
        }
    }

    // Workers start only once the handle is valid, so a failed open never leaves threads
    // waiting on a file they cannot read.
    io_stop_ = false;
    for (size_t i = 0; i < io_thread_count_; ++i) {
        io_threads_.emplace_back([this] { io_worker(); });
    }

    // Allocate UMA Layer Slots
    slots_.reserve(slot_count_);
    for (size_t i = 0; i < slot_count_; ++i) {
        auto slot = std::make_unique<LayerSlot>();
        // HOST_CACHED, not merely HOST_VISIBLE. These slots are the destination of a ReadFile
        // per streamed layer, and the first type matching HOST_VISIBLE|HOST_COHERENT on this
        // driver is write-combined. Measured on this machine (bench_gpu section 5), reading
        // four real layers:
        //     write-combined  1.95 GB/s
        //     HOST_CACHED     5.97 GB/s
        // The GPU reads both at the same speed (~30 GB/s, bench_gpu section 3), so there is no
        // trade here -- the uncached type was simply the wrong choice.
        slot->buffer = ctx_->allocate_buffer(layer_bytes_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostCachedMapped);
        slot->host_ptr = slot->buffer.mapped_ptr;
        slot->event = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        slot->layer_id = -1;
        slot->pinned = false;
        slot->permanently_pinned = false;
        slots_.push_back(std::move(slot));
    }
}

LayerSlot* LayerStreamer::find_or_evict_slot(int layer_id, bool& was_hit) {
    was_hit = false;
    // 1. Check if layer is already cached
    for (auto& slot : slots_) {
        if (slot->layer_id == layer_id) {
            was_hit = true;
            slot->last_used_timestamp = ++global_clock_;
            slot->frequency++;
            return slot.get();
        }
    }

    // 2. Evict unpinned slot
    LayerSlot* best_candidate = nullptr;
    uint64_t oldest_time = UINT64_MAX;
    uint32_t lowest_freq = UINT32_MAX;

    for (auto& slot : slots_) {
        if (slot->pinned || slot->permanently_pinned) continue;

        if (policy_ == EvictionPolicy::LRU) {
            if (slot->last_used_timestamp < oldest_time) {
                oldest_time = slot->last_used_timestamp;
                best_candidate = slot.get();
            }
        } else { // LFU
            if (slot->frequency < lowest_freq) {
                lowest_freq = slot->frequency;
                best_candidate = slot.get();
            }
        }
    }

    if (!best_candidate) {
        // Fallback: take any unpinned slot
        for (auto& slot : slots_) {
            if (!slot->pinned && !slot->permanently_pinned) {
                best_candidate = slot.get();
                break;
            }
        }
    }

    if (!best_candidate) {
        throw G4DenseFormatError("LayerStreamer: cache thrash! All layer slots are pinned");
    }

    best_candidate->layer_id = layer_id;
    best_candidate->last_used_timestamp = ++global_clock_;
    best_candidate->frequency = 1;
    return best_candidate;
}

LayerStreamer::LayerPlan LayerStreamer::plan_layers(const std::vector<int>& layer_ids) {
    std::lock_guard<std::mutex> lock(lock_);
    LayerPlan plan;
    plan.owner = this;

    for (int lid : layer_ids) {
        bool hit = false;
        LayerSlot* slot = find_or_evict_slot(lid, hit);
        slot->pinned = true; // Pin during execution

        size_t idx = plan.slots.size();
        plan.slots.push_back(slot);
        if (hit) {
            plan.hits.push_back(idx);
            total_hits_++;
        } else {
            plan.misses.push_back(idx);
            total_misses_++;
        }
    }
    return plan;
}

void LayerStreamer::io_worker() {
    for (;;) {
        IoJob job{};
        {
            std::unique_lock<std::mutex> lk(io_mutex_);
            io_cv_.wait(lk, [this] { return io_stop_ || !io_queue_.empty(); });
            if (io_stop_ && io_queue_.empty()) return;
            job = io_queue_.front();
            io_queue_.pop_front();
        }

        LayerSlot* slot = job.slot;
        DWORD err = 0;
        bool truncated = false;

        // The offset lives in the OVERLAPPED, so concurrent workers on different slots do not
        // share a file pointer. Blocking here is the entire point: this thread is not the one
        // submitting GPU work, so a read that completes inline from the page cache costs
        // nothing that matters.
        ResetEvent(slot->event);
        std::memset(&slot->ov, 0, sizeof(slot->ov));
        slot->ov.hEvent = slot->event;
        slot->ov.Offset = static_cast<DWORD>(job.offset & 0xFFFFFFFF);
        slot->ov.OffsetHigh = static_cast<DWORD>((job.offset >> 32) & 0xFFFFFFFF);

        DWORD bytes_read = 0;
        BOOL ok = ReadFile(file_handle_, slot->host_ptr,
                           static_cast<DWORD>(job.count), &bytes_read, &slot->ov);
        if (!ok) {
            err = GetLastError();
            if (err == ERROR_IO_PENDING) {
                err = 0;
                DWORD transferred = 0;
                if (!GetOverlappedResult(file_handle_, &slot->ov, &transferred, TRUE)) {
                    err = GetLastError();
                } else {
                    bytes_read = transferred;
                }
            }
        }
        if (err == 0 && bytes_read < job.count) truncated = true;

        {
            std::lock_guard<std::mutex> lk(io_mutex_);
            slot->io_error = err;
            slot->io_truncated = truncated;
            slot->io_pending = false;
        }
        io_done_cv_.notify_all();
    }
}

void LayerStreamer::stop_io_threads() {
    {
        std::lock_guard<std::mutex> lk(io_mutex_);
        io_stop_ = true;
    }
    io_cv_.notify_all();
    for (auto& t : io_threads_) {
        if (t.joinable()) t.join();
    }
    io_threads_.clear();
}

void LayerStreamer::issue_read(LayerSlot* slot, uint64_t file_offset, size_t count) {
    {
        std::lock_guard<std::mutex> lk(io_mutex_);
        slot->io_pending = true;
        slot->io_error = 0;
        slot->io_truncated = false;
        io_queue_.push_back(IoJob{slot, file_offset, count});
    }
    io_cv_.notify_one();
    slot->read_pending = true;
    total_io_calls_++;
    total_bytes_read_ += count;
}

void LayerStreamer::await_read(LayerSlot* slot, size_t count) {
    if (!slot->read_pending) return;
    DWORD err = 0;
    bool truncated = false;
    {
        std::unique_lock<std::mutex> lk(io_mutex_);
        io_done_cv_.wait(lk, [slot] { return !slot->io_pending; });
        err = slot->io_error;
        truncated = slot->io_truncated;
    }
    slot->read_pending = false;
    if (err != 0 || truncated) {
        std::stringstream ss;
        ss << "LayerStreamer: read failed (err=" << err
           << (truncated ? ", truncated" : "") << ") for " << count << " bytes";
        throw G4DenseFormatError(ss.str());
    }
}

void LayerStreamer::issue_reads(LayerPlan& plan) {
    for (size_t miss_idx : plan.misses) {
        LayerSlot* slot = plan.slots[miss_idx];
        const int lid = slot->layer_id;
        issue_read(slot, header_.layer_offsets[lid], header_.layer_sizes[lid]);
    }
}

void LayerStreamer::await_reads(LayerPlan& plan) {
    for (size_t miss_idx : plan.misses) {
        LayerSlot* slot = plan.slots[miss_idx];
        await_read(slot, header_.layer_sizes[slot->layer_id]);
    }
}

void LayerStreamer::fetch_misses(LayerPlan& plan) {
    issue_reads(plan);
    await_reads(plan);
}

void LayerStreamer::release_plan(LayerPlan& plan) {
    std::lock_guard<std::mutex> lock(lock_);
    for (LayerSlot* slot : plan.slots) {
        if (slot) slot->pinned = false;
    }
    plan.slots.clear();
    plan.hits.clear();
    plan.misses.clear();
    plan.owner = nullptr;
}

void LayerStreamer::apply_tier_pinning(const std::vector<int>& pinned_layers) {
    std::lock_guard<std::mutex> lock(lock_);
    for (auto& slot : slots_) {
        slot->permanently_pinned = false;
    }

    for (int lid : pinned_layers) {
        bool was_hit = false;
        LayerSlot* slot = find_or_evict_slot(lid, was_hit);
        slot->permanently_pinned = true;
        if (!was_hit) {
            uint64_t offset = header_.layer_offsets[lid];
            uint64_t size = header_.layer_sizes[lid];
            issue_read(slot, offset, size);
            await_read(slot, size);
        }
    }
}

void LayerStreamer::clear_cache() {
    std::lock_guard<std::mutex> guard(lock_);
    for (auto& slot : slots_) {
        if (slot->permanently_pinned || slot->pinned) continue;
        slot->layer_id = -1;          // forces a miss, and therefore a re-read, next plan
        slot->frequency = 0;
        slot->last_used_timestamp = 0;
    }
}

} // namespace g4dense
