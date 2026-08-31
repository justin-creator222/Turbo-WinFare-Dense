// Kernel- and memory-level attribution for the forward pass.
//
// Round 5 left the engine correct and 11.1 s per forward pass. Measuring where that goes
// (`run_turbo_dense --prompt "Hi" --max-tokens 2`) gave:
//
//     Layer stream I/O 10,064 ms | GPU queue wait 968 ms | LM head 51 ms | CPU other 21 ms
//
// This harness exists so the fixes for those are chosen by measurement rather than by
// reasoning about the hardware. Every previous round of this project that reasoned instead of
// measuring got the answer wrong.
//
//   bench_gpu [path-to-container]
//
// Sections:
//   1. Memory heaps and types, as the driver reports them right now.
//   2. Host-pointer import (VK_EXT_external_memory_host) -- can the whole model be mapped
//      into the GPU without copying it?
//   3. GemvInt4 achieved bandwidth per memory type, at the real projection shapes.
//   4. Dispatch and submission overhead: per-layer vkQueueWaitIdle vs one submit per token.
//   5. ReadFile throughput into write-combined vs HOST_CACHED memory.

#include "g4dense/vk_context.hpp"
#include "g4dense/vk_pipeline.hpp"
#include "g4dense/format.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace g4dense;
namespace fs = std::filesystem;

namespace {

using clk = std::chrono::high_resolution_clock;

std::string g_container_path;
std::vector<void*> g_import_regions;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

const char* residency_name(MemoryResidency r) {
    switch (r) {
        case MemoryResidency::HostVisibleMapped: return "HOST_VISIBLE (uncached/WC)";
        case MemoryResidency::HostCachedMapped:  return "HOST_CACHED";
        case MemoryResidency::DeviceLocalStaged: return "DEVICE_LOCAL";
        default:                                 return "Auto";
    }
}

std::string flags_to_string(VkMemoryPropertyFlags f) {
    std::string s;
    auto add = [&](VkMemoryPropertyFlags bit, const char* name) {
        if (f & bit) { if (!s.empty()) s += "|"; s += name; }
    };
    add(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL");
    add(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE");
    add(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT");
    add(VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED");
    if (s.empty()) s = "-";
    return s;
}

// ---------------------------------------------------------------------------------------
// 1. Heaps and types
// ---------------------------------------------------------------------------------------
void report_memory(VulkanContext& ctx) {
    std::cout << "\n=== 1. Memory heaps and types ===\n";
    const auto& mp = ctx.memory_properties();
    for (uint32_t h = 0; h < mp.memoryHeapCount; ++h) {
        std::cout << "  heap " << h << "  " << std::fixed << std::setprecision(2)
                  << (mp.memoryHeaps[h].size / (1024.0 * 1024.0 * 1024.0)) << " GiB"
                  << ((mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "  DEVICE_LOCAL" : "")
                  << "\n";
    }
    for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
        const auto flags = mp.memoryTypes[t].propertyFlags;
        // The AMD-only coherent/uncached variants duplicate every type; they are not candidates
        // for weight storage, so listing them only makes the table harder to read.
        if (flags & (VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD)) {
            continue;
        }
        std::cout << "  type " << std::setw(2) << t << "  heap " << mp.memoryTypes[t].heapIndex
                  << "  " << flags_to_string(flags) << "\n";
    }
    std::cout << "  VK_EXT_external_memory_host: "
              << (ctx.has_external_memory_host() ? "YES" : "NO")
              << "   minImportedHostPointerAlignment = "
              << ctx.min_imported_host_pointer_alignment() << "\n";
}

// ---------------------------------------------------------------------------------------
// 2. Import the whole container, layer by layer
// ---------------------------------------------------------------------------------------
struct MappedContainer {
    HANDLE file{INVALID_HANDLE_VALUE};
    HANDLE mapping{nullptr};
    const uint8_t* base{nullptr};
    uint64_t size{0};
    G4DenseHeader header{};
};

bool map_container(const std::string& path, MappedContainer& out) {
    out.file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (out.file == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER li;
    if (!GetFileSizeEx(out.file, &li)) return false;
    out.size = static_cast<uint64_t>(li.QuadPart);
    out.mapping = CreateFileMappingA(out.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!out.mapping) return false;
    out.base = static_cast<const uint8_t*>(MapViewOfFile(out.mapping, FILE_MAP_READ, 0, 0, 0));
    if (!out.base) return false;
    std::memcpy(&out.header, out.base, sizeof(G4DenseHeader));
    return true;
}

// Returns the imported layer buffers, or an empty vector if import is not possible.
std::vector<VkMemoryAllocation> bench_import(VulkanContext& ctx, MappedContainer& mc) {
    std::cout << "\n=== 2. Host-pointer import (the 90% question) ===\n";
    std::vector<VkMemoryAllocation> imported;
    if (!ctx.has_external_memory_host()) {
        std::cout << "  SKIPPED: extension unavailable\n";
        return imported;
    }

    // Which KINDS of host memory can be imported? The answer decides whether the weights can
    // stay in the page cache (import the mapped view) or must be copied once into private
    // memory at load (import a VirtualAlloc region).
    struct Probe { const char* what; void* ptr; bool owned; };
    std::vector<Probe> probes;
    probes.push_back({"MapViewOfFile view (page cache, read-only)",
                      const_cast<uint8_t*>(mc.base + mc.header.layer_offsets[0]), false});
    void* va = VirtualAlloc(nullptr, 64u << 20, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    probes.push_back({"VirtualAlloc PAGE_READWRITE", va, true});
    void* va_big = VirtualAlloc(nullptr, 512u << 20, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    probes.push_back({"VirtualAlloc PAGE_READWRITE, 512 MB", va_big, true});

    for (auto& pr : probes) {
        if (!pr.ptr) { std::cout << "  " << pr.what << ": host allocation failed\n"; continue; }
        try {
            VkMemoryAllocation t = ctx.import_host_buffer(pr.ptr, 64u << 20);
            std::cout << "  OK      " << pr.what << "\n";
            ctx.free_buffer(t);
        } catch (const std::exception& ex) {
            std::cout << "  REFUSED " << pr.what << "\n            " << ex.what() << "\n";
        }
    }
    for (auto& pr : probes) if (pr.owned && pr.ptr) VirtualFree(pr.ptr, 0, MEM_RELEASE);

    // Full-scale test of the only import that the driver accepts.
    //
    // The runner will do exactly this at load: one private region per layer, the container read
    // into it once, then imported so the GPU reads the weights in place. After that there is no
    // per-token I/O at all -- which is the 10,064 ms.
    const uint32_t n = mc.header.num_layers;
    uint64_t total = 0;
    std::vector<void*> regions;
    // Release the whole-file mapping first. Its pages are page-cache backed, and holding 17 GB
    // of them while also committing ~16 GiB of private memory is what ran the machine out
    // earlier -- the import failed at layer 58 of 60. The header is already copied out.
    if (mc.base) { UnmapViewOfFile(mc.base); mc.base = nullptr; }
    if (mc.mapping) { CloseHandle(mc.mapping); mc.mapping = nullptr; }
    if (mc.file != INVALID_HANDLE_VALUE) { CloseHandle(mc.file); mc.file = INVALID_HANDLE_VALUE; }

    // FILE_FLAG_NO_BUFFERING: the bytes are going straight into memory the GPU will read, so a
    // second copy in the page cache is pure waste -- 17 GB of it. Requires sector-aligned
    // offsets, sizes and destinations, which the 4096-aligned container and VirtualAlloc's
    // 64 KB granularity both satisfy.
    HANDLE fh = CreateFileA(g_container_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        std::cout << "  cannot reopen container for the full-scale test\n";
        return imported;
    }

    const auto t0 = clk::now();
    double read_ms = 0.0;
    for (uint32_t l = 0; l < n; ++l) {
        const uint64_t sz = mc.header.layer_sizes[l];
        void* region = VirtualAlloc(nullptr, sz, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!region) {
            std::cout << "  VirtualAlloc failed at layer " << l << " after "
                      << (total / (1024.0 * 1024.0 * 1024.0)) << " GiB\n";
            break;
        }
        regions.push_back(region);

        const auto tr = clk::now();
        LARGE_INTEGER pos;
        pos.QuadPart = static_cast<LONGLONG>(mc.header.layer_offsets[l]);
        SetFilePointerEx(fh, pos, nullptr, FILE_BEGIN);
        uint64_t done = 0;
        while (done < sz) {
            const DWORD chunk = static_cast<DWORD>(std::min<uint64_t>(sz - done, 1u << 30));
            DWORD got = 0;
            if (!ReadFile(fh, static_cast<uint8_t*>(region) + done, chunk, &got, nullptr) || got == 0) break;
            done += got;
        }
        read_ms += ms_since(tr);

        try {
            imported.push_back(ctx.import_host_buffer(region, sz));
            total += sz;
        } catch (const std::exception& ex) {
            std::cout << "  IMPORT FAILED at layer " << l << " after "
                      << (total / (1024.0 * 1024.0 * 1024.0)) << " GiB: " << ex.what() << "\n";
            break;
        }
    }
    const double dt = ms_since(t0);
    CloseHandle(fh);

    std::cout << std::fixed << std::setprecision(2)
              << "  imported " << imported.size() << " of " << n << " layers, "
              << (total / (1024.0 * 1024.0 * 1024.0)) << " GiB\n"
              << "  one-time load: " << (dt / 1000.0) << " s total, of which "
              << (read_ms / 1000.0) << " s ReadFile ("
              << (total / (read_ms / 1000.0) / 1e9) << " GB/s)\n"
              << "  per-token I/O after this: 0\n";
    if (!imported.empty()) {
        const auto& mp = ctx.memory_properties();
        const uint32_t ti = imported[0].memory_type_index;
        std::cout << "  memory type " << ti << " (heap " << mp.memoryTypes[ti].heapIndex << "): "
                  << flags_to_string(mp.memoryTypes[ti].propertyFlags) << "\n";
    }
    g_import_regions = regions;
    return imported;
}

// ---------------------------------------------------------------------------------------
// 3. GemvInt4 achieved bandwidth
// ---------------------------------------------------------------------------------------
struct GemvCase {
    const char* name;
    uint32_t rows;
    uint32_t in_dim;
};

void run_gemv_case(VulkanContext& ctx, VulkanPipelineManager& pm, VkCommandBuffer cmd, VkQueue q,
                   const GemvCase& c, MemoryResidency residency, int iters) {
    const uint32_t groups = c.in_dim / 64;
    const uint64_t w_bytes = static_cast<uint64_t>(c.rows) * (c.in_dim / 8) * 4;
    const uint64_t sb_bytes = static_cast<uint64_t>(c.rows) * groups * 2;

    VkMemoryAllocation w, s, b, x, o;
    try {
        w = ctx.allocate_buffer(w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, residency);
        s = ctx.allocate_buffer(sb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, residency);
        b = ctx.allocate_buffer(sb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, residency);
        x = ctx.allocate_buffer(static_cast<uint64_t>(c.in_dim) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                MemoryResidency::HostVisibleMapped);
        o = ctx.allocate_buffer(static_cast<uint64_t>(c.rows) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                MemoryResidency::HostVisibleMapped);
    } catch (const std::exception& ex) {
        std::cout << "    " << std::setw(28) << std::left << residency_name(residency)
                  << "  allocation failed: " << ex.what() << "\n";
        return;
    }

    if (x.mapped_ptr) {
        float* xp = static_cast<float*>(x.mapped_ptr);
        for (uint32_t i = 0; i < c.in_dim; ++i) xp[i] = 0.001f * static_cast<float>(i % 97);
    }

    VkDescriptorSet ds = pm.allocate_descriptor_set(ComputeKernel::GemvInt4);
    pm.update_storage_buffer(ds, 0, w.buffer, 0, w_bytes);
    pm.update_storage_buffer(ds, 1, s.buffer, 0, sb_bytes);
    pm.update_storage_buffer(ds, 2, b.buffer, 0, sb_bytes);
    pm.update_storage_buffer(ds, 3, x.buffer, 0, static_cast<uint64_t>(c.in_dim) * 4);
    pm.update_storage_buffer(ds, 6, o.buffer, 0, static_cast<uint64_t>(c.rows) * 4);

    uint32_t pc[8]{0};
    pc[0] = c.rows;
    pc[1] = c.in_dim;

    // One submission containing `iters` dispatches, so the number reflects sustained kernel
    // bandwidth rather than per-submit latency (measured separately in section 4).
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkResetCommandBuffer(cmd, 0);
    vkBeginCommandBuffer(cmd, &bi);
    pm.bind_kernel(cmd, ComputeKernel::GemvInt4);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pm.get_pipeline_layout(ComputeKernel::GemvInt4), 0, 1, &ds, 0, nullptr);
    pm.push_constants(cmd, ComputeKernel::GemvInt4, pc, sizeof(pc));
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    for (int i = 0; i < iters; ++i) {
        pm.dispatch(cmd, (c.rows + 7) / 8, 1, 1);   // GemvInt4: 8 rows per threadgroup
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);   // warm up
    vkQueueWaitIdle(q);

    const auto t0 = clk::now();
    vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(q);
    const double dt = ms_since(t0);

    const double bytes = static_cast<double>(w_bytes + 2 * sb_bytes) * iters;
    std::cout << "    " << std::setw(28) << std::left << residency_name(residency)
              << std::right << std::fixed << std::setprecision(1)
              << std::setw(9) << (dt / iters) << " ms/iter "
              << std::setw(8) << std::setprecision(2) << (bytes / (dt / 1000.0) / 1e9) << " GB/s\n";

    ctx.free_buffer(w); ctx.free_buffer(s); ctx.free_buffer(b);
    ctx.free_buffer(x); ctx.free_buffer(o);
}

void bench_gemv(VulkanContext& ctx, VulkanPipelineManager& pm, VkCommandBuffer cmd, VkQueue q) {
    std::cout << "\n=== 3. GemvInt4 achieved bandwidth (real projection shapes) ===\n";
    std::cout << "  Weights are read once per token, so this is the ceiling on the GPU phase.\n";
    const GemvCase cases[] = {
        {"gate/up_proj  21504 x 5376", 21504, 5376},
        {"down_proj      5376 x 21504", 5376, 21504},
    };
    const MemoryResidency modes[] = {
        MemoryResidency::HostVisibleMapped,
        MemoryResidency::HostCachedMapped,
        MemoryResidency::DeviceLocalStaged,
    };
    for (const auto& c : cases) {
        std::cout << "  " << c.name << "\n";
        for (auto m : modes) run_gemv_case(ctx, pm, cmd, q, c, m, 4);
    }
}

// ---------------------------------------------------------------------------------------
// 3b. Batched GEMM vs repeated GEMV -- does prefill batching actually pay?
// ---------------------------------------------------------------------------------------
void bench_batch(VulkanContext& ctx, VulkanPipelineManager& pm, VkCommandBuffer cmd, VkQueue q) {
    std::cout << "\n=== 3b. Prefill: M x GemvInt4 vs 1 x GemmInt4Batch ===\n";
    std::cout << "  Prefill runs one full weight pass per prompt token today. This is whether\n"
                 "  batching M positions against one weight pass actually recovers that.\n";

    const GemvCase cases[] = {
        {"gate/up_proj  21504 x 5376", 21504, 5376},
        {"down_proj      5376 x 21504", 5376, 21504},
    };
    const uint32_t M = 8;

    for (const auto& c : cases) {
        const uint32_t groups = c.in_dim / 64;
        const uint64_t w_bytes  = static_cast<uint64_t>(c.rows) * (c.in_dim / 8) * 4;
        const uint64_t sb_bytes = static_cast<uint64_t>(c.rows) * groups * 2;
        const uint64_t x_bytes  = static_cast<uint64_t>(c.in_dim) * 4 * M;
        const uint64_t o_bytes  = static_cast<uint64_t>(c.rows) * 4 * M;

        VkMemoryAllocation w, sb, bb, x, o;
        try {
            w  = ctx.allocate_buffer(w_bytes,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostCachedMapped);
            sb = ctx.allocate_buffer(sb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostCachedMapped);
            bb = ctx.allocate_buffer(sb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostCachedMapped);
            x  = ctx.allocate_buffer(x_bytes,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
            o  = ctx.allocate_buffer(o_bytes,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        } catch (const std::exception& ex) {
            std::cout << "    " << c.name << ": allocation failed: " << ex.what() << "\n";
            continue;
        }
        if (x.mapped_ptr) {
            float* xp = static_cast<float*>(x.mapped_ptr);
            for (uint64_t i = 0; i < x_bytes / 4; ++i) xp[i] = 0.001f * static_cast<float>(i % 97);
        }

        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;

        auto time_it = [&](bool batched) {
            pm.reset_descriptor_pool();
            const ComputeKernel k = batched ? ComputeKernel::GemmInt4Batch : ComputeKernel::GemvInt4;
            vkResetCommandBuffer(cmd, 0);
            vkBeginCommandBuffer(cmd, &bi);
            const uint32_t reps = batched ? 1u : M;
            for (uint32_t r = 0; r < reps; ++r) {
                VkDescriptorSet ds = pm.allocate_descriptor_set(k);
                pm.update_storage_buffer(ds, 0, w.buffer, 0, w_bytes);
                pm.update_storage_buffer(ds, 1, sb.buffer, 0, sb_bytes);
                pm.update_storage_buffer(ds, 2, bb.buffer, 0, sb_bytes);
                pm.update_storage_buffer(ds, 3, x.buffer, 0, x_bytes);
                pm.update_storage_buffer(ds, 6, o.buffer, 0, o_bytes);
                uint32_t pc[12]{0};
                pc[0] = c.rows;
                pc[1] = c.in_dim;
                if (batched) {
                    pc[8] = M; pc[9] = c.in_dim * 4; pc[10] = c.rows * 4;
                } else {
                    pc[5] = r * c.in_dim * 4;   // x_byte_off for this position
                    pc[6] = r * c.rows * 4;     // out_byte_off
                }
                pm.bind_kernel(cmd, k);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pm.get_pipeline_layout(k), 0, 1, &ds, 0, nullptr);
                pm.push_constants(cmd, k, pc, sizeof(pc));
                const uint32_t per_group = batched ? 8u : 8u;   // both pack 8 rows per group
                pm.dispatch(cmd, (c.rows + per_group - 1) / per_group, 1, 1);
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            }
            vkEndCommandBuffer(cmd);
            vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(q);
            const auto t0 = clk::now();
            vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(q);
            return ms_since(t0);
        };

        const double gemv_ms = time_it(false);
        const double gemm_ms = time_it(true);
        std::cout << "  " << c.name << "  (M = " << M << ")\n"
                  << std::fixed << std::setprecision(2)
                  << "    " << std::setw(28) << std::left << "M x GemvInt4"
                  << std::right << std::setw(8) << gemv_ms << " ms\n"
                  << "    " << std::setw(28) << std::left << "1 x GemmInt4Batch"
                  << std::right << std::setw(8) << gemm_ms << " ms"
                  << "   speedup " << std::setprecision(2) << (gemv_ms / gemm_ms) << "x\n";

        ctx.free_buffer(w); ctx.free_buffer(sb); ctx.free_buffer(bb);
        ctx.free_buffer(x); ctx.free_buffer(o);
    }
}

// ---------------------------------------------------------------------------------------
// 4. Submission overhead
// ---------------------------------------------------------------------------------------
void bench_submission(VulkanContext& ctx, VulkanPipelineManager& pm, VkCommandBuffer cmd, VkQueue q) {
    std::cout << "\n=== 4. Submission overhead ===\n";
    std::cout << "  The engine submits once per layer and calls vkQueueWaitIdle after each:\n"
                 "  61 full device drains per token.\n";

    const uint32_t DIM = 5376;
    VkMemoryAllocation a = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                               MemoryResidency::HostVisibleMapped);
    VkMemoryAllocation b = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                               MemoryResidency::HostVisibleMapped);
    VkMemoryAllocation o = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                               MemoryResidency::HostVisibleMapped);

    auto record_one = [&](VkCommandBuffer c) {
        VkDescriptorSet ds = pm.allocate_descriptor_set(ComputeKernel::ResidualAccum);
        pm.update_storage_buffer(ds, 0, a.buffer, 0, DIM * 4);
        pm.update_storage_buffer(ds, 1, b.buffer, 0, DIM * 4);
        pm.update_storage_buffer(ds, 6, o.buffer, 0, DIM * 4);
        uint32_t pc[8]{0};
        pc[0] = DIM;
        float one = 1.0f;
        std::memcpy(&pc[1], &one, 4);
        std::memcpy(&pc[2], &one, 4);
        pm.bind_kernel(c, ComputeKernel::ResidualAccum);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pm.get_pipeline_layout(ComputeKernel::ResidualAccum), 0, 1, &ds, 0, nullptr);
        pm.push_constants(c, ComputeKernel::ResidualAccum, pc, sizeof(pc));
        pm.dispatch(c, (DIM + 255) / 256, 1, 1);
    };

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    // (a) today's shape: one submit + one full drain per layer
    pm.reset_descriptor_pool();
    auto t0 = clk::now();
    for (int i = 0; i < 61; ++i) {
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &bi);
        record_one(cmd);
        vkEndCommandBuffer(cmd);
        vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(q);
    }
    const double per_layer = ms_since(t0);

    // (b) the target shape: one command buffer, one submit, one fence
    pm.reset_descriptor_pool();
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(ctx.device(), &fci, nullptr, &fence);

    t0 = clk::now();
    vkResetCommandBuffer(cmd, 0);
    vkBeginCommandBuffer(cmd, &bi);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    for (int i = 0; i < 61; ++i) {
        record_one(cmd);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    vkEndCommandBuffer(cmd);
    vkQueueSubmit(q, 1, &si, fence);
    vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX);
    const double single = ms_since(t0);
    vkDestroyFence(ctx.device(), fence, nullptr);

    std::cout << std::fixed << std::setprecision(2)
              << "    61 submits + 61 vkQueueWaitIdle : " << per_layer << " ms\n"
              << "    1 submit + 1 fence             : " << single << " ms\n"
              << "    saving per token               : " << (per_layer - single) << " ms\n";

    ctx.free_buffer(a); ctx.free_buffer(b); ctx.free_buffer(o);
}

// ---------------------------------------------------------------------------------------
// 5. ReadFile destination
// ---------------------------------------------------------------------------------------
void bench_readfile(VulkanContext& ctx, const std::string& path, const MappedContainer& mc) {
    std::cout << "\n=== 5. ReadFile into GPU-visible memory ===\n";
    std::cout << "  This is what the streaming path does 54 times per token.\n";
    if (mc.header.num_layers == 0) { std::cout << "  SKIPPED: no container\n"; return; }

    const uint64_t bytes = mc.header.layer_sizes[0];
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { std::cout << "  SKIPPED: cannot open container\n"; return; }

    const MemoryResidency modes[] = {MemoryResidency::HostVisibleMapped,
                                     MemoryResidency::HostCachedMapped};
    for (auto m : modes) {
        VkMemoryAllocation dst;
        try {
            dst = ctx.allocate_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, m);
        } catch (const std::exception& ex) {
            std::cout << "    " << std::setw(28) << std::left << residency_name(m)
                      << "  allocation failed: " << ex.what() << "\n";
            continue;
        }
        // Read four different layers so the page cache, not a repeated single range, is what
        // is being measured.
        const auto t0 = clk::now();
        uint64_t moved = 0;
        for (uint32_t l = 0; l < 4 && l < mc.header.num_layers; ++l) {
            LARGE_INTEGER pos;
            pos.QuadPart = static_cast<LONGLONG>(mc.header.layer_offsets[l]);
            SetFilePointerEx(f, pos, nullptr, FILE_BEGIN);
            DWORD got = 0;
            ReadFile(f, dst.mapped_ptr, static_cast<DWORD>(mc.header.layer_sizes[l]), &got, nullptr);
            moved += got;
        }
        const double dt = ms_since(t0);
        std::cout << "    " << std::setw(28) << std::left << residency_name(m)
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(9) << dt << " ms  "
                  << std::setw(6) << (moved / (dt / 1000.0) / 1e9) << " GB/s\n";
        ctx.free_buffer(dst);
    }
    CloseHandle(f);
}

} // namespace

int main(int argc, char** argv) {
    std::string model = (argc > 1) ? argv[1] : "models/gemma-4-31b-dense.g4dense";
    g_container_path = model;

    std::cout << "========================================================\n"
              << "  Turbo-WinFare Dense: GPU / memory attribution\n"
              << "========================================================\n";

    try {
        VulkanContext ctx;
        ctx.initialize();
        std::cout << "Device: " << ctx.device_name() << "\n";

        VulkanPipelineManager pm(ctx);
        pm.initialize_pipelines();

        VkCommandPoolCreateInfo cp_ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cp_ci.queueFamilyIndex = ctx.compute_queue_family();
        cp_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkCommandPool pool = VK_NULL_HANDLE;
        vkCreateCommandPool(ctx.device(), &cp_ci, nullptr, &pool);

        VkCommandBufferAllocateInfo cb_ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cb_ai.commandPool = pool;
        cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb_ai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(ctx.device(), &cb_ai, &cmd);
        VkQueue q = ctx.compute_queue();

        report_memory(ctx);

        MappedContainer mc;
        const bool have_model = fs::exists(model) && map_container(model, mc);
        if (!have_model) {
            std::cout << "\n  NOTE: no container at " << model
                      << " -- sections 2 and 5 are skipped.\n";
        } else {
            auto imported = bench_import(ctx, mc);
            for (auto& a : imported) ctx.free_buffer(a);
            for (void* r : g_import_regions) VirtualFree(r, 0, MEM_RELEASE);
            g_import_regions.clear();
        }

        bench_gemv(ctx, pm, cmd, q);
    bench_batch(ctx, pm, cmd, q);
        bench_submission(ctx, pm, cmd, q);
        if (have_model) bench_readfile(ctx, model, mc);

        vkDestroyCommandPool(ctx.device(), pool, nullptr);
        std::cout << "\nDone.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "EXCEPTION: " << ex.what() << "\n";
        return 1;
    }
}
