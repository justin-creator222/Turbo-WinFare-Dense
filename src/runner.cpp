#include "g4dense/runner.hpp"
#include "g4dense/format.hpp"

#include <chrono>
#include <iostream>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <thread>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <deque>

namespace g4dense {

namespace {

// Every non-quantized tensor in the MLX export is BF16, exactly as the safetensors header
// says: the LayerNorm family (input/post_attn/pre_ffn/post_ffn layernorms, q_norm, k_norm,
// the final model norm), the per-layer scalar, and the quantization scales and biases.
//
// A previous round decided the LayerNorm family was secretly IEEE FP16, because the BF16
// decode "looks wrong" (model.norm.weight median 6.28, max 510) while the FP16 decode looks
// like a textbook norm weight (median 2.393). That reasoning does not work: on this value
// range the two decodings are a BIJECTION, so both are self-consistent and neither can be
// chosen by appearance. Behaviour on the checkpoint settles it:
//
//                    pre-softmax |score|      hidden rms, layers 0..7
//   read as BF16     mean 5-16, max 24        0.81 1.69 1.68 1.66 1.66 1.76 2.08  (stable)
//   read as FP16     mean 192-327, max 549    4.3 8.2 15.6 30 57 106 197  (x2 per layer)
//
// Gemma4TextAttention sets self.scaling = 1.0, so nothing downstream rescales those scores:
// FP16 turns softmax into a hard argmax and makes the residual stream diverge as 2^60.

inline float bf16_to_f32(uint16_t val) {
    uint32_t u32 = static_cast<uint32_t>(val) << 16;
    float f;
    std::memcpy(&f, &u32, sizeof(float));
    return f;
}

} // namespace

ForwardRunner::ForwardRunner(std::shared_ptr<VulkanContext> vk_ctx,
                             std::shared_ptr<Tokenizer> tokenizer,
                             const std::string& container_path)
    : vk_ctx_(vk_ctx), tokenizer_(tokenizer), container_path_(container_path) {}

ForwardRunner::~ForwardRunner() {
    // Imported buffers must be destroyed before the pages behind them are released, and
    // free_buffer() deliberately does not touch host memory it did not allocate.
    for (auto& a : resident_layer_bufs_) {
        if (a.buffer != VK_NULL_HANDLE && vk_ctx_) vk_ctx_->free_buffer(a);
    }
    resident_layer_bufs_.clear();
    for (void* r : resident_regions_) {
        if (r) VirtualFree(r, 0, MEM_RELEASE);
    }
    resident_regions_.clear();

    if (vk_ctx_) {
        VkDevice dev = vk_ctx_->device();
        if (dev != VK_NULL_HANDLE) {
            if (fence_ != VK_NULL_HANDLE) {
                vkDestroyFence(dev, fence_, nullptr);
                fence_ = VK_NULL_HANDLE;
            }
            if (cmd_pool_ != VK_NULL_HANDLE) {
                vkDestroyCommandPool(dev, cmd_pool_, nullptr);
                cmd_pool_ = VK_NULL_HANDLE;
            }
        }
        vk_ctx_->free_buffer(buf_hidden_);
        vk_ctx_->free_buffer(buf_norm_);
        vk_ctx_->free_buffer(buf_q_);
        vk_ctx_->free_buffer(buf_k_);
        vk_ctx_->free_buffer(buf_v_);
        vk_ctx_->free_buffer(buf_attn_out_);
        vk_ctx_->free_buffer(buf_proj_out_);
        vk_ctx_->free_buffer(buf_gate_);
        vk_ctx_->free_buffer(buf_up_);
        vk_ctx_->free_buffer(buf_ffn_out_);
        vk_ctx_->free_buffer(buf_final_norm_w_);
        vk_ctx_->free_buffer(buf_logits_);
    }

    if (mapped_data_ != nullptr) {
        UnmapViewOfFile(mapped_data_);
        mapped_data_ = nullptr;
    }
    if (mapping_handle_ != NULL) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = NULL;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
}

void ForwardRunner::initialize() {
    // 1. Open and Memory-Map Container File
    file_handle_ = CreateFileA(
        container_path_.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw G4DenseFormatError("ForwardRunner: cannot open container file " + container_path_);
    }

    LARGE_INTEGER li_size;
    if (!GetFileSizeEx(file_handle_, &li_size)) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw G4DenseFormatError("ForwardRunner: failed to get file size for " + container_path_);
    }
    file_size_ = li_size.QuadPart;

    mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_handle_ == NULL) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw G4DenseFormatError("ForwardRunner: failed to create file mapping for " + container_path_);
    }

    mapped_data_ = static_cast<const uint8_t*>(MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
    if (mapped_data_ == nullptr) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = NULL;
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw G4DenseFormatError("ForwardRunner: failed to MapViewOfFile for " + container_path_);
    }

    std::memcpy(&header_, mapped_data_, sizeof(G4DenseHeader));
    validate_header(header_);

    pipeline_mgr_ = std::make_unique<VulkanPipelineManager>(*vk_ctx_);
    pipeline_mgr_->initialize_pipelines();

    // 2. Layer residency.
    //
    // Import first, streamer second. Both draw on the same device-memory ceiling, and a
    // resident layer is strictly better than a pinned slot: the slot still costs a 269 MB copy
    // per token, the resident layer costs nothing. Allocating the slot pool first cost 2.63 GiB
    // of ceiling and left only 35 of 60 layers resident.
    uint64_t max_layer_bytes = 0;
    for (uint32_t i = 0; i < header_.num_layers; ++i) {
        if (header_.layer_sizes[i] > max_layer_bytes) {
            max_layer_bytes = header_.layer_sizes[i];
        }
    }

    // 3. Initialize KV Cache
    KVCacheConfig kv_cfg;
    kv_cfg.num_layers = header_.num_layers;
    kv_cfg.num_kv_heads = header_.num_kv_heads;
    kv_cfg.head_dim = header_.head_dim;
    // Full-attention geometry, read from the container rather than assumed. A container
    // written before these fields existed reports 0; resolve_layer_geometry supplies the 31B
    // values in that case, and has_legacy_global_geometry below says so out loud.
    {
        const LayerGeometry g = resolve_layer_geometry(header_, 0);
        (void)g;
        kv_cfg.global_kv_heads = header_.global_kv_heads ? header_.global_kv_heads : 4u;
        kv_cfg.global_head_dim = header_.global_head_dim ? header_.global_head_dim : 512u;
    }
    if (has_legacy_global_geometry(header_)) {
        std::cout << "[ForwardRunner] NOTE: container predates the full-attention geometry "
                     "header fields; assuming head_dim=" << kv_cfg.global_head_dim
                  << " kv_heads=" << kv_cfg.global_kv_heads
                  << " for the full-attention layers. Reconvert to record them."
                  << std::endl;
    }
    kv_cfg.sliding_window = header_.sliding_window;
    // Capped at ATTN_MAX_SPAN, not the spec's 8192.
    //
    // Attention.hlsl stages scores in `groupshared float s_scores[ATTN_MAX_SPAN]` (4096) and
    // indexes it as `s_scores[t - first]`. Sliding-window layers are safe at any context --
    // their span is bounded by sliding_window -- but full-attention layers pass
    // capacity == max_context with first == 0, so a context above 4096 writes past the end of
    // a groupshared array. That is a buffer overflow, not a degraded result.
    //
    // Raising this needs an online-softmax (flash-style) rewrite that never materializes the
    // full score span. Until then the honest configuration is a 4096 ceiling, enforced below
    // rather than left as a comment.
    kv_cfg.max_context = 4096;
    if (kv_cfg.max_context > kAttentionMaxSpan) {
        throw G4DenseFormatError(
            "ForwardRunner: max_context " + std::to_string(kv_cfg.max_context) +
            " exceeds Attention.hlsl's ATTN_MAX_SPAN of " + std::to_string(kAttentionMaxSpan) +
            "; full-attention layers would overflow groupshared s_scores. Raise ATTN_MAX_SPAN "
            "(costs groupshared occupancy) or implement online softmax.");
    }
    kv_cfg.global_layer_mask = header_.global_layer_mask;
    // FP16 halves the cache to 1.10 GiB, which is worth about four more resident layers.
    // KVWrite converts on the way in and Attention reads it back with f16_load; the
    // argmax-exact oracle diff is what says the precision loss is acceptable.
    kv_cfg.dtype = KVDType::FP16;

    kv_cache_ = std::make_unique<KVCacheManager>(vk_ctx_, kv_cfg);
    kv_cache_->initialize();

    draft_runtime_ = std::make_unique<DraftRuntime>();
    speculator_ = std::make_unique<SpeculativeCoordinator>();

    compute_layer_offsets();
    allocate_gpu_resources();

    // Reserve the streaming pool BEFORE importing. Import is greedy -- it takes memory until
    // the device refuses -- so anything allocated after it can fail, and this pool did: four
    // 269 MB slots would not fit once 46 layers were resident, and the throw escaped as an
    // uncaught exception at startup.
    //
    // PREFETCH_DEPTH (3) in flight plus the one being consumed. Four slots cost 1.05 GiB of
    // ceiling, i.e. four layers that could otherwise have been resident; that is the price of
    // being able to stream the remainder at all.
    const size_t total_slots = std::min<size_t>(4, header_.num_layers);
    // Unbuffered. Two reasons, both measured:
    //
    //   * Buffered overlapped reads complete INLINE when the data is already in the page
    //     cache, so issue_reads() blocks and nothing overlaps. Splitting issue from await
    //     moved 956 ms out of the I/O phase and straight into "CPU other" without saving any
    //     of it. FILE_FLAG_NO_BUFFERING makes the read genuinely asynchronous.
    //   * The destination is memory the GPU reads directly, so a second copy of the same
    //     bytes in the page cache is waste -- and the machine is already at its memory
    //     ceiling.
    //
    // Requires sector-aligned offsets, sizes and destination pointers: the container is
    // 4096-aligned throughout and Vulkan maps these slots on a page boundary.
    streamer_ = std::make_unique<LayerStreamer>(vk_ctx_, container_path_, total_slots,
                                                max_layer_bytes, IOMode::Unbuffered);
    streamer_->initialize(header_);

    load_resident_layers();

    // Nothing left to stream: hand the pool's 1.05 GiB back rather than holding four slots
    // that will never be filled. This is the state a larger BIOS UMA frame buffer would put
    // the machine in, since the ceiling that bounds residency grows with it.
    if (streamed_layers_.empty()) {
        streamer_.reset();
        std::cout << "[ForwardRunner] All layers resident; streaming pool released."
                  << std::endl;
    }

    switch_memory_tier(active_tier_id_);
}

void ForwardRunner::compute_layer_offsets() {
    layer_gpu_offsets_.resize(header_.num_layers);

    for (uint32_t l = 0; l < header_.num_layers; ++l) {
        bool is_global = (header_.global_layer_mask & (1ULL << l)) != 0;
        const LayerGeometry geo = resolve_layer_geometry(header_, l);
        uint32_t head_dim = geo.head_dim;
        uint32_t q_heads = geo.q_heads;
        uint32_t kv_heads = geo.kv_heads;
        uint32_t d_model = header_.d_model;
        uint32_t d_ff = layer_ffn_width(header_, l);

        LayerOffsetsGPU& lo = layer_gpu_offsets_[l];
        uint32_t cur = 0;

        // Must mirror tools/convert_hf_to_g4dense.py exactly. A KV-sharing layer carries no
        // k_norm (it has no k_proj), and a PLE model appends one more norm.
        lo.kv_shared = is_kv_shared_layer(header_, l);
        lo.kv_donor = kv_donor_layer(header_, l);

        lo.in_norm_off = cur; cur += d_model * 2;
        lo.post_attn_norm_off = cur; cur += d_model * 2;
        lo.pre_ffn_norm_off = cur; cur += d_model * 2;
        lo.post_ffn_norm_off = cur; cur += d_model * 2;
        lo.q_norm_off = cur; cur += head_dim * 2;
        if (!lo.kv_shared) { lo.k_norm_off = cur; cur += head_dim * 2; }
        lo.layer_scalar_off = cur; cur += 2;
        if (header_.ple_dim) { lo.post_ple_norm_off = cur; cur += d_model * 2; }

        // Read layer scalar value
        if (mapped_data_ && header_.layer_offsets[l] > 0) {
            const uint8_t* layer_ptr = mapped_data_ + header_.layer_offsets[l];
            uint16_t ls_bf16 = *reinterpret_cast<const uint16_t*>(layer_ptr + lo.layer_scalar_off);
            // layer_scalar is genuinely BF16 -- it is NOT a LayerNorm weight despite sitting
            // beside them in the container. BF16 gives 0.089355 for layer 0 (essentially
            // 1/sqrt(2*num_layers)); FP16 gives 1.4287, a 16x over-scale applied to the whole
            // residual stream in EVERY layer, which compounds catastrophically over 60.
            lo.layer_scalar = bf16_to_f32(ls_bf16);
        } else {
            lo.layer_scalar = 1.0f / std::sqrt(2.0f * static_cast<float>(header_.num_layers));
        }

        auto setup_proj = [&](LayerOffsetsGPU::ProjOffsets& p, uint32_t rows, uint32_t in_dim) {
            p.rows = rows;
            p.in_dim = in_dim;
            uint32_t num_groups = in_dim / 64;
            p.w_off = cur; cur += (rows * in_dim) / 2;
            p.s_off = cur; cur += rows * num_groups * 2;
            p.b_off = cur; cur += rows * num_groups * 2;
        };

        setup_proj(lo.q_proj, q_heads * head_dim, d_model);
        if (!lo.kv_shared) {
            setup_proj(lo.k_proj, kv_heads * head_dim, d_model);
            // Full-attention layers omit v_proj only when attention_k_eq_v is set, which the
            // 31B does and E2B does not -- so this keys off the tensor actually being present,
            // which for the 31B is exactly the global layers.
            if (!is_global || header_.ple_dim != 0) {
                setup_proj(lo.v_proj, kv_heads * head_dim, d_model);
            }
        }
        setup_proj(lo.o_proj, d_model, q_heads * head_dim);
        setup_proj(lo.gate_proj, d_ff, d_model);
        setup_proj(lo.up_proj, d_ff, d_model);
        setup_proj(lo.down_proj, d_model, d_ff);
        if (header_.ple_dim) {
            setup_proj(lo.ple_gate, header_.ple_dim, d_model);
            setup_proj(lo.ple_proj, d_model, header_.ple_dim);
        }
    }
}

void ForwardRunner::allocate_gpu_resources() {
    VkDevice dev = vk_ctx_->device();

    // Command pool & buffer
    VkCommandPoolCreateInfo pool_ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_ci.queueFamilyIndex = vk_ctx_->compute_queue_family();
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(dev, &pool_ci, nullptr, &cmd_pool_);

    VkCommandBufferAllocateInfo cmd_ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_ai.commandPool = cmd_pool_;
    cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(dev, &cmd_ai, &cmd_);

    VkFenceCreateInfo fence_ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(dev, &fence_ci, nullptr, &fence_);

    // Allocate GPU activation buffers
    uint32_t d_model = header_.d_model;
    // Widest feed-forward across layers: E2B's last 20 are twice the rest.
    uint32_t d_ff = header_.d_ff;
    for (uint32_t i = 0; i < header_.num_layers && i < G4DenseHeader::MAX_LAYERS; ++i) {
        d_ff = std::max(d_ff, layer_ffn_width(header_, i));
    }
    uint32_t max_q_dim = 32 * 512;
    uint32_t max_kv_dim = 16 * 512;

    // Every per-token activation buffer is sized for a whole prefill batch, laid out
    // [position][dim] with no padding, so a batched dispatch addresses position m at
    // m * dim * 4 and the elementwise kernels can treat the whole thing as one n * M vector.
    // At kGemmMaxBatch = 8 the largest of these (d_ff) is under 700 KB, so this costs nothing
    // against a memory budget measured in gigabytes.
    const uint32_t B = kGemmMaxBatch;

    buf_hidden_ = vk_ctx_->allocate_buffer(d_model * 4 * B, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
    buf_norm_ = vk_ctx_->allocate_buffer(d_model * 4 * B, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
    buf_q_ = vk_ctx_->allocate_buffer(max_q_dim * 4 * B, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
    buf_k_ = vk_ctx_->allocate_buffer(max_kv_dim * 4 * B, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
    buf_v_ = vk_ctx_->allocate_buffer(max_kv_dim * 4 * B, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
    buf_attn_out_ = vk_ctx_->allocate_buffer(max_q_dim * 4 * B, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
    buf_proj_out_ = vk_ctx_->allocate_buffer(d_model * 4 * B, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
    buf_gate_ = vk_ctx_->allocate_buffer(d_ff * 4 * B, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
    buf_up_ = vk_ctx_->allocate_buffer(d_ff * 4 * B, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
    buf_ffn_out_ = vk_ctx_->allocate_buffer(d_model * 4 * B, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
    buf_final_norm_w_ = vk_ctx_->allocate_buffer(d_model * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
    buf_logits_ = vk_ctx_->allocate_buffer(static_cast<uint64_t>(header_.vocab_size) * 4 * B, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);

    // Upload the tied LM head once, so the per-token head becomes a GPU GEMV instead of a
    // scalar CPU loop. Layout in the container is [packed INT4 | BF16 scales | BF16 biases],
    // contiguous, so a single memcpy preserves the offsets the kernel expects.
    if (mapped_data_ && header_.lm_head_size > 0) {
        const uint32_t packed_cols = d_model / 8;
        const uint32_t groups_per_row = d_model / 64;
        lm_head_w_bytes_ = static_cast<uint64_t>(header_.vocab_size) * packed_cols * sizeof(uint32_t);
        lm_head_s_bytes_ = static_cast<uint64_t>(header_.vocab_size) * groups_per_row * sizeof(uint16_t);
        const uint64_t total = lm_head_w_bytes_ + 2 * lm_head_s_bytes_;

        if (total > header_.lm_head_size) {
            throw G4DenseFormatError(
                "ForwardRunner: computed LM head size " + std::to_string(total) +
                " exceeds header lm_head_size " + std::to_string(header_.lm_head_size));
        }

        buf_lm_head_ = vk_ctx_->allocate_buffer(total, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                MemoryResidency::HostVisibleMapped);
        std::memcpy(buf_lm_head_.mapped_ptr, mapped_data_ + header_.lm_head_offset,
                    static_cast<size_t>(total));
    }

    // Per-layer embedding resources.
    if (header_.ple_dim) {
        const uint32_t ple_rows = header_.num_layers * header_.ple_dim;   // 8960 on E2B
        const uint64_t emb_w = static_cast<uint64_t>(header_.ple_vocab) * ple_rows / 2;
        const uint64_t emb_sb = static_cast<uint64_t>(header_.ple_vocab) * (ple_rows / 64) * 2;
        ple_emb_w_bytes_ = emb_w;
        ple_emb_s_bytes_ = emb_sb;

        const uint64_t proj_w = static_cast<uint64_t>(ple_rows) * d_model / 2;
        const uint64_t proj_sb = static_cast<uint64_t>(ple_rows) * (d_model / 64) * 2;

        ple_model_proj_.rows = ple_rows;
        ple_model_proj_.in_dim = d_model;
        ple_model_proj_.w_off = 0;
        ple_model_proj_.s_off = static_cast<uint32_t>(proj_w);
        ple_model_proj_.b_off = static_cast<uint32_t>(proj_w + proj_sb);

        buf_ple_w_ = vk_ctx_->allocate_buffer(proj_w + 2 * proj_sb,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              MemoryResidency::HostCachedMapped);
        buf_ple_norm_w_ = vk_ctx_->allocate_buffer(header_.ple_dim * 2,
                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                   MemoryResidency::HostVisibleMapped);
        buf_ple_emb_ = vk_ctx_->allocate_buffer(static_cast<uint64_t>(ple_rows) * 4 * B,
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                MemoryResidency::HostVisibleMapped);
        buf_ple_ = vk_ctx_->allocate_buffer(static_cast<uint64_t>(ple_rows) * 4 * B,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            MemoryResidency::HostVisibleMapped);
        buf_ple_g_ = vk_ctx_->allocate_buffer(static_cast<uint64_t>(header_.ple_dim) * 4 * B,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              MemoryResidency::HostVisibleMapped);

        if (mapped_data_ && header_.ple_offset) {
            const uint8_t* base = mapped_data_ + header_.ple_offset + emb_w + 2 * emb_sb;
            std::memcpy(buf_ple_w_.mapped_ptr, base, static_cast<size_t>(proj_w + 2 * proj_sb));
            std::memcpy(buf_ple_norm_w_.mapped_ptr, base + proj_w + 2 * proj_sb,
                        header_.ple_dim * 2);
        }
        std::cout << "[ForwardRunner] per-layer embeddings active: ple_dim=" << header_.ple_dim
                  << ", " << header_.num_kv_shared_layers << " KV-sharing layers." << std::endl;
    }

    // Copy final norm weight (BF16) into buf_final_norm_w_
    if (mapped_data_ && header_.embed_offset > 0) {
        uint32_t packed_cols = d_model / 8;
        uint32_t groups_per_row = d_model / 64;
        size_t embed_w_bytes = static_cast<size_t>(header_.vocab_size) * packed_cols * sizeof(uint32_t);
        size_t embed_s_bytes = static_cast<size_t>(header_.vocab_size) * groups_per_row * sizeof(uint16_t);
        size_t embed_b_bytes = static_cast<size_t>(header_.vocab_size) * groups_per_row * sizeof(uint16_t);
        const uint8_t* norm_w_ptr = mapped_data_ + header_.embed_offset + embed_w_bytes + embed_s_bytes + embed_b_bytes;
        std::memcpy(buf_final_norm_w_.mapped_ptr, norm_w_ptr, d_model * 2);
    }
}

// Import as many layers as the device will take, once, at load.
//
// This is the fix for the 90% of the forward pass that was layer streaming. On a UMA APU the
// weights are already in system DRAM; copying them into a "GPU buffer" every token was pure
// waste. VK_EXT_external_memory_host lets a VkBuffer be backed by memory we already own, so
// after this runs the GPU reads the weights in place and the per-token copy disappears.
//
// Two constraints found by measurement (tools/bench_gpu.cpp), not assumed:
//
//   * The driver REFUSES file-mapping views (VK_ERROR_INVALID_EXTERNAL_HANDLE), so the pages
//     cannot simply be the memory-mapped container -- they must be private committed memory,
//     which means one copy at load. VirtualAlloc regions import fine.
//   * Import stops at a total-memory ceiling (~15.9 GiB here, the sum of both heaps) rather
//     than a per-allocation one. All 60 layers plus the LM head and KV cache do not fit under
//     it at the current BIOS UMA split, so this is greedy: it takes what fits and leaves the
//     rest to the streamer.
//
// Reads are unbuffered. The bytes are going straight into memory the GPU will read, so a
// second copy of the same 17 GB in the OS page cache is waste; the container's 4096-byte
// alignment satisfies the sector alignment that FILE_FLAG_NO_BUFFERING requires.
// Can the device still accept a submission? An empty command buffer is enough: what fails when
// the layer import has taken too much is the driver's own per-submit allocation, not the work
// itself.
bool ForwardRunner::can_submit_work() {
    static bool reported = false;
    auto note = [&](const char* stage, VkResult r) {
        if (!reported) {
            reported = true;
            std::cout << "[ForwardRunner] device probe failed at " << stage
                      << " (VkResult=" << static_cast<int>(r) << ")" << std::endl;
        }
        return false;
    };

    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VkResult r = vkResetCommandBuffer(cmd_, 0);
    if (r != VK_SUCCESS) return note("vkResetCommandBuffer", r);
    r = vkBeginCommandBuffer(cmd_, &begin_info);
    if (r != VK_SUCCESS) return note("vkBeginCommandBuffer", r);
    r = vkEndCommandBuffer(cmd_);
    if (r != VK_SUCCESS) return note("vkEndCommandBuffer", r);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    r = vkQueueSubmit(vk_ctx_->compute_queue(), 1, &si, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) return note("vkQueueSubmit", r);
    r = vkQueueWaitIdle(vk_ctx_->compute_queue());
    if (r != VK_SUCCESS) return note("vkQueueWaitIdle", r);
    return true;
}

void ForwardRunner::load_resident_layers() {
    resident_layer_bufs_.assign(header_.num_layers, VkMemoryAllocation{});
    streamed_layers_.clear();

    if (!vk_ctx_->has_external_memory_host()) {
        std::cout << "[ForwardRunner] VK_EXT_external_memory_host unavailable; every layer will "
                     "be streamed." << std::endl;
        for (uint32_t l = 0; l < header_.num_layers; ++l) streamed_layers_.push_back(l);
        return;
    }

    HANDLE fh = CreateFileA(container_path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        fh = CreateFileA(container_path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    }
    if (fh == INVALID_HANDLE_VALUE) {
        for (uint32_t l = 0; l < header_.num_layers; ++l) streamed_layers_.push_back(l);
        return;
    }

    // How much may be imported.
    //
    // Importing until the driver refuses does NOT work: at 8 GB UMA it accepted 58 of 60 layers
    // (14.81 GiB) and then failed every vkQueueSubmit with VK_ERROR_OUT_OF_DEVICE_MEMORY, and
    // freeing the imports again did not restore it. VK_EXT_memory_budget cannot warn us either
    // -- it reported 12.69 GiB still free immediately after that import, because it does not
    // account for imported host memory at all.
    //
    // So imports are budgeted as if they DID debit the heaps: take what the driver reports free
    // once every fixed-size allocation is in place (KV cache, LM head, activations, streaming
    // pool), keep a reserve for the driver's own per-submit working set, and import into what
    // is left. Conservative by construction, and it self-tunes to the BIOS UMA split.
    // Reserve held back from the heap-reported free space for the driver's own per-submit
    // working set. 1.5 GiB was over-cautious and cost five resident layers; 512 MB leaves the
    // driver's natural refusal (42 layers here) as the binding limit, which is the point
    // measured to be safe.
    constexpr uint64_t kImportReserveBytes = 512ull * 1024 * 1024;
    const uint64_t free_at_start = vk_ctx_->available_device_memory();
    uint64_t import_budget =
        (vk_ctx_->has_memory_budget() && free_at_start > kImportReserveBytes)
            ? (free_at_start - kImportReserveBytes)
            : 0;

    // ...and cap it against VISIBLE SYSTEM RAM, which is what actually binds.
    //
    // Imported pages are private committed memory that the driver pins, so they are spent from
    // the OS's RAM, not from a Vulkan heap -- and exceeding what the driver can pin makes every
    // vkQueueSubmit return VK_ERROR_OUT_OF_DEVICE_MEMORY, unrecoverably. Measured on this
    // machine, with everything else already allocated:
    //
    //   BIOS UMA   visible RAM   imported     result
    //   minimum    31.3 GiB      10.74 GiB    works   (34% of RAM)
    //   8 GB       23.8 GiB       7.67 GiB    works   (32%)
    //   8 GB       23.8 GiB       7.92 GiB    FAILS   (33%)
    //   8 GB       23.8 GiB       9.71 GiB    FAILS   (41%)
    //
    // Crossing the limit is not a soft failure: submission returns VK_ERROR_OUT_OF_DEVICE_MEMORY
    // and stays broken however much is freed afterwards, so the edge cannot be found safely by
    // walking up to it -- which is why this is a static budget rather than an adaptive one.
    //
    // 38% is chosen so that on the configuration this runs on (minimum UMA, 31.3 GiB visible)
    // the DRIVER's own refusal binds first, at 46 layers / 11.75 GiB / 37.5% -- a stopping point
    // measured to be safe. It moved from 42 when the KV cache went to FP16: the driver was
    // refusing on TOTAL device memory, of which the cache is part, so halving it bought four
    // resident layers. The fraction is a backstop for configurations where the driver
    // over-permits instead, as it does at 8 GB UMA: there it accepted 58 layers and then could
    // not submit any work at all.
    //
    // The consequence is worth stating plainly: a SMALLER BIOS UMA carve-out allows MORE
    // resident layers. Imported weights are pinned system RAM, not Vulkan heap memory, so
    // giving RAM back to Windows is what buys residency -- 42 layers at minimum UMA against 26
    // at 8 GB, i.e. 1,421 ms per forward pass against 2,428 ms.
    //
    // If the probe below ever fails, lower this or set G4DENSE_MAX_RESIDENT_LAYERS.
    MEMORYSTATUSEX mem{};
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        const uint64_t ram_cap = (mem.ullTotalPhys / 100) * 38;
        if (import_budget == 0 || ram_cap < import_budget) import_budget = ram_cap;
    }

    // Leave room for a draft model if one is going to be loaded beside this one.
    if (import_reserve_bytes_ > 0) {
        import_budget = (import_budget > import_reserve_bytes_)
                            ? (import_budget - import_reserve_bytes_) : 0;
        std::cout << "[ForwardRunner] holding back "
                  << (import_reserve_bytes_ / (1024.0 * 1024.0 * 1024.0))
                  << " GiB for a draft model." << std::endl;
    }
    if (import_budget == 0) {
        std::cout << "[ForwardRunner] no import budget available; streaming every layer."
                  << std::endl;
    }

    // Escape hatch for calibrating the safe import size on a given machine. The device-memory
    // limit that actually bites here is not one Vulkan reports, so the ceiling has to be found
    // by measurement; this is how.
    uint32_t max_resident = header_.num_layers;
    if (const char* env = std::getenv("G4DENSE_MAX_RESIDENT_LAYERS")) {
        max_resident = static_cast<uint32_t>(std::max(0, std::atoi(env)));
        // Authoritative: it must be able to raise the count as well as lower it, or it cannot
        // be used to find where the limit actually is on a given machine.
        import_budget = UINT64_MAX;
        std::cout << "[ForwardRunner] G4DENSE_MAX_RESIDENT_LAYERS=" << max_resident
                  << " (RAM budget overridden)" << std::endl;
    }

    const auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t resident_bytes = 0;
    bool ceiling_hit = false;

    // WHICH layers stream matters as much as how many.
    //
    // Importing greedily from layer 0 leaves the tail streaming back to back: at 45 of 60
    // resident, layers 45-59 each waited on a ~35 ms read with only the 3-deep prefetch queue
    // (~35 ms of GPU work) to hide behind, and the pass spent 522 ms blocked on I/O.
    //
    // Spreading the streamed layers evenly puts several resident layers' compute between
    // consecutive reads, so each read overlaps work that was going to happen anyway. The
    // streamer needs no change: issue_next_stream() already runs 3 reads ahead, which now
    // spans roughly 4x as many layers.
    //
    // The plan is an estimate -- a layer that fails to import still falls through to the
    // streamed set, so a wrong guess degrades to the old behaviour rather than breaking.
    std::vector<bool> plan_streamed(header_.num_layers, false);
    {
        uint64_t budget_left = import_budget;
        uint32_t fit = 0;
        for (uint32_t l = 0; l < header_.num_layers && l < max_resident; ++l) {
            if (budget_left < header_.layer_sizes[l]) break;
            budget_left -= header_.layer_sizes[l];
            ++fit;
        }
        const uint32_t to_stream = header_.num_layers - std::min(fit, header_.num_layers);
        if (to_stream > 0) {
            // Evenly spaced: layer floor((i + 0.5) * num_layers / to_stream) for each i, which
            // puts them at the centres of `to_stream` equal buckets rather than clumped at
            // either end.
            for (uint32_t i = 0; i < to_stream; ++i) {
                const uint32_t idx = static_cast<uint32_t>(
                    ((2ull * i + 1ull) * header_.num_layers) / (2ull * to_stream));
                plan_streamed[std::min(idx, header_.num_layers - 1)] = true;
            }
        }
    }

    for (uint32_t l = 0; l < header_.num_layers; ++l) {
        const uint64_t sz = header_.layer_sizes[l];
        if (ceiling_hit || plan_streamed[l]) { streamed_layers_.push_back(l); continue; }
        if (l >= max_resident || resident_bytes + sz > import_budget) {
            ceiling_hit = true;
            streamed_layers_.push_back(l);
            continue;
        }

        void* region = VirtualAlloc(nullptr, sz, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!region) { ceiling_hit = true; streamed_layers_.push_back(l); continue; }

        LARGE_INTEGER pos;
        pos.QuadPart = static_cast<LONGLONG>(header_.layer_offsets[l]);
        SetFilePointerEx(fh, pos, nullptr, FILE_BEGIN);
        uint64_t done = 0;
        bool read_ok = true;
        while (done < sz) {
            const DWORD chunk = static_cast<DWORD>(std::min<uint64_t>(sz - done, 1u << 30));
            DWORD got = 0;
            if (!ReadFile(fh, static_cast<uint8_t*>(region) + done, chunk, &got, nullptr) || got == 0) {
                read_ok = false;
                break;
            }
            done += got;
        }
        if (!read_ok) {
            VirtualFree(region, 0, MEM_RELEASE);
            ceiling_hit = true;
            streamed_layers_.push_back(l);
            continue;
        }

        try {
            resident_layer_bufs_[l] = vk_ctx_->import_host_buffer(region, sz);
            resident_regions_.push_back(region);
            resident_bytes += sz;
        } catch (const std::exception&) {
            // Expected once the device-memory ceiling is reached. Everything from here on is
            // streamed, and the engine stays correct -- just slower for those layers.
            VirtualFree(region, 0, MEM_RELEASE);
            ceiling_hit = true;
            streamed_layers_.push_back(l);
        }
    }
    CloseHandle(fh);

    const double secs = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_start).count();
    const size_t resident_count = header_.num_layers - streamed_layers_.size();
    std::cout << "[ForwardRunner] " << resident_count << " of " << header_.num_layers
              << " layers resident (" << (resident_bytes / (1024.0 * 1024.0 * 1024.0))
              << " GiB imported in " << secs << " s); " << streamed_layers_.size()
              << " streamed per token." << std::endl;

    // Publish what the web UI displays. These were struct defaults before, so the telemetry
    // endpoint reported 0 resident layers and a max_context of 8192 while the engine capped
    // at 4096 -- worse than reporting nothing, because it looks authoritative.
    TelemetryCollector::instance().record_model_state(
        static_cast<uint32_t>(resident_count),
        static_cast<uint32_t>(streamed_layers_.size()),
        kv_cache_ ? kv_cache_->max_context() : 0,
        vk_ctx_->device_name(),
        draft_runtime_ && draft_runtime_->is_loaded(),
        0);
    {
        const auto& mp = vk_ctx_->memory_properties();
        const double free_gib = static_cast<double>(vk_ctx_->available_device_memory());
        const double h0 = mp.memoryHeapCount > 0 ? static_cast<double>(mp.memoryHeaps[0].size) : 0.0;
        const double h1 = mp.memoryHeapCount > 1 ? static_cast<double>(mp.memoryHeaps[1].size) : 0.0;
        const double mb = 1024.0 * 1024.0;
        // The driver reports free space per heap only in aggregate here, so usage is derived
        // from the total rather than attributed per heap.
        const double used = (h0 + h1 - free_gib) / mb;
        TelemetryCollector::instance().record_heap_usage(
            used * (h0 / ((h0 + h1) > 0 ? (h0 + h1) : 1.0)), h0 / mb,
            used * (h1 / ((h0 + h1) > 0 ? (h0 + h1) : 1.0)), h1 / mb);
    }

    // One check that the device can still do work. It cannot be walked back from the edge --
    // once submission returns VK_ERROR_OUT_OF_DEVICE_MEMORY, freeing the imports does not
    // restore it -- so this is a loud assertion that the budget above was conservative enough,
    // not a recovery path.
    if (!can_submit_work()) {
        throw G4DenseFormatError(
            "ForwardRunner: the device cannot accept work after importing " +
            std::to_string(resident_bytes / (1024 * 1024)) +
            " MB of layers. The import budget is too aggressive for this machine; lower the "
            "fraction of system RAM used in load_resident_layers(), or set "
            "G4DENSE_MAX_RESIDENT_LAYERS.");
    }
}

bool ForwardRunner::load_draft_model(const std::string& path) {
    if (!draft_runtime_) return false;
    if (!std::filesystem::exists(path)) {
        std::cout << "[ForwardRunner] no draft model at " << path
                  << "; speculative decoding disabled." << std::endl;
        return false;
    }
    try {
        draft_runtime_->load_model(path, vk_ctx_, tokenizer_);
        std::cout << "[ForwardRunner] draft model loaded for speculative decoding: "
                  << path << std::endl;
        TelemetryCollector::instance().record_model_state(
            static_cast<uint32_t>(header_.num_layers - streamed_layers_.size()),
            static_cast<uint32_t>(streamed_layers_.size()),
            kv_cache_ ? kv_cache_->max_context() : 0,
            vk_ctx_->device_name(), true, 0);
        return true;
    } catch (const std::exception& ex) {
        std::cout << "[ForwardRunner] draft model failed to load (" << ex.what()
                  << "); continuing without speculation." << std::endl;
        return false;
    }
}

void ForwardRunner::switch_memory_tier(uint32_t tier_id) {
    active_tier_id_ = tier_id;
    pinned_layers_.clear();

    // pinned_layers_ stays empty, whatever the tier says.
    //
    // The streaming pool is sized for the prefetch queue and nothing else: PREFETCH_DEPTH (3)
    // reads in flight plus the one being consumed is exactly its four slots. Pinning even one
    // of them leaves the queue unable to issue and plan_layers() throws "cache thrash". A
    // fallback branch here pinned three when no layer was resident, which broke the
    // stream-everything path outright.
    //
    // Nothing is lost: pinning was only ever a way to avoid re-reading a layer, and residency
    // does that better -- a pinned slot still costs a 269 MB read per token, a resident layer
    // costs nothing.
    if (streamer_) {
        streamer_->apply_tier_pinning(pinned_layers_);
    }
}

void ForwardRunner::forward_single_token(uint32_t token_id, uint32_t position, float* out_logits) {
    forward_batch(&token_id, 1, position, out_logits);
}

// One pass over the weights for `batch` consecutive positions starting at `base_position`.
//
// Prefill used to call forward_single_token() once per prompt token, so a 15-token prompt read
// all 16 GB of weights 15 times -- and 19 of the 60 layers are streamed from disk, so that was
// 15 x 859 ms of I/O alone. Running the whole prompt through one weight pass is the fix, and it
// is worth far more than the batched GEMM itself: measured, GemmInt4Batch beats M separate
// GemvInt4 dispatches by only 1.17x, while amortizing the streaming is a multiple of M.
//
// `out_logits` receives the LAST position's logits, which is all a prefill needs -- the LM head
// is 756 MB and running it for every position would cost more than it saves.
//
// batch == 1 must stay bit-identical to the old single-token path; the 4-position oracle diff
// is what enforces that.
void ForwardRunner::forward_batch(const uint32_t* token_ids, uint32_t batch,
                                  uint32_t base_position, float* out_logits,
                                  bool all_positions) {
    if (batch == 0) return;
    if (batch > kGemmMaxBatch) {
        throw G4DenseFormatError("ForwardRunner::forward_batch: batch " + std::to_string(batch) +
                                 " exceeds kGemmMaxBatch (" + std::to_string(kGemmMaxBatch) + ")");
    }
    const uint32_t token_id = token_ids[0];
    const uint32_t position = base_position;
    (void)token_id; (void)position;
    // Reclaim every descriptor set from the previous pass. Sets are allocated per dispatch
    // and never freed individually, so without this the pool runs out partway through the
    // fifth or sixth token and generation dies with "failed to allocate descriptor set".
    // Safe here because each pass is fully submitted and waited on before the next begins.
    pipeline_mgr_->reset_descriptor_pool();

    // Phase attribution. The pass is ~9.6 s against a ~232 ms bandwidth floor; without
    // knowing which phase owns that gap, every optimization is a guess.
    using clk = std::chrono::high_resolution_clock;
    const auto t_pass_start = clk::now();
    double io_ms = 0.0, gpu_ms = 0.0, lm_ms = 0.0;

    uint32_t d_model = header_.d_model;
    uint32_t vocab_size = header_.vocab_size;
    float scale = std::sqrt(static_cast<float>(d_model));

    // 1. Embedding lookup -> buf_hidden_
    uint32_t packed_cols = d_model / 8;
    uint32_t groups_per_row = d_model / 64;
    const uint8_t* embed_ptr = mapped_data_ + header_.embed_offset;
    const uint32_t* embed_w = reinterpret_cast<const uint32_t*>(embed_ptr);
    const uint16_t* embed_s = reinterpret_cast<const uint16_t*>(embed_ptr + static_cast<size_t>(vocab_size) * packed_cols * sizeof(uint32_t));
    const uint16_t* embed_b = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(embed_s) + static_cast<size_t>(vocab_size) * groups_per_row * sizeof(uint16_t));

    // Embedding lookup, one row per position, written contiguously as [position][d_model].
    float* hidden_host = static_cast<float*>(buf_hidden_.mapped_ptr);
    for (uint32_t m = 0; m < batch; ++m) {
        const uint32_t safe_token = token_ids[m] % vocab_size;
        const uint32_t* token_packed = embed_w + safe_token * packed_cols;
        const uint16_t* token_scales = embed_s + safe_token * groups_per_row;
        const uint16_t* token_biases = embed_b + safe_token * groups_per_row;
        float* dst = hidden_host + static_cast<size_t>(m) * d_model;
        for (uint32_t c = 0; c < d_model; ++c) {
            uint32_t w_idx = c / 8;
            uint32_t n_idx = c % 8;
            uint8_t q_val = (token_packed[w_idx] >> (n_idx * 4)) & 0x0F;
            uint32_t g = c / 64;
            float s = bf16_to_f32(token_scales[g]);
            float b = bf16_to_f32(token_biases[g]);
            dst[c] = (static_cast<float>(q_val) * s + b) * scale;
        }
    }

    // Per-layer embedding precompute (E2B/E4B only). Upstream:
    //
    //   per_layer_inputs     = embed_tokens_per_layer(token) * sqrt(ple_dim)
    //   per_layer_projection = per_layer_model_projection(embeds) * hidden^-0.5
    //   per_layer_projection = per_layer_projection_norm(per_layer_projection)
    //   combined             = (per_layer_projection + per_layer_inputs) * 2^-0.5
    //
    // The hidden^-0.5 scale is dropped deliberately: an RMSNorm over each ple_dim slice follows
    // it immediately, and a uniform scalar cancels exactly there.
    const uint32_t ple_rows = header_.ple_dim ? header_.num_layers * header_.ple_dim : 0;
    if (header_.ple_dim && mapped_data_ && header_.ple_offset) {
        const uint8_t* ple_base = mapped_data_ + header_.ple_offset;
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(ple_base);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(ple_base + ple_emb_w_bytes_);
        const uint16_t* pb = reinterpret_cast<const uint16_t*>(ple_base + ple_emb_w_bytes_ + ple_emb_s_bytes_);
        const uint32_t ple_packed_cols = ple_rows / 8;
        const uint32_t ple_groups = ple_rows / 64;
        const float ple_scale = std::sqrt(static_cast<float>(header_.ple_dim));

        float* dst = static_cast<float*>(buf_ple_emb_.mapped_ptr);
        for (uint32_t m = 0; m < batch; ++m) {
            const uint32_t tok = token_ids[m] % header_.ple_vocab;
            const uint32_t* rw = pw + static_cast<size_t>(tok) * ple_packed_cols;
            const uint16_t* rs = ps + static_cast<size_t>(tok) * ple_groups;
            const uint16_t* rb = pb + static_cast<size_t>(tok) * ple_groups;
            float* out = dst + static_cast<size_t>(m) * ple_rows;
            for (uint32_t c = 0; c < ple_rows; ++c) {
                const uint8_t q = (rw[c / 8] >> ((c % 8) * 4)) & 0x0F;
                out[c] = (static_cast<float>(q) * bf16_to_f32(rs[c / 64]) +
                          bf16_to_f32(rb[c / 64])) * ple_scale;
            }
        }
    }

    VkQueue queue = vk_ctx_->compute_queue();

    // GPU half of the PLE precompute: project, normalize each ple_dim slice, combine.
    if (header_.ple_dim) {
        vkResetCommandBuffer(cmd_, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd_, &bi);

        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        // per_layer_model_projection(inputs_embeds) -> buf_ple_
        {
            const bool batched = (batch > 1);
            const ComputeKernel k = batched ? ComputeKernel::GemmInt4Batch : ComputeKernel::GemvInt4;
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(k);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_ple_w_.buffer, 0, buf_ple_w_.size_bytes);
            pipeline_mgr_->update_storage_buffer(ds, 1, buf_ple_w_.buffer, 0, buf_ple_w_.size_bytes);
            pipeline_mgr_->update_storage_buffer(ds, 2, buf_ple_w_.buffer, 0, buf_ple_w_.size_bytes);
            pipeline_mgr_->update_storage_buffer(ds, 3, buf_hidden_.buffer, 0,
                                                 static_cast<uint64_t>(d_model) * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_ple_.buffer, 0,
                                                 static_cast<uint64_t>(ple_rows) * 4 * batch);
            uint32_t pc[16]{0};
            pc[0] = ple_model_proj_.rows;
            pc[1] = ple_model_proj_.in_dim;
            pc[2] = ple_model_proj_.w_off;
            pc[3] = ple_model_proj_.s_off;
            pc[4] = ple_model_proj_.b_off;
            if (batched) { pc[8] = batch; pc[9] = d_model * 4; pc[10] = ple_rows * 4; }
            pipeline_mgr_->bind_kernel(cmd_, k);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(k), 0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, k, pc, sizeof(pc));
            const uint32_t per_group = batched ? kGemmRowsPerGroup : kGemvRowsPerGroup;
            pipeline_mgr_->dispatch(cmd_, (ple_model_proj_.rows + per_group - 1) / per_group, 1, 1);
        }
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

        // per_layer_projection_norm over each ple_dim slice: one group per (position, layer).
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::RMSNormK);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_ple_.buffer, 0,
                                                 static_cast<uint64_t>(ple_rows) * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, buf_ple_norm_w_.buffer, 0, header_.ple_dim * 2);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_ple_.buffer, 0,
                                                 static_cast<uint64_t>(ple_rows) * 4 * batch);
            uint32_t pc[8]{0};
            pc[0] = header_.ple_dim;
            pc[4] = 1;
            float eps = 1e-6f;
            std::memcpy(&pc[5], &eps, 4);
            pc[6] = header_.ple_dim * 4;
            pc[7] = header_.ple_dim * 4;
            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::RMSNormK);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::RMSNormK),
                                    0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::RMSNormK, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, header_.num_layers * batch, 1, 1);
        }
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

        // combined = (projection + token_identity) * 2^-0.5
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::ResidualAccum);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_ple_.buffer, 0,
                                                 static_cast<uint64_t>(ple_rows) * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, buf_ple_emb_.buffer, 0,
                                                 static_cast<uint64_t>(ple_rows) * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_ple_.buffer, 0,
                                                 static_cast<uint64_t>(ple_rows) * 4 * batch);
            uint32_t pc[8]{0};
            pc[0] = ple_rows * batch;
            float one = 1.0f;
            float inv_sqrt2 = 0.70710678118654752f;   // per_layer_input_scale = 2^-0.5
            std::memcpy(&pc[1], &one, 4);
            std::memcpy(&pc[2], &inv_sqrt2, 4);
            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::ResidualAccum);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::ResidualAccum),
                                    0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::ResidualAccum, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, (ple_rows * batch + 255) / 256, 1, 1);
        }

        vkEndCommandBuffer(cmd_);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        const VkResult r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        if (r != VK_SUCCESS) {
            throw G4DenseFormatError("ForwardRunner: PLE precompute submit failed (VkResult=" +
                                     std::to_string(r) + ")");
        }
        vkQueueWaitIdle(queue);
    }

    // 2. Transformer layers.
    //
    // Most layers are resident: imported once at load, so the pass reads their weights in place
    // and copies nothing. Only the layers that did not fit under the device-memory ceiling are
    // streamed, and the prefetch queue runs over just those.
    std::deque<std::pair<uint32_t, LayerStreamer::LayerPlan>> inflight;
    size_t stream_cursor = 0;
    constexpr uint32_t PREFETCH_DEPTH = 3;

    auto issue_next_stream = [&]() {
        if (!streamer_ || stream_cursor >= streamed_layers_.size()) return;
        const uint32_t lid = streamed_layers_[stream_cursor++];
        LayerStreamer::LayerPlan plan = streamer_->plan_layers({static_cast<int>(lid)});
        // Issue only. The wait happens when this layer reaches the head of the queue, so the
        // read overlaps the GPU work on the layers ahead of it. Issuing and awaiting in the
        // same breath -- which is what fetch_misses() does -- serialized all of it.
        streamer_->issue_reads(plan);
        inflight.emplace_back(lid, std::move(plan));
    };
    for (uint32_t p = 0; p < PREFETCH_DEPTH; ++p) issue_next_stream();

    for (uint32_t l = 0; l < header_.num_layers; ++l) {
        // Where this layer's weights live. Resident layers cost nothing to "fetch"; streamed
        // ones come off the front of the prefetch queue, which is refilled as each is consumed.
        VkBuffer layer_buf = VK_NULL_HANDLE;
        uint64_t layer_buf_size = 0;
        LayerStreamer::LayerPlan* active_plan = nullptr;

        if (resident_layer_bufs_[l].buffer != VK_NULL_HANDLE) {
            layer_buf = resident_layer_bufs_[l].buffer;
            layer_buf_size = resident_layer_bufs_[l].size_bytes;
        } else {
            if (inflight.empty() || inflight.front().first != l) {
                throw G4DenseFormatError(
                    "ForwardRunner: streamed layer " + std::to_string(l) +
                    " is not at the head of the prefetch queue");
            }
            // push_back below may reallocate the deque's index block, but references to
            // existing elements stay valid, so holding this pointer across issue_next_stream()
            // is safe.
            active_plan = &inflight.front().second;
            // Block only for what has not already landed while earlier layers were computing.
            { const auto t0 = clk::now();
            streamer_->await_reads(*active_plan);
            io_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
            LayerSlot* slot = active_plan->slots[0];
            layer_buf = slot->buffer.buffer;
            layer_buf_size = slot->buffer.size_bytes;
            issue_next_stream();
        }

        const auto& lo = layer_gpu_offsets_[l];
        bool is_global = (header_.global_layer_mask & (1ULL << l)) != 0;
        const LayerGeometry geo = resolve_layer_geometry(header_, l);
        uint32_t head_dim = geo.head_dim;
        uint32_t q_heads = geo.q_heads;
        uint32_t kv_heads = geo.kv_heads;
        uint32_t rotated_pairs = is_global ? 64 : (head_dim / 2);
        float rope_theta = is_global ? header_.rope_theta_global : header_.rope_theta_local;
        // Gemma 4's per-layer scalar. Upstream applies it ONCE at the end of the decoder
        // layer, to the whole hidden state: `hidden_states *= self.layer_scalar`. It is folded
        // into the FFN residual add's out_scale below; the attention residual add passes 1.0.
        //
        // Do not apply it to the residual branches (the engine did, attenuating each layer ~11x
        // twice over) and do not drop it (an earlier round did, letting the residual stream
        // grow unbounded to rms ~50 by layer 59).
        float layer_scalar = lo.layer_scalar;

        // Reset and begin command buffer
        vkResetCommandBuffer(cmd_, 0);
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd_, &begin_info);

        // A. Input RMSNorm: hidden -> norm_buf
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::RMSNormK);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_hidden_.buffer, 0, d_model * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_size);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_norm_.buffer, 0, d_model * 4 * batch);

            uint32_t pc[8]{0};
            pc[0] = d_model;
            pc[1] = 0;              // x_byte_off
            pc[2] = lo.in_norm_off; // w_byte_off
            pc[3] = 0;              // out_byte_off
            pc[4] = 1;              // has_weight
            float eps = 1e-6f;
            std::memcpy(&pc[5], &eps, 4);

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::RMSNormK);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::RMSNormK),
                                    0, 1, &ds, 0, nullptr);
            // One threadgroup per position; gp1.z/gp1.w are the per-position strides.
            pc[6] = d_model * 4;
            pc[7] = d_model * 4;
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::RMSNormK, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, batch, 1, 1);
        }

        // Barrier: norm_buf ready
        VkMemoryBarrier mem_bar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mem_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mem_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // B. Q, K, V Projections via GemvInt4
        // Projections. At batch == 1 this is the original GemvInt4 dispatch, unchanged; above
        // that it switches to GemmInt4Batch, which reads each weight row once for all
        // positions. Activations are [position][dim], so the strides are just the dimensions.
        auto dispatch_gemv = [&](const LayerOffsetsGPU::ProjOffsets& p, VkBuffer in_buf, VkBuffer out_buf) {
            const bool batched = (batch > 1);
            const ComputeKernel k = batched ? ComputeKernel::GemmInt4Batch : ComputeKernel::GemvInt4;

            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(k);
            pipeline_mgr_->update_storage_buffer(ds, 0, layer_buf, 0, layer_buf_size);
            pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_size);
            pipeline_mgr_->update_storage_buffer(ds, 2, layer_buf, 0, layer_buf_size);
            pipeline_mgr_->update_storage_buffer(ds, 3, in_buf, 0, static_cast<uint64_t>(p.in_dim) * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 6, out_buf, 0, static_cast<uint64_t>(p.rows) * 4 * batch);

            uint32_t pc[16]{0};
            pc[0] = p.rows;
            pc[1] = p.in_dim;
            pc[2] = p.w_off;
            pc[3] = p.s_off;
            pc[4] = p.b_off;
            pc[5] = 0; // x_off
            pc[6] = 0; // out_off
            pc[7] = 0; // row_base
            if (batched) {
                pc[8]  = batch;
                pc[9]  = p.in_dim * 4;   // x stride between positions
                pc[10] = p.rows * 4;     // out stride between positions
            }

            pipeline_mgr_->bind_kernel(cmd_, k);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(k), 0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, k, pc, sizeof(pc));
            const uint32_t per_group = batched ? kGemmRowsPerGroup : kGemvRowsPerGroup;
            pipeline_mgr_->dispatch(cmd_, (p.rows + per_group - 1) / per_group, 1, 1);
        };

        // A KV-sharing layer has no k_proj/v_proj at all: it reads the cache its donor built.
        dispatch_gemv(lo.q_proj, buf_norm_.buffer, buf_q_.buffer);
        if (!lo.kv_shared) {
            dispatch_gemv(lo.k_proj, buf_norm_.buffer, buf_k_.buffer);
            if (lo.v_proj.rows != 0) {
                dispatch_gemv(lo.v_proj, buf_norm_.buffer, buf_v_.buffer);
            }
        }

        // Barrier: Q, K (and non-global V) Gemvs complete before Epilogues
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        if (!lo.kv_shared) {
            // V norm. Upstream (Gemma4TextAttention.forward, modular_gemma4.py):
            //
            //   value_states = v_proj(x) if v_proj is not None else key_states
            //   key_states   = k_norm(key_states);  key_states = apply_rotary_pos_emb(...)
            //   value_states = v_norm(value_states)
            //
            // v_norm is `Gemma4RMSNorm(head_dim, with_scale=False)` -- unweighted -- so
            // has_weight = 0, and values are never rotated, so do_rope = 0.
            //
            // It runs on EVERY layer: `is_kv_shared_layer` is the only thing that skips it and
            // it is false throughout, because this config sets num_kv_shared_layers = 0. The
            // engine used to normalize V only on the 10 full-attention layers and hand the 50
            // sliding layers a raw v_proj output.
            //
            // Full-attention layers additionally have no v_proj (attention_k_eq_v), so V is
            // read from the RAW k_proj output. Python rebinds `key_states` when k_norm returns
            // a new tensor, so V never sees k_norm -- and running this block BEFORE the Q/K
            // epilogue is what keeps buf_k_ raw at this point. A round-4 change applied k_norm
            // here instead; that was wrong.
            // V comes from v_proj when the layer has one, and otherwise aliases the RAW
            // k_proj output (attention_k_eq_v). That is a property of the TENSOR being
            // present, not of the layer being full-attention: the 31B omits v_proj on its
            // global layers, E2B does not. Keying this off `is_global` broke E2B at layer 4,
            // its first full-attention layer, while layers 0-3 matched to 1e-3.
            VkBuffer v_src = (lo.v_proj.rows != 0) ? buf_v_.buffer : buf_k_.buffer;

            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::QKVEpilogue);
            pipeline_mgr_->update_storage_buffer(ds, 0, v_src, 0, kv_heads * head_dim * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_size);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_v_.buffer, 0, kv_heads * head_dim * 4 * batch);

            uint32_t pc[16]{0};   // gp0..gp3: the batched kernels read gp3, so it must be pushed
            pc[0] = head_dim;
            pc[1] = kv_heads;
            pc[2] = 0; // in_off (gp0.z)
            pc[3] = 0; // out_off (gp0.w)
            float eps = 1e-6f;
            std::memcpy(&pc[4], &eps, 4); // eps_bits (gp1.x)
            pc[5] = 0;                    // w_off (gp1.y) -- unused, v_norm is unweighted
            pc[6] = 0;                    // has_weight = 0 (gp1.z) -- v_norm has no weight
            pc[7] = 0;                    // do_rope = 0 (gp1.w) -- values are not rotated

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::QKVEpilogue);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::QKVEpilogue),
                                    0, 1, &ds, 0, nullptr);
            pc[12] = kv_heads * head_dim * 4;   // gp3.x: input stride per position
            pc[13] = kv_heads * head_dim * 4;   // gp3.y: output stride per position
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::QKVEpilogue, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, kv_heads * batch, 1, 1);

            // On global layers this read raw buf_k_, so it must complete before the K epilogue
            // below overwrites buf_k_ with k_norm + RoPE.
            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);
        }

        // C. Q/K Epilogue (RMSNorm + RoPE)
        // Q Norm & RoPE
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::QKVEpilogue);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_q_.buffer, 0, q_heads * head_dim * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_size);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_q_.buffer, 0, q_heads * head_dim * 4 * batch);

            uint32_t pc[16]{0};   // gp0..gp3: the batched kernels read gp3, so it must be pushed
            pc[0] = head_dim;
            pc[1] = q_heads;
            pc[2] = 0; // in_off (gp0.z)
            pc[3] = 0; // out_off (gp0.w)
            float eps = 1e-6f;
            std::memcpy(&pc[4], &eps, 4); // eps_bits (gp1.x)
            pc[5] = lo.q_norm_off;        // w_off (gp1.y)
            pc[6] = 1;                    // has_weight (gp1.z)
            pc[7] = 1;                    // do_rope (gp1.w)
            pc[8] = rotated_pairs;        // rotated_pairs (gp2.x)
            pc[9] = position;             // position (gp2.y)
            std::memcpy(&pc[10], &rope_theta, 4); // theta_bits (gp2.z)

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::QKVEpilogue);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::QKVEpilogue),
                                    0, 1, &ds, 0, nullptr);
            pc[12] = q_heads * head_dim * 4;    // gp3.x: input stride per position
            pc[13] = q_heads * head_dim * 4;    // gp3.y: output stride per position
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::QKVEpilogue, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, q_heads * batch, 1, 1);
        }

        // K Norm & RoPE -- absent on KV-sharing layers, which have no k_norm either.
        if (!lo.kv_shared) {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::QKVEpilogue);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_k_.buffer, 0, kv_heads * head_dim * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_size);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_k_.buffer, 0, kv_heads * head_dim * 4 * batch);

            uint32_t pc[16]{0};   // gp0..gp3: the batched kernels read gp3, so it must be pushed
            pc[0] = head_dim;
            pc[1] = kv_heads;
            pc[2] = 0; // in_off (gp0.z)
            pc[3] = 0; // out_off (gp0.w)
            float eps = 1e-6f;
            std::memcpy(&pc[4], &eps, 4); // eps_bits (gp1.x)
            pc[5] = lo.k_norm_off;        // w_off (gp1.y)
            pc[6] = 1;                    // has_weight (gp1.z)
            pc[7] = 1;                    // do_rope (gp1.w)
            pc[8] = rotated_pairs;        // rotated_pairs (gp2.x)
            pc[9] = position;             // position (gp2.y)
            std::memcpy(&pc[10], &rope_theta, 4); // theta_bits (gp2.z)

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::QKVEpilogue);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::QKVEpilogue),
                                    0, 1, &ds, 0, nullptr);
            pc[12] = kv_heads * head_dim * 4;   // gp3.x: input stride per position
            pc[13] = kv_heads * head_dim * 4;   // gp3.y: output stride per position
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::QKVEpilogue, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, kv_heads * batch, 1, 1);
        }

        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // D. Append K and V into the KV cache ring, one slot per position in the batch.
        //
        // Separate copies rather than one contiguous copy: the ring wraps at kv_cap, so a batch
        // straddling the wrap point is not contiguous in the destination.
        uint32_t kv_cap = kv_cache_->layer_capacity(l);
        const uint32_t kv_n = kv_heads * head_dim;              // elements per position
        const uint64_t kv_src_stride = static_cast<uint64_t>(kv_n) * 4;   // FP32 scratch
        const uint64_t kv_stride = static_cast<uint64_t>(kv_n) * 2;       // FP16 cache slot

        if (!lo.kv_shared) {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::KVWrite);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_k_.buffer, 0, kv_src_stride * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, buf_v_.buffer, 0, kv_src_stride * batch);
            pipeline_mgr_->update_storage_buffer(ds, 6, kv_cache_->k_buffer(l), 0,
                                                 static_cast<uint64_t>(kv_cap) * kv_stride);
            pipeline_mgr_->update_storage_buffer(ds, 7, kv_cache_->v_buffer(l), 0,
                                                 static_cast<uint64_t>(kv_cap) * kv_stride);

            uint32_t pc[8]{0};
            pc[0] = kv_n;
            pc[1] = batch;
            pc[2] = base_position;
            pc[3] = kv_cap;
            pc[4] = static_cast<uint32_t>(kv_src_stride);
            pc[5] = static_cast<uint32_t>(kv_stride);

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::KVWrite);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::KVWrite),
                                    0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::KVWrite, pc, sizeof(pc));
            const uint32_t kvw_total = (kv_n / 2) * batch;
            pipeline_mgr_->dispatch(cmd_, (kvw_total + 255) / 256, 1, 1);
        }

        // KVWrite is a compute dispatch now, not a copy, so this is a shader-write barrier.
        VkMemoryBarrier xfer_bar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        xfer_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        xfer_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &xfer_bar, 0, nullptr, 0, nullptr);

        // E. Attention
        uint32_t first_pos = (position >= kv_cap && !is_global) ? (position - kv_cap + 1) : 0;
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::Attention);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_q_.buffer, 0, q_heads * head_dim * 4 * batch);
            // Shared layers read the donor layer's cache. Donor type always matches, so the
            // geometry and capacity line up and nothing has to be copied.
            const uint32_t kv_src = lo.kv_donor;
            pipeline_mgr_->update_storage_buffer(ds, 1, kv_cache_->k_buffer(kv_src), 0, static_cast<uint64_t>(kv_cap) * kv_stride);
            pipeline_mgr_->update_storage_buffer(ds, 2, kv_cache_->v_buffer(kv_src), 0, static_cast<uint64_t>(kv_cap) * kv_stride);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_attn_out_.buffer, 0, q_heads * head_dim * 4 * batch);

            uint32_t pc[16]{0};   // gp0..gp3: the batched kernels read gp3, so it must be pushed
            pc[0] = q_heads;
            pc[1] = kv_heads;
            pc[2] = head_dim;
            pc[3] = position + 1;
            pc[4] = first_pos;
            pc[9] = kv_cap;
            // gp2.z: attention scale = head_dim^-0.5, per layer (1/16 for the 50 sliding
            // layers at head_dim 256, 1/sqrt(512) for the 10 global layers at 512). This was
            // hardcoded to 1.0 in the kernel on the inherited assumption that q_norm absorbs
            // it; this checkpoint's q_norm is a constant 1.8779, so it does not.
            // Gemma 4 sets self.scaling = 1.0 in Gemma4TextAttention -- NOT head_dim^-0.5.
            // Verified against transformers/models/gemma4/modular_gemma4.py. Passed as a push
            // constant rather than baked into the kernel so it stays a property of the model,
            // and so the parity test can prove the kernel honours a non-unit value.
            float attn_scale = 1.0f;
            std::memcpy(&pc[10], &attn_scale, 4);

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::Attention);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::Attention),
                                    0, 1, &ds, 0, nullptr);
            pc[12] = batch;                       // gp3.x
            pc[13] = q_heads * head_dim * 4;      // gp3.y: query stride per position
            pc[14] = q_heads * head_dim * 4;      // gp3.z: output stride per position
            // gp3.w: the sliding window, so each query in the batch can compute its own causal
            // start. Zero on full-attention layers, which see everything.
            pc[15] = is_global ? 0u : kv_cap;
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::Attention, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, q_heads * batch, 1, 1);
        }

        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // F. O_Proj: attn_out -> proj_out
        dispatch_gemv(lo.o_proj, buf_attn_out_.buffer, buf_proj_out_.buffer);

        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // G. Post-Attn Norm: proj_out -> proj_out
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::RMSNormK);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_proj_out_.buffer, 0, d_model * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_size);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_proj_out_.buffer, 0, d_model * 4 * batch);

            uint32_t pc[8]{0};
            pc[0] = d_model;
            pc[1] = 0;                  // x_byte_off
            pc[2] = lo.post_attn_norm_off; // w_byte_off
            pc[3] = 0;                  // out_byte_off
            pc[4] = 1;
            float eps = 1e-6f;
            std::memcpy(&pc[5], &eps, 4);

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::RMSNormK);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::RMSNormK),
                                    0, 1, &ds, 0, nullptr);
            // One threadgroup per position; gp1.z/gp1.w are the per-position strides.
            pc[6] = d_model * 4;
            pc[7] = d_model * 4;
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::RMSNormK, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, batch, 1, 1);
        }

        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // H. Residual Addition: hidden = hidden + attn_out   (unscaled)
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::ResidualAccum);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_hidden_.buffer, 0, d_model * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, buf_proj_out_.buffer, 0, d_model * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_hidden_.buffer, 0, d_model * 4 * batch);

            uint32_t pc[8]{0};
            pc[0] = d_model * batch;
            float res_scale = 1.0f;
            std::memcpy(&pc[1], &res_scale, 4);   // res_scale
            float attn_out_scale = 1.0f;          // layer_scalar is applied once, at layer end
            std::memcpy(&pc[2], &attn_out_scale, 4);

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::ResidualAccum);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::ResidualAccum),
                                    0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::ResidualAccum, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, (d_model * batch + 255) / 256, 1, 1);
        }

        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // I. Pre-FFN Norm: hidden -> norm_buf
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::RMSNormK);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_hidden_.buffer, 0, d_model * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_size);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_norm_.buffer, 0, d_model * 4 * batch);

            uint32_t pc[8]{0};
            pc[0] = d_model;
            pc[1] = 0;                 // x_byte_off
            pc[2] = lo.pre_ffn_norm_off; // w_byte_off
            pc[3] = 0;                 // out_byte_off
            pc[4] = 1;
            float eps = 1e-6f;
            std::memcpy(&pc[5], &eps, 4);

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::RMSNormK);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::RMSNormK),
                                    0, 1, &ds, 0, nullptr);
            // One threadgroup per position; gp1.z/gp1.w are the per-position strides.
            pc[6] = d_model * 4;
            pc[7] = d_model * 4;
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::RMSNormK, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, batch, 1, 1);
        }

        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // J. FFN Gate & Up Projections
        dispatch_gemv(lo.gate_proj, buf_norm_.buffer, buf_gate_.buffer);
        dispatch_gemv(lo.up_proj, buf_norm_.buffer, buf_up_.buffer);

        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // K. GeGLU
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::GeGLU);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_gate_.buffer, 0, lo.gate_proj.rows * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, buf_up_.buffer, 0, lo.gate_proj.rows * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_gate_.buffer, 0, lo.gate_proj.rows * 4 * batch);

            uint32_t pc[8]{0};
            pc[0] = lo.gate_proj.rows * batch;

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::GeGLU);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::GeGLU),
                                    0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::GeGLU, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, (lo.gate_proj.rows * batch + 255) / 256, 1, 1);
        }

        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // L. FFN Down Projection
        dispatch_gemv(lo.down_proj, buf_gate_.buffer, buf_ffn_out_.buffer);

        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // M. Post-FFN Norm: ffn_out -> ffn_out
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::RMSNormK);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_ffn_out_.buffer, 0, d_model * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_size);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_ffn_out_.buffer, 0, d_model * 4 * batch);

            uint32_t pc[8]{0};
            pc[0] = d_model;
            pc[1] = 0;                  // x_byte_off
            pc[2] = lo.post_ffn_norm_off; // w_byte_off
            pc[3] = 0;                  // out_byte_off
            pc[4] = 1;
            float eps = 1e-6f;
            std::memcpy(&pc[5], &eps, 4);

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::RMSNormK);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::RMSNormK),
                                    0, 1, &ds, 0, nullptr);
            // One threadgroup per position; gp1.z/gp1.w are the per-position strides.
            pc[6] = d_model * 4;
            pc[7] = d_model * 4;
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::RMSNormK, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, batch, 1, 1);
        }

        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // M. Residual Addition: hidden = (hidden + ffn_out) * layer_scalar   (layer end)
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::ResidualAccum);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_hidden_.buffer, 0, d_model * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 1, buf_ffn_out_.buffer, 0, d_model * 4 * batch);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_hidden_.buffer, 0, d_model * 4 * batch);

            uint32_t pc[8]{0};
            pc[0] = d_model * batch;
            float res_scale = 1.0f;
            std::memcpy(&pc[1], &res_scale, 4);   // res_scale
            // out_scale carries layer_scalar: upstream does `hidden_states *= self.layer_scalar`
            // once at the END of the decoder layer. Folding it into this add is exact ONLY
            // when the per-layer-input block is absent -- on a PLE model that block sits
            // between the two, so there the scale moves to its residual add instead.
            const float ffn_out_scale = header_.ple_dim ? 1.0f : layer_scalar;
            std::memcpy(&pc[2], &ffn_out_scale, 4);

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::ResidualAccum);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::ResidualAccum),
                                    0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::ResidualAccum, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, (d_model * batch + 255) / 256, 1, 1);
        }

        // N. Per-layer input block (E2B/E4B). Upstream, after the FFN residual add and before
        // layer_scalar:
        //
        //   residual = h
        //   h = per_layer_input_gate(h);  h = gelu(h);  h = h * per_layer_input[l]
        //   h = per_layer_projection(h);  h = post_per_layer_input_norm(h)
        //   h = residual + h
        //
        // gelu(gate) * per_layer_input is exactly what GeGLU computes, so no new kernel is
        // needed -- the combined tensor is passed as its `up` operand, strided by layer.
        if (header_.ple_dim) {
            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            dispatch_gemv(lo.ple_gate, buf_hidden_.buffer, buf_ple_g_.buffer);

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::GeGLU);
                pipeline_mgr_->update_storage_buffer(ds, 0, buf_ple_g_.buffer, 0,
                                                     static_cast<uint64_t>(header_.ple_dim) * 4 * batch);
                pipeline_mgr_->update_storage_buffer(ds, 1, buf_ple_.buffer, 0,
                                                     static_cast<uint64_t>(ple_rows) * 4 * batch);
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_ple_g_.buffer, 0,
                                                     static_cast<uint64_t>(header_.ple_dim) * 4 * batch);
                uint32_t pc[8]{0};
                pc[0] = header_.ple_dim;
                pc[2] = l * header_.ple_dim * 4;      // this layer's slice of `combined`
                pc[4] = batch;
                pc[5] = header_.ple_dim * 4;          // gate stride per position
                pc[6] = ple_rows * 4;                 // combined stride per position
                pc[7] = header_.ple_dim * 4;          // out stride per position
                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::GeGLU);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::GeGLU),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::GeGLU, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, (header_.ple_dim * batch + 255) / 256, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            dispatch_gemv(lo.ple_proj, buf_ple_g_.buffer, buf_ffn_out_.buffer);

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::RMSNormK);
                pipeline_mgr_->update_storage_buffer(ds, 0, buf_ffn_out_.buffer, 0, d_model * 4 * batch);
                pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_size);
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_ffn_out_.buffer, 0, d_model * 4 * batch);
                uint32_t pc[8]{0};
                pc[0] = d_model;
                pc[2] = lo.post_ple_norm_off;
                pc[4] = 1;
                float eps = 1e-6f;
                std::memcpy(&pc[5], &eps, 4);
                pc[6] = d_model * 4;
                pc[7] = d_model * 4;
                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::RMSNormK);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::RMSNormK),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::RMSNormK, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, batch, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::ResidualAccum);
                pipeline_mgr_->update_storage_buffer(ds, 0, buf_hidden_.buffer, 0, d_model * 4 * batch);
                pipeline_mgr_->update_storage_buffer(ds, 1, buf_ffn_out_.buffer, 0, d_model * 4 * batch);
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_hidden_.buffer, 0, d_model * 4 * batch);
                uint32_t pc[8]{0};
                pc[0] = d_model * batch;
                float one = 1.0f;
                std::memcpy(&pc[1], &one, 4);
                // layer_scalar lands here on a PLE model -- this add is the end of the layer.
                std::memcpy(&pc[2], &layer_scalar, 4);
                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::ResidualAccum);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::ResidualAccum),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::ResidualAccum, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, (d_model * batch + 255) / 256, 1, 1);
            }
        }

        vkEndCommandBuffer(cmd_);

        // Submit GPU Layer Execution
        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_;
        // Check the result. An unchecked submit that fails looks exactly like a submit that
        // succeeded instantly: the queue is idle, every vkQueueWaitIdle returns at once, and the
        // pass reports near-zero GPU time while producing empty output. That is precisely how
        // an over-greedy layer import presented itself.
        {
            const VkResult sres = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
            if (sres != VK_SUCCESS) {
                is_generating_ = false;
                throw G4DenseFormatError("ForwardRunner: vkQueueSubmit failed (VkResult=" +
                                         std::to_string(sres) + ")");
            }
        }
        { const auto t0 = clk::now();
        vkQueueWaitIdle(queue);
        gpu_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

        // Optional per-layer readback, mirroring the CPU reference's --dump-tensors so the two
        // can be diffed layer by layer. Set G4DENSE_GPU_DUMP_DIR to enable. buf_hidden_ is
        // host-visible and the queue has just been drained, so the mapped pointer is current.
        //
        // This is what localizes a GPU-only defect: the whole stack matches the oracle at
        // position 0 while later positions diverge, and only a per-layer diff at a diverging
        // position says which layer first goes wrong.
        if (const char* dump_dir = std::getenv("G4DENSE_GPU_DUMP_DIR")) {
            std::filesystem::create_directories(dump_dir);
            // One file per position in the batch, so a batched pass can be diffed against the
            // CPU oracle's per-position dumps layer by layer.
            const char* hp = static_cast<const char*>(buf_hidden_.mapped_ptr);
            for (uint32_t m = 0; m < batch; ++m) {
                const std::string name = std::string(dump_dir) + "/gpu_token" +
                                         std::to_string(base_position + m) + "_layer" +
                                         std::to_string(l) + "_hidden.bin";
                std::ofstream f(name, std::ios::binary);
                f.write(hp + static_cast<size_t>(m) * d_model * 4, d_model * 4);
            }
        }

        if (active_plan) {
            streamer_->release_plan(*active_plan);
            inflight.pop_front();
        }
    }

    // 3. Final RMSNorm: hidden -> norm_buf
    {
        vkResetCommandBuffer(cmd_, 0);
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd_, &begin_info);

        VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::RMSNormK);
        pipeline_mgr_->update_storage_buffer(ds, 0, buf_hidden_.buffer, 0, d_model * 4 * batch);
        pipeline_mgr_->update_storage_buffer(ds, 1, buf_final_norm_w_.buffer, 0, d_model * 2);
        pipeline_mgr_->update_storage_buffer(ds, 6, buf_norm_.buffer, 0, d_model * 4 * batch);

        // Normally only the LAST position's hidden state is normed and sent to the LM head:
        // prefill only needs the logits that follow the prompt, and the head is a 756 MB read.
        // Speculative verification is the exception -- it needs every position.
        uint32_t pc[8]{0};
        pc[0] = d_model;
        pc[1] = all_positions ? 0u : (batch - 1) * d_model * 4; // x_off
        pc[2] = 0; // w_off
        pc[3] = 0; // out_off
        pc[4] = 1; // has_weight
        float eps = 1e-6f;
        std::memcpy(&pc[5], &eps, 4);

        pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::RMSNormK);
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_mgr_->get_pipeline_layout(ComputeKernel::RMSNormK),
                                0, 1, &ds, 0, nullptr);
        if (all_positions) { pc[6] = d_model * 4; pc[7] = d_model * 4; }
        pipeline_mgr_->push_constants(cmd_, ComputeKernel::RMSNormK, pc, sizeof(pc));
        pipeline_mgr_->dispatch(cmd_, all_positions ? batch : 1, 1, 1);

        vkEndCommandBuffer(cmd_);

        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_;
        // Check the result. An unchecked submit that fails looks exactly like a submit that
        // succeeded instantly: the queue is idle, every vkQueueWaitIdle returns at once, and the
        // pass reports near-zero GPU time while producing empty output. That is precisely how
        // an over-greedy layer import presented itself.
        {
            const VkResult sres = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
            if (sres != VK_SUCCESS) {
                is_generating_ = false;
                throw G4DenseFormatError("ForwardRunner: vkQueueSubmit failed (VkResult=" +
                                         std::to_string(sres) + ")");
            }
        }
        { const auto t0 = clk::now();
        vkQueueWaitIdle(queue);
        gpu_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
    }

    const auto t_lm_start = clk::now();
    // 4. LM Head (tied INT4 head: GPU GEMV + softcap)
    //
    // This was a scalar CPU loop over vocab_size x d_model = 1.41e9 multiply-adds per token.
    // Measured at ~5 cores pegged, it was the dominant term in a 23.6 s forward pass -- while
    // LMHeadGreedy.hlsl sat compiled and unused. GemvInt4 and Softcap are both already
    // verified against the CPU reference by test_gpu_kernels, so this reuses parity-checked
    // numerics rather than introducing new ones.
    //
    // Full logits are produced (not the fused greedy argmax) because sampling, speculative
    // verification, and the oracle diff all need the distribution.
    {
        vkResetCommandBuffer(cmd_, 0);
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd_, &begin_info);

        // 4a. logits[v] = dot(lm_head_row_v, normed_hidden)
        {
            // Batched when every position's logits are wanted: the head is 756 MB, so running
            // it once per position would cost more than the layers did.
            const uint32_t lm_batch = all_positions ? batch : 1u;
            const bool lm_batched = (lm_batch > 1);
            const ComputeKernel lmk = lm_batched ? ComputeKernel::GemmInt4Batch : ComputeKernel::GemvInt4;

            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(lmk);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_lm_head_.buffer, 0, lm_head_w_bytes_);
            pipeline_mgr_->update_storage_buffer(ds, 1, buf_lm_head_.buffer,
                                                 lm_head_w_bytes_, lm_head_s_bytes_);
            pipeline_mgr_->update_storage_buffer(ds, 2, buf_lm_head_.buffer,
                                                 lm_head_w_bytes_ + lm_head_s_bytes_, lm_head_s_bytes_);
            pipeline_mgr_->update_storage_buffer(ds, 3, buf_norm_.buffer, 0,
                                                 static_cast<uint64_t>(d_model) * 4 * lm_batch);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_logits_.buffer, 0,
                                                 static_cast<uint64_t>(vocab_size) * 4 * lm_batch);

            // Byte offsets are all zero: each binding is already a sub-range view.
            uint32_t pc[16]{0};
            pc[0] = vocab_size;   // rows
            pc[1] = d_model;      // in_dim
            if (lm_batched) {
                pc[8]  = lm_batch;
                pc[9]  = d_model * 4;
                pc[10] = vocab_size * 4;
            }

            pipeline_mgr_->bind_kernel(cmd_, lmk);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(lmk), 0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, lmk, pc, sizeof(pc));
            const uint32_t lm_per_group = lm_batched ? kGemmRowsPerGroup : kGemvRowsPerGroup;
            pipeline_mgr_->dispatch(cmd_, (vocab_size + lm_per_group - 1) / lm_per_group, 1, 1);
        }

        // The softcap reads what the GEMV just wrote.
        VkMemoryBarrier bar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &bar, 0, nullptr, 0, nullptr);

        // 4b. z' = cap * tanh(z / cap), in place.
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::Softcap);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_logits_.buffer, 0,
                                                 static_cast<uint64_t>(vocab_size) * 4 * (all_positions ? batch : 1u));
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_logits_.buffer, 0,
                                                 static_cast<uint64_t>(vocab_size) * 4 * (all_positions ? batch : 1u));

            uint32_t pc[8]{0};
            pc[0] = vocab_size * (all_positions ? batch : 1u);
            float cap = header_.final_logit_softcapping;
            std::memcpy(&pc[3], &cap, 4);

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::Softcap);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::Softcap),
                                    0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::Softcap, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, (vocab_size * (all_positions ? batch : 1u) + 255) / 256, 1, 1);
        }

        vkEndCommandBuffer(cmd_);

        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_;
        // Check the result. An unchecked submit that fails looks exactly like a submit that
        // succeeded instantly: the queue is idle, every vkQueueWaitIdle returns at once, and the
        // pass reports near-zero GPU time while producing empty output. That is precisely how
        // an over-greedy layer import presented itself.
        {
            const VkResult sres = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
            if (sres != VK_SUCCESS) {
                is_generating_ = false;
                throw G4DenseFormatError("ForwardRunner: vkQueueSubmit failed (VkResult=" +
                                         std::to_string(sres) + ")");
            }
        }
        { const auto t0 = clk::now();
        vkQueueWaitIdle(queue);
        gpu_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
    }

    if (out_logits) {
        std::memcpy(out_logits, buf_logits_.mapped_ptr,
                    static_cast<size_t>(vocab_size) * 4 * (all_positions ? batch : 1u));
    }

    lm_ms = std::chrono::duration<double, std::milli>(clk::now() - t_lm_start).count();
    const double total_ms =
        std::chrono::duration<double, std::milli>(clk::now() - t_pass_start).count();
    // gpu_ms already includes the LM head's own queue wait, so subtract it out of the
    // 'other' bucket rather than double-counting.
    const double cpu_other_ms = total_ms - io_ms - gpu_ms;
    TelemetryCollector::instance().record_phase_breakdown(io_ms, gpu_ms, lm_ms,
                                                          cpu_other_ms > 0 ? cpu_other_ms : 0.0);
}

void ForwardRunner::generate(const std::string& prompt,
                             const GenerationOptions& options,
                             TokenCallback on_token,
                             std::atomic<bool>* cancel_flag) {
    is_generating_ = true;
    auto start_time = std::chrono::high_resolution_clock::now();

    if (!tokenizer_ || !tokenizer_->is_loaded()) {
        is_generating_ = false;
        throw std::runtime_error("ForwardRunner::generate: Tokenizer is not loaded or missing");
    }

    // Each generate() call is an independent conversation starting at position 0, so the KV
    // cache must not carry the previous one's keys and values. Without this, the second and
    // later calls attend over a mixture of two conversations.
    if (kv_cache_) {
        kv_cache_->reset();
    }

    // Instruction-tuned checkpoints need the turn framing. apply_chat_template() emits the
    // literal "<bos>" and the <|turn> / <turn|> markers, and encode() peels added tokens out
    // of the text verbatim, so add_bos must be false here or the sequence gets two.
    std::vector<uint32_t> prompt_tokens;
    if (options.use_chat_template) {
        Tokenizer::ChatMessage msg{"user", prompt};
        prompt_tokens = tokenizer_->encode(tokenizer_->apply_chat_template({msg}), false);
    } else {
        prompt_tokens = tokenizer_->encode(prompt, true);
    }
    if (prompt_tokens.empty()) {
        is_generating_ = false;
        throw std::runtime_error("ForwardRunner::generate: prompt encoded to empty token sequence");
    }

    for (auto& t : prompt_tokens) {
        if (t >= header_.vocab_size) {
            if (header_.vocab_size < 262144) {
                t = t % header_.vocab_size;
            } else {
                is_generating_ = false;
                throw std::runtime_error("ForwardRunner::generate: token id " + std::to_string(t) +
                                         " exceeds model vocab size " + std::to_string(header_.vocab_size));
            }
        }
    }

    auto is_stop = [this](uint32_t tok) -> bool {
        if (tokenizer_) {
            for (uint32_t stop_id : tokenizer_->stop_token_ids()) {
                if (tok == stop_id) return true;
            }
        }
        return false;
    };

    std::vector<uint32_t> history = prompt_tokens;
    std::vector<float> logits(header_.vocab_size);

    // Prefill the prompt in batches, one weight pass per chunk instead of one per token.
    //
    // This was the single largest end-to-end cost: a 15-token prompt ran 15 full passes over
    // all 60 layers, and 19 of those layers stream from disk at ~859 ms per pass. Batching
    // amortizes both the streaming and the weight reads across the whole chunk -- measured at
    // 532 ms per position against 1,456 ms sequential.
    for (size_t i = 0; i < prompt_tokens.size(); i += kGemmMaxBatch) {
        const uint32_t n = static_cast<uint32_t>(
            std::min<size_t>(kGemmMaxBatch, prompt_tokens.size() - i));
        forward_batch(prompt_tokens.data() + i, n, static_cast<uint32_t>(i), logits.data());
    }

    auto ttft_time = std::chrono::high_resolution_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(ttft_time - start_time).count();
    TelemetryCollector::instance().record_ttft(ttft_ms);

    // Speculative decoding, when a draft model is loaded.
    //
    // The draft proposes K tokens; the target verifies them all in ONE weight pass, and every
    // token whose argmax matches is accepted for free.
    //
    // The verify batch leads with the token that has not yet been through the target -- the
    // previous round's bonus -- rather than with the first draft. That is what keeps the round
    // to a SINGLE target pass: verify[i] is the distribution after that token plus i drafts,
    // which is exactly what is needed to check draft i+1. Leading with the first draft instead
    // costs a second full pass per round to catch the target up, and measured 1.74x SLOWER
    // than not speculating at all.
    //
    // Rejected drafts leave stale entries in both KV caches past the accepted length. They are
    // never read -- attention only looks at [first, current_position) -- and the next pass
    // overwrites those ring slots, so there is no rollback step.
    if (options.speculative_enabled && draft_runtime_ && draft_runtime_->is_loaded() &&
        speculator_ && options.draft_k > 0) {
        const uint32_t vocab_size = header_.vocab_size;
        const uint32_t K = std::min<uint32_t>(options.draft_k, kGemmMaxBatch);
        std::vector<float> verify(static_cast<size_t>(vocab_size) * K);
        std::vector<uint32_t> batch_tokens;
        batch_tokens.reserve(K);
        int emitted = 0;

        // The token the target has not processed yet. history's last entry has been forwarded
        // during prefill, so the first round leads with the token sampled from `logits`.
        uint32_t pending = sample_token(logits.data(), vocab_size, options.sampling,
                                        seed_for(options.sampling,
                                                 static_cast<uint32_t>(history.size() - 1)));
        history.push_back(pending);
        ++emitted;
        if (on_token && !on_token(pending, tokenizer_->decode_single(pending))) emitted = options.max_tokens;
        if (is_stop(pending)) emitted = options.max_tokens;

        while (emitted < options.max_tokens) {
            if (cancel_flag && cancel_flag->load()) break;
            const auto step_start = std::chrono::high_resolution_clock::now();

            DraftResult dr = draft_runtime_->generate_draft_tokens(history, K - 1, options.sampling);
            if (dr.draft_tokens.empty()) break;

            batch_tokens.clear();
            batch_tokens.push_back(pending);
            for (uint32_t t : dr.draft_tokens) batch_tokens.push_back(t);
            const uint32_t m = static_cast<uint32_t>(batch_tokens.size());

            // Positions: `pending` sits at history.size()-1, the drafts follow it.
            forward_batch(batch_tokens.data(), m,
                          static_cast<uint32_t>(history.size() - 1), verify.data(), true);

            // verify[i] is the distribution after batch_tokens[0..i], so verify[i] decides
            // draft i (0-based), i.e. batch_tokens[i+1].
            std::vector<const float*> per_pos;
            per_pos.reserve(dr.draft_tokens.size() + 1);
            for (uint32_t i = 0; i < m; ++i) {
                per_pos.push_back(verify.data() + static_cast<size_t>(i) * vocab_size);
            }

            SpeculativeEvaluation ev = speculator_->evaluate_verification(
                dr.draft_tokens, per_pos, vocab_size, options.sampling,
                seed_for(options.sampling, static_cast<uint32_t>(history.size())),
                dr.draft_logits);

            bool stop = false;
            for (uint32_t t : ev.accepted_tokens) {
                if (emitted >= options.max_tokens) { stop = true; break; }
                history.push_back(t);
                ++emitted;
                pending = t;
                if (on_token && !on_token(t, tokenizer_->decode_single(t))) { stop = true; break; }
                if (is_stop(t)) { stop = true; break; }
            }
            draft_runtime_->accept(history.size() - 1);

            const double lat = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - step_start).count();
            TelemetryCollector::instance().record_generation_step(
                lat, ev.accepted_tokens.size(),
                static_cast<uint32_t>(dr.draft_tokens.size()), ev.num_accepted);
            if (stop) break;
        }

        is_generating_ = false;
        return;
    }

    for (int step = 0; step < options.max_tokens; ++step) {
        if (cancel_flag && cancel_flag->load()) break;

        auto step_start = std::chrono::high_resolution_clock::now();
        uint32_t pos = static_cast<uint32_t>(history.size() - 1);

        if (options.sampling.repetition_penalty != 1.0f) {
            apply_repetition_penalty(logits.data(), header_.vocab_size, history,
                                     options.sampling.repetition_penalty, header_.final_logit_softcapping);
        }

        uint64_t seed = seed_for(options.sampling, pos);
        uint32_t next_token = sample_token(logits.data(), header_.vocab_size, options.sampling, seed);

        history.push_back(next_token);
        std::string piece = tokenizer_->decode_single(next_token);

        if (on_token) {
            bool keep_going = on_token(next_token, piece);
            if (!keep_going) break;
        }

        if (is_stop(next_token)) break;

        // Forward next token on GPU
        forward_single_token(next_token, static_cast<uint32_t>(history.size() - 1), logits.data());

        auto step_end = std::chrono::high_resolution_clock::now();
        double step_latency_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
        TelemetryCollector::instance().record_generation_step(step_latency_ms, 1, 0, 0);
    }

    is_generating_ = false;
}

TelemetrySnapshot ForwardRunner::get_latest_telemetry() const {
    return TelemetryCollector::instance().snapshot();
}

} // namespace g4dense
