#include "g4dense/mtp_runner.hpp"
#include "g4dense/format.hpp"
#include "g4dense/sampling.hpp"

#include <chrono>
#include <iostream>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <filesystem>
#include <windows.h>

namespace g4dense {

namespace {

inline float bf16_to_f32(uint16_t val) {
    uint32_t u32 = static_cast<uint32_t>(val) << 16;
    float f;
    std::memcpy(&f, &u32, sizeof(float));
    return f;
}

} // namespace

MtpRunner::MtpRunner() = default;

MtpRunner::~MtpRunner() {
    cleanup();
}

void MtpRunner::cleanup() {
    if (ctx_ && ctx_->device()) {
        vkDeviceWaitIdle(ctx_->device());

        auto free_buf = [&](VkMemoryAllocation& a) {
            if (a.buffer != VK_NULL_HANDLE) {
                ctx_->free_buffer(a);
                a = {};
            }
        };

        free_buf(buf_in_);
        free_buf(buf_hidden_);
        free_buf(buf_norm_act_);
        free_buf(buf_q_);
        free_buf(buf_attn_out_);
        free_buf(buf_proj_out_);
        free_buf(buf_gate_);
        free_buf(buf_up_);
        free_buf(buf_post_proj_act_);
        free_buf(buf_logits_);

        free_buf(buf_embed_);
        free_buf(buf_pre_proj_);
        free_buf(buf_post_proj_);
        free_buf(buf_norm_);
        for (auto& b : buf_layers_) free_buf(b);
        buf_layers_.clear();

        if (fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(ctx_->device(), fence_, nullptr);
            fence_ = VK_NULL_HANDLE;
        }
        if (cmd_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(ctx_->device(), cmd_pool_, nullptr);
            cmd_pool_ = VK_NULL_HANDLE;
            cmd_ = VK_NULL_HANDLE;
        }
    }

    pipeline_mgr_.reset();

    if (mapped_data_) {
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

    loaded_ = false;
    total_gpu_bytes_ = 0;
}

bool MtpRunner::load_model(const std::string& model_path, std::shared_ptr<VulkanContext> ctx) {
    cleanup();

    if (!std::filesystem::exists(model_path)) {
        throw G4DenseFormatError("MtpRunner: checkpoint not found: " + model_path);
    }
    if (!ctx) {
        throw G4DenseFormatError("MtpRunner: VulkanContext is required");
    }

    ctx_ = ctx;
    model_path_ = model_path;

    // 1. Open and memory-map container
    file_handle_ = CreateFileA(model_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw G4DenseFormatError("MtpRunner: failed to open " + model_path);
    }

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(file_handle_, &sz)) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw G4DenseFormatError("MtpRunner: failed to get file size for " + model_path);
    }
    file_size_ = static_cast<uint64_t>(sz.QuadPart);

    mapping_handle_ = CreateFileMappingA(file_handle_, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping_handle_) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw G4DenseFormatError("MtpRunner: CreateFileMapping failed for " + model_path);
    }

    mapped_data_ = static_cast<const uint8_t*>(MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
    if (!mapped_data_) {
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        mapping_handle_ = NULL;
        file_handle_ = INVALID_HANDLE_VALUE;
        throw G4DenseFormatError("MtpRunner: MapViewOfFile failed for " + model_path);
    }

    // 2. Validate header
    std::memcpy(&header_, mapped_data_, sizeof(G4MtpHeader));
    validate_mtp_header(header_, file_size_);

    // 3. Initialize Vulkan compute pipeline manager
    pipeline_mgr_ = std::make_unique<VulkanPipelineManager>(*ctx_);
    pipeline_mgr_->initialize_pipelines(1024);

    // 4. Create command pool and buffer
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = ctx_->compute_queue_family();
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(ctx_->device(), &pool_info, nullptr, &cmd_pool_) != VK_SUCCESS) {
        throw G4DenseFormatError("MtpRunner: failed to create compute command pool");
    }

    VkCommandBufferAllocateInfo alloc_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc_info.commandPool = cmd_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx_->device(), &alloc_info, &cmd_) != VK_SUCCESS) {
        throw G4DenseFormatError("MtpRunner: failed to allocate compute command buffer");
    }

    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(ctx_->device(), &fence_info, nullptr, &fence_) != VK_SUCCESS) {
        throw G4DenseFormatError("MtpRunner: failed to create fence");
    }

    // 5. Precompute per-layer offsets
    compute_layer_offsets();

    // 6. Allocate GPU memory and upload weights
    allocate_gpu_resources();

    loaded_ = true;
    return true;
}

void MtpRunner::compute_layer_offsets() {
    layer_offsets_.resize(header_.num_layers);

    for (uint32_t l = 0; l < header_.num_layers; ++l) {
        MtpLayerOffsets& lo = layer_offsets_[l];
        const bool is_global = (header_.global_layer_mask & (1ULL << l)) != 0;
        const uint32_t q_head_dim = is_global ? header_.global_head_dim : header_.head_dim;
        const uint32_t total_q_dim = header_.num_q_heads * q_head_dim;
        const uint32_t hidden_size = header_.hidden_size;
        const uint32_t intermediate_size = header_.intermediate_size;

        // Norms
        lo.in_norm_off = 0;
        lo.post_attn_norm_off = hidden_size * 2;
        lo.pre_ffn_norm_off = hidden_size * 4;
        lo.post_ffn_norm_off = hidden_size * 6;
        lo.q_norm_off = hidden_size * 8;
        lo.layer_scalar_off = lo.q_norm_off + q_head_dim * 2;

        // Read layer_scalar directly from container memory
        const uint8_t* layer_base = mapped_data_ + header_.layer_offsets[l];
        const uint16_t scalar_bf16 = *reinterpret_cast<const uint16_t*>(layer_base + lo.layer_scalar_off);
        lo.layer_scalar = bf16_to_f32(scalar_bf16);

        // Projections start at 16-byte aligned offset
        uint32_t cur = (lo.layer_scalar_off + 2 + 15) & ~15u;

        auto assign_proj = [&](uint32_t& w_off, uint32_t& s_off, uint32_t& b_off,
                               uint32_t& rows_out, uint32_t& in_dim_out,
                               uint32_t rows, uint32_t in_dim) {
            cur = (cur + 15) & ~15u;
            w_off = cur;
            rows_out = rows;
            in_dim_out = in_dim;
            const uint32_t w_bytes = rows * (in_dim / 8) * sizeof(uint32_t);
            const uint32_t s_bytes = rows * (in_dim / 64) * sizeof(uint16_t);
            const uint32_t b_bytes = rows * (in_dim / 64) * sizeof(uint16_t);
            s_off = cur + w_bytes;
            b_off = cur + w_bytes + s_bytes;
            cur += w_bytes + s_bytes + b_bytes;
        };

        assign_proj(lo.q_proj_w_off, lo.q_proj_s_off, lo.q_proj_b_off,
                    lo.q_proj_rows, lo.q_proj_in_dim, total_q_dim, hidden_size);

        assign_proj(lo.o_proj_w_off, lo.o_proj_s_off, lo.o_proj_b_off,
                    lo.o_proj_rows, lo.o_proj_in_dim, hidden_size, total_q_dim);

        assign_proj(lo.gate_proj_w_off, lo.gate_proj_s_off, lo.gate_proj_b_off,
                    lo.gate_proj_rows, lo.gate_proj_in_dim, intermediate_size, hidden_size);

        assign_proj(lo.up_proj_w_off, lo.up_proj_s_off, lo.up_proj_b_off,
                    lo.up_proj_rows, lo.up_proj_in_dim, intermediate_size, hidden_size);

        assign_proj(lo.down_proj_w_off, lo.down_proj_s_off, lo.down_proj_b_off,
                    lo.down_proj_rows, lo.down_proj_in_dim, hidden_size, intermediate_size);
    }
}

void MtpRunner::allocate_gpu_resources() {
    total_gpu_bytes_ = 0;

    auto alloc_weight = [&](uint64_t size, uint64_t file_offset) -> VkMemoryAllocation {
        VkMemoryAllocation a = ctx_->allocate_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                    MemoryResidency::HostVisibleMapped);
        if (a.mapped_ptr && mapped_data_) {
            std::memcpy(a.mapped_ptr, mapped_data_ + file_offset, size);
        }
        total_gpu_bytes_ += size;
        return a;
    };

    // 1. Weights
    buf_embed_ = alloc_weight(header_.embed_size, header_.embed_offset);
    buf_pre_proj_ = alloc_weight(header_.pre_proj_size, header_.pre_proj_offset);
    buf_post_proj_ = alloc_weight(header_.post_proj_size, header_.post_proj_offset);
    buf_norm_ = alloc_weight(header_.norm_size, header_.norm_offset);

    buf_layers_.resize(header_.num_layers);
    for (uint32_t l = 0; l < header_.num_layers; ++l) {
        buf_layers_[l] = alloc_weight(header_.layer_sizes[l], header_.layer_offsets[l]);
    }

    // 2. Activation scratch buffers
    const uint32_t backbone_dim = header_.backbone_hidden_size;
    const uint32_t hidden_size = header_.hidden_size;
    const uint32_t intermediate_size = header_.intermediate_size;
    const uint32_t max_q_dim = header_.num_q_heads * header_.global_head_dim;
    const uint32_t vocab_size = header_.vocab_size;

    auto alloc_act = [&](uint64_t size) -> VkMemoryAllocation {
        VkMemoryAllocation a = ctx_->allocate_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                    MemoryResidency::HostVisibleMapped);
        total_gpu_bytes_ += size;
        return a;
    };

    buf_in_ = alloc_act((backbone_dim * 2) * sizeof(float)); // [e(t); h]
    buf_hidden_ = alloc_act(hidden_size * sizeof(float));
    buf_norm_act_ = alloc_act(hidden_size * sizeof(float));
    buf_q_ = alloc_act(max_q_dim * sizeof(float));
    buf_attn_out_ = alloc_act(max_q_dim * sizeof(float));
    buf_proj_out_ = alloc_act(hidden_size * sizeof(float));
    buf_gate_ = alloc_act(intermediate_size * sizeof(float));
    buf_up_ = alloc_act(intermediate_size * sizeof(float));
    buf_post_proj_act_ = alloc_act(backbone_dim * sizeof(float));
    buf_logits_ = alloc_act(static_cast<uint64_t>(vocab_size) * sizeof(float));
}

DraftResult MtpRunner::generate_draft_tokens(
    const float* initial_target_hidden_state,
    uint32_t seed_token,
    uint32_t context_length,
    uint32_t k,
    const SamplingParams& sampling,
    const std::function<void(uint32_t, float*)>& get_embedding_fn,
    KVCacheManager* kv_cache) {

    const auto t0 = std::chrono::high_resolution_clock::now();
    DraftResult result;
    if (!loaded_ || k == 0 || !initial_target_hidden_state || !kv_cache) {
        return result;
    }

    result.draft_tokens.reserve(k);
    result.draft_logits.reserve(k);

    const uint32_t backbone_dim = header_.backbone_hidden_size;
    const uint32_t hidden_size = header_.hidden_size;
    const uint32_t intermediate_size = header_.intermediate_size;
    const uint32_t vocab_size = header_.vocab_size;
    VkQueue queue = ctx_->compute_queue();

    VkMemoryBarrier mem_bar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mem_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mem_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    uint32_t current_token = seed_token;

    for (uint32_t step = 0; step < k; ++step) {
        pipeline_mgr_->reset_descriptor_pool();

        // --- 1. Prepare input: [e(t); h] in buf_in_ ---
        float* in_host = static_cast<float*>(buf_in_.mapped_ptr);
        // Channels 0..5375: token embedding
        get_embedding_fn(current_token, in_host);
        // Channels 5376..10751: target hidden state
        if (step == 0) {
            std::memcpy(in_host + backbone_dim, initial_target_hidden_state,
                        backbone_dim * sizeof(float));
        } else {
            std::memcpy(in_host + backbone_dim, buf_post_proj_act_.mapped_ptr,
                        backbone_dim * sizeof(float));
        }

        // --- 2. Record GPU command buffer ---
        vkResetCommandBuffer(cmd_, 0);
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd_, &begin_info);

        // Pre-projection: [1024 x 10752] x [10752] -> buf_hidden_
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::GemvInt4);
            const uint32_t w_bytes = 1024 * (10752 / 8) * sizeof(uint32_t);
            const uint32_t s_bytes = 1024 * (10752 / 64) * sizeof(uint16_t);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_pre_proj_.buffer, 0, w_bytes);
            pipeline_mgr_->update_storage_buffer(ds, 1, buf_pre_proj_.buffer, w_bytes, s_bytes);
            pipeline_mgr_->update_storage_buffer(ds, 2, buf_pre_proj_.buffer, w_bytes + s_bytes, s_bytes);
            pipeline_mgr_->update_storage_buffer(ds, 3, buf_in_.buffer, 0, (backbone_dim * 2) * sizeof(float));
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_hidden_.buffer, 0, hidden_size * sizeof(float));

            uint32_t pc[16]{0};
            pc[0] = hidden_size;          // rows = 1024
            pc[1] = backbone_dim * 2;     // in_dim = 10752
            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::GemvInt4);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::GemvInt4),
                                    0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::GemvInt4, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, (hidden_size + kGemvRowsPerGroup - 1) / kGemvRowsPerGroup, 1, 1);
        }

        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // 4 Transformer Layers
        for (uint32_t l = 0; l < header_.num_layers; ++l) {
            const MtpLayerOffsets& lo = layer_offsets_[l];
            const bool is_global = (header_.global_layer_mask & (1ULL << l)) != 0;
            const uint32_t q_head_dim = is_global ? header_.global_head_dim : header_.head_dim;
            const uint32_t total_q_dim = header_.num_q_heads * q_head_dim;
            const uint32_t kv_heads = is_global ? header_.global_kv_heads : header_.num_kv_heads;
            const uint32_t donor_layer = is_global ? 59u : 58u;
            const uint32_t kv_cap = is_global ? kv_cache->layer_capacity(59) : 1024u;
            const uint32_t first_pos = (context_length >= kv_cap && !is_global) ? (context_length - kv_cap) : 0;
            const float rope_theta = is_global ? 1000000.0f : 10000.0f;
            const uint32_t rotated_pairs = is_global ? 64u : 128u;
            VkBuffer layer_buf = buf_layers_[l].buffer;
            const uint64_t layer_buf_sz = header_.layer_sizes[l];

            // A. Input layernorm: hidden -> norm_act
            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::RMSNormK);
                pipeline_mgr_->update_storage_buffer(ds, 0, buf_hidden_.buffer, 0, hidden_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_sz);
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_norm_act_.buffer, 0, hidden_size * sizeof(float));

                uint32_t pc[8]{0};
                pc[0] = hidden_size;
                pc[1] = 0;
                pc[2] = lo.in_norm_off;
                pc[3] = 0;
                pc[4] = 1;
                float eps = 1e-6f;
                std::memcpy(&pc[5], &eps, 4);

                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::RMSNormK);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::RMSNormK),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::RMSNormK, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, 1, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            // B. Q projection: norm_act -> q
            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::GemvInt4);
                const uint32_t w_bytes = lo.q_proj_rows * (lo.q_proj_in_dim / 8) * sizeof(uint32_t);
                const uint32_t s_bytes = lo.q_proj_rows * (lo.q_proj_in_dim / 64) * sizeof(uint16_t);
                pipeline_mgr_->update_storage_buffer(ds, 0, layer_buf, lo.q_proj_w_off, w_bytes);
                pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, lo.q_proj_s_off, s_bytes);
                pipeline_mgr_->update_storage_buffer(ds, 2, layer_buf, lo.q_proj_b_off, s_bytes);
                pipeline_mgr_->update_storage_buffer(ds, 3, buf_norm_act_.buffer, 0, hidden_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_q_.buffer, 0, total_q_dim * sizeof(float));

                uint32_t pc[16]{0};
                pc[0] = lo.q_proj_rows;
                pc[1] = lo.q_proj_in_dim;
                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::GemvInt4);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::GemvInt4),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::GemvInt4, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, (lo.q_proj_rows + kGemvRowsPerGroup - 1) / kGemvRowsPerGroup, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            // C. Q Norm & RoPE
            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::QKVEpilogue);
                pipeline_mgr_->update_storage_buffer(ds, 0, buf_q_.buffer, 0, total_q_dim * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_sz);
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_q_.buffer, 0, total_q_dim * sizeof(float));

                uint32_t pc[16]{0};
                pc[0] = q_head_dim;
                pc[1] = header_.num_q_heads;
                pc[2] = 0;
                pc[3] = 0;
                float eps = 1e-6f;
                std::memcpy(&pc[4], &eps, 4);
                pc[5] = lo.q_norm_off;
                pc[6] = 1;
                pc[7] = 1;
                pc[8] = rotated_pairs;
                pc[9] = context_length - 1; // position of last token in KV cache
                std::memcpy(&pc[10], &rope_theta, 4);

                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::QKVEpilogue);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::QKVEpilogue),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::QKVEpilogue, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, header_.num_q_heads, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            // D. Cross-Attention against target shared KV cache
            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::Attention);
                pipeline_mgr_->update_storage_buffer(ds, 0, buf_q_.buffer, 0, total_q_dim * sizeof(float));
                const uint64_t kv_stride = static_cast<uint64_t>(kv_heads) * q_head_dim * 2;
                pipeline_mgr_->update_storage_buffer(ds, 1, kv_cache->k_buffer(donor_layer), 0,
                                                     static_cast<uint64_t>(kv_cap) * kv_stride);
                pipeline_mgr_->update_storage_buffer(ds, 2, kv_cache->v_buffer(donor_layer), 0,
                                                     static_cast<uint64_t>(kv_cap) * kv_stride);
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_attn_out_.buffer, 0, total_q_dim * sizeof(float));

                uint32_t pc[16]{0};
                pc[0] = header_.num_q_heads;
                pc[1] = kv_heads;
                pc[2] = q_head_dim;
                pc[3] = context_length;
                pc[4] = first_pos;
                pc[9] = kv_cap;
                float scale = 1.0f;
                std::memcpy(&pc[10], &scale, 4);
                pc[12] = 1; // batch
                pc[15] = is_global ? 0u : kv_cap;

                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::Attention);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::Attention),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::Attention, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, header_.num_q_heads, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            // E. O projection: attn_out -> proj_out
            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::GemvInt4);
                const uint32_t w_bytes = lo.o_proj_rows * (lo.o_proj_in_dim / 8) * sizeof(uint32_t);
                const uint32_t s_bytes = lo.o_proj_rows * (lo.o_proj_in_dim / 64) * sizeof(uint16_t);
                pipeline_mgr_->update_storage_buffer(ds, 0, layer_buf, lo.o_proj_w_off, w_bytes);
                pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, lo.o_proj_s_off, s_bytes);
                pipeline_mgr_->update_storage_buffer(ds, 2, layer_buf, lo.o_proj_b_off, s_bytes);
                pipeline_mgr_->update_storage_buffer(ds, 3, buf_attn_out_.buffer, 0, total_q_dim * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_proj_out_.buffer, 0, hidden_size * sizeof(float));

                uint32_t pc[16]{0};
                pc[0] = lo.o_proj_rows;
                pc[1] = lo.o_proj_in_dim;
                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::GemvInt4);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::GemvInt4),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::GemvInt4, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, (lo.o_proj_rows + kGemvRowsPerGroup - 1) / kGemvRowsPerGroup, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            // F. Post-attention RMSNorm: proj_out -> proj_out
            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::RMSNormK);
                pipeline_mgr_->update_storage_buffer(ds, 0, buf_proj_out_.buffer, 0, hidden_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_sz);
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_proj_out_.buffer, 0, hidden_size * sizeof(float));

                uint32_t pc[8]{0};
                pc[0] = hidden_size;
                pc[1] = 0;
                pc[2] = lo.post_attn_norm_off;
                pc[3] = 0;
                pc[4] = 1;
                float eps = 1e-6f;
                std::memcpy(&pc[5], &eps, 4);

                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::RMSNormK);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::RMSNormK),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::RMSNormK, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, 1, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            // G. Attention Residual: hidden = hidden + proj_out
            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::ResidualAccum);
                pipeline_mgr_->update_storage_buffer(ds, 0, buf_hidden_.buffer, 0, hidden_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 1, buf_proj_out_.buffer, 0, hidden_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_hidden_.buffer, 0, hidden_size * sizeof(float));

                uint32_t pc[8]{0};
                pc[0] = hidden_size;
                float res_scale = 1.0f;
                float out_scale = 1.0f;
                std::memcpy(&pc[1], &res_scale, 4);
                std::memcpy(&pc[2], &out_scale, 4);

                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::ResidualAccum);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::ResidualAccum),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::ResidualAccum, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, (hidden_size + 255) / 256, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            // H. Pre-FFN layernorm: hidden -> norm_act
            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::RMSNormK);
                pipeline_mgr_->update_storage_buffer(ds, 0, buf_hidden_.buffer, 0, hidden_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_sz);
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_norm_act_.buffer, 0, hidden_size * sizeof(float));

                uint32_t pc[8]{0};
                pc[0] = hidden_size;
                pc[1] = 0;
                pc[2] = lo.pre_ffn_norm_off;
                pc[3] = 0;
                pc[4] = 1;
                float eps = 1e-6f;
                std::memcpy(&pc[5], &eps, 4);

                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::RMSNormK);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::RMSNormK),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::RMSNormK, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, 1, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            // I. MLP Gate & Up projections
            {
                // Gate proj
                VkDescriptorSet ds_g = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::GemvInt4);
                const uint32_t wg_bytes = lo.gate_proj_rows * (lo.gate_proj_in_dim / 8) * sizeof(uint32_t);
                const uint32_t sg_bytes = lo.gate_proj_rows * (lo.gate_proj_in_dim / 64) * sizeof(uint16_t);
                pipeline_mgr_->update_storage_buffer(ds_g, 0, layer_buf, lo.gate_proj_w_off, wg_bytes);
                pipeline_mgr_->update_storage_buffer(ds_g, 1, layer_buf, lo.gate_proj_s_off, sg_bytes);
                pipeline_mgr_->update_storage_buffer(ds_g, 2, layer_buf, lo.gate_proj_b_off, sg_bytes);
                pipeline_mgr_->update_storage_buffer(ds_g, 3, buf_norm_act_.buffer, 0, hidden_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds_g, 6, buf_gate_.buffer, 0, intermediate_size * sizeof(float));

                uint32_t pc_g[16]{0};
                pc_g[0] = lo.gate_proj_rows;
                pc_g[1] = lo.gate_proj_in_dim;
                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::GemvInt4);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::GemvInt4),
                                        0, 1, &ds_g, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::GemvInt4, pc_g, sizeof(pc_g));
                pipeline_mgr_->dispatch(cmd_, (lo.gate_proj_rows + kGemvRowsPerGroup - 1) / kGemvRowsPerGroup, 1, 1);

                // Up proj
                VkDescriptorSet ds_u = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::GemvInt4);
                const uint32_t wu_bytes = lo.up_proj_rows * (lo.up_proj_in_dim / 8) * sizeof(uint32_t);
                const uint32_t su_bytes = lo.up_proj_rows * (lo.up_proj_in_dim / 64) * sizeof(uint16_t);
                pipeline_mgr_->update_storage_buffer(ds_u, 0, layer_buf, lo.up_proj_w_off, wu_bytes);
                pipeline_mgr_->update_storage_buffer(ds_u, 1, layer_buf, lo.up_proj_s_off, su_bytes);
                pipeline_mgr_->update_storage_buffer(ds_u, 2, layer_buf, lo.up_proj_b_off, su_bytes);
                pipeline_mgr_->update_storage_buffer(ds_u, 3, buf_norm_act_.buffer, 0, hidden_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds_u, 6, buf_up_.buffer, 0, intermediate_size * sizeof(float));

                uint32_t pc_u[16]{0};
                pc_u[0] = lo.up_proj_rows;
                pc_u[1] = lo.up_proj_in_dim;
                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::GemvInt4);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::GemvInt4),
                                        0, 1, &ds_u, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::GemvInt4, pc_u, sizeof(pc_u));
                pipeline_mgr_->dispatch(cmd_, (lo.up_proj_rows + kGemvRowsPerGroup - 1) / kGemvRowsPerGroup, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            // J. GeGLU: act = gelu(gate) * up -> buf_gate_
            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::GeGLU);
                pipeline_mgr_->update_storage_buffer(ds, 0, buf_gate_.buffer, 0, intermediate_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 1, buf_up_.buffer, 0, intermediate_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_gate_.buffer, 0, intermediate_size * sizeof(float));

                uint32_t pc[8]{0};
                pc[0] = intermediate_size;
                pc[4] = 1; // batch = 1

                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::GeGLU);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::GeGLU),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::GeGLU, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, (intermediate_size + 255) / 256, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            // K. Down projection: gate -> proj_out
            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::GemvInt4);
                const uint32_t w_bytes = lo.down_proj_rows * (lo.down_proj_in_dim / 8) * sizeof(uint32_t);
                const uint32_t s_bytes = lo.down_proj_rows * (lo.down_proj_in_dim / 64) * sizeof(uint16_t);
                pipeline_mgr_->update_storage_buffer(ds, 0, layer_buf, lo.down_proj_w_off, w_bytes);
                pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, lo.down_proj_s_off, s_bytes);
                pipeline_mgr_->update_storage_buffer(ds, 2, layer_buf, lo.down_proj_b_off, s_bytes);
                pipeline_mgr_->update_storage_buffer(ds, 3, buf_gate_.buffer, 0, intermediate_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_proj_out_.buffer, 0, hidden_size * sizeof(float));

                uint32_t pc[16]{0};
                pc[0] = lo.down_proj_rows;
                pc[1] = lo.down_proj_in_dim;
                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::GemvInt4);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::GemvInt4),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::GemvInt4, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, (lo.down_proj_rows + kGemvRowsPerGroup - 1) / kGemvRowsPerGroup, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            // L. Post-FFN layernorm: proj_out -> proj_out
            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::RMSNormK);
                pipeline_mgr_->update_storage_buffer(ds, 0, buf_proj_out_.buffer, 0, hidden_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 1, layer_buf, 0, layer_buf_sz);
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_proj_out_.buffer, 0, hidden_size * sizeof(float));

                uint32_t pc[8]{0};
                pc[0] = hidden_size;
                pc[1] = 0;
                pc[2] = lo.post_ffn_norm_off;
                pc[3] = 0;
                pc[4] = 1;
                float eps = 1e-6f;
                std::memcpy(&pc[5], &eps, 4);

                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::RMSNormK);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::RMSNormK),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::RMSNormK, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, 1, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);

            // M. FFN Residual with layer_scalar: hidden = (hidden + proj_out) * layer_scalar
            {
                VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::ResidualAccum);
                pipeline_mgr_->update_storage_buffer(ds, 0, buf_hidden_.buffer, 0, hidden_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 1, buf_proj_out_.buffer, 0, hidden_size * sizeof(float));
                pipeline_mgr_->update_storage_buffer(ds, 6, buf_hidden_.buffer, 0, hidden_size * sizeof(float));

                uint32_t pc[8]{0};
                pc[0] = hidden_size;
                float res_scale = 1.0f;
                float out_scale = lo.layer_scalar;
                std::memcpy(&pc[1], &res_scale, 4);
                std::memcpy(&pc[2], &out_scale, 4);

                pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::ResidualAccum);
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_mgr_->get_pipeline_layout(ComputeKernel::ResidualAccum),
                                        0, 1, &ds, 0, nullptr);
                pipeline_mgr_->push_constants(cmd_, ComputeKernel::ResidualAccum, pc, sizeof(pc));
                pipeline_mgr_->dispatch(cmd_, (hidden_size + 255) / 256, 1, 1);
            }

            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, nullptr, 0, nullptr);
        }

        // --- 3. Final RMSNorm: hidden -> norm_act ---
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::RMSNormK);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_hidden_.buffer, 0, hidden_size * sizeof(float));
            pipeline_mgr_->update_storage_buffer(ds, 1, buf_norm_.buffer, 0, header_.norm_size);
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_norm_act_.buffer, 0, hidden_size * sizeof(float));

            uint32_t pc[8]{0};
            pc[0] = hidden_size;
            pc[1] = 0;
            pc[2] = 0;
            pc[3] = 0;
            pc[4] = 1;
            float eps = 1e-6f;
            std::memcpy(&pc[5], &eps, 4);

            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::RMSNormK);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::RMSNormK),
                                    0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::RMSNormK, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, 1, 1, 1);
        }

        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, nullptr, 0, nullptr);

        // --- 4. Post-projection: norm_act -> buf_post_proj_act_ ---
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::GemvInt4);
            const uint32_t w_bytes = backbone_dim * (hidden_size / 8) * sizeof(uint32_t);
            const uint32_t s_bytes = backbone_dim * (hidden_size / 64) * sizeof(uint16_t);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_post_proj_.buffer, 0, w_bytes);
            pipeline_mgr_->update_storage_buffer(ds, 1, buf_post_proj_.buffer, w_bytes, s_bytes);
            pipeline_mgr_->update_storage_buffer(ds, 2, buf_post_proj_.buffer, w_bytes + s_bytes, s_bytes);
            pipeline_mgr_->update_storage_buffer(ds, 3, buf_norm_act_.buffer, 0, hidden_size * sizeof(float));
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_post_proj_act_.buffer, 0, backbone_dim * sizeof(float));

            uint32_t pc[16]{0};
            pc[0] = backbone_dim;
            pc[1] = hidden_size;
            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::GemvInt4);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::GemvInt4),
                                    0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::GemvInt4, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, (backbone_dim + kGemvRowsPerGroup - 1) / kGemvRowsPerGroup, 1, 1);
        }

        // --- 5. LM Head: norm_act -> buf_logits_ ---
        {
            VkDescriptorSet ds = pipeline_mgr_->allocate_descriptor_set(ComputeKernel::GemvInt4);
            const uint64_t w_bytes = static_cast<uint64_t>(vocab_size) * (hidden_size / 8) * sizeof(uint32_t);
            const uint64_t s_bytes = static_cast<uint64_t>(vocab_size) * (hidden_size / 64) * sizeof(uint16_t);
            pipeline_mgr_->update_storage_buffer(ds, 0, buf_embed_.buffer, 0, w_bytes);
            pipeline_mgr_->update_storage_buffer(ds, 1, buf_embed_.buffer, w_bytes, s_bytes);
            pipeline_mgr_->update_storage_buffer(ds, 2, buf_embed_.buffer, w_bytes + s_bytes, s_bytes);
            pipeline_mgr_->update_storage_buffer(ds, 3, buf_norm_act_.buffer, 0, hidden_size * sizeof(float));
            pipeline_mgr_->update_storage_buffer(ds, 6, buf_logits_.buffer, 0,
                                                 static_cast<uint64_t>(vocab_size) * sizeof(float));

            uint32_t pc[16]{0};
            pc[0] = vocab_size;
            pc[1] = hidden_size;
            pipeline_mgr_->bind_kernel(cmd_, ComputeKernel::GemvInt4);
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_mgr_->get_pipeline_layout(ComputeKernel::GemvInt4),
                                    0, 1, &ds, 0, nullptr);
            pipeline_mgr_->push_constants(cmd_, ComputeKernel::GemvInt4, pc, sizeof(pc));
            pipeline_mgr_->dispatch(cmd_, (vocab_size + kGemvRowsPerGroup - 1) / kGemvRowsPerGroup, 1, 1);
        }

        vkEndCommandBuffer(cmd_);

        // Submit and wait for this draft step
        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_;
        if (vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw G4DenseFormatError("MtpRunner: vkQueueSubmit failed during draft step");
        }
        vkQueueWaitIdle(queue);

        // --- 6. Host sampling ---
        const float* logits = static_cast<const float*>(buf_logits_.mapped_ptr);
        const uint64_t step_seed = splitmix64(sampling.has_seed ? (sampling.seed + step) : (1337 + step));
        const uint32_t cand = sample_token(logits, vocab_size, sampling, step_seed);

        result.draft_tokens.push_back(cand);
        // Copy logits for speculative verification coordinator
        std::vector<float> step_logits(vocab_size);
        std::memcpy(step_logits.data(), logits, vocab_size * sizeof(float));
        result.draft_logits.push_back(std::move(step_logits));

        current_token = cand;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    result.draft_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

} // namespace g4dense
