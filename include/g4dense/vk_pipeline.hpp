#pragma once

#include "g4dense/vk_context.hpp"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace g4dense {

// Rows of the output that one GemvInt4 threadgroup computes, one wave each. MUST match
// GEMV_ROWS_PER_GROUP in shaders/GemvInt4.hlsl: dispatch too few groups and the tail of the
// output is silently left at whatever the buffer held.
inline constexpr uint32_t kGemvRowsPerGroup = 8;

// GemmInt4Batch's blocking factors. MUST match GEMM_ROWS_PER_GROUP and GEMM_MAX_BATCH in
// shaders/GemmInt4Batch.hlsl: the kernel clamps the batch to kGemmMaxBatch internally, so
// passing more positions than this silently drops the extra ones rather than failing.
inline constexpr uint32_t kGemmRowsPerGroup = 8;
inline constexpr uint32_t kGemmMaxBatch     = 8;

// Kernels the forward pass actually dispatches, plus three kept deliberately:
//   EmbedLookup   parity-tested; the embedding is still dequantized on the CPU, which measures
//                 ~0.1 ms of a ~1400 ms pass, so wiring it in was not worth the change.
//   LMHeadGreedy  a fused argmax head. Unused because sampling, speculative verification and
//                 the oracle diff all need the full distribution, not just the top token.
//   GemvInt8      for an INT8 KV path that does not exist.
// PostAttn and LayerTail were fused epilogues from the sibling MoE project; they were deleted
// in round 7 after fusion measured worthless against 3.6 ms of per-token submission overhead.
enum class ComputeKernel : uint32_t {
    EmbedLookup = 0,
    RMSNormK,
    GemvInt4,
    GemmInt4Batch,
    QKVEpilogue,
    Attention,
    GeGLU,
    ResidualAccum,
    Softcap,
    LMHeadGreedy,
    ArgmaxReduce,
    KVWrite,
    COUNT
};

struct DispatchParams {
    uint32_t u32_params[16]{0};
    float    f32_params[16]{0.0f};

    void set_u32(size_t index, uint32_t val) { if (index < 16) u32_params[index] = val; }
    void set_f32(size_t index, float val) { if (index < 16) f32_params[index] = val; }
    void clear() {
        for (auto& v : u32_params) v = 0;
        for (auto& v : f32_params) v = 0.0f;
    }
};

class VulkanPipelineManager {
public:
    explicit VulkanPipelineManager(VulkanContext& ctx);
    ~VulkanPipelineManager();

    // Initializes compute pipelines from SPIR-V bytecode directory (build/shaders/)
    // Requires explicit descriptor capacity argument to avoid runtime pool exhaustion
    void initialize_pipelines(size_t descriptor_capacity = 4096);

    VkPipeline get_pipeline(ComputeKernel kernel) const;
    VkPipelineLayout get_pipeline_layout(ComputeKernel kernel) const;

    // Command recording helpers
    void bind_kernel(VkCommandBuffer cmd, ComputeKernel kernel);
    void push_constants(VkCommandBuffer cmd, ComputeKernel kernel, const void* data, size_t size_bytes);
    void dispatch(VkCommandBuffer cmd, uint32_t group_x, uint32_t group_y = 1, uint32_t group_z = 1);

    // Descriptor set allocation and binding
    VkDescriptorSet allocate_descriptor_set(ComputeKernel kernel);

    // Frees every set handed out since the last reset, in one call.
    //
    // Sets are allocated per dispatch and never individually freed, so without this the pool
    // is exhausted mid-generation: one token costs roughly 60 layers x 13 dispatches ~= 800
    // sets, so a 4096-set pool dies after about five tokens with
    // "failed to allocate descriptor set". Single-token tests never reached it, which is why
    // it survived two validation rounds -- multi-token generation on the real model had in
    // fact never run to completion.
    //
    // Call this once per forward pass, before recording. Every set from the previous pass is
    // invalidated, which is safe only because the pass is fully submitted and waited on
    // before the next one begins.
    void reset_descriptor_pool();
    void update_storage_buffer(VkDescriptorSet ds, uint32_t binding, VkBuffer buffer,
                               uint64_t offset, uint64_t range);

private:
    VulkanContext& ctx_;
    VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
    size_t descriptor_capacity_{0};
    size_t sets_allocated_since_reset_{0};
    size_t peak_sets_per_pass_{0};

    // Kernels that could not be created at requiredSubgroupSize=32 and fell back to the
    // device default wave width. Empty is the expected state on RDNA 3.
    std::vector<std::string> subgroup_fallbacks_;
    std::unordered_map<ComputeKernel, VkShaderModule> shader_modules_;
    std::unordered_map<ComputeKernel, VkDescriptorSetLayout> desc_layouts_;
    std::unordered_map<ComputeKernel, VkPipelineLayout> pipeline_layouts_;
    std::unordered_map<ComputeKernel, VkPipeline> pipelines_;

    VkShaderModule load_shader_module(const std::string& spv_path);
};

} // namespace g4dense
