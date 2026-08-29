#pragma once

#include "g4dense/vk_context.hpp"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace g4dense {

enum class ComputeKernel : uint32_t {
    EmbedLookup = 0,
    RMSNormK,
    GemvInt4,
    GemmInt4Batch,
    QKVEpilogue,
    Attention,
    SwiGLU,
    PostAttn,
    LayerTail,
    Softcap,
    LMHeadGreedy,
    ArgmaxReduce,
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
    void update_storage_buffer(VkDescriptorSet ds, uint32_t binding, VkBuffer buffer,
                               uint64_t offset, uint64_t range);

private:
    VulkanContext& ctx_;
    VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
    std::unordered_map<ComputeKernel, VkShaderModule> shader_modules_;
    std::unordered_map<ComputeKernel, VkDescriptorSetLayout> desc_layouts_;
    std::unordered_map<ComputeKernel, VkPipelineLayout> pipeline_layouts_;
    std::unordered_map<ComputeKernel, VkPipeline> pipelines_;

    VkShaderModule load_shader_module(const std::string& spv_path);
};

} // namespace g4dense
