#include "g4dense/vk_pipeline.hpp"
#include "g4dense/format.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <cstring>

namespace g4dense {

namespace {

const char* get_kernel_spv_name(ComputeKernel kernel) {
    switch (kernel) {
        case ComputeKernel::EmbedLookup:    return "EmbedLookup.spv";
        case ComputeKernel::RMSNormK:       return "RMSNormK.spv";
        case ComputeKernel::GemvInt4:       return "GemvInt4.spv";
        case ComputeKernel::GemmInt4Batch:  return "GemvInt4.spv"; // Shared kernel with batch parameter
        case ComputeKernel::QKVEpilogue:    return "QKVEpilogue.spv";
        case ComputeKernel::Attention:      return "Attention.spv";
        case ComputeKernel::SwiGLU:         return "GeGLU.spv";
        case ComputeKernel::PostAttn:       return "PostAttn.spv";
        case ComputeKernel::LayerTail:      return "LayerTail.spv";
        case ComputeKernel::Softcap:        return "Softcap.spv";
        case ComputeKernel::LMHeadGreedy:   return "LMHeadGreedy.spv";
        case ComputeKernel::ArgmaxReduce:   return "ArgmaxReduce.spv";
        default: return "";
    }
}

} // namespace

VulkanPipelineManager::VulkanPipelineManager(VulkanContext& ctx)
    : ctx_(ctx) {}

VulkanPipelineManager::~VulkanPipelineManager() {
    VkDevice dev = ctx_.device();
    for (auto& [_, p] : pipelines_) vkDestroyPipeline(dev, p, nullptr);
    for (auto& [_, pl] : pipeline_layouts_) vkDestroyPipelineLayout(dev, pl, nullptr);
    for (auto& [_, dsl] : desc_layouts_) vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    for (auto& [_, sm] : shader_modules_) vkDestroyShaderModule(dev, sm, nullptr);
    if (descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, descriptor_pool_, nullptr);
}

VkShaderModule VulkanPipelineManager::load_shader_module(const std::string& spv_path) {
    std::ifstream f(spv_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw G4DenseFormatError("VulkanPipelineManager: cannot open SPIR-V file " + spv_path);
    }
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<char> bytecode(size);
    f.read(bytecode.data(), size);

    VkShaderModuleCreateInfo sm_ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm_ci.codeSize = size;
    sm_ci.pCode = reinterpret_cast<const uint32_t*>(bytecode.data());

    VkShaderModule mod = VK_NULL_HANDLE;
    VkResult res = vkCreateShaderModule(ctx_.device(), &sm_ci, nullptr, &mod);
    if (res != VK_SUCCESS) {
        throw G4DenseFormatError("VulkanPipelineManager: failed to create VkShaderModule for " + spv_path);
    }
    return mod;
}

void VulkanPipelineManager::initialize_pipelines(size_t descriptor_capacity) {
    VkDevice dev = ctx_.device();

    // 1. Create Descriptor Pool
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<uint32_t>(descriptor_capacity * 8) },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(descriptor_capacity * 2) }
    };
    VkDescriptorPoolCreateInfo pool_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_ci.maxSets = static_cast<uint32_t>(descriptor_capacity);
    pool_ci.poolSizeCount = 2;
    pool_ci.pPoolSizes = pool_sizes;
    pool_ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    VkResult res = vkCreateDescriptorPool(dev, &pool_ci, nullptr, &descriptor_pool_);
    if (res != VK_SUCCESS) {
        throw G4DenseFormatError("VulkanPipelineManager: failed to create descriptor pool");
    }

    // 2. Load SPIR-V modules and build pipelines
    std::vector<std::string> search_dirs = {
        "build/shaders",
        "shaders",
        "../build/shaders",
        "../../build/shaders"
    };

    for (uint32_t k = 0; k < static_cast<uint32_t>(ComputeKernel::COUNT); ++k) {
        ComputeKernel kernel = static_cast<ComputeKernel>(k);
        const char* spv_name = get_kernel_spv_name(kernel);
        if (!spv_name || strlen(spv_name) == 0) continue;

        std::string found_path;
        for (const auto& d : search_dirs) {
            std::filesystem::path p = std::filesystem::path(d) / spv_name;
            if (std::filesystem::exists(p)) {
                found_path = p.string();
                break;
            }
        }

        if (found_path.empty()) {
            throw G4DenseFormatError(std::string("VulkanPipelineManager: missing shader ") + spv_name);
        }

        VkShaderModule sm = load_shader_module(found_path);
        shader_modules_[kernel] = sm;

        // Create Descriptor Set Layout (Up to 8 Storage Buffers)
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t b = 0; b < 8; ++b) {
            VkDescriptorSetLayoutBinding bind{};
            bind.binding = b;
            bind.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bind.descriptorCount = 1;
            bind.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(bind);
        }

        VkDescriptorSetLayoutCreateInfo dsl_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsl_ci.bindingCount = static_cast<uint32_t>(bindings.size());
        dsl_ci.pBindings = bindings.data();

        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        vkCreateDescriptorSetLayout(dev, &dsl_ci, nullptr, &dsl);
        desc_layouts_[kernel] = dsl;

        // Push constant range (128 bytes)
        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc_range.offset = 0;
        pc_range.size = 128;

        VkPipelineLayoutCreateInfo pl_ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &dsl;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &pc_range;

        VkPipelineLayout pl = VK_NULL_HANDLE;
        vkCreatePipelineLayout(dev, &pl_ci, nullptr, &pl);
        pipeline_layouts_[kernel] = pl;

        // Pipeline stage with Wave32 Subgroup Control
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size_ci{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT
        };
        subgroup_size_ci.requiredSubgroupSize = 32;

        VkPipelineShaderStageCreateInfo stage_ci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage_ci.pNext = &subgroup_size_ci;
        stage_ci.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        stage_ci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_ci.module = sm;
        stage_ci.pName = "main";

        VkComputePipelineCreateInfo cp_ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp_ci.stage = stage_ci;
        cp_ci.layout = pl;

        VkPipeline pipe = VK_NULL_HANDLE;
        res = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &pipe);
        if (res != VK_SUCCESS) {
            // Fallback without subgroup size control if driver rejects
            stage_ci.pNext = nullptr;
            stage_ci.flags = 0;
            res = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &pipe);
            if (res != VK_SUCCESS) {
                throw G4DenseFormatError("VulkanPipelineManager: failed to create compute pipeline");
            }
        }
        pipelines_[kernel] = pipe;
    }
}

VkPipeline VulkanPipelineManager::get_pipeline(ComputeKernel kernel) const {
    auto it = pipelines_.find(kernel);
    return it != pipelines_.end() ? it->second : VK_NULL_HANDLE;
}

VkPipelineLayout VulkanPipelineManager::get_pipeline_layout(ComputeKernel kernel) const {
    auto it = pipeline_layouts_.find(kernel);
    return it != pipeline_layouts_.end() ? it->second : VK_NULL_HANDLE;
}

void VulkanPipelineManager::bind_kernel(VkCommandBuffer cmd, ComputeKernel kernel) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, get_pipeline(kernel));
}

void VulkanPipelineManager::push_constants(VkCommandBuffer cmd, ComputeKernel kernel,
                                          const void* data, size_t size_bytes) {
    vkCmdPushConstants(cmd, get_pipeline_layout(kernel), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, static_cast<uint32_t>(size_bytes), data);
}

void VulkanPipelineManager::dispatch(VkCommandBuffer cmd, uint32_t group_x, uint32_t group_y, uint32_t group_z) {
    vkCmdDispatch(cmd, group_x, group_y, group_z);
}

VkDescriptorSet VulkanPipelineManager::allocate_descriptor_set(ComputeKernel kernel) {
    VkDescriptorSetAllocateInfo ds_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ds_ai.descriptorPool = descriptor_pool_;
    ds_ai.descriptorSetCount = 1;
    ds_ai.pSetLayouts = &desc_layouts_[kernel];

    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkResult res = vkAllocateDescriptorSets(ctx_.device(), &ds_ai, &ds);
    if (res != VK_SUCCESS) {
        throw G4DenseFormatError("VulkanPipelineManager: failed to allocate descriptor set");
    }
    return ds;
}

void VulkanPipelineManager::update_storage_buffer(VkDescriptorSet ds, uint32_t binding,
                                                  VkBuffer buffer, uint64_t offset, uint64_t range) {
    VkDescriptorBufferInfo buf_info{};
    buf_info.buffer = buffer;
    buf_info.offset = offset;
    buf_info.range = range;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = ds;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buf_info;

    vkUpdateDescriptorSets(ctx_.device(), 1, &write, 0, nullptr);
}

} // namespace g4dense
