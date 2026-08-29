#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace g4dense {

struct VkMemoryAllocation {
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    void* mapped_ptr{nullptr};
    uint64_t size_bytes{0};
    uint32_t memory_type_index{0};
    bool is_host_visible{false};
    bool is_device_local{false};
};

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    // Initializes Vulkan 1.3, selects physical device, enables subgroup size control,
    // creates logical compute device and queue. Throws on missing required features.
    void initialize();

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physical_device() const { return physical_device_; }
    VkDevice device() const { return device_; }
    VkQueue compute_queue() const { return compute_queue_; }
    uint32_t compute_queue_family() const { return compute_queue_family_; }

    const std::string& device_name() const { return device_name_; }
    uint32_t subgroup_size() const { return subgroup_size_; }
    bool has_cooperative_matrix() const { return has_coop_matrix_; }

    // Allocates a zero-copy host-visible UMA buffer
    VkMemoryAllocation allocate_buffer(uint64_t size_bytes, VkBufferUsageFlags usage,
                                       bool prefer_device_local = true);
    void free_buffer(VkMemoryAllocation& alloc);

    // Layer circular pool (4 buffers of layer_bytes each)
    void init_layer_pool(size_t buffer_count, uint64_t layer_bytes);
    VkMemoryAllocation& get_layer_buffer(size_t index);
    size_t layer_pool_size() const { return layer_pool_.size(); }

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const;

private:
    VkInstance instance_{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
    VkDevice device_{VK_NULL_HANDLE};
    VkQueue compute_queue_{VK_NULL_HANDLE};
    uint32_t compute_queue_family_{0};

    std::string device_name_;
    uint32_t subgroup_size_{32};
    bool has_subgroup_size_control_{false};
    bool has_coop_matrix_{false};

    VkPhysicalDeviceMemoryProperties memory_properties_{};
    std::vector<VkMemoryAllocation> layer_pool_;
};

} // namespace g4dense
