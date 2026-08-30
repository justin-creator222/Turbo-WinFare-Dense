#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace g4dense {

enum class MemoryResidency {
    // Host-visible and CPU-writable. NOT necessarily cached: the first type matching
    // HOST_VISIBLE | HOST_COHERENT is usually write-combined, which is fine for GPU reads but
    // slow as a destination for large CPU writes. Use HostCachedMapped when the CPU writes it.
    HostVisibleMapped = 0,
    DeviceLocalStaged = 1, // DEVICE_LOCAL - fast for the GPU, populated via staging upload
    Auto = 2,              // Prefers DEVICE_LOCAL; falls back to host-visible, and says so
    // Host-visible AND HOST_CACHED. For buffers the CPU fills in bulk (streamed weights):
    // ReadFile into write-combined memory measured 1.44 GB/s on this machine.
    HostCachedMapped = 3
};

struct VkMemoryAllocation {
    VkBuffer buffer{VK_NULL_HANDLE};
    // True when the memory was imported from a host pointer the caller owns, so freeing this
    // allocation must not touch the underlying pages.
    bool is_imported{false};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    void* mapped_ptr{nullptr};
    uint64_t size_bytes{0};
    uint32_t memory_type_index{0};
    bool is_host_visible{false};
    bool is_device_local{false};
    MemoryResidency requested_residency{MemoryResidency::Auto};
    bool was_downgraded{false};
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
    const VkPhysicalDeviceMemoryProperties& memory_properties() const { return memory_properties_; }

    // Allocates a buffer on requested heap (Heap 0 DeviceLocal vs Heap 1 HostVisible)
    VkMemoryAllocation allocate_buffer(uint64_t size_bytes, VkBufferUsageFlags usage,
                                       MemoryResidency residency = MemoryResidency::HostVisibleMapped);
    void free_buffer(VkMemoryAllocation& alloc);

    // Layer circular pool (4 buffers of layer_bytes each)
    void init_layer_pool(size_t buffer_count, uint64_t layer_bytes,
                         MemoryResidency residency = MemoryResidency::HostVisibleMapped);
    VkMemoryAllocation& get_layer_buffer(size_t index);
    size_t layer_pool_size() const { return layer_pool_.size(); }

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags required_flags,
                              VkMemoryPropertyFlags preferred_flags = 0, bool* out_downgraded = nullptr) const;

    // Wraps an existing host allocation as a VkBuffer WITHOUT copying it
    // (VK_EXT_external_memory_host).
    //
    // This is the whole point on a UMA APU: the OS page cache already holds the model in the
    // same physical DRAM the GPU reads, so copying page-cache -> "GPU buffer" is RAM-to-RAM for
    // no benefit. It measured 10.06 s of the 11.1 s forward pass.
    //
    // `host_ptr` and `size_bytes` must both be multiples of min_imported_host_pointer_alignment()
    // (4096 here, which every .g4dense layer offset already satisfies). The caller owns the
    // memory and must keep it mapped and unmoved for the lifetime of the returned buffer;
    // free_buffer() releases the Vulkan objects but never the host pages.
    VkMemoryAllocation import_host_buffer(void* host_ptr, uint64_t size_bytes,
                                          VkBufferUsageFlags usage = 0);

    bool has_external_memory_host() const { return has_external_memory_host_; }

    // Device memory still allocatable, summed over all heaps, queried live.
    //
    // Needed because importing host pointers consumes the same budget as ordinary allocations
    // but is not bounded by any single heap's size, so "how much is left" cannot be inferred
    // from what we have asked for. Returns 0 if VK_EXT_memory_budget is unavailable, which
    // callers must read as "unknown", not "none".
    uint64_t available_device_memory() const;
    bool has_memory_budget() const { return has_memory_budget_; }
    uint64_t min_imported_host_pointer_alignment() const { return min_imported_host_ptr_align_; }

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
    bool has_external_memory_host_{false};
    bool has_memory_budget_{false};
    uint64_t min_imported_host_ptr_align_{4096};
    PFN_vkGetMemoryHostPointerPropertiesEXT vk_get_memory_host_pointer_properties_{nullptr};

    VkPhysicalDeviceMemoryProperties memory_properties_{};
    std::vector<VkMemoryAllocation> layer_pool_;
};

} // namespace g4dense
