#include "g4dense/vk_context.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace g4dense;

int main() {
    std::cout << "========================================================\n"
              << "  Turbo-WinFare Dense: Vulkan Device & Heap Verification \n"
              << "========================================================\n";

    VulkanContext ctx;
    ctx.initialize();

    std::cout << "Device Name:    " << ctx.device_name() << "\n"
              << "Default Wave:   " << ctx.subgroup_size() << "\n"
              << "Coop Matrix:    " << (ctx.has_cooperative_matrix() ? "Supported" : "Not supported") << "\n\n";

    // Query Subgroup Size Control properties
    VkPhysicalDeviceSubgroupSizeControlProperties size_ctrl_props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES
    };
    VkPhysicalDeviceProperties2 dev_props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    dev_props2.pNext = &size_ctrl_props;
    vkGetPhysicalDeviceProperties2(ctx.physical_device(), &dev_props2);

    std::cout << "Subgroup Control Properties:\n"
              << "  Min Subgroup Size:      " << size_ctrl_props.minSubgroupSize << "\n"
              << "  Max Subgroup Size:      " << size_ctrl_props.maxSubgroupSize << "\n"
              << "  Required Subgroup Size: " << ((size_ctrl_props.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) ? "Supported in COMPUTE" : "Not supported") << "\n\n";

    assert(size_ctrl_props.minSubgroupSize <= 32 && size_ctrl_props.maxSubgroupSize >= 32);

    const auto& mem_props = ctx.memory_properties();
    std::cout << "Memory Heaps (" << mem_props.memoryHeapCount << "):\n";
    std::cout << "  Heap | Size (GiB) | Flags\n";
    std::cout << "  -----+------------+-----------------------------------------\n";

    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        double gib = static_cast<double>(mem_props.memoryHeaps[i].size) / (1024.0 * 1024.0 * 1024.0);
        std::string flags;
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) flags += "DEVICE_LOCAL ";
        if (flags.empty()) flags = "HOST_ONLY / SYSTEM";

        std::cout << "    " << i << "  | " << std::setw(10) << std::fixed << std::setprecision(2) << gib << " | " << flags << "\n";
    }

    std::cout << "\nMemory Types (" << mem_props.memoryTypeCount << "):\n";
    std::cout << "  Type | Heap | Property Flags\n";
    std::cout << "  -----+------+-------------------------------------------------\n";

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        std::string props;
        VkMemoryPropertyFlags f = mem_props.memoryTypes[i].propertyFlags;
        if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) props += "DEVICE_LOCAL ";
        if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) props += "HOST_VISIBLE ";
        if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) props += "HOST_COHERENT ";
        if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) props += "HOST_CACHED ";
        if (props.empty()) props = "NONE";

        std::cout << "   " << std::setw(2) << i << "  |  " << mem_props.memoryTypes[i].heapIndex << "   | " << props << "\n";
    }

    // Verify Two-Heap Allocation
    std::cout << "\nVerifying Two-Heap Allocator:\n";

    // 1. Allocate on Heap 1 (HostVisibleMapped)
    VkMemoryAllocation host_alloc = ctx.allocate_buffer(64 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
    std::cout << "  Host-Visible Buffer (64 MB): type=" << host_alloc.memory_type_index
              << ", is_host_visible=" << host_alloc.is_host_visible
              << ", mapped=" << (host_alloc.mapped_ptr != nullptr ? "YES" : "NO") << "\n";
    assert(host_alloc.is_host_visible);
    assert(host_alloc.mapped_ptr != nullptr);

    // 2. Allocate on Heap 0 (DeviceLocalStaged)
    VkMemoryAllocation dev_alloc = ctx.allocate_buffer(64 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::DeviceLocalStaged);
    std::cout << "  Device-Local Buffer (64 MB): type=" << dev_alloc.memory_type_index
              << ", is_device_local=" << dev_alloc.is_device_local << "\n";
    assert(dev_alloc.is_device_local);

    ctx.free_buffer(host_alloc);
    ctx.free_buffer(dev_alloc);

    std::cout << "\n[test_vk_device] ALL VULKAN DEVICE AND HEAP CHECKS PASSED!\n";
    return 0;
}
