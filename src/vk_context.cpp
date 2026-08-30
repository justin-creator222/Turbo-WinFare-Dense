#include "g4dense/vk_context.hpp"
#include "g4dense/format.hpp"

#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>
#include <string>

namespace g4dense {

VulkanContext::VulkanContext() = default;

VulkanContext::~VulkanContext() {
    for (auto& alloc : layer_pool_) {
        free_buffer(alloc);
    }
    layer_pool_.clear();

    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

uint32_t VulkanContext::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags required_flags,
                                          VkMemoryPropertyFlags preferred_flags, bool* out_downgraded) const {
    if (out_downgraded) *out_downgraded = false;

    // 1. Try exact match for (required | preferred)
    VkMemoryPropertyFlags combined = required_flags | preferred_flags;
    for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (memory_properties_.memoryTypes[i].propertyFlags & combined) == combined) {
            return i;
        }
    }

    // 2. Match required flags only
    if (preferred_flags != 0) {
        for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
            if ((type_filter & (1 << i)) &&
                (memory_properties_.memoryTypes[i].propertyFlags & required_flags) == required_flags) {
                if (out_downgraded) *out_downgraded = true;
                return i;
            }
        }
    }

    std::stringstream ss;
    ss << "VulkanContext: failed to find suitable memory type (type_filter=0x"
       << std::hex << type_filter << ", required=0x" << required_flags
       << ", preferred=0x" << preferred_flags << std::dec << ")";
    throw G4DenseFormatError(ss.str());
}

void VulkanContext::initialize() {
    // 1. Create Instance
    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = "Turbo-WinFare Dense";
    app_info.applicationVersion = VK_MAKE_VERSION(2, 0, 0);
    app_info.pEngineName = "G4Dense";
    app_info.engineVersion = VK_MAKE_VERSION(2, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo inst_ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    inst_ci.pApplicationInfo = &app_info;

    VkResult res = vkCreateInstance(&inst_ci, nullptr, &instance_);
    if (res != VK_SUCCESS) {
        std::stringstream ss;
        ss << "VulkanContext: failed to create Vulkan instance (VkResult=" << res << ")";
        throw G4DenseFormatError(ss.str());
    }

    // 2. Select Physical Device
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance_, &dev_count, nullptr);
    if (dev_count == 0) {
        throw G4DenseFormatError("VulkanContext: no Vulkan-capable physical devices found");
    }

    std::vector<VkPhysicalDevice> devices(dev_count);
    vkEnumeratePhysicalDevices(instance_, &dev_count, devices.data());

    // Prefer integrated / discrete GPU over CPU software rasterizer
    physical_device_ = devices[0];
    VkPhysicalDeviceProperties dev_props{};
    for (VkPhysicalDevice dev : devices) {
        vkGetPhysicalDeviceProperties(dev, &dev_props);
        if (dev_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
            dev_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device_ = dev;
            break;
        }
    }

    vkGetPhysicalDeviceProperties(physical_device_, &dev_props);
    device_name_ = dev_props.deviceName;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);

    // 3. Find Compute Queue Family
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, qf_props.data());

    bool found_compute = false;
    for (uint32_t i = 0; i < qf_count; ++i) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family_ = i;
            found_compute = true;
            break;
        }
    }
    if (!found_compute) {
        throw G4DenseFormatError("VulkanContext: failed to find compute queue family");
    }

    // 4. Check Subgroup Size Control & Cooperative Matrix
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> extensions(ext_count);
    vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, extensions.data());

    std::vector<const char*> enabled_extensions;
    for (const auto& ext : extensions) {
        if (std::strcmp(ext.extensionName, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) == 0) {
            has_subgroup_size_control_ = true;
            enabled_extensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        }
        if (std::strcmp(ext.extensionName, "VK_KHR_cooperative_matrix") == 0) {
            has_coop_matrix_ = true;
            enabled_extensions.push_back("VK_KHR_cooperative_matrix");
        }
        // Lets a VkBuffer be backed by memory the application already owns -- here, the
        // memory-mapped .g4dense container -- so the GPU reads the weights straight out of the
        // OS page cache instead of us copying 14.5 GB into a "GPU buffer" every token.
        if (std::strcmp(ext.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0) {
            has_external_memory_host_ = true;
            enabled_extensions.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        }
        // Lets the layer import stop before it exhausts the device, instead of taking
        // everything and leaving the driver unable to submit work.
        if (std::strcmp(ext.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
            has_memory_budget_ = true;
            enabled_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        }
    }

    // Query subgroup properties (and, when available, the host-pointer import alignment)
    VkPhysicalDeviceSubgroupProperties subgroup_props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT ext_host_props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 dev_props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    dev_props2.pNext = &subgroup_props;
    if (has_external_memory_host_) {
        subgroup_props.pNext = &ext_host_props;
    }
    vkGetPhysicalDeviceProperties2(physical_device_, &dev_props2);
    if (has_external_memory_host_ && ext_host_props.minImportedHostPointerAlignment != 0) {
        min_imported_host_ptr_align_ = ext_host_props.minImportedHostPointerAlignment;
    }
    subgroup_size_ = subgroup_props.subgroupSize;

    // 5. Create Logical Device
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_ci.queueFamilyIndex = compute_queue_family_;
    queue_ci.queueCount = 1;
    queue_ci.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroup_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT
    };
    subgroup_features.subgroupSizeControl = VK_TRUE;
    subgroup_features.computeFullSubgroups = VK_TRUE;

    VkPhysicalDeviceVulkan13Features vk13_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    vk13_features.subgroupSizeControl = has_subgroup_size_control_ ? VK_TRUE : VK_FALSE;
    vk13_features.computeFullSubgroups = has_subgroup_size_control_ ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceFeatures2 dev_features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    dev_features2.pNext = &vk13_features;

    VkDeviceCreateInfo dev_ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dev_ci.pNext = &dev_features2;
    dev_ci.queueCreateInfoCount = 1;
    dev_ci.pQueueCreateInfos = &queue_ci;
    dev_ci.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    dev_ci.ppEnabledExtensionNames = enabled_extensions.data();

    res = vkCreateDevice(physical_device_, &dev_ci, nullptr, &device_);
    if (res != VK_SUCCESS) {
        std::stringstream ss;
        ss << "VulkanContext: failed to create logical device (VkResult=" << res << ")";
        throw G4DenseFormatError(ss.str());
    }

    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
}

VkMemoryAllocation VulkanContext::allocate_buffer(uint64_t size_bytes, VkBufferUsageFlags usage,
                                                  MemoryResidency residency) {
    VkMemoryAllocation alloc{};
    alloc.size_bytes = size_bytes;
    alloc.requested_residency = residency;

    VkBufferCreateInfo buf_ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buf_ci.size = size_bytes;
    buf_ci.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = vkCreateBuffer(device_, &buf_ci, nullptr, &alloc.buffer);
    if (res != VK_SUCCESS) {
        throw G4DenseFormatError("VulkanContext: failed to create VkBuffer");
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_, alloc.buffer, &mem_reqs);

    VkMemoryPropertyFlags required_flags = 0;
    VkMemoryPropertyFlags preferred_flags = 0;

    switch (residency) {
        case MemoryResidency::DeviceLocalStaged:
            required_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
        case MemoryResidency::HostVisibleMapped:
            required_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
        case MemoryResidency::HostCachedMapped:
            // HOST_CACHED is required, not preferred: the caller asked for this residency
            // because the CPU writes the buffer in bulk, and silently handing back
            // write-combined memory is the thing that cost 10 s per forward pass.
            required_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                             VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            break;
        case MemoryResidency::Auto:
        default:
            preferred_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            required_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
    }

    bool downgraded = false;
    alloc.memory_type_index = find_memory_type(mem_reqs.memoryTypeBits, required_flags, preferred_flags, &downgraded);
    alloc.was_downgraded = downgraded;
    alloc.is_host_visible = (memory_properties_.memoryTypes[alloc.memory_type_index].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    alloc.is_device_local = (memory_properties_.memoryTypes[alloc.memory_type_index].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;

    if (downgraded) {
        std::cerr << "[VulkanContext] WARNING: Residency downgraded for buffer ("
                  << size_bytes / (1024 * 1024) << " MB) from Heap 0 (DeviceLocal) to Heap 1 (HostVisible)"
                  << std::endl;
    }

    VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = alloc.memory_type_index;

    res = vkAllocateMemory(device_, &alloc_info, nullptr, &alloc.memory);
    if (res != VK_SUCCESS) {
        vkDestroyBuffer(device_, alloc.buffer, nullptr);
        std::stringstream ss;
        ss << "VulkanContext: failed to allocate VkDeviceMemory (" << mem_reqs.size << " bytes on type " << alloc.memory_type_index << ")";
        throw G4DenseFormatError(ss.str());
    }

    vkBindBufferMemory(device_, alloc.buffer, alloc.memory, 0);

    if (alloc.is_host_visible) {
        vkMapMemory(device_, alloc.memory, 0, size_bytes, 0, &alloc.mapped_ptr);
    }

    return alloc;
}

uint64_t VulkanContext::available_device_memory() const {
    if (!has_memory_budget_) return 0;

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    props2.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(physical_device_, &props2);

    uint64_t free_bytes = 0;
    for (uint32_t h = 0; h < props2.memoryProperties.memoryHeapCount; ++h) {
        if (budget.heapBudget[h] > budget.heapUsage[h]) {
            free_bytes += budget.heapBudget[h] - budget.heapUsage[h];
        }
    }
    return free_bytes;
}

VkMemoryAllocation VulkanContext::import_host_buffer(void* host_ptr, uint64_t size_bytes,
                                                     VkBufferUsageFlags usage) {
    if (!has_external_memory_host_) {
        throw G4DenseFormatError(
            "VulkanContext::import_host_buffer: VK_EXT_external_memory_host is not available");
    }
    if (!vk_get_memory_host_pointer_properties_) {
        vk_get_memory_host_pointer_properties_ =
            reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
                vkGetDeviceProcAddr(device_, "vkGetMemoryHostPointerPropertiesEXT"));
        if (!vk_get_memory_host_pointer_properties_) {
            throw G4DenseFormatError(
                "VulkanContext::import_host_buffer: vkGetMemoryHostPointerPropertiesEXT missing");
        }
    }

    const uint64_t align = min_imported_host_ptr_align_;
    if ((reinterpret_cast<uintptr_t>(host_ptr) % align) != 0) {
        throw G4DenseFormatError("VulkanContext::import_host_buffer: host pointer is not aligned to " +
                                 std::to_string(align));
    }
    // Vulkan requires the imported range itself to be a whole number of alignment units. Every
    // .g4dense layer is 4096-aligned and 4096-padded, so this rounds up by zero in practice; it
    // is here so a differently-padded container fails loudly rather than at bind time.
    const uint64_t aligned_size = ((size_bytes + align - 1) / align) * align;

    VkMemoryAllocation alloc{};
    alloc.size_bytes = aligned_size;
    alloc.is_imported = true;
    alloc.requested_residency = MemoryResidency::HostCachedMapped;

    VkExternalMemoryBufferCreateInfo ext_buf_ci{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext_buf_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

    VkBufferCreateInfo buf_ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buf_ci.pNext = &ext_buf_ci;
    buf_ci.size = aligned_size;
    buf_ci.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = vkCreateBuffer(device_, &buf_ci, nullptr, &alloc.buffer);
    if (res != VK_SUCCESS) {
        throw G4DenseFormatError("VulkanContext::import_host_buffer: vkCreateBuffer failed (VkResult=" +
                                 std::to_string(res) + ")");
    }

    VkMemoryHostPointerPropertiesEXT host_props{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    res = vk_get_memory_host_pointer_properties_(
        device_, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host_ptr, &host_props);
    if (res != VK_SUCCESS || host_props.memoryTypeBits == 0) {
        vkDestroyBuffer(device_, alloc.buffer, nullptr);
        throw G4DenseFormatError(
            "VulkanContext::import_host_buffer: this pointer cannot be imported (VkResult=" +
            std::to_string(res) + ", memoryTypeBits=" + std::to_string(host_props.memoryTypeBits) + ")");
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_, alloc.buffer, &mem_reqs);
    const uint32_t candidates = host_props.memoryTypeBits & mem_reqs.memoryTypeBits;
    if (candidates == 0) {
        vkDestroyBuffer(device_, alloc.buffer, nullptr);
        throw G4DenseFormatError("VulkanContext::import_host_buffer: no memory type satisfies both the "
                                 "buffer and the imported pointer");
    }

    // Prefer a cached type: these pages are the OS page cache, and the CPU may still fault them
    // in. Fall back to any host-visible type the import allows.
    bool downgraded = false;
    uint32_t type_index;
    try {
        type_index = find_memory_type(candidates,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    } catch (const G4DenseFormatError&) {
        type_index = find_memory_type(candidates, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        downgraded = true;
    }
    alloc.memory_type_index = type_index;
    alloc.was_downgraded = downgraded;
    alloc.is_host_visible = true;
    alloc.is_device_local =
        (memory_properties_.memoryTypes[type_index].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;

    VkImportMemoryHostPointerInfoEXT import_info{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    import_info.pHostPointer = host_ptr;

    VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.pNext = &import_info;
    alloc_info.allocationSize = aligned_size;
    alloc_info.memoryTypeIndex = type_index;

    res = vkAllocateMemory(device_, &alloc_info, nullptr, &alloc.memory);
    if (res != VK_SUCCESS) {
        vkDestroyBuffer(device_, alloc.buffer, nullptr);
        throw G4DenseFormatError("VulkanContext::import_host_buffer: vkAllocateMemory failed (VkResult=" +
                                 std::to_string(res) + ") importing " +
                                 std::to_string(aligned_size / (1024 * 1024)) + " MB");
    }

    res = vkBindBufferMemory(device_, alloc.buffer, alloc.memory, 0);
    if (res != VK_SUCCESS) {
        vkFreeMemory(device_, alloc.memory, nullptr);
        vkDestroyBuffer(device_, alloc.buffer, nullptr);
        throw G4DenseFormatError("VulkanContext::import_host_buffer: vkBindBufferMemory failed");
    }

    // Deliberately NOT mapped: the caller already has the host pointer, and mapping imported
    // memory a second time buys nothing.
    alloc.mapped_ptr = host_ptr;
    return alloc;
}

void VulkanContext::free_buffer(VkMemoryAllocation& alloc) {
    // Imported memory is owned by the caller (an mmap view, typically). Unmapping or freeing it
    // here would tear down memory Vulkan never allocated.
    if (alloc.is_imported) {
        if (alloc.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, alloc.buffer, nullptr);
            alloc.buffer = VK_NULL_HANDLE;
        }
        if (alloc.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, alloc.memory, nullptr);
            alloc.memory = VK_NULL_HANDLE;
        }
        alloc.mapped_ptr = nullptr;
        return;
    }
    if (alloc.mapped_ptr) {
        vkUnmapMemory(device_, alloc.memory);
        alloc.mapped_ptr = nullptr;
    }
    if (alloc.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, alloc.buffer, nullptr);
        alloc.buffer = VK_NULL_HANDLE;
    }
    if (alloc.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_, alloc.memory, nullptr);
        alloc.memory = VK_NULL_HANDLE;
    }
}

void VulkanContext::init_layer_pool(size_t buffer_count, uint64_t layer_bytes, MemoryResidency residency) {
    for (auto& a : layer_pool_) free_buffer(a);
    layer_pool_.clear();

    for (size_t i = 0; i < buffer_count; ++i) {
        layer_pool_.push_back(allocate_buffer(layer_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, residency));
    }
}

VkMemoryAllocation& VulkanContext::get_layer_buffer(size_t index) {
    if (index >= layer_pool_.size()) {
        throw G4DenseFormatError("VulkanContext: layer buffer index out of range");
    }
    return layer_pool_[index];
}

} // namespace g4dense
