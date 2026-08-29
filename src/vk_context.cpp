#include "g4dense/vk_context.hpp"
#include "g4dense/format.hpp"

#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>

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

uint32_t VulkanContext::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const {
    for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (memory_properties_.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    // Fallback: match any host-visible if requested
    for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (memory_properties_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            return i;
        }
    }
    throw G4DenseFormatError("VulkanContext: failed to find suitable memory type");
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
    }

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
                                                 bool prefer_device_local) {
    VkMemoryAllocation alloc{};
    alloc.size_bytes = size_bytes;

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

    VkMemoryPropertyFlags prop_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (prefer_device_local) {
        prop_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }

    alloc.memory_type_index = find_memory_type(mem_reqs.memoryTypeBits, prop_flags);
    alloc.is_host_visible = (memory_properties_.memoryTypes[alloc.memory_type_index].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    alloc.is_device_local = (memory_properties_.memoryTypes[alloc.memory_type_index].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;

    VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = alloc.memory_type_index;

    res = vkAllocateMemory(device_, &alloc_info, nullptr, &alloc.memory);
    if (res != VK_SUCCESS) {
        vkDestroyBuffer(device_, alloc.buffer, nullptr);
        throw G4DenseFormatError("VulkanContext: failed to allocate VkDeviceMemory");
    }

    vkBindBufferMemory(device_, alloc.buffer, alloc.memory, 0);

    if (alloc.is_host_visible) {
        vkMapMemory(device_, alloc.memory, 0, size_bytes, 0, &alloc.mapped_ptr);
    }

    return alloc;
}

void VulkanContext::free_buffer(VkMemoryAllocation& alloc) {
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

void VulkanContext::init_layer_pool(size_t buffer_count, uint64_t layer_bytes) {
    for (auto& a : layer_pool_) free_buffer(a);
    layer_pool_.clear();

    for (size_t i = 0; i < buffer_count; ++i) {
        layer_pool_.push_back(allocate_buffer(layer_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true));
    }
}

VkMemoryAllocation& VulkanContext::get_layer_buffer(size_t index) {
    if (index >= layer_pool_.size()) {
        throw G4DenseFormatError("VulkanContext: layer buffer index out of range");
    }
    return layer_pool_[index];
}

} // namespace g4dense
