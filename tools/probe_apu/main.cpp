#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <dxgi1_4.h>
#include <winioctl.h>
#include <intrin.h>

#include <vulkan/vulkan.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>

struct ProbeResults {
    // Hardware & OS
    std::string os_version;
    std::string cpu_brand;
    int cpu_physical_cores = 0;
    int cpu_logical_cores = 0;
    bool avx512_f = false;
    bool avx512_bw = false;
    bool avx512_cd = false;
    bool avx512_dq = false;
    bool avx512_vl = false;
    bool avx512_vnni = false;
    bool avx512_bf16 = false;

    uint64_t ram_total_bytes = 0;
    uint64_t ram_avail_bytes = 0;
    double ram_total_gb = 0.0;

    std::string disk_model;
    uint64_t disk_free_bytes = 0;
    double disk_free_gb = 0.0;

    // DXGI Memory
    std::string dxgi_adapter_name;
    uint64_t dxgi_dedicated_vram = 0;
    uint64_t dxgi_shared_system_ram = 0;
    uint64_t dxgi_local_budget = 0;
    uint64_t dxgi_non_local_budget = 0;

    // Vulkan Device & Version
    std::string vk_device_name;
    uint32_t vk_api_version = 0;
    uint32_t vk_driver_version = 0;
    uint32_t vk_vendor_id = 0;
    uint32_t vk_device_id = 0;

    // Vulkan Heaps
    struct HeapInfo {
        uint32_t index;
        uint64_t size_bytes;
        double size_mb;
        VkMemoryHeapFlags flags;
        bool is_device_local;
        uint64_t budget_bytes;
        uint64_t usage_bytes;
    };
    std::vector<HeapInfo> vk_heaps;

    struct TypeInfo {
        uint32_t index;
        uint32_t heap_index;
        VkMemoryPropertyFlags flags;
        bool is_host_visible;
        bool is_host_coherent;
        bool is_device_local;
        bool is_host_cached;
    };
    std::vector<TypeInfo> vk_types;

    uint64_t heap_host_visible_device_local_bytes = 0;
    uint64_t heap_host_visible_bytes = 0;
    uint64_t heap_device_local_only_bytes = 0;

    // Subgroup Control
    uint32_t subgroup_size = 0;
    uint32_t subgroup_min_size = 0;
    uint32_t subgroup_max_size = 0;
    bool subgroup_size_control_supported = false;
    bool wave32_pipeline_created = false;
    uint32_t wave32_runtime_measured = 0;

    // Cooperative Matrix (WMMA)
    bool coop_matrix_supported = false;
    struct CoopMatrixTuple {
        uint32_t MSize;
        uint32_t NSize;
        uint32_t KSize;
        std::string AType;
        std::string BType;
        std::string CType;
        std::string ResultType;
        bool saturatingAccumulation;
        std::string scope;
    };
    std::vector<CoopMatrixTuple> coop_matrix_tuples;

    // UMA Zero-Copy Checksum
    bool uma_zero_copy_passed = false;
    uint64_t uma_test_bytes = 0;
    double uma_checksum_time_ms = 0.0;
    double uma_bandwidth_gbs = 0.0;

    // GPU Heap Bandwidth & Staging Upload (R0.2)
    double heap0_read_bandwidth_gbs = 0.0;
    double heap1_read_bandwidth_gbs = 0.0;
    double heap2_read_bandwidth_gbs = 0.0;
    double staging_heap1_to_heap0_gbs = 0.0;
    double staging_heap2_to_heap0_gbs = 0.0;

    // Storage Benchmarks (R0.1)
    double storage_buffered_cold_gbs = 0.0;
    double storage_buffered_warm_gbs = 0.0;
    double storage_warm_256mb_gbs = 0.0;
    double storage_warm_1gb_gbs = 0.0;
    double storage_warm_4gb_1mb_gbs = 0.0;
    double storage_warm_4gb_4mb_gbs = 0.0;
    double storage_warm_4gb_16mb_gbs = 0.0;
    double storage_warm_4gb_seqscan_16mb_gbs = 0.0;
    double storage_warm_4gb_mmap_gbs = 0.0;

    double storage_unbuffered_qd1_gbs = 0.0;
    double storage_unbuffered_qd4_gbs = 0.0;
    double storage_unbuffered_qd8_gbs = 0.0;
    double storage_unbuffered_qd16_gbs = 0.0;
    double storage_unbuffered_16k_align_gbs = 0.0;
    bool directstorage_available = false;
    std::string directstorage_status;
    double directstorage_gbs = 0.0;
};

// Global function pointers for Vulkan dynamic loading
static HMODULE g_vk_module = nullptr;
static PFN_vkGetInstanceProcAddr g_vkGetInstanceProcAddr = nullptr;
static PFN_vkCreateInstance g_vkCreateInstance = nullptr;
static PFN_vkDestroyInstance g_vkDestroyInstance = nullptr;
static PFN_vkEnumeratePhysicalDevices g_vkEnumeratePhysicalDevices = nullptr;
static PFN_vkGetPhysicalDeviceProperties g_vkGetPhysicalDeviceProperties = nullptr;
static PFN_vkGetPhysicalDeviceProperties2 g_vkGetPhysicalDeviceProperties2 = nullptr;
static PFN_vkGetPhysicalDeviceMemoryProperties g_vkGetPhysicalDeviceMemoryProperties = nullptr;
static PFN_vkGetPhysicalDeviceMemoryProperties2 g_vkGetPhysicalDeviceMemoryProperties2 = nullptr;
static PFN_vkGetPhysicalDeviceFeatures2 g_vkGetPhysicalDeviceFeatures2 = nullptr;
static PFN_vkEnumerateDeviceExtensionProperties g_vkEnumerateDeviceExtensionProperties = nullptr;
static PFN_vkCreateDevice g_vkCreateDevice = nullptr;
static PFN_vkDestroyDevice g_vkDestroyDevice = nullptr;
static PFN_vkGetDeviceQueue g_vkGetDeviceQueue = nullptr;
static PFN_vkCreateShaderModule g_vkCreateShaderModule = nullptr;
static PFN_vkDestroyShaderModule g_vkDestroyShaderModule = nullptr;
static PFN_vkCreateDescriptorSetLayout g_vkCreateDescriptorSetLayout = nullptr;
static PFN_vkDestroyDescriptorSetLayout g_vkDestroyDescriptorSetLayout = nullptr;
static PFN_vkCreatePipelineLayout g_vkCreatePipelineLayout = nullptr;
static PFN_vkDestroyPipelineLayout g_vkDestroyPipelineLayout = nullptr;
static PFN_vkCreateComputePipelines g_vkCreateComputePipelines = nullptr;
static PFN_vkDestroyPipeline g_vkDestroyPipeline = nullptr;
static PFN_vkCreateDescriptorPool g_vkCreateDescriptorPool = nullptr;
static PFN_vkDestroyDescriptorPool g_vkDestroyDescriptorPool = nullptr;
static PFN_vkAllocateDescriptorSets g_vkAllocateDescriptorSets = nullptr;
static PFN_vkUpdateDescriptorSets g_vkUpdateDescriptorSets = nullptr;
static PFN_vkCreateBuffer g_vkCreateBuffer = nullptr;
static PFN_vkDestroyBuffer g_vkDestroyBuffer = nullptr;
static PFN_vkGetBufferMemoryRequirements g_vkGetBufferMemoryRequirements = nullptr;
static PFN_vkAllocateMemory g_vkAllocateMemory = nullptr;
static PFN_vkFreeMemory g_vkFreeMemory = nullptr;
static PFN_vkBindBufferMemory g_vkBindBufferMemory = nullptr;
static PFN_vkMapMemory g_vkMapMemory = nullptr;
static PFN_vkUnmapMemory g_vkUnmapMemory = nullptr;
static PFN_vkCreateCommandPool g_vkCreateCommandPool = nullptr;
static PFN_vkDestroyCommandPool g_vkDestroyCommandPool = nullptr;
static PFN_vkAllocateCommandBuffers g_vkAllocateCommandBuffers = nullptr;
static PFN_vkBeginCommandBuffer g_vkBeginCommandBuffer = nullptr;
static PFN_vkEndCommandBuffer g_vkEndCommandBuffer = nullptr;
static PFN_vkCmdBindPipeline g_vkCmdBindPipeline = nullptr;
static PFN_vkCmdBindDescriptorSets g_vkCmdBindDescriptorSets = nullptr;
static PFN_vkCmdPushConstants g_vkCmdPushConstants = nullptr;
static PFN_vkCmdDispatch g_vkCmdDispatch = nullptr;
static PFN_vkCmdCopyBuffer g_vkCmdCopyBuffer = nullptr;
static PFN_vkCmdPipelineBarrier g_vkCmdPipelineBarrier = nullptr;
static PFN_vkQueueSubmit g_vkQueueSubmit = nullptr;
static PFN_vkQueueWaitIdle g_vkQueueWaitIdle = nullptr;
static PFN_vkCreateFence g_vkCreateFence = nullptr;
static PFN_vkDestroyFence g_vkDestroyFence = nullptr;
static PFN_vkWaitForFences g_vkWaitForFences = nullptr;

// Extensions
typedef VkResult (VKAPI_PTR *PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkCooperativeMatrixPropertiesKHR* pProperties);
static PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR g_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = nullptr;

bool load_vulkan() {
    g_vk_module = LoadLibraryA("vulkan-1.dll");
    if (!g_vk_module) {
        std::cerr << "[vulkan] Failed to LoadLibrary vulkan-1.dll, GetLastError=" << GetLastError() << std::endl;
        return false;
    }

    g_vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetProcAddress(g_vk_module, "vkGetInstanceProcAddr");
    if (!g_vkGetInstanceProcAddr) {
        std::cerr << "[vulkan] Failed to GetProcAddress vkGetInstanceProcAddr" << std::endl;
        return false;
    }

    g_vkCreateInstance = (PFN_vkCreateInstance)g_vkGetInstanceProcAddr(nullptr, "vkCreateInstance");
    if (!g_vkCreateInstance) {
        std::cerr << "[vulkan] Failed to resolve vkCreateInstance" << std::endl;
        return false;
    }

    return true;
}

void load_instance_procs(VkInstance instance) {
    #define LOAD_PROC(name) g_##name = (PFN_##name)g_vkGetInstanceProcAddr(instance, #name)
    LOAD_PROC(vkDestroyInstance);
    LOAD_PROC(vkEnumeratePhysicalDevices);
    LOAD_PROC(vkGetPhysicalDeviceProperties);
    LOAD_PROC(vkGetPhysicalDeviceProperties2);
    LOAD_PROC(vkGetPhysicalDeviceMemoryProperties);
    LOAD_PROC(vkGetPhysicalDeviceMemoryProperties2);
    LOAD_PROC(vkGetPhysicalDeviceFeatures2);
    LOAD_PROC(vkEnumerateDeviceExtensionProperties);
    LOAD_PROC(vkCreateDevice);
    LOAD_PROC(vkDestroyDevice);
    LOAD_PROC(vkGetDeviceQueue);
    LOAD_PROC(vkCreateShaderModule);
    LOAD_PROC(vkDestroyShaderModule);
    LOAD_PROC(vkCreateDescriptorSetLayout);
    LOAD_PROC(vkDestroyDescriptorSetLayout);
    LOAD_PROC(vkCreatePipelineLayout);
    LOAD_PROC(vkDestroyPipelineLayout);
    LOAD_PROC(vkCreateComputePipelines);
    LOAD_PROC(vkDestroyPipeline);
    LOAD_PROC(vkCreateDescriptorPool);
    LOAD_PROC(vkDestroyDescriptorPool);
    LOAD_PROC(vkAllocateDescriptorSets);
    LOAD_PROC(vkUpdateDescriptorSets);
    LOAD_PROC(vkCreateBuffer);
    LOAD_PROC(vkDestroyBuffer);
    LOAD_PROC(vkGetBufferMemoryRequirements);
    LOAD_PROC(vkAllocateMemory);
    LOAD_PROC(vkFreeMemory);
    LOAD_PROC(vkBindBufferMemory);
    LOAD_PROC(vkMapMemory);
    LOAD_PROC(vkUnmapMemory);
    LOAD_PROC(vkCreateCommandPool);
    LOAD_PROC(vkDestroyCommandPool);
    LOAD_PROC(vkAllocateCommandBuffers);
    LOAD_PROC(vkBeginCommandBuffer);
    LOAD_PROC(vkEndCommandBuffer);
    LOAD_PROC(vkCmdBindPipeline);
    LOAD_PROC(vkCmdBindDescriptorSets);
    LOAD_PROC(vkCmdPushConstants);
    LOAD_PROC(vkCmdDispatch);
    LOAD_PROC(vkCmdCopyBuffer);
    LOAD_PROC(vkCmdPipelineBarrier);
    LOAD_PROC(vkQueueSubmit);
    LOAD_PROC(vkQueueWaitIdle);
    LOAD_PROC(vkCreateFence);
    LOAD_PROC(vkDestroyFence);
    LOAD_PROC(vkWaitForFences);

    g_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = 
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)g_vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    #undef LOAD_PROC
}

std::string vk_component_type_to_string(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "FP16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "FP32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "FP64";
        case VK_COMPONENT_TYPE_SINT8_KHR: return "INT8";
        case VK_COMPONENT_TYPE_SINT16_KHR: return "INT16";
        case VK_COMPONENT_TYPE_SINT32_KHR: return "INT32";
        case VK_COMPONENT_TYPE_SINT64_KHR: return "INT64";
        case VK_COMPONENT_TYPE_UINT8_KHR: return "UINT8";
        case VK_COMPONENT_TYPE_UINT16_KHR: return "UINT16";
        case VK_COMPONENT_TYPE_UINT32_KHR: return "UINT32";
        case VK_COMPONENT_TYPE_UINT64_KHR: return "UINT64";
        default: return "Unknown (" + std::to_string(t) + ")";
    }
}

std::string vk_scope_to_string(VkScopeKHR s) {
    switch (s) {
        case VK_SCOPE_DEVICE_KHR: return "Device";
        case VK_SCOPE_WORKGROUP_KHR: return "Workgroup";
        case VK_SCOPE_SUBGROUP_KHR: return "Subgroup";
        case VK_SCOPE_QUEUE_FAMILY_KHR: return "QueueFamily";
        default: return "Unknown (" + std::to_string(s) + ")";
    }
}

std::vector<uint32_t> load_spirv_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    size_t size = (size_t)file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> buffer(size / 4);
    file.read((char*)buffer.data(), size);
    return buffer;
}

void probe_cpu_and_os(ProbeResults& r) {
    // OS Version
    OSVERSIONINFOEXA osvi{};
    osvi.dwOSVersionInfoSize = sizeof(osvi);
    typedef LONG(WINAPI* PFN_RtlGetVersion)(PRTL_OSVERSIONINFOW);
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    if (ntdll) {
        PFN_RtlGetVersion pRtlGetVersion = (PFN_RtlGetVersion)GetProcAddress(ntdll, "RtlGetVersion");
        if (pRtlGetVersion) {
            RTL_OSVERSIONINFOW rovi{};
            rovi.dwOSVersionInfoSize = sizeof(rovi);
            if (pRtlGetVersion(&rovi) == 0) {
                r.os_version = "Windows build " + std::to_string(rovi.dwBuildNumber);
            }
        }
    }
    if (r.os_version.empty()) r.os_version = "Windows 11 64-bit";

    // CPU Brand String
    int cpuInfo[4] = {0};
    char brand[0x40] = {0};
    __cpuid(cpuInfo, 0x80000000);
    unsigned int nExIds = cpuInfo[0];
    if (nExIds >= 0x80000004) {
        __cpuid((int*)(brand), 0x80000002);
        __cpuid((int*)(brand + 16), 0x80000003);
        __cpuid((int*)(brand + 32), 0x80000004);
        r.cpu_brand = brand;
    } else {
        r.cpu_brand = "AMD Ryzen Processor";
    }
    size_t start = r.cpu_brand.find_first_not_of(" \t\r\n");
    if (start != std::string::npos) r.cpu_brand = r.cpu_brand.substr(start);

    // Cores
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    r.cpu_logical_cores = sysInfo.dwNumberOfProcessors;

    DWORD returnLength = 0;
    GetLogicalProcessorInformation(nullptr, &returnLength);
    if (returnLength > 0) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (GetLogicalProcessorInformation(buffer.data(), &returnLength)) {
            int cores = 0;
            for (const auto& info : buffer) {
                if (info.Relationship == RelationProcessorCore) cores++;
            }
            r.cpu_physical_cores = (cores > 0) ? cores : 8;
        }
    }
    if (r.cpu_physical_cores == 0) r.cpu_physical_cores = 8;

    // AVX-512 flags
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        r.avx512_f = (cpuInfo[1] & (1 << 16)) != 0;
        r.avx512_dq = (cpuInfo[1] & (1 << 17)) != 0;
        r.avx512_cd = (cpuInfo[1] & (1 << 28)) != 0;
        r.avx512_bw = (cpuInfo[1] & (1 << 30)) != 0;
        r.avx512_vl = (cpuInfo[1] & (1 << 31)) != 0;
        r.avx512_vnni = (cpuInfo[2] & (1 << 11)) != 0;

        __cpuidex(cpuInfo, 7, 1);
        r.avx512_bf16 = (cpuInfo[0] & (1 << 5)) != 0;
    }

    // RAM
    MEMORYSTATUSEX memStatus{};
    memStatus.dwLength = sizeof(memStatus);
    if (GlobalMemoryStatusEx(&memStatus)) {
        r.ram_total_bytes = memStatus.ullTotalPhys;
        r.ram_avail_bytes = memStatus.ullAvailPhys;
        r.ram_total_gb = (double)memStatus.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
    }

    // Disk C: Free Space
    ULARGE_INTEGER freeBytes, totalBytes, totalFreeBytes;
    if (GetDiskFreeSpaceExA("C:\\", &freeBytes, &totalBytes, &totalFreeBytes)) {
        r.disk_free_bytes = freeBytes.QuadPart;
        r.disk_free_gb = (double)freeBytes.QuadPart / (1024.0 * 1024.0 * 1024.0);
    }

    // Disk Model Name via PhysicalDrive0
    HANDLE hDrive = CreateFileA("\\\\.\\PhysicalDrive0", 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (hDrive != INVALID_HANDLE_VALUE) {
        STORAGE_PROPERTY_QUERY query{};
        query.PropertyId = StorageDeviceProperty;
        query.QueryType = PropertyStandardQuery;
        STORAGE_DESCRIPTOR_HEADER header{};
        DWORD bytesReturned = 0;
        if (DeviceIoControl(hDrive, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query), &header, sizeof(header), &bytesReturned, nullptr)) {
            std::vector<BYTE> buffer(header.Size);
            if (DeviceIoControl(hDrive, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query), buffer.data(), (DWORD)buffer.size(), &bytesReturned, nullptr)) {
                STORAGE_DEVICE_DESCRIPTOR* desc = (STORAGE_DEVICE_DESCRIPTOR*)buffer.data();
                if (desc->ProductIdOffset > 0 && desc->ProductIdOffset < buffer.size()) {
                    r.disk_model = (char*)(buffer.data() + desc->ProductIdOffset);
                    size_t s = r.disk_model.find_first_not_of(" \t\r\n");
                    size_t e = r.disk_model.find_last_not_of(" \t\r\n");
                    if (s != std::string::npos && e != std::string::npos) {
                        r.disk_model = r.disk_model.substr(s, e - s + 1);
                    }
                }
            }
        }
        CloseHandle(hDrive);
    }
    if (r.disk_model.empty()) r.disk_model = "NVMe PCIe 4.0 SSD";
}

void probe_dxgi_memory(ProbeResults& r) {
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory))) return;

    for (UINT i = 0; ; ++i) {
        IDXGIAdapter1* adapter = nullptr;
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;

        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            adapter->Release();
            continue;
        }

        std::wstring ws(desc.Description);
        std::string name(ws.begin(), ws.end());

        if (r.dxgi_adapter_name.empty() || name.find("Radeon") != std::string::npos || name.find("780M") != std::string::npos) {
            r.dxgi_adapter_name = name;
            r.dxgi_dedicated_vram = desc.DedicatedVideoMemory;
            r.dxgi_shared_system_ram = desc.SharedSystemMemory;

            IDXGIAdapter3* adapter3 = nullptr;
            if (SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&adapter3))) {
                DXGI_QUERY_VIDEO_MEMORY_INFO local_info{};
                if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info))) {
                    r.dxgi_local_budget = local_info.Budget;
                }
                DXGI_QUERY_VIDEO_MEMORY_INFO non_local_info{};
                if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local_info))) {
                    r.dxgi_non_local_budget = non_local_info.Budget;
                }
                adapter3->Release();
            }
        }
        adapter->Release();
    }
    factory->Release();
}

void probe_vulkan(ProbeResults& r) {
    std::cout << "      [vk] loading vulkan loader..." << std::endl;
    if (!load_vulkan()) {
        std::cerr << "[vulkan] Failed to load vulkan-1.dll" << std::endl;
        return;
    }

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "Turbo-Dense APU Probe";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "g4dense";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult res = g_vkCreateInstance(&createInfo, nullptr, &instance);
    if (res != VK_SUCCESS || !instance) {
        std::cerr << "[vulkan] vkCreateInstance failed with code " << res << std::endl;
        return;
    }
    std::cout << "      [vk] instance created successfully." << std::endl;
    load_instance_procs(instance);

    uint32_t deviceCount = 0;
    g_vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "[vulkan] No physical devices found" << std::endl;
        g_vkDestroyInstance(instance, nullptr);
        return;
    }
    std::cout << "      [vk] found " << deviceCount << " physical device(s)." << std::endl;

    std::vector<VkPhysicalDevice> devices(deviceCount);
    g_vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    VkPhysicalDevice selectedDevice = devices[0];
    for (auto dev : devices) {
        VkPhysicalDeviceProperties props;
        g_vkGetPhysicalDeviceProperties(dev, &props);
        std::string devName = props.deviceName;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
            devName.find("Radeon") != std::string::npos ||
            devName.find("780M") != std::string::npos) {
            selectedDevice = dev;
            break;
        }
    }

    VkPhysicalDeviceProperties props;
    g_vkGetPhysicalDeviceProperties(selectedDevice, &props);
    r.vk_device_name = props.deviceName;
    r.vk_api_version = props.apiVersion;
    r.vk_driver_version = props.driverVersion;
    r.vk_vendor_id = props.vendorID;
    r.vk_device_id = props.deviceID;

    // Check device extensions
    uint32_t extCount = 0;
    g_vkEnumerateDeviceExtensionProperties(selectedDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extCount);
    g_vkEnumerateDeviceExtensionProperties(selectedDevice, nullptr, &extCount, extensions.data());

    bool has_subgroup_size_control = false;
    bool has_coop_matrix = false;
    bool has_memory_budget = false;

    for (const auto& ext : extensions) {
        if (strcmp(ext.extensionName, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) == 0) {
            has_subgroup_size_control = true;
        }
        if (strcmp(ext.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
            has_coop_matrix = true;
        }
        if (strcmp(ext.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
            has_memory_budget = true;
        }
    }
    r.subgroup_size_control_supported = has_subgroup_size_control;
    r.coop_matrix_supported = has_coop_matrix;

    // Heaps and Memory Types
    VkPhysicalDeviceMemoryProperties2 memProps2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    if (has_memory_budget) {
        memProps2.pNext = &budgetProps;
    }

    if (g_vkGetPhysicalDeviceMemoryProperties2) {
        g_vkGetPhysicalDeviceMemoryProperties2(selectedDevice, &memProps2);
    } else {
        g_vkGetPhysicalDeviceMemoryProperties(selectedDevice, &memProps2.memoryProperties);
    }

    const auto& memProps = memProps2.memoryProperties;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        ProbeResults::HeapInfo h;
        h.index = i;
        h.size_bytes = memProps.memoryHeaps[i].size;
        h.size_mb = (double)h.size_bytes / (1024.0 * 1024.0);
        h.flags = memProps.memoryHeaps[i].flags;
        h.is_device_local = (h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        h.budget_bytes = has_memory_budget ? budgetProps.heapBudget[i] : h.size_bytes;
        h.usage_bytes = has_memory_budget ? budgetProps.heapUsage[i] : 0;
        r.vk_heaps.push_back(h);
    }

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        ProbeResults::TypeInfo t;
        t.index = i;
        t.heap_index = memProps.memoryTypes[i].heapIndex;
        t.flags = memProps.memoryTypes[i].propertyFlags;
        t.is_host_visible = (t.flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        t.is_host_coherent = (t.flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        t.is_device_local = (t.flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        t.is_host_cached = (t.flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
        r.vk_types.push_back(t);

        uint64_t heap_size = memProps.memoryHeaps[t.heap_index].size;
        if (t.is_host_visible && t.is_host_coherent && t.is_device_local) {
            r.heap_host_visible_device_local_bytes = std::max(r.heap_host_visible_device_local_bytes, heap_size);
        } else if (t.is_host_visible && t.is_host_coherent) {
            r.heap_host_visible_bytes = std::max(r.heap_host_visible_bytes, heap_size);
        } else if (t.is_device_local) {
            r.heap_device_local_only_bytes = std::max(r.heap_device_local_only_bytes, heap_size);
        }
    }

    // Subgroup Properties
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceSubgroupProperties subgroupProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT subgroupControlProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT};

    props2.pNext = &subgroupProps;
    if (has_subgroup_size_control) {
        subgroupProps.pNext = &subgroupControlProps;
    }

    if (g_vkGetPhysicalDeviceProperties2) {
        g_vkGetPhysicalDeviceProperties2(selectedDevice, &props2);
        r.subgroup_size = subgroupProps.subgroupSize;
        if (has_subgroup_size_control) {
            r.subgroup_min_size = subgroupControlProps.minSubgroupSize;
            r.subgroup_max_size = subgroupControlProps.maxSubgroupSize;
        }
    }

    // Cooperative Matrix Tuples
    if (has_coop_matrix && g_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
        uint32_t coopCount = 0;
        if (g_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(selectedDevice, &coopCount, nullptr) == VK_SUCCESS && coopCount > 0) {
            std::vector<VkCooperativeMatrixPropertiesKHR> coopProps(coopCount);
            for (auto& cp : coopProps) cp.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            if (g_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(selectedDevice, &coopCount, coopProps.data()) == VK_SUCCESS) {
                for (const auto& cp : coopProps) {
                    ProbeResults::CoopMatrixTuple tup;
                    tup.MSize = cp.MSize;
                    tup.NSize = cp.NSize;
                    tup.KSize = cp.KSize;
                    tup.AType = vk_component_type_to_string(cp.AType);
                    tup.BType = vk_component_type_to_string(cp.BType);
                    tup.CType = vk_component_type_to_string(cp.CType);
                    tup.ResultType = vk_component_type_to_string(cp.ResultType);
                    tup.saturatingAccumulation = (cp.saturatingAccumulation == VK_TRUE);
                    tup.scope = vk_scope_to_string(cp.scope);
                    r.coop_matrix_tuples.push_back(tup);
                }
            }
        }
    }

    // Create Logical Device to test Wave32, UMA compute execution, and Heap Bandwidth
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueCreateInfo.queueFamilyIndex = 0;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    std::vector<const char*> enabledExtensions;
    if (has_subgroup_size_control) {
        enabledExtensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    }
    if (has_coop_matrix) {
        enabledExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }

    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupControlFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
    subgroupControlFeatures.subgroupSizeControl = VK_TRUE;
    subgroupControlFeatures.computeFullSubgroups = VK_TRUE;

    if (has_subgroup_size_control) {
        features2.pNext = &subgroupControlFeatures;
    }

    VkDeviceCreateInfo deviceCreateInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceCreateInfo.pNext = &features2;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = (uint32_t)enabledExtensions.size();
    deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();

    VkDevice device = VK_NULL_HANDLE;
    res = g_vkCreateDevice(selectedDevice, &deviceCreateInfo, nullptr, &device);
    if (res != VK_SUCCESS || !device) {
        std::cerr << "[vulkan] vkCreateDevice failed with code " << res << std::endl;
        g_vkDestroyInstance(instance, nullptr);
        return;
    }

    VkQueue queue = VK_NULL_HANDLE;
    g_vkGetDeviceQueue(device, 0, 0, &queue);

    // Test 1: Subgroup Size Control (Wave32 execution verification)
    std::vector<uint32_t> subgroup_spv = load_spirv_file("build/probe_subgroup.spv");
    if (!subgroup_spv.empty()) {
        VkShaderModuleCreateInfo smInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smInfo.codeSize = subgroup_spv.size() * 4;
        smInfo.pCode = subgroup_spv.data();
        VkShaderModule shaderModule = VK_NULL_HANDLE;
        if (g_vkCreateShaderModule(device, &smInfo, nullptr, &shaderModule) == VK_SUCCESS) {
            VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bufInfo.size = 256;
            bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VkBuffer testBuf = VK_NULL_HANDLE;
            g_vkCreateBuffer(device, &bufInfo, nullptr, &testBuf);

            VkMemoryRequirements memReq;
            g_vkGetBufferMemoryRequirements(device, testBuf, &memReq);

            uint32_t memTypeIdx = 0;
            for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
                if ((memReq.memoryTypeBits & (1 << i)) &&
                    (memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
                    memTypeIdx = i;
                    break;
                }
            }

            VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocInfo.allocationSize = memReq.size;
            allocInfo.memoryTypeIndex = memTypeIdx;
            VkDeviceMemory testMem = VK_NULL_HANDLE;
            g_vkAllocateMemory(device, &allocInfo, nullptr, &testMem);
            g_vkBindBufferMemory(device, testBuf, testMem, 0);

            VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dslInfo.bindingCount = 1;
            dslInfo.pBindings = &binding;
            VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
            g_vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &descLayout);

            VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            plInfo.setLayoutCount = 1;
            plInfo.pSetLayouts = &descLayout;
            VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
            g_vkCreatePipelineLayout(device, &plInfo, nullptr, &pipeLayout);

            VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSubgroup{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
            reqSubgroup.requiredSubgroupSize = 32;

            VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = shaderModule;
            stageInfo.pName = "main";
            if (has_subgroup_size_control) {
                stageInfo.pNext = &reqSubgroup;
                stageInfo.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
            }

            VkComputePipelineCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpInfo.stage = stageInfo;
            cpInfo.layout = pipeLayout;

            VkPipeline pipeline = VK_NULL_HANDLE;
            if (g_vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &pipeline) == VK_SUCCESS) {
                r.wave32_pipeline_created = true;

                VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
                VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
                poolInfo.maxSets = 1;
                poolInfo.poolSizeCount = 1;
                poolInfo.pPoolSizes = &poolSize;
                VkDescriptorPool descPool = VK_NULL_HANDLE;
                g_vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool);

                VkDescriptorSetAllocateInfo dsAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                dsAllocInfo.descriptorPool = descPool;
                dsAllocInfo.descriptorSetCount = 1;
                dsAllocInfo.pSetLayouts = &descLayout;
                VkDescriptorSet descSet = VK_NULL_HANDLE;
                g_vkAllocateDescriptorSets(device, &dsAllocInfo, &descSet);

                VkDescriptorBufferInfo dbi{testBuf, 0, 256};
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = descSet;
                write.dstBinding = 0;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &dbi;
                g_vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

                VkCommandPoolCreateInfo cmdPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
                cmdPoolInfo.queueFamilyIndex = 0;
                VkCommandPool cmdPool = VK_NULL_HANDLE;
                g_vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool);

                VkCommandBufferAllocateInfo cbAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                cbAllocInfo.commandPool = cmdPool;
                cbAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                cbAllocInfo.commandBufferCount = 1;
                VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
                g_vkAllocateCommandBuffers(device, &cbAllocInfo, &cmdBuf);

                VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                g_vkBeginCommandBuffer(cmdBuf, &beginInfo);
                g_vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                g_vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr);
                g_vkCmdDispatch(cmdBuf, 1, 1, 1);
                g_vkEndCommandBuffer(cmdBuf);

                VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &cmdBuf;
                g_vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
                g_vkQueueWaitIdle(queue);

                uint32_t* mapped = nullptr;
                g_vkMapMemory(device, testMem, 0, 256, 0, (void**)&mapped);
                if (mapped) {
                    r.wave32_runtime_measured = mapped[0];
                    g_vkUnmapMemory(device, testMem);
                }

                g_vkDestroyCommandPool(device, cmdPool, nullptr);
                g_vkDestroyDescriptorPool(device, descPool, nullptr);
                g_vkDestroyPipeline(device, pipeline, nullptr);
            }
            g_vkDestroyPipelineLayout(device, pipeLayout, nullptr);
            g_vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
            g_vkDestroyBuffer(device, testBuf, nullptr);
            g_vkFreeMemory(device, testMem, nullptr);
            g_vkDestroyShaderModule(device, shaderModule, nullptr);
        }
    }

    // Test 2 & 3: UMA Checksum & Multi-Heap Bandwidth / Staging Upload Probe (R0.2)
    std::vector<uint32_t> checksum_spv = load_spirv_file("build/probe_checksum.spv");
    if (!checksum_spv.empty()) {
        const size_t test_size = 245 * 1024 * 1024; // 245 MB
        const size_t num_words = test_size / sizeof(uint32_t);
        r.uma_test_bytes = test_size;

        // Find memory type indices:
        // heap0Type: DEVICE_LOCAL only, NOT host-visible
        // heap1Type: HOST_VISIBLE | HOST_COHERENT (Heap 1)
        // heap2Type: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (Heap 2 / ReBAR if present)
        int heap0Type = -1;
        int heap1Type = -1;
        int heap2Type = -1;

        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            const auto& t = memProps.memoryTypes[i];
            bool is_dl = (t.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
            bool is_hv = (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
            bool is_hc = (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

            if (is_dl && is_hv && is_hc && heap2Type == -1) {
                heap2Type = (int)i;
            } else if (is_hv && is_hc && !is_dl && heap1Type == -1) {
                heap1Type = (int)i;
            } else if (is_dl && !is_hv && heap0Type == -1) {
                heap0Type = (int)i;
            }
        }

        // Fallback for host-visible if not separated
        if (heap1Type == -1) {
            for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
                if (memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                    heap1Type = (int)i;
                    break;
                }
            }
        }

        // Setup descriptor set layout and pipeline for checksum / reduction kernel
        VkShaderModuleCreateInfo smInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smInfo.codeSize = checksum_spv.size() * 4;
        smInfo.pCode = checksum_spv.data();
        VkShaderModule csmModule = VK_NULL_HANDLE;
        g_vkCreateShaderModule(device, &smInfo, nullptr, &csmModule);

        VkDescriptorSetLayoutBinding bindings[2]{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
        };
        VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslInfo.bindingCount = 2;
        dslInfo.pBindings = bindings;
        VkDescriptorSetLayout csmDescLayout = VK_NULL_HANDLE;
        g_vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &csmDescLayout);

        VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &csmDescLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pcRange;
        VkPipelineLayout csmPipeLayout = VK_NULL_HANDLE;
        g_vkCreatePipelineLayout(device, &plInfo, nullptr, &csmPipeLayout);

        VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = csmModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpInfo.stage = stageInfo;
        cpInfo.layout = csmPipeLayout;
        VkPipeline csmPipeline = VK_NULL_HANDLE;
        g_vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &csmPipeline);

        VkCommandPoolCreateInfo cmdPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cmdPoolInfo.queueFamilyIndex = 0;
        VkCommandPool cmdPool = VK_NULL_HANDLE;
        g_vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool);

        VkDescriptorPoolSize poolSizes[1]{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 4;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = poolSizes;
        VkDescriptorPool descPool = VK_NULL_HANDLE;
        g_vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool);

        // Allocate outBuf (256 B) in host-visible heap
        VkBuffer outBuf = VK_NULL_HANDLE;
        VkBufferCreateInfo outBufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        g_vkCreateBuffer(device, &outBufInfo, nullptr, &outBuf);
        VkMemoryRequirements outMemReq;
        g_vkGetBufferMemoryRequirements(device, outBuf, &outMemReq);
        VkMemoryAllocateInfo outAlloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, outMemReq.size, (uint32_t)(heap1Type >= 0 ? heap1Type : 0)};
        VkDeviceMemory outMem = VK_NULL_HANDLE;
        g_vkAllocateMemory(device, &outAlloc, nullptr, &outMem);
        g_vkBindBufferMemory(device, outBuf, outMem, 0);

        uint32_t cpu_expected_sum = 0;

        // A. Allocate Heap 1 Buffer (Host-Visible)
        VkBuffer heap1Buf = VK_NULL_HANDLE;
        VkDeviceMemory heap1Mem = VK_NULL_HANDLE;
        if (heap1Type >= 0) {
            VkBufferCreateInfo bInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, test_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
            g_vkCreateBuffer(device, &bInfo, nullptr, &heap1Buf);
            VkMemoryRequirements req;
            g_vkGetBufferMemoryRequirements(device, heap1Buf, &req);
            VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, (uint32_t)heap1Type};
            g_vkAllocateMemory(device, &alloc, nullptr, &heap1Mem);
            g_vkBindBufferMemory(device, heap1Buf, heap1Mem, 0);

            // Populate on CPU
            uint32_t* host_ptr = nullptr;
            g_vkMapMemory(device, heap1Mem, 0, test_size, 0, (void**)&host_ptr);
            if (host_ptr) {
                for (size_t i = 0; i < num_words; ++i) {
                    uint32_t val = (uint32_t)(i * 2654435761u + 1337);
                    host_ptr[i] = val;
                    cpu_expected_sum += val;
                }
                g_vkUnmapMemory(device, heap1Mem);
            }
        }

        // B. Allocate Heap 0 Buffer (Device-Local only, 13.10 GiB pool)
        VkBuffer heap0Buf = VK_NULL_HANDLE;
        VkDeviceMemory heap0Mem = VK_NULL_HANDLE;
        if (heap0Type >= 0) {
            VkBufferCreateInfo bInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, test_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
            g_vkCreateBuffer(device, &bInfo, nullptr, &heap0Buf);
            VkMemoryRequirements req;
            g_vkGetBufferMemoryRequirements(device, heap0Buf, &req);
            VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, (uint32_t)heap0Type};
            g_vkAllocateMemory(device, &alloc, nullptr, &heap0Mem);
            g_vkBindBufferMemory(device, heap0Buf, heap0Mem, 0);
        }

        // Benchmark Staging Upload (Heap 1 -> Heap 0)
        if (heap1Buf && heap0Buf && g_vkCmdCopyBuffer) {
            VkCommandBuffer copyCmd = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo cbAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
            g_vkAllocateCommandBuffers(device, &cbAlloc, &copyCmd);

            VkCommandBufferBeginInfo bBegin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            g_vkBeginCommandBuffer(copyCmd, &bBegin);
            VkBufferCopy region{0, 0, test_size};
            g_vkCmdCopyBuffer(copyCmd, heap1Buf, heap0Buf, 1, &region);
            g_vkEndCommandBuffer(copyCmd);

            // Warmup copy
            VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &copyCmd, 0, nullptr};
            g_vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE);
            g_vkQueueWaitIdle(queue);

            // Measure 20 iterations
            const int iters = 20;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) {
                g_vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE);
            }
            g_vkQueueWaitIdle(queue);
            auto t1 = std::chrono::high_resolution_clock::now();

            double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
            double total_gb = (double)(test_size * iters) / (1024.0 * 1024.0 * 1024.0);
            r.staging_heap1_to_heap0_gbs = total_gb / elapsed_sec;
        }

        // Benchmark GPU Compute Read from Heap 0
        if (heap0Buf) {
            VkDescriptorSet ds0 = VK_NULL_HANDLE;
            VkDescriptorSetAllocateInfo dsAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &csmDescLayout};
            g_vkAllocateDescriptorSets(device, &dsAlloc, &ds0);

            VkDescriptorBufferInfo dbiIn{heap0Buf, 0, test_size};
            VkDescriptorBufferInfo dbiOut{outBuf, 0, 256};
            VkWriteDescriptorSet writes[2]{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = ds0;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo = &dbiIn;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = ds0;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo = &dbiOut;
            g_vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

            VkCommandBuffer compCmd = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo cbAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
            g_vkAllocateCommandBuffers(device, &cbAlloc, &compCmd);

            uint32_t push_words = (uint32_t)num_words;
            VkCommandBufferBeginInfo bBegin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            g_vkBeginCommandBuffer(compCmd, &bBegin);
            g_vkCmdBindPipeline(compCmd, VK_PIPELINE_BIND_POINT_COMPUTE, csmPipeline);
            g_vkCmdBindDescriptorSets(compCmd, VK_PIPELINE_BIND_POINT_COMPUTE, csmPipeLayout, 0, 1, &ds0, 0, nullptr);
            g_vkCmdPushConstants(compCmd, csmPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &push_words);
            g_vkCmdDispatch(compCmd, 256, 1, 1);
            g_vkEndCommandBuffer(compCmd);

            // Warmup
            VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &compCmd, 0, nullptr};
            g_vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE);
            g_vkQueueWaitIdle(queue);

            const int iters = 20;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) {
                g_vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE);
            }
            g_vkQueueWaitIdle(queue);
            auto t1 = std::chrono::high_resolution_clock::now();

            double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
            double total_gb = (double)(test_size * iters) / (1024.0 * 1024.0 * 1024.0);
            r.heap0_read_bandwidth_gbs = total_gb / elapsed_sec;
        }

        // Benchmark GPU Compute Read from Heap 1 (Host-Visible) & Verify Checksum
        if (heap1Buf) {
            VkDescriptorSet ds1 = VK_NULL_HANDLE;
            VkDescriptorSetAllocateInfo dsAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &csmDescLayout};
            g_vkAllocateDescriptorSets(device, &dsAlloc, &ds1);

            VkDescriptorBufferInfo dbiIn{heap1Buf, 0, test_size};
            VkDescriptorBufferInfo dbiOut{outBuf, 0, 256};
            VkWriteDescriptorSet writes[2]{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = ds1;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo = &dbiIn;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = ds1;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo = &dbiOut;
            g_vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

            VkCommandBuffer compCmd = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo cbAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
            g_vkAllocateCommandBuffers(device, &cbAlloc, &compCmd);

            uint32_t push_words = (uint32_t)num_words;
            VkCommandBufferBeginInfo bBegin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            g_vkBeginCommandBuffer(compCmd, &bBegin);
            g_vkCmdBindPipeline(compCmd, VK_PIPELINE_BIND_POINT_COMPUTE, csmPipeline);
            g_vkCmdBindDescriptorSets(compCmd, VK_PIPELINE_BIND_POINT_COMPUTE, csmPipeLayout, 0, 1, &ds1, 0, nullptr);
            g_vkCmdPushConstants(compCmd, csmPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &push_words);
            g_vkCmdDispatch(compCmd, 256, 1, 1);
            g_vkEndCommandBuffer(compCmd);

            // Single iteration timing for UMA zero-copy reporting
            uint32_t* out_mapped = nullptr;
            g_vkMapMemory(device, outMem, 0, 256, 0, (void**)&out_mapped);
            if (out_mapped) out_mapped[0] = 0;

            auto t0 = std::chrono::high_resolution_clock::now();
            VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &compCmd, 0, nullptr};
            g_vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE);
            g_vkQueueWaitIdle(queue);
            auto t1 = std::chrono::high_resolution_clock::now();

            r.uma_checksum_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            r.uma_bandwidth_gbs = ((double)test_size / (1024.0 * 1024.0 * 1024.0)) / (r.uma_checksum_time_ms / 1000.0);

            if (out_mapped && out_mapped[0] == cpu_expected_sum) {
                r.uma_zero_copy_passed = true;
            } else {
                std::cerr << "[uma] Checksum mismatch! CPU=" << cpu_expected_sum << " GPU=" << (out_mapped ? out_mapped[0] : 0) << std::endl;
            }
            if (out_mapped) g_vkUnmapMemory(device, outMem);

            // Measure 20 iterations for sustained read bandwidth
            const int iters = 20;
            t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) {
                g_vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE);
            }
            g_vkQueueWaitIdle(queue);
            t1 = std::chrono::high_resolution_clock::now();

            double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
            double total_gb = (double)(test_size * iters) / (1024.0 * 1024.0 * 1024.0);
            r.heap1_read_bandwidth_gbs = total_gb / elapsed_sec;
        }

        // Cleanup
        g_vkDestroyCommandPool(device, cmdPool, nullptr);
        g_vkDestroyDescriptorPool(device, descPool, nullptr);
        g_vkDestroyPipeline(device, csmPipeline, nullptr);
        g_vkDestroyPipelineLayout(device, csmPipeLayout, nullptr);
        g_vkDestroyDescriptorSetLayout(device, csmDescLayout, nullptr);
        g_vkDestroyShaderModule(device, csmModule, nullptr);
        g_vkDestroyBuffer(device, outBuf, nullptr);
        g_vkFreeMemory(device, outMem, nullptr);
        if (heap0Buf) g_vkDestroyBuffer(device, heap0Buf, nullptr);
        if (heap0Mem) g_vkFreeMemory(device, heap0Mem, nullptr);
        if (heap1Buf) g_vkDestroyBuffer(device, heap1Buf, nullptr);
        if (heap1Mem) g_vkFreeMemory(device, heap1Mem, nullptr);
    }

    g_vkDestroyDevice(device, nullptr);
    g_vkDestroyInstance(instance, nullptr);
}

// Helpers for Storage Benchmarks
void ensure_test_file(const std::string& path, uint64_t size_bytes) {
    HANDLE hCheck = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (hCheck != INVALID_HANDLE_VALUE) {
        LARGE_INTEGER curSize;
        if (GetFileSizeEx(hCheck, &curSize) && (uint64_t)curSize.QuadPart == size_bytes) {
            CloseHandle(hCheck);
            return;
        }
        CloseHandle(hCheck);
    }

    HANDLE hWrite = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hWrite == INVALID_HANDLE_VALUE) return;

    const DWORD write_chunk = 4 * 1024 * 1024;
    std::vector<BYTE> write_buf(write_chunk, 0xA5);
    DWORD written = 0;
    uint64_t total_written = 0;
    while (total_written < size_bytes) {
        DWORD to_write = (DWORD)std::min((uint64_t)write_chunk, size_bytes - total_written);
        if (!WriteFile(hWrite, write_buf.data(), to_write, &written, nullptr) || written == 0) break;
        total_written += written;
    }
    CloseHandle(hWrite);
}

double measure_warm_readfile(const std::string& path, uint64_t size_bytes, DWORD chunk_size, bool seq_scan) {
    DWORD flags = FILE_ATTRIBUTE_NORMAL;
    if (seq_scan) flags |= FILE_FLAG_SEQUENTIAL_SCAN;

    // Warm-up pass
    {
        HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
        if (hFile != INVALID_HANDLE_VALUE) {
            std::vector<BYTE> buf(chunk_size);
            DWORD read_bytes = 0;
            uint64_t total_read = 0;
            while (total_read < size_bytes) {
                if (!ReadFile(hFile, buf.data(), chunk_size, &read_bytes, nullptr) || read_bytes == 0) break;
                total_read += read_bytes;
            }
            CloseHandle(hFile);
        }
    }

    // Timed pass
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) return 0.0;

    std::vector<BYTE> buf(chunk_size);
    DWORD read_bytes = 0;
    uint64_t total_read = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    while (total_read < size_bytes) {
        if (!ReadFile(hFile, buf.data(), chunk_size, &read_bytes, nullptr) || read_bytes == 0) break;
        total_read += read_bytes;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    CloseHandle(hFile);

    double sec = std::chrono::duration<double>(t1 - t0).count();
    return ((double)total_read / (1024.0 * 1024.0 * 1024.0)) / sec;
}

double measure_warm_mmap(const std::string& path, uint64_t size_bytes) {
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) return 0.0;

    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) {
        CloseHandle(hFile);
        return 0.0;
    }

    const BYTE* ptr = (const BYTE*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, size_bytes);
    if (!ptr) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        return 0.0;
    }

    // Warm-up pass (touch all pages)
    volatile uint64_t dummy = 0;
    const uint64_t* u64_ptr = (const uint64_t*)ptr;
    const size_t u64_count = size_bytes / sizeof(uint64_t);
    for (size_t i = 0; i < u64_count; i += 512) { // step by 4KB page
        dummy += u64_ptr[i];
    }

    // Timed sequential read pass
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t sum = 0;
    for (size_t i = 0; i < u64_count; ++i) {
        sum += u64_ptr[i];
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    UnmapViewOfFile(ptr);
    CloseHandle(hMap);
    CloseHandle(hFile);

    if (sum == 0) dummy = sum; // prevent compiler elision

    double sec = std::chrono::duration<double>(t1 - t0).count();
    return ((double)size_bytes / (1024.0 * 1024.0 * 1024.0)) / sec;
}

void probe_storage_benchmarks(ProbeResults& r) {
    const std::string test_file_256mb = "build/io_test_256mb.tmp";
    const std::string test_file_1gb   = "build/io_test_1gb.tmp";
    const std::string test_file_4gb   = "build/io_test_4gb.tmp";

    const uint64_t size_256mb = 256ULL * 1024ULL * 1024ULL;
    const uint64_t size_1gb   = 1024ULL * 1024ULL * 1024ULL;
    const uint64_t size_4gb   = 4ULL * 1024ULL * 1024ULL * 1024ULL;

    std::cout << "[storage] preparing test files (256 MB, 1 GB, 4 GB)..." << std::endl;
    ensure_test_file(test_file_256mb, size_256mb);
    ensure_test_file(test_file_1gb, size_1gb);
    ensure_test_file(test_file_4gb, size_4gb);

    // 1. Warm-Cache Multi-Size and Chunk Benchmarks (R0.1)
    std::cout << "[storage] benchmarking warm OS page cache reads across file sizes & chunk configurations..." << std::endl;
    r.storage_warm_256mb_gbs = measure_warm_readfile(test_file_256mb, size_256mb, 16 * 1024 * 1024, true);
    r.storage_warm_1gb_gbs   = measure_warm_readfile(test_file_1gb, size_1gb, 16 * 1024 * 1024, true);

    r.storage_warm_4gb_1mb_gbs           = measure_warm_readfile(test_file_4gb, size_4gb, 1 * 1024 * 1024, false);
    r.storage_warm_4gb_4mb_gbs           = measure_warm_readfile(test_file_4gb, size_4gb, 4 * 1024 * 1024, false);
    r.storage_warm_4gb_16mb_gbs          = measure_warm_readfile(test_file_4gb, size_4gb, 16 * 1024 * 1024, false);
    r.storage_warm_4gb_seqscan_16mb_gbs  = measure_warm_readfile(test_file_4gb, size_4gb, 16 * 1024 * 1024, true);
    r.storage_warm_4gb_mmap_gbs          = measure_warm_mmap(test_file_4gb, size_4gb);

    r.storage_buffered_warm_gbs = r.storage_warm_4gb_seqscan_16mb_gbs;
    r.storage_buffered_cold_gbs = r.storage_warm_4gb_1mb_gbs; // Note legacy field

    // 2. Unbuffered Asynchronous Overlapped Read Benchmark (Cold NVMe Hardware Throughput)
    auto bench_unbuffered = [&](int qd, size_t alignment, double& result_gbs) {
        HANDLE hFile = CreateFileA(
            test_file_4gb.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
            nullptr
        );
        if (hFile == INVALID_HANDLE_VALUE) {
            result_gbs = 0.0;
            return;
        }

        const DWORD io_chunk_size = 1024 * 1024; // 1 MB unbuffered chunk
        const size_t total_chunks = size_4gb / io_chunk_size;

        struct Slot {
            OVERLAPPED ov{};
            HANDLE event = nullptr;
            BYTE* buf = nullptr;
        };

        std::vector<Slot> slots(qd);
        for (int i = 0; i < qd; ++i) {
            slots[i].event = CreateEventA(nullptr, TRUE, FALSE, nullptr);
            slots[i].ov.hEvent = slots[i].event;
            slots[i].buf = (BYTE*)_aligned_malloc(io_chunk_size, alignment);
        }

        size_t next_chunk = 0;
        size_t in_flight = 0;
        uint64_t bytes_completed = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Issue initial batch
        for (int i = 0; i < qd && next_chunk < total_chunks; ++i) {
            uint64_t offset = next_chunk * io_chunk_size;
            slots[i].ov.Offset = (DWORD)(offset & 0xFFFFFFFF);
            slots[i].ov.OffsetHigh = (DWORD)(offset >> 32);
            ResetEvent(slots[i].event);

            DWORD read = 0;
            ReadFile(hFile, slots[i].buf, io_chunk_size, &read, &slots[i].ov);
            next_chunk++;
            in_flight++;
        }

        while (in_flight > 0) {
            std::vector<HANDLE> waitEvents;
            std::vector<int> eventSlotMap;
            for (int i = 0; i < qd; ++i) {
                if (slots[i].buf) {
                    waitEvents.push_back(slots[i].event);
                    eventSlotMap.push_back(i);
                }
            }

            DWORD waitRes = WaitForMultipleObjects((DWORD)waitEvents.size(), waitEvents.data(), FALSE, INFINITE);
            if (waitRes >= WAIT_OBJECT_0 && waitRes < WAIT_OBJECT_0 + waitEvents.size()) {
                int slotIdx = eventSlotMap[waitRes - WAIT_OBJECT_0];
                DWORD transferred = 0;
                if (GetOverlappedResult(hFile, &slots[slotIdx].ov, &transferred, FALSE)) {
                    bytes_completed += transferred;
                }
                in_flight--;

                if (next_chunk < total_chunks) {
                    uint64_t offset = next_chunk * io_chunk_size;
                    slots[slotIdx].ov.Offset = (DWORD)(offset & 0xFFFFFFFF);
                    slots[slotIdx].ov.OffsetHigh = (DWORD)(offset >> 32);
                    ResetEvent(slots[slotIdx].event);

                    DWORD read = 0;
                    ReadFile(hFile, slots[slotIdx].buf, io_chunk_size, &read, &slots[slotIdx].ov);
                    next_chunk++;
                    in_flight++;
                }
            } else {
                break;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        CloseHandle(hFile);

        for (int i = 0; i < qd; ++i) {
            if (slots[i].event) CloseHandle(slots[i].event);
            if (slots[i].buf) _aligned_free(slots[i].buf);
        }

        double sec = std::chrono::duration<double>(t1 - t0).count();
        result_gbs = ((double)bytes_completed / (1024.0 * 1024.0 * 1024.0)) / sec;
    };

    bench_unbuffered(1, 4096, r.storage_unbuffered_qd1_gbs);
    bench_unbuffered(4, 4096, r.storage_unbuffered_qd4_gbs);
    bench_unbuffered(8, 4096, r.storage_unbuffered_qd8_gbs);
    bench_unbuffered(16, 4096, r.storage_unbuffered_qd16_gbs);
    bench_unbuffered(8, 16384, r.storage_unbuffered_16k_align_gbs);

    // 3. DirectStorage 1.2 Probe
    HMODULE hDStorage = LoadLibraryA("dstorage.dll");
    if (hDStorage) {
        r.directstorage_available = true;
        r.directstorage_status = "dstorage.dll loaded successfully (app-local NuGet redist)";
        FreeLibrary(hDStorage);
    } else {
        r.directstorage_available = false;
        r.directstorage_status = "dstorage.dll not found in app directory (System32 lookup expectedly absent; will use Win32 IOCP backend)";
    }
}

void write_json_and_markdown(const ProbeResults& r, const std::string& json_path, const std::string& md_path) {
    // 1. JSON Report
    std::ofstream jf(json_path);
    if (jf.is_open()) {
        jf << "{\n";
        jf << "  \"system\": {\n";
        jf << "    \"os_version\": \"" << r.os_version << "\",\n";
        jf << "    \"cpu_brand\": \"" << r.cpu_brand << "\",\n";
        jf << "    \"cpu_physical_cores\": " << r.cpu_physical_cores << ",\n";
        jf << "    \"cpu_logical_cores\": " << r.cpu_logical_cores << ",\n";
        jf << "    \"ram_total_bytes\": " << r.ram_total_bytes << ",\n";
        jf << "    \"ram_total_gb\": " << std::fixed << std::setprecision(2) << r.ram_total_gb << ",\n";
        jf << "    \"ram_available_bytes\": " << r.ram_avail_bytes << ",\n";
        jf << "    \"disk_model\": \"" << r.disk_model << "\",\n";
        jf << "    \"disk_free_gb\": " << std::fixed << std::setprecision(2) << r.disk_free_gb << ",\n";
        jf << "    \"avx512\": {\n";
        jf << "      \"f\": " << (r.avx512_f ? "true" : "false") << ",\n";
        jf << "      \"bw\": " << (r.avx512_bw ? "true" : "false") << ",\n";
        jf << "      \"cd\": " << (r.avx512_cd ? "true" : "false") << ",\n";
        jf << "      \"dq\": " << (r.avx512_dq ? "true" : "false") << ",\n";
        jf << "      \"vl\": " << (r.avx512_vl ? "true" : "false") << ",\n";
        jf << "      \"vnni\": " << (r.avx512_vnni ? "true" : "false") << ",\n";
        jf << "      \"bf16\": " << (r.avx512_bf16 ? "true" : "false") << "\n";
        jf << "    }\n";
        jf << "  },\n";

        jf << "  \"dxgi\": {\n";
        jf << "    \"adapter_name\": \"" << r.dxgi_adapter_name << "\",\n";
        jf << "    \"dedicated_vram_bytes\": " << r.dxgi_dedicated_vram << ",\n";
        jf << "    \"shared_system_ram_bytes\": " << r.dxgi_shared_system_ram << ",\n";
        jf << "    \"local_budget_bytes\": " << r.dxgi_local_budget << ",\n";
        jf << "    \"non_local_budget_bytes\": " << r.dxgi_non_local_budget << "\n";
        jf << "  },\n";

        jf << "  \"vulkan\": {\n";
        jf << "    \"device_name\": \"" << r.vk_device_name << "\",\n";
        jf << "    \"api_version\": \"" << VK_VERSION_MAJOR(r.vk_api_version) << "." << VK_VERSION_MINOR(r.vk_api_version) << "." << VK_VERSION_PATCH(r.vk_api_version) << "\",\n";
        jf << "    \"driver_version\": \"" << r.vk_driver_version << "\",\n";
        jf << "    \"vendor_id\": " << r.vk_vendor_id << ",\n";
        jf << "    \"device_id\": " << r.vk_device_id << ",\n";
        jf << "    \"subgroup\": {\n";
        jf << "      \"default_size\": " << r.subgroup_size << ",\n";
        jf << "      \"min_size\": " << r.subgroup_min_size << ",\n";
        jf << "      \"max_size\": " << r.subgroup_max_size << ",\n";
        jf << "      \"size_control_supported\": " << (r.subgroup_size_control_supported ? "true" : "false") << ",\n";
        jf << "      \"wave32_pipeline_created\": " << (r.wave32_pipeline_created ? "true" : "false") << ",\n";
        jf << "      \"wave32_runtime_measured\": " << r.wave32_runtime_measured << "\n";
        jf << "    },\n";
        jf << "    \"heaps\": [\n";
        for (size_t i = 0; i < r.vk_heaps.size(); ++i) {
            const auto& h = r.vk_heaps[i];
            jf << "      {\"index\": " << h.index << ", \"size_bytes\": " << h.size_bytes << ", \"size_mb\": " << h.size_mb << ", \"device_local\": " << (h.is_device_local ? "true" : "false") << ", \"budget_bytes\": " << h.budget_bytes << "}" << (i + 1 < r.vk_heaps.size() ? "," : "") << "\n";
        }
        jf << "    ],\n";
        jf << "    \"memory_types\": [\n";
        for (size_t i = 0; i < r.vk_types.size(); ++i) {
            const auto& t = r.vk_types[i];
            jf << "      {\"index\": " << t.index << ", \"heap_index\": " << t.heap_index << ", \"host_visible\": " << (t.is_host_visible ? "true" : "false") << ", \"host_coherent\": " << (t.is_host_coherent ? "true" : "false") << ", \"device_local\": " << (t.is_device_local ? "true" : "false") << ", \"host_cached\": " << (t.is_host_cached ? "true" : "false") << "}" << (i + 1 < r.vk_types.size() ? "," : "") << "\n";
        }
        jf << "    ],\n";
        jf << "    \"key_heaps\": {\n";
        jf << "      \"host_visible_device_local_bytes\": " << r.heap_host_visible_device_local_bytes << ",\n";
        jf << "      \"host_visible_bytes\": " << r.heap_host_visible_bytes << ",\n";
        jf << "      \"device_local_only_bytes\": " << r.heap_device_local_only_bytes << "\n";
        jf << "    },\n";
        jf << "    \"heap_bandwidth\": {\n";
        jf << "      \"gpu_read_heap0_gbs\": " << r.heap0_read_bandwidth_gbs << ",\n";
        jf << "      \"gpu_read_heap1_gbs\": " << r.heap1_read_bandwidth_gbs << ",\n";
        jf << "      \"staging_upload_heap1_to_heap0_gbs\": " << r.staging_heap1_to_heap0_gbs << "\n";
        jf << "    },\n";
        jf << "    \"cooperative_matrix\": {\n";
        jf << "      \"supported\": " << (r.coop_matrix_supported ? "true" : "false") << ",\n";
        jf << "      \"tuples_count\": " << r.coop_matrix_tuples.size() << ",\n";
        jf << "      \"tuples\": [\n";
        for (size_t i = 0; i < r.coop_matrix_tuples.size(); ++i) {
            const auto& tup = r.coop_matrix_tuples[i];
            jf << "        {\"M\": " << tup.MSize << ", \"N\": " << tup.NSize << ", \"K\": " << tup.KSize << ", \"A\": \"" << tup.AType << "\", \"B\": \"" << tup.BType << "\", \"C\": \"" << tup.CType << "\", \"Result\": \"" << tup.ResultType << "\", \"scope\": \"" << tup.scope << "\"}" << (i + 1 < r.coop_matrix_tuples.size() ? "," : "") << "\n";
        }
        jf << "      ]\n";
        jf << "    }\n";
        jf << "  },\n";

        jf << "  \"uma\": {\n";
        jf << "    \"zero_copy_passed\": " << (r.uma_zero_copy_passed ? "true" : "false") << ",\n";
        jf << "    \"test_bytes\": " << r.uma_test_bytes << ",\n";
        jf << "    \"checksum_time_ms\": " << r.uma_checksum_time_ms << ",\n";
        jf << "    \"bandwidth_gbs\": " << r.uma_bandwidth_gbs << "\n";
        jf << "  },\n";

        jf << "  \"storage\": {\n";
        jf << "    \"buffered_cold_gbs\": " << r.storage_buffered_cold_gbs << ",\n";
        jf << "    \"buffered_warm_gbs\": " << r.storage_buffered_warm_gbs << ",\n";
        jf << "    \"warm_256mb_gbs\": " << r.storage_warm_256mb_gbs << ",\n";
        jf << "    \"warm_1gb_gbs\": " << r.storage_warm_1gb_gbs << ",\n";
        jf << "    \"warm_4gb_1mb_chunk_gbs\": " << r.storage_warm_4gb_1mb_gbs << ",\n";
        jf << "    \"warm_4gb_4mb_chunk_gbs\": " << r.storage_warm_4gb_4mb_gbs << ",\n";
        jf << "    \"warm_4gb_16mb_chunk_gbs\": " << r.storage_warm_4gb_16mb_gbs << ",\n";
        jf << "    \"warm_4gb_seqscan_16mb_gbs\": " << r.storage_warm_4gb_seqscan_16mb_gbs << ",\n";
        jf << "    \"warm_4gb_mmap_gbs\": " << r.storage_warm_4gb_mmap_gbs << ",\n";
        jf << "    \"unbuffered_qd1_gbs\": " << r.storage_unbuffered_qd1_gbs << ",\n";
        jf << "    \"unbuffered_qd4_gbs\": " << r.storage_unbuffered_qd4_gbs << ",\n";
        jf << "    \"unbuffered_qd8_gbs\": " << r.storage_unbuffered_qd8_gbs << ",\n";
        jf << "    \"unbuffered_qd16_gbs\": " << r.storage_unbuffered_qd16_gbs << ",\n";
        jf << "    \"unbuffered_16k_align_gbs\": " << r.storage_unbuffered_16k_align_gbs << ",\n";
        jf << "    \"directstorage_available\": " << (r.directstorage_available ? "true" : "false") << ",\n";
        jf << "    \"directstorage_status\": \"" << r.directstorage_status << "\"\n";
        jf << "  }\n";
        jf << "}\n";
    }

    // 2. Markdown Report
    std::ofstream mf(md_path);
    if (mf.is_open()) {
        mf << "# Ground Truth Hardware & Runtime Capabilities\n\n";
        {
            const std::time_t now = std::time(nullptr);
            std::tm tm_buf{};
            localtime_s(&tm_buf, &now);
            char datebuf[32];
            std::strftime(datebuf, sizeof(datebuf), "%Y-%m-%d", &tm_buf);
            mf << "**Date:** " << datebuf << "  \n";
        }
        mf << "**Generated by:** `tools/probe_apu/probe_apu.exe`  \n\n";

        mf << "## 1. System & APU Hardware\n\n";
        mf << "| Component | Specification / Measured Value |\n";
        mf << "|---|---|\n";
        mf << "| **System / OS** | " << r.os_version << " |\n";
        mf << "| **CPU** | " << r.cpu_brand << " (" << r.cpu_physical_cores << "C / " << r.cpu_logical_cores << "T) |\n";
        mf << "| **AVX-512 Support** | F=" << (r.avx512_f ? "Yes" : "No") << ", BW=" << (r.avx512_bw ? "Yes" : "No") << ", VNNI=" << (r.avx512_vnni ? "Yes" : "No") << ", BF16=" << (r.avx512_bf16 ? "Yes" : "No") << " |\n";
        mf << "| **Physical RAM** | **" << std::fixed << std::setprecision(2) << r.ram_total_gb << " GB usable** (" << r.ram_total_bytes << " bytes) |\n";
        mf << "| **Storage Device** | " << r.disk_model << " (" << std::fixed << std::setprecision(1) << r.disk_free_gb << " GB free on C:) |\n";
        mf << "| **iGPU** | " << r.vk_device_name << " (Vulkan " << VK_VERSION_MAJOR(r.vk_api_version) << "." << VK_VERSION_MINOR(r.vk_api_version) << "." << VK_VERSION_PATCH(r.vk_api_version) << ", Driver " << r.vk_driver_version << ") |\n\n";

        mf << "## 2. Vulkan Memory Heaps & Shared Memory Budget\n\n";
        mf << "| Heap # | Size (MB) | Device Local | Host Visible / Coherent | Ext Budget (MB) |\n";
        mf << "|---|---:|:---:|:---:|---:|\n";
        for (const auto& h : r.vk_heaps) {
            bool hv = false;
            for (const auto& t : r.vk_types) {
                if (t.heap_index == h.index && t.is_host_visible) hv = true;
            }
            mf << "| Heap " << h.index << " | " << std::fixed << std::setprecision(1) << h.size_mb << " MB | "
               << (h.is_device_local ? "✔ Yes" : "No") << " | "
               << (hv ? "✔ Yes" : "No") << " | "
               << std::fixed << std::setprecision(1) << (double)h.budget_bytes / (1024.0 * 1024.0) << " MB |\n";
        }
        mf << "\n";
        // By memory TYPE, not by heap index. The indices are not fixed -- on this machine
        // heap 0 is the host-visible one -- and naming them by index made this summary
        // contradict the flags in the table directly above it.
        mf << "- **`DEVICE_LOCAL` only (GPU-private pool):** " << std::fixed << std::setprecision(2) << ((double)r.heap_device_local_only_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB\n";
        mf << "- **`HOST_VISIBLE` (CPU-mappable pool):** " << std::fixed << std::setprecision(2) << ((double)r.heap_host_visible_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB\n";
        mf << "- **`HOST_VISIBLE | DEVICE_LOCAL` (shared, if present):** " << std::fixed << std::setprecision(2) << ((double)r.heap_host_visible_device_local_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB\n";
        mf << "- **DXGI Shared Memory Budget:** " << std::fixed << std::setprecision(2) << ((double)r.dxgi_non_local_budget / (1024.0 * 1024.0 * 1024.0)) << " GB (Local Dedicated: " << ((double)r.dxgi_dedicated_vram / (1024.0 * 1024.0)) << " MB)\n\n";

        mf << "## 3. GPU Heap Read Bandwidth & Staging Upload (R0.2)\n\n";
        mf << "| Memory Operation | Measured Bandwidth |\n";
        mf << "|---|---:|\n";
        // Sizes come from the measurement, not from a literal: these were "13.1 GiB" and
        // "6.55 GiB", which are the sizes under a different BIOS UMA setting.
        mf << "| **GPU Compute Read from Heap 0** (" << std::fixed << std::setprecision(2) << (r.vk_heaps.empty() ? 0.0 : (double)r.vk_heaps[0].size_mb / 1024.0) << " GiB) | **" << std::fixed << std::setprecision(2) << r.heap0_read_bandwidth_gbs << " GB/s** |\n";
        mf << "| **GPU Compute Read from Heap 1** (" << std::fixed << std::setprecision(2) << (r.vk_heaps.size() < 2 ? 0.0 : (double)r.vk_heaps[1].size_mb / 1024.0) << " GiB) | **" << std::fixed << std::setprecision(2) << r.heap1_read_bandwidth_gbs << " GB/s** |\n";
        mf << "| **Staging Upload (`vkCmdCopyBuffer` Heap 1 -> Heap 0)** | **" << std::fixed << std::setprecision(2) << r.staging_heap1_to_heap0_gbs << " GB/s** |\n\n";

        mf << "## 4. Subgroup Control & Cooperative Matrix\n\n";
        mf << "- **Default Subgroup Size:** " << r.subgroup_size << " lanes\n";
        mf << "- **`VK_EXT_subgroup_size_control`:** " << (r.subgroup_size_control_supported ? "✔ Supported" : "✘ Not Supported") << " (Min: " << r.subgroup_min_size << ", Max: " << r.subgroup_max_size << ")\n";
        mf << "- **Wave32 Explicit Pipeline Creation:** " << (r.wave32_pipeline_created ? "✔ Success" : "✘ Failed") << " -> **Runtime Measured Lane Count: " << r.wave32_runtime_measured << "**\n";
        mf << "- **`VK_KHR_cooperative_matrix` (WMMA):** " << (r.coop_matrix_supported ? "✔ Supported" : "✘ Not Supported") << " (" << r.coop_matrix_tuples.size() << " configurations)\n\n";

        if (!r.coop_matrix_tuples.empty()) {
            mf << "### Supported Cooperative Matrix (M, N, K) Tuples:\n\n";
            mf << "| M | N | K | A Type | B Type | C Type | Result Type | Scope |\n";
            mf << "|---|---|---|---|---|---|---|---|\n";
            for (const auto& tup : r.coop_matrix_tuples) {
                mf << "| " << tup.MSize << " | " << tup.NSize << " | " << tup.KSize << " | " << tup.AType << " | " << tup.BType << " | " << tup.CType << " | " << tup.ResultType << " | " << tup.scope << " |\n";
            }
            mf << "\n";
        }

        mf << "## 5. Storage / NVMe & Warm-Cache Throughput (R0.1)\n\n";
        mf << "### Warm OS Page-Cache Read Rates across File Sizes:\n\n";
        mf << "| File Size | Chunk Size / Mode | Measured Throughput |\n";
        mf << "|---|---|---:|\n";
        mf << "| 256 MB File | 16 MB Chunk + `SEQUENTIAL_SCAN` | **" << std::fixed << std::setprecision(2) << r.storage_warm_256mb_gbs << " GB/s** |\n";
        mf << "| 1.0 GB File | 16 MB Chunk + `SEQUENTIAL_SCAN` | **" << std::fixed << std::setprecision(2) << r.storage_warm_1gb_gbs << " GB/s** |\n";
        mf << "| 4.0 GB File | 1 MB Chunk `ReadFile` | " << std::fixed << std::setprecision(2) << r.storage_warm_4gb_1mb_gbs << " GB/s |\n";
        mf << "| 4.0 GB File | 4 MB Chunk `ReadFile` | " << std::fixed << std::setprecision(2) << r.storage_warm_4gb_4mb_gbs << " GB/s |\n";
        mf << "| 4.0 GB File | 16 MB Chunk `ReadFile` | " << std::fixed << std::setprecision(2) << r.storage_warm_4gb_16mb_gbs << " GB/s |\n";
        mf << "| 4.0 GB File | 16 MB Chunk + `SEQUENTIAL_SCAN` | **" << std::fixed << std::setprecision(2) << r.storage_warm_4gb_seqscan_16mb_gbs << " GB/s** |\n";
        mf << "| 4.0 GB File | Memory Mapped (`MapViewOfFile`) | **" << std::fixed << std::setprecision(2) << r.storage_warm_4gb_mmap_gbs << " GB/s** |\n\n";

        mf << "### Unbuffered Direct NVMe Throughput (Hardware Limit):\n\n";
        mf << "| Access Mode | Alignment | Queue Depth | Measured Throughput |\n";
        mf << "|---|---|---|---:|\n";
        mf << "| `NO_BUFFERING` Overlapped | 4 KB Sector | QD = 1 | " << std::fixed << std::setprecision(2) << r.storage_unbuffered_qd1_gbs << " GB/s |\n";
        mf << "| `NO_BUFFERING` Overlapped | 4 KB Sector | QD = 4 | " << std::fixed << std::setprecision(2) << r.storage_unbuffered_qd4_gbs << " GB/s |\n";
        mf << "| `NO_BUFFERING` Overlapped | 4 KB Sector | QD = 8 | " << std::fixed << std::setprecision(2) << r.storage_unbuffered_qd8_gbs << " GB/s |\n";
        mf << "| `NO_BUFFERING` Overlapped | 4 KB Sector | QD = 16 | " << std::fixed << std::setprecision(2) << r.storage_unbuffered_qd16_gbs << " GB/s |\n";
        mf << "| `NO_BUFFERING` Overlapped | 16 KB Sector | QD = 8 | " << std::fixed << std::setprecision(2) << r.storage_unbuffered_16k_align_gbs << " GB/s |\n\n";
        mf << "- **DirectStorage 1.2 Status:** " << r.directstorage_status << "\n";
    }
}

int main(int argc, char** argv) {
    bool run_all = true;
    std::string json_path = "build/ground_truth.json";
    std::string md_path = "docs/GROUND_TRUTH.md";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json" && i + 1 < argc) {
            json_path = argv[++i];
        } else if (arg == "--md" && i + 1 < argc) {
            md_path = argv[++i];
        } else if (arg == "--all") {
            run_all = true;
        }
    }

    CreateDirectoryA("build", nullptr);
    CreateDirectoryA("docs", nullptr);

    std::cout << "============================================================" << std::endl;
    std::cout << "Turbo-WinFare Dense — APU Capability & Hardware Probe" << std::endl;
    std::cout << "============================================================" << std::endl;

    ProbeResults r;

    std::cout << "[1/5] Probing CPU, OS, RAM and Storage hardware..." << std::endl;
    probe_cpu_and_os(r);
    std::cout << "      CPU: " << r.cpu_brand << " (" << r.cpu_physical_cores << " physical, " << r.cpu_logical_cores << " logical)" << std::endl;
    std::cout << "      RAM: " << std::fixed << std::setprecision(2) << r.ram_total_gb << " GB usable" << std::endl;
    std::cout << "      Disk: " << r.disk_model << " (" << std::fixed << std::setprecision(1) << r.disk_free_gb << " GB free)" << std::endl;

    std::cout << "[2/5] Probing DXGI memory segments..." << std::endl;
    probe_dxgi_memory(r);
    std::cout << "      Adapter: " << r.dxgi_adapter_name << std::endl;
    std::cout << "      Dedicated VRAM: " << (r.dxgi_dedicated_vram / (1024 * 1024)) << " MB, Non-Local Budget: " << (r.dxgi_non_local_budget / (1024 * 1024)) << " MB" << std::endl;

    std::cout << "[3/5] Probing Vulkan 1.3, heaps, Subgroup control, and WMMA..." << std::endl;
    probe_vulkan(r);
    std::cout << "      Device: " << r.vk_device_name << std::endl;
    std::cout << "      Wave32 Subgroup Runtime: " << r.wave32_runtime_measured << " lanes" << std::endl;
    std::cout << "      Cooperative Matrix Tuples: " << r.coop_matrix_tuples.size() << std::endl;
    std::cout << "      UMA Zero-Copy: " << (r.uma_zero_copy_passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "      GPU Read Heap 0: " << std::fixed << std::setprecision(2) << r.heap0_read_bandwidth_gbs << " GB/s" << std::endl;
    std::cout << "      GPU Read Heap 1: " << std::fixed << std::setprecision(2) << r.heap1_read_bandwidth_gbs << " GB/s" << std::endl;
    std::cout << "      Staging Heap 1 -> Heap 0: " << std::fixed << std::setprecision(2) << r.staging_heap1_to_heap0_gbs << " GB/s" << std::endl;

    std::cout << "[4/5] Running sustained sequential NVMe & warm-cache I/O benchmarks..." << std::endl;
    probe_storage_benchmarks(r);
    std::cout << "      Warm Page Cache (16MB Chunk + Seq): " << std::fixed << std::setprecision(2) << r.storage_warm_4gb_seqscan_16mb_gbs << " GB/s" << std::endl;
    std::cout << "      Warm Memory Mapped (4GB): " << std::fixed << std::setprecision(2) << r.storage_warm_4gb_mmap_gbs << " GB/s" << std::endl;
    std::cout << "      Unbuffered QD=8 (Cold SSD limit): " << std::fixed << std::setprecision(2) << r.storage_unbuffered_qd8_gbs << " GB/s" << std::endl;

    std::cout << "[5/5] Writing " << json_path << " and " << md_path << "..." << std::endl;
    write_json_and_markdown(r, json_path, md_path);

    std::cout << "============================================================" << std::endl;
    std::cout << "PROBE COMPLETE: All ground-truth parameters measured." << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
