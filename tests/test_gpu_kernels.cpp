#include "g4dense/vk_context.hpp"
#include "g4dense/vk_pipeline.hpp"
#include "g4dense/format.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>
#include <random>
#include <iomanip>

using namespace g4dense;

namespace {

inline float bf16_to_f32(uint16_t val) {
    uint32_t u32 = static_cast<uint32_t>(val) << 16;
    float f;
    std::memcpy(&f, &u32, sizeof(float));
    return f;
}

// Every non-quantized tensor in the container is BF16 -- the LayerNorm family included --
// exactly as the MLX safetensors header says. These tests previously encoded norm weights as
// IEEE FP16 to match a reader that had been changed to decode them that way; both were wrong.
// A fixture that disagrees with the real container stops testing anything useful.
inline uint16_t f32_to_bf16(float val) {
    uint32_t u32;
    std::memcpy(&u32, &val, sizeof(float));
    return static_cast<uint16_t>(u32 >> 16);
}

// IEEE FP16 <-> float, for the KV cache. Written out rather than using a compiler intrinsic
// so the test does not depend on one being available.
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    // Round to nearest even on the 13 bits being discarded.
    const uint32_t round_bit = (man >> 12) & 1u;
    const uint32_t sticky = (man & 0xFFFu) != 0u;
    uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13);
    if (round_bit && (sticky || (h & 1u))) ++h;
    return static_cast<uint16_t>(h);
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {
            exp = 1;
            while ((man & 0x400u) == 0) { man <<= 1; --exp; }
            man &= 0x3FFu;
            bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline float gelu_tanh_cpu(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    float inner = SQRT_2_OVER_PI * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

void rms_norm_cpu(const float* x, const float* weight, uint32_t dim, float eps, float* out) {
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) sum_sq += x[i] * x[i];
    float rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + eps);
    for (uint32_t i = 0; i < dim; ++i) {
        out[i] = x[i] * rms * (weight ? weight[i] : 1.0f);
    }
}

bool check_allclose(const float* a, const float* b, size_t count, float atol = 1e-4f, float rtol = 1e-3f, const std::string& name = "") {
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float diff = std::abs(a[i] - b[i]);
        float tol = atol + rtol * std::abs(b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > tol || std::isnan(a[i]) || std::isinf(a[i])) {
            std::cerr << "FAILED [" << name << "] at index " << i << ": GPU=" << a[i]
                      << ", CPU=" << b[i] << ", diff=" << diff << ", tol=" << tol << "\n";
            return false;
        }
    }
    std::cout << "  [PASS] " << std::setw(15) << std::left << name
              << " max_abs_diff=" << std::setprecision(6) << max_diff << "\n";
    return true;
}

} // namespace

int main() {
    std::cout << "========================================================\n"
              << "  Turbo-WinFare Dense: GPU Compute Kernel Parity Tests  \n"
              << "========================================================\n";

    VulkanContext ctx;
    ctx.initialize();

    VulkanPipelineManager pm(ctx);
    pm.initialize_pipelines();

    VkDevice dev = ctx.device();
    VkQueue queue = ctx.compute_queue();

    // Create Command Pool & Buffer
    VkCommandPool cmd_pool{VK_NULL_HANDLE};
    VkCommandPoolCreateInfo pool_ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_ci.queueFamilyIndex = ctx.compute_queue_family();
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(dev, &pool_ci, nullptr, &cmd_pool);

    VkCommandBuffer cmd{VK_NULL_HANDLE};
    VkCommandBufferAllocateInfo cmd_ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_ai.commandPool = cmd_pool;
    cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(dev, &cmd_ai, &cmd);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    // ----------------------------------------------------
    // Test 1: RMSNormK Kernel
    // ----------------------------------------------------
    {
        constexpr uint32_t DIM = 5376;
        std::vector<float> in_x(DIM);
        std::vector<uint16_t> in_w_bf16(DIM);
        std::vector<float> in_w_f32(DIM);
        std::vector<float> cpu_out(DIM);

        for (uint32_t i = 0; i < DIM; ++i) {
            in_x[i] = dist(rng);
            in_w_f32[i] = dist(rng) * 0.5f + 1.0f;
            in_w_bf16[i] = f32_to_bf16(in_w_f32[i]);   // norm weight: BF16, like every
            // non-quantized tensor in the container
            in_w_f32[i] = bf16_to_f32(in_w_bf16[i]);
        }

        rms_norm_cpu(in_x.data(), in_w_f32.data(), DIM, 1e-6f, cpu_out.data());

        VkMemoryAllocation buf_x = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_w = ctx.allocate_buffer(DIM * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_out = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);

        std::memcpy(buf_x.mapped_ptr, in_x.data(), DIM * 4);
        std::memcpy(buf_w.mapped_ptr, in_w_bf16.data(), DIM * 2);

        VkDescriptorSet ds = pm.allocate_descriptor_set(ComputeKernel::RMSNormK);
        pm.update_storage_buffer(ds, 0, buf_x.buffer, 0, DIM * 4);
        pm.update_storage_buffer(ds, 1, buf_w.buffer, 0, DIM * 2);
        pm.update_storage_buffer(ds, 6, buf_out.buffer, 0, DIM * 4); // g_out0 is binding 6

        uint32_t pc[8]{0};
        pc[0] = DIM;        // gp0.x = n
        pc[1] = 0;          // gp0.y = x_byte_off
        pc[2] = 0;          // gp0.z = w_byte_off
        pc[3] = 0;          // gp0.w = out_byte_off
        pc[4] = 1;          // gp1.x = has_weight
        float eps = 1e-6f;
        std::memcpy(&pc[5], &eps, 4); // gp1.y = eps_bits

        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &begin_info);
        pm.bind_kernel(cmd, ComputeKernel::RMSNormK);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm.get_pipeline_layout(ComputeKernel::RMSNormK), 0, 1, &ds, 0, nullptr);
        pm.push_constants(cmd, ComputeKernel::RMSNormK, pc, sizeof(pc));
        pm.dispatch(cmd, 1, 1, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        const float* gpu_res = static_cast<const float*>(buf_out.mapped_ptr);
        bool ok = check_allclose(gpu_res, cpu_out.data(), DIM, 1e-4f, 1e-3f, "RMSNormK");
        assert(ok);

        ctx.free_buffer(buf_x);
        ctx.free_buffer(buf_w);
        ctx.free_buffer(buf_out);
    }

    // ----------------------------------------------------
    // Test 2: GeGLU Kernel
    // ----------------------------------------------------
    {
        constexpr uint32_t DIM = 21504;
        std::vector<float> gate(DIM), up(DIM), cpu_out(DIM);
        for (uint32_t i = 0; i < DIM; ++i) {
            gate[i] = dist(rng);
            up[i] = dist(rng);
            cpu_out[i] = gelu_tanh_cpu(gate[i]) * up[i];
        }

        VkMemoryAllocation buf_gate = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_up = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_out = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);

        std::memcpy(buf_gate.mapped_ptr, gate.data(), DIM * 4);
        std::memcpy(buf_up.mapped_ptr, up.data(), DIM * 4);

        VkDescriptorSet ds = pm.allocate_descriptor_set(ComputeKernel::GeGLU);
        pm.update_storage_buffer(ds, 0, buf_gate.buffer, 0, DIM * 4);
        pm.update_storage_buffer(ds, 1, buf_up.buffer, 0, DIM * 4);
        pm.update_storage_buffer(ds, 6, buf_out.buffer, 0, DIM * 4); // g_out0 is binding 6

        uint32_t pc[8]{0};
        pc[0] = DIM; // gp0.x = n

        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &begin_info);
        pm.bind_kernel(cmd, ComputeKernel::GeGLU);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm.get_pipeline_layout(ComputeKernel::GeGLU), 0, 1, &ds, 0, nullptr);
        pm.push_constants(cmd, ComputeKernel::GeGLU, pc, sizeof(pc));
        pm.dispatch(cmd, (DIM + 255) / 256, 1, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        const float* gpu_res = static_cast<const float*>(buf_out.mapped_ptr);
        bool ok = check_allclose(gpu_res, cpu_out.data(), DIM, 1e-4f, 1e-3f, "GeGLU");
        assert(ok);

        ctx.free_buffer(buf_gate);
        ctx.free_buffer(buf_up);
        ctx.free_buffer(buf_out);
    }

    // ----------------------------------------------------
    // Test 3: GemvInt4 Kernel (Affine INT4 G64)
    // ----------------------------------------------------
    {
        constexpr uint32_t ROWS = 256;
        constexpr uint32_t COLS = 5376;
        constexpr uint32_t GSIZE = 64;
        constexpr uint32_t NUM_GROUPS = COLS / GSIZE;

        std::vector<uint32_t> packed_w(ROWS * (COLS / 8));
        std::vector<uint16_t> scales(ROWS * NUM_GROUPS);
        std::vector<uint16_t> biases(ROWS * NUM_GROUPS);
        std::vector<float> in_x(COLS);
        std::vector<float> cpu_out(ROWS, 0.0f);

        for (uint32_t i = 0; i < COLS; ++i) in_x[i] = dist(rng) * 0.1f;
        for (auto& w : packed_w) w = rng();
        for (auto& s : scales) s = f32_to_bf16(std::abs(dist(rng)) * 0.05f + 0.01f);
        for (auto& b : biases) b = f32_to_bf16(dist(rng) * 0.05f);

        // Compute CPU reference
        for (uint32_t r = 0; r < ROWS; ++r) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < COLS; ++c) {
                uint32_t w_idx = r * (COLS / 8) + (c / 8);
                uint32_t nibble_idx = c % 8;
                uint8_t q_val = (packed_w[w_idx] >> (nibble_idx * 4)) & 0x0F;

                uint32_t g_idx = r * NUM_GROUPS + (c / GSIZE);
                float s = bf16_to_f32(scales[g_idx]);
                float b = bf16_to_f32(biases[g_idx]);
                float w = static_cast<float>(q_val) * s + b;
                sum += w * in_x[c];
            }
            cpu_out[r] = sum;
        }

        VkMemoryAllocation buf_w = ctx.allocate_buffer(packed_w.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_s = ctx.allocate_buffer(scales.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_b = ctx.allocate_buffer(biases.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_x = ctx.allocate_buffer(in_x.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_out = ctx.allocate_buffer(ROWS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);

        std::memcpy(buf_w.mapped_ptr, packed_w.data(), packed_w.size() * 4);
        std::memcpy(buf_s.mapped_ptr, scales.data(), scales.size() * 2);
        std::memcpy(buf_b.mapped_ptr, biases.data(), biases.size() * 2);
        std::memcpy(buf_x.mapped_ptr, in_x.data(), in_x.size() * 4);

        VkDescriptorSet ds = pm.allocate_descriptor_set(ComputeKernel::GemvInt4);
        pm.update_storage_buffer(ds, 0, buf_w.buffer, 0, packed_w.size() * 4);
        pm.update_storage_buffer(ds, 1, buf_s.buffer, 0, scales.size() * 2);
        pm.update_storage_buffer(ds, 2, buf_b.buffer, 0, biases.size() * 2);
        pm.update_storage_buffer(ds, 3, buf_x.buffer, 0, in_x.size() * 4);
        pm.update_storage_buffer(ds, 6, buf_out.buffer, 0, ROWS * 4); // g_out0 is binding 6

        uint32_t pc[8]{0};
        pc[0] = ROWS; // gp0.x = rows
        pc[1] = COLS; // gp0.y = in_dim
        pc[2] = 0;    // gp0.z = w_byte_off
        pc[3] = 0;    // gp0.w = s_byte_off
        pc[4] = 0;    // gp1.x = b_byte_off
        pc[5] = 0;    // gp1.y = x_byte_off
        pc[6] = 0;    // gp1.z = out_byte_off
        pc[7] = 0;    // gp1.w = row_base

        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &begin_info);
        pm.bind_kernel(cmd, ComputeKernel::GemvInt4);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm.get_pipeline_layout(ComputeKernel::GemvInt4), 0, 1, &ds, 0, nullptr);
        pm.push_constants(cmd, ComputeKernel::GemvInt4, pc, sizeof(pc));
        pm.dispatch(cmd, ROWS, 1, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        const float* gpu_res = static_cast<const float*>(buf_out.mapped_ptr);
        bool ok = check_allclose(gpu_res, cpu_out.data(), ROWS, 1e-3f, 1e-3f, "GemvInt4");
        assert(ok);

        ctx.free_buffer(buf_w);
        ctx.free_buffer(buf_s);
        ctx.free_buffer(buf_b);
        ctx.free_buffer(buf_x);
        ctx.free_buffer(buf_out);
    }

    // ----------------------------------------------------
    // Test 4: QKVEpilogue (Head Norm + NeoX RoPE)
    // ----------------------------------------------------
    {
        constexpr uint32_t HEAD_DIM = 256;
        constexpr uint32_t NUM_HEADS = 32;
        constexpr uint32_t TOTAL_DIM = NUM_HEADS * HEAD_DIM;
        constexpr uint32_t ROTATED_PAIRS = HEAD_DIM / 2;
        constexpr float ROPE_THETA = 10000.0f;
        constexpr uint32_t POSITION = 5;

        std::vector<float> in_q(TOTAL_DIM);
        std::vector<uint16_t> in_norm_bf16(HEAD_DIM);
        std::vector<float> in_norm_f32(HEAD_DIM);
        std::vector<float> cpu_out(TOTAL_DIM);

        for (uint32_t i = 0; i < TOTAL_DIM; ++i) in_q[i] = dist(rng);
        for (uint32_t i = 0; i < HEAD_DIM; ++i) {
            in_norm_f32[i] = dist(rng) * 0.5f + 1.0f;
            in_norm_bf16[i] = f32_to_bf16(in_norm_f32[i]);   // q_norm/k_norm: BF16
            in_norm_f32[i] = bf16_to_f32(in_norm_bf16[i]);
        }

        // CPU reference
        for (uint32_t h = 0; h < NUM_HEADS; ++h) {
            float* q_head = in_q.data() + h * HEAD_DIM;
            float* out_head = cpu_out.data() + h * HEAD_DIM;
            rms_norm_cpu(q_head, in_norm_f32.data(), HEAD_DIM, 1e-6f, out_head);

            uint32_t half_dim = HEAD_DIM / 2;
            for (uint32_t p = 0; p < ROTATED_PAIRS; ++p) {
                float freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * p) / static_cast<float>(HEAD_DIM));
                float angle = POSITION * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                float x0 = out_head[p];
                float x1 = out_head[p + half_dim];
                out_head[p] = x0 * cos_a - x1 * sin_a;
                out_head[p + half_dim] = x0 * sin_a + x1 * cos_a;
            }
        }

        VkMemoryAllocation buf_in = ctx.allocate_buffer(TOTAL_DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_norm = ctx.allocate_buffer(HEAD_DIM * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_out = ctx.allocate_buffer(TOTAL_DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);

        std::memcpy(buf_in.mapped_ptr, in_q.data(), TOTAL_DIM * 4);
        std::memcpy(buf_norm.mapped_ptr, in_norm_bf16.data(), HEAD_DIM * 2);

        VkDescriptorSet ds = pm.allocate_descriptor_set(ComputeKernel::QKVEpilogue);
        pm.update_storage_buffer(ds, 0, buf_in.buffer, 0, TOTAL_DIM * 4);
        pm.update_storage_buffer(ds, 1, buf_norm.buffer, 0, HEAD_DIM * 2);
        pm.update_storage_buffer(ds, 6, buf_out.buffer, 0, TOTAL_DIM * 4);

        uint32_t pc[16]{0};   // gp0..gp3: the batched kernels read gp3, so it must be pushed
        pc[0] = HEAD_DIM;           // gp0.x = head_dim
        pc[1] = NUM_HEADS;          // gp0.y = num_heads
        pc[2] = 0;                  // gp0.z = in_off
        pc[3] = 0;                  // gp0.w = out_off
        float eps = 1e-6f;
        std::memcpy(&pc[4], &eps, 4); // gp1.x = eps_bits
        pc[5] = 0;                  // gp1.y = w_off
        pc[6] = 1;                  // gp1.z = has_weight
        pc[7] = 1;                  // gp1.w = do_rope
        pc[8] = ROTATED_PAIRS;      // gp2.x = rotated_pairs
        pc[9] = POSITION;           // gp2.y = position
        float theta = ROPE_THETA;
        std::memcpy(&pc[10], &theta, 4); // gp2.z = theta_bits

        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &begin_info);
        pm.bind_kernel(cmd, ComputeKernel::QKVEpilogue);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm.get_pipeline_layout(ComputeKernel::QKVEpilogue), 0, 1, &ds, 0, nullptr);
        pm.push_constants(cmd, ComputeKernel::QKVEpilogue, pc, sizeof(pc));
        pm.dispatch(cmd, NUM_HEADS, 1, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        const float* gpu_res = static_cast<const float*>(buf_out.mapped_ptr);
        bool ok = check_allclose(gpu_res, cpu_out.data(), TOTAL_DIM, 1e-4f, 1e-3f, "QKVEpilogue");
        assert(ok);

        ctx.free_buffer(buf_in);
        ctx.free_buffer(buf_norm);
        ctx.free_buffer(buf_out);
    }

    // ----------------------------------------------------
    // Test 5: Softcap Kernel (30.0 * tanh(x / 30.0))
    // ----------------------------------------------------
    {
        constexpr uint32_t DIM = 262144;
        constexpr float SOFTCAP = 30.0f;
        std::vector<float> in_x(DIM), cpu_out(DIM);

        for (uint32_t i = 0; i < DIM; ++i) {
            in_x[i] = dist(rng) * 20.0f; // test both saturated and linear regions
            cpu_out[i] = SOFTCAP * std::tanh(in_x[i] / SOFTCAP);
        }

        VkMemoryAllocation buf_x = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_out = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);

        std::memcpy(buf_x.mapped_ptr, in_x.data(), DIM * 4);

        VkDescriptorSet ds = pm.allocate_descriptor_set(ComputeKernel::Softcap);
        pm.update_storage_buffer(ds, 0, buf_x.buffer, 0, DIM * 4);
        pm.update_storage_buffer(ds, 6, buf_out.buffer, 0, DIM * 4);

        uint32_t pc[8]{0};
        pc[0] = DIM;                // gp0.x = n
        pc[1] = 0;                  // gp0.y = in_byte_off
        pc[2] = 0;                  // gp0.z = out_byte_off
        float cap = SOFTCAP;
        std::memcpy(&pc[3], &cap, 4); // gp0.w = cap_bits

        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &begin_info);
        pm.bind_kernel(cmd, ComputeKernel::Softcap);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm.get_pipeline_layout(ComputeKernel::Softcap), 0, 1, &ds, 0, nullptr);
        pm.push_constants(cmd, ComputeKernel::Softcap, pc, sizeof(pc));
        pm.dispatch(cmd, (DIM + 255) / 256, 1, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        const float* gpu_res = static_cast<const float*>(buf_out.mapped_ptr);
        bool ok = check_allclose(gpu_res, cpu_out.data(), DIM, 1e-4f, 1e-3f, "Softcap");
        assert(ok);

        ctx.free_buffer(buf_x);
        ctx.free_buffer(buf_out);
    }

    // ----------------------------------------------------
    // Test 3b: GemmInt4Batch -- the prefill kernel
    //
    // Exercised at M > 1 with a DIFFERENT activation vector per position, and with a non-zero
    // out_stride, so a kernel that computed only the first column (which is what the enum
    // silently did before this shader existed) or that mixed positions up cannot pass.
    //
    // in_dim 5376 is deliberately not a multiple of the 512-column tile: the last tile is 256
    // wide, so the partial-tile path is covered too.
    // ----------------------------------------------------
    {
        constexpr uint32_t ROWS = 256;
        constexpr uint32_t COLS = 5376;
        constexpr uint32_t GSIZE = 64;
        constexpr uint32_t NUM_GROUPS = COLS / GSIZE;
        constexpr uint32_t BATCH = 5;          // not a power of two, and < kGemmMaxBatch

        std::vector<uint32_t> packed_w(ROWS * (COLS / 8));
        std::vector<uint16_t> scales(ROWS * NUM_GROUPS);
        std::vector<uint16_t> biases(ROWS * NUM_GROUPS);
        std::vector<float> in_x(static_cast<size_t>(BATCH) * COLS);
        std::vector<float> cpu_out(static_cast<size_t>(BATCH) * ROWS, 0.0f);

        for (auto& v : in_x) v = dist(rng) * 0.1f;
        for (auto& w : packed_w) w = rng();
        for (auto& s : scales) s = f32_to_bf16(std::abs(dist(rng)) * 0.05f + 0.01f);
        for (auto& b : biases) b = f32_to_bf16(dist(rng) * 0.05f);

        for (uint32_t m = 0; m < BATCH; ++m) {
            for (uint32_t r = 0; r < ROWS; ++r) {
                float sum = 0.0f;
                for (uint32_t c = 0; c < COLS; ++c) {
                    const uint32_t w_idx = r * (COLS / 8) + (c / 8);
                    const uint8_t q_val = (packed_w[w_idx] >> ((c % 8) * 4)) & 0x0F;
                    const uint32_t g_idx = r * NUM_GROUPS + (c / GSIZE);
                    const float wv = static_cast<float>(q_val) * bf16_to_f32(scales[g_idx])
                                   + bf16_to_f32(biases[g_idx]);
                    sum += wv * in_x[static_cast<size_t>(m) * COLS + c];
                }
                cpu_out[static_cast<size_t>(m) * ROWS + r] = sum;
            }
        }

        VkMemoryAllocation buf_w = ctx.allocate_buffer(packed_w.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_s = ctx.allocate_buffer(scales.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_b = ctx.allocate_buffer(biases.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_x = ctx.allocate_buffer(in_x.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_out = ctx.allocate_buffer(cpu_out.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);

        std::memcpy(buf_w.mapped_ptr, packed_w.data(), packed_w.size() * 4);
        std::memcpy(buf_s.mapped_ptr, scales.data(), scales.size() * 2);
        std::memcpy(buf_b.mapped_ptr, biases.data(), biases.size() * 2);
        std::memcpy(buf_x.mapped_ptr, in_x.data(), in_x.size() * 4);
        std::memset(buf_out.mapped_ptr, 0, cpu_out.size() * 4);

        VkDescriptorSet ds = pm.allocate_descriptor_set(ComputeKernel::GemmInt4Batch);
        pm.update_storage_buffer(ds, 0, buf_w.buffer, 0, packed_w.size() * 4);
        pm.update_storage_buffer(ds, 1, buf_s.buffer, 0, scales.size() * 2);
        pm.update_storage_buffer(ds, 2, buf_b.buffer, 0, biases.size() * 2);
        pm.update_storage_buffer(ds, 3, buf_x.buffer, 0, in_x.size() * 4);
        pm.update_storage_buffer(ds, 6, buf_out.buffer, 0, cpu_out.size() * 4);

        uint32_t pc[16]{0};   // gp0..gp3: the batched kernels read gp3, so it must be pushed
        pc[0] = ROWS;
        pc[1] = COLS;
        pc[8]  = BATCH;
        pc[9]  = COLS * 4;   // x_stride between positions
        pc[10] = ROWS * 4;   // out_stride between positions

        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &begin_info);
        pm.bind_kernel(cmd, ComputeKernel::GemmInt4Batch);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pm.get_pipeline_layout(ComputeKernel::GemmInt4Batch), 0, 1, &ds, 0, nullptr);
        pm.push_constants(cmd, ComputeKernel::GemmInt4Batch, pc, sizeof(pc));
        pm.dispatch(cmd, (ROWS + kGemmRowsPerGroup - 1) / kGemmRowsPerGroup, 1, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        bool ok = check_allclose(static_cast<const float*>(buf_out.mapped_ptr), cpu_out.data(),
                                 cpu_out.size(), 1e-3f, 1e-2f, "GemmInt4Batch");
        assert(ok);
        ctx.free_buffer(buf_w); ctx.free_buffer(buf_s); ctx.free_buffer(buf_b);
        ctx.free_buffer(buf_x); ctx.free_buffer(buf_out);
    }

    // ----------------------------------------------------
    // Test 6: ResidualAccum -- out[i] = (hidden[i] + res[i] * res_scale) * out_scale
    //
    // Both scales are exercised with distinct non-unit values, so a kernel that ignores either
    // one (or that swaps them) fails here. The runner drives OUT_SCALE with the layer's
    // layer_scalar on the FFN residual add and 1.0 on the attention one.
    // ----------------------------------------------------
    {
        constexpr uint32_t DIM = 5376;
        constexpr float RES_SCALE = 0.75f;
        constexpr float OUT_SCALE = 0.089355f;   // the model's layer_scalar at layer 0
        std::vector<float> in_h(DIM), in_r(DIM), cpu_out(DIM);
        for (uint32_t i = 0; i < DIM; ++i) {
            in_h[i] = dist(rng);
            in_r[i] = dist(rng);
            cpu_out[i] = (in_h[i] + in_r[i] * RES_SCALE) * OUT_SCALE;
        }

        VkMemoryAllocation buf_h = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_r = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_out = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        std::memcpy(buf_h.mapped_ptr, in_h.data(), DIM * 4);
        std::memcpy(buf_r.mapped_ptr, in_r.data(), DIM * 4);

        VkDescriptorSet ds = pm.allocate_descriptor_set(ComputeKernel::ResidualAccum);
        pm.update_storage_buffer(ds, 0, buf_h.buffer, 0, DIM * 4);
        pm.update_storage_buffer(ds, 1, buf_r.buffer, 0, DIM * 4);
        pm.update_storage_buffer(ds, 6, buf_out.buffer, 0, DIM * 4);

        uint32_t pc[8]{0};
        pc[0] = DIM;
        float rsc = RES_SCALE;
        float osc = OUT_SCALE;
        std::memcpy(&pc[1], &rsc, 4);
        std::memcpy(&pc[2], &osc, 4);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &bi);
        pm.bind_kernel(cmd, ComputeKernel::ResidualAccum);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm.get_pipeline_layout(ComputeKernel::ResidualAccum), 0, 1, &ds, 0, nullptr);
        pm.push_constants(cmd, ComputeKernel::ResidualAccum, pc, sizeof(pc));
        pm.dispatch(cmd, (DIM + 255) / 256, 1, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        bool ok = check_allclose(static_cast<const float*>(buf_out.mapped_ptr), cpu_out.data(),
                                 DIM, 1e-4f, 1e-3f, "ResidualAccum");
        assert(ok);
        ctx.free_buffer(buf_h); ctx.free_buffer(buf_r); ctx.free_buffer(buf_out);
    }

    // ----------------------------------------------------
    // Test 7: EmbedLookup -- dequantize one row, scaled by sqrt(d_model)
    //
    // Gemma multiplies the embedding by sqrt(D) before layer 0; dropping that scale yields
    // fluent-but-wrong output rather than an obvious failure, so it is asserted here.
    // ----------------------------------------------------
    {
        constexpr uint32_t DIM = 5376;
        constexpr uint32_t VOCAB = 64;          // enough rows to exercise row addressing
        constexpr uint32_t TOKEN = 37;
        const uint32_t groups = DIM / 64;
        const float scale = std::sqrt(static_cast<float>(DIM));

        std::vector<uint8_t> w(static_cast<size_t>(VOCAB) * (DIM / 2));
        std::vector<uint16_t> s(static_cast<size_t>(VOCAB) * groups);
        std::vector<uint16_t> b(static_cast<size_t>(VOCAB) * groups);
        std::uniform_int_distribution<int> nib(0, 15);
        for (auto& byte : w) byte = static_cast<uint8_t>(nib(rng) | (nib(rng) << 4));
        for (auto& v : s) v = f32_to_bf16(dist(rng) * 0.01f);
        for (auto& v : b) v = f32_to_bf16(dist(rng) * 0.01f);

        std::vector<float> cpu_out(DIM);
        for (uint32_t c = 0; c < DIM; ++c) {
            const uint32_t g = c / 64;
            const size_t byte_off = static_cast<size_t>(TOKEN) * (DIM / 2) + (c >> 1);
            const uint8_t byte = w[byte_off];
            const uint32_t q = ((c & 1u) == 0u) ? (byte & 0x0Fu) : (byte >> 4);
            const float sv = bf16_to_f32(s[static_cast<size_t>(TOKEN) * groups + g]);
            const float bv = bf16_to_f32(b[static_cast<size_t>(TOKEN) * groups + g]);
            cpu_out[c] = (static_cast<float>(q) * sv + bv) * scale;
        }

        VkMemoryAllocation buf_w = ctx.allocate_buffer(w.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_s = ctx.allocate_buffer(s.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_b = ctx.allocate_buffer(b.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        VkMemoryAllocation buf_out = ctx.allocate_buffer(DIM * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
        std::memcpy(buf_w.mapped_ptr, w.data(), w.size());
        std::memcpy(buf_s.mapped_ptr, s.data(), s.size() * 2);
        std::memcpy(buf_b.mapped_ptr, b.data(), b.size() * 2);

        VkDescriptorSet ds = pm.allocate_descriptor_set(ComputeKernel::EmbedLookup);
        pm.update_storage_buffer(ds, 0, buf_w.buffer, 0, w.size());
        pm.update_storage_buffer(ds, 1, buf_s.buffer, 0, s.size() * 2);
        pm.update_storage_buffer(ds, 2, buf_b.buffer, 0, b.size() * 2);
        pm.update_storage_buffer(ds, 6, buf_out.buffer, 0, DIM * 4);

        uint32_t pc[8]{0};
        pc[0] = TOKEN;
        pc[1] = DIM;
        float sc = scale;
        std::memcpy(&pc[6], &sc, 4);   // gp1.z = scale_bits

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &bi);
        pm.bind_kernel(cmd, ComputeKernel::EmbedLookup);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm.get_pipeline_layout(ComputeKernel::EmbedLookup), 0, 1, &ds, 0, nullptr);
        pm.push_constants(cmd, ComputeKernel::EmbedLookup, pc, sizeof(pc));
        pm.dispatch(cmd, 1, 1, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        bool ok = check_allclose(static_cast<const float*>(buf_out.mapped_ptr), cpu_out.data(),
                                 DIM, 1e-3f, 1e-3f, "EmbedLookup");
        assert(ok);
        ctx.free_buffer(buf_w); ctx.free_buffer(buf_s); ctx.free_buffer(buf_b); ctx.free_buffer(buf_out);
    }

    // ----------------------------------------------------
    // Test 8: Attention -- GQA with ring-buffer KV and sliding-window masking
    //
    // The most complex kernel and, until now, the one with no parity coverage. A data race on
    // the reused `s_partial` groupshared slot lived here undetected through two validation
    // rounds: the end-to-end oracle diff only ever ran a single token, where the race window
    // is essentially zero. Both cases below use a long span precisely so that window is wide.
    //
    // Attention scale is 1.0, NOT 1/sqrt(head_dim) -- the query scaling is folded into q_norm.
    // ----------------------------------------------------
    {
        auto run_attention_case = [&](const char* name, uint32_t q_heads, uint32_t kv_heads,
                                      uint32_t head_dim, uint32_t n_pos, uint32_t first,
                                      uint32_t capacity) {
            const uint32_t kv_elems = capacity * kv_heads * head_dim;
            std::vector<float> q(q_heads * head_dim), kc(kv_elems), vc(kv_elems);
            for (auto& x : q) x = dist(rng);
            for (auto& x : kc) x = dist(rng);
            for (auto& x : vc) x = dist(rng);

            // The KV cache is FP16 on the device. Round the reference inputs through half
            // precision so this test measures the KERNEL, not the storage format -- otherwise
            // it would report the quantization error of the cache as a kernel defect. Whether
            // that error is acceptable end to end is a separate question, answered by the
            // argmax-exact oracle diff in run_gpu_forward_test.
            std::vector<uint16_t> kc_h(kv_elems), vc_h(kv_elems);
            for (uint32_t i = 0; i < kv_elems; ++i) {
                kc_h[i] = f32_to_f16(kc[i]);  kc[i] = f16_to_f32(kc_h[i]);
                vc_h[i] = f32_to_f16(vc[i]);  vc[i] = f16_to_f32(vc_h[i]);
            }

            // Attention scale is head_dim^-0.5, passed to the kernel in gp2.z. Exercising a
            // non-unit scale matters: the kernel hardcoded 1.0 until it was found to be the
            // cause of incoherent generation, and a test that only ever passes 1.0 would not
            // notice the scale being dropped again.
            const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // CPU reference
            std::vector<float> cpu_out(q_heads * head_dim, 0.0f);
            const uint32_t gqa = q_heads / kv_heads;
            for (uint32_t h = 0; h < q_heads; ++h) {
                const uint32_t kvh = h / gqa;
                std::vector<float> scores;
                scores.reserve(n_pos - first);
                float best = -1e30f;
                for (uint32_t t = first; t < n_pos; ++t) {
                    const uint32_t slot = t % capacity;
                    const float* kp = &kc[(static_cast<size_t>(slot) * kv_heads + kvh) * head_dim];
                    float d = 0.0f;
                    for (uint32_t i = 0; i < head_dim; ++i) d += q[h * head_dim + i] * kp[i];
                    scores.push_back(d * attn_scale);
                    best = std::max(best, d);
                }
                float sum = 0.0f;
                for (auto& s : scores) { s = std::exp(s - best); sum += s; }
                const float inv = 1.0f / sum;
                for (uint32_t t = first; t < n_pos; ++t) {
                    const uint32_t slot = t % capacity;
                    const float* vp = &vc[(static_cast<size_t>(slot) * kv_heads + kvh) * head_dim];
                    const float w = scores[t - first] * inv;
                    for (uint32_t i = 0; i < head_dim; ++i) cpu_out[h * head_dim + i] += w * vp[i];
                }
            }

            VkMemoryAllocation bq = ctx.allocate_buffer(q.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
            VkMemoryAllocation bk = ctx.allocate_buffer(kc_h.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
            VkMemoryAllocation bv = ctx.allocate_buffer(vc_h.size() * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
            VkMemoryAllocation bo = ctx.allocate_buffer(cpu_out.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryResidency::HostVisibleMapped);
            std::memcpy(bq.mapped_ptr, q.data(), q.size() * 4);
            std::memcpy(bk.mapped_ptr, kc_h.data(), kc_h.size() * 2);
            std::memcpy(bv.mapped_ptr, vc_h.data(), vc_h.size() * 2);
            std::memset(bo.mapped_ptr, 0, cpu_out.size() * 4);

            VkDescriptorSet ds = pm.allocate_descriptor_set(ComputeKernel::Attention);
            pm.update_storage_buffer(ds, 0, bq.buffer, 0, q.size() * 4);
            pm.update_storage_buffer(ds, 1, bk.buffer, 0, kc_h.size() * 2);
            pm.update_storage_buffer(ds, 2, bv.buffer, 0, vc_h.size() * 2);
            pm.update_storage_buffer(ds, 6, bo.buffer, 0, cpu_out.size() * 4);

            uint32_t pc[16]{0};   // gp0..gp3: the batched kernels read gp3, so it must be pushed
            pc[0] = q_heads; pc[1] = kv_heads; pc[2] = head_dim; pc[3] = n_pos;
            pc[4] = first;   pc[9] = capacity;
            std::memcpy(&pc[10], &attn_scale, 4);   // gp2.z: head_dim^-0.5

            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkResetCommandBuffer(cmd, 0);
            vkBeginCommandBuffer(cmd, &bi);
            pm.bind_kernel(cmd, ComputeKernel::Attention);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm.get_pipeline_layout(ComputeKernel::Attention), 0, 1, &ds, 0, nullptr);
            pm.push_constants(cmd, ComputeKernel::Attention, pc, sizeof(pc));
            pm.dispatch(cmd, q_heads, 1, 1);
            vkEndCommandBuffer(cmd);

            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);

            bool ok = check_allclose(static_cast<const float*>(bo.mapped_ptr), cpu_out.data(),
                                     cpu_out.size(), 1e-4f, 1e-3f, name);
            assert(ok);
            ctx.free_buffer(bq); ctx.free_buffer(bk); ctx.free_buffer(bv); ctx.free_buffer(bo);
        };

        // Full-attention shape: capacity exceeds the span, so `t % capacity` is an identity.
        run_attention_case("Attention/global", 8, 4, 64, /*n_pos=*/300, /*first=*/0, /*cap=*/512);
        // Sliding-window shape: the span wraps the ring, exercising the modulo indexing.
        run_attention_case("Attention/sliding", 8, 4, 64, /*n_pos=*/300, /*first=*/173, /*cap=*/128);
    }

    vkDestroyCommandPool(dev, cmd_pool, nullptr);

    // Deliberately NOT "ALL KERNELS": PostAttn, LayerTail, LMHeadGreedy, ArgmaxReduce,
    // GemvInt8 and GemmInt4Batch still have no parity coverage here. The end-to-end oracle
    // diff in test_gpu_forward exercises them as a chain, which catches gross errors but
    // cannot localize one -- and, as the Attention race showed, cannot even detect one that
    // only manifests beyond a single token. Claiming completeness we do not have is what let
    // a 5-of-13 suite print "ALL ... PASSED" through two validation rounds.
    std::cout << "\n[test_gpu_kernels] 9 of 11 kernels verified against the CPU reference: "
              << "RMSNormK, GeGLU, GemvInt4, GemmInt4Batch, QKVEpilogue, Softcap, ResidualAccum, EmbedLookup,"
              << " Attention.\n"
              << "  Not yet covered: LMHeadGreedy, "
              << "ArgmaxReduce, GemvInt8.\n";
    return 0;
}
