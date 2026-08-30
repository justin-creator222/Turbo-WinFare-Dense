# Architecture — Turbo-WinFare Dense

## 1. System Overview

Turbo-WinFare Dense is a high-performance, low-latency LLM streaming inference engine designed specifically for AMD APUs and Windows 11. It streams Gemma 4 31B Dense weights from NVMe storage over async DMA into Vulkan compute memory, enabling execution of 31B parameter models on hardware with memory constraints.

## 2. Component Diagram

```
+-------------------------------------------------------------------------+
|                        Application / Client Layers                      |
|  - Web GUI (gui/index.html)     - CLI (turbo-dense.exe)                 |
|  - OpenAI-compatible HTTP API   - C API (c_api.h)                       |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                           ForwardRunner                                 |
|  - Tokenizer & Detokenizer (SPM/BPE 262k)                               |
|  - Speculative Coordinator (Modified Rejection Sampling)                |
|  - Draft Runtime (E2B CPU Runner)                                       |
|  - Memory Tier Manager (Tiers 1, 2, 3, 4)                               |
+-------------------------------------------------------------------------+
         |                                           |
         v                                           v
+------------------------------------+   +--------------------------------+
|          LayerStreamer             |   |        Vulkan Compute          |
| - Depth-3 Prefetch Pipeline        |   | - VulkanContext (Vulkan 1.3)   |
| - Win32 Overlapped Async DMA       |   | - VulkanPipelineManager (DXC)  |
| - Two-Tier Residency (Heap 0 / 1)  |   | - 15 SPIR-V Compute Kernels    |
| - LRU / LFU Eviction & Slot Pinning|   | - GPU KV Cache Ring Buffer     |
+------------------------------------+   +--------------------------------+
```

## 3. Two-Tier Memory Architecture

Based on hardware ground truth from `tools/probe_apu`:
- **Heap 0 (`DEVICE_LOCAL` 13.10 GiB):** High-speed GPU compute memory pool (73.57 GB/s). Holds resident/pinned layers in Tier 2/3.
- **Heap 1 (`HOST_VISIBLE` 6.55 GiB):** Direct DMA target memory pool for streaming ring slots (65.30 GB/s GPU compute read).
- **Staging Pipeline:** At tier switch, pinned layers are uploaded from Heap 1 to Heap 0 via `vkCmdCopyBuffer` at 26.94 GB/s.

## 4. Compute Shaders (SPIR-V compiled with DXC)

1. `RMSNormK.hlsl`: Fused RMSNorm with optional weight scaling and epsilon.
2. `GemvInt4.hlsl`: Affine INT4 group-64 quantized matrix-vector multiplication with 2-byte aligned unaligned 32-bit load support (`u32_load`).
3. `GemvInt8.hlsl`: Affine INT8 group-64 quantized matrix-vector multiplication.
4. `QKVEpilogue.hlsl`: Per-head RMSNorm + NeoX rotary positional embeddings (RoPE). Supports full-attention partial rotary factor (0.25) and sliding window full rotary.
5. `Attention.hlsl`: GQA multi-head attention over KV ring buffer.
6. `GeGLU.hlsl`: GeLU-tanh gated feed-forward activation.
7. `ResidualAccum.hlsl`: Residual connection addition with layer scalar scaling.
8. `Softcap.hlsl`: Final logit softcapping ($\text{cap} \cdot \tanh(\text{logits} / \text{cap})$).
9. `LMHeadGreedy.hlsl`: Multi-threaded vocab projection with online argmax reduction.
