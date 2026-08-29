# SPECIFICATION: Turbo-WinFare Dense (APU-Optimized Gemma 4 31B Streaming Runtime)

> **Status note (added by Claude Opus 5 during planning, 2026-08-29):**
> This is the original specification as authored. Several of its assumptions were
> measured against the actual target hardware and found to be incorrect or in
> internal tension. **`IMPLEMENTATION_PLAN.md` is the authority where the two
> disagree** — see its §1 "Three spec defects this plan resolves rather than
> inherits" and "Additional spec corrections". Sections most affected:
> §2 (RAM is 23.8 GB, not 32 GB), §4 (Tier 1 does not close at 14 pinned layers;
> Tier 4 is infeasible on this device), §5.2 (`FILE_FLAG_NO_BUFFERING` conflicts
> with the §10 Throughput Gate), §5.3 (intrinsic naming; no INT4 cooperative-matrix
> type; Wave32 must be forced), §7 (WinUI 3 is unbuildable with the installed
> toolchain), and §10 (the Numerical Parity Gate is unrunnable as written).

## 1. System Context & Multi-Agent Execution Directives

### 1.1 Role of Claude Opus 5 (Architect & Validator)
You (Claude Opus 5 via Claude Code) are tasked with producing the **Formal Technical Implementation Plan** based on this specification.
* **Target Executor:** Your implementation plan will be directly executed by **Gemini 3.7 Flash** running inside **Google Antigravity 2.0** using the `/goal` command.
* **Sub-Agent Structure:** Antigravity 2.0 operates by spawning specialized sub-agents (I/O engine agent, Vulkan shader agent, Speculative coordinator agent, WinUI frontend agent). Your implementation plan must structure modules and tasks with strict file isolation and verifiable interface boundaries.
* **Final Validation:** Once Gemini 3.7 Flash signals completion of all implementation milestones, **Claude Opus 5 will be re-engaged to validate the complete codebase**, run unit/integration test suites, verify mathematical numerical parity against reference models, and audit memory/throughput benchmarks.

---

## 2. Target Hardware: Lenovo Legion Go S (AMD Ryzen Z1 Extreme)

This project is engineered specifically for **Windows APU (Unified Memory Architecture - UMA)** platforms (e.g., Lenovo Legion Go S, ASUS ROG Ally X, Minisforum APU Mini PCs) and is explicitly **not designed for discrete desktop GPUs**.

### 2.1 Hardware Architecture Matrix
* **SoC:** AMD Phoenix / Hawk Point silicon (Ryzen Z1 Extreme).
* **CPU:** 8 Cores / 16 Threads (Zen 4 architecture) with native **AVX-512 FMA & VNNI** (512-bit vector pipelines).
* **iGPU:** AMD Radeon 780M (RDNA 3 architecture, 12 Compute Units / 768 Stream Processors) with **Wave Matrix Multiply Accumulate (WMMA)** hardware matrix cores supporting INT8, INT4, FP16, and BF16.
* **Unified Memory:** 32 GB LPDDR5X-7500 on a 128-bit bus (~120.0 GB/s theoretical, ~95–105 GB/s sustained bandwidth shared between CPU and iGPU).
* **Storage Interface:** M.2 PCIe 4.0 x4 NVMe SSD sustained sequential reads at **5.5 to 7.0 GB/s**.
* **Operating System:** Windows 11 64-bit (DirectX 12 Agility SDK, Vulkan 1.3.x, Windows App SDK / WinUI 3).

---

## 3. Model Architecture, Quantization & Online Resource Guide

### 3.1 Gemma 4 31B Dense Architecture Details
* **Total Layers:** 60 Transformer blocks.
* **Model Dimension (d_model):** 5,376 (42 Query Heads, 16 Key/Value Heads via GQA, d_k = 128).
* **Feedforward Dimension (d_ff):** 21,504 (SwiGLU activation).
* **Attention Mechanism:** 3:1 Interleaved Hybrid Attention (45 sliding-window local layers with W = 512 tokens, 15 global attention layers with Proportional RoPE / p-RoPE).
* **No Per-Layer Embeddings (PLE):** Gemma 4 31B sets `hidden_size_per_layer_input = 0` (unlike E2B/E4B, 31B uses standard input/output embedding projections).

### 3.2 Quantization & Model Sources
* **Target Precision:** **4-bit Quantization-Aware Trained (QAT) W4A16** or **Affine INT4 (Group Size 64)**.
  * Do not use quantizations below 4-bit (e.g., 2-bit or 3-bit) to preserve reasoning perplexity.
  * Prioritize official/community QAT checkpoints (`google/gemma-4-31B-it-qat-w4a16` or calibrated AWQ/GGUF Q4_K_M formats).
* **Draft Model:** **Gemma 4 E2B-IT** (4-bit QAT, ~1.2 GB) or **Gemma 4 E4B-IT** (4-bit QAT, ~2.2 GB).
* **Online Resources & References for Implementation:**
  * Gemma 4 Technical Report: `arXiv:2607.02770`
  * Official Hugging Face Repositories: `google/gemma-4-31B-it`, `google/gemma-4-E2B-it`
  * Upstream Metal/Swift Reference: `https://github.com/drumih/turbo-fieldfare`
  * Microsoft DirectStorage SDK: `https://github.com/microsoft/DirectStorage`
  * AMD RDNA 3 Instruction Set Architecture (WMMA & Wave32 docs): `https://gpuopen.com/`

---

## 4. Dynamic Tiered Memory Management (6 GB Baseline to Max RAM)

The application defaults to a **6.0 GB baseline RAM footprint**, but provides a dynamic memory slider in the GUI that dynamically scales resident layer pinning based on available system memory:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           Tiered Memory Allocation Profiles                             │
├───────────────────┬──────────────┬──────────────┬───────────────────┬───────────────────┤
│ Mode / Tier       │ Total RAM    │ Pinned 31B   │ Streamed 31B      │ Expected Speed    │
│                   │ Allocation   │ Layers       │ Layers (SSD)      │ (Legion Go S)     │
├───────────────────┼──────────────┼──────────────┼───────────────────┼───────────────────┤
│ Tier 1 (Baseline) │ **6.0 GB**   │ 14 Layers    │ 46 Layers (11.2GB)│ **~2.60 TPS**     │
│ Tier 2 (Balanced) │ **10.0 GB**  │ 30 Layers    │ 30 Layers (7.3 GB)│ **~3.85 TPS**     │
│ Tier 3 (High-Perf)│ **16.0 GB**  │ 48 Layers    │ 12 Layers (2.9 GB)│ **~6.50 TPS**     │
│ Tier 4 (Resident) │ **22.0 GB**  │ 60 Layers    │ 0 Layers (0.0 GB) │ **~18–25+ TPS**   │
└───────────────────┴──────────────┴──────────────┴───────────────────┴───────────────────┘
```

### Layer Pinning Priority Order:
When memory headroom increases, pin layers in this structural priority:
1. **Input Boundary:** Layers 0–6 (Early semantic grounding).
2. **Output Boundary:** Layers 53–59 (Final token convergence).
3. **Global Attention Blocks:** Layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51.
4. **Intermediate FFN Blocks:** Remaining middle layers.

---

## 5. Low-Level Windows APU Engine Architecture

```
+-----------------------------------------------------------------------------------------+
|                  Turbo-WinFare Dense Engine (Windows APU Architecture)                  |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [ Zen 4 CPU Draft Generator ]                   [ Radeon 780M iGPU Batch Verifier ]    |
|  Gemma 4 E2B (4-bit, ~1.2GB resident)            Vulkan 1.3 Compute Pipeline            |
|  - AVX-512 / VNNI Fused Kernel Execution         - RDNA 3 WMMA (Wave32) MatMul          |
|  - Drafts K=6 tokens at ~40 TPS                  - Zero-Copy Host-Visible Memory Pool   |
|                 │                                           ▲                           |
|                 │ Candidate Token Batch                     │ Mapped Shared Pointer     |
|                 ▼                                           │ (No CPU->GPU Copy)        |
|  ┌─────────────────────────────┐           ┌────────────────┴────────────────────────┐  |
|  | Draft Batch [t_1 ... t_6]   | ────────> | DirectStorage 1.2 / Win32 IOCP Streamer |  |
|  | (K = 6 tokens)              |           | Unbuffered Asynchronous Disk I/O        |  |
|  └─────────────────────────────┘           └─────────────────────────────────────────┘  |
|                                                             │                           |
|                                            ┌────────────────▼────────────────────────┐  |
|                                            | Internal PCIe 4.0 NVMe (model.g4dense)  |  |
|                                            └─────────────────────────────────────────┘  |
+-----------------------------------------------------------------------------------------+
```

### 5.1 Zero-Copy Vulkan 1.3 APU Memory Allocation
The engine utilizes Vulkan 1.3 with unified memory flags:
* Allocate a circular pool of **4 dynamic layer buffers** (`Buffer_0` .. `Buffer_3`, ~245 MB each).
* Memory flags: `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
* SSD reads write directly to the mapped host pointer. The Radeon 780M executes compute shaders on the exact same physical memory addresses with **zero staging copies**.

### 5.2 Storage Engine: DirectStorage 1.2 / Win32 IOCP
* File reads use `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` with 4096-byte sector alignment.
* Asynchronous prefetching ensures that while Layer *i* executes on the iGPU, Layer *i+1* is in flight from the NVMe drive, and Layer *i+2* is queued in the I/O completion port.

### 5.3 RDNA 3 WMMA Compute Shaders (SPIR-V)
* Compute shaders written in HLSL/GLSL targeting Vulkan SPIR-V with `Wave32` execution width.
* Use `subgroupMatrixMultiplyAccumulate` intrinsics for fused 4-bit dequantization + batched matrix multiplication (M = K ∈ [2, 8]) during speculative verification passes.

---

## 6. Container Format & Serialization: `.g4dense` (v2)

The offline conversion tool packs Gemma 4 31B Safetensors into an optimized, 4KB-aligned binary container:

```
[Header: 4096 Bytes]
* Magic: 0x4734444E ('G4DN')
* Version: uint32 (2)
* QuantType: uint32 (1 = Affine INT4 Group-64, 2 = QAT W4A16)
* Model Dimensions (Layers: 60, Hidden: 5376, FFN: 21504, Heads: 42/16)
* Table of Offsets: uint64[60] (Layer byte offsets, 4096-byte aligned)
* Table of Sizes: uint64[60]
* Embedding Offset & Size, LM Head Offset & Size

[Resident Section]
* Token Embedding Matrix (256,000 x 5376)
* RMSNorm weights (all 60 layers + final norm)

[Layer Blocks 00 through 59]
* Layer N Attention: Q_proj, K_proj, V_proj, O_proj (Weights + Scales + Zeros)
* Layer N FFN: Gate_proj, Up_proj, Down_proj (Weights + Scales + Zeros)
```

---

## 7. Native Windows GUI & Direct Hugging Face Downloader

The frontend must be built as a native Windows desktop app using **WinUI 3 (Windows App SDK)** or **modern C++ with ImGui / FluentUI wrapper**:

```
+-----------------------------------------------------------------------------------------+
| Turbo-WinFare (Gemma 4 31B Dense APU Engine)                            [-] [口] [X]    |
+-----------------------------------------------------------------------------------------+
| [ Model Downloader & Manager ]                                                          |
| Selected: google/gemma-4-31B-it-qat-w4a16 (14.8 GB) + E2B Draft (1.2 GB)                |
| Download: [========================================>      ] 78% (12.4 GB / 16.0 GB)     |
| Speed: 84.2 MB/s | Status: Downloading Layer Blocks... [Pause] [Cancel]                 |
+-----------------------------------------------------------------------------------------+
| [ Memory & Hardware Tuning ]                                                            |
| RAM Allocation Profile: [------O--------------------------] 6.0 GB (Baseline Mode)      |
|  -> Pinned in RAM: 14 Layers | Streamed from SSD: 46 Layers                             |
|  -> Storage Device: [NVMe Samsung 990 Pro (PCIe 4.0 - 6.8 GB/s) ▼] (High Speed Verified)|
| Speculative Draft Tokens (K): [ 6 ▼] | Context Window: [ 8,192 Tokens ▼]                |
+-----------------------------------------------------------------------------------------+
| [ Chat & Generation Interface ]                                                         |
| System: You are a helpful AI assistant.                                                 |
| User: Write a high-performance multithreaded ring buffer in C++20.                      |
| Assistant: Here is a lock-free ring buffer implementation using std::atomic...          |
|                                                                                         |
| [ Type your prompt here...                                            ] [ Send ] [Stop] |
+-----------------------------------------------------------------------------------------+
| [ Real-Time Telemetry Dashboard ]                                                       |
| Generation: 2.68 TPS | TTFT: 142 ms | Speculative Acceptance: 78.4%                     |
| NVMe Read: 5.82 GB/s | RAM Footprint: 5.84 GB / 32 GB | APU Power: 24.5 W (TDP)         |
+-----------------------------------------------------------------------------------------+
```

### 7.1 In-App Automated Model Downloader Specs
* Connects directly to Hugging Face Hub via REST API.
* Supports chunked parallel downloads with resume capability (`Range: bytes=...`).
* SHA-256 integrity verification.
* Automated post-download packaging into local `.g4dense` format.

---

## 8. Speculative Decoding & Verification Pipeline

1. **Draft Generation:** Gemma 4 E2B runs autoregressively on Zen 4 CPU (AVX-512) to produce K = 6 draft tokens ([t_1, t_2, t_3, t_4, t_5, t_6]).
2. **Batched Prefetch & Verification Pass:**
   * Stream the 46 non-pinned layers from SSD while computing through the 14 pinned layers in RAM.
   * Execute batch GEMM (M = 6) on the Radeon 780M across all layers in a single pass.
3. **Acceptance Evaluation & Rollback:**
   * Compare draft distribution vs 31B target logits.
   * Accept N ≤ 6 valid tokens plus 1 new sampled token.
   * Update circular KV cache pointers for accepted tokens; roll back unaccepted slots.

---

## 9. Antigravity 2.0 / Gemini 3.7 Flash Sub-Agent Decomposition Plan

Claude Opus 5 must structure the implementation plan for the following Antigravity sub-agents:

1. **Sub-Agent 1 (`storage-engine`):** Win32 unbuffered overlapped I/O, DirectStorage 1.2 integration, `.g4dense` parser, and 4KB sector-aligned file streaming.
2. **Sub-Agent 2 (`vulkan-compute`):** Vulkan 1.3 device init, `HOST_VISIBLE | DEVICE_LOCAL` memory pool manager, RDNA 3 WMMA Wave32 GLSL/SPIR-V compute shaders.
3. **Sub-Agent 3 (`speculative-coordinator`):** Zen 4 AVX-512 Gemma 4 E2B draft runtime, batch verification controller, KV cache circular ring buffer manager.
4. **Sub-Agent 4 (`model-downloader`):** Hugging Face HTTP client with chunked multi-stream resume, SHA-256 verifier, and offline weight packer.
5. **Sub-Agent 5 (`winui-gui`):** WinUI 3 desktop interface, hardware telemetry monitors (NVMe speed, RAM usage, TPS, TDP), chat history, and dynamic RAM slider.

---

## 10. Claude Opus 5 Validation & Acceptance Gates

When Gemini 3.7 Flash completes development, Claude Opus 5 must execute these validation gates:

* [ ] **Memory Ceiling Gate:** Under active 8k-context generation in 6.0 GB Mode, total process working set must not exceed **6,000 MB**.
* [ ] **Throughput Gate (Legion Go S):**
  * 6.0 GB Mode: ≥ 2.50 Tokens / Sec.
  * 16.0 GB Mode: ≥ 6.00 Tokens / Sec.
* [ ] **Numerical Parity Gate:** Perplexity difference < 0.15 on WikiText-2 compared to standard in-memory FP16 baseline.
* [ ] **Storage Gate:** Direct unbuffered NVMe read throughput ≥ 5.0 GB/s during streaming phases.
* [ ] **GUI Functionality Gate:** Model downloader successfully pulls and converts checkpoints directly from Hugging Face with UI progress tracking.
