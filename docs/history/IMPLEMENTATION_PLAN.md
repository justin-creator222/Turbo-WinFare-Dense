# Turbo-WinFare Dense — Formal Technical Implementation Plan

**Spec:** `spec.md` (Turbo-WinFare Dense, Gemma 4 31B, APU streaming runtime)
**Executor:** Gemini 3.7 Flash / Google Antigravity 2.0, `/goal` with sub-agent delegation
**Validator:** Claude Opus 5 (re-engaged at Phase 6 against §10 Acceptance Gates)
**Target repo:** `<repo root>` (currently empty)

---

## 1. Context

### Why this project exists

The spec calls for a streaming inference engine that runs **Gemma 4 31B Dense** on a handheld
APU with a 6 GB RAM baseline — a model whose 4-bit weights are 15.2 GB, on a device with
23.8 GB of unified memory shared between CPU and iGPU. The only way that closes is to keep a
minority of layers resident and stream the rest from NVMe per forward pass, then amortize
that streaming cost across multiple tokens with speculative decoding.

### What already exists and shapes this plan

`<sibling engine>` is a **shipped, working** Turbo-WinFare for Gemma 4 26B-A4B
(MoE), targeting this exact hardware, at **8.8 tok/s**. Its `CLAUDE.md` is a 490-line record
of what was measured, what was tried and rejected, and which forward-pass constants produce
fluent-but-wrong output when missed. Dense Turbo is a **fork and port** of that codebase:
MoE router/expert paths out, dense SwiGLU + 60-layer streaming + speculative decoding in.

That decision is the single largest risk reduction in this plan. It carries over a verified
Gemma 4 tokenizer, a sampler with the correct (non-HuggingFace) truncation order, a KV ring
cache, an overlapped Win32 streamer with slot pinning, a CPU FP32 reference oracle, a
streaming HF converter, an HTTP/OpenAI server, and 15 GPU kernels already diffed against CPU.

### Ground truth measured on the target machine (not assumed)

| Property | Spec assumes | **Measured on this device** |
|---|---|---|
| System | Legion Go S | Lenovo **83N6** ✔ Legion Go S |
| CPU | Ryzen Z1 Extreme, 8C/16T | ✔ confirmed |
| RAM | **32 GB** LPDDR5X | **23.8 GB usable** ✘ |
| iGPU | Radeon 780M | ✔ AMD driver `32.0.23033.1002` (2026-04-29) |
| Vulkan | 1.3.x | ✔ ICD `amdvlk64.dll`, **api_version 1.4.344** |
| NVMe | 5.5–7.0 GB/s | Samsung `MZAL81T0HDLB` 1 TB PCIe 4.0, 334.9 GB free |
| OS | Win11 + Agility SDK | ✔ build 26200 |
| MSVC / Visual Studio | (WinUI 3 requires it) | **not installed** ✘ |
| Vulkan SDK | (glslc/glslang) | **not installed** ✘ |
| Toolchain | — | `C:\w64devkit\bin\g++.exe` ✔; cmake+ninja in sibling `.venv\Scripts` |

### Four decisions taken (confirmed with the user)

1. **Fork & port** from `<sibling engine>`, not greenfield.
2. **Vulkan 1.3 backend, kernels stay in HLSL**, compiled to SPIR-V by DXC
   (`-spirv -fspv-target-env=vulkan1.3`). Meets §5's Vulkan mandate without rewriting
   CPU-verified kernels into a second language or bootstrapping the Vulkan SDK.
3. **Embedded HTTP server + web GUI** in place of WinUI 3. WinUI 3 is unbuildable with the
   installed toolchain; the sibling project's server already implements §7's surface.
4. **Tier table is derived from measured hardware**, not hardcoded from §4.

### Three spec defects this plan resolves rather than inherits

These are stated up front because each one silently breaks an Acceptance Gate if implemented
as written.

**(a) The Tier 1 budget does not close at 14 pinned layers.**
Derived from the §3.1 geometry (`d_model` 5376, `d_ff` 21504, 42/16 heads, INT4 group-64 +
BF16 scale/zero): **239,984,640 B (228.9 MiB) per layer**, 14.40 GB for 60 layers,
774 MB for the 256k×5376 embedding table. KV at 8k context (45 SWA×512 + 15 global×8192,
16 KV heads × 128) is 8,192 B/token/layer → **1,195 MB at FP16**.

| Component | §4 as written (14 pinned, FP16 KV) | Derived (8 pinned, INT8 KV) |
|---|---:|---:|
| Embedding table (resident) | 774 MB | 774 MB |
| Pinned 31B layers | 3,360 MB | 1,920 MB |
| E2B draft model (4-bit) | 1,200 MB | 1,200 MB |
| Stream ring 4 × 245 MB | 980 MB | 980 MB |
| KV cache @ 8k | 1,195 MB | 610 MB |
| Activations (M=6 batch) | 120 MB | 120 MB |
| Runtime + GUI + heap overhead | 250 MB | 250 MB |
| **Total vs 6,000 MB gate** | **7,879 MB — fails by 1,879** | **5,854 MB — passes, 146 MB slack** |

The 6.0 GB ceiling is a real product constraint and is kept. **Pinned-layer count becomes an
output of a budget solver, not an input**, and INT8 KV is the lever that buys layers back.

**(b) `FILE_FLAG_NO_BUFFERING` is in direct tension with the Throughput Gate.**
Speculative decoding yields `(1 − α^(K+1))/(1 − α)` tokens per verification pass — at the
spec's α = 0.784, K = 6 that is **3.79 tokens/pass**, not 6. Tier 1 with 52 streamed layers
moves 12.48 GB per pass:

- Unbuffered at 6.0 GB/s → 2.08 s/pass → **1.82 TPS — fails the ≥2.50 gate**
- Buffered, ~60 % page-cache hit → ~1.13 s/pass → **~3.35 TPS — passes**

On a 23.8 GB machine the 11–12.5 GB streamed working set **nearly fits in the OS page cache**.
The sibling project measured exactly this: removing `FILE_FLAG_NO_BUFFERING` took expert I/O
from 237 ms to 87 ms. The spec mandates the flag that destroys the mechanism the throughput
target depends on. This plan implements **both paths**, measures them interleaved, and ships
whichever wins — reporting the unbuffered figure as a §10 diagnostic, not as the hot path.

**(c) The Numerical Parity Gate is unverifiable as written.** An in-memory FP16 31B baseline
is ~62 GB and cannot run on this machine. Redefined in Phase 6 (§6.3) around oracles that
actually exist.

### Additional spec corrections

- **`subgroupMatrixMultiplyAccumulate` is not the Vulkan spelling.** RDNA 3 WMMA is reached
  via `VK_KHR_cooperative_matrix` / `GL_KHR_cooperative_matrix` (`coopMatMulAdd`), or
  `SPV_KHR_cooperative_matrix` from DXC. **There is no INT4 cooperative-matrix component
  type** — 4-bit weights must dequantize to FP16 in LDS before the WMMA op. "Fused 4-bit
  dequant + WMMA" means fused-in-one-kernel, not native INT4 matrix cores.
- **Wave32 is not the default.** The sibling project's `CLAUDE.md`: *"RDNA 3 runs a 256-thread
  group as Wave64, not Wave32."* WMMA requires Wave32, so it must be **forced** via
  `VK_EXT_subgroup_size_control` with `requiredSubgroupSize = 32` and a pipeline created with
  `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`. A hardcoded assumption here
  produced output scaled by exactly `1/sqrt(2)` last time.
- **M = 6 batch GEMM is below the 16×16×16 WMMA tile.** Padding M 6→16 wastes 62 % of the
  tile. The win is **bandwidth amortization** (one weight read serves 6 rows), not ALU, and
  the plan should say so rather than expecting a 6× speedup.
- **Context 8,192 needs an online-softmax attention rewrite.** The sibling `Attention.hlsl`
  stages scores in 32 KB groupshared (`ATTN_MAX_SPAN = 4096`). 8k requires a flash-style
  kernel that never materializes the full span. Explicit task, not an afterthought.
- **Tier 4 (22.0 GB) is infeasible on 23.8 GB of RAM** and is marked *32 GB hardware only*.
  Tier 3 (16.0 GB) is contingent on the Phase 0 heap probe — the sibling project measured the
  GPU-visible shared-memory budget on a 24 GB box at **~12.2 GB**, below Tier 3's requirement.

---

## 2. Execution model for Antigravity 2.0

### 2.1 Anti-fabrication contract — read before writing any code

The sibling project's entire Stage 0 was *"remove every fabricating path"*: canned replies,
prompt echo, synthesized vocabulary, a `cs_5_0` shader downgrade that `#define`d wave
intrinsics to identity, invented telemetry, and a repacker that wrote zero-filled bundles.
Every one of those made the system *appear* to work. These rules are binding on every
sub-agent:

1. **No fallbacks that change numerics.** A missing shader compiler, missing weights, missing
   tokenizer, or an unsupported Vulkan feature is a **hard error with a diagnostic**, never a
   downgrade path.
2. **No synthetic data outside `tests/` and `tools/make_synthetic_model.py`.** Never
   zero-fill, never placeholder-generate a weight file.
3. **No invented telemetry.** A number the engine cannot measure is reported as `null` and
   rendered as `—`. Never interpolate, never estimate.
4. **A task is complete only when its verification command exits 0** on this machine, with
   the output pasted into the task's completion note. "Implemented" without a passing command
   is **not** complete.
5. **Report blocked work as blocked.** If a gate fails, say which and why. Do not narrow scope
   silently, and do not mark a phase done with a failing gate inside it.

### 2.2 File ownership map — no two agents write the same file

Contracts in `include/g4dense/` are **frozen at Gate 1**. A sub-agent that needs a contract
change files it as a change request to the orchestrator; it does not edit the header.

| Owner | Writes |
|---|---|
| **Orchestrator** (Phases 0–2, 4–6) | `include/g4dense/**`, `src/format.cpp`, `src/tokenizer.cpp`, `src/sampling.cpp`, `src/detokenizer.cpp`, `src/http.cpp`, `src/cpu_reference.cpp`, `src/c_api.cpp`, `src/main.cpp`, `CMakeLists.txt`, `docs/**` |
| **SA1 `storage-engine`** | `src/streamer.cpp`, `src/layer_cache.cpp`, `src/manifest.cpp`, `src/dstorage_backend.cpp`, `tests/test_streamer.cpp`, `tests/test_format.cpp` |
| **SA2 `vulkan-compute`** | `src/vk_context.cpp`, `src/vk_pipeline.cpp`, `src/spirv_cache.cpp`, `shaders/**`, `tests/test_gpu_kernels.cpp`, `tests/test_vk_device.cpp` |
| **SA3 `speculative-coordinator`** | `src/runner.cpp`, `src/draft_runtime.cpp`, `src/speculator.cpp`, `src/kv_cache.cpp`, `src/avx512_gemm.cpp`, `tests/test_speculative.cpp`, `tests/test_kv_cache.cpp` |
| **SA4 `model-downloader`** | `tools/**`, `src/hf_client.cpp`, `tests/test_hf_client.cpp` |
| **SA5 `gui`** | `gui/**`, `src/server.cpp`, `src/openai_api.cpp`, `src/telemetry.cpp`, `tests/test_server.cpp`, `tests/test_openai.cpp` |

### 2.3 Source of truth for porting

For every ported file, the sibling path is the reference. Read it before writing; its comments
encode measured findings, not opinions.

| New file | Port from | Change required |
|---|---|---|
| `src/tokenizer.cpp` | `Turbo/src/tokenizer.cpp` | none — Gemma 4 `<\|turn>`/`<turn\|>` markers already correct |
| `src/sampling.cpp` | `Turbo/src/sampling.cpp` | none — Top-P → Top-K → temperature order is load-bearing |
| `src/detokenizer.cpp` | `Turbo/src/detokenizer.cpp` | none |
| `src/http.cpp`, `json.hpp` | `Turbo/src/http.cpp`, `include/gturbo/json.hpp` | none |
| `src/kv_cache.cpp` | `Turbo/src/kv_cache.cpp` | D3D12→Vulkan buffers; add K+1 speculative rollback slots; INT8 option |
| `src/streamer.cpp` | `Turbo/src/streamer.cpp` | expert-slot pool → **layer**-slot pool; keep pin/evict/plan-fetch-release split verbatim |
| `src/cpu_reference.cpp` | `Turbo/src/cpu_reference.cpp` | MoE FFN → SwiGLU; add p-RoPE + 3:1 hybrid attention |
| `src/server.cpp`, `openai_api.cpp` | `Turbo/src/server.cpp`, `openai_api.cpp` | expert telemetry → layer/tier/speculative telemetry |
| `gui/**` | `Turbo/gui/**` | add RAM slider, HF downloader panel, acceptance-rate readout |
| `tools/convert_hf_to_g4dense.py` | `Turbo/tools/convert_hf_to_gturbo.py` | expert scatter → dense layer scatter; `.g4dense` v2 container |
| `shaders/*.hlsl` | `Turbo/shaders/*.hlsl` | drop `RouterTopK`/`MoEReduce`/`SharedExpert`/`ScaleAccum`; add batched M=6 variants + WMMA; SPIR-V-clean bindings |
| `CMakeLists.txt` | `Turbo/CMakeLists.txt` | d3d12/dxgi → vulkan-1; add SPIR-V build step |

---

## Phase 0 — Ground Truth & Feasibility

**No feature code in this phase.** Every downstream number depends on these measurements.

### T0.1 — Toolchain bootstrap
`tools/bootstrap.py`, ported from `Turbo/tools/download_toolchain.py` (its
`_w64devkit_asset` comment documents a real clean-machine failure — keep it).

Installs/verifies: w64devkit at `C:\w64devkit`; cmake + ninja (reuse
`<sibling engine>\.venv\Scripts` or install into a local venv); DXC
(`dxcompiler.dll`, `dxil.dll`) into `build/`; Vulkan headers (`Vulkan-Headers` repo — the
loader `vulkan-1.dll` 1.4.341 is already in System32, the SDK is not needed).

**DoD:** `python tools/bootstrap.py --verify` exits 0 and prints each tool's version.
**Verify:** `dxc.exe -T cs_6_6 -spirv -fspv-target-env=vulkan1.3 -E main tests/fixtures/probe.hlsl` produces valid SPIR-V.

### T0.2 — APU capability probe  ← **highest-value task in the plan**
`tools/probe_apu/` → standalone `probe_apu.exe`, writes `docs/GROUND_TRUTH.md` +
`build/ground_truth.json`.

Measures, on this machine:
- **Vulkan heaps:** every `VkMemoryHeap`/`VkMemoryType`, with the actual byte size of the
  `HOST_VISIBLE|HOST_COHERENT|DEVICE_LOCAL` heap versus the plain `HOST_VISIBLE` heap.
  *This single number decides whether Tiers 3 and 4 are reachable.* If the
  DEVICE_LOCAL|HOST_VISIBLE heap is the small BAR carve-out, record the largest heap the GPU
  can still read and plan cold layers there.
- **Subgroup control:** `VkPhysicalDeviceSubgroupSizeControlProperties`
  (min/max subgroup size, `requiredSubgroupSizeStages`). Confirm a Wave32 pipeline can be
  created and that it actually reports 32 at runtime.
- **Cooperative matrix:** `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` — every
  supported (M, N, K, A/B/C/result type, scope) tuple. Record verbatim. If the extension is
  absent, WMMA is off the table and the batched path falls back to a **measured, documented**
  scalar GEMM — not a silent one.
- **Storage:** sustained sequential read of a 4 GB file, three ways —
  (a) buffered `ReadFile`, (b) `FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED` at 4 KB
  alignment, (c) DirectStorage 1.2 if the redist resolves. Cold and warm cache, queue depths
  1/4/8/16. Report GB/s for each.
- **UMA zero-copy:** allocate a 245 MB host-visible buffer, `ReadFile` straight into the
  mapped pointer, dispatch a compute shader that checksums it, verify the GPU sees the bytes
  with no staging copy and no explicit flush.
- **DXGI shared-memory budget** for cross-checking against the sibling project's ~12.2 GB.

**DoD:** `docs/GROUND_TRUTH.md` exists with every number filled in and no `TBD`.
**Verify:** `.\build\probe_apu.exe --all --json build/ground_truth.json` exits 0.

### T0.3 — Model source resolution
Probe Hugging Face for what actually exists. `google/gemma-4-31B-it-qat-w4a16` is **assumed
by the spec, not verified** — resolve it, or the best available substitute, and record the
real repo id, revision SHA, file list, total bytes, tensor names, shapes, dtypes, and
quantization scheme. Same for the E2B/E4B draft. Pin a revision; the sibling project pins
`mlx-community/gemma-4-26b-a4b-it-4bit @ 0d77464e` for exactly this reason.

**DoD:** `docs/MODEL_SOURCES.md` records repo + pinned revision + verified byte counts, and
states which of {QAT W4A16, AWQ, MLX affine g64, GGUF Q4_K_M} the converter must accept.
**Verify:** `python tools/resolve_model.py --check` exits 0 and byte counts match the Hub API.

### T0.4 — Budget solver & tier derivation
`tools/plan_tiers.py` consumes `ground_truth.json` + `MODEL_SOURCES.md` geometry and emits
`config/tiers.json` plus a regenerated §10 gate table.

Computes per-layer bytes, embedding bytes, KV bytes at each (context, dtype), then for each
RAM ceiling solves for max pinned layers subject to the ceiling **and** the measured heap
budget. Projects TPS as `tokens_per_pass / max(stream_time, compute_time)` with
`tokens_per_pass = (1 − α^(K+1))/(1 − α)` — parameterized on α, not hardcoded to 0.784.

**DoD:** `config/tiers.json` with, per tier: RAM ceiling, pinned layer count, pinned layer
ids (per the §4 priority order), streamed bytes, KV dtype, projected TPS at α ∈ {0.6, 0.7,
0.78, 0.85}, and a `feasible: true|false` flag with a reason string.
**Verify:** `python tools/plan_tiers.py --self-test` exits 0; every tier's component sum is
under its ceiling.

### ▣ GATE 0 — Feasibility sign-off  *(blocking)*
- [ ] `docs/GROUND_TRUTH.md` complete, no `TBD`
- [ ] Vulkan heap budget recorded; Tier 3/4 feasibility decided **on evidence**
- [ ] Cooperative-matrix support confirmed or definitively ruled out
- [ ] Buffered vs unbuffered NVMe rates measured; the §5.2 mandate re-litigated against data
- [ ] `config/tiers.json` generated; Tier 1 closes under 6,000 MB
- [ ] Model sources resolved and pinned
- [ ] **Any §10 gate the measurements invalidate is renegotiated here, in writing, not later**

---

## Phase 1 — Contracts & Skeleton

Single-threaded, orchestrator only. No sub-agents until Gate 1 passes.

### T1.1 — Fork and rename
Copy the reusable subset of `<sibling engine>` into `Dense Turbo`. Rename
`gturbo::` → `g4dense::`, `include/gturbo/` → `include/g4dense/`. **Do not copy**: `build/`,
`.venv/`, `gemma-4-26b-a4b.gturbo/`, `turbo-fieldfare-main/`, `packed_experts.*`,
`resident_index.*` (superseded by the `.g4dense` v2 header), the MoE shaders. Carry `LICENSE`,
`NOTICE`, and `THIRD_PARTY_NOTICES.md` forward — the upstream Apache-2.0 §4(b) notice of
modification still applies. `git init`; `.gitignore` must exclude weights, bundles, and build
output before the first `git add`.

### T1.2 — `.g4dense` v2 container
`include/g4dense/format.hpp` + `src/format.cpp` + `docs/G4DENSE_FORMAT.md`, implementing §6.

Header 4096 B: magic `0x4734444E`, version 2, `QuantType`, dimensions, `uint64 offsets[60]`,
`uint64 sizes[60]`, embedding + LM-head offset/size, **plus** fields §6 omits that the engine
needs: `sliding_window`, `global_layer_mask` (bitmask), `rope_theta_local`/`rope_theta_global`,
`rope_scaling`, `quant_group_size`, `scale_dtype`, `tied_embeddings` flag, `vocab_size`,
`sha256[32]` of the payload, and a `reserved` tail padding to 4096.

**Alignment:** §6 specifies 4096 B. The sibling project uses 16 KB
(`GTurboFormatV1::ALIGNMENT_BYTES`) explicitly so unbuffered DMA lands on sector boundaries.
Set `ALIGNMENT_BYTES = 4096` per spec but make it a single named constant, and have T0.2's
storage probe report whether 4 KB vs 16 KB alignment changes measured throughput. Cost of
16 KB is ~1 MB of padding across the file.

**DoD:** round-trips a synthetic container; rejects truncated/misaligned/bad-magic files with
distinct errors. Reuse `checked_add`/`checked_multiply`/`PathValidator` from the sibling
`format.hpp` verbatim.

### T1.3 — Freeze interface contracts
Write every header in `include/g4dense/` with complete declarations and doc comments, and
make each compile standalone. These are the boundaries the five sub-agents build against:

`format.hpp`, `manifest.hpp`, `vk_context.hpp`, `vk_pipeline.hpp`, `streamer.hpp`,
`layer_cache.hpp`, `kv_cache.hpp`, `draft_runtime.hpp`, `speculator.hpp`, `runner.hpp`,
`cpu_reference.hpp`, `tokenizer.hpp`, `sampling.hpp`, `detokenizer.hpp`, `http.hpp`,
`json.hpp`, `openai_api.hpp`, `server.hpp`, `hf_client.hpp`, `telemetry.hpp`, `c_api.h`.

Carry these interface lessons across from the sibling headers:
- `ExpertStreamer::plan_experts` / `fetch_misses` / `release_plan` — the **plan/fetch/release
  split is what makes hit-first dispatch possible**. Keep it as `plan_layers` / `fetch_misses`
  / `release_plan`, including slot **pinning** (without it a plan evicts its own entries and
  the caller binds the wrong layer's weights — fluent, wrong output).
- `ComputePipelineManager::initialize_pipelines(descriptor_capacity)` takes capacity as a
  **required argument** so a resolve-order dependency is enforced by the compiler. The Vulkan
  equivalent is descriptor-pool sizing: same rule.
- `KVCacheManager` owns all K/V buffers and `physical_slot(layer, pos)`; `ForwardRunner` holds
  none of its own.
- `KernelDispatchParams::set()` zeroes unset constants so a kernel never reads a stale value.

**DoD:** `tests/test_contracts.cpp` includes every header alone and links; CI enforces it.

### T1.4 — Build system
`CMakeLists.txt` from the sibling's, with: `d3d12/dxgi/d3dcompiler` → `vulkan-1`; a
**build-time** SPIR-V compilation step (DXC per `shaders/*.hlsl` → `build/shaders/*.spv`);
`shaders/` and `gui/` copied on every **build**, not at configure time (the sibling comment
explains why: a configure-only copy makes shader edits silently no-op); MinGW
`-static-libgcc -static-libstdc++` on every target; CTest registration.

**Shader hygiene, inherited:** never save a shader as UTF-8 **with BOM** — DXC rejects it with
a misleading "non-ASCII characters" error at the wrong line. PowerShell's
`Set-Content -Encoding utf8` adds one silently; use `-Encoding ascii`. Same trap applies to
GUI assets, which must be UTF-8 **without** a cp1252 round-trip.

### T1.5 — Synthetic tiny model  ← **unblocks all five sub-agents**
`tools/make_synthetic_model.py` emits a real, structurally valid `.g4dense` with 4 layers,
`d_model` 256, `d_ff` 512, 4/2 heads, vocab 1024 — a few MB. Deterministic from a seed, with
a matching NumPy reference forward pass in `tests/fixtures/`.

Every sub-agent tests against this. **No sub-agent may require the 15 GB checkpoint to make
progress.** This is what keeps five agents working in parallel.

**DoD:** `python tools/make_synthetic_model.py --out tests/fixtures/tiny.g4dense --seed 42`
plus `--verify` round-trip; the NumPy reference emits fixed logits for a fixed input.

### ▣ GATE 1 — Skeleton  *(blocking)*
- [ ] `cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe` configures
- [ ] `cmake --build build` succeeds; every `shaders/*.hlsl` compiles to `.spv`
- [ ] `ctest --test-dir build --output-on-failure` passes (`test_contracts`, `test_format`,
      `test_tokenizer`, `test_sampling`, `test_detokenizer`)
- [ ] `tests/fixtures/tiny.g4dense` loads and round-trips
- [ ] **All `include/g4dense/**` headers frozen.** Changes now require a change request.

---

## Phase 2 — Weights Pipeline & Numerical Oracle

Two tracks in parallel. Both block Phase 3's GPU work.

### Track A — Offline converter (critical path)

**T2.1** `tools/convert_hf_to_g4dense.py`, ported from `Turbo/tools/convert_hf_to_gturbo.py`.
Preserve its **streaming scatter**: source tensors read in fixed chunks, written straight to
output offsets, peak memory a few MB regardless of a 15 GB input. Replace expert scatter with
dense per-layer packing into the §6 layout. Accept the source format T0.3 resolved.

**T2.2** `tools/test_convert_streaming.py`, ported. Guards the scatter offsets against
synthetic data — the sibling notes a wrong offset *"would silently corrupt 12 GB of weights"*,
and that its `expert_stride` was hardcoded wrong in several places, reading a completely
different tensor while still producing fluent text. **Same failure mode exists here per-layer.**

**T2.3** Convert the real 31B checkpoint and the E2B draft. Verify every layer offset is
4096-aligned, sizes sum to file length, SHA-256 matches the header.

**DoD:** `gemma-4-31b.g4dense` exists; `python tools/verify_bundle.py <path>` exits 0.

### Track B — CPU FP32 reference oracle  ← **the ground truth for everything after**

**T2.4** `src/cpu_reference.cpp`, ported from the sibling's 28 KB scalar FP32 forward pass.
Touches no GPU. Changes: MoE FFN → **SwiGLU** (`silu(gate) * up`, note Gemma 4 26B used
`gelu_tanh`; confirm the 31B activation from `config.json` in T0.3); routers/experts removed;
**3:1 hybrid attention** — 45 sliding-window layers at W=512, 15 global layers with p-RoPE.

**Forward-pass constants — verify each against the checkpoint, do not infer.** The sibling
lists these as *"produces fluent-but-wrong output if missed, the hardest failure to debug"*:
attention scale (1.0 there, absorbed into `q_norm` — **re-derive for 31B**); RMSNorm applies
`w`, not `1+w`, eps 1e-6; dequant `w = q*scale + bias`, packed low-nibble-first; embedding
scaled by `sqrt(d_model)`; tied LM head; partial-rotary fraction per layer class; **Top-P →
Top-K → temperature** sampling order.

**T2.5** `--dump-tensors <dir>` writes per-stage FP32 tensors for token 0 (`embed`,
`layerN_hidden`, `final_norm`, `logits`) so Phase 3 can diff each kernel individually. This is
the mechanism that let the sibling verify 15 kernels; it is not optional.

**T2.6** Run the reference on the real 31B bundle. It must produce **coherent Gemma 4 text**.
~0.1 tok/s is expected and fine — its job is to be right, not fast.

### ▣ GATE 2 — Oracle  *(blocking for SA2 + SA3)*
- [ ] `.\build\turbo-dense.exe --cpu --prompt "What is the capital of France?"` produces
      correct, coherent output
- [ ] `--dump-tensors` writes per-stage tensors for token 0
- [ ] Reference matches the NumPy tiny-model reference to `allclose(1e-5, 1e-4)`
- [ ] Every forward-pass constant in §T2.4 is **verified against the checkpoint** and recorded
      in `docs/FORWARD_PASS.md` with its source

---

## Phase 3 — Sub-Agent Parallel Implementation

Five agents, disjoint file sets (§2.2), each gated independently against the tiny model.

### SA1 — `storage-engine`

| Task | Description |
|---|---|
| **T3.1.1** | `src/manifest.cpp` — `.g4dense` v2 header parse/validate; reject bad magic, version, misalignment, overflow, size mismatch with distinct errors |
| **T3.1.2** | `src/streamer.cpp` — port the sibling `ExpertStreamer` to a **layer** streamer. Keep verbatim: persistent per-slot `HANDLE event` + `OVERLAPPED` (creating them per read cost 240 kernel-object pairs/token), plan/fetch/release, **slot pinning**, LFU/LRU eviction, skip-read-on-hit (`load_expert` used to read unconditionally, so a 51.6 % hit rate still re-read every byte) |
| **T3.1.3** | Dual I/O backend behind one interface: buffered `ReadFile` and `FILE_FLAG_NO_BUFFERING\|FILE_FLAG_OVERLAPPED` at 4096 alignment, selectable at runtime (`--io-mode buffered\|unbuffered\|auto`) |
| **T3.1.4** | `src/dstorage_backend.cpp` — DirectStorage 1.2 as a **third optional** backend (app-local redist; `dstorage.dll` is not in System32). Must degrade to Win32 IOCP with a logged reason, never silently |
| **T3.1.5** | Depth-3 prefetch pipeline per §5.2: layer *i* computing, *i+1* in flight, *i+2* queued. Reads land **directly in the mapped host pointer** from SA2's pool — zero staging |
| **T3.1.6** | Tiered pinning: consume `config/tiers.json`; pin in the §4 priority order (0–6, then 53–59, then global-attention layers, then middle FFN); support live re-tiering on the GUI slider without a reload |

**Gate SA1:** `tests/test_streamer.cpp` passes — eviction, pinning, plan-isolation (one plan
must not release another's slots), hit/miss accounting, and a **read-correctness** test
(fetched bytes equal file bytes). `probe_apu --io-bench` numbers reproduce inside the engine
within 10 %. Zero-copy asserted: the GPU checksums the fetched buffer without a copy.

### SA2 — `vulkan-compute`

| Task | Description |
|---|---|
| **T3.2.1** | `src/vk_context.cpp` — instance (Vulkan 1.3), physical-device select (prefer the 780M), queue setup, **`VK_EXT_subgroup_size_control` enabled and `requiredSubgroupSize = 32` asserted at runtime**. Feature/extension absence is a hard error naming the missing feature |
| **T3.2.2** | UMA memory pool. Enumerate heaps; allocate from `HOST_VISIBLE\|HOST_COHERENT\|DEVICE_LOCAL` per §5.1; if T0.2 shows that heap too small, fall back to the largest GPU-readable `HOST_VISIBLE` heap **with a logged, telemetry-visible reason**. Circular pool of 4 layer buffers (size from `tiers.json`, ~245 MB), persistently mapped, handed to SA1 as raw pointers |
| **T3.2.3** | `src/spirv_cache.cpp` — load `build/shaders/*.spv`, cache modules, strip a leading BOM defensively. **No runtime HLSL fallback** |
| **T3.2.4** | `src/vk_pipeline.cpp` — descriptor set layouts, pools sized from an explicit capacity argument (the sibling's `descriptors_for()` lesson: a hardcoded 65,536 threw **mid-generation**), push constants replacing the 16 root constants, pipeline barriers replacing UAV barriers, and a **binding-set cache keyed on the exact resource set with strong references** (raw-pointer keys served stale descriptors after allocator reuse) |
| **T3.2.5** | Port the 11 surviving kernels to SPIR-V-clean HLSL: `EmbedLookup`, `RMSNormK`, `GemvInt4`, `QKVEpilogue`, `Attention`, `SwiGLU`, `PostAttn`, `LayerTail`, `Softcap`, `LMHeadGreedy`, `ArgmaxReduce`. Keep **raw byte-address buffers with explicit offsets** — the sibling's mixed `StructuredBuffer<float16_t>`/`Texture2D<uint>` declarations silently disagreed with stride-4 host bindings |
| **T3.2.6** | Batched M∈[2,8] variants (`GemmInt4Batch`) for speculative verification. **`WaveGetLaneCount()`, never a hardcoded 32** — a fixed stride double-counted groups on Wave64 and produced a *lower* argmax rather than an obviously broken one |
| **T3.2.7** | WMMA path via `SPV_KHR_cooperative_matrix`, gated on the T0.2 tuple list. Dequant INT4 → FP16 into LDS, then `coopMatMulAdd` on 16×16×16 tiles with M padded 6→16. **A/B against T3.2.6 and keep whichever measures faster** — the win is bandwidth amortization, not ALU, and may be small |
| **T3.2.8** | Online-softmax (flash-style) `Attention` so 8,192 context does not stage the full score span in 32 KB groupshared |

**Gate SA2:** `tests/test_gpu_kernels.cpp` verifies **every** kernel against `cpu_reference`
using `allclose(atol=1e-5, rtol=1e-4)` — not a pure relative test (GPU transcendentals differ
by a couple of ULPs and near-zero results then show huge relative error for ~1e-7 absolute,
while a real logic bug is never marginal: the Wave64 bug was off by 0.29 relative,
everywhere). `test_vk_device` asserts subgroup size 32 and prints the heap table.

### SA3 — `speculative-coordinator`

| Task | Description |
|---|---|
| **T3.3.1** | `src/avx512_gemm.cpp` — Zen 4 AVX-512 FMA/VNNI INT4 GEMV/GEMM. **Runtime CPUID dispatch** with an AVX2 path; `-mavx512f -mavx512bw -mavx512vnni` on that TU only, so the binary still starts on non-AVX-512 hardware |
| **T3.3.2** | `src/draft_runtime.cpp` — Gemma 4 E2B autoregressive draft loop on CPU, its own small KV cache, thread pool sized to physical cores (8), not threads. Target ~40 TPS per §5 |
| **T3.3.3** | `src/kv_cache.cpp` — port `KVCacheManager`; keep `physical_slot(L,p) = p % capacity` (this is what let the sibling take context 1024 → 4096 for 560 MB instead of ~1.85 GB). Add: **K+1 speculative slots with rollback**, hybrid capacity (45 SWA layers get `sliding_window` slots regardless of context; only the 15 global layers scale), and an **INT8 KV** mode with per-head FP16 scales — the lever that closes the Tier 1 budget |
| **T3.3.4** | `src/speculator.cpp` — acceptance/rollback per §8.3. Modified rejection sampling comparing draft distribution against 31B logits; accept N ≤ K then 1 bonus token; commit accepted KV slots, roll back the rest. **Must be exact:** greedy speculative output must equal greedy non-speculative output token-for-token |
| **T3.3.5** | `src/runner.cpp` — `ForwardRunner` orchestration. Overlap draft generation with prefetch of the first streamed layers; submit pinned-layer and cache-hit work **before** issuing miss reads (this ordering was the sibling's single biggest scheduling win, 6.04 → 8.06 tok/s); minimize fence waits (~92 submissions but 31 fence waits per token there — **fence count is what costs**) |
| **T3.3.6** | Per-phase metrics: `stream_io_ms`, `gpu_wait_ms`, `draft_ms`, `verify_ms`, `lm_head_ms`, `cpu_other_ms`, acceptance rate, tokens/pass. Attribution, not guesswork |

**Gate SA3:** `test_kv_cache` covers ring wraparound, hybrid capacity, rollback correctness,
INT8 quant/dequant round-trip. `test_speculative` proves **greedy speculative == greedy
non-speculative, token-for-token, on the tiny model** — the gate that makes speculative
decoding trustworthy. Draft runtime hits ≥ 30 TPS on the E2B model.

### SA4 — `model-downloader`

| Task | Description |
|---|---|
| **T3.4.1** | `src/hf_client.cpp` — HF Hub REST client on the ported `http.cpp`: repo metadata, file listing, LFS resolution, optional `HF_TOKEN` auth |
| **T3.4.2** | Chunked parallel download, `Range: bytes=`, N concurrent streams, **resume from a partial file**, exponential backoff, cancel/pause |
| **T3.4.3** | Streaming SHA-256 verification against the Hub's declared hash, computed as bytes land — never by re-reading a 15 GB file |
| **T3.4.4** | Post-download conversion invoking the T2.1 converter, with progress events |
| **T3.4.5** | Progress/telemetry events (bytes, %, MB/s, ETA, phase) over the `telemetry.hpp` contract for SA5 |

**Gate SA4:** `test_hf_client` covers Range parsing, resume across a simulated disconnect,
SHA-256 mismatch → hard failure with the partial file kept for retry (not silently accepted),
and 404/401/429 handling. Live test: download one small real HF file end-to-end.

### SA5 — `gui`

| Task | Description |
|---|---|
| **T3.5.1** | Port `server.cpp` + `openai_api.cpp`. Keep `RequestCoordinator` FIFO serialization + `429 queue_full`, real HTTP framing (Content-Length, chunked, 1 MiB limit, 413/415), and **reject unsupported OpenAI params rather than ignoring them** (`n != 1`, `logprobs`, penalties, `tool_choice: required`) — silently accepting one is worse than a 400 because the caller believes it applied |
| **T3.5.2** | Port `gui/` and rebuild the §7 layout: downloader panel, **memory tier slider**, K selector, context selector, chat, telemetry dashboard |
| **T3.5.3** | Model Downloader panel — repo picker, progress bar, MB/s, phase, pause/cancel, SHA-256 status |
| **T3.5.4** | Memory & Hardware Tuning — RAM slider driven by `config/tiers.json`, showing live pinned/streamed layer split and the *feasible* flag; a tier the heap cannot satisfy is **disabled with a reason**, never offered and then failed |
| **T3.5.5** | Telemetry dashboard — TPS, TTFT, **acceptance %**, NVMe GB/s, RAM working set, APU package power. Any metric the platform will not expose is `null` → `—`. **No invented numbers** |
| **T3.5.6** | Chat with SSE streaming, stop button, multi-turn history |

**GUI hygiene, inherited:** all assets UTF-8 with `charset=utf-8` declared; the sibling's
`index.html` was saved through cp1252 and stored double-encoded emoji that rendered as
mojibake with no tool complaining. The sidebar must **hydrate from `GET /api/config`** — the
engine's *resolved* values, not HTML placeholders (the panel once advertised 62,000 tokens
while the engine was auto-sized to 4,096).

**Gate SA5:** `test_server` (framing, JSON, loopback) and `test_openai` (validation,
rendering, admission) pass. GUI loads, hydrates from `/api/config`, and streams a generation
end-to-end against the tiny model.

---

## Phase 4 — Integration

**T4.1** Wire all five modules through `ForwardRunner`; end-to-end generation on the tiny
model, GPU path.
**T4.2** **Token-for-token gate**: GPU greedy output == CPU-reference greedy output on the
real 31B bundle. This is the primary correctness gate; a per-kernel diff via `--dump-tensors`
localizes any failure to one kernel.
**T4.3** Enable speculative decoding on the real model; measure the true acceptance rate. If
α is materially below 0.78, re-run T0.4's projection and **renegotiate the throughput gates on
evidence** — do not quietly miss them.
**T4.4** Tier switching without a process restart; assert the working set after each switch.
**T4.5** `resolve_bundle_path()` port — validate each candidate by actually loading it, never
by an existence check (a stale placeholder bundle in `build/` cost the sibling real debugging
time), and validate a new bundle **before** releasing the old runner.

### ▣ GATE 4 — Integration
- [ ] GPU greedy == CPU greedy, token-for-token, on the real 31B bundle
- [ ] Greedy speculative == greedy non-speculative, token-for-token
- [ ] All tiers marked `feasible` load, generate, and switch cleanly
- [ ] Measured acceptance rate recorded in `docs/PERFORMANCE.md`

---

## Phase 5 — Performance

**Benchmark methodology is binding.** Two traps in the sibling's log each produced a confident
wrong conclusion:

1. **Background disk I/O invalidates everything.** A reported 40 % regression was a game
   download: 5.07 tok/s contended vs 6.03 idle on a bit-identical workload. Confirm
   `Get-Counter '\PhysicalDisk(_Total)\Current Disk Queue Length'` reads 0 first.
2. **A sequential sweep measures page-cache warmth, not your variable.** Sweeping tiers in
   order turned a real +27 % into an apparent +50 %. **Alternate A/B/A/B for ≥3 rounds and
   compare medians** — interleaved spread is ±0.05 tok/s, sequential is over 1 tok/s.

Also: **check the exit code.** A crashed run greps as an empty metric row that reads like a
slow result. And run-to-run drift on this machine exceeds most single optimizations, so
compare variants **within one session**, never across sessions.

| Task | Description |
|---|---|
| **T5.1** | I/O mode A/B: buffered vs unbuffered vs DirectStorage, per tier. **Expect buffered to win at Tier 1**, where the 11–12.5 GB streamed set nearly fits the page cache. Record both; ship the winner as default |
| **T5.2** | Prefetch depth and layer-slot pool sizing. Pools compete with the OS page cache for the same physical memory — the sibling's 64-slot config was *slower* than 44 despite a 90 % hit rate. Bigger is not free |
| **T5.3** | K sweep (4/6/8) against measured α; pick K maximizing tokens/pass ÷ pass-time, not α |
| **T5.4** | GPU: fence-count reduction, submission batching, descriptor-cache hit rate, WMMA vs scalar batched GEMM |
| **T5.5** | Overlap draft generation with layer streaming; overlap the LM head with the next draft |
| **T5.6** | Working-set reduction to close Tier 1 under 6,000 MB: INT8 KV, ring depth, embedding residency (the embedding **lookup** needs one row per token and need not be resident; the tied LM head needs the full matrix but only once per token and can stream) |

### ▣ GATE 5 — Performance
- [ ] Tier 1 ≥ 2.50 TPS **and** working set ≤ 6,000 MB, simultaneously, under 8k-context load
- [ ] Tier 3 ≥ 6.00 TPS **if** T0.2 showed the heap budget supports it; otherwise the
      shortfall is documented with the measured heap figure
- [ ] Every performance claim in `docs/PERFORMANCE.md` cites an interleaved measurement

---

## Phase 6 — Acceptance Gates & Opus 5 Validation Handoff

### 6.1 Deliverables for validation
`docs/GROUND_TRUTH.md`, `docs/FORWARD_PASS.md`, `docs/G4DENSE_FORMAT.md`,
`docs/PERFORMANCE.md`, `docs/ARCHITECTURE.md`, `config/tiers.json`, `CLAUDE.md`, plus
`bench/` scripts that reproduce every §10 number with one command each.

### 6.2 §10 gates as executable checks
`tools/acceptance.py` runs all five and emits `build/acceptance_report.json`.

| § 10 Gate | Executable check |
|---|---|
| **Memory Ceiling** | 8k-context generation in Tier 1; sample peak working set via `GetProcessMemoryInfo` every 100 ms; **peak ≤ 6,000 MB** |
| **Throughput** | Interleaved A/B/A/B ×3, idle disk verified; median Tier 1 ≥ 2.50 TPS, Tier 3 ≥ 6.00 TPS |
| **Numerical Parity** | See §6.3 — redefined |
| **Storage** | Sustained layer-fetch throughput ≥ 5.0 GB/s during streaming. **Report buffered and unbuffered separately**; if unbuffered is the slower path, that is the finding, and the gate is met by the shipped path |
| **GUI Functionality** | Scripted: pull a real checkpoint from HF with progress, verify SHA-256, convert to `.g4dense`, load, generate |

### 6.3 Numerical Parity Gate — redefined
§10's "in-memory FP16 baseline" is ~62 GB and cannot run on this machine. **Substituted with
three checks that are strictly stronger together than the unrunnable one:**

1. **Token-for-token greedy agreement**, GPU vs the CPU FP32 reference, over ≥ 200 tokens on
   ≥ 5 prompts. This is the check that actually catches kernel bugs.
2. **Per-kernel `allclose(1e-5, 1e-4)`** against the CPU reference for all 11 kernels, via
   `--dump-tensors`.
3. **Perplexity on a fixed WikiText-2 slice**, engine vs CPU FP32 reference on the same 4-bit
   weights → isolates *engine* error. Separately, compare the absolute figure against the
   checkpoint's published QAT perplexity → bounds *quantization* error. **Δ < 0.15** applies
   to check 3's first comparison.

If an FP16 baseline on external hardware becomes available, run the original §10 gate and
append the result. Do not claim it otherwise.

### 6.4 Handoff note to Claude Opus 5
Deliver, per gate: the command, its raw output, the pass/fail verdict, and — for any gate
renegotiated in Phase 0 or Phase 4 — the measurement that forced the change and the revised
threshold. **A renegotiated gate is acceptable; a silently-missed one is not.**

---

## 3. Risk Register

| # | Risk | Detection | Mitigation |
|---|---|---|---|
| R1 | Vulkan heap budget < Tier 3/4 needs | T0.2 | Cold layers in the large `HOST_VISIBLE` heap; mark infeasible tiers `feasible: false` and disable them in the GUI |
| R2 | Acceptance rate α ≪ 0.78 | T4.3 | Every tier's TPS scales with `(1−α^(K+1))/(1−α)`. Measure early — E2B-drafts-31B acceptance is measurable before the GPU path exists. Re-tune K; renegotiate gates on evidence |
| R3 | `google/gemma-4-31B-it-qat-w4a16` does not exist as specified | T0.3 | Converter accepts QAT W4A16 / AWQ / MLX affine g64 / GGUF Q4_K_M; pin whatever resolves |
| R4 | Unbuffered I/O < 5.0 GB/s, gate unreachable | T0.2 | Ship buffered; report unbuffered as a diagnostic (§6.2) |
| R5 | No `VK_KHR_cooperative_matrix` | T0.2 | Scalar batched GEMM at M=6 still amortizes weight reads ~6×; WMMA is an optimization, not a dependency |
| R6 | Wave64 assumed where Wave32 required | T3.2.1 / Gate SA2 | `requiredSubgroupSize=32` asserted at runtime; `WaveGetLaneCount()` everywhere; `test_vk_device` fails loudly |
| R7 | Tier 1 will not close under 6,000 MB | T0.4 / T5.6 | Levers in order: INT8 KV (−585 MB), fewer pinned layers (−240 MB each), 3-buffer ring (−245 MB), non-resident embeddings (−774 MB) |
| R8 | Sub-agent fabricates a stub and reports success | Every gate is a runnable command | §2.1 anti-fabrication contract; no gate passes without pasted output |
| R9 | Intermittent exit-139 crash under repeated runs (open in the sibling, unexplained) | Non-zero exit with no metrics | Capture full stdout+stderr and exit code on every bench run, never a grep of metric lines |
| R10 | 8k context exceeds the attention span cap | Gate SA2 | T3.2.8 online-softmax rewrite; until then cap context and say so |

---

## 4. Verification — end to end

```powershell
# Phase 0
python tools/bootstrap.py --verify
.\build\probe_apu.exe --all --json build/ground_truth.json
python tools/plan_tiers.py --self-test

# Build (add C:\w64devkit\bin to PATH first, or g++ cannot find 'as')
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe
cmake --build build --config Release
ctest --test-dir build --output-on-failure

# Correctness — the primary gate
.\build\turbo-dense.exe --cpu --prompt "What is the capital of France?"     # oracle
.\build\turbo-dense.exe --prompt "What is the capital of France?"           # must match
.\build\turbo-dense.exe --cpu --dump-tensors build/ref_dump --prompt "Hi"
.\build\test_gpu_kernels.exe                                                # all 11 vs CPU
.\build\test_speculative.exe                                                # spec == non-spec

# Performance — verify the disk is idle first
Get-Counter '\PhysicalDisk(_Total)\Current Disk Queue Length'
python bench/run_tiers.py --interleaved --rounds 3 --tiers 1,2,3

# Acceptance
python tools/acceptance.py --all --report build/acceptance_report.json
```

**Manual GUI check:** launch `turbo-dense.exe`, download a checkpoint from the HF panel with
live progress, watch SHA-256 verification and conversion, move the RAM slider across tiers and
confirm the pinned/streamed split updates, then generate and confirm TPS / TTFT / acceptance %
/ NVMe GB/s / working set all move with reality.

---

## 5. Summary for the Antigravity `/goal` invocation

- **Phase 0 and Phase 1 are single-threaded and blocking.** No sub-agent fans out before
  Gate 1. Phase 0 exists because four §10 gate values are currently unverified assumptions.
- **Phase 2 Track B (CPU reference) blocks SA2 and SA3.** Without the oracle, a GPU kernel bug
  is indistinguishable from a plausible activation.
- **T1.5's synthetic tiny model is what makes five parallel agents possible.** No agent should
  ever be blocked on a 15 GB download.
- **File ownership (§2.2) is strict.** Contracts freeze at Gate 1.
- **Read the sibling `CLAUDE.md` before porting any file.** Its comments record measurements,
  not preferences — several encode bugs that cost real debugging time and that reproduce
  exactly in the dense design.
