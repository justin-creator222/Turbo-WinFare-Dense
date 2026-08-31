# Performance — Turbo-WinFare Dense

**Last updated:** 2026-08-30, round 6 (speed) · **Machine:** Lenovo Legion Go S, Ryzen Z1 Extreme,
Radeon 780M, 32 GB LPDDR5X, BIOS UMA at minimum (32.06 GB visible to Windows).

## How to read this document

Every number below is either **measured** on this machine and labelled with the command that
produced it, or explicitly labelled **projected**. Nothing else appears.

The previous version of this file asserted *"Tier 1 … Sustained throughput ≥ 2.50 TPS"*, which
contradicted the project's own `config/tiers.json` (`meets_target: false`, 0.95–1.77 TPS
projected) and was supported by no measurement. It has been retracted.

---

## 1. Where the time actually goes

**Round 6 (speed). Baseline 11,104 ms per forward pass, now 1,421 ms -- 7.8x.**

Measured with `run_turbo_dense --model models/gemma-4-31b-dense.g4dense --prompt "Hi"
--max-tokens 2 --temp 0`, which prints a four-line phase breakdown. Every row below is one
change, measured on its own:

| after | stream I/O | GPU | LM head | CPU other | **total** | end-to-end |
|---|---:|---:|---:|---:|---:|---:|
| round 5 baseline | 10,064 | 968 | 51 | 21 | **11,104** | 176.9 s |
| slots to `HOST_CACHED` | 2,597 | 767 | 35 | 21 | **3,420** | 55.8 s |
| 42 of 60 layers resident | 956 | 534 | 35 | 18 | **1,544** | 24.1 s |
| unbuffered async streaming | 789 | 532 | 43 | 87 | **1,451** | 22.6 s |
| 8 rows per gemv threadgroup | 800 | 495 | 41 | 85 | **1,421** | 22.1 s |

### The correction this round makes

The previous version of this section said *"Streaming is **not** the bottleneck"*. That was
true when it was written -- the LM head was then a CPU loop costing 21 of 23.6 s -- but moving
the head to the GPU left streaming as **90%** of what remained, and the claim was never
re-measured. It is retracted.

### What actually mattered

**1. The streaming destination was uncached memory (3.9x).** `MemoryResidency::HostVisibleMapped`
asks for `HOST_VISIBLE | HOST_COHERENT`, and `find_memory_type` returns the first match --
`memoryTypes[1]`, which is write-combined. The cached type sits on the same heap and was never
chosen. Measured (`bench_gpu` section 5), reading four real layers:

| ReadFile destination | throughput |
|---|---:|
| write-combined | 1.95 GB/s |
| `HOST_CACHED` | 5.97 GB/s |

The GPU reads both at the same speed, so this was not a trade-off -- just the wrong type.
A `MemoryResidency::HostCachedMapped` residency now exists so the choice is explicit.

**2. Most layers never need copying at all (2.2x more).** On a UMA APU there is no discrete
VRAM: the weights already sit in system DRAM. `VK_EXT_external_memory_host` lets a `VkBuffer`
be backed by memory the process already owns, so the GPU reads the weights in place and the
per-token copy disappears. 42 of 60 layers are now imported once at load.

Two constraints, both found by measurement rather than assumed:

- **The driver refuses file-mapping views** (`VK_ERROR_INVALID_EXTERNAL_HANDLE`), so the pages
  cannot be the memory-mapped container. They must be private `VirtualAlloc` memory, which
  costs one copy at load (~5 s) instead of one per token.
- **Import stops at a total-device-memory ceiling**, not a per-allocation one: ~15.9 GiB here,
  which is the sum of both heaps. Anything allocated *after* the import can therefore fail, and
  the 4-slot streaming pool did -- as an uncaught exception at startup. Fixed-size allocations
  now all happen first, and the import is greedy with whatever is left.

**3. The prefetch never overlapped anything.** `fetch_misses()` issued its async reads and
awaited them in the same call, so I/O and GPU were fully serialized. Splitting it into
`issue_reads()` / `await_reads()` was not enough on its own: *buffered* overlapped reads
complete inline when the data is in the page cache, which moved 956 ms out of the I/O phase and
straight into "CPU other" without saving any of it. `FILE_FLAG_NO_BUFFERING` makes the read
genuinely asynchronous, and also stops a second copy of the same bytes competing for a machine
already at its memory ceiling.

### Hypotheses the measurements killed

Worth recording, because each would otherwise have been plausible work:

- **Submission batching is not worth doing.** 61 `vkQueueWaitIdle` drains per token sounded
  expensive. Measured (`bench_gpu` section 4): 61 submits with a full drain each cost 3.85 ms
  against 0.21 ms for one submit and one fence -- a **3.6 ms** saving on an 11,000 ms pass.
  Not implemented, and the debugging cost of losing per-layer fault attribution is not worth it.
- **Memory type barely affects GPU reads.** Write-combined, `HOST_CACHED` and `DEVICE_LOCAL`
  all land within ~10% of each other in `GemvInt4` (28-36 GB/s). Placement matters enormously
  for CPU writes and hardly at all for GPU reads.
- **`GemvInt4` was not occupancy-limited.** It ran one wave per threadgroup, which should cap
  workgroups-per-CU well below the SIMD capacity. Packing 8 rows per group changed the
  microbenchmark not at all (28-36 GB/s before and after), though it is worth a consistent
  ~37 ms on the real pass, so it was kept.

### Round 7 — prefill batched

Prefill ran one full weight pass per prompt token. `forward_batch()` now runs a chunk of
positions per pass: **532 ms per position against 1,456 ms sequential (2.74x)**, 2.96x end to
end, with identical output.

| | round 6 | round 7 |
|---|---:|---:|
| "What is the capital of France?" | 38.3 s | **17.0 s** |
| `--prompt "Hi" --max-tokens 2` | 22.8 s | **7.7 s** |

Two measurements worth keeping, because both contradict what theory predicted:

- **The batched GEMM contributes almost none of that.** `GemmInt4Batch` beats M separate
  `GemvInt4` dispatches by only **1.17x** — it is groupshared-read-bound, and raising
  rows-per-group makes it worse. The win is amortizing streaming I/O and weight reads.
- **Its first version was 0.58x — slower than what it replaced** — because each lane owned a
  contiguous 16-column block and `lane*16 mod 32` hits only two LDS banks. Consecutive lanes
  must read consecutive floats.

Activation loads are also NOT the `GemvInt4` bottleneck: vectorising them to `Load4` moved the
GPU phase from 476.7 ms to 477.8 ms. With round 6's occupancy result, that is two eliminated
hypotheses and no third.

### Round 7 — batched prefill, FP16 KV, a second architecture

| | round 6 | round 7 |
|---|---:|---:|
| "What is the capital of France?" | 38.3 s | **15.0 s** |
| `--prompt "Hi" --max-tokens 2` | 22.8 s | **6.9 s** |
| resident layers | 41 / 60 | **45 / 60** |

- **Batched prefill** replaced one weight pass per prompt token: 532 ms per position against
  1,456 ms sequential. This is nearly all of the gain.
- **FP16 KV cache** bought 4 resident layers -- not by relieving the RAM-fraction cap, but by
  lowering the *driver's own* refusal point, which counts the cache in total device memory.
- **E2B runs**, with per-layer embeddings and KV sharing: 13.2 tok/s, and as a speculative draft
  it gives **1.33x on a 24-token generation at 76.2% acceptance**. It is a *loss* on short
  outputs, where its 1.5 GiB import reserve costs the target 6 layers for too few tokens.

Three results that contradicted expectation, all measured:

- **The batched GEMM contributes almost none of the prefill win** (1.17x over M separate
  gemvs); amortizing streaming and weight reads does.
- Its first version was **0.58x -- slower than what it replaced** -- from a 16-way LDS bank
  conflict (`lane*16 mod 32` hits two banks).
- **Vectorising the gemv's activation loads did nothing** (476.7 -> 477.8 ms), eliminating the
  second hypothesis about that kernel after round 6 eliminated occupancy.

### Still on the table

- **19 layers still stream, costing ~850 ms per token.** They cannot be pinned: the driver
  stops accepting imports at ~10.5 GiB on this machine, and going past that breaks submission
  outright. Section 2 has the measurements. Reducing the KV cache from FP32 to FP16 would free
  ~1.1 GiB of the same budget (about 4 more layers), but the attention kernel reads the cache as
  FP32, so that is a shader change and a numerics change.
- **The GPU phase is 495 ms** for ~16 GB of weight reads, i.e. ~33 GB/s, against 65-74 GB/s for
  a pure streaming copy. The gap is the kernel's access pattern, and the occupancy experiment
  above says it is not simply workgroup count.
- **Prefill still runs one full weight pass per prompt token.** A 15-token prompt reads the
  whole model 15 times, which is most of the 22 s end-to-end above. Fixing it needs a batched
  INT4 GEMM; `ComputeKernel::GemmInt4Batch` is in the enum but no shader exists.
- **The embedding lookup is still on the CPU** (`EmbedLookup.hlsl` exists and is parity-tested
  but unreferenced), and `PostAttn.hlsl` / `LayerTail.hlsl` are unused fused epilogues.

## 2. Hardware baselines

From `tools/probe_apu`, re-verified independently during validation:

| Measurement | Value |
|---|---:|
| NVMe buffered read, cold | 6.52 GB/s |
| NVMe buffered read, warm | 5.68 GB/s |
| NVMe `NO_BUFFERING` overlapped, QD 4–16 | 5.73–5.75 GB/s |
| GPU read bandwidth, heap 0 (device-local) | 73.57 GB/s |
| GPU read bandwidth, heap 1 (host-visible) | 65.30 GB/s |
| Staging copy, heap 1 → heap 0 | 26.94 GB/s |

### Memory heaps — and why SAM does not apply

**These heaps change with the BIOS UMA frame buffer, but the heaps are NOT what bounds
residency.** Imported layers are pinned *system* RAM, not heap allocations, so what actually
limits them is how much Windows lets the driver pin -- and a larger UMA carve-out takes RAM away
from Windows. Measured in both directions on this machine:

| BIOS UMA | Vulkan heaps | visible RAM | resident layers | forward pass |
|---|---:|---:|---:|---:|
| **minimum** | 15.90 GiB | 31.3 GiB | **41 / 60** | **1,456 ms** |
| 8 GB | 19.90 GiB | 23.8 GiB | 26 / 60 | 2,428 ms |

**So minimum UMA is correct, and a bigger carve-out is 1.7x slower.** An earlier version of this
document recommended raising UMA to 8 GB on the reasoning that the Vulkan heap total bounds
residency. It does grow with UMA -- 15.90 to 19.90 GiB -- but it was never the binding
constraint, and that recommendation is retracted.

Crossing the pinning limit is not a soft failure. At 8 GB UMA the driver accepted 58 layers
(14.81 GiB) and then returned `VK_ERROR_OUT_OF_DEVICE_MEMORY` from **every** `vkQueueSubmit`,
and freeing the imports again did not restore it. `VK_EXT_memory_budget` cannot warn about this
either: it reported 12.69 GiB still free immediately after that import, because it does not
account for imported host memory at all. The import is therefore governed by a static budget
(the smaller of heap-reported free space less a 512 MB reserve, and 34% of visible RAM), with
`G4DENSE_MAX_RESIDENT_LAYERS` to override it for calibration.

## 3. Numerical parity — measured

`run_gpu_forward_test` against `models/gemma-4-31b-dense.g4dense`, token 2 (BOS) at position 0,
diffed against the CPU FP32 reference oracle:

| Metric | Value |
|---|---:|
| Mean absolute error | 1.885e-05 |
| Max absolute error | 3.514e-04 |
| CPU oracle argmax | token 1852 |
| GPU argmax | token 1852 |

Kernel-level parity (`run_gpu_kernels_test`): **7 of 13** kernels verified against the CPU
reference — RMSNormK, GeGLU, GemvInt4, QKVEpilogue, Softcap, ResidualAccum, EmbedLookup.
Not yet covered: Attention, PostAttn, LayerTail, LMHeadGreedy, ArgmaxReduce, GemvInt8,
GemmInt4Batch. The end-to-end oracle diff exercises those as a chain, which catches gross
errors but cannot localize one.

## 4. Memory footprint — measured

Peak working set sampled at 150 ms intervals during a real-model forward pass:

| Configuration | Peak working set |
|---|---:|
| Round 2 (CPU LM head) | 7,889 MB |
| Round 3 (GPU LM head, +756 MB resident head buffer) | ~8,855 MB |

Both exceed the 6,000 MB Tier 1 ceiling. Note these runs do not pin a tier explicitly and do
not reach 8k context, so they are not a Tier 1 measurement — they are the only real-model
figures that exist. `config/tiers.json` projects a Tier 1 footprint of 4,739.7 MB, which
remains unverified in practice.

## 5. Throughput — measured, and short of gate

At 9,614 ms per forward pass, single-token throughput is **0.104 TPS** against a 2.50 TPS gate.

`config/tiers.json` reports `meets_target: false` for **every** tier:

| Tier | Pinned | Projected TPS (α = 0.78) | Target | Meets |
|---|---:|---:|---:|:---:|
| 1 | 6 | 1.46 | 2.50 | ✘ |
| 2 | 21 | 1.97 | 3.85 | ✘ |
| 3 | 48 | 5.28 | 6.00 | ✘ |
| 4 | 60 | 9.36 | 18.0 | ✘ (infeasible: needs 15,414 MiB of a 13,414 MiB heap) |

Those projections additionally assume a batched K+1 speculative verification pass that **does
not exist** — `ComputeKernel::GemmInt4Batch` currently maps to `GemvInt4.spv`, the unbatched
kernel. Until that is built and `runner.cpp` reads `options.speculative_enabled`, the
speculative multiplier in these projections is unearned.

## 6. Benchmark methodology

Binding, and inherited from the sibling project's log where each rule cost a wrong conclusion:

- Interleave A/B/A/B for ≥3 rounds and compare medians. A sequential sweep measures page-cache
  warmth, not your variable.
- Confirm `Get-Counter '\PhysicalDisk(_Total)\Current Disk Queue Length'` reads 0 first.
  Background disk I/O once turned a clean run into an apparent 40% regression.
- **Never pipe a verification command.** `| head` reported exit 0 on a process crashing with
  exit 29, which hid a hard crash for an entire validation round.
- Compare variants within one session. Run-to-run drift on this machine exceeds most single
  optimizations.
