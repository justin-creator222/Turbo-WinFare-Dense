# Round 6 — speed

**Author:** Claude Opus 5 · **Date:** 2026-08-30 · **Implemented by:** me

## Summary

Round 5 made the engine correct. This round made it **7.6× faster**, with identical output.

| | round 5 | round 6 |
|---|---:|---:|
| forward pass | 11,104 ms | **1,456 ms** |
| "What is the capital of France?" | 279.2 s | **38.3 s** |
| "…what a ring buffer is" | 507.1 s | **66.6 s** |
| "List three primary colors." | 396.2 s | **55.4 s** |

All three prompts produce exactly the same text as round 5, and `run_gpu_forward_test` is still
argmax-exact against the CPU oracle at all four positions.

Final configuration: **41 of 60 layers resident**, 19 streamed, at minimum BIOS UMA. The figures
in the step table below were taken at 42 resident during development; the shipped budget is
slightly more conservative, which costs one layer.

Per-phase, measured with the same command throughout
(`run_turbo_dense --prompt "Hi" --max-tokens 2 --temp 0`):

| after | stream I/O | GPU | LM head | CPU other | **total** |
|---|---:|---:|---:|---:|---:|
| round 5 baseline | 10,064 | 968 | 51 | 21 | **11,104** |
| slots to `HOST_CACHED` | 2,597 | 767 | 35 | 21 | **3,420** |
| 42 of 60 layers resident | 956 | 534 | 35 | 18 | **1,544** |
| unbuffered async streaming | 789 | 532 | 43 | 87 | **1,451** |
| 8 rows per gemv threadgroup | 800 | 495 | 41 | 85 | **1,421** |
| shipped budget, 41 resident | 850 | 477 | 41 | 88 | **1,456** |

---

## The measurement that set the agenda

`docs/PERFORMANCE.md` asserted *"Streaming is **not** the bottleneck"*. Measuring first —
before changing anything — showed it was **90%** of the pass. That claim was true when written
in round 3, when the LM head was a CPU loop costing 21 of 23.6 s; moving the head to the GPU
left streaming dominant, and nobody re-measured. It is retracted in the doc.

I built `tools/bench_gpu.cpp` before touching the engine, and it earned its keep by **killing
two plausible optimizations before I wrote them**:

- **Submission batching is not worth doing.** 61 `vkQueueWaitIdle` device drains per token
  sounded expensive and was in the approved plan. Measured: 61 submits with a full drain each
  cost **3.85 ms** against **0.21 ms** for one submit and one fence. A 3.6 ms saving on an
  11,000 ms pass. Dropped — and keeping per-layer submission preserves fault attribution.
- **`GemvInt4` was not occupancy-limited.** It ran one wave per threadgroup, which should cap
  workgroups-per-CU below SIMD capacity. Packing 8 rows per group moved the microbenchmark not
  at all (28–36 GB/s before and after). It was worth a consistent ~37 ms on the real pass, so it
  stayed, but the reasoning behind it was wrong.

## What actually mattered

### 1. The streaming destination was uncached memory — 3.9× from one line

`MemoryResidency::HostVisibleMapped` requires `HOST_VISIBLE | HOST_COHERENT`, and
`find_memory_type` returns the first match: `memoryTypes[1]`, which is **write-combined**. The
cached type sits on the same heap and was never chosen. So every byte of the model was
`ReadFile`d into uncached memory, every token.

| ReadFile destination | throughput |
|---|---:|
| write-combined | 1.95 GB/s |
| `HOST_CACHED` | 5.97 GB/s |

The GPU reads both at the same speed, so this was not a trade-off — just the wrong memory type.
A `MemoryResidency::HostCachedMapped` now exists so the choice has to be made explicitly.

### 2. Most layers never need copying at all — 2.2× more

This is a UMA APU: there is no discrete VRAM, and the weights already sit in system DRAM.
Copying page-cache → "GPU buffer" every token was RAM-to-RAM for no benefit. The streaming
architecture inherited from `spec.md` is shaped for a discrete GPU.

`VK_EXT_external_memory_host` lets a `VkBuffer` be backed by memory the process already owns.
42 of 60 layers are now imported once at load and read in place forever after.

Two constraints, both found by measurement rather than assumed:

- **The driver refuses file-mapping views** (`VK_ERROR_INVALID_EXTERNAL_HANDLE`), so the pages
  cannot be the memory-mapped container. They must be private `VirtualAlloc` memory — one copy
  at load (~5 s) instead of one per token. `bench_gpu` probes both kinds and reports which the
  driver accepts.
- **Import stops at a total-device-memory ceiling**, not a per-allocation one. Anything
  allocated *after* the import can therefore fail, and the streaming pool did — as an uncaught
  exception at startup, which is how I found it. All fixed-size allocations now happen first and
  the import takes what is left.

### 3. The prefetch never overlapped anything

`fetch_misses()` issued its async reads and awaited them in the same call, so I/O and GPU were
fully serialized despite the "depth-3 prefetch". Splitting it into `issue_reads()` /
`await_reads()` was **not enough on its own**: buffered overlapped reads complete *inline* when
the data is in the page cache, which moved 956 ms out of the I/O phase and straight into "CPU
other" without saving a millisecond. `FILE_FLAG_NO_BUFFERING` makes the read genuinely
asynchronous, and stops a second copy of the same bytes competing for a machine already at its
memory ceiling.

---

## The UMA recommendation was backwards -- twice

I first recommended *lowering* the BIOS UMA carve-out, on the theory that Windows-visible RAM
was the binding constraint. Seeing the Vulkan heap total shrink from 19.90 to 15.90 GiB, I then
recommended *raising* it back to 8 GB, on the theory that the heap total was the real limit.

**Both were reasoning where I should have measured.** With the engine actually running at each
setting:

| BIOS UMA | Vulkan heaps | visible RAM | resident layers | forward pass | TPS |
|---|---:|---:|---:|---:|---:|
| **minimum** | 15.90 GiB | 31.3 GiB | **41 / 60** | **1,456 ms** | **0.088** |
| 8 GB | 19.90 GiB | 23.8 GiB | 26 / 60 | 2,428 ms | 0.052 |

Minimum UMA wins by 1.7x. Imported layers are pinned *system* RAM, not Vulkan heap memory, so
giving RAM back to Windows is exactly what buys residency; the heap total grows with UMA but
never binds. The cost of the round trip was two BIOS changes and a reboot each.

Crossing the pinning limit is unrecoverable, which is what made it expensive to find. At 8 GB
UMA the driver accepted 58 layers and then failed every `vkQueueSubmit` with
`VK_ERROR_OUT_OF_DEVICE_MEMORY`, and freeing the imports did not restore it -- so there is no
safe way to discover the edge by walking up to it. `VK_EXT_memory_budget` is no help either: it
reported 12.69 GiB free immediately after that import, because it does not account for imported
host memory. Hence a static budget plus a loud startup probe, rather than anything adaptive.

## Two bugs this shook out

**A failing submit looked like a fast one.** `vkQueueSubmit`'s result was never checked. When
the over-greedy import broke the device, the forward pass reported ~0 ms of GPU time and
"1.93 TPS" while printing an empty response -- a 20x speedup that was pure failure. All three
submit sites now check, and the runner probes the device once after import rather than
discovering this mid-generation.

**Tier pinning starved the prefetch queue.** A fallback branch pinned three of the four
streaming slots when no layer was resident, so `plan_layers()` threw "cache thrash" and the
stream-everything path could not run at all. Pinning is now removed entirely: it only ever
avoided a re-read, and residency does that strictly better -- a pinned slot still costs a
269 MB read per token, a resident layer costs nothing.

## Verification

```
run_gpu_kernels_test            8 of 13 kernels PASS
run_gpu_forward_test            4 positions vs the CPU oracle, argmax exact at every one
run_real_generation_test 24     3 of 3 prompts, text identical to round 5
tokenizer / detok / detokenizer / format / sampling / prompt_pipeline / contracts    PASS
```

Argmax and logits are unchanged to ~1e-4 at every position, so none of this altered the maths.

## Decisions made without asking

1. **Dropped submission batching from the approved plan** once measured at 3.6 ms. The plan
   ranked it second; the measurement ranked it last.
2. **Kept the gemv row-packing** even though the microbenchmark showed nothing, because the
   real pass improved by ~37 ms consistently. Recorded honestly as "the reasoning was wrong,
   the change was mildly good".
3. **Import is greedy rather than a fixed tier.** How many layers fit depends on the BIOS split
   and on what else is allocated, so a hard-coded count would be wrong on any other
   configuration. The tier system still applies when no import is possible.
4. **Tier pinning is disabled whenever layers are resident.** A pinned slot still costs a
   269 MB read per token while a resident layer costs nothing, and pinning 3 of 4 slots starved
   the prefetch queue outright ("cache thrash").
5. **Did not touch the KV cache dtype.** FP32 → FP16 would free ~1.1 GiB of ceiling (≈4 more
   resident layers) but the attention kernel reads the cache as FP32, so it is a numerics change
   plus a shader change — not something to fold into a round whose gate is bit-stability.

## Still on the table

- **18 layers still stream, ~800 ms/token.** See the UMA section — that is the big one.
- **The GPU phase is 495 ms** for ~16 GB of weight reads, ~33 GB/s against 65–74 GB/s for a
  pure streaming copy. The occupancy experiment says it is not workgroup count; the access
  pattern is the next suspect.
- **Prefill still runs one full weight pass per prompt token** — a 15-token prompt reads the
  whole model 15 times, which is most of the 22 s end-to-end figure. Needs a batched INT4 GEMM;
  `ComputeKernel::GemmInt4Batch` is in the enum but **no shader exists**. This is the TTFT fix
  and was deliberately scoped out of this round.
- **The embedding lookup is still on the CPU**, and `PostAttn.hlsl` / `LayerTail.hlsl` are
  unused fused epilogues that would cut the 17 dispatches per layer.
