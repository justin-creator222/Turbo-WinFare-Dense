# Round 6 — speed

**Author:** Claude Opus 5 · **Date:** 2026-08-30 · **Implemented by:** me

## Summary

Round 5 made the engine correct. This round made it **7.8× faster**, with byte-identical output.

| | round 5 | round 6 |
|---|---:|---:|
| forward pass | 11,104 ms | **1,421 ms** |
| "What is the capital of France?" | 279.2 s | **37.7 s** |
| "…what a ring buffer is" | 507.1 s | **64.7 s** |
| "List three primary colors." | 396.2 s | **49.5 s** |

All three prompts produce exactly the same text as round 5, and `run_gpu_forward_test` is still
argmax-exact against the CPU oracle at all four positions.

Per-phase, measured with the same command throughout
(`run_turbo_dense --prompt "Hi" --max-tokens 2 --temp 0`):

| after | stream I/O | GPU | LM head | CPU other | **total** |
|---|---:|---:|---:|---:|---:|
| round 5 baseline | 10,064 | 968 | 51 | 21 | **11,104** |
| slots to `HOST_CACHED` | 2,597 | 767 | 35 | 21 | **3,420** |
| 42 of 60 layers resident | 956 | 534 | 35 | 18 | **1,544** |
| unbuffered async streaming | 789 | 532 | 43 | 87 | **1,451** |
| 8 rows per gemv threadgroup | 800 | 495 | 41 | 85 | **1,421** |

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

## The UMA recommendation was backwards — and should be revisited

Before implementation I recommended lowering the BIOS UMA frame buffer, reasoning that
Windows-visible RAM was the binding constraint. **That was wrong.** The binding constraint is
the total memory Vulkan can allocate, and it *shrinks* as UMA shrinks:

| | DEVICE_LOCAL heap | host-visible heap | 256 MB heap | **total** |
|---|---:|---:|---:|---:|
| 8 GB UMA (before) | 13.10 GiB | 6.55 GiB | 0.25 GiB | **19.90 GiB** |
| minimum UMA (now) | 10.60 GiB | 5.30 GiB | — | **15.90 GiB** |

Lowering UMA did return 7.7 GB to Windows (24,381 → 32,061 MB visible), but it cost 4.0 GiB of
the ceiling that actually decides how many layers can be resident.

The budget under that ceiling:

```
weights          15.06 GiB      LM head           0.74 GiB
KV cache (FP32)   2.19 GiB      streaming pool    1.05 GiB     total  19.04 GiB
```

At **15.90 GiB** only 42 of 60 layers fit, and the other 18 cost ~800 ms per token.
At **19.90 GiB** all 60 fit, which would remove that 800 ms — roughly **1.6 tok/s instead of
0.70**, a further 2.3×. The code already handles this: it imports greedily and releases the
streaming pool when nothing is left to stream.

**This is a BIOS question now, not a code one**, and it is the single largest remaining win.

---

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
