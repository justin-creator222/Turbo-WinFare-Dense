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

Rewritten at the end of round 9.

- **The GPU is now the constraint**, at 631 ms of an 883 ms token. `GemvInt4` runs at 38–48
  GB/s against 65–74 GB/s for a pure streaming copy, and unlike in rounds 6–8, closing that gap
  would now show up as throughput.
- **15 layers still stream**, but at 89–92 ms per token rather than 322. A hard memory ceiling,
  not a code problem.
- **`EmbedLookup` is still unwired** — parity-tested, ~0.1 ms of an ~883 ms pass.

## 2. Hardware baselines

From `tools/probe_apu`, re-verified independently during validation:

| Measurement | Value |
|---|---:|
| NVMe buffered read, cold | 8.19 GB/s |
| NVMe buffered read, warm | 7.14 GB/s |
| NVMe `NO_BUFFERING` overlapped, QD 4–16 | 3.08–3.09 GB/s |
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
(the smaller of heap-reported free space less a 512 MB reserve, and 38% of visible RAM -- the
`ram_cap` in `ForwardRunner::initialize`), with `G4DENSE_MAX_RESIDENT_LAYERS` to override it for
calibration.

**Caveat on the 58-layer figure above.** It may have been taken through that override, which
until round 10 applied the layer count as an index and under-imported by ~36% (see §5). If so,
the driver's real refusal point is higher than recorded here, and the ~11.75 GiB ceiling in §1
is a lower bound rather than a measurement. Re-deriving it is open work.

### External baseline — the same model, another runtime, this machine

Every other number in this document is self-measured, so the engine has had no outside
reference point. One now exists. A separate application on this machine runs Gemma 4 through
Google's **LiteRT-LM 0.15.0** on the **MLDrift GPU backend**, and benchmarked the 31B dense
alongside smaller variants on 2026-09-01. Its report is at
`C:\Users\Justin\Code\LiteRT-LM\BENCHMARKS.md`; the hardware section there matches
`tools/probe_apu` on every field that appears in both (31.3 GB RAM, Radeon 780M `0x15BF`,
Windows build 26200), so it is the same box.

| 31B dense, greedy decode | LiteRT-LM / MLDrift | this engine (round 9) |
|---|---:|---:|
| Decode throughput | 0.8 t/s | **1.129–1.136 tok/s** |
| Context | 1024 | **4096** |
| TTFT | ~8.6 s | see §5 |
| Weights on disk | 17.90 GB (`-gpu` build) | ~15 GiB (INT4 g64) |
| Free RAM at low point | 0.9 GB | ~5 GB |
| Residency strategy | none — OS pagefile | 45 / 60 pinned, 15 streamed |

**Roughly 1.4x, at 4x the context.** Read it as an order-of-magnitude check rather than a
head-to-head: the quantizations differ, MLDrift is moving 2.9 GB more weight per token, and
1024 vs 4096 context favours the other side on KV pressure. What the comparison does establish
is that the alternative strategy — load the model and let Windows page it — was measured on
this exact hardware and lands *below* planned streaming, not above it.

The same report reaches this project's premise independently, from the failure side: a naive
bandwidth estimate for the 31B (102 GB/s ÷ 17.9 GB ≈ 5.7 t/s) overpredicts the observed 0.8 t/s
by ~7x, because it assumes a residency the machine cannot provide. That is the same reasoning
behind choosing *which* 15 layers stream rather than letting the pager choose.

**One number there is worth following up.** MLDrift ran the 26B A4B with a ~17.7 GB footprint,
~14.7 GB of it weights, and stayed usable — more resident weight than this engine's ~11.75 GiB
import ceiling admits. That is a different allocation path (the WDDM shared-GPU budget, ~15.7 GB
here) than `VK_EXT_external_memory_host` pinning, and it is not yet known whether the extra
~3 GB is reachable from Vulkan. Bound the prize before spending on it: streaming is now 89–92 ms
of an 883 ms token, so removing it entirely is worth **+29%** to the 684 ms GPU-bound ceiling
(§1), and the `GemvInt4` gap is the larger lever.

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

## 5. Throughput — measured, and which constraint binds

**Measured, three runs, `--prompt "Hi" --max-tokens 14 --temp 0 --no-spec`, 45 of 60 layers
resident:**

| | round 7 | round 8 | round 9 |
|---|---:|---:|---:|
| stream I/O | 566–585 ms | 318–326 ms | **89–92 ms** |
| GPU queue | 625–631 ms | 746–752 ms | 630–633 ms |
| LM head | 38 ms | 32 ms | 24 ms |
| CPU other | 110 ms | 100 ms | **28–29 ms** |
| **throughput** | 0.667–0.677 tok/s | 0.750–0.760 tok/s | **1.129–1.136 tok/s** |

**+50% over round 8, +68% over round 7**, from one change: the streaming reads were moved to
worker threads and switched from unbuffered to buffered.

### The engine was disk-bound, and that explains three rounds of neutral results

The streamer used `FILE_FLAG_NO_BUFFERING`, chosen in round 6 because buffered overlapped reads
*complete inline* from the page cache — so `issue_reads()` blocked and nothing overlapped. That
observation was correct. The conclusion was not, because it fixed a threading problem by giving
up more than half the read bandwidth:

```
unbuffered, measured      3.08–3.09 GB/s
buffered warm, measured   7.14 GB/s

15 streamed layers x 276.3 MB  =  4.145 GB per token
                 / 3.09 GB/s   =  1,341 ms
round-8 token time             =  1,328 ms      <- the read WAS the token
```

The GPU's 748 ms fitted inside the read with room to spare. So round 8's gemv vectorisation
(12% faster kernel) and its flash-attention rewrite both measured neutral for the same reason:
**the GPU was never the constraint.** Inline completion only mattered because the read was
issued on the thread that submits GPU work; performing it on a worker makes it irrelevant and
keeps the faster path.

Note `CPU other`, 100 ms → 29 ms. That is the blocking read leaving the main thread.

### The gemv converted, as predicted

Round 8 kept the vectorised weight loads on the argument that they would pay once streaming
stopped binding. Re-measured under the new regime, three runs each:

| | GPU phase | throughput |
|---|---:|---:|
| 8 × `u32_load` | 667–672 ms | 1.104–1.110 tok/s |
| 2 × `Load4` | 630–631 ms | 1.126–1.130 tok/s |

**+1.8%** where it was worth exactly nothing before. Modest, because the GPU is now 631 ms of
an 883 ms token rather than all of it.

### Tuning

I/O workers, and prefetch depth — each slot costs ~276 MB of import ceiling, i.e. a resident
layer, so depth trades overlap against residency:

| workers | tok/s | I/O | | depth | resident | tok/s | I/O |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 0.893 | 212 ms | | 2 (3 slots) | 46 | 1.102 | 126 ms |
| 2 | 1.079 | 127–142 ms | | **3 (4 slots)** | **45** | **1.125** | **92 ms** |
| **4** | **1.130** | **85–90 ms** | | 4 (5 slots) | 44 | 1.045 | 135 ms |
| 6 | 1.139 | 84–87 ms | | | | | |
| 8 | 1.127 | 89–92 ms | | | | | |

Past 4 workers the gain is noise: `PREFETCH_DEPTH` is 3, so the queue never holds more than
three jobs and further workers idle.

### The ceiling now, and the arithmetic behind it

Full residency needs **15.06 GiB** of layer imports against a driver ceiling of ~**11.75 GiB**,
inside a 15.90 GiB heap that also holds the KV cache, LM head and streaming pool:

```
imports 11.75 + KV (FP16, 4096) 1.17 + LM head 0.79 + streaming pool 1.05  ≈  14.8 GiB
```

All three BIOS UMA settings were measured; minimum is the best of them. With streaming no longer
binding, the ceiling is the GPU-bound pass: **631 + 24 + 29 = 684 ms, or 1.46 tok/s**. Further
gemv work now converts to throughput, which it did not before.

### The projection model is superseded, not merely unmet

`config/tiers.json`'s projections mark tier 3 (45 pinned layers) **infeasible** — the
configuration the engine actually runs — and project **5.30 TPS** for it. `projected_tps` is
kept only as a record of what was assumed before anything ran. Plan against
`measured_operating_point`.

### Speculative decoding — off by default, and why

**Retraction.** The previous version of this section said `draft_k` is **8**, on the strength of
K=8 at 20.58–20.81 s against K=4's 26.06–26.45 s over 24 tokens, reasoning that "the target pass
costs ~1,330 ms against a ~65 ms draft, a 20:1 ratio, so drafting more per round is nearly
free." Two things were wrong with it. The 1,330 ms was a round-8 target pass, already superseded
by round 9's 883 ms when the paragraph was written. And **`--no-spec` was never measured on the
same prompt**, so the comparison that decides whether to speculate at all was missing. Both K=4
and K=8 were slower than not speculating. The recommendation is withdrawn.

Replaced by a 132-run campaign: 10 prompts spanning continuation entropy (rote sequence,
rigid format, explanatory prose, technical reasoning, open creative) × {`--no-spec`, K=2,4,6,8},
48 tokens greedy, two interleaved rounds each. Zero failed runs; generated text is byte-identical
across every configuration within each prompt, verified by SHA.

#### The drafter costs 6 resident layers before it drafts anything

`--no-spec` pinned to the drafter's 39 layers with `G4DENSE_MAX_RESIDENT_LAYERS`, against its
natural 45:

| prompt | nospec @45 | nospec @39 | tax |
|---|---:|---:|---:|
| primes | 34.54 s | 42.44 s | 1.23× |
| counting | 35.17 s | 43.25 s | 1.23× |
| JSON | 36.31 s | 43.98 s | 1.21× |
| bicycle | 34.59 s | 42.59 s | 1.23× |

The 1.5 GiB import reserve leaves 21 layers streaming per token instead of 15. **A flat
~23%, prompt-independent, paid at load.** That is the hurdle drafting has to clear.

#### The matrix

Seconds, mean of two interleaved rounds:

| prompt | class | nospec | K=2 | K=4 | K=6 | K=8 | best | net |
|---|---|---:|---:|---:|---:|---:|---|---:|
| primes | rote | 34.54 | 51.28 | 33.69 | 33.08 | **23.01** | K=8 | **1.50×** |
| counting | rote | 35.17 | 49.33 | 36.33 | **29.62** | 31.51 | K=6 | **1.19×** |
| JSON | format | 36.31 | 52.44 | 36.04 | **30.88** | 40.56 | K=6 | **1.18×** |
| Python fn | format | 34.52 | 50.84 | 44.09 | **37.02** | 45.85 | K=6 | 0.93× |
| quicksort | technical | 34.03 | 50.57 | **41.98** | 49.84 | 63.20 | K=4 | 0.81× |
| salt/boiling | technical | 34.51 | 52.70 | 47.98 | **45.99** | 58.93 | K=6 | 0.75× |
| bicycle | explain | 34.59 | 49.91 | **46.23** | 52.87 | 61.63 | K=4 | 0.75× |
| photosynthesis | explain | 33.86 | 54.49 | **48.82** | 56.14 | 60.57 | K=4 | 0.69× |
| lighthouse | creative | 35.11 | 57.67 | **50.56** | 55.69 | 65.07 | K=4 | 0.69× |
| coffee shop | creative | 35.13 | **50.61** | 52.27 | 55.75 | 62.44 | K=2 | 0.69× |

Suite totals:

| policy | total | vs off |
|---|---:|---:|
| **`--no-spec`** | **347.77 s** | **1.000×** |
| K=4 (best fixed K) | 437.99 s | 1.259× slower |
| K=6 | 446.85 s | 1.285× slower |
| K=8 (the old default) | 512.77 s | 1.474× slower |
| K=2 | 519.83 s | 1.495× slower |
| oracle per-prompt K | 404.72 s | 1.164× slower |
| oracle gate *and* K | 325.25 s | 0.935× |

**No value of K beats leaving speculation off**, and neither does an oracle choosing the best K
per prompt. Only an oracle that also decides *whether* to speculate wins, by 6.5% — a policy
that requires knowing acceptance before generating. Speculation is therefore **off by default**,
and `--spec` opts in; K is **6**, the best value across the prompts where opting in is rational.

#### Best K is prompt-dependent and not monotonic

Acceptance does not fall smoothly as K rises — the JSON prompt accepts 60% at K=2, 71% at K=4,
76% at K=6 and 46% at K=8, deterministically. K decides where round boundaries fall in the text,
so an unpredictable token costs one wasted draft or seven depending on where it lands. This is
why "drafting more per round is nearly free" was wrong even on its own terms.

Break-even is roughly **70% acceptance at the best K**: JSON wins at 76%, counting at 80%, but
the Python function loses at 47% and quicksort at 45%. Every conversational prompt sits at
17–57%.

#### Generation length amplifies the sign, it does not change it

| | 16 tok | 48 tok | 128 tok |
|---|---:|---:|---:|
| primes (97.6% accept) | 1.22× | 1.50× | 1.50× |
| bicycle (17.7% accept) | 0.61× | 0.56× | 0.53× |

Fixed prefill dilutes both directions, so speculation does not become safe on long generations —
it becomes more decisive either way.

#### The adaptive gate

When `--spec` is passed and acceptance turns out low, the engine stops drafting mid-generation:
below **45%** measured over at least **24 drafted tokens**, it falls back to one token per target
pass. Thresholds derived from the matrix above — at K=6, wins run down to 47.1% acceptance and
losses start at 32.2%, so break-even against the 39-layer baseline is ~40%; at K=4 it is ~45%.
Both are overridable at runtime with `G4DENSE_SPEC_GATE_MIN_ACCEPT` and
`G4DENSE_SPEC_GATE_WINDOW`.

The gate **salvages a wrong opt-in; it does not replace the launch-time decision.** The reserve
is spent at load, so its ceiling is the 39-layer no-drafting baseline: ~42.6 s on the losing
prompts against ~52.9 s ungated at K=6. Because residency is fixed at load and drafting is
per-request, even perfectly gated speculation loses on a mixed workload — ≈381.7 s across this
suite against 347.8 s for never loading the drafter.

**On reading the acceptance rate.** Its denominator matters. This campaign's rates rest on
42–147 drafts per prompt; round 9's rested on 15–21, where "46.7%" is 7 of 15. Round 7's 76.2%
was 16 of 21 from a single default-temperature sample, which round 8 recorded as an unexplained
regression. `speculative_drafted` and `speculative_accepted` are published beside the rate, and
the counters reset per generation.

#### Two bugs this campaign found

- **`G4DENSE_MAX_RESIDENT_LAYERS` under-imported by ~36%.** The planner treats the value as a
  count and spreads the streamed layers evenly through the stack; the importer compared it
  against the layer *index* and refused everything numbered above it. `=39` produced 25 resident
  layers. The natural path was unaffected (the value is `num_layers` there, so the test never
  fired), so published throughput numbers stand — but the escape hatch documented in §2 for
  calibrating the ceiling was wrong whenever it was used. Fixed.
- **`--draft-k 1` generated exactly one token.** K is the verify-batch width and the loop asks
  the drafter for K−1 tokens, so K=1 asked for none, got an empty draft and stopped. The server
  clamped its API input to a minimum of 1, so `{"draft_k": 1}` returned a one-token response.
  Both now clamp to 2.

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
