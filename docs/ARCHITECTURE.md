# Architecture — Turbo-WinFare Dense

## 1. The constraint everything follows from

Turbo-WinFare Dense runs **Gemma 4 31B Dense** on an AMD APU (Lenovo Legion Go S / Ryzen Z1
Extreme / Radeon 780M / 32 GB LPDDR5X) using Vulkan 1.3 compute.

The model's weights are **15.06 GiB** in MLX affine INT4. The driver stops accepting host-memory
imports at about **11.75 GiB**, inside a 15.90 GiB heap that also holds the KV cache, the LM head
and the streaming pool. So **45 of 60 layers are resident and 15 stream from disk on every
token**, and most of the design below is a consequence of that.

Measured, greedy decode at 45/60 resident, 4096 context: **1.13 tok/s**. See
[PERFORMANCE.md](PERFORMANCE.md).

## 2. Components

```
+-------------------------------------------------------------------------+
|                        Application / client layers                      |
|  Web GUI (gui/)         CLI (run_turbo_dense.exe)                       |
|  OpenAI-compatible HTTP API      C API (include/g4dense/c_api.h)        |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                            ForwardRunner                                |
|  Tokenizer / Detokenizer (SentencePiece, 262k)                          |
|  Speculative coordinator (greedy match, or rejection sampling)          |
|  DraftRuntime -- a second ForwardRunner on the GPU, running E2B         |
+-------------------------------------------------------------------------+
         |                                           |
         v                                           v
+------------------------------------+   +--------------------------------+
|           LayerStreamer            |   |        Vulkan compute          |
|  4 slots, 3-deep prefetch queue    |   |  VulkanContext (Vulkan 1.3)    |
|  4 I/O worker threads, buffered    |   |  12 SPIR-V kernels (DXC)       |
|  LRU / LFU eviction, slot pinning  |   |  FP16 KV cache ring buffer     |
+------------------------------------+   +--------------------------------+
```

## 3. Memory

Layers are **not** heap allocations. They are pages of ordinary system RAM imported into Vulkan
through `VK_EXT_external_memory_host`, which is why residency is bounded by what the driver will
accept rather than by heap size.

Measured heaps at the minimum BIOS UMA setting, from `tools/probe_apu`:

| | size | GPU read |
|---|---:|---:|
| Heap 0, `HOST_VISIBLE` | 5.30 GiB | 85.3 GB/s |
| Heap 1, `DEVICE_LOCAL` | 10.60 GiB | 80.8 GB/s |

All three BIOS UMA settings were measured and **minimum is the best of them** — a larger
device-local heap comes at the cost of visible system RAM, and it is system RAM that bounds the
import. See PERFORMANCE.md §2.

**There is no tier pinning and no staging pipeline.** Earlier designs copied "pinned" layers
into device-local memory at tier-switch time. That was removed: a pinned slot still costs a
276 MB read per token while a resident layer costs nothing, and pinning starved the prefetch
queue badly enough to throw "cache thrash". `switch_memory_tier()` remains as an API and
adjusts nothing about residency.

## 4. Streaming

15 layers × 276.3 MB = **4.145 GB read per token**, which makes the read path a first-class
part of the forward pass rather than a detail.

* **Buffered reads on 4 worker threads.** The read is issued to a worker and the submit thread
  waits on a per-slot completion, so a read that completes inline from the page cache costs
  nothing that matters. This replaced `FILE_FLAG_NO_BUFFERING`, which was genuinely async but
  ran at 3.09 GB/s against buffered-warm's 7.14 — the engine was disk-bound and every GPU
  optimization measured neutral until this changed.
* **A 3-deep prefetch queue over 4 slots.** Each slot costs ~276 MB of import ceiling, i.e. a
  resident layer, so depth is a direct trade against residency. 3 measures best.
* **The streamed layers are evenly spaced, not the tail.** Importing greedily from layer 0 left
  layers 45–59 streaming back to back with nothing to overlap. Spreading them puts several
  resident layers' compute between consecutive reads.

## 5. Compute kernels

12 registered in `ComputeKernel` (`include/g4dense/vk_pipeline.hpp`), HLSL compiled to SPIR-V by
`tools/compile_shaders.py`. Nine are parity-tested against the CPU reference by
`run_gpu_kernels_test`.

| kernel | role |
|---|---|
| `RMSNormK` | RMSNorm, `x_hat * w` (not `1 + w`), batched |
| `GemvInt4` | affine INT4 group-64 matrix-vector; weights read via `Load4` |
| `GemmInt4Batch` | the batched form, used for prefill and speculative verification |
| `QKVEpilogue` | per-head RMSNorm + RoPE |
| `Attention` | GQA with sliding-window masking, as a single-pass online softmax |
| `KVWrite` | packs K/V into the FP16 cache |
| `GeGLU` | GeLU-tanh gated FFN, with per-operand strides for PLE models |
| `ResidualAccum` | residual add with layer scalar |
| `Softcap` | `cap * tanh(logits / cap)` |
| `LMHeadGreedy`, `ArgmaxReduce` | vocab projection and argmax |
| `EmbedLookup` | parity-tested, currently unwired (~0.1 ms of an ~883 ms pass) |

`GemvInt8`, `MulBF16` and `ScaleAccum` exist in `shaders/` but are not registered.

**Attention is a flash-style online softmax.** It walks keys in tiles of 256 carrying a running
max, denominator and accumulator, rescaling by `exp(m_old - m_new)` when a tile raises the max.
Nothing in it scales with the attended span, so context is bounded by KV cache cost rather than
by groupshared memory.

## 6. Two architectures

The engine runs both Gemma 4 variants, and the container describes which is which:

* **31B Dense** — 60 layers, `d_model` 5376, 32 Q / 16 KV heads, `head_dim` 256; global layers
  (every 6th, `[5, 11, … 59]`) use `head_dim` 512 and 4 KV heads. `attention_k_eq_v` is true,
  so V is taken from K.
* **E2B** — 35 layers, per-layer embeddings (PLE) and KV sharing across the last 20 layers, and
  a feed-forward width that **doubles at layer 15** in a way `config.json` does not state. The
  container records a measured `layer_d_ff` per layer rather than inferring a rule.

E2B is also the speculative draft model: it runs at ~15 tok/s against the 31B's ~1.13, on the
same Vulkan context.

## 7. Speculative decoding

The draft proposes `draft_k - 1` tokens and the target verifies them in **one** weight pass.
The verify batch leads with the previous round's bonus token — the one the target has not seen —
which is what keeps a round to a single target pass; leading with the first draft instead costs
a second pass and measured 1.74× *slower* than not speculating.

Rejected drafts leave stale entries past the accepted length in both KV caches. They are never
read, because attention only looks at `[first, current_position)`, and the next pass overwrites
those ring slots — so there is no rollback step.

`draft_k` is the width of the verify batch, capped at `kGemmMaxBatch` (8) and floored at 2 —
K=1 would ask the drafter for zero tokens. It defaults to **6**.

**Speculation is off unless `--spec` is passed.** The drafter is loaded once at startup and
reserves 1.5 GiB, which costs the target 6 resident layers — 21 streamed per token instead of
15, a flat ~23% before anything is drafted. Only rote and rigid-format output accepts often
enough (roughly 70%) to earn that back; over a 10-prompt suite no value of K beat leaving
speculation off. Residency is therefore a launch-time decision, and the server's per-request
`speculative_enabled` cannot change it — it only stops drafting.

When drafting is on, an **adaptive gate** stops it mid-generation if acceptance falls below 45%
over at least 24 drafted tokens, falling back to one token per target pass through the same
`forward_batch` call with a batch of 1. It salvages a wrong `--spec`; it cannot return the 6
layers. `docs/PERFORMANCE.md` §5 carries the measurements.

## 8. Where to look

| | |
|---|---|
| `src/runner.cpp` | the forward pass, residency, and the decode loops |
| `src/streamer.cpp` | slots, the prefetch queue, I/O workers |
| `src/cpu_reference.cpp` | the CPU oracle the GPU is graded against |
| `shaders/Common.hlsli` | shared load helpers and the gemv inner loop |
| `tools/numpy_reference.py` | upstream-transcribed reference, used to validate E2B |
| [`G4DENSE_FORMAT.md`](G4DENSE_FORMAT.md) | container format (v3) |
| [`FORWARD_PASS.md`](FORWARD_PASS.md) | the arithmetic, constant by constant |
| [`PERFORMANCE.md`](PERFORMANCE.md) | measurements, and the ideas they killed |
