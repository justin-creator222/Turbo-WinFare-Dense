# Round 7 — prefill, FP16 KV, and a real draft model

**Author:** Claude Opus 5 · **Date:** 2026-08-30 · **Implemented by:** me

## Summary

All of phases 1–6 are done. Two of them produced negative results that are reported as such
rather than papered over.

| | round 6 | round 7 |
|---|---:|---:|
| "What is the capital of France?" | 38.3 s | **15.0 s** |
| "…what a ring buffer is" | 66.6 s | **35.3 s** |
| "List three primary colors." | 55.4 s | **27.8 s** |
| `--prompt "Hi" --max-tokens 2` | 22.8 s | **6.9 s** |

All three prompts produce exactly the same text as rounds 5 and 6. Against the round-5 baseline,
"What is the capital of France?" has gone **279.2 s → 15.0 s, 18.6×**.

**The engine now runs a second architecture.** E2B — per-layer embeddings and KV sharing —
loads, generates coherent text at **13.0 tok/s**, and serves as a speculative draft model at a
measured **76.2% acceptance**.

---

## Phase 2 — batched prefill *(the largest win)*

Prefill ran one full 60-layer weight pass per prompt token. `forward_batch()` now runs a chunk
of positions per pass: **532 ms per position against 1,456 ms sequential (2.74×)**, 2.96× end
to end.

`shaders/GemmInt4Batch.hlsl` is new. The enum previously mapped `GemmInt4Batch` to
`GemvInt4.spv` with a comment claiming it took a batch parameter — it did not, so any use would
have silently computed only the first position.

**The win is not from the GEMM.** Measured against M separate `GemvInt4` dispatches,
`GemmInt4Batch` is only **1.17×** faster; it is groupshared-read-bound, and raising
rows-per-group makes it worse (1.07× at 16, 0.95× at 32). What pays is amortizing the streaming
I/O and weight reads across the chunk.

**A 2× mistake worth recording.** The kernel's first version gave each lane a contiguous
16-column block, so lane L read groupshared at float index `L*16`. LDS has 32 banks of 4 bytes
and `L*16 mod 32` takes two values — a 16-way bank conflict on every read, making the batched
kernel **0.58×, slower than what it replaced**. Walking one quantization group at a time with
lane L taking column L doubled throughput for identical arithmetic.

**Three batching bugs, all caught by the oracle.** First run gave argmax 494 against 3282, with
position 0 exact and 1–3 broken at layer 0. `GeGLU` and both `ResidualAccum` dispatches had
element counts scaled by the batch but **not their grid sizes**; the V-norm binding range was
left at one position. All are invisible at M = 1, which is why `run_gpu_forward_test` now runs
the batched path against the oracle as a standing check.

## Phase 1 — gemv access pattern: measured, and abandoned

The hypothesis was that `gemv_int4_row_lane` is bound by activation loads (64 of ~74 memory
instructions per 64 weights). Vectorising them to `Load4` cut that 4×.

**No effect**: GPU phase 476.7 ms before, 477.8 ms after, across runs agreeing to 0.2%. With
round 6's occupancy result that is two eliminated hypotheses; `GemvInt4` sustains ~33 GB/s
against 65–74 GB/s for a pure copy and I do not have a third theory worth a round. The
vectorised form is kept only because it is equally simple and the batched kernel wanted it.

## Phase 3 — KV cache to FP16

`KVWrite.hlsl` packs K/V into the cache as half precision, replacing two `vkCmdCopyBuffer`
calls per layer, and `Attention` reads it back with `f16_load`.

**It bought 4 resident layers, but not for the reason the plan gave.** The plan expected the
freed 1.09 GiB to relieve the residency budget directly; it did not, because the RAM-fraction
cap bound first. What actually moved was the **driver's own refusal point** — it was counting
the KV cache in the total device memory it refuses on, so halving the cache let the import go
from 42 layers to 46. The backstop was then raised from 34% to 38% of visible RAM to let that
through. Residency went 41 → 45 layers, ~6.8% faster.

**On the numerics.** Argmax is unchanged at all four positions, and the CPU reference produces
identical argmaxes with and without FP16 KV. But `max|diff|` rose from ~0.003 to ~0.2, because
half precision quantizes K/V to ~1e-3 relative and a 1e-6 CPU/GPU input difference can flip a
stored value by a whole ULP. Rather than simply widen the tolerance — which is how real defects
get waved through — the CPU reference now **rounds its own cache through FP16**, so the oracle
models the engine, and a **top-5 set agreement** check was added to buy back the sensitivity
lost on the numeric channel.

## Phase 4 — per-layer embeddings and KV sharing, so E2B runs

The container gained `ple_dim`, `ple_vocab`, `num_kv_shared_layers`, a model-level PLE block,
and per-layer PLE tensors. Models with none of these are written byte-for-byte as before, so the
31B container stayed valid and no version bump was needed.

Validated against `tools/numpy_reference.py` extended from the upstream source — **argmax and
top-5 agreement at four positions, plus batched prefill** — and then by reading the text.
Deliberately *not* validated against a second implementation of my own assumptions; that is the
round-5 lesson.

Three things had to be right, and only the first was in the plan:

1. **The PLE block** sits between the FFN residual add and `layer_scalar`, so on such a model
   `layer_scalar` moves from the FFN residual add to the PLE block's. `gelu(gate) * per_layer_input`
   is exactly what `GeGLU` computes, so no new kernel was needed — `GeGLU` gained per-operand
   strides because the combined tensor is strided by layer.
2. **KV sharing is an indirection, not storage.** A shared layer reads the cache of the last
   non-shared layer of the *same* attention type — donor type always matches, so geometry and
   capacity line up and nothing is copied.
3. **E2B's last 20 layers have twice the feed-forward width** (12288 against the 6144
   `intermediate_size` claims), which nothing in `config.json` states and which I had no reason
   to expect. The container now records a **measured `layer_d_ff` per layer** rather than
   inferring a rule.

One bug the bisect caught: V-from-K was gated on `is_global`, but that is the 31B's
`attention_k_eq_v` behaviour. E2B's full-attention layers *do* have `v_proj`. It is now keyed
off the tensor being present, which is what upstream actually tests.

## Phase 5 — speculative decoding

`DraftRuntime` wrapped `CpuReferenceRunner` — the minutes-per-token CPU oracle — which would
have made speculation catastrophically slower. It now runs a GPU `ForwardRunner` sharing the
target's Vulkan context, and only forwards context tokens it has not already seen (the previous
version re-ran the whole context per call, quadratic in generated length).

**Measured, on "Write one sentence explaining what a ring buffer is", 24 tokens:**

| | elapsed | acceptance |
|---|---:|---:|
| speculative | **26.5 s** | **76.2 %** |
| `--no-spec` | 35.1 s | — |

**1.33× faster, and 76.2% acceptance against the spec's assumed α = 0.784** — so that
assumption holds. Output is token-for-token identical under greedy, which `test_speculative`
asserts.

**It is not a win on short generations.** The same comparison over 8 tokens is *slower*
(18.6 s against 15.1 s): the draft costs 1.5 GiB of import reserve, which drops the target from
45 to 39 resident layers, and that fixed cost is not amortized over so few tokens.

**One structural mistake, measured and fixed.** My first loop verified `[d1..dK]` and then did a
second full target pass to catch the target up on the accepted bonus token — two target passes
per round, which measured **1.74× slower than not speculating**. Leading the verify batch with
the pending bonus token instead makes `verify[i]` exactly the distribution needed to check draft
`i+1`, and the round costs one target pass. That change alone took 26.3 s → 18.6 s.

## Phase 6 — small items

- **`PostAttn` and `LayerTail` deleted.** Fusing them would collapse 2 of 17 dispatches per
  layer, against 3.6 ms of per-token submission overhead measured in round 6 — noise. The plan
  sanctioned deleting them rather than leaving dead code inflating the untested-kernel list.
  Coverage is now **9 of 11**, and the three remaining are documented at the enum as
  deliberately unused.
- **`EmbedLookup` left unwired.** It is parity-tested, but the CPU-side embedding dequant is
  ~0.1 ms of a ~1400 ms pass. Wiring it would be churn for no measurable gain.
- **8k context still deferred**, as agreed. `ATTN_MAX_SPAN` is enforced, not silently wrong.

## Verification

```
run_gpu_kernels_test        9 of 11 kernels PASS (GemmInt4Batch added, at M = 5)
run_gpu_forward_test        31B: 4 positions + batched prefill, argmax and top-5 exact
                            E2B: same, against the NumPy reference
run_real_generation_test    3 of 3 prompts, text identical to rounds 5 and 6
run_speculative_test        speculative output equals non-speculative, token for token
tokenizer / detok / detokenizer / format / sampling / prompt_pipeline / contracts   PASS
```

## Two process notes

**The 31B container was reconverted** so it records `global_head_dim`, `global_kv_heads` and the
new per-layer `layer_d_ff`, which removes the startup warning about assuming them. The
non-PLE layout is unchanged, so the file is byte-identical in size and passes the same oracle;
the superseded copy was deleted (it regenerates from `models/gemma-4-31b-it-4bit`).

**A measurement scare worth recording.** Immediately after writing 17 GB, the same benchmark
read 12.1 s instead of 6.9 s, and E2B — which streams nothing at all — slowed by 25% too. That
looked like a code regression and I nearly reported it as one. It was disk contention from the
just-completed copy: once the drive went idle both returned to normal (6.82 / 6.90 s, and E2B
to 13.2 tok/s). Benchmarks taken while the storage is still settling are not measurements.

## Still open

- **`GemvInt4` at ~33 GB/s against a 65–74 GB/s ceiling** remains unexplained after two
  eliminated hypotheses.
- **Speculation is a loss on short outputs.** The import reserve is a fixed 1.5 GiB; making it
  adaptive, or loading the draft only for long generations, would fix that.
- **The 31B is still 15 of 60 layers streamed** (21 with a draft loaded). That is a hard memory
  ceiling on this machine, not a code problem.
- **8k context** needs flash-style online softmax.
