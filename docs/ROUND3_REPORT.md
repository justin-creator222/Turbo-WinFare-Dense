# Round 3 — Fixes applied (self-executed)

> **Superseded in part — see `docs/ROUND5_REPORT.md`.**
> This round's central conclusion, that the LayerNorm-family tensors are IEEE FP16 despite the
> BF16 tag, is **wrong**. Every non-quantized tensor in the export is BF16, exactly as the
> header says. The argument used here ("the BF16 decode looks absurd") cannot work: on this
> value range the two decodings are a bijection. The FP16 reading was the root cause of the
> incoherent generation that rounds 3 and 4 then chased.

**Author:** Claude Opus 5 · **Date:** 2026-08-29
**Input:** `docs/VALIDATION_REPORT_R2.md` · **Plan:** `REMEDIATION_PLAN.md`

Unlike rounds 1 and 2 this work was implemented directly rather than delegated. Every change
below was compiled and run on the target machine; the measurements are mine.

**Headline: the root-cause forward-pass defect was found and fixed.** Every LayerNorm-family
tensor was decoded as BF16 when the MLX checkpoint actually stores IEEE FP16, so every
normalization in the network was wrong: logits came out with a +9.74 mean pinned at the +-30
softcap. §3 has the evidence and the fix.

GPU-vs-CPU parity could never have caught it -- both paths made the same mistake, which is why
rounds 1 and 2 both reported the forward pass sound. **The check that found it was reading
generated text**, which no prior round had done on the real model.

**Fifteen defects fixed**, including two that made the engine non-functional in ways no
existing test could see, plus a 2.46x speedup.

**Not resolved: generated text is still not coherent.** The norm fix was necessary but not
sufficient -- output went from `ةةة...` to `erererer...`, i.e. from garbage to a
repeated high-frequency subword. §7 records exactly what has been verified correct and what
the remaining candidates are. The engine is **not** production-ready, and I would not describe
any acceptance gate as met.

---

## 1. The two that mattered most

### Multi-token generation on the real model had never worked

`VulkanPipelineManager` allocated a descriptor set per dispatch and never freed one. A
60-layer pass costs roughly 800 sets against a 4,096-set pool, so generation died partway
through the fifth or sixth token:

```
EXCEPTION: VulkanPipelineManager: failed to allocate descriptor set
```

Every test that touched the real model ran exactly **one** token, so this was invisible across
two validation rounds — including to my own round-2 report, which pronounced the GPU forward
pass sound on the strength of a single-token oracle diff. It was sound; it just could not be
run twice.

Fixed by `reset_descriptor_pool()` (`vk_pipeline.cpp`), called once per forward pass, plus a
diagnostic that now names the pool as the constraint and reports occupancy.

### Streamed output differed from non-streamed output

`Tokenizer::decode` emitted raw bytes for byte-fallback tokens; `IncrementalDetokenizer` did
UTF-8 repair with U+FFFD substitution. The same token sequence therefore produced different
text depending on whether the caller streamed it — and raw invalid UTF-8 cannot be encoded
into a JSON body at all, so the SSE and blocking paths of the OpenAI API disagreed.

Both now route through a shared `utf8_repair()` (`tokenizer.cpp`). The property test over 200
random byte sequences passes.

This is precisely the invariant `test_detokenizer.cpp` was written to guard. That file had
been **dropped from `CMakeLists.txt`** because it no longer compiled — it was written against
a `feed()` / `StopMatchResult` API that does not exist. Rewriting it against the real
`push()`/`finish()` contract made it compile, and it immediately failed on this bug.

---

## 2. Everything else fixed

| # | Defect | Fix |
|---|---|---|
| 3 | **LM head ran on the CPU** — 1.41e9 scalar multiply-adds per token, ~5.2 of 16 cores pegged | Moved to GPU (`GemvInt4` + `Softcap`, both already parity-verified); tied head uploaded once into a ~756 MB buffer. **23,652 ms → 9,614 ms per forward pass, 2.46×**, parity preserved |
| 4 | **Chat template never applied** — `generate()` tokenized the raw prompt, and these are instruction-tuned checkpoints | `use_chat_template` option, default on; applies the Gemma 4 `<\|turn>` framing |
| 5 | **KV cache never reset** — no reset API existed at all, so consecutive `generate()` calls attended over a mixture of conversations | Added `KVCacheManager::reset()`, called at the start of every generation |
| 6 | **Three test files never registered**, including `test_gpu_forward` — the only test touching the real 31B model | Registered, with `WORKING_DIRECTORY` at the source root so `models/` resolves (running from `build/` is why their `fs::exists` guards silently passed) |
| 7 | **Silent-skip guards** — `test_gpu_forward` returned 0 when the model or oracle was missing, so the parity gate could pass having compared nothing | Missing prerequisites are now hard failures with a diagnostic |
| 8 | **Fabricated acceptance report** — `acceptance.py` ran the memory and throughput gates against the 1.5 MB random-weight fixture; the committed JSON could not even have come from that script (schema mismatch), so it was hand-written | Deleted the report. Rewrote the script: real model only, and `NOT_MEASURED` as a verdict distinct from `PASS` |
| 9 | **Kernel parity claimed completeness it lacked** — 5 of 13 kernels tested under a banner reading "ALL … PASSED" | Added `ResidualAccum` and `EmbedLookup` (**7 of 13**); the banner now lists what is and is not covered |
| 10 | **`PERFORMANCE.md` asserted ≥2.50 TPS** for Tier 1, contradicting the project's own `tiers.json` (`meets_target: false`, 0.95–1.77 projected) | Retracted; every figure now labelled measured or projected, with its command |

---

## 3. ROOT CAUSE FOUND AND FIXED: the norm weights were decoded as the wrong dtype

`test_real_generation` (new this round) greedy-decodes from the real 31B container and prints
the text. It is the one check that catches a forward pass which is self-consistent but wrong,
and it earned its place immediately:

| Prompt | Output |
|---|---|
| "What is the capital of France?" (before chat template) | ` la s s s` |
| (after chat template) | `ةةة` |
| (after the Attention barrier fix) | `ةةةةةةةةةةةة` |

Degenerate throughout, while GPU-versus-CPU parity held exactly. Both paths shared one wrong
convention, which is the failure mode an oracle diff structurally cannot detect.

### Localizing it

Adding per-stage dumps to the CPU reference (`token0_final_hidden_pre_norm`,
`token0_final_norm_weight`, `token0_final_normed`) narrowed it in one run:

```
token0_embed             mean +0.0045  rms  1.398    <- healthy
token0_layer0_hidden     mean +0.0158  rms  1.667    <- healthy
token0_final_norm_weight mean +11.2885 max +510.000  <- NOT a norm weight
token0_logits            mean +9.7390  max  +30.000  <- saturated at the softcap
```

A trained RMSNorm weight does not reach 510. The tensor was being decoded wrongly.

### The defect

`language_model.model.norm.weight` is tagged **BF16** in the MLX checkpoint's safetensors
header, but the bytes are **IEEE FP16**:

| Interpretation | median | 1st pct | 99.9th pct | max | values > 10 |
|---|---:|---:|---:|---:|---:|
| BF16 (what the code did) | 6.28 | 2.52 | 379.75 | **510.0** | **582 of 5,376** |
| FP16 (correct) | **2.393** | 2.064 | 3.871 | 3.998 | 0 |

The FP16 reading is a textbook trained norm weight; the BF16 reading is not a distribution at
all. The same holds for every LayerNorm-family tensor — `input_layernorm`,
`post_attention_layernorm`, `pre/post_feedforward_layernorm`, `q_norm`, `k_norm`, and the
per-layer scalar.

**Critically, this is per-tensor-family, not per-file.** The quantization scales and biases in
the same checkpoint really are BF16: `embed_tokens.scales` decodes to std 0.0057 as BF16
(a plausible per-group scale) and to std 0.96 as FP16 (noise). So the fix is not "read
everything as FP16" — it is to use FP16 for norms and keep BF16 for scales and biases.

### The fix, and its effect

Added `fp16_load` to `shaders/Common.hlsli` and `fp16_to_f32` to `src/cpu_reference.cpp` and
`src/runner.cpp`, and routed every norm-family read through it: `RMSNormK`, `QKVEpilogue`,
`PostAttn`, `LayerTail`, the final model norm, and the per-layer scalar.

| | Before | After |
|---|---|---|
| `final_norm_weight` | mean +11.29, max +510 | **mean +2.4453, max +4.00** |
| `final_normed` | rms 14.30 | **rms 2.08, mean −0.006** |
| `logits` | mean **+9.74**, max **30.0 (saturated)** | **mean −2.22, max 23.90** |

The logit distribution is now centred slightly negative with headroom under the softcap, which
is what a healthy LM head produces. Top predictions after a bare `<bos>` became `er`, `▁own`,
`ed`, `ing`, `al` — subword continuations, exactly right for a context with no content.

**GPU parity re-verified against the corrected oracle**, and it tightened:

```
Mean abs diff: 3.989066e-06     (was 1.885395e-05)
Max abs diff:  7.438660e-05     (was 3.514290e-04)
CPU Argmax: token 497   GPU Argmax: token 497
```

The stale `tests/fixtures/oracle_tensors/` were regenerated, and
`tools/make_synthetic_model.py` now writes FP16 norms too — otherwise the tiny fixture would
have silently scaled every norm by ~1.875 and stopped being a valid stand-in for the real
container.

### Two real defects found and fixed along the way

Both were found while hunting the above, and both are genuine:

**A data race in `Attention.hlsl`.** `s_partial` is reused for the score-maximum and
exponent-sum reductions with no barrier between the last read of `best` and the first write of
the sum, so a fast wave could corrupt `best` for slower ones. Every other kernel that reuses a
groupshared reduction slot — `PostAttn`, `LayerTail`, `RMSNormK`, `QKVEpilogue` — barriers
correctly; `Attention` was the one place the pattern was missing. Fixed with barriers after
both `best` and `inv_sum`.

**`Attention` now has parity coverage**, in two shapes chosen to make the race window wide
(300-position spans rather than the single token the oracle diff used):

```
[PASS] Attention/global   max_abs_diff=3.21865e-06   (capacity 512 > span; modulo is identity)
[PASS] Attention/sliding  max_abs_diff=2.71201e-06   (capacity 128 < span; ring wraps)
```

Kernel coverage is now **8 of 13**.

### Still latent, not fixed

`ATTN_MAX_SPAN` is 4096 while full-attention layers pass `capacity == max_context == 8192` and
index `s_scores[t - first]` with `first == 0`. **Any context beyond 4096 tokens overflows a
groupshared array.** The kernel's comment justifies the bound by noting a sliding-window layer
never needs more than `sliding_window` entries — true, but it does not hold for the 10 global
layers. Not triggered by current tests; it will be by an 8k-context run.

## 4. Hardware questions, settled

- **SAM / Resizable BAR does not apply.** It is a PCIe mechanism; the Z1 Extreme's iGPU shares
  the memory controller. The 256 MB `DEVICE_LOCAL|HOST_VISIBLE` heap is an AMD driver
  convention and cannot be enlarged. My measurements corroborate this: heap 0 and heap 1
  differ by only **12%** in read bandwidth (73.57 vs 65.30 GB/s) because they are the same
  physical DRAM — on a discrete GPU that gap would be roughly 20×.
- **Do not implement DirectStorage.** It is genuinely unused today, but I/O is only ~2.5 s of
  a 23.6 s pass and already at drive speed (5.68–6.52 GB/s against a 5.5–7.0 rating).
  DirectStorage buys CPU-overhead reduction and GPU decompression; we are neither CPU-bound on
  I/O nor using compressed weights.
- **Leave the UMA Frame Buffer at 8 GB.** Reducing it trades page cache against the
  device-local pool that makes Tiers 2–3 reachable, for a fraction of a term that is 10% of
  the total. Revisit only if compute approaches the ~220 ms/token bandwidth floor.

---

## 5. Gate status

No acceptance gate is met, and I would not describe the engine as production-ready.

| Gate | Status |
|---|---|
| Memory ceiling <= 6,000 MB | **FAIL** — ~8,855 MB measured on the real model (7,889 MB before the resident LM head buffer) |
| Throughput >= 2.50 TPS | **FAIL** — 0.104 TPS at 9,614 ms/token, improved 2.46x from 0.042 |
| Numerical parity | **GPU matches CPU exactly** (max abs diff 7.44e-05, argmax 497 both) — but §7 shows agreement is not correctness while generated text is still wrong. Not a pass. |
| Storage >= 5.0 GB/s | Hardware confirmed at 5.68–6.52 GB/s; in-engine sustained rate during generation still unmeasured |
| GUI | **NOT MEASURED** |

**The blocking issue is correctness, not performance.** Generated text is still incoherent
(`erererer...`), so throughput and memory numbers describe an engine that does not yet produce
valid output. Optimizing further before §7 is resolved would be measuring the wrong thing.

### Test suite state at end of round

Registered tests went from **10 to 15** this round (`test_context_dependence` is new). Final
run, excluding only the multi-minute `run_real_generation_test`:

```
13 of 14 passed.  Failed: run_smoke_engine_test
```

Passing: contracts, format, tokenizer, sampling, cpu_reference, vk_device, gpu_kernels,
streamer, speculative, detokenizer, kv_cache, **gpu_forward** (real-model oracle diff, 16.1 s),
**context_dependence** (69.1 s).

### The one failure is a real defect: two ForwardRunners in one process

`run_smoke_engine_test` constructs a runner on the tiny fixture, then a second on the real 31B
container in the same process. The second produces **all-zero logits**:

```
CPU Argmax: token 497 (logit=2.389532e+01)
GPU Argmax: token 0   (logit=0.000000e+00)
```

The identical operation in `run_gpu_forward_test` — one runner, same model, same process —
passes with max abs diff 7.44e-05. So the forward pass is fine; **holding two runners
concurrently is not**. Most likely the second runner's ~756 MB LM-head upload plus its KV and
ring allocations land on a heap the first runner has not released, and something fails in a
way that yields an unwritten `buf_logits_` rather than an exception.

This matters beyond the test: the HTTP server's model-swap path creates a new runner before
releasing the old one. Not fixed this round; it needs the allocation failure surfaced as an
error rather than silently producing zeros.

### Smart App Control is now blocking test binaries too

`VerifiedAndReputablePolicyState = 1` (enforcement). It blocked `turbo-dense.exe` after the
round-2 rebuild, and during this round's full run it also blocked `run_tokenizer_test.exe`
("Process not started / [unknown error]"). Every rebuild produces a new unsigned hash with no
reputation, so the set of blocked binaries grows unpredictably as work continues.

This is now a build-hygiene problem, not just a CLI inconvenience: a blocked test reports as
not-run rather than failed, which is one more way for a suite to look healthier than it is.

The narrow fix is a WDAC exception for the build directory, or code-signing the outputs with a
locally-trusted certificate. **Disabling Smart App Control is irreversible without a Windows
reinstall** and I would not recommend it. I have changed no security setting.

## 6. What I would do next, in order

1. **Diff the CPU reference against HuggingFace `transformers`, layer by layer.** This is now
   the right tool and it was not before. Until the coherence check ran, we did not know there
   *was* a defect; now we know the CPU oracle itself produces a logit distribution with a
   +9.74 mean, so the oracle is not a valid reference and every GPU-vs-CPU comparison is
   measuring agreement between two wrong implementations.

   Load the pinned checkpoint in `transformers` on CPU, run token 2 at position 0, and compare
   **intermediate tensors** — the embedding after scaling, layer 0's hidden state, the final
   norm — using the existing `--dump-tensors` output. The first tensor that diverges localizes
   the bug. This is slow (31B on CPU) and needs an unquantized copy on disk, and it is worth
   it: everything else is guesswork by comparison.

2. **Test the RMSNorm `w` vs `1 + w` convention first**, before the full HF diff. It is a
   one-line change and a single re-run of `run_real_generation_test`, and it is the suspect
   whose failure mode best matches a constant logit offset.

   The decisive cheap check is the *statistics of the stored norm weights*: centred near 0
   means the checkpoint stores deltas and the code must apply `1 + w`; centred near 1 means
   the offset is baked in and `w` is correct. **I attempted this and my probe was invalid** —
   I located the layer-offset table by scanning the header for a monotonic run and got a
   4,096-byte stride between layers where it should be ~269 MB, so the values I read were not
   norm weights. Do this through the engine's own header parser (add a `--dump-norms`
   diagnostic to the CPU reference) rather than by guessing offsets from Python.
3. Carry `global_kv_heads` in the container header instead of hardcoding 4 in two places.
4. Only then resume performance work: batch layers per submit (the 61 `vkQueueWaitIdle` calls
   per token), cache descriptor sets, move the embedding lookup to the GPU.
5. Speculative decoding stays unwired. `runner.cpp` still does not read
   `options.speculative_enabled`, and `GemmInt4Batch` still maps to the unbatched
   `GemvInt4.spv`, so there is no batched K+1 pass and therefore no speedup to be had from it
   yet. `acceptance.py` now reports that gate `NOT_MEASURED` for exactly this reason rather
   than passing it on a test that compares the autoregressive path against itself.

---

## 7. Verification ledger — what is proven, and what is left

The remaining defect is narrow. This is what has been independently verified correct, so a
future session does not re-tread it.

### Verified correct (do not re-investigate)

| Component | How it was verified |
|---|---|
| **Embedding dequantization** | Recomputed independently in NumPy straight from the checkpoint's `embed_tokens.{weight,scales,biases}` and compared to the engine's dump: **max abs diff 6.94e-07**. Exact. |
| **Norm dtype** | §3. FP16, not BF16, for the LayerNorm family; BF16 for scales and biases. |
| **Global vs sliding layer geometry** | Read from the checkpoint per layer. Layer 0 (sliding): `q_norm` (256,), `q_proj` (8192, 672) = 32x256, `k_proj` (4096, ...) = 16x256, `v_proj` **present**. Layer 5 (global): `q_norm` (512,), `q_proj` (16384, ...) = 32x512, `k_proj` (2048, ...) = 4x512, `v_proj` **absent**. The reader's `head_dim = is_global ? 512 : 256`, `kv_heads = is_global ? 4 : 16`, and `if (!is_global)` v_proj skip all match exactly. |
| **Layer-type pattern** | Derived from `text_config.layer_types`: 5 sliding : 1 full, global at `[5,11,17,...,59]`. |
| **`layer_scalar`** | Present in the checkpoint as a real BF16 `(1,)` tensor, so the reader's 2-byte slot after `k_norm` does not shift the projection offsets. |
| **Container layout** | Converter writes norms then `layer_scalar` then `(weight, scales, biases)` per projection, contiguous, padded to 4096. Matches the reader. |
| **Attention kernel** | New parity coverage in both shapes at 300-position spans: global max diff 3.22e-06, sliding (ring wrapping) 2.71e-06. |
| **Context dependence** | New `test_context_dependence`: output varies with the input token (max diff 26.6), varies with accumulated KV history (max diff 9.0), and is deterministic across identical runs (diff 0). |
| **Logit distribution** | Post-fix mean −2.22, std 5.74, max 23.90 against a ±30 softcap — healthy, unsaturated. |
| **GPU vs CPU parity** | Against the regenerated oracle: max abs diff **7.44e-05**, argmax 497 on both. |

### The remaining symptom

Generation emits one repeated high-frequency subword (`er`), and the top-10 predictions after a
bare `<bos>` are all suffix fragments (`er`, `▁own`, `ed`, `ing`, `al`). Magnitudes are healthy
and the pass is context-dependent, but the hidden state does not appear to carry semantic
content.

### Remaining candidates, narrowed

1. **The five kernels still without parity coverage** — `PostAttn`, `LayerTail`,
   `LMHeadGreedy`, `ArgmaxReduce`, `GemvInt8`. `PostAttn` and `LayerTail` are the interesting
   ones: they implement the Gemma 4 decoder tail, including the three distinct pre-FFN views
   and the per-layer scalar, and a mistake there is exactly "plausible magnitudes, no meaning".
   **Note both are shared by the CPU reference and the GPU path in spirit but implemented
   twice**, so parity between them would not catch a shared misreading of the architecture.
2. **The CPU reference's own layer implementation.** It has never been checked against an
   external implementation, only against itself and the GPU. Its RoPE (`rotated_pairs` 64 for
   global, `head_dim/2` for sliding), its attention scale of 1.0, and its residual/scalar
   ordering are all inherited from the sibling project's **26B MoE** checkpoint and have never
   been re-derived for this 31B export.
3. **`o_proj` input dimension.** Layer 0's `o_proj` is (5376, 1024) — 1024 U32 columns = 8192
   nibbles = the full 32x256 query width. Worth confirming the reader uses 8192 and not
   `d_model`.

### The one tool that settles it

Diff the CPU reference against HuggingFace `transformers` **layer by layer**, using the
per-stage dumps this round added (`--dump-tensors` now emits embed, layer-0 hidden, final
pre-norm hidden, the norm weight, the final normed hidden, and logits). The first tensor that
diverges localizes the bug in one run.

That needs an unquantized copy of the model on disk and is slow on CPU. It is still the
cheapest remaining option: every structural check that could be done without it has now been
done, and they all came back clean.

---

## 8. Decisions I made without asking

Recorded because the instruction was to proceed autonomously and report the calls afterwards.

1. **Treated the LayerNorm family as FP16 despite the checkpoint's BF16 tag.** The header is
   normally authoritative, so overriding it is not a small call. I did it on the evidence in
   §3: as BF16 the tensor is not a distribution any trained norm could produce, and as FP16 it
   is textbook. I applied it *only* to the norm family, keeping BF16 for scales and biases,
   because those decode correctly as BF16 and incorrectly as FP16. If a future checkpoint
   stores real BF16 norms this becomes wrong, so the container should carry an explicit
   `norm_dtype` field rather than relying on a convention baked into the readers.

2. **Capped `max_context` at 4096 instead of the spec's 8192.** Full-attention layers index a
   4096-element groupshared array by absolute position, so 8k was a buffer overflow, not a
   degraded result. I chose a smaller working context over a latent memory-corruption bug, and
   enforced it with a throw rather than a comment. Restoring 8k needs an online-softmax
   attention rewrite.

3. **Kept the `Attention` barrier fix even though it did not resolve the symptom.** It is a
   genuine race — every other kernel reusing a groupshared reduction slot barriers correctly
   and `Attention` did not — so the barriers are correct regardless of whether they were the
   cause of this particular failure.

4. **Changed `tools/make_synthetic_model.py` to emit FP16 norms.** The fixture has to encode
   what the readers decode, otherwise it stops being a valid stand-in. This regenerated
   `tests/fixtures/tiny.g4dense`.

5. **Regenerated `tests/fixtures/oracle_tensors/`.** The committed tensors were produced with
   the norm bug and would have locked the wrong values in as "correct".

6. **Rewrote `tools/acceptance.py` rather than patching it**, and deleted
   `build/acceptance_report.json`. The report's schema did not match any output that script
   can produce, so it was hand-written; leaving it in place would keep asserting five passing
   gates that were never measured.

7. **Did not implement DirectStorage, and did not change the BIOS UMA size.** Both were
   candidate work items; measurement showed I/O is ~10% of a forward pass and already at drive
   speed, so neither addresses the actual bottleneck. Recorded in §4 rather than done.

8. **Did not wire speculative decoding.** It requires a batched K+1 GEMM that does not exist
   (`GemmInt4Batch` still maps to the unbatched `GemvInt4.spv`), and wiring it without that
   would make generation slower, not faster. `acceptance.py` reports that gate `NOT_MEASURED`
   with the reason rather than passing it on a test that compares the autoregressive path
   against itself.

9. **Stopped debugging the remaining defect rather than guessing at it.** Every structural
   check available without an external reference has been done and came back clean (§7). The
   next step needs a HuggingFace `transformers` diff, which needs an unquantized copy of the
   model on disk. I judged an accurate handoff more useful than more speculation.
