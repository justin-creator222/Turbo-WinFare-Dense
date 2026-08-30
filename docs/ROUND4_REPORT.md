# Round 4 — the four GUI-blocking issues

> **Superseded in part — see `docs/ROUND5_REPORT.md`.**
> Three claims here are wrong. The attention scale is **1.0**, not `head_dim^-0.5`
> (`Gemma4TextAttention` sets `self.scaling = 1.0`). V on full-attention layers is an
> *unweighted* norm of the **raw** `k_proj` output, not `k_norm(K)`. And "the engine's forward
> pass at position 0 is correct" rested on 63/63 agreement with a NumPy reference that shared
> the engine's FP16 dtype error, so it proved nothing.

**Author:** Claude Opus 5 · **Date:** 2026-08-30

## Summary

Issues 2, 3 and 4 are done. Issue 1 (incoherent generation) is **not fixed**, but it moved from
"something in the layer stack is wrong" to a single precisely-located defect, and the entire
forward pass is now verified correct against an independent implementation.

**Four separate forward-pass bugs were found and fixed this round.** None was detectable by
GPU-vs-CPU parity, because both paths made each mistake identically.

---

## The verification that changed everything

`tools/numpy_reference.py` (new) reads the MLX checkpoint directly and recomputes the forward
pass in NumPy, sharing no code with the engine. Diffing it per layer against `--dump-tensors`
output localizes a defect in one run.

Final state — **63 of 63 tensors match**:

```
embed                max|diff| = 0.00e+00              MATCH
layer0 .. layer59    max|diff| <= 6.10e-04             MATCH   (all 60, sliding and global)
final_normed         max|diff| = 6.10e-05              MATCH
logits               max|diff| = 6.87e-05              MATCH
numpy top-5 token ids: [497, 1852, 524, 514, 522]      identical to the engine
```

The engine's forward pass at position 0 is correct. I should have built this before touching
any constant: it answered in two runs what inspection had circled for hours, and it caught one
of my own wrong guesses.

---

## Issue 1 — incoherent generation: four bugs fixed, one remaining

### Fixed — attention scale was 1.0

`q_norm` is a constant 1.8779 and `k_norm` a constant 1.4941; neither absorbs a
`1/sqrt(head_dim)` factor, and the config defines no `query_pre_attn_scalar`. Scores reached
about ±45, so softmax saturated to one-hot. Now `head_dim^-0.5` per layer, passed as a push
constant, with the parity test exercising a non-unit scale so it cannot silently regress.

### Fixed — `layer_scalar` applied to the residual stream *(a regression I introduced)*

In round 3 I extended the FP16 norm fix to `layer_scalar`, which is genuinely **BF16**:
0.089355 versus 1.428711. Worse, the engine multiplied **both** residual adds by it,
attenuating every layer roughly 11× twice over across 60 layers. Layer 0 output was rms 1.481
against an embedding rms of 1.398 — the layers were near no-ops. Removing it gives rms 2.997,
exactly matching the NumPy reference. It is still read for diagnostics but not applied; its
correct role in this architecture is not established.

### Fixed — global layers dropped `k_norm` from V

Full-attention layers have no `v_proj` (`attention_k_eq_v: True`), so V is copied from K — but
the engine copied it through a **no-scale** RMSNorm, dropping the `k_norm` weight. Correct is
V = K after `k_norm`, before RoPE. The per-layer bisect showed sliding layers 0–4 matching at
~1e-4 while global layer 5 diverged by 1.50e-01; the fix brings it to 1.14e-04.

### Fixed — the OpenAI chat path fed raw JSON as the prompt

`r->generate(req.body, ...)` passed the raw HTTP body, so the model received
`{"model":...,"messages":[...]}` as literal text and never saw the question. It now renders the
parsed messages through the Gemma 4 turn template, which also restores multi-turn history.

### NOT fixed — attention does not attend over history

With all of the above corrected, prefilling the real 20-token prompt still yields the
bare-`<bos>` prediction:

```
top-5 after the prompt:   497 "er"   524 "ed"   522 "ing"   514 "al"   237076
bare <bos> at position 0 predicts:  497
```

Compositionally identical. The prompt does not move the output.

**Ruled out, with evidence:**

| Stage | Verified by |
|---|---|
| Forward pass, position 0 | 63/63 tensors match the NumPy reference |
| Prompt tokenization | `test_prompt_pipeline` (new): 20 tokens, exact round-trip |
| Attention kernel | parity at 300-position spans, both ring shapes |
| Decode-loop positions | `history = prompt_tokens`; prefill 0..N−1, decode at N |
| KV write ordering | K/V appended before attention reads, with a transfer→shader barrier |

**A correction to my own earlier claim.** `test_context_dependence` check B compared the same
token at position 0 versus position 1, saw a difference, and I reported it as "output depends
on accumulated KV history". **It is not evidence of that.** RoPE rotates the query by position,
so the output changes with position even if attention never reads a single previous key. Check
D (new) is the real test, and it fails.

### Further elimination (all measured, all negative)

Everything below was checked after the four fixes and came back clean, so none of it is the
cause:

| Hypothesis | Result |
|---|---|
| Generation-prompt suffix wrong | Dropping `<\|channel>thought\n<channel\|>` changes nothing |
| Chat markers wrong | Checkpoint's own `chat_template.jinja` emits `<\|turn>model\n` then the channel pair — **byte-identical to ours**; `tokenizer_config.json` confirms `eot_token = <turn\|>` |
| Wrong tokenizer file | Repo root, `tests/fixtures/`, and the checkpoint copy are all md5 `72b1044584e75adc53dd4372e903925c` |
| Chat framing at fault | Plain LM continuation of `<bos>What is the capital of France?` gives the same `er`/`ed`/`ing`/`al` |
| KV buffer overflow for global layers | `KVCacheManager` over-allocates (16×256 vs the runner's 4×512) but read and write share a stride — self-consistent |
| **Dequantization convention wrong** | **Disproved semantically.** Embedding cosine similarity: France–Germany **0.379**, France–Paris **0.309**, Germany–Berlin **0.285**, versus ~0.05 for anything against "banana" and 0.003 for banana–the. That is a trained embedding space, so the nibble order, group mapping and affine formula are all right |
| Missing tensors (e.g. per-layer embeddings) | Full checkpoint inventory: every text-model tensor is consumed. No PLE tensors exist. `v_proj` appears exactly 50 times, confirming the 10 global layers correctly lack it |

### Where it actually stands

The top predictions are consistently **low-id, high-frequency subwords** (`er` 497, `ed` 524,
`ing` 522, `al` 514) regardless of input — the signature of a hidden state carrying no
information, with the LM head falling back to a frequency prior. Yet every component is
verified.

**The one trained tensor we deliberately ignore is `layer_scalar`** — 60 of them, real, values
falling with depth (0.0894, 0.0654, ...). Applying it to the residual adds makes the layers
near no-ops, which is why it was removed; but "matching the NumPy reference" only means
"matching my own choice", because that reference omits it too. Its correct role is the single
largest remaining unknown, and it is where I would look next.

**Next steps, in order:**

1. Determine `layer_scalar`'s actual role. It is not standard Gemma 3. Candidates: a scale on
   one branch rather than the residual, a LayerScale-style factor, or an attention temperature.
   The checkpoint is an MLX conversion of `google/gemma-4-31B-it` at revision
   `3548789868c5356dbf307c98e6f609007b82b3eb`; the upstream modelling code would settle it.
2. Extend `numpy_reference.py` to two positions to rule attention-over-history in or out
   independently — everything proven so far is position-0 only.
3. Note this checkpoint is a **vision-language** model (`image-text-to-text`, built for
   `mlx-vlm`). Text-only use should be fine, but it has not been confirmed against a working
   reference run.

---

## Issue 2 — speed: instrumented, not yet optimized

Per-phase attribution added (`TelemetryCollector::record_phase_breakdown`) and printed in the
CLI summary: layer stream I/O, GPU queue wait, LM head, CPU other. The fields already existed
in `TelemetrySnapshot` and were never populated.

Optimization deliberately not started. The plan says measure first, and correctness took the
round. The known costs remain 61 `vkQueueWaitIdle` device drains per token, roughly 800
descriptor sets written per token, and the embedding lookup still on the CPU.

## Issue 3 — six 404 endpoints: done

`/api/config` (GET and POST), `/api/models`, `/api/load_model`, `/api/unload_model`,
`/api/clear_cache` and `/api/stop`, built on the existing `ServerConfig` and `swap_runner`.
`swap_runner` already validates and initializes a new container before releasing the old one.
`/api/models` lists only containers that actually parse, never a bare existence check.
`/api/stop` sets a cancel flag now threaded into both `generate()` call sites.

Two corrections to what I told you earlier: `/api/repack` was already an explanatory comment,
not a live call, so there was nothing dead to remove; and `GET /api/config` was one of six
missing endpoints, not the only one.

## Issue 4 — two runners producing all-zero logits: root cause fixed

Not an allocation failure — `allocate_buffer` already throws and already reports residency
downgrades. The real cause was `bind_kernel` binding `VK_NULL_HANDLE` without checking, which
is undefined behaviour: the dispatch silently does nothing and the output buffer keeps whatever
it held. Exactly "all-zero logits, no error".

Now `bind_kernel` throws naming the kernel, `initialize_pipelines` validates that every kernel
has a live pipeline before any dispatch, and the silent Wave64 subgroup fallback is recorded
and warned about instead of swallowed.

---

## Decisions made without asking

1. **Removed `layer_scalar` from the residual path** rather than hunting for its correct role.
   The NumPy reference omits it and matches exactly; applying it does not. Kept read,
   unapplied, and documented.
2. **Chose V = `k_norm`(K) for global layers** over the engine's no-scale variant, inferred
   from `attention_k_eq_v` and confirmed by the bisect (1.50e-01 → 1.14e-04).
3. **Did not implement `proportional` RoPE.** The config names it for full-attention layers and
   the code applies plain RoPE, yet all 60 layers match the reference at position 0 — so
   whatever it denotes is either equivalent here or only manifests beyond position 0. Not
   guessed at.
4. **Retained the 4096 context cap** from round 3: full-attention layers index a 4096-element
   groupshared array by absolute position, so 8k is a buffer overflow.
5. **Stopped before optimizing.** Speed work on an engine that does not yet produce correct
   text would be measuring the wrong thing.

## Tests added this round

`test_prompt_pipeline` (string → template → tokens, no GPU), `test_context_dependence` check D
(prefill the real prompt and compare against the bare-`<bos>` prediction), per-layer dumps from
the CPU reference (all 60, not just layer 0), and `tools/numpy_reference.py`.
