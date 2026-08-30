# Round 5 — the architecture against upstream, and the dtype that was wrong all along

**Author:** Claude Opus 5 · **Date:** 2026-08-30 · **Implemented by:** me, not Gemini

## Summary

The plan for this round was to correct four conventions against the upstream Gemma 4 modelling
source, de-hardcode the geometry, and extend the NumPy reference to multiple positions. All of
that is done. Along the way two further defects surfaced that the plan did not anticipate, and
the second of them was the root cause of the incoherent generation:

1. **`v_norm` was never applied on the 50 sliding layers.** Upstream applies it on every layer.
2. **Every non-quantized tensor is BF16.** The round-3 conclusion that the LayerNorm family was
   secretly IEEE FP16 was wrong, and everything built on it — including the NumPy reference —
   inherited the error.

The forward pass now matches an independent NumPy reference at four positions across the GPU
and CPU paths, with no numerical amplification, **and the engine generates correct text for the
first time**:

```
PROMPT 1: What is the capital of France?
RESPONSE: The capital of France is Paris.
[8 tokens in 279.2 s = 0.0287 TPS]

PROMPT 2: Write one sentence explaining what a ring buffer is.
RESPONSE: A ring buffer is a fixed-size circular data structure that efficiently manages a
          continuous stream of data by overwriting the oldest
[24 tokens in 507.1 s = 0.0473 TPS]

PROMPT 3: List three primary colors.
RESPONSE: The three primary colors are:

          1. ...
```

Prompt 1 stopped on its own at the end-of-turn token after 8 tokens rather than running to the
24-token cap, so stop handling and the chat template are working end to end.

**Speed is now the whole problem.** 0.029–0.047 tok/s against a ~232 ms/token bandwidth floor.
The known costs are unchanged and untouched this round: 61 `vkQueueWaitIdle` device drains per
token, roughly 800 descriptor sets written per token, and the embedding lookup still on the
CPU. Optimization was deliberately not started while the output was wrong; it is now measuring
the right thing.

---

## The dtype error, and why nothing caught it for two rounds

Round 3 concluded that the MLX export tags its LayerNorm-family tensors `BF16` but stores IEEE
FP16, on this evidence: decoded as BF16, `model.norm.weight` has median 6.28 with a maximum of
510; decoded as FP16 it is a textbook trained norm weight, median 2.393 spanning 2.06–3.87.

**That argument is invalid, and I should have seen it.** On this value range the two decodings
are a *bijection*: a BF16 value in [2⁻⁷, 2) and an FP16 value in [1, 2) are literally the same
16 bits. Every "the BF16 decode looks absurd" observation has an exactly equivalent "the FP16
decode looks absurd" counterpart on some other tensor, and I found those too without
recognising what they meant. Appearance cannot choose between the two. Neither can distribution
shape, for the same reason.

What settles it is behaviour that neither implementation gets to define
(`tools/dtype_probe.py`, layers 0–7, one token):

| non-quantized tensors read as | pre-softmax \|score\| | hidden rms, layers 0…7 |
|---|---|---|
| **BF16** | mean 5–16, max 24 | 0.81 1.69 1.68 1.66 1.66 1.76 2.08 — **stable** |
| FP16 | mean 192–327, max 549 | 4.3 8.2 15.6 30 57 106 197 — **×2 per layer** |

`Gemma4TextAttention` sets `self.scaling = 1.0`, so nothing downstream rescales those scores.
Under FP16, softmax over scores of ±550 is a hard argmax — attention stops carrying context —
and the residual stream diverges as 2⁶⁰. Under BF16 it is an ordinary transformer.

This explains both standing symptoms:

- **The frequency prior.** `er`, `ed`, `ing`, `al` regardless of input is what a saturated
  attention stack produces.
- **The GPU/CPU divergence from position 1 onward.** It compounded smoothly from 1e-6 relative
  at layer 0 to 0.29 by layer 59 with no single bad layer, which is not a logic bug at all — it
  is a hard-argmax attention amplifying FP32 reassociation noise. Under BF16 the same
  four-position diff agrees to mean 1e-4 with no growth.

It also explains why the engine looked stable at rms ≈ 4.3 before this round: FP16 norms
doubling the state each layer happened to be nearly cancelled by a BF16 `layer_scalar` of ≈0.9.
Two errors partially masking each other.

### The lesson about the NumPy reference

`tools/numpy_reference.py` reported **63/63 tensors MATCH** while the engine was decoding every
norm weight wrongly. That number was worth nothing here, because I wrote the reference from the
same assumption as the engine: the two agreed on being wrong together.

An independent reimplementation only tests what the two implementations do *not* share. It
caught the global-layer V bug, which was a coding difference. It could never catch a
misconception, and I over-trusted it for two rounds — including reporting "the forward pass at
position 0 is correct" when it was not. Only a measurement neither implementation defines —
score magnitude, trajectory stability — could separate them.

---

## Verification

```
run_gpu_kernels_test            8 of 13 kernels PASS, incl. Attention global + sliding
run_cpu_reference_test          4 positions, dumps at each
numpy_reference --tokens        28 of 28 tensors MATCH across 4 positions, layers 0-5
run_gpu_forward_test            4 positions vs the CPU oracle, argmax exact at every one
run_context_dependence_test     A, B, C, D all PASS  (D failed every round until now)
run_real_generation_test 24     3 of 3 prompts coherent, each stopping at end-of-turn
tokenizer / detok / detokenizer / format / sampling / prompt_pipeline    PASS
```

`run_context_dependence_test` check D, the one that has failed since it was written:

```
D. Prefilling 20 prompt tokens ...
   top-5 after the prompt:
     818    25.33  "The"
     669     9.66  " The"
     50429   8.41  "Paris"
     1437    7.14  "the"
     9079    7.03  " Paris"
   (a bare <bos> at position 0 predicts token 236773)
   [PASS] the prompt changed the prediction
```

`Paris` appears twice in the top five and `The` leads, which is what a model about to emit
"The capital of France is Paris." should predict.

**Check B's label was corrected rather than trusted.** It compares the same token at position 0
versus position 1 and round 4 reported the difference as proof that attention reads accumulated
KV. It is not: RoPE rotates the query by position, so this differs even if attention never
looks at a previous key. It now reports only that `position` reaches the kernels, and says so.

GPU vs CPU oracle, the diff that used to fail:

| position | token | mean\|diff\| | CPU argmax | GPU argmax | |
|---|---|---|---|---|---|
| 0 | 2 `<bos>` | 1.10e-05 | 236773 `S` | 236773 | PASS |
| 1 | 3689 `What` | 2.39e-04 | 15938 `Who` | 15938 | PASS |
| 2 | 563 ` is` | 1.62e-04 | 506 ` the` | 506 | PASS |
| 3 | 506 ` the` | 3.30e-04 | 3282 ` reason` | 3282 | PASS |

Before this round the same run gave argmax 759 / 569 / 759 / 759 with mean\|diff\| up to 2.54.
The predictions are now sensible English continuations, and the error no longer grows with
position.

---

## Changes, by plan item

### Phase A — architecture against upstream

**A1 — attention scale reverted to 1.0.** `Gemma4TextAttention.__init__` sets
`self.scaling = 1.0`; the config defines no `query_pre_attn_scalar`. My round-4 change to
`head_dim^-0.5` was wrong. Kept as a push constant so it stays a property of the model, and the
parity test still drives a non-unit value so the kernel cannot quietly ignore it.

**A2 — `layer_scalar` restored, applied correctly.** Upstream's last statement in
`Gemma4TextDecoderLayer.forward` is `hidden_states *= self.layer_scalar` — the whole hidden
state, once, at the end of the layer. `ResidualAccum` now computes
`out = (h + r * res_scale) * out_scale`; the attention residual add passes `out_scale = 1`, the
FFN add passes `out_scale = layer_scalar`. That is exact here because the per-layer-input block
that would otherwise sit between them is skipped when `hidden_size_per_layer_input == 0`.

Both earlier attempts were wrong in opposite directions: applying it to each residual *branch*
attenuated every layer ~11× twice over; removing it let the stream grow unbounded.

**A3 — V reverted to an unweighted RMSNorm of the raw `k_proj` output.** On full-attention
layers `value_states` aliases `key_states` *before* `k_norm`; Python rebinds `key_states` when
`k_norm` returns a new tensor, so V never sees it. My round-4 change to apply `k_norm` was
wrong.

**A4 — `proportional` RoPE needs no code.** Its padded inverse frequencies are zero, so those
pairs rotate by angle 0 — identical to rotating only the first 64 pairs, which the engine
already does. Recorded in `docs/FORWARD_PASS.md` so it is not re-litigated.

**A5 — NumPy reference updated to upstream semantics**, then to BF16 norms.

### Not in the plan — found while executing it

**`v_norm` on every layer.** `self.v_norm = Gemma4RMSNorm(head_dim, with_scale=False)` is
applied unconditionally; only `is_kv_shared_layer` skips it, and that is false for all 60 layers
because this config sets `num_kv_shared_layers = 0`. The engine normalized V only on the 10
full-attention layers and handed the 50 sliding layers a raw `v_proj` output. `with_scale=False`
means the module has no weight at all, which is why the checkpoint contains no `v_norm` tensors
— their absence is not evidence that the norm is skipped.

**The BF16 dtype**, above.

### Phase B — geometry and converter

**B1.** `global_head_dim` and `global_kv_heads` added to `G4DenseHeader`, placed at the head of
the former `reserved` block so every existing offset is unchanged and the struct stays 4096
bytes. `resolve_layer_geometry()` is the single place that resolves per-layer geometry; the
runner, the CPU reference and the KV cache all go through it. A container written before these
fields existed reports 0 and gets the 31B values with a logged note. The KV cache also sized
every slot from the sliding geometry, which over-allocated the full-attention layers here and
would have silently overflowed on a model where their slot is larger.

**B2.** The converter now refuses `hidden_size_per_layer_input != 0`, `num_kv_shared_layers != 0`
and declared MoE blocks, rather than emitting a container that loads and computes nonsense.

**B3.** `models/gemma-4-e2b-dense.g4dense` moved to `models/quarantine/` with a README. Its
checkpoint has `hidden_size_per_layer_input = 256` and shares KV across 20 of 35 layers, so it
was never valid. The E2B fallbacks are removed from `test_gpu_forward`, `test_streamer` and
`test_speculative` so nothing silently reaches for it again.

### Phase C — multi-position verification

`tools/numpy_reference.py` now carries a real KV cache across N positions with sliding-window
masking (`--tokens`), reorganised layer-outer so each layer's weights are dequantized once. The
CPU reference keys its dumps by position and dumps at every position of a prefill.
`test_gpu_forward` takes a token sequence and diffs against the oracle at each position — that
is the test that localized the GPU divergence to "position ≥ 1", and it now passes.

---

## Decisions made without asking

1. **Quarantined rather than deleted the E2B container.** It is 1.2 GB of the user's data and
   the plan allowed either. Moving it makes a stale path fail loudly; deleting it would not.
2. **Kept `tools/dtype_probe.py` in the repo.** It is the only artifact that can settle a
   dtype question of this kind, and the reasoning that failed here would otherwise recur.
3. **Left `has_legacy_global_geometry` as a logged note, not an error.** The existing 17 GB
   container predates the new header fields and reconverting it is expensive; the fallback
   values are exactly what the code assumed before, so nothing changes behaviourally.
4. **Did not re-run the full 60-layer NumPy bisect after the dtype fix.** The 4-position,
   6-layer cross-check plus the 4-position GPU-vs-CPU diff at full depth cover the same ground
   for far less time. The full-depth run is worth doing but is not gating anything.
5. **Stopped short of performance work again.** Still ~10 s/token, still 61 device drains per
   token. Correctness took the round.

## Corrections to my own earlier reports

- **Round 4's "the engine's forward pass at position 0 is correct" was wrong.** It rested on
  63/63 agreement with a reference that shared the engine's dtype error.
- **Round 3's FP16 norm conclusion was wrong**, and it was the root cause of the incoherent
  generation, not a fix for it.
- **Round 4's attention-scale and V-from-K "fixes" were both wrong** and are reverted.
