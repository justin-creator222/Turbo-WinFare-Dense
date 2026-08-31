# Forward Pass Specification & Authoritative Architecture

**Date:** 2026-08-29  
**Status:** Authoritative (Supersedes `docs/history/spec.md` §3.1 and §5.1)  
**Target Model:** `mlx-community/gemma-4-31b-it-4bit` (@ `696d436c404745a59f30e4939a658162b0a9e57f`)  
**Draft Model:** `mlx-community/gemma-4-e2b-it-4bit` (@ `238767527555cb75a05732a84dff5d6ba0dd6809`)

---

## 1. Checkpoint-Derived Model Geometry

All parameters below are derived directly from the official checkpoint `config.json` files and verified by `tools/resolve_model.py`:

| Parameter | Target 31B Dense Model | Draft E2B Model |
|---|---|---|
| **Architecture** | `Gemma4ForConditionalGeneration` | `Gemma4ForConditionalGeneration` |
| **`d_model` (hidden_size)** | **5,376** | **1,536** |
| **`d_ff` (intermediate_size)** | **21,504** | **6,144** |
| **Total Layers** | **60** | **35** |
| **Query Heads (`num_attention_heads`)** | **32** | **8** |
| **KV Heads (`num_key_value_heads`)** | **16** (GQA 2:1) | **1** (MQA 8:1) |
| **Head Dimension (`head_dim`)** | **256** (`32 × 256 = 8192 ≠ d_model`) | **256** |
| **Vocabulary Size** | **262,144** | **262,144** |
| **Sliding Window Size** | **1,024** tokens | **512** tokens |
| **Activation Function** | `gelu_pytorch_tanh` (`GeGLU`) | `gelu_pytorch_tanh` (`GeGLU`) |
| **Logit Softcapping** | **30.0** (`tanh(x / 30.0) * 30.0`) | **30.0** |
| **RMSNorm Epsilon** | `1e-6` | `1e-6` |
| **RoPE $\theta$ (Sliding Attention)** | `10,000.0` (Full RoPE: 128 pairs) | `10,000.0` |
| **RoPE $\theta$ (Global Attention)** | `1,000,000.0` (Partial RoPE: 0.25 = 64 pairs) | `1,000,000.0` (Partial RoPE: 0.25) |
| **BOS / EOS Token IDs** | BOS: `2`, EOS: `[1, 106, 50]` | BOS: `2`, EOS: `[1, 106, 50]` |

### Attention Layer Pattern (Global vs Sliding)

From checkpoint `text_config.layer_types`:

- **Target 31B (60 layers):** 5 Sliding : 1 Full repeating pattern.
  - **10 Full Attention Layers:** `[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]`
  - **50 Sliding Attention Layers:** all other 50 layers.
  - Global Layer Mask: `0x0820820820820820` (`(1<<5)|(1<<11)|(1<<17)|...|(1<<59)`).
- **Draft E2B (35 layers):** 4 Sliding : 1 Full repeating pattern.
  - **7 Full Attention Layers:** `[4, 9, 14, 19, 24, 29, 34]`
  - **28 Sliding Attention Layers:** all other 28 layers.

---

## 2. Quantization & Packing Specification

- **Scheme:** MLX Affine INT4 (Group Size 64).
- **Storage:** 4-bit unsigned integers packed low-nibble-first (8 weights per 32-bit word).
- **Scales & Biases:** `bfloat16` per group of 64 weights.
- **Dequantization formula:**
  $$\text{weight} = \text{quant} \times \text{scale} + \text{bias}$$
- **Size Breakdown (31B):**
  - Per Transformer Layer: **269,446,146 bytes** (256.96 MB, 4096-byte aligned).
  - Embedding Table (tied to LM Head): **792,723,456 bytes** (755.99 MB).
  - Total Converted 31B Model Container: **16,959,496,192 bytes** (~15.79 GiB).

---

## 3. Two-Tier Residency Memory Design (Resolves F4)

On AMD Ryzen Z1 Extreme APU / Radeon 780M, physical heaps measured via Vulkan 1.3:

| Vulkan Heap | Physical Pool | Flags | Size | Role in Engine |
|---|---|---|---:|---|
| **Heap 0** | VRAM Pool | `DEVICE_LOCAL` only (not host-visible) | **13.10 GiB** (13,417 MB) | **Pinned Layer Residency** |
| **Heap 1** | System RAM | `HOST_VISIBLE \| HOST_COHERENT` | **6.55 GiB** (6,708 MB) | **Streaming Ring (4 slots), Embeddings, KV Cache, Activations** |
| **Heap 2** | Small ReBAR | `DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT` | **256 MB** | Staging transfer window |

### Measured Bandwidth Baselines (R0.2)
- **GPU Compute Read from Heap 0:** **73.57 GB/s**
- **GPU Compute Read from Heap 1:** **65.30 GB/s**
- **Staging Upload (`vkCmdCopyBuffer` Heap 1 $\to$ Heap 0):** **26.94 GB/s**
  - Cost to stage 6 pinned layers (Tier 1, 1.54 GB) at startup: **~57 ms** (one-time).
  - Cost to stage 48 pinned layers (Tier 3, 12.33 GB) at startup: **~450 ms** (one-time).

---

## 4. Multi-Tier Budgets & Throughput Projection

| Tier | Pinned | Heap 0 (MB) | Heap 1 (MB) | Total Footprint (MB) | Ceiling (MB) | Projected TPS ($\alpha=0.78$) | Feasibility |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **Tier 1 (Baseline)** | 6 | 1,541.8 | 3,197.9 | 4,739.7 | 6,000.0 | **1.46 TPS** | ✔ **FEASIBLE** |
| **Tier 2 (Balanced)** | 21 | 5,396.2 | 3,197.9 | 8,594.2 | 10,000.0 | **1.97 TPS** | ✔ **FEASIBLE** |
| **Tier 3 (High-Perf)** | 48 | 12,334.3 | 3,197.9 | 15,532.2 | 16,000.0 | **5.28 TPS** | ✔ **FEASIBLE** |
| **Tier 4 (Resident)** | 60 | 15,417.8 | 3,206.0 | 18,623.8 | 22,000.0 | 9.36 TPS | ✘ Exceeds Heap 0 (32 GB only) |

> **Note on Tier 1 Throughput Projection:**  
> On sustained NVMe streaming at 5.68 GB/s with 54 streamed layers (13.87 GB I/O per pass), speculative verification with $K=6$ drafts at $\alpha=0.78$ yields **1.46 TPS** (or **1.73 TPS** at $\alpha=0.85$). The original spec assumption of $\ge 2.50$ TPS for Tier 1 assumed either an unphysically high streaming bandwidth (>10 GB/s) or higher pinned layer counts exceeding 6 GB RAM. Tier 3 achieves **5.28 TPS** ($\alpha=0.78$) and **6.29 TPS** ($\alpha=0.85$), meeting the high-performance gate.

---

## 5. Forward Pass Kernel Dispatch Graph

For each decoding pass ($M = K + 1 = 7$ tokens in batched speculative verification):

1. **Embedding Lookup (`EmbedLookup`):** Gather $M$ tokens from tied embedding table.
2. **Transformer Blocks (Layers $0 \dots 59$):**
   - **Input RMSNorm (`RMSNormK`):** $w_{\text{norm}} \times \text{rsqrt}(\text{mean}(x^2) + 1e-6)$.
   - **Q, K, V Projections (`GemvInt4` / Batched GEMM):**
     - $Q \in \mathbb{R}^{M \times (32 \times 256)}$, $K \in \mathbb{R}^{M \times (16 \times 256)}$, $V \in \mathbb{R}^{M \times (16 \times 256)}$.
   - **Q/K Norm & RoPE (`QKVEpilogue`):**
     - $Q \leftarrow \text{rmsnorm}(Q)$, $K \leftarrow \text{rmsnorm}(K)$.
     - Apply RoPE (full 128 pairs for SWA, partial 64 pairs for Global).
   - **KV Cache Update & Attention (`Attention`):**
     - Write K/V into ring buffer slot `pos % 1024` for SWA, full buffer for Global.
     - Softmax scale = $1.0$.
   - **Output Projection (`GemvInt4`):** $O \in \mathbb{R}^{M \times 5376}$.
   - **Post-Attention Residual & FFN Norm (`PostAttn` / `RMSNormK`).**
   - **Feedforward Network (`GemvInt4` + `GeGLU` + `GemvInt4`):**
     - Gate & Up Projections: $\mathbb{R}^{M \times 21504}$.
     - `GeGLU`: $\text{gelu\_pytorch\_tanh}(\text{gate}) \times \text{up}$.
     - Down Projection: $\mathbb{R}^{M \times 5376}$.
   - **Layer Tail Residual (`LayerTail`).**
3. **Final RMSNorm (`RMSNormK`):** Normalize hidden states across $M$ positions.
4. **LM Head / Argmax (`GemvInt4` / `Softcap` / `LMHeadGreedy`):**
   - Compute logits for $M$ positions against tied embedding weights.
   - Apply softcapping: $\text{softcapped} = 30.0 \times \tanh(\text{logits} / 30.0)$.
   - Rejection sampling verification against target logits.

---

## 6. Upstream-Verified Conventions

Every item below was settled by reading `transformers/models/gemma4/modular_gemma4.py`
(`Gemma4TextAttention`, `Gemma4TextDecoderLayer`) and `Gemma4RMSNorm`, which aliases
`Gemma3nRMSNorm`. They are recorded here because each one had already been guessed at wrongly
at least once, and because **GPU-vs-CPU parity cannot detect a convention both paths share** —
neither can the NumPy reference, which was written from the same assumptions as the engine.
Upstream is the authority; the reference only checks that we implement what we intended.

### Attention scale is 1.0, not `head_dim^-0.5`

`Gemma4TextAttention.__init__` sets `self.scaling = 1.0` outright, and that value is what is
handed to the attention interface. There is no `query_pre_attn_scalar` in this config.

A round-4 change made it `head_dim^-0.5`, reasoning that `q_norm` (a constant 1.8779) could not
be absorbing the factor. The premise was fine and the conclusion wrong: upstream simply does
not scale. The value is still a push constant rather than a kernel literal, so it stays a
property of the model and the parity test can prove the kernel honours a non-unit value.

### `v_norm` is unweighted and applies to every layer

```python
self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
...
value_states = self.v_norm(value_states)
```

`with_scale=False` means the module has no `weight` parameter at all — a pure RMS
normalization. That is why the checkpoint contains zero `v_norm` tensors; their absence is not
evidence that the norm is skipped.

It is guarded only by `is_kv_shared_layer`, which is false for all 60 layers because this
config sets `num_kv_shared_layers = 0`. **So V is normalized on the 50 sliding layers too.**
The engine normalized V only on the 10 full-attention layers and fed the sliding layers a raw
`v_proj` output.

### Full-attention layers take V from the RAW `k_proj` output

```python
key_states   = self.k_proj(hidden_states).view(hidden_shape)
value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
key_states   = self.k_norm(key_states)          # returns a NEW tensor; rebinds key_states
key_states   = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
value_states = self.v_norm(value_states)
```

`attention_k_eq_v` is true and the 10 full-attention layers have no `v_proj`, so `value_states`
aliases `key_states` **before** `k_norm` runs. Because `k_norm` returns a new tensor rather than
mutating in place, V never sees `k_norm` and never sees RoPE. A round-4 change applied `k_norm`
to V; that was wrong.

On the GPU this is why the V-norm dispatch runs *before* the Q/K epilogue — that ordering is
what keeps `buf_k_` raw at the point V reads it, and the barrier after it exists so the K
epilogue cannot overwrite `buf_k_` first.

### RMSNorm is `x_hat * w`, not `x_hat * (1 + w)`

`Gemma3nRMSNorm.forward` multiplies by `self.weight` directly. The `(1 + w)` form used by some
other Gemma generations does **not** apply here.

### `layer_scalar` scales the whole hidden state once, at the end of the layer

`hidden_states *= self.layer_scalar` is the final statement of
`Gemma4TextDecoderLayer.forward`, after both residual adds and after the per-layer-input block
(which is skipped entirely when `hidden_size_per_layer_input == 0`, as here).

The engine folds it into the FFN residual add's `out_scale`, which is exact for this model
since nothing sits between that add and the multiply. `ResidualAccum` therefore computes
`out = (h + r * res_scale) * out_scale`: the attention add passes `out_scale = 1`, the FFN add
passes `out_scale = layer_scalar`.

Two earlier attempts were wrong in opposite directions. Applying it to each residual *branch*
(`hidden += branch * scalar`, twice per layer) attenuated every layer roughly 11x twice over,
making the layers near no-ops. Removing it entirely let the residual stream grow unbounded to
rms ~50 by layer 59. The correct recurrence is bounded: the branches are computed from a
normalized hidden state, so their magnitude is roughly independent of |h| and the stream
converges to |h| ~= s*|branch|/(1-s).

`layer_scalar` is **BF16**, like every other non-quantized tensor here — 0.0894 at layer 0,
0.0654 at layer 1, then 0.75–0.99 through the stack and 0.0364 at layer 59. See the dtype
section below: for a while this file claimed the LayerNorm family was FP16 and `layer_scalar`
alone was BF16. All of it is BF16.

### `proportional` RoPE needs no separate implementation

`config.rope_parameters.full_attention` names `proportional` with
`partial_rotary_factor = 0.25`. From `modeling_rope_utils.py`:

```python
rope_angles      = int(rope_proportion * head_dim // 2)      # 0.25 * 512 // 2 = 64
inv_freq_rotated = 1.0 / (base ** (arange(0, 2*rope_angles, 2) / head_dim))
inv_freq         = cat(inv_freq_rotated, zeros(head_dim//2 - rope_angles))
```

The padded entries are zero inverse frequency, hence angle 0, hence `cos = 1, sin = 0` — an
identity rotation. That is exactly equivalent to rotating only the first 64 pairs and leaving
the rest alone, which is what the engine already does (`rotated_pairs = 64` on global layers).

**No code change is required.** This is recorded so it is not re-litigated; it also explains why
all 60 layers matched the reference at position 0 despite `proportional` never being
implemented by name.


### Every non-quantized tensor is BF16 — the header was right

The MLX safetensors header tags all 1,600 non-quantized tensors `BF16` and that tag is correct:
the LayerNorm family (input / post_attn / pre_ffn / post_ffn layernorms, `q_norm`, `k_norm`,
`model.norm`), `layer_scalar`, and the quantization scales and biases.

An earlier round concluded the LayerNorm family was secretly IEEE FP16, because the BF16 decode
"looks wrong" (`model.norm.weight` median 6.28, max 510) while the FP16 decode looks like a
textbook trained norm weight (median 2.393, range 2.06–3.87). **That reasoning does not work.**
On this value range the two decodings are a bijection — a BF16 value in [2⁻⁷, 2) and an FP16
value in [1, 2) are the same 16 bits — so each decode is internally self-consistent and neither
can be chosen by how it looks. The distribution-shape argument fails for the same reason.

What settles it is behaviour, measured on the checkpoint (`scratchpad/dtype_probe.py`, layers
0–7, one token):

| non-quantized tensors read as | pre-softmax \|score\| | hidden rms, layers 0…7 |
|---|---|---|
| **BF16** | mean 5–16, max 24 | 0.81 1.69 1.68 1.66 1.66 1.76 2.08 — **stable** |
| FP16 | mean 192–327, max 549 | 4.3 8.2 15.6 30 57 106 197 — **×2 per layer** |

`Gemma4TextAttention` sets `self.scaling = 1.0`, so nothing downstream rescales those scores.
Under FP16, softmax over scores of ±550 is a hard argmax: attention stops carrying context, and
the residual stream diverges as 2⁶⁰. Under BF16 it is an ordinary transformer.

Two consequences of the FP16 reading, both observed and both explained by it:

- **Generation collapsed to a frequency prior** (`er`, `ed`, `ing`, `al`) regardless of input —
  the signature of saturated attention.
- **The GPU and CPU paths diverged from position 1 onward**, compounding smoothly from 1e-6
  relative at layer 0 to 0.29 by layer 59 with no single bad layer. That is not a logic bug; it
  is a hard-argmax attention amplifying FP32 reassociation noise. With BF16 the same 4-position
  diff agrees to mean 1e-4 with no growth.

**The NumPy reference could not catch this**, and its 63/63 agreement was worth nothing here: it
was written from the same assumption as the engine, so the two agreed on being wrong together.
Only a measurement that neither implementation defines — score magnitude and trajectory
stability — could separate them. That is the general lesson from this round.
