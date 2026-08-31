# `.g4dense` (v3) Binary Container Specification

**Version:** 3.0  
**Magic:** `0x4734444E` (`'G4DN'`)  
**Sector/DMA Alignment:** 4096 Bytes (`ALIGNMENT_BYTES = 4096`)  
**Packed-weight Alignment:** 16 Bytes (see 4.1)

---

## 1. Overview

The `.g4dense` v3 container is an APU-optimized, single-file container for Gemma 4 Dense models. All layer offsets, embedding payloads, and LM head weights are 4096-byte sector-aligned to enable unbuffered direct memory access (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` and Win32 IOCP) straight into GPU host-visible mapped physical memory.

---

## 2. Header Layout (4096 Bytes Exact)

```
Offset (Hex) | Type       | Field Name             | Description
-------------+------------+------------------------+------------------------------------------
0x0000       | uint32     | magic                  | 0x4734444E ('G4DN')
0x0004       | uint32     | version                | 3 (see 4.1; v2 is REJECTED, not upgraded)
0x0008       | uint32     | quant_type             | 1=Affine INT4 g64, 2=QAT W4A16, 3=FP16
0x000C       | uint32     | num_layers             | 60
0x0010       | uint32     | d_model                | 5376
0x0014       | uint32     | d_ff                   | 21504
0x0018       | uint32     | num_q_heads            | 32
0x001C       | uint32     | num_kv_heads           | 16
0x0020       | uint32     | head_dim               | 256
0x0024       | uint32     | vocab_size             | 262144
0x0028       | uint32     | sliding_window         | 1024
0x002C       | uint32     | quant_group_size       | 64
0x0030       | uint32     | scale_dtype            | 1=BF16, 2=FP16
0x0034       | uint32     | tied_embeddings        | 1=True
0x0038       | uint64     | global_layer_mask      | 64-bit mask of full attention layers
0x0040       | float      | rope_theta_local       | 10000.0
0x0044       | float      | rope_theta_global      | 1000000.0
0x0048       | float      | rope_scaling           | 1.0
0x004C       | float      | final_logit_softcap    | 30.0
0x0050       | uint64     | embed_offset           | Byte offset of token embeddings
0x0058       | uint64     | embed_size             | Byte size of token embeddings
0x0060       | uint64     | lm_head_offset         | Byte offset of LM head
0x0068       | uint64     | lm_head_size           | Byte size of LM head
0x0070       | uint64[60] | layer_offsets          | Offsets for transformer blocks 0..59
0x0250       | uint64[60] | layer_sizes            | Sizes for transformer blocks 0..59
0x0430       | uint8[32]  | payload_sha256         | SHA-256 checksum of payload
0x0450       | uint8[...] | reserved               | Zero-padded tail to exactly 4096 bytes
```

---

## 3. Layer Internal Sub-Block Layout

Each Layer Block *N* (at `layer_offsets[N]`, size `layer_sizes[N]`) is structured with sub-tensors in fixed order.

**Every packed-weight block below is preceded by zero padding to the next 16-byte boundary,
relative to the start of the layer block** (v3; see 4.1). The norms are all multiples of 16
bytes, but `layer_scalar` is 2, so without this pad every projection would land at offset 2
(mod 16). Scales and biases follow their weights contiguously with no pad of their own.

1. **RMSNorm Weights (BF16):**
   - `input_layernorm.weight`: `[d_model]`
   - `post_attention_layernorm.weight`: `[d_model]`
   - `pre_feedforward_layernorm.weight`: `[d_model]`
   - `post_feedforward_layernorm.weight`: `[d_model]`
   - `q_norm.weight`: `[head_dim]`
   - `k_norm.weight`: `[head_dim]`
   - `layer_scalar`: `[1]`

2. **Self-Attention Sub-Block:**
   - `q_proj` (weights `[q_heads*head_dim, d_model/2]`, scales `[q_heads*head_dim, d_model/64]`, biases `[q_heads*head_dim, d_model/64]`)
   - `k_proj` (weights `[kv_heads*head_dim, d_model/2]`, scales, biases)
   - `v_proj` (weights `[kv_heads*head_dim, d_model/2]`, scales, biases)
   - `o_proj` (weights `[d_model, (q_heads*head_dim)/2]`, scales, biases)

3. **Feedforward Sub-Block (SwiGLU):**
   - `gate_proj` (weights `[d_ff, d_model/2]`, scales `[d_ff, d_model/64]`, biases `[d_ff, d_model/64]`)
   - `up_proj` (weights `[d_ff, d_model/2]`, scales `[d_ff, d_model/64]`, biases `[d_ff, d_model/64]`)
   - `down_proj` (weights `[d_model, d_ff/2]`, scales `[d_model, d_ff/64]`, biases `[d_model, d_ff/64]`)

---

## 4. Versioning

### 4.1 v3 — 16-byte alignment of packed-weight blocks

`GemvInt4` loads weights four 32-bit words at a time (`ByteAddressBuffer.Load4`), which requires
the packed-weight block to be 16-byte aligned. Through v2 it was not: the four RMSNorm weights
and the q/k norms are all multiples of 16 bytes, but `layer_scalar` is 2 bytes, so every
projection in every layer sat at offset 2 (mod 16) and every `Load4` was misaligned.

**A v2 container MUST be rejected, not read.** Unlike every earlier field addition — which
appended to `reserved` so that a zeroed field read as "unspecified" — this is a layout change.
A v2 container read by a v3 implementation decodes at the wrong offsets and produces
plausible-looking garbage rather than failing. `validate_header` enforces this.

### 4.2 Where this layout is written down

The per-layer layout is spelled out in five places, and they must be changed together. Updating
one in isolation is not hypothetical: it is how the v3 change was first made, and the CPU
reference oracle then silently graded the GPU against the wrong bytes.

| | |
|---|---|
| `src/runner.cpp` | `setup_proj` — the GPU offsets |
| `src/cpu_reference.cpp` | `parse_quant_block` — the oracle |
| `tools/convert_hf_to_g4dense.py` | the writer |
| `tools/make_synthetic_model.py` | the `tiny.g4dense` test fixture |
| `docs/G4DENSE_FORMAT.md` | this document |

`tools/verify_weights.py` checks the header and payload SHA-256 only; it does not walk the
per-layer layout, so it needs the version constant but not the padding rule.
