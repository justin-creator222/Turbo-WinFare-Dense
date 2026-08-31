"""
Generates a deterministic synthetic tiny .g4dense v3 model for fast testing.
4 layers, d_model 256, d_ff 512, 4/2 heads, vocab 1024.
"""

import argparse
import hashlib
import json
import struct
import sys
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADER_SIZE = 4096
ALIGNMENT = 4096

MAGIC = 0x4734444E  # 'G4DN'
VERSION = 3  # 3: packed-weight blocks are 16-byte aligned within each layer
QUANT_TYPE_AFFINE_INT4_G64 = 1


def pad_to_alignment(data: bytes, alignment: int = ALIGNMENT) -> bytes:
    remainder = len(data) % alignment
    if remainder != 0:
        data += b'\x00' * (alignment - remainder)
    return data


def quantize_affine_int4_g64(weights: np.ndarray, group_size: int = 64):
    """
    Quantizes 2D float32 weights [rows, cols] to affine INT4 (group_size=64).
    Returns (packed_u32, scales_bf16, biases_bf16)
    """
    rows, cols = weights.shape
    assert cols % group_size == 0, f"cols {cols} must be divisible by {group_size}"
    num_groups = cols // group_size

    # Reshape into groups
    w_grouped = weights.reshape(rows, num_groups, group_size)
    w_min = w_grouped.min(axis=-1, keepdims=True)
    w_max = w_grouped.max(axis=-1, keepdims=True)

    scales = (w_max - w_min) / 15.0
    scales = np.where(scales == 0, 1.0, scales)
    biases = w_min

    # Quantize to 0..15
    q = np.clip(np.round((w_grouped - biases) / scales), 0, 15).astype(np.uint8)
    q_flat = q.reshape(rows, cols)

    # Pack low-nibble-first into uint32 (8 nibbles per uint32)
    assert cols % 8 == 0
    packed_cols = cols // 8
    packed = np.zeros((rows, packed_cols), dtype=np.uint32)

    for i in range(8):
        nibbles = q_flat[:, i::8].astype(np.uint32)
        packed |= (nibbles & 0xF) << (i * 4)

    # Convert scales and biases to BF16
    def to_bf16_bytes(arr: np.ndarray) -> bytes:
        f32_u32 = arr.astype(np.float32).view(np.uint32)
        bf16_u16 = (f32_u32 >> 16).astype(np.uint16)
        return bf16_u16.tobytes()

    scales_flat = scales.reshape(rows, num_groups)
    biases_flat = biases.reshape(rows, num_groups)

    return packed.tobytes(), to_bf16_bytes(scales_flat), to_bf16_bytes(biases_flat)


def make_synthetic_model(out_path: str, seed: int = 42):
    np.random.seed(seed)

    num_layers = 4
    d_model = 256
    d_ff = 512
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    vocab_size = 1024
    sliding_window = 512
    quant_group_size = 64
    global_layer_mask = 0

    def to_bf16_bytes(arr: np.ndarray) -> bytes:
        """Quantization scales and biases. These really are BF16 in the real container."""
        f32_u32 = arr.astype(np.float32).view(np.uint32)
        bf16_u16 = (f32_u32 >> 16).astype(np.uint16)
        return bf16_u16.tobytes()

    def to_norm_bytes(arr: np.ndarray) -> bytes:
        """LayerNorm-family weights: input/post_attn/pre_ffn/post_ffn norms, q_norm, k_norm,
        the final model norm, and the per-layer scalar.

        BF16, same as the scales -- every non-quantized tensor in the real MLX container is
        BF16, exactly as its safetensors header says. A previous round wrote these as IEEE
        FP16 here to match a reader that had been changed to decode norms as FP16; both were
        wrong, and the fixture has to match the real container or it stops being a valid
        stand-in.
        """
        return to_bf16_bytes(arr)

    # Generate Embeddings [vocab_size, d_model] + Final RMSNorm [d_model]
    w_embed = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
    embed_packed, embed_scales, embed_biases = quantize_affine_int4_g64(w_embed, quant_group_size)
    norm_final = np.ones(d_model, dtype=np.float32)
    embed_bytes = embed_packed + embed_scales + embed_biases + to_norm_bytes(norm_final)
    embed_bytes = pad_to_alignment(embed_bytes)

    # Generate Layers 0..3
    layer_data_list = []
    layer_weights = []

    for l in range(num_layers):
        # Attention Projections
        w_q = np.random.randn(num_q_heads * head_dim, d_model).astype(np.float32) * 0.02
        w_k = np.random.randn(num_kv_heads * head_dim, d_model).astype(np.float32) * 0.02
        w_v = np.random.randn(num_kv_heads * head_dim, d_model).astype(np.float32) * 0.02
        w_o = np.random.randn(d_model, num_q_heads * head_dim).astype(np.float32) * 0.02

        # FFN Projections
        w_gate = np.random.randn(d_ff, d_model).astype(np.float32) * 0.02
        w_up = np.random.randn(d_ff, d_model).astype(np.float32) * 0.02
        w_down = np.random.randn(d_model, d_ff).astype(np.float32) * 0.02

        # Norms
        norm_in = np.ones(d_model, dtype=np.float32)
        norm_post_attn = np.ones(d_model, dtype=np.float32)
        norm_pre_ffn = np.ones(d_model, dtype=np.float32)
        norm_post_ffn = np.ones(d_model, dtype=np.float32)
        norm_q = np.ones(head_dim, dtype=np.float32)
        norm_k = np.ones(head_dim, dtype=np.float32)
        layer_scalar = np.array([1.0 / np.sqrt(2.0 * num_layers)], dtype=np.float32)

        # Assemble Layer Payload
        ldata = bytearray()
        # Norms + Layer Scalar
        ldata += to_norm_bytes(norm_in)
        ldata += to_norm_bytes(norm_post_attn)
        ldata += to_norm_bytes(norm_pre_ffn)
        ldata += to_norm_bytes(norm_post_ffn)
        ldata += to_norm_bytes(norm_q)
        ldata += to_norm_bytes(norm_k)
        ldata += to_bf16_bytes(layer_scalar)   # genuinely BF16, not a norm weight

        # Q, K, V, O, Gate, Up, Down
        for w in [w_q, w_k, w_v, w_o, w_gate, w_up, w_down]:
            # 16-byte align each packed-weight block, matching convert_hf_to_g4dense.py and the
            # runner's setup_proj. The gemv loads weights four words at a time.
            while len(ldata) % 16 != 0:
                ldata += b"\x00"
            p, s, b = quantize_affine_int4_g64(w, quant_group_size)
            ldata += p + s + b

        layer_bytes = pad_to_alignment(bytes(ldata))
        layer_data_list.append(layer_bytes)
        layer_weights.append({
            "w_q": w_q, "w_k": w_k, "w_v": w_v, "w_o": w_o,
            "w_gate": w_gate, "w_up": w_up, "w_down": w_down,
            "norm_in": norm_in, "norm_post_attn": norm_post_attn,
            "norm_pre_ffn": norm_pre_ffn, "norm_post_ffn": norm_post_ffn,
            "norm_q": norm_q, "norm_k": norm_k
        })

    # Assemble complete container
    payload = bytearray()
    embed_offset = HEADER_SIZE
    embed_size = len(embed_bytes)
    payload += embed_bytes

    layer_offsets = [0] * 60
    layer_sizes = [0] * 60

    cur_offset = HEADER_SIZE + len(embed_bytes)
    for l in range(num_layers):
        layer_offsets[l] = cur_offset
        layer_sizes[l] = len(layer_data_list[l])
        payload += layer_data_list[l]
        cur_offset += len(layer_data_list[l])

    payload_bytes = bytes(payload)
    sha256_hash = hashlib.sha256(payload_bytes).digest()

    # Build Header (4096 bytes)
    header_fmt = "<IIIIIIIIIIIIIIQffffQQQQ60Q60Q32s2992s"
    header_bytes = struct.pack(
        header_fmt,
        MAGIC,
        VERSION,
        QUANT_TYPE_AFFINE_INT4_G64,
        num_layers,
        d_model,
        d_ff,
        num_q_heads,
        num_kv_heads,
        head_dim,
        vocab_size,
        sliding_window,
        quant_group_size,
        1,  # scale_dtype = BF16
        1,  # tied_embeddings
        global_layer_mask,
        10000.0,    # rope_theta_local
        1000000.0,  # rope_theta_global
        1.0,        # rope_scaling
        30.0,       # final_logit_softcapping
        embed_offset,
        embed_size,
        embed_offset, # lm_head_offset tied
        embed_size,   # lm_head_size tied
        *layer_offsets,
        *layer_sizes,
        sha256_hash,
        b'\x00' * 2992
    )

    assert len(header_bytes) == HEADER_SIZE, f"Header size {len(header_bytes)} != 4096"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header_bytes)
        f.write(payload_bytes)

    print(f"Wrote synthetic model: {out_path} ({len(header_bytes) + len(payload_bytes):,} bytes)")
    return {
        "w_embed": w_embed,
        "layers": layer_weights
    }


def verify_synthetic_model(file_path: str):
    print(f"Verifying {file_path}...")
    with open(file_path, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        if len(header_bytes) != HEADER_SIZE:
            raise RuntimeError(f"Header too short: {len(header_bytes)}")

        header_fmt = "<IIIIIIIIIIIIIIQffffQQQQ60Q60Q32s2992s"
        unpacked = struct.unpack(header_fmt, header_bytes)

        magic, version, qtype, num_layers, d_model, d_ff, q_heads, kv_heads, head_dim, vocab_size, sw, gsize, s_dtype, tied, mask, r_local, r_global, r_scale, softcap, e_off, e_sz, lm_off, lm_sz = unpacked[:23]
        layer_offsets = unpacked[23:83]
        layer_sizes = unpacked[83:143]
        declared_sha = unpacked[143]

        if magic != MAGIC: raise RuntimeError(f"Bad magic 0x{magic:x}")
        if version != VERSION: raise RuntimeError(f"Bad version {version}")
        if num_layers != 4: raise RuntimeError(f"Expected 4 layers, got {num_layers}")

        payload = f.read()
        calc_sha = hashlib.sha256(payload).digest()
        if declared_sha != calc_sha:
            raise RuntimeError(f"SHA-256 mismatch! declared={declared_sha.hex()}, calc={calc_sha.hex()}")

    print(f"VERIFICATION SUCCESSFUL: {file_path} is structurally valid G4Dense v{VERSION} container.")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="tests/fixtures/tiny.g4dense", help="output path")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--verify", action="store_true", help="verify existing file")
    args = ap.parse_args()

    if args.verify:
        ok = verify_synthetic_model(args.out)
        return 0 if ok else 1

    make_synthetic_model(args.out, args.seed)
    if not verify_synthetic_model(args.out):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
