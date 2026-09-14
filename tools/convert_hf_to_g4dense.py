"""
Converts Hugging Face / MLX safetensors checkpoints of Gemma 4 Dense models
(31B or E2B) into an APU-optimized .g4dense v2 binary container.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
import numpy as np

HEADER_SIZE = 4096
ALIGNMENT = 4096
MAGIC = 0x4734444E  # 'G4DN'
VERSION = 3  # 3: packed-weight blocks are 16-byte aligned within each layer


def pad_to_alignment(data: bytes, alignment: int = ALIGNMENT) -> bytes:
    rem = len(data) % alignment
    if rem != 0:
        data += b'\x00' * (alignment - rem)
    return data


def load_safetensors_metadata(file_path: str):
    with open(file_path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"Invalid safetensors file: {file_path}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_json = f.read(header_len).decode("utf-8")
        return json.loads(header_json), 8 + header_len


def convert_checkpoint(input_dir: str, output_file: str, verify: bool = True):
    in_path = Path(input_dir)
    config_file = in_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing config.json in {input_dir}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Read from text_config if present
    text_cfg = config.get("text_config", config)

    # Validate architecture fields
    arch = config.get("architectures", ["Gemma4ForConditionalGeneration"])[0]
    is_muse = (arch in ["MuseGlimmerForConditionalGeneration", "MuseForConditionalGeneration"] or
               config.get("model_type") == "muse_glimmer")
    arch_type = 1 if is_muse else 0

    num_layers = int(text_cfg.get("num_hidden_layers", 52 if is_muse else 60))
    d_model = int(text_cfg.get("hidden_size", 6656 if is_muse else 5376))
    d_ff = int(text_cfg.get("intermediate_size", 19968 if is_muse else 21504))
    num_q_heads = int(text_cfg.get("num_attention_heads", 32))
    num_kv_heads = int(text_cfg.get("num_key_value_heads", 2 if is_muse else 16))
    head_dim = int(text_cfg.get("head_dim", 128 if is_muse else 256))
    attn_out_dim = num_q_heads * head_dim if is_muse else d_model
    global_head_dim = head_dim if is_muse else int(text_cfg.get("global_head_dim", 512))
    _gkv = text_cfg.get("num_global_key_value_heads")
    global_kv_heads = int(_gkv) if _gkv else num_kv_heads
    vocab_size = int(text_cfg.get("vocab_size", 202048 if is_muse else 262144))
    sliding_window = int(text_cfg.get("sliding_window", 2048 if is_muse else 1024))
    quant_group_size = 64
    final_logit_softcap = float(text_cfg.get("final_logit_softcapping", 20.0 if is_muse else 30.0))
    qk_scale_factor = float(text_cfg.get("qk_scale_factor", 3.87 if is_muse else 1.0))
    output_multiplier = float(text_cfg.get("output_multiplier", 0.196116 if is_muse else 1.0))
    tied_embeddings = not is_muse and bool(text_cfg.get("tie_word_embeddings", True))

    rope_params = text_cfg.get("rope_parameters", {})
    rope_theta_local = float(rope_params.get("sliding_attention", {}).get("rope_theta", 500000.0 if is_muse else 10000.0))
    rope_theta_global = float(rope_params.get("full_attention", {}).get("rope_theta", 0.0 if is_muse else 1000000.0))

    # Derive global layer mask directly from checkpoint's layer_types
    layer_types = text_cfg.get("layer_types", [])
    if not layer_types:
        raise ValueError(f"Missing 'layer_types' in checkpoint config at {config_file}")

    # Per-layer embeddings and KV sharing are the E2B/E4B architecture. They used to be
    # rejected here; they are supported now, and the container records enough to read them
    # back. A model with neither is written byte-for-byte as before.
    ple_dim = int(text_cfg.get("hidden_size_per_layer_input", 0) or 0)
    ple_vocab = int(text_cfg.get("vocab_size_per_layer_input", 0) or 0)
    kv_shared = int(text_cfg.get("num_kv_shared_layers", 0) or 0)
    first_shared_layer = num_layers - kv_shared if kv_shared else num_layers

    moe_declared = text_cfg.get("moe_layers") or text_cfg.get("num_experts")
    if moe_declared:
        raise ValueError(
            f"unsupported architecture: this checkpoint declares MoE blocks ({moe_declared}). "
            "This engine is the dense variant; the sibling MoE project handles those."
        )

    global_layer_mask = 0
    full_attn_indices = []
    for idx, ltype in enumerate(layer_types):
        if ltype == "full_attention":
            full_attn_indices.append(idx)
            if idx < 64:
                global_layer_mask |= (1 << idx)

    print(f"Converting checkpoint: {input_dir}")
    print(f"  Architecture: {arch} ({num_layers} layers, d_model={d_model}, d_ff={d_ff}, vocab={vocab_size})")
    print(f"  Layer Types: {len(layer_types)} total, {len(full_attn_indices)} global attention blocks {full_attn_indices}")
    print(f"  Global Layer Mask: 0x{global_layer_mask:016X}")
    print(f"  Sliding geometry: head_dim={head_dim} kv_heads={num_kv_heads}  |  "
          f"Global geometry: head_dim={global_head_dim} kv_heads={global_kv_heads}")

    # Locate safetensors files
    st_files = sorted(list(in_path.glob("*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {input_dir}")

    print(f"  Found {len(st_files)} safetensors shard(s)")

    # Build tensor index
    tensor_index = {}
    for st_path in st_files:
        meta, data_offset = load_safetensors_metadata(str(st_path))
        for tname, info in meta.items():
            if tname == "__metadata__":
                continue
            tensor_index[tname] = {
                "file": st_path,
                "data_offset": data_offset,
                "offsets": info["data_offsets"],
                "shape": info["shape"],
                "dtype": info["dtype"]
            }

    def find_key(tname: str):
        candidates = [
            tname,
            f"model.{tname}",
            f"language_model.{tname}",
            f"model.language_model.{tname}"
        ]
        for c in candidates:
            if c in tensor_index:
                return c
        return None

    def has_tensor(tname: str) -> bool:
        return find_key(tname) is not None

    def read_tensor_bytes(tname: str, optional: bool = False) -> bytes:
        key = find_key(tname)
        if key is None:
            if optional:
                return b""
            raise KeyError(f"Tensor {tname} not found in checkpoint shards")

        info = tensor_index[key]
        start, end = info["offsets"]
        length = end - start
        with open(info["file"], "rb") as f:
            f.seek(info["data_offset"] + start)
            data = f.read(length)
            if len(data) != length:
                raise IOError(f"Truncated read for tensor {key}")
            return data

    # Assemble Embeddings + Final RMSNorm
    print("  Packing embeddings and final RMSNorm...")
    embed_bytes = bytearray()
    embed_bytes += read_tensor_bytes("embed_tokens.weight")
    embed_bytes += read_tensor_bytes("embed_tokens.scales", optional=True)
    embed_bytes += read_tensor_bytes("embed_tokens.biases", optional=True)
    embed_bytes += read_tensor_bytes("norm.weight")
    embed_payload = pad_to_alignment(bytes(embed_bytes))

    # Untied LM head (Muse-Glimmer)
    lm_head_payload = b""
    if not tied_embeddings:
        print("  Packing untied LM head...")
        lh_bytes = bytearray()
        lh_bytes += read_tensor_bytes("lm_head.weight")
        lh_bytes += read_tensor_bytes("lm_head.scales", optional=True)
        lh_bytes += read_tensor_bytes("lm_head.biases", optional=True)
        lm_head_payload = pad_to_alignment(bytes(lh_bytes))

    # Model-level PLE tensors
    ple_payload = b""
    if ple_dim:
        print(f"  Packing per-layer embeddings (ple_dim={ple_dim}, vocab={ple_vocab})...")
        pb = bytearray()
        pb += read_tensor_bytes("model.embed_tokens_per_layer.weight")
        pb += read_tensor_bytes("model.embed_tokens_per_layer.scales")
        pb += read_tensor_bytes("model.embed_tokens_per_layer.biases")
        pb += read_tensor_bytes("model.per_layer_model_projection.weight")
        pb += read_tensor_bytes("model.per_layer_model_projection.scales")
        pb += read_tensor_bytes("model.per_layer_model_projection.biases")
        pb += read_tensor_bytes("model.per_layer_projection_norm.weight")
        ple_payload = pad_to_alignment(bytes(pb))

    # Assemble Layers 0..num_layers-1
    print(f"  Packing {num_layers} transformer blocks...")
    layer_d_ff = [0] * 60
    layer_payloads = []
    for l in range(num_layers):
        l_bytes = bytearray()
        prefix = f"layers.{l}."
        is_global = (layer_types[l] == "full_attention")

        # Norms + Layer Scalar (BF16)
        shared_kv = l >= first_shared_layer

        norm_names = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ]
        if not is_muse:
            norm_names.append("self_attn.q_norm.weight")
            if not shared_kv:
                norm_names.append("self_attn.k_norm.weight")
            norm_names.append("layer_scalar")
        if ple_dim:
            norm_names.append("post_per_layer_input_norm.weight")
        for n in norm_names:
            l_bytes += read_tensor_bytes(prefix + n)

        # Projections
        proj_names = [("self_attn.q_proj", False)]
        if not shared_kv:
            proj_names.append(("self_attn.k_proj", False))
            proj_names.append(("self_attn.v_proj", not is_muse))
        if is_muse:
            proj_names.append(("self_attn.gate_proj", False))
        proj_names.extend([
            ("self_attn.o_proj", False),
            ("mlp.gate_proj", False),
            ("mlp.up_proj", False),
            ("mlp.down_proj", False),
        ])
        if ple_dim:
            proj_names.extend([
                ("per_layer_input_gate", False),
                ("per_layer_projection", False),
            ])
        if l < 60:
            gate_key = find_key(f"{prefix}mlp.gate_proj.weight")
            gate_shape = tensor_index.get(gate_key) if gate_key else None
            layer_d_ff[l] = int(gate_shape["shape"][0]) if gate_shape else d_ff

        for p, opt in proj_names:
            if opt and not has_tensor(prefix + p + ".weight"):
                continue
            while len(l_bytes) % 16 != 0:
                l_bytes += b"\x00"
            l_bytes += read_tensor_bytes(prefix + p + ".weight")
            l_bytes += read_tensor_bytes(prefix + p + ".scales", optional=True)
            l_bytes += read_tensor_bytes(prefix + p + ".biases", optional=True)

        layer_payloads.append(pad_to_alignment(bytes(l_bytes)))

    # Compute Offsets (Support up to 60 layers)
    embed_offset = HEADER_SIZE
    embed_size = len(embed_payload)
    if tied_embeddings:
        lm_head_offset = embed_offset
        lm_head_size = embed_size
        cur_offset = HEADER_SIZE + embed_size + len(ple_payload)
    else:
        lm_head_offset = embed_offset + embed_size
        lm_head_size = len(lm_head_payload)
        cur_offset = lm_head_offset + lm_head_size + len(ple_payload)

    ple_offset = cur_offset - len(ple_payload) if ple_payload else 0
    ple_size = len(ple_payload)

    layer_offsets = [0] * 60
    layer_sizes = [0] * 60
    for l in range(num_layers):
        if l < 60:
            layer_offsets[l] = cur_offset
            layer_sizes[l] = len(layer_payloads[l])
            cur_offset += len(layer_payloads[l])

    # Build Header
    print("  Computing payload SHA-256 and writing container...")
    hasher = hashlib.sha256()
    hasher.update(embed_payload)
    if lm_head_payload:
        hasher.update(lm_head_payload)
    hasher.update(ple_payload)
    for lp in layer_payloads:
        hasher.update(lp)
    payload_sha256 = hasher.digest()

    header_fmt = "<IIIIIIIIIIIIIIQffffQQQQ60Q60Q32sIIIIIIQQ60IIIff2696s"
    header_bytes = struct.pack(
        header_fmt,
        MAGIC,
        VERSION,
        1,  # Affine INT4 G64
        num_layers,
        d_model,
        d_ff,
        num_q_heads,
        num_kv_heads,
        head_dim,
        vocab_size,
        sliding_window,
        quant_group_size,
        1,  # BF16 scales
        1 if tied_embeddings else 0,
        global_layer_mask,
        rope_theta_local,
        rope_theta_global,
        1.0,
        final_logit_softcap,
        embed_offset,
        embed_size,
        lm_head_offset,
        lm_head_size,
        *layer_offsets,
        *layer_sizes,
        payload_sha256,
        global_head_dim,
        global_kv_heads,
        ple_dim,
        ple_vocab,
        kv_shared,
        0,              # ple_reserved
        ple_offset,
        ple_size,
        *layer_d_ff,
        arch_type,
        attn_out_dim,
        qk_scale_factor,
        output_multiplier,
        b'\x00' * 2696
    )

    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        f.write(header_bytes)
        f.write(embed_payload)
        if lm_head_payload:
            f.write(lm_head_payload)
        f.write(ple_payload)
        for lp in layer_payloads:
            f.write(lp)

    total_size = HEADER_SIZE + len(embed_payload) + len(ple_payload) + sum(len(lp) for lp in layer_payloads)
    print(f"Successfully converted {output_file} ({total_size / (1024*1024):.2f} MB, {total_size:,} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Input Hugging Face checkpoint directory")
    ap.add_argument("--out", required=True, help="Output .g4dense file path")
    args = ap.parse_args()

    convert_checkpoint(args.input, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
