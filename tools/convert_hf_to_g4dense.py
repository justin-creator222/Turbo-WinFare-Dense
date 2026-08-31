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
VERSION = 2


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

    # Validate Gemma 4 architecture fields
    arch = config.get("architectures", ["Gemma4ForConditionalGeneration"])[0]
    num_layers = int(text_cfg.get("num_hidden_layers", 60))
    d_model = int(text_cfg.get("hidden_size", 5376))
    d_ff = int(text_cfg.get("intermediate_size", 21504))
    num_q_heads = int(text_cfg.get("num_attention_heads", 32))
    num_kv_heads = int(text_cfg.get("num_key_value_heads", 16))
    head_dim = int(text_cfg.get("head_dim", 256))
    global_head_dim = int(text_cfg.get("global_head_dim", 512))
    # Falls back to the sliding KV-head count: E2B leaves num_global_key_value_heads null
    # and its full-attention layers use the same single KV head (k_proj is 512 = 1 x 512).
    _gkv = text_cfg.get("num_global_key_value_heads")
    global_kv_heads = int(_gkv) if _gkv else num_kv_heads
    vocab_size = int(text_cfg.get("vocab_size", 262144))
    sliding_window = int(text_cfg.get("sliding_window", 1024))
    quant_group_size = 64
    final_logit_softcap = float(text_cfg.get("final_logit_softcapping", 30.0))

    rope_params = text_cfg.get("rope_parameters", {})
    rope_theta_local = float(rope_params.get("sliding_attention", {}).get("rope_theta", 10000.0))
    rope_theta_global = float(rope_params.get("full_attention", {}).get("rope_theta", 1000000.0))

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

    def has_tensor(tname: str) -> bool:
        return (tname in tensor_index) or (f"language_model.{tname}" in tensor_index)

    def read_tensor_bytes(tname: str, optional: bool = False) -> bytes:
        if tname in tensor_index:
            key = tname
        elif f"language_model.{tname}" in tensor_index:
            key = f"language_model.{tname}"
        else:
            if optional:
                return b""
            raise KeyError(f"Tensor {tname} (or language_model.{tname}) not found in checkpoint shards")

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
    embed_bytes += read_tensor_bytes("model.embed_tokens.weight")
    embed_bytes += read_tensor_bytes("model.embed_tokens.scales")
    embed_bytes += read_tensor_bytes("model.embed_tokens.biases")
    embed_bytes += read_tensor_bytes("model.norm.weight")
    embed_payload = pad_to_alignment(bytes(embed_bytes))

    # Model-level PLE tensors. embed_tokens_per_layer is a second embedding table holding
    # num_layers * ple_dim values per token; per_layer_model_projection maps the main embedding
    # into the same shape; per_layer_projection_norm normalizes the latter.
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
    # Measured feed-forward width per layer. E2B's last 20 layers are twice as wide as
    # intermediate_size claims, and config.json does not say so anywhere.
    layer_d_ff = [0] * 60
    layer_payloads = []
    for l in range(num_layers):
        l_bytes = bytearray()
        prefix = f"model.layers.{l}."
        is_global = (layer_types[l] == "full_attention")

        # Norms + Layer Scalar (BF16)
        shared_kv = l >= first_shared_layer

        # A KV-sharing layer keeps its own q_norm but has no k_norm, because it has no k_proj.
        norm_names = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_norm.weight",
        ]
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
            proj_names += [
                ("self_attn.k_proj", False),
                # v_proj is absent on full-attention layers when attention_k_eq_v is set
                ("self_attn.v_proj", True),
            ]
        proj_names += [
            ("self_attn.o_proj", False),
            ("mlp.gate_proj", False),
            ("mlp.up_proj", False),
            ("mlp.down_proj", False),
        ]
        if ple_dim:
            proj_names += [
                ("per_layer_input_gate", False),
                ("per_layer_projection", False),
            ]
        if l < 60:
            gate_shape = tensor_index.get(f"{prefix}mlp.gate_proj.weight") or \
                         tensor_index.get(f"language_model.{prefix}mlp.gate_proj.weight")
            layer_d_ff[l] = int(gate_shape["shape"][0]) if gate_shape else d_ff

        for p, opt in proj_names:
            if opt and not has_tensor(prefix + p + ".weight"):
                continue
            l_bytes += read_tensor_bytes(prefix + p + ".weight")
            l_bytes += read_tensor_bytes(prefix + p + ".scales")
            l_bytes += read_tensor_bytes(prefix + p + ".biases")

        layer_payloads.append(pad_to_alignment(bytes(l_bytes)))

    # Compute Offsets (Support up to 60 layers)
    embed_offset = HEADER_SIZE
    ple_offset = (embed_offset + len(embed_payload)) if ple_payload else 0
    ple_size = len(ple_payload)
    embed_size = len(embed_payload)
    layer_offsets = [0] * 60
    layer_sizes = [0] * 60

    cur_offset = HEADER_SIZE + embed_size + len(ple_payload)
    for l in range(num_layers):
        if l < 60:
            layer_offsets[l] = cur_offset
            layer_sizes[l] = len(layer_payloads[l])
            cur_offset += len(layer_payloads[l])

    # Build Header
    print("  Computing payload SHA-256 and writing container...")
    hasher = hashlib.sha256()
    hasher.update(embed_payload)
    hasher.update(ple_payload)
    for lp in layer_payloads:
        hasher.update(lp)
    payload_sha256 = hasher.digest()

    # The two uints after payload_sha256 are global_head_dim and global_kv_heads: the
    # full-attention layers' geometry, which differs from the sliding layers' (512 / 4
    # vs 256 / 16 on the 31B). They sit at the head of the former reserved block so
    # every preceding offset is unchanged and the struct stays 4096 bytes; a reader
    # that finds them zero treats the container as predating them.
    header_fmt = "<IIIIIIIIIIIIIIQffffQQQQ60Q60Q32sIIIIIIQQ60I2712s"
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
        1,  # Tied embeddings
        global_layer_mask,
        rope_theta_local,
        rope_theta_global,
        1.0,
        final_logit_softcap,
        embed_offset,
        embed_size,
        embed_offset, # LM Head tied
        embed_size,
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
        b'\x00' * 2712
    )

    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        f.write(header_bytes)
        f.write(embed_payload)
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
