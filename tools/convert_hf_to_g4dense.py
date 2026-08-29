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

    # Validate Gemma 4 architecture fields
    arch = config.get("architectures", ["Gemma4ForConditionalGeneration"])[0]
    num_layers = config.get("num_hidden_layers", 60)
    d_model = config.get("hidden_size", 5376)
    d_ff = config.get("intermediate_size", 21504)
    num_q_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", 16)
    head_dim = config.get("head_dim", 256)
    vocab_size = config.get("vocab_size", 262144)
    sliding_window = config.get("sliding_window", 1024)
    quant_group_size = 64
    final_logit_softcap = float(config.get("final_logit_softcapping", 30.0))
    rope_theta_local = 10000.0
    rope_theta_global = float(config.get("rope_theta", 1000000.0))

    # Global layer mask (10 layers for 60-layer model: 5, 11, 17, 23, 29, 35, 41, 47, 53, 59)
    global_layer_mask = 0
    full_attn_list = config.get("full_attention_layers", [5, 11, 17, 23, 29, 35, 41, 47, 53, 59])
    for l in full_attn_list:
        if l < 60:
            global_layer_mask |= (1 << l)

    print(f"Converting checkpoint: {input_dir}")
    print(f"  Architecture: {arch} ({num_layers} layers, d_model={d_model}, d_ff={d_ff}, vocab={vocab_size})")

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

    def read_tensor_bytes(tname: str) -> bytes:
        if tname not in tensor_index:
            raise KeyError(f"Tensor {tname} not found in checkpoint shards")
        info = tensor_index[tname]
        start, end = info["offsets"]
        length = end - start
        with open(info["file"], "rb") as f:
            f.seek(info["data_offset"] + start)
            data = f.read(length)
            if len(data) != length:
                raise IOError(f"Truncated read for tensor {tname}")
            return data

    # Assemble Embeddings
    print("  Packing embeddings...")
    embed_bytes = bytearray()
    embed_bytes += read_tensor_bytes("model.embed_tokens.weight")
    embed_bytes += read_tensor_bytes("model.embed_tokens.scales")
    embed_bytes += read_tensor_bytes("model.embed_tokens.biases")
    embed_payload = pad_to_alignment(bytes(embed_bytes))

    # Assemble Layers 0..num_layers-1
    print(f"  Packing {num_layers} transformer blocks...")
    layer_payloads = []
    for l in range(num_layers):
        l_bytes = bytearray()
        # Norms (BF16)
        prefix = f"model.layers.{l}."
        norm_names = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight"
        ]
        for n in norm_names:
            l_bytes += read_tensor_bytes(prefix + n)

        # Projections
        proj_names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj"
        ]
        for p in proj_names:
            l_bytes += read_tensor_bytes(prefix + p + ".weight")
            l_bytes += read_tensor_bytes(prefix + p + ".scales")
            l_bytes += read_tensor_bytes(prefix + p + ".biases")

        layer_payloads.append(pad_to_alignment(bytes(l_bytes)))

    # Compute Offsets
    embed_offset = HEADER_SIZE
    embed_size = len(embed_payload)
    layer_offsets = [0] * 60
    layer_sizes = [0] * 60

    cur_offset = HEADER_SIZE + embed_size
    for l in range(num_layers):
        layer_offsets[l] = cur_offset
        layer_sizes[l] = len(layer_payloads[l])
        cur_offset += len(layer_payloads[l])

    # Build Header
    print("  Computing payload SHA-256 and writing container...")
    hasher = hashlib.sha256()
    hasher.update(embed_payload)
    for lp in layer_payloads:
        hasher.update(lp)
    payload_sha256 = hasher.digest()

    header_fmt = "<IIIIIIIIIIIIIIQffffQQQQ60Q60Q32s2992s"
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
        b'\x00' * 2992
    )

    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        f.write(header_bytes)
        f.write(embed_payload)
        for lp in layer_payloads:
            f.write(lp)

    total_size = HEADER_SIZE + len(embed_payload) + sum(len(lp) for lp in layer_payloads)
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
