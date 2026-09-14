"""
Converts Hugging Face / MLX safetensors checkpoints of Gemma 4 31B MTP Assistant
into an APU-optimized .g4mtp v1 binary container.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

HEADER_SIZE = 4096
ALIGNMENT = 4096
MAGIC = 0x47344D54  # 'G4MT'
VERSION = 1


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


def convert_assistant_checkpoint(input_dir: str, output_file: str):
    in_path = Path(input_dir)
    config_file = in_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing config.json in {input_dir}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    text_cfg = config.get("text_config", config)

    arch = config.get("architectures", ["Gemma4AssistantForCausalLM"])[0]
    backbone_hidden_size = int(config.get("backbone_hidden_size", 5376))
    num_layers = int(text_cfg.get("num_hidden_layers", 4))
    hidden_size = int(text_cfg.get("hidden_size", 1024))
    intermediate_size = int(text_cfg.get("intermediate_size", 8192))
    vocab_size = int(text_cfg.get("vocab_size", 262144))
    num_q_heads = int(text_cfg.get("num_attention_heads", 32))
    num_kv_heads = int(text_cfg.get("num_key_value_heads", 16))
    head_dim = int(text_cfg.get("head_dim", 256))
    global_head_dim = int(text_cfg.get("global_head_dim", 512))
    global_kv_heads = int(text_cfg.get("num_global_key_value_heads", 4))
    sliding_window = int(text_cfg.get("sliding_window", 1024))
    quant_group_size = 64

    layer_types = text_cfg.get("layer_types", ["sliding_attention"] * (num_layers - 1) + ["full_attention"])
    global_layer_mask = 0
    for idx, ltype in enumerate(layer_types):
        if ltype == "full_attention":
            global_layer_mask |= (1 << idx)

    print(f"Converting Gemma 4 Assistant checkpoint: {input_dir}")
    print(f"  Architecture: {arch} ({num_layers} layers, hidden={hidden_size}, backbone={backbone_hidden_size})")
    print(f"  Layer Types: {layer_types}")

    st_files = sorted(list(in_path.glob("*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {input_dir}")

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
            raise KeyError(f"Tensor {tname} not found in checkpoint")

        info = tensor_index[key]
        start, end = info["offsets"]
        length = end - start
        with open(info["file"], "rb") as f:
            f.seek(info["data_offset"] + start)
            data = f.read(length)
            if len(data) != length:
                raise IOError(f"Truncated read for tensor {key}")
            return data

    # 1. Embeddings & tied LM head
    print("  Packing embed_tokens...")
    embed_bytes = bytearray()
    embed_bytes += read_tensor_bytes("model.embed_tokens.weight")
    embed_bytes += read_tensor_bytes("model.embed_tokens.scales")
    embed_bytes += read_tensor_bytes("model.embed_tokens.biases")
    embed_payload = pad_to_alignment(bytes(embed_bytes))

    # 2. Pre-projection (1024 x 10752)
    print("  Packing pre_projection...")
    pre_bytes = bytearray()
    pre_bytes += read_tensor_bytes("pre_projection.weight")
    pre_bytes += read_tensor_bytes("pre_projection.scales")
    pre_bytes += read_tensor_bytes("pre_projection.biases")
    pre_payload = pad_to_alignment(bytes(pre_bytes))

    # 3. Post-projection (5376 x 1024)
    print("  Packing post_projection...")
    post_bytes = bytearray()
    post_bytes += read_tensor_bytes("post_projection.weight")
    post_bytes += read_tensor_bytes("post_projection.scales")
    post_bytes += read_tensor_bytes("post_projection.biases")
    post_payload = pad_to_alignment(bytes(post_bytes))

    # 4. Final Norm
    print("  Packing final norm...")
    norm_bytes = bytearray()
    norm_bytes += read_tensor_bytes("model.norm.weight")
    norm_payload = pad_to_alignment(bytes(norm_bytes))

    # 5. Layers 0..3
    print(f"  Packing {num_layers} assistant transformer blocks...")
    layer_payloads = []
    for l in range(num_layers):
        l_bytes = bytearray()
        prefix = f"model.layers.{l}."

        norm_names = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_norm.weight",
            "layer_scalar",
        ]
        for n in norm_names:
            l_bytes += read_tensor_bytes(prefix + n)

        proj_names = [
            "self_attn.q_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        for p in proj_names:
            while len(l_bytes) % 16 != 0:
                l_bytes += b"\x00"
            l_bytes += read_tensor_bytes(prefix + p + ".weight")
            l_bytes += read_tensor_bytes(prefix + p + ".scales")
            l_bytes += read_tensor_bytes(prefix + p + ".biases")

        layer_payloads.append(pad_to_alignment(bytes(l_bytes)))

    # Compute Offsets
    embed_offset = HEADER_SIZE
    embed_size = len(embed_payload)

    pre_proj_offset = embed_offset + embed_size
    pre_proj_size = len(pre_payload)

    post_proj_offset = pre_proj_offset + pre_proj_size
    post_proj_size = len(post_payload)

    norm_offset = post_proj_offset + post_proj_size
    norm_size = len(norm_payload)

    layer_offsets = [0] * 8
    layer_sizes = [0] * 8

    cur_offset = norm_offset + norm_size
    for l in range(num_layers):
        layer_offsets[l] = cur_offset
        layer_sizes[l] = len(layer_payloads[l])
        cur_offset += len(layer_payloads[l])

    # SHA256 of all payloads
    hasher = hashlib.sha256()
    hasher.update(embed_payload)
    hasher.update(pre_payload)
    hasher.update(post_payload)
    hasher.update(norm_payload)
    for lp in layer_payloads:
        hasher.update(lp)
    payload_sha256 = hasher.digest()

    # Header layout: 296 bytes used, 3800 bytes reserved -> 4096 total
    header_fmt = "<14IQ2I8Q8Q8Q32s3800s"
    reserved_bytes = b"\x00" * 3800

    header_bytes = struct.pack(
        header_fmt,
        MAGIC,
        VERSION,
        1,  # AFFINE_INT4_G64
        num_layers,
        backbone_hidden_size,
        hidden_size,
        intermediate_size,
        vocab_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        global_head_dim,
        global_kv_heads,
        sliding_window,
        global_layer_mask,
        quant_group_size,
        1,  # BF16 scales
        embed_offset, embed_size,
        pre_proj_offset, pre_proj_size,
        post_proj_offset, post_proj_size,
        norm_offset, norm_size,
        *layer_offsets,
        *layer_sizes,
        payload_sha256,
        reserved_bytes
    )
    assert len(header_bytes) == HEADER_SIZE, f"Header size mismatch: {len(header_bytes)} != {HEADER_SIZE}"

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header_bytes)
        f.write(embed_payload)
        f.write(pre_payload)
        f.write(post_payload)
        f.write(norm_payload)
        for lp in layer_payloads:
            f.write(lp)

    total_size = out_path.stat().st_size
    print(f"\nContainer written successfully: {output_file}")
    print(f"  Total size: {total_size / (1024*1024):.2f} MiB ({total_size} bytes)")
    print(f"  Payload SHA-256: {payload_sha256.hex()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Gemma 4 Assistant safetensors to .g4mtp")
    parser.add_argument("--input-dir", default="models/gemma-4-31b-assistant-4bit", help="Path to checkpoint directory")
    parser.add_argument("--output-file", default="models/gemma-4-31b-assistant.g4mtp", help="Output .g4mtp path")
    args = parser.parse_args()
    convert_assistant_checkpoint(args.input_dir, args.output_file)
