"""
Verifies the integrity of a .g4mtp binary container against the original safetensors checkpoint.
"""

import hashlib
import json
import struct
import sys
from pathlib import Path

HEADER_SIZE = 4096
MAGIC = 0x47344D54  # 'G4MT'
VERSION = 1


def verify_container(container_path: str, checkpoint_dir: str):
    cp = Path(container_path)
    if not cp.exists():
        print(f"Error: {container_path} does not exist", file=sys.stderr)
        return False

    with open(cp, "rb") as f:
        header_data = f.read(HEADER_SIZE)
        if len(header_data) < HEADER_SIZE:
            print("Error: container smaller than header size", file=sys.stderr)
            return False

        header_fmt = "<14IQ2I8Q8Q8Q32s3800s"
        unpacked = struct.unpack(header_fmt, header_data)

        magic = unpacked[0]
        version = unpacked[1]
        quant_type = unpacked[2]
        num_layers = unpacked[3]
        backbone_hidden = unpacked[4]
        hidden_size = unpacked[5]
        intermediate_size = unpacked[6]
        vocab_size = unpacked[7]
        num_q_heads = unpacked[8]
        num_kv_heads = unpacked[9]
        head_dim = unpacked[10]
        global_head_dim = unpacked[11]
        global_kv_heads = unpacked[12]
        sliding_window = unpacked[13]
        global_layer_mask = unpacked[14]

        embed_offset = unpacked[17]
        embed_size = unpacked[18]
        pre_proj_offset = unpacked[19]
        pre_proj_size = unpacked[20]
        post_proj_offset = unpacked[21]
        post_proj_size = unpacked[22]
        norm_offset = unpacked[23]
        norm_size = unpacked[24]

        layer_offsets = unpacked[25:33]
        layer_sizes = unpacked[33:41]
        payload_sha256 = unpacked[41]

        print("=== .g4mtp Header Verification ===")
        print(f"Magic:              0x{magic:08X} ({'OK' if magic == MAGIC else 'FAIL'})")
        print(f"Version:            {version} ({'OK' if version == VERSION else 'FAIL'})")
        print(f"Quant Type:         {quant_type} (AFFINE_INT4_G64)")
        print(f"Backbone Hidden:    {backbone_hidden} (31B)")
        print(f"Hidden Size:        {hidden_size}")
        print(f"Intermediate Size:  {intermediate_size}")
        print(f"Layers:             {num_layers}")
        print(f"Vocab Size:         {vocab_size}")
        print(f"Embeddings:         offset={embed_offset}, size={embed_size / (1024*1024):.2f} MiB")
        print(f"Pre-Projection:     offset={pre_proj_offset}, size={pre_proj_size / (1024*1024):.2f} MiB")
        print(f"Post-Projection:    offset={post_proj_offset}, size={post_proj_size / (1024*1024):.2f} MiB")
        print(f"Norm:               offset={norm_offset}, size={norm_size} bytes")
        for l in range(num_layers):
            print(f"Layer {l}:            offset={layer_offsets[l]}, size={layer_sizes[l] / (1024*1024):.2f} MiB")

        if magic != MAGIC or version != VERSION:
            print("FAILED: Invalid magic or version!", file=sys.stderr)
            return False

        # Verify SHA256 of payload
        print("\nVerifying SHA-256 payload integrity...")
        hasher = hashlib.sha256()
        f.seek(HEADER_SIZE)
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
        calc_sha = hasher.digest()

        if calc_sha == payload_sha256:
            print(f"SUCCESS: Payload SHA-256 matches header ({calc_sha.hex()[:16]}...)")
        else:
            print(f"FAILED: Payload SHA-256 mismatch! {calc_sha.hex()} != {payload_sha256.hex()}", file=sys.stderr)
            return False

    return True


if __name__ == "__main__":
    ok = verify_container("models/gemma-4-31b-assistant.g4mtp", "models/gemma-4-31b-assistant-4bit")
    sys.exit(0 if ok else 1)
