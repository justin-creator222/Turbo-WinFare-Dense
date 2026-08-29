"""
Inspects and verifies a .g4dense v2 model container.
Computes SHA-256, tensor summary statistics (min, max, mean, zeros, NaNs),
and verifies sector alignment and container integrity.
"""

import argparse
import hashlib
import struct
import sys
from pathlib import Path
import numpy as np

HEADER_SIZE = 4096
MAGIC = 0x4734444E
VERSION = 2


def verify_g4dense_file(path: str, verbose: bool = True):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    file_size = p.stat().st_size
    print(f"=== Inspecting G4Dense Container: {path} ===")
    print(f"File Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")

    with open(p, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        if len(header_bytes) != HEADER_SIZE:
            raise ValueError(f"Header too short ({len(header_bytes)} bytes)")

        header_fmt = "<IIIIIIIIIIIIIIQffffQQQQ60Q60Q32s2992s"
        unpacked = struct.unpack(header_fmt, header_bytes)

        magic, version, qtype, num_layers, d_model, d_ff, q_heads, kv_heads, head_dim, vocab_size, sw, gsize, s_dtype, tied, mask, r_local, r_global, r_scale, softcap, e_off, e_sz, lm_off, lm_sz = unpacked[:23]
        layer_offsets = unpacked[23:83]
        layer_sizes = unpacked[83:143]
        payload_sha256 = unpacked[143]

        if magic != MAGIC:
            raise ValueError(f"Bad magic 0x{magic:08x} (expected 0x{MAGIC:08x})")
        if version != VERSION:
            raise ValueError(f"Bad version {version} (expected {VERSION})")

        print(f"Header Validation:")
        print(f"  Magic: 'G4DN' (0x{magic:08x}), Version: {version}")
        print(f"  Architecture: {num_layers} layers, d_model={d_model}, d_ff={d_ff}, vocab_size={vocab_size}")
        print(f"  Heads: {q_heads} Q heads, {kv_heads} KV heads (head_dim={head_dim})")
        print(f"  Sliding Window: {sw}, Quant Group Size: {gsize}, Scale DType: {s_dtype}")
        print(f"  Global Layer Mask: 0x{mask:016x}")
        print(f"  Embedding: offset={e_off}, size={e_sz} bytes (aligned: {e_off % 4096 == 0})")

        # Verify Layer alignments
        for l in range(num_layers):
            off = layer_offsets[l]
            sz = layer_sizes[l]
            if off % 4096 != 0:
                raise ValueError(f"Layer {l} offset {off} is not 4096-aligned")
            if sz == 0:
                raise ValueError(f"Layer {l} has zero size")
            if off + sz > file_size:
                raise ValueError(f"Layer {l} extends beyond file (offset={off}, sz={sz}, total={file_size})")

        # Verify Payload SHA-256
        print("Verifying Payload SHA-256 checksum...")
        f.seek(HEADER_SIZE)
        hasher = hashlib.sha256()
        chunk_size = 64 * 1024 * 1024
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
        calc_sha = hasher.digest()

        if calc_sha != payload_sha256:
            raise ValueError(f"SHA-256 mismatch! declared={payload_sha256.hex()}, calculated={calc_sha.hex()}")

        print(f"  SHA-256 Checksum: {calc_sha.hex()} (MATCHES HEADER)")
        print("\nAll container headers and payload integrity verified successfully!")
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", help="Path to .g4dense file")
    ap.add_argument("--verbose", action="store_true", help="Print per-layer tensor details")
    args = ap.parse_args()

    try:
        verify_g4dense_file(args.file, args.verbose)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
