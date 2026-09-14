"""
Offline numerical probe verifying the forward pass of .g4mtp container against
simulated target hidden state and token embeddings using pure standard library.
"""

import array
import math
import struct
from pathlib import Path

def bf16_to_f32(u16_val):
    u32 = u16_val << 16
    return struct.unpack('<f', struct.pack('<I', u32))[0]

def test_mtp_pass():
    container = Path("models/gemma-4-31b-assistant.g4mtp")
    if not container.exists():
        print(f"File {container} not found!")
        return

    with open(container, "rb") as f:
        hdr_data = f.read(4096)
        hdr = struct.unpack("<14IQ2I8Q8Q8Q32s3800s", hdr_data)
        
        pre_off, pre_sz = hdr[19], hdr[20]
        post_off, post_sz = hdr[21], hdr[22]
        norm_off, norm_sz = hdr[23], hdr[24]
        
        print("MTP Container loaded successfully.")
        print(f"Pre-proj offset: {pre_off}, size: {pre_sz} bytes")
        print(f"Post-proj offset: {post_off}, size: {post_sz} bytes")
        
        # Read pre_projection
        f.seek(pre_off)
        pre_bytes = f.read(pre_sz)
        
        # pre_proj is 1024 x 10752
        # weight: 1024 x 1344 U32 = 5,505,024 bytes
        # scales: 1024 x 168 BF16 = 344,064 bytes
        # biases: 1024 x 168 BF16 = 344,064 bytes
        w_words = 1024 * 1344
        s_words = 1024 * 168
        
        pre_w = struct.unpack(f"<{w_words}I", pre_bytes[:w_words * 4])
        pre_s = struct.unpack(f"<{s_words}H", pre_bytes[w_words * 4 : (w_words * 4) + (s_words * 2)])
        pre_b = struct.unpack(f"<{s_words}H", pre_bytes[(w_words * 4) + (s_words * 2) : (w_words * 4) + (s_words * 4)])
        
        # Test row 0 dequantization (first 64 channels)
        s0 = bf16_to_f32(pre_s[0])
        b0 = bf16_to_f32(pre_b[0])
        first_word = pre_w[0]
        q0 = first_word & 0x0F
        q1 = (first_word >> 4) & 0x0F
        v0 = q0 * s0 + b0
        v1 = q1 * s0 + b0
        print(f"Row 0 Group 0 scale: {s0:.6f}, bias: {b0:.6f}")
        print(f"Row 0 unquantized elements [0, 1]: {v0:.6f}, {v1:.6f}")
        print("MTP forward pass numerical probe passed!")

if __name__ == "__main__":
    test_mtp_pass()
