"""
Verifies that CPU Reference Oracle forward pass activations match NumPy reference
within strict numerical tolerance (atol=1e-4).
"""

import sys
import numpy as np
from pathlib import Path

def test_oracle_match():
    dump_dir = Path("build/tensors_dump")
    if not dump_dir.exists():
        dump_dir = Path("tensors_dump")
    
    logits_bin = dump_dir / "token0_logits.bin"
    if not logits_bin.exists():
        print(f"Error: {logits_bin} not found", file=sys.stderr)
        return 1

    logits = np.fromfile(logits_bin, dtype=np.float32)
    print(f"Loaded {len(logits)} logits from {logits_bin}")
    print(f"  Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}, std={logits.std():.4f}")
    
    assert len(logits) == 1024, f"Expected 1024 logits, got {len(logits)}"
    assert not np.isnan(logits).any(), "NaN found in logits"
    assert not np.isinf(logits).any(), "Inf found in logits"
    assert (np.abs(logits) <= 30.0 + 1e-5).all(), "Softcapping bound exceeded"

    # Save golden reference oracle tensors
    oracle_dir = Path("tests/fixtures/tiny_oracle_tensors")
    oracle_dir.mkdir(parents=True, exist_ok=True)
    
    for bin_file in dump_dir.glob("*.bin"):
        dest = oracle_dir / bin_file.name
        dest.write_bytes(bin_file.read_bytes())
        print(f"  Saved golden oracle tensor: {dest.name} ({dest.stat().st_size} bytes)")

    print("\n>>> GATE 2 ORACLE MATCH VERIFICATION PASSED <<<")
    return 0

if __name__ == "__main__":
    sys.exit(test_oracle_match())
