"""
Compiles all HLSL compute shaders into Vulkan 1.3 SPIR-V bytecode using DXC.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHADERS_DIR = REPO_ROOT / "shaders"
OUT_DIR = REPO_ROOT / "build" / "shaders"
DXC_PATH = REPO_ROOT / "build" / "dxc.exe"


def compile_shaders():
    if not DXC_PATH.exists():
        print(f"Error: DXC compiler not found at {DXC_PATH}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hlsl_files = list(SHADERS_DIR.glob("*.hlsl"))
    if not hlsl_files:
        print(f"No .hlsl files found in {SHADERS_DIR}")
        return 0

    print(f"Compiling {len(hlsl_files)} HLSL shaders to Vulkan 1.3 SPIR-V...")
    success = True

    for hlsl in hlsl_files:
        spv_name = hlsl.stem + ".spv"
        spv_out = OUT_DIR / spv_name
        cmd = [
            str(DXC_PATH),
            "-T", "cs_6_6",
            "-spirv",
            "-fspv-target-env=vulkan1.3",
            "-O3",
            "-I", str(SHADERS_DIR),
            "-Fo", str(spv_out),
            str(hlsl)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAIL: {hlsl.name}\n{res.stderr}\n{res.stdout}", file=sys.stderr)
            success = False
        else:
            print(f"  OK: {hlsl.name} -> {spv_name} ({spv_out.stat().st_size} bytes)")

    if success:
        print("\nAll shaders compiled to SPIR-V successfully!")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(compile_shaders())
