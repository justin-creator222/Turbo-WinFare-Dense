#!/usr/bin/env python3
"""
benchmark_mtp_layers.py - Benchmark Gemma 4 31B with MTP across all layer configurations.
Tests:
- 0 resident layers (100% streamed)
- 15 resident layers
- 25 resident layers
- 35 resident layers
- 39 resident layers (legacy E2B allocation)
- 42 resident layers
- 43 resident layers (natural automatic MTP default)
- 44 resident layers
- 45 resident layers (maximum driver limit)
- 46 resident layers (boundary condition: driver OOM graceful fallback)
"""

import os
import sys
import json
import time
import subprocess
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "build" / "run_turbo_dense.exe"
MODEL = ROOT / "models" / "gemma-4-31b-dense.g4dense"
MTP_MODEL = ROOT / "models" / "gemma-4-31b-assistant.g4mtp"
RESULTS_JSON = ROOT / "build" / "mtp_layer_benchmark_results.json"

CONFIGS = [
    {"name": "0 Layers (Pure Streamed)", "layers": 0},
    {"name": "15 Layers", "layers": 15},
    {"name": "25 Layers", "layers": 25},
    {"name": "35 Layers", "layers": 35},
    {"name": "39 Layers (E2B Baseline)", "layers": 39},
    {"name": "42 Layers", "layers": 42},
    {"name": "43 Layers (Auto MTP Default)", "layers": 43},
    {"name": "44 Layers", "layers": 44},
    {"name": "45 Layers (Driver Ceiling)", "layers": 45},
    {"name": "46 Layers (OOM Boundary)", "layers": 46},
]

PROMPTS = [
    {
        "id": "primes",
        "prompt": "List the first 10 prime numbers:",
        "max_tokens": 24,
    },
    {
        "id": "python",
        "prompt": "Write a Python function to check if a number is even:",
        "max_tokens": 24,
    },
    {
        "id": "capital",
        "prompt": "The capital of France is",
        "max_tokens": 8,
    },
]

def run_test(config, prompt_info):
    env = os.environ.copy()
    env["PATH"] = r"C:\w64devkit\bin;" + env.get("PATH", "")
    if config["layers"] is not None:
        env["G4DENSE_MAX_RESIDENT_LAYERS"] = str(config["layers"])
    else:
        env.pop("G4DENSE_MAX_RESIDENT_LAYERS", None)

    cmd = [
        str(EXE),
        "--model", str(MODEL),
        "--prompt", prompt_info["prompt"],
        "--max-tokens", str(prompt_info["max_tokens"]),
        "--spec",
        "--draft-k", "5",
    ]

    print(f"\n========================================================")
    print(f"Running: {config['name']} | Prompt: {prompt_info['id']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"========================================================")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=180,
            cwd=str(ROOT),
        )
        elapsed_proc = time.time() - t0
        output = proc.stdout
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: Process timed out after 180 seconds")
        return {
            "config": config,
            "prompt_id": prompt_info["id"],
            "status": "TIMEOUT",
            "error": "Process timed out after 180s",
        }
    except Exception as e:
        print(f"ERROR: Process execution failed: {e}")
        return {
            "config": config,
            "prompt_id": prompt_info["id"],
            "status": "FAILED",
            "error": str(e),
        }

    # Parse stdout
    print(output)

    # Extract metrics
    res = {
        "config": config,
        "prompt_id": prompt_info["id"],
        "exit_code": exit_code,
        "total_wall_time": elapsed_proc,
        "status": "OK" if exit_code == 0 else "ERROR",
        "resident_layers": None,
        "streamed_layers": None,
        "draft_loaded": False,
        "tokens_generated": None,
        "elapsed_time_s": None,
        "throughput_tps": None,
        "ram_footprint_mb": None,
        "draft_acceptance_rate": None,
        "draft_accepted": None,
        "draft_drafted": None,
        "io_time_ms": None,
        "gpu_wait_ms": None,
        "lm_head_ms": None,
        "cpu_other_ms": None,
        "response_text": "",
        "oom_detected": False,
    }

    # Residency
    m_res = re.search(r"(\d+) of (\d+) layers resident .* (\d+) streamed per token", output)
    if m_res:
        res["resident_layers"] = int(m_res.group(1))
        res["streamed_layers"] = int(m_res.group(3))

    # Draft model loaded
    if "MTP assistant loaded for speculative decoding" in output:
        res["draft_loaded"] = True
    elif "draft model failed to load" in output:
        res["draft_loaded"] = False
        res["oom_detected"] = True

    # Tokens generated
    m_tok = re.search(r"Tokens Generated:\s+(\d+)", output)
    if m_tok:
        res["tokens_generated"] = int(m_tok.group(1))

    # Elapsed time
    m_el = re.search(r"Elapsed Time:\s+([\d\.]+)\s+s", output)
    if m_el:
        res["elapsed_time_s"] = float(m_el.group(1))

    # Throughput
    m_tps = re.search(r"Throughput \(TPS\):\s+([\d\.]+)\s+tokens/s", output)
    if m_tps:
        res["throughput_tps"] = float(m_tps.group(1))

    # RAM Footprint
    m_ram = re.search(r"RAM Footprint:\s+([\d\.]+)\s+MB", output)
    if m_ram:
        res["ram_footprint_mb"] = float(m_ram.group(1))

    # Draft acceptance
    m_acc = re.search(r"Draft acceptance:\s+([\d\.]+)\s+%\s+\((\d+)/(\d+)\)", output)
    if m_acc:
        res["draft_acceptance_rate"] = float(m_acc.group(1))
        res["draft_accepted"] = int(m_acc.group(2))
        res["draft_drafted"] = int(m_acc.group(3))

    # Per forward pass breakdown
    m_io = re.search(r"Layer stream I/O:\s+([\d\.]+)", output)
    if m_io:
        res["io_time_ms"] = float(m_io.group(1))
    m_gpu = re.search(r"GPU queue wait:\s+([\d\.]+)", output)
    if m_gpu:
        res["gpu_wait_ms"] = float(m_gpu.group(1))
    m_lm = re.search(r"LM head:\s+([\d\.]+)", output)
    if m_lm:
        res["lm_head_ms"] = float(m_lm.group(1))
    m_cpu = re.search(r"CPU other:\s+([\d\.]+)", output)
    if m_cpu:
        res["cpu_other_ms"] = float(m_cpu.group(1))

    # Response text
    m_resp = re.search(r"Response:\n(.*?)\n\n====", output, re.DOTALL)
    if m_resp:
        res["response_text"] = m_resp.group(1).strip()

    return res

def main():
    if not EXE.exists():
        print(f"Error: {EXE} not found! Build first.")
        sys.exit(1)
    if not MODEL.exists():
        print(f"Error: {MODEL} not found!")
        sys.exit(1)
    if not MTP_MODEL.exists():
        print(f"Error: {MTP_MODEL} not found!")
        sys.exit(1)

    all_results = []
    print("Starting Gemma 4 31B MTP Multi-Layer Configuration Benchmark Suite...")

    for config in CONFIGS:
        for prompt_info in PROMPTS:
            res = run_test(config, prompt_info)
            all_results.append(res)
            # Short cooldown to allow driver memory to fully reset
            time.sleep(1)

    # Save to JSON
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results saved to {RESULTS_JSON}")

if __name__ == "__main__":
    main()
