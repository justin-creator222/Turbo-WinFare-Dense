"""
Interleaved Tier Benchmarking Tool for Turbo-WinFare Dense.
Runs A/B/A/B benchmark passes across memory tiers (1, 2, 3) and records median metrics.
"""

import argparse
import json
import subprocess
import time
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
EXE_PATH = BUILD_DIR / "turbo-dense.exe"


def check_disk_queue():
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Counter '\\PhysicalDisk(_Total)\\Current Disk Queue Length' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue"],
            capture_output=True, text=True, timeout=5
        )
        if res.returncode == 0 and res.stdout.strip():
            val = float(res.stdout.strip())
            print(f"[bench] Current Disk Queue Length: {val:.2f}")
    except Exception:
        pass


def run_benchmark_pass(model_path, tier, prompt, max_tokens=10):
    cmd = [
        str(EXE_PATH),
        "--model", str(model_path),
        "--tier", str(tier),
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0.0"
    ]
    env = dict(sys.modules["os"].environ)
    env["PATH"] = r"C:\w64devkit\bin;" + env.get("PATH", "")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    t1 = time.perf_counter()

    if proc.returncode != 0:
        print(f"[ERROR] turbo-dense failed with exit code {proc.returncode} for Tier {tier}")
        print("STDERR:\n", proc.stderr)
        print("STDOUT:\n", proc.stdout)
        return None

    # Parse stdout for telemetry metrics
    tokens_gen = max_tokens
    tps = 0.0
    ram_mb = 0.0
    for line in proc.stdout.splitlines():
        line_clean = line.strip()
        if "Throughput (TPS):" in line_clean:
            try:
                tps = float(line_clean.split(":")[1].split()[0])
            except Exception:
                pass
        elif "RAM Footprint:" in line_clean:
            try:
                ram_mb = float(line_clean.split(":")[1].split("/")[0].replace("MB", "").strip())
            except Exception:
                pass
        elif "Tokens Generated:" in line_clean:
            try:
                tokens_gen = int(line_clean.split(":")[1].strip())
            except Exception:
                pass

    if tps == 0.0:
        elapsed = max(t1 - t0, 0.001)
        tps = tokens_gen / elapsed

    return {
        "tier": tier,
        "tokens": tokens_gen,
        "elapsed_sec": t1 - t0,
        "tps": tps,
        "ram_footprint_mb": ram_mb
    }


def main():
    parser = argparse.ArgumentParser(description="Turbo-WinFare Dense Interleaved Tier Benchmark")
    parser.add_argument("--model", type=str, default="tests/fixtures/tiny.g4dense", help="Path to model file")
    parser.add_argument("--tiers", type=str, default="1,2,3", help="Comma-separated tier list")
    parser.add_argument("--rounds", type=int, default=3, help="Number of interleaved rounds")
    parser.add_argument("--interleaved", action="store_true", help="Run in interleaved A/B/A/B order")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms.", help="Prompt text")
    parser.add_argument("--tokens", type=int, default=8, help="Tokens per generation")
    args = parser.parse_args()

    if not EXE_PATH.exists():
        print(f"Error: {EXE_PATH} not found. Please build the project first.")
        sys.exit(1)

    tier_list = [int(t.strip()) for t in args.tiers.split(",") if t.strip()]

    print("========================================================")
    print("  Turbo-WinFare Dense: Interleaved Tier Benchmarks     ")
    print("========================================================")
    print(f"Model:   {args.model}")
    print(f"Tiers:   {tier_list}")
    print(f"Rounds:  {args.rounds}")
    print(f"Mode:    {'Interleaved (A/B/A/B)' if args.interleaved else 'Sequential'}")
    print("--------------------------------------------------------")

    check_disk_queue()

    results_by_tier = {t: [] for t in tier_list}

    for r in range(1, args.rounds + 1):
        print(f"\n--- Round {r}/{args.rounds} ---")
        for t in tier_list:
            print(f"  Running Tier {t}...", end="", flush=True)
            res = run_benchmark_pass(args.model, t, args.prompt, args.tokens)
            if res:
                results_by_tier[t].append(res)
                print(f" TPS: {res['tps']:.2f}, Latency: {res['elapsed_sec']*1000:.1f}ms, RAM: {res['ram_footprint_mb']:.1f}MB")
            else:
                print(" FAILED")

    print("\n========================================================")
    print("  Benchmark Summary (Medians across rounds)            ")
    print("========================================================")
    print(f"{'Tier':<8} | {'Median TPS':<12} | {'Median RAM (MB)':<16} | {'Samples':<8}")
    print("-" * 52)

    for t in tier_list:
        runs = results_by_tier[t]
        if not runs:
            print(f"Tier {t:<3} | {'FAILED':<12} | {'N/A':<16} | 0")
            continue
        tps_sorted = sorted([r["tps"] for r in runs])
        ram_sorted = sorted([r["ram_footprint_mb"] for r in runs])
        mid = len(tps_sorted) // 2
        med_tps = tps_sorted[mid]
        med_ram = ram_sorted[mid]
        print(f"Tier {t:<3} | {med_tps:<12.2f} | {med_ram:<16.2f} | {len(runs)}")

    print("========================================================\n")


if __name__ == "__main__":
    main()
