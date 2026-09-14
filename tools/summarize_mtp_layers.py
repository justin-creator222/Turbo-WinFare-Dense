#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
JSON_FILE = ROOT / "build" / "mtp_layer_benchmark_results.json"

def main():
    if not JSON_FILE.exists():
        print("JSON results file not found!")
        return

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total benchmark runs recorded: {len(data)}")

    # Check output consistency across runs for each prompt
    print("\n--- Output Consistency Audit ---")
    by_prompt = defaultdict(dict)
    for item in data:
        cfg = item["config"]["name"]
        pid = item["prompt_id"]
        text = item.get("response_text", "")
        by_prompt[pid][cfg] = text

    for pid, cfgs in by_prompt.items():
        unique_texts = set(cfgs.values())
        print(f"Prompt '{pid}': {len(unique_texts)} unique text output(s) across all {len(cfgs)} layer configurations.")
        if len(unique_texts) == 1:
            sample = next(iter(unique_texts)).replace('\n', ' ')
            print(f"  [100% IDENTICAL ACROSS ALL CONFIGS] Sample: {sample[:70]}...")
        else:
            for text in unique_texts:
                matched_cfgs = [c for c, t in cfgs.items() if t == text]
                print(f"  Variant ({len(matched_cfgs)} configs: {', '.join(matched_cfgs)}): {text[:60]!r}")

    # Summary table
    print("\n--- Performance & Telemetry Table ---")
    by_config = defaultdict(list)
    for item in data:
        cfg = item["config"]["name"]
        by_config[cfg].append(item)

    header = "| Configuration | Resident | Streamed | RAM (MB) | Primes TPS | Python TPS | Capital TPS | Primes Acc % | Python Acc % | Avg I/O (ms) | MTP Status |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|"
    print(header)
    print(sep)

    for cfg, runs in by_config.items():
        res_l = runs[0]["resident_layers"]
        str_l = runs[0]["streamed_layers"]
        ram = runs[0]["ram_footprint_mb"]
        
        primes_run = next((r for r in runs if r["prompt_id"] == "primes"), None)
        python_run = next((r for r in runs if r["prompt_id"] == "python"), None)
        cap_run = next((r for r in runs if r["prompt_id"] == "capital"), None)
        
        p_tps = f"{primes_run['throughput_tps']:.2f}" if primes_run and primes_run['throughput_tps'] else "N/A"
        py_tps = f"{python_run['throughput_tps']:.2f}" if python_run and python_run['throughput_tps'] else "N/A"
        c_tps = f"{cap_run['throughput_tps']:.2f}" if cap_run and cap_run['throughput_tps'] else "N/A"
        
        p_acc = f"{primes_run['draft_acceptance_rate']:.1f}%" if primes_run and primes_run['draft_acceptance_rate'] is not None else "N/A"
        py_acc = f"{python_run['draft_acceptance_rate']:.1f}%" if python_run and python_run['draft_acceptance_rate'] is not None else "N/A"
        
        ios = [r["io_time_ms"] for r in runs if r.get("io_time_ms") is not None]
        avg_io = f"{sum(ios)/len(ios):.1f}" if ios else "N/A"
        
        mtp_status = "Active" if runs[0]["draft_loaded"] else ("OOM Fallback" if runs[0]["oom_detected"] else "Off")
        
        print(f"| {cfg} | {res_l} | {str_l} | {ram:.0f} | {p_tps} | {py_tps} | {c_tps} | {p_acc} | {py_acc} | {avg_io} | {mtp_status} |")

if __name__ == "__main__":
    main()
