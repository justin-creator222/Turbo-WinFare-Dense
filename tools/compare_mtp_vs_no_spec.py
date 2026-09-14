#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
MTP_JSON = ROOT / "build" / "mtp_layer_benchmark_results.json"
NO_SPEC_JSON = ROOT / "build" / "no_spec_layer_benchmark_results.json"

def main():
    if not MTP_JSON.exists():
        print(f"Error: {MTP_JSON} not found!")
        return
    if not NO_SPEC_JSON.exists():
        print(f"Error: {NO_SPEC_JSON} not found! (Benchmark may still be in progress)")
        return

    with open(MTP_JSON, "r", encoding="utf-8") as f:
        mtp_data = json.load(f)
    with open(NO_SPEC_JSON, "r", encoding="utf-8") as f:
        no_spec_data = json.load(f)

    print(f"Loaded {len(mtp_data)} MTP runs and {len(no_spec_data)} No-Spec runs.")

    # Cross-Mode Text Consistency Audit
    print("\n========================================================")
    print("  Cross-Mode Text Consistency Audit (MTP vs No-Spec)")
    print("========================================================")
    mtp_by_prompt = defaultdict(dict)
    for item in mtp_data:
        cfg = item["config"]["name"]
        pid = item["prompt_id"]
        mtp_by_prompt[pid][cfg] = item.get("response_text", "")

    no_spec_by_prompt = defaultdict(dict)
    for item in no_spec_data:
        cfg = item["config"]["name"]
        pid = item["prompt_id"]
        no_spec_by_prompt[pid][cfg] = item.get("response_text", "")

    for pid in mtp_by_prompt:
        mtp_texts = set(mtp_by_prompt[pid].values())
        no_spec_texts = set(no_spec_by_prompt[pid].values())
        all_texts = mtp_texts.union(no_spec_texts)
        print(f"Prompt '{pid}': {len(all_texts)} distinct text(s) across all MTP and No-Spec runs.")
        if len(all_texts) == 1:
            sample = next(iter(all_texts)).replace('\n', ' ')
            print(f"  [100% BIT-IDENTICAL ACROSS BOTH MODES] Sample: {sample[:70]}...")
        else:
            print(f"  WARNING: Divergence detected between MTP and No-Spec for prompt '{pid}'!")

    # Performance Comparison Table
    print("\n========================================================")
    print("  Throughput (TPS) & Latency Head-to-Head Comparison")
    print("========================================================")

    mtp_by_layers = defaultdict(lambda: {})
    for item in mtp_data:
        l = item["config"]["layers"]
        mtp_by_layers[l][item["prompt_id"]] = item

    no_spec_by_layers = defaultdict(lambda: {})
    for item in no_spec_data:
        l = item["config"]["layers"]
        no_spec_by_layers[l][item["prompt_id"]] = item

    header = (
        "| Configuration | Resident | RAM MTP | RAM NoSpec | RAM Delta | "
        "Primes MTP | Primes NoSpec | Primes Speedup | "
        "Python MTP | Python NoSpec | Python Speedup | "
        "Capital MTP | Capital NoSpec | Capital Speedup | "
        "I/O MTP (ms) | I/O NoSpec |"
    )
    sep = "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    print(header)
    print(sep)

    all_layers = sorted(set(mtp_by_layers.keys()).intersection(set(no_spec_by_layers.keys())))

    for l in all_layers:
        mtp_runs = mtp_by_layers[l]
        no_runs = no_spec_by_layers[l]

        m_p = mtp_runs.get("primes")
        n_p = no_runs.get("primes")
        if not m_p or not n_p:
            continue

        cfg_name = f"{l} Layers"
        if l == 0: cfg_name = "0 Layers (Pure Streamed)"
        elif l == 39: cfg_name = "39 Layers (E2B Baseline)"
        elif l == 43: cfg_name = "43 Layers (Auto MTP Default)"
        elif l == 45: cfg_name = "45 Layers (Natural AR Default)"
        elif l == 46: cfg_name = "46 Layers (Max Coexistence)"
        elif l == 47: cfg_name = "47 Layers (Driver OOM Boundary)"

        res_l = m_p["resident_layers"]
        m_ram = m_p["ram_footprint_mb"]
        n_ram = n_p["ram_footprint_mb"]
        ram_delta = m_ram - n_ram

        m_p_tps = m_p["throughput_tps"]
        n_p_tps = n_p["throughput_tps"]
        p_ratio = (m_p_tps / n_p_tps) if (m_p_tps and n_p_tps) else 0.0

        m_py = mtp_runs.get("python")
        n_py = no_runs.get("python")
        m_py_tps = m_py["throughput_tps"] if m_py else 0.0
        n_py_tps = n_py["throughput_tps"] if n_py else 0.0
        py_ratio = (m_py_tps / n_py_tps) if (m_py_tps and n_py_tps) else 0.0

        m_cap = mtp_runs.get("capital")
        n_cap = no_runs.get("capital")
        m_cap_tps = m_cap["throughput_tps"] if m_cap else 0.0
        n_cap_tps = n_cap["throughput_tps"] if n_cap else 0.0
        cap_ratio = (m_cap_tps / n_cap_tps) if (m_cap_tps and n_cap_tps) else 0.0

        m_ios = [r["io_time_ms"] for r in mtp_runs.values() if r.get("io_time_ms") is not None]
        n_ios = [r["io_time_ms"] for r in no_runs.values() if r.get("io_time_ms") is not None]
        m_avg_io = sum(m_ios)/len(m_ios) if m_ios else 0.0
        n_avg_io = sum(n_ios)/len(n_ios) if n_ios else 0.0

        row = (
            f"| {cfg_name} | {res_l}/60 | {m_ram:.0f} MB | {n_ram:.0f} MB | {ram_delta:+.0f} MB | "
            f"{m_p_tps:.2f} | {n_p_tps:.2f} | **{p_ratio:.2f}x** | "
            f"{m_py_tps:.2f} | {n_py_tps:.2f} | **{py_ratio:.2f}x** | "
            f"{m_cap_tps:.2f} | {n_cap_tps:.2f} | **{cap_ratio:.2f}x** | "
            f"{m_avg_io:.1f} | {n_avg_io:.1f} |"
        )
        print(row)

if __name__ == "__main__":
    main()
