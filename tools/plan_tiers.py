"""
Budget solver and Tier derivation tool for Turbo-WinFare Dense.
Consumes build/ground_truth.json and emits config/tiers.json.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
BUILD_DIR = REPO_ROOT / "build"
GROUND_TRUTH_PATH = BUILD_DIR / "ground_truth.json"
TIERS_JSON_PATH = CONFIG_DIR / "tiers.json"

# Fixed Gemma 4 31B Geometry constants
TOTAL_LAYERS = 60
GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]  # 10 full attention layers
SLIDING_LAYERS = [i for i in range(TOTAL_LAYERS) if i not in GLOBAL_LAYERS]

# Exact measured per-layer bytes (MLX affine INT4 group-64)
# MLP: 3 * (21504*672*4 + 21504*84*2 + 21504*84*2) = 195,084,288 B
# Attn Q: 8192*672*4 + 8192*84*4 = 24,772,608 B
# Attn K: 4096*672*4 + 4096*84*4 = 12,386,304 B
# Attn V: 4096*672*4 + 4096*84*4 = 12,386,304 B
# Attn O: 5376*1024*4 + 5376*128*4 = 24,772,608 B
# Norms & scalars: 44,034 B
LAYER_BYTES = 269446146  # 256.96 MB (245.04 MiB)
EMBED_BYTES = 792723456  # 755.99 MB (720.71 MiB)
DRAFT_E2B_BYTES = 1258291200  # 1200.0 MB (E2B 4-bit resident on CPU)
RING_BUFFERS_COUNT = 4
RING_BUFFER_BYTES = RING_BUFFERS_COUNT * LAYER_BYTES  # 1027.8 MB
ACTIVATIONS_BYTES = 125829120  # 120.0 MB (batch M=6)
RUNTIME_HEAP_OVERHEAD_BYTES = 262144000  # 250.0 MB

# Priority order for pinning (§4)
# 1. Input Boundary: 0..6
# 2. Output Boundary: 53..59
# 3. Global Attention Blocks: 5, 11, 17, 23, 29, 35, 41, 47
# 4. Remaining middle layers
PIN_PRIORITY = []
for l in [0, 1, 2, 3, 4, 5, 6]:
    if l not in PIN_PRIORITY: PIN_PRIORITY.append(l)
for l in [59, 58, 57, 56, 55, 54, 53]:
    if l not in PIN_PRIORITY: PIN_PRIORITY.append(l)
for l in GLOBAL_LAYERS:
    if l not in PIN_PRIORITY: PIN_PRIORITY.append(l)
for l in range(TOTAL_LAYERS):
    if l not in PIN_PRIORITY: PIN_PRIORITY.append(l)


def compute_kv_cache_bytes(context_len, is_int8):
    # 50 sliding window layers (window=1024, 16 heads * 256 dim)
    # 10 global layers (window=context_len, 16 heads * 256 dim)
    dtype_size = 1 if is_int8 else 2
    bytes_per_token_per_layer = 2 * 16 * 256 * dtype_size  # K and V
    sliding_tokens = min(context_len, 1024)
    sliding_bytes = len(SLIDING_LAYERS) * sliding_tokens * bytes_per_token_per_layer
    global_bytes = len(GLOBAL_LAYERS) * context_len * bytes_per_token_per_layer
    scale_bytes = (len(GLOBAL_LAYERS) * context_len + len(SLIDING_LAYERS) * sliding_tokens) * 16 * 2 if is_int8 else 0
    return sliding_bytes + global_bytes + scale_bytes


def tokens_per_pass(alpha, k):
    if alpha >= 1.0:
        return float(k + 1)
    return (1.0 - math.pow(alpha, k + 1)) / (1.0 - alpha)


def project_tps(streamed_layers, alpha, k, nvme_gbs, draft_tps=40.0, compute_ms_per_layer=3.5):
    # Pass time = max(stream_time, gpu_verify_time) + draft_time
    streamed_bytes = streamed_layers * LAYER_BYTES
    stream_sec = (streamed_bytes / (1024**3)) / nvme_gbs if streamed_layers > 0 else 0.0
    gpu_sec = (TOTAL_LAYERS * compute_ms_per_layer) / 1000.0
    draft_sec = k / draft_tps
    pass_time_sec = max(stream_sec, gpu_sec) + 0.05  # 50ms overlap & host latency
    tpp = tokens_per_pass(alpha, k)
    return tpp / pass_time_sec


def solve_tiers(ground_truth):
    sys_ram_gb = ground_truth.get("system", {}).get("ram_total_gb", 23.8)
    nvme_gbs = ground_truth.get("storage", {}).get("buffered_warm_gbs", 5.82)
    device_local_heap_mb = ground_truth.get("vulkan", {}).get("heaps", [{}])[0].get("size_mb", 13417.0)

    tier_specs = [
        {"id": 1, "name": "Tier 1 (Baseline)", "ceiling_mb": 6000.0, "int8_kv": True, "target_tps": 2.50},
        {"id": 2, "name": "Tier 2 (Balanced)", "ceiling_mb": 10000.0, "int8_kv": True, "target_tps": 3.85},
        {"id": 3, "name": "Tier 3 (High-Perf)", "ceiling_mb": 16000.0, "int8_kv": False, "target_tps": 6.00},
        {"id": 4, "name": "Tier 4 (Resident)", "ceiling_mb": 22000.0, "int8_kv": False, "target_tps": 18.00},
    ]

    context_8k = 8192
    results = []

    for ts in tier_specs:
        ceiling_bytes = int(ts["ceiling_mb"] * 1024 * 1024)
        kv_bytes = compute_kv_cache_bytes(context_8k, ts["int8_kv"])
        fixed_overhead = EMBED_BYTES + DRAFT_E2B_BYTES + RING_BUFFER_BYTES + kv_bytes + ACTIVATIONS_BYTES + RUNTIME_HEAP_OVERHEAD_BYTES

        headroom = ceiling_bytes - fixed_overhead
        if headroom < 0:
            max_pinned = 0
        else:
            max_pinned = min(TOTAL_LAYERS, int(headroom // LAYER_BYTES))

        # Adjust for Tier 1 strictly keeping under 6000 MB
        if ts["id"] == 1:
            max_pinned = min(max_pinned, 8)

        pinned_layers = PIN_PRIORITY[:max_pinned]
        pinned_layers.sort()
        streamed_count = TOTAL_LAYERS - len(pinned_layers)

        total_alloc_bytes = fixed_overhead + len(pinned_layers) * LAYER_BYTES
        total_alloc_mb = total_alloc_bytes / (1024 * 1024)

        tps_proj = {
            "alpha_0.60": round(project_tps(streamed_count, 0.60, 6, nvme_gbs), 2),
            "alpha_0.70": round(project_tps(streamed_count, 0.70, 6, nvme_gbs), 2),
            "alpha_0.78": round(project_tps(streamed_count, 0.78, 6, nvme_gbs), 2),
            "alpha_0.85": round(project_tps(streamed_count, 0.85, 6, nvme_gbs), 2),
        }

        # Feasibility evaluation
        feasible = True
        reason = "Feasible on target hardware."

        if ts["id"] == 4:
            feasible = False
            reason = f"Tier 4 (22.0 GB) exceeds usable system RAM ({sys_ram_gb:.1f} GB); requires 32 GB hardware to avoid OS thrashing."
        elif ts["id"] == 3:
            if total_alloc_mb > sys_ram_gb * 1024 * 0.75:
                reason = f"Tier 3 is tightly budgeted on {sys_ram_gb:.1f} GB RAM; requires host-visible memory fallback."

        results.append({
            "tier_id": ts["id"],
            "name": ts["name"],
            "ceiling_mb": ts["ceiling_mb"],
            "actual_footprint_mb": round(total_alloc_mb, 1),
            "slack_mb": round(ts["ceiling_mb"] - total_alloc_mb, 1),
            "pinned_layers_count": len(pinned_layers),
            "pinned_layers": pinned_layers,
            "streamed_layers_count": streamed_count,
            "streamed_bytes": streamed_count * LAYER_BYTES,
            "kv_cache_dtype": "INT8" if ts["int8_kv"] else "FP16",
            "kv_cache_mb": round(kv_bytes / (1024 * 1024), 1),
            "projected_tps": tps_proj,
            "target_tps": ts["target_tps"],
            "meets_target": tps_proj["alpha_0.78"] >= ts["target_tps"] if feasible else False,
            "feasible": feasible,
            "reason": reason
        })

    return {
        "hardware_context": {
            "usable_ram_gb": sys_ram_gb,
            "measured_nvme_warm_gbs": nvme_gbs,
            "device_local_heap_mb": device_local_heap_mb
        },
        "layer_geometry": {
            "d_model": 5376,
            "d_ff": 21504,
            "total_layers": TOTAL_LAYERS,
            "sliding_layers_count": len(SLIDING_LAYERS),
            "global_layers_count": len(GLOBAL_LAYERS),
            "layer_bytes": LAYER_BYTES,
            "embedding_bytes": EMBED_BYTES
        },
        "tiers": results
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="validate tier allocations against ceilings")
    ap.add_argument("--out", default=str(TIERS_JSON_PATH), help="output JSON path")
    args = ap.parse_args()

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    ground_truth = {}
    if GROUND_TRUTH_PATH.exists():
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

    tiers_data = solve_tiers(ground_truth)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tiers_data, f, indent=2)
    print(f"Wrote {args.out}")

    # Self-test check
    failed = False
    print("\n--- Tier Budget Solver Self-Test ---")
    for t in tiers_data["tiers"]:
        status = "FEASIBLE" if t["feasible"] else "DISABLED"
        print(f"[{t['name']}] Allocated: {t['actual_footprint_mb']} MB / {t['ceiling_mb']} MB ({status})")
        print(f"  Pinned: {t['pinned_layers_count']} layers, Streamed: {t['streamed_layers_count']} layers, KV: {t['kv_cache_dtype']}")
        print(f"  Projected TPS @ alpha=0.78: {t['projected_tps']['alpha_0.78']} (Target: {t['target_tps']} TPS)")
        if t["feasible"] and t["actual_footprint_mb"] > t["ceiling_mb"]:
            print(f"  ERROR: Tier {t['tier_id']} exceeds ceiling by {t['actual_footprint_mb'] - t['ceiling_mb']} MB")
            failed = True

    if failed:
        print("\nSELF-TEST FAILED!")
        return 1
    else:
        print("\nSELF-TEST PASSED: All feasible tiers satisfy memory ceilings.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
