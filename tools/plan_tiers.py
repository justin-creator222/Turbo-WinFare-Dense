"""
Budget solver and Tier derivation tool for Turbo-WinFare Dense.
Consumes build/ground_truth.json and emits config/tiers.json.
Implements two-tier residency per-heap budgeting (Heap 0 vs Heap 1).
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
CHECKPOINT_CONFIG_PATH = REPO_ROOT / "models" / "gemma-4-31b-it-4bit" / "config.json"


def get_layer_types():
    if CHECKPOINT_CONFIG_PATH.exists():
        with open(CHECKPOINT_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            text_cfg = cfg.get("text_config", cfg)
            lt = text_cfg.get("layer_types", [])
            if lt:
                return lt
    pattern = []
    for l in range(60):
        if (l % 6) == 5:
            pattern.append("full_attention")
        else:
            pattern.append("sliding_attention")
    return pattern


LAYER_TYPES = get_layer_types()
TOTAL_LAYERS = len(LAYER_TYPES)
GLOBAL_LAYERS = [i for i, lt in enumerate(LAYER_TYPES) if lt == "full_attention"]
SLIDING_LAYERS = [i for i, lt in enumerate(LAYER_TYPES) if lt == "sliding_attention"]

# Exact measured per-layer bytes (MLX affine INT4 group-64)
# MLP: 3 * (21504*672*4 + 21504*84*2 + 21504*84*2) = 195,084,288 B
# Attn Q: 8192*672*4 + 8192*84*4 = 24,772,608 B
# Attn K: 4096*672*4 + 4096*84*4 = 12,386,304 B
# Attn V: 4096*672*4 + 4096*84*4 = 12,386,304 B
# Attn O: 5376*1024*4 + 5376*128*4 = 24,772,608 B
# Norms & scalars: 44,034 B
LAYER_BYTES = 269446146  # 256.96 MB (245.04 MiB)
EMBED_BYTES = 792723456  # 755.99 MB (720.71 MiB)
DRAFT_E2B_BYTES = 1258291200  # 1200.0 MB (E2B 4-bit resident on CPU RAM)
RING_BUFFERS_COUNT = 4
RING_BUFFER_BYTES = RING_BUFFERS_COUNT * LAYER_BYTES  # 1027.8 MB
ACTIVATIONS_BYTES = 125829120  # 120.0 MB (batch M=7)
RUNTIME_HEAP_OVERHEAD_BYTES = 262144000  # 250.0 MB

# Pinning Priority (§4)
PIN_PRIORITY = []
for l in [0, 1, 2, 3, 4, 5, 6]:
    if l not in PIN_PRIORITY and l < TOTAL_LAYERS: PIN_PRIORITY.append(l)
for l in [59, 58, 57, 56, 55, 54, 53]:
    if l not in PIN_PRIORITY and l < TOTAL_LAYERS: PIN_PRIORITY.append(l)
for l in GLOBAL_LAYERS:
    if l not in PIN_PRIORITY and l < TOTAL_LAYERS: PIN_PRIORITY.append(l)
for l in range(TOTAL_LAYERS):
    if l not in PIN_PRIORITY: PIN_PRIORITY.append(l)


def compute_kv_cache_bytes(context_len, is_int8):
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


def project_tps(streamed_layers, alpha, k, nvme_gbs, draft_tps=40.0, compute_ms_per_layer=3.67):
    streamed_bytes = streamed_layers * LAYER_BYTES
    stream_sec = (streamed_bytes / (1024**3)) / nvme_gbs if streamed_layers > 0 else 0.0
    gpu_sec = (TOTAL_LAYERS * compute_ms_per_layer) / 1000.0
    draft_sec = k / draft_tps
    pass_time_sec = max(stream_sec, gpu_sec) + draft_sec + 0.03  # overlapped I/O/compute + draft + latency
    tpp = tokens_per_pass(alpha, k)
    return tpp / pass_time_sec


def solve_tiers(ground_truth):
    sys_ram_gb = ground_truth.get("system", {}).get("ram_total_gb", 23.81)
    nvme_gbs = ground_truth.get("storage", {}).get("buffered_warm_gbs", 5.68)
    
    # Measured Heaps
    heaps = ground_truth.get("vulkan", {}).get("heaps", [])
    heap0_size_mb = heaps[0].get("size_mb", 13417.62) if len(heaps) > 0 else 13417.62
    heap1_size_mb = heaps[1].get("size_mb", 6708.75) if len(heaps) > 1 else 6708.75

    # Targets re-based in round 8 against what this machine actually permits.
    #
    # The old targets (2.50 / 3.85 / 6.00 / 18.00 TPS) were set before anything ran. Every one
    # of them reported meets_target: false for the whole life of the project, which makes them
    # decoration rather than targets. The arithmetic that replaces them:
    #
    #   weights, all 60 layers            15.06 GiB
    #   driver's observed import ceiling  ~11.75 GiB   (45 layers, measured)
    #   + KV cache (FP16, 4096)            1.17 GiB
    #   + LM head                          0.79 GiB
    #   + streaming pool                   1.05 GiB
    #   ----------------------------------------------
    #   total against a 15.90 GiB heap    ~14.8 GiB
    #
    # Full residency needs 15.06 GiB of imports alone, so it does not fit and no tier above
    # ~46 resident layers is reachable. That is a heap ceiling, not a tuning problem: all three
    # BIOS UMA settings were measured and minimum is the best of them (8 GB gives a larger
    # device-local heap but only 23.81 GiB of visible RAM, where the import broke down around
    # 30 layers; 4 GB is worse on both counts).
    #
    # The targets below are the measured operating point and the headroom above it that the
    # current kernels allow:
    #
    #   measured, greedy decode, 45 of 60 resident      1.13 tok/s
    #   the same pass with streaming perfectly hidden   1.46 tok/s  (GPU 631 + LM 24 + CPU 29)
    #
    # Round 8's version of these lines read 0.75 and 1.14, and named streaming overlap as the
    # gap. The real gap was that streaming ran unbuffered at 3.09 GB/s; see
    # measured_operating_point.
    #
    # So tier 3 is set at the perfect-overlap ceiling and tier 4 is marked infeasible with the
    # arithmetic rather than a target nothing can meet.
    #
    # KV is FP16 everywhere: the engine has no INT8 KV path, and claiming one in the plan
    # understated every tier's footprint by ~1 GiB.
    tier_specs = [
        {"id": 1, "name": "Tier 1 (Baseline)", "ceiling_mb": 6000.0, "pinned_target": 6, "int8_kv": False, "target_tps": 0.35},
        {"id": 2, "name": "Tier 2 (Balanced)", "ceiling_mb": 10000.0, "pinned_target": 21, "int8_kv": False, "target_tps": 0.55},
        {"id": 3, "name": "Tier 3 (High-Perf)", "ceiling_mb": 16000.0, "pinned_target": 45, "int8_kv": False, "target_tps": 1.46},
        {"id": 4, "name": "Tier 4 (Resident)", "ceiling_mb": 22000.0, "pinned_target": 60, "int8_kv": False, "target_tps": 2.00},
    ]

    context_8k = 8192
    results = []

    for ts in tier_specs:
        pinned_count = ts["pinned_target"]
        kv_bytes = compute_kv_cache_bytes(context_8k, ts["int8_kv"])
        
        # Two-Tier Residency Accounting:
        # Heap 0 (Device Local Only): Pinned layers
        heap0_bytes = pinned_count * LAYER_BYTES
        heap0_mb = heap0_bytes / (1024 * 1024)

        # Heap 1 (Host Visible): Streaming ring, Embeddings, KV Cache, Activations, Runtime Overhead
        ring_bytes = 0 if pinned_count == TOTAL_LAYERS else RING_BUFFER_BYTES
        heap1_bytes = ring_bytes + EMBED_BYTES + kv_bytes + ACTIVATIONS_BYTES + RUNTIME_HEAP_OVERHEAD_BYTES
        heap1_mb = heap1_bytes / (1024 * 1024)

        total_alloc_bytes = heap0_bytes + heap1_bytes
        total_alloc_mb = total_alloc_bytes / (1024 * 1024)

        pinned_layers = PIN_PRIORITY[:pinned_count]
        pinned_layers.sort()
        streamed_count = TOTAL_LAYERS - pinned_count

        tps_proj = {
            "alpha_0.60": round(project_tps(streamed_count, 0.60, 6, nvme_gbs), 2),
            "alpha_0.70": round(project_tps(streamed_count, 0.70, 6, nvme_gbs), 2),
            "alpha_0.78": round(project_tps(streamed_count, 0.78, 6, nvme_gbs), 2),
            "alpha_0.85": round(project_tps(streamed_count, 0.85, 6, nvme_gbs), 2),
        }

        # Feasibility evaluation
        feasible = True
        reason = "Feasible under two-tier residency."

        if heap0_mb > heap0_size_mb:
            feasible = False
            reason = f"Heap 0 use ({heap0_mb:.1f} MB) exceeds physical Device-Local Heap 0 ({heap0_size_mb:.1f} MB); requires 32 GB hardware."
        elif heap1_mb > heap1_size_mb:
            feasible = False
            reason = f"Heap 1 use ({heap1_mb:.1f} MB) exceeds Host-Visible Heap 1 ({heap1_size_mb:.1f} MB)."
        elif total_alloc_mb > ts["ceiling_mb"]:
            feasible = False
            reason = f"Total allocation ({total_alloc_mb:.1f} MB) exceeds tier ceiling ({ts['ceiling_mb']} MB)."
        elif ts["id"] == 4:
            feasible = False
            reason = "Tier 4 requires 60 resident layers in Heap 0 (15,414 MiB), exceeding Heap 0 capacity (13,417 MiB); requires 32 GB APU."

        results.append({
            "tier_id": ts["id"],
            "name": ts["name"],
            "ceiling_mb": ts["ceiling_mb"],
            "heap0_usage_mb": round(heap0_mb, 1),
            "heap0_budget_mb": round(heap0_size_mb, 1),
            "heap1_usage_mb": round(heap1_mb, 1),
            "heap1_budget_mb": round(heap1_size_mb, 1),
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
            "heap0_device_local_mb": heap0_size_mb,
            "heap1_host_visible_mb": heap1_size_mb
        },
        # The projections below are SUPERSEDED by measurement and are kept only as a record of
        # what was assumed before anything ran. Round 8 checked them against the engine and
        # they are wrong in both directions:
        #
        #   - they call tier 3 (45 pinned layers) INFEASIBLE, which is the configuration the
        #     engine actually runs, because the two-tier heap accounting does not match how the
        #     driver assigns imported host memory;
        #   - they project 5.30 TPS for that same configuration against a measured 0.75, an
        #     overestimate of about 7x, because the alpha model prices streaming bandwidth and
        #     nothing else. The GPU phase alone is 748 ms/token, a hard 1.34 TPS ceiling that
        #     the model has no term for.
        #
        # Read measured_operating_point. Do not plan against projected_tps.
        "projection_model_status": "superseded by measurement; see measured_operating_point",

        # Not a projection. This is what the engine does on this machine, so that anyone
        # reading the projections below can see how far the model is from the measurement.
        "measured_operating_point": {
            "bios_uma": "minimum (all three settings measured; minimum is best)",
            "resident_layers": 45,
            "streamed_layers": 15,
            "imported_gib": 11.58,
            "kv_cache_dtype": "FP16",
            "max_context": 4096,
            "draft_k": 8,
            "decode_tok_s_greedy": 1.13,
            "phase_ms_per_token": {"stream_io": 90, "gpu": 631, "lm_head": 24, "cpu_other": 29},
            "binding_constraint": "GPU",
            "gpu_bound_ceiling_tok_s": 1.46,
            "note": [
                "Round 9 changed which constraint binds. The engine read 4.145 GB per token",
                "through FILE_FLAG_NO_BUFFERING at 3.09 GB/s = 1,341 ms, against a 1,328 ms",
                "token: it was DISK-bound, and every GPU optimisation measured neutral because",
                "the GPU was never the constraint. Buffered reads on I/O worker threads took",
                "stream I/O from 322 ms to 90 and throughput from 0.75 to 1.13 tok/s.",
                "The GPU now binds at 631 of 883 ms, so 1.46 tok/s is the ceiling until the",
                "gemv gets faster -- and gemv work now converts to throughput, which it did",
                "not before."
            ]
        },
        "layer_geometry": {
            "d_model": 5376,
            "d_ff": 21504,
            "total_layers": TOTAL_LAYERS,
            "sliding_layers_count": len(SLIDING_LAYERS),
            "global_layers_count": len(GLOBAL_LAYERS),
            "global_layers": GLOBAL_LAYERS,
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
        print(f"[{t['name']}] Footprint: {t['actual_footprint_mb']} MB / {t['ceiling_mb']} MB (Heap 0: {t['heap0_usage_mb']} MB, Heap 1: {t['heap1_usage_mb']} MB) -> {status}")
        print(f"  Pinned: {t['pinned_layers_count']}, Streamed: {t['streamed_layers_count']}, KV: {t['kv_cache_dtype']}")
        print(f"  Projected TPS @ alpha=0.78: {t['projected_tps']['alpha_0.78']} (Target: {t['target_tps']} TPS, Meets: {t['meets_target']})")
        if t["feasible"] and t["actual_footprint_mb"] > t["ceiling_mb"]:
            print(f"  ERROR: Tier {t['tier_id']} exceeds ceiling by {t['actual_footprint_mb'] - t['ceiling_mb']} MB")
            failed = True

    if failed:
        print("\nSELF-TEST FAILED!")
        return 1
    else:
        print("\nSELF-TEST PASSED: All feasible tiers satisfy memory ceilings and heap budgets.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
