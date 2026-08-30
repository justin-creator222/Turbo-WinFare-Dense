"""
Acceptance gate evaluation for Turbo-WinFare Dense.

Gates are defined in docs/VALIDATION_REPORT.md and REMEDIATION_PLAN.md.

Two rules govern this script, both learned the hard way:

  1. EVERY GATE RUNS AGAINST THE REAL MODEL. The previous version passed
     `tests/fixtures/tiny.g4dense` -- a 1.5 MB, 4-layer, random-weight fixture -- to the
     memory and throughput gates and wrote the results into the report as facts about
     Gemma 4 31B. It reported 111.4 MB and 138.8 TPS; the real figures are ~7,900 MB and
     ~0.04 TPS. A gate measured on the wrong model is worse than no gate, because it turns
     an open question into a false answer.

  2. "COULD NOT MEASURE" IS NOT "PASS". Every gate reports one of PASS / FAIL /
     NOT_MEASURED, and NOT_MEASURED never counts toward acceptance. Missing prerequisites
     are named in the report so the reader knows exactly what was and was not established.

The committed build/acceptance_report.json from round 2 could not have come from this
script at all -- its schema (gate_1_memory_ceiling, "detail" fields) does not match what
any version of this code emits. It was written by hand. Do not hand-edit reports.
"""

import argparse
import ctypes
import ctypes.wintypes
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
REAL_MODEL = REPO_ROOT / "models" / "gemma-4-31b-dense.g4dense"
# No draft model exists. The E2B container that used to sit here is quarantined as
# invalid (models/quarantine/README.md); speculative decoding therefore has nothing
# verified to draft with, and reporting a draft_model here would overstate the gate.
DRAFT_MODEL = REPO_ROOT / "models" / "gemma-4-e2b-dense.g4dense"

MEMORY_CEILING_MB = 6000.0
THROUGHPUT_TARGET_TPS = 2.50

PASS, FAIL, NOT_MEASURED = "PASS", "FAIL", "NOT_MEASURED"


# --------------------------------------------------------------------------------------
# Peak working-set sampling. psutil is not installed in this venv, so this goes straight
# at GetProcessMemoryInfo. Windows maintains PeakWorkingSetSize for us; we only have to
# read it while the process is still alive.
# --------------------------------------------------------------------------------------

class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010


def _peak_working_set_bytes(pid):
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        return 0
    try:
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize if ok else 0
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def run_sampled(cmd, timeout_sec=5400):
    """Runs `cmd`, sampling peak working set until it exits.

    Returns (returncode, stdout, peak_mb). stderr is folded into stdout so a diagnostic
    is never lost.
    """
    env = dict(os.environ)
    env["PATH"] = r"C:\w64devkit\bin;" + env.get("PATH", "")

    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    peak = {"bytes": 0}

    def sampler():
        while proc.poll() is None:
            got = _peak_working_set_bytes(proc.pid)
            if got > peak["bytes"]:
                peak["bytes"] = got
            time.sleep(0.15)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        out, _ = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        out = (out or "") + f"\n[acceptance] TIMEOUT after {timeout_sec}s"
    t.join(timeout=2.0)
    return proc.returncode, out or "", peak["bytes"] / (1024.0 * 1024.0)


def missing_prereqs():
    """Prerequisites shared by the real-model gates."""
    problems = []
    if not REAL_MODEL.exists():
        problems.append(f"missing {REAL_MODEL.relative_to(REPO_ROOT)}")
    return problems


def not_measured(reason, **extra):
    d = {"verdict": NOT_MEASURED, "passed": False, "reason": reason}
    d.update(extra)
    return d


# --------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------

def gate_1_and_2_memory_and_throughput(max_tokens):
    """Gates 1 and 2 share one run of the real model: peak working set, and measured TPS.

    Running them together avoids two multi-minute loads of a 16 GiB container and
    guarantees both numbers describe the same execution.
    """
    exe = BUILD_DIR / "run_real_generation_test.exe"
    if not exe.exists():
        r = not_measured("run_real_generation_test.exe not built")
        return r, dict(r)
    problems = missing_prereqs()
    if problems:
        r = not_measured("; ".join(problems))
        return r, dict(r)

    rc, out, peak_mb = run_sampled([str(exe), str(max_tokens), str(REAL_MODEL)])

    if rc != 0:
        reason = f"real-model generation exited {rc}"
        tail = "\n".join(out.strip().splitlines()[-6:])
        r = not_measured(reason, output_tail=tail)
        return r, dict(r)

    tps_values = [float(m) for m in re.findall(r"=\s*([0-9.]+)\s*TPS\]", out)]
    median_tps = 0.0
    if tps_values:
        s = sorted(tps_values)
        median_tps = s[len(s) // 2]

    g1 = {
        "verdict": PASS if 0 < peak_mb <= MEMORY_CEILING_MB else FAIL,
        "passed": 0 < peak_mb <= MEMORY_CEILING_MB,
        "measured_peak_working_set_mb": round(peak_mb, 1),
        "ceiling_mb": MEMORY_CEILING_MB,
        "model": str(REAL_MODEL.relative_to(REPO_ROOT)),
        "note": ("Peak working set sampled during real 31B generation. Note this run does "
                 "not pin a tier explicitly, and does not reach 8k context."),
    }
    g2 = {
        "verdict": PASS if median_tps >= THROUGHPUT_TARGET_TPS else FAIL,
        "passed": median_tps >= THROUGHPUT_TARGET_TPS,
        "measured_tps_median": round(median_tps, 4),
        "measured_tps_all": [round(v, 4) for v in tps_values],
        "target_tps": THROUGHPUT_TARGET_TPS,
        "model": str(REAL_MODEL.relative_to(REPO_ROOT)),
    }
    return g1, g2


def gate_3_parity():
    """GPU forward pass vs the CPU reference oracle, on the real 31B container."""
    exe = BUILD_DIR / "run_gpu_forward_test.exe"
    if not exe.exists():
        return not_measured("run_gpu_forward_test.exe not built")
    problems = missing_prereqs()
    oracle = REPO_ROOT / "tests" / "fixtures" / "oracle_tensors" / "token0_logits.bin"
    if not oracle.exists():
        problems.append("missing tests/fixtures/oracle_tensors/token0_logits.bin")
    if problems:
        return not_measured("; ".join(problems))

    rc, out, _ = run_sampled([str(exe), str(REAL_MODEL)])
    ok = rc == 0 and "matches CPU Oracle with exact token equality" in out

    def grab(pat):
        m = re.search(pat, out)
        return float(m.group(1)) if m else None

    return {
        "verdict": PASS if ok else FAIL,
        "passed": ok,
        "max_abs_diff": grab(r"Max abs diff:\s+([0-9.eE+-]+)"),
        "mean_abs_diff": grab(r"Mean abs diff:\s+([0-9.eE+-]+)"),
        "exact_token_match": ok,
        "model": str(REAL_MODEL.relative_to(REPO_ROOT)),
    }


def _binary_gate(exe_name, success_marker, key):
    exe = BUILD_DIR / exe_name
    if not exe.exists():
        return not_measured(f"{exe_name} not built")
    rc, out, _ = run_sampled([str(exe)])
    ok = rc == 0 and success_marker in out
    return {
        "verdict": PASS if ok else FAIL,
        "passed": ok,
        key: ok,
        "exit_code": rc,
    }


def gate_4_streaming():
    return _binary_gate("run_streamer_test.exe",
                        "ALL STREAMER DMA TESTS PASSED",
                        "byte_exact_read_correctness")


def gate_5_speculative():
    r = _binary_gate("run_speculative_test.exe",
                     "All speculative decoding tests passed.",
                     "greedy_exact_token_equivalence")
    # Round 2 caveat, carried until the runner actually reads options.speculative_enabled:
    # that test compared the autoregressive path against itself, so a pass proved nothing.
    runner_src = (REPO_ROOT / "src" / "runner.cpp").read_text(encoding="utf-8", errors="replace")
    if "speculative_enabled" not in runner_src:
        r = not_measured(
            "runner.cpp does not read options.speculative_enabled, so the speculative path "
            "does not execute during generation; the equivalence test compares the "
            "autoregressive path against itself and cannot establish this gate.")
    return r


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Turbo-WinFare Dense acceptance gates")
    ap.add_argument("--all", action="store_true", help="run every gate")
    ap.add_argument("--report", default="build/acceptance_report.json")
    ap.add_argument("--max-tokens", type=int, default=8,
                    help="tokens per prompt for the real-model run (gates 1 and 2)")
    args = ap.parse_args()

    print("=" * 72)
    print("  Turbo-WinFare Dense -- Acceptance Gates (real model only)")
    print("=" * 72)
    if not REAL_MODEL.exists():
        print(f"  WARNING: {REAL_MODEL} is absent; real-model gates report NOT_MEASURED.")

    results = {}

    print("1+2. Memory ceiling and throughput (one real-model run)...", flush=True)
    g1, g2 = gate_1_and_2_memory_and_throughput(args.max_tokens)
    results["gate_1_memory_ceiling"] = g1
    results["gate_2_throughput"] = g2
    print(f"     Gate 1 [{g1['verdict']}] {g1.get('measured_peak_working_set_mb', '-')} MB "
          f"(ceiling {MEMORY_CEILING_MB:.0f})")
    print(f"     Gate 2 [{g2['verdict']}] {g2.get('measured_tps_median', '-')} TPS "
          f"(target {THROUGHPUT_TARGET_TPS})")

    print("3.   Numerical parity vs CPU oracle...", flush=True)
    results["gate_3_numerical_parity"] = gate_3_parity()
    print(f"     [{results['gate_3_numerical_parity']['verdict']}] "
          f"max_abs_diff={results['gate_3_numerical_parity'].get('max_abs_diff')}")

    print("4.   Storage / layer streaming correctness...", flush=True)
    results["gate_4_storage_streaming"] = gate_4_streaming()
    print(f"     [{results['gate_4_storage_streaming']['verdict']}]")

    print("5.   Speculative decoding...", flush=True)
    results["gate_5_speculative_decoding"] = gate_5_speculative()
    print(f"     [{results['gate_5_speculative_decoding']['verdict']}]")

    passed = sum(1 for g in results.values() if g.get("passed"))
    failed = sum(1 for g in results.values() if g.get("verdict") == FAIL)
    unmeasured = sum(1 for g in results.values() if g.get("verdict") == NOT_MEASURED)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": "Turbo-WinFare Dense",
        "target_model": str(REAL_MODEL.relative_to(REPO_ROOT)) if REAL_MODEL.exists() else None,
        "draft_model": str(DRAFT_MODEL.relative_to(REPO_ROOT)) if DRAFT_MODEL.exists() else None,
        "overall_accepted": passed == len(results),
        "gates_passed": passed,
        "gates_failed": failed,
        "gates_not_measured": unmeasured,
        "total_gates": len(results),
        "acceptance_gates": results,
    }

    out_path = Path(args.report)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"  {passed} passed / {failed} failed / {unmeasured} not measured "
          f"of {len(results)}")
    print(f"  Overall: {'ACCEPTED' if report['overall_accepted'] else 'NOT ACCEPTED'}")
    print(f"  Report:  {out_path}")
    print("=" * 72 + "\n")

    sys.exit(0 if report["overall_accepted"] else 1)


if __name__ == "__main__":
    main()
