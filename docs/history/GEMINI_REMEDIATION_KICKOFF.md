# Antigravity 2.0 remediation kickoff

Paste the block below into Antigravity 2.0 as a single `/goal`.
Working directory: `<repo root>`.

---

```
/goal Remediate Turbo-WinFare Dense. The previous round was VALIDATED AND REJECTED — 0 of 5
acceptance gates pass. You are fixing a real codebase, not starting over. Much of it is good
and must be preserved.

═══════════════════════════════════════════════════════════════
READ IN THIS ORDER BEFORE TOUCHING ANY FILE
═══════════════════════════════════════════════════════════════

1. docs/VALIDATION_REPORT.md  — what was found wrong, with evidence. Nine findings, F1–F9.
2. REMEDIATION_PLAN.md        — THE AUTHORITY. Phases R0–R6, gates, task ids.
3. <sibling engine>\CLAUDE.md — the sibling engine. READ-ONLY. Never edit or build it.

spec.md is superseded wherever it disagrees with REMEDIATION_PLAN.md. Its §3.1 geometry and
§5.1 memory design are both known-wrong on this hardware.

═══════════════════════════════════════════════════════════════
WHAT WENT WRONG LAST TIME — READ THIS TWICE
═══════════════════════════════════════════════════════════════

The tree you delivered built cleanly, passed 5/5 tests, and had zero TODO or stub markers.
It still failed every gate, because:

  · runner.cpp calls the CPU reference for every token. Zero vkCmdDispatch. The GPU is
    initialized, printed to the banner, and then ignored.
  · The streamer is fully implemented and never called once.
  · Speculative decoding passes ONE logits vector to verify K drafts, so k = min(K,1) = 1.
  · DraftRuntime::load_model() ignores its path and returns true. Drafts come from
    (last_token * 31337 + 101 + step) % 1024 — a hash function presented as a neural network.
  · test_e2e_engine.cpp prints "ALL END-TO-END ENGINE & GATE VERIFICATIONS PASSED!" while
    asserting only token_count > 0.
  · nvme_read_gbs reports cumulative bytes, in a field named GB/s, that is always zero.

None of this was caught by "no stubs" or "tests pass". The rules below target THAT failure
mode specifically. They are not boilerplate — each one names a thing that actually happened.

═══════════════════════════════════════════════════════════════
SEVEN RULES — VIOLATING ANY ONE FAILS THE ROUND
═══════════════════════════════════════════════════════════════

1. NO TEST MAY PRINT A GATE VERDICT IT DID NOT EVALUATE.
   A test asserting token_count > 0 may not print "[PASS] Gate 3". Gate names in output must
   match gate definitions in REMEDIATION_PLAN.md. If you did not check it, do not claim it.

2. EVERY REPORTED METRIC MUST TRACE TO A MEASUREMENT.
   If you cannot measure it, emit null. Naming a field _gbs and assigning cumulative bytes is
   fabrication, not a rounding error.

3. A FUNCTION THAT IGNORES ITS ARGUMENT IS A STUB, HOWEVER SHORT.
   load_model(path) that never opens path is the canonical case. If a component is not
   implemented, LEAVE IT UNIMPLEMENTED AND SAY SO. Never synthesize plausible-looking output
   to make a call site work.

4. RUN VERIFICATION COMMANDS WITHOUT A PIPE, AND PASTE THE OUTPUT.
   `./turbo-dense.exe ... | head` reported EXIT=0 while the process was crashing with exit 29.
   That masked a hard crash for an entire round. Redirect to a file, then check $? and cat it.

5. DO NOT REMOVE A FAILING TEST FROM THE BUILD.
   Three test files were dropped from CMakeLists.txt last round. Report failures as failures.

6. HALT AT GATE R0 AND GATE R2. Report and wait.
   R0 re-baselines the throughput gate against measurement. R2 is the first moment any claim
   about the real model is possible. These are human decision points.

7. DO NOT REWRITE WHAT ALREADY WORKS. The plan §6 lists what to keep. In particular
   streamer.cpp, vk_pipeline.cpp, kv_cache.cpp and cpu_reference.cpp are sound — R3 and R4 are
   WIRING jobs, not rewrites. If you find yourself rewriting one of these, stop and re-read.

═══════════════════════════════════════════════════════════════
ENVIRONMENT — VERIFIED, DO NOT REDISCOVER
═══════════════════════════════════════════════════════════════

  PATH first:  C:\w64devkit\bin    (else g++ fails with "cannot execute 'as'")
  cmake/ninja/python:  <sibling engine>\.venv\Scripts\   (has torch, numpy,
                       safetensors, huggingface_hub)
  DXC:         build\dxc.exe, dxcompiler.dll, dxil.dll — already present
  Vulkan:      loader 1.4.341; ICD registered under the DISPLAY CLASS key, NOT under
               HKLM\SOFTWARE\Khronos\Vulkan\Drivers. That key being empty is NORMAL.
  NOT PRESENT: MSVC, Visual Studio, Vulkan SDK. Do not try to install or use them.
  Disk:        330 GB free. You need ~35 GB for the checkpoint plus the converted bundle.

  MEASURED HEAPS — the whole memory design turns on these:
      Heap 0   13.10 GiB   DEVICE_LOCAL only, NOT host-visible
      Heap 1    6.55 GiB   HOST_VISIBLE | HOST_COHERENT, not device-local
      Heap 2     256 MB    both  <- what spec §5.1 assumed was large. It is not.

═══════════════════════════════════════════════════════════════
PHASE ORDER
═══════════════════════════════════════════════════════════════

R0  Re-measure and reconcile.  YOU, ALONE. No sub-agents.
    The warm-cache benchmark reports 5.82 GB/s against 6.52 cold — backwards and
    implausible for a 4 GB file in 23.8 GB of RAM. Fix it; Tier 1's throughput gate depends
    entirely on that number. Probe Heap 0 vs Heap 1 GPU read bandwidth. Derive the
    SWA/global layer split from the checkpoint's layer_types — the current code falls back
    to a hardcoded guess every time. Regenerate tiers.json with per-heap accounting.
    ▶ HALT. Report the re-baselined throughput projection.

R1  Restore verification integrity.  YOU, ALONE.
    Fix the lying test, restore the dropped tests, fix the telemetry, delete the fabricated
    prompt fallback, fix the exit-29 CWD crash, add --cpu and --dump-tensors, add the
    build-time SPIR-V step. Nothing downstream is trustworthy until this is done.

R2  Real weights. Start the 17 GiB download DURING R0/R1 — it is on the critical path for
    every remaining gate.
    ▶ HALT after the CPU reference produces coherent Gemma 4 text on the real bundle.

R3  GPU forward pass — sub-agents may fan out here, not before.
    Two-heap allocator, then wire the existing pipeline manager into runner.cpp ONE KERNEL
    AT A TIME, each diffed against the CPU reference via --dump-tensors before adding the
    next. Then test_gpu_kernels.cpp at allclose(1e-5, 1e-4).

R4  Streaming. Call plan_layers/fetch_misses/release_plan. Depth-3 prefetch. Pinned layers
    to Heap 0 via staging; ring to Heap 1, zero-copy.

R5  Real draft model + correct speculative verification. Delete the LCG. K+1 logit vectors
    from one batched pass. Modified rejection sampling, not exact-match.

R6  Remaining components, benchmarks, acceptance gates.

═══════════════════════════════════════════════════════════════
YOU HAVE GONE WRONG IF…
═══════════════════════════════════════════════════════════════

· A test prints PASS for something it did not check.
· A metric has a value the engine did not measure.
· A function takes a path, filename, or handle and never opens it.
· You are rewriting streamer.cpp, vk_pipeline.cpp, kv_cache.cpp, or cpu_reference.cpp.
· You piped a verification command through head, tail, or grep and read the exit code.
· You proceeded past Gate R0 or Gate R2 without reporting.
· You edited anything under <sibling engine>.
· You spawned sub-agents before R3.
· You hardcoded a subgroup width of 32, or a vocab size, or a layer-type pattern.

═══════════════════════════════════════════════════════════════
START HERE
═══════════════════════════════════════════════════════════════

1. Read the three documents above.
2. Reply in under 250 words with: (a) the four things that made the last round fail
   validation despite passing its own tests, (b) which files you must NOT rewrite, and
   (c) what R0.1 is measuring and why it matters. Do not restate the plan.
3. Then begin R0.1.
```

---

## Why this prompt differs from the first one

The first kickoff guarded against *not doing the work*. It failed to guard against **doing
work that reports itself as complete** — which is what actually happened. The changes:

| Change | Targets |
|---|---|
| A "what went wrong last time" section quoting real code | Abstract rules did not bind; concrete ones might |
| Rule 1: no gate verdict without evaluation | `[PASS] Gate 3` on a `token_count > 0` assert |
| Rule 3: ignoring an argument makes it a stub | `load_model()` returning true, F9 |
| Rule 4: no pipes on verification commands | `\| head` masked exit 29 for a whole round |
| Rule 7 + "do not rewrite" list | Preserves the ~60% that is genuinely sound |
| Sub-agents forbidden until R3 | R0–R2 are sequential and each halts |
| Confirm-back asks what *failed*, not what the plan says | Tests comprehension of the failure mode, not recall |
