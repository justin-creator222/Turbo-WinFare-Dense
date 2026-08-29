# Antigravity 2.0 kickoff prompt

Paste the block below into Google Antigravity 2.0 as a single `/goal` invocation.
Working directory must be `c:\Users\Justin\Code\Dense Turbo`.

---

```
/goal Build Turbo-WinFare Dense: an APU-optimized streaming inference engine for Gemma 4 31B, running on THIS machine (Lenovo Legion Go S, Ryzen Z1 Extreme, Radeon 780M, 23.8 GB unified memory). You are running on the target hardware — every benchmark you take is real, not simulated.

═══════════════════════════════════════════════════════════════
READ THESE THREE DOCUMENTS BEFORE WRITING ANY CODE
═══════════════════════════════════════════════════════════════

1. IMPLEMENTATION_PLAN.md (repo root) — THE AUTHORITY. Phases, task IDs (T0.1, T3.2.4…),
   gates, and the file-ownership map. Follow it task by task.

2. spec.md (repo root) — the original product spec. WHERE IT DISAGREES WITH THE PLAN,
   THE PLAN WINS. Its header lists which sections are superseded and why.

3. c:\Users\Justin\Code\Turbo\CLAUDE.md — a shipped, working sibling engine for the Gemma 4
   26B MoE variant on this exact hardware, at 8.8 tok/s. You are FORKING this codebase.
   Its 490 lines record measurements, not opinions. Several document bugs that reproduce
   exactly in the dense design. Read it before porting any file or interpreting any benchmark.

═══════════════════════════════════════════════════════════════
FIVE RULES THAT OVERRIDE EVERYTHING ELSE
═══════════════════════════════════════════════════════════════

1. THE SIBLING REPO IS READ-ONLY.
   c:\Users\Justin\Code\Turbo is the user's shipped product. Read it, copy from it — never
   edit, build, benchmark, git-commit, or run anything inside it. Copy files OUT to
   "Dense Turbo" and modify them there.

2. NO FABRICATION. This is the failure mode that killed the previous project's first stage.
   - A missing tool, missing weights, or unsupported GPU feature is a HARD ERROR with a
     diagnostic. Never a stub. Never a fallback that changes numerics.
   - Never write zero-filled or placeholder weight files.
   - Never invent a telemetry number. If you cannot measure it, report null.
   - Never write a test that passes without exercising the thing it names.

3. A TASK IS DONE ONLY WHEN ITS VERIFICATION COMMAND EXITS 0 AND YOU PASTE THE OUTPUT.
   "Implemented" without a passing command is NOT done. Never mark a phase complete with a
   failing gate inside it. If something is blocked, say which gate and why — do not narrow
   scope silently.

4. DO NOT SPAWN SUB-AGENTS BEFORE GATE 2 PASSES.
   Phases 0, 1 and 2 are single-threaded, sequential and blocking. You do all of that work
   yourself. Sub-agents fan out only at Phase 3.

5. HALT AT GATE 0 AND REPORT TO THE USER.
   Gate 0 is where acceptance criteria get renegotiated against measurement. That is a human
   decision, not yours. Stop, present the measured numbers, wait.

═══════════════════════════════════════════════════════════════
ENVIRONMENT — ALREADY VERIFIED, DO NOT REDISCOVER
═══════════════════════════════════════════════════════════════

TOOLCHAIN
  C:\w64devkit\bin\g++.exe          exists, NOT on PATH.
                                    Add C:\w64devkit\bin to PATH first or g++ dies with
                                    "cannot execute 'as'". This is the C++ compiler. Use it.
  cmake.exe / ninja.exe             c:\Users\Justin\Code\Turbo\.venv\Scripts\ (not on PATH)
  python 3.12.13                    same venv; has torch, numpy, safetensors, huggingface_hub
  dxcompiler.dll + dxil.dll         c:\Users\Justin\Code\Turbo\build\ — copy them, do not
                                    re-download unless bootstrap says they are stale

NOT INSTALLED — DO NOT ATTEMPT TO USE
  MSVC / Visual Studio              → WinUI 3 and MSVC builds are OFF THE TABLE. The GUI is
                                      an embedded C++ HTTP server + HTML/JS, ported from the
                                      sibling's server.cpp and gui/. This is decided.
  Vulkan SDK                        → no glslc, no glslangValidator. Compile HLSL to SPIR-V
                                      with DXC: -spirv -fspv-target-env=vulkan1.3

GPU / VULKAN
  Loader   C:\Windows\System32\vulkan-1.dll, version 1.4.341
  ICD      registered under the DISPLAY CLASS key (VulkanDriverName → amdvlk64.dll), NOT
           under HKLM\SOFTWARE\Khronos\Vulkan\Drivers. That key being absent is NORMAL.
           Do not conclude Vulkan is missing.
  Driver   AMD 32.0.23033.1002, dated 2026-04-29
  dstorage.dll is NOT in System32 — it is an app-local NuGet redist. That is expected.

HARDWARE
  RAM 23.8 GB usable (the spec's 32 GB is WRONG). Disk: 334.9 GB free on C:.
  NVMe: Samsung MZAL81T0HDLB, PCIe 4.0. CPU: 8C/16T Zen 4 with AVX-512.

═══════════════════════════════════════════════════════════════
EXECUTION SEQUENCE
═══════════════════════════════════════════════════════════════

PHASE 0 — Ground truth. YOU, ALONE. No feature code.
  T0.1 toolchain bootstrap · T0.2 APU capability probe · T0.3 model source resolution
  · T0.4 budget solver
  T0.2 is the highest-value task in the whole plan. It measures the Vulkan heap budget,
  cooperative-matrix support, subgroup size control, and buffered-vs-unbuffered NVMe
  throughput. Four acceptance-gate values currently rest on unverified assumptions and this
  is what replaces them with numbers.
  ▶ GATE 0 → HALT. Report every measured number. Wait for the user.

PHASE 1 — Contracts and skeleton. YOU, ALONE.
  Fork and rename · .g4dense v2 container · freeze all include/g4dense/ headers ·
  CMake + SPIR-V build · synthetic tiny model (4 layers, d_model 256 — this is what lets
  five sub-agents work in parallel without a 15 GB download; build it before Phase 3)
  ▶ GATE 1 → build succeeds, ctest passes, headers FROZEN.

PHASE 2 — Weights pipeline and the numerical oracle. YOU, ALONE.
  Track A: streaming HF→.g4dense converter, then convert the real checkpoints.
  Track B: the CPU FP32 reference forward pass. THIS IS THE ORACLE. Without it, a GPU
  kernel bug is indistinguishable from a plausible activation — that is exactly how the
  sibling's Wave64 bug survived visual inspection.
  ▶ GATE 2 → CPU path produces coherent Gemma 4 text on the real 31B bundle.

PHASE 3 — NOW spawn five sub-agents, in parallel. Disjoint file sets, no overlaps:
  SA1 storage-engine   streamer, layer_cache, manifest, dstorage_backend, test_streamer,
                       test_format
  SA2 vulkan-compute   vk_context, vk_pipeline, spirv_cache, shaders/**, test_gpu_kernels,
                       test_vk_device
  SA3 speculative      runner, draft_runtime, speculator, kv_cache, avx512_gemm,
                       test_speculative, test_kv_cache
  SA4 downloader       tools/**, hf_client, test_hf_client
  SA5 gui              gui/**, server, openai_api, telemetry, test_server, test_openai
  Give each sub-agent: its plan section, the tiny model, and the sibling files it ports from.
  NO TWO SUB-AGENTS MAY WRITE THE SAME FILE. Headers in include/g4dense/ are frozen — a
  sub-agent needing a contract change reports it to you; it does not edit the header.

PHASE 4 — Integration. The primary correctness gate: GPU greedy output must equal CPU
  reference greedy output TOKEN FOR TOKEN on the real bundle. Also measure the true
  speculative acceptance rate; if it is materially below 0.78, re-run the throughput
  projection and report — do not quietly miss the gate.

PHASE 5 — Performance. Benchmark methodology in the plan is BINDING: interleave A/B/A/B for
  ≥3 rounds and compare medians, verify the disk queue length reads 0 first, and check exit
  codes (a crash greps as an empty metric row that reads like a slow result).

PHASE 6 — Acceptance gates as executable checks, then hand off for validation.

═══════════════════════════════════════════════════════════════
YOU HAVE GONE WRONG IF…
═══════════════════════════════════════════════════════════════

· You are writing Vulkan or kernel code and Phase 0 is not finished.
· You spawned sub-agents before Gate 2 passed.
· You edited, built, or ran anything inside c:\Users\Justin\Code\Turbo.
· You hardcoded a wave/subgroup width of 32. RDNA 3 runs a 256-thread group as Wave64.
  Query WaveGetLaneCount(); force Wave32 explicitly where WMMA needs it.
· You wrote a placeholder, stub, or zero-filled file to make something "work".
· You marked a task done without pasting a passing command's output.
· You are trying to install Visual Studio, MSVC, WinUI 3, or the Vulkan SDK.
· Your ground-truth document still contains "TBD".
· You concluded Vulkan is unavailable because HKLM\SOFTWARE\Khronos\Vulkan\Drivers is empty.

═══════════════════════════════════════════════════════════════
START HERE
═══════════════════════════════════════════════════════════════

1. Read IMPLEMENTATION_PLAN.md, spec.md, and the sibling CLAUDE.md.
2. Confirm back to me, in under 200 words: the phase order, where sub-agents fan out, and
   the three spec defects the plan resolves. Do not restate the whole plan.
3. Then begin T0.1.
```

---

## Why the prompt is shaped this way

| Guard | Failure it prevents |
|---|---|
| Read-order block first | Flash writing code before reading the plan — the single most likely failure |
| Sibling repo READ-ONLY, stated twice | Destructive edits to the user's shipped product |
| "No sub-agents before Gate 2" | `/goal` fanning out immediately because the spec mentions sub-agents |
| HALT at Gate 0 | Gate renegotiation happening without human review |
| Environment block | Wasted cycles rediscovering the toolchain, or wrongly concluding Vulkan is missing |
| "You have gone wrong if…" | Cheap self-check that catches drift without re-reading the plan |
| Confirm-back step | Verifies comprehension before any file is written, at low cost |
