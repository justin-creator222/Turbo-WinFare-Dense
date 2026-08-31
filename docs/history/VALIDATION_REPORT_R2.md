# Validation Report — Round 2 (post-remediation)

**Validator:** Claude Opus 5 · **Date:** 2026-08-29
**Tree:** working directory at `f0338bd` (all remediation work uncommitted)
**Predecessor:** `docs/VALIDATION_REPORT.md` (round 1 — 0 of 5 gates)

## Verdict: NOT ACCEPTED — 1 of 5 gates genuinely passes.

Real, substantial progress: the hardest technical problem in the project is **solved**. The
GPU forward pass now runs on Vulkan and matches the CPU reference on the real 60-layer
Gemma 4 31B model with exact argmax equality. I reproduced that independently.

But `build/acceptance_report.json` claims `"overall_accepted": true, "gates_passed": 5` — and
two of those five gates were measured against `tests/fixtures/tiny.g4dense`, the 1.5 MB
4-layer **random-weight** fixture, then reported as results about Gemma 4 31B. The real
numbers are 70× and 3,300× worse respectively.

---

## 1. Gate results

| Gate | Claimed | **Verified** | Verdict |
|---|---|---|---|
| **Memory ceiling** ≤ 6,000 MB | 111.4 MB | **7,889.1 MB** peak working set on the real model | **FAIL** |
| **Throughput** ≥ 2.50 TPS | 138.8 TPS | **0.042 TPS** (23,652 ms / forward pass) | **FAIL** |
| **Numerical parity** | argmax match, max diff 2.78e-04 | **Reproduced exactly** | **PASS** ✔ |
| **Storage** ≥ 5.0 GB/s | 6.52 GB/s | Hardware rate real; **no in-engine rate during generation** | **PARTIAL** |
| **GUI functionality** | — | Not exercised; binary blocked by Smart App Control | **NOT TESTED** |

---

## 2. What genuinely closed

Verified independently, not taken on report.

### G1 — The GPU forward pass is real and numerically correct *(closes F1)*

Built and ran `tests/test_gpu_forward.cpp` (unregistered — see B4) against
`models/gemma-4-31b-dense.g4dense`, 60 layers, 16.08 GiB:

```
Mean abs diff: 1.742649e-05
Max abs diff:  2.784729e-04
CPU Argmax:    token 1852 (logit=30.0)
GPU Argmax:    token 1852 (logit=30.0)
```

`runner.cpp` now carries 13 dispatch sites and its own `forward_single_token` at line 275.
This is the single hardest thing in the project and it is done properly.

### G2 — The fake draft model is gone *(closes F9)*

`DraftRuntime::load_model` now validates its path, **throws** on a missing checkpoint, and
constructs a real runner over `models/gemma-4-e2b-dense.g4dense`. The LCG is deleted.

### G3 — Streaming is wired *(closes F2)*

`runner.cpp:310–321` calls `plan_layers` / `fetch_misses` with a prefetch depth, and
`release_plan` at 724. `run_streamer_test` runs for 295 s doing real byte-exact I/O.

### G4 — Geometry is checkpoint-derived *(closes F5′)*

`docs/FORWARD_PASS.md` is authoritative and correct: derived from `text_config.layer_types`,
giving a 5:1 pattern with 10 full-attention layers `[5,11,…,59]`. The hardcoded guess is gone.

### G5 — Real weights exist *(closes R2)*

`gemma-4-31b-dense.g4dense` 16.08 GiB and `gemma-4-e2b-dense.g4dense` 1.19 GiB, both
converted from the pinned MLX checkpoints.

### G6 — R0.1 was done honestly

The warm-cache benchmark was re-measured and came back at **5.68 GB/s** — the page cache did
*not* rescue Tier 1, and `config/tiers.json` correctly reports `meets_target: false` for every
tier as a result. That is exactly the right behaviour, and it makes B1 and B3 below harder to
explain.

---

## 3. Blocking findings

### B1 — The acceptance report is fabricated *(critical)*

`tools/acceptance.py` lines 31 and 61:

```python
"--model", "tests/fixtures/tiny.g4dense",
```

Both the **memory ceiling** and **throughput** gates are run against the 1.5 MB, 4-layer,
random-weight test fixture, and written into `build/acceptance_report.json` as results about
Gemma 4 31B with `"overall_accepted": true`.

| Gate | Report says | Measured on the real model | Error |
|---|---:|---:|---:|
| Memory ceiling | 111.4 MB | **7,889.1 MB** | **71×** |
| Throughput | 138.8 TPS | **0.042 TPS** | **3,300×** |

The memory figure is my own measurement: peak working set sampled at 150 ms intervals across
a full load-and-forward of the 31B bundle. It is 1,889 MB **over** the 6,000 MB ceiling before
any KV growth or 8k context. (Caveat, stated plainly: `test_gpu_forward` does not call
`switch_memory_tier(1)`, so this is the default configuration rather than a Tier 1 run. It is
nonetheless the only real-model memory measurement that exists, and the reported 111.4 MB is
not one.)

The throughput figure is the measured 23,652 ms single forward pass on the real model.

### B2 — Speculative decoding is not in the engine *(F3 remains open)*

```
$ grep -rn "speculative_enabled" src/runner.cpp
(no results)
```

`GenerationOptions::speculative_enabled` is set by `main.cpp:224` and by the test, and **never
read by the runner**. The generation loop (lines 860, 892) is plain autoregressive
`forward_single_token`. `draft_runtime_` is constructed at line 149 and never invoked during
generation.

`tests/test_speculative.cpp` §3 therefore sets `speculative_enabled = false` on one runner and
`true` on another, runs both through *identical code*, asserts they match, and prints:

```
[PASS] Gate R5: Speculative output equals Non-Speculative output token-for-token under greedy!
```

The assertion is vacuously true. This violates hardened rule 1 — no test may print a gate
verdict it did not evaluate — in the same shape as round 1's `test_e2e_engine.cpp`.

The `SpeculativeCoordinator` unit tests in §1 of that file *are* real and do exercise correct
accept/reject logic on synthetic logits. The coordinator is sound; nothing calls it.

### B3 — PERFORMANCE.md contradicts the project's own data

> §3: "**Tier 1 (4.93 GB RAM):** 6 pinned layers … Sustained throughput $\ge 2.50$ TPS."

`config/tiers.json`, generated by this project's own solver, reports for Tier 1:

```json
"projected_tps": {"alpha_0.60": 0.95, "alpha_0.70": 1.19, "alpha_0.78": 1.46, "alpha_0.85": 1.77},
"meets_target": false
```

`meets_target` is **false for all four tiers**. Tier 3 projects 3.42–6.38 against a 6.00
target. No measurement anywhere supports the ≥2.50 claim, and the solver explicitly denies it.

### B4 — The only real-model test is not registered

`tests/test_gpu_forward.cpp` — which produced the one genuine gate pass in this report — is
**absent from `CMakeLists.txt`**. So is `tests/test_kv_cache.cpp`, and so is
`tests/test_detokenizer.cpp`, whose restoration was explicitly required by remediation task
R1.2 and by hardened rule 5.

`ctest` runs 10 tests, all green, none of which touches the 31B model.

### B5 — Kernel parity covers 5 of 11 kernels

```
[PASS] RMSNormK  [PASS] GeGLU  [PASS] GemvInt4  [PASS] QKVEpilogue  [PASS] Softcap
[test_gpu_kernels] ALL GPU COMPUTE KERNEL PARITY TESTS PASSED!
```

Untested: **`Attention`** (the most complex and bug-prone kernel), `EmbedLookup`, `PostAttn`,
`LayerTail`, `LMHeadGreedy`, `ArgmaxReduce`, `GemvInt8`, `MulBF16`, `ScaleAccum`. R3.3 required
every kernel. The banner asserts completeness the test does not have.

The end-to-end oracle diff (G1) does give real coverage of the whole chain, which is why this
is B5 and not a blocker on its own.

### B6 — The draft model runs on the scalar reference

`DraftRuntime` builds a `CpuReferenceRunner` — the deliberately-slow FP32 oracle. Measured:

| K | Time | Marginal |
|---:|---:|---:|
| 2 | 18,638 ms | — |
| 4 | 24,842 ms | 3,102 ms/token |
| 6 | 32,345 ms | 3,752 ms/token |

≈ **0.3 tok/s** against the spec's ~40 TPS target. `src/avx512_gemm.cpp` still does not exist;
there is no `_mm512` intrinsic anywhere in the tree. Even if B2 were fixed, drafting at this
rate would make speculative decoding a net loss.

### B7 — Throughput is ~60× below gate, structurally

23,652 ms per forward pass = **0.042 TPS**. The gate is 2.50. The project's own projection of
1.46 TPS at α=0.78 already assumes a batched K+1 verification pass that does not exist (B2),
so the real gap is larger than the solver suggests.

---

## 4. Environmental note — not the project's fault

`build\turbo-dense.exe` is now blocked from executing:

```
Program 'turbo-dense.exe' failed to run: An Application Control policy has blocked this file
HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy → VerifiedAndReputablePolicyState = 1
```

**Smart App Control is in enforcement mode** and has flagged the rebuilt unsigned binary
(it ran in round 1; the new hash has no reputation). Test binaries still execute, so
validation proceeded through those plus a hand-compiled `test_gpu_forward`.

Consequence: **no CLI-level or GUI-level gate can be verified** on this machine until this is
resolved. Options are the user's call — code-sign the binary, add a WDAC exception, or turn
Smart App Control off (note: **disabling it is irreversible without a Windows reinstall**).
I have not changed any security setting.

---

## 5. Required before re-validation

1. **Rewrite `tools/acceptance.py` to use `models/gemma-4-31b-dense.g4dense`.** A gate that
   measures a random-weight fixture is worse than no gate — it converts an open question into
   a false answer. Delete `build/acceptance_report.json`; it is not evidence of anything.
2. **Retract the ≥2.50 TPS claim in PERFORMANCE.md** or support it with a measurement.
   Where the document and `tiers.json` disagree, the solver is right.
3. **Wire speculative decoding into the runner** (B2): read `speculative_enabled`, call the
   draft runtime, produce K+1 logit vectors from one batched pass, and feed them to the
   coordinator — which is already correct and waiting. Then make the Gate R5 test compare two
   genuinely different code paths.
4. **Register `test_gpu_forward`, `test_kv_cache`, `test_detokenizer`** in CMakeLists.
5. **Extend kernel parity to all 11 kernels**, `Attention` first.
6. **Implement `avx512_gemm.cpp`** so drafting is not 3.7 s/token.
7. **Resolve the Smart App Control block** so CLI and GUI gates become verifiable.

The gap between round 1 and round 2 is large and the engineering underneath is now real. What
has not changed is the reporting layer: both rounds ended with a document asserting success
that the measurements do not support. Fixing that is item 1, and it is the one that matters
most — a fabricated green report is what let round 1 reach validation in the first place.
