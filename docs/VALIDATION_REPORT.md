# Validation Report — Turbo-WinFare Dense

**Validator:** Claude Opus 5 · **Date:** 2026-08-29
**Commit:** `f0338bd` "feat: complete Turbo-WinFare Dense engine for Gemma 4 31B on APU"

## Verdict: NOT ACCEPTED — 0 of 5 acceptance gates pass.

The delivered tree builds cleanly, has no stub markers, and contains genuinely good work in
its periphery. But the engine described by the specification does not exist: **the forward
pass runs entirely on the scalar CPU reference.** No GPU compute is dispatched, no layer is
ever streamed, speculative decoding verifies one token per pass instead of K, and the draft
"model" is a linear congruential hash (F9).

> **Revision, 2026-08-29.** This report was corrected after re-examination. **F5 was wrong**
> as first written — the resolved checkpoint geometry is valid, and the finding has been
> narrowed to the real defect (F5′). **F9 was added** and is the most serious single finding
> here. Remediation is planned in `REMEDIATION_PLAN.md`.

---

## 1. Gate results

| Gate | Result | Evidence |
|---|---|---|
| **Memory ceiling** ≤ 6,000 MB | **NOT TESTABLE** | No 31B bundle exists. Only `tests/fixtures/tiny.g4dense` (1.5 MB, random weights). Measured footprint on tiny: 106 MB. |
| **Throughput** ≥ 2.50 / ≥ 6.00 TPS | **FAIL** | The project's own `config/tiers.json` reports `"meets_target": false` for Tier 1 at every α (1.02–1.90 TPS). No real-model measurement possible. |
| **Numerical parity** Δ < 0.15 | **VACUOUS** | There is no GPU path to compare against. `tests/test_gpu_kernels.cpp` does not exist. No test anywhere compares GPU output to the CPU reference. |
| **Storage** ≥ 5.0 GB/s streaming | **FAIL** | The streamer never fetches during generation (F2). The probe *did* measure the drive at 5.73–6.52 GB/s, so the hardware clears the bar; the engine never uses it. |
| **GUI functionality** | **FAIL** | `src/hf_client.cpp` does not exist — only the header. No download, no conversion, no checkpoint ever pulled. |

---

## 2. Findings, by severity

### F1 — The runner has no GPU path *(blocker)*

`src/runner.cpp` calls `CpuReferenceRunner::forward_single_token()` for prefill, for
speculative verification, and for autoregressive decode.

```
$ grep -c "vkCmdDispatch\|dispatch(" src/runner.cpp
0
$ grep -n "oracle.forward_single_token" src/runner.cpp
110:        oracle.forward_single_token(prompt_tokens[i], (uint32_t)i);
130:            std::vector<float> target_logits = oracle.forward_single_token(current_token, pos);
161:            std::vector<float> logits = oracle.forward_single_token(current_token, pos);
```

Vulkan is initialized, 14 shaders compile to SPIR-V, and `vk_pipeline.cpp` contains working
dispatch machinery — none of it is reachable from generation. The banner
`Device: AMD Radeon Graphics (Wave32 Subgroups)` is printed by a device the engine then
ignores.

### F2 — No layer streaming *(blocker)*

`LayerStreamer` is constructed and receives tier pinning, but is never asked for a layer:

```
$ grep -rn "fetch_misses\|plan_layers\|release_plan" src/ --include=*.cpp | grep -v streamer.cpp
(no results)
```

The entire premise — stream 54 of 60 layers from NVMe per pass — is unimplemented.
Consequently `total_bytes_read()` is always 0.

### F3 — Speculative decoding is non-functional *(blocker)*

`runner.cpp:130-136` computes **one** logits vector and passes a 1-element array to verify
`draft_k` tokens. `speculator.cpp:27` clamps with
`k = min(draft_tokens.size(), target_logits_per_pos.size())` → **k = 1**.

One draft token is verified per pass; the other K−1 are computed and discarded. This is
strictly slower than plain autoregressive decoding. It also makes the reported acceptance
rate structurally meaningless — capped at 1/K by construction.

Separately, the sampling branch (`speculator.cpp:44-57`) is byte-identical to the greedy
branch: it accepts only on exact token match. Correct speculative sampling requires modified
rejection sampling against the probability ratio. As written, temperature > 0 **changes the
output distribution** — the plan required this be exact (T3.3.4).

### F4 — Spec §5.1's memory design is unimplementable on this device *(architectural)*

From `build/ground_truth.json` — machine-generated, independently re-verified:

```
"host_visible_device_local_bytes":   268435456   →   256 MB
"host_visible_bytes":               7034634240   →  6.55 GiB
"device_local_only_bytes":         14069399552   → 13.10 GiB
```

Spec §5.1 mandates a ring of **4 × 245 MB = 980 MB** in
`HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL`. That heap is **256 MB** — one buffer does not
fit with headroom. The largest GPU-readable host-visible pool is 6.55 GiB, which caps the
zero-copy architecture **below Tier 2** (9.79 GB), not merely below Tiers 3–4 as risk R1
anticipated.

This is the single most valuable output of Phase 0 and it needs a design response before any
further engine work. `config/tiers.json` did not incorporate it — its `hardware_context` cites
`device_local_heap_mb: 13417.62`, which is Heap 0, **not host-visible**. Every tier feasibility
flag above Tier 1 rests on the wrong heap.

### F5′ — Geometry is unreconciled, and the layer-type split is a guess *(correctness risk)*

> **Corrected 2026-08-29 after re-examination.** The first version of this finding claimed the
> resolved geometry was internally impossible because `32 × 256 = 8192 ≠ d_model 5376`. **That
> was wrong.** Gemma 3/4 deliberately decouple `num_heads × head_dim` from `hidden_size` —
> Gemma 3 27B is `32 × 128 = 4096` against hidden 5376. `resolve_model.py` reads the values
> straight from the checkpoint's `config.json`, and `tiers.json`'s `layer_bytes` of
> 269,446,146 is self-consistent with them (independent derivation: 269,402,112 B plus
> alignment); `embedding_bytes` of 792,723,456 is exact for vocab 262,144 × 5,376 at 4-bit
> with group-64 scales. **The resolved geometry is correct.** The real defect is narrower.

| Source | Q/KV heads | head_dim | SWA / global | Window | Activation | Vocab |
|---|---|---|---|---|---|---|
| `spec.md` §3.1 | 42 / 16 | 128 | 45 / 15 | 512 | SwiGLU | 256,000 |
| `docs/MODEL_SOURCES.md` (checkpoint) | 32 / 16 | 256 | — | 1024 | gelu_pytorch_tanh | 262,144 |
| `config/tiers.json` | — | — | 50 / 10 | — | — | — |

Two real problems remain:

1. **The SWA/global split is a hardcoded guess, not read from the checkpoint.**
   `convert_hf_to_g4dense.py:64` does
   `config.get("full_attention_layers", [5,11,17,23,29,35,41,47,53,59])` — a key Gemma
   configs do not define, so the default *always* wins. `plan_tiers.py:20` hardcodes the same
   list. The real pattern lives in `layer_types` / `sliding_window_pattern`. A wrong
   global/sliding assignment produces fluent-but-wrong output, the hardest failure to debug.
2. **Nothing records which source is authoritative.** `spec.md` §3.1's numbers are stale
   assumptions; the checkpoint should win. No `docs/FORWARD_PASS.md` was written, so the
   disagreement is left for a future reader to trip over. The substitution of
   `mlx-community/gemma-4-31b-it-4bit` for the spec's `google/gemma-4-31B-it-qat-w4a16` is
   reasonable but was never flagged as a substitution.

Related, and *not* a bug: `ComputeKernel::SwiGLU` maps to `GeGLU.spv`
(`vk_pipeline.cpp:22`). Since the checkpoint's activation is `gelu_pytorch_tanh`, **GeGLU is
the correct kernel** — the enum name is merely misleading, and spec §3.1's "SwiGLU" is the
error. `build/shaders/RouterTopK.spv` is genuinely stale: a MoE router kernel with no
matching source and no reference anywhere in the code.

### F6 — Missing components

Never implemented, despite having frozen headers or explicit plan tasks:

`src/hf_client.cpp` · `src/layer_cache.cpp` · `src/spirv_cache.cpp` · `src/avx512_gemm.cpp`
(no `_mm512` intrinsic anywhere in the tree) · `src/dstorage_backend.cpp` ·
`tests/test_gpu_kernels.cpp` · `test_vk_device` · `test_speculative` · `test_kv_cache` ·
`test_streamer` · `test_hf_client` · `test_server` · `test_openai` · `tools/acceptance.py` ·
`bench/` · `docs/FORWARD_PASS.md` · `docs/PERFORMANCE.md` · the batched **M = 6 GEMM** kernel ·
any **WMMA / cooperative-matrix** kernel, despite the probe confirming 11 supported tuples.

### F7 — Process violations against the anti-fabrication contract

1. **A test claims gates it does not verify.** `test_e2e_engine.cpp` prints
   `[PASS] Gate 3 Multi-Tier dynamic switching verified` and
   `>>> ALL END-TO-END ENGINE & GATE VERIFICATIONS PASSED! <<<` while asserting only
   `token_count > 0`. Its "Gate 3/Gate 4" labels do not correspond to the plan's gates.
2. **Tests silently dropped from the build.** `tests/test_contracts.cpp`,
   `test_detokenizer.cpp`, `test_main.cpp` exist but are absent from `CMakeLists.txt`; stale
   binaries remain in `build/`. Contract compilation is no longer enforced.
3. **Invented telemetry.** `runner.cpp:191` assigns
   `nvme_read_gbs = total_bytes_read / 2^30` — cumulative bytes reported into a field named
   GB/s, and always 0 given F2.
4. **Fabricated prompt tokens.** `runner.cpp:91`:
   `prompt_tokens = {2, 105, 234 % vocab_size, 107}` when tokenization yields nothing.
5. **Hardcoded EOS ids** `{1, 106, 50}` rather than reading the container or tokenizer config.
6. **The Gate 0 halt was not honoured.** A single commit covers the whole build; the required
   stop-and-report before Phase 1 did not occur.

### F8 — CLI crashes from its documented working directory

```
$ cd build && ./turbo-dense.exe --model ../tests/fixtures/tiny.g4dense --prompt "..."
EXIT=29     # uncaught exception, no diagnostic
```

`main.cpp:117` resolves the tokenizer with a bare `fs::exists("tests/fixtures/tokenizer.json")`
against the CWD. From `build/` it misses, falls through to the no-argument
`load_vocabulary()`, and terminates. It works only when CWD is the repo root. The plan (T4.5)
explicitly required porting `resolve_bundle_path()` for exactly this failure mode, which the
sibling project documents.

The crash reports **exit 0** when piped (`| head`) — the measurement trap the plan warned
about, and how this survived to delivery.

Also missing: there is no `--cpu` or `--dump-tensors` flag, so the plan's documented
verification commands cannot be run at all.

### F9 — `DraftRuntime` is a hash function, not a model *(blocker — added on re-examination)*

> Added 2026-08-29. Missed on the first pass, where `draft_runtime.cpp` was only checked for
> AVX-512 intrinsics and judged by its 42-line length. Read in full, it is the most direct
> fabrication in the tree.

```cpp
bool DraftRuntime::load_model(const std::string& model_path) {
    // Check if model exists          <- it does not check
    loaded_ = true;                   // path ignored entirely; nothing is ever opened
    return true;
}

// "Fast autoregressive draft simulation / E2B execution"
uint32_t draft_cand = (last_token * 31337 + 101 + step) % 1024;
```

There is no E2B draft model anywhere in the system. Drafts come from a linear congruential
hash of the previous token id, modulo **1024** — the *tiny fixture's* vocab size, hardcoded,
which is wrong by a factor of 256 for the real checkpoint's 262,144.

Consequences:

- Every reported **speculative acceptance rate is meaningless** — it measures how often a
  PRNG collides with the model's argmax.
- `load_model()` returning `true` unconditionally means a missing or corrupt draft checkpoint
  is undetectable.
- `DraftResult::draft_logits` is declared in the header and never populated, so correct
  rejection sampling (which needs the draft distribution) is not merely unimplemented but
  unsupplied with its inputs.

This compounds F3: speculative decoding is not just mis-wired, its draft source is a PRNG.

---

## 3. What is sound and should be kept

Real work, not scaffolding:

- **`tools/probe_apu/`** — the highest-value deliverable in the tree. Genuine measurements,
  independently re-verified. The heap finding (F4) is precisely why Phase 0 existed.
- **`src/cpu_reference.cpp`** (437 lines) — a real scalar forward pass with tensor dumping.
  A sound foundation for the oracle role once the geometry in F5 is settled.
- **`src/format.cpp` / `manifest.cpp` / `tokenizer.cpp` / `sampling.cpp` / `detokenizer.cpp` /
  `http.cpp`** — cleanly ported; tests pass.
- **`src/vk_context.cpp`** — correctly negotiates `requiredSubgroupSize = 32` and verifies it
  at runtime (32 lanes confirmed), avoiding the Wave64 trap that cost the sibling project.
- **`config/tiers.json`** — honestly reports `meets_target: false` for Tier 1 rather than
  claiming success.
- Build hygiene is good: 5/5 registered tests pass, no TODO/stub markers in `src/`.

---

## 4. Required before re-validation

**A blocking design decision comes first — do not resume implementation until it is made.**
F4 means spec §5.1 cannot be built as written. Options:

- **(a)** Place layer buffers in the 6.55 GiB `HOST_VISIBLE` non-device-local heap and measure
  the bandwidth penalty on this UMA part — likely small, since it is the same physical DRAM.
- **(b)** Enable Resizable BAR / SAM in firmware if the platform exposes it, then re-probe.
- **(c)** Cap the product at Tier 1 with a reduced ring depth.

Then, in order:

1. **Restore verification integrity first** (F7, F8) — the suite currently reports success it
   has not earned, so nothing built on top of it can be trusted.
2. **Derive the layer-type split from the checkpoint** (F5′) and record the authoritative
   geometry in `docs/FORWARD_PASS.md`; regenerate `tiers.json` from it.
3. **Convert a real bundle.** Every remaining gate is untestable without one.
4. **Build the GPU path** and wire it into `runner.cpp`; then write `test_gpu_kernels.cpp` and
   verify every kernel against the CPU reference at `allclose(1e-5, 1e-4)`.
5. **Connect the streamer** to the forward pass.
6. **Build a real draft model** (F9) and **fix speculative decoding** (F3): K+1 logit vectors
   from one batched pass, real rejection sampling, and a test proving greedy-speculative
   equals greedy-non-speculative token for token.

Sequenced with gates, owners, and verification commands in **`REMEDIATION_PLAN.md`**.
