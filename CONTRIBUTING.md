# Contributing to Turbo-WinFare Dense

Thanks for taking a look. A few things are specific to this project and will save you time.

## Scope

Turbo-WinFare Dense is a **Vulkan 1.3 compute** inference engine for **Gemma 4 31B Dense**,
built for AMD APUs on Windows. Shaders are HLSL compiled to SPIR-V with DXC. There is no
upstream project and no other platform: this is a self-contained engine, and the design follows
almost entirely from one constraint — the model's weights are 15.06 GiB in INT4, the driver
stops accepting host-memory imports at about 11.75 GiB, so 15 of 60 layers stream from disk on
every token.

Read [`README.md`](README.md) first, then [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The
per-round reports in `docs/` record *why* things are the way they are, which is usually the
question that matters.

## Setting up

```powershell
python tools/bootstrap.py          # fetch w64devkit, DXC, Vulkan headers
python tools/bootstrap.py --verify # confirm the toolchain works
$env:PATH = "C:\w64devkit\bin;" + $env:PATH
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe
cmake --build build
ctest --test-dir build --output-on-failure
```

MSVC is not supported.

Shaders are compiled separately, and only when you change one:

```powershell
python tools/compile_shaders.py
```

### Which tests need what

There is no CI. Everything below runs locally.

| | |
|---|---|
| **No GPU, no model** | `run_format_test`, `run_tokenizer_test`, `run_detokenizer_test`, `run_sampling_test`, `run_contracts_test`, `run_prompt_pipeline_test`, `run_kv_cache_test`, `run_streamer_test` |
| **Vulkan device** | `run_vk_device_test`, `run_gpu_kernels_test` |
| **Vulkan device + a real container** | `run_gpu_forward_test`, `run_smoke_engine_test`, `run_speculative_test`, `run_real_generation_test`, `run_context_dependence_test`, `run_cpu_reference_test` |

`run_gpu_kernels_test` needs a Vulkan 1.3 device with subgroup (wave) intrinsics.

### Test fixtures are not in git

`*.g4dense` is gitignored, which includes two things the suite depends on. Neither regenerates
automatically, and a stale copy fails in ways that look like engine bugs — a stale
`oracle_tensors/` had two ctest cases failing for an entire round against an oracle that
predated a softcapping change.

```powershell
python tools\make_synthetic_model.py --out tests\fixtures\tiny.g4dense
.\build\run_cpu_reference_test.exe models\gemma-4-31b-dense.g4dense tests\fixtures\oracle_tensors 2
```

Regenerate both after any change to the container layout. `docs/G4DENSE_FORMAT.md` section 4.2
lists the five places that layout is written down; they must move together.

## The correctness gate

This project's whole claim is that GPU output is *right*, not just plausible. Wrong inference
output is fluent, and fluent-but-wrong is the hardest failure mode to notice.

**If you touch a kernel, a forward-pass constant, the streamer, the KV cache, or the container
layout, run all three of these and say so in the PR:**

```powershell
.\build\run_gpu_kernels_test.exe

.\build\run_gpu_forward_test.exe models\gemma-4-31b-dense.g4dense tests\fixtures\oracle_multi "2,3689,563,506"
.\build\run_gpu_forward_test.exe models\gemma-4-e2b-dense.g4dense  tests\fixtures\oracle_e2b   "2,3689,563,506"

.\build\run_real_generation_test.exe 24
```

The oracle diff must be **argmax- and top-5-exact at every position**, on **both** models —
the engine runs two architectures (31B Dense, and E2B with per-layer embeddings and KV sharing)
and it is easy to fix one while breaking the other. `run_real_generation_test` asserts only that
tokens appeared; **a human has to read the text.**

`--dump-tensors <dir>` (with `--cpu`) writes per-stage FP32 tensors for the first token, which
is how you find *where* two paths diverge rather than just that they do.

Kernel agreement uses numpy's `allclose` rule (`|got-want| <= 1e-5 + 1e-4*|want|`), not a pure
relative test — GPU transcendentals differ from libm by a couple of ULPs, and near-zero results
then show a huge relative error for ~1e-7 absolute. A real logic bug is never marginal.

### A test that cannot fail is worse than no test

Two have shipped here. The gemv parity test passed 0 for all three byte offsets, so it never
exercised the offset arithmetic the real model uses — and it passed while the engine produced
garbage. The attention cases all fit inside two tiles, so they could not detect a broken
online-softmax rescale. **When you add a test, break the thing it covers on purpose and confirm
it fails.**

Equally: a reference implementation written from the same assumptions as the engine cannot
catch a shared misconception. Seven forward-pass bugs survived GPU-vs-CPU parity in round 5
because both paths shared each mistake. `tools/numpy_reference.py` is transcribed from the
upstream modelling source for exactly this reason.

## Rules that exist because something broke

* **No fabricating paths.** Never let a missing model, tokenizer, shader compiler, or telemetry
  value degrade into a plausible-looking substitute. Missing weights, a missing tokenizer and a
  missing DXC are all hard errors on purpose. Telemetry that reports struct defaults is worse
  than telemetry that reports nothing, because it looks authoritative.
* **Check every `vkQueueSubmit`.** An unchecked failing submit once reported ~0 ms of GPU time
  and "1.93 TPS" alongside empty output.
* **Save shaders as ASCII with no BOM.** `Set-Content -Encoding ascii`, never `-Encoding utf8`.
  DXC rejects a BOM with a misleading "non-ASCII characters are not allowed" pointing at the
  wrong line.
* **Do not assume a wave width.** RDNA 3 runs a 256-thread group as Wave64. Use
  `WaveGetLaneCount()` / `WaveIsFirstLane()`, and remember that a `GroupMemoryBarrierWithGroupSync`
  guarantees arrival, not that every thread has finished the *read* that follows it.
* **Do not add a flag that parses and does nothing.** A clean "unknown argument" error is better
  than a silently ignored option. Unknown arguments exit 2 by design.
* **Never commit weights, `.g4dense` bundles, or build output.** `.gitignore` covers these;
  check `git status` before committing anyway.

## Benchmarking

Run-to-run drift on this class of machine is larger than most single optimizations, and this
project has twice drawn a conclusion from a single run that later evaporated.

* **Three runs minimum per variant**, compared **within one session**. A before-session vs
  after-session comparison is not evidence.
* **Confirm the disk is idle first.** Benchmarks taken while storage is still settling once
  turned a clean run into an apparent 40% regression.
* **Never pipe a verification command.** `| head` once reported exit 0 for a process crashing
  with exit 29, hiding a hard crash for an entire round.
* **Check which resource is saturated before optimizing it.** Three rounds of GPU work measured
  neutral here because the engine was disk-bound and nobody had done the division.

[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) §6 keeps this list current.

## Style

Match the surrounding code: C++20, `g4dense::` namespace, headers in `include/g4dense/`.
Comments in this codebase explain *why*, especially where a previous approach was wrong — a
comment recording a measurement that killed an idea saves the next person from re-running it.
That convention is worth keeping.

## Licensing of contributions

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the same license as the project. If you port logic from another
project, say so in the PR so it can be recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
