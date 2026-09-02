<div align="center">
  <img src="docs/assets/banner.png" alt="Turbo-WinFare Dense" width="800" />

  # Turbo-WinFare Dense

  **APU-Optimized Native Vulkan Streaming Inference Engine for Gemma 4 31B Dense**

  [![Vulkan 1.3](https://img.shields.io/badge/Vulkan-1.3-red?style=flat-square&logo=vulkan&logoColor=white)](https://www.vulkan.org/)
  [![Gemma 4 31B](https://img.shields.io/badge/Target_Model-Gemma_4_31B_Dense-blue?style=flat-square)](https://huggingface.co/google/gemma-4-31b-it)
  [![Gemma 4 E2B](https://img.shields.io/badge/Draft_Model-Gemma_4_E2B-green?style=flat-square)]()
  [![AMD Radeon 780M](https://img.shields.io/badge/Hardware-AMD_Radeon_780M-ED1C24?style=flat-square&logo=amd&logoColor=white)](https://www.amd.com/)
  [![INT4 Quant](https://img.shields.io/badge/Quantization-MLX_INT4_G64-blueviolet?style=flat-square)]()

</div>

---

**Run a 31-billion-parameter model on a handheld gaming PC.**

Turbo-WinFare Dense is a Vulkan 1.3 inference engine for **Gemma 4 31B Dense**, built for AMD
APUs — specifically a Lenovo Legion Go S (Ryzen Z1 Extreme, Radeon 780M, 32 GB LPDDR5X). It also
runs **Gemma 4 E2B**, both on its own and as a speculative draft model for the 31B.

### The problem, and the approach

Quantized to 4 bits the 31B still needs **15 GiB** of weights, and the graphics driver stops
accepting memory around **11.75 GiB**. The model does not fit. There is no configuration of this
hardware where it fits.

So the engine does not try to hold it all. **45 of the 60 transformer layers stay resident in
memory, and the other 15 are read from NVMe on every single token** — about 4 GB per token,
continuously, while the GPU is working. The whole design follows from that one constraint:

- **Nothing is copied twice.** On an APU the CPU and GPU share physical memory, so resident
  layers are handed to the GPU as ordinary system memory it reads in place, rather than being
  uploaded to a separate pool.
- **The layers that stream are chosen, not left over.** They are spread evenly through the
  stack so each disk read overlaps the compute of the layers around it, instead of arriving in
  one stall at the end.
- **Reading and computing happen at once.** Disk reads run on their own threads, so the GPU
  never waits on the filesystem, and a read served from the OS cache costs nothing on the
  critical path.
- **A small model can guess ahead, when the text is predictable enough.** Gemma 4 E2B drafts
  several tokens and the 31B verifies them in a single pass. This is **off by default** and
  `--spec` turns it on: loading the drafter reserves 1.5 GiB, which costs the 31B 6 resident
  layers, and only rote or rigid-format output is predicted well enough to earn that back.
  Measured across 10 prompts it wins on primes (1.50×) and JSON (1.18×) and loses on every
  conversational one. [PERFORMANCE.md](docs/PERFORMANCE.md) §5 has the full matrix.

### What that costs

Streaming a quarter of the model from disk every token gives up **roughly a third of the speed**
full residency would deliver. That is the interesting part: the weights are 28% larger than the
memory the driver will give them, and the penalty is nowhere near proportionate.

**Measured: 1.08–1.13 tok/s** greedy decode, 4096 context, at 45 of 60 layers resident — the
spread is between sessions on the same machine, which is why nothing here is concluded from a
single run. Full residency is estimated at **1.7–2.0 tok/s** and is *not reachable on this
hardware*: the driver refuses long before 60 layers, so that figure is an extrapolation from
two measured endpoints rather than something anyone has observed.
[PERFORMANCE.md](docs/PERFORMANCE.md) shows the working.

It also ships an **OpenAI-compatible API and a web UI**, so anything that speaks to a local LLM
server speaks to this.

## Documentation

| | |
|---|---|
| [**Performance**](docs/PERFORMANCE.md) | Every measured number: where the time goes, hardware baselines, memory footprint, throughput, and the benchmarking rules. |
| [**Architecture**](docs/ARCHITECTURE.md) | How the engine is put together, and the constraint the design follows from. |
| [**Forward pass**](docs/FORWARD_PASS.md) | The arithmetic, constant by constant, against the upstream modelling source. |
| [**Container format**](docs/G4DENSE_FORMAT.md) | The `.g4dense` v3 spec. |
| [**Model sources**](docs/MODEL_SOURCES.md) | Pinned checkpoint revisions. |
| [**Hardware ground truth**](docs/GROUND_TRUTH.md) | What `tools/probe_apu` measured on this machine. |
| [**Contributing**](CONTRIBUTING.md) | Build setup, the correctness gate, and rules that exist because something broke. |

**Round reports** — what changed and why, newest first. These carry the reasoning, including
the ideas that did not work:
[10](docs/ROUND10_REPORT.md) · [9](docs/ROUND9_REPORT.md) · [8](docs/ROUND8_REPORT.md) ·
[7](docs/ROUND7_REPORT.md) · [6](docs/ROUND6_REPORT.md) · [5](docs/ROUND5_REPORT.md) ·
[4](docs/ROUND4_REPORT.md) · [3](docs/ROUND3_REPORT.md). Earlier planning documents are archived in
[docs/history/](docs/history/).

## Sibling project — Turbo-WinFare

[**Turbo-WinFare**](https://github.com/justin-creator222/Turbo-WinFare) applies the same idea to a
*mixture-of-experts* model instead of a dense one. It is a **Direct3D 12** (DirectCompute) engine
that runs **Gemma 4 26B-A4B** on the same class of hardware (Legion Go S, Radeon 780M). Where this
repository keeps 45 of 60 layers resident and streams the other 15, Turbo-WinFare keeps only the
1.35 GB of non-expert weights resident and reads the eight experts each layer actually routes to
from NVMe per token. Because so little is active per token, it is far faster: **9.9 tok/s** at 24
expert slots, **16.2 tok/s** at 44.

Separate codebase, separate container format, separate GUI and OpenAI-compatible server. The two
projects share the design brief — an APU whose driver will not hand out as much memory as the model
needs — not code.

## System requirements

The engine streams whatever it cannot hold, so it runs on far less memory than the model's
size suggests — it just gets slower. Measured on this machine, forcing residency down with
`G4DENSE_MAX_RESIDENT_LAYERS`:

| resident layers | process RAM | throughput | |
|---:|---:|---:|---|
| 0 / 60 | 3.9 GB | 0.40 tok/s | reads all 16.8 GB of weights every token |
| 8 / 60 | 4.2 GB | 0.43 tok/s | |
| 21 / 60 | 5.7 GB | 0.46 tok/s | what a 16 GB machine allows |
| **45 / 60** | **~9.7 GB** | **1.13 tok/s** | 32 GB, the configuration this was built for |

**~3.9 GB is the floor**, and it is not the layers — it is the KV cache (1,120 MB at 4096
context), the LM head (756 MB) and the four streaming slots (1,146 MB), all allocated whatever
the residency. Below that the engine cannot start.

In practice:

- **32 GB** — what this targets. 45 of 60 layers resident.
- **16 GB** — works. Residency is capped at 38% of system RAM, so ~21 layers, and expect
  roughly **0.45 tok/s**. Most of the model is read from disk every token, so the result
  depends heavily on your SSD and on how much of the 16.5 GB file the OS can keep cached.
- **8 GB** — the process fits, but only just, and with almost nothing left for page cache
  every token reads the whole model from storage. Treat it as possible, not usable.

Note the flat part of that curve: 0 → 21 resident layers barely moves throughput, because at
low residency the pass is bound by reading weights, not by computing with them. Residency only
starts paying once enough layers stay put to leave the disk idle.



```powershell
.\build\run_turbo_dense.exe --model models\gemma-4-e2b-dense.g4dense --prompt "Hi" --max-tokens 24
```

Also needed, regardless of RAM: a **Vulkan 1.3** GPU with subgroup intrinsics (this targets an
AMD Radeon 780M), **Windows**, and about **20 GB of disk** for the two containers.

## Build

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full setup. In short:

```powershell
python tools/bootstrap.py          # fetch w64devkit, DXC, Vulkan headers
python tools/bootstrap.py --verify # confirm the toolchain works
$env:PATH = "C:\w64devkit\bin;" + $env:PATH
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe
cmake --build build
```

## Run

```powershell
.\build\run_turbo_dense.exe --model models\gemma-4-31b-dense.g4dense --prompt "Hi" --max-tokens 24
.\build\run_turbo_dense.exe --model models\gemma-4-31b-dense.g4dense --gui     # web UI + OpenAI API
```

Useful flags: `--spec` (enable speculative decoding; off by default because the drafter costs
the 31B 6 resident layers), `--draft-k N` (verify-batch width, 2–8, default 6), `--max-context N`
(default 4096; 8192 costs ~336 MB of KV cache, about one resident layer), `--temp`, `--top-p`,
`--top-k`.

### The web UI

`--gui` serves a control panel and an OpenAI-compatible API on the same port. Everything the
CLI can set is settable there — sampling (temperature, top-p, top-k, repetition penalty, seed),
generation length, speculation on/off and its K, and context length — plus live telemetry:
per-token phase breakdown, memory by pool, which layers are resident against which stream, and
speculative acceptance with its denominator.

Two things it deliberately does not do: it never substitutes a plausible value for telemetry it
does not have (missing readings show `—`), and it does not claim a speculative speedup, because
measuring one honestly means running the same prompt both ways and the server does not do that.

| endpoint | |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible, streaming or not |
| `GET /v1/models`, `GET /api/models` | available containers |
| `GET/POST /api/config` | sampling, speculation, context; a context change reports `requires_reload` |
| `GET /api/telemetry` | throughput, phase breakdown, memory, speculation counters |
| `GET /api/model_info` | geometry, and which layers stream |
| `POST /api/load_model`, `/api/unload_model` | swap the container in place |
| `POST /api/reset_kv`, `/api/clear_cache`, `/api/stop` | reset context, drop stream slots, cancel |
| `GET /api/setup` | what is installed, and whether Python/`huggingface_hub` are available |
| `POST /api/download_model` | start a download + conversion; `GET /api/download_status` polls it |
| `POST /api/download_cancel`, `/api/delete_checkpoint` | stop a running job; reclaim a source checkpoint |

## Models

**The easiest way is the web UI.** Start the server with `--gui` and, if no model is installed,
a **Get Models** panel appears with each model's download size, total disk cost, and a button.
It downloads from Hugging Face and converts locally — nothing is uploaded, and neither
repository is gated, so no account or token is needed.

| | download | on disk when done | roughly |
|---|---:|---:|---|
| Gemma 4 E2B | 3.4 GB | ~6 GB | start here — it is also the draft model |
| Gemma 4 31B Dense | 18 GB | ~35 GB | the main model |

Conversion leaves both the source checkpoint and the container on disk. The panel offers to
reclaim the checkpoint afterwards; you would only need it again to re-convert.

That feature runs the Python tools below as a child process, so it needs Python 3 with
`huggingface_hub` installed. The UI says so plainly if either is missing. To do it by hand
instead:

```powershell
python tools/convert_hf_to_g4dense.py --input models\gemma-4-31b-it-4bit --out models\gemma-4-31b-dense.g4dense
python tools/verify_weights.py models\gemma-4-31b-dense.g4dense

# or download, convert and verify in one step -- this is exactly what the UI runs:
python tools/fetch_model.py --model 31b
python tools/fetch_model.py --check     # report what is installed
```

[docs/G4DENSE_FORMAT.md](docs/G4DENSE_FORMAT.md) is the format spec. **A v2 container is rejected, not upgraded** — v3
16-byte-aligns the packed-weight blocks, so a v2 file read as v3 decodes at the wrong offsets
and produces plausible-looking garbage. Weights and `.g4dense` bundles are gitignored and never
committed.

## Test fixtures are not in git, and go stale silently

`*.g4dense` is gitignored, which includes two things the test suite depends on. Neither
regenerates automatically, and a stale copy fails in ways that look like engine bugs — a stale
`oracle_tensors/` had `run_gpu_forward_test` and `run_smoke_engine_test` failing for an entire
round against an oracle that predated a softcapping change.

```powershell
# The 4-layer synthetic fixture (fast, no GPU or real model needed)
python tools\make_synthetic_model.py --out tests\fixtures\tiny.g4dense
python tools\make_synthetic_model.py --out tests\fixtures\tiny.g4dense --verify

# The CPU-oracle dumps the GPU forward pass is graded against.
# Minutes per token: it runs the full 31B on the CPU.
.\build\run_cpu_reference_test.exe models\gemma-4-31b-dense.g4dense tests\fixtures\oracle_tensors 2
```

**Regenerate both after any change to the container layout.** The per-layer layout is spelled
out in five places ([G4DENSE_FORMAT.md §4.2](docs/G4DENSE_FORMAT.md#42-where-this-layout-is-written-down) lists them) and they must move together.

## Verifying a change

```powershell
ctest --test-dir build --output-on-failure     # 16 of 16

.\build\run_gpu_forward_test.exe models\gemma-4-31b-dense.g4dense tests\fixtures\oracle_multi "2,3689,563,506"
.\build\run_gpu_forward_test.exe models\gemma-4-e2b-dense.g4dense  tests\fixtures\oracle_e2b   "2,3689,563,506"
.\build\run_real_generation_test.exe 24        # READ THE TEXT -- it asserts only that tokens appeared
```

Both models must stay argmax- and top-5-exact against their oracles. `run_real_generation_test`
cannot judge coherence; a human has to read the output.

**On benchmarking**, from [PERFORMANCE.md](docs/PERFORMANCE.md) §6 — each rule cost a wrong conclusion:
three runs minimum per variant, compare variants within one session, confirm the disk is idle
first, and never pipe a verification command (`| head` once reported exit 0 for a process
crashing with exit 29).

## Layout

| | |
|---|---|
| [`src/`](src/), [`include/g4dense/`](include/g4dense/) | engine — [`runner.cpp`](src/runner.cpp) is the forward pass, [`streamer.cpp`](src/streamer.cpp) the layer I/O |
| [`shaders/`](shaders/) | HLSL compiled to SPIR-V by [`tools/compile_shaders.py`](tools/compile_shaders.py) |
| [`tools/`](tools/) | conversion, verification, the NumPy reference, benchmarks |
| [`docs/`](docs/) | [PERFORMANCE.md](docs/PERFORMANCE.md), [G4DENSE_FORMAT.md](docs/G4DENSE_FORMAT.md), [FORWARD_PASS.md](docs/FORWARD_PASS.md), per-round reports |
