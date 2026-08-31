# Third-party software and model terms

Turbo-WinFare Dense's own source and documentation are licensed under the
[Apache License 2.0](LICENSE). That license applies to this repository's source only. It does
not relicense model weights, the upstream work this is derived from, or any third-party
component listed below.

This file is an attribution aid, not legal advice. Anyone distributing a compiled build must
also preserve the license and NOTICE material required by the exact component versions included
in that build.

## Upstream work

Turbo-WinFare Dense is a derivative work of
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) by Andrey Mikhaylov, licensed under
the Apache License 2.0, by way of Turbo-WinFare — a Direct3D 12 Windows port of that work by
the same copyright holder as this repository.

No Swift or Metal source from the original is vendored here. This is an independent C++20 /
Vulkan 1.3 implementation targeting a different model, and it defines its own container format
rather than retaining the upstream `.gturbo` bundle. See [NOTICE](NOTICE) for the attribution
and the statement of modification required by Apache-2.0 §4(b).

## Model weights

**Model weights are not included in this repository and must not be redistributed with releases
of it.** `.g4dense` containers are excluded by [`.gitignore`](.gitignore) for exactly this
reason.

[`tools/convert_hf_to_g4dense.py`](tools/convert_hf_to_g4dense.py) repacks a locally downloaded
checkpoint into a `.g4dense` container on your machine. Downloaded weights, and any container
built from them, remain a separate work governed by their source terms.

**The two models this engine runs are under different terms. Check both.**

| Model | Pinned revision | Terms as published on the model card |
|---|---|---|
| [`mlx-community/gemma-4-31b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-31b-it-4bit) | `696d436c404745a59f30e4939a658162b0a9e57f` | `apache-2.0`, with a [Gemma 4 license link](https://ai.google.dev/gemma/docs/gemma_4_license) |
| [`mlx-community/gemma-4-e2b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit) | `238767527555cb75a05732a84dff5d6ba0dd6809` | **`gemma`** — Google's Gemma Terms of Use, *not* Apache 2.0 |

The Gemma Terms of Use carry a prohibited-use policy and redistribution conditions that the
Apache License does not. If you ship, host, or offer a service built on the draft model, read
those terms; they are the binding ones for it.

Each container also carries the checkpoint's `tokenizer.json` and chat template, copied through
unchanged and governed by the same terms as the weights they came with.

## Runtime dependencies

| Component | Source | Terms |
| --- | --- | --- |
| `vulkan-1` | Vulkan loader, installed with the GPU driver | System component; not redistributed here |
| `ws2_32`, `dxgi` | Windows SDK import libraries | Microsoft Windows SDK license; system components, not redistributed here |
| Vulkan headers | [KhronosGroup/Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | Apache-2.0. Fetched into `third_party/` at setup time by [`tools/bootstrap.py`](tools/bootstrap.py); **not committed to this repository** |
| `dxcompiler.dll`, `dxil.dll` | [microsoft/DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler) release assets | DXC is under the LLVM/University of Illinois Open Source License with Microsoft additions; `dxil.dll` is a Microsoft-signed binary under its own redistribution terms. Fetched at setup time by `tools/bootstrap.py`; **never committed**. Check the exact release's terms before redistributing them in a binary package |

DXC is used at build time to compile HLSL to SPIR-V. It is not linked into the engine.

No third-party source is vendored into the engine itself. [`CMakeLists.txt`](CMakeLists.txt)
declares no external package dependency beyond the system libraries above; every file under
`src/`, `include/`, `shaders/`, `gui/`, `tools/` and `tests/` is original to this project or
derived from the upstream work described in [NOTICE](NOTICE).

## Build-time tools

Used to build the software; not linked into it or distributed with it.

| Tool | Source | Terms |
| --- | --- | --- |
| w64devkit (MinGW-w64 GCC toolchain) | [skeeto/w64devkit](https://github.com/skeeto/w64devkit) | Unlicense for the distribution; bundled GCC/binutils are GPL, with the GCC Runtime Library Exception covering compiled output |
| CMake | Kitware | BSD-3-Clause |
| Ninja | ninja-build | Apache-2.0 |
| Python 3 | Python Software Foundation | PSF License |

`tools/bootstrap.py` and `tools/convert_hf_to_g4dense.py` use only the Python standard library.
The optional validation tooling (`tools/numpy_reference.py`, `tools/dtype_probe.py`) uses NumPy
(BSD-3-Clause) and is not required to build or run the engine.

## Binary distribution

If you publish a compiled release:

* Include [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
* Include the license text for the exact DXC release whose `dxcompiler.dll` / `dxil.dll` you
  ship, and confirm that release permits redistribution.
* Do **not** include model weights or a prebuilt `.g4dense` container.
