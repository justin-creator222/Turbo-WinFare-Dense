# Round 8 — the outstanding issues, closed

**Author:** Claude Opus 5 · **Date:** 2026-08-30 · **Implemented by:** me

## Summary

All seven phases are done. The headline is smaller than the plan projected and the reasons are
measured rather than guessed.

**Measured, three runs each, `--prompt "Hi" --max-tokens 14 --temp 0 --no-spec`:**

| | round 7 | round 8 |
|---|---:|---:|
| stream I/O | 566–585 ms | **318–326 ms** |
| GPU queue | 625–631 ms | 746–752 ms |
| throughput | 0.667–0.677 tok/s | **0.750–0.760 tok/s** |

**+12%**, entirely from choosing *which* layers stream. Two of the plan's other three
performance ideas turned out not to help, and saying so is most of this report.

The context ceiling is gone: attention is an online softmax and `--max-context 8192` works.

---

## Phase 1 — round 7 committed

Four commits on `round-7-engine`. Nothing had been committed, so every phase below had a floor
to fall back to.

## Phase 2 — the server reserved for a draft it never loaded

`--server` / `--gui` reserved 1.5 GiB for a draft and then returned before the line that loads
it, so the GUI ran at **39 resident layers instead of 45 and never speculated** — it paid the
whole cost and got none of the benefit. The draft now loads before the server branch.

Telemetry was reporting struct defaults at the same time (0 layers, `max_context` 8192 against
an actual 4096) and usage counts of 0/0/0. Both fixed; verified end to end against the running
GUI.

## Phase 3 — spread the streamed layers *(the round's only real win)*

`load_resident_layers()` imported greedily from layer 0, so at 45 of 60 resident the streamed
layers were **45–59: all at the end of the pass, back to back**. Each read is ~35 ms and the
prefetch queue runs 3 deep, so there was ~35 ms of GPU work available to hide 15 consecutive
reads behind.

Choosing evenly-spaced layers to stream puts several resident layers' compute between
consecutive reads. Same 45 layers, same bytes, purely a scheduling change:

| | greedy tail | evenly spaced |
|---|---:|---:|
| stream I/O | 566–585 ms | **318–326 ms** |
| throughput | 0.667–0.677 tok/s | **0.750–0.760 tok/s** |

## Phase 4 — the gemv, explained at last, and it does not matter yet

`gemv_int4_row_lane` issued **eight separate 4-byte loads per quantization group** for its
weights — 256 requests per wave for 1024 contiguous bytes. Rounds 6 and 7 had eliminated
occupancy and activation loads; this was the remaining candidate and it was the right one.

| | GPU phase | throughput |
|---|---:|---:|
| 8 × `u32_load` | 848–856 ms | 0.746–0.751 tok/s |
| 2 × `Load4` | **746–748 ms** | 0.750–0.760 tok/s |

**The kernel is ~12% faster and generation is not.** At 45 of 60 resident the pass is bound by
streaming, so GPU time saved is returned as I/O wait. E2B, which is fully resident, is unchanged
too (15.9 against 15.8 tok/s) — at 2.5 GB it is not bandwidth-bound to begin with.

The plan said to revert if this measured neutral. It is neutral on throughput but not on the
kernel, so it stays: strictly less work for identical arithmetic, and it converts to throughput
as soon as streaming stops binding.

### It took a container format change, and that found two real defects

Two `Load4`s need 16-byte alignment. The weights had none: the norms are all multiples of 16
bytes but `layer_scalar` is 2, so **every projection in every layer sat at offset 2 (mod 16)**.
The container now pads each packed-weight block, which is a layout change, so the version goes
to **3** — unlike every previous field addition, an old container decodes at the wrong offsets
and produces plausible-looking garbage rather than failing.

The first build of this produced garbage on the real model **while the gemv parity test
passed**, because that test passed 0 for all three byte offsets and so never exercised the
offset arithmetic the real model uses. It now runs at a non-zero offset, and that is the second
time this project has shipped a test that could not fail.

Separately, `tests/fixtures/oracle_tensors` is a gitignored local dump that had gone **stale**,
so `run_gpu_forward_test` and `run_smoke_engine_test` — which default to it — had been failing
before any of this round's work, against an oracle whose argmax logit (16.9) predates round 7's
softcapping. Regenerated. The CPU reference and the GPU now agree exactly at argmax 236773,
logit 29.9723. **ctest is 16 of 16 for the first time this round.**

## Phase 5 — both gates turned out to be unnecessary

**Speculation now wins at every length**, so the length gate would only ever switch off a win:

| tokens | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| `--no-spec` | 10.1 | 14.3 | 18.5 | 22.6 | 30.8 | 38.1 s |
| speculative | 9.2 | 13.0 | 16.6 | 20.5 | 26.2 | 30.1 s |
| ratio | 1.10× | 1.10× | 1.11× | 1.10× | 1.18× | 1.26× |

Round 7 measured a *loss* at 8 tokens (18.6 against 15.1). Phases 3 and 4 removed enough
per-layer cost that the six resident layers the draft displaces no longer dominate.

**Trimming the draft's import reserve makes things worse**, which the plan had backwards. The
reserve is what keeps the *draft* fully resident, and the draft runs K times per verify round:

| reserve | target | draft | 24 tokens |
|---|---:|---:|---:|
| 1536 MiB | 39/60 | 35/35 | 26.2 s |
| 1280 MiB | 40/60 | 32/35 | 26.1 s |
| 1152 MiB | 41/60 | 27/35 | 29.2 s |
| 1024 MiB | 41/60 | 27/35 | 29.6 s |

A streamed draft costs more than an extra resident target layer gains. 1536 stays.

**One unexplained number.** Draft acceptance measures 46.7% greedy and 55.6% at default
temperature, against the single 76.2% figure in the round-7 report. Both of today's runs agree
with each other. Speculative output is still token-for-token identical to non-speculative
(`run_speculative_test`), so this is a throughput observation, not a correctness one, but I
have not accounted for the difference.

## Phase 6 — flash attention

Attention staged every score in `groupshared float s_scores[4096]` and made three passes. That
is one float of LDS per attended key, which is why `max_context` was hard-capped: a
full-attention layer above 4096 wrote **past the end of the array**.

The kernel now walks keys in tiles of 256, carrying a running max, a running denominator and
the accumulator itself, rescaling by `exp(m_old - m_new)` whenever a tile raises the max.
Groupshared drops from 16 KB to ~3 KB and nothing scales with the attended span.
`kAttentionMaxSpan` is replaced by `kAttentionMaxHeadDim` — still a bound, but one set by the
architecture rather than the conversation.

**Performance is neutral**, measured three runs each way: 0.751/748 ms old, 0.753/748 ms new.
That is the correct outcome for a ceiling-removal change.

`--max-context 8192` works: correct text, 45 → 44 resident layers, exactly the ~336 MB of extra
KV cache predicted. It stays opt-in because that is a tax on every generation.

**The new parity cases have teeth.** Deliberately breaking the rescale to a constant leaves
`Attention/global` and `Attention/sliding` **passing** — both fit in about two tiles, where the
running max rarely revises — and fails `Attention/long-span` at 1100 keys. The pre-existing
cases had no power over the path the rewrite introduces.

## Phase 7 — the target re-based, and the model retired

`config/tiers.json` set tier 1 at 2.50 TPS. Full residency needs 15.06 GiB of imports against a
~11.75 GiB driver ceiling, inside a 15.90 GiB heap that also holds the KV cache, LM head and
streaming pool. All three BIOS UMA settings were measured; minimum is the best of them.

Targets are now the measured operating point and the headroom above it: **1.14 tok/s** is the
pass with streaming perfectly hidden, and that is tier 3's target.

Checking the projections against the engine was worth doing on its own — **they are wrong in
both directions**. They mark tier 3 (45 pinned layers) *infeasible*, which is the configuration
the engine actually runs; and they project **5.30 TPS** for it against a measured **0.75**, an
overestimate of ~7×, because the α model prices streaming bandwidth and has no term for the GPU
phase — which alone is 748 ms/token, a hard 1.34 TPS ceiling. `projected_tps` is kept only as a
record of what was assumed before anything ran.

## Verification

```
run_gpu_kernels_test     9 of 11 kernels; 5 of 5 attention cases, max_abs_diff <= 6.3e-07
run_gpu_forward_test     31B and E2B: argmax and top-5 exact at 4 positions + batched prefill
run_real_generation_test 3 of 3 prompts coherent, unchanged in substance
ctest                    16 of 16
```

## What I got wrong this round

- **I quoted single runs as measurements.** Three commits carried numbers from one run on each
  side of a ~0.75 tok/s mean with ±0.015 of spread, and one of them ("GPU 841 → 626 ms") was an
  outlier I nearly built a conclusion on. Re-measured three runs each way and corrected the
  record in `01a4092`. Comparing single runs of a 1.3-second-per-token pass is not measurement,
  and section 6 of `PERFORMANCE.md` has said so since round 3.
- **I changed the container layout in one of four places.** The runner, the converter, the
  synthetic generator and the CPU reference each spell out the per-layer layout; I updated the
  runner, and the CPU oracle then silently graded against the wrong bytes.

## Still open

- **15 layers stream, ~322 ms/token.** A hard memory ceiling on this machine, not a code
  problem.
- **Draft acceptance is 46.7–55.6%**, against 76.2% reported in round 7, unexplained.
- **The gemv's 12% is unrealized** until streaming stops binding.
- **`EmbedLookup` is still unwired** — parity-tested, ~0.1 ms of a ~1,300 ms pass.
