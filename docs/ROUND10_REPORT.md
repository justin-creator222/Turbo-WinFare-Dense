# Round 10 — speculation was costing more than it saved

**Author:** Claude Opus 5 · **Date:** 2026-09-01 · **Implemented by:** me

## Summary

Speculative decoding was on by default at `draft_k` 8. Measured across 10 prompts, that default
was **1.474× slower than turning speculation off**.

| policy, 10 prompts × 48 tokens | total | vs off |
|---|---:|---:|
| **`--no-spec`** | **347.77 s** | **1.000×** |
| K=4 (best fixed K) | 437.99 s | 1.259× slower |
| K=6 | 446.85 s | 1.285× slower |
| K=8 (the shipped default) | 512.77 s | 1.474× slower |
| K=2 | 519.83 s | 1.495× slower |
| oracle per-prompt K | 404.72 s | 1.164× slower |
| oracle gate *and* K | 325.25 s | 0.935× |

Speculation is now **off by default**, `--spec` opts in, K is **6**, and an adaptive gate stops
drafting when acceptance does not justify it.

---

## The finding

Enabling speculation does two things, and round 9 measured only one of them. It adds draft
compute, and it drops the target from 45 resident layers to 39, because the drafter's 1.5 GiB
import reserve comes out of the same budget. Round 9 compared K=4 against K=8 and never ran
`--no-spec` on the same prompt, so the second cost was invisible.

Isolating it with `G4DENSE_MAX_RESIDENT_LAYERS=39` and no drafting:

| prompt | nospec @45 | nospec @39 | tax |
|---|---:|---:|---:|
| primes | 34.54 s | 42.44 s | 1.23× |
| counting | 35.17 s | 43.25 s | 1.23× |
| JSON | 36.31 s | 43.98 s | 1.21× |
| bicycle | 34.59 s | 42.59 s | 1.23× |

**A flat ~23%, prompt-independent, paid before a single token is drafted.** 21 layers stream per
token instead of 15. Drafting has to clear that before it is worth anything, and it only does so
above roughly 70% acceptance — which rote sequences and rigid formats reach, and ordinary prose
does not.

The full matrix and the length sweep are in [PERFORMANCE.md](PERFORMANCE.md) §5.

## Best K is prompt-dependent, and not monotonic

The JSON prompt accepts 60% at K=2, 71% at K=4, 76% at K=6 and 46% at K=8 — deterministically,
not as noise. K decides where verify-round boundaries fall in the text, so an unpredictable token
costs one wasted draft or seven depending on where it lands.

That is why round 9's reasoning — "the target pass costs ~1,330 ms against a ~65 ms draft, a 20:1
ratio, so drafting more per round is nearly free" — was wrong twice over. The ratio was a round-8
number, already superseded by round 9's own 883 ms pass when the paragraph was written; and
acceptance is not independent of K, which the arithmetic assumed.

K=6 wins on the prompts where opting in is rational (33.08 / 29.62 / 30.88 s on the three net
winners, against K=8's 23.01 / 31.51 / 40.56). K=8 wins only on near-deterministic sequences.

## What shipped

- **Off by default.** `--spec` opts in; `--no-spec` is retained as a no-op so existing scripts
  keep working. The CLI, `GenerationOptions` and `ServerConfig` all carry the same value —
  round 9's defect was that they did not.
- **`draft_k` 8 → 6**, clamped to [2, 8] everywhere.
- **An adaptive gate.** Below 45% acceptance over at least 24 drafted tokens, drafting stops for
  the rest of the generation and the loop runs one token per target pass. Thresholds override
  from `G4DENSE_SPEC_GATE_MIN_ACCEPT` and `G4DENSE_SPEC_GATE_WINDOW`. Measured: the bicycle
  prompt goes 52.87 s ungated at K=6 to **44.18 s** gated, against a 42.59 s floor.

  It runs inside the speculative loop via `forward_batch` with a batch of 1, rather than falling
  through to the non-speculative loop — that loop samples from `logits` before forwarding, while
  the speculative path carries an unforwarded `pending` token, so handing over would emit it
  twice.
- **The GUI stops offering controls that do nothing.** The speculation toggle and the K slider
  are disabled when no drafter is loaded, which is now the common case, and the telemetry panel
  distinguishes *gated off* from *idle*.

## Two bugs the campaign found

- **`G4DENSE_MAX_RESIDENT_LAYERS` under-imported by ~36%.** The planner treats the value as a
  count and spreads streamed layers evenly through the stack; the importer compared it against
  the layer *index* and refused everything numbered above it, so `=39` produced 25 resident
  layers. The two readings agreed only before rounds 6/7 changed streamed layers from
  tail-clumped to evenly spread, and this check was never updated. The natural path was
  unaffected (`max_resident` is `num_layers` there, so the test never fired), so published
  throughput numbers stand — but §2's "58 layers accepted at 8 GB UMA" is suspect if it came
  through the override, which would make the ~11.75 GiB ceiling a lower bound.
- **`--draft-k 1` generated exactly one token.** K is the verify-batch width and the loop asks
  for K−1 drafts, so K=1 asked for none, got an empty draft and broke out of the loop. The
  server clamped API input to a minimum of 1, so `{"draft_k": 1}` returned a one-token response
  to any client that sent it.

## Method

132 runs, two interleaved rounds per cell, medians compared; 10 prompts chosen to span
continuation entropy (rote sequence, rigid format, explanatory prose, technical reasoning, open
creative), 48 tokens greedy. Zero non-zero exits. Every run records a SHA of its generated text,
and **output is byte-identical across every configuration within each prompt** — speculation,
K, and the gate change speed only.

Acceptance rates rest on 42–147 drafts per prompt. Round 9's rested on 15–21, where "46.7%" is
7 of 15.

## What I got wrong

- **I called the K=8 result prompt-general after one prompt.** The first sweep said K=4 beat K=8
  and that speculation lost; a second prompt showed K=8 winning by 1.35×. Both were true of their
  prompt and neither was true in general. The 10-prompt matrix was the minimum that could have
  answered it, and I should have started there.
- **The first thing I built measured nothing.** The residency-control arm depended on
  `G4DENSE_MAX_RESIDENT_LAYERS`, which was broken. Smoke-testing the harness against a known
  value caught it; running the campaign first would have produced 100 clean-looking runs of a
  configuration I was not testing.

## Still open

- **The residency tax is the real target.** Removing it is what turns speculation from "loses on
  7 of 10" into "wins on ~4, neutral on 2" — intrinsically, against the 39-layer baseline, it
  already wins on primes (1.85×), counting (1.46×), JSON (1.42×) and the Python function (1.14×).
  `tools/probe_apu` reports AVX-512 with VNNI and BF16, and E2B is ~1.5 GiB of INT4, so a
  CPU-resident drafter would free the reserve entirely and restore all 45 target layers. That is
  a hypothesis, not a measurement.
- **Re-derive the import ceiling** now that the override works.
- **The speculative path ignores `sampling.repetition_penalty`**, which the non-speculative loop
  applies. Invisible at temp 0 with the default 1.0, and untouched by this round.
