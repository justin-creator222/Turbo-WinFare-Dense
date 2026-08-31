# Historical documents

These are a record of how the project was built, kept because the reasoning in them is
sometimes still the best explanation of why something is the way it is. **They are not current
documentation and some of them are wrong.**

For the engine as it exists now, read [`../../README.md`](../../README.md),
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) and [`../PERFORMANCE.md`](../PERFORMANCE.md).

| | |
|---|---|
| `spec.md` | The original specification. Its throughput targets (2.5–18 TPS) were set before anything ran and none were reachable; see PERFORMANCE.md §5 for the arithmetic and the measured figures that replaced them. |
| `IMPLEMENTATION_PLAN.md` | The plan the first build followed. |
| `VALIDATION_REPORT.md` | That build validated and **rejected** — 0 of 5 gates passed. |
| `REMEDIATION_PLAN.md` | The plan that followed the rejection. |
| `VALIDATION_REPORT_R2.md` | The re-validation after remediation. |
| `GEMINI_KICKOFF.md`, `GEMINI_REMEDIATION_KICKOFF.md` | Prompts used to drive a coding agent through those two rounds. Of historical interest only. |

Absolute paths in these files have been replaced with `<repo root>` and `<sibling engine>`.
The latter refers to a separate, private engine for a mixture-of-experts model that this
project borrowed structure from; it is not part of this repository and is not required to
build or run anything here.

## What happened after

The numbered round reports in `../` continue the story and supersede everything in this
directory:

| | |
|---|---|
| `ROUND3_REPORT.md` – `ROUND5_REPORT.md` | Correcting the forward pass against the upstream modelling source. Round 5 found seven bugs that GPU-vs-CPU parity could not catch, because both implementations shared the same misconceptions. |
| `ROUND6_REPORT.md` | Performance: 11,104 → 1,456 ms per pass. |
| `ROUND7_REPORT.md` | Batched prefill, FP16 KV cache, a second architecture (E2B), speculative decoding. |
| `ROUND8_REPORT.md` | Residency scheduling, flash attention, and three optimizations that measured neutral. |
| `ROUND9_REPORT.md` | Why they measured neutral: the engine was disk-bound, not GPU-bound. |
