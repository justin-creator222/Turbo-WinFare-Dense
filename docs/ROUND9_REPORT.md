# Round 9 — the engine was disk-bound

**Author:** Claude Opus 5 · **Date:** 2026-08-30 · **Implemented by:** me

## Summary

| | round 8 | round 9 |
|---|---:|---:|
| stream I/O | 318–326 ms | **89–92 ms** |
| GPU queue | 746–752 ms | 630–633 ms |
| LM head | 32 ms | 24 ms |
| CPU other | 100 ms | **28–29 ms** |
| **throughput** | 0.750–0.760 tok/s | **1.129–1.136 tok/s** |

**+50%.** Three runs each, `--prompt "Hi" --max-tokens 14 --temp 0 --no-spec`, 45 of 60 layers
resident — the same command every round has used.

Speculative generation of 24 tokens: **30.8 s → 20.2 s**.

This round began as a cleanup of round 8's leftovers. The cleanup found that round 8 had been
optimising the wrong resource for its entire duration.

---

## The finding

The streamer opened the container with `FILE_FLAG_NO_BUFFERING`. Ground truth measures that
path at **3.08–3.09 GB/s**, against **7.14 GB/s** buffered-warm. At 15 streamed layers the
engine reads **4.145 GB per token**:

```
4.145 GB / 3.09 GB/s  =  1,341 ms
round-8 token time    =  1,328 ms
```

The read alone was the entire token, to within 1%. The GPU's 748 ms fitted inside it with room
to spare. **The engine was disk-bound, and had been since the streaming path was written.**

That one division explains every negative result of round 8:

- the gemv vectorisation made the kernel 12% faster and changed nothing;
- flash attention was neutral;
- spreading the streamed layers was the only thing that helped — because it was an
  I/O-scheduling fix, not a GPU one.

Round 8 measured all three honestly and drew the right local conclusion each time. What it never
asked was why two independent GPU improvements would *both* come to nothing. The saturated
resource was never checked.

### Why unbuffered had been chosen, and what was wrong with it

Round 6 recorded a real measurement: buffered overlapped reads *complete inline* when the data
is in the page cache, so `issue_reads()` blocked and nothing overlapped — 956 ms moved out of
the I/O phase into "CPU other" without being saved.

The observation was correct; the conclusion did not follow. Inline completion is only fatal
because the read was issued on the thread that submits GPU work. That is a threading problem,
and it was solved by surrendering more than half the read bandwidth.

`LayerStreamer` now owns four I/O workers. `issue_read()` enqueues and returns; `await_read()`
waits on a per-slot completion. The `OVERLAPPED` carries the file offset, so workers never share
a file pointer, and the destructor joins them before closing the handle or freeing slots.

`CPU other` at 100 ms → 29 ms is the blocking read leaving the main thread.

## Tuning

| workers | tok/s | I/O | | depth | resident | tok/s | I/O |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 0.893 | 212 ms | | 2 (3 slots) | 46 | 1.102 | 126 ms |
| 2 | 1.079 | 127–142 ms | | **3 (4 slots)** | **45** | **1.125** | **92 ms** |
| **4** | **1.130** | **85–90 ms** | | 4 (5 slots) | 44 | 1.045 | 135 ms |
| 6 | 1.139 | 84–87 ms | | | | | |
| 8 | 1.127 | 89–92 ms | | | | | |

Past 4 workers the gain is noise, structurally: `PREFETCH_DEPTH` is 3, so the queue never holds
more than three jobs. `PREFETCH_DEPTH` itself was already right — each slot costs ~276 MB of
import ceiling, i.e. a resident layer, and the trade is real in both directions.

## The gemv converted

Round 8 kept the vectorised weight loads against its own plan, which said to revert anything
neutral, on the argument that they would pay once streaming stopped binding. Re-measured:

| | GPU phase | throughput |
|---|---:|---:|
| 8 × `u32_load` | 667–672 ms | 1.104–1.110 tok/s |
| 2 × `Load4` | 630–631 ms | 1.126–1.130 tok/s |

**+1.8%** where it was worth nothing before. Small, because the GPU is now 631 ms of an 883 ms
token rather than all of it — but the reasoning for keeping it held up.

## `draft_k`, and an acceptance rate that can be read

Three entry points disagreed: the CLI defaulted to **4**, `GenerationOptions` to **6**, and the
server set neither — so every speculation measurement taken on the CLI described a configuration
the GUI did not run. Measured, 24 tokens greedy, three runs each:

| | elapsed | acceptance |
|---|---:|---:|
| K = 4 | 26.06–26.45 s | 46.7% |
| **K = 8** | **20.58–20.81 s** | 51.4% |

**21% faster.** My planning arithmetic had said K=4 was near-optimal; it assumed acceptance was
independent of K and priced the draft pass too high. The target pass costs ~1,330 ms against a
~65 ms draft, so drafting more per round is nearly free. K is capped by `kGemmMaxBatch` because
the verify batch is K wide, so 8 is also the ceiling. It costs 1–3% on prompts that stop early,
where the last round over-drafts.

**The acceptance "mystery" was not one.** 76.2%, 46.7% and 55.6% are 16/21, 7/15 and 5/9 —
every denominator a multiple of 3, so all three ran at `draft_k` 4, on ~20-draft samples. Greedy
is deterministic (46.6667% twice), so round 7's 76.2% was a single default-temperature sample.
Round 8 recorded it as an unexplained regression and left it open; it was neither.

Two reporting defects behind that, both fixed: the counters were cumulative for the life of the
process and never cleared, so in server mode the rate drifted across requests; and the
denominator was hidden, which is what let a 7-of-15 be read as a stable percentage.

## Documentation and hygiene

- **`docs/G4DENSE_FORMAT.md` was still v2.0** and documented the per-layer layout with no
  16-byte pad — the authoritative spec, and the one place not updated when the format went to
  v3. Now correct, with a section naming all five places the layout is written down, because
  changing one in isolation is exactly how the CPU oracle came to grade against the wrong bytes.
- **`README.md` added.** `tests/fixtures/tiny.g4dense` and `oracle_tensors/` are gitignored and
  go stale silently — a stale `oracle_tensors` had two ctest cases failing for a whole round
  before anyone noticed.
- **Rounds 7 and 8 are pushed and merged to `master`.** They had existed only on this machine.

## Verification

```
run_gpu_kernels_test     9 of 11 kernels; 5 of 5 attention cases
run_gpu_forward_test     31B and E2B: argmax and top-5 exact, 4 positions + batched prefill
run_real_generation_test 3 of 3 prompts coherent, text unchanged
ctest                    16 of 16
```

## What I got wrong

- **I did not check which resource was saturated** before optimising, for three rounds. The
  check is one division and it was available the whole time — `bench_gpu` and `ground_truth.json`
  had both halves of it.
- **My planning arithmetic for `draft_k` was wrong**, and measurement contradicted it. The plan
  said unify at 4; the answer was 8, which is 21% faster.
- **A doc change rode along in an unrelated commit** (`9825732` carries the `G4DENSE_FORMAT.md`
  rewrite without mentioning it).

## Still open

- **The GPU now binds**, at 631 ms of an 883 ms token. `GemvInt4` sustains 38–48 GB/s against
  65–74 GB/s for a pure copy, and closing that gap would now show up as throughput — which it
  would not have in rounds 6–8.
- **The ceiling is 1.46 tok/s** (GPU 631 + LM 24 + CPU 29) at this residency.
- **`EmbedLookup` is still unwired** — ~0.1 ms of an ~883 ms pass.
