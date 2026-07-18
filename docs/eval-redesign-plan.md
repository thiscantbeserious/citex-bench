# Eval redesign plan

**Status:** Planned. Not implemented yet. Current (single-pass) eval committed and produces the [scout results](../reports/00-results-2026-07-18.md). This document is the agreed shape of the rebuild before anyone touches the working harness.

## Why redesign

The scout run answered the direction questions but has three load-bearing weaknesses:

1. **No variance.** Every cell is one sample. A single slow/odd case dominates (e.g. 1.7b-direct t=0 avg 14.5s vs 2.7s at t=0.5, likely one case, unmeasured). You cannot tell signal from noise.
2. **Collapsed failures.** The 8B tier shows as "0% valid JSON" with no way to see *why*. The root cause (8B emitting a reasoning preamble into stdout, never the array) had to be diagnosed by hand. Captures should make this automatic.
3. **Partial param sweep.** Only temperature was swept. top-k/top-p/repeat-penalty/presence-penalty held at card defaults. The user wants the full Cartesian across all of them.

## Goals

- Every `(tier x mode x params)` cell runs **N passes**. report mean +/- std (and for ratio metrics, the full distribution, since they're bounded [0,1] and skewed).
- Sweep the full Cartesian of sampling parameters from the model card's suggested ranges.
- Capture raw model I/O per run to a file, so any surprising number can be re-diagnosed deterministically, "not through judgement only."
- Deterministic re-verification path: feed captured raw output back through the scorer and assert it reproduces the logged number.

## Grid (agreed)

- **Tiers:** `1.7b-1bit`, `4b-1bit`, `8b-1bit` (the 1-bit family. ternary is a separate matrix). 27B excluded from the accuracy grid (latency-prohibitive per-case. keep in speed bench only).
- **Modes:** `direct` and `quote` both, in the same run, for the architecture comparison.
- **Params, full Cartesian** over the model card's suggested ranges:
 - `temperature` in {0.0, 0.5, 0.7} (0 = greedy control. 0.5 default. 0.7 top of range)
 - `top_k` in {20, 40}
 - `top_p` in {0.85, 0.95}
 - `repeat_penalty` = 1.0 (fixed, card has no range)
 - `presence_penalty` = 0.5 for 1.7b tiers (card flags it), 0.0 otherwise (fixed)
 - `seed` = varied per pass for temp>0 (to sample the distribution), fixed at 0 for temp=0 (greedy is deterministic)
 - **3 x 2 x 2 = 12 param configs per (tier x mode)**.
- **Passes (reps):** N=5 per cell. 12 configs x 2 modes x 5 reps = **120 runs per tier**. x 3 tiers = **360 runs**.

## Run-cost estimate (rough, single-pass scout latencies x cells)

- 1.7b ~5s/case x 5 cases x 120 ~ 50 min
- 4b ~9s/case x 5 cases x 120 ~ 90 min
- 8b ~36s/case x 5 cases x 120 ~ 6 hr *(8b is the dominant cost. consider reducing its reps or params once prompt fix lands)*
- **~8 hours total on this VM.** Reducible by parallelizing tiers across containers or trimming 8b.

## Concrete changes to `eval/eval.py`

1. **`--reps N`** arg. Loop the per-case run N times at each (params) config. collect per-rep raw output + score.
2. **`--params`** drives the Cartesian product instead of `--temps`. Parse a small spec or just iterate the hardcoded card ranges above (start hardcoded, config-file later if needed).
3. **Capture:** write one JSONL record per rep to `reports/captures/<tier>-<mode>-<config>-rep<k>.json`:
 `{tier, mode, config{temp,top_k,top_p,rep_penalty,presence,seed}, case_id, raw_output, expected, score, seconds}`.
 This is the "reverify deterministically, not through judgement only" path: raw I/O is inspectable per cell.
4. **Stats:** instead of one number per cell, emit `mean, std, min, max, n` for ratio metrics and latency.
5. **Seed policy:** for `temp>0`, vary `seed` per rep (e.g. `seed = base + rep`). For `temp=0`, fixed `seed=0` (greedy deterministic, reps should produce identical output, so N reps there is a *consistency check*, verifying determinism, not variance). Flag if greedy reps diverge (would indicate a bug).
6. **Summary table:** rows = `(tier, mode, config)`, columns = `valid_json_mean+/-std, recall_mean+/-std, cite_acc_mean+/-std, quote_hit_mean+/-std, secs_mean+/-std`.

## Deterministic re-verification (`eval/verify.py`, exists, expand)

- Self-test (passes): scorer is a pure function, same raw -> same score.
- Add a `--replay <captures/>` mode that reloads every capture, re-runs `extract_json_array` + `score_case` against the captured `raw_output`, and asserts the recomputed score equals the stored `score`. Any mismatch = a bug or non-determinism in the scorer. the cell is flagged.
- This is the check that answers "did the logged number actually come from this raw output" without trusting me.

## 8B prompt fix (prerequisite for 8B in the grid)

8B `llama-completion` emits a reasoning preamble before answering (observed: "Okay, the user wants me to reply with a JSON array..."), so `extract_json_array` returns `None`. Options, in order of preference:
1. Add 2-3 few-shot examples to the prompt showing immediate JSON output, likely suppresses the preamble cheapest.
2. Pass `--jinja` with the model's chat template so the "thinking" goes into a proper channel not stdout.
3. Stop-string / regex to trim preamble before extraction (fragile, only if 1-2 fail).
This must land before 8B enters the multi-pass grid, or its cells are meaningless.

## Out of scope for this plan (deliberately)

- 27B in the accuracy grid (latency).
- Ternary tiers beyond the existing 1.7b/4b speed rows.
- Real (non-synthetic) test cases, synthesize more first, or source a labeled set. tracked separately.
- GPU/Vulkan backends, this is the CPU floor.

## Next steps, after the multi-pass refactoring

These are the two gaps called out in the README's Background that the
single-pass scout does not address. Both should land after the multi-pass
harness is stable, because each changes what a "case" is and would invalidate
the single-pass numbers being held as the baseline.

1. **Chunking.** A document that exceeds the model's context window must be
 split into context chunks before extraction. Design questions to settle:
 chunk size (token-budgeted, not char), overlap or not, and how a claim that
 cites a marker first introduced in an earlier chunk is resolved when the
 chunk fed to the model no longer contains that marker. Likely needs the
 resolver (or a pre-pass) to carry marker context across chunks rather than
 per-chunk.
2. **Footer reference parsing.** Real documents carry their sources in a
 references section at the end (`[1] Title. URL.`, `[2] ...`). Today the
 harness treats markers as opaque labels. Resolving a marker to its actual
 source means detecting the references section, parsing it, and mapping each
 `[N]` marker to a `{title, url, ...}` entry. This is a deterministic pre/post
 step, not a model step.

Order them: footer parsing first (it is the smaller, fully deterministic
addition and unblocks real-world cases), chunking second (it interacts with
marker context and the resolver, and benefits from the multi-pass variance
numbers to measure chunk-boundary regressions against).
