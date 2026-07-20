# Eval redesign plan

**Status.** Amended and current. This is the single plan doc. It merges the
original plan, the reviewed refinements (R1 through R8), and the config review
(strict_schema default-on, greedy demoted to a determinism check, reporting
split out, KISS cuts). The single-pass scout already ran and produced
[reports/00-results-2026-07-18.md](../reports/00-results-2026-07-18.md), a
frozen, raw, untemplated baseline. Multi-pass numbers are a new baseline, not
comparable to the scout. The multi-pass grid runs once, after R1 through R7
land, never before.

## Scope and invariants (do not break)

- ARCHITECTURE.md stage contracts are inviolable. Stage 1 (extraction) emits
  `{claim, quote, marker}` only. The model never emits a resolved source, URL,
  or trust score. The resolver (Stage 2) is deterministic, never a model.
- `score_case` in `eval/eval.py` stays a pure function of
  `(raw_output, expected, mode, doc)`. That purity is what `eval/verify.py`
  depends on. Any new metric is computed from those inputs only.
- The `direct` mode prompt asks the model to emit `{claim, source_ref}`, which
  means the model does its own localization. That breaks the Stage 1 contract
  on purpose, to measure the cost of letting the model localize. `direct` is
  labeled as a contract-deviating baseline everywhere it appears and is never
  reported as compliant Stage 1 output. `quote` mode is the
  contract-conforming architecture.
- The scout baseline is frozen. Single pass, raw prompts, no chat template.
  The scout report is not edited and its numbers are not reused. Multi-pass
  numbers are a separate baseline and the report says so.

## Why redesign

The scout run answered the direction questions but has three load-bearing
weaknesses:

1. **No variance.** Every cell is one sample. A single slow or odd case
   dominates (e.g. 1.7b-direct t=0 avg 14.5s vs 2.7s at t=0.5, likely one
   case, unmeasured). You cannot tell signal from noise.
2. **Collapsed failures.** The 8B tier shows as 0% valid JSON with no way to
   see why. The root cause (8B emitting a reasoning preamble into stdout,
   never the array) had to be diagnosed by hand. Captures make this automatic.
3. **Partial param sweep.** Only temperature was swept. The card's other
   parameters (top-k, top-p, penalties) were never exercised.

## Fork capability check (verified, not assumed)

Checked against the built binary in the image `bonsai-floor:prism`, from the
PrismML fork (`prism` branch, CPU only). Every flag below appears in
`llama-completion --help`. No upstream-parity assumption was made.

- **Chat template (for R3):** `--jinja` / `--no-jinja`, `--chat-template`,
  `--chat-template-file`, `-cnv` / `-no-cnv`, `--special`, `-st`. Supported.
- **Schema constraint (for R4):** `--grammar`, `--grammar-file`,
  `-j` / `--json-schema`, `-jf` / `--json-schema-file`. Supported. The
  `--help` note says schemas with external `$ref` should use `--grammar` plus
  `example/json_schema_to_grammar.py` instead. Our schemas are flat, no
  `$ref`, so `-j` / `--json-schema` is the right tool.
- **Penalties CLI (for R2):** `--repeat-penalty` maps to `penalty_repeat`
  and `--presence-penalty` maps to `penalty_present` (verified in
  `common/arg.cpp`). `penalty_last_n` defaults to 64 (verified in
  `common/common.h`), and eval.py never overrides it, so penalties scan the
  last 64 context tokens, not the whole document. This bounds the penalty's
  reach but does not change R2's claim.
- **Sampler chain (for R2 and R7):** the default chain order (verified in
  `common/common.h`) is `PENALTIES` first, then `TOP_K`, `TOP_P`,
  `TEMPERATURE`, then `dist`. The temperature sampler at temp less than or
  equal to 0 does `ggml_argmax` over the logits it receives
  (`llama-sampler.cpp`), which have already passed through the penalties
  sampler. So at greedy, penalties modify logits before argmax. They are NOT
  inert. only top-k and top-p are inert at greedy because they filter the
  candidate set, which argmax ignores.
- **Model metadata:** all three 1-bit tiers (1.7B, 4B, 8B) are Qwen3
  architecture with a ChatML template (`<|im_start|>`, `<|im_end|>`) and the
  Qwen3 reasoning block. The embedded `tokenizer.chat_template` is identical
  across the three GGUFs. At `add_generation_prompt` it emits
  `<|im_start|>assistant\n` plus a closed reasoning block to steer generation
  toward the answer channel. Whether generation still emits thinking depends
  on sampler logits, so R3's thinking-off path is confirmed by the smoke test,
  not assumed.

## R1, precision plus one-to-one matching (blocks everything)

**Problem.** `score_case()` measures recall and citation accuracy but never
penalizes fabricated claims. A model that emits 20 claims, 3 correct and 17
invented, scores perfect recall. For a pipeline whose purpose is catching
hallucination (see ARCHITECTURE.md), the primary failure class is currently
unmeasured. The `no_citations` case covers only the degenerate instance
(`expected == []`).

**Change 1, precision.**

```
precision = matched_predictions / total_predictions   (0.0 if no predictions)
```

where `matched_predictions` counts the predictions assigned to an expected
claim (see Change 2). Precision is the primary anti-hallucination metric. A
fabricated claim is a precision failure. Recall is secondary: missing a claim
is less damaging than inventing one with a citation, per the Stage 1 hard
rule that uncited and fabricated sentences are the exact failure class to
discard.

**Change 2, one-to-one assignment.** Current matching is any-match. Each
expected claim scans all predictions (`key in claim.lower()` or
`SequenceMatcher` ratio above 0.5), so one prediction can satisfy several
expected claims, and an expected claim can match the wrong prediction and
inherit its ref. `citation_acc` then measures luck. Replace with greedy
one-to-one assignment:

1. Compute similarity for every (expected, prediction) pair. Keep the existing
   rule: `key_phrase in claim.lower()` scores 1.0, else
   `SequenceMatcher(None, key, claim.lower()).ratio()`.
2. Sort pairs by similarity descending. Assign greedily. Each expected and
   each prediction is used at most once. Pairs below 0.5 never assign.
3. `recall = assigned / len(expected)`,
   `precision = assigned / len(predictions)`,
   `citation_acc = correct_refs_among_assigned / assigned`.

**F1, secondary and labeled.** F1 is emitted as a secondary convenience sort
key, not as a decision metric. F1 equal-weights precision and recall, which
masks precision failures: a high-recall fabricator scores mid-F1 and looks
acceptable when it is not. Because this eval's primary failure is precision,
F1 is labeled in the report as non-decision-grade, and ranking is done on
precision. Anyone who needs F1 finds it in the output, but no conclusion is
drawn from it alone.

**Proof, not assertion.** Extend `eval/verify.py::self_test()` with two cases:

- A hallucination case. 2 expected, 5 predictions, 2 correct. Old scorer has
  no precision (it returns none) and gives recall 1.0. New scorer gives
  precision 0.4, recall 1.0.
- A wrong-ref decoy case where any-match would credit the wrong prediction
  and one-to-one must not.

The new self-test must FAIL against the old scorer and PASS against the new
one. This is demonstrated, not asserted, before R1 is marked done.

## R2, greedy is a determinism check, not an accuracy cell

At temp=0 the sampler is greedy. top-k and top-p are inert (they filter the
candidate set, which argmax ignores). Penalties are NOT inert: they modify
logits before argmax (verified in the sampler chain, see Fork capability
check). So a greedy set must state its penalties explicitly. The 1.7B greedy
set carries `presence_penalty: 0.5`, matching the card and the scout's
effective sampler. 4B and 8B keep `repeat_penalty: 1.0` and
`presence_penalty: 0.0`, so penalties are a no-op there.

Greedy on Q1_0 models is degenerate-prone: repetition loops, empty output,
and under `strict_schema_validation` a trivially-valid `[]` that satisfies the
constraint but carries no content. So greedy is treated as a determinism
check, not an accuracy cell:

- Greedy runs N reps (same as accuracy reps, to have something to compare).
- The check is: do the N reps reproduce identically? Yes is the expected
  deterministic outcome. No, with an identical recorded config including
  threads (see R7), flags the cell as a finding, not as variance to average.
- Greedy cells are reported in a side section, not in the accuracy comparison.
  A greedy `[]` output is a passed determinism check, not a scored result.

This separates "is the model deterministic" from "how good is greedy
extraction," which greedy-on-noisy-models cannot answer meaningfully.

## R3, decide template-vs-raw once, globally (replaces the 8B prompt fix)

**Problem.** `run_model()` feeds a raw `-p` prompt to instruct-tuned,
thinking-capable Qwen3-based models, with no chat template, for any tier. The
8B reasoning preamble is not a prompting quirk. It is what an untemplated
thinking model does. Fixing 8B alone makes its cells incomparable to 1.7B and
4B, which are also measured raw. The decision is global, applied to every
tier identically.

**Primary path.** Chat template with thinking disabled for all tiers. Use the
fork's `--jinja` with the model's embedded chat template (verified present in
all three GGUFs). The template's `add_generation_prompt` pre-seeds a closed
reasoning block. Confirm by the R3 smoke test that no generation emits a
reasoning preamble into stdout.

**Fallback path (only if the primary regresses or fails).** Few-shot examples
in the prompt string, or a `/no_think` token in the prompt. Applied to every
tier identically.

**Decision procedure, not a silent pick.** The plan implements the primary
(`--jinja`) path and runs the smoke test. The smoke result, not this doc,
decides. The smoke test is 1 case per tier, t=0, direct mode, template-on vs
template-off. 1 case is enough to see template regress 1.7B/4B or unblock 8B.
If it is ambiguous, dig deeper. If template-on does not regress 1.7B and 4B
and unblocks 8B, template is used everywhere and `defaults.template` gets the
path. If it regresses or fails to unblock 8B, switch to the fallback. Record
the decision and the smoke numbers in the reports. The template decision
lands before the grid, because it changes what every cell measures.

**8B interaction.** The 8B raw sets (see Config) carry
`strict_schema_validation: false` and show the 0-percent preamble failure
unconstrained. That is a template-independent data point on the 8B failure
mode. It informs but does not replace the smoke-test decision.

## R4, schema-validated JSON as the default, raw as the opt-out

**Problem.** The dominant observed failure is JSON validity (1.7b-1bit 40 to
60 percent valid, 8b 0 percent). The old sweep never touched generation
format.

**Change.** `strict_schema_validation` defaults to `true`. When true, eval.py
passes a JSON schema to `llama-completion` via `--json-schema` matching the
mode's exact output schema: `[{claim, source_ref}]` for direct,
`[{claim, quote}]` for quote. Schemas are flat, no `$ref`. A set that wants
unconstrained generation opts out with `strict_schema_validation: false`.

Schema validation constrains only JSON syntax, not content. The schema says
`{"claim": string}`, which the grammar treats as "any string." So it forces
the array and object shape but does NOT force the claim to be correct or the
quote to be verbatim. The model can still emit a fabricated claim as a valid
string. Schema validation kills the JSON-validity failure mode, not the
hallucination one. That is why precision (R1) is still the primary metric.

**Raw opt-out placement.** The opt-out sets sit where the JSON failure is
instructive: once on 1.7B (heavy failure, 40 to 60 percent valid), once on 8B
(the 0-percent preamble failure), and once on 4B as a control, does constraining
a model that already emits perfect JSON change its content?

**Rationale guard.** This does not contradict FullCite. FullCite showed
grammar-constrained span selection loses to post-hoc alignment. Constraining
JSON syntax while quote plus post-hoc still does all localization conflicts
with nothing. The model still never localizes.

## Config (the grid is a file)

The grid is `eval/config.json`. Three blocks: `run` (how the batch runs),
`defaults` (generation defaults every model inherits), and `models` keyed by
`repo:quant`, value a list of param sets. eval.py iterates
models, then sets, then modes, then reps. There is no enumerated product in
code. Each param set is the values it sets, everything else falls to
`defaults`.

```json
{
  "run": {
    "threads": 7,
    "reps": 5,
    "timeout": 600,
    "modes": ["direct", "quote"]
  },
  "defaults": {
    "repeat_penalty": 1.0,
    "top_k": 20,
    "top_p": 0.9,
    "presence_penalty": 0.0,
    "strict_schema_validation": true,
    "template": ""
  },
  "models": {
    "prism-ml/Bonsai-1.7B-gguf:Q1_0": [
      {"temp": 0.0, "presence_penalty": 0.5},
      {"temp": 0.5, "presence_penalty": 0.5},
      {"temp": 0.7, "top_p": 0.85, "top_k": 40, "presence_penalty": 0.5},
      {"temp": 0.5, "presence_penalty": 0.5, "strict_schema_validation": false}
    ],
    "prism-ml/Bonsai-4B-gguf:Q1_0": [
      {"temp": 0.0},
      {"temp": 0.5},
      {"temp": 0.5, "strict_schema_validation": false}
    ],
    "prism-ml/Bonsai-8B-gguf:Q1_0": [
      {"temp": 0.0},
      {"temp": 0.5},
      {"temp": 0.5, "strict_schema_validation": false}
    ]
  }
}
```

**Loader rules (normative, not implementation detail):**

- `defaults` MUST define every samplable key (`repeat_penalty`, `top_k`,
  `top_p`, `presence_penalty`, `strict_schema_validation`, `template`). A
  param set may only override keys that exist in `defaults` or set `temp`.
  The loader hard-fails on an unknown key in a set and on a missing default.
  No key ever falls through to an eval.py CLI default or an implicit value.
  This is the opposite of vibecoding: silent drift is blocked loudly.
- `seed` is auto-assigned by the loader, not configured. At temp greater than
  0, each of the 5 reps needs a different seed so they are independent rolls
  of the sampler. The loader assigns rep 0 to seed 0, rep 1 to seed 1, and so
  on. This is what makes the variance measurement real: if all reps used one
  seed they would be identical and variance would read zero. At temp equal to
  0 (greedy), seed is inert, so it is fixed at 0 for consistency. Re-running
  the grid reproduces the same seeds. The seed is written into every capture
  so it is auditable.
- `run.threads` is pinned once, written into every capture (R7). Not a
  sampling param.
- `run.reps` is the pass count. Not a sampling param.
- `run.timeout` is the per-invocation subprocess timeout, in the config so it
  cannot drift per invocation.
- `run.modes` is which architectures to test, a batch decision, not a per-
  model sampling default.
- `model` key is the HF repo, with `:quant` suffix. eval.py splits repo from
  quant, lists the repo's GGUFs, picks the file matching the quant.
- `template` is a path to a jinja template file, or empty for raw prompt. Per
  R3, the smoke test decides whether a path is set (in `defaults`, so it
  applies globally).

**Config id is a readable slug.** Each set is identified by a readable slug
derived from its param values, not an opaque hash. The slug format is
`t{temp}-tk{top_k}-tp{top_p}-rp{repeat_penalty}-pp{presence_penalty}-s{0|1}`
where the `s` flag is 1 when `strict_schema_validation` is true and 0
otherwise. Example: `t0.5-tk20-tp0.9-rp1.0-pp0.5-s1`. Capture filenames are
`reports/captures/<model>-<mode>-<slug>.jsonl` and summary rows are keyed by
the same slug. The slug must be unique per set within a model. The loader
hard-fails on a duplicate slug.

**Run count is the product of what is listed.** 1.7B has 4 sets, 4B has 3, 8B
has 3. At 2 modes and 5 reps that is 40 plus 30 plus 30 = 100 runs, each run
5 cases. 8B stays at 3 sets. If 8B time is tight, drop the 8B
`strict_schema_validation: false` raw set first (it duplicates the scout's
known 0-percent failure). Do not drop the 4B sets: 4B is the scout's best
performer and the leading floor candidate, its absence from an earlier config
draft was an error, not a scope choice.

**Models in scope:** the three 1-bit tiers (`prism-ml/Bonsai-1.7B-gguf:Q1_0`,
`prism-ml/Bonsai-4B-gguf:Q1_0`, `prism-ml/Bonsai-8B-gguf:Q1_0`), all present
above. 27B excluded from accuracy (latency-prohibitive per case, kept in the
speed bench only). Ternary tiers stay in speed only.

**Param sets are curated, not enumerated.** The card gives ranges (temp 0.5
to 0.7, top_k 20 to 40, top_p 0.85 to 0.95, 1.7b needs presence penalty). It
does not publish combos. The sets above are representative points picked from
those ranges. This honors the card's "tune together" intent, at the cost of
not measuring one parameter's marginal effect holding others fixed. Known
consequence: the 1.7B `temp 0.7` set changes three knobs versus the
`temp 0.5` set (temp, top_p, top_k), so a difference between them cannot be
attributed to any single knob. Accepted deliberately. If attribution ever
matters, add a set differing only in temp.

## R5, capture schema, one record per (case, rep), addressable

The original filename lost the case dimension. Each rep runs 5 cases, so
replay could not address a single case, which is the stated purpose of
captures.

**Change.** One JSONL file per (model, mode, slug):
`reports/captures/<model>-<mode>-<slug>.jsonl`. One record per (case, rep):

```json
{"model":"prism-ml/Bonsai-1.7B-gguf:Q1_0", "mode":"direct",
 "slug":"t0.5-tk20-tp0.9-rp1.0-pp0.5-s1",
 "config":{"temp":0.5,"top_k":20,"top_p":0.9,"repeat_penalty":1.0,
          "presence_penalty":0.5,"seed":3,"strict_schema_validation":true,
          "template":"","threads":7},
 "case_id":"energy", "rep":3, "raw_output":"...", "expected":[...],
 "doc":"...", "score":{...}, "seconds":0.0, "scored":true}
```

`config` includes everything that affects output. `strict_schema_validation`,
`template`, `threads`, and the auto-assigned `seed` are included, or replay
cannot explain divergence. `expected` and `doc` are stored in every record so
captures are self-contained and do not depend on cases.jsonl resolving the
same way at replay time. The trade-off is larger records and duplication of
case data, accepted for self-containment.

## R6, implement verify.py --replay

`verify.py` is an honest skeleton. Replay exits 2 today. Implement it. Load
every capture record. Re-run `extract_json_array` plus `score_case` on the
stored `raw_output` and `doc`. Assert the recomputed `score` equals the stored
`score`. Any mismatch names the `(model, mode, slug, case_id, rep)` cell and
exits nonzero. Run it at the end of every grid run, not manually.

**Timeout records.** A timed-out rep is stored with a synthesized score
(`valid_json=False, timed_out=True`) that was never produced by the scorer,
because the process was killed and there is no array. Every record carries a
`scored` flag. Replay skips the equality check on records where `scored` is
false, asserting only that `timed_out=True` reproduces. Records where `scored`
is true must pass the equality check.

## R7, pin and record threads, refine the greedy-divergence rule

t=0 reps diverging does not necessarily mean a bug. Thread-count-dependent
floating point accumulation in prompt processing can legitimately shift
logits.

**Change.** Pin `--threads` explicitly for the whole grid. One value, chosen
once (`run.threads` in Config, container `nproc`). Write it into every
capture (R5). Never let `os.cpu_count()` float between runs. Then: greedy
divergence with an identical recorded config, including threads, flags the
cell as a finding (see R2). Divergence across different thread counts is
expected, not a defect.

## Reporting, split into eval/summarize.py

The accuracy summary is no longer inline in eval.py. eval.py runs the grid and
writes captures. A separate `eval/summarize.py` reads captures and emits the
report. This mirrors the speed bench's `summarize.py` and keeps the runner and
the reporter decoupled, sharing only the capture format contract.

`summarize.py` computes, from captures:

- **Two-level variance.** Per (case, slug) stats over reps first (mean, std),
  then aggregated across cases. This separates "the model is unstable" (high
  within-case std) from "case C is hard" (low within-case std but low mean).
  Reporting both levels, not a single conflated mean and std.
- **Per-case latency.** A per-case latency table, seconds per case_id, from
  the `seconds` field in captures. This answers the scout's "14.5s vs 2.7s,
  one slow case" puzzle directly, which reps alone could not.
- **Precision, recall, F1-secondary.** Precision and recall are primary.
  F1 is emitted as a secondary, non-decision-grade convenience column, see
  R1.
- **Determinism check (side section).** Greedy sets (temp equal to 0) are not
  scored for accuracy. They are reported as a determinism check: do the N reps
  reproduce identically. Yes is expected. No, with an identical recorded
  config including threads, is a finding.

## R8, single-source the model list (config), packing stays in bench.sh

The repo+quant mapping lives in one place: `eval/config.json`'s `models`
keys. eval.py reads it for accuracy. bench.sh reads it for the speed bench,
then applies its own packing logic (g128 vs g64, PQ2_0 exclusion) locally.
The packing logic was never in `tier_spec()`, it was in bench.sh's resolve
loop, so sharing the model list does not couple packing into the accuracy
config. bench.sh's `tier_spec()` is removed. One source of truth for which
models and which quants. R8 lands alongside Config.

## Revised sequencing

1. R1, scorer precision plus one-to-one assignment, with the extended
   self-test that proves the old scorer fails and the new one passes.
2. R8 plus Config. Build `eval/config.json` (the sets above) and the loader
   in `eval.py`, including the loader rules (hard-fail on unknown keys and
   duplicate slugs, auto-assigned seed, readable slug). This is the grid's
   home and the model+param single source.
3. R3, template decision. Implement the `--jinja` primary, then smoke test
   template-on vs template-off per tier, 1 case each. The smoke result picks
   primary or fallback and sets the `template` path in `defaults`. Record the
   decision and numbers.
4. R4, schema wiring. JSON schemas for direct and quote, `--json-schema`
   flag wired in. `strict_schema_validation` is a default-on flag in Config.
   Fork capability already verified.
5. R5, R6, R7. Captures (R5), replay (R6, with the `scored` flag), thread
   pinning (R7, `run.threads` in Config). The capture schema carries
   `strict_schema_validation`, `template`, `threads`, and the auto-assigned
   seed.
6. Reporting. `eval/summarize.py` reading captures, two-level variance,
   per-case latency, F1-secondary, determinism side section.
7. R2 review. Confirm the sets in Config. Greedy is one set per model and a
   determinism check, penalties stated explicitly on greedy sets.
8. Run the multi-pass grid once.
9. Then, unchanged from the original plan: footer parsing, then chunking.

Footer parsing is a deterministic Stage 2 step and does not change what a case
is. Chunking does. The original ordering, footer first then chunking, stands.

## Open questions not resolved by this amendment

Surfaced so they are not silently dropped.

- **Cold model load per subprocess.** Each `run_model` spawns
  `llama-completion` cold. Every listed run is a cold load. The per-case
  latency figure for 8B is unclear on whether it includes load. Amortizing
  load via a persistent process per (model, set) changes the harness more than
  the config loader but may be where the real time is. Out of scope here.
  Flagged as future.
- **Case-set generalization.** N=5 reps on 5 hand-written cases is a weak
  base. The variance numbers describe these 5 cases, not the model. "More
  cases" is out of scope per the original plan. Recorded as a limitation.

## Out of scope (unchanged from the original)

- 27B in the accuracy grid (latency).
- Ternary tiers beyond the existing 1.7b and 4b speed rows.
- Real (non-synthetic) test cases. Synthesize more first, or source a labeled
  set. Tracked separately.
- GPU and Vulkan backends. This is the CPU floor.
