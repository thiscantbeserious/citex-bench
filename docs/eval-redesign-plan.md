# Eval redesign plan

A build spec for the multi-pass accuracy eval. Execute through the repo's own
work loop (see AGENTS.md): each step is scoped, implemented by the
`implementer` subagent, then verified by `work-verifier` and `arch-checker`
(both always run), and `run-verifier` when executable code changed. The implementer
picks the code. This doc fixes the scope, invariants, decisions, and the
facts that must be proven, not the line-by-line how. Over-specifying the how
makes an implementer overconfident and stops it reading the real code.

Build prerequisites: the `bonsai-floor:prism` image and the 1-bit GGUFs in
`models/`. If absent for a step, skip that step's smoke, say so in the commit
body, do not invent output.

## Invariants (do not break)

1. ARCHITECTURE.md stage contracts. Stage 1 (extraction) emits
   `{claim, quote, marker}` only. The model never emits a resolved source,
   URL, or trust score. The resolver (Stage 2) is deterministic, never a model.
2. `score_case` in `eval/eval.py` is a pure function of
   `(raw_output, expected, mode, doc)`. `eval/verify.py` depends on it. Any
   new metric is computed from those inputs only.
3. `direct` mode (model emits `{claim, source_ref}`) breaks the Stage 1
   contract on purpose, to measure the cost of letting the model localize.
   Labeled "contract-deviating baseline" wherever it appears, never reported
   as compliant Stage 1. `quote` is the contract-conforming architecture.
4. The scout baseline ([reports/00-results-2026-07-18.md](../reports/00-results-2026-07-18.md))
   is frozen. Do not edit, do not reuse its numbers as comparable.

## Engineering principles

- KISS. Shortest working diff. No abstraction before a second caller exists.
- YAGNI. No config knob, schema field, fallback branch, or report column that
  nothing reads. Cut anything a later step does not consume.
- TDD. Test first, prove it red against current code, implement to green,
  clean up. The test stays as the regression guard. Pick the right test for
  the logic, do not follow a fixed recipe.
- Skeptic. Re-verify fork flag behavior with `--help` or source when a step
  depends on it. Missing capability is recorded as a finding with the
  documented fallback, never a silent skip.
- Honest output. Paste smoke numbers. If a smoke did not run, say so.

## Verified fork capabilities (re-verify when a step depends on one)

Image `bonsai-floor:prism`, PrismML `prism` branch, CPU only.

- `llama-completion` supports `--jinja`, `--chat-template*`, `-cnv`/`-no-cnv`,
  `--special`. Re-verify with `--help | grep -E 'jinja|chat-template'`.
- `llama-completion` supports `--json-schema`/`-j`, `--json-schema-file`/`-jf`,
  `--grammar`, `--grammar-file`. Re-verify with `--help | grep -E
  'json-schema|grammar'`. Flat schemas, no `$ref`, use `--json-schema`.
- Default sampler chain: `PENALTIES` then `TOP_K` `TOP_P` `TEMPERATURE` then
  `dist` (`common/common.h`). At temp <= 0 the temp sampler does `ggml_argmax`
  over post-penalty logits (`llama-sampler.cpp`). Penalties are NOT inert at
  greedy. Only top-k and top-p are.
- `--repeat-penalty` -> `penalty_repeat`, `--presence-penalty` ->
  `penalty_present` (`common/arg.cpp`). `penalty_last_n` defaults to 64
  (`common/common.h`), never overridden by eval.py, so penalties scan the last
  64 context tokens.
- All three 1-bit GGUFs are Qwen3, ChatML template, with the reasoning block.
  The embedded template pre-seeds a closed reasoning block at
  `add_generation_prompt`. Whether generation still emits a preamble is
  sampler-dependent, so R3 confirms by smoke, not by reading the template.

## Commit convention

One commit per step, message `<step-id> <imperative>`. No trailers. Push when
the user asks.

## Step 1, R1, scorer precision and one-to-one matching

Scope: `eval/eval.py::score_case`, `eval/verify.py::self_test`.

The scorer measures recall and citation accuracy but never penalizes fabricated
claims. For a hallucination-catching pipeline this is the primary failure
class and it is unmeasured.

After this step: greedy one-to-one assignment between expected and predicted
claims. Each expected and each prediction used at most once. Similarity keeps
the existing rule (`key_phrase in claim.lower()` = 1.0, else
`SequenceMatcher` ratio). Pairs below 0.5 never assign. Metrics:
`recall = assigned / len(expected)`, `precision = assigned / len(predictions)`
(0 if none), `citation_acc = correct_refs_among_assigned / assigned`.
Precision is primary (fabrication is the failure). F1 is a labeled
non-decision-grade convenience column, no conclusion from it alone.

Normative self-test fixtures (the implementer builds tests around these):
- Hallucination: doc `"Per a report [1], X happened. Per a study [2], Y
  happened."`, 2 expected, 5 predictions with 2 correct. New scorer gives
  precision 0.4, recall 1.0.
- Wrong-ref decoy: 2 expected with distinct keys, 1 prediction whose claim
  contains both keys with one ref. New scorer gives recall 0.5, precision 1.0
  (one-to-one cannot match both expected to one prediction).

Must prove: the new self-test is RED against the current scorer (capture the
failure in the commit body) and GREEN after the change. `score_case` stays
pure.

Done: verifiers clean, `verify.py selftest` green, old-failure captured in the
commit body, a 1-case smoke on `1.7b-1bit` direct temp 0 shows precision (or
stated absent if models missing).

## Step 2, R8 plus Config, the grid is a file

Scope: new `eval/config.json`, a loader in `eval/eval.py`, removal of
`tier_spec()` from `bench.sh`.

The grid is `eval/config.json`. Three blocks: `run` (how the batch runs),
`defaults` (generation defaults every model inherits), `models` mapping
`repo:quant` to a list of param sets. eval.py iterates models, sets, modes,
reps. No enumerated product in code.

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

Loader is normative, the implementer writes the code:

- `defaults` must define every samplable key (`repeat_penalty`, `top_k`,
  `top_p`, `presence_penalty`, `strict_schema_validation`, `template`). A set
  overrides only keys in `defaults` plus `temp`. Unknown key in a set raises.
  Missing default raises. Nothing falls through to a CLI default.
- `seed` is loader-owned. temp > 0 gets seed = rep index (0..N-1) so the N
  reps are independent rolls. temp == 0 is inert, fixed at 0. Re-run
  reproduces the seeds. Seed is written into every capture.
- `model` key is HF repo with `:quant` suffix. The loader resolves it to a
  file the same way bench.sh does (list repo GGUFs, pick the matching quant).
- Set identity is a readable slug
  `t{temp}-tk{top_k}-tp{top_p}-rp{repeat_penalty}-pp{presence_penalty}-s{0|1}`,
  `s`=1 when strict. Unique per set within a model, the loader raises on a
  duplicate.
- bench.sh reads `eval/config.json` for the model list and applies its packing
  logic (g128 vs g64, PQ2_0 exclusion) in its existing resolve loop. The
  `tier_spec()` function is removed.

Must prove: the loader raises on unknown key, missing default, and duplicate
slug (test cases the implementer chooses). bench.sh still resolves the same
GGUFs.

Done: verifiers clean, `verify.py selftest` green on loader behavior, bench
resolve path unchanged. Run budget is 100 runs of 5 cases (1.7B 4 sets, 4B 3,
8B 3, at 2 modes 5 reps). 8B stays at 3 sets. If 8B time is tight later, drop
its `strict_schema_validation: false` set first, never drop 4B.

## Step 3, R3, template-vs-raw, global, smoke-gated

Scope: `eval/eval.py::run_model`, `defaults.template` in config, a decision
record under `reports/`.

`run_model` feeds a raw `-p` prompt to thinking-capable Qwen3 models with no
chat template, every tier. The 8B reasoning preamble is the untemplated
behavior, not a quirk. The decision is global, one path for every tier.

Primary path: chat template via `--jinja` with the model's embedded template,
all tiers. Fallback (only if the primary regresses 1.7B/4B or fails to
unblock 8B): few-shot in the prompt, or `/no_think` in the prompt. Whichever
path applies to every tier identically.

`defaults.template` values: `""` raw, `"embedded"` use `--jinja` with the
model template, or a path to a jinja file. The smoke decides which.

Smoke: 1 case per tier, temp 0, direct mode, template-on vs template-off.
Compare valid_json and recall for 1.7B and 4B, and whether 8B produces
nonempty JSON with template on.

Decision rule:
- Template-on does not regress 1.7B/4B and unblocks 8B: set
  `defaults.template: "embedded"`.
- Otherwise: implement the fallback, rerun the smoke, record that.
- Both fail: record the finding and stop. Do not run the grid with an
  unresolved 8B prompt. Surface to the user.

Done: verifiers clean, `reports/r3-template-decision.md` holds the smoke lines
and the explicit decision, `defaults.template` in config matches it. The 8B
raw opt-out sets keep `strict_schema_validation: false` as a
template-independent failure data point.

## Step 4, R4, schema-validated JSON, default-on

Scope: `eval/eval.py::run_model`, the two flat schemas, `strict_schema_validation`
wiring.

`strict_schema_validation` defaults true. When true, pass `--json-schema` with
the mode's flat schema: direct is an array of `{claim, source_ref}`, quote is
an array of `{claim, quote}`, all string properties, no `$ref`. When false, no
constraint, post-hoc extract as today.

Schema constrains JSON syntax, not content. A fabricated claim passes as a
valid string. Schema kills the validity failure mode, not the hallucination
one, so precision stays primary.

Done: verifiers clean, a 1-case smoke on `1.7b-1bit` quote temp 0.5 with
strict true shows valid_json true, and the same case with strict false shows
the raw behavior, both recorded in the commit body. Fork capability confirmed
(`--json-schema` present), or a recorded finding if absent.

## Step 5, R5 plus R7, captures with seed and pinned threads

Scope: `eval/eval.py` capture writing, seed and threads in the run loop,
`reports/captures/`.

One JSONL file per (model, mode, slug):
`reports/captures/<model>-<mode>-<slug>.jsonl` where `<model>` is the
`repo:quant` with `/` and `:` replaced by `-`. One record per (case, rep)
carrying: model, mode, slug, config (with the auto-assigned seed, pinned
threads, strict_schema, template), case_id, rep, raw_output, expected, doc,
score, seconds, and a `scored` flag (true when the score came from the scorer,
false when synthesized for a timeout).

`expected` and `doc` stored per record so captures are self-contained and
replay does not re-resolve cases.jsonl. Seed and threads from Step 2's rules.
On `subprocess.TimeoutExpired`, write a record with `scored: false` and
`score: {"valid_json": false, "timed_out": true, ...}`. Threads pinned from
`run.threads`, never `os.cpu_count()`.

Done: verifiers clean, `verify.py selftest` green on the capture round-trip, a
2-rep smoke on `1.7b-1bit` direct temp 0.5 writes 2 records per case to one
file, one record shown in the commit body.

## Step 6, R6, deterministic replay

Scope: `eval/verify.py`, a `--replay <dir>` mode.

Replay loads every capture record, re-runs `extract_json_array` and
`score_case` on the stored `raw_output` and `doc`, and asserts the recomputed
score equals the stored score. On mismatch it names the
`(model, mode, slug, case_id, rep)` cell and exits nonzero. Records with
`scored: false` skip the equality check, asserting only `timed_out` present.
Replay runs at the end of every grid run.

Must prove: a capture with a mutated score makes replay exit nonzero and name
the cell (the implementer's test), before the implementation lands.

Done: verifiers clean, `verify.py selftest` green, replay on the Step 5 smoke
captures passes, replay output shown in the commit body.

## Step 7, reporting in eval/summarize.py

Scope: new `eval/summarize.py`. eval.py runs the grid and writes captures and
no longer prints an accuracy summary. `summarize.py` reads captures and emits
the report (mirrors the speed bench's `summarize.py`, decoupling runner and
reporter over the capture format).

`summarize.py <captures-dir>` computes: two-level variance (per (case, slug)
mean and std over reps first, then across cases, reporting within-case
stability separately from cross-case difficulty), per-case latency (seconds per
case_id, answering the scout's one-slow-case puzzle), precision and recall
primary with F1 secondary labeled non-decision-grade, and a determinism side
section reporting whether greedy (temp 0) reps reproduce identically
(identical recorded config incl threads with divergence is a finding).

Done: verifiers clean, `verify.py selftest` green on a small fixtures capture
dir, `summarize.py` on the Step 5 smoke produces readable output, a snippet
in the commit body.

## Step 8, R2 review

Scope: read-only review of `eval/config.json`.

Confirm greedy (temp 0) is one set per model, with penalties stated explicitly
(the 1.7B greedy set carries `presence_penalty: 0.5`, 4B and 8B inherit
`repeat_penalty: 1.0`, `presence_penalty: 0.0`), and greedy is treated as a
determinism check (Step 7 side section), not an accuracy cell. Fix the config
if it does not match. No new code expected.

Done: config matches, one line in the commit body.

## Step 9, run the grid once

Run the grid end to end, then `verify.py --replay reports/captures`, then
`summarize.py reports/captures`. This is the only grid run. If a cell fails
unexpectedly, record it in the report, do not edit config mid-run. The summary
states it is a new baseline, not comparable to the scout.

Done: captures populated, replay passes, summary in `reports/`, committed.

## After the grid (unchanged from the original)

1. Footer parsing. Deterministic Stage 2. Detect the references section, parse
   it, map each `[N]` to `{marker, title, url, ...}`. Does not change what a
   case is. Next.
2. Chunking. Token-budgeted, marker context crosses boundaries, a straddling
   claim stays in one chunk. Changes what a case is, lands after the baseline.

Footer first, chunking second.

## Known limitations

- Cold model load per subprocess. Every run is a cold load. 8B latency is
  unclear whether it includes load. Amortizing is out of scope, flagged future.
- Case-set generalization. N=5 reps on 5 hand-written cases. Variance describes
  these 5 cases, not the model. More cases out of scope.
- Curated param sets, not enumerated. The 1.7B `temp 0.7` set changes three
  knobs versus `temp 0.5`, so differences are not attributable to one knob.

## Out of scope (unchanged)

- 27B in the accuracy grid (latency).
- Ternary tiers beyond the existing 1.7b and 4b speed rows.
- Real (non-synthetic) test cases.
- GPU and Vulkan backends. This is the CPU floor.
