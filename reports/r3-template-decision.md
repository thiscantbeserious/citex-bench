# R3 Template-vs-Raw Decision

Status: decided. Smoke data recorded below. The coordinator finalized this
record after the smoke numbers were in.

## Problem

`run_model` feeds a raw `-p` prompt to Qwen3 models with no chat template,
every tier. The 8B reasoning preamble is the untemplated behavior, not a quirk.
The original plan assumed one global path for every tier. The smoke proved
this assumption false.

## Fork Capability Findings

Re-verified on `bonsai-floor:prism` via:

```
docker run --rm --platform linux/arm64 --entrypoint llama-completion \
  bonsai-floor:prism --help 2>&1 | grep -Ei 'jinja|chat-template|json-schema|grammar|-cnv|no-cnv|special'
```

Relevant `--help` lines (full output in `/tmp/step3-fork-help.txt`):

```
--grammar GRAMMAR                       BNF-like grammar to constrain generations
--grammar-file FNAME                    file to read grammar from
-j,    --json-schema SCHEMA             JSON schema to constrain generations
-jf,   --json-schema-file FILE          File containing a JSON schema
-sp,   --special                        special tokens output enabled (default: false)
-cnv,  --conversation, -no-cnv, --no-conversation
                                        - does not print special tokens and suffix/prefix
--jinja, --no-jinja                     whether to use jinja template engine for chat (default: disabled)
                                        (env: LLAMA_ARG_JINJA)
--chat-template JINJA_TEMPLATE          set custom jinja chat template (default: template taken from model's
--chat-template-file JINJA_TEMPLATE_FILE
                                        set custom jinja chat template file (default: template taken from
```

All flags confirmed present: `--jinja`, `--chat-template`, `--chat-template-file`,
`-cnv`/`-no-cnv`, `--special`, `--json-schema`/`-j`, `--json-schema-file`/`-jf`,
`--grammar`, `--grammar-file`.

### Runtime confirmation (1.7B, 10 tokens)

Ran `--jinja` alone and `--jinja -cnv` on `Bonsai-1.7B-Q1_0.gguf`. Both produced
the same behavior: the log printed `chat template is available, enabling
conversation mode` and showed the ChatML template example
(`<|im_start|>system ... <|im_start|>user ... <|im_start|>assistant`). This
confirms `--jinja` alone auto-enables conversation mode when the model has a
chat template, treating `-p` as a user turn. `-cnv` is added for explicitness.

## Flags Chosen per Template Value

| `template` value | Flags added to `llama-completion` | Why |
| --- | --- | --- |
| `""` (raw) | none | Current behavior preserved. No new flags. |
| `"embedded"` | `--jinja -cnv` | `--jinja` enables the Jinja engine so the model's embedded ChatML template is applied. `-cnv` explicitly enables conversation mode so `-p` is treated as a user turn with `add_generation_prompt` (pre-seeds the closed reasoning block). |
| `<path>` | `--jinja --chat-template-file <path> -cnv` | `--jinja` enables the Jinja engine, `--chat-template-file` loads the custom template, `-cnv` enables conversation mode. |

## Smoke Procedure

1 case per tier, temp 0, direct mode. Four configurations tried: raw (no
`/no_think`), embedded, embedded with `/no_think`, and raw with `/no_think`.
Compare `valid_json` and `recall` across all tiers.

## Smoke Results

| Config | 1.7B | 4B | 8B |
| --- | --- | --- | --- |
| raw (no /no_think) | json=True recall 0% | json=True recall 100% | json=False recall 0% |
| embedded | json=True recall 100% | json=False recall 0% | json=True recall 100% |
| embedded + /no_think | json=True recall 100% | json=False recall 0% | json=True recall 100% |
| raw + /no_think | json=True recall 100% | json=True recall 100% | json=False recall 0% |

Key findings:

- 1.7B: fails on raw alone (0% recall), works on embedded, embedded+/no_think,
  and raw+/no_think (all 100%). Needs embedded.
- 4B: works on raw and raw+/no_think (100%), fails on embedded and
  embedded+/no_think (json=False, 0% recall). The embedded template corrupts 4B
  JSON output. Needs raw.
- 8B: fails on raw and raw+/no_think (0% recall), works on embedded and
  embedded+/no_think (100%). Needs embedded.
- `/no_think` did not change either failure mode (embedded still breaks 4B, raw
  still breaks 8B). It stays implemented but is not enabled by default.

## Decision Rule

- Template-on does not regress 1.7B/4B and unblocks 8B: set
  `defaults.template: "embedded"` in `eval/config.json`.
- Otherwise: implement the fallback (few-shot in the prompt, or `/no_think` in
  the prompt), rerun the smoke, record that.
- Both fail: record the finding and stop. Do not run the grid with an
  unresolved 8B prompt. Surface to the user.

## DECISION

Per-tier template. No single template path works for all three tiers.

- `defaults.template` = `"embedded"`. This is the majority path, working for
  1.7B and 8B.
- 4B is the sole exception. Each of its 3 param sets overrides `template` to
  `""` (raw) because the embedded template corrupts 4B JSON output (recall
  drops to 0%). The loader already supports per-set template overrides
  (`template` is in `_SAMPLABLE_KEYS`).
- The `/no_think` flag stays implemented but is not enabled by default. It did
  not change outcomes in any smoke configuration.
- This contradicts the plan's original "one path for every tier identically"
  rule. The plan doc (`docs/eval-redesign-plan.md` Step 3) is updated to match.
