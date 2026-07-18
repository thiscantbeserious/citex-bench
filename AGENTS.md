# AGENTS.md

Write efficient, short sentences. Be token efficient, not verbose. Apply this to everything you write here and in the repo.

CLAUDE.md is a symlink to this file. Other harnesses (Codex, pi.dev, opencode) read AGENTS.md directly.

## What this is

citex-bench: CPU-floor benchmark for LLM claim extraction with citation mapping, aimed at hallucinated citations in LLM-produced research. Measures Stage 1 (extraction) today, against a full concept in [ARCHITECTURE.md](ARCHITECTURE.md). Read it and [README.md](README.md) before touching the pipeline.

## The rule this repo stands on

Same-document support is real but always lower-weight than external verification. Top trust needs external sources, verified by the harness, not embedded here. Same principle for your work: the agent that produced a change does not judge it done. Verify independently.

## Work loop (before marking done)

Dispatch these, then address blocking findings before commit.

- work-verifier: adversarial diff review. Hunts fabricated claims, bugs, gaps. Read-only.
- arch-checker: conformance to ARCHITECTURE.md. Flags contract breaks. Read-only.
- run-verifier (runtime changes only): execute the changed code. Do not trust the diff's claims about behavior. Independent run is the check.

Mirrors the system: extract, verify against external, score.

## Prose rules

No em-dashes. No semicolons in prose. No `--`. Use commas and periods. Shell and C semicolons are fine.

## Commits

Commit only when asked. Configured author identity. Short imperative messages. Don't add attribution trailers. Push only when asked.

## Doubt

If a change would make extraction emit something ARCHITECTURE.md assigns to a later stage (resolved source, trust score, verification), stop and raise it. The benchmark measures one stage, output must feed the rest.
