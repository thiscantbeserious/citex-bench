# AGENTS.md

Write efficient, short sentences. Be token efficient, not verbose. Apply this to everything in this repo.

This file is the source of truth for any agent working here. `CLAUDE.md` symlinks to it. Do not restate README or ARCHITECTURE content, read those for context. This file is the operating procedure.

## Coordinator principle

The main session coordinates. It does not write code, prose, or run multi-step work itself. Delegate implementation to subagents. This keeps the coordinator's context for decisions, not for diffs. The coordinator's job ends after dispatching, relaying results, and committing when the user asks.

## Work units

A work unit is one discrete change: a feature, a fix, a doc section. Split work into units. One unit per subagent dispatch.

## Work loop (enforced, numbered)

Every work unit, in order, no skipping:

1. Scope. Coordinator names the unit, the files in scope, and the done condition in one or two sentences. No more.
2. Implement. Coordinator dispatches an implementer subagent with that scope. The implementer writes the change. It does not commit, declare done, or self-verify.
3. Verify-code. Coordinator dispatches `work-verifier` on the diff. Adversarial, read-only. Blocking findings go back to the implementer.
4. Verify-arch. Coordinator dispatches `arch-checker` on the diff. Conformance to ARCHITECTURE.md hard rules, read-only. Blocking findings go back to the implementer.
5. Verify-run. For runtime changes only, coordinator dispatches `run-verifier`. It executes the changed code. Real output is the check, not the diff's claims.
6. Resolve. Implementer fixes all blocking findings from steps 3-5, re-dispatch verifiers until clean.
7. Report. Coordinator tells the user what changed, in short. Then, only if asked, commits and pushes.

Steps 3 and 4 always run. Step 5 runs when code that executes changed. A work unit is not done until the verifiers that apply are clean.

## The rule this repo stands on

The agent that produced a change does not judge it done. Verify independently, against external reality (run) and against the spec (ARCHITECTURE.md). This is the same principle the system enforces for claims: same-source is weak, external check is the check.

## Agents

Implementation (do the work, out of coordinator context):

- `implementer`: writes a scoped change. Writes code and docs. Does not commit, does not declare done, does not verify itself.

Verification (independent of the implementer):

- `work-verifier`: adversarial diff review. Hunts fabricated claims, correctness bugs, logic gaps. Read-only: Read, Grep, Glob.
- `arch-checker`: conformance to ARCHITECTURE.md hard rules. Flags contract breaks. Read-only: Read, Grep, Glob.
- `run-verifier`: executes changed code for runtime changes. Does not trust the diff's claims. Has Bash, Read, Grep, Glob.

All agents live in `.agents/`. `.claude` and `.pi` symlink to it.

## Prose rules

No em-dashes. No semicolons in prose. No `--` double hyphens. Use commas and periods. Shell and C statement semicolons are fine. These rules apply to all markdown and docstring prose the agents write. Reverting them wastes a round trip.

## Commits

Commit only when the user asks. Configured author identity. Short imperative messages. No attribution trailers the user has not asked for. Push only when the user asks.

## Doubt

If a change would make Stage 1 (extraction) emit something ARCHITECTURE.md assigns to a later stage (resolved source, trust score, verification verdict), stop and raise it. The benchmark measures one stage, its output must feed the rest without contract breaks.
