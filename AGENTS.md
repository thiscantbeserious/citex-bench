# AGENTS.md

Write efficient, short sentences. Be token efficient, not verbose. Apply this to everything here. `CLAUDE.md` symlinks to this file. Read README and ARCHITECTURE for context, do not restate them.

## Work loop (always, numbered)

Every change, in order, no skipping:

1. Scope. Name the change, the files in scope, the done condition. One or two sentences.
2. Implement. Write the change with the `implementer` agent. Code and docs. It does not commit or self-verify.
3. Verify-code. Dispatch `work-verifier` on the diff. Adversarial, read-only. Blocking findings go back to step 2.
4. Verify-arch. Dispatch `arch-checker` on the diff. Conformance to ARCHITECTURE.md, read-only. Blocking findings go back to step 2.
5. Verify-run. Runtime changes only. Dispatch `run-verifier`. It executes the changed code. Real output is the check, not the diff's claims.
6. Resolve. Fix all blocking findings from steps 3 to 5. Re-dispatch verifiers until clean.
7. Report. Say what changed, short. Commit and push only if the user asks.

Steps 3 and 4 always run. Step 5 runs when executable code changed. A change is not done until the verifiers that apply are clean.

## Agents

All in `.agents/`. `.claude` and `.pi` symlink to it.

- `implementer`: writes a scoped change. Does not commit, declare done, or self-verify.
- `work-verifier`: adversarial diff review. Hunts fabricated claims, bugs, gaps. Read-only: Read, Grep, Glob.
- `arch-checker`: conformance to ARCHITECTURE.md. Flags contract breaks. Read-only: Read, Grep, Glob.
- `run-verifier`: executes changed code. Does not trust the diff's claims. Bash, Read, Grep, Glob.

## Prose rules

No em-dashes. No semicolons in prose. No `--`. Use commas and periods. Shell and C semicolons are fine.

## Commits

Commit and push only when the user asks. Short imperative messages. No attribution trailers.
