# citex-bench

A CPU-floor benchmark for **claim extraction with citation mapping** on small
quantized language models. Measures both **speed** (can it finish under an
interactive latency budget?) and **accuracy** (does it extract the right claims
and attribute them to the right citations?), and compares two extraction
architectures head-to-head.

Not a generic llama.cpp throughput benchmark. It exists to answer one question
for a specific pipeline: **can a small ternary/1-bit model, running CPU-only with
no GPU, extract cited claims fast enough and accurately enough to be the floor of
a fact-checking tool?**

---

## The intention

Stage (1) of a document fact-checking pipeline: read a document, pull every
factual claim that carries an inline citation marker (`[1]`, `[2]`), ignore
uncited opinion, emit structured JSON `{claim, source_ref}`. Stages (2) and (3)
- claim verification (MiniCheck) and domain credibility, are out of scope here.
this bench measures stage (1) only.

**Hard constraint:** CPU floor, interactive latency, **under 30s per ~1-2k
token document.** That constraint is what makes model selection non-trivial: a
large model would trivially do the extraction well, and trivially fail the
budget.

The bench compares two architectures for the extraction:

- **direct**, the model emits `{claim, source_ref}` directly. The model does the
 localization. (What you'd reach for first.)
- **quote**, the model emits `{claim, quote}` (a verbatim span from the
 document). a deterministic post-processor then locates the quote in the
 document and reads the nearest `[N]` marker. The model never sees a marker.
 This is what two independent 2026 preprints (FullCite, CAMS) converge on -
 see [FINDINGS.md](FINDINGS.md).

For the full research basis, model-card numbers, the FullCite/CAMS finding, ruled-out
alternatives, and the honest limits of that research, read
[**FINDINGS.md**](FINDINGS.md). For how/why models were chosen, read
[**docs/model-selection.md**](docs/model-selection.md).

---

## The models

The PrismML Bonsai family: ternary (`{-1,0,+1}`, ~1.71 bits/weight) and 1-bit
(`{-1,+1}`, ~1.125 bits/weight) representations of Qwen3.6-based models at four
sizes. Run CPU-only via the PrismML llama.cpp fork (the `prism` branch carries
the low-bit CPU kernels. stock upstream does not).

| tier | model | quant | deployed size |
|---|---|---|---|
| `1.7b` | Ternary-Bonsai-1.7B | Q2_0 | ~0.4 GB |
| `1.7b-1bit` | Bonsai-1.7B | Q1_0 | ~0.2 GB |
| `4b` | Ternary-Bonsai-4B | Q2_0 | ~1.0 GB |
| `4b-1bit` | Bonsai-4B | Q1_0 | ~0.5 GB |
| `8b-1bit` | Bonsai-8B | Q1_0 | ~1.1 GB |
| `27b-1bit` | Bonsai-27B | Q1_0 | ~3.6 GB |

See [docs/model-selection.md](docs/model-selection.md) for the full rationale,
the tier-resolution rules, and why 27B is in the speed bench but not the accuracy
grid.

---

## What's in the box

```
Dockerfile CPU-only build of the PrismML llama.cpp fork (llama-bench + llama-completion)
run.sh host wrapper: build image, run the speed benchmark
bench.sh in-container: resolve/download models, run llama-bench, summarize
summarize.py turn llama-bench rows into wall-clock seconds + bandwidth-model check
membw.c STREAM-triad memory-bandwidth probe (decode's explanatory variable)
eval.sh host wrapper: run the accuracy eval across tiers/modes/temps
eval/
 eval.py runs cases via llama-completion, scores JSON/recall/citation-acc
 cases.jsonl 5 synthetic documents with expected {source_ref, key_phrase}
 verify.py deterministic re-verification of the scorer (pure-function check)
docs/
 model-selection.md why these models/tiers
 eval-redesign-plan.md the multi-pass + full Cartesian redesign (planned, not built)
reports/
 00-results-2026-07-18.md the scout-run results writeup
 accuracy-arch-temp.log raw accuracy eval log
results/
 raw-host.json / summary-host.txt llama-bench speed output
```

Models cache to `./models/` (bind-mounted. survives rebuilds, gitignored).

---

## Quickstart

Requires Docker. Native build only, do not pass `--platform` to cross-build.
an emulated container benchmarks QEMU, not your CPU.

```bash
# Speed benchmark (default tiers):
./run.sh

# One tier, scoped workload:
TIERS="1.7b-1bit" DOC_TOKENS=700 OUT_TOKENS=300 REPS=1 ./run.sh

# Label the machine for cross-machine comparison (do NOT use your real hostname):
LABEL=thinkpad-t14 ./run.sh

# Accuracy eval, direct vs quote architectures, temps 0/0.5/0.7:
TIERS="1.7b-1bit 4b-1bit 8b-1bit" ./eval.sh
```

Docker Desktop users: Settings -> Resources, give the VM all cores and >=12 GB
RAM, or the larger tiers OOM or thrash.

---

## The measured results (scout run)

Single-pass, directional only, see
[reports/00-results-2026-07-18.md](reports/00-results-2026-07-18.md) for the full
tables and [docs/eval-redesign-plan.md](docs/eval-redesign-plan.md) for why a
multi-pass rebuild is planned.

**Speed** (700-token doc in, 300 out, 7 threads, CPU only):

| tier | size | total | <30s |
|---|---|---|---|
| 1.7b-1bit | 0.2G | 9.7s | OK |
| 4b-1bit | 0.5G | 22.2s | OK |
| 4b (ternary) | 1.0G | 42.4s | NO |

**Accuracy** architecture comparison (1.7b-1bit):
- **quote** beats **direct** on citation accuracy at every temperature (+100pts)
 , direct mode at 1.7B stops after one claim or misattributes.
- At **4B**, direct mode is perfect and *faster*. quote mode degrades at temp>0.
- **8B** emits a reasoning preamble before answering and never produces the JSON
 array, a prompting failure to fix before it's benchmarkable.

The headline finding's important nuance: the research's "never let the model
localize" conclusion was drawn at 8B-12B+ on *exact-span F1*, not *marker
matching* (our metric, strictly easier). So at marker-matching on small docs, 4B
direct genuinely works, even though it would likely still fail FullCite's
exact-span test. The research and our measurement aren't contradicting. they
measure different things.

---

## Honesty

- The accuracy case set is **5 synthetic documents**. Small, hand-authored, not
 representative of real-world documents. Treat accuracy numbers as
 direction, not decision-grade.
- **Single-pass.** No variance reported. The planned multi-pass redesign exists
 precisely to make the numbers trustworthy.
- The low-bit CPU kernels measured **7-12% bandwidth-model efficiency** vs the
 50-80% a tuned kernel gets, meaning the kernels are immature on this ARM path
 and there's headroom left. Absolute numbers are real but don't extrapolate to
 x86 by bandwidth ratio alone. rerun on the target box.
- This is a research/benchmarking tool, not production code. The eval harness
 has been correctness-reviewed (scorer is a pure, deterministic function.
 see `eval/verify.py`) but has not been hardened against adversarial input.

## License

Model weights are Apache-2.0 (PrismML / Bonsai). This benchmark harness code is
released under the MIT License.
