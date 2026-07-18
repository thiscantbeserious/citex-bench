# citex-bench

![model](https://img.shields.io/badge/model-GLM%205.2-blue)
![model](https://img.shields.io/badge/model-Ornith%201.0%2035B-purple)
![harness](https://img.shields.io/badge/harness-Claude%20Code-orange)
![harness](https://img.shields.io/badge/harness-pi.dev-green)

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

## Background

This benchmark comes out of work on verifying the credibility of research and
writing that LLMs produce or scaffold. As more reports and prose are generated
or assisted by large language models, the citations in that text stop being a
guarantee and become a claim of their own: does this source actually say what
the text says it does? citex-bench tests whether a small local model can be the
first step in answering that, by extracting cited claims and mapping them to
their sources so each can be checked rather than taken on trust.

The deployment target is deliberately the CPU floor, any machine, not a GPU rig
or a billed API. The intended host is a coding-agent harness (Claude Code at
present, Codex, pi.dev, opencode, and similar), where a local verification step
has to fit inside the agent loop's interactive latency budget. The harness is a
swappable runtime, not a fixed choice, nothing here depends on it.

See [**FINDINGS.md**](FINDINGS.md) for the research basis.

---

## The intention

The verification pipeline has three stages. This bench measures stage (1) only,
the others are named so the scope is explicit:

1. **Claim extraction**, read a document, pull every factual claim that carries
 an inline citation marker (`[1]`, `[2]`), ignore uncited opinion, emit
 structured JSON `{claim, source_ref}`. *What this bench measures.*
2. **Claim verification**, check each extracted claim against its cited source
 for support. Out of scope here.
3. **Domain credibility**, score sources for trustworthiness. Out of scope here.

**Hard constraint:** CPU floor, interactive latency, **under 30s per ~1-2k
token document.** A large model would trivially do the extraction well, and
trivially fail the budget.

The bench compares two architectures for the extraction:

- **direct**, the model emits `{claim, source_ref}` directly. The model does the
 localization. (What you'd reach for first.)
- **quote**, the model emits `{claim, quote}` (a verbatim span from the
 document). a deterministic post-processor then locates the quote in the
 document and reads the nearest `[N]` marker. The model never sees a marker.
 This is what two independent 2026 preprints (FullCite, CAMS) converge on -
 see [FINDINGS.md](FINDINGS.md).

For how/why the model tiers were chosen, read
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
citex-bench/
├── Dockerfile           build definition for the CPU-only llama.cpp image
├── run.sh / eval.sh     host wrappers for speed / accuracy
├── bench.sh             in-container speed benchmark
├── summarize.py         turns bench output into wall-clock seconds + bandwidth check
├── membw.c              STREAM-triad memory-bandwidth probe
├── eval/                accuracy harness, test cases, scorer self-check
├── docs/                rationale and plans
├── reports/             result writeups and raw logs
├── results/             raw speed output
├── FINDINGS.md          research writeup
├── LICENSE              MIT (harness code)
└── .gitignore
```

Models cache to `./models/` (bind-mounted, survives rebuilds, gitignored).

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
