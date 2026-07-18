# citex-bench - research for fast LLM claim extraction & validation

![created with](https://img.shields.io/badge/created%20with-GLM%205.2-blue)
![created with](https://img.shields.io/badge/created%20with-Ornith%201.0%2035B-purple)
![created with](https://img.shields.io/badge/created%20with-Claude%20Code-orange)
![created with](https://img.shields.io/badge/created%20with-pi.dev-green)

## Credits

The current model and low-bit-kernel provider is **PrismML** (Bonsai). Thanks to
PrismML for pushing for efficiency. Third-party licenses and notices:
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)

<a href="https://huggingface.co/prism-ml"><img src="https://raw.githubusercontent.com/PrismML-Eng/Bonsai-demo/main/assets/bonsai-logo.svg" width="220" alt="PrismML Bonsai"></a>

---

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

### Sample data

What goes in and what should come out (from `eval/cases.jsonl`):

**Document (in):** inline citation markers in the body text.

> Germany's grid operators reported record renewable output last month.
> According to a report from CleanEnergyWire [1], wind and solar covered 68%
> of national electricity demand on April 14th, a new national record. The
> report also notes that grid curtailment costs reached 412 million euros in
> the first quarter [1]. Some analysts believe this trend will continue for
> years, though such predictions are inherently speculative. A separate
> statement from the grid operator TenneT [2] confirmed that no blackouts
> occurred despite the surge.

**Expected extraction (out):** each cited factual claim mapped to its marker.
The uncited analyst opinion is dropped.

```json
[
  {"source_ref": "[1]", "key_phrase": "68%"},
  {"source_ref": "[1]", "key_phrase": "412 million"},
  {"source_ref": "[2]", "key_phrase": "no blackouts"}
]
```

In `direct` mode the model emits this JSON itself. In `quote` mode it emits
`{claim, quote}` and a deterministic resolver recovers the marker from the
document text (see [FINDINGS.md](FINDINGS.md)).

### Known gaps (not yet handled)

The bench's current sample is a single short excerpt with markers inline. Real
documents differ in two ways that matter, and neither is handled yet:

- **Chunking.** Longer documents exceed a small model's context window and must
 be split into context chunks before extraction. The claim-to-marker mapping
 then has to survive chunk boundaries (a claim may cite a marker introduced in
 an earlier chunk). Nothing here chunks today.
- **Footer references.** Real documents usually carry their sources in a
 references section at the end (`[1] ...`, `[2] ...`), not as inline markers
 alone. Resolving a marker to its actual source (URL, title, author) means
 parsing that footer. The bench treats markers as opaque labels today and does
 not resolve them to footer entries at all.

Both are tracked as the next steps after the multi-pass refactoring, see
[docs/eval-redesign-plan.md](docs/eval-redesign-plan.md).

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

## Tested models

The current configured provider is **PrismML** (Bonsai): ternary
(`{-1,0,+1}`, ~1.71 bits/weight) and 1-bit (`{-1,+1}`, ~1.125 bits/weight)
representations of Qwen3.6-based models at four sizes, run CPU-only via the
PrismML llama.cpp fork (the `prism` branch carries the low-bit CPU kernels,
stock upstream does not).

| tier | model | quant | deployed size |
|---|---|---|---|
| `1.7b` | Ternary-Bonsai-1.7B | Q2_0 | ~0.4 GB |
| `1.7b-1bit` | Bonsai-1.7B | Q1_0 | ~0.2 GB |
| `4b` | Ternary-Bonsai-4B | Q2_0 | ~1.0 GB |
| `4b-1bit` | Bonsai-4B | Q1_0 | ~0.5 GB |
| `8b-1bit` | Bonsai-8B | Q1_0 | ~1.1 GB |
| `27b-1bit` | Bonsai-27B | Q1_0 | ~3.6 GB |

Providers and tiers are configured in `bench.sh`'s `tier_spec()`. That is the
single place to point at a different vendor or model family, the rest of the
harness is provider-agnostic. See [docs/model-selection.md](docs/model-selection.md)
for the rationale.

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
├── THIRD_PARTY_LICENSES.md  third-party licenses and notices
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

## Measured results

See [reports/](reports/) for dated result writeups and raw logs. The current
entry is a single-pass scout run, directional only. A multi-pass rebuild with
full parameter sweep is the planned next step in
[docs/eval-redesign-plan.md](docs/eval-redesign-plan.md).

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
