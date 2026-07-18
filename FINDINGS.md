# Research: Claim Extraction + Citation-Marker Mapping on a CPU Floor

**Date:** 2026-07-18
**Method:** 5 search angles -> 9 primary sources fetched -> 45 claims extracted -> 12 verified adversarially -> **5 survived, 7 refuted**
**Verification:** single adversarial verifier per claim, instructed to refute. Claims below carry their vote and confidence.

---

## The question

For a document fact-checking pipeline, stage (1) extract factual claims carrying inline citation markers (`[1]`, `[2]`), ignore uncited opinion, emit structured JSON. stage (2) verify with MiniCheck-RoBERTa. stage (3) domain credibility, running **CPU-only, under 30s per 1-2k token document**:

1. Is there a purpose-built model/dataset for attribution extraction / citation-span detection / claim-to-source mapping?
2. How do classical/lightweight NLP techniques compare to a generative LLM, in accuracy and CPU latency?
3. What existing benchmark does this most resemble, whose methods could be reused?
4. Fine-tuning a small model on a few hundred examples vs zero-shot prompting a small LLM?

---

## Executive summary

**No purpose-built off-the-shelf model for stage (1) surfaced.** The nearest named benchmark, AttributionBench, is attribution *evaluation*, the stage (2) job MiniCheck already covers.

**The strongest finding is architectural, not model-selection.** Two independent 2026 primary sources converge on the same design: **decouple *what is claimed* from *where it is stated*.** Have the LLM emit claim text plus a **verbatim quote**, then recover the span/citation **deterministically by string alignment**, rather than asking the model for offsets or trusting spans it emits. That post-hoc alignment step is worth roughly **5x on evidence-span F1** (12.80 -> 61.87 snippet-F1 on ASQA) in one pipeline and a **97% quote-match rate** in the other, and it **beats decoding-time constraints**.

**Practical recommendation for a CPU-only sub-30s budget:** keep the small prompted LLM for claim decomposition *only*, and move citation-marker resolution out of the model into deterministic post-processing (regex marker extraction + exact-then-fuzzy quote matching). Cheap, near-zero latency, and the evidence says it outperforms letting the model do the localization.

---

## FINDING 1: The headline

> **Do not ask the LLM for spans, offsets, or the citation mapping. Ask it for a verbatim quote, then resolve the location deterministically by string matching.**

**Confidence: HIGH** | Vote 1-0 (merged from two independently-verified claims)
**Sources:** [FullCite, arXiv 2606.07130](https://arxiv.org/abs/2606.07130) | [CAMS, arXiv 2606.23989](https://arxiv.org/abs/2606.23989)

Two independent 2026 primary sources arrive at this design separately. That convergence is the reason this finding carries the most weight in the whole run.

### FullCite (arXiv 2606.07130)

Compares three strategies for producing structured inline citations that link each claim to its source document **and** its supporting evidence span:

| Strategy | Snippet-F1 (ASQA, Qwen3-8B) |
|---|---|
| Prompt-based generation (model emits spans directly) | **12.80** |
| Grammar-constrained decoding over a citation grammar | **55.11** |
| **Post-hoc span alignment** | **61.87** |

The paper's abstract states *"posthoc yields the largest gains in correct evidence identification."*

**Two things this establishes:**
1. Asking a model to emit spans directly is catastrophically bad (12.80).
2. **Post-hoc alignment beats constrained decoding** (61.87 vs 55.11), so the fix is *not* to work harder at forcing correctness during generation. Move the job out of the model entirely.

### CAMS (arXiv 2606.23989)

Extracts atomic factual claims with token-level provenance, linking each claim back to specific source spans. It **explicitly refuses** to request offsets from the model:

> Sec. 3.2: *"A direct demand for character or token offsets is unreliable... instruction-tuned models routinely miscount positions and hallucinate spans. We therefore decouple what is claimed from where it is stated."*

Its resolution mechanism, directly implementable, and the concrete recipe worth copying:

> Sec. 4.4: *"quotes are resolved by exact then rapidfuzz indel matching (ratio=0.85) before conversion to token spans"*

**Reported outcome: 97% quote-match rate** (Sec. 5).

### What this implies concretely

**Instead of:**
```json
{"claim": "<restatement>", "source_ref": "[1]"}
```

**Do:**
```json
{"claim": "<restatement>", "quote": "<verbatim span copied from the document>"}
```

Then, with no model involved:
1. Locate `quote` in the source document, **exact match first**, then **fuzzy** (`rapidfuzz` indel, ratio ~ 0.85).
2. Read the nearest citation marker via regex (`\[\d+\]`).
3. Emit `source_ref` from that.

**Secondary benefit:** the model stops emitting citation bookkeeping tokens, which cuts decode, directly relevant when decode is a large share of a tight latency budget.

### Caveats on Finding 1 (read these before committing)

- **Both are unrefereed 2026 preprints**, single-author or single-lab, self-reporting their own numbers (including the 97% figure).
- **FullCite's gain is uneven across datasets.** Gemma-3-12B on BioASQ actually *degrades* under post-hoc alignment: 28.84 -> 20.90. The technique is not universally positive.
- **A simpler Generate-then-Retrieve baseline beats FullCite post-hoc on ASQA** (75.07 vs 61.87 snippet-F1). If anything this *strengthens* the core argument, cheap deterministic recovery outperforms model-side span emission, but it means FullCite's specific method is not the ceiling.
- **All figures come from 8B-12B+ models on GPU/API.** Whether a 1.7B ternary-quantized model emits quotes verbatim enough for exact-then-fuzzy matching to work is **untested** (see Open Question 1).

---

## FINDING 2: No purpose-built model for stage (1)

**Confidence: MEDIUM** (absence-of-evidence, not proven non-existence) | Vote 1-0
**Source:** [AttributionBench, arXiv 2402.15089](https://arxiv.org/abs/2402.15089)

The benchmark most often conflated with this task is **attribution evaluation**, not extraction. Its abstract frames the problem as *"evaluating the answer's attribution, i.e., whether every claim within the generated responses is fully supported by its cited evidence"*, precisely the stage (2) role MiniCheck already fills.

Its construction is a unified reformatting of existing attribution-**classification** datasets (AttributedQA, HAGRID, ExpertQA, Stanford-GenSearch, AttrScore, BEGIN). All supply *both* the claim and the evidence and ask for an attributable/not-attributable label. **Nothing in it segments a document into claims or resolves inline `[N]` markers.**

**Why only medium confidence:** the verifier's adversarial search tooling errored, so this rests on entailment from the primary abstract plus knowledge of the benchmark format, not a fresh search for a later extraction-subtask variant. Treat as "none found in a partially-degraded search," not a settled negative.

### The broader benchmark landscape (RQ3)
Every adjacent benchmark family found runs the **inverse** direction of stage (1), they *generate* citations or *evaluate* given pairs. none *extracts* pre-existing markers:

| Benchmark | What it actually does | Relation to stage (1) |
|---|---|---|
| [ALCE 2305.14627](https://arxiv.org/abs/2305.14627) | LLMs retrieve evidence + generate cited answers | Inverse (generation) |
| [LongCite 2409.02897](https://arxiv.org/abs/2409.02897) | Trains LLMs to generate fine-grained sentence-level citations | Inverse (generation) |
| [AttributedQA 2212.08037](https://arxiv.org/abs/2212.08037) | Formal task: is a generated answer attributable to a source? | Stage (2) |
| [HAGRID 2307.16883](https://arxiv.org/abs/2307.16883) | GPT-3.5 generates attributed explanations over MIRACL. humans label | Inverse (generate-then-verify) |
| [AttributionBench 2402.15089](https://arxiv.org/abs/2402.15089) | Binary attributable/not classification | Stage (2) |
| [Citation-needed 1902.11116](https://arxiv.org/abs/1902.11116) | Sentence-level *"does this need a citation?"* classification | Adjacent, different output |

---

## FINDING 3, Fine-tuning beats prompting, but not at the scale you asked about

**Confidence: HIGH** | Vote 1-0
**Source:** [ReClaim, arXiv 2407.01796](https://arxiv.org/abs/2407.01796)

Table 2, ASQA (MAUVE / EM / **CAS** = citation accuracy / CRS):

| System | MAUVE | EM | **Citation acc.** | CRS |
|---|---|---|---|---|
| 0-shot GPT-4o | 72.9 | 52.8 | **74.8** | 51.6 |
| 3-shot GPT-4o | 91.3 | 56.6 | **77.4** | 58.0 |
| ReClaim w/IG **Llama2-7B** | 71.4 | 55.0 | **89.5** | 78.7 |
| ReClaim w/IG **Llama3-8B** | 88.1 | 53.5 | **92.1** | 86.1 |

A fine-tuned 7B/8B model beats GPT-4o by **~15 points on citation accuracy**.

**Three load-bearing qualifications:**
- **(a)** The fine-tuning set derives from **WebGLM-QA, orders of magnitude beyond "a few hundred examples."** This is explicitly **NOT** evidence that a small hand-labeled set suffices. RQ4 as asked remains **unanswered**.
- **(b)** ReClaim is *worse* than GPT-4o on answer correctness (EM 53.5 vs 56.6) and fluency (MAUVE 88.1 vs 91.3). **The win is specifically on citation attribution**, nothing else.
- **(c)** The task is attributed *generation* (writing cited answers from retrieved passages), not extracting pre-existing markers. Transfer is plausible but unproven.
- **(d)** The GPT-4o baselines are the ReClaim authors' own prompt implementations, not an optimized GPT-4o ceiling.

---

## FINDING 4, Classical vs LLM decomposition: weak, unmeasured

**Confidence: MEDIUM** | Vote 1-0
**Source:** [CAMS, arXiv 2606.23989](https://arxiv.org/abs/2606.23989)

The only direct comparison found favors the LLM, **as a design assertion, not a measured result.**

> Sec. 3.2: *"Unlike off-the-shelf OpenIE (Angeli et al., 2015) used in prior modular work (Guan and Padmakumar, 2023), LLM decomposition improves recall and yields fluent, normalized claims while still anchoring each to its source at the first stage rather than recovering it later."*

**Why this is weak evidence:**
- **The paper runs NO OpenIE-vs-LLM ablation.** Sec. 5.9 ablations cover verification, clustering, and selector only. Appendix F reports extraction diagnostics with no classical baseline.
- The prior work being dismissed is **the same author's own**.
- It says **nothing about CPU latency**, CAMS's extraction stage runs a frozen `claude-opus-4-8` API model.

**Treat as:** weak directional support for keeping an LLM in the decomposition loop. **Not** proof that classical segmentation fails.

---

## REFUTED, 7 claims killed in verification. Do not act on these.

These looked plausible and did **not** survive adversarial checking:

1. ~~"LLMs prompted zero-shot reliably identify the correct source *document* but only fail at the precise *span*, document-level attribution is much easier than span-level."~~
2. ~~"The post-hoc alignment baseline is built from classical CPU-friendly components: sentence splitting + BM25 + all-MiniLM-L6-v2 retrieval."~~ ⚠️ **Consequence: you cannot assume the alignment step is cheap on the basis of the cited systems.**
3. ~~"ALCE performs claim-to-citation mapping with pure rule-based sentence segmentation + bracket-marker parsing, no learned model."~~ ⚠️ **Consequence: the claim that "classical regex is the accepted method for the mapping step" is NOT supported.**
4. ~~"ALCE's citation-quality evaluation reduces to an NLI/entailment model over (cited passages, statement) pairs, confirming extraction is deterministic and only verification needs a model."~~
5. ~~"The extraction stage is a 3-shot prompt over a frozen LLM returning JSON with fields claim/quote/doc_id."~~
6. ~~"Constrained decoding over a token-level prefix tree eliminates fabricated references, a cheap decoding-time technique."~~
7. ~~"No purpose-built model existed as of this work. the authors explicitly declined to train one for lack of supervised data."~~

**Note the pattern:** refutations #2 and #3 are exactly the claims that would have made the "just use regex, skip the model" story airtight. **They did not hold.** The deterministic-alignment recommendation in Finding 1 stands on FullCite/CAMS's *measured* results, not on any verified claim that regex alone is sufficient.

---

## CAVEATS, the honest limits of this research run

- **Three of the five surviving claims rest on 2026 arXiv preprints** (2606.07130, 2606.23989) that are **unrefereed, single-author or single-lab, and self-reporting their own numbers**, including the 97% quote-match rate. The GPT-4o baselines in 2407.01796 are likewise the authors' own prompt implementations rather than an optimized GPT-4o ceiling.

- **RQ2 is effectively unanswered.** Seven claims were refuted, and the casualties include every claim asserting that ALCE performs mapping with rule-based segmentation + bracket regex, and that the post-hoc alignment baseline was built from CPU-friendly BM25 + MiniLM. **The CPU-friendliness of the recommended alignment step is an inference from the technique's nature (string matching, sentence splitting), not a verified property of the cited systems.**

- **NO source in this run reports latency measurements on CPU-only hardware at all.** Every cited experiment ran large models (GPT-4o, claude-opus-4-8, Qwen3-8B, Gemma-3-12B) on **GPU or API**. None is direct evidence about the 1.7B-27B quantized llama.cpp regime. **The accuracy gaps reported here may narrow, widen, or invert at your model scale.**

- **The verifier's WebSearch/WebFetch tooling errored repeatedly.** Claims were checked by direct fetch of primary sources **without an independent adversarial search for contradicting evidence**. The "no purpose-built stage-1 model" conclusion is absence-of-evidence within a partially-degraded search.

- **None of the sources studies the exact task**, extracting claims and resolving *pre-existing* inline markers in a user-supplied document, as opposed to *generating* cited text or *evaluating* given claim/evidence pairs.

---

## OPEN QUESTIONS, ranked by value

1. **Does the post-hoc alignment gain survive at small model scale?**
 All reported figures use 8B-12B+ models. Whether a **1.7B ternary-quantized model emits verbatim quotes accurate enough** for exact-then-fuzzy matching to approach a 97% match rate is **untested, and is the single highest-value thing to measure** in an empirical harness.

2. **What is the accuracy floor of the pure-classical baseline?**
 Sentence segmentation + regex bracket-marker attachment + nearest-preceding-marker heuristic. **No source verified this** (the ALCE claims asserting it were refuted), and it is cheap enough to implement in an afternoon. **It should be measured as the control before any model is chosen**, a sub-30s CPU budget makes a good-enough deterministic baseline very hard to beat.

3. **Does the fine-tuning advantage hold at a few-hundred-example scale?**
 ReClaim's win used a WebGLM-QA-derived set orders of magnitude larger. The label-efficiency curve for citation-mapping fine-tuning is unknown, and it determines whether labeling a small set is viable at all.

4. **How should uncited opinion/prediction filtering be handled**, inside the LLM prompt, or as a separate lightweight sentence classifier?
 **None of the surveyed work addresses discarding non-factual or unattributed sentences**, which is an explicit requirement of stage (1). It may be a cheaper, more reliable target for a small BERT-sized classifier than the extraction itself.

---

## Sources (9 primary, all fetched and read)

| # | Source | Angle |
|---|---|---|
| 1 | [arXiv 2606.07130](https://arxiv.org/abs/2606.07130), FullCite | purpose-built extraction models |
| 2 | [arXiv 2606.23989](https://arxiv.org/abs/2606.23989), CAMS | purpose-built extraction models |
| 3 | [arXiv 2407.01796](https://arxiv.org/abs/2407.01796), ReClaim | purpose-built extraction models |
| 4 | [arXiv 2305.14627](https://arxiv.org/abs/2305.14627), ALCE | purpose-built extraction models |
| 5 | [arXiv 2402.15089](https://arxiv.org/abs/2402.15089), AttributionBench | purpose-built extraction models |
| 6 | [arXiv 1902.11116](https://arxiv.org/abs/1902.11116), Citation Needed | purpose-built extraction models |
| 7 | [arXiv 2409.02897](https://arxiv.org/abs/2409.02897), LongCite | closest benchmark family |
| 8 | [arXiv 2212.08037](https://arxiv.org/abs/2212.08037), Attributed QA | closest benchmark family |
| 9 | [arXiv 2307.16883](https://arxiv.org/abs/2307.16883), HAGRID | closest benchmark family |

**Run stats:** 5 angles | 9 sources fetched (2 URL dupes removed) | 45 claims extracted | 12 verified | 5 confirmed | 7 refuted | 0 unverified | 28 agent calls
