# ARCHITECTURE.md

Hard rules and stage contracts for citex-bench. Every component must follow
these. The system exists to catch LLM hallucination, and hallucination thrives
in self-reference, so the core invariant below is non-negotiable.

**Scope note.** citex-bench is a benchmark, not the full pipeline. It measures
Stage 1 (extraction) today and the stages following it only as their contracts
need to be exercised. This document defines the **correct concept end to end**
so the benchmark measures the right thing even where it does not implement the
whole thing. The rule below is what keeps the benchmark honest: it must never
score "verified" on self-reference, even while it only measures extraction.

---

## The one hard rule (read this first)

**A claim is never "verified" by the document it was extracted from.** The
document the claim came from is, at best, a pointer to where to find a source.
Verification only comes from independently fetching and checking an external
source against the claim.

This is the anti-hallucination invariant, and it is the reason the system is
architected as separated stages rather than one LLM pass. An LLM can be fluent
and confident while fabricating citations out of its own context, and the only
defense is to check each claim against something the model did not generate
and cannot see.

```
                 EXTRACT        RESOLVE MARKER        VERIFY EXTERNAL
document  -->  {claim, marker}  -->  source pointer -->  external doc  --> trust score
              (model)              (deterministic)       (MiniCheck)        (calculated)
```

---

## Stage contracts

### Stage 1, Extraction (model)

**Input:** one context chunk of a document, with inline citation markers
(`[1]`, `[2]`) in its body.

**Output:** a set of claims, each with a verbatim `quote` (a span copied from
the chunk) and the `marker` it attributes the claim to.

```json
{"claim": "<restated factual claim>", "quote": "<verbatim span>", "marker": "[1]"}
```

**Hard rules for extraction:**
- The model never emits a `source_ref`, resolved source, URL, or trust score.
 It emits a `marker` token and a `quote`. Resolving the marker to an actual
 source is NOT the model's job (models miscount positions and hallucinate
 spans, see [FINDINGS.md](FINDINGS.md)).
- Uncited opinion, prediction, and hedging are dropped here, not later. If a
 sentence has no marker, it produces no claim.
- The `quote` MUST be a verbatim span present in the chunk, because the
 resolver relies on locating it by string matching.

### Stage 2, Marker resolution (deterministic, NOT a model)

**Input:** the extraction output plus the document's footer references section.

**Output:** each claim's `marker` resolved to a `source` pointer,
`{marker, title, url, ...}`, or `null` if the marker has no footer entry.

**Hard rules for resolution:**
- Parse the document footer into a map `{ "[1]": {title,url,...} }`.
 If no footer exists, every marker resolves to `null` and trust is capped
 (see Trust scoring).
- Resolution is pure string and structure matching. The model is not involved.
- A `source` at this stage is only a POINTER. Resolving a marker to a footer
 entry is not verification, it is "where do I fetch the real source."

### Stage 3, External verification (model, separate from extraction)

**Input:** one claim plus a fetched external document (the source the marker
pointed to, or a retrieval result for the claim).

**Output:** per-source support label from a consistency model (MiniCheck
family), plus the fetched document.

**Hard rules for verification:**
- The verification source MUST be fetched, not generated. It is external to the
 claim document. The model cannot verify a claim against text it also produced.
- A claim verified only against the same document it came from does NOT count
 as verified (see Trust scoring).

### Stage 4, Trust scoring (deterministic)

Compute a `trust_score` per claim, 0..1, from the collected evidence.
Calibrated like a deep-research pass, not a binary pass/fail.

**Hard rules for trust scoring:**
- **Same-document support is devalued.** A claim mentioned multiple times in the
 same document does get some credit (cross-reference within the doc is weak
 corroboration), but it is rated LOWER than a claim with independently fetched
 and verified sources.
- **No full score without external sources.** A claim is NEVER assigned the top
 trust tier on same-document support alone, no matter how many times it recurs.
 The top tier requires at least one external source that was fetched and that
 MiniCheck (or equivalent) judged as supporting the claim.
- Additional research / retrieval sources found beyond the document's own
 citations can raise the score up to the full tier.
- The score is published with its components, not as an opaque number.
 `{same_doc_hits, external_sources_verified, external_sources_checked, tier}`.

**Trust tiers (illustrative, calibration TBD by measurement):**

| tier | rule |
|---|---|
| unverified | no supporting source found at all |
| low | only same-document recurrence, no external source |
| medium | >=1 external source fetched and verified |
| high | >=1 external source verified AND same-doc cross-reference, or multiple independent external sources |

A hallucinated citation (marker points nowhere, or external source does not
support the claim) maps to `unverified`.

---

## Document handling rules

### Footers

A real document carries its sources in a references section at the end
(`[1] Title. URL.`, `[2] ...`). Stage 2 MUST parse this footer. Today the bench
does not (the footer is a documented gap, see
[docs/eval-redesign-plan.md](docs/eval-redesign-plan.md)). Until it is built,
no claim can reach above the `low` tier on same-document support alone, by the
rule above. This is enforced so the gap is visible in scores, not hidden.

### Chunking

Documents exceeding a model's context window are split into context chunks
before Stage 1. Hard rules:
- Chunks are token-budgeted, not character-counted.
- Marker context crosses chunk boundaries. A claim in chunk N that cites a
 marker `[1]` introduced in chunk M (M < N) must still resolve correctly.
 The resolver carries marker definitions across chunks, or a pre-pass extracts
 the footer / marker map before chunking.
- A single claim must not be split across chunks. If a claim straddles a
 boundary, the chunker must keep its full sentence(s) in one chunk.

### What counts as a "claim"

A factual assertion tied to a citation marker. Opinions ("analysts believe"),
predictions ("this trend will continue"), and hedged speculation ("could", "may")
are dropped at Stage 1, not scored. This is a hard rule because these are
exactly the sentence types an LLM produces fluently while fabricating support.

---

## Implementation status (what the benchmark builds vs. specifies)

| stage | in the repo | role |
|---|---|---|
| 1 Extraction | built, measured | `eval.py`, direct and quote architectures, the thing this benchmark runs |
| 2 Marker resolution (footer) | contract only | `resolve_ref_from_quote` returns the marker token today, never a source. The bench must NOT report "`[1]` resolved" as if a source was verified. |
| 3 External verification | contract only, out of scope of a bench run | MiniCheck is the chosen model. A real pipeline fetches and checks. The bench names the contract so Stage 1 output is shaped to feed it. |
| 4 Trust scoring | contract only | the rules above are the spec a real pipeline and a future full-suite benchmark mode fill in |

**Why the benchmark only implements Stage 1 and still needs the full concept:**
the extraction output is the input to Stages 2-4. If Stage 1 emits a schema that
cannot carry a resolved source, or lets the model emit "verification," the later
stages cannot be added without an extraction contract break. So the benchmark
measures one stage but against a schema and invariant set designed for the whole
pipeline. Concretely, that is why extraction emits `{claim, quote, marker}` and
not `{claim, source_ref:"[1]"}` treated as resolved, and why no score in this
bench is ever called a trust score.
