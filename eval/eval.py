#!/usr/bin/env python3
"""Claim-extraction accuracy eval, the half that raw tok/s can't answer.

Compares two architectures for stage (1) on the SAME cases:

  direct : model emits {claim, source_ref}          model does the localization
  quote  : model emits {claim, quote}, then we      model never sees a marker
           locate the quote in the doc and read       citation resolved by string
           the nearest [N] marker with a regex        matching + regex

The `quote` mode is what FullCite (arXiv 2606.07130) and CAMS (arXiv 2606.23989)
converge on: don't ask the model for spans/offsets, ask for a verbatim quote and
recover the location deterministically. CAMS resolves by exact match then fuzzy
matching at ratio 0.85. We do the same with difflib (stdlib) instead of rapidfuzz
so the container needs no extra dependency.

Open question this measures (research called it the highest-value unknown):
those results are all from 8B-12B+ models. Whether a 1.7B ternary/1-bit model
emits quotes verbatim enough for the matching to land is untested.
"""
import argparse, difflib, json, os, re, subprocess, sys, time

MARKER_RE = re.compile(r"\[\d+\]")

PROMPT_DIRECT = """Extract every factual claim in the DOCUMENT that is attributed to a citation marker like [1] or [2]. Ignore claims, opinions, or predictions with no citation marker.

Output ONLY a JSON array. Each element must be:
{{"claim": "<concise restatement>", "source_ref": "<the citation marker, e.g. [1]>"}}

If there are no cited claims, output exactly: []

DOCUMENT:
{doc}

JSON:"""

PROMPT_QUOTE = """Extract every factual claim in the DOCUMENT that is attributed to a citation marker like [1] or [2]. Ignore claims, opinions, or predictions with no citation marker.

Output ONLY a JSON array. Each element must be:
{{"claim": "<concise restatement>", "quote": "<the exact sentence from the DOCUMENT, copied word for word>"}}

Copy the quote EXACTLY as it appears in the document. Do not include the citation marker in the quote.

If there are no cited claims, output exactly: []

DOCUMENT:
{doc}

JSON:"""


def run_model(binary, model, prompt, threads, ctx, max_tokens, timeout, sample):
    """One-shot completion. Uses llama-completion, NOT llama-cli: in this fork
    llama-cli is an interactive REPL that rejects -no-cnv and spins forever on
    EOF stdin (~1GB of "> " prompts, then OOM).

    `sample` carries the full Bonsai-recommended sampling config, because the
    model card tunes temperature together with top-k/top-p/penalties, sweeping
    temperature in isolation at temp>0 is off-spec. At temp=0 (greedy) the
    sampler short-circuits and top-k/top-p are inert, which is fine."""
    cmd = [binary, "-m", model, "-p", prompt, "-n", str(max_tokens),
           "-t", str(threads), "-c", str(ctx), "-ngl", "0",
           "--temp", str(sample["temp"]),
           "--top-k", str(sample["top_k"]),
           "--top-p", str(sample["top_p"]),
           "--repeat-penalty", str(sample["repeat_penalty"]),
           "--presence-penalty", str(sample["presence_penalty"]),
           "--seed", str(sample["seed"])]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=timeout, stdin=subprocess.DEVNULL)
    return r.stdout


def extract_json_array(raw):
    tail = raw.rsplit("JSON:", 1)[-1]
    m = re.search(r"\[.*\]", tail, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def resolve_ref_from_quote(quote, doc, cutoff=0.85):
    """Deterministic citation resolution, the CAMS recipe.
    Exact match first, then fuzzy. Then read the nearest following [N] marker.
    Returns (source_ref|None, matched: bool)."""
    if not quote:
        return None, False
    idx = doc.find(quote)
    if idx >= 0:
        end = idx + len(quote)
    else:
        # Fuzzy: align the quote against the whole doc by matching character
        # blocks. Comparing against whole sentences fails here, a sentence's
        # "According to X [1], " prefix dilutes the ratio below cutoff even when
        # the quote matches the rest verbatim. Summing matched blocks measures
        # "how much of the quote appears in the doc", which is the actual question.
        sm = difflib.SequenceMatcher(None, quote.lower(), doc.lower(), autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
        if not blocks or sum(b.size for b in blocks) < cutoff * len(quote):
            return None, False
        idx = blocks[0].b
        end = max(b.b + b.size for b in blocks)
    # A citation marker belongs to the SENTENCE containing the quote, it may sit
    # mid-sentence ("According to X [1], wind and solar...") or terminate it
    # ("...in the first quarter [1]."). Scoping to the sentence is what keeps an
    # UNCITED sentence from inheriting a neighbour's marker, which would fabricate
    # citations for exactly the opinion sentences stage (1) must discard.
    bounds = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+", doc)] + [len(doc)]
    s_start, s_end = 0, len(doc)
    for i in range(len(bounds) - 1):
        if bounds[i] <= idx < bounds[i + 1]:
            s_start, s_end = bounds[i], bounds[i + 1]
            break
    s_end = max(s_end, end)              # quote may run past one sentence boundary
    markers = MARKER_RE.findall(doc[s_start:s_end])
    return (markers[-1] if markers else None), True


def score_case(predicted, expected, mode, doc):
    base = {"valid_json": False, "recall": 0.0, "precision": 0.0,
            "citation_acc": None, "matched": 0, "expected_n": len(expected),
            "quote_match": None, "f1": 0.0}
    if predicted is None or not isinstance(predicted, list):
        return base
    if not expected:
        recall = 1.0 if len(predicted) == 0 else 0.0
        return {**base, "valid_json": True, "recall": recall,
                "precision": 0.0, "f1": 0.0}

    # Resolve each prediction to a source_ref (differs per architecture).
    resolved, quote_hits = [], 0
    for p in predicted:
        if not isinstance(p, dict):
            continue
        claim = str(p.get("claim", ""))
        if mode == "quote":
            ref, hit = resolve_ref_from_quote(str(p.get("quote", "")), doc)
            quote_hits += hit
        else:
            ref = str(p.get("source_ref", "")).strip()
        resolved.append((claim, ref))

    # Greedy one-to-one assignment between expected and resolved predictions.
    # Each expected and each prediction used at most once. Pairs below 0.5
    # similarity never assign. Sort by similarity descending, assign in that
    # order skipping any side already used. Python's sort is stable, so ties
    # break in expected-then-prediction iteration order.
    candidates = []
    for ei, exp in enumerate(expected):
        key = exp["key_phrase"].lower()
        for pi, (claim, ref) in enumerate(resolved):
            c = claim.lower()
            if key in c:
                sim = 1.0
            else:
                sim = difflib.SequenceMatcher(None, key, c).ratio()
            if sim >= 0.5:
                candidates.append((sim, ei, pi, ref))
    candidates.sort(key=lambda t: -t[0])

    used_exp, used_pred = set(), set()
    assigned, citation_correct = 0, 0
    for sim, ei, pi, ref in candidates:
        if ei in used_exp or pi in used_pred:
            continue
        used_exp.add(ei)
        used_pred.add(pi)
        assigned += 1
        if ref == expected[ei]["source_ref"]:
            citation_correct += 1

    n_pred = len(predicted)
    recall = assigned / len(expected)
    precision = (assigned / n_pred) if n_pred else 0.0
    citation_acc = (citation_correct / assigned) if assigned else 0.0
    f1 = (2 * precision * recall / (precision + recall)) \
        if (precision + recall) > 0 else 0.0

    return {"valid_json": True,
            "recall": recall,
            "precision": precision,
            "citation_acc": citation_acc,
            "matched": assigned, "expected_n": len(expected),
            "quote_match": (quote_hits / len(resolved)) if (mode == "quote" and resolved) else None,
            "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", default="/opt/eval/cases.jsonl")
    ap.add_argument("--threads", default=str(os.cpu_count() or 4))
    ap.add_argument("--ctx", default="2048")
    ap.add_argument("--max-tokens", default="400")
    ap.add_argument("--binary", default="llama-completion")
    ap.add_argument("--tier", default="")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--mode", choices=["direct", "quote", "both"], default="both")
    ap.add_argument("--temps", default="0",
                   help="comma-separated temperatures to sweep (default 0 = greedy)")
    # Bonsai model-card recommended sampling: temp 0.5 default (0.5-0.7), top-k 20
    # (20-40), top-p 0.9 (0.85-0.95), repeat-penalty 1.0. The 1.7B needs a presence
    # penalty (card flags it without a value). 0.5 is a modest nonzero default.
    ap.add_argument("--top-k", type=float, default=20)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repeat-penalty", type=float, default=1.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # The 1.7B card specifically calls for a presence penalty. Apply one unless the
    # caller explicitly set --presence-penalty. Detect by tier, not model path, so
    # the ternary 1.7B is covered too.
    presence = args.presence_penalty
    if presence == 0.0 and args.tier and "1.7b" in args.tier:
        presence = 0.5
        print(f"  (1.7b tier: applying presence-penalty={presence} per model card)", flush=True)

    cases = [json.loads(l) for l in open(args.cases) if l.strip()]
    modes = ["direct", "quote"] if args.mode == "both" else [args.mode]
    temps = [float(x) for x in args.temps.split(",") if x.strip() != ""]
    label = args.tier or args.model
    summary = {}   # key: f"{mode}@t={temp}" -> metrics

    for temp in temps:
        sample = {"temp": temp, "top_k": args.top_k, "top_p": args.top_p,
                  "repeat_penalty": args.repeat_penalty,
                  "presence_penalty": presence, "seed": args.seed}
        for mode in modes:
            template = PROMPT_QUOTE if mode == "quote" else PROMPT_DIRECT
            print(f"\n=== {label}  [{mode}]  temp={temp} top-k={args.top_k} top-p={args.top_p} "
                  f"rep={args.repeat_penalty} presence={presence} seed={args.seed} ===", flush=True)
            results = []
            for case in cases:
                t0 = time.time()
                try:
                    raw = run_model(args.binary, args.model,
                                    template.format(doc=case["text"]),
                                    args.threads, args.ctx, args.max_tokens,
                                    args.timeout, sample)
                    s = score_case(extract_json_array(raw), case["expected"], mode, case["text"])
                    dt = time.time() - t0
                except subprocess.TimeoutExpired:
                    dt = args.timeout
                    s = {"valid_json": False, "recall": 0.0, "precision": 0.0,
                         "citation_acc": None,
                         "matched": 0, "expected_n": len(case["expected"]),
                         "quote_match": None, "f1": 0.0, "timed_out": True}
                s["id"], s["seconds"] = case["id"], round(dt, 1)
                results.append(s)
                cite = f"{s['citation_acc']:.0%}" if s.get("citation_acc") is not None else " n/a"
                qm = f"  quote_match={s['quote_match']:.0%}" if s.get("quote_match") is not None else ""
                f1 = f"  f1={s['f1']:.0%}" if s['valid_json'] else ""
                flag = "  TIMEOUT" if s.get("timed_out") else ""
                print(f"  {s['id']:<16} json={str(s['valid_json']):<5} recall={s['recall']:>4.0%}  "
                      f"precision={s['precision']:>4.0%}  cite_acc={cite:>4}  "
                      f"({s['matched']}/{s['expected_n']}, {s['seconds']}s){qm}{f1}{flag}",
                      flush=True)

            n = len(results)
            cites = [r["citation_acc"] for r in results if r["citation_acc"] is not None]
            qms = [r["quote_match"] for r in results if r.get("quote_match") is not None]
            summary[f"{mode}@t={temp}"] = {
                "temp": temp, "mode": mode,
                "json": sum(r["valid_json"] for r in results) / n,
                "recall": sum(r["recall"] for r in results) / n,
                "precision": sum(r["precision"] for r in results) / n,
                "cite": (sum(cites) / len(cites)) if cites else None,
                "quote": (sum(qms) / len(qms)) if qms else None,
                "secs": sum(r["seconds"] for r in results) / n,
            }

    print(f"\n{'='*84}\n  {label} >> architecture x temperature\n{'='*84}")
    print(f"  {'mode':<8}{'temp':>6}{'valid_json':>12}{'recall':>10}{'precision':>11}{'cite_acc':>11}{'quote_hit':>11}{'avg_s':>8}")
    print("  " + "-" * 77)
    for key, v in summary.items():
        cite = f"{v['cite']:.0%}" if v["cite"] is not None else "n/a"
        quote = f"{v['quote']:.0%}" if v["quote"] is not None else "-"
        print(f"  {v['mode']:<8}{v['temp']:>6.1f}{v['json']:>12.0%}{v['recall']:>10.0%}{v['precision']:>11.0%}{cite:>11}{quote:>11}{v['secs']:>7.1f}s")

    # Per temperature, direct-vs-quote delta (only when both modes ran at that temp).
    for temp in temps:
        d = summary.get(f"direct@t={temp}")
        q = summary.get(f"quote@t={temp}")
        if d and q and d["cite"] is not None and q["cite"] is not None:
            delta = (q["cite"] - d["cite"]) * 100
            verdict = "quote WINS" if delta > 0 else ("direct WINS" if delta < 0 else "tie")
            print(f"  t={temp}: citation-accuracy delta (quote - direct) = {delta:+.0f} pts  -> {verdict}")


if __name__ == "__main__":
    main()
