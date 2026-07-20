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
import argparse, difflib, json, os, re, subprocess, sys, time, urllib.request, urllib.error

MARKER_RE = re.compile(r"\[\d+\]")

# Step 4 flat JSON schemas for --json-schema. No $ref, all string properties.
# The fork accepts a schema string via --json-schema (confirmed via --help on
# bonsai-floor:prism). Schemas with external $ref need --grammar instead, so
# these are kept flat. Schema constrains JSON syntax, not content: a fabricated
# claim passes as a valid string. It kills the validity failure mode, not the
# hallucination one, so precision stays primary.
DIRECT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "source_ref": {"type": "string"},
        },
    },
}
QUOTE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "quote": {"type": "string"},
        },
    },
}
_MODE_SCHEMAS = {"direct": DIRECT_SCHEMA, "quote": QUOTE_SCHEMA}

# Keys every defaults block must define. A param set may override these plus
# "temp". Nothing else is allowed.
_SAMPLABLE_KEYS = ["repeat_penalty", "top_k", "top_p",
                   "presence_penalty", "strict_schema_validation", "template"]


# Packing suffix after the quant: _g64, _g128, etc. Digits only, per the
# llama.cpp/PrismML naming convention.
_PACK_SUFFIX_RE = re.compile(r"_g\d+$", re.IGNORECASE)

_HF_API_TIMEOUT = 10  # seconds, don't hang the grid if HF is slow or offline


def _list_repo_ggufs(repo):
    """Fetch the list of .gguf filenames from the HuggingFace API for `repo`.
    Mirrors bench.sh's list_ggufs(). Returns a list of filenames, or None on
    network/parse error (offline, rate-limited, repo not found)."""
    url = f"https://huggingface.co/api/models/{repo}"
    try:
        with urllib.request.urlopen(url, timeout=_HF_API_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return [s["rfilename"] for s in data.get("siblings", [])
            if s.get("rfilename", "").endswith(".gguf")]


def _is_excluded_variant(filename_lower):
    """True for non-loadable variant files: mmproj, Dspark/Drafter, F16/BF16,
    PQ2_0. Matches bench.sh's exclusion case statement."""
    return ("mmproj" in filename_lower or "spark" in filename_lower
            or "drafter" in filename_lower or "f16" in filename_lower
            or "bf16" in filename_lower or "pq2_0" in filename_lower)


def _pick_gguf(filenames, quant):
    """Pick the GGUF matching `quant` from a list of filenames, applying
    bench.sh's exclusion and packing preference. The list is assumed to be
    scoped to one repo (so no cross-size collision is possible). Returns the
    filename, or None."""
    quant_l = quant.lower()
    cands = []
    for f in filenames:
        lf = f.lower()
        if quant_l not in lf:
            continue
        if _is_excluded_variant(lf):
            continue
        cands.append(f)
    if not cands:
        return None
    # Prefer g128 (native pack): the file WITHOUT _g64. Fall back to _g64.
    for f in cands:
        if "_g64" not in f.lower():
            return f
    return cands[0]


def _tail_matches_quant(tail, quant_l):
    """True if `tail` (the filename segment between the model-name prefix and
    .gguf, lowercased) is exactly the quant, or the quant plus a packing suffix
    like _g64 / _g128. Strict, not a substring: q1_0 matches "q1_0" and
    "q1_0_g64" but NOT "q1_0-f16" or "q1_0-instruct"."""
    if tail == quant_l:
        return True
    if not tail.startswith(quant_l + "_g"):
        return False
    return bool(_PACK_SUFFIX_RE.search(tail))


def _resolve_model_file(models_dir, repo, quant):
    """Pick the GGUF for `repo` + `quant`.

    Primary: fetch the repo's file list from the HuggingFace API. The list is
    scoped to one repo, so Bonsai-1.7B and Bonsai-8B (different repos) cannot
    collide. This is exactly how bench.sh resolves, and it needs no model-name
    matching. The first grid run (commit d8ef19f) loaded the 1.7B for every
    model key because the loader scanned the local models/ dir (all sizes
    mixed) and matched on quant alone.

    Fallback (offline or API failure): scan the local models_dir with a strict
    model-name prefix + quant-tail match. The prefix (repo basename with -gguf
    stripped, lowercased) rejects cross-size collisions; the strict tail rejects
    variant files (mmproj, F16, etc.) without a separate exclusion list.

    Returns the filename, or None when no match."""
    files = _list_repo_ggufs(repo)
    if files:
        pick = _pick_gguf(files, quant)
        if pick:
            return pick
    if not models_dir or not os.path.isdir(models_dir):
        return None
    model_name = repo.rsplit("/", 1)[-1].lower()
    if model_name.endswith("-gguf"):
        model_name = model_name[:-len("-gguf")]
    prefix = model_name + "-"
    quant_l = quant.lower()
    try:
        entries = sorted(os.listdir(models_dir))
    except OSError:
        return None
    cands = []
    for f in entries:
        if not f.endswith(".gguf"):
            continue
        lf = f.lower()
        if not lf.startswith(prefix):
            continue
        tail = lf[len(prefix):-len(".gguf")]
        if not _tail_matches_quant(tail, quant_l):
            continue
        cands.append(f)
    if not cands:
        return None
    for f in cands:
        if "_g64" not in f.lower():
            return f
    return cands[0]


def _set_slug(s):
    """Readable identity slug for a resolved param set."""
    return (f"t{s['temp']:.1f}"
            f"-tk{s['top_k']}"
            f"-tp{s['top_p']}"
            f"-rp{s['repeat_penalty']}"
            f"-pp{s['presence_penalty']}"
            f"-s{1 if s['strict_schema_validation'] else 0}")


def load_config(path, models_dir=None):
    """Load eval/config.json and resolve it into a structured grid.

    Returns a dict with three keys:

        {
          "run": {"threads", "reps", "timeout", "modes"},
          "defaults": {repeat_penalty, top_k, top_p, presence_penalty,
                      strict_schema_validation, template},
          "cells": [ ... one per (model, mode, set, rep) ... ]
        }

    Each cell is:

        {
          "model": "repo:quant",      # the HF model key as written in config
          "repo": "prism-ml/...",     # HF repo, split on the LAST colon
          "quant": "Q1_0",            # quant suffix
          "resolved_file": "file.gguf" or None,
          "mode": "direct"|"quote",
          "rep": 0,                    # 0..N-1 where N = run.reps
          "seed": 0,                   # temp>0 -> rep index, temp==0 -> 0
          "slug": "t0.0-tk20-...-s1", # unique per set within a model
          "set": {temp, repeat_penalty, top_k, top_p,
                  presence_penalty, strict_schema_validation, template},
        }

    Validation:
      - defaults must define every key in _SAMPLABLE_KEYS, else ValueError.
      - a param set may override only _SAMPLABLE_KEYS plus "temp". Unknown key
        -> ValueError naming the key and model.
      - duplicate slug within a model -> ValueError.

    Model resolution: when models_dir is given, resolve the quant to a local
    GGUF file by filename substring with packing preference (g128 over g64,
    PQ2_0/mmproj/F16/BF16 excluded). When models_dir is None or no file matches,
    resolved_file is None (no error).
    """
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot load config from {path}: {e}")

    run = cfg["run"]
    defaults = cfg["defaults"]
    reps = run["reps"]
    modes = run["modes"]

    # Validate defaults: every samplable key must be present.
    for k in _SAMPLABLE_KEYS:
        if k not in defaults:
            raise ValueError(f"defaults missing required key: {k}")

    cells = []
    for model_key, sets in cfg["models"].items():
        # Split on the LAST colon to separate repo and quant.
        idx = model_key.rfind(":")
        if idx < 0:
            raise ValueError(f"model key missing ':quant' suffix: {model_key}")
        repo = model_key[:idx]
        quant = model_key[idx + 1:]
        resolved_file = _resolve_model_file(models_dir, repo, quant)

        seen_slugs = set()
        for s in sets:
            # Validate: only _SAMPLABLE_KEYS + "temp" allowed in a set.
            for k in s:
                if k != "temp" and k not in defaults:
                    raise ValueError(
                        f"unknown key '{k}' in param set for model {model_key}")
            # Merge: start from defaults, apply overrides, add temp.
            merged = dict(defaults)
            merged.update({k: v for k, v in s.items() if k != "temp"})
            merged["temp"] = s["temp"]
            slug = _set_slug(merged)
            if slug in seen_slugs:
                raise ValueError(
                    f"duplicate slug '{slug}' in model {model_key}")
            seen_slugs.add(slug)
            for mode in modes:
                for rep in range(reps):
                    seed = rep if merged["temp"] > 0 else 0
                    cells.append({
                        "model": model_key,
                        "repo": repo,
                        "quant": quant,
                        "resolved_file": resolved_file,
                        "mode": mode,
                        "rep": rep,
                        "seed": seed,
                        "slug": slug,
                        "set": merged,
                    })

    return {"run": run, "defaults": defaults, "cells": cells}

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


def _build_cmd(binary, model, prompt, threads, ctx, max_tokens, sample,
              template="", strict_schema_validation=True, mode=None):
    """Build the llama-completion argv list. Does not run the model. Extracted
    from run_model so verify.py can assert on the cmd without a GPU.

    `strict_schema_validation` (default True, matching config): when True, pass
    --json-schema with the flat schema for `mode`. When False, no constraint,
    post-hoc extract as today. The caller passes `mode` ("direct" or "quote")
    so this helper can pick the right schema from _MODE_SCHEMAS. Passing the
    mode is cleaner than passing the schema dict: the schemas are module-level
    constants keyed by mode, and main already knows the mode.

    See run_model for `sample` and `template` docs."""
    cmd = [binary, "-m", model, "-p", prompt, "-n", str(max_tokens),
           "-t", str(threads), "-c", str(ctx), "-ngl", "0",
           "--temp", str(sample["temp"]),
           "--top-k", str(sample["top_k"]),
           "--top-p", str(sample["top_p"]),
           "--repeat-penalty", str(sample["repeat_penalty"]),
           "--presence-penalty", str(sample["presence_penalty"]),
           "--seed", str(sample["seed"])]
    if template == "embedded":
        cmd += ["--jinja", "-cnv"]
    elif template:
        cmd += ["--jinja", "--chat-template-file", template, "-cnv"]
    if strict_schema_validation:
        if mode not in _MODE_SCHEMAS:
            raise ValueError(
                f"strict_schema_validation=True requires mode in "
                f"{list(_MODE_SCHEMAS)}, got {mode!r}")
        cmd += ["--json-schema", json.dumps(_MODE_SCHEMAS[mode])]
    return cmd


# Timing lines emitted by llama-completion on stderr (common_perf_print).
# prompt eval time = prefill; eval time = decode (generation). The decode TPS
# is the number users mean by "how fast does the model generate". The regexes
# are lenient on whitespace (the fork pads numbers variably) and never raise:
# any parse failure yields None for that field, so a format change in a future
# fork build degrades to missing TPS, never a crash.
_PROMPT_TIMING_RE = re.compile(
    r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens?\s*"
    r"\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
_DECODE_TIMING_RE = re.compile(
    r"\beval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*runs?\s*"
    r"\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)")


def _parse_timing(stderr):
    """Parse llama-completion's common_perf_print lines from stderr into a
    timing dict: {prefill_tps, prompt_tokens, decode_tps, decode_tokens}.

    Never raises. Any field not found is None, so a stderr format change or a
    truncated capture degrades to missing TPS rather than crashing the grid.
    The decode TPS is the generation throughput, the primary speed metric for
    the per-case line and summary."""
    # Typed so the float/int reassignments below do not trip the type checker
    # narrowing the dict values to None.
    result: dict[str, float | int | None] = {
        "prefill_tps": None, "prompt_tokens": None,
        "decode_tps": None, "decode_tokens": None,
    }
    if not stderr:
        return result
    try:
        m = _PROMPT_TIMING_RE.search(stderr)
        if m:
            result["prompt_tokens"] = int(m.group(1))
            result["prefill_tps"] = float(m.group(2))
        m = _DECODE_TIMING_RE.search(stderr)
        if m:
            result["decode_tokens"] = int(m.group(1))
            result["decode_tps"] = float(m.group(2))
    except (ValueError, AttributeError):
        # int/float conversion or match group access failed; keep the Nones.
        pass
    return result


def run_model(binary, model, prompt, threads, ctx, max_tokens, timeout, sample,
             template="", strict_schema_validation=True, mode=None):
    """One-shot completion. Uses llama-completion, NOT llama-cli: in this fork
    llama-cli is an interactive REPL that rejects -no-cnv and spins forever on
    EOF stdin (~1GB of "> " prompts, then OOM).

    `sample` carries the full Bonsai-recommended sampling config, because the
    model card tunes temperature together with top-k/top-p/penalties, sweeping
    temperature in isolation at temp>0 is off-spec. At temp=0 (greedy) the
    sampler short-circuits and top-k/top-p are inert, which is fine.

    `template` controls chat-template application:
      ""        raw -p prompt, no template (current behavior).
      "embedded"  apply the model's embedded chat template via --jinja -cnv.
                  --jinja enables the Jinja engine, -cnv enables conversation
                  mode so -p is treated as a user turn with add_generation_prompt.
      <path>    use a custom jinja template file via --jinja --chat-template-file
                <path> -cnv.
    `strict_schema_validation` (default True): when True, pass --json-schema
      with the flat schema for `mode` (direct or quote). When False, no
      constraint, post-hoc extract as today. Schema constrains JSON syntax,
      not content. See _MODE_SCHEMAS.
    `mode` ("direct" or "quote"): selects the schema when strict is True.
      Required when strict_schema_validation is True.
    Flags confirmed via `llama-completion --help` on bonsai-floor:prism."""
    cmd = _build_cmd(binary, model, prompt, threads, ctx, max_tokens, sample,
                    template, strict_schema_validation, mode)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=timeout, stdin=subprocess.DEVNULL)
    # Parse real TPS from llama-completion's stderr (common_perf_print). The
    # decode TPS is the generation throughput, the primary speed metric.
    timing = _parse_timing(r.stderr)
    return r.stdout, timing


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


def capture_path(capture_dir, model, mode, slug):
    """File path for the JSONL capture of one (model, mode, slug) group.
    The model key (repo:quant) has / and : replaced by -."""
    safe = model.replace("/", "-").replace(":", "-")
    return os.path.join(capture_dir, f"{safe}-{mode}-{slug}.jsonl")


def build_record(model, mode, slug, cell_config, case, rep, raw_output, score,
                 seconds, scored, timing=None):
    """Construct one capture record dict. cell_config carries temp, top_k,
    top_p, repeat_penalty, presence_penalty, strict_schema_validation,
    template, seed, threads. timing carries prefill_tps, prompt_tokens,
    decode_tps, decode_tokens parsed from llama-completion's stderr (None when
    not captured, e.g. a timeout)."""
    return {
        "model": model,
        "mode": mode,
        "slug": slug,
        "config": dict(cell_config),
        "case_id": case["id"],
        "rep": rep,
        "raw_output": raw_output,
        "expected": case["expected"],
        "doc": case["text"],
        "score": score,
        "seconds": seconds,
        "scored": scored,
        "timing": timing or {},
    }


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
    ap.add_argument("--config", default="eval/config.json",
                   help="path to the config file")
    ap.add_argument("--models-dir", default="models",
                   help="directory to resolve model files from")
    ap.add_argument("--cases", default="eval/cases.jsonl",
                   help="the cases file")
    ap.add_argument("--binary", default="llama-completion")
    ap.add_argument("--ctx", default="2048")
    ap.add_argument("--max-tokens", default="400")
    ap.add_argument("--timeout", type=int, default=None,
                   help="override run.timeout from config")
    ap.add_argument("--capture-dir", default="reports/captures",
                   help="where to write JSONL capture files")
    args = ap.parse_args()

    cfg = load_config(args.config, models_dir=args.models_dir)
    cells = cfg["cells"]
    threads = cfg["run"]["threads"]
    timeout = args.timeout if args.timeout is not None else cfg["run"]["timeout"]

    with open(args.cases) as f:
        cases = [json.loads(l) for l in f if l.strip()]

    os.makedirs(args.capture_dir, exist_ok=True)

    for cell in cells:
        model_key = cell["model"]
        resolved = cell["resolved_file"]
        if resolved is None:
            print(f"  {model_key}: no resolved model file, skipping", flush=True)
            continue
        model_path = os.path.join(args.models_dir, resolved)
        mode = cell["mode"]
        slug = cell["slug"]
        rep = cell["rep"]
        seed = cell["seed"]
        s = cell["set"]
        sample = dict(s)
        sample["seed"] = seed
        cell_config = {
            "temp": s["temp"],
            "top_k": s["top_k"],
            "top_p": s["top_p"],
            "repeat_penalty": s["repeat_penalty"],
            "presence_penalty": s["presence_penalty"],
            "strict_schema_validation": s["strict_schema_validation"],
            "template": s["template"],
            "seed": seed,
            "threads": threads,
        }
        template = PROMPT_QUOTE if mode == "quote" else PROMPT_DIRECT
        print(f"\n=== {model_key} [{mode}] {slug} rep={rep} ===", flush=True)
        cap = capture_path(args.capture_dir, model_key, mode, slug)
        with open(cap, "a") as cf:
            for case in cases:
                t0 = time.time()
                timing = {}
                try:
                    prompt = template.format(doc=case["text"])
                    raw, timing = run_model(args.binary, model_path, prompt,
                                    threads, args.ctx, args.max_tokens,
                                    timeout, sample,
                                    template=s["template"],
                                    strict_schema_validation=s["strict_schema_validation"],
                                    mode=mode)
                    score = score_case(extract_json_array(raw),
                                       case["expected"], mode, case["text"])
                    dt = round(time.time() - t0, 1)
                    scored = True
                except subprocess.TimeoutExpired:
                    dt = timeout
                    raw = ""
                    score = {"valid_json": False, "timed_out": True,
                             "recall": 0.0, "precision": 0.0,
                             "citation_acc": None, "matched": 0,
                             "expected_n": len(case["expected"]),
                             "quote_match": None, "f1": 0.0}
                    scored = False
                rec = build_record(model_key, mode, slug, cell_config, case,
                                   rep, raw, score, dt, scored, timing)
                cf.write(json.dumps(rec) + "\n")
                cf.flush()
                cite = f"{score['citation_acc']:.0%}" if score.get("citation_acc") is not None else " n/a"
                qm = f"  quote_match={score['quote_match']:.0%}" if score.get("quote_match") is not None else ""
                f1 = f"  f1={score['f1']:.0%}" if score['valid_json'] else ""
                tps = timing.get("decode_tps")
                tps_s = f"  {tps:.1f} tok/s" if tps is not None else ""
                flag = "  TIMEOUT" if not scored else ""
                print(f"  {case['id']:<16} json={str(score['valid_json']):<5} recall={score['recall']:>4.0%}  "
                      f"precision={score['precision']:>4.0%}  cite_acc={cite:>4}  "
                      f"({score['matched']}/{score['expected_n']}, {dt}s){qm}{tps_s}{f1}{flag}",
                      flush=True)

    # Step 6: deterministic replay. verify.py imports eval (import eval as ev),
    # so importing verify here would create a circular import. Use a subprocess
    # call instead: python3 eval/verify.py --replay <capture_dir>.
    verify_script = os.path.join(os.path.dirname(__file__), "verify.py")
    replay_cmd = [sys.executable, verify_script, "--replay", args.capture_dir]
    replay_result = subprocess.run(replay_cmd)
    if replay_result.returncode != 0:
        print("replay failed, exiting nonzero", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
