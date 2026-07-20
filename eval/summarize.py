#!/usr/bin/env python3
"""Summarize accuracy eval captures into a readable report.

Reads JSONL capture files (one per model/mode/slug, written by eval.py) and
emits a compact report:

  1. One summary table: one row per (model, mode, slug) with the decision-grade
     metrics (recall, precision, citation accuracy), mean latency, mean decode
     TPS, rep count, and across-case std (stability). Scannable in one screen.
  2. A determinism section: for greedy (temp 0) sets, whether reps reproduce
     identical raw_output. Identical recorded config with divergence is a
     finding.
  3. An appendix: per-case detail (within-case variance across reps, per-case
     latency, per-case TPS) for readers who need the breakdown.

Precision and recall are primary (fabrication is the failure class). F1 is a
labeled non-decision-grade convenience, reported in the appendix only.

Mirrors the speed bench's root-level summarize.py: decoupled from the runner
over the capture format. The runner writes captures, this reads them.

Usage:
    python3 eval/summarize.py <captures-dir>
"""
import glob, json, os, statistics, sys
from collections import defaultdict


def load_records(captures_dir):
    """Load all .jsonl records from captures_dir. Returns a list of dicts, or
    None on error (missing dir, unreadable file, malformed JSON)."""
    if not os.path.isdir(captures_dir):
        return None
    files = sorted(glob.glob(os.path.join(captures_dir, "*.jsonl")))
    records = []
    for fpath in files:
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as e:
            print(f"summarize: cannot read {fpath}: {e}", file=sys.stderr)
            return None
    return records


def _mean(values):
    return statistics.mean(values) if values else 0.0


def _std(values):
    """Sample std. Returns 0.0 for single-element or empty lists, because
    one rep has no variance to measure."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _short_model(model_key):
    """prism-ml/Bonsai-8B-gguf:Q1_0 -> 8B. Compact for the summary table."""
    base = model_key.rsplit("/", 1)[-1]
    base = base.replace("-gguf", "").replace("-Bonsai", "Bonsai")
    # Pull the size token (1.7B, 4B, 8B, 27B) out of the basename.
    for tok in ("1.7B", "4B", "8B", "27B"):
        if tok in base:
            return tok
    return base


def is_greedy(record):
    """A greedy set: slug starts with t0.0- or config temp == 0.0."""
    if record.get("slug", "").startswith("t0.0-"):
        return True
    cfg = record.get("config", {})
    return cfg.get("temp") == 0.0


def _fmt_pct(x):
    """Format a 0..1 fraction as a percentage, or 'n/a' for None."""
    return f"{x*100:.0f}%" if x is not None else "n/a"


def summarize(captures_dir):
    """Read captures and print the report. Returns 0 on success, 1 on error."""
    records = load_records(captures_dir)
    if records is None:
        print(f"summarize: no captures found in {captures_dir}",
              file=sys.stderr)
        return 1
    if not records:
        print(f"summarize: no records in {captures_dir}", file=sys.stderr)
        return 1

    # Group records by (model, mode, slug).
    groups = defaultdict(list)
    for r in records:
        key = (r["model"], r["mode"], r["slug"])
        groups[key].append(r)

    print(f"\n{'='*92}")
    print(f"  ACCURACY EVAL SUMMARY")
    print(f"  captures: {captures_dir}")
    print(f"  records: {len(records)}, groups: {len(groups)}")
    print(f"  metrics: recall and precision primary (fabrication is the failure class), F1 non-decision-grade")
    print(f"{'='*92}")

    # ---- Summary table: one row per (model, mode, slug) ----
    header = (f"  {'model':<6}{'mode':<8}{'slug':<40}"
              f"{'recall':>8}{'prec':>8}{'cite':>8}"
              f"{'latency':>9}{'tps':>9}{'reps':>6}{'std':>7}")
    print(header)
    print(f"  {'-'*88}")

    greedy_groups = []  # (model, mode, slug, by_case) for the determinism section
    appendix_rows = []  # per-case detail rows for the appendix

    for (model, mode, slug) in sorted(groups):
        recs = groups[(model, mode, slug)]
        by_case = defaultdict(list)
        for r in recs:
            by_case[r["case_id"]].append(r)
        for cid in by_case:
            by_case[cid].sort(key=lambda r: r["rep"])

        # Per-case means, then aggregate across cases.
        case_recalls, case_precisions, case_latencies, case_tps = [], [], [], []
        for cid in sorted(by_case):
            reps = by_case[cid]
            recalls = [r["score"].get("recall", 0.0) for r in reps]
            precisions = [r["score"].get("precision", 0.0) for r in reps]
            secs = [r.get("seconds", 0.0) for r in reps]
            tps_vals = [r.get("timing", {}).get("decode_tps")
                        for r in reps if r.get("timing", {}).get("decode_tps") is not None]
            case_recalls.append(_mean(recalls))
            case_precisions.append(_mean(precisions))
            case_latencies.append(_mean(secs))
            case_tps.append(_mean(tps_vals) if tps_vals else 0.0)
            appendix_rows.append((model, mode, slug, cid, reps,
                                  _mean(recalls), _std(recalls),
                                  _mean(precisions), _std(precisions),
                                  _mean(secs), _mean(tps_vals) if tps_vals else None,
                                  len(reps)))

        recall_mean = _mean(case_recalls)
        precision_mean = _mean(case_precisions)
        # Citation accuracy: mean over records that have a non-None cite_acc.
        cites = [r["score"].get("citation_acc") for r in recs
                 if r["score"].get("citation_acc") is not None]
        cite_mean = _mean(cites) if cites else None
        latency_mean = _mean(case_latencies)
        tps_mean = _mean(case_tps) if case_tps else 0.0
        reps_count = len(recs) // len(by_case) if by_case else 0
        # Across-case std of recall (stability signal).
        recall_std = _std(case_recalls)

        print(f"  {_short_model(model):<6}{mode:<8}{slug:<40}"
              f"{recall_mean:>7.2f}{precision_mean:>8.2f}{_fmt_pct(cite_mean):>8}"
              f"{latency_mean:>7.1f}s{tps_mean:>8.1f}{reps_count:>6}{recall_std:>7.2f}")

        if is_greedy(recs[0]):
            greedy_groups.append((model, mode, slug, by_case))

    # ---- Determinism section ----
    print(f"\n  DETERMINISM (greedy / temp 0 sets)")
    if not greedy_groups:
        print(f"  no greedy sets found, skipping.")
    else:
        divergences = 0
        total = 0
        for (model, mode, slug, by_case) in greedy_groups:
            for cid in sorted(by_case):
                reps = by_case[cid]
                outputs = [r.get("raw_output", "") for r in reps]
                total += 1
                if len(set(outputs)) != 1:
                    divergences += 1
                    print(f"  NO  {_short_model(model)} {mode} {slug} {cid}: "
                          f"reps diverge")
        if divergences == 0:
            print(f"  all {total} greedy cells deterministic across reps "
                  f"({len(greedy_groups)} sets). 0 divergences.")
        else:
            print(f"  {divergences}/{total} greedy cells diverged. "
                  f"Identical config with divergence is a finding.")

    # ---- Appendix: per-case detail ----
    print(f"\n  APPENDIX: per-case detail (within-case variance across reps)")
    print(f"  {'model':<6}{'mode':<8}{'slug':<34}{'case':<16}"
          f"{'rec':>6}{'rec_std':>9}{'prec':>6}{'prec_std':>10}"
          f"{'lat':>6}{'tps':>7}{'reps':>5}")
    print(f"  {'-'*100}")
    for (model, mode, slug, cid, reps, r_mean, r_std, p_mean, p_std,
         lat, tps, n) in appendix_rows:
        tps_s = f"{tps:.1f}" if tps is not None else "-"
        print(f"  {_short_model(model):<6}{mode:<8}{slug:<34}{cid:<16}"
              f"{r_mean:>5.2f}{r_std:>9.2f}{p_mean:>5.2f}{p_std:>10.2f}"
              f"{lat:>5.1f}s{tps_s:>7}{n:>5}")

    print()
    return 0


def main():
    if len(sys.argv) != 2:
        print("usage: summarize.py <captures-dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(summarize(sys.argv[1]))


if __name__ == "__main__":
    main()
