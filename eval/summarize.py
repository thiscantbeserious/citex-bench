#!/usr/bin/env python3
"""Summarize accuracy eval captures into a readable report.

Reads JSONL capture files (one per model/mode/slug, written by eval.py) and
computes:

  - Two-level variance: within-case (across reps) then across cases.
    Level 1 measures within-case stability. Level 2 measures cross-case
    difficulty. Reported separately, per the plan.
  - Per-case latency: mean seconds per case_id, answering the scout's
    one-slow-case puzzle.
  - Precision and recall as primary metrics. F1 is reported but labeled
    non-decision-grade, no conclusion drawn from it alone.
  - Determinism side section: for greedy (temp 0) sets, checks whether reps
    reproduce identical raw_output. Identical recorded config with divergence
    is a finding.

Mirrors the speed bench's root-level summarize.py: decoupled from the runner
over the capture format. The runner writes captures, this reads them.

Usage:
    python3 eval/summarize.py <captures-dir>
"""
import glob, json, os, statistics, sys
from collections import defaultdict


def load_records(captures_dir):
    """Load all .jsonl records from captures_dir. Returns a list of dicts."""
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


def is_greedy(record):
    """A greedy set: slug starts with t0.0- or config temp == 0.0."""
    if record.get("slug", "").startswith("t0.0-"):
        return True
    cfg = record.get("config", {})
    return cfg.get("temp") == 0.0


def summarize(captures_dir):
    """Read captures and print the report. Returns 0 on success, 1 on error."""
    records = load_records(captures_dir)
    if not records:
        print(f"summarize: no captures found in {captures_dir}",
              file=sys.stderr)
        return 1

    # Group records by (model, mode, slug).
    groups = defaultdict(list)
    for r in records:
        key = (r["model"], r["mode"], r["slug"])
        groups[key].append(r)

    print(f"\n{'='*84}")
    print(f"  ACCURACY EVAL SUMMARY")
    print(f"  captures: {captures_dir}")
    print(f"  records: {len(records)}, groups: {len(groups)}")
    print(f"{'='*84}")

    has_greedy = False

    for (model, mode, slug) in sorted(groups):
        recs = groups[(model, mode, slug)]
        print(f"\n--- {model} / {mode} / {slug} ---")

        # Group by case_id within this group, sort reps.
        by_case = defaultdict(list)
        for r in recs:
            by_case[r["case_id"]].append(r)
        for cid in by_case:
            by_case[cid].sort(key=lambda r: r["rep"])

        # --- Level 1: within-case stability (across reps) ---
        print(f"\n  LEVEL 1: within-case stability (mean and std across reps)")
        hdr1 = (f"  {'case_id':<16}{'recall_mean':>12}{'recall_std':>12}"
                f"{'precision_mean':>16}{'precision_std':>16}")
        print(hdr1)
        print(f"  {'-'*70}")
        case_recall_means = []
        case_precision_means = []
        case_latency = {}
        for cid in sorted(by_case):
            reps = by_case[cid]
            recalls = [r["score"].get("recall", 0.0) for r in reps]
            precisions = [r["score"].get("precision", 0.0) for r in reps]
            r_mean = _mean(recalls)
            r_std = _std(recalls)
            p_mean = _mean(precisions)
            p_std = _std(precisions)
            case_recall_means.append(r_mean)
            case_precision_means.append(p_mean)
            secs = [r.get("seconds", 0.0) for r in reps]
            case_latency[cid] = _mean(secs)
            print(f"  {cid:<16}{r_mean:>12.2f}{r_std:>12.2f}"
                  f"{p_mean:>16.2f}{p_std:>16.2f}")

        # --- Level 2: across-cases difficulty ---
        print(f"\n  LEVEL 2: across-cases difficulty (mean and std of per-case means)")
        hdr2 = (f"  {'recall_mean':>12}{'recall_std':>12}"
                f"{'precision_mean':>16}{'precision_std':>16}")
        print(hdr2)
        print(f"  {'-'*56}")
        r2_mean = _mean(case_recall_means)
        r2_std = _std(case_recall_means)
        p2_mean = _mean(case_precision_means)
        p2_std = _std(case_precision_means)
        print(f"  {r2_mean:>12.2f}{r2_std:>12.2f}"
              f"{p2_mean:>16.2f}{p2_std:>16.2f}")

        # --- F1 (non-decision-grade) ---
        f1s = []
        for cid in sorted(by_case):
            for r in by_case[cid]:
                f1s.append(r["score"].get("f1", 0.0))
        print(f"\n  F1 (non-decision-grade): mean={_mean(f1s):.2f}")

        # --- Per-case latency ---
        print(f"\n  PER-CASE LATENCY (mean seconds across reps):")
        hdr3 = f"  {'case_id':<16}{'mean_seconds':>14}"
        print(hdr3)
        print(f"  {'-'*30}")
        for cid in sorted(by_case):
            print(f"  {cid:<16}{case_latency[cid]:>14.1f}")

        # --- Determinism (greedy sets only) ---
        if is_greedy(recs[0]):
            has_greedy = True
            print(f"\n  DETERMINISM (greedy / temp 0):")
            hdr4 = f"  {'case_id':<16}{'deterministic':>14}"
            print(hdr4)
            print(f"  {'-'*30}")
            for cid in sorted(by_case):
                reps = by_case[cid]
                outputs = [r.get("raw_output", "") for r in reps]
                det = len(set(outputs)) == 1
                label = "yes" if det else "NO"
                print(f"  {cid:<16}{label:>14}")
                if not det:
                    print(f"    diverging cell: {model} / {mode} / "
                          f"{slug} / {cid}")

    if not has_greedy:
        print(f"\n  DETERMINISM: no greedy (temp 0) sets found, skipping.")

    print()
    return 0


def main():
    if len(sys.argv) != 2:
        print("usage: summarize.py <captures-dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(summarize(sys.argv[1]))


if __name__ == "__main__":
    main()
