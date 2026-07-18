#!/usr/bin/env python3
"""Deterministic re-verification of the eval's scoring logic.

Inputs are plain text on stdin: one record per line,
    <raw_model_output>\t<expected_json_or_empty>\t<doc_text...>
or read from a capture file produced by --capture mode of eval.py.

It replays extract_json_array() + score_case() against raw outputs and asserts
the harness would score them identically. This is the "reverify deterministically,
not through judgement" path: feed the captured raw model output back through the
exact scoring code and check the numbers match the logged summary.

Usage:
    python3 verify.py --capture /tmp/captures   # eval.py --capture writes here
    python3 verify.py --replay reports/accuracy-arch-temp.log

A real implementation would record raw I/O during the eval. For now this is a
skeleton asserting the scorer is deterministic: same raw output -> same score.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import eval as ev  # noqa: E402


def self_test():
    """The scorer must be a pure function of (raw_output, expected, doc).
    Re-scoring the same input twice must give identical results."""
    doc = "Per a report [1], X happened. Per a study [2], Y happened."
    raw = '[{"claim":"X happened","source_ref":"[1]"},{"claim":"Y","source_ref":"[2]"}]'
    expected = [{"source_ref": "[1]", "key_phrase": "X happened"},
                {"source_ref": "[2]", "key_phrase": "Y"}]
    pred = ev.extract_json_array(raw)
    s1 = ev.score_case(pred, expected, "direct", doc)
    s2 = ev.score_case(ev.extract_json_array(raw), expected, "direct", doc)
    assert s1 == s2, "scorer is non-deterministic"
    assert s1["valid_json"] is True and s1["recall"] == 1.0
    print("self-test: scorer is deterministic, scores known input correctly")
    return True


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if mode == "selftest":
        sys.exit(0 if self_test() else 1)
    print("capture/replay modes need --capture wiring in eval.py; run selftest", file=sys.stderr)
    sys.exit(2)
