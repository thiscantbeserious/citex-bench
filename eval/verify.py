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
    raw = '[{"claim":"X happened","source_ref":"[1]"},{"claim":"Y happened","source_ref":"[2]"}]'
    expected = [{"source_ref": "[1]", "key_phrase": "X happened"},
                {"source_ref": "[2]", "key_phrase": "Y happened"}]
    pred = ev.extract_json_array(raw)
    s1 = ev.score_case(pred, expected, "direct", doc)
    s2 = ev.score_case(ev.extract_json_array(raw), expected, "direct", doc)
    assert s1 == s2, "scorer is non-deterministic"
    assert s1["valid_json"] is True and s1["recall"] == 1.0

    # Hallucination fixture: 2 expected, 5 predictions (2 correct, 3 fabricated).
    # Greedy one-to-one assigns 2, precision penalizes the 3 fabrications.
    raw_h = ('[{"claim":"X happened","source_ref":"[1]"},'
             '{"claim":"Y happened","source_ref":"[2]"},'
             '{"claim":"The sky is purple","source_ref":"[3]"},'
             '{"claim":"Dinosaurs built spaceships","source_ref":"[4]"},'
             '{"claim":"Water flows uphill","source_ref":"[5]"}]')
    pred_h = ev.extract_json_array(raw_h)
    sh = ev.score_case(pred_h, expected, "direct", doc)
    _p = sh.get("precision")
    assert _p is not None and abs(_p - 0.4) < 1e-9, \
        f"hallucination precision: expected 0.4, got {_p}"
    assert abs(sh["recall"] - 1.0) < 1e-9, \
        f"hallucination recall: expected 1.0, got {sh['recall']}"

    # Wrong-ref decoy: 1 prediction contains both key phrases but carries one
    # ref. One-to-one assignment uses it once, the other expected stays unmatched.
    expected_d = [{"source_ref": "[1]", "key_phrase": "alpha"},
                  {"source_ref": "[2]", "key_phrase": "beta"}]
    raw_d = '[{"claim":"alpha and beta together","source_ref":"[1]"}]'
    pred_d = ev.extract_json_array(raw_d)
    sd = ev.score_case(pred_d, expected_d, "direct", doc)
    assert abs(sd["recall"] - 0.5) < 1e-9, \
        f"decoy recall: expected 0.5, got {sd['recall']}"
    _p = sd.get("precision")
    assert _p is not None and abs(_p - 1.0) < 1e-9, \
        f"decoy precision: expected 1.0, got {_p}"

    print("self-test: scorer is deterministic, one-to-one assignment works")
    return True


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if mode == "selftest":
        sys.exit(0 if self_test() else 1)
    print("capture/replay modes need --capture wiring in eval.py; run selftest", file=sys.stderr)
    sys.exit(2)
