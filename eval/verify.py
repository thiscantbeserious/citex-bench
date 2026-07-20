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
import json, os, sys, tempfile, shutil
sys.path.insert(0, os.path.dirname(__file__))
import eval as ev  # noqa: E402

# -------------------------------------------------------------------- helpers

def _base_config():
    """A minimal valid config dict for building error-case variants."""
    return {
        "run": {"threads": 7, "reps": 5, "timeout": 600,
                "modes": ["direct", "quote"]},
        "defaults": {
            "repeat_penalty": 1.0, "top_k": 20, "top_p": 0.9,
            "presence_penalty": 0.0, "strict_schema_validation": True,
            "template": "",
        },
        "models": {
            "prism-ml/Bonsai-1.7B-gguf:Q1_0": [
                {"temp": 0.0, "presence_penalty": 0.5},
                {"temp": 0.5, "presence_penalty": 0.5},
            ],
        },
    }


# Temp config files created by _write_tmp_config. Tracked so self_test can
# clean them all up in a finally block instead of leaking into /tmp.
_tmp_config_paths = []


def _write_tmp_config(cfg_dict):
    """Write a config dict to a temp file, return the path. The path is tracked
    in _tmp_config_paths for cleanup by self_test."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(cfg_dict, f)
    _tmp_config_paths.append(path)
    return path


def _expect_value_error(fn, name=None):
    """Assert that fn() raises ValueError, optionally checking the message."""
    try:
        fn()
    except ValueError as e:
        if name and name not in str(e):
            raise AssertionError(f"ValueError did not name '{name}': {e}")
        return
    raise AssertionError("expected ValueError, none raised")


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

    # ================================================================ Step 2
    # load_config behavior: validation, slug, seed, resolution.
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        # 1. Real config loads, returns 3 models with expected set counts.
        cfg = ev.load_config(config_path)
        assert "run" in cfg and "defaults" in cfg and "cells" in cfg
        assert cfg["run"]["reps"] == 5
        assert cfg["run"]["threads"] == 7
        assert cfg["run"]["modes"] == ["direct", "quote"]
        sets_by_model = {}
        for cell in cfg["cells"]:
            sets_by_model.setdefault(cell["model"], set()).add(cell["slug"])
        assert len(sets_by_model["prism-ml/Bonsai-1.7B-gguf:Q1_0"]) == 4, \
            f"1.7B sets: expected 4, got {len(sets_by_model['prism-ml/Bonsai-1.7B-gguf:Q1_0'])}"
        assert len(sets_by_model["prism-ml/Bonsai-4B-gguf:Q1_0"]) == 3, \
            f"4B sets: expected 3, got {len(sets_by_model['prism-ml/Bonsai-4B-gguf:Q1_0'])}"
        assert len(sets_by_model["prism-ml/Bonsai-8B-gguf:Q1_0"]) == 3, \
            f"8B sets: expected 3, got {len(sets_by_model['prism-ml/Bonsai-8B-gguf:Q1_0'])}"

        # 2. Unknown key in a set raises ValueError.
        bad = _base_config()
        bad["models"]["prism-ml/Bonsai-1.7B-gguf:Q1_0"][0]["temperature"] = 0.5
        _expect_value_error(lambda: ev.load_config(_write_tmp_config(bad)))

        # 3. Missing default raises ValueError naming the key.
        bad_def = _base_config()
        del bad_def["defaults"]["top_k"]
        _expect_value_error(lambda: ev.load_config(_write_tmp_config(bad_def)), name="top_k")

        # 4. Duplicate slug within a model raises ValueError.
        dup = _base_config()
        # Two sets that merge to the same resolved params (both temp 0.5, defaults
        # fill the rest identically) produce the same slug.
        dup["models"]["prism-ml/Bonsai-1.7B-gguf:Q1_0"] = [
            {"temp": 0.5, "top_k": 20},
            {"temp": 0.5},
        ]
        _expect_value_error(lambda: ev.load_config(_write_tmp_config(dup)))

        # 5. Seed assignment: temp 0.5 with reps=3 yields seeds [0,1,2] across reps.
        #    temp 0.0 yields seed 0 for all reps.
        seed_cfg = _base_config()
        seed_cfg["run"]["reps"] = 3
        seed_cfg["run"]["modes"] = ["direct"]
        seed_cfg["models"]["prism-ml/Bonsai-1.7B-gguf:Q1_0"] = [
            {"temp": 0.0, "presence_penalty": 0.5},
            {"temp": 0.5, "presence_penalty": 0.5},
        ]
        sc = ev.load_config(_write_tmp_config(seed_cfg))
        temp05_seeds = sorted(c["seed"] for c in sc["cells"]
                              if abs(c["set"]["temp"] - 0.5) < 1e-9)
        assert temp05_seeds == [0, 1, 2], \
            f"temp 0.5 seeds: expected [0,1,2], got {temp05_seeds}"
        temp0_seeds = [c["seed"] for c in sc["cells"]
                      if abs(c["set"]["temp"]) < 1e-9]
        assert all(s == 0 for s in temp0_seeds), \
            f"temp 0.0 seeds: expected all 0, got {temp0_seeds}"

        # 6. Slug format on one known set.
        slug_cell = next(c for c in cfg["cells"]
                         if c["model"] == "prism-ml/Bonsai-1.7B-gguf:Q1_0"
                         and abs(c["set"]["temp"]) < 1e-9
                         and abs(c["set"]["presence_penalty"] - 0.5) < 1e-9)
        assert slug_cell["slug"] == "t0.0-tk20-tp0.9-rp1.0-pp0.5-s1", \
            f"slug: expected t0.0-tk20-tp0.9-rp1.0-pp0.5-s1, got {slug_cell['slug']}"

        # 7. Model resolution against a temp dir with fake files. The excluded
        #    candidates all CONTAIN the quant (Q1_0) so they pass the substring
        #    filter and reach the exclusion logic. If the exclusion is deleted,
        #    _resolve_model_file would return one of them instead of the valid
        #    file, failing the assertion.
        tmpdir = tempfile.mkdtemp()
        try:
            open(os.path.join(tmpdir, "Bonsai-1.7B-Q1_0.gguf"), "w").close()
            open(os.path.join(tmpdir, "Bonsai-1.7B-Q1_0-F16.gguf"), "w").close()
            open(os.path.join(tmpdir, "Bonsai-1.7B-Q1_0-mmproj.gguf"), "w").close()
            open(os.path.join(tmpdir, "Bonsai-1.7B-Q1_0-PQ2_0.gguf"), "w").close()
            open(os.path.join(tmpdir, "Bonsai-1.7B-Q1_0-BF16.gguf"), "w").close()
            open(os.path.join(tmpdir, "Bonsai-1.7B-Q1_0-Drafter.gguf"), "w").close()
            open(os.path.join(tmpdir, "Bonsai-1.7B-Q1_0-Dspark.gguf"), "w").close()
            rc = ev.load_config(config_path, models_dir=tmpdir)
            cell17 = next(c for c in rc["cells"]
                          if c["model"] == "prism-ml/Bonsai-1.7B-gguf:Q1_0")
            assert cell17["resolved_file"] == "Bonsai-1.7B-Q1_0.gguf", \
                f"resolved_file: expected Bonsai-1.7B-Q1_0.gguf, got {cell17['resolved_file']}"
        finally:
            shutil.rmtree(tmpdir)

        # 8. Per-set template override (Step 3 per-tier decision). The real
        #    config has defaults.template="embedded". 1.7B and 8B sets inherit
        #    it. 4B sets override to "" (raw) because the embedded template
        #    corrupts 4B output.
        for cell in cfg["cells"]:
            model = cell["model"]
            tmpl = cell["set"]["template"]
            if "4B" in model:
                assert tmpl == "", \
                    f"4B set {cell['slug']} template: expected '' (raw override), got {tmpl!r}"
            else:
                assert tmpl == "embedded", \
                    f"{model} set {cell['slug']} template: expected 'embedded' (inherited), got {tmpl!r}"
    finally:
        for p in _tmp_config_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        _tmp_config_paths.clear()

    # ================================================================ Step 4
    # strict_schema_validation wires --json-schema into the llama-completion
    # cmd. We test by calling _build_cmd (the helper run_model delegates to)
    # directly and asserting on the returned list. No subprocess, no model, no
    # GPU. This isolates the flag-to-arg wiring from the model run.
    sample = {"temp": 0.5, "top_k": 20, "top_p": 0.9, "repeat_penalty": 1.0,
              "presence_penalty": 0.5, "seed": 0}

    # strict=True, mode=quote: cmd carries --json-schema with QUOTE_SCHEMA.
    cmd_q = ev._build_cmd("llama-completion", "m.gguf", "prompt", 7, 2048,
                          400, sample, template="",
                          strict_schema_validation=True, mode="quote")
    assert "--json-schema" in cmd_q, "strict quote: --json-schema missing"
    qi = cmd_q.index("--json-schema")
    q_schema = cmd_q[qi + 1]
    assert '"quote"' in q_schema, "strict quote: schema lacks quote field"
    assert '"source_ref"' not in q_schema, \
        "strict quote: schema has source_ref (wrong schema picked)"

    # strict=True, mode=direct: cmd carries --json-schema with DIRECT_SCHEMA.
    cmd_d = ev._build_cmd("llama-completion", "m.gguf", "prompt", 7, 2048,
                          400, sample, template="",
                          strict_schema_validation=True, mode="direct")
    di = cmd_d.index("--json-schema")
    d_schema = cmd_d[di + 1]
    assert '"source_ref"' in d_schema, \
        "strict direct: schema lacks source_ref field"
    assert '"quote"' not in d_schema, \
        "strict direct: schema has quote (wrong schema picked)"

    # strict=False: cmd must NOT carry --json-schema (post-hoc extract path).
    cmd_f = ev._build_cmd("llama-completion", "m.gguf", "prompt", 7, 2048,
                          400, sample, template="",
                          strict_schema_validation=False, mode="quote")
    assert "--json-schema" not in cmd_f, \
        "strict=False: --json-schema should be absent"

    # ================================================================ Step 5
    # Capture round-trip: build_record and capture_path produce valid JSONL
    # that reads back with all required fields. 2 cases x 2 reps = 4 records.
    tmpdir5 = tempfile.mkdtemp()
    try:
        capture_dir = os.path.join(tmpdir5, "captures")
        model = "prism-ml/Bonsai-1.7B-gguf:Q1_0"
        mode = "direct"
        slug = "t0.5-tk20-tp0.9-rp1.0-pp0.5-s1"
        cell_config = {
            "temp": 0.5, "top_k": 20, "top_p": 0.9,
            "repeat_penalty": 1.0, "presence_penalty": 0.5,
            "strict_schema_validation": True, "template": "embedded",
            "seed": 0, "threads": 7,
        }
        cases_5 = [
            {"id": "c1", "text": "doc1 text", "expected": []},
            {"id": "c2", "text": "doc2 text", "expected": []},
        ]
        cap = ev.capture_path(capture_dir, model, mode, slug)
        assert cap.endswith(
            "prism-ml-Bonsai-1.7B-gguf-Q1_0-direct-"
            "t0.5-tk20-tp0.9-rp1.0-pp0.5-s1.jsonl"), \
            f"capture_path: wrong filename, got {cap}"
        os.makedirs(capture_dir, exist_ok=True)
        with open(cap, "w") as f:
            for rep in range(2):
                for case in cases_5:
                    rec = ev.build_record(model, mode, slug, cell_config, case,
                                          rep, "raw_output_here",
                                          {"valid_json": True}, 1.5, True)
                    f.write(json.dumps(rec) + "\n")
        records = [json.loads(l) for l in open(cap) if l.strip()]
        assert len(records) == 4, \
            f"expected 4 records, got {len(records)}"
        required = {"model", "mode", "slug", "config", "case_id", "rep",
                    "raw_output", "expected", "doc", "score", "seconds",
                    "scored"}
        for i, rec in enumerate(records):
            assert required.issubset(rec.keys()), \
                f"record {i} missing fields: {required - rec.keys()}"
            assert "seed" in rec["config"], \
                f"record {i} config missing seed"
            assert "threads" in rec["config"], \
                f"record {i} config missing threads"
            assert rec["scored"] is True
            assert rec["case_id"] in ("c1", "c2")
            assert rec["rep"] in (0, 1)
    finally:
        shutil.rmtree(tmpdir5)

    print("self-test: scorer is deterministic, one-to-one assignment works")
    print("self-test: capture round-trip writes and reads back all required fields")
    print("self-test: load_config validates, resolves, and assigns seeds correctly")
    print("self-test: strict_schema_validation wires --json-schema per mode")
    return True


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if mode == "selftest":
        sys.exit(0 if self_test() else 1)
    print("capture/replay modes need --capture wiring in eval.py; run selftest", file=sys.stderr)
    sys.exit(2)
