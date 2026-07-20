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
import json, os, subprocess, sys, tempfile, shutil
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


def replay(captures_dir):
    """Deterministic replay: re-run extract_json_array + score_case on every
    captured record and assert the recomputed score equals the stored score.

    For scored=true records: exact dict equality. score_case is a pure function
    of (pred, expected, mode, doc), so the recomputed dict must be identical to
    the stored dict. No float tolerance is needed because the same arithmetic
    produces the same float bits. If a future metric introduces nondeterminism,
    switch to a tolerance comparison and document why.

    For scored=false (timeout) records: skip the equality check, assert only
    that score["timed_out"] is present and true.

    On mismatch: name the (model, mode, slug, case_id, rep) cell and the field
    that differs. Print to stderr. Return False.

    On success: print a summary and return True.
    """
    import glob
    if not os.path.isdir(captures_dir):
        print(f"replay: capture dir does not exist: {captures_dir}",
              file=sys.stderr)
        return False
    files = sorted(glob.glob(os.path.join(captures_dir, "*.jsonl")))
    n_files = len(files)
    if n_files == 0:
        print(f"replay: no .jsonl captures found in {captures_dir} "
              f"(zero captures means the grid run produced nothing to verify)",
              file=sys.stderr)
        return False
    total_records = 0
    scored_verified = 0
    timeout_checked = 0

    for fpath in files:
        with open(fpath) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"replay: malformed JSON in {fpath} line {lineno}: {e}",
                          file=sys.stderr)
                    return False
                total_records += 1
                model = record.get("model", "?")
                mode = record.get("mode", "?")
                slug = record.get("slug", "?")
                case_id = record.get("case_id", "?")
                rep = record.get("rep", "?")
                cell = (f"model={model}, mode={mode}, slug={slug}, "
                        f"case_id={case_id}, rep={rep}")
                stored_score = record["score"]
                if record.get("scored"):
                    pred = ev.extract_json_array(record["raw_output"])
                    recomputed = ev.score_case(
                        pred, record["expected"],
                        record["mode"], record["doc"])
                    if recomputed != stored_score:
                        all_keys = sorted(set(list(recomputed.keys())
                                             + list(stored_score.keys())))
                        diff_field = "(unknown)"
                        for k in all_keys:
                            if recomputed.get(k) != stored_score.get(k):
                                diff_field = k
                                break
                        print(f"REPLAY MISMATCH: {cell}", file=sys.stderr)
                        print(f"  field '{diff_field}': "
                              f"stored={stored_score.get(diff_field)!r} "
                              f"recomputed={recomputed.get(diff_field)!r}",
                              file=sys.stderr)
                        return False
                    scored_verified += 1
                else:
                    if not stored_score.get("timed_out"):
                        print(f"REPLAY MISMATCH: {cell}", file=sys.stderr)
                        print("  scored=false but score['timed_out'] "
                              "is not true", file=sys.stderr)
                        return False
                    timeout_checked += 1

    print(f"replay: {n_files} files, {total_records} records, "
          f"{scored_verified} scored verified, {timeout_checked} timeout checked")
    return True


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

        # 7b. Regression: multiple same-quant different-size files must resolve
        #     to the RIGHT size, not the first alphabetically. The first grid
        #     run (commit d8ef19f) silently loaded Bonsai-1.7B-Q1_0.gguf for
        #     every model key because the resolver matched on quant alone. This
        #     test fails against the old resolver: all three resolve to 1.7B.
        tmpdir2 = tempfile.mkdtemp()
        try:
            for name in ("Bonsai-1.7B-Q1_0.gguf", "Bonsai-4B-Q1_0.gguf",
                         "Bonsai-8B-Q1_0.gguf"):
                open(os.path.join(tmpdir2, name), "w").close()
            rc2 = ev.load_config(config_path, models_dir=tmpdir2)
            for cell in rc2["cells"]:
                if cell["rep"] != 0 or cell["mode"] != "direct":
                    continue
                model = cell["model"]
                got = cell["resolved_file"]
                if "1.7B" in model:
                    expected = "Bonsai-1.7B-Q1_0.gguf"
                elif "4B" in model:
                    expected = "Bonsai-4B-Q1_0.gguf"
                elif "8B" in model:
                    expected = "Bonsai-8B-Q1_0.gguf"
                else:
                    continue
                assert got == expected, \
                    f"{model}: expected {expected}, got {got} (cross-size collision bug)"
        finally:
            shutil.rmtree(tmpdir2)

        # 7c. HF API primary path: mock _list_repo_ggufs to return a scoped
        #     file list (one repo's files) and confirm _pick_gguf selects the
        #     right quant with exclusion and packing preference. This tests the
        #     path that runs when online, where the scoped list makes cross-size
        #     collisions impossible.
        from unittest.mock import patch
        hf_files = [
            "Bonsai-1.7B-Q1_0.gguf",
            "Bonsai-1.7B-Q1_0_K.gguf",       # wrong quant variant
            "Bonsai-1.7B-F16.gguf",           # excluded variant
            "Bonsai-1.7B-mmproj.gguf",       # excluded variant
            "Bonsai-1.7B-Q2_0.gguf",         # different quant
            "Bonsai-1.7B-Q2_0_g64.gguf",    # g64 pack (fallback)
        ]
        with patch.object(ev, "_list_repo_ggufs", return_value=hf_files):
            assert ev._resolve_model_file(None, "prism-ml/Bonsai-1.7B-gguf", "Q1_0") == "Bonsai-1.7B-Q1_0.gguf", \
                "HF path: Q1_0 should pick Bonsai-1.7B-Q1_0.gguf"
            assert ev._resolve_model_file(None, "prism-ml/Bonsai-1.7B-gguf", "Q2_0") == "Bonsai-1.7B-Q2_0.gguf", \
                "HF path: Q2_0 should prefer g128 (no _g64 suffix)"
            # g64 fallback: only the _g64 file present for this quant.
            with patch.object(ev, "_list_repo_ggufs", return_value=["Bonsai-1.7B-Q2_0_g64.gguf"]):
                assert ev._resolve_model_file(None, "prism-ml/Bonsai-1.7B-gguf", "Q2_0") == "Bonsai-1.7B-Q2_0_g64.gguf", \
                    "HF path: should fall back to _g64 when no native pack"
            # No match.
            with patch.object(ev, "_list_repo_ggufs", return_value=["Bonsai-1.7B-F16.gguf"]):
                assert ev._resolve_model_file(None, "prism-ml/Bonsai-1.7B-gguf", "Q1_0") is None, \
                    "HF path: excluded-only files should return None"

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

    # ================================================================ Step 6
    # Replay: re-run extract_json_array + score_case on captured records and
    # assert the recomputed score matches the stored score.
    doc6 = "Per a report [1], X happened. Per a study [2], Y happened."
    raw6 = '[{"claim":"X happened","source_ref":"[1]"},{"claim":"Y happened","source_ref":"[2]"}]'
    expected6 = [{"source_ref": "[1]", "key_phrase": "X happened"},
                 {"source_ref": "[2]", "key_phrase": "Y happened"}]
    pred6 = ev.extract_json_array(raw6)
    score6 = ev.score_case(pred6, expected6, "direct", doc6)

    base_rec = {
        "model": "test-model", "mode": "direct",
        "slug": "t0.0-tk20-tp0.9-rp1.0-pp0.0-s1",
        "config": {"temp": 0.0, "seed": 0, "threads": 7},
        "case_id": "c1", "rep": 0,
        "raw_output": raw6, "expected": expected6, "doc": doc6,
        "score": score6, "seconds": 1.0, "scored": True,
    }

    # Test 11: replay on valid captures passes.
    tmpdir11 = tempfile.mkdtemp()
    try:
        cap11 = os.path.join(tmpdir11, "test-model-direct-t0.0.jsonl")
        with open(cap11, "w") as f:
            f.write(json.dumps(base_rec) + "\n")
        ok11 = replay(tmpdir11)
        assert ok11 is True, "test 11: replay should pass on valid captures"
    finally:
        shutil.rmtree(tmpdir11)

    # Test 12: replay on a mutated score fails and names the cell.
    tmpdir12 = tempfile.mkdtemp()
    try:
        import io
        from contextlib import redirect_stderr
        mutated = dict(base_rec)
        mutated["score"] = dict(score6)
        mutated["score"]["recall"] = 0.5
        cap12 = os.path.join(tmpdir12, "test-model-direct-t0.0.jsonl")
        with open(cap12, "w") as f:
            f.write(json.dumps(mutated) + "\n")
        buf = io.StringIO()
        with redirect_stderr(buf):
            ok12 = replay(tmpdir12)
        assert ok12 is False, "test 12: replay should fail on mutated score"
        err = buf.getvalue()
        assert "REPLAY MISMATCH" in err, "test 12: error should name the mismatch"
        assert "test-model" in err and "direct" in err and "c1" in err \
            and "rep=0" in err, \
            f"test 12: error should name the cell, got: {err}"
        assert "recall" in err, \
            f"test 12: error should name the differing field, got: {err}"
    finally:
        shutil.rmtree(tmpdir12)

    # Test 13: replay on scored=false (timeout) record checks timed_out.
    tmpdir13 = tempfile.mkdtemp()
    try:
        rec13 = dict(base_rec)
        rec13["scored"] = False
        rec13["raw_output"] = ""
        rec13["score"] = {"valid_json": False, "timed_out": True,
                          "recall": 0.0, "precision": 0.0,
                          "citation_acc": None, "matched": 0,
                          "expected_n": 2, "quote_match": None, "f1": 0.0}
        cap13 = os.path.join(tmpdir13, "test-model-direct-t0.0.jsonl")
        with open(cap13, "w") as f:
            f.write(json.dumps(rec13) + "\n")
        ok13 = replay(tmpdir13)
        assert ok13 is True, "test 13: replay should pass on timeout record"
    finally:
        shutil.rmtree(tmpdir13)

    # ================================================================ Step 7
    # summarize.py: reads captures dir, emits the report.
    # Test 14: summarize on a small fixtures capture dir. 2 JSONL files,
    # each with 2 cases x 2 reps (4 records per file). Known scores. Run as
    # subprocess, assert stdout contains the required sections, exit 0.
    tmpdir14 = tempfile.mkdtemp()
    try:
        summarize_script = os.path.join(os.path.dirname(__file__), "summarize.py")

        # File 1: greedy (temp 0.0), direct mode, 2 cases x 2 reps.
        # raw_output identical across reps -> deterministic: yes.
        cap14a = os.path.join(tmpdir14, "modelA-direct-t0.0-tk20-tp0.9-rp1.0-pp0.0-s1.jsonl")
        cell_cfg_a = {"temp": 0.0, "top_k": 20, "top_p": 0.9,
                      "repeat_penalty": 1.0, "presence_penalty": 0.0,
                      "strict_schema_validation": True, "template": "",
                      "seed": 0, "threads": 7}
        raw_a = '[{"claim":"X","source_ref":"[1]"}]'
        expected_a = [{"source_ref": "[1]", "key_phrase": "X"}]
        doc_a = "Per a report [1], X happened."
        score_a = ev.score_case(ev.extract_json_array(raw_a), expected_a,
                                "direct", doc_a)
        with open(cap14a, "w") as f:
            for rep in range(2):
                for cid in ("c1", "c2"):
                    rec = ev.build_record("modelA", "direct",
                                          "t0.0-tk20-tp0.9-rp1.0-pp0.0-s1",
                                          cell_cfg_a,
                                          {"id": cid, "text": doc_a,
                                           "expected": expected_a},
                                          rep, raw_a, score_a, 2.0, True)
                    f.write(json.dumps(rec) + "\n")

        # File 2: non-greedy (temp 0.5), quote mode, 2 cases x 2 reps.
        cap14b = os.path.join(tmpdir14, "modelB-quote-t0.5-tk20-tp0.9-rp1.0-pp0.0-s1.jsonl")
        cell_cfg_b = {"temp": 0.5, "top_k": 20, "top_p": 0.9,
                      "repeat_penalty": 1.0, "presence_penalty": 0.0,
                      "strict_schema_validation": True, "template": "",
                      "seed": 0, "threads": 7}
        raw_b = '[{"claim":"Y","quote":"Y happened"}]'
        expected_b = [{"source_ref": "[1]", "key_phrase": "Y"}]
        doc_b = "Per a report [1], Y happened."
        score_b = ev.score_case(ev.extract_json_array(raw_b), expected_b,
                                "quote", doc_b)
        with open(cap14b, "w") as f:
            for rep in range(2):
                for cid in ("c1", "c2"):
                    rec = ev.build_record("modelB", "quote",
                                          "t0.5-tk20-tp0.9-rp1.0-pp0.0-s1",
                                          cell_cfg_b,
                                          {"id": cid, "text": doc_b,
                                           "expected": expected_b},
                                          rep, raw_b, score_b, 3.0, True)
                    f.write(json.dumps(rec) + "\n")

        result14 = subprocess.run(
            [sys.executable, summarize_script, tmpdir14],
            capture_output=True, text=True)
        assert result14.returncode == 0, \
            f"test 14: summarize exited {result14.returncode}, " \
            f"stderr: {result14.stderr}"
        out14 = result14.stdout.lower()
        for keyword in ("precision", "recall", "within-case",
                        "across-cases", "latency", "determinis"):
            assert keyword in out14, \
                f"test 14: output missing '{keyword}'"
    finally:
        shutil.rmtree(tmpdir14)

    # Test 15: summarize on empty dir exits nonzero with a message.
    tmpdir15 = tempfile.mkdtemp()
    try:
        summarize_script = os.path.join(os.path.dirname(__file__), "summarize.py")
        result15 = subprocess.run(
            [sys.executable, summarize_script, tmpdir15],
            capture_output=True, text=True)
        assert result15.returncode != 0, \
            "test 15: summarize on empty dir should exit nonzero"
        assert result15.stderr.strip() or result15.stdout.strip(), \
            "test 15: summarize on empty dir should print a message"
    finally:
        shutil.rmtree(tmpdir15)

    print("self-test: scorer is deterministic, one-to-one assignment works")
    print("self-test: capture round-trip writes and reads back all required fields")
    print("self-test: load_config validates, resolves, and assigns seeds correctly")
    print("self-test: strict_schema_validation wires --json-schema per mode")
    print("self-test: replay verifies scores, names mismatches, checks timeouts")
    print("self-test: summarize reports variance, latency, f1, determinism")
    return True


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if mode in ("--replay", "replay"):
        if len(sys.argv) < 3:
            print("replay requires a directory argument", file=sys.stderr)
            sys.exit(2)
        ok = replay(sys.argv[2])
        sys.exit(0 if ok else 1)
    if mode == "selftest":
        sys.exit(0 if self_test() else 1)
    print("usage: verify.py [selftest|--replay <dir>]", file=sys.stderr)
    sys.exit(2)
