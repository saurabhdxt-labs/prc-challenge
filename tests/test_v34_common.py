"""scripts/analysis/v34_common.py — the shared v34 discipline: the no-seconds invariant, the output-path rules (new paths
only, never a shipped-artefact directory), the registration check and the step-report chain of custody.
Nothing here loads real data or takes a lock. Mutation rehearsal 2026-09-24: see docstrings."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("v34_common", "scripts/analysis/v34_common.py")


def test_seconds_cols_are_nsc_folds():
    src = (ROOT / "scripts/analysis/nsc_fold.py").read_text()
    assert 'SECONDS_COLS = ("aobt_sec", "mvt_sec", "sched_sec")' in src
    assert C.SECONDS_COLS == ("aobt_sec", "mvt_sec", "sched_sec")


def test_assert_no_seconds_refuses_each_seconds_column_and_any_sec_name():
    """Rehearsed RED: checking only `mvt_sec` (aobt_sec passes) and dropping the *_sec suffix rule (x_sec passes)."""
    assert C.assert_no_seconds(("proxy", "sp"), "ok") == ["proxy", "sp"]
    for bad in ("aobt_sec", "mvt_sec", "sched_sec", "renamed_sec"):
        with pytest.raises(AssertionError, match=f"label-seconds column.*{bad}"):
            C.assert_no_seconds(["proxy", bad], "design")
    with pytest.raises(AssertionError, match="duplicate"):
        C.assert_no_seconds(["a", "a"], "design")


def test_amendment_heading_per_step_and_refuses_unknown():
    assert C.amendment_heading(1) == "## AMENDMENT V34.1 " and C.amendment_heading(7) == "## AMENDMENT V34.7 "
    with pytest.raises(ValueError):
        C.amendment_heading(8)


def test_output_roots_are_outside_every_protected_directory():
    for r in (C.OUT_ROOT, C.SMOKE_ROOT):
        assert not any(C._under(r.resolve(), d.resolve()) for d in C.PROTECTED_DIRS)
    assert C.V34_FINAL.name == "merry-quicksand_v34.parquet"


def test_check_output_path_allows_only_new_paths_under_the_roots(tmp_path):
    """Rehearsed RED: dropping the exists check (overwrite passes) and dropping the root check (/etc passes)."""
    root = tmp_path / "out"
    ok = root / "a" / "x.parquet"
    assert C.check_output_path(ok, allow_roots=(root,), allow_files=()) == ok.resolve()
    ok.parent.mkdir(parents=True)
    ok.write_text("x")
    with pytest.raises(FileExistsError, match="never overwrites"):
        C.check_output_path(ok, allow_roots=(root,), allow_files=())
    with pytest.raises(PermissionError, match="outside the v34 output roots"):
        C.check_output_path(tmp_path / "elsewhere.json", allow_roots=(root,), allow_files=())


def test_check_output_path_refuses_shipped_artefacts_even_if_a_root_contains_them():
    """The real defaults: the shipped F boosters and the v33 file are refused; the registered v34 files are allowed."""
    with pytest.raises(PermissionError):
        C.check_output_path(ROOT / "data/cache_stand/lgbm_v6_queue_order_seed0.txt")
    with pytest.raises((PermissionError, FileExistsError)):
        C.check_output_path(ROOT / "submissions/merry-quicksand_v33.parquet")
    with pytest.raises(PermissionError, match="shipped-artefact"):
        C.check_output_path(ROOT / "data/adsb/v2/new.txt", allow_roots=(ROOT / "data",), allow_files=())
    if not C.V34_CANDIDATE.exists():
        assert C.check_output_path(C.V34_CANDIDATE) == C.V34_CANDIDATE.resolve()
    assert C.check_output_path(C.OUT_ROOT / "step1_f_fold" / "never_written.json")


def test_write_parquet_and_json_new_write_once_and_return_the_sha(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
    p = tmp_path / "x.parquet"
    sha = C.write_parquet_new(df, p, allow_roots=(tmp_path,), allow_files=())
    assert sha == C.sha256(p) and pd.read_parquet(p).equals(df)
    with pytest.raises(FileExistsError):
        C.write_parquet_new(df, p, allow_roots=(tmp_path,), allow_files=())
    j = tmp_path / "r.json"
    C.write_json_new({"k": 1.5, "n": C.np.int64(3), "ok": C.np.bool_(True)}, j, allow_roots=(tmp_path,), allow_files=())
    assert json.loads(j.read_text()) == {"k": 1.5, "n": 3, "ok": True}
    with pytest.raises(ValueError, match="non-finite"):
        C.write_json_new({"k": float("nan")}, tmp_path / "n.json", allow_roots=(tmp_path,), allow_files=())


def _registered(tmp_path, text):
    pre = tmp_path / "PREREG_x.md"
    pre.write_text(text)
    log = tmp_path / "hashes.log"
    log.write_text(f"{C.sha256(pre)}  2026  AMENDMENT  PREREG_x.md\n")
    return pre, log


def test_require_registration_needs_the_logged_sha_and_the_step_heading(tmp_path):
    """Rehearsed RED: skipping the heading check lets step 2 run under a step-1-only registration."""
    pre, log = _registered(tmp_path, "# x\n\n## AMENDMENT V34.1 — step 1\n")
    assert C.require_registration(1, pre, log) == C.sha256(pre)
    with pytest.raises(RuntimeError, match="AMENDMENT V34.2"):
        C.require_registration(2, pre, log)
    pre.write_text(pre.read_text() + "\nappended, not hashed\n")
    with pytest.raises(RuntimeError, match="not in"):
        C.require_registration(1, pre, log)


def test_read_step_report_verifies_status_smoke_and_output_shas(tmp_path, monkeypatch):
    """Rehearsed RED: dropping the sha re-check (a changed output passes)."""
    monkeypatch.setattr(C, "ROOT", tmp_path)
    d = C.step_dir(1, tmp_path)
    d.mkdir(parents=True)
    out = d / "rec.parquet"
    out.write_bytes(b"abc")
    rep = {"status": "PASS", "smoke": False, "outputs": {str(out.relative_to(tmp_path)): C.sha256(out)}}
    (d / C.REPORT_NAME).write_text(json.dumps(rep))
    assert C.read_step_report(1, tmp_path)["status"] == "PASS"
    out.write_bytes(b"abd")
    with pytest.raises(AssertionError, match="changed since"):
        C.read_step_report(1, tmp_path)
    (d / C.REPORT_NAME).write_text(json.dumps({**rep, "status": "FAIL"}))
    with pytest.raises(RuntimeError, match="not PASS"):
        C.read_step_report(1, tmp_path)
    (d / C.REPORT_NAME).write_text(json.dumps({**rep, "smoke": True}))
    with pytest.raises(RuntimeError, match="smoke"):
        C.read_step_report(1, tmp_path)
    with pytest.raises(FileNotFoundError, match="run it first"):
        C.read_step_report(2, tmp_path)


def test_taxi_floors_at_one_second():
    assert C.taxi([100.0, 5.0], [40.0, 50.0]).tolist() == [60.0, 1.0]


def test_jsonable_converts_numpy_and_refuses_non_finite():
    """Rehearsed RED: letting NaN through (json.dumps would write NaN, which allow_nan=False then refuses late)."""
    out = C.jsonable({1: (C.np.float32(1.5), C.np.array([1, 2])), "p": ROOT})
    assert out == {"1": [1.5, [1, 2]], "p": str(ROOT)}
    with pytest.raises(ValueError, match="non-finite"):
        C.jsonable({"x": [C.np.inf]})
