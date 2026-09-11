"""prc.pipeline.assemble / validate / write -- one frame per scored row with lane provenance; the contract
checks; the manifest's shape and its deterministic digest; nothing is ever overwritten; and the no-splice rule
(no input under submissions/, no splice call anywhere in the package).

Mutation rehearsal (2026-09-11, scratchpad p2a/mutate.py): the break named in each docstring turned the test
RED against the module under test, GREEN on restore.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("pipeline_world", pathlib.Path(__file__).with_name("pipeline_world.py"))
pw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pw)

from prc.pipeline import assemble as A  # noqa: E402
from prc.pipeline import config as C  # noqa: E402
from prc.pipeline import errors as E  # noqa: E402
from prc.pipeline import lanes, validate, write  # noqa: E402

IDS = np.arange(100.0, 108.0)
AP = np.array(["EDDF", "EDDF", "LIRF", "LIRF", "EKCH", "EKCH", "LTFM", "LTFM"], dtype=object)
LANE = np.array(["matched", "unmatched", "matched", "lirf_rules", "matched", "unmatched", "matched", "unmatched"], dtype=object)


def _routing():
    rc = np.where(LANE == "matched", "matched", "unmatched").astype(object)
    return lanes.Routing(ids=IDS.copy(), airport=AP.copy(), row_class=rc, lane=LANE.copy(), fallback=AP == "EKCH")


def _cfg():
    raw = pw.fresh(pw.real_raw())
    raw["airports"]["EKCH"] = {"tz": "Europe/Copenhagen", "history": False,
                               "lanes": {"matched": "matched", "unmatched": "unmatched"}}
    return C.parse_config(raw, root=ROOT)


def _result(lane, values=None, ids=None):
    ids = IDS[LANE == lane] if ids is None else np.asarray(ids, dtype="float64")
    values = np.full(len(ids), 700.0) if values is None else np.asarray(values, dtype="float64")
    return lanes.LaneResult(lane=lane, ids=ids, values=values, provenance=lanes._prov(ids, lane, "m", "s"))


def _results():
    return [_result("matched", [900, 901, 902, 903]), _result("unmatched", [500, 501, 502]), _result("lirf_rules", [40000])]


def _template():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"MVT_ID_mvt": IDS[rng.permutation(len(IDS))]})


# ---- assemble -------------------------------------------------------------------------------------------

def test_assemble_gives_one_int32_row_per_template_id_in_template_order_with_provenance():
    """Rehearsed: assemble() leaving the rows in lane order (no template reindex) -> RED."""
    t = _template()
    a = A.assemble(_results(), _routing(), t)
    assert np.array_equal(a.submission.MVT_ID_mvt.to_numpy(), t.MVT_ID_mvt.to_numpy())
    assert a.submission.TAXITIME_SEC_mvt.dtype == np.int32 and list(a.submission.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    got = dict(zip(a.provenance.MVT_ID_mvt, a.provenance.TAXITIME_SEC_mvt))
    assert got[IDS[3]] == 40000 and got[IDS[0]] == 900 and got[IDS[7]] == 502
    assert list(a.provenance.columns) == A.PROVENANCE_OUT and a.lane_counts() == {"lirf_rules": 1, "matched": 4, "unmatched": 3}
    assert a.provenance.set_index("MVT_ID_mvt").fallback.loc[IDS[4]] and not a.provenance.set_index("MVT_ID_mvt").fallback.loc[IDS[0]]


def test_assemble_refuses_a_lane_that_returns_a_row_routed_elsewhere():
    """Rehearsed: assemble()'s per-lane id-set comparison deleted -> RED."""
    res = _results()
    res[1] = _result("unmatched", [500, 501, 502], ids=[IDS[1], IDS[5], IDS[0]])      # IDS[0] is matched's
    with pytest.raises(E.LaneError, match="unmatched"):
        A.assemble(res, _routing(), _template())


def test_assemble_refuses_a_missing_lane_and_an_unrouted_lane():
    """Rehearsed: the missing-lane check deleted -> RED."""
    with pytest.raises(E.LaneError, match="lirf_rules"):
        A.assemble(_results()[:2], _routing(), _template())
    extra = _results() + [lanes.LaneResult(lane="spare", ids=np.array([]), values=np.array([]),
                                           provenance=lanes._prov([], "spare", "m", "s"))]
    with pytest.raises(E.LaneError, match="spare"):
        A.assemble(extra, _routing(), _template())


def test_a_lane_result_refuses_fractional_nonfinite_or_misaligned_output():
    """Rehearsed: LaneResult's whole-seconds check deleted -> RED."""
    ids = IDS[:2]
    with pytest.raises(E.LaneError, match="whole seconds"):
        lanes.LaneResult("x", ids, [1.5, 2.0], lanes._prov(ids, "x", "m", "s"))
    with pytest.raises(E.LaneError, match="non-finite"):
        lanes.LaneResult("x", ids, [np.nan, 2.0], lanes._prov(ids, "x", "m", "s"))
    with pytest.raises(E.LaneError, match="aligned"):
        lanes.LaneResult("x", ids, [1.0, 2.0], lanes._prov(ids[::-1], "x", "m", "s"))
    with pytest.raises(E.LaneError, match="duplicate"):
        lanes.LaneResult("x", [1.0, 1.0], [1.0, 2.0], lanes._prov([1.0, 1.0], "x", "m", "s"))


# ---- validate -------------------------------------------------------------------------------------------

def test_validate_passes_and_reports_per_lane_per_airport_and_fallback_rows():
    """Rehearsed: validate()'s per-lane report computed over all rows (not per lane) -> RED."""
    t = _template()
    a = A.assemble(_results(), _routing(), t)
    rep = validate.validate(a, t, _routing(), _cfg())
    assert rep["per_lane"]["lirf_rules"] == {"n": 1, "min": 40000, "median": 40000.0, "max": 40000, "by_stage": {"s": 1}}
    assert rep["per_lane"]["matched"]["n"] == 4 and rep["fallback_rows"] == {"EKCH": 2}
    assert rep["per_airport"]["EDDF"] == 2 and rep["check_submission"] == "passed"
    assert any("LSZH" in w for w in rep["warnings"])                  # configured, no scored rows here


def test_validate_refuses_a_value_outside_the_range():
    """Rehearsed: validate()'s range check deleted -> RED."""
    t = _template()
    res = _results()
    res[2] = _result("lirf_rules", [validate.VALUE_RANGE_S[1] + 1])
    with pytest.raises(E.ValidationError, match="outside"):
        validate.validate(A.assemble(res, _routing(), t), t, _routing(), _cfg())


def test_validate_refuses_counts_order_or_dtype_that_disagree_with_the_routing():
    """Rehearsed: the per-lane count comparison deleted -> RED (the relabelled row passes)."""
    t = _template()
    a = A.assemble(_results(), _routing(), t)
    prov = a.provenance.copy()
    prov.loc[prov.lane == "lirf_rules", "lane"] = "matched"
    with pytest.raises(E.ValidationError, match="per-lane row counts"):
        validate.validate(dataclasses.replace(a, provenance=prov), t, _routing(), _cfg())
    with pytest.raises(E.ValidationError, match="row order"):
        validate.validate(dataclasses.replace(a, submission=a.submission.iloc[::-1].reset_index(drop=True)), t,
                          _routing(), _cfg())
    with pytest.raises(E.ValidationError, match="int32"):
        validate.validate(dataclasses.replace(a, submission=a.submission.astype({"TAXITIME_SEC_mvt": "int64"})), t,
                          _routing(), _cfg())


def test_validate_refuses_an_airport_missing_predictions():
    """Rehearsed: the per-airport comparison deleted -> RED."""
    t = _template()
    a = A.assemble(_results(), _routing(), t)
    prov = a.provenance.copy()
    prov.loc[prov.airport == "LTFM", "airport"] = "EDDF"
    with pytest.raises(E.ValidationError, match="LTFM"):
        validate.validate(dataclasses.replace(a, provenance=prov), t, _routing(), _cfg())


def test_compare_to_reference_counts_equal_and_differing_rows_per_lane(tmp_path):
    """Rehearsed: compare_to_reference counting |diff| <= 1 as equal -> RED."""
    t = _template()
    a = A.assemble(_results(), _routing(), t)
    ref = a.submission.copy()
    ref.loc[ref.MVT_ID_mvt == IDS[7], "TAXITIME_SEC_mvt"] += 1
    p = tmp_path / "ref.parquet"
    ref.to_parquet(p, index=False)
    out = validate.compare_to_reference(a, p)
    assert out["unmatched"] == {"n": 3, "n_equal": 2, "n_differ": 1, "max_abs_diff_s": 1, "first_differing_ids": [IDS[7]]}
    assert out["matched"]["n_differ"] == 0 and out["ALL"] == {"n": 8, "n_equal": 7, "n_differ": 1}


# ---- the no-splice rule ---------------------------------------------------------------------------------

def test_no_pipeline_input_may_come_from_submissions(tmp_path):
    """Rehearsed: assert_no_splice returning without checking -> RED."""
    ok = tmp_path / "data" / "x.parquet"
    assert validate.assert_no_splice([ok]) == [str(ok)]
    with pytest.raises(E.SpliceRefusedError, match="lgbm_submit --base"):
        validate.assert_no_splice([ok, ROOT / "submissions" / "merry-quicksand_v2.parquet"])


def test_the_package_calls_no_splice_function_and_names_no_submission_file():
    """Static guard: no module of prc/pipeline calls a script's splice / main / build-on-a-base path, or names a
    submission file. (rome_bandfloor.build IS called -- on the lane's own frame of freshly computed values; that
    call site is the one allowed, and is pinned here.) Rehearsed: a `B.splice_unmatched(` call added to
    lanes.py -> RED."""
    forbidden = [r"splice_unmatched\(", r"\bls\.splice\(", r"lgbm_submit\.main\(", r"\.main\(\s*\[",
                 r"rome_ship\.build\(", r"\brs\.build\(", r"ucs\.build\(", r"merry-quicksand_v\d",
                 r"add_argument\(\s*[\"']--base"]
    hits = []
    for p in sorted((ROOT / "prc" / "pipeline").glob("*.py")):
        text = p.read_text()
        code = re.sub(r'"""[\s\S]*?"""', "", text)                        # docstrings may explain the rule
        code = "\n".join(line.split("#", 1)[0] for line in code.splitlines())
        for pat in forbidden:
            if re.search(pat, code):
                hits.append(f"{p.name}: {pat}")
    assert not hits, hits
    assert len(re.findall(r"rb\.build\(", (ROOT / "prc" / "pipeline" / "lanes.py").read_text())) == 1


# ---- manifest and writing -------------------------------------------------------------------------------

def _manifest(tmp_path, cfg, a, rep, timings, input_bytes=b"abc"):
    f = tmp_path / "input.bin"
    f.write_bytes(input_bytes)
    inputs = write.describe_inputs({"x": f}, tmp_path)
    outputs = {"submission": {"file": "submission.parquet", "sha256": "0" * 64, "rows": 8}}
    return write.build_manifest(cfg, inputs, outputs, {"matched": {"n_rows": 4}}, _routing(), rep, timings, [str(f)])


def test_the_manifest_has_every_required_field_and_a_digest_that_ignores_only_wall_clock(tmp_path):
    """Rehearsed: manifest_digest hashing the whole manifest (timings included) -> RED. The two manifests hash the live
    repo's code, so the comparison goes through pw.assert_same_run (BC-8: another session may save a loaded module between
    the two builds)."""
    import time
    cfg, t = _cfg(), _template()
    a = A.assemble(_results(), _routing(), t)
    rep = validate.validate(a, t, _routing(), cfg)
    since = time.time()
    m1 = _manifest(tmp_path, cfg, a, rep, {"lane_matched": 1.0})
    m2 = _manifest(tmp_path, cfg, a, rep, {"lane_matched": 99.0})
    write.check_manifest_shape(m1)
    assert set(write.REQUIRED_KEYS) <= set(m1)
    pw.assert_same_run(m1, m2, since, ROOT)
    assert m1["config"]["hash"] == cfg.config_hash and m1["fallback_airports"] == ["EKCH"]
    assert m1["airports"]["EKCH"]["fallback"] is True and m1["airports"]["LIRF"]["rules"] == list(C.RULE_ORDER)
    assert m1["routing"]["n_fallback_rows"] == 2 and set(m1["git"]) == {"sha", "dirty"}
    assert m1["inputs"]["x"]["sha256"] == __import__("hashlib").sha256(b"abc").hexdigest()


def test_the_manifest_hashes_the_source_files_that_ran(tmp_path):
    """Review finding 2026-09-11: the manifest stamped a git sha that did not contain the untracked code that ran.
    Rehearsed 2026-09-11 (c4): build_manifest without the code block -> RED; loaded_code_files ignoring sys.modules
    (prc/pipeline only) -> RED (the scripts file is missing); check_manifest_shape without the code check -> RED."""
    from prc.pipeline import legacy
    legacy.matched()                                    # the matched stack loaded HERE, so no test order is assumed
    cfg, t = _cfg(), _template()
    a = A.assemble(_results(), _routing(), t)
    rep = validate.validate(a, t, _routing(), cfg)
    m = _manifest(tmp_path, cfg, a, rep, {})
    write.check_manifest_shape(m)
    files = m["code"]["files"]
    assert files["prc/pipeline/write.py"] == write.sha256_file(ROOT / "prc" / "pipeline" / "write.py")
    assert {"scripts/lgbm_submit.py", "scripts/stand_ab.py"} <= set(files)   # and lgbm_submit's own import
    assert all(p.split("/")[0] in write.CODE_DIRS for p in files)
    assert m["code"]["differs_from_head"] is None or isinstance(m["code"]["differs_from_head"], list)
    broken = dict(m, code={"files": {}, "differs_from_head": None})
    broken["digest"] = write.manifest_digest(broken)
    with pytest.raises(E.ValidationError, match="code"):
        write.check_manifest_shape(broken)


def test_code_state_follows_a_loaded_scripts_module_byte_for_byte(tmp_path):
    """A module imported from <root>/scripts is hashed; changing one byte of it changes its hash."""
    import importlib.util as iu
    import sys as _sys
    (tmp_path / "prc" / "pipeline").mkdir(parents=True)
    (tmp_path / "prc" / "pipeline" / "write.py").write_text("# stand-in\n")
    (tmp_path / "scripts").mkdir()
    src = tmp_path / "scripts" / "zz_code_state_probe.py"
    src.write_text("X = 1\n")
    spec = iu.spec_from_file_location("zz_code_state_probe", src)
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _sys.modules["zz_code_state_probe"] = mod
    try:
        c1 = write.code_state(tmp_path)
        src.write_text("X = 2\n")
        c2 = write.code_state(tmp_path)
    finally:
        del _sys.modules["zz_code_state_probe"]
    assert set(c1["files"]) == {"prc/pipeline/write.py", "scripts/zz_code_state_probe.py"}
    assert c1["files"]["scripts/zz_code_state_probe.py"] != c2["files"]["scripts/zz_code_state_probe.py"]


@pytest.mark.parametrize("text,want", [
    ("", []),
    (" M prc/pipeline/lanes.py\n?? scripts/new.py\n", ["prc/pipeline/lanes.py", "scripts/new.py"]),
    ('R  scripts/old.py -> scripts/renamed.py\n', ["scripts/renamed.py"]),
    ('?? "scripts/with space.py"\n', ["scripts/with space.py"]),
])
def test_porcelain_paths_reads_modified_untracked_and_renamed(text, want):
    assert write.porcelain_paths(text) == want


def test_the_manifest_digest_changes_when_an_input_byte_changes_and_tampering_is_caught(tmp_path):
    """Rehearsed: describe_inputs recording the file size instead of its sha256 -> RED (same size, other bytes)."""
    cfg, t = _cfg(), _template()
    a = A.assemble(_results(), _routing(), t)
    rep = validate.validate(a, t, _routing(), cfg)
    m1 = _manifest(tmp_path, cfg, a, rep, {}, b"abc")
    m2 = _manifest(tmp_path, cfg, a, rep, {}, b"abd")
    assert m1["digest"] != m2["digest"]
    m1["validation"]["n_rows"] = 9
    with pytest.raises(E.ValidationError, match="digest"):
        write.check_manifest_shape(m1)


def test_regression_a_wall_clock_value_outside_timings_is_refused_by_the_manifest_check(tmp_path):
    """Regression, 2026-09-11 (found as a FLAKY end-to-end test: 1 failure in ~8 full-suite runs): lanes.py's
    run_matched_lane put `design_s` / `predict_s` into the lane info, which the manifest records under `lanes`,
    inside the digest -- two identical runs produced different digests whenever the rounded timings differed.
    Class: a volatile value inside the deterministic part of a stamped record. Fix at the origin (no timing in lane
    info) plus this tripwire. Rehearsed: check_manifest_shape without the volatile_leaks check -> RED."""
    cfg, t = _cfg(), _template()
    a = A.assemble(_results(), _routing(), t)
    rep = validate.validate(a, t, _routing(), cfg)
    f = tmp_path / "in.bin"
    f.write_bytes(b"x")
    m = write.build_manifest(cfg, write.describe_inputs({"x": f}, tmp_path), {}, {"matched": {"n_rows": 4, "predict_s": 0.2}},
                             _routing(), rep, {"lane_matched": 0.2}, [str(f)])
    with pytest.raises(E.ValidationError, match=r"lanes\.matched\.predict_s"):
        write.check_manifest_shape(m)
    assert write.volatile_leaks({"timings_s": {"x_wall_s": 1}, "validation": {"value_range_s": [1, 2]}}) == []


def test_writing_names_the_files_checks_what_landed_and_never_overwrites(tmp_path):
    """Rehearsed: write_submission's exists() refusal deleted -> RED."""
    t = _template()
    a = A.assemble(_results(), _routing(), t)
    assert write.output_names("merry-quicksand", None) == ("submission.parquet", "submission.provenance.parquet",
                                                           "submission.manifest.json")
    assert write.output_names("merry-quicksand", 11)[0] == "merry-quicksand_v11.parquet"
    out = write.write_submission(a, t, tmp_path, "merry-quicksand")
    back = pd.read_parquet(tmp_path / "submission.parquet")
    assert back.equals(a.submission) and out["submission"]["rows"] == 8
    assert pd.read_parquet(tmp_path / "submission.provenance.parquet").lane.tolist() == a.provenance.lane.tolist()
    with pytest.raises(E.ValidationError, match="overwrite"):
        write.write_submission(a, t, tmp_path, "merry-quicksand")
    cfg = _cfg()
    rep = validate.validate(a, t, _routing(), cfg)
    m = _manifest(tmp_path, cfg, a, rep, {})
    path = write.write_manifest(m, tmp_path, "merry-quicksand")
    assert json.loads(path.read_text())["digest"] == m["digest"]
    with pytest.raises(E.ValidationError, match="overwrite"):
        write.write_manifest(m, tmp_path, "merry-quicksand")
