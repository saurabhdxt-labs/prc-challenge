"""Tests for scripts/analysis/fwp_build.py — FWP's G2 reproduction check and the FW/v19 splice (build steps 3-4;
plans/PREREG_fw_production_build_2026_09_16.md). No model is loaded here; every fixture is a tiny hand-built frame.
Each test names the mutation it must catch.

Mutation rehearsal 2026-09-17 (scripts/analysis/__pycache__ and tests/__pycache__ cleared before every run; backup,
break, run the full test file, confirm RED, restore, confirm GREEN again — only fwp_build.py was ever touched, never
the imported cbv_ship_build/afj_ship_build/build_submission modules). 8/8 mutants caught RED:
  1. compute_R: `cbv_candidate == v19v` -> `!=` (R membership inverted)              -> caught (test_compute_R_*)
  2. fw_candidate: `CBV.taxi(proxy, c)` -> `CBV.taxi(c, proxy)` (args swapped)        -> caught (test_fw_candidate_formula_*)
  3. assert_outside_R_unchanged: `~mask_r` -> `mask_r` (inverted mask)               -> caught (test_outside_R_identity_*)
  4. repro_check: `"identical": bool((diff == 0).all())` -> `"identical": True`      -> caught (test_repro_check_one_differing_row_*)
  5. fw_values_for: `v < 1.0` -> `v < 0.0` (floor check weakened)                    -> caught (test_fw_values_for_refuses_*[subfloor])
  6. fw_values_for: `d > G4_MAX_ABS_DELTA_S` -> `d > G4_MAX_ABS_DELTA_S * 10`         -> caught (test_fw_values_for_refuses_*[toofar])
  7. projection_P: `sse_gain = (...) * n_rows` -> `* n_pop` (wrong row count)         -> caught (test_projection_P_hand_example)
  8. stop_reason: `p >= STOP_P` -> `p > STOP_P` (boundary flips)                      -> caught (test_stop_reason_boundary_at_50)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import fwp_build as F  # noqa: E402


# ---------------------------------------------------------------- fw_output_path / naming

def test_fw_output_path_matches_lgbm_submits_version_6_name(tmp_path):
    """G2's/G4's input file must be exactly what `lgbm_submit.py --version 6 ... --out-dir <d>` writes. Fails if the
    version or naming convention drifts."""
    assert F.fw_output_path(tmp_path) == tmp_path / "merry-quicksand_v6.parquet"
    assert F.FW_SUBMISSION_NAME == "merry-quicksand_v6.parquet"


# ---------------------------------------------------------------- value formula (rounding, floor)

def test_fw_candidate_formula_rounding_and_floor():
    """rint(0.5*max(proxy-C,1) + 0.5*FW). Case A: no floor (proxy > C), exact rounding. Case B: the floor binds
    (proxy < C) and a banker's-rounding tie (150.5 -> 150, even). Fails if the floor is dropped, the weight is not
    0.5, or rounding breaks."""
    out = F.fw_candidate([900.0, 50.0], [300.0, 900.0], [400.0, 300.0])
    assert out.tolist() == [500.0, 150.0]
    with pytest.raises(AssertionError, match="S5"):
        F.fw_candidate([900.0], [300.0], [np.nan])


# ---------------------------------------------------------------- R membership

def _scope(ids, taxi_c, v13, airport=None):
    return pd.DataFrame({"MVT_ID_mvt": ids, "proxy": 0.0, "C": 0.0, "taxi_C": taxi_c, "v13": v13,
                         "airport": airport or ["EDDF"] * len(ids)})


def test_compute_R_excludes_a_row_a_later_lever_touched():
    """Three scope rows; row 2's v19 value was moved by a later lever (e.g. AFJ) and must be excluded from R, not
    raised on. Fails if R takes every scope row regardless of v19, or if the excluded count is wrong."""
    scope = _scope([1.0, 2.0, 3.0], [601.0, 100.0, 900.0], [400.0, 1.0, 300.0])
    v19 = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "TAXITIME_SEC_mvt": np.array([500, 999, 600], dtype="int32")})
    r, counts = F.compute_R(scope, v19)
    assert r.MVT_ID_mvt.tolist() == [1.0, 3.0]
    assert counts == {"n_scope": 3, "n_R": 2, "n_excluded_later_lever": 1}
    assert r.cbv_candidate.tolist() == r.v19.tolist() == [500.0, 600.0]


def test_compute_R_raises_if_a_scope_id_is_missing_from_v19():
    scope = _scope([1.0, 2.0], [601.0, 100.0], [400.0, 1.0])
    v19 = pd.DataFrame({"MVT_ID_mvt": [1.0], "TAXITIME_SEC_mvt": np.array([500], dtype="int32")})
    with pytest.raises(AssertionError, match="G6"):
        F.compute_R(scope, v19)


# ---------------------------------------------------------------- outside-R identity guard (G3)

def test_outside_R_identity_guard_fires_on_injected_change():
    """A row not in R must stay byte-identical between v19 and the candidate. Fails if the mask is inverted or the
    comparison is skipped."""
    v19 = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "TAXITIME_SEC_mvt": np.array([100, 200, 300], dtype="int32")})
    cand_ok = v19.copy()
    cand_ok.loc[cand_ok.MVT_ID_mvt == 1.0, "TAXITIME_SEC_mvt"] = 111
    F.assert_outside_R_unchanged(v19, cand_ok, [1.0])          # row 1 is in R: no error
    cand_bad = v19.copy()
    cand_bad.loc[cand_bad.MVT_ID_mvt == 2.0, "TAXITIME_SEC_mvt"] = 222   # row 2 is OUTSIDE R: injected change
    with pytest.raises(AssertionError, match="G3"):
        F.assert_outside_R_unchanged(v19, cand_bad, [1.0])


# ---------------------------------------------------------------- G2 (repro-check)

def _write_v13(tmp_path, ids, values, stage):
    v13 = pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": np.asarray(values, dtype="int32")})
    prov = pd.DataFrame({"MVT_ID_mvt": ids, "stage": stage})
    v13_p, prov_p = tmp_path / "v13.parquet", tmp_path / "v13.prov.parquet"
    v13.to_parquet(v13_p, index=False)
    prov.to_parquet(prov_p, index=False)
    return v13_p, prov_p


def test_repro_check_one_differing_row_is_not_identical(tmp_path, monkeypatch):
    """G2: every scope-stage row must reproduce v13 exactly. One row off by 5 s must be caught, named, and counted;
    it must not be swallowed by a tolerance. Fails if equality is loosened to a tolerance or the diff sign is lost."""
    monkeypatch.setattr(F.CBV, "STAGE_COUNTS", {"lightgbm:delta:mean3": 3, "lightgbm:delta:mean3+adsb": 0})
    v13_p, prov_p = _write_v13(tmp_path, [1.0, 2.0, 3.0], [100, 200, 300], "lightgbm:delta:mean3")
    repro_dir = tmp_path / "repro"
    repro_dir.mkdir()
    repro = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "TAXITIME_SEC_mvt": np.array([100, 205, 300], dtype="int32")})
    repro.to_parquet(F.fw_output_path(repro_dir), index=False)
    v = F.repro_check(repro_dir, v13_sub_path=v13_p, v13_prov_path=prov_p)
    assert v["identical"] is False
    assert v["n_diff"] == 1 and v["max_abs_diff"] == 5.0
    assert v["first20_diff_ids"] == [{"MVT_ID_mvt": 2.0, "diff": 5.0}]
    assert F.cmd_repro_check(repro_dir, v13_sub_path=v13_p, v13_prov_path=prov_p) == 1
    assert json.loads((repro_dir / F.G2_VERDICT_NAME).read_text())["identical"] is False


def test_repro_check_identical_case_exits_0(tmp_path, monkeypatch):
    monkeypatch.setattr(F.CBV, "STAGE_COUNTS", {"lightgbm:delta:mean3": 2, "lightgbm:delta:mean3+adsb": 0})
    v13_p, prov_p = _write_v13(tmp_path, [1.0, 2.0], [100, 200], "lightgbm:delta:mean3")
    repro_dir = tmp_path / "repro"
    repro_dir.mkdir()
    repro = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "TAXITIME_SEC_mvt": np.array([100, 200], dtype="int32")})
    repro.to_parquet(F.fw_output_path(repro_dir), index=False)
    v = F.repro_check(repro_dir, v13_sub_path=v13_p, v13_prov_path=prov_p)
    assert v["identical"] is True and v["n_diff"] == 0 and v["first20_diff_ids"] == []
    assert F.cmd_repro_check(repro_dir, v13_sub_path=v13_p, v13_prov_path=prov_p) == 0


def test_repro_check_refuses_a_scope_count_that_drifted(tmp_path, monkeypatch):
    monkeypatch.setattr(F.CBV, "STAGE_COUNTS", {"lightgbm:delta:mean3": 99, "lightgbm:delta:mean3+adsb": 0})
    v13_p, prov_p = _write_v13(tmp_path, [1.0], [100], "lightgbm:delta:mean3")
    with pytest.raises(AssertionError, match="G2"):
        F.repro_check(tmp_path, v13_sub_path=v13_p, v13_prov_path=prov_p)


# ---------------------------------------------------------------- G4 (FW bounds)

@pytest.mark.parametrize("name, value, v13_val, match", [
    ("nonfinite", float("nan"), 100.0, "non-finite"),
    ("subfloor", 0.5, 100.0, "below 1"),
    ("toofar", 5000.0, 100.0, "differ from v13"),
])
def test_fw_values_for_refuses_out_of_bound_values(tmp_path, name, value, v13_val, match):
    """G4: finite, >= 1, and within 3,600 s of v13. Fails if any bound is loosened."""
    d = tmp_path / name
    d.mkdir()
    fw = pd.DataFrame({"MVT_ID_mvt": [1.0], "TAXITIME_SEC_mvt": [value]})
    fw.to_parquet(F.fw_output_path(d), index=False)
    with pytest.raises(AssertionError, match=match):
        F.fw_values_for([1.0], d, [v13_val])


def test_fw_values_for_accepts_in_bound_values_and_refuses_missing_ids(tmp_path):
    fw = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "TAXITIME_SEC_mvt": np.array([120, 3600 + 100], dtype="int32")})
    fw.to_parquet(F.fw_output_path(tmp_path), index=False)
    ok = F.fw_values_for([1.0, 2.0], tmp_path, [100.0, 100.0])   # id2 diff exactly 3,600: at the bound, passes
    assert ok.tolist() == [120.0, 3700.0]
    with pytest.raises(AssertionError, match="missing"):
        F.fw_values_for([1.0, 9.0], tmp_path, [100.0, 100.0])
    with pytest.raises(FileNotFoundError, match="G4"):
        F.fw_values_for([1.0], tmp_path / "nope", [100.0])


# ---------------------------------------------------------------- P formula

def test_projection_P_hand_example():
    """P = (rmse_inc^2 - rmse_cand^2) * n_rows / n_pop * |R| / board_n. Fails if any of the four terms is swapped,
    squared incorrectly, or the row-count normalisation drifts."""
    comp = {"n_rows": 1000, "n_pop": 100, "main_mean3": {"rmse_inc": 10.0, "rmse_cand": 8.0}}
    out = F.projection_P(comp, n_r=50, board_n=200.0)
    assert out["sse_gain_s2"] == pytest.approx(36_000.0)          # (100 - 64) * 1000
    assert out["per_pop_row_s2"] == pytest.approx(360.0)          # 36,000 / 100
    assert out["P"] == pytest.approx(90.0)                        # 360 * 50 / 200
    assert out["n_R"] == 50 and out["board_n"] == 200.0


def test_projection_P_zero_gain_is_zero():
    comp = {"n_rows": 1000, "n_pop": 100, "main_mean3": {"rmse_inc": 8.0, "rmse_cand": 8.0}}
    assert F.projection_P(comp, n_r=50)["P"] == 0.0


# ---------------------------------------------------------------- stop bar

def test_stop_reason_boundary_at_50():
    """The registered stop bar is P < 50 (strict): exactly 50 does NOT stop. Fails if the comparison becomes <=."""
    assert F.stop_reason(50.0) is None
    assert F.stop_reason(100.0) is None
    assert F.stop_reason(49.9) is not None and "49.9" in F.stop_reason(49.9)
    assert F.STOP_P == 50.0


# ---------------------------------------------------------------- pre-upload rules / move split reuse

def test_move_report_reuses_afj_move_split_and_duplicates_stage_only():
    """Reuses afj_ship_build.move_split (bounds tested there); FWP has no separate stage-only decomposition, so
    `stage_only` must equal `total` (move passed twice). Fails if the wrong array is passed or a bound is loosened
    locally instead of deferring to move_split."""
    move = np.array([50.0] * 9 + [700.0])            # the AFJ test's own fails-both fixture
    out = F.move_report(move)
    assert out["rules"] == {"concentration_gt300_le_47_8pct": False, "cap_rows_gt600_le_0_19pct": False, "recommendation": "do not upload"}
    assert out["stage_only"] == out["total"]
    ok = F.move_report(np.array([40.0, -60.0, 120.0, -30.0]))
    assert ok["rules"]["recommendation"] == "owner decision"


def test_per_airport_mean_move():
    r = pd.DataFrame({"airport": ["EDDF", "EDDF", "EDDM"]})
    out = F.per_airport_mean_move(r, [10.0, 20.0, -5.0])
    assert out == {"EDDF": {"n": 2, "mean_move_s": 15.0}, "EDDM": {"n": 1, "mean_move_s": -5.0}}


# ---------------------------------------------------------------- G1

def test_g1_checks_reports_and_cross_checks_v19_against_its_own_manifest(tmp_path, monkeypatch):
    v13_p, _ = _write_v13(tmp_path, [1.0], [100], "lightgbm:delta:mean3")
    monkeypatch.setattr(F, "V13_SHA", F.sha256(v13_p))
    v19_p = tmp_path / "merry-quicksand_v19.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0], "TAXITIME_SEC_mvt": np.array([100], dtype="int32")}).to_parquet(v19_p, index=False)
    man_p = tmp_path / "v19.manifest.json"
    man_p.write_text(json.dumps({"outputs": {"submissions/merry-quicksand_v19.parquet": F.sha256(v19_p)}}))
    c_p = tmp_path / "c.npy"
    np.save(c_p, np.array([1.0]))
    out = F.g1_checks(v13_p, v19_p, man_p, c_p)
    assert out["v13_sha256"] == F.sha256(v13_p) and out["v19_sha256"] == F.sha256(v19_p)
    assert out["c_sha256_recorded_only"] == F.sha256(c_p)
    bad_man = tmp_path / "bad.manifest.json"
    bad_man.write_text(json.dumps({"outputs": {"submissions/merry-quicksand_v19.parquet": "0" * 64}}))
    with pytest.raises(AssertionError, match="G1"):
        F.g1_checks(v13_p, v19_p, bad_man, c_p)


def test_g1_checks_refuses_a_drifted_v13(tmp_path):
    v13_p, _ = _write_v13(tmp_path, [1.0], [100], "lightgbm:delta:mean3")     # V13_SHA is NOT patched here
    with pytest.raises(AssertionError, match="G1"):
        F.g1_checks(v13_p, v13_p, v13_p, v13_p)


# ---------------------------------------------------------------- synthetic end-to-end (splice)

def _e2e_world(tmp_path):
    """6 scored matched rows: ids 1,2,4,5 are the frozen v13-stage scope AND stay unmoved by any later lever (in R);
    id 3 is the same scope-stage but v19 already differs from the CBV formula (a later lever touched it: excluded);
    id 6 carries the `+adsb` stage (matched but never scope: the "row in another stage" case)."""
    ids = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    proxy = {1: 900.0, 2: 700.0, 3: 1000.0, 4: 500.0, 5: 2000.0, 6: 300.0}
    c = {1: 300.0, 2: 600.0, 3: 970.0, 4: 440.0, 5: 1900.0, 6: 200.0}
    v13v = {1: 400, 2: 1, 3: 50, 4: 60, 5: 100, 6: 650}
    # cbv_candidate = rint(0.5*taxi_C + 0.5*v13): id1 500, id2 50, id3 40 (v19 set to 999, excluded), id4 60, id5 100
    v19v = {1: 500, 2: 50, 3: 999, 4: 60, 5: 100, 6: 777}
    stage = {1: "lightgbm:delta:mean3", 2: "lightgbm:delta:mean3", 3: "lightgbm:delta:mean3",
             4: "lightgbm:delta:mean3", 5: "lightgbm:delta:mean3", 6: "lightgbm:delta:mean3+adsb"}
    airport = {1: "EDDF", 2: "EDDF", 3: "EDDF", 4: "EDDM", 5: "EDDM", 6: "EDDF"}
    fw = {1: 420.0, 2: 21.0, 3: 999.0, 4: 90.0, 5: 140.0, 6: 700.0}   # within 3,600 s of v13 on every row

    d = tmp_path
    ship = pd.DataFrame({"MVT_ID_mvt": ids, "score_proxy": [proxy[i] for i in ids], "holdout": True,
                        "delta": np.nan})
    ship_p = d / "ship_input.parquet"
    ship.to_parquet(ship_p, index=False)
    c_p = d / "c.npy"
    np.save(c_p, np.array([c[i] for i in ids]))
    v13_p = d / "v13.parquet"
    pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": np.array([v13v[i] for i in ids], dtype="int32")}).to_parquet(v13_p, index=False)
    prov_p = d / "v13.prov.parquet"
    pd.DataFrame({"MVT_ID_mvt": ids, "airport": [airport[i] for i in ids], "stage": [stage[i] for i in ids]}).to_parquet(prov_p, index=False)
    v19_p = d / "merry-quicksand_v19.parquet"
    pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": np.array([v19v[i] for i in ids], dtype="int32")}).to_parquet(v19_p, index=False)
    man_p = d / "v19.manifest.json"
    man_p.write_text(json.dumps({"outputs": {"submissions/merry-quicksand_v19.parquet": F.sha256(v19_p)}}))
    repro_dir = d / "repro"
    repro_dir.mkdir()
    repro = pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": np.array([v13v[i] for i in ids], dtype="int32")})   # exact G2 reproduction
    repro.to_parquet(F.fw_output_path(repro_dir), index=False)
    fw_dir = d / "fw"
    fw_dir.mkdir()
    fw_out = pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": np.array([fw[i] for i in ids], dtype="int32")})
    fw_out.to_parquet(F.fw_output_path(fw_dir), index=False)
    return dict(ship_input_path=ship_p, c_path=c_p, v13_sub_path=v13_p, v13_prov_path=prov_p, v19_path=v19_p,
                v19_manifest_path=man_p, repro_dir=repro_dir, fw_dir=fw_dir)


def _run_repro_check_for(world, monkeypatch):
    monkeypatch.setattr(F.CBV, "STAGE_COUNTS", {"lightgbm:delta:mean3": 5, "lightgbm:delta:mean3+adsb": 1})
    v = F.repro_check(world["repro_dir"], v13_sub_path=world["v13_sub_path"], v13_prov_path=world["v13_prov_path"])
    (world["repro_dir"] / F.G2_VERDICT_NAME).write_text(json.dumps(v))
    assert v["identical"] is True
    return v


@pytest.mark.devtree
def test_build_splice_writes_a_candidate_and_never_touches_submissions(tmp_path, monkeypatch):
    """Full synthetic build: G1/G2/G6 pass, R = {1,2,4,5} (id3 excluded: a later lever; id6 excluded: another
    stage), the FW/CBV blend is written only on R, id6 (never in scope) is untouched, the candidate re-reads
    byte-identical, and nothing under submissions/ is created."""
    world = _e2e_world(tmp_path)
    monkeypatch.setattr(F, "V13_SHA", F.sha256(world["v13_sub_path"]))
    _run_repro_check_for(world, monkeypatch)
    runs_dir = tmp_path / "pipeline_runs"
    runs_dir.mkdir()                     # production RUNS (data/pipeline_runs/) always pre-exists; the fixture matches that
    comp_p = tmp_path / "comp.json"
    comp_p.write_text(json.dumps({"n_rows": 100, "n_pop": 10, "main_mean3": {"rmse_inc": 10.0, "rmse_cand": 8.0}}))
    result = F.build_splice(world["fw_dir"], ship_input_path=world["ship_input_path"], c_path=world["c_path"],
                            v13_sub_path=world["v13_sub_path"], v13_prov_path=world["v13_prov_path"],
                            v19_path=world["v19_path"], v19_manifest_path=world["v19_manifest_path"],
                            comp_json_path=comp_p, repro_dir=world["repro_dir"], runs_dir=runs_dir, board_n=10.0)
    assert result["stopped"] is False
    assert result["scope"] == {"n_scope": 5, "n_R": 4, "n_excluded_later_lever": 1}
    assert result["projection"]["P"] == pytest.approx(144.0)      # 360 per-pop-row * 4 / 10
    assert result["by_airport"] == {"EDDF": {"n": 2, "mean_move_s": 10.0}, "EDDM": {"n": 2, "mean_move_s": 17.5}}
    out_dir = ROOT / result["out_dir"] if not Path(result["out_dir"]).is_absolute() else Path(result["out_dir"])
    cand = pd.read_parquet(out_dir / "merry-quicksand_fwp.parquet")
    got = cand.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    assert got.loc[1.0] == 510 and got.loc[2.0] == 60 and got.loc[4.0] == 75 and got.loc[5.0] == 120
    assert got.loc[3.0] == 999                 # excluded from R: untouched
    assert got.loc[6.0] == 777                 # never in scope (another stage): untouched
    manifest = json.loads((out_dir / "fwp_manifest.json").read_text())
    assert manifest["scope"] == {"n_scope": 5, "n_R": 4, "n_excluded_later_lever": 1}
    assert manifest["upload"].startswith("NOT AUTHORISED")
    assert not (ROOT / "submissions" / "merry-quicksand_fwp.parquet").exists()
    assert not any(p.name.startswith("merry-quicksand_fwp") for p in (ROOT / "submissions").iterdir())


def test_build_splice_stops_and_writes_nothing_when_P_below_50(tmp_path, monkeypatch):
    world = _e2e_world(tmp_path)
    monkeypatch.setattr(F, "V13_SHA", F.sha256(world["v13_sub_path"]))
    _run_repro_check_for(world, monkeypatch)
    runs_dir = tmp_path / "pipeline_runs"
    comp_p = tmp_path / "comp.json"
    comp_p.write_text(json.dumps({"n_rows": 100, "n_pop": 10, "main_mean3": {"rmse_inc": 8.0, "rmse_cand": 8.0}}))
    result = F.build_splice(world["fw_dir"], ship_input_path=world["ship_input_path"], c_path=world["c_path"],
                            v13_sub_path=world["v13_sub_path"], v13_prov_path=world["v13_prov_path"],
                            v19_path=world["v19_path"], v19_manifest_path=world["v19_manifest_path"],
                            comp_json_path=comp_p, repro_dir=world["repro_dir"], runs_dir=runs_dir)
    assert result["stopped"] is True and "STOP" in result["reason"]
    assert not runs_dir.exists()


def test_build_splice_refuses_when_g2_verdict_missing_or_failed(tmp_path, monkeypatch):
    world = _e2e_world(tmp_path)
    monkeypatch.setattr(F, "V13_SHA", F.sha256(world["v13_sub_path"]))
    comp_p = tmp_path / "comp.json"
    comp_p.write_text(json.dumps({"n_rows": 100, "n_pop": 10, "main_mean3": {"rmse_inc": 10.0, "rmse_cand": 8.0}}))
    with pytest.raises(AssertionError, match="G2"):
        F.build_splice(world["fw_dir"], ship_input_path=world["ship_input_path"], c_path=world["c_path"],
                       v13_sub_path=world["v13_sub_path"], v13_prov_path=world["v13_prov_path"],
                       v19_path=world["v19_path"], v19_manifest_path=world["v19_manifest_path"],
                       comp_json_path=comp_p, repro_dir=world["repro_dir"], runs_dir=tmp_path / "pipeline_runs")
    (world["repro_dir"] / F.G2_VERDICT_NAME).write_text(json.dumps({"identical": False, "n_diff": 3}))
    with pytest.raises(AssertionError, match="NOT-identical"):
        F.build_splice(world["fw_dir"], ship_input_path=world["ship_input_path"], c_path=world["c_path"],
                       v13_sub_path=world["v13_sub_path"], v13_prov_path=world["v13_prov_path"],
                       v19_path=world["v19_path"], v19_manifest_path=world["v19_manifest_path"],
                       comp_json_path=comp_p, repro_dir=world["repro_dir"], runs_dir=tmp_path / "pipeline_runs")
