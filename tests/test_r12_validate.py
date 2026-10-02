"""Tests for scripts/analysis/r12_validate.py — the R12 validation harness (coverage-reweighted holdout, the
forward-chronological December check, comparators, candidate configs).

Synthetic fixtures only in this process; the fit is injected (no LightGBM fit here). Mutation rehearsals are recorded
in the docstrings ("Fails when ...").
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("r12_validate", "scripts/analysis/r12_validate.py")
B = R.B


# ---------------------------------------------------------------- cells and target shares

def test_cell_keys_month_and_pooled():
    ap, m = np.array(["EGLL", "EHAM"]), np.array([1, 7])
    assert R.cell_keys(ap, m).tolist() == ["EGLL|1", "EHAM|7"]
    assert R.cell_keys(ap, m, pooled=True).tolist() == ["EGLL", "EHAM"]


def _serve():
    return pd.DataFrame({"ap": ["EGLL"] * 4 + ["EHAM"] * 2, "month": [1, 1, 7, 7, 1, 1],
                         "coverage": [3, 0, 3, 3, 2, 1]})


def test_target_shares_by_month_and_pooled():
    """Fails when the share counts coverage >= 2 (or any coverage) instead of coverage == 3."""
    t = R.target_shares(_serve(), pooled=False)
    assert t == {"EGLL|1": 0.5, "EGLL|7": 1.0, "EHAM|1": 0.0}
    p = R.target_shares(_serve(), pooled=True)
    assert p == {"EGLL": 0.75, "EHAM": 0.0}


# ---------------------------------------------------------------- weights

def test_coverage_weights_match_the_target_share_and_keep_cell_mass():
    """In each cell the weighted coverage-3 share equals the target and the weights sum to the cell's row count.
    Fails when the non-cov3 weight (1-p26)/(1-p25) is replaced by 1 (the cell mass then drifts)."""
    keys = np.array(["A|1"] * 10 + ["B|1"] * 4)
    cov3 = np.array([1, 1] + [0] * 8 + [1, 0, 0, 0])
    w, cells = R.coverage_weights(keys, cov3, {"A|1": 0.6, "B|1": 0.25})
    a, b = keys == "A|1", keys == "B|1"
    assert abs(w[a].sum() - 10) < 1e-9 and abs(w[b].sum() - 4) < 1e-9        # exact arithmetic up to fp rounding
    assert abs((w[a] * cov3[a]).sum() / w[a].sum() - 0.6) < 1e-12
    np.testing.assert_allclose(w[b], 1.0)                                       # 0.25 -> 0.25: identity
    assert cells["A|1"]["p_source"] == 0.2 and cells["A|1"]["p_target"] == 0.6
    assert cells["A|1"]["status"] == "reweighted" and cells["A|1"]["n_eff"] < 10


def test_coverage_weights_unrepresentable_cells_are_flagged_not_faked():
    """p_source 0 with p_target > 0 (LEMD/LIRF January in the v4 tables): no row can carry the target share -> weight
    1 and status 'unrepresentable'. Same for p_source 1 with p_target < 1. Fails when such a cell divides by zero or
    is silently reported as matched."""
    keys = np.array(["Z|1"] * 3 + ["O|1"] * 2 + ["N|1"] * 2)
    cov3 = np.array([0, 0, 0, 1, 1, 0, 0])
    w, cells = R.coverage_weights(keys, cov3, {"Z|1": 0.24, "O|1": 0.5, "N|1": 0.0})
    np.testing.assert_allclose(w, 1.0)
    assert cells["Z|1"]["status"] == "unrepresentable" and cells["O|1"]["status"] == "unrepresentable"
    assert cells["N|1"]["status"] == "matched"
    assert np.isfinite(w).all()


def test_coverage_weights_refuse_a_cell_with_no_target():
    with pytest.raises(KeyError, match="no 2026 target share for cell"):
        R.coverage_weights(np.array(["Q|7"]), np.array([1]), {"Q|1": 0.5})


def test_coverage_weights_smoke_may_lack_a_target_cell():
    """Smoke tables hold a few days, so some cells have no 2026 target: weight 1, status no_target. Real runs refuse."""
    w, cells = R.coverage_weights(np.array(["Q|7", "Q|7"]), np.array([1, 0]), {}, allow_missing=True)
    np.testing.assert_allclose(w, 1.0)
    assert cells["Q|7"]["status"] == "no_target" and cells["Q|7"]["p_target"] is None


def test_coverage_weights_refuse_bad_inputs():
    with pytest.raises(ValueError, match="cov3 must be 0/1"):
        R.coverage_weights(np.array(["A|1"]), np.array([2]), {"A|1": 0.5})
    with pytest.raises(ValueError, match="length"):
        R.coverage_weights(np.array(["A|1", "A|1"]), np.array([1]), {"A|1": 0.5})


def test_weighted_rmse():
    y, p = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    assert R.weighted_rmse(y, p) == pytest.approx(np.sqrt(12.5), abs=1e-12)          # exact up to fp
    assert R.weighted_rmse(y, p, np.array([1.0, 0.0])) == pytest.approx(3.0, abs=1e-12)
    with pytest.raises(ValueError, match="non-finite"):
        R.weighted_rmse(y, np.array([np.nan, 1.0]))


# ---------------------------------------------------------------- paired gain

def test_paired_gain_sign_units_and_ci():
    """gain = (SSE_comparator - SSE_candidate) / 344,841: positive = the candidate is better. Fails when the sign or
    the unit (SSE/344,841, the board-MSE unit) is changed."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(500, 100, n)
    comp = y + rng.normal(0, 50, n)
    cand = y + rng.normal(0, 20, n)
    blocks = np.repeat(np.arange(40), 10)
    g = R.paired_gain(y, cand, comp, blocks, n_boot=300, seed=1)
    want = (((comp - y) ** 2).sum() - ((cand - y) ** 2).sum()) / 344_841
    assert g["gain_bmse"] == pytest.approx(want, rel=1e-12)
    assert g["ci95"][0] < g["gain_bmse"] < g["ci95"][1] and g["ci95"][0] > 0
    assert g["n"] == n and g["n_blocks"] == 40
    worse = R.paired_gain(y, comp, cand, blocks, n_boot=300, seed=1)
    assert worse["gain_bmse"] == pytest.approx(-want, rel=1e-12)


def test_paired_gain_weighted_and_nan_comparator_rows_dropped():
    y = np.array([0.0, 0.0, 0.0])
    cand, comp = np.array([1.0, 1.0, 1.0]), np.array([2.0, np.nan, 3.0])
    g = R.paired_gain(y, cand, comp, np.array([0, 1, 2]), w=np.array([2.0, 1.0, 1.0]), n_boot=50, seed=0)
    assert g["n"] == 2
    assert g["gain_bmse"] == pytest.approx((2 * (4 - 1) + 1 * (9 - 1)) / 344_841, rel=1e-12)


# ---------------------------------------------------------------- attaching the ADS-B table to a load_fold frame

def _table(months_counts, cov=3):
    rows = []
    for m, n in months_counts:
        for p in range(n):
            rows.append({"month": m, "pos": p, "MVT_ID_mvt": float(m * 1000 + p), "ap": "EGLL", "mvt_day": f"2025-{m:02d}-02",
                         "adsb_day_present": 1, "coverage": cov})
    t = pd.DataFrame(rows)
    for c in B.FEATURES:
        t[c] = t.MVT_ID_mvt * 2.0
    return t[B.TABLE_COLS]


def test_month_positions_and_non_monotone_refusal():
    assert R.month_positions(np.array([1, 1, 2, 2, 2])).tolist() == [0, 1, 0, 1, 2]
    with pytest.raises(ValueError, match="not in cache-file order"):
        R.month_positions(np.array([1, 2, 1]))


def test_attach_adsb_joins_by_month_and_position():
    """Fails when the join uses the frame's global index instead of (month, position in the month file)."""
    d = pd.DataFrame({"month": [2, 2, 3], "x": [0, 0, 0]})
    t = _table([(1, 5), (2, 2), (3, 1)])
    info = R.attach_adsb(d, t, strict=True)
    assert d.dwell_s.tolist() == [4000.0, 4002.0, 6000.0]
    assert d.coverage.tolist() == [3, 3, 3] and info["n_without_table"] == 0
    assert info["pos"].tolist() == [0, 1, 0]
    k = info["keys"]
    assert list(k.columns) == ["month", "pos", "MVT_ID_mvt", "mvt_day", "coverage"]
    assert k.MVT_ID_mvt.tolist() == [2000.0, 2001.0, 3000.0]


def test_attach_adsb_strict_refuses_missing_rows_and_count_mismatch():
    """Real runs are strict: every frame row needs a table row, and each month's row count must equal the table's."""
    d = pd.DataFrame({"month": [2, 2, 2]})
    with pytest.raises(ValueError, match="month 2: frame has 3 rows, table 2"):
        R.attach_adsb(d, _table([(2, 2)]), strict=True)
    d = pd.DataFrame({"month": [4]})
    with pytest.raises(ValueError, match="month 4"):
        R.attach_adsb(d, _table([(2, 2)]), strict=True)


def test_attach_adsb_non_strict_marks_rows_without_a_table():
    """Smoke only: rows the table does not cover get NaN features, adsb_day_present 0, coverage NaN — counted."""
    d = pd.DataFrame({"month": [1, 1, 1]})
    t = _table([(1, 1)])
    info = R.attach_adsb(d, t, strict=False)
    assert info["n_without_table"] == 2
    assert d.adsb_day_present.tolist() == [1, 0, 0]
    assert np.isnan(d.dwell_s.iloc[1]) and np.isnan(d.coverage.iloc[2])


def test_rows_to_month_pos():
    """Step-1 record rows are global indices over the 12 concatenated months."""
    counts = {1: 3, 2: 2, 3: 4}
    m, p = R.rows_to_month_pos(np.array([0, 2, 3, 4, 5, 8]), counts)
    assert m.tolist() == [1, 1, 2, 2, 3, 3] and p.tolist() == [0, 2, 0, 1, 0, 3]
    with pytest.raises(ValueError, match="out of range"):
        R.rows_to_month_pos(np.array([9]), counts)


def test_cache_month_counts_reads_file_metadata(tmp_path):
    """The step-1 record's global rows map through the CACHE files' row counts (a partial R12 table must not be used).
    Fails when the counts come from anything but the cache files."""
    for m, n in ((1, 3), (2, 5)):
        pd.DataFrame({"month": [m] * n}).to_parquet(tmp_path / f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet")
    assert R.cache_month_counts(tmp_path) == {1: 3, 2: 5}
    with pytest.raises(FileNotFoundError):
        R.cache_month_counts(tmp_path / "none")


def _hold():
    return pd.DataFrame({"MVT_ID_mvt": [11.0, 12.0, 13.0], "month": [1, 1, 7], "pos": [0, 1, 0],
                         "y": [100.0, 200.0, 300.0]})


def test_regression_bug_r12_1_join_comparators_reads_the_pipe_column():
    """2026-09-25 smoke: `lane.pipe` resolved to DataFrame.pipe (a METHOD), AttributeError in real_comparators
    (scripts/analysis/r12_validate.py, the untested real-artefact wrapper). Bug class: attribute access on a column
    whose name shadows a DataFrame method. The join is now a pure, tested function. Fails when the column is read by
    attribute or when rows are joined by position instead of MVT id / (month, pos)."""
    lane = pd.DataFrame({"MVT_ID_mvt": [13.0, 11.0], "y": [300.0, 100.0], "pipe": [310.0, 90.0]})
    rec = pd.DataFrame({"row": [0, 1, 2], "y": [100.0, 200.0, 300.0], "proxy": [600.0, 700.0, 800.0],
                        "treatment": [450.0, 480.0, 490.0]})
    out = R.join_comparators(_hold(), lane, rec, {1: 2, 7: 1})
    np.testing.assert_array_equal(out["PIPE_M"], [90.0, np.nan, 310.0])
    np.testing.assert_array_equal(out["F_NSC_step1"], [150.0, 220.0, 310.0])


def test_join_comparators_expected_pipe_rows_tripwire():
    """A real holdout run must join PIPE_M on all of lane M (334,132 rows); fewer means the ids are incomplete."""
    lane = pd.DataFrame({"MVT_ID_mvt": [13.0, 11.0], "y": [300.0, 100.0], "pipe": [310.0, 90.0]})
    rec = pd.DataFrame({"row": [0], "y": [100.0], "proxy": [600.0], "treatment": [450.0]})
    R.join_comparators(_hold(), lane, rec, {1: 2, 7: 1}, expect_pipe_n=2)
    with pytest.raises(AssertionError, match="PIPE_M joined 2 test rows, expected 3"):
        R.join_comparators(_hold(), lane, rec, {1: 2, 7: 1}, expect_pipe_n=3)


def test_join_comparators_refuses_misaligned_labels():
    lane = pd.DataFrame({"MVT_ID_mvt": [11.0], "y": [999.0], "pipe": [90.0]})
    rec = pd.DataFrame({"row": [0], "y": [100.0], "proxy": [600.0], "treatment": [450.0]})
    with pytest.raises(AssertionError, match="PIPE_M y differs"):
        R.join_comparators(_hold(), lane, rec, {1: 2, 7: 1})
    lane = pd.DataFrame({"MVT_ID_mvt": [11.0], "y": [100.0], "pipe": [90.0]})
    rec = pd.DataFrame({"row": [0], "y": [555.0], "proxy": [600.0], "treatment": [450.0]})
    with pytest.raises(AssertionError, match="step-1 record y differs"):
        R.join_comparators(_hold(), lane, rec, {1: 2, 7: 1})


def test_frame_scope():
    """Real: every month + every ranking row. Smoke: 3 months + 5,000 ranking rows. Full-frame smoke (the memory
    measurement / 1-step dry run): the real frame with smoke trees. Full-frame without smoke is refused (meaningless)."""
    assert R.frame_scope("holdout", smoke=False, full_frame=False) == (None, None)
    assert R.frame_scope("holdout", smoke=True, full_frame=False) == ((1, 2, 3), R.LF.L.SMOKE["n_rank"])
    assert R.frame_scope("forward", smoke=True, full_frame=False) == ((3, 11, 12), R.LF.L.SMOKE["n_rank"])
    assert R.frame_scope("forward", smoke=True, full_frame=True) == (None, None)
    with pytest.raises(ValueError, match="full-frame"):
        R.frame_scope("holdout", smoke=False, full_frame=True)


# ---------------------------------------------------------------- splits and designs

def test_split_forward_is_feb_to_nov_without_jan_jul_testing_december():
    """Fails when January / July rows leak into the fit (they are the holdout) or December into training."""
    month = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12])
    te, tr, fit, es = R.split_forward(month)
    assert month[te].tolist() == [12, 12]
    assert sorted(set(month[tr])) == [2, 3, 4, 5, 6, 8, 9, 10, 11]
    assert sorted(set(month[es])) == [3, 9] and sorted(set(month[fit])) == [2, 4, 5, 6, 8, 10, 11]
    assert not (te & tr).any() and not (fit & es).any()


def test_split_holdout_is_fold_a():
    month = np.array([1, 2, 3, 7, 9, 12])
    te, tr, fit, es = R.split_holdout(month)
    assert month[te].tolist() == [1, 7] and month[es].tolist() == [3, 9] and month[fit].tolist() == [2, 12]


def test_designs_have_no_seconds_and_differ_only_by_the_adsb_block():
    r12, ctl = R.design_for("r12"), R.design_for("control")
    assert r12[:len(ctl)] == ctl and r12[len(ctl):] == R.ADSB_COLS
    assert not [c for c in r12 if c.endswith("_sec")]
    assert R.design_for("yhead") == r12
    with pytest.raises(ValueError, match="unknown arm"):
        R.design_for("nope")


# ---------------------------------------------------------------- serve rows, A' masking, sensitivity

def _serve_table(ids, month=1, cov=3, ap="LFPG"):
    t = pd.DataFrame({"month": month, "pos": np.arange(len(ids)), "MVT_ID_mvt": np.asarray(ids, float), "ap": ap,
                      "mvt_day": "2026-01-02", "adsb_day_present": 1, "coverage": cov})
    for c in B.FEATURES:
        t[c] = t.MVT_ID_mvt * 3.0
    return t[B.TABLE_COLS]


def test_adsb_by_id_joins_serve_rows_by_mvt_id():
    """Ranking rows may be a sample (lgbm_submit smoke) and arrive in any order: joined by id, not position."""
    j = R.adsb_by_id(np.array([7.0, 5.0]), _serve_table([5.0, 6.0, 7.0]), strict=True)
    assert j.dwell_s.tolist() == [21.0, 15.0] and j.pos.tolist() == [2, 0]
    with pytest.raises(ValueError, match="1 serve ids have no table row"):
        R.adsb_by_id(np.array([9.0]), _serve_table([5.0]), strict=True)
    k = R.adsb_by_id(np.array([9.0]), _serve_table([5.0]), strict=False)
    assert k.adsb_day_present.tolist() == [0.0] and np.isnan(k.dwell_s.iloc[0])


def test_attach_all_training_then_ranking_rows():
    """Fails when ranking rows are joined through the training table's (month, pos) (their month 1/7 would collide
    with January / July 2025) or when ranking rows are not last."""
    d = pd.DataFrame({"month": [1, 1, 2, 1]})
    is_rank = np.array([False, False, False, True])
    info = R.attach_all(d, is_rank, np.array([5.0]), _table([(1, 2), (2, 1)]), _serve_table([5.0]), strict=True)
    assert d.dwell_s.tolist() == [2000.0, 2002.0, 4000.0, 15.0]
    assert info["keys"].MVT_ID_mvt.tolist() == [1000.0, 1001.0, 2000.0, 5.0]
    with pytest.raises(ValueError, match="ranking rows must come last"):
        R.attach_all(pd.DataFrame({"month": [1, 1]}), np.array([True, False]), np.array([5.0]), _table([(1, 1)]),
                     _serve_table([5.0]), strict=True)


def test_eobt_cols_are_the_eobt_derived_inputs():
    assert R.EOBT_COLS == ["eobt_p", "eobt_sched", "aobt_eobt", "eobt_lobt"]
    assert set(R.EOBT_COLS) <= set(R.BASE)


def test_mask_eobt_only_at_the_named_airports_and_copies():
    """A': the EOBT-derived columns are NaN on LFPG/LTFM rows (fit, stop, test AND serve), untouched elsewhere; the
    input matrix is not modified in place. Fails when the mask hits every airport or edits X in place."""
    feats = R.design_for("r12")
    X = np.ones((4, len(feats)), dtype=np.float32)
    ap_code, airports = np.array([0, 1, 2, 1]), ["EGLL", "LFPG", "LTFM"]
    X2, n = R.mask_eobt(X, feats, ap_code, airports)
    idx = [feats.index(c) for c in R.EOBT_COLS]
    other = [i for i in range(len(feats)) if i not in idx]
    assert n == 3 and np.isnan(X2[1:, idx]).all() and (X2[0, idx] == 1).all()
    assert (X2[:, other] == 1).all() and (X == 1).all()
    assert R.design_for("r12_noeobt") == feats


def test_mask_janjul_blanks_the_adsb_block_outside_janjul_training_rows():
    """janjul_adsb = today's situation: ADS-B only on January / July 2025 training rows (and on the 2026 serve rows),
    NaN + adsb_day_present 0 on every other training row; X itself untouched. Fails when ranking rows are blanked,
    when Jan/Jul rows are blanked, or when the flag is not set."""
    feats = R.design_for("r12")
    X = np.ones((5, len(feats)), dtype=np.float32)
    month = np.array([1, 3, 7, 12, 7])
    rank = np.array([False, False, False, False, True])
    X2, n = R.mask_janjul(X, feats, month, rank)
    a = [feats.index(c) for c in R.ADSB_COLS if c != "adsb_day_present"]
    flag = feats.index("adsb_day_present")
    assert n == 2
    assert np.isnan(X2[[1, 3]][:, a]).all() and (X2[[1, 3], flag] == 0).all()
    assert (X2[[0, 2, 4]] == 1).all() and (X == 1).all()
    assert R.design_for("janjul_adsb") == feats


def test_mask_janjul_never_blanks_ranking_rows_even_with_an_unknown_month():
    """A serve row absent from a (smoke) table has month NaN; it is a ranking row and keeps its inputs (mutation
    rehearsal #17: the rank exemption was untested). Fails when the exemption is dropped."""
    feats = R.design_for("r12")
    X = np.ones((2, len(feats)), dtype=np.float32)
    X2, n = R.mask_janjul(X, feats, np.array([3.0, np.nan]), np.array([False, True]))
    assert n == 1 and (X2[1] == 1).all()


def test_fit_config_screen_and_smoke():
    """--screen: learning rate 0.05 with the production tree budget and patience; --smoke keeps its 60-tree values."""
    s = R.fit_config(smoke=False, screen=True)
    assert s["params"]["learning_rate"] == 0.05 and s["nest"] == R.LF.L.NEST and s["patience"] == R.LF.L.PATIENCE
    p = R.fit_config(smoke=False)
    assert p["params"]["learning_rate"] == R.LF.L.P["learning_rate"]
    assert R.fit_config(smoke=True, screen=True)["nest"] == R.LF.SMOKE["nest"]


def test_seeds_for_screen_and_real():
    """--screen fits one seed (0) unless --seeds says otherwise; a real run fits 0, 1, 2."""
    assert R.seeds_for(None, screen=True) == (0,)
    assert R.seeds_for(None, screen=False) == (0, 1, 2)
    assert R.seeds_for("0,1", screen=True) == (0, 1)
    with pytest.raises(ValueError, match="seeds"):
        R.seeds_for("a", screen=False)


@pytest.mark.parametrize("arg", ["", " , ", ","])
def test_seeds_for_refuses_an_empty_seed_list(arg):
    """An empty --seeds would fit no seed and report an empty arm. Mutation: return () again -> red."""
    with pytest.raises(ValueError, match="no seed"):
        R.seeds_for(arg, screen=False)


def test_split_refit_trains_every_month_and_tests_none():
    month = np.array([1, 2, 3, 7, 9, 12])
    te, tr, fit, es = R.split_refit(month)
    assert not te.any() and tr.all() and month[es].tolist() == [3, 9] and month[fit].tolist() == [1, 2, 7, 12]


def test_prediction_shift_cells_and_values():
    ref = pd.DataFrame({"ap": ["LFPG"] * 4, "month": [7] * 4, "taxi": [100.0, 200.0, 300.0, 400.0],
                        "proxy": [500.0] * 4})
    new = pd.DataFrame({"ap": ["LFPG"] * 2, "month": [7] * 2, "taxi": [400.0, 600.0], "proxy": [700.0] * 2})
    out = R.prediction_shift(ref, new, pooled=False)
    c = out["LFPG|7"]
    assert c["n_2025"] == 4 and c["n_2026"] == 2
    assert c["mean_taxi_2025"] == 250.0 and c["mean_taxi_2026"] == 500.0 and c["shift_mean_taxi"] == 250.0
    assert c["shift_mean_proxy"] == 200.0
    pooled = R.prediction_shift(ref.assign(month=12), new, pooled=True)
    assert list(pooled) == ["LFPG"]


def test_input_range_share_outside_the_2025_training_range():
    """2026 rows outside [min, max] (and [p0.5, p99.5]) of the 2025 TRAINING rows at the same airport. Fails when the
    range is taken from all airports or from the 2026 rows themselves."""
    feats = ["aobt_eobt", "x"]
    X = np.array([[0.0, 0], [100, 0], [50, 0], [5000, 0],      # rows 0-2 LFPG training, row 3 EGLL training
                  [150, 0], [60, 0], [np.nan, 0]], dtype=np.float32)
    ap = np.array(["LFPG", "LFPG", "LFPG", "EGLL", "LFPG", "LFPG", "LFPG"])
    month = np.array([2, 3, 4, 2, 7, 7, 7])
    train = np.array([True, True, True, True, False, False, False])
    new = ~train
    r = R.input_range(X, feats, ["aobt_eobt"], train, new, ap, month, pooled=False)
    c = r["LFPG|7"]["aobt_eobt"]
    assert c["n_2026"] == 3 and c["share_nan_2026"] == pytest.approx(1 / 3)
    assert c["share_outside_minmax"] == pytest.approx(1 / 2)          # 150 > 100; the NaN row is not counted
    assert c["min_2025"] == 0.0 and c["max_2025"] == 100.0
    assert c["median_2025"] == 50.0 and c["median_2026"] == 105.0


def test_arm_differences_on_2026_and_2025_rows():
    ref = pd.DataFrame({"ap": ["LFPG", "LFPG"], "month": [7, 7]})
    out = R.arm_differences({"r12": np.array([10.0, 20.0]), "r12_noeobt": np.array([13.0, 16.0])}, ref, pooled=False,
                            pairs=[("r12", "r12_noeobt")])
    c = out["r12_minus_r12_noeobt"]["LFPG|7"]
    assert c["mean"] == pytest.approx(0.5) and c["rms"] == pytest.approx(np.sqrt((9 + 16) / 2)) and c["n"] == 2


# ---------------------------------------------------------------- the run, with an injected loader and fit

def _fake_fold(split, months=(1, 2, 3, 7, 9, 12), n=40, n_rank=20, seed=0):
    rng = np.random.default_rng(seed)
    month_tr = np.repeat(np.array(months), n)
    ap_tr = np.where(np.arange(len(month_tr)) % 2, "LFPG", "EGLL")
    rank_ids = 90_000.0 + np.arange(n_rank)
    month_rk = np.where(np.arange(n_rank) % 2, 7, 1)
    ap_rk = np.where(np.arange(n_rank) % 4 < 2, "LFPG", "EGLL")
    d = pd.DataFrame({"month": np.r_[month_tr, month_rk]})
    is_rank = np.r_[np.zeros(len(month_tr), bool), np.ones(n_rank, bool)]
    serve = _serve_table(rank_ids).assign(month=month_rk, ap=ap_rk)
    info = R.attach_all(d, is_rank, rank_ids, _table([(m, n) for m in months]), serve, strict=True)
    te, tr, fit, es = (np.r_[m, np.zeros(n_rank, bool)] for m in split(month_tr))
    N = len(d)
    proxy = rng.uniform(400, 900, N)
    y = np.r_[proxy[:len(month_tr)] - rng.uniform(0, 300, len(month_tr)), np.full(n_rank, np.nan)]
    feats = R.design_for("r12")
    X = rng.normal(size=(N, len(feats))).astype(np.float32)
    X[:, feats.index("doy")] = (np.arange(N) // 10 % 7 + 1).astype(np.float32)     # 7 days per month
    airports = ["EGLL", "LFPG"]
    ap_all = np.r_[ap_tr, ap_rk]
    ap_code = np.searchsorted(airports, ap_all)
    base = pd.DataFrame({"row": np.flatnonzero(te), "month": d.month.to_numpy()[te], "ap": ap_all[te], "y": y[te],
                         "proxy": proxy[te], "delta": (proxy - y)[te], "sp": 0.0})
    rank_base = pd.DataFrame({"month": month_rk, "ap": ap_rk, "proxy": proxy[is_rank], "MVT_ID_mvt": rank_ids})
    return dict(X=X, y=y, dlt=proxy - y, proxy=proxy, te=te, tr=tr, fit=fit, es=es, rank=is_rank, feats=feats,
                month=d.month.to_numpy(),
                base=base, rank_base=rank_base, ap_code=ap_code, airports=airports, _keys=info["keys"])


def _fake_fit(fold, X, params, nest, patience, pa_floor, seeds, per_airport, pa_trees, name, target="delta"):
    te = fold["te"]
    true = fold["dlt"][te] if target == "delta" else fold["y"][te]
    true = np.where(np.isfinite(true), true, 100.0 if target == "delta" else 500.0)
    off = 5.0 if "control" in name else (2.0 if "noeobt" in name else (3.0 if "janjul" in name else 1.0))
    return {"delta": true + off, "best_iter": 3, "n_ref": 4, "single": {s: true + off for s in seeds},
            "single_rmse": {s: off for s in seeds}}


def _serve_small():
    return pd.DataFrame({"ap": ["EGLL", "EGLL", "LFPG", "LFPG"], "month": [1, 7, 1, 7], "coverage": [3, 3, 3, 0]})


def test_run_holdout_writes_report_and_predictions(tmp_path):
    """End to end on a synthetic fold: arms r12/control/yhead/r12_noeobt, composition B, weights, comparators on common
    rows, the label-free 2026 sensitivity block. Observable state: predictions.parquet, predictions_2026.parquet and
    report.json written once; a second run into the same dir refuses."""
    loader = lambda feats, split: _fake_fold(split)

    def comps(hold):
        c = hold.y.to_numpy() + 3.0
        return {"PIPE_M": np.where(hold.month.to_numpy() == 1, c, np.nan), "NONE": np.full(len(hold), np.nan)}
    out = tmp_path / "logs" / "r12_scratch" / "v1"
    R.run("holdout", ["r12", "control", "yhead", "r12_noeobt"], smoke=True, serve_table=_serve_small(), out_dir=out,
          load=loader, fit=_fake_fit, comparators=comps, seeds=(0,), root=tmp_path, log=lambda m: None)
    back = json.loads((out / "report.json").read_text())
    assert back["split"] == "holdout" and back["smoke"] is True
    assert set(back["arms"]) == {"r12", "control", "yhead", "r12_noeobt", "B_blend_y"}
    assert back["arms"]["r12"]["rmse_unweighted"] == pytest.approx(1.0, abs=1e-9)   # fake fit: every row off by 1 s
    assert back["arms"]["control"]["rmse_unweighted"] == pytest.approx(5.0, abs=1e-9)
    g = back["pairs"]["r12_vs_control"]
    assert g["unweighted"]["gain_bmse"] > 0 and g["weighted"]["gain_bmse"] > 0
    assert back["pairs"]["r12_vs_PIPE_M"]["unweighted"]["n"] == 40          # only January rows have the comparator
    assert "r12_vs_NONE" not in back["pairs"] and "r12_vs_NONE" in back["pairs_skipped"]  # no finite value: listed
    assert "LFPG|7" in back["weights"]["cells"]
    sens = back["sensitivity_2026"]
    assert sens["prediction_shift"]["r12"]["LFPG|7"]["n_2026"] == 5
    assert sens["input_range"]["LFPG|7"]["aobt_eobt"]["n_2026"] == 5
    assert sens["arm_differences_2026"]["r12_minus_r12_noeobt"]["LFPG|7"]["mean"] == pytest.approx(1.0)
    assert back["n_rank"] == 20
    pr = pd.read_parquet(out / "predictions.parquet")
    assert {"month", "pos", "MVT_ID_mvt", "y", "proxy", "w", "coverage", "taxi_r12", "taxi_control", "taxi_yhead",
            "taxi_r12_noeobt", "taxi_B_blend_y", "comp_PIPE_M"} <= set(pr.columns)
    assert len(pr) == 80
    f = _fake_fold(R.split_holdout)
    np.testing.assert_allclose(pr.pred_r12.to_numpy(), f["dlt"][f["te"]] + 1.0)     # the raw delta the stack needs
    assert back["fits"]["yhead"]["target"] == "y" and "pred_yhead" in pr.columns
    p26 = pd.read_parquet(out / "predictions_2026.parquet")
    assert "pred_r12" in p26.columns
    assert len(p26) == 20 and "y" not in p26.columns and p26.MVT_ID_mvt.is_unique
    with pytest.raises(FileExistsError):
        R.run("holdout", ["r12"], smoke=True, serve_table=_serve_small(), out_dir=out, load=loader, fit=_fake_fit,
              comparators=comps, seeds=(0,), root=tmp_path, log=lambda m: None)


def test_run_day_blocks_come_from_the_frame_doy_not_the_table(tmp_path):
    """Bootstrap blocks = month * 1000 + doy of the frame (every row has it; the NSC/v34 pairing's day blocks). The
    2026-09-25 smoke fell back to one block per month for rows without a table row and gave a zero-width CI. Fails
    when blocks come from the table's mvt_day again."""
    out = tmp_path / "logs" / "r12_scratch" / "blocks"
    rep = R.run("holdout", ["r12", "control"], smoke=True, serve_table=_serve_small(), out_dir=out,
                load=lambda feats, split: _fake_fold(split), fit=_fake_fit, comparators=None, seeds=(0,),
                root=tmp_path, log=lambda m: None)
    f = _fake_fold(R.split_holdout)
    want = len(np.unique(f["month"][f["te"]] * 1000 + f["X"][f["te"], f["feats"].index("doy")].astype(int)))
    assert want == 8                                                                   # 2 test months x 4 fixture days
    assert rep["pairs"]["r12_vs_control"]["unweighted"]["n_blocks"] == want


def test_run_refuses_a_fold_with_another_design(tmp_path):
    """Fails when run trusts the loader's column order instead of checking it (control's column slice would be wrong)."""
    def loader(feats, split):
        f = _fake_fold(split)
        f["feats"] = list(reversed(f["feats"]))
        return f
    with pytest.raises(AssertionError, match="design"):
        R.run("holdout", ["r12"], smoke=True, serve_table=_serve_small(), out_dir=tmp_path / "logs" / "r12_scratch" / "z",
              load=loader, fit=_fake_fit, comparators=None, seeds=(0,), root=tmp_path, log=lambda m: None)


def test_run_refuses_ranking_rows_before_test_rows(tmp_path):
    """The te|rank prediction is split by position: every test row must precede every ranking row."""
    def loader(feats, split):
        f = _fake_fold(split)
        f["rank"] = np.roll(f["rank"], -len(f["rank"]) // 2)
        return f
    with pytest.raises(AssertionError, match="precede"):
        R.run("holdout", ["r12"], smoke=True, serve_table=_serve_small(), out_dir=tmp_path / "logs" / "r12_scratch" / "q",
              load=loader, fit=_fake_fit, comparators=None, seeds=(0,), root=tmp_path, log=lambda m: None)


def test_run_refuses_ranking_rows_before_a_test_row_without_overlap(tmp_path):
    """Ranking rows placed between January and July test rows (no row is both): the position split would hand
    July's predictions to ranking rows (mutation rehearsal #31). Fails when only the overlap is checked."""
    def loader(feats, split):
        f = _fake_fold(split)
        rank = np.zeros(len(f["rank"]), bool)
        rank[40:60] = True                                  # February training rows, after Jan, before Jul tests
        f["rank"] = rank
        assert not (f["te"] & rank).any()
        return f
    with pytest.raises(AssertionError, match="precede"):
        R.run("holdout", ["r12"], smoke=True, serve_table=_serve_small(), out_dir=tmp_path / "logs" / "r12_scratch" / "p",
              load=loader, fit=_fake_fit, comparators=None, seeds=(0,), root=tmp_path, log=lambda m: None)


def test_run_janjul_arm(tmp_path):
    out = tmp_path / "logs" / "r12_scratch" / "jj"
    rep = R.run("holdout", ["r12", "janjul_adsb"], smoke=True, serve_table=_serve_small(), out_dir=out,
                load=lambda feats, split: _fake_fold(split), fit=_fake_fit, comparators=None, seeds=(0,), root=tmp_path,
                log=lambda m: None)
    assert rep["arms"]["janjul_adsb"]["rmse_unweighted"] == pytest.approx(3.0, abs=1e-9)
    assert rep["fits"]["janjul_adsb"]["n_rows_adsb_masked"] > 0
    assert "r12_vs_janjul_adsb" in rep["pairs"]


def test_run_refit_predicts_2026_only(tmp_path):
    """split refit (the serve-side F): every 2025 month trains, nothing is scored, predictions_2026 is written."""
    out = tmp_path / "logs" / "r12_scratch" / "rf"
    rep = R.run("refit", ["r12"], smoke=True, serve_table=_serve_small(), out_dir=out,
                load=lambda feats, split: _fake_fold(split), fit=_fake_fit, comparators=None, seeds=(0,), root=tmp_path,
                log=lambda m: None)
    assert rep["n_test"] == 0 and rep["pairs"] == {} and rep["n_rank"] == 20
    assert not (out / "predictions.parquet").exists()
    p26 = pd.read_parquet(out / "predictions_2026.parquet")
    assert len(p26) == 20 and np.allclose(p26.pred_r12, 101.0)


def test_run_forward_pools_the_target_share_and_tests_december(tmp_path):
    out = tmp_path / "logs" / "r12_scratch" / "fwd"
    rep = R.run("forward", ["r12"], smoke=True, serve_table=_serve_small(), out_dir=out,
                load=lambda feats, split: _fake_fold(split), fit=_fake_fit, comparators=None, seeds=(0,), root=tmp_path,
                log=lambda m: None)
    assert rep["n_test"] == 40 and sorted(rep["weights"]["cells"]) == ["EGLL", "LFPG"]
    assert rep["weights"]["cells"]["LFPG"]["p_target"] == 0.5
    assert list(rep["sensitivity_2026"]["prediction_shift"]["r12"]) == ["EGLL", "LFPG"]


def test_run_refuses_an_unknown_split_and_outside_output(tmp_path):
    with pytest.raises(ValueError, match="split"):
        R.run("sideways", ["r12"], smoke=True, serve_table=_serve_small(), out_dir=tmp_path / "logs" / "r12_scratch" / "x",
              load=None, fit=None, comparators=None, seeds=(0,), root=tmp_path, log=lambda m: None)
    with pytest.raises(PermissionError, match="outside"):
        R.run("holdout", ["r12"], smoke=True, serve_table=_serve_small(), out_dir=tmp_path / "x", load=None, fit=None,
              comparators=None, seeds=(0,), root=tmp_path, log=lambda m: None)


def test_run_real_mode_requires_comparators_on_holdout(tmp_path):
    """A real (non-smoke) holdout run without the PIPE_M comparator is refused: the report would have no anchor."""
    with pytest.raises(ValueError, match="comparator"):
        R.run("holdout", ["r12"], smoke=False, serve_table=_serve_small(), out_dir=tmp_path / "logs" / "r12_scratch" / "y",
              load=None, fit=None, comparators=None, seeds=(0,), root=tmp_path, log=lambda m: None)


# ---------------------------------------------------------------- month-out splits for the LTFM de-icing test

def test_split_feb_holds_out_february_only_and_trains_jan_jul():
    """LDI: the February-2025 storm out of fold. Every other 2025 month trains (Jan / Jul too: production-like, the
    served refit sees them), stopping on 3 + 9. Fails when February leaks into fit / stopping, when another month is
    tested, or when Jan / Jul are dropped from training.
    Mutation rehearsal 2026-09-27: tr = all rows, split_feb on (1, 2),
    split_decmar on (12,), POOLED_SPLITS without feb, no-test / no-stop / no-fit refusals removed, decmar smoke months
    without a fit month: 8/8 KILLED. The ES-month filter mutant was equivalent (tr already excludes the test months);
    the filter was removed as redundant."""
    month = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    te, tr, fit, es = R.split_feb(month)
    assert month[te].tolist() == [2, 2]
    assert sorted(set(month[tr])) == [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert sorted(set(month[es])) == [3, 9] and sorted(set(month[fit])) == [1, 4, 5, 6, 7, 8, 10, 11, 12]
    assert not (te & tr).any() and not (fit & es).any() and ((fit | es) == tr).all()


def test_split_decmar_tests_december_and_march_with_february_trained():
    """LDI controls: December + March 2025 out of fold, February (the storm) IN training (as in the served refit),
    stopping on 9 only (3 is a test month). Fails when March stays a stopping month (test rows would steer early
    stopping) or February is dropped from training."""
    month = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 3])
    te, tr, fit, es = R.split_decmar(month)
    assert sorted(month[te].tolist()) == [3, 3, 12, 12]
    assert 2 in set(month[fit]) and sorted(set(month[es])) == [9]
    assert sorted(set(month[fit])) == [1, 2, 4, 5, 6, 7, 8, 10, 11]
    assert not (te & tr).any() and not (fit & es).any() and ((fit | es) == tr).all()


def test_split_months_out_refuses_missing_test_or_stopping_rows():
    """A month-out split with no test rows, no stopping rows or no fit rows is refused, never silently empty."""
    with pytest.raises(ValueError, match="no test rows"):
        R.split_feb(np.array([1, 3, 9]))
    with pytest.raises(ValueError, match="no early-stopping rows"):
        R.split_decmar(np.array([1, 3, 12]))
    with pytest.raises(ValueError, match="no fit rows"):
        R.split_decmar(np.array([3, 9, 12]))


def test_every_split_has_smoke_months_that_make_a_valid_split():
    """Each split's SMOKE_MONTHS must yield fit and stopping rows (and test rows unless refit): a smoke that cannot
    split would fail only at launch time."""
    assert set(R.SMOKE_MONTHS) == set(R.SPLITS) and {"feb", "decmar"} <= set(R.SPLITS)
    for s in R.SPLITS:
        month = np.repeat(np.array(R.SMOKE_MONTHS[s]), 3)
        te, tr, fit, es = R.SPLIT_FNS[s](month)
        assert fit.any() and es.any(), s
        assert te.any() == (s != "refit"), s
    assert R.parse_args(["--split", "feb", "--arms", "r12", "--tables", "t", "--out", "o",
                         "--rss-limit-gib", "11"]).split == "feb"


def test_run_feb_pools_the_target_share_and_scores_february(tmp_path):
    """A month-out split has no 2026 month to key a coverage target on: weights pool per airport (as forward).
    Fails when the run keys Feb cells by month (KeyError: no 2026 target for EGLL|2) or tests another month."""
    out = tmp_path / "logs" / "r12_scratch" / "feb"
    rep = R.run("feb", ["r12"], smoke=False, serve_table=_serve_small(), out_dir=out,
                load=lambda feats, split: _fake_fold(split), fit=_fake_fit, comparators=None, seeds=(0,), root=tmp_path,
                log=lambda m: None)
    assert rep["n_test"] == 40 and sorted(rep["weights"]["cells"]) == ["EGLL", "LFPG"]
    assert rep["weights"]["pooled_target"] is True
    pred = pd.read_parquet(out / "predictions.parquet")
    assert set(pred.month) == {2} and len(pred) == 40
    rep2 = R.run("decmar", ["r12"], smoke=False, serve_table=_serve_small(), out_dir=tmp_path / "logs" / "r12_scratch" / "dm",
                 load=lambda feats, split: _fake_fold(split), fit=_fake_fit, comparators=None, seeds=(0,), root=tmp_path,
                 log=lambda m: None)
    assert rep2["n_test"] == 80
    assert set(pd.read_parquet(tmp_path / "logs" / "r12_scratch" / "dm" / "predictions.parquet").month) == {3, 12}


def test_r13_cat_arm_is_a_delta_arm_on_the_r12_design():
    """PREREG R13 arm 1: CatBoost sees exactly the r12 design (FW-NSC + the ADS-B block) with the delta target."""
    assert "cat" in R.ARMS and R.TARGETS["cat"] == "delta"
    assert R.design_for("cat") == R.design_for("r12")


def test_r13_move_arm_drops_exactly_the_heard_columns():
    """AMEND-A2: r12_move = r12 minus the reception / sample-count / first-heard / join-quality columns,
    keeping the movement and timing block. Fails when a heard column leaks back in or a movement column is lost."""
    heard = {"coverage", "adsb_day_present", "joined", "join_callsign", "join_mvt_anchored", "mvt_anchor_gap_s",
             "first_seen_rel_mvt", "n_ground", "ground_rate_pm", "mlat_share", "first_ground_rel_mvt",
             "first_ground_gs", "first_ground_slow", "first_ground_rel_aobt3", "ground_seen_mvt_m10",
             "moving_mvt_m10", "stand_visible", "stand_ref_parked_share", "stand_others_n",
             "first_ground_to_stand_m", "travel_s_first_to_stand"}
    assert set(R.HEARD_COLS) == heard
    move = R.design_for("r12_move")
    assert set(R.design_for("r12")) - set(move) == heard
    assert {"dwell_s", "n_stops", "path_m", "taxi_med_gs", "implied_offblock_rel_aobt3", "implied_taxi_s",
            "dt_takeoff", "first_move_rel_mvt"} <= set(move)
    assert R.TARGETS["r12_move"] == "delta" and move[:len(R.BASE)] == R.BASE
