"""scripts/catboost_native.py + scripts/catboost_native_worker.py - arms C_delta / C_sched of
plans/PREREG_catboost_native_2026_09_10.md (native-categorical CatBoost, two targets, SCREEN tier).

Everything testable without the CatBoost venv is tested here: the registered column sets (no in-fold
encoding enters, so BC-1 cannot apply), the categorical construction and its airport scoping, the
identity join, target construction and recovery for both targets, the worker's input contract and
refit rule, and the screening rule.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import catboost_native as cn  # noqa: E402
import catboost_native_worker as cw  # noqa: E402


def test_numeric_columns_are_f_design_minus_every_encoding():
    """Rehearsed 2026-09-10 RED: the te_/de_ filter dropped (24 encodings leak in)."""
    import lgbm_fold as lf
    num = cn.numeric_columns()
    assert len(num) == 59 and not any(c.startswith(("te_", "de_")) for c in num)
    assert set(num) == {c for c in lf.FEATS_QUEUE_ORDER if not c.startswith(("te_", "de_"))}


def _frame():
    return pd.DataFrame({"ap": ["LIRF", "EGLL", "LIRF"], "STAND_mvt": ["210", "210", None],
                         "RUNWAY_mvt": ["25", "27L", "25"], "ADES_mvt": ["EGLL", "LIRF", None],
                         "FLIGHT_mvt": ["AZ1", None, "AZ3"], "CALLSIGN_flt": ["ITY1", "BAW2", None],
                         "airline": ["AZ1", "BA2", "AZ3"], "AIRCRAFT_OPERATOR_flt": ["ITY", "BAW", None],
                         "AIRCRAFT_TYPE_mvt": ["A320", "A320", "B738"], "MARKET_SEGMENT_flt": ["M", "M", None],
                         "WK_TBL_CAT_flt": ["M", "M", "M"], "FLIGHT_TYPE_flt": ["S", "S", "S"],
                         "stand_pref": ["21", "21", None]})


def test_categoricals_are_the_registered_13_with_airport_scoping_and_na():
    """Stand 210 at LIRF and at EGLL are different levels; a null stays inside its airport scope.
    Rehearsed 2026-09-10 RED: the airport prefix dropped from ap|STAND."""
    c = cn.categoricals(_frame())
    assert list(c.columns) == cn.CAT_COLS and len(cn.CAT_COLS) == 13
    assert c.ap_stand.tolist() == ["LIRF|210", "EGLL|210", "LIRF|NA"]
    assert c.ap_ades.tolist()[2] == "LIRF|NA" and c.FLIGHT_mvt.tolist()[1] == "NA"
    assert all(c[col].map(type).eq(str).all() for col in c.columns)


def test_identity_join_is_by_mvt_id_and_refuses_missing_ids():
    raw = pd.DataFrame({"MVT_ID_mvt": [3.0, 1.0, 2.0], "FLIGHT_mvt": ["C", "A", "B"], "CALLSIGN_flt": ["c", "a", None]})
    got = cn.attach_identity(np.array([1.0, 2.0, 3.0]), raw)
    assert got.FLIGHT_mvt.tolist() == ["A", "B", "C"] and pd.isna(got.CALLSIGN_flt.iloc[1])   # null survives as null; categoricals() maps it to "NA"
    with pytest.raises(ValueError, match="absent from the raw"):
        cn.attach_identity(np.array([1.0, 9.0]), raw)


def test_targets_and_recovery_for_both_formulations():
    """delta = y reconstructed via proxy; G = sp - y. Recovery floors at 1 s. Rehearsed 2026-09-10
    RED: G's recovery written proxy - pred."""
    y, proxy, sp, delta = np.array([900.0, 1200.0]), np.array([1000.0, 1500.0]), np.array([5000.0, 1250.0]), np.array([100.0, 300.0])
    t = cn.targets(y=y, sp=sp, delta=delta)
    np.testing.assert_allclose(t["G"], sp - y)
    np.testing.assert_allclose(cn.recover(t["delta"], proxy, sp, "delta"), y)
    np.testing.assert_allclose(cn.recover(t["G"], proxy, sp, "G"), y)
    np.testing.assert_allclose(cn.recover(np.array([5000.0, 0.0]), proxy, sp, "G"), [1.0, 1250.0])
    with pytest.raises(ValueError):
        cn.recover(t["G"], proxy, sp, "y")


def test_worker_contract_refuses_overlapping_masks_and_non_string_categoricals(tmp_path):
    n = 8
    d = pd.DataFrame({"x1": np.arange(n, dtype="float32"), "c1": list("abcdabcd"), "delta": np.arange(n, dtype="float64"),
                      "train": [1, 1, 1, 1, 1, 1, 0, 0], "fit": [1, 1, 1, 1, 0, 0, 0, 0], "es": [0, 0, 0, 0, 1, 1, 0, 0],
                      "holdout": [0, 0, 0, 0, 0, 0, 1, 1]})
    for c in ("train", "fit", "es", "holdout"):
        d[c] = d[c].astype(bool)
    p = tmp_path / "in.parquet"
    d.to_parquet(p)
    out = cw.load_input(p, num=["x1"], cat=["c1"], target="delta")
    assert out["X"].shape == (8, 2)
    bad = d.copy()
    bad.loc[6, "train"] = True
    bad.to_parquet(p)
    with pytest.raises(ValueError, match="overlap"):
        cw.load_input(p, num=["x1"], cat=["c1"], target="delta")
    bad2 = d.copy()
    bad2["c1"] = np.arange(n)
    bad2.to_parquet(p)
    with pytest.raises(ValueError, match="must be strings"):
        cw.load_input(p, num=["x1"], cat=["c1"], target="delta")


def test_refit_count_is_arm_f_rule():
    """n_ref = round(best_iter x n_train / n_fit) - arm F's scaled rule, not 'best + 1'."""
    assert cw.refit_count(best_iter=1000, n_train=1_200, n_fit=1_000) == 1200
    assert cw.refit_count(best_iter=25_316, n_train=2_062_440, n_fit=1_717_600) == round(25_316 * 2_062_440 / 1_717_600)


def test_screen_rule_needs_interval_and_1000_mse():
    """A consistent gain below 1,000 weighted MSE is NOT shortlisted. Rehearsed 2026-09-10: with an
    independent-noise 'tiny' arm the floor deletion SURVIVED (its interval already held zero); the
    shrunk-error arm pins the floor itself, and goes RED on its deletion."""
    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.uniform(400, 2000, n)
    F = y + rng.normal(0, 220, n)
    tiny = y + 0.99 * (F - y)                      # a CONSISTENT gain (interval excludes 0) of ~948 weighted MSE < 1,000
    big = y + rng.normal(0, 180, n)
    res = cn.screen(y, F, {"tiny": tiny, "big": big}, ap=np.array(["LIRF"] * n), fill=np.zeros(n, bool),
                    delta=np.zeros(n), n_boot=200)
    assert res["big"]["shortlist"] is True
    assert res["tiny"]["ci95"][0] > 0 and res["tiny"]["net_weighted_fold_mse"] < 1_000
    assert res["tiny"]["shortlist"] is False


def test_output_keeps_every_feature_float32_and_scoring_columns_apart():
    """`proxy` and `sp` are FEATURES and also needed for scoring. The first build wrote the float64 scoring
    copies over the float32 features (name collision); the worker's contract refused it. Scoring columns now
    carry a `score_` prefix. Rehearsed 2026-09-10 RED: scoring columns written under the feature names."""
    num = cn.numeric_columns()
    n = 4
    d = pd.DataFrame({c: np.arange(n, dtype="float64") + 0.5 for c in num})
    d = pd.concat([d, _frame().iloc[[0, 1, 2, 0]].reset_index(drop=True)], axis=1)
    d["y"], d["delta"], d["month"], d["MVT_ID_mvt"] = [900.0, 1000.0, 1100.0, 1200.0], [1.0, 2.0, 3.0, 4.0], [1, 2, 3, 7], [1.0, 2.0, 3.0, 4.0]
    masks = tuple(np.array(v, bool) for v in ([0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]))   # te, tr, fit, es
    o = cn.assemble_output(d, *masks)
    assert all(o[c].dtype == np.float32 for c in num)
    assert {"score_y", "score_proxy", "score_sp"} <= set(o.columns) and o.score_sp.dtype == np.float64
    np.testing.assert_array_equal(o.G.to_numpy(), d.sp.to_numpy() - d.y.to_numpy())
