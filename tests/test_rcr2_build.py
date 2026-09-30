"""Tests for scripts/analysis/rcr2_build.py (arm RCR2, plans/PREREG_rcr2_direct_target_production_2026_09_22.md).

Every test names the failure it catches. The post-compute lines (composition, delta statistics, verdict clauses,
prediction scoring, gate G1) are tested on synthetic frames BEFORE the first real run, because a crash or a wrong
sign there after a 2-hour fit is the BC-24 shape (MXI's `_stats` crashed post-compute).
"""
from __future__ import annotations

import importlib.util, json, os, pathlib, sys
import numpy as np, pandas as pd, pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(n, r):
    if n in sys.modules:
        return sys.modules[n]
    s = importlib.util.spec_from_file_location(n, ROOT / r); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m


R = _load("rcr2_build", "scripts/analysis/rcr2_build.py")


def test_imported_module_is_the_one_under_test():
    """Catches the AGS symlink-mirror landmine: the module under test must be the file next to this ROOT."""
    assert pathlib.Path(R.__file__).resolve() == (ROOT / "scripts/analysis/rcr2_build.py").resolve()


# ---- constants and the registration -------------------------------------------------------------------------
def test_legs_are_the_registered_two_directions():
    """Catches a leg silently added or a month moved after the hash (registration Stage 1)."""
    assert list(R.LEGS) == ["L-Jul", "L-Jan"]
    assert R.LEGS["L-Jul"] == {"fit_months": (1, 2, 3, 4, 5, 6), "test_month": 7}
    assert R.LEGS["L-Jan"] == {"fit_months": (8, 9, 10, 11, 12), "test_month": 1}


def test_stage1_deviation_is_rcrs():
    """Catches Stage 1 drifting from the registered CHR/RCR deviation (lr 0.05 / 8,000 / 200, ES 15 % days)."""
    assert (R.LR, R.MAX_ROUNDS, R.PATIENCE, R.ES_DAY_FRAC, R.ES_SEED) == (0.05, 8000, 200, 0.15, 1716)


def test_constants_match_the_registration():
    assert R.SEEDS == (0, 1, 2) and R.N_BOOT == 2000 and R.BOOT_SEED == 0
    assert R.INC_RMSE == 208.31040447352203 and (R.N_INC, R.N_POP) == (334_132, 161_956)
    assert R.RCR_A2_GAIN == 1.5094285641585827 and R.BOARD_N == 344_841.0
    assert R.FLOOR_BIND_MAX == 0.005 and R.SEED_SPREAD_MAX_S == 2.0 and R.RSS_LIMIT == 6 * 1024 ** 3


def test_registration_hash_matches_the_log():
    """Catches the registration being edited after hashing (the runner refuses, and so does this test)."""
    log = (ROOT / "reports/rcr2_prereg_hashes.log").read_text()
    assert f"(full {R.PREREG_SHA})" in log
    assert f"(full {R._prereg_ok()})" in log, "the current registration file (with its appended sections) is not in the hash log"


def test_prereg_ok_refuses_a_tampered_registration(tmp_path, monkeypatch):
    t = tmp_path / "p.md"; t.write_text("tampered\n")
    monkeypatch.setattr(R, "PREREG", t)
    with pytest.raises(AssertionError, match="registration text changed"):
        R._prereg_ok()


def test_prereg_ok_accepts_a_result_appended_after_the_hash(tmp_path, monkeypatch):
    """The RESULT is appended below the hashed text; the hash of the text above '## RESULT' must still verify."""
    import hashlib
    body = "registered text\n"; t = tmp_path / "p.md"; t.write_text(body + "\n## RESULT\nlater\n")
    monkeypatch.setattr(R, "PREREG", t); monkeypatch.setattr(R, "PREREG_SHA", hashlib.sha256(body.encode()).hexdigest())
    assert R._prereg_ok()
    t.write_text(body + "\n## ERRATUM 1\nfix\n\n## RESULT\nlater\n")           # an erratum before the result
    assert R._prereg_ok()
    t.write_text(body + "edited\n\n## RESULT\nlater\n")                        # the hashed text itself changed
    with pytest.raises(AssertionError, match="registration text changed"):
        R._prereg_ok()


# ---- feature list and labels --------------------------------------------------------------------------------
def test_production_feats_is_the_shipped_93_in_order():
    S = _load("stand_ab", "scripts/stand_ab.py")
    f = R.production_feats(S)
    assert len(f) == 93 and f == list(S.BASELINE_FEATS) + list(S.QUEUE_FEATS) + list(S.ORDER_FEATS) + list(S.WEATHER_FEATS)
    assert "proxy" in f and "y" not in f and "delta" not in f


def test_production_feats_refuses_a_label_column():
    class S:
        BASELINE_FEATS = ["proxy", "y"]; QUEUE_FEATS = []; ORDER_FEATS = []; WEATHER_FEATS = []
    with pytest.raises(AssertionError, match="leak gate 1"):
        R.production_feats(S)


def test_production_feats_refuses_a_wrong_width():
    class S:
        BASELINE_FEATS = ["proxy"]; QUEUE_FEATS = []; ORDER_FEATS = []; WEATHER_FEATS = []
    with pytest.raises(AssertionError, match="93 features"):
        R.production_feats(S)


def test_check_labels_accepts_and_summarises():
    assert R.check_labels([1.0, 500.0, 86400.0]) == {"n": 3, "min": 1.0, "max": 86400.0, "n_over_one_day": 0}


def test_check_labels_counts_labels_above_one_day_without_excluding_them():
    """ERRATUM 1: the LIRF doy-196 row (y = 87,002 s) is production's row; it is counted and reported, never dropped."""
    assert R.check_labels([10.0, 87002.0, 90000.0]) == {"n": 3, "min": 10.0, "max": 90000.0, "n_over_one_day": 2}


@pytest.mark.parametrize("bad, msg", [([1.0, np.nan], "non-finite"), ([0.5, 10.0], "below 1 s")])
def test_check_labels_refuses_bad_labels(bad, msg):
    with pytest.raises(AssertionError, match=msg):
        R.check_labels(bad)


# ---- leg masks ----------------------------------------------------------------------------------------------
def _months_days():
    month = np.repeat(np.arange(1, 13), 40); day = np.concatenate([np.repeat(np.arange(1, 21) + 31 * (m - 1), 2) for m in range(1, 13)])
    return month, day


def test_leg_masks_partition_and_never_touch_the_test_month():
    month, day = _months_days()
    fit, es, test = R.leg_masks(month, day, (1, 2, 3, 4, 5, 6), 7)
    assert not (fit & es).any() and not (fit & test).any() and not (es & test).any()
    assert set(month[test]) == {7} and set(month[fit]) <= set(range(1, 7)) and set(month[es]) <= set(range(1, 7))
    assert not np.isin(month[fit | es], [7]).any() and (fit | es).sum() == 240


def test_leg_masks_es_is_whole_days_at_the_registered_fraction():
    """ES is 15 % of the FIT months' real DAYS, whole days (a row-level ES would leak same-day rows)."""
    month, day = _months_days()
    fit, es, test = R.leg_masks(month, day, (8, 9, 10, 11, 12), 1)
    es_days, fit_days = set(day[es]), set(day[fit])
    assert not (es_days & fit_days)
    assert len(es_days) == round(0.15 * 100)


def test_leg_masks_is_deterministic():
    month, day = _months_days()
    a = R.leg_masks(month, day, (1, 2, 3), 4); b = R.leg_masks(month, day, (1, 2, 3), 4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_leg_masks_refuses_test_inside_fit_and_empty_sets():
    month, day = _months_days()
    with pytest.raises(ValueError, match="inside the fit months"):
        R.leg_masks(month, day, (1, 2, 7), 7)
    with pytest.raises(ValueError, match="empty"):
        R.leg_masks(month, day, (1, 2), 13)


# ---- fitting and taxi ---------------------------------------------------------------------------------------
def test_fit_leg_learns_a_linear_signal_and_reports_best_iter():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3)).astype(np.float32); label = 3 * X[:, 0] + rng.normal(scale=.1, size=600)
    fit = np.zeros(600, bool); fit[:400] = True; es = ~fit
    p = dict(objective="regression", learning_rate=0.3, num_leaves=7, min_data_in_leaf=5, verbosity=-1, num_threads=1, seed=0)
    raw, best = R.fit_leg(X, label, fit, es, X[es], p, max_rounds=100, patience=10)
    assert raw.shape == (200,) and best >= 1
    assert R.rmse(label[es], raw) < 1.0


def test_fit_leg_refuses_non_finite_predictions(monkeypatch):
    import lightgbm as lgb
    class M:
        best_iteration = 3
        def predict(self, X, num_iteration=None): return np.full(len(X), np.nan)
    monkeypatch.setattr(lgb, "train", lambda *a, **k: M())
    X = np.zeros((10, 2), np.float32); m = np.ones(10, bool)
    with pytest.raises(RuntimeError, match="non-finite"):
        R.fit_leg(X, np.zeros(10), m, m, X, {})


def test_taxi_floors_both_targets_like_lgbm_submit():
    proxy = np.array([100.0, 100.0, 100.0]); raw = np.array([20.0, 150.0, -5.0])
    assert np.array_equal(R.taxi(raw, proxy, "delta"), [80.0, 1.0, 105.0])
    assert np.array_equal(R.taxi(raw, proxy, "y"), [20.0, 150.0, 1.0])
    with pytest.raises(ValueError):
        R.taxi(raw, proxy, "z")


# ---- paired statistics --------------------------------------------------------------------------------------
def _paired():
    rng = np.random.default_rng(1)
    n = 400; y = rng.normal(300, 50, n); ref = y + rng.normal(0, 20, n); arm = y + rng.normal(0, 10, n)
    days = np.repeat(np.arange(20), 20); ap = np.where(np.arange(n) % 4 == 0, "LIRF", np.where(np.arange(n) % 4 == 1, "LFPG", "EGLL"))
    return y, days, ap, ref, arm


def test_paired_stats_gain_sign_and_ci_when_arm_is_better():
    y, days, ap, ref, arm = _paired()
    st = R.paired_stats(y, days, ap, ref, arm, n_boot=300)
    assert st["gain_s"] > 0 and st["ci95"][0] > 0 and st["n"] == 400
    assert st["rmse_ref"] == pytest.approx(R.rmse(y, ref)) and st["rmse_arm"] == pytest.approx(R.rmse(y, arm))
    assert st["sse_gain_per_board_row"] == pytest.approx(((y - ref) ** 2 - (y - arm) ** 2).sum() / 344841)
    assert set(st["per_airport_gain_s"]) == {"EGLL", "LFPG", "LIRF"}


def test_paired_stats_minus_top_removes_the_largest_rows():
    """One planted monster row makes the raw gain; minus top-1 must drop it and the share must say so."""
    y, days, ap, ref, arm = _paired()
    ref = y.copy(); arm = y.copy(); ref[0] += 5000                      # the arm wins on ONE row only
    st = R.paired_stats(y, days, ap, ref, arm, n_boot=100)
    assert st["gain_s"] > 0 and st["gain_minus_top1_s"] == pytest.approx(0.0) and st["top1_share"] == pytest.approx(1.0)
    assert st["gain_minus_top3_s"] == pytest.approx(0.0) and st["top3_share"] == pytest.approx(1.0)


def test_paired_stats_airport_exclusions_exclude_the_named_airport():
    y, days, ap, ref, arm = _paired()
    arm = ref.copy(); arm[ap == "LIRF"] = y[ap == "LIRF"]           # the arm only helps at LIRF
    st = R.paired_stats(y, days, ap, ref, arm, n_boot=100)
    assert st["gain_s"] > 0 and st["gain_ex_LIRF_s"] == pytest.approx(0.0) and st["gain_ex_LFPG_s"] > 0
    assert st["per_airport_gain_s"]["LIRF"] > 0 and st["per_airport_gain_s"]["EGLL"] == pytest.approx(0.0)


def test_paired_stats_refuses_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        R.paired_stats([1, 2], [1, 2], ["a", "b"], [1, 2], [1])


def test_minus_top_share_is_nan_when_nothing_moves():
    y = np.array([1.0, 2.0]); g, share = R._minus_top(y, y, y, 1)
    assert g == 0.0 and np.isnan(share)


def test_move_energy_keys_and_values():
    e = R.move_energy([1.0, 1.0, 1.0], [3.0, 1.0, 5.0], ["A", "A", "B"], [1, 1, 7])
    assert e == {"1:A": 2.0, "7:B": 16.0}


# ---- gate G1 and RCR reproduction ---------------------------------------------------------------------------
def _stage1(jul_raw, jul_m3, jan_raw, jan_m3):
    mk = lambda r, m: {"y_vs_delta": {"gain_s": r, "gain_minus_top3_s": m, "per_airport_gain_s": {"LIRF": -9, "EGLL": 1}}}
    return {"L-Jul": mk(jul_raw, jul_m3), "L-Jan": mk(jan_raw, jan_m3)}


@pytest.mark.parametrize("args, want", [((1, 1, 1, 1), True), ((0, 0, 0, 0), True), ((1, -0.1, 1, 1), False),
                                        ((-0.1, 1, 1, 1), False), ((1, 1, -0.1, 1), False), ((1, 1, 1, -0.1), False)])
def test_gate_g1_requires_raw_and_minus_top3_on_both_legs(args, want):
    assert R.gate_g1(_stage1(*args)) is want


def test_reproduce_rcr_a2_matches_the_stored_report():
    """The runner's artifact check: A2's gain must reproduce to 1e-6 from reports/rcr_test_predictions.parquet."""
    st = R.reproduce_rcr_a2()
    rep = json.loads((ROOT / "reports/rcr_report.json").read_text())
    assert st["gain_s"] == pytest.approx(rep["vs_A0"]["A2"]["gain_s"], abs=1e-9)
    assert st["ci95"] == pytest.approx(rep["vs_A0"]["A2"]["ci95"], abs=1e-9)
    assert st["top3_share"] > 0.7 and st["gain_minus_top3_s"] < 0.5        # the baseline-read finding
    assert "per_airport_gain_s" not in st


def test_reproduce_rcr_a2_refuses_a_drifted_artifact(tmp_path):
    p = pd.read_parquet(ROOT / "reports/rcr_test_predictions.parquet").iloc[:5000]
    t = tmp_path / "x.parquet"; p.to_parquet(t, index=False)
    with pytest.raises(AssertionError, match="does not reproduce"):
        R.reproduce_rcr_a2(t)


# ---- Stage 2: incumbent, composition, delta statistics, verdict, predictions --------------------------------
def _inc_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(100, 900, n); proxy = y + rng.normal(0, 50, n)
    F = rng.normal(0, 30, n); v13 = F.copy(); pop = np.arange(n) % 2 == 0
    v13[~pop] = F[~pop] + 1.0                                        # stage-served rows differ from F
    return pd.DataFrame({"row": np.arange(n), "MVT_ID_mvt": np.arange(n) + 10, "ap": np.where(np.arange(n) % 3 == 0, "LFPG", "LIRF"),
                         "month": np.where(np.arange(n) < n // 2, 1, 7), "day": pd.to_datetime("2025-01-01") + pd.to_timedelta(np.arange(n) % 7, "D"),
                         "fold": 0.0, "y": y, "proxy": proxy, "F": F, "v13_delta": v13,
                         "taxi_v13_delta": np.maximum(proxy - v13, 1), "C": rng.normal(0, 30, n), "FW": F + rng.normal(0, 5, n)})


def test_finish_incumbent_composes_the_registered_formula():
    f = _inc_frame(); f.loc[f.index[:40], "v13_delta"] = f.F[:40] + 2.0        # 20 more stage-served rows: pop = 80
    s = R.finish_incumbent(f, subset_ok=True)
    pop = s["pop"].to_numpy()
    assert pop.sum() == 80 and np.array_equal(pop, s.v13_delta.eq(s.F).to_numpy())
    want = np.where(pop, .5 * (np.maximum(s.proxy - s.C, 1) + np.maximum(s.proxy - s.FW, 1)), s.taxi_v13_delta)
    assert np.allclose(s.inc, want) and (s.doy.to_numpy()[:7] == np.arange(1, 8)).all()


def test_finish_incumbent_refuses_the_wrong_row_count_or_rmse(monkeypatch):
    with pytest.raises(AssertionError):
        R.finish_incumbent(_inc_frame(), subset_ok=False)
    monkeypatch.setattr(R, "N_INC", 200); monkeypatch.setattr(R, "N_POP", 100)
    with pytest.raises(AssertionError, match="incumbent does not reproduce"):
        R.finish_incumbent(_inc_frame(), subset_ok=False)               # counts right, RMSE wrong


def test_attach_doy_joins_and_refuses_a_missing_day():
    fw = pd.DataFrame({"row": [0, 1], "FW": [1.0, 2.0]})
    oof = pd.DataFrame({"row": [0, 1], "day": pd.to_datetime(["2025-01-03", "2025-07-01"])})
    j = R.attach_doy(fw, oof)
    assert list(j.columns) == ["row", "FW", "doy"] and j.doy.tolist() == [3, 182]
    with pytest.raises(AssertionError, match="have no day"):
        R.attach_doy(fw, oof.iloc[:1])


def test_compose_replaces_fw_by_fy_on_pop_rows_only():
    s = R.finish_incumbent(_inc_frame(), subset_ok=True)
    fy = s.y.to_numpy() * 0 + 5.0
    c = R.compose(s, fy)
    pop = s["pop"].to_numpy()
    assert np.allclose(c[pop], .5 * (s.taxi_C.to_numpy()[pop] + 5.0)) and np.allclose(c[~pop], s.taxi_v13_delta.to_numpy()[~pop])
    c0 = R.compose(s, np.full(len(s), 0.2))
    assert np.allclose(c0[pop], .5 * (s.taxi_C.to_numpy()[pop] + 1.0)), "the floor at 1 must apply to Fy"
    with pytest.raises(ValueError, match="shape"):
        R.compose(s, fy[:3])


def test_block_bootstrap_delta_units_and_ci():
    g = np.array([10.0, -2.0, 4.0, 8.0]); blocks = [1, 1, 2, 2]
    d, ci = R.block_bootstrap_delta(g, blocks, n_boot=500, seed=0)
    assert d == pytest.approx(20 / 344841) and ci[0] <= d <= ci[1]
    assert ci[0] >= min(8, 12) * 2 / 344841 - 1e-12 and ci[1] <= max(8, 12) * 2 / 344841 + 1e-12


def test_mde_is_2_8_half_width_over_1_96():
    assert R.mde([-1.96, 1.96]) == pytest.approx(2.8)


def test_delta_stats_sign_units_cuts_and_k():
    s = R.finish_incumbent(_inc_frame(), subset_ok=True)
    y = s.y.to_numpy(); base = s.inc.to_numpy(); cand = y.copy()                          # the truth itself: k = Σ(D·e)/ΣD² = 1 (D = e)
    st = R.delta_stats(y, s.doy, s.ap, s.month, base, cand, n_boot=300)
    g = (y - base) ** 2 - (y - cand) ** 2
    assert st["delta"] == pytest.approx(g.sum() / 344841) and st["delta"] > 0 and st["ci95"][0] > 0
    half = R.delta_stats(y, s.doy, s.ap, s.month, base, base + 0.5 * (y - base), n_boot=50)
    assert half["k"] == pytest.approx(2.0), "halfway to the truth is k = 2 under the LSC identity"
    assert st["k"] == pytest.approx(1.0) and st["S"] == pytest.approx(((cand - base) ** 2).sum() / 344841) and st["n_moved"] == 200
    assert set(st["per_month"]) == {1, 7} and st["per_month"][1] + st["per_month"][7] == pytest.approx(st["delta"])
    assert st["delta_ex_LFPG"] == pytest.approx(g[s.ap.to_numpy() != "LFPG"].sum() / 344841)
    assert st["delta_ex_LIRF"] + st["delta_ex_LFPG"] == pytest.approx(st["delta"])
    assert st["per_airport"]["LFPG"] + st["per_airport"]["LIRF"] == pytest.approx(st["delta"])
    assert st["gain_s"] == pytest.approx(R.rmse(y, base) - R.rmse(y, cand)) and st["mde"] == pytest.approx(R.mde(st["ci95"]))
    o = np.argsort(-np.abs(g))
    assert st["delta_minus_top1"] == pytest.approx(g.sum() / 344841 - g[o[0]] / 344841)
    assert st["delta_minus_top3"] == pytest.approx((g.sum() - g[o[:3]].sum()) / 344841)
    assert st["top1_share"] == pytest.approx(abs(g[o[0]]) / abs(g.sum()))


def test_delta_stats_negative_when_the_candidate_is_worse_and_k_zero_for_noise():
    s = R.finish_incumbent(_inc_frame(), subset_ok=True)
    y = s.y.to_numpy(); base = y.copy(); rng = np.random.default_rng(3); cand = base + rng.normal(0, 10, len(y))
    st = R.delta_stats(y, s.doy, s.ap, s.month, base, cand, n_boot=200)
    assert st["delta"] < 0 and st["k"] == pytest.approx(0.0) and st["ci_excludes_zero"]
    e = rng.choice([-5.0, 5.0], len(y))
    st2 = R.delta_stats(y, s.doy, s.ap, s.month, y + e, y - e, n_boot=200)      # identical errors: no effect at all
    assert abs(st2["delta"]) < 1e-12 and st2["ci95"] == pytest.approx([0.0, 0.0], abs=1e-12) and st2["ci_excludes_zero"] is False


def test_delta_stats_blocks_are_month_and_day_not_day_alone():
    """Jan 3 and Jul 3 are different blocks: the CI must equal the explicit (month*1000 + doy) block bootstrap."""
    s = R.finish_incumbent(_inc_frame(), subset_ok=True)
    y = s.y.to_numpy(); base = s.inc.to_numpy(); rng = np.random.default_rng(5); cand = base + rng.normal(0, 30, len(y))
    st = R.delta_stats(y, s.doy, s.ap, s.month, base, cand, n_boot=300)
    g = (y - base) ** 2 - (y - cand) ** 2
    _, ci_md = R.block_bootstrap_delta(g, s.month.to_numpy() * 1000 + s.doy.to_numpy(), n_boot=300)
    _, ci_d = R.block_bootstrap_delta(g, s.doy.to_numpy(), n_boot=300)
    assert st["ci95"] == pytest.approx(ci_md) and st["ci95"] != pytest.approx(ci_d)


def _comp(**kw):
    d = {"delta": 50.0, "ci95": [10.0, 90.0], "per_month": {1: 20.0, 7: 30.0}, "delta_minus_top3": 40.0, "delta_ex_LFPG": 30.0,
         "ci_excludes_zero": True, "top1_share": 0.1}
    d.update(kw); return d


def test_stage2_verdict_working_when_every_clause_holds():
    v = R.stage2_verdict(_comp(), _comp(), 0.0, [200.0, 200.5, 199.8])
    assert v["verdict"] == "WORKING" and all(v["clauses"].values()) and v["seed_spread_s"] == pytest.approx(0.4, abs=0.01)


@pytest.mark.parametrize("comp_kw, f_kw, floor, seeds, failing", [
    ({"ci95": [-1.0, 90.0], "ci_excludes_zero": False}, {}, 0.0, [1, 1, 1], "T2_comp_positive_ci"),
    ({"delta": -5.0, "ci95": [-9.0, -1.0]}, {}, 0.0, [1, 1, 1], "T2_comp_positive_ci"),
    ({"per_month": {1: -1.0, 7: 51.0}}, {}, 0.0, [1, 1, 1], "T3_both_months_nonneg"),
    ({"delta_minus_top3": -1.0}, {}, 0.0, [1, 1, 1], "T4_minus_top3_and_ex_LFPG"),
    ({"delta_ex_LFPG": 0.0}, {}, 0.0, [1, 1, 1], "T4_minus_top3_and_ex_LFPG"),
    ({}, {"ci95": [-1.0, 5.0]}, 0.0, [1, 1, 1], "T5_F_alone_positive_ci"),
    ({}, {}, 0.006, [1, 1, 1], "T6_floor_and_seeds"),
    ({}, {}, 0.0, [1, 1, 4.1], "T6_floor_and_seeds"),
])
def test_stage2_verdict_each_clause_fails_alone(comp_kw, f_kw, floor, seeds, failing):
    v = R.stage2_verdict(_comp(**comp_kw), _comp(**f_kw), floor, seeds)
    assert v["verdict"].startswith("NOT WORKING") and v["clauses"][failing] is False
    assert sum(1 for c in v["clauses"].values() if not c) == 1


def test_stage2_verdict_names_not_established_when_ci_includes_zero():
    v = R.stage2_verdict(_comp(ci95=[-1.0, 90.0], ci_excludes_zero=False), _comp(), 0.0, [1, 1, 1])
    assert "not established" in v["verdict"]
    v2 = R.stage2_verdict(_comp(delta=-5.0, ci95=[-9.0, -1.0]), _comp(), 0.0, [1, 1, 1])
    assert v2["verdict"] == "NOT WORKING"


def test_predictions_scored_stage1_hits_and_misses():
    s = _stage1(1.0, 0.8, -2.0, -2.5)
    p = R.predictions_scored(s, None)
    assert p == {"P1_LJan_raw_negative": "HIT", "P2_LJul_band": "HIT", "P3_LIRF_largest_both_legs": "HIT", "P4_stops_at_stage1": "HIT"}
    s2 = _stage1(3.0, 0.5, 1.0, 1.0)
    s2["L-Jul"]["y_vs_delta"]["per_airport_gain_s"] = {"LIRF": 1, "EGLL": 9}
    p2 = R.predictions_scored(s2, None)
    assert p2 == {"P1_LJan_raw_negative": "MISS", "P2_LJul_band": "MISS", "P3_LIRF_largest_both_legs": "MISS", "P4_stops_at_stage1": "MISS"}


def test_predictions_scored_stage2_bands():
    s2 = {"composed": _comp(delta=100.0, ci95=[0.0, 200.0]), "f_alone": _comp(delta=300.0), "best_iter": 21101}
    assert R.predictions_scored(None, s2) == {"P5_bands": "HIT", "P6_best_iter": "HIT", "P7_top_row_share": "HIT"}
    s3 = {"composed": _comp(delta=250.0, ci95=[0.0, 20.0], top1_share=0.3), "f_alone": _comp(delta=500.0), "best_iter": 30000}
    assert R.predictions_scored(None, s3) == {"P5_bands": "MISS", "P6_best_iter": "MISS", "P7_top_row_share": "MISS"}
    assert R.predictions_scored(None, None) == {}


# ---- guards and real artifacts ------------------------------------------------------------------------------
def test_guards_take_and_release_a_private_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("PHANTOM_HEAVY_LOCK_PATH", str(tmp_path / "lock"))
    fh, G = R._guards()
    with pytest.raises(RuntimeError, match="held by another job"):
        G.acquire_lock()
    G.release_lock(fh)
    fh2 = G.acquire_lock(); G.release_lock(fh2)


def test_load_fw_all_covers_every_fold_row_with_a_day():
    j = R.load_fw_all()
    assert len(j) == 339_015 and list(j.columns) == ["row", "FW", "doy"] and j.doy.between(1, 212).all()
    assert set(pd.read_parquet(ROOT / "data/cache_stand/fold_preds_queue_order_weather.parquet", columns=["month"]).month.unique()) == {1, 7}


@pytest.mark.slow
def test_load_incumbent_reproduces_opc_screens_anchor():
    """The incumbent anchor (208.31040447352203, 334,132 rows, pop 161,956) must reproduce or the run aborts."""
    s = R.load_incumbent()
    assert len(s) == 334_132 and int(s["pop"].sum()) == 161_956 and abs(R.rmse(s.y, s.inc) - 208.31040447352203) < 1e-8
