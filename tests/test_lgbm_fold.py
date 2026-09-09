"""`scripts/lgbm_fold.py` is the fold measurement for v4 (Amendment 12): seed averaging and
the per-airport blend on the LightGBM path, on fold A exactly as `lgbm_ab.py` measured it.

What the harness must get right, and what each test pins:
  * arm bookkeeping - A0/A1a/A1b are the single seeds, A2 their mean, A3 the blend - so that a
    reported gain belongs to the arm it is attributed to;
  * the paired row bootstrap must detect a planted signal and reject noise (this project's
    falsify rule for any harness) and its interval must be the width a paired bootstrap has;
  * the repo's ESTABLISHED rule - gain > 2 x the sd over the three single seeds - computed with
    the sample sd, strictly;
  * the fold: holdout months 1 and 7, early-stopping months 3 and 9 carved from the TRAINING
    fold, never from the holdout.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_stand"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lf = _load("lgbm_fold")


def _load_test_helper(name):
    """A non-test helper module under tests/, loaded like the scripts are."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_arms_are_exactly_what_their_definitions_say():
    """A0 = seed 0, A1a = seed 1, A1b = seed 2, A2 = the arithmetic mean of the three, A3 =
    (1 - BLEND) * A2 + BLEND * per-airport on fitted rows and A2 on fallback rows. Checked on
    non-constant arrays so a swapped or dropped seed cannot hide.

    Fails when A2 is a single seed, when A1a/A1b are swapped, or when A3 blends the fallback
    rows or blends against A0 instead of A2. Rehearsed 2026-09-09, each RED: `"A2": single[0]`;
    `"A3": L.blend(single[0], ...)`; `A1a`/`A1b` swapped.
    """
    n = 6
    single = {0: np.arange(n, dtype="float64"), 1: np.arange(n, dtype="float64") * 2 + 1,
              2: np.full(n, 10.0)}
    pa = np.array([100.0, np.nan, 300.0, np.nan, 500.0, 600.0])
    fitted = np.array([True, False, True, False, True, True])
    arms = lf.build_arms(single, pa, fitted)
    assert list(arms) == ["A0", "A1a", "A1b", "A2", "A3"]
    assert arms["A0"].tolist() == single[0].tolist()
    assert arms["A1a"].tolist() == single[1].tolist()
    assert arms["A1b"].tolist() == single[2].tolist()
    a2 = (single[0] + single[1] + single[2]) / 3.0
    assert np.allclose(arms["A2"], a2, atol=0, rtol=0)     # exact: same float64 arithmetic
    want = np.where(fitted, 0.5 * a2 + 0.5 * pa, a2)
    assert np.allclose(arms["A3"], want, atol=0, rtol=0)
    assert lf.ARM_DEFINITIONS["A2"].startswith("mean of seeds")
    with pytest.raises(ValueError, match="seeds"):
        lf.build_arms({0: single[0], 1: single[1]}, pa, fitted)


def _analytic_paired_se(sa, sb):
    """Delta-method SE of rmse(a) - rmse(b) under paired row resampling."""
    ma, mb = sa.mean(), sb.mean()
    c = np.cov(sa, sb) / len(sa)
    return float(np.sqrt(c[0, 0] / (4 * ma) + c[1, 1] / (4 * mb) - 2 * c[0, 1] / (4 * np.sqrt(ma * mb))))


def test_paired_bootstrap_detects_a_planted_ten_sigma_gain_and_rejects_noise():
    """The falsify rule for a harness: it must find a planted signal and must not find one in
    noise. Two arms whose residuals have the same variance and correlation 0.95 (a noise
    pair - the same model refit) must get an interval that includes zero, with a half-width
    within 10% of 1.96 x the delta-method paired SE (measured 0.996 on this fixture: point
    -0.21 s, SE 0.22 s, CI [-0.66, +0.21]); an arm whose residuals are 10% smaller (a planted
    gain of 9.77 s = 44.8 x its paired SE, asserted >= 10 x) must get an interval that
    excludes zero and a point estimate equal to the actual RMSE difference. The same row
    draws serve every pair.

    Fails when the draws are not paired (independent indices per arm inflate the width ~5x),
    when the percentiles are not 2.5/97.5, or when the interval degenerates to the point.
    Rehearsed 2026-09-09, each RED: `idx_b = rng.integers(0, n, n)` for the second arm;
    `np.percentile(gains, [5, 95])`; `lo = hi = point`.
    """
    n, rng = 20_000, np.random.default_rng(0)
    ra = rng.normal(0, 100, n)
    eps = rng.normal(0, 100, n)
    rb = 0.95 * ra + np.sqrt(1 - 0.95 ** 2) * eps        # same variance, corr 0.95: noise
    rc = 0.9 * rb                                        # planted: 10% smaller residuals
    se = {"A": ra ** 2, "B": rb ** 2, "C": rc ** 2}
    res = lf.paired_bootstrap(se, [("B", "A"), ("C", "A")], n_draws=2_000, seed=0)
    assert set(res) == {"B_vs_A", "C_vs_A"}

    noise = res["B_vs_A"]
    se_noise = _analytic_paired_se(se["A"], se["B"])
    point = float(np.sqrt(se["A"].mean()) - np.sqrt(se["B"].mean()))
    assert abs(point) < 3 * se_noise, "fixture is not a noise pair"
    assert noise["gain_s"] == pytest.approx(point, abs=1e-12)
    lo, hi = noise["ci95"]
    assert lo <= 0.0 <= hi, f"noise interval excludes zero: [{lo:.3f}, {hi:.3f}]"
    assert noise["excludes_zero"] is False and noise["improving"] is False
    ratio = ((hi - lo) / 2) / (1.96 * se_noise)
    assert 0.9 < ratio < 1.1, f"interval half-width is {ratio:.2f} x the paired SE"

    planted = res["C_vs_A"]
    se_planted = _analytic_paired_se(se["A"], se["C"])
    gain = float(np.sqrt(se["A"].mean()) - np.sqrt(se["C"].mean()))
    assert gain >= 10 * se_planted, "fixture does not plant a 10-sigma gain"
    assert planted["gain_s"] == pytest.approx(gain, abs=1e-12)
    assert planted["ci95"][0] > 0 and planted["excludes_zero"] is True and planted["improving"] is True
    assert planted["n_draws"] == 2_000 and planted["seed"] == 0

    # the draws are shared: the same seed must reproduce the same intervals exactly
    again = lf.paired_bootstrap(se, [("B", "A")], n_draws=2_000, seed=0)
    assert again["B_vs_A"]["ci95"] == noise["ci95"]


def test_established_rule_requires_the_gain_to_exceed_twice_the_seed_sd():
    """Seed sd is the SAMPLE sd (ddof=1) of the three single-seed RMSEs - three values give
    two degrees of freedom - and the repo's ESTABLISHED rule needs gain > 2 x that sd,
    strictly. Pinned by hand: [226.24, 226.50, 226.80] -> sd 0.2803; a 0.5 s gain does not
    clear 0.5606, a 0.6 s gain does, equality does not.

    Fails when the rule slips to 1 x (0.5 > 0.28 would pass), when ddof=0 (sd 0.229), or when
    `>=` replaces `>`. Rehearsed 2026-09-09, each RED: `gain > seed_sd`; `np.std(v)`; `>=`.
    """
    vals = [226.24, 226.50, 226.80]
    sd = lf.seed_sd(vals)
    assert sd == pytest.approx(0.28031, abs=1e-4)          # hand: sqrt(0.15707 / 2)
    assert lf.exceeds_2x_seed_sd(0.5, sd) is False
    assert lf.exceeds_2x_seed_sd(0.6, sd) is True
    assert lf.exceeds_2x_seed_sd(2 * sd, sd) is False        # equality is not exceeding
    assert lf.exceeds_2x_seed_sd(-1.0, sd) is False
    with pytest.raises(ValueError, match="three"):
        lf.seed_sd([226.24, 226.50])


def test_fold_holds_out_jan_and_jul_and_carves_es_months_from_the_training_fold():
    """Holdout = months 1 and 7; training = everything else; early-stop = months 3 and 9 of
    the TRAINING fold; fit = training minus early-stop. The four masks partition the rows.

    Fails when the ES months are taken from all rows (a July row could stop the model on
    itself), or when the holdout leaks into the fit rows. Rehearsed 2026-09-09: `es =
    np.isin(month, es_months)` without `& tr` went RED (month 7 is not an ES month, so the
    rehearsal plants month 3 rows as holdout via `holdout=(1, 3)`).
    """
    month = np.array([1, 2, 3, 7, 9, 12, 1, 3])
    te, tr, fit, es = lf.fold_masks(month)
    assert te.tolist() == [1, 0, 0, 1, 0, 0, 1, 0]
    assert tr.tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    assert es.tolist() == [0, 0, 1, 0, 1, 0, 0, 1]
    assert fit.tolist() == [0, 1, 0, 0, 0, 1, 0, 0]
    assert ((te | fit | es) == np.ones(8, bool)).all() and not (te & tr).any()
    te2, tr2, fit2, es2 = lf.fold_masks(month, holdout=(1, 3))
    assert not (es2 & te2).any(), "an early-stopping month leaked out of the training fold"
    assert lf.HOLDOUT == (1, 7) and lf.L.ES_MONTHS == (3, 9)


def test_per_airport_rmse_and_the_count_of_airports_that_improve():
    """RMSE per arm is reported pooled and per airport, and "airports improving" counts the
    airports whose RMSE is strictly lower under the second arm - a tie is not an improvement
    (Amendment 12.3 needs >= 6 of 10 to improve).

    Fails when the per-airport cut is pooled, when the count uses >= (the planted tie would
    count), or when the labels are cast to float32 before the residual. Rehearsed 2026-09-09,
    each RED: `rmse_new[a] <= rmse_ref[a]`; `y = np.asarray(y, dtype="float32")` in
    squared_errors (the 100,000.3 label rounds to 100,000.297 and the error becomes 0.0918).
    """
    y = np.array([1000.0, 1000.0, 2000.0, 2000.0, 100_000.0, 500.0])
    proxy = np.array([1100.0, 1100.0, 2100.0, 2100.0, 100_100.0, 600.0])
    ap_code = np.array([0, 0, 1, 1, 2, 2], np.int8)
    airports = ["AAA", "BBB", "CCC"]
    arms = {"X": np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),   # perfect
            "Y": np.array([110.0, 90.0, 100.0, 100.0, 100.0, 100.0])}    # AAA worse, others tied
    se = lf.squared_errors(arms, y, proxy)
    assert se["X"].tolist() == [0.0] * 6
    assert se["Y"].tolist() == [100.0, 100.0, 0.0, 0.0, 0.0, 0.0]
    r = lf.arm_rmses(se, ap_code, airports)
    assert r["X"]["pooled"] == 0.0 and r["Y"]["pooled"] == pytest.approx(np.sqrt(200 / 6), abs=1e-12)
    assert r["Y"]["per_airport"] == {"AAA": 10.0, "BBB": 0.0, "CCC": 0.0}
    assert lf.airports_improving(r["X"]["per_airport"], r["Y"]["per_airport"]) == 1   # X over Y: AAA only
    assert lf.airports_improving(r["Y"]["per_airport"], r["X"]["per_airport"]) == 0   # ties do not count
    # a 1e5-scale row whose label is NOT exact in float32 (100,000.3 -> 100,000.297): the squared
    # error must be formed in float64 (0.09), not from a float32 label (0.0918)
    big = lf.squared_errors({"Z": np.array([0.0])}, np.array([100_000.3]), np.array([100_000.6]))
    assert big["Z"][0] == pytest.approx(0.09, rel=1e-9)


def test_taxi_time_floors_at_one_second_without_rounding_and_guards_the_floor():
    """y_hat = max(proxy - delta_hat, 1) in float64, unrounded - lgbm_ab scored unrounded and A0
    must reproduce it. The 0.5% floor-bind guard and the finite check are those of
    lgbm_submit.recover_taxi_time.

    Fails when the floor is dropped or the values are rounded. Rehearsed 2026-09-09:
    `np.rint(...)` went RED on 100.4.
    """
    proxy = np.full(1_000, 1_000.0)
    delta_hat = np.full(1_000, 899.6)
    delta_hat[0] = 1_500.0
    got = lf.taxi_time(proxy, delta_hat)
    assert got.dtype == np.float64
    assert got[0] == 1.0 and got[1] == pytest.approx(100.4, abs=1e-12)
    with pytest.raises(AssertionError, match="positivity floor"):
        lf.taxi_time(np.full(1_000, 100.0), np.full(1_000, 200.0))


def test_no_predict_outside_predict_delta():
    """The fold harness issues no prediction of its own: every delta_hat comes out of
    lgbm_submit.fit_seeds / fit_per_airport, whose only predict is predict_delta (which
    forwards num_threads; a bare .predict( would run single-threaded under the repo's OMP cap
    for 20+ minutes per arm - the v3 incident). So the fold measures the code that ships.

    Fails when the fold script calls Booster.predict directly, or grows its own fitting path
    instead of using the shared one. Rehearsed 2026-09-09: `b.predict(X_te)` added to the
    seed loop went RED."""
    src = (ROOT / "scripts" / "lgbm_fold.py").read_text()
    assert src.count(".predict(") == 0, "a call site bypasses lgbm_submit.predict_delta"
    assert "L.fit_seeds(" in src and "L.fit_per_airport(" in src, "the fold must fit through lgbm_submit"
    assert "lgb.train(" not in src and "import lightgbm" not in src, "the fold must not fit on its own"
    sub = (ROOT / "scripts" / "lgbm_submit.py").read_text()
    assert sub.count(".predict(") == 1


_smoke_inputs = [CACHE / f"training_2025-0{m}-01_2025-0{m + 1}-01.parquet" for m in (1, 2, 3)]


@pytest.mark.skipif(not all(p.exists() for p in _smoke_inputs),
                    reason="needs the January-March month caches (scripts/stand_ab.py cache)")
def test_smoke_end_to_end_writes_json_log_and_per_arm_predictions(tmp_path, capsys):
    """Three months (holdout 1, early-stop 3, fit 2), tiny trees, 200 bootstrap draws, a
    scratch directory. The JSON, the log and the per-arm prediction parquet all land; the
    arms in the parquet obey their definitions; the JSON's RMSEs are recomputable from the
    parquet; the reproduction check reports itself skipped; the real report and cache paths
    are untouched; the banner is printed and written to the log.

    Fails when any output is missing, when the JSON and parquet disagree, when the smoke
    writes to reports/ or data/cache_stand/, or when the banner does not open and close the
    log (it also sits in the table header, so presence alone is not a guard). Rehearsed
    2026-09-09, each RED: `json_path = REPORTS / JSON_NAME` under --out-dir; dropping the
    closing `log(L.SMOKE_BANNER)` alone; dropping the opening one alone; writing `A2` as
    `single[0]` into the parquet.
    """
    real = [ROOT / "reports" / "lgbm_fold_v4.json", ROOT / "reports" / "lgbm_fold_v4.log",
            CACHE / "fold_v4_preds.parquet"]
    before = [p.stat().st_mtime_ns if p.exists() else None for p in real]
    rc = lf.main(["--smoke", "--out-dir", str(tmp_path)])
    assert rc == 0
    assert [p.stat().st_mtime_ns if p.exists() else None for p in real] == before, \
        "the smoke touched a real output path"

    j = json.loads((tmp_path / "lgbm_fold_v4.json").read_text())
    log = (tmp_path / "lgbm_fold_v4.log").read_text()
    preds = pd.read_parquet(tmp_path / "fold_v4_preds.parquet")
    for key in ("smoke", "git_sha", "fold", "config", "a0_reproduction", "arms", "seed_sd",
                "pairs", "per_airport_models", "command", "estimate"):
        assert key in j, f"json lacks {key}"
    assert j["smoke"] is True and j["config"]["n_boot"] == lf.SMOKE["n_boot"]
    assert j["config"]["seeds"] == [0, 1, 2] and j["config"]["blend"] == 0.5
    assert j["config"]["pa_tree_rule"] == lf.L.PA_TREE_RULE and j["config"]["pa_trees"] == "share"
    assert lf.parse_args([]).pa_trees == "share" and lf.parse_args(["--pa-trees", "es"]).pa_trees == "es"
    assert j["a0_reproduction"]["status"] == "skipped (smoke)"
    assert j["fold"]["holdout_months"] == [1, 7] and j["fold"]["es_months"] == [3, 9]
    assert set(preds.month.unique()) == {1}
    assert len(preds) == j["fold"]["n_test"] > 100_000

    cols = ["row", "month", "ap", "y", "proxy", "delta", "delta_hat_seed0", "delta_hat_seed1",
            "delta_hat_seed2", "pa_seed0", "pa_seed1", "pa_seed2", "pa_fitted",
            "A0", "A1a", "A1b", "A2", "A3"]
    assert list(preds.columns) == cols
    s = [preds[f"delta_hat_seed{i}"].to_numpy() for i in range(3)]
    assert np.array_equal(preds.A0.to_numpy(), s[0]) and np.array_equal(preds.A1a.to_numpy(), s[1])
    assert np.array_equal(preds.A1b.to_numpy(), s[2])
    assert np.allclose(preds.A2.to_numpy(), (s[0] + s[1] + s[2]) / 3, atol=1e-9, rtol=0)
    pa = np.mean([preds[f"pa_seed{i}"].to_numpy() for i in range(3)], axis=0)
    fitted = preds.pa_fitted.to_numpy()
    assert fitted.any() and (~fitted).any(), "smoke exercised only one per-airport branch"
    assert np.isnan(pa[~fitted]).all() and np.isfinite(pa[fitted]).all()
    want = np.where(fitted, 0.5 * preds.A2.to_numpy() + 0.5 * np.nan_to_num(pa), preds.A2.to_numpy())
    assert np.allclose(preds.A3.to_numpy(), want, atol=1e-9, rtol=0)
    for arm in ("A0", "A1a", "A1b", "A2", "A3"):
        yhat = np.maximum(preds.proxy.to_numpy() - preds[arm].to_numpy(), 1.0)
        rm = float(np.sqrt(((preds.y.to_numpy() - yhat) ** 2).mean()))
        assert j["arms"][arm]["rmse"] == pytest.approx(rm, abs=1e-9), arm
        assert set(j["arms"][arm]["per_airport"]) == set(preds.ap.unique())
    for a, row in j["per_airport_models"].items():
        assert row["fitted"] == (row["n_train"] >= 20_000), a
        rows = preds.ap.to_numpy() == a
        assert fitted[rows].all() == row["fitted"]
    assert set(j["pairs"]) == {"A2_vs_A0", "A3_vs_A2", "A3_vs_A0"}
    for pr in j["pairs"].values():
        assert pr["ci95"][0] <= pr["gain_s"] <= pr["ci95"][1]
        assert isinstance(pr["exceeds_2x_seed_sd"], bool) and isinstance(pr["airports_improving"], int)
        assert set(pr["per_airport"]) == set(preds.ap.unique())
    assert j["seed_sd"]["sd"] >= 0 and sorted(j["seed_sd"]["values"]) == ["A0", "A1a", "A1b"]

    text = capsys.readouterr().out
    lines = [l for l in log.splitlines() if l.strip()]
    assert lf.L.SMOKE_BANNER in lines[0] and lf.L.SMOKE_BANNER in lines[-1], \
        "the banner must open AND close the smoke log"
    assert text.count(lf.L.SMOKE_BANNER) >= 2
    assert "python3.11" in j["command"] and "reports/lgbm_fold_v4.json" in log
    assert "per-airport tree rule" in log


@pytest.mark.skipif(not all(p.exists() for p in _smoke_inputs),
                    reason="needs the January-March month caches (scripts/stand_ab.py cache)")
def test_smoke_with_the_per_airport_early_stopping_rule(tmp_path, capsys):
    """`--smoke --pa-trees es`: the per-airport models are early-stopped on their own rows and
    refit at their own n_ref; the JSON records the rule, every fitted airport's best_iter and
    n_ref, and A3 is still the blend of A2 with the (now differently sized) per-airport mean.

    Fails when the rule is not threaded through to fit_per_airport, or when the JSON does not
    record it. Rehearsed 2026-09-09: `trees="share"` hard-coded in the fold's call went RED.
    """
    assert lf.main(["--smoke", "--out-dir", str(tmp_path), "--pa-trees", "es"]) == 0
    j = json.loads((tmp_path / "lgbm_fold_v4.json").read_text())
    assert j["config"]["pa_trees"] == "es" and j["config"]["pa_tree_rule"] == lf.L.PA_TREE_RULE_ES
    fitted = {a: r for a, r in j["per_airport_models"].items() if r["fitted"]}
    assert fitted and all(r["rule"] == "es" and r["es"]["best_iter"] >= 1
                          and r["n_trees"] == r["es"]["n_ref"] >= 50 for r in fitted.values())
    assert all(r["es"] is None for r in j["per_airport_models"].values() if not r["fitted"])
    preds = pd.read_parquet(tmp_path / "fold_v4_preds.parquet")
    pa = np.mean([preds[f"pa_seed{i}"].to_numpy() for i in range(3)], axis=0)
    f = preds.pa_fitted.to_numpy()
    want = np.where(f, 0.5 * preds.A2.to_numpy() + 0.5 * np.nan_to_num(pa), preds.A2.to_numpy())
    assert np.allclose(preds.A3.to_numpy(), want, atol=1e-9, rtol=0)
    assert lf.L.SMOKE_BANNER in capsys.readouterr().out


# =============================================================================================
# Amendments 14 and 15: the queue arm, the CatBoost arm, the capacity sweep and its confirmation
# tier. Every fold-A arm below measures against the shipped v4 arm's own per-row predictions
# (read from fold_v4_preds.parquet on the identical fold) or refits it in-process; nothing is
# measured against a re-typed number. The end-to-end tests run on a SYNTHETIC cache that has the
# real caches' columns and dtypes, so every arm's code path is exercised without the real months
# (the real-month smokes are the owner's to run beside the live fold).
# =============================================================================================

import argparse
import sys
import textwrap

APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
QUEUE_FEATS = ["q_apt_at_push", "q_rwy_at_push", "q_pushed_after_me", "q_rwy_tko_in_taxi",
               "q_rwy_push_pre20", "q_dep_tko_sym15", "q_rwy_tko_sym10", "q_arr_taxiing_at_push",
               "q_arr_taxiin_sym30_push", "q_arr_taxiin_sym30_tko", "q_rwy_ambient_proxy",
               "q_next_arr_onblock_gap"]
VENV_PY = pathlib.Path.home() / ".venvs" / "prc-catboost" / "bin" / "python"


def test_modes_are_mutually_exclusive_and_name_their_own_outputs():
    """One run measures one arm. No flag is the v4 path and keeps its legacy file names (the
    running v4 fold and its tests keep their contract); --queue / --catboost / --sweep /
    --confirm each name a sibling json + log under reports/ and a predictions parquet under
    data/cache_stand/ (none for the screening tier, which has no holdout predictions worth a
    cut); two arms at once are refused. --baseline names the v4 arm, --seeds the seed set.

    Fails when a mode inherits the v4 names (it would overwrite the v4 record), when two
    modes are accepted together, or when the sweep grows a parquet. Rehearsed 2026-09-09,
    each RED: `"queue": OUTPUT_NAMES["v4"]`; removing the exclusivity check; giving the
    sweep a parquet name.
    """
    assert lf.mode_of(lf.parse_args([])) == "v4"
    assert lf.mode_of(lf.parse_args(["--queue"])) == "queue"
    assert lf.mode_of(lf.parse_args(["--catboost"])) == "catboost"
    assert lf.mode_of(lf.parse_args(["--sweep"])) == "sweep"
    assert lf.mode_of(lf.parse_args(["--confirm", "255,40,0.8"])) == "confirm"
    for argv in (["--queue", "--catboost"], ["--sweep", "--confirm", "255,40,0.8"], ["--queue", "--sweep"]):
        with pytest.raises(SystemExit):
            lf.parse_args(argv)
    out = pathlib.Path("/o")
    assert lf.output_paths("v4", out) == (out / "lgbm_fold_v4.json", out / "lgbm_fold_v4.log",
                                          out / "fold_v4_preds.parquet")
    assert lf.output_paths("queue", out) == (out / "lgbm_fold_queue.json", out / "lgbm_fold_queue.log",
                                             out / "fold_preds_queue.parquet")
    assert lf.output_paths("catboost", out)[2] == out / "fold_preds_catboost.parquet"
    assert lf.output_paths("confirm", out)[0] == out / "lgbm_fold_confirm.json"
    assert lf.output_paths("sweep", out) == (out / "lgbm_fold_sweep.json", out / "lgbm_fold_sweep.log", None)
    j, l, p = lf.output_paths("queue", None)
    assert j == lf.REPORTS / "lgbm_fold_queue.json" and l == lf.REPORTS / "lgbm_fold_queue.log"
    assert p == lf.CACHE / "fold_preds_queue.parquet"
    assert lf.output_paths("v4", None) == (lf.REPORTS / "lgbm_fold_v4.json", lf.REPORTS / "lgbm_fold_v4.log",
                                           lf.CACHE / "fold_v4_preds.parquet")
    with pytest.raises(ValueError):
        lf.output_paths("magic", out)
    assert lf.parse_args([]).baseline == "A3" and lf.parse_args(["--baseline", "A2"]).baseline == "A2"
    with pytest.raises(SystemExit):
        lf.parse_args(["--baseline", "A0"])
    assert lf.parse_args([]).seeds == (0, 1, 2) and lf.parse_args(["--seeds", "0,1"]).seeds == (0, 1)
    assert lf.parse_args([]).n_perm == 20 and lf.parse_args(["--n-perm", "3"]).n_perm == 3
    assert lf.parse_args([]).refit_baseline is False
    assert lf.parse_args([]).baseline_preds is None and lf.parse_args([]).v4_json is None
    assert lf.V4_PREDS == lf.CACHE / "fold_v4_preds.parquet" and lf.V4_JSON == lf.REPORTS / "lgbm_fold_v4.json"


def test_sweep_grid_has_18_unique_settings_and_confirm_refuses_a_setting_outside_it():
    """Amendment 15.2's grid: num_leaves {127, 255, 511} x min_data_in_leaf {20, 40, 100} x
    feature_fraction {0.6, 0.8} = 18 unique settings, the incumbent (255, 40, 0.8) among
    them. A setting is written "leaves,min_data,fraction"; --confirm takes one or more and
    refuses anything outside the grid (a setting that was never screened cannot be
    "confirmed"). setting_params changes exactly those three keys of the measured P.

    Fails when the grid is not the registered one, when an off-grid setting is accepted, or
    when setting_params touches another parameter. Rehearsed 2026-09-09, each RED:
    `(127, 255)` for the leaves axis; `if False:` for the grid membership check;
    `dict(params, ..., learning_rate=0.05)` in setting_params.
    """
    grid = lf.sweep_grid()
    assert len(grid) == 18 and len(set(grid)) == 18
    assert {nl for nl, _, _ in grid} == {127, 255, 511}
    assert {m for _, m, _ in grid} == {20, 40, 100}
    assert {f for _, _, f in grid} == {0.6, 0.8}
    assert (255, 40, 0.8) in grid and lf.INCUMBENT_SETTING == (255, 40, 0.8)
    assert lf.parse_setting("255,40,0.8") == (255, 40, 0.8)
    assert lf.parse_setting(" 511, 20, 0.6 ") == (511, 20, 0.6)
    assert lf.setting_label((255, 40, 0.8)) == "255,40,0.8" and lf.setting_label((511, 20, 0.6)) == "511,20,0.6"
    for bad in ("300,40,0.8", "255,40", "255,40,0.7", "a,b,c", "", "255,40,0.8,1"):
        with pytest.raises(argparse.ArgumentTypeError):
            lf.parse_setting(bad)
    args = lf.parse_args(["--confirm", "255,40,0.8", "511,20,0.6"])
    assert args.confirm == [(255, 40, 0.8), (511, 20, 0.6)]
    with pytest.raises(SystemExit):
        lf.parse_args(["--confirm", "300,40,0.8"])
    with pytest.raises(SystemExit):
        lf.parse_args(["--confirm"])
    p = lf.setting_params(lf.L.P, (511, 20, 0.6))
    assert (p["num_leaves"], p["min_data_in_leaf"], p["feature_fraction"]) == (511, 20, 0.6)
    keys = ("num_leaves", "min_data_in_leaf", "feature_fraction")
    assert {k: v for k, v in p.items() if k not in keys} == {k: v for k, v in lf.L.P.items() if k not in keys}
    assert (lf.L.P["num_leaves"], lf.L.P["min_data_in_leaf"], lf.L.P["feature_fraction"]) == lf.INCUMBENT_SETTING
    assert lf.SWEEP == dict(train=(2, 3, 4, 5), stop=6, learning_rate=0.05)


def test_screen_ranking_is_by_stopping_set_rmse_ascending_and_every_line_carries_the_banner(tmp_path):
    """The screening tier ranks the 18 settings by their stopping-set matched RMSE, ascending,
    ties in input order, and every number it emits - each ranking entry in the JSON, every
    line of its log - carries "RANKING ONLY — no magnitude from this tier is a result",
    because this project has twice quoted a subsample magnitude as a result.

    Fails when the sort is descending or by best_iter, when an entry lacks the banner, or
    when the log's suffix is dropped. Rehearsed 2026-09-09, each RED: `reverse=True`;
    `key=lambda r: r["best_iter"]`; removing `banner=RANKING_BANNER` from the entry;
    `suffix=None` in the sweep log.
    """
    results = [dict(setting=(127, 20, 0.6), rmse_stop=230.5, best_iter=3),
               dict(setting=(255, 40, 0.8), rmse_stop=229.0, best_iter=5),
               dict(setting=(511, 100, 0.8), rmse_stop=229.0, best_iter=2),
               dict(setting=(127, 100, 0.6), rmse_stop=231.2, best_iter=9)]
    ranked = lf.rank_settings(results)
    assert [r["rank"] for r in ranked] == [1, 2, 3, 4]
    assert [r["setting"] for r in ranked] == ["255,40,0.8", "511,100,0.8", "127,20,0.6", "127,100,0.6"]
    assert [r["rmse_stop_RANKING_ONLY"] for r in ranked] == [229.0, 229.0, 230.5, 231.2]
    assert [r["best_iter"] for r in ranked] == [5, 2, 3, 9]
    assert all(r["banner"] == lf.RANKING_BANNER for r in ranked)
    assert lf.RANKING_BANNER == "RANKING ONLY — no magnitude from this tier is a result"
    assert all("rmse_stop" not in r for r in ranked), "a bare magnitude key survived the rename"
    log = lf._Log(tmp_path / "sweep.log", suffix=lf.RANKING_BANNER)
    log("setting 255,40,0.8 best_iter 5")
    log("no digits in this message")
    log.close()
    lines = (tmp_path / "sweep.log").read_text().splitlines()
    assert len(lines) == 2 and all(line.endswith(lf.RANKING_BANNER) for line in lines)
    plain = lf._Log(tmp_path / "plain.log")
    plain("x 1")
    plain.close()
    assert lf.RANKING_BANNER not in (tmp_path / "plain.log").read_text()


def test_sweep_subfold_trains_feb_to_may_and_stops_on_june():
    """The screening subfold: training rows are months 2-5, the stopping set is month 6 and
    nothing else; the early-stopping call sees fit == train and es == stop (there is no
    refit in the screen). A stop month inside the training months, or an absent month, is
    refused.

    Fails when the stop month leaks into the training rows or the masks are the fold-A ones.
    Rehearsed 2026-09-09: `tr = np.isin(month, train_months + (stop,))` went RED.
    """
    month = np.array([1, 2, 3, 4, 5, 6, 7, 6, 2])
    te, tr, fit, es = lf.sweep_masks(month)
    assert te.tolist() == [0, 0, 0, 0, 0, 1, 0, 1, 0]
    assert tr.tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 1]
    assert np.array_equal(fit, tr) and np.array_equal(es, te) and not (tr & te).any()
    te2, tr2, _, _ = lf.sweep_masks(month, train_months=(2,), stop=3)
    assert te2.tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0] and tr2.tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 1]
    with pytest.raises(ValueError, match="stop"):
        lf.sweep_masks(month, train_months=(2, 3), stop=3)
    with pytest.raises(ValueError, match="absent"):
        lf.sweep_masks(np.array([2, 3]), train_months=(2,), stop=6)
    assert lf.SMOKE_SWEEP == dict(train=(2,), stop=3)


def test_quarter_capacity_and_within_group_row_permutation_of_the_queue_block():
    """The negative control refits at best_iter // 4 trees (17,557 -> 4,389; never below 1) and
    permutes the twelve-column queue block as WHOLE ROWS within each group (airport x fold
    side): every group keeps exactly its own rows, a row's twelve values travel together
    (the joint distribution is kept, only the link to the row is broken), a singleton group
    is unchanged, and seed 0 reproduces the same permutation.

    Fails when columns are permuted independently (the row identity 10x+1 breaks), when rows
    cross groups, or when the count is not integer-quartered. Rehearsed 2026-09-09, each
    RED: permuting each column with its own draw; `rng.permutation(len(block))` over all
    rows; `round(best_iter / 4)` (17,557 -> 4,389 still, but 4 -> 1 and 6 -> 2 differ: pinned
    with 6 -> 1).
    """
    assert lf.reduced_trees(17_557) == 4_389 and lf.reduced_trees(8) == 2
    assert lf.reduced_trees(4) == 1 and lf.reduced_trees(3) == 1 and lf.reduced_trees(6) == 1
    with pytest.raises(ValueError):
        lf.reduced_trees(0)
    n = 41
    groups = np.r_[np.repeat([0, 1, 2, 3], 10), 7]              # four groups of ten and a singleton
    block = np.column_stack([np.arange(n, dtype="float32"), np.arange(n, dtype="float32") * 10 + 1])
    orig = block.copy()
    got = lf.permute_within_groups(block, groups, np.random.default_rng(0))
    assert got is block, "the permutation must be in place (the block is 100 MB on the real fold)"
    assert not np.array_equal(got, orig), "nothing moved"
    for g in np.unique(groups):
        rows = groups == g
        assert sorted(map(tuple, got[rows].tolist())) == sorted(map(tuple, orig[rows].tolist())), g
    assert got[40].tolist() == orig[40].tolist()
    assert np.array_equal(got[:, 1], got[:, 0] * 10 + 1), "columns were permuted independently"
    again = lf.permute_within_groups(orig.copy(), groups, np.random.default_rng(0))
    assert np.array_equal(again, got)
    with pytest.raises(ValueError, match="length"):
        lf.permute_within_groups(block, groups[:-1], np.random.default_rng(0))


def test_negative_control_compares_the_reduced_capacity_gain_to_the_95th_percentile():
    """The treatment's reduced-capacity gain must EXCEED the 95th percentile of the permuted
    gains (np.percentile, linear interpolation: 0..20 -> 19.0), strictly; equality does not
    count. Because every gain subtracts the SAME baseline RMSE, this is identically "the
    treatment's reduced-capacity RMSE is below the 5th percentile of the permuted RMSEs" -
    the form the summary also reports, because the baseline at 68 columns is not like-for-like
    with the 80-column runs (LightGBM's feature-sampling stream depends on the column count:
    on the synthetic fold every permuted run beat the baseline by the same ~8 s) and it cancels
    in the treatment-vs-permuted comparison. The summary records every permuted gain and
    RMSE, the count at or above the treatment, and the registered deviation from Amendment
    14.4's text ("20 permutations at quarter capacity, paired").

    Fails when the percentile is the median or the maximum, when `>=` replaces `>`, or when
    the two forms disagree. Rehearsed 2026-09-09, each RED: `np.percentile(perm, 50)`;
    `perm.max()`; `>=`; `p5 = np.percentile(rmses, 95)`.
    """
    base = 256.0                                          # every value below is an exact binary float
    perm_rmse = [base - float(i) for i in range(21)]       # permuted gains 0..20 -> p95 = 19.0, p5 rmse = base - 19
    c = lf.control_summary(base, base - 19.5, perm_rmse)
    assert c["p95_permuted_gain_s"] == 19.0 and c["exceeds_p95"] is True
    assert c["gain_reduced_s"] == 19.5 and c["p5_permuted_rmse"] == base - 19.0
    assert c["treatment_below_p5"] is True and c["treatment_below_p5"] == c["exceeds_p95"]
    assert lf.control_summary(base, base - 18.0, perm_rmse)["exceeds_p95"] is False
    assert lf.control_summary(base, base - 19.0, perm_rmse)["exceeds_p95"] is False      # equality is not exceeding
    assert c["n_perm"] == 21 and c["permuted_gains_s"] == [float(i) for i in range(21)]
    assert c["permuted_rmses"] == perm_rmse and c["n_permuted_at_or_above"] == 1 and c["permuted_gain_max_s"] == 20.0
    assert lf.control_summary(base, base - 25.0, perm_rmse)["n_permuted_at_or_above"] == 0
    assert "feature-sampling" in c["note"] and "cancels" in c["note"]
    # a shared offset in the baseline moves every gain but not the verdict: the baseline cancels
    shifted = lf.control_summary(base + 8.0, base - 19.5, perm_rmse)
    assert shifted["gain_reduced_s"] == 27.5 and shifted["exceeds_p95"] == c["exceeds_p95"]
    assert lf.CONTROL_DEVIATION.startswith("20 permutations at quarter capacity, paired")
    assert lf.N_PERM == 20 and lf.PERM_SEED == 0
    with pytest.raises(ValueError):
        lf.control_summary(base, base - 1.0, [base - 1.0])


def test_tail_subset_is_non_fill_and_beyond_twenty_minutes_strictly():
    """Amendment 14.3's pre-defined tail: non-fill (|y - sp| > 60 s) AND |delta| > 1,200 s,
    both strict, so the boundary rows (exactly 60, exactly 1,200) are outside it.

    Fails when either inequality is relaxed to >=. Rehearsed 2026-09-09, each RED:
    `>= TAIL_NONFILL_S` (row 2 enters); `>= TAIL_DELTA_S` (row 3 enters).
    """
    y = np.full(5, 1_000.0)
    sp = np.array([1_000.0, 1_061.0, 1_060.0, 1_061.0, 900.0])
    delta = np.array([1_500.0, 1_500.0, 1_500.0, 1_200.0, -1_201.0])
    assert lf.tail_mask(y, sp, delta).tolist() == [False, True, False, False, True]
    assert lf.TAIL_NONFILL_S == 60.0 and lf.TAIL_DELTA_S == 1_200.0
    assert lf.NAMED_AIRPORTS == ("LEBL", "LTFM", "EGLL", "LSZH")


def _base_frame(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"row": np.arange(n, dtype="int64"), "month": np.full(n, 1, "int32"),
                         "ap": np.array(APTS, dtype=object)[np.arange(n) % 10].astype(str),
                         "y": rng.uniform(300, 900, n), "proxy": rng.uniform(600, 1_200, n),
                         "delta": rng.normal(size=n), "sp": rng.uniform(300, 900, n)})


def _fake_v4_preds(base, seed=0):
    """A fold_v4_preds.parquet-shaped frame whose A2/A3 are formed exactly as build_arms forms
    them (left-to-right float64 sum / 3, blend 0.5 on fitted rows)."""
    rng = np.random.default_rng(seed)
    n = len(base)
    p = base.copy()
    for s in (0, 1, 2):
        p[f"delta_hat_seed{s}"] = rng.normal(size=n) + s
    fitted = rng.random(n) < 0.5
    for s in (0, 1, 2):
        p[f"pa_seed{s}"] = np.where(fitted, rng.normal(size=n) + 10 + s, np.nan)
    p["pa_fitted"] = fitted
    p["A0"], p["A1a"], p["A1b"] = p.delta_hat_seed0, p.delta_hat_seed1, p.delta_hat_seed2
    p["A2"] = (p.delta_hat_seed0.to_numpy() + p.delta_hat_seed1.to_numpy() + p.delta_hat_seed2.to_numpy()) / 3
    pa = (p.pa_seed0.to_numpy() + p.pa_seed1.to_numpy() + p.pa_seed2.to_numpy()) / 3
    p["A3"] = np.where(fitted, 0.5 * p.A2.to_numpy() + 0.5 * np.nan_to_num(pa), p.A2.to_numpy())
    return p


def test_baseline_is_rebuilt_from_the_v4_per_seed_predictions_on_the_identical_fold():
    """The baseline arm is the v4 run's own prediction, rebuilt from its per-seed columns for
    the seed set asked for (A2 = their mean, A3 = A2 blended with the per-airport mean on
    fitted rows) and checked against the parquet's stored arm column when the seed set is
    the full one; the fold must be IDENTICAL (row index, month, airport, y, proxy, delta, all
    exact) or the pairing is meaningless; a seed the v4 run did not fit is refused; the v4
    JSON's configuration must match the treatment's (same seeds, same per-airport rule for
    A3, same smoke flag, A0 reproduced).

    Fails when the fold check is dropped, when the rebuild ignores the seed subset, or when
    the configuration check is skipped. Rehearsed 2026-09-09, each RED: `if False:` for the
    fold comparison; `single = {s: preds[f"delta_hat_seed{s}"] for s in SEEDS}` ignoring
    the argument; `if False:` for the pa_trees comparison.
    """
    base = _base_frame()
    v4 = _fake_v4_preds(base.drop(columns=["sp"]))
    b3 = lf.baseline_from_preds(v4, base, "A3", (0, 1, 2))
    assert np.allclose(b3["delta"], v4.A3.to_numpy(), atol=1e-12, rtol=0)
    assert np.array_equal(b3["fitted"], v4.pa_fitted.to_numpy()) and sorted(b3["single"]) == [0, 1, 2]
    b2 = lf.baseline_from_preds(v4, base, "A2", (0, 1, 2))
    assert np.allclose(b2["delta"], v4.A2.to_numpy(), atol=1e-12, rtol=0) and b2["fitted"] is None
    b01 = lf.baseline_from_preds(v4, base, "A2", (0, 1))
    assert np.allclose(b01["delta"], (v4.delta_hat_seed0.to_numpy() + v4.delta_hat_seed1.to_numpy()) / 2,
                       atol=1e-12, rtol=0)
    assert sorted(b01["single"]) == [0, 1]
    wrong = v4.copy()
    wrong.loc[3, "y"] += 1.0
    with pytest.raises(ValueError, match="fold"):
        lf.baseline_from_preds(wrong, base, "A3", (0, 1, 2))
    with pytest.raises(ValueError, match="fold"):
        lf.baseline_from_preds(v4.iloc[:-1], base, "A3", (0, 1, 2))
    with pytest.raises(ValueError, match="seed 5"):
        lf.baseline_from_preds(v4, base, "A2", (0, 5))
    bad = v4.copy()
    bad["A3"] = bad.A3 + 1e-3
    with pytest.raises(ValueError, match="disagrees"):
        lf.baseline_from_preds(bad, base, "A3", (0, 1, 2))
    j = dict(smoke=False, config=dict(seeds=[0, 1, 2], pa_trees="es"), a0_reproduction=dict(status="pass"))
    lf.check_baseline_config(j, "A3", (0, 1, 2), "es", smoke=False)
    lf.check_baseline_config(j, "A2", (0, 1), "share", smoke=False)          # A2 has no per-airport rule
    with pytest.raises(ValueError, match="tree rule"):
        lf.check_baseline_config(j, "A3", (0, 1, 2), "share", smoke=False)
    with pytest.raises(ValueError, match="seed"):
        lf.check_baseline_config(j, "A2", (0, 3), "es", smoke=False)
    with pytest.raises(ValueError, match="smoke"):
        lf.check_baseline_config(j, "A2", (0,), "es", smoke=True)
    with pytest.raises(ValueError, match="reproduc"):
        lf.check_baseline_config(dict(j, a0_reproduction=dict(status="FAIL")), "A2", (0,), "es", smoke=False)
    lf.check_baseline_config(dict(j, smoke=True, a0_reproduction={"status": "skipped (smoke)"}),
                             "A3", (0,), "es", smoke=True)


def test_blend_curve_and_residual_correlation_are_computed_from_the_recovered_taxi_times():
    """The blend curve is the matched RMSE of the recovered taxi time at CatBoost weight 0
    (the LightGBM arm), 0.3, 0.5, 0.7 and 1 (CatBoost alone): a perfect LightGBM arm and a
    CatBoost arm off by +-10 s give exactly 0, 3, 5, 7, 10. "Increases with weight" is the
    strict monotone rise over 0.3/0.5/0.7 that Amendment 15.1 reads as dilution. The residual
    correlation is Pearson's r between the two arms' residuals.

    Fails when the curve is computed on delta instead of on the recovered taxi time (it
    would still be 0/3/5/7/10 here, so the taxi-time floor is pinned separately), when the
    weights are reversed, or when monotonicity is read over the wrong points. Rehearsed
    2026-09-09, each RED: `w * base + (1 - w) * cat`; the window `(0.0, 0.3, 0.5)` in place of
    BLEND_WEIGHTS (the 0.7-turn case).
    """
    y = np.array([100.0, 200.0, 300.0, 400.0])
    proxy = np.array([150.0, 250.0, 350.0, 450.0])
    base = np.full(4, 50.0)                                  # y_hat = proxy - 50 = y exactly
    cat = np.array([60.0, 40.0, 60.0, 40.0])                 # off by +-10
    curve = lf.blend_curve(base, cat, y, proxy)
    assert list(curve) == [0.0, 0.3, 0.5, 0.7, 1.0]
    assert curve[0.0] == 0.0 and curve[1.0] == pytest.approx(10.0)
    assert curve[0.3] == pytest.approx(3.0) and curve[0.5] == pytest.approx(5.0) and curve[0.7] == pytest.approx(7.0)
    assert lf.curve_increases_with_weight(curve) is True
    assert lf.curve_increases_with_weight({0.0: 5.0, 0.3: 4.0, 0.5: 3.5, 0.7: 4.5, 1.0: 6.0}) is False
    assert lf.curve_increases_with_weight({0.0: 5.0, 0.3: 4.0, 0.5: 4.0, 0.7: 4.5, 1.0: 6.0}) is False
    assert lf.curve_increases_with_weight({0.0: 1.0, 0.3: 2.0, 0.5: 3.0, 0.7: 2.5, 1.0: 4.0}) is False   # 0.7 turns
    n = 1_000
    yy, pp, big = np.full(n, 300.0), np.full(n, 350.0), np.full(n, 50.0)
    big[0] = 10_000.0                                        # binds the 1 s floor on one row in a thousand
    floored = lf.blend_curve(np.full(n, 50.0), big, yy, pp)
    assert floored[1.0] == pytest.approx(float(np.sqrt((300.0 - 1.0) ** 2 / n)))
    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 6.0, 8.0])
    assert lf.residual_correlation(y, y - a, y - b) == pytest.approx(1.0)
    assert lf.residual_correlation(y, y - a, y + b) == pytest.approx(-1.0)
    assert lf.BLEND_WEIGHTS == (0.3, 0.5, 0.7) and lf.CURVE_WEIGHTS == (0.0, 0.3, 0.5, 0.7, 1.0)


def _fake_worker(path, mode):
    """A stand-in for scripts/catboost_fold_worker.py: honours the argv contract, reads the npz,
    writes a constant prediction (the training-target mean) and the sidecar."""
    path.write_text(textwrap.dedent(f'''\
        import json, pathlib, sys
        import numpy as np
        a = sys.argv
        inp, out = a[a.index("--input") + 1], a[a.index("--output") + 1]
        z = np.load(inp)
        t, hold, train = z["target"], z["holdout"], z["train"]
        assert np.isnan(t[hold]).all() and np.isfinite(t[train]).all()
        n = int(hold.sum())
        mode = {mode!r}
        if mode == "crash":
            print("boom")
            sys.exit(3)
        pred = np.full(n - 1 if mode == "short" else n, float(t[train].mean()))
        np.save(out, pred)
        pathlib.Path(out).with_suffix(".json").write_text(json.dumps(
            dict(best_iteration=3, n_refit=4, peak_rss_gb=0.1, fake=True, depth=a[a.index("--depth") + 1])))
        print("peak RSS 0.10 GB")
        '''))
    return path


def test_catboost_worker_contract_input_file_subprocess_and_output_shape(tmp_path):
    """The fold hands CatBoost a .npz (X float32, target float64 with the HOLDOUT targets
    blanked to NaN so they never leave the parent, four boolean masks) and launches the
    worker as a subprocess under the venv interpreter; it checks the exit code and the
    prediction shape, and reads the sidecar. A short prediction vector, a non-zero exit and
    a missing interpreter are refusals with specific messages.

    Fails when the holdout targets are written (a worker could read them), when the exit
    code is ignored (the crash case must name the code, not a later "no predictions" error),
    or when the shape is not checked. Rehearsed 2026-09-09, each RED: dropping
    `blanked[holdout] = np.nan`; `if False:` for the exit-code check; `if False:` for the
    shape check.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4)).astype("float32")
    target = rng.normal(size=12)
    te = np.zeros(12, bool)
    te[:3] = True
    tr = ~te
    es = np.zeros(12, bool)
    es[3:5] = True
    fit = tr & ~es
    npz = tmp_path / "in.npz"
    lf.write_worker_input(npz, X, target, tr, fit, es, te)
    z = np.load(npz)
    assert set(z.files) == {"X", "target", "train", "fit", "es", "holdout"}
    assert z["X"].dtype == np.float32 and z["X"].shape == (12, 4) and z["target"].dtype == np.float64
    assert np.isnan(z["target"][te]).all() and np.array_equal(z["target"][tr], target[tr])
    assert all(z[k].dtype == bool for k in ("train", "fit", "es", "holdout"))
    with pytest.raises(ValueError, match="overlap"):
        lf.write_worker_input(npz, X, target, tr, fit, es, tr)
    with pytest.raises(ValueError, match="fit"):
        lf.write_worker_input(npz, X, target, tr, tr, es, te)
    with pytest.raises(ValueError, match="rows"):
        lf.write_worker_input(npz, X[:-1], target, tr, fit, es, te)
    lf.write_worker_input(npz, X, target, tr, fit, es, te)

    quiet = lambda m: None
    npy = tmp_path / "out.npy"
    preds, sidecar, run = lf.run_catboost_worker(sys.executable, _fake_worker(tmp_path / "ok.py", "ok"),
                                                 npz, npy, dict(lf.CATBOOST), 3, quiet)
    assert preds.shape == (3,) and preds.dtype == np.float64 and np.allclose(preds, target[tr].mean())
    assert sidecar["fake"] is True and sidecar["depth"] == "8" and run["returncode"] == 0
    cmd = run["command"]
    assert cmd[0] == sys.executable and cmd[cmd.index("--iterations") + 1] == "10000"
    assert cmd[cmd.index("--od-wait") + 1] == "200" and cmd[cmd.index("--threads") + 1] == "4"
    assert cmd[cmd.index("--learning-rate") + 1] == "0.03" and cmd[cmd.index("--seed") + 1] == "0"
    with pytest.raises(ValueError, match="shape"):
        lf.run_catboost_worker(sys.executable, _fake_worker(tmp_path / "short.py", "short"), npz,
                               tmp_path / "s.npy", dict(lf.CATBOOST), 3, quiet)
    with pytest.raises(RuntimeError, match="code 3"):
        lf.run_catboost_worker(sys.executable, _fake_worker(tmp_path / "crash.py", "crash"), npz,
                               tmp_path / "c.npy", dict(lf.CATBOOST), 3, quiet)
    with pytest.raises(FileNotFoundError, match="venv"):
        lf.run_catboost_worker(tmp_path / "no-such-python", tmp_path / "ok.py", npz, npy,
                               dict(lf.CATBOOST), 3, quiet)
    assert lf.CATBOOST == dict(depth=8, learning_rate=0.03, iterations=10_000, od_wait=200, seed=0, threads=4)
    assert str(lf.CATBOOST_PYTHON).endswith(".venvs/prc-catboost/bin/python")
    assert lf.CATBOOST_WORKER == ROOT / "scripts" / "catboost_fold_worker.py"


def test_catboost_and_lightgbm_never_share_an_interpreter():
    """CatBoost needs numpy < 2 and lives only in ~/.venvs/prc-catboost; the global
    interpreter is numpy 2.4 and shared with a live fleet. So: the fold never imports
    catboost (it launches the worker through a subprocess under the venv interpreter), the
    worker never imports lightgbm, and the fold still issues no prediction of its own.

    Fails when either import appears, or when the fold grows an in-process CatBoost path.
    Rehearsed 2026-09-09: `import catboost` added to the fold went RED.
    """
    fold = (ROOT / "scripts" / "lgbm_fold.py").read_text()
    worker = (ROOT / "scripts" / "catboost_fold_worker.py").read_text()
    assert "import catboost" not in fold and "from catboost" not in fold
    assert "lightgbm" not in worker and "lgb." not in worker
    assert "from catboost import" in worker
    assert fold.count(".predict(") == 0
    assert (ROOT / "scripts" / "lgbm_submit.py").read_text().count(".predict(") == 1
    assert "subprocess.Popen" in fold and "CATBOOST_PYTHON" in fold


@pytest.mark.skipif(not VENV_PY.exists(), reason="the isolated CatBoost venv is absent")
def test_real_catboost_worker_fits_a_micro_problem_under_the_venv(tmp_path):
    """The real worker, under the venv interpreter, on 400 rows x 5 features with a linear
    target: early-stops on the ES rows, refits on all training rows at the found count,
    writes finite holdout predictions of the right length that beat the target's own spread,
    writes the sidecar (best iteration, refit count = best + 1, row counts, peak RSS,
    versions) and prints its peak RSS. Runs in seconds; the venv's numpy must be 1.x.

    Fails when the refit is skipped (n_refit absent), when the holdout length is wrong, or
    when the venv cannot import catboost. Rehearsed 2026-09-09: `n_refit = best`
    (off by one) went RED.
    """
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 5)).astype("float32")
    target = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, 0.1, n)
    te = np.zeros(n, bool)
    te[:80] = True
    es = np.zeros(n, bool)
    es[80:160] = True
    tr = ~te
    fit = tr & ~es
    npz, npy = tmp_path / "in.npz", tmp_path / "out.npy"
    lf.write_worker_input(npz, X, target, tr, fit, es, te)
    lines = []
    preds, sidecar, run = lf.run_catboost_worker(VENV_PY, lf.CATBOOST_WORKER, npz, npy,
                                                 dict(lf.CATBOOST, iterations=60, od_wait=10), 80, lines.append)
    truth = target[te]
    assert preds.shape == (80,) and np.isfinite(preds).all()
    assert lf.rmse(truth, preds) < 0.6 * float(truth.std()), "the worker learned nothing"
    assert sidecar["best_iteration"] >= 1 and sidecar["n_refit"] == sidecar["best_iteration"] + 1
    assert (sidecar["n_fit"], sidecar["n_es"], sidecar["n_train"], sidecar["n_holdout"]) == (240, 80, 320, 80)
    assert sidecar["peak_rss_gb"] > 0 and sidecar["numpy_version"].startswith("1.")
    assert sidecar["catboost_version"] and sidecar["params"]["depth"] == 8
    assert any("peak RSS" in line for line in lines) and run["returncode"] == 0


# ---- synthetic caches with the real caches' columns and dtypes ---------------------------------
# The generator lives in tests/synthetic_caches.py (shared with tests/test_lgbm_submit.py since the
# Amendment 19 arms); the names below are the ones the tests in this file were written against.

_syn = _load_test_helper("synthetic_caches")
_INT_COLS, _KEYS = _syn.INT_COLS, _syn.KEYS
_stand_columns, _queue_block, _synthetic_caches = _syn.stand_columns, _syn.queue_block, _syn.synthetic_caches
DAY_FEATS = list(_syn.DAY_FEATS)


def _fold_env(monkeypatch, tmp_path, months=(1, 2, 3), n=600, **kw):
    stand, queue = _synthetic_caches(tmp_path / "data", months, n=n, **kw)
    monkeypatch.setattr(lf, "CACHE", stand)
    monkeypatch.setattr(lf, "QCACHE", queue)
    if hasattr(lf, "DCACHE"):
        monkeypatch.setattr(lf, "DCACHE", tmp_path / "data" / "cache_day")
    monkeypatch.setattr(lf.L, "PA_MIN_ROWS", 100)      # every synthetic airport (120 training rows) is fitted
    return stand, queue


def _recovered_rmse(preds, col):
    yhat = np.maximum(preds.proxy.to_numpy() - preds[col].to_numpy(), 1.0)
    return float(np.sqrt(((preds.y.to_numpy() - yhat) ** 2).mean()))


V4_COLUMNS = ["row", "month", "ap", "y", "proxy", "delta", "delta_hat_seed0", "delta_hat_seed1",
              "delta_hat_seed2", "pa_seed0", "pa_seed1", "pa_seed2", "pa_fitted", "A0", "A1a", "A1b", "A2", "A3"]


def test_v4_path_end_to_end_on_a_synthetic_cache(monkeypatch, tmp_path):
    """The default (v4) path on the synthetic cache: the loader was extracted for the new arms
    and this pins that the v4 path still writes its legacy files with their legacy columns,
    its three pairs, its per-airport records and a recomputable RMSE per arm.

    Fails when the refactor changes the v4 parquet columns or file names, or when the fold
    rows are not the holdout month. Rehearsed 2026-09-09: `base` written with the `sp`
    column on the v4 path went RED.
    """
    _fold_env(monkeypatch, tmp_path, queue_signal=0.0)
    out = tmp_path / "out"
    assert lf.main(["--smoke", "--out-dir", str(out)]) == 0
    j = json.loads((out / "lgbm_fold_v4.json").read_text())
    preds = pd.read_parquet(out / "fold_v4_preds.parquet")
    assert list(preds.columns) == V4_COLUMNS
    assert j["fold"]["n_test"] == 600 == len(preds) and set(preds.month) == {1}
    assert set(j["pairs"]) == {"A2_vs_A0", "A3_vs_A2", "A3_vs_A0"}
    assert j["a0_reproduction"]["status"] == "skipped (smoke)" and j["config"]["n_features"] == 68
    for arm in ("A0", "A1a", "A1b", "A2", "A3"):
        assert j["arms"][arm]["rmse"] == pytest.approx(_recovered_rmse(preds, arm), abs=1e-9), arm
    assert sorted(j["per_airport_models"]) == APTS and all(r["fitted"] for r in j["per_airport_models"].values())
    assert preds.pa_fitted.all()
    log = (out / "lgbm_fold_v4.log").read_text()
    assert lf.L.SMOKE_BANNER in log.splitlines()[0] and lf.L.SMOKE_BANNER in log.splitlines()[-1]


def test_queue_arm_end_to_end_detects_a_planted_queue_signal_and_runs_the_control(monkeypatch, tmp_path):
    """The Amendment 14 arm on a synthetic cache whose target carries 60 s x q_apt_at_push -
    visible only through the queue block. In-process baseline (A3, seeds 0-2, share rule),
    treatment = baseline + QUEUE_FEATS with best_iter re-found: the paired interval must
    exclude zero in the improving direction with a double-digit gain, >= 4 airports must
    improve, the tail subset is reported for both arms with its own interval, and the
    negative control (4 permutations here, 20 on the real run) at quarter capacity must put
    the treatment above the 95th percentile with the registered deviation on record. The
    parquet holds every column a later cut needs, and the treatment is recomputable from
    them. Then the same arm reads a v4 record as its baseline: the v4 smoke's parquet and
    JSON, the fold-identity check, the seed sd from the JSON; a v4 record from another fold
    and a mismatched per-airport rule are refused.

    Fails when the queue block is not in the treatment's design matrix (no gain), when the
    control permutes nothing (p95 equals the treatment gain), when the baseline is not the
    v4 record on the read path, or when the fold check is dropped. Rehearsed 2026-09-09,
    each RED: the treatment fitted on `X[:, :nb]` (the 68 columns); `permute_within_groups`
    replaced by a no-op (the permuted RMSEs collapse onto the treatment's); the baseline
    column overwritten with the treatment's delta; `if False:` for the fold comparison.
    """
    _fold_env(monkeypatch, tmp_path, queue_signal=60.0)
    out = tmp_path / "out"
    argv = ["--queue", "--smoke", "--out-dir", str(out), "--refit-baseline", "--n-perm", "4", "--pa-trees", "share"]
    assert lf.main(argv) == 0
    j = json.loads((out / "lgbm_fold_queue.json").read_text())
    preds = pd.read_parquet(out / "fold_preds_queue.parquet")
    log = (out / "lgbm_fold_queue.log").read_text()
    assert j["mode"] == "queue" and j["config"]["baseline_arm"] == "A3" and j["config"]["seeds"] == [0, 1, 2]
    assert j["config"]["n_features"] == 80 and j["config"]["features"][-12:] == QUEUE_FEATS
    assert j["config"]["features"][:68] == list(lf.L.FEATS) and j["config"]["queue_feats"] == QUEUE_FEATS
    assert j["baseline"]["source"] == "in-process" and j["treatment"]["best_iter"] >= 1
    p = j["pairs"]["treatment_vs_baseline"]
    assert p["gain_s"] > 10.0 and p["excludes_zero"] is True and p["improving"] is True and p["ci95"][0] > 0
    assert p["airports_improving"] >= 4 and p["at_least_4_airports"] is True and p["n_airports"] == 10
    assert set(p["named_airports"]) == {"LEBL", "LTFM", "EGLL", "LSZH"} and set(p["per_airport"]) == set(APTS)
    assert p["exceeds_2x_seed_sd"] is True and p["gain_over_seed_sd"] > 2
    assert j["seed_sd"]["source"] == "baseline single seeds (in-process)" and j["seed_sd"]["sd"] >= 0
    assert sorted(j["seed_sd"]["treatment_values"]) == ["0", "1", "2"]
    tail = j["tail_subset"]
    assert tail["n_rows"] > 0 and "60" in tail["definition"] and "1200" in tail["definition"]
    assert set(tail["rmse"]) == {"baseline", "treatment"} and tail["pair"]["ci95"][0] <= tail["pair"]["gain_s"]
    c = j["negative_control"]
    assert c["n_perm"] == 4 and c["seed"] == 0 and c["reduced_trees"] == max(1, j["treatment"]["best_iter"] // 4)
    assert len(c["permuted_gains_s"]) == 4 and c["exceeds_p95"] is True and c["treatment_below_p5"] is True
    assert c["gain_reduced_s"] > c["p95_permuted_gain_s"] and c["deviation"] == lf.CONTROL_DEVIATION
    assert c["treatment_reduced_rmse"] < c["p5_permuted_rmse"] and len(c["permuted_rmses"]) == 4
    assert c["groups"] == "airport x fold side" and c["baseline_reduced_rmse"] > c["treatment_reduced_rmse"]
    assert min(c["permuted_rmses"]) - c["treatment_reduced_rmse"] > 5.0, "the planted signal must separate clearly"
    assert "registered deviation" in log and lf.CONTROL_DEVIATION in log
    cols = (["row", "month", "ap", "y", "proxy", "delta", "sp"] + [f"baseline_seed{s}" for s in range(3)]
            + [f"baseline_pa_seed{s}" for s in range(3)] + ["baseline_pa_fitted", "baseline"]
            + [f"delta_hat_seed{s}" for s in range(3)] + [f"pa_seed{s}" for s in range(3)]
            + ["pa_fitted", "treatment", "control_baseline_red", "control_treatment_red"]
            + [f"control_perm{k}_red" for k in range(4)])
    assert list(preds.columns) == cols
    bs = [preds[f"baseline_seed{i}"].to_numpy() for i in range(3)]
    bpa = np.mean([preds[f"baseline_pa_seed{i}"].to_numpy() for i in range(3)], axis=0)
    bf = preds.baseline_pa_fitted.to_numpy()
    b2 = (bs[0] + bs[1] + bs[2]) / 3
    assert np.allclose(preds.baseline.to_numpy(), np.where(bf, 0.5 * b2 + 0.5 * bpa, b2), atol=1e-9, rtol=0)
    s = [preds[f"delta_hat_seed{i}"].to_numpy() for i in range(3)]
    a2 = (s[0] + s[1] + s[2]) / 3
    pa = np.mean([preds[f"pa_seed{i}"].to_numpy() for i in range(3)], axis=0)
    f = preds.pa_fitted.to_numpy()
    assert f.all(), "every synthetic airport is above the (patched) floor"
    assert np.allclose(preds.treatment.to_numpy(), np.where(f, 0.5 * a2 + 0.5 * pa, a2), atol=1e-9, rtol=0)
    for arm in ("baseline", "treatment"):
        assert j["arms"][arm]["rmse"] == pytest.approx(_recovered_rmse(preds, arm), abs=1e-9), arm
    assert c["baseline_reduced_rmse"] == pytest.approx(_recovered_rmse(preds, "control_baseline_red"), abs=1e-9)
    assert not (preds.treatment.to_numpy() == preds.baseline.to_numpy()).all()
    lines = [line for line in log.splitlines() if line.strip()]
    assert lf.L.SMOKE_BANNER in lines[0] and lf.L.SMOKE_BANNER in lines[-1]

    out4 = tmp_path / "v4"
    assert lf.main(["--smoke", "--out-dir", str(out4)]) == 0
    v4p = pd.read_parquet(out4 / "fold_v4_preds.parquet")
    v4j = json.loads((out4 / "lgbm_fold_v4.json").read_text())
    read = ["--queue", "--smoke", "--out-dir", str(tmp_path / "q2"), "--baseline-preds",
            str(out4 / "fold_v4_preds.parquet"), "--v4-json", str(out4 / "lgbm_fold_v4.json"),
            "--n-perm", "2", "--pa-trees", "share"]
    assert lf.main(read) == 0
    j2 = json.loads((tmp_path / "q2" / "lgbm_fold_queue.json").read_text())
    p2 = pd.read_parquet(tmp_path / "q2" / "fold_preds_queue.parquet")
    assert j2["baseline"]["source"] == "v4 predictions" and j2["baseline"]["preds"] == str(out4 / "fold_v4_preds.parquet")
    assert j2["seed_sd"]["source"] == "v4 json" and j2["seed_sd"]["sd"] == v4j["seed_sd"]["sd"]
    assert np.array_equal(p2.baseline.to_numpy(), v4p.A3.to_numpy())
    assert np.array_equal(p2.baseline_seed1.to_numpy(), v4p.delta_hat_seed1.to_numpy())
    assert np.array_equal(p2.baseline_pa_fitted.to_numpy(), v4p.pa_fitted.to_numpy())
    assert list(p2.columns) == cols[:-4] + ["control_perm0_red", "control_perm1_red"]
    assert j2["negative_control"]["n_perm"] == 2
    bad = v4p.copy()
    bad.loc[0, "y"] += 1.0
    bad.to_parquet(tmp_path / "bad.parquet", index=False)
    with pytest.raises(ValueError, match="fold"):
        lf.main(["--queue", "--smoke", "--out-dir", str(tmp_path / "q_bad"), "--n-perm", "2", "--pa-trees", "share",
                 "--baseline-preds", str(tmp_path / "bad.parquet"), "--v4-json", str(out4 / "lgbm_fold_v4.json")])
    with pytest.raises(ValueError, match="tree rule"):
        lf.main(["--queue", "--smoke", "--out-dir", str(tmp_path / "q_es"), "--n-perm", "2", "--pa-trees", "es",
                 "--baseline-preds", str(out4 / "fold_v4_preds.parquet"), "--v4-json", str(out4 / "lgbm_fold_v4.json")])
    with pytest.raises(FileNotFoundError, match="refit-baseline"):
        lf.main(["--queue", "--smoke", "--out-dir", str(tmp_path / "q3"), "--n-perm", "2",
                 "--baseline-preds", str(tmp_path / "missing.parquet")])


def test_catboost_arm_end_to_end_with_a_fake_worker(monkeypatch, tmp_path):
    """The Amendment 15.1 arm with the worker replaced by a fake that predicts the training
    mean: the fold writes the npz under the output directory (holdout targets blanked),
    frees its design matrix, launches the worker under the given interpreter, and blends
    the returned predictions 0.3/0.5/0.7 with the baseline; the JSON holds the five arms,
    the four pairs against the baseline, the weight curve at 0/0.3/0.5/0.7/1, the residual
    correlation, the worker's sidecar and exit code; the parquet holds every column.

    Fails when the blends are not the stated convex combinations, when the curve's ends are
    not the two arms, or when the worker's output is not the CatBoost column. Rehearsed
    2026-09-09, each RED: `w * base + (1 - w) * cat`; `curve[1.0] = rmses["baseline"]["pooled"]`
    after blend_curve.
    """
    _fold_env(monkeypatch, tmp_path, queue_signal=0.0)
    worker = _fake_worker(tmp_path / "fake_worker.py", "ok")
    out = tmp_path / "out"
    assert lf.main(["--catboost", "--smoke", "--out-dir", str(out), "--refit-baseline", "--baseline", "A2",
                    "--catboost-python", sys.executable, "--catboost-worker", str(worker)]) == 0
    j = json.loads((out / "lgbm_fold_catboost.json").read_text())
    preds = pd.read_parquet(out / "fold_preds_catboost.parquet")
    assert j["mode"] == "catboost" and j["config"]["baseline_arm"] == "A2"
    assert j["worker"]["sidecar"]["fake"] is True and j["worker"]["returncode"] == 0
    assert j["worker"]["command"][0] == sys.executable and j["worker"]["input"] == str(out / "catboost_fold_input.npz")
    z = np.load(out / "catboost_fold_input.npz")
    assert z["X"].shape == (1_800, 68) and np.isnan(z["target"][z["holdout"]]).all()
    assert (out / "catboost_fold_preds.npy").exists()
    assert set(j["arms"]) == {"baseline", "catboost", "blend_0.3", "blend_0.5", "blend_0.7"}
    assert set(j["pairs"]) == {"blend_0.3_vs_baseline", "blend_0.5_vs_baseline", "blend_0.7_vs_baseline",
                               "catboost_vs_baseline"}
    curve = j["blend_curve"]
    assert list(curve) == ["0.0", "0.3", "0.5", "0.7", "1.0"]
    assert curve["0.0"] == j["arms"]["baseline"]["rmse"] and curve["1.0"] == j["arms"]["catboost"]["rmse"]
    assert curve["0.5"] == j["arms"]["blend_0.5"]["rmse"]
    assert isinstance(j["curve_rmse_increases_with_weight"], bool) and -1.0 <= j["residual_correlation"] <= 1.0
    assert j["config"]["catboost"] == dict(lf.CATBOOST, iterations=lf.SMOKE_CATBOOST["iterations"],
                                           od_wait=lf.SMOKE_CATBOOST["od_wait"])
    assert list(preds.columns) == (["row", "month", "ap", "y", "proxy", "delta", "sp"]
                                   + [f"baseline_seed{s}" for s in range(3)]
                                   + ["baseline", "catboost", "blend_0.3", "blend_0.5", "blend_0.7"])
    b, c = preds.baseline.to_numpy(), preds.catboost.to_numpy()
    for w in (0.3, 0.5, 0.7):
        assert np.allclose(preds[f"blend_{w}"].to_numpy(), (1 - w) * b + w * c, atol=1e-9, rtol=0)
        assert j["arms"][f"blend_{w}"]["rmse"] == pytest.approx(_recovered_rmse(preds, f"blend_{w}"), abs=1e-9)
    assert np.allclose(c, c[0]), "the fake worker predicts one constant"
    assert j["seed_sd"]["source"] == "baseline single seeds (in-process)"
    log = (out / "lgbm_fold_catboost.log").read_text()
    assert "[catboost] peak RSS" in log and "freed the design matrix" in log


def test_sweep_screen_end_to_end_ranks_18_settings_with_the_banner_on_every_line(monkeypatch, tmp_path):
    """The Amendment 15.2 screening tier on a synthetic two-month subfold (train 2, stop on 3
    in smoke; Feb-May / Jun on the real run) at lr 0.05: 18 settings early-stopped on the
    stop month, ranked ascending by their stopping-set matched RMSE; every entry and every
    log line carries the RANKING ONLY banner; the incumbent's rank is recorded; no
    predictions parquet is written.

    Fails when a setting is missing from the ranking, when the ranking is not sorted, when
    any log line lacks the banner, or when the tier writes a parquet. Rehearsed 2026-09-09,
    each RED: `sweep_grid()[:-1]`; `suffix=None`; writing the fold's preds.
    """
    _fold_env(monkeypatch, tmp_path, months=(2, 3), n=400, queue_signal=0.0)
    out = tmp_path / "out"
    assert lf.main(["--sweep", "--smoke", "--out-dir", str(out)]) == 0
    j = json.loads((out / "lgbm_fold_sweep.json").read_text())
    log = (out / "lgbm_fold_sweep.log").read_text()
    assert j["mode"] == "sweep" and j["banner"] == lf.RANKING_BANNER and j["tier"] == "screening"
    assert j["subfold"] == {"train_months": [2], "stop_month": 3} and j["config"]["learning_rate"] == 0.05
    assert j["config"]["grid_size"] == 18 and j["config"]["n_features"] == 68
    r = j["ranking"]
    assert len(r) == 18 and [e["rank"] for e in r] == list(range(1, 19))
    vals = [e["rmse_stop_RANKING_ONLY"] for e in r]
    assert vals == sorted(vals) and all(v > 0 for v in vals)
    assert all(e["banner"] == lf.RANKING_BANNER and e["best_iter"] >= 1 for e in r)
    assert {e["setting"] for e in r} == {lf.setting_label(s) for s in lf.sweep_grid()}
    assert j["incumbent_setting"] == "255,40,0.8" and 1 <= j["incumbent_rank"] <= 18
    assert j["top_two"] == [r[0]["setting"], r[1]["setting"]]
    lines = [line for line in log.splitlines() if line.strip()]
    assert len(lines) > 18 and all(lf.RANKING_BANNER in line for line in lines)
    assert lf.L.SMOKE_BANNER in lines[0] and lf.L.SMOKE_BANNER in lines[-1]
    assert not list(out.glob("*.parquet")) and lf.output_paths("sweep", out)[2] is None
    assert j["fold"]["n_train"] == 400 and j["fold"]["n_stop"] == 400


def test_confirm_tier_end_to_end_against_the_incumbent(monkeypatch, tmp_path):
    """The Amendment 15.2 confirmation tier: each candidate setting is a full fold-A arm
    (early stopping seed 0, refit per seed, mean over seeds), paired against the incumbent
    setting's arm over the same seeds - in-process here, the v4 record when it exists - with
    the paired interval and the 2x-seed-sd rule (seed sd from the v4 JSON when present, else
    the incumbent's own three seeds); every candidate's own seed sd is recorded too.

    Fails when a candidate is fitted with the incumbent's parameters, when the pairs are not
    against the incumbent, or when the seed sd source is misreported. Rehearsed 2026-09-09,
    each RED: `setting_params(params, INCUMBENT_SETTING)` for every candidate (the two
    candidates' best_iter coincide); `source = "v4 json"` on the in-process path.
    """
    _fold_env(monkeypatch, tmp_path, queue_signal=0.0)
    out = tmp_path / "out"
    assert lf.main(["--confirm", "127,20,0.6", "511,100,0.6", "--smoke", "--out-dir", str(out),
                    "--refit-baseline"]) == 0
    j = json.loads((out / "lgbm_fold_confirm.json").read_text())
    preds = pd.read_parquet(out / "fold_preds_confirm.parquet")
    assert j["mode"] == "confirm" and j["incumbent"]["setting"] == "255,40,0.8"
    assert j["incumbent"]["source"] == "in-process" and j["incumbent"]["rmse"] > 0
    assert set(j["candidates"]) == {"127,20,0.6", "511,100,0.6"}
    for label, c in j["candidates"].items():
        nl, md, ff = label.split(",")
        assert (c["params"]["num_leaves"], c["params"]["min_data_in_leaf"], c["params"]["feature_fraction"]) \
            == (int(nl), int(md), float(ff))
        assert c["best_iter"] >= 1 and c["n_ref"] >= 1 and sorted(c["single_seed_rmses"]) == ["0", "1", "2"]
        assert c["seed_sd_own"] >= 0 and c["rmse"] == pytest.approx(_recovered_rmse(preds, label), abs=1e-9)
    assert j["candidates"]["127,20,0.6"]["best_iter"] != j["candidates"]["511,100,0.6"]["best_iter"] or \
        j["candidates"]["127,20,0.6"]["rmse"] != j["candidates"]["511,100,0.6"]["rmse"]
    assert set(j["pairs"]) == {"127,20,0.6_vs_incumbent", "511,100,0.6_vs_incumbent"}
    for p in j["pairs"].values():
        assert p["ci95"][0] <= p["gain_s"] <= p["ci95"][1] and isinstance(p["exceeds_2x_seed_sd"], bool)
        assert p["n_airports"] == 10 and isinstance(p["airports_improving"], int)
    assert j["seed_sd"]["source"] == "incumbent single seeds (in-process)" and j["seed_sd"]["sd"] >= 0
    assert list(preds.columns) == (["row", "month", "ap", "y", "proxy", "delta", "sp", "incumbent_seed0",
                                    "incumbent_seed1", "incumbent_seed2", "incumbent"]
                                   + [f"127,20,0.6_seed{s}" for s in range(3)] + ["127,20,0.6"]
                                   + [f"511,100,0.6_seed{s}" for s in range(3)] + ["511,100,0.6"])
    assert j["incumbent"]["rmse"] == pytest.approx(_recovered_rmse(preds, "incumbent"), abs=1e-9)

    out4 = tmp_path / "v4"
    assert lf.main(["--smoke", "--out-dir", str(out4)]) == 0
    v4p = pd.read_parquet(out4 / "fold_v4_preds.parquet")
    v4j = json.loads((out4 / "lgbm_fold_v4.json").read_text())
    assert lf.main(["--confirm", "127,20,0.6", "--smoke", "--out-dir", str(tmp_path / "c2"),
                    "--baseline-preds", str(out4 / "fold_v4_preds.parquet"),
                    "--v4-json", str(out4 / "lgbm_fold_v4.json")]) == 0
    j2 = json.loads((tmp_path / "c2" / "lgbm_fold_confirm.json").read_text())
    p2 = pd.read_parquet(tmp_path / "c2" / "fold_preds_confirm.parquet")
    assert j2["incumbent"]["source"] == "v4 predictions" and j2["seed_sd"]["source"] == "v4 json"
    assert j2["seed_sd"]["sd"] == v4j["seed_sd"]["sd"]
    assert np.array_equal(p2.incumbent.to_numpy(), v4p.A2.to_numpy())
    assert list(p2.columns)[7:11] == ["incumbent_seed0", "incumbent_seed1", "incumbent_seed2", "incumbent"]


# =============================================================================================
# Amendment 16: the schedule-fill mixture head. Treatment = the shipped v4 arm + a LightGBM
# binary head p = P(|y - sp| <= 60 s | x) on FEATS + nmdelay, mixed as p * sp + (1 - p) *
# max(proxy - baseline, 1). Reported per 16.2/16.3: the pooled and per-airport paired interval,
# the FILL and NON-FILL holdout subsets with their own intervals, LIRF alone, the reliability
# deciles of p, the 2 x seed sd check against the baseline's seed sd.
# =============================================================================================

def test_fillhead_mode_names_its_outputs_and_is_exclusive_with_the_other_arms():
    """--fillhead is the Amendment 16 arm: its own json + log under reports/ and
    fold_preds_fillhead.parquet under data/cache_stand/, never another mode's names; one arm
    per run; --baseline A2/A3 like the other arms; the registered command reads the v4 record
    with the per-airport rule the v4 fold ran (`--pa-trees es`); the head's design is
    FEATS + [nmdelay] (69 columns).

    Fails when the mode inherits another mode's names or when two arms are accepted.
    Rehearsed 2026-09-09, each RED: `"fillhead": OUTPUT_NAMES["queue"]` (the queue names);
    `("fillhead", args.fillhead)` left out of _arm_flags (exclusivity and mode_of share it).
    """
    assert lf.mode_of(lf.parse_args(["--fillhead"])) == "fillhead"
    assert lf.parse_args(["--fillhead"]).baseline == "A3"
    assert lf.parse_args(["--fillhead", "--baseline", "A2"]).baseline == "A2"
    assert lf.parse_args([]).fillhead is False and lf.mode_of(lf.parse_args([])) == "v4"
    for argv in (["--fillhead", "--queue"], ["--fillhead", "--catboost"], ["--fillhead", "--sweep"],
                 ["--fillhead", "--confirm", "255,40,0.8"]):
        with pytest.raises(SystemExit):
            lf.parse_args(argv)
    out = pathlib.Path("/o")
    assert lf.output_paths("fillhead", out) == (out / "lgbm_fold_fillhead.json", out / "lgbm_fold_fillhead.log",
                                                out / "fold_preds_fillhead.parquet")
    j, l, p = lf.output_paths("fillhead", None)
    assert j == lf.REPORTS / "lgbm_fold_fillhead.json" and l == lf.REPORTS / "lgbm_fold_fillhead.log"
    assert p == lf.CACHE / "fold_preds_fillhead.parquet"
    assert len({lf.OUTPUT_NAMES[m] for m in lf.MODES}) == len(lf.MODES), "two modes share an output name"
    assert "fillhead" in lf.MODES and "fillhead" in lf.RUNNERS and "fillhead" in lf.COMMANDS and "fillhead" in lf.ESTIMATES
    assert "--fillhead --baseline A3 --pa-trees es" in lf.COMMANDS["fillhead"]
    assert "lgbm_fold_fillhead.console.log" in lf.COMMANDS["fillhead"]
    assert lf.FEATS_HEAD == list(lf.L.FEATS) + ["nmdelay"] and len(lf.FEATS_HEAD) == 69


def test_fill_mask_is_the_complement_of_the_tail_rule_non_fill_and_partitions_the_holdout():
    """Amendment 16.2's fill subset is |y - sp| <= 60 s inclusive - exactly the complement of
    the tail rule's non-fill (|y - sp| > 60, Amendment 14.3), so the two amendments cannot
    disagree about a row: fill + non-fill = every holdout row, the boundary row (exactly 60 s)
    is a fill, and the tolerance is lgbm_submit.FILL_TOL_S - the same number the head's training
    label uses. The tail is a subset of the non-fills.

    Fails when the rule is strict (the 60 s row then belongs to neither subset) or when the two
    constants drift apart. Rehearsed 2026-09-09, each RED: `< L.FILL_TOL_S` in fill_mask;
    `TAIL_NONFILL_S = 61.0`.
    """
    y = np.full(6, 1_000.0)
    sp = np.array([1_000.0, 1_060.0, 940.0, 1_060.001, 939.999, 5_000.0])
    fill = lf.fill_mask(y, sp)
    assert fill.dtype == bool and fill.tolist() == [True, True, True, False, False, False]
    nonfill = ~fill
    assert int(fill.sum()) + int(nonfill.sum()) == 6 and not (fill & nonfill).any()
    assert np.array_equal(nonfill, np.abs(y - sp) > lf.TAIL_NONFILL_S)
    assert lf.TAIL_NONFILL_S == lf.L.FILL_TOL_S == 60.0
    tail = lf.tail_mask(y, sp, np.full(6, 2_000.0))
    assert not (tail & fill).any() and tail.tolist() == [False, False, False, True, True, True]


def test_reliability_deciles_are_equal_count_bins_of_p_with_mean_p_and_realised_fill_rate():
    """Reliability of p on the holdout: the rows sorted by p (stable) and cut into ten
    equal-count bins; per bin the count, the p range, the mean p and the realised fill rate;
    plus the overall mean p and fill rate (a calibrated head has them equal). Pinned by hand
    on 100 rows with p = i/99 and fills exactly where p > 0.5 (i = 50..99, fifty rows): bins of
    10, mean p rising, fill rate 0 in the first five bins and 1 in the last five, overall fill
    rate 0.50 and mean p 0.50.
    Ties break by row order so the bins are always equal-count (a p that is constant on 91% of
    the rows would otherwise collapse the quantile bins - pinned with 91 rows at 0.01). Fewer
    than ten rows, or mismatched lengths, are refusals.

    Fails when the bins are fixed-width (91 rows at 0.01 then fall into one bin), when the fill
    rate is computed on every row, or when the sort is descending. Rehearsed 2026-09-09, each
    RED: `np.linspace(0, 1, 11)` edges with np.digitize; `fill.mean()` for every bin; `order[::-1]`.
    """
    p = np.arange(100) / 99.0
    fill = p > 0.5
    r = lf.reliability_deciles(p, fill)
    assert r["n_rows"] == 100 and r["n_bins"] == 10 and len(r["deciles"]) == 10
    assert [d["n"] for d in r["deciles"]] == [10] * 10 and sum(d["n"] for d in r["deciles"]) == 100
    assert [d["decile"] for d in r["deciles"]] == list(range(1, 11))
    means = [d["mean_p"] for d in r["deciles"]]
    assert means == sorted(means)
    assert means[0] == pytest.approx(np.mean(p[:10])) and means[-1] == pytest.approx(np.mean(p[90:]))
    assert [d["fill_rate"] for d in r["deciles"]] == [0.0] * 5 + [1.0] * 5
    assert r["deciles"][0]["p_min"] == 0.0 and r["deciles"][-1]["p_max"] == 1.0
    assert r["deciles"][4]["p_max"] == pytest.approx(49 / 99) and r["deciles"][5]["p_min"] == pytest.approx(50 / 99)
    assert r["mean_p"] == pytest.approx(0.5) and r["fill_rate"] == pytest.approx(0.50)
    skew = np.r_[np.full(91, 0.01), np.linspace(0.6, 1.0, 9)]
    rs = lf.reliability_deciles(skew, skew > 0.5)
    assert [d["n"] for d in rs["deciles"]] == [10] * 10
    assert rs["deciles"][-1]["fill_rate"] == 0.9 and rs["deciles"][0]["fill_rate"] == 0.0
    assert rs["deciles"][-1]["p_min"] == 0.01
    with pytest.raises(ValueError, match="ten"):
        lf.reliability_deciles(p[:9], fill[:9])
    with pytest.raises(ValueError, match="length"):
        lf.reliability_deciles(p, fill[:99])
    assert lf.N_DECILES == 10


def test_subset_record_scores_both_arms_on_the_masked_rows_with_their_own_paired_interval():
    """A subset record (the fill / non-fill cuts, LIRF's cuts) holds the row count and share,
    the subset's share of the baseline's SSE, the RMSE of each arm on the subset's rows only,
    and the paired interval of treatment - baseline on those rows; an empty subset has no RMSE
    and no interval, a single row has an RMSE but no interval (a bootstrap needs two rows).
    Pinned by hand on four rows: baseline errors 3, 4, 0, 10 and treatment errors 0, 0, 0, 10 on
    a mask of the first two rows -> RMSE 3.5355 vs 0, SSE share 25 / 125, gain 3.5355.

    Fails when the RMSEs are pooled instead of masked, or when the interval is drawn on every
    row. Rehearsed 2026-09-09, each RED: `v.mean()` for `v[mask].mean()`; the pair drawn on
    `se` instead of the masked `se`.
    """
    se = {"baseline": np.array([9.0, 16.0, 0.0, 100.0]), "treatment": np.array([0.0, 0.0, 0.0, 100.0])}
    mask = np.array([True, True, False, False])
    r = lf.subset_record("first two", mask, se, n_boot=200)
    assert r["definition"] == "first two" and r["n_rows"] == 2 and r["share_of_holdout_rows"] == 0.5
    assert r["share_of_baseline_sse"] == pytest.approx(25 / 125)
    assert r["rmse"]["baseline"] == pytest.approx(np.sqrt(12.5)) and r["rmse"]["treatment"] == 0.0
    assert r["pair"]["gain_s"] == pytest.approx(np.sqrt(12.5)) and r["pair"]["n_draws"] == 200
    assert r["pair"]["ci95"][0] <= r["pair"]["gain_s"] <= r["pair"]["ci95"][1]
    assert r["pair"]["ci95"][0] >= 3.0 - 1e-9, "an interval drawn on every row would reach down to the 0-gain rows"
    assert r["pair"]["ci95"][1] <= 4.0 + 1e-9
    one = lf.subset_record("one row", np.array([False, False, False, True]), se, n_boot=200)
    assert one["n_rows"] == 1 and one["rmse"] == {"baseline": 10.0, "treatment": 10.0} and one["pair"] is None
    none = lf.subset_record("no rows", np.zeros(4, bool), se, n_boot=200)
    assert none["n_rows"] == 0 and none["rmse"] is None and none["pair"] is None


def test_load_fold_derive_hook_appends_nmdelay_as_the_last_design_column(monkeypatch, tmp_path):
    """load_fold(..., derive=L.add_nmdelay) adds nmdelay = proxy - sp to the frame after the
    in-fold encodings and before the design matrix, so column 68 of the 69-column matrix is
    exactly float32(proxy - sp) row by row and the first 68 columns are the regressor's own
    matrix; the regressors then read X[:, :68] and the head reads all of X. Without the hook a
    request for nmdelay is refused - there is no such column in the caches.

    Fails when the hook is not called (KeyError: the matrix cannot be built) or when the column
    lands elsewhere. Rehearsed 2026-09-09: `if False:` around the derive call went RED.
    """
    stand, _ = _fold_env(monkeypatch, tmp_path, queue_signal=0.0)
    fold = lf.load_fold(stand, (1, 2, 3), lf.FEATS_HEAD, derive=lf.L.add_nmdelay)
    X = fold["X"]
    assert X.shape == (1_800, 69) and fold["feats"] == lf.FEATS_HEAD
    assert np.array_equal(X[:, 68], (fold["proxy"] - fold["sp"]).astype(np.float32))
    assert not np.array_equal(X[:, 68], X[:, 0]), "nmdelay must not be a copy of proxy"
    base = lf.load_fold(stand, (1, 2, 3), list(lf.L.FEATS))
    assert np.array_equal(X[:, :68], base["X"])
    with pytest.raises(KeyError):
        lf.load_fold(stand, (1, 2, 3), lf.FEATS_HEAD)


def test_fillhead_arm_end_to_end_gains_on_the_planted_fills_and_leaves_the_non_fills_alone(monkeypatch, tmp_path):
    """The Amendment 16 arm on a synthetic cache whose fills are PLANTED: sp == y exactly on the
    rows with f_callsign > 1.28 (about 10%), every other row's sp at least 120 s from y, so the
    60 s label is exactly the plant and one feature identifies it. In-process baseline (A3,
    seeds 0-2, share rule), the head fitted on the training rows, treatment = the mixture:
      * the FILL subset (|y - sp| <= 60 on the holdout) must gain >= 5 s with a paired interval
        excluding zero - the mechanism must show where it is claimed (16.3);
      * the NON-FILL subset must not be worse by more than 0.5 s - 16.3's "must not poison the
        majority", one-sided as the amendment writes it (on this fixture the small p left on the
        tail rows pulls them toward y, so the non-fills may gain a little: measured +1.0 s);
      * fill + non-fill partition the 600 holdout rows exactly;
      * the reliability deciles are computed on the 600 holdout rows only (their counts sum to
        600, not to the 1,800 rows loaded), the top decile is almost all fills, the bottom almost none;
      * p is in [0, 1]; the treatment column is the delta form of the mixture, i.e.
        max(proxy - treatment, 1) == p * sp + (1 - p) * max(proxy - baseline, 1) row by row;
      * the pooled pair carries the 2 x seed sd check against the baseline's seed sd; the head is
        single-seed and says so; LIRF is reported alone with its own fill / non-fill cuts;
      * the JSON's RMSEs are recomputable from the parquet; the banner opens and closes the log.
    Then the arm reads a v4 record as its baseline (the v4 smoke's parquet + JSON on the same
    synthetic cache): baseline == the v4 A3 column, seed sd from the JSON, A2 on request; a v4
    record from a different fold is refused.

    Fails when the head is disabled (p = 0: the fill gain vanishes), when the mixture weights
    are swapped (the non-fill subset is poisoned by about -100 s), when the deciles are cut on
    the wrong rows, or when the treatment is not the mixture. Rehearsed 2026-09-09, each RED:
    `p = np.zeros_like(p)` after the head fit; `L.mix_fill(1.0 - p, ...)`;
    `reliability_deciles(p, fill_mask(y, sp))` (every row: the length check); `treat = b["delta"]`.
    """
    _fold_env(monkeypatch, tmp_path, queue_signal=0.0, fill_feature="f_callsign")
    out = tmp_path / "out"
    assert lf.main(["--fillhead", "--smoke", "--out-dir", str(out), "--refit-baseline", "--pa-trees", "share"]) == 0
    j = json.loads((out / "lgbm_fold_fillhead.json").read_text())
    preds = pd.read_parquet(out / "fold_preds_fillhead.parquet")
    log = (out / "lgbm_fold_fillhead.log").read_text()
    assert j["mode"] == "fillhead" and j["amendment"] == "16" and j["config"]["baseline_arm"] == "A3"
    assert j["config"]["seeds"] == [0, 1, 2] and j["config"]["features"] == lf.FEATS_HEAD
    assert j["config"]["n_features"] == 69 and j["config"]["features_regressor"] == list(lf.L.FEATS)
    assert j["config"]["fill_tolerance_s"] == 60.0 and j["config"]["nmdelay"] == "nmdelay"
    assert j["config"]["params_head"]["objective"] == "binary" and j["config"]["params_head"]["metric"] == "auc"
    assert j["config"]["params_head"]["learning_rate"] == lf.SMOKE["learning_rate"]
    assert j["baseline"]["source"] == "in-process"
    h = j["head"]
    assert h["best_iter"] >= 1 and h["n_ref"] >= 50 and h["n_features"] == 69
    assert h["n_all"] == 1_200 and h["n_fit"] == 600 and h["n_es"] == 600
    assert 0.05 < h["fill_share_train"] < 0.2 and h["es_auc_NOT_A_RESULT"] > 0.9 and h["holdout_auc"] > 0.95
    p = preds.p.to_numpy()
    assert len(preds) == 600 == j["fold"]["n_test"] and ((p >= 0) & (p <= 1)).all()
    fill = np.abs(preds.y.to_numpy() - preds.sp.to_numpy()) <= 60.0
    sub = j["subsets"]
    assert sub["fill"]["n_rows"] == int(fill.sum()) and sub["nonfill"]["n_rows"] == int((~fill).sum())
    assert sub["fill"]["n_rows"] + sub["nonfill"]["n_rows"] == 600
    assert 30 <= sub["fill"]["n_rows"] <= 90, "the plant is about 10% of the holdout"
    assert "60" in sub["fill"]["definition"] and "60" in sub["nonfill"]["definition"]
    fp, nfp = sub["fill"]["pair"], sub["nonfill"]["pair"]
    assert fp["gain_s"] >= 5.0 and fp["ci95"][0] > 0 and fp["improving"] is True
    assert sub["fill"]["rmse"]["treatment"] < sub["fill"]["rmse"]["baseline"]
    assert nfp["gain_s"] >= -0.5, f"the head poisoned the non-fills: {nfp['gain_s']:+.3f} s"
    assert nfp["gain_s"] <= 3.0, f"a non-fill gain this large is not the head's mechanism: {nfp['gain_s']:+.3f} s"
    pooled = j["pairs"]["treatment_vs_baseline"]
    assert pooled["gain_s"] > 0 and pooled["improving"] is True and isinstance(pooled["exceeds_2x_seed_sd"], bool)
    assert pooled["gain_over_seed_sd"] is not None and pooled["n_airports"] == 10
    assert set(pooled["per_airport"]) == set(APTS)
    assert j["seed_sd"]["source"] == "baseline single seeds (in-process)" and j["seed_sd"]["treatment_sd"] is None
    assert sorted(j["seed_sd"]["baseline_values"]) == ["0", "1", "2"]
    rel = j["reliability"]
    assert rel["n_rows"] == 600 and sum(d["n"] for d in rel["deciles"]) == 600 and len(rel["deciles"]) == 10
    assert rel["deciles"][-1]["fill_rate"] >= 0.8 and rel["deciles"][0]["fill_rate"] <= 0.1
    assert rel["deciles"][-1]["mean_p"] > 0.8 and rel["deciles"][0]["mean_p"] < 0.1
    assert rel["fill_rate"] == pytest.approx(fill.mean())
    lirf = j["lirf"]
    assert lirf["n_rows"] == 60 and set(lirf["rmse"]) == {"baseline", "treatment"}
    assert lirf["fill"]["n_rows"] + lirf["nonfill"]["n_rows"] == 60
    assert lirf["pair"] == pooled["per_airport"]["LIRF"]
    assert lirf["rmse"]["baseline"] == j["arms"]["baseline"]["per_airport"]["LIRF"]
    hold = j["holdout"]
    assert hold["n_rows"] == 600 and hold["auc"] == h["holdout_auc"] and hold["fill_share"] == pytest.approx(fill.mean())
    assert hold["mean_p_on_fills"] > hold["mean_p_on_nonfills"] + 0.5
    cols = (["row", "month", "ap", "y", "proxy", "delta", "sp"] + [f"baseline_seed{s}" for s in range(3)]
            + [f"baseline_pa_seed{s}" for s in range(3)] + ["baseline_pa_fitted", "baseline", "p", "treatment"])
    assert list(preds.columns) == cols
    base_yhat = np.maximum(preds.proxy.to_numpy() - preds.baseline.to_numpy(), 1.0)
    mix = p * preds.sp.to_numpy() + (1.0 - p) * base_yhat
    got = np.maximum(preds.proxy.to_numpy() - preds.treatment.to_numpy(), 1.0)
    assert np.allclose(got, np.maximum(mix, 1.0), atol=1e-9, rtol=0)
    for arm in ("baseline", "treatment"):
        assert j["arms"][arm]["rmse"] == pytest.approx(_recovered_rmse(preds, arm), abs=1e-9), arm
    assert "mixture" in j["arms"]["treatment"]["definition"]
    lines = [line for line in log.splitlines() if line.strip()]
    assert lf.L.SMOKE_BANNER in lines[0] and lf.L.SMOKE_BANNER in lines[-1]
    assert "reliability deciles" in log and "fill subset" in log and "non-fill subset" in log and "LIRF" in log
    assert "fill head" in log and "single-seed" in log

    out4 = tmp_path / "v4"
    assert lf.main(["--smoke", "--out-dir", str(out4)]) == 0
    v4p = pd.read_parquet(out4 / "fold_v4_preds.parquet")
    v4j = json.loads((out4 / "lgbm_fold_v4.json").read_text())
    read = ["--fillhead", "--smoke", "--out-dir", str(tmp_path / "f2"), "--baseline-preds",
            str(out4 / "fold_v4_preds.parquet"), "--v4-json", str(out4 / "lgbm_fold_v4.json"), "--pa-trees", "share"]
    assert lf.main(read) == 0
    j2 = json.loads((tmp_path / "f2" / "lgbm_fold_fillhead.json").read_text())
    p2 = pd.read_parquet(tmp_path / "f2" / "fold_preds_fillhead.parquet")
    assert j2["baseline"]["source"] == "v4 predictions" and j2["baseline"]["preds"] == str(out4 / "fold_v4_preds.parquet")
    assert j2["seed_sd"]["source"] == "v4 json" and j2["seed_sd"]["sd"] == v4j["seed_sd"]["sd"]
    assert np.array_equal(p2.baseline.to_numpy(), v4p.A3.to_numpy())
    assert np.array_equal(p2.baseline_pa_fitted.to_numpy(), v4p.pa_fitted.to_numpy())
    assert list(p2.columns) == cols and ((p2.p.to_numpy() >= 0) & (p2.p.to_numpy() <= 1)).all()
    assert j2["subsets"]["fill"]["pair"]["gain_s"] >= 5.0 and j2["subsets"]["nonfill"]["pair"]["gain_s"] >= -0.5
    assert lf.main(read[:3] + [str(tmp_path / "f3")] + read[4:] + ["--baseline", "A2"]) == 0
    p3 = pd.read_parquet(tmp_path / "f3" / "fold_preds_fillhead.parquet")
    assert np.array_equal(p3.baseline.to_numpy(), v4p.A2.to_numpy())
    assert list(p3.columns) == cols[:7] + [f"baseline_seed{s}" for s in range(3)] + ["baseline", "p", "treatment"]
    bad = v4p.copy()
    bad.loc[0, "y"] += 1.0
    bad.to_parquet(tmp_path / "bad.parquet", index=False)
    with pytest.raises(ValueError, match="fold"):
        lf.main(["--fillhead", "--smoke", "--out-dir", str(tmp_path / "f_bad"), "--pa-trees", "share",
                 "--baseline-preds", str(tmp_path / "bad.parquet"), "--v4-json", str(out4 / "lgbm_fold_v4.json")])
    with pytest.raises(ValueError, match="tree rule"):
        lf.main(["--fillhead", "--smoke", "--out-dir", str(tmp_path / "f_es"), "--pa-trees", "es",
                 "--baseline-preds", str(out4 / "fold_v4_preds.parquet"), "--v4-json", str(out4 / "lgbm_fold_v4.json")])
