"""Tests for scripts/adsb_stack.py — arm ADN's harness: day folds, arm S, stackers G / N, day-block scoring, clauses.

Written before the module (red first). Synthetic worlds only; every test names the mutation it must kill.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("adsb_stack", ROOT / "scripts" / "adsb_stack.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["adsb_stack"] = mod
    spec.loader.exec_module(mod)
    return mod


st = _load()


def _world(n_days=10, per_day=300, signal=1.0, seed=0):
    """A fold-like frame: F predicts delta with error e; ADS-B feature `x` (joined rows only) carries signal*e."""
    rng = np.random.default_rng(seed)
    days = [f"2025-01-{d:02d}" for d in range(1, n_days // 2 + 1)] + [f"2025-07-{d:02d}" for d in range(1, n_days // 2 + 1)]
    rows = []
    for d in days:
        for _ in range(per_day):
            rows.append(d)
    n = len(rows)
    delta = rng.normal(0, 200, n)
    e = rng.normal(0, 150, n)
    F = delta - e                                   # F's delta_hat; residual (delta - F) = e
    proxy = np.abs(delta) + 1200 + rng.uniform(0, 300, n)
    joined = rng.uniform(size=n) < 0.6
    x = np.where(joined, signal * e + rng.normal(0, 30, n), np.nan)
    ap = rng.choice(["EHAM", "LSZH", "EDDM"], n)
    fr = pd.DataFrame({"day": rows, "ap": ap, "hour": rng.integers(0, 24, n), "y": proxy - delta, "proxy": proxy,
                       "delta": delta, "F": F, "coverage": np.where(joined, 3, 1), "x": x,
                       "noise": np.where(joined, rng.normal(0, 1, n), np.nan)})
    fr["month"] = fr.day.str[5:7].astype(int)
    fr["fold"] = st.day_folds(fr.day)
    return fr


# ---------------------------------------------------------------- folds

def test_day_folds_hold_both_months_exclude_the_spent_day_and_are_deterministic():
    """Kills: month-pure folds; the spent day entering a fold; a random assignment."""
    days = pd.Series([f"2025-01-{d:02d}" for d in range(1, 32)] + [f"2025-07-{d:02d}" for d in range(1, 32)])
    f = st.day_folds(days)
    assert np.isnan(f[days == "2025-01-09"]).all()
    for k in range(5):
        dk = days[f == k]
        assert dk.str.startswith("2025-01").any() and dk.str.startswith("2025-07").any()
    assert (np.nan_to_num(st.day_folds(days), nan=-1) == np.nan_to_num(f, nan=-1)).all()
    ok = days[days != "2025-01-09"]
    assert st.day_folds(ok).tolist() == [i % 5 for i in range(30)] + [i % 5 for i in range(31)]


# ---------------------------------------------------------------- arm S

def test_arm_s_anchors_on_the_dwell_end_when_one_exists_and_biases_from_training_folds_only():
    """ADN.2: taxi_S = (dwell_end_rel_mvt if finite else first_ground_rel_mvt) - b, b = training folds' median of
    (anchor - y) per airport. Kills: first_ground used where a dwell exists; b computed with the test fold."""
    fr = pd.DataFrame({"fold": [0, 0, 1, 1, 2, 2], "ap": ["EHAM"] * 6, "coverage": [3] * 6,
                       "y": [600.0, 600, 600, 600, 600, 9999], "proxy": [2000.0] * 6, "F": [1000.0] * 6,
                       "dwell_end_rel_mvt": [np.nan, 700, np.nan, np.nan, np.nan, np.nan],
                       "first_ground_rel_mvt": [650.0, 5000, 650, 650, 650, 650]})
    d = st.arm_s(fr)
    # fold 0's b from folds 1, 2: anchor - y = 50, 50, 50, -9349 -> median 50
    assert d[0] == pytest.approx(2000 - (650 - 50))
    assert d[1] == pytest.approx(2000 - (700 - 50))                    # the dwell end, not the 5,000 s first sample
    # fold 2's b from folds 0 and 1: 50, 100, 50, 50 -> median 50; the test fold's own 9,999 never enters
    assert d[4] == pytest.approx(2000 - 600)


def test_arm_s_leaves_rows_not_seen_from_the_stand_at_f():
    """Kills: applying the sensor to mid-taxi (code 2) or unjoined rows."""
    fr = pd.DataFrame({"fold": [0, 1, 1], "ap": ["EHAM"] * 3, "coverage": [2, 3, 1], "y": [600.0] * 3,
                       "proxy": [2000.0] * 3, "F": [1111.0] * 3, "dwell_end_rel_mvt": [np.nan] * 3,
                       "first_ground_rel_mvt": [650.0] * 3})
    d = st.arm_s(fr)
    assert d[0] == 1111.0 and d[2] == 1111.0


# ---------------------------------------------------------------- stackers are out-of-fold and leave unjoined rows at F

@pytest.mark.parametrize("arm", ["G", "N"])
def test_stacker_learns_a_planted_signal_and_is_out_of_fold(arm):
    """A feature carrying the residual must cut the error on joined rows; a test fold's labels must not move its own
    predictions. Kills: fitting on the test fold; ignoring the feature; touching unjoined rows."""
    fr = _world(signal=1.0)
    pred = st.fit_stacker(fr, ["x"], arm=arm, seed=0)
    j = fr.coverage >= 2
    base = np.sqrt(np.mean((fr.delta - fr.F)[j] ** 2))
    mine = np.sqrt(np.mean((fr.delta - pred)[j] ** 2))
    assert mine < 0.6 * base
    assert (pred[~j.to_numpy()] == fr.F[~j].to_numpy()).all()
    fr2 = fr.copy()
    k0 = fr2.fold == 0
    fr2.loc[k0, "delta"] = fr2.loc[k0, "delta"].sample(frac=1, random_state=1).to_numpy()   # scramble fold-0 labels
    pred2 = st.fit_stacker(fr2, ["x"], arm=arm, seed=0)
    np.testing.assert_allclose(pred2[k0.to_numpy()], pred[k0.to_numpy()], rtol=0, atol=1e-6)  # exact refit, fixed seed


def test_nn_aborts_on_a_non_finite_loss():
    """In-flight runtime guard. Kills: training on through an inf target. The guard is two layers in
    adsb_nn_worker.train_predict (a target pre-check and the in-loop loss check); removing either ONE alone is an
    equivalent mutant because the other still aborts (rehearsed 2026-09-11: M10 survived by design)."""
    fr = _world()
    victim = fr.index[(fr.coverage >= 2) & (fr.fold == 1)][0]      # a joined row that is in fold 0's TRAINING set
    fr.loc[victim, "delta"] = np.inf
    with pytest.raises(FloatingPointError, match="non-finite"):
        st.fit_stacker(fr, ["x"], arm="N", seed=0)


# ---------------------------------------------------------------- scoring

def test_taxi_recovery_is_floored_proxy_minus_delta():
    """BC-2 convention. Kills: a missing floor; using delta as a taxi time."""
    np.testing.assert_array_equal(st.taxi(np.array([1000.0, 50.0]), np.array([400.0, 80.0])), [600.0, 1.0])


def test_day_block_bootstrap_detects_a_planted_gain_and_rejects_noise():
    """Kills: a bootstrap that is not paired, or resamples rows instead of dates, or mixes months."""
    fr = _world(n_days=20, per_day=200)
    y = fr.y.to_numpy()
    base = st.taxi(fr.proxy.to_numpy(), fr.F.to_numpy())
    good = y + 0.3 * (base - y)
    lo, hi = st.day_block_ci(y, base, good, fr.day.to_numpy(), n_draws=400, seed=0)
    assert lo > 0
    rng = np.random.default_rng(3)
    noisy = base + rng.normal(0, 1.0, len(y))
    lo2, hi2 = st.day_block_ci(y, base, noisy, fr.day.to_numpy(), n_draws=400, seed=0)
    assert lo2 < 0 < hi2


def test_clauses_and_verdict():
    """C1..C5 as registered. Kills: WORKING with a failed clause; NOT WORKING when only a later clause fails."""
    ok = dict(ci=(0.4, 1.9), net=800.0, month_gain={1: 0.5, 7: 1.2}, ap_gain={"EHAM": 2.0, "LSZH": 0.4, "LTFM": 0.0},
              gain=1.1, seed_sd=0.2)
    assert st.verdict(st.clauses(**ok)) == "WORKING"
    assert st.verdict(st.clauses(**{**ok, "ci": (-0.1, 1.0)})) == "NOT WORKING"
    assert st.verdict(st.clauses(**{**ok, "net": 300.0})) == "INCONCLUSIVE"
    assert st.verdict(st.clauses(**{**ok, "month_gain": {1: -0.1, 7: 1.0}})) == "INCONCLUSIVE"
    assert st.verdict(st.clauses(**{**ok, "ap_gain": {"EHAM": 3.0, "LSZH": -1.2}})) == "INCONCLUSIVE"
    assert st.verdict(st.clauses(**{**ok, "seed_sd": 0.6})) == "INCONCLUSIVE"
    assert st.clauses(**{**ok, "seed_sd": None})["C5"] is None                    # arm S has no seeds: C5 n/a
    assert st.verdict(st.clauses(**{**ok, "seed_sd": None})) == "WORKING"


def test_net_weighted_fold_mse_matches_the_registered_formula():
    """net = 0.98466 * dSSE / n_rows. Kills: a sign flip or a missing weight."""
    y = np.array([0.0, 0.0]); base = np.array([10.0, 10.0]); arm = np.array([0.0, 10.0])
    assert st.net_weighted(y, base, arm) == pytest.approx(0.98466 * (200 - 100) / 2)


def test_regression_bc5_torch_never_shares_a_process_with_lightgbm():
    """BC-5 (two OpenMP runtimes in one process), surfaced 2026-09-11 in this harness: fitting arm G (lightgbm, its own
    libomp) then arm N (torch, its bundled libomp) in ONE process deadlocked in torch indexing (all OpenMP workers parked
    in __kmp_fork_barrier, 0% CPU). Fix: the NN trains in scripts/adsb_nn_worker.py, a subprocess that never imports
    lightgbm. This test runs G then N in a fresh interpreter under a timeout and checks torch never entered it.
    Kills: training the NN in-process (hangs -> timeout)."""
    import subprocess, textwrap
    code = textwrap.dedent(f"""
        import importlib.util, sys
        sys.path.insert(0, {str(ROOT / 'tests')!r})
        spec = importlib.util.spec_from_file_location('adsb_stack', {str(ROOT / 'scripts' / 'adsb_stack.py')!r})
        st = importlib.util.module_from_spec(spec); sys.modules['adsb_stack'] = st; spec.loader.exec_module(st)
        import test_adsb_stack as T
        fr = T._world(n_days=10, per_day=120)
        st.fit_stacker(fr, ['x'], arm='G', seed=0)
        st.fit_stacker(fr, ['x'], arm='N', seed=0)
        import tempfile, pathlib
        md = pathlib.Path(tempfile.mkdtemp(dir={str(ROOT / 'data')!r})) / 'm'
        st.train_artifacts(fr, ['x'], md, seeds=(0,), gate={{'EHAM': True}})
        st.predict_residual(fr, md)
        import shutil; shutil.rmtree(md.parent)
        print('TORCH_IN_PARENT', 'torch' in sys.modules, 'LGB_IN_PARENT', 'lightgbm' in sys.modules)
    """)
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=90,
                       cwd=str(ROOT))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "TORCH_IN_PARENT False LGB_IN_PARENT True" in r.stdout


def test_arm_s_bias_never_sees_the_test_fold():
    """Kills: b_ap computed over all folds including the test fold (M1). Fold 2 holds the majority of rows and a
    different offset, so including it would move the median."""
    fr = pd.DataFrame({"fold": [0, 1, 2, 2, 2], "ap": ["EHAM"] * 5, "coverage": [3] * 5,
                       "y": [600.0, 600.0, 600.0, 600.0, 600.0], "proxy": [3000.0] * 5, "F": [1.0] * 5,
                       "dwell_end_rel_mvt": [np.nan] * 5,
                       "first_ground_rel_mvt": [600.0, 600.0, 1600.0, 1600.0, 1600.0]})
    d = st.arm_s(fr)
    assert d[2] == pytest.approx(3000 - (1600 - 0))          # b from folds 0 and 1 only (offset 0), not 1,000


def test_day_block_bootstrap_is_paired():
    """Kills: an unpaired bootstrap (M5). Strong day effects shared by both arms, and a small but consistent gain:
    only a paired resample (the same dates for both arms) keeps the interval above zero."""
    rng = np.random.default_rng(7)
    days = np.repeat([f"2025-01-{d:02d}" for d in range(1, 16)] + [f"2025-07-{d:02d}" for d in range(1, 16)], 100)
    day_err = pd.Series(rng.normal(0, 400, 30), index=pd.unique(days)).reindex(days).to_numpy()
    y = np.zeros(len(days))
    base = y + day_err + rng.normal(0, 50, len(days))
    arm = y + 0.97 * (base - y)
    lo, hi = st.day_block_ci(y, base, arm, days, n_draws=500, seed=0)
    assert lo > 0


def test_smoke_subset_keeps_the_first_days_of_each_month_and_all_five_folds():
    """The smoke run uses the first 5 fold-days of each month. Kills: a subset missing a month or a fold (the stacker
    would silently skip it); keeping the spent day."""
    days = pd.Series([f"2025-01-{d:02d}" for d in range(1, 32)] + [f"2025-07-{d:02d}" for d in range(1, 32)])
    fr = pd.DataFrame({"day": days, "fold": st.day_folds(days)})
    sub = st.smoke_subset(fr, per_month=5)
    assert sorted(sub.day.unique()) == ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05",
                                        "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04", "2025-07-05"]
    assert sorted(sub.fold.unique()) == [0, 1, 2, 3, 4]


def test_score_arm_compares_against_the_given_baseline_not_f():
    """ADN.5: the ADS-B claim is arm_ads vs arm_base (the same stacker without ADS-B). Kills: scoring always against F,
    which credits the stacker's own recalibration of F to ADS-B (the smoke's K0 failure)."""
    fr = _world(n_days=10, per_day=200)
    base = fr.delta.to_numpy() - 0.5 * (fr.delta - fr.F).to_numpy()        # a recalibrated F: already better than F
    same = base.copy()
    vs_f = st.score_arm(fr, same)
    vs_base = st.score_arm(fr, same, base_delta=base)
    assert vs_f["gain_s"] > 10                                              # looks like a big win against F...
    assert vs_base["gain_s"] == pytest.approx(0.0, abs=1e-9)                # ...and is nothing against its own base
    assert vs_base["verdict"] == "NOT WORKING"
    assert vs_base["rmse_base"] == pytest.approx(vs_base["rmse_arm"])


def test_airport_transfer_never_trains_on_the_holdout_airport():
    """ADN.7: predictions for holdout airport A come from models that never saw A's rows. Kills: training on A;
    predicting rows outside A; leaving A at F."""
    fr = _world(n_days=10, per_day=200)
    A = "EHAM"
    pred = st.fit_stacker(fr, ["x"], arm="G", seed=0, train_rows=(fr.ap != A).to_numpy(),
                          predict_rows=(fr.ap == A).to_numpy())
    isA = (fr.ap == A).to_numpy() & (fr.coverage >= 2).to_numpy()
    assert (pred[~isA] == fr.F.to_numpy()[~isA]).all()
    assert (pred[isA] != fr.F.to_numpy()[isA]).mean() > 0.9
    fr2 = fr.copy()
    fr2.loc[fr2.ap == A, "delta"] = fr2.loc[fr2.ap == A, "delta"].sample(frac=1, random_state=3).to_numpy()
    pred2 = st.fit_stacker(fr2, ["x"], arm="G", seed=0, train_rows=(fr2.ap != A).to_numpy(),
                           predict_rows=(fr2.ap == A).to_numpy())
    np.testing.assert_allclose(pred2[isA], pred[isA], rtol=0, atol=1e-9)       # A's labels never entered its model


def test_transfer_rule_as_registered():
    """ADN.7: >= 3 of 4 airports need gain >= 0.5 x in-sample AND a CI excluding 0. Kills: a >= 2 rule; ignoring the CI;
    a 0.25 fraction."""
    good = {"gain_s": 20.0, "ci95_day_block": (1.0, 30.0)}
    res = {a: dict(good) for a in st.TRANSFER_AIRPORTS}
    assert st.transfer_rule(res)["PASS"] is True
    res["EHAM"] = {"gain_s": 15.0, "ci95_day_block": (1.0, 30.0)}             # 15 < 0.5 x 31.63: fails there
    res["EDDM"] = {"gain_s": 20.0, "ci95_day_block": (-0.1, 30.0)}            # CI spans 0: fails there
    r = st.transfer_rule(res)
    assert r["n_pass"] == 2 and r["PASS"] is False
    assert st.transfer_rule({**res, "EHAM": {"gain_s": 15.9, "ci95_day_block": (1.0, 2.0)}})["per_airport_pass"]["EHAM"]


# ---------------------------------------------------------------- ship artefacts (the pipeline hand-off contract)

def test_artefacts_round_trip_and_predict_residual_obeys_the_contract(tmp_path):
    """train_artifacts saves 5 folds x seeds per learner; predict_residual reloads them and returns the B residual
    (mean of G and N) in seconds: 0.0 where coverage < 2 or the airport is not gated in, finite elsewhere, and equal to
    the in-memory CV-bag. Kills: a gate or coverage row getting a non-zero residual; artefacts that do not reproduce."""
    fr = _world(n_days=10, per_day=150)
    md = tmp_path / "models"
    st.train_artifacts(fr, ["x"], md, seeds=(0,), gate={"EHAM": True, "LSZH": True, "EDDM": False})
    import json
    man = json.loads((md / "MANIFEST.json").read_text())
    assert man["feats"] == ["x"] and man["seeds"] == [0] and man["gate"]["EDDM"] is False
    assert len(list(md.glob("G_s0_k*.txt"))) == 5 and len(list(md.glob("N_s0_k*.pt"))) == 5
    res = st.predict_residual(fr, md)
    ok = (fr.coverage >= 2).to_numpy() & fr.ap.isin(["EHAM", "LSZH"]).to_numpy()
    assert (res[~ok] == 0.0).all() and np.isfinite(res[ok]).all() and (res[ok] != 0).mean() > 0.9
    again = st.predict_residual(fr, md)
    np.testing.assert_array_equal(res, again)                                   # deterministic reload


def test_predict_residual_refuses_a_frame_missing_a_contract_column(tmp_path):
    """Kills: silently predicting without F / proxy / coverage (a NaN input would become a real residual)."""
    fr = _world(n_days=10, per_day=60)
    md = tmp_path / "models"
    st.train_artifacts(fr, ["x"], md, seeds=(0,), gate={"EHAM": True})
    with pytest.raises(ValueError, match="missing contract column"):
        st.predict_residual(fr.drop(columns=["F"]), md)


def test_predict_residual_reads_the_gate_table_form(tmp_path):
    """The manifest gate may be {ap: bool} or the gate_table form {ap: {allowed, rule, ...}}. Kills: a dict entry being
    read as truthy (every airport would ship)."""
    import json
    fr = _world(n_days=10, per_day=60)
    md = tmp_path / "models"
    st.train_artifacts(fr, ["x"], md, seeds=(0,), gate={})
    man = json.loads((md / "MANIFEST.json").read_text())
    man["gate"] = {"EHAM": {"allowed": True, "rule": "0.7x"}, "LSZH": {"allowed": False, "rule": "0.7x"}}
    (md / "MANIFEST.json").write_text(json.dumps(man))
    res = st.predict_residual(fr, md)
    assert (res[(fr.ap == "LSZH").to_numpy()] == 0.0).all()
    assert (res[((fr.ap == "EHAM") & (fr.coverage >= 2)).to_numpy()] != 0.0).mean() > 0.9


def test_predict_residual_oof_mode_reproduces_the_in_memory_cv_blend(tmp_path):
    """With oof=True each row is scored only by the models whose held-out fold contains it, which reproduces the
    harness's out-of-fold B = mean(G, N) (seed means). Kills: scoring 2025 rows with the full bag (mostly in-sample:
    each model saw 3 of 5 folds), which would make the reproduction look better than it is."""
    fr = _world(n_days=10, per_day=150)
    md = tmp_path / "models"
    gate = {"EHAM": True, "LSZH": True, "EDDM": True}
    st.train_artifacts(fr, ["x"], md, seeds=(0,), gate=gate)
    g = st.fit_stacker(fr, ["x"], "G", 0) - fr.F.to_numpy()
    n = st.fit_stacker(fr, ["x"], "N", 0) - fr.F.to_numpy()
    want = np.where((fr.coverage >= 2).to_numpy(), 0.5 * (g + n), 0.0)
    got = st.predict_residual(fr, md, oof=True)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-4)      # float32 NN inputs through npz: 1e-4 s is far below
    full = st.predict_residual(fr, md)                             # the full bag is NOT the OOF answer
    assert np.abs(full - want).max() > 1e-3


def test_regression_predict_g_never_applies_the_booster_below_coverage_2(tmp_path):
    """Bug found 2026-09-18 building scripts/analysis/acx_g_build.py (arm ACX-G): `_predict_g` used to apply the
    shipped booster to EVERY row with a finite fold, with no coverage check at all -- unlike `fit_stacker`'s own
    `joined = (coverage >= 2) & isfinite(fold)` mask, which never assigns a real residual to a coverage<2 row.
    `predict_residual` only got the right SERVED answer because it applied its own `coverage>=2 & gate` mask
    AFTERWARD; any direct caller of `_predict_g` (bypassing `predict_residual`, as an analysis script legitimately
    does to inspect a component in isolation) that didn't reapply at least `coverage>=2` got nonzero, unvetted
    predictions on rows the model was never trained to predict on. Concretely: an ungated pull of `_predict_g`/
    `_predict_n` corrupted a blended-RMSE reproduction by ~5 seconds (214.03 vs an independently confirmed
    209.27) before the mask was found. Fixed by making `coverage` a REQUIRED argument that `_predict_g` masks
    with internally, so this class of misuse is no longer possible from outside.

    Kills: `_predict_g` (or `_combine`) losing its internal coverage mask -- this test fails if `_predict_g`
    returns a nonzero residual on a coverage<2 row, while `fit_stacker`'s own OOF value for that identical row is
    exactly 0 (that row is outside `joined`, so `fit_stacker` never touches it)."""
    fr = _world(n_days=10, per_day=150)
    md = tmp_path / "models"
    st.train_artifacts(fr, ["x"], md, seeds=(0,), gate={"EHAM": True, "LSZH": True, "EDDM": True})
    import json
    man = json.loads((md / "MANIFEST.json").read_text())
    X = st._design(fr, ["x"])
    fold = fr.fold.to_numpy(float)
    cov = fr.coverage.to_numpy()
    cov_lt2 = (cov < 2) & np.isfinite(fold)
    assert cov_lt2.any()   # the fixture must actually exercise this population, or the test proves nothing

    g_fit_stacker_resid = st.fit_stacker(fr, ["x"], "G", 0) - fr.F.to_numpy()
    assert (g_fit_stacker_resid[cov_lt2] == 0.0).all()   # ground truth: fit_stacker never predicts here

    g_direct = st._predict_g(X, md, man, cov, fold)
    n_bad = int((g_direct[cov_lt2] != 0).sum())
    assert n_bad == 0, (
        f"_predict_g applied the shipped booster to {n_bad} of {int(cov_lt2.sum())} coverage<2 rows "
        "(missing coverage>=2 mask)")

    with pytest.raises(TypeError):
        st._predict_g(X, md, man)   # `coverage` must be required, not silently defaultable/omittable


def test_regression_predict_n_never_applies_the_network_below_coverage_2(tmp_path):
    """Same class and same call site as test_regression_predict_g_never_applies_the_booster_below_coverage_2, for
    the N (neural) learner -- `_predict_n` had the identical missing-mask defect, fixed in the same commit."""
    fr = _world(n_days=10, per_day=150)
    md = tmp_path / "models"
    st.train_artifacts(fr, ["x"], md, seeds=(0,), gate={"EHAM": True, "LSZH": True, "EDDM": True})
    import json
    man = json.loads((md / "MANIFEST.json").read_text())
    X = st._design(fr, ["x"])
    fold = fr.fold.to_numpy(float)
    cov = fr.coverage.to_numpy()
    cov_lt2 = (cov < 2) & np.isfinite(fold)
    assert cov_lt2.any()

    n_fit_stacker_resid = st.fit_stacker(fr, ["x"], "N", 0) - fr.F.to_numpy()
    assert (n_fit_stacker_resid[cov_lt2] == 0.0).all()

    n_direct = st._predict_n(X, md, man, cov, fold)
    n_bad = int((n_direct[cov_lt2] != 0).sum())
    assert n_bad == 0, (
        f"_predict_n applied the shipped network to {n_bad} of {int(cov_lt2.sum())} coverage<2 rows "
        "(missing coverage>=2 mask)")

    with pytest.raises(TypeError):
        st._predict_n(X, md, man)   # `coverage` must be required, not silently defaultable/omittable


# ---------------------------------------------------------------- arm ADU (unmatched lane, taxi scale)

def test_unmatched_mapping_recovers_taxi_and_s1c_through_the_matched_machinery():
    """ADU maps delta := -y, F := -S1C, proxy := 0 so ADN's taxi() returns y and S1C. Kills: a sign slip (the stacker
    would learn the residual backwards)."""
    rec = pd.DataFrame({"y": [900.0, 2000.0], "S1C": [800.0, 2500.0]})
    m = st.map_unmatched(rec)
    np.testing.assert_allclose(st.taxi(m.proxy, m.delta), [900.0, 2000.0])
    np.testing.assert_allclose(st.taxi(m.proxy, m.F), [800.0, 2500.0])
    np.testing.assert_allclose(m.delta - m.F, -(rec.y - rec.S1C))


def test_score_unmatched_prices_at_2026_rows_excludes_the_top_airport_day_and_caps_airport_losses():
    """C2's net = dSSE / 344,841 x (5,290 / n); event-excluded drops the airport-day with the largest dSSE;
    C4 fails if any airport's MSE worsens > 5%. Kills: matched weighting; no event exclusion; a missing C4 cap."""
    fr = pd.DataFrame({"ap": ["EHAM"] * 4 + ["LSZH"] * 4, "day": ["2025-01-01"] * 2 + ["2025-07-01"] * 2
                       + ["2025-01-02"] * 2 + ["2025-07-02"] * 2, "month": [1, 1, 7, 7, 1, 1, 7, 7],
                       "fold": [0.0] * 8, "y": [1000.0] * 8, "proxy": [0.0] * 8})
    base = -np.array([1100.0] * 8)                      # base errs by 100 s everywhere
    arm = -np.array([1000, 1000, 1100, 1100, 1000, 1000, 1010, 1010.0])   # fixes EHAM Jan fully, LSZH mostly
    r = st.score_unmatched(fr, arm, base)
    dsse = (100 ** 2 - 0) * 2 + 0 + (100 ** 2) * 2 + (100 ** 2 - 10 ** 2) * 2
    assert r["net_full"] == pytest.approx(dsse / 344_841 * (5_290 / 8))
    top = (100 ** 2) * 2                                 # EHAM 2025-01-01 and LSZH 2025-01-02 tie; one is removed
    assert r["net_event_excluded"] == pytest.approx((dsse - top) / 344_841 * (5_290 / 8))
    worse = -np.array([1000, 1000, 1100, 1100, 1100, 1100, 1300, 1300.0])  # LSZH much worse than base
    assert st.score_unmatched(fr, worse, base)["clauses"]["C4"] is False


# ---------------------------------------------------------------- arm ADX (plans/PREREG_adsb_taxistart_2026_09_11.md)

def _adx_frame():
    """8 scored rows at one airport: 4 joined mid-taxi (coverage 2), 2 seen from the stand (3), 2 uncovered (1)."""
    n = 8
    return pd.DataFrame({"y": np.array([600.0, 700, 800, 900, 500, 550, 400, 450]),
                         "proxy": np.full(n, 1500.0), "F": np.full(n, 800.0),
                         "coverage": np.array([2, 2, 2, 2, 3, 3, 1, 1]),
                         "day": ["2025-01-02"] * 4 + ["2025-01-03"] * 4, "month": [1] * 8,
                         "ap": ["EHAM"] * 8, "fold": [0.0] * 8})


def test_subset_net_scales_a_subsets_gain_by_the_FULL_scored_row_count():
    """C6 is in weighted FOLD MSE units, so a subset's net is its own SSE change over ALL scored rows, not over the
    subset. Kills: dividing by the subset size (here 4 instead of 8, doubling the number); counting unscored rows;
    the sign flipped."""
    st = _load()
    fr = _adx_frame()
    base = np.full(len(fr), 800.0)
    arm = base.copy()
    arm[fr.coverage.to_numpy() == 3] = 700.0                       # only the code-3 rows move, by +100 s of delta
    net_sub = st.subset_net(fr, fr.coverage.to_numpy() == 3, base, arm)
    y, px = fr.y.to_numpy(), fr.proxy.to_numpy()
    m = fr.coverage.to_numpy() == 3
    d = np.sum((st.taxi(px[m], base[m]) - y[m]) ** 2) - np.sum((st.taxi(px[m], arm[m]) - y[m]) ** 2)
    assert net_sub == pytest.approx(st.MATCHED_WEIGHT * d / 8, rel=1e-12)
    assert st.subset_net(fr, fr.coverage.to_numpy() == 3, arm, base) == pytest.approx(-net_sub, rel=1e-12)


def test_adx_breakdown_reports_each_coverage_code_separately():
    """The pre-registered breakdown. Kills: code 2 and 3 pooled; uncovered rows counted as moved."""
    st = _load()
    fr = _adx_frame()
    base = np.full(len(fr), 800.0)
    arm = base.copy()
    arm[fr.coverage.to_numpy() == 2] = 750.0
    b = st.adx_breakdown(fr, base, arm)
    assert set(b) == {"2", "3"}
    assert b["2"]["n"] == 4 and b["3"]["n"] == 2
    assert b["2"]["net"] != 0.0 and b["3"]["net"] == 0.0


def test_c6_refuses_an_arm_that_pays_for_code_2_out_of_code_3():
    """C6 (the no-harm bar): code-3 net >= -150 weighted fold MSE. Kills: C6 missing from the verdict; the bar applied
    to code 2; -150 itself failing (the bar is inclusive)."""
    st = _load()
    base = {"C1": True, "C2": True, "C3": True, "C4": True, "C5": True}
    assert st.clauses_adx(base, code3_net=0.0)["C6"] is True
    assert st.clauses_adx(base, code3_net=-150.0)["C6"] is True
    assert st.clauses_adx(base, code3_net=-150.1)["C6"] is False
    assert st.verdict(st.clauses_adx(base, code3_net=-200.0)) == "INCONCLUSIVE"
    assert st.verdict(st.clauses_adx({**base, "C1": False}, code3_net=0.0)) == "NOT WORKING"
    assert st.verdict(st.clauses_adx(base, code3_net=-10.0)) == "WORKING"


def test_run_adx_compares_the_block_against_its_own_twin_and_records_c6(tmp_path, monkeypatch):
    """The whole arm on a stub fitter: the twin gets the ADN features, the arm gets them PLUS the block, the breakdown
    and C6 come from the same fit, and the record names the early-stopping convention.
    Kills: the twin fitted with the same feature list as the arm (no comparison); C6 read from a later fit; the record
    written without the block or the controls; an existing report overwritten."""
    st = _load()
    fr = _adx_frame()
    fr["delta"], fr["MVT_ID_mvt"] = fr.F + 50.0, np.arange(len(fr), dtype=float)
    twin_feats = ["first_ground_rel_aobt3"]
    arm_feats = twin_feats + ["travel_s_first_to_stand"]
    for c in arm_feats:
        fr[c] = np.linspace(0.0, 1.0, len(fr))
    seen = []

    def fake_fit(frame, feats, arm, seed, **kw):
        seen.append(tuple(feats))
        d = frame.F.to_numpy(float).copy()
        if "travel_s_first_to_stand" in feats:                     # the arm: exact on code-2 rows, WORSE on code-3
            m = frame.coverage.to_numpy() == 2
            d[m] = (frame.proxy.to_numpy(float) - frame.y.to_numpy(float))[m]
            d[frame.coverage.to_numpy() == 3] -= 20.0              # pays for code 2 out of code 3: C6 must catch it
        return d
    monkeypatch.setattr(st, "fit_stacker", fake_fit)
    out = tmp_path / "adx.json"
    assert st.run_adx(fr, twin_feats, arm_feats, seeds=(0,), out_json=out) == 0
    import json
    rec = json.loads(out.read_text())
    assert rec["block"] == ["travel_s_first_to_stand"]
    assert set(rec["arms"]) == {"G", "N", "B"} and "C6" in rec["arms"]["G"]["clauses"]
    assert rec["arms"]["G"]["breakdown_by_coverage"]["2"]["net"] > 0
    assert rec["arms"]["G"]["breakdown_by_coverage"]["3"]["net"] < st.ADX_C6_BAR      # code 3 harmed
    assert rec["arms"]["G"]["clauses"]["C6"] is False and rec["arms"]["G"]["verdict"] != "WORKING"
    assert "early_stopping" in rec and rec["controls"]["K_zero"]["pass"] in (True, False)
    assert any("travel_s_first_to_stand" in f for f in seen) and any("travel_s_first_to_stand" not in f for f in seen)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        st.run_adx(fr, twin_feats, arm_feats, seeds=(0,), out_json=out)


# ---------------------------------------------------------------- arm ADJ (plans/PREREG_adsb_join_recall_2026_09_12.md)

def _adj_frame():
    """8 scored rows: 2 joined today (coverage 3), 3 recovered by the second join (coverage 2), 3 never joined."""
    n = 8
    return pd.DataFrame({"y": np.array([600.0, 700, 800, 900, 500, 550, 400, 450]),
                         "proxy": np.full(n, 1500.0), "F": np.full(n, 800.0), "delta": np.full(n, 850.0),
                         "coverage": np.array([3, 3, 2, 2, 2, 1, 1, 1]),
                         "join_mvt_anchored": np.array([0.0, 0, 1, 1, 1, np.nan, np.nan, np.nan]),
                         "feat": np.arange(n, dtype=float),
                         "day": ["2025-01-02"] * 4 + ["2025-01-03"] * 4, "month": [1] * 8,
                         "ap": ["EHAM"] * 8, "fold": [0.0] * 8})


def test_the_twin_puts_the_recovered_rows_back_to_unjoined_and_leaves_everything_else_alone():
    """ADJ's twin is 'the same stacker on today's coverage'. Kills: the twin built by dropping rows (changing the
    fit population); rows joined today also reset; the input frame mutated in place."""
    st = _load()
    fr = _adj_frame()
    tw = st.today_coverage(fr)
    assert tw.coverage.tolist() == [3, 3, 1, 1, 1, 1, 1, 1]
    assert fr.coverage.tolist() == [3, 3, 2, 2, 2, 1, 1, 1]          # the caller's frame is untouched
    assert len(tw) == len(fr) and tw.y.tolist() == fr.y.tolist()


def test_c7_refuses_an_arm_that_pays_for_new_rows_out_of_the_rows_already_joined():
    """C7, the no-harm bar on rows joined today (net >= -150). Kills: C7 missing; the bar read off the new rows;
    -150 itself failing."""
    st = _load()
    base = {"C1": True, "C2": True, "C3": True, "C4": True, "C5": True}
    assert st.clauses_adj(base, old_rows_net=-150.0)["C7"] is True
    assert st.clauses_adj(base, old_rows_net=-150.1)["C7"] is False
    assert st.verdict(st.clauses_adj(base, old_rows_net=-400.0)) == "INCONCLUSIVE"
    assert st.verdict(st.clauses_adj(base, old_rows_net=0.0)) == "WORKING"


def test_the_noise_control_permutes_only_the_rows_under_test():
    """BC-10 applied: ADJ's K0 must scramble the RECOVERED rows' features and leave every other row's untouched,
    because the twin keeps those. Kills: permuting the whole column (that destroys the ADN signal the twin has and the
    control reads as a huge loss, exactly how ADX's harness died); permuting nothing."""
    st = _load()
    fr = _adj_frame()
    new = fr.join_mvt_anchored.to_numpy() == 1
    out = st.permute_rows(fr, ["feat"], new, seed=0)
    assert out.feat[~new].tolist() == fr.feat[~new].tolist()
    assert sorted(out.feat[new].tolist()) == sorted(fr.feat[new].tolist())
    assert out.feat[new].tolist() != fr.feat[new].tolist() or len(fr) < 3


def test_run_adj_fits_the_twin_on_todays_coverage_and_records_both_row_groups(tmp_path, monkeypatch):
    """The whole arm on a stub fitter. Kills: the twin fitted on the extended coverage (no comparison); C7 read from
    the recovered rows; the record missing the row counts or the controls; an existing report overwritten."""
    st = _load()
    fr = _adj_frame()
    fr["MVT_ID_mvt"] = np.arange(len(fr), dtype=float)
    seen = []

    def fake_fit(frame, feats, arm, seed, **kw):
        seen.append(int((frame.coverage.to_numpy() >= 2).sum()))
        d = frame.F.to_numpy(float).copy()
        m = frame.coverage.to_numpy() >= 2
        d[m] = (frame.proxy.to_numpy(float) - frame.y.to_numpy(float))[m]     # exact wherever it may act
        return d
    monkeypatch.setattr(st, "fit_stacker", fake_fit)
    out = tmp_path / "adj.json"
    assert st.run_adj(fr, ["feat"], seeds=(0,), out_json=out) == 0
    import json
    rec = json.loads(out.read_text())
    assert rec["n_recovered"] == 3 and rec["n_joined_today"] == 2
    assert 2 in seen and 5 in seen                                            # twin sees 2 joined rows, the arm 5
    g = rec["arms"]["G"]
    assert g["net_recovered_rows"] > 0 and g["clauses"]["C7"] is True
    assert g["net_rows_joined_today"] == pytest.approx(0.0, abs=1e-9)         # unchanged rows, so C7 is comfortable
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        st.run_adj(fr, ["feat"], seeds=(0,), out_json=out)


def test_c7_catches_an_arm_that_improves_recovered_rows_while_harming_the_rows_joined_today(tmp_path, monkeypatch):
    """The case C7 exists for. Kills: C7 computed from the RECOVERED rows (then this arm passes); C7 computed but not
    fed into the verdict."""
    st = _load()
    fr = _adj_frame()
    fr["MVT_ID_mvt"] = np.arange(len(fr), dtype=float)

    def harming_fit(frame, feats, arm, seed, **kw):
        d = frame.F.to_numpy(float).copy()
        m = frame.coverage.to_numpy() >= 2
        d[m] = (frame.proxy.to_numpy(float) - frame.y.to_numpy(float))[m]      # exact wherever it may act ...
        if m.sum() > 2:                                                        # ... but the extended fit also spoils
            d[frame.coverage.to_numpy() == 3] += 30.0                          #     the rows that were already joined
        return d
    monkeypatch.setattr(st, "fit_stacker", harming_fit)
    out = tmp_path / "adj_harm.json"
    st.run_adj(fr, ["feat"], seeds=(0,), out_json=out)
    import json
    g = json.loads(out.read_text())["arms"]["G"]
    assert g["net_recovered_rows"] > 0 and g["net_rows_joined_today"] < st.ADJ_C7_BAR
    assert g["clauses"]["C7"] is False and g["verdict"] != "WORKING"


def test_the_adj_noise_control_passes_when_noise_loses_and_fails_when_noise_wins(tmp_path, monkeypatch):
    """ADJ.1: K0 passes iff its net is < +150. Against this twin a noise arm must LOSE, so a negative net is a pass;
    only noise that WINS fails. Kills: the old rule (interval spanning zero) that no ADJ run can satisfy; a bar that
    accepts a winning noise arm."""
    st = _load()
    fr = _adj_frame()
    fr["MVT_ID_mvt"] = np.arange(len(fr), dtype=float)
    import json

    def fit(frame, feats, arm, seed, **kw):
        d = frame.F.to_numpy(float).copy()
        m = frame.coverage.to_numpy() >= 2
        d[m] = (frame.proxy.to_numpy(float) - frame.y.to_numpy(float))[m]
        if "feat" in feats and frame.feat.is_monotonic_increasing is False:      # the permuted (noise) fit
            d[m] = frame.F.to_numpy(float)[m] + 400.0                            # noise loses badly
        return d
    monkeypatch.setattr(st, "fit_stacker", fit)
    out = tmp_path / "adj_k0.json"
    st.run_adj(fr, ["feat"], seeds=(0,), out_json=out)
    k0 = json.loads(out.read_text())["controls"]["K_zero"]
    assert k0["net"] < 0 and k0["pass"] is True                                  # noise lost: the control passes
    assert st.ADJ_K0_BAR == 150.0


def test_predict_residual_ships_the_learner_the_manifest_names(tmp_path, monkeypatch):
    """SHIP-ADJ ships the TREES alone, because only J_G has a verdict. The manifest must say which learner ships and
    predict_residual must obey it — and default to the blend, so v11/v12's models keep their meaning.
    Kills: the blend hard-coded (the MLP would ship with no verdict); an unknown learner accepted silently; the
    default changing for a manifest without the key."""
    st = _load()
    calls = {}
    monkeypatch.setattr(st, "_predict_g", lambda X, md, man, coverage, fold=None: (calls.setdefault("g", 0), np.full(len(X), 10.0))[1])
    monkeypatch.setattr(st, "_predict_n", lambda X, md, man, coverage, fold=None: (calls.setdefault("n", 0), np.full(len(X), 20.0))[1])
    fr = pd.DataFrame({"ap": ["EHAM", "EHAM"], "hour": [10.0, 11], "proxy": [900.0, 800], "F": [600.0, 500],
                       "coverage": [3, 3], "feat": [1.0, 2.0]})
    import json
    md = tmp_path / "models"; md.mkdir()
    base = {"feats": ["feat"], "seeds": [0], "folds": 1, "gate": {"EHAM": {"allowed": True}}, "target_scale": 300.0,
            "nn": {"hidden": 8, "dropout": 0.1, "batch": 64}}
    (md / "MANIFEST.json").write_text(json.dumps({**base, "ship_learner": "G"}))
    np.testing.assert_allclose(st.predict_residual(fr, md), [10.0, 10.0])
    assert "n" not in calls                                            # the MLP is never even run
    (md / "MANIFEST.json").write_text(json.dumps(base))                # no key: the blend, as v11/v12 shipped
    np.testing.assert_allclose(st.predict_residual(fr, md), [15.0, 15.0])
    (md / "MANIFEST.json").write_text(json.dumps({**base, "ship_learner": "Q"}))
    with pytest.raises(ValueError, match="ship_learner"):
        st.predict_residual(fr, md)


def test_ship_delta_compares_two_gated_configurations_not_two_raw_predictions():
    """SHIP-ADJ.1: v13 vs v12 is a difference of two SHIPPED configurations — each prediction applies only where that
    version's coverage and gate allow it, and F elsewhere. Kills: the gate ignored (LIRF then counts although v13 keeps
    F there); the coverage mask taken from the wrong version; F not used outside the gate."""
    st = _load()
    fr = pd.DataFrame({"ap": ["EHAM", "LIRF", "EHAM", "LTFM"], "coverage_old": [3, 3, 1, 1],
                       "coverage": [3, 3, 2, 1], "F": [800.0, 800, 800, 800]})
    pred = np.array([700.0, 600, 750, 500])
    out = st.gated_prediction(pred, fr.F.to_numpy(float), fr.coverage.to_numpy(), {"EHAM": True, "LIRF": False},
                              fr.ap.to_numpy())
    assert out.tolist() == [700.0, 800.0, 750.0, 800.0]          # LIRF closed -> F; LTFM uncovered -> F
    old = st.gated_prediction(pred, fr.F.to_numpy(float), fr.coverage_old.to_numpy(),
                              {"EHAM": True, "LIRF": True}, fr.ap.to_numpy())
    assert old.tolist() == [700.0, 600.0, 800.0, 800.0]          # v12: LIRF open, row 3 not yet joined


def test_the_ship_comparison_can_ship_the_blend_as_v_next(tmp_path, monkeypatch):
    """SHIP-ADJ2 ships B on the extended coverage, not G. Kills: v_next hard-coded to the trees (the package would be
    measured as SHIP-ADJ's, which already failed); the CURRENT side changed away from the blend."""
    st = _load()
    fr = _adj_frame()
    fr["MVT_ID_mvt"] = np.arange(len(fr), dtype=float)

    def fit(frame, feats, arm, seed, **kw):
        d = frame.F.to_numpy(float).copy()
        m = frame.coverage.to_numpy() >= 2
        d[m] = frame.F.to_numpy(float)[m] + (100.0 if arm == "G" else 300.0)      # the two learners differ
        return d
    monkeypatch.setattr(st, "fit_stacker", fit)
    gates = ({"EHAM": True}, {"EHAM": True})
    out_g, out_b = tmp_path / "g.json", tmp_path / "b.json"
    st.run_adj(fr, ["feat"], seeds=(0,), out_json=out_g, ship_gates=gates, next_learner="G")
    st.run_adj(fr, ["feat"], seeds=(0,), out_json=out_b, ship_gates=gates, next_learner="B")
    import json
    g = json.loads(out_g.read_text())["arms"]["SHIP_v_next_vs_current"]
    b = json.loads(out_b.read_text())["arms"]["SHIP_v_next_vs_current"]
    assert g["next_learner"] == "G" and b["next_learner"] == "B"
    assert g["net_weighted_fold_mse"] != b["net_weighted_fold_mse"]                # the choice reaches the number


# ---------------------------------------------------------------- arm AFJ: the standardised-input clip (defence in depth)

def test_regression_nn_matrix_clips_standardised_inputs_at_z_clip_and_leaves_everything_below_it_bit_identical():
    """Regression (reports/RESEARCH_ADSBFI_JULY_2026_09_16.md §3; plans/PREREG_adsbfi_july_fallback_2026_09_16.md §2): a
    ground_rate_pm of 5e8 standardised to z ~ 3e7 and the MLP returned -1.9e6 s. _nn_matrix now clips z to ±Z_CLIP = 300,
    the next hundred above the largest |z| any shipped table ever fed the MLP (272.6: ground_rate_pm 12,000 under fold
    stats). Below the clip the matrix is bit-identical to the unclipped computation (float32 of (x - mu) / sd), the missing
    indicator and one-hot are untouched, and a NaN input still maps to 0 with its indicator set. Fails when the clip is
    removed, or when it is set at the report's ±10 (the |z| = 250 column would change: 459 / 347 shipped joined rows sit
    above 10). Rehearsed 2026-09-16 17:1x EDT (backup, break, run, restore byte-identical): clip removed -> RED; ±10 -> RED;
    a clip applied to the whole matrix (indicators included) SURVIVES at Z_CLIP = 300 because the indicator and one-hot
    columns are in {0, 1}: an equivalent mutant, not a coverage gap."""
    assert st.Z_CLIP == 300.0
    rng = np.random.default_rng(0)
    fit = pd.DataFrame({"F": rng.normal(0, 100, 4000), "proxy": rng.normal(1500, 300, 4000), "hour": rng.integers(0, 24, 4000),
                        "x": rng.normal(0, 1, 4000), "ap": ["EHAM"] * 4000})
    A, stats = st._nn_matrix(st._design(fit, ["x"]))
    mu, sd = stats
    cols = list(st._design(fit, ["x"]).drop(columns=["ap", "hour"]).columns) + ["hs", "hc"]   # the numeric block's order
    xi = cols.index("x")
    probe = pd.DataFrame({"F": [0.0] * 5, "proxy": [1500.0] * 5, "hour": [12] * 5,
                          "x": [mu[xi] + 250.0 * sd[xi], mu[xi] - 250.0 * sd[xi], mu[xi] + 3e7 * sd[xi], -3e7, np.nan],
                          "ap": ["EHAM"] * 5})
    T, _ = st._nn_matrix(st._design(probe, ["x"]), stats)
    n_num = len(mu)
    z = T[:, :n_num]
    want_unclipped = ((probe.x.to_numpy(float) - mu[xi]) / sd[xi]).astype(np.float32)
    assert z[0, xi] == want_unclipped[0] and z[1, xi] == want_unclipped[1]               # |z| = 250: bit-identical
    assert abs(want_unclipped[2]) > 1e6 and z[2, xi] == np.float32(300.0) and z[3, xi] == np.float32(-300.0)
    assert z[4, xi] == 0.0 and T[4, n_num + xi] == 1.0 and T[:4, n_num + xi].sum() == 0.0
    assert T[:, 2 * n_num:].sum(axis=1).tolist() == [1.0] * 5                            # the one-hot block untouched
    again, _ = st._nn_matrix(st._design(probe.iloc[:2], ["x"]), stats)
    np.testing.assert_array_equal(again, T[:2])
