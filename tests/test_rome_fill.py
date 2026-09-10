"""scripts/rome_fill.py - arm R of plans/PREREG_rome_fill_2026_09_10.md: a stake-weighted LightGBM
fill classifier for Rome's unmatched rows, swapped into S1's mixture in place of p_hat only.

The tests pin: the registered weight formula; the serve-time contract (no feature reads BLOCK or
the target); unseen categorical levels at test time; that the classifier recovers a planted
airline-level fill signal and the permuted-label control does not; the month-block bootstrap;
that every fold trains without its own test month; and that the harness rebuilds S1 exactly
before it reports anything.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rome_fill as rf  # noqa: E402
import stratum_fold as sf  # noqa: E402


@pytest.fixture(scope="module")
def frames():
    """derive()-shaped synthetic frames with the stratum harness's planted LIRF fill structure:
    fill probability ITY 0.7, RYR 0.2, others 0.45 at LIRF. Read-only across tests.

    Sized so LIRF carries ~150 unmatched rows a month, as the real stratum does. The first version
    used 400 rows/month over ten airports (~40 LIRF rows/month): under the stake weights that is a
    Kish effective sample of 398 of 5,428 training rows, and the permuted control's airline gap
    swung -0.30 / +0.00 / -0.07 / -0.10 across four seeds - a statistic too small to test."""
    unm, lirf = sf.synthetic_frame(n_per_month=1500, n_matched_per_month=2000, seed=3)
    return unm, lirf


def test_stake_weight_is_the_registered_formula():
    """w = clip(1 + (max(sp,0)/3600)^2, 1, 25) x (8 if unmatched else 1). Rehearsed 2026-09-10
    RED: the cap removed; the unmatched factor applied to matched rows."""
    sp = np.array([-500.0, 0.0, 1800.0, 3600.0, 7200.0, 36_000.0])
    un = np.array([False, False, False, True, False, True])
    w = rf.stake_weight(sp, un)
    np.testing.assert_allclose(w, [1.0, 1.0, 1.25, 16.0, 5.0, 200.0], rtol=0, atol=1e-12)


def test_features_never_read_block_or_the_target(frames):
    """Serve-time contract: scrambling BLOCK, TAXITIME and y changes no feature value.
    Rehearsed 2026-09-10 RED: `(BLOCK - SCHED)` added as a feature column."""
    unm, _ = frames
    a = rf.feature_frame(unm)
    scr = unm.copy()
    scr["BLOCK_TIME_UTC_mvt"] = pd.NaT
    scr["TAXITIME_SEC_mvt"] = np.nan
    scr["y"] = -1.0
    b = rf.feature_frame(scr)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == rf.FEATS_NUM + rf.FEATS_CAT
    assert not {"y", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "delta", "proxy"} & set(a.columns)


def test_fill_label_is_the_shipped_definition(frames):
    unm, _ = frames
    d = (unm.BLOCK_TIME_UTC_mvt - unm.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()
    np.testing.assert_array_equal(rf.fill_label(unm), (d <= 60).to_numpy())


def test_classifier_recovers_the_planted_airline_signal_and_the_permuted_control_does_not(frames):
    """Trained on months 1-10 of LIRF rows (matched + unmatched), scored on ALL of months 11-12's
    LIRF rows: mean p on ITY must exceed mean p on RYR by a wide margin (planted 0.7 vs 0.2); with
    labels permuted within month, the gap averaged over four seeds collapses. A single permuted fit
    is not tested: its airline effect is fitted noise whose size depends on the weighted effective
    sample, so the control is the seed-average. Rehearsed 2026-09-10 RED: the categorical columns
    dropped from the design (the airline signal is invisible)."""
    unm, lirf = frames
    tr = rf.rome_rows(unm, lirf, months=range(1, 11))
    te = rf.rome_rows(unm, lirf, months=[11, 12])
    ity, ryr = (te.airline == "ITY").to_numpy(), (te.airline == "RYR").to_numpy()
    p = rf.fit_predict_fill(tr, te, seed=0)
    assert p.shape == (len(te),) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    gap = p[ity].mean() - p[ryr].mean()
    perm = [rf.fit_predict_fill(tr, te, seed=s, permute=True) for s in range(4)]
    gaps_perm = [pp[ity].mean() - pp[ryr].mean() for pp in perm]
    print(f"\nplanted gap recovered {gap:.3f}; permuted gaps {np.round(gaps_perm, 3)}")
    assert gap > 0.25
    assert abs(np.mean(gaps_perm)) < 0.10


def test_unseen_categorical_levels_at_test_time_do_not_break_prediction(frames):
    unm, lirf = frames
    tr = rf.rome_rows(unm, lirf, months=range(1, 11))
    te = unm[(unm.ADEP_mvt == "LIRF") & (unm.month == 12)].copy()
    te["airline"] = "ZZZ"
    te["STAND_mvt"] = "Q99"
    p = rf.fit_predict_fill(tr, te, seed=0)
    assert np.isfinite(p).all()


def test_rome_rows_takes_only_lirf_and_only_the_training_months(frames):
    """Rehearsed 2026-09-10 RED: the month filter dropped (the test month leaks into training)."""
    unm, lirf = frames
    tr = rf.rome_rows(unm, lirf, months=[2, 3])
    assert set(tr.ADEP_mvt) == {"LIRF"} and set(tr.month) == {2, 3}
    assert tr.unmatched.any() and (~tr.unmatched).any()
    n_expect = int(((unm.ADEP_mvt == "LIRF") & unm.month.isin([2, 3])).sum() + lirf.month.isin([2, 3]).sum())
    assert len(tr) == n_expect


def test_month_block_bootstrap_is_zero_for_identical_arms_and_positive_for_a_consistent_gain():
    """Rehearsed 2026-09-10 RED: the gain's sign flipped (new - ref)."""
    base = np.array([10.0, 12.0, 8.0, 30.0, 5.0, 7.0, 9.0, 11.0, 6.0, 14.0, 13.0, 4.0])
    g = rf.month_block_bootstrap(base, base, n_draws=500, seed=0)
    assert g["gain"] == 0.0 and g["ci95"] == [0.0, 0.0]
    g2 = rf.month_block_bootstrap(base, 0.8 * base, n_draws=500, seed=0)
    assert g2["gain"] == pytest.approx(0.2 * base.sum()) and g2["ci95"][0] > 0
    g3 = rf.month_block_bootstrap(0.8 * base, base, n_draws=500, seed=0)
    assert g3["ci95"][1] < 0


def test_harness_rebuilds_s1_exactly_and_reports_every_registered_clause(frames):
    """End to end on the synthetic frames with a synthetic stored S1 record: the rebuilt S1
    equals the stored column on every row (else the run refuses), every arm is present, R beats
    a deliberately weak stored p_hat, R_perm does not, and all five clauses are reported.
    Rehearsed 2026-09-10 RED: the mixture floor removed (S1 no longer rebuilds; refuses)."""
    unm, lirf = frames
    stored = rf.synthetic_stored(unm, seed=1)
    res = rf.run(unm, lirf, stored, seeds=(0, 1, 2), n_boot=300)
    assert res["controls"]["s1_rebuilds_exactly"] is True
    assert set(res["arms"]) >= {"S1", "R", "R_seed0", "R_seed1", "R_seed2", "R_unm", "R_perm"}
    lomo = res["lomo"]
    assert lomo["sse"]["R"] < lomo["sse"]["S1"]
    assert res["controls"]["perm_does_not_beat_s1"] is True
    assert set(res["clauses"]) == {"c1_month_block_interval", "c2_lomo_reduction_ge_15pct",
                                   "c3_foldA_reduction_ge_15pct", "c4_gain_over_2x_seed_sd",
                                   "c5_months_improving_ge_8"}
    assert res["verdict"] in {"ESTABLISHED", "INCONCLUSIVE", "NOT WORKING"}


def test_mixture_is_the_shipped_function_and_floors_at_one_second():
    """The harness must score arms with the shipped mixture, floor included. Rehearsed 2026-09-10:
    a local copy without the floor SURVIVED the end-to-end test (the fixture shared the copy);
    now the function is imported and this test binds the floor on a constructed row."""
    assert rf.mixture is sf.bs.mixture
    out = rf.mixture(np.array([0.5, 0.0]), np.array([-5000.0, 100.0]), np.array([10.0, 300.0]))
    np.testing.assert_allclose(out, [1.0, 300.0], rtol=0, atol=0)


def test_harness_refuses_when_s1_does_not_rebuild(frames):
    unm, lirf = frames
    stored = rf.synthetic_stored(unm, seed=1)
    stored.loc[stored.index[0], "S1"] += 50.0
    with pytest.raises(AssertionError, match="S1 does not rebuild"):
        rf.run(unm, lirf, stored, seeds=(0,), n_boot=50)


def test_cached_frames_carry_every_column_derive_produces(frames, tmp_path, monkeypatch):
    """Round trip through load_frames' cache: every derive() column comes back, and sp_band /
    sp_fine equal derive()'s own. Rehearsed 2026-09-10 RED: with_sp_bins removed from the cached
    read path (the ship step died on KeyError('sp_band') before this existed)."""
    unm, lirf = frames
    monkeypatch.setattr(rf, "FRAME_CACHE", tmp_path)
    for d, name in ((unm, "unm"), (lirf, "lirf")):
        d.drop(columns=["sp_band", "sp_fine"]).to_parquet(tmp_path / f"{name}.parquet")
    u2, l2 = rf.load_frames(lambda m: None)
    for orig, back in ((unm, u2), (lirf, l2)):
        assert set(orig.columns) <= set(back.columns)
        for c in ("sp_band", "sp_fine"):
            np.testing.assert_array_equal(orig[c].astype(str).to_numpy(), back[c].astype(str).to_numpy())
