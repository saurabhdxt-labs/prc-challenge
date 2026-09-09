"""The consolidated stratum estimator: contract tests for the hierarchical cells + LIRF model.

`fit_unmatched` is the estimator for the 5,290 scored rows that carry roughly 40% of the
total squared error. It reads BLOCK_TIME to build its TRAINING statistics; it must never
read it for the rows it is predicting. It also shrinks toward parent cells, and a shrinkage
that silently falls through to the global prior would look like a working model while
throwing away every airport-specific rate.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bs", ROOT / "scripts" / "build_submission.py")
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)

EP = pd.Timestamp("2025-03-01", tz="UTC")


def _frame(n=4000, seed=0, airports=("EDDF", "LIRF", "EHAM")):
    rng = np.random.default_rng(seed)
    ap = rng.choice(airports, n)
    # work entirely in epoch seconds and convert once - mixing tz-aware datetimes through
    # np.where silently drops the timezone and derive() then raises on the subtraction
    s0 = (EP - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds()
    sched_s = s0 + rng.integers(0, 86400 * 20, n).astype(float)
    # LIRF copies the schedule into off-block far more often than anywhere else
    is_fill = rng.random(n) < np.where(ap == "LIRF", 0.45, 0.03)
    taxi = rng.gamma(4, 220, n) + 200
    late = rng.gamma(1.5, 1800, n)
    # a fill row stamps SCHED into off-block, so its apparent taxi is late + real taxi
    block_s = np.where(is_fill, sched_s, sched_s + late)
    mvt_s = np.where(is_fill, sched_s + late + taxi, block_s + taxi)
    sched = pd.to_datetime(sched_s, unit="s", utc=True)
    block = pd.to_datetime(block_s, unit="s", utc=True)
    mvt = pd.to_datetime(mvt_s, unit="s", utc=True)
    df = pd.DataFrame({
        "ADEP_mvt": ap, "ADES_mvt": rng.choice(["EGLL", "LFPG", "LEMD"], n),
        "STAND_mvt": rng.choice([f"{c}{i}" for c in "ABC" for i in range(9)], n),
        "RUNWAY_mvt": rng.choice(["09L", "27R"], n),
        "FLIGHT_mvt": rng.choice(["RYR123", "ITY456", "AEZ789", "BAW321"], n),
        "AIRCRAFT_TYPE_mvt": "A320", "FLIGHT_RULE_mvt": "I",
        "MVT_TIME_UTC_mvt": mvt,
        "SCHED_TIME_UTC_mvt": sched,
        "BLOCK_TIME_UTC_mvt": block,
        "AIRCRAFT_OPERATOR_flt": "OP", "MARKET_SEGMENT_flt": "S",
        "WK_TBL_CAT_flt": "M", "FLIGHT_TYPE_flt": "S",
    })
    # the NM clocks are null on every unmatched row, but they must be tz-AWARE nulls: a bare
    # pd.NaT creates a tz-naive column and derive() then raises subtracting it from MVT_TIME.
    for c in ("AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt"):
        df[c] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    df = bs.derive(df)
    df["y"] = (df.MVT_TIME_UTC_mvt - df.BLOCK_TIME_UTC_mvt).dt.total_seconds()
    return df[df.y > 0].reset_index(drop=True)


@pytest.fixture(scope="module")
def split():
    d = _frame()
    return d.iloc[: len(d) // 2].copy(), d.iloc[len(d) // 2:].copy()


def test_predictions_ignore_the_held_out_off_block_clock(split):
    """Permuting the TEST rows' BLOCK_TIME must not move a single prediction.

    BLOCK is the target and is null on every scored row. Fails if any statistic is computed
    over `test` rather than `train`. Rehearsed 2026-09-08 by computing the fill rate over
    pd.concat([tr, te]): predictions moved on 1,676 rows and this test went red.
    """
    tr, te = split
    a = bs.fit_unmatched(tr, te, train_matched=tr.iloc[:0])
    te2 = te.copy()
    rng = np.random.default_rng(1)
    te2["BLOCK_TIME_UTC_mvt"] = rng.permutation(te2.BLOCK_TIME_UTC_mvt.to_numpy())
    b = bs.fit_unmatched(tr, te2, train_matched=tr.iloc[:0])
    assert np.allclose(a, b), f"{int((a != b).sum())} predictions moved with the test labels"

    # control: perturbing TRAINING labels MUST move them, else the test is vacuous
    tr2 = tr.copy()
    tr2["BLOCK_TIME_UTC_mvt"] = tr2.BLOCK_TIME_UTC_mvt - pd.Timedelta(seconds=900)
    assert not np.allclose(a, bs.fit_unmatched(tr2, te, train_matched=tr.iloc[:0])), \
        "training-label perturbation changed nothing - the test proves nothing"


def test_hierarchy_shrinks_toward_the_parent_not_the_global_prior(split):
    """An airport whose fine cells are empty must still get its OWN airport-level rate.

    Fails when the levels overwrite instead of shrink (the flat estimator's behaviour):
    LIRF's 45% fill rate would collapse toward the ~17% global prior.
    """
    tr, te = split
    pred = bs.fit_unmatched(tr, te, train_matched=tr.iloc[:0])
    # the estimator is pred = p*sp + (1-p)*nf, so p is recoverable wherever sp is far from
    # the non-fill mean. That is the quantity the hierarchy must carry per airport.
    nf0 = float(tr.y[(tr.BLOCK_TIME_UTC_mvt - tr.SCHED_TIME_UTC_mvt)
                     .dt.total_seconds().abs() > 60].mean())
    sp = te.sp.to_numpy()
    usable = np.abs(sp - nf0) > 1800
    implied_p = (pred[usable] - nf0) / (sp[usable] - nf0)
    lirf = (te.ADEP_mvt == "LIRF").to_numpy()[usable]
    p_lirf, p_other = implied_p[lirf].mean(), implied_p[~lirf].mean()
    # fixture truth: LIRF fills 45% of the time, everywhere else 3%
    assert p_lirf > 0.25, f"LIRF fill rate collapsed to {p_lirf:.3f} (truth 0.45)"
    assert p_lirf > 3 * p_other, (
        f"airport-specific fill rate not preserved: LIRF {p_lirf:.3f} vs other {p_other:.3f}")


def test_lirf_model_touches_only_lirf_rows(split):
    """The logistic is LIRF-only; no other airport's prediction may change because of it."""
    tr, te = split
    base = bs.fit_unmatched(tr, te, train_matched=tr.iloc[:0])
    matched = _frame(n=3000, seed=7)              # stands in for LIRF's matched rows
    with_model = bs.fit_unmatched(tr, te, train_matched=matched)
    other = (te.ADEP_mvt != "LIRF").to_numpy()
    assert np.allclose(base[other], with_model[other]), \
        "the LIRF model altered predictions at other airports"


def test_predictions_are_positive_and_finite(split):
    """Taxi-out is strictly positive; the submission contract rejects anything else."""
    tr, te = split
    pred = bs.fit_unmatched(tr, te, train_matched=tr.iloc[:0])
    assert np.isfinite(pred).all() and (pred >= 1.0).all()
    assert len(pred) == len(te)
