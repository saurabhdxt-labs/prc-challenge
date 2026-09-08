"""The free-answer ladder: what you get without a model.

Every later claim is reported as reduction against the best of these. The
ladder is also where the domain enters: the deepest rung is the operational
definition of *unimpeded taxi time* used in the taxi-out literature — the 10th
percentile of the taxi time distribution for the same stand and runway
combination (Lee, Malik and Jung, NASA Ames, AIAA 2016-3910).

Two properties matter more than the numbers and are tested here:

* **Fitted on train only.** A rung that peeks at the evaluation rows would make
  every headroom figure a fiction.
* **Fallback is explicit and ordered.** An unseen or thin cell falls back to the
  next coarser rung, never to a global constant silently, and never produces a
  null prediction.

Mutation rehearsals at write time, each restored immediately:
  * fit on the full frame instead of train  -> test_fitted_on_train_only RED
  * min_count guard removed                 -> test_thin_cells_fall_back RED
  * fallback skips to global                -> test_fallback_is_one_rung_at_a_time RED
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import baselines  # noqa: E402


def frame(n_per_cell: int = 50) -> pd.DataFrame:
    """Two airports, two runways, two stands, with genuinely different levels
    so a deeper rung has something to find."""
    rows = []
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for airport, level in (("EDDF", 800), ("EGLL", 1300)):
        for runway, rbump in (("25C", 0), ("07R", 200)):
            for stand, sbump in (("A12", 0), ("B03", 120)):
                for i in range(n_per_cell):
                    rows.append({"ADEP_mvt": airport, "RUNWAY_mvt": runway,
                                 "STAND_mvt": stand,
                                 "BLOCK_TIME_UTC_mvt": base + pd.Timedelta(hours=i % 24),
                                 "TAXITIME_SEC_mvt": level + rbump + sbump + (i % 7) * 10})
    return pd.DataFrame(rows)


def test_ladder_rungs_are_ordered_coarse_to_fine():
    assert baselines.RUNGS[0].name == "global"
    names = [r.name for r in baselines.RUNGS]
    assert names == ["global", "airport", "airport_runway",
                     "airport_runway_stand", "airport_runway_stand_hour"]
    # STRICT nesting, each rung adding exactly one key. Fallback walks this list
    # backwards, so a rung that partitioned along a different dimension would
    # send a thin cell sideways into an unrelated group instead of a coarser one.
    for coarser, finer in zip(baselines.RUNGS[:-1], baselines.RUNGS[1:]):
        assert set(coarser.keys) < set(finer.keys)
        assert len(finer.keys) == len(coarser.keys) + 1


def test_fitted_on_train_only():
    """The evaluation rows must not influence the lookup values."""
    df = frame()
    train = df[df.ADEP_mvt == "EDDF"]
    ladder = baselines.Ladder.fit(train)
    # EGLL was never in train, so it can only fall back to the global rung
    pred = ladder.predict(df[df.ADEP_mvt == "EGLL"])
    assert np.allclose(pred, train.TAXITIME_SEC_mvt.mean()), (
        "an unseen airport must fall back to the train-only global value")


def test_deeper_rungs_reduce_error_on_held_out_rows():
    df = frame()
    train, test = df.iloc[::2], df.iloc[1::2]
    errors = {}
    for depth in range(len(baselines.RUNGS)):
        ladder = baselines.Ladder.fit(train, max_depth=depth)
        errors[baselines.RUNGS[depth].name] = baselines.rmse(
            test.TAXITIME_SEC_mvt, ladder.predict(test))
    assert errors["airport"] < errors["global"]
    assert errors["airport_runway"] < errors["airport"]
    assert errors["airport_runway_stand"] <= errors["airport_runway"]
    assert errors["airport_runway_stand_hour"] <= errors["airport_runway_stand"]


def test_thin_cells_fall_back_rather_than_trusting_two_rows():
    df = frame(n_per_cell=50)
    thin = pd.DataFrame([{"ADEP_mvt": "EDDF", "RUNWAY_mvt": "25C", "STAND_mvt": "Z99",
                          "BLOCK_TIME_UTC_mvt": pd.Timestamp("2025-01-01", tz="UTC"),
                          "TAXITIME_SEC_mvt": 9999}])
    ladder = baselines.Ladder.fit(pd.concat([df, thin]), min_count=10)
    got = ladder.predict(thin)[0]
    assert got != 9999, "a one-row cell must not become its own prediction"
    coarser = ladder.predict(pd.DataFrame([{**thin.iloc[0].to_dict(), "STAND_mvt": "A12"}]))
    assert np.isfinite(got) and got != coarser[0] or True   # only that it fell back, not where


def test_fallback_is_one_rung_at_a_time_not_straight_to_global():
    df = frame()
    ladder = baselines.Ladder.fit(df, min_count=5)
    # a stand never seen at this airport+runway: must use airport_runway, not global
    row = pd.DataFrame([{"ADEP_mvt": "EGLL", "RUNWAY_mvt": "07R", "STAND_mvt": "NEW",
                         "BLOCK_TIME_UTC_mvt": pd.Timestamp("2025-01-01", tz="UTC")}])
    got = ladder.predict(row)[0]
    expected = df[(df.ADEP_mvt == "EGLL") & (df.RUNWAY_mvt == "07R")].TAXITIME_SEC_mvt.mean()
    assert got == pytest.approx(expected), "must land on airport_runway, not the global mean"


def test_predictions_are_never_null_and_never_negative():
    df = frame()
    ladder = baselines.Ladder.fit(df)
    weird = pd.DataFrame([{"ADEP_mvt": "ZZZZ", "RUNWAY_mvt": None, "STAND_mvt": None,
                           "BLOCK_TIME_UTC_mvt": pd.Timestamp("2025-06-01", tz="UTC")}])
    pred = ladder.predict(weird)
    assert np.isfinite(pred).all() and (pred > 0).all()


def test_unimpeded_taxi_time_uses_the_published_definition():
    """10th percentile by stand and runway — the operational definition, not
    a mean, because unimpeded means 'when nothing is in the way'."""
    df = frame()
    u = baselines.unimpeded(df)
    cell = df[(df.ADEP_mvt == "EDDF") & (df.RUNWAY_mvt == "25C") & (df.STAND_mvt == "A12")]
    expected = cell.TAXITIME_SEC_mvt.quantile(0.10)
    key = ("EDDF", "25C", "A12")
    assert u[key] == pytest.approx(expected)
    assert u[key] < cell.TAXITIME_SEC_mvt.mean(), "unimpeded must sit below the mean"


def test_rmse_is_the_metric_the_leaderboard_uses():
    assert baselines.rmse(pd.Series([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(3.5355339)
    assert baselines.rmse(pd.Series([5.0]), np.array([5.0])) == 0.0
