"""Every feature a fitted model consults must be computable on the rows we must predict.

This is the train-serve skew guard. It fired RED from its writing until 2026-09-12 against
`prc.baselines`, whose deepest rung keyed on `_hour` derived from BLOCK_TIME_UTC_mvt — a column
that is 100% null on the evaluation file. The rung therefore never fired in production and the
published 627.7 denominator was not deliverable (PREREG Amendment 3 section 3).

FIXED 2026-09-12: the ladder is re-anchored to take-off hour (`baselines.HOUR_SOURCE`, measured
0.000% null on all 344,841 scored rows, 24 distinct hours). This test is what proves the fix
reached the code rather than the docs, and it stays live so a re-anchor back to any hidden column
goes red again. It resolves `_hour` through `baselines.HOUR_SOURCE` rather than naming a column
itself - duplicating the mapping here is how the original defect survived review.
"""
from __future__ import annotations

import glob
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from prc import baselines

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
pytestmark = pytest.mark.skipif(not (RAW / "ranking.parquet").exists(),
                                reason="challenge data not present")


def _scored_rows(columns):
    ids = set(pq.read_table(RAW / "submitting.parquet",
                            columns=["MVT_ID_mvt"]).to_pandas().MVT_ID_mvt)
    need = sorted(set(columns) | {"MVT_ID_mvt"})
    ev = pq.read_table(RAW / "ranking.parquet", columns=need).to_pandas()
    return ev[ev.MVT_ID_mvt.isin(ids)]


def test_every_ladder_key_is_populated_on_the_scored_rows():
    """Each column any rung keys on must be non-null on the evaluation file.

    `_hour` is derived, so the test resolves it back to its source column rather
    than accepting the derived name at face value — accepting the derived name is
    exactly how the defect survived review.
    """
    source_of = {"_hour": baselines.HOUR_SOURCE}   # never hard-code the source here
    keys = {source_of.get(k, k) for rung in baselines.RUNGS for k in rung.keys}
    ev = _scored_rows(keys)

    dead = {k: float(ev[k].isna().mean()) for k in sorted(keys)
            if ev[k].isna().mean() > 0.01}
    assert not dead, (
        "these ladder keys are unusable on the 344,841 scored rows: "
        + ", ".join(f"{k} is {100 * v:.3f}% null" for k, v in dead.items())
        + ". A rung keyed on a null column silently returns the rung above it, so "
          "the reported denominator is not the one the submission would achieve."
    )


def _training(files):
    cols = {baselines.AIRPORT, baselines.RUNWAY, baselines.STAND,
            baselines.HOUR_SOURCE, baselines.LABEL}
    train = pd.concat([pq.read_table(f, columns=sorted(cols | {"PHASE_mvt"})).to_pandas()
                       for f in files], ignore_index=True)
    train = train[(train.PHASE_mvt == "DEP") & train[baselines.LABEL].notna()
                  & train[baselines.HOUR_SOURCE].notna()]
    return train[train[baselines.LABEL] > 0]


def test_ladder_predictions_differ_when_the_finest_rung_is_dropped():
    """If the deepest rung is alive, dropping it must change predictions.

    Identical output from two different depths is the signature of a dead rung. Uses real evaluation
    rows, because on training rows the rung works fine and the defect is invisible.

    Fits on ALL training months, which is what a benchmark actually has. Measured 2026-09-12: on one
    month, 0 of 52,460 deepest-rung cells reach min_count=30, so the rung is inert from SPARSITY even
    after the anchor was repaired -- two stacked defects. On twelve months, 22,152 of 129,118 cells
    qualify and 12,632 of 20,000 scored rows change. `min_count` was NOT lowered to reach this;
    lowering it would have made the test pass while leaving the rung untrustworthy.
    """
    files = sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
    assert len(files) == 12, f"this benchmark fits on all 12 training months, found {len(files)}"
    train = _training(files)
    deep = baselines.Ladder.fit(train, max_depth=len(baselines.RUNGS) - 1)
    shallow = baselines.Ladder.fit(train, max_depth=len(baselines.RUNGS) - 2)

    cols = {baselines.AIRPORT, baselines.RUNWAY, baselines.STAND, baselines.HOUR_SOURCE}
    ev = _scored_rows(cols).head(20_000)
    p_deep, p_shallow = deep.predict(ev), shallow.predict(ev)

    assert not np.allclose(p_deep, p_shallow), (
        f"the deepest rung '{baselines.RUNGS[-1].name}' changes nothing on the "
        f"scored rows: {len(p_deep)} predictions are identical to the rung above "
        f"it. It keys on {baselines.RUNGS[-1].keys}, and the source column is "
        f"{100 * ev[baselines.HOUR_SOURCE].isna().mean():.1f}% null here.")


def test_the_deepest_rung_is_inert_on_a_single_month_and_that_is_sparsity_not_skew():
    """The second defect, kept as a fact rather than lost in the fix.

    After the anchor repair the deepest rung is STILL inert when fitted on one month: no
    (airport, runway, stand, hour) cell reaches min_count. This is why the original test could not
    distinguish "the column is null" from "the cells are thin" -- both produced identical predictions.
    Fails if someone lowers min_count to force the rung alive on thin data.
    """
    one = _training(sorted(glob.glob(str(RAW / "training_2025-*.parquet")))[:1])
    with_hour = baselines._with_hour(one)
    counts = with_hour.groupby([baselines.AIRPORT, baselines.RUNWAY, baselines.STAND, "_hour"],
                               dropna=False)[baselines.LABEL].size()
    assert (counts >= baselines.DEFAULT_MIN_COUNT).sum() == 0, (
        "a one-month fit now has populated deepest-rung cells; min_count may have been lowered")
    deep = baselines.Ladder.fit(one, max_depth=len(baselines.RUNGS) - 1)
    shallow = baselines.Ladder.fit(one, max_depth=len(baselines.RUNGS) - 2)
    ev = _scored_rows({baselines.AIRPORT, baselines.RUNWAY, baselines.STAND,
                       baselines.HOUR_SOURCE}).head(20_000)
    assert np.allclose(deep.predict(ev), shallow.predict(ev))
