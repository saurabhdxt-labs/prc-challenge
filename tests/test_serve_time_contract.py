"""Every feature a fitted model consults must be computable on the rows we must predict.

This is the train-serve skew guard. It fires RED today against `prc.baselines`, whose
deepest rung keys on `_hour`, derived from BLOCK_TIME_UTC_mvt — a column that is
100% null on the evaluation file. The rung therefore never fires in production and
the published 627.7 denominator is not deliverable (PREREG Amendment 3 section 3).

Fails when the ladder is re-anchored to take-off hour: that is the intended fix, and
this test is what proves the fix reached the code rather than the docs.
"""
from __future__ import annotations

import pathlib

import numpy as np
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
    source_of = {"_hour": baselines.OFF_BLOCK}
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


def test_ladder_predictions_differ_when_the_finest_rung_is_dropped():
    """If the deepest rung is alive, dropping it must change predictions.

    Identical output from two different depths is the signature of a dead rung.
    Uses real evaluation rows, because on training rows the rung works fine and
    the defect is invisible — which is the whole point.
    """
    cols = {baselines.AIRPORT, baselines.RUNWAY, baselines.STAND,
            baselines.OFF_BLOCK, baselines.LABEL}
    train = pq.read_table(RAW / "training_2025-01-01_2025-02-01.parquet",
                          columns=sorted(cols | {"PHASE_mvt"})).to_pandas()
    train = train[(train.PHASE_mvt == "DEP") & train[baselines.LABEL].notna()
                  & train[baselines.OFF_BLOCK].notna()]
    train = train[train[baselines.LABEL] > 0]

    deep = baselines.Ladder.fit(train, max_depth=len(baselines.RUNGS) - 1)
    shallow = baselines.Ladder.fit(train, max_depth=len(baselines.RUNGS) - 2)

    ev = _scored_rows(cols).head(20_000)
    p_deep, p_shallow = deep.predict(ev), shallow.predict(ev)

    assert not np.allclose(p_deep, p_shallow), (
        f"the deepest rung '{baselines.RUNGS[-1].name}' changes nothing on the "
        f"scored rows: {len(p_deep)} predictions are identical to the rung above "
        f"it. It keys on {baselines.RUNGS[-1].keys}, and the source column is "
        f"{100 * ev[baselines.OFF_BLOCK].isna().mean():.1f}% null here."
    )
