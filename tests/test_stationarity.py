"""The stratum that decides the score is not stationary between 2025 and 2026.

Pre-registered in Amendment 3 to fire RED. It is kept red deliberately: it is the
only 2026-side measurement available without spending a submission, and a green
here would mean the regime had stopped moving, not that the test had stopped
mattering.

The threshold is not a guess. It is the 95% confidence interval of the difference
of two binomial proportions, computed inside the test from the actual row counts,
so the assertion can never be narrower than the noise of the quantity it tests
(PREREG Amendment 3 §5).

Fails when the per-month unmatched rate is replaced by the pooled rate — the exact
mistake the amendment names, since the pooled rate moves 1.545% -> 1.534% while
January nearly doubles.
"""
from __future__ import annotations

import math
import pathlib

import pyarrow.parquet as pq
import pytest

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
COLS = ["PHASE_mvt", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "TAXITIME_SEC_mvt", "AOBT_3_flt"]
pytestmark = pytest.mark.skipif(not (RAW / "ranking.parquet").exists(),
                                reason="challenge data not present")


def _admissible_departures(path):
    d = pq.read_table(path, columns=COLS).to_pandas()
    d = d[d.PHASE_mvt == "DEP"]
    d = d[d.TAXITIME_SEC_mvt.notna() & d.BLOCK_TIME_UTC_mvt.notna()
          & d.MVT_TIME_UTC_mvt.notna()]
    d = d[d.TAXITIME_SEC_mvt > 0]
    return d[~d.MVT_ID_mvt.duplicated(keep="first")]


def _unmatched_rate(frame):
    return float(frame.AOBT_3_flt.isna().mean()), int(len(frame))


def _ci95_of_difference(p1, n1, p2, n2):
    """Half-width of the 95% interval for p2 - p1, two independent binomials."""
    return 1.959964 * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)


def _evaluation_rows():
    ids = set(pq.read_table(RAW / "submitting.parquet",
                            columns=["MVT_ID_mvt"]).to_pandas().MVT_ID_mvt)
    ev = pq.read_table(RAW / "ranking.parquet", columns=COLS).to_pandas()
    return ev[ev.MVT_ID_mvt.isin(ids)]


@pytest.mark.parametrize("month,training_file", [
    (1, "training_2025-01-01_2025-02-01.parquet"),
    (7, "training_2025-07-01_2025-08-01.parquet"),
])
def test_stratum_prevalence_is_stationary(month, training_file):
    """The unmatched-stratum rate must not move between years by more than the
    sampling interval of that move. Asserted PER MONTH, never pooled."""
    train = _admissible_departures(RAW / training_file)
    p25, n25 = _unmatched_rate(train)

    ev = _evaluation_rows()
    ev_month = ev[ev.MVT_TIME_UTC_mvt.dt.month == month]
    p26, n26 = _unmatched_rate(ev_month)

    tol = _ci95_of_difference(p25, n25, p26, n26)
    shift = p26 - p25
    assert abs(shift) <= tol, (
        f"month {month}: unmatched-stratum rate moved {100 * p25:.2f}% (2025, "
        f"n={n25}) -> {100 * p26:.2f}% (2026, n={n26}), a shift of "
        f"{100 * shift:+.2f} percentage points against a 95% interval of "
        f"+/-{100 * tol:.3f}pp ({abs(shift) / tol:.1f}x). The stratum carrying "
        f"most of the squared error is not the same size in the evaluation year, "
        f"so every stratum estimate measured on 2025 is projected, not observed."
    )


def test_pooled_rate_hides_the_shift():
    """Why the test above must be per-month: pooled, the shift is invisible.

    This test PASSES, and that is the point — it documents the trap rather than
    guarding against it.
    """
    jan = _admissible_departures(RAW / "training_2025-01-01_2025-02-01.parquet")
    jul = _admissible_departures(RAW / "training_2025-07-01_2025-08-01.parquet")
    import pandas as pd
    pooled_25, n25 = _unmatched_rate(pd.concat([jan, jul]))
    pooled_26, n26 = _unmatched_rate(_evaluation_rows())
    tol = _ci95_of_difference(pooled_25, n25, pooled_26, n26)
    assert abs(pooled_26 - pooled_25) <= tol, (
        "pooled rates were expected to agree; if this fires the trap has changed "
        f"shape: {100 * pooled_25:.3f}% -> {100 * pooled_26:.3f}%")
