"""The stratum that decides the score is not stationary between 2025 and 2026.

Pre-registered in Amendment 3 to fire RED. RESTRUCTURED 2026-09-12 (session
prc-challenge-9962-b) under the owner's ruling `reports/RXC_REVIEW_RULING_2026_09_12.md`:

    "retain the measured drift as a scientific diagnostic. A software test can verify that
     the diagnostic detects it; a claim that deployment data must be stationary is not a
     general software invariant. Any method that requires stationarity must still fail its
     transfer gate. Do not widen thresholds to make the measured drift disappear."

So the DRIFT IS PRESERVED, not suppressed, and NO threshold was widened. What changed is what
the software asserts: it now verifies the DETECTOR works -- that it flags the real 2025->2026
move, and that it stays silent on a stationary control. The drift itself is recorded as a
measured transfer limitation in `test_the_measured_drift_is_recorded`, which is the number any
method requiring stationarity must answer to.

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


#: The measured 2025 -> 2026 move, recorded 2026-09-12 so it is a FACT in the repo rather than a
#: red test nobody reads. Shift is in percentage points; `x_ci` is |shift| / the 95% interval.
MEASURED_DRIFT = {1: {"p25": 0.0092, "p26": 0.0161, "shift_pp": +0.69, "x_ci": 8.7},
                  7: {"p25": 0.0205, "p26": 0.0148, "shift_pp": -0.58, "x_ci": 6.9}}


@pytest.mark.parametrize("month,training_file", [
    (1, "training_2025-01-01_2025-02-01.parquet"),
    (7, "training_2025-07-01_2025-08-01.parquet"),
])
def test_the_drift_detector_flags_the_real_2025_to_2026_move(month, training_file):
    """The DETECTOR must fire on the real data. Asserted PER MONTH, never pooled.

    This is the inversion the ruling asked for: the old assertion was `abs(shift) <= tol`, which
    demanded that deployment data be stationary -- not a software invariant, and permanently red.
    The invariant that IS software's business is that the instrument detects the move. The
    threshold `tol` is unchanged: it is still the 95% interval of the difference of two binomial
    proportions computed from the actual row counts.
    """
    train = _admissible_departures(RAW / training_file)
    p25, n25 = _unmatched_rate(train)
    ev_month = _evaluation_rows()
    ev_month = ev_month[ev_month.MVT_TIME_UTC_mvt.dt.month == month]
    p26, n26 = _unmatched_rate(ev_month)
    tol = _ci95_of_difference(p25, n25, p26, n26)
    shift = p26 - p25
    assert abs(shift) > tol, (
        f"month {month}: the unmatched-stratum rate is no longer detectably different between "
        f"2025 ({100 * p25:.2f}%, n={n25}) and 2026 ({100 * p26:.2f}%, n={n26}): shift "
        f"{100 * shift:+.2f}pp against a 95% interval of +/-{100 * tol:.3f}pp. Either the regime "
        f"stopped moving -- which would be news -- or the detector broke. Investigate before "
        f"treating any 2025 stratum estimate as observed rather than projected.")


def test_the_drift_detector_stays_silent_on_a_stationary_control():
    """A detector that always fires is not a detector.

    Splits ONE month's admissible departures in half at random and runs the same statistic. Two
    halves of one month are stationary by construction, so the detector must NOT flag them. Fails
    if `tol` is ever shrunk toward zero to manufacture a detection.
    """
    import numpy as np
    d = _admissible_departures(RAW / "training_2025-01-01_2025-02-01.parquet")
    rng = np.random.default_rng(0)
    half = rng.random(len(d)) < 0.5
    pa, na = _unmatched_rate(d[half])
    pb, nb = _unmatched_rate(d[~half])
    tol = _ci95_of_difference(pa, na, pb, nb)
    assert abs(pb - pa) <= tol, (
        f"two random halves of one month differ by {100 * (pb - pa):+.3f}pp against a 95% "
        f"interval of +/-{100 * tol:.3f}pp -- the detector fires on stationary data, so its "
        f"detections on real data mean nothing.")


@pytest.mark.parametrize("month", [1, 7])
def test_the_measured_drift_is_recorded(month):
    """The drift as a MEASURED TRANSFER LIMITATION, pinned so it cannot silently change.

    This is the number any method that assumes stationarity must answer to. It is not a target and
    it is not something to fix: January's unmatched rate nearly doubles (0.92% -> 1.61%) while July
    falls (2.05% -> 1.48%), which is why the pooled rate hides it. Fails if the real data moves
    materially from what was recorded, which is itself worth knowing.
    """
    f = {1: "training_2025-01-01_2025-02-01.parquet", 7: "training_2025-07-01_2025-08-01.parquet"}[month]
    p25, _ = _unmatched_rate(_admissible_departures(RAW / f))
    ev = _evaluation_rows()
    p26, _ = _unmatched_rate(ev[ev.MVT_TIME_UTC_mvt.dt.month == month])
    want = MEASURED_DRIFT[month]
    assert abs(p25 - want["p25"]) < 5e-4, f"2025 rate moved: {p25:.4f} vs recorded {want['p25']}"
    assert abs(p26 - want["p26"]) < 5e-4, f"2026 rate moved: {p26:.4f} vs recorded {want['p26']}"
    assert abs(100 * (p26 - p25) - want["shift_pp"]) < 0.05


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
