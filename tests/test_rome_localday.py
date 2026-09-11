"""Tests for scripts/rome_localday.py (arm LD). Exact small cases, including summer-time boundaries."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rome_localday as R  # noqa: E402

TZ = {"LIRF": "Europe/Rome", "EGLL": "Europe/London"}


def _t(s):
    return pd.Series(pd.to_datetime(s, utc=True))


def test_local_dayoff_uses_the_airports_own_date_in_winter_and_summer():
    """Jan 22:30 UTC sched / 23:30 UTC take-off: same UTC day, next Rome day (UTC+1). Jul 21:30 / 22:30: next Rome day
    (UTC+2). London in winter = UTC: same day. Fails if UTC dates or a single zone are used."""
    mvt = _t(["2025-01-10 23:30", "2025-07-10 22:30", "2025-01-10 23:30", "2025-01-11 00:30"])
    sch = _t(["2025-01-10 22:30", "2025-07-10 21:30", "2025-01-10 22:30", "2025-01-10 22:00"])
    adep = ["LIRF", "LIRF", "EGLL", "LIRF"]
    assert R.local_dayoff(mvt, sch, adep, TZ).tolist() == [1, 1, 0, 1]
    # UTC would say [0, 0, 0, 1]; clip keeps a 2-day gap at 1
    assert R.local_dayoff(_t(["2025-01-12 12:00"]), _t(["2025-01-10 12:00"]), ["LIRF"], TZ).tolist() == [1]


def test_local_dayoff_refusals():
    with pytest.raises(ValueError, match="no time zone"):
        R.local_dayoff(_t(["2025-01-10 12:00"]), _t(["2025-01-10 11:00"]), ["LTFM"], TZ)
    with pytest.raises(ValueError, match="tz-aware UTC"):
        R.local_dayoff(pd.Series(pd.to_datetime(["2025-01-10 12:00"])), _t(["2025-01-10 11:00"]), ["LIRF"], TZ)


def test_localize_changes_only_dayoff():
    f = pd.DataFrame({"MVT_TIME_UTC_mvt": _t(["2025-01-10 23:30"]), "SCHED_TIME_UTC_mvt": _t(["2025-01-10 22:30"]),
                      "ADEP_mvt": ["LIRF"], "dayoff": np.array([0], dtype="int64"), "sp": [3600.0]})
    g = R.localize(f, TZ)
    assert g.dayoff.tolist() == [1] and g.dayoff.dtype == f.dayoff.dtype
    pd.testing.assert_frame_equal(g.drop(columns="dayoff"), f.drop(columns="dayoff"))


def test_e1_floor_band_and_rounding():
    assert R.e1_floor([1000.4, 1000.0, 1000.0, 90_000.0], [30_000.2, 23_999.0, 86_400.0, 30_000.0]).tolist() == \
        [30_000.0, 1000.0, 1000.0, 90_000.0]


def test_price_2026_scales_the_2025_cell_mean_by_the_2026_counts():
    g = np.array([10.0, 30.0, 5.0])
    cell = np.array(["utc0_local1", "utc0_local1", "utc0_local0"])
    c26 = np.array(["utc0_local1"] * 3 + ["utc0_local0"] + ["utc1_local0"])
    p = R.price_2026(g, cell, c26, n_scored=10)
    assert p["by_cell"]["utc0_local1"] == pytest.approx(20.0 * 3 / 10)
    assert p["by_cell"]["utc0_local0"] == pytest.approx(5.0 / 10) and p["by_cell"]["utc1_local0"] == 0.0
    assert p["total"] == pytest.approx(6.5) and p["cells_without_2025_rows"] == ["utc1_local0"]


def test_cell_of():
    assert R.cell_of([0, 1], [1, 1]).tolist() == ["utc0_local1", "utc1_local1"]
