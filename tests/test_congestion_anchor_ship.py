"""Tests for SHIP-CA (scripts/analysis/congestion_anchor_ship.py; plans/PREREG_ship_congestion_anchor_2026_09_13.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import congestion_anchor_ship as C  # noqa: E402


def _rows():
    t = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    recs = []
    # airport X, Jan: days 1-3 normal proxy 600 s, day 4 storm proxy 1,500 s; one unmatched row; one out-of-range proxy; one ARR row
    def add(day, prox, phase="DEP", k=0):
        mvt = t(f"2026-01-0{day}T10:{k:02d}:00")
        recs.append(dict(ADEP_mvt="X", PHASE_mvt=phase, MVT_TIME_UTC_mvt=mvt, AOBT_3_flt=mvt - pd.Timedelta(seconds=prox)))
    for day in (1, 2, 3):
        for k in range(3):
            add(day, 600, k=k)
    for k, prox in enumerate((1400, 1500, 2600)):          # median 1,500, mean 1,833 (fails if median -> mean)
        add(4, prox, k=k)
    for k in range(2):
        add(4, 86_400, k=10 + k)                            # out of range: with the cap dropped the day-4 median becomes 2,600
    for k in range(4):
        add(1, 1_000, phase="ARR", k=20 + k)                # arrivals: if included, day-1 median becomes 1,000
    recs.append(dict(ADEP_mvt="X", PHASE_mvt="DEP", MVT_TIME_UTC_mvt=t("2026-01-04T11:00:00"), AOBT_3_flt=pd.NaT))
    return pd.DataFrame(recs)


def test_day_anomaly_uses_matched_departures_in_range_relative_to_month_median():
    """Fails if unmatched/ARR rows or proxies >= 10,800 s enter, the median is replaced by a mean, or the month baseline is dropped."""
    a = C.day_anomaly(_rows()).set_index("date").A
    assert a.to_dict() == {"2026-01-01": 0.0, "2026-01-02": 0.0, "2026-01-03": 0.0, "2026-01-04": 900.0}


def test_shift_knot_and_missing():
    """max(A − 300, 0); NaN → 0. Fails if the knot moves or the ratio differs from 1."""
    assert C.shift([900.0, 300.0, 250.0, np.nan, -50.0]).tolist() == [600.0, 0.0, 0.0, 0.0, 0.0]


def test_anchored_values_round_and_floor_guard():
    assert C.anchored_values([1000.0, 1000.0], [900.4, np.nan]).tolist() == [1600.0, 1000.0]
    with pytest.raises(AssertionError, match="G3"):
        C.anchored_values([np.nan], [0.0])
