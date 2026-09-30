"""Tests for scripts/rome_local_day.py — rule RLD (plans/PREREG_rome_local_day_rule_2026_09_11.md).

Written before the module (red first). Each test names the mutation it must kill.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("rome_local_day", ROOT / "scripts" / "rome_local_day.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rome_local_day"] = mod
    spec.loader.exec_module(mod)
    return mod


rl = _load()


def _row(mid, sched, mvt, base, ap="LIRF", unmatched=True):
    return dict(MVT_ID_mvt=float(mid), ADEP_mvt=ap, unmatched=unmatched, base=base,
                SCHED_TIME_UTC_mvt=pd.Timestamp(sched, tz="UTC"), MVT_TIME_UTC_mvt=pd.Timestamp(mvt, tz="UTC"))


def test_the_rule_keys_on_the_rome_local_day_not_the_utc_day():
    """Kills: dayoff computed in UTC (the review's R-2 finding). Row 1: UTC day 0, Rome day 1 -> untouched (a date-slip
    is possible). Row 2: UTC day 1, Rome day 0 -> the rule applies."""
    fr = pd.DataFrame([
        _row(1, "2026-07-14 00:30", "2026-07-14 23:00", 90000),   # CEST: 02:30 -> 01:00 next local day; sp 81,000
        _row(2, "2026-01-14 23:30", "2026-01-15 16:00", 90000),   # CET: 00:30 Jan 15 -> 17:00 Jan 15; sp 59,400
    ])
    out, changed = rl.apply_rld(fr)
    assert out.tolist() == [90000, 59400] and changed.tolist() == [False, True]


def test_band_edges_lane_and_airport():
    """Kills: >= 24,000 flipped to >; matched rows or other airports touched. The sp < 86,400 bound is an EQUIVALENT
    mutant in scope (rehearsed 2026-09-11, L3 survived by design): sp >= 24 h already implies a later local day in
    January / July, so no local-dayoff-0 row can reach it. The 25-hour autumn DST day is outside the scored months."""
    base = 100
    fr = pd.DataFrame([
        _row(1, "2026-01-14 00:00", "2026-01-14 06:39:59", base),   # sp 23,999: out of band
        _row(2, "2026-01-14 00:00", "2026-01-14 06:40:00", base),   # sp 24,000: in band
        _row(3, "2026-01-14 00:00", "2026-01-14 06:40:00", base, unmatched=False),
        _row(4, "2026-01-14 00:00", "2026-01-14 06:40:00", base, ap="LSZH"),
    ])
    out, changed = rl.apply_rld(fr)
    assert changed.tolist() == [False, True, False, False] and out.tolist() == [100, 24000, 100, 100]


def test_the_rule_lowers_as_well_as_raises_and_rounds():
    """Kills: an up-only rule (E1's shape) — the review's row 202685083 must come DOWN; kills truncation."""
    fr = pd.DataFrame([_row(1, "2026-01-14 00:40:00", "2026-01-14 14:14:01", 62215),       # sp 48,841
                       _row(2, "2026-01-14 00:40:00.4", "2026-01-14 14:14:01", 30000)])    # sp 48,840.6 -> 48,841
    out, changed = rl.apply_rld(fr)
    assert out.tolist() == [48841, 48841] and changed.tolist() == [True, True]
    assert out.dtype == np.int32
