"""Rule RLD (plans/PREREG_rome_local_day_rule_2026_09_11.md): scored LIRF UNMATCHED departures with
24,000 <= sp < 86,400 s whose take-off falls on the schedule's LOCAL (Europe/Rome) calendar day take taxi = rint(sp).

A 24 h date-slip needs take-off on a later local day than the schedule, so here it cannot occur; the 2025 rows in this
cell were schedule fills 14 of 14. The rule lowers as well as raises; E1 (rome_bandfloor, up-only) is untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AIRPORT, TZ = "LIRF", "Europe/Rome"
SP_LO, SP_HI = 24_000.0, 86_400.0


def local_dayoff(sched_utc: pd.Series, mvt_utc: pd.Series, tz: str = TZ) -> np.ndarray:
    """Calendar days from the schedule's local date to the take-off's local date."""
    s = pd.to_datetime(sched_utc, utc=True).dt.tz_convert(tz).dt.normalize()
    m = pd.to_datetime(mvt_utc, utc=True).dt.tz_convert(tz).dt.normalize()
    return (m - s).dt.days.to_numpy()


def apply_rld(fr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """fr: MVT_ID_mvt, ADEP_mvt, unmatched (bool), base (current taxi), SCHED_TIME_UTC_mvt, MVT_TIME_UTC_mvt.
    Returns (new taxi values as int32, mask of rows the rule changed)."""
    sp = (pd.to_datetime(fr.MVT_TIME_UTC_mvt, utc=True) - pd.to_datetime(fr.SCHED_TIME_UTC_mvt, utc=True)) \
        .dt.total_seconds().to_numpy()
    scope = ((fr.ADEP_mvt.to_numpy() == AIRPORT) & fr.unmatched.to_numpy(bool) & (sp >= SP_LO) & (sp < SP_HI)
             & (local_dayoff(fr.SCHED_TIME_UTC_mvt, fr.MVT_TIME_UTC_mvt) == 0))
    base = fr.base.to_numpy()
    new = np.where(scope, np.rint(sp), base).astype(np.int32)
    return new, new != base.astype(np.int32)
