"""Agent E (contrastive twins) rules for the LIRF unmatched fill / ordinary call.

Signal: the "reactionary turnaround" signature read from the departure's own
stand.  For an unmatched LIRF departure with scheduled off-block ``s``,
take-off ``m`` and stand ``S``:

* ``k_in``  = number of LIRF arrivals whose in-block (ARR BLOCK_TIME_UTC_mvt)
  at stand ``S`` falls in (s, m - 900 s)  -> "the inbound came onto the stand
  after the departure should already have left".
* ``ta``    = m - (latest ARR in-block at stand ``S`` that is <= m - 900 s)
  -> time from the inbound reaching the stand to take-off.
* ``turn``  = 2,700 <= ta < 5,700 s  (a minimum-time turnaround + taxi).

Late departures whose delay is a quick turn of a late inbound are recorded as
FILL (BLOCK := SCHED) much more often than R predicts; late departures whose
aircraft sat on the stand for a long time are more often ORDINARY.

Serve-available inputs only: DEP rows' STAND / SCHED / MVT (floored to the
minute) and ARR rows' STAND / BLOCK (floored to the minute).  No DEP label,
no sub-minute seconds.  Context is read from data/raw/ for the rows' own
months (plus the previous month for rows scheduled on the 1st).

Returned arrays are p_fill in [0, 1] or NaN (no opinion -> keep R), in row order.
"""
from __future__ import annotations

import glob
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd

RAW_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "raw"
)
AIRPORT = "LIRF"
GUARD_S = 900.0          # inbound must reach the stand >= 15 min before take-off
TURN_LO_S = 2700.0       # minimum-time turn + taxi window (LOMO-stable 2,400-6,000)
TURN_HI_S = 5700.0
SP_LO_S, SP_HI_S = 3600.0, 24000.0   # rules speak only in the hard band

# rule_E1 constant: fill share when (k_in >= 1 & turn) in the 2025 discovery
# hard band = 216 / 269 = 0.803 (LOMO constants 0.79-0.83).
E1_P_FILL = 0.80
# rule_E2 logistic stack fitted on all 962 discovery hard-band rows:
# logit p = a*logit(p_R) + b*k_in_flag + c*turn + d
E2_A, E2_B, E2_C, E2_D = 0.685, 0.601, 1.254, -0.451

_T0 = pd.Timestamp("2025-01-01", tz="UTC")


def _sec_minute(ts: pd.Series) -> np.ndarray:
    """UTC timestamps -> seconds since 2025-01-01, floored to the minute."""
    ts = pd.to_datetime(ts, utc=True)
    return (ts.dt.floor("min") - _T0).dt.total_seconds().to_numpy(dtype=float)


def _files_for(year_months: Iterable[tuple[int, int]], raw_dir: str) -> list[str]:
    out = []
    for y, mth in sorted(set(year_months)):
        tag = f"{y:04d}-{mth:02d}-01"
        hits = glob.glob(os.path.join(raw_dir, f"training_{tag}_*.parquet"))
        if hits:
            out.extend(hits)
        elif y >= 2026:
            p = os.path.join(raw_dir, "ranking.parquet")
            if os.path.exists(p):
                out.append(p)
    return sorted(set(out))


def _load_arrivals(year_months: Iterable[tuple[int, int]], raw_dir: str) -> pd.DataFrame:
    cols = ["ADES_mvt", "PHASE_mvt", "STAND_mvt", "BLOCK_TIME_UTC_mvt"]
    parts = []
    for f in _files_for(year_months, raw_dir):
        r = pd.read_parquet(f, columns=cols)
        r = r[(r.PHASE_mvt == "ARR") & (r.ADES_mvt == AIRPORT)]
        parts.append(r.dropna(subset=["STAND_mvt", "BLOCK_TIME_UTC_mvt"]))
    if not parts:
        raise FileNotFoundError(f"no raw files for {sorted(set(year_months))} in {raw_dir}")
    a = pd.concat(parts, ignore_index=True)
    a["ib"] = _sec_minute(a.BLOCK_TIME_UTC_mvt)
    a["STAND_mvt"] = a.STAND_mvt.astype(str)
    return a[["STAND_mvt", "ib"]].drop_duplicates().sort_values("ib", kind="mergesort")


def stand_inbound_features(
    rows: pd.DataFrame,
    months: Optional[Iterable[tuple[int, int]]] = None,
    raw_dir: str = RAW_DIR_DEFAULT,
) -> pd.DataFrame:
    """Per-row k_in, ta (seconds) and turn flag; NaN where the stand is unknown.

    ``months``: (year, month) pairs whose raw files supply ARR context; default =
    the rows' own SCHED months, plus the previous month for rows on the 1st.
    """
    for c in ("STAND_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"):
        if c not in rows.columns:
            raise KeyError(f"rows missing required column {c}")
    n = len(rows)
    s = _sec_minute(rows.SCHED_TIME_UTC_mvt)
    m = _sec_minute(rows.MVT_TIME_UTC_mvt)
    if months is None:
        sched = pd.to_datetime(rows.SCHED_TIME_UTC_mvt, utc=True)
        ym = set()
        for t in sched.dropna():
            ym.add((t.year, t.month))
            if t.day == 1:  # an inbound may have in-blocked the previous evening
                p = t - pd.Timedelta(days=1)
                ym.add((p.year, p.month))
        months = ym
    arr = _load_arrivals(months, raw_dir)
    by_stand = {k: g.ib.to_numpy() for k, g in arr.groupby("STAND_mvt", sort=False)}
    stand = rows.STAND_mvt.astype(object).where(rows.STAND_mvt.notna(), None)
    k_in = np.full(n, np.nan)
    ta = np.full(n, np.nan)
    for i in range(n):
        st = stand.iloc[i]
        if st is None or np.isnan(s[i]) or np.isnan(m[i]):
            continue
        ib = by_stand.get(str(st))
        if ib is None:
            k_in[i] = 0.0
            continue
        cut = m[i] - GUARD_S
        j = np.searchsorted(ib, cut, side="right")   # ib[:j] <= cut
        if j > 0:
            ta[i] = m[i] - ib[j - 1]
        k_in[i] = float(np.sum((ib[:j] > s[i]) & (ib[:j] < cut)))
    turn = np.where(np.isnan(ta), np.nan, ((ta >= TURN_LO_S) & (ta < TURN_HI_S)).astype(float))
    return pd.DataFrame({"k_in": k_in, "ta": ta, "turn": turn}, index=rows.index)


def _sp(rows: pd.DataFrame) -> np.ndarray:
    if "sp" in rows.columns:
        return rows.sp.to_numpy(dtype=float)
    return (pd.to_datetime(rows.MVT_TIME_UTC_mvt, utc=True)
            - pd.to_datetime(rows.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds().to_numpy()


def rule_E1(rows: pd.DataFrame, months=None, raw_dir: str = RAW_DIR_DEFAULT) -> np.ndarray:
    """If a 1-7 h late departure's stand received an arrival after its scheduled
    off-block and that arrival reached the stand 45-95 min before take-off
    (quick turn of a late inbound), say fill with p = 0.80; otherwise no opinion.
    Standalone (does not see R); weak on its own - see the report."""
    f = stand_inbound_features(rows, months, raw_dir)
    sp = _sp(rows)
    fire = ((sp >= SP_LO_S) & (sp <= SP_HI_S)
            & (f.k_in.to_numpy() >= 1) & (f.turn.to_numpy() == 1))
    out = np.full(len(rows), np.nan)
    out[fire] = E1_P_FILL
    return out


def rule_E2(rows: pd.DataFrame, months=None, raw_dir: str = RAW_DIR_DEFAULT) -> np.ndarray:
    """Update R's fill probability in the 1-7 h band with the stand-inbound
    signature: logit p = 0.685*logit(p_R) + 0.601*[inbound in-blocked after
    SCHED] + 1.254*[inbound-to-take-off 45-95 min] - 0.451.
    Needs a ``p_R`` column (R's fill probability); returns all-NaN without it."""
    out = np.full(len(rows), np.nan)
    if "p_R" not in rows.columns:
        return out
    f = stand_inbound_features(rows, months, raw_dir)
    sp = _sp(rows)
    p = np.clip(rows.p_R.to_numpy(dtype=float), 1e-4, 1 - 1e-4)
    lp = np.log(p / (1 - p))
    kin = (f.k_in.to_numpy() >= 1).astype(float)
    turn = np.nan_to_num(f.turn.to_numpy(), nan=0.0)
    z = E2_A * lp + E2_B * kin + E2_C * turn + E2_D
    ok = (sp >= SP_LO_S) & (sp <= SP_HI_S) & np.isfinite(lp) & f.k_in.notna().to_numpy()
    out[ok] = 1.0 / (1.0 + np.exp(-z[ok]))
    return out
