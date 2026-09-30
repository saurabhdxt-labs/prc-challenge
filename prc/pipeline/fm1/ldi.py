"""FM1 LDI — the LTFM de-icing correction v41 shipped (plans/PREREG_ldi_ltfm_deicing_2026_09_27.md; owner decision
2026-09-29: the REGISTERED constants, the target rows computed here, the constants never re-derived from raw).

The definitions come from scripts/analysis/ldi_correction.py's public functions (shipped.ldi_correction(): to_epoch_s,
local_day, band, severe_days; tested in tests/test_ldi_correction.py), composed exactly as its frame assembly did for v41:

    rows      LTFM's matched departures (the matched lane's frame: AOBT_3 present), any month
    sp        MVT - SCHED (s)                          band      (-inf, 1800], (1800, 3600], (3600, inf) s
    day       the local day, UTC + 3 h (LTFM has no DST)
    severe    over the day's LTFM matched departures: share(sp > 1800) >= 0.5 AND min w_temp_c <= 1.0
    target    w_freezing == 1 on a severe day (missing weather: never)
    value     v + delta[band] on the target rows whose delta is positive (ldi_run.values_for_v39); decided_by 'ldi'

Whole-day contract. `severe` reads EVERY LTFM matched departure of the day, so the rows a predict call receives decide
which rows fire. A batch drawn from the scored file must hold every LTFM departure of each local day it touches (serve
the whole scored file, as the unmatched lane's set-level rules require); a partial day, or a batch mixing scored-file
rows with others, raises PartialDayError. A batch with no scored-file LTFM departure (a 2025 validation month) is its own
population, and the report says which population was used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import errors as E
from . import shipped

AIRPORT = "LTFM"
#: band -> the knob holding its shift (the band names are ldi_correction.BANDS)
DELTA_KNOBS = {"<30": "modules.ldi.delta_le30_s", "30-60": "modules.ldi.delta_30_60_s", ">60": "modules.ldi.delta_gt60_s"}
FRAME_COLS = ("MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "w_freezing", "w_temp_c")


def _lc():
    return shipped.ldi_correction()


def deltas(cfg) -> dict:
    """{band: shift in s} from the config's knobs."""
    return {b: float(cfg[k]) for b, k in DELTA_KNOBS.items()}


def _check_deltas(d: dict) -> None:
    if set(d) != set(_lc().BANDS):
        raise E.ConfigValueError(f"LDI deltas must name the bands {list(_lc().BANDS)}, got {sorted(d)}")
    v = np.array([d[b] for b in _lc().BANDS], dtype="float64")
    if not np.isfinite(v).all() or (v < 0).any():
        raise E.ConfigValueError(f"LDI deltas must be finite and >= 0 (the registered rule applies max(0, .)), got {d}")


def _check_matched(matched: pd.DataFrame) -> None:
    for c in ("MVT_ID_mvt", "ap", "w_freezing", "w_temp_c"):
        if c not in matched.columns:
            raise E.MissingColumnError(f"LDI reads the matched frame's {c!r} (features.weather must be on)")


def frame(matched: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """The LTFM matched departures with the columns the rule reads: the matched frame's weather + the raw rows' clocks."""
    _check_matched(matched)
    m = matched.loc[matched.ap.astype(str).to_numpy() == AIRPORT, ["MVT_ID_mvt", "w_freezing", "w_temp_c"]]
    ids = m.MVT_ID_mvt.to_numpy(dtype="float64")
    raw = rows.set_index(rows.MVT_ID_mvt.to_numpy(dtype="float64"))
    missing = ~pd.Index(ids).isin(raw.index)
    if missing.any():
        raise E.RoutingError(f"{int(missing.sum())} LTFM matched row(s) are not in the scored rows (e.g. "
                             f"{ids[missing][:3].tolist()})")
    t = raw.loc[ids, ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]].reset_index(drop=True)   # Series keep the tz dtype
    for c in t.columns:
        if t[c].isna().any():
            raise E.NullabilityError(f"{int(t[c].isna().sum())} LTFM matched row(s) without {c}")
    return pd.DataFrame({"MVT_ID_mvt": ids, "MVT_TIME_UTC_mvt": t.MVT_TIME_UTC_mvt,
                         "SCHED_TIME_UTC_mvt": t.SCHED_TIME_UTC_mvt,
                         "w_freezing": m.w_freezing.to_numpy(), "w_temp_c": m.w_temp_c.to_numpy()})


def flags(f: pd.DataFrame) -> pd.DataFrame:
    """f (FRAME_COLS) + sp, day, band, severe, target: ldi_correction's public definitions, composed as its frame
    assembly composed them for v41."""
    lc = _lc()
    f = f.copy()
    t = lc.to_epoch_s(f.MVT_TIME_UTC_mvt)
    f["sp"] = t - lc.to_epoch_s(f.SCHED_TIME_UTC_mvt)
    f["day"] = lc.local_day(t)
    f["band"] = lc.band(f.sp)
    sev = lc.severe_days(f.day, f.sp, f.w_temp_c)
    f["severe"] = f.day.map(sev).astype(bool)
    f["target"] = (f.w_freezing == 1) & f.severe
    return f


def _ltfm_departure_days(rows: pd.DataFrame) -> pd.DataFrame:
    """MVT_ID_mvt + local day of every LTFM departure with a movement time."""
    r = rows.loc[(rows.PHASE_mvt.astype(str).to_numpy() == "DEP") & (rows.ADEP_mvt.astype(str).to_numpy() == AIRPORT)
                 & rows.MVT_TIME_UTC_mvt.notna().to_numpy(), ["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]]
    return pd.DataFrame({"MVT_ID_mvt": r.MVT_ID_mvt.to_numpy(dtype="float64"),
                         "day": _lc().local_day(_lc().to_epoch_s(r.MVT_TIME_UTC_mvt))})


def population(rows: pd.DataFrame, scored: pd.DataFrame) -> str:
    """'scored_file' when the batch's LTFM departures come from `scored` and cover every local day they touch; 'batch'
    when none of them does (another file: its own population). PartialDayError otherwise."""
    b, s = _ltfm_departure_days(rows), _ltfm_departure_days(scored)
    inside = b.MVT_ID_mvt.isin(s.MVT_ID_mvt)
    if len(b) == 0 or not inside.any():
        return "batch"
    if not inside.all():
        raise E.PartialDayError(f"the batch mixes {int(inside.sum())} scored-file LTFM departure(s) with "
                                f"{int((~inside).sum())} from elsewhere; LDI's day test needs one population")
    nb = b.groupby("day").size()
    ns = s.groupby("day").size().reindex(nb.index)
    short = nb[nb < ns]
    if len(short):
        d = short.index[0]
        raise E.PartialDayError(f"LDI's severe-day test reads every LTFM departure of a local day; on {d}: the batch "
                                f"holds {int(nb[d])} of the scored file's {int(ns[d])} ({len(short)} day(s) short). "
                                "Serve whole days (the whole scored file).")
    return "scored_file"


def serve(matched, rows: pd.DataFrame, ledger, stage, delta: dict, scored: pd.DataFrame) -> tuple:
    """Shift the target rows' ledger values by their band's delta. Returns (per-row components indexed by MVT_ID_mvt:
    ldi_band, ldi_severe, ldi_delta; report)."""
    _check_deltas(delta)
    comp = pd.DataFrame(columns=["ldi_band", "ldi_severe", "ldi_delta"], index=pd.Index([], name="MVT_ID_mvt"))
    rep = {"airport": AIRPORT, "deltas": dict(delta), "n_rows": 0, "n_target": 0, "severe_days": [],
           "n_target_by_band": {b: 0 for b in _lc().BANDS}}
    if matched is not None:
        _check_matched(matched)
    if matched is None or not (matched.ap.astype(str).to_numpy() == AIRPORT).any():
        rep["population"] = None
        return comp, rep
    rep["population"] = population(rows, scored)
    f = flags(frame(matched, rows))
    add = f.band.map(delta).astype("float64").to_numpy()
    hit = f.target.to_numpy() & (add > 0)
    ids = f.MVT_ID_mvt.to_numpy(dtype="float64")
    before = ledger.values_of(ids[hit])
    if np.isnan(before).any():
        raise E.RoutingError(f"{int(np.isnan(before).sum())} LDI target row(s) have no value to shift (no value stage "
                             "served them)")
    if hit.any():
        ledger.write(stage, ids[hit], before + add[hit])
    comp = pd.DataFrame({"ldi_band": f.band.to_numpy(), "ldi_severe": f.severe.to_numpy(),
                         "ldi_delta": np.where(hit, add, 0.0)}, index=pd.Index(ids, name="MVT_ID_mvt"))
    by_band = pd.Series(f.band.to_numpy()[hit]).value_counts()
    rep.update(n_rows=int(len(f)), n_target=int(hit.sum()),
               severe_days=sorted(str(d) for d in pd.unique(f.day[f.severe.to_numpy()])),
               n_target_by_band={b: int(by_band.get(b, 0)) for b in _lc().BANDS})
    return comp, rep


def apply(cfg, matched, rows: pd.DataFrame, ledger, stage) -> tuple:
    """The engine's entry point: the deltas from the knobs, the whole-day population from data.scored_file."""
    scored = pd.read_parquet(cfg.path("data.scored_file"),
                             columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    return serve(matched, rows, ledger, stage, deltas(cfg), scored)
