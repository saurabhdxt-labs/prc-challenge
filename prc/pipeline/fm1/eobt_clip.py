"""FM1 EOBT clip — a serve-time repair of the EOBT_1 field where its production changed
(plans/PREREG_clp_lfpg_eobt_clip_2026_10_02.md).

From July 2026 the EOBT_1 field at LFPG carries delay revisions it did not carry in 2025: the share of matched departures
with EOBT_1 later than LOBT is 3.9 - 7.1 % in every 2025 month and 6.1 % in January 2026, 35.1 % in July 2026 (DRF audit),
while FM1 learned the 2025 field (`aobt_eobt` is its largest input). In a cell whose field carries revisions the rule
restores the 2025 meaning on every eligible row; elsewhere it does nothing:

    eligible  a departure at an airport in modules.eobt_clip.airports with AOBT_3, EOBT_1, LOBT and MVT_TIME present
              (EPX's row selection, scripts/analysis/epx_price.py, plus the movement time that keys the cell)
    cell      (airport, UTC year-month of MVT_TIME)
    share     eligible rows with EOBT_1 > LOBT / eligible rows of the cell
    fires     share >= modules.eobt_clip.min_share; a cell with 1 ... min_rows - 1 eligible rows is refused (DetectorError)
    clip      EOBT_1 := min(EOBT_1, LOBT) on every eligible row of a fired cell (epx_price.clip_eobt); nothing else moves

Population (LDI's contract). When the batch's eligible rows come from data.scored_file, the shares are read from the scored
file's whole cells, so every batch of the scored file gets the same decision; a batch with no scored-file eligible row is
its own population (a 2025 validation month); a batch mixing the two is refused (PopulationError).

Predict time only: engine.predict clips the raw rows before the feature path, so every stage sees the clipped field, as in
EPX's pricing; fit never calls this module (the model is trained on the unclipped field).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import errors as E

#: the raw columns the rule reads
COLS = ("MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt")
TABLE_COLS = ["airport", "month", "n", "n_late", "share"]


def _check(rows: pd.DataFrame) -> None:
    missing = [c for c in COLS if c not in rows.columns]
    if missing:
        raise E.MissingColumnError(f"EOBT clip reads the raw column(s) {missing}")


def eligible(rows: pd.DataFrame, airports) -> np.ndarray:
    """Boolean mask of the rows the rule may read and clip."""
    _check(rows)
    return ((rows.PHASE_mvt.astype(str) == "DEP") & rows.ADEP_mvt.astype(str).isin(list(airports))
            & rows.AOBT_3_flt.notna() & rows.EOBT_1_flt.notna() & rows.LOBT_flt.notna()
            & rows.MVT_TIME_UTC_mvt.notna()).to_numpy()


def _cell_keys(rows: pd.DataFrame, mask: np.ndarray) -> tuple:
    r = rows.loc[mask]
    return r.ADEP_mvt.astype(str).to_numpy(), pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m").to_numpy()


def cell_table(rows: pd.DataFrame, airports) -> pd.DataFrame:
    """One row per (airport, month) cell with eligible rows: n, n_late (EOBT_1 strictly later than LOBT), share."""
    mask = eligible(rows, airports)
    if not mask.any():
        return pd.DataFrame(columns=TABLE_COLS)
    ap, month = _cell_keys(rows, mask)
    late = (rows.EOBT_1_flt[mask] > rows.LOBT_flt[mask]).to_numpy()
    g = pd.DataFrame({"airport": ap, "month": month, "late": late}).groupby(["airport", "month"], sort=True).late
    t = pd.DataFrame({"n": g.size(), "n_late": g.sum()}).reset_index()
    t["n"], t["n_late"] = t.n.astype(int), t.n_late.astype(int)
    t["share"] = t.n_late / t.n
    return t[TABLE_COLS]


def decide(table: pd.DataFrame, min_share: float, min_rows: int) -> pd.DataFrame:
    """table + `fired`. A cell below the row floor cannot be decided and is refused by name."""
    short = table[table.n < min_rows]
    if len(short):
        c = short.iloc[0]
        raise E.DetectorError(f"EOBT clip: cell {c.airport} {c.month} has {int(c.n)} eligible row(s), fewer than "
                              f"modules.eobt_clip.min_rows = {min_rows}; its share cannot be read ({len(short)} cell(s) "
                              "short). Serve whole months, or lower the floor explicitly.")
    out = table.copy()
    out["fired"] = (out.share >= min_share).to_numpy(dtype=bool)
    return out


def population(rows: pd.DataFrame, scored: pd.DataFrame, airports) -> str:
    """'scored_file' when the batch's eligible rows all come from `scored`, 'batch' when none does; refused otherwise."""
    ids = rows.MVT_ID_mvt.to_numpy(dtype="float64")[eligible(rows, airports)]
    inside = pd.Index(ids).isin(scored.MVT_ID_mvt.to_numpy(dtype="float64"))
    if not inside.any():
        return "batch"
    if not inside.all():
        raise E.PopulationError(f"EOBT clip: the batch mixes {int(inside.sum())} scored-file eligible departure(s) with "
                                f"{int((~inside).sum())} from elsewhere; the cell test needs one population")
    return "scored_file"


def serve(rows: pd.DataFrame, airports, min_share: float, min_rows: int, scored: pd.DataFrame) -> tuple:
    """(rows with EOBT_1 clipped in the fired cells — a new frame, the input is not modified; the ids whose EOBT_1 moved;
    report)."""
    airports = list(airports)
    mask = eligible(rows, airports)
    rep = {"airports": airports, "min_share": float(min_share), "min_rows": int(min_rows), "population": None,
           "cells": [], "n_clipped": 0}
    if not mask.any():
        return rows, np.array([], dtype="float64"), rep
    rep["population"] = population(rows, scored, airports)
    pop = scored if rep["population"] == "scored_file" else rows
    cells = decide(cell_table(pop, airports), min_share, min_rows)
    fired = {(a, m) for a, m, f in zip(cells.airport, cells.month, cells.fired) if f}
    ap, month = _cell_keys(rows, mask)
    in_fired = np.zeros(len(rows), dtype=bool)
    in_fired[np.flatnonzero(mask)] = [(a, m) in fired for a, m in zip(ap, month)]
    clip = in_fired & (rows.EOBT_1_flt > rows.LOBT_flt).to_numpy()
    eobt = rows.EOBT_1_flt
    new = rows.assign(EOBT_1_flt=eobt.mask(clip, rows.LOBT_flt).astype(eobt.dtype))
    ids = rows.MVT_ID_mvt.to_numpy(dtype="float64")[clip]
    rep["cells"] = [{"airport": str(r.airport), "month": str(r.month), "n": int(r.n), "n_late": int(r.n_late),
                     "share": float(r.share), "fired": bool(r.fired)} for r in cells.itertuples()]
    rep["n_clipped"] = int(clip.sum())
    return new, ids, rep


def apply(cfg, rows: pd.DataFrame) -> tuple:
    """The engine's entry point: the knobs, and the scored file's cells as the population for a batch drawn from it."""
    scored = pd.read_parquet(cfg.path("data.scored_file"), columns=list(COLS))
    return serve(rows, cfg["modules.eobt_clip.airports"], cfg["modules.eobt_clip.min_share"],
                 cfg["modules.eobt_clip.min_rows"], scored)
