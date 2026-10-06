"""TC — the unmatched lane's tail-cell rule: whether the learned body beats the shipped hybrid's cell-mean branch
on the rows it decides.

Registration: plans/PREREG_tc_tail_cell_routing_2026_10_05.md (REG 9eaf025b). H-TC: on the population (lane
unmatched, decided_by in SCOPE, unm_nf_cells >= T_TAIL_S -- exactly the rows `build_submission.nf_hybrid` routes to
the cell mean, never below it), the learned body (`alt_value`) is the better not-copy value than the shipped cell
mean.

    python research/analysis/tc_price.py count --lomo-dir D --months 1,7 --out counts.json
    python research/analysis/tc_price.py verdict --lomo-dir D --raw-dir R --out report.json

`count` is label-free: it opens no `y` column and is the one command safe to run on the real scored months.
`verdict` needs `y` and must run only on data a caller has the right to label-read; this repository's own binding
rule is that it is never pointed at the real January/July 2025 LOMO screen files.

T_TAIL_S and `mixture` / `nf_hybrid` are the SHIPPED module's (imported below as `bs`, never copied or
redefined): a local duplicate of the threshold or the formula could drift from the rule being priced.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bs = _load("build_submission", "scripts/build_submission.py")

REGISTRATION = "9eaf025b"
N_BOARD = 344_841
SCOPE = ("unmatched_body", "storm")
SCORED_MONTHS = (1, 7)
RECON_TOL_S = 1.0
NOLABEL_COLS = ["MVT_ID_mvt", "lane", "decided_by", "month", "day", "ap", "pred", "calibration_shift", "unm_p_fill",
                "unm_nf_cells", "unm_body", "storm_shift"]


# ---------------------------------------------------------------------------------------------- frame assembly

def _read_months(lomo_dir: pathlib.Path, months, with_labels: bool) -> pd.DataFrame:
    """NOLABEL_COLS (+ 'y' iff with_labels) from each requested month's oof_M{mm:02d}.parquet in lomo_dir,
    concatenated in month order. Refuses a missing month file or a duplicated MVT_ID_mvt across the requested
    files."""
    lomo_dir = pathlib.Path(lomo_dir)
    cols = list(NOLABEL_COLS) + (["y"] if with_labels else [])
    frames = []
    for m in months:
        f = lomo_dir / f"oof_M{m:02d}.parquet"
        if not f.exists():
            raise ValueError(f"tc_price: missing LOMO month file {f}")
        frames.append(pd.read_parquet(f, columns=cols))
    frame = pd.concat(frames, ignore_index=True)
    dup = frame["MVT_ID_mvt"].duplicated()
    if dup.any():
        raise ValueError(f"tc_price: {int(dup.sum())} duplicate MVT_ID_mvt across the requested month file(s) "
                         f"{list(months)}")
    return frame


def load(lomo_dir, raw_dir, months, with_labels: bool) -> pd.DataFrame:
    """The requested months' LOMO rows (NOLABEL_COLS, + y iff with_labels), with `sp` (seconds from
    SCHED_TIME_UTC_mvt to MVT_TIME_UTC_mvt) attached from the single raw file matching
    `<raw_dir>/training_2025-{mm:02d}-*.parquet` for each requested month.

    Refuses: a missing LOMO month file; zero or several raw files matching one month; a duplicated MVT_ID_mvt in
    the LOMO files or in a raw file; a population row (population(frame)) whose id has no raw row, or whose raw
    row's SCHED or MVT time is null. A non-population row with no raw match, or a null SCHED/MVT, keeps sp NaN
    without raising."""
    lomo_dir = pathlib.Path(lomo_dir)
    raw_dir = pathlib.Path(raw_dir)
    months_sorted = sorted(int(m) for m in months)
    frame = _read_months(lomo_dir, months_sorted, with_labels)
    frame["sp"] = np.nan
    pop = population(frame)
    month_col = frame["month"].astype(int).to_numpy()

    for m in months_sorted:
        matches = sorted(raw_dir.glob(f"training_2025-{m:02d}-*.parquet"))
        if len(matches) != 1:
            raise ValueError(f"load: {len(matches)} raw file(s) match training_2025-{m:02d}-*.parquet in "
                             f"{raw_dir} (need exactly 1)")
        raw = pd.read_parquet(matches[0], columns=["MVT_ID_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"])
        raw_dup = raw["MVT_ID_mvt"].duplicated()
        if raw_dup.any():
            raise ValueError(f"load: {int(raw_dup.sum())} duplicate MVT_ID_mvt in {matches[0].name}")

        in_month = month_col == m
        ids = frame.loc[in_month, "MVT_ID_mvt"]
        # DataFrame.reindex (not a per-column detour through .to_numpy()) keeps each column's own tz-aware dtype
        # even where every reindexed row is NaT; subtracting the two resulting Series stays inside pandas' own
        # datetime arithmetic, which is what tz-aware NaT needs (BC-class: a numpy round-trip silently drops tz
        # on an all-NaT slice, raising "Cannot subtract tz-naive and tz-aware datetime-like objects").
        aligned = raw.set_index("MVT_ID_mvt").reindex(ids.to_numpy())
        sp = (aligned["MVT_TIME_UTC_mvt"] - aligned["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
        frame.loc[in_month, "sp"] = sp.to_numpy(dtype="float64")

    missing = pop & frame["sp"].isna().to_numpy()
    if missing.any():
        raise ValueError(f"load: {int(missing.sum())} population row(s) have no raw row or a null SCHED/MVT")
    return frame


# -------------------------------------------------------------------------------------------------- population

def population(frame: pd.DataFrame) -> np.ndarray:
    """Rows the shipped hybrid routes to the cell mean (`build_submission.nf_hybrid` keeps `cells` exactly when it
    is NOT below T_tail -- this mirrors that comparison, never `< T_tail`): lane unmatched, decided_by in SCOPE,
    unm_nf_cells >= bs.T_TAIL_S."""
    lane = frame["lane"].to_numpy()
    decided = frame["decided_by"]
    cells = frame["unm_nf_cells"].to_numpy(dtype="float64")
    return (lane == "unmatched") & decided.isin(SCOPE).to_numpy() & (cells >= bs.T_TAIL_S)


# --------------------------------------------------------------------------------------------------- alt_value

def alt_value(pred, p, cells, body) -> np.ndarray:
    """rint(max(pred - (1 - p) * (cells - body), 1)): pred with the shipped hybrid's cell-mean contribution
    replaced by the learned body's. Refuses when any of the four inputs is not finite (NaN or +/-inf)."""
    pred = np.asarray(pred, dtype="float64")
    p = np.asarray(p, dtype="float64")
    cells = np.asarray(cells, dtype="float64")
    body = np.asarray(body, dtype="float64")
    if not (np.isfinite(pred).all() and np.isfinite(p).all() and np.isfinite(cells).all()
           and np.isfinite(body).all()):
        raise ValueError("alt_value: pred, p, cells and body must all be finite")
    return np.rint(np.maximum(pred - (1.0 - p) * (cells - body), 1.0))


# ------------------------------------------------------------------------------------------- check_reconstruction

def check_reconstruction(frame: pd.DataFrame, mask) -> dict:
    """Label-free proof that, on the rows `mask` selects, the lane's served `pred` is still at the shipped hybrid's
    cell-mean mixture: expected = rint(bs.mixture(p, sp, cells)) + (storm_shift, 0 where null), within
    RECON_TOL_S seconds. Refuses a non-zero, non-null calibration_shift on those rows (a later stage would have
    moved `pred` for a reason this reconstruction does not model) and any row over tolerance. n == 0 (no row
    selected) short-circuits to {"n": 0, "max_abs_diff_s": 0.0} before reading any column."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "max_abs_diff_s": 0.0}
    sub = frame.loc[mask]
    cal = sub["calibration_shift"].to_numpy(dtype="float64")
    bad_cal = np.isfinite(cal) & (cal != 0.0)
    if bad_cal.any():
        raise ValueError(f"check_reconstruction: {int(bad_cal.sum())} row(s) have a non-zero, non-null "
                         f"calibration_shift")
    p = sub["unm_p_fill"].to_numpy(dtype="float64")
    sp = sub["sp"].to_numpy(dtype="float64")
    cells = sub["unm_nf_cells"].to_numpy(dtype="float64")
    shift = np.nan_to_num(sub["storm_shift"].to_numpy(dtype="float64"), nan=0.0)
    expected = np.rint(bs.mixture(p, sp, cells)) + shift
    pred = sub["pred"].to_numpy(dtype="float64")
    diff = np.abs(pred - expected)
    max_diff = float(diff.max())
    if max_diff > RECON_TOL_S:
        over = int((diff > RECON_TOL_S).sum())
        raise ValueError(f"check_reconstruction: {over} row(s) over tolerance ({RECON_TOL_S}s); "
                         f"max |diff| = {max_diff}")
    return {"n": n, "max_abs_diff_s": max_diff}


# ------------------------------------------------------------------------------------------------------- count

def count(lomo_dir, months) -> dict:
    """Label-free population counts of `months` (no raw join, no sp; works on files with no `y` column at all)."""
    months_sorted = sorted(int(m) for m in months)
    frame = _read_months(lomo_dir, months_sorted, with_labels=False)
    pop = population(frame)
    by_month = {m: 0 for m in months_sorted}
    for m, n in frame.loc[pop, "month"].astype(int).value_counts().items():
        by_month[int(m)] = int(n)
    by_airport: dict = {}
    for a, n in frame.loc[pop, "ap"].astype(str).value_counts().items():
        by_airport[str(a)] = int(n)
    return {"by_month": by_month, "by_airport": by_airport, "n": int(pop.sum())}


# ----------------------------------------------------------------------------------------------------- verdict

def verdict(frame: pd.DataFrame) -> dict:
    """H-TC on the population rows of `frame` (needs the label column y). term = (y-pred)**2 - (y-alt)**2;
    G = sum(term) / N_BOARD; k = count(term > 0). "NOT TESTABLE" if n == 0 (no population row); "TRUE" if G > 0
    and k >= 2; "FALSE" if G <= 0; else "NOT ESTABLISHED" (G > 0 on a single row)."""
    if "y" not in frame.columns:
        raise ValueError("verdict needs the label column y")
    pop = population(frame)
    sub = frame.loc[pop]
    n = int(len(sub))
    if n == 0:
        return {"verdict": "NOT TESTABLE", "n": 0, "k": 0, "G_bmse": 0.0, "rows": [], "registration": REGISTRATION}

    pred = sub["pred"].to_numpy(dtype="float64")
    p = sub["unm_p_fill"].to_numpy(dtype="float64")
    cells = sub["unm_nf_cells"].to_numpy(dtype="float64")
    body = sub["unm_body"].to_numpy(dtype="float64")
    y = sub["y"].to_numpy(dtype="float64")
    alt = alt_value(pred, p, cells, body)
    term = (y - pred) ** 2 - (y - alt) ** 2
    G = float(term.sum() / N_BOARD)
    k = int((term > 0).sum())

    if G > 0 and k >= 2:
        v = "TRUE"
    elif G <= 0:
        v = "FALSE"
    else:
        v = "NOT ESTABLISHED"

    rows = [{"ap": str(a), "day": str(d), "pred": float(pr), "alt": float(al), "cells": float(c), "body": float(b),
            "p": float(pp), "y": float(yy), "term": float(t)}
           for a, d, pr, al, c, b, pp, yy, t in
           zip(sub["ap"].to_numpy(), sub["day"].to_numpy(), pred, alt, cells, body, p, y, term)]
    return {"verdict": v, "n": n, "k": k, "G_bmse": G, "rows": rows, "registration": REGISTRATION}


# ---------------------------------------------------------------------------------------------------------- CLI

def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("count", help="label-free population counts by month and airport")
    pc.add_argument("--lomo-dir", required=True)
    pc.add_argument("--months", required=True, help="comma-separated, e.g. 1,7")
    pc.add_argument("--out", required=True)

    pv = sub.add_parser("verdict", help="H-TC on SCORED_MONTHS (never point this at real data)")
    pv.add_argument("--lomo-dir", required=True)
    pv.add_argument("--raw-dir", required=True)
    pv.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    out_path = pathlib.Path(a.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")

    if a.command == "count":
        lomo_dir = pathlib.Path(a.lomo_dir).resolve()
        months = tuple(sorted(int(x) for x in a.months.split(",")))
        result = count(lomo_dir, months)
        out_path.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
        print(json.dumps(result, indent=1, sort_keys=True, default=str))
        return 0

    lomo_dir = pathlib.Path(a.lomo_dir).resolve()
    raw_dir = pathlib.Path(a.raw_dir).resolve()
    frame = load(lomo_dir, raw_dir, SCORED_MONTHS, with_labels=True)
    recon = check_reconstruction(frame, population(frame))
    result = verdict(frame)
    result["check_reconstruction"] = recon

    files = [lomo_dir / f"oof_M{m:02d}.parquet" for m in SCORED_MONTHS]
    for m in SCORED_MONTHS:
        files.extend(sorted(raw_dir.glob(f"training_2025-{m:02d}-*.parquet")))
    result["input_sha256"] = {f.name: _sha256(f) for f in files}

    out_path.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(json.dumps({"verdict": result["verdict"], "n": result["n"], "k": result["k"]}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
