"""Scoring for the evaluation harness (P3): paired comparisons on identical rows, day-block intervals, and fold records
that carry their own prediction convention.

Every comparison in this project is paired (two arms, the same holdout rows) and is priced in WEIGHTED FOLD MSE —
`w · ΔSSE / n_fold`, w the lane's share of the scored file — because MSE is additive across rows and lanes and seconds
are not. Intervals resample calendar DATES within each holdout month (rows within a day share weather and congestion;
arm F's row interval was 4.6× too narrow — MSE_LEDGER 2026-09-10 19:55).

Fold records are parquet files whose metadata names the prediction convention (`taxi_time` or `delta`), the tag and the
prereg; `read_record` refuses a file without one, or with another convention than the caller expects. This is the
structural fix BC-2 asked for ("no fold-prediction writer stamps its convention into the file").
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from . import errors as E

W_MATCHED = 339_551 / 344_841          # the matched lane's share of the 2026 scored file
W_UNMATCHED = 5_290 / 344_841
CONVENTIONS = ("taxi_time", "delta")
META_KEYS = (b"convention", b"tag", b"prereg")


def weighted_fold_mse(sigma_g: float, n_fold: int, w: float = W_MATCHED) -> float:
    """w · Σg / n_fold, Σg = ΔSSE (control − treatment): positive = the treatment removes error."""
    if n_fold <= 0:
        raise ValueError(f"n_fold must be positive, got {n_fold}")
    if not 0.0 < w <= 1.0:
        raise ValueError(f"lane weight {w} outside (0, 1]")
    return float(sigma_g) * float(w) / int(n_fold)


def dayblock_interval(g, day, month, n_boot: int = 2000, seed: int = 0) -> tuple:
    """(Σg, 2.5%, 97.5%, n_dates): Σg re-summed over dates drawn with replacement within each month separately."""
    g = np.asarray(g, dtype="float64")
    if len(g) != len(day) or len(g) != len(month):
        raise ValueError(f"g ({len(g)}), day ({len(day)}) and month ({len(month)}) differ in length")
    if not np.isfinite(g).all():
        raise ValueError(f"{int((~np.isfinite(g)).sum())} non-finite per-row gains")
    codes, uniq = pd.factorize(pd.Series(day))
    if (codes < 0).any():
        raise ValueError("a row has no date")
    per_date = np.bincount(codes, weights=g)
    mon = pd.Series(np.asarray(month)).groupby(codes).agg(["min", "max"])
    if (mon["min"] != mon["max"]).any():
        raise ValueError("a date spans two months")
    mon = mon["min"].to_numpy()
    groups = [np.flatnonzero(mon == m) for m in sorted(set(mon.tolist()))]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = sum(per_date[rng.choice(gr, size=len(gr), replace=True)].sum() for gr in groups)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(g.sum()), float(lo), float(hi), int(len(uniq))


def paired(y, control, treatment, day, month, n_fold: int | None = None, w: float = W_MATCHED, cuts=None,
           n_boot: int = 2000, seed: int = 0) -> dict:
    """The paired comparison of two arms on identical rows: net weighted fold MSE, its day-block interval, both RMSEs
    and the net of every named cut (`cuts`: {name: boolean mask})."""
    y, a, b = (np.asarray(v, dtype="float64") for v in (y, control, treatment))
    if not (y.shape == a.shape == b.shape):
        raise ValueError(f"y {y.shape}, control {a.shape}, treatment {b.shape}: not the same rows")
    n = int(n_fold if n_fold is not None else len(y))
    g = (y - a) ** 2 - (y - b) ** 2
    tot, lo, hi, nd = dayblock_interval(g, day, month, n_boot, seed)
    out = {"n_rows": int(len(y)), "n_fold": n, "net": weighted_fold_mse(tot, n, w),
           "dayblock95": [weighted_fold_mse(lo, n, w), weighted_fold_mse(hi, n, w)], "n_dates": nd,
           "rmse_control": float(np.sqrt(((y - a) ** 2).mean())), "rmse_treatment": float(np.sqrt(((y - b) ** 2).mean())),
           "cuts": {}}
    for name, m in (cuts or {}).items():
        m = np.asarray(m, dtype=bool)
        if m.shape != g.shape:
            raise ValueError(f"cut {name!r} has shape {m.shape}, expected {g.shape}")
        out["cuts"][name] = {"n": int(m.sum()), "net": weighted_fold_mse(float(g[m].sum()), n, w)}
    return out


def write_record(frame: pd.DataFrame, path, convention: str, tag: str, prereg: str, meta: dict | None = None) -> None:
    """A fold record: the frame plus convention / tag / prereg (and any `meta`) in the parquet metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    if not tag or not prereg:
        raise ValueError("a fold record needs a tag and the prereg it answers to")
    t = pa.Table.from_pandas(frame, preserve_index=False)
    md = {**(t.schema.metadata or {}), b"convention": convention.encode(), b"tag": tag.encode(),
          b"prereg": prereg.encode(), b"meta": json.dumps(meta or {}, default=str).encode()}
    pq.write_table(t.replace_schema_metadata(md), pathlib.Path(path))


def read_record(path, convention: str, tag: str | None = None) -> tuple:
    """(frame, meta). Refuses a record without a convention stamp, with another convention, or of another tag."""
    import pyarrow.parquet as pq
    t = pq.read_table(pathlib.Path(path))
    md = t.schema.metadata or {}
    got = md.get(b"convention", b"").decode()
    if not got:
        raise E.SchemaError(f"{pathlib.Path(path).name} carries no convention stamp (BC-2): refusing to guess its scale")
    if got != convention:
        raise E.SchemaError(f"{pathlib.Path(path).name} is on the {got!r} scale, the caller expects {convention!r} (BC-2)")
    if tag is not None and md.get(b"tag", b"").decode() != tag:
        raise E.SchemaError(f"{pathlib.Path(path).name} is tagged {md.get(b'tag', b'').decode()!r}, expected {tag!r}")
    meta = {k.decode(): v.decode() for k, v in md.items() if k in META_KEYS}
    meta["meta"] = json.loads(md.get(b"meta", b"{}").decode())
    return t.to_pandas(), meta
