"""RUL — does our offline 2025 gain predict the public board, and which part of it?

Registration: plans/PREREG_rul_ruler_validation_2026_10_04.md (REG 7d789c25, 2026-10-04 13:02:13Z). For each board
version that was priced on a per-row 2025 pair, the offline gain is recomputed from that pair and split into the body
(label <= 3,600 s), the tail and the gain without its top-10 rows; the board delta comes from our own whole-file scores.
Every version must first reproduce its recorded price (a data contract); only versions that pass are split.

    python scripts/analysis/rul_ruler.py --out reports/rul_ruler_report.json

Reads only per-row prediction files and labels of 2025; no board read, no upload. Standard numerics (numpy, pandas,
scipy).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOARD_N = 344_841.0          # scored departures on the board: every pair is scaled to board MSE units the same way
BODY_MAX_S = 3600.0          # the body: flights of at most one hour (the cut fixed in the registration)
TOP_K = 10                   # rows dropped by |gain| for the robustness view
TOL = 0.5                    # reproduction tolerance in the price's own units


# ---------------------------------------------------------------------------------------------- statistics

def row_gain(y, a, b) -> np.ndarray:
    """Per-row gain (y - a)^2 - (y - b)^2: > 0 where the new prediction b is closer than the reference a."""
    y, a, b = (np.asarray(v, dtype="float64") for v in (y, a, b))
    if not (len(y) == len(a) == len(b)):
        raise ValueError(f"lengths differ: y {len(y)}, a {len(a)}, b {len(b)}")
    if not (np.isfinite(y).all() and np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite label or prediction")
    return (y - a) ** 2 - (y - b) ** 2


def gain_split(y, a, b, body_max: float = BODY_MAX_S, top_k: int = TOP_K, n_board: float = BOARD_N) -> dict:
    """G (all rows), B (y <= body_max), T = G - B, and G without the top_k rows by |gain| (stable order), in board
    MSE units (sum / n_board)."""
    g = row_gain(y, a, b)
    body = np.asarray(y, dtype="float64") <= body_max
    top = np.argsort(-np.abs(g), kind="stable")[:top_k]
    keep = np.ones(len(g), dtype=bool)
    keep[top] = False
    return {"G": float(g.sum() / n_board), "B": float(g[body].sum() / n_board), "T": float(g[~body].sum() / n_board),
            "G_minus_top": float(g[keep].sum() / n_board), "n": int(len(g)), "n_body": int(body.sum())}


def board_delta(rmse_base: float, rmse_new: float) -> float:
    """Board MSE change, > 0 = the new file is better."""
    return float(rmse_base) ** 2 - float(rmse_new) ** 2


def reproduces(g_sum: float, n_rows: int, recorded: float, tol: float = TOL):
    """'bmse' or 'per_row' — the convention in which sum(gain) reproduces the recorded price within tol — else None."""
    if abs(g_sum / BOARD_N - recorded) <= tol:
        return "bmse"
    if n_rows and abs(g_sum / n_rows - recorded) <= tol:
        return "per_row"
    return None


def spearman(x, y) -> float:
    return float(stats.spearmanr(np.asarray(x, float), np.asarray(y, float)).statistic)


def pearson(x, y) -> float:
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def sign_agreement(x, y) -> float:
    return float(np.mean(np.sign(np.asarray(x, float)) == np.sign(np.asarray(y, float))))


def ratio_through_origin(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float((x * y).sum() / (x * x).sum())


def verdict(rho: float, sign: float, ratio: float) -> str:
    """The registered shapes: TRUE needs rho >= 0.6, sign agreement >= 0.75 and the ratio in [0.3, 3]; FALSE is
    rho < 0.4 or sign agreement < 0.6; anything else is NOT ESTABLISHED."""
    if rho < 0.4 or sign < 0.6:
        return "FALSE"
    if rho >= 0.6 and sign >= 0.75 and 0.3 <= ratio <= 3.0:
        return "TRUE"
    return "NOT ESTABLISHED"


# ---------------------------------------------------------------------------------------------- pairs

@dataclasses.dataclass(frozen=True)
class Side:
    file: str
    col: str
    delta: bool = False          # the column holds a predicted delta: taxi = max(proxy - delta, 1)


@dataclasses.dataclass(frozen=True)
class Spec:
    version: int
    base_rmse: float
    new_rmse: float
    ref: Side
    new: Side
    key: str | None = None       # join the two files on this column; None = same file or the same row order
    rows: dict | None = None     # rows the split uses: {"ap": (...), "month": (...), "leg": (...), "cells": ((ap, m),)}
    price_rows: dict | None = None
    price_on_all: bool = False   # the recorded price is on every row of the pair (rows only select the split)
    recorded: float | None = None
    note: str = ""


def _read(side: Side, extra) -> pd.DataFrame:
    path = pathlib.Path(side.file)
    path = path if path.is_absolute() else ROOT / path
    cols = {side.col, "y", *extra} | ({"proxy"} if side.delta else set())
    have = set(pq.ParquetFile(path).schema_arrow.names)
    missing = {side.col, "y"} - have
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    return pd.read_parquet(path, columns=sorted(cols & have))


def _taxi(frame: pd.DataFrame, side: Side) -> np.ndarray:
    v = frame[side.col].to_numpy(dtype="float64")
    return np.maximum(frame["proxy"].to_numpy(dtype="float64") - v, 1.0) if side.delta else v


def _mask(frame: pd.DataFrame, rows) -> np.ndarray:
    m = np.ones(len(frame), dtype=bool)
    for k, vals in (rows or {}).items():
        if k == "cells":
            pairs = set(map(tuple, vals))
            m &= np.array([(a, int(mo)) in pairs for a, mo in zip(frame["ap"].astype(str), frame["month"])])
        else:
            m &= frame[k].astype(type(vals[0])).isin(vals).to_numpy()
    return m


def load_pair(spec: Spec, rows) -> tuple:
    """(y, reference taxi, new taxi) on the selected rows. Two files are joined on spec.key or, without a key, must
    hold the same rows in the same order (checked on 'row' when present) with identical labels."""
    want = {"ap", "month", "leg", "row", "MVT_ID_mvt"}
    same = spec.ref.file == spec.new.file
    r = _read(spec.ref, want | ({spec.new.col} if same else set()))
    if same:
        n = r
    else:
        n = _read(spec.new, want)
        if spec.key:
            n = n.set_index(spec.key).reindex(r[spec.key].to_numpy())
            if n["y"].isna().any():
                raise ValueError(f"v{spec.version}: {int(n['y'].isna().sum())} reference rows have no match on {spec.key}")
            n = n.reset_index()
        else:
            if len(n) != len(r) or ("row" in r and "row" in n and not np.array_equal(r["row"], n["row"])):
                raise ValueError(f"v{spec.version}: the two files' row order differs")
        if not np.allclose(r["y"].to_numpy(float), n["y"].to_numpy(float)):
            raise ValueError(f"v{spec.version}: labels differ between the two files")
    m = _mask(r, rows)
    return r["y"].to_numpy(float)[m], _taxi(r, spec.ref)[m], _taxi(n, spec.new)[m]


# ---------------------------------------------------------------------------------------------- evaluation

def evaluate(specs) -> tuple:
    """(checks, rows). Every version's reproduction check is decided first; only versions that reproduce their
    recorded price are then split (a spec with recorded=None is an unverified pair: split, flagged)."""
    checks, passed = [], []
    for s in specs:
        c = {"version": s.version, "recorded": s.recorded, "convention": None, "excluded": None}
        if s.recorded is None:
            c["excluded"] = None
            c["unverified"] = True
            passed.append(s)
        else:
            price_rows = None if s.price_on_all else (s.price_rows if s.price_rows is not None else s.rows)
            y, a, b = load_pair(s, rows=price_rows)
            g_sum = float(row_gain(y, a, b).sum())
            c.update(g_bmse=g_sum / BOARD_N, g_per_row=g_sum / len(y), n_price=int(len(y)))
            c["convention"] = reproduces(g_sum, len(y), s.recorded)
            if c["convention"] is None:
                c["excluded"] = (f"does not reproduce its recorded price {s.recorded} (bmse {g_sum / BOARD_N:.2f}, "
                                 f"per row {g_sum / len(y):.2f})")
            else:
                passed.append(s)
        checks.append(c)
    rows = []
    for s in passed:
        y, a, b = load_pair(s, rows=s.rows)
        rows.append({"version": s.version, **gain_split(y, a, b), "D": board_delta(s.base_rmse, s.new_rmse),
                     "note": s.note})
    return checks, rows


def summarise(rows, key: str = "B") -> dict:
    x = [r[key] for r in rows]
    d = [r["D"] for r in rows]
    if len(rows) < 3:
        return {"n": len(rows)}
    return {"n": len(rows), "spearman": spearman(x, d), "pearson": pearson(x, d), "sign": sign_agreement(x, d),
            "ratio": ratio_through_origin(x, d)}


# ---------------------------------------------------------------------------------------------- the registered set

R12_SCREEN = "data/r12/stack_801f9bcac9a5_screen/oof_r12.parquet"
R12_FULL = "data/r12/stack_801f9bcac9a5_full/oof_r12.parquet"
FIVE = ("EDDF", "EDDM", "EHAM", "LEBL", "LSZH")
CELLS15 = tuple((a, m) for a in FIVE + ("LEMD", "LIRF") for m in (1, 7)) + (("EGLL", 1),)


def registered_specs() -> tuple:
    """(primary, secondary) exactly as in the registration's table."""
    primary = [
        Spec(24, 257.5532, 257.4310, Side("reports/mau_test_predictions.parquet", "control"),
             Side("reports/mau_test_predictions.parquet", "candidate"), recorded=94.38),
        Spec(25, 257.4310, 257.2917, Side("reports/mxi_predictions.parquet", "it200"),
             Side("reports/mxi_predictions.parquet", "it1000"), rows={"leg": ("forward",)}, recorded=33.95),
        Spec(26, 257.2917, 256.9698, Side("reports/mxj_predictions.parquet", "inc"),
             Side("reports/mxj_predictions.parquet", "arm"), rows={"leg": ("forward",)}, recorded=67.41),
        Spec(27, 256.9698, 257.2683, Side("reports/agf_forward_predictions.parquet", "control"),
             Side("reports/agf_forward_predictions.parquet", "real"), recorded=69.69),
        Spec(29, 256.9698, 256.8981, Side("data/cache_stand/fold_preds_queue_order_weather.parquet", "treatment", True),
             Side("reports/orl_fold_predictions.parquet", "F_orl", True), key="row", recorded=137.24),
        Spec(34, 254.1149, 254.0762, Side("data/nsc_run/nsc_fold_preds.parquet", "incumbent", True),
             Side("data/nsc_run/nsc_fold_preds.parquet", "nsc", True), recorded=99.31),
        Spec(35, 254.0762, 252.0706, Side(R12_SCREEN, "PIPE_M"), Side(R12_SCREEN, "stacked_v34lanes"),
             rows={"ap": FIVE}, recorded=1155.0, note="reference PIPE_M"),
        Spec(36, 254.0762, 251.3146, Side(R12_FULL, "PIPE_M"), Side(R12_FULL, "stacked_v34lanes"),
             rows={"ap": FIVE}, recorded=1456.2, note="reference PIPE_M; cumulative from v34"),
        Spec(40, 251.3146, 251.1991, Side(R12_FULL, "PIPE_M"), Side(R12_FULL, "stacked_v34lanes"),
             rows={"ap": ("EGLL", "LEMD", "LIRF")}, recorded=289.0, note="reference PIPE_M"),
        Spec(46, 250.8082, 250.7314, Side(R12_FULL, "stacked_v34lanes"),
             Side("data/r12/r13_stack_full/oof_lgbcat.parquet", "stacked_v34lanes"), rows={"cells": CELLS15},
             price_on_all=True, recorded=251.6, note="price on all rows; split on the 15 served cells"),
        Spec(48, 250.7314, 252.1294, Side("logs/hpx_scratch/run/preds.parquet", "pipe"),
             Side("logs/hpx_scratch/run/preds.parquet", "T"), recorded=293.9, note="reference PIPE_M"),
        Spec(49, 250.8082, 253.0326, Side(R12_FULL, "stacked_v34lanes"),
             Side("data/r12/move_stack_51348d3e1cb3_full/oof_r12_move.parquet", "stacked_v34lanes"),
             rows={"cells": CELLS15}, price_on_all=True, recorded=-512.1,
             note="price on all rows; split on the 15 served cells"),
    ]
    secondary = {
        "a_v20_unverified": [Spec(20, 258.1806, 257.6686,
                                  Side("data/cache_stand/fold_preds_queue_order_weather.parquet", "baseline", True),
                                  Side("data/cache_stand/fold_preds_queue_order_weather.parquet", "treatment", True),
                                  recorded=None, note="unverified: priced inside a 0.5 CatBoost blend")],
        "d_screen_legs": [
            Spec(25, 257.4310, 257.2917, Side("reports/mxi_predictions.parquet", "it200"),
                 Side("reports/mxi_predictions.parquet", "it1000"), rows={"leg": ("screen",)}, recorded=32.75),
            Spec(26, 257.2917, 256.9698, Side("reports/mxj_predictions.parquet", "inc"),
                 Side("reports/mxj_predictions.parquet", "arm"), rows={"leg": ("screen",)}, recorded=41.45)],
    }
    return primary, secondary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    primary, secondary = registered_specs()
    checks, rows = evaluate(primary)                     # checks decided before any split (see evaluate)
    keys = ("B", "G", "T", "G_minus_top")
    summary = {k: summarise(rows, k) for k in keys}
    s = summary["B"]
    report = {"registration": "plans/PREREG_rul_ruler_validation_2026_10_04.md", "checks": checks, "rows": rows,
              "summary": summary,
              "verdict": verdict(s["spearman"], s["sign"], s["ratio"]) if s.get("n", 0) >= 3 else "NOT ESTABLISHED",
              "secondary": {}}
    pipe_m = {35, 36, 40, 48}
    report["secondary"]["b_without_pipe_m"] = {k: summarise([r for r in rows if r["version"] not in pipe_m], k)
                                                for k in keys}
    report["secondary"]["c_without_v36"] = {k: summarise([r for r in rows if r["version"] != 36], k) for k in keys}
    for name, specs in secondary.items():
        c2, r2 = evaluate(specs)
        report["secondary"][name] = {"checks": c2, "rows": r2}
    out = pathlib.Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        raise SystemExit(f"{out} exists; a result is never overwritten")
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({"verdict": report["verdict"], "summary_B": s,
                      "excluded": [c["version"] for c in checks if c["excluded"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
