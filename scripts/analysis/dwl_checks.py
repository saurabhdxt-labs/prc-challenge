"""DWL (plans/PREREG_dwl_lgb_half_on_dwg_2026_09_29.md) — label-free refit acceptance R1 / R2 and the v47-vs-v46 change
report. R3 is the assemble key check inside r13_cat.py assemble (it raises on any id / proxy difference); R4 is
blf_build.py match (v46's own vals must reproduce v46).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/dwl_checks.py r1 --report data/r12/refit_51348d3e1cb3_full/report.json \
        --preds data/r12/refit_51348d3e1cb3_full/predictions_2026.parquet --log logs/r12_scratch/r12_val_run_20260929T193637.log
    /opt/homebrew/bin/python3.11 -B scripts/analysis/dwl_checks.py r2 --new data/r12/refit_51348d3e1cb3_full/predictions_2026.parquet \
        --old data/r12/refit_801f9bcac9a5_full/predictions_2026.parquet
    /opt/homebrew/bin/python3.11 -B scripts/analysis/dwl_checks.py report --new submissions/merry-quicksand_v47.parquet \
        --base submissions/merry-quicksand_v46.parquet --out logs/campaign/v47_change_report.json

Reads no label. Exit 0 = the clause passes.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
N_RANK = 339_551
CAP = 50_000                  # r12_validate FULL tier: "max 50,000 rounds" (the refit log's early-stopping line)
R2_LIMIT_S = 81.6             # 2 x the LightGBM refit-vs-holdout analog 40.8 s (the BLF R2 anchor), registered
TABLE_RE = re.compile(r"rows without an ADS-B table row ([\d,]+)")


def r1(report: dict, preds: pd.DataFrame, log_text, *, n_rank_expected: int = N_RANK, cap: int = CAP,
       arm: str = "r12") -> dict:
    """R1: rc-0 refit (the caller checks rc), best_iter below the cap, finite predictions on every rank row, the
    registered rank-row count, and no rank row without an ADS-B table row (from the run log; `log_text=None` skips only
    that check and records it). Also MOVE's M1 with arm='r12_move', cap=49,800. Raises ValueError on the first defect."""
    if report.get("split") != "refit":
        raise ValueError(f"report split {report.get('split')!r}, expected 'refit'")
    best = int(report["fits"][arm]["best_iter"])
    if best >= cap:
        raise ValueError(f"best_iter {best} reached the cap {cap}")
    if int(report["n_rank"]) != n_rank_expected:
        raise ValueError(f"report n_rank {report['n_rank']}, expected {n_rank_expected}")
    if int(report["n_test_without_table"]) != 0:
        raise ValueError(f"report n_test_without_table {report['n_test_without_table']}, expected 0")
    if len(preds) != n_rank_expected:
        raise ValueError(f"predictions have {len(preds)} rows, expected {n_rank_expected}")
    dup = int(preds.MVT_ID_mvt.duplicated().sum())
    if dup:
        raise ValueError(f"predictions carry {dup} duplicate MVT_ID_mvt")
    bad = int((~np.isfinite(preds[f"pred_{arm}"].to_numpy(float))).sum())
    if bad:
        raise ValueError(f"{bad} non-finite pred_{arm}")
    wo = None
    if log_text is not None:
        m = TABLE_RE.findall(log_text)
        if not m:
            raise ValueError("log has no 'rows without an ADS-B table row' line")
        wo = int(m[-1].replace(",", ""))
        if wo:
            raise ValueError(f"log: {wo} rows without an ADS-B table row")
    return {"arm": arm, "best_iter": best, "cap": cap, "n_rank": int(report["n_rank"]), "rows_without_table": wo,
            "log_checked": log_text is not None, "n_finite": int(len(preds)), "ok": True}


def r2(new: pd.DataFrame, old: pd.DataFrame, *, limit: float = R2_LIMIT_S, arm: str = "r12") -> dict:
    """R2: RMS of the LightGBM delta difference on the 2026 rank rows, aligned by MVT_ID_mvt (same ids, same proxy)."""
    a, b = new.set_index("MVT_ID_mvt"), old.set_index("MVT_ID_mvt")
    only_a, only_b = a.index.difference(b.index), b.index.difference(a.index)
    if len(only_a) or len(only_b):
        raise ValueError(f"id sets differ: {len(only_a)} only in new, {len(only_b)} only in old")
    b = b.loc[a.index]
    npx = int((a.proxy.to_numpy(float) != b.proxy.to_numpy(float)).sum())
    if npx:
        raise ValueError(f"proxy differs on {npx} rows")
    col = f"pred_{arm}"
    d = a[col].to_numpy(float) - b[col].to_numpy(float)
    rms = float(np.sqrt(np.mean(d ** 2)))
    return {"arm": arm, "n": int(len(d)), "rms_s": rms, "limit_s": limit, "ok": bool(rms <= limit),
            "mean_s": float(d.mean()), "p99_abs_s": float(np.quantile(np.abs(d), 0.99))}


def serve_coverage(serve: dict, *, n_expected: int = N_RANK) -> dict:
    """MOVE M3: the stack serve covers every matched rank row and was served from the refit F."""
    if int(serve["n_matched"]) != n_expected:
        raise ValueError(f"serve n_matched {serve['n_matched']}, expected {n_expected}")
    if serve.get("f_source") != "refit":
        raise ValueError(f"serve f_source {serve.get('f_source')!r}, expected 'refit'")
    return {"n_matched": int(serve["n_matched"]), "f_source": serve["f_source"], "ok": True}


def change_report(new: pd.DataFrame, base: pd.DataFrame, cell: pd.Series, *, id_col: str = "MVT_ID_mvt",
                  val_col: str = "TAXITIME_SEC_mvt") -> dict:
    """Rows changed, RMS move and the per-cell (airport|month) mean shift of `new` against `base` (same row order)."""
    if len(new) != len(base) or not np.array_equal(new[id_col].to_numpy(float), base[id_col].to_numpy(float)):
        raise ValueError("row order / ids differ from the base")
    mv = new[val_col].to_numpy(float) - base[val_col].to_numpy(float)
    c = cell.reindex(base[id_col].to_numpy(float)).to_numpy()
    cells = {}
    for k in sorted(pd.unique(c)):
        sel = c == k
        cells[str(k)] = {"n": int(sel.sum()), "n_changed": int((mv[sel] != 0).sum()),
                         "mean_shift_s": float(mv[sel].mean())}
    return {"rows": int(len(mv)), "n_changed": int((mv != 0).sum()), "rms_move_s": float(np.sqrt(np.mean(mv ** 2))),
            "cells": cells}


def _cells() -> pd.Series:  # pragma: no cover - real file
    rk = pd.read_parquet(ROOT / "data" / "raw" / "ranking.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    m = pd.to_datetime(rk.MVT_TIME_UTC_mvt, utc=True).dt.month.astype(str)
    return pd.Series((rk.ADEP_mvt.astype(str) + "|" + m).to_numpy(), index=rk.MVT_ID_mvt.to_numpy(float))


def main(argv=None) -> int:  # pragma: no cover - thin I/O shell around the tested functions
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("r1")
    s.add_argument("--report", required=True)
    s.add_argument("--preds", required=True)
    s.add_argument("--log", default=None, help="the run log; omitted = the table-row check is skipped (recorded)")
    s.add_argument("--arm", default="r12")
    s.add_argument("--cap", type=int, default=CAP)
    s = sub.add_parser("r2")
    s.add_argument("--new", required=True)
    s.add_argument("--old", required=True)
    s.add_argument("--arm", default="r12")
    s = sub.add_parser("serve")
    s.add_argument("--json", required=True)
    s = sub.add_parser("report")
    s.add_argument("--new", required=True)
    s.add_argument("--base", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "r1":
        cols = ["MVT_ID_mvt", f"pred_{a.arm}"]
        r = r1(json.loads(pathlib.Path(a.report).read_text()), pd.read_parquet(a.preds, columns=cols),
               pathlib.Path(a.log).read_text() if a.log else None, cap=a.cap, arm=a.arm)
        print(json.dumps(r))
        return 0
    if a.cmd == "r2":
        cols = ["MVT_ID_mvt", f"pred_{a.arm}", "proxy"]
        r = r2(pd.read_parquet(a.new, columns=cols), pd.read_parquet(a.old, columns=cols), arm=a.arm)
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    if a.cmd == "serve":
        print(json.dumps(serve_coverage(json.loads(pathlib.Path(a.json).read_text()))))
        return 0
    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    r = change_report(pd.read_parquet(a.new), pd.read_parquet(a.base), _cells())
    r.update(new=a.new, base=a.base)
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ("rows", "n_changed", "rms_move_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
