"""LCM — error map of v13's SHIPPED LIRF unmatched chain (plans/PREREG_lirf_chain_map_2026_09_12.md; descriptive).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/lirf_chain_map.py --out reports/lirf_chain_map_2026_09_12.json

Composes the stored out-of-fold R2 record with the shipped rules E1 (rome_localday.e1_floor) and RLD
(rome_local_day.apply_rld); edits none of them. Decomposes squared error by label class, by serve-time cell, and by row
concentration; prices per-row-target oracles; projects to 2026 by serve-time cell. Claims no gain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import rome_local_day as RLD  # noqa: E402
import rome_localday as LD  # noqa: E402
import unm_diag as U  # noqa: E402

R2_PREDS = ROOT / "data" / "cache_stand" / "rome_dateslip_preds.parquet"
UNM = ROOT / "data" / "cache_rome" / "unm.parquet"      # AMENDMENT 1: LIRF's UNMATCHED rows (lirf.parquet is matched only)
RANKING = ROOT / "data" / "raw" / "ranking.parquet"
N_SCORED = 344_841
WANT_RMSE_R2E1 = 3916.615967013426
WANT_N_LOMO = 1_488
WANT_N_2026 = 383
DAY_S = 86_400.0


def cell(sp, local_dayoff) -> np.ndarray:
    """Serve-time cell: sp band × (Rome local dayoff 0 | >=1)."""
    d = np.asarray(local_dayoff)
    return np.char.add(U.sp_band(sp).astype(str), np.where(d >= 1, "|next", "|same"))


def reconcile(parts: dict, total: float, tol: float = 1e-6) -> None:
    s = float(sum(parts.values()))
    if abs(s - total) > tol * max(1.0, abs(total)):
        raise AssertionError(f"decomposition does not reconcile: {s} vs {total}")


def slip_target(sp) -> tuple[np.ndarray, np.ndarray]:
    """The two per-row candidates for a 24 h slip: rint(sp) and rint(sp) + 86,400."""
    sp = np.rint(np.asarray(sp, dtype="float64"))
    return sp, sp + DAY_S


def oracle_saving(y, pred, target, mask) -> float:
    """SSE saved by replacing pred with target on mask (can be negative if the target is worse)."""
    y, pred, target, mask = (np.asarray(y, float), np.asarray(pred, float), np.asarray(target, float),
                             np.asarray(mask, bool))
    return float((((y - pred) ** 2) - ((y - target) ** 2))[mask].sum())


def project_2026(se, cells_2025, cells_2026: dict, n_scored: int = N_SCORED) -> dict:
    df = pd.DataFrame({"se": np.asarray(se, float), "c": np.asarray(cells_2025)})
    mse = df.groupby("c").se.mean()
    by, unsupported = {}, []
    for c, n in cells_2026.items():
        if c in mse.index:
            by[c] = float(mse[c] * n / n_scored)
        else:
            unsupported.append(c)
    return {"total": float(sum(by.values())), "by_cell": by, "cells_without_2025_rows": sorted(unsupported)}


def load_2025() -> pd.DataFrame:
    p = pd.read_parquet(R2_PREDS)
    l = pd.read_parquet(UNM, columns=["MVT_ID_mvt", "ADEP_mvt", "unmatched", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "y"])
    return join_frames(p, l)


def join_frames(p: pd.DataFrame, l: pd.DataFrame) -> pd.DataFrame:
    """LOMO rows of the stored R2 record, with SCHED / MVT times from the LIRF UNMATCHED frame rows."""
    p = p[p.fold == "lomo"].reset_index(drop=True)
    l = l[l.unmatched.astype(bool) & (l.ADEP_mvt == "LIRF")].drop(columns=["ADEP_mvt"])
    if l.MVT_ID_mvt.duplicated().any():
        raise AssertionError("duplicate MVT_ID_mvt among LIRF unmatched frame rows")
    m = p.merge(l, on="MVT_ID_mvt", how="left", suffixes=("", "_l"), validate="one_to_one")
    if m.SCHED_TIME_UTC_mvt.isna().any():
        raise AssertionError(f"{int(m.SCHED_TIME_UTC_mvt.isna().sum())} R2 rows have no LIRF frame row")
    if not np.array_equal(m.y.to_numpy(float), m.y_l.to_numpy(float)):
        raise AssertionError("labels differ between the R2 record and the LIRF frame")
    return m


def load_2026() -> pd.DataFrame:
    r = pd.read_parquet(RANKING, columns=["ADEP_mvt", "PHASE_mvt", "AOBT_3_flt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"])
    return r[(r.PHASE_mvt == "DEP") & (r.ADEP_mvt == "LIRF") & r.AOBT_3_flt.isna()].reset_index(drop=True)


def sp_of(fr) -> np.ndarray:
    return (pd.to_datetime(fr.MVT_TIME_UTC_mvt, utc=True) - pd.to_datetime(fr.SCHED_TIME_UTC_mvt, utc=True)) \
        .dt.total_seconds().to_numpy()


def run(m: pd.DataFrame, r26: pd.DataFrame, want_rmse: float = WANT_RMSE_R2E1, want_n: int = WANT_N_LOMO,
        want_n26: int = WANT_N_2026) -> dict:
    if len(m) != want_n:
        raise AssertionError(f"expected {want_n} LOMO rows, got {len(m)}")
    if len(r26) != want_n26:
        raise AssertionError(f"expected {want_n26} scored 2026 LIRF unmatched rows, got {len(r26)}")
    y, sp = m.y.to_numpy(float), m.sp.to_numpy(float)
    if not np.allclose(sp, sp_of(m), atol=1e-6, rtol=0):
        raise AssertionError("stored sp is not MVT - SCHED (convention mismatch)")
    e1 = LD.e1_floor(m.R2.to_numpy(float), sp)
    got = float(np.sqrt(np.mean((y - e1) ** 2)))
    if abs(got - want_rmse) > 1e-6:
        raise AssertionError(f"guard R2E1 LOMO RMSE {got:.9f} vs recorded {want_rmse:.9f}")
    fr = pd.DataFrame({"MVT_ID_mvt": m.MVT_ID_mvt, "ADEP_mvt": "LIRF", "unmatched": True, "base": e1,
                       "SCHED_TIME_UTC_mvt": m.SCHED_TIME_UTC_mvt, "MVT_TIME_UTC_mvt": m.MVT_TIME_UTC_mvt})
    chain, rld_changed = RLD.apply_rld(fr)
    chain = chain.astype(float)
    se = (y - chain) ** 2
    total = float(se.sum())
    cls = U.row_class(y, sp)
    ld = RLD.local_dayoff(m.SCHED_TIME_UTC_mvt, m.MVT_TIME_UTC_mvt)
    cl = cell(sp, ld)
    by_class = {c: float(se[cls == c].sum()) for c in sorted(set(cls.tolist()))}
    reconcile(by_class, total)
    by_cell = {c: float(se[cl == c].sum()) for c in sorted(set(cl.tolist()))}
    reconcile(by_cell, total)
    order = np.sort(se)[::-1]
    s_lo, s_hi = slip_target(sp)
    slip = cls == "dateslip"
    rows = pd.DataFrame({"MVT_ID_mvt": m.MVT_ID_mvt, "month": m.month, "sp": sp, "y": y, "chain": chain, "se": se,
                         "class": cls, "cell": cl, "local_dayoff": ld})
    sp26 = sp_of(r26)
    c26 = pd.Series(cell(sp26, RLD.local_dayoff(r26.SCHED_TIME_UTC_mvt, r26.MVT_TIME_UTC_mvt))).value_counts().to_dict()
    lomo_board_on_2026_rows = total / len(m) * want_n26 / N_SCORED
    return {
        "tag": "LCM", "prereg": "plans/PREREG_lirf_chain_map_2026_09_12.md", "n_lomo": int(len(m)),
        "guard_R2E1_rmse": got, "rld_rows_changed_2025": int(rld_changed.sum()),
        "rmse_chain": float(np.sqrt(se.mean())), "sse_total": total,
        "naive_2026_board_mse_all_rows_mean": lomo_board_on_2026_rows,
        "by_class": {c: {"n": int((cls == c).sum()), "sse": v, "share": v / total,
                         "mean_err": float((y - chain)[cls == c].mean())} for c, v in by_class.items()},
        "by_cell": {c: {"n": int((cl == c).sum()), "sse": v, "share": v / total} for c, v in by_cell.items()},
        "top_k_share": {k: float(order[:k].sum() / total) for k in (1, 5, 10, 22)},
        "oracle_fill_sp_sse_saved": oracle_saving(y, chain, np.rint(sp), cls == "fill"),
        "oracle_slip_rint_sp_sse_saved": oracle_saving(y, chain, s_lo, slip),
        "oracle_slip_sp_plus_day_sse_saved": oracle_saving(y, chain, s_hi, slip),
        "cells_2026": {k: int(v) for k, v in c26.items()},
        "projection_2026_by_cell": project_2026(se, cl, c26),
        "top_rows": rows.sort_values("se", ascending=False).head(25).to_dict("records"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    res = run(load_2025(), load_2026())
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("n_lomo", "rmse_chain", "naive_2026_board_mse_all_rows_mean", "by_class",
                                          "top_k_share")}, default=float, indent=1))
    print(json.dumps(res["projection_2026_by_cell"], default=float, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
