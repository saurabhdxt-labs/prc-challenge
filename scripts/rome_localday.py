"""LD — `dayoff` on the airport's LOCAL date for the Rome lane (plans/PREREG_local_dayoff_2026_09_11.md).

    python3.11 -B scripts/rome_localday.py        -> reports/rome_localday.json

Re-runs v10's LIRF chain end to end on the 2025 frames — stratum_fold S1 (LOMO + fold A) -> rome_fill (R) -> rome_dateslip
(R2) -> E1's floor — twice: on the UNCHANGED frames (must reproduce the stored R2 record on every row: the guard) and on
frames whose `dayoff` is computed on each airport's local date (the arm). Only that column differs.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CACHE = ROOT / "data" / "cache_stand"
OUT = ROOT / "reports" / "rome_localday.json"
STORED_R2 = CACHE / "rome_dateslip_preds.parquet"
E1_LO, E1_HI = 24_000.0, 86_400.0
N_SCORED = 344_841
C2_BAR = 300.0
GUARD_TOL = 1e-6


# ------------------------------------------------------------------ pure pieces (unit-tested)

def local_dayoff(mvt_utc, sched_utc, adep, tz_of: dict) -> np.ndarray:
    """clip(local date(MVT) − local date(SCHED), 0, 1) in days, each row in its airport's zone. Refuses an airport
    without a zone and a timestamp that is not tz-aware UTC."""
    mvt, sch = pd.Series(pd.to_datetime(mvt_utc)), pd.Series(pd.to_datetime(sched_utc))
    for s, name in ((mvt, "MVT"), (sch, "SCHED")):
        tz = getattr(s.dt, "tz", None)
        if tz is None or str(tz) != "UTC":
            raise ValueError(f"{name} times must be tz-aware UTC, got {tz}")
    adep = pd.Series(np.asarray(adep).astype(str))
    missing = sorted(set(adep) - set(tz_of))
    if missing:
        raise ValueError(f"no time zone for airports {missing}")
    out = np.empty(len(adep), dtype="int64")
    for a in adep.unique():
        m = (adep == a).to_numpy()
        lm = mvt[m].dt.tz_convert(tz_of[a]).dt.tz_localize(None).dt.normalize()
        ls = sch[m].dt.tz_convert(tz_of[a]).dt.tz_localize(None).dt.normalize()
        out[m] = np.clip((lm - ls).dt.days.to_numpy(), 0, 1)
    return out


def localize(frame: pd.DataFrame, tz_of: dict) -> pd.DataFrame:
    """A copy whose `dayoff` is the local-date version; every other column untouched."""
    f = frame.copy()
    f["dayoff"] = local_dayoff(f.MVT_TIME_UTC_mvt, f.SCHED_TIME_UTC_mvt, f.ADEP_mvt, tz_of).astype(frame.dayoff.dtype)
    return f


def e1_floor(pred, sp) -> np.ndarray:
    """rome_bandfloor's rule on rounded values: max(rint(v), rint(sp)) on LO <= sp < HI (every row here is LIRF)."""
    v, sp = np.rint(np.asarray(pred, dtype="float64")), np.asarray(sp, dtype="float64")
    rule = (sp >= E1_LO) & (sp < E1_HI)
    return np.where(rule, np.maximum(v, np.rint(sp)), v)


def price_2026(g, cell, cell_2026, n_scored: int = N_SCORED) -> dict:
    """Σ_cells mean(g | cell, 2025) × n_2026(cell) / n_scored; cells absent from 2025 add 0 and are listed."""
    g = pd.Series(np.asarray(g, dtype="float64"))
    cell = pd.Series(np.asarray(cell))
    mean_g = g.groupby(cell.to_numpy()).mean()
    n26 = pd.Series(np.asarray(cell_2026)).value_counts()
    contrib = {str(c): float(mean_g.get(c, 0.0) * n / n_scored) for c, n in n26.items()}
    return {"total": float(sum(contrib.values())), "by_cell": contrib,
            "cells_without_2025_rows": sorted(str(c) for c in n26.index if c not in mean_g.index)}


def cell_of(utc_dayoff, local_dayoff_) -> np.ndarray:
    return np.char.add(np.char.add("utc", np.asarray(utc_dayoff).astype(int).astype(str)),
                       np.char.add("_local", np.asarray(local_dayoff_).astype(int).astype(str)))


# ------------------------------------------------------------------ the chain (real data)

def chain(unm: pd.DataFrame, lirf: pd.DataFrame, log) -> pd.DataFrame:
    import rome_dateslip as rds
    import rome_fill as rf
    import stratum_fold as sf
    stored = pd.concat([sf.score_lomo(unm, lirf, sf.SEEDS, log), sf.score_fold_a(unm, lirf, sf.SEEDS, log)],
                       ignore_index=True)
    r = rf.run(unm, lirf, stored, seeds=rf.SEEDS, n_boot=200, log=log)
    d = rds.run(r["preds"], unm, n_boot=200)
    p = d["preds"].copy()
    p["R2E1"] = e1_floor(p.R2, p.sp)
    return p


def main() -> int:
    import rome_fill as rf
    from prc.pipeline import config as C, ingest
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    cfg = C.load_config(ROOT / "configs" / "pipeline.yaml")
    tz_of = {code: a.tz for code, a in cfg.airports.items()}
    unm, lirf = rf.load_frames(log)
    base = chain(unm, lirf, log)
    ref = pd.read_parquet(STORED_R2, columns=["MVT_ID_mvt", "fold", "R2"])
    chk = base.merge(ref, on=["MVT_ID_mvt", "fold"], suffixes=("", "_stored"), validate="one_to_one")
    if len(chk) != len(ref) or len(chk) != len(base):
        raise AssertionError(f"row sets differ: chain {len(base)}, stored {len(ref)}, joined {len(chk)}")
    err = float(np.abs(chk.R2 - chk.R2_stored).max())
    if err > GUARD_TOL:
        raise AssertionError(f"GUARD: the UTC chain does not reproduce the stored R2 (max |Δ| {err}); nothing is scored")
    log(f"guard: UTC chain reproduces the stored R2 on {len(chk):,} rows (max |Δ| {err})")
    arm = chain(localize(unm, tz_of), localize(lirf, tz_of), log)
    j = base.merge(arm[["MVT_ID_mvt", "fold", "R2", "R2E1", "dayoff"]], on=["MVT_ID_mvt", "fold"], suffixes=("_utc", "_ld"),
                   validate="one_to_one")
    y = j.y.to_numpy(dtype="float64")
    j["g"] = (y - j.R2E1_utc) ** 2 - (y - j.R2E1_ld) ** 2
    j["cell"] = cell_of(j.dayoff_utc, j.dayoff_ld)
    lomo, fa = j[j.fold == "lomo"], j[j.fold == "A"]
    bm_u = ((lomo.y - lomo.R2E1_utc) ** 2).groupby(lomo.month).sum()
    bm_l = ((lomo.y - lomo.R2E1_ld) ** 2).groupby(lomo.month).sum()
    boot = rf.month_block_bootstrap(bm_u.to_numpy(), bm_l.to_numpy(), 2000, 0)
    sd = ingest.ingest_scored(cfg)
    s = sd.scored[(sd.scored.ADEP_mvt == "LIRF") & sd.scored.unmatched.to_numpy(dtype=bool)]
    c26 = cell_of(s.dayoff.to_numpy(), local_dayoff(s.MVT_TIME_UTC_mvt, s.SCHED_TIME_UTC_mvt, s.ADEP_mvt, tz_of))
    price = price_2026(lomo.g, lomo.cell, c26)
    clauses = {"C1_month_block_lower_gt_0": bool(boot["ci95"][0] > 0), "C2_2026_price_ge_300": bool(price["total"] >= C2_BAR),
               "C3_foldA_sign": bool(fa.g.sum() > 0)}
    verdict = "WORKING" if all(clauses.values()) else ("NOT WORKING" if not clauses["C1_month_block_lower_gt_0"] else "INCONCLUSIVE")
    top = lomo.reindex(lomo.g.abs().sort_values(ascending=False).index)[:10]
    res = {"tag": "LD", "prereg": "plans/PREREG_local_dayoff_2026_09_11.md",
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "guard_max_abs_diff": err, "n_rows": {"lomo": int(len(lomo)), "A": int(len(fa))},
           "rmse_lomo": {"utc": float(np.sqrt(((lomo.y - lomo.R2E1_utc) ** 2).mean())),
                         "local": float(np.sqrt(((lomo.y - lomo.R2E1_ld) ** 2).mean()))},
           "sigma_g": {"lomo": float(lomo.g.sum()), "A": float(fa.g.sum())}, "month_block_lomo": boot,
           "by_cell_lomo": {c: {"n": int((lomo.cell == c).sum()), "sigma_g": float(lomo.g[lomo.cell == c].sum())}
                            for c in sorted(set(lomo.cell))},
           "by_month_lomo": {int(m): float(v) for m, v in lomo.g.groupby(lomo.month).sum().items()},
           "price_2026": price, "top10_rows_lomo": top[["MVT_ID_mvt", "month", "sp", "y", "R2E1_utc", "R2E1_ld", "g", "cell"]]
           .to_dict(orient="records"), "cells_2026": pd.Series(c26).value_counts().to_dict(),
           "clauses": clauses, "verdict": verdict, "wall_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("rmse_lomo", "sigma_g", "price_2026", "by_cell_lomo", "clauses", "verdict")}, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
