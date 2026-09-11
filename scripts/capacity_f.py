"""`scripts/capacity_f.py` -- arm CAP of plans/PREREG_capacity_f_2026_09_10.md.

    python scripts/capacity_f.py fit [--smoke]     # heavy: arm F's design refitted at setting 127,20,0.6
    python scripts/capacity_f.py score             # light: the five registered clauses against arm F's record

`fit` is arm F exactly -- FEATS + QUEUE + ORDER (83 columns), L.P, A2 (no per-airport models), seeds 0,1,2,
best_iter re-found through lgbm_fold.fit_arm -- with num_leaves / min_data_in_leaf / feature_fraction replaced by
the registered setting (lgbm_fold.setting_params). It writes `data/cache_stand/capacity_f_preds.parquet` (the
fold's base columns + the seed-mean delta `cap` + `cap_seed{s}`) and `reports/capacity_f_fit.json`.

`score` joins that record with arm F's (`fold_preds_queue_order.parquet`) on `row` (labels must match; F must
recover 222.5632), and writes `reports/capacity_f.json`. DELTA convention on both sides (reports/bug_classes.md
BC-2): taxi time = max(proxy - delta, 1).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SETTING = (127, 20, 0.6)
SEEDS = (0, 1, 2)
F_RECORD = ROOT / "data" / "cache_stand" / "fold_preds_queue_order.parquet"
F_RMSE = 222.5632
PREDS = ROOT / "data" / "cache_stand" / "capacity_f_preds.parquet"
FIT_JSON = ROOT / "reports" / "capacity_f_fit.json"
SCORE_JSON = ROOT / "reports" / "capacity_f.json"
W_MATCHED, N_FOLD = 0.98466, 339_015
MONTHS = (1, 7)
MIN_AIRPORTS = 7
CUT_TOL_S = 1.0
FILL_TOL_S = 60.0
TAIL_S = 600.0


def taxi(proxy, delta) -> np.ndarray:
    return np.maximum(np.asarray(proxy, dtype="float64") - np.asarray(delta, dtype="float64"), 1.0)


def rmse(e, mask=None) -> float:
    e = np.asarray(e, dtype="float64")
    return float(np.sqrt(e.mean() if mask is None else e[mask].mean()))


def align(f: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """Arm F's record joined with CAP's on `row`; refuses a different row set or different labels."""
    need_f = {"row", "month", "ap", "y", "proxy", "sp", "delta", "treatment"} | {f"delta_hat_seed{s}" for s in SEEDS}
    need_c = {"row", "y", "cap"} | {f"cap_seed{s}" for s in SEEDS}
    for name, frame, need in (("F", f, need_f), ("CAP", c, need_c)):
        miss = sorted(need - set(frame.columns))
        if miss:
            raise ValueError(f"{name} record lacks columns {miss}")
    m = f.merge(c[sorted(need_c)], on="row", suffixes=("", "_c"), validate="one_to_one")
    if len(m) != len(f) or len(m) != len(c):
        raise AssertionError(f"row sets differ: F {len(f):,}, CAP {len(c):,}, joined {len(m):,}")
    if not np.array_equal(m.y.to_numpy(dtype="float64"), m.y_c.to_numpy(dtype="float64")):
        raise AssertionError("labels differ between arm F's record and CAP's on `row`")
    return m


def clauses(m: pd.DataFrame, dates, n_boot: int = 2_000, seed: int = 0) -> dict:
    """The five registered clauses on the aligned frame (DELTA convention both sides). C1 is the paired
    calendar-date block bootstrap (Amendment CAP.1); the row bootstrap is reported, not decisional."""
    if dates is None or len(dates) != len(m):
        raise ValueError("C1 needs one take-off date per row (Amendment CAP.1)")
    import lgbm_fold as lf
    y, proxy, sp = (m[c].to_numpy(dtype="float64") for c in ("y", "proxy", "sp"))
    arms = {"F": taxi(proxy, m.treatment), "CAP": taxi(proxy, m.cap)}
    for s in SEEDS:
        arms[f"F_seed{s}"] = taxi(proxy, m[f"delta_hat_seed{s}"])
        arms[f"CAP_seed{s}"] = taxi(proxy, m[f"cap_seed{s}"])
    se = {a: (v - y) ** 2 for a, v in arms.items()}
    r = {a: rmse(e) for a, e in se.items()}
    gain = r["F"] - r["CAP"]
    boot = lf.paired_bootstrap({"F": se["F"], "CAP": se["CAP"]}, [("CAP", "F")], n_boot, seed)["CAP_vs_F"]
    sd_cap = lf.seed_sd([r[f"CAP_seed{s}"] for s in SEEDS])
    sd_f = lf.seed_sd([r[f"F_seed{s}"] for s in SEEDS])
    month, ap = m.month.to_numpy(), m.ap.astype(str).to_numpy()
    by_month = {int(k): rmse(se["F"], month == k) - rmse(se["CAP"], month == k) for k in MONTHS if (month == k).any()}
    by_ap = {a: rmse(se["F"], ap == a) - rmse(se["CAP"], ap == a) for a in sorted(set(ap))}
    fill = np.abs(y - sp) <= FILL_TOL_S
    tail = np.abs(m.delta.to_numpy(dtype="float64")) > TAIL_S
    for name, k in (("fill", fill), ("tail", tail)):
        if not k.any():
            raise ValueError(f"the decisional {name} cut is empty; C5 cannot be evaluated")
    cuts = {name: {"n": int(k.sum()), "F": rmse(se["F"], k), "CAP": rmse(se["CAP"], k),
                   "loss_s": rmse(se["CAP"], k) - rmse(se["F"], k)} for name, k in (("fill", fill), ("tail", tail),
                                                                                   ("nonfill", ~fill))}
    day_ci = date_block(None, se["F"], se["CAP"], dates, n_boot, seed)
    c = {"C1_interval": bool(day_ci[0] > 0.0),
         "C2_gain_over_2x_seed_sd": bool(gain > 2.0 * sd_cap),
         "C3_both_months": bool(len(by_month) == len(MONTHS) and all(v > 0 for v in by_month.values())),
         "C4_airports": bool(sum(v > 0 for v in by_ap.values()) >= MIN_AIRPORTS),
         "C5_cuts_bounded": bool(cuts["fill"]["loss_s"] <= CUT_TOL_S and cuts["tail"]["loss_s"] <= CUT_TOL_S)}
    verdict = "ESTABLISHED" if all(c.values()) else ("NOT WORKING" if not c["C1_interval"] else "INCONCLUSIVE")
    out = {"n_rows": int(len(m)), "rmse": r, "gain_s": gain, "ci95_day_block": day_ci,
           "ci95_row_bootstrap_reported": boot["ci95"], "seed_sd_cap": sd_cap,
           "seed_sd_f": sd_f, "gain_by_month_s": by_month, "gain_by_airport_s": by_ap,
           "airports_improving": int(sum(v > 0 for v in by_ap.values())), "cuts": cuts,
           "net_weighted_fold_mse": float(W_MATCHED * (se["F"].sum() - se["CAP"].sum()) / N_FOLD),
           "clauses": c, "verdict": verdict}
    return out


def date_block(_d, se_f, se_c, dates, n_boot, seed) -> list:
    """C1 (Amendment CAP.1): calendar-date blocks resampled; gain = RMSE(F) - RMSE(CAP) per draw. Percentile 2.5/97.5."""
    dates = np.asarray(dates).astype(str)
    blocks, code = np.unique(dates, return_inverse=True)
    sf = np.bincount(code, weights=se_f, minlength=len(blocks))
    sc = np.bincount(code, weights=se_c, minlength=len(blocks))
    cnt = np.bincount(code, minlength=len(blocks)).astype("float64")
    rng = np.random.default_rng(seed)
    g = np.empty(n_boot)
    for i in range(n_boot):
        k = np.bincount(rng.integers(0, len(blocks), len(blocks)), minlength=len(blocks)).astype("float64")
        n = (k * cnt).sum()
        g[i] = np.sqrt((k * sf).sum() / n) - np.sqrt((k * sc).sum() / n)
    return [float(v) for v in np.percentile(g, [2.5, 97.5])]


def fit(smoke: bool) -> int:
    import lgbm_fold as lf
    import lgbm_submit as L
    params, nest, patience, months = lf.setting_params(dict(L.P), SETTING), L.NEST, L.PATIENCE, None
    if smoke:
        params["learning_rate"] = lf.SMOKE["learning_rate"]
        nest, patience, months = lf.SMOKE["nest"], lf.SMOKE["patience"], lf.SMOKE["months"]
    out = ROOT / "data" / "smoke_capacity_f" if smoke else None
    preds_path = (out / "capacity_f_preds.parquet") if out else PREDS
    json_path = (out / "capacity_f_fit.json") if out else FIT_JSON
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    if preds_path.exists() and not smoke:
        raise SystemExit(f"refusing to overwrite {preds_path}")
    log = lf._Log(json_path.with_suffix(".log"))
    lf.log = L.log = log
    t0 = time.time()
    if smoke:
        log(L.SMOKE_BANNER)
    feats = lf.FEATS_QUEUE_ORDER
    fold = lf.load_fold(lf.CACHE, months, feats, qcache=lf.QCACHE, ocache=lf.OCACHE, check_counts=not smoke,
                        label="holdout (months (1, 7))")
    log(f"arm CAP: setting {SETTING} on arm F's {len(feats)}-column design; params {params}")
    base, cols = fold["base"].copy(), {}

    def write():
        pd.concat([base, pd.DataFrame(cols, index=base.index)], axis=1).to_parquet(preds_path)

    def on_seed(s, pred):
        cols[f"cap_seed{s}"] = pred
        write()
    arm = lf.fit_arm(fold, fold["X"], params, nest, patience, L.PA_TREE_FLOOR, SEEDS, per_airport=False,
                     pa_trees="es", name="cap", on_seed=on_seed)
    cols["cap"] = arm["delta"]
    write()
    rec = {"prereg": "plans/PREREG_capacity_f_2026_09_10.md", "smoke": smoke, "setting": list(SETTING), "params": params,
           "features": feats, "best_iter": arm["best_iter"], "n_ref": arm["n_ref"],
           "single_seed_rmse_all_holdout": {str(k): v for k, v in arm["single_rmse"].items()},
           "wall_s": round(time.time() - t0, 1), "peak_rss_gb": round(L.peak_rss_gb(), 2), "preds": str(preds_path),
           "written": dt.datetime.now().isoformat(timespec="seconds")}
    json_path.write_text(json.dumps(rec, indent=1, default=float))
    log(f"json -> {json_path}   best_iter {arm['best_iter']:,} n_ref {arm['n_ref']:,}   wall {rec['wall_s']:.0f}s")
    if smoke:
        log(L.SMOKE_BANNER)
    log.close()
    return 0


def fold_dates(rows) -> np.ndarray:
    """Calendar date of each fold row's take-off, through the order cache's row -> MVT_ID map (reported cut only)."""
    import pyarrow.parquet as pq
    import rome_matched as rm
    ids = rm.row_to_mvt_id(ROOT / "data" / "cache_stand", ROOT / "data" / "cache_order")[np.asarray(rows)]
    raw = pd.concat([pq.read_table(p, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).to_pandas()
                     for p in (ROOT / "data" / "raw").glob("training_2025-0[17]-*.parquet")])
    d = raw.drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    got = d.reindex(ids).to_numpy()
    if pd.isna(got).any():
        raise AssertionError(f"{int(pd.isna(got).sum())} fold rows without a take-off date")
    return got


def score() -> int:
    f = pd.read_parquet(F_RECORD)
    r = rmse((taxi(f.proxy, f.treatment) - f.y) ** 2)
    if abs(r - F_RMSE) > 1e-3:
        raise AssertionError(f"arm F's record recovers to {r:.4f}, not {F_RMSE}: wrong record or convention")
    m = align(f, pd.read_parquet(PREDS))
    res = clauses(m, fold_dates(m.row.to_numpy()))
    res.update(prereg="plans/PREREG_capacity_f_2026_09_10.md", setting=list(SETTING),
               written=dt.datetime.now().isoformat(timespec="seconds"))
    SCORE_JSON.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("rmse", "gain_s", "ci95_day_block", "ci95_row_bootstrap_reported", "seed_sd_cap",
                                          "gain_by_month_s", "airports_improving", "cuts", "net_weighted_fold_mse",
                                          "clauses", "verdict")}, indent=1, default=float))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("fit", "score"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    return fit(a.smoke) if a.step == "fit" else score()


if __name__ == "__main__":
    sys.exit(main())
