"""NSC — the shipped FW fold recipe minus the three label-seconds columns (plans/PREREG_nsc_no_seconds_2026_09_24.md).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/nsc_fold.py --smoke   # months 1-3, tiny trees, no lock, < 5 min
    /opt/homebrew/bin/python3.11 -B logs/nsc_scratch/launch_nsc.py         # the real run, detached (~2.3 h, ~4.6 GB)

What it measures. `stand_ab.BASE` carries `aobt_sec, mvt_sec, sched_sec` (the `.dt.second` of AOBT_3 / MVT / SCHED), so the
shipped FW boosters (`lgbm_fold.FEATS_QUEUE_ORDER_WEATHER`, 93 columns, indices 14-16) read the label-seconds channel that
the project's own ruling refuses (MVT and BLOCK seconds are coupled by the organiser's jitter). NSC refits fold A with the
IDENTICAL recipe - `lgbm_fold.load_fold` (holdout months 1 and 7, early stopping on 3 and 9 carved from training, in-fold
encodings), `lgbm_fold.fit_arm` (early stopping seed 0, three seeds 0, 1, 2 refitted at n_ref, the pooled mean, no
per-airport blend: the FW run's baseline arm was A2) and the production parameters `lgbm_submit.P` - on the 90 columns
left after dropping the three, and pairs it row for row against the FW run's OWN fold predictions
(data/cache_stand/fold_preds_queue_order_weather.parquet, column `treatment`, 48,391.5 board MSE). It never reads the
66-feature cache as a baseline (memory feedback_prc_66_feature_cache_overstates).

Units: cost = (SSE_nsc - SSE_fw) / 344,841 (board MSE; positive = removing the seconds costs). Interval: paired
bootstrap, 2,000 draws, seed 0, by whole (month, day-of-year) days and by rows. The decision rule is the registration's.

Imports lgbm_fold (LightGBM) only: no torch, no CatBoost in this process (bug class BC-5).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LF = _load("lgbm_fold", "scripts/lgbm_fold.py")

PREREG = ROOT / "plans" / "PREREG_nsc_no_seconds_2026_09_24.md"
HASHLOG = ROOT / "reports" / "nsc_prereg_hashes.log"
RECORD = ROOT / "data" / "cache_stand" / "fold_preds_queue_order_weather.parquet"
RECORD_SHA = "1cfd1596ccfa05a73d7a527c4bcebf4703b9871f963c6c4128b3fb57e8692010"
OUT_DIR = ROOT / "data" / "nsc_run"
REPORT_NAME, PREDS_NAME, LOG_NAME = "nsc_fold_report.json", "nsc_fold_preds.parquet", "nsc_fold.log"
N_BOARD = 344_841
N_BOOT, BOOT_SEED = 2_000, 0
BOOT_CELLS = 2_000_000          # index cells per bootstrap chunk: 16 MB of int64 (+ 16 MB gathered)
SEEDS = (0, 1, 2)
RSS_LIMIT_BYTES = 6 * 1024 ** 3

#: the label-seconds channel (stand_ab.py:198-200)
SECONDS_COLS = ("aobt_sec", "mvt_sec", "sched_sec")
#: registered bars (plans/PREREG_nsc_no_seconds_2026_09_24.md "Decision rule")
BOUND_BMSE = 337.0          # explainable sub-minute variance of y on fold A, (ap, mvt_sec) cells: 343.0 s^2 x 339,015/344,841
ESCALATE_LOWER_BMSE = 150.0
MDE_Z = 2.8                 # 80 % power, two-sided alpha 0.05

#: v33's production lists that hold a seconds column, frozen so v33 stays reproducible. A NEW list with a seconds column
#: fails tests/test_nsc_fold.py; v34 retires these by switching its consumers to the FEATS_NSC_* lists below.
LEGACY_SECONDS_LISTS = (
    "lgbm_fold.FEATS_QUEUE", "lgbm_fold.FEATS_HEAD", "lgbm_fold.FEATS_DAY", "lgbm_fold.FEATS_QUEUE_DAY",
    "lgbm_fold.FEATS_ORDER", "lgbm_fold.FEATS_QUEUE_ORDER", "lgbm_fold.FEATS_WEATHER", "lgbm_fold.FEATS_QUEUE_WEATHER",
    "lgbm_fold.FEATS_QUEUE_ORDER_WEATHER", "lgbm_submit.FEATS", "lgbm_submit.UNMATCHED_COLS", "stand_ab.BASE",
    "stand_ab.BASELINE_FEATS", "stand_ab.VARIANT_FEATS", "stand_ab.UNMATCHED_NAN_COLS", "stand_ab.UNMATCHED_COLS",
    "stand_ab.UNMATCHED_FLOAT32",
)
#: sources of the production unmatched lane and Rome chain: no seconds column may appear in them at all (AST check)
NO_SECONDS_SOURCES = ("scripts/build_submission.py", "scripts/unm_congestion.py", "scripts/rome_fill.py",
                      "scripts/rome_local_day.py", "scripts/rome_dateslip.py", "scripts/analysis/mau_build.py")
#: names in THIS module that hold the seconds names but are not feature lists
_NOT_FEATURE_LISTS = {"nsc_fold.SECONDS_COLS"}


def drop_seconds(feats) -> list:
    """`feats` without the three seconds columns, order kept. Refuses a list that does not hold all three (a silent no-op
    on the wrong list is the failure this guards) and a list with duplicates."""
    feats = list(feats)
    if len(set(feats)) != len(feats):
        raise ValueError("feature list has duplicate names")
    missing = [c for c in SECONDS_COLS if c not in feats]
    if missing:
        raise ValueError(f"feature list does not hold all three seconds columns: missing {missing}")
    return [c for c in feats if c not in SECONDS_COLS]


FEATS_NSC_FW = drop_seconds(LF.FEATS_QUEUE_ORDER_WEATHER)     # 90: the ablation's design, and v34's FW
FEATS_NSC_F = drop_seconds(LF.FEATS_QUEUE_ORDER)              # 80: v34's F (ADS-B stage base)


def nsc_catboost_numeric() -> list:
    """v34's CatBoost numeric columns: catboost_native.numeric_columns() minus the three (56)."""
    import catboost_native as cn
    return drop_seconds(cn.numeric_columns())


def production_modules() -> dict:
    """The modules whose module-level lists define the production matched designs (as lgbm_fold loads them)."""
    import types
    return {"lgbm_fold": LF, "lgbm_submit": LF.L, "stand_ab": LF.S, "nsc_fold": types.SimpleNamespace(**globals())}


def lists_with_seconds(modules: dict) -> set:
    """'module.NAME' for every module-level list/tuple holding a seconds column name."""
    out = set()
    for mname, mod in modules.items():
        for k, v in vars(mod).items():
            if isinstance(v, (list, tuple)) and any(isinstance(x, str) and x in SECONDS_COLS for x in v):
                name = f"{mname}.{k}"
                if name not in _NOT_FEATURE_LISTS:
                    out.add(name)
    return out


# ---------------------------------------------------------------- scoring (pure)

def _taxi(proxy, delta) -> np.ndarray:
    return np.maximum(np.asarray(proxy, float) - np.asarray(delta, float), 1.0)


def _boot(g: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> list:
    """95 % percentile interval of sum(g)/N_BOARD, resampling whole groups with replacement."""
    uniq, inv = np.unique(groups, return_inverse=True)
    per = np.bincount(inv, weights=g, minlength=len(uniq))
    k = len(uniq)
    rng = np.random.default_rng(seed)
    # chunked: an (n_boot x k) index matrix is 5.4 GB at k = 339,015 (the 2026-09-24 smoke peaked at 7.1 GB)
    step = max(1, BOOT_CELLS // k)
    draws = np.concatenate([per[rng.integers(0, k, size=(min(step, n_boot - i), k))].sum(axis=1)
                            for i in range(0, n_boot, step)]) / N_BOARD
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def pair_cost(y, proxy, delta_inc, delta_nsc, day, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Paired cost of NSC vs the incumbent in board MSE; day-block and row-bootstrap intervals."""
    arrs = [np.asarray(a, float) for a in (y, proxy, delta_inc, delta_nsc)]
    day = np.asarray(day)
    if len({len(a) for a in arrs} | {len(day)}) != 1:
        raise ValueError(f"length mismatch: {[len(a) for a in arrs] + [len(day)]}")
    if not all(np.isfinite(a).all() for a in arrs):
        raise ValueError("non-finite input to pair_cost")
    y, proxy, inc, nsc = arrs
    se_inc = (y - _taxi(proxy, inc)) ** 2
    se_nsc = (y - _taxi(proxy, nsc)) ** 2
    g = se_nsc - se_inc
    return {"n": int(len(y)), "sse_inc": float(se_inc.sum()), "sse_nsc": float(se_nsc.sum()),
            "cost_bmse": float(g.sum() / N_BOARD), "inc_bmse": float(se_inc.sum() / N_BOARD),
            "nsc_bmse": float(se_nsc.sum() / N_BOARD),
            "ci95_day": _boot(g, day, n_boot, seed), "ci95_row": _boot(g, np.arange(len(g)), n_boot, seed),
            "n_days": int(len(np.unique(day)))}


def check_alignment(base: pd.DataFrame, record: pd.DataFrame) -> None:
    """The fold must be the record's fold: same rows in the same order, identical row/month/ap/y/proxy/delta."""
    if len(base) != len(record):
        raise AssertionError(f"fold has {len(base):,} holdout rows, the record {len(record):,}")
    for c in ("row", "month", "ap", "y", "proxy", "delta"):
        a, b = base[c].to_numpy(), record[c].to_numpy()
        same = (a.astype(str) == b.astype(str)).all() if c == "ap" else np.array_equal(a, b)
        if not same:
            raise AssertionError(f"fold and record differ in column {c}")


def mde_from_ci(ci) -> float:
    """2.8 standard errors, the SE read off a 95 % interval as its width / 3.92."""
    return MDE_Z * (float(ci[1]) - float(ci[0])) / 3.92


def decide(cost: float, ci_lower: float) -> dict:
    """The registered rule: ESCALATE to the owner if cost >= 337 (above the sub-minute channel's own bound) or the
    day-block lower bound >= 150; otherwise PROCEED to the v34 retrain without the seconds, whatever the sign."""
    if not (math.isfinite(cost) and math.isfinite(ci_lower)):
        raise ValueError("decide needs a finite cost and lower bound")
    reasons = []
    if cost >= BOUND_BMSE:
        reasons.append(f"cost {cost:+.1f} >= bound {BOUND_BMSE}")
    if ci_lower >= ESCALATE_LOWER_BMSE:
        reasons.append(f"day-block lower bound {ci_lower:+.1f} >= {ESCALATE_LOWER_BMSE}")
    return {"decision": "ESCALATE" if reasons else "PROCEED", "reasons": reasons,
            "bound_bmse": BOUND_BMSE, "escalate_lower_bmse": ESCALATE_LOWER_BMSE}


# ---------------------------------------------------------------- the run

def _load_real(feats, smoke: bool) -> dict:
    months = LF.SMOKE["months"] if smoke else None
    return LF.load_fold(LF.CACHE, months, list(feats), qcache=LF.QCACHE, ocache=LF.OCACHE, wcache=LF.WCACHE,
                        check_counts=not smoke, label=f"holdout (months {LF.HOLDOUT})")


def _fit_cfg(smoke: bool) -> tuple:
    params, nest, patience, pa_floor = dict(LF.L.P), LF.L.NEST, LF.L.PATIENCE, LF.L.PA_TREE_FLOOR
    if smoke:
        params["learning_rate"] = LF.SMOKE["learning_rate"]
        nest, patience, pa_floor = LF.SMOKE["nest"], LF.SMOKE["patience"], LF.SMOKE["pa_floor"]
    return params, nest, patience, pa_floor


def _fit(fit, fold, smoke, name):
    params, nest, patience, pa_floor = _fit_cfg(smoke)
    return fit(fold, fold["X"], params, nest, patience, pa_floor, seeds=SEEDS, per_airport=False, pa_trees="es",
               name=name)


def run(smoke: bool, out_dir, load, fit, record, n_boot: int = N_BOOT, log=print) -> dict:
    """Fit NSC, pair it with the incumbent (the FW record; an in-process 93-column refit under --smoke), write the
    report and the per-row predictions. `load(feats, smoke)` -> an lgbm_fold.load_fold dict; `fit` = lgbm_fold.fit_arm."""
    out_dir = pathlib.Path(out_dir)
    report_path, preds_path = out_dir / REPORT_NAME, out_dir / PREDS_NAME
    if report_path.exists() or preds_path.exists():
        raise FileExistsError(f"NSC outputs exist in {out_dir}; no silent rerun")
    t0 = time.time()
    fold = load(FEATS_NSC_FW, smoke)
    if any(c in SECONDS_COLS for c in fold["feats"]):
        raise AssertionError("the NSC design carries a seconds column")
    assert fold["feats"] == FEATS_NSC_FW, "the fold's design is not FEATS_NSC_FW"
    te = fold["te"]
    base = fold["base"].reset_index(drop=True)
    doy = fold["X"][te, fold["feats"].index("doy")].astype(int)
    day = base.month.to_numpy().astype(int) * 1000 + doy
    res = _fit(fit, fold, smoke, "nsc")
    nsc = np.asarray(res["delta"], float)
    fold["X"] = None
    gc.collect()
    if smoke:
        f93 = load(LF.FEATS_QUEUE_ORDER_WEATHER, smoke)
        r93 = _fit(fit, f93, smoke, "incumbent (smoke refit)")
        check_alignment(base, f93["base"].reset_index(drop=True))
        inc, source = np.asarray(r93["delta"], float), "in-process refit (smoke)"
        del f93
    else:
        check_alignment(base, record)
        inc, source = record.treatment.to_numpy(float), f"{RECORD.relative_to(ROOT)} column treatment"
    y, proxy = base.y.to_numpy(float), base.proxy.to_numpy(float)
    pooled = pair_cost(y, proxy, inc, nsc, day, n_boot, BOOT_SEED)
    by_ap, by_month = {}, {}
    for key, col, out in (("ap", base.ap.astype(str).to_numpy(), by_ap), ("month", base.month.to_numpy(), by_month)):
        for v in np.unique(col):
            m = col == v
            out[str(v)] = pair_cost(y[m], proxy[m], inc[m], nsc[m], day[m], n_boot, BOOT_SEED)
    rep = {"tag": "NSC", "smoke": bool(smoke), "prereg": str(PREREG.relative_to(ROOT)),
           "incumbent_source": source, "record_sha256": None if smoke else RECORD_SHA,
           "features": FEATS_NSC_FW, "n_features": len(FEATS_NSC_FW), "dropped": list(SECONDS_COLS),
           "n_rows": int(len(base)), "seeds": list(SEEDS), "best_iter": int(res["best_iter"]), "n_ref": int(res["n_ref"]),
           "single_seed_rmse": {str(k): float(v) for k, v in res["single_rmse"].items()},
           "pooled": pooled, "by_airport": by_ap, "by_month": by_month,
           "mde_bmse": mde_from_ci(pooled["ci95_day"]),
           "decision": decide(pooled["cost_bmse"], pooled["ci95_day"][0]),
           "wall_s": round(time.time() - t0, 1), "peak_rss_gb": round(LF.L.peak_rss_gb(), 2),
           "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = base[["row", "month", "ap", "y", "proxy", "delta"]].copy()
    preds["doy"] = doy
    preds["incumbent"] = inc
    for s in SEEDS:
        preds[f"delta_hat_seed{s}"] = np.asarray(res["single"][s], float)
    preds["nsc"] = nsc
    preds.to_parquet(preds_path, index=False)
    report_path.write_text(json.dumps(rep, indent=1))
    log(f"NSC cost {pooled['cost_bmse']:+.2f} board MSE, day CI [{pooled['ci95_day'][0]:+.2f}, "
        f"{pooled['ci95_day'][1]:+.2f}], MDE {rep['mde_bmse']:.1f}; decision {rep['decision']['decision']}")
    return rep


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="months 1-3, tiny trees, no lock, a scratch output dir")
    ap.add_argument("--out-dir", default=None, help=f"default {OUT_DIR} (real) or logs/nsc_scratch/smoke_<ts> (smoke)")
    return ap.parse_args(argv)


def main(argv=None) -> int:  # pragma: no cover - real-data entry point; exercised by the --smoke run, not pytest
    import run_guards as G
    args = parse_args(argv)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (
        ROOT / "logs" / "nsc_scratch" / f"smoke_{stamp}" if args.smoke else OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = LF._Log(out_dir / LOG_NAME)
    LF.log = log
    LF.L.log = log
    fh = None
    if not args.smoke:
        sha = G.assert_prereg_registered(PREREG, HASHLOG)
        log(f"registration {PREREG.name} sha256 {sha} is in {HASHLOG.name}")
        if G.sha256_file(RECORD) != RECORD_SHA:
            raise RuntimeError(f"{RECORD} is not the registered FW fold record")
        fh = G.acquire_lock()
        log("heavy-compute lock acquired (nonblocking)")
    G.start_rss_guard(RSS_LIMIT_BYTES)
    try:
        record = None if args.smoke else pd.read_parquet(
            RECORD, columns=["row", "month", "ap", "y", "proxy", "delta", "treatment"])
        rep = run(args.smoke, out_dir, _load_real, LF.fit_arm, record, log=log)
        log(f"report -> {out_dir / REPORT_NAME}   wall {rep['wall_s']:.0f}s   peak RSS {rep['peak_rss_gb']:.2f} GB")
    finally:
        if fh is not None:
            G.release_lock(fh)
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
