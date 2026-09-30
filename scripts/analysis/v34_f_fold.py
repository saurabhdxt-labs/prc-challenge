"""v34 step 1 — F-NSC fold A (plans/PREREG_nsc_no_seconds_2026_09_24.md §7 step 1, AMENDMENT V34.1).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/v34_launch.py --step 1 --detach [--smoke]      # always via the launcher
    (the launcher runs: v34_f_fold.py --rss-limit-gib 6 [--smoke])

The shipped F record (data/cache_stand/fold_preds_queue_order.parquet, sha 8536bb87...) was written by
`lgbm_fold.py --orderfeats --queue --baseline A2 --pa-trees es`: fold A (holdout months 1, 7; early stopping on 3, 9 carved
from training; in-fold encodings), `lgbm_fold.fit_arm` with lgbm_submit.P, early stopping seed 0, seeds 0/1/2 refitted at
n_ref, the pooled mean, NO per-airport blend (A2). The ADS-B stack is trained on that record's residuals with its
`treatment` as the input F, so v34 needs the same record without the seconds. This runner repeats the recipe on
nsc_fold.FEATS_NSC_F (the 83 minus aobt_sec / mvt_sec / sched_sec, order kept, 80 columns) and writes:

  data/nsc_v34/step1_f_fold/fold_preds_queue_order_nsc.parquet   row, month, ap, y, proxy, delta, sp, delta_hat_seed0..2,
                                                                  treatment   (the columns adsb_stack.load_frame reads)
  data/nsc_v34/step1_f_fold/report.json                           pairing vs the shipped F record (nsc_fold.pair_cost,
                                                                  day-block CI), the registered NSC decision rule, status

Acceptance (V34.1): rows aligned exactly to the shipped record (row / month / ap / y / proxy / delta), no seconds column in
the design, finite treatment, and nsc_fold.decide on the pairing reads PROCEED (point < 337 and day-block lower < 150);
ESCALATE writes the report with status FAIL and step 2 refuses to start.
Imports lgbm_fold (LightGBM) only: no torch, no CatBoost here (BC-5).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.util
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


C = _load("v34_common", "scripts/analysis/v34_common.py")
NSC = _load("nsc_fold", "scripts/analysis/nsc_fold.py")
LF = NSC.LF

STEP = 1
FEATS = C.assert_no_seconds(NSC.FEATS_NSC_F, "FEATS_NSC_F")             # 80
RECORD = ROOT / "data" / "cache_stand" / "fold_preds_queue_order.parquet"
RECORD_SHA = "8536bb872ebae4770ee705bcbcb7af25aa8fe083d19dd507bf9a8bcda3fb6c06"
RECORD_NAME = "fold_preds_queue_order_nsc.parquet"
LOG_NAME = "v34_f_fold.log"
SEEDS = (0, 1, 2)
RECORD_COLS = ["row", "month", "ap", "y", "proxy", "delta", "sp", "delta_hat_seed0", "delta_hat_seed1", "delta_hat_seed2",
               "treatment"]
N_BOOT = NSC.N_BOOT


def fit_config(smoke: bool) -> tuple:
    """(params, nest, patience, pa_floor): lgbm_submit's production values, or lgbm_fold's SMOKE values."""
    params, nest, patience, pa_floor = dict(LF.L.P), LF.L.NEST, LF.L.PATIENCE, LF.L.PA_TREE_FLOOR
    if smoke:
        params["learning_rate"] = LF.SMOKE["learning_rate"]
        nest, patience, pa_floor = LF.SMOKE["nest"], LF.SMOKE["patience"], LF.SMOKE["pa_floor"]
    return params, nest, patience, pa_floor


def load_real(feats, smoke: bool) -> dict:  # pragma: no cover - real caches; exercised by the smoke run
    months = LF.SMOKE["months"] if smoke else None
    return LF.load_fold(LF.CACHE, months, list(feats), qcache=LF.QCACHE, ocache=LF.OCACHE, wcache=LF.WCACHE,
                        check_counts=not smoke, label=f"holdout (months {LF.HOLDOUT})")


def _fit(fit, fold, smoke: bool, name: str) -> dict:
    params, nest, patience, pa_floor = fit_config(smoke)
    return fit(fold, fold["X"], params, nest, patience, pa_floor, seeds=SEEDS, per_airport=False, pa_trees="es",
               name=name)


def run(smoke: bool, out_dir, load, fit, record, n_boot: int = N_BOOT, log=print,
        allow_roots=(C.OUT_ROOT, C.SMOKE_ROOT)) -> dict:
    """Fit F-NSC, pair it with the shipped F record (an in-process 83-column refit under --smoke), write the new record and
    the report. `load(feats, smoke)` -> an lgbm_fold.load_fold dict; `fit` = lgbm_fold.fit_arm."""
    out_dir = pathlib.Path(out_dir)
    rec_path = C.check_output_path(out_dir / RECORD_NAME, allow_roots=allow_roots, allow_files=())
    rep_path = C.check_output_path(out_dir / C.REPORT_NAME, allow_roots=allow_roots, allow_files=())
    t0 = time.time()
    fold = load(FEATS, smoke)
    C.assert_no_seconds(fold["feats"], "the step-1 fold design")
    if list(fold["feats"]) != FEATS:
        raise AssertionError("the fold's design is not FEATS_NSC_F")
    te = fold["te"]
    base = fold["base"].reset_index(drop=True)
    doy = fold["X"][te, fold["feats"].index("doy")].astype(int)
    day = base.month.to_numpy().astype(int) * 1000 + doy
    res = _fit(fit, fold, smoke, "f_nsc")
    new = np.asarray(res["delta"], float)
    if not np.isfinite(new).all():
        raise AssertionError("non-finite F-NSC prediction")
    fold["X"] = None
    gc.collect()
    if smoke:
        f83 = load(LF.FEATS_QUEUE_ORDER, smoke)
        r83 = _fit(fit, f83, smoke, "f83 (smoke refit)")
        NSC.check_alignment(base, f83["base"].reset_index(drop=True))
        inc, source = np.asarray(r83["delta"], float), "in-process 83-column refit (smoke)"
        del f83
    else:
        NSC.check_alignment(base, record)
        inc, source = record.treatment.to_numpy(float), f"{C.rel(RECORD)} column treatment"
    y, proxy = base.y.to_numpy(float), base.proxy.to_numpy(float)
    pooled = NSC.pair_cost(y, proxy, inc, new, day, n_boot, NSC.BOOT_SEED)
    by_ap, by_month = {}, {}
    for col, out in ((base.ap.astype(str).to_numpy(), by_ap), (base.month.to_numpy(), by_month)):
        for v in np.unique(col):
            m = col == v
            out[str(v)] = NSC.pair_cost(y[m], proxy[m], inc[m], new[m], day[m], n_boot, NSC.BOOT_SEED)
    decision = NSC.decide(pooled["cost_bmse"], pooled["ci95_day"][0])
    move = C.taxi(proxy, new) - C.taxi(proxy, inc)
    rec = base[["row", "month", "ap", "y", "proxy", "delta", "sp"]].copy()
    for s in SEEDS:
        rec[f"delta_hat_seed{s}"] = np.asarray(res["single"][s], float)
    rec["treatment"] = new
    rec = rec[RECORD_COLS]
    sha = C.write_parquet_new(rec, rec_path, allow_roots=allow_roots, allow_files=())
    status = "SMOKE" if smoke else ("PASS" if decision["decision"] == "PROCEED" else "FAIL")
    rep = {"step": STEP, "tag": "V34.1 F-NSC fold A", "smoke": bool(smoke), "status": status,
           "prereg": C.rel(C.PREREG), "incumbent_source": source, "record_sha256": None if smoke else RECORD_SHA,
           "features": FEATS, "n_features": len(FEATS), "dropped": list(C.SECONDS_COLS), "n_rows": int(len(base)),
           "seeds": list(SEEDS), "best_iter": int(res["best_iter"]), "n_ref": int(res["n_ref"]),
           "single_seed_rmse": {str(k): float(v) for k, v in res["single_rmse"].items()},
           "treatment_rmse": float(np.sqrt(np.mean((y - C.taxi(proxy, new)) ** 2))),
           "rms_move_s": float(np.sqrt(np.mean(move ** 2))),
           "pairing": pooled, "by_airport": by_ap, "by_month": by_month, "mde_bmse": NSC.mde_from_ci(pooled["ci95_day"]),
           "decision": decision, "outputs": {C.rel(rec_path): sha},
           "wall_s": round(time.time() - t0, 1), "peak_rss_gb": round(LF.L.peak_rss_gb(), 2),
           "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    C.write_json_new(rep, rep_path, allow_roots=allow_roots, allow_files=())
    log(f"F-NSC vs the F record: cost {pooled['cost_bmse']:+.2f} bmse, day CI [{pooled['ci95_day'][0]:+.2f}, "
        f"{pooled['ci95_day'][1]:+.2f}], decision {decision['decision']}; status {status}")
    return rep


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rss-limit-gib", type=float, required=True, help="in-process peak-RSS abort (the launcher passes it)")
    ap.add_argument("--smoke", action="store_true", help="months 1-3, tiny trees, no lock, a scratch output dir")
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:  # pragma: no cover - real-data entry point; exercised by the --smoke run, not pytest
    import run_guards as G
    args = parse_args(argv)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (
        C.SMOKE_ROOT / f"smoke_step1_{stamp}" if args.smoke else C.step_dir(STEP))
    out_dir.mkdir(parents=True, exist_ok=True)
    log = LF._Log(out_dir / LOG_NAME)
    LF.log = log
    LF.L.log = log
    fh = None
    if not args.smoke:
        sha = C.require_registration(STEP)
        log(f"registration {C.PREREG.name} sha256 {sha} logged; AMENDMENT V34.1 present")
        if G.sha256_file(RECORD) != RECORD_SHA:
            raise RuntimeError(f"{RECORD} is not the shipped F fold record")
        fh = G.acquire_lock()
        log("heavy-compute lock acquired (nonblocking)")
    G.start_rss_guard(int(args.rss_limit_gib * 1024 ** 3))
    log(f"in-process RSS abort at {args.rss_limit_gib:g} GiB")
    try:
        record = None if args.smoke else pd.read_parquet(RECORD, columns=["row", "month", "ap", "y", "proxy", "delta",
                                                                          "treatment"])
        rep = run(args.smoke, out_dir, load_real, LF.fit_arm, record, log=log)
        log(f"report -> {out_dir / C.REPORT_NAME}   wall {rep['wall_s']:.0f}s   peak RSS {rep['peak_rss_gb']:.2f} GB")
    finally:
        if fh is not None:
            G.release_lock(fh)
        log.close()
    return 0 if rep["status"] in ("PASS", "SMOKE") else 3


if __name__ == "__main__":
    raise SystemExit(main())
