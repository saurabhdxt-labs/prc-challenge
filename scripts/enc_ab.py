#!/usr/bin/env python3.11
"""Priority 0 + the scoped-key A/B: three encoding variants on one fold load.

    $ENV scripts/enc_ab.py --screen            # subfold ranking, 1 seed, no refit (~20 min/arm)
    $ENV scripts/enc_ab.py --full              # fold A, 3 seeds, refit (hours/arm)
  where $ENV is  OMP_NUM_THREADS=1 nice -n 19 python3.11 -B -u

The screening tier fits months 2-5, stops on 6 and SCORES ON MONTH 8, which no arm ever sees.
It refits on 2-6 at n_ref and reports a real holdout RMSE. Ranking encoders by a stopping metric
would be invalid: the incumbent's stopping rows carry their own labels in their features.

WHAT THIS ANSWERS

Two questions that share a fold load, because the expensive part is reading twelve months of
cache and the cheap part is recomputing 24 encoding columns:

  incumbent   stand_ab.infold_encodings fitted on `tr` -- exactly what ships today, INCLUDING
              the early-stopping leak. This is the control and it must reproduce.
  separated   prc.encoding.separated_encodings: the ES rows are encoded by a fit-months-only
              encoder, training rows are cross-fitted leave-one-MONTH-out. Priority 0.
  scoped      separated, plus airport-scoped composites of the keys whose names are reused
              across airports (55.3% of rows carry a shared STAND_mvt name). Priority 1.

Each variant early-stops on its STAGE 1 columns and refits on its STAGE 2 columns. A single
design matrix serves both because only the encoding columns differ between stages, and they are
written in place -- see prc/encoding.py's module docstring for why two stages are needed.

WHAT IT DOES NOT DO

It does not touch scripts/lgbm_fold.py, scripts/stand_ab.py or any cache; it imports them. It
writes no submission. Screen-tier magnitudes are RANKING ONLY and are never a result -- the
project has already paid once for quoting a subsample magnitude (MSE_LEDGER lever 4).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LF = _load("lgbm_fold")
L, S = LF.L, LF.S
from prc import encoding as E  # noqa: E402

VARIANTS = ("incumbent", "separated", "scoped")
REPORTS = ROOT / "reports"
_t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} + {time.time() - _t0:7.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- encoding variants

def encoding_columns(variant: str, d: pd.DataFrame, y, dlt, month, fit, es, tr,
                     enc_keys=None) -> tuple:
    """(stage1 cols, stage2 cols, feature names) for one variant.

    `incumbent` returns the SAME dict for both stages -- that identity is what makes it the
    control: the shipped harness uses one column set for early stopping and refit alike, and
    that column set was fitted on `tr`, which contains the stopping rows.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    enc_keys = list(S.ENC_KEYS if enc_keys is None else enc_keys)
    keys = d[enc_keys]
    if variant == "incumbent":
        cols = S.infold_encodings(keys, y, dlt, tr, enc_keys=enc_keys)
        return cols, cols, sorted(cols)
    if variant == "scoped":
        keys, made = E.add_airport_scoped(keys)
        enc_keys = enc_keys + made
    stages = E.separated_encodings(keys, y, dlt, month, fit_mask=fit, es_mask=es, tr_mask=tr,
                                   enc_keys=enc_keys)
    return stages["stage1"], stages["stage2"], sorted(stages["stage2"])


def write_enc(X: np.ndarray, feats, cols: dict) -> int:
    """Overwrite X's encoding columns in place. Returns how many were written.

    Raises rather than silently skipping: a name in `cols` that is not in `feats` means the
    design matrix and the encoder disagree, which would leave stale columns in X and produce a
    result that looks like a model difference but is a wiring bug.
    """
    idx = {f: i for i, f in enumerate(feats)}
    missing = [c for c in cols if c not in idx]
    if missing:
        raise KeyError(f"{len(missing)} encoding columns absent from the design matrix: "
                       f"{missing[:5]}")
    for c, v in cols.items():
        X[:, idx[c]] = np.asarray(v, dtype=np.float32)
    return len(cols)


def enc_checksum(X: np.ndarray, feats, enc_names) -> str:
    """A cheap fingerprint of the encoding columns currently in X.

    Recorded at early stopping and again at the refit so the stage swap is an OBSERVABLE, not
    an assumption. `separated` and `scoped` must differ between the two; `incumbent` must not.
    """
    idx = [feats.index(c) for c in enc_names]
    h = hashlib.blake2b(digest_size=8)
    for i in idx:
        h.update(np.ascontiguousarray(X[:, i]).tobytes())
    return h.hexdigest()


#: Screening tier. `lgbm_fold.sweep_masks` cannot serve this comparison: it returns
#: (stop, train, train, stop), so its "holdout" IS its stopping set and there is no independent
#: score at all. Ranking the incumbent against the repaired encoder on a STOPPING metric is
#: worse than useless -- the incumbent's stopping rows carry their own labels in their features,
#: so the leaking arm wins by construction. This tier therefore holds out a month that neither
#: arm ever sees, and every arm is scored there.
SCREEN = dict(fit=(2, 3, 4, 5), stop=6, score=8)


def screen_masks(month, cfg=None):
    """(holdout, training, fit, early-stop) for the screening tier, with a real holdout."""
    cfg = SCREEN if cfg is None else cfg
    month = np.asarray(month)
    fit = np.isin(month, cfg["fit"])
    es = month == cfg["stop"]
    te = month == cfg["score"]
    if (te & (fit | es)).any():
        raise ValueError("the screening score month overlaps the fit or stopping months")
    if not te.any():
        raise ValueError(f"screening score month {cfg['score']} absent from the loaded months")
    return te, fit | es, fit, es


# --------------------------------------------------------------------------- the fold

def load_base(cache, months, feats, qcache, split):
    """_load_months + concat + masks, WITHOUT the encodings. The keys survive for the encoder."""
    frames, paths = LF._load_months(cache, months, feats, qcache)
    d = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    month = d.month.to_numpy()
    te, tr, fit, es = split(month)
    ap_code, airports = L.airport_codes(d.ap)
    log(f"loaded {len(paths)} months ({len(d):,} rows): holdout {te.sum():,}; training "
        f"{tr.sum():,}; fit {fit.sum():,}; early-stop {es.sum():,}")
    return dict(d=d, month=month, te=te, tr=tr, fit=fit, es=es, ap_code=ap_code,
                airports=airports, y=d.y.to_numpy(), dlt=d.delta.to_numpy(),
                proxy=d.proxy.to_numpy(), sp=d.sp.to_numpy(), n_months=len(paths))


def fit_variant(base: dict, variant: str, feats_num, params, nest, patience, seeds,
                refit: bool) -> dict:
    """One variant end to end: encode, build X, early-stop on stage 1, refit on stage 2."""
    t = time.time()
    s1, s2, enc_names = encoding_columns(variant, base["d"], base["y"], base["dlt"],
                                         base["month"], base["fit"], base["es"], base["tr"])
    feats = list(feats_num) + enc_names
    d = base["d"].copy()
    for c, v in s2.items():
        d[c] = v
    X = L.design_matrix(d, feats)
    del d
    gc.collect()
    log(f"[{variant}] {len(enc_names)} encoding columns, design matrix {X.shape[0]:,} x "
        f"{X.shape[1]} ({X.nbytes / 1e9:.2f} GB)   [{time.time() - t:.0f}s]")

    write_enc(X, feats, s1)                      # stage 1: the stopping-safe columns
    es_sum = enc_checksum(X, feats, enc_names)
    info = L.early_stop(X, base["dlt"], base["y"], base["proxy"], base["tr"], base["fit"],
                        base["es"], dict(params, seed=L.ES_SEED), nest, patience)
    n_ref = int(info["n_ref"])
    # the stopping metric keeps lgbm_submit's deliberate name: it is an in-fold number and,
    # under the incumbent variant, a LEAKY one. It ranks nothing and is never a result.
    es_key = "es_rmse_taxi_time_NOT_A_RESULT"
    log(f"[{variant}] best_iter {info['best_iter']:,} -> n_ref {n_ref:,}; "
        f"stopping RMSE {info.get(es_key, float('nan')):.4f} (NOT A RESULT)")
    out = dict(variant=variant, n_features=int(X.shape[1]), n_enc=len(enc_names),
               best_iter=int(info["best_iter"]), n_ref=n_ref,
               es_rmse_NOT_A_RESULT=float(info.get(es_key, np.nan)),
               es_enc_checksum=es_sum)
    if not refit:
        del X
        gc.collect()
        return out

    write_enc(X, feats, s2)                      # stage 2: cross-fitted for the refit
    out["refit_enc_checksum"] = enc_checksum(X, feats, enc_names)
    if variant != "incumbent" and out["refit_enc_checksum"] == es_sum:
        raise RuntimeError(
            f"[{variant}] the refit matrix is bit-identical to the stopping matrix: the stage "
            "swap did not happen, so one of the two fits is using the wrong encodings")
    label = L.regression_label(base["dlt"], base["y"])
    single = L.fit_seeds(X, label, base["tr"], X[base["te"]], params, n_ref, tuple(seeds),
                         n_features=X.shape[1])
    pooled = L.mean_delta([single[s] for s in seeds])
    yhat = LF.taxi_time(base["proxy"][base["te"]], pooled)
    y_te = base["y"][base["te"]]
    per_seed = {int(s): float(LF.rmse(y_te, LF.taxi_time(base["proxy"][base["te"]], single[s])))
                for s in seeds}
    out.update(holdout_rmse=float(LF.rmse(y_te, yhat)), per_seed=per_seed,
               seed_sd=LF.seed_sd(list(per_seed.values())) if len(per_seed) > 1 else None)
    # LIRF carries 22.6% of matched SSE on 7.7% of rows; a pooled number hides it. Repo rule.
    se = (yhat - y_te) ** 2
    out["per_airport"] = LF.arm_rmses({"arm": se}, base["ap_code"][base["te"]],
                                      base["airports"])["arm"]
    out["_pred"] = yhat
    log(f"[{variant}] holdout matched RMSE {out['holdout_rmse']:.4f}")
    del X
    gc.collect()
    return out


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--screen", action="store_true",
                   help="subfold ranking tier: no refit, 1 seed. RANKING ONLY.")
    g.add_argument("--full", action="store_true", help="fold A, refit, 3 seeds")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--queue", action="store_true", help="include the shipped queue block")
    ap.add_argument("--smoke", action="store_true",
                    help="three synthetic months, 60 rounds, one seed: proves the code runs. "
                         "Its magnitudes are NEVER quoted (MSE_LEDGER lever 4).")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variant(s) {bad}; expected from {VARIANTS}")

    feats_num = list(LF.FEATS_QUEUE if a.queue else L.FEATS)
    feats_num = [f for f in feats_num if not f.startswith(("te_", "de_"))]
    qcache = LF.QCACHE if a.queue else None
    if a.smoke:
        months, split, seeds, refit = LF.SMOKE["months"], LF.fold_masks, (0,), a.full
        params = dict(L.P, learning_rate=LF.SMOKE["learning_rate"])
        nest, patience = LF.SMOKE["nest"], LF.SMOKE["patience"]
    elif a.screen:
        months = SCREEN["fit"] + (SCREEN["stop"], SCREEN["score"])
        split, seeds, refit = screen_masks, (0,), True
        params = dict(L.P, learning_rate=LF.SWEEP["learning_rate"])
        nest, patience = 30_000, 200
    else:
        months, split, seeds, refit = tuple(range(1, 13)), LF.fold_masks, LF.SEEDS, True
        params, nest, patience = dict(L.P), 60_000, 400

    log(f"mode {'smoke' if a.smoke else 'screen' if a.screen else 'full'}; variants {variants}; "
        f"{len(feats_num)} numeric features; queue block {'on' if a.queue else 'off'}")
    base = load_base(LF.CACHE, months, feats_num, qcache, split)

    results, preds = {}, {}
    for v in variants:
        r = fit_variant(base, v, feats_num, params, nest, patience, seeds, refit)
        preds[v] = r.pop("_pred", None)
        results[v] = r

    if refit and len(variants) > 1 and all(preds[v] is not None for v in variants):
        y_te = base["y"][base["te"]]
        se = {v: (preds[v] - y_te) ** 2 for v in variants}
        pairs = [(v, "incumbent") for v in variants if v != "incumbent"
                 and "incumbent" in variants]
        results["pairs"] = LF.paired_bootstrap(se, pairs) if pairs else {}

    tier = "smoke" if a.smoke else "screen" if a.screen else "full"
    out = pathlib.Path(a.out) if a.out else REPORTS / f"enc_ab_{tier}.json"
    out.write_text(json.dumps(dict(
        mode=tier, git=LF._git(), variants=variants,
        queue=bool(a.queue), months=list(months), seeds=list(seeds),
        ranking_only=bool(a.screen or a.smoke), results=results,
        note=("screen/smoke magnitudes are RANKING ONLY and are not a result"
              if (a.screen or a.smoke) else
              "fold A holdout, paired against the incumbent encoder")),
        indent=2, default=LF._jsonable))
    log(f"json -> {out}   wall {time.time() - _t0:.0f}s   peak RSS {L.peak_rss_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
