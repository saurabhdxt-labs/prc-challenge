#!/usr/bin/env python3.11
"""Priority 2 — properly conditioned fill / non-fill experts.

    $ENV scripts/cond_experts.py --smoke --out /tmp/x.json    # synthetic, proves it runs
    $ENV scripts/cond_experts.py --full [--queue] [--calibrate]
  where $ENV is  OMP_NUM_THREADS=1 nice -n 19 python3.11 -B -u

WHY THE OLD ARM WAS NOT A TEST OF THIS

RESULT 8 built `p·sp + (1−p)·m(X)` where `m` is the ALL-ROWS model. `m` already approximates
`E[y|X]`, which is itself the mixture, so the formula applies fill behaviour twice. Recomputed
per subset (reports/PRIORITY2_FILL_LANE_PRICED.md): the fill branch gains **+1,782 MSE** on the
30,167 schedule-fill rows and the same rule loses **1,719 MSE** on the 308,848 body rows,
netting +63 -- which is what got verdicted NOT WORKING.

The classifier was never the problem: holdout AUC 0.8957, decile fill rates monotone 0.000 ->
0.416. The body expert is what was missing.

WHAT THIS FITS

    p        = P(F=1 | X)                       binary, early-stopped on AUC
    mu_fill  = sp + shrunk mean(y - sp | F=1)   no model: |y - sp| <= 60 s by definition of F,
                                                so the correction is bounded and a shrunk
                                                per-airport mean is the whole of it
    mu_body  = proxy - E[delta | X, F=0]        fitted on NON-FILL TRAINING ROWS ONLY  <-- the
                                                piece that has never been fitted
    yhat     = p·mu_fill + (1 - p)·mu_body

against a `baseline` arm -- the all-rows delta model on the same rows, same encodings, same
seed -- so the comparison is paired and attributable to the conditioning alone.

Both arms use prc.encoding's separated stages (Priority 0): early stopping on fit-months-only
encodings, refit on cross-fitted ones. Running this against the leaky encoder would have to be
re-run, so it is not offered.

CALIBRATION, AND ITS ONE HONEST COMPROMISE

`--calibrate` fits an isotonic map on the EARLY-STOPPING months, using the fit-months
classifier's predictions there, and applies it to the holdout. Those months are also the
stopping set, so they are used twice. That is a real compromise and it is named rather than
hidden; the alternative -- leave-one-month-out out-of-fold predictions across all ten training
months -- costs ten extra classifier fits and is the right thing to do if this arm survives.
The HOLDOUT is never used to fit anything, which is the property that keeps the score honest.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import enc_ab as EA  # noqa: E402  (loads lgbm_fold, lgbm_submit, stand_ab, prc.encoding)
from prc import reference as REF  # noqa: E402

LF, L, S, E = EA.LF, EA.L, EA.S, EA.E
REPORTS = ROOT / "reports"
FILL_TOL_S = L.FILL_TOL_S
SHRINK = 50.0          # the repo's SMOOTH; a per-airport mean of at most +/-60 s needs no more
_t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} + {time.time() - _t0:7.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- the fill branch

def fill_correction(ap_code, y, sp, rows, n_airports: int, shrink: float = SHRINK) -> np.ndarray:
    """Shrunk per-airport mean of (y - sp) over `rows`, as a per-airport vector.

    Fitted on TRAINING rows only. Shrunk toward the pooled mean so an airport with few fills
    cannot swing its own correction; the quantity is bounded by FILL_TOL_S by construction, so
    this can never move a prediction by more than a minute.
    """
    ap_code, rows = np.asarray(ap_code), np.asarray(rows, dtype=bool)
    if not rows.any():
        raise ValueError("no rows to fit the fill correction on")
    r = np.asarray(y, dtype="float64")[rows] - np.asarray(sp, dtype="float64")[rows]
    if not np.isfinite(r).all():
        raise ValueError("non-finite (y - sp) on the fill-correction rows")
    prior = float(r.mean())
    out = np.full(int(n_airports), prior, dtype="float64")
    codes = ap_code[rows]
    for a in range(int(n_airports)):
        m = codes == a
        n = int(m.sum())
        if n:
            out[a] = (r[m].sum() + prior * shrink) / (n + shrink)
    return out


def assemble(p, mu_fill, mu_body) -> np.ndarray:
    """p·mu_fill + (1-p)·mu_body, with p clipped to [0, 1].

    Refuses non-finite inputs rather than letting a NaN reach the score, and refuses a length
    mismatch, which would broadcast silently and produce a plausible-looking wrong answer.
    """
    p, mu_fill, mu_body = (np.asarray(v, dtype="float64") for v in (p, mu_fill, mu_body))
    if not (p.shape == mu_fill.shape == mu_body.shape):
        raise ValueError(f"shape mismatch: p {p.shape}, mu_fill {mu_fill.shape}, "
                         f"mu_body {mu_body.shape}")
    for name, v in (("p", p), ("mu_fill", mu_fill), ("mu_body", mu_body)):
        if not np.isfinite(v).all():
            raise ValueError(f"{int((~np.isfinite(v)).sum()):,} non-finite values in {name}")
    p = np.clip(p, 0.0, 1.0)
    return p * mu_fill + (1.0 - p) * mu_body


def isotonic_map(p_cal, fill_cal):
    """Isotonic calibration fitted on a set that is NOT the holdout. Returns a callable."""
    from sklearn.isotonic import IsotonicRegression
    p_cal, fill_cal = np.asarray(p_cal, dtype="float64"), np.asarray(fill_cal, dtype="float64")
    if p_cal.shape != fill_cal.shape:
        raise ValueError(f"calibration length mismatch: {p_cal.shape} vs {fill_cal.shape}")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal, fill_cal)
    return lambda v: np.asarray(iso.predict(np.asarray(v, dtype="float64")), dtype="float64")


# --------------------------------------------------------------------------- the classifier

def fit_classifier(X, label, fit, es, tr, te, params, nest, patience, swap_to_refit=None):
    """p = P(F=1|X): early stop on the ES rows, refit on all training rows, predict ES and holdout.

    Returns (p_te, p_es, info). `p_es` comes from the FIT-MONTHS model and is what calibration
    may legitimately be fitted on; the refit model is used only for the holdout.
    """
    import lightgbm as lgb
    cp = dict(params, objective="binary", metric="auc")
    cp.pop("alpha", None)
    dfit = lgb.Dataset(X[fit], label=label[fit], free_raw_data=True)
    des = lgb.Dataset(X[es], label=label[es], reference=dfit, free_raw_data=True)
    ev = {}
    b = lgb.train(cp, dfit, num_boost_round=nest, valid_sets=[des],
                  callbacks=[lgb.early_stopping(patience, verbose=False),
                             lgb.record_evaluation(ev)])
    best = int(b.best_iteration)
    es_auc = float(ev["valid_0"]["auc"][best - 1])
    if es_auc <= L.FILL_HEAD_MIN_ES_AUC:
        raise RuntimeError(f"the classifier's stopping AUC {es_auc:.4f} is not above "
                           f"{L.FILL_HEAD_MIN_ES_AUC}: it learned nothing, refusing the refit")
    p_es = np.asarray(b.predict(X[es], num_iteration=best), dtype="float64")
    n_ref = int(L.n_refit(best, int(tr.sum()), int(fit.sum())))
    del dfit, des, b
    gc.collect()
    if swap_to_refit is not None:
        swap_to_refit()
    dtr = lgb.Dataset(X[tr], label=label[tr], free_raw_data=True)
    rb = lgb.train(cp, dtr, num_boost_round=n_ref)
    p_te = np.asarray(rb.predict(X[te], num_iteration=n_ref), dtype="float64")
    del dtr, rb
    gc.collect()
    return p_te, p_es, dict(best_iter=best, n_ref=n_ref,
                            es_auc_NOT_A_RESULT=es_auc)


# --------------------------------------------------------------------------- main

def apply_exclusions(yhat_cond, yhat_base, ap_code, airports, exclude) -> "tuple":
    """Keep the baseline untouched at named airports; return (blended, mask, names).

    Four pooled rules have now died on LIRF alone (RESULTS 16, 17, 20 and Arm U's LIRF
    384.07 -> 622.84), so an arm that cannot express "leave Rome alone" has to be re-run to
    answer the only question its per-airport table will raise. This is NOT airport selection
    after the fact: the excluded set is passed on the command line and belongs in a
    pre-registration written before the run.
    """
    names = [a.strip() for a in exclude if a.strip()]
    idx = {a: i for i, a in enumerate(airports)}
    bad = [a for a in names if a not in idx]
    if bad:
        raise ValueError(f"unknown airport(s) to exclude: {bad}; known {list(airports)}")
    keep = np.zeros(len(yhat_cond), dtype=bool)
    for a in names:
        keep |= (np.asarray(ap_code) == idx[a])
    out = np.asarray(yhat_cond, dtype="float64").copy()
    out[keep] = np.asarray(yhat_base, dtype="float64")[keep]
    return out, keep, names


def holdout_fingerprint(base: dict) -> str:
    """Order-sensitive identity of the holdout rows, from arrays `load_base` always provides.

    `MVT_ID` is not carried through the fold loader, so the fingerprint is taken over the
    holdout's (month, y, proxy, sp) — enough to make it impossible to pair a stored baseline
    against a different or reordered row set.
    """
    te = base["te"]
    cols = [np.asarray(base[k], dtype="float64")[te] for k in ("y", "proxy", "sp")]
    cols.append(np.asarray(base["month"], dtype="float64")[te])
    return REF.row_checksum(np.concatenate(cols))


def run(base: dict, feats_num, params, nest, patience, seeds, calibrate: bool,
        exclude=(), ref_path=None, months=(), split_name="fold_masks") -> dict:
    s1, s2, enc = EA.encoding_columns("separated", base["d"], base["y"], base["dlt"],
                                      base["month"], base["fit"], base["es"], base["tr"])
    feats = list(feats_num) + enc
    d = base["d"].copy()
    for c, v in s2.items():
        d[c] = v
    X = L.design_matrix(d, feats)
    del d
    gc.collect()
    te, tr, fit, es = base["te"], base["tr"], base["fit"], base["es"]
    y, dlt, proxy, sp = base["y"], base["dlt"], base["proxy"], base["sp"]
    F = LF.fill_mask(y, sp)
    log(f"design {X.shape[0]:,} x {X.shape[1]}; fills {F[tr].sum():,}/{tr.sum():,} training "
        f"({100 * F[tr].mean():.2f}%), {F[te].sum():,}/{te.sum():,} holdout")
    out = {"n_features": int(X.shape[1]), "fill_share_train": float(F[tr].mean()),
           "fill_share_holdout": float(F[te].mean())}

    def _delta_arm(name, train_m, fit_m, es_m):
        EA.write_enc(X, feats, s1)
        info = L.early_stop(X, dlt, y, proxy, train_m, fit_m, es_m,
                            dict(params, seed=L.ES_SEED), nest, patience)
        EA.write_enc(X, feats, s2)
        single = L.fit_seeds(X, L.regression_label(dlt, y), train_m, X[te], params,
                             int(info["n_ref"]), tuple(seeds), n_features=X.shape[1])
        log(f"[{name}] best_iter {info['best_iter']:,} -> n_ref {int(info['n_ref']):,}")
        ps = {int(s): float(LF.rmse(y[te], LF.taxi_time(proxy[te], single[s]))) for s in seeds}
        return L.mean_delta([single[s] for s in seeds]), dict(
            best_iter=int(info["best_iter"]), n_ref=int(info["n_ref"]),
            n_train=int(train_m.sum()), per_seed=ps,
            seed_sd=LF.seed_sd(list(ps.values())) if len(ps) > 1 else None)

    rows_fp = holdout_fingerprint(base)
    ref_meta = dict(feats=list(feats), params={k: v for k, v in sorted(params.items())},
                    seeds=list(seeds), months=list(months), split=split_name,
                    variant="separated", scale="taxi_time", rows=rows_fp, n_rows=int(te.sum()))
    # `n_ref` is INHERITED from the stored reference, NOT verified. Reusing a baseline skips
    # the stopping run, so this process never computes an n_ref to compare against; an earlier
    # version read n_ref out of the file and passed it back as the required value, which made
    # that one identity check a guaranteed no-op (a reference stored at n_ref=100 was reused
    # when 999 was required). It is now excluded from the match and reported instead, so the
    # inheritance is visible in the record rather than disguised as a check.
    # Every OTHER field -- ordered features, params, seeds, months, split, variant, scale and
    # the holdout row fingerprint -- is matched strictly and raises on any difference.
    reused = None
    if ref_path is not None and pathlib.Path(ref_path).exists():
        with np.load(ref_path, allow_pickle=False) as z:
            stored_n_ref = json.loads(str(z["meta"])).get("n_ref")
        reused = REF.load(ref_path, dict(ref_meta, n_ref=stored_n_ref))
        if reused is not None:
            log(f"[baseline] n_ref {stored_n_ref} INHERITED from the reference, not verified")
    if reused is not None:
        yhat_base, meta_stored = reused
        out["baseline"] = dict(meta_stored, reused_from=str(ref_path))
        out["baseline"]["holdout_rmse"] = float(LF.rmse(y[te], yhat_base))
        log(f"[baseline] REUSED from {ref_path}: n_ref {meta_stored.get('n_ref')}, "
            f"holdout matched RMSE {out['baseline']['holdout_rmse']:.4f}")
    else:
        dh_base, out["baseline"] = _delta_arm("baseline", tr, fit, es)
        yhat_base = LF.taxi_time(proxy[te], dh_base)
        out["baseline"]["holdout_rmse"] = float(LF.rmse(y[te], yhat_base))
        log(f"[baseline] holdout matched RMSE {out['baseline']['holdout_rmse']:.4f}")
        if ref_path is not None:
            REF.save(ref_path, yhat_base, dict(ref_meta, n_ref=out["baseline"]["n_ref"],
                                               holdout_rmse=out["baseline"]["holdout_rmse"]))
            log(f"[baseline] saved as a reusable reference -> {ref_path}")

    dh_body, out["body_expert"] = _delta_arm("mu_body", tr & ~F, fit & ~F, es & ~F)
    mu_body = LF.taxi_time(proxy[te], dh_body)

    EA.write_enc(X, feats, s1)
    lab = np.where(tr, F.astype("float64"), np.nan)
    p_te, p_es, out["classifier"] = fit_classifier(
        X, np.nan_to_num(lab), fit, es, tr, te, params, nest, patience,
        swap_to_refit=lambda: EA.write_enc(X, feats, s2))
    log(f"[p] stopping AUC {out['classifier']['es_auc_NOT_A_RESULT']:.4f} (NOT A RESULT); "
        f"mean p on holdout {p_te.mean():.4f} against a fill rate of {F[te].mean():.4f}")
    if calibrate:
        cal = isotonic_map(p_es, F[es])
        p_raw, p_te = p_te, cal(p_te)
        out["calibration"] = dict(
            fitted_on="early-stopping months, fit-months model",
            caveat="those months are also the stopping set; the holdout is never used",
            mean_p_before=float(p_raw.mean()), mean_p_after=float(p_te.mean()),
            holdout_fill_rate=float(F[te].mean()))
        log(f"[p] calibrated: mean p {p_raw.mean():.4f} -> {p_te.mean():.4f}")

    corr = fill_correction(base["ap_code"], y, sp, tr & F, len(base["airports"]))
    mu_fill = sp[te] + corr[base["ap_code"][te]]
    out["fill_correction_s"] = {a: float(c) for a, c in zip(base["airports"], corr)}
    # The SHIPPED positivity floor, lgbm_fold.taxi_time's max(raw, 1). `mu_body` carries it
    # already (it comes through taxi_time); `mu_fill` and the assembled mixture did not, which
    # would have compared a floored baseline against an unfloored treatment -- the same
    # floored-vs-unfloored error that made the first version of PRIORITY2_FILL_LANE_PRICED.md
    # wrong, re-introduced here. It is bug class BC-1's neighbour and it is why both arms are
    # floored at the same point now.
    yhat_cond = np.maximum(assemble(p_te, np.maximum(mu_fill, 1.0), mu_body), 1.0)
    out["floor_binds_rows"] = int((assemble(p_te, mu_fill, mu_body) < 1.0).sum())

    se = {"baseline": (yhat_base - y[te]) ** 2, "cond": (yhat_cond - y[te]) ** 2}
    out["cond"] = dict(holdout_rmse=float(LF.rmse(y[te], yhat_cond)))
    pairs = [("cond", "baseline")]
    if exclude:
        yhat_ex, keep, names = apply_exclusions(yhat_cond, yhat_base, base["ap_code"][te],
                                                base["airports"], exclude)
        se["cond_ex"] = (yhat_ex - y[te]) ** 2
        out["cond_ex"] = dict(holdout_rmse=float(LF.rmse(y[te], yhat_ex)),
                              excluded=names, n_rows_kept_at_baseline=int(keep.sum()))
        pairs.append(("cond_ex", "baseline"))
        log(f"[cond_ex] excluding {names}: {int(keep.sum()):,} rows keep the baseline; "
            f"holdout matched RMSE {out['cond_ex']['holdout_rmse']:.4f}")
    out["pairs"] = LF.paired_bootstrap(se, pairs)
    out["reliability"] = LF.reliability_deciles(p_te, F[te])
    # LIRF behaves unlike the other nine and a pooled number hides it. Repo rule.
    out["per_airport"] = {k: LF.arm_rmses({k: v}, base["ap_code"][te], base["airports"])[k]
                          for k, v in se.items()}
    w = LF.W_MATCHED_2026
    out["subsets"] = {}
    for nm, m in (("fill", F[te]), ("body", ~F[te])):
        out["subsets"][nm] = dict(
            n=int(m.sum()),
            baseline_rmse=float(np.sqrt(se["baseline"][m].mean())),
            cond_rmse=float(np.sqrt(se["cond"][m].mean())),
            mse_gain=float(w * (se["baseline"][m].sum() - se["cond"][m].sum()) / m.size))
    log(f"[cond] holdout matched RMSE {out['cond']['holdout_rmse']:.4f} against baseline "
        f"{out['baseline']['holdout_rmse']:.4f}")
    del X
    gc.collect()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--full", action="store_true")
    g.add_argument("--smoke", action="store_true",
                   help="three months, 60 rounds, one seed: proves the code runs. Its "
                        "magnitudes are NEVER quoted (MSE_LEDGER lever 4).")
    ap.add_argument("--queue", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                   help="isotonic on the ES months; see the module docstring's compromise")
    ap.add_argument("--exclude", default="",
                    help="comma-separated airports that keep the baseline untouched (e.g. LIRF). "
                         "Reported as a SEPARATE arm beside the unexcluded one; the excluded set "
                         "must be named in a pre-registration written before the run.")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds; one seed is the development setting and three "
                         "the confirmation. Default: 0 for --smoke, 0,1,2 for --full.")
    ap.add_argument("--ref", default=None,
                    help="path to a reusable baseline reference. Reused only when the ordered "
                         "feature list, params, seeds, n_ref, months, split, encoder variant, "
                         "prediction scale and holdout row fingerprint ALL match; any mismatch "
                         "raises naming the field. Written on a miss.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    feats_num = [f for f in (LF.FEATS_QUEUE if a.queue else L.FEATS)
                 if not f.startswith(("te_", "de_"))]
    if a.smoke:
        months, seeds = LF.SMOKE["months"], (0,)
        params = dict(L.P, learning_rate=LF.SMOKE["learning_rate"])
        nest, patience = LF.SMOKE["nest"], LF.SMOKE["patience"]
    else:
        months, seeds = tuple(range(1, 13)), LF.SEEDS
        params, nest, patience = dict(L.P), 60_000, 400
    if a.seeds:
        seeds = tuple(int(s) for s in a.seeds.split(",") if s.strip() != "")
        if not seeds:
            raise SystemExit("--seeds was given but parsed to nothing")
    log(f"mode {'smoke' if a.smoke else 'full'}; {len(feats_num)} numeric features; "
        f"queue {'on' if a.queue else 'off'}; calibrate {a.calibrate}; seeds {list(seeds)}")
    base = EA.load_base(LF.CACHE, months, feats_num, LF.QCACHE if a.queue else None,
                        LF.fold_masks)
    res = run(base, feats_num, params, nest, patience, seeds, a.calibrate,
              exclude=tuple(x for x in a.exclude.split(",") if x.strip()),
              ref_path=a.ref, months=months, split_name="fold_masks")

    out = pathlib.Path(a.out) if a.out else REPORTS / (
        f"cond_experts_{'smoke' if a.smoke else 'full'}.json")
    out.write_text(json.dumps(dict(
        mode="smoke" if a.smoke else "full", git=LF._git(), months=list(months),
        seeds=list(seeds), queue=bool(a.queue), calibrated=bool(a.calibrate),
        excluded=[x for x in a.exclude.split(",") if x.strip()],
        ranking_only=bool(a.smoke), results=res,
        note=("smoke magnitudes are RANKING ONLY and are not a result" if a.smoke else
              "fold A holdout; cond paired against the all-rows baseline on identical rows")),
        indent=2, default=LF._jsonable))
    log(f"json -> {out}   wall {time.time() - _t0:.0f}s   peak RSS {L.peak_rss_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
