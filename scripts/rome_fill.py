"""`scripts/rome_fill.py` -- arm R of plans/PREREG_rome_fill_2026_09_10.md.

    python scripts/rome_fill.py            # the real run: 12 months, LOMO-12 + fold A, seeds 0,1,2
    python scripts/rome_fill.py --smoke    # synthetic frames, code path only, NOT a result

A stake-weighted LightGBM fill classifier for Rome's unmatched rows, trained on ALL LIRF departures
of the training months (the schedule copied into BLOCK is visible on matched rows too), swapped
into S1's mixture in place of `p_hat` ONLY:

    pred = max(p * sp + (1 - p) * nf, 1)      nf = S1's stored non-fill term

so any difference against S1 is the classifier's. S1 must rebuild exactly from its stored parts
before anything is reported (`stratum_fold_v7_preds.parquet`; verified 0.0 max abs error).

Arms: S1 (control), R (mean p over seeds), R_seed{s}, R_unm (unmatched rows only, diagnostic),
R_perm (labels permuted within month, negative control). Folds: LOMO-12 and fold A (1, 7). The
five registered clauses are computed here; the design is frozen in the prereg, nothing is tuned.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import pathlib
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stratum_fold as sf  # noqa: E402

REPORTS = ROOT / "reports"
CACHE = ROOT / "data" / "cache_stand"
FRAME_CACHE = ROOT / "data" / "cache_rome"
STORED = CACHE / "stratum_fold_v7_preds.parquet"

# ---- the registered design (plans/PREREG_rome_fill_2026_09_10.md section 1), frozen ----
FILL_TOL_S = 60.0
W_UNMATCHED = 8.0
W_CAP = 25.0
FEATS_NUM = ["sp", "log_sp", "dayoff", "hr", "tmin", "dow", "sched_min", "unmatched_i"]
FEATS_CAT = ["airline", "STAND_mvt", "stand_c1", "ADES_mvt", "ades_p2", "RUNWAY_mvt", "FLIGHT_RULE_mvt"]
PARAMS = dict(objective="binary", num_leaves=15, learning_rate=0.05, n_estimators=400, min_child_samples=100,
              reg_lambda=30.0, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, cat_smooth=50,
              min_data_per_group=50, max_cat_to_onehot=4, n_jobs=4, verbose=-1)
SEEDS = (0, 1, 2)
FOLD_A = (1, 7)
N_BOOT, BOOT_SEED = 2_000, 0
# ---- registered thresholds (section 3) ----
MIN_REDUCTION = 0.15
MIN_MONTHS = 8
# ---- reported, not decisional ----
PUBLIC_JULY_ROME_RMSE = 3437.9       # elegant-alligator missing_v2.json, LIRF/7
SCORED_ROME_UNMATCHED = 383          # LIRF rows without AOBT_3 in the 2026 scored file
SCORED_ROWS = 344_841
SP_BANDS = (-1e18, 0.0, 1800.0, 3600.0, 10_800.0, 24_000.0, 1e18)


def stake_weight(sp, unmatched) -> np.ndarray:
    """clip(1 + (max(sp,0)/3600)^2, 1, W_CAP) x (W_UNMATCHED if unmatched else 1)."""
    sp = np.asarray(sp, dtype="float64")
    w = np.clip(1.0 + (np.maximum(sp, 0.0) / 3600.0) ** 2, 1.0, W_CAP)
    return w * np.where(np.asarray(unmatched, dtype=bool), W_UNMATCHED, 1.0)


def fill_label(frame: pd.DataFrame) -> np.ndarray:
    """The shipped fill flag: |BLOCK - SCHED| <= 60 s (build_submission.schedule_fill). Training only."""
    d = (frame.BLOCK_TIME_UTC_mvt - frame.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()
    return (d <= FILL_TOL_S).to_numpy()


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The registered features, all populated on scored rows; nothing reads BLOCK or the target."""
    sp = frame.sp.to_numpy(dtype="float64")
    out = pd.DataFrame({
        "sp": sp,
        "log_sp": np.log1p(np.maximum(sp, 0.0)),
        "dayoff": frame.dayoff.to_numpy(dtype="float64"),
        "hr": frame.hr.to_numpy(dtype="float64"),
        "tmin": frame.tmin.to_numpy(dtype="float64"),
        "dow": frame.dow.to_numpy(dtype="float64"),
        "sched_min": frame.SCHED_TIME_UTC_mvt.dt.minute.to_numpy(dtype="float64"),
        "unmatched_i": frame.unmatched.to_numpy(dtype="float64"),
    }, index=frame.index)
    for c in FEATS_CAT:
        out[c] = frame[c].astype("object").where(frame[c].notna(), "NA").astype(str)
    return out


def _design(train_f: pd.DataFrame, test_f: pd.DataFrame) -> tuple:
    """Categorical columns as pandas categories with the TRAINING levels; an unseen test level is NaN."""
    a, b = train_f.copy(), test_f.copy()
    for c in FEATS_CAT:
        levels = pd.Index(sorted(a[c].unique()))
        a[c] = pd.Categorical(a[c], categories=levels)
        # an unseen level becomes NaN EXPLICITLY: pandas' implicit coercion is deprecated and will raise
        b[c] = pd.Categorical(b[c].where(b[c].isin(levels)), categories=levels)
    return a, b


def rome_rows(unm: pd.DataFrame, lirf: pd.DataFrame, months) -> pd.DataFrame:
    """Every LIRF departure of `months`: unmatched rows from the stratum, matched rows from `lirf`."""
    months = list(months)
    u = unm[(unm.ADEP_mvt == "LIRF") & unm.month.isin(months)]
    m = lirf[lirf.month.isin(months)]
    return pd.concat([u, m], ignore_index=True)


def fit_predict_fill(train: pd.DataFrame, test: pd.DataFrame, seed: int, permute: bool = False,
                     extra_num=()) -> np.ndarray:
    """P(fill) for `test` from a stake-weighted LightGBM fitted on `train` (the registered design).
    `extra_num`: additional numeric columns (arm M's NM-clock fields); empty = arm R exactly."""
    import lightgbm as lgb

    y = fill_label(train).astype(int)
    if permute:
        rng = np.random.default_rng(10_000 + seed)
        y = y.copy()
        for mo in np.unique(train.month.to_numpy()):
            idx = np.flatnonzero(train.month.to_numpy() == mo)
            y[idx] = y[rng.permutation(idx)]
    w = stake_weight(train.sp, train.unmatched)
    xtr, xte = _design(feature_frame(train), feature_frame(test))
    for c in extra_num:
        xtr[c] = train[c].to_numpy(dtype="float64")
        xte[c] = test[c].to_numpy(dtype="float64")
    clf = lgb.LGBMClassifier(random_state=int(seed), **PARAMS)
    clf.fit(xtr, y, sample_weight=w, categorical_feature=FEATS_CAT)
    return clf.predict_proba(xte)[:, 1]


#: the SHIPPED mixture, max(p*sp + (1-p)*nf, 1) -- imported, never re-implemented: a local copy
#: let a floor deletion pass the rebuild check, because the fixture built S1 with the same copy
mixture = sf.bs.mixture


def stored_nf(stored: pd.DataFrame) -> np.ndarray:
    return np.where(stored.routed_nf_fit.to_numpy(dtype=bool), stored.nf_fit.to_numpy(dtype="float64"),
                    stored.nf_cells.to_numpy(dtype="float64"))


def month_block_bootstrap(sse_ref_by_month, sse_new_by_month, n_draws=N_BOOT, seed=BOOT_SEED) -> dict:
    """Paired month-block bootstrap of SSE(ref) - SSE(new): months resampled with replacement."""
    a = np.asarray(sse_ref_by_month, dtype="float64")
    b = np.asarray(sse_new_by_month, dtype="float64")
    rng = np.random.default_rng(seed)
    k = len(a)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        idx = rng.integers(0, k, k)
        draws[i] = a[idx].sum() - b[idx].sum()
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    return {"gain": float(a.sum() - b.sum()), "ci95": [lo, hi], "n_draws": int(n_draws), "seed": int(seed)}


def _weighted_auc(y, p, w) -> float | None:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=bool)
    if y.all() or (~y).all():
        return None
    return float(roc_auc_score(y, p, sample_weight=w))


def synthetic_stored(unm: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A stored-record stand-in for tests and --smoke: LIRF unmatched rows under fold 'lomo' (every
    month) and fold 'A' (months 1, 7), S1 built with the shipped mixture so it rebuilds exactly.

    nf is a PERFECT normal expert (the row's own y on non-fill rows, the non-fill mean on fill
    rows, where nf carries no weight once p is right), so the classifier is the only thing that
    can move SSE. A constant nf was tried first and made the comparison meaningless: a wrong p
    partly compensates for a biased nf inside the mixture. p_hat is deliberately WEAK but
    informative: half the fill rate, half the synthetic airline propensity, plus noise."""
    rng = np.random.default_rng(seed)
    u = unm[unm.ADEP_mvt == "LIRF"]
    prop = {"ITY": 0.7, "RYR": 0.2}                       # stratum_fold._P_FILL_LIRF, others 0.45
    parts = []
    for fold, rows in (("lomo", u), ("A", u[u.month.isin(FOLD_A)])):
        n = len(rows)
        fill = fill_label(rows)
        air = rows.airline.map(prop).fillna(0.45).to_numpy(dtype="float64")
        p = np.clip(0.5 * fill_label(u).mean() + 0.5 * air + rng.normal(0, 0.05, n), 0.01, 0.99)
        nf = np.where(fill, float(u.y[~fill_label(u)].mean()), rows.y.to_numpy(dtype="float64"))
        parts.append(pd.DataFrame({
            "MVT_ID_mvt": rows.MVT_ID_mvt.to_numpy(), "fold": fold, "month": rows.month.to_numpy(),
            "ADEP_mvt": "LIRF", "y": rows.y.to_numpy(dtype="float64"), "sp": rows.sp.to_numpy(dtype="float64"),
            "p_hat": p, "nf_cells": nf, "nf_fit": nf, "routed_nf_fit": False,
            "S1": mixture(p, rows.sp.to_numpy(dtype="float64"), nf)}))
    return pd.concat(parts, ignore_index=True)


def _fold_predictions(unm, lirf, test_ids, train_months, seeds, log) -> dict:
    """{arm: p} for the unmatched LIRF rows with `test_ids` (in that order)."""
    u_lirf = unm[unm.ADEP_mvt == "LIRF"].set_index("MVT_ID_mvt", drop=False)
    missing = np.setdiff1d(test_ids, u_lirf.index.to_numpy())
    if len(missing):
        raise ValueError(f"{len(missing)} stored test rows are absent from the unmatched frame")
    te = u_lirf.loc[test_ids]
    tr = rome_rows(unm, lirf, train_months)
    out = {}
    for s in seeds:
        out[f"R_seed{s}"] = fit_predict_fill(tr, te, seed=s)
    out["R"] = np.mean([out[f"R_seed{s}"] for s in seeds], axis=0)
    out["R_unm"] = fit_predict_fill(tr[tr.unmatched], te, seed=seeds[0])
    out["R_perm"] = fit_predict_fill(tr, te, seed=seeds[0], permute=True)
    log(f"  trained on {len(tr):,} LIRF rows ({int(tr.unmatched.sum()):,} unmatched), months {sorted(set(train_months))}; "
        f"scored {len(te):,}")
    return out


def run(unm: pd.DataFrame, lirf: pd.DataFrame, stored: pd.DataFrame, seeds=SEEDS, n_boot=N_BOOT, log=print) -> dict:
    seeds = tuple(seeds)
    st = stored[stored.ADEP_mvt == "LIRF"].reset_index(drop=True)
    nf = stored_nf(st)
    rebuilt = mixture(st.p_hat.to_numpy(dtype="float64"), st.sp.to_numpy(dtype="float64"), nf)
    err = float(np.abs(rebuilt - st.S1.to_numpy(dtype="float64")).max())
    if err != 0.0:
        raise AssertionError(f"S1 does not rebuild from its stored parts (max abs error {err}); nothing is reported")

    arms = ["S1"] + [f"R_seed{s}" for s in seeds] + ["R", "R_unm", "R_perm"]
    P = {a: np.full(len(st), np.nan) for a in arms}
    P["S1"] = st.p_hat.to_numpy(dtype="float64")
    all_months = sorted(set(unm.month.unique()) | set(lirf.month.unique()))
    for fold in ("lomo", "A"):
        if fold == "lomo":
            groups = [(int(m), [int(m)]) for m in sorted(st[st.fold == "lomo"].month.unique())]
        else:
            groups = [("A", list(FOLD_A))]
        for key, held in groups:
            rows = np.flatnonzero((st.fold == fold).to_numpy() & st.month.isin(held).to_numpy())
            if not len(rows):
                continue
            log(f"fold {fold} {key}:")
            got = _fold_predictions(unm, lirf, st.MVT_ID_mvt.to_numpy()[rows],
                                    [m for m in all_months if m not in held], seeds, log)
            for a, p in got.items():
                P[a][rows] = p

    y, sp = st.y.to_numpy(dtype="float64"), st.sp.to_numpy(dtype="float64")
    pred = {a: mixture(P[a], sp, nf) for a in arms}
    se = {a: (pred[a] - y) ** 2 for a in arms}
    fill = (np.abs(y - sp) <= FILL_TOL_S)

    def fold_block(fold):
        m = (st.fold == fold).to_numpy()
        months = st.month.to_numpy()[m]
        by_month = {a: pd.Series(se[a][m]).groupby(months).sum() for a in arms}
        rec = {"n_rows": int(m.sum()),
               "sse": {a: float(se[a][m].sum()) for a in arms},
               "rmse": {a: float(np.sqrt(se[a][m].mean())) for a in arms},
               "sse_by_month": {a: {str(k): float(v) for k, v in by_month[a].items()} for a in arms},
               "weighted_auc_sp2": {a: _weighted_auc(fill[m & (sp > 0)], P[a][m & (sp > 0)], sp[m & (sp > 0)] ** 2)
                                    for a in arms}}
        bands = pd.cut(sp[m], SP_BANDS)
        rec["sse_by_sp_band"] = {a: {str(k): float(v) for k, v in pd.Series(se[a][m]).groupby(bands, observed=True).sum().items()}
                                 for a in ("S1", "R")}
        return rec, by_month

    lomo, lomo_bm = fold_block("lomo")
    fa, _ = fold_block("A")
    boot = month_block_bootstrap(lomo_bm["S1"].to_numpy(), lomo_bm["R"].to_numpy(), n_boot, BOOT_SEED)
    seed_rmse = [lomo["rmse"][f"R_seed{s}"] for s in seeds]
    seed_sd = float(np.std(seed_rmse, ddof=1)) if len(seeds) >= 2 else None
    gain_rmse = lomo["rmse"]["S1"] - lomo["rmse"]["R"]
    red_lomo = 1 - lomo["sse"]["R"] / lomo["sse"]["S1"]
    red_a = 1 - fa["sse"]["R"] / fa["sse"]["S1"]
    months_improving = int((lomo_bm["R"] < lomo_bm["S1"]).sum())
    clauses = {
        "c1_month_block_interval": bool(boot["ci95"][0] > 0),
        "c2_lomo_reduction_ge_15pct": bool(red_lomo >= MIN_REDUCTION),
        "c3_foldA_reduction_ge_15pct": bool(red_a >= MIN_REDUCTION),
        "c4_gain_over_2x_seed_sd": bool(seed_sd is not None and gain_rmse > 2 * seed_sd),
        "c5_months_improving_ge_8": bool(months_improving >= MIN_MONTHS),
    }
    verdict = ("ESTABLISHED" if all(clauses.values()) else
               "NOT WORKING" if not clauses["c1_month_block_interval"] else "INCONCLUSIVE")
    jul = ((st.fold == "A") & (st.month == 7)).to_numpy()
    controls = {"s1_rebuilds_exactly": True, "s1_rebuild_max_abs_error": err,
                "perm_does_not_beat_s1": bool(lomo["sse"]["R_perm"] >= lomo["sse"]["S1"])}
    preds = st[["MVT_ID_mvt", "fold", "month", "y", "sp"]].copy()
    preds["nf"] = nf
    for a in arms:
        preds[f"p_{a}"] = P[a]
        preds[a] = pred[a]
    return {
        "arms": arms, "controls": controls, "lomo": lomo, "fold_A": fa,
        "month_block_bootstrap_lomo": boot, "seed_rmse_lomo": dict(zip([f"R_seed{s}" for s in seeds], seed_rmse)),
        "seed_sd": seed_sd, "gain_rmse_lomo": gain_rmse, "reduction_lomo": red_lomo, "reduction_foldA": red_a,
        "months_improving": months_improving, "n_months": int(len(lomo_bm["S1"])),
        "clauses": clauses, "verdict": verdict,
        "reported": {
            "foldA_july_rome_rmse": {a: float(np.sqrt(se[a][jul].mean())) for a in ("S1", "R")} if jul.any() else None,
            "public_july_rome_rmse": PUBLIC_JULY_ROME_RMSE,
            "projected_board_mse": float((fa["sse"]["S1"] - fa["sse"]["R"]) * (SCORED_ROME_UNMATCHED / max(fa["n_rows"], 1))
                                         / SCORED_ROWS),
        },
        "preds": preds,
    }


# =============================================================================================
# entry point
# =============================================================================================

def _peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def with_sp_bins(d: pd.DataFrame) -> pd.DataFrame:
    """Re-attach derive()'s `sp_band` / `sp_fine` from `sp` with the SHIPPED edges and labels.
    The cache stores neither (categoricals), and fit_unmatched's cells read both: a cache without
    them broke the ship step with KeyError('sp_band') on 2026-09-10."""
    d = d.copy()
    d["sp_band"] = pd.cut(d.sp, sf.bs.SP_EDGES, labels=sf.bs.SP_LABELS)
    d["sp_fine"] = pd.cut(d.sp, sf.bs.SP_FINE, labels=[str(i) for i in range(len(sf.bs.SP_FINE) - 1)])
    return d


def load_frames(log) -> tuple:
    """(unmatched stratum, LIRF matched) through stratum_fold.load_real, cached in data/cache_rome/.
    Every column derive() produces is present on return, from the cache or not."""
    fu, fl = FRAME_CACHE / "unm.parquet", FRAME_CACHE / "lirf.parquet"
    if fu.exists() and fl.exists():
        log(f"frames from {FRAME_CACHE}")
        return with_sp_bins(pd.read_parquet(fu)), with_sp_bins(pd.read_parquet(fl))
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "training_2025-*.parquet")))
    if len(files) != 12:
        raise FileNotFoundError(f"expected 12 training months, found {len(files)}")
    unm, lirf = sf.load_real(files, log)
    FRAME_CACHE.mkdir(parents=True, exist_ok=True)
    for d, p in ((unm, fu), (lirf, fl)):
        d.drop(columns=[c for c in ("sp_band", "sp_fine") if c in d.columns]).to_parquet(p)
    return unm, lirf


def report(res: dict, log) -> None:
    for name, blk in (("LOMO-12", res["lomo"]), ("fold A", res["fold_A"])):
        log(f"{name}: Rome unmatched rows {blk['n_rows']:,}")
        for a in res["arms"]:
            log(f"  {a:8s} RMSE {blk['rmse'][a]:9.1f}  SSE {blk['sse'][a] / 1e6:9.1f} M  "
                f"sp2-weighted AUC {blk['weighted_auc_sp2'][a] if blk['weighted_auc_sp2'][a] is None else round(blk['weighted_auc_sp2'][a], 3)}")
    b = res["month_block_bootstrap_lomo"]
    log(f"month-block bootstrap (LOMO, S1 - R SSE): {b['gain'] / 1e6:+.1f} M  CI [{b['ci95'][0] / 1e6:+.1f}, {b['ci95'][1] / 1e6:+.1f}] M")
    log(f"reduction LOMO {100 * res['reduction_lomo']:.1f}%  fold A {100 * res['reduction_foldA']:.1f}%  "
        f"months improving {res['months_improving']}/{res['n_months']}  gain RMSE {res['gain_rmse_lomo']:+.1f} vs 2 x seed sd "
        f"{2 * res['seed_sd'] if res['seed_sd'] is not None else float('nan'):.1f}")
    log(f"controls: {res['controls']}")
    for k, v in res["clauses"].items():
        log(f"  {k:30s} {'PASS' if v else 'fail'}")
    log(f"VERDICT (R): {res['verdict']}")
    log(f"reported: {res['reported']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="synthetic frames + synthetic stored record; NOT a result")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out_dir) if a.out_dir else (ROOT / "data" / "smoke_rome" if a.smoke else REPORTS)
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "rome_fill.log"
    fh = open(logf, "w")
    t0 = time.time()

    def log(msg):
        line = f"[{dt.datetime.now():%H:%M:%S} +{time.time() - t0:6.0f}s] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    if a.smoke:
        log("SMOKE - code path only, NOT a result")
        unm, lirf = sf.synthetic_frame()
        stored = synthetic_stored(unm)
        res = run(unm, lirf, stored, seeds=SEEDS, n_boot=200, log=log)
    else:
        unm, lirf = load_frames(log)
        stored = pd.read_parquet(STORED)
        res = run(unm, lirf, stored, seeds=SEEDS, n_boot=N_BOOT, log=log)
    report(res, log)
    preds = res.pop("preds")
    preds_path = (out if a.smoke else CACHE) / "rome_fill_preds.parquet"
    preds.to_parquet(preds_path)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    res.update({"prereg": "plans/PREREG_rome_fill_2026_09_10.md", "smoke": bool(a.smoke), "git_sha": sha,
                "wall_s": round(time.time() - t0, 1), "peak_rss_gb": round(_peak_rss_gb(), 2),
                "params": PARAMS, "features": FEATS_NUM + FEATS_CAT, "w_unmatched": W_UNMATCHED, "w_cap": W_CAP,
                "preds_parquet": str(preds_path)})
    (out / "rome_fill.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"json -> {out / 'rome_fill.json'}   wall {res['wall_s']:.0f}s   peak RSS {res['peak_rss_gb']} GB")
    if a.smoke:
        log("SMOKE - code path only, NOT a result")
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
