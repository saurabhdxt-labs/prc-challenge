"""Arm ADN harness (plans/PREREG_adsb_nn_2026_09_10.md, amendments ADN.1–ADN.3).

    python3.11 scripts/adsb_stack.py run          # arms S, G (3 seeds), N (3 seeds), B; controls K+ / K0; clauses

Rows: fold A's 339,015 matched holdout rows (data/cache_stand/fold_preds_queue_order.parquet), joined row-for-row to the
label-free ADS-B table (data/adsb/v2/features_2025janjul.parquet). F = the record's `treatment` (delta scale).
Every arm predicts delta; taxi = max(proxy − delta, 1) (BC-2 convention). Rows with no joined ADS-B flight keep F.
Scored rows exclude the spent day 2025-01-09.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECORD = ROOT / "data" / "cache_stand" / "fold_preds_queue_order.parquet"
FEATURES = ROOT / "data" / "adsb" / "v2" / "features_2025janjul.parquet"
RAW_MONTHS = [ROOT / "data" / "raw" / "training_2025-01-01_2025-02-01.parquet",
              ROOT / "data" / "raw" / "training_2025-07-01_2025-08-01.parquet"]
OUT_JSON = ROOT / "reports" / "adsb_nn.json"
OUT_PREDS = ROOT / "data" / "adsb" / "v2" / "adn_fold_preds.parquet"
SPENT_DAYS = ("2025-01-09",)
N_FOLDS = 5
MATCHED_WEIGHT = 0.98466
F_GUARD_RMSE = 222.5632
SEEDS = (0, 1, 2)
TARGET_SCALE = 300.0
LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbosity=-1, num_threads=4,
                  deterministic=True, force_row_wise=True)
LGB_ROUNDS, LGB_PATIENCE = 3000, 100
NN_HIDDEN, NN_DROPOUT, NN_LR, NN_BATCH, NN_EPOCHS, NN_PATIENCE = 128, 0.1, 1e-3, 8192, 40, 4   # ADN.6: 1,024 tripped the guard
GRAD_SPIKE = 10.0
Z_CLIP = 300.0                 # AFJ (plans/PREREG_adsbfi_july_fallback_2026_09_16.md §2): _nn_matrix clips standardised
#                                inputs to ±Z_CLIP, the next hundred above the largest |z| any shipped table ever fed the
#                                MLP (272.6 = ground_rate_pm 12,000 under fold stats), so every shipped row is bit-identical.
#                                A duplicated adsb.fi fix reached z ~ 3e7 and an MLP residual of -1.9e6 s. The clip bounds
#                                the input; the output guard is prc/pipeline/adsb_stage.MAX_ABS_RESID_S
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


# ---------------------------------------------------------------- folds

def day_folds(days) -> np.ndarray:
    """Fold per row: sorted days within each month, spent days excluded (NaN), fold = index mod N_FOLDS."""
    days = pd.Series(days).astype(str).reset_index(drop=True)
    m = {}
    for month in sorted(days.str[:7].unique()):
        for i, d in enumerate(sorted(d for d in days[days.str[:7] == month].unique() if d not in SPENT_DAYS)):
            m[d] = i % N_FOLDS
    return days.map(m).to_numpy(dtype=float)


# ---------------------------------------------------------------- helpers

def taxi(proxy, delta) -> np.ndarray:
    return np.maximum(np.asarray(proxy, float) - np.asarray(delta, float), 1.0)


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def net_weighted(y, base, arm) -> float:
    y, base, arm = (np.asarray(v, float) for v in (y, base, arm))
    return MATCHED_WEIGHT * (np.sum((base - y) ** 2) - np.sum((arm - y) ** 2)) / len(y)


def day_block_ci(y, base, arm, days, n_draws=2000, seed=0, q=(0.025, 0.975)) -> tuple:
    """Paired day-block bootstrap of RMSE(base) − RMSE(arm): whole DATES resampled, January and July separately."""
    y, base, arm = (np.asarray(v, float) for v in (y, base, arm))
    days = pd.Series(np.asarray(days).astype(str))
    codes, uniq = pd.factorize(days)
    eb = np.bincount(codes, (base - y) ** 2, minlength=len(uniq))
    ea = np.bincount(codes, (arm - y) ** 2, minlength=len(uniq))
    nn = np.bincount(codes, minlength=len(uniq)).astype(float)
    months = pd.Series(uniq).str[:7].to_numpy()
    groups = [np.flatnonzero(months == m) for m in sorted(set(months))]
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for i in range(n_draws):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        n = nn[idx].sum()
        out[i] = np.sqrt(eb[idx].sum() / n) - np.sqrt(ea[idx].sum() / n)
    return float(np.quantile(out, q[0])), float(np.quantile(out, q[1]))


# ---------------------------------------------------------------- arm S

def arm_s(fr: pd.DataFrame) -> np.ndarray:
    """ADN.2: on coverage-3 rows, taxi_S = anchor − b_ap with anchor = dwell_end_rel_mvt if finite else
    first_ground_rel_mvt, b_ap = the OTHER folds' median of (anchor − y) per airport; delta_S = proxy − taxi_S."""
    out = fr.F.to_numpy(dtype=float).copy()
    anchor = np.where(np.isfinite(fr.dwell_end_rel_mvt.to_numpy(float)), fr.dwell_end_rel_mvt.to_numpy(float),
                      fr.first_ground_rel_mvt.to_numpy(float))
    use = (fr.coverage.to_numpy() == 3) & np.isfinite(anchor) & np.isfinite(fr.fold.to_numpy(float))
    r = anchor - fr.y.to_numpy(float)
    for k in range(N_FOLDS):
        te = use & (fr.fold.to_numpy() == k)
        for ap in np.unique(fr.ap.to_numpy()[te]):
            tr = use & (fr.fold.to_numpy() != k) & (fr.ap.to_numpy() == ap)
            if not tr.any():
                continue
            b = float(np.median(r[tr]))
            m = te & (fr.ap.to_numpy() == ap)
            out[m] = fr.proxy.to_numpy(float)[m] - (anchor[m] - b)
    return out


# ---------------------------------------------------------------- stackers

def _design(fr: pd.DataFrame, feats: list) -> pd.DataFrame:
    x = pd.DataFrame({"F": fr.F.to_numpy(float), "proxy": fr.proxy.to_numpy(float),
                      "hour": fr.hour.to_numpy(float)})
    for c in feats:
        x[c] = fr[c].to_numpy(float)
    x["ap"] = pd.Categorical(fr.ap, categories=AIRPORTS).codes.astype(int)
    return x


def _fit_lgb(X, r, fit, es, seed, params=None):
    import lightgbm as lgb
    p = dict(params or LGB_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
    dtr = lgb.Dataset(X[fit], r[fit], categorical_feature=["ap"], free_raw_data=False)
    des = lgb.Dataset(X[es], r[es], categorical_feature=["ap"], reference=dtr, free_raw_data=False)
    return lgb.train(p, dtr, LGB_ROUNDS, valid_sets=[des],
                     callbacks=[lgb.early_stopping(LGB_PATIENCE, verbose=False)])


def _nn_matrix(X: pd.DataFrame, stats=None):
    num = X.drop(columns=["ap"])
    num = num.assign(hs=np.sin(2 * np.pi * X.hour / 24), hc=np.cos(2 * np.pi * X.hour / 24)).drop(columns=["hour"])
    miss = num.isna().to_numpy().astype(np.float32)
    if stats is None:
        mu = np.nanmean(num.to_numpy(float), axis=0)
        sd = np.nanstd(num.to_numpy(float), axis=0)
        mu, sd = np.nan_to_num(mu), np.where(np.nan_to_num(sd) > 0, np.nan_to_num(sd), 1.0)
        stats = (mu, sd)
    z = (num.to_numpy(float) - stats[0]) / stats[1]
    z = np.where(np.isfinite(z), z, 0.0)            # a missing input is carried by its indicator, never by the 0
    z = np.clip(z, -Z_CLIP, Z_CLIP)                 # AFJ: no standardised input beyond every shipped version's support
    onehot = np.eye(len(AIRPORTS), dtype=np.float32)[np.clip(X.ap.to_numpy(), 0, len(AIRPORTS) - 1)]
    return np.hstack([z.astype(np.float32), miss, onehot]), stats


def _run_worker(args: list, d: pathlib.Path):
    import subprocess
    p = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "adsb_nn_worker.py"), *args],
                       capture_output=True, text=True)
    if p.returncode == 3:
        raise FloatingPointError(str(np.load(d / "out.npz")["error"]))
    if p.returncode != 0:
        raise RuntimeError(f"NN worker failed ({p.returncode}): {p.stderr[-1500:]}")


def _fit_nn(X, r, fit, es, te, seed, save_path=None) -> np.ndarray:
    """BC-5: train in scripts/adsb_nn_worker.py (a subprocess that never imports lightgbm); returns the predicted
    residual (seconds) on rows `te`. A worker guard trip is re-raised here as FloatingPointError."""
    import shutil
    import tempfile
    A, stats = _nn_matrix(X[fit])
    B, _ = _nn_matrix(X[es], stats)
    T, _ = _nn_matrix(X[te], stats)
    work = ROOT / "data" / "adsb" / "v2" / ".nn_work"
    work.mkdir(parents=True, exist_ok=True)
    d = pathlib.Path(tempfile.mkdtemp(dir=work))
    try:
        extra = {} if save_path is None else {"save_path": str(save_path)}
        np.savez(d / "in.npz", A=A, ya=r[fit] / TARGET_SCALE, B=B, yb=r[es] / TARGET_SCALE, T=T, seed=seed,
                 hidden=NN_HIDDEN, dropout=NN_DROPOUT, lr=NN_LR, batch=NN_BATCH, epochs=NN_EPOCHS,
                 patience=NN_PATIENCE, grad_spike=GRAD_SPIKE, **extra)
        _run_worker([str(d / "in.npz"), str(d / "out.npz")], d)
        if save_path is not None:
            np.savez(str(save_path).replace(".pt", "_stats.npz"), mu=stats[0], sd=stats[1])
        return TARGET_SCALE * np.load(d / "out.npz")["pred"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def fit_stacker(fr: pd.DataFrame, feats: list, arm: str, seed: int, train_rows=None, predict_rows=None,
                lgb_params=None) -> np.ndarray:
    """Out-of-fold delta predictions: F + predicted residual on joined rows (coverage >= 2) with a fold; F elsewhere.
    `train_rows` / `predict_rows` (boolean masks) restrict which rows may train and which are predicted (ADN.7's
    leave-one-airport-out transfer test); by default every joined row does both."""
    X = _design(fr, feats)
    r = fr.delta.to_numpy(float) - fr.F.to_numpy(float)
    fold = fr.fold.to_numpy(float)
    joined = (fr.coverage.to_numpy() >= 2) & np.isfinite(fold)
    trn = joined if train_rows is None else joined & np.asarray(train_rows, bool)
    prd = joined if predict_rows is None else joined & np.asarray(predict_rows, bool)
    out = fr.F.to_numpy(float).copy()
    for k in range(N_FOLDS):
        te = prd & (fold == k)
        if not te.any():
            continue
        es_fold = max(f for f in range(N_FOLDS) if f != k)
        fit = trn & (fold != k) & (fold != es_fold)
        es = trn & (fold == es_fold)
        if arm == "G":
            model = _fit_lgb(X, r, fit, es, seed, lgb_params)
            out[te] = out[te] + model.predict(X[te], num_iteration=model.best_iteration)
        elif arm == "N":
            out[te] = out[te] + _fit_nn(X, r, fit, es, te, seed)
        else:
            raise ValueError(f"unknown arm {arm!r}")
    return out


# ---------------------------------------------------------------- clauses

def clauses(ci, net, month_gain, ap_gain, gain, seed_sd) -> dict:
    c = {"C1": ci[0] > 0, "C2": net >= 500.0, "C3": all(v > 0 for v in month_gain.values()),
         "C4": sum(v > 0 for v in ap_gain.values()) >= 2 and all(v >= -1.0 for v in ap_gain.values()),
         "C5": None if seed_sd is None else bool(gain > 2 * seed_sd)}
    return {k: (None if v is None else bool(v)) for k, v in c.items()}


def verdict(c: dict) -> str:
    if not c["C1"]:
        return "NOT WORKING"
    return "WORKING" if all(v for v in c.values() if v is not None) else "INCONCLUSIVE"


def score_arm(fr: pd.DataFrame, delta_hat: np.ndarray, seed_rmses=None, base_delta=None) -> dict:
    """Paired comparison of an arm against a baseline (F by default; ADN.5: the same stacker without ADS-B)."""
    s = np.isfinite(fr.fold.to_numpy(float))
    y, px = fr.y.to_numpy(float)[s], fr.proxy.to_numpy(float)[s]
    bd = fr.F.to_numpy(float) if base_delta is None else np.asarray(base_delta, float)
    base, arm = taxi(px, bd[s]), taxi(px, np.asarray(delta_hat, float)[s])
    days, mon, ap = fr.day.to_numpy()[s], fr.month.to_numpy()[s], fr.ap.to_numpy()[s]
    gain = _rmse(base, y) - _rmse(arm, y)
    mg = {int(m): _rmse(base[mon == m], y[mon == m]) - _rmse(arm[mon == m], y[mon == m]) for m in np.unique(mon)}
    ag = {a: _rmse(base[ap == a], y[ap == a]) - _rmse(arm[ap == a], y[ap == a]) for a in np.unique(ap)}
    ci = day_block_ci(y, base, arm, days)
    seed_sd = float(np.std(seed_rmses, ddof=1)) if seed_rmses is not None and len(seed_rmses) > 1 else None
    c = clauses(ci, net_weighted(y, base, arm), mg, ag, gain, seed_sd)
    return {"baseline": "F" if base_delta is None else "given", "rmse_base": _rmse(base, y),
            "rmse_F": _rmse(taxi(px, fr.F.to_numpy(float)[s]), y), "rmse_arm": _rmse(arm, y), "gain_s": gain, "ci95_day_block": ci,
            "net_weighted_fold_mse": net_weighted(y, base, arm), "month_gain_s": mg, "airport_gain_s": ag,
            "seed_rmses": seed_rmses, "seed_sd": seed_sd, "clauses": c, "verdict": verdict(c),
            "n_rows_scored": int(s.sum()), "n_rows_changed": int(np.sum(np.asarray(delta_hat)[s] != bd[s]))}


ADJ_C7_BAR = -150.0            # arm ADJ: the no-harm bar on rows joined today, in weighted fold MSE
ADJ_KPLUS_BAR = 1000.0         # arm ADJ: the planted signal can only act on the recovered rows (BC-10: derived here)
ADJ_K0_BAR = 150.0             # arm ADJ.1: noise must not WIN; against this twin a noise arm must lose, so only the
#                                positive side can fail. See the amendment for the derivation.


def today_coverage(fr: pd.DataFrame) -> pd.DataFrame:
    """Arm ADJ's twin frame: the recovered rows (join_mvt_anchored == 1) put back to coverage 1, so the stacker neither
    trains on them nor moves them, and every other row untouched. The caller's frame is not modified."""
    out = fr.copy()
    new = out.join_mvt_anchored.to_numpy(dtype=float) == 1
    cov = out.coverage.to_numpy().copy()
    cov[new] = 1
    out["coverage"] = cov
    return out


def clauses_adj(base_clauses: dict, old_rows_net: float) -> dict:
    """ADJ's clauses: ADN's C1-C5 plus C7, the no-harm bar on the rows that were already joined."""
    return {**base_clauses, "C7": bool(old_rows_net >= ADJ_C7_BAR)}


def permute_rows(fr: pd.DataFrame, feats: list, mask, seed: int = 0) -> pd.DataFrame:
    """K0 for a rows-under-test design: shuffle `feats` AMONG the masked rows only; every other row keeps its values.
    Permuting the whole column would destroy signal the twin still has (BC-10)."""
    out = fr.copy()
    m = np.asarray(mask, bool)
    idx = np.flatnonzero(m)
    if len(idx) > 1:
        rng = np.random.default_rng(seed)
        for c in feats:
            v = out[c].to_numpy(dtype=float).copy()
            v[idx] = v[rng.permutation(idx)]
            out[c] = v
    return out


ADX_C6_BAR = -150.0            # arm ADX: the code-3 no-harm bar, in weighted fold MSE


def subset_net(fr: pd.DataFrame, mask, base_delta, delta_hat) -> float:
    """The weighted fold-MSE change contributed by a SUBSET of the scored rows: its own SSE change divided by ALL
    scored rows, so subset nets add up to the arm's net (the units C6 is registered in)."""
    s = np.isfinite(fr.fold.to_numpy(float))
    m = s & np.asarray(mask, bool)
    if not m.any():
        return 0.0
    y, px = fr.y.to_numpy(float), fr.proxy.to_numpy(float)
    base, arm = taxi(px[m], np.asarray(base_delta, float)[m]), taxi(px[m], np.asarray(delta_hat, float)[m])
    return float(MATCHED_WEIGHT * (np.sum((base - y[m]) ** 2) - np.sum((arm - y[m]) ** 2)) / int(s.sum()))


def adx_breakdown(fr: pd.DataFrame, base_delta, delta_hat) -> dict:
    """Per coverage code (2 = joined mid-taxi, 3 = seen from the stand): rows, net and both RMSEs."""
    s = np.isfinite(fr.fold.to_numpy(float))
    y, px = fr.y.to_numpy(float), fr.proxy.to_numpy(float)
    out = {}
    for code in (2, 3):
        m = s & (fr.coverage.to_numpy() == code)
        base, arm = taxi(px[m], np.asarray(base_delta, float)[m]), taxi(px[m], np.asarray(delta_hat, float)[m])
        out[str(code)] = {"n": int(m.sum()), "net": subset_net(fr, fr.coverage.to_numpy() == code, base_delta, delta_hat),
                          "rmse_base": _rmse(base, y[m]) if m.any() else None,
                          "rmse_arm": _rmse(arm, y[m]) if m.any() else None}
    return out


def clauses_adx(base_clauses: dict, code3_net: float) -> dict:
    """ADX's clauses: ADN's C1-C5 plus C6, the code-3 no-harm bar (net >= ADX_C6_BAR)."""
    return {**base_clauses, "C6": bool(code3_net >= ADX_C6_BAR)}


def smoke_subset(fr: pd.DataFrame, per_month: int = 5) -> pd.DataFrame:
    """The first `per_month` scored days of each month (all five folds present when per_month >= N_FOLDS)."""
    keep = []
    for month in sorted(fr.day.str[:7].unique()):
        ds = sorted(d for d in fr.day[fr.day.str[:7] == month].unique() if d not in SPENT_DAYS)
        keep += ds[:per_month]
    return fr[fr.day.isin(keep)].reset_index(drop=True)


# ---------------------------------------------------------------- data

def load_frame(features_path=None) -> pd.DataFrame:
    import importlib.util
    spec = importlib.util.spec_from_file_location("adsb_features", ROOT / "scripts" / "adsb_features.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    rec = pd.read_parquet(RECORD, columns=["row", "month", "ap", "y", "proxy", "delta", "treatment"])
    feat = pd.read_parquet(FEATURES if features_path is None else features_path)
    if not (feat.row.to_numpy() == rec.row.to_numpy()).all():
        raise ValueError("feature table rows are not the record's rows in order")
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]) for p in RAW_MONTHS])
    mvt = raw.drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").MVT_TIME_UTC_mvt
    t = pd.to_datetime(feat.MVT_ID_mvt.map(mvt), utc=True)
    if t.isna().any():
        raise ValueError("feature rows without an MVT time")
    fr = pd.concat([rec.reset_index(drop=True),
                    feat.drop(columns=["row"]).reset_index(drop=True)], axis=1).rename(columns={"treatment": "F"})
    fr["day"] = t.dt.strftime("%Y-%m-%d").to_numpy()
    fr["hour"] = t.dt.hour.to_numpy()
    fr["fold"] = day_folds(fr.day)
    guard = _rmse(taxi(fr.proxy, fr.F), fr.y)
    if abs(guard - F_GUARD_RMSE) > 1e-3:
        raise ValueError(f"F guard failed: {guard:.4f} != {F_GUARD_RMSE}")
    return fr


CONTRACT_COLS = ["ap", "hour", "proxy", "F", "coverage"]


def train_artifacts(fr: pd.DataFrame, feats: list, out_dir, seeds=SEEDS, gate=None, ship_learner: str = "B") -> dict:
    """The ship hand-off: the evaluated CV configuration saved per (learner, seed, fold): G boosters (txt), N
    state_dicts (trained in the worker, BC-5) + standardisation stats, and MANIFEST.json (feats, seeds, gate)."""
    import subprocess
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X = _design(fr, feats)
    r = fr.delta.to_numpy(float) - fr.F.to_numpy(float)
    fold = fr.fold.to_numpy(float)
    joined = (fr.coverage.to_numpy() >= 2) & np.isfinite(fold)
    for s in seeds:
        for k in range(N_FOLDS):
            es_fold = max(f for f in range(N_FOLDS) if f != k)
            fit, es = joined & (fold != k) & (fold != es_fold), joined & (fold == es_fold)
            b = _fit_lgb(X, r, fit, es, s)
            b.save_model(str(out_dir / f"G_s{s}_k{k}.txt"), num_iteration=b.best_iteration)
            _fit_nn(X, r, fit, es, es, s, save_path=out_dir / f"N_s{s}_k{k}.pt")
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    man = {"feats": list(feats), "seeds": list(seeds), "folds": N_FOLDS, "airports": AIRPORTS,
           "target_scale": TARGET_SCALE, "nn": {"hidden": NN_HIDDEN, "dropout": NN_DROPOUT, "batch": NN_BATCH},
           "gate": dict(gate or {}), "git_sha": sha + "-dirty", "trained": dt.datetime.now().astimezone().isoformat(),
           "n_train_joined_rows": int(joined.sum()), "blend": "B = mean(G, N) residual, seconds",
           "ship_learner": ship_learner}
    (out_dir / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man


def _combine(per_model: dict, man: dict, fold, n_rows: int) -> np.ndarray:
    """per_model[(seed, fold)] -> array. Full bag = the mean of all; oof = per row, the mean over seeds of that row's
    held-out fold's models."""
    if fold is None:
        return np.mean(list(per_model.values()), axis=0)
    out = np.zeros(n_rows)
    for k in range(man["folds"]):
        rows = fold == k
        out[rows] = np.mean([per_model[(s, k)][rows] for s in man["seeds"]], axis=0)
    return out


def _predict_g(X, model_dir, man: dict, coverage, fold=None) -> np.ndarray:
    """The LightGBM half of the bag. `coverage` is REQUIRED (not optional/defaulted): rows with coverage<2 are
    forced to residual 0.0, matching `fit_stacker`'s own `joined = (coverage>=2) & isfinite(fold)` mask, which
    never assigns a real residual to a coverage<2 row in the first place. Before this, `_predict_g` applied the
    shipped booster to EVERY row with no coverage check at all, relying entirely on `predict_residual` (its only
    caller at the time) to mask it afterward -- a new direct caller that didn't know to reapply that mask got
    silently wrong values (found 2026-09-18 building scripts/analysis/acx_g_build.py: an unmasked pull corrupted a
    blended-RMSE reproduction by ~5 seconds). Making `coverage` a required positional argument turns that silent
    misuse into an immediate TypeError instead of a wrong number."""
    import lightgbm as lgb
    raw = _combine({(s, k): lgb.Booster(model_file=str(pathlib.Path(model_dir) / f"G_s{s}_k{k}.txt")).predict(X)
                    for s in man["seeds"] for k in range(man["folds"])}, man, fold, len(X))
    return np.where(np.asarray(coverage, dtype=float) >= 2, raw, 0.0)


def _predict_n(X, model_dir, man: dict, coverage, fold=None) -> np.ndarray:
    """The MLP half, in its own worker process (BC-5: torch never enters this one). `coverage` is REQUIRED — see
    `_predict_g`'s docstring for why (the identical defect existed here, fixed in the same commit)."""
    import shutil
    import tempfile
    model_dir = pathlib.Path(model_dir)
    work = ROOT / "data" / "adsb" / "v2" / ".nn_work"
    work.mkdir(parents=True, exist_ok=True)
    d = pathlib.Path(tempfile.mkdtemp(dir=work))
    try:
        pairs = [(s, k) for s in man["seeds"] for k in range(man["folds"])]
        per = {}
        for s, k in pairs:
            st_ = np.load(model_dir / f"N_s{s}_k{k}_stats.npz")
            T, _ = _nn_matrix(X, (st_["mu"], st_["sd"]))
            np.savez(d / "in.npz", n_models=1, hidden=man["nn"]["hidden"], dropout=man["nn"]["dropout"],
                     T_0=T, path_0=np.array(str(model_dir / f"N_s{s}_k{k}.pt")))
            _run_worker(["predict", str(d / "in.npz"), str(d / "out.npz")], d)
            per[(s, k)] = man["target_scale"] * np.load(d / "out.npz")["pred"]
        raw = _combine(per, man, fold, len(X))
        return np.where(np.asarray(coverage, dtype=float) >= 2, raw, 0.0)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def predict_residual(frame: pd.DataFrame, model_dir, oof: bool = False) -> np.ndarray:
    """Contract entry point for the pipeline: the shipped residual (seconds) per row; 0.0 where coverage < 2 or the
    airport is not gated in.

    Which learner ships is the manifest's `ship_learner`: "B" (mean of G and N, what v11/v12 ship and the default when
    the key is absent) or "G" (LightGBM alone, SHIP-ADJ, because only J_G has a verdict).
    oof=True (2025 reproduction only; needs a `fold` column) scores each row with its held-out fold's models."""
    model_dir = pathlib.Path(model_dir)
    man = json.loads((model_dir / "MANIFEST.json").read_text())
    missing = [c for c in CONTRACT_COLS + man["feats"] if c not in frame.columns]
    if missing:
        raise ValueError(f"missing contract column(s): {missing}")
    learner = man.get("ship_learner", "B")
    if learner not in ("B", "G"):
        raise ValueError(f"ship_learner must be 'B' or 'G', got {learner!r}")
    X = _design(frame, man["feats"])
    if oof and "fold" not in frame.columns:
        raise ValueError("oof=True needs a `fold` column")
    fold = frame.fold.to_numpy(float) if oof else None
    cov = frame.coverage.to_numpy()
    g = _predict_g(X, model_dir, man, cov, fold)
    b = g if learner == "G" else 0.5 * (g + _predict_n(X, model_dir, man, cov, fold))
    if oof:
        b = np.where(np.isfinite(fold), b, 0.0)

    def allowed(a):
        v = man["gate"].get(a, False)
        return bool(v["allowed"]) if isinstance(v, dict) else bool(v)
    ok = (frame.coverage.to_numpy() >= 2) & frame.ap.map(allowed).to_numpy(bool)
    return np.where(ok, b, 0.0)


# ---------------------------------------------------------------- arm ADU (plans/PREREG_adsb_unmatched_2026_09_11.md)

UNM_RECORD = ROOT / "data" / "cache_stand" / "unm_congestion_lomo.parquet"
UNM_FEATURES = ROOT / "data" / "adsb" / "v2" / "features_unm_2025janjul.parquet"
OUT_UNM = ROOT / "reports" / "adsb_unmatched.json"
UNM_ROWS_2026, BOARD_ROWS = 5_290, 344_841
UNM_C2_BAR, UNM_C4_WORSE = 150.0, 1.05
LGB_PARAMS_UNM = dict(LGB_PARAMS, min_data_in_leaf=50)


def map_unmatched(rec: pd.DataFrame) -> pd.DataFrame:
    """Taxi-scale lane onto ADN's delta machinery: delta := -y, F := -S1C, proxy := 0 (so taxi(proxy, delta) = y and
    the stacker's residual is -(y - S1C))."""
    return pd.DataFrame({"delta": -rec.y.to_numpy(float), "F": -rec.S1C.to_numpy(float),
                         "proxy": np.zeros(len(rec))}, index=rec.index)


def score_unmatched(fr: pd.DataFrame, arm_delta, base_delta, seed_rmses=None) -> dict:
    """ADU's clauses: day-block C1; C2 on the event-excluded net priced at the 2026 unmatched row count; C3 months;
    C4 >= 2 airports improve and none worsens its MSE by > 5%; C5 seeds."""
    s = np.isfinite(fr.fold.to_numpy(float))
    y, px = fr.y.to_numpy(float)[s], fr.proxy.to_numpy(float)[s]
    base, arm = taxi(px, np.asarray(base_delta, float)[s]), taxi(px, np.asarray(arm_delta, float)[s])
    days, mon, ap = fr.day.to_numpy()[s], fr.month.to_numpy()[s], fr.ap.to_numpy()[s]
    d = (base - y) ** 2 - (arm - y) ** 2
    n = int(s.sum())
    scale = UNM_ROWS_2026 / (BOARD_ROWS * n)
    grp = pd.Series(d).groupby([ap, days]).sum()
    net_full, net_ex = float(d.sum() * scale), float((d.sum() - grp.max()) * scale)
    gain = _rmse(base, y) - _rmse(arm, y)
    mg = {int(m): _rmse(base[mon == m], y[mon == m]) - _rmse(arm[mon == m], y[mon == m]) for m in np.unique(mon)}
    ratio = {a: float(np.mean((arm[ap == a] - y[ap == a]) ** 2) / max(np.mean((base[ap == a] - y[ap == a]) ** 2), 1e-9))
             for a in np.unique(ap)}
    ci = day_block_ci(y, base, arm, days)
    seed_sd = float(np.std(seed_rmses, ddof=1)) if seed_rmses is not None and len(seed_rmses) > 1 else None
    c = {"C1": bool(ci[0] > 0), "C2": bool(net_ex >= UNM_C2_BAR), "C3": bool(all(v > 0 for v in mg.values())),
         "C4": bool(sum(v < 1 for v in ratio.values()) >= 2 and all(v <= UNM_C4_WORSE for v in ratio.values())),
         "C5": None if seed_sd is None else bool(gain > 2 * seed_sd)}
    return {"rmse_base": _rmse(base, y), "rmse_arm": _rmse(arm, y), "gain_s": gain, "ci95_day_block": ci,
            "net_full": net_full, "net_event_excluded": net_ex, "top_airport_day": [str(x) for x in grp.idxmax()],
            "month_gain_s": mg, "airport_mse_ratio": ratio, "seed_rmses": seed_rmses, "seed_sd": seed_sd,
            "clauses": c, "verdict": verdict(c), "n_rows_scored": n}


OUT_ADJ = ROOT / "reports" / "adsb_join_recall.json"


def gated_prediction(pred, F, coverage, gate: dict, ap) -> np.ndarray:
    """What a VERSION actually ships: the stacker's delta where that version's coverage >= 2 and its gate allows the
    airport, F everywhere else (SHIP-ADJ.1)."""
    ok = (np.asarray(coverage) >= 2) & np.array([bool(gate.get(a, False)) for a in np.asarray(ap)])
    return np.where(ok, np.asarray(pred, float), np.asarray(F, float))


def run_adj(fr: pd.DataFrame, feats: list, seeds=SEEDS, out_json=None, ship_gates=None, next_learner: str = "G") -> int:
    """Arm ADJ (plans/PREREG_adsb_join_recall_2026_09_12.md): the ADN stacker on the EXTENDED coverage against the same
    stacker on today's coverage (the recovered rows keeping F). C1-C5 as ADN, plus C7 on the rows joined today.
    Controls act on the recovered rows only (BC-10)."""
    out_json = OUT_ADJ if out_json is None else pathlib.Path(out_json)
    if out_json.exists():
        raise SystemExit(f"refusing to overwrite {out_json}")
    now = lambda: dt.datetime.now().astimezone().strftime("%H:%M:%S")
    new = fr.join_mvt_anchored.to_numpy(dtype=float) == 1
    old_joined = (fr.coverage.to_numpy() >= 2) & ~new
    tw_fr = today_coverage(fr)
    print(f"[{now()}] ADJ rows {len(fr):,}; scored {int(np.isfinite(fr.fold).sum()):,}; joined today "
          f"{int(old_joined.sum()):,}; recovered {int(new.sum()):,}", flush=True)
    res, preds = {}, {}
    for arm in ("G", "N"):
        twin = [fit_stacker(tw_fr, feats, arm, sd) for sd in seeds]
        per = [fit_stacker(fr, feats, arm, sd) for sd in seeds]
        preds[arm], preds[arm + "_twin"] = np.mean(per, axis=0), np.mean(twin, axis=0)
        rm = [score_arm(fr, p, base_delta=preds[arm + "_twin"])["rmse_arm"] for p in per]
        r = score_arm(fr, preds[arm], seed_rmses=rm, base_delta=preds[arm + "_twin"])
        r["net_recovered_rows"] = subset_net(fr, new, preds[arm + "_twin"], preds[arm])
        r["net_rows_joined_today"] = subset_net(fr, old_joined, preds[arm + "_twin"], preds[arm])
        r["breakdown_by_coverage"] = adx_breakdown(fr, preds[arm + "_twin"], preds[arm])
        r["clauses"] = clauses_adj(r["clauses"], r["net_rows_joined_today"])
        r["verdict"] = verdict(r["clauses"])
        res[arm] = r
        print(f"[{now()}] {arm} {r['verdict']} gain {r['gain_s']:+.3f} ci {r['ci95_day_block']} net "
              f"{r['net_weighted_fold_mse']:+.0f} | recovered {r['net_recovered_rows']:+.0f} "
              f"today {r['net_rows_joined_today']:+.0f}", flush=True)
    b = score_arm(fr, 0.5 * (preds["G"] + preds["N"]), base_delta=0.5 * (preds["G_twin"] + preds["N_twin"]))
    b["net_recovered_rows"] = subset_net(fr, new, 0.5 * (preds["G_twin"] + preds["N_twin"]), 0.5 * (preds["G"] + preds["N"]))
    res["B"] = b
    if ship_gates is not None:                                   # SHIP-ADJ.1: v_next vs the CURRENT shipped version
        g_new, g_old = ship_gates
        ap = fr.ap.to_numpy()
        if next_learner not in ("G", "B"):
            raise ValueError(f"next_learner must be 'G' or 'B', got {next_learner!r}")
        nxt = preds["G"] if next_learner == "G" else 0.5 * (preds["G"] + preds["N"])
        v_next = gated_prediction(nxt, fr.F.to_numpy(float), fr.coverage.to_numpy(), g_new, ap)
        v_cur = gated_prediction(0.5 * (preds["G_twin"] + preds["N_twin"]), fr.F.to_numpy(float),
                                 tw_fr.coverage.to_numpy(), g_old, ap)
        sd = score_arm(fr, v_next, base_delta=v_cur)
        sd["net_recovered_rows"] = subset_net(fr, new, v_cur, v_next)
        sd["net_rows_joined_today"] = subset_net(fr, old_joined, v_cur, v_next)
        sd["net_LIRF"] = subset_net(fr, ap == "LIRF", v_cur, v_next)
        sd["gate_next"], sd["gate_current"], sd["next_learner"] = g_new, g_old, next_learner
        res["SHIP_v_next_vs_current"] = sd
        print(f"[{now()}] SHIP v_next vs current: gain {sd['gain_s']:+.3f} ci {sd['ci95_day_block']} net "
              f"{sd['net_weighted_fold_mse']:+.0f} | recovered {sd['net_recovered_rows']:+.0f} today "
              f"{sd['net_rows_joined_today']:+.0f} LIRF {sd['net_LIRF']:+.0f}", flush=True)
    rng = np.random.default_rng(0)
    resid = fr.delta.to_numpy(float) - fr.F.to_numpy(float)
    plant = fr.assign(planted=np.where(new, resid + rng.normal(0, 0.5 * np.nanstd(resid[new]), len(fr)), np.nan))
    kp = score_arm(fr, fit_stacker(plant, feats + ["planted"], "G", 0), base_delta=preds["G_twin"])
    k0 = score_arm(fr, fit_stacker(permute_rows(fr, feats, new, seed=0), feats, "G", 0), base_delta=preds["G_twin"])
    controls = {"K_plus": {"net": kp["net_weighted_fold_mse"], "C1": kp["clauses"]["C1"], "bar": ADJ_KPLUS_BAR,
                           "pass": bool(kp["clauses"]["C1"] and kp["net_weighted_fold_mse"] >= ADJ_KPLUS_BAR)},
                "K_zero": {"net": k0["net_weighted_fold_mse"], "ci": k0["ci95_day_block"], "rule": "ADJ.1: noise must "
                           "not win, net < +150; a negative net is the expected sign when the twin leaves these rows at F",
                           "pass": bool(k0["net_weighted_fold_mse"] < ADJ_K0_BAR)}}
    print(f"[{now()}] controls {controls}", flush=True)
    import subprocess
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out_json.write_text(json.dumps(
        {"prereg": "plans/PREREG_adsb_join_recall_2026_09_12.md (ADJ.1)", "git_sha": sha + "-dirty", "seeds": list(seeds),
         "n_recovered": int(new.sum()), "n_joined_today": int(old_joined.sum()),
         "controls": controls, "harness_valid": bool(controls["K_plus"]["pass"] and controls["K_zero"]["pass"]),
         "arms": res}, indent=1, default=float))
    return 0


OUT_ADX = ROOT / "reports" / "adsb_taxistart.json"


def run_adx(fr: pd.DataFrame, feats_twin: list, feats_arm: list, seeds=SEEDS, out_json=None) -> int:
    """Arm ADX (plans/PREREG_adsb_taxistart_2026_09_11.md): the ADN stacker WITH the taxi-start block against the same
    stacker WITHOUT it, on the same rows, folds and seeds. C1-C5 as ADN, plus C6 on code-3 rows. The breakdown and C6
    come from the SAME fit as the verdict."""
    out_json = OUT_ADX if out_json is None else pathlib.Path(out_json)
    if out_json.exists():
        raise SystemExit(f"refusing to overwrite {out_json}")
    now = lambda: dt.datetime.now().astimezone().strftime("%H:%M:%S")
    block = [c for c in feats_arm if c not in feats_twin]
    print(f"[{now()}] ADX rows {len(fr):,}; scored {int(np.isfinite(fr.fold).sum()):,}; joined "
          f"{int((fr.coverage >= 2).sum()):,}; block {block}", flush=True)
    nan_share = {c: float(fr.loc[fr.coverage >= 2, c].isna().mean()) for c in block}
    print(f"[{now()}] block NaN share on joined rows: { {k: round(v, 3) for k, v in nan_share.items()} }", flush=True)
    res, preds = {}, {}
    for arm in ("G", "N"):
        twin = [fit_stacker(fr, feats_twin, arm, sd) for sd in seeds]
        per = [fit_stacker(fr, feats_arm, arm, sd) for sd in seeds]
        preds[arm], preds[arm + "_twin"] = np.mean(per, axis=0), np.mean(twin, axis=0)
        rm = [score_arm(fr, p, base_delta=preds[arm + "_twin"])["rmse_arm"] for p in per]
        r = score_arm(fr, preds[arm], seed_rmses=rm, base_delta=preds[arm + "_twin"])
        bd = adx_breakdown(fr, preds[arm + "_twin"], preds[arm])
        r["breakdown_by_coverage"] = bd
        r["clauses"] = clauses_adx(r["clauses"], bd["3"]["net"])
        r["verdict"] = verdict(r["clauses"])
        res[arm] = r
        print(f"[{now()}] {arm} {r['verdict']} gain {r['gain_s']:+.3f} ci {r['ci95_day_block']} net "
              f"{r['net_weighted_fold_mse']:+.0f} | code2 {bd['2']['net']:+.0f} code3 {bd['3']['net']:+.0f}", flush=True)
    b = score_arm(fr, 0.5 * (preds["G"] + preds["N"]), base_delta=0.5 * (preds["G_twin"] + preds["N_twin"]))
    b["breakdown_by_coverage"] = adx_breakdown(fr, 0.5 * (preds["G_twin"] + preds["N_twin"]), 0.5 * (preds["G"] + preds["N"]))
    res["B"] = b
    rng = np.random.default_rng(0)
    resid = fr.delta.to_numpy(float) - fr.F.to_numpy(float)
    j = fr.coverage.to_numpy() >= 2
    plant = fr.assign(planted=np.where(j, resid + rng.normal(0, 0.5 * np.nanstd(resid[j]), len(fr)), np.nan))
    kp = score_arm(fr, fit_stacker(plant, feats_arm + ["planted"], "G", 0), base_delta=preds["G_twin"])
    noise = fr.copy()
    grp = noise.ap.astype(str) + "|" + noise.month.astype(str)
    for c in feats_arm:
        noise[c] = noise.groupby(grp)[c].transform(lambda x: x.sample(frac=1.0, random_state=0).to_numpy())
    k0 = score_arm(fr, fit_stacker(noise, feats_arm, "G", 0), base_delta=preds["G_twin"])
    controls = {"K_plus": {"net": kp["net_weighted_fold_mse"], "C1": kp["clauses"]["C1"],
                           "pass": bool(kp["clauses"]["C1"] and kp["net_weighted_fold_mse"] >= 5000.0)},
                "K_zero": {"net": k0["net_weighted_fold_mse"], "ci": k0["ci95_day_block"],
                           "pass": bool(k0["ci95_day_block"][0] <= 0 <= k0["ci95_day_block"][1]
                                        or abs(k0["net_weighted_fold_mse"]) < 150.0)}}
    print(f"[{now()}] controls {controls}", flush=True)
    import subprocess
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out_json.write_text(json.dumps(
        {"prereg": "plans/PREREG_adsb_taxistart_2026_09_11.md", "git_sha": sha + "-dirty",
         "block": block, "block_nan_share_joined": nan_share, "seeds": list(seeds),
         "early_stopping": "ADN.3's rule for both arms (the last training fold); each arm stops at its own round, which "
                           "is part of the comparison",
         "controls": controls, "harness_valid": bool(controls["K_plus"]["pass"] and controls["K_zero"]["pass"]),
         "arms": res}, indent=1, default=float))
    return 0


def load_unmatched_frame() -> pd.DataFrame:
    rec = pd.read_parquet(UNM_RECORD)
    rec = rec[rec.month.isin([1, 7])].reset_index(drop=True)
    feat = pd.read_parquet(UNM_FEATURES)
    if not (feat.MVT_ID_mvt.to_numpy() == rec.MVT_ID_mvt.to_numpy()).all():
        raise ValueError("unmatched feature table is not in the record's order")
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]) for p in RAW_MONTHS])
    t = pd.to_datetime(rec.MVT_ID_mvt.map(raw.drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").MVT_TIME_UTC_mvt),
                       utc=True)
    fr = pd.concat([rec[["MVT_ID_mvt", "month", "y", "sp", "S1C", "ADEP_mvt"]].rename(columns={"ADEP_mvt": "ap"}),
                    map_unmatched(rec), feat.drop(columns=["MVT_ID_mvt"])], axis=1)
    fr["day"], fr["hour"] = t.dt.strftime("%Y-%m-%d").to_numpy(), t.dt.hour.to_numpy()
    fr["fold"] = day_folds(fr.day)
    return fr


def run_unmatched(feats: list) -> int:
    if OUT_UNM.exists():
        raise SystemExit(f"refusing to overwrite {OUT_UNM}")
    fr = load_unmatched_frame()
    s = np.isfinite(fr.fold.to_numpy(float))
    guard = _rmse(np.maximum(fr.S1C[s], 1), fr.y[s])
    print(f"[ADU] rows {len(fr):,}, scored {int(s.sum()):,}, joined {int((fr.coverage >= 2).sum()):,}; S1C RMSE "
          f"{guard:.3f}", flush=True)
    cov = fr.groupby("ap").coverage.apply(lambda c: round(float((c >= 2).mean()), 3)).to_dict()
    print(f"[ADU] joined share by airport: {cov}", flush=True)
    uf = feats + ["sp"]
    res, preds = {}, {}
    for arm in ("G", "N"):
        base = [fit_stacker(fr, ["sp"], arm, sd, lgb_params=LGB_PARAMS_UNM) for sd in SEEDS]
        per = [fit_stacker(fr, uf, arm, sd, lgb_params=LGB_PARAMS_UNM) for sd in SEEDS]
        preds[arm], preds[arm + "_base"] = np.mean(per, axis=0), np.mean(base, axis=0)
        rm = [score_unmatched(fr, p, preds[arm + "_base"])["rmse_arm"] for p in per]
        res[arm] = score_unmatched(fr, preds[arm], preds[arm + "_base"], seed_rmses=rm)
        print(f"[ADU] {arm}: {res[arm]['verdict']} gain {res[arm]['gain_s']:+.2f} ci {res[arm]['ci95_day_block']} "
              f"net_ex {res[arm]['net_event_excluded']:+.0f} net {res[arm]['net_full']:+.0f}", flush=True)
    res["B"] = score_unmatched(fr, 0.5 * (preds["G"] + preds["N"]), 0.5 * (preds["G_base"] + preds["N_base"]))
    rng = np.random.default_rng(0)
    resid = fr.delta.to_numpy(float) - fr.F.to_numpy(float)
    j = fr.coverage.to_numpy() >= 2
    plant = fr.assign(planted=np.where(j, resid + rng.normal(0, 0.5 * np.nanstd(resid[j]), len(fr)), np.nan))
    kp = score_unmatched(fr, fit_stacker(plant, uf + ["planted"], "G", 0, lgb_params=LGB_PARAMS_UNM), preds["G_base"])
    noise = fr.copy()
    grp = noise.ap.astype(str) + "|" + noise.month.astype(str)
    for c in feats:
        noise[c] = noise.groupby(grp)[c].transform(lambda x: x.sample(frac=1.0, random_state=0).to_numpy())
    k0 = score_unmatched(fr, fit_stacker(noise, uf, "G", 0, lgb_params=LGB_PARAMS_UNM), preds["G_base"])
    controls = {"K_plus": {"net_full": kp["net_full"], "C1": kp["clauses"]["C1"],
                           "pass": bool(kp["clauses"]["C1"] and kp["gain_s"] > 0.2 * kp["rmse_base"])},
                "K_zero": {"net_full": k0["net_full"], "ci": k0["ci95_day_block"],
                           "pass": bool(k0["ci95_day_block"][0] <= 0 <= k0["ci95_day_block"][1]
                                        or abs(k0["net_full"]) < 50)}}
    print(f"[ADU] controls {controls}", flush=True)
    OUT_UNM.write_text(json.dumps({"prereg": "plans/PREREG_adsb_unmatched_2026_09_11.md", "coverage_joined": cov,
                                   "s1c_guard_rmse": guard, "controls": controls,
                                   "harness_valid": bool(controls["K_plus"]["pass"] and controls["K_zero"]["pass"]),
                                   "arms": res}, indent=1, default=float))
    return 0


TRANSFER_AIRPORTS = ("EHAM", "EDDM", "LSZH", "LEBL")
TRANSFER_IN_SAMPLE = {"EHAM": 31.63, "EDDM": 30.00, "LSZH": 31.49, "LEBL": 20.67}   # RESULT ADN, arm G per airport (s)
OUT_TRANSFER = ROOT / "reports" / "adsb_transfer.json"


def transfer_rule(results: dict) -> dict:
    """ADN.7: PASS iff at >= 3 of the 4 holdout airports gain >= 0.5 x in-sample gain AND the day-block CI excludes 0."""
    ok = {a: bool(r["gain_s"] >= 0.5 * TRANSFER_IN_SAMPLE[a] and r["ci95_day_block"][0] > 0) for a, r in results.items()}
    return {"per_airport_pass": ok, "n_pass": int(sum(ok.values())), "PASS": bool(sum(ok.values()) >= 3)}


def transfer_test(fr: pd.DataFrame, feats: list) -> int:
    """ADN.7's leave-one-airport-out transfer test (arm G, seed 0), written to reports/adsb_transfer.json."""
    if OUT_TRANSFER.exists():
        raise SystemExit(f"refusing to overwrite {OUT_TRANSFER}")
    res = {}
    for A in TRANSFER_AIRPORTS:
        tr, pr = (fr.ap != A).to_numpy(), (fr.ap == A).to_numpy()
        g = fit_stacker(fr, feats, "G", 0, train_rows=tr, predict_rows=pr)
        b = fit_stacker(fr, [], "G", 0, train_rows=tr, predict_rows=pr)
        sub = fr[pr].reset_index(drop=True)
        res[A] = score_arm(sub, g[pr], base_delta=b[pr])
        print(f"[transfer] {A}: gain {res[A]['gain_s']:+.3f} ci {res[A]['ci95_day_block']} "
              f"(in-sample {TRANSFER_IN_SAMPLE[A]})", flush=True)
    rule = transfer_rule(res)
    OUT_TRANSFER.write_text(json.dumps({"prereg": "ADN.7", "rule": rule, "airports": res}, indent=1, default=float))
    print(f"[transfer] {rule}", flush=True)
    return 0


def main(argv=None) -> int:
    import argparse
    import importlib.util
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "transfer", "train-artifacts", "run-unmatched", "run-adx", "run-adj"])
    ap.add_argument("--table", default=None, help="feature table (run-adx: the v3 table with the taxi-start block)")
    ap.add_argument("--out", default=None, help="output JSON (run-adx)")
    ap.add_argument("--models", default=None, help="artefact directory (train-artifacts)")
    ap.add_argument("--next-learner", default="G", choices=["G", "B"],
                    help="run-adj --ship-gates: which learner v_next ships (SHIP-ADJ used G, SHIP-ADJ2 uses B)")
    ap.add_argument("--ship-gates", default=None,
                    help="run-adj: '<next models MANIFEST.json>,<current models MANIFEST.json>' to also score the "
                         "shipped configurations against each other (SHIP-ADJ.1)")
    ap.add_argument("--ship-learner", default="B", choices=["B", "G"],
                    help="which learner the shipped manifest declares: B = mean(G, N) as v11/v12, G = LightGBM alone "
                         "(SHIP-ADJ, because only J_G has a verdict)")
    ap.add_argument("--smoke", action="store_true", help="first 5 days per month, seed 0, *_smoke outputs")
    a = ap.parse_args(argv)
    out_json = OUT_JSON.with_name("adsb_nn_smoke.json") if a.smoke else OUT_JSON
    out_preds = OUT_PREDS.with_name("adn_fold_preds_smoke.parquet") if a.smoke else OUT_PREDS
    seeds = (0,) if a.smoke else SEEDS
    spec = importlib.util.spec_from_file_location("adsb_features", ROOT / "scripts" / "adsb_features.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    feats = [c for c in fx.FEATURES]
    if a.cmd == "run" and out_json.exists() and not a.smoke:
        raise SystemExit(f"refusing to overwrite {out_json}")
    now = lambda: dt.datetime.now().astimezone().strftime("%H:%M:%S")
    if a.cmd == "run-unmatched":
        return run_unmatched(feats)
    fr = load_frame(a.table) if a.table else load_frame()
    if a.cmd == "run-adj":
        gates = None
        if a.ship_gates:                                        # SHIP-ADJ.1: "<next manifest>,<current manifest>"
            nxt, cur = a.ship_gates.split(",")
            g = lambda q: {k: (v["allowed"] if isinstance(v, dict) else bool(v))
                           for k, v in json.loads(pathlib.Path(q).read_text())["gate"].items()}
            gates = (g(nxt), g(cur))
        return run_adj(fr, feats, seeds=seeds, out_json=a.out, ship_gates=gates, next_learner=a.next_learner)
    if a.cmd == "run-adx":
        twin = [c for c in feats if c not in fx.ADX_BLOCK]
        return run_adx(fr, twin, feats, seeds=seeds, out_json=a.out)
    if a.cmd == "transfer":
        return transfer_test(fr, feats)
    if a.cmd == "train-artifacts":
        md = pathlib.Path(a.models) if a.models else ROOT / "data" / "adsb" / "v2" / "models"
        if (md / "MANIFEST.json").exists():
            raise SystemExit(f"refusing to overwrite {md}")
        man = train_artifacts(fr, feats, md, seeds=SEEDS, gate={}, ship_learner=a.ship_learner)
        print(f"[{now()}] artefacts -> {md} ({man['n_train_joined_rows']:,} joined training rows); gate empty until "
              f"`adsb_ship.py gate` runs on the 2026 table", flush=True)
        return 0
    if a.smoke:
        fr = smoke_subset(fr)
    print(f"[{now()}] frame {len(fr):,} rows; scored {int(np.isfinite(fr.fold).sum()):,}; joined "
          f"{int((fr.coverage >= 2).sum()):,}; F guard ok", flush=True)
    preds = {"F": fr.F.to_numpy(float)}
    res = {}
    import subprocess
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    def save(stage, controls=None):
        """ADN.6: the JSON is rewritten after every stage, so an abort never discards completed arms."""
        out = {"prereg": "plans/PREREG_adsb_nn_2026_09_10.md (ADN.1-ADN.6)", "smoke": bool(a.smoke),
               "git_sha": sha + "-dirty", "features_version": fx.FEATURES_VERSION, "stage": stage,
               "written": dt.datetime.now().astimezone().isoformat(), "controls": controls,
               "harness_valid": None if controls is None else
               bool(controls["K_plus"]["pass"] and controls["K_zero"]["pass"]), "arms": res}
        out_json.write_text(json.dumps(out, indent=1, default=float))
        return out
    preds["S"] = arm_s(fr)
    res["S"] = score_arm(fr, preds["S"])
    res["S"]["status"] = "CLOSED FAILED at the instrument control (ADN.2/ADN.4); reported, not a candidate"
    print(f"[{now()}] S {res['S']['verdict']} gain {res['S']['gain_s']:+.3f}", flush=True)
    for arm in ("G", "N"):
        base_per = [fit_stacker(fr, [], arm, s) for s in seeds]            # ADN.5: the same stacker without ADS-B
        per = [fit_stacker(fr, feats, arm, s) for s in seeds]
        for s, p, q in zip(seeds, per, base_per):
            preds[f"{arm}_seed{s}"], preds[f"{arm}_base_seed{s}"] = p, q
        preds[arm], preds[f"{arm}_base"] = np.mean(per, axis=0), np.mean(base_per, axis=0)
        rm = [score_arm(fr, p, base_delta=preds[f"{arm}_base"])["rmse_arm"] for p in per]
        res[arm] = score_arm(fr, preds[arm], seed_rmses=rm, base_delta=preds[f"{arm}_base"])
        res[f"{arm}_vs_F_context"] = score_arm(fr, preds[arm])
        res[f"{arm}_base_vs_F_context"] = score_arm(fr, preds[f"{arm}_base"])
        print(f"[{now()}] {arm} vs {arm}_base: {res[arm]['verdict']} gain {res[arm]['gain_s']:+.3f} "
              f"ci {res[arm]['ci95_day_block']} net {res[arm]['net_weighted_fold_mse']:+.0f}", flush=True)
        save(f"after {arm}")
    preds["B"], preds["B_base"] = 0.5 * (preds["G"] + preds["N"]), 0.5 * (preds["G_base"] + preds["N_base"])
    res["B"] = score_arm(fr, preds["B"], base_delta=preds["B_base"])
    # harness controls (ADN.3)
    rng = np.random.default_rng(0)
    resid = fr.delta.to_numpy(float) - fr.F.to_numpy(float)
    joined = fr.coverage.to_numpy() >= 2
    plant = fr.copy()
    plant["planted"] = np.where(joined, resid + rng.normal(0, 0.5 * np.nanstd(resid[joined]), len(fr)), np.nan)
    g_base0 = preds["G_base_seed0"]
    kplus = score_arm(fr, fit_stacker(plant, feats + ["planted"], "G", 0), base_delta=g_base0)
    noise = fr.copy()
    grp = noise.ap.astype(str) + "|" + noise.month.astype(str)
    for c in feats:
        noise[c] = noise.groupby(grp)[c].transform(lambda s: s.sample(frac=1.0, random_state=0).to_numpy())
    k0 = score_arm(fr, fit_stacker(noise, feats, "G", 0), base_delta=g_base0)
    controls = {"K_plus": {"net": kplus["net_weighted_fold_mse"], "C1": kplus["clauses"]["C1"],
                           "pass": bool(kplus["net_weighted_fold_mse"] >= 5000 and kplus["clauses"]["C1"])},
                "K_zero": {"net": k0["net_weighted_fold_mse"], "ci": k0["ci95_day_block"],
                           "pass": bool(k0["ci95_day_block"][0] <= 0 <= k0["ci95_day_block"][1]
                                        or abs(k0["net_weighted_fold_mse"]) < 150)}}
    print(f"[{now()}] controls {controls}", flush=True)
    out = save("finished", controls)
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.Table.from_pandas(pd.DataFrame(preds).assign(row=fr.row.to_numpy()),
                             preserve_index=False)
    t = t.replace_schema_metadata({**(t.schema.metadata or {}),
                                   b"convention": b"DELTA scale; recover taxi with max(proxy - col, 1)"})
    pq.write_table(t, out_preds)
    print(f"[{now()}] wrote {out_json} and {out_preds}; harness_valid={out['harness_valid']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
