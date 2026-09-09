"""Is the LEARNER the bottleneck? HGB control vs LightGBM, encoded and native-categorical.

Pre-registered in plans/PREREG_taxiout_2026_09_08.md Amendment 7.

    python3.11 scripts/lgbm_ab.py [--smoke]

Identical cached rows, fold, delta target, scoring code and leakage guards as stand_ab.py.
Early stopping uses two months carved from the TRAINING fold, never the held-out Jan+Jul.
"""
from __future__ import annotations

import argparse, gc, glob, json, pathlib
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor as HGR

import importlib.util
ROOT = pathlib.Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("stand_ab", ROOT / "scripts" / "stand_ab.py")
S = importlib.util.module_from_spec(_s); _s.loader.exec_module(S)

HOLDOUT, ES_MONTHS = (1, 7), (3, 9)      # ES months come out of the TRAINING fold
STRATUM = 1527.2                          # held fixed; this experiment touches matched rows only
W_M, W_U = 339551 / 344841, 5290 / 344841
CATS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt",
        "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt",
        "FLIGHT_TYPE_flt", "stand_pref", "airline", "ars"]


def total_rmse(matched_rmse):
    return float(np.sqrt(W_M * matched_rmse ** 2 + W_U * STRATUM ** 2))


def main(smoke: bool):
    d = pd.concat([pd.read_parquet(p) for p in
                   sorted(glob.glob(str(S.CACHE / "training_2025-*.parquet")))],
                  ignore_index=True)
    if smoke:
        d = d[d.month.isin([1, 2, 3, 9])].reset_index(drop=True)
    te = d.month.isin(HOLDOUT).to_numpy()
    tr = ~te
    es = tr & d.month.isin(ES_MONTHS).to_numpy()
    fit = tr & ~es
    print(f"rows {len(d):,}   fit {fit.sum():,}   early-stop {es.sum():,}   test {te.sum():,}")
    assert not (es & te).any(), "early-stopping months overlap the held-out fold"

    y, proxy, dlt = d.y.to_numpy(), d.proxy.to_numpy(), d.delta.to_numpy()
    for col, vals in S.infold_encodings(d, y, dlt, tr).items():
        d[col] = vals
    d["stand_slack_med"], d["stand_slack_iqr"] = S.infold_slack_stats(
        d.gapa.to_numpy(), dlt, d.ars.to_numpy(), tr)

    def score(pred_delta, mask):
        raw = proxy[mask] - pred_delta
        assert np.isfinite(raw).all(), "non-finite predictions"
        bound = float((raw < 1.0).mean())
        assert bound < 0.005, f"positivity floor binds on {100 * bound:.2f}% of rows"
        pred = np.maximum(raw, 1.0)
        return float(np.sqrt(((y[mask] - pred) ** 2).mean())), y[mask] - pred

    feats = S.BASELINE_FEATS
    X = np.empty((len(d), len(feats)), dtype=np.float32)
    for i, c in enumerate(feats):
        X[:, i] = pd.to_numeric(d[c], errors="coerce").to_numpy(np.float32)

    out, res = {}, {}

    # ---- control: the learner every experiment in this project has used ----
    m = HGR(max_iter=60 if smoke else 400, learning_rate=0.05, max_leaf_nodes=127,
            min_samples_leaf=20, random_state=0, early_stopping=False).fit(X[tr], dlt[tr])
    out["hgb_control"], res["hgb_control"] = score(m.predict(X[te]), te)
    print(f"  hgb_control        {out['hgb_control']:8.2f}   (expect ~235.6)")
    del m; gc.collect()

    P = dict(objective="regression", metric="rmse", learning_rate=0.05 if smoke else 0.01,
             num_leaves=255, min_data_in_leaf=40, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, cat_smooth=20, min_data_per_group=100,
             max_cat_threshold=64, num_threads=4, verbosity=-1, seed=0)
    NEST = 400 if smoke else 50_000
    CB = [lgb.early_stopping(50 if smoke else 200, verbose=False), lgb.log_evaluation(0)]

    # ---- arm 2: LightGBM on the identical encoded matrix ----
    ds = lgb.Dataset(X[fit], dlt[fit])
    b = lgb.train(P, ds, num_boost_round=NEST, valid_sets=[lgb.Dataset(X[es], dlt[es])],
                  callbacks=CB)
    out["lgb_enc"], res["lgb_enc"] = score(b.predict(X[te], num_iteration=b.best_iteration), te)
    bi_enc = b.best_iteration
    print(f"  lgb_enc            {out['lgb_enc']:8.2f}   best_iter {bi_enc:,}")
    del ds, b; gc.collect()

    # ---- arm 3: native categoricals, target encodings dropped ----
    num = [c for c in feats if not c.startswith(("te_", "de_"))]
    Xn = np.empty((len(d), len(num) + len(CATS)), dtype=np.float32)
    for i, c in enumerate(num):
        Xn[:, i] = pd.to_numeric(d[c], errors="coerce").to_numpy(np.float32)
    for j, c in enumerate(CATS):
        Xn[:, len(num) + j] = pd.factorize(d[c].astype(str))[0].astype(np.float32)
    cidx = list(range(len(num), len(num) + len(CATS)))
    print(f"    native arm: {len(num)} numeric + {len(CATS)} categorical "
          f"(STAND_mvt levels: {d.STAND_mvt.nunique():,})")
    ds = lgb.Dataset(Xn[fit], dlt[fit], categorical_feature=cidx, free_raw_data=False)
    b = lgb.train(P, ds, num_boost_round=NEST,
                  valid_sets=[lgb.Dataset(Xn[es], dlt[es], categorical_feature=cidx,
                                          reference=ds)], callbacks=CB)
    out["lgb_native"], res["lgb_native"] = score(
        b.predict(Xn[te], num_iteration=b.best_iteration), te)
    bi_nat = b.best_iteration
    print(f"  lgb_native         {out['lgb_native']:8.2f}   best_iter {bi_nat:,}")
    del ds, b, Xn; gc.collect()

    # ---- arm 4: refit on ALL training months at the discovered iteration count ----
    # early stopping costs two months of training data; the control does not pay that, so a
    # like-for-like comparison refits on the full training fold at best_iter scaled by the
    # data ratio. This is what the prior-year winners' pipelines do.
    scale = tr.sum() / fit.sum()
    n_ref = max(50, int(bi_enc * scale))
    print(f"    refit arm: {n_ref:,} trees on all {tr.sum():,} training rows "
          f"(best_iter {bi_enc:,} x {scale:.2f})")
    ds = lgb.Dataset(X[tr], dlt[tr])
    b = lgb.train(P, ds, num_boost_round=n_ref, callbacks=[lgb.log_evaluation(0)])
    out["lgb_refit"], res["lgb_refit"] = score(b.predict(X[te]), te)
    print(f"  lgb_refit          {out['lgb_refit']:8.2f}   {n_ref:,} trees, no early stop")
    del ds, b; gc.collect()

    base = out["hgb_control"]
    print("\n" + "=" * 96)
    print("AMENDMENT 7 RESULT — fold A, matched rows, stratum held fixed at 1527.2")
    print("=" * 96)
    print(f"{'arm':16s}{'matched':>10s}{'vs control':>12s}{'%':>8s}{'total':>9s}"
          f"{'MSE removed':>14s}{'resid corr':>12s}")
    for k, v in out.items():
        pct = 100 * (base - v) / base
        dmse = W_M * (base ** 2 - v ** 2)
        rc = np.corrcoef(res["hgb_control"], res[k])[0, 1]
        print(f"{k:16s}{v:10.2f}{base - v:+12.2f}{pct:+7.1f}%{total_rmse(v):9.2f}"
              f"{dmse:+14,.0f}{rc:12.3f}")
    best = min(out, key=out.get)
    pct = 100 * (base - out[best]) / base
    band = ("PIVOT (>=10%)" if pct >= 10 else "MAJOR (5-10%)" if pct >= 5
            else "USEFUL (2-5%)" if pct >= 2 else "NOT THE MECHANISM (<2%)")
    print(f"\nbest arm: {best}  {pct:+.1f}%  ->  Amendment 7 band: {band}")

    ap = d.ap.to_numpy()[te]
    print(f"\n{'airport':10s}{'n':>9s}" + "".join(f"{k:>14s}" for k in out))
    for a in sorted(set(ap)):
        m_ = ap == a
        print(f"{a:10s}{m_.sum():9,d}" + "".join(
            f"{np.sqrt((res[k][m_] ** 2).mean()):14.1f}" for k in out))
    tail = (dlt[te] < -600)
    print(f"\n{'delta<-600':10s}{tail.sum():9,d}" + "".join(
        f"{np.sqrt((res[k][tail] ** 2).mean()):14.1f}" for k in out))
    (ROOT / "reports" / "lgbm_ab_result.json").write_text(json.dumps(
        {"matched": out, "best_iter": {"lgb_enc": bi_enc, "lgb_native": bi_nat}}, indent=2))


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--smoke", action="store_true")
    main(a.parse_args().smoke)
