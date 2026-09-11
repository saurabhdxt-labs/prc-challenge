#!/usr/bin/env python3.11
"""Arm REG — a regime-gated multi-model matched lane (plans/PREREG_regime_experts_2026_09_11.md, SCREEN tier).

    OMP_NUM_THREADS=1 nice -n 10 python3.11 -B -u scripts/regime_experts.py --smoke   # 3 months, proves it runs
    OMP_NUM_THREADS=1 nice -n 10 python3.11 -B -u scripts/regime_experts.py --full    # fold A, ~1 h, ~5 GB

The matched rows come from four data-generating situations (delta = BLOCK − AOBT_3): the clocks agree, the airport
stamped the schedule (fill), the airport's off-block is > 10 min BEFORE NM's (early), or > 10 min after (late). One
global model averages them. Here a four-class gate p_k(x) and one expert per regime, each fitted ONLY on its own rows
(no all-rows expert: RESULT 8's double counting), combine as REG = max(Σ_k p_k μ_k, 1), paired against BASE — one delta
model on all rows with the same encodings, params and seed — so a gain is the architecture, not the settings.
Generalises scripts/cond_experts.py (fill / body) and reuses its pieces and enc_ab's BC-1-clean SEPARATED encodings.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cond_experts as CE  # noqa: E402  (fill_correction; loads enc_ab -> lgbm_fold, lgbm_submit, stand_ab)
import enc_ab as EA  # noqa: E402
import nm_param_diag as N  # noqa: E402  (holdout_days, dayblock_sum, to_board)

LF, L = EA.LF, EA.L
REGIMES = ("agree", "fill", "early", "late")
TAIL_S = 600.0
C2_NET, C3_AGREE_FLOOR, C4_AIRPORTS = 1_000.0, -300.0, 7
F_RMSE = 222.56316251601282
EXPECTED_HOLDOUT = {"agree": 285_942, "fill": 30_167, "early": 12_754, "late": 10_152}
PREREG = "plans/PREREG_regime_experts_2026_09_11.md"
CONVENTION = "taxi_time"
_t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} + {time.time() - _t0:7.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- pure pieces (unit-tested)

def regime_label(y, sp, dlt) -> np.ndarray:
    """0 agree, 1 fill, 2 early, 3 late; fill takes precedence. NaN in any input on a row raises."""
    y, sp, dlt = (np.asarray(v, dtype="float64") for v in (y, sp, dlt))
    if not (np.isfinite(y).all() and np.isfinite(sp).all() and np.isfinite(dlt).all()):
        raise ValueError("regime_label needs finite y, sp and delta on every row it labels")
    fill = np.abs(y - sp) <= L.FILL_TOL_S
    return np.select([fill, dlt < -TAIL_S, dlt > TAIL_S], [1, 2, 3], default=0).astype("int64")


def expert_taxi(proxy, delta_hat) -> tuple:
    """(max(proxy − δ̂, 1), floor-bind count). The floor binds legitimately for a tail expert applied to every row
    (the late expert on short proxies); the mixture weight on those rows is what decides, so no share assert here."""
    raw = np.asarray(proxy, dtype="float64") - np.asarray(delta_hat, dtype="float64")
    if not np.isfinite(raw).all():
        raise ValueError(f"{int((~np.isfinite(raw)).sum())} non-finite expert predictions")
    return np.maximum(raw, 1.0), int((raw < 1.0).sum())


def assemble_mixture(P, MU) -> np.ndarray:
    """max(Σ_k P[:, k] · MU[:, k], 1). Refuses shape mismatch, non-finite values, negative or non-normalised rows."""
    P, MU = np.asarray(P, dtype="float64"), np.asarray(MU, dtype="float64")
    if P.shape != MU.shape or P.ndim != 2 or P.shape[1] != len(REGIMES):
        raise ValueError(f"P {P.shape} and MU {MU.shape} must both be (n, {len(REGIMES)})")
    for name, v in (("P", P), ("MU", MU)):
        if not np.isfinite(v).all():
            raise ValueError(f"{int((~np.isfinite(v)).sum())} non-finite values in {name}")
    if (P < 0).any() or not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("gate probabilities must be non-negative and sum to 1 per row")
    return np.maximum((P * MU).sum(axis=1), 1.0)


def one_vs_rest_auc(P, lab) -> dict:
    from sklearn.metrics import roc_auc_score
    P, lab = np.asarray(P, dtype="float64"), np.asarray(lab)
    out = {}
    for k, name in enumerate(REGIMES):
        pos = lab == k
        out[name] = float(roc_auc_score(pos, P[:, k])) if 0 < pos.sum() < len(pos) else float("nan")
    return out


def reliability(p, pos, bins: int = 10) -> list:
    """Deciles of p: mean p against the observed rate (the gate is uncalibrated by design; this shows how far)."""
    p, pos = np.asarray(p, dtype="float64"), np.asarray(pos, dtype=bool)
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    return [{"bin": int(b), "n": int((idx == b).sum()), "mean_p": float(p[idx == b].mean()),
             "rate": float(pos[idx == b].mean())} for b in range(len(q) - 1) if (idx == b).any()]


def score(y, base, reg, regime, ap, month, day, n_fold: int | None = None) -> dict:
    """The locked clauses of REG vs BASE. g = SE(BASE) − SE(REG) per row; weighted fold MSE = N.to_board(Σg, n)."""
    y, base, reg = (np.asarray(v, dtype="float64") for v in (y, base, reg))
    regime, ap, month = np.asarray(regime), np.asarray(ap).astype(str), np.asarray(month)
    n = int(n_fold if n_fold is not None else len(y))
    g = (y - base) ** 2 - (y - reg) ** 2
    tot, lo, hi, nd = N.dayblock_sum(g, day, month)
    by_regime = {name: N.to_board(float(g[regime == k].sum()), n) for k, name in enumerate(REGIMES)}
    by_airport = {a: N.to_board(float(g[ap == a].sum()), n) for a in sorted(set(ap))}
    by_month = {int(m): N.to_board(float(g[month == m].sum()), n) for m in sorted(set(month.tolist()))}
    net = N.to_board(tot, n)
    clauses = {"C1_dayblock_lower_gt_0": bool(lo > 0),
               "C2_net_ge_1000": bool(net >= C2_NET),
               "C3_agree_net_ge_minus_300": bool(by_regime["agree"] >= C3_AGREE_FLOOR),
               "C4_airports_ge_7": bool(sum(v > 0 for v in by_airport.values()) >= C4_AIRPORTS)}
    return {"net_weighted_fold_mse": net, "dayblock95": [N.to_board(lo, n), N.to_board(hi, n)], "n_dates": nd,
            "rmse_base": float(np.sqrt(((y - base) ** 2).mean())), "rmse_reg": float(np.sqrt(((y - reg) ** 2).mean())),
            "by_regime": by_regime, "by_airport": by_airport, "by_month": by_month,
            "airports_improving": int(sum(v > 0 for v in by_airport.values())), "clauses": clauses,
            "verdict": "SHORTLIST" if all(clauses.values()) else "NOT SHORTLISTED"}


# --------------------------------------------------------------------------- fitting (smoke-tested, run once)

def fit_gate(X, lab, fit, es, tr, te, params, nest, patience, swap_to_refit=None) -> tuple:
    """Four-class gate: early stop on the ES rows (multi_logloss), refit on all training rows at n_refit, predict the
    holdout. `lab` is read on training rows only."""
    import lightgbm as lgb
    cp = dict(params, objective="multiclass", num_class=len(REGIMES), metric="multi_logloss")
    cp.pop("alpha", None)
    dfit = lgb.Dataset(X[fit], label=lab[fit], free_raw_data=True)
    des = lgb.Dataset(X[es], label=lab[es], reference=dfit, free_raw_data=True)
    b = lgb.train(cp, dfit, num_boost_round=nest, valid_sets=[des],
                  callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
    best = int(b.best_iteration)
    if best < 1:
        raise RuntimeError(f"gate best_iteration {best}")
    n_ref = int(L.n_refit(best, int(tr.sum()), int(fit.sum())))
    del dfit, des, b
    gc.collect()
    if swap_to_refit is not None:
        swap_to_refit()
    rb = lgb.train(cp, lgb.Dataset(X[tr], label=lab[tr], free_raw_data=True), num_boost_round=n_ref)
    P = np.asarray(rb.predict(X[te], num_iteration=n_ref), dtype="float64")
    del rb
    gc.collect()
    return P, {"best_iter": best, "n_ref": n_ref}


def load_base(months) -> dict:
    """Arm F's rows: stand cache + queue + order blocks joined positionally (lgbm_fold._load_months), fold A masks, the
    encoding keys kept for enc_ab's separated encoder (as enc_ab.load_base, plus the order block)."""
    feats_num = [f for f in LF.FEATS_QUEUE_ORDER if not f.startswith(("te_", "de_"))]
    frames, paths = LF._load_months(LF.CACHE, months, feats_num, LF.QCACHE, ocache=LF.OCACHE)
    d = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    month = d.month.to_numpy()
    te, tr, fit, es = LF.fold_masks(month)
    ap_code, airports = L.airport_codes(d.ap)
    log(f"loaded {len(paths)} months ({len(d):,} rows): holdout {te.sum():,}; training {tr.sum():,}; "
        f"fit {fit.sum():,}; early-stop {es.sum():,}")
    return dict(d=d, month=month, te=te, tr=tr, fit=fit, es=es, ap_code=ap_code, airports=airports,
                y=d.y.to_numpy(), dlt=d.delta.to_numpy(), proxy=d.proxy.to_numpy(), sp=d.sp.to_numpy(),
                feats_num=feats_num)


def run(base: dict, params: dict, nest: int, patience: int, seed: int) -> dict:
    s1, s2, enc = EA.encoding_columns("separated", base["d"], base["y"], base["dlt"], base["month"],
                                      base["fit"], base["es"], base["tr"])
    feats = list(base["feats_num"]) + enc
    d = base["d"].copy()
    for c, v in s2.items():
        d[c] = v
    X = L.design_matrix(d, feats)
    del d
    gc.collect()
    te, tr, fit, es = base["te"], base["tr"], base["fit"], base["es"]
    y, dlt, proxy, sp = base["y"], base["dlt"], base["proxy"], base["sp"]
    lab = np.full(len(y), -1, dtype="int64")
    lab[tr] = regime_label(y[tr], sp[tr], dlt[tr])
    lab_te = regime_label(y[te], sp[te], dlt[te])
    log(f"design {X.shape[0]:,} x {X.shape[1]}; training regimes "
        f"{ {n: int((lab[tr] == k).sum()) for k, n in enumerate(REGIMES)} }; holdout "
        f"{ {n: int((lab_te == k).sum()) for k, n in enumerate(REGIMES)} }")
    out = {"n_features": int(X.shape[1]), "holdout_regimes": {n: int((lab_te == k).sum()) for k, n in enumerate(REGIMES)}}

    def delta_arm(name, rows):
        EA.write_enc(X, feats, s1)
        info = L.early_stop(X, dlt, y, proxy, tr & rows, fit & rows, es & rows, dict(params, seed=L.ES_SEED), nest, patience)
        EA.write_enc(X, feats, s2)
        pred = L.fit_seeds(X, L.regression_label(dlt, y), tr & rows, X[te], params, int(info["n_ref"]), (seed,),
                           n_features=X.shape[1])[seed]
        mu, binds = expert_taxi(proxy[te], pred)
        log(f"[{name}] rows {int((tr & rows).sum()):,}; best_iter {info['best_iter']:,} -> n_ref {int(info['n_ref']):,}; "
            f"floor binds on {binds:,} holdout rows")
        return mu, {"best_iter": int(info["best_iter"]), "n_ref": int(info["n_ref"]), "n_train": int((tr & rows).sum()),
                    "floor_binds_holdout": binds}

    everyone = np.ones(len(y), dtype=bool)
    base_pred, out["BASE"] = delta_arm("BASE all rows", everyone)
    mus = {}
    for k, name in enumerate(REGIMES):
        if name == "fill":
            continue
        mus[name], out[f"expert_{name}"] = delta_arm(f"expert {name}", lab == k)
    corr = CE.fill_correction(base["ap_code"], y, sp, tr & (lab == 1), len(base["airports"]))
    mus["fill"] = np.maximum(sp[te] + corr[base["ap_code"][te]], 1.0)
    out["fill_correction_s"] = {a: float(c) for a, c in zip(base["airports"], corr)}
    EA.write_enc(X, feats, s1)
    P, out["gate"] = fit_gate(X, lab, fit, es, tr, te, params, nest, patience,
                              swap_to_refit=lambda: EA.write_enc(X, feats, s2))
    del X
    gc.collect()
    MU = np.column_stack([mus[n] for n in REGIMES])
    reg = assemble_mixture(P, MU)
    oracle = assemble_mixture(np.eye(len(REGIMES))[lab_te], MU)
    out["gate"]["auc_one_vs_rest"] = one_vs_rest_auc(P, lab_te)
    out["gate"]["mean_p"] = {n: float(P[:, k].mean()) for k, n in enumerate(REGIMES)}
    out["gate"]["reliability_early"] = reliability(P[:, 2], lab_te == 2)
    log(f"gate AUC one-vs-rest {out['gate']['auc_one_vs_rest']}; mean p {out['gate']['mean_p']}")
    te_idx = np.flatnonzero(te)
    frame = pd.DataFrame({"row": te_idx.astype("int64"), "month": base["month"][te].astype("int32"),
                          "ap": np.asarray(base["airports"], dtype=object)[base["ap_code"][te]].astype(str),
                          "y": y[te], "proxy": proxy[te], "delta": dlt[te], "sp": sp[te], "regime": lab_te,
                          "BASE": base_pred, "REG": reg, "ORACLE_GATE_NOT_A_RESULT": oracle,
                          **{f"mu_{n}": mus[n] for n in REGIMES}, **{f"p_{n}": P[:, k] for k, n in enumerate(REGIMES)}})
    return out, frame


def write_preds(frame: pd.DataFrame, path: pathlib.Path, meta: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.Table.from_pandas(frame, preserve_index=False)
    md = {**(t.schema.metadata or {}), b"convention": CONVENTION.encode(), b"prereg": PREREG.encode(), b"tag": b"REG",
          b"meta": json.dumps(meta, default=str).encode()}
    pq.write_table(t.replace_schema_metadata(md), path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--full", action="store_true")
    g.add_argument("--smoke", action="store_true", help="three months, 60 rounds: proves the code runs; never a result")
    a = ap.parse_args(argv)
    if a.smoke:
        months, nest, patience = LF.SMOKE["months"], LF.SMOKE["nest"], LF.SMOKE["patience"]
        outdir = ROOT / "data" / "smoke_regime_experts"
    else:
        months, nest, patience = tuple(range(1, 13)), 60_000, 400
        outdir = LF.CACHE
    params = dict(L.P, learning_rate=0.05)
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"mode {'smoke' if a.smoke else 'full'}; lr {params['learning_rate']}; seed 0; months {list(months)}")
    base = load_base(months)
    res, frame = run(base, params, nest, patience, seed=0)
    del base
    gc.collect()
    day = N.holdout_days(frame.row.to_numpy(), frame.month.to_numpy(), frame.y.to_numpy())
    if not a.smoke:
        f = pd.read_parquet(LF.CACHE / "fold_preds_queue_order.parquet", columns=["row", "y", "proxy", "treatment"])
        if not (np.array_equal(f.row.to_numpy(), frame.row.to_numpy()) and np.array_equal(f.y.to_numpy(), frame.y.to_numpy())
                and np.array_equal(f.proxy.to_numpy(), frame.proxy.to_numpy())):
            raise AssertionError("holdout rows differ from arm F's record")
        if res["holdout_regimes"] != EXPECTED_HOLDOUT:
            raise AssertionError(f"holdout regimes {res['holdout_regimes']} != the registered {EXPECTED_HOLDOUT}")
        F = np.maximum(f.proxy.to_numpy() - f.treatment.to_numpy(), 1.0)
        if abs(LF.rmse(frame.y, F) - F_RMSE) > 5e-4:
            raise AssertionError("arm F does not reproduce")
        frame["F"] = F
    sc = score(frame.y, frame.BASE, frame.REG, frame.regime, frame.ap, frame.month, day)
    res["score_REG_vs_BASE"] = sc
    res["oracle_gate_NOT_A_RESULT"] = score(frame.y, frame.BASE, frame.ORACLE_GATE_NOT_A_RESULT, frame.regime, frame.ap,
                                             frame.month, day)
    if "F" in frame:
        res["context_vs_F"] = {"REG_vs_F": score(frame.y, frame.F, frame.REG, frame.regime, frame.ap, frame.month, day),
                               "BASE_vs_F": score(frame.y, frame.F, frame.BASE, frame.regime, frame.ap, frame.month, day)}
    name = "smoke" if a.smoke else "full"
    rec = {"tag": "REG", "prereg": PREREG, "mode": name, "git": LF._git(), "months": list(months),
           "params": params, "wall_s": round(time.time() - _t0, 1), "peak_rss_gb": L.peak_rss_gb(), "results": res,
           "note": "smoke magnitudes are never a result" if a.smoke else "SCREEN tier: one seed, one fold"}
    write_preds(frame, outdir / f"regime_experts_preds{'_smoke' if a.smoke else ''}.parquet", {"mode": name})
    jpath = (outdir if a.smoke else ROOT / "reports") / f"regime_experts_{name}.json"
    jpath.write_text(json.dumps(rec, indent=1, default=LF._jsonable))
    log(f"REG vs BASE: net {sc['net_weighted_fold_mse']:+.1f} {sc['dayblock95']} regimes "
        f"{ {k: round(v) for k, v in sc['by_regime'].items()} } -> {sc['verdict']}; json {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
