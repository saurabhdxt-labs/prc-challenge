"""RWC: does ADN's 2025 gain survive the 2026 ADS-B feature shift? (plans/PREREG_reweight_check_2026_09_11.md)

    /opt/homebrew/bin/python3.11 scripts/reweight_check.py        # K0, K+, then the verdict at EDDM / EDDF / LEBL / LSZH

2025 rows are fold A's scored matched rows with ADS-B joined (coverage >= 2). Their per-row ADN gain comes from the frozen
record (adn_fold_preds B vs F). Each airport's 2025 rows are reweighted to its 2026 joined rows by a cross-fitted
LightGBM density ratio on the three key features plus the coverage code. No 2026 label exists or is read.
Writes reports/rwc.json; refuses to overwrite it.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEY = ["first_ground_rel_aobt3", "first_move_rel_mvt", "first_ground_gs"]
XCOLS = KEY + ["coverage"]
D = ("EDDM", "EDDF", "LEBL", "LSZH")
REPORT_ONLY = ("EHAM", "EGLL", "LEMD", "LFPG")
KPLUS_MIN_ROWS = 1000
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=1000, verbosity=-1, num_threads=4,
              deterministic=True, force_row_wise=True, seed=0)
ROUNDS, FOLDS = 100, 5                          # RWC.2: the less overfit classifier (synthetic calibration)
ESS_MIN, TRANSFER, K0_AUC, K0_ESS, KPLUS_MIN_PLANT = 0.20, 0.5, 0.55, 0.90, 0.02
OUT = ROOT / "reports" / "rwc.json"


# ---------------------------------------------------------------- the machinery

def crossfit_ratio(X_target: pd.DataFrame, X_base: pd.DataFrame, seed: int = 0, n_folds: int = FOLDS) -> tuple:
    """(weights on the base rows, mean 1; cross-fitted AUC). Every row is scored by the fold model that did not see it;
    w = p / (1 - p) x n_base / n_target, normalised. Refuses non-finite weights (a p of exactly 1)."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    X = pd.concat([X_target[XCOLS], X_base[XCOLS]], ignore_index=True).astype("float64")
    lab = np.r_[np.ones(len(X_target)), np.zeros(len(X_base))]
    fold = np.random.default_rng(seed).permutation(len(X)) % n_folds
    p = np.empty(len(X))
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], lab[tr]), ROUNDS)
        p[te] = m.predict(X[te])
    pb = p[len(X_target):]
    w = pb / (1.0 - pb) * len(X_base) / len(X_target)
    if not np.isfinite(w).all():
        raise FloatingPointError(f"{int((~np.isfinite(w)).sum())} non-finite weights (classifier p = 1)")
    return w / w.mean(), float(roc_auc_score(lab, p))


def kish_ess(w) -> float:
    w = np.asarray(w, dtype="float64")
    return float(w.sum() ** 2 / (w ** 2).sum())


def weighted_gain(delta, w) -> float:
    delta, w = np.asarray(delta, dtype="float64"), np.asarray(w, dtype="float64")
    if not (np.isfinite(delta).all() and np.isfinite(w).all()):
        raise ValueError("non-finite gain or weight")
    return float((w * delta).sum() / w.sum())


def normal_score(x, ap) -> np.ndarray:
    """Within each airport, the normal score of x's rank ((r - 0.5) / n); 0 where x is missing."""
    from scipy.special import ndtri
    x, ap = np.asarray(x, dtype="float64"), np.asarray(ap)
    z = np.zeros(len(x))
    for a in pd.unique(ap):
        idx = np.flatnonzero((ap == a) & np.isfinite(x))
        if len(idx):
            r = pd.Series(x[idx]).rank(method="average").to_numpy()
            z[idx] = ndtri((r - 0.5) / len(idx))
    return z


# ---------------------------------------------------------------- the registered rules

def verdict(G, Gw, lo, ess, n) -> dict:
    """V1 ESS >= 0.2 n (else INCONCLUSIVE); C1 day-block lower bound > 0; C2 Gw >= 0.5 G. SHIP iff all three."""
    c = {"V1": bool(ess >= ESS_MIN * n), "C1": bool(lo > 0), "C2": bool(Gw >= TRANSFER * G)}
    if not c["V1"]:
        return {**c, "decision": "INCONCLUSIVE"}
    return {**c, "decision": "SHIP" if c["C1"] and c["C2"] else "KEEP_F"}


def k0_rule(auc, ess_frac, inside) -> str:
    """PASS iff AUC <= 0.55, ESS >= 0.90 n and Gw inside the unweighted interval. RWC.1 names a failure on ESS alone."""
    auc_ok, ess_ok = auc <= K0_AUC, ess_frac >= K0_ESS
    if auc_ok and ess_ok and inside:
        return "PASS"
    return "FAIL_ESS_ONLY" if auc_ok and inside and not ess_ok else "FAIL"


def kplus_rule(G, Gw, Gtrue) -> str:
    planted = abs(G - Gtrue)
    if planted < KPLUS_MIN_PLANT * abs(G):
        return "INCONCLUSIVE"
    return "PASS" if abs(Gw - Gtrue) <= 0.5 * planted else "FAIL"


# ---------------------------------------------------------------- data

def load_2025() -> pd.DataFrame:
    """Fold A's scored matched rows with coverage >= 2: ap, day, month, gain (unrounded, B vs F), the key features."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import adsb_stack as A
    c = ROOT / "data" / "cache_stand"
    rec = pd.read_parquet(c / "fold_preds_queue_order.parquet", columns=["row", "month", "ap", "y", "proxy"])
    cache = pd.concat([pd.read_parquet(p, columns=["doy"]) for p in sorted(c.glob("training_2025-*.parquet"))], ignore_index=True)
    if len(cache) != 2_062_440:
        raise ValueError(f"stand caches hold {len(cache):,} rows, not arm F's fold (2,062,440)")
    days = [(dt.date(2025, 1, 1) + dt.timedelta(days=int(d) - 1)).isoformat() for d in cache.doy.to_numpy()[rec.row.to_numpy()]]
    preds = pd.read_parquet(ROOT / "data" / "adsb" / "v2" / "adn_fold_preds.parquet", columns=["row", "F", "B"])
    feat = pd.read_parquet(ROOT / "data" / "adsb" / "v2" / "features_2025janjul.parquet", columns=["row", *XCOLS])
    if not (np.array_equal(preds.row, rec.row) and np.array_equal(feat.row, rec.row)):
        raise ValueError("the ADN record / feature table rows are not the fold record's rows in order")
    y, px = rec.y.to_numpy(float), rec.proxy.to_numpy(float)
    gain = (np.maximum(px - preds.F.to_numpy(), 1) - y) ** 2 - (np.maximum(px - preds.B.to_numpy(), 1) - y) ** 2
    f = pd.DataFrame({"ap": rec.ap.astype(str).to_numpy(), "day": days, "month": rec.month.to_numpy(), "gain": gain,
                      **{k: feat[k].to_numpy() for k in XCOLS}})
    keep = np.isfinite(A.day_folds(days)) & (f.coverage.to_numpy() >= 2)
    return f[keep].reset_index(drop=True)


def load_2026() -> pd.DataFrame:
    t = pd.read_parquet(ROOT / "data" / "adsb" / "v2" / "features_2026janjul.parquet", columns=["MVT_ID_mvt", *XCOLS])
    rk = pd.read_parquet(ROOT / "data" / "cache_stand" / "ranking.parquet", columns=["MVT_ID_mvt", "ap"])
    ap = rk.set_index(rk.MVT_ID_mvt.to_numpy(dtype="float64")).ap.reindex(t.MVT_ID_mvt.to_numpy(dtype="float64"))
    if ap.isna().any():
        raise ValueError(f"{int(ap.isna().sum())} 2026 table rows without a ranking-cache airport")
    t = t.assign(ap=ap.astype(str).to_numpy())
    return t[t.coverage.to_numpy() >= 2].reset_index(drop=True)


# ---------------------------------------------------------------- the three stages

def interval_of_mean(values, day, month) -> tuple:
    from prc.pipeline.scoring import dayblock_interval
    s, lo, hi, _ = dayblock_interval(values, day, month)
    n = len(values)
    return s / n, lo / n, hi / n


def run_k0(f25: pd.DataFrame, airports=D) -> dict:
    out = {}
    for a in airports:
        g = f25[f25.ap == a]
        odd = pd.to_datetime(g.day).dt.day.to_numpy() % 2 == 1
        ev = g[~odd]
        w, auc = crossfit_ratio(g[odd], ev)
        G, lo, hi = interval_of_mean(ev.gain.to_numpy(), ev.day.to_numpy(), ev.month.to_numpy())
        Gw = weighted_gain(ev.gain.to_numpy(), w)
        ess_frac = kish_ess(w) / len(w)
        out[a] = {"auc": auc, "ess_frac": ess_frac, "G_even": G, "ci_even": [lo, hi], "Gw_even": Gw,
                  "rule": k0_rule(auc, ess_frac, lo <= Gw <= hi)}
    shapes = {r["rule"] for r in out.values()}
    out["overall"] = "PASS" if shapes == {"PASS"} else ("FAIL_ESS_ONLY" if shapes <= {"PASS", "FAIL_ESS_ONLY"} else "FAIL")
    return out


def run_kplus(f25: pd.DataFrame, airports) -> dict:
    rng = np.random.default_rng(0)
    num_w = num_t = n_tot = 0.0
    per = {}
    for a in airports:
        g = f25[f25.ap == a].reset_index(drop=True)
        if len(g) < KPLUS_MIN_ROWS:
            continue
        pi = np.exp(-normal_score(g.first_ground_gs.to_numpy(), np.full(len(g), a)))
        idx = rng.choice(len(g), size=len(g), replace=True, p=pi / pi.sum())
        w, auc = crossfit_ratio(g.iloc[idx], g)
        d = g.gain.to_numpy()
        gt = float((pi * d).sum() / pi.sum())
        num_w += (w * d).sum()
        num_t += len(g) * gt
        n_tot += len(g)
        per[a] = {"n": len(g), "auc": auc, "G": float(d.mean()), "G_true": gt, "Gw": weighted_gain(d, w)}
    G = float(sum(p["G"] * p["n"] for p in per.values()) / n_tot)
    Gw, Gt = float(num_w / n_tot), float(num_t / n_tot)
    return {"per_airport": per, "G": G, "G_true": Gt, "Gw": Gw, "planted_change": abs(G - Gt), "rule": kplus_rule(G, Gw, Gt)}


def run_verdict(f25: pd.DataFrame, f26: pd.DataFrame, airports) -> dict:
    out = {}
    for a in airports:
        g, t = f25[f25.ap == a], f26[f26.ap == a]
        if len(g) < KPLUS_MIN_ROWS or len(t) < KPLUS_MIN_ROWS:
            out[a] = {"n25": len(g), "n26": len(t), "decision": "NOT_COMPUTED (fewer than 1,000 joined rows in a year)"}
            continue
        w, auc = crossfit_ratio(t, g)
        d = g.gain.to_numpy()
        G = float(d.mean())
        Gw, lo, hi = interval_of_mean(w * d, g.day.to_numpy(), g.month.to_numpy())
        ess = kish_ess(w)
        out[a] = {"n25": len(g), "n26": len(t), "auc": auc, "ess": ess, "ess_frac": ess / len(g), "G": G, "Gw": Gw,
                  "ci_Gw": [lo, hi], "ratio": Gw / G if G else None, **verdict(G, Gw, lo, ess, len(g))}
    return out


def main(argv=None) -> int:
    """Default: RWC on D. `--airports LIRF --out reports/rwc_lirf.json` is RWC.3 (same machinery, same clauses)."""
    import argparse
    ap_ = argparse.ArgumentParser(description=main.__doc__)
    ap_.add_argument("--airports", default=",".join(D))
    ap_.add_argument("--out", default=str(OUT))
    args = ap_.parse_args(argv)
    decided = tuple(args.airports.split(","))
    out_path = pathlib.Path(args.out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}")
    sys.path.insert(0, str(ROOT))
    f25, f26 = load_2025(), load_2026()
    print(f"2025 joined scored rows {len(f25):,}; 2026 joined rows {len(f26):,}", flush=True)
    k0 = run_k0(f25, decided)
    print("K0", json.dumps(k0, default=float), flush=True)
    kp = run_kplus(f25, sorted(set(D) | set(REPORT_ONLY)))
    print("K+", json.dumps({k: v for k, v in kp.items() if k != "per_airport"}), flush=True)
    harness = "PASS" if k0["overall"] == "PASS" and kp["rule"] == "PASS" else "FAIL"
    report = list(D) + list(REPORT_ONLY) if decided == D else list(decided)
    res = run_verdict(f25, f26, report)
    for a, r in res.items():
        print(a, json.dumps(r, default=float), flush=True)
    ship = sorted(a for a in decided if harness == "PASS" and res[a].get("decision") == "SHIP")
    rec = {"prereg": "plans/PREREG_reweight_check_2026_09_11.md", "written": dt.datetime.now().astimezone().isoformat(),
           "k0": k0, "kplus": kp, "harness": harness, "results": res, "decided_set": list(decided), "ship_from_D": ship}
    if decided == D:
        rec["gate_final"] = sorted(["EHAM", "EGLL", "LEMD", "LFPG", "LTFM", *ship])
    out_path.write_text(json.dumps(rec, indent=1, default=float))
    print(f"harness {harness}; ship from {list(decided)}: {ship}; final gate {rec.get('gate_final', 'n/a (RWC.3 scope)')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
