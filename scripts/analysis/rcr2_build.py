"""RCR2 — target `y` on the PRODUCTION recipe, composed against the production incumbent
(plans/PREREG_rcr2_direct_target_production_2026_09_22.md, registered sha256 48d3b93b).

Stage 1: both month-forward directions at the CHR/RCR deviation (lr 0.05), delta vs y on identical test rows; gate G1.
Stage 2: fold A on the shipped parameters, 3 seeds, composed on the CBV rows against opc_screen's incumbent (208.3104).
Nothing here uploads or writes under submissions/; a WORKING verdict authorises a candidate build only.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/rcr2_build.py stage1 [--smoke]
    /opt/homebrew/bin/python3.11 -B scripts/analysis/rcr2_build.py stage2 [--smoke]
"""
from __future__ import annotations

import argparse, gc, hashlib, importlib.util, json, pathlib, resource, sys, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
PREREG = ROOT / "plans" / "PREREG_rcr2_direct_target_production_2026_09_22.md"
PREREG_SHA = "48d3b93b90d6373d4e729aa0a6e83e6f15f7bf44f09273b3dc05054235ef30b9"
REPORT1 = ROOT / "reports" / "rcr2_stage1_report.json"
REPORT2 = ROOT / "reports" / "rcr2_stage2_report.json"
PREDS1 = ROOT / "reports" / "rcr2_stage1_predictions.parquet"
PREDS2 = ROOT / "reports" / "rcr2_fold_predictions.parquet"
RCR_PREDS = ROOT / "reports" / "rcr_test_predictions.parquet"
RCR_A2_GAIN = 1.5094285641585827                  # reports/rcr_report.json vs_A0.A2.gain_s
INC_RMSE = 208.31040447352203                     # opc_screen.load_data's asserted incumbent
N_INC, N_POP = 334_132, 161_956
BOARD_N = 344_841.0
MAX_LABEL_S = 86_400.0

LR, MAX_ROUNDS, PATIENCE = 0.05, 8000, 200        # Stage 1: the registered CHR/RCR deviation
ES_DAY_FRAC, ES_SEED = 0.15, 1709 + 7
LEGS = {"L-Jul": {"fit_months": (1, 2, 3, 4, 5, 6), "test_month": 7},
        "L-Jan": {"fit_months": (8, 9, 10, 11, 12), "test_month": 1}}
TARGETS = ("delta", "y")
SEEDS = (0, 1, 2)
N_BOOT, BOOT_SEED = 2000, 0
BANNED_IN_FEATURES = ("y", "delta", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "regime")
FLOOR_BIND_MAX = 0.005
SEED_SPREAD_MAX_S = 2.0
RSS_LIMIT = 6 * 1024 ** 3


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


def production_feats(S=None) -> list:
    """The shipped 93: BASELINE(68) + QUEUE(12) + ORDER(3) + WEATHER(10); leak gate 1 by name."""
    S = S or _load("stand_ab", "scripts/stand_ab.py")
    feats = list(S.BASELINE_FEATS) + list(S.QUEUE_FEATS) + list(S.ORDER_FEATS) + list(S.WEATHER_FEATS)
    leaks = [c for c in feats if c in BANNED_IN_FEATURES]
    if leaks:
        raise AssertionError(f"leak gate 1: feature list contains label column(s) {leaks}")
    if len(feats) != 93:
        raise AssertionError(f"production recipe is 93 features, got {len(feats)}")
    return feats


def check_labels(y) -> dict:
    """Label integrity: finite and >= 1 s (refused otherwise); labels above one day are COUNTED, not excluded
    (registration ERRATUM 1: the one such row, LIRF doy 196 y = 87,002 s, is production's training/holdout row)."""
    y = np.asarray(y, float)
    nonfinite = int((~np.isfinite(y)).sum())
    if nonfinite:
        raise AssertionError(f"VOID: {nonfinite} non-finite label(s)")
    lo, hi = float(y.min()), float(y.max())
    if lo < 1.0:
        raise AssertionError(f"VOID: label minimum {lo} below 1 s")
    over = int((y > MAX_LABEL_S).sum())          # ERRATUM 1: counted and reported, never excluded (production keeps them)
    return {"n": int(len(y)), "min": lo, "max": hi, "n_over_one_day": over}


def leg_masks(month, day, fit_months, test_month, es_frac=ES_DAY_FRAC, seed=ES_SEED):
    """(fit, es, test) row masks for one month-forward leg. ES = a random `es_frac` of the FIT months' real
    days; the test month is never in fit or ES (leak gate 2). Every row is in at most one set."""
    month = np.asarray(month); day = np.asarray(day)
    if test_month in fit_months:
        raise ValueError(f"test month {test_month} is inside the fit months {fit_months}")
    train = np.isin(month, list(fit_months))
    test = month == test_month
    if not train.any() or not test.any():
        raise ValueError("empty fit or test set")
    days = np.array(sorted(set(day[train].tolist())))
    rng = np.random.default_rng(seed)
    es_days = set(rng.choice(days, size=max(1, int(round(es_frac * len(days)))), replace=False).tolist())
    es = train & np.isin(day, list(es_days))
    fit = train & ~es
    assert not (fit & test).any() and not (es & test).any() and not (fit & es).any(), "leak gate 2"
    assert fit.any() and es.any()
    return fit, es, test


def fit_leg(X, label, fit, es, X_test, params, max_rounds=MAX_ROUNDS, patience=PATIENCE):
    """One early-stopped LightGBM fit on `fit`, stopping on `es`; returns (raw predictions on X_test, best_iter)."""
    import lightgbm as lgb
    ds = lgb.Dataset(X[fit], label=label[fit], free_raw_data=True)
    dv = lgb.Dataset(X[es], label=label[es], reference=ds, free_raw_data=True)
    m = lgb.train(params, ds, max_rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(patience, verbose=False)])
    raw = m.predict(X_test, num_iteration=m.best_iteration)
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("non-finite predictions")            # Part 5 rule 1: in-flight abort
    return raw, int(m.best_iteration)


def taxi(raw, proxy, target):
    """The floored taxi time a prediction implies: max(proxy - raw, 1) under delta, max(raw, 1) under y."""
    LS = _load("lgbm_submit", "scripts/lgbm_submit.py")
    return np.maximum(LS.raw_taxi_time(raw, proxy, target), 1.0)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def _minus_top(y, ref, arm, k):
    """RMSE gain (ref - arm) with the k rows of largest |per-row SSE gain| removed, and their share of the SSE gain."""
    y = np.asarray(y, float); g = (y - ref) ** 2 - (y - arm) ** 2
    order = np.argsort(-np.abs(g))
    keep = np.ones(len(g), bool); keep[order[:k]] = False
    total = g.sum()
    share = float(np.abs(g[order[:k]]).sum() / abs(total)) if total != 0 else float("nan")
    return float(rmse(y[keep], ref[keep]) - rmse(y[keep], arm[keep])), share


def paired_stats(y, days, ap, ref, arm, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """RMSE-seconds comparison of `arm` against `ref` on identical rows: gain, day-blocked CI, remove-and-recompute
    (minus top-1 / top-3), airport exclusions, per-airport gains, and the SSE gain in board units (SSE/344,841)."""
    CHR = _load("chr_build", "scripts/analysis/chr_build.py")
    y = np.asarray(y, float); ref = np.asarray(ref, float); arm = np.asarray(arm, float); ap = np.asarray(ap)
    if not (len(y) == len(ref) == len(arm) == len(ap) == len(days)):
        raise ValueError("length mismatch")
    out = {"n": int(len(y)), "rmse_ref": rmse(y, ref), "rmse_arm": rmse(y, arm)}
    out["gain_s"] = out["rmse_ref"] - out["rmse_arm"]
    out["ci95"] = CHR.day_block_ci(y, days, ref, arm, n=n_boot, seed=seed)
    out["gain_minus_top1_s"], out["top1_share"] = _minus_top(y, ref, arm, 1)
    out["gain_minus_top3_s"], out["top3_share"] = _minus_top(y, ref, arm, 3)
    for a in ("LFPG", "LIRF"):
        m = ap != a
        out[f"gain_ex_{a}_s"] = float(rmse(y[m], ref[m]) - rmse(y[m], arm[m])) if m.any() else float("nan")
    out["per_airport_gain_s"] = {str(a): float(rmse(y[ap == a], ref[ap == a]) - rmse(y[ap == a], arm[ap == a]))
                                 for a in sorted(set(ap.tolist()))}
    out["sse_gain_per_board_row"] = float(((y - ref) ** 2 - (y - arm) ** 2).sum() / BOARD_N)
    return out


def move_energy(ref, arm, ap, month) -> dict:
    """mean((arm - ref)^2) by (month, airport): the label-free tripwire quantity (INV section 6.3)."""
    d = pd.DataFrame({"m": np.asarray(month), "ap": np.asarray(ap), "e": (np.asarray(arm, float) - np.asarray(ref, float)) ** 2})
    g = d.groupby(["m", "ap"]).e.mean()
    return {f"{int(m)}:{a}": float(v) for (m, a), v in g.items()}


def gate_g1(stage1: dict) -> bool:
    """G1: Stage 2 runs only if BOTH legs have raw gain >= 0 AND minus-top-3 gain >= 0."""
    legs = [stage1[leg]["y_vs_delta"] for leg in LEGS]
    return all(s["gain_s"] >= 0 and s["gain_minus_top3_s"] >= 0 for s in legs)


def reproduce_rcr_a2(path=RCR_PREDS) -> dict:
    """The artifact check the runner performs before any fit: RCR's A2 gain must reproduce to 1e-6."""
    p = pd.read_parquet(path)
    y = p.y.to_numpy(float)
    gain = rmse(y, p.yhat_A0.to_numpy(float)) - rmse(y, p.yhat_A2.to_numpy(float))
    if abs(gain - RCR_A2_GAIN) > 1e-6:
        raise AssertionError(f"RCR A2 does not reproduce: {gain} vs {RCR_A2_GAIN}")
    ap = np.full(len(p), "?", dtype=object)          # the RCR artifact carries no airport column
    st = paired_stats(y, p.real_day.to_numpy(), ap, p.yhat_A0.to_numpy(float), p.yhat_A2.to_numpy(float))
    st.pop("per_airport_gain_s"); st.pop("gain_ex_LFPG_s"); st.pop("gain_ex_LIRF_s")
    return st


# ---- Stage 2 helpers ------------------------------------------------------------------------------------------
def load_incumbent(subset_ok: bool = False) -> pd.DataFrame:
    """opc_screen.load_data's composed incumbent, re-derived here from the same three artifacts and asserted the
    same way (334,132 rows, pop 161,956, inc RMSE 208.31040447352203). `subset_ok` (smoke only) relaxes the counts."""
    O = _load("opc_screen", "scripts/analysis/opc_screen.py")
    s = O.read("data/adsb/v2/v13_matched_oof.parquet",
               ["row", O.ID, "ap", "month", "day", "fold", "y", "proxy", "F", "v13_delta", "taxi_v13_delta"])
    h = O.read("data/cache_stand/catboost_native_input.parquet", ["row", "holdout"], filters=[("holdout", "=", True)])
    h["C"] = np.load(ROOT / "data/cache_stand/catboost_native_delta.npy")
    s = O.joined(s, h.drop(columns="holdout"), "row")
    fw = O.read("data/cache_stand/fold_preds_queue_order_weather.parquet", ["row", "y", "proxy", "treatment"])
    fw = fw.rename(columns={"y": "fw_y", "proxy": "fw_proxy", "treatment": "FW"})
    s = O.joined(s, fw, "row")
    assert np.array_equal(s.y, s.fw_y) and np.array_equal(s.proxy, s.fw_proxy)
    s = s[np.isfinite(s.fold)].copy().reset_index(drop=True)
    return finish_incumbent(s, subset_ok)


def finish_incumbent(s: pd.DataFrame, subset_ok: bool = False) -> pd.DataFrame:
    """The composition and its assertions, separated so a synthetic frame can test them."""
    s = s.copy()
    s["pop"] = s.v13_delta.eq(s.F).to_numpy()
    s["taxi_C"] = np.maximum(s.proxy - s.C, 1)
    s["taxi_FW"] = np.maximum(s.proxy - s.FW, 1)
    s["inc"] = np.where(s["pop"], .5 * (s.taxi_C + s.taxi_FW), s.taxi_v13_delta)
    if not subset_ok:
        assert len(s) == N_INC and int(s["pop"].sum()) == N_POP, (len(s), int(s["pop"].sum()))
        r = rmse(s.y, s.inc)
        assert abs(r - INC_RMSE) < 1e-8, f"incumbent does not reproduce: {r}"
    s["doy"] = pd.to_datetime(s.day).dt.dayofyear.astype(int)
    return s


def compose(frame: pd.DataFrame, fy) -> np.ndarray:
    """cand = where(pop, 0.5*(taxi_C + max(Fy, 1)), taxi_v13_delta): Fy replaces FW on the CBV rows only."""
    fy = np.asarray(fy, float)
    if fy.shape != (len(frame),):
        raise ValueError(f"Fy shape {fy.shape} != {len(frame)} rows")
    return np.where(frame["pop"].to_numpy(), .5 * (frame.taxi_C.to_numpy() + np.maximum(fy, 1.0)),
                    frame.taxi_v13_delta.to_numpy())


def block_bootstrap_delta(g, blocks, n_boot=N_BOOT, seed=BOOT_SEED):
    """Block bootstrap of the summed per-row SSE gain `g` over `blocks` -> (delta, ci95) in SSE/344,841 units."""
    fr = pd.DataFrame({"b": np.asarray(blocks), "g": np.asarray(g, float)})
    per = fr.groupby("b").g.sum().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    sums = per[idx].sum(1) / BOARD_N
    return float(per.sum() / BOARD_N), [float(np.quantile(sums, .025)), float(np.quantile(sums, .975))]


def mde(ci) -> float:
    """Minimum detectable effect at 80 % power from a 95 % CI: 2.8 x half-width / 1.96 (result-writing rule g)."""
    return 2.8 * ((ci[1] - ci[0]) / 2) / 1.96


def delta_stats(y, doy, ap, month, base, cand, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """Composed comparison in SSE/344,841 (convention A: positive = improvement): raw, minus top-1/3 rows, per month,
    ex-LFPG / ex-LIRF, per airport, (month, day)-blocked CI, MDE, k and S on the moved rows, RMSE seconds."""
    y = np.asarray(y, float); base = np.asarray(base, float); cand = np.asarray(cand, float)
    ap = np.asarray(ap); month = np.asarray(month); doy = np.asarray(doy)
    g = (y - base) ** 2 - (y - cand) ** 2
    blocks = month.astype(int) * 1000 + doy.astype(int)
    delta, ci = block_bootstrap_delta(g, blocks, n_boot, seed)
    order = np.argsort(-np.abs(g))
    out = {"n": int(len(y)), "delta": delta, "ci95": ci, "mde": mde(ci), "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
           "rmse_base": rmse(y, base), "rmse_cand": rmse(y, cand)}
    out["gain_s"] = out["rmse_base"] - out["rmse_cand"]
    for k in (1, 3):
        keep = np.ones(len(g), bool); keep[order[:k]] = False
        out[f"delta_minus_top{k}"] = float(g[keep].sum() / BOARD_N)
    out["top1_share"] = float(abs(g[order[0]]) / abs(g.sum())) if g.sum() != 0 else float("nan")
    out["per_month"] = {int(m): float(g[month == m].sum() / BOARD_N) for m in sorted(set(month.tolist()))}
    for a in ("LFPG", "LIRF"):
        out[f"delta_ex_{a}"] = float(g[ap != a].sum() / BOARD_N)
    out["per_airport"] = {str(a): float(g[ap == a].sum() / BOARD_N) for a in sorted(set(ap.tolist()))}
    D = cand - base; e = y - base; moved = D != 0
    out["S"] = float((D ** 2).sum() / BOARD_N)
    out["k"] = float((D * e).sum() / (D ** 2).sum()) if moved.any() else float("nan")
    out["n_moved"] = int(moved.sum())
    return out


def stage2_verdict(comp: dict, f_alone: dict, floor_bind_rate: float, seed_rmses) -> dict:
    """T2-T6 of the registration, each named; WORKING iff every clause holds. T1 (gate G1) is checked by the caller."""
    seed_rmses = [float(v) for v in seed_rmses]
    spread = max(abs(v - float(np.mean(seed_rmses))) for v in seed_rmses) if seed_rmses else float("inf")
    clauses = {
        "T2_comp_positive_ci": bool(comp["delta"] > 0 and comp["ci95"][0] > 0),
        "T3_both_months_nonneg": bool(all(v >= 0 for v in comp["per_month"].values())),
        "T4_minus_top3_and_ex_LFPG": bool(comp["delta_minus_top3"] > 0 and comp["delta_ex_LFPG"] > 0),
        "T5_F_alone_positive_ci": bool(f_alone["delta"] > 0 and f_alone["ci95"][0] > 0),
        "T6_floor_and_seeds": bool(floor_bind_rate < FLOOR_BIND_MAX and spread <= SEED_SPREAD_MAX_S),
    }
    verdict = "WORKING" if all(clauses.values()) else "NOT WORKING"
    if verdict == "NOT WORKING" and not comp["ci_excludes_zero"]:
        verdict = "NOT WORKING (not established: CI includes 0; see mde)"
    return {"verdict": verdict, "clauses": clauses, "seed_spread_s": spread}


def predictions_scored(stage1: dict | None, stage2: dict | None) -> dict:
    """P1-P7 as HIT / MISS / n.a. from the measured numbers (P8 is scored from the run record by the caller)."""
    out = {}
    if stage1 is not None:
        jan, jul = stage1["L-Jan"]["y_vs_delta"], stage1["L-Jul"]["y_vs_delta"]
        out["P1_LJan_raw_negative"] = "HIT" if jan["gain_s"] < 0 else "MISS"
        out["P2_LJul_band"] = "HIT" if (-5 <= jul["gain_s"] <= 2 and abs(jul["gain_minus_top3_s"] - jul["gain_s"]) <= 1) else "MISS"
        biggest = [max(s["per_airport_gain_s"], key=lambda a: abs(s["per_airport_gain_s"][a])) for s in (jan, jul)]
        out["P3_LIRF_largest_both_legs"] = "HIT" if biggest == ["LIRF", "LIRF"] else "MISS"
        out["P4_stops_at_stage1"] = "HIT" if not gate_g1(stage1) else "MISS"
    if stage2 is not None:
        c, f = stage2["composed"], stage2["f_alone"]
        hw = (c["ci95"][1] - c["ci95"][0]) / 2
        out["P5_bands"] = "HIT" if (-200 <= f["delta"] <= 400 and -100 <= c["delta"] <= 200 and 60 <= hw <= 250) else "MISS"
        out["P6_best_iter"] = "HIT" if 0.7 * 21101 <= stage2["best_iter"] <= 1.1 * 21101 else "MISS"
        out["P7_top_row_share"] = "HIT" if c["top1_share"] < 0.25 else "MISS"
    return out


# ---- orchestration --------------------------------------------------------------------------------------------
def _guards():
    G = _load("run_guards", "scripts/run_guards.py")
    fh = G.acquire_lock()
    G.start_rss_guard(RSS_LIMIT)
    return fh, G


def _prereg_ok():
    h = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    text = PREREG.read_text()
    cut = [text.index(m) for m in ("\n## ERRATUM", "\n## RESULT") if m in text]   # appended sections, never edits
    reg = text[:min(cut)] if cut else text
    hreg = hashlib.sha256(reg.encode()).hexdigest()
    if PREREG_SHA not in (h, hreg):
        raise AssertionError("registration text changed after hashing")
    return h


def _load_months_frame(months, feats):
    S = _load("stand_ab", "scripts/stand_ab.py")
    LF = _load("lgbm_fold", "scripts/lgbm_fold.py")
    frames, _ = LF._load_months(S.CACHE, months, feats, S.QCACHE, None, S.OCACHE, S.WCACHE)
    keep = [c for c in feats if c not in S.ENC] + list(S.ENC_KEYS) + ["y", "delta", "proxy", "month", "doy", "ap"]
    keep = list(dict.fromkeys(keep))
    frames = [f[keep] for f in frames]                     # the in-fold encodings are recomputed per leg
    d = pd.concat(frames, ignore_index=True); del frames; gc.collect()
    d = d[np.isfinite(d.delta.to_numpy(float))].reset_index(drop=True)
    return d


def load_fw_all() -> pd.DataFrame:
    """The shipped FW fold delta on every fold-A holdout row (339,015) with the real day: (row, FW, doy)."""
    fw = pd.read_parquet(ROOT / "data/cache_stand/fold_preds_queue_order_weather.parquet", columns=["row", "treatment"])
    oof = pd.read_parquet(ROOT / "data/adsb/v2/v13_matched_oof.parquet", columns=["row", "day"])
    return attach_doy(fw.rename(columns={"treatment": "FW"}), oof)


def attach_doy(fw: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Join the real day onto every FW row (one-to-one on `row`); a row without a day is refused, never dropped."""
    j = fw.merge(oof, on="row", how="left", validate="one_to_one")
    if j.day.isna().any():
        raise AssertionError(f"{int(j.day.isna().sum())} fold row(s) have no day in v13_matched_oof")
    j["doy"] = pd.to_datetime(j.day).dt.dayofyear.astype(int)
    return j.drop(columns="day")


def run_stage1(smoke=False):
    started = time.monotonic()
    if not smoke and REPORT1.exists():
        raise FileExistsError("Stage 1 result exists; no silent rerun")
    fh, G = _guards()
    sha = _prereg_ok()
    S = _load("stand_ab", "scripts/stand_ab.py"); LS = _load("lgbm_submit", "scripts/lgbm_submit.py")
    feats = production_feats(S)
    rcr = reproduce_rcr_a2()
    print(json.dumps({"stage": "rcr_a2_reproduced", **{k: v for k, v in rcr.items() if k != "ci95"}, "ci95": rcr["ci95"]}), flush=True)
    months = sorted({m for leg in LEGS.values() for m in leg["fit_months"]} | {leg["test_month"] for leg in LEGS.values()})
    d = _load_months_frame(months, feats)
    if smoke:
        d = d.iloc[::40].reset_index(drop=True)
    labels = check_labels(d.y.to_numpy(float))
    month = d.month.to_numpy().astype(int); day = d.doy.to_numpy().astype(int)
    y_all = d.y.to_numpy(float); delta_all = d.delta.to_numpy(float); proxy = d.proxy.to_numpy(float)
    ap_all = d.ap.to_numpy().astype(str)
    keys = d[S.ENC_KEYS]
    report, preds_out = {"labels": labels}, []
    for leg, spec in LEGS.items():
        fit, es, test = leg_masks(month, day, spec["fit_months"], spec["test_month"])
        enc = S.infold_encodings(keys, y_all, delta_all, fit)          # FIT rows only (leak gate 2)
        for c, v in (enc.items() if isinstance(enc, dict) else zip(S.ENC, np.asarray(enc).T)):
            d[c] = v                                                   # overwritten per leg; no frame copy
        X = LS.design_matrix(d, feats); gc.collect()
        p = dict(LS.P); p.update(learning_rate=LR, num_threads=4, verbosity=-1)
        taxis, iters = {}, {}
        for tgt in TARGETS:
            label = delta_all if tgt == "delta" else y_all
            raw, iters[tgt] = fit_leg(X, label, fit, es, X[test], p, 60 if smoke else MAX_ROUNDS, PATIENCE)
            taxis[tgt] = taxi(raw, proxy[test], tgt)
            print(json.dumps({"leg": leg, "target": tgt, "best_iter": iters[tgt], "rmse": rmse(y_all[test], taxis[tgt]),
                              "n_fit": int(fit.sum()), "n_es": int(es.sum()), "n_test": int(test.sum()),
                              "secs": round(time.monotonic() - started, 1), "peak_GiB": round(G.peak_rss() / 1024 ** 3, 2)}), flush=True)
        st = paired_stats(y_all[test], day[test], ap_all[test], taxis["delta"], taxis["y"])
        report[leg] = {"fit_months": list(spec["fit_months"]), "test_month": spec["test_month"], "n_fit": int(fit.sum()),
                       "n_es": int(es.sum()), "n_test": int(test.sum()), "best_iter": iters, "y_vs_delta": st,
                       "move_energy": move_energy(taxis["delta"], taxis["y"], ap_all[test], month[test])}
        preds_out.append(pd.DataFrame({"leg": leg, "doy": day[test], "ap": ap_all[test], "y": y_all[test], "proxy": proxy[test],
                                       "taxi_delta": taxis["delta"], "taxi_y": taxis["y"]}))
        print(json.dumps({"leg": leg, "y_vs_delta": {k: v for k, v in st.items() if k != "per_airport_gain_s"}}), flush=True)
        del X; gc.collect()
    report["gate_G1_pass"] = gate_g1(report)
    report["predictions"] = predictions_scored(report, None)
    report.update(tag="RCR2", stage=1, rcr_a2_reproduced=rcr, seconds=time.monotonic() - started,
                  peak_rss_GiB=G.peak_rss() / 1024 ** 3, prereg_sha256=sha, smoke=smoke,
                  scope="Stage 1: single seed, lr 0.05 (registered deviation), 2025 only; a recipe-shape gate, not a board number")
    report["predictions"]["P8_stage1_wall_and_rss"] = "HIT" if (report["seconds"] <= 1800 and report["peak_rss_GiB"] < 6) else "MISS"
    if smoke:
        print(json.dumps({"smoke": "passed; scores not inspected", "gate_G1_pass": report["gate_G1_pass"]})); G.release_lock(fh); return report
    pd.concat(preds_out, ignore_index=True).to_parquet(PREDS1, index=False)
    with REPORT1.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({"gate_G1_pass": report["gate_G1_pass"], "predictions": report["predictions"]}, indent=1), flush=True)
    G.release_lock(fh)
    return report


def run_stage2(smoke=False):
    started = time.monotonic()
    if not smoke and REPORT2.exists():
        raise FileExistsError("Stage 2 result exists; no silent rerun")
    if not smoke:
        s1 = json.loads(REPORT1.read_text())
        if not s1["gate_G1_pass"]:
            raise RuntimeError("gate G1 failed in Stage 1: Stage 2 is not run (registration)")
    fh, G = _guards()
    sha = _prereg_ok()
    S = _load("stand_ab", "scripts/stand_ab.py"); LS = _load("lgbm_submit", "scripts/lgbm_submit.py")
    LF = _load("lgbm_fold", "scripts/lgbm_fold.py")
    feats = production_feats(S)
    fold = LF.load_fold(S.CACHE, (1, 2, 3) if smoke else None, feats, qcache=S.QCACHE, ocache=S.OCACHE, wcache=S.WCACHE,
                        check_counts=not smoke, label="holdout (months 1, 7)")
    X, y, dlt, proxy, te, tr, fit, es = (fold[k] for k in ("X", "y", "dlt", "proxy", "te", "tr", "fit", "es"))
    labels = check_labels(y)
    p = dict(LS.P)
    nest, patience = (60, 50) if smoke else (LS.NEST, LS.PATIENCE)
    if smoke:
        p["learning_rate"] = LS.SMOKE["learning_rate"]
    info = LS.early_stop(X, dlt, y, proxy, tr, fit, es, p, nest, patience, target="y")
    label = LS.regression_label(dlt, y, "y")
    preds = LS.fit_seeds(X, label, tr, X[te], p, info["n_ref"], SEEDS, n_features=len(feats), target="y")
    del X; gc.collect()
    base = fold["base"].copy()
    for s_ in SEEDS:
        base[f"y_seed{s_}"] = preds[s_]
    fy = np.mean([preds[s_] for s_ in SEEDS], axis=0)
    if not np.all(np.isfinite(fy)):
        raise RuntimeError("non-finite Fy")
    base["Fy"] = fy
    base.to_parquet(PREDS2 if not smoke else ROOT / "reports" / "rcr2_fold_predictions_smoke.parquet", index=False)
    seed_rmses = [rmse(base.y, np.maximum(base[f"y_seed{s_}"], 1)) for s_ in SEEDS]
    floor_rate = float((fy < 1.0).mean())
    inc = load_incumbent(subset_ok=smoke)
    j = inc.merge(base[["row", "y", "proxy", "Fy"]].rename(columns={"y": "b_y", "proxy": "b_proxy"}), on="row",
                  how="inner" if smoke else "left", validate="one_to_one")
    if not smoke:
        assert len(j) == N_INC and j.Fy.notna().all(), "fold rows do not cover the incumbent frame"
    assert np.array_equal(j.y, j.b_y) and np.array_equal(j.proxy, j.b_proxy), "row join is not the same row"
    cand = compose(j, j.Fy.to_numpy())
    comp = delta_stats(j.y, j.doy, j.ap, j.month, j.inc.to_numpy(), cand)
    fa = base.merge(load_fw_all(), on="row", how="inner" if smoke else "left", validate="one_to_one")
    if not smoke:
        assert len(fa) == len(base) and fa.FW.notna().all(), "FW does not cover every fold row"
    f_alone = delta_stats(fa.y, fa.doy, fa.ap, fa.month, np.maximum(fa.proxy - fa.FW, 1), np.maximum(fa.Fy, 1))
    verdict = stage2_verdict(comp, f_alone, floor_rate, seed_rmses)
    report = {"tag": "RCR2", "stage": 2, "verdict": verdict, "composed": comp, "f_alone": f_alone, "labels": labels,
              "best_iter": info["best_iter"], "n_ref": info["n_ref"], "es_info": info, "seed_rmses": seed_rmses,
              "floor_bind_rate": floor_rate, "seconds": time.monotonic() - started, "peak_rss_GiB": G.peak_rss() / 1024 ** 3,
              "prereg_sha256": sha, "smoke": smoke, "n_incumbent_rows": int(len(j)), "n_pop": int(j["pop"].sum())}
    report["predictions"] = predictions_scored(None, report)
    report["predictions"]["P8_stage2_wall_and_rss"] = "HIT" if (report["seconds"] <= 10800 and report["peak_rss_GiB"] < 6) else "MISS"
    if smoke:
        print(json.dumps({"smoke": "passed; scores not inspected", "verdict": verdict["verdict"]})); G.release_lock(fh); return report
    with REPORT2.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False, default=float)
    print(json.dumps({"verdict": verdict, "composed": {k: v for k, v in comp.items() if k != "per_airport"},
                      "f_alone": {k: v for k, v in f_alone.items() if k != "per_airport"}, "predictions": report["predictions"]}, indent=1), flush=True)
    G.release_lock(fh)
    return report


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("mode", choices=["stage1", "stage2"]); a.add_argument("--smoke", action="store_true")
    args = a.parse_args()
    (run_stage1 if args.mode == "stage1" else run_stage2)(args.smoke)
