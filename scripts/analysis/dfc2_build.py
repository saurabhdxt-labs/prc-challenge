"""DFC.2 — the six network-flow columns (fixed `dfc_build`, BC-25) on the shipped 93-feature delta base
(plans/PREREG_dfc2_networkflow_forward_2026_09_23.md, registered sha256 2c0d0b2d).

Stage 1: both month-forward directions at the CHR/RCR deviation (lr 0.05); per leg the base (93), A6 (99 = + six DFC)
and A7 (96 = + the three globally available dep_* columns, reported only) fit on identical rows; gate G1 on A6 vs base.
Stage 2 (only if G1 passes): fold A on the shipped parameters, 3 seeds, composed against opc_screen's incumbent
(208.3104), plus the T7 label-free 2026 construction tripwires. Nothing here uploads or writes under submissions/.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/dfc2_build.py stage1 [--smoke]
"""
from __future__ import annotations

import argparse, gc, glob, hashlib, importlib.util, json, pathlib, sys, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
PREREG = ROOT / "plans" / "PREREG_dfc2_networkflow_forward_2026_09_23.md"
PREREG_SHA = "2c0d0b2dbee69e6299f283b5fe1fb31869b921797c93888010f4ae4cabe0b38e"
REPORT1 = ROOT / "reports" / "dfc2_stage1_report.json"
REPORT2 = ROOT / "reports" / "dfc2_stage2_report.json"
PREDS1 = ROOT / "reports" / "dfc2_stage1_predictions.parquet"
PREDS2 = ROOT / "reports" / "dfc2_fold_predictions.parquet"
YEAR = 2025
N_ARRIVAL_FILES = 12
RSS_LIMIT = 6 * 1024 ** 3
#: CHR forward-leg incumbents: the 93-feature delta base at lr 0.05 (reports/rcr2_stage1_report.json, rmse_ref)
RCR2_BASE_RMSE = {"L-Jul": 252.42168727704052, "L-Jan": 210.61355035331607}
FW_BEST_ITER = 21_101
#: smoke subsample stride. 1-in-40 (rcr2's) left hour 23 with ZERO in-set fit rows on L-Jan, so the arrival _dev
#: fallback had nothing to fall back on and the NaN-pattern assertion fired on a sparsity artefact the full frame does
#: not have (full-frame minimum in-set fit rows per hour: 79 L-Jul, 82 L-Jan, 160 fold-A tr; measured 2026-09-23).
#: 1-in-8 keeps >= 8 rows in every hour of every leg.
SMOKE_STRIDE = 8


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


D = _load("dfc_build", "scripts/analysis/dfc_build.py")
R = _load("rcr2_build", "scripts/analysis/rcr2_build.py")

DEP_ONLY = ["dfc_ades_dep_1h", "dfc_ades_dep_3h", "dfc_ades_dep_1h_dev"]
ARMS = {"A6": list(D.DFC_FEATS), "A7": DEP_ONLY}
ARM_ORDER = ("base", "A6", "A7")


# ---- feature lists and inputs --------------------------------------------------------------------------------
def arm_feats(arm: str, S=None) -> list:
    """base = the shipped 93 (rcr2_build.production_feats); A6/A7 append their DFC block. Leak gate 1 by name."""
    base = R.production_feats(S)
    block = [] if arm == "base" else ARMS[arm]                      # KeyError for an unknown arm
    feats = base + list(block)
    leaks = [c for c in feats if c in R.BANNED_IN_FEATURES]
    if leaks:
        raise AssertionError(f"leak gate 1: feature list contains label column(s) {leaks}")
    return feats


def arrival_paths(root=ROOT) -> list:
    """The 12 monthly 2025 raw files, explicit and sorted — never a hidden glob inside the builder (BC-25)."""
    paths = sorted(glob.glob(str(pathlib.Path(root) / "data/raw/training_2025-*.parquet")))
    if len(paths) != N_ARRIVAL_FILES:
        raise AssertionError(f"expected {N_ARRIVAL_FILES} monthly 2025 arrival files, found {len(paths)}")
    return paths


def load_arrivals_2025() -> "D.ArrivalStream":
    return D.load_arrivals(arrival_paths(), year=YEAR)


def dfc_block(frame: pd.DataFrame, fit_mask, arrivals, year: int = YEAR, dev_tables=None) -> pd.DataFrame:
    """The six columns for `frame` from the fixed builder — medians on `fit_mask` rows only (training, leak gate 2) or
    from explicit `dev_tables` (serve) — with the registered NaN pattern asserted: departure columns finite everywhere;
    arrival columns NaN exactly where in_set == 0 given a stream, and NaN everywhere with `arrivals=None` (DFC.3: no
    stream, nothing is observed — the with-stream contract cannot hold there)."""
    out = D.build_dfc(frame[list(D.INPUT_COLS)], fit_mask, year, arrivals, dev_tables=dev_tables)
    if list(out.columns) != list(D.DFC_FEATS) or len(out) != len(frame):
        raise AssertionError("builder returned an unexpected block shape")
    in_set = out.dfc_ades_in_set.to_numpy()
    if not set(np.unique(in_set).tolist()) <= {0.0, 1.0}:
        raise AssertionError("dfc_ades_in_set is not a 0/1 flag")
    for c in DEP_ONLY:
        if not np.isfinite(out[c].to_numpy(float)).all():
            raise AssertionError(f"{c} has non-finite values (F2 contract: never NaN)")
    for c in ("dfc_ades_arr_1h", "dfc_ades_arr_1h_dev"):
        v = out[c].to_numpy(float)
        if arrivals is None:
            if not np.isnan(v).all():
                raise AssertionError(f"{c}: finite value with no arrival stream (must be NaN everywhere)")
        elif not np.isnan(v[in_set == 0]).all() or not np.isfinite(v[in_set == 1]).all():
            raise AssertionError(f"{c}: NaN pattern is not 'unobservable iff out of set'")
    return out


def attach(frame: pd.DataFrame, block: pd.DataFrame) -> None:
    """Write the six columns onto `frame` in place (row-aligned by position; the block was built from this frame)."""
    if len(block) != len(frame):
        raise ValueError(f"block length {len(block)} != frame length {len(frame)}")
    for c in D.DFC_FEATS:
        frame[c] = block[c].to_numpy()


# ---- fitting -------------------------------------------------------------------------------------------------
def fit_arm(X, label, fit, es, X_test, params, feature_names, max_rounds=R.MAX_ROUNDS, patience=R.PATIENCE):
    """One early-stopped LightGBM fit -> (raw test predictions, best_iter, {feature: gain importance})."""
    import lightgbm as lgb
    if len(feature_names) != X.shape[1]:
        raise ValueError(f"{len(feature_names)} feature names for a {X.shape[1]}-column design")
    ds = lgb.Dataset(X[fit], label=label[fit], free_raw_data=True)
    dv = lgb.Dataset(X[es], label=label[es], reference=ds, free_raw_data=True)
    m = lgb.train(params, ds, max_rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(patience, verbose=False)])
    raw = m.predict(X_test, num_iteration=m.best_iteration)
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("non-finite predictions")            # in-flight abort
    imp = {n: float(v) for n, v in zip(feature_names, m.feature_importance("gain"))}
    return raw, int(m.best_iteration), imp


def importance_share(imp: dict, cols) -> float:
    total = float(sum(imp.values()))
    return float(sum(imp.get(c, 0.0) for c in cols) / total) if total > 0 else 0.0


# ---- gate, predictions, tripwires, verdict --------------------------------------------------------------------
def gate_g1(stage1: dict) -> bool:
    """G1: Stage 2 runs only if A6 vs base has raw gain >= 0 AND minus-top-3 gain >= 0 on BOTH legs. A7 never enters."""
    legs = [stage1[leg]["A6_vs_base"] for leg in R.LEGS]
    return all(s["gain_s"] >= 0 and s["gain_minus_top3_s"] >= 0 for s in legs)


def predictions_scored(stage1: dict | None, stage2: dict | None) -> dict:
    """P1-P7 (Stage 1) and P9-P11 (Stage 2) as HIT / MISS; P8 (wall, RSS) is scored from the run record by the caller."""
    out = {}
    if stage1 is not None:
        jul, jan = stage1["L-Jul"], stage1["L-Jan"]
        out["P1_LJul_band"] = "HIT" if -0.5 <= jul["A6_vs_base"]["gain_s"] <= 1.5 else "MISS"
        out["P2_LJan_band"] = "HIT" if -0.5 <= jan["A6_vs_base"]["gain_s"] <= 1.5 else "MISS"
        out["P3_diffuse"] = "HIT" if all(abs(s["A6_vs_base"]["gain_minus_top3_s"] - s["A6_vs_base"]["gain_s"]) <= 0.3
                                        for s in (jul, jan)) else "MISS"
        out["P4_g1_passes"] = "HIT" if gate_g1(stage1) else "MISS"
        out["P5_arrival_contribution_null"] = "HIT" if all(
            -0.5 <= s["A6_vs_A7"]["gain_s"] <= 0.5 and s["A6_vs_A7"]["ci95"][0] <= 0 <= s["A6_vs_A7"]["ci95"][1]
            for s in (jul, jan)) else "MISS"
        out["P6_base_reproduces"] = "HIT" if all(abs(s["base_rmse_minus_rcr2"]) <= 0.3 for s in (jul, jan)) else "MISS"
        out["P7_importance_share"] = "HIT" if all(s["dfc_importance_share_A6"] < 0.03 for s in (jul, jan)) else "MISS"
    if stage2 is not None:
        c, f = stage2["composed"], stage2["f_alone"]
        hw = (c["ci95"][1] - c["ci95"][0]) / 2
        out["P9_bands"] = "HIT" if (-100 <= f["delta"] <= 300 and -50 <= c["delta"] <= 150 and 40 <= hw <= 200) else "MISS"
        out["P10_best_iter"] = "HIT" if 0.8 * FW_BEST_ITER <= stage2["best_iter"] <= 1.2 * FW_BEST_ITER else "MISS"
        out["P11_tripwires"] = "HIT" if stage2["tripwires"]["pass"] else "MISS"
    return out


def tripwire_eval(t: dict) -> dict:
    """T7 (a)-(e) of the registration on the measured 2026 construction quantities; every clause named."""
    clauses = {
        "a": bool(t["guard_raised_on_2025_stream"] and t["guard_raised_on_wrong_year"]),
        "b": bool(all(t["nan_share"][c] == 0.0 for c in DEP_ONLY)),
        "c": bool(all(t["out_of_range_share"][c] <= 0.001 for c in D.DFC_FEATS)
                  and all(abs(t["mean_shift_in_sd"][c]) <= 0.25 for c in D.DFC_FEATS)),
        "d": bool(abs(t["in_set_share_2026"] - t["in_set_share_2025"]) <= 0.01),
        "e": bool(t["unseen_ades_share"] <= 0.001),
    }
    return {"pass": all(clauses.values()), "clauses": clauses, "measured": t}


def stage2_verdict(comp: dict, f_alone: dict, floor_bind_rate: float, seed_rmses, tripwires: dict) -> dict:
    """T2-T6 via rcr2_build.stage2_verdict, AND T7 (the 2026 construction tripwires). WORKING iff every clause holds."""
    v = R.stage2_verdict(comp, f_alone, floor_bind_rate, seed_rmses)
    v["clauses"]["T7_tripwires"] = bool(tripwires["pass"])
    if not tripwires["pass"] and v["verdict"] == "WORKING":
        v["verdict"] = "NOT WORKING"
    return v


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


def load_frame(months, feats) -> pd.DataFrame:
    """The monthly caches with the queue/order/weather blocks, keeping the 93 inputs, the encoding keys, the labels,
    the leg columns AND the DFC inputs (hr, tmin, mvt_sec are not among the 93); finite-delta rows only."""
    S = _load("stand_ab", "scripts/stand_ab.py")
    LF = _load("lgbm_fold", "scripts/lgbm_fold.py")
    frames, _ = LF._load_months(S.CACHE, list(months), feats, S.QCACHE, None, S.OCACHE, S.WCACHE)
    keep = ([c for c in feats if c not in S.ENC] + list(S.ENC_KEYS) + ["y", "delta", "proxy", "month", "doy", "ap"]
            + list(D.INPUT_COLS))
    keep = list(dict.fromkeys(keep))
    frames = [f[keep] for f in frames]                     # the in-fold encodings are recomputed per leg
    d = pd.concat(frames, ignore_index=True); del frames; gc.collect()
    return d[np.isfinite(d.delta.to_numpy(float))].reset_index(drop=True)


def run_stage1(smoke=False):
    started = time.monotonic()
    if not smoke and REPORT1.exists():
        raise FileExistsError("Stage 1 result exists; no silent rerun")
    fh, G = _guards()
    sha = _prereg_ok()
    S = _load("stand_ab", "scripts/stand_ab.py"); LS = _load("lgbm_submit", "scripts/lgbm_submit.py")
    feats = {arm: arm_feats(arm, S) for arm in ARM_ORDER}
    months = sorted({m for leg in R.LEGS.values() for m in leg["fit_months"]} | {leg["test_month"] for leg in R.LEGS.values()})
    d = load_frame(months, feats["base"])
    if smoke:
        d = d.iloc[::SMOKE_STRIDE].reset_index(drop=True)
    arrivals = load_arrivals_2025()
    print(json.dumps({"stage": "arrivals", "years": sorted(arrivals.years), "n_airports": len(arrivals.by_airport),
                      "secs": round(time.monotonic() - started, 1)}), flush=True)
    labels = R.check_labels(d.y.to_numpy(float))
    month = d.month.to_numpy().astype(int); day = d.doy.to_numpy().astype(int)
    y_all = d.y.to_numpy(float); delta_all = d.delta.to_numpy(float); proxy = d.proxy.to_numpy(float)
    ap_all = d.ap.to_numpy().astype(str)
    keys = d[S.ENC_KEYS]
    report, preds_out = {"labels": labels}, []
    for leg, spec in R.LEGS.items():
        fit, es, test = R.leg_masks(month, day, spec["fit_months"], spec["test_month"])
        enc = S.infold_encodings(keys, y_all, delta_all, fit)          # FIT rows only (leak gate 2)
        for c, v in (enc.items() if isinstance(enc, dict) else zip(S.ENC, np.asarray(enc).T)):
            d[c] = v
        attach(d, dfc_block(d, fit, arrivals))                         # medians on FIT rows only (leak gate 2)
        print(json.dumps({"leg": leg, "stage": "dfc", "in_set_share": float(d.dfc_ades_in_set.mean()),
                          "dep_1h_mean": float(d.dfc_ades_dep_1h.mean()),
                          "arr_1h_mean_where_obs": float(np.nanmean(d.dfc_ades_arr_1h)),
                          "secs": round(time.monotonic() - started, 1)}), flush=True)
        p = dict(LS.P); p.update(learning_rate=R.LR, num_threads=4, verbosity=-1)
        taxis, iters, imps = {}, {}, {}
        for arm in ARM_ORDER:
            X = LS.design_matrix(d, feats[arm]); gc.collect()
            raw, iters[arm], imps[arm] = fit_arm(X, delta_all, fit, es, X[test], p, feats[arm],
                                                 60 if smoke else R.MAX_ROUNDS, R.PATIENCE)
            del X; gc.collect()
            taxis[arm] = R.taxi(raw, proxy[test], "delta")
            print(json.dumps({"leg": leg, "arm": arm, "n_feats": len(feats[arm]), "best_iter": iters[arm],
                              "rmse": R.rmse(y_all[test], taxis[arm]), "n_fit": int(fit.sum()), "n_es": int(es.sum()),
                              "n_test": int(test.sum()), "secs": round(time.monotonic() - started, 1),
                              "peak_GiB": round(G.peak_rss() / 1024 ** 3, 2)}), flush=True)
        yt, dt, at = y_all[test], day[test], ap_all[test]
        legrep = {"fit_months": list(spec["fit_months"]), "test_month": spec["test_month"], "n_fit": int(fit.sum()),
                  "n_es": int(es.sum()), "n_test": int(test.sum()), "best_iter": iters,
                  "A6_vs_base": R.paired_stats(yt, dt, at, taxis["base"], taxis["A6"]),
                  "A7_vs_base": R.paired_stats(yt, dt, at, taxis["base"], taxis["A7"]),
                  "A6_vs_A7": R.paired_stats(yt, dt, at, taxis["A7"], taxis["A6"]),
                  "move_energy_A6_vs_base": R.move_energy(taxis["base"], taxis["A6"], at, month[test]),
                  "base_rmse": R.rmse(yt, taxis["base"]),
                  "base_rmse_minus_rcr2": R.rmse(yt, taxis["base"]) - RCR2_BASE_RMSE[leg],
                  "dfc_importance_share_A6": importance_share(imps["A6"], D.DFC_FEATS),
                  "dfc_gain_importance_A6": {c: imps["A6"][c] for c in D.DFC_FEATS},
                  "dfc_importance_share_A7": importance_share(imps["A7"], DEP_ONLY),
                  "top_row_y_A6_vs_base": float(yt[np.argmax(np.abs((yt - taxis["base"]) ** 2 - (yt - taxis["A6"]) ** 2))])}
        report[leg] = legrep
        preds_out.append(pd.DataFrame({"leg": leg, "doy": dt, "ap": at, "y": yt, "proxy": proxy[test],
                                       "taxi_base": taxis["base"], "taxi_A6": taxis["A6"], "taxi_A7": taxis["A7"]}))
        print(json.dumps({"leg": leg, **{k: {kk: vv for kk, vv in legrep[k].items() if kk != "per_airport_gain_s"}
                                         for k in ("A6_vs_base", "A7_vs_base", "A6_vs_A7")},
                          "base_rmse_minus_rcr2": legrep["base_rmse_minus_rcr2"],
                          "dfc_importance_share_A6": legrep["dfc_importance_share_A6"]}), flush=True)
    report["gate_G1_pass"] = gate_g1(report)
    report["predictions"] = predictions_scored(report, None)
    report.update(tag="DFC2", stage=1, seconds=time.monotonic() - started, peak_rss_GiB=G.peak_rss() / 1024 ** 3,
                  prereg_sha256=sha, smoke=smoke, rcr2_base_rmse=RCR2_BASE_RMSE,
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


# ---- Stage 2 -------------------------------------------------------------------------------------------------
RANKING = ROOT / "data" / "raw" / "ranking.parquet"
SERVE_APTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
YEAR_SERVE = 2026
SEEDS = R.SEEDS


def compose_delta(frame: pd.DataFrame, fd) -> np.ndarray:
    """cand = where(pop, 0.5*(taxi_C + max(proxy - Fd, 1)), taxi_v13_delta): rcr2_build.compose with fy = proxy - Fd."""
    fd = np.asarray(fd, float)
    if fd.shape != (len(frame),):
        raise ValueError(f"Fd shape {fd.shape} != {len(frame)} rows")
    return R.compose(frame, frame.proxy.to_numpy(float) - fd)


def serve_frame_2026(path=RANKING) -> pd.DataFrame:
    """The 2026 DEP rows under stand_ab's SERVE filter (stand_ab.py build_features serve=True), as INPUT_COLS with
    the clock derived exactly as stand_ab derives it (tmin = hour*60 + minute); a null ADES is the string 'NA'."""
    d = pd.read_parquet(path, columns=["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "SCHED_TIME_UTC_mvt"])
    s = d[(d.PHASE_mvt == "DEP") & d.AOBT_3_flt.notna() & d.MVT_TIME_UTC_mvt.notna() & d.SCHED_TIME_UTC_mvt.notna()
          & d.ADEP_mvt.isin(SERVE_APTS)]
    t = pd.to_datetime(s.MVT_TIME_UTC_mvt, utc=True)
    if set(t.dt.year.unique()) != {YEAR_SERVE}:
        raise AssertionError(f"serve frame is not {YEAR_SERVE}-only: {sorted(t.dt.year.unique())}")
    f = pd.DataFrame({"ADES_mvt": s.ADES_mvt.fillna("NA").astype(str).to_numpy(), "doy": t.dt.dayofyear.to_numpy(),
                      "hr": t.dt.hour.to_numpy(), "tmin": (t.dt.hour * 60 + t.dt.minute).to_numpy(),
                      "mvt_sec": t.dt.second.to_numpy()})
    if not np.array_equal(f.tmin.to_numpy() // 60, f.hr.to_numpy()):
        raise AssertionError("tmin // 60 != hr on the serve frame")
    return f[list(D.INPUT_COLS)]


def serve_guards_fire(frame26: pd.DataFrame, stream25, stream26, tables) -> tuple:
    """T7 (a): (the 2025 stream on the 2026 frame raises, a 2025 `year` declared for the 2026 stream raises)."""
    def raised(**kw):
        try:
            D.build_dfc(frame26, fit_mask=None, arrivals=kw["arrivals"], year=kw["year"], dev_tables=tables)
        except AssertionError as e:
            return "F1" in str(e)
        return False
    return raised(arrivals=stream25, year=YEAR_SERVE), raised(arrivals=stream26, year=YEAR)


def tripwire_measure(block25: pd.DataFrame, block26: pd.DataFrame, ades_fit, ades26, guards, cols=None) -> dict:
    """The T7 quantities: NaN share of the departure columns on 2026, per-column 2026 share outside the 2025 [min, max]
    (NaN-aware: the arrival columns over their observable rows), |mean26 - mean25| in 2025-sd units, in_set shares,
    unseen-ADES share (2026 rows whose destination never appears among the fit rows), and the two guard outcomes."""
    out = {"guard_raised_on_2025_stream": bool(guards[0]), "guard_raised_on_wrong_year": bool(guards[1]),
           "n_2025": int(len(block25)), "n_2026": int(len(block26)), "nan_share": {}, "out_of_range_share": {},
           "mean_shift_in_sd": {}, "range_2025": {}, "mean_2025": {}, "mean_2026": {}}
    for c in DEP_ONLY:
        out["nan_share"][c] = float(np.isnan(block26[c].to_numpy(float)).mean())
    for c in (list(D.DFC_FEATS) if cols is None else list(cols)):     # DFC.3 (no arrival stream) passes DEP_ONLY
        a = block25[c].to_numpy(float); b = block26[c].to_numpy(float)
        a = a[np.isfinite(a)]; bf = b[np.isfinite(b)]
        lo, hi = float(a.min()), float(a.max())
        out["range_2025"][c] = [lo, hi]
        out["out_of_range_share"][c] = float(((bf < lo) | (bf > hi)).sum() / len(b))
        sd = float(a.std())
        out["mean_2025"][c] = float(a.mean()); out["mean_2026"][c] = float(bf.mean())
        out["mean_shift_in_sd"][c] = float((bf.mean() - a.mean()) / sd) if sd > 0 else 0.0
    out["in_set_share_2025"] = float(block25.dfc_ades_in_set.mean())
    out["in_set_share_2026"] = float(block26.dfc_ades_in_set.mean())
    seen = set(np.asarray(ades_fit).astype(str).tolist())
    a26 = np.asarray(ades26).astype(str)
    out["unseen_ades_share"] = float(np.mean([x not in seen for x in a26]))
    return out


def fold_a_frame(feats) -> tuple:
    """Fold A exactly as lgbm_fold.load_fold builds it (12 months, no row filter, fold_masks, counts = LGBM_AB) but
    keeping the DFC inputs (load_fold drops ADES_mvt before its derive hook). Returns (frame, (te, tr, fit, es))."""
    S = _load("stand_ab", "scripts/stand_ab.py"); LF = _load("lgbm_fold", "scripts/lgbm_fold.py")
    frames, _ = LF._load_months(S.CACHE, list(range(1, 13)), feats, S.QCACHE, None, S.OCACHE, S.WCACHE)
    keep = ([c for c in feats if c not in S.ENC] + list(S.ENC_KEYS) + ["y", "delta", "proxy", "month", "doy", "ap"]
            + list(D.INPUT_COLS))
    keep = list(dict.fromkeys(keep))
    frames = [f[keep] for f in frames]
    d = pd.concat(frames, ignore_index=True); del frames; gc.collect()
    te, tr, fit, es = LF.fold_masks(d.month.to_numpy())
    for k, want in (("n_test", int(te.sum())), ("n_fit", int(fit.sum())), ("n_es", int(es.sum())), ("n_all", int(tr.sum()))):
        if LF.LGBM_AB[k] != want:
            raise RuntimeError(f"fold row count {k}={want:,} differs from lgbm_ab's {LF.LGBM_AB[k]:,}: not the same fold")
    return d, (te, tr, fit, es)


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
    feats = arm_feats("A6", S)
    if smoke:
        LF = _load("lgbm_fold", "scripts/lgbm_fold.py")
        # months 1-3 in full (as rcr2's smoke): January's fold `row` numbers then coincide with the 12-month
        # numbering the incumbent frame uses, so the row join below is exact even in the smoke
        frames, _ = LF._load_months(S.CACHE, [1, 2, 3], arm_feats("base", S), S.QCACHE, None, S.OCACHE, S.WCACHE)
        d = pd.concat(frames, ignore_index=True); del frames
        te, tr, fit, es = LF.fold_masks(d.month.to_numpy())
    else:
        d, (te, tr, fit, es) = fold_a_frame(arm_feats("base", S))
    labels = R.check_labels(d.y.to_numpy(float))
    y, dlt, proxy = d.y.to_numpy(float), d.delta.to_numpy(float), d.proxy.to_numpy(float)
    month = d.month.to_numpy().astype(int); ap_all = d.ap.to_numpy().astype(str); doy = d.doy.to_numpy().astype(int)
    arrivals = load_arrivals_2025()
    for c, v in S.infold_encodings(d[S.ENC_KEYS], y, dlt, tr).items():          # TRAINING rows, as load_fold
        d[c] = v
    attach(d, dfc_block(d, tr, arrivals))                                        # medians on tr only (leak gate 2)
    dev_tables = D.fit_dev_tables(d[list(D.INPUT_COLS)], tr, YEAR, arrivals)   # the serve reference (T7 provenance)
    block25_holdout = d.loc[te, list(D.DFC_FEATS)].reset_index(drop=True)      # Jan+Jul 2025, built with the tr table
    ades_fit = d.ADES_mvt.astype(str).to_numpy()[tr]
    X = LS.design_matrix(d, feats); del d; gc.collect()
    print(json.dumps({"stage": "fold", "n": int(len(y)), "n_feats": len(feats), "secs": round(time.monotonic() - started, 1),
                      "peak_GiB": round(G.peak_rss() / 1024 ** 3, 2)}), flush=True)
    p = dict(LS.P)
    nest, patience = (60, 50) if smoke else (LS.NEST, LS.PATIENCE)
    if smoke:
        p["learning_rate"] = LS.SMOKE["learning_rate"]
    info = LS.early_stop(X, dlt, y, proxy, tr, fit, es, p, nest, patience, target="delta")
    label = LS.regression_label(dlt, y, "delta")
    preds = LS.fit_seeds(X, label, tr, X[te], p, info["n_ref"], SEEDS, n_features=len(feats), target="delta")
    del X; gc.collect()
    te_idx = np.flatnonzero(te)
    base = pd.DataFrame({"row": te_idx.astype("int64"), "month": month[te], "ap": ap_all[te], "y": y[te], "proxy": proxy[te], "doy": doy[te]})
    for s_ in SEEDS:
        base[f"d_seed{s_}"] = preds[s_]
    fd = np.mean([preds[s_] for s_ in SEEDS], axis=0)
    if not np.all(np.isfinite(fd)):
        raise RuntimeError("non-finite Fd")
    base["Fd"] = fd
    base.to_parquet(PREDS2 if not smoke else ROOT / "reports" / "dfc2_fold_predictions_smoke.parquet", index=False)
    seed_rmses = [R.rmse(base.y, np.maximum(base.proxy - base[f"d_seed{s_}"], 1)) for s_ in SEEDS]
    taxi_fd = np.maximum(base.proxy.to_numpy() - fd, 1)
    floor_rate = float((base.proxy.to_numpy() - fd < 1.0).mean())
    # ---- T7: the 2026 serve construction, label-free, before any verdict ----
    f26 = serve_frame_2026()
    if smoke:
        f26 = f26.iloc[::SMOKE_STRIDE].reset_index(drop=True)
    arrivals26 = D.load_arrivals([RANKING], year=YEAR_SERVE)
    guards = serve_guards_fire(f26.iloc[:2000], arrivals, arrivals26, dev_tables)
    block26 = dfc_block(f26, None, arrivals26, year=YEAR_SERVE, dev_tables=dev_tables)
    tw = tripwire_eval(tripwire_measure(block25_holdout, block26, ades_fit, f26.ADES_mvt.to_numpy(), guards))
    print(json.dumps({"stage": "T7", "pass": tw["pass"], "clauses": tw["clauses"], "n_2026": int(len(f26))}), flush=True)
    # ---- composition against the incumbent ----
    inc = R.load_incumbent(subset_ok=smoke)
    j = inc.merge(base[["row", "y", "proxy", "Fd"]].rename(columns={"y": "b_y", "proxy": "b_proxy"}), on="row",
                  how="inner" if smoke else "left", validate="one_to_one")
    if not smoke:
        assert len(j) == R.N_INC and j.Fd.notna().all(), "fold rows do not cover the incumbent frame"
    assert np.array_equal(j.y, j.b_y) and np.array_equal(j.proxy, j.b_proxy), "row join is not the same row"
    cand = compose_delta(j, j.Fd.to_numpy())
    comp = R.delta_stats(j.y, j.doy, j.ap, j.month, j.inc.to_numpy(), cand)
    fa = base.merge(R.load_fw_all().drop(columns="doy"), on="row", how="inner" if smoke else "left", validate="one_to_one")
    if not smoke:
        assert len(fa) == len(base) and fa.FW.notna().all(), "FW does not cover every fold row"
    f_alone = R.delta_stats(fa.y, fa.doy, fa.ap, fa.month, np.maximum(fa.proxy - fa.FW, 1), np.maximum(fa.proxy - fa.Fd, 1))
    verdict = stage2_verdict(comp, f_alone, floor_rate, seed_rmses, tw)
    report = {"tag": "DFC2", "stage": 2, "verdict": verdict, "composed": comp, "f_alone": f_alone, "tripwires": tw, "labels": labels,
              "best_iter": info["best_iter"], "n_ref": info["n_ref"], "es_info": info, "seed_rmses": seed_rmses,
              "floor_bind_rate": floor_rate, "rmse_taxi_fd_all_fold_rows": R.rmse(base.y, taxi_fd),
              "seconds": time.monotonic() - started, "peak_rss_GiB": G.peak_rss() / 1024 ** 3, "prereg_sha256": sha, "smoke": smoke,
              "n_incumbent_rows": int(len(j)), "n_pop": int(j["pop"].sum())}
    report["predictions"] = predictions_scored(None, report)
    report["predictions"]["P8_stage2_wall_and_rss"] = "HIT" if (report["seconds"] <= 10800 and report["peak_rss_GiB"] < 6) else "MISS"
    if smoke:
        print(json.dumps({"smoke": "passed; scores not inspected", "verdict": verdict["verdict"], "T7": tw["clauses"]})); G.release_lock(fh); return report
    with REPORT2.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False, default=float)
    print(json.dumps({"verdict": verdict, "composed": {k: v for k, v in comp.items() if k != "per_airport"},
                      "f_alone": {k: v for k, v in f_alone.items() if k != "per_airport"}, "T7": tw["clauses"],
                      "predictions": report["predictions"]}, indent=1), flush=True)
    G.release_lock(fh)
    return report


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("mode", choices=["stage1", "stage2"]); a.add_argument("--smoke", action="store_true")
    args = a.parse_args()
    (run_stage1 if args.mode == "stage1" else run_stage2)(args.smoke)
