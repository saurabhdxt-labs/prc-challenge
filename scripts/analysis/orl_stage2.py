"""ORL Stage 2 — fold A on the production recipe with the six ORL columns (plans/PREREG_orl_osm_route_length_2026_09_23.md,
"Stage 2", clauses T1-T6, predictions P5-P7).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/orl_stage2.py [--smoke]

`F_orl` = the 3-seed mean delta of the shipped `lgbm_submit.P` fitted on fold A (months 1, 7 held out; ES seed 0 on months
3, 9; in-fold encodings on the 10 training months) with the production 93 + the six ORL columns attached by
`orl_stage1.attach_to_cache_frame`. **No control refit**: the control is the stored FW fold prediction (registration).
Composed on the CBV rows against opc_screen's incumbent (208.31040447352203, re-derived by `rcr2_build.load_incumbent`),
reported in SSE/344,841 (old - new, positive = improvement) with the (month, day)-blocked CI and MDE; F-alone on every
fold row; both split into ORL-matched and unmatched rows. Label-free tripwires: the 2025 fold-A move energy on the CBV
rows by (month, airport) — the Stage-3 reference — and the 2026 (stand, runway) pairs with zero 2025 rows. The 2026 move
energy itself needs the Stage-3 12-month refit and is measured there (registration, Stage 3).

Reused by import and tested where they live: dfc2_build.fold_a_frame / compose_delta, rcr2_build (labels, incumbent,
delta_stats, stage2_verdict, move_energy, load_fw_all), orl_stage1 (arm_feats, attach, matched_share, gate_g1),
orl_build.unseen_key_share, lgbm_submit (design matrix, early_stop, fit_seeds), run_guards (heavy lock, RSS abort).
Refuses a silent rerun; refuses to run unless Stage 1's T1 passed. No upload, nothing under submissions/, no git.
"""
from __future__ import annotations

import argparse, gc, hashlib, importlib.util, json, pathlib, sys, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


S1 = _load("orl_stage1", "scripts/analysis/orl_stage1.py")
O = S1.O
R = _load("rcr2_build", "scripts/analysis/rcr2_build.py")
B = _load("dfc2_build", "scripts/analysis/dfc2_build.py")

PREREG, HASHLOG, REPORT1 = S1.PREREG, S1.HASHLOG, S1.REPORT1
REPORT2 = ROOT / "reports" / "orl_stage2_report.json"
PREDS2 = ROOT / "reports" / "orl_fold_predictions.parquet"
PREDS2_SMOKE = ROOT / "reports" / "orl_fold_predictions_smoke.parquet"
REPORT2_SMOKE = ROOT / "reports" / "orl_stage2_report_smoke.json"
RANKING = ROOT / "data" / "raw" / "ranking.parquet"
ROUTES_SHA = "4436b00edf558e8f451dba400cfb949c06935fe7fbed7dc20b232d0537f93590"   # leak gate 6
SEEDS = (0, 1, 2)
FW_BEST_ITER = 21_101
N_SERVE = 339_551                    # orl_audit_report.json n_serve: the serve key frame must be the audit's
SMOKE_STRIDE = 40
RSS_LIMIT = 6 * 1024 ** 3
WALL_MAX_S = 3.0 * 3600              # P7: Stage 2 <= 3.0 h
KEY_COLS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]


def prereg_sha(prereg=None, hashlog=None) -> str:
    """The registration's CURRENT sha256 must be logged as `(full <sha>)` on a line naming the file (appends are hashed)."""
    prereg = pathlib.Path(PREREG if prereg is None else prereg); hashlog = pathlib.Path(HASHLOG if hashlog is None else hashlog)
    h = hashlib.sha256(prereg.read_bytes()).hexdigest()
    if not any(f"(full {h})" in line and prereg.name in line for line in hashlog.read_text().splitlines()):
        raise AssertionError(f"registration sha256 {h} is not in {hashlog} on a line naming {prereg.name}")
    return h


def stage1_t1(stage1: dict) -> dict:
    """T1: G1 (both legs raw >= 0 AND minus-top-3 >= 0) AND in both legs the gain on matched rows >= 0."""
    g1 = bool(S1.gate_g1(stage1))
    mt = bool(all(stage1[leg]["arm_vs_ctrl"]["gain_matched_rows_s"] >= 0 for leg in S1.LEGS))
    return {"G1": g1, "matched_rows_nonneg": mt, "pass": g1 and mt}


def stage2_feats(S) -> tuple:
    """(control 93, arm 99): orl_stage1.arm_feats (leak gate by name, 93 asserted), and the arm width asserted."""
    ctrl, arm = S1.arm_feats(S)
    if len(arm) != 99 or arm[:93] != ctrl:
        raise AssertionError(f"the ORL arm is the production 93 + 6 columns, got {len(arm)}")
    return ctrl, arm


def split_by_matched(y, doy, ap, month, base, cand, matched) -> dict:
    """rcr2_build.delta_stats on the ORL-matched rows and on the unmatched rows separately (None when there are none)."""
    m = np.asarray(matched, bool)
    share = S1.matched_share(m)
    args = [np.asarray(a) for a in (y, doy, ap, month, base, cand)]
    out = {"matched_share": share, "matched": R.delta_stats(*(a[m] for a in args))}
    out["unmatched"] = R.delta_stats(*(a[~m] for a in args)) if (~m).any() else None
    return out


def seed_summary(y, proxy, preds: dict, seeds) -> dict:
    """Per-seed RMSE (s) of the floored taxi time max(proxy - d, 1), their mean, sd (ddof 1) and max |seed - mean|."""
    r = [R.rmse(y, np.maximum(np.asarray(proxy, float) - np.asarray(preds[s], float), 1.0)) for s in seeds]
    mean = float(np.mean(r))
    return {"rmse": r, "mean": mean, "sd": float(np.std(r, ddof=1)), "spread": float(max(abs(v - mean) for v in r))}


def floor_bind_rate(proxy, fd) -> float:
    """Share of rows where the prediction floor binds (proxy - F_orl < 1 s)."""
    return float((np.asarray(proxy, float) - np.asarray(fd, float) < 1.0).mean())


def verdict(t1: dict, comp: dict, f_alone: dict, floor_rate: float, seed_rmses, fd) -> dict:
    """T1-T6 of the registration, each named; WORKING iff every clause holds. T2-T5 and the floor/seed half of T6 are
    rcr2_build.stage2_verdict's (same thresholds: floor < 0.5 %, every seed within 2 s of the mean); T6 here adds
    'every prediction finite'. FALSE-shape wording: CI including 0 -> 'not established'; T5 failing while T2 passes is
    flagged (blend-only gain, v22's shape)."""
    v = R.stage2_verdict(comp, f_alone, floor_rate, seed_rmses)
    c = dict(v["clauses"])
    t6 = bool(c.pop("T6_floor_and_seeds") and np.all(np.isfinite(np.asarray(fd, float))))
    clauses = {"T1_stage1": bool(t1["pass"]), **c, "T6_floor_seeds_finite": t6}
    ok = all(clauses.values())
    word = "WORKING" if ok else ("NOT WORKING (not established: CI includes 0; see mde)" if not comp["ci_excludes_zero"] else "NOT WORKING")
    return {"verdict": word, "clauses": clauses, "seed_spread_s": v["seed_spread_s"],
            "blend_only_flag": bool(clauses["T2_comp_positive_ci"] and not clauses["T5_F_alone_positive_ci"])}


def predictions_scored(rep: dict) -> dict:
    """P5 (Delta_F in [-80, +200], Delta_comp in [-40, +120], CI half-width in [15, 120]), P6 (best_iter in
    [0.8, 1.2] x 21,101, largest row < 25 % of |Delta_comp|), P7's Stage-2 part (<= 3.0 h, peak RSS < 6 GiB)."""
    c, f = rep["composed"], rep["f_alone"]
    hw = (c["ci95"][1] - c["ci95"][0]) / 2
    h = lambda b: "HIT" if b else "MISS"
    p = {"P5_dF_band": h(-80 <= f["delta"] <= 200), "P5_dcomp_band": h(-40 <= c["delta"] <= 120),
         "P5_halfwidth": h(15 <= hw <= 120)}
    p["P5"] = h(all(p[k] == "HIT" for k in ("P5_dF_band", "P5_dcomp_band", "P5_halfwidth")))
    p["P6_best_iter"] = h(0.8 * FW_BEST_ITER <= rep["best_iter"] <= 1.2 * FW_BEST_ITER)
    p["P6_top1_share"] = h(c["top1_share"] < 0.25)
    p["P6"] = h(p["P6_best_iter"] == "HIT" and p["P6_top1_share"] == "HIT")
    p["P7_stage2_wall_rss"] = h(rep["seconds"] <= WALL_MAX_S and rep["peak_rss_GiB"] < 6)
    return p


def serve_keys_2026(path=RANKING) -> pd.DataFrame:
    """The 2026 DEP rows under stand_ab's SERVE filter (the filter of dfc2_build.serve_frame_2026 and the ORL audit's
    serve side), as the three ORL join keys. A non-2026 movement time is refused."""
    d = pd.read_parquet(path, columns=["PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt",
                                       "SCHED_TIME_UTC_mvt", "AOBT_3_flt"])
    s = d[(d.PHASE_mvt == "DEP") & d.AOBT_3_flt.notna() & d.MVT_TIME_UTC_mvt.notna() & d.SCHED_TIME_UTC_mvt.notna()
          & d.ADEP_mvt.isin(B.SERVE_APTS)]
    years = set(pd.to_datetime(s.MVT_TIME_UTC_mvt, utc=True).dt.year.unique().tolist())
    if years != {B.YEAR_SERVE}:
        raise AssertionError(f"serve frame is not {B.YEAR_SERVE}-only: {sorted(years)}")
    return pd.DataFrame({"ADEP_mvt": s.ADEP_mvt.astype(str).to_numpy(), "STAND_mvt": s.STAND_mvt.to_numpy(),
                         "RUNWAY_mvt": s.RUNWAY_mvt.to_numpy()})


def unseen_pairs(fit_keys_tr: pd.DataFrame, fit_keys_all: pd.DataFrame, serve_keys: pd.DataFrame) -> dict:
    """Per airport, the share of 2026 serve rows whose (stand, runway) pair / stand has zero rows in the fold-A training
    months and in all twelve 2025 months (orl_build.unseen_key_share; INV section 6.3, label-free)."""
    return {"unseen_2026_vs_foldA_training": O.unseen_key_share(fit_keys_tr, serve_keys),
            "unseen_2026_vs_all_2025": O.unseen_key_share(fit_keys_all, serve_keys)}


def tripwires(j: pd.DataFrame, cand, unseen: dict) -> dict:
    """Label-free (INV section 6.3): the 2025 fold-A move energy mean((cand - inc)^2) by (month, airport) on the CBV
    (`pop`) rows only — the registered Stage-3 reference — plus the unseen-pair shares (`unseen_pairs`)."""
    pop = j["pop"].to_numpy(bool)
    cand = np.asarray(cand, float)
    return {"move_energy_2025_pop_by_month_airport": R.move_energy(j.inc.to_numpy(float)[pop], cand[pop],
                                                                   j.ap.to_numpy()[pop], j.month.to_numpy()[pop]),
            **unseen,
            "move_energy_2026": "owed at Stage 3: needs the 99-feature 12-month refit's 2026 predictions (registration Stage 3)"}


def score_fold(base: pd.DataFrame, inc: pd.DataFrame, fw_all: pd.DataFrame, smoke: bool) -> dict:
    """Everything after the fits: the composed comparison vs the incumbent (joined by `row`, one-to-one, y/proxy
    re-asserted), F-alone vs FW on every fold row, both split by `orl_matched`, seeds, floor rate, 2025 move energy.
    `base` has row, month, ap, y, proxy, orl_matched, d_seed{s}, F_orl; `fw_all` has row, FW, doy (load_fw_all)."""
    fd = base.F_orl.to_numpy(float)
    if not np.all(np.isfinite(fd)):
        raise RuntimeError("non-finite F_orl")
    j = inc.merge(base[["row", "y", "proxy", "F_orl", "orl_matched"]].rename(columns={"y": "b_y", "proxy": "b_proxy"}),
                  on="row", how="inner" if smoke else "left", validate="one_to_one")
    if not smoke:
        assert len(j) == len(inc) and j.F_orl.notna().all(), "fold rows do not cover the incumbent frame"
    assert np.array_equal(j.y.to_numpy(), j.b_y.to_numpy()) and np.array_equal(j.proxy.to_numpy(), j.b_proxy.to_numpy()), \
        "row join is not the same row"
    cand = B.compose_delta(j, j.F_orl.to_numpy(float))
    args = (j.y.to_numpy(), j.doy.to_numpy(), j.ap.to_numpy(), j.month.to_numpy(), j.inc.to_numpy(float), cand)
    comp = R.delta_stats(*args)
    comp_split = split_by_matched(*args, j.orl_matched.to_numpy() == 1)
    fa = base.drop(columns=[c for c in ("doy",) if c in base.columns]).merge(
        fw_all[["row", "FW", "doy"]], on="row", how="inner" if smoke else "left", validate="one_to_one")
    if not smoke:
        assert len(fa) == len(base) and fa.FW.notna().all(), "FW does not cover every fold row"
    fargs = (fa.y.to_numpy(), fa.doy.to_numpy(), fa.ap.to_numpy(), fa.month.to_numpy(),
             np.maximum(fa.proxy.to_numpy(float) - fa.FW.to_numpy(float), 1.0),
             np.maximum(fa.proxy.to_numpy(float) - fa.F_orl.to_numpy(float), 1.0))
    f_alone = R.delta_stats(*fargs)
    f_split = split_by_matched(*fargs, fa.orl_matched.to_numpy() == 1)
    seeds = seed_summary(base.y.to_numpy(float), base.proxy.to_numpy(float),
                         {s: base[f"d_seed{s}"].to_numpy(float) for s in SEEDS}, SEEDS)
    pop = j["pop"].to_numpy(bool)
    return {"composed": comp, "composed_by_match": comp_split, "f_alone": f_alone, "f_alone_by_match": f_split,
            "seeds": seeds, "floor_bind_rate": floor_bind_rate(base.proxy.to_numpy(float), fd),
            "n_incumbent_rows": int(len(j)), "n_pop": int(pop.sum()), "n_fold_rows": int(len(base)),
            "_j": j, "_cand": cand}


def run_stage2(smoke: bool = False) -> dict:
    started = time.monotonic()
    if not smoke and REPORT2.exists():
        raise FileExistsError("Stage 2 result exists; no silent rerun")
    t1 = stage1_t1(json.loads(pathlib.Path(REPORT1).read_text()))
    if not t1["pass"]:
        raise RuntimeError(f"Stage 1 T1 failed ({t1}): Stage 2 is not run (registration)")
    G = _load("run_guards", "scripts/run_guards.py")
    fh = G.acquire_lock(); G.start_rss_guard(RSS_LIMIT)
    sha = prereg_sha()
    rsha = G.sha256_file(O.ROUTES)
    if rsha != ROUTES_SHA:
        raise AssertionError(f"route table sha256 {rsha} is not the registered {ROUTES_SHA}")
    S = _load("stand_ab", "scripts/stand_ab.py"); LS = _load("lgbm_submit", "scripts/lgbm_submit.py")
    ctrl, feats = stage2_feats(S)
    d, (te, tr, fit, es) = B.fold_a_frame(ctrl)                    # counts asserted against LGBM_AB inside
    d = S1.attach_to_cache_frame(d, pd.read_parquet(O.ROUTES))
    rowid = np.arange(len(d), dtype="int64")
    keys = d[["ap", "STAND_mvt", "RUNWAY_mvt"]].rename(columns={"ap": "ADEP_mvt"})
    serve = serve_keys_2026()
    if not smoke and len(serve) != N_SERVE:
        raise AssertionError(f"serve key frame has {len(serve):,} rows, the audit's has {N_SERVE:,}")
    unseen = unseen_pairs(keys[tr].reset_index(drop=True), keys.reset_index(drop=True), serve)   # before the fits:
    del keys, serve; gc.collect()                                                                # no 2 M key strings held
    if smoke:
        keep = rowid % SMOKE_STRIDE == 0
        d = d[keep].reset_index(drop=True); rowid = rowid[keep]
        te, tr, fit, es = te[keep], tr[keep], fit[keep], es[keep]
    labels = R.check_labels(d.y.to_numpy(float))
    y, dlt, proxy = d.y.to_numpy(float), d.delta.to_numpy(float), d.proxy.to_numpy(float)
    month = d.month.to_numpy().astype(int); ap_all = d.ap.to_numpy().astype(str); doy = d.doy.to_numpy().astype(int)
    flag = d[O.ORL_FLAG].to_numpy()
    share_all, share_test = S1.matched_share(flag), S1.matched_share(flag[te])    # leak gate 5
    for c, v in S.infold_encodings(d[S.ENC_KEYS], y, dlt, tr).items():          # TRAINING rows, as load_fold
        d[c] = v
    X = LS.design_matrix(d, feats); del d; gc.collect()
    print(json.dumps({"stage": "fold", "n": int(len(y)), "n_feats": len(feats), "n_test": int(te.sum()),
                      "matched_share_test": share_test, "secs": round(time.monotonic() - started, 1),
                      "peak_GiB": round(G.peak_rss() / 1024 ** 3, 2)}), flush=True)
    p = dict(LS.P)
    nest, patience = (60, 50) if smoke else (LS.NEST, LS.PATIENCE)
    if smoke:
        p["learning_rate"] = LS.SMOKE["learning_rate"]
    info = LS.early_stop(X, dlt, y, proxy, tr, fit, es, p, nest, patience, target="delta")
    label = LS.regression_label(dlt, y, "delta")
    preds = LS.fit_seeds(X, label, tr, X[te], p, info["n_ref"], SEEDS, n_features=len(feats), target="delta")
    del X; gc.collect()
    base = pd.DataFrame({"row": rowid[te], "month": month[te], "ap": ap_all[te], "y": y[te], "proxy": proxy[te],
                         "doy": doy[te], "orl_matched": flag[te]})
    for s_ in SEEDS:
        base[f"d_seed{s_}"] = preds[s_]
    base["F_orl"] = np.mean([preds[s_] for s_ in SEEDS], axis=0)
    base.to_parquet(PREDS2_SMOKE if smoke else PREDS2, index=False)
    sc = score_fold(base, R.load_incumbent(subset_ok=smoke), R.load_fw_all(), smoke)
    tw = tripwires(sc.pop("_j"), sc.pop("_cand"), unseen)
    v = verdict(t1, sc["composed"], sc["f_alone"], sc["floor_bind_rate"], sc["seeds"]["rmse"], base.F_orl.to_numpy())
    report = {"tag": "ORL", "stage": 2, "verdict": v, "stage1_T1": t1, **sc, "tripwires": tw, "labels": labels,
              "best_iter": info["best_iter"], "n_ref": info["n_ref"], "es_info": info, "matched_share_all": share_all,
              "matched_share_test": share_test, "n_feats": len(feats), "seconds": time.monotonic() - started,
              "peak_rss_GiB": G.peak_rss() / 1024 ** 3, "prereg_sha256": sha, "routes_sha256": rsha, "smoke": smoke,
              "scope": "Stage 2: fold A, shipped P, 3 seeds, F_orl vs the stored FW (no control refit); 2025 only, not a board number"}
    report["predictions"] = predictions_scored(report)
    if smoke:
        REPORT2_SMOKE.write_text(json.dumps(report, indent=2, allow_nan=True, default=float))   # serialisation exercised
        print(json.dumps({"smoke": "passed; scores not inspected", "keys": sorted(report)}), flush=True)
        G.release_lock(fh); return report
    with REPORT2.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=True, default=float)
    print(json.dumps({"verdict": v, "composed": {k: x for k, x in sc["composed"].items() if k != "per_airport"},
                      "f_alone": {k: x for k, x in sc["f_alone"].items() if k != "per_airport"},
                      "predictions": report["predictions"]}, indent=1, default=float), flush=True)
    G.release_lock(fh)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    run_stage2(smoke=a.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
