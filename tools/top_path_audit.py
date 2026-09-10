"""Read-only diagnostics for plans/TOP_PATH_2026_09_10.md; never fit or submit.

Run: OMP_NUM_THREADS=1 python3.11 -B tools/top_path_audit.py
Writes only reports/top_path_audit.json. Existing predictions/caches stay unchanged.
Ceilings below use visible 2025 labels and are NOT estimates of achievable gains.
"""
from __future__ import annotations

import json
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WM = 339551 / 344841
WU = 5290 / 344841
RAW_COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
            "ADES_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt",
            "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]


def counterpart_candidates(t):
    """Exact route + flight number, one arrival in (takeoff, takeoff+6h).

    This is a candidate census, not a trusted matching algorithm. Times are
    observed movement times; no departure target or off-block enters the join.
    """
    d = t[t.PHASE_mvt.eq("DEP")].copy()
    a = t[t.PHASE_mvt.eq("ARR") & t.AOBT_3_flt.notna()].copy()
    # Restrict the merge to two candidate dates to avoid a month-wide Cartesian
    # expansion for flights repeated daily. The six-hour window can cross midnight.
    d["arrival_day"] = d.MVT_TIME_UTC_mvt.dt.floor("D")
    a["arrival_day"] = a.MVT_TIME_UTC_mvt.dt.floor("D")
    d = pd.concat([d, d.assign(arrival_day=d.arrival_day + pd.Timedelta(days=1))])
    keys = ["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "arrival_day"]
    c = d.dropna(subset=keys).merge(a.dropna(subset=keys), on=keys,
                                   suffixes=("_dep", "_arr"))
    gap = (c.MVT_TIME_UTC_mvt_arr - c.MVT_TIME_UTC_mvt_dep).dt.total_seconds()
    c = c[(gap > 0) & (gap < 21600)]
    counts = c.groupby("MVT_ID_mvt_dep").size()
    return c[c.MVT_ID_mvt_dep.isin(counts[counts.eq(1)].index)]


def matched_raw(t):
    d = t[t.PHASE_mvt.eq("DEP") & t.AOBT_3_flt.notna()
          & t.TAXITIME_SEC_mvt.gt(0)]
    return d.sort_values("MVT_TIME_UTC_mvt", kind="stable").reset_index(drop=True)


def daily_bootstrap(frame, draws=2000):
    """Paired date blocks, stratified by month; all airports travel together.

    Conditional on these two observed months. Does not measure year shift or
    erase selection bias from repeatedly inspecting the same fold.
    """
    rng = np.random.default_rng(0)
    old = np.zeros(draws)
    new = np.zeros(draws)
    for _, month in frame.groupby("month"):
        g = month.groupby("date").agg(n=("se", "size"), old=("old_se", "sum"),
                                       new=("se", "sum"))
        idx = rng.integers(0, len(g), size=(draws, len(g)))
        n = g.n.to_numpy()[idx].sum(axis=1)
        weight = len(month) / len(frame)
        old += weight * g.old.to_numpy()[idx].sum(axis=1) / n
        new += weight * g.new.to_numpy()[idx].sum(axis=1) / n
    gain = np.sqrt(old) - np.sqrt(new)
    return {"draws": draws, "seed": 0, "date_blocks": int(frame.date.nunique()),
            "gain_s_ci95": np.quantile(gain, [.025, .975]).tolist(),
            "weighted_mse_gain_ci95": (WM * np.quantile(old-new, [.025, .975])).tolist()}


def main():
    q = pd.read_parquet(ROOT / "data/cache_stand/fold_preds_queue.parquet",
                        columns=["month", "ap", "y", "proxy", "delta", "sp",
                                 "baseline", "treatment"])
    u = pd.read_parquet(ROOT / "data/cache_stand/stratum_fold_v7_preds.parquet")
    u = u[u.fold.eq("A")].copy()
    assert len(q) == 339015 and len(u) == 5321
    q["prediction"] = np.maximum(q.proxy - q.treatment, 1)
    q["se"] = (q.y - q.prediction) ** 2
    q["old_se"] = (q.y - np.maximum(q.proxy - q.baseline, 1)) ** 2
    u["se"] = (u.y - u.S1) ** 2
    report = {"status": "diagnostic; no new model fitted; no score projection",
              "run_utc": datetime.now(timezone.utc).isoformat(),
              "score_source": "User screenshot / local September 10 snapshot; not a live API refresh",
              "score_snapshot": {"ours": 288.1406, "tenth": 275.9373, "leader": 245.2901},
              "gaps_mse": {"tenth": 288.1406**2-275.9373**2,
                           "leader": 288.1406**2-245.2901**2},
              "counterpart_census": {}}
    spec = importlib.util.spec_from_file_location("stand_ab", ROOT / "scripts/stand_ab.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    keys = pd.DataFrame({"k": ["x"] * 4})
    y = np.array([100., 200., 300., 400.])
    tr = np.array([True, True, True, False])  # two fit, one inner stop, one outer holdout
    encode = lambda v: s.infold_encodings(keys, v, v, tr, enc_keys=["k"])["de_k"]
    stop_change, hold_change = y.copy(), y.copy()
    stop_change[2] += 1000
    hold_change[3] += 1000
    assert not np.array_equal(encode(y), encode(stop_change))
    assert np.array_equal(encode(y), encode(hold_change))
    report["encoding_dependency_check"] = {
        "inner_stop_label_changes_fit_and_stop_features": True,
        "outer_holdout_label_changes_features": False,
        "scope": "Synthetic dependency check of existing encoder; no model gain measured."}
    raw_holdout = []
    candidate_ids = []
    for name in ["training_2025-01-01_2025-02-01.parquet",
                 "training_2025-07-01_2025-08-01.parquet", "ranking.parquet"]:
        t = pd.read_parquet(ROOT / "data/raw" / name, columns=RAW_COLS)
        d = t[t.PHASE_mvt.eq("DEP")]
        c = counterpart_candidates(t)
        uc = c[c.AOBT_3_flt_dep.isna()]
        mc = c[c.FLIGHT_ID_mvt_dep.notna() & c.FLIGHT_ID_mvt_arr.notna()]
        same = mc.FLIGHT_ID_mvt_dep.eq(mc.FLIGHT_ID_mvt_arr)
        a = t[t.PHASE_mvt.eq("ARR") & t.FLIGHT_ID_mvt.notna()]
        byid = d[d.FLIGHT_ID_mvt.notna()].merge(a, on="FLIGHT_ID_mvt",
                                               suffixes=("_dep", "_arr"))
        offsets = (byid.AOBT_3_flt_dep - byid.AOBT_3_flt_arr).dt.total_seconds()
        report["counterpart_census"][name] = {
            "departures": len(d), "unmatched": int(d.AOBT_3_flt.isna().sum()),
            "unmatched_unique_candidates": len(uc),
            "candidate_precision_on_known_id_pairs": float(same.mean()),
            "known_id_candidates": len(mc),
            "same_id_pairs": len(byid),
            "same_id_aobt_disagreements": int(offsets.abs().gt(0).sum()),
            "caveat": "Known-id precision need not transfer to unmatched rows."}
        if name != "ranking.parquet":
            raw_holdout.append(matched_raw(t))
            candidate_ids.extend(uc.MVT_ID_mvt_dep.tolist())
    raw = pd.concat(raw_holdout, ignore_index=True)
    assert len(raw) == len(q)
    np.testing.assert_array_equal(raw.TAXITIME_SEC_mvt.to_numpy(), q.y.to_numpy())
    np.testing.assert_array_equal(raw.ADEP_mvt.to_numpy(), q.ap.to_numpy())
    px = (raw.MVT_TIME_UTC_mvt - raw.AOBT_3_flt).dt.total_seconds().to_numpy()
    np.testing.assert_array_equal(px, q.proxy.to_numpy())
    q["date"] = raw.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy()
    report["matched_v5_fold_rmse"] = float(np.sqrt(q.se.mean()))
    report["matched_by_airport"] = []
    for ap, g in q.groupby("ap"):
        report["matched_by_airport"].append({"airport": ap, "n": len(g),
            "rmse": float(np.sqrt(g.se.mean())),
            "sse_share": float(g.se.sum()/q.se.sum()),
            "contribution_at_scored_stratum_weight": float(WM*g.se.sum()/len(q))})
    report["queue_vs_previous_date_bootstrap"] = daily_bootstrap(q)
    report["matched_diagnostic_ceilings_mse"] = {
        "perfect_on_schedule_fills_only": float(WM*q.loc[(q.y-q.sp).abs().le(60), "se"].sum()/len(q)),
        "perfect_on_clock_disagreement_gt_600_only": float(WM*q.loc[q.delta.abs().gt(600), "se"].sum()/len(q)),
        "perfect_on_lirf_only": float(WM*q.loc[q.ap.eq("LIRF"), "se"].sum()/len(q))}
    uc = u[u.MVT_ID_mvt.isin(candidate_ids)]
    report["unmatched_counterpart_candidate_ceiling"] = {
        "fold_A_candidates": len(uc),
        "perfect_prediction_gain_at_scored_stratum_weight": float(WU*uc.se.sum()/len(u)),
        "all_unmatched_fold_mse_at_scored_stratum_weight": float(WU*u.se.mean()),
        "caveat": "Perfect 2025 subset prediction, not an achievable or 2026 gain."}
    report["unmatched_fold_top_error_share"] = {
        str(n): float(u.nlargest(n, "se").se.sum()/u.se.sum()) for n in (1, 2, 5, 20)}
    report["unmatched_fold_by_airport"] = [
        {"airport": ap, "n": len(g), "rmse": float(np.sqrt(g.se.mean())),
         "sse_share": float(g.se.sum()/u.se.sum())} for ap, g in u.groupby("ADEP_mvt")]
    out = ROOT / "reports/top_path_audit.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
