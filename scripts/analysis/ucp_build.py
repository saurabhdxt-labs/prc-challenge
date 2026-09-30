"""One registered uncapped production-body comparison; no submission path."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import unm_congestion as uc
import congestion_anchor_ship as ca
import pbl_build as pbl

PREREG = ROOT / "plans/PREREG_ucp_uncapped_production_body_2026_09_20.md"
HASHLOG = ROOT / "reports/ucp_prereg_hashes.log"
REPORT = ROOT / "reports/ucp_report.json"
PREDS = ROOT / "reports/ucp_test_predictions.parquet"
MONTHS = (1, 7)
SEEDS = (0, 1, 2)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decide(st):
    if st["ci95"][1] < 0:
        return "WORSE"
    if (st["gain"] > 0 and st["ci95"][0] > 0
            and all(v["gain"] >= 0 for v in st["monthly"].values())):
        return "DEVELOPMENT IMPROVEMENT"
    return "NOT ESTABLISHED"


def score(frame, pred):
    return pbl.day_block_bootstrap(
        frame.y.to_numpy(float), frame.control.to_numpy(float),
        np.asarray(pred, float), frame.day.to_numpy(), frame.month.to_numpy())


def main():
    started = time.monotonic()
    signal.alarm(300)
    if REPORT.exists() or PREDS.exists():
        raise FileExistsError("UCP output already exists; refuses overwrite")
    registered = sha(PREREG)
    assert f"sha256={registered}" in HASHLOG.read_text(), "registration not hashed"
    source = ROOT / "data/cache_rome/unm.parquet"
    saved_path = ROOT / "data/cache_stand/unm_physics_lomo.parquet"
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (source, saved_path)}
    frame = pd.read_parquet(source)
    saved = pd.read_parquet(saved_path)
    assert len(frame) == len(saved) == 22219
    assert not frame.MVT_ID_mvt.duplicated().any()
    assert not saved.MVT_ID_mvt.duplicated().any()
    ref = saved.set_index("MVT_ID_mvt").reindex(frame.MVT_ID_mvt).reset_index()
    assert ref.y.notna().all()
    assert np.array_equal(frame.y.to_numpy(), ref.y.to_numpy())
    assert np.array_equal(frame.sp.to_numpy(), ref.sp.to_numpy())
    assert np.array_equal(frame.month.to_numpy(), ref.month.to_numpy())
    for c in uc.WITNESS:
        frame[c] = ref[c].to_numpy()
    assert uc.bs.WINSOR_S == uc.bs.T_TAIL_S == 3000

    # Actual shipped downstream rule, computed without target or off-block labels.
    anomalies = []
    raw_cols = ["ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
    for m in MONTHS:
        files = list((ROOT / "data/raw").glob(f"training_2025-{m:02d}-01_*.parquet"))
        assert len(files) == 1
        raw = pd.read_parquet(files[0], columns=raw_cols)
        anomalies.append(ca.day_anomaly(raw))
        del raw
    anomaly = pd.concat(anomalies).set_index(["ADEP_mvt", "date"])["A"]
    frame["date"] = frame.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    keys = pd.MultiIndex.from_arrays([frame.ADEP_mvt, frame.date])
    frame["A"] = anomaly.reindex(keys).to_numpy()

    scope = frame.month.isin(MONTHS).to_numpy() & frame.ADEP_mvt.ne("LIRF").to_numpy()
    oracle_frame = frame.loc[scope, ["y", "month"]].copy()
    oracle_frame["day"] = frame.loc[scope, "MVT_TIME_UTC_mvt"].dt.dayofyear.to_numpy()
    oracle_frame["control"] = ca.anchored_values(
        np.rint(ref.loc[scope, "S1C"].to_numpy()), frame.loc[scope, "A"].to_numpy())
    identity = score(oracle_frame, oracle_frame.control)
    oracle = score(oracle_frame, oracle_frame.y)
    assert identity["gain"] == 0 and identity["ci95"] == [0.0, 0.0]
    assert decide(oracle) == "DEVELOPMENT IMPROVEMENT"

    records, guards = [], []
    for m in MONTHS:
        test = frame.month.eq(m).to_numpy()
        train = frame.loc[~test]
        train = train.loc[~uc.bs.schedule_fill(train)]
        te, rr = frame.loc[test], ref.loc[test]
        predictions = {}
        for arm, cap in (("control", 3000.0), ("candidate", np.inf)):
            per_seed = []
            for seed in SEEDS:
                old_cap = uc.bs.WINSOR_S
                try:
                    uc.bs.WINSOR_S = cap
                    model = uc.fit_congestion_regressor(train, seed)
                finally:
                    uc.bs.WINSOR_S = old_cap
                per_seed.append(model.predict(te))
                del model
            predictions[arm] = np.asarray(per_seed)
            body = predictions[arm].mean(axis=0)
            mix = uc.bs.mixture(rr.p_hat, rr.sp, uc.bs.nf_hybrid(rr.nf_cells, body))
            if arm == "control":
                body_diff = float(np.max(np.abs(body - rr.nf_fit_c.to_numpy())))
                mix_diff = float(np.max(np.abs(mix - rr.S1C.to_numpy())))
                assert body_diff < 1e-6, (m, "body reproduction", body_diff)
                assert mix_diff < 1e-6, (m, "mixture reproduction", mix_diff)
                guards.append({"month": m, "body_max_abs_diff": body_diff,
                               "mixture_max_abs_diff": mix_diff})
            print(json.dumps({"month": m, "arm": arm, "seconds": time.monotonic()-started}), flush=True)
        out = te[["MVT_ID_mvt", "ADEP_mvt", "date", "month", "y", "sp", "A"]].copy()
        out["day"] = te.MVT_TIME_UTC_mvt.dt.dayofyear.to_numpy()
        for c in ("p_hat", "nf_cells", "routed_nf_fit"):
            out[c] = rr[c].to_numpy()
        for arm, stack in predictions.items():
            out[f"body_{arm}"] = stack.mean(axis=0)
            mix = uc.bs.mixture(rr.p_hat, rr.sp, uc.bs.nf_hybrid(rr.nf_cells, stack.mean(axis=0)))
            out[arm] = ca.anchored_values(np.rint(mix), te.A)
            for j, seed in enumerate(SEEDS):
                single = uc.bs.mixture(rr.p_hat, rr.sp, uc.bs.nf_hybrid(rr.nf_cells, stack[j]))
                out[f"{arm}_seed{seed}"] = ca.anchored_values(np.rint(single), te.A)
        out = out.loc[out.ADEP_mvt.ne("LIRF")].copy()
        out["gain_sse"] = (out.y-out.control)**2 - (out.y-out.candidate)**2
        # Routing remains fixed; a cell-routed row cannot change in this experiment.
        assert (out.loc[~out.routed_nf_fit, "control"] == out.loc[~out.routed_nf_fit, "candidate"]).all()
        records.append(out)
    result_rows = pd.concat(records, ignore_index=True)
    assert len(result_rows) == int(scope.sum())
    result_rows.to_parquet(PREDS, index=False)
    reread = pd.read_parquet(PREDS)
    pd.testing.assert_frame_equal(result_rows, reread)
    stats = score(reread, reread.candidate)
    seed_gains = {str(s): pbl.board_gain(reread.y, reread[f"control_seed{s}"], reread[f"candidate_seed{s}"])
                  for s in SEEDS}
    airport = (reread.groupby("ADEP_mvt").gain_sse.sum()/344841).to_dict()
    gains = reread.gain_sse.to_numpy()
    desc = np.sort(gains)[::-1]
    report = {"tag": "UCP", "verdict": decide(stats), "stats": stats,
              "seed_gains": seed_gains, "airport_gains": airport,
              "control_guards": guards, "instrument_oracle": oracle,
              "largest_positive_row_contribution": float(desc[0]/344841),
              "gain_excluding_top5_positive_rows": float((gains.sum()-desc[:5].sum())/344841),
              "scope": "Jan/Jul 2025 LOMO non-LIRF unmatched, S1C + shipped storm rule. Development SSE/344841; not a board forecast.",
              "prereg_sha256": registered, "runner_sha256": sha(__file__),
              "inputs": inputs, "prediction_sha256": sha(PREDS),
              "seconds": time.monotonic()-started,
              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3}
    with REPORT.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    signal.alarm(0)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
