"""MXJ: v25's unmatched body with early-stopping patience 50 instead of sklearn's default 10.

Registration: plans/PREREG_mxj_unmatched_patience_2026_09_21.md (sha256 b6da70bd0a98b95cdb5181aa82aa266dbe5e0ba31891794816a0c8c266f337fc),
hashed 2026-09-21T13:47:20Z before this runner existed or any number was measured.

Same two legs as MXI (mau_build's Jan/Jul screen, mav_build's Oct-Dec forward), with MXI's own pure helpers imported from
`mxi_build` (composition, stats, verdict, gates, preflight). The incumbent is v25's body and must reproduce MXI's persisted
`it1000` columns EXACTLY on both legs, candidate and per seed, or the run is void.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/mxj_build.py
"""
from __future__ import annotations

import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts", ROOT / "scripts/analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import mxi_build as XI       # noqa: E402  (MXI's runner: imported, never edited)

MB, U, uc = XI.MB, XI.U, XI.uc
PREREG = ROOT / "plans/PREREG_mxj_unmatched_patience_2026_09_21.md"
REGISTERED_SHA = "b6da70bd0a98b95cdb5181aa82aa266dbe5e0ba31891794816a0c8c266f337fc"
HASHLOG = ROOT / "reports/mxj_prereg_hashes.log"
REPORT = ROOT / "reports/mxj_report.json"
PREDS = ROOT / "reports/mxj_predictions.parquet"
MXI_REPORT = ROOT / "reports/mxi_report.json"
MXI_PREDS = ROOT / "reports/mxi_predictions.parquet"
ARMS = {"inc": XI.body_params(XI.MXI_ITER), "arm": {**XI.body_params(XI.MXI_ITER), "n_iter_no_change": 50}}
CAP = XI.MXI_ITER
ALARM_S = 300


def fit_body(x_fit, y_fit, design_extra, y_extra, seed, x_val, y_val, params):
    """mxi_build.fit_body with the whole parameter dict free (MAU's evaluation order for the weight)."""
    weight = 0.25 * len(x_fit) / len(design_extra)
    tx = pd.concat([x_fit, design_extra], ignore_index=True)
    ty = np.r_[np.asarray(y_fit, float), np.clip(np.asarray(y_extra, float), -3000, 3000)]
    sw = np.r_[np.ones(len(x_fit)), np.full(len(design_extra), weight)]
    model = MB.HGR(random_state=seed, early_stopping=True, **params)
    model.fit(tx, ty, sample_weight=sw, X_val=x_val, y_val=y_val, sample_weight_val=np.ones(len(x_val)))
    return model


def _fit_both(train, te, ex, seed, fits, leg, month):
    ctrl = uc.fit_congestion_regressor(train, seed)
    x = ctrl.design(train)
    yy = np.clip(train.y.to_numpy(float), -3000, 3000)
    fi, va = MB.stopping_split(len(train), seed)
    dex, dte = ctrl.design(ex), ctrl.design(te)
    out = {}
    for name, params in ARMS.items():
        t = time.monotonic()
        model = fit_body(x.iloc[fi], yy[fi], dex, ex.y.to_numpy(float), seed, x.iloc[va], yy[va], params)
        out[name] = model.predict(dte)
        fits.append({"leg": leg, "month": int(month), "seed": int(seed), "arm": name, "n_iter": int(model.n_iter_),
                     "seconds": round(time.monotonic() - t, 2)})
    return out


def _stack(te, rows_keep, p_hat, sp, cells, preds):
    out = te[["MVT_ID_mvt", "ADEP_mvt", "date", "month", "y", "sp", "A"]].copy()
    out["day"] = te.MVT_TIME_UTC_mvt.dt.dayofyear.to_numpy()
    for name in ARMS:
        stack = np.asarray(preds[name])
        out[name] = XI.compose(p_hat, sp, cells, stack.mean(axis=0), te.A)
        for j, seed in enumerate(U.SEEDS):
            out[f"{name}_seed{seed}"] = XI.compose(p_hat, sp, cells, stack[j], te.A)
    return out.loc[rows_keep].copy()


def _gate(result, reference, what):
    XI.assert_reproduces(result.MVT_ID_mvt, reference.MVT_ID_mvt, f"{what} row ids")
    XI.assert_reproduces(result.inc, reference.it1000, f"{what} it1000")
    for s in U.SEEDS:
        XI.assert_reproduces(result[f"inc_seed{s}"], reference[f"it1000_seed{s}"], f"{what} seed {s}")


def _stats(frame):
    y, day, month = frame.y.to_numpy(float), frame.day.to_numpy(), frame.month.to_numpy()
    inc, arm = frame.inc.to_numpy(float), frame.arm.to_numpy(float)
    per_row = pd.Series(((y - inc) ** 2 - (y - arm) ** 2) / 344841)
    return {"delta": XI.delta_stats(y, inc, arm, day, month), "oracle": XI.delta_stats(y, inc, y, day, month),
            "identity": XI.delta_stats(y, inc, inc.copy(), day, month),
            "delta_without_monsters": XI.without_monsters(frame.MVT_ID_mvt, y, inc, arm),
            "seed_deltas": {str(s): U.pbl.board_gain(y, frame[f"inc_seed{s}"], frame[f"arm_seed{s}"]) for s in U.SEEDS},
            "airport_deltas": {str(k): float(v) for k, v in per_row.groupby(frame.ADEP_mvt.to_numpy()).sum().items()},
            "rows_changed": int((inc != arm).sum()), "n": int(len(frame))}


def assemble(screen, forward, fits):
    s, f = _stats(screen), _stats(forward)
    for leg in (s, f):
        assert leg["identity"]["gain"] == 0.0 and leg["identity"]["ci95"] == [0.0, 0.0], "VOID: identity"
    lf = screen.ADEP_mvt.ne("LFPG").to_numpy()
    lfpg = XI.delta_stats(screen.y[lf], screen.inc[lf], screen.arm[lf], screen.day[lf], screen.month[lf])
    report = {"tag": "MXJ", "verdict": XI.verdict(s["delta"], f["delta"], lfpg), "screen": s, "forward": f,
              "screen_lfpg_excluded": lfpg,
              "arm_fits_at_cap": sum(x["n_iter"] >= CAP for x in fits if x["arm"] == "arm"),
              "arm_n_iter": sorted(x["n_iter"] for x in fits if x["arm"] == "arm"),
              "inc_n_iter": sorted(x["n_iter"] for x in fits if x["arm"] == "inc"), "fits": fits}
    preds = pd.concat([screen.assign(leg="screen"), forward.assign(leg="forward")], ignore_index=True)
    return report, preds


def screen_leg(extra, fits):
    """mxi_build.screen_leg with both arms."""
    frame = pd.read_parquet(ROOT / "data/cache_rome/unm.parquet")
    saved = pd.read_parquet(ROOT / "data/cache_stand/unm_physics_lomo.parquet")
    ref = saved.set_index("MVT_ID_mvt").reindex(frame.MVT_ID_mvt).reset_index()
    for c in ("y", "sp", "month"):
        np.testing.assert_array_equal(frame[c].to_numpy(), ref[c].to_numpy())
    for c in uc.WITNESS:
        frame[c] = ref[c].to_numpy()
    XI._attach_anomaly(frame, U.MONTHS)
    records = []
    for m in U.MONTHS:
        test = frame.month.eq(m).to_numpy()
        train = frame.loc[~test & ~uc.bs.schedule_fill(frame)]
        ex = extra.loc[extra.month.ne(m)]
        te, rr = frame.loc[test], ref.loc[test]
        preds = {name: [] for name in ARMS}
        for seed in U.SEEDS:
            for name, p in _fit_both(train, te, ex, seed, fits, "screen", m).items():
                preds[name].append(p)
        records.append(_stack(te, te.ADEP_mvt.ne("LIRF").to_numpy(), rr.p_hat, rr.sp, rr.nf_cells, preds))
        print(json.dumps({"leg": "screen", "month": m}), flush=True)
    return pd.concat(records, ignore_index=True)


def forward_leg(extra, fits):
    """mxi_build.forward_leg with both arms."""
    frame = uc.bs.derive(pd.read_parquet(ROOT / "data/cache_rome/unm.parquet"))
    saved = pd.read_parquet(ROOT / "data/cache_stand/unm_physics_lomo.parquet")
    ref = saved.set_index("MVT_ID_mvt").reindex(frame.MVT_ID_mvt)
    np.testing.assert_array_equal(frame.y.to_numpy(), ref.y.to_numpy())
    for c in uc.WITNESS:
        frame[c] = ref[c].to_numpy()
    XI._attach_anomaly(frame, XI.FORWARD_MONTHS)
    records = []
    for m in XI.FORWARD_MONTHS:
        tr = frame.loc[frame.month.lt(m)]
        te = frame.loc[frame.month.eq(m) & frame.ADEP_mvt.ne("LIRF")]
        ex = extra.loc[extra.month.lt(m)]
        train = tr.loc[~uc.bs.schedule_fill(tr)]
        assert set(train.month) == set(ex.month) == set(range(1, m))
        parts = {}
        uc.bs.fit_unmatched(tr, te, train_matched=None, hybrid=False, parts=parts)
        preds = {name: [] for name in ARMS}
        for seed in U.SEEDS:
            for name, p in _fit_both(train, te, ex, seed, fits, "forward", m).items():
                preds[name].append(p)
        records.append(_stack(te, np.ones(len(te), bool), parts["p"], te.sp, parts["nf_cells"], preds))
        print(json.dumps({"leg": "forward", "month": m}), flush=True)
    return pd.concat(records, ignore_index=True)


def main() -> dict:
    started = time.monotonic()
    signal.alarm(ALARM_S)
    XI.preflight(REPORT, PREDS, PREREG, HASHLOG, REGISTERED_SHA)
    mau, mav, mxi = (json.loads(p.read_text()) for p in (MB.REPORT, XI.MAV_REPORT, MXI_REPORT))
    for name, h in {**mau["input_hashes"], **mav["input_hashes"]}.items():
        assert U.sha(ROOT / name) == h, ("input changed since MAU/MAV", name)
    assert U.sha(MB.TRAIN) == mau["train_sha256"] and U.sha(MXI_PREDS) == mxi["predictions_sha256"]
    extra = pd.read_parquet(MB.TRAIN)
    mxi_preds = pd.read_parquet(MXI_PREDS)
    fits = []
    screen = screen_leg(extra, fits)
    _gate(screen, mxi_preds[mxi_preds.leg == "screen"].reset_index(drop=True), "MXI screen")
    forward = forward_leg(extra, fits)
    _gate(forward, mxi_preds[mxi_preds.leg == "forward"].reset_index(drop=True), "MXI forward")
    report, preds = assemble(screen, forward, fits)
    report.update(prereg_sha256=REGISTERED_SHA, runner_sha256=U.sha(__file__), mxi_runner_sha256=U.sha(XI.__file__),
                  seconds=round(time.monotonic() - started, 1),
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3)
    preds.to_parquet(PREDS, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(PREDS), preds)
    report["predictions_sha256"] = U.sha(PREDS)
    with REPORT.open("x") as fh:
        json.dump(report, fh, indent=1, allow_nan=False)
    signal.alarm(0)
    print(json.dumps({k: report[k] for k in ("verdict", "arm_fits_at_cap", "seconds")}), flush=True)
    return report


if __name__ == "__main__":
    main()
