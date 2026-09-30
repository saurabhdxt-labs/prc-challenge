"""MXI: v24's unmatched body with max_iter 1000 (early stopping ends the fit) vs the shipped 200.

Registration: plans/PREREG_mxi_unmatched_capacity_2026_09_21.md (sha256 e98627ad8184f115f7952463824f6a03e4b1af09cdb0899f2ac1afded426f3e1),
hashed 2026-09-21T13:17:51Z before this runner existed or any number was measured.

Two legs, each a line-for-line replica of an existing validated harness with ONE extra fit per seed:
  screen  = mau_build.main's 2025 Jan/Jul leave-one-month-out (4,924 non-LIRF unmatched rows);
  forward = mav_build.main's Oct/Nov/Dec 2025, every fit on strictly earlier months.
The incumbent (max_iter 200) must reproduce reports/mau_test_predictions.parquet and reports/mav_test_predictions.parquet
EXACTLY, candidate and per-seed columns, or the run is void. mau_build.py and mav_build.py are imported, never edited.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/mxi_build.py
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
import mau_build as MB       # noqa: E402  (another lane's validated file; imported, never edited)
import arscat_build as A     # noqa: E402  (assert_day_blocks_are_real, the BC-23 guard; numpy/pandas only)

U, uc, ca, pbl = MB.U, MB.uc, MB.ca, MB.U.pbl
PREREG = ROOT / "plans/PREREG_mxi_unmatched_capacity_2026_09_21.md"
REGISTERED_SHA = "e98627ad8184f115f7952463824f6a03e4b1af09cdb0899f2ac1afded426f3e1"
HASHLOG = ROOT / "reports/mxi_prereg_hashes.log"
REPORT = ROOT / "reports/mxi_report.json"
PREDS = ROOT / "reports/mxi_predictions.parquet"
MAV_REPORT = ROOT / "reports/mav_report.json"
MAV_PREDS = ROOT / "reports/mav_test_predictions.parquet"

INCUMBENT_ITER, MXI_ITER = 200, 1000
ITERS = (INCUMBENT_ITER, MXI_ITER)
FORWARD_MONTHS = (10, 11, 12)
MONSTER_IDS = (182378789.0, 182377662.0)       # the two LFPG label rows (y = 84,240 s and 58,206 s)
ANOMALY_COLS = ["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
ALARM_S = 300


def body_params(max_iter: int) -> dict:
    return {**uc.bs.NF_PARAMS, "max_iter": max_iter}


def fit_body(x_fit, y_fit, design_extra, y_extra, seed, x_val, y_val, max_iter):
    """mau_build.py:192-199 / mav_build.py:78-85 (v24's candidate body) with max_iter the only free parameter.
    `y_fit`/`y_val` arrive already capped (MAU's `yy`); the extra target is capped here, as MAU does."""
    weight = 0.25 * len(x_fit) / len(design_extra)          # MAU's evaluation order, so 0.25 is bit-identical
    tx = pd.concat([x_fit, design_extra], ignore_index=True)
    ty = np.r_[np.asarray(y_fit, float), np.clip(np.asarray(y_extra, float), -3000, 3000)]
    sw = np.r_[np.ones(len(x_fit)), np.full(len(design_extra), weight)]
    model = MB.HGR(random_state=seed, early_stopping=True, **body_params(max_iter))
    model.fit(tx, ty, sample_weight=sw, X_val=x_val, y_val=y_val, sample_weight_val=np.ones(len(x_val)))
    return model


def compose(p_hat, sp, nf_cells, body, anomaly) -> np.ndarray:
    """body -> shipped mixture -> storm anchor (mau_build.py:216/223, mav_build.py:102-103)."""
    mix = uc.bs.mixture(p_hat, sp, uc.bs.nf_hybrid(nf_cells, body))
    return ca.anchored_values(np.rint(mix), anomaly)


def delta_stats(y, incumbent, arm, day, month) -> dict:
    """(SSE_incumbent - SSE_arm)/344,841 with the day-blocked CI; refuses a degenerate day vector (BC-23)."""
    A.assert_day_blocks_are_real(day, month)
    frame = pd.DataFrame({"y": np.asarray(y, float), "control": np.asarray(incumbent, float),
                          "day": np.asarray(day), "month": np.asarray(month)})
    return U.score(frame, np.asarray(arm, float))


def without_monsters(ids, y, incumbent, arm) -> float:
    keep = ~np.isin(np.asarray(ids, float), MONSTER_IDS)
    y, incumbent, arm = (np.asarray(v, float)[keep] for v in (y, incumbent, arm))
    return float((((y - incumbent) ** 2) - ((y - arm) ** 2)).sum() / 344841)


def verdict(screen, forward, lfpg_excluded) -> str:
    if screen["ci95"][1] < 0:
        return "WORSE"
    ok = (U.decide(screen) == "DEVELOPMENT IMPROVEMENT" and U.decide(forward) == "DEVELOPMENT IMPROVEMENT"
          and lfpg_excluded["gain"] > 0)
    return "TRUE" if ok else "NOT WORKING"


def assert_reproduces(got, want, what) -> None:
    got, want = np.asarray(got, float), np.asarray(want, float)
    if got.shape != want.shape or not np.array_equal(got, want):
        raise AssertionError(f"VOID: incumbent does not reproduce {what}")


def preflight(report, preds, prereg, hashlog, expected_sha) -> None:
    if Path(report).exists() or Path(preds).exists():
        raise FileExistsError("MXI outputs exist; no silent rerun")
    current = U.sha(prereg)
    if current != expected_sha:
        raise AssertionError(f"registration changed since it was hashed: {current}")
    if expected_sha not in Path(hashlog).read_text():
        raise AssertionError("registration sha is not in the hash log")


def _anomaly(months) -> pd.Series:
    parts = []
    for m in months:
        files = list((ROOT / "data/raw").glob(f"training_2025-{m:02d}-01_*.parquet"))
        assert len(files) == 1, (m, files)
        parts.append(ca.day_anomaly(pd.read_parquet(files[0], columns=ANOMALY_COLS)))
    return pd.concat(parts).set_index(["ADEP_mvt", "date"]).A


def _attach_anomaly(frame, months):
    anomaly = _anomaly(months)
    frame["date"] = frame.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    frame["A"] = anomaly.reindex(pd.MultiIndex.from_arrays([frame.ADEP_mvt, frame.date])).to_numpy()


def _fit_both(train, te, ex, seed, fits, leg, month):
    """One seed: production's control fit (for the design), then the v24 body at 200 and at 1000."""
    ctrl = uc.fit_congestion_regressor(train, seed)
    x = ctrl.design(train)
    yy = np.clip(train.y.to_numpy(float), -3000, 3000)
    fi, va = MB.stopping_split(len(train), seed)
    dex, dte = ctrl.design(ex), ctrl.design(te)
    out = {}
    for it in ITERS:
        t = time.monotonic()
        model = fit_body(x.iloc[fi], yy[fi], dex, ex.y.to_numpy(float), seed, x.iloc[va], yy[va], it)
        out[it] = model.predict(dte)
        fits.append({"leg": leg, "month": int(month), "seed": int(seed), "max_iter": it, "n_iter": int(model.n_iter_),
                     "seconds": round(time.monotonic() - t, 2), "fit_n": int(len(fi)), "extra_n": int(len(ex))})
    return out


def _stack(te, rows_keep, p_hat, sp, cells, preds):
    out = te[["MVT_ID_mvt", "ADEP_mvt", "date", "month", "y", "sp", "A"]].copy()
    out["day"] = te.MVT_TIME_UTC_mvt.dt.dayofyear.to_numpy()
    for it in ITERS:
        stack = np.asarray(preds[it])
        out[f"body_{it}"] = stack.mean(axis=0)
        out[f"it{it}"] = compose(p_hat, sp, cells, stack.mean(axis=0), te.A)
        for j, seed in enumerate(U.SEEDS):
            out[f"it{it}_seed{seed}"] = compose(p_hat, sp, cells, stack[j], te.A)
    return out.loc[rows_keep].copy()


def screen_leg(extra, fits):
    """mau_build.main:135-230 with the second fit."""
    frame = pd.read_parquet(ROOT / "data/cache_rome/unm.parquet")
    saved = pd.read_parquet(ROOT / "data/cache_stand/unm_physics_lomo.parquet")
    ref = saved.set_index("MVT_ID_mvt").reindex(frame.MVT_ID_mvt).reset_index()
    for c in ("y", "sp", "month"):
        np.testing.assert_array_equal(frame[c].to_numpy(), ref[c].to_numpy())
    for c in uc.WITNESS:
        frame[c] = ref[c].to_numpy()
    _attach_anomaly(frame, U.MONTHS)
    records = []
    for m in U.MONTHS:
        test = frame.month.eq(m).to_numpy()
        train = frame.loc[~test & ~uc.bs.schedule_fill(frame)]
        ex = extra.loc[extra.month.ne(m)]
        te, rr = frame.loc[test], ref.loc[test]
        preds = {it: [] for it in ITERS}
        for seed in U.SEEDS:
            for it, p in _fit_both(train, te, ex, seed, fits, "screen", m).items():
                preds[it].append(p)
        records.append(_stack(te, te.ADEP_mvt.ne("LIRF").to_numpy(), rr.p_hat, rr.sp, rr.nf_cells, preds))
        print(json.dumps({"leg": "screen", "month": m}), flush=True)
    return pd.concat(records, ignore_index=True)


def forward_leg(extra, fits):
    """mav_build.main:25-109 with the second fit."""
    frame = uc.bs.derive(pd.read_parquet(ROOT / "data/cache_rome/unm.parquet"))
    saved = pd.read_parquet(ROOT / "data/cache_stand/unm_physics_lomo.parquet")
    ref = saved.set_index("MVT_ID_mvt").reindex(frame.MVT_ID_mvt)
    np.testing.assert_array_equal(frame.y.to_numpy(), ref.y.to_numpy())
    for c in uc.WITNESS:
        frame[c] = ref[c].to_numpy()
    _attach_anomaly(frame, FORWARD_MONTHS)
    records = []
    for m in FORWARD_MONTHS:
        tr = frame.loc[frame.month.lt(m)]
        te = frame.loc[frame.month.eq(m) & frame.ADEP_mvt.ne("LIRF")]
        ex = extra.loc[extra.month.lt(m)]
        train = tr.loc[~uc.bs.schedule_fill(tr)]
        assert set(train.month) == set(ex.month) == set(range(1, m))
        parts = {}
        uc.bs.fit_unmatched(tr, te, train_matched=None, hybrid=False, parts=parts)
        preds = {it: [] for it in ITERS}
        for seed in U.SEEDS:
            for it, p in _fit_both(train, te, ex, seed, fits, "forward", m).items():
                preds[it].append(p)
        records.append(_stack(te, np.ones(len(te), bool), parts["p"], te.sp, parts["nf_cells"], preds))
        print(json.dumps({"leg": "forward", "month": m}), flush=True)
    return pd.concat(records, ignore_index=True)


def _gate(result, reference, what):
    assert_reproduces(result.MVT_ID_mvt, reference.MVT_ID_mvt, f"{what} row ids")
    assert_reproduces(result[f"it{INCUMBENT_ITER}"], reference.candidate, f"{what} candidate")
    for s in U.SEEDS:
        assert_reproduces(result[f"it{INCUMBENT_ITER}_seed{s}"], reference[f"candidate_seed{s}"], f"{what} seed {s}")


def _stats(frame):
    y, day, month = frame.y.to_numpy(float), frame.day.to_numpy(), frame.month.to_numpy()
    inc, arm = frame[f"it{INCUMBENT_ITER}"].to_numpy(float), frame[f"it{MXI_ITER}"].to_numpy(float)
    return {"delta": delta_stats(y, inc, arm, day, month), "oracle": delta_stats(y, inc, y, day, month),
            "identity": delta_stats(y, inc, inc.copy(), day, month),
            "delta_without_monsters": without_monsters(frame.MVT_ID_mvt, y, inc, arm),
            "seed_deltas": {str(s): pbl.board_gain(y, frame[f"it{INCUMBENT_ITER}_seed{s}"], frame[f"it{MXI_ITER}_seed{s}"])
                            for s in U.SEEDS},
            "airport_deltas": {str(k): float(v) for k, v in
                               pd.Series(((y - inc) ** 2 - (y - arm) ** 2) / 344841).groupby(frame.ADEP_mvt.to_numpy()).sum().items()},
            "rows_changed": int((inc != arm).sum()), "n": int(len(frame))}


def assemble(screen, forward, fits):
    """Everything after the fits: both legs' stats, the LFPG-excluded leg, the verdict and the prediction frame.
    Pure function of the leg frames, so it is tested before any real run (bug class BC-24)."""
    s, f = _stats(screen), _stats(forward)
    for leg in (s, f):
        assert leg["identity"]["gain"] == 0.0 and leg["identity"]["ci95"] == [0.0, 0.0], "VOID: identity"
    lf = screen.ADEP_mvt.ne("LFPG").to_numpy()
    lfpg_excluded = delta_stats(screen.y[lf], screen[f"it{INCUMBENT_ITER}"][lf], screen[f"it{MXI_ITER}"][lf],
                                screen.day[lf], screen.month[lf])
    mxi_fits = [x for x in fits if x["max_iter"] == MXI_ITER]
    report = {"tag": "MXI", "verdict": verdict(s["delta"], f["delta"], lfpg_excluded), "screen": s, "forward": f,
              "screen_lfpg_excluded": lfpg_excluded,
              "mxi_fits_at_cap": sum(x["n_iter"] >= MXI_ITER for x in mxi_fits),
              "incumbent_fits_at_cap": sum(x["n_iter"] >= INCUMBENT_ITER for x in fits if x["max_iter"] == INCUMBENT_ITER),
              "mxi_n_iter": sorted(x["n_iter"] for x in mxi_fits), "fits": fits}
    preds = pd.concat([screen.assign(leg="screen"), forward.assign(leg="forward")], ignore_index=True)
    return report, preds


def main() -> dict:
    started = time.monotonic()
    signal.alarm(ALARM_S)
    preflight(REPORT, PREDS, PREREG, HASHLOG, REGISTERED_SHA)
    mau, mav = json.loads(MB.REPORT.read_text()), json.loads(MAV_REPORT.read_text())
    for name, h in {**mau["input_hashes"], **mav["input_hashes"]}.items():
        assert U.sha(ROOT / name) == h, ("input changed since MAU/MAV", name)
    assert U.sha(MB.TRAIN) == mau["train_sha256"]
    extra = pd.read_parquet(MB.TRAIN)
    fits = []
    screen = screen_leg(extra, fits)
    _gate(screen, pd.read_parquet(MB.PREDS), "MAU screen")
    forward = forward_leg(extra, fits)
    _gate(forward, pd.read_parquet(MAV_PREDS), "MAV forward")
    report, preds = assemble(screen, forward, fits)
    report.update(prereg_sha256=REGISTERED_SHA, runner_sha256=U.sha(__file__), mau_runner_sha256=U.sha(MB.__file__),
                  seconds=round(time.monotonic() - started, 1),
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3)
    preds.to_parquet(PREDS, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(PREDS), preds)
    report["predictions_sha256"] = U.sha(PREDS)
    with REPORT.open("x") as fh:
        json.dump(report, fh, indent=1, allow_nan=False)
    signal.alarm(0)
    print(json.dumps({k: report[k] for k in ("verdict", "mxi_fits_at_cap", "seconds")}), flush=True)
    return report


if __name__ == "__main__":
    main()
