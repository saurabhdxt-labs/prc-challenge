"""Arm MTX driver (plans/PREREG_mtx_monthly_calibration_2026_09_28.md).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/mtx_run.py validate   # 2025 Jan+Jul OOF, LMO mapping
    /opt/homebrew/bin/python3.11 -B scripts/analysis/mtx_run.py build      # v42 = v41 + registered calibration

Writes reports/mtx_validation_2026_09_28.json and (build) submissions/merry-quicksand_v42.parquet + .meta.json.
Uploads NOTHING.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import mtx_calib as mc  # noqa: E402
import mtx_source as ms  # noqa: E402

BOARD_N = 344_841
RAW = ROOT / "data" / "mtx" / "raw"
PREREG = "plans/PREREG_mtx_monthly_calibration_2026_09_28.md"
V41 = ROOT / "submissions" / "merry-quicksand_v41.parquet"
V42 = ROOT / "submissions" / "merry-quicksand_v42.parquet"


def _load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def label_cells(dep: pd.DataFrame, pub: pd.DataFrame) -> pd.DataFrame:
    """Our 2025 label mean per airport × LOCAL month joined to the published cells: ap, month, n, mean_label, TF,
    NR, INV, P, pub_mean_s. `dep` needs ADEP_mvt, MVT_TIME_UTC_mvt, TAXITIME_SEC_mvt. Refuses unjoined cells."""
    y, m = ms.local_year_month(dep.MVT_TIME_UTC_mvt.reset_index(drop=True), dep.ADEP_mvt.reset_index(drop=True))
    d = pd.DataFrame({"ap": dep.ADEP_mvt.to_numpy(), "year": y.to_numpy(), "month": m.to_numpy(),
                      "y": dep.TAXITIME_SEC_mvt.to_numpy(float)})
    d = d[d.year == 2025]
    g = d.groupby(["ap", "month"]).y.agg(n="size", mean_label="mean").reset_index()
    out = g.merge(pub[pub.year == 2025], on=["ap", "month"], how="left", validate="one_to_one")
    if out.TF.isna().any():
        raise ValueError(f"unjoined label cells: {out[out.TF.isna()][['ap', 'month']].values.tolist()}")
    return out


def lmo_params(cells_ap: pd.DataFrame, month: int):
    """For scoring `month`: (c, e) fitted without it, and the nested LMO errors of the other months (each fitted
    without `month` AND itself) for sigma — no label of `month` reaches T or sigma."""
    rest = cells_ap[cells_ap.month != month]
    c, e = mc.fit_excess(rest)
    return c, e, mc.lmo_targets(rest)


def _ap_month_key(ap, month):
    return np.char.add(np.char.add(np.asarray(ap, str), "|"), np.asarray(month).astype(str))


def calibrate(frame: pd.DataFrame, pub_cells: pd.DataFrame, params: dict, runway: pd.DataFrame | None,
              mode: str, full_lambda: bool = False):
    """Apply the registered rule to `frame` (ap, month, pred[, rwy]). pub_cells: published rows for the scored
    months; params[(ap, month)] = (c, e, lmo_df). Returns (new pred, per-cell report)."""
    target, lam, sig = {}, {}, {}
    key = _ap_month_key(frame.ap, frame.month)
    for (ap, mo), (c, e, lmo) in params.items():
        p = pub_cells[(pub_cells.ap == ap) & (pub_cells.month == mo)].iloc[0]
        T = float(mc.target_mean(p.pub_mean_s, p.NR, p.INV, p.TF, c, e))
        k = f"{ap}|{mo}"
        sel = key == k
        if not sel.any():
            continue
        s = mc.cell_sigma(lmo, float(p.INV / p.TF))
        g = T - float(frame.pred.to_numpy()[sel].mean())
        target[k], sig[k] = T, s
        lam[k] = 1.0 if full_lambda else mc.shrink(g, s)
    out, rep = mc.shift(frame.pred.to_numpy(), key, target, lam, mode=mode)
    for k in rep:
        rep[k]["T"], rep[k]["sigma"] = target[k], sig[k]
    if runway is None:
        return out, rep
    # R: per-runway targets where published; other rows keep the airport shift just computed
    rk = np.char.add(np.char.add(key, "|"), frame.rwy.astype(str).to_numpy())
    rt, rl = {}, {}
    for (ap, mo), (c, e, lmo) in params.items():
        p = pub_cells[(pub_cells.ap == ap) & (pub_cells.month == mo)].iloc[0]
        exc = float(mc.target_mean(0.0, p.NR, p.INV, p.TF, c, e))
        s = mc.cell_sigma(lmo, float(p.INV / p.TF))
        for _, r in runway[(runway.ap == ap) & (runway.month == mo)].iterrows():
            k = f"{ap}|{mo}|{r.runway}"
            sel = rk == k
            if not sel.any():
                continue
            T = float(r.pub_mean_s) + exc
            g = T - float(frame.pred.to_numpy()[sel].mean())
            rt[k] = T
            rl[k] = 1.0 if full_lambda else mc.shrink(g, s)
    has_r = np.isin(rk, list(rt))
    out_r, rep_r = mc.shift(frame.pred.to_numpy(), rk, rt, rl, mode=mode)
    out = np.where(has_r, out_r, out)
    rep.update({k: v for k, v in rep_r.items()})
    return out, rep


def _pub(year):
    return ms.load_monthly_csv(RAW / f"taxi_out_additional_time_{year}.csv")


def _runway(year):
    rw = pd.concat([ms.parse_dashboard_runway((RAW / "dashboard" / f"{a}.html").read_text(), a) for a in ms.AIRPORTS])
    return rw[rw.year == year]


def _train_dep():
    cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt"]
    t = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))])
    return t[t.PHASE_mvt == "DEP"].reset_index(drop=True)


def validate(n_boot=2000):
    dep = _train_dep()
    pub25 = _pub(2025)
    cells = label_cells(dep, pub25)
    cln = _load_mod("cln_build", "scripts/analysis/cln_build.py")
    M = cln.load_comparators()["M"][["MVT_ID_mvt", "y", "pipe"]].rename(columns={"pipe": "pred"})
    U = pd.read_parquet(ROOT / "data/cache_stand/u_fold_current_v33.parquet")
    U = U[~U.cln_spent_day][["MVT_ID_mvt", "y", "after_v33"]].rename(columns={"after_v33": "pred"})
    fr = pd.concat([M.assign(lane="M"), U.assign(lane="U")], ignore_index=True)
    meta = dep[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt"]]
    fr = fr.merge(meta, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if fr.ADEP_mvt.isna().any() or not np.array_equal(fr.y.to_numpy(float), fr.TAXITIME_SEC_mvt.to_numpy(float)):
        raise ValueError("OOF rows do not join the raw 2025 labels exactly")
    t = pd.to_datetime(fr.MVT_TIME_UTC_mvt, utc=True)
    fr["ap"], fr["month"], fr["day"], fr["rwy"] = fr.ADEP_mvt, t.dt.month.to_numpy(), t.dt.strftime("%Y-%m-%d"), fr.RUNWAY_mvt
    if set(fr.month) != {1, 7}:
        raise ValueError(f"OOF months {sorted(set(fr.month))} != Jan+Jul")
    params = {}
    for ap in ms.AIRPORTS:
        for mo in (1, 7):
            params[(ap, mo)] = lmo_params(cells[cells.ap == ap], mo)
    pubm = pub25[pub25.month.isin([1, 7])]
    rw = _runway(2025)
    y, before = fr.y.to_numpy(float), fr.pred.to_numpy(float)
    res = {"n_rows": int(len(fr)), "denominator": BOARD_N, "variants": {}}
    outs = {}
    for name, kw in {"V1": dict(runway=None, mode="add"), "V1F": dict(runway=None, mode="add", full_lambda=True),
                     "V2": dict(runway=None, mode="mul"), "R": dict(runway=rw, mode="add")}.items():
        after, rep = calibrate(fr, pubm, params, **kw)
        outs[name] = after
        G, lo, hi = mc.gain_ci(y, before, after, fr.day, fr.month, BOARD_N, n_boot)
        cellsG = {}
        for k in [k for k in rep if k.count("|") == 1]:
            ap, mo = k.split("|")
            s = ((fr.ap == ap) & (fr.month == int(mo))).to_numpy()
            g1 = mc.gain_ci(y[s], before[s], after[s], fr.day[s], fr.month[s], BOARD_N, n_boot)
            cellsG[k] = {**{kk: round(v, 3) for kk, v in rep[k].items()}, "G": g1[0], "lo": g1[1], "hi": g1[2]}
        res["variants"][name] = {"G": G, "lo": lo, "hi": hi, "cells": cellsG}
    for name in ("V1F", "V2", "R"):
        d = mc.gain_ci(y, outs["V1"], outs[name], fr.day, fr.month, BOARD_N, n_boot)
        res["variants"][name]["diff_vs_V1"] = d
    # oracle: exact all-rows mean of the scored rows, lambda = 1
    true_T = {f"{a}|{m}": v for (a, m), v in fr.groupby(["ap", "month"]).y.mean().items()}
    key = _ap_month_key(fr.ap, fr.month)
    oracle, _ = mc.shift(before, key, true_T, {k: 1.0 for k in true_T}, mode="add")
    res["oracle"] = dict(zip(("G", "lo", "hi"), mc.gain_ci(y, before, oracle, fr.day, fr.month, BOARD_N, n_boot)))
    lanes = fr.assign(res=fr.y - fr.pred).groupby(["ap", "month", "lane"]).res.agg(["size", "mean"]).reset_index()
    res["lane_bias"] = lanes.round(2).values.tolist()
    return res


def build(variant="V1"):
    dep = _train_dep()
    pub25, pub26 = _pub(2025), _pub(2026)
    cells = label_cells(dep, pub25)
    params = {}
    for ap in ms.AIRPORTS:
        ca = cells[cells.ap == ap]
        c, e = mc.fit_excess(ca)
        for mo in (1, 7):
            params[(ap, mo)] = (c, e, mc.lmo_targets(ca))
    r = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt",
                                                                     "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
    r = r[r.PHASE_mvt == "DEP"]
    v41 = pd.read_parquet(V41)
    fr = v41.merge(r, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if fr.ADEP_mvt.isna().any():
        raise ValueError("v41 ids missing from the ranking file")
    t = pd.to_datetime(fr.MVT_TIME_UTC_mvt, utc=True)
    fr["ap"], fr["month"], fr["rwy"] = fr.ADEP_mvt, t.dt.month.to_numpy(), fr.RUNWAY_mvt
    fr["pred"] = fr.TAXITIME_SEC_mvt.astype(float)
    pubm = pub26[(pub26.year == 2026) & pub26.month.isin([1, 7])]
    rw = _runway(2026) if variant == "R" else None
    after, rep = calibrate(fr, pubm, params, runway=rw, mode="mul" if variant == "V2" else "add",
                           full_lambda=variant == "V1F")
    vb = _load_mod("v35_build", "scripts/analysis/v35_build.py")
    moved = np.abs(after - fr.pred.to_numpy()) > 0
    vals = pd.DataFrame({"MVT_ID_mvt": fr.MVT_ID_mvt.to_numpy()[moved], "taxi": after[moved]})
    out, crep = vb.compose(v41, vals)
    out.to_parquet(V42, index=False)
    meta = {"base": str(V41.relative_to(ROOT)), "prereg": vb.check_prereg(PREREG), "variant": variant,
            "compose": crep, "cells": rep, "params": {f"{a}|{m}": {"c": v[0], "e": v[1]} for (a, m), v in params.items()},
            "sha256": hashlib.sha256(V42.read_bytes()).hexdigest()}
    (V42.parent / (V42.name + ".meta.json")).write_text(json.dumps(meta, indent=1, default=float))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "build"])
    ap.add_argument("--variant", default="V1", choices=["V1", "V1F", "V2", "R"])
    a = ap.parse_args()
    if a.cmd == "validate":
        res = validate()
        (ROOT / "reports" / "mtx_validation_2026_09_28.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cells"} for k, v in res["variants"].items()},
                         indent=1, default=float))
        print("oracle", res["oracle"])
    else:
        m = build(a.variant)
        print(json.dumps({"compose": m["compose"], "sha256": m["sha256"]}, indent=1))
