"""`scripts/catboost_native.py` -- arms C_delta / C_sched of plans/PREREG_catboost_native_2026_09_10.md.

    python scripts/catboost_native.py build [--months 1,2,3,7,8,9]   # the input parquet (heavy-ish)
    python scripts/catboost_native.py fit --target delta|G           # runs the venv worker (heavy)
    python scripts/catboost_native.py score                          # screens every arm that exists vs F

`build` joins the stand cache with the queue and order blocks exactly as lgbm_fold does (same row
order, so a holdout row's `row` equals arm F's), attaches full flight / callsign from the raw files by
MVT_ID through the order cache, and writes the 59 numeric columns (arm F's design minus the 24 in-fold
encodings -- BC-1 cannot apply), the 13 categoricals, both targets and arm F's fold masks.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

INPUT = ROOT / "data" / "cache_stand" / "catboost_native_input.parquet"
PRED = {t: ROOT / "data" / "cache_stand" / f"catboost_native_{t}.npy" for t in ("delta", "G")}
VENV_PY = pathlib.Path.home() / ".venvs" / "prc-catboost" / "bin" / "python"
WORKER = ROOT / "scripts" / "catboost_native_worker.py"
CAT_COLS = ["ap", "ap_stand", "ap_rwy", "ap_ades", "FLIGHT_mvt", "CALLSIGN_flt", "airline", "AIRCRAFT_OPERATOR_flt",
            "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "stand_pref"]
W_MATCHED = 0.9846596
SHORTLIST_MSE = 1_000.0
BLEND = 0.5


def numeric_columns() -> list:
    import lgbm_fold as lf
    return [c for c in lf.FEATS_QUEUE_ORDER if not c.startswith(("te_", "de_"))]


def _s(v: pd.Series) -> pd.Series:
    return v.astype(object).where(v.notna(), "NA").astype(str)


def categoricals(f: pd.DataFrame) -> pd.DataFrame:
    """The registered 13 categoricals as strings; airport-scoped keys carry the airport, and a null
    becomes the literal "NA" inside its scope."""
    ap = _s(f.ap)
    out = pd.DataFrame(index=f.index)
    out["ap"] = ap
    out["ap_stand"] = ap + "|" + _s(f.STAND_mvt)
    out["ap_rwy"] = ap + "|" + _s(f.RUNWAY_mvt)
    out["ap_ades"] = ap + "|" + _s(f.ADES_mvt)
    for c in ("FLIGHT_mvt", "CALLSIGN_flt", "airline", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt",
              "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "stand_pref"):
        out[c] = _s(f[c])
    return out[CAT_COLS]


def attach_identity(ids, raw: pd.DataFrame) -> pd.DataFrame:
    """FLIGHT_mvt / CALLSIGN_flt for `ids` (in order) from a raw frame keyed by MVT_ID."""
    r = raw.drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    missing = int((~pd.Index(ids).isin(r.index)).sum())
    if missing:
        raise ValueError(f"{missing} ids absent from the raw frame")
    return r.loc[np.asarray(ids), ["FLIGHT_mvt", "CALLSIGN_flt"]].reset_index(drop=True)


def targets(y, sp, delta) -> dict:
    y, sp, delta = (np.asarray(v, dtype="float64") for v in (y, sp, delta))
    return {"delta": delta, "G": sp - y}


def recover(pred, proxy, sp, target) -> np.ndarray:
    pred, proxy, sp = (np.asarray(v, dtype="float64") for v in (pred, proxy, sp))
    if target == "delta":
        return np.maximum(proxy - pred, 1.0)
    if target == "G":
        return np.maximum(sp - pred, 1.0)
    raise ValueError(f"unknown target {target!r}")


def screen(y, F, arms: dict, ap, fill, delta, n_boot=2_000) -> dict:
    """The registered screening rule for every arm against F on the same rows."""
    import lgbm_fold as lf
    y, F = np.asarray(y, dtype="float64"), np.asarray(F, dtype="float64")
    se = {"F": (F - y) ** 2, **{a: (np.asarray(v, dtype="float64") - y) ** 2 for a, v in arms.items()}}
    boot = lf.paired_bootstrap(se, [(a, "F") for a in arms], n_boot, lf.BOOT_SEED)
    n = len(y)
    out = {}
    for a in arms:
        gain_mse = float(W_MATCHED * (se["F"].sum() - se[a].sum()) / n)
        b = boot[f"{a}_vs_F"]
        cut = lambda m: float(np.sqrt(se["F"][m].mean()) - np.sqrt(se[a][m].mean())) if m.any() else None
        out[a] = {"rmse": float(np.sqrt(se[a].mean())), "rmse_F": float(np.sqrt(se["F"].mean())),
                  "gain_s": b["gain_s"], "ci95": b["ci95"], "net_weighted_fold_mse": gain_mse,
                  "shortlist": bool(b["ci95"][0] > 0 and gain_mse >= SHORTLIST_MSE),
                  "gain_fill_s": cut(np.asarray(fill, bool)), "gain_nonfill_s": cut(~np.asarray(fill, bool)),
                  "gain_by_airport_s": {str(k): cut(np.asarray(ap) == k) for k in sorted(set(ap))},
                  "gain_by_band_s": {k: cut(m) for k, m in lf.delta_bands(delta).items()}}
    return out


def build(months=None, out=INPUT, log=print) -> None:
    import lgbm_fold as lf
    import lgbm_submit as L
    lf.log = L.log = log
    frames, paths = lf._load_months(lf.CACHE, months, lf.FEATS_QUEUE_ORDER, lf.QCACHE, None, lf.OCACHE, None)
    for f, p in zip(frames, paths):
        ids = pq.read_table(L.order_cache_path(lf.OCACHE, p), columns=["MVT_ID_mvt"]).to_pandas().MVT_ID_mvt.to_numpy()
        if len(ids) != len(f):
            raise AssertionError(f"{p.name}: order cache {len(ids)} rows vs stand cache {len(f)}")
        raw = pq.read_table(ROOT / "data" / "raw" / p.name, columns=["MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt"]).to_pandas()
        idn = attach_identity(ids, raw)
        f["FLIGHT_mvt"], f["CALLSIGN_flt"] = idn.FLIGHT_mvt.to_numpy(), idn.CALLSIGN_flt.to_numpy()
        f["MVT_ID_mvt"] = ids
    d = pd.concat(frames, ignore_index=True)
    del frames
    te, tr, fit, es = lf.fold_masks(d.month.to_numpy())
    o = assemble_output(d, te, tr, fit, es)
    num = numeric_columns()
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    o.to_parquet(out)
    pathlib.Path(str(out) + ".cols.json").write_text(json.dumps({"num": num, "cat": CAT_COLS}))
    log(f"input -> {out}: {len(o):,} rows ({int(te.sum()):,} holdout, {int(fit.sum()):,} fit, {int(es.sum()):,} es), "
        f"{len(num)} numeric + {len(CAT_COLS)} categorical; peak RSS {L.peak_rss_gb():.2f} GB")


def assemble_output(d: pd.DataFrame, te, tr, fit, es) -> pd.DataFrame:
    """The worker's input frame. Feature columns stay float32 under their own names; the columns the
    SCORING step needs (y, proxy, sp in float64) carry a `score_` prefix so they can never overwrite a
    feature of the same name (`proxy` and `sp` are both)."""
    num = numeric_columns()
    o = pd.DataFrame({c: d[c].to_numpy(dtype="float32") for c in num})
    o = pd.concat([o, categoricals(d).reset_index(drop=True)], axis=1)
    t = targets(d.y, d.sp, d.delta)
    o["delta"], o["G"] = t["delta"], t["G"]
    for k in ("y", "proxy", "sp"):
        o[f"score_{k}"] = d[k].to_numpy(dtype="float64")
    o["month"], o["row"], o["MVT_ID_mvt"] = d.month.to_numpy(), np.arange(len(d)), d.MVT_ID_mvt.to_numpy()
    o["train"], o["fit"], o["es"], o["holdout"] = tr, fit, es, te
    return o


def fit(target: str, inp=INPUT, out=None) -> int:
    out = out or PRED[target]
    cmd = [str(VENV_PY), "-u", str(WORKER), "--input", str(inp), "--target", target, "--output", str(out)]
    return subprocess.call(cmd)


def score(inp=INPUT) -> dict:
    """Screens every arm whose predictions exist, against arm F's treatment on the same rows."""
    import lgbm_fold as lf
    h = pd.read_parquet(inp, columns=["row", "score_y", "score_proxy", "score_sp", "delta", "holdout"])
    h = h[h.holdout].reset_index(drop=True).rename(columns={"score_y": "y", "score_proxy": "proxy", "score_sp": "sp"})
    f = pd.read_parquet(lf.CACHE / "fold_preds_queue_order.parquet", columns=["row", "ap", "y", "proxy", "treatment"])
    m = h.merge(f, on="row", suffixes=("", "_f"))
    if len(m) != len(h) or not np.array_equal(m.y.to_numpy(), m.y_f.to_numpy()):
        raise AssertionError("holdout rows do not align with arm F's record on `row`")
    F = np.maximum(m.proxy - m.treatment, 1.0).to_numpy()
    arms = {}
    for t, p in PRED.items():
        if p.exists():
            pred = np.load(p)
            if len(pred) != len(m):
                raise AssertionError(f"{p.name}: {len(pred)} predictions for {len(m)} holdout rows")
            arms[f"C_{t}"] = recover(pred, m.proxy, m.sp, t)
            arms[f"C_{t}_blend"] = BLEND * arms[f"C_{t}"] + (1 - BLEND) * F
    fill = (np.abs(m.y - m.sp) <= 60).to_numpy()
    return screen(m.y.to_numpy(), F, arms, m.ap.to_numpy(), fill, m.delta.to_numpy())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("build", "fit", "score"))
    ap.add_argument("--months", default=None)
    ap.add_argument("--target", choices=("delta", "G"), default="delta")
    ap.add_argument("--input", default=str(INPUT))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.step == "build":
        months = tuple(int(x) for x in a.months.split(",")) if a.months else None
        build(months, a.input)
        return 0
    if a.step == "fit":
        return fit(a.target, a.input, a.out)
    res = score(a.input)
    for k, v in res.items():
        print(f"{k:14s} RMSE {v['rmse']:8.3f} (F {v['rmse_F']:.3f})  gain {v['gain_s']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  "
              f"net {v['net_weighted_fold_mse']:+8.0f} MSE  fill {v['gain_fill_s']:+.2f}  non-fill {v['gain_nonfill_s']:+.2f}  "
              f"-> {'SHORTLIST' if v['shortlist'] else 'not shortlisted'}")
    (ROOT / "reports" / "catboost_native.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
