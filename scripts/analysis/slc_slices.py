"""SLC — fixed slices of the January + July 2025 holdout, to compare our out-of-month predictions with a public
reference solution's published per-slice RMSE.

Registration: plans/PREREG_slc_slice_comparison_2026_10_04.md (REG 3ab3460f). The slice definitions are the
reference's, re-implemented from their description: no flight-plan record (FLIGHT_ID_mvt null), tail (y > 3,600 s),
normal (0 < y < 10,800 s), near (y within 300 s of MVT − a planned time), lottery (y > 10,800 s and not near).

    python scripts/analysis/slc_slices.py --out reports/slc_report.json

Reads our stored OOF records and the raw January / July 2025 files; no fit, no board read, no upload.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]

PLAN_REFS = ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]
TAIL_S = 3600.0
MAX_NORMAL_S = 10800.0
NEAR_S = 300.0
Y_TOL_S = 0.5
REF_NORMAIS_NM = 189.04      # the reference's four members, mean (registered)
BEHIND_S = 2.0               # TRUE: ours − reference >= this
NOT_BEHIND_S = 0.5           # FALSE: ours − reference <= this
SLICES = ("completo", "sem_outliers", "nm_presente", "nm_ausente", "y_gt_1h", "normais_nm", "cauda_copia", "sem_loteria")
RAW_COLS = ["MVT_ID_mvt", "PHASE_mvt", "TAXITIME_SEC_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt"] + PLAN_REFS


def masks(frame: pd.DataFrame) -> dict:
    """Boolean arrays nm / tail / normal / near / lottery for rows with y, FLIGHT_ID_mvt, MVT time and the planned times."""
    y = frame.y.to_numpy(dtype="float64")
    mvt = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
    gaps = np.column_stack([(mvt - pd.to_datetime(frame[c], utc=True)).dt.total_seconds()
                            .to_numpy(dtype="float64", na_value=np.nan) for c in PLAN_REFS])
    with np.errstate(invalid="ignore"):
        near = (np.abs(y[:, None] - gaps) <= NEAR_S).any(axis=1)        # a null planned time is never near
    return {"nm": frame.FLIGHT_ID_mvt.isna().to_numpy(), "tail": y > TAIL_S, "normal": (y > 0) & (y < MAX_NORMAL_S),
            "near": near, "lottery": (y > MAX_NORMAL_S) & ~near}


def _rmse(err2: np.ndarray) -> float:
    return float(np.sqrt(err2.mean())) if err2.size else float("nan")


def metrics(frame: pd.DataFrame) -> dict:
    """RMSE on each registered slice, the false alarms and the per-airport RMSE."""
    m = masks(frame)
    y, pred = frame.y.to_numpy(dtype="float64"), frame.pred.to_numpy(dtype="float64")
    err2 = (y - pred) ** 2
    sel = {"completo": np.ones(len(y), bool), "sem_outliers": m["normal"], "nm_presente": ~m["nm"], "nm_ausente": m["nm"],
           "y_gt_1h": m["tail"], "normais_nm": ~m["tail"] & ~m["nm"], "cauda_copia": m["tail"] & m["near"],
           "sem_loteria": ~m["lottery"]}
    out = {k: _rmse(err2[v]) for k, v in sel.items()}
    false_alarm = ~m["tail"] & (pred > TAIL_S)
    out["n"] = int(len(y))
    out["n_by_slice"] = {**{k: int(v.sum()) for k, v in sel.items()}, "lottery": int(m["lottery"].sum())}
    out["sse_by_slice"] = {k: float(err2[v].sum()) for k, v in sel.items()}
    out["alarmes_falsos"] = {"n": int(false_alarm.sum()),
                             "parte_erro2": float(err2[false_alarm].sum() / err2.sum()) if err2.sum() else 0.0}
    ap = frame.ap.to_numpy().astype(str)
    out["por_aeroporto"] = {a: _rmse(err2[ap == a]) for a in sorted(set(ap))}
    out["normais_nm_por_aeroporto"] = {a: _rmse(err2[(ap == a) & sel["normais_nm"]]) for a in sorted(set(ap))}
    return out


def attach(oof: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """The OOF rows with their raw FLIGHT_ID / MVT time / planned times, joined by MVT_ID_mvt (order kept); refuses a
    row without a raw row or a label that disagrees with the raw one."""
    r = raw.drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    absent = ~oof.MVT_ID_mvt.isin(r.index)
    if absent.any():
        raise ValueError(f"{int(absent.sum())} OOF row(s) have no raw row")
    r = r.reindex(oof.MVT_ID_mvt.to_numpy())
    bad = np.abs(r.TAXITIME_SEC_mvt.to_numpy(dtype="float64") - oof.y.to_numpy(dtype="float64")) > Y_TOL_S
    if bad.any():
        raise ValueError(f"labels disagree on {int(bad.sum())} row(s) between the OOF records and the raw files")
    out = oof.reset_index(drop=True).copy()
    for c in ["FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt"] + PLAN_REFS:
        out[c] = r[c].to_numpy()
    return out


def verdict_of(ours_normais_nm: float) -> str:
    """H-SLC on the registered slice: TRUE = behind by >= 2.0 s; FALSE = within +0.5 s (AHEAD if <= −2.0 s)."""
    d = round(ours_normais_nm - REF_NORMAIS_NM, 2)
    if d >= BEHIND_S:
        return "TRUE"
    if d <= -BEHIND_S:
        return "FALSE (AHEAD)"
    return "FALSE" if d <= NOT_BEHIND_S else "NOT ESTABLISHED"


def load_raw(files) -> pd.DataFrame:
    f = pd.concat([pd.read_parquet(p, columns=RAW_COLS) for p in files], ignore_index=True)
    return f[f.PHASE_mvt == "DEP"].drop(columns=["PHASE_mvt"])


def census(raw: pd.DataFrame, oof: pd.DataFrame) -> dict:
    """The departures the reference scores and we do not (no valid label): count, labels, flight-plan record."""
    dep = raw[raw.MVT_TIME_UTC_mvt.notna()]
    extra = dep[~dep.MVT_ID_mvt.isin(oof.MVT_ID_mvt)]
    y = extra.TAXITIME_SEC_mvt.to_numpy(dtype="float64")
    return {"reference_rows": int(len(dep)), "ours": int(len(oof)), "extra": int(len(extra)),
            "extra_y_null": int(np.isnan(y).sum()), "extra_y_le_0": int((y <= 0).sum()),
            "extra_with_plan_record": int(extra.FLIGHT_ID_mvt.notna().sum()),
            "extra_y_min": float(np.nanmin(y)) if np.isfinite(y).any() else None,
            "extra_y_max": float(np.nanmax(y)) if np.isfinite(y).any() else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--primary", nargs="+", default=[str(ROOT / "data/fm1/sel/wsc_S0/oof_F1.parquet"),
                                                     str(ROOT / "data/fm1/sel/wsc_S0/oof_F2.parquet")])
    ap.add_argument("--second", nargs="+", default=[str(ROOT / "data/fm1/lomo_1004/lomo_screen/oof_M01.parquet"),
                                                    str(ROOT / "data/fm1/lomo_1004/lomo_screen/oof_M07.parquet")])
    ap.add_argument("--raw", nargs="+", default=[str(ROOT / "data/raw/training_2025-01-01_2025-02-01.parquet"),
                                                 str(ROOT / "data/raw/training_2025-07-01_2025-08-01.parquet")])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    raw = load_raw([pathlib.Path(p).resolve() for p in a.raw])
    rep = {"registration": "plans/PREREG_slc_slice_comparison_2026_10_04.md", "reference_normais_nm": REF_NORMAIS_NM}
    for name, files in (("primary", a.primary), ("second", a.second)):
        oof = pd.concat([pd.read_parquet(pathlib.Path(p).resolve(), columns=["MVT_ID_mvt", "y", "pred", "ap", "month", "lane"])
                         for p in files], ignore_index=True)
        rep[name] = {"files": [str(p) for p in files], "metrics": metrics(attach(oof, raw)), "census": census(raw, oof)}
    ours = rep["primary"]["metrics"]["normais_nm"]
    rep["delta_normais_nm"] = round(ours - REF_NORMAIS_NM, 2)
    rep["verdict"] = verdict_of(ours)
    pathlib.Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"verdict": rep["verdict"], "delta_normais_nm": rep["delta_normais_nm"],
                      "primary": {k: round(rep["primary"]["metrics"][k], 2) for k in SLICES},
                      "second": {k: round(rep["second"]["metrics"][k], 2) for k in SLICES}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
