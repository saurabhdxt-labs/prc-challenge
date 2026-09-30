"""SHIP-CA — congestion anchor on v14's non-LIRF unmatched rows (plans/PREREG_ship_congestion_anchor_2026_09_13.md). NO upload.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/congestion_anchor_ship.py

value = rint(v14 + max(A − 300, 0)) where A = airport-day median of matched (MVT − AOBT_3) minus the airport-month median of day medians.
Post hoc fixed ratio; the board is the only independent test.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "plans" / "PREREG_ship_congestion_anchor_2026_09_13.md"
V14 = ROOT / "submissions" / "merry-quicksand_v14.parquet"
V14_SHA = "574265d5560d060fcf79372d9fae94affa4d2983237106a479fc4f8e27c61247"
PROV = ROOT / "data" / "pipeline_runs" / "v13_20260912T113635Z" / "merry-quicksand_v13.provenance.parquet"
RANK = ROOT / "data" / "raw" / "ranking.parquet"
RUNS = ROOT / "data" / "pipeline_runs"
SCOPE_STAGES = {"S1C:body": 4_897, "S1C:cells": 10}
KNOT, PROXY_MAX = 300.0, 10_800.0
EXPECT_TOUCHED, EXPECT_MAX_SHIFT = 581, 1_258.0
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def day_anomaly(rows: pd.DataFrame) -> pd.DataFrame:
    """(ADEP_mvt, date, A) from DEP rows with AOBT_3: day median of MVT − AOBT_3 in (0, 10,800) minus the airport-month median of day medians."""
    d = rows[(rows.PHASE_mvt == "DEP") & rows.AOBT_3_flt.notna()].copy()
    d["proxy"] = (d.MVT_TIME_UTC_mvt - EPOCH).dt.total_seconds() - (d.AOBT_3_flt - EPOCH).dt.total_seconds()
    d = d[(d.proxy > 0) & (d.proxy < PROXY_MAX)]
    d["date"] = d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    d["ym"] = d.date.str[:7]
    dm = d.groupby(["ADEP_mvt", "date", "ym"]).proxy.median().rename("dmed").reset_index()
    dm["A"] = dm.dmed - dm.groupby(["ADEP_mvt", "ym"]).dmed.transform("median")
    return dm[["ADEP_mvt", "date", "A"]]


def shift(A) -> np.ndarray:
    """max(A − 300, 0); missing A → 0."""
    a = np.asarray(A, dtype="float64")
    return np.where(np.isfinite(a), np.maximum(a - KNOT, 0.0), 0.0)


def anchored_values(base, A) -> np.ndarray:
    v = np.rint(np.asarray(base, dtype="float64") + shift(A))
    if not (np.isfinite(v).all() and (v >= 1).all()):
        raise AssertionError("G3: non-finite or sub-1 value")
    return v


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if sha256(V14) != V14_SHA:
        raise AssertionError("G1: v14 sha differs")
    v14 = pd.read_parquet(V14)
    prov = pd.read_parquet(PROV, columns=["MVT_ID_mvt", "stage"])
    counts = prov.stage.value_counts().to_dict()
    for k, n in SCOPE_STAGES.items():
        if counts.get(k) != n:
            raise AssertionError(f"G1: stage {k} has {counts.get(k)} rows, registered {n}")
    rank = pd.read_parquet(RANK, columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
    A = day_anomaly(rank)
    s = v14.merge(prov, on="MVT_ID_mvt", how="left", validate="one_to_one") \
           .merge(rank[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]], on="MVT_ID_mvt", how="left", validate="one_to_one")
    scope = s.stage.isin(list(SCOPE_STAGES)).to_numpy()
    if s.AOBT_3_flt[scope].notna().any() or (s.ADEP_mvt[scope] == "LIRF").any():
        raise AssertionError("G2: a scope row has AOBT_3 or is LIRF")
    s["date"] = s.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    s = s.merge(A, on=["ADEP_mvt", "date"], how="left", validate="many_to_one")
    base = v14.TAXITIME_SEC_mvt.to_numpy().astype("float64")
    new = base.copy()
    new[scope] = anchored_values(base[scope], s.A.to_numpy()[scope])
    touched = new != base
    sh = new - base
    if int(touched.sum()) != EXPECT_TOUCHED or abs(float(sh.max()) - EXPECT_MAX_SHIFT) > 1.0:
        raise AssertionError(f"G4: touched {int(touched.sum())} / max shift {sh.max():.0f}, registered {EXPECT_TOUCHED} / {EXPECT_MAX_SHIFT:.0f}")
    if (touched & ~scope).any() or (sh < 0).any():
        raise AssertionError("G3: a row outside scope changed or a shift is negative")
    cand = v14.copy()
    cand["TAXITIME_SEC_mvt"] = new.astype(v14.TAXITIME_SEC_mvt.dtype)
    if not (np.array_equal(cand.MVT_ID_mvt.to_numpy(), v14.MVT_ID_mvt.to_numpy()) and cand.dtypes.equals(v14.dtypes)):
        raise AssertionError("G3: ids / order / dtype")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RUNS / f"sca_{stamp}"
    out.mkdir(parents=False, exist_ok=False)
    sub, rows_p = out / "merry-quicksand_sca.parquet", out / "sca_touched_rows.parquet"
    cand.to_parquet(sub, index=False)
    tr = s.loc[touched, ["MVT_ID_mvt", "ADEP_mvt", "date", "stage", "A"]].assign(v14=base[touched], candidate=new[touched])
    tr.to_parquet(rows_p, index=False)
    back = pd.read_parquet(sub).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(tr.MVT_ID_mvt.to_numpy()).to_numpy("float64")
    if not np.array_equal(back, tr.candidate.to_numpy("float64")):
        raise AssertionError("G5: re-read does not reproduce touched values")
    by_day = tr.groupby(["ADEP_mvt", "date"]).agg(n=("A", "size"), shift_s=("A", lambda a: float(np.maximum(a.iloc[0] - KNOT, 0)))).reset_index()
    manifest = dict(tag="SCA", prereg=str(PREREG.relative_to(ROOT)), prereg_sha256=sha256(PREREG), created_utc=stamp,
                    guards=dict(G1=True, G2=True, G3=True, G4=True, G5=True),
                    touched=int(touched.sum()), mean_shift_s=float(sh[touched].mean()), max_shift_s=float(sh.max()),
                    stake_board_mse=dict(no_effect=-float((sh[touched] ** 2).sum() / 344_841)),
                    by_airport_day=by_day.to_dict(orient="records"),
                    outputs={str(p.relative_to(ROOT)): sha256(p) for p in (sub, rows_p)},
                    inputs={str(p.relative_to(ROOT)): sha256(p) for p in (V14, PROV, RANK)},
                    evidence_statement="post hoc fixed ratio (k=1, knot 300); fitted CAN NOT WORKING; 2025 sensitivity +1,180, worst event -1.9; "
                                       "board is the only independent test; stake -1,088 (no effect) / +1,088 (1x) / +3,264 (2x)",
                    upload="NOT AUTHORISED - separate owner decision")
    (out / "sca_manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    print(json.dumps({k: manifest[k] for k in ("touched", "mean_shift_s", "max_shift_s", "stake_board_mse", "outputs")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
