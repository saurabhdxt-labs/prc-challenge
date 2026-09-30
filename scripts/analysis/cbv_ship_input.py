"""SHIP-CBV input — all-2025 training + 2026 scored matched rows for the unchanged CatBoost worker
(plans/PREREG_ship_cbv_2026_09_13.md §2, guards S2/S3).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/cbv_ship_input.py

Reuses data/cache_stand/catboost_native_input.parquet (the screen's own 2025 frame) unchanged except the masks, appends the 2026
rows built with the screen's own column functions, and writes data/cache_stand/cbv_ship_input.parquet (+ .cols.json).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import catboost_native as CN  # noqa: E402

SRC = ROOT / "data" / "cache_stand" / "catboost_native_input.parquet"
OUT = ROOT / "data" / "cache_stand" / "cbv_ship_input.parquet"
RANK_STAND = ROOT / "data" / "cache_stand" / "ranking.parquet"
RANK_QUEUE = ROOT / "data" / "cache_queue" / "ranking.parquet"
RANK_ORDER = ROOT / "data" / "cache_order" / "ranking.parquet"
RAW_RANK = ROOT / "data" / "raw" / "ranking.parquet"
ES_MONTHS = (3, 9)
N_2025, N_2026 = 2_062_440, 339_551


def ship_masks(month) -> dict:
    """All 2025 rows train; early stop on months 3 and 9; fit on the rest; no 2025 holdout."""
    month = np.asarray(month)
    es = np.isin(month, ES_MONTHS)
    return {"train": np.ones(len(month), dtype=bool), "fit": ~es, "es": es, "holdout": np.zeros(len(month), dtype=bool)}


def assemble_2026(rank: pd.DataFrame, num, row_offset: int) -> pd.DataFrame:
    """The 2026 holdout frame in the worker's contract: numeric float32, categoricals via the screen's function, targets NaN."""
    missing = [c for c in num if c not in rank.columns]
    if missing:
        raise ValueError(f"2026 frame lacks numeric columns {missing}")
    o = pd.DataFrame({c: rank[c].to_numpy(dtype="float32") for c in num})
    o = pd.concat([o, CN.categoricals(rank.reset_index(drop=True)).reset_index(drop=True)], axis=1)
    n = len(rank)
    o["delta"], o["G"] = np.full(n, np.nan), np.full(n, np.nan)
    o["score_y"] = np.full(n, np.nan)
    o["score_proxy"] = rank.proxy.to_numpy(dtype="float64")
    o["score_sp"] = rank.sp.to_numpy(dtype="float64")
    o["month"] = rank.month.to_numpy()
    o["row"] = row_offset + np.arange(n)
    o["MVT_ID_mvt"] = rank.MVT_ID_mvt.to_numpy()
    o["train"], o["fit"], o["es"], o["holdout"] = (np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool), np.ones(n, bool))
    return o


def check_contract(d: pd.DataFrame, num, cat) -> None:
    """The worker's input contract, asserted before the worker ever reads the file."""
    for c in num:
        if d[c].dtype != np.float32:
            raise ValueError(f"numeric {c} is {d[c].dtype}")
    for c in cat:
        if not d[c].map(lambda v: isinstance(v, str)).all():
            raise ValueError(f"categorical {c} has a non-string")
    tr, fit, es, ho = (d[m].to_numpy() for m in ("train", "fit", "es", "holdout"))
    if (tr & ho).any() or (fit & es).any() or not np.array_equal(fit | es, tr):
        raise ValueError("mask contract violated")
    if np.isfinite(d.delta.to_numpy()[ho]).any() or np.isfinite(d.score_y.to_numpy()[ho]).any():
        raise ValueError("a 2026 holdout row carries a label")
    if not np.isfinite(d.delta.to_numpy()[tr]).all():
        raise ValueError("non-finite target on a training row")


def main() -> int:
    import lgbm_submit as L
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    t0 = time.time()
    cols = json.loads(Path(str(SRC) + ".cols.json").read_text())
    num, cat = cols["num"], cols["cat"]
    d25 = pd.read_parquet(SRC)
    if len(d25) != N_2025:
        raise AssertionError(f"2025 input has {len(d25)} rows, registered {N_2025}")
    for k, v in ship_masks(d25.month.to_numpy()).items():
        d25[k] = v
    rank = pd.read_parquet(RANK_STAND)
    if not (rank.y.isna().all() and rank.delta.isna().all()):
        raise AssertionError("S3: the ranking stand cache carries labels")
    ids = rank.MVT_ID_mvt.to_numpy(dtype="float64")
    q = L.attach_queue_by_id(ids, L.read_queue_cache(RANK_QUEUE), subset_ok=False)
    o = L.attach_order_by_id(ids, L.read_order_cache(RANK_ORDER), subset_ok=False)
    for c in L.QUEUE_FEATS:
        rank[c] = q[c].to_numpy()
    for c in L.ORDER_FEATS:
        rank[c] = o[c].to_numpy()
    raw = pd.read_parquet(RAW_RANK, columns=["MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "PHASE_mvt"])
    raw = raw[raw.PHASE_mvt == "DEP"]
    idn = CN.attach_identity(ids, raw)
    rank["FLIGHT_mvt"], rank["CALLSIGN_flt"] = idn.FLIGHT_mvt.to_numpy(), idn.CALLSIGN_flt.to_numpy()
    r = raw.drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").loc[ids]
    px = (pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True) - pd.to_datetime(r.AOBT_3_flt, utc=True)).dt.total_seconds().to_numpy()
    if not np.allclose(px, rank.proxy.to_numpy(dtype="float64"), rtol=0, atol=0.5):
        raise AssertionError("S3: 2026 proxy is not MVT − AOBT_3 from the raw ranking file")
    d26 = assemble_2026(rank, num, row_offset=N_2025)
    if len(d26) != N_2026 or not pd.Index(d26.MVT_ID_mvt).is_unique:
        raise AssertionError(f"S3: 2026 part has {len(d26)} rows / duplicate ids")
    d = pd.concat([d25, d26[d25.columns]], ignore_index=True)
    del d25, d26, rank, q, o, raw
    check_contract(d, num, cat)
    d.to_parquet(OUT)
    Path(str(OUT) + ".cols.json").write_text(json.dumps({"num": num, "cat": cat}))
    del d
    # S2: every feature / target column of the 2025 part equals the source, re-read one column at a time
    for c in num + cat + ["delta", "G", "score_y", "score_proxy", "score_sp", "month", "row", "MVT_ID_mvt"]:
        a = pq.read_table(SRC, columns=[c]).column(0).to_pandas()
        b = pq.read_table(OUT, columns=[c]).column(0).to_pandas().iloc[:N_2025].reset_index(drop=True)
        if not a.equals(b):
            raise AssertionError(f"S2: column {c} of the 2025 part differs from the source input")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "n_2025": N_2025, "n_2026": N_2026, "wall_s": round(time.time() - t0, 1),
                      "es_rows": int(np.isin(pq.read_table(OUT, columns=["month", "es"]).to_pandas().month.iloc[:N_2025], ES_MONTHS).sum())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
