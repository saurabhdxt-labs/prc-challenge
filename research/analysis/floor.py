"""Model-free floor: how much of delta is irreducible given the observable context?

Repeat-pair estimator. Take rows that are twins in the observable feature space
(same airport, stand, runway, airline, hour-of-day, same NM-delay band, same month).
Any remaining variation between twins cannot be explained by ANY model built on those
columns, so sd(difference)/sqrt(2) is a lower bound on achievable RMSE.
Coarsening the key upward traces the bias/variance of the bound.
"""
import gc, glob, pathlib
import pathlib
import numpy as np, pandas as pd, pyarrow.parquet as pq

RAW = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw") + "/"
C = ["PHASE_mvt","ADEP_mvt","ADES_mvt","STAND_mvt","RUNWAY_mvt","FLIGHT_mvt",
     "AIRCRAFT_TYPE_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt",
     "TAXITIME_SEC_mvt","AOBT_3_flt","EOBT_1_flt"]
sec = lambda a, b: (a - b).dt.total_seconds()

rows = []
for p in sorted(glob.glob(RAW + "training_2025-*.parquet")):
    t = pq.read_table(p, columns=C).to_pandas()
    t = t[(t.PHASE_mvt == "DEP") & t.TAXITIME_SEC_mvt.notna()
          & t.BLOCK_TIME_UTC_mvt.notna() & (t.TAXITIME_SEC_mvt > 0) & t.AOBT_3_flt.notna()]
    t["delta"] = sec(t.BLOCK_TIME_UTC_mvt, t.AOBT_3_flt)
    t["nmdelay"] = sec(t.AOBT_3_flt, t.SCHED_TIME_UTC_mvt)
    t["blk_sec"] = t.BLOCK_TIME_UTC_mvt.dt.second
    t["aobt_sec"] = t.AOBT_3_flt.dt.second
    t["airline"] = t.FLIGHT_mvt.astype(str).str[:3]
    t["hr"] = t.MVT_TIME_UTC_mvt.dt.hour
    t["month"] = t.MVT_TIME_UTC_mvt.dt.month
    t["dband"] = np.clip((t.nmdelay // 300), -2, 24)
    rows.append(t[["ADEP_mvt","STAND_mvt","RUNWAY_mvt","airline","AIRCRAFT_TYPE_mvt",
                   "ADES_mvt","hr","month","dband","delta","blk_sec","aobt_sec"]])
    del t; gc.collect()
d = pd.concat(rows, ignore_index=True); del rows; gc.collect()
print(f"matched labelled 2025 departures: {len(d):,}\n")

print("=" * 92)
print("TIMESTAMP GRANULARITY  (is BLOCK a clock reading or a minute-rounded entry?)")
print("=" * 92)
print(f"{'apt':6s}{'BLOCK sec==0':>14s}{'AOBT_3 sec==0':>15s}{'delta mult of 60':>19s}")
for ap in sorted(d.ADEP_mvt.unique()):
    a = d[d.ADEP_mvt == ap]
    print(f"{ap:6s}{100*(a.blk_sec==0).mean():13.1f}%{100*(a.aobt_sec==0).mean():14.1f}%"
          f"{100*(a.delta % 60 == 0).mean():18.1f}%")

print()
print("=" * 92)
print("REPEAT-PAIR FLOOR on delta.  sd(within-twin difference)/sqrt(2) = irreducible RMSE bound")
print("=" * 92)
KEYS = [
    ("apt+month",                 ["ADEP_mvt","month"]),
    ("apt+stand+month",           ["ADEP_mvt","STAND_mvt","month"]),
    ("apt+stand+rwy+month",       ["ADEP_mvt","STAND_mvt","RUNWAY_mvt","month"]),
    ("+airline",                  ["ADEP_mvt","STAND_mvt","RUNWAY_mvt","airline","month"]),
    ("+airline+hr",               ["ADEP_mvt","STAND_mvt","RUNWAY_mvt","airline","month","hr"]),
    ("+airline+hr+delayband",     ["ADEP_mvt","STAND_mvt","RUNWAY_mvt","airline","month","hr","dband"]),
    ("+type+dest (finest)",       ["ADEP_mvt","STAND_mvt","RUNWAY_mvt","airline","month","hr",
                                   "dband","AIRCRAFT_TYPE_mvt","ADES_mvt"]),
]
print(f"{'key':26s}{'groups>=2':>11s}{'rows used':>11s}{'floor RMSE':>12s}{'trimmed(99%)':>14s}")
for name, k in KEYS:
    g = d.groupby(k, observed=True).delta
    n = g.transform("size")
    m = d.delta[n >= 2]
    grp_mean = g.transform("mean")[n >= 2]
    grp_n = n[n >= 2]
    # unbiased within-group variance -> irreducible sd
    resid = (m - grp_mean) * np.sqrt(grp_n / (grp_n - 1))
    floor = float(np.sqrt((resid ** 2).mean()))
    q = np.quantile(np.abs(resid), 0.99)
    tr = float(np.sqrt((resid[np.abs(resid) <= q] ** 2).mean()))
    ngrp = int(d[n >= 2].groupby(k, observed=True).ngroups)
    print(f"{name:26s}{ngrp:11,d}{len(m):11,d}{floor:12.1f}{tr:14.1f}")

print()
print("Reference: constant-per-airport 364.3 | shipped model on this target 230.4")
print("A floor materially below 230 means the matched side is NOT exhausted.")
