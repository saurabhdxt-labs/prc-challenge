"""Train/serve audit: do the features the matched model leans on behave the same in 2026?

Cut PER AIRPORT throughout (pooling has hidden the truth four times in this project).
Reads only the needed columns; peak RSS target < 1.5 GB.
"""
import gc
import pathlib
import numpy as np, pandas as pd, pyarrow.parquet as pq

RAW = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw") + "/"
C = ["PHASE_mvt","MVT_ID_mvt","ADEP_mvt","ADES_mvt","STAND_mvt","RUNWAY_mvt",
     "AIRCRAFT_TYPE_mvt","FLIGHT_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt",
     "BLOCK_TIME_UTC_mvt","TAXITIME_SEC_mvt","AOBT_3_flt","EOBT_1_flt","LOBT_flt",
     "IOBT_flt","AIRCRAFT_OPERATOR_flt"]

def sec(a, b):
    return (a - b).dt.total_seconds()

def prep(df):
    df = df[df.PHASE_mvt == "DEP"].copy()
    df["proxy"] = sec(df.MVT_TIME_UTC_mvt, df.AOBT_3_flt)
    df["sp"]    = sec(df.MVT_TIME_UTC_mvt, df.SCHED_TIME_UTC_mvt)
    df["aobt_eobt"] = sec(df.AOBT_3_flt, df.EOBT_1_flt)
    df["lobt_sched"] = sec(df.LOBT_flt, df.SCHED_TIME_UTC_mvt)
    df["nmdelay"] = sec(df.AOBT_3_flt, df.SCHED_TIME_UTC_mvt)
    df["ars"] = (df.ADEP_mvt.astype(str) + "|" + df.RUNWAY_mvt.astype(str)
                 + "|" + df.STAND_mvt.astype(str))
    df["matched"] = df.AOBT_3_flt.notna()
    return df

# ---- 2026 scored rows (label null by construction) ----
r = pq.read_table(RAW + "ranking.parquet", columns=C).to_pandas()
r = prep(r)
print(f"2026 DEP rows {len(r):,}  unmatched {(~r.matched).sum():,} "
      f"({100*(~r.matched).mean():.3f}%)")

# ---- 2025 Jan+Jul (the validation fold) and the full training year ----
tr = pd.concat([pq.read_table(RAW + p, columns=C).to_pandas() for p in
                ["training_2025-01-01_2025-02-01.parquet",
                 "training_2025-07-01_2025-08-01.parquet"]], ignore_index=True)
tr = prep(tr)
lab = tr.TAXITIME_SEC_mvt.notna() & tr.BLOCK_TIME_UTC_mvt.notna() & (tr.TAXITIME_SEC_mvt > 0)
tr = tr[lab]
print(f"2025 Jan+Jul DEP labelled {len(tr):,}  unmatched {(~tr.matched).sum():,} "
      f"({100*(~tr.matched).mean():.3f}%)\n")

# ---- 1. nullity of every clock, per airport, 2025 vs 2026 ----
print("=" * 100)
print("1. CLOCK AVAILABILITY  (% non-null)   2025 Jan+Jul  ->  2026 Jan+Jul")
print("=" * 100)
cols = ["AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt", "STAND_mvt",
        "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"]
print(f"{'apt':6s}" + "".join(f"{c.replace('_flt','').replace('_mvt',''):>22s}" for c in cols))
for ap in sorted(r.ADEP_mvt.unique()):
    a, b = tr[tr.ADEP_mvt == ap], r[r.ADEP_mvt == ap]
    row = f"{ap:6s}"
    for c in cols:
        p25, p26 = 100*a[c].notna().mean(), 100*b[c].notna().mean()
        flag = "  <<" if abs(p25 - p26) > 5 else ""
        row += f"{p25:8.1f}->{p26:6.1f}{flag:>8s}"
    print(row)

# ---- 2. distribution of the load-bearing features, per airport ----
print()
print("=" * 100)
print("2. FEATURE DISTRIBUTIONS, matched rows only   median [p10, p90]   2025 -> 2026")
print("=" * 100)
for feat in ["proxy", "aobt_eobt", "nmdelay", "sp"]:
    print(f"\n--- {feat} ---")
    for ap in sorted(r.ADEP_mvt.unique()):
        a = tr[(tr.ADEP_mvt == ap) & tr.matched][feat].dropna()
        b = r[(r.ADEP_mvt == ap) & r.matched][feat].dropna()
        if len(a) < 100 or len(b) < 100:
            continue
        d = b.median() - a.median()
        flag = "  <<<< SHIFT" if abs(d) > 60 else ""
        print(f"  {ap}  {a.median():8.0f} [{a.quantile(.1):7.0f},{a.quantile(.9):7.0f}]"
              f"  ->  {b.median():8.0f} [{b.quantile(.1):7.0f},{b.quantile(.9):7.0f}]"
              f"   d={d:+7.0f}{flag}")

# ---- 3. categorical vocabulary coverage: do 2026 keys exist in 2025? ----
print()
print("=" * 100)
print("3. ENCODING VOCABULARY COVERAGE  (share of 2026 scored rows whose key was seen in 2025)")
print("=" * 100)
full = []
for p in sorted(__import__("glob").glob(RAW + "training_2025-*.parquet")):
    t = pq.read_table(p, columns=["PHASE_mvt","ADEP_mvt","ADES_mvt","STAND_mvt","RUNWAY_mvt",
                                  "AIRCRAFT_TYPE_mvt","FLIGHT_mvt","AIRCRAFT_OPERATOR_flt",
                                  "TAXITIME_SEC_mvt","BLOCK_TIME_UTC_mvt"]).to_pandas()
    t = t[(t.PHASE_mvt == "DEP") & t.TAXITIME_SEC_mvt.notna() & t.BLOCK_TIME_UTC_mvt.notna()]
    t["ars"] = (t.ADEP_mvt.astype(str) + "|" + t.RUNWAY_mvt.astype(str)
                + "|" + t.STAND_mvt.astype(str))
    t["airline"] = t.FLIGHT_mvt.astype(str).str[:3]
    full.append(t[["ADEP_mvt","ADES_mvt","STAND_mvt","RUNWAY_mvt","AIRCRAFT_TYPE_mvt",
                   "AIRCRAFT_OPERATOR_flt","ars","airline"]])
    del t; gc.collect()
full = pd.concat(full, ignore_index=True)
print(f"(2025 training vocabulary built from {len(full):,} labelled departures)")
r["airline"] = r.FLIGHT_mvt.astype(str).str[:3]

keys = ["STAND_mvt", "RUNWAY_mvt", "ars", "airline", "ADES_mvt",
        "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]
print(f"{'apt':6s}{'n2026':>9s}" + "".join(f"{k.replace('_mvt','').replace('_flt',''):>12s}"
                                           for k in keys))
for ap in sorted(r.ADEP_mvt.unique()):
    b = r[r.ADEP_mvt == ap]
    v = full[full.ADEP_mvt == ap]
    row = f"{ap:6s}{len(b):9,d}"
    for k in keys:
        seen = set(v[k].dropna().unique())
        cov = 100 * b[k].isin(seen).mean()
        row += f"{cov:11.1f}%"
    print(row)

print("\nTotals over all 344,841 scored rows:")
for k in keys:
    seen = set(full[k].dropna().unique())
    print(f"  {k:24s} {100*r[k].isin(seen).mean():6.2f}%  "
          f"(2026 distinct {r[k].nunique():,}, 2025 distinct {full[k].nunique():,})")
