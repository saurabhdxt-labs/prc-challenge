"""Does the thing we actually model - delta = BLOCK - AOBT_3 - drift within 2025?

The shipped fold holds out Jan+Jul 2025 and trains on the ten months AROUND them: it
interpolates in time. The real task trains on all of 2025 and extrapolates forward into
2026. If delta drifts month to month, the fold is optimistic by construction.
No model is fitted here - these are the raw conditional statistics.
"""
import gc, glob, pathlib
import pathlib
import numpy as np, pandas as pd, pyarrow.parquet as pq

RAW = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw") + "/"
C = ["PHASE_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt",
     "TAXITIME_SEC_mvt","AOBT_3_flt","EOBT_1_flt"]
sec = lambda a, b: (a - b).dt.total_seconds()

rows = []
for p in sorted(glob.glob(RAW + "training_2025-*.parquet")):
    t = pq.read_table(p, columns=C).to_pandas()
    t = t[(t.PHASE_mvt == "DEP") & t.TAXITIME_SEC_mvt.notna()
          & t.BLOCK_TIME_UTC_mvt.notna() & (t.TAXITIME_SEC_mvt > 0)]
    t = t[t.AOBT_3_flt.notna()]
    t["month"] = t.MVT_TIME_UTC_mvt.dt.month
    t["delta"] = sec(t.BLOCK_TIME_UTC_mvt, t.AOBT_3_flt)
    t["aobt_eobt"] = sec(t.AOBT_3_flt, t.EOBT_1_flt)
    t["y"] = t.TAXITIME_SEC_mvt.astype(float)
    rows.append(t[["ADEP_mvt","month","delta","aobt_eobt","y"]])
    del t; gc.collect()
d = pd.concat(rows, ignore_index=True); del rows; gc.collect()
print(f"matched labelled 2025 departures: {len(d):,}\n")

print("=" * 108)
print("A. delta = BLOCK - AOBT_3 : the quantity the matched model predicts.  Per airport, per month.")
print("   sd(delta) is the RMSE a constant-per-airport predictor would achieve.")
print("=" * 108)
print(f"{'apt':6s}{'sd(delta)':>10s}{'|':>2s}" + "".join(f"{m:>7d}" for m in range(1, 13))
      + f"{'  spread':>9s}")
for ap in sorted(d.ADEP_mvt.unique()):
    a = d[d.ADEP_mvt == ap]
    med = [a[a.month == m].delta.median() for m in range(1, 13)]
    sd = a.delta.std()
    print(f"{ap:6s}{sd:10.0f}{'|':>2s}" + "".join(f"{v:7.0f}" for v in med)
          + f"{max(med)-min(med):9.0f}")

print()
print("=" * 108)
print("B. P(delta < -600) by month - the slot-hold tail that carries 30.5% of matched SSE")
print("=" * 108)
print(f"{'apt':6s}{'|':>2s}" + "".join(f"{m:>7d}" for m in range(1, 13)) + f"{'  ratio':>9s}")
for ap in sorted(d.ADEP_mvt.unique()):
    a = d[d.ADEP_mvt == ap]
    r = [100*(a[a.month == m].delta < -600).mean() for m in range(1, 13)]
    print(f"{ap:6s}{'|':>2s}" + "".join(f"{v:7.2f}" for v in r)
          + f"{(max(r)/max(min(r),1e-9)):9.1f}x")

print()
print("=" * 108)
print("C. What a constant-per-airport delta predictor scores vs the model's 230.4")
print("=" * 108)
tot_n = tot_sse = 0
for ap in sorted(d.ADEP_mvt.unique()):
    a = d[d.ADEP_mvt == ap]
    sse = ((a.delta - a.delta.mean())**2).sum()
    tot_n += len(a); tot_sse += sse
    print(f"  {ap}  n={len(a):8,d}  sd(delta)={a.delta.std():7.1f}  "
          f"sd(y)={a.y.std():8.1f}  ratio={a.delta.std()/a.y.std():5.3f}")
print(f"\n  POOLED constant-per-airport delta RMSE = {np.sqrt(tot_sse/tot_n):.2f}")
print(f"  best model (fold A, matched)           = 230.41")
print(f"  -> variance explained by 70 features   = "
      f"{100*(1 - 230.41**2/(tot_sse/tot_n)):.1f}%")

print()
print("=" * 108)
print("D. LFPG aobt_eobt by month (the feature that shifted -300 s into 2026)")
print("=" * 108)
for ap in ["LFPG", "EGLL", "EHAM"]:
    a = d[d.ADEP_mvt == ap]
    print(f"  {ap} median aobt_eobt: "
          + " ".join(f"{a[a.month==m].aobt_eobt.median():.0f}" for m in range(1, 13)))
    print(f"  {ap} median delta    : "
          + " ".join(f"{a[a.month==m].delta.median():.0f}" for m in range(1, 13)))
