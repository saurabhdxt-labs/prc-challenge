"""Stand-occupancy evidence for the catastrophic negative-delta tail.

STAGE 1 (new): does the physical event happen at all?  Uses the hidden BLOCK_dep to ask
whether another aircraft in-blocked at the same stand between our off-block and AOBT_3.
RED_TEAM measured a FEATURE's tail coverage (prev_arr_gap<1500s -> 3.5%) and concluded the
bound was dead; that cannot distinguish "rare event" from "bad detector".  This can.

STAGE 2 (new): serve-time detectors that prev_arr_gap cannot express -
  COUNTS of arrivals at the stand in windows (>=2 arrivals => reoccupation is certain),
  arrivals AFTER AOBT_3 and before take-off,
  restricted to stands whose ARR/DEP alternation is historically clean.
"""
import gc, glob, pathlib
import pathlib
import numpy as np, pandas as pd, pyarrow.parquet as pq

RAW = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw") + "/"
C = ["PHASE_mvt","ADEP_mvt","ADES_mvt","STAND_mvt","MVT_TIME_UTC_mvt",
     "SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","TAXITIME_SEC_mvt","AOBT_3_flt"]
APTS = ["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH","LTFM"]
EPOCH = pd.Timestamp("2024-12-01", tz="UTC")
es = lambda s: (s - EPOCH).dt.total_seconds().to_numpy()   # unit-safe epoch seconds

out = []
for p in sorted(glob.glob(RAW + "training_2025-*.parquet")):
    t = pq.read_table(p, columns=C).to_pandas()

    arr = t[(t.PHASE_mvt == "ARR") & t.BLOCK_TIME_UTC_mvt.notna() & t.STAND_mvt.notna()]
    arr = arr[arr.ADES_mvt.isin(APTS)]
    a_key = (arr.ADES_mvt.astype(str) + "|" + arr.STAND_mvt.astype(str)).to_numpy()
    a_t = es(arr.BLOCK_TIME_UTC_mvt)
    o = np.argsort(a_t, kind="mergesort"); a_key, a_t = a_key[o], a_t[o]

    dep = t[(t.PHASE_mvt == "DEP") & t.TAXITIME_SEC_mvt.notna()
            & t.BLOCK_TIME_UTC_mvt.notna() & (t.TAXITIME_SEC_mvt > 0)
            & t.AOBT_3_flt.notna() & t.STAND_mvt.notna()]
    dep = dep[dep.ADEP_mvt.isin(APTS)].copy()
    d_key = (dep.ADEP_mvt.astype(str) + "|" + dep.STAND_mvt.astype(str)).to_numpy()
    blk, aobt = es(dep.BLOCK_TIME_UTC_mvt), es(dep.AOBT_3_flt)
    mvt, sch = es(dep.MVT_TIME_UTC_mvt), es(dep.SCHED_TIME_UTC_mvt)

    # per-stand sorted arrival times -> counts in arbitrary windows by searchsorted
    idx = {}
    for k, grp in pd.Series(np.arange(len(a_t))).groupby(a_key, sort=False):
        idx[k] = a_t[grp.to_numpy()]
    # guard: a NUL separator silently collapses arrow-backed string keys (10 groups, not 2277)
    assert len(idx) > 500, f"stand index collapsed to {len(idx)} keys - bad separator?"
    assert len(set(d_key) & set(idx)) > 500, "arrival/departure stand keys do not join"
    def cnt(lo, hi):
        r = np.zeros(len(d_key), dtype=np.int32)
        for k, rows in pd.Series(np.arange(len(d_key))).groupby(d_key, sort=False):
            ta = idx.get(k)
            if ta is None:
                continue
            j = rows.to_numpy()
            r[j] = np.searchsorted(ta, hi[j], "right") - np.searchsorted(ta, lo[j], "right")
        return r

    dep["delta"] = blk - aobt
    dep["apt"] = dep.ADEP_mvt
    # STAGE 1: truth-only diagnostic
    dep["n_blk_aobt"] = cnt(blk, aobt)          # arrivals between our off-block and AOBT_3
    dep["n_blk_mvt"]  = cnt(blk, mvt)           # ... and before take-off
    # STAGE 2: serve-time only (no BLOCK_dep anywhere)
    dep["n_sch_aobt"] = cnt(sch, aobt)          # arrivals between SCHED and AOBT_3
    dep["n_aobt_mvt"] = cnt(aobt, mvt)          # arrivals between AOBT_3 and take-off
    dep["n_2h_mvt"]   = cnt(mvt - 7200, mvt)    # arrivals in the 2 h before take-off
    out.append(dep[["apt","STAND_mvt","delta","n_blk_aobt","n_blk_mvt",
                    "n_sch_aobt","n_aobt_mvt","n_2h_mvt"]])
    del t, arr, dep, idx; gc.collect()

d = pd.concat(out, ignore_index=True); del out; gc.collect()
d['tl'] = d.delta < -600
base = d['tl'].mean()
print(f"matched labelled 2025 departures with a stand: {len(d):,}   "
      f"tail (delta<-600): {d['tl'].sum():,} ({100*base:.2f}%)\n")

print("=" * 100)
print("STAGE 1 - DOES THE PHYSICAL EVENT HAPPEN?  (uses hidden BLOCK; diagnostic only)")
print("  reocc = >=1 arrival in-blocked at this exact stand between our off-block and AOBT_3")
print("=" * 100)
print(f"{'apt':6s}{'n':>9s}{'tail n':>8s}{'P(reocc|tail)':>15s}{'P(reocc|ctrl)':>15s}"
      f"{'lift':>8s}{'P(tail|reocc)':>15s}")
for ap in APTS + ["ALL"]:
    a = d if ap == "ALL" else d[d.apt == ap]
    if not len(a): continue
    r = a.n_blk_aobt > 0
    pt, pc = r[a['tl']].mean(), r[~a['tl']].mean()
    prec = a['tl'][r].mean() if r.any() else np.nan
    print(f"{ap:6s}{len(a):9,d}{a['tl'].sum():8,d}{100*pt:14.2f}%{100*pc:14.2f}%"
          f"{(pt/pc if pc else np.inf):8.1f}{100*prec:14.2f}%")

print()
print("=" * 100)
print("STAGE 2 - SERVE-TIME COUNT DETECTORS (nothing here reads BLOCK_dep)")
print("=" * 100)
print(f"{'rule':34s}{'n':>10s}{'% rows':>8s}{'precision':>11s}{'tail cover':>12s}"
      f"{'median delta':>14s}{'mean delta':>12s}")
rules = [("n_sch_aobt >= 1", d.n_sch_aobt >= 1), ("n_sch_aobt >= 2", d.n_sch_aobt >= 2),
         ("n_aobt_mvt >= 1", d.n_aobt_mvt >= 1), ("n_aobt_mvt >= 2", d.n_aobt_mvt >= 2),
         ("n_2h_mvt   >= 2", d.n_2h_mvt >= 2),   ("n_2h_mvt   >= 3", d.n_2h_mvt >= 3),
         ("n_sch_aobt>=1 & n_aobt_mvt>=1", (d.n_sch_aobt >= 1) & (d.n_aobt_mvt >= 1))]
for name, m in rules:
    s = d[m]
    if len(s) < 50: continue
    print(f"{name:34s}{len(s):10,d}{100*m.mean():7.2f}%{100*s['tl'].mean():10.2f}%"
          f"{100*s['tl'].sum()/d['tl'].sum():11.2f}%{s.delta.median():14.0f}{s.delta.mean():12.0f}")
print(f"{'BASE RATE (no rule)':34s}{len(d):10,d}{100.0:7.2f}%{100*base:10.2f}%"
      f"{100.0:11.2f}%{d.delta.median():14.0f}{d.delta.mean():12.0f}")

print()
print("=" * 100)
print("STAGE 3 - IS THE CONSTRAINT SOUND?  overlap rate = arrivals landing on an occupied stand")
print("  (if stands were exclusive and correctly assigned this would be ~0)")
print("=" * 100)
for ap in APTS:
    a = d[d.apt == ap]
    print(f"  {ap}  P(>=1 arrival between our own off-block and take-off) = "
          f"{100*(a.n_blk_mvt > 0).mean():6.2f}%   "
          f"P(>=1 between off-block and AOBT_3) = {100*(a.n_blk_aobt > 0).mean():6.2f}%")
