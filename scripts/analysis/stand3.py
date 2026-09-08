"""Two questions left:
  1. Anchoring the witness at TAKE-OFF instead of AOBT_3 enlarges the pool ~20x
     (0.5% -> ~10% of rows).  Is the bound still tight enough to be worth anything?
  2. Is the right vehicle a hard CONSTRAINT or a derived FEATURE?  Compare
     y = MVT - T_c  (the bound)  against  y = MVT - T_c + E[slack]  (bound + learned
     correction) against a per-airport constant, on the witnessed rows.
"""
import gc, glob, pathlib
import pathlib
import numpy as np, pandas as pd, pyarrow.parquet as pq

RAW = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw") + "/"
C = ["PHASE_mvt","ADEP_mvt","ADES_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt","MVT_TIME_UTC_mvt",
     "SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","TAXITIME_SEC_mvt","AOBT_3_flt"]
APTS = ["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH","LTFM"]
EPOCH = pd.Timestamp("2024-12-01", tz="UTC")
es = lambda s: (s - EPOCH).dt.total_seconds().to_numpy()

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
    mvt = es(dep.MVT_TIME_UTC_mvt)

    idx = {}
    for k, grp in pd.Series(np.arange(len(a_t))).groupby(a_key, sort=False):
        idx[k] = grp.to_numpy()
    assert len(idx) > 500, f"index collapsed to {len(idx)}"

    n = len(d_key); Tm = np.full(n, np.nan)
    for k, rows in pd.Series(np.arange(n)).groupby(d_key, sort=False):
        pos = idx.get(k)
        if pos is None: continue
        ta = a_t[pos]; j = rows.to_numpy()
        h = np.searchsorted(ta, mvt[j], "right") - 1     # last arrival <= TAKE-OFF
        ok = h >= 0
        Tm[j[ok]] = ta[h[ok]]
    dep["apt"] = dep.ADEP_mvt
    dep["y"] = dep.TAXITIME_SEC_mvt.astype(float)
    dep["delta"] = blk - aobt
    dep["wit"] = np.where(np.isnan(Tm), False, Tm > blk)   # TRUTH: retaken before take-off
    dep["slack"] = Tm - blk                                 # TRUTH
    dep["ybound"] = mvt - Tm                                # serve-time
    dep["gapm"] = mvt - Tm                                  # serve-time (same thing)
    dep["gapa"] = aobt - Tm                                 # serve-time
    out.append(dep[["apt","y","delta","wit","slack","ybound","gapm","gapa"]])
    del t, arr, dep, idx; gc.collect()

d = pd.concat(out, ignore_index=True); del out; gc.collect()
d = d[d.gapm.notna()]
d["tl"] = d.delta < -600
print(f"rows with any arrival at the stand before take-off: {len(d):,}")
print(f"of those, TRUE witnesses (arrival landed after our off-block): "
      f"{d.wit.sum():,} ({100*d.wit.mean():.2f}%)   [AOBT-anchored was 0.53%]\n")

print("=" * 100)
print("1. WITNESS POOL anchored at TAKE-OFF, and how tight the bound is")
print("=" * 100)
W = d[d.wit]
print(f"{'apt':6s}{'witnesses':>11s}{'% of rows':>11s}{'P(tail|wit)':>13s}"
      f"{'slack p10':>11s}{'p50':>8s}{'p90':>8s}")
for ap in APTS + ["ALL"]:
    a = W if ap == "ALL" else W[W.apt == ap]
    b = d if ap == "ALL" else d[d.apt == ap]
    if len(a) < 50: continue
    print(f"{ap:6s}{len(a):11,d}{100*len(a)/len(b):10.2f}%{100*a.tl.mean():12.2f}%"
          f"{a.slack.quantile(.1):11.0f}{a.slack.quantile(.5):8.0f}{a.slack.quantile(.9):8.0f}")

print()
print("=" * 100)
print("2. CONSTRAINT vs FEATURE on the witnessed rows.  RMSE against true y")
print("   bound      = MVT - T_c                (commit to the inequality)")
print("   bound+corr = MVT - T_c + median slack (bound plus a learned offset)")
print("   const      = per-airport mean of y on the same rows (no stand info at all)")
print("=" * 100)
print(f"{'apt':6s}{'n':>9s}{'bound':>10s}{'bound+corr':>12s}{'const':>10s}"
      f"{'verdict':>26s}")
for ap in APTS + ["ALL"]:
    a = W if ap == "ALL" else W[W.apt == ap]
    if len(a) < 50: continue
    corr = a.slack.median()
    r_b = np.sqrt(((a.y - a.ybound) ** 2).mean())
    r_c = np.sqrt(((a.y - (a.ybound + corr)) ** 2).mean())
    r_k = np.sqrt(((a.y - a.y.mean()) ** 2).mean())
    v = "feature beats constant" if r_c < r_k else "NO GAIN over a constant"
    print(f"{ap:6s}{len(a):9,d}{r_b:10.1f}{r_c:12.1f}{r_k:10.1f}{v:>26s}")

print()
print("=" * 100)
print("3. SERVE-TIME DETECTION of a true witness, take-off anchored")
print("=" * 100)
print(f"  base rate P(witness) = {100*d.wit.mean():.2f}%")
print(f"\n{'rule':34s}{'n':>10s}{'P(witness)':>12s}{'recall':>9s}{'median true slack':>19s}")
for name, m in [("gapm <= 600", d.gapm <= 600), ("gapm <= 1200", d.gapm <= 1200),
                ("gapm <= 1800", d.gapm <= 1800), ("gapm <= 3600", d.gapm <= 3600),
                ("gapa <= 600", d.gapa <= 600), ("gapa <= 1800", d.gapa <= 1800)]:
    m = m.fillna(False); s = d[m]
    if len(s) < 50: continue
    print(f"{name:34s}{len(s):10,d}{100*s.wit.mean():11.2f}%"
          f"{100*s.wit.sum()/d.wit.sum():8.2f}%{s[s.wit].slack.median():19.0f}")

print()
print("=" * 100)
print("4. CEILING: MSE removed on the 344,841-row score if witnessed rows were PERFECT")
print("=" * 100)
scale = 344841 / len(d)
for name, sub in [("AOBT-anchored witnesses (0.53%)", W[W.gapa <= 1800]),
                  ("take-off anchored witnesses", W)]:
    nrows = len(sub) * scale
    for assumed in (1400.0, 1000.0):
        ceil = nrows * assumed ** 2 / 344841
        print(f"  {name:34s} n2026~{nrows:7.0f}  if model now scores {assumed:6.0f} "
              f"-> ceiling {ceil:8.0f} MSE  ({296.7 - np.sqrt(max(88036-ceil,1)):5.1f} s)")
