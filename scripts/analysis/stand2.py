"""Stage 1 proved reoccupation happens (lift 39x, precision 61%).  Two questions decide
whether it is worth MSE:

  A. ORACLE VALUE. On reoccupied rows, how close is the reoccupying arrival T_c to our
     true off-block?  If T_c - BLOCK is small, y_hat = MVT - T_c is a near-exact predictor
     and the constraint attacks tail MAGNITUDE, not just probability.
  B. DETECTABILITY. Can we tell "T_c is a reoccupation" from "T_c is our own inbound"
     using serve-time columns only?  The constraint is only safe where this is precise.
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
    a_ty = arr.AIRCRAFT_TYPE_mvt.astype(str).to_numpy()
    o = np.argsort(a_t, kind="mergesort"); a_key, a_t, a_ty = a_key[o], a_t[o], a_ty[o]

    dep = t[(t.PHASE_mvt == "DEP") & t.TAXITIME_SEC_mvt.notna()
            & t.BLOCK_TIME_UTC_mvt.notna() & (t.TAXITIME_SEC_mvt > 0)
            & t.AOBT_3_flt.notna() & t.STAND_mvt.notna()]
    dep = dep[dep.ADEP_mvt.isin(APTS)].copy()
    d_key = (dep.ADEP_mvt.astype(str) + "|" + dep.STAND_mvt.astype(str)).to_numpy()
    blk, aobt = es(dep.BLOCK_TIME_UTC_mvt), es(dep.AOBT_3_flt)
    mvt, sch = es(dep.MVT_TIME_UTC_mvt), es(dep.SCHED_TIME_UTC_mvt)

    idx = {}
    for k, grp in pd.Series(np.arange(len(a_t))).groupby(a_key, sort=False):
        idx[k] = grp.to_numpy()
    assert len(idx) > 500, f"stand index collapsed to {len(idx)}"

    n = len(d_key)
    Tc = np.full(n, np.nan); Tc_ty = np.empty(n, dtype=object)
    Tp = np.full(n, np.nan)                       # the arrival BEFORE Tc
    ncnt = np.zeros(n, dtype=np.int32)            # arrivals in (SCHED, AOBT_3]
    for k, rows in pd.Series(np.arange(n)).groupby(d_key, sort=False):
        pos = idx.get(k)
        if pos is None: continue
        ta = a_t[pos]; j = rows.to_numpy()
        h = np.searchsorted(ta, aobt[j], "right") - 1        # last arrival <= AOBT_3
        ok = h >= 0
        Tc[j[ok]] = ta[h[ok]]
        Tc_ty[j[ok]] = a_ty[pos[h[ok]]]
        ok2 = h >= 1
        Tp[j[ok2]] = ta[h[ok2] - 1]
        ncnt[j] = (np.searchsorted(ta, aobt[j], "right")
                   - np.searchsorted(ta, sch[j], "right"))
    dep["apt"] = dep.ADEP_mvt
    dep["y"] = dep.TAXITIME_SEC_mvt.astype(float)
    dep["delta"] = blk - aobt
    dep["reocc"] = np.where(np.isnan(Tc), False, Tc > blk)   # TRUTH
    dep["gap"] = aobt - Tc                                   # serve-time
    dep["gap_mvt"] = mvt - Tc                                # serve-time
    dep["gap_prev"] = aobt - Tp                              # serve-time
    dep["ncnt"] = ncnt                                       # serve-time
    dep["ty_match"] = (dep.AIRCRAFT_TYPE_mvt.astype(str).to_numpy() == Tc_ty)
    dep["y_con"] = mvt - Tc                                  # prediction if we trust Tc
    dep["y_proxy"] = mvt - aobt
    dep["slack"] = Tc - blk                                  # TRUTH: how tight is the bound
    out.append(dep[["apt","y","delta","reocc","gap","gap_mvt","gap_prev","ncnt",
                    "ty_match","y_con","y_proxy","slack"]])
    del t, arr, dep, idx; gc.collect()

d = pd.concat(out, ignore_index=True); del out; gc.collect()
d = d[d.gap.notna()]
print(f"rows with an arrival at the stand before AOBT_3: {len(d):,}\n")

print("=" * 96)
print("A. ORACLE VALUE - on truly reoccupied rows, is MVT - T_c a good predictor of y?")
print("=" * 96)
R = d[d.reocc]
print(f"reoccupied rows: {len(R):,}   slack = T_c - BLOCK (s): "
      f"p10 {R.slack.quantile(.1):.0f}  p50 {R.slack.quantile(.5):.0f}  "
      f"p90 {R.slack.quantile(.9):.0f}")
print(f"{'apt':6s}{'n':>8s}{'RMSE(MVT-T_c)':>15s}{'RMSE(MVT-AOBT)':>16s}{'RMSE(y=const)':>15s}"
      f"{'median y':>10s}")
for ap in APTS + ["ALL"]:
    a = R if ap == "ALL" else R[R.apt == ap]
    if len(a) < 50: continue
    r1 = np.sqrt(((a.y - a.y_con) ** 2).mean())
    r2 = np.sqrt(((a.y - a.y_proxy) ** 2).mean())
    r3 = a.y.std()
    print(f"{ap:6s}{len(a):8,d}{r1:15.1f}{r2:16.1f}{r3:15.1f}{a.y.median():10.0f}")

print()
print("=" * 96)
print("B. DETECTABILITY - P(reocc) given serve-time quantities only")
print("=" * 96)
print(f"  base rate P(reocc) = {100*d.reocc.mean():.3f}%")
print(f"\n{'serve-time rule':46s}{'n':>10s}{'P(reocc)':>11s}{'recall':>9s}")
rules = [
    ("gap <= 300",                       d.gap <= 300),
    ("gap <= 600",                       d.gap <= 600),
    ("gap <= 900",                       d.gap <= 900),
    ("gap <= 1800",                      d.gap <= 1800),
    ("gap <= 600 & ncnt >= 1",           (d.gap <= 600) & (d.ncnt >= 1)),
    ("gap <= 600 & ~ty_match",           (d.gap <= 600) & (~d.ty_match)),
    ("gap <= 900 & ~ty_match",           (d.gap <= 900) & (~d.ty_match)),
    ("gap <= 900 & ~ty_match & ncnt>=1", (d.gap <= 900) & (~d.ty_match) & (d.ncnt >= 1)),
    ("gap <= 1800 & ~ty_match & ncnt>=1",(d.gap <= 1800) & (~d.ty_match) & (d.ncnt >= 1)),
    ("gap <= 1800 & gap_prev >= 7200",   (d.gap <= 1800) & (d.gap_prev >= 7200)),
]
for name, m in rules:
    m = m.fillna(False); s = d[m]
    if len(s) < 30: continue
    print(f"{name:46s}{len(s):10,d}{100*s.reocc.mean():10.2f}%"
          f"{100*s.reocc.sum()/d.reocc.sum():8.2f}%")

print()
print("=" * 96)
print("C. PER-AIRPORT detectability at gap<=900 & ~ty_match")
print("=" * 96)
m = ((d.gap <= 900) & (~d.ty_match)).fillna(False)
print(f"{'apt':6s}{'n flagged':>11s}{'P(reocc)':>11s}{'recall':>9s}"
      f"{'RMSE(MVT-T_c) on flagged':>26s}{'RMSE(MVT-AOBT) same':>21s}")
for ap in APTS:
    s = d[m & (d.apt == ap)]
    if len(s) < 30: continue
    r1 = np.sqrt(((s.y - s.y_con) ** 2).mean()); r2 = np.sqrt(((s.y - s.y_proxy) ** 2).mean())
    rec = 100*s.reocc.sum()/max(d[d.apt == ap].reocc.sum(), 1)
    print(f"{ap:6s}{len(s):11,d}{100*s.reocc.mean():10.2f}%{rec:8.2f}%{r1:26.1f}{r2:21.1f}")
