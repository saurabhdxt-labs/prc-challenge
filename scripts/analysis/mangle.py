"""Mangled-designator separator for the unmatched stratum - LOMO validation.

FAMILY_B.md §9.1 found that LIRF unmatched rows with a doubled-prefix designator
(ITYTY680, RYRR90BN, ...) are 91% fill-or-slip vs 64%, and carry 9 of the 12 24h-slip
rows.  It was never validated out of sample.  This is that test.

Incumbent  : per (airport, coarse sp bucket) fill-rate mixture, hierarchical k=5.
Variant    : identical, with `mangled` added as a cell dimension at LIRF only.
Separator  : regex on FLIGHT_mvt - a serve-time column.
Reported as GLOBAL MSE removed on the 344,841-row score, against the 26,284 target.
"""
import glob, pathlib, re
import pathlib
import numpy as np, pandas as pd, pyarrow.parquet as pq

RAW = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw") + "/"
C = ["PHASE_mvt","ADEP_mvt","STAND_mvt","FLIGHT_mvt","MVT_TIME_UTC_mvt",
     "SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","TAXITIME_SEC_mvt","AOBT_3_flt"]
EDGES = [-np.inf, 900, 1800, 3600, 7200, 10800, 21600, 43200, np.inf]
LAB = ["<15m","15-30m","30-60m","1-2h","2-3h","3-6h","6-12h","12h+"]
K = 5.0
N_SCORED, N_UNMATCHED_2026 = 344841, 5290
W_U = N_UNMATCHED_2026 / N_SCORED
MANGLED = re.compile(r"^[A-Z]{3}[A-Z]{1,3}[0-9]")
S = lambda a, b: (a - b).dt.total_seconds()

fr = []
for p in sorted(glob.glob(RAW + "training_2025-*.parquet")):
    t = pq.read_table(p, columns=C).to_pandas()
    t = t[(t.PHASE_mvt == "DEP") & t.TAXITIME_SEC_mvt.notna()
          & t.BLOCK_TIME_UTC_mvt.notna() & (t.TAXITIME_SEC_mvt > 0)
          & t.AOBT_3_flt.isna()]
    fr.append(t)
d = pd.concat(fr, ignore_index=True); del fr
d = d.sort_values(["BLOCK_TIME_UTC_mvt","MVT_TIME_UTC_mvt"], kind="mergesort")
d["y"] = d.TAXITIME_SEC_mvt.astype(float)
d["sp"] = S(d.MVT_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)
d["dd"] = S(d.BLOCK_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)
d["fill"] = d.dd.abs() <= 60
d["cb"] = pd.cut(d.sp, EDGES, labels=LAB).astype(str)
d["month"] = d.MVT_TIME_UTC_mvt.dt.month
d["ap"] = d.ADEP_mvt.astype(str)
f = d.FLIGHT_mvt.astype(str)
d["mangled"] = f.str.match(MANGLED).fillna(False) & (f.str.len() >= 7)
d = d.reset_index(drop=True)
print(f"unmatched 2025 departures: {len(d):,}   (report says 22,219)")
print(f"mangled overall {d.mangled.mean()*100:.2f}%   at LIRF "
      f"{d[d.ap=='LIRF'].mangled.mean()*100:.2f}%  "
      f"({d[d.ap=='LIRF'].mangled.sum()} of {(d.ap=='LIRF').sum()})")
for ap in sorted(d.ap.unique()):
    a = d[d.ap == ap]
    print(f"    {ap} mangled {100*a.mangled.mean():5.2f}%   "
          f"median y mangled {a[a.mangled].y.median() if a.mangled.any() else float('nan'):8.0f}"
          f"   non-mangled {a[~a.mangled].y.median():8.0f}")

def stats(tr, keys):
    g = tr.groupby(keys, observed=True)
    return pd.DataFrame({"p": g.fill.mean(), "nf": g.apply(
        lambda x: x.y[~x.fill].mean() if (~x.fill).any() else np.nan,
        include_groups=False), "n": g.size()})

def predict(tr, te, use_mangled):
    """Hierarchical shrink [ap] -> [ap,cb] -> (LIRF only) [ap,cb,mangled], k=5."""
    p0, nf0 = tr.fill.mean(), tr.y[~tr.fill].mean()
    L1 = stats(tr, ["ap"]); L2 = stats(tr, ["ap", "cb"])
    out_p = np.full(len(te), p0); out_nf = np.full(len(te), nf0)
    a1 = L1.reindex(te.ap.to_numpy())
    n1 = a1.n.to_numpy(); n1 = np.where(np.isnan(n1), 0, n1)
    out_p = (np.nan_to_num(a1.p.to_numpy()) * n1 + p0 * K) / (n1 + K)
    out_nf = (np.nan_to_num(a1.nf.to_numpy(), nan=nf0) * n1 + nf0 * K) / (n1 + K)
    idx2 = pd.MultiIndex.from_arrays([te.ap.to_numpy(), te.cb.to_numpy()])
    a2 = L2.reindex(idx2)
    n2 = np.nan_to_num(a2.n.to_numpy())
    out_p = (np.nan_to_num(a2.p.to_numpy()) * n2 + out_p * K) / (n2 + K)
    out_nf = (np.where(np.isnan(a2.nf.to_numpy()), out_nf, a2.nf.to_numpy()) * n2
              + out_nf * K) / (n2 + K)
    if use_mangled:
        trL = tr[tr.ap == "LIRF"]
        L3 = stats(trL, ["cb", "mangled"])
        m = (te.ap == "LIRF").to_numpy()
        idx3 = pd.MultiIndex.from_arrays([te.cb.to_numpy()[m], te.mangled.to_numpy()[m]])
        a3 = L3.reindex(idx3)
        n3 = np.nan_to_num(a3.n.to_numpy())
        out_p[m] = (np.nan_to_num(a3.p.to_numpy()) * n3 + out_p[m] * K) / (n3 + K)
        nf3 = a3.nf.to_numpy()
        out_nf[m] = (np.where(np.isnan(nf3), out_nf[m], nf3) * n3 + out_nf[m] * K) / (n3 + K)
    return np.maximum(out_p * te.sp.to_numpy() + (1 - out_p) * out_nf, 1.0)

pred = {False: np.zeros(len(d)), True: np.zeros(len(d))}
for mth in range(1, 13):
    te_m = d.month == mth
    tr, te = d[~te_m], d[te_m]
    for use in (False, True):
        pred[use][te_m.to_numpy()] = predict(tr, te, use)

y = d.y.to_numpy()
e0, e1 = y - pred[False], y - pred[True]
r0, r1 = np.sqrt((e0**2).mean()), np.sqrt((e1**2).mean())
dmse = W_U * (r0**2 - r1**2)
print("\n" + "=" * 92)
print("RESULT - LOMO over 12 months, 22k unmatched rows")
print("=" * 92)
print(f"  incumbent  (ap x coarse bucket)      stratum RMSE = {r0:8.2f}")
print(f"  + mangled dimension at LIRF          stratum RMSE = {r1:8.2f}")
print(f"  paired difference                                 = {r0-r1:+8.2f} s")

rng = np.random.default_rng(0)
bs = np.array([(lambda i: np.sqrt((e0[i]**2).mean()) - np.sqrt((e1[i]**2).mean()))
               (rng.integers(0, len(y), len(y))) for _ in range(2000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  paired row bootstrap 95% CI                       = [{lo:+.2f}, {hi:+.2f}]"
      f"   {'ESTABLISHED' if lo > 0 else 'NOT ESTABLISHED (interval spans zero)'}")
print(f"\n  GLOBAL MSE REMOVED = {dmse:+,.0f}   of the 26,284 needed "
      f"({100*dmse/26284:+.1f}%)")
v = ("MAJOR" if dmse > 4000 else "USEFUL" if dmse > 1000 else "ESSENTIALLY CLOSED")
print(f"  verdict against the owner's bands: {v}")

print(f"\n{'apt':6s}{'n':>7s}{'RMSE base':>11s}{'RMSE mangled':>14s}{'diff':>9s}")
for ap in sorted(d.ap.unique()):
    m = (d.ap == ap).to_numpy()
    a, b = np.sqrt((e0[m]**2).mean()), np.sqrt((e1[m]**2).mean())
    print(f"{ap:6s}{m.sum():7,d}{a:11.1f}{b:14.1f}{a-b:+9.1f}")
