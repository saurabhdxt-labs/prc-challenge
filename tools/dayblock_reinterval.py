"""Re-interval the overnight arms with a DATE-block bootstrap (dates resampled within January and
July separately, as the audit's queue check did), next to the row bootstrap every RESULT used.

Each fold record row is dated through the positional twins: cache_stand month file (y, in take-off
order) == cache_order month file (MVT_ID, same order) -> raw month file (MVT_ID -> MVT_TIME). The
alignment is VERIFIED by requiring the record's y to equal the stand cache's y row for row.
Light: one record at a time, numpy, no fitting.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq, sys

MONTHS = {1: "training_2025-01-01_2025-02-01.parquet", 7: "training_2025-07-01_2025-08-01.parquet"}

def dates_for_holdout():
    out = {}
    for m, f in MONTHS.items():
        y_stand = pq.read_table(f"data/cache_stand/{f}", columns=["y"]).column("y").to_numpy()
        ids = pq.read_table(f"data/cache_order/{f}", columns=["MVT_ID_mvt"]).column("MVT_ID_mvt").to_numpy()
        assert len(ids) == len(y_stand), (m, len(ids), len(y_stand))
        raw = pq.read_table(f"data/raw/{f}", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).to_pandas().set_index("MVT_ID_mvt")
        day = raw.MVT_TIME_UTC_mvt.reindex(ids).dt.strftime("%Y-%m-%d").to_numpy()
        assert pd.notna(day).all(), f"month {m}: some ids have no take-off time"
        out[m] = (y_stand.astype("float64"), day)
    return out

HOLD = dates_for_holdout()

def attach_dates(rec):
    day = np.empty(len(rec), dtype=object)
    for m, (y_stand, d) in HOLD.items():
        s = (rec.month.to_numpy() == m)
        yr = rec.y.to_numpy()[s].astype("float64")
        assert len(yr) == len(y_stand), f"month {m}: record {len(yr)} rows vs cache {len(y_stand)}"
        assert np.allclose(yr, y_stand), f"month {m}: record y does not match the stand cache row for row"
        day[s] = d
    return day

def taxi(rec, col):                  # delta-scale records: recover exactly as the harness does
    return np.maximum(rec.proxy.to_numpy() - rec[col].to_numpy(), 1.0)

def intervals(y, pa, pb, day, month, n_boot=2000, seed=0):
    se_a, se_b = (y - pa) ** 2, (y - pb) ** 2
    gain = np.sqrt(se_a.mean()) - np.sqrt(se_b.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    g_row = np.sqrt(se_a[idx].mean(1)) - np.sqrt(se_b[idx].mean(1))
    # date blocks, resampled within each month (stratified)
    keys = pd.Series(day)
    codes, uniq = pd.factorize(keys)
    Sa = np.bincount(codes, se_a); Sb = np.bincount(codes, se_b); N = np.bincount(codes).astype(float)
    mon_of = pd.Series(month).groupby(codes).first().to_numpy()
    groups = [np.flatnonzero(mon_of == mm) for mm in sorted(set(mon_of))]
    rng2 = np.random.default_rng(seed + 7)
    g_day = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([rng2.choice(g, size=len(g), replace=True) for g in groups])
        g_day[b] = np.sqrt(Sa[pick].sum() / N[pick].sum()) - np.sqrt(Sb[pick].sum() / N[pick].sum())
    lo_r, hi_r = np.percentile(g_row, [2.5, 97.5]); lo_d, hi_d = np.percentile(g_day, [2.5, 97.5])
    return gain, (lo_r, hi_r), (lo_d, hi_d), len(uniq)

rows = []
def run(name, rec, pa, pb, recorded=None):
    day = attach_dates(rec)
    g, r, dd, nd = intervals(rec.y.to_numpy().astype("float64"), pa, pb, day, rec.month.to_numpy())
    ratio = (dd[1] - dd[0]) / (r[1] - r[0])
    rows.append((name, g, r, dd, ratio, nd, recorded))

q = pd.read_parquet("data/cache_stand/fold_preds_queue.parquet")
run("queue block (v5)", q, taxi(q, "baseline"), taxi(q, "treatment"), "[+1.316, +1.812]")
for f, nm, rec_ci in (("queue_order", "arm F (record ordering)", "[+1.407, +1.993]"),
                      ("queue_weather", "arm W (weather)", "[+0.495, +0.947]"),
                      ("queue_day", "arm D (day regime)", "[-0.311, +0.210]")):
    r_ = pd.read_parquet(f"data/cache_stand/fold_preds_{f}.parquet")
    run(nm, r_, taxi(r_, "baseline"), taxi(r_, "treatment"), rec_ci)
yv = pd.read_parquet("data/cache_stand/fold_preds_queue_ytarget.parquet")
pipe, Yh = taxi(yv, "baseline"), taxi(yv, "Y")
run("arm Y blend (0.5 everywhere)", yv, pipe, taxi(yv, "blend"), "[-27.996, +1.094]")
gate = (np.abs(yv.baseline.to_numpy()) > 600) & (yv.ap.to_numpy() != "LIRF")
run("gated blend on Y, ex-LIRF (A27)", yv, pipe, np.where(gate, 0.5 * Yh + 0.5 * pipe, pipe), "[+0.234, +0.820]")

print(f"{'arm':34s} {'gain':>8s}  {'row-bootstrap 95%':>22s}  {'DATE-block 95%':>22s}  {'width x':>7s}  {'excl 0 (date)':>13s}")
for name, g, r, dd, ratio, nd, rec_ci in rows:
    print(f"{name:34s} {g:+8.3f}  [{r[0]:+8.3f}, {r[1]:+8.3f}]  [{dd[0]:+8.3f}, {dd[1]:+8.3f}]  {ratio:7.2f}  "
          f"{'yes' if dd[0] > 0 or dd[1] < 0 else 'NO':>13s}")
print(f"\nblocks: {rows[0][5]} dates (January and July resampled separately); recorded row intervals reproduced where shown")
