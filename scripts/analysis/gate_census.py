"""Ten-airport ADS-B stand-gate census. Pre-registered in PREREG Amendment 6.

    python3.11 scripts/analysis/gate_census.py --day 2025-01-09 [--radius-from <json>]

Stand centroids are learned from the data (position of each aircraft at its own recorded
off-block), LEAVE-ONE-FLIGHT-OUT. The estimate is the last ADS-B sample inside radius R of
this flight's own stand. RMSE is bias-corrected, as a calibrated feature would be.
"""
import argparse, glob, json, pathlib
import numpy as np, pandas as pd, pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[2]
EP = pd.Timestamp('1970-01-01', tz='UTC')
M_PER_DEG = 111_320.0
#: per-airport matched-model RMSE, fold A baseline (reports/MSE_LEDGER.md A/B run)
MODEL = {'EDDF':189.6,'EDDM':178.0,'EGLL':243.6,'EHAM':173.2,'LEBL':225.9,
         'LEMD':182.5,'LFPG':256.3,'LIRF':397.8,'LSZH':187.7,'LTFM':257.1}
COV_BAR, RMSE_BAR, N_BAR = 0.50, 250.0, 6      # Amendment 6 section 4/5

ap_ = argparse.ArgumentParser(); ap_.add_argument('--day', required=True)
ap_.add_argument('--radius-from', default=None)
A = ap_.parse_args()

ad = pd.read_parquet(ROOT/'data'/'adsb'/f'adsb_{A.day}.parquet').sort_values('t')
assert 1.7e9 < ad.t.median() < 1.8e9, 'adsb epoch seconds wrong'
ad['cs'] = ad.callsign.fillna('NA').astype(str).str.strip()
C = ['PHASE_mvt','ADEP_mvt','STAND_mvt','CALLSIGN_flt','MVT_TIME_UTC_mvt',
     'BLOCK_TIME_UTC_mvt','TAXITIME_SEC_mvt','AOBT_3_flt']
src = [p for p in glob.glob(str(ROOT/'data'/'raw'/'training_2025-*.parquet'))
       if A.day[:7].replace('-','-') in pathlib.Path(p).stem]
t = pd.concat([pq.read_table(p, columns=C).to_pandas() for p in src], ignore_index=True)
d = t[(t.PHASE_mvt=='DEP') & t.TAXITIME_SEC_mvt.notna() & t.BLOCK_TIME_UTC_mvt.notna()
      & (t.TAXITIME_SEC_mvt>0) & t.AOBT_3_flt.notna() & t.STAND_mvt.notna()].copy()
d['blk'] = pd.to_datetime(d.BLOCK_TIME_UTC_mvt, utc=True)
d = d[(d.blk >= pd.Timestamp(A.day, tz='UTC')) &
      (d.blk < pd.Timestamp(A.day, tz='UTC') + pd.Timedelta(days=1))]
d['cs'] = d.CALLSIGN_flt.fillna('NA').astype(str).str.strip()
d['blk_s'] = (d.blk - EP).dt.total_seconds()
d['mvt_s'] = (pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True) - EP).dt.total_seconds()
d['stand'] = d.ADEP_mvt.astype(str) + '|' + d.STAND_mvt.astype(str)
d = d.reset_index(drop=True)
print(f'{A.day}: {len(d):,} labelled matched departures across '
      f'{d.ADEP_mvt.nunique()} airports; {len(ad):,} in-box ADS-B samples')

g = {k: v for k, v in ad.groupby(['airport','cs'])}
joined = np.array([(r.ADEP_mvt, r.cs) in g for r in d.itertuples()])
print(f'callsign join rate: {100*joined.mean():.1f}%')

pos = {}
for r in d.itertuples():
    tr = g.get((r.ADEP_mvt, r.cs))
    if tr is None: continue
    tt = tr.t.to_numpy(); i = int(np.abs(tt - r.blk_s).argmin())
    if abs(tt[i] - r.blk_s) < 300:
        pos.setdefault(r.stand, []).append((r.Index, tr.lat.to_numpy()[i], tr.lon.to_numpy()[i]))

def gated(r, R):
    tr = g.get((r.ADEP_mvt, r.cs))
    v = [p for p in pos.get(r.stand, []) if p[0] != r.Index]
    if tr is None or not v: return np.nan
    cla, clo = float(np.median([p[1] for p in v])), float(np.median([p[2] for p in v]))
    w = tr[(tr.t > r.mvt_s - 10800) & (tr.t <= r.mvt_s + 60)]
    if len(w) < 2: return np.nan
    la, lo, tt = w.lat.to_numpy(), w.lon.to_numpy(), w.t.to_numpy()
    dist = np.hypot((la-cla)*M_PER_DEG, (lo-clo)*M_PER_DEG*np.cos(np.radians(cla)))
    ins = dist <= R
    return tt[np.where(ins)[0][-1]] if ins.any() else np.nan

rows = list(d.itertuples())
best = {}
hdr = ''.join(('R=%d' % R).rjust(22) for R in (50, 100, 200))
print(f"\n{'apt':6s}{'dep':>6s}{'join':>7s}" + hdr)
print(f"{'':19s}" + ''.join(f"{'cov':>8s}{'RMSE':>7s}{'flag':>7s}" for _ in range(3)))
for ap in sorted(MODEL):
    m = (d.ADEP_mvt == ap).to_numpy()
    if m.sum() < 20: 
        print(f'{ap:6s}{m.sum():6,d}   too few departures'); continue
    line = f'{ap:6s}{m.sum():6,d}{100*joined[m].mean():6.0f}%'
    for R in (50, 100, 200):
        est = np.array([gated(r, R) if m[i] else np.nan for i, r in enumerate(rows)])
        res = est - d.blk_s.to_numpy()
        res = np.where(np.abs(res) > 10800, np.nan, res)[m]
        v = res[~np.isnan(res)]
        if len(v) < 20:
            line += f"{100*len(v)/m.sum():7.0f}%{'--':>7s}{'':>7s}"; continue
        rmse = float(np.sqrt(((v - np.median(v))**2).mean())); cov = len(v)/m.sum()
        ok = (cov >= COV_BAR) and (rmse <= RMSE_BAR)
        if ok and (ap not in best or rmse < best[ap][1]): best[ap] = (R, rmse, cov)
        line += f'{100*cov:7.0f}%{rmse:7.0f}{"PASS" if ok else "":>7s}'
    print(line)

n_pass = len(best)
print(f'\nAmendment 6 bars: coverage >= {COV_BAR:.0%}, bias-corrected RMSE <= {RMSE_BAR:.0f} s')
print(f'airports clearing both: {n_pass} of 10  (threshold {N_BAR})')
for ap, (R, rm, cv) in sorted(best.items()):
    print(f'   {ap}  R={R:3d}m  RMSE {rm:6.0f}  cov {cv:.0%}   model {MODEL[ap]:.0f}'
          f'   {"BEATS MODEL" if rm < MODEL[ap] else ""}')
print(f'\nVERDICT: {"GO" if n_pass >= N_BAR else "NO-GO"}')
if best:
    (ROOT/'reports'/f'gate_census_{A.day}.json').write_text(json.dumps(
        {k: {'R': v[0], 'rmse': v[1], 'cov': v[2]} for k, v in best.items()}, indent=2))
