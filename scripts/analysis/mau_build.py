"""MAU: a fixed training-data transfer test on the actual unmatched composition."""
from __future__ import annotations

import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
import ucp_build as U

uc, ca = U.uc, U.ca
PREREG = ROOT / 'plans/PREREG_mau_masked_training_transfer_2026_09_20.md'
HASHLOG = ROOT / 'reports/mau_prereg_hashes.log'
REPORT = ROOT / 'reports/mau_report.json'
PREDS = ROOT / 'reports/mau_test_predictions.parquet'
TRAIN = ROOT / 'reports/mau_training_rows.parquet'
FEATURES = list(uc.CongestionRegressor.NUMERIC) + list(uc.bs.NF_ENCODED)
RAW_COLS = ['PHASE_mvt', 'MVT_ID_mvt', 'ADEP_mvt', 'RUNWAY_mvt',
            'STAND_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
            'BLOCK_TIME_UTC_mvt', 'TAXITIME_SEC_mvt', 'AOBT_3_flt']


def median_without_each(values):
    """Median after removing each entry, in original order; no row labels used."""
    values = np.asarray(values, float)
    n = len(values)
    if n < 2:
        return np.full(n, np.nan)
    order = np.argsort(values, kind='stable')
    v = values[order]
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    mid = n // 2
    if n % 2 == 0:
        return np.where(rank < mid, v[mid], v[mid-1])
    return np.where(rank < mid, (v[mid]+v[mid+1])/2,
                    np.where(rank > mid, (v[mid-1]+v[mid])/2,
                             (v[mid-1]+v[mid+1])/2))


def loo_witness(raw):
    """Vectorised exact leave-one-flight-out median in the original row order."""
    m = raw.PHASE_mvt.eq('DEP') & raw.AOBT_3_flt.notna() & raw.MVT_TIME_UTC_mvt.notna()
    w = pd.DataFrame({'idx': raw.index[m], 'ap': raw.loc[m, 'ADEP_mvt'].astype(str),
                      'hour': raw.loc[m, 'MVT_TIME_UTC_mvt'].dt.floor('h'),
                      'proxy': (raw.loc[m, 'MVT_TIME_UTC_mvt']-raw.loc[m, 'AOBT_3_flt']).dt.total_seconds()})
    w = w.sort_values(['ap', 'hour', 'proxy'], kind='stable').reset_index(drop=True)
    g = w.groupby(['ap', 'hour'], sort=False)
    n = g.proxy.transform('size').to_numpy(dtype=int)
    rank = g.cumcount().to_numpy(dtype=int)
    start = np.arange(len(w)) - rank
    mid = n // 2
    v = w.proxy.to_numpy(float)
    lo = v[np.maximum(start+mid-1, 0)]
    center = v[start+mid]
    hi = v[np.minimum(start+mid+1, len(v)-1)]
    even = np.where(rank < mid, center, lo)
    odd = np.where(rank < mid, (center+hi)/2,
                   np.where(rank > mid, (lo+center)/2, (lo+hi)/2))
    med = np.where(n % 2 == 0, even, odd)
    med[n-1 < uc.MIN_N] = np.nan
    return pd.DataFrame({'hprox_med': med, 'hprox_n': n-1}, index=w.idx)


def construction_guards():
    rng = np.random.default_rng(0)
    for n in range(1, 30):
        v = rng.integers(-10, 11, n).astype(float)
        expected = np.array([np.median(np.delete(v, i)) if n > 1 else np.nan for i in range(n)])
        np.testing.assert_allclose(median_without_each(v), expected, equal_nan=True)
        times = pd.Series(pd.to_datetime(['2025-01-01T10:30:00Z'] * n))
        f = pd.DataFrame({'PHASE_mvt': 'DEP', 'ADEP_mvt': 'TEST',
                          'MVT_TIME_UTC_mvt': times, 'AOBT_3_flt': times-pd.to_timedelta(v, unit='s')})
        actual = loo_witness(f).reindex(f.index)
        if n-1 < uc.MIN_N:
            expected[:] = np.nan
        np.testing.assert_allclose(actual.hprox_med, expected, equal_nan=True)
        assert (actual.hprox_n == n-1).all()
    # Different hours and airports must not mix.
    f = pd.concat([f.assign(ADEP_mvt='ONE'), f.assign(ADEP_mvt='TWO'),
                   f.assign(MVT_TIME_UTC_mvt=f.MVT_TIME_UTC_mvt+pd.Timedelta(hours=1),
                            AOBT_3_flt=f.AOBT_3_flt+pd.Timedelta(hours=1))], ignore_index=True)
    got = loo_witness(f).reindex(f.index)
    np.testing.assert_allclose(got.hprox_med, np.tile(expected, 3))
    assert (got.hprox_n == n-1).all()


def extra_rows(raw):
    witness = loo_witness(raw)
    d = raw.loc[raw.PHASE_mvt.eq('DEP') & raw.AOBT_3_flt.notna()].copy()
    d['hprox_med'] = witness.hprox_med.reindex(d.index).to_numpy()
    d['hprox_n'] = witness.hprox_n.reindex(d.index).to_numpy()
    d = uc.bs.admissible(d)
    ids = d.MVT_ID_mvt.to_numpy(float)
    assert np.isfinite(ids).all() and (ids == np.rint(ids)).all()
    d = d.loc[(ids.astype('int64') % 20 == 0) & ~uc.bs.schedule_fill(d)].copy()
    t = d.MVT_TIME_UTC_mvt
    d['sp'] = (t-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    d['dayoff'] = np.clip((t.dt.normalize()-d.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days, 0, 1)
    d['hr'] = t.dt.hour
    d['month'] = t.dt.month
    d['airline'] = d.FLIGHT_mvt.astype(str).str[:3]
    d['stand_pref'] = d.STAND_mvt.astype(str).str[:2]
    d['AIRCRAFT_OPERATOR_flt'] = np.nan
    d['y'] = d.TAXITIME_SEC_mvt
    # Restrict columns structurally; hidden own-row NM clocks cannot enter the model.
    return d[['MVT_ID_mvt', 'month', 'y', *FEATURES]].copy()


def stopping_split(n, seed):
    assert sklearn.__version__ == '1.8.0'
    rs = np.random.RandomState(seed).randint(np.iinfo(np.uint32).max, dtype='u8')
    return train_test_split(np.arange(n), test_size=0.1, stratify=None, random_state=rs)


def main():
    started = time.monotonic()
    signal.alarm(300)
    if any(p.exists() for p in (REPORT, PREDS, TRAIN)):
        raise FileExistsError('MAU outputs exist; no silent rerun')
    reg = U.sha(PREREG)
    assert f'sha256={reg}' in HASHLOG.read_text()
    construction_guards()
    frame = pd.read_parquet(ROOT / 'data/cache_rome/unm.parquet')
    saved = pd.read_parquet(ROOT / 'data/cache_stand/unm_physics_lomo.parquet')
    assert frame.MVT_ID_mvt.is_unique and saved.MVT_ID_mvt.is_unique
    ref = saved.set_index('MVT_ID_mvt').reindex(frame.MVT_ID_mvt).reset_index()
    for c in ('y', 'sp', 'month'):
        np.testing.assert_array_equal(frame[c].to_numpy(), ref[c].to_numpy())
    assert frame.AIRCRAFT_OPERATOR_flt.isna().all()
    for c in uc.WITNESS:
        frame[c] = ref[c].to_numpy()
    sources = {name: U.sha(ROOT / name) for name in (
        'data/cache_rome/unm.parquet', 'data/cache_stand/unm_physics_lomo.parquet')}
    parts, anomalies = [], []
    for path in sorted((ROOT / 'data/raw').glob('training_2025-*.parquet')):
        sources[str(path.relative_to(ROOT))] = U.sha(path)
        raw = pd.read_parquet(path, columns=RAW_COLS)
        parts.append(extra_rows(raw))
        if any(path.name.startswith(f'training_2025-{m:02d}-01_') for m in U.MONTHS):
            anomalies.append(ca.day_anomaly(raw))
        print(json.dumps({'loaded': path.name, 'sample_n': len(parts[-1]),
                          'seconds': time.monotonic()-started}), flush=True)
        del raw
    assert len(parts) == 12
    extra = pd.concat(parts, ignore_index=True)
    assert extra.MVT_ID_mvt.is_unique
    assert not extra.MVT_ID_mvt.isin(frame.MVT_ID_mvt).any()
    extra.to_parquet(TRAIN, index=False)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3
    assert peak < 0.8, ('pre-fit RSS limit', peak)
    anomaly = pd.concat(anomalies).set_index(['ADEP_mvt', 'date']).A
    frame['date'] = frame.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    frame['A'] = anomaly.reindex(pd.MultiIndex.from_arrays([frame.ADEP_mvt, frame.date])).to_numpy()
    scope = frame.month.isin(U.MONTHS) & frame.ADEP_mvt.ne('LIRF')
    instrument = frame.loc[scope, ['y', 'month']].copy()
    instrument['day'] = frame.loc[scope, 'MVT_TIME_UTC_mvt'].dt.dayofyear.to_numpy()
    instrument['control'] = ca.anchored_values(np.rint(ref.loc[scope, 'S1C']), frame.loc[scope, 'A'])
    identity, oracle = U.score(instrument, instrument.control), U.score(instrument, instrument.y)
    assert identity['gain'] == 0 and identity['ci95'] == [0.0, 0.0]
    assert U.decide(oracle) == 'DEVELOPMENT IMPROVEMENT'
    records, guards, fits = [], [], []
    for m in U.MONTHS:
        test = frame.month.eq(m).to_numpy()
        train = frame.loc[~test & ~uc.bs.schedule_fill(frame)]
        ex = extra.loc[extra.month.ne(m)]
        te, rr = frame.loc[test], ref.loc[test]
        assert not train.month.eq(m).any() and not ex.month.eq(m).any()
        stacks = {'control': [], 'candidate': []}
        for seed in U.SEEDS:
            ctrl = uc.fit_congestion_regressor(train, seed)
            cp = ctrl.predict(te)
            x = ctrl.design(train)
            yy = np.clip(train.y.to_numpy(float), -3000, 3000)
            fi, va = stopping_split(len(train), seed)
            replay = HGR(random_state=seed, early_stopping=True, **uc.bs.NF_PARAMS)
            replay.fit(x.iloc[fi], yy[fi], X_val=x.iloc[va], y_val=yy[va])
            replay_diff = float(np.max(np.abs(replay.predict(ctrl.design(te))-cp)))
            assert replay_diff < 1e-6, (m, seed, 'explicit split reproduction', replay_diff)
            assert set(ctrl.design(ex).columns) == set(ctrl.design(train).columns)
            weight = 0.25*len(fi)/len(ex)
            tx = pd.concat([x.iloc[fi], ctrl.design(ex)], ignore_index=True)
            ty = np.r_[yy[fi], np.clip(ex.y.to_numpy(float), -3000, 3000)]
            sw = np.r_[np.ones(len(fi)), np.full(len(ex), weight)]
            assert np.isclose(sw[len(fi):].sum()/sw[:len(fi)].sum(), 0.25)
            model = HGR(random_state=seed, early_stopping=True, **uc.bs.NF_PARAMS)
            model.fit(tx, ty, sample_weight=sw, X_val=x.iloc[va], y_val=yy[va],
                      sample_weight_val=np.ones(len(va)))
            stacks['control'].append(cp)
            stacks['candidate'].append(model.predict(ctrl.design(te)))
            fits.append({'month': m, 'seed': seed, 'unmatched_fit_n': len(fi),
                         'validation_n': len(va), 'extra_n': len(ex), 'matched_weight': weight,
                         'extra_mass_ratio': float(sw[len(fi):].sum()/len(fi)),
                         'control_iter': int(ctrl.model.n_iter_), 'candidate_iter': int(model.n_iter_),
                         'replay_max_abs_diff': replay_diff})
            del ctrl, replay, model, tx, ty, sw, x
            print(json.dumps({'month': m, 'seed': seed, 'seconds': time.monotonic()-started}), flush=True)
        out = te[['MVT_ID_mvt', 'ADEP_mvt', 'date', 'month', 'y', 'sp', 'A']].copy()
        out['day'] = te.MVT_TIME_UTC_mvt.dt.dayofyear.to_numpy()
        for c in ('p_hat', 'nf_cells', 'routed_nf_fit'):
            out[c] = rr[c].to_numpy()
        for arm in stacks:
            stack = np.asarray(stacks[arm])
            body = stack.mean(axis=0)
            mix = uc.bs.mixture(rr.p_hat, rr.sp, uc.bs.nf_hybrid(rr.nf_cells, body))
            if arm == 'control':
                bd = float(np.max(np.abs(body-rr.nf_fit_c.to_numpy())))
                md = float(np.max(np.abs(mix-rr.S1C.to_numpy())))
                assert bd < 1e-6 and md < 1e-6, (m, bd, md)
                guards.append({'month': m, 'body_max_abs_diff': bd, 'mixture_max_abs_diff': md})
            out[f'body_{arm}'] = body
            out[arm] = ca.anchored_values(np.rint(mix), te.A)
            for j, seed in enumerate(U.SEEDS):
                z = uc.bs.mixture(rr.p_hat, rr.sp, uc.bs.nf_hybrid(rr.nf_cells, stack[j]))
                out[f'{arm}_seed{seed}'] = ca.anchored_values(np.rint(z), te.A)
        out = out.loc[out.ADEP_mvt.ne('LIRF')].copy()
        assert (out.loc[~out.routed_nf_fit, 'control'] == out.loc[~out.routed_nf_fit, 'candidate']).all()
        out['gain_sse'] = (out.y-out.control)**2-(out.y-out.candidate)**2
        records.append(out)
    result = pd.concat(records, ignore_index=True)
    assert len(result) == 4924 and result.MVT_ID_mvt.is_unique
    identity, oracle = U.score(result, result.control), U.score(result, result.y)
    assert identity['gain'] == 0 and identity['ci95'] == [0.0, 0.0]
    assert U.decide(oracle) == 'DEVELOPMENT IMPROVEMENT'
    result.to_parquet(PREDS, index=False)
    reread = pd.read_parquet(PREDS)
    pd.testing.assert_frame_equal(result, reread)
    stats = U.score(reread, reread.candidate)
    report = {'tag': 'MAU', 'verdict': U.decide(stats), 'stats': stats, 'fits': fits,
              'guards': guards, 'identity': identity, 'oracle': oracle,
              'seed_gains': {str(s): U.pbl.board_gain(reread.y, reread[f'control_seed{s}'],
                              reread[f'candidate_seed{s}']) for s in U.SEEDS},
              'airport_gains': (reread.groupby('ADEP_mvt').gain_sse.sum()/344841).to_dict(),
              'scope': '2025 Jan/Jul LOMO non-Rome unmatched S1C plus storm. SSE/344841; not board forecast.',
              'prereg_sha256': reg, 'runner_sha256': U.sha(__file__), 'input_hashes': sources,
              'pred_sha256': U.sha(PREDS), 'train_sha256': U.sha(TRAIN),
              'seconds': time.monotonic()-started,
              'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3}
    with REPORT.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    signal.alarm(0)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
