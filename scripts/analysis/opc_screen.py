"""OPC: fixed month-swap residual screen. No production fit or submission."""
from pathlib import Path
import argparse
import hashlib
import json
import resource
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/opc_screen_2026_09_17.json"
PREDS = ROOT / "reports/opc_predictions_2026_09_17.parquet"
PREREG = ROOT / "plans/PREREG_ordered_peer_clocks_opc_2026_09_17.md"
ID = "MVT_ID_mvt"
WEIGHT = 339551 / 344841
OWN_NUM = ['proxy', 'sp', 'eobt_p', 'lobt_p', 'iobt_p', 'aobt_sched', 'eobt_sched',
           'aobt_eobt', 'lobt_sched', 'eobt_lobt', 'iobt_lobt', 'airborne3', 'airborne1',
           'arvt_diff', 'aobt_sec', 'mvt_sec', 'sched_sec', 'hr', 'tmin', 'dow', 'doy',
           'actype_null', 'f_callsign', 'f_ades', 'f_type', 'f_rule', 'n_push', 'rwy_b30',
           'apt_b30', 'arr_b30', 'sched_prev60', 'sched_next60', 'sched_day', 'rwy_rate',
           'dep_dur', 'rwy_dur', 'arr_dur', 'arr_ob60_taxiin', 'arr_ob60_cdiff',
           'arr_ob180_cdiff', 'prev_arr_gap', 'prev_arr_taxiin', 'prev_arr_cdiff', 'prev_dep_gap',
           'q_apt_at_push', 'q_rwy_at_push', 'q_pushed_after_me', 'q_rwy_tko_in_taxi',
           'q_rwy_push_pre20', 'q_dep_tko_sym15', 'q_rwy_tko_sym10', 'q_arr_taxiing_at_push',
           'q_arr_taxiin_sym30_push', 'q_arr_taxiin_sym30_tko', 'q_rwy_ambient_proxy',
           'q_next_arr_onblock_gap', 'o_dev_mvt', 'o_dev_flt', 'o_n_line']
OWN_CAT = ['ap', 'ap_stand', 'ap_rwy', 'ap_ades', 'airline', 'AIRCRAFT_OPERATOR_flt',
           'AIRCRAFT_TYPE_mvt', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt', 'FLIGHT_TYPE_flt', 'stand_pref']
RAW_COLS = [ID, 'PHASE_mvt', 'ADEP_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt',
            'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt', 'LOBT_flt']
INPUTS = []


def read(rel, columns=None, **kw):
    path = ROOT / rel
    INPUTS.append(path)
    return pd.read_parquet(path, columns=columns, **kw)


def joined(left, right, key):
    d = left.merge(right, on=key, how='left', validate='one_to_one', indicator=True)
    assert len(d) == len(left) and d._merge.eq('both').all()
    return d.drop(columns='_merge')


def peer_features(raw):
    """Reads only label-free columns. Selection excludes simultaneous departures."""
    r = raw.loc[raw.PHASE_mvt.eq('DEP'), RAW_COLS].copy().reset_index(drop=True)
    assert r[ID].is_unique
    times = {}
    epoch = pd.Timestamp('1970-01-01', tz='UTC')
    for c in ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt', 'LOBT_flt']:
        times[c] = (pd.to_datetime(r[c], utc=True).dt.floor('s') - epoch).dt.total_seconds().to_numpy()
    t = times['MVT_TIME_UTC_mvt']
    assert np.isfinite(t).all()
    proxy = t - times['AOBT_3_flt']
    values = {'proxy': proxy, 'sp': t - times['SCHED_TIME_UTC_mvt'],
              'aobt_eobt': times['AOBT_3_flt'] - times['EOBT_1_flt'],
              'eobt_lobt': times['EOBT_1_flt'] - times['LOBT_flt']}
    month = pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m')
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 10800)
    stand = r.STAND_mvt.astype('string')
    out = {ID: r[ID].to_numpy()}
    for scope, cols in [('ap', ['ADEP_mvt']), ('rwy', ['ADEP_mvt', 'RUNWAY_mvt'])]:
        keys = r[cols].copy()
        keys['month_key'] = month
        for direction in [-1, 1]:
            for rank in range(1, 5):
                prefix = f'opc_{scope}_{"prev" if direction < 0 else "next"}{rank}_'
                for field in ['gap', 'proxy', 'sp', 'aobt_eobt', 'eobt_lobt', 'same_stand']:
                    out[prefix + field] = np.full(len(r), np.nan, dtype=np.float32)
        for _, ix in keys.groupby(list(keys.columns), observed=True).indices.items():
            pool = ix[valid[ix]]
            order = pool[np.argsort(t[pool], kind='stable')]
            if not len(order):
                continue
            pt = t[order]
            left = np.searchsorted(pt, t[ix], side='left')
            right = np.searchsorted(pt, t[ix], side='right')
            for direction in [-1, 1]:
                for rank in range(1, 5):
                    pos = left - rank if direction < 0 else right + rank - 1
                    exists = (pos >= 0) & (pos < len(order))
                    target = ix[exists]
                    source = order[pos[exists]]
                    gap = t[source] - t[target]
                    keep = np.abs(gap) <= 5400
                    target, source, gap = target[keep], source[keep], gap[keep]
                    assert (direction * gap > 0).all()
                    assert (r[ID].to_numpy()[target] != r[ID].to_numpy()[source]).all()
                    prefix = f'opc_{scope}_{"prev" if direction < 0 else "next"}{rank}_'
                    out[prefix + 'gap'][target] = gap
                    for field, array in values.items():
                        out[prefix + field][target] = array[source]
                    same = (stand.iloc[target].reset_index(drop=True) == stand.iloc[source].reset_index(drop=True)).astype('Float32')
                    out[prefix + 'same_stand'][target] = same.to_numpy(dtype=np.float32, na_value=np.nan)
    return pd.DataFrame(out)


def load_data():
    s = read('data/adsb/v2/v13_matched_oof.parquet',
             ['row', ID, 'ap', 'month', 'day', 'fold', 'y', 'proxy', 'F', 'v13_delta', 'taxi_v13_delta'])
    h = read('data/cache_stand/catboost_native_input.parquet',
             list(dict.fromkeys(['row', ID, 'holdout', 'score_y', 'score_proxy', 'month'] + OWN_NUM + OWN_CAT)),
             filters=[('holdout', '=', True)])
    cp = ROOT / 'data/cache_stand/catboost_native_delta.npy'
    INPUTS.append(cp)
    h['C'] = np.load(cp)
    overlap = (set(h.columns) & set(s.columns)) - {'row'}
    h = h.rename(columns={c: f'native_{c}' for c in overlap})
    s = joined(s, h.drop(columns='holdout'), 'row')
    for c in [ID, 'ap', 'month', 'proxy']:
        assert np.array_equal(s[c], s[f'native_{c}'])
    assert np.array_equal(s.y, s.score_y) and np.array_equal(s.proxy, s.score_proxy)
    fw = read('data/cache_stand/fold_preds_queue_order_weather.parquet', ['row', 'y', 'proxy', 'treatment'])
    fw = fw.rename(columns={'y': 'fw_y', 'proxy': 'fw_proxy', 'treatment': 'FW'})
    s = joined(s, fw, 'row')
    assert np.array_equal(s.y, s.fw_y) and np.array_equal(s.proxy, s.fw_proxy)
    s = s[np.isfinite(s.fold)].copy().reset_index(drop=True)
    pop = s.v13_delta.eq(s.F).to_numpy()
    assert len(s) == 334132 and pop.sum() == 161956
    s['taxi_C'] = np.maximum(s.proxy - s.C, 1)
    s['taxi_FW'] = np.maximum(s.proxy - s.FW, 1)
    s['inc'] = np.where(pop, .5 * (s.taxi_C + s.taxi_FW), s.taxi_v13_delta)
    rmse = float(np.sqrt(np.mean((s.y - s.inc) ** 2)))
    assert abs(rmse - 208.31040447352203) < 1e-8
    raw = pd.concat([read(f'data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet', RAW_COLS) for m in [1, 7]], ignore_index=True)
    peers = peer_features(raw)
    s = joined(s, peers, ID)
    return s, pop


def bootstrap_stats(d, base, cand):
    y = d.y.to_numpy(float)
    gain = (y - base) ** 2 - (y - cand) ** 2
    frame = pd.DataFrame({'gain': gain, 'day': d.day.to_numpy(), 'month': d.month.to_numpy()})
    daily = frame.groupby(['month', 'day'], sort=True).agg(gain=('gain', 'sum'), n=('gain', 'size')).reset_index()
    rng = np.random.default_rng(0)
    vals = np.zeros(2000)
    ns = np.zeros(2000)
    monthly = {}
    for m in [1, 7]:
        a = daily[daily.month.eq(m)]
        idx = rng.integers(0, len(a), size=(2000, len(a)))
        sums = a.gain.to_numpy()[idx].sum(axis=1)
        counts = a.n.to_numpy()[idx].sum(axis=1)
        vals += sums
        ns += counts
        monthly[str(m)] = {'gain_weighted_mse': float(WEIGHT * a.gain.sum() / a.n.sum()),
                           'ci95': np.quantile(WEIGHT * sums / counts, [.025, .975]).tolist()}
    move = cand - base
    grouped = pd.DataFrame({'gain': gain, 'ap': d.ap.to_numpy(), 'day': d.day.to_numpy()}).groupby(['ap', 'day']).gain.sum()
    return {'gain_weighted_mse': float(WEIGHT * gain.mean()),
            'gain_rmse_s': float(np.sqrt(np.mean((y - base)**2)) - np.sqrt(np.mean((y - cand)**2))),
            'ci95': np.quantile(WEIGHT * vals / ns, [.025, .975]).tolist(), 'months': monthly,
            'changed_rows': int((move != 0).sum()), 'move_abs_max_s': float(np.abs(move).max()),
            'squared_move_share_gt300': float((move[np.abs(move) > 300]**2).sum() / max((move**2).sum(), 1e-30)),
            'largest_positive_row_gain_weighted_mse': float(WEIGHT * gain.max() / len(d)),
            'top3_airport_day_gain_weighted_mse': float(WEIGHT * grouped.nlargest(3).sum() / len(d)),
            'by_airport_gain_weighted_mse': {str(ap): float(WEIGHT * gain[d.ap.eq(ap).to_numpy()].sum() / len(d)) for ap in sorted(d.ap.unique())}}


def run(smoke=False):
    import lightgbm as lgb
    started = time.monotonic()
    if not smoke and (OUT.exists() or PREDS.exists()):
        raise FileExistsError('OPC result already exists')
    s, pop = load_data()
    peer_cols = [c for c in s if c.startswith('opc_')]
    assert len(peer_cols) == 96
    own_cols = OWN_NUM + OWN_CAT + ['inc', 'taxi_C', 'taxi_FW']
    X = s.loc[pop, own_cols + peer_cols].copy().reset_index(drop=True)
    for c in OWN_CAT:
        X[c] = X[c].astype('category')
    for c in set(X.columns) - set(OWN_CAT):
        X[c] = X[c].astype(np.float32)
    month = s.loc[pop, 'month'].to_numpy()
    airport = s.loc[pop, 'ap'].to_numpy()
    target = (s.y - s.inc).to_numpy()[pop]
    shuffled = X.copy()
    rng = np.random.default_rng(1710)
    for ap in sorted(set(airport)):
        for m in [1, 7]:
            ix = np.flatnonzero((airport == ap) & (month == m))
            shuffled.loc[ix, peer_cols] = X.loc[rng.permutation(ix), peer_cols].to_numpy()
    params = dict(objective='regression', learning_rate=.03, num_leaves=15, min_data_in_leaf=500,
                  lambda_l2=20, feature_fraction=1, bagging_fraction=1, max_bin=63,
                  boost_from_average=False, seed=1709, deterministic=True, force_row_wise=True,
                  num_threads=2, verbosity=-1)
    predictions = {}
    model_details = []
    for name, features in [('OWN', X[own_cols]), ('PEER', X), ('SHUFFLE', shuffled)]:
        pred = np.full(len(X), np.nan)
        for held in [1, 7]:
            tr, te = month != held, month == held
            if smoke:
                tr &= np.arange(len(X)) % 20 == 0
                te &= np.arange(len(X)) % 20 == 0
            t0 = time.monotonic()
            fit = lgb.Dataset(features.loc[tr], label=target[tr], categorical_feature=OWN_CAT, free_raw_data=True)
            model = lgb.train(params, fit, num_boost_round=3 if smoke else 300)
            pred[te] = model.predict(features.loc[te], num_threads=2)
            elapsed = time.monotonic() - t0
            model_details.append({'arm': name, 'held_month': held, 'train_n': int(tr.sum()), 'test_n': int(te.sum()),
                                  'seconds': elapsed, 'n_features': len(features.columns),
                                  'peer_split_gain': float(sum(g for f, g in zip(model.feature_name(), model.feature_importance('gain')) if f.startswith('opc_')))})
            print(json.dumps({k: v for k, v in model_details[-1].items() if k != 'peer_split_gain'}), flush=True)
            del model, fit
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
            if rss >= 3.0 or time.monotonic() - started > 280:
                raise RuntimeError(f'OPC bounded resource scope exceeded: peak {rss:.3f} GiB, elapsed {time.monotonic()-started:.1f}s')
        if not smoke:
            assert np.isfinite(pred).all()
            candidate = s.inc.to_numpy().copy()
            candidate[pop] = np.maximum(candidate[pop] + .5 * pred, 1)
            predictions[name] = candidate
    if smoke:
        print(json.dumps({'smoke': 'passed; scores not inspected', 'seconds': time.monotonic()-started,
                          'peak_rss_GiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3}), flush=True)
        return
    scores = {f'{name}_vs_inc': bootstrap_stats(s, s.inc.to_numpy(), pred) for name, pred in predictions.items()}
    for name in ['OWN', 'SHUFFLE']:
        scores[f'PEER_vs_{name}'] = bootstrap_stats(s, predictions[name], predictions['PEER'])
    p = scores['PEER_vs_inc']
    passes = p['gain_weighted_mse'] >= 100 and p['ci95'][0] > 0 and all(v['ci95'][0] > 0 for v in p['months'].values())
    passes &= scores['PEER_vs_OWN']['ci95'][0] > 0 and scores['SHUFFLE_vs_inc']['ci95'][0] <= 0
    result = {'tag': 'OPC', 'status': 'PROMISING DEVELOPMENT LEVER' if passes else 'NO LEVER ESTABLISHED BY THIS SCREEN',
              'scope': '2025 development screen; fixed month swap and composed incumbent; not a measured 2026 board gain',
              'n': len(s), 'n_pop': int(pop.sum()), 'inc_rmse': float(np.sqrt(np.mean((s.y-s.inc)**2))),
              'scores': scores, 'model_details': model_details,
              'seconds': time.monotonic()-started, 'peak_rss_GiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,
              'peer_missing_fraction': float(X[peer_cols].isna().to_numpy().mean()),
              'prereg_sha256': hashlib.sha256(PREREG.read_bytes()).hexdigest(), 'input_sha256': {}}
    for path in dict.fromkeys(INPUTS):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1<<20), b''):
                h.update(block)
        result['input_sha256'][str(path.relative_to(ROOT))] = h.hexdigest()
    detail = s[[ID, 'row', 'month', 'day', 'ap', 'y', 'inc']].copy()
    for name, pred in predictions.items():
        detail[name] = pred
    detail.to_parquet(PREDS, index=False)
    with OUT.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'scores': scores, 'seconds': result['seconds'],
                      'peak_rss_GiB': result['peak_rss_GiB']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args().smoke)
