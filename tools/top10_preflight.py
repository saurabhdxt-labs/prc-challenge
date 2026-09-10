"""Audit the next-run assumptions without training or changing any prediction.

OMP_NUM_THREADS=1 python3.11 -B tools/top10_preflight.py
Output: reports/top10_preflight.json. Scores use the user's September 10 snapshot.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WM = 339551 / 344841


def main():
    path = ROOT / 'data/cache_stand/fold_preds_fillhead.parquet'
    m = pd.read_parquet(path, columns=['ap', 'y', 'sp', 'proxy', 'baseline', 'treatment', 'p'])
    y = m.y.to_numpy()
    fill = (m.y - m.sp).abs().le(60).to_numpy()
    p = m.p.to_numpy()
    # This record stores delta, including a recovered/floored treatment in delta form.
    old = np.maximum(m.proxy.to_numpy() - m.baseline.to_numpy(), 1)
    new = np.maximum(m.proxy.to_numpy() - m.treatment.to_numpy(), 1)
    record = json.loads((ROOT / 'reports/lgbm_fold_fillhead.json').read_text())
    for name, pred in [('baseline', old), ('treatment', new)]:
        np.testing.assert_allclose(np.sqrt(np.mean((y-pred)**2)),
                                   record['arms'][name]['rmse'], rtol=0, atol=1e-8)
    gain = (y-old)**2 - (y-new)**2
    w = WM / len(m)
    per_airport = []
    for ap in sorted(m.ap.unique()):
        rows = m.ap.eq(ap).to_numpy()
        per_airport.append(dict(airport=ap,
            fill_gain_mse=float(w*gain[rows & fill].sum()),
            nonfill_gain_mse=float(w*gain[rows & ~fill].sum()),
            net_gain_mse=float(w*gain[rows].sum())))
    fill_gain = float(w*gain[fill].sum())
    rome_gain = next(r['fill_gain_mse'] for r in per_airport if r['airport'] == 'LIRF')
    reliability = []
    for indices in np.array_split(np.argsort(p, kind='stable'), 10):
        reliability.append(dict(n=len(indices), mean_p=float(p[indices].mean()),
                                 actual_fill_rate=float(fill[indices].mean())))
    keys = []
    for path in sorted((ROOT / 'data/cache_stand').glob('training_2025-*.parquet')) + [
            ROOT / 'data/cache_stand/ranking.parquet']:
        d = pd.read_parquet(path, columns=['ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'ars'])
        parts = [d[c].fillna('NA').astype(str) for c in ['ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt']]
        expected = parts[0] + '|' + parts[1] + '|' + parts[2]
        same = d.ars.eq(expected)
        keys.append(dict(file=path.name, n=len(d), ars_already_airport_scoped=int(same.sum())))
    q = pd.read_parquet(ROOT / 'data/cache_stand/fold_preds_queue.parquet',
                        columns=['ap', 'y', 'proxy', 'treatment'])
    e = q.y - np.maximum(q.proxy-q.treatment, 1)
    matched = float(np.sqrt(np.mean(e**2)))
    rome = q.ap.eq('LIRF')
    rmse_rome = float(np.sqrt(np.mean(e[rome]**2)))
    ours = 288.1406
    gap = ours**2 - 275.9373**2
    scenarios = [dict(mse_gain=g, board_rmse_if_gain_transfers=float(np.sqrt(ours**2-g)),
                      matched_rmse_if_only_matched_changes=float(np.sqrt(matched**2-g/WM)))
                 for g in [232., 322., 1782., gap, 9000., 10000.]]
    output = dict(run_utc=datetime.now(timezone.utc).isoformat(),
        scope='Read-only diagnostics of existing predictions. No new model, gate, or calibration fit.',
        score_source='User September 10 snapshot; live leaderboard refresh unavailable.',
        fill_branch_gain_mse=fill_gain,
        rome_share_of_fill_branch_gain=rome_gain/fill_gain,
        old_fill_blend_by_airport=per_airport,
        probability_diagnostics=dict(mean_p=float(p.mean()), prevalence=float(fill.mean()),
            reliability_deciles=reliability,
            caveat='Pooled deciles do not establish calibration by airport or error severity.'),
        ars_census=keys, budget_scenarios=scenarios,
        scenario_caveat='Planning arithmetic conditional on equal MSE transfer; not a leaderboard prediction.',
        rome_targets=[dict(rmse=t, matched_fold_mse_gain=float(WM*rome.mean()*(rmse_rome**2-t**2)))
                      for t in [350., 325., 300., 280.]],
        rome_target_caveat='Desired improvements, not demonstrated or necessarily achievable.')
    out = ROOT / 'reports/top10_preflight.json'
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(f'Baseline/treatment scales reproduced. Rome carries {100*rome_gain/fill_gain:.2f}% of fill-branch gain.')
    print(f'Airport-scoped ars: {sum(r["ars_already_airport_scoped"] for r in keys):,} / '
          f'{sum(r["n"] for r in keys):,} rows across {len(keys)} caches.')
    print(f'Snapshot top-ten gap: {gap:,.1f} MSE. Results: {out}')


if __name__ == '__main__':
    main()
