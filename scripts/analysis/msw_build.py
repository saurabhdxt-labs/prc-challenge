"""MSW: sweep MAU's two hand-set constants -- extra-row sample rate (m) x extra-row mass ratio (r).

Registration: plans/PREREG_msw_mau_sweep_2026_09_21.md (sha256 e338746478cf54ce5ee89cfb5b143edb585700ce67d23f188721213a41755473,
hashed before measuring) plus its appended pre-measurement operationalisation A1.

Exactly two things vary from MAU (`mau_build.py`, another lane's validated file, imported and NEVER edited):
the modulus in `MVT_ID % m == 0` (mau_build.py:107) and the mass ratio (mau_build.py:192). Everything else --
the ten features, the 3,000 s cap, fill probabilities, cell routing, storm anchor, seeds, the unmatched-only
stopping split -- is MAU's own code or a line-for-line copy of it, and the incumbent cell (20, 0.25) must
reproduce MAU's persisted predictions EXACTLY or the sweep is void.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / 'scripts', ROOT / 'scripts/analysis'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import mau_build as MB      # noqa: E402
import arscat_build as A    # noqa: E402  (assert_day_blocks_are_real: the BC-23 guard)

U, uc, ca, pbl = MB.U, MB.uc, MB.ca, MB.U.pbl
# The verbatim copy below resolves these names to MAU's OWN objects, not to copies of them.
loo_witness, FEATURES = MB.loo_witness, MB.FEATURES

PREREG = ROOT / 'plans/PREREG_msw_mau_sweep_2026_09_21.md'
REGISTERED_SHA = 'e338746478cf54ce5ee89cfb5b143edb585700ce67d23f188721213a41755473'
HASHLOG = ROOT / 'reports/msw_prereg_hashes.log'
REPORT = ROOT / 'reports/msw_report.json'
PREDS = ROOT / 'reports/msw_cell_predictions.parquet'

MODULI = (5, 10, 20, 40)
RATIOS = (0.15, 0.25, 0.40)
INCUMBENT = (20, 0.25)
RERUN = 'incumbent_rerun'
EXPECTED_EXTRA_AT_20 = 93_165
TARGET_CAP_S = 3000            # mau_build.py:185,194 -- unchanged
# The two LFPG label outliers (84,240 s and 58,206 s of "taxi"); audit_mau_mav_rcv_2026_09_20.py:35.
MONSTER_IDS = (182378789.0, 182377662.0)
ALARM_S = 300                  # the registration's 5-minute bar; a longer run needs the pre-run checks
RSS_LIMIT_GIB = 2.0            # the box has ~6 GB reclaimable and Phantom's aggregate spikes 6.2-6.6 GB


def grid():
    return [(m, r) for m in MODULI for r in RATIOS]


def cell_key(m, r):
    return f'm{m}_r{r:.2f}'


def extra_rows_at(raw, modulus):
    """`mau_build.extra_rows` with the sample modulus as a parameter. The body below the marker is a
    VERBATIM copy (tests/test_msw_build.py pins it to MAU's source line-for-line, `% 20` -> `% modulus`)."""
    if isinstance(modulus, bool) or not isinstance(modulus, (int, np.integer)) or modulus <= 0:
        raise ValueError(f'modulus must be a positive int, got {modulus!r}')
    # ---- verbatim mau_build.extra_rows body, modulus parameterised ----
    witness = loo_witness(raw)
    d = raw.loc[raw.PHASE_mvt.eq('DEP') & raw.AOBT_3_flt.notna()].copy()
    d['hprox_med'] = witness.hprox_med.reindex(d.index).to_numpy()
    d['hprox_n'] = witness.hprox_n.reindex(d.index).to_numpy()
    d = uc.bs.admissible(d)
    ids = d.MVT_ID_mvt.to_numpy(float)
    assert np.isfinite(ids).all() and (ids == np.rint(ids)).all()
    d = d.loc[(ids.astype('int64') % modulus == 0) & ~uc.bs.schedule_fill(d)].copy()
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


def _gate(got, want, what):
    try:
        pd.testing.assert_frame_equal(got, want, check_exact=True)
    except AssertionError as err:
        raise AssertionError(f'GATE FAILED ({what}): the sweep does not run.\n{err}') from None


def extras_for_file(raw):
    """Every modulus for one raw month, with the construction-fidelity GATE enforced at run time:
    m = 20 is MAU's own rows, m = 40 is MAU's rows filtered, m in {5, 10} contain MAU's rows exactly."""
    ref = MB.extra_rows(raw)
    out = {m: extra_rows_at(raw, m) for m in MODULI}
    _gate(out[20], ref, 'm=20 vs mau_build.extra_rows')
    for m in MODULI:
        ids = out[m].MVT_ID_mvt.astype('int64')
        if m > 20:
            _gate(out[m], ref.loc[ref.MVT_ID_mvt.astype('int64') % m == 0], f'm={m} nested in MAU')
        elif m < 20:
            _gate(out[m].loc[ids % 20 == 0], ref, f'm={m} contains MAU')
    return out


def extra_weight(n_fit, n_extra, ratio):
    """Per-row weight so the extra rows carry `ratio` of the unmatched fit mass. MAU's evaluation order
    (`0.25*len(fi)/len(ex)`), so at ratio 0.25 the float is bit-identical to MAU's."""
    if n_fit <= 0 or n_extra <= 0:
        raise ValueError(f'need non-empty fit and extra sets, got n_fit={n_fit}, n_extra={n_extra}')
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError(f'ratio must be finite and > 0, got {ratio!r}')
    return ratio*n_fit/n_extra


def fit_candidate(x_fit, y_fit, design_extra, y_extra, ratio, seed, x_val, y_val):
    """mau_build.py:192-199 with `ratio` in place of 0.25. `y_fit`/`y_val` arrive already capped (MAU's `yy`)."""
    if set(design_extra.columns) != set(x_fit.columns):
        raise ValueError('the extra design does not match the fit design columns')
    weight = extra_weight(len(x_fit), len(design_extra), ratio)
    tx = pd.concat([x_fit, design_extra], ignore_index=True)
    ty = np.r_[np.asarray(y_fit, float), np.clip(np.asarray(y_extra, float), -TARGET_CAP_S, TARGET_CAP_S)]
    sw = np.r_[np.ones(len(x_fit)), np.full(len(design_extra), weight)]
    mass = float(sw[len(x_fit):].sum()/sw[:len(x_fit)].sum())
    if not np.isclose(mass, ratio):
        raise AssertionError(f'extra mass ratio {mass} != registered {ratio}')
    model = MB.HGR(random_state=seed, early_stopping=True, **uc.bs.NF_PARAMS)
    model.fit(tx, ty, sample_weight=sw, X_val=x_val, y_val=y_val, sample_weight_val=np.ones(len(x_val)))
    return model, {'weight': weight, 'mass_ratio': mass, 'iter': int(model.n_iter_)}


def assert_rerun_identical(a, b, what):
    """The determinism null: the same cell fitted twice must agree to the last bit, or the sweep is VOID."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
        n = int((a != b).sum()) if a.shape == b.shape else -1
        raise AssertionError(f'VOID: {what} rerun differs from the first fit on {n} values; '
                             f'the harness is not deterministic')


def delta_stats(y, incumbent, cell, day, month):
    """(SSE_incumbent - SSE_cell)/344,841, day-blocked bootstrap (2,000 draws, seed 0), BC-23 guarded."""
    A.assert_day_blocks_are_real(np.asarray(day), np.asarray(month))
    return pbl.day_block_bootstrap(y, incumbent, cell, day, month)


def monster_delta(ids, y, incumbent, cell):
    """The part of a cell's delta carried by the two LFPG label outliers. A diagnostic, never a clause."""
    mask = np.isin(np.asarray(ids, float), MONSTER_IDS)
    y, incumbent, cell = (np.asarray(v, float)[mask] for v in (y, incumbent, cell))
    return pbl.board_gain(y, incumbent, cell)


def screen_clauses(st):
    """Registration clauses 1 and 2, on the Jan/Jul selection screen."""
    return {'c1_beats_incumbent': bool(st['gain'] > 0), 'c2_ci_excludes_0': bool(st['ci95'][0] > 0)}


def confirm_clause(st):
    """Clauses 3 and 4, 'remains ahead', as pinned in A1: ahead on the point AND the CI excludes 0.
    A CI spanning 0 is ambiguous on sign, and ambiguous = NOT WORKING (registration section 2)."""
    return bool(st['gain'] > 0 and st['ci95'][0] > 0)


def pick_winner(screens):
    """Among non-incumbent cells passing clauses 1 AND 2, the largest delta; None if none passes."""
    inc = cell_key(*INCUMBENT)
    passing = sorted(k for k, st in screens.items() if k != inc and all(screen_clauses(st).values()))
    if not passing:
        return None
    return max(passing, key=lambda k: screens[k]['gain'])


def verdict(winner, forward, lfpg_excluded):
    if winner is None:
        return 'NOT WORKING'
    if forward is None or lfpg_excluded is None:
        raise ValueError('a winner needs both section M16 legs (forward-chronological AND LFPG-excluded)')
    return 'WORKING' if confirm_clause(forward) and confirm_clause(lfpg_excluded) else 'NOT WORKING'


def compose(rr, te, body):
    """mau_build.py:216/223/225-226: body -> shipped mixture -> storm anchor."""
    mix = uc.bs.mixture(rr.p_hat, rr.sp, uc.bs.nf_hybrid(rr.nf_cells, body))
    return ca.anchored_values(np.rint(mix), te.A)


def registration_sha():
    """The registered text is everything above the first appended hash line; it must still be e338...
    Appendices start '\\n---\\nsha256 of everything above this line' after the registration's final newline."""
    text = PREREG.read_bytes()
    marker = b'\n\n---\nsha256 of everything above this line'
    cut = text.index(marker) + 1
    return hashlib.sha256(text[:cut]).hexdigest(), U.sha(PREREG)


def _rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3


def main():
    started = time.monotonic()
    signal.alarm(ALARM_S)
    if any(p.exists() for p in (REPORT, PREDS)):
        raise FileExistsError('MSW outputs exist; no silent rerun')
    registered, current = registration_sha()
    assert registered == REGISTERED_SHA, ('registration text changed', registered)
    log = HASHLOG.read_text().split()
    assert REGISTERED_SHA in log and current in log, 'amended registration not hashed before measuring'
    MB.construction_guards()
    mau_report = json.loads(MB.REPORT.read_text())
    frame = pd.read_parquet(ROOT / 'data/cache_rome/unm.parquet')
    saved = pd.read_parquet(ROOT / 'data/cache_stand/unm_physics_lomo.parquet')
    assert frame.MVT_ID_mvt.is_unique and saved.MVT_ID_mvt.is_unique
    ref = saved.set_index('MVT_ID_mvt').reindex(frame.MVT_ID_mvt).reset_index()
    for c in ('y', 'sp', 'month'):
        np.testing.assert_array_equal(frame[c].to_numpy(), ref[c].to_numpy())
    assert frame.AIRCRAFT_OPERATOR_flt.isna().all()
    for c in uc.WITNESS:
        frame[c] = ref[c].to_numpy()
    sources = {}
    for name, h in mau_report['input_hashes'].items():
        sources[name] = U.sha(ROOT / name)
        assert sources[name] == h, ('input changed since MAU', name)
    parts, anomalies = {m: [] for m in MODULI}, []
    files = sorted((ROOT / 'data/raw').glob('training_2025-*.parquet'))
    assert len(files) == 12
    for path in files:
        raw = pd.read_parquet(path, columns=MB.RAW_COLS)
        for m, part in extras_for_file(raw).items():
            parts[m].append(part)
        if any(path.name.startswith(f'training_2025-{mm:02d}-01_') for mm in U.MONTHS):
            anomalies.append(ca.day_anomaly(raw))
        del raw
    extras = {m: pd.concat(parts[m], ignore_index=True) for m in MODULI}
    del parts
    for m, e in extras.items():
        assert e.MVT_ID_mvt.is_unique and not e.MVT_ID_mvt.isin(frame.MVT_ID_mvt).any(), m
    assert len(extras[20]) == EXPECTED_EXTRA_AT_20, len(extras[20])
    _gate(extras[20], pd.read_parquet(MB.TRAIN), 'm=20 vs MAU persisted training rows')
    print(json.dumps({'gate': 'PASS', 'extra_n': {m: len(e) for m, e in extras.items()},
                      'seconds': time.monotonic()-started, 'rss_gib': _rss_gib()}), flush=True)
    assert _rss_gib() < RSS_LIMIT_GIB, ('pre-fit RSS limit', _rss_gib())
    anomaly = pd.concat(anomalies).set_index(['ADEP_mvt', 'date']).A
    frame['date'] = frame.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    frame['A'] = anomaly.reindex(pd.MultiIndex.from_arrays([frame.ADEP_mvt, frame.date])).to_numpy()
    mau_preds = pd.read_parquet(MB.PREDS)
    inc_key = cell_key(*INCUMBENT)
    cells = {cell_key(m, r): (m, r) for m, r in grid()}
    order = [inc_key, RERUN, *[k for k in cells if k != inc_key]]     # incumbent first: fail fast
    params = {**cells, RERUN: INCUMBENT}
    records, fits = [], []
    for month in U.MONTHS:
        test = frame.month.eq(month).to_numpy()
        train = frame.loc[~test & ~uc.bs.schedule_fill(frame)]
        te, rr = frame.loc[test], ref.loc[test]
        exm = {m: extras[m].loc[extras[m].month.ne(month)] for m in MODULI}
        assert not train.month.eq(month).any() and not any(e.month.eq(month).any() for e in exm.values())
        keep = te.ADEP_mvt.ne('LIRF').to_numpy()
        mau_m = mau_preds.loc[mau_preds.month.eq(month)]
        np.testing.assert_array_equal(te.MVT_ID_mvt.to_numpy()[keep], mau_m.MVT_ID_mvt.to_numpy())
        stacks = {k: [] for k in ('control', *order)}
        for seed in U.SEEDS:
            ctrl = uc.fit_congestion_regressor(train, seed)
            cp = ctrl.predict(te)
            x = ctrl.design(train)
            yy = np.clip(train.y.to_numpy(float), -TARGET_CAP_S, TARGET_CAP_S)
            fi, va = MB.stopping_split(len(train), seed)
            replay = MB.HGR(random_state=seed, early_stopping=True, **uc.bs.NF_PARAMS)
            replay.fit(x.iloc[fi], yy[fi], X_val=x.iloc[va], y_val=yy[va])
            dte = ctrl.design(te)
            replay_diff = float(np.max(np.abs(replay.predict(dte)-cp)))
            assert replay_diff < 1e-6, (month, seed, 'explicit split reproduction', replay_diff)
            dex = {m: ctrl.design(exm[m]) for m in MODULI}
            for m in MODULI:
                assert set(dex[m].columns) == set(x.columns)
            stacks['control'].append(cp)
            for key in order:
                m, r = params[key]
                model, info = fit_candidate(x.iloc[fi], yy[fi], dex[m], exm[m].y.to_numpy(float), r, seed,
                                            x.iloc[va], yy[va])
                pred = model.predict(dte)
                stacks[key].append(pred)
                fits.append({'cell': key, 'm': m, 'r': r, 'month': month, 'seed': seed,
                             'unmatched_fit_n': len(fi), 'extra_n': len(dex[m]), **info})
                if key == inc_key:     # fail fast: MAU's own per-seed composed prediction, every row
                    got = np.asarray(compose(rr, te, pred), float)[keep]
                    np.testing.assert_array_equal(got, mau_m[f'candidate_seed{seed}'].to_numpy(float),
                                                  err_msg='incumbent does not reproduce MAU: VOID')
                if key == RERUN:
                    assert_rerun_identical(pred, stacks[inc_key][-1], f'{inc_key} month {month} seed {seed}')
                del model
            print(json.dumps({'month': month, 'seed': seed, 'seconds': time.monotonic()-started,
                              'rss_gib': _rss_gib()}), flush=True)
            del ctrl, replay, x, dte, dex
        out = te[['MVT_ID_mvt', 'ADEP_mvt', 'date', 'month', 'y', 'sp', 'A']].copy()
        out['day'] = te.MVT_TIME_UTC_mvt.dt.dayofyear.to_numpy()
        for c in ('p_hat', 'nf_cells', 'routed_nf_fit'):
            out[c] = rr[c].to_numpy()
        for arm, values in stacks.items():
            stack = np.asarray(values)
            body = stack.mean(axis=0)
            if arm == 'control':       # mau_build.py:217-221, production's own recipe
                mix = uc.bs.mixture(rr.p_hat, rr.sp, uc.bs.nf_hybrid(rr.nf_cells, body))
                assert float(np.max(np.abs(body-rr.nf_fit_c.to_numpy()))) < 1e-6
                assert float(np.max(np.abs(mix-rr.S1C.to_numpy()))) < 1e-6
            out[arm] = compose(rr, te, body)
            for j, seed in enumerate(U.SEEDS):
                out[f'{arm}_seed{seed}'] = compose(rr, te, stack[j])
        out = out.loc[out.ADEP_mvt.ne('LIRF')].copy()
        for arm in stacks:
            assert (out.loc[~out.routed_nf_fit, 'control'] == out.loc[~out.routed_nf_fit, arm]).all(), arm
        records.append(out)
    result = pd.concat(records, ignore_index=True)
    assert len(result) == 4924 and result.MVT_ID_mvt.is_unique
    # ---- the incumbent must BE MAU's persisted run, every row, every seed ----
    for ours, theirs in [('control', 'control'), (inc_key, 'candidate'),
                         *[(f'control_seed{s}', f'control_seed{s}') for s in U.SEEDS],
                         *[(f'{inc_key}_seed{s}', f'candidate_seed{s}') for s in U.SEEDS],
                         ('y', 'y'), ('day', 'day'), ('month', 'month'), ('MVT_ID_mvt', 'MVT_ID_mvt')]:
        np.testing.assert_array_equal(result[ours].to_numpy(float), mau_preds[theirs].to_numpy(float),
                                      err_msg=f'{ours} != MAU {theirs}: VOID')
    y, day, month_v = result.y.to_numpy(float), result.day.to_numpy(), result.month.to_numpy()
    inc = result[inc_key].to_numpy(float)
    incumbent_abs = delta_stats(y, result.control, inc, day, month_v)
    assert incumbent_abs['gain'] == mau_report['stats']['gain'], incumbent_abs['gain']
    assert incumbent_abs['ci95'] == mau_report['stats']['ci95'], incumbent_abs['ci95']
    # ---- the determinism null: incumbent - incumbent must be EXACTLY 0.0 with a degenerate CI ----
    assert_rerun_identical(result[RERUN], inc, f'{inc_key} composed')
    null = delta_stats(y, inc, result[RERUN], day, month_v)
    if not (null['gain'] == 0.0 and null['ci95'] == [0.0, 0.0]):
        raise AssertionError(f'VOID: incumbent-incumbent null is {null}')
    oracle = delta_stats(y, inc, y, day, month_v)
    ids = result.MVT_ID_mvt.to_numpy(float)
    grid_out = {}
    for key, (m, r) in cells.items():
        cell = result[key].to_numpy(float)
        d = delta_stats(y, inc, cell, day, month_v)
        grid_out[key] = {
            'm': m, 'r': r, 'sample_rate': 1/m,
            'delta_vs_incumbent': d,
            'screen': screen_clauses(d),
            'absolute_vs_control': delta_stats(y, result.control, cell, day, month_v),
            'seed_deltas': {str(s): pbl.board_gain(y, result[f'{inc_key}_seed{s}'], result[f'{key}_seed{s}'])
                            for s in U.SEEDS},
            'monster_rows_delta': monster_delta(ids, y, inc, cell),
            'values_differ_from_incumbent': int((cell != inc).sum()),
            'candidate_iters': [f['iter'] for f in fits if f['cell'] == key],
            'extra_n_by_month': {str(f['month']): f['extra_n'] for f in fits if f['cell'] == key},
        }
    winner = pick_winner({k: v['delta_vs_incumbent'] for k, v in grid_out.items()})
    result.to_parquet(PREDS, index=False)
    reread = pd.read_parquet(PREDS)
    pd.testing.assert_frame_equal(result, reread)
    report = {
        'tag': 'MSW',
        'screen_winner': winner,
        'verdict': 'NOT WORKING' if winner is None else 'PENDING: section M16 legs on the winner only',
        'selection_warning': 'A winner\'s Jan/Jul delta is BIASED UPWARD by selection over 12 cells and is '
                             'never the arm\'s value; its value is whatever the section M16 pair says.',
        'grid': grid_out, 'incumbent_vs_control': incumbent_abs, 'null_incumbent_vs_rerun': null,
        'oracle_vs_incumbent': oracle, 'fits': fits,
        'scope': '2025 Jan/Jul LOMO non-Rome unmatched S1C plus storm; delta = (SSE_incumbent - SSE_cell)/344841. '
                 'Not a board forecast.',
        'prereg_registered_sha256': REGISTERED_SHA, 'prereg_file_sha256': current,
        'runner_sha256': U.sha(__file__), 'mau_runner_sha256': U.sha(MB.__file__), 'input_hashes': sources,
        'pred_sha256': U.sha(PREDS), 'seconds': time.monotonic()-started, 'peak_rss_gib': _rss_gib(),
    }
    with REPORT.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    signal.alarm(0)
    print(json.dumps({k: report[k] for k in ('screen_winner', 'verdict', 'seconds', 'peak_rss_gib')}), flush=True)


if __name__ == '__main__':
    main()
