"""Synthetic caches with the REAL caches' columns and dtypes, for the end-to-end tests of
`scripts/lgbm_fold.py` and `scripts/lgbm_submit.py` that must run beside a live fold without the
real months (a second >= 2 GB job is not allowed on this machine; the real-month smokes are the
owner's to run).

Not a test module (no `test_` prefix): both test files load it through importlib, exactly as
they load the scripts. The generator was lifted out of tests/test_lgbm_fold.py on 2026-09-09
for the Amendment 19 arms, UNCHANGED in every draw it already made: the day block below is drawn
from its own RNG stream and every new option is applied after the existing draws, so a fixture
built with the options off is bit-identical to the one the queue / fill-head / CatBoost / sweep /
confirm tests were written against (test_lgbm_fold.py asserts planted magnitudes on it).

`synthetic_caches(root, months, ...)` writes data/cache_stand + data/cache_queue + data/cache_day
twins, one calendar month each: ten airports, in-fold keys, a target that is 250 x n_push + 150 x
[rwy_b30 > 0] (+ queue_signal x q_apt_at_push, visible only through the queue block; + day_signal
x a per-day effect, visible only through the day block) + N(0, 25), a 2% tail at |delta|
1,300-1,400 s, and 10% exact fills (sp == y). `proxy` is drawn independently of `delta` - an
earlier draft added max(delta, 0) to keep y positive and thereby leaked the target into a feature,
which shrank the planted queue gain from ~40 s to 6 s.

`synthetic_submission(root, ...)` adds what `lgbm_submit.main` needs on top: a serve-mode ranking
cache (labels NaN, months 1 and 7, MVT_ID_mvt) with its queue and day twins, the raw ranking file
(ids, phase, AOBT_3: matched departures, unmatched departures, arrivals), the submitting template
and a base submission `<team>_v1.parquet` (int32, ids shuffled).

The unified all-rows arm (the Amendment 19 addition) adds data/cache_unmatched: one file per month
in the unmatched cache's column contract (UNMATCHED_COLS) with the unmatched NaN pattern applied
(UNMATCHED_NAN_COLS and the *_flt-derived columns NaN, every other feature drawn), a target that is
the MATCHED structure with the proxy anchor replaced by its mean - 3,150 - 250 x n_push - 150 x
[rwy_b30 > 0] + N(0, 25) - as the real stratum is the same departures without an NM off-block
(an earlier draft planted an unrelated sp-linear target and the unified smoke could only get it
wrong: the matched rule "y is near sp" contradicted it), with ~2% monsters above 10,800 s, and
- for the fold - a stratum record shaped like stratum_fold_v7_preds.parquet whose fold-A S0 is the
target plus N(0, 500) (the "as shipped" unmatched baseline the unified arm is paired against).
"""
from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("stand_ab")

APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
#: spelled out, not read from the module, so the fixture does not certify the module's own list
QUEUE_FEATS = ["q_apt_at_push", "q_rwy_at_push", "q_pushed_after_me", "q_rwy_tko_in_taxi",
               "q_rwy_push_pre20", "q_dep_tko_sym15", "q_rwy_tko_sym10", "q_arr_taxiing_at_push",
               "q_arr_taxiin_sym30_push", "q_arr_taxiin_sym30_tko", "q_rwy_ambient_proxy",
               "q_next_arr_onblock_gap"]
QUEUE_VALUE_FEATS = ["q_arr_taxiin_sym30_push", "q_arr_taxiin_sym30_tko", "q_rwy_ambient_proxy",
                     "q_next_arr_onblock_gap"]
DAY_FEATS = ["d_prx_p50", "d_prx_p90", "d_prx_le0", "d_arr_p50", "d_arr_p90", "d_arr_long",
             "d_n_dep", "d_n_arr", "d_prx_minus_p50"]
#: the three day columns that are NaN together when a day has no arrival with a taxi-in
DAY_ARR_FEATS = ["d_arr_p50", "d_arr_p90", "d_arr_long"]
N_DAYS = 30                       # synthetic airport-days per month for the day block
DAY_RNG_OFFSET = 7_919            # the day block's own RNG stream: seed * 100 + month + this

#: the unmatched cache's NaN pattern, spelled out (the module's own lists are certified by
#: tests/test_unmatched_features.py, not by this fixture)
UNMATCHED_NAN_COLS = ["proxy", "aobt_sched", "aobt_eobt", "aobt_sec", "q_apt_at_push", "q_rwy_at_push",
                      "q_pushed_after_me", "q_rwy_tko_in_taxi", "q_rwy_push_pre20", "q_arr_taxiing_at_push",
                      "q_arr_taxiin_sym30_push", "q_next_arr_onblock_gap", "d_prx_minus_p50"]
UNMATCHED_FLT_COLS = ["eobt_p", "lobt_p", "iobt_p", "eobt_sched", "lobt_sched", "eobt_lobt", "iobt_lobt",
                      "airborne3", "airborne1", "arvt_diff"]
MONSTER_S = 10_800.0
N_UNMATCHED = 40                  # synthetic unmatched rows per month (the real stratum: ~1.8k of ~170k)
UNMATCHED_RNG_OFFSET = 31_337     # the unmatched rows' own RNG stream

INT_COLS = {"aobt_sec", "mvt_sec", "sched_sec", "hr", "tmin", "dow", "doy"}
KEYS = ["STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt",
        "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "ADEP_mvt", "stand_pref", "airline", "ars"]
TEAM = "merry-quicksand"
MONTH_NAME = "training_2025-{:02d}-01_2025-{:02d}-01.parquet"


def stand_columns() -> list:
    """The real stand cache's column order (labels first, then BASE / SURF / STANDHIST / the
    non-in-fold STAND_BLOCK, month, ap, the string keys)."""
    return (["delta", "y", "proxy", "sp"] + [c for c in S.BASE if c not in ("proxy", "sp")] + S.SURF
            + S.STANDHIST + [c for c in S.STAND_BLOCK if c not in S.STAND_INFOLD] + ["month", "ap"] + KEYS)


def unmatched_columns() -> list:
    """The unmatched cache's column order: id, labels, BASE / SURF / STANDHIST, month, ap, the
    string keys, then the queue and day blocks - one file per month, no twins."""
    return (["MVT_ID_mvt", "delta", "y"] + list(S.BASE) + list(S.SURF) + list(S.STANDHIST) + ["month", "ap"] + KEYS
            + QUEUE_FEATS + DAY_FEATS)


def unmatched_block(rng_u, n, m, cols, eff_day) -> pd.DataFrame:
    """`n` synthetic unmatched rows of month `m` in the unmatched cache's contract: every drawn
    feature as the matched rows have it, the NaN pattern applied, the target the matched
    structure without its proxy anchor (3,150 - 250 x n_push - 150 x [rwy_b30 > 0] + N(0, 25)),
    ~2% monsters. `eff_day` (N_DAYS values) is the month's day effect (the day block's values)."""
    d = {}
    for c in cols:
        if c in INT_COLS:
            d[c] = rng_u.integers(0, 1_440, n).astype("float64")
        elif c not in KEYS and c not in ("MVT_ID_mvt", "delta", "y", "proxy", "sp", "month", "ap"):
            d[c] = rng_u.normal(size=n)
    ap = np.array(APTS, dtype=object)[rng_u.integers(0, 10, n)].astype(str)
    stand_id = np.array([f"S{v}" for v in rng_u.integers(0, 12, n)])
    d.update(STAND_mvt=stand_id, RUNWAY_mvt=rng_u.choice(["27L", "09R", "25C"], n),
             AIRCRAFT_TYPE_mvt=rng_u.choice(["A320", "B738", "A21N", "E190"], n),
             ADES_mvt=rng_u.choice(["LFPG", "EGLL", "EDDF", "LEMD", "LIRF"], n),
             AIRCRAFT_OPERATOR_flt=np.array(["NA"] * n), MARKET_SEGMENT_flt=np.array(["NA"] * n),
             WK_TBL_CAT_flt=np.array(["NA"] * n), FLIGHT_TYPE_flt=np.array(["NA"] * n),
             ADEP_mvt=ap, stand_pref=np.array([s[:2] for s in stand_id]), airline=rng_u.choice(["DLH", "BAW", "AFR", "RYR", "VLG"], n),
             ars=np.array([f"{a}|{s}|27L" for a, s in zip(ap, stand_id)]))
    di = rng_u.integers(0, N_DAYS, n)
    eff = eff_day[di]
    for c, v in zip(DAY_FEATS, (600.0 + 100.0 * eff, 1_200.0 + 200.0 * eff, np.clip(0.05 + 0.02 * eff, 0, 1),
                                500.0 + 50.0 * eff, 900.0 + 100.0 * eff, np.clip(0.10 + 0.05 * eff, 0, 1),
                                np.round(300.0 + 30.0 * eff), np.round(250.0 + 25.0 * eff), np.full(n, np.nan))):
        d[c] = v
    y = 3_150.0 - 250.0 * d["n_push"] - 150.0 * (d["rwy_b30"] > 0) + rng_u.normal(0, 25, n)
    sp = y + rng_u.normal(0, 300, n)
    monster = rng_u.random(n) < 0.02
    y[monster] = rng_u.uniform(12_000, 15_000, monster.sum())
    y = np.maximum(y, 60.0)
    d.update(sp=sp, y=y, delta=np.full(n, np.nan), proxy=np.full(n, np.nan), month=np.full(n, m, "int32"), ap=ap)
    for c in UNMATCHED_NAN_COLS + UNMATCHED_FLT_COLS:
        d[c] = np.full(n, np.nan)
    d["f_callsign"] = d["f_ades"] = d["f_type"] = d["f_rule"] = np.ones(n)
    return pd.DataFrame(d)


def queue_block(rng, n) -> dict:
    q = {"q_apt_at_push": rng.integers(0, 8, n).astype("float64")}
    for c in QUEUE_FEATS[1:]:
        if c in QUEUE_VALUE_FEATS:
            v = rng.normal(600.0, 100.0, n)
            v[rng.random(n) < 0.1] = np.nan
            q[c] = v
        else:
            q[c] = rng.integers(0, 6, n).astype("float64")
    return q


def day_block(rng_day, n) -> tuple:
    """The day block from its OWN RNG stream: every row belongs to one of N_DAYS synthetic
    airport-days with a N(0, 1) effect `eff`; the eight day-level columns are functions of the
    row's day effect alone (constant within a day, as the real block is), the three arrival
    levels are NaN together on ~5% of days (a day without an arrival taxi-in), and
    d_prx_minus_p50 is per row. Returns (block dict, eff per row)."""
    eff_day = rng_day.normal(size=N_DAYS)
    di = rng_day.integers(0, N_DAYS, n)
    no_arr = rng_day.random(N_DAYS) < 0.05
    minus = rng_day.normal(0.0, 300.0, n)
    eff = eff_day[di]
    nan_rows = no_arr[di]
    blk = {"d_prx_p50": 600.0 + 100.0 * eff, "d_prx_p90": 1_200.0 + 200.0 * eff,
           "d_prx_le0": np.clip(0.05 + 0.02 * eff, 0.0, 1.0),
           "d_arr_p50": np.where(nan_rows, np.nan, 500.0 + 50.0 * eff),
           "d_arr_p90": np.where(nan_rows, np.nan, 900.0 + 100.0 * eff),
           "d_arr_long": np.where(nan_rows, np.nan, np.clip(0.10 + 0.05 * eff, 0.0, 1.0)),
           "d_n_dep": np.round(300.0 + 30.0 * eff), "d_n_arr": np.round(250.0 + 25.0 * eff),
           "d_prx_minus_p50": minus}
    return blk, eff


def _month(rng, rng_day, m, n, cols, queue_signal, fill_feature, day_signal, proxy_levels, serve) -> tuple:
    """One synthetic month: (stand frame dict, queue block, day block). The order of every draw
    from `rng` is the original generator's; the day block comes from `rng_day`."""
    d = {}
    for c in cols:
        if c in INT_COLS:
            d[c] = rng.integers(0, 1_440, n).astype("int32")
        elif c not in KEYS and c not in ("delta", "y", "proxy", "sp", "month", "ap"):
            d[c] = rng.normal(size=n)
    ap = np.array(APTS, dtype=object)[np.arange(n) % 10].astype(str)
    stand_id = np.array([f"S{v}" for v in rng.integers(0, 12, n)])
    operator = rng.choice(["DLH", "BAW", "AFR", "RYR", "VLG"], n)
    d.update(STAND_mvt=stand_id, RUNWAY_mvt=rng.choice(["27L", "09R", "25C"], n),
             AIRCRAFT_TYPE_mvt=rng.choice(["A320", "B738", "A21N", "E190"], n),
             ADES_mvt=rng.choice(["LFPG", "EGLL", "EDDF", "LEMD", "LIRF"], n),
             AIRCRAFT_OPERATOR_flt=operator, MARKET_SEGMENT_flt=rng.choice(["Sched", "Lowcost", "Cargo"], n),
             WK_TBL_CAT_flt=rng.choice(["M", "H", "L"], n), FLIGHT_TYPE_flt=rng.choice(["S", "N"], n),
             ADEP_mvt=ap, stand_pref=np.array([s[:2] for s in stand_id]), airline=operator,
             ars=np.array([f"{a}|{s}" for a, s in zip(ap, stand_id)]))
    q = queue_block(rng, n)
    delta = (250.0 * d["n_push"] + 150.0 * (d["rwy_b30"] > 0) + queue_signal * q["q_apt_at_push"]
             + rng.normal(0, 25, n))
    blk, eff = day_block(rng_day, n)
    if day_signal:
        delta = delta + day_signal * eff              # a day-level shift, visible only through the day block
    tail = rng.random(n) < 0.02
    delta[tail] = rng.choice([-1.0, 1.0], tail.sum()) * rng.uniform(1_300, 1_400, tail.sum())
    u = rng.random(n)
    if proxy_levels is None:
        proxy = 3_000.0 + 300.0 * u                    # independent of delta; y = proxy - delta >= 1,000 s
    else:
        # a few exact proxy levels: a tree recovers y = proxy - f(x) from proxy with two splits,
        # so the y-target arm can reproduce the delta-target arm (the Amendment 19.2 test)
        proxy = 3_000.0 + 300.0 * np.floor(u * int(proxy_levels)) / int(proxy_levels)
    assert np.abs(delta).max() < 2_000.0, "a delta beyond the proxy floor would put y below 1 s"
    y = proxy - delta
    sp = y + rng.normal(0, 300, n)
    fill = rng.random(n) < 0.1
    if fill_feature is not None:
        fill = d[fill_feature] > 1.2816
        near = ~fill & (np.abs(sp - y) <= 60.0)
        sp[near] = y[near] + np.where(sp[near] >= y[near], 120.0, -120.0)
    sp[fill] = y[fill]
    if serve:
        y, delta = np.full(n, np.nan), np.full(n, np.nan)   # the scored file carries no label
    d.update(delta=delta, y=y, proxy=proxy, sp=sp, month=np.full(n, m, "int32"), ap=ap)
    return d, q, blk


def _write_twins(qdir, ddir, name, ids, q, blk) -> None:
    qf = pd.DataFrame({"MVT_ID_mvt": ids})
    for c in QUEUE_FEATS:
        qf[c] = q[c].astype("float32")
    qf.to_parquet(qdir / name, index=False)
    df = pd.DataFrame({"MVT_ID_mvt": ids})
    for c in DAY_FEATS:
        df[c] = blk[c].astype("float32")
    df.to_parquet(ddir / name, index=False)


def synthetic_caches(root, months, n=600, seed=0, queue_signal=60.0, fill_feature=None, day_signal=0.0,
                     proxy_levels=None):
    """data/cache_stand + data/cache_queue + data/cache_day twins under `root`, one file per
    calendar month in `months`. Returns (stand dir, queue dir); the day dir is root/cache_day.

    With `fill_feature` (the Amendment 16 arm's test) the fills are PLANTED instead of drawn:
    fill iff that feature > 1.2816 (the top 10% of its N(0, 1) draw), and every other row's sp is
    pushed at least 120 s away from y, so the 60 s label is exactly the plant and one feature
    identifies it. With `day_signal` the target carries day_signal x the row's day effect (the
    Amendment 19.1 arm's test); with `proxy_levels` the proxy takes that many exact levels (the
    Amendment 19.2 arm's test). The main RNG stream is untouched by every option (every draw
    still happens, in the original order).
    """
    stand, queue, day = root / "cache_stand", root / "cache_queue", root / "cache_day"
    unmatched = root / "cache_unmatched"
    stand.mkdir(parents=True), queue.mkdir(parents=True), day.mkdir(parents=True), unmatched.mkdir(parents=True)
    assert set(KEYS) == set(S.ENC_KEYS)
    cols = stand_columns()
    ucols = unmatched_columns()
    next_id = 1.0
    for m in months:
        rng = np.random.default_rng(seed * 100 + m)
        rng_day = np.random.default_rng(seed * 100 + m + DAY_RNG_OFFSET)
        d, q, blk = _month(rng, rng_day, m, n, cols, queue_signal, fill_feature, day_signal, proxy_levels,
                           serve=False)
        name = MONTH_NAME.format(m, m + 1)
        pd.DataFrame(d)[cols].to_parquet(stand / name, index=False)
        _write_twins(queue, day, name, next_id + np.arange(n, dtype="float64"), q, blk)
        next_id += n
        # the unmatched rows of the month: their own RNG stream, ids from 500,000 up
        rng_u = np.random.default_rng(seed * 100 + m + UNMATCHED_RNG_OFFSET)
        eff_day = np.random.default_rng(seed * 100 + m + DAY_RNG_OFFSET).normal(size=N_DAYS)
        u = unmatched_block(rng_u, N_UNMATCHED, m, ucols, eff_day)
        u.insert(0, "MVT_ID_mvt", 500_000.0 + m * 1_000 + np.arange(N_UNMATCHED, dtype="float64"))
        u[ucols].to_parquet(unmatched / name, index=False)
    return stand, queue


def synthetic_stratum_record(root, months, seed=0, holdout=(1, 7)) -> pathlib.Path:
    """A stratum_fold_v7_preds.parquet-shaped record for the synthetic unmatched rows: every
    unmatched row of a holdout month as a fold-"A" row (the others as "lomo" rows) with S0 = the
    target + N(0, 500) floored at 1 s (the "as shipped" unmatched prediction the unified arm is
    paired against), S1 = S0, the monster flag, p_hat / nf_* placeholders. Written to
    root/cache_stand/stratum_fold_v7_preds.parquet; returns the path."""
    unmatched = root / "cache_unmatched"
    rng = np.random.default_rng(seed * 100 + 77)
    parts = []
    for m in months:
        u = pd.read_parquet(unmatched / MONTH_NAME.format(m, m + 1))
        s0 = np.maximum(u.y.to_numpy() + rng.normal(0, 500, len(u)), 1.0)
        f = pd.DataFrame({"MVT_ID_mvt": u.MVT_ID_mvt.to_numpy(), "fold": "A" if m in holdout else "lomo",
                          "month": u.month.to_numpy().astype("int64"), "ADEP_mvt": u.ap.to_numpy(),
                          "y": u.y.to_numpy(), "sp": u.sp.to_numpy(), "monster": u.y.to_numpy() > MONSTER_S,
                          "p_hat": np.full(len(u), 0.05), "nf_cells": s0, "nf_fit": s0, "routed_nf_fit": True,
                          "S0": s0, "S1": s0, "S1_seed0": s0, "S1_seed1": s0, "S1_seed2": s0})
        parts.append(f)
    out = root / "cache_stand" / "stratum_fold_v7_preds.parquet"
    pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
    return out


def synthetic_submission(root, months=(1, 2, 3), n=600, seed=0, n_rank=6_000, n_unmatched=200, n_arr=300,
                         **kw) -> SimpleNamespace:
    """The caches of `synthetic_caches` plus everything `lgbm_submit.main` reads:

      * <stand>/ranking.parquet - serve-mode rows (y, delta NaN; months 1 and 7, half each;
        MVT_ID_mvt first) with <queue>/ranking.parquet and <day>/ranking.parquet twins in the
        same row order;
      * <raw>/ranking.parquet - MVT_ID_mvt / PHASE_mvt / AOBT_3_flt for the matched departures
        (AOBT_3 set), `n_unmatched` departures without one and `n_arr` arrivals;
      * <raw>/submitting.parquet - the template: every departure's id with a placeholder time;
      * <subs>/<team>_v1.parquet - the base submission: the template ids shuffled, int32 times.

    Returns SimpleNamespace(stand, queue, day, raw, subs, base, rank_ids, matched_ids, unmatched_ids).
    """
    stand, queue = synthetic_caches(root, months, n=n, seed=seed, **kw)
    day = root / "cache_day"
    raw, subs = root / "raw", root / "submissions"
    raw.mkdir(parents=True), subs.mkdir(parents=True)
    cols = stand_columns()
    first_id = 1_000_000.0
    parts, ids_all, q_all, blk_all = [], [], {c: [] for c in QUEUE_FEATS}, {c: [] for c in DAY_FEATS}
    per_month = n_rank // 2
    for i, m in enumerate((1, 7)):
        rng = np.random.default_rng(seed * 100 + 50 + m)
        rng_day = np.random.default_rng(seed * 100 + 50 + m + DAY_RNG_OFFSET)
        d, q, blk = _month(rng, rng_day, m, per_month, cols, kw.get("queue_signal", 60.0), kw.get("fill_feature"),
                           kw.get("day_signal", 0.0), kw.get("proxy_levels"), serve=True)
        ids = first_id + i * per_month + np.arange(per_month, dtype="float64")
        frame = pd.DataFrame(d)[cols]
        frame.insert(0, "MVT_ID_mvt", ids)
        parts.append(frame)
        ids_all.append(ids)
        for c in QUEUE_FEATS:
            q_all[c].append(q[c])
        for c in DAY_FEATS:
            blk_all[c].append(blk[c])
    rank = pd.concat(parts, ignore_index=True)
    rank.to_parquet(stand / "ranking.parquet", index=False)
    rank_ids = np.concatenate(ids_all)
    _write_twins(queue, day, "ranking.parquet", rank_ids,
                 {c: np.concatenate(v) for c, v in q_all.items()}, {c: np.concatenate(v) for c, v in blk_all.items()})

    rng = np.random.default_rng(seed * 100 + 99)
    unmatched = first_id + n_rank + np.arange(n_unmatched, dtype="float64")
    # the unmatched rows of the scored file: the unmatched cache's contract in serve mode (y NaN)
    ucols = unmatched_columns()
    rng_u = np.random.default_rng(seed * 100 + 50 + UNMATCHED_RNG_OFFSET)
    u_parts = []
    for i, m in enumerate((1, 7)):
        k = n_unmatched // 2 if i == 0 else n_unmatched - n_unmatched // 2
        u = unmatched_block(rng_u, k, m, ucols, np.random.default_rng(seed * 100 + 50 + m + DAY_RNG_OFFSET).normal(size=N_DAYS))
        u["y"] = np.nan
        u_parts.append(u)
    ur = pd.concat(u_parts, ignore_index=True)
    ur.insert(0, "MVT_ID_mvt", unmatched)
    ur[ucols].to_parquet(root / "cache_unmatched" / "ranking.parquet", index=False)
    arrivals = first_id + n_rank + n_unmatched + np.arange(n_arr, dtype="float64")
    t0 = pd.Timestamp("2026-01-05 10:00:00", tz="UTC")
    raw_rank = pd.DataFrame({
        "MVT_ID_mvt": np.concatenate([rank_ids, unmatched, arrivals]),
        "PHASE_mvt": ["DEP"] * (len(rank_ids) + n_unmatched) + ["ARR"] * n_arr,
        "AOBT_3_flt": [t0 + pd.Timedelta(seconds=int(k)) for k in range(len(rank_ids))]
                      + [pd.NaT] * (n_unmatched + n_arr)})
    raw_rank["AOBT_3_flt"] = pd.to_datetime(raw_rank.AOBT_3_flt, utc=True)
    raw_rank = raw_rank.iloc[rng.permutation(len(raw_rank))].reset_index(drop=True)
    raw_rank.to_parquet(raw / "ranking.parquet", index=False)
    dep_ids = np.concatenate([rank_ids, unmatched])
    template = pd.DataFrame({"MVT_ID_mvt": dep_ids, "TAXITIME_SEC_mvt": np.full(len(dep_ids), 600, "int32")})
    template.to_parquet(raw / "submitting.parquet", index=False)
    order = rng.permutation(len(dep_ids))
    base = pd.DataFrame({"MVT_ID_mvt": dep_ids[order],
                         "TAXITIME_SEC_mvt": rng.integers(300, 2_000, len(dep_ids)).astype("int32")})
    base_path = subs / f"{TEAM}_v1.parquet"
    base.to_parquet(base_path, index=False)
    return SimpleNamespace(stand=stand, queue=queue, day=day, unmatched=root / "cache_unmatched", raw=raw, subs=subs,
                           base=base_path, rank_ids=rank_ids, matched_ids=set(rank_ids.tolist()),
                           unmatched_ids=set(unmatched.tolist()))
