"""The Amendment 19 arm D block: airport-day regime features (PREREG Amendment 19.1).

`stand_ab.build_day(table, serve)` emits `MVT_ID_mvt` + `DAY_FEATS` for exactly the rows
`build_features` would emit, in the same take-off order. Per (airport, UTC calendar day of the
row's own MVT_TIME), from the SAME file the row lives in - transductive, as 19.1 permits - and
WITHOUT self-exclusion (a day-level aggregate over hundreds of rows is not moved by the row
itself beyond the percentile's own resolution, and the scored file is built the same way):
the day's departure `proxy` p50 / p90 / share <= 0, the day's arrival taxi-in p50 / p90 / share
> 1,200 s, the two counts, and the row's own proxy minus the day's p50. Everything read belongs
to other departures' MVT_TIME / AOBT_3 or to arrivals' MVT_TIME / BLOCK_TIME / TAXITIME, all
populated on the evaluation file; a departure's own BLOCK_TIME / TAXITIME are the target and
must never be touched.

What is pinned, each on a fixture whose every value is derived by hand, or on real rows:

  1. the arithmetic of every feature in BOTH modes, with the rows that separate the day boundary
     (a take-off at 23:59:59 and one at 00:00:00, a push on the day before its take-off), the
     `<= 0` proxy edge (exactly 0, and a negative), the strict `> 1,200 s` taxi-in edge (exactly
     1,200), a NaN taxi-in, a day without arrivals, an arrival landing at 23:59:59 and going on
     block the next day, and rows at an airport outside the ten;
  2. the hidden-clock invariant on the fixture: blanking AND permuting departure BLOCK/TAXITIME
     leaves the serve build byte-identical, while training mode's row set moves under the
     permutation (the control that proves the permutation is real);
  3. the per-calendar-month evaluation build, and its refusal of a row without MVT_TIME;
  4. the cache command: one month under --smoke, the contract columns, a smoke ranking refused;
  5. on real rows (needs the challenge data): the row contract against build_month in both
     modes, serve/train parity with the causal pruning proof, the hidden clocks on a January
     slice, and the evaluation build of exactly the 339,551 matched rows with NaN rates within
     2 pp of March training. WRITTEN 2026-09-09 AND NOT YET RUN - a live ~5 GB fold occupied
     the machine and the owner runs the real-month tests after it; each names the mutation to
     rehearse on its first run.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
JAN = RAW / "training_2025-01-01_2025-02-01.parquet"
MARCH = RAW / "training_2025-03-01_2025-04-01.parquet"
RANKING = RAW / "ranking.parquet"
TEMPLATE = RAW / "submitting.parquet"

_spec = importlib.util.spec_from_file_location("stand_ab", ROOT / "scripts" / "stand_ab.py")
stand_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stand_ab)

needs_data = pytest.mark.skipif(
    not (JAN.exists() and MARCH.exists() and RANKING.exists() and TEMPLATE.exists()),
    reason="challenge data is not redistributable and is absent from a clean checkout",
)

#: the contract, spelled out rather than read back from the module so a typo in the module's
#: list cannot certify itself
DAY_FEATS = ["d_prx_p50", "d_prx_p90", "d_prx_le0", "d_arr_p50", "d_arr_p90", "d_arr_long",
             "d_n_dep", "d_n_arr", "d_prx_minus_p50"]
#: mechanism per column, for the parity tests: DEP-stream features may legitimately differ
#: between modes on the days the serve filter admits extra reference departures for; ARR-stream
#: features read arrivals only and must not move at all.
DEP_STREAM = ["d_prx_p50", "d_prx_p90", "d_prx_le0", "d_n_dep", "d_prx_minus_p50"]
ARR_STREAM = ["d_arr_p50", "d_arr_p90", "d_arr_long", "d_n_arr"]
COUNTS = ["d_n_dep", "d_n_arr"]
#: the serve/train parity test holds the DEP-stream count and the DEP-stream levels / shares to
#: different instruments: a day that gains one reference departure moves EVERY row of that
#: airport-day's count by exactly +1 (an identity, checked row by row against the extra rows),
#: while it moves the day's percentiles only by their own resolution (a rate, held to
#: MIN_DAY_MATCH). On March 2025, 6 extra rows moved 1.5% of the rows' count - a rate bar cannot
#: tell that from a bug; the identity can.
DEP_COUNT = "d_n_dep"
DEP_LEVELS = [c for c in DEP_STREAM if c != DEP_COUNT]
MIN_DAY_MATCH = 0.99
MAX_NAN_RATE_DRIFT = 0.02
QCOLS = ["PHASE_mvt", "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt",
         "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
         "AOBT_3_flt"]

# ---------------------------------------------------------------------------------------------
# the micro-fixture: two airports (EGLL, LFPG), two UTC days, thirteen admissible departures
# plus one at EBBR (outside the ten) and one without a label (serve mode only), seven arrivals
# plus one at EBBR and one without an on-block. Seconds are offsets from T0 = day 1, 00:00 UTC;
# day 2 starts at 86,400. Every value below is derived by hand from these rows.
# ---------------------------------------------------------------------------------------------
T0 = pd.Timestamp("2025-03-10 00:00:00", tz="UTC")
DAY = 86_400
#: (MVT_ID, airport, AOBT_3 = push, MVT_TIME = take-off, labelled) - proxy = take-off - push
DEPS = [(1, "EGLL", 1_000, 1_600, True),        # EGLL day 1: proxies 600, 800, 300, 1000, 0
        (2, "EGLL", 2_000, 2_800, True),
        (3, "EGLL", 3_000, 3_300, True),
        (4, "EGLL", 4_000, 5_000, True),
        (5, "EGLL", 6_000, 6_000, True),        # proxy exactly 0: inside `<= 0`
        (7, "EGLL", 90_000, 90_500, True),      # EGLL day 2: proxies 500, 300, -100 (+400 in serve mode)
        (8, "EGLL", 91_000, 91_300, True),
        (9, "EGLL", 95_000, 94_900, True),      # take-off before push: proxy -100
        (15, "EGLL", 92_000, 92_400, False),    # no label: admitted by the serve filter only
        (10, "LFPG", 85_000, 86_399, True),     # take-off 23:59:59 of day 1: day 1
        (11, "LFPG", 80_000, 81_000, True),
        (12, "LFPG", 86_000, 86_400, True),     # pushed on day 1, take-off 00:00:00 of day 2: day 2
        (13, "LFPG", 100_000, 100_600, True),
        (14, "EBBR", 5_000, 5_600, True)]       # outside the ten airports: not a row, not a reference
#: (MVT_ID, airport = ADES, landing = MVT_TIME, on-block = BLOCK_TIME, taxi-in or NaN, has on-block)
ARRS = [(101, "EGLL", 500, 900, 400.0, True),          # EGLL day 1 taxi-ins 400, 1200, 1500, 301
        (102, "EGLL", 2_000, 3_200, 1_200.0, True),    # exactly 1,200: NOT long (strict >)
        (103, "EGLL", 4_000, 5_500, 1_500.0, True),
        (104, "EGLL", 86_399, 86_700, 301.0, True),    # lands 23:59:59 day 1, on block day 2: day 1
        (105, "EGLL", 90_000, 90_600, 600.0, True),    # EGLL day 2
        (106, "EGLL", 91_000, 92_000, np.nan, True),   # taxi-in NaN: counted, not in the levels
        (107, "LFPG", 50_000, 51_300, 1_300.0, True),  # LFPG day 1: one arrival, long
        (108, "EBBR", 1_000, 1_500, 500.0, True),      # outside the ten airports
        (109, "EGLL", 3_000, 3_500, 500.0, False)]     # no on-block: not in the arrival stream
#: hand derivation (numpy linear-interpolation percentiles: rank 0.9 x (n - 1) between neighbours)
#:   EGLL day 1  proxies sorted 0 300 600 800 1000 -> p50 600, p90 800 + 0.6 x 200 = 920,
#:               le0 1/5; arrivals 301 400 1200 1500 -> p50 800, p90 1200 + 0.7 x 300 = 1410,
#:               long 1/4 (1500 only), n_arr 4 (109 has no on-block; 108 is EBBR)
#:   EGLL day 2  training: proxies -100 300 500 -> p50 300, p90 300 + 0.8 x 200 = 460, le0 1/3,
#:               n_dep 3; serve: -100 300 400 500 -> p50 350, p90 400 + 0.7 x 100 = 470, le0 1/4,
#:               n_dep 4; arrivals: 600 and a NaN -> p50 600, p90 600, long 0, n_arr 2
#:   LFPG day 1  proxies 1000 1399 -> p50 1199.5, p90 1000 + 0.9 x 399 = 1359.1, le0 0;
#:               arrivals 1300 -> p50 1300, p90 1300, long 1, n_arr 1
#:   LFPG day 2  proxies 400 600 -> p50 500, p90 400 + 0.9 x 200 = 580, le0 0; no arrivals ->
#:               NaN levels, n_arr 0
nan = np.nan
_D = dict(EGLL1=(600, 920, 1 / 5, 800, 1410, 1 / 4, 5, 4),
          EGLL2_train=(300, 460, 1 / 3, 600, 600, 0.0, 3, 2),
          EGLL2_serve=(350, 470, 1 / 4, 600, 600, 0.0, 4, 2),
          LFPG1=(1_199.5, 1_359.1, 0.0, 1_300, 1_300, 1.0, 2, 1),
          LFPG2=(500, 580, 0.0, nan, nan, nan, 2, 0))
#: MVT_ID -> (day key, own proxy)
_ROWS = {1: ("EGLL1", 600), 2: ("EGLL1", 800), 3: ("EGLL1", 300), 4: ("EGLL1", 1_000), 5: ("EGLL1", 0),
         7: ("EGLL2", 500), 8: ("EGLL2", 300), 9: ("EGLL2", -100), 15: ("EGLL2", 400),
         10: ("LFPG1", 1_399), 11: ("LFPG1", 1_000), 12: ("LFPG2", 400), 13: ("LFPG2", 600)}
#: take-off order of the admissible departures, training mode and serve mode
TAKEOFF_ORDER_TRAIN = [1, 2, 3, 4, 5, 11, 10, 12, 7, 8, 9, 13]
TAKEOFF_ORDER_SERVE = [1, 2, 3, 4, 5, 11, 10, 12, 7, 8, 15, 9, 13]


def expected(serve: bool) -> dict:
    """{MVT_ID: [the nine values]} for the mode."""
    out = {}
    for mid, (key, px) in _ROWS.items():
        if mid == 15 and not serve:
            continue
        stats = _D[key + ("_serve" if serve else "_train")] if key == "EGLL2" else _D[key]
        p50, p90, le0, a50, a90, long, n_dep, n_arr = stats
        out[mid] = [p50, p90, le0, a50, a90, long, n_dep, n_arr, px - p50]
    return out


def _ts(seconds):
    return T0 + pd.Timedelta(seconds=int(seconds))


def micro_fixture(blank_departure_clocks: bool = False, shift_days: int = 0, id_offset: int = 0) -> pd.DataFrame:
    """The rows above as a movement table in the raw schema (QCOLS), deliberately shuffled.
    `shift_days` moves every clock by whole days and `id_offset` renumbers, for a second month."""
    rows = []
    s = shift_days * DAY
    for mid, apt, a, t, labelled in DEPS:
        block = _ts(a - 120 + s) if labelled else pd.NaT      # arbitrary: no feature may depend on it
        rows.append(dict(PHASE_mvt="DEP", MVT_ID_mvt=float(mid + id_offset), ADEP_mvt=apt, ADES_mvt="LEMD",
                         STAND_mvt="A1", RUNWAY_mvt="27L", MVT_TIME_UTC_mvt=_ts(t + s),
                         SCHED_TIME_UTC_mvt=_ts(a - 300 + s), BLOCK_TIME_UTC_mvt=block,
                         TAXITIME_SEC_mvt=float(t - (a - 120)) if labelled else np.nan, AOBT_3_flt=_ts(a + s)))
    for mid, apt, land, onb, taxi, has_block in ARRS:
        rows.append(dict(PHASE_mvt="ARR", MVT_ID_mvt=float(mid + id_offset), ADEP_mvt="LEMD", ADES_mvt=apt,
                         STAND_mvt="B2", RUNWAY_mvt="27R", MVT_TIME_UTC_mvt=_ts(land + s),
                         SCHED_TIME_UTC_mvt=_ts(land - 600 + s),
                         BLOCK_TIME_UTC_mvt=_ts(onb + s) if has_block else pd.NaT,
                         TAXITIME_SEC_mvt=taxi, AOBT_3_flt=pd.NaT))
    t = pd.DataFrame(rows)
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"):
        t[c] = pd.to_datetime(t[c], utc=True)
    t = t.iloc[np.random.default_rng(5).permutation(len(t))].reset_index(drop=True)
    if blank_departure_clocks:
        dep = (t.PHASE_mvt == "DEP").to_numpy()
        t.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
        t.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    return t[QCOLS]


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Row-wise equality that treats NaN == NaN, for float and string columns alike."""
    x, z = a.to_numpy(), b.to_numpy()
    if a.dtype.kind in "fi" and b.dtype.kind in "fi":
        x, z = x.astype(float), z.astype(float)
        return (x == z) | (np.isnan(x) & np.isnan(z))
    return x == z


def _aligned(train: pd.DataFrame, serve: pd.DataFrame) -> pd.DataFrame:
    assert train.MVT_ID_mvt.is_unique and serve.MVT_ID_mvt.is_unique
    missing = set(train.MVT_ID_mvt) - set(serve.MVT_ID_mvt)
    assert not missing, f"{len(missing)} training rows are absent from the serve build"
    return serve.set_index("MVT_ID_mvt").loc[train.MVT_ID_mvt.to_numpy()].reset_index()


def _utc_day(clock: pd.Series) -> np.ndarray:
    """The UTC calendar day of a clock as whole days since the epoch - the floor of its epoch
    seconds over 86,400 - derived HERE and not by stand_ab._day_index, so the parity test's
    expected count shift cannot mirror a day boundary that drifts in the module."""
    seconds = (clock - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    return np.floor_divide(seconds, 86_400.0).astype("int64")


def _int_counts(s: pd.Series) -> np.ndarray:
    """A float32 count column as int64; a NaN or a non-integer is not a count and is refused."""
    v = s.to_numpy().astype("float64")
    assert np.isfinite(v).all() and (v == np.rint(v)).all(), f"{s.name} is not a count column"
    return v.astype("int64")


def _count_shift_from_extra_rows(t: pd.DataFrame, extra_ids: set, ids: np.ndarray) -> np.ndarray:
    """For every departure id in `ids`, the number of `extra_ids` departures of the same raw
    file on the same (airport, UTC day of MVT_TIME): the shift the serve filter's extra
    reference rows must give that row's day departure count - and nothing else may."""
    dep = t[t.PHASE_mvt == "DEP"].set_index("MVT_ID_mvt")
    assert dep.index.is_unique, "duplicate departure ids in the raw month"
    is_extra = dep.index.isin(list(extra_ids))
    assert int(is_extra.sum()) == len(extra_ids), "an extra row is not a departure of the file"
    keys = pd.MultiIndex.from_arrays([dep.ADEP_mvt.fillna("NA").astype(str).to_numpy(),
                                      _utc_day(dep.MVT_TIME_UTC_mvt)])
    per_day = pd.Series(1, index=keys[is_extra]).groupby(level=[0, 1]).sum()
    pos = dep.index.get_indexer(ids)
    assert (pos >= 0).all(), "a built row is not a departure of the file"
    return per_day.reindex(keys.take(pos)).fillna(0).to_numpy().astype("int64")


def _slice(src: pathlib.Path, lo: str, hi: str) -> pd.DataFrame:
    t = pq.read_table(src, columns=stand_ab.DCOLS).to_pandas()
    return t[(t.MVT_TIME_UTC_mvt >= pd.Timestamp(lo, tz="UTC"))
             & (t.MVT_TIME_UTC_mvt < pd.Timestamp(hi, tz="UTC"))]


def _blank_departures(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    dep = (t.PHASE_mvt == "DEP").to_numpy()
    t.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    t.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    return t


def _write(frame: pd.DataFrame, path: pathlib.Path) -> pathlib.Path:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    return path


def _check_values(o: pd.DataFrame, want: dict, mode: str) -> None:
    """Every column of every row against the hand values: counts exact, levels at float32
    resolution (2e-3 absolute is 16 float32 ulps at 1,400 s and a thousand times below the gap
    to any other percentile definition), NaN only where NaN is expected."""
    by_id = o.set_index("MVT_ID_mvt")
    wrong = []
    for mid, vals in want.items():
        for c, w in zip(DAY_FEATS, vals):
            g = float(by_id.loc[float(mid), c])
            ok = (np.isnan(g) and np.isnan(w)) if (np.isnan(g) or np.isnan(w)) else \
                (g == w if c in COUNTS else abs(g - w) <= 2e-3)
            if not ok:
                wrong.append((mid, c, g, w))
    assert not wrong, f"{mode}: hand computation disagrees (MVT_ID, column, got, expected): {wrong}"


# ---------------------------------------------------------------------------------------------
# 0. the contract itself
# ---------------------------------------------------------------------------------------------

def test_day_feature_list_is_the_contract_and_every_column_is_classified():
    """DAY_FEATS is exactly the nine pre-registered names, in this order; every one is assigned
    a mechanism (DEP_STREAM or ARR_STREAM) so it cannot escape the parity test; no name shadows
    an existing feature (BASELINE_FEATS, STAND_BLOCK, QUEUE_FEATS); the long-taxi-in edge is
    1,200 s; the raw columns the block reads exclude STAND and RUNWAY (it never keys on them).

    Fails when a name is misspelt, dropped, reordered or added in the module without being
    classified here. Rehearsed 2026-09-09: `"d_prx_p9O"` in the module's list went RED.
    """
    assert stand_ab.DAY_FEATS == DAY_FEATS
    assert sorted(DEP_STREAM + ARR_STREAM) == sorted(DAY_FEATS)
    assert not set(DEP_STREAM) & set(ARR_STREAM)
    assert not set(DAY_FEATS) & set(stand_ab.BASELINE_FEATS + stand_ab.STAND_BLOCK + stand_ab.QUEUE_FEATS), \
        "a day feature shadows an existing feature name"
    assert stand_ab.DAY_ARR_LONG_S == 1_200.0 and stand_ab.DAY_PERCENTILES == (50, 90)
    assert set(stand_ab.DCOLS) <= set(stand_ab.QCOLS) and "STAND_mvt" not in stand_ab.DCOLS
    assert stand_ab.DCACHE == ROOT / "data" / "cache_day"


# ---------------------------------------------------------------------------------------------
# 1. arithmetic, by hand, both modes
# ---------------------------------------------------------------------------------------------

def test_micro_fixture_every_feature_equals_the_hand_computation_in_both_modes():
    """Every one of the nine features on the thirteen admissible departures equals the value
    derived by hand in `expected`, in take-off order, float32, keyed by MVT_ID - and in serve
    mode, where the filter admits the unlabelled departure 15, the EGLL day-2 statistics move
    exactly as the hand derivation says (p50 300 -> 350, p90 460 -> 470, share <= 0 1/3 -> 1/4,
    n_dep 3 -> 4) while every other row and every arrival column is unchanged.

    The rows pin: the UTC day boundary (take-off at 23:59:59 vs 00:00:00, a push on the day
    before its take-off), proxy exactly 0 and negative inside `<= 0`, taxi-in exactly 1,200
    outside `> 1,200`, a NaN taxi-in counted but not levelled, a day without arrivals (NaN
    levels, count 0), an arrival landing on day 1 and going on block on day 2 (day 1), an
    arrival without an on-block (not in the stream), and EBBR rows (neither rows nor references).

    Fails when p50 and p90 are swapped, when the day is not the UTC date of the row's own
    MVT_TIME, when arrivals are keyed on their origin, when the `<= 0` / `> 1,200` edges move,
    when the row is excluded from its own day, or when the serve filter differs from
    build_features's. Rehearsed 2026-09-09, each RED: p50 and p90 swapped in `build_day`
    (`_percentiles` reversed); `np.floor_divide(tk - 1.0, DAY_S)` (row 12 falls into day 1);
    `sarr(arr.ADEP_mvt)` for the arrival airport (every arrival level NaN); `d_n_arr` taken from
    the departure group (`len(idx)`); `tx > DAY_ARR_LONG_S` -> `>=` (row 102 becomes long:
    EGLL day 1 long 0.25 -> 0.5).
    """
    o = stand_ab.build_day(micro_fixture(), serve=False)
    assert list(o.columns) == ["MVT_ID_mvt"] + DAY_FEATS
    assert o.MVT_ID_mvt.tolist() == TAKEOFF_ORDER_TRAIN, "rows must come out in take-off order"
    assert all(o[c].dtype == np.float32 for c in DAY_FEATS) and o.MVT_ID_mvt.dtype == np.float64
    _check_values(o, expected(serve=False), "training")

    served = stand_ab.build_day(micro_fixture(blank_departure_clocks=True), serve=True)
    assert served.MVT_ID_mvt.tolist() == TAKEOFF_ORDER_SERVE
    _check_values(served, expected(serve=True), "serve")
    aligned = _aligned(o, served)
    for c in ARR_STREAM:
        assert _same(o[c], aligned[c]).all(), f"{c}: an arrival column moved between modes"
    moved = {c: o.MVT_ID_mvt[~_same(o[c], aligned[c])].tolist() for c in DEP_STREAM}
    assert moved == {"d_prx_p50": [7, 8, 9], "d_prx_p90": [7, 8, 9], "d_prx_le0": [7, 8, 9],
                     "d_n_dep": [7, 8, 9], "d_prx_minus_p50": [7, 8, 9]}, \
        "serve mode must move exactly the EGLL day-2 rows (the day that gained a reference departure)"


def test_build_day_refuses_an_empty_departure_stream():
    """A table with no admissible departure is a broken input, not an empty feature frame.

    Fails when build_day returns silently on zero rows (a month cache of nothing would then be
    written and joined). Rehearsed 2026-09-09: replacing the raise with `pass` went RED
    (an IndexError from the empty groupby instead of the ValueError).
    """
    arrivals_only = micro_fixture()
    arrivals_only = arrivals_only[arrivals_only.PHASE_mvt == "ARR"]
    with pytest.raises(ValueError, match="no admissible departures"):
        stand_ab.build_day(arrivals_only, serve=False)
    with pytest.raises(ValueError, match="no admissible departures"):
        stand_ab.build_day(micro_fixture(blank_departure_clocks=True), serve=False)   # training needs the label


# ---------------------------------------------------------------------------------------------
# 2. the hidden clocks, on the fixture
# ---------------------------------------------------------------------------------------------

def test_day_features_never_read_the_departure_own_hidden_clocks_on_the_fixture():
    """Permuting BLOCK_TIME and TAXITIME across departures, or blanking both on every departure
    as the evaluation file does, leaves the serve build byte-identical - rows, order and every
    column. The control proves the permutation is real: training mode, whose filter reads
    TAXITIME > 0, admits a different row set under it (the fixture's one unlabelled departure
    lends its NaN to another row).

    Fails when any day feature or the serve filter reads a departure's BLOCK or TAXITIME.
    Rehearsed 2026-09-09: `+ 1e-3 * es(d.BLOCK_TIME_UTC_mvt)` on d_prx_minus_p50 went RED,
    the blanked variant naming the column (NaN) and the permuted one too.
    """
    base_tbl = micro_fixture()
    base = stand_ab.build_day(base_tbl, serve=True)
    t = base_tbl.copy()
    dep = (t.PHASE_mvt == "DEP").to_numpy()
    perm = np.random.default_rng(0).permutation(dep.sum())
    for col in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"):
        vals = t.loc[dep, col].to_numpy()
        t.loc[dep, col] = vals[perm]
    assert not _same(t.TAXITIME_SEC_mvt, base_tbl.TAXITIME_SEC_mvt).all(), "the permutation moved nothing"
    for name, tbl in (("permuted", t), ("blanked", micro_fixture(blank_departure_clocks=True))):
        served = stand_ab.build_day(tbl, serve=True)
        assert served.MVT_ID_mvt.tolist() == base.MVT_ID_mvt.tolist(), f"{name}: the serve row set changed"
        moved = [c for c in DAY_FEATS if not _same(base[c], served[c]).all()]
        assert not moved, f"{name}: day features read the departure's own hidden clocks: {moved}"
    ref = stand_ab.build_day(base_tbl, serve=False)
    trained = stand_ab.build_day(t, serve=False)
    assert set(trained.MVT_ID_mvt) != set(ref.MVT_ID_mvt), \
        "training mode admitted the same rows under the permutation - the control proved nothing"


# ---------------------------------------------------------------------------------------------
# 3. the evaluation build is per calendar month
# ---------------------------------------------------------------------------------------------

def test_ranking_build_is_per_calendar_month_and_refuses_a_row_without_mvt_time(tmp_path, monkeypatch):
    """Built one calendar month at a time, serve mode, as the training files are: on a fixture of
    two months (the micro-fixture and a copy shifted by 31 days, departures blanked as on the
    evaluation file) build_day is called once per month (a spy sees [3] then [4]) and the result
    equals each month built alone and concatenated, MVT_ID unique across the months. Unlike the
    queue block, the single-unit build coincides with it - a UTC day never straddles two calendar
    months - which is why the split can only be pinned structurally (the spy), and that
    coincidence is asserted rather than assumed. A row without MVT_TIME cannot be assigned to a
    month and is refused; the same ids in two months are refused.

    Fails when build_day_ranking builds the file as one unit (one spy call spanning both months;
    the OUTPUT would be identical, so on 2026-09-09 this mutation survived the value assertions
    alone and the spy was added), when a month is built in training mode, or when a NaT MVT_TIME
    is silently dropped. Rehearsed 2026-09-09, each RED: `parts = [build_day(t, serve=True)]`;
    removing the NaT check (the build then refuses with the wrong message).
    """
    calls = []
    real = stand_ab.build_day

    def spy(frame, serve=False):
        calls.append((sorted(frame.MVT_TIME_UTC_mvt.dt.month.unique().tolist()), serve))
        return real(frame, serve=serve)
    monkeypatch.setattr(stand_ab, "build_day", spy)
    m1 = _blank_departures(micro_fixture())
    m2 = _blank_departures(micro_fixture(shift_days=31, id_offset=1_000))
    two = pd.concat([m1, m2], ignore_index=True)
    got = stand_ab.build_day_ranking(_write(two, tmp_path / "ranking.parquet"))
    assert calls == [([3], True), ([4], True)], "the evaluation file must be built one calendar month at a time"
    want = pd.concat([stand_ab.build_day(m1, serve=True), stand_ab.build_day(m2, serve=True)], ignore_index=True)
    assert len(got) == len(want) == 2 * len(TAKEOFF_ORDER_SERVE) and got.MVT_ID_mvt.is_unique
    assert np.array_equal(got.MVT_ID_mvt.to_numpy(), want.MVT_ID_mvt.to_numpy())
    assert not [c for c in DAY_FEATS if not _same(got[c], want[c]).all()]
    single = stand_ab.build_day(two, serve=True)
    assert np.array_equal(single.MVT_ID_mvt.to_numpy(), want.MVT_ID_mvt.to_numpy())
    assert not [c for c in DAY_FEATS if not _same(single[c], want[c]).all()], \
        "a day aggregate crossed a month boundary"
    dup = pd.concat([m1, _blank_departures(micro_fixture(shift_days=31))], ignore_index=True)   # same ids twice
    with pytest.raises(AssertionError, match="duplicate MVT_ID"):
        stand_ab.build_day_ranking(_write(dup, tmp_path / "dup.parquet"))
    bad = two.copy()
    bad.loc[bad.index[0], "MVT_TIME_UTC_mvt"] = pd.NaT
    with pytest.raises(ValueError, match="calendar month"):
        stand_ab.build_day_ranking(_write(bad, tmp_path / "bad.parquet"))


# ---------------------------------------------------------------------------------------------
# 4. the cache command
# ---------------------------------------------------------------------------------------------

def test_cmd_day_cache_writes_the_contract_columns_and_refuses_a_smoke_ranking(tmp_path, monkeypatch):
    """`day-cache --smoke` writes exactly one file, named after the first training month,
    holding `MVT_ID_mvt` + DAY_FEATS (float32) for the rows build_day_month gives; `--ranking
    --smoke` is refused before anything is written; `--ranking` writes only ranking.parquet,
    equal to build_day_ranking (serve mode). Nothing lands outside the day cache dir.

    Fails when the column list is misspelt, when the smoke writes more than one month, when a
    smoke ranking cache reaches the real path, or when the command forgets serve mode for the
    ranking file. Rehearsed 2026-09-09, each RED: removing the `if smoke: raise ValueError`
    guard; `build_day_month(RAW / "ranking.parquet")` (training mode: the blanked file has no
    training row and the build refuses); `paths[:2]` for the smoke.
    """
    raw, cache = tmp_path / "raw", tmp_path / "cache_day"
    raw.mkdir()
    month = _write(micro_fixture(), raw / "training_2025-03-01_2025-04-01.parquet")
    _write(micro_fixture(shift_days=31, id_offset=1_000), raw / "training_2025-04-01_2025-05-01.parquet")
    two = pd.concat([_blank_departures(micro_fixture()),
                     _blank_departures(micro_fixture(shift_days=31, id_offset=1_000))], ignore_index=True)
    _write(two, raw / "ranking.parquet")
    monkeypatch.setattr(stand_ab, "RAW", raw)
    monkeypatch.setattr(stand_ab, "DCACHE", cache)
    monkeypatch.setattr(stand_ab, "CACHE", tmp_path / "must_not_be_touched")
    monkeypatch.setattr(stand_ab, "QCACHE", tmp_path / "must_not_be_touched_either")

    stand_ab.cmd_day_cache(smoke=True, ranking=False)
    assert sorted(p.name for p in cache.iterdir()) == ["training_2025-03-01_2025-04-01.parquet"]
    o = pd.read_parquet(cache / "training_2025-03-01_2025-04-01.parquet")
    assert list(o.columns) == ["MVT_ID_mvt"] + DAY_FEATS
    assert all(o[c].dtype == np.float32 for c in DAY_FEATS)
    want = stand_ab.build_day_month(month, serve=False)
    assert o.MVT_ID_mvt.tolist() == want.MVT_ID_mvt.tolist() == TAKEOFF_ORDER_TRAIN
    assert all(_same(o[c], want[c]).all() for c in DAY_FEATS)

    with pytest.raises(ValueError, match="smoke"):
        stand_ab.cmd_day_cache(smoke=True, ranking=True)
    assert sorted(p.name for p in cache.iterdir()) == ["training_2025-03-01_2025-04-01.parquet"]

    stand_ab.cmd_day_cache(smoke=False, ranking=True)
    assert sorted(p.name for p in cache.iterdir()) == ["ranking.parquet", "training_2025-03-01_2025-04-01.parquet"]
    r = pd.read_parquet(cache / "ranking.parquet")
    want_r = stand_ab.build_day_ranking(raw / "ranking.parquet")
    assert list(r.columns) == ["MVT_ID_mvt"] + DAY_FEATS and len(r) == 2 * len(TAKEOFF_ORDER_SERVE)
    assert np.array_equal(r.MVT_ID_mvt.to_numpy(), want_r.MVT_ID_mvt.to_numpy())
    assert all(_same(r[c], want_r[c]).all() for c in DAY_FEATS)
    assert not (tmp_path / "must_not_be_touched").exists() and not (tmp_path / "must_not_be_touched_either").exists()
    # the CLI knows the command
    assert "day-cache" in (ROOT / "scripts" / "stand_ab.py").read_text().split('ap.add_argument("cmd"')[1].split(")")[0]


# ---------------------------------------------------------------------------------------------
# 5. real rows. WRITTEN 2026-09-09 AND NOT YET RUN (see the module docstring).
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def march_pair():
    """March 2025 built both ways. Shared by the parity tests."""
    return (stand_ab.build_day_month(MARCH, serve=False), stand_ab.build_day_month(MARCH, serve=True))


@needs_data
def test_day_rows_are_exactly_the_stand_ab_reference_stream_in_both_modes(march_pair):
    """build_day's MVT_ID sequence equals build_features's, training and serve mode alike - the
    proof that the positional join of the day cache onto the v4 stand caches (which predate
    MVT_ID_mvt) is valid, exactly as tests/test_queue_features.py proves it for the queue cache.

    NOT YET RUN (needs ~1.3 GB for build_month on March; run after the live fold). Mutation to
    rehearse on the first run: requiring `proxy > 0` in `_dep_stream` (163,310 vs 163,361 rows).
    """
    for serve, mine in zip((False, True), march_pair):
        theirs = stand_ab.build_month(MARCH, serve=serve)
        assert len(mine) == len(theirs) > 150_000
        assert np.array_equal(mine.MVT_ID_mvt.to_numpy(), theirs.MVT_ID_mvt.to_numpy()), \
            f"serve={serve}: build_day's rows or order differ from build_features"


@needs_data
def test_serve_mode_differences_are_caused_only_by_the_extra_reference_rows_on_march(march_pair, tmp_path, capsys):
    """ARR-stream features (d_n_arr included) are byte-identical between modes; the DEP-stream
    levels and shares agree on >= 99% of rows (a day that gains a reference departure moves its
    percentiles only by their own resolution); the DEP-stream count d_n_dep moves, row by row,
    by EXACTLY the number of serve-only reference departures on that row's (airport, UTC day) -
    an identity, not a rate: every extra departure shifts every row of its airport-day by +1,
    and on March 2025 the 6 extra rows moved 1.5% of the 163,361 rows (98.49% agreement, which
    the 99% rate bar this identity replaced on 2026-09-09 read as a failure and which no rate bar
    can tell from a bug); and pruning the rows only the serve filter admits makes the two modes
    agree on EVERY column in row order.

    The expected shift is derived in the test from the raw rows - the airport and the floor of
    the epoch seconds over 86,400 - independently of stand_ab's day index, so a day boundary
    that drifts in the module is caught here, not mirrored. Fails when serve mode changes an
    arrival feature, when a level moves on more than 1% of rows, when the count shifts on a row
    whose airport-day gained no reference departure or by the wrong amount, when the module's
    day boundary leaves the UTC midnight, or when anything beyond the extra rows differs.
    Rehearsed 2026-09-09 on the module's own stand_ab instance (nothing under scripts/ is written
    while a fit runs), each RED at the count identity while every level stayed >= 99.24%: the
    day boundary moved from midnight to noon (`+ DAY_S / 2` inside _day_index) named 2,078 rows
    (serve - train 2 against 0 extra rows on the first); a serve-only +1 on d_n_dep for the
    first 500 rows (0.3% of rows - invisible to the 99% rate bar) named exactly those 500.
    """
    train, serve = march_pair
    assert len(serve) > len(train), "the serve filter admitted no extra rows on March"
    aligned = _aligned(train, serve)
    broken = [c for c in ARR_STREAM if not _same(train[c], aligned[c]).all()]
    assert not broken, f"serve mode changed arrival-stream features: {broken}"
    rates = {c: float(_same(train[c], aligned[c]).mean()) for c in DEP_LEVELS}
    extra_ids = set(serve.MVT_ID_mvt) - set(train.MVT_ID_mvt)
    t = pq.read_table(MARCH, columns=stand_ab.DCOLS).to_pandas()
    expected = _count_shift_from_extra_rows(t, extra_ids, train.MVT_ID_mvt.to_numpy())
    observed = _int_counts(aligned[DEP_COUNT]) - _int_counts(train[DEP_COUNT])
    print(f"\nday DEP-stream features, serve vs training on March 2025 ({len(train):,} rows; "
          f"{len(extra_ids)} extra reference rows admitted, on the airport-days of "
          f"{int((expected > 0).sum()):,} rows):")
    for c, r in rates.items():
        print(f"  {c:18s} equal on {100 * r:7.3f}% of rows")
    print(f"  {DEP_COUNT:18s} serve - train == the airport-day's extra rows on "
          f"{int((observed == expected).sum()):,} of {len(train):,} rows")
    low = {c: r for c, r in rates.items() if r < MIN_DAY_MATCH}
    assert not low, f"day levels below {MIN_DAY_MATCH:.0%} agreement: {low}"
    assert 0 < int((expected > 0).sum()) < len(train), \
        "the extra rows shift no row, or every row - the count identity proves nothing"
    wrong = np.flatnonzero(observed != expected)
    assert not len(wrong), (
        f"{DEP_COUNT} moved by other than the airport-day's serve-only reference rows on {len(wrong):,} rows "
        f"(first: id {train.MVT_ID_mvt.iloc[wrong[0]]:.0f}, serve - train {observed[wrong[0]]}, "
        f"extra rows on its airport-day {expected[wrong[0]]})")
    pruned = stand_ab.build_day_month(_write(t[~t.MVT_ID_mvt.isin(extra_ids)], tmp_path / "pruned.parquet"), serve=True)
    assert np.array_equal(pruned.MVT_ID_mvt.to_numpy(), train.MVT_ID_mvt.to_numpy())
    moved = [c for c in DAY_FEATS if not _same(train[c], pruned[c]).all()]
    assert not moved, f"serve mode differs beyond the extra reference rows on: {moved}"


@needs_data
def test_day_features_never_read_the_departure_own_hidden_clocks_on_real_rows(tmp_path):
    """The hidden-clock invariant on 1-4 January 2025: permuting or blanking departure
    BLOCK/TAXITIME leaves the serve build byte-identical; training mode's row set moves under
    the permutation (the control).

    NOT YET RUN (~0.4 GB expected: one raw-month read). Mutation to rehearse on the first run:
    `+ 1e-3 * es(d.BLOCK_TIME_UTC_mvt)` on d_prx_minus_p50.
    """
    base_path = _write(_slice(JAN, "2025-01-01", "2025-01-04"), tmp_path / "slice.parquet")
    base = stand_ab.build_day_month(base_path, serve=True)
    t = pq.read_table(base_path, columns=stand_ab.DCOLS).to_pandas()
    dep = (t.PHASE_mvt == "DEP").to_numpy()
    perm = np.random.default_rng(0).permutation(dep.sum())
    for col in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"):
        vals = t.loc[dep, col].to_numpy()
        t.loc[dep, col] = vals[perm]
    perm_path = _write(t, tmp_path / "permuted.parquet")
    blank_path = _write(_blank_departures(pq.read_table(base_path, columns=stand_ab.DCOLS).to_pandas()),
                        tmp_path / "blanked.parquet")
    for name, path in (("permuted", perm_path), ("blanked", blank_path)):
        served = stand_ab.build_day_month(path, serve=True)
        assert len(served) == len(base) > 5_000, f"{name}: the serve row set changed"
        assert np.array_equal(served.MVT_ID_mvt.to_numpy(), base.MVT_ID_mvt.to_numpy())
        moved = [c for c in DAY_FEATS if not _same(base[c], served[c]).all()]
        assert not moved, f"{name}: day features read the departure's own hidden clocks: {moved}"
    ref = stand_ab.build_day_month(base_path, serve=False)
    trained = stand_ab.build_day_month(perm_path, serve=False)
    assert set(trained.MVT_ID_mvt) != set(ref.MVT_ID_mvt), "the control proved nothing"


@needs_data
def test_ranking_build_yields_every_matched_row_with_training_like_nan_rates(march_pair, capsys):
    """Exactly the 339,551 scored departures with an NM off-block, the same id set as the raw
    file's matched rows, counts never NaN and never negative, per-column NaN rates within 2 pp
    of the March 2025 training rates (the arrival levels are NaN only on an airport-day without
    an arrival taxi-in, which is as rare on the scored file as in training).

    NOT YET RUN (~0.6 GB expected: the raw ranking file at DCOLS plus the March pair). Mutation
    to rehearse on the first run: dropping `t.AOBT_3_flt.notna()` from `_dep_stream`'s serve
    filter (344,841 rows).
    """
    train, _ = march_pair
    rank = stand_ab.build_day_ranking(RANKING)
    raw = pq.read_table(RANKING, columns=["MVT_ID_mvt", "PHASE_mvt", "AOBT_3_flt"]).to_pandas()
    ids = set(pq.read_table(TEMPLATE, columns=["MVT_ID_mvt"]).to_pandas().MVT_ID_mvt)
    matched = raw[(raw.PHASE_mvt == "DEP") & raw.AOBT_3_flt.notna() & raw.MVT_ID_mvt.isin(ids)]
    assert len(rank) == 339_551, f"expected 339,551 matched rows, got {len(rank):,}"
    assert rank.MVT_ID_mvt.is_unique and set(rank.MVT_ID_mvt) == set(matched.MVT_ID_mvt)
    assert list(rank.columns) == ["MVT_ID_mvt"] + DAY_FEATS
    print(f"\nNaN rate per day feature, training (March 2025, n={len(train):,}) vs ranking (n={len(rank):,}):")
    drift = {}
    for c in DAY_FEATS:
        r_tr, r_rk = float(train[c].isna().mean()), float(rank[c].isna().mean())
        if abs(r_rk - r_tr) > MAX_NAN_RATE_DRIFT:
            drift[c] = (r_tr, r_rk)
        print(f"  {c:18s} train {100 * r_tr:7.3f}%   ranking {100 * r_rk:7.3f}%{'   <-- DRIFT' if c in drift else ''}")
    assert not drift, f"NaN rate on the evaluation rows differs from training by > 2 pp: {drift}"
    for c in COUNTS + ["d_prx_p50", "d_prx_p90", "d_prx_le0", "d_prx_minus_p50"]:
        assert rank[c].notna().all() and train[c].notna().all(), f"{c} has NaN"
    for c in COUNTS:
        assert (rank[c] >= 0).all() and (train[c] >= 0).all(), f"negative count in {c}"
    assert (rank.d_n_dep >= 1).all(), "a row's own day counts at least the row itself"
