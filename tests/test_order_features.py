"""Amendment 22: the record-ordering block (stand_ab.build_order / ORDER_FEATS / order-cache).

Three invariants, each measured rather than asserted from the code:
  * the block is exactly the stand_ab departure stream (same rows, same take-off order) with
    three float32 columns, and every value is the hand computation on a micro fixture;
  * the reference line is per airport and per file, over EVERY departure with a take-off time -
    so a row's values are IDENTICAL in training and serve mode (the day block cannot promise
    this; the order block must);
  * nothing reads a departure's own hidden clocks (BLOCK, TAXITIME) beyond the training filter.
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
MARCH = RAW / "training_2025-03-01_2025-04-01.parquet"
RANKING = RAW / "ranking.parquet"

_spec = importlib.util.spec_from_file_location("stand_ab", ROOT / "scripts" / "stand_ab.py")
stand_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stand_ab)

needs_data = pytest.mark.skipif(not (MARCH.exists() and RANKING.exists()),
                                reason="challenge data is not redistributable and is absent from a clean checkout")

#: the contract, spelled out rather than read back from the module
ORDER_FEATS = ["o_dev_mvt", "o_dev_flt", "o_n_line"]
OCOLS = ["PHASE_mvt", "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
         "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"]
T0 = pd.Timestamp("2025-03-10 06:00:00", tz="UTC")
N_PER_APT = 80                  # >= the line's min_periods (50) and < its window (501): the line is the whole airport
PLANTED = 20                    # the EGLL departure whose MVT_ID is far off the line
PLANTED_ID = 5_000.0
NULL_FLT = 10                   # the EGLL departure without a FLIGHT_ID


def micro_fixture(blank_departure_clocks: bool = False) -> pd.DataFrame:
    """EGLL and LEMD: 80 departures each, five minutes apart, MVT_ID = base + k in take-off order
    except EGLL's row 20 (MVT_ID 5,000, an out-of-order record) and FLIGHT_ID null on EGLL's row
    10; one unmatched EGLL departure (no AOBT_3, on the line: it is a reference row, never a
    feature row); one unlabelled EGLL departure (no TAXITIME: a reference row in training mode, a
    feature row in serve mode); one arrival (ignored). Deliberately shuffled."""
    rows = []
    for apt, base in (("EGLL", 1_000.0), ("LEMD", 2_000.0)):
        for k in range(N_PER_APT):
            t = T0 + pd.Timedelta(minutes=5 * k)
            mid = PLANTED_ID if (apt == "EGLL" and k == PLANTED) else base + k
            fid = np.nan if (apt == "EGLL" and k == NULL_FLT) else 300.0 + base + k
            rows.append(dict(PHASE_mvt="DEP", MVT_ID_mvt=mid, FLIGHT_ID_mvt=fid, ADEP_mvt=apt, ADES_mvt="LFPG",
                             MVT_TIME_UTC_mvt=t, SCHED_TIME_UTC_mvt=t - pd.Timedelta(minutes=20),
                             BLOCK_TIME_UTC_mvt=t - pd.Timedelta(minutes=12), TAXITIME_SEC_mvt=720.0,
                             AOBT_3_flt=t - pd.Timedelta(minutes=11)))
    t = T0 + pd.Timedelta(minutes=5 * N_PER_APT + 2)
    rows.append(dict(PHASE_mvt="DEP", MVT_ID_mvt=1_000.0 + N_PER_APT, FLIGHT_ID_mvt=1_300.0 + N_PER_APT, ADEP_mvt="EGLL",
                     ADES_mvt="LFPG", MVT_TIME_UTC_mvt=t, SCHED_TIME_UTC_mvt=t, BLOCK_TIME_UTC_mvt=t - pd.Timedelta(minutes=9),
                     TAXITIME_SEC_mvt=540.0, AOBT_3_flt=pd.NaT))                                     # unmatched
    t = T0 + pd.Timedelta(minutes=5 * N_PER_APT + 7)
    rows.append(dict(PHASE_mvt="DEP", MVT_ID_mvt=1_001.0 + N_PER_APT, FLIGHT_ID_mvt=1_301.0 + N_PER_APT, ADEP_mvt="EGLL",
                     ADES_mvt="LFPG", MVT_TIME_UTC_mvt=t, SCHED_TIME_UTC_mvt=t, BLOCK_TIME_UTC_mvt=pd.NaT,
                     TAXITIME_SEC_mvt=np.nan, AOBT_3_flt=t - pd.Timedelta(minutes=10)))              # unlabelled
    rows.append(dict(PHASE_mvt="ARR", MVT_ID_mvt=9_000.0, FLIGHT_ID_mvt=9_300.0, ADEP_mvt="LFPG", ADES_mvt="EGLL",
                     MVT_TIME_UTC_mvt=T0, SCHED_TIME_UTC_mvt=T0, BLOCK_TIME_UTC_mvt=T0 + pd.Timedelta(minutes=8),
                     TAXITIME_SEC_mvt=480.0, AOBT_3_flt=pd.NaT))
    t = pd.DataFrame(rows)
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"):
        t[c] = pd.to_datetime(t[c], utc=True)
    t = t.iloc[np.random.default_rng(11).permutation(len(t))].reset_index(drop=True)
    if blank_departure_clocks:
        dep = (t.PHASE_mvt == "DEP").to_numpy()
        t.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
        t.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    return t[OCOLS]


def hand_line(ids: np.ndarray) -> tuple:
    """(deviation per id, n) when the whole airport fits in one window: (id - median) / max(p90 - p10, 1)."""
    med = np.median(ids)
    spread = max(np.percentile(ids, 90) - np.percentile(ids, 10), 1.0)
    return (ids - med) / spread, len(ids)


def _egll_reference_ids(train: bool) -> np.ndarray:
    ids = [PLANTED_ID if k == PLANTED else 1_000.0 + k for k in range(N_PER_APT)]
    return np.array(ids + [1_000.0 + N_PER_APT, 1_001.0 + N_PER_APT])     # the unmatched and the unlabelled rows are on the line


def test_order_feature_list_is_the_contract():
    """The module's list is exactly the contract, in this order, and OCOLS is the raw column set
    the builder reads (a superset of the day block's, plus FLIGHT_ID). Fails on any rename."""
    assert list(stand_ab.ORDER_FEATS) == ORDER_FEATS
    assert list(stand_ab.OCOLS) == OCOLS
    assert stand_ab.ORDER_WINDOW == 501 and stand_ab.ORDER_MIN_ROWS == 50


def test_micro_fixture_every_feature_equals_the_hand_computation_in_both_modes():
    """On the micro fixture the line is the whole airport (80 + 2 EGLL reference rows < the 501
    window, >= the 50 minimum), so every value equals the hand computation: the planted record
    sits > 30 spreads off the line, every other EGLL row within one, LEMD's own line is separate
    (a pooled line would put every LEMD row a thousand ids high), FLIGHT_ID's deviation is NaN on
    the null row and finite elsewhere, o_n_line is the airport's reference count. Rows are the
    stand_ab departure stream in take-off order: 79 + 1 matched EGLL rows and 80 LEMD rows in
    training mode (the unlabelled one excluded), the unlabelled row joining in serve mode. And the
    values of every shared row are identical between the modes.

    Fails when the line is pooled across airports, when the reference stream excludes the unmatched
    or unlabelled rows, when the spread floor is dropped, or when serve mode recomputes the line
    from feature rows only. Rehearsed 2026-09-09, each RED: grouping on nothing instead of the
    airport; `_order_stream` filtered on AOBT_3; the `max(spread, 1.0)` floor removed; the serve
    reference taken from `_dep_stream`.
    """
    t = micro_fixture()
    o = stand_ab.build_order(t, serve=False)
    assert list(o.columns) == ["MVT_ID_mvt"] + ORDER_FEATS and all(o[c].dtype == np.float32 for c in ORDER_FEATS)
    want_rows = stand_ab._dep_stream(t, False).MVT_ID_mvt.to_numpy()
    assert o.MVT_ID_mvt.tolist() == want_rows.tolist() and len(o) == 2 * N_PER_APT   # 80 EGLL (unlabelled excluded, unmatched excluded) + 80 LEMD
    ref = _egll_reference_ids(True)
    dev, n = hand_line(ref)
    by_id = dict(zip(ref, dev))
    egll = o[o.MVT_ID_mvt.isin(ref)]
    assert len(egll) == N_PER_APT
    for mid, v, nl in zip(egll.MVT_ID_mvt, egll.o_dev_mvt, egll.o_n_line):
        assert v == pytest.approx(by_id[mid], rel=1e-6, abs=1e-6) and nl == n       # float32 of a float64 hand value
    planted = egll[egll.MVT_ID_mvt == PLANTED_ID]
    assert len(planted) == 1 and float(planted.o_dev_mvt.iloc[0]) > 30
    assert (egll[egll.MVT_ID_mvt != PLANTED_ID].o_dev_mvt.abs() < 1.0).all()
    lemd = o[~o.MVT_ID_mvt.isin(ref)]
    dev_l, n_l = hand_line(np.array([2_000.0 + k for k in range(N_PER_APT)]))
    assert (lemd.o_dev_mvt.abs() < 1.0).all() and (lemd.o_n_line == n_l).all()
    null_id = 1_000.0 + NULL_FLT
    assert np.isnan(float(egll[egll.MVT_ID_mvt == null_id].o_dev_flt.iloc[0]))
    assert egll[egll.MVT_ID_mvt != null_id].o_dev_flt.notna().all() and lemd.o_dev_flt.notna().all()
    # the planted MVT_ID row has an ordinary FLIGHT_ID: only the MVT line flags it
    assert abs(float(planted.o_dev_flt.iloc[0])) < 1.0
    s = stand_ab.build_order(micro_fixture(blank_departure_clocks=True), serve=True)
    assert s.MVT_ID_mvt.tolist() == stand_ab._dep_stream(micro_fixture(blank_departure_clocks=True), True).MVT_ID_mvt.to_numpy().tolist()
    assert len(s) == len(o) + 1, "serve mode admits the unlabelled row"
    shared = s.set_index("MVT_ID_mvt").loc[o.MVT_ID_mvt.to_numpy()]
    for c in ORDER_FEATS:
        a, b = o[c].to_numpy(), shared[c].to_numpy()
        assert np.array_equal(a, b, equal_nan=True), f"{c} differs between the modes"


def test_order_line_floor_and_window_count_on_degenerate_sequences():
    """_order_line on a constant sequence (spread 0) gives deviation 0 everywhere, not NaN - the
    spread floor of 1 id; on 60 ascending ids the window count is 60 for every row (the whole
    sequence fits) and the deviations are antisymmetric about the middle; NaN ids get NaN and
    leave the count of non-null ids alone. Fails when the floor is removed (0 / 0). Rehearsed
    2026-09-09, RED: `/ spread` without the floor."""
    dev, n = stand_ab._order_line(np.full(60, 7.0))
    assert np.array_equal(dev, np.zeros(60)) and np.array_equal(n, np.full(60, 60.0))
    dev, n = stand_ab._order_line(np.arange(60, dtype=float))
    assert np.array_equal(n, np.full(60, 60.0)) and np.allclose(dev, -dev[::-1], atol=1e-12)
    ids = np.arange(60, dtype=float); ids[5] = np.nan
    dev, n = stand_ab._order_line(ids)
    assert np.isnan(dev[5]) and np.isfinite(np.delete(dev, 5)).all() and np.array_equal(n, np.full(60, 60.0))


def test_build_order_refuses_an_empty_departure_stream_and_a_row_without_a_take_off_time():
    """No admissible departure -> ValueError naming the mode; a departure without MVT_TIME cannot
    sit on a time line and is refused rather than silently placed. Fails when the guard is dropped
    or the NaT row is sorted to an end and given a deviation. Rehearsed 2026-09-09, each RED."""
    t = micro_fixture()
    with pytest.raises(ValueError, match="no admissible departures"):
        stand_ab.build_order(t[t.PHASE_mvt == "ARR"], serve=False)
    bad = t.copy()
    i = bad.index[(bad.PHASE_mvt == "DEP") & (bad.ADEP_mvt == "LEMD")][0]
    bad.loc[i, "MVT_TIME_UTC_mvt"] = pd.NaT
    with pytest.raises(ValueError, match="MVT_TIME"):
        stand_ab.build_order(bad, serve=False)


def test_order_features_never_read_the_departure_own_hidden_clocks():
    """Blanking every departure's BLOCK and TAXITIME and building in serve mode changes no value
    of any row present in both builds (the training filter reads them; no feature does).
    Fails when a feature reads either clock. Rehearsed 2026-09-09, RED: o_n_line counting rows
    with a TAXITIME."""
    a = stand_ab.build_order(micro_fixture(), serve=False)
    b = stand_ab.build_order(micro_fixture(blank_departure_clocks=True), serve=True).set_index("MVT_ID_mvt")
    for c in ORDER_FEATS:
        assert np.array_equal(a[c].to_numpy(), b.loc[a.MVT_ID_mvt.to_numpy(), c].to_numpy(), equal_nan=True)


def _write(frame: pd.DataFrame, path: pathlib.Path) -> pathlib.Path:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    return path


def test_cmd_order_cache_writes_the_contract_columns_and_refuses_a_smoke_ranking(tmp_path, monkeypatch):
    """`order-cache --smoke` writes exactly one file named after the first training month with
    MVT_ID_mvt + ORDER_FEATS (float32) equal to build_order_month; `--ranking --smoke` is refused
    before anything is written; `--ranking` writes only ranking.parquet, equal to
    build_order_ranking (serve mode, per calendar month, unique ids asserted). Nothing lands
    outside the order cache dir. Fails when the smoke writes two months, when the ranking cache
    is built in training mode (the blanked file has no training row and the build refuses), or
    when the guard is dropped. Rehearsed 2026-09-09, each RED."""
    raw, cache = tmp_path / "raw", tmp_path / "cache_order"
    raw.mkdir()
    month = _write(micro_fixture(), raw / "training_2025-03-01_2025-04-01.parquet")
    second = micro_fixture(); second["MVT_ID_mvt"] += 10_000; second["FLIGHT_ID_mvt"] += 10_000
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"):
        second[c] = second[c] + pd.Timedelta(days=31)
    _write(second, raw / "training_2025-04-01_2025-05-01.parquet")
    _write(pd.concat([micro_fixture(blank_departure_clocks=True), second.assign(BLOCK_TIME_UTC_mvt=pd.NaT, TAXITIME_SEC_mvt=np.nan)],
                     ignore_index=True), raw / "ranking.parquet")
    monkeypatch.setattr(stand_ab, "RAW", raw)
    monkeypatch.setattr(stand_ab, "OCACHE", cache)
    monkeypatch.setattr(stand_ab, "CACHE", tmp_path / "must_not_be_touched")
    stand_ab.cmd_order_cache(smoke=True, ranking=False)
    assert sorted(p.name for p in cache.iterdir()) == ["training_2025-03-01_2025-04-01.parquet"]
    o = pd.read_parquet(cache / "training_2025-03-01_2025-04-01.parquet")
    want = stand_ab.build_order_month(month, serve=False)
    assert list(o.columns) == ["MVT_ID_mvt"] + ORDER_FEATS and all(o[c].dtype == np.float32 for c in ORDER_FEATS)
    assert o.MVT_ID_mvt.tolist() == want.MVT_ID_mvt.tolist()
    assert all(np.array_equal(o[c].to_numpy(), want[c].to_numpy(), equal_nan=True) for c in ORDER_FEATS)
    with pytest.raises(ValueError, match="smoke"):
        stand_ab.cmd_order_cache(smoke=True, ranking=True)
    assert sorted(p.name for p in cache.iterdir()) == ["training_2025-03-01_2025-04-01.parquet"]
    stand_ab.cmd_order_cache(smoke=False, ranking=True)
    r = pd.read_parquet(cache / "ranking.parquet")
    wr = stand_ab.build_order_ranking(raw / "ranking.parquet")
    assert r.MVT_ID_mvt.tolist() == wr.MVT_ID_mvt.tolist() and r.MVT_ID_mvt.is_unique
    assert len(r) == 2 * (2 * N_PER_APT + 1), "serve mode: both months, the unlabelled rows admitted"
    assert all(np.array_equal(r[c].to_numpy(), wr[c].to_numpy(), equal_nan=True) for c in ORDER_FEATS)
    assert not (tmp_path / "must_not_be_touched").exists()


@needs_data
def test_order_rows_are_exactly_the_stand_ab_reference_stream_and_identical_across_modes_on_march(tmp_path):
    """Real March: the training build's rows are build_month's rows in its order; the serve build
    on the same file with the departures' clocks blanked admits the serve stream; every row
    present in both carries IDENTICAL values (the line reads MVT_ID, FLIGHT_ID and MVT_TIME of
    every departure, blanked or not); o_n_line is 501 on interior rows and >= 50 everywhere;
    FLIGHT_ID's deviation is NaN exactly where FLIGHT_ID is null. Fails when the reference stream
    depends on the mode or when the window/min-periods drift."""
    t = pq.read_table(MARCH, columns=OCOLS).to_pandas()
    o = stand_ab.build_order(t, serve=False)
    ref = stand_ab.build_month(MARCH)
    assert o.MVT_ID_mvt.tolist() == ref.MVT_ID_mvt.tolist() if "MVT_ID_mvt" in ref.columns else len(o) == len(ref)
    dep = (t.PHASE_mvt == "DEP").to_numpy()
    s = t.copy(); s.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT; s.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    so = stand_ab.build_order(s, serve=True).set_index("MVT_ID_mvt")
    for c in ORDER_FEATS:
        assert np.array_equal(o[c].to_numpy(), so.loc[o.MVT_ID_mvt.to_numpy(), c].to_numpy(), equal_nan=True), c
    assert o.o_n_line.min() >= 50 and (o.o_n_line == 501).mean() > 0.9
    fid = t.set_index("MVT_ID_mvt").FLIGHT_ID_mvt.loc[o.MVT_ID_mvt.to_numpy()]
    assert np.array_equal(o.o_dev_flt.isna().to_numpy(), fid.isna().to_numpy())
    assert o.o_dev_mvt.notna().all() and o.o_dev_mvt.abs().median() < 1.0


@needs_data
def test_ranking_order_build_yields_every_matched_scored_row(tmp_path):
    """The serve build on the real scored file yields one row per matched scored departure (the
    stand cache's ranking rows), unique ids, finite o_dev_mvt everywhere, and FLIGHT_ID's
    deviation NaN on the rows without a FLIGHT_ID at a rate within 2 pp of the training months'."""
    r = stand_ab.build_order_ranking(RANKING)
    stand_rank = pd.read_parquet(stand_ab.CACHE / "ranking.parquet", columns=["MVT_ID_mvt"])
    assert r.MVT_ID_mvt.is_unique and set(stand_rank.MVT_ID_mvt) <= set(r.MVT_ID_mvt)
    assert r.o_dev_mvt.notna().all() and r.o_n_line.min() >= 50
    train = stand_ab.build_order_month(MARCH)
    assert abs(r.o_dev_flt.isna().mean() - train.o_dev_flt.isna().mean()) < 0.02
