"""Tests for ARN (scripts/analysis/rome_arn.py; plans/PREREG_rome_arcn_xj_2026_09_16.md).

Written before scripts/analysis/rome_arn.py existed. Run at that point (`/opt/homebrew/bin/python3.11 -B -m pytest
tests/test_rome_arn.py -q`, with scripts/analysis/rome_arn.py moved aside): every test failed with
`ModuleNotFoundError: No module named 'rome_arn'` (collection error) — the RED evidence for "seen failing before the
module exists". Verbatim log: reports/rome_arn_2026_09_16.console.log (RED section) and the session transcript.

Mutation rehearsal (2026-09-16, __pycache__ cleared before each run): 8 mutants, 8/8 caught by this file alone —
1) norm(): drop the `group(1).endswith(group(2))` guard (always substitute) -> caught by
   test_norm_group1_does_not_end_with_group2_unchanged.
2) norm(): `{1,2}` -> `{1,1}` in NORM_RE (only ever a single-letter group2) -> caught by
   test_norm_cases (WMTMT306/ITYTY680 need a 2-letter group2).
3) in_scope(): `sp < SP_HI` -> `sp <= SP_HI` -> caught by test_in_scope_boundaries (24,000 case).
4) in_scope(): `sp >= SP_LO` -> `sp > SP_LO` -> caught by test_in_scope_boundaries (3,600 case).
5) arcn_join(): drop the FLIGHT_mvt normalisation (`dep_n["FLIGHT_mvt"] = dep.FLIGHT_mvt...` i.e. a no-op) -> caught
   by test_arcn_join_normalises_before_arcs_own_join.
6) xj_join(): `dt >= 0.0` -> `dt > 0.0` -> caught by test_xj_join_window_inclusive_both_ends.
7) xj_join(): drop the `h.host_fn != h.CALLSIGN_flt` filter (host self-consistency guard) -> caught by
   test_xj_join_host_must_disagree_with_own_callsign.
8) verdict(): swap the `n_wrong > 0` precedence below the `n_held < 2` check (so a single wrong call at n_held=1
   reports INCONCLUSIVE instead of NOT WORKING) -> caught by test_verdict.
All 8 reverted after their catch was confirmed; `diff scripts/analysis/rome_arn.py <pre-mutation copy>` byte-identical
afterwards (the file is new/untracked this session, so `git diff` itself shows nothing either way).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import rome_arn as N  # noqa: E402


# --------------------------------------------------------------------------- norm()

@pytest.mark.parametrize("s,want", [
    ("RYRR88TM", "RYR88TM"),
    ("WMTMT306", "WMT306"),
    ("ITYTY680", "ITY680"),
    ("AZA123", "AZA123"),      # no letters between the 3-letter prefix and the first digit -> regex doesn't match
    ("BAW12A", "BAW12A"),      # same: digit immediately follows the 3-letter prefix
])
def test_norm_cases(s, want):
    assert N.norm(s) == want


@pytest.mark.parametrize("s", [None, 123, 123.0, np.nan, ["RYR88TM"]])
def test_norm_non_string_is_none(s):
    assert N.norm(s) is None


def test_norm_group1_does_not_end_with_group2_unchanged():
    """'ABCXY123' matches the regex structurally (group1='ABC', group2='XY', group3='123') but 'ABC' does not end
    with 'XY' -> unchanged. Fails if the endswith guard is dropped (mutant 1)."""
    assert N.norm("ABCXY123") == "ABCXY123"


def test_norm_arr_vectorised():
    out = N.norm_arr(["RYRR88TM", None, "AZA123", 5])
    assert list(out) == ["RYR88TM", None, "AZA123", None]


# --------------------------------------------------------------------------- in_scope()

@pytest.mark.parametrize("aobt3_own,sp,want", [
    (np.nan, 3_600.0, True),        # lower bound inclusive
    (np.nan, 3_599.999, False),
    (np.nan, 24_000.0, False),      # upper bound exclusive
    (np.nan, 23_999.999, True),
    (0.0, 10_000.0, False),         # matched (aobt3_own not null) -> never in scope
    (np.nan, np.nan, False),        # sp missing
])
def test_in_scope_boundaries(aobt3_own, sp, want):
    assert N.in_scope([aobt3_own], [sp]).tolist() == [want]


# --------------------------------------------------------------------------- arcn_join()

def _dep():
    return pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "ADEP_mvt": ["LIRF", "LIRF"], "ADES_mvt": ["EDDF", "EDDF"],
                         "FLIGHT_mvt": ["RYRR88TM", "AZA123"], "mvt": [10_000.0, 10_000.0]})


def _arr():
    return pd.DataFrame({"ADEP_flt": ["LIRF", "LIRF"], "ADES_mvt": ["EDDF", "EDDF"],
                         "CALLSIGN_flt": ["RYR88TM", "AZA123"], "mvt": [10_000 + 3_000.0, 10_000 + 3_000.0],
                         "lobt": [111.0, 222.0], "aobt3": [1_111.0, 2_222.0]})


def test_arcn_join_normalises_before_arcs_own_join():
    """RYRR88TM's raw designator does not equal any ARR CALLSIGN_flt (ARC's own join misses it); normalised to
    RYR88TM it matches row 0. AZA123 is unchanged by norm() and matches row 1 directly either way. Fails if the
    normalisation is dropped (mutant 5)."""
    j = N.arcn_join(_dep(), _arr())
    assert j.arr_lobt.tolist() == [111.0, 222.0]
    assert j.arr_aobt3.tolist() == [1_111.0, 2_222.0]
    # ARC's own (unnormalised) join misses row 0 entirely
    import arr_row_clocks as ARC
    j0 = ARC.join_arr(_dep(), _arr())
    assert np.isnan(j0.arr_lobt[0])


def test_arcn_join_inherits_window_and_earliest_wins():
    """Window and tie-break come from `join_arr` itself — ARC-N does not reimplement them."""
    dep = pd.DataFrame({"MVT_ID_mvt": [1.0], "ADEP_mvt": ["LIRF"], "ADES_mvt": ["EDDF"],
                        "FLIGHT_mvt": ["RYRR88TM"], "mvt": [10_000.0]})
    arr = pd.DataFrame({"ADEP_flt": ["LIRF", "LIRF", "LIRF"], "ADES_mvt": ["EDDF", "EDDF", "EDDF"],
                        "CALLSIGN_flt": ["RYR88TM", "RYR88TM", "RYR88TM"],
                        "mvt": [10_000 + 600.0, 10_000 + 5_000.0, 10_000 + 3_000.0],   # first at boundary (excluded)
                        "lobt": [999.0, 222.0, 333.0], "aobt3": [9_999.0, 2_222.0, 3_333.0]})
    j = N.arcn_join(dep, arr)
    assert j.arr_lobt.tolist() == [333.0]     # earliest of the two in-window candidates (3,000 s), not 999


def test_arcn_join_missing_lobt_or_aobt3_excluded():
    dep = pd.DataFrame({"MVT_ID_mvt": [1.0], "ADEP_mvt": ["LIRF"], "ADES_mvt": ["EDDF"],
                        "FLIGHT_mvt": ["RYRR88TM"], "mvt": [10_000.0]})
    arr = pd.DataFrame({"ADEP_flt": ["LIRF"], "ADES_mvt": ["EDDF"], "CALLSIGN_flt": ["RYR88TM"],
                        "mvt": [13_000.0], "lobt": [np.nan], "aobt3": [1_111.0]})
    assert np.isnan(N.arcn_join(dep, arr).arr_lobt[0])


# --------------------------------------------------------------------------- xj_join()

def _host(**over):
    base = dict(FLIGHT_mvt=["RYRR88TM"], CALLSIGN_flt=["RYR88TM_OTHER"], ADES_flt=["EDDF"], LOBT_flt=[500.0],
                AOBT_3_flt=[1_000.0], src=[1])
    base.update(over)
    return pd.DataFrame(base)


def _x(**over):
    base = dict(MVT_ID_mvt=[1.0], fn=["RYR88TM_OTHER"], ADES_mvt=["EDDF"], mvt=[1_500.0], src=[1])
    base.update(over)
    return pd.DataFrame(base)


def test_xj_join_resolves_a_single_qualifying_host():
    """Host's own norm(FLIGHT_mvt)='RYR88TM' != its CALLSIGN_flt='RYR88TM_OTHER' -> qualifies; CALLSIGN_flt matches
    X.fn; ADES agree; dt = 1,500 - 1,000 = 500, in [0, 10,800]."""
    j = N.xj_join(_x(), _host())
    assert j.xj_lobt.tolist() == [500.0] and j.xj_aobt3.tolist() == [1_000.0]


def test_xj_join_host_must_disagree_with_own_callsign():
    """Host whose own norm(FLIGHT_mvt) equals its CALLSIGN_flt never qualifies, even if everything else matches.
    Fails if this guard is dropped (mutant 7)."""
    h = _host(FLIGHT_mvt=["RYR88TM_OTHER"])   # norm() no-op, equals CALLSIGN_flt already
    j = N.xj_join(_x(), h)
    assert np.isnan(j.xj_lobt[0])


def test_xj_join_ades_mismatch_untouched():
    j = N.xj_join(_x(ADES_mvt=["LEMD"]), _host())
    assert np.isnan(j.xj_lobt[0])


@pytest.mark.parametrize("dt,ok", [(0.0, True), (10_800.0, True), (-0.001, False), (10_800.001, False)])
def test_xj_join_window_inclusive_both_ends(dt, ok):
    """Fails if either bound flips exclusive (mutant 6 for the lower bound)."""
    h = _host(AOBT_3_flt=[1_500.0 - dt])
    j = N.xj_join(_x(mvt=[1_500.0]), h)
    assert (not np.isnan(j.xj_lobt[0])) == ok


def test_xj_join_exactly_one_host_else_untouched():
    h2 = pd.concat([_host(), _host(FLIGHT_mvt=["WMTMT306"], CALLSIGN_flt=["RYR88TM_OTHER"], LOBT_flt=[777.0],
                                    AOBT_3_flt=[1_200.0])], ignore_index=True)
    j = N.xj_join(_x(), h2)
    assert np.isnan(j.xj_lobt[0])          # two qualifying hosts -> untouched
    j1 = N.xj_join(_x(), _host())
    assert not np.isnan(j1.xj_lobt[0])     # exactly one -> resolved


def test_xj_join_different_src_never_matches():
    """The 'same file' rule: a host in a different src never qualifies."""
    j = N.xj_join(_x(src=[1]), _host(src=[2]))
    assert np.isnan(j.xj_lobt[0])


def test_xj_join_x_fn_none_never_joins():
    j = N.xj_join(_x(fn=[None]), _host())
    assert np.isnan(j.xj_lobt[0])


def test_xj_join_refuses_duplicate_x_ids():
    x2 = pd.concat([_x(), _x()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate MVT_ID_mvt"):
        N.xj_join(x2, _host())


# --------------------------------------------------------------------------- rule_value(), sub_rule (ARC's, reused)

def test_rule_value_ord_and_fill():
    rule = np.array(["ord", "fill", "none", ""])
    mvt = np.array([10_000.0, 10_000.0, 10_000.0, 10_000.0])
    aobt3 = np.array([9_000.4, 0.0, 0.0, 0.0])
    sp = np.array([4_000.0, 5_000.6, 6_000.0, 7_000.0])
    out = N.rule_value(rule, mvt, aobt3, sp)
    assert out[0] == pytest.approx(1_000.0) and out[1] == pytest.approx(5_001.0)
    assert np.isnan(out[2]) and np.isnan(out[3])


def test_rule_value_ord_floors_at_one():
    out = N.rule_value(["ord"], [10_000.0], [10_500.0], [4_000.0])   # AOBT_3 after take-off
    assert out[0] == 1.0


@pytest.mark.parametrize("sp,sched,lobt,aobt3,want", [
    (5_000.0, 0.0, 3_600.0, 99_999.0, "ord"),          # |lobt-sched| == 3,600 -> ord (inclusive)
    (5_000.0, 0.0, 3_600.5, 99_999.0, "none"),          # 3,600.5 -> not ord (registration's own boundary example)
    (5_000.0, 0.0, 3_601.0, 3_601.0 + 3_606, "fill"),   # |aobt3-lobt| == 3,606 -> fill (inclusive)
    (5_000.0, 0.0, 3_601.0, 3_601.0 + 3_607, "none"),   # 3,607 -> untouched ('none': joined, in scope, no rule)
])
def test_sub_rule_boundaries_arn_relies_on(sp, sched, lobt, aobt3, want):
    """ARN reuses `arr_row_clocks.sub_rule` unchanged (registration: 'ARC's, unchanged'). This pins the exact
    boundary values the registration's sub-rules section names, directly against ARC's own function."""
    import arr_row_clocks as ARC
    assert ARC.sub_rule([sp], [sched], [lobt], [aobt3]).tolist() == [want]


# --------------------------------------------------------------------------- call_correct(), verdict()

def test_call_correct():
    rule = np.array(["ord", "ord", "fill", "fill", "none"])
    cls = np.array(["ordinary", "fill", "fill", "ordinary", "ordinary"])
    assert N.call_correct(rule, cls).tolist() == [True, False, True, False, False]


@pytest.mark.parametrize("n_held,n_wrong,want", [
    (0, 0, "INCONCLUSIVE"),
    (1, 0, "INCONCLUSIVE"),
    (1, 1, "NOT WORKING"),         # a single wrong call is decisive even though n_held < 2 (mutant 8)
    (2, 0, "WORKING"),
    (2, 1, "NOT WORKING"),
    (4, 0, "WORKING"),
])
def test_verdict(n_held, n_wrong, want):
    assert N.verdict(n_held, n_wrong) == want


# --------------------------------------------------------------------------- gain_sse(), stake_alt()

def test_gain_sse_matches_arc_pattern():
    assert N.gain_sse([100.0], [90.0], [100.0]).tolist() == [100.0]     # (100-90)^2 - 0
    assert N.gain_sse([100.0], [90.0], [90.0]).tolist() == [0.0]        # pred==inc -> no gain


def test_stake_alt_fill_uses_mvt_minus_aobt3_else_v19():
    rule = np.array(["fill", "fill", "ord"])
    mvt = np.array([10_000.0, 10_000.0, 10_000.0])
    aobt3 = np.array([9_000.0, np.nan, 9_000.0])
    sp = np.array([5_000.0, 5_000.0, 5_000.0])
    v19 = np.array([1_234.0, 1_234.0, 1_234.0])
    out = N.stake_alt(rule, mvt, aobt3, sp, v19)
    assert out[0] == pytest.approx(1_000.0)      # mvt - aobt3
    assert out[1] == pytest.approx(1_234.0)      # aobt3 missing -> v19
    assert out[2] == pytest.approx(5_000.0)      # ord -> rint(sp)


# --------------------------------------------------------------------------- rule_rows() end-to-end on synthetic raw

def _raw_2025(rows: list[dict]) -> pd.DataFrame:
    """Builds a raw-shaped DataFrame with the columns rule_rows()/lirf_frame() read. Every row not overridden is a
    plain non-LIRF filler so PHASE_mvt=='ARR' pooling and dep_frame's `isin` scans have something realistic to skip."""
    base = dict(MVT_ID_mvt=0.0, PHASE_mvt="DEP", ADEP_mvt="EHAM", ADES_mvt="EHAM", FLIGHT_mvt="XXX000",
                MVT_TIME_UTC_mvt=pd.Timestamp("2025-02-10 10:00:00", tz="UTC"),
                SCHED_TIME_UTC_mvt=pd.Timestamp("2025-02-10 08:00:00", tz="UTC"), ADEP_flt="EHAM", ADES_flt="EHAM",
                CALLSIGN_flt="XXX000", LOBT_flt=pd.NaT, AOBT_3_flt=pd.NaT, TAXITIME_SEC_mvt=np.nan)
    recs = [{**base, **r} for r in rows]
    df = pd.DataFrame.from_records(recs)
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def test_rule_rows_arcn_only_touches_rows_arc_itself_misses():
    """A candidate whose RAW designator exactly matches an ARR CALLSIGN_flt is reached by ARC's own join and must
    never be touched by ARC-N/XJ, even though its scope conditions otherwise qualify. A second candidate whose raw
    designator ARC misses, but whose *normalised* designator matches, is resolved by ARC-N."""
    t0 = pd.Timestamp("2025-02-10 08:00:00", tz="UTC")
    rows = [
        # candidate reached by ARC directly (exact CALLSIGN_flt match) -> must stay untouched by ARN
        dict(MVT_ID_mvt=1.0, ADEP_mvt="LIRF", ADES_mvt="EDDF", FLIGHT_mvt="RYR88TM",
             MVT_TIME_UTC_mvt=t0 + pd.Timedelta(seconds=5_000), SCHED_TIME_UTC_mvt=t0, TAXITIME_SEC_mvt=5_000.0),
        dict(MVT_ID_mvt=2.0, PHASE_mvt="ARR", ADEP_flt="LIRF", ADES_mvt="EDDF", CALLSIGN_flt="RYR88TM",
             MVT_TIME_UTC_mvt=t0 + pd.Timedelta(seconds=5_000 + 3_000),
             LOBT_flt=t0 + pd.Timedelta(seconds=1_000), AOBT_3_flt=t0 + pd.Timedelta(seconds=5_500)),
        # candidate ARC misses (raw != any CALLSIGN_flt) but ARC-N resolves via normalisation
        dict(MVT_ID_mvt=3.0, ADEP_mvt="LIRF", ADES_mvt="LEMD", FLIGHT_mvt="RYRR88TM",
             MVT_TIME_UTC_mvt=t0 + pd.Timedelta(seconds=5_000), SCHED_TIME_UTC_mvt=t0, TAXITIME_SEC_mvt=5_000.0),
        dict(MVT_ID_mvt=4.0, PHASE_mvt="ARR", ADEP_flt="LIRF", ADES_mvt="LEMD", CALLSIGN_flt="RYR88TM",
             MVT_TIME_UTC_mvt=t0 + pd.Timedelta(seconds=5_000 + 3_000),
             LOBT_flt=t0 + pd.Timedelta(seconds=1_000), AOBT_3_flt=t0 + pd.Timedelta(seconds=5_500)),
    ]
    raw = _raw_2025(rows)
    out = N.rule_rows(raw, src=1)
    got = dict(zip(out.MVT_ID_mvt, out.method))
    assert 1.0 not in got                 # ARC reached it -> ARN must never touch MVT_ID 1
    assert got.get(3.0) == "ARC-N"


def test_rule_rows_xj_only_after_arc_and_arcn_both_miss():
    """A candidate that neither ARC's raw join nor ARC-N can join (no ARR row at all) but that XJ resolves via a
    matched sibling LIRF departure H in the same file."""
    t0 = pd.Timestamp("2025-02-10 08:00:00", tz="UTC")
    rows = [
        # H: a matched LIRF departure whose own norm(FLIGHT_mvt) != its CALLSIGN_flt
        dict(MVT_ID_mvt=10.0, ADEP_mvt="LIRF", ADES_mvt="EDDF", FLIGHT_mvt="RYRR88TM", ADES_flt="EDDF",
             CALLSIGN_flt="RYR88TM_H", MVT_TIME_UTC_mvt=t0 + pd.Timedelta(seconds=2_000),
             SCHED_TIME_UTC_mvt=t0, LOBT_flt=t0 + pd.Timedelta(seconds=100),
             AOBT_3_flt=t0 + pd.Timedelta(seconds=1_000), TAXITIME_SEC_mvt=2_000.0),
        # X: unmatched LIRF departure whose normalised designator matches H's CALLSIGN_flt; no ARR row exists at all
        dict(MVT_ID_mvt=11.0, ADEP_mvt="LIRF", ADES_mvt="EDDF", FLIGHT_mvt="RYR88TM_H",
             MVT_TIME_UTC_mvt=t0 + pd.Timedelta(seconds=5_000), SCHED_TIME_UTC_mvt=t0, TAXITIME_SEC_mvt=5_000.0),
    ]
    raw = _raw_2025(rows)
    out = N.rule_rows(raw, src=1)
    got = dict(zip(out.MVT_ID_mvt, out.method))
    assert got.get(11.0) == "XJ"


def test_rule_rows_empty_scope_returns_empty_frame():
    raw = _raw_2025([dict(MVT_ID_mvt=1.0, ADEP_mvt="EHAM")])   # not LIRF -> no candidates at all
    out = N.rule_rows(raw, src=1)
    assert len(out) == 0
    assert list(out.columns) == list(N.RULED_DTYPES.keys())


# --------------------------------------------------------------------------- real-data: 2026 ARN never touches ARC's own 42

@pytest.mark.devtree
def test_rows_2026_never_touches_arcs_42_shipped_rows():
    """SLOW (real data: reads ranking.parquet's needed columns, ~0.5-0.8 s; all other tests in this file are pure/
    synthetic and run in well under 1 s). ARC's own raw join changed 42 rows between v16 and v17 (submissions/merry-quicksand_v{16,17}.parquet) — the
    real, shipped result of `arr_row_clocks.main()`'s ARC rule on 2026's ranking.parquet, computed exactly (single
    file, no per-file approximation, unlike the 2025 pass). This directly tests the registration's 'NOT reached by
    ARC's raw join' scope clause against ARC's own real production output, not just against a synthetic mock of
    `join_arr`: whatever `rows_2026()` rules must be disjoint from what ARC itself already ruled."""
    v16 = pd.read_parquet(ROOT / "submissions" / "merry-quicksand_v16.parquet").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    v17 = pd.read_parquet(ROOT / "submissions" / "merry-quicksand_v17.parquet").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    arc_touched = set(v16.index[(v16 != v17).to_numpy()])
    assert len(arc_touched) == 42
    r26 = N.rows_2026()
    assert set(r26.MVT_ID_mvt.tolist()).isdisjoint(arc_touched)
