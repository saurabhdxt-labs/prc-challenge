"""The unmatched cache for the unified all-rows arm (the Amendment 19 addition): stand_ab's
features for the departures WITHOUT an NM off-block, so one LightGBM can be trained on every
admissible training row and score every row of the evaluation file.

`stand_ab.build_unmatched(table, serve)` emits, for the admissible unmatched departures (the
stratum's own row set: DEP, AOBT_3 null, labelled in training mode, one row per MVT_ID by
build_submission.admissible's keep-first rule, at the ten airports, in take-off order), the
matched cache's feature columns computed against the SAME reference streams the matched cache
uses (the matched departures and the arrivals), plus the queue features anchored on the row's
take-off and the day block. Everything that reads the row's own AOBT_3 - directly or as the
anchor of a window - is NaN by construction (UNMATCHED_NAN_COLS); everything that reads the row's
*_flt clocks is NaN whenever those are null, which on the data is every unmatched row
(Amendment 3 section 2, measured 2026-09-09 on January and on ranking.parquet).

What is pinned:
  1. formula parity on a generated 27-column fixture: an unmatched row CLONED from a matched
     row (the same clocks, AOBT_3 and every *_flt column null) gets the matched row's window
     features exactly where the window is anchored on the take-off and counts the reference
     stream - the two symmetric take-off counts see the original as one more neighbour (+1) and
     the ambient proxy median includes it (asserted from an independent re-derivation) - and
     the NaN pattern on the AOBT_3-anchored columns;
  2. the contract: the column list, the NaN lists, the take-off order, dtypes, ids unique;
  3. the row set: admissible's keep-first duplicate rule reproduced; the unlabelled row admitted
     in serve mode only; the hidden-clock invariant (blank / permute BLOCK and TAXITIME);
  4. the in-fold encodings under a delta mask (delta is NaN on unmatched rows: the delta
     encodings must be fitted on the finite rows and the y encodings on every training row);
  5. the per-calendar-month evaluation build and the cache command;
  6. on real rows (NOT YET RUN - the live fold): the ranking build yields exactly the 5,290
     unmatched scored rows with the identical NaN pattern to training, and the training row set
     equals derive(admissible(...))[unmatched]'s per month.
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
RANKING = RAW / "ranking.parquet"
TEMPLATE = RAW / "submitting.parquet"
MONTHS = sorted(RAW.glob("training_2025-*.parquet"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stand_ab = _load("stand_ab")
bs = _load("build_submission")

needs_data = pytest.mark.skipif(not (JAN.exists() and RANKING.exists() and TEMPLATE.exists()),
                                reason="challenge data is not redistributable and is absent from a clean checkout")
needs_year = pytest.mark.skipif(len(MONTHS) != 12 or MONTHS[0] != JAN,
                                reason="the NaN-rate envelope needs the twelve 2025 training months")
#: a non-anchored feature's NaN rate on the scored rows must sit inside the envelope of the
#: twelve 2025 training months' rates, widened by this on each side - the January-only bar the
#: envelope replaced (test_ranking_unmatched_build_yields_the_5290_scored_rows_...)
MAX_NAN_RATE_DRIFT = 0.02

#: the contracts, spelled out
UNMATCHED_NAN_COLS = ["proxy", "aobt_sched", "aobt_eobt", "aobt_sec", "q_apt_at_push", "q_rwy_at_push",
                      "q_pushed_after_me", "q_rwy_tko_in_taxi", "q_rwy_push_pre20", "q_arr_taxiing_at_push",
                      "q_arr_taxiin_sym30_push", "q_next_arr_onblock_gap", "d_prx_minus_p50"]
UNMATCHED_FLT_COLS = ["eobt_p", "lobt_p", "iobt_p", "eobt_sched", "lobt_sched", "eobt_lobt", "iobt_lobt",
                      "airborne3", "airborne1", "arvt_diff"]
#: take-off-anchored features that must equal the cloned matched row's value exactly
EQUAL_ON_CLONE = ["sp", "mvt_sec", "sched_sec", "hr", "tmin", "dow", "doy", "actype_null",
                  "n_push", "rwy_b30", "apt_b30", "arr_b30", "sched_prev60", "sched_next60", "sched_day", "rwy_rate",
                  "dep_dur", "rwy_dur", "arr_dur", "arr_ob60_taxiin", "arr_ob60_cdiff", "arr_ob180_cdiff",
                  "prev_arr_gap", "prev_arr_taxiin", "prev_arr_cdiff", "prev_dep_gap",
                  "q_arr_taxiin_sym30_tko", "d_prx_p50", "d_prx_p90", "d_prx_le0", "d_arr_p50", "d_arr_p90",
                  "d_arr_long", "d_n_dep", "d_n_arr"]
PLUS_ONE_ON_CLONE = ["q_dep_tko_sym15", "q_rwy_tko_sym10"]      # the clone sees the original as a neighbour
FLAGS = ["f_callsign", "f_ades", "f_type", "f_rule"]             # "NA" != the mvt value on every unmatched row
T0 = pd.Timestamp("2025-03-10 00:00:00", tz="UTC")


def _ts(seconds):
    return T0 + pd.to_timedelta(np.asarray(seconds, dtype="float64"), unit="s")


def parity_fixture(seed=0, dup=False, unlabelled=True) -> tuple:
    """A movement table in the full COLS schema: 120 matched departures (90 EGLL, 30 LFPG) with
    DISTINCT take-off seconds, 150 arrivals over 150 distinct stands (build_features's stand
    index needs > 100 keys), 15 unmatched CLONES of matched rows (new ids, AOBT_3 and every *_flt
    column null, the same clocks), 5 genuine unmatched rows at fresh times, optionally one
    unlabelled unmatched row (serve mode only) and one duplicated unmatched id (the keep-first
    rule). Returns (table, {clone id: original id}, genuine ids)."""
    rng = np.random.default_rng(seed)
    n_m = 120
    apt = np.array(["EGLL"] * 90 + ["LFPG"] * 30)
    t = np.sort(rng.choice(np.arange(3_600, 170_000), n_m, replace=False)).astype("float64")
    proxy = rng.integers(300, 1_500, n_m).astype("float64")
    a = t - proxy
    sched = a - rng.integers(0, 900, n_m)
    block = a - rng.integers(60, 300, n_m)
    stands = np.array([f"S{i:03d}" for i in range(150)])
    rwy = rng.choice(["27L", "09R"], n_m)
    stand = rng.choice(stands[:120], n_m)
    ac = rng.choice(["A320", "B738", "A21N"], n_m)
    flight = np.array([f"BAW{k}" for k in rng.integers(100, 999, n_m)])
    ades = rng.choice(["LFPG", "EDDF", "LEMD"], n_m)
    rows = pd.DataFrame({
        "PHASE_mvt": "DEP", "MVT_ID_mvt": np.arange(1, n_m + 1, dtype="float64"), "ADEP_mvt": apt, "ADES_mvt": ades,
        "STAND_mvt": stand, "RUNWAY_mvt": rwy, "AIRCRAFT_TYPE_mvt": ac, "FLIGHT_mvt": flight, "FLIGHT_RULE_mvt": "I",
        "MVT_TIME_UTC_mvt": _ts(t), "SCHED_TIME_UTC_mvt": _ts(sched), "BLOCK_TIME_UTC_mvt": _ts(block),
        "TAXITIME_SEC_mvt": t - block, "AOBT_3_flt": _ts(a), "EOBT_1_flt": _ts(a + rng.integers(-600, 600, n_m)),
        "LOBT_flt": _ts(a + rng.integers(-900, 300, n_m)), "IOBT_flt": _ts(a + rng.integers(-1200, 0, n_m)),
        "ARVT_1_flt": _ts(t + rng.integers(20, 120, n_m)), "ARVT_3_flt": _ts(t + rng.integers(30, 150, n_m)),
        "CALLSIGN_flt": flight, "ADES_FILED_flt": ades, "AIRCRAFT_TYPE_flt": ac, "FLIGHT_RULE_flt": "I",
        "AIRCRAFT_OPERATOR_flt": "BAW", "MARKET_SEGMENT_flt": "Sched", "WK_TBL_CAT_flt": "M", "FLIGHT_TYPE_flt": "S"})
    n_a = 170
    a_apt = np.array(["EGLL"] * 150 + ["LFPG"] * 20)
    land = rng.integers(0, 170_000, n_a).astype("float64")
    onb = land + rng.integers(200, 1_600, n_a)
    arr = pd.DataFrame({
        "PHASE_mvt": "ARR", "MVT_ID_mvt": np.arange(1_001, 1_001 + n_a, dtype="float64"), "ADEP_mvt": "LEMD",
        "ADES_mvt": a_apt, "STAND_mvt": np.r_[stands, stands[:20]], "RUNWAY_mvt": "27R", "AIRCRAFT_TYPE_mvt": "A320",
        "FLIGHT_mvt": "IBE123", "FLIGHT_RULE_mvt": "I", "MVT_TIME_UTC_mvt": _ts(land), "SCHED_TIME_UTC_mvt": _ts(land - 600),
        "BLOCK_TIME_UTC_mvt": _ts(onb), "TAXITIME_SEC_mvt": onb - land, "AOBT_3_flt": pd.NaT, "EOBT_1_flt": pd.NaT,
        "LOBT_flt": pd.NaT, "IOBT_flt": pd.NaT, "ARVT_1_flt": pd.NaT, "ARVT_3_flt": _ts(onb - rng.integers(0, 120, n_a)),
        "CALLSIGN_flt": None, "ADES_FILED_flt": None, "AIRCRAFT_TYPE_flt": None, "FLIGHT_RULE_flt": None,
        "AIRCRAFT_OPERATOR_flt": None, "MARKET_SEGMENT_flt": None, "WK_TBL_CAT_flt": None, "FLIGHT_TYPE_flt": None})
    clone_src = rng.choice(np.arange(n_m), 15, replace=False)
    clones = rows.iloc[clone_src].copy().reset_index(drop=True)
    clones["MVT_ID_mvt"] = np.arange(2_001, 2_016, dtype="float64")
    clone_map = {2_001.0 + i: float(rows.MVT_ID_mvt.iloc[j]) for i, j in enumerate(clone_src)}
    genuine = rows.iloc[rng.choice(np.arange(n_m), 5, replace=False)].copy().reset_index(drop=True)
    genuine["MVT_ID_mvt"] = np.arange(3_001, 3_006, dtype="float64")
    gt = np.sort(rng.choice(np.setdiff1d(np.arange(3_600, 170_000), t.astype(int)), 5, replace=False)).astype("float64")
    genuine["MVT_TIME_UTC_mvt"] = _ts(gt)
    genuine["SCHED_TIME_UTC_mvt"] = _ts(gt - 1_500)
    genuine["BLOCK_TIME_UTC_mvt"] = _ts(gt - 700)
    genuine["TAXITIME_SEC_mvt"] = 700.0
    extra = [clones, genuine]
    if unlabelled:
        u = rows.iloc[[3]].copy()
        u["MVT_ID_mvt"] = 4_001.0
        u["BLOCK_TIME_UTC_mvt"] = pd.NaT
        u["TAXITIME_SEC_mvt"] = np.nan
        extra.append(u)
    if dup:
        d1 = rows.iloc[[7]].copy()
        d1["MVT_ID_mvt"] = 5_001.0
        d2 = d1.copy()
        d2["BLOCK_TIME_UTC_mvt"] = d1.BLOCK_TIME_UTC_mvt - pd.Timedelta(seconds=100)      # sorts FIRST on BLOCK
        d2["TAXITIME_SEC_mvt"] = d1.TAXITIME_SEC_mvt + 100.0
        extra += [d1, d2]
    unm = pd.concat(extra, ignore_index=True)
    unm["AOBT_3_flt"] = pd.NaT
    for c in ("EOBT_1_flt", "LOBT_flt", "IOBT_flt", "ARVT_1_flt", "ARVT_3_flt"):
        unm[c] = pd.NaT
    for c in ("CALLSIGN_flt", "ADES_FILED_flt", "AIRCRAFT_TYPE_flt", "FLIGHT_RULE_flt", "AIRCRAFT_OPERATOR_flt",
              "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt"):
        unm[c] = None
    table = pd.concat([rows, arr, unm], ignore_index=True)
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt",
              "IOBT_flt", "ARVT_1_flt", "ARVT_3_flt"):
        table[c] = pd.to_datetime(table[c], utc=True)
    table = table.iloc[np.random.default_rng(seed + 1).permutation(len(table))].reset_index(drop=True)
    return table[stand_ab.COLS], clone_map, [3_001.0 + i for i in range(5)]


def _same(a, b) -> np.ndarray:
    x, z = np.asarray(a), np.asarray(b)
    if x.dtype.kind in "fi" and z.dtype.kind in "fi":
        x, z = x.astype(float), z.astype(float)
        return (x == z) | (np.isnan(x) & np.isnan(z))
    return x == z


def _write(frame, path):
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    return path


def _blank_departures(t):
    t = t.copy()
    dep = (t.PHASE_mvt == "DEP").to_numpy()
    t.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    t.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    return t


# ---------------------------------------------------------------------------------------------
# 0. the contract
# ---------------------------------------------------------------------------------------------

def test_unmatched_contract_lists_and_column_order():
    """UNMATCHED_COLS is id, labels, BASE, SURF, STANDHIST, month, ap, the twelve string keys,
    the twelve QUEUE_FEATS, the nine DAY_FEATS (every feature of the full design, so the loader
    aligns by name); UNMATCHED_NAN_COLS is exactly the thirteen columns anchored on the row's own
    AOBT_3; UNMATCHED_FLT_COLS the ten that read its *_flt clocks; the two are disjoint and both
    inside the contract; the four computed queue features are the take-off-anchored ones.

    Fails when a list drifts. Rehearsed 2026-09-09: `"d_prx_minus_p5O"` in UNMATCHED_NAN_COLS went RED.
    """
    S = stand_ab
    want = (["MVT_ID_mvt", "delta", "y"] + S.BASE + S.SURF + S.STANDHIST + ["month", "ap"]
            + ["STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
               "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "ADEP_mvt", "stand_pref", "airline", "ars"] + S.QUEUE_FEATS + S.DAY_FEATS)
    assert S.UNMATCHED_COLS == want
    assert S.UNMATCHED_NAN_COLS == UNMATCHED_NAN_COLS and S.UNMATCHED_FLT_COLS == UNMATCHED_FLT_COLS
    assert not set(UNMATCHED_NAN_COLS) & set(UNMATCHED_FLT_COLS)
    assert set(UNMATCHED_NAN_COLS + UNMATCHED_FLT_COLS) <= set(want)
    assert S.UNMATCHED_QUEUE_FEATS == ["q_dep_tko_sym15", "q_rwy_tko_sym10", "q_arr_taxiin_sym30_tko", "q_rwy_ambient_proxy"]
    assert set(S.UNMATCHED_QUEUE_FEATS) | set(c for c in UNMATCHED_NAN_COLS if c.startswith("q_")) == set(S.QUEUE_FEATS)
    assert set(S.ENC_KEYS) <= set(want) and S.UCACHE == ROOT / "data" / "cache_unmatched"


# ---------------------------------------------------------------------------------------------
# 1. formula parity on clones, and the NaN pattern
# ---------------------------------------------------------------------------------------------

def test_cloned_unmatched_rows_reproduce_the_matched_rows_window_features(capsys):
    """build_unmatched against build_features on the generated fixture: for every clone, the 35
    take-off-anchored features listed in EQUAL_ON_CLONE equal the original's exactly (the clone
    is not in the reference stream but its original is, so every window counts the same rows);
    the two symmetric take-off counts are the original's + 1 (the original excludes itself, the
    clone sees it); the runway ambient proxy median equals the median over the window INCLUDING
    the original, re-derived here from the fixture; the thirteen AOBT_3-anchored columns and the
    ten *_flt-derived ones are NaN; the four mismatch flags are 1.0 (the *_flt value is null);
    delta and proxy are NaN, y is the label. The genuine unmatched rows get finite values on every
    computable column (their windows are not empty), and apt_b30 of each equals an independent
    count from the fixture. Rows come out in take-off order with unique ids, float32 features.

    Fails when a window is anchored on the wrong clock, when the reference stream differs from
    build_features's, when the +1 / self-exclusion semantics drift, or when a NaN column carries
    a value. Rehearsed 2026-09-09, each RED: `_win_count(tu, ta, -1800, 0)` -> `(-1800, 1)` on
    apt_b30 (a clone whose original has a neighbour 1 s later moves); `_count_le(tj, tu + W)` ->
    `_count_lt` on q_dep_tko_sym15; the proxy column left as `tu - a_u` (a NaN check on
    proxy fails only if a value appears: rehearsed by filling proxy with 0.0); `d_stand` built
    from ADES on the unmatched rows (prev_dep_gap moves).
    """
    table, clone_map, genuine = parity_fixture()
    m = (stand_ab.build_features(table).set_index("MVT_ID_mvt")
         .join(stand_ab.build_queue(table).set_index("MVT_ID_mvt"))
         .join(stand_ab.build_day(table).set_index("MVT_ID_mvt")).reset_index())
    u = stand_ab.build_unmatched(table)
    assert list(u.columns) == stand_ab.UNMATCHED_COLS and u.MVT_ID_mvt.is_unique
    assert u.MVT_ID_mvt.tolist() == sorted(u.MVT_ID_mvt.tolist(), key=lambda i: float(table.set_index("MVT_ID_mvt").MVT_TIME_UTC_mvt[i].value)), \
        "rows must come out in take-off order"
    assert set(u.MVT_ID_mvt) == set(clone_map) | set(genuine), "training mode: the labelled unmatched rows only"
    assert all(u[c].dtype == np.float32 for c in stand_ab.BASELINE_FEATS[:-len(stand_ab.ENC)] + stand_ab.QUEUE_FEATS + stand_ab.DAY_FEATS
               if c in u.columns)
    mi, ui = m.set_index("MVT_ID_mvt"), u.set_index("MVT_ID_mvt")
    bad = {}
    for cid, oid in clone_map.items():
        for c in EQUAL_ON_CLONE:      # the cache stores float32: the original's float64 cast to float32 must match exactly
            if not _same([np.float32(ui.loc[cid, c])], [np.float32(mi.loc[oid, c])])[0]:
                bad.setdefault(c, []).append((cid, float(ui.loc[cid, c]), float(mi.loc[oid, c])))
        for c in PLUS_ONE_ON_CLONE:
            if not float(ui.loc[cid, c]) == float(mi.loc[oid, c]) + 1.0:
                bad.setdefault(c, []).append((cid, float(ui.loc[cid, c]), float(mi.loc[oid, c])))
        for c in UNMATCHED_NAN_COLS + UNMATCHED_FLT_COLS + ["delta", "proxy"]:
            if not np.isnan(float(ui.loc[cid, c])):
                bad.setdefault(c + " (NaN)", []).append((cid, float(ui.loc[cid, c])))
        for c in FLAGS:
            if float(ui.loc[cid, c]) != 1.0:
                bad.setdefault(c, []).append((cid, float(ui.loc[cid, c])))
        assert float(ui.loc[cid, "y"]) == float(mi.loc[oid, "y"]) and float(ui.loc[cid, "month"]) == 3
    assert not bad, "clone parity broken (column -> [(clone id, clone value, original value)]): " + \
        "; ".join(f"{c}: {v[:3]}" for c, v in bad.items())
    # the ambient proxy median, re-derived: matched on the clone's runway within +-3,600 s of its take-off,
    # the original INCLUDED (the matched row's own value excludes itself)
    dep = table[(table.PHASE_mvt == "DEP") & table.AOBT_3_flt.notna()]
    tko = (dep.MVT_TIME_UTC_mvt - T0).dt.total_seconds().to_numpy()
    px = tko - (dep.AOBT_3_flt - T0).dt.total_seconds().to_numpy()
    for cid, oid in clone_map.items():
        row = table[table.MVT_ID_mvt == cid].iloc[0]
        tu = (row.MVT_TIME_UTC_mvt - T0).total_seconds()
        win = (dep.ADEP_mvt.to_numpy() == row.ADEP_mvt) & (dep.RUNWAY_mvt.to_numpy() == row.RUNWAY_mvt) & (np.abs(tko - tu) <= 3_600)
        want = float(np.median(px[win])) if win.any() else np.nan
        assert _same([ui.loc[cid, "q_rwy_ambient_proxy"]], [np.float32(want)])[0], (cid, ui.loc[cid, "q_rwy_ambient_proxy"], want)
    n_moved = sum(1 for cid, oid in clone_map.items() if float(ui.loc[cid, "q_rwy_ambient_proxy"]) != float(mi.loc[oid, "q_rwy_ambient_proxy"]))
    assert n_moved > 0, "the fixture never separates the with-self from the self-excluded median"
    always = ["sp", "mvt_sec", "sched_sec", "hr", "tmin", "dow", "doy", "actype_null", "n_push", "rwy_b30", "apt_b30",
              "arr_b30", "sched_prev60", "sched_next60", "sched_day", "rwy_rate", "arr_dur", "arr_ob60_taxiin",
              "arr_ob60_cdiff", "arr_ob180_cdiff", "q_dep_tko_sym15", "q_rwy_tko_sym10", "d_n_dep", "d_n_arr"]
    for gid in genuine:
        row = table[table.MVT_ID_mvt == gid].iloc[0]
        tu = (row.MVT_TIME_UTC_mvt - T0).total_seconds()
        same_apt = dep.ADEP_mvt.to_numpy() == row.ADEP_mvt
        want = int((same_apt & (tko > tu - 1_800) & (tko <= tu)).sum())
        assert float(ui.loc[gid, "apt_b30"]) == want, (gid, ui.loc[gid, "apt_b30"], want)
        assert float(ui.loc[gid, "sp"]) == 1_500.0 and float(ui.loc[gid, "y"]) == 700.0
        nans = [c for c in always if np.isnan(float(ui.loc[gid, c]))]
        assert not nans, f"genuine unmatched row {gid}: unexpected NaN on an always-defined column: {nans}"
        # the day levels are NaN exactly when the airport-day has no matched departure (row 3001: LFPG on a
        # day with none), the trailing medians when fewer than three matched departures precede the row
        n_day = int((same_apt & (np.floor_divide(tko, 86_400) == np.floor_divide(tu, 86_400))).sum())
        assert float(ui.loc[gid, "d_n_dep"]) == n_day
        assert np.isnan(float(ui.loc[gid, "d_prx_p50"])) == (n_day == 0), gid
        assert np.isnan(float(ui.loc[gid, "dep_dur"])) == (int((same_apt & (tko <= tu)).sum()) < 3), gid
    assert any(np.isnan(float(ui.loc[g, "d_prx_p50"])) for g in genuine), "the fixture never exercises an empty airport-day"
    print(f"\nclone parity: {len(clone_map)} clones x {len(EQUAL_ON_CLONE)} equal + {len(PLUS_ONE_ON_CLONE)} plus-one columns; "
          f"ambient median moved on {n_moved} clones")


def test_row_set_keep_first_duplicates_unlabelled_and_serve_mode():
    """Training mode keeps one row per MVT_ID by build_submission.admissible's rule - sort by
    BLOCK, MVT_TIME, TAXITIME, MVT_ID (mergesort) and keep the first - checked against
    bs.admissible itself on the fixture's duplicated id; the unlabelled unmatched row is dropped
    in training mode and admitted in serve mode; the serve build of the blanked table equals the
    training build on every column for the shared rows; an empty unmatched stream is refused.

    Fails when the duplicate rule keeps the last, when the serve filter reads the label, or when
    an empty stream returns silently. Rehearsed 2026-09-09, each RED: `keep="last"`;
    `t.TAXITIME_SEC_mvt.notna()` left in the serve filter; the raise replaced by `pass`.
    """
    table, clone_map, genuine = parity_fixture(dup=True)
    u = stand_ab.build_unmatched(table)
    assert (u.MVT_ID_mvt == 5_001.0).sum() == 1
    kept = float(u.set_index("MVT_ID_mvt").loc[5_001.0, "y"])
    adm = bs.admissible(table[(table.PHASE_mvt == "DEP") & table.AOBT_3_flt.isna()])
    assert kept == float(adm.set_index("MVT_ID_mvt").loc[5_001.0, "TAXITIME_SEC_mvt"]), "not admissible's keep-first row"
    assert 4_001.0 not in set(u.MVT_ID_mvt), "the unlabelled row must not be a training row"
    table_serve, _, _ = parity_fixture(dup=False)
    served = stand_ab.build_unmatched(_blank_departures(table_serve), serve=True)
    assert 4_001.0 in set(served.MVT_ID_mvt) and served.y.isna().all() and served.delta.isna().all()
    train = stand_ab.build_unmatched(table_serve)
    aligned = served.set_index("MVT_ID_mvt").loc[train.MVT_ID_mvt.to_numpy()].reset_index()
    moved = [c for c in stand_ab.UNMATCHED_COLS if c not in ("y", "delta") and not _same(train[c], aligned[c]).all()]
    assert not moved, f"serve mode differs from training mode on: {moved}"
    with pytest.raises(ValueError, match="no admissible unmatched"):
        stand_ab.build_unmatched(table_serve[table_serve.AOBT_3_flt.notna() | (table_serve.PHASE_mvt == "ARR")])


def test_unmatched_features_never_read_the_departure_own_hidden_clocks():
    """Permuting BLOCK_TIME and TAXITIME across departures, or blanking both, leaves the serve
    build byte-identical; training mode's row set moves under the permutation (the control).

    Fails when any feature or the serve filter reads a departure's BLOCK or TAXITIME. Rehearsed
    2026-09-09: `+ 1e-3 * es(u.BLOCK_TIME_UTC_mvt)` on sp went RED (blanked: NaN; permuted: moved).
    """
    table, _, _ = parity_fixture()
    base = stand_ab.build_unmatched(table, serve=True)
    t = table.copy()
    dep = (t.PHASE_mvt == "DEP").to_numpy()
    perm = np.random.default_rng(0).permutation(dep.sum())
    for col in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"):
        vals = t.loc[dep, col].to_numpy()
        t.loc[dep, col] = vals[perm]
    for name, tbl in (("permuted", t), ("blanked", _blank_departures(table))):
        served = stand_ab.build_unmatched(tbl, serve=True)
        assert served.MVT_ID_mvt.tolist() == base.MVT_ID_mvt.tolist(), f"{name}: the serve row set changed"
        moved = [c for c in stand_ab.UNMATCHED_COLS if not _same(base[c], served[c]).all()]
        assert not moved, f"{name}: unmatched features read the departure's own hidden clocks: {moved}"
    assert set(stand_ab.build_unmatched(t).MVT_ID_mvt) != set(stand_ab.build_unmatched(table).MVT_ID_mvt), \
        "training mode admitted the same rows under the permutation - the control proved nothing"


# ---------------------------------------------------------------------------------------------
# 2. the in-fold encodings under a delta mask
# ---------------------------------------------------------------------------------------------

def test_infold_encodings_fit_the_delta_encodings_on_the_masked_rows_only():
    """delta is NaN on every unmatched row, so the unified design fits the 12 delta encodings on
    the rows `delta_mask` names (the matched training rows) and the 12 y encodings on every
    training row. Pinned by hand with SMOOTH = 50: keys A A B B A B, y 1..6, delta 10 20 30 40 NaN
    NaN, delta_mask = finite -> prior_d = 25, de_A = (30 + 25 x 50) / 52 = 24.6154, de_B =
    (70 + 1250) / 52 = 25.3846; te_A = (1 + 2 + 5 + 3 x 3.5) / 53 (prior_y = 3.5 over all six).
    Without a mask a NaN delta on a training row is refused; the default (no NaN) is unchanged.

    Fails when the delta prior or the encodings read the NaN rows (NaN everywhere), or when the
    mask is applied to the y encodings too. Rehearsed 2026-09-09, each RED: `d_tr = delta[tr_mask]`
    ignoring the mask; the mask applied to `y_tr`.
    """
    keys = pd.DataFrame({k: np.array(["A", "A", "B", "B", "A", "B"]) for k in stand_ab.ENC_KEYS})
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    delta = np.array([10.0, 20.0, 30.0, 40.0, np.nan, np.nan])
    tr = np.ones(6, bool)
    enc = stand_ab.infold_encodings(keys, y, delta, tr, delta_mask=np.isfinite(delta))
    assert enc["de_ADEP_mvt"][0] == pytest.approx((30 + 25 * 50) / 52, abs=1e-5)
    assert enc["de_ADEP_mvt"][2] == pytest.approx((70 + 25 * 50) / 52, abs=1e-5)
    assert np.isfinite(enc["de_ADEP_mvt"]).all() and enc["de_ADEP_mvt"][4] == enc["de_ADEP_mvt"][0]
    assert enc["te_ADEP_mvt"][0] == pytest.approx((8 + 3.5 * 50) / 53, abs=1e-5)
    with pytest.raises(ValueError, match="NaN delta"):
        stand_ab.infold_encodings(keys, y, delta, tr)
    finite = np.isfinite(delta)
    a = stand_ab.infold_encodings(keys[finite].reset_index(drop=True), y[finite], delta[finite], tr[finite])
    b = stand_ab.infold_encodings(keys[finite].reset_index(drop=True), y[finite], delta[finite], tr[finite], delta_mask=tr[finite])
    assert all(np.array_equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------------------------
# 3. the evaluation build and the cache command
# ---------------------------------------------------------------------------------------------

def test_unmatched_ranking_build_is_per_calendar_month_and_the_cache_command_writes_the_contract(tmp_path, monkeypatch):
    """build_unmatched_ranking builds one calendar month at a time in serve mode (a spy sees the
    months) and refuses a NaT MVT_TIME and duplicate ids across months; `unmatched-cache --smoke`
    writes one month in the contract, `--ranking --smoke` is refused, `--ranking` writes
    ranking.parquet equal to build_unmatched_ranking; nothing lands in the other cache dirs.

    Fails when the file is built as one unit, when the ranking cache is built in training mode,
    or when the smoke guard is dropped. Rehearsed 2026-09-09, each RED: `parts =
    [build_unmatched(t, serve=True)]`; `build_unmatched_month(RAW / "ranking.parquet")`;
    the `if smoke: raise` guard removed (4th occurrence).
    """
    table, _, _ = parity_fixture()
    m1 = _blank_departures(table)
    m2 = m1.copy()
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt",
              "IOBT_flt", "ARVT_1_flt", "ARVT_3_flt"):
        m2[c] = m2[c] + pd.Timedelta(days=31)
    m2["MVT_ID_mvt"] = m2.MVT_ID_mvt + 10_000
    calls = []
    real = stand_ab.build_unmatched

    def spy(frame, serve=False, u=None):
        calls.append((sorted(frame.MVT_TIME_UTC_mvt.dt.month.unique().tolist()), serve))
        return real(frame, serve=serve, u=u)
    monkeypatch.setattr(stand_ab, "build_unmatched", spy)
    two = pd.concat([m1, m2], ignore_index=True)
    got = stand_ab.build_unmatched_ranking(_write(two, tmp_path / "ranking.parquet"))
    assert calls == [([3], True), ([4], True)]
    want = pd.concat([real(m1, serve=True), real(m2, serve=True)], ignore_index=True)
    assert got.MVT_ID_mvt.tolist() == want.MVT_ID_mvt.tolist() and got.MVT_ID_mvt.is_unique
    # the two-read month path (RCOLS for the streams, COLS for the unmatched rows) equals the one-table path
    one = real(table)
    two_read = stand_ab.build_unmatched_month(_write(table, tmp_path / "one.parquet"))
    assert one.MVT_ID_mvt.tolist() == two_read.MVT_ID_mvt.tolist()
    assert not [c for c in stand_ab.UNMATCHED_COLS if not _same(one[c], two_read[c]).all()]
    assert not [c for c in stand_ab.UNMATCHED_COLS if not _same(got[c], want[c]).all()]
    dup = pd.concat([m1, m2.assign(MVT_ID_mvt=m1.MVT_ID_mvt.to_numpy())], ignore_index=True)
    with pytest.raises(AssertionError, match="duplicate MVT_ID"):
        stand_ab.build_unmatched_ranking(_write(dup, tmp_path / "dup.parquet"))
    bad = two.copy()
    bad.loc[bad.index[0], "MVT_TIME_UTC_mvt"] = pd.NaT
    with pytest.raises(ValueError, match="calendar month"):
        stand_ab.build_unmatched_ranking(_write(bad, tmp_path / "bad.parquet"))
    monkeypatch.setattr(stand_ab, "build_unmatched", real)

    raw, cache = tmp_path / "raw", tmp_path / "cache_unmatched"
    raw.mkdir()
    month = _write(table, raw / "training_2025-03-01_2025-04-01.parquet")
    _write(m2.assign(TAXITIME_SEC_mvt=600.0, BLOCK_TIME_UTC_mvt=m2.MVT_TIME_UTC_mvt - pd.Timedelta(seconds=600)),
           raw / "training_2025-04-01_2025-05-01.parquet")
    _write(two, raw / "ranking.parquet")
    monkeypatch.setattr(stand_ab, "RAW", raw)
    monkeypatch.setattr(stand_ab, "UCACHE", cache)
    for name in ("CACHE", "QCACHE", "DCACHE"):
        monkeypatch.setattr(stand_ab, name, tmp_path / f"untouched_{name}")
    stand_ab.cmd_unmatched_cache(smoke=True, ranking=False)
    assert sorted(p.name for p in cache.iterdir()) == ["training_2025-03-01_2025-04-01.parquet"]
    o = pd.read_parquet(cache / "training_2025-03-01_2025-04-01.parquet")
    assert list(o.columns) == stand_ab.UNMATCHED_COLS
    w = stand_ab.build_unmatched_month(month)
    assert o.MVT_ID_mvt.tolist() == w.MVT_ID_mvt.tolist() and all(_same(o[c], w[c]).all() for c in stand_ab.UNMATCHED_COLS)
    with pytest.raises(ValueError, match="smoke"):
        stand_ab.cmd_unmatched_cache(smoke=True, ranking=True)
    stand_ab.cmd_unmatched_cache(smoke=False, ranking=True)
    r = pd.read_parquet(cache / "ranking.parquet")
    wr = stand_ab.build_unmatched_ranking(raw / "ranking.parquet")
    assert r.MVT_ID_mvt.tolist() == wr.MVT_ID_mvt.tolist() and all(_same(r[c], wr[c]).all() for c in stand_ab.UNMATCHED_COLS)
    assert not any((tmp_path / f"untouched_{n}").exists() for n in ("CACHE", "QCACHE", "DCACHE"))
    assert "unmatched-cache" in (ROOT / "scripts" / "stand_ab.py").read_text().split('ap.add_argument("cmd"')[1].split(")")[0]


# ---------------------------------------------------------------------------------------------
# 4. real rows. WRITTEN 2026-09-09 AND NOT YET RUN (a live fold occupied the machine).
# ---------------------------------------------------------------------------------------------

@needs_data
@needs_year
def test_ranking_unmatched_build_yields_the_5290_scored_rows_with_the_training_nan_pattern(capsys):
    """Exactly the 5,290 scored departures without an NM off-block, the same id set as the raw
    file's; the NaN pattern by construction - UNMATCHED_NAN_COLS (the 13 AOBT_3-anchored
    columns) and UNMATCHED_FLT_COLS 100% NaN on the scored rows and on January training - and
    every other feature's scored NaN rate inside the envelope of the twelve 2025 training
    months' rates, widened by MAX_NAN_RATE_DRIFT (2 pp) on each side; the January training row
    set equals derive(admissible(...))[unmatched]'s (the stratum's own rows), 1,408 rows.

    The envelope replaced a January-only 2 pp bar on 2026-09-09, when that bar failed on
    prev_arr_cdiff - NaN on 14.35% of January's 1,408 rows against 10.47% of the 5,290 scored
    rows - with no bug behind it: the column is NaN when the stand's previous arrival carries no
    ARVT_3, and that varies by month. Measured with this builder on 2026-09-09, prev_arr_cdiff's
    NaN rate on the unmatched training rows, Jan..Dec 2025: 14.3, 20.4, 25.9, 22.3, 13.6, 12.6,
    7.4, 12.8, 9.3, 11.7, 15.1, 12.3% (12-month pooled 13.07%; Jan+Jul pooled 9.25%, the scored
    file being Jan+Jul 2026); every other non-anchored feature spans <= 4.6 pp across the twelve
    months and the scored rows sit within 1.31 pp of its Jan+Jul rate. A scored rate inside the
    envelope is indistinguishable from a training month's; a fill (0%) or a serve path that never
    finds the previous arrival (100%) falls outside it. The 100%-NaN assertions are the bug
    detectors and are unchanged.

    Fails when the ranking build admits a matched row or drops an unmatched one, when an
    anchored or *_flt column carries a value on any scored or January row, or when a feature's
    scored NaN rate is one no training month shows. Rehearsed 2026-09-09 on the module's own
    stand_ab instance (nothing under scripts/ is written while a fit runs), each RED: `proxy`
    (an AOBT_3-anchored column) filled with 0 on the scored rows - the 100%-NaN assertion names
    it; prev_arr_cdiff's NaNs filled with 0 on the scored rows - 0% against the envelope's floor
    of 7.41% - 2 pp, named with its (min, max, ranking) triple.
    """
    rank = stand_ab.build_unmatched_ranking(RANKING)
    raw = pq.read_table(RANKING, columns=["MVT_ID_mvt", "PHASE_mvt", "AOBT_3_flt"]).to_pandas()
    ids = set(pq.read_table(TEMPLATE, columns=["MVT_ID_mvt"]).to_pandas().MVT_ID_mvt)
    want = raw[(raw.PHASE_mvt == "DEP") & raw.AOBT_3_flt.isna() & raw.MVT_ID_mvt.isin(ids)]
    assert len(rank) == 5_290 and set(rank.MVT_ID_mvt) == set(want.MVT_ID_mvt)
    jan = stand_ab.build_unmatched_month(JAN)
    adm = bs.derive(bs.admissible(bs.load_movements([JAN])))
    adm = adm[adm.unmatched]
    assert len(jan) == len(adm) == 1_408 and set(jan.MVT_ID_mvt) == set(adm.MVT_ID_mvt)
    for c in UNMATCHED_NAN_COLS + UNMATCHED_FLT_COLS:
        assert rank[c].isna().all() and jan[c].isna().all(), c
    feats = [c for c in stand_ab.UNMATCHED_COLS if c not in ("y", "delta")]
    monthly = {JAN.name[9:16]: {c: float(jan[c].isna().mean()) for c in feats}}
    for p in MONTHS[1:]:
        m = stand_ab.build_unmatched_month(p)
        monthly[p.name[9:16]] = {c: float(m[c].isna().mean()) for c in feats}
        del m
    assert len(monthly) == 12
    print(f"\nNaN rate per unmatched feature: the twelve 2025 training months' envelope vs the scored "
          f"rows (n={len(rank):,}); features NaN nowhere or everywhere are omitted:")
    outside = {}
    for c in feats:
        lo, hi = min(r[c] for r in monthly.values()), max(r[c] for r in monthly.values())
        r_rk = float(rank[c].isna().mean())
        if not lo - MAX_NAN_RATE_DRIFT <= r_rk <= hi + MAX_NAN_RATE_DRIFT:
            outside[c] = (lo, hi, r_rk)
        if lo != hi or r_rk != lo:
            print(f"  {c:24s} months {100 * lo:6.2f}%..{100 * hi:6.2f}%   ranking {100 * r_rk:6.2f}%"
                  f"{'   <-- OUTSIDE' if c in outside else ''}")
    assert not outside, ("NaN rate on the scored rows outside the twelve training months' envelope "
                         f"(+/- {MAX_NAN_RATE_DRIFT:.0%}), {{feature: (min, max, ranking)}}: {outside}")
