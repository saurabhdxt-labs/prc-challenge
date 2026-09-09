"""The v6 queue block: push-anchored surface-congestion features (PREREG Amendment 14).

`stand_ab.build_queue(table, serve)` emits `MVT_ID_mvt` + `QUEUE_FEATS` for exactly the rows
`build_features` would emit, in the same take-off order. Everything it reads belongs to OTHER
movements and is populated on the evaluation file: other departures' `AOBT_3` / `MVT_TIME`, and
arrivals' landing (`MVT_TIME`), on-block (`BLOCK_TIME`) and taxi-in (`TAXITIME`). A departure's
own `BLOCK_TIME` / `TAXITIME` are the target and are blank on every scored row; nothing here may
touch them.

What is pinned, each on real rows or on a fixture whose every value is derived by hand:

  1. the arithmetic of every feature, on a micro-fixture with the boundary cases that separate a
     strict from a non-strict inequality and self-inclusion from self-exclusion;
  2. the row contract: build_queue's rows ARE build_features's rows, both modes, same order -
     which is also the proof that a positional join onto the v4 caches (which predate
     `MVT_ID_mvt`) is valid;
  3. serve/train parity on March 2025, with the causal proof that pruning the rows only the
     serve filter admits makes the two modes identical on every column;
  4. the hidden-clock invariant: blanking AND permuting departure BLOCK/TAXITIME leaves the
     serve build byte-identical;
  5. the evaluation build: exactly the 339,551 matched rows, per calendar month, with NaN rates
     within 2 pp of the March training rates;
  6. the cache command writes the contract columns and refuses a smoke ranking cache.
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
FEB = RAW / "training_2025-02-01_2025-03-01.parquet"
MARCH = RAW / "training_2025-03-01_2025-04-01.parquet"
RANKING = RAW / "ranking.parquet"
TEMPLATE = RAW / "submitting.parquet"

_spec = importlib.util.spec_from_file_location("stand_ab", ROOT / "scripts" / "stand_ab.py")
stand_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stand_ab)

needs_data = pytest.mark.skipif(
    not (JAN.exists() and FEB.exists() and MARCH.exists() and RANKING.exists() and TEMPLATE.exists()),
    reason="challenge data is not redistributable and is absent from a clean checkout",
)

#: the contract, spelled out rather than read back from the module so a typo in the module's
#: list cannot certify itself
QUEUE_FEATS = ["q_apt_at_push", "q_rwy_at_push", "q_pushed_after_me", "q_rwy_tko_in_taxi",
               "q_rwy_push_pre20", "q_dep_tko_sym15", "q_rwy_tko_sym10", "q_arr_taxiing_at_push",
               "q_arr_taxiin_sym30_push", "q_arr_taxiin_sym30_tko", "q_rwy_ambient_proxy",
               "q_next_arr_onblock_gap"]
#: mechanism of every feature, for the parity test: DEP-stream features may legitimately differ
#: between modes on the rows the serve filter admits extra reference departures for; ARR-stream
#: features read arrivals only and must not move at all.
DEP_STREAM = ["q_apt_at_push", "q_rwy_at_push", "q_pushed_after_me", "q_rwy_tko_in_taxi",
              "q_rwy_push_pre20", "q_dep_tko_sym15", "q_rwy_tko_sym10", "q_rwy_ambient_proxy"]
ARR_STREAM = ["q_arr_taxiing_at_push", "q_arr_taxiin_sym30_push", "q_arr_taxiin_sym30_tko",
              "q_next_arr_onblock_gap"]
MIN_WINDOW_MATCH = 0.995
MAX_NAN_RATE_DRIFT = 0.02

# ---------------------------------------------------------------------------------------------
# the micro-fixture: one airport (EGLL), two runways, nine departures, six arrivals.
# Seconds are offsets from T0. Every feature value below is derived by hand from these rows.
# ---------------------------------------------------------------------------------------------
T0 = pd.Timestamp("2025-03-10 10:00:00", tz="UTC")
#: (MVT_ID, AOBT_3 = push, MVT_TIME = take-off, runway, stand)
DEPS = [(1, 0, 600, "27L", "A1"),
        (2, 100, 900, "27L", "A2"),      # pushes at the same second as 3: a tie, strict "<"
        (3, 100, 1500, "09R", "A3"),
        (4, 600, 1200, "27L", "A1"),     # pushes exactly at 1's take-off: 1 is NOT on the ground
        (5, 1800, 2100, "27L", "A4"),    # push = 4's push + 1200 exactly: pre20 boundary
        (6, 1600, 2000, "09R", "A5"),
        (7, 5100, 5400, "27L", "A6"),    # pushes exactly at 8's take-off
        (8, 4900, 5100, "27L", "A2"),
        (9, 20000, 20600, "27L", "A7"),  # alone: every window empty
        (10, 30000, 29900, "27L", "A8"), # take-off BEFORE push (proxy = -100): 177 such rows on
        (11, 29950, 29990, "27L", "A9")] # the evaluation file; its own intervals are empty
#: (MVT_ID, landing = MVT_TIME, on-block = BLOCK_TIME, stand); taxi-in = on-block - landing
ARRS = [(101, -300, 200, "A1"),
        (102, 50, 100, "A2"),            # on-block exactly at 2's push: not taxiing at the push
        (103, 1000, 1800, "A1"),         # on-block exactly at 5's push
        (104, 2000, 5000, "A4"),
        (105, 89000, 91500, "A6"),       # on-block exactly 24 h after 7's push: inside
        (106, 91000, 91301, "A2")]       # on-block 24 h + 1 s after 8's push: outside

#: by MVT_ID 1..9. Derivations (a = push, t = take-off; 27L = {1,2,4,5,7,8,9}, 09R = {3,6}):
#:   q_apt_at_push      #{j: a_j < a_i < t_j}      4: 2 (100<600<900) and 3 (100<600<1500); 1's
#:                                                  t=600 is not > 600.  5: 6 (1600<1800<2000).
#:   q_rwy_at_push      same on the runway          4: only 2 (3 is on 09R).  5: 6 is on 09R -> 0.
#:   q_pushed_after_me  #{j: a_i < a_j < t_i}      1: 2 and 3 (4's a=600 is not < 600).
#:                                                  6: 5 (1600<1800<2000).
#:   q_rwy_tko_in_taxi  #{j rwy: a_i < t_j < t_i}  2: 1 (100<600<900). 4: 2 (600<900<1200), not
#:                                                  1 (600 is not > 600). 7: 8's t=5100 not > 5100.
#:   q_rwy_push_pre20   #{j rwy: a_i-1200 <= a_j < a_i}  4: 1,2. 5: 4 (600 = 1800-1200). 7: 8.
#:   q_dep_tko_sym15    #{j: |t_j-t_i| <= 900} - 1  take-offs 600 900 1200 1500 2000 2100 5100
#:                                                  5400 20600; 4 (1200): [300,2100] -> 6-1 = 5.
#:   q_rwy_tko_sym10    #{j rwy: |t_j-t_i| <= 600} - 1  1 (600): 600,900,1200 -> 2. 5 (2100):
#:                                                  only itself -> 0. 3 (09R,1500): 1500,2000 -> 1.
#:   q_arr_taxiing_at_push #{arr: land < a_i < onblock}  2: 101 only (102's on-block 100 is not
#:                                                  > 100). 5: 103's on-block 1800 is not > 1800.
#:                                                  8: 104 (2000<4900<5000). 7: 5100 is not < 5000.
#:   q_arr_taxiin_sym30_push  mean taxi-in, on-block in [a-1800, a+1800]  taxi-ins 500,50,800,
#:                                                  3000,2500,301. 1..6: 101,102,103 -> 450
#:                                                  (1: 103's 1800 = 0+1800). 7,8: 104 -> 3000.
#:   q_arr_taxiin_sym30_tko   same around t         6 (2000): [200,3800] -> 101 (200 = 2000-1800)
#:                                                  and 103 -> 650. 5 (2100): [300,3900] -> 103 -> 800.
#:   q_rwy_ambient_proxy   median proxy_j on rwy, t_j in [t-3600, t+3600], self excluded
#:                                                  proxies 27L: 1:600 2:800 4:600 5:300 8:200 7:300
#:                                                  9:600; 09R: 3:1400 6:400.  1 (600): {2,4,5} ->
#:                                                  median(800,600,300)=600. 8 (5100): {5,7} -> 300.
#:                                                  7 (5400): {5,8} -> 250. 3: {6} -> 400. 9: none.
#:   q_next_arr_onblock_gap (first on-block at my stand after a, within 24 h) - t
#:                                                  1 (A1): 200-600 = -400. 4 (A1): 1800-1200 = 600.
#:                                                  5 (A4): 5000-2100 = 2900. 7 (A6): 91500-5100 =
#:                                                  86400 exactly -> 91500-5400 = 86100. 8 (A2):
#:                                                  106 is 86401 after the push -> NaN. 2 (A2):
#:                                                  102's on-block 100 is not > 100 -> NaN.
#:   10 (push 30000, take-off 29900) and 11 (29950, 29990): 10's intervals (a, t) are EMPTY, so
#:   q_pushed_after_me and q_rwy_tko_in_taxi are 0 - a difference of two searchsorted counts
#:   gives -2 for both, which is the bug the evaluation rows surfaced on 2026-09-09. 10's pre20
#:   window [28800, 30000) holds 11's push; each is the other's only sym15/sym10 neighbour and
#:   the other's only ambient neighbour (proxies -100 and 40). Nothing arrives near either.
nan = np.nan
EXPECTED = {
    "q_apt_at_push":           [0, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0],
    "q_rwy_at_push":           [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "q_pushed_after_me":       [2, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "q_rwy_tko_in_taxi":       [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "q_rwy_push_pre20":        [0, 1, 0, 2, 1, 0, 1, 0, 0, 1, 0],
    "q_dep_tko_sym15":         [3, 3, 5, 5, 3, 3, 1, 1, 0, 1, 1],
    "q_rwy_tko_sym10":         [2, 2, 1, 2, 0, 1, 1, 1, 0, 1, 1],
    "q_arr_taxiing_at_push":   [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    "q_arr_taxiin_sym30_push": [450, 450, 450, 450, 450, 450, 3000, 3000, nan, nan, nan],
    "q_arr_taxiin_sym30_tko":  [450, 450, 450, 450, 800, 650, 3000, 3000, nan, nan, nan],
    "q_rwy_ambient_proxy":     [600, 600, 400, 600, 600, 1400, 250, 300, nan, 40, -100],
    "q_next_arr_onblock_gap":  [-400, nan, nan, 600, 2900, nan, 86100, nan, nan, nan, nan],
}
#: take-off order of the eleven departures
TAKEOFF_ORDER = [1, 2, 4, 3, 6, 5, 8, 7, 9, 10, 11]


def _ts(seconds):
    return T0 + pd.Timedelta(seconds=int(seconds))


def micro_fixture(blank_departure_clocks: bool = False) -> pd.DataFrame:
    """The rows above as a movement table in the raw schema (QCOLS), deliberately shuffled."""
    rows = []
    for mid, a, t, rwy, stand in DEPS:
        block = _ts(a - 120)                     # arbitrary: no feature may depend on it
        rows.append(dict(PHASE_mvt="DEP", MVT_ID_mvt=float(mid), ADEP_mvt="EGLL", ADES_mvt="LFPG",
                         STAND_mvt=stand, RUNWAY_mvt=rwy, MVT_TIME_UTC_mvt=_ts(t),
                         SCHED_TIME_UTC_mvt=_ts(a - 300), BLOCK_TIME_UTC_mvt=block,
                         TAXITIME_SEC_mvt=float(t - (a - 120)), AOBT_3_flt=_ts(a)))
    for mid, land, onb, stand in ARRS:
        rows.append(dict(PHASE_mvt="ARR", MVT_ID_mvt=float(mid), ADEP_mvt="LFPG", ADES_mvt="EGLL",
                         STAND_mvt=stand, RUNWAY_mvt="27R", MVT_TIME_UTC_mvt=_ts(land),
                         SCHED_TIME_UTC_mvt=_ts(land - 600), BLOCK_TIME_UTC_mvt=_ts(onb),
                         TAXITIME_SEC_mvt=float(onb - land), AOBT_3_flt=pd.NaT))
    t = pd.DataFrame(rows)
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"):
        t[c] = pd.to_datetime(t[c], utc=True)
    t = t.iloc[np.random.default_rng(3).permutation(len(t))].reset_index(drop=True)
    if blank_departure_clocks:
        dep = (t.PHASE_mvt == "DEP").to_numpy()
        t.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
        t.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    return t[stand_ab.QCOLS]


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Row-wise equality that treats NaN == NaN, for float and string columns alike."""
    x, z = a.to_numpy(), b.to_numpy()
    if a.dtype.kind in "fi" and b.dtype.kind in "fi":
        x, z = x.astype(float), z.astype(float)
        return (x == z) | (np.isnan(x) & np.isnan(z))
    return x == z


def _aligned(train: pd.DataFrame, serve: pd.DataFrame) -> pd.DataFrame:
    """Serve rows re-ordered to the training rows' MVT_IDs (every training row must exist)."""
    assert train.MVT_ID_mvt.is_unique and serve.MVT_ID_mvt.is_unique
    missing = set(train.MVT_ID_mvt) - set(serve.MVT_ID_mvt)
    assert not missing, f"{len(missing)} training rows are absent from the serve build"
    return serve.set_index("MVT_ID_mvt").loc[train.MVT_ID_mvt.to_numpy()].reset_index()


def _slice(src: pathlib.Path, lo: str, hi: str) -> pd.DataFrame:
    t = pq.read_table(src, columns=stand_ab.QCOLS).to_pandas()
    return t[(t.MVT_TIME_UTC_mvt >= pd.Timestamp(lo, tz="UTC"))
             & (t.MVT_TIME_UTC_mvt < pd.Timestamp(hi, tz="UTC"))]


def _blank_departures(t: pd.DataFrame) -> pd.DataFrame:
    """The evaluation file's shape: no BLOCK, no TAXITIME on any departure."""
    t = t.copy()
    dep = (t.PHASE_mvt == "DEP").to_numpy()
    t.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    t.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    return t


def _write(frame: pd.DataFrame, path: pathlib.Path) -> pathlib.Path:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    return path


# ---------------------------------------------------------------------------------------------
# 0. the contract itself
# ---------------------------------------------------------------------------------------------

def test_queue_feature_list_is_the_contract_and_every_column_is_classified():
    """QUEUE_FEATS is exactly the twelve pre-registered names, in this order, and every one is
    assigned a mechanism (DEP_STREAM or ARR_STREAM) so it cannot escape the parity test.

    Fails when a name is misspelt, dropped, reordered or added in the module without being
    classified here. Rehearsed 2026-09-09: `"q_rwy_tko_sym1O"` in the module's list went RED.
    """
    assert stand_ab.QUEUE_FEATS == QUEUE_FEATS
    assert sorted(DEP_STREAM + ARR_STREAM) == sorted(QUEUE_FEATS)
    assert not set(DEP_STREAM) & set(ARR_STREAM)
    assert not set(QUEUE_FEATS) & set(stand_ab.BASELINE_FEATS + stand_ab.STAND_BLOCK), \
        "a queue feature shadows an existing feature name"


# ---------------------------------------------------------------------------------------------
# 1. arithmetic, by hand
# ---------------------------------------------------------------------------------------------

def test_micro_fixture_every_feature_equals_the_hand_computation():
    """Every one of the twelve features, on nine departures and six arrivals, equals the value
    derived by hand in EXPECTED - including the boundary rows that pin each inequality: a push
    at the same second as another push, a push exactly at another aircraft's take-off, an
    on-block exactly at a push, a push exactly 1,200 s after another, take-offs exactly 900 s
    and 600 s apart, an arrival exactly 24 h after a push and one 24 h + 1 s after.

    Output is in take-off order, float32, keyed by MVT_ID; blanking the departures' BLOCK and
    TAXITIME (the evaluation file's shape) and building in serve mode gives the same values.

    Fails when any strict/non-strict inequality is flipped, when self-exclusion is dropped, when
    an empty interval is counted as negative, when airport- and runway-level keys are confused,
    or when a window width is wrong. Rehearsed 2026-09-09, each RED: `_count_lt` for `_count_le`
    on the lower bound of `_count_open` (q_pushed_after_me then counts the tie at 100 s and the
    row itself; rows 1-3 off by one); dropping the `- 1` in q_dep_tko_sym15 (every row +1);
    including the row itself in q_rwy_ambient_proxy (3: 400 -> 900, 7: 250 -> 300); `<= 86400`
    -> `< 86400` in q_next_arr_onblock_gap (7 -> NaN); removing the empty-interval clamp in
    `_count_open` (row 10: -2 on q_pushed_after_me and q_rwy_tko_in_taxi - the bug the
    evaluation rows surfaced before this row existed).
    """
    o = stand_ab.build_queue(micro_fixture(), serve=False)
    assert list(o.columns) == ["MVT_ID_mvt"] + QUEUE_FEATS
    assert o.MVT_ID_mvt.tolist() == TAKEOFF_ORDER, "rows must come out in take-off order"
    assert all(o[c].dtype == np.float32 for c in QUEUE_FEATS)
    assert o.MVT_ID_mvt.dtype == np.float64

    by_id = o.set_index("MVT_ID_mvt").loc[[float(i) for i in range(1, 12)]]
    wrong = {}
    for c, want in EXPECTED.items():
        got = by_id[c].to_numpy(dtype=float)
        want = np.asarray(want, dtype=float)
        ok = (got == want) | (np.isnan(got) & np.isnan(want))   # exact: every value is integral
        if not ok.all():
            wrong[c] = [(i + 1, g, w) for i, (g, w, k) in enumerate(zip(got, want, ok)) if not k]
    assert not wrong, "hand computation disagrees (MVT_ID, got, expected): " + \
        "; ".join(f"{c}: {v}" for c, v in wrong.items())

    served = stand_ab.build_queue(micro_fixture(blank_departure_clocks=True), serve=True)
    assert served.MVT_ID_mvt.tolist() == TAKEOFF_ORDER
    for c in QUEUE_FEATS:
        assert _same(o[c], served[c]).all(), f"{c} differs between training and serve mode"


def test_build_queue_refuses_an_empty_departure_stream():
    """A table with no admissible departure is a broken input, not an empty feature frame.

    Fails when build_queue returns silently on zero rows (a month cache of nothing would then be
    written and joined). Rehearsed 2026-09-09: replacing the raise with `pass` went RED - the
    key-collapse guard then fires an AssertionError ("0 airports") instead of the ValueError.
    """
    arrivals_only = micro_fixture()
    arrivals_only = arrivals_only[arrivals_only.PHASE_mvt == "ARR"]
    with pytest.raises(ValueError, match="no admissible departures"):
        stand_ab.build_queue(arrivals_only, serve=False)
    blanked = micro_fixture(blank_departure_clocks=True)
    with pytest.raises(ValueError, match="no admissible departures"):
        stand_ab.build_queue(blanked, serve=False)      # training mode needs the label


# ---------------------------------------------------------------------------------------------
# 2. the row contract and serve/train parity on real rows
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def march_pair():
    """March 2025 built both ways. Shared by the parity tests."""
    return (stand_ab.build_queue_month(MARCH, serve=False),
            stand_ab.build_queue_month(MARCH, serve=True))


@needs_data
def test_queue_rows_are_exactly_the_stand_ab_reference_stream_in_both_modes(march_pair):
    """build_queue's MVT_ID sequence equals build_features's, training and serve mode alike.

    This is the shared-stream contract of Amendment 14 ("the SAME reference stream stand_ab
    already builds") and it is load-bearing twice: it is what makes the queue features an
    honest addition to the v4 design matrix, and - because the on-disk training caches in
    data/cache_stand predate `MVT_ID_mvt` - it is the proof that a positional join of the
    queue cache onto those caches is valid.

    Fails when either mode's departure filter or sort key drifts from build_features. Rehearsed
    2026-09-09: requiring `proxy > 0` in `_dep_stream` went RED - 163,310 rows against
    build_month's 163,361 on March in training mode (51 departures with take-off <= push).
    """
    for serve, mine in zip((False, True), march_pair):
        theirs = stand_ab.build_month(MARCH, serve=serve)
        assert len(mine) == len(theirs) > 150_000
        assert np.array_equal(mine.MVT_ID_mvt.to_numpy(), theirs.MVT_ID_mvt.to_numpy()), \
            f"serve={serve}: build_queue's rows or order differ from build_features"


@needs_data
def test_serve_mode_reproduces_training_mode_on_march(march_pair, capsys):
    """ARR-stream features are byte-identical between modes; DEP-stream features agree on
    >= 99.5% of rows, the only admissible differences being the extra reference departures
    the serve filter admits (those without a positive label).

    Fails when serve mode builds either stream differently. Rehearsed 2026-09-09, each RED:
    a serve-only `arr = arr[arr.TAXITIME_SEC_mvt > 60]` arrival filter (the two taxi-in levels
    moved on 1.52% of rows, q_arr_taxiing_at_push on 0.007%); a serve-only `proxy > 0`
    departure filter (the serve build then has FEWER rows than training: RED at the first
    assertion).
    """
    train, serve = march_pair
    assert len(serve) > len(train), "the serve filter admitted no extra rows on March - the " \
                                    "window tolerance below would be untested"
    aligned = _aligned(train, serve)

    broken = {c: float(1 - _same(train[c], aligned[c]).mean()) for c in ARR_STREAM
              if not _same(train[c], aligned[c]).all()}
    assert not broken, "serve mode changed arrival-stream features: " + ", ".join(
        f"{c} differs on {100 * v:.3f}% of rows" for c, v in broken.items())

    rates = {c: float(_same(train[c], aligned[c]).mean()) for c in DEP_STREAM}
    n_extra = len(serve) - len(train)
    print(f"\nqueue DEP-stream features, serve vs training on March 2025 ({len(train):,} rows; "
          f"{n_extra} extra reference rows admitted):")
    for c, r in rates.items():
        print(f"  {c:24s} equal on {100 * r:7.3f}% of rows")
    low = {c: r for c, r in rates.items() if r < MIN_WINDOW_MATCH}
    assert not low, f"window features below {MIN_WINDOW_MATCH:.1%} agreement: {low}"


@needs_data
def test_serve_mode_differences_are_caused_only_by_the_extra_reference_rows(march_pair, tmp_path):
    """Causal proof: remove the rows only the serve filter admits, and the two modes agree on
    EVERY column and in row order.

    Fails when serve mode differs from training mode for any reason other than the extra
    reference rows. Rehearsed 2026-09-09: `+ (1.0 if serve else 0.0)` on q_rwy_push_pre20
    went RED naming the column.
    """
    train, serve = march_pair
    extra_ids = set(serve.MVT_ID_mvt) - set(train.MVT_ID_mvt)
    assert extra_ids, "no extra rows on March - the pruning proves nothing"
    t = pq.read_table(MARCH, columns=stand_ab.QCOLS).to_pandas()
    pruned = stand_ab.build_queue_month(
        _write(t[~t.MVT_ID_mvt.isin(extra_ids)], tmp_path / "pruned.parquet"), serve=True)
    assert len(pruned) == len(train)
    assert np.array_equal(pruned.MVT_ID_mvt.to_numpy(), train.MVT_ID_mvt.to_numpy())
    moved = [c for c in QUEUE_FEATS if not _same(train[c], pruned[c]).all()]
    assert not moved, f"serve mode differs beyond the extra reference rows on: {moved}"


# ---------------------------------------------------------------------------------------------
# 3. the hidden clocks
# ---------------------------------------------------------------------------------------------

@needs_data
def test_queue_features_never_read_the_departure_own_hidden_clocks(tmp_path):
    """Permuting BLOCK_TIME and TAXITIME across departures, or blanking both on every
    departure as the evaluation file does, leaves the serve build byte-identical.

    The blanked variant is the load-bearing one (a serve filter reading `BLOCK.notna()` empties
    the row set there and a permutation cannot see it); the permutation catches a feature that
    reads the value. The control proves the permutation is real: training mode, whose filter
    reads TAXITIME > 0, admits a different row set under it.

    Fails when any queue feature or the serve filter reads a departure's BLOCK or TAXITIME.
    Rehearsed 2026-09-09: adding `1e-3 * (es(d.BLOCK_TIME_UTC_mvt)[j] - aj)` to
    q_rwy_push_pre20 went RED, the permuted variant (checked first) naming the column.
    """
    base_path = _write(_slice(JAN, "2025-01-01", "2025-01-04"), tmp_path / "slice.parquet")
    base = stand_ab.build_queue_month(base_path, serve=True)

    t = pq.read_table(base_path, columns=stand_ab.QCOLS).to_pandas()
    dep = (t.PHASE_mvt == "DEP").to_numpy()
    rng = np.random.default_rng(0)
    perm = rng.permutation(dep.sum())
    for col in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"):
        vals = t.loc[dep, col].to_numpy()
        t.loc[dep, col] = vals[perm]
    perm_path = _write(t, tmp_path / "permuted.parquet")
    blank_path = _write(_blank_departures(pq.read_table(base_path, columns=stand_ab.QCOLS).to_pandas()),
                        tmp_path / "blanked.parquet")

    for name, path in (("permuted", perm_path), ("blanked", blank_path)):
        served = stand_ab.build_queue_month(path, serve=True)
        assert len(served) == len(base) > 5_000, f"{name}: the serve row set changed"
        assert np.array_equal(served.MVT_ID_mvt.to_numpy(), base.MVT_ID_mvt.to_numpy())
        moved = [c for c in QUEUE_FEATS if not _same(base[c], served[c]).all()]
        assert not moved, f"{name}: queue features read the departure's own hidden clocks: {moved}"

    ref = stand_ab.build_queue_month(base_path, serve=False)
    trained = stand_ab.build_queue_month(perm_path, serve=False)
    assert set(trained.MVT_ID_mvt) != set(ref.MVT_ID_mvt), \
        "training mode admitted the same rows under the permutation - the test proved nothing"


# ---------------------------------------------------------------------------------------------
# 4. the evaluation build
# ---------------------------------------------------------------------------------------------

@needs_data
def test_ranking_build_yields_every_matched_row_with_training_like_nan_rates(march_pair, capsys):
    """Exactly the 339,551 scored departures with an NM off-block, the same id set as the raw
    file's matched rows, every feature finite where it is finite in training, and per-column
    NaN rates within 2 pp of the March 2025 training rates.

    Fails when the serve filter is wrong (row count / id set), when a feature's inputs are
    blank on the evaluation rows (NaN rate jumps), or when the build is not per calendar month
    (see the adjacent-month test for the causal version). Rehearsed 2026-09-09: dropping
    `t.AOBT_3_flt.notna()` from `_dep_stream`'s serve filter went RED (344,841 rows).
    """
    train, _ = march_pair
    rank = stand_ab.build_queue_ranking(RANKING)
    raw = pq.read_table(RANKING, columns=["MVT_ID_mvt", "PHASE_mvt", "AOBT_3_flt"]).to_pandas()
    ids = set(pq.read_table(TEMPLATE, columns=["MVT_ID_mvt"]).to_pandas().MVT_ID_mvt)
    matched = raw[(raw.PHASE_mvt == "DEP") & raw.AOBT_3_flt.notna() & raw.MVT_ID_mvt.isin(ids)]
    assert len(rank) == 339_551, f"expected 339,551 matched rows, got {len(rank):,}"
    assert rank.MVT_ID_mvt.is_unique
    assert set(rank.MVT_ID_mvt) == set(matched.MVT_ID_mvt), "row set != raw matched ids"
    assert list(rank.columns) == ["MVT_ID_mvt"] + QUEUE_FEATS

    print(f"\nNaN rate per queue feature, training (March 2025, n={len(train):,}) vs ranking "
          f"serve build (n={len(rank):,}):")
    drift = {}
    for c in QUEUE_FEATS:
        r_tr, r_rk = float(train[c].isna().mean()), float(rank[c].isna().mean())
        flag = ""
        if abs(r_rk - r_tr) > MAX_NAN_RATE_DRIFT:
            drift[c] = (r_tr, r_rk)
            flag = "   <-- DRIFT"
        print(f"  {c:24s} train {100 * r_tr:7.3f}%   ranking {100 * r_rk:7.3f}%{flag}")
    assert not drift, "NaN rate on the evaluation rows differs from training by > 2 pp: " + \
        ", ".join(f"{c} (train {100 * a:.3f}%, ranking {100 * b:.3f}%)" for c, (a, b) in drift.items())
    counts = [c for c in QUEUE_FEATS if c not in ("q_arr_taxiin_sym30_push", "q_arr_taxiin_sym30_tko",
                                                   "q_rwy_ambient_proxy", "q_next_arr_onblock_gap")]
    for c in counts:
        assert rank[c].notna().all() and train[c].notna().all(), f"count feature {c} has NaN"
        assert (rank[c] >= 0).all() and (train[c] >= 0).all(), f"negative count in {c}"


@needs_data
def test_ranking_build_is_per_calendar_month(tmp_path):
    """Built one calendar month at a time, as the training files are: on a fixture of two
    ADJACENT months (31 Jan - 2 Feb 2025, departures blanked as on the evaluation file) the
    ranking build equals each month built alone and concatenated - and it must NOT equal the
    single-unit build, which lets windows and the 24 h stand look-ahead cross the boundary that
    the training caches cannot cross.

    The real evaluation file holds January and July, five months apart, where the two builds
    coincide; that is why the causal check needs adjacent months. Fails when
    build_queue_ranking builds the file as one unit. Rehearsed 2026-09-09:
    `parts = [build_queue(t, serve=True)]` went RED.
    """
    jan = _blank_departures(_slice(JAN, "2025-01-30", "2025-02-01"))
    feb = _blank_departures(_slice(FEB, "2025-02-01", "2025-02-03"))
    two = pd.concat([jan, feb], ignore_index=True)
    path = _write(two, tmp_path / "ranking.parquet")

    got = stand_ab.build_queue_ranking(path)
    want = pd.concat([stand_ab.build_queue(jan, serve=True), stand_ab.build_queue(feb, serve=True)],
                     ignore_index=True)
    assert len(got) == len(want) > 10_000        # four days at ten airports is ~20k departures
    assert np.array_equal(got.MVT_ID_mvt.to_numpy(), want.MVT_ID_mvt.to_numpy())
    moved = [c for c in QUEUE_FEATS if not _same(got[c], want[c]).all()]
    assert not moved, f"ranking build differs from the per-month builds on: {moved}"

    single = stand_ab.build_queue(two, serve=True)
    assert np.array_equal(single.MVT_ID_mvt.to_numpy(), want.MVT_ID_mvt.to_numpy())
    differs = [c for c in QUEUE_FEATS if not _same(single[c], want[c]).all()]
    assert differs, "single-unit and per-month builds agree on every column - the fixture " \
                    "cannot tell them apart and this test proves nothing"


# ---------------------------------------------------------------------------------------------
# 5. the cache command
# ---------------------------------------------------------------------------------------------

@needs_data
def test_cmd_queue_cache_writes_the_contract_columns_and_refuses_a_smoke_ranking(tmp_path,
                                                                                 monkeypatch):
    """`queue-cache --smoke` writes exactly one file, named after the first training month,
    holding `MVT_ID_mvt` + QUEUE_FEATS (float32) for the rows build_queue_month gives;
    `--ranking --smoke` is refused before anything is written; `--ranking` writes only
    ranking.parquet, equal to build_queue_ranking. Nothing lands outside the queue cache dir.

    Fails when the column list is misspelt, when the smoke writes more than one month, when a
    smoke ranking cache reaches the real path, or when the command forgets serve mode for the
    ranking file. Rehearsed 2026-09-09, each RED: `"q_rwy_tko_sym1O"` in QUEUE_FEATS; removing
    the `if smoke: raise ValueError` guard; `build_queue_month(RAW / "ranking.parquet")`
    (training mode) for the ranking cache.
    """
    raw, cache = tmp_path / "raw", tmp_path / "cache_queue"
    raw.mkdir()
    month = _write(_slice(JAN, "2025-01-01", "2025-01-03"), raw / "training_2025-01-01_2025-02-01.parquet")
    _write(_slice(FEB, "2025-02-01", "2025-02-02"), raw / "training_2025-02-01_2025-03-01.parquet")
    two = pd.concat([_blank_departures(_slice(JAN, "2025-01-30", "2025-02-01")),
                     _blank_departures(_slice(FEB, "2025-02-01", "2025-02-02"))], ignore_index=True)
    _write(two, raw / "ranking.parquet")
    monkeypatch.setattr(stand_ab, "RAW", raw)
    monkeypatch.setattr(stand_ab, "QCACHE", cache)
    monkeypatch.setattr(stand_ab, "CACHE", tmp_path / "must_not_be_touched")

    stand_ab.cmd_queue_cache(smoke=True, ranking=False)
    assert sorted(p.name for p in cache.iterdir()) == ["training_2025-01-01_2025-02-01.parquet"]
    o = pd.read_parquet(cache / "training_2025-01-01_2025-02-01.parquet")
    assert list(o.columns) == ["MVT_ID_mvt"] + QUEUE_FEATS
    assert all(o[c].dtype == np.float32 for c in QUEUE_FEATS)
    want = stand_ab.build_queue_month(month, serve=False)
    assert len(o) == len(want) > 5_000
    assert np.array_equal(o.MVT_ID_mvt.to_numpy(), want.MVT_ID_mvt.to_numpy())
    assert all(_same(o[c], want[c]).all() for c in QUEUE_FEATS)

    with pytest.raises(ValueError, match="smoke"):
        stand_ab.cmd_queue_cache(smoke=True, ranking=True)
    assert sorted(p.name for p in cache.iterdir()) == ["training_2025-01-01_2025-02-01.parquet"]

    stand_ab.cmd_queue_cache(smoke=False, ranking=True)
    assert sorted(p.name for p in cache.iterdir()) == ["ranking.parquet",
                                                       "training_2025-01-01_2025-02-01.parquet"]
    r = pd.read_parquet(cache / "ranking.parquet")
    want_r = stand_ab.build_queue_ranking(raw / "ranking.parquet")
    assert list(r.columns) == ["MVT_ID_mvt"] + QUEUE_FEATS
    assert np.array_equal(r.MVT_ID_mvt.to_numpy(), want_r.MVT_ID_mvt.to_numpy())
    assert all(_same(r[c], want_r[c]).all() for c in QUEUE_FEATS)
    assert not (tmp_path / "must_not_be_touched").exists(), "the stand cache dir was touched"
