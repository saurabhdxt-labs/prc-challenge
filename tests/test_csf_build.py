"""CSF — registered invariants of plans/PREREG_csf_callsign_frequency_fill_2026_09_19.md (c672efca).

This arm's entire signal is a CALLSIGN-IDENTITY COUNT, so its characteristic failure is a counting bug
that invents the result: null callsigns grouped together (BC-15), a row counting ITSELF as its own
twin, or counts leaking across calendar months. Each has its own test below. The second load-bearing
piece is `decide` — all three registered clauses required, and it must be able to return NOT WORKING.

Mutation rehearsal is recorded at the end of this docstring after the escapes were pinned.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


@pytest.fixture(scope="module")
def M():
    return _load("csf_build", "scripts/analysis/csf_build.py")


T = pd.Timestamp


def _mv(mid, cs, when, ap="LIRF", phase="DEP"):
    return {"MVT_ID_mvt": float(mid), "FLIGHT_mvt": cs,
            "MVT_TIME_UTC_mvt": T(when, tz="UTC"), "ADEP_mvt": ap, "PHASE_mvt": phase}


def _frames(recs):
    d = pd.DataFrame(recs)
    return d, d


def test_bars_match_the_registration(M):
    assert M.AUC_BAR == 0.020 and M.FOLDS_BAR == 10 and M.N_FOLDS == 12
    assert M.TWIN_H == 36 and M.NO_TWIN_SENTINEL == 72.0
    assert M.EXPECT_ROWS == 1488 and M.REPLICATION_ATOL == 1e-10
    assert M.MIN_ROWS_CARRYING == 20


# ------------------------------------------------------------------ n_cs

def test_n_cs_counts_same_callsign_in_the_same_month(M):
    recs = [_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "AZA1", "2025-03-20 10:00"),
            _mv(3, "AZA1", "2025-03-28 10:00")]
    rows, pool = _frames(recs)
    assert M.callsign_features(rows, pool, pool).n_cs.tolist() == [3, 3, 3]


def test_n_cs_does_NOT_count_across_calendar_months(M):
    """The registration says 'within the same calendar month'. Leaking across months would make a
    daily airline callsign look rarer or commoner than it is in the month being scored."""
    recs = [_mv(1, "AZA1", "2025-03-31 23:00"), _mv(2, "AZA1", "2025-04-01 01:00")]
    rows, pool = _frames(recs)
    assert M.callsign_features(rows, pool, pool).n_cs.tolist() == [1, 1]


def test_n_cs_is_zero_for_a_NULL_callsign(M):
    """BC-15: `pandas` groups NaN with NaN. Two unrelated no-callsign flights must NOT count each
    other — this arm's whole signal is callsign identity, so that would invent the result."""
    recs = [_mv(1, None, "2025-03-04 10:00"), _mv(2, None, "2025-03-05 10:00"),
            _mv(3, None, "2025-03-06 10:00")]
    rows, pool = _frames(recs)
    assert M.callsign_features(rows, pool, pool).n_cs.tolist() == [0, 0, 0]


def test_n_cs_counts_only_DEP_rows_at_the_ten(M):
    recs = [_mv(1, "AZA1", "2025-03-04 10:00"),
            _mv(2, "AZA1", "2025-03-05 10:00", phase="ARR"),
            _mv(3, "AZA1", "2025-03-06 10:00", ap="ZZZZ")]
    rows = pd.DataFrame([recs[0]])
    pool = pd.DataFrame(recs)
    assert M.callsign_features(rows, pool, pool).n_cs.tolist() == [1]


# ------------------------------------------------------------------ twins

def test_a_row_is_not_its_OWN_twin(M):
    """A single movement has no twin. Self-matching would make has_twin36 identically 1."""
    rows, pool = _frames([_mv(1, "AZA1", "2025-03-04 10:00")])
    f = M.callsign_features(rows, pool, pool)
    assert f.has_twin36.tolist() == [0] and f.twin_h.tolist() == [M.NO_TWIN_SENTINEL]


def test_twin_found_inside_36h(M):
    recs = [_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "AZA1", "2025-03-05 22:00")]
    rows, pool = _frames(recs)
    f = M.callsign_features(rows, pool, pool)
    assert f.has_twin36.tolist() == [1, 1]
    assert f.twin_h.iloc[0] == pytest.approx(36.0)


def test_twin_just_OUTSIDE_36h_is_not_a_twin(M):
    recs = [_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "AZA1", "2025-03-05 22:00:01")]
    rows, pool = _frames(recs)
    f = M.callsign_features(rows, pool, pool)
    assert f.has_twin36.tolist() == [0, 0]
    assert f.twin_h.iloc[0] == pytest.approx(36.0 + 1 / 3600)


def test_twin_uses_the_NEAREST_movement(M):
    recs = [_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "AZA1", "2025-03-04 12:00"),
            _mv(3, "AZA1", "2025-03-05 10:00")]
    rows, pool = _frames(recs)
    assert M.callsign_features(rows, pool, pool).twin_h.iloc[0] == pytest.approx(2.0)


def test_twin_counts_ARR_rows_too(M):
    """`has_twin36` is 'another MOVEMENT', not another departure — the inbound leg is the point."""
    recs = [_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "AZA1", "2025-03-04 08:00", phase="ARR")]
    rows = pd.DataFrame([recs[0]]); pool = pd.DataFrame(recs)
    assert M.callsign_features(rows, pool, pool).has_twin36.tolist() == [1]


def test_NULL_callsign_has_no_twin(M):
    recs = [_mv(1, None, "2025-03-04 10:00"), _mv(2, None, "2025-03-04 11:00")]
    rows, pool = _frames(recs)
    f = M.callsign_features(rows, pool, pool)
    assert f.has_twin36.tolist() == [0, 0]
    assert f.twin_h.tolist() == [M.NO_TWIN_SENTINEL, M.NO_TWIN_SENTINEL]


def test_features_are_aligned_to_the_row_frame(M):
    recs = [_mv(1, "ZZZ9", "2025-03-04 10:00"), _mv(2, "AZA1", "2025-03-04 10:00"),
            _mv(3, "AZA1", "2025-03-04 11:00")]
    rows = pd.DataFrame(recs); pool = pd.DataFrame(recs)
    f = M.callsign_features(rows, pool, pool)
    assert list(f.index) == list(rows.index)
    assert f.has_twin36.tolist() == [0, 1, 1]


def test_features_never_read_a_label(M):
    """Serve-time contract: the runner must not be able to see y even if a caller passes it."""
    recs = [_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "AZA1", "2025-03-04 11:00")]
    rows = pd.DataFrame(recs); pool = pd.DataFrame(recs)
    a = M.callsign_features(rows, pool, pool)
    rows2 = rows.assign(y=[99999.0, 1.0], fill=[True, False], TAXITIME_SEC_mvt=[5.0, 6.0])
    b = M.callsign_features(rows2, pool, pool)
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------ the decision

def test_decide_needs_all_three_clauses(M):
    assert M.decide(0.03, 11, True, 500) == "WORKING"
    assert M.decide(0.019, 11, True, 500) == "NOT WORKING"    # AUC
    assert M.decide(0.03, 9, True, 500) == "NOT WORKING"      # folds
    assert M.decide(0.03, 11, False, 500) == "NOT WORKING"    # trim sign


def test_decide_boundaries_are_the_registered_ones(M):
    assert M.decide(0.020, 10, True, 20) == "WORKING"
    assert M.decide(0.020, 10, True, 19) == "NOT WORKING"


def test_decide_applies_the_tail_rule(M):
    """Pre-committed third shape: a gain carried by <20 rows is NOT WORKING however clean."""
    assert M.decide(0.20, 12, True, 19) == "NOT WORKING"


def test_decide_rejects_a_negative_auc_gain(M):
    assert M.decide(-0.05, 12, True, 500) == "NOT WORKING"


# ------------------------------------------------------------------ the replication guard

def test_replication_guard_rejects_a_mismatch(M):
    a = np.array([0.1, 0.2, 0.3])
    M.assert_replicates(a, a.copy())
    with pytest.raises(ValueError, match="replicate"):
        M.assert_replicates(a, a + 1e-9)


def test_replication_guard_accepts_float_noise_below_atol(M):
    a = np.array([0.1, 0.2, 0.3])
    M.assert_replicates(a, a + 1e-12)


# ---------------------------------------------- pinned after the null-guard mutants ESCAPED

def test_pandas_astype_str_PROPAGATES_nan_which_is_what_makes_the_null_path_safe():
    """The assumption the null-callsign guards rest on, pinned so a pandas change cannot hide it.

    On pandas 3.0.1 `Series.astype(str)` on an object column containing None yields **NaN**, not the
    string "None". That is why a null callsign cannot collide with another null in the
    `astype(str) -> value_counts -> map` path, and why the explicit `notna()` / `known` guards in
    `callsign_features` are currently EQUIVALENT MUTANTS (recorded in tests/mutation_exemptions.md).

    On pandas 2.x the same expression yields the STRING "None", and every no-callsign flight would
    count every other no-callsign flight as the same aircraft — silently inventing this arm's signal.
    If this test goes RED, the guards stop being redundant and become load-bearing: do not remove them.
    """
    s = pd.Series([None, None], dtype=object)
    assert s.astype(str).isna().all(), (
        f"pandas {pd.__version__} no longer propagates NaN through astype(str); the null-callsign "
        "guards in callsign_features are now LOAD-BEARING, not redundant")
    assert s.astype(str).value_counts().empty


def test_null_callsign_stays_zero_even_with_a_literal_nan_string_in_the_pool(M):
    """Defence in depth the pandas behaviour does NOT provide: a pool row whose callsign is the
    literal text "nan" must not be counted for a row whose callsign is genuinely missing."""
    recs = [_mv(1, None, "2025-03-04 10:00"), _mv(2, "nan", "2025-03-05 10:00"),
            _mv(3, "nan", "2025-03-06 10:00")]
    rows = pd.DataFrame([recs[0]]); pool = pd.DataFrame(recs)
    f = M.callsign_features(rows, pool, pool)
    assert f.n_cs.tolist() == [0]
    assert f.has_twin36.tolist() == [0] and f.twin_h.tolist() == [M.NO_TWIN_SENTINEL]
