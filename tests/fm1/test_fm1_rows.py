"""FM1's input contract (prc/pipeline/fm1/rows.py) — AT2. Real-shaped rows (the e2e world's slice of the real 2025 files);
each violation raises its named error with the offending column / count."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import features as F
from prc.pipeline.fm1 import knobs as K
from prc.pipeline.fm1 import rows as R

APS = ("EDDF", "LSZH")


@pytest.fixture(scope="module")
def train(e2e_world):
    return pd.read_parquet(e2e_world[0] / "raw" / "training_2025-01-01_2025-02-01.parquet")


@pytest.fixture(scope="module")
def scored(e2e_world):
    return pd.read_parquet(e2e_world[0] / "raw" / "scored.parquet")


@pytest.fixture(scope="module")
def cfg(e2e_world, fm1_world_mod):
    return K.parse_config(fm1_world_mod.config_raw(e2e_world[0]))


def test_real_rows_pass_in_their_mode(train, scored):
    rep = R.validate_raw(train, "train", APS)
    assert rep["n_dep"] + rep["n_arr"] == len(train) and rep["n_dep_matched"] > 0 and rep["n_dep_unmatched"] > 0
    served_rep = R.validate_raw(scored, "serve", APS)
    assert served_rep["mode"] == "serve"
    # BC-28: n_labels_beyond_a_day is a training-mode-only count; a serve call (no labels at all) must report 0,
    # not an initialization leftover. Mutation: initialize n_beyond_a_day to a nonzero default -> this goes red.
    assert served_rep["n_labels_beyond_a_day"] == 0


def test_labelled_departures_are_refused_at_serve(train):
    """A scored file never carries a departure label (no 2026 label exists; a 2025 file passed as scored is refused).
    Mutation: check only TAXITIME (not BLOCK) -> the BLOCK-only case goes green -> red."""
    with pytest.raises(E.LabelLeakError, match=r"\d+ scored departure\(s\) carry a label"):
        R.validate_raw(train, "serve", APS)
    served, labels = R.hide_labels(train)
    R.validate_raw(served, "serve", APS)
    block_only = served.copy()
    i = block_only.index[block_only.PHASE_mvt == "DEP"][0]
    block_only.loc[i, "BLOCK_TIME_UTC_mvt"] = block_only.loc[i, "MVT_TIME_UTC_mvt"]
    with pytest.raises(E.LabelLeakError, match="1 scored departure"):
        R.validate_raw(block_only, "serve", APS)


def test_hide_labels_keeps_arrivals_and_returns_the_departure_labels(train):
    served, labels = R.hide_labels(train)
    dep = train.PHASE_mvt == "DEP"
    assert served.loc[dep, "TAXITIME_SEC_mvt"].isna().all() and served.loc[dep, "BLOCK_TIME_UTC_mvt"].isna().all()
    # arrival taxi-in stays an input: every arrival value unchanged. The label column becomes float64 because it now
    # holds nulls -- exactly how the real scored file reads (data/raw/ranking.parquet: TAXITIME_SEC_mvt float64).
    assert served.TAXITIME_SEC_mvt.dtype == "float64"
    pd.testing.assert_frame_equal(served[~dep], train[~dep], check_dtype=False)
    assert np.array_equal(served.loc[~dep, "TAXITIME_SEC_mvt"].to_numpy(float), train.loc[~dep, "TAXITIME_SEC_mvt"].to_numpy(float))
    assert labels.index.tolist() == train.MVT_ID_mvt[dep].tolist()
    assert np.array_equal(labels.to_numpy(), train.TAXITIME_SEC_mvt[dep].to_numpy(dtype="float64", na_value=np.nan),
                          equal_nan=True)
    assert train.loc[dep, "TAXITIME_SEC_mvt"].notna().any()                 # the input was not modified


@pytest.mark.parametrize("mutate,err,match", [
    (lambda f: f.drop(columns=["STAND_mvt"]), E.MissingColumnError, r"missing column\(s\) \['STAND_mvt'\]"),
    (lambda f: f.assign(extra=1), E.SchemaError, r"unexpected column\(s\) \['extra'\]"),
    (lambda f: f.assign(MVT_TIME_UTC_mvt=f.MVT_TIME_UTC_mvt.dt.tz_localize(None)), E.DtypeError,
     r"MVT_TIME_UTC_mvt: dtype datetime64.* is not a tz-aware UTC timestamp"),
    (lambda f: f.assign(MVT_TIME_UTC_mvt=f.MVT_TIME_UTC_mvt.dt.tz_convert("Europe/Berlin")), E.DtypeError,
     "not a tz-aware UTC timestamp"),
    (lambda f: f.assign(MVT_ID_mvt=f.MVT_ID_mvt.astype(str)), E.DtypeError, r"MVT_ID_mvt: dtype .* is not float"),
    (lambda f: f.iloc[:0], E.SchemaError, "no rows"),
    (lambda f: pd.concat([f, f.iloc[:2]]), E.DuplicateIdError, r"2 duplicate id\(s\)"),
    (lambda f: f.assign(PHASE_mvt=np.where(np.arange(len(f)) == 0, "TAXI", f.PHASE_mvt)), E.SchemaError,
     r"unknown phase\(s\) \['TAXI'\]"),
])
def test_contract_violations_are_named(train, mutate, err, match):
    with pytest.raises(err, match=match):
        R.validate_raw(mutate(train), "train", APS)


def test_unknown_departure_airport_and_missing_mvt_time(train):
    bad = train.copy()
    i = bad.index[bad.PHASE_mvt == "DEP"][0]
    bad.loc[i, "ADEP_mvt"] = "EKCH"
    with pytest.raises(E.UnknownAirportError, match=r"\['EKCH'\] outside the registry"):
        R.validate_raw(bad, "train", APS)
    bad = train.copy()
    bad.loc[i, "MVT_TIME_UTC_mvt"] = pd.NaT
    with pytest.raises(E.RangeError, match=r"1 departure\(s\) without MVT_TIME"):
        R.validate_raw(bad, "train", APS)


def test_training_labels_beyond_a_day_are_accepted_no_upper_bound(train):
    """Production parity (BC-28): a real 2025 date-slip label (Rome-shaped: BLOCK computed across a UTC day
    rollover) can exceed 86,400 s -- stand_ab.py:236/589/1268 keeps every row with TAXITIME_SEC_mvt > 0 and BLOCK
    not null, with NO upper bound, and FM1's AT2 contract (plans/PREREG_fm1_2026_09_28.md sec3) never required
    one. rows.validate_raw used to raise RangeError above 86,400 s (the pre-fix behaviour); the contract now
    mirrors production and accepts it. Mutation: reintroduce the > MAX_TAXI_S check -> this goes red."""
    bad = train.copy()
    i = bad.index[bad.TAXITIME_SEC_mvt.notna()][0]
    bad.loc[i, "TAXITIME_SEC_mvt"] = 90_000
    R.validate_raw(bad, "train", APS)          # must not raise


def test_a_label_exactly_at_the_day_boundary_does_not_count_as_beyond_it(train):
    """DAY_S (86,400 s) is the informational threshold's NAME, not an enforced bound (BC-28): a label exactly at
    the boundary is not "beyond" it. Mutation: `>` -> `>=` in the n_labels_beyond_a_day comparator -> red."""
    base = R.validate_raw(train, "train", APS)["n_labels_beyond_a_day"]
    bad = train.copy()
    i = bad.index[bad.TAXITIME_SEC_mvt.notna()][0]
    bad.loc[i, "TAXITIME_SEC_mvt"] = 86_400.0
    rep = R.validate_raw(bad, "train", APS)
    assert rep["n_labels_beyond_a_day"] == base


def test_n_labels_beyond_a_day_counts_without_refusing(train):
    """`validate_raw`'s report carries n_labels_beyond_a_day as information (BC-28), never a refusal: pins the
    exact count of real training rows (DEP or ARR -- the check is not phase-masked) whose |TAXITIME_SEC_mvt|
    exceeds DAY_S. Mutation: off-by-one in the comparator, dropping `np.abs` (a large-negative label would then
    be missed), or always returning 0 -> the exact-count assert goes red."""
    base = R.validate_raw(train, "train", APS)["n_labels_beyond_a_day"]
    bad = train.copy()
    idx = bad.index[bad.TAXITIME_SEC_mvt.notna()][:2]
    bad.loc[idx[0], "TAXITIME_SEC_mvt"] = 90_000            # +3,600 s over DAY_S, positive
    bad.loc[idx[1], "TAXITIME_SEC_mvt"] = -90_000            # sign must not matter (abs)
    rep = R.validate_raw(bad, "train", APS)
    assert rep["n_labels_beyond_a_day"] == base + 2


def test_regression_bug_28_real_labels_beyond_a_day_are_training_rows(train, cfg):
    """A LIRF-shaped date-slip label is a real, valid training label: 15 real DEP labels across the 12 months of
    2025 exceed 86,400 s (max 131,167 s, Feb 2025, LIRF); 14 of the 15 are this Rome date-slip shape (BLOCK on the
    previous UTC day's clock), 1 is LSZH. Production (stand_ab.py) keeps every row with TAXITIME_SEC_mvt > 0 and
    BLOCK not null -- no upper bound. Before the fix, rows.py:33 MAX_TAXI_S = 86_400 raised RangeError and refused
    FM1 on the real 12-month dry run (logs/cml_scratch/fm1_dryrun_20260928T135914.log). This asserts both halves
    of the contract: (1) validate_raw accepts the row, and (2) the training-row filter (features.unmatched_frame,
    mirroring stand_ab's TAXITIME > 0 & BLOCK notna keep rule) keeps it with the label value UNCHANGED, exactly as
    production would. Mutation: clip TAXITIME in unmatched_frame's `y` column -> the exact-value assert goes red."""
    bad = train.copy()
    dep_unmatched = bad.index[(bad.PHASE_mvt == "DEP") & bad.TAXITIME_SEC_mvt.notna() & bad.AOBT_3_flt.isna()]
    assert len(dep_unmatched) > 0, "fixture must contain an unmatched DEP row with a label to mutate"
    i = dep_unmatched[0]
    mvt_id = bad.loc[i, "MVT_ID_mvt"]
    bad.loc[i, "TAXITIME_SEC_mvt"] = 131_167.0
    bad.loc[i, "BLOCK_TIME_UTC_mvt"] = bad.loc[i, "MVT_TIME_UTC_mvt"] + pd.Timedelta(seconds=131_167)

    rep = R.validate_raw(bad, "train", APS)          # must not raise
    assert rep["mode"] == "train"
    assert rep["n_labels_beyond_a_day"] == 1

    u = F.unmatched_frame(bad, False, cfg)
    row = u[u.MVT_ID_mvt == float(mvt_id)]
    assert len(row) == 1, "the date-slip row must survive the training-row filter"
    assert row.y.iloc[0] == 131_167.0, "the label must reach the training frame unchanged (no clip / no drop)"


def test_bad_arguments():
    with pytest.raises(ValueError, match="mode"):
        R.validate_raw(pd.DataFrame(), "score", APS)
    with pytest.raises(E.SchemaError, match="expected a pandas DataFrame"):
        R.validate_raw([1, 2], "train", APS)


@pytest.mark.parametrize("cols,bad", [
    (["proxy", "mvt_sec"], ["mvt_sec"]), (["y", "sp"], ["y"]), (["delta"], ["delta"]),
    (["TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "w_age_s"], ["TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"]),
    (["aobt_sec", "sched_sec", "dwell_s", "implied_taxi_s"], ["aobt_sec", "sched_sec"]),
])
def test_forbidden_design_columns(cols, bad):
    """Negative invariant (AT2): labels, label derivatives and `*_sec` never enter a design; `*_s` feature names do.
    Mutation: match the suffix `_s` instead of `_sec` -> dwell_s is refused -> red."""
    assert R.forbidden(cols) == bad
    with pytest.raises(E.ForbiddenInputError, match="forbidden input column"):
        R.assert_design_clean(cols)
    R.assert_design_clean([c for c in cols if c not in bad])
