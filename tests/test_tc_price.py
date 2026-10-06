"""scripts/analysis/tc_price.py — TC: the unmatched lane's tail-cell rule (the shipped hybrid's cell-mean branch,
`build_submission.nf_hybrid`, on rows whose cell estimate is at or above T_TAIL_S).

Registration: plans/PREREG_tc_tail_cell_routing_2026_10_05.md (REG 9eaf025b). Constructed rows only (synthetic
parquet files in tmp_path): `population`'s three-way filter and its T_TAIL_S boundary, `alt_value`'s formula and
floor, `check_reconstruction`'s mixture/storm-shift reconstruction and its two refusals, `load`'s LOMO read, raw
join and its four refusals, `count`'s label-free aggregation, `verdict`'s four classes and its label-column
refusal, and the CLI's two commands, overwrite guard and sha256 manifest. The label rule is absolute: no test here
ever reads a real file, and `verdict` is exercised only on data this file constructs.

Mutation rehearsal (every decision line, one at a time, module restored and re-imported after each): recorded in
logs/tc_1005/mutation_tc.txt, not here.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TC = _load("tc_price", "research/analysis/tc_price.py")
T_TAIL_S = TC.bs.T_TAIL_S                     # the shipped threshold (3,000.0 s); never duplicated as a literal


# ==================================================================================== shared fixture helpers

def _rows(specs, with_y=False) -> pd.DataFrame:
    """A NOLABEL_COLS (+y) frame from `specs` (list of dicts). Each spec may override any of: id, lane,
    decided_by, month, day, ap, pred, calibration_shift, unm_p_fill, unm_nf_cells, unm_body, storm_shift, y.
    Defaults make a row population-EXCLUDED (lane "matched", decided_by "rome", cells 0.0) unless a spec opts in,
    so a test's other rows never silently join the population. Builds every column explicitly (never a bare
    `pd.DataFrame(list_of_dicts)`) so a ZERO-row call still yields the full NOLABEL_COLS schema instead of a
    columnless frame -- the shape `load`/`count` actually request from a real (possibly empty) month file."""
    dtypes = {"MVT_ID_mvt": "float64", "lane": "object", "decided_by": "object", "month": "int64", "day": "object",
              "ap": "object", "pred": "float64", "calibration_shift": "float64", "unm_p_fill": "float64",
              "unm_nf_cells": "float64", "unm_body": "float64", "storm_shift": "float64"}
    if with_y:
        dtypes["y"] = "float64"
    cols: dict = {k: [] for k in dtypes}
    for i, s in enumerate(specs):
        cols["MVT_ID_mvt"].append(float(s.get("id", i)))
        cols["lane"].append(str(s.get("lane", "matched")))
        cols["decided_by"].append(str(s.get("decided_by", "rome")))
        cols["month"].append(int(s.get("month", 1)))
        cols["day"].append(str(s.get("day", "2025-01-10")))
        cols["ap"].append(str(s.get("ap", "EDDF")))
        cols["pred"].append(float(s.get("pred", 1000.0)))
        cols["calibration_shift"].append(float(s.get("calibration_shift", np.nan)))
        cols["unm_p_fill"].append(float(s.get("unm_p_fill", np.nan)))
        cols["unm_nf_cells"].append(float(s.get("unm_nf_cells", 0.0)))
        cols["unm_body"].append(float(s.get("unm_body", 1000.0)))
        cols["storm_shift"].append(float(s.get("storm_shift", np.nan)))
        if with_y:
            cols["y"].append(float(s.get("y", 1000.0)))
    return pd.DataFrame({k: pd.Series(v, dtype=dtypes[k]) for k, v in cols.items()})


def _lomo_dir(tmp_path, by_month: dict, with_y=False, subdir="lomo") -> pathlib.Path:
    """by_month: {month: [specs]}. Writes oof_M{mm:02d}.parquet per key, forcing each row's `month` to match the
    file (a spec's own `month` override is for cross-month tests and is otherwise harmless)."""
    d = tmp_path / subdir
    d.mkdir(exist_ok=True)
    for m, specs in by_month.items():
        frame = _rows(specs, with_y=with_y)
        frame["month"] = int(m)
        frame.to_parquet(d / f"oof_M{int(m):02d}.parquet", index=False)
    return d


def _write_raw(path: pathlib.Path, specs) -> pathlib.Path:
    """specs: [(id, sched_iso_or_None, mvt_iso_or_None), ...] -> the 3-column raw file `load` reads."""
    ids = [float(s[0]) for s in specs]
    sched = [pd.Timestamp(s[1], tz="UTC") if s[1] else pd.NaT for s in specs]
    mvt = [pd.Timestamp(s[2], tz="UTC") if s[2] else pd.NaT for s in specs]
    df = pd.DataFrame({"MVT_ID_mvt": pd.Series(ids, dtype="float64"),
                       "SCHED_TIME_UTC_mvt": pd.Series(sched, dtype="datetime64[us, UTC]"),
                       "MVT_TIME_UTC_mvt": pd.Series(mvt, dtype="datetime64[us, UTC]")})
    df.to_parquet(path, index=False)
    return path


def _raw_dir(tmp_path, by_month: dict, subdir="raw") -> pathlib.Path:
    """by_month: {month: [(id, sched_iso, mvt_iso), ...]}. Writes exactly one training_2025-{mm:02d}-01_a.parquet
    per key (the single-file case; tests of the 'several files' refusal write a second file directly)."""
    d = tmp_path / subdir
    d.mkdir(exist_ok=True)
    for m, specs in by_month.items():
        _write_raw(d / f"training_2025-{int(m):02d}-01_a.parquet", specs)
    return d


POP_SPEC = {"lane": "unmatched", "decided_by": "unmatched_body", "unm_nf_cells": T_TAIL_S}


# ============================================================================================ population()

def test_population_requires_lane_unmatched():
    """A matched row with every other population qualifier set is excluded. Mutation: drop the lane check from
    population() -> this row joins the population -> red."""
    frame = _rows([{**POP_SPEC, "lane": "matched"}])
    assert TC.population(frame).tolist() == [False]


def test_population_requires_decided_by_in_scope():
    """decided_by outside SCOPE (here 'rome') excludes an otherwise-qualifying row. Mutation: drop the decided_by
    check -> red."""
    frame = _rows([{**POP_SPEC, "decided_by": "rome"}])
    assert TC.population(frame).tolist() == [False]


def test_population_accepts_either_scope_value():
    """Both SCOPE values ('unmatched_body' and 'storm') qualify. Mutation: narrow the scope check to one literal
    value -> the second row drops out -> red."""
    frame = _rows([{**POP_SPEC, "decided_by": "unmatched_body"}, {**POP_SPEC, "id": 1, "decided_by": "storm"}])
    assert TC.population(frame).tolist() == [True, True]


def test_population_boundary_at_t_tail_s_is_in():
    """unm_nf_cells exactly at T_TAIL_S is IN the population (nf_hybrid keeps the cell mean at the boundary: 'a
    cell AT the threshold keeps its own load'). Mutation: change >= to > in population() -> red."""
    frame = _rows([{**POP_SPEC, "unm_nf_cells": T_TAIL_S}])
    assert TC.population(frame).tolist() == [True]


def test_population_just_below_t_tail_s_is_out():
    """unm_nf_cells 1 s below T_TAIL_S is OUT (nf_hybrid would route it to the fit, not the cell). Mutation: change
    >= to > in population() -> this row flips to True -> red."""
    frame = _rows([{**POP_SPEC, "unm_nf_cells": T_TAIL_S - 1.0}])
    assert TC.population(frame).tolist() == [False]


def test_population_mirrors_nf_hybrid_on_a_grid():
    """Cross-check against the REAL shipped nf_hybrid: a row is in TC's population iff nf_hybrid keeps its cell mean
    (output == cells, not == fit). Mutation: any divergence between population()'s comparison and nf_hybrid's own
    -> red."""
    cells = np.array([T_TAIL_S - 500.0, T_TAIL_S - 1.0, T_TAIL_S, T_TAIL_S + 1.0, T_TAIL_S + 500.0])
    fit = np.full(cells.shape, 10.0)                       # far from any cell value, so output identity is unambiguous
    kept_cell = TC.bs.nf_hybrid(cells, fit) == cells
    frame = _rows([{**POP_SPEC, "id": i, "unm_nf_cells": float(c)} for i, c in enumerate(cells)])
    assert TC.population(frame).tolist() == kept_cell.tolist()


# ================================================================================================ alt_value()

def test_alt_value_formula():
    """rint(max(pred - (1-p)*(cells-body), 1)) on hand numbers: pred=4000, p=0.5, cells=5000, body=1000 ->
    4000 - 0.5*4000 = 2000. Mutation: change (1 - p) to p -> 4000 - 0.5*4000 is still 2000 by coincidence at
    p=0.5, so this test alone would not catch it; test_alt_value_one_minus_p below does."""
    out = TC.alt_value(np.array([4000.0]), np.array([0.5]), np.array([5000.0]), np.array([1000.0]))
    assert out.tolist() == [2000.0]


def test_alt_value_one_minus_p():
    """p=0.25: pred - 0.75*(cells-body) = 4000 - 0.75*4000 = 1000, which differs from the p-not-(1-p) mistake
    (4000 - 0.25*4000 = 3000). Mutation: change (1 - p) to p in alt_value -> 3000.0 != 1000.0 -> red."""
    out = TC.alt_value(np.array([4000.0]), np.array([0.25]), np.array([5000.0]), np.array([1000.0]))
    assert out.tolist() == [1000.0]


def test_alt_value_floors_at_one():
    """pred far below the (1-p)*(cells-body) term floors at 1, not a negative number. Mutation: drop the
    np.maximum floor -> red."""
    out = TC.alt_value(np.array([10.0]), np.array([0.0]), np.array([5000.0]), np.array([0.0]))
    assert out.tolist() == [1.0]


def test_alt_value_rounds_to_the_nearest_integer():
    """pred - (1-p)*(cells-body) = 2000.6 rounds to 2001. Mutation: drop np.rint -> 2000.6 != 2001.0 -> red."""
    out = TC.alt_value(np.array([4000.6]), np.array([0.5]), np.array([5000.0]), np.array([1000.0]))
    assert out.tolist() == [2001.0]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("slot", ["pred", "p", "cells", "body"])
def test_alt_value_refuses_non_finite_input(bad, slot):
    """Any one of the four inputs non-finite refuses, naming no row count beyond the type. Mutation: drop the
    isfinite check (or check only one array) -> red for the slot(s) left unchecked."""
    vals = {"pred": 4000.0, "p": 0.5, "cells": 5000.0, "body": 1000.0}
    vals[slot] = bad
    with pytest.raises(ValueError) as exc:
        TC.alt_value(np.array([vals["pred"]]), np.array([vals["p"]]), np.array([vals["cells"]]),
                     np.array([vals["body"]]))
    assert "finite" in str(exc.value)


# ========================================================================================= check_reconstruction()

def _recon_row(p, sp, cells, storm_shift=np.nan, calibration_shift=np.nan, decided_by="unmatched_body", **over):
    pred = float(np.rint(TC.bs.mixture(np.array([p]), np.array([sp]), np.array([cells]))[0]))
    pred += 0.0 if (storm_shift is None or (isinstance(storm_shift, float) and np.isnan(storm_shift))) else storm_shift
    row = {**POP_SPEC, "decided_by": decided_by, "unm_p_fill": p, "unm_nf_cells": cells, "pred": pred,
           "storm_shift": storm_shift, "calibration_shift": calibration_shift}
    row.update(over)
    return row, pred


def _frame_with_sp(specs):
    frame = _rows(specs)
    frame["sp"] = [s.get("sp", np.nan) for s in specs]
    return frame


def test_check_reconstruction_empty_mask_returns_zero():
    """No population row short-circuits before reading any column. Mutation: drop the n == 0 branch -> a .max() on
    an empty array raises instead of returning {"n": 0, ...} -> red."""
    frame = _frame_with_sp([{**POP_SPEC, "sp": 100.0}])
    out = TC.check_reconstruction(frame, np.array([False]))
    assert out == {"n": 0, "max_abs_diff_s": 0.0}


def test_check_reconstruction_passes_within_tolerance():
    """pred built exactly from rint(mixture(p, sp, cells)) reconstructs with max_abs_diff_s == 0.0. Mutation:
    change the mixture call's argument order (e.g. mixture(sp, p, cells)) -> diff != 0 -> red."""
    row, pred = _recon_row(p=0.4, sp=500.0, cells=T_TAIL_S)
    row["sp"] = 500.0
    frame = _frame_with_sp([row])
    out = TC.check_reconstruction(frame, np.array([True]))
    assert out["n"] == 1
    assert out["max_abs_diff_s"] == pytest.approx(0.0, abs=1e-9)


def test_check_reconstruction_storm_row_adds_shift():
    """A storm row's expected value is rint(mixture(...)) PLUS storm_shift. Built with shift=75.0 and reconstructs
    exactly. Mutation: drop '+ storm_shift' from expected -> max_abs_diff_s == 75.0 != 0 -> red."""
    row, pred = _recon_row(p=0.4, sp=500.0, cells=T_TAIL_S, storm_shift=75.0, decided_by="storm")
    row["sp"] = 500.0
    frame = _frame_with_sp([row])
    out = TC.check_reconstruction(frame, np.array([True]))
    assert out["n"] == 1
    assert out["max_abs_diff_s"] == pytest.approx(0.0, abs=1e-9)


def test_check_reconstruction_without_the_storm_shift_fails_tolerance():
    """Same row as above but `pred` omits the storm shift: proves the shift is load-bearing for this check, not
    incidentally zero. Mutation: none needed -- this IS the mutation (the shift term removed from `pred`); a
    passing check_reconstruction here would mean the function ignores storm_shift -> red."""
    pred_no_shift = float(np.rint(TC.bs.mixture(np.array([0.4]), np.array([500.0]), np.array([T_TAIL_S]))[0]))
    row = {**POP_SPEC, "decided_by": "storm", "unm_p_fill": 0.4, "unm_nf_cells": T_TAIL_S, "pred": pred_no_shift,
          "storm_shift": 75.0, "calibration_shift": np.nan, "sp": 500.0}
    frame = _frame_with_sp([row])
    with pytest.raises(ValueError) as exc:
        TC.check_reconstruction(frame, np.array([True]))
    assert "tolerance" in str(exc.value)


def test_check_reconstruction_refuses_nonzero_nonnull_calibration_shift():
    """A population row with calibration_shift=5.0 (non-zero, non-null) refuses before any tolerance check.
    Mutation: drop the calibration_shift guard -> no raise -> red."""
    row, pred = _recon_row(p=0.4, sp=500.0, cells=T_TAIL_S, calibration_shift=5.0)
    row["sp"] = 500.0
    frame = _frame_with_sp([row])
    with pytest.raises(ValueError) as exc:
        TC.check_reconstruction(frame, np.array([True]))
    assert "calibration_shift" in str(exc.value)


def test_check_reconstruction_allows_null_calibration_shift():
    """calibration_shift left NaN (null) is not a refusal. Mutation: treat NaN as non-zero (e.g. drop the isfinite
    guard in the check) -> a passing reconstruction instead raises -> red."""
    row, pred = _recon_row(p=0.4, sp=500.0, cells=T_TAIL_S, calibration_shift=np.nan)
    row["sp"] = 500.0
    frame = _frame_with_sp([row])
    out = TC.check_reconstruction(frame, np.array([True]))
    assert out["n"] == 1


def test_check_reconstruction_refuses_over_tolerance_with_count_and_max():
    """pred deliberately off by 50 s (RECON_TOL_S is 1.0) refuses, naming the over-tolerance count and the max
    diff. Mutation: compare with >= instead of > (harmless here) or drop the message's count/max -> the substring
    assertions go red."""
    row, pred = _recon_row(p=0.4, sp=500.0, cells=T_TAIL_S)
    row["pred"] = pred + 50.0
    row["sp"] = 500.0
    frame = _frame_with_sp([row])
    with pytest.raises(ValueError) as exc:
        TC.check_reconstruction(frame, np.array([True]))
    msg = str(exc.value)
    assert "1 row" in msg
    assert "50" in msg


# =================================================================================================== load()

def test_load_concatenates_months_and_attaches_sp_for_population_rows(tmp_path):
    """One population row per month (1 and 7); sp attaches from the matching raw file. Mutation: swap the
    subtraction order (sched - mvt instead of mvt - sched) -> sign flips -> red."""
    lomo = _lomo_dir(tmp_path, {1: [{**POP_SPEC, "id": 1.0}], 7: [{**POP_SPEC, "id": 7.0}]})
    raw = _raw_dir(tmp_path, {1: [(1.0, "2025-01-10T10:00:00", "2025-01-10T10:05:00")],
                              7: [(7.0, "2025-07-05T08:00:00", "2025-07-05T08:20:00")]})
    frame = TC.load(lomo, raw, (1, 7), with_labels=False)
    assert len(frame) == 2
    got = dict(zip(frame.MVT_ID_mvt, frame.sp))
    assert got[1.0] == pytest.approx(300.0)
    assert got[7.0] == pytest.approx(1200.0)


def test_load_with_labels_true_reads_y(tmp_path):
    """with_labels=True surfaces the y column from a (synthetic) file that has one. Mutation: drop the +['y'] in
    the column list -> KeyError/missing column instead of a populated y -> red."""
    lomo = _lomo_dir(tmp_path, {1: [{**POP_SPEC, "id": 1.0, "y": 777.0}]}, with_y=True)
    raw = _raw_dir(tmp_path, {1: [(1.0, "2025-01-10T10:00:00", "2025-01-10T10:05:00")]})
    frame = TC.load(lomo, raw, (1,), with_labels=True)
    assert "y" in frame.columns
    assert frame.y.tolist() == [777.0]


def test_load_with_labels_false_never_reads_y_even_when_the_file_has_one(tmp_path):
    """A file that DOES have a y column, loaded with with_labels=False, never surfaces it (pandas/pyarrow never
    materialise a column outside `columns=`). Mutation: always read +['y'] regardless of with_labels -> 'y' appears
    in frame.columns -> red."""
    lomo = _lomo_dir(tmp_path, {1: [{**POP_SPEC, "id": 1.0, "y": 999999.0}]}, with_y=True)
    raw = _raw_dir(tmp_path, {1: [(1.0, "2025-01-10T10:00:00", "2025-01-10T10:05:00")]})
    frame = TC.load(lomo, raw, (1,), with_labels=False)
    assert "y" not in frame.columns


def test_load_refuses_missing_month_file(tmp_path):
    """Month 7 requested but no oof_M07.parquet exists. Mutation: drop the exists() check (let pandas raise its
    own, unnamed error) -> the message no longer names the file -> red."""
    lomo = _lomo_dir(tmp_path, {1: [{**POP_SPEC, "id": 1.0}]})
    raw = _raw_dir(tmp_path, {1: [(1.0, "2025-01-10T10:00:00", "2025-01-10T10:05:00")]})
    with pytest.raises(ValueError) as exc:
        TC.load(lomo, raw, (1, 7), with_labels=False)
    assert "oof_M07" in str(exc.value)


def test_load_refuses_no_raw_file_for_a_month(tmp_path):
    """Month 1's LOMO file exists but no raw file matches training_2025-01-*.parquet. Mutation: drop the
    'exactly 1' check (len(matches) != 1) -> a silent empty join instead of a refusal -> red."""
    lomo = _lomo_dir(tmp_path, {1: [{**POP_SPEC, "id": 1.0}]})
    raw_empty_dir = tmp_path / "raw_empty"
    raw_empty_dir.mkdir()
    with pytest.raises(ValueError) as exc:
        TC.load(lomo, raw_empty_dir, (1,), with_labels=False)
    assert "01" in str(exc.value)
    assert "0 raw file" in str(exc.value)


def test_load_refuses_several_raw_files_for_a_month(tmp_path):
    """Two files both match training_2025-01-*.parquet. Mutation: drop the 'exactly 1' check -> the first match is
    silently used -> red."""
    lomo = _lomo_dir(tmp_path, {1: [{**POP_SPEC, "id": 1.0}]})
    raw = _raw_dir(tmp_path, {1: [(1.0, "2025-01-10T10:00:00", "2025-01-10T10:05:00")]})
    _write_raw(raw / "training_2025-01-02_b.parquet", [(1.0, "2025-01-10T10:00:00", "2025-01-10T10:05:00")])
    with pytest.raises(ValueError) as exc:
        TC.load(lomo, raw, (1,), with_labels=False)
    assert "2 raw file" in str(exc.value)


def test_load_refuses_duplicated_mvt_id_across_lomo_files(tmp_path):
    """The same MVT_ID_mvt appears in both month files. Mutation: drop the LOMO-side duplicate check -> the
    concatenated frame silently carries two rows for one id -> red."""
    lomo = _lomo_dir(tmp_path, {1: [{**POP_SPEC, "id": 1.0}], 7: [{**POP_SPEC, "id": 1.0}]})
    raw = _raw_dir(tmp_path, {1: [(1.0, "2025-01-10T10:00:00", "2025-01-10T10:05:00")],
                              7: [(1.0, "2025-07-05T08:00:00", "2025-07-05T08:20:00")]})
    with pytest.raises(ValueError) as exc:
        TC.load(lomo, raw, (1, 7), with_labels=False)
    assert "duplicate" in str(exc.value)


def test_load_refuses_duplicated_mvt_id_in_the_raw_file(tmp_path):
    """The raw file for month 1 has two rows sharing an id. The message must name the count AND the raw file, not
    just the word 'duplicate': pandas' own reindex() happens to raise a ValueError mentioning 'duplicate labels'
    too (cannot reindex on an axis with duplicate labels), so a weaker assertion would not tell the two apart.
    Mutation: drop the raw-side duplicate check -> pandas' own error still says 'duplicate', but never names the
    raw file -> the filename assertion goes red (found live: the plain 'duplicate' substring check alone
    survived this exact mutant)."""
    lomo = _lomo_dir(tmp_path, {1: [{**POP_SPEC, "id": 1.0}]})
    raw = _raw_dir(tmp_path, {1: [(1.0, "2025-01-10T10:00:00", "2025-01-10T10:05:00"),
                                   (1.0, "2025-01-10T11:00:00", "2025-01-10T11:05:00")]})
    with pytest.raises(ValueError) as exc:
        TC.load(lomo, raw, (1,), with_labels=False)
    assert "1 duplicate" in str(exc.value)
    assert "training_2025-01-01_a.parquet" in str(exc.value)


def test_load_refuses_population_row_with_no_matching_raw_row(tmp_path):
    """A population row's id is not present in the raw file at all. Mutation: drop the post-join isna() refusal on
    population rows -> the row silently keeps sp = NaN -> red."""
    lomo = _lomo_dir(tmp_path, {1: [{**POP_SPEC, "id": 1.0}]})
    raw = _raw_dir(tmp_path, {1: [(999.0, "2025-01-10T10:00:00", "2025-01-10T10:05:00")]})
    with pytest.raises(ValueError) as exc:
        TC.load(lomo, raw, (1,), with_labels=False)
    assert "population row" in str(exc.value)


def test_load_refuses_population_row_with_null_sched_or_mvt(tmp_path):
    """The id matches, but the raw row's SCHED_TIME_UTC_mvt is null. Mutation: compute sp without going through
    pandas NaT propagation (e.g. .fillna before the subtraction) -> the refusal never fires -> red."""
    lomo = _lomo_dir(tmp_path, {1: [{**POP_SPEC, "id": 1.0}]})
    raw = _raw_dir(tmp_path, {1: [(1.0, None, "2025-01-10T10:05:00")]})
    with pytest.raises(ValueError) as exc:
        TC.load(lomo, raw, (1,), with_labels=False)
    assert "population row" in str(exc.value)


def test_load_leaves_sp_nan_for_non_population_rows_without_raising(tmp_path):
    """A non-population row (lane matched) with NO matching raw row does not raise, and keeps sp NaN. Mutation:
    apply the population-row refusal to every row regardless of mask -> this non-population row now raises -> red."""
    lomo = _lomo_dir(tmp_path, {1: [{**POP_SPEC, "id": 1.0}, {"id": 2.0, "lane": "matched"}]})
    raw = _raw_dir(tmp_path, {1: [(1.0, "2025-01-10T10:00:00", "2025-01-10T10:05:00")]})
    frame = TC.load(lomo, raw, (1,), with_labels=False)
    row2 = frame.loc[frame.MVT_ID_mvt == 2.0].iloc[0]
    assert np.isnan(row2.sp)


# ==================================================================================================== count()

def test_count_on_files_without_a_y_column_at_all(tmp_path):
    """The written LOMO files have no 'y' column whatsoever; count must not need raw_dir or labels. Mutation:
    have count() request NOLABEL_COLS + ['y'] -> a column-not-found error on these files -> red."""
    lomo = _lomo_dir(tmp_path, {1: [{**POP_SPEC, "id": 1.0}], 7: []}, with_y=False)
    out = TC.count(lomo, (1, 7))
    assert out["n"] == 1
    assert "y" not in pd.read_parquet(lomo / "oof_M01.parquet").columns   # the fixture itself has no label column


def test_count_by_month_and_airport_zero_fills_requested_months(tmp_path):
    """January has zero qualifying rows, July has two at different airports; by_month zero-fills January, by_airport
    only lists airports that occur. Mutation: build by_month only from value_counts() (no zero-fill) -> the '1' key
    is missing from by_month -> red."""
    lomo = _lomo_dir(tmp_path, {1: [{"lane": "matched"}],
                                7: [{**POP_SPEC, "id": 1.0, "ap": "LTFM"}, {**POP_SPEC, "id": 2.0, "ap": "LTFM"}]})
    out = TC.count(lomo, (1, 7))
    assert out["by_month"] == {1: 0, 7: 2}
    assert out["by_airport"] == {"LTFM": 2}
    assert out["n"] == 2


# =================================================================================================== verdict()

def test_verdict_refuses_a_frame_without_y():
    """Exact message match (the brief's wording). Mutation: change the message text -> the exact-match assertion
    goes red."""
    frame = _rows([{**POP_SPEC}], with_y=False)
    with pytest.raises(ValueError) as exc:
        TC.verdict(frame)
    assert str(exc.value) == "verdict needs the label column y"


def test_verdict_not_testable_when_n_is_zero():
    """No row satisfies population() (lane matched): NOT TESTABLE, n=0, k=0, G_bmse=0.0, rows=[]. Mutation: return
    FALSE instead of NOT TESTABLE for n == 0 -> red."""
    frame = _rows([{"lane": "matched"}], with_y=True)
    out = TC.verdict(frame)
    assert out == {"verdict": "NOT TESTABLE", "n": 0, "k": 0, "G_bmse": 0.0, "rows": [],
                   "registration": TC.REGISTRATION}


def test_verdict_g_bmse_is_the_term_sum_divided_by_n_board():
    """Pins the exact divisor, not just G's sign: one row with a hand-computable term ((0-5000)**2-(0-1000)**2 =
    25,000,000-1,000,000 = 24,000,000, every operand an exactly-representable integer) gives G_bmse ==
    24,000,000 / N_BOARD exactly -- a single deterministic float64 division of the same two values is bit-identical
    whether taken here or inside verdict(), so '==' (not approx) is the correct, justified assertion. Mutation:
    divide by N_BOARD * 2 (or any other constant) instead of N_BOARD -> red (found live: a doubled divisor
    survived every other test in this file, since they only check G's sign or a sum that happens to be zero)."""
    row = {**POP_SPEC, "id": 1.0, "unm_p_fill": 0.0, "unm_nf_cells": 5000.0, "unm_body": 1000.0, "pred": 5000.0,
          "y": 0.0}
    frame = _rows([row], with_y=True)
    out = TC.verdict(frame)
    assert out["G_bmse"] == 24_000_000.0 / TC.N_BOARD


def test_verdict_true_when_g_positive_and_k_at_least_two():
    """Two population rows, each with y much closer to alt than to pred (p=0 so alt = pred - (cells - body); large
    gain): G > 0 and k == 2. Mutation: require k >= 1 instead of k >= 2 -> would still read TRUE here (not a
    discriminating case), but test_verdict_not_established_when_k_is_one below catches that mutant; this test pins
    the TRUE branch itself: flip the G > 0 check to G >= 0 and a FALSE-only frame would wrongly read TRUE -> the
    FALSE test below catches it; here, dropping the k >= 2 term entirely (always True) cannot be distinguished from
    this single test, so it is combined with test_verdict_not_established_when_k_is_one for full coverage."""
    rows = [{**POP_SPEC, "id": 1.0, "unm_p_fill": 0.0, "unm_nf_cells": 5000.0, "unm_body": 1000.0, "pred": 5000.0,
            "y": 1000.0},
           {**POP_SPEC, "id": 2.0, "unm_p_fill": 0.0, "unm_nf_cells": 5000.0, "unm_body": 1000.0, "pred": 5000.0,
            "y": 1000.0}]
    frame = _rows(rows, with_y=True)
    out = TC.verdict(frame)
    assert out["verdict"] == "TRUE"
    assert out["n"] == 2 and out["k"] == 2
    assert out["G_bmse"] > 0
    assert out["registration"] == TC.REGISTRATION
    assert len(out["rows"]) == 2
    for r in out["rows"]:
        assert set(r) == {"ap", "day", "pred", "alt", "cells", "body", "p", "y", "term"}


def test_verdict_false_when_g_not_positive():
    """One population row where alt is FARTHER from y than pred (pred==y exactly, alt far off): term < 0, G <= 0.
    Mutation: change the FALSE condition from G <= 0 to G < 0 -> a G exactly 0.0 case would misclassify, but this
    case has G strictly negative, so a >= -> > typo is what this test is built to catch -> red."""
    row = {**POP_SPEC, "id": 1.0, "unm_p_fill": 0.0, "unm_nf_cells": 5000.0, "unm_body": 1000.0, "pred": 1000.0,
          "y": 1000.0}
    frame = _rows([row], with_y=True)
    out = TC.verdict(frame)
    assert out["verdict"] == "FALSE"
    assert out["G_bmse"] < 0


def test_verdict_false_when_g_is_exactly_zero_with_k_at_least_two():
    """term is linear in y once pred/alt are fixed ((alt-pred)*(2y-pred-alt)), so y is chosen per row to hit an
    EXACT term: rows 1-2 at +100,000 each (k=2), row 3 at -200,000 (cells kept >= T_TAIL_S so it stays IN the
    population too), summing to exactly 0 -> G_bmse == 0.0 exactly (0.0 / N_BOARD is exact in IEEE754). G == 0
    must read FALSE, not TRUE. Mutation: change the TRUE condition's G > 0 to G >= 0 -> this k=2, G=0.0 case
    misreads TRUE -> red (found live: this mutant survived the suite before this test was added; every other
    verdict test happened to use a strictly negative or strictly positive G)."""
    row_a = {**POP_SPEC, "id": 1.0, "unm_p_fill": 0.0, "unm_nf_cells": 5000.0, "unm_body": 1000.0, "pred": 5000.0,
            "y": 2987.5}
    row_b = {**POP_SPEC, "id": 2.0, "unm_p_fill": 0.0, "unm_nf_cells": 5000.0, "unm_body": 1000.0, "pred": 5000.0,
            "y": 2987.5}
    row_c = {**POP_SPEC, "id": 3.0, "unm_p_fill": 0.0, "unm_nf_cells": 5000.0, "unm_body": 9000.0, "pred": 1000.0,
            "y": 2975.0}
    frame = _rows([row_a, row_b, row_c], with_y=True)
    assert TC.population(frame).tolist() == [True, True, True]          # all three rows must count
    out = TC.verdict(frame)
    assert out["k"] == 2
    assert out["G_bmse"] == 0.0
    assert out["verdict"] == "FALSE"


def test_verdict_not_established_when_k_is_one_and_g_positive():
    """Two rows: one with a large positive term (dominates the sum, G > 0) and one with a small negative term
    (k stays at 1). Mutation: change k >= 2 to k >= 1 in the TRUE condition -> this case misreads TRUE -> red."""
    big_gain = {**POP_SPEC, "id": 1.0, "unm_p_fill": 0.0, "unm_nf_cells": 5000.0, "unm_body": 1000.0, "pred": 5000.0,
               "y": 1000.0}
    small_loss = {**POP_SPEC, "id": 2.0, "unm_p_fill": 0.0, "unm_nf_cells": 3000.0, "unm_body": 2999.0,
                 "pred": 3000.0, "y": 3000.0}
    frame = _rows([big_gain, small_loss], with_y=True)
    out = TC.verdict(frame)
    assert out["k"] == 1
    assert out["G_bmse"] > 0
    assert out["verdict"] == "NOT ESTABLISHED"


# ======================================================================================================== CLI


def _cli_lomo_and_raw(tmp_path, with_labels: bool, n_pop_rows_by_month=(0, 3)):
    specs_by_month = {}
    raw_by_month = {}
    next_id = 1.0
    for month, n in zip((1, 7), n_pop_rows_by_month):
        specs = []
        raw_specs = []
        for _ in range(n):
            rid = next_id
            next_id += 1.0
            sp = 500.0 + rid
            pred = float(np.rint(TC.bs.mixture(np.array([0.3]), np.array([sp]), np.array([T_TAIL_S]))[0]))
            spec = {**POP_SPEC, "id": rid, "unm_p_fill": 0.3, "unm_nf_cells": T_TAIL_S, "pred": pred, "month": month}
            if with_labels:
                spec["y"] = pred - 100.0
            specs.append(spec)
            sched = f"2025-{month:02d}-10T10:00:00"
            mvt_ts = pd.Timestamp(sched, tz="UTC") + pd.Timedelta(seconds=sp)
            raw_specs.append((rid, sched, mvt_ts.isoformat()))
        specs_by_month[month] = specs
        raw_by_month[month] = raw_specs
    lomo = _lomo_dir(tmp_path, specs_by_month, with_y=with_labels)
    raw = _raw_dir(tmp_path, raw_by_month)
    return lomo, raw


def test_cli_count_round_trip(tmp_path):
    """main(['count', ...]) writes exactly what count() returns. Mutation: write a different dict (e.g. drop a key)
    -> the equality check goes red."""
    lomo = _lomo_dir(tmp_path, {1: [], 7: [{**POP_SPEC, "id": 1.0}]})
    out_path = tmp_path / "count.json"
    rc = TC.main(["count", "--lomo-dir", str(lomo), "--months", "1,7", "--out", str(out_path)])
    assert rc == 0
    written = json.loads(out_path.read_text())
    direct = TC.count(lomo, (1, 7))
    assert written["n"] == direct["n"]
    assert {int(k): v for k, v in written["by_month"].items()} == direct["by_month"]


def test_cli_count_refuses_an_existing_output(tmp_path):
    """Running count twice at the same --out refuses the second time. Mutation: drop the exists() guard -> the
    second run silently overwrites -> red (no FileExistsError raised)."""
    lomo = _lomo_dir(tmp_path, {1: [], 7: []})
    out_path = tmp_path / "count.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert TC.main(["count", "--lomo-dir", str(lomo), "--months", "1,7", "--out", str(out_path)]) == 0
    with pytest.raises(FileExistsError):
        TC.main(["count", "--lomo-dir", str(lomo), "--months", "1,7", "--out", str(out_path)])


def test_cli_verdict_round_trip_on_synthetic_data_only(tmp_path):
    """main(['verdict', ...]) on SYNTHETIC files only: runs check_reconstruction first (stored), then verdict
    (stored), then sha256 of the 4 input files (2 LOMO + 2 raw). Mutation: skip storing check_reconstruction's
    result -> the key is absent -> red."""
    lomo, raw = _cli_lomo_and_raw(tmp_path, with_labels=True, n_pop_rows_by_month=(0, 2))
    out_path = tmp_path / "verdict.json"
    rc = TC.main(["verdict", "--lomo-dir", str(lomo), "--raw-dir", str(raw), "--out", str(out_path)])
    assert rc == 0
    written = json.loads(out_path.read_text())
    assert written["registration"] == TC.REGISTRATION
    assert written["n"] == 2
    assert "check_reconstruction" in written and written["check_reconstruction"]["n"] == 2
    assert set(written["input_sha256"]) == {"oof_M01.parquet", "oof_M07.parquet",
                                            "training_2025-01-01_a.parquet", "training_2025-07-01_a.parquet"}
    for sha in written["input_sha256"].values():
        assert isinstance(sha, str) and len(sha) == 64


def test_cli_verdict_refuses_an_existing_output(tmp_path):
    """Mutation: drop the exists() guard on the verdict path -> no FileExistsError on the second run -> red."""
    lomo, raw = _cli_lomo_and_raw(tmp_path, with_labels=True, n_pop_rows_by_month=(0, 1))
    out_path = tmp_path / "verdict.json"
    assert TC.main(["verdict", "--lomo-dir", str(lomo), "--raw-dir", str(raw), "--out", str(out_path)]) == 0
    with pytest.raises(FileExistsError):
        TC.main(["verdict", "--lomo-dir", str(lomo), "--raw-dir", str(raw), "--out", str(out_path)])


def test_cli_verdict_runs_check_reconstruction_before_verdict(tmp_path):
    """A frame that fails check_reconstruction (pred corrupted after being built consistently) must raise from the
    reconstruction check, never reach verdict's label read. Mutation: call verdict() before check_reconstruction()
    -> this still raises (both would, from different causes) only if verdict also happens to fail; the real
    discriminator is the message, asserted below -> swapping the order without changing messages would pass this
    test only coincidentally, so the message assertion is load-bearing."""
    lomo, raw = _cli_lomo_and_raw(tmp_path, with_labels=True, n_pop_rows_by_month=(0, 1))
    # corrupt the one population row's pred so it fails check_reconstruction's tolerance
    f = lomo / "oof_M07.parquet"
    df = pd.read_parquet(f)
    df.loc[0, "pred"] = df.loc[0, "pred"] + 500.0
    df.to_parquet(f, index=False)
    out_path = tmp_path / "verdict.json"
    with pytest.raises(ValueError) as exc:
        TC.main(["verdict", "--lomo-dir", str(lomo), "--raw-dir", str(raw), "--out", str(out_path)])
    assert "tolerance" in str(exc.value)
    assert not out_path.exists()
