"""scripts/analysis/fmf_checks.py — the FM1 final-fit prereg's label-free gates (plans/PREREG_fm1_final_fit_2026_09_30.md,
amendment FMF.2): `rss` (dry-run peak memory), `same` (determinism: two predict outputs identical), `accept` (A3 / A4 / A6).
Synthetic frames only; the real files are read by the CLI.

Seen red first 2026-09-30 (collection error, module absent). Mutation rehearsal 2026-09-30 (scratch runner, -x, source
restored): 19 of 19 RED — footprint ignored, < for <=, first time block only, missing-line refusal off, float == for the
bit compare, dtype / row-count checks off, template ids by set not order, values >= 0, vocabulary check off, global
demean, lane filter off, scope filter off, raw RMS gated, missing-cell / base-order / empty-population / provenance-id
checks off, ok constant.
2026-10-01 (BC-36: current-run segment, guard pause, implausible footprint): 5 tests seen red first; rehearsal 10 of 10
RED — whole log read, first header instead of last, first MEMGUARD START instead of last, pause read from any run,
limit not divided, < for <=, pause branch off, implausibility check off, footprint in the paused peak (survived until the
12 GB-footprint assertion was added), ratio 1.5."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fmf_checks", ROOT / "research/analysis/fmf_checks.py")
C = importlib.util.module_from_spec(_spec)
sys.modules["fmf_checks"] = C
_spec.loader.exec_module(C)

TIME_L = """heavy lock ACQUIRED pid 1 -> running: /usr/bin/time -l python -m prc fit
      190.41 real      7681.06 user       410.44 sys
         {rss}  maximum resident set size
                   0  average shared memory size
             4396784  page reclaims
         {foot}  peak memory footprint
heavy lock RELEASED
"""


# ---------------------------------------------------------------- rss

def test_rss_takes_the_larger_of_resident_set_and_footprint():
    r = C.rss(TIME_L.format(rss=12_532_334_592, foot=14_346_120_856), limit=16_000_000_000)
    assert r == {"peak_bytes": 14_346_120_856, "max_rss_bytes": 12_532_334_592, "footprint_bytes": 14_346_120_856,
                 "footprint_used": True, "paused": False, "limit_bytes": 16_000_000_000, "ok": True}


def test_rss_fails_above_the_limit_on_the_footprint_alone():
    r = C.rss(TIME_L.format(rss=12_000_000_000, foot=16_000_000_001), limit=16_000_000_000)
    assert r["ok"] is False and r["peak_bytes"] == 16_000_000_001


def test_rss_at_exactly_the_limit_passes():
    assert C.rss(TIME_L.format(rss=16_000_000_000, foot=15_000_000_000), limit=16_000_000_000)["ok"] is True


def test_rss_uses_the_largest_of_repeated_time_blocks():
    """the 09-28 run log printed its time block twice; a second, larger block (a second command) must count."""
    text = TIME_L.format(rss=1_000, foot=2_000) + TIME_L.format(rss=17_000_000_000, foot=16_500_000_000)
    r = C.rss(text, limit=16_000_000_000)
    assert r["max_rss_bytes"] == 17_000_000_000 and r["ok"] is False


def test_rss_without_a_resident_set_line_is_refused():
    with pytest.raises(ValueError, match="no 'maximum resident set size' line"):
        C.rss("heavy lock ACQUIRED\nTraceback ...\n", limit=16_000_000_000)


def test_rss_without_a_footprint_line_still_gates_on_the_resident_set():
    text = "        17000000000  maximum resident set size\n"
    r = C.rss(text, limit=16_000_000_000)
    assert r["footprint_bytes"] is None and r["peak_bytes"] == 17_000_000_000 and r["ok"] is False


# ---------------------------------------------------------------- rss: appended logs and guard pauses (2026-10-01)

HEAD_1 = "[2026-10-01 09:26:38] lock /usr/bin/time -l python -m prc fit\n"
HEAD_2 = "[2026-10-01 16:26:41] lock /usr/bin/time -l python -m prc fit\n"
GUARD_QUIET = "[2026-10-01 16:26:41] MEMGUARD START pid=1 pause<350M\n[2026-10-01 16:31:42] MEMGUARD END target exited\n"
GUARD_PAUSED = ("[2026-10-01 16:26:41] MEMGUARD START pid=1 pause<350M\n"
                "[2026-10-01 16:26:41] MEMGUARD PAUSED reason=phantom:aggregate_cell_hour pids=[1]\n"
                "[2026-10-01 16:28:42] MEMGUARD RESUMED after 120s\n[2026-10-01 16:31:42] MEMGUARD END target exited\n")


def test_regression_bug_36_rss_reads_only_the_last_queue_run_of_an_appended_log():
    """BC-36 (2026-10-01): the queue appends to <step>.log, so run 2's log still held run 1's 17.8 GB block above its
    own 9.2 GB one and the check failed a run that was within the limit. Only the text after the last queue header
    ("[date time] lock|light|r12 ...") is the current run. Seen red: 17,803,278,592 reported for the second run."""
    text = (HEAD_1 + TIME_L.format(rss=11_652_874_240, foot=17_803_278_592)
            + HEAD_2 + TIME_L.format(rss=9_158_557_696, foot=9_949_127_536))
    r = C.rss(text, limit=16_000_000_000)
    assert r["max_rss_bytes"] == 9_158_557_696 and r["peak_bytes"] == 9_949_127_536 and r["ok"] is True


def test_rss_still_takes_the_largest_block_inside_the_current_run():
    text = HEAD_1 + TIME_L.format(rss=1_000, foot=2_000) + HEAD_2 + TIME_L.format(rss=5_000_000_000, foot=5_100_000_000) \
        + TIME_L.format(rss=17_000_000_000, foot=16_500_000_000)
    assert C.rss(text, limit=16_000_000_000)["max_rss_bytes"] == 17_000_000_000


def test_regression_bug_36_a_guard_pause_voids_the_footprint_and_the_resident_set_is_gated_at_limit_over_1_6():
    """`/usr/bin/time -l` snapshots real / instructions / peak footprint at the child's first stop: mem_guard's SIGSTOP
    truncates them (measured 2026-10-01: 433 MB reported for a 1.2 GB process paused at launch; run 2 reported 18.7 MB
    beside a 9.16 GB resident set). Max RSS comes from wait4 and stays right. With a PAUSED line in the guard log's
    last run the footprint is not used and RSS is gated at limit / 1.6 (worst measured footprint / RSS 1.53, run 1)."""
    text = HEAD_2 + TIME_L.format(rss=9_158_557_696, foot=18_710_912)
    r = C.rss(text, limit=16_000_000_000, guard_text=GUARD_PAUSED)
    assert r == {"peak_bytes": 9_158_557_696, "max_rss_bytes": 9_158_557_696, "footprint_bytes": 18_710_912,
                 "footprint_used": False, "paused": True, "limit_bytes": 10_000_000_000, "ok": True}
    assert C.rss(HEAD_2 + TIME_L.format(rss=10_000_000_000, foot=1), limit=16_000_000_000,
                 guard_text=GUARD_PAUSED)["ok"] is True
    assert C.rss(HEAD_2 + TIME_L.format(rss=10_000_000_001, foot=1), limit=16_000_000_000,
                 guard_text=GUARD_PAUSED)["ok"] is False
    r = C.rss(HEAD_2 + TIME_L.format(rss=9_000_000_000, foot=12_000_000_000), limit=16_000_000_000, guard_text=GUARD_PAUSED)
    assert r["peak_bytes"] == 9_000_000_000 and r["ok"] is True      # a truncated footprint is never the reported peak


def test_rss_a_pause_in_an_earlier_guard_run_does_not_void_this_runs_footprint():
    text = HEAD_2 + TIME_L.format(rss=9_000_000_000, foot=15_000_000_000)
    r = C.rss(text, limit=16_000_000_000, guard_text=GUARD_PAUSED + GUARD_QUIET)
    assert r["paused"] is False and r["footprint_used"] is True and r["peak_bytes"] == 15_000_000_000
    r = C.rss(text, limit=16_000_000_000, guard_text=GUARD_QUIET + GUARD_PAUSED)
    assert r["paused"] is True and r["footprint_used"] is False


def test_a_pause_before_the_guard_was_restarted_still_belongs_to_the_run():
    """The queue driver restarts a guard that died while the step runs (`mem_guard.py --restart`, logged RESTART): the
    run continues, so a pause logged before the restart still voids this run's footprint."""
    restarted = GUARD_PAUSED.replace("MEMGUARD END target exited", "MEMGUARD RESTART pid=1 pause<350M") + \
        "[2026-10-01 16:40:00] MEMGUARD END target exited\n"
    assert C.guard_paused(restarted) is True
    assert C.guard_paused(restarted + GUARD_QUIET) is False            # a new START is a new run


def test_rss_refuses_an_implausible_footprint_when_no_pause_explains_it():
    """A footprint under a quarter of the resident set cannot be a peak (every credible pair measured: 0.93 – 1.53);
    without a pause on record the block is refused, never silently gated on RSS alone."""
    text = HEAD_2 + TIME_L.format(rss=9_158_557_696, foot=18_710_912)
    with pytest.raises(ValueError, match="peak memory footprint 18,710,912 is not credible"):
        C.rss(text, limit=16_000_000_000)
    with pytest.raises(ValueError, match="not credible"):
        C.rss(text, limit=16_000_000_000, guard_text=GUARD_QUIET)


# ---------------------------------------------------------------- same

def _pred(n=5):
    return pd.DataFrame({"MVT_ID_mvt": np.arange(1, n + 1, dtype=float),
                         "TAXITIME_SEC_mvt": np.arange(600, 600 + n, dtype="int64"),
                         "decided_by": ["base"] * (n - 1) + ["unmatched_body"],
                         "F": [1.5, np.nan, 2.5, 3.5, np.nan][:n]})


def test_same_frames_with_nans_in_the_same_places_are_identical():
    assert C.same(_pred(), _pred()) == {"rows": 5, "columns": 4, "n_diff_cells": 0, "diff_columns": [], "ok": True}


def test_same_counts_a_last_bit_float_difference():
    b = _pred()
    b.loc[2, "F"] = np.nextafter(2.5, 3.0)
    r = C.same(_pred(), b)
    assert r["ok"] is False and r["n_diff_cells"] == 1 and r["diff_columns"] == ["F"]


def test_same_counts_a_nan_against_a_value():
    b = _pred()
    b.loc[1, "F"] = 0.0
    assert C.same(_pred(), b)["diff_columns"] == ["F"]


def test_same_counts_a_string_difference():
    b = _pred()
    b.loc[4, "decided_by"] = "rome"
    r = C.same(_pred(), b)
    assert r["diff_columns"] == ["decided_by"] and r["n_diff_cells"] == 1


def test_same_refuses_different_columns_dtypes_or_lengths():
    with pytest.raises(ValueError, match="columns differ"):
        C.same(_pred(), _pred().drop(columns="F"))
    b = _pred()
    b["TAXITIME_SEC_mvt"] = b.TAXITIME_SEC_mvt.astype("int32")
    with pytest.raises(ValueError, match="dtype of TAXITIME_SEC_mvt differs"):
        C.same(_pred(), b)
    with pytest.raises(ValueError, match="rows differ: 5 vs 4"):
        C.same(_pred(), _pred(4))


def test_same_detects_reordered_rows():
    b = _pred().iloc[::-1].reset_index(drop=True)
    r = C.same(_pred(), b)
    assert r["ok"] is False and "MVT_ID_mvt" in r["diff_columns"]


# ---------------------------------------------------------------- accept

VOCAB = frozenset({"base", "adsb_stack", "cbv_blend", "unmatched_body", "rome"})
SCOPE = (("EDDF", 1), ("EDDF", 7))


def _case(n_per=4, shift=0.0, noise=None):
    """3 cells x n_per rows: EDDF-1 and EDDF-7 in scope, LEMD-1 not; the last row of each cell is unmatched."""
    rows = []
    for ap, m in (("EDDF", 1), ("EDDF", 7), ("LEMD", 1)):
        for i in range(n_per):
            rows.append((ap, m, "unmatched" if i == n_per - 1 else "matched"))
    n = len(rows)
    ids = np.arange(101, 101 + n, dtype=float)
    template = pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": np.zeros(n, "int32")})
    base = pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": np.full(n, 600, "int32")})
    add = np.full(n, shift) if noise is None else shift + np.asarray(noise, float)
    sub = pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": (600 + add).round().astype("int32")})
    prov = pd.DataFrame({"MVT_ID_mvt": ids[::-1], "decided_by": ["base"] * n,
                         "lane": [r[2] for r in rows][::-1]})
    cells = pd.DataFrame({"MVT_ID_mvt": ids, "ADEP_mvt": [r[0] for r in rows], "month": [r[1] for r in rows]})
    return sub, prov, template, base, cells


def _accept(sub, prov, template, base, cells, limit=81.6):
    return C.accept(sub, prov, template, base, cells, scope=SCOPE, vocab=VOCAB, limit=limit)


def test_accept_passes_a_per_cell_constant_shift_of_any_size():
    """MTX is a per-cell constant: a −50 s cell shift is not composition drift (FMF.2)."""
    sub, prov, template, base, cells = _case(shift=-50.0)
    r = _accept(sub, prov, template, base, cells)
    assert r["ok"] is True and r["a6"]["rms_demeaned_s"] == 0.0 and r["a6"]["rms_raw_s"] == 50.0
    assert r["a6"]["cells"] == {"EDDF|1": {"n": 3, "mean_diff_s": -50.0}, "EDDF|7": {"n": 3, "mean_diff_s": -50.0}}


def test_accept_a6_counts_only_matched_rows_in_scope_cells():
    """a huge move on an unmatched row or on an out-of-scope cell does not enter A6."""
    noise = [0, 0, 0, 5000,   0, 0, 0, 5000,   9000, 9000, 9000, 9000]
    sub, prov, template, base, cells = _case(noise=noise)
    r = _accept(sub, prov, template, base, cells)
    assert r["a6"]["n_rows"] == 6 and r["a6"]["rms_demeaned_s"] == 0.0 and r["ok"] is True


def test_accept_a6_fails_on_row_level_drift_above_the_limit():
    noise = [-100, 100, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0]
    sub, prov, template, base, cells = _case(noise=noise)
    r = _accept(sub, prov, template, base, cells)
    exp = float(np.sqrt((100 ** 2 + 100 ** 2) / 6))
    assert r["a6"]["rms_demeaned_s"] == pytest.approx(exp, abs=1e-9)   # 57.7 s
    assert r["ok"] is True
    r = _accept(sub, prov, template, base, cells, limit=57.0)
    assert r["ok"] is False and r["a6"]["ok"] is False


def test_accept_a6_demeans_within_each_cell_not_globally():
    """cell means −50 and +50: demeaned globally the RMS would be 50, within cell it is 0."""
    noise = [-50, -50, -50, 0,   50, 50, 50, 0,   0, 0, 0, 0]
    sub, prov, template, base, cells = _case(noise=noise)
    r = _accept(sub, prov, template, base, cells, limit=1.0)
    assert r["a6"]["rms_demeaned_s"] == 0.0 and r["ok"] is True


def test_accept_a3_refuses_wrong_ids_order_or_values():
    sub, prov, template, base, cells = _case()
    bad = sub.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="A3: ids / order differ from the template"):
        _accept(bad, prov, template, base, cells)
    bad = sub.copy()
    bad.loc[0, "TAXITIME_SEC_mvt"] = 0
    with pytest.raises(ValueError, match="A3: 1 value"):
        _accept(bad, prov, template, base, cells)
    with pytest.raises(ValueError, match="A3: ids / order differ from the template"):
        _accept(sub.iloc[:-1], prov, template, base, cells)


def test_accept_a4_refuses_a_missing_or_unknown_decided_by():
    sub, prov, template, base, cells = _case()
    bad = prov.copy()
    bad.loc[3, "decided_by"] = None
    with pytest.raises(ValueError, match="A4: 1 row"):
        _accept(sub, bad, template, base, cells)
    bad = prov.copy()
    bad.loc[0, "decided_by"] = "hand_edit"
    with pytest.raises(ValueError, match="A4: 1 row"):
        _accept(sub, bad, template, base, cells)
    with pytest.raises(ValueError, match="A4: provenance ids differ"):
        _accept(sub, prov.iloc[1:], template, base, cells)


def test_accept_reports_the_decided_by_counts():
    sub, prov, template, base, cells = _case()
    prov.loc[0, "decided_by"] = "rome"
    assert _accept(sub, prov, template, base, cells)["a4"] == {"rows": 12, "decided_by": {"base": 11, "rome": 1}}


def test_accept_refuses_a_base_that_is_not_the_same_rows():
    sub, prov, template, base, cells = _case()
    with pytest.raises(ValueError, match="A6: base ids / order differ"):
        _accept(sub, prov, template, base.iloc[::-1].reset_index(drop=True), cells)


def test_accept_refuses_an_empty_a6_population():
    sub, prov, template, base, cells = _case()
    prov["lane"] = "unmatched"
    with pytest.raises(ValueError, match="A6: no matched rows in the scope cells"):
        _accept(sub, prov, template, base, cells)


def test_accept_refuses_submission_ids_without_a_cell():
    sub, prov, template, base, cells = _case()
    with pytest.raises(ValueError, match="A6: 1 submission id\\(s\\) without an airport / month cell"):
        _accept(sub, prov, template, base, cells.iloc[1:])


# ---------------------------------------------------------------- min_changed (BSV: the variant size rule)

def test_min_changed_gates_the_registered_ten_thousand_rows():
    """plans/PREREG_bsv_board_selected_variants_2026_10_01.md: a variant must change >= 10,000 scored rows against its
    base, or its board delta reads a few rows' labels. Mutation: > for >= -> red; a constant ok -> red."""
    assert C.min_changed({"n_changed": 10_000, "rows": 344_841}) == {"n_changed": 10_000, "minimum": 10_000, "ok": True}
    assert C.min_changed({"n_changed": 9_999, "rows": 344_841})["ok"] is False
    assert C.min_changed({"n_changed": 42, "rows": 344_841}, minimum=40)["ok"] is True
    with pytest.raises(ValueError, match="no n_changed"):
        C.min_changed({"rows": 344_841})


def test_r12_scope_is_the_fifteen_registered_airport_months():
    """A6's population is fixed in this module (no load from a build script): 7 airports x {1, 7} + EGLL January."""
    s = set(C.R12_SCOPE)
    assert len(C.R12_SCOPE) == len(s) == 15
    assert ("EGLL", 1) in s and ("EGLL", 7) not in s
    assert not any(a in ("LFPG", "LTFM") for a, _ in s)
    assert all(m in (1, 7) for _, m in s) and {a for a, _ in s} == {"EDDF", "EDDM", "EHAM", "LEBL", "LSZH", "LEMD", "LIRF", "EGLL"}
