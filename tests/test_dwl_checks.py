"""scripts/analysis/dwl_checks.py — the DWL prereg's label-free refit acceptance (plans/PREREG_dwl_lgb_half_on_dwg_2026_09_29.md
R1 / R2) and the v47-vs-v46 change report. Synthetic frames only; the real files are read by the CLI.

Seen red first 2026-09-29 (collection error, module absent). Mutation rehearsal 2026-09-29 (scratch runner, -x, source
restored): 17 of 17 RED — cap > instead of >=, split / n_rank / without-table / row-count / duplicate / finite checks
off, first frame line instead of last, log count ignored, no id alignment in R2, ok at < instead of <=, id-set and
proxy checks off, mean |d| for RMS, order check reduced to length, n_changed = all rows, cells read by position
(the last four killed only after the tests added here: last-line, RMS-vs-MAE, exact limit, shuffled cell index).
2026-09-29 MOVE extensions (arm, optional log, serve coverage): seen red (4 failed), then 6 of 6 new mutants RED (arm
ignored in R1 / R2, None log still parsed, log_checked constant, coverage / f_source checks off); the 17 above re-run
RED (finite-check mutant re-targeted to the arm-named message)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dwl_checks", ROOT / "research/analysis/dwl_checks.py")
D = importlib.util.module_from_spec(_spec)
sys.modules["dwl_checks"] = D
_spec.loader.exec_module(D)

N = 339_551


def _preds(n=6, shift=0.0, ids=None):
    ids = np.arange(1, n + 1, dtype=float) if ids is None else np.asarray(ids, float)
    return pd.DataFrame({"MVT_ID_mvt": ids, "pred_r12": np.linspace(-50, 50, len(ids)) + shift,
                         "proxy": np.full(len(ids), 900.0)})


def _report(best=39_422, n_rank=6, n_wo=0, split="refit"):
    return {"split": split, "smoke": False, "fits": {"r12": {"best_iter": best}}, "n_rank": n_rank,
            "n_test_without_table": n_wo}


LOG_OK = "[19:36:46] R12 frame: ... + 339,551 ranking rows; design (2401991, 127); rows without an ADS-B table row 0; peak\n"


def test_r1_passes_a_clean_refit():
    r = D.r1(_report(), _preds(), LOG_OK, n_rank_expected=6, cap=50_000)
    assert r == {"arm": "r12", "best_iter": 39_422, "cap": 50_000, "n_rank": 6, "rows_without_table": 0,
                 "log_checked": True, "n_finite": 6, "ok": True}


@pytest.mark.parametrize("kw,preds,log,msg", [
    (dict(best=50_000), None, LOG_OK, "best_iter 50000 reached the cap 50000"),
    (dict(n_rank=5), None, LOG_OK, "report n_rank 5, expected 6"),
    (dict(n_wo=1), None, LOG_OK, "report n_test_without_table 1, expected 0"),
    (dict(split="holdout"), None, LOG_OK, "report split 'holdout', expected 'refit'"),
    ({}, "nan", LOG_OK, "1 non-finite pred_r12"),
    ({}, "dup", LOG_OK, "predictions carry 1 duplicate MVT_ID_mvt"),
    ({}, "short", LOG_OK, "predictions have 5 rows, expected 6"),
    ({}, None, LOG_OK.replace("table row 0", "table row 12"), "log: 12 rows without an ADS-B table row"),
    ({}, None, "no frame line here\n", "log has no 'rows without an ADS-B table row' line"),
])
def test_r1_refuses_each_defect(kw, preds, log, msg):
    p = _preds()
    if preds == "nan":
        p.loc[2, "pred_r12"] = np.nan
    elif preds == "dup":
        p.loc[3, "MVT_ID_mvt"] = p.loc[2, "MVT_ID_mvt"]
    elif preds == "short":
        p = p.iloc[:5]
    with pytest.raises(ValueError) as e:
        D.r1(_report(**kw), p, log, n_rank_expected=6, cap=50_000)
    assert str(e.value) == msg


def test_r1_reads_the_last_frame_line_of_the_log():
    two = LOG_OK.replace("table row 0", "table row 5") + LOG_OK
    assert D.r1(_report(), _preds(), two, n_rank_expected=6, cap=50_000)["rows_without_table"] == 0
    with pytest.raises(ValueError, match="log: 5 rows without an ADS-B table row"):
        D.r1(_report(), _preds(), LOG_OK + LOG_OK.replace("table row 0", "table row 5"), n_rank_expected=6, cap=50_000)


def test_r2_is_a_root_mean_square_not_a_mean_absolute_difference():
    old = _preds(n=4)
    new = old.copy()
    new["pred_r12"] = old.pred_r12 + np.array([3.0, -4.0, 0.0, 0.0])
    assert D.r2(new, old, limit=81.6)["rms_s"] == pytest.approx(np.sqrt(25 / 4))


def test_r2_passes_at_exactly_the_limit():
    assert D.r2(_preds(shift=81.6), _preds(), limit=81.6)["ok"] is True


def test_r2_is_the_rms_of_the_delta_difference_aligned_by_id():
    """Rows are matched by MVT_ID_mvt, not position: the new file shuffled must give the same number."""
    old = _preds()
    new = _preds(shift=3.0).sample(frac=1.0, random_state=0)
    r = D.r2(new, old, limit=81.6)
    assert r["rms_s"] == pytest.approx(3.0, abs=1e-12) and r["ok"] is True and r["n"] == 6
    assert r["limit_s"] == 81.6


def test_r2_flags_above_the_limit_and_refuses_mismatched_ids():
    r = D.r2(_preds(shift=90.0), _preds(), limit=81.6)
    assert r["ok"] is False and r["rms_s"] == pytest.approx(90.0)
    with pytest.raises(ValueError, match="id sets differ: 1 only in new, 1 only in old"):
        D.r2(_preds(ids=[1, 2, 3, 4, 5, 7]), _preds(), limit=81.6)
    with pytest.raises(ValueError, match="proxy differs on 1 rows"):
        bad = _preds()
        bad.loc[0, "proxy"] = 901.0
        D.r2(bad, _preds(), limit=81.6)


def test_change_report_counts_moves_and_splits_them_by_cell():
    base = pd.DataFrame({"ID": [1.0, 2.0, 3.0, 4.0], "TAXITIME": [100, 200, 300, 400]})
    new = pd.DataFrame({"ID": [1.0, 2.0, 3.0, 4.0], "TAXITIME": [100, 210, 290, 400]})
    cell = pd.Series(["LIRF|7", "LIRF|7", "EDDF|1", "EDDF|1"], index=[3.0, 4.0, 1.0, 2.0])   # not in base order
    r = D.change_report(new, base, cell, id_col="ID", val_col="TAXITIME")
    assert r["rows"] == 4 and r["n_changed"] == 2
    assert r["rms_move_s"] == pytest.approx(np.sqrt((100 + 100) / 4))
    assert r["cells"] == {"EDDF|1": {"n": 2, "n_changed": 1, "mean_shift_s": 5.0},
                          "LIRF|7": {"n": 2, "n_changed": 1, "mean_shift_s": -5.0}}


def test_change_report_refuses_misaligned_files():
    base = pd.DataFrame({"ID": [1.0, 2.0], "TAXITIME": [1, 2]})
    new = pd.DataFrame({"ID": [2.0, 1.0], "TAXITIME": [2, 1]})
    with pytest.raises(ValueError, match="row order / ids differ from the base"):
        D.change_report(new, base, pd.Series(["a", "a"], index=[1.0, 2.0]), id_col="ID", val_col="TAXITIME")


# ------------------------------------------------------------------ MOVE (plans/PREREG_move_v47_2026_09_28.md M1 / M2 / M3)

def _arm_preds(arm, n=6, shift=0.0):
    p = _preds(n=n, shift=shift)
    return p.rename(columns={"pred_r12": f"pred_{arm}"})


def test_r1_reads_the_named_arm_and_its_own_cap():
    """MOVE's M1: arm r12_move, best_iter < 50,000 - 200 (cap 49,800 as registered)."""
    rep = {"split": "refit", "fits": {"r12_move": {"best_iter": 49_799}}, "n_rank": 6, "n_test_without_table": 0}
    r = D.r1(rep, _arm_preds("r12_move"), LOG_OK, n_rank_expected=6, cap=49_800, arm="r12_move")
    assert r["best_iter"] == 49_799 and r["arm"] == "r12_move"
    rep["fits"]["r12_move"]["best_iter"] = 49_800
    with pytest.raises(ValueError, match="best_iter 49800 reached the cap 49800"):
        D.r1(rep, _arm_preds("r12_move"), LOG_OK, n_rank_expected=6, cap=49_800, arm="r12_move")


def test_r1_without_a_log_skips_only_the_log_check_and_says_so():
    r = D.r1(_report(), _preds(), None, n_rank_expected=6, cap=50_000)
    assert r["rows_without_table"] is None and r["log_checked"] is False and r["ok"] is True
    assert D.r1(_report(), _preds(), LOG_OK, n_rank_expected=6, cap=50_000)["log_checked"] is True


def test_r2_reads_the_named_arm():
    r = D.r2(_arm_preds("r12_move", shift=2.0), _arm_preds("r12_move"), limit=81.6, arm="r12_move")
    assert r["rms_s"] == pytest.approx(2.0) and r["arm"] == "r12_move"


def test_serve_coverage_needs_every_matched_rank_row():
    """M3: the stack serve covers every matched MVT id (v46's serve: n_matched 339,551)."""
    assert D.serve_coverage({"n_matched": 339_551, "f_source": "refit"}, n_expected=339_551) == {
        "n_matched": 339_551, "f_source": "refit", "ok": True}
    with pytest.raises(ValueError, match="serve n_matched 339550, expected 339551"):
        D.serve_coverage({"n_matched": 339_550, "f_source": "refit"}, n_expected=339_551)
    with pytest.raises(ValueError, match="serve f_source 'holdout', expected 'refit'"):
        D.serve_coverage({"n_matched": 339_551, "f_source": "holdout"}, n_expected=339_551)
