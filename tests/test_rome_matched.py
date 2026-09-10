"""scripts/rome_matched.py - arm M of plans/PREREG_rome_matched_fill_2026_09_10.md: a stake-weighted
fill mixture on Rome's MATCHED holdout rows over arm F's record.

Pins: the fold-row -> MVT_ID mapping refuses a length mismatch; the NM-clock features never read
BLOCK or the target; the registered clauses, including the body bound that exists to catch
RESULT 8's double-counting; the permuted control; and that non-LIRF rows are untouched.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rome_fill as rf  # noqa: E402
import rome_matched as rm  # noqa: E402
import stratum_fold as sf  # noqa: E402


@pytest.fixture(scope="module")
def world():
    """Synthetic LIRF matched frame with planted airline fill propensity; a synthetic 'F' record on
    months 1 and 7 that HEDGES on every row (0.5 sp + 0.5 body) - the incumbent's shape."""
    unm, lirf = sf.synthetic_frame(n_per_month=1500, n_matched_per_month=2000, seed=7)
    te = lirf[lirf.month.isin([1, 7])].reset_index(drop=True)
    fill = rf.fill_label(te)
    body = te.y.to_numpy(dtype="float64") + np.random.default_rng(1).normal(0, 60, len(te))
    body = np.where(fill, 900.0, body)
    f_pred = np.where(fill, 0.5 * te.sp + 0.5 * body, body)
    rec = pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.to_numpy(), "month": te.month.to_numpy(), "ap": "LIRF",
                        "y": te.y.to_numpy(dtype="float64"), "sp": te.sp.to_numpy(dtype="float64"), "F": f_pred})
    return unm, lirf, rec


def test_mapping_refuses_a_length_mismatch(tmp_path):
    a, b = tmp_path / "s", tmp_path / "o"
    a.mkdir(); b.mkdir()
    pd.DataFrame({"y": [1.0, 2.0]}).to_parquet(a / "training_2025-01.parquet")
    pd.DataFrame({"MVT_ID_mvt": [7.0]}).to_parquet(b / "training_2025-01.parquet")
    with pytest.raises(AssertionError, match="row count"):
        rm.row_to_mvt_id(a, b)


def test_nm_features_never_read_block_or_the_target(world):
    _, lirf, _ = world
    a = rm.nm_frame(lirf)
    scr = lirf.copy()
    scr["BLOCK_TIME_UTC_mvt"] = pd.NaT
    scr["y"] = -1.0
    pd.testing.assert_frame_equal(a, rm.nm_frame(scr))
    assert list(a.columns) == rm.NM_FEATS


def test_m_beats_a_hedging_f_and_the_permuted_control_does_not(world):
    """F hedges on fill rows; a real classifier sharpens them. Rehearsed 2026-09-10 RED: the
    mixture written as p*F + (1-p)*sp (branches swapped)."""
    unm, lirf, rec = world
    res = rm.run(unm, lirf, rec, seeds=(0, 1), n_boot=300)
    assert res["rmse"]["M"] < res["rmse"]["F"]
    assert res["clauses"]["M"]["c1_interval"] is True
    assert res["control_perm_passes_c1"] is False
    assert set(res["clauses"]["M"]) == {"c1_interval", "c2_gain_ge_5s", "c3_both_months", "c4_body_bounded",
                                        "c5_gain_over_2x_seed_sd"}


def test_the_body_clause_catches_double_counting(world):
    """If F were already right on every row, blending sp in can only hurt: M must not win and the
    verdict must not be ESTABLISHED. This does NOT pin clause 4 (a c4-on-fill-rows mutation passed
    it); test_c4_reads_body_rows_... below does."""
    unm, lirf, rec = world
    exact = rec.assign(F=rec.y)
    res = rm.run(unm, lirf, exact, seeds=(0,), n_boot=100)
    assert res["rmse"]["M"] >= res["rmse"]["F"]
    assert res["verdict"]["M"] != "ESTABLISHED"


def test_c4_reads_body_rows_and_fails_when_the_blend_damages_them(world):
    """F hedges on fill rows but is EXACT on body rows: M wins on fills, and any non-zero p on a
    body row is pure damage there - clause 4 (body bounded) must be False even though M's fill
    rows improve. Rehearsed 2026-09-10: c4 computed on FILL rows SURVIVED the test above; this
    one goes RED on that mutation."""
    unm, lirf, rec = world
    fill = (rec.y - rec.sp).abs() <= rf.FILL_TOL_S
    body_exact = rec.assign(F=np.where(fill, rec.F, rec.y))
    res = rm.run(unm, lirf, body_exact, seeds=(0, 1), n_boot=100)
    d = res["detail"]["M"]
    assert d["fill_rows_rmse"]["M"] < d["fill_rows_rmse"]["F"]
    assert d["body_loss_s"] > rm.BODY_TOL_S
    assert res["clauses"]["M"]["c4_body_bounded"] is False
