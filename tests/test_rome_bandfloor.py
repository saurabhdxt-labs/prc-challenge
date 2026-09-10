"""scripts/rome_bandfloor.py - arm E1 (plans/PREREG_rome_bandfloor_2026_09_10.md): a base submission
with LIRF unmatched rows in [24,000, 86,400) raised to rint(sp) where below it, nothing else touched.

Pins: the mask's edges and scope; the values (max, never down); every other row byte-identical;
the base-identity guard; the output passes check_submission.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rome_bandfloor as bf  # noqa: E402

N = 400


@pytest.fixture()
def world():
    """A scored frame with rows placed on every edge the rule has, and a base submission."""
    rng = np.random.default_rng(3)
    ap = np.where(rng.random(N) < 0.5, "LIRF", "EHAM")
    unm = rng.random(N) < 0.4
    sp = rng.uniform(0, 120_000, N).round()
    # edge rows, all LIRF unmatched, base below sp so a wrong edge would move them
    edges = [23_999.0, 24_000.0, 86_399.0, 86_400.0]
    ap[:4], unm[:4], sp[:4] = "LIRF", True, edges
    # a LIRF MATCHED row and an EHAM unmatched row inside the band, base below sp
    ap[4], unm[4], sp[4] = "LIRF", False, 39_365.0
    ap[5], unm[5], sp[5] = "EHAM", True, 39_365.0
    # a LIRF unmatched in-band row whose base is ABOVE sp (a date-slip hedge): must stay
    ap[6], unm[6], sp[6] = "LIRF", True, 48_841.0
    ids = np.arange(1_000_000, 1_000_000 + N, dtype="float64")
    scored = pd.DataFrame({"MVT_ID_mvt": ids, "ADEP_mvt": ap, "unmatched": unm, "sp": sp})
    vals = rng.integers(200, 3_000, N).astype("int32")
    vals[6] = 62_215
    base = pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": vals})
    return scored, base


def test_rule_mask_edges_and_scope(world):
    """Rehearsed 2026-09-10 RED: `sp >= LO` -> `sp > LO` (the 24,000 row drops out) and `sp < HI` ->
    `sp <= HI` (the 86,400 row comes in); dropping the `unmatched` term admits row 4."""
    scored, _ = world
    m = bf.rule_mask(scored.sp, scored.ADEP_mvt, scored.unmatched)
    assert m[:7].tolist() == [False, True, True, False, False, False, True]
    want = (scored.ADEP_mvt == "LIRF") & scored.unmatched & (scored.sp >= 24_000) & (scored.sp < 86_400)
    np.testing.assert_array_equal(m, want.to_numpy())


def test_build_raises_rule_rows_to_rint_sp_and_copies_everything_else(world):
    """Rehearsed 2026-09-10 RED: `np.maximum(before, floor)` -> `floor` (row 6 moves DOWN to sp;
    the build's own down-move guard fires, and this test's value check would too)."""
    scored, base = world
    out, info = bf.build(base, scored)
    m = bf.rule_mask(scored.sp, scored.ADEP_mvt, scored.unmatched)
    b, o = base.TAXITIME_SEC_mvt.to_numpy(), out.TAXITIME_SEC_mvt.to_numpy()
    np.testing.assert_array_equal(o[~m], b[~m])
    np.testing.assert_array_equal(o[m], np.maximum(b[m], np.rint(scored.sp.to_numpy()[m]).astype("int64")))
    assert o[6] == 62_215 and o[1] == 24_000 and o[2] == 86_399 and o[0] == b[0] and o[3] == b[3]
    assert o[4] == b[4] and o[5] == b[5]
    assert out.TAXITIME_SEC_mvt.dtype == base.TAXITIME_SEC_mvt.dtype and list(out.columns) == list(base.columns)
    moved = o != b
    assert info["n_values_differ"] == int(moved.sum()) and info["n_rule_rows"] == int(m.sum())
    np.testing.assert_array_equal(np.sort(info["ids_changed"]), np.sort(scored.MVT_ID_mvt.to_numpy()[moved]))
    want_gain = float(((b[moved].astype("int64") - np.rint(scored.sp.to_numpy()[moved])) ** 2).sum() / N)
    assert info["fill_gain_board_mse"] == pytest.approx(want_gain, rel=1e-12)  # same integers, fp64 sum


def test_build_does_not_mutate_the_base(world):
    scored, base = world
    snap = base.copy()
    bf.build(base, scored)
    pd.testing.assert_frame_equal(base, snap)


def test_build_refuses_rule_rows_missing_from_the_base(world):
    scored, base = world
    with pytest.raises(AssertionError, match="missing from the base"):
        bf.build(base[base.MVT_ID_mvt != scored.MVT_ID_mvt[1]].reset_index(drop=True), scored)


def _detail(scored, base):
    lu = scored[(scored.ADEP_mvt == "LIRF") & scored.unmatched]
    after = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(lu.MVT_ID_mvt).to_numpy()
    return lu, pd.DataFrame({"MVT_ID_mvt": lu.MVT_ID_mvt.to_numpy(), "sp": lu.sp.to_numpy(), "after": after})


def test_verify_base_passes_on_the_described_base_and_refuses_any_other(world):
    """Rehearsed 2026-09-10 RED: the value comparison deleted from verify_base (the perturbed base
    passes); the sp comparison deleted (the shifted-sp detail passes)."""
    scored, base = world
    lu, det = _detail(scored, base)
    bf.verify_base(base, det, lu)
    bad = base.copy()
    bad.loc[bad.MVT_ID_mvt == lu.MVT_ID_mvt.iloc[0], "TAXITIME_SEC_mvt"] += 1
    with pytest.raises(AssertionError, match="not the version it claims to be"):
        bf.verify_base(bad, det, lu)
    d2 = det.copy()
    d2.loc[0, "sp"] += 1
    with pytest.raises(AssertionError, match="sp differs"):
        bf.verify_base(base, d2, lu)
    with pytest.raises(AssertionError, match="different row sets"):
        bf.verify_base(base, det.iloc[1:], lu)


def test_the_output_passes_the_submission_checker(world):
    import stratum_fold as sf
    scored, base = world
    out, _ = bf.build(base, scored)
    sf.bs.check_submission(out, base[["MVT_ID_mvt"]].assign(TAXITIME_SEC_mvt=0))
