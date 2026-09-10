"""scripts/rome_ship.py - the shipping step for Rome arm R2 (plans/PREREG_rome_dateslip_2026_09_10.md,
owner decision 2026-09-10): a base submission with ONLY the LIRF unmatched rows replaced by
rint(apply_DS(mixture(p_R, sp, nf))), where nf and the rest of S1 come from the shipped
fit_unmatched parts.

Pins the three guards: S1's parts must rebuild the base exactly on every scored unmatched row or
nothing is written; only LIRF unmatched rows change; the output passes check_submission.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rome_ship as rs  # noqa: E402
import stratum_fold as sf  # noqa: E402


@pytest.fixture(scope="module")
def world():
    """Training frames (months 1-12) plus a 'scored' set: the synthetic month-12 rows relabelled as
    serve rows, and a base submission built from S1's own parts so the rebuild guard can pass."""
    unm, lirf = sf.synthetic_frame(n_per_month=1500, n_matched_per_month=2000, seed=5)
    train_unm, train_lirf = unm[unm.month <= 11], lirf[lirf.month <= 11]
    scored = unm[unm.month == 12].reset_index(drop=True)
    parts = {}
    s1 = sf.bs.fit_unmatched(train_unm, scored, train_matched=train_lirf, hybrid=True, seeds=(0,), parts=parts)
    matched_ids = np.arange(9_000_000, 9_000_050, dtype="float64")
    base = pd.DataFrame({"MVT_ID_mvt": np.concatenate([scored.MVT_ID_mvt.to_numpy(), matched_ids]),
                         "TAXITIME_SEC_mvt": np.concatenate([np.rint(s1), np.full(50, 900.0)]).astype("int32")})
    return train_unm, train_lirf, scored, parts, base


def test_rebuild_guard_passes_on_the_true_base_and_refuses_a_perturbed_one(world):
    """Rehearsed 2026-09-10 RED: the guard's comparison replaced by a tolerance of 5 s."""
    _, _, scored, parts, base = world
    rs.verify_parts_rebuild(parts, scored, base)
    bad = base.copy()
    bad.loc[0, "TAXITIME_SEC_mvt"] += 1
    with pytest.raises(AssertionError, match="do not rebuild the base"):
        rs.verify_parts_rebuild(parts, scored, bad)


def test_only_lirf_unmatched_rows_change_and_values_follow_the_r2_formula(world):
    """Rehearsed 2026-09-10 RED: the LIRF mask dropped (every unmatched row is rewritten)."""
    train_unm, train_lirf, scored, parts, base = world
    out, info = rs.build(train_unm, train_lirf, scored, parts, base, seeds=(0,))
    lirf_ids = set(scored.MVT_ID_mvt[scored.ADEP_mvt == "LIRF"])
    b = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    o = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    changed = set(o.index[(o != b.loc[o.index]).to_numpy()])
    assert changed <= lirf_ids and len(changed) > 0
    others = [i for i in o.index if i not in lirf_ids]
    assert (o.loc[others] == b.loc[others]).all()
    assert out.TAXITIME_SEC_mvt.dtype == base.TAXITIME_SEC_mvt.dtype and len(out) == len(base)
    assert info["n_rows_replaced"] == len(lirf_ids) and info["n_in_segment_g"] >= 0
    # the values are rint(apply(mixture(p, sp, nf))) on the LIRF rows
    m = (scored.ADEP_mvt == "LIRF").to_numpy()
    want = np.rint(rs.ds.apply(scored[m], rs.rf.mixture(info["p_r"], scored.sp.to_numpy()[m], parts["nf"][m]),
                               info["q"], info["m"]))
    np.testing.assert_array_equal(o.loc[scored.MVT_ID_mvt[m]].to_numpy(), want.astype("int32"))


def test_the_output_passes_the_submission_checker(world):
    train_unm, train_lirf, scored, parts, base = world
    out, _ = rs.build(train_unm, train_lirf, scored, parts, base, seeds=(0,))
    sf.bs.check_submission(out, base[["MVT_ID_mvt"]].assign(TAXITIME_SEC_mvt=0))


def test_build_itself_refuses_a_base_that_s1_does_not_rebuild(world):
    """The guard must sit INSIDE build(), not only in main(). Rehearsed 2026-09-10: deleting the
    verify call from build() SURVIVED the other tests (they all pass a correct base); this one
    goes RED on that deletion."""
    train_unm, train_lirf, scored, parts, base = world
    bad = base.copy()
    bad.loc[3, "TAXITIME_SEC_mvt"] += 7
    with pytest.raises(AssertionError, match="do not rebuild the base"):
        rs.build(train_unm, train_lirf, scored, parts, bad, seeds=(0,))
