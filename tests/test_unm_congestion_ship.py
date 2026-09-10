"""scripts/unm_congestion_ship.py - the shipping step for arm E3C (plans/PREREG_unm_congestion_2026_09_10.md
RESULT E3C, owner decision 2026-09-10): a base submission with ONLY the scored non-LIRF unmatched rows replaced
by rint(S1C) = rint(mixture(p, sp, nf_hybrid(nf_cells, mean over seeds of the congestion body))), where p, sp,
nf_cells and the seeds are the shipped fit_unmatched parts that rebuild the base.

Pins: the rebuild guard passes on the true base, refuses a perturbed non-LIRF row, ignores LIRF rows (they are
R2 / E1's in v7 / v8) and sits INSIDE build(); only non-LIRF unmatched rows change and to exactly rint(S1C)
recomputed independently; LIRF unmatched rows and matched rows are byte-identical; the dtype is preserved;
check_submission passes; the base is not mutated; the 2026 witness is the SERVE file's (build attaches it
itself, refuses pre-attached witness columns and a witness that covers no scored hour); the writer refuses to
overwrite and re-reads what it wrote. Every RNG is seeded; mutation rehearsals are in the docstrings.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import unm_congestion_ship as us  # noqa: E402
import unm_congestion as uc  # noqa: E402

bs = uc.bs
LIRF_OFFSET = 123           # the base's LIRF unmatched rows are NOT S1's (v7 = R2, v8 = E1 on top)
SERVE_SHIFT_S = 900.0       # the serve companion's prox is 900 s hotter than the 2025 world's


@pytest.fixture(scope="module")
def world():
    """Training frames (months 1-11) with the 2025 witness attached, a 'scored' set (the synthetic month-12 rows
    relabelled as serve rows, witness columns dropped), a serve frame (a matched companion of the scored rows
    whose prox is shifted so the 2026 witness differs from 2025's), S1's parts on the scored rows, and a base
    built from S1 on non-LIRF rows and from S1 + LIRF_OFFSET on LIRF rows."""
    unm, lirf, _ = uc.synthetic_world(seed=5, n_per_month=1000, n_matched_per_month=600)
    train_unm, train_lirf = unm[unm.month <= 11], lirf[lirf.month <= 11]
    scored_25w = unm[unm.month == 12].reset_index(drop=True)                # carries the 2025-world witness
    scored = scored_25w.drop(columns=list(uc.WITNESS))
    serve = uc.smoke_matched_companion(scored, seed=77)
    serve["AOBT_3_flt"] = serve.AOBT_3_flt - pd.Timedelta(seconds=SERVE_SHIFT_S)
    parts = {}
    s1 = bs.fit_unmatched(train_unm, scored, train_matched=train_lirf, hybrid=True, seeds=(0,), parts=parts)
    vals = np.rint(s1)
    vals[(scored.ADEP_mvt == "LIRF").to_numpy()] += LIRF_OFFSET
    matched_ids = np.arange(9_000_000, 9_000_050, dtype="float64")
    base = pd.DataFrame({"MVT_ID_mvt": np.concatenate([scored.MVT_ID_mvt.to_numpy(), matched_ids]),
                         "TAXITIME_SEC_mvt": np.concatenate([vals, np.full(50, 900.0)]).astype("int32")})
    return dict(train_unm=train_unm, train_lirf=train_lirf, scored=scored, scored_25w=scored_25w, serve=serve,
                parts=parts, base=base, matched_ids=matched_ids)


@pytest.fixture(scope="module")
def built(world):
    snap = world["base"].copy()
    out, info = us.build(world["train_unm"], world["scored"], world["parts"], world["base"], world["serve"])
    return out, info, snap


def _independent_s1c(w) -> np.ndarray:
    """rint(S1C) recomputed here from the shipped pieces: the serve witness attached to the scored rows, the body
    fitted on train[~schedule_fill(train)] at the parts' seeds, S1's p / sp / nf_cells."""
    sw = uc.attach_witness(w["scored"], uc.airport_hour_witness(w["serve"]))
    nonfill = w["train_unm"][~bs.schedule_fill(w["train_unm"])]
    body = bs.mean_over_seeds({s: uc.fit_congestion_regressor(nonfill, s).predict(sw) for s in w["parts"]["seeds"]})
    return np.rint(bs.mixture(w["parts"]["p"], w["parts"]["sp"], bs.nf_hybrid(w["parts"]["nf_cells"], body)))


# =============================================================================================
# guard 1
# =============================================================================================

def test_rebuild_guard_passes_on_the_true_base_refuses_a_perturbed_non_lirf_row_and_ignores_lirf(world):
    """Rehearsed 2026-09-10 RED: the guard's comparison replaced by a tolerance of 5 s (the perturbed base passes);
    the non-LIRF mask dropped (the true base, whose LIRF rows are S1 + 123, is refused)."""
    scored, parts, base = world["scored"], world["parts"], world["base"]
    non = us.non_lirf_mask(scored)
    got = us.verify_parts_rebuild_non_lirf(parts, scored, base)
    assert got == {"n_checked": int(non.sum()), "n_lirf_not_compared": int((~non).sum()), "seeds": [0]}
    assert 0 < got["n_lirf_not_compared"] < got["n_checked"]
    i = int(np.flatnonzero(non)[0])
    bad = base.copy()
    bad.loc[i, "TAXITIME_SEC_mvt"] += 1
    with pytest.raises(AssertionError, match="do not rebuild the base on 1 of"):
        us.verify_parts_rebuild_non_lirf(parts, scored, bad)
    j = int(np.flatnonzero(~non)[0])
    worse = base.copy()
    worse.loc[j, "TAXITIME_SEC_mvt"] += 500                                  # a LIRF row: not this arm's business
    us.verify_parts_rebuild_non_lirf(parts, scored, worse)
    with pytest.raises(AssertionError, match="missing from the base"):
        us.verify_parts_rebuild_non_lirf(parts, scored, base.iloc[1:].reset_index(drop=True))
    with pytest.raises(ValueError, match="not the hybrid path"):
        us.verify_parts_rebuild_non_lirf({**parts, "hybrid": False}, scored, base)


def test_build_itself_refuses_a_base_that_s1_does_not_rebuild(world):
    """The guard must sit INSIDE build(), not only in main(). Rehearsed 2026-09-10 RED: the verify call deleted
    from build() (the other tests all pass a correct base and SURVIVE that deletion; this one goes RED)."""
    w = world
    non = us.non_lirf_mask(w["scored"])
    bad = w["base"].copy()
    bad.loc[int(np.flatnonzero(non)[3]), "TAXITIME_SEC_mvt"] += 7
    with pytest.raises(AssertionError, match="do not rebuild the base"):
        us.build(w["train_unm"], w["scored"], w["parts"], bad, w["serve"])


# =============================================================================================
# guard 2: what changes, and to what
# =============================================================================================

def test_only_non_lirf_unmatched_rows_change_and_values_equal_rint_s1c_recomputed_independently(world, built):
    """Rehearsed 2026-09-10 RED: the body built from parts["nf_fit"] instead of the congestion body (the values no
    longer match the independent S1C and nothing changes); the witness overwritten with a constant 700 s before
    the body predicts (the values drift from the independent recompute)."""
    w, (out, info, _) = world, built
    non = us.non_lirf_mask(w["scored"])
    o = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    b = w["base"].set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    changed = set(o.index[(o != b.loc[o.index]).to_numpy()])
    non_ids = set(w["scored"].MVT_ID_mvt[non])
    assert changed <= non_ids and len(changed) > 0.5 * len(non_ids)      # the body moves on (most of) the routed rows
    want = _independent_s1c(w)
    np.testing.assert_array_equal(o.loc[w["scored"].MVT_ID_mvt[non]].to_numpy(), want[non].astype("int32"))
    assert len(out) == len(w["base"]) and list(out.columns) == list(w["base"].columns)
    assert info["n_rows_replaced"] == int(non.sum()) and info["n_values_differ"] == len(changed)
    assert info["seeds"] == [0] and info["n_train_nonfill"] == int((~bs.schedule_fill(w["train_unm"])).sum())


def test_lirf_unmatched_rows_and_matched_rows_are_byte_identical(world, built):
    """Rehearsed 2026-09-10 RED: the non-LIRF mask replaced by all-True in build() (LIRF rows are rewritten to S1C)."""
    w, (out, _, _) = world, built
    lirf_ids = w["scored"].MVT_ID_mvt[~us.non_lirf_mask(w["scored"])].to_numpy()
    o = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    b = w["base"].set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    assert len(lirf_ids) > 0
    np.testing.assert_array_equal(o.loc[lirf_ids].to_numpy(), b.loc[lirf_ids].to_numpy())
    np.testing.assert_array_equal(o.loc[w["matched_ids"]].to_numpy(), np.full(50, 900, dtype="int32"))
    np.testing.assert_array_equal(out.MVT_ID_mvt.to_numpy(), w["base"].MVT_ID_mvt.to_numpy())   # order kept too


def test_dtype_is_preserved(world, built):
    """Rehearsed 2026-09-10 RED: the spliced column cast to int64 after splice_unmatched."""
    out = built[0]
    assert out.TAXITIME_SEC_mvt.dtype == world["base"].TAXITIME_SEC_mvt.dtype == np.dtype("int32")
    assert out.MVT_ID_mvt.dtype == world["base"].MVT_ID_mvt.dtype


def test_the_output_passes_the_submission_checker(world, built):
    """Rehearsed 2026-09-10 RED: the first replaced value forced to 0 before the splice. With build's own
    finite-and-floor check intact that check fires first (the fixture errors); with it deleted as well,
    check_submission refuses the non-positive prediction here."""
    out = built[0]
    bs.check_submission(out, world["base"][["MVT_ID_mvt"]].assign(TAXITIME_SEC_mvt=0))


def test_build_does_not_mutate_the_base(world, built):
    """Rehearsed 2026-09-10 RED: build() writing the new values into base.TAXITIME_SEC_mvt in place."""
    _, _, snap = built
    pd.testing.assert_frame_equal(world["base"], snap)


def test_info_shift_metrics_and_detail_are_consistent(world, built):
    """Rehearsed 2026-09-10 RED: sse_shift divided by the number of replaced rows instead of len(base)."""
    w, (out, info, _) = world, built
    d = info["detail"]
    assert list(d.columns) == us.DETAIL_COLUMNS and len(d) == info["n_rows_replaced"]
    non = us.non_lirf_mask(w["scored"])
    np.testing.assert_array_equal(d.MVT_ID_mvt.to_numpy(), w["scored"].MVT_ID_mvt.to_numpy()[non])
    np.testing.assert_array_equal(d.p.to_numpy(), np.asarray(w["parts"]["p"])[non])
    np.testing.assert_array_equal(d.sp.to_numpy(), w["scored"].sp.to_numpy()[non])
    b = w["base"].set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.loc[d.MVT_ID_mvt.to_numpy()].to_numpy()
    o = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.loc[d.MVT_ID_mvt.to_numpy()].to_numpy()
    np.testing.assert_array_equal(d.before.to_numpy(), b)
    np.testing.assert_array_equal(d.after.to_numpy(), o)
    shift = o.astype("float64") - b.astype("float64")
    assert info["sse_shift_vs_base_board_mse"] == pytest.approx((shift ** 2).sum() / len(w["base"]), rel=1e-12)  # integers, fp64 sum
    assert info["rms_shift_vs_base_s"] == pytest.approx(np.sqrt((shift ** 2).mean()), rel=1e-12)
    assert info["mean_shift_s"] == pytest.approx(shift.mean(), rel=1e-12)
    per_ap = pd.Series(shift).groupby(d.ADEP_mvt.to_numpy()).mean()
    assert set(info["per_airport_mean_shift_s"]) == set(per_ap.index) and "LIRF" not in per_ap.index
    for k, v in per_ap.items():
        assert info["per_airport_mean_shift_s"][k] == pytest.approx(v, rel=1e-12)
    assert sum(info["bins_2026_changed_rows"].values()) == info["n_values_differ"]
    assert sum(info["bins_2026_replaced_rows"].values()) == info["n_rows_replaced"]
    np.testing.assert_array_equal(d.bin.to_numpy(), uc.bin_of(d.hprox_med))
    t = us.shift_tables(d)
    assert int(t["by_bin"].n.sum()) == len(d) and int(t["by_airport"].n_changed.sum()) == info["n_values_differ"]
    assert list(t["by_bin"].index) == [b_ for b_ in uc.BINS if b_ in set(d.bin)]
    lines = []
    us.render_tables(t, lines.append)
    assert sum(1 for line in lines if line.startswith("---")) == 3 and any("EHAM" in line for line in lines)


# =============================================================================================
# the 2026 witness comes from the serve file, never from 2025
# =============================================================================================

def test_the_witness_on_the_replaced_rows_is_the_serve_files_not_2025s(world, built):
    """Rehearsed 2026-09-10 RED: hprox_med overwritten with a constant 700 s after attaching (the detail no longer
    equals the serve witness lookup); the witness computed from the TRAINING frame instead of the serve frame
    (its rows made 'matched' with a fabricated prox) -- the coverage guard fires, and with that guard deleted
    as well the detail carries no serve hour and this test goes RED on the lookup equality."""
    w, (out, info, _) = world, built
    d = info["detail"]
    non = us.non_lirf_mask(w["scored"])
    serve_w = uc.attach_witness(w["scored"], uc.airport_hour_witness(w["serve"]))
    np.testing.assert_array_equal(d.hprox_med.to_numpy(), serve_w.hprox_med.to_numpy()[non])
    np.testing.assert_array_equal(d.hprox_n.to_numpy(), serve_w.hprox_n.to_numpy()[non])
    w25 = w["scored_25w"].hprox_med.to_numpy()[non]
    both = np.isfinite(w25) & np.isfinite(d.hprox_med.to_numpy())
    assert both.sum() > 100 and (d.hprox_med.to_numpy()[both] != w25[both]).mean() > 0.95
    assert d.hprox_med.dropna().median() > w["scored_25w"].hprox_med.dropna().median() + 0.5 * SERVE_SHIFT_S
    # S1C recomputed with the 2025 witness is NOT what shipped
    nonfill = w["train_unm"][~bs.schedule_fill(w["train_unm"])]
    body25 = uc.fit_congestion_regressor(nonfill, 0).predict(w["scored_25w"])
    s1c25 = np.rint(bs.mixture(w["parts"]["p"], w["parts"]["sp"], bs.nf_hybrid(w["parts"]["nf_cells"], body25)))
    shipped = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.loc[w["scored"].MVT_ID_mvt[non]].to_numpy()
    assert (shipped != s1c25[non].astype("int32")).mean() > 0.3
    assert info["witness_2026"]["n_scored_covered"] == int((serve_w.hprox_n.to_numpy() > 0).sum())
    assert info["witness_2026"]["n_serve_matched_rows"] == len(w["serve"])


def test_build_refuses_scored_rows_that_arrive_with_a_witness_and_a_serve_frame_that_covers_no_scored_hour(world):
    """Rehearsed 2026-09-10 RED: the pre-attached refusal deleted from attach_serve_witness (the 2025 witness
    is accepted); the coverage check deleted (the 2025 LIRF frame is accepted as the serve file)."""
    w = world
    with pytest.raises(ValueError, match="already carry .*hprox_med.*build attaches the serve file's witness"):
        us.build(w["train_unm"], w["scored_25w"], w["parts"], w["base"], w["serve"])
    with pytest.raises(ValueError, match="covers none of the scored airport-hours"):
        us.build(w["train_unm"], w["scored"], w["parts"], w["base"], w["train_lirf"])     # 2025 hours never match
    unmatched_serve = w["serve"].assign(AOBT_3_flt=pd.Series(pd.NaT, index=w["serve"].index, dtype="datetime64[ns, UTC]"))
    with pytest.raises(ValueError, match="no matched departure"):
        us.build(w["train_unm"], w["scored"], w["parts"], w["base"], unmatched_serve)
    with pytest.raises(ValueError, match="serve frame needs"):
        us.build(w["train_unm"], w["scored"], w["parts"], w["base"], w["serve"].drop(columns=["AOBT_3_flt"]))
    with pytest.raises(ValueError, match="lack the witness column"):
        us.build(w["train_unm"].drop(columns=["hprox_n"]), w["scored"], w["parts"], w["base"], w["serve"])


# =============================================================================================
# writing
# =============================================================================================

def test_write_outputs_round_trips_writes_detail_and_meta_and_refuses_to_overwrite(world, built, tmp_path):
    """Rehearsed 2026-09-10 RED: the `dest.exists()` refusal deleted (the second call overwrites); the frame
    written with its last row dropped (the re-read check_submission fails on the row count)."""
    w, (out, info, _) = world, built
    template = w["base"][["MVT_ID_mvt"]].assign(TAXITIME_SEC_mvt=0)
    dest, meta = us.write_outputs(out, info, template, 99, "merry-quicksand_v8.parquet", tmp_path, 0.0)
    assert dest == tmp_path / "merry-quicksand_v99.parquet" and dest.exists()
    back = pd.read_parquet(dest)
    pd.testing.assert_frame_equal(back, out)
    detail = pd.read_parquet(tmp_path / "merry-quicksand_v99.unm_rows.parquet")
    pd.testing.assert_frame_equal(detail, info["detail"])
    disk = json.loads((tmp_path / "merry-quicksand_v99.meta.json").read_text())
    assert disk == meta
    assert meta["version"] == 99 and meta["base"] == "merry-quicksand_v8.parquet" and meta["file"] == dest.name
    assert "RESULT E3C" in meta["arm"] and meta["prereg"] == uc.PREREG
    for k in ("n_rows_replaced", "n_values_differ", "per_airport_mean_shift_s", "rms_shift_vs_base_s",
              "bins_2026_changed_rows", "sse_shift_vs_base_board_mse", "rebuild_guard", "witness_2026", "seeds"):
        assert meta[k] == info[k if k != "rebuild_guard" else "rebuild"] or k == "witness_2026"
    assert meta["witness_2026"]["n_airport_hours"] == info["witness_2026"]["n_airport_hours"] and "ranking.parquet" in meta["witness_2026"]["source"]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        us.write_outputs(out, info, template, 99, "merry-quicksand_v8.parquet", tmp_path, 0.0)
    with pytest.raises(ValueError, match="row count"):
        us.write_outputs(out.iloc[:-1], info, template, 98, "merry-quicksand_v8.parquet", tmp_path, 0.0)
    assert not (tmp_path / "merry-quicksand_v98.parquet").exists()
