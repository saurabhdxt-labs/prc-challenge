"""scripts/unm_congestion_resid_ship.py - the shipping step for arm E3R (plans/PREREG_unm_congestion_resid_2026_09_10.md,
only if RESULT E3R is WORKING): a base submission with ONLY the scored non-LIRF unmatched rows replaced by rint(S1R),
behind a guard that the S1C path rebuilds the base on every one of those rows.

Pins: guard 1 (the S1C path with the serve witness rebuilds the base) passes on the true base, refuses a perturbed
non-LIRF row, ignores LIRF rows and sits INSIDE build(); only non-LIRF unmatched rows change and to exactly rint(S1R)
recomputed independently (the offset body fitted here on the non-fill rows WITH a witness, S1C's per-seed body as the
fallback); scored rows WITHOUT a witness keep the base value; LIRF unmatched rows and matched rows are byte-identical;
the dtype is preserved; check_submission passes; the base is not mutated; the 2026 witness is the SERVE file's; the
writer refuses to overwrite and re-reads what it wrote. Every RNG is seeded; mutation rehearsals are in the docstrings.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import unm_congestion_resid_ship as rs  # noqa: E402
import unm_congestion_resid as ur  # noqa: E402
import unm_congestion_ship as us  # noqa: E402
import unm_congestion as uc  # noqa: E402

bs = uc.bs
LIRF_OFFSET = 123           # the base's LIRF unmatched rows are NOT S1C's (v7 = R2, v8 = E1, v10 = the band floor on top)
SERVE_SHIFT_S = 900.0       # the serve companion's prox is 900 s hotter than the 2025 world's


@pytest.fixture(scope="module")
def world():
    """Training frames (months 1-11) with the 2025 witness attached, a 'scored' set (the synthetic month-12 rows relabelled
    as serve rows, witness columns dropped), a serve frame (a matched companion of the scored rows whose prox is shifted),
    S1's parts on the scored rows, and a base built from rint(S1C with the serve witness) on non-LIRF rows -- v10's
    non-LIRF unmatched rows are v9's S1C -- and S1C + LIRF_OFFSET on LIRF rows."""
    unm, lirf, _ = uc.synthetic_world(seed=5, n_per_month=1000, n_matched_per_month=600)
    train_unm, train_lirf = unm[unm.month <= 11], lirf[lirf.month <= 11]
    scored_25w = unm[unm.month == 12].reset_index(drop=True)                # carries the 2025-world witness
    scored = scored_25w.drop(columns=list(uc.WITNESS))
    serve = uc.smoke_matched_companion(scored, seed=77)
    serve["AOBT_3_flt"] = serve.AOBT_3_flt - pd.Timedelta(seconds=SERVE_SHIFT_S)
    parts = {}
    bs.fit_unmatched(train_unm, scored, train_matched=train_lirf, hybrid=True, seeds=(0,), parts=parts)
    scored_w = uc.attach_witness(scored, uc.airport_hour_witness(serve))
    s1c, _, _, _ = us.s1c_from_parts(train_unm, scored_w, parts)
    vals = np.rint(s1c)
    vals[(scored.ADEP_mvt == "LIRF").to_numpy()] += LIRF_OFFSET
    matched_ids = np.arange(9_000_000, 9_000_050, dtype="float64")
    base = pd.DataFrame({"MVT_ID_mvt": np.concatenate([scored.MVT_ID_mvt.to_numpy(), matched_ids]),
                         "TAXITIME_SEC_mvt": np.concatenate([vals, np.full(50, 900.0)]).astype("int32")})
    return dict(train_unm=train_unm, train_lirf=train_lirf, scored=scored, scored_25w=scored_25w, scored_w=scored_w, serve=serve,
                parts=parts, base=base, matched_ids=matched_ids, s1c=s1c)


@pytest.fixture(scope="module")
def built(world):
    snap = world["base"].copy()
    out, info = rs.build(world["train_unm"], world["scored"], world["parts"], world["base"], world["serve"])
    return out, info, snap


def _independent_s1r(w) -> np.ndarray:
    """rint(S1R) recomputed here from the pieces: the serve witness attached to the scored rows; S1C's body fitted on
    train[~schedule_fill(train)]; the offset body fitted on those rows WITH a witness on clip(y - h, +-3000); per seed
    nf = h + r_hat where the scored row has a witness, S1C's body otherwise; S1's p / sp / nf_cells."""
    sw = uc.attach_witness(w["scored"], uc.airport_hour_witness(w["serve"]))
    h = sw.hprox_med.to_numpy(dtype="float64")
    nonfill = w["train_unm"][~bs.schedule_fill(w["train_unm"])]
    fitted = nonfill[nonfill.hprox_med.notna()]
    per_seed = {}
    for s in w["parts"]["seeds"]:
        body_c = uc.fit_congestion_regressor(nonfill, s).predict(sw)
        off = ur.ResidualRegressor(fitted, s).predict_offset(sw)
        per_seed[s] = np.where(np.isnan(h), body_c, h + off)
    body = bs.mean_over_seeds(per_seed)
    return np.rint(bs.mixture(w["parts"]["p"], w["parts"]["sp"], bs.nf_hybrid(w["parts"]["nf_cells"], body)))


# =============================================================================================
# guard 1
# =============================================================================================

def test_s1c_rebuild_guard_passes_on_the_true_base_refuses_a_perturbed_non_lirf_row_and_ignores_lirf(world):
    """Rehearsed 2026-09-10 RED: the guard's comparison replaced by a tolerance of 5 s (the perturbed base passes); the
    non-LIRF mask dropped (the true base, whose LIRF rows are S1C + 123, is refused)."""
    scored, base, s1c = world["scored"], world["base"], world["s1c"]
    non = rs.non_lirf_mask(scored)
    got = rs.verify_s1c_rebuilds_non_lirf(s1c, scored, base)
    assert got == {"n_checked": int(non.sum()), "n_lirf_not_compared": int((~non).sum()), "path": "S1C (unm_congestion_ship.s1c_from_parts)"}
    assert 0 < got["n_lirf_not_compared"] < got["n_checked"]
    i = int(np.flatnonzero(non)[0])
    bad = base.copy()
    bad.loc[i, "TAXITIME_SEC_mvt"] += 1
    with pytest.raises(AssertionError, match="S1C path does not rebuild the base on 1 of"):
        rs.verify_s1c_rebuilds_non_lirf(s1c, scored, bad)
    j = int(np.flatnonzero(~non)[0])
    worse = base.copy()
    worse.loc[j, "TAXITIME_SEC_mvt"] += 500                                  # a LIRF row: not this arm's business
    rs.verify_s1c_rebuilds_non_lirf(s1c, scored, worse)
    with pytest.raises(AssertionError, match="missing from the base"):
        rs.verify_s1c_rebuilds_non_lirf(s1c, scored, base.iloc[1:].reset_index(drop=True))
    with pytest.raises(ValueError, match="S1C has .* values for .* scored rows"):
        rs.verify_s1c_rebuilds_non_lirf(s1c[1:], scored, base)
    # S1 (the v7 path) does NOT rebuild this base: the guard is the S1C path, not S1's
    with pytest.raises(AssertionError, match="do not rebuild the base"):
        us.verify_parts_rebuild_non_lirf(world["parts"], scored, base)


def test_build_itself_refuses_a_base_that_the_s1c_path_does_not_rebuild_and_the_cell_path(world):
    """The guard must sit INSIDE build(). Rehearsed 2026-09-10 RED: the verify call deleted from build() (the other tests
    pass a correct base and SURVIVE that deletion; this one goes RED)."""
    w = world
    non = rs.non_lirf_mask(w["scored"])
    bad = w["base"].copy()
    bad.loc[int(np.flatnonzero(non)[3]), "TAXITIME_SEC_mvt"] += 7
    with pytest.raises(AssertionError, match="S1C path does not rebuild the base"):
        rs.build(w["train_unm"], w["scored"], w["parts"], bad, w["serve"])
    with pytest.raises(ValueError, match="not the hybrid path"):
        rs.build(w["train_unm"], w["scored"], {**w["parts"], "hybrid": False}, w["base"], w["serve"])


# =============================================================================================
# guard 2: what changes, and to what
# =============================================================================================

def test_only_non_lirf_unmatched_rows_change_and_values_equal_rint_s1r_recomputed_independently(world, built):
    """Rehearsed 2026-09-10 RED: the body built from S1C's nf_fit_c instead of the residual body (nothing changes and the
    values no longer match the independent S1R); the fallback taken from S1's nf_fit_by_seed (NaN-witness rows move and
    build's own no-witness guard fires)."""
    w, (out, info, _) = world, built
    non = rs.non_lirf_mask(w["scored"])
    o = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    b = w["base"].set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    changed = set(o.index[(o != b.loc[o.index]).to_numpy()])
    non_ids = set(w["scored"].MVT_ID_mvt[non])
    assert changed <= non_ids and len(changed) > 0.5 * len(non_ids)      # the offset body moves (most of) the routed witness rows
    want = _independent_s1r(w)
    np.testing.assert_array_equal(o.loc[w["scored"].MVT_ID_mvt[non]].to_numpy(), want[non].astype("int32"))
    assert len(out) == len(w["base"]) and list(out.columns) == list(w["base"].columns)
    assert info["n_rows_replaced"] == int(non.sum()) and info["n_values_differ"] == len(changed)
    nonfill = w["train_unm"][~bs.schedule_fill(w["train_unm"])]
    assert info["seeds"] == [0] and info["n_train_nonfill"] == len(nonfill) and info["n_train_witness"] == int(nonfill.hprox_med.notna().sum())
    assert 0 < info["n_train_witness"] < info["n_train_nonfill"] and info["n_clipped_high"] > 0 and info["n_clipped_low"] >= 0


def test_scored_rows_without_a_witness_keep_the_base_value(world, built):
    """Rehearsed 2026-09-10 RED: residual_nf_fit's fallback replaced by the parent's nf_fit (S1's) -- the NaN rows move."""
    w, (out, info, _) = world, built
    d = info["detail"]
    nan = np.isnan(d.hprox_med.to_numpy())
    assert 5 < nan.sum() < len(d) and info["n_nan_witness_rows"] == int(nan.sum())
    np.testing.assert_array_equal(d.after.to_numpy()[nan], d.before.to_numpy()[nan])
    assert np.isnan(d.r_hat.to_numpy()[nan]).all() and np.isfinite(d.r_hat.to_numpy()[~nan]).all()
    assert (d.r_hat.to_numpy()[~nan] >= -3_000.0).all() and (d.r_hat.to_numpy()[~nan] <= 3_000.0).all()
    o = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    b = w["base"].set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    ids = d.MVT_ID_mvt.to_numpy()[nan]
    np.testing.assert_array_equal(o.loc[ids].to_numpy(), b.loc[ids].to_numpy())


def test_lirf_unmatched_rows_and_matched_rows_are_byte_identical(world, built):
    """Rehearsed 2026-09-10 RED: the non-LIRF mask replaced by all-True in build() (guard 1 then compares LIRF rows and
    refuses; with guard 1 relaxed too, LIRF rows are rewritten to S1R and this test goes RED)."""
    w, (out, _, _) = world, built
    lirf_ids = w["scored"].MVT_ID_mvt[~rs.non_lirf_mask(w["scored"])].to_numpy()
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
    """Rehearsed 2026-09-10 RED: the first replaced value forced to 0 before the splice (build's finite-and-floor check
    fires first; with it deleted too, check_submission refuses the non-positive prediction here)."""
    out = built[0]
    bs.check_submission(out, world["base"][["MVT_ID_mvt"]].assign(TAXITIME_SEC_mvt=0))


def test_build_does_not_mutate_the_base(world, built):
    """Rehearsed 2026-09-10 RED: build() writing the new values into base.TAXITIME_SEC_mvt in place."""
    _, _, snap = built
    pd.testing.assert_frame_equal(world["base"], snap)


def test_info_shift_metrics_detail_and_hot_table_are_consistent(world, built):
    """Rehearsed 2026-09-10 RED: sse_shift divided by the number of replaced rows instead of len(base); hot_table's
    threshold `>=` -> `>` on a detail with rows at exactly 2,000 s."""
    w, (out, info, _) = world, built
    d = info["detail"]
    assert list(d.columns) == rs.DETAIL_COLUMNS and len(d) == info["n_rows_replaced"]
    non = rs.non_lirf_mask(w["scored"])
    np.testing.assert_array_equal(d.MVT_ID_mvt.to_numpy(), w["scored"].MVT_ID_mvt.to_numpy()[non])
    np.testing.assert_array_equal(d.p.to_numpy(), np.asarray(w["parts"]["p"])[non])
    np.testing.assert_array_equal(d.nf_cells.to_numpy(), np.asarray(w["parts"]["nf_cells"])[non])
    np.testing.assert_array_equal(d.sp.to_numpy(), w["scored"].sp.to_numpy()[non])
    b = w["base"].set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.loc[d.MVT_ID_mvt.to_numpy()].to_numpy()
    o = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.loc[d.MVT_ID_mvt.to_numpy()].to_numpy()
    np.testing.assert_array_equal(d.before.to_numpy(), b)
    np.testing.assert_array_equal(d.after.to_numpy(), o)
    np.testing.assert_array_equal(np.rint(w["s1c"][non]), d.before.to_numpy())          # before IS rint(S1C): guard 1's premise
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
    # nf_fit_r on the routed witness rows is h + r_hat (one seed: exact); S1R = mixture(p, sp, nf_hybrid(nf_cells, nf_fit_r))
    h = d.hprox_med.to_numpy()
    has = ~np.isnan(h)
    np.testing.assert_array_equal(info["nf_fit_r"][has], (h + d.r_hat.to_numpy())[has])
    np.testing.assert_array_equal(info["nf_fit_r"][~has], d.nf_fit_c.to_numpy()[~has])
    np.testing.assert_array_equal(d.after.to_numpy(), np.rint(bs.mixture(d.p, d.sp, bs.nf_hybrid(d.nf_cells, info["nf_fit_r"]))))
    # the hot table: per airport over rows with witness >= 2,000
    hot = d[d.hprox_med >= 2_000.0]
    assert set(info["hot_rows"]) == set(hot.ADEP_mvt) and len(hot) > 0
    for ap, r in info["hot_rows"].items():
        part = hot[hot.ADEP_mvt == ap]
        assert r["n"] == len(part) and r["witness_mean"] == pytest.approx(part.hprox_med.mean(), rel=1e-12)
        assert r["before_mean"] == pytest.approx(part.before.mean(), rel=1e-12) and r["after_mean"] == pytest.approx(part.after.mean(), rel=1e-12)
        assert r["after_max"] == part.after.max()
    edge = pd.DataFrame({"ADEP_mvt": ["EHAM", "EHAM", "LFPG"], "hprox_med": [2_000.0, 1_999.9, np.nan], "before": [1.0, 2.0, 3.0], "after": [4.0, 5.0, 6.0]})
    assert rs.hot_table(edge) == {"EHAM": {"n": 1, "witness_mean": 2_000.0, "before_mean": 1.0, "after_mean": 4.0, "after_max": 4.0}}
    t = us.shift_tables(d)
    assert int(t["by_bin"].n.sum()) == len(d) and int(t["by_airport"].n_changed.sum()) == info["n_values_differ"]
    lines = []
    rs.render_hot(info["hot_rows"], lines.append)
    assert lines[0].startswith("---") and len(lines) == 2 + len(info["hot_rows"])


# =============================================================================================
# the 2026 witness comes from the serve file, never from 2025
# =============================================================================================

def test_the_witness_on_the_replaced_rows_is_the_serve_files_not_2025s(world, built):
    """Rehearsed 2026-09-10 RED: hprox_med overwritten with a constant 700 s after attaching (the detail no longer equals
    the serve witness lookup)."""
    w, (out, info, _) = world, built
    d = info["detail"]
    non = rs.non_lirf_mask(w["scored"])
    serve_w = w["scored_w"]
    np.testing.assert_array_equal(d.hprox_med.to_numpy(), serve_w.hprox_med.to_numpy()[non])
    np.testing.assert_array_equal(d.hprox_n.to_numpy(), serve_w.hprox_n.to_numpy()[non])
    w25 = w["scored_25w"].hprox_med.to_numpy()[non]
    both = np.isfinite(w25) & np.isfinite(d.hprox_med.to_numpy())
    assert both.sum() > 100 and (d.hprox_med.to_numpy()[both] != w25[both]).mean() > 0.95
    assert info["witness_2026"]["n_scored_covered"] == int((serve_w.hprox_n.to_numpy() > 0).sum())
    assert info["witness_2026"]["n_serve_matched_rows"] == len(w["serve"])
    # the shipped values move TOWARD the serve witness on the hot rows: the offset model carries the level from the witness
    hot = d[(d.hprox_med >= 2_000.0) & (d.nf_cells < bs.T_TAIL_S)]
    assert len(hot) > 20 and (hot.after - hot.before).mean() > 0 and abs(hot.after - hot.hprox_med).mean() < abs(hot.before - hot.hprox_med).mean()


def test_build_refuses_scored_rows_that_arrive_with_a_witness_and_a_serve_frame_that_covers_no_scored_hour(world):
    w = world
    with pytest.raises(ValueError, match="already carry .*hprox_med.*build attaches the serve file's witness"):
        rs.build(w["train_unm"], w["scored_25w"], w["parts"], w["base"], w["serve"])
    with pytest.raises(ValueError, match="covers none of the scored airport-hours"):
        rs.build(w["train_unm"], w["scored"], w["parts"], w["base"], w["train_lirf"])     # 2025 hours never match
    with pytest.raises(ValueError, match="serve frame needs"):
        rs.build(w["train_unm"], w["scored"], w["parts"], w["base"], w["serve"].drop(columns=["AOBT_3_flt"]))
    with pytest.raises(ValueError, match="lack the witness column"):
        rs.build(w["train_unm"].drop(columns=["hprox_n"]), w["scored"], w["parts"], w["base"], w["serve"])


# =============================================================================================
# writing
# =============================================================================================

def test_write_outputs_round_trips_writes_detail_and_meta_and_refuses_to_overwrite(world, built, tmp_path):
    """Rehearsed 2026-09-10 RED: the `dest.exists()` refusal deleted (the second call overwrites); the frame written with
    its last row dropped (the re-read check_submission fails on the row count)."""
    w, (out, info, _) = world, built
    template = w["base"][["MVT_ID_mvt"]].assign(TAXITIME_SEC_mvt=0)
    dest, meta = rs.write_outputs(out, info, template, 99, "merry-quicksand_v10.parquet", tmp_path, 0.0)
    assert dest == tmp_path / "merry-quicksand_v99.parquet" and dest.exists()
    back = pd.read_parquet(dest)
    pd.testing.assert_frame_equal(back, out)
    detail = pd.read_parquet(tmp_path / "merry-quicksand_v99.unm_rows.parquet")
    pd.testing.assert_frame_equal(detail, info["detail"])
    disk = json.loads((tmp_path / "merry-quicksand_v99.meta.json").read_text())
    assert disk == meta
    assert meta["version"] == 99 and meta["base"] == "merry-quicksand_v10.parquet" and meta["file"] == dest.name
    assert meta["arm"].startswith("E3R S1R") and meta["prereg"] == ur.PREREG and meta["parent_prereg"] == uc.PREREG and meta["clip_s"] == 3_000.0
    for k in ("n_rows_replaced", "n_values_differ", "n_nan_witness_rows", "n_train_witness", "n_clipped_low", "n_clipped_high",
              "per_airport_mean_shift_s", "rms_shift_vs_base_s", "bins_2026_changed_rows", "sse_shift_vs_base_board_mse", "seeds", "hot_rows"):
        assert meta[k] == info[k], k
    assert meta["rebuild_guard"] == info["rebuild"] and "S1C" in meta["rebuild_guard"]["path"]
    assert meta["witness_2026"]["n_airport_hours"] == info["witness_2026"]["n_airport_hours"] and "ranking.parquet" in meta["witness_2026"]["source"]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        rs.write_outputs(out, info, template, 99, "merry-quicksand_v10.parquet", tmp_path, 0.0)
    with pytest.raises(ValueError, match="row count"):
        rs.write_outputs(out.iloc[:-1], info, template, 98, "merry-quicksand_v10.parquet", tmp_path, 0.0)
    assert not (tmp_path / "merry-quicksand_v98.parquet").exists()
