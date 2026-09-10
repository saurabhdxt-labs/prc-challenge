"""scripts/unm_congestion.py - arm E3C (plans/PREREG_unm_congestion_2026_09_10.md): a congestion-aware
unmatched body, S1's regressor with an airport-hour witness appended, scored by seven locked clauses.

What the tests pin, and why each exists:
  * the witness: median of MVT - AOBT_3 per airport x floored UTC hour over MATCHED rows only, NaN below
    five, unmatched rows never counted, an empty airport-hour is (NaN, 0);
  * S1C differs from S1 only through the body: p, nf_cells and sp are S1's own parts, the regressor's
    columns are NF_NUMERIC + the witness + the same encodings, same params, same non-fill rows;
  * the reproduction guard: the harness's S1 equals stratum_fold's on every lomo row to 1e-6 s, the
    reference's fold-A rows are ignored (BC-2), a perturbed or mis-shaped reference raises;
  * each clause's arithmetic on a hand-built table at its exact boundary, the bin borrowing rule, the
    date-block bootstrap's determinism and power, the verdict mapping;
  * the shipped modules are untouched (NF_NUMERIC == ["sp", "dayoff", "hr"], git-clean files);
  * smoke paths never point at the real outputs; the parquet carries its convention and a reader
    refuses one without it.
Every RNG is seeded. Mutation rehearsals are recorded per test ("Rehearsed 2026-09-10 RED: ...").
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import unm_congestion as uc  # noqa: E402
import stratum_fold as sf  # noqa: E402

bs = uc.bs
T0 = pd.Timestamp("2025-03-04 10:00:00", tz="UTC")


# =============================================================================================
# fixtures
# =============================================================================================

def _rows(ap, offsets_s, prox_s, matched=True):
    """Movement rows at one airport: MVT = T0 + offset; AOBT_3 = MVT - prox for matched rows, NaT otherwise."""
    mvt = T0 + pd.to_timedelta(np.asarray(offsets_s, dtype="float64"), unit="s")
    aobt = mvt - pd.to_timedelta(np.asarray(prox_s, dtype="float64"), unit="s")
    if not matched:
        aobt = pd.Series([pd.NaT] * len(mvt), dtype="datetime64[ns, UTC]")
    return pd.DataFrame({"ADEP_mvt": ap, "MVT_TIME_UTC_mvt": pd.Series(mvt), "AOBT_3_flt": pd.Series(aobt)})


@pytest.fixture()
def witness_frame():
    """EHAM hour 10: six matched rows (prox 100..600) plus two UNMATCHED rows in the same hour; EHAM hour 11:
    four matched rows (below the five threshold); EDDF hour 10: five matched rows; one EHAM matched row at
    10:59:59 and one at 11:00:00 on either side of the floor."""
    parts = [
        _rows("EHAM", [0, 60, 120, 180, 240, 3599], [100, 200, 300, 400, 500, 600]),   # 6 in hour 10 (3599 s floors to 10)
        _rows("EHAM", [30, 90], [0, 0], matched=False),                                # unmatched, hour 10
        _rows("EHAM", [3600, 3660, 3720, 3780], [1000, 1100, 1200, 1300]),             # 4 in hour 11
        _rows("EDDF", [0, 1, 2, 3, 4], [900, 900, 900, 2000, 2000]),                    # 5 in hour 10
    ]
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def world():
    """A small synthetic world (stratum_fold.synthetic_frame + a dense matched companion) and one LOMO split."""
    unm, lirf, summary = uc.synthetic_world(seed=3, n_per_month=350, n_matched_per_month=300)
    month = unm.month.to_numpy()
    m, tr, te = sf.lomo_folds(month)[11]           # month 12 out
    return unm, lirf, tr, te, m, summary


# =============================================================================================
# the witness
# =============================================================================================

def test_witness_median_count_nan_rule_floor_and_matched_only(witness_frame):
    """Rehearsed 2026-09-10 RED: MIN_N 5 -> 4 (EHAM hour 11 gets a median); `dt.floor("h")` -> `dt.round("h")`
    (the 10:59:59 row moves to hour 11: EHAM 10 has n 5, med 300); the matched filter reduced to MVT present
    (the two unmatched rows are counted: EHAM 10 has n 8)."""
    w = uc.airport_hour_witness(witness_frame)
    h10, h11 = T0, T0 + pd.Timedelta(hours=1)
    assert w.loc[("EHAM", h10), "hprox_n"] == 6 and w.loc[("EHAM", h10), "hprox_med"] == 350.0
    assert w.loc[("EHAM", h11), "hprox_n"] == 4 and np.isnan(w.loc[("EHAM", h11), "hprox_med"])
    assert w.loc[("EDDF", h10), "hprox_n"] == 5 and w.loc[("EDDF", h10), "hprox_med"] == 900.0
    assert len(w) == 3 and list(w.columns) == ["hprox_med", "hprox_n"]


def test_witness_attaches_by_airport_hour_and_empty_hours_are_nan_zero(witness_frame):
    """Rehearsed 2026-09-10 RED: attach keyed on the airport alone (every EHAM row gets hour 10's median)."""
    w = uc.airport_hour_witness(witness_frame)
    rows = pd.DataFrame({"ADEP_mvt": ["EHAM", "EHAM", "EDDF", "LFPG", "EHAM"],
                         "MVT_TIME_UTC_mvt": [T0 + pd.Timedelta(seconds=1799), T0 + pd.Timedelta(hours=1, minutes=5), T0,
                                              T0, T0 + pd.Timedelta(hours=5)]})
    rows["MVT_TIME_UTC_mvt"] = pd.to_datetime(rows.MVT_TIME_UTC_mvt, utc=True)
    got = uc.attach_witness(rows, w)
    np.testing.assert_array_equal(got.hprox_n.to_numpy(), [6, 4, 5, 0, 0])
    med = got.hprox_med.to_numpy()
    assert med[0] == 350.0 and med[2] == 900.0 and np.isnan(med[[1, 3, 4]]).all()
    assert got.hprox_n.dtype == np.int64 and list(got.index) == list(rows.index)


def test_witness_never_reads_block_or_the_label(witness_frame):
    """The witness is computed from a frame with neither BLOCK nor TAXITIME present, so it cannot read them."""
    assert "BLOCK_TIME_UTC_mvt" not in witness_frame.columns and "TAXITIME_SEC_mvt" not in witness_frame.columns
    uc.airport_hour_witness(witness_frame)
    with pytest.raises(ValueError, match="needs column 'AOBT_3_flt'"):
        uc.airport_hour_witness(witness_frame.drop(columns=["AOBT_3_flt"]))


def test_bin_edges_are_left_closed_and_nan_is_its_own_bin():
    """Rehearsed 2026-09-10 RED: `right=False` -> `right=True` (900 falls into <900, 2400 into 2000-2400)."""
    h = [np.nan, 0.0, 899.999, 900.0, 1099.9, 1100.0, 1299.9, 1300.0, 1499.9, 1500.0, 1999.9, 2000.0, 2399.9, 2400.0, 9e9]
    want = ["NaN", "<900", "<900", "900-1100", "900-1100", "1100-1300", "1100-1300", "1300-1500", "1300-1500",
            "1500-2000", "1500-2000", "2000-2400", "2000-2400", ">=2400", ">=2400"]
    assert uc.bin_of(h).tolist() == want
    assert uc.BINS == ["NaN", "<900", "900-1100", "1100-1300", "1300-1500", "1500-2000", "2000-2400", ">=2400"]


# =============================================================================================
# S1C: the body only
# =============================================================================================

def test_s1c_shares_p_nf_cells_and_sp_with_s1_and_differs_only_through_the_body(world):
    """Rehearsed 2026-09-10 RED: S1C built from `parts["nf_fit"]` instead of `nf_fit_c` (S1C == S1 everywhere);
    `mixture(parts["p"] * 0.99, ...)` for S1C (the rebuild from S1's parts no longer matches)."""
    unm, lirf, tr, te, m, _ = world
    tr_mat = lirf[lirf.month != m]
    arms, parts = uc.congestion_arms(unm[tr], unm[te], tr_mat, seeds=(0,))
    ref_arms, ref_parts = sf.stratum_arms(unm[tr], unm[te], tr_mat, seeds=(0,))
    np.testing.assert_array_equal(arms["S1"], ref_arms["S1"])
    np.testing.assert_array_equal(arms["S1_seed0"], ref_arms["S1_seed0"])
    np.testing.assert_array_equal(parts["p"], ref_parts["p"])
    np.testing.assert_array_equal(parts["nf_cells"], ref_parts["nf_cells"])
    want = bs.mixture(ref_parts["p"], ref_parts["sp"], bs.nf_hybrid(ref_parts["nf_cells"], parts["nf_fit_c"]))
    np.testing.assert_array_equal(arms["S1C"], want)
    np.testing.assert_array_equal(arms["S1C_seed0"], want)               # one seed: the mean is the seed
    np.testing.assert_array_equal(parts["nf_fit_c"], parts["nf_fit_c_by_seed"][0])
    body = parts["nf_cells"] < bs.T_TAIL_S
    assert (arms["S1C"][body] != arms["S1"][body]).mean() > 0.9         # the body differs on (almost) every routed row
    np.testing.assert_array_equal(arms["S1C"][~body], arms["S1"][~body])  # the tail is the cells' in both arms


def test_congestion_regressor_columns_params_and_refusals(world):
    """Rehearsed 2026-09-10 RED: `NUMERIC = [*bs.NF_NUMERIC]` (the witness dropped from the design)."""
    unm, _, tr, _, _, _ = world
    nonfill = unm[tr][~bs.schedule_fill(unm[tr])]
    reg = uc.fit_congestion_regressor(nonfill, 1)
    assert isinstance(reg, bs.NonFillRegressor)
    assert reg.columns == ["sp", "dayoff", "hr", "hprox_med", "hprox_n", *["te_" + c for c in bs.NF_ENCODED]]
    assert list(reg.design(nonfill.head(3)).columns) == reg.columns
    assert reg.model.get_params()["max_leaf_nodes"] == 15 and reg.model.get_params()["max_iter"] == 200
    assert reg.model.get_params()["min_samples_leaf"] == 100 and reg.model.get_params()["random_state"] == 1
    assert reg.n_train == len(nonfill) and reg.seed == 1
    ref = bs.fit_nf_regressor(nonfill, 1)
    assert reg.prior == ref.prior and all(reg.maps[c].equals(ref.maps[c]) for c in bs.NF_ENCODED)
    with pytest.raises(ValueError, match="NON-FILL rows only"):
        uc.fit_congestion_regressor(unm[tr], 1)
    with pytest.raises(ValueError, match="needs the witness columns"):
        uc.fit_congestion_regressor(nonfill.drop(columns=["hprox_med"]), 1)
    with pytest.raises(ValueError, match="needs non-fill training rows"):
        uc.fit_congestion_regressor(nonfill.iloc[:0], 1)


def test_shipped_modules_are_untouched():
    """The prereg composes the shipped code; it never edits it."""
    assert bs.NF_NUMERIC == ["sp", "dayoff", "hr"]
    assert bs.NF_ENCODED == ["ADEP_mvt", "RUNWAY_mvt", "stand_pref", "airline", "AIRCRAFT_OPERATOR_flt"]
    assert bs.NF_PARAMS == dict(max_leaf_nodes=15, max_iter=200, min_samples_leaf=100) and bs.WINSOR_S == 3_000.0
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "scripts/build_submission.py", "scripts/stratum_fold.py"],
                       cwd=ROOT, capture_output=True)
    if r.returncode not in (0, 1):
        pytest.skip("git unavailable")
    assert r.returncode == 0, "scripts/build_submission.py or scripts/stratum_fold.py differs from HEAD"


# =============================================================================================
# LOMO and the reproduction guard
# =============================================================================================

@pytest.fixture(scope="module")
def lomo_preds(world):
    unm, lirf, _, _, _, _ = world
    small = unm[unm.month.isin([1, 2, 3])].reset_index(drop=True)      # three folds keep this under a few seconds
    lirf3 = lirf[lirf.month.isin([1, 2, 3])].reset_index(drop=True)
    preds = uc.score_lomo(small, lirf3, uc.SEEDS, lambda s: None)
    ref = sf.score_lomo(small, lirf3, sf.SEEDS, lambda s: None)
    return small, preds, ref


def test_lomo_scores_every_row_once_and_s1_reproduces_stratum_fold(lomo_preds):
    """Rehearsed 2026-09-10 RED: the guard's tolerance 1e-6 -> 1e-3 survives the equality (S1 IS bit-identical),
    so the reference is perturbed by 1e-5 below: RED when the `bad` check is dropped from check_reproduction."""
    small, preds, ref = lomo_preds
    assert len(preds) == len(small) and preds.MVT_ID_mvt.is_unique and set(preds.MVT_ID_mvt) == set(small.MVT_ID_mvt)
    assert list(preds.columns) == uc.PRED_COLUMNS and (preds.fold == "lomo").all()
    got = uc.check_reproduction(preds, ref)
    assert got["status"] == "reproduced" and got["max_abs_diff_s"] == 0.0 and got["n_rows"] == len(small)
    bad = ref.copy()
    bad.loc[5, "S1"] += 1e-5
    with pytest.raises(AssertionError, match="does not reproduce the stored S1 on 1 of"):
        uc.check_reproduction(preds, bad)
    with pytest.raises(AssertionError, match="different row sets"):
        uc.check_reproduction(preds, ref.iloc[1:])


def test_reproduction_guard_ignores_the_references_fold_a_rows(lomo_preds):
    """BC-2: the stored reference holds the same ids under fold 'A' too. Rehearsed 2026-09-10 RED: the
    `fold == "lomo"` filter dropped from the reference (duplicate ids -> the guard raises on a correct harness)."""
    _, preds, ref = lomo_preds
    fold_a = ref.copy()
    fold_a["fold"] = "A"
    fold_a["S1"] = fold_a.S1 + 50.0
    uc.check_reproduction(preds, pd.concat([ref, fold_a], ignore_index=True))


def test_lomo_row_columns_carry_the_witness_date_and_per_seed_arms(lomo_preds):
    small, preds, _ = lomo_preds
    p = preds.set_index("MVT_ID_mvt").loc[small.MVT_ID_mvt.to_numpy()]
    np.testing.assert_array_equal(p.hprox_n.to_numpy(), small.hprox_n.to_numpy())
    np.testing.assert_array_equal(p.hprox_med.to_numpy(), small.hprox_med.to_numpy())
    np.testing.assert_array_equal(p.date.to_numpy(), small.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy())
    # S1C is the mixture of the seed-MEAN body; every per-seed arm is the same mixture of that seed's body
    body = bs.nf_hybrid(p.nf_cells.to_numpy(), p.nf_fit_c.to_numpy())
    np.testing.assert_array_equal(p.S1C.to_numpy(), bs.mixture(p.p_hat.to_numpy(), p.sp.to_numpy(), body))
    np.testing.assert_array_equal(p.S1.to_numpy(), bs.mixture(p.p_hat.to_numpy(), p.sp.to_numpy(), bs.nf_hybrid(p.nf_cells.to_numpy(), p.nf_fit.to_numpy())))
    assert (p.S1C >= 1.0).all() and np.isfinite(p.S1C).all() and {f"S1C_seed{s}" for s in (0, 1, 2)} <= set(p.columns)
    with pytest.raises(ValueError, match="registered seeds"):
        uc.score_lomo(small, small.iloc[:0], (0,), lambda s: None)


def test_lomo_parquet_round_trips_with_its_convention_and_a_reader_refuses_an_unstamped_file(lomo_preds, tmp_path):
    """Rehearsed 2026-09-10 RED: `write_lomo` stamping convention 'delta' (the reader refuses its own file)."""
    _, preds, _ = lomo_preds
    path = tmp_path / "lomo.parquet"
    uc.write_lomo(preds, path, {"reproduction": {"status": "reproduced"}})
    back, meta = uc.read_lomo(path)
    pd.testing.assert_frame_equal(back, preds[uc.PRED_COLUMNS])
    assert meta["reproduction"]["status"] == "reproduced"
    assert pq.read_schema(path).metadata[b"convention"] == b"taxi_time"
    bare = tmp_path / "bare.parquet"
    pq.write_table(pa.Table.from_pandas(preds, preserve_index=False), bare)
    with pytest.raises(ValueError, match="carries convention ''"):
        uc.read_lomo(bare)


# =============================================================================================
# clauses on a hand-built table
# =============================================================================================

#: exact encodings of a per-row gain g = SE(S1) - SE(S1C) = a^2 - c^2 with a, c integers or halves, so every
#: sum below is exact in fp64 (no sqrt round-trip): S1 = y + a, S1C = y + c.
_ENC = {40.0: (7.0, 3.0), 60.0: (8.0, 2.0), -5.0: (2.0, 3.0), -12.0: (2.0, 4.0), 30.0: (6.5, 3.5), 26.0: (7.5, 5.5),
        -6.0: (0.5, 2.5), -700.0: (7.5, 27.5)}
_APS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LSZH", "LTFM"]


def _table(override: dict | None = None) -> pd.DataFrame:
    """36 non-LIRF rows, three per month (j = 0: NaN bin, 1: calm 800 s, 2: hot 2,500 s), airports cycling over
    the nine, one date per month, built so every clause sits at or near its boundary:
      g(m, j): months 1..8 -> NaN +40, calm -5, hot +60 (month sum +95); months 9..12 -> -12, -5, -12 (-29);
      row 0 (month 1, NaN) is the one monster (y = 10,801) and carries +30 instead of +40.
      pooled +634; months positive 8 of 12 (C3 at the floor); airports positive 6 of 9 (EDDM, LEBL, LSZH negative;
      C4 at the floor); calm sum -60 against a bound of -63.4 (C5); ex-monster +604 (C6); the per-seed arms equal
      the pooled arms, so C7's sd is 0. `override` = {(month, j): g} replaces cells with another _ENC value."""
    rows, k = [], 0
    for m in range(1, 13):
        for j in range(3):
            g = [40.0, -5.0, 60.0][j] if m <= 8 else [-12.0, -5.0, -12.0][j]
            if (m, j) == (1, 0):
                g = 30.0
            if override and (m, j) in override:
                g = override[(m, j)]
            a, c = _ENC[g]
            y = 10_801.0 if (m, j) == (1, 0) else 1_000.0
            rows.append(dict(month=m, date=f"2025-{m:02d}-15", ADEP_mvt=_APS[k % 9], y=y, hprox_med=[np.nan, 800.0, 2500.0][j],
                             S1=y + a, S1C=y + c))
            k += 1
    t = pd.DataFrame(rows)
    for s in (0, 1, 2):
        t[f"S1_seed{s}"], t[f"S1C_seed{s}"] = t.S1, t.S1C
    return t


def test_hand_table_is_what_its_docstring_says():
    t = _table()
    g = uc.gains(t)
    assert g.sum() == 634.0 and len(t) == 36                              # exact by construction (_ENC)
    by_m = pd.Series(g).groupby(t.month.to_numpy()).sum()
    assert (by_m > 0).sum() == 8 and (by_m.loc[9:] == -29.0).all() and by_m.loc[1] == 85.0 and by_m.loc[2] == 95.0
    by_a = pd.Series(g).groupby(t.ADEP_mvt.to_numpy()).sum()
    assert (by_a > 0).sum() == 6 and list(by_a.index[by_a < 0]) == ["EDDM", "LEBL", "LSZH"]
    assert g[t.hprox_med.to_numpy() < 1100.0].sum() == -60.0
    assert g[t.y.to_numpy() <= 10_800.0].sum() == 604.0 and g[0] == 30.0 and t.y[0] == 10_801.0


def test_clauses_c3_to_c7_arithmetic_at_the_boundaries():
    """Rehearsed 2026-09-10 RED: MIN_MONTHS 8 -> 9; MIN_AIRPORTS 6 -> 7; CALM_TOLERANCE 0.10 -> 0.05 (calm -60
    against a bound of -31.7 fails); C6's `y <= MONSTER_S` -> `y < MONSTER_S` (the added row at exactly 10,800
    leaves the sum: +604 - 0 instead of -96)."""
    t = _table()
    counts = {b: 100 for b in uc.BINS}
    c = uc.clauses(t, counts, n_boot=200, seed=0, n_scored=1000)
    assert c["pooled_sum_g"] == 634.0 and c["n_rows"] == 36
    c3 = c["C3_months"]
    assert c3["months_positive"] == 8 and c3["n_months"] == 12 and c3["passes"]
    assert c3["per_month"]["1"] == {"sum_g": 85.0, "n": 3} and c3["per_month"]["12"] == {"sum_g": -29.0, "n": 3}
    c4 = c["C4_airports"]
    assert c4["airports_positive"] == 6 and c4["n_airports"] == 9 and c4["passes"]
    assert c4["per_airport"]["EDDF"] == {"sum_g": 98.0, "n": 4} and c4["per_airport"]["LSZH"] == {"sum_g": -20.0, "n": 4}
    c5 = c["C5_calm_control"]
    assert c5["calm_sum_g"] == -60.0 and c5["n_calm"] == 12 and c5["pooled_sum_g"] == 634.0
    assert c5["bound"] == pytest.approx(-63.4, abs=1e-12) and c5["passes"]        # 0.1 x 634 is not exact in binary
    c6 = c["C6_not_the_monsters"]
    assert c6["exmonster_sum_g"] == 604.0 and c6["n_exmonster"] == 35 and c6["passes"]
    c7 = c["C7_seeds"]
    assert c7["seed_sd"] == 0.0 and c7["passes"] and c7["per_seed_sum_g"] == {"0": 634.0, "1": 634.0, "2": 634.0}
    # a row at exactly y = 10,800 is NOT a monster: it belongs to C6's sum and, at g = -700, sinks it
    a, cc = _ENC[-700.0]
    extra = t.iloc[[1]].assign(y=10_800.0, S1=10_800.0 + a, S1C=10_800.0 + cc)
    for s in (0, 1, 2):
        extra[f"S1_seed{s}"], extra[f"S1C_seed{s}"] = extra.S1, extra.S1C
    c6b = uc.clauses(pd.concat([t, extra], ignore_index=True), counts, n_boot=200, seed=0, n_scored=1000)["C6_not_the_monsters"]
    assert c6b["n_exmonster"] == 36 and c6b["exmonster_sum_g"] == -96.0 and not c6b["passes"]


def test_c5_at_its_exact_bound_and_c3_c4_need_the_full_panel():
    """Rehearsed 2026-09-10 RED: C5 `>=` -> `>` (calm -60 against a bound of exactly -60.0 fails)."""
    counts = {b: 100 for b in uc.BINS}
    t = _table({(2, 2): 26.0})                      # month 2's hot row 60 -> 26: pooled 600, bound -0.1 x 600 = -60.0 exactly
    c = uc.clauses(t, counts, n_boot=200, seed=0, n_scored=1000)
    c5 = c["C5_calm_control"]
    assert c5["pooled_sum_g"] == 600.0 and c5["bound"] == -60.0 and c5["calm_sum_g"] == -60.0 and c5["passes"]
    assert c["C3_months"]["passes"] and c["C4_airports"]["passes"]       # month 2 is still +61, EHAM/LEMD still positive
    t = _table({(2, 2): 26.0, (3, 1): -6.0})        # one calm row -5 -> -6: calm -61, pooled 599, bound -59.9
    c5 = uc.clauses(t, counts, n_boot=200, seed=0, n_scored=1000)["C5_calm_control"]
    assert c5["calm_sum_g"] == -61.0 and c5["pooled_sum_g"] == 599.0 and not c5["passes"]
    # a panel with eleven months cannot pass C3 whatever the signs; eight airports cannot pass C4
    t = _table()
    c = uc.clauses(t[t.month != 12], counts, n_boot=200, seed=0, n_scored=1000)
    assert c["C3_months"]["n_months"] == 11 and c["C3_months"]["months_positive"] == 8 and not c["C3_months"]["passes"]
    c = uc.clauses(t[t.ADEP_mvt != "LTFM"], counts, n_boot=200, seed=0, n_scored=1000)
    assert c["C4_airports"]["n_airports"] == 8 and not c["C4_airports"]["passes"]
    with pytest.raises(ValueError, match="non-LIRF rows only"):
        uc.clauses(t.assign(ADEP_mvt=np.where(t.index == 3, "LIRF", t.ADEP_mvt)), counts, n_boot=200)


def test_c2_pricing_and_the_downward_borrowing_rule():
    """Rehearsed 2026-09-10 RED: borrowing from the next HIGHER bin (`i + 1`); the thin threshold 30 -> 29
    (a 29-row bin prices itself); the NaN bin borrowing from <900."""
    g = np.concatenate([np.full(50, 2.0), np.full(40, -1.0), np.full(31, 5.0), np.full(29, 100.0), np.full(3, 200.0),
                        np.full(35, 7.0), np.zeros(0), np.full(10, 9.0)])
    h = np.concatenate([np.full(50, np.nan), np.full(40, 500.0), np.full(31, 1000.0), np.full(29, 1200.0), np.full(3, 1400.0),
                        np.full(35, 1700.0), np.zeros(0), np.full(10, 3000.0)])
    table = uc.bin_table(g, h)
    assert {k: v["n_2025"] for k, v in table.items()} == {"NaN": 50, "<900": 40, "900-1100": 31, "1100-1300": 29,
                                                          "1300-1500": 3, "1500-2000": 35, "2000-2400": 0, ">=2400": 10}
    assert table["2000-2400"]["per_row_gain"] is None and table["<900"]["per_row_gain"] == -1.0
    counts = {"NaN": 1000, "<900": 2000, "900-1100": 300, "1100-1300": 200, "1300-1500": 100, "1500-2000": 50,
              "2000-2400": 20, ">=2400": 10}
    p = uc.price_2026(table, counts, n_scored=10_000)
    used = {k: v["per_row_gain_used"] for k, v in p["bins"].items()}
    assert used == {"NaN": 2.0, "<900": -1.0, "900-1100": 5.0, "1100-1300": 5.0, "1300-1500": 5.0, "1500-2000": 7.0,
                    "2000-2400": 7.0, ">=2400": 7.0}
    src = {k: v["borrowed_from"] for k, v in p["bins"].items()}
    assert src == {"NaN": "NaN", "<900": "<900", "900-1100": "900-1100", "1100-1300": "900-1100", "1300-1500": "900-1100",
                   "1500-2000": "1500-2000", "2000-2400": "1500-2000", ">=2400": "1500-2000"}
    want = (2.0 * 1000 - 1.0 * 2000 + 5.0 * 300 + 5.0 * 200 + 5.0 * 100 + 7.0 * 50 + 7.0 * 20 + 7.0 * 10) / 10_000
    assert p["priced_mse"] == pytest.approx(want, rel=1e-12) and p["passes"] is (want >= 500.0)
    assert p["bins"]["1100-1300"]["thin"] and not p["bins"]["900-1100"]["thin"]
    assert p["bins"]["1100-1300"]["priced_mse"] == pytest.approx(5.0 * 200 / 10_000, rel=1e-12)
    # a thin NaN bin has no lower bin: priced at zero and flagged, even when <900 is well populated
    g2, h2 = np.concatenate([np.full(5, 50.0), np.full(40, 50.0), np.full(40, 3.0)]), np.concatenate([np.full(5, np.nan), np.full(40, 100.0), np.full(40, 950.0)])
    p2 = uc.price_2026(uc.bin_table(g2, h2), {"NaN": 100, "<900": 100, "900-1100": 100}, n_scored=100)
    assert p2["bins"]["NaN"]["per_row_gain_used"] == 0.0 and p2["bins"]["NaN"]["borrowed_from"] is None
    assert p2["bins"]["<900"]["per_row_gain_used"] == 50.0 and p2["priced_mse"] == pytest.approx(53.0, rel=1e-12)
    # a thin <900 bin has no lower bin either: priced at zero and flagged, whatever the NaN bin holds
    g3, h3 = np.concatenate([np.full(40, 50.0), np.full(5, 50.0), np.full(40, 3.0)]), np.concatenate([np.full(40, np.nan), np.full(5, 100.0), np.full(40, 950.0)])
    p3 = uc.price_2026(uc.bin_table(g3, h3), {"NaN": 100, "<900": 100, "900-1100": 100}, n_scored=100)
    assert p3["bins"]["<900"]["per_row_gain_used"] == 0.0 and p3["bins"]["<900"]["borrowed_from"] is None
    assert p3["bins"]["NaN"]["per_row_gain_used"] == 50.0 and p3["priced_mse"] == pytest.approx(53.0, rel=1e-12)


def test_c2_passes_at_exactly_500_and_uses_the_registered_denominator():
    """Rehearsed 2026-09-10 RED: `>= C2_MIN_MSE` -> `>` (the total is EXACTLY 500.0: per-row 500 x 344,841 rows
    / 344,841); N_SCORED_2026 default 344,841 -> 344,842."""
    g, h = np.full(30, 500.0), np.full(30, 2500.0)                                 # 30 rows in >=2400, mean exactly 500
    table = uc.bin_table(g, h)
    p = uc.price_2026(table, {">=2400": 344_841})
    assert p["n_scored_2026"] == 344_841 and p["priced_mse"] == 500.0 and p["passes"]
    p = uc.price_2026(table, {">=2400": 344_840})
    assert p["priced_mse"] < 500.0 and not p["passes"]


def test_date_block_bootstrap_is_deterministic_finds_a_planted_gain_and_rejects_noise():
    """Rehearsed 2026-09-10 RED: `seed` ignored (default_rng() unseeded: two calls differ); blocks = rows instead
    of dates (the block count reads 3000, not 30)."""
    rng = np.random.default_rng(11)
    dates = np.repeat([f"2025-01-{d:02d}" for d in range(1, 31)], 100)
    g = rng.normal(5.0, 1.0, 3000)
    a = uc.date_block_bootstrap(g, dates, n_draws=500, seed=0)
    b = uc.date_block_bootstrap(g, dates, n_draws=500, seed=0)
    assert a == b and a["n_blocks"] == 30 and a["n_draws"] == 500 and a["seed"] == 0
    assert a["sum_g"] == pytest.approx(g.sum(), rel=1e-12) and a["ci95"][0] > 0 and a["passes"]
    assert uc.date_block_bootstrap(g, dates, n_draws=500, seed=1)["ci95"] != a["ci95"]
    noise = rng.normal(0.0, 1.0, 3000)
    z = uc.date_block_bootstrap(noise, dates, n_draws=500, seed=0)
    assert z["ci95"][0] < 0 < z["ci95"][1] and not z["passes"]
    # a date-level artefact: one date carries the whole gain -> the interval includes zero, a row bootstrap would not
    art = np.zeros(3000)
    art[:100] = 50.0
    assert not uc.date_block_bootstrap(art, dates, n_draws=500, seed=0)["passes"]
    with pytest.raises(ValueError, match="at least two dates"):
        uc.date_block_bootstrap(g[:100], dates[:100], n_draws=10, seed=0)
    # hand check: with two dates the draw sums are one of {2 s1, s1 + s2, 2 s2}
    two = uc.date_block_bootstrap([1.0, 2.0, 10.0], ["a", "a", "b"], n_draws=50, seed=0)
    assert two["sum_g"] == 13.0 and 6.0 <= two["ci95"][0] <= two["ci95"][1] <= 20.0


def test_verdict_mapping():
    """Rehearsed 2026-09-10 RED: `if not passes[0]` -> `if not passes[1]` (a C2-only failure reads NOT WORKING)."""
    keys = ["C1_date_block_bootstrap", "C2_priced_2026", "C3_months", "C4_airports", "C5_calm_control",
            "C6_not_the_monsters", "C7_seeds"]
    ok = {k: {"passes": True} for k in keys}
    assert uc.verdict(ok) == "WORKING"
    for k in keys[1:]:
        one = {kk: {"passes": kk != k} for kk in keys}
        assert uc.verdict(one) == "INCONCLUSIVE", k
    assert uc.verdict({**ok, "C1_date_block_bootstrap": {"passes": False}}) == "NOT WORKING"
    assert uc.verdict({k: {"passes": False} for k in keys}) == "NOT WORKING"
    assert uc.verdict({**ok, "C1_date_block_bootstrap": {"passes": False}, "C7_seeds": {"passes": False}}) == "NOT WORKING"


def test_clauses_end_to_end_on_the_synthetic_world_carry_a_verdict_and_every_table(lomo_preds):
    small, preds, _ = lomo_preds
    non = preds[preds.ADEP_mvt != "LIRF"].reset_index(drop=True)
    counts = uc.bin_counts(uc.bin_of(non.hprox_med))
    c = uc.clauses(non, counts, n_boot=50, seed=0, n_scored=len(non))
    assert c["verdict"] in ("WORKING", "NOT WORKING", "INCONCLUSIVE") and c["verdict"] == uc.verdict(c)
    assert set(c["C2_priced_2026"]["bins"]) == set(uc.BINS) and sum(v["n_2025"] for v in c["C2_priced_2026"]["bins"].values()) == len(non)
    assert c["C3_months"]["n_months"] == 3 and not c["C3_months"]["passes"]      # three months can never clear 8 of 12
    assert c["C1_date_block_bootstrap"]["n_blocks"] == non.date.nunique()


# =============================================================================================
# the 2026 witness, smoke conventions and paths
# =============================================================================================

def test_witness_2026_from_frames_bins_the_scored_non_lirf_unmatched_rows(world):
    """Rehearsed 2026-09-10 RED: the LIRF exclusion dropped (LIRF rows appear in the 2026 table)."""
    unm, lirf, _, _, _, _ = world
    rows = unm[unm.month == 7]
    companion = uc.smoke_matched_companion(rows, seed=9)
    ranking = pd.concat([rows.drop(columns=["hprox_med", "hprox_n"]), companion.assign(unmatched=False)], ignore_index=True)
    ranking["unmatched"] = ranking.AOBT_3_flt.isna()
    for c in ("month", "sp"):
        ranking[c] = ranking[c].fillna(0)
    ids = rows.MVT_ID_mvt.to_numpy()[::2]
    w = uc.witness_2026_from_frames(ranking, ids)
    want = rows[rows.MVT_ID_mvt.isin(ids) & (rows.ADEP_mvt != "LIRF")]
    assert len(w) == len(want) and set(w.MVT_ID_mvt) == set(want.MVT_ID_mvt) and (w.ADEP_mvt != "LIRF").all()
    assert list(w.columns) == ["MVT_ID_mvt", "ADEP_mvt", "month", "sp", "hprox_med", "hprox_n", "bin"]
    np.testing.assert_array_equal(w.bin.to_numpy(), uc.bin_of(w.hprox_med))
    assert w.hprox_n.max() >= 5 and sum(uc.bin_counts(w.bin).values()) == len(w)
    with pytest.raises(ValueError, match="joined .* scored rows"):
        uc.witness_2026_from_frames(ranking, np.append(ids, -1.0))


def test_smoke_paths_never_touch_the_real_outputs():
    smoke, real = uc.output_paths(True), uc.output_paths(False)
    assert set(smoke) == set(real) == {"lomo", "witness_2026", "json", "lomo_log", "score_log"}
    assert all(p.parent == ROOT / "data" / "smoke_unm_congestion" for p in smoke.values())
    assert real["lomo"] == ROOT / "data" / "cache_stand" / "unm_congestion_lomo.parquet"
    assert real["json"] == ROOT / "reports" / "unm_congestion.json"
    assert not (set(smoke.values()) & set(real.values()))


def test_synthetic_world_populates_the_witness_and_leaves_some_hours_nan(world):
    unm, _, _, _, _, summary = world
    assert 0.0 < unm.hprox_med.isna().mean() < 0.9 and (unm.hprox_n >= 0).all()
    assert summary["unmatched_nan_share"] == unm.hprox_med.isna().mean()
    assert set(summary["unmatched_bin_counts"]) <= set(uc.BINS)
    assert unm.hprox_med.dropna().between(300, 3000).all()                # the planted prox: 700 / 1,600 plus noise
