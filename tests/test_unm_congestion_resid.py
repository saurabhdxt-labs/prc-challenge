"""scripts/unm_congestion_resid.py - arm E3R (plans/PREREG_unm_congestion_resid_2026_09_10.md): the congestion body
learns the clipped offset to the witness, scored by seven locked clauses with an event-excluded C2.

What the tests pin, and why each exists:
  * the residual target: r = clip(y - hprox_med, -3000, +3000) on non-fill rows WITH a witness; the prior and the
    encodings are computed on r; the prediction adds hprox_med back; NaN-witness rows fall back to S1C bit-exact;
  * S1R differs from S1C only through the body: p, nf_cells, sp, seeds and the tail routing are S1C's own;
  * the reproduction guard: S1C (and S1, the per-seed S1C arms) equal E3C's on every lomo row to 1e-6 s; a perturbed
    or mis-shaped reference raises; the reference must carry the taxi-time stamp and its own 'reproduced' status;
  * the event-excluded C2 on a hand-built table: the dropped airport-month is the largest |sum g| (a NEGATIVE event
    here), the excluded table is re-binned so a bin turns thin and borrows, the price sits EXACTLY at +300;
  * C1 / C3-C7 through E3C's arithmetic on the renamed view, the verdict mapping with the decisional C2;
  * the shipped modules are untouched (git-clean tracked files; content hashes for the untracked E3C harness);
  * smoke paths never point at the real outputs; the parquet carries its convention and a reader refuses one without it.
Every RNG is seeded. Mutation rehearsals are recorded per test ("Rehearsed 2026-09-10 RED: ...").
"""
from __future__ import annotations

import hashlib
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
import unm_congestion_resid as ur  # noqa: E402
import unm_congestion as uc  # noqa: E402
import stratum_fold as sf  # noqa: E402

bs = uc.bs
SEEDS = (0, 1, 2)
#: sha256 of the E3C harness files at E3R's writing (2026-09-10). They are untracked, so `git diff` cannot see an edit
#: to them; the prereg composes them and never edits them. A red here means one of them changed: re-read before trusting.
E3C_HASHES = {"scripts/unm_congestion.py": "e938e8bca54ee8ee9791c75fa57d956131461451fd9e63136538c56140cf6bc6",
              "scripts/unm_congestion_ship.py": "79cf63366f8ba7fea5933861ab5ac0dda70435fe5259435fdd1babdac36e271b"}


# =============================================================================================
# fixtures
# =============================================================================================

@pytest.fixture(scope="module")
def world():
    """E3C's small synthetic world (stratum_fold.synthetic_frame + a dense matched companion) and one LOMO split."""
    unm, lirf, summary = uc.synthetic_world(seed=3, n_per_month=350, n_matched_per_month=300)
    month = unm.month.to_numpy()
    m, tr, te = sf.lomo_folds(month)[11]           # month 12 out
    return unm, lirf, tr, te, m, summary


@pytest.fixture(scope="module")
def arms_and_parts(world):
    unm, lirf, tr, te, m, _ = world
    tr_mat = lirf[lirf.month != m]
    arms, parts = ur.residual_arms(unm[tr], unm[te], tr_mat, seeds=(0,))
    ref_arms, ref_parts = uc.congestion_arms(unm[tr], unm[te], tr_mat, seeds=(0,))
    return arms, parts, ref_arms, ref_parts


@pytest.fixture(scope="module")
def lomo_preds(world):
    unm, lirf, _, _, _, _ = world
    small = unm[unm.month.isin([1, 2, 3])].reset_index(drop=True)      # three folds keep this under a few seconds
    lirf3 = lirf[lirf.month.isin([1, 2, 3])].reset_index(drop=True)
    preds = ur.score_lomo(small, lirf3, ur.SEEDS, lambda s: None)
    ref = uc.score_lomo(small, lirf3, uc.SEEDS, lambda s: None)
    return small, preds, ref


# =============================================================================================
# the residual target
# =============================================================================================

def test_residual_target_is_the_clipped_offset_and_the_encodings_are_computed_on_it(world):
    """Rehearsed 2026-09-10 RED: the clip in ResidualRegressor 3000 -> 2000 (the prior and the maps no longer match the
    hand-computed r); `frame.assign(y=r)` -> `frame.assign(y=raw)` (predicted to survive because the parent re-clips at
    WINSOR_S == CLIP_S -- it went RED instead: the parent's `n_winsorised` counts the 57 rows above the cap in what it
    was handed, and this test pins it at 0, so the unclipped hand-off is visible); the `hprox_med +` dropped from
    predict (predict != h + predict_offset)."""
    unm, _, tr, te, _, _ = world
    nonfill = unm[tr][~bs.schedule_fill(unm[tr])]
    fitted = nonfill[nonfill.hprox_med.notna()].copy()
    idx = fitted.index[:3]
    fitted.loc[idx, "y"] = fitted.loc[idx, "hprox_med"] - 5_000.0          # three rows below the cap; the world has rows above it
    raw = fitted.y.to_numpy() - fitted.hprox_med.to_numpy()
    assert (raw < -3000).sum() == 3 and (raw > 3000).sum() > 10
    r = np.clip(raw, -3000.0, 3000.0)
    reg = ur.fit_residual_regressor(fitted, 1)
    assert isinstance(reg, uc.CongestionRegressor) and reg.seed == 1 and reg.n_train == len(fitted) == reg.n_train_witness
    assert reg.n_clipped_low == 3 and reg.n_clipped_high == int((raw > 3000).sum()) and reg.n_winsorised == 0
    assert reg.prior == r.mean()
    f = fitted.assign(r=r)
    for c in bs.NF_ENCODED:
        g = f.groupby(c, observed=True)["r"].agg(["mean", "size"])
        want = (g["mean"] * g["size"] + r.mean() * bs.SMOOTH) / (g["size"] + bs.SMOOTH)
        pd.testing.assert_series_equal(reg.maps[c], want, check_names=False)
    assert reg.columns == ["sp", "dayoff", "hr", "hprox_med", "hprox_n", *["te_" + c for c in bs.NF_ENCODED]]
    assert reg.model.get_params()["max_leaf_nodes"] == 15 and reg.model.get_params()["max_iter"] == 200
    assert reg.model.get_params()["min_samples_leaf"] == 100 and reg.model.get_params()["random_state"] == 1
    test = unm[te]
    off = reg.predict_offset(test)
    assert off.shape == (len(test),) and np.isfinite(off).all() and r.min() <= off.min() and off.max() <= r.max()
    pred = reg.predict(test)
    h = test.hprox_med.to_numpy()
    np.testing.assert_array_equal(pred[~np.isnan(h)], (h + off)[~np.isnan(h)])
    assert np.isnan(pred[np.isnan(h)]).all() and np.isnan(h).sum() > 0
    assert ur.CLIP_S == bs.WINSOR_S == 3_000.0


def test_fit_residual_regressor_refusals(world):
    """Rehearsed 2026-09-10 RED: the NaN-witness refusal deleted (a frame with NaN witness rows is fitted)."""
    unm, _, tr, _, _, _ = world
    nonfill = unm[tr][~bs.schedule_fill(unm[tr])]
    with pytest.raises(ValueError, match="rows WITH a witness only; .* rows have hprox_med NaN"):
        ur.fit_residual_regressor(nonfill, 0)
    fitted = nonfill[nonfill.hprox_med.notna()]
    with pytest.raises(ValueError, match="NON-FILL rows only"):
        ur.fit_residual_regressor(unm[tr][unm[tr].hprox_med.notna()], 0)
    with pytest.raises(ValueError, match="needs non-fill training rows"):
        ur.fit_residual_regressor(fitted.iloc[:0], 0)
    with pytest.raises(ValueError, match="needs columns \\['hprox_n'\\]"):
        ur.fit_residual_regressor(fitted.drop(columns=["hprox_n"]), 0)
    with pytest.raises(ValueError, match="needs columns \\['y'"):
        ur.ResidualRegressor(fitted.drop(columns=["y"]), 0)


def test_residual_nf_fit_adds_the_witness_back_and_falls_back_to_s1c_without_one():
    """Rehearsed 2026-09-10 RED: `np.where(has, h + r, c)` -> `np.where(has, c, h + r)` (the NaN row becomes NaN and the
    finite check raises; the witness rows read the fallback)."""
    got = ur.residual_nf_fit([np.nan, 1000.0, 2000.0], [5.0, -100.0, 300.0], [700.0, 800.0, 900.0])
    np.testing.assert_array_equal(got, [700.0, 900.0, 2300.0])
    assert got.dtype == np.float64
    with pytest.raises(ValueError, match="shape mismatch"):
        ur.residual_nf_fit([1.0, 2.0], [1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="not finite"):
        ur.residual_nf_fit([1000.0], [np.inf], [1.0])
    with pytest.raises(ValueError, match="not finite"):
        ur.residual_nf_fit([np.nan], [0.0], [np.nan])


# =============================================================================================
# S1R: the body only
# =============================================================================================

def test_s1r_shares_p_nf_cells_sp_and_the_tail_with_s1c_and_equals_s1c_on_nan_witness_rows(arms_and_parts, world):
    """Rehearsed 2026-09-10 RED: the fallback taken from parts["nf_fit_by_seed"] (S1's body) instead of S1C's (the
    NaN-witness rows no longer equal S1C); S1R built from mixture(p * 0.99, ...) (the rebuild from S1C's parts breaks);
    the body fitted on ALL non-fill rows (ResidualRegressor refuses the NaN rows)."""
    arms, parts, ref_arms, ref_parts = arms_and_parts
    unm, _, tr, te, _, _ = world
    for a in ("S1", "S1_seed0", "S1C", "S1C_seed0"):
        np.testing.assert_array_equal(arms[a], ref_arms[a])
    for k in ("p", "sp", "nf_cells", "nf_fit", "nf_fit_c"):
        np.testing.assert_array_equal(parts[k], ref_parts[k])
    want = bs.mixture(ref_parts["p"], ref_parts["sp"], bs.nf_hybrid(ref_parts["nf_cells"], parts["nf_fit_r"]))
    np.testing.assert_array_equal(arms["S1R"], want)
    np.testing.assert_array_equal(arms["S1R_seed0"], want)                    # one seed: the mean is the seed
    np.testing.assert_array_equal(parts["nf_fit_r"], parts["nf_fit_r_by_seed"][0])
    h = unm[te].hprox_med.to_numpy()
    nan, body = np.isnan(h), parts["nf_cells"] < bs.T_TAIL_S
    assert 20 < nan.sum() < len(h) and (~nan & body).sum() > 100
    np.testing.assert_array_equal(arms["S1R"][nan], arms["S1C"][nan])          # no witness: S1C bit-exact
    np.testing.assert_array_equal(parts["nf_fit_r"][nan], parts["nf_fit_c"][nan])
    assert np.isnan(parts["r_hat"][nan]).all() and np.isfinite(parts["r_hat"][~nan]).all()
    np.testing.assert_array_equal(parts["nf_fit_r"][~nan], (h + parts["r_hat"])[~nan])   # witness: h + r_hat
    assert (arms["S1R"][~nan & body] != arms["S1C"][~nan & body]).mean() > 0.9
    np.testing.assert_array_equal(arms["S1R"][~body], arms["S1C"][~body])      # the tail is the cells' in both arms
    assert (arms["S1R"] >= 1.0).all() and np.isfinite(arms["S1R"]).all()
    # the body recomputed independently: fitted on the non-fill rows WITH a witness, in fit_unmatched's row order
    nonfill = unm[tr][~bs.schedule_fill(unm[tr])]
    fitted = nonfill[nonfill.hprox_med.notna()]
    assert parts["n_train_nonfill"] == len(nonfill) and parts["n_train_witness"] == len(fitted) < len(nonfill)
    off = ur.ResidualRegressor(fitted, 0).predict_offset(unm[te])
    np.testing.assert_array_equal(parts["r_hat"][~nan], off[~nan])
    np.testing.assert_array_equal(parts["nf_fit_r"], np.where(nan, ref_parts["nf_fit_c_by_seed"][0], h + off))


def test_residual_body_refusals(world):
    unm, _, tr, te, _, _ = world
    fake = {0: np.zeros(int(te.sum()))}
    with pytest.raises(ValueError, match="carries seeds \\[0\\], the body needs \\(0, 1\\)"):
        ur.residual_body(unm[tr], unm[te], fake, seeds=(0, 1))
    with pytest.raises(ValueError, match="witness column 'hprox_med' missing"):
        ur.residual_body(unm[tr].drop(columns=["hprox_med"]), unm[te], fake, seeds=(0,))
    with pytest.raises(ValueError, match="shape mismatch"):
        ur.residual_body(unm[tr], unm[te], {0: np.zeros(3)}, seeds=(0,))


def test_shipped_modules_are_untouched():
    """The prereg composes the shipped code and E3C's harness; it never edits them. build_submission / stratum_fold are
    tracked (git diff); unm_congestion / unm_congestion_ship are untracked, so their content is hashed."""
    assert bs.NF_NUMERIC == ["sp", "dayoff", "hr"] and bs.WINSOR_S == 3_000.0 and bs.T_TAIL_S == 3_000.0
    assert uc.CongestionRegressor.NUMERIC == ["sp", "dayoff", "hr", "hprox_med", "hprox_n"] and uc.C2_MIN_MSE == 500.0
    assert ur.ResidualRegressor.NUMERIC == uc.CongestionRegressor.NUMERIC
    for rel, want in E3C_HASHES.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == want, f"{rel} changed since E3R was written"
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "scripts/build_submission.py", "scripts/stratum_fold.py"],
                       cwd=ROOT, capture_output=True)
    if r.returncode not in (0, 1):
        pytest.skip("git unavailable")
    assert r.returncode == 0, "scripts/build_submission.py or scripts/stratum_fold.py differs from HEAD"


# =============================================================================================
# LOMO and the reproduction guard
# =============================================================================================

def test_lomo_scores_every_row_once_and_s1c_reproduces_e3c(lomo_preds):
    """Rehearsed 2026-09-10 RED: the `bad` check dropped from check_reproduction (the perturbed reference passes);
    GUARD_COLUMNS reduced to ("S1C",) (a perturbed S1 passes)."""
    small, preds, ref = lomo_preds
    assert len(preds) == len(small) and preds.MVT_ID_mvt.is_unique and set(preds.MVT_ID_mvt) == set(small.MVT_ID_mvt)
    assert list(preds.columns) == ur.PRED_COLUMNS and (preds.fold == "lomo").all()
    assert ur.PRED_COLUMNS[:len(uc.PRED_COLUMNS)] == uc.PRED_COLUMNS
    got = ur.check_reproduction(preds, ref)
    assert got["status"] == "reproduced" and got["n_rows"] == len(small) and got["columns"] == list(ur.GUARD_COLUMNS)
    assert got["max_abs_diff_s"] == {c: 0.0 for c in ("S1C", "S1", "S1C_seed0", "S1C_seed1", "S1C_seed2")}
    for col in ("S1C", "S1", "S1C_seed2"):
        bad = ref.copy()
        bad.loc[5, col] += 1e-5
        with pytest.raises(AssertionError, match=f"{col} does not reproduce E3C's stored {col} on 1 of"):
            ur.check_reproduction(preds, bad)
    with pytest.raises(AssertionError, match="different row sets"):
        ur.check_reproduction(preds, ref.iloc[1:])
    with pytest.raises(AssertionError, match="duplicate MVT_ID"):
        ur.check_reproduction(preds, pd.concat([ref, ref.iloc[:1]], ignore_index=True))
    with pytest.raises(AssertionError, match="no column 'S1C_seed1'"):
        ur.check_reproduction(preds, ref.drop(columns=["S1C_seed1"]))
    # BC-2: the reference's fold-A rows (same ids) are ignored
    fold_a = ref.copy()
    fold_a["fold"], fold_a["S1C"] = "A", fold_a.S1C + 50.0
    ur.check_reproduction(preds, pd.concat([ref, fold_a], ignore_index=True))


def test_lomo_row_columns_carry_the_residual_parts(lomo_preds):
    """Rehearsed 2026-09-10 RED: `r_hat` stored from `parts["nf_fit_r"]` (the h + r_hat identity fails)."""
    small, preds, _ = lomo_preds
    p = preds.set_index("MVT_ID_mvt").loc[small.MVT_ID_mvt.to_numpy()]
    h = p.hprox_med.to_numpy()
    nan = np.isnan(h)
    np.testing.assert_array_equal(h, small.hprox_med.to_numpy())
    np.testing.assert_array_equal(p.S1R.to_numpy(), bs.mixture(p.p_hat.to_numpy(), p.sp.to_numpy(), bs.nf_hybrid(p.nf_cells.to_numpy(), p.nf_fit_r.to_numpy())))
    np.testing.assert_array_equal(p.S1C.to_numpy(), bs.mixture(p.p_hat.to_numpy(), p.sp.to_numpy(), bs.nf_hybrid(p.nf_cells.to_numpy(), p.nf_fit_c.to_numpy())))
    # nf_fit_r is the prereg's mean over seeds of (h + r_hat_s); the stored r_hat is mean(r_hat_s): h + mean(r_hat_s) differs
    # from mean(h + r_hat_s) by fp64 associativity only (observed <= 5e-13 s on values ~3,000 s); 1e-9 s is far above that
    # and far below anything the clauses can see
    np.testing.assert_allclose(p.nf_fit_r.to_numpy()[~nan], (h + p.r_hat.to_numpy())[~nan], rtol=0.0, atol=1e-9)
    np.testing.assert_array_equal(p.nf_fit_r.to_numpy()[nan], p.nf_fit_c.to_numpy()[nan])
    np.testing.assert_array_equal(p.S1R.to_numpy()[nan], p.S1C.to_numpy()[nan])
    assert nan.sum() > 0 and np.isnan(p.r_hat.to_numpy()[nan]).all()
    assert (p.S1R >= 1.0).all() and np.isfinite(p.S1R).all() and {f"S1R_seed{s}" for s in SEEDS} <= set(p.columns)
    with pytest.raises(ValueError, match="registered seeds"):
        ur.score_lomo(small, small.iloc[:0], (0,), lambda s: None)


def test_lomo_parquet_round_trips_with_its_convention_and_a_reader_refuses_an_unstamped_file(lomo_preds, tmp_path):
    """Rehearsed 2026-09-10 RED: `write_lomo` stamping convention 'delta' (the reader refuses its own file)."""
    _, preds, _ = lomo_preds
    path = tmp_path / "lomo.parquet"
    ur.write_lomo(preds, path, {"reproduction": {"status": "reproduced"}})
    back, meta = ur.read_lomo(path)
    pd.testing.assert_frame_equal(back, preds[ur.PRED_COLUMNS])
    assert meta["reproduction"]["status"] == "reproduced"
    schema_meta = pq.read_schema(path).metadata
    assert schema_meta[b"convention"] == b"taxi_time" and schema_meta[b"tag"] == b"E3R" and schema_meta[b"prereg"] == ur.PREREG.encode()
    bare = tmp_path / "bare.parquet"
    pq.write_table(pa.Table.from_pandas(preds, preserve_index=False), bare)
    with pytest.raises(ValueError, match="carries convention ''"):
        ur.read_lomo(bare)


def test_load_reference_refuses_a_reference_whose_own_guard_did_not_pass(lomo_preds, tmp_path, monkeypatch):
    """Rehearsed 2026-09-10 RED: the status check deleted from load_reference (a 'skipped' reference is accepted)."""
    _, _, ref = lomo_preds
    ok = tmp_path / "ok.parquet"
    uc.write_lomo(ref, ok, {"reproduction": {"status": "reproduced"}})
    monkeypatch.setattr(ur, "REFERENCE", ok)
    pd.testing.assert_frame_equal(ur.load_reference(), ref[uc.PRED_COLUMNS])
    skipped = tmp_path / "skipped.parquet"
    uc.write_lomo(ref, skipped, {"reproduction": {"status": "skipped (synthetic)"}})
    monkeypatch.setattr(ur, "REFERENCE", skipped)
    with pytest.raises(AssertionError, match="reproduction status 'skipped \\(synthetic\\)'"):
        ur.load_reference()


# =============================================================================================
# the event-excluded C2 on a hand-built table
# =============================================================================================

#: exact encodings of g = a^2 - c^2 (S1C = y + a, S1R = y + c), integers so every sum is exact in fp64.
_ENC = {1_000.0: (35.0, 15.0), 2_000.0: (45.0, 5.0), 10_000.0: (100.0, 0.0), -300_000.0: (50.0, 550.0), 100_000.0: (350.0, 150.0),
        40.0: (7.0, 3.0), 60.0: (8.0, 2.0), -5.0: (2.0, 3.0), -12.0: (2.0, 4.0), 30.0: (6.5, 3.5), 26.0: (7.5, 5.5), -6.0: (0.5, 2.5)}


def _event_rows() -> pd.DataFrame:
    """Five airport-months, one bin each:
      EHAM-1: 5 rows, h 1,700 (1500-2000), g -300,000 each -> sum -1.5e6 (the largest |sum g|, NEGATIVE);
      LTFM-2: 10 rows, h 2,500 (>=2400), g +100,000 each -> +1.0e6 (a thin bin: 10 < 30);
      EDDF-3: 29 rows, h 1,800 (1500-2000), g +10,000 each -> +2.9e5;
      LFPG-4: 30 rows, h 1,400 (1300-1500), g +2,000 each -> +6e4;
      EGLL-5: 40 rows, h 500 (<900), g +1,000 each -> +4e4.
    Full table: 1500-2000 has 34 rows (mean -35,588.2); with EHAM-1 dropped it has 29 (thin -> borrows 1300-1500 = 2,000)."""
    spec = [("EHAM", 1, 1_700.0, -300_000.0, 5), ("LTFM", 2, 2_500.0, 100_000.0, 10), ("EDDF", 3, 1_800.0, 10_000.0, 29),
            ("LFPG", 4, 1_400.0, 2_000.0, 30), ("EGLL", 5, 500.0, 1_000.0, 40)]
    rows = []
    for ap, m, h, g, n in spec:
        a, c = _ENC[g]
        for i in range(n):
            rows.append(dict(month=m, date=f"2025-{m:02d}-{1 + i % 28:02d}", ADEP_mvt=ap, y=1_000.0, hprox_med=h, S1C=1_000.0 + a, S1R=1_000.0 + c))
    t = pd.DataFrame(rows)
    for s in SEEDS:
        t[f"S1C_seed{s}"], t[f"S1R_seed{s}"] = t.S1C, t.S1R
    return t


#: 2026 counts chosen so the event-excluded price is EXACTLY 300 at n_scored 4,000:
#: 1,000 x 100 + 2,000 x (50 + 50 + 450) = 1.2e6; / 4,000 = 300.0 (every term exact). The NaN bin is priced 0 (thin).
_COUNTS_26 = {"NaN": 999, "<900": 100, "900-1100": 0, "1100-1300": 0, "1300-1500": 50, "1500-2000": 50, "2000-2400": 0, ">=2400": 450}


def test_event_table_is_what_its_docstring_says():
    t = _event_rows()
    g = ur.gains(t)
    assert len(t) == 114 and g.sum() == -1_500_000.0 + 1_000_000.0 + 290_000.0 + 60_000.0 + 40_000.0
    am = ur.airport_month_table(g, t.ADEP_mvt, t.month)
    assert am[["ADEP_mvt", "month", "sum_g", "n"]].to_dict(orient="records") == [
        {"ADEP_mvt": "EHAM", "month": 1, "sum_g": -1_500_000.0, "n": 5}, {"ADEP_mvt": "LTFM", "month": 2, "sum_g": 1_000_000.0, "n": 10},
        {"ADEP_mvt": "EDDF", "month": 3, "sum_g": 290_000.0, "n": 29}, {"ADEP_mvt": "LFPG", "month": 4, "sum_g": 60_000.0, "n": 30},
        {"ADEP_mvt": "EGLL", "month": 5, "sum_g": 40_000.0, "n": 40}]
    assert {k: v["n_2025"] for k, v in uc.bin_table(g, t.hprox_med).items()} == {"NaN": 0, "<900": 40, "900-1100": 0, "1100-1300": 0,
                                                                                 "1300-1500": 30, "1500-2000": 34, "2000-2400": 0, ">=2400": 10}


def test_largest_event_is_by_absolute_sum_with_a_deterministic_tie_break():
    """Rehearsed 2026-09-10 RED: `abs_sum_g` sorted by `sum_g` (the positive LTFM-2 is dropped instead of the negative
    EHAM-1); the tie-break order reversed (ties resolve to the LAST airport)."""
    t = _event_rows()
    assert ur.largest_event(ur.gains(t), t.ADEP_mvt, t.month) == {"ADEP_mvt": "EHAM", "month": 1, "sum_g": -1_500_000.0, "n": 5}
    # ties: three airport-months at |sum g| = 10 -> airport first, then month
    g = np.array([10.0, -10.0, 10.0, 3.0])
    assert ur.largest_event(g, ["LTFM", "EDDF", "EDDF", "EHAM"], [3, 9, 2, 1]) == {"ADEP_mvt": "EDDF", "month": 2, "sum_g": 10.0, "n": 1}
    # two rows of +10 / -10 in EDDF-9 sum to 0; the single LTFM-3 row at +10 then wins
    assert ur.largest_event(g, ["LTFM", "EDDF", "EDDF", "EHAM"], [3, 9, 9, 1]) == {"ADEP_mvt": "LTFM", "month": 3, "sum_g": 10.0, "n": 1}
    with pytest.raises(ValueError, match="no airport-month to exclude"):
        ur.largest_event(np.zeros(0), [], [])


def test_event_excluded_price_rebins_borrows_and_passes_at_exactly_300():
    """Rehearsed 2026-09-10 RED: the exclusion mask inverted (`keep = ...` without the `~`: only EHAM-1 is priced);
    `>= C2_MIN_MSE` -> `>` (300.0 exactly fails); C2_MIN_MSE 300 -> 500 (the parent's bar: 300.0 fails);
    the price computed on the FULL table (1500-2000 is not thin, ≥2400 borrows -35,588 and the total is negative)."""
    t = _event_rows()
    g, h = ur.gains(t), t.hprox_med.to_numpy()
    p = ur.event_excluded_price(g, h, t.ADEP_mvt, t.month, _COUNTS_26, n_scored=4_000)
    assert p["excluded_airport_month"] == {"ADEP_mvt": "EHAM", "month": 1, "sum_g": -1_500_000.0, "n": 5, "n_rows_kept": 109,
                                           "n_rows_dropped": 5, "sum_g_kept": 1_390_000.0}
    assert {k: v["n_2025"] for k, v in p["bins"].items()} == {"NaN": 0, "<900": 40, "900-1100": 0, "1100-1300": 0, "1300-1500": 30,
                                                              "1500-2000": 29, "2000-2400": 0, ">=2400": 10}
    assert {k: v["per_row_gain_used"] for k, v in p["bins"].items()} == {"NaN": 0.0, "<900": 1_000.0, "900-1100": 1_000.0, "1100-1300": 1_000.0,
                                                                         "1300-1500": 2_000.0, "1500-2000": 2_000.0, "2000-2400": 2_000.0,
                                                                         ">=2400": 2_000.0}
    assert {k: v["borrowed_from"] for k, v in p["bins"].items()} == {"NaN": None, "<900": "<900", "900-1100": "<900", "1100-1300": "<900",
                                                                     "1300-1500": "1300-1500", "1500-2000": "1300-1500", "2000-2400": "1300-1500",
                                                                     ">=2400": "1300-1500"}
    assert p["bins"]["1500-2000"]["thin"] and p["bins"]["1500-2000"]["per_row_gain"] == 10_000.0    # its own 29 rows are not used
    assert p["priced_mse"] == 300.0 and p["passes"] and p["c2_min_mse"] == 300.0 and p["n_scored_2026"] == 4_000
    assert p["rule"].startswith("event-excluded")
    q = ur.event_excluded_price(g, h, t.ADEP_mvt, t.month, _COUNTS_26, n_scored=4_001)
    assert q["priced_mse"] < 300.0 and not q["passes"]
    # the full (non-excluded) price on the same table is negative: 1500-2000 keeps its 34 rows and ≥2400 borrows their mean
    full = uc.price_2026(uc.bin_table(g, h), _COUNTS_26, n_scored=4_000)
    assert full["bins"]["1500-2000"]["per_row_gain"] == pytest.approx(-1_210_000.0 / 34, rel=1e-12) and not full["bins"]["1500-2000"]["thin"]
    assert full["bins"][">=2400"]["borrowed_from"] == "1500-2000" and full["priced_mse"] < 0.0


def test_event_excluded_price_uses_the_registered_denominator_by_default():
    """Rehearsed 2026-09-10 RED: N_SCORED_2026 default -> 344,842 (per-row 300 x 344,841 rows / 344,842 < 300)."""
    g, h = np.full(30, 300.0), np.full(30, 2_500.0)
    ap, mo = np.array(["EHAM"] * 15 + ["LFPG"] * 15), np.array([1] * 15 + [2] * 15)
    p = ur.event_excluded_price(g, h, ap, mo, {">=2400": 344_841})
    # EHAM-1 (+4,500) ties LFPG-2 (+4,500): EHAM first; 15 rows remain in ≥2400 -> thin -> borrow chain ends at 0
    assert p["excluded_airport_month"]["ADEP_mvt"] == "EHAM" and p["n_scored_2026"] == 344_841 and p["priced_mse"] == 0.0
    g2 = np.concatenate([np.full(30, 300.0), np.full(31, 1.0)])
    h2 = np.concatenate([np.full(30, 2_500.0), np.full(31, 1_000.0)])
    ap2, mo2 = np.array(["EHAM"] * 30 + ["LFPG"] * 31), np.array([1] * 30 + [2] * 31)
    p2 = ur.event_excluded_price(g2, h2, ap2, mo2, {">=2400": 344_841})          # EHAM-1 (+9,000) dropped; ≥2400 empty -> borrows 900-1100 = 1
    assert p2["excluded_airport_month"]["ADEP_mvt"] == "EHAM" and p2["priced_mse"] == 1.0 and not p2["passes"]
    ap3, mo3 = np.array(["EHAM"] * 30 + ["LFPG"] * 31), np.array([1] * 30 + [2] * 31)
    g3 = np.concatenate([np.full(30, 300.0), np.full(31, 1_000.0)])             # now LFPG-2 (+31,000) is the event; ≥2400 keeps its 30 rows
    p3 = ur.event_excluded_price(g3, h2, ap3, mo3, {">=2400": 344_841})
    assert p3["excluded_airport_month"]["ADEP_mvt"] == "LFPG" and p3["priced_mse"] == 300.0 and p3["passes"]


# =============================================================================================
# the seven clauses on E3C's boundary table, renamed
# =============================================================================================

_APS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LSZH", "LTFM"]


def _panel(override: dict | None = None) -> pd.DataFrame:
    """E3C's 36-row boundary table with E3R's names: 3 rows per month (NaN / calm 800 / hot 2,500), airports cycling,
    one date per month; pooled +634; 8 of 12 months and 6 of 9 airports positive; calm -60 vs bound -63.4; ex-monster
    +604; per-seed arms equal the pooled ones. Every airport-month is one row; the largest |g| is 60 (eight ties) ->
    the tie-break picks EGLL month 1."""
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
                             S1C=y + a, S1R=y + c))
            k += 1
    t = pd.DataFrame(rows)
    for s in SEEDS:
        t[f"S1C_seed{s}"], t[f"S1R_seed{s}"] = t.S1C, t.S1R
    return t


def test_clauses_c3_to_c7_are_e3c_arithmetic_on_the_renamed_view_and_c2_is_the_excluded_price():
    """Rehearsed 2026-09-10 RED: the view mapping swapped (S1 := S1R, S1C := S1C -> the view's gains are negated and
    clauses() raises); C2 left as the full price (the record's C2 lacks the excluded airport-month; here the full price
    on 100-count bins is +100 x ... and the key sets differ)."""
    t = _panel()
    counts = {b: 100 for b in uc.BINS}
    c = ur.clauses(t, counts, n_boot=200, seed=0, n_scored=1_000)
    assert c["pooled_sum_g"] == 634.0 and c["n_rows"] == 36
    c3, c4, c5, c6, c7 = c["C3_months"], c["C4_airports"], c["C5_calm_control"], c["C6_not_the_monsters"], c["C7_seeds"]
    assert c3["months_positive"] == 8 and c3["n_months"] == 12 and c3["passes"] and c3["per_month"]["1"] == {"sum_g": 85.0, "n": 3}
    assert c4["airports_positive"] == 6 and c4["n_airports"] == 9 and c4["passes"] and c4["per_airport"]["LSZH"] == {"sum_g": -20.0, "n": 4}
    assert c5["calm_sum_g"] == -60.0 and c5["n_calm"] == 12 and c5["bound"] == pytest.approx(-63.4, abs=1e-12) and c5["passes"]  # 0.1 x 634 inexact
    assert c6["exmonster_sum_g"] == 604.0 and c6["n_exmonster"] == 35 and c6["passes"]
    assert c7["seed_sd"] == 0.0 and c7["passes"] and c7["per_seed_sum_g"] == {"0": 634.0, "1": 634.0, "2": 634.0}
    c1 = c["C1_date_block_bootstrap"]
    assert c1["n_blocks"] == 12 and c1["n_draws"] == 200 and c1["seed"] == 0 and c1["sum_g"] == 634.0
    # C2: EGLL month 1 (g +60, the first of eight ties) is dropped; every bin is then thin -> priced 0 -> fails
    c2 = c["C2_priced_2026"]
    assert c2["excluded_airport_month"] == {"ADEP_mvt": "EGLL", "month": 1, "sum_g": 60.0, "n": 1, "n_rows_kept": 35, "n_rows_dropped": 1,
                                            "sum_g_kept": 574.0}
    assert c2["priced_mse"] == 0.0 and not c2["passes"] and c2["c2_min_mse"] == 300.0
    assert {k: v["n_2025"] for k, v in c2["bins"].items()} == {"NaN": 12, "<900": 12, "900-1100": 0, "1100-1300": 0, "1300-1500": 0,
                                                               "1500-2000": 0, "2000-2400": 0, ">=2400": 11}
    full = c["C2_full_price_not_decisional"]
    assert "passes" not in full and full["rule"].startswith("full price") and full["bins"][">=2400"]["n_2025"] == 12
    want_full = uc.price_2026(uc.bin_table(ur.gains(t), t.hprox_med), counts, 1_000)
    want_full.pop("passes")
    assert full == {**want_full, "rule": full["rule"]}
    assert c["verdict"] == ("INCONCLUSIVE" if c1["passes"] else "NOT WORKING") and c["verdict"] == uc.verdict(c)
    assert [r["ADEP_mvt"] for r in c["airport_month_top"][:3]] == ["EGLL", "EGLL", "EGLL"] and len(c["airport_month_top"]) == 12
    # the view: control in the S1 slot, treatment in the S1C slot, and its gains are E3R's
    v = ur.e3c_view(t)
    assert list(v.columns) == ["y", "month", "date", "ADEP_mvt", "hprox_med", "S1", "S1C", "S1_seed0", "S1C_seed0", "S1_seed1", "S1C_seed1",
                               "S1_seed2", "S1C_seed2"]
    np.testing.assert_array_equal(v.S1.to_numpy(), t.S1C.to_numpy())
    np.testing.assert_array_equal(v.S1C.to_numpy(), t.S1R.to_numpy())
    np.testing.assert_array_equal(uc.gains(v), ur.gains(t))
    with pytest.raises(ValueError, match="non-LIRF rows only"):
        ur.clauses(t.assign(ADEP_mvt=np.where(t.index == 3, "LIRF", t.ADEP_mvt)), counts, n_boot=200)


def test_gains_are_control_minus_treatment_on_the_raw_label():
    """Rehearsed 2026-09-10 RED: CONTROL and TREATMENT swapped in gains() (the sign flips)."""
    t = pd.DataFrame({"y": [1_000.0, 20_000.0], "S1C": [1_100.0, 20_500.0], "S1R": [1_050.0, 21_000.0]})
    np.testing.assert_array_equal(ur.gains(t), [100.0 ** 2 - 50.0 ** 2, 500.0 ** 2 - 1_000.0 ** 2])


def test_verdict_mapping_with_the_decisional_c2():
    """Rehearsed 2026-09-10 RED: in clauses(), the verdict computed BEFORE C2 is replaced by the event-excluded price
    (a table whose full price clears 500 but whose excluded price is 0 reads WORKING)."""
    keys = ["C1_date_block_bootstrap", "C2_priced_2026", "C3_months", "C4_airports", "C5_calm_control", "C6_not_the_monsters", "C7_seeds"]
    ok = {k: {"passes": True} for k in keys}
    assert ur.verdict(ok) == "WORKING"
    for k in keys[1:]:
        assert ur.verdict({kk: {"passes": kk != k} for kk in keys}) == "INCONCLUSIVE", k
    assert ur.verdict({**ok, "C1_date_block_bootstrap": {"passes": False}}) == "NOT WORKING"
    assert ur.verdict({**ok, "C1_date_block_bootstrap": {"passes": False}, "C7_seeds": {"passes": False}}) == "NOT WORKING"
    # a table whose FULL price clears the parent's 500 but whose event-excluded price does not clear 300: not WORKING.
    # Every (airport, month) has one calm row (h 800, g +1); LTFM month 2 adds 100 hot rows (h 2,500, g +60): full price
    # 60 x 100,000 / 1,000 = 6,000; with LTFM-2 dropped the >=2400 bin is empty and borrows the calm +1 -> 100.
    rows = [dict(month=m, date=f"2025-{m:02d}-15", ADEP_mvt=ap, y=1_000.0, hprox_med=800.0, S1C=1_001.0, S1R=1_000.0)
            for m in range(1, 13) for ap in _APS]
    rows += [dict(month=2, date="2025-02-15", ADEP_mvt="LTFM", y=1_000.0, hprox_med=2_500.0, S1C=1_008.0, S1R=1_002.0) for _ in range(100)]
    t = pd.DataFrame(rows)
    for s in SEEDS:
        t[f"S1C_seed{s}"], t[f"S1R_seed{s}"] = t.S1C, t.S1R
    counts = {">=2400": 100_000, **{b: 0 for b in uc.BINS if b != ">=2400"}}
    c = ur.clauses(t, counts, n_boot=200, seed=0, n_scored=1_000)
    assert c["pooled_sum_g"] == 108.0 + 6_000.0
    assert c["C2_full_price_not_decisional"]["priced_mse"] == 6_000.0 >= uc.C2_MIN_MSE
    c2 = c["C2_priced_2026"]
    assert c2["excluded_airport_month"] == {"ADEP_mvt": "LTFM", "month": 2, "sum_g": 6_001.0, "n": 101, "n_rows_kept": 107,
                                            "n_rows_dropped": 101, "sum_g_kept": 107.0}
    assert c2["bins"][">=2400"]["n_2025"] == 0 and c2["bins"][">=2400"]["borrowed_from"] == "<900" and c2["priced_mse"] == 100.0 and not c2["passes"]
    for k in ("C1_date_block_bootstrap", "C3_months", "C4_airports", "C5_calm_control", "C6_not_the_monsters", "C7_seeds"):
        assert c[k]["passes"], k
    assert c["verdict"] == "INCONCLUSIVE"


def test_bin_means_and_rmse_arms_on_the_panel():
    """Rehearsed 2026-09-10 RED: bin_means' mean_S1R read from the S1C column (the two means coincide)."""
    t = _panel()
    b = ur.bin_means(t)
    assert set(b) == set(uc.BINS) and b["NaN"]["n"] == 12 and b["<900"]["n"] == 12 and b[">=2400"]["n"] == 12 and b["900-1100"]["n"] == 0
    assert b["NaN"]["mean_hprox_med"] is None and b["<900"]["mean_hprox_med"] == 800.0 and b[">=2400"]["mean_hprox_med"] == 2_500.0
    assert b["<900"]["mean_S1C"] == 1_002.0 and b["<900"]["mean_S1R"] == 1_003.0 and b["<900"]["sum_g"] == -60.0 and b["<900"]["mean_y"] == 1_000.0
    hot_a = np.mean([8.0] * 8 + [2.0] * 4)
    hot_c = np.mean([2.0] * 8 + [4.0] * 4)
    assert b[">=2400"]["mean_S1C"] == pytest.approx(1_000.0 + hot_a, rel=1e-12) and b[">=2400"]["mean_S1R"] == pytest.approx(1_000.0 + hot_c, rel=1e-12)
    assert b[">=2400"]["sum_g"] == 8 * 60.0 - 4 * 12.0 and b["900-1100"]["mean_y"] is None
    r = ur.rmse_arms(t)
    assert r["n"] == 36 and r["S1C"] == pytest.approx(np.sqrt(((t.S1C - t.y) ** 2).mean()), rel=1e-12)
    assert r["S1R"] == pytest.approx(np.sqrt(((t.S1R - t.y) ** 2).mean()), rel=1e-12) and r["gain_s"] == r["S1C"] - r["S1R"]
    assert ur.rmse_arms(t, np.zeros(36, dtype=bool)) == {"n": 0, "S1C": None, "S1R": None, "gain_s": None}


def test_check_counts_match_e3c():
    """Rehearsed 2026-09-10 RED: the inequality dropped (a different stored table is accepted)."""
    bins = np.array(["<900"] * 3 + [">=2400"] * 2 + ["NaN"], dtype=object)
    counts = uc.bin_counts(bins)
    assert ur.check_counts_match_e3c(counts, bins) == counts
    with pytest.raises(AssertionError, match="differ from E3C's stored"):
        ur.check_counts_match_e3c(counts, bins[:-1])


def test_clauses_end_to_end_on_the_synthetic_world_carry_a_verdict_and_every_table(lomo_preds):
    small, preds, _ = lomo_preds
    non = preds[preds.ADEP_mvt != "LIRF"].reset_index(drop=True)
    counts = uc.bin_counts(uc.bin_of(non.hprox_med))
    c = ur.clauses(non, counts, n_boot=50, seed=0, n_scored=len(non))
    assert c["verdict"] in ("WORKING", "NOT WORKING", "INCONCLUSIVE") and c["verdict"] == uc.verdict(c)
    ev = c["C2_priced_2026"]["excluded_airport_month"]
    assert ev["n_rows_kept"] + ev["n_rows_dropped"] == len(non) and ev["ADEP_mvt"] in set(non.ADEP_mvt) and ev["month"] in (1, 2, 3)
    assert sum(v["n_2025"] for v in c["C2_priced_2026"]["bins"].values()) == ev["n_rows_kept"]
    assert sum(v["n_2025"] for v in c["C2_full_price_not_decisional"]["bins"].values()) == len(non)
    assert c["C3_months"]["n_months"] == 3 and not c["C3_months"]["passes"]      # three months can never clear 8 of 12
    assert c["C1_date_block_bootstrap"]["n_blocks"] == non.date.nunique()
    b = ur.bin_means(non)
    assert sum(v["n"] for v in b.values()) == len(non) and sum(v["sum_g"] for v in b.values()) == pytest.approx(c["pooled_sum_g"], rel=1e-9)


# =============================================================================================
# smoke conventions and paths
# =============================================================================================

def test_smoke_paths_never_touch_the_real_outputs_nor_e3cs():
    smoke, real = ur.output_paths(True), ur.output_paths(False)
    assert set(smoke) == set(real) == {"lomo", "json", "lomo_log", "score_log"}
    assert all(p.parent == ROOT / "data" / "smoke_unm_congestion_resid" for p in smoke.values())
    assert real["lomo"] == ROOT / "data" / "cache_stand" / "unm_congestion_resid_lomo.parquet"
    assert real["json"] == ROOT / "reports" / "unm_congestion_resid.json"
    e3c = set(uc.output_paths(False).values()) | set(uc.output_paths(True).values())
    assert not (set(smoke.values()) & set(real.values())) and not ((set(smoke.values()) | set(real.values())) & e3c)
    assert ur.REFERENCE == ROOT / "data" / "cache_stand" / "unm_congestion_lomo.parquet" == uc.output_paths(False)["lomo"]
    assert ur.E3C_WITNESS_2026 == uc.output_paths(False)["witness_2026"]
    for cmd in (ur.COMMAND_LOMO, ur.COMMAND_SCORE):
        assert "OMP_NUM_THREADS=1 nice -n 19 /opt/homebrew/bin/python3.11 -B -u scripts/unm_congestion_resid.py" in cmd and 'echo "exit $?"' in cmd


def test_registered_constants_are_the_preregs():
    """Rehearsed 2026-09-10 RED: C2_MIN_MSE 300 -> 500."""
    assert ur.C2_MIN_MSE == 300.0 and ur.CLIP_S == 3_000.0 and ur.SEEDS == (0, 1, 2) and ur.N_BOOT == 2_000 and ur.BOOT_SEED == 0
    assert ur.N_SCORED_2026 == 344_841 and ur.CONTROL == "S1C" and ur.TREATMENT == "S1R" and ur.CONVENTION == "taxi_time"
    assert ur.GUARD_COLUMNS == ("S1C", "S1", "S1C_seed0", "S1C_seed1", "S1C_seed2") and ur.PREREG.endswith("PREREG_unm_congestion_resid_2026_09_10.md")
