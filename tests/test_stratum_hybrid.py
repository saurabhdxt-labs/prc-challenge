"""Amendment 18 (plans/PREREG_taxiout_2026_09_08.md): the v7 stratum hybrid — a fitted
non-fill regressor for the BODY of the unmatched stratum with the cell estimator's tail load
preserved — and its 12-fold LOMO harness `scripts/stratum_fold.py`.

What the tests pin, and why each exists:
  * the DEFAULT `fit_unmatched` path is byte-identical to a golden computed from the PRISTINE
    (pre-hybrid) module — v2/v3's 5,290 unmatched predictions must stay reproducible;
  * `nf_hybrid` routes on `nf_cells < T_tail` strictly, `nf_fit` below and `nf_cells` at or
    above, so the tail load the cells carry to 80,989 s on LIRF is never replaced by a
    winsorised prediction;
  * seed averaging is the arithmetic mean; winsorisation touches the FIT target only, never
    scoring; the encodings and the regressor are fitted on NON-FILL training rows only;
  * the monster mask is `y > 10,800` strictly; LOMO leaves exactly one month out and never
    scores a training row; the month-block bootstrap is paired by month, finds a planted
    10-sigma gain and rejects noise (this repo's falsify rule for any harness);
  * `--smoke --synthetic` runs end to end on a planted structure where the hybrid must beat
    the cells ex-monster and must NOT lose pooled, and prints the 18.3 clauses with numbers
    and no verdict word.
"""
from __future__ import annotations

import hashlib
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bs = _load("build_submission")

AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
EPOCH_2025 = pd.Timestamp("2025-01-01", tz="UTC")
_S0 = (EPOCH_2025 - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds()


# =============================================================================================
# the frozen golden frame — the default-path golden was computed on it from the PRISTINE module
# =============================================================================================

def _raw_block(rng, n, matched: bool) -> pd.DataFrame:
    """Raw movement columns in the shape `derive()` expects. Unmatched rows have every NM clock
    and every `*_flt` column null together, exactly as the scored file does (Amendment 3 §2);
    matched rows are LIRF only, with the schedule-copy pattern the L-e logistic reads."""
    ap = np.full(n, "LIRF") if matched else rng.choice(AIRPORTS, n)
    airline = rng.choice(["RYR", "ITY", "AEZ", "BAW", "DLH", "AFR"], n)
    sched_s = _S0 + rng.integers(0, 365 * 86400, n).astype(float)
    p_fill = np.where(ap == "LIRF", np.where(airline == "ITY", 0.7, np.where(airline == "RYR", 0.2, 0.45)), 0.03)
    is_fill = rng.random(n) < p_fill
    taxi = rng.gamma(4, 220, n) + 200
    late = rng.gamma(1.5, 1800, n)
    block_s = np.where(is_fill, sched_s, sched_s + late)
    mvt_s = np.where(is_fill, sched_s + late + taxi, block_s + taxi)
    df = pd.DataFrame({
        "PHASE_mvt": "DEP",
        "MVT_ID_mvt": np.arange(n, dtype="float64") + (1e6 if matched else 0.0),
        "ADEP_mvt": ap,
        "ADES_mvt": rng.choice(["EGLL", "LFPG", "LEMD", "KJFK", "OMDB", "EDDF"], n),
        "STAND_mvt": rng.choice([f"{c}{i:02d}" for c in "ABCDEF" for i in range(1, 31)], n),
        "RUNWAY_mvt": rng.choice(["09L", "27R", "16", "34", "07L", "25R"], n),
        "AIRCRAFT_TYPE_mvt": rng.choice(["A320", "B738", "A21N"], n),
        "FLIGHT_mvt": [f"{a}{k}" for a, k in zip(airline, rng.integers(100, 9999, n))],
        "FLIGHT_RULE_mvt": "I",
        "MVT_TIME_UTC_mvt": pd.to_datetime(mvt_s, unit="s", utc=True),
        "SCHED_TIME_UTC_mvt": pd.to_datetime(sched_s, unit="s", utc=True),
        "BLOCK_TIME_UTC_mvt": pd.to_datetime(block_s, unit="s", utc=True),
        "TAXITIME_SEC_mvt": np.rint(mvt_s - block_s).astype("int32"),
    })
    for c in ("AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt"):
        df[c] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for c in ("AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt"):
        df[c] = pd.Series([None] * n, index=df.index, dtype="str")
    if matched:
        # a copy row stamps SCHED into off-block while NM disagrees by > 300 s; a genuine row
        # has NM within a minute of the airport clock
        aobt_s = block_s + np.where(is_fill, 600.0, rng.normal(0.0, 60.0, n))
        df["AOBT_3_flt"] = pd.to_datetime(aobt_s, unit="s", utc=True)
        df["AIRCRAFT_OPERATOR_flt"] = pd.Series(airline, index=df.index, dtype="str")
        df["MARKET_SEGMENT_flt"] = pd.Series(["S"] * n, index=df.index, dtype="str")
        df["WK_TBL_CAT_flt"] = pd.Series(["M"] * n, index=df.index, dtype="str")
        df["FLIGHT_TYPE_flt"] = pd.Series(["S"] * n, index=df.index, dtype="str")
    return df


def _golden_frame():
    """FROZEN 2026-09-09. (train unmatched, test unmatched, train LIRF matched) split by
    `bs.HOLDOUT_MONTHS`, the fold the default path shipped on. Do NOT edit: a changed frame
    invalidates tests/fixtures/stratum_hybrid_default_golden.npz, which can only be regenerated
    from the PRISTINE build_submission.py (sha256 cb3a1f7f..., HEAD a570407)."""
    rng = np.random.default_rng(20260909)
    unm = bs.derive(_raw_block(rng, 9_000, matched=False))
    mat = bs.derive(_raw_block(rng, 4_000, matched=True))
    for d in (unm, mat):
        d["y"] = (d.MVT_TIME_UTC_mvt - d.BLOCK_TIME_UTC_mvt).dt.total_seconds()
        d["delta"] = (d.BLOCK_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
    unm = unm[unm.y > 0].reset_index(drop=True)
    mat = mat[mat.y > 0].reset_index(drop=True)
    assert unm.unmatched.all() and not mat.unmatched.any()
    held = unm.month.isin(bs.HOLDOUT_MONTHS)
    return (unm[~held].reset_index(drop=True), unm[held].reset_index(drop=True),
            mat[~mat.month.isin(bs.HOLDOUT_MONTHS)].reset_index(drop=True))


GOLDEN = FIXTURES / "stratum_hybrid_default_golden.npz"
#: sha256 over the concatenated float64 bytes of the two golden arrays; pinned here so a
#: regenerated fixture file cannot pass unnoticed
GOLDEN_SHA256 = "3bf82839db45abc1c2d2e7662111a6114e69476bd787b195debe685922166809"


def _sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype="float64").tobytes())
    return h.hexdigest()


def test_default_path_is_byte_identical_to_the_pristine_golden():
    """`fit_unmatched(train, test, train_matched=...)` and `fit_unmatched(train, test)` must
    return, bit for bit, what the PRISTINE module returned on the frozen frame — the hybrid is
    opt-in and the default call is the one that produced v2/v3's 5,290 unmatched rows.
    The golden was computed from build_submission.py at sha256 cb3a1f7f... (HEAD a570407)
    BEFORE the hybrid edit, under sklearn 1.8.0 / pandas 3.0.1 / numpy 2.4.3; a different
    library version that moves the LIRF logistic at the 1e-12 level is a real finding, not a
    reason to loosen this to allclose.

    Fails when the default flips to hybrid, when the mixture's arithmetic is reordered, or
    when the cell path or the LIRF model changes at all. Rehearsed 2026-09-09: `hybrid: bool = True`
    as the default went RED (every row moved); `parts = {}` after the return line (dead code)
    stayed GREEN as it must.
    """
    tr, te, mat = _golden_frame()
    with_lirf = bs.fit_unmatched(tr, te, train_matched=mat)
    cells_only = bs.fit_unmatched(tr, te)
    g = np.load(GOLDEN)
    assert _sha(g["with_lirf"], g["cells_only"]) == GOLDEN_SHA256, "the fixture file was regenerated"
    assert with_lirf.dtype == np.float64 and with_lirf.shape == g["with_lirf"].shape
    moved = int((with_lirf != g["with_lirf"]).sum())
    assert np.array_equal(with_lirf, g["with_lirf"]), f"default path moved on {moved} of {len(te)} rows"
    moved = int((cells_only != g["cells_only"]).sum())
    assert np.array_equal(cells_only, g["cells_only"]), f"cells-only path moved on {moved} of {len(te)} rows"
    assert not np.array_equal(with_lirf, cells_only), "the LIRF model did nothing on the frozen frame"


# =============================================================================================
# helpers shared by the unit tests (NOT the frozen golden frame)
# =============================================================================================

def _fill(d: pd.DataFrame) -> np.ndarray:
    return ((d.BLOCK_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60).to_numpy()


def _small_split(seed=3, n_unm=3_000, n_mat=1_500):
    """A quick (train unmatched, test unmatched, train LIRF matched) triple for unit tests that
    need real `derive()` dtypes but not the frozen frame."""
    rng = np.random.default_rng(seed)
    unm = bs.derive(_raw_block(rng, n_unm, matched=False))
    mat = bs.derive(_raw_block(rng, n_mat, matched=True))
    for d in (unm, mat):
        d["y"] = (d.MVT_TIME_UTC_mvt - d.BLOCK_TIME_UTC_mvt).dt.total_seconds()
        d["delta"] = (d.BLOCK_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
    unm = unm[unm.y > 0].reset_index(drop=True)
    mat = mat[mat.y > 0].reset_index(drop=True)
    held = unm.month.isin((1, 7))
    return (unm[~held].reset_index(drop=True), unm[held].reset_index(drop=True),
            mat[~mat.month.isin((1, 7))].reset_index(drop=True))


sf = _load("stratum_fold")


# =============================================================================================
# A. build_submission: the hybrid pieces
# =============================================================================================

def test_nf_hybrid_routes_on_nf_cells_below_t_tail_strictly_and_counts_both_branches():
    """`nf_hybrid` takes `nf_fit` where `nf_cells < T_tail` and `nf_cells` at or above it, so
    a cell that carries tail load at exactly the winsorisation point keeps that load. Hand
    values straddle the 3,000 s boundary; `routing_counts` reports both branches.

    Fails when `<` slips to `<=` (the 3,000.0 row would route to the fitted 20.0), when the
    branches are swapped, or when T_tail drifts from 3,000. Rehearsed 2026-09-09, each RED:
    `nf_cells <= T_tail`; `np.where(..., nf_cells, nf_fit)`; `T_TAIL_S = 3_001.0`.
    """
    nf_cells = np.array([2_999.999, 3_000.0, 3_000.001, 100.0, 50_000.0, 0.0])
    nf_fit = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    got = bs.nf_hybrid(nf_cells, nf_fit)
    assert got.dtype == np.float64
    assert got.tolist() == [10.0, 3_000.0, 3_000.001, 40.0, 50_000.0, 60.0]
    assert bs.nf_hybrid(nf_cells, nf_fit, T_tail=3_000.002).tolist() == [10.0, 20.0, 30.0, 40.0, 50_000.0, 60.0]
    assert bs.T_TAIL_S == 3_000.0 == bs.WINSOR_S, "T_tail is the winsorisation point, not tuned"
    assert bs.routing_counts(nf_cells) == {"n_nf_fit": 3, "n_nf_cells": 3, "T_tail": 3_000.0}
    assert bs.routing_counts(np.array([1.0, 2.0]))["n_nf_cells"] == 0
    with pytest.raises(ValueError, match="shape"):
        bs.nf_hybrid(nf_cells, nf_fit[:-1])
    with pytest.raises(ValueError, match="finite"):
        bs.nf_hybrid(nf_cells, np.array([np.nan, 20.0, 30.0, 40.0, 50.0, 60.0]))
    with pytest.raises(ValueError, match="finite"):
        bs.nf_hybrid(np.array([np.inf] + [1.0] * 5), nf_fit)


def test_mixture_is_p_sp_plus_one_minus_p_nf_floored_at_one_second():
    """The mixture the default path has always returned: `max(p * sp + (1 - p) * nf, 1)`.

    Fails when the weights are swapped or the floor is dropped. Rehearsed 2026-09-09, each
    RED: `(1 - p) * sp + p * nf`; `np.maximum(..., 0.0)`.
    """
    p, sp, nf = np.array([0.0, 1.0, 0.25, 0.5]), np.array([100.0, 100.0, 1_000.0, -50.0]), np.array([10.0, 10.0, 200.0, -50.0])
    assert bs.mixture(p, sp, nf).tolist() == [10.0, 100.0, 400.0, 1.0]


def test_mean_over_seeds_is_the_arithmetic_mean_in_seed_order():
    """Seed averaging is the plain mean over the seeds given, float64, exact on small ints.

    Fails when only the first seed is used or when the mean is over a wrong axis. Rehearsed
    2026-09-09, each RED: `return stack[0]`; `np.mean(stack, axis=1)`.
    """
    got = bs.mean_over_seeds({0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0]), 2: np.array([8.0, 12.0])})
    assert got.dtype == np.float64 and got.tolist() == [4.0, 6.0]
    assert bs.mean_over_seeds({7: np.array([5.0])}).tolist() == [5.0]
    with pytest.raises(ValueError, match="seed"):
        bs.mean_over_seeds({})


def test_regressor_is_exactly_screen_b_b2():
    """Amendment 18.1: `nf_fit` is Screen B's B2 regressor so the ex-monster number is
    comparable to RESULT 3.2 — HistGradientBoostingRegressor(max_leaf_nodes=15, max_iter=200,
    min_samples_leaf=100, random_state=seed) with sklearn's other defaults (early_stopping
    'auto', i.e. ON above 10,000 rows on a seed-dependent 10% split — that is what Screen B
    ran), on float64 [sp, dayoff, hr] + SMOOTH-smoothed target encodings of ADEP_mvt,
    RUNWAY_mvt, stand_pref, airline, AIRCRAFT_OPERATOR_flt in that order; an unseen category
    encodes to the prior exactly; predictions are float64 and unfloored.

    Fails when a hyper-parameter, the column set or its order drifts, or when an unseen level
    falls to NaN instead of the prior. Rehearsed 2026-09-09, each RED: `max_leaf_nodes=31`;
    `"stand_c1"` in NF_ENCODED; `.fillna(0.0)`.
    """
    tr, te, _ = _small_split()
    nonfill = tr[~_fill(tr)].reset_index(drop=True)
    reg = bs.fit_nf_regressor(nonfill, seed=5)
    params = reg.model.get_params()
    assert (params["max_leaf_nodes"], params["max_iter"], params["min_samples_leaf"]) == (15, 200, 100)
    assert params["random_state"] == 5 and params["learning_rate"] == 0.1 and params["early_stopping"] == "auto"
    assert params["loss"] == "squared_error" and params["l2_regularization"] == 0.0
    assert bs.NF_PARAMS == {"max_leaf_nodes": 15, "max_iter": 200, "min_samples_leaf": 100}
    assert bs.NF_NUMERIC == ["sp", "dayoff", "hr"]
    assert bs.NF_ENCODED == ["ADEP_mvt", "RUNWAY_mvt", "stand_pref", "airline", "AIRCRAFT_OPERATOR_flt"]
    assert reg.columns == ["sp", "dayoff", "hr", "te_ADEP_mvt", "te_RUNWAY_mvt", "te_stand_pref", "te_airline",
                           "te_AIRCRAFT_OPERATOR_flt"]
    x = reg.design(te)
    assert list(x.columns) == reg.columns and all(str(d) == "float64" for d in x.dtypes)
    assert np.array_equal(x["sp"].to_numpy(), te.sp.to_numpy(dtype="float64"))
    unseen = te.iloc[:3].copy()
    unseen["airline"] = "ZZZ"
    assert (reg.design(unseen)["te_airline"].to_numpy() == reg.prior).all()
    pred = reg.predict(te)
    assert pred.dtype == np.float64 and pred.shape == (len(te),) and np.isfinite(pred).all()
    assert reg.seed == 5 and reg.n_train == len(nonfill)


def test_winsorisation_applies_to_the_fit_target_only_and_never_to_scoring():
    """The regressor is fitted on `clip(y, -3000, 3000)`: its prior and its target encodings
    are means of the CLIPPED target (a 50,000 s row counts as 3,000), and `n_winsorised`
    counts the clipped rows. Scoring, in the harness, uses the RAW label: a 50,000 s row
    scored at 1,000 s has squared error 49,000², not 2,000².

    Fails when the prior or the encodings are computed from the raw target, when the count
    is wrong, or when the harness clips the label before the residual. Rehearsed 2026-09-09,
    each RED: `frame.y.mean()` for the prior; `y_fit = y` (no clip); `np.minimum(y, WINSOR_S)` in
    stratum_fold.squared_errors.
    """
    tr, _, _ = _small_split()
    nonfill = tr[~_fill(tr)].reset_index(drop=True).copy()
    nonfill.loc[:4, "y"] = 50_000.0                   # five raw rows far above the clip
    reg = bs.fit_nf_regressor(nonfill, seed=0)
    y = nonfill.y.to_numpy(dtype="float64")
    clipped = np.clip(y, -bs.WINSOR_S, bs.WINSOR_S)
    assert reg.prior == float(clipped.mean()) and reg.prior != float(y.mean())
    assert reg.n_winsorised == int((y > bs.WINSOR_S).sum()) >= 5
    ap = nonfill.ADEP_mvt.to_numpy()
    for a in np.unique(ap):
        m = ap == a
        want = (clipped[m].mean() * m.sum() + reg.prior * bs.SMOOTH) / (m.sum() + bs.SMOOTH)
        assert reg.maps["ADEP_mvt"][a] == pytest.approx(want, abs=1e-9), a   # float64 groupby mean
    preds = pd.DataFrame({"y": [50_000.0, 100.0], "S0": [1_000.0, 100.0]})
    se = sf.squared_errors(preds, ["S0"])
    assert se["S0"].tolist() == [49_000.0 ** 2, 0.0] and se["S0"].dtype == np.float64


def test_encodings_and_the_regressor_are_fitted_on_non_fill_training_rows_only():
    """Inside `fit_unmatched(hybrid=True)` the regressor sees `train[~fill]` and nothing else:
    its prediction equals `fit_nf_regressor(train[~fill], seed).predict(test)` bit for bit,
    and `fit_nf_regressor` refuses a frame that still contains fill rows (a fill row's label
    is `sp`, thousands of seconds, and would poison every encoding).

    Fails when the regressor is fitted on all unmatched rows, or when the guard is dropped.
    Rehearsed 2026-09-09, each RED: `fit_nf_regressor(tr, s)` (ValueError from the guard);
    guard removed AND `fit_nf_regressor(tr, s)` (predictions differ on every row).
    """
    tr, te, mat = _small_split()
    parts = {}
    bs.fit_unmatched(tr, te, train_matched=mat, hybrid=True, seeds=(0,), parts=parts)
    fill = _fill(tr)
    assert fill.any(), "fixture has no fill rows"
    want = bs.fit_nf_regressor(tr[~fill].reset_index(drop=True), 0).predict(te)
    assert np.array_equal(parts["nf_fit"], want)
    assert np.array_equal(parts["nf_fit_by_seed"][0], want) and list(parts["nf_fit_by_seed"]) == [0]
    with pytest.raises(ValueError, match="fill"):
        bs.fit_nf_regressor(tr, 0)
    with pytest.raises(ValueError, match="non-fill"):
        bs.fit_nf_regressor(tr.iloc[:0], 0)


def test_hybrid_mixture_keeps_p_hat_and_nf_cells_unchanged_and_averages_the_seeds(monkeypatch):
    """With `hybrid=True` only the non-fill term changes: `p` and `nf_cells` are bit-identical
    to the default path's, `nf_fit` is the mean over the seeds given, `nf` is the hybrid, the
    routing counts are exposed, and each per-seed arm the harness builds is the mixture with
    that seed's regressor. Seeds are stubbed so the three predictions differ by construction.

    EDDF's non-fill training rows are planted at 50,000 s so EDDF's cells sit above T_tail and
    the test rows there take the `nf_cells` branch — without that plant the fixture has no cell
    above 3,000 s and "routing applied" would be vacuous (found by the rehearsal: `nf = nf_fit`
    stayed GREEN until the plant was added).

    Fails when the hybrid touches `p`, when only the first seed is used, or when routing is
    not applied to the averaged prediction. Rehearsed 2026-09-09, each RED:
    `nf_fit = by_seed[used[0]]`; `p = p * 0.5` under `if hybrid:`; `nf = nf_fit`.
    """
    tr, te, mat = _small_split()
    tr = tr.copy()
    tr.loc[(tr.ADEP_mvt == "EDDF") & ~_fill(tr), "y"] = 50_000.0        # EDDF's cells -> tail branch
    base = {}
    s0 = bs.fit_unmatched(tr, te, train_matched=mat, parts=base)
    assert base["hybrid"] is False and base["nf_fit"] is None and base["routing"] is None
    assert np.array_equal(base["nf"], base["nf_cells"]) and base["seeds"] == ()

    class Stub:
        def __init__(self, seed):
            self.seed = seed

        def predict(self, frame):
            return np.full(len(frame), 100.0 * (self.seed + 1)) + np.arange(len(frame), dtype="float64")

    seen = []
    monkeypatch.setattr(bs, "fit_nf_regressor", lambda frame, seed: seen.append(seed) or Stub(seed))
    parts = {}
    s1 = bs.fit_unmatched(tr, te, train_matched=mat, hybrid=True, seeds=(0, 1, 2), parts=parts)
    assert seen == [0, 1, 2] and parts["seeds"] == (0, 1, 2) and parts["hybrid"] is True
    assert np.array_equal(parts["p"], base["p"]) and np.array_equal(parts["nf_cells"], base["nf_cells"])
    stubs = {s: Stub(s).predict(te) for s in (0, 1, 2)}
    want_mean = (stubs[0] + stubs[1] + stubs[2]) / 3.0
    assert np.allclose(parts["nf_fit"], want_mean, atol=0, rtol=0)
    assert np.array_equal(parts["nf"], bs.nf_hybrid(parts["nf_cells"], parts["nf_fit"]))
    eddf = (te.ADEP_mvt == "EDDF").to_numpy()
    assert (parts["nf_cells"][eddf] >= bs.T_TAIL_S).all() and (parts["nf_cells"][~eddf] < bs.T_TAIL_S).any()
    assert np.array_equal(parts["nf"][eddf], parts["nf_cells"][eddf]), "EDDF rows must take the cells branch"
    assert np.array_equal(parts["nf"][~eddf], parts["nf_fit"][~eddf]), "the body must take the fitted branch"
    assert parts["routing"] == bs.routing_counts(parts["nf_cells"])
    assert parts["routing"]["n_nf_fit"] > 0 and parts["routing"]["n_nf_cells"] == int(eddf.sum()) > 0
    assert parts["routing"]["n_nf_fit"] + parts["routing"]["n_nf_cells"] == len(te)
    assert np.array_equal(s1, bs.mixture(parts["p"], te.sp.to_numpy(), parts["nf"]))
    assert not np.array_equal(s0, s1)
    with pytest.raises(ValueError, match="seed"):
        bs.fit_unmatched(tr, te, train_matched=mat, hybrid=True, seeds=())
    with pytest.raises(ValueError, match="seed"):
        bs.fit_unmatched(tr, te, train_matched=mat, hybrid=True, seeds=(0, 0))


def test_predict_forwards_the_hybrid_flag_to_the_unmatched_rows_only(monkeypatch):
    """`predict(train, test, hybrid=True)` routes the unmatched rows through the hybrid with
    `SEEDS` and leaves the matched rows to `fit_matched`; the default is the incumbent.

    Fails when the flag is not forwarded or when `seeds` differ from SEEDS. Rehearsed
    2026-09-09: `fit_unmatched(..., hybrid=False, ...)` inside predict went RED.
    """
    tr, te, mat = _small_split()
    monkeypatch.setattr(bs, "fit_matched", lambda train, test: np.full(len(test), 777.0))
    train = pd.concat([tr, mat], ignore_index=True)
    test = pd.concat([te, mat.iloc[:50]], ignore_index=True)
    um = test.unmatched.to_numpy()
    got = bs.predict(train, test, hybrid=True)
    want = bs.fit_unmatched(tr, te, train_matched=mat, hybrid=True, seeds=bs.SEEDS)
    assert np.array_equal(got[um], want) and (got[~um] == 777.0).all()
    assert np.array_equal(bs.predict(train, test)[um], bs.fit_unmatched(tr, te, train_matched=mat))
    assert bs.SEEDS == (0, 1, 2)


def test_splice_unmatched_replaces_exactly_the_named_rows_and_keeps_int32():
    """The v7 shipping path: a predecessor submission with ONLY the unmatched rows replaced.
    Every id must be present exactly once; every other row and the dtype are untouched.

    Fails when a matched row moves, when an unknown id is silently ignored, or when the
    column is upcast. Rehearsed 2026-09-09, each RED: `col[:] = values[0]` (every row); the
    membership check dropped (`if False:`); `.astype("float64")`.
    """
    base = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                         "TAXITIME_SEC_mvt": np.array([900, 901, 902, 903, 904, 905], dtype="int32")})
    out = bs.splice_unmatched(base, np.array([5.0, 2.0]), np.array([50, 20], dtype="int32"))
    assert list(out.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"] and out.TAXITIME_SEC_mvt.dtype == np.int32
    assert out.TAXITIME_SEC_mvt.tolist() == [900, 20, 902, 903, 50, 905]
    assert base.TAXITIME_SEC_mvt.tolist() == [900, 901, 902, 903, 904, 905], "the base was mutated"
    with pytest.raises(ValueError, match="not in the base"):
        bs.splice_unmatched(base, np.array([9.0]), np.array([1], dtype="int32"))
    with pytest.raises(ValueError, match="duplicate"):
        bs.splice_unmatched(base, np.array([2.0, 2.0]), np.array([1, 2], dtype="int32"))
    with pytest.raises(ValueError, match="length"):
        bs.splice_unmatched(base, np.array([2.0]), np.array([1, 2], dtype="int32"))


def _write_raw_months(rng, raw: pathlib.Path, months=(1, 2, 3), n=900):
    """Synthetic raw parquet files in the shape fetch_data leaves them: one training file per
    month (DEP rows, matched LIRF + unmatched everywhere), a ranking file whose scored DEP
    rows have BLOCK/TAXITIME null, and the two-column template."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    raw.mkdir(parents=True, exist_ok=True)
    next_id = 10.0
    for m in months:
        unm = _raw_block(rng, n, matched=False)
        mat = _raw_block(rng, n // 2, matched=True)
        df = pd.concat([unm, mat], ignore_index=True)
        df["MVT_ID_mvt"] = next_id + np.arange(len(df), dtype="float64")
        next_id += len(df)
        pq.write_table(pa.Table.from_pandas(df), raw / f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet")
    scored = pd.concat([_raw_block(rng, 120, matched=False), _raw_block(rng, 240, matched=True)], ignore_index=True)
    scored["MVT_ID_mvt"] = next_id + np.arange(len(scored), dtype="float64")
    scored["BLOCK_TIME_UTC_mvt"] = pd.Series(pd.NaT, index=scored.index, dtype="datetime64[ns, UTC]")
    scored["TAXITIME_SEC_mvt"] = pd.array([None] * len(scored), dtype="Int32")
    arr = scored.iloc[:5].copy()
    arr["PHASE_mvt"] = "ARR"
    arr["MVT_ID_mvt"] = arr.MVT_ID_mvt + 1e6
    pq.write_table(pa.Table.from_pandas(pd.concat([scored, arr], ignore_index=True)), raw / "ranking.parquet")
    template = pd.DataFrame({"MVT_ID_mvt": scored.MVT_ID_mvt.to_numpy(),
                             "TAXITIME_SEC_mvt": pd.array([None] * len(scored), dtype="Int32")})
    pq.write_table(pa.Table.from_pandas(template), raw / "submitting.parquet")
    return template


def test_shipping_path_base_splice_changes_only_the_unmatched_rows(monkeypatch, tmp_path, capsys):
    """`build_submission.py --hybrid --base <v_prev> --version N` writes v_N as the base with
    ONLY the scored unmatched rows replaced by the hybrid stratum (seeds 0-2), rounded like
    every submission; the matched rows are byte-identical to the base, the matched model is
    never fitted, the routing counts and the replaced-row count are printed, and the file
    passes the submission contract on re-read. Without `--hybrid` the same path reproduces the
    incumbent's rounded predictions; without `--base` the full predictor runs.

    Fails when the base's matched rows change, when the stratum is the incumbent under
    `--hybrid`, or when `--base` still fits the matched model. Rehearsed 2026-09-09, each RED:
    `hybrid=False` on the --base path; `predict(frame, scored, hybrid=args.hybrid)` under
    --base (the matched stub's 777 lands on matched rows).
    """
    rng = np.random.default_rng(11)
    root = tmp_path / "repo"
    raw = root / "data" / "raw"
    template = _write_raw_months(rng, raw)
    monkeypatch.setattr(bs, "ROOT", root)
    monkeypatch.setattr(bs, "RAW", raw)
    monkeypatch.setattr(bs, "fit_matched", lambda train, test: np.full(len(test), 777.0))
    (root / "submissions").mkdir()
    base = pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt.to_numpy(),
                         "TAXITIME_SEC_mvt": np.full(len(template), 900, dtype="int32")})
    base.to_parquet(root / "submissions" / "merry-quicksand_v1.parquet", index=False)

    assert bs.main(["--hybrid", "--base", "submissions/merry-quicksand_v1.parquet", "--version", "2"]) == 0
    out = capsys.readouterr().out
    v2 = pd.read_parquet(root / "submissions" / "merry-quicksand_v2.parquet")
    bs.check_submission(v2, template)
    # the truth, rebuilt independently from the same raw files through the same recipe
    import glob
    frame = bs.derive(bs.admissible(bs.load_movements(sorted(glob.glob(str(raw / "training_*.parquet"))))))
    frame["y"] = frame.TAXITIME_SEC_mvt.astype("float64")
    ranking = bs.derive(bs.load_movements([raw / "ranking.parquet"]))
    scored = ranking[ranking.MVT_ID_mvt.isin(set(template.MVT_ID_mvt))].reset_index(drop=True)
    um = scored.unmatched.to_numpy()
    assert um.sum() == 120 and (~um).sum() == 240
    want = bs.fit_unmatched(frame[frame.unmatched], scored[um], train_matched=frame[~frame.unmatched],
                            hybrid=True, seeds=bs.SEEDS)
    got = v2.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    assert np.array_equal(got.loc[scored.MVT_ID_mvt[um]].to_numpy(), np.rint(want).astype("int32"))
    assert (got.loc[scored.MVT_ID_mvt[~um]].to_numpy() == 900).all(), "a matched row moved"
    assert "routed" in out and "nf_fit" in out and "replaced 120" in out and "777" not in out
    incumbent = bs.fit_unmatched(frame[frame.unmatched], scored[um], train_matched=frame[~frame.unmatched])
    assert not np.array_equal(np.rint(incumbent), np.rint(want)), "the hybrid changed nothing on the fixture"

    assert bs.main(["--base", "submissions/merry-quicksand_v1.parquet", "--version", "3"]) == 0
    v3 = pd.read_parquet(root / "submissions" / "merry-quicksand_v3.parquet").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    assert np.array_equal(v3.loc[scored.MVT_ID_mvt[um]].to_numpy(), np.rint(incumbent).astype("int32"))
    assert (v3.loc[scored.MVT_ID_mvt[~um]].to_numpy() == 900).all()

    assert bs.main(["--hybrid", "--version", "4"]) == 0                # no base: the full predictor
    v4 = pd.read_parquet(root / "submissions" / "merry-quicksand_v4.parquet").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    assert (v4.loc[scored.MVT_ID_mvt[~um]].to_numpy() == 777).all()
    assert np.array_equal(v4.loc[scored.MVT_ID_mvt[um]].to_numpy(), np.rint(want).astype("int32"))
    with pytest.raises(SystemExit):
        bs.main(["--base", "submissions/does_not_exist_v1.parquet", "--version", "5"])


# =============================================================================================
# B. stratum_fold: the Amendment 18.2 harness
# =============================================================================================

def test_monster_mask_is_strictly_above_three_hours():
    """Amendment 9 rule 4 / STRATUM_MONSTERS §0: a monster is `y > 10,800 s`, strictly — a
    row at exactly three hours is NOT a monster.

    Fails when `>` slips to `>=` (the 10,800.0 row) or the constant drifts. Rehearsed
    2026-09-09, each RED: `>= MONSTER_S`; `MONSTER_S = 10_801.0`.
    """
    y = np.array([10_799.9, 10_800.0, 10_800.1, 0.0, 131_167.0, -1.0])
    assert sf.monster_mask(y).tolist() == [False, False, True, False, True, False]
    assert sf.MONSTER_S == 10_800.0 and sf.monster_mask(y).dtype == np.bool_


def test_lomo_leaves_exactly_one_month_out_each_fold_and_never_scores_a_training_row(monkeypatch):
    """Twelve folds, one per month present, in month order; the test mask is that month and
    the training mask its complement; the masks partition the rows; across folds every row
    is scored exactly once. Then `score_lomo`, with the arm builder stubbed to record what it
    was given, must never hand a scored row to the training side and must fill every row of
    the out-of-fold frame once.

    Fails when a fold trains on its own month, when a month is skipped or scored twice, or
    when the out-of-fold assembly overwrites rows. Rehearsed 2026-09-09, each RED:
    `np.ones_like(month, dtype=bool)` for the training mask; `month <= m` for the test mask;
    `lomo_folds(month)[1:]` in score_lomo.
    """
    month = np.array([3, 1, 12, 7, 1, 3, 12, 7, 5])
    folds = sf.lomo_folds(month)
    assert [m for m, _, _ in folds] == [1, 3, 5, 7, 12]
    scored = np.zeros(len(month), dtype="int64")
    for m, tr, te in folds:
        assert te.tolist() == (month == m).tolist() and tr.tolist() == (month != m).tolist()
        assert not (tr & te).any() and (tr | te).all()
        scored += te
    assert (scored == 1).all()
    with pytest.raises(ValueError, match="month"):
        sf.lomo_folds(np.array([4, 4, 4]))

    unm, mat = sf.synthetic_frame(n_per_month=60, seed=1, n_matched_per_month=30)
    calls = []

    def stub(train_unm, test_unm, train_matched, seeds):
        calls.append((set(train_unm.MVT_ID_mvt), set(test_unm.MVT_ID_mvt), set(train_matched.month)))
        n = len(test_unm)
        arms = {a: np.full(n, float(i + 1)) for i, a in enumerate(sf.ARMS)}
        parts = {"p": np.full(n, 0.5), "nf_cells": np.full(n, 10.0), "nf_fit": np.full(n, 20.0),
                 "nf": np.full(n, 20.0), "routing": bs.routing_counts(np.full(n, 10.0))}
        return arms, parts

    monkeypatch.setattr(sf, "stratum_arms", stub)
    oof = sf.score_lomo(unm, mat, seeds=sf.SEEDS, log=lambda *_: None)
    assert len(calls) == 12 and len(oof) == len(unm) and oof.fold.eq("lomo").all()
    ids = unm.MVT_ID_mvt
    for (train_ids, test_ids, matched_months), (m, _, _) in zip(calls, sf.lomo_folds(unm.month.to_numpy())):
        assert not (train_ids & test_ids), "a scored row was also a training row"
        assert test_ids == set(ids[unm.month == m]) and train_ids == set(ids[unm.month != m])
        assert m not in matched_months, "the held-out month's matched rows reached the LIRF model"
    assert sorted(oof.MVT_ID_mvt) == sorted(ids) and oof.MVT_ID_mvt.is_unique
    assert set(oof.month) == set(range(1, 13)) and (oof.S0 == 1.0).all() and (oof.S1 == 2.0).all()
    assert oof.routed_nf_fit.all() and (oof.nf_cells == 10.0).all()


def test_fold_a_holds_out_january_and_july_and_trains_on_the_other_ten():
    """Fold A is `bs.HOLDOUT_MONTHS` = (1, 7) — Screen B's fold, so S0 there must reproduce
    RESULT 3.2's B0. Fails when the holdout or the complement drifts. Rehearsed 2026-09-09:
    `holdout=(1, 6)` default went RED."""
    month = np.array([1, 2, 7, 12, 7, 1, 3])
    tr, te = sf.fold_a_masks(month)
    assert te.tolist() == [1, 0, 1, 0, 1, 1, 0] and tr.tolist() == [0, 1, 0, 1, 0, 0, 1]
    assert sf.FOLD_A == (1, 7) == bs.HOLDOUT_MONTHS
    with pytest.raises(ValueError, match="holdout"):
        sf.fold_a_masks(np.array([2, 3]))


def _analytic_block_se(se_a, se_b, month):
    """Delta-method SE of rmse(a) - rmse(b) under month-block resampling with EQUAL month
    sizes (the denominator is then fixed): the mean squares vary through the 12 resampled
    per-month SSE sums, each an iid draw from the 12-point empirical distribution."""
    months = np.unique(month)
    k, n = len(months), len(month)
    sa = np.array([se_a[month == m].sum() for m in months])
    sb = np.array([se_b[month == m].sum() for m in months])
    c = k * np.cov(sa, sb, ddof=0) / n ** 2                 # cov of the resampled mean squares
    ma, mb = se_a.mean(), se_b.mean()
    return float(np.sqrt(c[0, 0] / (4 * ma) + c[1, 1] / (4 * mb) - 2 * c[0, 1] / (4 * np.sqrt(ma * mb))))


def test_month_block_bootstrap_is_paired_by_month_finds_a_planted_gain_and_rejects_noise():
    """The falsify rule for a harness, at the block level. Twelve equal months. A noise pair
    (same variance, residual correlation 0.95) must get an interval that includes zero with a
    half-width within 20% of 1.96 x the delta-method block SE (12 blocks make the percentile
    interval lumpier than a row bootstrap's; measured 1.02 on this fixture); an arm whose
    residuals are 10% smaller (a gain of 9.8 s, >= 10 x its block SE) must get an interval
    that excludes zero and a point equal to the actual RMSE difference. The same month draws
    serve every pair and every arm (two identical arms give an interval of exactly [0, 0]).
    Block, not row: an arm whose whole gain sits in ONE month gets a lower bound of exactly
    zero, because the month is absent from (11/12)^12 = 35% of the draws — a row bootstrap
    would exclude zero. `block_draws` resamples the 12 blocks WITH replacement: block 12 is
    absent from 31-40% of 2,000 draws and some draw carries a block twice. A `mask` restricts
    the rows BEFORE the per-month sums.

    Fails when the draws are not paired, when rows rather than months are resampled, when the
    resampling is without replacement, or when the mask is applied after the sums. Rehearsed
    2026-09-09, each RED: an independent `block_draws` per arm; every row made its own block (a
    row bootstrap); `rng.permutation(n_blocks)` (without replacement); mask dropped (`if False:`).
    """
    k, per, rng = 12, 500, np.random.default_rng(0)
    n = k * per
    month = np.repeat(np.arange(1, k + 1), per)
    ra = rng.normal(0, 100, n)
    eps = rng.normal(0, 100, n)
    rb = 0.95 * ra + np.sqrt(1 - 0.95 ** 2) * eps                # noise pair
    rc = 0.9 * rb                                                # planted 10% smaller residuals
    rd = np.where(month == 12, 0.5 * ra, ra)                     # the gain lives in one month
    se = {"A": ra ** 2, "B": rb ** 2, "C": rc ** 2, "D": rd ** 2, "A2": ra ** 2}
    res = sf.month_block_bootstrap(se, month, [("B", "A"), ("C", "A"), ("D", "A"), ("A2", "A")], n_draws=2_000, seed=0)
    assert set(res) == {"B_vs_A", "C_vs_A", "D_vs_A", "A2_vs_A"}

    noise = res["B_vs_A"]
    point = float(np.sqrt(se["A"].mean()) - np.sqrt(se["B"].mean()))
    se_noise = _analytic_block_se(se["A"], se["B"], month)
    assert abs(point) < 3 * se_noise, "fixture is not a noise pair"
    assert noise["gain_s"] == pytest.approx(point, abs=1e-12)
    lo, hi = noise["ci95"]
    assert lo <= 0.0 <= hi and noise["excludes_zero"] is False and noise["improving"] is False
    ratio = ((hi - lo) / 2) / (1.96 * se_noise)
    assert 0.8 < ratio < 1.2, f"interval half-width is {ratio:.2f} x the block SE"
    assert noise["n_draws"] == 2_000 and noise["seed"] == 0 and noise["n_blocks"] == 12

    planted = res["C_vs_A"]
    gain = float(np.sqrt(se["A"].mean()) - np.sqrt(se["C"].mean()))
    assert gain >= 10 * _analytic_block_se(se["A"], se["C"], month), "fixture does not plant a 10-sigma gain"
    assert planted["gain_s"] == pytest.approx(gain, abs=1e-12)
    assert planted["ci95"][0] > 0 and planted["excludes_zero"] is True and planted["improving"] is True

    one_month = res["D_vs_A"]
    assert one_month["gain_s"] > 0 and one_month["ci95"][0] == 0.0 and one_month["excludes_zero"] is False
    assert res["A2_vs_A"]["ci95"] == [0.0, 0.0] and res["A2_vs_A"]["gain_s"] == 0.0

    draws = sf.block_draws(k, 2_000, 0)
    assert draws.shape == (2_000, k) and (draws.sum(axis=1) == k).all() and draws.dtype.kind == "i"
    absent = float((draws[:, 11] == 0).mean())
    assert 0.31 < absent < 0.40, f"block 12 absent from {absent:.3f} of draws; (11/12)^12 = 0.352"
    assert (draws >= 2).any(), "no block was ever drawn twice: not with replacement"
    again = sf.month_block_bootstrap(se, month, [("B", "A")], n_draws=2_000, seed=0)
    assert again["B_vs_A"]["ci95"] == noise["ci95"]

    mask = month != 12
    masked = sf.month_block_bootstrap(se, month, [("D", "A")], n_draws=500, seed=0, mask=mask)
    subset = sf.month_block_bootstrap({k_: v[mask] for k_, v in se.items()}, month[mask], [("D", "A")], n_draws=500, seed=0)
    assert masked["D_vs_A"] == subset["D_vs_A"] and masked["D_vs_A"]["gain_s"] == 0.0 and masked["D_vs_A"]["n_blocks"] == 11
    with pytest.raises(ValueError, match="block"):
        sf.month_block_bootstrap(se, month, [("B", "A")], n_draws=10, seed=0, mask=month == 1)


def test_seed_sd_two_sigma_rule_and_airports_improving_match_the_repo_rule():
    """Seed sd is the SAMPLE sd (ddof=1) of the three per-seed S1 LOMO RMSEs; the rule is
    gain > 2 x sd strictly; a zero sd (seeds identical below sklearn's 10,000-row early-stop
    switch) makes any positive gain clear it and is reported, not hidden. Airports improving
    counts strict improvements only.

    Fails when ddof=0, when `>=` replaces `>`, or when a tie counts. Rehearsed 2026-09-09,
    each RED: `np.std(v)`; `>=`; `new[a] <= ref[a]`.
    """
    sd = sf.seed_sd([226.24, 226.50, 226.80])
    assert sd == pytest.approx(0.28031, abs=1e-4)
    assert sf.exceeds_2x_seed_sd(0.6, sd) is True and sf.exceeds_2x_seed_sd(0.5, sd) is False
    assert sf.exceeds_2x_seed_sd(2 * sd, sd) is False and sf.exceeds_2x_seed_sd(-1.0, sd) is False
    assert sf.seed_sd([1.0, 1.0, 1.0]) == 0.0 and sf.exceeds_2x_seed_sd(1e-9, 0.0) is True
    assert sf.exceeds_2x_seed_sd(0.0, 0.0) is False
    with pytest.raises(ValueError, match="three"):
        sf.seed_sd([1.0, 2.0])
    assert sf.airports_improving({"A": 1.0, "B": 2.0, "C": 3.0}, {"A": 2.0, "B": 2.0, "C": 2.0}) == 1


def test_arm_table_cuts_pooled_ex_monster_per_airport_and_per_month_from_raw_labels():
    """Every arm is reported pooled and ex-monster, per airport and per month, with the row
    and monster counts, all from the raw label. Pinned by hand on six rows with one monster.

    Fails when the ex-monster cut keeps the monster, when a cut is pooled instead of sliced,
    or when the monster count is wrong. Rehearsed 2026-09-09, each RED: `exm = np.ones_like(mon)`;
    per-airport `rmse(e)` on all rows.
    """
    preds = pd.DataFrame({"y": [1_000.0, 1_000.0, 2_000.0, 2_000.0, 50_000.0, 500.0],
                          "ADEP_mvt": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
                          "month": [1, 2, 1, 2, 1, 2],
                          "S0": [1_100.0, 900.0, 2_000.0, 2_000.0, 1_000.0, 500.0],
                          "S1": [1_000.0, 1_000.0, 2_100.0, 1_900.0, 1_000.0, 600.0]})
    t = sf.arm_table(preds, ["S0", "S1"])
    assert t["S0"]["pooled"] == pytest.approx(np.sqrt((100 ** 2 * 2 + 49_000 ** 2) / 6), rel=1e-12)
    assert t["S0"]["exmonster"] == pytest.approx(np.sqrt(100 ** 2 * 2 / 5), rel=1e-12)
    assert t["S1"]["exmonster"] == pytest.approx(np.sqrt((100 ** 2 * 3) / 5), rel=1e-12)
    assert t["S0"]["n"] == 6 and t["S0"]["n_monsters"] == 1
    assert t["S0"]["per_airport"]["AAA"] == {"pooled": 100.0, "exmonster": 100.0, "n": 2, "n_monsters": 0}
    assert t["S0"]["per_airport"]["CCC"]["exmonster"] == 0.0 and t["S0"]["per_airport"]["CCC"]["n_monsters"] == 1
    assert t["S1"]["per_month"][2]["pooled"] == pytest.approx(np.sqrt((0 + 100 ** 2 + 100 ** 2) / 3), rel=1e-12)
    assert t["S1"]["per_month"][1]["exmonster"] == pytest.approx(np.sqrt((0 + 100 ** 2) / 2), rel=1e-12)
    assert list(t["S0"]["per_month"]) == [1, 2]


def test_clauses_18_3_are_numbers_and_booleans_with_no_verdict_word():
    """The three ESTABLISHED clauses and the two NOT WORKING triggers of Amendment 18.3 are
    evaluated to numbers and booleans — the owner writes the verdict. Pinned by hand: an
    ex-monster gain of 25 s with interval [5, 45], seed sd 2 (2 x sd = 4 < 25), a pooled
    interval [-15, 30] (loss bound 15 < 20), 7 of 10 airports improving; then the same with
    the pooled interval [-25, 30] (the tail was NOT preserved under the interval reading) and
    a gain of 19.9 s (below the 20 s bar). The rendered lines carry no verdict word.

    Fails when a threshold drifts, when the loss bound uses the wrong end of the interval, or
    when a verdict word appears. Rehearsed 2026-09-09, each RED: `gain >= 19.0`;
    `loss_upper_bound_s = -phi` (the upper end); a "VERDICT: ..." line added to render_clauses.
    """
    exm = {"gain_s": 25.0, "ci95": [5.0, 45.0], "excludes_zero": True, "improving": True}
    pooled = {"gain_s": 2.0, "ci95": [-15.0, 30.0], "excludes_zero": False, "improving": False}
    s0 = {a: 100.0 for a in AIRPORTS}
    s1 = {a: (90.0 if i < 7 else 100.0) for i, a in enumerate(AIRPORTS)}
    c = sf.clauses_18_3(exm, pooled, s0, s1, seed_sd_exmonster=2.0)
    a, b, cc = c["a_exmonster"], c["b_pooled_tail_preserved"], c["c_airports"]
    assert a["gain_s"] == 25.0 and a["ci95"] == [5.0, 45.0] and a["excludes_zero_improving"] is True
    assert a["gain_ge_20_s"] is True and a["seed_sd_s"] == 2.0 and a["gain_gt_2x_seed_sd"] is True
    assert a["holds"] is True
    assert b["gain_s"] == 2.0 and b["loss_upper_bound_s"] == 15.0 and b["loss_upper_bound_lt_20_s"] is True
    assert b["holds"] is True
    assert cc["airports_improving_exmonster"] == 7 and cc["n_airports"] == 10 and cc["at_least_6"] is True
    assert c["all_three_hold"] is True
    nw = c["not_working_triggers"]
    assert nw["exmonster_interval_includes_zero"] is False
    assert nw["pooled_point_worse_by_more_than_20_s"] is False and nw["pooled_interval_worse_by_more_than_20_s"] is False
    assert nw["any"] is False

    pooled2 = {"gain_s": -1.0, "ci95": [-25.0, 30.0], "excludes_zero": False, "improving": False}
    exm2 = dict(exm, gain_s=19.9)
    c2 = sf.clauses_18_3(exm2, pooled2, s0, s1, seed_sd_exmonster=10.0)
    assert c2["a_exmonster"]["gain_ge_20_s"] is False and c2["a_exmonster"]["gain_gt_2x_seed_sd"] is False
    assert c2["a_exmonster"]["holds"] is False and c2["all_three_hold"] is False
    assert c2["b_pooled_tail_preserved"]["loss_upper_bound_s"] == 25.0
    assert c2["b_pooled_tail_preserved"]["loss_upper_bound_lt_20_s"] is False and c2["b_pooled_tail_preserved"]["holds"] is False
    assert c2["not_working_triggers"]["pooled_interval_worse_by_more_than_20_s"] is True
    assert c2["not_working_triggers"]["pooled_point_worse_by_more_than_20_s"] is False
    assert c2["not_working_triggers"]["any"] is True
    text = "\n".join(sf.render_clauses(c) + sf.render_clauses(c2))
    for word in sf.VERDICT_WORDS:
        assert word.lower() not in text.lower(), word
    assert "25.0" in text and "(a)" in text and "(b)" in text and "(c)" in text
    assert sf.VERDICT_WORDS == ("ESTABLISHED", "NOT WORKING", "INCONCLUSIVE")


def test_synthetic_frame_is_shaped_like_derive_and_plants_the_registered_structure():
    """The smoke's frame: 12 months, ten airports, unmatched rows with every NM clock null and
    the `*_flt` columns null together, LIRF matched rows with the copy pattern, a non-fill
    body whose label depends on airline / stand prefix / hour (invisible to the cells), and a
    LIRF long-wait tail whose cells carry loads above T_tail and above the monster line.

    Fails when a derive column is missing, when the tail is absent (nothing to preserve), or
    when the body structure is absent (nothing for the regressor to find). Rehearsed
    2026-09-09: `_TAIL_SHARE = 0.0` went RED on the monster count.
    """
    unm, mat = sf.synthetic_frame(n_per_month=200, seed=0, n_matched_per_month=80)
    need = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "FLIGHT_mvt",
            "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt",
            "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt",
            "y", "delta", "proxy", "sp", "sp_band", "sp_fine", "dayoff", "hr", "month", "airline", "stand_pref",
            "stand_c1", "ades_p2", "unmatched", "MVT_ID_mvt"]
    for d in (unm, mat):
        assert all(c in d.columns for c in need), [c for c in need if c not in d.columns]
    assert unm.unmatched.all() and unm.AOBT_3_flt.isna().all() and unm.AIRCRAFT_OPERATOR_flt.isna().all()
    assert unm.delta.isna().all() and (unm.y > 0).all() and (unm.y == unm.TAXITIME_SEC_mvt).all()
    assert not mat.unmatched.any() and (mat.ADEP_mvt == "LIRF").all() and mat.AOBT_3_flt.notna().all()
    assert set(unm.month) == set(range(1, 13)) == set(mat.month) and set(unm.ADEP_mvt) == set(AIRPORTS)
    assert unm.MVT_ID_mvt.is_unique and not set(unm.MVT_ID_mvt) & set(mat.MVT_ID_mvt)
    assert unm.sp_band.dtype.name == "category" and unm.sp_fine.dtype.name == "category"
    fill = _fill(unm)
    lirf = (unm.ADEP_mvt == "LIRF").to_numpy()
    assert 0.3 < fill[lirf].mean() < 0.6 and fill[~lirf].mean() < 0.08
    stamped = (unm.BLOCK_TIME_UTC_mvt == unm.SCHED_TIME_UTC_mvt).to_numpy()
    assert stamped.sum() > 0.9 * fill.sum() and (unm.y[stamped] == unm.sp[stamped]).all(), "a stamped row's label is sp"
    assert (unm.y[fill] - unm.sp[fill]).abs().max() <= 60, "a fill row is within the 60 s tolerance of sp"
    monsters = sf.monster_mask(unm.y.to_numpy())
    assert monsters.sum() >= 20 and monsters[lirf].sum() > 0.8 * monsters.sum()
    body = ~fill & ~monsters & ~lirf
    g = unm[body].groupby("airline", observed=True).y.mean()
    assert g.max() - g.min() > 300, "airline structure not planted"
    g = unm[body].groupby("hr", observed=True).y.mean()
    assert g.max() - g.min() > 200, "hour structure not planted"
    cop = ((mat.BLOCK_TIME_UTC_mvt - mat.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60) & \
          ((mat.BLOCK_TIME_UTC_mvt - mat.AOBT_3_flt).dt.total_seconds().abs() > 300)
    assert 0.2 < cop.mean() < 0.7
    u2, m2 = sf.synthetic_frame(n_per_month=200, seed=0, n_matched_per_month=80)
    assert u2.equals(unm) and m2.equals(mat), "the synthetic frame is not deterministic"


def test_output_paths_and_flags():
    """Real runs write reports/stratum_fold_v7.{json,log} and data/cache_stand/
    stratum_fold_v7_preds.parquet; an out dir redirects all three; --smoke takes 200 draws
    and --synthetic swaps the loader. Fails when a name collides with another harness's or
    the smoke keeps 2,000 draws. Rehearsed 2026-09-09: `"lgbm_fold_v4.json"` went RED."""
    assert sf.output_paths(None) == (sf.REPORTS / "stratum_fold_v7.json", sf.REPORTS / "stratum_fold_v7.log",
                                     sf.CACHE / "stratum_fold_v7_preds.parquet")
    out = pathlib.Path("/o")
    assert sf.output_paths(out) == (out / "stratum_fold_v7.json", out / "stratum_fold_v7.log",
                                    out / "stratum_fold_v7_preds.parquet")
    a = sf.parse_args([])
    assert a.smoke is False and a.synthetic is False and a.out_dir is None and a.n_boot == 2_000
    b = sf.parse_args(["--smoke", "--synthetic", "--out-dir", "/x"])
    assert b.smoke and b.synthetic and b.out_dir == "/x" and b.n_boot == sf.SMOKE["n_boot"] == 200
    assert sf.parse_args(["--n-boot", "50"]).n_boot == 50
    with pytest.raises(SystemExit):
        sf.parse_args(["--n-boot", "1"])
    assert sf.N_BOOT == 2_000 and sf.BOOT_SEED == 0 and sf.SEEDS == (0, 1, 2)


def _recomputed(preds: pd.DataFrame, arm: str, mask=None) -> float:
    e = (preds.y.to_numpy(dtype="float64") - preds[arm].to_numpy(dtype="float64")) ** 2
    return float(np.sqrt(e[mask].mean() if mask is not None else e.mean()))


def test_smoke_synthetic_end_to_end_hybrid_beats_the_cells_ex_monster_and_keeps_the_tail(tmp_path, capsys):
    """`--smoke --synthetic`: the whole harness on the planted frame in a scratch directory.
    The JSON, the log and the parquet land and the real paths are untouched; the LOMO covers
    every stratum row once and fold A the (1, 7) rows; S0 and S1 are recomputable from the
    parquet's `p_hat`, `nf_cells`, `nf_fit` and `routed_nf_fit` columns (so p_hat is shared
    and routing is what the record says); the JSON's RMSEs are recomputable from the parquet;
    the ex-monster S1 - S0 month-block interval excludes zero in the improving direction with
    a double-digit gain, the pooled loss bound is below 20 s (the tail load is preserved),
    >= 6 of 10 airports improve ex-monster, both routing branches are used, the three seeds
    are recorded with their sd, the LIRF cut is present, the 18.3 clauses are printed with
    their numbers, the reproduction check reports itself skipped, and NO verdict word appears
    in the log or the JSON.

    Fails when routing is disabled in either direction (always `nf_fit`: the pooled loss
    bound explodes; always `nf_cells`: the ex-monster gain vanishes), when the parquet and
    the JSON disagree, when the smoke writes to reports/ or data/cache_stand/, or when a
    verdict word leaks. Rehearsed 2026-09-09, each RED: `np.where(True, nf_fit, nf_cells)`;
    `np.where(False, ...)`; `json_path = REPORTS / JSON_NAME` under --out-dir; a
    `log("VERDICT: ESTABLISHED")` line.
    """
    real = [sf.REPORTS / "stratum_fold_v7.json", sf.REPORTS / "stratum_fold_v7.log", sf.CACHE / "stratum_fold_v7_preds.parquet"]
    before = [p.stat().st_mtime_ns if p.exists() else None for p in real]
    out = tmp_path / "out"
    rc = sf.main(["--smoke", "--synthetic", "--out-dir", str(out)])
    assert rc == 0
    assert [p.stat().st_mtime_ns if p.exists() else None for p in real] == before, "the smoke touched a real output path"
    import json
    j = json.loads((out / "stratum_fold_v7.json").read_text())
    log = (out / "stratum_fold_v7.log").read_text()
    preds = pd.read_parquet(out / "stratum_fold_v7_preds.parquet")
    for key in ("smoke", "synthetic", "git_sha", "command", "estimate", "config", "data", "reproduction",
                "lomo", "fold_a", "judgement_calls", "peak_rss_gb", "wall_s"):
        assert key in j, key
    assert j["smoke"] is True and j["synthetic"] is True and j["config"]["n_boot"] == 200
    assert j["config"]["seeds"] == [0, 1, 2] and j["config"]["T_tail_s"] == 3_000.0 and j["config"]["monster_s"] == 10_800.0
    assert j["config"]["regressor"] == bs.NF_PARAMS and j["config"]["encoded"] == bs.NF_ENCODED
    assert j["reproduction"]["status"].startswith("skipped")
    text = (log + json.dumps(j)).lower()
    for word in sf.VERDICT_WORDS:
        assert word.lower() not in text, f"verdict word {word!r} leaked"
    lines = [l for l in log.splitlines() if l.strip()]
    assert sf.SMOKE_BANNER in lines[0] and sf.SMOKE_BANNER in lines[-1]

    lomo, fa = preds[preds.fold == "lomo"], preds[preds.fold == "A"]
    assert len(lomo) == j["data"]["n_stratum"] and lomo.MVT_ID_mvt.is_unique and set(lomo.month) == set(range(1, 13))
    assert set(fa.month) == {1, 7} and len(fa) == int((lomo.month.isin((1, 7))).sum())
    cols = ["MVT_ID_mvt", "fold", "month", "ADEP_mvt", "y", "sp", "monster", "p_hat", "nf_cells", "nf_fit",
            "routed_nf_fit", "S0", "S1", "S1_seed0", "S1_seed1", "S1_seed2"]
    assert list(preds.columns) == cols
    for part in (lomo, fa):
        p, sp = part.p_hat.to_numpy(), part.sp.to_numpy()
        nf = np.where(part.routed_nf_fit.to_numpy(), part.nf_fit.to_numpy(), part.nf_cells.to_numpy())
        assert np.allclose(part.S1.to_numpy(), bs.mixture(p, sp, nf), atol=1e-9, rtol=0)
        assert np.allclose(part.S0.to_numpy(), bs.mixture(p, sp, part.nf_cells.to_numpy()), atol=1e-9, rtol=0)
        assert np.array_equal(part.routed_nf_fit.to_numpy(), part.nf_cells.to_numpy() < 3_000.0)
        assert np.array_equal(part.monster.to_numpy(), part.y.to_numpy() > 10_800.0)
        s = np.stack([part[f"S1_seed{k}"].to_numpy() for k in range(3)])
        assert np.allclose(part.S1.to_numpy(), s.mean(axis=0), atol=1e-6, rtol=0)   # mean of routed per-seed arms
    for block, part in (("lomo", lomo), ("fold_a", fa)):
        exm = ~part.monster.to_numpy()
        for arm in sf.ARMS:
            assert j[block]["arms"][arm]["pooled"] == pytest.approx(_recomputed(part, arm), abs=1e-9), (block, arm)
            assert j[block]["arms"][arm]["exmonster"] == pytest.approx(_recomputed(part, arm, exm), abs=1e-9), (block, arm)
        r = j[block]["routing"]
        assert r["n_nf_fit"] > 0 and r["n_nf_cells"] > 0 and r["n_nf_fit"] + r["n_nf_cells"] == len(part)
        assert r["n_nf_fit"] == int(part.routed_nf_fit.sum())

    pair = j["lomo"]["pairs"]["S1_vs_S0"]
    exm, pooled = pair["exmonster"], pair["pooled"]
    assert exm["gain_s"] > 10.0 and exm["improving"] is True and exm["ci95"][0] > 0.0, exm
    assert -pooled["ci95"][0] < 20.0, f"pooled loss bound {-pooled['ci95'][0]:.1f} s: the tail was not preserved"
    assert pair["airports_improving_exmonster"] >= 6 and pair["n_airports"] == 10
    assert set(pair["per_airport_exmonster"]) == set(AIRPORTS)
    c = j["lomo"]["clauses_18_3"]
    assert c["a_exmonster"]["gain_s"] == exm["gain_s"] and c["c_airports"]["airports_improving_exmonster"] == pair["airports_improving_exmonster"]
    assert c["b_pooled_tail_preserved"]["loss_upper_bound_lt_20_s"] is True and "(a)" in log and "(c)" in log
    sd = j["lomo"]["seed_sd"]
    assert sorted(sd["values_exmonster"]) == ["0", "1", "2"] and sd["exmonster"] >= 0.0 and sd["pooled"] >= 0.0
    assert sd["exmonster"] == pytest.approx(sf.seed_sd([j["lomo"]["arms"][f"S1_seed{k}"]["exmonster"] for k in range(3)]), abs=1e-12)
    assert set(j["lomo"]["pairs"]) == {"S1_vs_S0", "S1_seed0_vs_S0", "S1_seed1_vs_S0", "S1_seed2_vs_S0"}
    assert j["lomo"]["lirf_cut"]["n"] > 0 and "S0" in j["lomo"]["lirf_cut"]["pooled"] and "exmonster" in j["lomo"]["lirf_cut"]
    assert set(j["lomo"]["per_month"]) == {str(m) for m in range(1, 13)}
    assert j["fold_a"]["pairs"]["S1_vs_S0"]["exmonster"]["gain_s"] > 0
    assert j["fold_a"]["r3_2_reference"] == sf.R3_2
    assert "routed" in log and "seed sd" in log and "(b)" in log
