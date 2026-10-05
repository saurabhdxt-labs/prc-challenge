"""scripts/analysis/blend_exact.py — exact, label-free convex blending of board-scored submission files.

Written first and seen red 2026-10-05 (no module: importing blend_exact.py fails).

Covers: load_members's sort-by-id alignment and five refusals, load_scores's happy path and five refusals,
pair_d's direct-computation equivalence, exact_mse's board-MSE identity (exact on unrounded board scores, bounded
under 4-decimal rounding) and its two refusals, eligibility's statistic computation and the four pass/fail
conditions at their exact boundaries, coefficients' base/top mapping and four refusals, solve_convex's agreement
with a brute-force grid optimum, its handling of a duplicate (collinear) member, its weight constraints, and its
KKT optimality check forced to fire via a patched optimiser, the solved combination's exact-vs-true-MSE identity,
build_values's unmatched/matched split, the floor-at-1, the rounding-energy bookkeeping and two refusals, a full
truth-based decomposition of the built file's true MSE, move_report's hand-computed numbers, plan's two refusals
(via a patched move_report) and its output contract (via a genuine small multi-member solve), write_submission's
template-shape fidelity, id-reordering, overwrite refusal and lossy-cast refusal, and the CLI's
eligibility -> plan -> build round trip including build's three drift refusals and its never-overwrite guarantee.
Section 17 is a real-data check against the actual board-scored submission files; it skips with a written reason
when those files are absent.

Fixtures never go through `member_path` (so a bug in that function cannot hide itself from a test that uses it to
both write and read a fixture); every member parquet and result json is written with the literal real names
("merry-quicksand_vN.parquet", "..._result.json") and the real dtypes (id float64, prediction int32).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BE = _load("blend_exact", "scripts/analysis/blend_exact.py")


# ==================================================================================== shared fixture helpers

def _truth(n, seed=0):
    """A gamma-like positive 'seconds' truth vector. The tool under test never reads this; it exists only so the
    tests can check the tool's output against an independent ground truth."""
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=2.0, scale=250.0, size=n) + 30.0


def _write_member(sub_dir, version, ids, pred):
    """Writes one member parquet with the real column names and dtypes, in whatever row order `ids`/`pred` are
    given (callers shuffle to exercise load_members's sort-by-id)."""
    path = sub_dir / f"merry-quicksand_v{version}.parquet"
    pd.DataFrame({"MVT_ID_mvt": np.asarray(ids, dtype=np.float64),
                 "TAXITIME_SEC_mvt": np.asarray(pred, dtype=np.int32)}).to_parquet(path, index=False)
    return path


def _write_result(sub_dir, version, score, used_pairs, status="Succeeded", file=None):
    path = sub_dir / f"merry-quicksand_v{version}.parquet_result.json"
    payload = {"status": status,
              "file": f"merry-quicksand_v{version}.parquet" if file is None else file,
              "bucket": "test-bucket", "used_pairs": used_pairs, "inputs": {}, "score": score}
    path.write_text(json.dumps(payload))
    return path


def _members_world(tmp_path, n, members, seed=0):
    """members: {version: (own_scale, shared_weight)}. One shared noise term (fixed by `seed`) feeds every member
    at its own `shared_weight`, plus independent noise at `own_scale`, so members correlate through the shared
    term and differ in quality by scale -- the brief's 'different scales and a shared component'. Predictions are
    rounded to int32 (the real on-disk dtype) and floored at 1 before writing. Each member's file is written in
    its own shuffled row order with a result.json holding the TRUE (unrounded) rmse against `y`.
    Returns (sub_dir, y, ids, pred {version: int32 array in ids order}, rmse {version: float})."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    y = _truth(n, seed)
    ids = np.arange(5_000_000, 5_000_000 + n, dtype=np.int64)
    shared = np.random.default_rng(seed * 7 + 1).normal(0.0, 40.0, size=n)
    pred, rmse = {}, {}
    for version, (scale, shared_weight) in members.items():
        rng = np.random.default_rng(seed * 7 + 2 + version)
        raw = y + shared_weight * shared + rng.normal(0.0, scale, size=n)
        p = np.rint(np.maximum(raw, 1.0)).astype(np.int32)
        pred[version] = p
        r = float(np.sqrt(np.mean((p.astype(np.float64) - y) ** 2)))
        rmse[version] = r
        shuf = np.random.default_rng(seed * 7 + 9000 + version).permutation(n)
        _write_member(sub_dir, version, ids[shuf], p[shuf])
        _write_result(sub_dir, version, r, n)
    return sub_dir, y, ids, pred, rmse


def _synthetic_mse_d(n, versions, order, seed=0):
    """Pure in-memory fixture for solve_convex's array-level tests (no parquet/json files): independent noise per
    version (no shared term -- the shared-component fixture lives in `_members_world` and is used wherever member
    files are written). Returns (pred {version: float64}, mse {version: float}, D)."""
    y = _truth(n, seed)
    pred = {}
    for i, v in enumerate(versions):
        rng = np.random.default_rng(seed * 13 + v)
        pred[v] = y + rng.normal(0.0, 50.0 + 30.0 * i, size=n)
    mse = {v: float(np.mean((pred[v] - y) ** 2)) for v in versions}
    D = BE.pair_d(pred, order)
    return y, pred, mse, D


# ==================================================================================== member_path

def test_member_path_builds_the_standard_filename(tmp_path):
    """Fails when member_path stops joining sub_dir with the exact real filename pattern. Mutation: change the
    f-string to drop the leading 'v' (e.g. f"merry-quicksand_{version}.parquet")."""
    got = BE.member_path(tmp_path, 57)
    assert got == tmp_path / "merry-quicksand_v57.parquet"
    assert isinstance(got, pathlib.Path)


# ==================================================================================== 1-2: exact_mse identity

def test_exact_mse_matches_true_mse_for_random_coefficients_including_negative(tmp_path):
    """Fails when the 0.5 factor or the sign of the D term in exact_mse's formula is wrong (for ANY random
    affine combination, including one with a negative coefficient, the board formula must equal the true MSE
    computed directly against the truth, to 1e-9 relative). Mutation: change `- 0.5 * a @ D @ a` to
    `+ 0.5 * a @ D @ a`, or to `- a @ D @ a` (drop the 0.5)."""
    sub_dir, y, ids, pred, rmse = _members_world(
        tmp_path, 4000, {1: (60.0, 1.0), 2: (90.0, 0.6), 3: (120.0, 0.3), 4: (70.0, 0.8)}, seed=11)
    order = [1, 2, 3, 4]
    mse = {v: r ** 2 for v, r in rmse.items()}
    D = BE.pair_d(pred, order)

    rng = np.random.default_rng(2026)
    for _ in range(5):
        raw = rng.normal(0.0, 1.0, size=4)
        raw[0] -= 2.0   # bias one coefficient negative so the combination is a genuine affine (not convex) blend
        a = raw / raw.sum()
        coef = dict(zip(order, a))
        val = BE.exact_mse(coef, mse, D, order)
        full = sum(coef[v] * pred[v].astype(np.float64) for v in order)
        true_mse = float(np.mean((full - y) ** 2))
        assert abs(val - true_mse) <= 1e-9 * max(1.0, abs(true_mse))


def test_exact_mse_with_rounded_scores_is_within_the_score_rounding_bound(tmp_path):
    """Fails when exact_mse computed from 4-decimal-rounded board scores drifts from the true MSE by more than
    _score_rounding_bound predicts (the bound: each member's MSE error from rounding its rmse to SCORE_DECIMALS
    decimals is at most 2*rmse*0.5*10**-SCORE_DECIMALS; summed with |coefficient| weights). Mutation: round the
    bound's half-ULP factor from 0.5 to 1.0 (or drop it), making the predicted bound too tight to hold."""
    sub_dir, y, ids, pred, rmse_true = _members_world(
        tmp_path, 4000, {1: (60.0, 1.0), 2: (90.0, 0.6), 3: (55.0, 0.9)}, seed=12)
    order = [1, 2, 3]
    rmse_rounded = {v: round(r, BE.SCORE_DECIMALS) for v, r in rmse_true.items()}
    mse_rounded = {v: r ** 2 for v, r in rmse_rounded.items()}
    D = BE.pair_d(pred, order)

    rng = np.random.default_rng(99)
    for _ in range(5):
        raw = rng.normal(0.0, 1.0, size=3)
        a = raw / raw.sum()
        coef = dict(zip(order, a))
        val_rounded = BE.exact_mse(coef, mse_rounded, D, order)
        full = sum(coef[v] * pred[v].astype(np.float64) for v in order)
        true_mse = float(np.mean((full - y) ** 2))
        bound = BE._score_rounding_bound(coef, rmse_rounded)
        assert abs(val_rounded - true_mse) <= bound + 1e-9


# ==================================================================================== 3: exact_mse refusals

def test_exact_mse_refuses_coefficients_not_summing_to_one():
    """Fails when exact_mse accepts coefficients that do not sum to 1, or raises the wrong message. Mutation:
    change the sum tolerance check from `> 1e-9` to `> 1e-3` (hiding this exact violation)."""
    mse = {1: 100.0, 2: 200.0}
    D = np.array([[0.0, 10.0], [10.0, 0.0]])
    with pytest.raises(ValueError, match="coefficients must sum to 1"):
        BE.exact_mse({1: 0.5, 2: 0.6}, mse, D, [1, 2])


def test_exact_mse_refuses_a_key_outside_order():
    """Fails when exact_mse silently ignores (or crashes ungracefully on) a coefficient key that is not in
    `order`, instead of raising ValueError. Mutation: delete the `bad = set(coef) - set(order)` check."""
    mse = {1: 100.0, 2: 200.0}
    D = np.array([[0.0, 10.0], [10.0, 0.0]])
    with pytest.raises(ValueError, match="not in order"):
        BE.exact_mse({1: 0.5, 3: 0.5}, mse, D, [1, 2])


# ==================================================================================== 4: pair_d

def test_pair_d_matches_direct_computation_and_is_symmetric_with_zero_diagonal():
    """Fails when pair_d's entries do not equal mean((X[i]-X[j])**2) computed directly, or when it is not exactly
    symmetric with a zero diagonal. Mutation: swap the stored (i, j) / (j, i) assignment so D is no longer
    symmetric, or compute mean(X[i]-X[j])**2 instead of mean((X[i]-X[j])**2)."""
    rng = np.random.default_rng(7)
    X = {10: rng.normal(100.0, 20.0, size=500), 20: rng.normal(100.0, 20.0, size=500),
        30: rng.normal(100.0, 20.0, size=500)}
    order = [10, 20, 30]
    D = BE.pair_d(X, order)
    assert D.shape == (3, 3)
    for i in range(3):
        assert D[i, i] == 0.0
        for j in range(3):
            direct = float(np.mean((X[order[i]] - X[order[j]]) ** 2))
            assert abs(D[i, j] - direct) < 1e-9
            assert D[i, j] == D[j, i]


# ==================================================================================== 5: load_members

def test_load_members_aligns_shuffled_files_by_id(tmp_path):
    """Fails when load_members does not sort each file by id before returning, so two files written in different
    row orders come back misaligned. Mutation: delete the `np.argsort` step (return rows in on-disk order)."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    ids = np.array([103, 101, 104, 102, 100], dtype=np.int64)
    pred_a = np.array([50, 10, 60, 20, 0], dtype=np.int32)      # pred_a[k] belongs to ids[k]
    pred_b = np.array([550, 510, 560, 520, 500], dtype=np.int32)
    order_a = [4, 1, 3, 0, 2]   # a different shuffle per file
    order_b = [0, 2, 4, 1, 3]
    _write_member(sub_dir, 1, ids[order_a], pred_a[order_a])
    _write_member(sub_dir, 2, ids[order_b], pred_b[order_b])

    out_ids, X = BE.load_members(sub_dir, [1, 2])
    assert list(out_ids) == [100, 101, 102, 103, 104]
    assert out_ids.dtype == np.int64
    # row k of X[1] / X[2] must both belong to out_ids[k], i.e. to the SAME id
    expect_a = {100: 0, 101: 10, 102: 20, 103: 50, 104: 60}
    expect_b = {100: 500, 101: 510, 102: 520, 103: 550, 104: 560}
    for k, i in enumerate(out_ids):
        assert X[1][k] == expect_a[int(i)]
        assert X[2][k] == expect_b[int(i)]


def test_load_members_refuses_a_missing_file(tmp_path):
    """Fails when load_members does not raise ValueError naming the missing file, e.g. if it let pandas raise its
    own (unnamed) FileNotFoundError instead. Mutation: delete the `if not path.exists()` check."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    missing = sub_dir / "merry-quicksand_v1.parquet"
    with pytest.raises(ValueError, match=str(missing).replace("\\", "\\\\")):
        BE.load_members(sub_dir, [1])


def test_load_members_refuses_a_null_id(tmp_path):
    """Fails when load_members does not catch a null id before casting to int64 (a NaN->int64 cast is undefined
    and would silently corrupt the id). Mutation: delete the `np.isnan(id_col).any()` check."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    path = sub_dir / "merry-quicksand_v1.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, np.nan, 3.0], "TAXITIME_SEC_mvt": np.array([10, 20, 30], dtype=np.int32)}
                ).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="null id"):
        BE.load_members(sub_dir, [1])


def test_load_members_refuses_a_non_integer_id(tmp_path):
    """Fails when load_members does not catch a fractional id before casting to int64 (ids are stored as doubles;
    a non-integer id would silently truncate under `.astype(np.int64)` instead of being refused). Mutation: delete
    the `np.array_equal(id_col, np.trunc(id_col))` check."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    path = sub_dir / "merry-quicksand_v1.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, 2.5, 3.0], "TAXITIME_SEC_mvt": np.array([10, 20, 30], dtype=np.int32)}
                ).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="non-integer id"):
        BE.load_members(sub_dir, [1])


def test_load_members_refuses_a_duplicated_id(tmp_path):
    """Fails when load_members does not catch two rows sharing one id within a single file. Mutation: delete the
    `len(np.unique(ids_i64)) != len(ids_i64)` check."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    path = sub_dir / "merry-quicksand_v1.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 2.0], "TAXITIME_SEC_mvt": np.array([10, 20, 30], dtype=np.int32)}
                ).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate id"):
        BE.load_members(sub_dir, [1])


def test_load_members_refuses_a_non_finite_prediction(tmp_path):
    """Fails when load_members does not catch an inf/nan prediction value. Mutation: delete the
    `np.isfinite(pred).all()` check."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    path = sub_dir / "merry-quicksand_v1.parquet"
    # TAXITIME_SEC_mvt must stay castable: use a float column to inject a non-finite value past int32 storage.
    pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "TAXITIME_SEC_mvt": [10.0, np.inf, 30.0]}).to_parquet(
        path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        BE.load_members(sub_dir, [1])


def test_load_members_refuses_a_different_id_set(tmp_path):
    """Fails when load_members does not catch a second file whose id set does not exactly match the first file's.
    Mutation: delete the `np.array_equal(ids_sorted, ids_ref)` check."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    _write_member(sub_dir, 1, [1, 2, 3], [10, 20, 30])
    _write_member(sub_dir, 2, [1, 2, 4], [10, 20, 40])   # id 4 instead of 3
    with pytest.raises(ValueError, match="does not match"):
        BE.load_members(sub_dir, [1, 2])


# ==================================================================================== 6: load_scores

def test_load_scores_happy_path_returns_rmse_mse_and_result_sha256(tmp_path):
    """Fails when load_scores does not correctly return {rmse, mse=rmse**2, result_sha256} per version, or
    computes result_sha256 over the wrong bytes. Mutation: change `rmse ** 2` to `rmse * 2`."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    path = _write_result(sub_dir, 5, 123.4567, 1000)
    out = BE.load_scores(sub_dir, [5], 1000)
    assert out[5]["rmse"] == 123.4567
    assert abs(out[5]["mse"] - 123.4567 ** 2) < 1e-9
    import hashlib
    assert out[5]["result_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_scores_refuses_a_missing_result_file(tmp_path):
    """Fails when load_scores does not raise ValueError for a missing result.json. Mutation: delete the
    `if not result_path.exists()` check."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (sub_dir / "merry-quicksand_v5.parquet").write_bytes(b"")
    with pytest.raises(ValueError, match="missing result file"):
        BE.load_scores(sub_dir, [5], 1000)


def test_load_scores_refuses_a_non_succeeded_status(tmp_path):
    """Fails when load_scores accepts a result whose status is not 'Succeeded'. Mutation: delete the status
    check."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    _write_result(sub_dir, 5, 100.0, 1000, status="Failed")
    with pytest.raises(ValueError, match="Failed"):
        BE.load_scores(sub_dir, [5], 1000)


def test_load_scores_refuses_a_wrong_used_pairs(tmp_path):
    """Fails when load_scores accepts a result whose used_pairs does not equal n_rows. Mutation: delete the
    used_pairs check."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    _write_result(sub_dir, 5, 100.0, 999)
    with pytest.raises(ValueError, match="999"):
        BE.load_scores(sub_dir, [5], 1000)


def test_load_scores_refuses_a_wrong_file_name(tmp_path):
    """Fails when load_scores accepts a result whose 'file' field names a different file than the member's own.
    Mutation: delete the file-name check."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    _write_result(sub_dir, 5, 100.0, 1000, file="merry-quicksand_v6.parquet")
    with pytest.raises(ValueError, match="merry-quicksand_v6.parquet"):
        BE.load_scores(sub_dir, [5], 1000)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1.0, 0.0])
def test_load_scores_refuses_a_non_finite_or_non_positive_score(tmp_path, score):
    """Fails when load_scores accepts a score that is not a finite, strictly positive number (nan, inf, negative,
    or zero). Mutation: change the check from `> 0` to `>= 0` (accepting zero)."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    _write_result(sub_dir, 5, score, 1000)
    with pytest.raises(ValueError, match="finite positive"):
        BE.load_scores(sub_dir, [5], 1000)


@pytest.mark.parametrize("score", [True, False, "100.0", None])
def test_load_scores_refuses_a_bool_or_non_numeric_score(tmp_path, score):
    """Fails when load_scores accepts a bool score -- in Python `isinstance(True, (int, float))` is True, so
    without an explicit exclusion `True` would silently become mse=1.0 -- or any score that is not an int/float at
    all. Mutation: delete the `not isinstance(score, bool)` half of the `is_number` guard, so a bare
    `isinstance(score, (int, float))` lets True/False back in."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    _write_result(sub_dir, 5, score, 1000)
    with pytest.raises(ValueError, match="finite positive"):
        BE.load_scores(sub_dir, [5], 1000)


# ==================================================================================== 7: eligibility

def test_eligibility_statistics_match_direct_computation_on_a_hand_made_case():
    """Fails when eligibility's n_diff / d / top20_share / max_abs_diff do not match a hand-computable case.
    20 rows differ by 1000 (dominant), 980 rows differ by 20 (body), 4000 rows identical. Mutation: compute
    top20_share as the 20 largest raw differences (not squared)."""
    n = 5000
    base = np.full(n, 1000.0)
    other = base.copy()
    other[:20] += 1000.0     # 20 big movers
    other[20:1000] += 20.0   # 980 small movers
    # remaining 4000 rows identical
    X = {1: base, 2: other}
    out = BE.eligibility(X, 1, [1, 2])
    assert 1 not in out                      # the base itself is not in the result
    e = out[2]
    assert e["n_diff"] == 1000
    assert abs(e["d"] - np.mean((other - base) ** 2)) < 1e-9
    top20_sq = 20 * 1000.0 ** 2
    total_sq = 20 * 1000.0 ** 2 + 980 * 20.0 ** 2
    assert abs(e["top20_share"] - top20_sq / total_sq) < 1e-12
    assert e["max_abs_diff"] == 1000.0


def test_eligibility_all_identical_member_has_zero_top20_share_and_fails_n_diff():
    """Fails when eligibility divides by zero (or otherwise mishandles) an all-identical member instead of
    returning top20_share == 0.0. Mutation: delete the `if total == 0` guard in the shared top20-share helper."""
    base = np.full(2000, 500.0)
    X = {1: base, 2: base.copy()}
    out = BE.eligibility(X, 1, [1, 2])
    e = out[2]
    assert e["n_diff"] == 0
    assert e["d"] == 0.0
    assert e["top20_share"] == 0.0
    assert e["max_abs_diff"] == 0.0
    assert e["ok"] is False
    assert e["why"] == ["n_diff"]


def test_eligibility_a_body_wide_different_member_is_ok():
    """Fails when a realistic body-wide member (every row differs by a modest, spread-out amount, as real
    submission files do) is wrongly marked ineligible, e.g. if n_diff were computed by counting agreements
    instead of disagreements. Mutation: compute n_diff via `X[v] == X[base]` (count agreements, not
    disagreements)."""
    n = 150_000
    base = np.full(n, 1000.0)
    other = base + (np.arange(n) % 300) + 1.0   # every row differs, spread 1..300
    X = {1: base, 2: other}
    out = BE.eligibility(X, 1, [1, 2])
    e = out[2]
    assert e["n_diff"] == n
    assert e["ok"] is True
    assert e["why"] == []


def test_eligibility_few_differing_rows_fails_only_n_diff():
    """Fails when a member differing on far fewer than MIN_ROWS rows is marked eligible, or when it also
    (wrongly) fails top20_share/max_abs_diff due to a fixture or formula bug. Mutation: change `n_diff >= MIN_ROWS`
    to `n_diff >= 0` (always true)."""
    n = 150_000
    base = np.full(n, 1000.0)
    other = base.copy()
    other[:500] += 20.0   # 500 << MIN_ROWS rows differ, by a small, non-concentrated, non-huge amount
    X = {1: base, 2: other}
    e = BE.eligibility(X, 1, [1, 2])[2]
    assert e["n_diff"] == 500
    assert e["ok"] is False
    assert e["why"] == ["n_diff"]


def test_eligibility_concentrated_moves_fail_only_top20_share():
    """Fails when a member with enough differing rows (passes n_diff) and no single huge move (passes
    max_abs_diff) but whose error is concentrated in 20 rows is wrongly marked eligible. Mutation: change
    `top20_share <= MAX_TOP20` to `top20_share <= 1.0` (always true)."""
    n = 150_000
    base = np.full(n, 1000.0)
    other = base + 20.0          # every row moves by a small amount (passes n_diff, body of the distribution)
    other[:20] = base[:20] + 1000.0   # those 20 rows instead carry a dominant move (1000, still <= BIG_S)
    X = {1: base, 2: other}
    e = BE.eligibility(X, 1, [1, 2])[2]
    assert e["n_diff"] == n
    assert e["max_abs_diff"] == 1000.0
    assert e["top20_share"] > BE.MAX_TOP20   # 20*1000^2 / (20*1000^2 + 149980*20^2) = 0.250.., verified by hand
    assert e["ok"] is False
    assert e["why"] == ["top20_share"]


def test_eligibility_a_single_huge_move_fails_only_max_abs_diff():
    """Fails when a member with one move bigger than BIG_S (a rule, not a model, per the brief) is wrongly marked
    eligible even though its n_diff and top20_share both pass. Mutation: change `max_abs_diff <= BIG_S` to
    `max_abs_diff <= BIG_S * 10` (too permissive)."""
    n = 150_000
    base = np.full(n, 1000.0)
    other = base.copy()
    other[:n] += 100.0            # every row moves modestly (n_diff passes; keeps top20_share small later)
    other[0] = base[0] + 7200.0   # one 2-hour move, far beyond BIG_S = 3600
    X = {1: base, 2: other}
    e = BE.eligibility(X, 1, [1, 2])[2]
    assert e["n_diff"] == n
    assert e["max_abs_diff"] == 7200.0
    assert e["top20_share"] <= BE.MAX_TOP20   # isolates max_abs_diff: this must NOT also fail top20_share
    assert e["ok"] is False
    assert e["why"] == ["max_abs_diff"]


def test_eligibility_every_condition_can_fail_at_once():
    """Fails when `why` drops a failed condition when more than one fails simultaneously. Mutation: change `why`
    construction to `return [first_failed]` (stop at the first)."""
    n = 50          # tiny: guarantees n_diff fails too
    base = np.full(n, 1000.0)
    other = base.copy()
    other[:n] += 50.0
    other[0] = base[0] + 7200.0   # also trips max_abs_diff; the 50-row body also trips top20_share (n is tiny)
    X = {1: base, 2: other}
    e = BE.eligibility(X, 1, [1, 2])[2]
    assert e["ok"] is False
    assert set(e["why"]) == {"n_diff", "top20_share", "max_abs_diff"}


@pytest.mark.parametrize("n_diff, want_ok", [(99_999, False), (100_000, True)])
def test_eligibility_ok_n_diff_boundary_is_inclusive(n_diff, want_ok):
    """Fails when the n_diff boundary is not inclusive (>= MIN_ROWS), i.e. exactly MIN_ROWS must pass. Tests the
    private comparison helper directly at the literal threshold value, so no fixture-construction float noise can
    hide an off-by-one. Mutation: change `n_diff >= MIN_ROWS` to `n_diff > MIN_ROWS`."""
    ok, why = BE._eligibility_ok(n_diff, 0.01, 10.0)
    assert ok is want_ok
    assert ("n_diff" in why) is (not want_ok)


@pytest.mark.parametrize("top20_share, want_ok", [(0.05, True), (0.0500001, False)])
def test_eligibility_ok_top20_share_boundary_is_inclusive(top20_share, want_ok):
    """Fails when the top20_share boundary is not inclusive (<= MAX_TOP20). Mutation: change
    `top20_share <= MAX_TOP20` to `top20_share < MAX_TOP20`."""
    ok, why = BE._eligibility_ok(BE.MIN_ROWS, top20_share, 10.0)
    assert ok is want_ok
    assert ("top20_share" in why) is (not want_ok)


@pytest.mark.parametrize("max_abs_diff, want_ok", [(3600.0, True), (3600.0001, False)])
def test_eligibility_ok_max_abs_diff_boundary_is_inclusive(max_abs_diff, want_ok):
    """Fails when the max_abs_diff boundary is not inclusive (<= BIG_S). Mutation: change
    `max_abs_diff <= BIG_S` to `max_abs_diff < BIG_S`."""
    ok, why = BE._eligibility_ok(BE.MIN_ROWS, 0.01, max_abs_diff)
    assert ok is want_ok
    assert ("max_abs_diff" in why) is (not want_ok)


def test_eligibility_ok_all_pass_returns_true_and_empty_why():
    """Fails when the all-pass case returns a non-empty why list or ok is not a plain bool. Mutation: initialise
    `why` with a stray entry that is never cleared."""
    ok, why = BE._eligibility_ok(BE.MIN_ROWS, BE.MAX_TOP20, BE.BIG_S)
    assert ok is True
    assert why == []
    assert isinstance(ok, bool)   # not np.bool_, so json.dumps works without a custom encoder


# ==================================================================================== 8: coefficients

def test_coefficients_maps_weights_to_combination_coefficients():
    """Fails when coefficients does not compute a_base = w_base - 1 and a_top = 1 exactly, or mis-maps the other
    pool members. Mutation: change `weights[base] - 1.0` to `weights[base]` (drop the -1)."""
    coef = BE.coefficients({10: 0.3, 20: 0.7}, base=10, top=99)
    assert coef[10] == pytest.approx(0.3 - 1.0)
    assert coef[20] == pytest.approx(0.7)
    assert coef[99] == 1.0
    assert abs(sum(coef.values()) - 1.0) < 1e-12


def test_coefficients_drops_exact_zero_pool_weights_but_keeps_base_and_top():
    """Fails when coefficients keeps a pool member with exactly zero weight (should be dropped), or drops base
    even though base's own coefficient can legitimately be zero (w_base == 1). Mutation: drop base/top whenever
    their coefficient is 0, or keep every zero-weight pool member."""
    coef = BE.coefficients({10: 1.0, 20: 0.0, 30: 0.0}, base=10, top=99)
    assert 20 not in coef
    assert 30 not in coef
    assert coef[10] == 0.0          # w_base - 1 == 0, but base is still present
    assert coef[99] == 1.0


def test_coefficients_refuses_when_base_is_missing():
    """Fails when coefficients does not raise ValueError for a weights dict missing the base. Mutation: delete
    the `base not in weights` check."""
    with pytest.raises(ValueError, match="base"):
        BE.coefficients({20: 1.0}, base=10, top=99)


def test_coefficients_refuses_when_top_is_present():
    """Fails when coefficients does not raise ValueError when `top` appears in the weights dict (top is never a
    pool member). Mutation: delete the `top in weights` check."""
    with pytest.raises(ValueError, match="top"):
        BE.coefficients({10: 0.5, 99: 0.5}, base=10, top=99)


def test_coefficients_refuses_a_negative_weight():
    """Fails when coefficients accepts a weight below -1e-12. Mutation: change the negativity check's threshold
    from -1e-12 to -1.0 (too permissive)."""
    with pytest.raises(ValueError, match="negative"):
        BE.coefficients({10: 1.1, 20: -0.1}, base=10, top=99)


def test_coefficients_refuses_a_sum_off_from_one():
    """Fails when coefficients accepts weights summing to something more than 1e-9 away from 1. Mutation: change
    the sum tolerance from 1e-9 to 1e-1."""
    with pytest.raises(ValueError, match="sum"):
        BE.coefficients({10: 0.5, 20: 0.4}, base=10, top=99)


# ==================================================================================== 9-10: solve_convex

def test_solve_convex_matches_a_brute_force_grid_optimum():
    """Fails when solve_convex's objective is worse than the best point found by an independent 0.01-resolution
    grid search over the 3-member simplex. Mutation: break the polish step's KKT linear system (e.g. swap the
    sign of the `1.0` constraint row), which would silently install a non-optimal polished solution."""
    pool, base, top = [1, 2, 3], 1, 4
    order = pool + [top]
    y, pred, mse, D = _synthetic_mse_d(600, [1, 2, 3, 4], order, seed=3)
    w = BE.solve_convex(mse, D, order, pool, base, top)
    assert set(w.keys()) == set(pool)
    assert all(wi >= -1e-9 for wi in w.values())
    assert abs(sum(w.values()) - 1.0) < 1e-9
    obj = BE.exact_mse(BE.coefficients(w, base, top), mse, D, order)

    best = None
    step = 0.01
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        w1 = round(i * step, 10)
        for j in range(n_steps + 1 - i):
            w2 = round(j * step, 10)
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            cand = {1: w1, 2: w2, 3: w3}
            val = BE.exact_mse(BE.coefficients(cand, base, top), mse, D, order)
            if best is None or val < best:
                best = val
    assert obj <= best + 1e-9


def test_solve_convex_identical_copy_member_does_not_change_the_optimum():
    """Fails when adding an exact duplicate of an existing pool member (a collinear D row/column, the kind of
    thing np.linalg.lstsq must absorb in the polish step) changes the achievable objective, or breaks the solve
    outright (e.g. np.linalg.solve on a singular matrix would raise). Mutation: replace the polish step's
    np.linalg.lstsq with np.linalg.solve."""
    pool, base, top = [1, 2], 1, 3
    order = pool + [top]
    y, pred, mse, D = _synthetic_mse_d(600, [1, 2, 3], order, seed=4)
    w_nodup = BE.solve_convex(mse, D, order, pool, base, top)
    obj_nodup = BE.exact_mse(BE.coefficients(w_nodup, base, top), mse, D, order)

    pred_dup = dict(pred)
    pred_dup[99] = pred[2].copy()
    mse_dup = dict(mse)
    mse_dup[99] = mse[2]
    pool_dup = [1, 2, 99]
    order_dup = pool_dup + [top]
    D_dup = BE.pair_d(pred_dup, order_dup)
    w_dup = BE.solve_convex(mse_dup, D_dup, order_dup, pool_dup, base, top)
    obj_dup = BE.exact_mse(BE.coefficients(w_dup, base, top), mse_dup, D_dup, order_dup)

    assert all(wi >= -1e-9 for wi in w_dup.values())
    assert abs(sum(w_dup.values()) - 1.0) < 1e-9
    assert abs(obj_dup - obj_nodup) < 1e-6
    assert abs((w_dup[2] + w_dup[99]) - w_nodup[2]) < 1e-4


def test_solve_convex_raises_when_the_optimiser_does_not_move_from_the_start(monkeypatch):
    """Fails when solve_convex accepts a non-optimal solution instead of raising RuntimeError. Patches
    scipy.optimize.minimize (imported at module scope precisely so this is possible, matching this codebase's own
    convention of patching a module-level dependency) to return a point that collapses all weight onto the base
    alone, ignoring two other pool members that genuinely help here (verified numerically: member 3's gradient is
    below the active set's lambda by far more than KKT_TOL). Mutation: delete the
    `if not ok: raise RuntimeError(...)` at the end of solve_convex."""
    pool, base, top = [1, 2, 3], 1, 4
    order = pool + [top]
    y, pred, mse, D = _synthetic_mse_d(500, [1, 2, 3, 4], order, seed=5)

    class _FakeResult:
        def __init__(self, x):
            self.x = np.asarray(x, dtype=np.float64)

    def _fake_minimize(fun, x0, **kwargs):
        return _FakeResult([1.0 - 2e-7, 1e-7, 1e-7])

    monkeypatch.setattr(BE, "minimize", _fake_minimize)
    with pytest.raises(RuntimeError, match="solve_convex: the solver did not reach the optimum"):
        BE.solve_convex(mse, D, order, pool, base, top)


def test_solve_convex_solution_exact_mse_equals_true_mse_on_truth():
    """Fails when the weights solve_convex returns, once mapped through coefficients and applied to the member
    arrays, do not achieve the exact_mse value it implies -- catching e.g. a pool/weight ordering mismatch between
    solve_convex's internal variable order and its returned dict's keys. Mutation: zip `pool` with the weight
    array in a different (e.g. sorted) order before building the returned dict."""
    pool, base, top = [1, 2], 1, 3
    order = pool + [top]
    y, pred, mse, D = _synthetic_mse_d(700, [1, 2, 3], order, seed=6)
    w = BE.solve_convex(mse, D, order, pool, base, top)
    coef = BE.coefficients(w, base, top)
    reported = BE.exact_mse(coef, mse, D, order)
    full = sum(coef[v] * pred[v] for v in coef)
    true_mse = float(np.mean((full - y) ** 2))
    assert abs(reported - true_mse) < 1e-6


# ==================================================================================== 11: build_values

def test_build_values_unmatched_rows_equal_the_top_file_exactly():
    """Fails when an unmatched row's built value is anything other than the top file's own value, even when the
    full combination would have produced something different there. Mutation: change
    `np.where(is_unmatched, X[top], ...)` to always use `full` (ignore is_unmatched)."""
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    X = {10: np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        20: np.array([150.0, 250.0, 350.0, 450.0, 550.0])}   # top
    coef = {10: 0.5, 20: 0.5}
    unmatched = np.array([2, 4], dtype=np.int64)
    out = BE.build_values(ids, X, coef, top=20, unmatched=unmatched)
    assert out["values"][1] == X[20][1]   # id 2 -> index 1
    assert out["values"][3] == X[20][3]   # id 4 -> index 3
    assert list(out["is_unmatched"]) == [False, True, False, True, False]


def test_build_values_matched_rows_equal_rint_of_max_full_and_one():
    """Fails when a matched row's built value is not exactly round-half-to-even(max(full, 1)). Mutation: change
    `np.rint(np.maximum(full, 1.0))` to `np.floor(np.maximum(full, 1.0))`."""
    ids = np.array([1, 2, 3], dtype=np.int64)
    X = {10: np.array([10.3, 10.7, 11.5]), 20: np.array([0.0, 0.0, 0.0])}
    coef = {10: 1.0, 20: 0.0}
    unmatched = np.array([], dtype=np.int64)
    out = BE.build_values(ids, X, coef, top=20, unmatched=unmatched)
    assert list(out["values"]) == [10.0, 11.0, 12.0]   # rint: 10.3->10, 10.7->11, 11.5->12 (round-half-to-even)


def test_build_values_floors_matched_rows_at_one():
    """Fails when a matched row whose full combination is below 1 is not floored up to 1. Mutation: change
    `np.maximum(full, 1.0)` to `full` (no floor)."""
    ids = np.array([1, 2], dtype=np.int64)
    X = {10: np.array([-5.0, 0.3]), 20: np.array([0.0, 0.0])}
    coef = {10: 1.0, 20: 0.0}
    unmatched = np.array([], dtype=np.int64)
    out = BE.build_values(ids, X, coef, top=20, unmatched=unmatched)
    assert list(out["values"]) == [1.0, 1.0]
    assert out["n_floored"] == 2


def test_build_values_e_g_rounding_energy_and_n_floored_match_direct_computation():
    """Fails when e_g, rounding_energy or n_floored diverge from their own stated formulas, including e_g being
    computed over the wrong row set (all rows, or the matched rows, instead of only the unmatched ones).
    Mutation: compute e_g over `~is_unmatched` instead of `is_unmatched`."""
    ids = np.array([1, 2, 3, 4], dtype=np.int64)
    # member 10 (base, weight contributes via coef) and 20 (top)
    X = {10: np.array([100.0, 0.5, 100.0, 300.0]), 20: np.array([140.0, 0.0, 142.0, 0.0])}
    coef = {10: 0.5, 20: 0.5}
    unmatched = np.array([1, 4], dtype=np.int64)   # ids 1 and 4 -> indices 0, 3
    out = BE.build_values(ids, X, coef, top=20, unmatched=unmatched)
    full = 0.5 * X[10] + 0.5 * X[20]
    is_un = np.array([True, False, False, True])
    n = len(ids)
    expect_e_g = float(np.sum((full - X[20])[is_un] ** 2) / n)
    expect_values = np.where(is_un, X[20], np.rint(np.maximum(full, 1.0)))
    expect_rounding_energy = float(np.sum((expect_values - full)[~is_un] ** 2) / n)
    expect_n_floored = int(np.sum((full < 1.0) & ~is_un))
    assert abs(out["e_g"] - expect_e_g) < 1e-9
    assert abs(out["rounding_energy"] - expect_rounding_energy) < 1e-9
    assert out["n_floored"] == expect_n_floored
    assert list(out["values"]) == list(expect_values)


def test_build_values_not_applied_describes_the_unmatched_gap():
    """Fails when not_applied's n_nonzero / n_over_30s / max_abs_s do not describe (full - top) on exactly the
    unmatched rows. Mutation: compute not_applied over all rows instead of only the unmatched ones."""
    ids = np.array([1, 2, 3], dtype=np.int64)
    X = {10: np.array([1000.0, 1000.0, 1000.0]), 20: np.array([1000.0, 960.0, 999.0])}
    coef = {10: 1.0, 20: 0.0}   # full == X[10] == [1000,1000,1000] everywhere
    unmatched = np.array([2, 3], dtype=np.int64)   # indices 1, 2: full-top = [40, 1]
    out = BE.build_values(ids, X, coef, top=20, unmatched=unmatched)
    na = out["not_applied"]
    assert na["n_nonzero"] == 2
    assert na["n_over_30s"] == 1
    assert na["max_abs_s"] == 40.0


def test_build_values_not_applied_is_zero_when_there_are_no_unmatched_rows():
    """Fails when not_applied's max_abs_s raises (np.max on an empty array) instead of defaulting to 0.0 when
    there are no unmatched rows. Mutation: delete the `if diff_un.size else 0.0` guard."""
    ids = np.array([1, 2], dtype=np.int64)
    X = {10: np.array([100.0, 200.0]), 20: np.array([110.0, 210.0])}
    coef = {10: 1.0, 20: 0.0}
    out = BE.build_values(ids, X, coef, top=20, unmatched=np.array([], dtype=np.int64))
    assert out["not_applied"] == {"n_nonzero": 0, "n_over_30s": 0, "max_abs_s": 0.0}


def test_build_values_refuses_a_coefficient_sum_off_from_one():
    """Fails when build_values accepts coefficients that do not sum to 1. Mutation: delete the coefficient-sum
    check in build_values."""
    ids = np.array([1, 2], dtype=np.int64)
    X = {10: np.array([100.0, 200.0]), 20: np.array([110.0, 210.0])}
    with pytest.raises(ValueError, match="coefficients must sum to 1"):
        BE.build_values(ids, X, {10: 0.4, 20: 0.4}, top=20, unmatched=np.array([], dtype=np.int64))


def test_build_values_refuses_an_unmatched_id_not_in_ids():
    """Fails when build_values accepts an unmatched id that is not present in `ids`. Mutation: delete the
    `np.isin(unmatched, ids).all()` check."""
    ids = np.array([1, 2], dtype=np.int64)
    X = {10: np.array([100.0, 200.0]), 20: np.array([110.0, 210.0])}
    coef = {10: 0.0, 20: 1.0}
    with pytest.raises(ValueError, match="not present"):
        BE.build_values(ids, X, coef, top=20, unmatched=np.array([999], dtype=np.int64))


# ==================================================================================== 12: decomposition on truth

def test_build_values_decomposition_holds_against_truth():
    """Fails when the built file's true MSE against the truth does not decompose exactly as:
        true_mse == exact_full_mse - e_g - 2*X + 2*Y + rounding_energy
    where exact_full_mse = mean((full-y)**2) [the unsplit combination's true MSE], e_g as returned by
    build_values, X = sum((full-top)*(top-y))[unmatched] / n, and Y = sum((full-y)*(values-full))[matched] / n
    (both X and Y use the truth y and so are only ever computed in this test, never by the tool). Derived
    from the unmatched/matched split and verified numerically to machine precision before being written here.
    Checked to 1e-6. Mutation: any sign error in build_values's e_g or rounding_energy formulas would break this
    identity, since X and Y are computed independently here."""
    rng = np.random.default_rng(321)
    n = 3000
    y = rng.gamma(2.0, 200.0, size=n) + 20.0
    f1 = y + rng.normal(0.0, 55.0, size=n)
    f2 = y + rng.normal(0.0, 80.0, size=n)
    top = y + rng.normal(0.0, 65.0, size=n)
    a1, a2, atop = 0.55, 0.5, -0.05
    assert abs(a1 + a2 + atop - 1.0) < 1e-12
    full = a1 * f1 + a2 * f2 + atop * top
    exact_full_mse = float(np.mean((full - y) ** 2))

    is_unmatched = rng.random(n) < 0.08
    ids = np.arange(n, dtype=np.int64)
    X = {1: f1, 2: f2, 99: top}
    coef = {1: a1, 2: a2, 99: atop}
    unmatched_ids = ids[is_unmatched]
    out = BE.build_values(ids, X, coef, top=99, unmatched=unmatched_ids)

    matched = ~is_unmatched
    term_x = float(np.sum(((full - top) * (top - y))[is_unmatched]) / n)
    term_y = float(np.sum(((full - y) * (out["values"] - full))[matched]) / n)
    true_mse = float(np.mean((out["values"] - y) ** 2))
    predicted = exact_full_mse - out["e_g"] - 2 * term_x + 2 * term_y + out["rounding_energy"]
    assert abs(true_mse - predicted) < 1e-6


# ==================================================================================== 13: move_report, plan refusals

def test_move_report_matches_a_hand_made_case():
    """Fails when move_report's n_changed / rms_s / top20_share / max_abs_s / n_over_600s / n_changed_unmatched do
    not match a hand-computable case. Mutation: compute rms_s over only the changed rows instead of all rows."""
    values = np.array([100.0, 200.0, 300.0, 1000.0, 500.0])
    top_values = np.array([100.0, 250.0, 300.0, 100.0, 500.0])   # diffs: 0, -50, 0, 900, 0
    is_unmatched = np.array([False, False, False, False, True])
    out = BE.move_report(values, top_values, is_unmatched)
    assert out["n_changed"] == 2
    diff = values - top_values
    assert abs(out["rms_s"] - float(np.sqrt(np.mean(diff ** 2)))) < 1e-9
    assert out["max_abs_s"] == 900.0
    assert out["n_over_600s"] == 1
    assert out["n_changed_unmatched"] == 0   # the one changed-unmatched candidate (index 4) didn't change


def test_move_report_counts_a_changed_unmatched_row():
    """Fails when move_report does not count a row that both changed and is unmatched. Mutation: compute
    n_changed_unmatched as `is_unmatched.sum()` (ignore whether it changed)."""
    values = np.array([100.0, 999.0])
    top_values = np.array([100.0, 200.0])
    is_unmatched = np.array([False, True])
    out = BE.move_report(values, top_values, is_unmatched)
    assert out["n_changed_unmatched"] == 1


def _tiny_plan_world(tmp_path, n=1500, seed=1):
    """A minimal two-member world (base 10, top 77) with an all-matched lane file, for tests that use pool=[base]
    alone (so no pool member other than base needs to pass eligibility, letting these tests use a small, fast
    n) to exercise plan/CLI mechanics that don't depend on a genuine multi-member solve."""
    sub_dir, y, ids, pred, rmse = _members_world(tmp_path, n, {10: (60.0, 1.0), 77: (70.0, 0.9)}, seed=seed)
    lane_file = tmp_path / "lane.parquet"
    pd.DataFrame({"MVT_ID_mvt": ids.astype(np.float64), "lane": ["matched"] * n}).to_parquet(lane_file, index=False)
    return sub_dir, lane_file, ids


def test_plan_refuses_a_concentrated_move_against_top(tmp_path, monkeypatch):
    """Fails when plan does not refuse a built file whose move against the top file is too concentrated.
    Patches move_report (a module-level dependency of plan, called by its bare name so monkeypatch.setattr on the
    module intercepts it) to return a top20_share above MOVE_MAX_TOP20, isolating plan's own refusal-checking
    logic from the unrelated question of whether real data can produce such a move. Mutation: delete the
    `move["top20_share"] > MOVE_MAX_TOP20` half of plan's move refusal."""
    sub_dir, lane_file, ids = _tiny_plan_world(tmp_path)
    monkeypatch.setattr(BE, "move_report", lambda *a, **k: {
        "n_changed": 5, "rms_s": 1.0, "top20_share": BE.MOVE_MAX_TOP20 + 0.01, "max_abs_s": 10.0,
        "n_over_600s": 0, "n_changed_unmatched": 0})
    with pytest.raises(ValueError, match="plan:"):
        BE.plan(sub_dir, base=10, top=77, pool=[10], lane_file=lane_file)


def test_plan_refuses_a_built_file_that_changes_an_unmatched_row(tmp_path, monkeypatch):
    """Fails when plan does not refuse a built file that changed an unmatched row (unmatched rows must always
    equal the top file exactly). Mutation: delete the `move["n_changed_unmatched"] != 0` half of plan's move
    refusal."""
    sub_dir, lane_file, ids = _tiny_plan_world(tmp_path)
    monkeypatch.setattr(BE, "move_report", lambda *a, **k: {
        "n_changed": 1, "rms_s": 1.0, "top20_share": 0.0, "max_abs_s": 10.0,
        "n_over_600s": 0, "n_changed_unmatched": 1})
    with pytest.raises(ValueError, match="plan:"):
        BE.plan(sub_dir, base=10, top=77, pool=[10], lane_file=lane_file)


# ==================================================================================== 14: write_submission

def _template(tmp_path, ids, pred_i32, name="merry-quicksand_v77.parquet", extra_col=True):
    path = tmp_path / name
    data = {"MVT_ID_mvt": np.asarray(ids, dtype=np.float64), "TAXITIME_SEC_mvt": np.asarray(pred_i32, dtype=np.int32)}
    if extra_col:
        data["other_col"] = np.arange(len(ids), dtype=np.float64)   # exercises "template's columns/order" fully
    pd.DataFrame(data).to_parquet(path, index=False)
    return path


def test_write_submission_matches_template_columns_order_dtypes_and_row_order(tmp_path):
    """Fails when the written file's columns, their order, their dtypes, or the row order do not exactly match
    the template. Mutation: write columns in a different order (e.g. sorted alphabetically) than the template's
    own."""
    ids = np.array([3, 1, 2], dtype=np.int64)        # template's own (on-disk) row order
    pred = np.array([30, 10, 20], dtype=np.int32)
    template_path = _template(tmp_path, ids, pred)
    out_path = tmp_path / "out.parquet"
    values_ids = np.array([1, 2, 3], dtype=np.int64)   # caller's ids in a DIFFERENT order than the template
    values = np.array([111.0, 222.0, 333.0])
    BE.write_submission(out_path, template_path, values_ids, values)

    template = pd.read_parquet(template_path)
    out = pd.read_parquet(out_path)
    assert list(out.columns) == list(template.columns)
    assert [str(out[c].dtype) for c in out.columns] == [str(template[c].dtype) for c in template.columns]
    assert list(out["MVT_ID_mvt"]) == list(template["MVT_ID_mvt"])   # template's own row order, ids [3,1,2]


def test_write_submission_places_values_by_id_regardless_of_input_order(tmp_path):
    """Fails when write_submission does not correctly map each value to ITS id when the caller's id order differs
    from the template's on-disk order (a transposition bug here would put the right values in the wrong rows
    without raising anything). Mutation: assign `values` into the template's row order positionally instead of
    by id (e.g. `out_df[PRED] = values` without reordering)."""
    ids = np.array([3, 1, 2], dtype=np.int64)
    pred = np.array([30, 10, 20], dtype=np.int32)
    template_path = _template(tmp_path, ids, pred, extra_col=False)
    out_path = tmp_path / "out.parquet"
    values_ids = np.array([1, 2, 3], dtype=np.int64)
    values = np.array([111.0, 222.0, 333.0])   # id 1->111, id 2->222, id 3->333
    BE.write_submission(out_path, template_path, values_ids, values)
    out = pd.read_parquet(out_path)
    got = dict(zip(out["MVT_ID_mvt"].astype(np.int64), out["TAXITIME_SEC_mvt"]))
    assert got == {1: 111, 2: 222, 3: 333}


def test_write_submission_refuses_to_overwrite_an_existing_file(tmp_path):
    """Fails when write_submission overwrites an existing out_path instead of refusing. Mutation: delete the
    `if out_path.exists()` check."""
    ids = np.array([1, 2], dtype=np.int64)
    template_path = _template(tmp_path, ids, np.array([10, 20], dtype=np.int32), extra_col=False)
    out_path = tmp_path / "out.parquet"
    out_path.write_bytes(b"pre-existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        BE.write_submission(out_path, template_path, ids, np.array([10.0, 20.0]))


def test_write_submission_refuses_a_template_id_set_mismatch(tmp_path):
    """Fails when write_submission does not check that the template's id set exactly equals `ids`. Mutation:
    delete the template-id-set check."""
    ids = np.array([1, 2], dtype=np.int64)
    template_path = _template(tmp_path, ids, np.array([10, 20], dtype=np.int32), extra_col=False)
    out_path = tmp_path / "out.parquet"
    with pytest.raises(ValueError, match="does not match"):
        BE.write_submission(out_path, template_path, np.array([1, 3], dtype=np.int64), np.array([10.0, 30.0]))


def test_write_submission_refuses_a_null_template_id(tmp_path):
    """Fails when write_submission does not catch a null id in the template before casting it to int64 (ids are
    stored as doubles; a NaN->int64 cast is undefined). Mutation: delete the null-id half of _check_whole_ids as
    called from write_submission."""
    template_path = _write_member(tmp_path, 77, [1.0, np.nan], [10, 20])
    out_path = tmp_path / "out.parquet"
    with pytest.raises(ValueError, match="null id"):
        BE.write_submission(out_path, template_path, np.array([1, 2], dtype=np.int64), np.array([10.0, 20.0]))


def test_write_submission_refuses_a_non_integer_template_id(tmp_path):
    """Fails when write_submission does not catch a fractional id in the template before casting it to int64.
    Mutation: delete the non-integer half of _check_whole_ids as called from write_submission."""
    template_path = _write_member(tmp_path, 77, [1.0, 2.5], [10, 20])
    out_path = tmp_path / "out.parquet"
    with pytest.raises(ValueError, match="non-integer id"):
        BE.write_submission(out_path, template_path, np.array([1, 2], dtype=np.int64), np.array([10.0, 20.0]))


def test_write_submission_refuses_a_lossy_cast(tmp_path):
    """Fails when write_submission silently truncates a fractional value while casting to the template's integer
    dtype instead of refusing. Mutation: delete the post-cast round-trip equality check."""
    ids = np.array([1, 2], dtype=np.int64)
    template_path = _template(tmp_path, ids, np.array([10, 20], dtype=np.int32), extra_col=False)
    out_path = tmp_path / "out.parquet"
    with pytest.raises(ValueError, match="cast"):
        BE.write_submission(out_path, template_path, ids, np.array([10.5, 20.0]))   # 10.5 truncates under int32


def test_write_submission_returned_sha256_equals_the_written_files_hash(tmp_path):
    """Fails when the returned sha256 does not equal the actual written file's hash (e.g. hashing the template
    instead of the output). Mutation: hash `template_path` instead of `out_path`."""
    import hashlib
    ids = np.array([1, 2], dtype=np.int64)
    template_path = _template(tmp_path, ids, np.array([10, 20], dtype=np.int32), extra_col=False)
    out_path = tmp_path / "out.parquet"
    sha = BE.write_submission(out_path, template_path, ids, np.array([10.0, 20.0]))
    assert sha == hashlib.sha256(out_path.read_bytes()).hexdigest()


def test_write_submission_makes_paths_absolute_first(tmp_path, monkeypatch):
    """Fails when write_submission does not resolve a relative out_path/template_path against the current working
    directory before using them. Mutation: delete the `.resolve()` calls on out_path/template_path."""
    ids = np.array([1, 2], dtype=np.int64)
    _template(tmp_path, ids, np.array([10, 20], dtype=np.int32), name="merry-quicksand_v77.parquet",
             extra_col=False)
    monkeypatch.chdir(tmp_path)
    BE.write_submission("out_rel.parquet", "merry-quicksand_v77.parquet", ids, np.array([10.0, 20.0]))
    assert (tmp_path / "out_rel.parquet").exists()


# ==================================================================================== 15: plan output contract

def test_plan_refuses_an_ineligible_pool_member_naming_it(tmp_path):
    """Fails when plan does not refuse (or does not name the member and its failed condition) an ineligible pool
    member. Mutation: delete plan's eligibility loop / refusal entirely."""
    sub_dir, y, ids, pred, rmse = _members_world(tmp_path, 1000, {10: (60.0, 1.0), 77: (70.0, 0.9)}, seed=2)
    # a third member differing on only 3 rows vs base 10: clearly fails n_diff at this small n
    bad = pred[10].copy()
    bad[:3] += 5
    _write_member(sub_dir, 11, np.arange(5_000_000, 5_000_000 + 1000, dtype=np.int64), bad)
    r = float(np.sqrt(np.mean((bad.astype(np.float64) - y) ** 2)))
    _write_result(sub_dir, 11, r, 1000)
    lane_file = tmp_path / "lane.parquet"
    pd.DataFrame({"MVT_ID_mvt": ids.astype(np.float64), "lane": ["matched"] * 1000}).to_parquet(
        lane_file, index=False)
    with pytest.raises(ValueError, match="11"):
        BE.plan(sub_dir, base=10, top=77, pool=[10, 11], lane_file=lane_file)


def test_plan_output_has_the_documented_keys_and_predicted_mse_relationship(tmp_path):
    """Fails when plan's output is missing a documented key, when
    predicted_mse != exact_full_mse - e_g + rounding_energy, when rounding_cross_sd does not match
    2*sqrt(rounding_energy * exact_full_mse / n_rows), or when code_sha256 is not the real sha256 of
    blend_exact.py. Exercises a GENUINE (non-monkeypatched) two-member solve so this is also the suite's one
    non-real-data integration check that solve_convex + build_values + move_report compose correctly inside plan.
    Mutation: compute predicted_mse as `exact_full_mse + e_g` (wrong sign), or drop the `+ rounding_energy` term."""
    n = 105_000   # comfortably >= MIN_ROWS even after the ~0.5% of rows that coincidentally round-tie
    sub_dir, y, ids, pred, rmse = _members_world(
        tmp_path, n, {10: (60.0, 1.0), 11: (90.0, 0.7), 77: (70.0, 0.9)}, seed=21)
    lane_file = tmp_path / "lane.parquet"
    n_unmatched = 50
    # np.array(["matched"] * n) would freeze a <U7 dtype from the initial fill and silently truncate the later
    # "unmatched" (9 chars) assignment to "unmatch" -- np.where sizes the dtype from both choices up front.
    lane = np.where(np.arange(n) < n_unmatched, "unmatched", "matched")
    pd.DataFrame({"MVT_ID_mvt": ids.astype(np.float64), "lane": lane}).to_parquet(lane_file, index=False)

    out = BE.plan(sub_dir, base=10, top=77, pool=[10, 11], lane_file=lane_file)
    expected_keys = {"base", "top", "pool", "weights", "coefficients", "exact_full_mse", "exact_full_rmse",
                     "top_mse", "gain_vs_top", "e_g", "rounding_energy", "n_floored", "not_applied",
                     "predicted_mse", "predicted_rmse", "rounding_cross_sd", "score_rounding_bound", "move",
                     "eligibility", "members", "lane_file_sha256", "n_rows", "n_unmatched", "code_sha256"}
    assert set(out.keys()) == expected_keys   # exact set: catches a dropped OR an undocumented extra key
    assert abs(out["predicted_mse"] - (out["exact_full_mse"] - out["e_g"] + out["rounding_energy"])) < 1e-9
    expect_cross_sd = 2.0 * (out["rounding_energy"] * out["exact_full_mse"] / out["n_rows"]) ** 0.5
    assert out["rounding_cross_sd"] == pytest.approx(expect_cross_sd, rel=1e-9)
    import hashlib
    assert out["code_sha256"] == hashlib.sha256(BE.SELF_PATH.read_bytes()).hexdigest()
    assert out["n_rows"] == n
    assert out["n_unmatched"] == n_unmatched
    assert out["base"] == 10 and out["top"] == 77 and out["pool"] == [10, 11]
    json.dumps(out, sort_keys=True)   # every value must be JSON-serialisable (plain python types, not numpy)


# ==================================================================================== build's drift check (direct)

def test_check_plan_matches_refuses_a_code_change_with_everything_else_unchanged():
    """Fails when _check_plan_matches does not compare code_sha256 at all, so blend_exact.py itself changing
    between plan-time and build-time (with every data file byte-identical) would go undetected. Mutation: delete
    the `stored["code_sha256"] != fresh["code_sha256"]` check."""
    stored = {"code_sha256": "aaa", "members": {"10": {"file_sha256": "a", "result_sha256": "b"}},
             "lane_file_sha256": "c", "weights": {"10": 0.5}}
    fresh = {"code_sha256": "bbb", "members": {10: {"file_sha256": "a", "result_sha256": "b"}},
            "lane_file_sha256": "c", "weights": {10: 0.5}}
    with pytest.raises(ValueError, match="blend_exact.py changed since the plan"):
        BE._check_plan_matches(stored, fresh)


def test_check_plan_matches_checks_code_before_members():
    """Fails when the member-file check runs before (or instead of) the code-hash check: given BOTH a changed
    code hash AND a changed member file at once, a function that checked members first would report the member
    drift and never surface the sharper 'code changed' signal. Mutation: move the code_sha256 check to after the
    members loop."""
    stored = {"code_sha256": "aaa", "members": {"10": {"file_sha256": "a", "result_sha256": "b"}},
             "lane_file_sha256": "c", "weights": {"10": 0.5}}
    fresh = {"code_sha256": "bbb", "members": {10: {"file_sha256": "DIFFERENT", "result_sha256": "b"}},
            "lane_file_sha256": "c", "weights": {10: 0.5}}
    with pytest.raises(ValueError, match="blend_exact.py changed since the plan"):
        BE._check_plan_matches(stored, fresh)


def test_check_plan_matches_refuses_a_weight_drift_with_every_file_unchanged():
    """Fails when _check_plan_matches does not compare weights at all (only files/results/lane), so a weight that
    changed between plan-time and build-time with every underlying file byte-identical (e.g. from a code change
    to the solver) would go undetected. Hits the weight check in isolation, by hand, so no file-hash check fires
    first and masks it. Mutation: delete the weight-comparison loop in _check_plan_matches."""
    stored = {"code_sha256": "same", "members": {"10": {"file_sha256": "a", "result_sha256": "b"}},
             "lane_file_sha256": "c", "weights": {"10": 0.5}}
    fresh = {"code_sha256": "same", "members": {10: {"file_sha256": "a", "result_sha256": "b"}},
            "lane_file_sha256": "c", "weights": {10: 0.5 + 10 * BE.WEIGHT_DRIFT_TOL}}
    with pytest.raises(ValueError, match="weight for 10 changed"):
        BE._check_plan_matches(stored, fresh)


def test_check_plan_matches_accepts_weight_noise_within_tolerance():
    """Fails when the weight-drift check is tighter than WEIGHT_DRIFT_TOL, rejecting benign float noise from
    re-running a deterministic solver rather than only real drift. Mutation: change the comparison from
    `> WEIGHT_DRIFT_TOL` to `> 0` (reject any difference at all)."""
    stored = {"code_sha256": "same", "members": {"10": {"file_sha256": "a", "result_sha256": "b"}},
             "lane_file_sha256": "c", "weights": {"10": 0.5}}
    fresh = {"code_sha256": "same", "members": {10: {"file_sha256": "a", "result_sha256": "b"}},
            "lane_file_sha256": "c", "weights": {10: 0.5 + 0.1 * BE.WEIGHT_DRIFT_TOL}}
    BE._check_plan_matches(stored, fresh)   # must not raise


# ==================================================================================== 16: CLI round trip

def test_cli_eligibility_writes_the_table_and_refuses_to_overwrite(tmp_path):
    """Fails when the eligibility CLI subcommand does not write eligibility()'s table, or overwrites an existing
    --out. Mutation: delete the `if out_path.exists()` check in main()'s eligibility branch."""
    sub_dir, y, ids, pred, rmse = _members_world(tmp_path, 1000, {10: (60.0, 1.0), 11: (90.0, 0.7)}, seed=31)
    out_path = tmp_path / "elig.json"
    rc = BE.main(["eligibility", "--sub-dir", str(sub_dir), "--base", "10", "--versions", "11", "--out",
                 str(out_path)])
    assert rc == 0
    rep = json.loads(out_path.read_text())
    assert "11" in rep
    assert "ok" in rep["11"]
    with pytest.raises(FileExistsError):
        BE.main(["eligibility", "--sub-dir", str(sub_dir), "--base", "10", "--versions", "11", "--out",
                str(out_path)])


def test_cli_round_trip_eligibility_plan_build(tmp_path):
    """Fails when the eligibility -> plan -> build CLI round trip breaks anywhere: plan's --out is not the sorted-
    keys plan dict, build does not reproduce the same file, the meta's sha256 does not equal the written file's
    hash, or any of the three outputs can be overwritten. Mutation: in the build command, write the meta json
    before checking out_path/meta_path both don't already exist (a partial-write-then-fail hazard)."""
    sub_dir, lane_file, ids = _tiny_plan_world(tmp_path, n=1200, seed=41)

    elig_out = tmp_path / "elig.json"
    assert BE.main(["eligibility", "--sub-dir", str(sub_dir), "--base", "10", "--versions", "77", "--out",
                   str(elig_out)]) == 0

    plan_out = tmp_path / "plan.json"
    assert BE.main(["plan", "--sub-dir", str(sub_dir), "--base", "10", "--top", "77", "--pool", "10",
                   "--lane-file", str(lane_file), "--out", str(plan_out)]) == 0
    stored = json.loads(plan_out.read_text())
    assert stored == json.loads(json.dumps(stored, sort_keys=True))   # written with sorted keys (round-trips)
    with pytest.raises(FileExistsError):
        BE.main(["plan", "--sub-dir", str(sub_dir), "--base", "10", "--top", "77", "--pool", "10",
                "--lane-file", str(lane_file), "--out", str(plan_out)])

    build_out = tmp_path / "built.parquet"
    assert BE.main(["build", "--plan", str(plan_out), "--sub-dir", str(sub_dir), "--lane-file", str(lane_file),
                   "--out", str(build_out)]) == 0
    assert build_out.exists()
    meta = json.loads(pathlib.Path(str(build_out) + ".meta.json").read_text())
    assert meta["kind"] == "blend_exact"
    import hashlib
    assert meta["sha256"] == hashlib.sha256(build_out.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        BE.main(["build", "--plan", str(plan_out), "--sub-dir", str(sub_dir), "--lane-file", str(lane_file),
                "--out", str(build_out)])


def test_cli_build_refuses_when_a_member_file_changed_after_the_plan(tmp_path):
    """Fails when build does not detect that a member parquet changed after the plan was written. Mutation:
    delete the member file_sha256 comparison in build's drift check."""
    sub_dir, lane_file, ids = _tiny_plan_world(tmp_path, n=1200, seed=42)
    plan_out = tmp_path / "plan.json"
    assert BE.main(["plan", "--sub-dir", str(sub_dir), "--base", "10", "--top", "77", "--pool", "10",
                   "--lane-file", str(lane_file), "--out", str(plan_out)]) == 0

    member_path = sub_dir / "merry-quicksand_v10.parquet"
    t = pd.read_parquet(member_path)
    t.loc[0, "TAXITIME_SEC_mvt"] = int(t.loc[0, "TAXITIME_SEC_mvt"]) + 1
    member_path.unlink()
    t.to_parquet(member_path, index=False)

    build_out = tmp_path / "built.parquet"
    with pytest.raises(ValueError, match="changed since the plan"):
        BE.main(["build", "--plan", str(plan_out), "--sub-dir", str(sub_dir), "--lane-file", str(lane_file),
                "--out", str(build_out)])


def test_cli_build_refuses_when_a_result_file_changed_after_the_plan(tmp_path):
    """Fails when build does not detect that a result.json changed after the plan was written (the score moved,
    so the member's reported MSE would silently go stale). Mutation: delete the result_sha256 comparison in
    build's drift check."""
    sub_dir, lane_file, ids = _tiny_plan_world(tmp_path, n=1200, seed=43)
    plan_out = tmp_path / "plan.json"
    assert BE.main(["plan", "--sub-dir", str(sub_dir), "--base", "10", "--top", "77", "--pool", "10",
                   "--lane-file", str(lane_file), "--out", str(plan_out)]) == 0

    result_path = sub_dir / "merry-quicksand_v10.parquet_result.json"
    data = json.loads(result_path.read_text())
    data["score"] = data["score"] + 0.0001
    result_path.write_text(json.dumps(data))

    build_out = tmp_path / "built.parquet"
    with pytest.raises(ValueError, match="changed since the plan"):
        BE.main(["build", "--plan", str(plan_out), "--sub-dir", str(sub_dir), "--lane-file", str(lane_file),
                "--out", str(build_out)])


def test_cli_build_refuses_when_the_lane_file_changed_after_the_plan(tmp_path):
    """Fails when build does not detect that the lane (provenance) file changed after the plan was written.
    Mutation: delete the lane_file_sha256 comparison in build's drift check."""
    sub_dir, lane_file, ids = _tiny_plan_world(tmp_path, n=1200, seed=44)
    plan_out = tmp_path / "plan.json"
    assert BE.main(["plan", "--sub-dir", str(sub_dir), "--base", "10", "--top", "77", "--pool", "10",
                   "--lane-file", str(lane_file), "--out", str(plan_out)]) == 0

    lane = pd.read_parquet(lane_file)
    lane.loc[0, "lane"] = "unmatched"
    lane_file.unlink()
    lane.to_parquet(lane_file, index=False)

    build_out = tmp_path / "built.parquet"
    with pytest.raises(ValueError, match="changed since the plan"):
        BE.main(["build", "--plan", str(plan_out), "--sub-dir", str(sub_dir), "--lane-file", str(lane_file),
                "--out", str(build_out)])


# ==================================================================================== unmatched_ids

def test_unmatched_ids_returns_only_the_unmatched_rows(tmp_path):
    """Fails when unmatched_ids returns matched ids, drops unmatched ones, or returns the wrong dtype. Mutation:
    flip the filter from `lane == "unmatched"` to `lane == "matched"`."""
    lane_file = tmp_path / "lane.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0], "lane": ["matched", "unmatched", "matched", "unmatched"]}
                ).to_parquet(lane_file, index=False)
    out = BE.unmatched_ids(lane_file)
    assert out.dtype == np.int64
    assert sorted(out.tolist()) == [2, 4]


def test_unmatched_ids_refuses_a_missing_lane_column(tmp_path):
    """Fails when unmatched_ids does not raise a clear ValueError for a file with no lane column. Mutation:
    delete the lane-column presence check (letting pandas/pyarrow raise its own unrelated error)."""
    lane_file = tmp_path / "lane.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0]}).to_parquet(lane_file, index=False)
    with pytest.raises(ValueError, match="lane column"):
        BE.unmatched_ids(lane_file)


def test_unmatched_ids_refuses_a_null_lane(tmp_path):
    """Fails when unmatched_ids accepts a null lane value. Mutation: delete the null-lane check."""
    lane_file = tmp_path / "lane.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "lane": ["matched", None]}).to_parquet(lane_file, index=False)
    with pytest.raises(ValueError, match="null lane"):
        BE.unmatched_ids(lane_file)


def test_unmatched_ids_refuses_a_lane_value_outside_the_two_allowed(tmp_path):
    """Fails when unmatched_ids accepts a lane value other than 'matched'/'unmatched'. Mutation: delete the
    allowed-value check."""
    lane_file = tmp_path / "lane.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "lane": ["matched", "unknown"]}).to_parquet(lane_file, index=False)
    with pytest.raises(ValueError, match="outside"):
        BE.unmatched_ids(lane_file)


def test_unmatched_ids_refuses_a_null_id(tmp_path):
    """Fails when unmatched_ids does not catch a null id before casting to int64 (unmatched_ids had no id-null
    check at all before this change -- a NaN->int64 cast is undefined). Mutation: delete the null-id half of
    _check_whole_ids as called from unmatched_ids."""
    lane_file = tmp_path / "lane.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, np.nan], "lane": ["matched", "unmatched"]}).to_parquet(lane_file, index=False)
    with pytest.raises(ValueError, match="null id"):
        BE.unmatched_ids(lane_file)


def test_unmatched_ids_refuses_a_non_integer_id(tmp_path):
    """Fails when unmatched_ids does not catch a fractional id before casting to int64. Mutation: delete the
    non-integer half of _check_whole_ids as called from unmatched_ids."""
    lane_file = tmp_path / "lane.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, 2.5], "lane": ["matched", "unmatched"]}).to_parquet(lane_file, index=False)
    with pytest.raises(ValueError, match="non-integer id"):
        BE.unmatched_ids(lane_file)


def test_unmatched_ids_refuses_a_duplicated_id(tmp_path):
    """Fails when unmatched_ids accepts two rows sharing one id. Mutation: delete the duplicate-id check."""
    lane_file = tmp_path / "lane.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, 1.0], "lane": ["matched", "unmatched"]}).to_parquet(lane_file, index=False)
    with pytest.raises(ValueError, match="duplicate id"):
        BE.unmatched_ids(lane_file)


# ==================================================================================== 17: real data (skip if absent)

REAL_SUB_DIR = ROOT / "submissions"
REAL_LANE_FILE = ROOT / "data" / "fm1" / "v57" / "merry-quicksand_v57.provenance.parquet"
REAL_FILES_PRESENT = (REAL_SUB_DIR / "merry-quicksand_v57.parquet").exists() and REAL_LANE_FILE.exists()


@pytest.mark.devtree
def test_real_data_v57_alone_matches_its_own_board_mse():
    """Fails when exact_mse of v57 alone does not reproduce the board's published MSE for v57 (247.3187**2),
    using the real submission file and its real result json. Skips when those files are absent (a fresh clone
    does not carry data/ or submissions/)."""
    if not REAL_FILES_PRESENT:
        pytest.skip("submissions/merry-quicksand_v57.parquet or the v57 provenance file is absent")
    ids, X = BE.load_members(REAL_SUB_DIR, [57])
    order = [57]
    D = BE.pair_d(X, order)
    scores = BE.load_scores(REAL_SUB_DIR, order, len(ids))
    mse = {v: s["mse"] for v, s in scores.items()}
    val = BE.exact_mse({57: 1.0}, mse, D, order)
    assert abs(val - 247.3187 ** 2) < 1e-6


@pytest.mark.devtree
def test_real_data_pool_36_55_matches_the_registered_weights_and_mse():
    """Fails when the registered pool [36, 55], base 55, top 57 does not reproduce the weights (36: 0.387,
    55: 0.613, abs 0.002) and exact_full_mse (60,703.26, abs 0.05) recorded in the worker brief. Skips when the
    real files are absent."""
    if not REAL_FILES_PRESENT:
        pytest.skip("submissions/ or the v57 provenance file is absent")
    out = BE.plan(REAL_SUB_DIR, base=55, top=57, pool=[36, 55], lane_file=REAL_LANE_FILE)
    assert abs(out["weights"][36] - 0.387) < 0.002
    assert abs(out["weights"][55] - 0.613) < 0.002
    assert abs(out["exact_full_mse"] - 60703.26) < 0.05


@pytest.mark.devtree
def test_real_data_pool_six_members_matches_the_registered_weights_and_mse():
    """Fails when the registered 6-member pool [33, 36, 41, 43, 46, 55], base 55, top 57 does not reproduce the
    exact_full_mse (60,624.16, abs 0.05) and the approximate weights recorded in the worker brief (v33 0.100,
    v36 0.147, v41 0.084, v43 0.159, v46 0, v55 0.509, abs 0.003). Skips when the real files are absent."""
    if not REAL_FILES_PRESENT:
        pytest.skip("submissions/ or the v57 provenance file is absent")
    out = BE.plan(REAL_SUB_DIR, base=55, top=57, pool=[33, 36, 41, 43, 46, 55], lane_file=REAL_LANE_FILE)
    assert abs(out["exact_full_mse"] - 60624.16) < 0.05
    expected = {33: 0.100, 36: 0.147, 41: 0.084, 43: 0.159, 46: 0.0, 55: 0.509}
    for v, want in expected.items():
        assert abs(out["weights"][v] - want) < 0.003


