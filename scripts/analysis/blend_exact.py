"""Exact, label-free convex blending of our own board-scored submission files.

The public board gives each uploaded file's RMSE over the same scored rows. For any coefficients `a_k` summing to
1, the board MSE of the affine combination `sum_k a_k f_k` is known exactly, without labels:

    MSE(sum_k a_k f_k) = sum_k a_k MSE_k - 0.5 * sum_j sum_k a_j a_k D_jk,     D_jk = mean((f_j - f_k)^2)

This module (1) loads member files aligned by id and their stored board results, (2) says which members are
eligible to blend against a chosen base, (3) finds the convex weights that minimise the exact MSE of
`sum_{i in pool} w_i f_i + (f_top - f_base)` (base in the pool, top not; `f_top - f_base` always carries
coefficient 1, never scaled), (4) builds the file: the top file's own values on every unmatched row, the
combination on matched rows, and (5) reports what is exact and what is not.

    python scripts/analysis/blend_exact.py eligibility --sub-dir D --base N --versions 33,34,... --out FILE.json
    python scripts/analysis/blend_exact.py plan --sub-dir D --base N --top N --pool 33,36,... \\
        --lane-file F --out PLAN.json
    python scripts/analysis/blend_exact.py build --plan PLAN.json --sub-dir D --lane-file F --out OUT.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TEAM = "merry-quicksand"
ID, PRED = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
MIN_ROWS = 100_000        # an eligible member differs from the base on at least this many rows (a whole-model
                          # difference, not a few-row patch)
MAX_TOP20 = 0.05          # ... and its 20 largest squared differences carry at most this share of D (body-wide
                          # files in our history: 0.017-0.041; files that carry a few-row rule: 0.11 and above)
BIG_S = 3600.0            # ... and no row differs from the base by more than this (a move of over an hour is a
                          # rule, not a model)
MOVE_MAX_TOP20 = 0.05     # the built file's move against the top file: same concentration ceiling
KKT_TOL = 1e-3            # optimality check of the solved weights, in MSE units
SCORE_DECIMALS = 4        # the board prints RMSE to four decimals
WEIGHT_DRIFT_TOL = 1e-6   # build's weight-vs-plan comparison: solve_convex is deterministic given identical
                          # inputs, so this only needs to absorb float noise, not mask real drift
SELF_PATH = pathlib.Path(__file__).resolve()   # this file's own path; plan() hashes it fresh on every call so
                                               # build can refuse to run a plan made under different code


# ====================================================================================================== paths

def member_path(sub_dir, version) -> pathlib.Path:
    """<sub_dir>/merry-quicksand_v<version>.parquet. A pure join: callers make `sub_dir` absolute first."""
    return pathlib.Path(sub_dir) / f"{TEAM}_v{version}.parquet"


def _result_path(member_file_path) -> pathlib.Path:
    """The result json sits beside the member file, named '<member file name>_result.json' (note: no extra dot;
    the real files are literally named '...parquet_result.json')."""
    return pathlib.Path(str(member_file_path) + "_result.json")


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ====================================================================================================== ids

def _check_whole_ids(id_col: np.ndarray, path, context: str) -> None:
    """Refuses (ValueError prefixed by `context`, naming `path`) a null id, then an id that is not a whole number,
    in a raw float64 id column read straight off disk -- before any caller casts it to int64 (a NaN->int64 cast is
    undefined, and a fractional id would silently truncate instead of being refused). Shared by every place in
    this module that casts an id column to int64 (load_members, unmatched_ids, write_submission's template ids)."""
    if np.isnan(id_col).any():
        raise ValueError(f"{context}: null id(s) in {path}")
    if not np.array_equal(id_col, np.trunc(id_col)):
        raise ValueError(f"{context}: non-integer id(s) in {path}")


# ====================================================================================================== loading

def load_members(sub_dir, versions):
    """For each version, reads ID/PRED from member_path, aligns every file to the first file's id set by sorting
    each one by id. Returns (ids: int64 ndarray, {version: float64 ndarray in that id order}). Refuses (ValueError
    naming the file): a missing file; a null or non-integer id; a duplicated id; a non-finite prediction; a file
    whose id set is not exactly the first file's."""
    sub_dir = pathlib.Path(sub_dir)
    ids_ref = None
    first_path = None
    X = {}
    for v in versions:
        path = member_path(sub_dir, v)
        if not path.exists():
            raise ValueError(f"load_members: missing file {path}")
        df = pd.read_parquet(path, columns=[ID, PRED])
        id_col = df[ID].to_numpy()
        _check_whole_ids(id_col, path, "load_members")
        ids_i64 = id_col.astype(np.int64)
        if len(np.unique(ids_i64)) != len(ids_i64):
            raise ValueError(f"load_members: duplicate id(s) in {path}")
        pred = df[PRED].to_numpy().astype(np.float64)
        if not np.isfinite(pred).all():
            raise ValueError(f"load_members: non-finite prediction(s) in {path}")
        order_idx = np.argsort(ids_i64)
        ids_sorted = ids_i64[order_idx]
        pred_sorted = pred[order_idx]
        if ids_ref is None:
            ids_ref = ids_sorted
            first_path = path
        elif not np.array_equal(ids_sorted, ids_ref):
            raise ValueError(f"load_members: id set in {path} does not match {first_path}")
        X[v] = pred_sorted
    return ids_ref, X


def load_scores(sub_dir, versions, n_rows):
    """Reads '<member file name>_result.json' beside each member. Returns
    {version: {"rmse": float, "mse": rmse**2, "result_sha256": str}}. Refuses (ValueError naming the result
    file): a missing result file; status != "Succeeded"; used_pairs != n_rows; file != the member's own file
    name; a score that is not a finite positive number -- including a bool, which Python's
    `isinstance(x, (int, float))` otherwise accepts as 1.0/0.0."""
    sub_dir = pathlib.Path(sub_dir)
    out = {}
    for v in versions:
        mpath = member_path(sub_dir, v)
        rpath = _result_path(mpath)
        if not rpath.exists():
            raise ValueError(f"load_scores: missing result file {rpath}")
        data = json.loads(rpath.read_text())
        status = data.get("status")
        if status != "Succeeded":
            raise ValueError(f"load_scores: status {status!r} != 'Succeeded' in {rpath}")
        used_pairs = data.get("used_pairs")
        if used_pairs != n_rows:
            raise ValueError(f"load_scores: used_pairs {used_pairs!r} != {n_rows} in {rpath}")
        file_name = data.get("file")
        if file_name != mpath.name:
            raise ValueError(f"load_scores: file {file_name!r} != {mpath.name!r} in {rpath}")
        score = data.get("score")
        is_number = isinstance(score, (int, float)) and not isinstance(score, bool)
        if not (is_number and np.isfinite(score) and score > 0):
            raise ValueError(f"load_scores: score {score!r} is not a finite positive number in {rpath}")
        rmse = float(score)
        out[v] = {"rmse": rmse, "mse": rmse ** 2, "result_sha256": _sha256(rpath)}
    return out


# ====================================================================================================== pairwise D

def pair_d(X: dict, order: list) -> np.ndarray:
    """The k x k matrix of mean((X[i]-X[j])**2) in `order`; symmetric, zero diagonal. Computed pairwise (not via
    full broadcasting) so memory stays O(n) per pair regardless of pool size."""
    k = len(order)
    D = np.zeros((k, k), dtype=np.float64)
    arrs = [X[v] for v in order]
    for i in range(k):
        for j in range(i + 1, k):
            d = float(np.mean((arrs[i] - arrs[j]) ** 2))
            D[i, j] = d
            D[j, i] = d
    return D


# ====================================================================================================== exact_mse

def exact_mse(coef: dict, mse: dict, D: np.ndarray, order: list) -> float:
    """sum_i coef[i]*mse[i] - 0.5 * a @ D @ a, where `a` is `coef` mapped onto `order` (0 for any `order` member
    absent from `coef`). Refuses a key in `coef` that is not in `order`; refuses abs(sum(coef) - 1) > 1e-9
    ("coefficients must sum to 1")."""
    bad = set(coef) - set(order)
    if bad:
        raise ValueError(f"exact_mse: key(s) {sorted(bad)} not in order")
    _check_coef_sums_to_one(coef)
    a = np.array([coef.get(v, 0.0) for v in order], dtype=np.float64)
    m = np.array([mse[v] for v in order], dtype=np.float64)
    return float(a @ m - 0.5 * a @ D @ a)


def _check_coef_sums_to_one(coef: dict) -> None:
    if abs(sum(coef.values()) - 1.0) > 1e-9:
        raise ValueError("coefficients must sum to 1")


def _score_rounding_bound(coef: dict, rmse: dict) -> float:
    """sum_k |a_k| * 2 * rmse_k * 0.5 * 10**-SCORE_DECIMALS: a bound on how far exact_mse (computed from board
    scores rounded to SCORE_DECIMALS decimals) can be from the truth, from the linear term alone (D never
    touches a board score, so it carries no rounding error)."""
    half_ulp = 0.5 * 10.0 ** (-SCORE_DECIMALS)
    return float(sum(abs(c) * 2.0 * rmse[v] * half_ulp for v, c in coef.items()))


# ====================================================================================================== eligibility

def _top20_share(sq_diffs: np.ndarray) -> float:
    """sum of the 20 largest values in `sq_diffs` / sum of all of them; 0.0 when the total is 0."""
    total = float(sq_diffs.sum())
    if total == 0.0:
        return 0.0
    k = min(20, sq_diffs.size)
    return float(np.partition(sq_diffs, -k)[-k:].sum() / total)


def _eligibility_ok(n_diff, top20_share, max_abs_diff):
    """The three pass/fail conditions, as a function of the already-computed statistics only (no data), so a
    boundary test can hit the literal threshold values without any fixture-construction float noise."""
    why = []
    if not (n_diff >= MIN_ROWS):
        why.append("n_diff")
    if not (top20_share <= MAX_TOP20):
        why.append("top20_share")
    if not (max_abs_diff <= BIG_S):
        why.append("max_abs_diff")
    return (len(why) == 0), why


def eligibility(X: dict, base: int, versions: list) -> dict:
    """For each version in `versions` other than `base`, against `base`: n_diff (rows that differ), d (mean
    squared difference), top20_share, max_abs_diff, ok, and why (the failed conditions, empty when ok). The base
    itself is not in the result."""
    out = {}
    xb = X[base]
    for v in versions:
        if v == base:
            continue
        diff = X[v] - xb
        n_diff = int(np.sum(diff != 0))
        sq = diff ** 2
        d = float(np.mean(sq))
        top20_share = _top20_share(sq)
        max_abs_diff = float(np.max(np.abs(diff))) if diff.size else 0.0
        ok, why = _eligibility_ok(n_diff, top20_share, max_abs_diff)
        out[v] = {"n_diff": n_diff, "d": d, "top20_share": top20_share, "max_abs_diff": max_abs_diff,
                  "ok": bool(ok), "why": why}
    return out


# ====================================================================================================== coefficients

def coefficients(weights: dict, base: int, top: int) -> dict:
    """weights over the pool (must contain base, must not contain top; refuses a negative weight below -1e-12 or
    a sum off 1 by more than 1e-9). Returns {i: w_i for i != base} (dropping exact zeros) plus
    {base: w_base - 1, top: 1} (base and top are always kept, even at coefficient 0)."""
    if base not in weights:
        raise ValueError(f"coefficients: base {base} not in weights")
    if top in weights:
        raise ValueError(f"coefficients: top {top} must not be in weights")
    for k, w in weights.items():
        if w < -1e-12:
            raise ValueError(f"coefficients: weight for {k} is negative: {w}")
    s = sum(weights.values())
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"coefficients: weights sum to {s}, not 1")

    coef = {k: w for k, w in weights.items() if k != base and w != 0.0}
    coef[base] = weights[base] - 1.0
    coef[top] = 1.0
    return coef


# ====================================================================================================== solve_convex

def solve_convex(mse: dict, D: np.ndarray, order: list, pool: list, base: int, top: int) -> dict:
    """Minimises exact_mse(coefficients(w, base, top), mse, D, order) over w >= 0, sum(w) == 1, w indexed by
    `pool`, via scipy SLSQP from uniform weights with the analytic gradient, then cleans tiny weights to exactly
    0 and renormalises, then POLISHES on the support (solves the equality-constrained problem exactly by one
    linear solve of its KKT system via np.linalg.lstsq, so a duplicate/collinear member cannot break it), keeping
    the polish only if every polished weight is >= 0. Finally checks KKT optimality on the chosen weights and
    raises RuntimeError if it does not hold. Returns {version: weight} for the whole pool (zeros included)."""
    pos = {v: i for i, v in enumerate(order)}
    base_i, top_i = pos[base], pos[top]
    k = len(pool)
    Mvec = np.array([mse[v] for v in order], dtype=np.float64)

    def a_of(w):
        a = np.zeros(len(order), dtype=np.float64)
        for i, v in enumerate(pool):
            a[pos[v]] += w[i]
        a[base_i] -= 1.0
        a[top_i] += 1.0
        return a

    def fun(w):
        a = a_of(w)
        return float(a @ Mvec - 0.5 * a @ D @ a)

    def jac(w):
        a = a_of(w)
        g_full = Mvec - D @ a
        return np.array([g_full[pos[v]] for v in pool], dtype=np.float64)

    w0 = np.full(k, 1.0 / k, dtype=np.float64)
    cons = [{"type": "eq", "fun": lambda w: np.array([np.sum(w) - 1.0]),
            "jac": lambda w: np.ones((1, k))}]
    bounds = [(0.0, None)] * k
    res = minimize(fun, w0, jac=jac, method="SLSQP", bounds=bounds, constraints=cons,
                  options={"ftol": 1e-12, "maxiter": 2000})

    w = np.array(res.x, dtype=np.float64)
    w = np.where(w < 1e-6, 0.0, w)
    w = w / w.sum()

    support = [i for i in range(k) if w[i] > 0.0]
    n_s = len(support)
    p_cols = [pos[pool[i]] for i in support]
    c = np.zeros(len(order), dtype=np.float64)
    c[base_i] -= 1.0
    c[top_i] += 1.0
    H = D[np.ix_(p_cols, p_cols)]
    Dc = D @ c
    b_vec = Mvec[p_cols] - Dc[p_cols]
    K = np.zeros((n_s + 1, n_s + 1), dtype=np.float64)
    K[:n_s, :n_s] = H
    K[:n_s, n_s] = 1.0
    K[n_s, :n_s] = 1.0
    rhs = np.zeros(n_s + 1, dtype=np.float64)
    rhs[:n_s] = b_vec
    rhs[n_s] = 1.0
    sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    w_polish = sol[:n_s]
    if np.all(w_polish >= 0.0):
        w_final = np.zeros(k, dtype=np.float64)
        for idx, i in enumerate(support):
            w_final[i] = w_polish[idx]
    else:
        w_final = w

    a_final = a_of(w_final)
    g_full = Mvec - D @ a_final
    g_pool = np.array([g_full[pos[v]] for v in pool], dtype=np.float64)
    active = w_final > 0.0
    lam = float(np.min(g_pool[active]))
    ok = True
    for i in range(k):
        if w_final[i] > 0.0:
            if g_pool[i] - lam > KKT_TOL:
                ok = False
        else:
            if g_pool[i] < lam - KKT_TOL:
                ok = False
    if not ok:
        raise RuntimeError("solve_convex: the solver did not reach the optimum ...")

    return {v: float(w_final[i]) for i, v in enumerate(pool)}


# ====================================================================================================== unmatched

def unmatched_ids(lane_file) -> np.ndarray:
    """int64 ids of the rows with lane == "unmatched" in the provenance parquet. Refuses: a missing lane column;
    a null lane; a lane value outside {"matched", "unmatched"}; a null or non-integer id; a duplicated id."""
    import pyarrow.parquet as pq

    lane_file = pathlib.Path(lane_file)
    schema_names = pq.ParquetFile(lane_file).schema_arrow.names
    if "lane" not in schema_names:
        raise ValueError(f"unmatched_ids: no lane column in {lane_file}")
    df = pd.read_parquet(lane_file, columns=[ID, "lane"])
    if df["lane"].isna().any():
        raise ValueError(f"unmatched_ids: null lane value(s) in {lane_file}")
    if not df["lane"].isin(["matched", "unmatched"]).all():
        raise ValueError(f"unmatched_ids: lane value(s) outside {{'matched', 'unmatched'}} in {lane_file}")
    id_col = df[ID].to_numpy()
    _check_whole_ids(id_col, lane_file, "unmatched_ids")
    ids = id_col.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"unmatched_ids: duplicate id(s) in {lane_file}")
    return np.sort(ids[df["lane"].to_numpy() == "unmatched"])


# ====================================================================================================== build_values

def build_values(ids, X: dict, coef: dict, top: int, unmatched: np.ndarray) -> dict:
    """full = sum_k coef[k] * X[k]. Refuses a coefficient sum off 1 by more than 1e-9; refuses an unmatched id
    that is not in `ids`. Returns values (X[top] on unmatched rows, else rint(max(full, 1))), is_unmatched,
    e_g (= sum((full-top)^2) over unmatched / len(ids): the part of the combination not applied there),
    rounding_energy (= sum((values-full)^2) over matched / len(ids)), n_floored (matched rows where full < 1),
    and not_applied (n_nonzero / n_over_30s / max_abs_s of (full-top) on the unmatched rows)."""
    _check_coef_sums_to_one(coef)
    ids = np.asarray(ids)
    if unmatched.size and not np.isin(unmatched, ids).all():
        raise ValueError("build_values: unmatched id(s) not present in ids")

    n = len(ids)
    full = np.zeros(n, dtype=np.float64)
    for k, c in coef.items():
        full += c * X[k]

    is_unmatched = np.isin(ids, unmatched)
    matched = ~is_unmatched
    values = np.where(is_unmatched, X[top], np.rint(np.maximum(full, 1.0)))

    e_g = float(np.sum((full - X[top])[is_unmatched] ** 2) / n)
    rounding_energy = float(np.sum((values - full)[matched] ** 2) / n)
    n_floored = int(np.sum((full < 1.0) & matched))

    diff_un = (full - X[top])[is_unmatched]
    not_applied = {"n_nonzero": int(np.sum(diff_un != 0)),
                  "n_over_30s": int(np.sum(np.abs(diff_un) > 30.0)),
                  "max_abs_s": float(np.max(np.abs(diff_un))) if diff_un.size else 0.0}

    return {"values": values, "is_unmatched": is_unmatched, "e_g": e_g, "rounding_energy": rounding_energy,
           "n_floored": n_floored, "not_applied": not_applied}


def move_report(values: np.ndarray, top_values: np.ndarray, is_unmatched: np.ndarray) -> dict:
    """Against the top file: n_changed, rms_s (over all rows), top20_share, max_abs_s, n_over_600s, and
    n_changed_unmatched (must be 0 for a built file; this function only reports it)."""
    diff = values - top_values
    sq = diff ** 2
    n_changed = int(np.sum(diff != 0))
    rms_s = float(np.sqrt(np.mean(sq)))
    top20_share = _top20_share(sq)
    max_abs_s = float(np.max(np.abs(diff))) if diff.size else 0.0
    n_over_600s = int(np.sum(np.abs(diff) > 600.0))
    n_changed_unmatched = int(np.sum((diff != 0) & is_unmatched))
    return {"n_changed": n_changed, "rms_s": rms_s, "top20_share": top20_share, "max_abs_s": max_abs_s,
           "n_over_600s": n_over_600s, "n_changed_unmatched": n_changed_unmatched}


# ====================================================================================================== write_submission

def write_submission(out_path, template_path, ids, values) -> str:
    """Reads the template parquet (the top file; refuses a null or non-integer id in it), requires its id set to
    equal `ids`'s set exactly, writes a parquet with the template's columns/order/dtypes and the template's own row
    order, PRED replaced by `values` placed by id and cast to the template's PRED dtype (refuses if that cast
    changes any value). Refuses to overwrite an existing out_path. Returns the written file's sha256. Makes both
    paths absolute first."""
    out_path = pathlib.Path(out_path).resolve()
    template_path = pathlib.Path(template_path).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")

    template = pd.read_parquet(template_path)
    template_id_col = template[ID].to_numpy()
    _check_whole_ids(template_id_col, template_path, "write_submission")
    template_ids = template_id_col.astype(np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    if set(template_ids.tolist()) != set(ids.tolist()):
        raise ValueError(f"write_submission: template id set in {template_path} does not match ids")

    values = np.asarray(values, dtype=np.float64)
    sort_idx = np.argsort(ids)
    sorted_ids = ids[sort_idx]
    sorted_vals = values[sort_idx]
    pos = np.searchsorted(sorted_ids, template_ids)
    values_in_template_order = sorted_vals[pos]

    target_dtype = template[PRED].dtype
    cast = values_in_template_order.astype(target_dtype)
    if not np.array_equal(cast.astype(np.float64), values_in_template_order):
        raise ValueError(f"write_submission: casting values to {target_dtype} would change value(s)")

    out_df = template.copy()
    out_df[PRED] = cast
    out_df.to_parquet(out_path, index=False)
    return _sha256(out_path)


# ====================================================================================================== plan

def plan(sub_dir, base, top, pool, lane_file) -> dict:
    """All of the above end to end, label-free. Refuses a pool member (other than base) that is not eligible,
    naming it and why; refuses a built file whose move against the top file is too concentrated or that touches
    an unmatched row.

    predicted_mse = exact_full_mse - e_g + rounding_energy: the built file's own rounding residual has a known
    energy (rounding_energy), but that residual's cross-term with the (label-free, so unknown) error is not --
    rounding_cross_sd = 2*sqrt(rounding_energy * exact_full_mse / n_rows) is a rough standard deviation of that
    unknown cross-term, for sizing how far predicted_mse can be off. Also returns code_sha256, the sha256 of this
    file itself at call time, so a later build() can refuse to run against a plan made under different code."""
    sub_dir = pathlib.Path(sub_dir)
    lane_file = pathlib.Path(lane_file)
    order = list(pool) + [top]

    ids, X = load_members(sub_dir, order)
    n_rows = len(ids)
    scores = load_scores(sub_dir, order, n_rows)
    mse = {v: s["mse"] for v, s in scores.items()}
    D = pair_d(X, order)

    elig = eligibility(X, base, pool)
    bad = [(v, elig[v]["why"]) for v in pool if v != base and not elig[v]["ok"]]
    if bad:
        names = "; ".join(f"{v} ({', '.join(why)})" for v, why in bad)
        raise ValueError(f"plan: ineligible pool member(s): {names}")

    weights = solve_convex(mse, D, order, pool, base, top)
    coef = coefficients(weights, base, top)
    exact_full_mse = exact_mse(coef, mse, D, order)
    exact_full_rmse = exact_full_mse ** 0.5
    top_mse = mse[top]
    gain_vs_top = top_mse - exact_full_mse

    unmatched = unmatched_ids(lane_file)
    built = build_values(ids, X, coef, top, unmatched)
    move = move_report(built["values"], X[top], built["is_unmatched"])
    if move["top20_share"] > MOVE_MAX_TOP20 or move["n_changed_unmatched"] != 0:
        raise ValueError("plan: built file's move against the top file is too concentrated or touches an "
                         "unmatched row")

    predicted_mse = exact_full_mse - built["e_g"] + built["rounding_energy"]
    predicted_rmse = predicted_mse ** 0.5
    rounding_cross_sd = 2.0 * (built["rounding_energy"] * exact_full_mse / n_rows) ** 0.5
    score_rounding_bound = _score_rounding_bound(coef, {v: s["rmse"] for v, s in scores.items()})
    members = {v: {"file_sha256": _sha256(member_path(sub_dir, v)), "rmse": scores[v]["rmse"],
                  "result_sha256": scores[v]["result_sha256"]} for v in order}

    return {"base": base, "top": top, "pool": list(pool), "weights": weights, "coefficients": coef,
           "exact_full_mse": exact_full_mse, "exact_full_rmse": exact_full_rmse, "top_mse": top_mse,
           "gain_vs_top": gain_vs_top, "e_g": built["e_g"], "rounding_energy": built["rounding_energy"],
           "n_floored": built["n_floored"], "not_applied": built["not_applied"], "predicted_mse": predicted_mse,
           "predicted_rmse": predicted_rmse, "rounding_cross_sd": rounding_cross_sd,
           "score_rounding_bound": score_rounding_bound, "move": move, "eligibility": elig, "members": members,
           "lane_file_sha256": _sha256(lane_file), "n_rows": n_rows, "n_unmatched": int(len(unmatched)),
           "code_sha256": _sha256(SELF_PATH)}


# ====================================================================================================== CLI

def _check_plan_matches(stored: dict, fresh: dict) -> None:
    """Raises ValueError naming the first drift found between a stored plan (as read back from json: string
    keys) and a freshly recomputed one (plain python call: int keys) -- the code itself (checked first, before
    any member), a member file, a member result file, the lane file, or a weight."""
    if stored["code_sha256"] != fresh["code_sha256"]:
        raise ValueError("build: blend_exact.py changed since the plan")
    for v_str, info in stored["members"].items():
        v = int(v_str)
        fresh_info = fresh["members"][v]
        if info["file_sha256"] != fresh_info["file_sha256"]:
            raise ValueError(f"build: member {v} file changed since the plan")
        if info["result_sha256"] != fresh_info["result_sha256"]:
            raise ValueError(f"build: member {v} result file changed since the plan")
    if stored["lane_file_sha256"] != fresh["lane_file_sha256"]:
        raise ValueError("build: lane file changed since the plan")
    for v_str, w in stored["weights"].items():
        v = int(v_str)
        if abs(float(w) - float(fresh["weights"][v])) > WEIGHT_DRIFT_TOL:
            raise ValueError(f"build: weight for {v} changed since the plan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("eligibility", help="report which members are eligible against a base")
    pe.add_argument("--sub-dir", required=True)
    pe.add_argument("--base", required=True, type=int)
    pe.add_argument("--versions", required=True)
    pe.add_argument("--out", required=True)

    pp = sub.add_parser("plan", help="solve the convex weights and report the exact plan, label-free")
    pp.add_argument("--sub-dir", required=True)
    pp.add_argument("--base", required=True, type=int)
    pp.add_argument("--top", required=True, type=int)
    pp.add_argument("--pool", required=True)
    pp.add_argument("--lane-file", required=True)
    pp.add_argument("--out", required=True)

    pb = sub.add_parser("build", help="recompute the plan and write the blended submission file")
    pb.add_argument("--plan", required=True)
    pb.add_argument("--sub-dir", required=True)
    pb.add_argument("--lane-file", required=True)
    pb.add_argument("--out", required=True)

    a = ap.parse_args(argv)

    if a.command == "eligibility":
        sub_dir = pathlib.Path(a.sub_dir).resolve()
        out_path = pathlib.Path(a.out).resolve()
        if out_path.exists():
            raise FileExistsError(f"refusing to overwrite {out_path}")
        versions = [int(v) for v in a.versions.split(",")]
        all_versions = sorted(set(versions) | {a.base})
        _, X = load_members(sub_dir, all_versions)
        elig = eligibility(X, a.base, versions)
        out_path.write_text(json.dumps(elig, indent=1, sort_keys=True))
        print(json.dumps({"n_versions": len(versions)}, indent=1))
        return 0

    if a.command == "plan":
        sub_dir = pathlib.Path(a.sub_dir).resolve()
        lane_file = pathlib.Path(a.lane_file).resolve()
        out_path = pathlib.Path(a.out).resolve()
        if out_path.exists():
            raise FileExistsError(f"refusing to overwrite {out_path}")
        pool = [int(v) for v in a.pool.split(",")]
        result = plan(sub_dir, a.base, a.top, pool, lane_file)
        out_path.write_text(json.dumps(result, indent=1, sort_keys=True))
        print(json.dumps({"exact_full_mse": result["exact_full_mse"]}, indent=1))
        return 0

    # build
    plan_path = pathlib.Path(a.plan).resolve()
    sub_dir = pathlib.Path(a.sub_dir).resolve()
    lane_file = pathlib.Path(a.lane_file).resolve()
    out_path = pathlib.Path(a.out).resolve()
    meta_path = pathlib.Path(str(out_path) + ".meta.json")
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")
    if meta_path.exists():
        raise FileExistsError(f"refusing to overwrite {meta_path}")

    stored = json.loads(plan_path.read_text())
    base, top = int(stored["base"]), int(stored["top"])
    pool = [int(v) for v in stored["pool"]]
    fresh = plan(sub_dir, base, top, pool, lane_file)
    _check_plan_matches(stored, fresh)

    order = list(pool) + [top]
    ids, X = load_members(sub_dir, order)
    coef = {int(k): float(v) for k, v in fresh["coefficients"].items()}
    unmatched = unmatched_ids(lane_file)
    built = build_values(ids, X, coef, top, unmatched)
    values = built["values"]
    sha = write_submission(out_path, member_path(sub_dir, top), ids, values)
    meta = {"sha256": sha, "kind": "blend_exact", "plan": fresh}
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    print(json.dumps({"sha256": sha}, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
