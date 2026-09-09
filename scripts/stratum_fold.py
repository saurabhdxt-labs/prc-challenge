"""`scripts/stratum_fold.py` -- the Amendment 18.2 harness for the v7 stratum hybrid.

    python scripts/stratum_fold.py                       # the real run: 12 months from data/raw
    python scripts/stratum_fold.py --smoke --synthetic   # code path only, < 60 s, < 0.8 GB

H-18 (plans/PREREG_taxiout_2026_09_08.md, Amendment 18): the mixture with a HYBRID non-fill
term -- Screen B's fitted regressor `nf_fit` where the row's cell estimate is below T_tail
(3,000 s), the cell estimate `nf_cells` otherwise -- beats the incumbent `fit_unmatched` on BOTH
the ex-monster and the pooled stratum RMSE. `p_hat` is the incumbent's, unchanged.

Arms, on the stratum (all ten airports' unmatched rows):
  S0        the incumbent `build_submission.fit_unmatched` (default path, byte-identical to v2-v6)
  S1        the hybrid, `nf_fit` averaged over seeds 0, 1, 2
  S1_seed*  the hybrid with a single seed (the seed sd comes from these)
Folds: 12-fold leave-one-month-out over 2025 (the STRATUM_MONSTERS 3b design) -- every cell
statistic, encoding and classifier fitted inside the eleven training months; fold A (holdout
months 1 and 7) alongside, where S0 must reproduce RESULT 3.2's B0 (1867.90 pooled / 995.60
ex-monster on 5,321 rows, 56 monsters) or the harness is wrong and nothing else counts.
Metrics: pooled and EX-MONSTER (y > 10,800 s, Amendment 9 rule 4), per airport, per month; a
paired MONTH-BLOCK bootstrap (12 blocks resampled with replacement, 2,000 draws, seed 0, the
same draws for every pair) on S1 - S0 pooled and ex-monster; the seed sd from the per-seed S1
LOMO RMSEs; the LIRF-only cut and the rows routed to each branch (reported, not decisional).
The 18.3 clauses are evaluated and printed with their numbers -- (a) ex-monster interval,
>= 20 s, > 2 x seed sd; (b) the pooled loss bound < 20 s; (c) >= 6 of 10 airports -- and NO
verdict word is printed: the owner writes the verdict.

Outputs: reports/stratum_fold_v7.json, reports/stratum_fold_v7.log and the per-row
predictions data/cache_stand/stratum_fold_v7_preds.parquet (both folds, every arm, p_hat,
nf_cells, nf_fit and the routing flag, so any later cut can be made without a refit).
`--smoke` uses 200 draws and a scratch directory and brackets the log with the banner;
`--synthetic` swaps the loader for a planted frame shaped like `derive()`'s output.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import glob
import importlib.util
import json
import os
import pathlib
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
CACHE = ROOT / "data" / "cache_stand"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bs = _load("build_submission")

MONSTER_S = 10_800.0                     # y > 3 h, Amendment 9 rule 4 / STRATUM_MONSTERS section 0
SEEDS = (0, 1, 2)                        # Amendment 18.2: three seeds of nf_fit
FOLD_A = bs.HOLDOUT_MONTHS               # (1, 7): Screen B's fold, RESULT 3.2
N_BOOT, BOOT_SEED = 2_000, 0
ARMS = ("S0", "S1", "S1_seed0", "S1_seed1", "S1_seed2")
ARM_DEFINITIONS = {
    "S0": "incumbent fit_unmatched: p_hat x sp + (1 - p_hat) x nf_cells",
    "S1": "hybrid: p_hat x sp + (1 - p_hat) x nf_hybrid(nf_cells, mean over seeds 0,1,2 of nf_fit)",
    "S1_seed0": "hybrid with nf_fit at seed 0 alone",
    "S1_seed1": "hybrid with nf_fit at seed 1 alone",
    "S1_seed2": "hybrid with nf_fit at seed 2 alone",
}
PAIRS = [("S1", "S0"), ("S1_seed0", "S0"), ("S1_seed1", "S0"), ("S1_seed2", "S0")]
#: the words this harness must never print -- the owner writes the verdict
VERDICT_WORDS = ("ESTABLISHED", "NOT WORKING", "INCONCLUSIVE")
SMOKE_BANNER = "SMOKE — code path only, NOT a result"
SMOKE = dict(n_boot=200)
SYNTHETIC = dict(n_per_month=1_400, seed=0, n_matched_per_month=500)
#: RESULT 3.2's B0 on fold A: the reproduction target for S0 on the real run
R3_2 = {"pooled": 1867.90, "exmonster": 995.60, "n": 5_321, "n_monsters": 56, "tolerance_s": 0.05}
JSON_NAME, LOG_NAME, PREDS_NAME = "stratum_fold_v7.json", "stratum_fold_v7.log", "stratum_fold_v7_preds.parquet"
PRED_COLUMNS = ["MVT_ID_mvt", "fold", "month", "ADEP_mvt", "y", "sp", "monster", "p_hat", "nf_cells", "nf_fit",
                "routed_nf_fit", *ARMS]
COMMAND = ("cd ~/Projects/prc-challenge && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 "
           "/usr/bin/time -l python3.11 -B -u scripts/stratum_fold.py 2>&1 | tee reports/stratum_fold_v7.console.log")
ESTIMATE = (
    "~5 min end to end, peak RSS ~3-4 GB, single thread; run ALONE (the load beside a 5.5 GB fold "
    "is a panic risk on this machine). Load = the shipped recipe derive(admissible(load_movements(12 "
    "months))) ~2-3 min (amend11 measured 2.86 GB peak on the same recipe; the frame then drops to "
    "22,219 unmatched + ~159k LIRF matched rows); 12 LOMO folds x (2 cell paths + 2 LIRF logistics + "
    "3 HGR fits of ~18k rows at ~0.1 s each, early stopping on above 10k rows) ~1-3 s per fold; fold "
    "A the same; the month-block bootstrap is vectorised (2,000 draws x 4 pairs x (pooled, ex-monster, "
    "10 airports, LIRF) well under a minute). Writes reports/stratum_fold_v7.{json,log} and "
    "data/cache_stand/stratum_fold_v7_preds.parquet.")

_T0 = time.time()


class _Log:
    """print + append to the readable log, flushing both."""
    def __init__(self, path: pathlib.Path | None = None):
        self.fh = open(path, "a") if path else None

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.0f}s] {msg}"
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


log = _Log()


# =============================================================================================
# arguments and output paths
# =============================================================================================

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE['n_boot']} bootstrap draws, a scratch directory, the banner")
    ap.add_argument("--synthetic", action="store_true",
                    help="a planted synthetic frame instead of data/raw (the loader is the only difference)")
    ap.add_argument("--out-dir", default=None,
                    help="directory for the json, the log and the parquet; default reports/ + "
                         "data/cache_stand/, a temp dir for --smoke")
    ap.add_argument("--n-boot", type=int, default=None,
                    help=f"bootstrap draws (default {N_BOOT}, {SMOKE['n_boot']} under --smoke)")
    ap.add_argument("--continue-on-repro-fail", action="store_true",
                    help="keep going when fold-A S0 misses RESULT 3.2's B0 (default: write what "
                         "exists and stop with exit code 2)")
    args = ap.parse_args(argv)
    if args.n_boot is None:
        args.n_boot = SMOKE["n_boot"] if args.smoke else N_BOOT
    if args.n_boot < 2:
        ap.error("--n-boot must be >= 2 (a percentile needs a distribution)")
    return args


def output_paths(out_dir) -> tuple:
    """(json, log, parquet). Real runs: reports/ for the json and log, data/cache_stand/ for
    the parquet; with an out dir everything goes there."""
    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        return out_dir / JSON_NAME, out_dir / LOG_NAME, out_dir / PREDS_NAME
    return REPORTS / JSON_NAME, REPORTS / LOG_NAME, CACHE / PREDS_NAME


# =============================================================================================
# fold arithmetic
# =============================================================================================

def monster_mask(y) -> np.ndarray:
    """Amendment 9 rule 4: a monster is y > 10,800 s, strictly."""
    return np.asarray(y, dtype="float64") > MONSTER_S


def lomo_folds(month) -> list:
    """[(held-out month, training mask, test mask)] in month order: one month out per fold."""
    month = np.asarray(month)
    months = sorted(int(m) for m in np.unique(month))
    if len(months) < 2:
        raise ValueError(f"LOMO needs at least two months, got {months}")
    return [(m, month != m, month == m) for m in months]


def fold_a_masks(month, holdout=FOLD_A) -> tuple:
    """(training mask, test mask) for fold A: the holdout months out, everything else in."""
    month = np.asarray(month)
    te = np.isin(month, holdout)
    if not te.any():
        raise ValueError(f"no holdout rows: months {holdout} are absent")
    return ~te, te


def stratum_arms(train_unm: pd.DataFrame, test_unm: pd.DataFrame, train_matched: pd.DataFrame, seeds=SEEDS) -> tuple:
    """Every arm from the code that ships: S0 = the default fit_unmatched, S1 = the hybrid
    over `seeds`, S1_seed* = the mixture rebuilt from that seed's regressor. Checks, on every
    call, that the hybrid left p_hat and nf_cells bit-identical (Amendment 18.1)."""
    base: dict = {}
    s0 = bs.fit_unmatched(train_unm, test_unm, train_matched=train_matched, parts=base)
    parts: dict = {}
    s1 = bs.fit_unmatched(train_unm, test_unm, train_matched=train_matched, hybrid=True, seeds=seeds, parts=parts)
    if not (np.array_equal(base["p"], parts["p"]) and np.array_equal(base["nf"], parts["nf_cells"])):
        raise AssertionError("the hybrid moved p_hat or nf_cells; Amendment 18.1 requires both unchanged")
    arms = {"S0": s0, "S1": s1}
    for s in seeds:
        arms[f"S1_seed{s}"] = bs.mixture(parts["p"], parts["sp"], bs.nf_hybrid(parts["nf_cells"], parts["nf_fit_by_seed"][s]))
    return arms, parts


def _score_split(unm, lirf_matched, tr, te, test_months, seeds, fold_label: str) -> pd.DataFrame:
    tr_unm, te_unm = unm[tr], unm[te]
    tr_mat = lirf_matched[~lirf_matched.month.isin(list(test_months))]
    arms, parts = stratum_arms(tr_unm, te_unm, tr_mat, seeds=seeds)
    out = pd.DataFrame({"MVT_ID_mvt": te_unm.MVT_ID_mvt.to_numpy(), "fold": fold_label,
                        "month": te_unm.month.to_numpy().astype("int64"), "ADEP_mvt": te_unm.ADEP_mvt.to_numpy(),
                        "y": te_unm.y.to_numpy(dtype="float64"), "sp": te_unm.sp.to_numpy(dtype="float64")})
    out["monster"] = monster_mask(out.y.to_numpy())
    out["p_hat"] = np.asarray(parts["p"], dtype="float64")
    out["nf_cells"] = np.asarray(parts["nf_cells"], dtype="float64")
    out["nf_fit"] = np.asarray(parts["nf_fit"], dtype="float64")
    out["routed_nf_fit"] = out.nf_cells.to_numpy() < bs.T_TAIL_S
    for a in ARMS:
        out[a] = np.asarray(arms[a], dtype="float64")
    return out[PRED_COLUMNS]


def score_lomo(unm: pd.DataFrame, lirf_matched: pd.DataFrame, seeds, log) -> pd.DataFrame:
    """Out-of-fold predictions for every stratum row: each month scored once by a model that
    never saw it (unmatched rows AND the LIRF matched rows of that month are both held out)."""
    month = unm.month.to_numpy()
    parts, scored = [], np.zeros(len(unm), dtype="int64")
    for m, tr, te in lomo_folds(month):
        t = time.time()
        part = _score_split(unm, lirf_matched, tr, te, [m], seeds, "lomo")
        scored += te
        parts.append(part)
        n_fit = int(part.routed_nf_fit.sum())
        log(f"  fold {m:2d}: test {int(te.sum()):5,}  train {int(tr.sum()):6,}  routed {n_fit:5,} -> nf_fit, "
            f"{len(part) - n_fit:4,} -> nf_cells  ({time.time() - t:.1f}s)")
    if not (scored == 1).all():
        raise AssertionError("LOMO did not score every row exactly once")
    return pd.concat(parts, ignore_index=True)[PRED_COLUMNS]


def score_fold_a(unm: pd.DataFrame, lirf_matched: pd.DataFrame, seeds, log) -> pd.DataFrame:
    tr, te = fold_a_masks(unm.month.to_numpy())
    t = time.time()
    part = _score_split(unm, lirf_matched, tr, te, FOLD_A, seeds, "A")
    n_fit = int(part.routed_nf_fit.sum())
    log(f"  fold A {FOLD_A}: test {int(te.sum()):5,}  train {int(tr.sum()):6,}  routed {n_fit:5,} -> nf_fit, "
        f"{len(part) - n_fit:4,} -> nf_cells  ({time.time() - t:.1f}s)")
    return part


# =============================================================================================
# scoring
# =============================================================================================

def squared_errors(preds: pd.DataFrame, arms) -> dict:
    """Per-row squared error of each arm against the RAW label, float64. Never winsorised."""
    y = preds.y.to_numpy(dtype="float64")
    return {a: (y - preds[a].to_numpy(dtype="float64")) ** 2 for a in arms}


def rmse(e, mask=None):
    e = np.asarray(e, dtype="float64")
    if mask is not None:
        e = e[np.asarray(mask, dtype=bool)]
    return float(np.sqrt(e.mean())) if len(e) else None


def arm_table(preds: pd.DataFrame, arms) -> dict:
    """{arm: {pooled, exmonster, n, n_monsters, per_airport: {...}, per_month: {...}}}."""
    se = squared_errors(preds, arms)
    mon = monster_mask(preds.y.to_numpy())
    exm = ~mon
    ap, month = preds.ADEP_mvt.to_numpy(), preds.month.to_numpy()
    out = {}
    for a in arms:
        e = se[a]
        rec = {"pooled": rmse(e), "exmonster": rmse(e, exm), "n": int(len(e)), "n_monsters": int(mon.sum()),
               "per_airport": {}, "per_month": {}}
        for code in sorted(str(v) for v in np.unique(ap)):
            m = ap == code
            rec["per_airport"][code] = {"pooled": rmse(e, m), "exmonster": rmse(e, m & exm), "n": int(m.sum()),
                                        "n_monsters": int((m & mon).sum())}
        for mo in sorted(int(v) for v in np.unique(month)):
            m = month == mo
            rec["per_month"][mo] = {"pooled": rmse(e, m), "exmonster": rmse(e, m & exm), "n": int(m.sum()),
                                    "n_monsters": int((m & mon).sum())}
        out[a] = rec
    return out


def block_draws(n_blocks: int, n_draws: int, seed: int) -> np.ndarray:
    """(n_draws x n_blocks) counts: each draw resamples the blocks WITH replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_blocks, size=(n_draws, n_blocks))
    counts = np.zeros((n_draws, n_blocks), dtype="int64")
    np.add.at(counts, (np.repeat(np.arange(n_draws), n_blocks), idx.ravel()), 1)
    return counts


def month_block_bootstrap(se: dict, month, pairs, n_draws=N_BOOT, seed=BOOT_SEED, mask=None) -> dict:
    """Paired MONTH-BLOCK bootstrap of gain = rmse(ref) - rmse(new) for each (new, ref) pair.

    The months are the blocks (STRATUM_MONSTERS section 0): each draw resamples them with
    replacement and the RMSE is taken over the concatenated rows (per-month SSE and n summed).
    One set of block draws serves every arm and every pair -- that is what makes it paired.
    `mask` restricts the rows BEFORE the per-month sums. Percentile interval, 2.5 / 97.5.
    Returns {"new_vs_ref": {gain_s, ci95, excludes_zero, improving, n_draws, seed, n_blocks}}.
    """
    month = np.asarray(month)
    arms = sorted({a for pair in pairs for a in pair})
    se = {a: np.asarray(se[a], dtype="float64") for a in arms}
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        month = month[mask]
        se = {a: v[mask] for a, v in se.items()}
    months = np.unique(month)
    k = len(months)
    if k < 2:
        raise ValueError(f"month-block bootstrap needs at least two populated blocks, got {k}")
    code = np.searchsorted(months, month)
    n_m = np.bincount(code, minlength=k).astype("float64")
    sse = np.stack([np.bincount(code, weights=se[a], minlength=k) for a in arms])       # arms x k
    counts = block_draws(k, n_draws, seed)                                              # draws x k
    rm = np.sqrt((counts @ sse.T) / (counts @ n_m)[:, None])                            # draws x arms
    point = {a: float(np.sqrt(se[a].mean())) for a in arms}
    out = {}
    for new, ref in pairs:
        gains = rm[:, arms.index(ref)] - rm[:, arms.index(new)]
        lo, hi = (float(v) for v in np.percentile(gains, [2.5, 97.5]))
        gain = point[ref] - point[new]
        out[f"{new}_vs_{ref}"] = dict(gain_s=gain, ci95=[lo, hi], excludes_zero=bool(lo > 0.0 or hi < 0.0),
                                      improving=bool(gain > 0.0 and lo > 0.0), n_draws=int(n_draws),
                                      seed=int(seed), n_blocks=int(k))
    return out


def paired_row_bootstrap(se: dict, pairs, n_draws=N_BOOT, seed=BOOT_SEED, mask=None) -> dict:
    """The row-level paired bootstrap of lgbm_fold / RESULT 3.2 (row draws shared by every
    arm), reported on fold A alongside the month-block one for comparability with R3.2's
    [18.6, 169.9] -- with two blocks a month-block interval is not a measurement."""
    arms = sorted({a for pair in pairs for a in pair})
    M = np.stack([np.asarray(se[a], dtype="float64") for a in arms])
    if mask is not None:
        M = M[:, np.asarray(mask, dtype=bool)]
    n = M.shape[1]
    if n < 2:
        raise ValueError("paired row bootstrap needs at least two rows")
    rng = np.random.default_rng(seed)
    means = np.empty((n_draws, len(arms)), dtype="float64")
    for i in range(n_draws):
        means[i] = M[:, rng.integers(0, n, n)].mean(axis=1)
    rm = np.sqrt(means)
    point = {a: float(np.sqrt(M[j].mean())) for j, a in enumerate(arms)}
    out = {}
    for new, ref in pairs:
        gains = rm[:, arms.index(ref)] - rm[:, arms.index(new)]
        lo, hi = (float(v) for v in np.percentile(gains, [2.5, 97.5]))
        gain = point[ref] - point[new]
        out[f"{new}_vs_{ref}"] = dict(gain_s=gain, ci95=[lo, hi], excludes_zero=bool(lo > 0.0 or hi < 0.0),
                                      improving=bool(gain > 0.0 and lo > 0.0), n_draws=int(n_draws),
                                      seed=int(seed), n_rows=int(n))
    return out


def seed_sd(values) -> float:
    """Sample sd (ddof=1) of the three per-seed S1 RMSEs; the rule needs all three seeds."""
    values = np.asarray(list(values), dtype="float64")
    if len(values) != 3:
        raise ValueError(f"seed sd needs the three per-seed RMSEs, got {len(values)}")
    return float(np.std(values, ddof=1))


def exceeds_2x_seed_sd(gain, sd) -> bool:
    """The repo's rule (reports/MSE_LEDGER.md): gain > 2 x seed sd, strictly. A zero sd
    (seeds identical below sklearn's 10,000-row early-stop switch) is cleared by any positive gain."""
    return bool(float(gain) > 2.0 * float(sd))


def airports_improving(rmse_new: dict, rmse_ref: dict) -> int:
    """How many airports the new arm strictly improves; a tie is not an improvement."""
    return int(sum(1 for a in rmse_ref if a in rmse_new and rmse_new[a] is not None and rmse_ref[a] is not None
                   and rmse_new[a] < rmse_ref[a]))


def clauses_18_3(exm_pair: dict, pooled_pair: dict, per_airport_exm_s0: dict, per_airport_exm_s1: dict,
                 seed_sd_exmonster: float) -> dict:
    """Amendment 18.3, evaluated to numbers and booleans. No verdict word here or anywhere."""
    gain, (lo, hi) = float(exm_pair["gain_s"]), exm_pair["ci95"]
    sd = float(seed_sd_exmonster)
    a = {"gain_s": gain, "ci95": [float(lo), float(hi)], "excludes_zero_improving": bool(exm_pair["improving"]),
         "gain_ge_20_s": bool(gain >= 20.0), "seed_sd_s": sd, "gain_gt_2x_seed_sd": exceeds_2x_seed_sd(gain, sd)}
    a["holds"] = bool(a["excludes_zero_improving"] and a["gain_ge_20_s"] and a["gain_gt_2x_seed_sd"])
    pg, (plo, phi) = float(pooled_pair["gain_s"]), pooled_pair["ci95"]
    b = {"gain_s": pg, "ci95": [float(plo), float(phi)], "loss_upper_bound_s": float(-plo),
         "loss_upper_bound_lt_20_s": bool(-plo < 20.0)}
    b["holds"] = b["loss_upper_bound_lt_20_s"]
    k = airports_improving(per_airport_exm_s1, per_airport_exm_s0)
    c = {"airports_improving_exmonster": k, "n_airports": len(per_airport_exm_s0), "at_least_6": bool(k >= 6)}
    nw = {"exmonster_interval_includes_zero": not bool(exm_pair["excludes_zero"]),
          "pooled_point_worse_by_more_than_20_s": bool(pg < -20.0),
          "pooled_interval_worse_by_more_than_20_s": bool(-plo > 20.0)}
    nw["any"] = bool(nw["exmonster_interval_includes_zero"] or nw["pooled_point_worse_by_more_than_20_s"]
                     or nw["pooled_interval_worse_by_more_than_20_s"])
    return {"a_exmonster": a, "b_pooled_tail_preserved": b, "c_airports": c,
            "all_three_hold": bool(a["holds"] and b["holds"] and c["at_least_6"]), "not_working_triggers": nw}


def render_clauses(c: dict) -> list:
    a, b, cc, nw = c["a_exmonster"], c["b_pooled_tail_preserved"], c["c_airports"], c["not_working_triggers"]
    return [
        f"(a) ex-monster S1 - S0: gain {a['gain_s']:+.2f} s, month-block 95% [{a['ci95'][0]:+.2f}, {a['ci95'][1]:+.2f}]; "
        f"excludes zero improving: {a['excludes_zero_improving']}; gain >= 20 s: {a['gain_ge_20_s']}; "
        f"seed sd {a['seed_sd_s']:.3f} s, gain > 2 x seed sd: {a['gain_gt_2x_seed_sd']}; clause holds: {a['holds']}",
        f"(b) pooled S1 - S0: gain {b['gain_s']:+.2f} s, month-block 95% [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}]; "
        f"loss upper bound {b['loss_upper_bound_s']:.2f} s (< 20 s: {b['loss_upper_bound_lt_20_s']}); clause holds: {b['holds']}",
        f"(c) airports improving ex-monster: {cc['airports_improving_exmonster']} of {cc['n_airports']} (>= 6: {cc['at_least_6']})",
        f"all three 18.3 clauses hold: {c['all_three_hold']}",
        f"18.3 triggers -- ex-monster interval includes zero: {nw['exmonster_interval_includes_zero']}; pooled point worse "
        f"by > 20 s: {nw['pooled_point_worse_by_more_than_20_s']}; pooled interval worse by > 20 s: "
        f"{nw['pooled_interval_worse_by_more_than_20_s']}; any: {nw['any']}",
    ]


# =============================================================================================
# the synthetic frame (--synthetic): shaped like derive(), with the registered structure planted
# =============================================================================================

AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
_BASE_AP = {"EDDF": 900.0, "EDDM": 700.0, "EGLL": 1200.0, "EHAM": 800.0, "LEBL": 600.0, "LEMD": 650.0,
            "LFPG": 1000.0, "LIRF": 1100.0, "LSZH": 500.0, "LTFM": 1300.0}
_AIRLINES = ("RYR", "ITY", "AEZ", "BAW", "DLH", "AFR")
_AIRLINE_EFF = {"RYR": -250.0, "ITY": 300.0, "AEZ": 0.0, "BAW": 150.0, "DLH": -100.0, "AFR": 200.0}
_P_FILL_LIRF = {"ITY": 0.7, "RYR": 0.2}          # other airlines 0.45; every other airport 0.03
_TAIL_SHARE = 0.30                                # LIRF non-fill rows that are genuine long waits
_OTHER_MONSTER_RATE = 0.002                       # rare unexplained monsters elsewhere (LFPG 2, LSZH 2 in 2025)


def _month_starts() -> list:
    """Epoch seconds of the first instant of 2025-01 .. 2026-01 (thirteen edges, twelve months)."""
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return [int((pd.Timestamp(2025 + (m - 1) // 12, (m - 1) % 12 + 1, 1, tz="UTC") - epoch).total_seconds())
            for m in range(1, 14)]


def _body_taxi(rng, ap, airline, stand, sched_hr, late) -> np.ndarray:
    """The non-fill body: airport base + airline + stand prefix + hour effects, a slope on the
    lateness, noise. The cells see only the airport and the sp band; the regressor sees the rest."""
    base = np.array([_BASE_AP[a] for a in ap])
    air = np.array([_AIRLINE_EFF[a] for a in airline])
    letter = np.array(["ABCDEF".index(s[0]) for s in stand], dtype="float64")
    digit = np.array([int(s[1]) for s in stand], dtype="float64")
    st = 120.0 * (letter - 2.5) + 80.0 * (digit - 1.5)
    hr = np.where((sched_hr >= 6) & (sched_hr <= 9) | (sched_hr >= 16) & (sched_hr <= 19), 350.0,
                  np.where(sched_hr <= 4, -150.0, 0.0))
    y = base + air + st + hr + 0.08 * late + rng.normal(0.0, 150.0, len(ap))
    return np.maximum(np.rint(y), 60.0)


def _synthetic_block(rng, n_per_month: int, matched: bool, id_offset: float) -> pd.DataFrame:
    starts = _month_starts()
    n = 12 * n_per_month
    sched_s = np.concatenate([rng.integers(starts[m - 1], starts[m], n_per_month) for m in range(1, 13)]).astype("float64")
    ap = np.full(n, "LIRF") if matched else rng.choice(AIRPORTS, n)
    airline = rng.choice(_AIRLINES, n)
    stand = rng.choice([f"{c}{i:02d}" for c in "ABCDEF" for i in range(1, 31)], n)
    sched_hr = ((sched_s % 86400) // 3600).astype("int64")
    p_fill = np.where(ap == "LIRF", np.array([_P_FILL_LIRF.get(a, 0.45) for a in airline]), 0.03)
    is_fill = rng.random(n) < p_fill
    late = np.rint(rng.gamma(1.5, 1800.0, n))
    y = _body_taxi(rng, ap, airline, stand, sched_hr, late)
    if not matched:
        tail = (ap == "LIRF") & ~is_fill & (rng.random(n) < _TAIL_SHARE)
        late = np.where(tail, np.rint(rng.uniform(10_800.0, 70_000.0, n)), late)
        y = np.where(tail, np.rint(0.5 * late + rng.normal(0.0, 0.05 * late)), y)
        other = (ap != "LIRF") & ~is_fill & (rng.random(n) < _OTHER_MONSTER_RATE)
        y = np.where(other, np.rint(rng.uniform(12_000.0, 40_000.0, n)), y)
    block_s = np.where(is_fill, sched_s, sched_s + late)
    mvt_s = np.where(is_fill, sched_s + late + y, block_s + y)
    df = pd.DataFrame({
        "PHASE_mvt": "DEP",
        "MVT_ID_mvt": id_offset + np.arange(n, dtype="float64"),
        "ADEP_mvt": ap,
        "ADES_mvt": rng.choice(["EGLL", "LFPG", "LEMD", "KJFK", "OMDB", "EDDF", "LIRF", "EHAM"], n),
        "STAND_mvt": stand,
        "RUNWAY_mvt": rng.choice(["09L", "27R", "16", "34", "07L", "25R"], n),
        "AIRCRAFT_TYPE_mvt": rng.choice(["A320", "B738", "A21N", "B77W"], n),
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
        aobt_s = block_s + np.where(is_fill, 600.0, np.rint(rng.normal(0.0, 60.0, n)))
        df["AOBT_3_flt"] = pd.to_datetime(aobt_s, unit="s", utc=True)
        df["AIRCRAFT_OPERATOR_flt"] = pd.Series(airline, index=df.index, dtype="str")
        df["MARKET_SEGMENT_flt"] = pd.Series(["S"] * n, index=df.index, dtype="str")
        df["WK_TBL_CAT_flt"] = pd.Series(["M"] * n, index=df.index, dtype="str")
        df["FLIGHT_TYPE_flt"] = pd.Series(["S"] * n, index=df.index, dtype="str")
    return df


def synthetic_frame(n_per_month: int = SYNTHETIC["n_per_month"], seed: int = SYNTHETIC["seed"],
                    n_matched_per_month: int = SYNTHETIC["n_matched_per_month"]) -> tuple:
    """(unmatched stratum, LIRF matched) frames through the real `derive()`, so every dtype and
    derived column is the shipped one. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    unm = bs.derive(_synthetic_block(rng, n_per_month, matched=False, id_offset=0.0))
    mat = bs.derive(_synthetic_block(rng, n_matched_per_month, matched=True, id_offset=1e7))
    for d in (unm, mat):
        d["y"] = (d.MVT_TIME_UTC_mvt - d.BLOCK_TIME_UTC_mvt).dt.total_seconds()
        d["delta"] = (d.BLOCK_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
    return unm.reset_index(drop=True), mat.reset_index(drop=True)


# =============================================================================================
# the real frame
# =============================================================================================

def load_real(files, log) -> tuple:
    """The shipped recipe on the twelve months, reduced to what the stratum needs: the
    unmatched rows and LIRF's matched rows (the only matched rows fit_unmatched reads -- checked
    bit-exact on the first fold before the full frame is released, as amend11 did)."""
    t = time.time()
    f = bs.derive(bs.admissible(bs.load_movements(files)))
    f["y"] = f.TAXITIME_SEC_mvt.astype("float64")
    f["delta"] = (f.BLOCK_TIME_UTC_mvt - f.AOBT_3_flt).dt.total_seconds()
    log(f"frame: {len(f):,} admissible DEP rows from {len(files)} files in {time.time() - t:.0f}s; peak RSS {peak_rss_gb():.2f} GB")
    unm = f[f.unmatched].reset_index(drop=True)
    lirf = f[(~f.unmatched) & (f.ADEP_mvt == "LIRF")].reset_index(drop=True)
    m0 = int(unm.month.min())
    a = bs.fit_unmatched(unm[unm.month != m0], unm[unm.month == m0], train_matched=f[(~f.unmatched) & (f.month != m0)])
    b = bs.fit_unmatched(unm[unm.month != m0], unm[unm.month == m0], train_matched=lirf[lirf.month != m0])
    if not np.array_equal(a, b):
        raise AssertionError("fit_unmatched(all matched) != fit_unmatched(LIRF matched) on the first fold")
    log(f"fold {m0}: fit_unmatched(all matched) == fit_unmatched(LIRF matched) bit-exact; releasing the full frame")
    del f, a, b
    gc.collect()
    return unm, lirf


# =============================================================================================
# records
# =============================================================================================

def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def _git() -> dict:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"git_sha": run("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _interval_or_none(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except ValueError as e:
        return {"unavailable": str(e)}


def summarise(preds: pd.DataFrame, n_boot: int, log, title: str, row_bootstrap: bool = False) -> dict:
    """Every number Amendment 18.2 asks for, from one predictions frame."""
    table = arm_table(preds, ARMS)
    se = squared_errors(preds, ARMS)
    y, month, ap = preds.y.to_numpy(), preds.month.to_numpy(), preds.ADEP_mvt.to_numpy()
    exm = ~monster_mask(y)
    pooled = month_block_bootstrap(se, month, PAIRS, n_boot, BOOT_SEED)
    exmb = month_block_bootstrap(se, month, PAIRS, n_boot, BOOT_SEED, mask=exm)
    airports = list(table["S0"]["per_airport"])
    pairs = {}
    for new, ref in PAIRS:
        key = f"{new}_vs_{ref}"
        s_new = {a: table[new]["per_airport"][a]["exmonster"] for a in airports}
        s_ref = {a: table[ref]["per_airport"][a]["exmonster"] for a in airports}
        rec = {"pooled": pooled[key], "exmonster": exmb[key],
               "airports_improving_exmonster": airports_improving(s_new, s_ref), "n_airports": len(airports),
               "per_airport_exmonster": {}}
        for a in airports:
            m = ap == a
            got = _interval_or_none(month_block_bootstrap, se, month, [(new, ref)], n_boot, BOOT_SEED, mask=m & exm)
            rec["per_airport_exmonster"][a] = {"rmse_ref": s_ref[a], "rmse_new": s_new[a],
                                               "gain_s": (s_ref[a] - s_new[a]) if s_ref[a] is not None and s_new[a] is not None else None,
                                               "month_block": got.get(key, got)}
        if row_bootstrap:
            rec["pooled_row"] = paired_row_bootstrap(se, [(new, ref)], n_boot, BOOT_SEED)[key]
            rec["exmonster_row"] = paired_row_bootstrap(se, [(new, ref)], n_boot, BOOT_SEED, mask=exm)[key]
        pairs[key] = rec
    sd_pooled = seed_sd([table[f"S1_seed{s}"]["pooled"] for s in SEEDS])
    sd_exm = seed_sd([table[f"S1_seed{s}"]["exmonster"] for s in SEEDS])
    routed = preds.routed_nf_fit.to_numpy()
    routing = {"n_nf_fit": int(routed.sum()), "n_nf_cells": int((~routed).sum()), "T_tail": bs.T_TAIL_S,
               "share_nf_fit": float(routed.mean()), "n_nf_fit_exmonster": int((routed & exm).sum()),
               "n_nf_cells_monsters": int((~routed & ~exm).sum())}
    lirf = ap == "LIRF"
    lirf_cut = {"n": int(lirf.sum()), "n_monsters": int((lirf & ~exm).sum()),
                "pooled": {a: rmse(se[a], lirf) for a in ARMS}, "exmonster": {a: rmse(se[a], lirf & exm) for a in ARMS},
                "S1_vs_S0": {"pooled": _interval_or_none(month_block_bootstrap, se, month, [("S1", "S0")], n_boot, BOOT_SEED, mask=lirf),
                             "exmonster": _interval_or_none(month_block_bootstrap, se, month, [("S1", "S0")], n_boot, BOOT_SEED, mask=lirf & exm)}}
    per_month = {}
    for mo in table["S0"]["per_month"]:
        per_month[mo] = {"n": table["S0"]["per_month"][mo]["n"], "n_monsters": table["S0"]["per_month"][mo]["n_monsters"],
                         **{a: {"pooled": table[a]["per_month"][mo]["pooled"], "exmonster": table[a]["per_month"][mo]["exmonster"]}
                            for a in ARMS}}
    s0_exm = {a: table["S0"]["per_airport"][a]["exmonster"] for a in airports}
    s1_exm = {a: table["S1"]["per_airport"][a]["exmonster"] for a in airports}
    clauses = clauses_18_3(pairs["S1_vs_S0"]["exmonster"], pairs["S1_vs_S0"]["pooled"], s0_exm, s1_exm, sd_exm)

    # ---- readable tables ----
    log(f"=== {title}: stratum RMSE, n={len(preds):,}, monsters {int((~exm).sum())} ===")
    log(f"  {'arm':10s} {'pooled':>10s} {'ex-monster':>11s}")
    for a in ARMS:
        log(f"  {a:10s} {table[a]['pooled']:10.2f} {table[a]['exmonster']:11.2f}")
    log(f"  seed sd over S1_seed0..2: pooled {sd_pooled:.3f} s, ex-monster {sd_exm:.3f} s")
    for key, rec in pairs.items():
        for cut in ("pooled", "exmonster"):
            r = rec[cut]
            log(f"  {key:16s} {cut:10s} gain {r['gain_s']:+8.2f} s  month-block 95% [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  "
                f"excludes zero {r['excludes_zero']}  improving {r['improving']}")
        if row_bootstrap:
            r = rec["exmonster_row"]
            log(f"  {key:16s} {'exm (row)':10s} gain {r['gain_s']:+8.2f} s  row 95% [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  (R3.2-comparable)")
        log(f"  {key:16s} airports improving ex-monster: {rec['airports_improving_exmonster']} of {rec['n_airports']}")
    log(f"  routed {routing['n_nf_fit']:,} rows -> nf_fit ({100 * routing['share_nf_fit']:.1f}%), {routing['n_nf_cells']:,} -> nf_cells "
        f"(of which {routing['n_nf_cells_monsters']} monsters); T_tail {bs.T_TAIL_S:.0f} s")
    log(f"  --- per airport: ex-monster RMSE [pooled], S0 vs S1 ---")
    for a in airports:
        r0, r1 = table["S0"]["per_airport"][a], table["S1"]["per_airport"][a]
        log(f"  {a:6s} n {r0['n']:5,} mon {r0['n_monsters']:3d}  S0 {r0['exmonster']:8.1f} [{r0['pooled']:8.1f}]  "
            f"S1 {r1['exmonster']:8.1f} [{r1['pooled']:8.1f}]  gain {r0['exmonster'] - r1['exmonster']:+7.1f}")
    log(f"  --- per month: ex-monster RMSE [pooled], S0 vs S1 ---")
    for mo, r in per_month.items():
        log(f"  {mo:2d}     n {r['n']:5,} mon {r['n_monsters']:3d}  S0 {r['S0']['exmonster']:8.1f} [{r['S0']['pooled']:8.1f}]  "
            f"S1 {r['S1']['exmonster']:8.1f} [{r['S1']['pooled']:8.1f}]")
    log(f"  --- LIRF only: n {lirf_cut['n']:,} (monsters {lirf_cut['n_monsters']}): S0 {lirf_cut['exmonster']['S0']} ex-monster "
        f"[{lirf_cut['pooled']['S0']} pooled]  S1 {lirf_cut['exmonster']['S1']} [{lirf_cut['pooled']['S1']}] ---")
    log(f"  --- Amendment 18.3 clauses ({title}) ---")
    for line in render_clauses(clauses):
        log("  " + line)
    return {"arms": {a: {"definition": ARM_DEFINITIONS[a], "pooled": table[a]["pooled"], "exmonster": table[a]["exmonster"],
                         "n": table[a]["n"], "n_monsters": table[a]["n_monsters"], "per_airport": table[a]["per_airport"]}
                     for a in ARMS},
            "pairs": pairs,
            "seed_sd": {"pooled": sd_pooled, "exmonster": sd_exm,
                        "values_pooled": {str(s): table[f"S1_seed{s}"]["pooled"] for s in SEEDS},
                        "values_exmonster": {str(s): table[f"S1_seed{s}"]["exmonster"] for s in SEEDS}},
            "routing": routing, "lirf_cut": lirf_cut, "per_month": per_month, "clauses_18_3": clauses}


def check_reproduction(fold_a: dict, synthetic: bool) -> dict:
    """Fold-A S0 against RESULT 3.2's B0. The default path is byte-identical to what Screen B
    called, so a miss means the frame, not the model, differs."""
    if synthetic:
        return {"status": "skipped (synthetic)", "reference": R3_2}
    s0 = fold_a["arms"]["S0"]
    got = {"pooled": s0["pooled"], "exmonster": s0["exmonster"], "n": s0["n"], "n_monsters": s0["n_monsters"]}
    ok = (abs(got["pooled"] - R3_2["pooled"]) <= R3_2["tolerance_s"] and abs(got["exmonster"] - R3_2["exmonster"]) <= R3_2["tolerance_s"]
          and got["n"] == R3_2["n"] and got["n_monsters"] == R3_2["n_monsters"])
    return {"status": "reproduced" if ok else "FAILED", "reference": R3_2, "measured": got}


def run(args, paths, started) -> int:
    json_path, log_path, preds_path = paths
    n_boot = args.n_boot
    record = {"mode": "stratum_v7", "smoke": bool(args.smoke), "synthetic": bool(args.synthetic), **_git(),
              "started_utc": started.isoformat(), "command": COMMAND, "estimate": ESTIMATE,
              "config": {"seeds": list(SEEDS), "T_tail_s": bs.T_TAIL_S, "winsor_s": bs.WINSOR_S, "monster_s": MONSTER_S,
                         "n_boot": int(n_boot), "boot_seed": BOOT_SEED, "n_blocks": "the months present",
                         "regressor": dict(bs.NF_PARAMS), "numeric": list(bs.NF_NUMERIC), "encoded": list(bs.NF_ENCODED),
                         "smooth": bs.SMOOTH, "k_shrink": bs.K_SHRINK, "fold_a_holdout": list(FOLD_A),
                         "sklearn_early_stopping": "'auto' -- on above 10,000 rows on a seed-dependent 10% split (Screen B's setting)",
                         "synthetic": dict(SYNTHETIC) if args.synthetic else None},
              "judgement_calls": [
                  "stand prefix = derive's `stand_pref` (two characters), as Screen B's B2 code used; the task text said "
                  "`stand_c1` -- Amendment 18.1 says 'exactly Screen B', so the two-character prefix is what is measured",
                  "the '2 x seed sd' of clause (a) uses the sd of the three per-seed S1 EX-MONSTER LOMO RMSEs (the gain it "
                  "qualifies is ex-monster); the pooled sd is reported beside it",
                  "clause (b) / the pooled trigger: 'the pooled interval shows S1 worse by more than 20 s' is reported under "
                  "both readings, the point loss and the interval's loss bound (-lower bound of the gain)",
                  "month-block bootstrap = the months resampled with replacement, RMSE over the concatenated rows (per-month "
                  "SSE and n summed), 2,000 draws, default_rng(0), one set of block draws for every pair and cut",
                  "fold A additionally carries a paired ROW bootstrap (2,000 draws, seed 0) because RESULT 3.2's [18.6, 169.9] "
                  "was a row bootstrap and a two-block interval is not a measurement",
                  "the LIRF model is fitted from LIRF's matched rows only; fit_unmatched(all matched) == fit_unmatched(LIRF "
                  "matched) is asserted bit-exact on the first fold before the full frame is released",
                  "the held-out month's LIRF MATCHED rows are held out too (the L-e propensity table cannot see them)",
                  "nf_fit's HistGradientBoostingRegressor keeps sklearn's early_stopping='auto': ON above 10,000 training rows "
                  "(the real stratum), OFF below -- identical to Screen B; not a tuning choice"]}
    if args.synthetic:
        unm, lirf = synthetic_frame()
        log(f"synthetic frame: {len(unm):,} unmatched rows, {len(lirf):,} LIRF matched rows ({SYNTHETIC})")
    else:
        files = sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
        if len(files) != 12:
            raise SystemExit(f"expected the 12 training months under {RAW}, found {len(files)}")
        unm, lirf = load_real(files, log)
    fill = bs.schedule_fill(unm)
    mon = monster_mask(unm.y.to_numpy())
    record["data"] = {"n_stratum": int(len(unm)), "n_lirf_matched": int(len(lirf)),
                      "months": sorted(int(m) for m in unm.month.unique()),
                      "airports": sorted(str(a) for a in unm.ADEP_mvt.unique()),
                      "fill_share": float(fill.mean()), "n_monsters": int(mon.sum()),
                      "monsters_per_airport": {str(k): int(v) for k, v in unm[mon].ADEP_mvt.value_counts().items()},
                      "n_nonfill": int((~fill).sum())}
    log(f"stratum: {len(unm):,} rows, fill share {100 * fill.mean():.1f}%, non-fill {int((~fill).sum()):,}, monsters "
        f"{int(mon.sum())} {record['data']['monsters_per_airport']}; LIRF matched {len(lirf):,}; bootstrap {n_boot:,} draws seed {BOOT_SEED}")

    log("--- 12-fold LOMO ---")
    t = time.time()
    lomo = score_lomo(unm, lirf, SEEDS, log)
    log(f"LOMO scored {len(lomo):,} rows in {time.time() - t:.0f}s; peak RSS {peak_rss_gb():.2f} GB")
    log("--- fold A ---")
    fa = score_fold_a(unm, lirf, SEEDS, log)
    preds = pd.concat([lomo, fa], ignore_index=True)[PRED_COLUMNS]
    preds.to_parquet(preds_path, index=False)
    log(f"predictions -> {preds_path} ({len(preds):,} rows: {len(lomo):,} lomo + {len(fa):,} fold A)")

    record["fold_a"] = summarise(fa, n_boot, log, "fold A (holdout 1, 7)", row_bootstrap=True)
    record["fold_a"]["r3_2_reference"] = R3_2
    record["reproduction"] = check_reproduction(record["fold_a"], args.synthetic)
    log(f"reproduction of RESULT 3.2's B0 on fold A: {record['reproduction']}")
    rc = 0
    if record["reproduction"]["status"] == "FAILED":
        log("fold-A S0 does not reproduce RESULT 3.2's B0 -- the frame differs from Screen B's; nothing below counts")
        rc = 2
        if not args.continue_on_repro_fail:
            _finish(record, json_path, started)
            return rc
    record["lomo"] = summarise(lomo, n_boot, log, "12-fold LOMO")
    _finish(record, json_path, started)
    log(f"json -> {json_path}   wall {record['wall_s']:.0f}s   peak RSS {record['peak_rss_gb']:.2f} GB")
    return rc


def _finish(record: dict, json_path: pathlib.Path, started) -> None:
    finished = dt.datetime.now(dt.timezone.utc)
    record["finished_utc"] = finished.isoformat()
    record["wall_s"] = round((finished - started).total_seconds(), 1)
    record["peak_rss_gb"] = round(peak_rss_gb(), 2)
    text = json.dumps(record, indent=2, default=_jsonable)
    for word in VERDICT_WORDS:
        if word.lower() in text.lower():
            raise AssertionError(f"the record contains the verdict word {word!r}; the owner writes the verdict")
    json_path.write_text(text)


def main(argv=None) -> int:
    global log
    args = parse_args(argv)
    started = dt.datetime.now(dt.timezone.utc)
    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
    elif args.smoke:
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="stratum_fold_smoke_"))
    else:
        out_dir = None
    json_path, log_path, preds_path = output_paths(out_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    log = _Log(log_path)
    try:
        if args.smoke:
            log(SMOKE_BANNER)
        log(f"stratum_fold  smoke={args.smoke}  synthetic={args.synthetic}  json={json_path}  log={log_path}  preds={preds_path}")
        log(f"command: {COMMAND}")
        log(f"estimate: {ESTIMATE}")
        rc = run(args, (json_path, log_path, preds_path), started)
    finally:
        if args.smoke:
            log(SMOKE_BANNER)
        log.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
