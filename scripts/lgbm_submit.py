"""Ship the measured LightGBM configuration on the matched rows; copy every other row from v2.

    python3.11 -u scripts/lgbm_submit.py --version 3           # full fit (~1 h), writes submissions/
    python3.11 -u scripts/lgbm_submit.py --version 3 --smoke   # 2 months + 5,000-row sample, scratch dir

`lgb_refit` in scripts/lgbm_ab.py measured 226.24 on the fold's matched rows against the
shipped 237.46 (reports/lgbm_ab_full.log). This script reproduces that procedure on all twelve
training months and splices the result into submissions/merry-quicksand_v2.parquet, changing
ONLY the 339,551 matched rows, so the board measures the matched lane and nothing else.

The procedure, identical to lgbm_ab's refit arm:
  1. the 24 smoothed target encodings, fitted on every training row (stand_ab.infold_encodings)
     and applied to the ranking rows, unseen keys falling back to the prior;
  2. an early-stopping run with months (3, 9) held out of the training months as the stopping
     set - patience 200, up to 50,000 rounds - to find best_iter;
  3. a refit on ALL training months at n_ref = max(50, int(best_iter * n_all / n_fit)) rounds,
     no early stopping;
  4. y_hat = max(proxy - delta_hat, 1), rounded to int32 exactly as build_submission does.

v4 (Amendment 12) adds two OPTIONS, both off by default so the v3 command above is unchanged:

    --seeds 0,1,2   the early-stopping run is done ONCE (seed 0) to find best_iter/n_ref; the
                    refit at n_ref is then done once per seed and delta_hat is the arithmetic
                    mean of the per-seed predictions. Each booster is saved as
                    lgbm_v{N}_seed{s}.txt with its own fit.json; --reuse-booster reuses all of
                    them or none (a partial set is refused). With the default seed set (0,) the
                    booster keeps v3's name, lgbm_v{N}.txt.
    --per-airport   after the pooled prediction, every airport with >= 20,000 training matched
                    rows gets its own LightGBM (same params; tree count = its row share of
                    n_ref, floor 200 - see per_airport_trees), averaged over the same seeds;
                    delta_hat = (1 - BLEND) * pooled + BLEND * per_airport with BLEND = 0.5
                    from build_submission; airports under the floor keep the pooled value.
                    Per-airport models are cheap and are refit on every run, never cached.
    --pa-trees es   replaces the share tree rule with a per-airport early-stopping run on the
                    airport's own rows (PA_TREE_RULE_ES); the default `share` is the rule above.

v6 (Amendment 14) adds one more OPTION, also off by default:

    --queue         append stand_ab.QUEUE_FEATS (the twelve push-anchored queue features cached
                    by `stand_ab.py queue-cache` in data/cache_queue/) to the design matrix and
                    change nothing else. Training months join their queue twin by POSITION
                    (the caches predate MVT_ID_mvt; equal row counts are asserted per month and
                    the identical order is proven by tests/test_queue_features.py); the ranking
                    caches both carry MVT_ID_mvt and join by id with set equality and order
                    asserted. Boosters are named lgbm_v{N}_queue[_seed{s}].txt and their
                    fit.json records the feature count, so a queue booster is never reused by a
                    68-feature run or vice versa.

v5 (Amendment 16) adds the schedule-fill mixture head, also off by default:

    --fillhead      a LightGBM binary classifier p = P(fill | x), fill := |y - sp| <= 60 s on the
                    TRAINING rows (sp = MVT_TIME - SCHED_TIME: the taxi time if the block time is
                    the schedule), on the regressor's design plus one explicit column
                    nmdelay = proxy - sp (= SCHED - AOBT_3), early-stopped on the same ES months
                    (metric auc, patience 200, seed 0) and refit ONCE on all training rows at
                    n_ref exactly like the regressor; the matched prediction becomes the MSE
                    mixture y_hat = p * sp + (1 - p) * max(proxy - delta_hat, 1), rounded as
                    before. The design matrix is built once with 69 columns and the regressors
                    read the first 68 as a view, so their boosters are the v4 ones in every
                    respect. The head booster is lgbm_v{N}[_queue]_fillhead.txt with its own
                    fit.json (params, best_iter, n_ref, n_features, the fill share of the
                    training rows); --reuse-booster reuses it under the same provenance check.

Amendment 19 adds two more OPTIONS, both off by default (the default command is byte-reproducible
against a golden of the pristine file, tests/test_lgbm_submit.py):

    --dayfeats      append stand_ab.DAY_FEATS (the nine airport-day regime features cached by
                    `stand_ab.py day-cache` in data/cache_day/) to the design matrix, after the
                    queue block when both are given: FEATS [+ QUEUE_FEATS] + DAY_FEATS. Joined
                    exactly as the queue block is (months by position, ranking by id; the row
                    contract is tests/test_day_features.py's). Boosters carry a `_day` tag
                    (`_queue_day` with --queue).
    --weatherfeats  append stand_ab.WEATHER_FEATS (the ten airport-hour weather features cached by
                    `stand_ab.py weather-cache` in data/cache_weather/, Amendment 24) after the
                    order block. Boosters carry a `_weather` tag. Not for --all-rows.
    --orderfeats    append stand_ab.ORDER_FEATS (the three record-ordering features cached by
                    `stand_ab.py order-cache` in data/cache_order/, Amendment 22) after the day
                    block: FEATS [+ QUEUE_FEATS] [+ DAY_FEATS] + ORDER_FEATS, joined like the
                    other blocks. Boosters carry an `_order` tag after `_day`. Not built for
                    --all-rows (the unmatched cache carries no order columns).
    --target y      arm Y: the regressor's label is y (TAXITIME) instead of delta, the stopping
                    metric is therefore RMSE on y directly, and the matched prediction is
                    y_hat = max(prediction, 1) with NO proxy subtraction (Amendment 19.2). The
                    splice is unchanged. Boosters carry a `_ytarget` tag and their fit.json
                    records the target, so a y booster is never reused by a delta run or the
                    reverse. Not combinable with --fillhead (not pre-registered; RESULT 8).

The UNIFIED all-rows arm (the Amendment 19 addition) adds two more OPTIONS:

    --all-rows      with --target y only: ONE regressor on EVERY admissible training row, matched
                    AND unmatched - the unmatched rows from data/cache_unmatched (stand_ab.py
                    unmatched-cache) with every AOBT_3-anchored column NaN exactly as LightGBM
                    handles missing values (the pattern is asserted on every cache file, never
                    filled) and an explicit is_unmatched 0/1 column appended LAST to the design;
                    the delta encodings fitted on the matched training rows only; y_hat =
                    max(prediction, 1) on every scored row, matched and unmatched alike, and the
                    splice replaces ALL 344,841 rows. No mixture, no proxy anywhere. Boosters carry
                    `_allrows` (and `_w{W}` below). Not combinable with --fillhead / --per-airport.
    --unmatched-weight W  the sample weight of the unmatched training rows (default 1.0; they are
                    ~22k of 2.08M rows); the stopping set is unweighted (the competition's metric).

Deliberately NOT here: the stand block (not measured on LightGBM). A smoke run proves the code
path and nothing else - it prints a banner saying so, because this project has twice quoted a
smoke magnitude as a result.

Memory: the design matrix is one preallocated float32 array filled column by column; string
key columns are dropped right after the encodings are fitted; the concatenated frame is
released before LightGBM allocates. Every print flushes - a long run's log otherwise stays
empty until exit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.util
import json
import pathlib
import resource
import subprocess
import sys
import tempfile
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache_stand"
SUBS = ROOT / "submissions"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("stand_ab")
B = _load("build_submission")

ES_MONTHS = (3, 9)                       # stopping set, carved out of the training months
N_MATCHED, N_UNMATCHED = 339_551, 5_290  # ranking.parquet: departures with / without an NM off-block
FEATS = S.BASELINE_FEATS
#: v6 (Amendment 14): the queue block's cache directory and column contract, one place.
QCACHE = S.QCACHE
QUEUE_FEATS = list(S.QUEUE_FEATS)
#: Amendment 19 arm D: the day block's cache directory and column contract, one place.
DCACHE = S.DCACHE
DAY_FEATS = list(S.DAY_FEATS)
#: Amendment 22 arm F: the order block's cache directory and column contract, one place.
OCACHE = S.OCACHE
ORDER_FEATS = list(S.ORDER_FEATS)
#: Amendment 24 arm W: the weather block's cache directory and column contract, one place.
WCACHE = S.WCACHE
WEATHER_FEATS = list(S.WEATHER_FEATS)
#: Amendment 19 arm Y: the regressor's label. `delta` (the default, v3 onwards) anchors the
#: prediction on proxy, y_hat = max(proxy - delta_hat, 1); `y` predicts the taxi time itself,
#: y_hat = max(prediction, 1), no proxy anchor, the stopping metric on y directly.
TARGET_DELTA, TARGET_Y = "delta", "y"
TARGETS = (TARGET_DELTA, TARGET_Y)
#: the unified all-rows arm: the unmatched cache's directory and contract, the flag column
UCACHE = S.UCACHE
UNMATCHED_COLS = list(S.UNMATCHED_COLS)
IS_UNMATCHED = "is_unmatched"
UNMATCHED_HINT = "stand_ab.py unmatched-cache [--ranking]"
#: a fit.json before the unified arm is a matched-only, weight-1 booster
LEGACY_PROVENANCE = {"all_rows": False, "unmatched_weight": 1.0}
#: the parameters lgbm_ab.py measured (lgb_refit 226.24). Do not tune here.
P = dict(objective="regression", metric="rmse", learning_rate=0.01, num_leaves=255,
         min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
         cat_smooth=20, min_data_per_group=100, max_cat_threshold=64, num_threads=4,
         verbosity=-1, seed=0)
NEST, PATIENCE = 50_000, 200
ES_SEED = 0                              # the early-stopping run is always seed 0 (Amendment 12)
#: per-airport models (Amendment 12, H-12b). The row floor is build_submission.fit_matched's;
#: the blend weight is build_submission's so one place knows it; the tree rule is the
#: implementer's judgement call recorded in per_airport_trees.
PA_MIN_ROWS = 20_000
PA_TREE_FLOOR = 200
BLEND = B.BLEND
PA_TREE_RULE = "max(floor, int(n_ref * n_airport / n_all))"
#: --pa-trees es: the measured alternative - each airport early-stops on its OWN fit/ES rows
#: (the pooled masks intersected with the airport, same ES months, seed 0) and refits at its own
#: n_ref. Costs about one more pooled ES run plus larger per-airport refits.
PA_TREE_RULE_ES = ("per-airport early stopping on the airport's own fit/ES rows (same ES months, "
                   "seed 0); n_ref_a = max(50, int(best_iter_a * n_all_a / n_fit_a))")
#: the smoke deviates only where a code-path check needs it to finish in seconds: pa_floor
#: replaces PA_TREE_FLOOR (200 trees x 10 airports x seeds is a minute; 20 is seconds)
SMOKE = dict(months=(1, 3), n_rank=5_000, learning_rate=0.05, nest=60, patience=50, pa_floor=20)
SMOKE_BANNER = "SMOKE — code path only, NOT a result"
#: gross-breakage guard on the stopping set. Measured 2026-09-09 on the caches: the proxy-only
#: predictor (delta_hat = mean delta) scores 342 (Mar) / 388 (Sep) / 449 (Jul), and the
#: measured configuration scores 226 on Jan+Jul, i.e. 0.5-0.65 of proxy-only. A model that
#: fails to beat proxy-only by 15% on its own stopping set has dead features, not bad luck.
ES_MAX_RATIO_TO_PROXY_ONLY = 0.85
PROGRESS_EVERY = 1_000

# ---- v5 (Amendment 16): the schedule-fill mixture head --------------------------------------
#: the one column the head sees that the regressors do not: proxy - sp = SCHED - AOBT_3, the NM
#: off-block delay against the schedule; appended LAST so the regressors read X[:, :len(FEATS)]
NMDELAY = "nmdelay"
#: fill := |y - sp| <= 60 s, inclusive (Amendment 16.2). The head's training label and the
#: fold's fill / non-fill holdout subsets both read this one number.
FILL_TOL_S = 60.0
#: a head whose stopping-set AUC is at or below chance learned nothing: refuse to refit it
FILL_HEAD_MIN_ES_AUC = 0.5


def head_params(params: dict) -> dict:
    """The head's LightGBM parameters: the regressor's with exactly objective and metric
    replaced (binary, auc) - the same learning rate, leaves, min_data_in_leaf, sampling, seed
    and threads (Amendment 16.1)."""
    return dict(params, objective="binary", metric="auc")


P_HEAD = head_params(P)

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.0f}s] {msg}", flush=True)


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}")
    if v < 1:
        raise argparse.ArgumentTypeError(f"version must be >= 1, got {v}")
    return v


def parse_seeds(text: str) -> tuple:
    """"0,1,2" -> (0, 1, 2): non-negative integers, in the order given, no repeats."""
    parts = [p.strip() for p in str(text).split(",")]
    if not parts or any(p == "" for p in parts):
        raise argparse.ArgumentTypeError(f"--seeds must be a comma-separated list, got {text!r}")
    try:
        seeds = tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--seeds must be integers, got {text!r}")
    if any(s < 0 for s in seeds):
        raise argparse.ArgumentTypeError(f"--seeds must be >= 0, got {text!r}")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"--seeds repeats a seed (it would be double counted "
                                         f"in the mean): {text!r}")
    return seeds


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", type=_positive_int, required=True,
                    help="submission version N; written as <team>_vN.parquet")
    ap.add_argument("--smoke", action="store_true",
                    help="two months, a 5,000-row ranking sample, a tiny booster, scratch dir")
    ap.add_argument("--out-dir", default=None,
                    help="where the parquet, meta.json (and in smoke mode the booster) go; "
                         "default submissions/ for a real run, a temp dir for --smoke")
    ap.add_argument("--base", default=str(SUBS / "merry-quicksand_v2.parquet"),
                    help="the submission whose unmatched rows are copied verbatim")
    ap.add_argument("--reuse-booster", action="store_true",
                    help="skip the fit if every seed's saved booster and fit.json exist "
                         "(a partial seed set is refused)")
    ap.add_argument("--seeds", type=parse_seeds, default="0",
                    help="refit seeds; delta_hat is the mean over them (default: 0 = v3)")
    ap.add_argument("--per-airport", action="store_true",
                    help=f"blend the pooled prediction {1 - BLEND:.1f}/{BLEND:.1f} with a "
                         f"per-airport model at airports with >= {PA_MIN_ROWS:,} training rows")
    ap.add_argument("--pa-trees", choices=("share", "es"), default="share",
                    help="per-airport tree count: 'share' = the airport's row share of n_ref "
                         "(floor 200); 'es' = its own early-stopping run (default: share)")
    ap.add_argument("--queue", action="store_true",
                    help="append stand_ab.QUEUE_FEATS from data/cache_queue/ to the design matrix "
                         "(v6, Amendment 14); boosters are tagged _queue")
    ap.add_argument("--fillhead", action="store_true",
                    help=f"v5 (Amendment 16): fit the schedule-fill head p = P(|y - sp| <= {FILL_TOL_S:.0f} s | x) "
                         f"on the design + {NMDELAY} and ship p * sp + (1 - p) * max(proxy - delta_hat, 1); "
                         "the head booster is lgbm_vN[_queue]_fillhead.txt")
    ap.add_argument("--dayfeats", action="store_true",
                    help="append stand_ab.DAY_FEATS from data/cache_day/ to the design matrix, after the "
                         "queue block (Amendment 19 arm D); boosters are tagged _day / _queue_day")
    ap.add_argument("--weatherfeats", action="store_true",
                    help="append stand_ab.WEATHER_FEATS from data/cache_weather/ after the order block "
                         "(Amendment 24 arm W); boosters carry a _weather tag; not available with --all-rows")
    ap.add_argument("--orderfeats", action="store_true",
                    help="append stand_ab.ORDER_FEATS from data/cache_order/ after the day block (Amendment 22 "
                         "arm F); boosters carry an _order tag; not available with --all-rows")
    ap.add_argument("--target", choices=TARGETS, default=TARGET_DELTA,
                    help="the regressor's label: delta (default; y_hat = max(proxy - delta_hat, 1)) or y "
                         "(Amendment 19 arm Y; y_hat = max(prediction, 1), no proxy anchor); boosters are "
                         "tagged _ytarget")
    ap.add_argument("--all-rows", action="store_true",
                    help="with --target y: ONE regressor on matched AND unmatched training rows (data/cache_unmatched), "
                         "is_unmatched appended to the design, every scored row re-predicted; boosters tagged _allrows")
    ap.add_argument("--unmatched-weight", type=float, default=1.0,
                    help="--all-rows: the sample weight of the unmatched training rows (default 1.0); boosters tagged _w{W}")
    args = ap.parse_args(argv)
    if args.fillhead and args.target == TARGET_Y:
        ap.error("--fillhead with --target y is not pre-registered (the head was NOT WORKING in RESULT 8 "
                 "and Amendment 19.2 measures the formulation on the regressor alone)")
    if args.all_rows:
        if args.target != TARGET_Y:
            ap.error("--all-rows needs --target y: the delta formulation has no proxy on the unmatched rows")
        if args.fillhead or args.per_airport:
            ap.error("--all-rows does not combine with --fillhead or --per-airport (not pre-registered for the unified arm)")
        if args.orderfeats or args.weatherfeats:
            ap.error("--all-rows does not combine with --orderfeats or --weatherfeats: the unmatched cache "
                     "carries neither block's columns")
        if not (np.isfinite(args.unmatched_weight) and args.unmatched_weight > 0):
            ap.error(f"--unmatched-weight must be a positive finite number, got {args.unmatched_weight}")
    return args


def booster_tag(queue: bool = False, day: bool = False, target: str = TARGET_DELTA, all_rows: bool = False,
                unmatched_weight: float = 1.0, order: bool = False, weather: bool = False) -> str:
    """The configuration tag in a booster's file name: `_queue` / `_day` / `_order` (any
    combination, in that order) for the block(s) in the design, `_ytarget` for the y
    formulation, `_allrows` for the unified arm and `_w{W}` for a non-unit unmatched weight, in
    that order, empty for the v3/v4 design on delta - so a booster can never be reused by another
    configuration."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    tag = (("_queue" if queue else "") + ("_day" if day else "") + ("_order" if order else "")
           + ("_weather" if weather else "") + ("_ytarget" if target == TARGET_Y else ""))
    if all_rows:
        tag += "_allrows"
        if float(unmatched_weight) != 1.0:
            tag += f"_w{float(unmatched_weight):g}"
    return tag


def booster_files(directory: pathlib.Path, version: int, seeds, queue: bool = False, day: bool = False,
                  target: str = TARGET_DELTA, all_rows: bool = False, unmatched_weight: float = 1.0,
                  order: bool = False, weather: bool = False) -> list:
    """(model file, fit.json) per seed. The default seed set keeps v3's names so that
    --reuse-booster still finds lgbm_v3.txt; any other set names one file per seed. A --queue
    run carries a `_queue` tag so its 80-feature boosters can never be reused by a 68-feature
    run (nor the reverse); --dayfeats, --target y and --all-rows add `_day`, `_ytarget`,
    `_allrows[_w{W}]` (booster_tag)."""
    directory = pathlib.Path(directory)
    tag = booster_tag(queue, day, target, all_rows, unmatched_weight, order=order, weather=weather)
    if tuple(seeds) == (ES_SEED,):
        return [(directory / f"lgbm_v{version}{tag}.txt", directory / f"lgbm_v{version}{tag}.fit.json")]
    return [(directory / f"lgbm_v{version}{tag}_seed{s}.txt",
             directory / f"lgbm_v{version}{tag}_seed{s}.fit.json") for s in seeds]


def head_booster_files(directory: pathlib.Path, version: int, queue: bool = False, day: bool = False,
                       order: bool = False, weather: bool = False) -> tuple:
    """(model file, fit.json) of the fill head: lgbm_v{N}[_queue][_day][_order]_fillhead.txt - its
    own name, so it can never be mistaken for a regressor booster by --reuse-booster, and the
    block tags for the same reason the regressors carry them (the head is delta-only)."""
    directory = pathlib.Path(directory)
    tag = booster_tag(queue, day, order=order, weather=weather)
    return (directory / f"lgbm_v{version}{tag}_fillhead.txt", directory / f"lgbm_v{version}{tag}_fillhead.fit.json")


def output_name(version: int) -> str:
    """One place knows the filename rule: build_submission.submission_name."""
    return B.submission_name(version)


def n_refit(best_iter: int, n_all: int, n_fit: int) -> int:
    """lgbm_ab's refit count: best_iter scaled by the data ratio, truncated, floored at 50."""
    return max(50, int(best_iter * n_all / n_fit))


def split_masks(month, is_rank, es_months=ES_MONTHS):
    """(train, fit, early-stop) masks. Ranking rows are in none of the fit/es sets."""
    month, is_rank = np.asarray(month), np.asarray(is_rank, dtype=bool)
    train = ~is_rank
    es = train & np.isin(month, es_months)
    fit = train & ~es
    if not es.any():
        raise ValueError(f"no early-stopping rows: months {es_months} are absent")
    if not fit.any():
        raise ValueError("no fit rows: every training month is an early-stopping month")
    return train, fit, es


def finalise_taxi_time(raw) -> np.ndarray:
    """A predicted taxi time (float64) -> the submission's int32: max(raw, 1) rounded with np.rint,
    exactly as build_submission.main rounds. Both the regressor path (recover_taxi_time) and the
    fill-head mixture (a taxi time already, not a delta) finish here, so v5's rows are on v4's scale.

    The floor at 1 s is required (taxi-out is strictly positive) but must not do real work:
    if it binds on more than 0.5% of rows the model is broken, not merely imprecise.
    """
    raw = np.asarray(raw, dtype="float64")
    assert np.isfinite(raw).all(), f"non-finite predictions: {int((~np.isfinite(raw)).sum())}"
    bound = float((raw < 1.0).mean())
    assert bound < 0.005, f"positivity floor binds on {100 * bound:.2f}% of rows"
    return np.rint(np.maximum(raw, 1.0)).astype("int32")


def recover_taxi_time(proxy, delta_hat) -> np.ndarray:
    """y_hat = max(proxy - delta_hat, 1), rounded like build_submission.main (np.rint, int32):
    finalise_taxi_time on the recovered taxi time."""
    return finalise_taxi_time(np.asarray(proxy, dtype="float64") - np.asarray(delta_hat, dtype="float64"))


def regression_label(dlt, y, target: str = TARGET_DELTA) -> np.ndarray:
    """The regressor's label under the formulation (Amendment 19.2): delta = BLOCK - AOBT_3 by
    default - the anchor the prediction is subtracted from proxy - or y = TAXITIME itself under
    `y`. Float64, the values untouched; an unknown target is refused."""
    if target == TARGET_DELTA:
        return np.asarray(dlt, dtype="float64")
    if target == TARGET_Y:
        return np.asarray(y, dtype="float64")
    raise ValueError(f"target must be one of {TARGETS}, got {target!r}")


def raw_taxi_time(pred, proxy, target: str = TARGET_DELTA) -> np.ndarray:
    """The unfloored taxi time a prediction implies: proxy - pred under `delta`; pred ITSELF under
    `y` - proxy is not read there (no anchor, 19.2). finalise_taxi_time / np.maximum(., 1) floor
    it; under `delta` finalise_taxi_time(raw_taxi_time(d, p)) is recover_taxi_time(p, d) exactly."""
    if target == TARGET_DELTA:
        return np.asarray(proxy, dtype="float64") - np.asarray(pred, dtype="float64")
    if target == TARGET_Y:
        return np.asarray(pred, dtype="float64")
    raise ValueError(f"target must be one of {TARGETS}, got {target!r}")


def splice(base: pd.DataFrame, ids, values, expected_ids) -> pd.DataFrame:
    """Replace TAXITIME_SEC_mvt on exactly `expected_ids`; leave every other row untouched.

    Refuses rather than partially writes: the predicted id set must equal the matched id
    set, every id must exist in the base, no id may repeat, and the values must already be
    positive int32 (the base's dtype - a float intermediate would rewrite unmatched rows).
    Row order and the identifier column are preserved exactly.
    """
    if list(base.columns) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]:
        raise ValueError(f"base columns {list(base.columns)}")
    if base.TAXITIME_SEC_mvt.dtype != np.int32:
        raise ValueError(f"base predictions are {base.TAXITIME_SEC_mvt.dtype}, expected int32")
    ids = np.asarray(ids, dtype="float64")
    values = np.asarray(values)
    if values.dtype != np.int32:
        raise ValueError(f"predictions must be int32 like the base, got {values.dtype}")
    if len(ids) != len(values):
        raise ValueError(f"{len(ids)} ids for {len(values)} predictions")
    if not pd.Index(ids).is_unique:
        raise ValueError(f"duplicate ids among the predictions: {len(ids) - pd.Index(ids).nunique()}")
    if (values <= 0).any():
        raise ValueError(f"{int((values <= 0).sum())} non-positive predictions")
    base_ids = base.MVT_ID_mvt.to_numpy()
    foreign = set(ids) - set(base_ids)
    if foreign:
        raise ValueError(f"{len(foreign)} predicted ids are not in the base submission")
    expected_ids = set(expected_ids)
    missing, extra = expected_ids - set(ids), set(ids) - expected_ids
    if missing or extra:
        raise ValueError(f"predicted ids != matched id set: {len(missing)} matched ids without "
                         f"a prediction, {len(extra)} predictions outside the matched set")
    lookup = pd.Series(values, index=ids)
    hit = base.MVT_ID_mvt.isin(ids).to_numpy()
    new = base.TAXITIME_SEC_mvt.to_numpy().copy()
    new[hit] = lookup.loc[base_ids[hit]].to_numpy()
    assert new.dtype == np.int32
    return pd.DataFrame({"MVT_ID_mvt": base_ids.copy(), "TAXITIME_SEC_mvt": new})


def matched_ids(raw_ranking: pathlib.Path, template: pd.DataFrame) -> set:
    """The matched id set from the RAW evaluation file, independent of any cache."""
    r = pq.read_table(raw_ranking, columns=["MVT_ID_mvt", "PHASE_mvt", "AOBT_3_flt"]).to_pandas()
    keep = (r.PHASE_mvt == "DEP") & r.AOBT_3_flt.notna() & r.MVT_ID_mvt.isin(set(template.MVT_ID_mvt))
    return set(r.MVT_ID_mvt[keep])


def _month_of(path: pathlib.Path) -> int:
    return int(path.name.split("_")[1][5:7])          # training_2025-MM-01_...


def training_frames(cache_dir: pathlib.Path, months=None):
    """The training month caches, schema-checked against each other, labels asserted present.

    Returns (frames, paths) unconcatenated so each caller concatenates exactly once (the
    submission appends the ranking cache; the fold harness uses the months alone).
    """
    paths = sorted(cache_dir.glob("training_2025-*.parquet"))
    if months is not None:
        paths = [p for p in paths if _month_of(p) in months]
    if not paths:
        raise FileNotFoundError(f"no training caches under {cache_dir} (run stand_ab.py cache)")
    frames = [pd.read_parquet(p) for p in paths]
    cols = list(frames[0].columns)
    for f, p in zip(frames, paths):
        if list(f.columns) != cols:
            raise ValueError(f"{p.name}: cache schema differs from {paths[0].name}")
        if f.y.isna().any() or f.delta.isna().any():
            raise ValueError(f"{p.name}: training cache has NaN labels - built in serve mode?")
    return frames, paths


def queue_cache_path(qcache: pathlib.Path, training_path: pathlib.Path) -> pathlib.Path:
    """The queue twin of a stand cache file: the same file name under data/cache_queue/."""
    return pathlib.Path(qcache) / pathlib.Path(training_path).name


def day_cache_path(dcache: pathlib.Path, training_path: pathlib.Path) -> pathlib.Path:
    """The day twin of a stand cache file: the same file name under data/cache_day/."""
    return pathlib.Path(dcache) / pathlib.Path(training_path).name


def order_cache_path(ocache: pathlib.Path, training_path: pathlib.Path) -> pathlib.Path:
    """The order twin of a stand cache file: the same file name under data/cache_order/."""
    return pathlib.Path(ocache) / pathlib.Path(training_path).name


def weather_cache_path(wcache: pathlib.Path, training_path: pathlib.Path) -> pathlib.Path:
    """The weather twin of a stand cache file: the same file name under data/cache_weather/."""
    return pathlib.Path(wcache) / pathlib.Path(training_path).name


# ---- feature-block caches. The queue block (v6) and the day block (Amendment 19) share one join
# ---- discipline; each is an instance with its own column contract, directory and build hint. ----

def _check_block_columns(q: pd.DataFrame, name: str, feats, what: str) -> None:
    want = ["MVT_ID_mvt"] + list(feats)
    if list(q.columns) != want:
        raise ValueError(f"{name}: {what} cache columns {list(q.columns)} != the contract {want}")


def read_block_cache(path: pathlib.Path, feats, what: str, hint: str) -> pd.DataFrame:
    """One block cache file, its column contract asserted; a missing file names the build command."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{what} cache {path} missing (run {hint})")
    q = pd.read_parquet(path)
    _check_block_columns(q, path.name, feats, what)
    return q


def attach_block_positional(n_rows: int, q: pd.DataFrame, name: str, feats, what: str) -> pd.DataFrame:
    """A block of a training month, joined by POSITION: the training caches predate MVT_ID_mvt,
    and each block's builder emits the stand cache's rows in the stand cache's order (same
    filter, same mergesort; proven on March both ways by tests/test_queue_features.py and
    tests/test_day_features.py). The only guard a positional join can carry is the row count,
    so it is asserted per month; a mismatch is a refusal, never a truncation."""
    _check_block_columns(q, name, feats, what)
    if len(q) != n_rows:
        raise ValueError(f"{name}: {what} cache has {len(q):,} rows, training cache {n_rows:,}: not the "
                         "same build, refusing a positional join")
    return q[list(feats)].reset_index(drop=True)


def attach_block_by_id(ids, q: pd.DataFrame, feats, what: str, subset_ok: bool = False) -> pd.DataFrame:
    """A block for ranking rows, joined by MVT_ID_mvt and returned in the order of `ids`. The
    full join (subset_ok=False) requires the SAME id set in the SAME order - every ranking cache
    is built per calendar month from the raw file in take-off order, so a difference means one
    was rebuilt differently and is refused; a sampled join (the smoke) aligns the subset by id.
    Duplicates and missing ids are refusals."""
    _check_block_columns(q, f"ranking {what} cache", feats, what)
    ids = np.asarray(ids, dtype="float64")
    if not pd.Index(ids).is_unique:
        raise ValueError(f"duplicate ids among the rows to join: {len(ids) - pd.Index(ids).nunique()}")
    qi = q.MVT_ID_mvt.to_numpy(dtype="float64")
    if not pd.Index(qi).is_unique:
        raise ValueError(f"duplicate MVT_ID in the {what} cache: {len(qi) - pd.Index(qi).nunique()}")
    missing = set(ids) - set(qi)
    if missing:
        raise ValueError(f"{len(missing):,} rows are missing from the {what} cache (no {what} features)")
    if not subset_ok:
        extra = set(qi) - set(ids)
        if extra:
            raise ValueError(f"{len(extra):,} extra rows in the {what} cache: the id sets differ")
        if not np.array_equal(qi, ids):
            raise ValueError(f"the {what} cache's rows are in a different order from the stand cache's: "
                             "not the same build")
    return q.set_index("MVT_ID_mvt").loc[ids, list(feats)].reset_index(drop=True)


QUEUE_HINT, DAY_HINT = "stand_ab.py queue-cache [--ranking]", "stand_ab.py day-cache [--ranking]"
ORDER_HINT = "stand_ab.py order-cache [--ranking]"
WEATHER_HINT = "stand_ab.py weather-cache [--ranking]"


def _check_queue_columns(q: pd.DataFrame, name: str) -> None:
    _check_block_columns(q, name, QUEUE_FEATS, "queue")


def read_queue_cache(path: pathlib.Path) -> pd.DataFrame:
    """One queue cache file, its column contract asserted."""
    return read_block_cache(path, QUEUE_FEATS, "queue", QUEUE_HINT)


def attach_queue_positional(n_rows: int, q: pd.DataFrame, name: str) -> pd.DataFrame:
    """The queue block of a training month, joined by position (attach_block_positional)."""
    return attach_block_positional(n_rows, q, name, QUEUE_FEATS, "queue")


def attach_queue_by_id(ids, q: pd.DataFrame, subset_ok: bool = False) -> pd.DataFrame:
    """The queue block for ranking rows, joined by id (attach_block_by_id)."""
    return attach_block_by_id(ids, q, QUEUE_FEATS, "queue", subset_ok)


def read_day_cache(path: pathlib.Path) -> pd.DataFrame:
    """One day cache file, its column contract asserted."""
    return read_block_cache(path, DAY_FEATS, "day", DAY_HINT)


def attach_day_positional(n_rows: int, q: pd.DataFrame, name: str) -> pd.DataFrame:
    """The day block of a training month, joined by position (attach_block_positional)."""
    return attach_block_positional(n_rows, q, name, DAY_FEATS, "day")


def attach_day_by_id(ids, q: pd.DataFrame, subset_ok: bool = False) -> pd.DataFrame:
    """The day block for ranking rows, joined by id (attach_block_by_id)."""
    return attach_block_by_id(ids, q, DAY_FEATS, "day", subset_ok)


def read_order_cache(path: pathlib.Path) -> pd.DataFrame:
    """One order cache file (Amendment 22), its column contract asserted."""
    return read_block_cache(path, ORDER_FEATS, "order", ORDER_HINT)


def attach_order_positional(n_rows: int, q: pd.DataFrame, name: str) -> pd.DataFrame:
    """The order block of a training month, joined by position (attach_block_positional)."""
    return attach_block_positional(n_rows, q, name, ORDER_FEATS, "order")


def attach_order_by_id(ids, q: pd.DataFrame, subset_ok: bool = False) -> pd.DataFrame:
    """The order block for ranking rows, joined by id (attach_block_by_id)."""
    return attach_block_by_id(ids, q, ORDER_FEATS, "order", subset_ok)


def read_weather_cache(path: pathlib.Path) -> pd.DataFrame:
    """One weather cache file (Amendment 24), its column contract asserted."""
    return read_block_cache(path, WEATHER_FEATS, "weather", WEATHER_HINT)


def attach_weather_positional(n_rows: int, q: pd.DataFrame, name: str) -> pd.DataFrame:
    """The weather block of a training month, joined by position (attach_block_positional)."""
    return attach_block_positional(n_rows, q, name, WEATHER_FEATS, "weather")


def attach_weather_by_id(ids, q: pd.DataFrame, subset_ok: bool = False) -> pd.DataFrame:
    """The weather block for ranking rows, joined by id (attach_block_by_id)."""
    return attach_block_by_id(ids, q, WEATHER_FEATS, "weather", subset_ok)


# ---- the unified all-rows arm: the unmatched cache ----------------------------------------------

def unmatched_cache_path(ucache: pathlib.Path, training_path: pathlib.Path) -> pathlib.Path:
    """The unmatched twin of a stand cache file: the same file name under data/cache_unmatched/."""
    return pathlib.Path(ucache) / pathlib.Path(training_path).name


def read_unmatched_cache(path: pathlib.Path) -> pd.DataFrame:
    """One unmatched cache file, its contract (stand_ab.UNMATCHED_COLS, in order) asserted."""
    return read_block_cache(path, UNMATCHED_COLS[1:], "unmatched", UNMATCHED_HINT)


def check_unmatched_nan_pattern(frame: pd.DataFrame, name: str = "unmatched cache") -> None:
    """Refuse an unmatched frame in which an AOBT_3-anchored column (stand_ab.UNMATCHED_NAN_COLS)
    or a *_flt-derived one (UNMATCHED_FLT_COLS) carries a value: the unified model must see the
    same missingness on the scored unmatched rows as in training, and a cache with those columns
    filled - 0, a mean, anything - would train silently on a pattern that never serves."""
    bad = []
    for c in S.UNMATCHED_NAN_COLS + S.UNMATCHED_FLT_COLS:
        if c in frame.columns:
            n = int(frame[c].notna().sum())
            if n:
                bad.append(f"{c} ({n:,} of {len(frame):,} rows)")
    if bad:
        raise ValueError(f"{name}: unmatched rows must carry NaN on the AOBT_3-anchored and *_flt-derived columns, "
                         "but these carry values: " + ", ".join(bad))


def row_weights(is_unmatched, unmatched_weight) -> np.ndarray:
    """LightGBM sample weights: 1.0 on matched rows, `unmatched_weight` on unmatched rows, float64;
    a non-positive or non-finite weight is refused."""
    w = float(unmatched_weight)
    if not np.isfinite(w) or w <= 0.0:
        raise ValueError(f"the unmatched weight must be a positive finite number, got {unmatched_weight!r}")
    return np.where(np.asarray(is_unmatched).astype(bool), w, 1.0).astype("float64")


def load_frames(cache_dir: pathlib.Path, months=None, n_rank=None, seed=0, queue=False, qcache=None,
                day=False, dcache=None, all_rows=False, ucache=None, order=False, ocache=None,
                weather=False, wcache=None):
    """Concatenate the training caches and the ranking cache.

    Returns (frame, is_rank, rank_ids). The training caches on disk predate MVT_ID_mvt and
    the ranking cache carries it; the id column is lifted out before the concat so every
    frame has the same schema and the ranking cache's schema is asserted against training.
    With queue=True the twelve QUEUE_FEATS are appended to every training month by position
    and to the ranking rows by id (attach_queue_positional / attach_queue_by_id); with day=True
    the nine DAY_FEATS follow them the same way (attach_day_positional / attach_day_by_id), and
    with order=True the three ORDER_FEATS follow those (Amendment 22), so the column order is
    always FEATS [+ QUEUE_FEATS] [+ DAY_FEATS] [+ ORDER_FEATS]; order with all_rows is refused
    (the unmatched cache carries no order columns).

    With all_rows=True (the unified arm) the unmatched months from `ucache` follow the matched
    months and the unmatched scored rows follow the matched ranking rows, aligned by name (the
    matched caches' STAND_BLOCK columns are NaN on unmatched rows; blocks not asked for are
    dropped; the unmatched cache carries its own queue and day columns), their NaN pattern
    checked file by file, and the frame carries IS_UNMATCHED (1.0 on them, 0.0 elsewhere) as
    its LAST column; a sampled ranking then samples matched and unmatched rows together.
    """
    rank_path = cache_dir / "ranking.parquet"
    if not rank_path.exists():
        raise FileNotFoundError(f"{rank_path} missing (run stand_ab.py cache --ranking)")
    if (order or weather) and all_rows:
        raise ValueError("the order and weather blocks are not built for the unmatched cache: "
                         "order/weather with all_rows=True")
    frames, paths = training_frames(cache_dir, months)
    if queue:
        qcache = QCACHE if qcache is None else pathlib.Path(qcache)
        for f, p in zip(frames, paths):
            qf = attach_queue_positional(len(f), read_queue_cache(queue_cache_path(qcache, p)), p.name)
            for c in QUEUE_FEATS:
                f[c] = qf[c].to_numpy()
    if day:
        dcache = DCACHE if dcache is None else pathlib.Path(dcache)
        for f, p in zip(frames, paths):
            df_ = attach_day_positional(len(f), read_day_cache(day_cache_path(dcache, p)), p.name)
            for c in DAY_FEATS:
                f[c] = df_[c].to_numpy()
    if order:
        ocache = OCACHE if ocache is None else pathlib.Path(ocache)
        for f, p in zip(frames, paths):
            of = attach_order_positional(len(f), read_order_cache(order_cache_path(ocache, p)), p.name)
            for c in ORDER_FEATS:
                f[c] = of[c].to_numpy()
    if weather:
        wcache = WCACHE if wcache is None else pathlib.Path(wcache)
        for f, p in zip(frames, paths):
            wf = attach_weather_positional(len(f), read_weather_cache(weather_cache_path(wcache, p)), p.name)
            for c in WEATHER_FEATS:
                f[c] = wf[c].to_numpy()
    u_frames, u_rank = [], None
    if all_rows:
        ucache = UCACHE if ucache is None else pathlib.Path(ucache)
        for p in paths:
            uf = read_unmatched_cache(unmatched_cache_path(ucache, p))
            check_unmatched_nan_pattern(uf, p.name)
            u_frames.append(uf)
        u_rank = read_unmatched_cache(unmatched_cache_path(ucache, rank_path))
        check_unmatched_nan_pattern(u_rank, "ranking.parquet (unmatched)")
    rank = pd.read_parquet(rank_path)
    sample_matched = n_rank is not None and not all_rows
    if sample_matched:
        rank = rank.sample(n=n_rank, random_state=seed)
    rank_ids = rank.MVT_ID_mvt.to_numpy(dtype="float64")
    rank = rank.drop(columns=["MVT_ID_mvt"])
    if queue:
        qr = attach_queue_by_id(rank_ids, read_queue_cache(queue_cache_path(qcache, rank_path)),
                                subset_ok=sample_matched)
        for c in QUEUE_FEATS:
            rank[c] = qr[c].to_numpy()
    if day:
        dr = attach_day_by_id(rank_ids, read_day_cache(day_cache_path(dcache, rank_path)),
                              subset_ok=sample_matched)
        for c in DAY_FEATS:
            rank[c] = dr[c].to_numpy()
    if order:
        orr = attach_order_by_id(rank_ids, read_order_cache(order_cache_path(ocache, rank_path)),
                                 subset_ok=sample_matched)
        for c in ORDER_FEATS:
            rank[c] = orr[c].to_numpy()
    if weather:
        wr = attach_weather_by_id(rank_ids, read_weather_cache(weather_cache_path(wcache, rank_path)),
                                  subset_ok=sample_matched)
        for c in WEATHER_FEATS:
            rank[c] = wr[c].to_numpy()
    cols = list(frames[0].columns)
    if set(rank.columns) != set(cols):
        raise ValueError(f"ranking cache schema differs: {set(rank.columns) ^ set(cols)}")
    if not (rank.y.isna().all() and rank.delta.isna().all()):
        raise ValueError("ranking cache carries labels - not a serve-mode build")
    if not set(rank.month.unique()) <= {1, 7}:
        raise ValueError(f"ranking cache months {sorted(rank.month.unique())}, expected 1 and 7")
    if not pd.Index(rank_ids).is_unique:
        raise ValueError("duplicate MVT_ID in the ranking cache")
    if not all_rows:
        d = pd.concat(frames + [rank[cols]], ignore_index=True)
        is_rank = np.zeros(len(d), dtype=bool)
        is_rank[len(d) - len(rank):] = True
        log(f"loaded {len(paths)} training months ({(~is_rank).sum():,} rows) + ranking "
            f"({is_rank.sum():,} rows); peak RSS {peak_rss_gb():.2f} GB")
        return d, is_rank, rank_ids

    # ---- the unified frame: matched months, unmatched months, matched + unmatched scored rows ----
    if not (u_rank.y.isna().all() and u_rank.delta.isna().all()):
        raise ValueError("the unmatched ranking cache carries labels - not a serve-mode build")
    if not set(u_rank.month.unique()) <= {1, 7}:
        raise ValueError(f"unmatched ranking cache months {sorted(u_rank.month.unique())}, expected 1 and 7")
    u_ids = u_rank.MVT_ID_mvt.to_numpy(dtype="float64")
    if not pd.Index(u_ids).is_unique or set(u_ids) & set(rank_ids):
        raise ValueError("the unmatched ranking cache repeats an id, or shares one with the matched ranking cache")
    n_m_train = sum(len(f) for f in frames)
    n_u_train = sum(len(f) for f in u_frames)
    rank_all = pd.concat([rank[cols].assign(**{IS_UNMATCHED: 0.0, "MVT_ID_mvt": rank_ids}),
                          u_rank.reindex(columns=cols).assign(**{IS_UNMATCHED: 1.0, "MVT_ID_mvt": u_ids})],
                         ignore_index=True)
    if n_rank is not None:
        rank_all = rank_all.sample(n=n_rank, random_state=seed)
    rank_ids = rank_all.MVT_ID_mvt.to_numpy(dtype="float64")
    rank_all = rank_all.drop(columns=["MVT_ID_mvt"])
    cols_u = cols + [IS_UNMATCHED]
    parts = [f.assign(**{IS_UNMATCHED: 0.0})[cols_u] for f in frames]
    parts += [uf.reindex(columns=cols).assign(**{IS_UNMATCHED: 1.0})[cols_u] for uf in u_frames]
    parts.append(rank_all[cols_u])
    d = pd.concat(parts, ignore_index=True)
    del parts, frames, u_frames
    gc.collect()
    is_rank = np.zeros(len(d), dtype=bool)
    is_rank[len(d) - len(rank_all):] = True
    log(f"loaded {len(paths)} training months ({n_m_train:,} matched + {n_u_train:,} unmatched rows) + ranking "
        f"({is_rank.sum():,} rows, {int(d[IS_UNMATCHED].to_numpy()[is_rank].sum()):,} unmatched); the NaN pattern "
        f"checked on {len(paths) + 1} unmatched cache files; peak RSS {peak_rss_gb():.2f} GB")
    return d, is_rank, rank_ids


def design_matrix(d: pd.DataFrame, feats) -> np.ndarray:
    X = np.empty((len(d), len(feats)), dtype=np.float32)
    for i, c in enumerate(feats):
        X[:, i] = pd.to_numeric(d[c], errors="coerce").to_numpy(np.float32)
    return X


def _progress(period: int, what: str = "delta"):
    def cb(env):
        if env.iteration % period == 0 and env.evaluation_result_list:
            name, metric, value, _ = env.evaluation_result_list[0]
            log(f"    iter {env.iteration:>6,}  {name} {metric} ({what}) {value:.3f}")
    cb.order = 30
    return cb


def _rmse(a, b) -> float:
    return float(np.sqrt(((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64")) ** 2).mean()))


def predict_delta(booster, X, params, num_iteration=None):
    """Every prediction goes through here so it is threaded like training.

    lightgbm's Booster.predict takes its thread count from the OpenMP runtime, not from the
    training params; under this repo's OMP_NUM_THREADS=1 cap a 23k-tree predict on 345k rows
    ran single-threaded for 40+ minutes on 2026-09-09. Forwarding num_threads fixes it at the
    only place a predict is issued."""
    return booster.predict(X, num_iteration=num_iteration, num_threads=params["num_threads"])


def early_stop(X, dlt, y, proxy, train, fit, es, params, nest, patience, target: str = TARGET_DELTA,
               weight=None, es_eval=None) -> dict:
    """lgbm_ab's early-stopping run on `es` -> best_iter, and n_ref for the refit on `train`.

    `dlt` is always the true delta (the proxy-only reference of the breakage guard reads it);
    the label LightGBM sees is regression_label(dlt, y, target): delta by default, y under
    Amendment 19.2's `--target y` - the stopping metric (rmse on the label) is then rmse on y
    directly, and the stopping-set taxi time is max(prediction, 1) with no proxy anchor.
    `weight` (the unified arm) is a per-row sample weight for the FIT rows; the stopping set is
    never weighted - its metric is the competition's plain RMSE. On an all-rows design the
    proxy-only breakage guard reads the stopping rows that HAVE a proxy (the matched ones).
    `es_eval` (Amendment 20.5, the all-rows arm) restricts the rows the stopping METRIC is
    evaluated on to a subset of `es` - the matched stopping rows - because the unmatched
    stopping rows' squared errors (monsters, bet variance) are of the order of the matched
    rows' whole MSE and decide the stop by noise (best_iter 1,354 against the delta arm's
    24,517 on the real months); the fit rows and their weights are untouched. None evaluates
    on every stopping row, which is every matched-only design, byte for byte as before.
    Returns the fit info (best_iter, n_ref, row counts - n_es evaluated, n_es_months all stopping
    rows - the target, the stopping-set guard numbers); the booster itself is discarded - the
    refit is what ships.
    """
    label = regression_label(dlt, y, target)
    w = None if weight is None else np.asarray(weight, dtype="float64")
    if w is not None and w.shape != label.shape:
        raise ValueError(f"weight shape {w.shape} != label shape {label.shape}")
    restricted = es_eval is not None
    if restricted:
        es_eval = np.asarray(es_eval, dtype=bool)
        if es_eval.shape != es.shape:
            raise ValueError(f"es_eval shape {es_eval.shape} != stopping mask shape {es.shape}")
        if (es_eval & ~es).any():
            raise ValueError(f"es_eval must be a subset of the stopping rows: {int((es_eval & ~es).sum()):,} rows outside them")
        if not es_eval.any():
            raise ValueError("es_eval selects no stopping rows")
    else:
        es_eval = es
    log(f"early-stopping run (seed {params['seed']}, target {target}): fit {fit.sum():,} rows, stop on "
        f"{es_eval.sum():,} rows (months {ES_MONTHS}"
        + (f"; the MATCHED stopping rows only, {es.sum() - es_eval.sum():,} unmatched stopping rows excluded "
           f"from the metric, Amendment 20.5" if restricted else "")
        + f"), lr {params['learning_rate']}, max {nest:,} rounds, patience {patience}"
        + ("" if w is None else f", row weights (unmatched x{w.max():g})"))
    ds = lgb.Dataset(X[fit], label[fit], weight=None if w is None else w[fit])
    b = lgb.train(params, ds, num_boost_round=nest, valid_sets=[lgb.Dataset(X[es_eval], label[es_eval])],
                  callbacks=[lgb.early_stopping(patience, verbose=False),
                             lgb.log_evaluation(0), _progress(PROGRESS_EVERY, target)])
    best_iter = int(b.best_iteration)
    assert best_iter >= 1, f"best_iteration {best_iter}"
    pred_es = np.maximum(raw_taxi_time(predict_delta(b, X[es_eval], params, num_iteration=best_iter), proxy[es_eval], target), 1.0)
    es_rmse = _rmse(y[es_eval], pred_es)
    fin_es, fin_fit = np.isfinite(proxy[es_eval]), np.isfinite(dlt[fit])
    if fin_es.all() and fin_fit.all():
        proxy_only = _rmse(y[es_eval], np.maximum(proxy[es_eval] - dlt[fit].mean(), 1.0))
        es_guard, n_guard = es_rmse, int(es_eval.sum())
    else:                                   # the unified arm: the rows with a proxy are the matched ones
        proxy_only = _rmse(y[es_eval][fin_es], np.maximum(proxy[es_eval][fin_es] - dlt[fit][fin_fit].mean(), 1.0))
        es_guard, n_guard = _rmse(y[es_eval][fin_es], pred_es[fin_es]), int(fin_es.sum())
    log(f"best_iter {best_iter:,}   stopping-set RMSE (taxi-time) {es_rmse:.2f}   "
        f"proxy-only on the {n_guard:,} rows with a proxy {proxy_only:.2f}   ratio {es_guard / proxy_only:.3f}   "
        f"[stopping set, optimistic by construction; NOT a fold result]   "
        f"peak RSS {peak_rss_gb():.2f} GB")
    if es_guard > ES_MAX_RATIO_TO_PROXY_ONLY * proxy_only:
        raise RuntimeError(f"stopping-set RMSE {es_guard:.2f} is not {ES_MAX_RATIO_TO_PROXY_ONLY:.0%} "
                           f"of proxy-only {proxy_only:.2f}: the feature path is broken, "
                           "refusing to refit")
    del ds, b
    gc.collect()
    n_ref = n_refit(best_iter, int(train.sum()), int(fit.sum()))
    log(f"n_ref {n_ref:,}: refit on all {train.sum():,} training rows "
        f"(best_iter {best_iter:,} x {train.sum() / fit.sum():.3f}), no early stopping")
    return dict(best_iter=best_iter, n_ref=n_ref, n_fit=int(fit.sum()), n_es=int(es_eval.sum()),
                n_es_months=int(es.sum()), n_all=int(train.sum()), es_months=list(ES_MONTHS), target=target,
                es_rmse_taxi_time_NOT_A_RESULT=es_rmse, es_proxy_only_rmse=proxy_only, es_guard_rows=n_guard)


def refit(X, dlt, mask, params, n_trees, weight=None):
    """A fixed-count fit on the rows in `mask`, no early stopping: the pooled refit at n_ref,
    and the per-airport models at their share count, are both this call. `weight` (the unified
    arm) is the per-row sample weight, masked like the rows."""
    ds = lgb.Dataset(X[mask], dlt[mask], weight=None if weight is None else np.asarray(weight, dtype="float64")[mask])
    b = lgb.train(params, ds, num_boost_round=n_trees, callbacks=[lgb.log_evaluation(0)])
    del ds
    gc.collect()
    return b


def fit_seeds(X, label, train, X_pred, params, n_ref, seeds, files=None, info=None, smoke=None,
              n_features=None, target: str = TARGET_DELTA, weight=None, provenance=None):
    """Refit on `train` at n_ref once per seed and predict X_pred with each -> {seed: prediction}.

    `label` is the regressor's label (regression_label: delta by default, y under --target y;
    the prediction is on the same scale). When `files` is given each booster is saved with its
    own fit.json (the ES info plus this seed's params, the feature count and the target) before
    being freed; only one booster is alive at a time. `n_features` is the width the caller
    built (default len(FEATS); FEATS + QUEUE_FEATS under --queue, + DAY_FEATS under --dayfeats,
    + ORDER_FEATS under --orderfeats, + is_unmatched under --all-rows) and the booster must agree with it. `weight` is the
    per-row sample weight (the unified arm); `provenance` extra fit.json fields (all_rows,
    unmatched_weight) checked on reuse.
    """
    n_features = len(FEATS) if n_features is None else int(n_features)
    out = {}
    for i, s in enumerate(seeds):
        t = time.time()
        b = refit(X, label, train, dict(params, seed=s), n_ref, weight=weight)
        wall = time.time() - t
        if b.num_feature() != n_features:
            raise ValueError(f"booster has {b.num_feature()} features, expected {n_features}")
        log(f"refit seed {s} done: {b.num_trees():,} trees on {int(np.sum(train)):,} rows in "
            f"{wall:.0f}s   peak RSS {peak_rss_gb():.2f} GB")
        if files is not None:
            path, fjson = files[i]
            b.save_model(str(path))
            fjson.write_text(json.dumps({**info, "fit_wall_s": round(wall, 1), "smoke": smoke,
                                         "params": dict(params, seed=s), "es_seed": ES_SEED,
                                         "n_features": n_features, "target": target, **(provenance or {})},
                                        indent=2))
            log(f"saved booster -> {path}")
        out[s] = predict_delta(b, X_pred, params)
        del b
        gc.collect()
    return out


def reusable(files) -> bool:
    """True when every seed's booster AND fit.json are on disk, False when none are. A partial
    set is refused: a mean over reused and freshly fitted boosters has no single provenance."""
    present = [p.exists() and j.exists() for p, j in files]
    if all(present):
        return True
    if any(present):
        have = [p.name for (p, _), ok in zip(files, present) if ok]
        missing = [p.name for (p, _), ok in zip(files, present) if not ok]
        raise ValueError(f"--reuse-booster: partial seed set on disk (present: {have}; missing: "
                         f"{missing}); refusing to mix reused and fresh boosters")
    return False


def reuse_seeds(files, seeds, X_pred, params, smoke, n_all, n_features=None, target: str = TARGET_DELTA,
                provenance=None):
    """Load each seed's saved booster, check its provenance against ITS seed's params (and the
    run's feature count and target), predict.

    Returns ({seed: prediction}, the shared fit info from the first fit.json). Every fit.json
    must agree on best_iter and n_ref - they came from one early-stopping run.
    """
    n_features = len(FEATS) if n_features is None else int(n_features)
    preds, info = {}, None
    for s, (path, fjson) in zip(seeds, files):
        b = lgb.Booster(model_file=str(path))
        i = json.loads(fjson.read_text())
        check_provenance(i, b.num_trees(), smoke, n_all, dict(params, seed=s), n_features=n_features, target=target,
                         extra=provenance)
        if b.num_feature() != n_features:
            raise ValueError(f"{path.name} has {b.num_feature()} features, expected {n_features}")
        log(f"reused booster {path} ({b.num_trees():,} trees), best_iter {i['best_iter']:,}, "
            f"n_ref {i['n_ref']:,}")
        shared = {k: v for k, v in i.items() if k not in ("params", "smoke")}
        if info is None:
            info = shared
        elif (shared.get("best_iter"), shared.get("n_ref")) != (info.get("best_iter"), info.get("n_ref")):
            raise ValueError(f"{fjson.name} disagrees with the first seed on best_iter/n_ref")
        preds[s] = predict_delta(b, X_pred, params)
        del b
        gc.collect()
    return preds, info


def mean_delta(preds) -> np.ndarray:
    """delta_hat over seeds: the arithmetic mean of the per-seed predictions, in float64,
    summed left to right so the value is the one the tests compute by hand."""
    preds = [np.asarray(p, dtype="float64") for p in preds]
    if not preds:
        raise ValueError("no predictions to average")
    if any(p.shape != preds[0].shape for p in preds):
        raise ValueError(f"prediction shape mismatch: {[p.shape for p in preds]}")
    acc = preds[0].copy()
    for p in preds[1:]:
        acc += p
    return acc / len(preds)


def per_airport_trees(n_ref, n_airport, n_all, floor=PA_TREE_FLOOR) -> int:
    """Tree count for one airport's own model: max(floor, int(n_ref * n_airport / n_all)).

    JUDGEMENT CALL, not a measurement (recorded per Amendment 12.4). The pooled refit's
    n_ref is the early-stopped count scaled to the training rows; an airport gets its row
    share of those trees, on the reading that the optimal count at a fixed learning rate
    grows with the rows. build_submission's HGB uses 400 per-airport vs 900 pooled (0.44),
    while the row share here is 0.06-0.12, so this rule is the more conservative of the two
    and may under-train the per-airport models. The measured alternative - an early-stopping
    run per airport on the same ES months - costs about one more pooled ES run (~25 min on
    the full data) and is the first thing to try if A3 disappoints.
    """
    if n_all <= 0 or n_airport < 0 or n_ref < 1:
        raise ValueError(f"per_airport_trees(n_ref={n_ref}, n_airport={n_airport}, n_all={n_all})")
    return max(int(floor), int(n_ref * n_airport / n_all))


def airport_codes(series: pd.Series):
    """(int8 code per row, sorted airport names): the airport column as an integer array so the
    string column can be dropped before the design matrix is allocated."""
    if series.isna().any():
        raise ValueError("airport column has missing values")
    airports = sorted(str(a) for a in series.unique())
    if len(airports) > 120:
        raise ValueError(f"{len(airports)} airports do not fit an int8 code")
    codes = series.map({a: i for i, a in enumerate(airports)}).to_numpy(np.int8)
    return codes, airports


def fit_per_airport(X, label, train, pred, ap_code, airports, params, n_ref, seeds,
                    min_rows=PA_MIN_ROWS, floor=PA_TREE_FLOOR, trees="share", es=None, target: str = TARGET_DELTA):
    """One LightGBM per airport with >= min_rows training rows, refit per seed on `label` (the
    regressor's label: delta by default, y under --target y), predicting that airport's `pred`
    rows. Airports under the floor (or with no prediction rows) are left NaN with fitted=False -
    blend() keeps the pooled value there.

    Tree count per airport: trees="share" -> per_airport_trees(n_ref, n_airport, n_all, floor);
    trees="es" -> the airport's own early-stopping run (early_stop on the pooled fit/es masks
    intersected with the airport, seed ES_SEED, the same target) and its own n_ref; `es` then
    supplies dict(y=, proxy=, fit=, es=, nest=, patience=) and, under a non-delta target, dlt=
    (the true delta, which early_stop's proxy-only guard reads; under delta the label is it).

    Returns ({seed: per-airport prediction over the pred rows, NaN where not fitted},
             fitted mask over the pred rows, {airport: info incl. the rule it was sized by}).
    """
    if trees not in ("share", "es"):
        raise ValueError(f"trees must be 'share' or 'es', got {trees!r}")
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    if trees == "es":
        need = ("y", "proxy", "fit", "es", "nest", "patience")
        if es is None or any(k not in es for k in need):
            raise ValueError(f"trees='es' needs es={{{', '.join(k + '=' for k in need)}}}")
        if target != TARGET_DELTA and "dlt" not in es:
            raise ValueError(f"trees='es' under target {target!r} needs es['dlt'] (the true delta) as well")
    train, pred = np.asarray(train, dtype=bool), np.asarray(pred, dtype=bool)
    ap_code = np.asarray(ap_code)
    if (train & pred).any():
        raise ValueError("training and prediction rows overlap")
    n_all = int(train.sum())
    pred_idx = np.flatnonzero(pred)
    ap_pred = ap_code[pred_idx]
    pa_by_seed = {s: np.full(len(pred_idx), np.nan) for s in seeds}
    fitted = np.zeros(len(pred_idx), dtype=bool)
    info = {}
    for code, name in enumerate(airports):
        tr_mask = train & (ap_code == code)
        n_tr = int(tr_mask.sum())
        rows = ap_pred == code
        n_pr = int(rows.sum())
        if n_tr < min_rows or n_pr == 0:
            why = f"{n_tr:,} training rows < {min_rows:,}" if n_tr < min_rows else "no prediction rows"
            log(f"  per-airport {name}: {why} -> keeps pooled (fallback)")
            info[name] = dict(n_train=n_tr, n_pred=n_pr, fitted=False, n_trees=None, seeds=[],
                              rule=trees, es=None)
            continue
        es_info = None
        if trees == "share":
            n_trees = per_airport_trees(n_ref, n_tr, n_all, floor)
        else:
            fit_a, es_a = es["fit"] & (ap_code == code), es["es"] & (ap_code == code)
            if not fit_a.any() or not es_a.any():
                raise ValueError(f"{name}: no fit or no early-stopping rows for a per-airport ES run")
            log(f"  per-airport {name}: early stopping on its own rows")
            i = early_stop(X, es["dlt"] if "dlt" in es else label, es["y"], es["proxy"], tr_mask, fit_a, es_a,
                           dict(params, seed=ES_SEED), es["nest"], es["patience"], target=target)
            n_trees = int(i["n_ref"])
            es_info = dict(best_iter=int(i["best_iter"]), n_ref=n_trees, n_fit=int(i["n_fit"]),
                           n_es=int(i["n_es"]))
        Xp = X[pred_idx[rows]]
        t = time.time()
        for s in seeds:
            b = refit(X, label, tr_mask, dict(params, seed=s), n_trees)
            pa_by_seed[s][rows] = predict_delta(b, Xp, params)
            del b
            gc.collect()
        fitted[rows] = True
        info[name] = dict(n_train=n_tr, n_pred=n_pr, fitted=True, n_trees=n_trees, seeds=list(seeds),
                          rule=trees, es=es_info)
        log(f"  per-airport {name}: {n_tr:,} training rows, {n_trees:,} trees ({trees}) x "
            f"{len(seeds)} seed(s), {n_pr:,} prediction rows, {time.time() - t:.0f}s")
    return pa_by_seed, fitted, info


def blend(pooled, pa, fitted, weight=BLEND) -> np.ndarray:
    """(1 - weight) * pooled + weight * per-airport on fitted rows; pooled on the rest."""
    pooled = np.asarray(pooled, dtype="float64")
    pa = np.asarray(pa, dtype="float64")
    fitted = np.asarray(fitted, dtype=bool)
    if not (pooled.shape == pa.shape == fitted.shape):
        raise ValueError(f"shape mismatch: pooled {pooled.shape}, per-airport {pa.shape}, "
                         f"fitted {fitted.shape}")
    if not np.isfinite(pooled).all():
        raise ValueError("non-finite pooled prediction")
    if not np.isfinite(pa[fitted]).all():
        raise ValueError("non-finite per-airport prediction on a fitted row")
    out = pooled.copy()
    out[fitted] = (1.0 - weight) * pooled[fitted] + weight * pa[fitted]
    return out


def check_provenance(info: dict, n_trees: int, smoke: bool, n_all: int, params: dict,
                     n_features=None, target=None, extra=None) -> None:
    """A saved booster may be reused only by the configuration that produced it.

    A smoke booster at the real path, a booster fitted on a different row set, different
    parameters, a different feature count (a fit.json that records one), another target, or a
    truncated model file would otherwise be spliced into a submission silently. Refuse on any
    mismatch. A legacy fit.json without n_features falls through to the booster's own
    num_feature check; one without `target` is a delta booster (every booster before Amendment
    19), reusable by a delta run only.
    """
    problems = []
    if n_features is not None and "n_features" in info and info["n_features"] != n_features:
        problems.append(f"n_features={info['n_features']} (run has {n_features})")
    if target is not None and info.get("target", TARGET_DELTA) != target:
        problems.append(f"target={info.get('target', TARGET_DELTA)!r} (run has {target!r})")
    for k, v in (extra or {}).items():
        have = info.get(k, LEGACY_PROVENANCE.get(k))
        if have != v:
            problems.append(f"{k}={have!r} (run has {v!r})")
    if info.get("smoke") != smoke:
        problems.append(f"smoke={info.get('smoke')} (run is smoke={smoke})")
    if info.get("n_all") != n_all:
        problems.append(f"n_all={info.get('n_all')} (run has {n_all})")
    if info.get("params") != params:
        problems.append("params differ")
    if info.get("n_ref") != n_trees:
        problems.append(f"n_ref={info.get('n_ref')} but the model file holds {n_trees} trees")
    if problems:
        raise ValueError("saved booster was not produced by this configuration: "
                         + "; ".join(problems) + "; refusing to reuse it")


# ---- v5 (Amendment 16): the schedule-fill mixture head --------------------------------------

def head_features(feats) -> list:
    """The head's design: `feats` + [NMDELAY], appended once, LAST; refuses a list that already
    carries it (the booster's feature count would otherwise disagree with the design silently)."""
    feats = list(feats)
    if NMDELAY in feats:
        raise ValueError(f"{NMDELAY!r} is already among the features: it is appended exactly once")
    return feats + [NMDELAY]


def add_nmdelay(d: pd.DataFrame) -> pd.DataFrame:
    """d[NMDELAY] = proxy - sp in float64, in place (returns d); refuses a frame that already has
    the column or lacks proxy / sp."""
    if NMDELAY in d.columns:
        raise ValueError(f"{NMDELAY!r} is already a column: it is appended exactly once")
    for c in ("proxy", "sp"):
        if c not in d.columns:
            raise ValueError(f"cannot form {NMDELAY} = proxy - sp: column {c!r} is missing")
    d[NMDELAY] = d.proxy.to_numpy(dtype="float64") - d.sp.to_numpy(dtype="float64")
    return d


def fill_label(y, sp, train) -> np.ndarray:
    """The head's target: 1.0 where |y - sp| <= FILL_TOL_S (inclusive), 0.0 otherwise, on the
    TRAINING rows; NaN on every other row - y is never read there (it is NaN on ranking rows
    and must stay unseen on holdout rows). Refuses a non-finite y or sp on a training row
    (the caches have none, and a NaN label would train silently) and a single class."""
    y, sp = np.asarray(y, dtype="float64"), np.asarray(sp, dtype="float64")
    train = np.asarray(train, dtype=bool)
    if not (y.shape == sp.shape == train.shape):
        raise ValueError(f"shape mismatch: y {y.shape}, sp {sp.shape}, train {train.shape}")
    if not train.any():
        raise ValueError("no training rows for the fill label")
    yt, st = y[train], sp[train]
    bad = ~(np.isfinite(yt) & np.isfinite(st))
    if bad.any():
        raise ValueError(f"non-finite y or sp on {int(bad.sum()):,} training rows: the fill label is undefined there")
    label = np.full(y.shape, np.nan)
    label[train] = (np.abs(yt - st) <= FILL_TOL_S).astype("float64")
    n_fill, n_train = int(label[train].sum()), int(train.sum())
    if n_fill == 0 or n_fill == n_train:
        raise ValueError(f"the training rows hold a single class ({n_fill:,} fills of {n_train:,}): nothing to learn")
    return label


def rank_auc(score, label) -> float:
    """AUC as the Mann-Whitney statistic with average ranks for ties (a constant score is 0.5):
    the head's diagnostic on its stopping set and on the fold's holdout."""
    score, label = np.asarray(score, dtype="float64"), np.asarray(label, dtype="float64")
    if score.shape != label.shape:
        raise ValueError(f"shape mismatch: {score.shape} scores for {label.shape} labels")
    pos = label > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _check_probabilities(p: np.ndarray) -> None:
    if not np.isfinite(p).all():
        raise ValueError(f"fill head produced {int((~np.isfinite(p)).sum()):,} non-finite probabilities")
    if (p < 0.0).any() or (p > 1.0).any():
        raise ValueError(f"fill head produced probabilities outside [0, 1]: min {p.min():.4g}, max {p.max():.4g}")


def fit_fill_head(X, label, train, fit, es, X_pred, params, nest, patience, files=None, smoke=None,
                  n_features=None):
    """The Amendment 16 head, p = P(fill | x), fitted like the regressor:

      1. an early-stopping run (params: objective binary, metric auc) on the `fit` rows, stopped
         on the `es` rows with `patience` -> best_iter and the stopping-set AUC (NOT a result);
      2. n_ref = n_refit(best_iter, n_all, n_fit), the regressor's rule;
      3. ONE refit on all `train` rows at n_ref with params["seed"] (the amendment's seed 0; the
         head is not seed-averaged);
      4. p = predict_delta(booster, X_pred) - the one threaded predict site, here a probability.

    `label` is fill_label's array (0/1 on the training rows, NaN elsewhere). A head whose
    stopping-set AUC is not above FILL_HEAD_MIN_ES_AUC learned nothing and is refused before
    the refit. With `files` = (model path, fit.json) the booster is saved with its provenance
    (the ES info, the head's params, n_features, the fill shares) before being freed.
    Returns (p, info).
    """
    n_features = len(head_features(FEATS)) if n_features is None else int(n_features)
    X_pred = np.asarray(X_pred)
    label = np.asarray(label, dtype="float64")
    train, fit, es = (np.asarray(m, dtype=bool) for m in (train, fit, es))
    if X.shape[1] != n_features or X_pred.shape[1] != n_features:
        raise ValueError(f"the head's design has {X.shape[1]} columns and the prediction rows {X_pred.shape[1]}; "
                         f"expected {n_features} features")
    if (fit & es).any() or not np.array_equal(fit | es, train):
        raise ValueError("the fit and early-stopping rows must partition the training rows")
    if not np.isfinite(label[train]).all():
        raise ValueError(f"non-finite fill label on {int((~np.isfinite(label[train])).sum()):,} training rows")
    for name, m in (("fit", fit), ("early-stopping", es)):
        if label[m].min() == label[m].max():
            raise ValueError(f"the {name} rows hold a single class: the head has nothing to learn")
    shares = dict(fill_share_train=float(label[train].mean()), fill_share_fit=float(label[fit].mean()),
                  fill_share_es=float(label[es].mean()))
    log(f"fill head: early-stopping run (seed {params['seed']}, objective {params['objective']}, metric "
        f"{params['metric']}): fit {fit.sum():,} rows ({100 * shares['fill_share_fit']:.2f}% fills), stop on "
        f"{es.sum():,} rows ({100 * shares['fill_share_es']:.2f}% fills, months {ES_MONTHS}), lr "
        f"{params['learning_rate']}, max {nest:,} rounds, patience {patience}, {n_features} features")
    t = time.time()
    ds = lgb.Dataset(X[fit], label[fit])
    b = lgb.train(params, ds, num_boost_round=nest, valid_sets=[lgb.Dataset(X[es], label[es])],
                  callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0),
                             _progress(PROGRESS_EVERY, "P(fill)")])
    best_iter = int(b.best_iteration)
    assert best_iter >= 1, f"best_iteration {best_iter}"
    p_es = np.asarray(predict_delta(b, X[es], params, num_iteration=best_iter), dtype="float64")
    es_auc = rank_auc(p_es, label[es])
    es_auc_lgb = float(b.best_score["valid_0"][params["metric"]])
    es_wall = time.time() - t
    log(f"fill head: best_iter {best_iter:,}   stopping-set AUC {es_auc:.4f} (lightgbm {es_auc_lgb:.4f})   "
        f"mean p {p_es.mean():.4f} vs fill rate {shares['fill_share_es']:.4f}   [stopping set, optimistic by "
        f"construction; NOT a fold result]   {es_wall:.0f}s   peak RSS {peak_rss_gb():.2f} GB")
    del ds, b
    gc.collect()
    if es_auc <= FILL_HEAD_MIN_ES_AUC:
        raise RuntimeError(f"fill head: stopping-set AUC {es_auc:.4f} is not above {FILL_HEAD_MIN_ES_AUC}: the "
                           "head learned nothing, refusing to refit")
    n_ref = n_refit(best_iter, int(train.sum()), int(fit.sum()))
    log(f"fill head: n_ref {n_ref:,}: one refit (seed {params['seed']}) on all {train.sum():,} training rows "
        f"({100 * shares['fill_share_train']:.2f}% fills; best_iter {best_iter:,} x {train.sum() / fit.sum():.3f}), "
        "no early stopping")
    t = time.time()
    b = refit(X, label, train, params, n_ref)
    wall = time.time() - t
    if b.num_feature() != n_features:
        raise ValueError(f"head booster has {b.num_feature()} features, expected {n_features}")
    info = dict(best_iter=best_iter, n_ref=n_ref, n_fit=int(fit.sum()), n_es=int(es.sum()), n_all=int(train.sum()),
                es_months=list(ES_MONTHS), es_auc_NOT_A_RESULT=es_auc, es_auc_lightgbm_NOT_A_RESULT=es_auc_lgb,
                es_mean_p=float(p_es.mean()), es_fill_rate=shares["fill_share_es"], **shares,
                n_features=n_features, objective=params["objective"], metric=params["metric"],
                es_wall_s=round(es_wall, 1), refit_wall_s=round(wall, 1))
    log(f"fill head: refit done: {b.num_trees():,} trees on {int(train.sum()):,} rows in {wall:.0f}s   "
        f"peak RSS {peak_rss_gb():.2f} GB")
    if files is not None:
        path, fjson = files
        b.save_model(str(path))
        fjson.write_text(json.dumps({**info, "smoke": smoke, "params": dict(params), "es_seed": params["seed"]},
                                    indent=2))
        log(f"saved fill head booster -> {path}")
    p = np.asarray(predict_delta(b, X_pred, params), dtype="float64")
    del b
    gc.collect()
    _check_probabilities(p)
    return p, info


def reuse_fill_head(files, X_pred, params, smoke, n_all, n_features=None):
    """Load the saved head booster, check its fit.json against THIS run's head params, feature
    count, training row count and smoke flag (check_provenance, as for the regressors), and
    predict p. Returns (p, the fit info without params / smoke)."""
    n_features = len(head_features(FEATS)) if n_features is None else int(n_features)
    path, fjson = files
    b = lgb.Booster(model_file=str(path))
    i = json.loads(fjson.read_text())
    check_provenance(i, b.num_trees(), smoke, n_all, dict(params), n_features=n_features)
    if b.num_feature() != n_features:
        raise ValueError(f"{path.name} has {b.num_feature()} features, expected {n_features}")
    log(f"reused fill head booster {path} ({b.num_trees():,} trees), best_iter {i['best_iter']:,}, n_ref "
        f"{i['n_ref']:,}, training fill share {i.get('fill_share_train', float('nan')):.4f}")
    p = np.asarray(predict_delta(b, X_pred, params), dtype="float64")
    del b
    gc.collect()
    _check_probabilities(p)
    return p, {k: v for k, v in i.items() if k not in ("params", "smoke")}


def mix_fill(p, sp, base_yhat) -> np.ndarray:
    """The Amendment 16 mixture, y_hat = p * sp + (1 - p) * base_yhat in float64, row by row:
    p = P(fill | x) from the head, sp = MVT_TIME - SCHED_TIME (the taxi time if the block time
    is the schedule), base_yhat = the regressor arm's own prediction max(proxy - delta_hat, 1).
    The result is a TAXI TIME, not a delta. p outside [0, 1], non-finite inputs and a shape
    mismatch are refusals."""
    p, sp, base = (np.asarray(v, dtype="float64") for v in (p, sp, base_yhat))
    if not (p.shape == sp.shape == base.shape):
        raise ValueError(f"shape mismatch: p {p.shape}, sp {sp.shape}, base_yhat {base.shape}")
    if not np.isfinite(p).all():
        raise ValueError(f"non-finite p on {int((~np.isfinite(p)).sum()):,} rows")
    if (p < 0.0).any() or (p > 1.0).any():
        raise ValueError(f"p must be in [0, 1]: min {p.min():.4g}, max {p.max():.4g}")
    if not np.isfinite(sp).all():
        raise ValueError(f"non-finite sp on {int((~np.isfinite(sp)).sum()):,} rows")
    if not np.isfinite(base).all():
        raise ValueError(f"non-finite baseline prediction on {int((~np.isfinite(base)).sum()):,} rows")
    return p * sp + (1.0 - p) * base


def _git() -> dict:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"git_sha": run("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


def main(argv=None) -> int:
    args = parse_args(argv)
    started = dt.datetime.now(dt.timezone.utc)
    smoke = args.smoke
    seeds = tuple(args.seeds)
    params, nest, patience, months, n_rank = dict(P), NEST, PATIENCE, None, None
    pa_floor = PA_TREE_FLOOR
    if smoke:
        params["learning_rate"] = SMOKE["learning_rate"]
        nest, patience, months, n_rank = SMOKE["nest"], SMOKE["patience"], SMOKE["months"], SMOKE["n_rank"]
        pa_floor = SMOKE["pa_floor"]
        log(SMOKE_BANNER)
    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
    else:
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="lgbm_submit_smoke_")) if smoke else SUBS
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = pathlib.Path(args.base)
    if not B.SUBMISSION_RE.match(base_path.name):
        raise SystemExit(f"--base {base_path.name!r} is not a submission file")
    dest = out_dir / output_name(args.version)
    if dest.resolve() == base_path.resolve():
        raise SystemExit("refusing to overwrite the base submission")
    feats = (list(FEATS) + (QUEUE_FEATS if args.queue else []) + (DAY_FEATS if args.dayfeats else [])
             + (ORDER_FEATS if args.orderfeats else []) + (WEATHER_FEATS if args.weatherfeats else [])
             + ([IS_UNMATCHED] if args.all_rows else []))
    feats_all = head_features(feats) if args.fillhead else feats   # the matrix; the regressors read [:nb]
    nb = len(feats)
    target = args.target
    provenance = {"all_rows": bool(args.all_rows), "unmatched_weight": float(args.unmatched_weight) if args.all_rows else 1.0}
    booster_dir = out_dir if smoke else CACHE
    files = booster_files(booster_dir, args.version, seeds, queue=args.queue, day=args.dayfeats, target=target,
                          all_rows=args.all_rows, unmatched_weight=args.unmatched_weight, order=args.orderfeats,
                          weather=args.weatherfeats)
    head_files = (head_booster_files(booster_dir, args.version, queue=args.queue, day=args.dayfeats,
                                     order=args.orderfeats, weather=args.weatherfeats)
                  if args.fillhead else None)
    log(f"version {args.version}  smoke={smoke}  seeds={list(seeds)}  per-airport={args.per_airport}"
        f"  queue={args.queue} ({len(feats)} features)  dayfeats={args.dayfeats}  orderfeats={args.orderfeats}  weatherfeats={args.weatherfeats}  target={target}"
        f"  all-rows={args.all_rows}" + (f" (unmatched weight {args.unmatched_weight:g})" if args.all_rows else "")
        + f"  fillhead={args.fillhead}"
        + (f" ({len(feats_all)} columns, the regressors read the first {nb}; head booster {head_files[0].name})"
           if args.fillhead else "")
        + f"  out={dest}  boosters={[p.name for p, _ in files]} in {files[0][0].parent}")

    # ---- 1. rows ----
    d, is_rank, rank_ids = load_frames(CACHE, months=months, n_rank=n_rank, queue=args.queue, day=args.dayfeats,
                                       all_rows=args.all_rows, order=args.orderfeats, weather=args.weatherfeats)
    train, fit, es = split_masks(d.month.to_numpy(), is_rank)
    y, dlt, proxy, sp = d.y.to_numpy(), d.delta.to_numpy(), d.proxy.to_numpy(), d.sp.to_numpy()
    label = regression_label(dlt, y, target)         # delta (the default) or y (Amendment 19.2)
    is_um = (d[IS_UNMATCHED].to_numpy() == 1.0) if args.all_rows else np.zeros(len(d), dtype=bool)
    weight = row_weights(is_um, args.unmatched_weight) if args.all_rows else None
    ap_code, airports = airport_codes(d.ap)          # before the string columns go

    # ---- 2. encodings: fitted on every training row, applied to the ranking rows; the delta
    # ---- encodings on the matched training rows only under --all-rows (delta is NaN elsewhere) ----
    for col, vals in S.infold_encodings(d, y, dlt, train, delta_mask=(~is_um if args.all_rows else None)).items():
        d[col] = vals
    nan_enc = [c for c in S.ENC if np.isnan(d[c].to_numpy()[is_rank]).any()]
    assert not nan_enc, f"NaN encodings on ranking rows: {nan_enc}"
    d = d.drop(columns=[c for c in set(S.ENC_KEYS) | {"ars", "ap"} if c in d.columns])
    gc.collect()

    # ---- 3. design matrix: one float32 array; the regressors see a view of its first nb columns ----
    if args.fillhead:
        add_nmdelay(d)
    X = design_matrix(d, feats_all)
    del d
    gc.collect()
    log(f"design matrix {X.shape[0]:,} x {X.shape[1]} float32 ({X.nbytes / 1e9:.2f} GB); "
        f"peak RSS {peak_rss_gb():.2f} GB")
    X_rank = X[is_rank]
    Xb, Xb_rank = (X[:, :nb], X_rank[:, :nb]) if args.fillhead else (X, X_rank)

    # ---- 4. fit once per seed, or reuse every seed's saved booster ----
    if args.reuse_booster and reusable(files):
        preds, info = reuse_seeds(files, seeds, Xb_rank, params, smoke, int(train.sum()), n_features=len(feats),
                                  target=target, provenance=provenance)
    else:
        t_fit = time.time()
        info = early_stop(Xb, dlt, y, proxy, train, fit, es, dict(params, seed=ES_SEED), nest, patience, target=target,
                          weight=weight, es_eval=(es & ~is_um) if args.all_rows else None)   # Amendment 20.5
        preds = fit_seeds(Xb, label, train, Xb_rank, params, info["n_ref"], seeds,
                          files=files, info=info, smoke=smoke, n_features=len(feats), target=target, weight=weight,
                          provenance=provenance)
        info["fit_wall_s"] = round(time.time() - t_fit, 1)
        log(f"fit wall {info['fit_wall_s']:.0f}s for the early-stopping run + {len(seeds)} refit(s)")
    pooled = mean_delta([preds[s] for s in seeds])
    if len(seeds) > 1:
        spread = np.stack([preds[s] for s in seeds]).std(axis=0, ddof=1)
        log(f"seed mean over {len(seeds)} refits: mean per-row seed sd of delta_hat "
            f"{spread.mean():.2f}s (p99 {np.percentile(spread, 99):.1f}s)")

    # ---- 5. per-airport models, blended into the pooled prediction ----
    pa_info = None
    pa_rule = PA_TREE_RULE if args.pa_trees == "share" else PA_TREE_RULE_ES
    if args.per_airport:
        log(f"per-airport models: floor {PA_MIN_ROWS:,} training rows, blend {BLEND}, "
            f"tree rule [{args.pa_trees}] {pa_rule} (share floor {pa_floor}), seeds {list(seeds)}")
        t_pa = time.time()
        pa_by_seed, fitted, pa_info = fit_per_airport(
            Xb, label, train, is_rank, ap_code, airports, params, info["n_ref"], seeds,
            min_rows=PA_MIN_ROWS, floor=pa_floor, trees=args.pa_trees,
            es=dict(y=y, proxy=proxy, dlt=dlt, fit=fit, es=es, nest=nest, patience=patience), target=target)
        delta_hat = blend(pooled, mean_delta([pa_by_seed[s] for s in seeds]), fitted, BLEND)
        n_fitted = sum(1 for r in pa_info.values() if r["fitted"])
        log(f"per-airport done: {n_fitted}/{len(airports)} airports fitted, {int(fitted.sum()):,} "
            f"rows blended, {int((~fitted).sum()):,} rows keep pooled, {time.time() - t_pa:.0f}s; "
            f"mean |per-airport - pooled| on blended rows "
            f"{np.abs(delta_hat - pooled)[fitted].mean() / BLEND:.2f}s")
    else:
        delta_hat = pooled

    # ---- 5b. the fill head (v5): p on the ranking rows, mixed with the regressor's prediction ----
    head_meta = None
    if args.fillhead:
        t_head = time.time()
        n_all = int(train.sum())
        params_head = head_params(params)
        label = fill_label(y, sp, train)
        reused = bool(args.reuse_booster and reusable([head_files]))
        if reused:
            p, head_info = reuse_fill_head(head_files, X_rank, params_head, smoke, n_all, n_features=len(feats_all))
        else:
            p, head_info = fit_fill_head(X, label, train, fit, es, X_rank, params_head, nest, patience,
                                         files=head_files, smoke=smoke, n_features=len(feats_all))
        sp_rank = sp[is_rank]
        assert np.isfinite(sp_rank).all(), "sp is not finite on every ranking row: the head cannot mix"
        base_yhat = np.maximum(raw_taxi_time(delta_hat, proxy[is_rank], target), 1.0)   # the regressor arm's own prediction
        y_mix = mix_fill(p, sp_rank, base_yhat)
        pred = finalise_taxi_time(y_mix)
        bind = float((y_mix < 1.0).mean())
        nonpos = sp_rank <= 0.0
        head_meta = {**head_info, "params": params_head, "booster_file": head_files[0].name, "reused": reused,
                     "fill_tolerance_s": FILL_TOL_S, "nmdelay": NMDELAY, "head_wall_s": round(time.time() - t_head, 1),
                     "rank_mean_p": float(p.mean()),
                     "rank_p_quantiles": {str(q): float(np.percentile(p, q)) for q in (1, 10, 50, 90, 99)},
                     "rank_p_share_above_half": float((p > 0.5).mean()),
                     "mixture_shift_vs_regressor_mean_s": float((y_mix - base_yhat).mean()),
                     "mixture_shift_vs_regressor_rms_s": float(np.sqrt(((y_mix - base_yhat) ** 2).mean())),
                     "n_rank_sp_nonpositive": int(nonpos.sum()),
                     "mean_p_on_sp_nonpositive": float(p[nonpos].mean()) if nonpos.any() else None,
                     "floor_bind_rate": bind}
        log(f"fill head on {len(p):,} ranking rows: mean p {head_meta['rank_mean_p']:.4f} (training fill share "
            f"{head_info['fill_share_train']:.4f}), p > 0.5 on {100 * head_meta['rank_p_share_above_half']:.2f}%; "
            f"mixture shift vs the regressor: mean {head_meta['mixture_shift_vs_regressor_mean_s']:+.2f}s, RMS "
            f"{head_meta['mixture_shift_vs_regressor_rms_s']:.2f}s; {int(nonpos.sum()):,} rows with sp <= 0 carry "
            "mean p " + (f"{head_meta['mean_p_on_sp_nonpositive']:.4f}" if nonpos.any() else "n/a")
            + f"; {time.time() - t_head:.0f}s")
    else:
        raw = raw_taxi_time(delta_hat, proxy[is_rank], target)    # proxy - delta_hat, or the y prediction itself
        pred = finalise_taxi_time(raw)
        bind = float((raw < 1.0).mean())

    # ---- 6. the matched ranking rows ----
    log(f"predicted {len(pred):,} matched rows: median {np.median(pred):.0f}s  mean {pred.mean():.0f}s  "
        f"min {pred.min()}  max {pred.max()}  floor binds on {100 * bind:.3f}%")
    del X, X_rank, Xb, Xb_rank
    gc.collect()

    # ---- 7. splice into the base submission ----
    template = pq.read_table(RAW / "submitting.parquet").to_pandas()
    base = pq.read_table(base_path).to_pandas()
    B.check_submission(base, template)
    if args.all_rows:
        expected = set(template.MVT_ID_mvt)          # EVERY scored row: the unified model predicts them all
    else:
        expected = matched_ids(RAW / "ranking.parquet", template)
    if smoke:
        expected &= set(rank_ids)
    elif args.all_rows:
        assert len(expected) == len(template) == N_MATCHED + N_UNMATCHED, f"scored ids {len(expected):,} != {N_MATCHED + N_UNMATCHED:,}"
        assert len(rank_ids) == len(expected), f"{len(rank_ids):,} predictions for {len(expected):,} scored rows"
    else:
        assert len(expected) == N_MATCHED, f"matched ids {len(expected):,} != {N_MATCHED:,}"
        assert len(base) - len(expected) == N_UNMATCHED, "unmatched count != 5,290"
    out = splice(base, rank_ids, pred, expected)
    B.check_submission(out, template)
    hit = base.MVT_ID_mvt.isin(rank_ids).to_numpy()
    changed = out.TAXITIME_SEC_mvt.to_numpy() != base.TAXITIME_SEC_mvt.to_numpy()
    assert hit.sum() == len(rank_ids) == len(expected), "replaced-row count != matched id count"
    assert not changed[~hit].any(), "a row outside the matched set changed"
    assert np.array_equal(out.MVT_ID_mvt.to_numpy(), base.MVT_ID_mvt.to_numpy())
    diff = out.TAXITIME_SEC_mvt.to_numpy()[hit].astype("float64") - base.TAXITIME_SEC_mvt.to_numpy()[hit].astype("float64")
    log(f"splice: {hit.sum():,} rows replaced ({int(changed.sum()):,} values differ from the base), "
        f"{(~hit).sum():,} rows untouched; matched-row shift vs base: mean {diff.mean():+.1f}s, "
        f"RMS {np.sqrt((diff ** 2).mean()):.1f}s")

    out.to_parquet(dest, index=False)
    reread = pq.read_table(dest).to_pandas()            # verify what actually landed on disk
    B.check_submission(reread, template)
    base_disk = pq.read_table(base_path).to_pandas()
    assert np.array_equal(reread.MVT_ID_mvt.to_numpy(), base_disk.MVT_ID_mvt.to_numpy())
    assert reread.TAXITIME_SEC_mvt.dtype == base_disk.TAXITIME_SEC_mvt.dtype == np.int32
    assert np.array_equal(reread.TAXITIME_SEC_mvt.to_numpy(), out.TAXITIME_SEC_mvt.to_numpy())
    assert np.array_equal(reread.TAXITIME_SEC_mvt.to_numpy()[~hit], base_disk.TAXITIME_SEC_mvt.to_numpy()[~hit])
    log(f"wrote {dest}  rows={len(reread):,}  median={reread.TAXITIME_SEC_mvt.median():.0f}s  "
        f"mean={reread.TAXITIME_SEC_mvt.mean():.0f}s")

    # ---- 8. provenance ----
    finished = dt.datetime.now(dt.timezone.utc)
    meta = {"version": args.version, "file": dest.name, "base": base_path.name, "smoke": smoke,
            **_git(), "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
            "wall_s": round((finished - started).total_seconds(), 1),
            "peak_rss_gb": round(peak_rss_gb(), 2), **info, "params": params,
            "seeds": list(seeds), "n_seeds": len(seeds), "es_seed": ES_SEED,
            "booster_files": [p.name for p, _ in files],
            "per_airport": pa_info, "blend": BLEND if args.per_airport else None,
            "pa_min_rows": PA_MIN_ROWS if args.per_airport else None,
            "pa_tree_floor": pa_floor if args.per_airport else None,
            "pa_tree_rule": pa_rule if args.per_airport else None,
            "pa_trees": args.pa_trees if args.per_airport else None,
            "max_rounds": nest, "patience": patience,
            "features": feats, "n_features": len(feats),
            "queue": args.queue, "queue_feats": QUEUE_FEATS if args.queue else None,
            "dayfeats": args.dayfeats, "day_feats": DAY_FEATS if args.dayfeats else None,
            "orderfeats": args.orderfeats, "order_feats": ORDER_FEATS if args.orderfeats else None,
            "weatherfeats": args.weatherfeats, "weather_feats": WEATHER_FEATS if args.weatherfeats else None,
            "target": target,
            "all_rows": bool(args.all_rows), "unmatched_weight": float(args.unmatched_weight) if args.all_rows else None,
            "n_unmatched_train": int(is_um[train].sum()), "n_unmatched_rank": int(is_um[is_rank].sum()),
            "unmatched_nan_cols": list(S.UNMATCHED_NAN_COLS) if args.all_rows else None,
            "unmatched_flt_cols": list(S.UNMATCHED_FLT_COLS) if args.all_rows else None,
            "unmatched_nan_pattern_checked": bool(args.all_rows),
            "fillhead": args.fillhead, "fill_head": head_meta,
            "features_head": feats_all if args.fillhead else None,
            "n_features_head": len(feats_all) if args.fillhead else None,
            "months": sorted(months) if months else "all twelve", "n_rank_rows": int(is_rank.sum()),
            "n_replaced": int(hit.sum()), "n_value_changed": int(changed.sum()),
            "n_unchanged": int((~hit).sum()), "floor_bind_rate": bind,
            "pred_median": float(np.median(pred)), "pred_mean": float(pred.mean()),
            "shift_vs_base_mean_s": float(diff.mean()), "shift_vs_base_rms_s": float(np.sqrt((diff ** 2).mean())),
            "replaced_ids": [float(i) for i in rank_ids] if smoke else None}
    meta_path = dest.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"meta -> {meta_path}")
    if smoke:
        log(SMOKE_BANNER)
    log(f"done: wall {meta['wall_s']:.0f}s, peak RSS {meta['peak_rss_gb']:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
