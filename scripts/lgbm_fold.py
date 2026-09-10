"""FOLD MEASUREMENTS on fold A, exactly as scripts/lgbm_ab.py measured it: the v4 arms
(Amendment 12), the queue arm (Amendment 14), the CatBoost arm (Amendment 15.1) and the
capacity sweep's two tiers (Amendment 15.2). One run measures ONE arm; every arm reports
measurements, never verdict labels - the amendments hold the thresholds and apply them.

    ENV="PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 python3.11 -B -u"
    $ENV scripts/lgbm_fold.py [--pa-trees es]                       # v4: A0/A1a/A1b/A2/A3
    $ENV scripts/lgbm_fold.py --queue    [--baseline A3] [--pa-trees es]   # Amendment 14
    $ENV scripts/lgbm_fold.py --catboost [--baseline A3] [--pa-trees es]   # Amendment 15.1
    $ENV scripts/lgbm_fold.py --sweep                                       # 15.2 screening tier
    $ENV scripts/lgbm_fold.py --confirm "255,40,0.8" "511,20,0.6"           # 15.2 confirmation tier
    $ENV scripts/lgbm_fold.py --fillhead [--baseline A3] [--pa-trees es]   # Amendment 16
    $ENV scripts/lgbm_fold.py --dayfeats [--queue] [--baseline A2] [--pa-trees es]   # Amendment 19.1, arm D
    $ENV scripts/lgbm_fold.py --orderfeats [--queue] [--baseline A2] [--pa-trees es] # Amendment 22, arm F
    $ENV scripts/lgbm_fold.py --ytarget  [--queue] [--baseline A2] [--pa-trees es]   # Amendment 19.2, arm Y
    $ENV scripts/lgbm_fold.py --ytarget --all-rows [--queue] [--dayfeats] --unmatched-weight 1,10 [--baseline A2]
                                                                            # the UNIFIED all-rows arm
    any of the above with --smoke                                           # code path only

Fold A: holdout months (1, 7) of the 2025 training caches; early-stopping months (3, 9) carved
out of the TRAINING fold; the 24 encodings fitted in-fold on the ten training months
(stand_ab.infold_encodings); matched rows only; target delta; y_hat = max(proxy - delta_hat, 1),
unrounded, as lgbm_ab scored it. The fitting code is lgbm_submit's (early_stop, refit,
fit_seeds, fit_per_airport, blend, predict_delta) so the fold measures the code that ships.

v4 arms - each a per-row delta_hat on the fold's 339,015 matched holdout rows:
  A0   seed 0 alone: the lgbm_ab refit procedure. Must reproduce lgb_refit 226.24 within 0.5 s
       (reports/lgbm_ab_result.json) or the harness is wrong and nothing below counts. LightGBM
       at num_threads=4 is not bit-deterministic run to run on this machine (the two v3 ES logs
       diverge from iteration 15,000), so the check is a tolerance, not an equality.
  A1a  seed 1 alone  }  with A0 these three single-seed RMSEs give the seed sd
  A1b  seed 2 alone  }
  A2   mean of seeds 0, 1, 2
  A3   A2 blended (1 - BLEND)/BLEND = 0.5/0.5 with a per-airport LightGBM at airports with
       >= 20,000 training rows (others keep A2); the per-airport model is itself the mean over
       the three seeds.
Reported, to reports/lgbm_fold_v4.json and a readable reports/lgbm_fold_v4.log: matched RMSE per
arm, pooled AND per airport; the seed sd (sample sd, ddof=1, of the three single-seed pooled
RMSEs); PAIRED row-bootstrap intervals (2,000 draws, seed 0, one set of draws for every pair) of
the gain rmse(ref) - rmse(new) for A2 vs A0, A3 vs A2 and A3 vs A0, pooled and per airport; per
pair gain / seed sd, the repo's ESTABLISHED check gain > 2 x seed sd, and the number of airports
whose own RMSE strictly improves. Every arm's per-row prediction - and each seed's, and each
per-airport seed's - goes to data/cache_stand/fold_v4_preds.parquet so any later cut can be made
without a refit; it is rewritten after every arm so a crash loses one arm, not the run.
Per-airport tree count (Amendment 12.4 says the rule must be recorded): --pa-trees `share`
(default) = max(200, int(n_ref * n_airport / n_all)) (lgbm_submit.per_airport_trees), `es` =
each airport's own early-stopping run on its own fit/ES rows (lgbm_submit.PA_TREE_RULE_ES).

THE BASELINE OF EVERY LATER ARM is the v4 arm that shipped, --baseline A3 (default) or A2 per
Amendment 12.3's rule, taken as the v4 run's OWN per-row predictions from
data/cache_stand/fold_v4_preds.parquet (rebuilt from its per-seed columns for --seeds, checked
against the stored arm column) with the seed sd from reports/lgbm_fold_v4.json. The fold must
be identical - row index, month, airport, y, proxy, delta all exact - and the v4 configuration
must match the treatment's (same seed set, same --pa-trees for A3, A0 reproduced), else the run
refuses. --refit-baseline fits the baseline in-process instead (a smoke without v4 outputs does
so automatically); the seed sd is then the in-process baseline's own three seeds.

--queue (Amendment 14, H-14): treatment = baseline + stand_ab.QUEUE_FEATS (12 columns from
data/cache_queue/, joined onto each training month by POSITION - equal row counts asserted, the
identical order proven by tests/test_queue_features.py), everything else identical, best_iter
re-found with the extra columns. Reported per 14.2/14.3: the paired interval and point gain of
treatment - baseline, pooled and per airport; airports improving (>= 4 of 10 with LEBL, LTFM,
EGLL, LSZH named in advance); the pre-defined tail subset (non-fill := |y - sp| > 60 s, AND
|delta| > 1,200 s) RMSE for both arms with its own paired interval; and the 14.4 negative
control with a REGISTERED DEVIATION (CONTROL_DEVIATION): 20 within-airport row permutations of
the queue block (seed 0; the block moves as whole rows within airport x fold side), each a
single-seed pooled refit at best_iter // 4 trees, compared like-for-like with the unpermuted
treatment and the baseline refit at the same reduced count. The treatment's reduced-capacity
gain vs the 95th percentile of the permuted gains is reported. Predictions ->
data/cache_stand/fold_preds_queue.parquet; record -> reports/lgbm_fold_queue.json.

--catboost (Amendment 15.1, H-15a): the design matrix, target (holdout targets blanked) and
masks go to data/cache_stand/catboost_fold_input.npz; scripts/catboost_fold_worker.py runs
UNDER ~/.venvs/prc-catboost/bin/python as a subprocess (catboost is never imported here - it
needs numpy < 2 and the global site-packages are shared with a live fleet), early-stops on the
ES rows (depth 8, lr 0.03, up to 10,000 iterations, od_wait 200, seed 0, 4 threads), refits on
all training rows at the found count and writes the holdout predictions. The design matrix is
freed before the worker starts so the two never hold it together. Blends (1 - w) baseline +
w CatBoost at w = 0.3/0.5/0.7; reported: the five arms, the paired intervals of each blend and
of CatBoost alone vs the baseline, the weight curve at 0/0.3/0.5/0.7/1 with whether it rises
monotonically over 0.3/0.5/0.7 (dilution), and the residual correlation between the families.

--sweep (Amendment 15.2 screening tier): 18 settings (num_leaves x min_data_in_leaf x
feature_fraction) early-stopped at lr 0.05 on a subfold that trains on Feb-May and stops on Jun
(encodings fitted on Feb-May only), RANKED by their stopping-set matched RMSE. Every log line
and every ranking entry carries "RANKING ONLY — no magnitude from this tier is a result": this
project has twice quoted a subsample magnitude as a result. No predictions parquet.

--confirm S [S ...] (15.2 confirmation tier): each setting is a full fold-A arm (early stopping
seed 0, refit per seed, mean over --seeds) paired against the incumbent setting's arm over the
same seeds (the v4 A2 record, or --refit-baseline), with the paired interval and the
2 x seed sd rule (seed sd from the v4 JSON when present, else the incumbent's own seeds);
each candidate's own seed sd is recorded too. A setting outside the grid is refused.

--fillhead (Amendment 16, H-16): treatment = baseline + the schedule-fill mixture head. The head
is lgbm_submit.fit_fill_head - a LightGBM binary classifier p = P(|y - sp| <= 60 s | x) on
FEATS + [nmdelay = proxy - sp] (69 columns; a baseline refit in-process reads the first 68 of
the same matrix), early-stopped on the ES months (metric auc, seed 0), refit ONCE on all training
rows at n_ref - and the treatment's taxi time is p * sp + (1 - p) * max(proxy - baseline, 1),
stored in the parquet as `treatment` = proxy - mixture (the delta form every arm uses, so
max(proxy - treatment, 1) recovers the mixture) beside `p`. Reported per 16.2/16.3: the paired
interval and point gain of treatment - baseline, pooled and per airport; the FILL (|y - sp| <= 60)
and NON-FILL holdout subsets with each arm's RMSE and their own paired intervals; LIRF alone with
the same two cuts; the reliability deciles of p (equal-count bins, mean p vs realised fill rate);
the 2 x seed sd check against the v4 JSON's seed sd (the head is single-seed, so the treatment
has no seed sd of its own). Predictions -> data/cache_stand/fold_preds_fillhead.parquet; record
-> reports/lgbm_fold_fillhead.json.

--dayfeats (Amendment 19.1, arm D): treatment = the current best design + stand_ab.DAY_FEATS (9
airport-day regime columns from data/cache_day/, joined onto each training month by POSITION
exactly as the queue block is; the row contract is tests/test_day_features.py's), best_iter
re-found. The current best design is the v4 arm (--baseline A2/A3, read from the v4 record) or,
with --queue, the queue fold's own treatment A2/A3 + QUEUE_FEATS read from ITS record
(--queue-preds / --queue-json, default data/cache_stand/fold_preds_queue.parquet +
reports/lgbm_fold_queue.json; the record's baseline_arm must be the arm asked for); --queue makes
the mode queue_day with its own output names. Reported per 19.3: the paired interval and point
gain pooled and per airport with the >= 6 of 10 count, the seed sd rule, and both arms' RMSE with
the treatment's paired interval on each |delta| band of 19.0 (< 2 min, 2-10, 10-20, > 20 min;
left-closed, right-open in seconds) plus their union over10 (|delta| >= 600 s, the "> 10 min
bands" 19.3 needs to move by >= +5 s). Predictions -> data/cache_stand/fold_preds_day.parquet
(fold_preds_queue_day.parquet), record -> reports/lgbm_fold_day.json (lgbm_fold_queue_day.json).

--ytarget (Amendment 19.2, arm Y): the same design, configuration and seeds as the delta arm
(the v4 arm, or the queue treatment under --queue, read from its record - it is both the pair's
reference and the blend partner, so --blend-with is the same knob as --baseline), fitted with
target y (lgbm_submit's --target y: label y, the stopping metric on y directly, y_hat_Y =
max(prediction, 1), no proxy anchor); and the 0.5/0.5 blend of y_hat_Y with the delta arm's
y_hat (YBLEND). Reported per 19.2/19.3: Y alone and the blend, each paired against the delta arm
pooled, per airport (LTFM and EDDM named) and on the |delta| bands; the seed sd rule. The parquet
carries the per-seed y predictions (y_seed{s}, y scale) and the arms in the delta form every arm's
column uses (max(proxy - col, 1) recovers the taxi time). Predictions ->
data/cache_stand/fold_preds_ytarget.parquet (fold_preds_queue_ytarget.parquet), record ->
reports/lgbm_fold_ytarget.json (lgbm_fold_queue_ytarget.json).

--ytarget --all-rows (the UNIFIED arm, the Amendment 19 addition): ONE regressor with target y
on EVERY admissible training row - the matched caches AND the unmatched cache
(data/cache_unmatched, stand_ab.py unmatched-cache: the stratum's rows with every AOBT_3-anchored
column NaN, asserted, never filled) - on the design FEATS [+ QUEUE_FEATS with --queue] [+
DAY_FEATS with --dayfeats] + is_unmatched, the delta encodings fitted on the matched training
rows only, an optional sample weight W on the unmatched rows (--unmatched-weight, a list: each W
is its own arm U_w{W}, three seeds), y_hat = max(prediction, 1) on every row, no proxy anywhere.
Paired against the CURRENT PIPELINE's per-row predictions: the matched holdout rows from the v4 /
queue record (as arms D and Y), the unmatched holdout rows from the stratum fold's shipped
fit_unmatched (S0 of data/cache_stand/stratum_fold_v7_preds.parquet, fold "A", joined by id;
--stratum-preds). Reported THREE ways, each arm and its 0.5/0.5 blend with the pipeline
(Ublend_w{W}): (i) the matched rows, (ii) the unmatched rows pooled AND ex-monster (y <= 10,800
s), both with per-airport cuts and paired row intervals on the subset's rows, (iii) the fold
TOTAL at the 2026 scored-file weights (w_m 0.984660 / w_u 0.015340) with a STRATIFIED paired row
bootstrap (each stratum resampled within itself); plus the |delta| bands on the matched rows.
The parquet's arm columns are TAXI TIMES (the unmatched rows have no proxy) with is_unmatched
and MVT_ID_mvt. Modes allrows / queue_allrows (+ _day) with their own output names.

Smoke (--smoke): months (1, 2, 3) - holdout 1, early-stop 3, fit 2 (the sweep: train 2, stop
3); tiny trees; 200 bootstrap draws; a scratch directory; the banner. The reproduction check is
skipped there. A smoke proves the code path and nothing else.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.util
import json
import os
import pathlib
import subprocess
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_stand"
QCACHE = ROOT / "data" / "cache_queue"
DCACHE = ROOT / "data" / "cache_day"
UCACHE = ROOT / "data" / "cache_unmatched"
REPORTS = ROOT / "reports"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L = _load("lgbm_submit")
S = _load("stand_ab")

HOLDOUT = (1, 7)                          # the evaluation is January and July; hold out their 2025 twins
SEEDS = (0, 1, 2)
N_BOOT, BOOT_SEED = 2_000, 0
PAIRS = (("A2", "A0"), ("A3", "A2"), ("A3", "A0"))       # (new, reference): gain = rmse(ref) - rmse(new)
ARM_DEFINITIONS = {
    "A0": "seed 0 alone (lgbm_ab's lgb_refit procedure)",
    "A1a": "seed 1 alone",
    "A1b": "seed 2 alone",
    "A2": "mean of seeds 0, 1, 2",
    "A3": f"A2 blended {1 - L.BLEND:.1f}/{L.BLEND:.1f} with the per-airport model (mean over seeds "
          f"0, 1, 2) at airports with >= {L.PA_MIN_ROWS:,} training rows; A2 elsewhere",
}
#: reports/lgbm_ab_full.log + reports/lgbm_ab_result.json, 2026-09-08
LGBM_AB = dict(lgb_refit=226.24494650121161, best_iter=17_557, n_ref=21_948, n_fit=1_378_585,
               n_es=344_840, n_all=1_723_425, n_test=339_015)
REPRO_TOL_S = 0.5
SMOKE = dict(months=(1, 2, 3), learning_rate=0.05, nest=60, patience=50, n_boot=200, pa_floor=20)
JSON_NAME, LOG_NAME, PREDS_NAME = "lgbm_fold_v4.json", "lgbm_fold_v4.log", "fold_v4_preds.parquet"
V4_JSON, V4_PREDS = REPORTS / JSON_NAME, CACHE / PREDS_NAME

# ---- Amendment 14: the queue arm --------------------------------------------------------------
QUEUE_FEATS = list(L.QUEUE_FEATS)
FEATS_QUEUE = list(L.FEATS) + QUEUE_FEATS
BASELINE_ARMS = ("A2", "A3")
NAMED_AIRPORTS = ("LEBL", "LTFM", "EGLL", "LSZH")         # 14.3: the four named in advance as likely
#: 14.3: non-fill := |y - sp| > 60 (the complement of 16.2's fill, one number); |delta| > 1,200
TAIL_NONFILL_S, TAIL_DELTA_S = L.FILL_TOL_S, 1_200.0
N_PERM, PERM_SEED = 20, 0
CONTROL_DEVIATION = (
    "20 permutations at quarter capacity, paired: Amendment 14.4 registers 100 within-airport row "
    "permutations of QUEUE_FEATS at full capacity. A permutation must RETRAIN - re-predicting the trained "
    "treatment booster on permuted columns measures nothing about training - and 100 full retrains are "
    "infeasible on this machine, so the control is 20 permutations (seed 0; the twelve columns move as "
    "whole rows within airport x fold side), each a single-seed pooled refit on all training rows at "
    "best_iter // 4 trees, compared like-for-like with the unpermuted treatment AND the baseline refit at "
    "the same reduced count; no per-airport blend. To be amended into 14.4 before the run.")

# ---- Amendment 15.1: the CatBoost arm ---------------------------------------------------------
CATBOOST = dict(depth=8, learning_rate=0.03, iterations=10_000, od_wait=200, seed=0, threads=4)
SMOKE_CATBOOST = dict(iterations=60, od_wait=20)
CATBOOST_PYTHON = pathlib.Path.home() / ".venvs" / "prc-catboost" / "bin" / "python"
CATBOOST_WORKER = ROOT / "scripts" / "catboost_fold_worker.py"
CATBOOST_INPUT, CATBOOST_OUTPUT = "catboost_fold_input.npz", "catboost_fold_preds.npy"
BLEND_WEIGHTS = (0.3, 0.5, 0.7)                           # 15.1: the registered blend weights
CURVE_WEIGHTS = (0.0, 0.3, 0.5, 0.7, 1.0)

# ---- Amendment 15.2: the capacity sweep -------------------------------------------------------
SWEEP_AXES = dict(num_leaves=(127, 255, 511), min_data_in_leaf=(20, 40, 100), feature_fraction=(0.6, 0.8))
INCUMBENT_SETTING = (L.P["num_leaves"], L.P["min_data_in_leaf"], L.P["feature_fraction"])
SWEEP = dict(train=(2, 3, 4, 5), stop=6, learning_rate=0.05)
SMOKE_SWEEP = dict(train=(2,), stop=3)
RANKING_BANNER = "RANKING ONLY — no magnitude from this tier is a result"

# ---- Amendment 16: the schedule-fill mixture head ------------------------------------------
FEATS_HEAD = L.head_features(L.FEATS)                     # FEATS + [nmdelay]: the head's 69 columns
N_DECILES = 10                                            # reliability: equal-count bins of p on the holdout

# ---- Amendment 19: arm D (the airport-day regime block) and arm Y (the y formulation) -------
DAY_FEATS = list(L.DAY_FEATS)
FEATS_DAY = list(L.FEATS) + DAY_FEATS                     # 77 columns
FEATS_QUEUE_DAY = FEATS_QUEUE + DAY_FEATS                 # 89 columns: the queue design + the day block
#: Amendment 22 arm F: the record-ordering block
OCACHE = ROOT / "data" / "cache_order"
ORDER_FEATS = list(L.ORDER_FEATS)
FEATS_ORDER = list(L.FEATS) + ORDER_FEATS                 # 71 columns
FEATS_QUEUE_ORDER = FEATS_QUEUE + ORDER_FEATS             # 83 columns: the queue design + the order block
YBLEND = 0.5                                              # 19.2: the 0.5/0.5 blend of y_hat_Y with the delta arm's y_hat
YTARGET_NAMED_AIRPORTS = ("LTFM", "EDDM")                 # 19.2: corr(proxy, y) 0.08 / 0.16 - called out per airport
DAY_MIN_AIRPORTS = 6                                      # 19.3: arm D needs >= 6 of 10 airports to improve
#: 19.0's |delta| bands in seconds, left-closed / right-open, a partition of the holdout; over10
#: is the union of the upper two (19.3's "|delta| > 10 min bands"), |delta| >= OVER10_S
DELTA_BANDS = (("lt2", 0.0, 120.0), ("2to10", 120.0, 600.0), ("10to20", 600.0, 1_200.0), ("gt20", 1_200.0, float("inf")))
OVER10_S = 600.0
#: the queue fold's record: under --queue it is the current best design's own predictions
QUEUE_JSON, QUEUE_PREDS = REPORTS / "lgbm_fold_queue.json", CACHE / "fold_preds_queue.parquet"

# ---- the unified all-rows arm -------------------------------------------------------------------
IS_UNMATCHED = L.IS_UNMATCHED
STRATUM_PREDS = CACHE / "stratum_fold_v7_preds.parquet"     # the shipped unmatched predictions, fold "A" rows
#: the 2026 scored file's shares: 339,551 matched and 5,290 unmatched of 344,841 rows
W_MATCHED_2026, W_UNMATCHED_2026 = 339_551 / 344_841, 5_290 / 344_841
MONSTER_S = 10_800.0                                        # y > 3 h: Amendment 9 rule 4, the stratum's ex-monster cut
UNIFIED_AMENDMENT = "19-unified"
ALLROWS_MODES = ("allrows", "queue_allrows", "allrows_day", "queue_allrows_day")

MODES = ("v4", "queue", "catboost", "sweep", "confirm", "fillhead", "day", "queue_day", "ytarget", "queue_ytarget",
         "order", "queue_order", *ALLROWS_MODES)
OUTPUT_NAMES = {
    "v4": (JSON_NAME, LOG_NAME, PREDS_NAME),
    "queue": ("lgbm_fold_queue.json", "lgbm_fold_queue.log", "fold_preds_queue.parquet"),
    "catboost": ("lgbm_fold_catboost.json", "lgbm_fold_catboost.log", "fold_preds_catboost.parquet"),
    "sweep": ("lgbm_fold_sweep.json", "lgbm_fold_sweep.log", None),
    "confirm": ("lgbm_fold_confirm.json", "lgbm_fold_confirm.log", "fold_preds_confirm.parquet"),
    "fillhead": ("lgbm_fold_fillhead.json", "lgbm_fold_fillhead.log", "fold_preds_fillhead.parquet"),
    "day": ("lgbm_fold_day.json", "lgbm_fold_day.log", "fold_preds_day.parquet"),
    "queue_day": ("lgbm_fold_queue_day.json", "lgbm_fold_queue_day.log", "fold_preds_queue_day.parquet"),
    "order": ("lgbm_fold_order.json", "lgbm_fold_order.log", "fold_preds_order.parquet"),
    "queue_order": ("lgbm_fold_queue_order.json", "lgbm_fold_queue_order.log", "fold_preds_queue_order.parquet"),
    "ytarget": ("lgbm_fold_ytarget.json", "lgbm_fold_ytarget.log", "fold_preds_ytarget.parquet"),
    "queue_ytarget": ("lgbm_fold_queue_ytarget.json", "lgbm_fold_queue_ytarget.log", "fold_preds_queue_ytarget.parquet"),
    "allrows": ("lgbm_fold_allrows.json", "lgbm_fold_allrows.log", "fold_preds_allrows.parquet"),
    "queue_allrows": ("lgbm_fold_queue_allrows.json", "lgbm_fold_queue_allrows.log", "fold_preds_queue_allrows.parquet"),
    "allrows_day": ("lgbm_fold_allrows_day.json", "lgbm_fold_allrows_day.log", "fold_preds_allrows_day.parquet"),
    "queue_allrows_day": ("lgbm_fold_queue_allrows_day.json", "lgbm_fold_queue_allrows_day.log",
                          "fold_preds_queue_allrows_day.parquet"),
}
_ENV = ("cd ~/Projects/prc-challenge && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 "
        "/usr/bin/time -l python3.11 -B -u scripts/lgbm_fold.py")
COMMAND = f"{_ENV} 2>&1 | tee reports/lgbm_fold_v4.console.log"
COMMANDS = {
    "v4": COMMAND,
    "queue": f"{_ENV} --queue --baseline A3 --pa-trees es 2>&1 | tee reports/lgbm_fold_queue.console.log",
    "catboost": f"{_ENV} --catboost --baseline A3 --pa-trees es 2>&1 | tee reports/lgbm_fold_catboost.console.log",
    "sweep": f"{_ENV} --sweep 2>&1 | tee reports/lgbm_fold_sweep.console.log",
    "confirm": f"{_ENV} --confirm \"<screen rank 1>\" \"<screen rank 2>\" 2>&1 | tee reports/lgbm_fold_confirm.console.log",
    "fillhead": f"{_ENV} --fillhead --baseline A3 --pa-trees es 2>&1 | tee reports/lgbm_fold_fillhead.console.log",
    "day": f"{_ENV} --dayfeats --baseline A2 --pa-trees es 2>&1 | tee reports/lgbm_fold_day.console.log",
    "queue_day": f"{_ENV} --dayfeats --queue --baseline A2 --pa-trees es 2>&1 | tee reports/lgbm_fold_queue_day.console.log",
    "order": f"{_ENV} --orderfeats --baseline A2 --pa-trees es 2>&1 | tee reports/lgbm_fold_order.console.log",
    "queue_order": f"{_ENV} --orderfeats --queue --baseline A2 --pa-trees es 2>&1 | tee reports/lgbm_fold_queue_order.console.log",
    "ytarget": f"{_ENV} --ytarget --baseline A2 --pa-trees es 2>&1 | tee reports/lgbm_fold_ytarget.console.log",
    "queue_ytarget": f"{_ENV} --ytarget --queue --baseline A2 --pa-trees es 2>&1 | tee reports/lgbm_fold_queue_ytarget.console.log",
    "allrows": f"{_ENV} --ytarget --all-rows --unmatched-weight 1,10 --baseline A2 --pa-trees es 2>&1 | tee reports/lgbm_fold_allrows.console.log",
    "queue_allrows": f"{_ENV} --ytarget --all-rows --queue --unmatched-weight 1,10 --baseline A2 --pa-trees es 2>&1 | tee reports/lgbm_fold_queue_allrows.console.log",
    "allrows_day": f"{_ENV} --ytarget --all-rows --dayfeats --unmatched-weight 1,10 --baseline A2 --pa-trees es 2>&1 | tee reports/lgbm_fold_allrows_day.console.log",
    "queue_allrows_day": f"{_ENV} --ytarget --all-rows --queue --dayfeats --unmatched-weight 1,10 --baseline A2 --pa-trees es 2>&1 | tee reports/lgbm_fold_queue_allrows_day.console.log",
}
ESTIMATE = (
    "full run ~1.4 h (range 1.2-2 h), peak RSS ~5 GB, 4 threads; writes reports/lgbm_fold_v4.json, "
    "reports/lgbm_fold_v4.log, data/cache_stand/fold_v4_preds.parquet. Arithmetic from "
    "reports/lgbm_submit_v3.log (ES 2.5e-8 s/row/iter, refit 2.2e-8 s/row/iter, predict 453 s for "
    "30,846 trees x 339k rows): ES ~17.8k iters on 1.38M rows ~15 min incl. the stopping-set predict; "
    "3 refits x (21,948 trees on 1.72M rows ~14 min + holdout predict ~5.5 min) ~58 min; per-airport "
    "3 seeds x 10 airports at the share count ~8 min; bootstrap + writes ~2 min. With --pa-trees es "
    "add ~30 min (per-airport ES runs ~6 min, larger per-airport refits ~8 min per seed): ~1.9 h.")
ESTIMATES = {
    "v4": ESTIMATE,
    "queue": (
        "~3.6 h (range 3-4.5 h), peak RSS ~5.5 GB (the v4 footprint + 12 float32 columns), 4 threads; run "
        "ALONE. Same per-row/iter rates as the v4 estimate x1.15 for 80 columns: ES ~16 min; 3 refits at "
        "~21.9k trees ~21 min each (64 min); per-airport `es` ~35 min; the control = 22 pooled refits at "
        "~4.4k trees (~4.3 min each incl. predict) ~95 min; bootstrap + 22 parquet rewrites ~3 min. The "
        "baseline is READ from the v4 record (no refit). Writes reports/lgbm_fold_queue.{json,log} and "
        "data/cache_stand/fold_preds_queue.parquet."),
    "catboost": (
        "UNMEASURED - no CatBoost timing exists in this repo. Parent: load + npz write ~2 min at ~3 GB peak, "
        "then the design matrix is freed (~0.3 GB while the worker runs). Worker (venv, 4 threads): depth 8 on "
        "1.38M x 68 at a guessed 0.1-0.3 s/iteration -> up to 10k ES iterations 17-50 min + the refit on "
        "1.72M rows 20-60 min: ~1-2 h; worker RSS ~2-3 GB (guess: raw float32 + CatBoost's quantized pool). "
        "Bootstrap ~1 min. Writes reports/lgbm_fold_catboost.{json,log}, data/cache_stand/"
        "fold_preds_catboost.parquet, catboost_fold_input.npz (0.56 GB, kept) and catboost_fold_preds.npy."),
    "sweep": (
        "~30-45 min, peak RSS ~1.5 GB (five months), 4 threads. Per setting: early stopping at lr 0.05 on "
        "~664k Feb-May rows (best_iter ~3.5k at lr 0.05 for 255 leaves; ~x0.6 for 127, ~x1.7 for 511) ~1 min "
        "+ the 181k-row stopping-set predict ~0.5 min: ~1.5-2.5 min x 18. RANKING ONLY. Writes "
        "reports/lgbm_fold_sweep.{json,log}; no parquet."),
    "confirm": (
        "~1.5 h per setting (range 1.2-2 h; x1.7 for 511 leaves), peak RSS ~5 GB, 4 threads; run ALONE: "
        "ES ~15 min + 3 refits x (~14 min + 5.5 min predict). The incumbent is READ from the v4 record. "
        "Writes reports/lgbm_fold_confirm.{json,log} and data/cache_stand/fold_preds_confirm.parquet."),
    "fillhead": (
        "~30-50 min (range 20-70 min), peak RSS ~4 GB (the v4 load footprint; one 69-column matrix), 4 threads; "
        "run ALONE. The baseline is READ from the v4 record (no regressor refit). The head at the v4 per-row/iter "
        "rates (ES 2.5e-8 s, refit 2.2e-8 s) x1.015 for 69 columns; the round count of a binary head at lr 0.01 is "
        "UNMEASURED (the regressor took 17.5k): 5k-15k ES rounds on 1.38M rows ~3-9 min + the stopping-set predict "
        "1-3 min; one refit on 1.72M rows at n_ref = 1.25 x best_iter ~4-11 min + the holdout predict 1-3 min; "
        "load + encodings ~3 min; bootstrap (pooled + 10 airports + 2 subsets + 3 LIRF cuts, 2,000 draws) ~3 min. "
        "With --refit-baseline add the full v4 run (~1.9 h with --pa-trees es). Writes "
        "reports/lgbm_fold_fillhead.{json,log} and data/cache_stand/fold_preds_fillhead.parquet."),
    "day": (
        "~1.7 h (range 1.4-2.2 h), peak RSS ~4.5 GB (the v4 footprint + 9 float32 columns), 4 threads; run ALONE. "
        "The v4 per-row/iter rates x1.13 for 77 columns: load + encodings + the day join ~3 min; ES ~17 min; 3 refits "
        "at ~22k trees ~16 min + holdout predict ~6 min each (~66 min); bootstrap (pooled + 10 airports + 5 bands, "
        "2,000 draws) ~3 min. With --baseline A3 --pa-trees es add ~35 min for the treatment's per-airport models. "
        "The baseline is READ from the v4 record (no refit). Writes reports/lgbm_fold_day.{json,log} and "
        "data/cache_stand/fold_preds_day.parquet."),
    "order": (
        "~1.6 h (range 1.3-2.1 h), peak RSS ~4.4 GB (the v4 footprint + 3 float32 columns), 4 threads; run ALONE. "
        "The v4 per-row/iter rates x1.04 for 71 columns: load + encodings + the order join ~3 min; ES ~16 min; "
        "3 refits ~15 min + holdout predict ~6 min each (~63 min); bootstrap ~3 min. The baseline is READ from the "
        "v4 record (no refit). Writes reports/lgbm_fold_order.{json,log} and "
        "data/cache_stand/fold_preds_order.parquet."),
    "queue_order": (
        "~1.9 h (range 1.5-2.5 h), peak RSS ~4.9 GB (the queue fold's 4.8 GB + 3 float32 columns), 4 threads; run "
        "ALONE. x1.22 for 83 columns: load + the two joins ~3 min; ES ~19 min; 3 refits ~18 min + predict ~6 min "
        "each (~72 min); bootstrap ~3 min. The baseline is READ from the queue record. Writes "
        "reports/lgbm_fold_queue_order.{json,log} and data/cache_stand/fold_preds_queue_order.parquet."),
    "queue_day": (
        "~2.0 h (range 1.6-2.6 h), peak RSS ~5 GB (the queue fold's 4.8 GB + 9 float32 columns), 4 threads; run "
        "ALONE. x1.3 for 89 columns: load + the two joins ~3 min; ES ~20 min; 3 refits ~19 min + predict ~6 min each "
        "(~75 min); bootstrap ~3 min. The baseline is READ from the queue record (fold_preds_queue.parquet's "
        "treatment column; no refit). Writes reports/lgbm_fold_queue_day.{json,log} and "
        "data/cache_stand/fold_preds_queue_day.parquet."),
    "ytarget": (
        "~1.4 h (range 1.1-2 h), peak RSS ~4.2 GB (the v4 footprint), 4 threads; run ALONE. The round count of a "
        "y-target early stop is UNMEASURED (the label's variance is larger than delta's; at lr 0.01 a best_iter "
        "near the delta arm's 17.5k is the guess): ES ~15 min; 3 refits ~14 min + holdout predict ~5.5 min each "
        "(~58 min); bootstrap (2 pairs x (pooled + 10 airports + 5 bands), 2,000 draws) ~5 min. The delta arm is "
        "READ from the v4 record. Writes reports/lgbm_fold_ytarget.{json,log} and "
        "data/cache_stand/fold_preds_ytarget.parquet."),
    "queue_ytarget": (
        "~1.6 h (range 1.3-2.3 h), peak RSS ~4.8 GB (the queue fold's footprint), 4 threads; run ALONE. x1.15 for "
        "80 columns: ES ~16 min; 3 refits ~21 min each; bootstrap ~5 min. The delta arm is READ from the queue "
        "record (fold_preds_queue.parquet's treatment column). Writes reports/lgbm_fold_queue_ytarget.{json,log} "
        "and data/cache_stand/fold_preds_queue_ytarget.parquet."),
    "allrows": (
        "~2.7 h (range 2.2-3.5 h) for --unmatched-weight 1,10: two unified arms, each an ES run (~15 min) + 3 refits "
        "(~14 min + 5.5 min predict each) on 1.75M rows at 69 columns; load (12 stand + 12 unmatched caches) + "
        "encodings ~4 min; bootstrap (3 subsets x 4 pairs x (pooled + per airport) + the stratified total + 5 bands, "
        "2,000 draws) ~6 min. Peak RSS ~4.5 GB (the v4 footprint + ~22k rows + is_unmatched), 4 threads; run ALONE. "
        "PREREQUISITES: data/cache_unmatched/ (stand_ab.py unmatched-cache, 12 months + --ranking: UNMEASURED, expect "
        "~3-6 s and ~0.5 GB per month - two reads, the unmatched rows filtered at the Arrow level) and "
        "data/cache_stand/stratum_fold_v7_preds.parquet (scripts/stratum_fold.py, ~5 min, ~3-4 GB, ALONE). The round "
        "count of a y-target early stop is UNMEASURED. The matched pipeline is READ from the v4 record. Writes "
        "reports/lgbm_fold_allrows.{json,log} and data/cache_stand/fold_preds_allrows.parquet."),
    "queue_allrows": (
        "~3.1 h (range 2.5-4 h): the allrows estimate x1.15 for 81 columns; peak RSS ~4.9 GB. The matched pipeline is "
        "READ from the queue record; prerequisites as for allrows plus data/cache_queue/. Writes "
        "reports/lgbm_fold_queue_allrows.{json,log} and data/cache_stand/fold_preds_queue_allrows.parquet."),
    "allrows_day": (
        "~3.0 h (range 2.4-3.9 h): the allrows estimate x1.13 for 78 columns; peak RSS ~4.7 GB; prerequisites as for "
        "allrows plus data/cache_day/. Writes reports/lgbm_fold_allrows_day.{json,log} and "
        "data/cache_stand/fold_preds_allrows_day.parquet."),
    "queue_allrows_day": (
        "~3.5 h (range 2.8-4.5 h): the allrows estimate x1.3 for 90 columns; peak RSS ~5.1 GB; prerequisites as for "
        "queue_allrows plus data/cache_day/. Writes reports/lgbm_fold_queue_allrows_day.{json,log} and "
        "data/cache_stand/fold_preds_queue_allrows_day.parquet."),
}

_T0 = time.time()


class _Log:
    """print + append to the readable log, flushing both; installed into lgbm_submit as well so
    the shared fitting code writes to the same log. `suffix` is appended to EVERY line (the
    screening tier's RANKING ONLY banner)."""
    def __init__(self, path: pathlib.Path | None = None, suffix: str | None = None):
        self.path = path
        self.suffix = suffix
        self.fh = open(path, "a") if path else None

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.0f}s] {msg}"
        if self.suffix:
            line = f"{line}   {self.suffix}"
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
# arguments, modes, output paths
# =============================================================================================

def sweep_grid() -> list:
    """The 18 registered settings, (num_leaves, min_data_in_leaf, feature_fraction), grid order."""
    return [(nl, md, ff) for nl in SWEEP_AXES["num_leaves"] for md in SWEEP_AXES["min_data_in_leaf"]
            for ff in SWEEP_AXES["feature_fraction"]]


def setting_label(setting) -> str:
    nl, md, ff = setting
    return f"{int(nl)},{int(md)},{float(ff):g}"


def parse_setting(text: str) -> tuple:
    """"255,40,0.8" -> (255, 40, 0.8); anything outside the registered grid is refused."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"a setting is 'num_leaves,min_data_in_leaf,feature_fraction', got {text!r}")
    try:
        setting = (int(parts[0]), int(parts[1]), float(parts[2]))
    except ValueError:
        raise argparse.ArgumentTypeError(f"a setting is 'int,int,float', got {text!r}")
    if setting not in sweep_grid():
        raise argparse.ArgumentTypeError(f"{text!r} is not in the registered grid "
                                         f"{ {k: list(v) for k, v in SWEEP_AXES.items()} }")
    return setting


def parse_weights(text: str) -> tuple:
    """"1,10" -> (1.0, 10.0): positive finite floats, in the order given, no repeats."""
    parts = [p.strip() for p in str(text).split(",")]
    if not parts or any(p == "" for p in parts):
        raise argparse.ArgumentTypeError(f"--unmatched-weight must be a comma list of weights, got {text!r}")
    try:
        ws = tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--unmatched-weight must be numbers, got {text!r}")
    if any(not np.isfinite(w) or w <= 0 for w in ws):
        raise argparse.ArgumentTypeError(f"--unmatched-weight must be positive finite numbers, got {text!r}")
    if len(set(ws)) != len(ws):
        raise argparse.ArgumentTypeError(f"--unmatched-weight repeats a weight: {text!r}")
    return ws


def setting_params(params: dict, setting) -> dict:
    """The measured P with exactly num_leaves / min_data_in_leaf / feature_fraction replaced."""
    nl, md, ff = setting
    return dict(params, num_leaves=int(nl), min_data_in_leaf=int(md), feature_fraction=float(ff))


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="months 1-3, tiny trees, 200 bootstrap draws, scratch dir, banner")
    ap.add_argument("--out-dir", default=None,
                    help="directory for the json, the log and the predictions parquet; default "
                         "reports/ (+ data/cache_stand/ for the parquet), a temp dir for --smoke")
    ap.add_argument("--pa-trees", choices=("share", "es"), default="share",
                    help="per-airport tree count: 'share' = row share of n_ref, floor 200 "
                         "(lgbm_submit.PA_TREE_RULE); 'es' = each airport's own early-stopping "
                         "run (lgbm_submit.PA_TREE_RULE_ES). Part of the tested configuration.")
    ap.add_argument("--continue-on-repro-fail", action="store_true",
                    help="keep going when A0 misses lgbm_ab's 226.24 by more than 0.5 s (default: "
                         "write what exists and stop with exit code 2)")
    arm = ap.add_argument_group("arms (at most one; none = the v4 arms)")
    arm.add_argument("--queue", action="store_true", help="Amendment 14: baseline + QUEUE_FEATS")
    arm.add_argument("--catboost", action="store_true",
                     help="Amendment 15.1: CatBoost (venv subprocess) blended with the baseline")
    arm.add_argument("--sweep", action="store_true",
                     help="Amendment 15.2 screening tier: rank the 18 settings on the Feb-May/Jun subfold")
    arm.add_argument("--confirm", nargs="+", type=parse_setting, default=None, metavar="SETTING",
                     help="Amendment 15.2 confirmation tier: full fold-A arms for these grid settings "
                          "('num_leaves,min_data_in_leaf,feature_fraction') against the incumbent")
    arm.add_argument("--fillhead", action="store_true",
                     help="Amendment 16: baseline + the schedule-fill mixture head (lgbm_submit.fit_fill_head)")
    arm.add_argument("--orderfeats", action="store_true",
                     help="Amendment 22 arm F: the current best design + ORDER_FEATS from data/cache_order/ "
                          "(composes with --queue like --dayfeats)")
    arm.add_argument("--dayfeats", action="store_true",
                     help="Amendment 19.1 arm D: the current best design + DAY_FEATS (composes with --queue, which "
                          "then names the queue treatment as the design and reads its record)")
    arm.add_argument("--ytarget", action="store_true",
                     help="Amendment 19.2 arm Y: the current best design fitted with target y, and its 0.5/0.5 blend "
                          "with the delta arm (composes with --queue like --dayfeats)")
    base = ap.add_argument_group("baseline (queue / catboost / confirm / fillhead / dayfeats / ytarget)")
    base.add_argument("--baseline", "--blend-with", dest="baseline", choices=BASELINE_ARMS, default="A3",
                      help="the v4 arm that shipped per Amendment 12.3 (default A3); under --queue the queue "
                           "treatment built on that arm. --blend-with is the same knob: 19.2's blend partner IS the "
                           "delta arm the Y arm is paired against")
    base.add_argument("--seeds", type=L.parse_seeds, default="0,1,2",
                      help="the treatment's seeds; the baseline is rebuilt over the same seeds (default 0,1,2)")
    base.add_argument("--baseline-preds", default=None,
                      help=f"the v4 predictions parquet (default {V4_PREDS})")
    base.add_argument("--v4-json", default=None, help=f"the v4 record (default {V4_JSON})")
    base.add_argument("--refit-baseline", action="store_true",
                      help="fit the baseline in-process instead of reading the v4 record (a smoke "
                           "without --baseline-preds/--v4-json does this automatically)")
    base.add_argument("--queue-preds", default=None,
                      help=f"under --queue with --dayfeats/--ytarget: the queue fold's predictions parquet (default {QUEUE_PREDS})")
    base.add_argument("--queue-json", default=None,
                      help=f"under --queue with --dayfeats/--ytarget: the queue fold's record (default {QUEUE_JSON})")
    uni = ap.add_argument_group("the unified all-rows arm (--ytarget --all-rows)")
    uni.add_argument("--all-rows", action="store_true",
                     help="with --ytarget: ONE y-target regressor on matched AND unmatched training rows "
                          "(data/cache_unmatched), paired against the current pipeline three ways; --queue and "
                          "--dayfeats then name the design (FEATS + blocks)")
    uni.add_argument("--unmatched-weight", type=parse_weights, default="1",
                     help="the sample weight(s) of the unmatched training rows, a comma list; each is its own arm "
                          "(default 1; the registered run: 1,10)")
    uni.add_argument("--stratum-preds", default=None,
                     help=f"the stratum fold's predictions parquet whose fold-A S0 is the unmatched rows' pipeline "
                          f"prediction (default {STRATUM_PREDS})")
    ap.add_argument("--n-perm", type=int, default=N_PERM,
                    help=f"queue arm: permutations in the negative control (default {N_PERM})")
    ap.add_argument("--catboost-python", default=str(CATBOOST_PYTHON),
                    help="the isolated venv interpreter that runs the CatBoost worker")
    ap.add_argument("--catboost-worker", default=str(CATBOOST_WORKER), help="the worker script")
    args = ap.parse_args(argv)
    on = [m for m, flag in _arm_flags(args) if flag]
    if args.all_rows:
        if not args.ytarget:
            ap.error("--all-rows is the unified y-target arm: it needs --ytarget")
        if args.orderfeats:
            ap.error("--all-rows does not combine with --orderfeats: the unmatched cache carries no order columns")
        if not set(on) <= {"ytarget", "queue", "day"}:
            ap.error(f"--all-rows composes only with --queue and --dayfeats as its design: {on}")
    elif len(on) > 1 and set(on) not in COMPOSITIONS:
        ap.error(f"one arm per run: {on} (--queue composes only with --dayfeats or --ytarget, Amendment 19)")
    if args.n_perm < 2:
        ap.error("--n-perm must be >= 2 (a percentile needs a distribution)")
    return args


#: the only flag pairs one run may carry: --queue as the DESIGN of an Amendment 19 arm
COMPOSITIONS = ({"queue", "day"}, {"queue", "ytarget"}, {"queue", "order"})


def _arm_flags(args) -> tuple:
    """(mode, flag) per arm, one place, so exclusivity and mode_of cannot disagree."""
    return (("queue", args.queue), ("catboost", args.catboost), ("sweep", args.sweep),
            ("confirm", bool(args.confirm)), ("fillhead", args.fillhead), ("day", args.dayfeats),
            ("order", args.orderfeats), ("ytarget", args.ytarget))


def mode_of(args) -> str:
    if getattr(args, "all_rows", False):
        return ("queue_" if args.queue else "") + "allrows" + ("_day" if args.dayfeats else "")
    if args.orderfeats:
        return "queue_order" if args.queue else "order"
    if args.dayfeats:
        return "queue_day" if args.queue else "day"
    if args.ytarget:
        return "queue_ytarget" if args.queue else "ytarget"
    for mode, flag in _arm_flags(args):
        if flag:
            return mode
    return "v4"


def output_paths(mode: str, out_dir) -> tuple:
    """(json, log, preds parquet or None). Real runs: reports/ for the json and log,
    data/cache_stand/ for the parquet; with an out_dir everything goes there."""
    if mode not in OUTPUT_NAMES:
        raise ValueError(f"unknown mode {mode!r}; one of {MODES}")
    jn, ln, pn = OUTPUT_NAMES[mode]
    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        return out_dir / jn, out_dir / ln, (out_dir / pn if pn else None)
    return REPORTS / jn, REPORTS / ln, (CACHE / pn if pn else None)


# =============================================================================================
# fold arithmetic (shared by every arm)
# =============================================================================================

def fold_masks(month, holdout=HOLDOUT, es_months=L.ES_MONTHS):
    """(holdout, training, fit, early-stop) masks. ES months come out of the TRAINING fold."""
    month = np.asarray(month)
    te = np.isin(month, holdout)
    if not te.any():
        raise ValueError(f"no holdout rows: months {holdout} are absent")
    tr, fit, es = L.split_masks(month, te, es_months)
    assert not (es & te).any() and not (fit & te).any()
    return te, tr, fit, es


def sweep_masks(month, train_months=SWEEP["train"], stop=SWEEP["stop"]):
    """The screening subfold: (stop, train, train, stop) in the (holdout, train, fit, es) shape
    the loader expects - the screen early-stops on the stop month and never refits."""
    month = np.asarray(month)
    train_months = tuple(int(m) for m in train_months)
    if int(stop) in train_months:
        raise ValueError(f"the stop month {stop} is inside the training months {train_months}")
    tr = np.isin(month, train_months)
    st = month == int(stop)
    if not tr.any():
        raise ValueError(f"no training rows: months {train_months} are absent")
    if not st.any():
        raise ValueError(f"no stopping rows: month {stop} is absent")
    assert not (tr & st).any()
    return st, tr, tr, st


def taxi_time(proxy, delta_hat, target=L.TARGET_DELTA) -> np.ndarray:
    """y_hat = max(proxy - delta_hat, 1) in float64, unrounded (lgbm_ab scored it so); the same
    floor-bind guard as lgbm_submit.recover_taxi_time. Under target "y" (Amendment 19.2) the
    second argument is the y prediction itself and y_hat = max(prediction, 1) - proxy is not
    read (lgbm_submit.raw_taxi_time)."""
    raw = L.raw_taxi_time(delta_hat, proxy, target)
    assert np.isfinite(raw).all(), f"non-finite predictions: {int((~np.isfinite(raw)).sum())}"
    bound = float((raw < 1.0).mean())
    assert bound < 0.005, f"positivity floor binds on {100 * bound:.2f}% of rows"
    return np.maximum(raw, 1.0)


def rmse(y, yhat) -> float:
    d = np.asarray(y, dtype="float64") - np.asarray(yhat, dtype="float64")
    return float(np.sqrt((d ** 2).mean()))


def build_arms(single: dict, pa, fitted, weight=L.BLEND) -> dict:
    """The five arms from the per-seed pooled predictions and the (seed-averaged) per-airport
    prediction: A0/A1a/A1b the singles, A2 their mean, A3 = blend(A2, per-airport)."""
    if sorted(single) != list(SEEDS):
        raise ValueError(f"the arms need seeds {SEEDS}, got {sorted(single)}")
    a2 = L.mean_delta([single[s] for s in SEEDS])
    return {"A0": np.asarray(single[0], dtype="float64"),
            "A1a": np.asarray(single[1], dtype="float64"),
            "A1b": np.asarray(single[2], dtype="float64"),
            "A2": a2,
            "A3": L.blend(a2, pa, fitted, weight)}


def squared_errors(arms: dict, y, proxy) -> dict:
    """Per-row squared error of the recovered taxi time, float64, per arm."""
    y = np.asarray(y, dtype="float64")
    return {name: (y - taxi_time(proxy, d)) ** 2 for name, d in arms.items()}


def arm_rmses(se: dict, ap_code, airports) -> dict:
    """{arm: {"pooled": rmse, "per_airport": {airport: rmse}}} from the squared errors."""
    ap_code = np.asarray(ap_code)
    out = {}
    for name, e in se.items():
        per = {a: float(np.sqrt(e[ap_code == i].mean())) for i, a in enumerate(airports)
               if (ap_code == i).any()}
        out[name] = {"pooled": float(np.sqrt(e.mean())), "per_airport": per}
    return out


def paired_bootstrap(se: dict, pairs, n_draws=N_BOOT, seed=BOOT_SEED) -> dict:
    """Paired row bootstrap of gain = rmse(ref) - rmse(new) for each (new, ref) pair.

    One set of row draws serves every arm and every pair (that is what makes it paired: the
    two arms are re-scored on the same resampled rows). Percentile interval, 2.5/97.5.
    Returns {"new_vs_ref": {gain_s, ci95, excludes_zero, improving, n_draws, seed}}.
    """
    arms = sorted({a for pair in pairs for a in pair})
    M = np.stack([np.asarray(se[a], dtype="float64") for a in arms])
    n = M.shape[1]
    if n < 2:
        raise ValueError("paired bootstrap needs at least two rows")
    rng = np.random.default_rng(seed)
    means = np.empty((n_draws, len(arms)), dtype="float64")
    for i in range(n_draws):
        idx = rng.integers(0, n, n)
        means[i] = M[:, idx].mean(axis=1)
    rm = np.sqrt(means)
    point = {a: float(np.sqrt(M[j].mean())) for j, a in enumerate(arms)}
    out = {}
    for new, ref in pairs:
        gains = rm[:, arms.index(ref)] - rm[:, arms.index(new)]
        lo, hi = (float(v) for v in np.percentile(gains, [2.5, 97.5]))
        gain = point[ref] - point[new]
        out[f"{new}_vs_{ref}"] = dict(gain_s=gain, ci95=[lo, hi],
                                      excludes_zero=bool(lo > 0.0 or hi < 0.0),
                                      improving=bool(gain > 0.0 and lo > 0.0),
                                      n_draws=int(n_draws), seed=int(seed))
    return out


def seed_sd(values) -> float:
    """Sample sd (ddof=1) of the single-seed RMSEs; the rule needs all three seeds."""
    values = np.asarray(list(values), dtype="float64")
    if len(values) != 3:
        raise ValueError(f"seed sd needs the three single-seed RMSEs, got {len(values)}")
    return float(np.std(values, ddof=1))


def exceeds_2x_seed_sd(gain, sd) -> bool:
    """The repo's ESTABLISHED rule (reports/MSE_LEDGER.md rule 1): gain > 2 x seed sd, strictly."""
    return bool(float(gain) > 2.0 * float(sd))


def airports_improving(rmse_new: dict, rmse_ref: dict) -> int:
    """How many airports the new arm strictly improves; a tie is not an improvement."""
    return int(sum(1 for a in rmse_ref if a in rmse_new and rmse_new[a] < rmse_ref[a]))


def score_pairs(se: dict, rmses: dict, pairs, ap_te, airports, n_boot, sd) -> tuple:
    """Pooled and per-airport paired intervals for every (new, ref) pair, with gain / seed sd,
    the 2 x seed sd rule (None when no seed sd exists) and the airports-improving count.
    Returns (pairs dict, [(code, airport)] present in the holdout)."""
    ap_te = np.asarray(ap_te)
    pooled = paired_bootstrap(se, pairs, n_boot, BOOT_SEED)
    present = [(i, a) for i, a in enumerate(airports) if (ap_te == i).any()]
    # an airport with a single row has an RMSE but no interval (a bootstrap needs two rows): the
    # unmatched strata of the unified arm have such airports; every matched fold has thousands
    per_airport = {a: (paired_bootstrap({k: v[ap_te == i] for k, v in se.items()}, pairs, n_boot, BOOT_SEED)
                       if int((ap_te == i).sum()) >= 2 else None) for i, a in present}
    out = {}
    for new, ref in pairs:
        key = f"{new}_vs_{ref}"
        p = dict(pooled[key])
        p["gain_over_seed_sd"] = float(p["gain_s"] / sd) if (sd is not None and sd > 0) else None
        p["exceeds_2x_seed_sd"] = exceeds_2x_seed_sd(p["gain_s"], sd) if sd is not None else None
        p["airports_improving"] = airports_improving(rmses[new]["per_airport"], rmses[ref]["per_airport"])
        p["n_airports"] = len(present)
        p["per_airport"] = {a: (per_airport[a][key] if per_airport[a] is not None else None) for _, a in present}
        out[key] = p
    return out, present


# ---- Amendment 14 arithmetic ----

def tail_mask(y, sp, delta) -> np.ndarray:
    """14.3's pre-defined tail: non-fill (|y - sp| > 60 s) AND |delta| > 1,200 s, both strict."""
    y, sp, delta = (np.asarray(v, dtype="float64") for v in (y, sp, delta))
    return (np.abs(y - sp) > TAIL_NONFILL_S) & (np.abs(delta) > TAIL_DELTA_S)


def reduced_trees(best_iter) -> int:
    """The control's tree count: best_iter // 4, never below one tree."""
    best_iter = int(best_iter)
    if best_iter < 1:
        raise ValueError(f"best_iter {best_iter}")
    return max(1, best_iter // 4)


def permute_within_groups(block: np.ndarray, groups, rng) -> np.ndarray:
    """Permute the ROWS of `block` in place within each group (all columns travel together)."""
    groups = np.asarray(groups)
    if len(groups) != len(block):
        raise ValueError(f"length mismatch: {len(groups)} group labels for {len(block)} rows")
    order = np.argsort(groups, kind="stable")
    bounds = np.flatnonzero(np.diff(groups[order])) + 1
    for idx in np.split(order, bounds):
        if len(idx) > 1:
            block[idx] = block[rng.permutation(idx)]
    return block


CONTROL_NOTE = (
    "Every gain subtracts the same baseline_reduced_rmse, so 'gain exceeds the 95th percentile of the permuted "
    "gains' is identically 'the treatment's reduced-capacity RMSE is below the 5th percentile of the permuted "
    "RMSEs'. The second form is the like-for-like one: the baseline is fitted on 68 columns and LightGBM's "
    "feature-sampling (feature_fraction) stream depends on the column count, so every 80-column run - permuted or "
    "not - shares an offset against it that is luck, not information; that offset cancels between the treatment "
    "and the permuted runs. The gains vs the baseline are reported for 14.4's wording; the verdict rests on the "
    "treatment-vs-permuted comparison.")


def control_summary(baseline_reduced_rmse, treatment_reduced_rmse, permuted_rmses) -> dict:
    """The negative control at reduced capacity: the treatment's gain over the baseline against
    the 95th percentile of the permuted gains - and, identically, the treatment's RMSE against
    the 5th percentile of the permuted RMSEs (the baseline cancels; see CONTROL_NOTE)."""
    rmses = np.asarray(list(permuted_rmses), dtype="float64")
    if len(rmses) < 2:
        raise ValueError(f"a percentile needs at least two permuted runs, got {len(rmses)}")
    base, treat = float(baseline_reduced_rmse), float(treatment_reduced_rmse)
    gain = base - treat
    gains = base - rmses
    p95 = float(np.percentile(gains, 95))
    p5 = float(np.percentile(rmses, 5))
    return dict(baseline_reduced_rmse=base, treatment_reduced_rmse=treat, gain_reduced_s=gain,
                permuted_rmses=rmses.tolist(), permuted_gains_s=gains.tolist(), n_perm=int(len(rmses)),
                p95_permuted_gain_s=p95, exceeds_p95=bool(gain > p95),
                p5_permuted_rmse=p5, treatment_below_p5=bool(treat < p5),
                permuted_gain_mean_s=float(gains.mean()), permuted_gain_max_s=float(gains.max()),
                n_permuted_at_or_above=int((gains >= gain).sum()), note=CONTROL_NOTE)


# ---- Amendment 15.1 arithmetic ----

def blend_curve(base_delta, cat_delta, y, proxy, weights=CURVE_WEIGHTS) -> dict:
    """{w: matched RMSE of the recovered taxi time at delta = (1 - w) base + w CatBoost}."""
    base = np.asarray(base_delta, dtype="float64")
    cat = np.asarray(cat_delta, dtype="float64")
    if base.shape != cat.shape:
        raise ValueError(f"shape mismatch: baseline {base.shape}, catboost {cat.shape}")
    return {float(w): rmse(y, taxi_time(proxy, (1.0 - w) * base + w * cat)) for w in weights}


def curve_increases_with_weight(curve: dict, weights=BLEND_WEIGHTS) -> bool:
    """True when the RMSE rises strictly over 0.3 -> 0.5 -> 0.7: CatBoost merely dilutes."""
    v = [curve[float(w)] for w in weights]
    return bool(all(b > a for a, b in zip(v, v[1:])))


def residual_correlation(y, yhat_a, yhat_b) -> float:
    """Pearson r between the two arms' residuals (15.1's named mechanism: below 1)."""
    y = np.asarray(y, dtype="float64")
    ra, rb = y - np.asarray(yhat_a, dtype="float64"), y - np.asarray(yhat_b, dtype="float64")
    return float(np.corrcoef(ra, rb)[0, 1])


def write_worker_input(path, X, target, train, fit, es, holdout) -> pathlib.Path:
    """The CatBoost worker's .npz: X float32, target float64 with the HOLDOUT targets blanked to
    NaN (they never leave the parent), four boolean masks; the masks are checked first."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be a matrix, got shape {X.shape}")
    n = X.shape[0]
    target = np.asarray(target, dtype="float64")
    train, fit, es, holdout = (np.asarray(m, dtype=bool) for m in (train, fit, es, holdout))
    for name, arr in (("target", target), ("train", train), ("fit", fit), ("es", es), ("holdout", holdout)):
        if arr.shape != (n,):
            raise ValueError(f"rows differ: X has {n:,} rows, {name} has shape {arr.shape}")
    if (train & holdout).any():
        raise ValueError("training and holdout rows overlap")
    if (fit & es).any():
        raise ValueError("fit and early-stopping rows overlap")
    if not np.array_equal(fit | es, train):
        raise ValueError("fit + early-stop rows do not make up the training rows")
    if not holdout.any() or not fit.any() or not es.any():
        raise ValueError("empty holdout, fit or early-stopping rows")
    blanked = target.copy()
    blanked[holdout] = np.nan
    if not np.isfinite(blanked[train]).all():
        raise ValueError("non-finite target on training rows")
    path = pathlib.Path(path)
    np.savez(path, X=X.astype(np.float32, copy=False), target=blanked, train=train, fit=fit, es=es,
             holdout=holdout)
    return path


def run_catboost_worker(python, worker, npz, npy, params: dict, n_expected: int, log) -> tuple:
    """Launch the worker under the venv interpreter, stream its output to `log`, check the
    exit code and the prediction shape. Returns (predictions, sidecar dict, run dict)."""
    python, worker, npz, npy = (pathlib.Path(p).expanduser() for p in (python, worker, npz, npy))
    if not python.exists():
        raise FileNotFoundError(f"CatBoost interpreter {python} is missing: the arm runs ONLY under the isolated "
                                "venv ~/.venvs/prc-catboost (never install catboost globally)")
    if not worker.exists():
        raise FileNotFoundError(f"CatBoost worker {worker} is missing")
    cmd = [str(python), "-u", str(worker), "--input", str(npz), "--output", str(npy),
           "--depth", str(int(params["depth"])), "--learning-rate", str(float(params["learning_rate"])),
           "--iterations", str(int(params["iterations"])), "--od-wait", str(int(params["od_wait"])),
           "--seed", str(int(params["seed"])), "--threads", str(int(params["threads"]))]
    log(f"catboost worker: {' '.join(cmd)}")
    t = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))
    for line in proc.stdout:
        log(f"    [catboost] {line.rstrip()}")
    rc = proc.wait()
    wall = time.time() - t
    if rc != 0:
        raise RuntimeError(f"catboost worker exited with code {rc} after {wall:.0f}s (see the log above)")
    if not npy.exists():
        raise RuntimeError(f"catboost worker exited 0 but wrote no predictions at {npy}")
    preds = np.asarray(np.load(npy), dtype="float64")
    if preds.shape != (int(n_expected),):
        raise ValueError(f"worker predictions have shape {preds.shape}, expected ({int(n_expected)},)")
    if not np.isfinite(preds).all():
        raise ValueError(f"worker predictions carry {int((~np.isfinite(preds)).sum())} non-finite values")
    sidecar_path = npy.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    log(f"catboost worker done in {wall:.0f}s: {len(preds):,} predictions; sidecar "
        f"{'read' if sidecar else 'MISSING'} ({sidecar_path.name})")
    return preds, sidecar, dict(command=cmd, returncode=int(rc), wall_s=round(wall, 1), python=str(python),
                                worker=str(worker), input=str(npz), output=str(npy),
                                input_bytes=int(npz.stat().st_size))


# ---- Amendment 15.2 arithmetic ----

def rank_settings(results) -> list:
    """Rank the screened settings by their stopping-set RMSE, ascending, ties in input order.
    Every entry carries the RANKING ONLY banner and its magnitude key says so."""
    ranked = sorted(results, key=lambda r: float(r["rmse_stop"]))
    out = []
    for i, r in enumerate(ranked, 1):
        nl, md, ff = r["setting"]
        entry = dict(rank=i, setting=setting_label(r["setting"]), num_leaves=int(nl), min_data_in_leaf=int(md),
                     feature_fraction=float(ff), best_iter=int(r["best_iter"]),
                     rmse_stop_RANKING_ONLY=float(r["rmse_stop"]))
        if "proxy_only" in r:
            entry["proxy_only_rmse_stop_RANKING_ONLY"] = float(r["proxy_only"])
        if "wall_s" in r:
            entry["wall_s"] = float(r["wall_s"])
        entry["banner"] = RANKING_BANNER
        out.append(entry)
    return out


# ---- Amendment 16 arithmetic ----

def fill_mask(y, sp) -> np.ndarray:
    """16.2's fill subset on holdout rows: |y - sp| <= lgbm_submit.FILL_TOL_S (60 s), inclusive -
    the complement of the tail rule's non-fill (tail_mask), so the two amendments agree."""
    y, sp = (np.asarray(v, dtype="float64") for v in (y, sp))
    return np.abs(y - sp) <= L.FILL_TOL_S


def reliability_deciles(p, fill, n_bins=N_DECILES) -> dict:
    """Reliability of p on the holdout: the rows sorted by p (stable, ties in row order) and cut
    into `n_bins` equal-count bins; per bin the count, the p range, the mean p and the realised
    fill rate; overall the mean p and the fill rate (a calibrated head has them equal)."""
    p, fill = np.asarray(p, dtype="float64"), np.asarray(fill, dtype=bool)
    if p.shape != fill.shape:
        raise ValueError(f"length mismatch: {p.shape} p values for {fill.shape} fill flags")
    n = int(len(p))
    if n < n_bins:
        raise ValueError(f"reliability deciles need at least ten rows ({n_bins} bins), got {n}")
    order = np.argsort(p, kind="stable")
    out = []
    for i, idx in enumerate(np.array_split(order, n_bins), 1):
        out.append(dict(decile=i, n=int(len(idx)), p_min=float(p[idx].min()), p_max=float(p[idx].max()),
                        mean_p=float(p[idx].mean()), fill_rate=float(fill[idx].mean())))
    return dict(n_rows=n, n_bins=int(n_bins), mean_p=float(p.mean()), fill_rate=float(fill.mean()), deciles=out,
                note="equal-count bins of the holdout rows sorted by p (ties in row order); mean p vs the realised "
                     "fill rate per bin; a calibrated head has them equal")


def subset_record(definition: str, mask, se: dict, n_boot: int, pair=("treatment", "baseline")) -> dict:
    """One holdout subset (fill, non-fill, LIRF's cuts): its row count and share, its share of the
    reference arm's SSE, each arm's RMSE on ITS rows only, and the paired interval of new - ref
    drawn on those rows only (None below two rows; no RMSE at all below one)."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    new, ref = pair
    total = float(se[ref].sum())
    return {"definition": definition, "n_rows": n,
            "share_of_holdout_rows": float(mask.mean()) if len(mask) else 0.0,
            f"share_of_{ref}_sse": float(se[ref][mask].sum() / total) if total > 0 else None,
            "rmse": {k: float(np.sqrt(v[mask].mean())) for k, v in se.items()} if n else None,
            "pair": (paired_bootstrap({k: v[mask] for k, v in se.items()}, [pair], n_boot, BOOT_SEED)[f"{new}_vs_{ref}"]
                     if n >= 2 else None)}


# ---- Amendment 19 arithmetic ----

def delta_bands(delta) -> dict:
    """19.0's |delta| bands as boolean masks: lt2 [0, 120), 2to10 [120, 600), 10to20 [600, 1200),
    gt20 [1200, inf) - left-closed, right-open in seconds, a partition of the rows - and over10 =
    10to20 | gt20 (|delta| >= OVER10_S), the "> 10 min bands" of 19.3."""
    abs_d = np.abs(np.asarray(delta, dtype="float64"))
    out = {name: (abs_d >= lo) & (abs_d < hi) for name, lo, hi in DELTA_BANDS}
    assert (np.stack(list(out.values())).sum(axis=0) == 1).all(), "the |delta| bands must partition the rows"
    out["over10"] = abs_d >= OVER10_S
    return out


def band_definition(name: str) -> str:
    if name == "over10":
        return f"|delta| >= {OVER10_S:.0f} s (10to20 + gt20)"
    lo, hi = next((lo, hi) for n, lo, hi in DELTA_BANDS if n == name)
    return f"|delta| >= {lo:.0f} s" if hi == float("inf") else f"{lo:.0f} s <= |delta| < {hi:.0f} s"


def blend_taxi_times(y_hat_y, y_hat_delta, weight=YBLEND) -> np.ndarray:
    """19.2's blend: weight x the y arm's taxi time + (1 - weight) x the delta arm's, float64,
    row by row; both are taxi times already floored at 1 s. Shape mismatch and non-finite
    inputs are refusals."""
    a, b = np.asarray(y_hat_y, dtype="float64"), np.asarray(y_hat_delta, dtype="float64")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: y arm {a.shape}, delta arm {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite taxi time in a blend operand")
    return float(weight) * a + (1.0 - float(weight)) * b


def band_record(definition: str, mask, se: dict, n_boot: int, pairs, ref: str) -> dict:
    """One |delta| band: its row count and share, its share of the reference arm's SSE, every
    arm's RMSE on ITS rows only, and one paired interval PER PAIR drawn on those rows only
    (None below two rows; no RMSE at all below one) - subset_record for several pairs."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    total = float(se[ref].sum())
    out = {"definition": definition, "n_rows": n,
           "share_of_holdout_rows": float(mask.mean()) if len(mask) else 0.0,
           f"share_of_{ref}_sse": float(se[ref][mask].sum() / total) if total > 0 else None,
           "rmse": {k: float(np.sqrt(v[mask].mean())) for k, v in se.items()} if n else None,
           "pairs": None}
    if n >= 2:
        out["pairs"] = paired_bootstrap({k: v[mask] for k, v in se.items()}, list(pairs), n_boot, BOOT_SEED)
    return out


def band_records(delta, se: dict, n_boot: int, pairs, ref: str) -> dict:
    """{band: band_record} over the five bands of delta_bands, cut on the TRUE delta."""
    return {name: band_record(band_definition(name), mask, se, n_boot, pairs, ref)
            for name, mask in delta_bands(delta).items()}


# ---- the unified all-rows arm's arithmetic ----

def total_rmse_2026(mse_matched, mse_unmatched, w_m=W_MATCHED_2026, w_u=W_UNMATCHED_2026) -> float:
    """The fold TOTAL at the 2026 scored-file shares: sqrt(w_m x MSE_matched + w_u x MSE_unmatched)."""
    return float(np.sqrt(w_m * float(mse_matched) + w_u * float(mse_unmatched)))


def stratified_paired_bootstrap(se_m: dict, se_u: dict, pairs, n_draws=N_BOOT, seed=BOOT_SEED, w_m=W_MATCHED_2026,
                                w_u=W_UNMATCHED_2026) -> dict:
    """Paired row bootstrap of the 2026-weighted TOTAL: the matched rows and the unmatched rows
    are each resampled within their own stratum (the same draws for every arm and pair), the
    total per draw is total_rmse_2026 of the two resampled means, the point estimate the actual
    total gain. Returns {"new_vs_ref": {gain_s, ci95, excludes_zero, improving, n_draws, seed,
    n_matched, n_unmatched, w_matched, w_unmatched}}."""
    arms = sorted({a for pair in pairs for a in pair})
    M = np.stack([np.asarray(se_m[a], dtype="float64") for a in arms])
    U = np.stack([np.asarray(se_u[a], dtype="float64") for a in arms])
    n_m, n_u = M.shape[1], U.shape[1]
    if n_m < 2 or n_u < 2:
        raise ValueError(f"each stratum needs at least two rows: matched {n_m}, unmatched {n_u}")
    rng = np.random.default_rng(seed)
    tot = np.empty((n_draws, len(arms)), dtype="float64")
    for i in range(n_draws):
        im, iu = rng.integers(0, n_m, n_m), rng.integers(0, n_u, n_u)
        tot[i] = np.sqrt(w_m * M[:, im].mean(axis=1) + w_u * U[:, iu].mean(axis=1))
    point = {a: total_rmse_2026(M[j].mean(), U[j].mean(), w_m, w_u) for j, a in enumerate(arms)}
    out = {}
    for new, ref in pairs:
        gains = tot[:, arms.index(ref)] - tot[:, arms.index(new)]
        lo, hi = (float(v) for v in np.percentile(gains, [2.5, 97.5]))
        gain = point[ref] - point[new]
        out[f"{new}_vs_{ref}"] = dict(gain_s=gain, ci95=[lo, hi], excludes_zero=bool(lo > 0.0 or hi < 0.0),
                                      improving=bool(gain > 0.0 and lo > 0.0), n_draws=int(n_draws), seed=int(seed),
                                      n_matched=int(n_m), n_unmatched=int(n_u), w_matched=float(w_m), w_unmatched=float(w_u))
    return out


def unmatched_baseline_from_stratum(preds: pd.DataFrame, base_u: pd.DataFrame) -> dict:
    """The unmatched holdout rows' pipeline prediction: the stratum record's fold-"A" S0 (the
    shipped fit_unmatched), joined by MVT_ID_mvt onto `base_u` in ITS order. Every row must be
    present exactly once among the fold-A rows and y / month must agree row by row (the same
    fold); lomo rows are ignored. Returns dict(yhat, n_record_fold_a, n_joined)."""
    if "S0" not in preds.columns:
        raise ValueError("the stratum record has no S0 column (the shipped fit_unmatched): not stratum_fold's parquet")
    fa = preds[preds.fold == "A"]
    if not pd.Index(fa.MVT_ID_mvt).is_unique:
        raise ValueError(f"duplicate MVT_ID among the record's fold-A rows: {int(fa.MVT_ID_mvt.duplicated().sum())}")
    ids = base_u.MVT_ID_mvt.to_numpy(dtype="float64")
    missing = set(ids) - set(fa.MVT_ID_mvt.to_numpy(dtype="float64"))
    if missing:
        raise ValueError(f"{len(missing):,} unmatched holdout rows are missing from the record's fold-A rows")
    rec = fa.set_index("MVT_ID_mvt").loc[ids]
    if not (np.array_equal(rec.y.to_numpy(dtype="float64"), base_u.y.to_numpy(dtype="float64"))
            and np.array_equal(rec.month.to_numpy().astype("int64"), base_u.month.to_numpy().astype("int64"))):
        raise ValueError("the record's fold-A rows are not the same fold as the unmatched holdout rows: y or month differs")
    return dict(yhat=rec.S0.to_numpy(dtype="float64"), n_record_fold_a=int(len(fa)), n_joined=int(len(ids)))


def _pad_matched(v, um_te) -> np.ndarray:
    """A per-row column of the MATCHED holdout rows widened to every holdout row: NaN (False for
    a boolean column) on the unmatched rows. A full-length column passes through."""
    v = np.asarray(v)
    if len(v) == len(um_te):
        return v
    if len(v) != int((~um_te).sum()):
        raise ValueError(f"a column of {len(v):,} rows is neither the matched ({int((~um_te).sum()):,}) nor the "
                         f"full ({len(um_te):,}) holdout")
    out = np.zeros(len(um_te), dtype=bool) if v.dtype == bool else np.full(len(um_te), np.nan)
    out[~um_te] = v
    return out


# =============================================================================================
# the baseline: the v4 record, or an in-process refit
# =============================================================================================

def baseline_from_preds(preds: pd.DataFrame, base: pd.DataFrame, arm: str, seeds, arm_col=None) -> dict:
    """Rebuild the v4 arm from the v4 predictions parquet on the IDENTICAL fold.

    A2 = mean over `seeds` of delta_hat_seed{s}; A3 = A2 blended with the mean of pa_seed{s} on
    pa_fitted rows. The fold columns (row, month, ap, y, proxy, delta) must match `base`
    exactly; when the seed set is the full v4 one the parquet's own stored arm column must
    agree with the rebuild - `arm_col` names it (the arm itself in a v4 record; "treatment" in
    the queue fold's record, whose delta_hat_seed{s} / pa_seed{s} ARE its treatment's seeds).
    Returns dict(delta, single, pa_by_seed, pa, fitted).
    """
    if arm not in BASELINE_ARMS:
        raise ValueError(f"baseline arm must be one of {BASELINE_ARMS}, got {arm!r}")
    arm_col = arm if arm_col is None else str(arm_col)
    need = ["row", "month", "ap", "y", "proxy", "delta"]
    for c in need:
        if c not in preds.columns:
            raise ValueError(f"the v4 predictions lack the fold column {c!r}")
    if len(preds) != len(base):
        raise ValueError(f"the v4 predictions are a different fold: {len(preds):,} rows vs {len(base):,} here")
    for c in need:
        a, b = preds[c].to_numpy(), base[c].to_numpy()
        if c == "ap":
            same = np.array_equal(a.astype(str), b.astype(str))
        else:
            same = np.array_equal(np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64"))
        if not same:
            raise ValueError(f"the v4 predictions are a different fold: column {c!r} differs")
    single = {}
    for s in seeds:
        col = f"delta_hat_seed{s}"
        if col not in preds.columns:
            raise ValueError(f"the v4 run did not fit seed {s} ({col} missing); pass --seeds within its set "
                             "or --refit-baseline")
        single[s] = preds[col].to_numpy(dtype="float64")
    pooled = L.mean_delta([single[s] for s in seeds])
    out = dict(delta=pooled, single=single, pa_by_seed=None, pa=None, fitted=None)
    if arm == "A3":
        for c in [f"pa_seed{s}" for s in seeds] + ["pa_fitted"]:
            if c not in preds.columns:
                raise ValueError(f"the v4 predictions lack {c!r}: no per-airport record for A3")
        pa_by_seed = {s: preds[f"pa_seed{s}"].to_numpy(dtype="float64") for s in seeds}
        pa = L.mean_delta([pa_by_seed[s] for s in seeds])
        fitted = preds["pa_fitted"].to_numpy(dtype=bool)
        out.update(delta=L.blend(pooled, pa, fitted), pa_by_seed=pa_by_seed, pa=pa, fitted=fitted)
    if arm_col in preds.columns and sorted(seeds) == sorted(SEEDS):
        if not np.allclose(out["delta"], preds[arm_col].to_numpy(dtype="float64"), atol=1e-9, rtol=0):
            raise ValueError(f"the record parquet's own {arm_col} column disagrees with its rebuild from the "
                             "per-seed columns: the record is inconsistent")
    return out


def check_baseline_config(v4: dict, arm: str, seeds, pa_trees: str, smoke: bool) -> None:
    """The v4 record may serve as the baseline only for a treatment measured the same way."""
    if bool(v4.get("smoke")) != bool(smoke):
        raise ValueError(f"the v4 record is smoke={v4.get('smoke')} but this run is smoke={smoke}")
    have = set(int(s) for s in v4.get("config", {}).get("seeds", []))
    if not set(int(s) for s in seeds) <= have:
        raise ValueError(f"the v4 record has seeds {sorted(have)}; this run asks for {list(seeds)}: a seed the "
                         "baseline never fitted")
    if arm == "A3" and v4.get("config", {}).get("pa_trees") != pa_trees:
        raise ValueError(f"per-airport tree rule differs: the v4 record used --pa-trees "
                         f"{v4.get('config', {}).get('pa_trees')!r}, this run {pa_trees!r}; the treatment must "
                         "use the same procedure as the baseline (Amendment 14.2)")
    status = (v4.get("a0_reproduction") or {}).get("status")
    if status == "FAIL":
        raise ValueError("the v4 record's A0 did not reproduce lgbm_ab (a0_reproduction FAIL): its predictions "
                         "carry that caveat and are refused as a baseline; pass --refit-baseline to measure "
                         "both arms in one process, knowing the harness may be wrong")


def check_queue_record(qj: dict, arm: str, seeds, pa_trees: str, smoke: bool) -> None:
    """The queue fold's record may serve as the --queue design baseline only when it IS that
    design: a queue-mode record whose baseline_arm is the arm asked for (its treatment is that
    arm + QUEUE_FEATS), whose design is FEATS + the twelve QUEUE_FEATS at 80 columns, and which
    passes the v4 record's own checks (smoke flag, seeds, the per-airport rule for A3)."""
    if qj.get("mode") != "queue":
        raise ValueError(f"not a queue record (mode {qj.get('mode')!r}): under --queue the design baseline is the "
                         "Amendment 14 fold's own record")
    cfg = qj.get("config") or {}
    if cfg.get("baseline_arm") != arm:
        raise ValueError(f"the queue record's baseline_arm is {cfg.get('baseline_arm')!r}, this run asks for {arm!r}: "
                         f"its treatment is {cfg.get('baseline_arm')} + QUEUE_FEATS, not {arm} + QUEUE_FEATS")
    feats = list(cfg.get("features") or [])
    if len(feats) != len(FEATS_QUEUE) or feats[-len(QUEUE_FEATS):] != QUEUE_FEATS:
        raise ValueError(f"the queue record's design ({len(feats)} features) is not FEATS + the {len(QUEUE_FEATS)} "
                         f"QUEUE_FEATS ({len(FEATS_QUEUE)} columns)")
    check_baseline_config(qj, arm, seeds, pa_trees, smoke)


def _reads_record(args, cfg, queue: bool = False) -> bool:
    """Read the baseline from its record unless --refit-baseline, or a smoke that names no record."""
    if args.refit_baseline:
        return False
    given = (args.queue_preds, args.queue_json) if queue else (args.baseline_preds, args.v4_json)
    return not (cfg.smoke and all(p is None for p in given))


def _reads_v4(args, cfg) -> bool:
    return _reads_record(args, cfg, queue=False)


def _queue_record(args, cfg, arm: str, seeds, pa_trees: str) -> dict:
    """Locate and validate the queue record BEFORE any data is loaded (fail fast)."""
    preds_path = pathlib.Path(args.queue_preds) if args.queue_preds else QUEUE_PREDS
    json_path = pathlib.Path(args.queue_json) if args.queue_json else QUEUE_JSON
    for p in (preds_path, json_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run the queue fold first (Amendment 14), or pass --refit-baseline "
                                    f"to fit the {arm} + QUEUE_FEATS baseline in-process")
    qj = json.loads(json_path.read_text())
    check_queue_record(qj, arm, seeds, pa_trees, smoke=cfg.smoke)
    tr = qj.get("treatment") or {}
    log(f"baseline: the queue record {json_path} (sha {qj.get('git_sha')}, treatment best_iter "
        f"{tr.get('best_iter')}, n_ref {tr.get('n_ref')}, baseline_arm {qj['config'].get('baseline_arm')!r}, seeds "
        f"{qj['config']['seeds']}, pa_trees {qj['config'].get('pa_trees')!r}); predictions {preds_path}")
    return dict(json=qj, preds_path=preds_path, json_path=json_path, arm_col="treatment", kind="queue")


def _v4_record(args, cfg, arm: str, seeds, pa_trees: str) -> dict:
    """Locate and validate the v4 record BEFORE any data is loaded (fail fast)."""
    preds_path = pathlib.Path(args.baseline_preds) if args.baseline_preds else V4_PREDS
    json_path = pathlib.Path(args.v4_json) if args.v4_json else V4_JSON
    for p in (preds_path, json_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run the v4 fold first, or pass --refit-baseline to fit the "
                                    f"{arm} baseline in-process")
    v4 = json.loads(json_path.read_text())
    check_baseline_config(v4, arm, seeds, pa_trees, smoke=cfg.smoke)
    log(f"baseline: the v4 record {json_path} (sha {v4.get('git_sha')}, best_iter {v4['config']['best_iter']:,}, "
        f"n_ref {v4['config']['n_ref']:,}, seeds {v4['config']['seeds']}, pa_trees {v4['config']['pa_trees']!r}, "
        f"A0 reproduction {(v4.get('a0_reproduction') or {}).get('status')!r}); predictions {preds_path}")
    return dict(json=v4, preds_path=preds_path, json_path=json_path, arm_col=None, kind="v4")


def fit_arm(fold: dict, X, params, nest, patience, pa_floor, seeds, per_airport: bool, pa_trees, name: str,
            on_seed=None, target=L.TARGET_DELTA, weight=None, es_eval=None) -> dict:
    """One fold-A arm on the design matrix X: early stopping (seed ES_SEED) -> n_ref; a refit per
    seed predicting the holdout rows; the mean over seeds; optionally the per-airport blend.
    `on_seed(seed, prediction)` runs after every seed (the callers write the parquet there).
    `target` is lgbm_submit's formulation: delta (the label is delta, the prediction a delta_hat)
    or y (Amendment 19.2: the label is y, the prediction a y_hat before the 1 s floor); the
    returned "delta" is the pooled prediction on that scale. `es_eval` (Amendment 20.5) is the
    subset of the stopping rows the early-stopping metric is evaluated on - the all-rows arm
    passes its matched stopping rows; None (every other arm) evaluates on all of them."""
    te, tr, fit, es = fold["te"], fold["tr"], fold["fit"], fold["es"]
    y, dlt, proxy = fold["y"], fold["dlt"], fold["proxy"]
    label = L.regression_label(dlt, y, target)
    X_te = X[te]
    log(f"[{name}] early stopping on {X.shape[1]} features, target {target}")
    info = L.early_stop(X, dlt, y, proxy, tr, fit, es, dict(params, seed=L.ES_SEED), nest, patience, target=target,
                        weight=weight, es_eval=es_eval)
    n_ref = int(info["n_ref"])
    single, single_rmse, timing = {}, {}, {}
    for s in seeds:
        t = time.time()
        single.update(L.fit_seeds(X, label, tr, X_te, params, n_ref, (s,), n_features=X.shape[1], target=target,
                                  weight=weight))
        timing[f"seed{s}_s"] = round(time.time() - t, 1)
        single_rmse[s] = rmse(y[te], taxi_time(proxy[te], single[s], target))
        log(f"[{name}] seed {s} alone: matched RMSE {single_rmse[s]:.4f} on {te.sum():,} holdout rows   "
            f"[{timing[f'seed{s}_s']:.0f}s]")
        if on_seed is not None:
            on_seed(s, single[s])
    pooled = L.mean_delta([single[s] for s in seeds])
    out = dict(info=info, best_iter=int(info["best_iter"]), n_ref=n_ref, single=single, single_rmse=single_rmse,
               pooled=pooled, delta=pooled, pa_by_seed=None, fitted=None, pa_info=None, timing=timing,
               n_features=int(X.shape[1]), seeds=list(seeds), per_airport=bool(per_airport), target=target)
    if per_airport:
        t = time.time()
        pa_by_seed, fitted, pa_info = L.fit_per_airport(
            X, label, tr, te, fold["ap_code"], fold["airports"], params, n_ref, seeds, min_rows=L.PA_MIN_ROWS,
            floor=pa_floor, trees=pa_trees, es=dict(y=y, proxy=proxy, dlt=dlt, fit=fit, es=es, nest=nest,
                                                     patience=patience), target=target)
        timing["per_airport_s"] = round(time.time() - t, 1)
        out.update(pa_by_seed=pa_by_seed, fitted=fitted, pa_info=pa_info,
                   delta=L.blend(pooled, L.mean_delta([pa_by_seed[s] for s in seeds]), fitted))
        n_fitted = sum(1 for r in pa_info.values() if r["fitted"])
        log(f"[{name}] per-airport: {n_fitted}/{len(fold['airports'])} airports fitted, {int(fitted.sum()):,} "
            f"holdout rows blended, {int((~fitted).sum()):,} keep the pooled mean   [{timing['per_airport_s']:.0f}s]")
    del X_te
    gc.collect()
    return out


def resolve_baseline(args, cfg, fold: dict, X_base, arm: str, seeds, pa_trees, v4, name: str, cols: dict,
                     write) -> tuple:
    """The baseline arm's predictions, written into `cols` as {name}_seed{s} [+ {name}_pa_*] + {name}.
    Returns (arm dict, record for the json, seed sd or None, seed sd source or None)."""
    if v4 is not None:
        kind = v4.get("kind", "v4")                  # "v4": the v4 arm; "queue": the queue fold's treatment
        b = baseline_from_preds(pd.read_parquet(v4["preds_path"]), fold["base"], arm, seeds, arm_col=v4.get("arm_col"))
        sd, source = float(v4["json"]["seed_sd"]["sd"]), f"{kind} json"
        cfg_j, tr_j = v4["json"].get("config") or {}, v4["json"].get("treatment") or {}
        record = dict(source=f"{kind} predictions", preds=str(v4["preds_path"]), json=str(v4["json_path"]), arm=arm,
                      seeds=list(seeds), record_kind=kind, record_git_sha=v4["json"].get("git_sha"),
                      record_best_iter=cfg_j.get("best_iter", tr_j.get("best_iter")),
                      record_n_ref=cfg_j.get("n_ref", tr_j.get("n_ref")), record_pa_trees=cfg_j.get("pa_trees"),
                      record_a0_reproduction=v4["json"].get("a0_reproduction"),
                      design=("FEATS + QUEUE_FEATS" if kind == "queue" else "FEATS"))
        if kind == "v4":
            record.update(v4_git_sha=record["record_git_sha"], v4_best_iter=record["record_best_iter"],
                          v4_n_ref=record["record_n_ref"], v4_pa_trees=record["record_pa_trees"],
                          v4_a0_reproduction=record["record_a0_reproduction"])
        log(f"[{name}] {arm}{' + QUEUE_FEATS' if kind == 'queue' else ''} rebuilt from the {kind} predictions over "
            f"seeds {list(seeds)} on the identical fold ({len(fold['base']):,} rows); seed sd {sd:.4f} s from the "
            f"{kind} json")
    else:
        b = fit_arm(fold, X_base, cfg.params, cfg.nest, cfg.patience, cfg.pa_floor, seeds, per_airport=(arm == "A3"),
                    pa_trees=pa_trees, name=name)
        sd = seed_sd(b["single_rmse"].values()) if len(seeds) == 3 else None
        source = f"{name} single seeds (in-process)" if sd is not None else None
        record = dict(source="in-process", arm=arm, seeds=list(seeds), best_iter=b["best_iter"], n_ref=b["n_ref"],
                      single_seed_rmses={str(s): r for s, r in b["single_rmse"].items()},
                      per_airport_models=b["pa_info"], timing_s=b["timing"],
                      es=dict(es_rmse_taxi_time_NOT_A_RESULT=b["info"]["es_rmse_taxi_time_NOT_A_RESULT"],
                              es_proxy_only_rmse=b["info"]["es_proxy_only_rmse"]))
        log(f"[{name}] {arm} fitted in-process over seeds {list(seeds)}"
            + (f"; seed sd {sd:.4f} s from its single seeds" if sd is not None else "; no seed sd (needs 3 seeds)"))
    for s in seeds:
        cols[f"{name}_seed{s}"] = b["single"][s]
    if arm == "A3":
        for s in seeds:
            cols[f"{name}_pa_seed{s}"] = b["pa_by_seed"][s]
        cols[f"{name}_pa_fitted"] = b["fitted"]
    cols[name] = b["delta"]
    write()
    return b, record, sd, source


# =============================================================================================
# loading one fold
# =============================================================================================

def _load_months(cache, months, feats, qcache=None, dcache=None, ocache=None) -> tuple:
    """The training month caches with the queue / day / order blocks among `feats` joined
    positionally (the row contracts of tests/test_queue_features.py, tests/test_day_features.py
    and tests/test_order_features.py); returns (frames, paths)."""
    frames, paths = L.training_frames(cache, months)
    queue = [c for c in feats if c in QUEUE_FEATS]
    if queue:
        if qcache is None:
            raise ValueError("queue features asked for without a queue cache directory")
        for f, p in zip(frames, paths):
            qf = L.attach_queue_positional(len(f), L.read_queue_cache(L.queue_cache_path(qcache, p)), p.name)
            for c in QUEUE_FEATS:
                f[c] = qf[c].to_numpy()
            del qf
        log(f"queue block: {len(QUEUE_FEATS)} columns joined POSITIONALLY onto {len(paths)} months from {qcache} "
            f"(row counts asserted per month; the order is the v6 builder's, proven on March both ways)")
    order = [c for c in feats if c in ORDER_FEATS]
    day = [c for c in feats if c in DAY_FEATS]
    if day:
        if dcache is None:
            raise ValueError("day features asked for without a day cache directory")
        for f, p in zip(frames, paths):
            df_ = L.attach_day_positional(len(f), L.read_day_cache(L.day_cache_path(dcache, p)), p.name)
            for c in DAY_FEATS:
                f[c] = df_[c].to_numpy()
            del df_
        log(f"day block: {len(DAY_FEATS)} columns joined POSITIONALLY onto {len(paths)} months from {dcache} "
            f"(row counts asserted per month; the order is build_features's, tests/test_day_features.py)")
    if order:
        if ocache is None:
            raise ValueError("order features asked for without an order cache directory")
        for f, p in zip(frames, paths):
            of = L.attach_order_positional(len(f), L.read_order_cache(L.order_cache_path(ocache, p)), p.name)
            for c in ORDER_FEATS:
                f[c] = of[c].to_numpy()
            del of
        log(f"order block: {len(ORDER_FEATS)} columns joined POSITIONALLY onto {len(paths)} months from {ocache} "
            f"(row counts asserted per month; the order is build_features's, tests/test_order_features.py)")
    return frames, paths


def load_fold(cache, months, feats, qcache=None, split=None, check_counts=False, label="holdout",
              derive=None, dcache=None, ocache=None) -> dict:
    """Rows, masks, in-fold encodings and the float32 design matrix for one fold.

    `split(month)` -> (holdout, train, fit, es); fold_masks by default. Queue columns among
    `feats` are joined positionally from `qcache` per month before the concat, and day columns
    from `dcache` after them (Amendment 19.1; the same positional discipline, the row contract
    of tests/test_day_features.py). `derive(frame)`, when given, adds derived columns to the
    frame after the in-fold encodings and before the design matrix (the fill head's nmdelay =
    proxy - sp: lgbm_submit.add_nmdelay).
    """
    frames, paths = _load_months(cache, months, feats, qcache, dcache, ocache)
    d = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    month = d.month.to_numpy()
    te, tr, fit, es = (split or fold_masks)(month)
    y, dlt, proxy, sp = d.y.to_numpy(), d.delta.to_numpy(), d.proxy.to_numpy(), d.sp.to_numpy()
    ap_code, airports = L.airport_codes(d.ap)
    log(f"loaded {len(paths)} months ({len(d):,} rows): {label} {te.sum():,}; training {tr.sum():,}; "
        f"fit {fit.sum():,}; early-stop {es.sum():,}; airports {airports}; peak RSS {L.peak_rss_gb():.2f} GB")
    if check_counts:
        for k, want in (("n_test", int(te.sum())), ("n_fit", int(fit.sum())), ("n_es", int(es.sum())),
                        ("n_all", int(tr.sum()))):
            if LGBM_AB[k] != want:
                raise RuntimeError(f"fold row count {k}={want:,} differs from lgbm_ab's {LGBM_AB[k]:,}: "
                                   "not the same fold, A0 cannot reproduce")
    for col, vals in S.infold_encodings(d, y, dlt, tr).items():
        d[col] = vals
    d = d.drop(columns=[c for c in set(S.ENC_KEYS) | {"ars", "ap"} if c in d.columns])
    gc.collect()
    if derive is not None:
        derive(d)
    X = L.design_matrix(d, feats)
    del d
    gc.collect()
    log(f"design matrix {X.shape[0]:,} x {X.shape[1]} float32 ({X.nbytes / 1e9:.2f} GB); peak RSS "
        f"{L.peak_rss_gb():.2f} GB")
    te_idx = np.flatnonzero(te)
    base = pd.DataFrame({"row": te_idx.astype("int64"), "month": month[te].astype("int32"),
                         "ap": np.asarray(airports, dtype=object)[ap_code[te]].astype(str),
                         "y": y[te], "proxy": proxy[te], "delta": dlt[te], "sp": sp[te]})
    return dict(X=X, y=y, dlt=dlt, proxy=proxy, sp=sp, month=month, te=te, tr=tr, fit=fit, es=es,
                ap_code=ap_code, airports=airports, feats=list(feats), base=base, n_months=len(paths))


def load_fold_all_rows(cache, ucache, months, feats, qcache=None, dcache=None, check_counts=False,
                       label="holdout") -> dict:
    """The unified fold: the matched months (with the blocks among `feats`) followed by the
    unmatched months from `ucache` - aligned by name (the matched caches' STAND_BLOCK columns
    NaN on unmatched rows, the unmatched cache's own queue / day columns used where asked for),
    every file's NaN pattern checked - with is_unmatched appended as the LAST design column and
    the delta encodings fitted on the matched training rows only. `row` numbers the matched
    rows exactly as load_fold does (so the v4 / queue record pairs on them) and the unmatched
    rows after them; `base` carries is_unmatched and MVT_ID_mvt (NaN on matched rows: the stand
    caches predate it). With check_counts the MATCHED subset must be lgbm_ab's fold."""
    if not feats or feats[-1] != IS_UNMATCHED:
        raise ValueError(f"the unified design ends with {IS_UNMATCHED!r}")
    frames, paths = _load_months(cache, months, feats[:-1], qcache, dcache)
    cols = list(frames[0].columns)
    n_m = sum(len(f) for f in frames)
    u_frames = []
    for p in paths:
        uf = L.read_unmatched_cache(L.unmatched_cache_path(ucache, p))
        L.check_unmatched_nan_pattern(uf, p.name)
        u_frames.append(uf)
    d_u = pd.concat(u_frames, ignore_index=True)
    mvt_id = np.r_[np.full(n_m, np.nan), d_u.MVT_ID_mvt.to_numpy(dtype="float64")]
    d = pd.concat(frames + [d_u.reindex(columns=cols)], ignore_index=True)
    del frames, u_frames, d_u
    gc.collect()
    is_um = np.zeros(len(d), dtype=bool)
    is_um[n_m:] = True
    d[IS_UNMATCHED] = is_um.astype("float64")
    log(f"unmatched rows: {int(is_um.sum()):,} appended after the {n_m:,} matched rows from {ucache} ({len(paths)} "
        f"files; the NaN pattern - {len(S.UNMATCHED_NAN_COLS)} AOBT_3-anchored + {len(S.UNMATCHED_FLT_COLS)} *_flt-derived "
        f"columns - checked on each); {IS_UNMATCHED} appended as the last design column")
    month = d.month.to_numpy()
    te, tr, fit, es = fold_masks(month)
    y, dlt, proxy, sp = d.y.to_numpy(), d.delta.to_numpy(), d.proxy.to_numpy(), d.sp.to_numpy()
    ap_code, airports = L.airport_codes(d.ap)
    log(f"loaded {len(paths)} months ({len(d):,} rows): {label} {te.sum():,} ({int((te & is_um).sum()):,} unmatched); "
        f"training {tr.sum():,} ({int((tr & is_um).sum()):,} unmatched); fit {fit.sum():,}; early-stop {es.sum():,}; "
        f"airports {airports}; peak RSS {L.peak_rss_gb():.2f} GB")
    if check_counts:
        mm = ~is_um
        for k, want in (("n_test", int((te & mm).sum())), ("n_fit", int((fit & mm).sum())), ("n_es", int((es & mm).sum())),
                        ("n_all", int((tr & mm).sum()))):
            if LGBM_AB[k] != want:
                raise RuntimeError(f"matched fold row count {k}={want:,} differs from lgbm_ab's {LGBM_AB[k]:,}: "
                                   "not the same fold, the record cannot pair")
    for col, vals in S.infold_encodings(d, y, dlt, tr, delta_mask=~is_um).items():
        d[col] = vals
    d = d.drop(columns=[c for c in set(S.ENC_KEYS) | {"ars", "ap"} if c in d.columns])
    gc.collect()
    X = L.design_matrix(d, feats)
    del d
    gc.collect()
    log(f"design matrix {X.shape[0]:,} x {X.shape[1]} float32 ({X.nbytes / 1e9:.2f} GB); peak RSS {L.peak_rss_gb():.2f} GB")
    te_idx = np.flatnonzero(te)
    base = pd.DataFrame({"row": te_idx.astype("int64"), "month": month[te].astype("int32"),
                         "ap": np.asarray(airports, dtype=object)[ap_code[te]].astype(str),
                         "y": y[te], "proxy": proxy[te], "delta": dlt[te], "sp": sp[te],
                         IS_UNMATCHED: is_um[te].astype("float64"), "MVT_ID_mvt": mvt_id[te]})
    return dict(X=X, y=y, dlt=dlt, proxy=proxy, sp=sp, month=month, te=te, tr=tr, fit=fit, es=es,
                ap_code=ap_code, airports=airports, feats=list(feats), base=base, n_months=len(paths), is_unmatched=is_um)


# =============================================================================================
# records and tables
# =============================================================================================

def _git() -> dict:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"git_sha": run("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


def _current_rss_gb() -> float:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(out.strip()) / (1024 ** 2)
    except Exception:                                   # ps missing or refused: the peak is still logged
        return float("nan")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _write_json(path: pathlib.Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))


def _write_preds(path: pathlib.Path, base: pd.DataFrame, columns: dict) -> None:
    frame = base.copy()
    for k, v in columns.items():
        frame[k] = v
    frame.to_parquet(path, index=False)


def _record_head(mode: str, cfg, started, fold: dict | None, extra: dict | None = None) -> dict:
    finished = dt.datetime.now(dt.timezone.utc)
    head = {"mode": mode, "smoke": cfg.smoke, **_git(), "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(), "wall_s": round((finished - started).total_seconds(), 1),
            "peak_rss_gb": round(L.peak_rss_gb(), 2), "command": COMMANDS[mode], "estimate": ESTIMATES[mode]}
    if fold is not None:
        head["fold"] = {"holdout_months": list(HOLDOUT), "es_months": list(L.ES_MONTHS),
                        "months": sorted(cfg.months) if cfg.months else "all twelve", "n_test": int(fold["te"].sum()),
                        "n_fit": int(fold["fit"].sum()), "n_es": int(fold["es"].sum()), "n_all": int(fold["tr"].sum()),
                        "airports": fold["airports"]}
    if extra:
        head.update(extra)
    return head


def _arm_records(arms: dict, rmses: dict, definitions: dict, proxy_te) -> dict:
    return {name: {"definition": definitions.get(name, name), "rmse": rmses[name]["pooled"],
                   "per_airport": rmses[name]["per_airport"],
                   "floor_bind_rate": float(((proxy_te - v) < 1.0).mean())} for name, v in arms.items()}


def _log_tables(arms: dict, rmses: dict, pairs: dict, present, ap_te, definitions: dict, sd, title: str) -> None:
    w = max(10, max(len(n) for n in arms) + 2)
    pw = max(14, max(len(k) for k in pairs) + 2) if pairs else 14
    log("=" * 100)
    log(title)
    log("=" * 100)
    log(f"{'arm':{w}s}{'matched RMSE':>14s}   definition")
    for name in arms:
        log(f"{name:{w}s}{rmses[name]['pooled']:14.4f}   {definitions.get(name, '')}")
    if sd is not None:
        log(f"seed sd: {sd:.4f} s (sample sd, ddof=1)   2 x seed sd = {2 * sd:.4f} s")
    else:
        log("seed sd: none (fewer than three single seeds); the 2 x seed sd rule is not applied")
    log(f"{'pair':{pw}s}{'gain s':>9s}{'95% paired CI':>22s}{'excl 0':>8s}{'gain/sd':>9s}{'>2x sd':>8s}"
        f"{'airports improving':>20s}")
    for key, p in pairs.items():
        lo, hi = p["ci95"]
        gs = f"{p['gain_over_seed_sd']:.2f}" if p["gain_over_seed_sd"] is not None else "n/a"
        ex = str(p["exceeds_2x_seed_sd"]) if p["exceeds_2x_seed_sd"] is not None else "n/a"
        log(f"{key:{pw}s}{p['gain_s']:+9.4f}   [{lo:+8.4f}, {hi:+8.4f}]{str(p['excludes_zero']):>8s}"
            f"{gs:>9s}{ex:>8s}{p['airports_improving']:>10d} / {p['n_airports']:<7d}")
    aw = max(11, max(len(n) for n in arms) + 1)
    log(f"{'airport':8s}{'n':>9s}" + "".join(f"{a:>{aw}s}" for a in arms) + "".join(f"{k:>{pw}s}" for k in pairs))
    for i, a in present:
        n_a = int((np.asarray(ap_te) == i).sum())
        log(f"{a:8s}{n_a:9,d}" + "".join(f"{rmses[name]['per_airport'][a]:{aw}.2f}" for name in arms)
            + "".join((f"{pairs[k]['per_airport'][a]['gain_s']:+{pw - 5}.2f}"
                       f"{'*' if pairs[k]['per_airport'][a]['excludes_zero'] else ' ':<5s}")
                      if pairs[k]["per_airport"][a] is not None else f"{'one row':>{pw - 5}s}     " for k in pairs))
    log("   (* = that airport's own paired interval excludes zero)")


# =============================================================================================
# the v4 arms (Amendment 12)
# =============================================================================================

def run_v4(args, cfg, paths, started) -> int:
    json_path, log_path, preds_path = paths
    smoke = cfg.smoke
    pa_rule = L.PA_TREE_RULE if args.pa_trees == "share" else L.PA_TREE_RULE_ES
    log(f"per-airport tree rule [{args.pa_trees}]: {pa_rule}; share floor {cfg.pa_floor} (real floor "
        f"{L.PA_TREE_FLOOR}); row floor {L.PA_MIN_ROWS:,}; blend {L.BLEND}; seeds {list(SEEDS)}; "
        f"ES seed {L.ES_SEED}; bootstrap {cfg.n_boot:,} draws, seed {BOOT_SEED}")

    # ---- 1-2. rows, encodings, design matrix (fold A) ----
    fold = load_fold(CACHE, cfg.months, list(L.FEATS), check_counts=not smoke, label=f"holdout (months {HOLDOUT})")
    X, y, dlt, proxy = fold["X"], fold["y"], fold["dlt"], fold["proxy"]
    te, tr, fit, es = fold["te"], fold["tr"], fold["fit"], fold["es"]
    ap_code, airports = fold["ap_code"], fold["airports"]
    X_te = X[te]
    log(f"holdout copy {X_te.nbytes / 1e6:.0f} MB")
    base = fold["base"].drop(columns=["sp"])           # the v4 parquet keeps its legacy columns
    params, nest, patience = cfg.params, cfg.nest, cfg.patience
    cols = {}

    # ---- 3. early stopping once (seed 0) -> n_ref; then A0, A1a, A1b ----
    info = L.early_stop(X, dlt, y, proxy, tr, fit, es, dict(params, seed=L.ES_SEED), nest, patience)
    n_ref = info["n_ref"]
    single, timing = {}, {}
    repro = {"status": "skipped (smoke)"} if smoke else None
    for s in SEEDS:
        t = time.time()
        single.update(L.fit_seeds(X, dlt, tr, X_te, params, n_ref, (s,)))
        timing[f"seed{s}_s"] = round(time.time() - t, 1)
        cols[f"delta_hat_seed{s}"] = single[s]
        _write_preds(preds_path, base, cols)
        r = rmse(y[te], taxi_time(proxy[te], single[s]))
        log(f"seed {s} alone: matched RMSE {r:.4f} on {te.sum():,} holdout rows   [{timing[f'seed{s}_s']:.0f}s]")
        if s == L.ES_SEED and not smoke:
            diff = r - LGBM_AB["lgb_refit"]
            ok = abs(diff) <= REPRO_TOL_S
            repro = dict(status="pass" if ok else "FAIL", expected_rmse=LGBM_AB["lgb_refit"], got_rmse=r,
                         diff_s=diff, tol_s=REPRO_TOL_S, expected_best_iter=LGBM_AB["best_iter"],
                         got_best_iter=info["best_iter"], expected_n_ref=LGBM_AB["n_ref"], got_n_ref=n_ref,
                         note="LightGBM at num_threads=4 is not bit-deterministic run to run on this "
                              "machine; a few hundredths of a second is expected, seconds are a bug")
            log(f"A0 reproduction of lgbm_ab lgb_refit: got {r:.4f} vs {LGBM_AB['lgb_refit']:.4f} "
                f"(diff {diff:+.4f} s, tolerance {REPRO_TOL_S} s) -> {repro['status']}; best_iter "
                f"{info['best_iter']:,} vs {LGBM_AB['best_iter']:,}; n_ref {n_ref:,} vs {LGBM_AB['n_ref']:,}")
            if not ok:
                partial = {"mode": "v4", "smoke": smoke, **_git(), "started_utc": started.isoformat(),
                           "a0_reproduction": repro, "fold": {"n_test": int(te.sum())},
                           "note": "A0 missed lgbm_ab's number; the harness is wrong or the run is "
                                   "not deterministic beyond tolerance. Nothing else was measured."}
                _write_json(json_path, partial)
                if not args.continue_on_repro_fail:
                    log(f"STOP: A0 does not reproduce lgbm_ab within {REPRO_TOL_S} s; partial json -> "
                        f"{json_path}; pass --continue-on-repro-fail to measure the other arms anyway")
                    return 2
                log("continuing on --continue-on-repro-fail: every number below carries this caveat")

    # ---- 4. per-airport models (mean over seeds) and the blend -> A3 ----
    t = time.time()
    pa_by_seed, fitted, pa_info = L.fit_per_airport(
        X, dlt, tr, te, ap_code, airports, params, n_ref, SEEDS, min_rows=L.PA_MIN_ROWS, floor=cfg.pa_floor,
        trees=args.pa_trees, es=dict(y=y, proxy=proxy, fit=fit, es=es, nest=nest, patience=patience))
    timing["per_airport_s"] = round(time.time() - t, 1)
    for s in SEEDS:
        cols[f"pa_seed{s}"] = pa_by_seed[s]
    cols["pa_fitted"] = fitted
    arms = build_arms(single, L.mean_delta([pa_by_seed[s] for s in SEEDS]), fitted)
    for name, v in arms.items():
        cols[name] = v
    _write_preds(preds_path, base, cols)
    del X, X_te
    fold["X"] = None
    gc.collect()
    n_fitted = sum(1 for r in pa_info.values() if r["fitted"])
    log(f"per-airport: {n_fitted}/{len(airports)} airports fitted, {int(fitted.sum()):,} holdout rows "
        f"blended, {int((~fitted).sum()):,} keep A2   [{timing['per_airport_s']:.0f}s]; predictions -> "
        f"{preds_path}")

    # ---- 5. scoring: pooled and per airport, seed sd, paired bootstrap, the 2x rule ----
    t = time.time()
    se = squared_errors(arms, y[te], proxy[te])
    rmses = arm_rmses(se, ap_code[te], airports)
    singles = {a: rmses[a]["pooled"] for a in ("A0", "A1a", "A1b")}
    sd = seed_sd(singles.values())
    pairs, present = score_pairs(se, rmses, PAIRS, ap_code[te], airports, cfg.n_boot, sd)
    timing["scoring_s"] = round(time.time() - t, 1)
    bind = {name: float(((proxy[te] - v) < 1.0).mean()) for name, v in arms.items()}

    # ---- 6. the readable tables ----
    _log_tables(arms, rmses, pairs, present, ap_code[te], ARM_DEFINITIONS, sd,
                f"FOLD A, matched rows ({te.sum():,}), holdout months {HOLDOUT}   "
                + (L.SMOKE_BANNER if smoke else "n_ref {:,} (best_iter {:,})".format(n_ref, info["best_iter"])))
    log(f"floor-bind rate per arm: " + ", ".join(f"{k} {100 * v:.3f}%" for k, v in bind.items()))
    if repro and repro.get("status") == "FAIL":
        log(f"CAVEAT: A0 missed lgbm_ab by {repro['diff_s']:+.4f} s; every number above inherits that")

    # ---- 7. the record ----
    result = _record_head("v4", cfg, started, fold)
    result.update({
        "config": {"params": params, "seeds": list(SEEDS), "es_seed": L.ES_SEED, "max_rounds": nest,
                   "patience": patience, "best_iter": info["best_iter"], "n_ref": n_ref,
                   "es_rmse_taxi_time_NOT_A_RESULT": info["es_rmse_taxi_time_NOT_A_RESULT"],
                   "es_proxy_only_rmse": info["es_proxy_only_rmse"],
                   "pa_min_rows": L.PA_MIN_ROWS, "pa_tree_floor": cfg.pa_floor, "pa_trees": args.pa_trees,
                   "pa_tree_rule": pa_rule,
                   "blend": L.BLEND, "n_boot": cfg.n_boot, "boot_seed": BOOT_SEED,
                   "features": list(L.FEATS), "n_features": len(L.FEATS)},
        "a0_reproduction": repro,
        "arms": {name: {"definition": ARM_DEFINITIONS[name], "rmse": rmses[name]["pooled"],
                        "per_airport": rmses[name]["per_airport"], "floor_bind_rate": bind[name]}
                 for name in arms},
        "seed_sd": {"values": singles, "sd": sd, "ddof": 1},
        "pairs": pairs,
        "per_airport_models": pa_info,
        "timing_s": timing,
        "preds_parquet": str(preds_path), "log": str(log_path),
    })
    _write_json(json_path, result)
    log(f"json -> {json_path}   wall {result['wall_s']:.0f}s   peak RSS {result['peak_rss_gb']:.2f} GB")
    return 0


# =============================================================================================
# Amendment 14: the queue arm
# =============================================================================================

def run_queue(args, cfg, paths, started) -> int:
    json_path, log_path, preds_path = paths
    seeds, arm, n_perm = tuple(args.seeds), args.baseline, int(args.n_perm)
    pa_rule = L.PA_TREE_RULE if args.pa_trees == "share" else L.PA_TREE_RULE_ES
    log(f"Amendment 14 arm: baseline {arm} over seeds {list(seeds)}; treatment = baseline + {len(QUEUE_FEATS)} "
        f"QUEUE_FEATS ({len(FEATS_QUEUE)} features), best_iter re-found; per-airport rule [{args.pa_trees}] "
        f"{pa_rule} (used by A3 only); bootstrap {cfg.n_boot:,} draws seed {BOOT_SEED}; negative control "
        f"{n_perm} permutations seed {PERM_SEED}")
    v4 = _v4_record(args, cfg, arm, seeds, args.pa_trees) if _reads_v4(args, cfg) else None
    if v4 is None:
        log("baseline: fitted in-process (--refit-baseline, or a smoke without a v4 record)")

    fold = load_fold(CACHE, cfg.months, FEATS_QUEUE, qcache=QCACHE, check_counts=not cfg.smoke,
                     label=f"holdout (months {HOLDOUT})")
    X, y, dlt, proxy, sp = fold["X"], fold["y"], fold["dlt"], fold["proxy"], fold["sp"]
    te, tr = fold["te"], fold["tr"]
    ap_code, airports = fold["ap_code"], fold["airports"]
    y_te, proxy_te, sp_te, dlt_te, ap_te = y[te], proxy[te], sp[te], dlt[te], ap_code[te]
    nb = len(L.FEATS)
    assert fold["feats"][:nb] == list(L.FEATS) and fold["feats"][nb:] == QUEUE_FEATS
    base, cols = fold["base"], {}
    write = lambda: _write_preds(preds_path, base, cols)
    timing = {}

    # ---- baseline ----
    t = time.time()
    b, b_record, sd, sd_source = resolve_baseline(args, cfg, fold, X[:, :nb], arm, seeds, args.pa_trees, v4,
                                                  "baseline", cols, write)
    timing["baseline_s"] = round(time.time() - t, 1)

    # ---- treatment: the same procedure on FEATS + QUEUE_FEATS ----
    t = time.time()

    def on_seed(s, pred):
        cols[f"delta_hat_seed{s}"] = pred
        write()
    tr_arm = fit_arm(fold, X, cfg.params, cfg.nest, cfg.patience, cfg.pa_floor, seeds, per_airport=(arm == "A3"),
                     pa_trees=args.pa_trees, name="treatment", on_seed=on_seed)
    if arm == "A3":
        for s in seeds:
            cols[f"pa_seed{s}"] = tr_arm["pa_by_seed"][s]
        cols["pa_fitted"] = tr_arm["fitted"]
    cols["treatment"] = tr_arm["delta"]
    write()
    timing["treatment_s"] = round(time.time() - t, 1)
    t_sd = seed_sd(tr_arm["single_rmse"].values()) if len(seeds) == 3 else None
    if sd is None and t_sd is not None:
        sd, sd_source = t_sd, "treatment single seeds"

    # ---- scoring: pooled, per airport, the named airports, the tail subset ----
    t = time.time()
    arms = {"baseline": b["delta"], "treatment": tr_arm["delta"]}
    definitions = {"baseline": f"the v4 arm {arm} over seeds {list(seeds)} ({b_record['source']})",
                   "treatment": f"baseline + QUEUE_FEATS, best_iter {tr_arm['best_iter']:,} -> n_ref {tr_arm['n_ref']:,}"}
    se = squared_errors(arms, y_te, proxy_te)
    rmses = arm_rmses(se, ap_te, airports)
    pairs, present = score_pairs(se, rmses, [("treatment", "baseline")], ap_te, airports, cfg.n_boot, sd)
    p = pairs["treatment_vs_baseline"]
    p["at_least_4_airports"] = bool(p["airports_improving"] >= 4)
    p["named_airports"] = {a: bool(rmses["treatment"]["per_airport"][a] < rmses["baseline"]["per_airport"][a])
                           for a in NAMED_AIRPORTS if a in rmses["baseline"]["per_airport"]}
    tail = tail_mask(y_te, sp_te, dlt_te)
    n_tail = int(tail.sum())
    tail_record = {"definition": f"non-fill (|y - sp| > {TAIL_NONFILL_S:.0f} s) AND |delta| > {TAIL_DELTA_S:.0f} s",
                   "n_rows": n_tail, "share_of_holdout_rows": float(tail.mean()),
                   "share_of_baseline_sse": float(se["baseline"][tail].sum() / se["baseline"].sum()),
                   "rmse": {k: float(np.sqrt(v[tail].mean())) for k, v in se.items()} if n_tail else None,
                   "pair": (paired_bootstrap({k: v[tail] for k, v in se.items()}, [("treatment", "baseline")],
                                             cfg.n_boot, BOOT_SEED)["treatment_vs_baseline"] if n_tail >= 2 else None)}
    timing["scoring_s"] = round(time.time() - t, 1)

    # ---- the negative control: quarter capacity, pooled, seed 0, registered deviation ----
    t = time.time()
    n_red = reduced_trees(tr_arm["best_iter"])
    log(f"negative control - registered deviation: {CONTROL_DEVIATION}")
    log(f"negative control: {n_perm} permutations (seed {PERM_SEED}) of the queue block as whole rows within "
        f"airport x fold side; reduced tree count {n_red:,} = best_iter {tr_arm['best_iter']:,} // 4; single-seed "
        f"(seed {L.ES_SEED}) pooled refits on all {tr.sum():,} training rows; no per-airport blend")
    params0 = dict(cfg.params, seed=L.ES_SEED)
    Xb = X[:, :nb]
    base_red = L.fit_seeds(Xb, dlt, tr, Xb[te], params0, n_red, (L.ES_SEED,), n_features=nb)[L.ES_SEED]
    treat_red = L.fit_seeds(X, dlt, tr, X[te], params0, n_red, (L.ES_SEED,), n_features=X.shape[1])[L.ES_SEED]
    r_base_red = rmse(y_te, taxi_time(proxy_te, base_red))
    r_treat_red = rmse(y_te, taxi_time(proxy_te, treat_red))
    gain_red = r_base_red - r_treat_red
    cols["control_baseline_red"], cols["control_treatment_red"] = base_red, treat_red
    write()
    log(f"  at {n_red:,} trees: baseline {r_base_red:.4f}, treatment {r_treat_red:.4f}, gain {gain_red:+.4f} s")
    groups = fold["ap_code"].astype(np.int64) * 2 + te.astype(np.int64)
    orig = X[:, nb:].copy()
    rng = np.random.default_rng(PERM_SEED)
    perm_gains, perm_rmses = [], []
    for k in range(n_perm):
        blk = orig.copy()
        permute_within_groups(blk, groups, rng)
        X[:, nb:] = blk
        del blk
        pk = L.fit_seeds(X, dlt, tr, X[te], params0, n_red, (L.ES_SEED,), n_features=X.shape[1])[L.ES_SEED]
        rk = rmse(y_te, taxi_time(proxy_te, pk))
        perm_rmses.append(rk)
        perm_gains.append(r_base_red - rk)
        cols[f"control_perm{k}_red"] = pk
        write()
        log(f"  permutation {k + 1}/{n_perm}: matched RMSE {rk:.4f}   gain vs the baseline at {n_red:,} trees "
            f"{r_base_red - rk:+.4f} s")
    X[:, nb:] = orig
    del orig
    control = control_summary(r_base_red, r_treat_red, perm_rmses)
    assert control["exceeds_p95"] == control["treatment_below_p5"]
    control.update(seed=PERM_SEED, reduced_trees=n_red, best_iter=tr_arm["best_iter"], deviation=CONTROL_DEVIATION,
                   groups="airport x fold side", registered_text=(
                       "Amendment 14.4: 100 within-airport row permutations at full capacity; the treatment's "
                       "gain must exceed the 95th percentile of the permuted gains"))
    timing["control_s"] = round(time.time() - t, 1)
    del X, Xb
    fold["X"] = None
    gc.collect()

    # ---- tables ----
    _log_tables(arms, rmses, pairs, present, ap_te, definitions, sd,
                f"AMENDMENT 14, fold A matched rows ({te.sum():,}), holdout months {HOLDOUT}   "
                + (L.SMOKE_BANNER if cfg.smoke else f"treatment n_ref {tr_arm['n_ref']:,} (best_iter {tr_arm['best_iter']:,})"))
    log(f"seed sd source: {sd_source}   treatment's own seed sd: "
        + (f"{t_sd:.4f} s" if t_sd is not None else "n/a"))
    log(f"airports improving {p['airports_improving']} / {p['n_airports']} (>= 4: {p['at_least_4_airports']}); "
        f"named in advance: " + ", ".join(f"{a} {'improves' if v else 'does not'}" for a, v in p["named_airports"].items()))
    if n_tail:
        tp = tail_record["pair"]
        log(f"tail subset [{tail_record['definition']}]: {n_tail:,} rows ({100 * tail.mean():.2f}% of rows, "
            f"{100 * tail_record['share_of_baseline_sse']:.1f}% of the baseline's SSE): baseline "
            f"{tail_record['rmse']['baseline']:.2f}, treatment {tail_record['rmse']['treatment']:.2f}"
            + (f", gain {tp['gain_s']:+.2f} s, paired CI [{tp['ci95'][0]:+.2f}, {tp['ci95'][1]:+.2f}]" if tp else ""))
    else:
        log(f"tail subset [{tail_record['definition']}]: no rows")
    log(f"negative control at {n_red:,} trees: treatment gain {control['gain_reduced_s']:+.4f} s vs the 95th "
        f"percentile of {n_perm} permuted gains {control['p95_permuted_gain_s']:+.4f} s (max "
        f"{control['permuted_gain_max_s']:+.4f}, {control['n_permuted_at_or_above']} at or above) -> exceeds p95: "
        f"{control['exceeds_p95']}   [registered deviation: {CONTROL_DEVIATION.split(':')[0]}]")
    log(f"  like-for-like form (the baseline cancels): treatment RMSE {r_treat_red:.4f} vs the 5th percentile of the "
        f"permuted RMSEs {control['p5_permuted_rmse']:.4f} (min {min(perm_rmses):.4f}) -> below p5: "
        f"{control['treatment_below_p5']}. The permuted gains vs the 68-column baseline carry the column-count "
        f"effect on LightGBM's feature-sampling stream; see negative_control.note in the json")

    # ---- the record ----
    result = _record_head("queue", cfg, started, fold, {"amendment": "14"})
    result.update({
        "config": {"params": cfg.params, "seeds": list(seeds), "es_seed": L.ES_SEED, "max_rounds": cfg.nest,
                   "patience": cfg.patience, "baseline_arm": arm, "pa_trees": args.pa_trees, "pa_tree_rule": pa_rule,
                   "pa_min_rows": L.PA_MIN_ROWS, "pa_tree_floor": cfg.pa_floor, "blend": L.BLEND,
                   "n_boot": cfg.n_boot, "boot_seed": BOOT_SEED, "features": fold["feats"],
                   "n_features": len(fold["feats"]), "queue_feats": QUEUE_FEATS, "queue_cache": str(QCACHE),
                   "n_perm": n_perm, "perm_seed": PERM_SEED, "reduced_trees": n_red,
                   "control_deviation": CONTROL_DEVIATION, "named_airports": list(NAMED_AIRPORTS),
                   "tail_definition": tail_record["definition"]},
        "baseline": b_record,
        "treatment": {"best_iter": tr_arm["best_iter"], "n_ref": tr_arm["n_ref"],
                      "es_rmse_taxi_time_NOT_A_RESULT": tr_arm["info"]["es_rmse_taxi_time_NOT_A_RESULT"],
                      "es_proxy_only_rmse": tr_arm["info"]["es_proxy_only_rmse"],
                      "single_seed_rmses": {str(s): r for s, r in tr_arm["single_rmse"].items()},
                      "per_airport_models": tr_arm["pa_info"], "timing_s": tr_arm["timing"]},
        "arms": _arm_records(arms, rmses, definitions, proxy_te),
        "seed_sd": {"sd": sd, "source": sd_source,
                    "treatment_values": {str(s): r for s, r in tr_arm["single_rmse"].items()}, "treatment_sd": t_sd,
                    "baseline_values": b_record.get("single_seed_rmses"), "ddof": 1},
        "pairs": pairs,
        "tail_subset": tail_record,
        "negative_control": control,
        "timing_s": timing,
        "preds_parquet": str(preds_path), "log": str(log_path),
    })
    _write_json(json_path, result)
    log(f"json -> {json_path}   wall {result['wall_s']:.0f}s   peak RSS {result['peak_rss_gb']:.2f} GB")
    return 0


# =============================================================================================
# Amendment 15.1: the CatBoost arm
# =============================================================================================

def run_catboost(args, cfg, paths, started) -> int:
    json_path, log_path, preds_path = paths
    seeds, arm = tuple(args.seeds), args.baseline
    cb = dict(CATBOOST)
    if cfg.smoke:
        cb.update(SMOKE_CATBOOST)
    python, worker = pathlib.Path(args.catboost_python).expanduser(), pathlib.Path(args.catboost_worker)
    if not python.exists():
        raise FileNotFoundError(f"CatBoost interpreter {python} is missing: the arm runs ONLY under the isolated "
                                "venv ~/.venvs/prc-catboost (never install catboost globally)")
    log(f"Amendment 15.1 arm: baseline {arm} over seeds {list(seeds)}; CatBoost {cb} under {python}; blends "
        f"{list(BLEND_WEIGHTS)}; bootstrap {cfg.n_boot:,} draws seed {BOOT_SEED}")
    v4 = _v4_record(args, cfg, arm, seeds, args.pa_trees) if _reads_v4(args, cfg) else None
    if v4 is None:
        log("baseline: fitted in-process (--refit-baseline, or a smoke without a v4 record)")

    fold = load_fold(CACHE, cfg.months, list(L.FEATS), check_counts=not cfg.smoke, label=f"holdout (months {HOLDOUT})")
    X, y, dlt, proxy = fold["X"], fold["y"], fold["dlt"], fold["proxy"]
    te, tr, fit, es = fold["te"], fold["tr"], fold["fit"], fold["es"]
    ap_te, airports = fold["ap_code"][te], fold["airports"]
    y_te, proxy_te = y[te], proxy[te]
    base, cols = fold["base"], {}
    write = lambda: _write_preds(preds_path, base, cols)
    timing = {}

    work_dir = preds_path.parent
    npz, npy = work_dir / CATBOOST_INPUT, work_dir / CATBOOST_OUTPUT
    t = time.time()
    write_worker_input(npz, X, dlt, tr, fit, es, te)
    timing["npz_write_s"] = round(time.time() - t, 1)
    log(f"worker input -> {npz} ({npz.stat().st_size / 1e9:.2f} GB; holdout targets blanked)   "
        f"[{timing['npz_write_s']:.0f}s]")

    t = time.time()
    b, b_record, sd, sd_source = resolve_baseline(args, cfg, fold, X, arm, seeds, args.pa_trees, v4, "baseline",
                                                  cols, write)
    timing["baseline_s"] = round(time.time() - t, 1)
    del X
    fold["X"] = None
    gc.collect()
    log(f"freed the design matrix before the worker starts: current RSS {_current_rss_gb():.2f} GB "
        f"(peak so far {L.peak_rss_gb():.2f} GB)")

    t = time.time()
    cat, sidecar, run = run_catboost_worker(python, worker, npz, npy, cb, int(te.sum()), log)
    timing["worker_s"] = round(time.time() - t, 1)
    cols["catboost"] = cat
    arms = {"baseline": b["delta"], "catboost": cat}
    definitions = {"baseline": f"the v4 arm {arm} over seeds {list(seeds)} ({b_record['source']})",
                   "catboost": f"CatBoost alone, best iteration {sidecar.get('best_iteration')}, refit {sidecar.get('n_refit')}"}
    for w in BLEND_WEIGHTS:
        arms[f"blend_{w}"] = (1.0 - w) * b["delta"] + w * cat
        cols[f"blend_{w}"] = arms[f"blend_{w}"]
        definitions[f"blend_{w}"] = f"{1 - w:.1f} baseline + {w:.1f} CatBoost"
    write()

    t = time.time()
    se = squared_errors(arms, y_te, proxy_te)
    rmses = arm_rmses(se, ap_te, airports)
    pair_list = [(f"blend_{w}", "baseline") for w in BLEND_WEIGHTS] + [("catboost", "baseline")]
    pairs, present = score_pairs(se, rmses, pair_list, ap_te, airports, cfg.n_boot, sd)
    curve = blend_curve(b["delta"], cat, y_te, proxy_te)
    increases = curve_increases_with_weight(curve)
    corr = residual_correlation(y_te, taxi_time(proxy_te, b["delta"]), taxi_time(proxy_te, cat))
    timing["scoring_s"] = round(time.time() - t, 1)

    _log_tables(arms, rmses, pairs, present, ap_te, definitions, sd,
                f"AMENDMENT 15.1, fold A matched rows ({te.sum():,}), holdout months {HOLDOUT}   "
                + (L.SMOKE_BANNER if cfg.smoke else f"CatBoost best iteration {sidecar.get('best_iteration')}"))
    log("blend weight curve (CatBoost weight -> matched RMSE): "
        + ", ".join(f"{w:.1f} -> {v:.4f}" for w, v in curve.items())
        + f"   rises monotonically over 0.3/0.5/0.7 (dilution): {increases}")
    log(f"residual correlation baseline vs CatBoost: {corr:.4f}   (15.1's mechanism needs < 1)")

    result = _record_head("catboost", cfg, started, fold, {"amendment": "15.1"})
    result.update({
        "config": {"params_lgbm": cfg.params, "seeds": list(seeds), "es_seed": L.ES_SEED, "max_rounds": cfg.nest,
                   "patience": cfg.patience, "baseline_arm": arm, "pa_trees": args.pa_trees, "catboost": cb,
                   "blend_weights": list(BLEND_WEIGHTS), "n_boot": cfg.n_boot, "boot_seed": BOOT_SEED,
                   "features": fold["feats"], "n_features": len(fold["feats"])},
        "baseline": b_record,
        "worker": {**run, "sidecar": sidecar},
        "arms": _arm_records(arms, rmses, definitions, proxy_te),
        "seed_sd": {"sd": sd, "source": sd_source, "baseline_values": b_record.get("single_seed_rmses"), "ddof": 1},
        "pairs": pairs,
        "blend_curve": {str(w): v for w, v in curve.items()},
        "curve_rmse_increases_with_weight": increases,
        "residual_correlation": corr,
        "timing_s": timing,
        "preds_parquet": str(preds_path), "log": str(log_path),
    })
    _write_json(json_path, result)
    log(f"json -> {json_path}   wall {result['wall_s']:.0f}s   peak RSS {result['peak_rss_gb']:.2f} GB")
    return 0


# =============================================================================================
# Amendment 15.2: the screening tier
# =============================================================================================

def run_sweep(args, cfg, paths, started) -> int:
    json_path, log_path, _ = paths
    sub = SMOKE_SWEEP if cfg.smoke else SWEEP
    train_months, stop = tuple(int(m) for m in sub["train"]), int(sub["stop"])
    months = train_months + (stop,)
    params = dict(cfg.params, learning_rate=SWEEP["learning_rate"])
    grid = sweep_grid()
    log(f"Amendment 15.2 screening tier: {len(grid)} settings {SWEEP_AXES} early-stopped at lr "
        f"{params['learning_rate']} on the subfold train {list(train_months)} / stop {stop}; encodings fitted on the "
        f"training months only; no refit; ranked by the stopping-set matched RMSE")
    fold = load_fold(CACHE, months, list(L.FEATS), split=lambda m: sweep_masks(m, train_months, stop),
                     check_counts=False, label=f"stop month {stop}")
    X, y, dlt, proxy = fold["X"], fold["y"], fold["dlt"], fold["proxy"]
    st, tr = fold["te"], fold["tr"]
    results = []
    for i, s in enumerate(grid, 1):
        p = setting_params(params, s)
        t = time.time()
        log(f"screen {i}/{len(grid)}: setting {setting_label(s)}")
        info = L.early_stop(X, dlt, y, proxy, tr, tr, st, dict(p, seed=L.ES_SEED), cfg.nest, cfg.patience)
        log("screen: no refit follows; only best_iter and the stopping-set RMSE are kept")
        results.append(dict(setting=s, rmse_stop=info["es_rmse_taxi_time_NOT_A_RESULT"], best_iter=info["best_iter"],
                            proxy_only=info["es_proxy_only_rmse"], wall_s=round(time.time() - t, 1)))
    del X
    fold["X"] = None
    gc.collect()
    ranked = rank_settings(results)
    inc = setting_label(INCUMBENT_SETTING)
    inc_rank = next(e["rank"] for e in ranked if e["setting"] == inc)
    log("=" * 100)
    log(f"SCREEN RANKING, subfold train {list(train_months)} / stop {stop}, {st.sum():,} stopping rows   "
        + (L.SMOKE_BANNER if cfg.smoke else ""))
    log("=" * 100)
    log(f"{'rank':>5s}  {'setting':14s}{'best_iter':>10s}{'stop RMSE':>12s}")
    for e in ranked:
        log(f"{e['rank']:5d}  {e['setting']:14s}{e['best_iter']:10,d}{e['rmse_stop_RANKING_ONLY']:12.4f}"
            + ("   <- incumbent" if e["setting"] == inc else ""))
    log(f"incumbent {inc} ranks {inc_rank}; top two for --confirm: {ranked[0]['setting']} {ranked[1]['setting']}")
    result = _record_head("sweep", cfg, started, None, {"amendment": "15.2", "tier": "screening", "banner": RANKING_BANNER})
    result.update({
        "subfold": {"train_months": list(train_months), "stop_month": stop},
        "fold": {"months": sorted(months), "n_train": int(tr.sum()), "n_stop": int(st.sum()), "airports": fold["airports"]},
        "config": {"learning_rate": params["learning_rate"], "grid_size": len(grid), "axes": {k: list(v) for k, v in SWEEP_AXES.items()},
                   "max_rounds": cfg.nest, "patience": cfg.patience, "es_seed": L.ES_SEED,
                   "base_params": {k: v for k, v in params.items() if k not in ("num_leaves", "min_data_in_leaf", "feature_fraction")},
                   "features": fold["feats"], "n_features": len(fold["feats"])},
        "ranking": ranked,
        "incumbent_setting": inc, "incumbent_rank": inc_rank,
        "top_two": [ranked[0]["setting"], ranked[1]["setting"]],
        "log": str(log_path),
    })
    _write_json(json_path, result)
    log(f"json -> {json_path}   wall {result['wall_s']:.0f}s   peak RSS {result['peak_rss_gb']:.2f} GB")
    return 0


# =============================================================================================
# Amendment 15.2: the confirmation tier
# =============================================================================================

def run_confirm(args, cfg, paths, started) -> int:
    json_path, log_path, preds_path = paths
    seeds = tuple(args.seeds)
    settings = list(dict.fromkeys(tuple(s) for s in args.confirm))          # unique, in the order given
    labels = [setting_label(s) for s in settings]
    log(f"Amendment 15.2 confirmation tier: {labels} as full fold-A arms over seeds {list(seeds)} against the "
        f"incumbent {setting_label(INCUMBENT_SETTING)}; bootstrap {cfg.n_boot:,} draws seed {BOOT_SEED}")
    v4 = _v4_record(args, cfg, "A2", seeds, args.pa_trees) if _reads_v4(args, cfg) else None
    if v4 is None:
        log("incumbent: fitted in-process (--refit-baseline, or a smoke without a v4 record)")

    fold = load_fold(CACHE, cfg.months, list(L.FEATS), check_counts=not cfg.smoke, label=f"holdout (months {HOLDOUT})")
    X, y, proxy, te = fold["X"], fold["y"], fold["proxy"], fold["te"]
    ap_te, airports = fold["ap_code"][te], fold["airports"]
    y_te, proxy_te = y[te], proxy[te]
    base, cols = fold["base"], {}
    write = lambda: _write_preds(preds_path, base, cols)
    timing = {}

    t = time.time()
    inc, inc_record, sd, sd_source = resolve_baseline(args, cfg, fold, X, "A2", seeds, args.pa_trees, v4, "incumbent",
                                                      cols, write)
    timing["incumbent_s"] = round(time.time() - t, 1)
    arms = {"incumbent": inc["delta"]}
    definitions = {"incumbent": f"setting {setting_label(INCUMBENT_SETTING)} over seeds {list(seeds)} ({inc_record['source']})"}
    candidates = {}
    for s, label in zip(settings, labels):
        p = setting_params(cfg.params, s)
        t = time.time()

        def on_seed(sd_, pred, label=label):
            cols[f"{label}_seed{sd_}"] = pred
            write()
        c = fit_arm(fold, X, p, cfg.nest, cfg.patience, cfg.pa_floor, seeds, per_airport=False, pa_trees=None,
                    name=label, on_seed=on_seed)
        cols[label] = c["delta"]
        write()
        arms[label] = c["delta"]
        definitions[label] = f"setting {label}, best_iter {c['best_iter']:,} -> n_ref {c['n_ref']:,}, mean over seeds {list(seeds)}"
        candidates[label] = dict(setting=list(s), params=p, best_iter=c["best_iter"], n_ref=c["n_ref"],
                                 single_seed_rmses={str(k): r for k, r in c["single_rmse"].items()},
                                 seed_sd_own=(seed_sd(c["single_rmse"].values()) if len(seeds) == 3 else None),
                                 es_rmse_taxi_time_NOT_A_RESULT=c["info"]["es_rmse_taxi_time_NOT_A_RESULT"],
                                 timing_s=c["timing"], wall_s=round(time.time() - t, 1))
    del X
    fold["X"] = None
    gc.collect()

    t = time.time()
    se = squared_errors(arms, y_te, proxy_te)
    rmses = arm_rmses(se, ap_te, airports)
    pairs, present = score_pairs(se, rmses, [(label, "incumbent") for label in labels], ap_te, airports, cfg.n_boot, sd)
    for label in labels:
        candidates[label]["rmse"] = rmses[label]["pooled"]
        candidates[label]["per_airport"] = rmses[label]["per_airport"]
    timing["scoring_s"] = round(time.time() - t, 1)
    _log_tables(arms, rmses, pairs, present, ap_te, definitions, sd,
                f"AMENDMENT 15.2 CONFIRMATION, fold A matched rows ({te.sum():,}), holdout months {HOLDOUT}   "
                + (L.SMOKE_BANNER if cfg.smoke else ""))
    log(f"seed sd source: {sd_source}; candidates' own seed sd: "
        + ", ".join(f"{k} {v['seed_sd_own']:.4f}" if v["seed_sd_own"] is not None else f"{k} n/a"
                    for k, v in candidates.items()))

    result = _record_head("confirm", cfg, started, fold, {"amendment": "15.2", "tier": "confirmation"})
    result.update({
        "config": {"params_incumbent": cfg.params, "seeds": list(seeds), "es_seed": L.ES_SEED, "max_rounds": cfg.nest,
                   "patience": cfg.patience, "settings": labels, "n_boot": cfg.n_boot, "boot_seed": BOOT_SEED,
                   "features": fold["feats"], "n_features": len(fold["feats"])},
        "incumbent": {"setting": setting_label(INCUMBENT_SETTING), **inc_record, "rmse": rmses["incumbent"]["pooled"],
                      "per_airport": rmses["incumbent"]["per_airport"]},
        "candidates": candidates,
        "arms": _arm_records(arms, rmses, definitions, proxy_te),
        "seed_sd": {"sd": sd, "source": sd_source, "incumbent_values": inc_record.get("single_seed_rmses"), "ddof": 1},
        "pairs": pairs,
        "timing_s": timing,
        "preds_parquet": str(preds_path), "log": str(log_path),
    })
    _write_json(json_path, result)
    log(f"json -> {json_path}   wall {result['wall_s']:.0f}s   peak RSS {result['peak_rss_gb']:.2f} GB")
    return 0


# =============================================================================================
# Amendment 16: the schedule-fill mixture head
# =============================================================================================

def _log_subset(title: str, rec: dict) -> None:
    if not rec["n_rows"]:
        log(f"{title} [{rec['definition']}]: no rows")
        return
    pr = rec["pair"]
    sse = rec.get("share_of_baseline_sse")
    log(f"{title} [{rec['definition']}]: {rec['n_rows']:,} rows ({100 * rec['share_of_holdout_rows']:.2f}% of rows, "
        + (f"{100 * sse:.1f}% of the baseline's SSE" if sse is not None else "SSE share n/a")
        + f"): baseline {rec['rmse']['baseline']:.2f}, treatment {rec['rmse']['treatment']:.2f}"
        + (f", gain {pr['gain_s']:+.2f} s, paired CI [{pr['ci95'][0]:+.2f}, {pr['ci95'][1]:+.2f}]"
           f"{' *' if pr['excludes_zero'] else ''}" if pr else ", no interval (one row)"))


def run_fillhead(args, cfg, paths, started) -> int:
    json_path, log_path, preds_path = paths
    seeds, arm = tuple(args.seeds), args.baseline
    pa_rule = L.PA_TREE_RULE if args.pa_trees == "share" else L.PA_TREE_RULE_ES
    params_head = L.head_params(cfg.params)
    nb = len(L.FEATS)
    log(f"Amendment 16 arm: baseline {arm} over seeds {list(seeds)}; head = LightGBM {params_head['objective']} "
        f"(metric {params_head['metric']}) on FEATS + [{L.NMDELAY}] ({len(FEATS_HEAD)} features), early-stopped on "
        f"months {L.ES_MONTHS} (seed {L.ES_SEED}, patience {cfg.patience}), one refit on all training rows at n_ref; "
        f"treatment = p * sp + (1 - p) * max(proxy - baseline, 1); fill := |y - sp| <= {L.FILL_TOL_S:.0f} s; "
        f"per-airport rule [{args.pa_trees}] (A3 baseline only); bootstrap {cfg.n_boot:,} draws seed {BOOT_SEED}; "
        f"reliability in {N_DECILES} equal-count bins")
    v4 = _v4_record(args, cfg, arm, seeds, args.pa_trees) if _reads_v4(args, cfg) else None
    if v4 is None:
        log("baseline: fitted in-process (--refit-baseline, or a smoke without a v4 record)")

    fold = load_fold(CACHE, cfg.months, FEATS_HEAD, derive=L.add_nmdelay, check_counts=not cfg.smoke,
                     label=f"holdout (months {HOLDOUT})")
    X, y, dlt, proxy, sp = fold["X"], fold["y"], fold["dlt"], fold["proxy"], fold["sp"]
    te, tr, fit, es = fold["te"], fold["tr"], fold["fit"], fold["es"]
    ap_code, airports = fold["ap_code"], fold["airports"]
    y_te, proxy_te, sp_te, ap_te = y[te], proxy[te], sp[te], ap_code[te]
    assert fold["feats"][:nb] == list(L.FEATS) and fold["feats"][nb:] == [L.NMDELAY]
    assert np.array_equal(X[:, nb], (proxy - sp).astype(np.float32)), "the nmdelay column is not proxy - sp"
    base, cols = fold["base"], {}
    write = lambda: _write_preds(preds_path, base, cols)
    timing = {}

    # ---- baseline: the shipped v4 arm, read from its record or refit on the first nb columns ----
    t = time.time()
    b, b_record, sd, sd_source = resolve_baseline(args, cfg, fold, X[:, :nb], arm, seeds, args.pa_trees, v4,
                                                  "baseline", cols, write)
    timing["baseline_s"] = round(time.time() - t, 1)
    base_yhat = taxi_time(proxy_te, b["delta"])

    # ---- the head: the fill label on the training rows, one fit, p on the holdout rows ----
    t = time.time()
    label = L.fill_label(y, sp, tr)
    p, head = L.fit_fill_head(X, label, tr, fit, es, X[te], params_head, cfg.nest, cfg.patience,
                              n_features=X.shape[1])
    timing["head_s"] = round(time.time() - t, 1)
    del X
    fold["X"] = None
    gc.collect()
    fill_te = fill_mask(y_te, sp_te)
    y_mix = L.mix_fill(p, sp_te, base_yhat)
    treat = proxy_te - y_mix                       # the delta form: max(proxy - treat, 1) recovers the mixture
    cols["p"], cols["treatment"] = p, treat
    write()
    nonpos = sp_te <= 0.0
    holdout = dict(n_rows=int(te.sum()), fill_share=float(fill_te.mean()), n_fills=int(fill_te.sum()),
                   mean_p=float(p.mean()), auc=L.rank_auc(p, fill_te),
                   p_quantiles={str(q): float(np.percentile(p, q)) for q in (1, 10, 50, 90, 99)},
                   p_share_above_half=float((p > 0.5).mean()),
                   mean_p_on_fills=float(p[fill_te].mean()) if fill_te.any() else None,
                   mean_p_on_nonfills=float(p[~fill_te].mean()) if (~fill_te).any() else None,
                   n_sp_nonpositive=int(nonpos.sum()),
                   mean_p_on_sp_nonpositive=float(p[nonpos].mean()) if nonpos.any() else None,
                   treatment_floor_bind_rate=float((y_mix < 1.0).mean()),
                   mixture_shift_vs_baseline_mean_s=float((y_mix - base_yhat).mean()),
                   mixture_shift_vs_baseline_rms_s=float(np.sqrt(((y_mix - base_yhat) ** 2).mean())))
    fmt = lambda v: f"{v:.4f}" if v is not None else "n/a"
    log(f"head on the {holdout['n_rows']:,} holdout rows: AUC {holdout['auc']:.4f}; mean p {holdout['mean_p']:.4f} vs "
        f"fill share {holdout['fill_share']:.4f} ({holdout['n_fills']:,} fills); mean p on fills "
        f"{fmt(holdout['mean_p_on_fills'])}, on non-fills {fmt(holdout['mean_p_on_nonfills'])}; "
        f"{holdout['n_sp_nonpositive']:,} rows with sp <= 0 carry mean p {fmt(holdout['mean_p_on_sp_nonpositive'])}; "
        f"mixture shift vs the baseline RMS {holdout['mixture_shift_vs_baseline_rms_s']:.2f} s; the treatment's floor "
        f"binds on {100 * holdout['treatment_floor_bind_rate']:.3f}%   [{timing['head_s']:.0f}s]; predictions -> "
        f"{preds_path}")

    # ---- scoring: pooled, per airport, the two subsets, LIRF alone, reliability ----
    t = time.time()
    arms = {"baseline": b["delta"], "treatment": treat}
    definitions = {"baseline": f"the v4 arm {arm} over seeds {list(seeds)} ({b_record['source']})",
                   "treatment": f"baseline + fill head: the mixture p * sp + (1 - p) * max(proxy - baseline, 1), "
                                f"head best_iter {head['best_iter']:,} -> n_ref {head['n_ref']:,}; stored as "
                                "proxy - mixture"}
    se = squared_errors(arms, y_te, proxy_te)
    rmses = arm_rmses(se, ap_te, airports)
    pairs, present = score_pairs(se, rmses, [("treatment", "baseline")], ap_te, airports, cfg.n_boot, sd)
    pooled = pairs["treatment_vs_baseline"]
    fill_def = f"fill: |y - sp| <= {L.FILL_TOL_S:.0f} s (holdout rows)"
    nonfill_def = f"non-fill: |y - sp| > {L.FILL_TOL_S:.0f} s (holdout rows)"
    subsets = {"fill": subset_record(fill_def, fill_te, se, cfg.n_boot),
               "nonfill": subset_record(nonfill_def, ~fill_te, se, cfg.n_boot)}
    assert subsets["fill"]["n_rows"] + subsets["nonfill"]["n_rows"] == int(te.sum()), "fill + non-fill != holdout"
    lirf = None
    if "LIRF" in airports:
        m = ap_te == airports.index("LIRF")
        lirf = {"n_rows": int(m.sum()), "share_of_holdout_rows": float(m.mean()),
                "rmse": {k: rmses[k]["per_airport"].get("LIRF") for k in arms},
                "pair": pooled["per_airport"].get("LIRF"),
                "fill": subset_record(f"LIRF, {fill_def}", m & fill_te, se, cfg.n_boot),
                "nonfill": subset_record(f"LIRF, {nonfill_def}", m & ~fill_te, se, cfg.n_boot)}
    rel = reliability_deciles(p, fill_te)
    timing["scoring_s"] = round(time.time() - t, 1)

    # ---- tables ----
    _log_tables(arms, rmses, pairs, present, ap_te, definitions, sd,
                f"AMENDMENT 16, fold A matched rows ({te.sum():,}), holdout months {HOLDOUT}   "
                + (L.SMOKE_BANNER if cfg.smoke else f"head best_iter {head['best_iter']:,} -> n_ref {head['n_ref']:,}"))
    log(f"seed sd source: {sd_source}; the head is single-seed (Amendment 16: seed {L.ES_SEED}), so the 2 x seed sd "
        "rule reads the baseline's seed sd")
    _log_subset("fill subset", subsets["fill"])
    _log_subset("non-fill subset", subsets["nonfill"])
    if lirf is not None:
        lp = lirf["pair"]
        log(f"LIRF alone: {lirf['n_rows']:,} rows: baseline {lirf['rmse']['baseline']:.2f}, treatment "
            f"{lirf['rmse']['treatment']:.2f}"
            + (f", gain {lp['gain_s']:+.2f} s, paired CI [{lp['ci95'][0]:+.2f}, {lp['ci95'][1]:+.2f}]" if lp else ""))
        _log_subset("LIRF fill subset", lirf["fill"])
        _log_subset("LIRF non-fill subset", lirf["nonfill"])
    log(f"reliability deciles of p on the {rel['n_rows']:,} holdout rows (equal-count bins; mean p vs realised "
        "fill rate):")
    log(f"{'decile':>7s}{'n':>9s}{'p range':>22s}{'mean p':>9s}{'fill rate':>11s}")
    for d in rel["deciles"]:
        log(f"{d['decile']:7d}{d['n']:9,d}    [{d['p_min']:.4f}, {d['p_max']:.4f}]{d['mean_p']:9.4f}{d['fill_rate']:11.4f}")
    log(f"   overall: mean p {rel['mean_p']:.4f} vs fill rate {rel['fill_rate']:.4f}")

    # ---- the record ----
    result = _record_head("fillhead", cfg, started, fold, {"amendment": "16"})
    result.update({
        "config": {"params": cfg.params, "params_head": params_head, "seeds": list(seeds), "es_seed": L.ES_SEED,
                   "max_rounds": cfg.nest, "patience": cfg.patience, "baseline_arm": arm, "pa_trees": args.pa_trees,
                   "pa_tree_rule": pa_rule, "pa_min_rows": L.PA_MIN_ROWS, "pa_tree_floor": cfg.pa_floor,
                   "blend": L.BLEND, "n_boot": cfg.n_boot, "boot_seed": BOOT_SEED, "features": fold["feats"],
                   "n_features": len(fold["feats"]), "features_regressor": list(L.FEATS), "nmdelay": L.NMDELAY,
                   "fill_tolerance_s": L.FILL_TOL_S,
                   "mixture": "y_hat = p * sp + (1 - p) * max(proxy - delta_hat_baseline, 1); the parquet's "
                              "`treatment` = proxy - y_hat (the delta form)",
                   "n_deciles": N_DECILES},
        "baseline": b_record,
        "head": {**head, "params": params_head, "features": fold["feats"], "seeds": [params_head["seed"]],
                 "holdout_auc": holdout["auc"], "timing_s": {"head_s": timing["head_s"]}},
        "holdout": holdout,
        "arms": _arm_records(arms, rmses, definitions, proxy_te),
        "seed_sd": {"sd": sd, "source": sd_source, "treatment_sd": None,
                    "baseline_values": b_record.get("single_seed_rmses"), "ddof": 1,
                    "note": "the head is single-seed (Amendment 16: seed 0), so the treatment has no seed sd of "
                            "its own; the 2 x seed sd rule reads the baseline's seed sd"},
        "pairs": pairs,
        "subsets": subsets,
        "lirf": lirf,
        "reliability": rel,
        "timing_s": timing,
        "preds_parquet": str(preds_path), "log": str(log_path),
    })
    _write_json(json_path, result)
    log(f"json -> {json_path}   wall {result['wall_s']:.0f}s   peak RSS {result['peak_rss_gb']:.2f} GB")
    return 0


# =============================================================================================
# Amendment 19: arm D (--dayfeats) and arm Y (--ytarget)
# =============================================================================================

def _log_bands(bands: dict, arm_names, pair_keys, ref: str) -> None:
    log(f"|delta| bands (19.0; left-closed, right-open; over10 = 10to20 + gt20); RMSE per arm on the band's rows, "
        f"paired gain vs {ref} with its 95% CI (* = excludes zero):")
    aw = max(11, max(len(a) for a in arm_names) + 1)
    log(f"{'band':8s}{'n':>9s}{'rows%':>7s}{'SSE%':>7s}" + "".join(f"{a:>{aw}s}" for a in arm_names)
        + "".join(f"{k:>30s}" for k in pair_keys))
    for name, rec in bands.items():
        if not rec["n_rows"]:
            log(f"{name:8s}{0:9,d}   no rows")
            continue
        sse = rec.get(f"share_of_{ref}_sse")
        line = (f"{name:8s}{rec['n_rows']:9,d}{100 * rec['share_of_holdout_rows']:7.2f}"
                + (f"{100 * sse:7.1f}" if sse is not None else f"{'n/a':>7s}")
                + "".join(f"{rec['rmse'][a]:{aw}.2f}" for a in arm_names))
        for k in pair_keys:
            pr = (rec["pairs"] or {}).get(k)
            line += (f"   {pr['gain_s']:+8.2f} [{pr['ci95'][0]:+7.2f}, {pr['ci95'][1]:+7.2f}]{'*' if pr['excludes_zero'] else ' '}"
                     if pr else f"{'no interval (one row)':>30s}")
        log(line)


def _design(args) -> tuple:
    """(base feature list, queue flag, the record reader) of the current best design: FEATS with
    the v4 record, or FEATS + QUEUE_FEATS with the queue fold's record under --queue."""
    queue = bool(args.queue)
    return (FEATS_QUEUE if queue else list(L.FEATS)), queue


def _baseline_record(args, cfg, arm, seeds, queue: bool):
    if not _reads_record(args, cfg, queue):
        log("baseline: fitted in-process (--refit-baseline, or a smoke without a record)")
        return None
    return _queue_record(args, cfg, arm, seeds, args.pa_trees) if queue else _v4_record(args, cfg, arm, seeds, args.pa_trees)


#: a BLOCK arm: the current best design as the baseline, the same procedure on the design plus a
#: cached feature block as the treatment. Amendment 19.1 (arm D, the airport-day block) and
#: Amendment 22 (arm F, the record-ordering block) are the same experiment on different columns,
#: so they are one runner: a copy would drift, and the intervals must be comparable.
#: `cache_attr` is the MODULE ATTRIBUTE NAME, not the path: the runner resolves it at call time
#: (globals()[...]), so a test's monkeypatch of DCACHE / OCACHE - or any later re-pointing - is
#: seen by the arm. A frozen path here read the real 152,250-row cache under a 600-row fixture.
BLOCK_ARMS = {
    "day": SimpleNamespace(name="day", feats=DAY_FEATS, cache_attr="DCACHE", loader_kw="dcache",
                           amendment="19.1", title="AMENDMENT 19.1 ARM D", label="Amendment 19.1 arm D",
                           min_airports=DAY_MIN_AIRPORTS, clause="19.3"),
    "order": SimpleNamespace(name="order", feats=ORDER_FEATS, cache_attr="OCACHE", loader_kw="ocache",
                             amendment="22", title="AMENDMENT 22 ARM F", label="Amendment 22 arm F",
                             min_airports=None, clause="22.3"),
}


def run_block(args, cfg, paths, started, block) -> int:
    """One BLOCK arm (19.1 arm D / 22 arm F): baseline = the current best design from its record,
    treatment = the same procedure on the design plus `block.feats`, best_iter re-found."""
    json_path, log_path, preds_path = paths
    seeds, arm = tuple(args.seeds), args.baseline
    base_feats, queue = _design(args)
    mode = f"queue_{block.name}" if queue else block.name
    block_cache = globals()[block.cache_attr]          # resolved now, never frozen at import
    feats, nb = base_feats + block.feats, len(base_feats)
    up = f"{block.name.upper()}_FEATS"
    pa_rule = L.PA_TREE_RULE if args.pa_trees == "share" else L.PA_TREE_RULE_ES
    design = f"{arm}{' + QUEUE_FEATS' if queue else ''}"
    log(f"{block.label}: baseline = {design} over seeds {list(seeds)} ({nb} features, "
        f"{'the queue record' if queue else 'the v4 record'}); treatment = baseline + {len(block.feats)} {up} "
        f"({len(feats)} features), best_iter re-found; per-airport rule [{args.pa_trees}] {pa_rule} (A3 only); "
        f"bootstrap {cfg.n_boot:,} draws seed {BOOT_SEED}; bands {[b[0] for b in DELTA_BANDS]} + over10 "
        f"(|delta| >= {OVER10_S:.0f} s)"
        + (f"; {block.clause} needs >= {block.min_airports} of 10 airports" if block.min_airports else
           f"; {block.clause} has no airport-count clause (the over10 band's own interval is the mechanism)"))
    rec = _baseline_record(args, cfg, arm, seeds, queue)

    fold = load_fold(CACHE, cfg.months, feats, qcache=QCACHE if queue else None,
                     **{block.loader_kw: block_cache},
                     check_counts=not cfg.smoke, label=f"holdout (months {HOLDOUT})")
    X, y, dlt, proxy = fold["X"], fold["y"], fold["dlt"], fold["proxy"]
    te, tr = fold["te"], fold["tr"]
    ap_code, airports = fold["ap_code"], fold["airports"]
    y_te, proxy_te, dlt_te, ap_te = y[te], proxy[te], dlt[te], ap_code[te]
    assert fold["feats"][:nb] == base_feats and fold["feats"][nb:] == block.feats
    base, cols = fold["base"], {}
    write = lambda: _write_preds(preds_path, base, cols)
    timing = {}

    # ---- baseline: the current best design, from its record or refit on the first nb columns ----
    t = time.time()
    b, b_record, sd, sd_source = resolve_baseline(args, cfg, fold, X[:, :nb], arm, seeds, args.pa_trees, rec,
                                                  "baseline", cols, write)
    timing["baseline_s"] = round(time.time() - t, 1)

    # ---- treatment: the same procedure on base_feats + the block ----
    t = time.time()

    def on_seed(s, pred):
        cols[f"delta_hat_seed{s}"] = pred
        write()
    tr_arm = fit_arm(fold, X, cfg.params, cfg.nest, cfg.patience, cfg.pa_floor, seeds, per_airport=(arm == "A3"),
                     pa_trees=args.pa_trees, name="treatment", on_seed=on_seed)
    if arm == "A3":
        for s in seeds:
            cols[f"pa_seed{s}"] = tr_arm["pa_by_seed"][s]
        cols["pa_fitted"] = tr_arm["fitted"]
    cols["treatment"] = tr_arm["delta"]
    write()
    timing["treatment_s"] = round(time.time() - t, 1)
    t_sd = seed_sd(tr_arm["single_rmse"].values()) if len(seeds) == 3 else None
    if sd is None and t_sd is not None:
        sd, sd_source = t_sd, "treatment single seeds"
    del X
    fold["X"] = None
    gc.collect()

    # ---- scoring: pooled, per airport, the 6-of-10 count, the |delta| bands ----
    t = time.time()
    arms = {"baseline": b["delta"], "treatment": tr_arm["delta"]}
    definitions = {"baseline": f"{design} over seeds {list(seeds)} ({b_record['source']})",
                   "treatment": f"baseline + {up}, best_iter {tr_arm['best_iter']:,} -> n_ref {tr_arm['n_ref']:,}"}
    pair_list = [("treatment", "baseline")]
    se = squared_errors(arms, y_te, proxy_te)
    rmses = arm_rmses(se, ap_te, airports)
    pairs, present = score_pairs(se, rmses, pair_list, ap_te, airports, cfg.n_boot, sd)
    p = pairs["treatment_vs_baseline"]
    if block.min_airports:
        p[f"at_least_{block.min_airports}_airports"] = bool(p["airports_improving"] >= block.min_airports)
    bands = band_records(dlt_te, se, cfg.n_boot, pair_list, "baseline")
    timing["scoring_s"] = round(time.time() - t, 1)

    # ---- tables ----
    _log_tables(arms, rmses, pairs, present, ap_te, definitions, sd,
                f"{block.title} [{mode}], fold A matched rows ({te.sum():,}), holdout months {HOLDOUT}   "
                + (L.SMOKE_BANNER if cfg.smoke else f"treatment n_ref {tr_arm['n_ref']:,} (best_iter {tr_arm['best_iter']:,})"))
    log(f"seed sd source: {sd_source}   treatment's own seed sd: " + (f"{t_sd:.4f} s" if t_sd is not None else "n/a"))
    log(f"airports improving {p['airports_improving']} / {p['n_airports']}"
        + (f" (>= {block.min_airports}: {p[f'at_least_{block.min_airports}_airports']})" if block.min_airports else ""))
    _log_bands(bands, list(arms), ["treatment_vs_baseline"], "baseline")
    o10 = bands["over10"]["pairs"]["treatment_vs_baseline"] if bands["over10"]["pairs"] else None
    log(f"over10 (the |delta| > 10 min bands of {block.clause}): "
        + (f"gain {o10['gain_s']:+.2f} s, paired CI [{o10['ci95'][0]:+.2f}, {o10['ci95'][1]:+.2f}]" if o10 else "no interval"))

    # ---- the record ----
    result = _record_head(mode, cfg, started, fold, {"amendment": block.amendment})
    result.update({
        "config": {"params": cfg.params, "seeds": list(seeds), "es_seed": L.ES_SEED, "max_rounds": cfg.nest,
                   "patience": cfg.patience, "baseline_arm": arm, "queue": queue, "design_baseline": design,
                   "pa_trees": args.pa_trees, "pa_tree_rule": pa_rule, "pa_min_rows": L.PA_MIN_ROWS,
                   "pa_tree_floor": cfg.pa_floor, "blend": L.BLEND, "n_boot": cfg.n_boot, "boot_seed": BOOT_SEED,
                   "features": fold["feats"], "n_features": len(fold["feats"]),
                   f"{block.name}_feats": block.feats, f"{block.name}_cache": str(block_cache),
                   "queue_feats": QUEUE_FEATS if queue else None,
                   "queue_cache": str(QCACHE) if queue else None, "min_airports": block.min_airports,
                   "bands": {name: [lo, hi] for name, lo, hi in DELTA_BANDS}, "over10_s": OVER10_S},
        "baseline": b_record,
        "treatment": {"best_iter": tr_arm["best_iter"], "n_ref": tr_arm["n_ref"],
                      "es_rmse_taxi_time_NOT_A_RESULT": tr_arm["info"]["es_rmse_taxi_time_NOT_A_RESULT"],
                      "es_proxy_only_rmse": tr_arm["info"]["es_proxy_only_rmse"],
                      "single_seed_rmses": {str(s): r for s, r in tr_arm["single_rmse"].items()},
                      "per_airport_models": tr_arm["pa_info"], "timing_s": tr_arm["timing"]},
        "arms": _arm_records(arms, rmses, definitions, proxy_te),
        "seed_sd": {"sd": sd, "source": sd_source,
                    "treatment_values": {str(s): r for s, r in tr_arm["single_rmse"].items()}, "treatment_sd": t_sd,
                    "baseline_values": b_record.get("single_seed_rmses"), "ddof": 1},
        "pairs": pairs,
        "bands": bands,
        "timing_s": timing,
        "preds_parquet": str(preds_path), "log": str(log_path),
    })
    _write_json(json_path, result)
    log(f"json -> {json_path}   wall {result['wall_s']:.0f}s   peak RSS {result['peak_rss_gb']:.2f} GB")
    return 0


def run_day(args, cfg, paths, started) -> int:
    """Amendment 19.1 arm D: the airport-day regime block."""
    return run_block(args, cfg, paths, started, BLOCK_ARMS["day"])


def run_order(args, cfg, paths, started) -> int:
    """Amendment 22 arm F: the record-ordering block."""
    return run_block(args, cfg, paths, started, BLOCK_ARMS["order"])


def run_ytarget(args, cfg, paths, started) -> int:
    json_path, log_path, preds_path = paths
    seeds, arm = tuple(args.seeds), args.baseline
    feats, queue = _design(args)
    mode = "queue_ytarget" if queue else "ytarget"
    pa_rule = L.PA_TREE_RULE if args.pa_trees == "share" else L.PA_TREE_RULE_ES
    design = f"{arm}{' + QUEUE_FEATS' if queue else ''}"
    log(f"Amendment 19.2 arm Y: delta arm = {design} over seeds {list(seeds)} ({len(feats)} features, "
        f"{'the queue record' if queue else 'the v4 record'}); Y = the same design, configuration and seeds with "
        f"target {L.TARGET_Y} (y_hat_Y = max(prediction, 1), no proxy anchor); blend = {YBLEND} y_hat_Y + "
        f"{1 - YBLEND} y_hat_delta; per-airport rule [{args.pa_trees}] {pa_rule} (A3 only); named airports "
        f"{list(YTARGET_NAMED_AIRPORTS)}; bootstrap {cfg.n_boot:,} draws seed {BOOT_SEED}; bands "
        f"{[b[0] for b in DELTA_BANDS]} + over10")
    rec = _baseline_record(args, cfg, arm, seeds, queue)

    fold = load_fold(CACHE, cfg.months, feats, qcache=QCACHE if queue else None, check_counts=not cfg.smoke,
                     label=f"holdout (months {HOLDOUT})")
    X, y, dlt, proxy = fold["X"], fold["y"], fold["dlt"], fold["proxy"]
    te, tr = fold["te"], fold["tr"]
    ap_code, airports = fold["ap_code"], fold["airports"]
    y_te, proxy_te, dlt_te, ap_te = y[te], proxy[te], dlt[te], ap_code[te]
    assert fold["feats"] == feats
    base, cols = fold["base"], {}
    write = lambda: _write_preds(preds_path, base, cols)
    timing = {}

    # ---- the delta arm: from its record, or refit in-process ----
    t = time.time()
    b, b_record, sd, sd_source = resolve_baseline(args, cfg, fold, X, arm, seeds, args.pa_trees, rec, "baseline",
                                                  cols, write)
    timing["baseline_s"] = round(time.time() - t, 1)

    # ---- the Y arm: the same procedure with target y ----
    t = time.time()

    def on_seed(s, pred):
        cols[f"y_seed{s}"] = pred
        write()
    y_arm = fit_arm(fold, X, cfg.params, cfg.nest, cfg.patience, cfg.pa_floor, seeds, per_airport=(arm == "A3"),
                    pa_trees=args.pa_trees, name="Y", on_seed=on_seed, target=L.TARGET_Y)
    if arm == "A3":
        for s in seeds:
            cols[f"y_pa_seed{s}"] = y_arm["pa_by_seed"][s]
        cols["y_pa_fitted"] = y_arm["fitted"]
    del X
    fold["X"] = None
    gc.collect()
    yh_base = taxi_time(proxy_te, b["delta"])
    yh_y = taxi_time(proxy_te, y_arm["delta"], target=L.TARGET_Y)
    yh_blend = blend_taxi_times(yh_y, yh_base, YBLEND)
    cols["Y"], cols["blend"] = proxy_te - yh_y, proxy_te - yh_blend      # the delta form every arm's column uses
    write()
    timing["y_arm_s"] = round(time.time() - t, 1)
    y_sd = seed_sd(y_arm["single_rmse"].values()) if len(seeds) == 3 else None
    if sd is None and y_sd is not None:
        sd, sd_source = y_sd, "Y single seeds"

    # ---- scoring: pooled, per airport (LTFM / EDDM named), the |delta| bands, both pairs ----
    t = time.time()
    arms = {"baseline": b["delta"], "Y": cols["Y"], "blend": cols["blend"]}
    definitions = {"baseline": f"the delta arm {design} over seeds {list(seeds)} ({b_record['source']})",
                   "Y": f"target y, best_iter {y_arm['best_iter']:,} -> n_ref {y_arm['n_ref']:,}, mean over seeds "
                        f"{list(seeds)}, y_hat = max(prediction, 1); stored as proxy - y_hat",
                   "blend": f"{YBLEND} y_hat_Y + {1 - YBLEND} y_hat_delta; stored as proxy - y_hat"}
    pair_list = [("blend", "baseline"), ("Y", "baseline")]
    se = squared_errors(arms, y_te, proxy_te)
    rmses = arm_rmses(se, ap_te, airports)
    pairs, present = score_pairs(se, rmses, pair_list, ap_te, airports, cfg.n_boot, sd)
    for new, ref in pair_list:
        pairs[f"{new}_vs_{ref}"]["named_airports"] = {
            a: bool(rmses[new]["per_airport"][a] < rmses[ref]["per_airport"][a])
            for a in YTARGET_NAMED_AIRPORTS if a in rmses[ref]["per_airport"]}
    bands = band_records(dlt_te, se, cfg.n_boot, pair_list, "baseline")
    timing["scoring_s"] = round(time.time() - t, 1)

    # ---- tables ----
    _log_tables(arms, rmses, pairs, present, ap_te, definitions, sd,
                f"AMENDMENT 19.2 ARM Y [{mode}], fold A matched rows ({te.sum():,}), holdout months {HOLDOUT}   "
                + (L.SMOKE_BANNER if cfg.smoke else f"Y n_ref {y_arm['n_ref']:,} (best_iter {y_arm['best_iter']:,})"))
    log(f"seed sd source: {sd_source}   Y's own seed sd: " + (f"{y_sd:.4f} s" if y_sd is not None else "n/a"))
    for key in ("Y_vs_baseline", "blend_vs_baseline"):
        log(f"named airports (19.2) for {key}: " + ", ".join(
            f"{a} {rmses['baseline']['per_airport'][a]:.2f} -> {rmses[key.split('_vs_')[0]]['per_airport'][a]:.2f} "
            f"({'improves' if v else 'does not'})" for a, v in pairs[key]["named_airports"].items()))
    _log_bands(bands, list(arms), ["Y_vs_baseline", "blend_vs_baseline"], "baseline")

    # ---- the record ----
    result = _record_head(mode, cfg, started, fold, {"amendment": "19.2"})
    result.update({
        "config": {"params": cfg.params, "seeds": list(seeds), "es_seed": L.ES_SEED, "max_rounds": cfg.nest,
                   "patience": cfg.patience, "baseline_arm": arm, "queue": queue, "design_baseline": design,
                   "target": L.TARGET_Y, "blend": YBLEND, "pa_trees": args.pa_trees, "pa_tree_rule": pa_rule,
                   "pa_min_rows": L.PA_MIN_ROWS, "pa_tree_floor": cfg.pa_floor, "n_boot": cfg.n_boot,
                   "boot_seed": BOOT_SEED, "features": fold["feats"], "n_features": len(fold["feats"]),
                   "queue_feats": QUEUE_FEATS if queue else None, "named_airports": list(YTARGET_NAMED_AIRPORTS),
                   "bands": {name: [lo, hi] for name, lo, hi in DELTA_BANDS}, "over10_s": OVER10_S,
                   "columns": "y_seed{s}: the per-seed y predictions (y scale); Y and blend: proxy - y_hat (the "
                              "delta form), so max(proxy - col, 1) recovers the taxi time"},
        "baseline": b_record,
        "y_arm": {"target": L.TARGET_Y, "best_iter": y_arm["best_iter"], "n_ref": y_arm["n_ref"],
                  "es_rmse_taxi_time_NOT_A_RESULT": y_arm["info"]["es_rmse_taxi_time_NOT_A_RESULT"],
                  "es_proxy_only_rmse": y_arm["info"]["es_proxy_only_rmse"],
                  "single_seed_rmses": {str(s): r for s, r in y_arm["single_rmse"].items()},
                  "per_airport_models": y_arm["pa_info"], "timing_s": y_arm["timing"],
                  "floor_bind_rate": float((np.asarray(y_arm["delta"], dtype="float64") < 1.0).mean())},
        "arms": _arm_records(arms, rmses, definitions, proxy_te),
        "seed_sd": {"sd": sd, "source": sd_source, "y_values": {str(s): r for s, r in y_arm["single_rmse"].items()},
                    "y_sd": y_sd, "baseline_values": b_record.get("single_seed_rmses"), "ddof": 1},
        "pairs": pairs,
        "bands": bands,
        "timing_s": timing,
        "preds_parquet": str(preds_path), "log": str(log_path),
    })
    _write_json(json_path, result)
    log(f"json -> {json_path}   wall {result['wall_s']:.0f}s   peak RSS {result['peak_rss_gb']:.2f} GB")
    return 0


# =============================================================================================
# The unified all-rows arm: --ytarget --all-rows
# =============================================================================================

def _subset_report(name: str, definition: str, mask, se: dict, ap_te, airports, pair_list, n_boot, sd) -> tuple:
    """One subset of the holdout: its row count and share, every arm's RMSE, the per-airport
    RMSEs, and score_pairs' paired intervals on the subset's rows. Returns (record, rmses, pairs,
    present) - the last three for the log tables."""
    mask = np.asarray(mask, dtype=bool)
    se_s = {k: v[mask] for k, v in se.items()}
    rm = arm_rmses(se_s, ap_te[mask], airports)
    pr, present = score_pairs(se_s, rm, pair_list, ap_te[mask], airports, n_boot, sd)
    rec = {"definition": definition, "n_rows": int(mask.sum()), "share_of_holdout_rows": float(mask.mean()),
           "rmse": {k: rm[k]["pooled"] for k in se}, "per_airport": {k: rm[k]["per_airport"] for k in se},
           "pairs": pr, "airports": [a for _, a in present]}
    return rec, rm, pr, present


def run_allrows(args, cfg, paths, started) -> int:
    json_path, log_path, preds_path = paths
    seeds, arm = tuple(args.seeds), args.baseline
    queue, day = bool(args.queue), bool(args.dayfeats)
    mode = mode_of(args)
    base_feats = list(L.FEATS) + (QUEUE_FEATS if queue else []) + (DAY_FEATS if day else [])
    feats = base_feats + [IS_UNMATCHED]
    nb = len(list(L.FEATS) + (QUEUE_FEATS if queue else []))       # the matched pipeline's design width
    weights = tuple(args.unmatched_weight)
    design = f"{arm}{' + QUEUE_FEATS' if queue else ''}"
    log(f"UNIFIED all-rows arm: ONE y-target regressor on matched + unmatched training rows, design FEATS"
        f"{' + QUEUE_FEATS' if queue else ''}{' + DAY_FEATS' if day else ''} + {IS_UNMATCHED} ({len(feats)} features); "
        f"unmatched weights {list(weights)} (one arm each, seeds {list(seeds)}); y_hat = max(prediction, 1), no proxy; "
        f"pipeline = matched {design} ({'the queue record' if queue else 'the v4 record'}) + unmatched S0 of the "
        f"stratum record; blend {YBLEND}; bootstrap {cfg.n_boot:,} draws seed {BOOT_SEED}; 2026 weights "
        f"{W_MATCHED_2026:.6f} / {W_UNMATCHED_2026:.6f}; monsters y > {MONSTER_S:.0f} s; early stopping evaluated on the "
        f"MATCHED stopping rows only (Amendment 20.5: the unmatched stopping rows' monster errors decide the stop by noise)")
    rec = _baseline_record(args, cfg, arm, seeds, queue)
    strat_path = pathlib.Path(args.stratum_preds) if args.stratum_preds else STRATUM_PREDS
    if not strat_path.exists():
        raise FileNotFoundError(f"{strat_path} missing: the unmatched rows' pipeline prediction is the stratum fold's "
                                "S0 (run scripts/stratum_fold.py first, or pass --stratum-preds)")
    strat = pd.read_parquet(strat_path)

    fold = load_fold_all_rows(CACHE, UCACHE, cfg.months, feats, qcache=QCACHE if queue else None,
                              dcache=DCACHE if day else None, check_counts=not cfg.smoke, label=f"holdout (months {HOLDOUT})")
    X, y, dlt, proxy = fold["X"], fold["y"], fold["dlt"], fold["proxy"]
    te, tr, is_um = fold["te"], fold["tr"], fold["is_unmatched"]
    ap_code, airports = fold["ap_code"], fold["airports"]
    um_te = is_um[te]
    y_te, proxy_te, dlt_te, ap_te = y[te], proxy[te], dlt[te], ap_code[te]
    n_te, n_u = int(te.sum()), int(um_te.sum())
    if n_u < 2 or n_te - n_u < 2:
        raise ValueError(f"the holdout needs both strata: {n_te - n_u:,} matched, {n_u:,} unmatched rows")
    base = fold["base"]
    base_m = base[~um_te].reset_index(drop=True)
    base_u = base[um_te].reset_index(drop=True)
    cols = {}
    write = lambda: _write_preds(preds_path, base, {k: _pad_matched(v, um_te) for k, v in cols.items()})
    timing = {}

    # ---- the matched pipeline: the record, or refit in-process on the matched rows only ----
    t = time.time()
    mm = ~is_um
    fold_m = dict(te=te[mm], tr=tr[mm], fit=fold["fit"][mm], es=fold["es"][mm], y=y[mm], dlt=dlt[mm], proxy=proxy[mm],
                  ap_code=ap_code[mm], airports=airports, base=base_m)
    X_m = None if rec is not None else X[mm][:, :nb]
    b, b_record, sd, sd_source = resolve_baseline(args, cfg, fold_m, X_m, arm, seeds, args.pa_trees, rec, "baseline",
                                                  cols, write)
    del X_m, fold_m
    gc.collect()
    yhat_pipe = np.empty(n_te, dtype="float64")
    yhat_pipe[~um_te] = taxi_time(proxy_te[~um_te], b["delta"])
    ub = unmatched_baseline_from_stratum(strat, base_u)
    yhat_pipe[um_te] = ub["yhat"]
    cols["pipeline"] = yhat_pipe
    write()
    timing["pipeline_s"] = round(time.time() - t, 1)
    log(f"pipeline: matched rows from {b_record['source']}, unmatched rows from {strat_path} (fold A: "
        f"{ub['n_record_fold_a']:,} rows, {ub['n_joined']:,} joined by id, y and month identical)")

    # ---- the unified arms, one per weight ----
    arms, u_arms, definitions = {"pipeline": yhat_pipe}, {}, {}
    definitions["pipeline"] = f"matched: {design} ({b_record['source']}); unmatched: the stratum record's S0"
    es_eval = fold["es"] & ~is_um                     # 20.5: the stopping metric reads the matched stopping rows
    for W in weights:
        key = f"w{W:g}"
        t = time.time()
        w = L.row_weights(is_um, W)

        def on_seed(s, pred, key=key):
            cols[f"U_{key}_seed{s}"] = pred
            write()
        try:
            ua = fit_arm(fold, X, cfg.params, cfg.nest, cfg.patience, cfg.pa_floor, seeds, per_airport=False, pa_trees=None,
                         name=f"U_{key}", on_seed=on_seed, target=L.TARGET_Y, weight=w, es_eval=es_eval)
        except RuntimeError as e:                     # the stopping-set breakage guard: recorded, the run goes on
            log(f"[U_{key}] NOT FITTED: {e}")
            u_arms[key] = dict(target=L.TARGET_Y, weight=float(W), error=str(e), fitted=False,
                               wall_s=round(time.time() - t, 1))
            continue
        raw = np.asarray(ua["delta"], dtype="float64")
        yhat_u = np.maximum(raw, 1.0)                                   # y_hat = max(prediction, 1): nothing else
        yhat_bl = blend_taxi_times(yhat_u, yhat_pipe, YBLEND)
        cols[f"U_{key}"], cols[f"Ublend_{key}"] = yhat_u, yhat_bl
        write()
        arms[f"U_{key}"], arms[f"Ublend_{key}"] = yhat_u, yhat_bl
        definitions[f"U_{key}"] = (f"unified y-target regressor, unmatched weight {W:g}, best_iter {ua['best_iter']:,} -> "
                                   f"n_ref {ua['n_ref']:,}, mean over seeds {list(seeds)}, max(., 1)")
        definitions[f"Ublend_{key}"] = f"{YBLEND} U_{key} + {1 - YBLEND} pipeline"
        per_seed_total = {}
        for s in seeds:
            yh = np.maximum(np.asarray(ua["single"][s], dtype="float64"), 1.0)
            e = (y_te - yh) ** 2
            per_seed_total[str(s)] = total_rmse_2026(e[~um_te].mean(), e[um_te].mean())
        u_arms[key] = dict(target=L.TARGET_Y, weight=float(W), fitted=True, best_iter=ua["best_iter"], n_ref=ua["n_ref"],
                           n_es=ua["info"]["n_es"], n_es_months=ua["info"]["n_es_months"],   # 20.5: matched stopping rows / all
                           n_features=ua["n_features"], single_seed_rmses=str_keys(ua["single_rmse"]),
                           single_seed_rmses_total=per_seed_total,
                           seed_sd_total=(seed_sd(per_seed_total.values()) if len(seeds) == 3 else None),
                           es_rmse_taxi_time_NOT_A_RESULT=ua["info"]["es_rmse_taxi_time_NOT_A_RESULT"],
                           es_proxy_only_rmse=ua["info"]["es_proxy_only_rmse"], es_guard_rows=ua["info"].get("es_guard_rows"),
                           floor_bind_rate=float((raw < 1.0).mean()), timing_s=ua["timing"], wall_s=round(time.time() - t, 1))
        log(f"[U_{key}] done in {time.time() - t:.0f}s; floor binds on {100 * u_arms[key]['floor_bind_rate']:.3f}% of rows")
    del X
    fold["X"] = None
    gc.collect()
    fitted_w = [W for W in weights if u_arms[f"w{W:g}"].get("fitted")]
    if not fitted_w:
        _write_json(json_path, {**_record_head(mode, cfg, started, fold, {"amendment": UNIFIED_AMENDMENT}), "u_arms": u_arms,
                                "note": "no unified arm cleared the stopping-set breakage guard; nothing was scored"})
        log(f"STOP: no unified arm fitted; partial json -> {json_path}")
        return 2
    if sd is None:
        first = u_arms[f"w{fitted_w[0]:g}"]
        if first["seed_sd_total"] is not None:
            sd, sd_source = first["seed_sd_total"], f"U_w{fitted_w[0]:g} single seeds (2026 total)"

    # ---- scoring: the three subsets, the total, the bands ----
    t = time.time()
    se = {k: (y_te - v) ** 2 for k, v in arms.items()}
    pair_list = [(f"{kind}_w{W:g}", "pipeline") for W in fitted_w for kind in ("U", "Ublend")]
    exm = um_te & (y_te <= MONSTER_S)
    subsets, tables = {}, {}
    for name, mask, definition in (("matched", ~um_te, "rows with an NM off-block (the matched record's fold)"),
                                   ("unmatched", um_te, "rows without one (the stratum), pooled"),
                                   ("unmatched_exmonster", exm, f"unmatched rows with y <= {MONSTER_S:.0f} s (ex-monster)")):
        rec_s, rm, pr, present = _subset_report(name, definition, mask, se, ap_te, airports, pair_list, cfg.n_boot, sd)
        subsets[name], tables[name] = rec_s, (rm, pr, present)
    assert subsets["matched"]["n_rows"] + subsets["unmatched"]["n_rows"] == n_te, "matched + unmatched != holdout"
    assert subsets["unmatched_exmonster"]["n_rows"] <= subsets["unmatched"]["n_rows"]
    total = dict(definition="sqrt(w_m x MSE_matched + w_u x MSE_unmatched) at the 2026 scored-file shares "
                            "339,551 / 5,290 of 344,841, stratified paired row bootstrap",
                 w_matched=W_MATCHED_2026, w_unmatched=W_UNMATCHED_2026,
                 rmse={k: total_rmse_2026(v[~um_te].mean(), v[um_te].mean()) for k, v in se.items()},
                 pairs=stratified_paired_bootstrap({k: v[~um_te] for k, v in se.items()}, {k: v[um_te] for k, v in se.items()},
                                                   pair_list, cfg.n_boot, BOOT_SEED))
    bands = band_records(dlt_te[~um_te], {k: v[~um_te] for k, v in se.items()}, cfg.n_boot, pair_list, "pipeline")
    timing["scoring_s"] = round(time.time() - t, 1)

    # ---- tables ----
    for name in ("matched", "unmatched", "unmatched_exmonster"):
        rm, pr, present = tables[name]
        _log_tables(arms, rm, pr, present, ap_te[{"matched": ~um_te, "unmatched": um_te, "unmatched_exmonster": exm}[name]],
                    definitions, sd, f"UNIFIED ALL-ROWS ARM [{mode}] - {name.upper().replace('_', ' ')} rows "
                                     f"({subsets[name]['n_rows']:,}), holdout months {HOLDOUT}   "
                    + (L.SMOKE_BANNER if cfg.smoke else ""))
    log("=" * 100)
    log(f"TOTAL at the 2026 weights (w_m {W_MATCHED_2026:.6f}, w_u {W_UNMATCHED_2026:.6f}; stratified paired bootstrap):")
    for k, v in total["rmse"].items():
        log(f"  {k:14s} {v:10.4f}")
    for k, pr in total["pairs"].items():
        log(f"  {k:28s} gain {pr['gain_s']:+8.4f} s   95% CI [{pr['ci95'][0]:+8.4f}, {pr['ci95'][1]:+8.4f}]   "
            f"excludes 0: {pr['excludes_zero']}")
    log(f"ex-monster: {subsets['unmatched_exmonster']['n_rows']:,} of the {n_u:,} unmatched holdout rows have y <= "
        f"{MONSTER_S:.0f} s; seed sd source: {sd_source}")
    log(f"NaN pattern on the unmatched rows: {len(S.UNMATCHED_NAN_COLS)} AOBT_3-anchored columns "
        f"{S.UNMATCHED_NAN_COLS} + {len(S.UNMATCHED_FLT_COLS)} *_flt-derived columns NaN, checked on every unmatched cache file")
    _log_bands(bands, list(arms), [f"{new}_vs_{ref}" for new, ref in pair_list], "pipeline")

    # ---- the record ----
    result = _record_head(mode, cfg, started, fold, {"amendment": UNIFIED_AMENDMENT})
    result["fold"]["n_unmatched_test"] = n_u
    result["fold"]["n_unmatched_all"] = int((tr & is_um).sum())
    result.update({
        "config": {"params": cfg.params, "seeds": list(seeds), "es_seed": L.ES_SEED, "max_rounds": cfg.nest,
                   "patience": cfg.patience, "baseline_arm": arm, "queue": queue, "day": day, "design_baseline": design,
                   "target": L.TARGET_Y, "blend": YBLEND, "unmatched_weights": [float(W) for W in weights],
                   "pa_trees": args.pa_trees, "n_boot": cfg.n_boot, "boot_seed": BOOT_SEED, "features": fold["feats"],
                   "n_features": len(fold["feats"]), "is_unmatched_column": IS_UNMATCHED,
                   "unmatched_cache": str(UCACHE), "unmatched_nan_cols": list(S.UNMATCHED_NAN_COLS),
                   "unmatched_flt_cols": list(S.UNMATCHED_FLT_COLS), "w_matched_2026": W_MATCHED_2026,
                   "w_unmatched_2026": W_UNMATCHED_2026, "monster_s": MONSTER_S, "stratum_preds": str(strat_path),
                   "columns": "every arm column is a TAXI TIME (the unmatched rows have no proxy); U_w{W}_seed{s} the "
                              "per-seed y predictions before the 1 s floor; baseline_* the matched record in the delta "
                              "form, NaN on unmatched rows"},
        "baseline": b_record,
        "unmatched_baseline": {"source": str(strat_path), "arm": "S0 (the shipped fit_unmatched)",
                               "n_record_fold_a": ub["n_record_fold_a"], "n_joined": ub["n_joined"]},
        "u_arms": u_arms,
        "arms": {k: {"definition": definitions[k], "rmse_total_2026": total["rmse"][k],
                     "rmse_matched": subsets["matched"]["rmse"][k], "rmse_unmatched": subsets["unmatched"]["rmse"][k],
                     "rmse_unmatched_exmonster": subsets["unmatched_exmonster"]["rmse"][k]} for k in arms},
        "subsets": subsets,
        "total_2026": total,
        "bands": bands,
        "seed_sd": {"sd": sd, "source": sd_source, "baseline_values": b_record.get("single_seed_rmses"),
                    "u_arms_total_sd": {k: v.get("seed_sd_total") for k, v in u_arms.items()}, "ddof": 1},
        "timing_s": timing,
        "preds_parquet": str(preds_path), "log": str(log_path),
    })
    _write_json(json_path, result)
    log(f"json -> {json_path}   wall {result['wall_s']:.0f}s   peak RSS {result['peak_rss_gb']:.2f} GB")
    return 0


def str_keys(d: dict) -> dict:
    return {str(k): v for k, v in d.items()}


RUNNERS = {"v4": run_v4, "queue": run_queue, "catboost": run_catboost, "sweep": run_sweep, "confirm": run_confirm,
           "fillhead": run_fillhead, "day": run_day, "queue_day": run_day,
           "order": run_order, "queue_order": run_order, "ytarget": run_ytarget,
           "queue_ytarget": run_ytarget, **{m: run_allrows for m in ALLROWS_MODES}}


# =============================================================================================
# entry point
# =============================================================================================

def main(argv=None) -> int:
    global log
    args = parse_args(argv)
    mode = mode_of(args)
    started = dt.datetime.now(dt.timezone.utc)
    smoke = args.smoke
    params, nest, patience, months = dict(L.P), L.NEST, L.PATIENCE, None
    n_boot, pa_floor = N_BOOT, L.PA_TREE_FLOOR
    if smoke:
        params["learning_rate"] = SMOKE["learning_rate"]
        nest, patience, months = SMOKE["nest"], SMOKE["patience"], SMOKE["months"]
        n_boot, pa_floor = SMOKE["n_boot"], SMOKE["pa_floor"]
    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
    elif smoke:
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"lgbm_fold_{mode}_smoke_"))
    else:
        out_dir = None
    json_path, log_path, preds_path = output_paths(mode, out_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if preds_path is not None:
        preds_path.parent.mkdir(parents=True, exist_ok=True)
    log = _Log(log_path, suffix=RANKING_BANNER if mode == "sweep" else None)
    L.log = log                                     # the shared fitting code logs here too
    cfg = SimpleNamespace(smoke=smoke, params=params, nest=nest, patience=patience, months=months, n_boot=n_boot,
                          pa_floor=pa_floor)
    try:
        if smoke:
            log(L.SMOKE_BANNER)
        log(f"lgbm_fold [{mode}]  smoke={smoke}  json={json_path}  log={log_path}  preds={preds_path}")
        log(f"command: {COMMANDS[mode]}")
        log(f"estimate: {ESTIMATES[mode]}")
        rc = RUNNERS[mode](args, cfg, (json_path, log_path, preds_path), started)
    finally:
        if smoke:
            log(L.SMOKE_BANNER)
        log.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
