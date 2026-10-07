"""FM1's ONE config: the knob schema, the loader and the resolved, hashed, immutable view (plans/PREREG_fm1_2026_09_28.md §1).

Every knob is declared once, here, with its type, default, legal range or choice set, effect, evidence pointer (a repo file,
optionally `:line` or ` §section`) and implementation status. `docs/fm1/CONFIG_REFERENCE.md` is generated from this table
(prc/pipeline/fm1/refdoc.py) and `configs/fm1.yaml` states every knob at its default; tests/fm1/test_fm1_knobs.py and
test_fm1_refdoc.py fail on any gap in either direction.

Loud at the boundary (code-design rule 5): an unknown key, a wrong type (a YAML `true` is never an int), an out-of-range value,
an unknown choice, a path into `submissions/` or a splice key raises a named error carrying the dotted path. A knob a YAML
leaves out takes its documented default; the resolved values (all knobs) are what is hashed, so a stated default and an
omitted one hash the same and a changed value never does.

Overrides (`--set path=value`) are parsed as YAML scalars/lists, validated like the file, and recorded in the hash.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import errors as E

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1
STATUSES = ("implemented", "provisional", "not_ported", "schema_only")
KINDS = ("bool", "int", "float", "str", "choice", "path", "int_list", "str_list", "tz_map")
AIRPORT_RE = re.compile(r"^[A-Z]{4}$")
TEAM_RE = re.compile(r"^[a-z]+-[a-z]+$")
SPLICE_KEYS = ("base", "base_submission", "base_file", "splice", "previous_submission")
SUBMISSIONS_DIR = "submissions"
TEN = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
TEN_TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam",
          "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LIRF": "Europe/Rome",
          "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}
#: learners refused by name, with the reason (owner decision 5; FMA §4.6 cut list)
CUT_LEARNERS = {"xgboost": "is CUT (owner decision 5, 2026-09-28: not installed; first on the FMA §4.6 cut list)",
                "mlp": "is not a days 1-3 learner (the MLP enters only through the ADS-B stack's worker; a generic MLP "
                       "learner is selection-study arm A6, not built yet)"}
PREREG = "plans/PREREG_fm1_2026_09_28.md"
FMA = "reports/final_model/AUDIT_AND_PLAN_2026_09_28.md"


def _on(v) -> bool:
    return v is True


@dataclass(frozen=True)
class Knob:
    path: str
    kind: str
    default: Any
    effect: str
    evidence: str
    status: str = "implemented"
    lo: float | None = None
    hi: float | None = None
    choices: tuple = ()
    #: for a component switch: True when this value turns on the component (used by FM1Config.not_ported)
    activates: Callable[[Any], bool] | None = field(default=None, compare=False)
    unique: bool = False                 # list knobs: no repeats
    #: False only for an output location: it cannot change a prediction, so it stays out of the model hash (FM1 notes N6)
    hashed: bool = True
    #: The generation of a knob added after stored artefacts existed (1 = modules.eobt_clip.*, 2 = adsb.train_dropout.*,
    #: both 2026-10-02; 3 = modules.uad.*, 2026-10-05; 4 = modules.tail_cells.*, 2026-10-05); 0 = present from the
    #: start. With h = the highest generation
    #: holding a knob off its default, parse_config hashes the late knobs of generations 1 ... h and leaves later ones
    #: out, which is exactly the hash a config had when it was written: a new generation never moves a stored hash
    #: (BC-41). A late knob's default must be the behaviour before it existed (a module that is off); defaults and
    #: generations are pinned by tests/fm1/test_fm1_eobt_clip.py.
    late: int = 0


def _K(*a, **k) -> Knob:
    return Knob(*a, **k)


SCHEMA: tuple = (
    # ---- run ----------------------------------------------------------------------------------------------------
    _K("run.purpose", "choice", "forecaster",
       "What the run is for. `forecaster` = the documented model (no target-period information); `scored_period` = the "
       "board build for a scored period, which may read that period's published aggregates (turns MTX on when "
       "calibration.mtx.enabled is auto).", f"{PREREG} §2", choices=("forecaster", "scored_period")),
    _K("run.team", "str", "merry-quicksand",
       "Team name in the submission file name `<team>_v<N>.parquet` (the organiser's convention).",
       "scripts/build_submission.py"),
    _K("run.seeds", "int_list", [0, 1, 2],
       "Seeds of every stochastic learner; the matched and unmatched predictions are the mean over seeds. The early-"
       "stopping run always uses the first seed.", "scripts/lgbm_submit.py:179", lo=0, hi=2**31 - 1, unique=True),
    _K("run.threads", "int", 4,
       "Thread count of every learner (LightGBM num_threads, CatBoost thread_count). Fixed so a fit is bit-reproducible "
       "on this machine (AT3); changing it can change LightGBM's floating-point summation order.",
       "scripts/lgbm_fold.py:32", lo=1, hi=64),
    _K("run.artifacts_dir", "path", "artifacts/fm1",
       "Where fitted artefacts live, one directory per resolved-config hash; an existing one is never overwritten. An "
       "output location: it is NOT part of the config hash (it cannot change a prediction).",
       f"{PREREG} §3", hashed=False),
    # ---- data ---------------------------------------------------------------------------------------------------
    _K("data.raw_dir", "path", "data/raw",
       "Directory of the organiser's raw monthly parquet files (training months and the scored file).",
       "data/raw/FETCH_MANIFEST.json"),
    _K("data.training_glob", "str", "training_2025-*.parquet",
       "Glob (inside data.raw_dir) of the labelled training months.", "configs/pipeline.yaml"),
    _K("data.train_months", "int_list", list(range(1, 13)),
       "Calendar months (of the training files) that fit the models; a validation protocol replaces this per fold.",
       f"{FMA} §4.3", lo=1, hi=12, unique=True),
    _K("data.scored_file", "path", "data/raw/ranking.parquet",
       "The raw file whose departures are scored (it holds DEP and ARR rows; ARR rows feed the stand/queue features).",
       "configs/pipeline.yaml"),
    _K("data.template_file", "path", "data/raw/submitting.parquet",
       "The id template the scorer expects; build-submission writes exactly its ids, in its order.",
       "configs/pipeline.yaml"),
    _K("data.weather_dir", "path", "data/weather",
       "Frozen METAR archive (IEM) read through prc.weather; the weather block's as-of join reads it.",
       "scripts/stand_ab.py:1076"),
    _K("data.dayoff_tz", "choice", "UTC",
       "Time zone of the `dayoff` feature (take-off date minus scheduled date, clipped to [0, 1]) of the unmatched body. "
       "UTC reproduces every shipped file; `local` uses airports.tz.", f"{FMA} §2.2 G11", choices=("UTC", "local")),
    # ---- airports -----------------------------------------------------------------------------------------------
    _K("airports.codes", "str_list", list(TEN),
       "The airport registry. A scored row at an airport outside it is refused (input contract).",
       "reports/final_model/AUDIT_AND_PLAN_2026_09_28.md", unique=True),
    _K("airports.tz", "tz_map", dict(TEN_TZ),
       "IANA time zone per airport; must cover exactly airports.codes. Read by dayoff_tz=local (the provisional hgb body). "
       "MTX keys scored rows by UTC month although the published figures are local months (LIMITATIONS 8).",
       "scripts/analysis/mtx_source.py:24"),
    # ---- lanes --------------------------------------------------------------------------------------------------
    _K("lanes.matched", "bool", True,
       "Serve matched departures (AOBT_3 present): base model -> ADS-B stack -> CatBoost half-blend.",
       f"{FMA} §4.2"),
    _K("lanes.unmatched", "bool", True,
       "Serve unmatched departures (AOBT_3 absent) with the unmatched body.", f"{FMA} §4.2"),
    # ---- features -----------------------------------------------------------------------------------------------
    _K("features.queue", "bool", True,
       "Push-anchored surface-congestion block (12 q_* columns, stand_ab.build_queue).", "scripts/stand_ab.py:106"),
    _K("features.order", "bool", True,
       "Movement-order block (o_dev_mvt, o_dev_flt, o_n_line; stand_ab.build_order).", "scripts/stand_ab.py:967"),
    _K("features.weather", "bool", True,
       "METAR block (10 w_* columns, as-of the row's AOBT_3; stand_ab.build_weather).", "scripts/stand_ab.py:1096"),
    _K("features.orl_route_length", "bool", True,
       "OSM stand->runway route-length block (6 orl_* columns; orl_build.attach, a static lookup).",
       "plans/PREREG_orl_osm_route_length_2026_09_23.md"),
    _K("features.orl_routes_file", "path", "data/osm/orl_routes.parquet",
       "The ORL route table built from OpenStreetMap (ODbL) by scripts/analysis/orl_build.py build.",
       "data/osm/manifest.json"),
    _K("features.adsb", "bool", True,
       "12-month ADS-B block (33 features + coverage + adsb_day_present) joined by MVT id from the R12 tables; NaN where "
       "a row has no table row.", "plans/PREREG_v35_r12_2026_09_26.md"),
    # ---- adsb ---------------------------------------------------------------------------------------------------
    _K("adsb.sources.adsb_lol", "bool", True,
       "adsb.lol globe_history (ODbL 1.0), the only source the R12 tables hold; required by features.adsb.",
       "plans/PREREG_adsb_nn_2026_09_10.md:5"),
    _K("adsb.sources.adsb_fi", "bool", False,
       "Declares that the ADS-B tables hold adsb.fi points (the owner's July-2026 opendata.adsb.fi capture, merged with "
       "adsb.lol by rule AUN.1; non-commercial terms, never redistributed, cannot be re-fetched by others). Checked against "
       "the tables' report both ways at fit and predict; the stand-exit stage's table (modules.uad.table) may not hold "
       "adsb.fi points while this is false. Off in the documented forecaster; a scored-period build may turn it on with "
       "tables built from the union (docs/fm1/REPRODUCE.md).", "plans/PREREG_afi_adsbfi_into_fm1_2026_10_03.md"),
    _K("adsb.table_dir", "path", "data/r12/tables_801f9bcac9a5",
       "Directory with the ADS-B tables train.parquet (2025) and serve.parquet (scored rows), keyed by MVT_ID_mvt.",
       "data/r12/tables_801f9bcac9a5/report.json"),
    _K("adsb.serve_missing_max_share", "float", 0.001,
       "At predict time, the largest share of scored matched rows that may lack an ADS-B table row; such a row is served "
       "as 'no ADS-B data' (coverage 0) and counted. Above it the run refuses. At fit a missing row is always refused.",
       "reports/final_model/FM1_PACKAGE_NOTES_2026_09_28.md §N1", lo=0.0, hi=1.0),
    _K("adsb.min_coverage", "int", 2,
       "A row is ADS-B-covered iff its coverage code >= this (2 = seen on the ground). The stack serves covered rows only.",
       "prc/pipeline/adsb_stage.py:32", lo=1, hi=3),
    _K("adsb.support_gate.rule", "choice", "manifest",
       "Which airports the ADS-B stack may serve. `manifest` = the gate stored with the stack models (ADN.7 rule, as "
       "v36 served); `coverage_ratio` = allowed iff the scored rows' covered share is within a factor `threshold` of the "
       "training share. The label-free ESS rule is a day 4-6 item (FMA §4.2 item 2).",
       "data/r12/stack_801f9bcac9a5_full/models_r12/MANIFEST.json", choices=("manifest", "coverage_ratio")),
    _K("adsb.support_gate.threshold", "float", 0.7,
       "coverage_ratio rule: allowed iff threshold <= covered_share_serve / covered_share_train <= 1 / threshold.",
       "data/r12/stack_801f9bcac9a5_full/models_r12/MANIFEST.json", lo=0.01, hi=1.0),
    _K("adsb.train_dropout.flight", "float", 0.0,
       "Training-time ADS-B dropout, flight level: a seeded random share of the JOINED matched training rows is fitted as an "
       "unheard flight (joined 0, coverage 1, flight-level ADS-B columns NaN; the airport-level counts kept), so the model "
       "learns to predict without the flight's own track at every airport and season. Fit only; predict never hides ADS-B. "
       "Why: which flights have ADS-B differs by airport-month between 2025 and 2026 (LIRF July 93 % -> 0.2 %).",
       "plans/PREREG_acr_adsb_coverage_regime_2026_10_02.md", lo=0.0, hi=0.9, late=2),
    _K("adsb.train_dropout.day", "float", 0.0,
       "Training-time ADS-B dropout, network level: every matched training row of a seeded random share of (airport, UTC "
       "day) groups is fitted as a network blackout (the flight-level state plus ground_seen_mvt_m10 = 0 and "
       "moving_mvt_m10 = NaN), the state of LIRF in July 2026. Fit only.", "plans/PREREG_acr_adsb_coverage_regime_2026_10_02.md", lo=0.0, hi=0.9, late=2),
    _K("adsb.stack.enabled", "bool", True,
       "Serve the ADS-B stack residual (LightGBM + MLP; MLP in its worker, BC-5) on covered, gated matched rows: "
       "value = rint(max(proxy - (F + resid), 1)).", "plans/PREREG_adsb_ship_v11_2026_09_11.md"),
    _K("adsb.stack.train_months", "int_list", [1, 7],
       "2025 months whose labels trained the stored stack (r12_stack: the val run's Jan + Jul records). A validation fold "
       "that tests one of them with the stack on is refused (leakage).", "scripts/analysis/r12_stack.py oof", lo=1, hi=12,
       unique=True),
    _K("adsb.stack.models_dir", "path", "data/r12/stack_801f9bcac9a5_full/models_r12",
       "The stack's stored models (G_*.txt LightGBM, N_*.pt MLP, MANIFEST.json with feats and gate).",
       "data/r12/stack_801f9bcac9a5_full/report_r12.json"),
    _K("adsb.stack.f_model", "choice", "lightgbm",
       "The F the ADS-B stack corrects on its rows: `lightgbm` = the base model's F (v36's stack, models_r12); `lgbcat` = "
       "0.5 F + 0.5 C, C = the matched CatBoost's delta (v46: R13's blend; its stack r13_stack_full/models_lgbcat was "
       "trained on the DWG tables 51348d3e1cb3). The stack must be the one trained on that base (r12_stack writes "
       "models_<arm>); fit fits the CatBoost whenever the blend needs it.",
       "plans/PREREG_blf_blend_serve_2026_09_28.md RESULT (v46 = 250.7314, +38.5 board MSE vs v43)",
       choices=("lightgbm", "lgbcat")),
    # ---- matched ------------------------------------------------------------------------------------------------
    _K("matched.learner", "choice", "lightgbm",
       "Matched base learner: `lightgbm` (in-process) or `catboost` (~/.venvs/prc-catboost worker). xgboost is CUT.",
       f"{PREREG} §2", choices=("lightgbm", "catboost")),
    _K("matched.target", "choice", "delta",
       "`delta` = proxy - taxi, recovered as max(proxy - F, 1) (every shipped matched model); `y` = taxi directly "
       "(selection arm A4; lost three times before).", "scripts/lgbm_submit.py:161", choices=("delta", "y")),
    _K("matched.es_months", "int_list", [3, 9],
       "Training months carved out for early stopping (the refit then uses every training month).",
       "scripts/lgbm_submit.py:146", lo=1, hi=12, unique=True),
    _K("matched.lightgbm.learning_rate", "float", 0.01,
       "LightGBM learning rate (0.01 = full tier; protocol S screens at validate.screen_learning_rate).",
       "scripts/lgbm_submit.py:174", lo=1e-4, hi=1.0),
    _K("matched.lightgbm.num_leaves", "int", 255, "LightGBM num_leaves: the tree size; larger fits finer interactions and needs more rows per leaf.", "scripts/lgbm_submit.py:174", lo=2, hi=4096),
    _K("matched.lightgbm.min_data_in_leaf", "int", 40, "LightGBM min_data_in_leaf: the smallest leaf; a floor against fitting single-flight noise.", "scripts/lgbm_submit.py:175",
       lo=1, hi=100_000),
    _K("matched.lightgbm.feature_fraction", "float", 0.8, "LightGBM feature_fraction (column subsample per tree).",
       "scripts/lgbm_submit.py:175", lo=0.05, hi=1.0),
    _K("matched.lightgbm.bagging_fraction", "float", 0.8, "LightGBM bagging_fraction (row subsample).",
       "scripts/lgbm_submit.py:175", lo=0.05, hi=1.0),
    _K("matched.lightgbm.bagging_freq", "int", 1, "LightGBM bagging_freq (0 = no bagging).",
       "scripts/lgbm_submit.py:175", lo=0, hi=100),
    _K("matched.lightgbm.max_rounds", "int", 50_000,
       "Cap on boosting rounds of the early-stopping run.", "scripts/lgbm_submit.py:178", lo=1, hi=500_000),
    _K("matched.lightgbm.patience", "int", 200, "Early-stopping patience (rounds without improvement on es_months).",
       "scripts/lgbm_submit.py:178", lo=1, hi=100_000),
    _K("matched.catboost.learning_rate", "float", 0.02,
       "CatBoost learning rate of the matched half-blend. The final fit used 0.02 and reached its round cap (LIMITATIONS "
       "13); the stored ADS-B stack was trained on a CatBoost holdout fitted at 0.05 (the R13 registration's amended full "
       "tier), so the stack is served on a differently made F (LIMITATIONS 16).", "scripts/analysis/r13_cat.py CB_LR",
       lo=1e-4, hi=1.0),
    _K("matched.catboost.depth", "int", 8, "CatBoost symmetric-tree depth (2^depth leaves per tree).", "scripts/analysis/r13_cat.py CB_BASE", lo=1, hi=16),
    _K("matched.catboost.l2_leaf_reg", "float", 3.0, "CatBoost L2 regularisation of the leaf values (shrinks small leaves).",
       "scripts/analysis/r13_cat.py CB_BASE", lo=0.0, hi=1000.0),
    _K("matched.catboost.subsample", "float", 0.8, "CatBoost Bernoulli row subsample per tree (stochastic regularisation).",
       "scripts/analysis/r13_cat.py CB_BASE", lo=0.05, hi=1.0),
    _K("matched.catboost.max_rounds", "int", 30_000, "Cap on CatBoost iterations of the early-stopping run.",
       "scripts/analysis/r13_cat.py CB_CAP", lo=1, hi=500_000),
    _K("matched.catboost.patience", "int", 200, "CatBoost early-stopping patience (od_wait).",
       "scripts/analysis/r13_cat.py CB_PATIENCE", lo=1, hi=100_000),
    _K("matched.blend.catboost_weight", "float", 0.5,
       "Weight w of the CatBoost half-blend on matched rows the ADS-B stack does not serve: "
       "rint(w * max(proxy - C, 1) + (1 - w) * taxi_F). 0 turns the blend (and the CatBoost fit) off.",
       "plans/PREREG_ship_cbv_2026_09_13.md", lo=0.0, hi=1.0),
    # ---- unmatched ----------------------------------------------------------------------------------------------
    _K("unmatched.learner", "choice", "v26",
       "Unmatched body. `v26` = the SHIPPED v26 body (Amendment A3): build_submission.fit_unmatched's hierarchical S1 "
       "cells give the fill rate p and the tail cells; the non-fill term is the MJS body (unm_congestion's congestion "
       "design + MAU's 1-in-20 matched extra rows at 25 % mass, HistGradientBoosting with the unmatched.hgb.* knobs); "
       "value = rint(max(p * sp + (1 - p) * nf_hybrid(cells, body), 1)); replayed exactly on all 4,907 non-LIRF 2026 "
       "rows. `hgb` = the days 1-3 PROVISIONAL E3C-shaped body on FM1's own unmatched frame (never replayed against a "
       "shipped file; kept only as a selection-study alternative).",
       "plans/PREREG_mjs_mxj_ship_2026_09_21.md", choices=("v26", "hgb")),
    _K("unmatched.hgb.max_iter", "int", 1000, "HGB max_iter of the unmatched body (MXI raised it 200 -> 1000; v26).",
       "plans/PREREG_mjs_mxj_ship_2026_09_21.md", lo=1, hi=100_000),
    _K("unmatched.hgb.n_iter_no_change", "int", 50, "HGB early-stopping patience of the unmatched body (MJS: 50; v26).",
       "plans/PREREG_mjs_mxj_ship_2026_09_21.md", lo=1, hi=10_000),
    _K("unmatched.hgb.max_leaf_nodes", "int", 15, "HGB max_leaf_nodes: the tree size of the unmatched body (small: ~5k training rows per year).", "scripts/build_submission.py:65", lo=2,
       hi=4096),
    _K("unmatched.hgb.min_samples_leaf", "int", 100, "HGB min_samples_leaf: the smallest leaf of the unmatched body.", "scripts/build_submission.py:65", lo=1,
       hi=100_000),
    _K("unmatched.hgb.learning_rate", "float", 0.1, "HGB learning rate (sklearn's default, as every shipped body).",
       "scripts/build_submission.py:65", lo=1e-4, hi=1.0),
    _K("unmatched.encoding_smooth", "float", 50.0,
       "Smoothing count k of the y target encodings of the `hgb` body. The `v26` body encodes with the shipped "
       "build_submission.SMOOTH = 50 inside its estimators, so v26 refuses any other value (it would be ignored).",
       "scripts/build_submission.py:35", lo=0.0, hi=100_000.0),
    # ---- optional modules ---------------------------------------------------------------------------------------
    _K("modules.arc.enabled", "bool", True,
       "ARC (v17): an unmatched departure whose own flight's ARR row (callsign-matched, at its destination) carries LOBT "
       "and AOBT_3 takes a clock-recovered value: `ord` -> max(rint(MVT - AOBT_3), 1), `fill` -> max(rint(sp), 1) "
       "(arr_row_clocks.sub_rule, arc_ship_build.arc_values). Runs after the Rome rules and before E2 (the shipped order).",
       "plans/PREREG_arr_row_clocks_2026_09_16.md"),
    _K("modules.arc.airports", "str_list", ["LIRF"],
       "Departure airports ARC rules at. The shipped scope is LIRF only (v17 changed 42 LIRF rows; its thresholds were "
       "set on Rome). Other airports are UNVALIDATED: the FMA's all-airport ARC (CORE item 3) needs a LOMO outside Rome "
       "first. Every code must be in airports.codes.", "scripts/analysis/arc_ship_build.py:43", unique=True),
    _K("modules.rome.enabled", "bool", True,
       "LIRF recording-regime module (optional, default ON: owner decision 1), documented as modelling Rome's label "
       "recording, in the shipped order on LIRF's unmatched rows: R2 fill classifier (rome_fill, LightGBM, mean of the "
       "run seeds) -> mixture with S1's non-fill term -> 24 h date-slip expectation (rome_dateslip) -> rint -> E1 schedule "
       "floor (24,000 <= sp < 86,400) -> RLD local-day rule; then (after ARC) E2 and E2W. ARN (v21) is DROPPED. The "
       "rules' constants were selected on 2025 labels of all months (in-sample, PREREG_urp_u_chain_replay).",
       "plans/PREREG_rome_fill_2026_09_10.md"),
    _K("modules.rome.e2", "bool", True,
       "E2 / RSE (v31), read only while modules.rome.enabled: on LIRF rows still at their Rome value with 3,600 <= sp < "
       "24,000 (rse_build.row_set), the stand-inbound logit (agent_E_rules.rule_E2, frozen coefficients) replaces the "
       "fill probability; a UTC day whose move energy per moved row exceeds e2_day_p99 is reverted. Reads LIRF arrival "
       "in-blocks from data.raw_dir itself (the training file of the row's month; ranking.parquet for 2026).",
       "plans/PREREG_rse_rome_stand_inbound_e2_2026_09_23.md"),
    _K("modules.rome.e2w", "bool", True,
       "E2W (v33), read only while modules.rome.enabled: on the rows E2 moved, E2's form with coefficients beta_W refitted "
       "at FIT time (stake-weighted Brier, e2w_build.fit_brier_w) on the training months' LIRF band rows, their p_R and "
       "non-fill term nested leave-one-month-out; then its own per-day check. Needs modules.rome.e2.",
       "plans/PREREG_e2w_ship_v33_2026_09_24.md"),
    _K("modules.rome.e2_day_p99", "float", 15.941366958491995,
       "E2's per-day revert threshold (RSE.1): the 99th percentile of 2025's per-day move energy per moved row "
       "(sum of squared moves / 344,841 / rows moved). A 2025-derived constant frozen at the v31 ship; FM1 does not "
       "refit it.", "submissions/merry-quicksand_v31_rse.meta.json", lo=0.0, hi=1e9),
    _K("modules.rome.e2w_day_p99", "float", 9.042848041672684,
       "E2W's per-day revert threshold, the same statistic for the beta_W move. A 2025-derived constant frozen at the v33 "
       "ship; FM1 does not refit it.", "submissions/merry-quicksand_v33_e2w.meta.json", lo=0.0, hi=1e9),
    _K("modules.storm.mode", "choice", "off",
       "Storm/congestion anchor (v16, re-applied inside the v24-v26 bodies): `off` (the default until the leave-one-event-"
       "out re-validation, owner decision 2) or `anchor`: every row the unmatched body decided takes "
       "rint(v + max(A - 300, 0)), A = the airport-day median MVT - AOBT_3 of the scored rows' matched departures minus the "
       "airport-month median of those day medians (congestion_anchor_ship.day_anomaly / anchored_values; knot 300 s "
       "fixed by the shipped function). The constants were chosen post hoc on 2026 label-free data (FMA §1.3 #4).",
       "plans/PREREG_ship_congestion_anchor_2026_09_13.md", choices=("off", "anchor")),
    _K("modules.storm.feature", "bool", False,
       "The anomaly A as a body FEATURE with a leave-one-event-out re-validation (owner decision 2's replacement for "
       "the anchor rule). NOT PORTED (not built): fit refuses while this is true.",
       "plans/PREREG_congestion_anchor_loeo_2026_09_13.md", status="not_ported", activates=_on),
    _K("modules.tail_cells.enabled", "bool", True,
       "The unmatched body's not-copy value IN THE BODY MIXTURE is the learned body, except that with this knob true "
       "(the shipped rule) a row whose cell mean is 3,000 s or more keeps the cell mean (`build_submission.nf_hybrid`); "
       "false uses the learned body there instead. A LIRF row is always decided by the Rome module from S1's own "
       "hybrid, never from this mixture, so this knob never changes a LIRF value. Registered 2026-10-05 after a look "
       "at the ten design months of 2025 (25 rows, all LTFM, 22 with labels under 2,000 s against values of "
       "3,100-5,700 s). Default true.",
       "plans/PREREG_tc_tail_cell_routing_2026_10_05.md", late=4),
    _K("modules.uad.enabled", "bool", False,
       "ADS-B stand-exit estimate for departures without a flight-plan record (UAD). A row that the unmatched body or "
       "the storm anchor decided, and whose row in modules.uad.table has class A1 or A2, moves from its value v to "
       "rint(max(v + w (u' - v), 1)): u' = p (MVT - SCHED) + (1 - p)(exit_rel_s + the airport's offset), p = the lane's "
       "fill probability; w = the calm weight, or the storm weight where the airport-day anomaly is above 300 s; a "
       "class-A2 row takes half of w. Offsets and weights come from modules.uad.constants. Measured once on 317 "
       "January + July 2025 rows against the leave-one-month-out baseline: RMSE 432 -> 274 s, body +53 bmse [22, 85]. "
       "Runs after the storm anchor and before LDI; off by default.",
       "plans/PREREG_uad_unmatched_adsb_offblock_2026_10_04.md", late=3),
    _K("modules.uad.table", "path", "data/uad/2026_table_a3/table.parquet",
       "The stand-exit table of the scored rows (MVT_ID_mvt, unmatched, cls, exit_rel_s), built from the ADS-B day "
       "store by scripts/analysis/uad_build.py. A prediction-time input: read only while modules.uad.enabled is true, "
       "its sha256 recorded in the build manifest.",
       "plans/PREREG_uad_unmatched_adsb_offblock_2026_10_04.md", late=3),
    _K("modules.uad.constants", "path", "configs/uad_constants_2025.json",
       "The per-airport offsets (seconds from off-block to leaving the stand) and the calm / storm weights of the UAD "
       "rule, fitted on the ten design months of 2025 by scripts/analysis/uad_price.py; FM1 does not refit them. Read "
       "only while modules.uad.enabled is true, its sha256 recorded in the build manifest.",
       "plans/PREREG_uad_unmatched_adsb_offblock_2026_10_04.md", late=3),
    _K("modules.ldi.enabled", "bool", False,
       "LTFM de-icing correction (v41): on LTFM's matched departures, a row with w_freezing == 1 on a SEVERE local day "
       "(UTC + 3 h; share(MVT - SCHED > 1800 s) >= 0.5 AND min w_temp_c <= 1.0 over the day's LTFM matched departures) "
       "takes value + the delta of its MVT - SCHED band (the three modules.ldi.delta_* knobs), before MTX. The day test "
       "reads every LTFM departure of the day, so a batch holding part of a scored-file day is refused "
       "(PartialDayError). Optional, off; with LTFM outside airports.codes it serves no row (reported).",
       "plans/PREREG_ldi_ltfm_deicing_2026_09_27.md", activates=_on),
    _K("modules.ldi.delta_le30_s", "float", 8.003118629519271,
       "LDI's shift (s) for target rows with MVT - SCHED <= 30 min. The registered Delta_apply = max(0, 0.5 x (February "
       "2025 OOF cell gap - the 2026 transfer adjustment)); a constant frozen at the v41 ship, not refitted by FM1.",
       "logs/ldi_scratch/result_20260927T1410/result.json jan26.delta_apply", lo=0.0, hi=3600.0),
    _K("modules.ldi.delta_30_60_s", "float", 35.44180751815995,
       "LDI's shift (s) for target rows with 30 min < MVT - SCHED <= 60 min (registered Delta_apply; frozen at v41).",
       "logs/ldi_scratch/result_20260927T1410/result.json jan26.delta_apply", lo=0.0, hi=3600.0),
    _K("modules.ldi.delta_gt60_s", "float", 122.4855891190231,
       "LDI's shift (s) for target rows with MVT - SCHED > 60 min (registered Delta_apply; frozen at v41).",
       "logs/ldi_scratch/result_20260927T1410/result.json jan26.delta_apply", lo=0.0, hi=3600.0),
    _K("modules.eobt_clip.enabled", "bool", False,
       "Serve-time EOBT clip (prc/pipeline/fm1/eobt_clip.py): in an (airport, UTC month) cell of the served rows whose "
       "EOBT_1 field carries delay revisions (share of eligible departures with EOBT_1 > LOBT >= min_share), every "
       "eligible departure takes EOBT_1 := min(EOBT_1, LOBT) before the feature path. Eligible = a departure at an airport "
       "in modules.eobt_clip.airports with AOBT_3, EOBT_1, LOBT and MVT_TIME present. Predict time only (the model was "
       "trained on the unclipped field). A batch of the scored file is decided on the scored file's whole cell. Why: from "
       "July 2026 LFPG's EOBT_1 carries revisions it did not carry in 2025 (35.1 % vs 3.8 - 7.1 %); priced on a held-out "
       "2025 simulation at about +195 ... +290 bmse (LightGBM only).", "plans/PREREG_clp_lfpg_eobt_clip_2026_10_02.md", late=1),
    _K("modules.eobt_clip.airports", "str_list", ["LFPG"],
       "Airports the EOBT clip may act on. Default LFPG only: elsewhere EOBT_1 > LOBT is 8 - 35 % in both 2025 and 2026 "
       "(normal practice the model trained on), and the clip was priced at LFPG only. Each must be in airports.codes "
       "while the clip is on.", "plans/PREREG_clp_lfpg_eobt_clip_2026_10_02.md", unique=True, late=1),
    _K("modules.eobt_clip.min_share", "float", 0.15,
       "Share of a cell's eligible departures with EOBT_1 > LOBT at or above which the clip acts in that cell. 0.15 is "
       "more than twice the highest unrevised LFPG month (7.1 % in 2025, 6.1 % in January 2026) and less than half of "
       "the revised one (35.1 %, July 2026).", "plans/PREREG_clp_lfpg_eobt_clip_2026_10_02.md", lo=0.0, hi=1.0, late=1),
    _K("modules.eobt_clip.min_rows", "int", 1000,
       "Fewest eligible departures a cell needs before the clip's share test is read; a cell with 1 ... min_rows - 1 is "
       "refused (DetectorError), never passed through. At 1,000 rows the share's standard error near 0.15 is 1.1 points.",
       "plans/PREREG_clp_lfpg_eobt_clip_2026_10_02.md", lo=1, hi=100_000_000, late=1),
    # ---- calibration --------------------------------------------------------------------------------------------
    _K("calibration.mtx.enabled", "choice", "auto",
       "MTX monthly calibration (shift each airport-month mean toward the published EUROCONTROL taxi-out mean). `auto` = "
       "on iff run.purpose is scored_period. ALWAYS the last value stage.",
       "plans/PREREG_mtx_monthly_calibration_2026_09_28.md", choices=("auto", "on", "off")),
    _K("calibration.mtx.reference_csv", "path", "data/mtx/raw/taxi_out_additional_time_2025.csv",
       "Published monthly CSV of the training year (fits the excess mapping c, e per airport).",
       "data/mtx/raw/MANIFEST.json"),
    _K("calibration.mtx.published_csv", "path", "data/mtx/raw/taxi_out_additional_time_2026.csv",
       "Published monthly CSV of the scored period (the target means). Exists only after the month is published.",
       "data/mtx/raw/MANIFEST.json"),
    _K("calibration.mtx.lambda_rule", "choice", "james_stein",
       "Shrink of each cell's shift: `james_stein` = max(0, 1 - sigma^2/gap^2) (V1, registered); `full` = 1 (V1F).",
       "scripts/analysis/mtx_calib.py:80", choices=("james_stein", "full")),
    _K("calibration.mtx.mode", "choice", "add",
       "`add` = uniform shift per cell (V1, registered); `mul` = scale (V2, not separable on 2025).",
       "scripts/analysis/mtx_calib.py:87", choices=("add", "mul")),
    # ---- validation ---------------------------------------------------------------------------------------------
    _K("validate.screen_learning_rate", "float", 0.05,
       "Learning rate of the screen tier (protocol S ranks arms at this rate, one seed).",
       "scripts/analysis/r12_validate.py SCREEN_LR", lo=1e-4, hi=1.0),
    _K("validate.tier", "choice", "screen",
       "`screen` = screen_learning_rate + the first seed only; `full` = the configured learner rates and all seeds.",
       f"{FMA} §4.3", choices=("screen", "full")),
    _K("validate.n_boot", "int", 2000, "Day-block bootstrap draws of every paired comparison.",
       "scripts/analysis/mtx_calib.py:111", lo=100, hi=100_000),
    _K("validate.boot_seed", "int", 20260928, "Seed of the day-block bootstrap.", f"{PREREG} §5", lo=0, hi=2**31 - 1),
)

KNOBS: Mapping[str, Knob] = types.MappingProxyType({k.path: k for k in SCHEMA})
LATE_KNOBS: tuple = tuple(k for k in SCHEMA if k.late)

#: Knobs a derived fit (`fit(..., reuse_from=donor)`) may change relative to its donor: predict-time or light-fit settings
#: only (plans/PREREG_bsv_board_selected_variants_2026_10_01.md, "Mechanism"). A `.`-terminated entry covers a group. A
#: knob that feeds the matched learners' X, y, masks or parameters must never be listed (tests/fm1/test_fm1_reuse.py).
#: adsb.table_dir is listed because what the learners read from it is guarded by DATA, not by the path: the
#: adsb/train.parquet input hash, the design-column check and the encodings check (a new serve table only, AFI).
#: data.scored_file / data.template_file: FP-W1 item 8 addendum, 2026-10-06. Neither is read anywhere in engine.fit's
#: call graph -- data.scored_file is read only by eobt_clip.apply and ldi.apply (both reachable only from
#: engine.predict) and compared-by-path (never read) in unmatched.check_e2_context (reachable only from
#: engine.predict, via unmatched.serve); data.template_file is read only by engine.build_submission. Verified
#: statically (this call-graph trace) and dynamically (tests/fm1/test_fm1_reuse.py::
#: test_fit_never_reads_the_scored_or_template_file, a fixture-scale fit with pandas.read_parquet trapped for both
#: paths). Listing them here means a new scored period's config no longer forces a full refit just to change which
#: file predict-time / build-submission-time code reads.
_REUSE_FREE_PREFIXES = ("run.purpose", "run.artifacts_dir", "calibration.mtx.", "modules.", "matched.blend.catboost_weight",
                        "adsb.stack.", "adsb.support_gate.", "adsb.min_coverage", "adsb.sources.adsb_fi", "adsb.table_dir",
                        "data.scored_file", "data.template_file")
REUSE_FREE_KNOBS = frozenset(p for p in KNOBS
                             if any(p == q or (q.endswith(".") and p.startswith(q)) for q in _REUSE_FREE_PREFIXES))


def reuse_blockers(cfg, donor_cfg) -> list:
    """The knobs outside the reuse allow-list on which two resolved configs differ, sorted; empty = the donor's matched
    learners are the ones a full fit of `cfg` would train (given the same input data)."""
    return sorted(p for p in KNOBS if p not in REUSE_FREE_KNOBS and cfg[p] != donor_cfg[p])


# =================================================================================================================
# schema completeness (the generator and the tests both call this)
# =================================================================================================================

def evidence_file(evidence: str) -> pathlib.Path:
    """The repo file an evidence pointer names: the text before the first ':' or ' ' (`file:line`, `file §4.3`)."""
    return ROOT / re.split(r"[: ]", evidence.strip(), maxsplit=1)[0]


# =================================================================================================================
# value checks (each names the dotted path)
# =================================================================================================================

def _tname(v) -> str:
    return type(v).__name__


def _check_bool(v, p):
    if not isinstance(v, bool):
        raise E.ConfigTypeError(f"{p}: expected true/false, got {_tname(v)} {v!r}")
    return v


def _check_num(v, p, lo, hi, integer: bool):
    if isinstance(v, bool) or not isinstance(v, int if integer else (int, float)):
        want = "an integer" if integer else "a number"
        raise E.ConfigTypeError(f"{p}: expected {want}, got {_tname(v)} {v!r}")
    v = int(v) if integer else float(v)
    if not (lo <= v <= hi):
        lo_s, hi_s = (int(lo), int(hi)) if integer else (float(lo), float(hi))
        raise E.ConfigValueError(f"{p}: {v!r} is outside [{lo_s}, {hi_s}]")
    return v


def _check_str(v, p):
    if not isinstance(v, str) or not v:
        raise E.ConfigTypeError(f"{p}: expected a non-empty string, got {_tname(v)} {v!r}")
    return v


def _check_choice(v, p, choices):
    if isinstance(v, bool) and ({"on", "off"} & set(choices)):
        raise E.ConfigTypeError(f"{p}: got bool {v!r}; YAML reads an unquoted on/off/yes/no as a boolean, write it quoted "
                                f"(one of {list(choices)})")
    _check_str(v, p)
    if p == "matched.learner" and v in CUT_LEARNERS:
        raise E.LearnerCutError(f"{p}: {v!r} {CUT_LEARNERS[v]}")
    if v not in choices:
        raise E.ConfigValueError(f"{p}: {v!r} is not one of {list(choices)}")
    return v


def _check_path(v, p):
    s = _check_str(v, p)
    parts = pathlib.PurePosixPath(s.replace("\\", "/")).parts
    if SUBMISSIONS_DIR in parts:
        raise E.SpliceRefusedError(f"{p}: {v!r} points into {SUBMISSIONS_DIR}/; FM1 never reads a previous submission "
                                   "(no splice)")
    return s


def _check_list(v, p, item, unique):
    if not isinstance(v, list):
        raise E.ConfigTypeError(f"{p}: expected a list, got {_tname(v)} {v!r}")
    if not v:
        raise E.ConfigValueError(f"{p}: must not be empty")
    out = tuple(item(x, f"{p}[{i}]") for i, x in enumerate(v))
    if unique and len(set(out)) != len(out):
        raise E.ConfigValueError(f"{p}: repeated value in {list(out)}")
    return out


def _check_tz_map(v, p):
    if not isinstance(v, Mapping):
        raise E.ConfigTypeError(f"{p}: expected a mapping of airport -> IANA zone, got {_tname(v)} {v!r}")
    out = {}
    for code, tz in v.items():
        _check_str(code, f"{p}.<key>")
        _check_str(tz, f"{p}.{code}")
        try:
            ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError):
            raise E.ConfigValueError(f"{p}.{code}: {tz!r} is not an IANA time zone") from None
        out[code] = tz
    return types.MappingProxyType(dict(sorted(out.items())))


def validate_value(k: Knob, v):
    """The validated, normalised value of knob `k` (lists -> tuples, maps -> read-only)."""
    p = k.path
    if k.kind == "bool":
        return _check_bool(v, p)
    if k.kind in ("int", "float"):
        return _check_num(v, p, k.lo, k.hi, integer=k.kind == "int")
    if k.kind == "str":
        return _check_str(v, p)
    if k.kind == "choice":
        return _check_choice(v, p, k.choices)
    if k.kind == "path":
        return _check_path(v, p)
    if k.kind == "int_list":
        return _check_list(v, p, lambda x, q: _check_num(x, q, k.lo, k.hi, integer=True), k.unique)
    if k.kind == "str_list":
        if p in ("airports.codes", "modules.eobt_clip.airports"):
            def item(x, q):
                _check_str(x, q)
                if not AIRPORT_RE.match(x):
                    raise E.UnknownAirportError(f"{q}: {x!r} is not a four-letter ICAO code")
                return x
            return _check_list(v, p, item, k.unique)
        return _check_list(v, p, _check_str, k.unique)
    if k.kind == "tz_map":
        return _check_tz_map(v, p)
    raise E.ConfigError(f"{p}: unknown kind {k.kind!r}")        # pragma: no cover - check_schema forbids it


def normalise_default(k: Knob):
    return validate_value(k, k.default)


def check_schema(schema) -> None:
    """Raise ConfigError on any undocumented, unevidenced, unranged or self-invalid knob, a duplicate path, or a path
    that is both a knob and a section. Every default must validate against its own knob. Structural only: it runs at
    import, so it reads no file (BC-38: the evidence files are the development record, which a clone does not carry;
    check_evidence_files checks them in the development tree)."""
    seen: set = set()
    for k in schema:
        if k.path in seen:
            raise E.ConfigError(f"duplicate knob {k.path!r}")
        seen.add(k.path)
        if k.kind not in KINDS:
            raise E.ConfigError(f"{k.path}: kind {k.kind!r} is not one of {KINDS}")
        if not k.effect or len(k.effect.strip()) < 20:
            raise E.ConfigError(f"{k.path}: effect is missing or shorter than a sentence")
        if not k.evidence:
            raise E.ConfigError(f"{k.path}: evidence is missing")
        if k.status not in STATUSES:
            raise E.ConfigError(f"{k.path}: status {k.status!r} is not one of {STATUSES}")
        if k.kind in ("int", "float", "int_list") and (k.lo is None or k.hi is None):
            raise E.ConfigError(f"{k.path}: numeric knob without a range")
        if k.kind == "choice" and (not k.choices or k.default not in k.choices):
            raise E.ConfigError(f"{k.path}: choice knob without choices or with a default outside them")
        if k.status == "not_ported" and k.activates is None:
            raise E.ConfigError(f"{k.path}: a not_ported knob must say which values activate it")
        if isinstance(k.late, bool) or not isinstance(k.late, int) or k.late < 0:
            raise E.ConfigError(f"{k.path}: late generation {k.late!r} must be a whole number >= 0 (0 = not late)")
        if k.late and not k.hashed:
            raise E.ConfigError(f"{k.path}: a late knob changes predictions, so it must be hashed")
        validate_value(k, k.default)
    for p in seen:
        for q in seen:
            if q != p and q.startswith(p + "."):
                raise E.ConfigError(f"{p!r} is both a knob and a section (of {q!r})")


def check_evidence_files(schema) -> None:
    """Raise ConfigError on a knob whose evidence pointer names a file that does not exist. A development-tree check
    (tests/fm1/test_fm1_knobs.py): the evidence is the project's record (registrations, logs), not part of the package."""
    for k in schema:
        if not evidence_file(k.evidence).exists():
            raise E.ConfigError(f"{k.path}: evidence file {evidence_file(k.evidence)} does not exist")


# =================================================================================================================
# flatten / nest
# =================================================================================================================

def flatten(raw, prefix: str = "") -> dict:
    """Nested mapping -> {dotted path: leaf}. Stops at a knob path (so a tz_map knob stays one leaf). Refuses splice keys
    and non-string keys anywhere."""
    if not isinstance(raw, Mapping):
        raise E.ConfigTypeError(f"{prefix or '<config>'}: expected a mapping, got {_tname(raw)} {raw!r}")
    out = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            raise E.ConfigTypeError(f"{prefix or '<config>'}: keys must be strings, got {_tname(k)} {k!r}")
        path = f"{prefix}.{k}" if prefix else k
        if k in SPLICE_KEYS:
            raise E.SpliceRefusedError(f"{path}: FM1 never splices a previous submission (the lgbm_submit --base "
                                       "landmine); every row is built from data")
        if path in KNOBS or not isinstance(v, Mapping):
            out[path] = v
        else:
            out.update(flatten(v, path))
    return out


def nest(flat: Mapping) -> dict:
    out: dict = {}
    for path, v in flat.items():
        node = out
        parts = path.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        if isinstance(v, tuple):
            v = list(v)
        elif isinstance(v, Mapping):
            v = dict(v)
        node[parts[-1]] = v
    return out


def _jsonable(v):
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, Mapping):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def canonical_hash(values: Mapping) -> str:
    """sha256 of the canonical JSON of the resolved values (every hashed knob; sorted keys; no whitespace)."""
    blob = json.dumps({k: _jsonable(v) for k, v in values.items()}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# =================================================================================================================
# the resolved view
# =================================================================================================================

@dataclass(frozen=True)
class FM1Config:
    values: Mapping[str, Any]          # every knob, validated and normalised (read-only)
    hash: str
    root: pathlib.Path
    overrides: tuple = ()
    source_path: pathlib.Path | None = None

    def __getitem__(self, path: str):
        if path not in self.values:
            raise E.UnknownKeyError(f"{path}: not a knob (see docs/fm1/CONFIG_REFERENCE.md)")
        return self.values[path]

    def path(self, knob: str) -> pathlib.Path:
        if KNOBS.get(knob) is None or KNOBS[knob].kind != "path":
            raise E.ConfigTypeError(f"{knob} is not a path knob")
        p = pathlib.Path(self.values[knob])
        return p if p.is_absolute() else self.root / p

    @property
    def short_hash(self) -> str:
        return self.hash[:12]

    @property
    def mtx_enabled(self) -> bool:
        v = self.values["calibration.mtx.enabled"]
        return v == "on" or (v == "auto" and self.values["run.purpose"] == "scored_period")

    def not_ported(self) -> list:
        """Every knob whose current value turns on a component FM1 has not ported (sorted)."""
        return sorted(k.path for k in SCHEMA if k.status == "not_ported" and k.activates(self.values[k.path]))

    def nested(self) -> dict:
        return nest(self.values)

    def replace(self, **updates) -> "FM1Config":
        """A new, re-validated config with {dotted path: value} updates (used by the validation protocols per fold)."""
        flat = {k: _jsonable(v) for k, v in self.values.items()}
        for k, v in updates.items():
            if k not in KNOBS:
                raise E.UnknownKeyError(f"{k}: not a knob")
            flat[k] = _jsonable(v)
        return parse_config(nest(flat), root=self.root, source_path=self.source_path)


def differs_from_defaults(cfg: FM1Config) -> list:
    return sorted(k.path for k in SCHEMA if cfg.values[k.path] != normalise_default(k))


def parse_override(text: str) -> tuple:
    """'a.b=value' -> ('a.b', value parsed as a YAML scalar or list)."""
    import yaml

    if not isinstance(text, str) or "=" not in text:
        raise E.ConfigError(f"override {text!r}: expected key=value")
    key, raw = text.split("=", 1)
    key = key.strip()
    if key not in KNOBS:
        raise E.UnknownKeyError(f"{key}: unknown knob in override {text!r}")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise E.ConfigError(f"override {text!r}: not a YAML value ({exc})") from None
    return key, value


def _cross_checks(v: dict) -> None:
    months = v["data.train_months"]
    stray = [m for m in v["matched.es_months"] if m not in months]
    if stray:
        raise E.ConfigValueError(f"matched.es_months: {stray} not in data.train_months {list(months)}")
    if not [m for m in months if m not in v["matched.es_months"]]:
        raise E.ConfigValueError("matched.es_months: every training month is an early-stopping month; no fit month left")
    codes, tz = set(v["airports.codes"]), set(v["airports.tz"])
    if codes != tz:
        raise E.ConfigValueError(f"airports.tz: must cover exactly airports.codes; missing {sorted(codes - tz)}, "
                                 f"extra {sorted(tz - codes)}")
    if not TEAM_RE.match(v["run.team"]):
        raise E.ConfigValueError(f"run.team: {v['run.team']!r} does not match {TEAM_RE.pattern}")
    if not (v["lanes.matched"] or v["lanes.unmatched"]):
        raise E.ConfigValueError("lanes: at least one lane must be on")
    if (v["adsb.train_dropout.flight"] > 0 or v["adsb.train_dropout.day"] > 0) and not v["features.adsb"]:
        raise E.ConfigValueError("adsb.train_dropout needs features.adsb (it hides the ADS-B block on training rows)")
    if v["adsb.stack.enabled"] and not v["features.adsb"]:
        raise E.ConfigValueError("adsb.stack.enabled needs features.adsb (the stack reads the ADS-B block)")
    if v["features.adsb"] and not v["adsb.sources.adsb_lol"]:
        raise E.ConfigValueError("features.adsb needs adsb.sources.adsb_lol (the R12 tables are adsb.lol)")
    if v["adsb.sources.adsb_fi"] and not v["features.adsb"]:
        raise E.ConfigValueError("adsb.sources.adsb_fi needs features.adsb (it declares what the ADS-B tables hold)")
    if v["matched.learner"] == "catboost" and v["matched.blend.catboost_weight"] != 0.0:
        raise E.ConfigValueError("matched.blend.catboost_weight must be 0 when matched.learner is catboost (the half-blend "
                                 "partner would be the base model itself)")
    _unmatched_cross_checks(v)


def _unmatched_cross_checks(v: dict) -> None:
    """The ported unmatched lane (Amendment A3): every combination it cannot honour is refused, never silently ignored."""
    codes = set(v["airports.codes"])
    if v["unmatched.learner"] == "v26":
        if v["data.dayoff_tz"] != "UTC":
            raise E.ConfigValueError("data.dayoff_tz: unmatched.learner v26 reads build_submission.derive's UTC dayoff (the "
                                     "shipped body); `local` is read only by the `hgb` body")
        if v["unmatched.encoding_smooth"] != 50.0:
            raise E.ConfigValueError("unmatched.encoding_smooth: unmatched.learner v26 encodes with build_submission.SMOOTH "
                                     "= 50 inside the shipped estimators; any other value would be ignored")
    if v["modules.rome.enabled"]:
        if not v["lanes.unmatched"]:
            raise E.ConfigValueError("modules.rome.enabled needs lanes.unmatched (the Rome module re-values LIRF's "
                                     "unmatched rows)")
        if v["unmatched.learner"] != "v26":
            raise E.ConfigValueError("modules.rome.enabled needs unmatched.learner v26 (its R2 mixture reads the v26 "
                                     "lane's S1 non-fill term and training frames)")
        if "LIRF" not in codes:
            raise E.ConfigValueError("modules.rome.enabled needs LIRF in airports.codes (every Rome rule is LIRF's); "
                                     "switch it off for a registry without LIRF")
        if v["modules.rome.e2w"] and not v["modules.rome.e2"]:
            raise E.ConfigValueError("modules.rome.e2w needs modules.rome.e2 (E2W re-values the rows E2 moved)")
    if v["modules.arc.enabled"]:
        if not v["lanes.unmatched"]:
            raise E.ConfigValueError("modules.arc.enabled needs lanes.unmatched (ARC re-values unmatched departures)")
        if v["unmatched.learner"] != "v26":
            raise E.ConfigValueError("modules.arc.enabled needs unmatched.learner v26 (ARC runs inside the ported lane)")
        stray = sorted(set(v["modules.arc.airports"]) - codes)
        if stray:
            raise E.ConfigValueError(f"modules.arc.airports: {stray} not in airports.codes {sorted(codes)}")
    if v["modules.storm.mode"] == "anchor":
        if not v["lanes.unmatched"]:
            raise E.ConfigValueError("modules.storm.mode anchor needs lanes.unmatched (it shifts the unmatched body's rows)")
        if v["unmatched.learner"] != "v26":
            raise E.ConfigValueError("modules.storm.mode anchor needs unmatched.learner v26 (the anchor runs inside the "
                                     "ported lane)")
    if not v["modules.tail_cells.enabled"]:
        if not v["lanes.unmatched"]:
            raise E.ConfigValueError("modules.tail_cells.enabled false needs lanes.unmatched (the knob changes the "
                                     "unmatched lane's not-copy value)")
        if v["unmatched.learner"] != "v26":
            raise E.ConfigValueError("modules.tail_cells.enabled false needs unmatched.learner v26 (not_fill_value "
                                     "runs inside the v26 lane's body_values)")
    if v["modules.uad.enabled"]:
        if not v["lanes.unmatched"]:
            raise E.ConfigValueError("modules.uad.enabled needs lanes.unmatched (the stage re-values the unmatched "
                                     "lane's rows with an ADS-B stand-exit estimate)")
        if v["unmatched.learner"] != "v26":
            raise E.ConfigValueError("modules.uad.enabled needs unmatched.learner v26 (the stage reads the v26 lane's "
                                     "fill probability)")
        if v["modules.storm.mode"] != "anchor":
            raise E.ConfigValueError("modules.uad.enabled needs modules.storm.mode anchor (the stage reads the "
                                     "anchor's airport-day anomaly)")
    if v["adsb.stack.enabled"]:
        arm = pathlib.PurePosixPath(str(v["adsb.stack.models_dir"]).replace("\\", "/")).name
        if v["adsb.stack.f_model"] == "lgbcat":
            if arm != "models_lgbcat":
                raise E.ConfigValueError(f"adsb.stack.f_model lgbcat needs a stack trained on the blend (r12_stack's "
                                         f"models_lgbcat); adsb.stack.models_dir is {v['adsb.stack.models_dir']!r}")
            if v["matched.learner"] != "lightgbm":
                raise E.ConfigValueError("adsb.stack.f_model lgbcat needs matched.learner lightgbm (the blend's F half)")
        elif arm == "models_lgbcat":
            raise E.ConfigValueError("adsb.stack.f_model lightgbm cannot serve models_lgbcat (a stack trained on the "
                                     "LightGBM + CatBoost blend); set adsb.stack.f_model lgbcat")
    if v["modules.ldi.enabled"]:
        if not v["lanes.matched"]:
            raise E.ConfigValueError("modules.ldi.enabled needs lanes.matched (LDI shifts matched LTFM departures)")
        if not v["features.weather"]:
            raise E.ConfigValueError("modules.ldi.enabled needs features.weather (the rule reads the matched frame's "
                                     "w_freezing and w_temp_c)")
    if v["modules.eobt_clip.enabled"]:
        out = [a for a in v["modules.eobt_clip.airports"] if a not in v["airports.codes"]]
        if out:
            raise E.ConfigValueError(f"modules.eobt_clip.airports: {out} not in airports.codes "
                                     f"{list(v['airports.codes'])}")


def parse_config(raw: Mapping, root: pathlib.Path | None = None, overrides=(), source_path=None) -> FM1Config:
    """Validate a nested mapping (+ `path=value` overrides) into an FM1Config; raise a named ConfigError otherwise."""
    root = pathlib.Path(root) if root is not None else ROOT
    raw = dict(raw) if raw is not None else {}
    if "schema_version" in raw:
        sv = raw.pop("schema_version")
        if isinstance(sv, bool) or sv != SCHEMA_VERSION:
            raise E.ConfigValueError(f"schema_version: {sv!r} (this loader reads {SCHEMA_VERSION})")
    flat = flatten(raw)
    unknown = sorted(p for p in flat if p not in KNOBS)
    if unknown:
        raise E.UnknownKeyError(f"unknown key(s) {unknown}; every knob is listed in docs/fm1/CONFIG_REFERENCE.md")
    parsed = [parse_override(o) for o in overrides]
    for k, val in parsed:
        flat[k] = val
    values = {}
    for k in SCHEMA:
        values[k.path] = validate_value(k, flat[k.path]) if k.path in flat else normalise_default(k)
    _cross_checks(values)
    frozen = types.MappingProxyType(values)
    hashed = {k: v for k, v in values.items() if KNOBS[k].hashed}
    in_use = max((k.late for k in LATE_KNOBS if values[k.path] != normalise_default(k)), default=0)
    for k in LATE_KNOBS:                         # later generations: the hash the config had before they existed (BC-41)
        if k.late > in_use:
            del hashed[k.path]
    return FM1Config(values=frozen, hash=canonical_hash(hashed), root=root, overrides=tuple(parsed),
                     source_path=source_path)


def load_config(path, root: pathlib.Path | None = None, overrides=()) -> FM1Config:
    import yaml

    path = pathlib.Path(path)
    if not path.exists():
        raise E.ConfigError(f"config file {path} does not exist")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise E.ConfigError(f"{path}: not valid YAML ({exc})") from None
    if raw is not None and not isinstance(raw, Mapping):
        raise E.ConfigTypeError(f"<config>: expected a mapping at the top of {path}, got {_tname(raw)}")
    return parse_config(raw, root=root, overrides=overrides, source_path=path.resolve())


check_schema(SCHEMA)                 # import-time: a broken schema never loads
