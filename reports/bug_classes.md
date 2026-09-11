# Bug classes — active tripwires

Each entry lists EVERY file in the codebase that uses the same pattern, whether currently buggy or
not. A change to any listed file forces a re-audit against its class before shipping. When a new
file starts using a listed pattern, it is added here in the same commit.

---

## BC-1 · A statistic fitted on rows that include the rows it is later STOPPED on

**Surfaced:** 2026-09-10, by the owner's audit (`reports/top_path_audit.json`,
`encoding_dependency_check.inner_stop_label_changes_fit_and_stop_features = true`), reproduced
independently in `reports/PRIORITY0_ENCODING_SEPARATION.md`.

**Root cause.** `fold_masks` puts the early-stopping months INSIDE the training mask — its own
docstring says so (`lgbm_fold.py:657`, *"ES months come out of the TRAINING fold"*). Any target-reading
statistic fitted on `tr` and then applied to every row therefore gives the ES rows features computed
from their own labels. Existing guards in this codebase all check the **outer holdout**, which is a
different mask and is not affected. The inner stopping set had no guard at all.

**What it does and does not invalidate.** Outer-holdout RMSEs and paired bootstrap intervals stay
valid — the holdout is excluded from `tr` and never contaminated. What is distorted is `best_iter`,
so tree counts and any cross-arm comparison of tree counts or stopping RMSE are biased by an
unmeasured amount.

**Measured size** (861,290 real rows, 5 months, all 12 keys): 24 of 24 encoding columns on ES rows
move when ES labels move, against 0 of 24 after repair. Mean displacement 17.8 s (`te_STAND_mvt`),
15.5 s (`de_STAND_mvt`), against `sd(delta)` 378.4 s.

### Sibling list — every site with this shape

| # | site | status |
|---|---|---|
| 1 | `lgbm_fold.py:1464` — `load_fold`, matched path | **repaired in a new module** (`prc/encoding.py`), NOT yet wired; the incumbent line is unchanged while the arms queue runs |
| 2 | `lgbm_fold.py:1528` — `load_fold_all_rows`, the all-rows path (Arm U) | **LIVE, unrepaired.** Same call, `delta_mask=~is_um`. Arm U's verdicts (RESULT 12, 15) were selected under it |
| 3 | `lgbm_submit.py:1452` — the SUBMISSION path | **LIVE, unrepaired, and it touched every shipped version.** `train` excludes the ranking rows, so serve-row features are clean and **the board scores are real**; but `n_ref` for v3/v4/v5 was chosen against a biased stopping metric |
| 4 | `lgbm_submit.py:1106–1124` — the per-airport arms | **LIVE, unrepaired.** `tr_mask = train & (ap_code == code)` then `early_stop(..., fit_a, es_a)`; the encodings were already fitted over all of `train` |
| 5 | `stand_ab.py:402` — `infold_slack_stats` | **DORMANT.** Same class exactly — `slack = -(gapa + delta)` reads the target and is fitted on `tr_mask` — but `STAND_INFOLD` is **not in `L.FEATS` (68) or `FEATS_QUEUE` (80)**, so no live arm or submission uses it. Verified 2026-09-10. Becomes live the moment any feature list adds `stand_slack_med` / `stand_slack_iqr` |

### Tripwire

Before changing any file above, or before adding a feature list that includes a target-reading
in-fold statistic, re-audit against this class. The test that pins it is
`tests/test_encoding_separation.py::test_incumbent_leak_is_real`, which asserts the defect itself so
it cannot be silently "fixed" and thereby hidden.

### Not yet done

Sites 2, 3 and 4 are unrepaired. Site 3 is the one that touched shipped artifacts. None can be
changed while the arms queue is running against those files.

---

## BC-2 · Two prediction columns compared under DIFFERENT recovery conventions

**Surfaced twice in one night, 2026-09-10, by two different agents working this repo in parallel.**
Both instances were caught by a *control*, not by inspection — which is the only reason either was
caught at all.

| # | instance | symptom | how it was caught |
|---|---|---|---|
| 1 | `fold_preds_fillhead.parquet`: the stored `treatment` column was built from an already-floored baseline branch (`p·sp + (1−p)·max(proxy−baseline, 1)`), while the stored `baseline` column is a raw delta. Recovering both as `proxy − col` floors one arm and not the other. | reported fill/body/net as **+1,789 / −1,534 / +256**; the truth is **+1,782 / −1,719 / +63**. 23 rows where the floor binds. | the pooled baseline read 226.258 against the record's own JSON value of 225.825 — **the tell was visible and was explained away as "seed averaging"** |
| 2 | `fold_preds_queue_allrows.parquet` stores arms as **taxi times** (its unmatched rows have no proxy, so it must); `fold_preds_queue_ytarget.parquet` stores them on the **delta scale**. Reading the second as the first. | a reported gain of **−109.0 s**, VOID | a negative control scored −23.9 s where it must score ~0 by construction |

**Root cause, shared: the convention is not recorded in the file.** Nine fold-prediction records
exist and at least three conventions are in use, with nothing machine-readable distinguishing them.

### The convention map — verified 2026-09-10 by reading each schema

| record | has `proxy`? | arm columns are | recover with |
|---|---|---|---|
| `fold_preds_queue_allrows.parquet` | yes | **TAXI TIMES** | use the column directly |
| `fold_preds_queue_allrows_20_2.parquet` | yes | **TAXI TIMES** | use the column directly |
| `fold_preds_queue_ytarget.parquet` | yes | **DELTA** (`y_seed*`) | `max(proxy − col, 1)` |
| `fold_preds_queue{,_day,_order,_weather}.parquet` | yes | **DELTA** (`delta_hat_seed*`) | `max(proxy − col, 1)` |
| `fold_v4_preds.parquet` | yes | **DELTA** (`delta_hat_seed*`) | `max(proxy − col, 1)` |
| **`fold_preds_fillhead.parquet`** | yes | **DELTA — but `treatment` already carries the floor and `baseline` does not** | floor BOTH: `max(proxy − col, 1)` |
| `stratum_fold_v7_preds.parquet` | **no** | taxi times (`S0`, `S1`) | use directly; there is no proxy on unmatched rows |

### Tripwire

Before comparing any two columns from any fold record:

1. **Reproduce a known number from the record's own JSON first.** Both instances above would have
   died instantly — instance 1's baseline reproduces 225.8254 only when floored, instance 2's
   arms reproduce 224.2576 / 260.6511 / 233.3405 only on the delta reading.
2. **Run a negative control that must score ~0 by construction** (a random-subset blend). This is
   what caught both. A lane without a control would have shipped the finding.
3. **Floor every arm at the same point.** `lgbm_fold.taxi_time` is `max(raw, 1)`; an arm that skips
   it is not comparable to one that does not.

### Repaired here

`scripts/cond_experts.py:290` floors `mu_fill` and the assembled mixture at the same point as
`mu_body`, and records `floor_binds_rows`. `prc/reference.py` makes `scale` a required field of any
stored baseline and refuses a load whose scale differs, with the field named in the error.

### NOT repaired

**No fold-prediction writer stamps its convention into the file.** The map above is documentation,
not a guard, and it will decay as new records are written. The structural fix is a `convention`
key in the parquet metadata plus a reader that refuses a record without one — not attempted here
because `scripts/lgbm_fold.py` is shared with a concurrent session.

---

## BC-3 · A provider's placeholder ingested as a measurement (`sentinel-ingested-as-measurement`)

**Surfaced:** 2026-09-10, found by the taxi-factor atlas session (`reports/TAXI_FACTOR_ATLAS_2026_09_10.md`,
"Arm W's precipitation column is dead"). The code was this session's (Amendment 24, 2026-09-09).
Post-mortem and fix 2026-09-10/11.

**What happened.** `stand_ab.load_weather` read the Iowa Mesonet field `p01i` as one-hour
precipitation. For every European station the archive returns the literal `"0.00"` — not `"M"` —
because European METARs carry no US hourly-precipitation group: 205,417 of 205,417 observations,
0 of 140 station-months with any other value, while the present-weather group reported
precipitation at the station on 18,947 of them. `w_precip_mm` was 0.0 on every row of every cache,
and the registered freezing rule (temp <= 3 C AND (precipitation OR a frozen code)) silently lost
its precipitation clause: cold rain and cold drizzle, the conditions under which aircraft are
de-iced, never flagged. **Arm W (RESULT 22) was measured on that block.**

**Root cause.** An external field was trusted because it was well-formed. Every check the
pipeline had — the fetch's header/station/row-count check, dtype, range (`>= 0`) — passes on a
placeholder, because a placeholder is syntactically a valid value. Nothing asked whether the field
carried *information*. The unit fixture used US-format `p01i` values (0.04, "T"), the shape the
code was written for, not the shape the archive emits for the ten scored airports.

**Why no test caught it** (post-mortem step 3): the fixture was too minimal in exactly the
dimension that mattered — it modelled the provider's documentation, not its output — and no test
looked at a real cache column's distribution. A constant column has no failing value to assert
against; only a distributional check (the class guard below) can see it.

**The class.** A value that means "not reported" or "unknown" arrives looking like a measurement:
a literal zero, a cap, a default, or a comparison on NaN that casts to a definite 0.

### Sibling list — every site with this shape (sweep 2026-09-11: `nan_to_num`, `fillna(`, `errors="coerce"`, comparisons cast to float on nullable columns, external fields; `scripts/adsb_*` owned by another session and excluded)

| # | site | class | status |
|---|---|---|---|
| 1 | `stand_ab.load_weather` — `w_precip_mm = p01i * 25.4` | (a) the defect | **fixed**: replaced by `w_precip_int` from the weather group; `p01i` gated per station-month in `prc/weather.py` |
| 2 | `stand_ab.load_weather` — `w_gust_kt = nan_to_num(gust)` when the WIND is also missing (7 rows) | (a) | **fixed**: NaN when the wind is unknown |
| 3 | `stand_ab.load_weather` — `(x <= c).astype(float)` for `w_freezing` / `w_lowvis` / `w_thunder`: unknown temperature, visibility or weather group became a definite 0 (9 / 4 / 15 rows) | (a) | **fixed**: three-valued `le3/and3/or3` in `prc/weather.py` |
| 4 | `stand_ab._wx_num` — a trace `"T"` substituted as 0.05 BEFORE the inch->mm conversion (1.27 mm) | (a), latent | **fixed**: `TRACE_MM` applied after conversion; unreachable on the frozen archive (no `T`) |
| 5 | `stand_ab._wx_num` — `errors="coerce"` turned a garbage value into NaN silently | (a), latent | **fixed**: `prc.weather._num` raises on anything but a number or `M` |
| 6 | `fetch_weather.check` — shape checks only | (a) | **fixed**: `information_report` prints any single-valued field with the contradicting evidence |
| 7 | `w_vis_km` — 6.21 mi (the "9999"/CAVOK 10 km cap) on 87% of observations | (b) right-censored, true | pinned: `test_visibility_cap_is_far_from_the_low_visibility_flag` |
| 8 | `w_thunder` — 0 on 99.94% of January rows | (b) rare, true | pinned: equal to the TS codes on all 205,417 observations |
| 9 | `cache_unmatched` — columns all-NaN by construction | (b) NaN is the honest unknown | pinned by `tests/test_unmatched_features.py`; on `ALL_NAN_OK` |
| 10 | `cache_unmatched.sched_sec` — constant 0 | (b) schedules are minute-precise | pinned: `test_schedule_seconds_are_zero_because_schedules_are_minute_precise` |
| 11 | `build_submission.py:234` — the L-e design matrix ends in `nan_to_num` | (b) inputs complete on every scored row | pinned: `test_scored_rows_have_complete_schedule_and_movement_times` |
| 12 | `build_submission.py:146, 377, 380`; `prc/encoding.py:66, 72` — unseen level -> prior, support -> 0 | (b) the encoding's definition: an unseen level has zero support | by design |
| 13 | `stand_ab.py:279, 1375` — `nancumsum(nan_to_num(ac))` in the queue sums | (b) a missing contribution adds nothing to a count | classified 2026-09-10 |
| 14 | `unm_congestion.py:160` — `hprox_n` NaN -> 0 | (b) an airport-hour absent from a count table has count 0 (its docstring) | by design |
| 15 | `cond_experts.py:275` — `nan_to_num(lab)` outside `tr` | (b) `fit_classifier` reads `label[fit]`, `label[es]`, `label[tr]` only, all inside `tr` | read 2026-09-11 |
| 16 | `lgbm_submit.design_matrix:832`, `stand_ab.py:461`, `lgbm_ab.py:64, 95` — `to_numeric(errors="coerce")` would turn a text column into an all-NaN feature | (b) no feature list names a text cache column | pinned: `test_no_feature_list_sends_a_text_column_through_the_coercing_design_matrix` (rehearsed RED with `ADEP_mvt` added) |
| 17 | `stand_ab.py:527` — diagnostic cut mask NaN -> False | (b) a printout, never a feature | — |
| 18 | `rome_fill.py:186` — airline propensity default 0.45 | (b) a synthetic harness's documented prior | — |

### The guards

* `tests/test_sentinel_values.py::test_no_cache_column_is_a_constant_sentinel` — every numeric
  column of every cache must vary on a real month, or sit on `CONSTANT_OK` with its reason. It
  would have caught this defect the day the block was built.
* `prc/weather.py` — the one METAR parser (VERSION 2.0.0): information content per
  station-month, unknown is NaN, and `tests/test_weather_parser.py` (23 mutants, all killed).
* `data/cache_weather/manifest.json` — a weather cache is stamped with the parser version and the
  archive digest, and `cmd_weather_cache` refuses to mix months from two parses.

### Tripwire

Any new external field (a new archive column, an ADS-B field, a schedule attribute) gets an
information-content check on the REAL source before a feature reads it: distinct values per
natural unit (station-month, airport-day), and a cross-check against an independent field that
should agree (here: the weather group). Any new `nan_to_num`, `fillna(<constant>)` or
`(x <op> c).astype(float)` on a nullable column is added to the table above with its class.

### NOT repaired

* The four coercing design-matrix builders (site 16) still coerce; the invariant is pinned, not
  enforced at the boundary. `lgbm_submit.py` / `lgbm_fold.py` are shared with concurrent sessions.
* `scripts/adsb_*` were not swept: they are owned by the ADS-B session, which edits them alone. The
  class was sent to that session on 2026-09-11 so it can run the same sweep on its own files.
* RESULT 22 (arm W) stands as measured on the v1 block; what it can and cannot claim is corrected in
  the pre-registration (note appended 2026-09-11).
