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
