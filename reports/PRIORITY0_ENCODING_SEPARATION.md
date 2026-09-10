# Priority 0 — the encoder / early-stopping separation

**2026-09-10 01:00 local.** Follows `plans/TOP_PATH_2026_09_10.md` Priority 0. Nothing beneath the
running arms queue was touched: this is a new module (`prc/encoding.py`) and a new test file, and
`scripts/lgbm_fold.py`, `scripts/stand_ab.py` and every cache are byte-unchanged. Wiring waits for
the queue to drain.

**This is a model-selection repair. It has no measured predictive gain and must not be reported as
one.**

**Scope correction, 2026-09-10 02:20.** Stage 2 also replaces the TRAINING rows' encodings with
leave-one-month-out values, so `separated` differs from `incumbent` in two ways at once: the
stopping fix AND a change to the refit's own features. An A/B between them does **not** isolate the
early-stopping repair, and "model-selection only" is too narrow a framing. Both changes are wanted
(in-sample encodings on training rows are their own defect), but the comparison must be reported as
the joint effect, or a third arm — separated stopping, in-sample refit features — must split them.

---

## 1. The defect, confirmed in the shipped code

`scripts/lgbm_fold.py::load_fold` computes the encodings with

```python
for col, vals in S.infold_encodings(d, y, dlt, tr).items():
```

and `fold_masks` (lgbm_fold.py:656–664) puts the early-stopping months **inside** `tr` — its own
docstring says *"ES months come out of the TRAINING fold"*. `stand_ab.infold_encodings` therefore
fits all 24 target/delta encodings on a row set that contains the early-stopping rows' own labels.

The guard in that function's docstring (*"the leak lives at this call site"*) is about the **outer
holdout** and it holds. The early-stopping leak is a different one and is unguarded.

### Measured on the real cache — 861,290 rows, all 12 keys, 5 months

| check | incumbent | repaired |
|---|---|---|
| early-stopping rows sensitive to their **own** labels | **24 of 24 columns** | **0 of 24** |
| outer holdout rows sensitive to their own labels | 0 of 24 | 0 of 24 |
| mean shift on ES rows, `te_STAND_mvt` / `de_STAND_mvt` | — | **17.76 s / 15.51 s** |
| mean shift on ES rows, `te_ADEP_mvt` / `de_ADEP_mvt` | — | 11.20 s / 10.30 s |

For scale, `sd(delta)` on these rows is 378.4 s, so the ES features are displaced by roughly 3–5% of
the target's spread — in a consistent direction, because a row's own label always pulls its encoding
toward itself. That is what biases the stopping metric optimistically and inflates `best_iter`.

### What this does and does not invalidate

- **Every reported holdout RMSE and every paired bootstrap interval remains a valid measurement.**
  The outer Jan+Jul rows are excluded from `tr` and were never contaminated. Nothing needs
  re-scoring, and no shipped board result is in question.
- **`best_iter` was selected against a biased signal.** Tree counts, and any comparison of tree
  counts or stopping RMSE *across arms*, are distorted by an unmeasured amount.
- **Suggestive but not evidence:** arm D's live stopping metric reads `valid_0 rmse (delta) 200.78`
  at iter 22,000 against a holdout matched RMSE of 224.26. Part of that 23 s is the leak and part is
  that March/September are genuinely easier than January/July. **The split is unmeasured** and must
  not be attributed to either cause without the corrected re-run.

---

## 2. The repair

`prc/encoding.py`. A change of **scope**, not of formula.

```
stage1 (early stopping)   fit rows -> cross-fitted, leave-one-MONTH-out within the fit months
                          ES rows  -> encoder fitted on the fit months ONLY
stage2 (refit + score)    training rows -> cross-fitted, leave-one-MONTH-out within all training months
                          holdout/serve -> encoder fitted on all training months
```

Whole-month exclusion, not random row folds: random folds would leak through same-day, same-stand
structure — the dependence the date-block bootstrap in `tools/top_path_audit.py` exists to respect.

Three design notes:

1. **A single design matrix cannot serve both stages.** A stopping row is an evaluation row in
   stage 1 and a training row in stage 2, and needs different encodings in each. The caller should
   hold the 24 encoding columns separately and write the stage-appropriate set into the matrix
   before each fit — ~200 MB per stage at twelve months, against several GB for a second matrix.
2. **Cross-fitting is implemented by subtracting each month's aggregate from the pooled one**, so
   the cost is O(months × levels), not O(months × rows).
3. **Counts and fallbacks are emitted** (`n_<key>`). **Correction, 2026-09-10 02:20:** an earlier
   version of this line implied the counts are cross-fitted like the encodings. They are not.
   `separated_encodings` cross-fits only `te_`/`de_` and leaves `n_` at its in-fold value, so the
   counts describe the pooled support rather than the support the row's own encoding was actually
   fitted on. Named here rather than quietly changed: changing it changes a feature's meaning and
   belongs in its own A/B.

### Formula equivalence — the control is a real control

Given identical fitting rows, the new encoder is **byte-identical to `stand_ab.infold_encodings` on
all 24 columns over 861,290 real rows** (`rtol=0, atol=0`). The incumbent encoded baseline therefore
remains a valid control: any difference in a future A/B is attributable to the scope change alone.

### Cost

9.1 s for both stages with leave-one-month-out on 861,290 rows, against 2.3 s for the incumbent's
single fit. Extrapolating to twelve months, ~25 s against a 2–3 h fit. Not a scheduling factor.

---

## 3. Tests

`tests/test_encoding_separation.py` — **15 tests**, all passing (10 at first writing; 5 added 2026-09-10 02:20 after an adversarial review of this session's diff). 38 tests across the three new files.

`test_incumbent_leak_is_real` asserts the **defect itself**, so the repair cannot later be declared
unnecessary, and so a silent change to `infold_encodings` is caught.

**Mutation rehearsal, run 2026-09-10 00:58** — every mutation restored afterwards, suite green:

| mutation | tests red |
|---|---|
| stage 1's ES encoder fitted on `tr` (i.e. re-introduce the defect) | 1 |
| drop the y cross-fit | 3 |
| smoothing constant 50 → 40 | 1 |
| unseen level returns NaN instead of the prior | 2 |
| NaN-delta guard removed | 1 |
| counts computed wrongly | 1 |
| single-fit-month fallback broken | 1 |

No mutation survived. No test is decoration.

---

## 4. What is NOT done

- **Not wired into any harness.** `scripts/lgbm_fold.py` is untouched by design while the queue runs.
- **No baseline reproduction under the corrected recipe.** Priority 0's exit criterion — *"the
  baseline reproduces under its original recipe and the corrected recipe has independent stopping
  features"* — is half met: the second clause is measured above, the first needs a fold run.
- **The performance effect is unmeasured**, in either direction. The corrected stopping metric will
  read worse than 200.78 because the optimism is gone; that is the repair working, not a regression,
  and the only number that decides anything is the outer holdout RMSE at the corrected `best_iter`.
- **The data-ratio tree scaling has not been checked** against an unscaled refit, as Priority 0 asks.

## 5. Next, in order

1. Let the queue finish under its original rules (arm D running; Y, F, W to follow).
2. Wire `stage1`/`stage2` into a corrected fold path as a **new mode**, leaving the incumbent path
   intact as the control.
3. Re-run the A2 baseline both ways on fold A: same rows, same seed, same features — the only
   difference being the encoding scope. Report `best_iter`, the stopping RMSE and the **outer
   holdout RMSE** with a date-block paired interval.
4. Check the data-ratio tree scaling against an unscaled refit within development data.
5. Only then start Priority 1 (rich categorical) and Priority 2 (conditional experts), which both
   need this baseline to compare against.
