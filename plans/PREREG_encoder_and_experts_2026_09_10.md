# Pre-registration — the corrected encoder, the scoped keys, and the conditional experts

**Written 2026-09-10 06:20 local, BEFORE any of these runs has started and before any of their
numbers exists.** Standalone file: the concurrent session owns
`plans/PREREG_taxiout_2026_09_08.md` and is actively appending RESULTs 18–20 to it.

Covers the three lanes of `plans/TOP_PATH_2026_09_10.md` that the arms queue is not touching:
Priority 0 (encoder/stopping separation), Priority 1's scoped keys, and Priority 2 (conditional
experts). All three run on fold A after the queue drains.

---

## 0. Order and cost

| # | run | command | cost |
|---|---|---|---|
| 1 | screen the three encoders | `scripts/enc_ab.py --screen --queue` | ~20 min per arm |
| 2 | the two best on the full fold | `scripts/enc_ab.py --full --queue --variants ...` | ~2–3 h per arm |
| 3 | conditional experts | `scripts/cond_experts.py --full --queue --exclude LIRF` | ~3 fits |

One heavy job at a time, `OMP_NUM_THREADS=1 nice -n 19`, and a fresh quality-gate pass before each.
**If the screen tier and the full tier disagree, the full tier decides** — screen magnitudes are
ranking only and this project has already paid once for quoting a subsample magnitude.

---

## 1. Hypothesis H-E1 — the corrected encoder

> Repairing the early-stopping leak changes `best_iter` materially, and the corrected design does
> not lose holdout accuracy against the incumbent.

**Arms.** `incumbent` (the shipped encoder, leak included — the control), `separated`
(`prc.encoding` two-stage).

**Registered before the run:** this comparison is **not** a clean isolation of the stopping repair.
Stage 2 also replaces the training rows' encodings with leave-one-month-out values, so `separated`
differs in two ways at once. The measured difference is the JOINT effect and will be reported as
such. A third arm splitting them is named in §5 as follow-up, not run here.

### 1.1 Thresholds, locked

- **`best_iter` moves** iff `|best_iter(separated) − best_iter(incumbent)| / best_iter(incumbent)
  >= 10%`. This is the primary read: it is the quantity the defect distorts.
- **ACCEPTED as the new baseline** iff the paired holdout interval against `incumbent` does **not**
  exclude zero in the incumbent's favour — i.e. the repair costs nothing measurable. A repair does
  not have to WIN; it has to not lose. **This is deliberately an inferiority test, not a
  superiority test**, and it is registered that way before the run so a null result is a pass.
- **BLOCKED** iff `separated` loses by an interval excluding zero AND the point loss exceeds
  **1.0 s**. Then the leak repair costs real accuracy and the reason must be understood before
  anything is built on it.
- Reported, not decisional: the stopping RMSE of each arm (the incumbent's is a LEAKY number and is
  never compared across arms as a quality measure), per airport, per `|delta|` band.

---

## 2. Hypothesis H-E2 — airport-scoped keys

> Adding airport-scoped composites of the five colliding keys improves matched holdout RMSE against
> the corrected encoder.

**Arms.** `separated` (reference), `scoped` (= separated + 10 columns).

### 2.1 Thresholds, locked

- **ESTABLISHED** iff (i) the paired interval excludes zero in `scoped`'s favour, (ii) point gain
  **>= +1.0 s**, (iii) gain > 2 × seed sd, AND (iv) **>= 6 of 10 airports improve**.
- **NOT WORKING** iff (i) fails.
- Anything between is INCONCLUSIVE and is reported with every number.

**A confound registered now, before the result exists.** `scoped` carries **34 encoding columns
against 24** — ten of them new information, but ten of them also extra capacity. If H-E2 clears, the
gain is **not attributable to scoping** until a capacity control is run: ten columns of the same
shape carrying permuted values, which adds capacity and no information. **A positive H-E2 without
that control is reported as "scoped design helps", never as "airport scoping helps".**

---

## 3. Hypothesis H-E3 — conditional fill / non-fill experts

> `p·mu_fill + (1−p)·mu_body`, with `mu_body` fitted on NON-FILL rows only, beats the all-rows model
> on matched holdout rows.

**Arms.** `baseline` (all-rows delta model, same encodings, same seeds), `cond`, and `cond_ex`
(= `cond` with LIRF held at the baseline).

**AMENDED 2026-09-10 06:40, before the run.** Measured: LIRF holds **80.7% of the fill-row gain
(+1,439 of +1,782)** and **−1,208 of the −1,719 body loss**. Ex-LIRF the arm nets **−168 MSE**
against **+63** with Rome in. **`cond` (Rome included) is the PRIMARY arm and `cond_ex` is a
diagnostic, not a fallback** — the paragraph below described it as a safer bet, which was wrong.
The bars are unchanged; only which arm is primary is corrected, and it is corrected before any
number exists.

**Why `cond_ex` is registered NOW and not chosen later.** Four pooled rules have died on LIRF alone
(RESULTS 16, 17, 20, and Arm U's LIRF 384.07 → 622.84). Naming the exclusion in advance is the only
way to test it without selecting an airport after inspecting a fold, which this project's rules
forbid and which Amendment 26 set the precedent for.

### 3.1 Thresholds, locked

Paired row bootstrap, 2,000 draws, seed 0, on the matched holdout rows.

- **`cond` ESTABLISHED** iff (i) interval excludes zero in its favour, (ii) point gain **>= +1.0 s**,
  (iii) gain > 2 × seed sd, AND (iv) **>= 6 of 10 airports improve**.
- **`cond_ex` ESTABLISHED** iff the same four clauses hold on the FULL matched set with LIRF scored
  by the baseline, with the point bar raised to **+1.5 s** and **>= 7 of the 9** non-excluded
  airports improving — stricter, because it is the second cut of one record.
- **NOT WORKING** for either iff its (i) fails.
- **Decisional subsets, both reported:** fill rows (`|y − sp| <= 60`) and body rows separately. The
  mechanism claim is that the fill branch gains and the body branch does not lose; a pooled number
  that hides opposite-signed subsets is exactly what made RESULT 8 unreadable.

### 3.2 The prize, and what would refute it

`reports/PRIORITY2_FILL_LANE_PRICED.md` measures the fill branch at **+1,782 MSE** and the old
blend's body damage at **−1,719**, for a net of **+63**. **The +1,782 is not a model gain** — it is
a subset figure, and the only number here with a control behind it is the gated rule's ~322 MSE. H-E3 asserts a properly fitted `mu_body` removes the second
without losing the first. **It is refuted if `mu_body` underperforms the all-rows model on body
rows** — it trains on ~9% fewer rows against a shifted distribution, and that is the single most
likely way this returns nothing. That number (body-row RMSE, `cond` vs `baseline`) is the one to
read first.

### 3.3 Calibration

`--calibrate` is **OFF** for the primary run. The isotonic map would be fitted on the
early-stopping months, which are also the stopping set — a double use that is acceptable for a
diagnostic and not for a primary result. If `cond` clears its bars, the calibrated variant is a
registered follow-up with leave-one-month-out out-of-fold predictions, not the ES-month shortcut.

---

## 4. What none of these license

No submission. Fold TOTALS do not transfer (Amendment 9); only paired matched-row gains do, at the
measured 0.97–1.17×. A cleared bar licenses a twelve-month fit and a board version, decided by the
owner, and nothing more.

## 5. Follow-ups, named now so they are not presented later as new ideas

1. **The isolation arm for H-E1** — separated stopping with in-sample refit features — which splits
   the joint effect §1 reports.
2. **The capacity control for H-E2** — ten permuted columns.
3. **LOMO calibration for H-E3** if it clears.
4. **BC-1 sites 2–4** (`reports/bug_classes.md`): `load_fold_all_rows`, the submission path, and the
   per-airport arms all still carry the early-stopping leak. The submission path is the one that
   touched every shipped version's tree counts. None can be changed while the queue runs.
