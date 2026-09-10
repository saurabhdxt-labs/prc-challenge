# Pre-registration — the p-gated fill mixture

**Written 2026-09-10 01:25 local, BEFORE any number below the line is computed.** Standalone file
so it does not collide with the concurrent session's edits to
`plans/PREREG_taxiout_2026_09_08.md`.

## 0. Why this exists

`reports/PRIORITY2_FILL_LANE_PRICED.md` measured, on the existing RESULT 8 record: the fill branch
gains **+1,782 MSE** on the 30,167 schedule-fill rows and the same rule loses **−1,719 MSE** on the
308,848 body rows, netting +63. (Those are the CORRECTED figures of 2026-09-10 02:05; the version
of this line written at 01:25 read +1,789 / −1,534 / +256, from a comparison that floored only one
arm. RESULT G below is unaffected — both of its arms derive from the same baseline, so the floor
cancels exactly; re-measured, +322 MSE and both intervals unchanged to three decimals.) The classifier is sound (holdout AUC 0.8957, decile fill rates
monotone 0.000 → 0.416).

`plans/TOP_PATH_2026_09_10.md` Priority 2 gives the mechanism: `m(X)` already approximates
`E[y|X]`, which is itself a mixture, so `p·sp + (1−p)·m(X)` applies fill behaviour twice. The
correct design needs `mu_body = proxy − E[delta|X,F=0]` **fitted on non-fill rows only**, which
requires a fit and therefore waits for the queue.

**This pre-registration is not that design.** It is the cheapest available probe of the same
mechanism, using only columns already in `data/cache_stand/fold_preds_fillhead.parquet`. It costs
no compute and can be run while the queue holds the machine.

## 1. Hypothesis H-G

> Applying the fill mixture **only where the classifier is confident** keeps most of the fill-row
> gain while leaving the body rows at the baseline, and therefore beats the ungated rule.

Note what a positive result would and would not mean. It would be evidence that the body damage is
caused by applying the mixture to low-`p` rows, which is the mechanism Priority 2 asserts. It would
**not** be a substitute for the fitted body expert, and it would not be shippable on this evidence:
this record was produced under the leaky early-stopping recipe (`reports/PRIORITY0_ENCODING_SEPARATION.md`).

## 2. The rule, fixed now

```
yhat = baseline_y                       where p <= T
yhat = p·sp + (1 − p)·baseline_y        where p >  T
```

`baseline_y = proxy − baseline`, `p` and `sp` are the stored columns. `T = 0` reproduces the
shipped ungated arm exactly, which is the control.

**Primary setting, chosen before measuring and not tunable afterwards: T = 0.5.** It is the natural
decision boundary of a probability, it was not selected by inspecting any result, and it is the only
value the verdict may be read from.

## 3. Thresholds, locked

Paired row bootstrap, 2,000 draws, seed 0, on the 339,015 matched holdout rows, against the
**ungated arm** (`T = 0`) and separately against the **baseline**.

- **SUPPORTED** iff, at T = 0.5: (i) the paired interval against the ungated arm excludes zero in
  the gate's favour, AND (ii) the paired interval against the baseline excludes zero in the gate's
  favour, AND (iii) the negative control below is cleared.
- **NOT SUPPORTED** iff (i) or (ii) fails.
- (i) and (ii) holding with (iii) failing is **NOT SUPPORTED**: the gate would not be doing the work.

**(iii) Negative control, specified before running.** Apply the identical mixture to a RANDOM subset
of matched rows of the same size, 1,000 draws. The real gate's gain must exceed the **95th
percentile** of that null. This separates "the classifier finds the fill rows" from "blending fewer
rows helps".

**Sensitivity, reported but NOT decisional:** T ∈ {0.2, 0.3, 0.4, 0.6, 0.8}. A better number at a
non-primary T is reported and changes nothing. Choosing T after seeing the grid is the exact failure
the pre-registration exists to prevent.

## 4. Multiplicity, recorded against myself

This is the **first** rule evaluated on the fillhead record. `RESULT 17a` closed the
`queue_allrows` record to further variants after four cuts; that closure applies to that record and
this is a different one. **This record gets one primary setting and one sensitivity grid. If H-G is
NOT SUPPORTED, no second gate variant will be evaluated on it** — the honest next step would be the
fitted body expert, not another threshold.

## 5. What a positive result licenses

Exactly one thing: raising Priority 2's fitted conditional-expert design above the scoped-key A/B in
the run order, on the grounds that its mechanism has independent support. It licenses **no
submission**, and it is not a measured gain for any shippable artifact.

---

# RESULT G · 2026-09-10 01:30 local · H-G SUPPORTED on all three clauses — and the magnitude is the finding

Evaluated exactly as §2/§3 fixed it, on the 339,015 matched holdout rows of
`data/cache_stand/fold_preds_fillhead.parquet`. No new fit. Primary setting T = 0.5, no other
setting read for the verdict.

| clause | measured | passes |
|---|---|---|
| (i) paired interval vs the UNGATED arm excludes zero in the gate's favour | **+0.581 s [+0.133, +1.013]** | **yes** |
| (ii) paired interval vs the BASELINE excludes zero in the gate's favour | **+0.724 s [+0.322, +1.126]** | **yes** |
| (iii) gain exceeds p95 of 1,000 same-size random subsets | **+0.7239 vs p95 +0.0609** (null mean +0.0006, null max +0.1797) | **yes** |

The negative control is decisive: the real gate's gain is **12x the 95th percentile** of blending
the same number of randomly chosen rows, and **4x the null's maximum over 1,000 draws**. The
classifier is doing the work, not the blending.

**VERDICT: SUPPORTED.** The mechanism `plans/TOP_PATH_2026_09_10.md` Priority 2 asserts is
confirmed: the body damage is caused by applying the mixture to low-`p` rows.

| arm | RMSE | MSE vs baseline |
|---|---|---|
| baseline | 225.825 | 0 |
| ungated (T = 0, the shipped RESULT 8 arm) | 225.683 | +63 |
| **gated, T = 0.5** | **225.100** | **+322** |

*(Re-measured 2026-09-10 02:10 with the shipped `max(proxy − pred, 1)` floor on every arm. The
verdict is unchanged and so is every clause: all three arms here derive from the same baseline, so
the floor cancels exactly in the paired comparisons — (i) +0.583 [+0.133, +1.003], (ii) +0.725
[+0.313, +1.129], both still excluding zero, and the gain still +322 MSE. Only the absolute RMSE
column moved, from the unfloored 226.258 / 226.114 / 225.534.)*

Per subset at T = 0.5, against the baseline: fill rows **+587 MSE**, body rows **−265 MSE** — the
body damage falls from −1,534 to −265, and that is the whole of the improvement.

## G.1 The magnitude is the real result, and it points at calibration

**+322 MSE recovers only 18% of the +1,782 MSE the fill branch is worth.** The reason is visible in
one number: at T = 0.5 the gate selects **5,053 rows (1.49%)**, 62.0% of them true fills — but that
is **10.4% recall** against 30,167 fills.

`p` ranks well and is calibrated badly. Mean `p` is **0.297 on true fills** against a base rate of
8.90%, so a probability threshold at the natural decision boundary never reaches most of the fills.
Sensitivity (reported, not decisional) is flat across the grid — T = 0.2: +262, T = 0.3: +289,
T = 0.4: +341, T = 0.6: +363, T = 0.8: +313 — which is what a mis-calibrated score looks like: no
threshold is good, because the ordering is fine and the scale is wrong.

**Consequence for the run order.** This does not make the gate a lane; at +322 MSE it is below the
plan's 1,000 MSE promotion bar and is a maintenance candidate at best, on a leaky-recipe record.
What it establishes is that the two components the plan names are the ones that carry the prize:

1. **Calibrate `p`** on out-of-fold predictions (isotonic), so the mixture can reach the other 89.6%
   of fills. `plans/TOP_PATH_2026_09_10.md` already requires out-of-fold components before any
   calibration or blend weight is fitted.
2. **Fit `mu_body` on non-fill rows only.** Even gated, the body still loses 265 MSE, because 38% of
   the gated rows are not fills. A real body expert is the only thing that removes that term rather
   than shrinking it.

## G.2 Multiplicity — this record is now closed to gate variants

Per §4, one primary setting and one sensitivity grid were evaluated. **No second gate variant will
be evaluated on the fillhead record.** The next step for this idea is the fitted conditional-expert
design on the corrected Priority 0 baseline, not another threshold.

## G.3 What this does NOT license

No submission. No claimed gain for any shippable artifact. Every number here came from a record
produced under the leaky early-stopping recipe, so both the head's and the baseline's `best_iter`
were chosen against a biased signal (`reports/PRIORITY0_ENCODING_SEPARATION.md`). The +322 must be
re-measured on the corrected baseline before it means anything for a version.
