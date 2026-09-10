# Priority 2 — the fill lane, priced from the record before any new fit

**2026-09-10 01:10 local.** Follows `plans/TOP_PATH_2026_09_10.md` Priority 2. No new model was
fitted. Every number below is recomputed from `data/cache_stand/fold_preds_fillhead.parquet` and
`reports/lgbm_fold_fillhead.json` — the record RESULT 8 declared NOT WORKING.

Nothing beneath the running arms queue was touched.

---

## 1. What RESULT 8 actually measured

The arm implemented `treatment = p·sp + (1−p)·m(X)` where `m` is the all-rows model. Its pooled gain
was +0.142 s and it was verdicted NOT WORKING. **The pooled number hides two large, opposite
effects.** Recomputed per subset, with `F = (|y − sp| ≤ 60)` — the schedule-fill definition the arm
itself used, and reading the parquet's stored delta form (`ŷ = proxy − pred`):

> **CORRECTED 2026-09-10 02:05, before this file was acted on.** The first version of this table
> compared the stored `treatment` column — which has `max(proxy − pred, 1)` applied inside it — against
> a `baseline` I recovered WITHOUT that floor. Floored against unfloored, on 23 rows where the floor
> binds. The tell was visible and I explained it away: my pooled baseline read 226.258 against the
> record's own 225.825, and I attributed the 0.43 s to seed averaging instead of chasing it. That is
> the exact failure this repo's ledger documents — a known correction in the repo, unapplied, with a
> mechanism invented for the residue. Found by an adversarial review of this session's diff.

| subset | rows | baseline RMSE | treatment RMSE | MSE at the 2026 matched weight |
|---|---|---|---|---|
| non-fill (`F=0`) | 308,848 | 218.714 | 223.051 | **−1,719** |
| **fill (`F=1`)** | 30,167 | **288.725** | **251.047** | **+1,782** |
| pooled | 339,015 | 225.825 | 225.683 | **+63** |

Both arms now carry the shipped positivity floor, and the pooled baseline reproduces
`reports/lgbm_fold_fillhead.json`'s `arms/baseline/rmse` = 225.8254 exactly.

**What changed and what did not.** The prize — the fill branch at **+1,782 MSE** — is unchanged in
substance (it was +1,789). The body damage is **larger** than first reported, −1,719 rather than
−1,534, which strengthens rather than weakens the case that the blend is the broken part. What was
materially wrong is the NET: the old arm delivered **+63 MSE**, not +256.

For scale — and this mixes instruments, so read it as an order of magnitude and nothing more: the
fill figures above are **fold-A** MSE at the 2026 matched weight, while the v5 shipment's **886 MSE**
is a **board** delta. This file's own rules forbid building a total across instruments; the sentence
is kept because the ratio is informative about where to spend a day, and it enters no arithmetic.

## 2. The classifier is not the problem

| | measured |
|---|---|
| holdout AUC | **0.8957** |
| ES AUC | 0.8930 |
| decile of `p` → actual fill rate | 0.000, 0.000, 0.000, 0.002, 0.007, 0.023, 0.063, 0.139, 0.241, **0.416** |

Monotone across all ten deciles with a 0 → 0.416 spread. **Fills are identifiable.** The earlier
closure of the fill lane rested on a pooled RMSE, not on the classifier, and the classifier is good.

One defect visible in the same numbers: mean `p` is **0.297 on true fills** against a base rate of
8.90%. It ranks well but is under-confident in the large, so the mixture is systematically
under-weighting `sp` where it should commit. Calibration on out-of-fold predictions is a free
sharpening step the old arm never applied.

## 3. Why the blend damages the body — the mechanism, stated exactly

`m(X)` is trained on all rows, so it already approximates `E[y|X]`, which is itself a mixture:

```
E[y|X] = p(X)·E[y|X,F=1] + (1−p(X))·E[y|X,F=0]
```

The arm then computes `p·sp + (1−p)·m(X)`. On a body row this **adds a second dose of fill
behaviour** to a prediction that already contains one. The body expert the formula needs is
`E[y|X,F=0]`, and `m(X)` is not it. That is the whole of the −1,534 MSE.

Confirmed independently on the current v5 record: within deciles of our predicted delta, the true
body-only mean `E[delta|X,F=0]` sits **above** our prediction in all ten deciles (bias −4.5 s to
−18.7 s), while the bias against the all-rows mean is near zero (−1.2 s to −7.1 s). The model is
sitting on the mixture mean, exactly as the algebra says.

A second observation from the same cut: **the fill rate is nearly flat across our model's predicted
delta — 6.8% to 10.4% across deciles.** The incumbent architecture carries almost no information
about fill membership in its output. A dedicated `p(X)` at AUC 0.896 is information the pipeline
currently cannot access at all.

## 4. What this licenses, and what it does not

**Licenses:** fitting the Priority 2 design —

```
p       = P(F=1 | X)                       # exists, AUC 0.896, needs calibration
mu_fill = sp + E[y − sp | X, F=1]          # shrunk, training-only
mu_body = proxy − E[delta | X, F=0]        # fitted on NON-FILL ROWS ONLY  <- the untested piece
new     = p·mu_fill + (1−p)·mu_body
```

**Does not license** a claimed gain of 1,782 MSE. That figure is the value of the fill branch
*conditional on the body expert being at least as good as the current model on body rows*, and the
body expert has never been fitted. It trains on ~9% fewer rows against a shifted distribution; it
may land better or worse than the incumbent's 218.714. **The +1,782 is the prize, not the
prediction.**

Two ways this could still return nothing:

1. `mu_body` underperforms `m(X)` on body rows by more than the fill branch gains.
2. The fill gain does not survive the corrected encoder (Priority 0). Every number in this file was
   produced under the leaky early-stopping recipe, so `best_iter` for both the head and the baseline
   was chosen against a biased signal.

Both are reasons to run it, not reasons to skip it.

## 4b. Rome IS the experiment — measured 2026-09-10 06:35

Per-airport decomposition of the fill-row gain and the body-row loss, both floored:

| | LIRF | the other nine | total |
|---|---|---|---|
| fill-row gain | **+1,439 (80.7%)** | +343 | +1,782 |
| body-row loss | **−1,208** | −511 | −1,719 |
| **net** | **+231** | **−168** | **+63** |

**Excluding Rome does not de-risk this arm; it destroys it.** An ex-LIRF variant nets **−168 MSE**
against the Rome-inclusive **+63**. The `--exclude LIRF` arm registered in
`plans/PREREG_encoder_and_experts_2026_09_10.md` is therefore a diagnostic, **not a fallback**, and
the Rome-inclusive arm is the primary. My earlier framing had this backwards.

Rome carries 80.7% of the opportunity **and** 70% of the damage. The conditional design's entire job
there is to keep the +1,439 while a properly fitted body expert removes the −1,208.

**And the +1,782 is not a model gain.** It is the improvement on true-fill rows before the losses
elsewhere; the arm as built nets +63. The number with a control behind it is the gated rule's
**~322 MSE** (RESULT G, beating its null by 12x). Quoting 1,782 as an expected gain would be
quoting a subset.

## 5. Order

This does not jump the queue. Priority 0's wiring and the baseline re-run come first, because a
conditional-expert A/B measured against a contaminated baseline would have to be re-run anyway.
Then this, as the first substantive bet, alongside Priority 1.

## 6. Reproduce

```bash
OMP_NUM_THREADS=1 nice -n 19 python3.11 -B - <<'PY'
import pandas as pd, numpy as np
m = pd.read_parquet('data/cache_stand/fold_preds_fillhead.parquet')
m['F'] = (m.y - m.sp).abs() <= 60
for c in ('baseline', 'treatment'):
    # DELTA form AND the shipped positivity floor: lgbm_fold.taxi_time is max(proxy - pred, 1).
    # Omitting it on one arm only is what made the first version of this file wrong.
    m['r_' + c] = np.maximum(m.proxy - m[c], 1.0) - m.y
for lab, g in m.groupby('F'):
    b, t = (g.r_baseline**2).sum(), (g.r_treatment**2).sum()
    print(lab, len(g), np.sqrt(b/len(g)), np.sqrt(t/len(g)), 0.9846596*(b-t)/len(m))
PY
```

---

## 7. A weakness in a shipped constant, found while testing the guard (2026-09-10 01:45)

`lgbm_submit.FILL_HEAD_MIN_ES_AUC = 0.5`, and `fit_fill_head` refuses a head whose stopping AUC is
`<= 0.5`. **A classifier that has learned nothing clears that bar roughly half the time by chance.**

Measured while writing `tests/test_cond_experts.py`: a LightGBM binary head on 3,000 rows of pure
noise, with labels drawn independently of every feature, produced a stopping AUC just above 0.5 and
**passed the guard**. The test now raises the threshold to 0.70 to exercise the branch.

The shipped constant is **not changed here.** It is a registered value in code the running arms
queue depends on, and changing it mid-queue is exactly the kind of silent edit this project's rules
forbid. Recorded so it is a decision rather than an oversight:

- **Risk:** any fill-head arm whose classifier is genuinely uninformative can pass the guard and go
  on to blend a coin-flip weight into predictions. The measured head is not in that position
  (AUC 0.8957), so nothing already shipped is affected.
- **Fix when the queue is clear:** set the floor from the sampling distribution of AUC under the
  null at the stopping set's size, not from the value 0.5 — e.g. the 95th percentile of permuted
  labels, which for n = 344,840 stopping rows is a few thousandths above 0.5, and for a small
  smoke fold is much further above it. That makes the guard scale with the evidence available.
