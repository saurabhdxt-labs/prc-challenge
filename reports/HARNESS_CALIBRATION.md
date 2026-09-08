# Is the Jan+Jul 2025 fold calibrated against the 2026 leaderboard?

Adversarial review, 2026-09-08 (leaderboard pulled ~17:30 UTC). Every number below was produced by a
script under the session scratchpad
(`.../scratchpad/{lb_analysis,extract,folds,burden,extra_rungs}.py`, outputs in the matching `.out`
files). Runs used `OMP_NUM_THREADS=1 nice -n 19 python3.11`; peak RSS of the heaviest run was
**1.50 GB** (`resource.getrusage`). Nothing under the repository was modified except this file.
No gradient-boosting model was fitted. The conclusion under attack was:

> "Our model scores ~306 on held-out Jan+Jul 2025, the leader scores 260.93, so we are ~30th of 71 and
> winning needs a 27% improvement on matched rows."

## 0. Verdict in one paragraph

**The 2025 Jan+Jul fold is a FAIR estimate of the 2026 score, calibrated to within about ±5 s at the
naive-predictor level, and the fold is not pessimistic.** Four different reproducible naive
predictors can be identified on the live leaderboard (a global constant, the all-rows mean including
arrivals, the per-airport mean, and a stand-level mean) and each lands within 0.5–4 s of the same
predictor scored on our fold — in both directions. The strongest datum is exact: two teams, one of
them **`merry-quicksand_v1` at 15:11:13 UTC today**, scored **689.6901** to four decimals, which is a
constant-prediction submission; on our fold a constant near the mean scores **686.1–686.5**. So
2026's total variance is, if anything, **0.5% higher** than the fold's. Half the monster burden is
ruled out: it would require a non-monster RMSE of ~611 s, when no 2025 month exceeds 475. Jan+Jul
IS the hardest of the six two-month folds by a wide margin (686 vs 437–576 for a constant), but
2026 matches Jan+Jul, not the other folds — so the other folds are the optimistic ones. Two
corrections to the conclusion itself: the leader is now **248.48** (youthful-giraffe v13, today
17:21 UTC), not 260.93, and 306 ranks **27th of 74** on today's board.

## 1. Reproduction of the ladder (does not depend on the caller's numbers)

`folds.py` re-runs `prc.baselines.ladder_report` on the admissible rows (`prc.labels.clean`,
2,084,659 rows, 202 above 3 h — identical to the handoff), fitted on the ten other months, scored on
Jan+Jul 2025 (n = 344,336):

| rung | RMSE | 95% row bootstrap (1,000) |
|---|---|---|
| global mean | **686.53** | [604, 782] |
| airport | **659.91** | [575, 747] |
| airport × runway | **653.06** | [558, 749] |
| airport × runway × stand | **628.98** | [539, 731] |
| + off-block hour | 627.61 | [533, 717] |

The caller's 686.2 / 660.0 / 653.1 / 629.1 reproduce (the handoff's 686.6 and the prompt's 686.2 are
both within rounding of 686.5). The bootstrap intervals are ±14% because 6 rows carry 29.7% of the
squared error; this is why a row bootstrap of a fold score is nearly useless for calibration, and
why the leaderboard comparison below is the right instrument.

Constant predictors on the fold (`folds.py`, `burden.py`): std(y) = **686.07**; c = 10-month mean
(987.2) → 686.53; c = 12-month mean (991.4) → 686.39; c = mean ± 50 → 686.7 / 689.7; c = mean ± 100
→ 690.6 / 696.7; c = median (912) → 693.4; c = 0 → 1,223.0; c = **all-rows raw mean including
arrivals (764.1) → 729.62**.

## 2. The leaderboard, in full

`lb_analysis.py` paginated the API with `nextCursor`: **441 submissions, 74 teams**, all with
`usedPairs = 344841`, processed 2026-09-04 09:58 to 2026-09-08 17:21 UTC. Team-best quantiles:
min 248.48, 5% 269.1, 25% 295.0, **median 328.0**, 75% 512.0, 95% 626.7, max 689.69. First-submission
quantiles: min 265.8, 25% 340.5, median 422.4, 75% 562.2, 95% 668.2, max 726.25.

### 2a. Rung matches — the calibration evidence

| naive predictor | our fold (2025 Jan+Jul) | leaderboard (2026) | who | gap 2026 − fold |
|---|---|---|---|---|
| constant near the mean | 686.1–686.5 (std 686.07) | **689.6901 ×2, identical to 4 dp** | nice-umbrella_v1 (09-06), **merry-quicksand_v1 (09-08 15:11 UTC)** | +3.2 to +3.6 |
| all-rows mean incl. arrivals (c ≈ 764) | 729.62 | 726.25, 726.70 | tidy-nugget_v1, nice-umbrella_v3 | −2.9 to −3.4 |
| airport mean (variants 659.1–659.9; airport median 666.0; airport × month 657.0) | 659.91 | 660.53, 660.89, 661.03, 661.16, 661.57, 663.04 | nice-umbrella v4/v5/v7, gracious-tractor_v2, unique-rose_v1, victorious-quartz_v1 | +0.6 to +3.1 |
| airport × runway (653.1) | 653.06 | 650.59 | enthusiastic-snowflake_v1 | −2.5 |
| stand-level mean (629.0; hour 627.6) | 628.98 | 625.34, 625.43, 626.14, 626.46, 627.65 | kind-mango_v15, sincere-donkey_v1, delightful-dragon_v3, enthusiastic-apple_v29, organized-hedgehog_v1 | −1.3 to −3.6 |

Two identical scores to four decimals means two identical prediction vectors; the only vector two
independent teams produce identically is a constant (the sensitivity of RMSE to the constant is
Δc²/(2·RMSE), so any c within ±20 s of the 2026 mean gives the same four decimals). The submission
template is 344,841 NaNs (`template.txt`), so 689.6901 is either the template with NaN imputed by the
scorer or the global mean submitted by both — either way a constant, and either way it bounds
**std(y₂₀₂₆) ≤ 689.69**, with equality if c is near the mean. Against std(y) = 686.07 on our fold,
the 2026 evaluation has the **same total variance to within +0.5%**. The airport-mean cluster
(six submissions from four teams within 2.5 s of each other) says the same thing one level down.

A cluster at **559.97–565.7** (kind-mango ×10, joyful-goblin ×7, kind-earthquake_v1,
trustworthy-unicorn_v1, versatile-radio) matches **no** group-mean predictor I could construct
(`extra_rungs.py` tried 16 cell definitions with mean/median and min_count 1–30: every one scores
613.6–666.0 on the fold; the AOBT proxy blended with the airport mean scores 640.6–652.0). It is
most likely a fitted model with no stratum handling; it is not usable for calibration.

### 2b. Distribution shape

All-submission histogram: 57% of submissions are in 260–340; a second mode sits at 560–640
(the naive-baseline shelf). Team bests: 23 teams below 300, 12 in 300–320, 7 in 320–340. Today
**306 ranks 27th of 74; 300 → 24th; 290 → 16th; 280 → 9th; 270 → 5th; 260 → 2nd** (`folds.py`).

## 3. Attack 1 — the monster burden

### 3a. What the fold actually contains

Jan+Jul 2025: 60 rows > 3 h (Jan 10, Jul 50; 54 of the 60 at LIRF, 56 in the unmatched stratum).
Predicted at the airport mean they contribute **500.5 s** in quadrature (502.3 s at a flat 1,000 s);
the non-monster rows contribute 429.8 s. The caller's "502" reproduces.

### 3b. Plug-in estimate of the 2026 burden and its sampling distribution (`burden.py`)

For each cell (airport × `sp` bucket, `sp = MVT_TIME − SCHED_TIME`, computable on every scored row)
the 2025 unmatched-stratum monster rate and the empirical squared errors of its monsters at a
1,000 s prediction are applied to the 2026 unmatched cell counts; the matched stratum uses its
pooled rate (16 monsters in 2.06 M rows). Counts are simulated Poisson per cell, squared errors
resampled from the cell's monster rows, 4,000 draws. Nine variants (three training windows × three
cell definitions):

| training window | cells | E[#monsters 2026] | point | sim median | 10–90% | 5–95% | P(B ≤ 250) | P(B ≤ 350) |
|---|---|---|---|---|---|---|---|---|
| 12 months | airport | 50.5 | 417 | 411 | [331, 494] | [306, 518] | 0.004 | 0.16 |
| 12 months | airport × coarse sp | 61.4 | 489 | 485 | [403, 567] | [383, 588] | 0.000 | 0.015 |
| 12 months | airport × fine sp | 61.5 | 480 | 475 | [400, 553] | [378, 575] | 0.000 | 0.017 |
| Jan+Jul only | airport | 57.9 | 490 | 486 | [407, 564] | [382, 585] | 0.000 | 0.016 |
| Jan+Jul only | airport × coarse | 56.8 | **504** | 501 | [424, 579] | [400, 600] | 0.000 | 0.006 |
| Jan+Jul only | airport × fine | 56.0 | 501 | 498 | [423, 574] | [404, 596] | 0.000 | 0.004 |
| 10 months excl. Jan+Jul | airport | 48.1 | 392 | 385 | [305, 470] | [284, 496] | 0.011 | 0.28 |
| 10 months excl. Jan+Jul | airport × coarse | 64.2 | 493 | 488 | [407, 574] | [385, 598] | 0.000 | 0.013 |
| 10 months excl. Jan+Jul | airport × fine | 64.3 | 479 | 473 | [398, 552] | [376, 575] | 0.000 | 0.020 |

Sensitivity: the airport-only cells (which ignore that the 2026 file's `sp > 3 h` counts differ from
2025's) are the only variants below 470, and they are wrong on their face — the burden is carried by
rows with `sp > 3 h` and the 2026 file has 668 such unmatched rows against 383 in Jan+Jul 2025. Every
variant that conditions on `sp` gives a point estimate of **479–504** with a 10–90% interval of
roughly **[400, 575]**. The caller's 518 is at the upper edge of the point estimates, not outside.

Self-check: the same machinery predicting the 2025 Jan+Jul burden from the other ten months gives
476–482 (sim median 472–476) against the realised 500.5 — the method under-predicts by ~5%, so if
anything the 2026 numbers above are slightly low.

**Could 2026 plausibly have half the burden (≈250 s)?** P(B ≤ 250) is ≤ 0.011 in every variant
and 0.000 in every `sp`-conditioned one. P(B ≤ 350) ≤ 0.02 when `sp` is used.

### 3c. The leaderboard-implied burden — independent of any 2025 rate

Airport-mean submissions score S = 660.5–663.0 on 2026. With B² = S² − NM² and NM the non-monster
RMSE under the airport predictor:

| NM assumed | 2025 basis | implied B₂₀₂₆ |
|---|---|---|
| 346 | best 2025 month (April) | 563 |
| 430 | Jan+Jul 2025 | **502** |
| 475 | worst 2025 month (February) | 459 |
| 611 | *required for B = 250* | 250 |

No 2025 month has NM above 475, so B₂₀₂₆ = 250 needs a non-monster regime 29% worse than the worst
month of 2025 while the total stays exactly on the fold's — not plausible. The same identity rules
out the opposite fear: EHAM's 2026 unmatched `sp > 3 h` count is **297** (vs 62 in Jan+Jul 2025); if
those rows were schedule-filled the airport-mean score would be ~790, not 660.5. They are not fills,
which is what the stratum report assumed.

Composition note for the model-level transfer: on the fold the model's residual monster error is
~150 s in quadrature (STRATUM_MONSTERS §4: 153 s under E0 at 312.4 total). Scaling that by the
plug-in range [400, 575]/490 gives [122, 176] and a 2026 model score of **sqrt(306² − 150² + [122,
176]²) = [298, 314]** from monster sampling alone; Family B alone (λ ≈ 2.4 rows, 115 s) has a Poisson
10–90% of 75–167 s.

## 4. Attack 2 — is Jan+Jul a representative fold?

`folds.py`, each fold scored with rungs fitted on the other ten months:

| fold | n | rows > 3 h | std(y) | std non-monster | global | airport [95% boot] | ap×rwy | stand [95% boot] | burden (airport pred) | non-monster RMSE | top-6 rows' SSE share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **01+07** | 344,336 | **60** | **686.1** | 467.5 | **686.5** | **659.9** [569, 760] | 653.1 | **629.0** [537, 720] | **500.6** | 430.0 | 0.297 |
| 02+08 | 334,829 | 37 | 576.2 | 461.5 | 576.2 | 546.9 [473, 646] | 534.6 | 506.7 [425, 618] | 343.5 | 425.6 | 0.280 |
| 03+09 | 348,355 | 28 | 516.1 | 409.3 | 516.2 | 481.9 [414, 559] | 473.6 | 443.4 [371, 525] | 313.0 | 366.5 | 0.323 |
| 04+10 | 360,910 | 24 | 436.6 | 401.5 | 436.7 | 397.1 [372, 430] | 385.5 | 348.9 [321, 387] | 170.0 | 358.9 | 0.139 |
| 05+11 | 347,495 | 23 | 499.6 | 408.7 | 499.6 | 468.5 [394, 550] | 456.8 | 425.9 [340, 515] | 286.5 | 370.8 | 0.333 |
| 06+12 | 348,734 | 30 | 536.2 | 427.5 | 536.2 | 504.5 [432, 588] | 496.8 | 465.9 [393, 548] | 322.1 | 388.3 | 0.319 |

Per month (fixed airport mean over all 12 months): July has **50** of the year's 202 monsters
(47 at LIRF), burden 566.9, RMSE 714.6; January has 10, burden 403.2, RMSE 584.3; the quietest month
is October (burden 121.6, RMSE 390.0). Non-monster RMSE by month ranges 345.9 (April) to 475.4
(February); Jan 422.9, Jul 435.2.

**So yes, Jan+Jul is the outlier — it is the hardest fold by 110 s for a constant and by 113 s at
the airport rung, and its burden is 46% above the next fold's.** But this cuts the other way from
what the attack hoped: the 2026 evaluation is January and July, the leaderboard's constant score
(689.69) matches the Jan+Jul fold (686.1–686.5) and not the 437–576 of the other folds, and the
seasonality is structural (LIRF's schedule-fill monsters peak in July; the unmatched rate is 2.05% in
July against 0.61–1.35% in the other months, and the 2026 file's unmatched rate is 1.534% vs
1.545% on the fold). Any fold other than Jan+Jul would have been **optimistic by 100–250 s** at the
naive level. The headline is measured on the right fold.

## 5. Attack 3 — "the leader's 260.93 implies matched-row RMSE ≈ 185"

The derivation is `sqrt(T² − F²)` with T the leader's total and F an assumed floor on the rows the
leader cannot fix. It rests on:

1. **F ≈ 185 s.** Nothing in the repository supports 185; STRATUM_MONSTERS §4 puts the irreducible
   floor at ≈145 s (Family B 115–119 + 24 h slips 80–91, in quadrature). Sensitivity is brutal
   because T and F are comparable: at T = 260.93, F = 100 → 241, 145 → **217**, 185 → 184, 200 → 168,
   230 → 123. The "≈185" is entirely the choice of F.
2. **The floor is the same for the leader as for us.** It is not: whether a 24 h date-slip row costs
   the leader 86,000 s or 1,000 s depends on their prediction, and the leader's total already
   proves their burden on all monster rows is < 260.93 (a team predicting ~1,000 s on every
   monster would score ≥ 500).
3. **The evaluation's monster composition equals the fold's.** Poisson with λ ≈ 2.4 for Family B —
   the 10–90% range of that term alone is 75–167 s.
4. **Quadrature over disjoint row sets** — the only assumption that holds exactly.
5. **T = 260.93.** Stale: the leader is 248.48 as of 17:21 UTC today; at that T the same table reads
   F = 145 → 202, F = 185 → 166.

Verdict: the 185 is not a measurement; it is F restated. The defensible statement is "the leader's
matched-row RMSE lies somewhere in 170–240 depending on an unobservable floor", and the leader's
number will keep falling.

## 6. What this does and does not establish

Established (external evidence, leaderboard vs fold):
- Total variance of the 2026 evaluation ≈ Jan+Jul 2025 fold, +0.5% (constant-predictor match,
  exact to four decimals across two teams including ours).
- Airport-level and stand-level naive predictors transfer within ±4 s, in both directions.
- The 2026 monster burden under a naive prediction is ~480–505 s (plug-in) / ~460–560 s
  (leaderboard-implied); half the fold's burden has probability ≤ 1%.
- Jan+Jul is the hardest 2025 fold and the correct analogue; other folds are optimistic.

Not established, and cannot be without a submission:
- The transfer of a *model* score depends on the 2026 mix of fill / slip / Family B monsters,
  which the naive rungs cannot resolve; the model-level range from monster sampling alone is
  roughly [298, 314] around 306.
- What predictor the 560–565 cluster is.
- Whether 689.6901 was a submitted constant or a NaN template imputed by the scorer (the bound on
  std(y₂₀₂₆) holds either way).

One operational note: a `merry-quicksand_v1` submission was processed at 15:11:13 UTC today and
scored 689.6901. Either the 403 has cleared or someone submitted through another route; the caller
should know that our team now has a scored constant submission on the public board.

## 7. Commands

```
curl -sS ".../leaderboard?limit=200[&cursor=…]"  → scratchpad/lb_page{1,2,3}.json (441 items, 74 teams)
OMP_NUM_THREADS=1 nice -n 19 python3.11 lb_analysis.py > lb_analysis.out
OMP_NUM_THREADS=1 nice -n 19 python3.11 extract.py     > extract.out     (peak RSS 1.05 GB)
OMP_NUM_THREADS=1 nice -n 19 python3.11 folds.py       > folds.out       (22.6 s)
OMP_NUM_THREADS=1 nice -n 19 python3.11 burden.py      > burden.out
OMP_NUM_THREADS=1 nice -n 19 python3.11 extra_rungs.py > extra_rungs.out (peak RSS 1.50 GB)
```
Epoch seconds were taken as `ts.astype("int64") // 1_000_000` throughout.
