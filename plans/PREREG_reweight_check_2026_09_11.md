# Pre-registration: RWC, does ADN's 2025 gain survive the 2026 feature shift at the four disputed airports?

**Written 2026-09-11 16:19:23 EDT (from `date`) by prc-challenge-70, BEFORE any reweighted number is computed.**
- **Owner decision, 2026-09-11:** "Reweighting check first" (option D). A was no hold (5,680); C was the corrected KS rule (2,493).
- **Parent:** `plans/PREREG_adsb_ship_v11_2026_09_11.md` (SHIP-ADN, amendments .1–.3).
- **What has been seen already:** the KS values and three gate outcomes. What has NOT been seen: any reweighted gain, weight, ESS or AUC.

## Hypothesis RWC (one per airport a ∈ D = {EDDM, EDDF, LEBL, LSZH})

> At airport a, ADN's 2025 fold-A gain over arm F on joined rows, reweighted to the 2026 joint distribution of the key ADS-B
> features and the coverage code, is at least half the unweighted gain and its day-block interval excludes zero.

D is exactly the set where the gate rules disagree. The other coverage-passing airports keep the decision that A and C share:
- EHAM, EGLL, LEMD, LFPG, LTFM ship;
- LIRF keeps F on coverage.
Their RWC numbers are reported, not acted on.

## Data (all existing; no fit touches a label except the frozen ADN record)

- **2025 rows:** fold A's scored matched rows (spent day excluded) at a, with coverage ≥ 2. Taken row-aligned from:
  - the record `data/cache_stand/fold_preds_queue_order.parquet` (y, proxy, ap);
  - `data/adsb/v2/adn_fold_preds.parquet` (F, B);
  - `data/adsb/v2/features_2025janjul.parquet` (coverage, features);
  - the stand caches (doy → date).
- **2026 rows:** `data/adsb/v2/features_2026janjul.parquet` at a with coverage ≥ 2; ap from `data/cache_stand/ranking.parquet` by MVT_ID. No 2026 label exists or is read.
- **Per-row gain:** Δ_i = (max(proxy − F, 1) − y)² − (max(proxy − B, 1) − y)², unrounded as in RESULT ADN.
  - G = mean Δ over a's 2025 joined rows;
  - G_w = Σ wΔ / Σ w.

## Weights (fixed now)

- **Features:** X = the three key features of SHIP-ADN (`first_ground_rel_aobt3`, `first_move_rel_mvt`, `first_ground_gs`) plus the coverage code (2 / 3). NaN is left to LightGBM. Joined-row missingness is ≤ 1.1% in both years.
- **Classifier:** a LightGBM binary classifier, 2026 (label 1) vs 2025 (label 0), at a. Settings:
  - objective binary, learning_rate 0.05, num_leaves 15, min_data_in_leaf 200, 300 rounds, no early stopping;
  - deterministic, seed 0, 4 threads.
  - Cross-fitted: 5 folds, random, seed 0, over the pooled rows. Each 2025 row is scored by the model that did not see it.
- **Weight:** w_i = p_i / (1 − p_i) × n25 / n26, normalised to mean 1 over a's 2025 rows.
  - No clipping. The ESS clause below is the guard.

## Clauses per airport (SHIP a ⇔ V1 ∧ C1 ∧ C2; anything else keeps F at a)

- **V1 (the weights carry information):** Kish ESS = (Σw)² / Σw² ≥ 0.20 × n25_a. Otherwise INCONCLUSIVE, which keeps F (the default is NOT WORKING).
- **C1:** the 95% day-block interval of G_w excludes 0 (lower bound > 0).
  - Dates are resampled with replacement within each month, 2,000 draws, seed 0 (`prc.pipeline.scoring.dayblock_interval` on w·Δ).
  - The weights are held fixed across draws, so classifier uncertainty is not in the interval (a named limit).
- **C2:** G_w ≥ 0.5 × G. This is ADN.7's transfer bar, "gain ≥ 0.5 × in-sample gain", reused unchanged.

Reported and not acted on: G, G_w, the interval, ESS / n, cross-fitted AUC, n25 and n26.

## Harness checks (both must pass before any airport's verdict is read; if either fails, D keeps F everywhere)

- **K0 (null, no shift):**
  - At each a ∈ D, split a's 2025 joined rows by date parity (odd vs even day of month) and treat "odd" as the target.
  - PASS iff at every a: cross-fitted AUC ≤ 0.55, AND ESS ≥ 0.90 n, AND G_w on the even rows lies inside the even rows' own unweighted day-block interval of G.
  - Guards against a machine that invents shift.
- **K+ (planted, known shift):** pooled over the 2025 joined rows of all coverage-passing airports.
  - Draw a pseudo-2026 sample of the same size, with replacement, with probability π_i ∝ exp(−z_i), seed 0. z_i = the within-airport normal score of `first_ground_gs`; NaN → 0.
  - Known target: G_true = Σ πΔ / Σ π, from the labels, which the classifier never sees.
  - The machinery (same classifier and cross-fitting, pseudo-2026 vs 2025) yields G_w on the 2025 rows.
  - PASS iff |G_w − G_true| ≤ 0.5 × |G − G_true|, i.e. it recovers at least half of the planted change. The planted change |G − G_true| is reported.
  - If it is below 2% of G, the plant is too weak to test anything: K+ is INCONCLUSIVE, and D keeps F.

## Shapes written before computing

- **TRUE (shift benign at a):** ESS well above 0.2n, G_w ≈ G (within about ±30%), interval above zero. Expected where the 2026 shift is "more stand-side first samples", the regime where ADN gained most in 2025.
- **FALSE (shift harmful at a):** G_w < 0.5 G, or the interval reaches zero. The 2026 mix sits where the stacker gained little in 2025 (e.g. code-2 mid-taxi joins with fast first samples).
- **Uninformative:** ESS < 0.2n, i.e. 2026 at a sits largely outside 2025's support. F is kept, and "the stacker has not seen this mix" is recorded.

## After the verdict (owner's pre-agreed mapping)

- **Final gate:** EHAM, EGLL, LEMD, LFPG, LTFM, plus every a ∈ D with SHIP. prc-challenge-53 registers it as SHIP-ADN.4, naming the owner's choice "D, per RWC", before building.
- **Next steps:** my dry run → rebuild → `adsb_v11_check` → the owner's upload decision (SHIP-ADN.2 still holds).
- **Projection:** SHIP-ADN's P formula, unchanged.
- **Board verdict:** by SHIP-ADN's clauses, unchanged. No threshold above moves after a number is seen. No second RWC variant is run.

## Clarification RWC.0 · 2026-09-11 16:20:38 EDT, before any number is computed

K+ runs the SAME per-airport classifier as the verdict. That is the only way "the same classifier" can see a plant drawn per
airport. The procedure:
1. For each coverage-passing airport with ≥ 1,000 2025 joined rows, draw pseudo-2026_a from a's rows with π ∝ exp(−z), n_a draws.
2. Weight a's rows by the per-airport cross-fitted classifier, normalised to mean 1 within a.
3. Pool: G_w = Σ_a Σ_i wΔ / Σ_a n_a; G_true = Σ_a n_a · G_true_a / Σ_a n_a; G = the pooled mean Δ.
The PASS / INCONCLUSIVE rule is unchanged.

## Note RWC.1 · 2026-09-11 16:20:56 EDT, before any number is computed (from prc-challenge-53's review)

- **No clause changes.** If K0 fails on its ESS clause alone, with AUC ≤ 0.55 and G_w inside the interval at every a, the
  record names that shape explicitly. The risk is 300-round null classifiers spreading p through overfitting. The mapping
  still applies (a K0 FAIL keeps F everywhere, which is option C's outcome); the owner reads the shape.
- **Named limits:**
  - The weights see 3 of the stacker's 29 inputs (plus the coverage code). A 2026 shift in the other inputs is invisible to RWC.
  - Δ is unrounded, while the ship rounds to whole seconds: a negligible difference.

## Amendment RWC.2 · 2026-09-11 16:24:04 EDT, before any REAL number is computed

The unit test on synthetic data went RED. Under a pure null (n = 4,000 per side), the registered classifier (300 rounds,
min_data_in_leaf 200) overfits: weights 0.31–5.88, ESS 0.885 n, below K0's 0.90 bar. That confirms prc-challenge-53's
warning (RWC.1). Calibration was on SYNTHETIC data only: three unit-variance features plus a coverage code; the null is two
independent samples; the shift is a −0.3 mean shift in first_ground_gs, whose true density ratio is known.

| rounds | min_leaf | null n=4k: AUC / ESS | null n=10k: AUC / ESS | known shift: AUC / corr(w, true w) |
|---|---|---|---|---|
| 300 | 200 (registered) | 0.483 / 0.895 | 0.506 / 0.948 | 0.566 / 0.752 |
| 100 | 200 | 0.490 / 0.953 | 0.505 / 0.980 | 0.574 / 0.864 |
| 300 | 1000 | 0.500 / 0.979 | 0.507 / 0.970 | 0.573 / 0.834 |
| **100** | **1000** | **0.504 / 0.990** | **0.509 / 0.986** | **0.579 / 0.902** |

**Change:** the classifier becomes 100 rounds, min_data_in_leaf 1000. That is the least overfit setting tried, and the best on
BOTH the null's noise and a known shift's recovery; the registered one is worst on both. The root cause, overfitting noise in
the weights, is fixed at its source. **Every threshold is unchanged:** V1 0.20, C1, C2 0.5, K0 0.55 / 0.90 / interval, K+ 0.5 / 2%.
No real reweighted number, weight, ESS or AUC has been computed.

# RESULT RWC · 2026-09-11 16:26 EDT (`reports/rwc.json`, `reports/rwc.console.log`; 9.9 s, 0.37 GB)

- **Rows:** 2025 joined scored 127,736; 2026 joined 166,999.
- **Harness PASS.**
  - **K0:** at EDDM, EDDF, LEBL and LSZH, AUC 0.499–0.512, ESS 0.980–0.987 n, and G_w inside the even rows' interval at every airport.
  - **K+:** planted change 954 (6.8% of G); G_w 14,883 vs G_true 14,980, so 90% of the plant was recovered.

| a | n25 / n26 | AUC | ESS/n | G | G_w | G_w / G | 95% day-block of G_w | V1 C1 C2 | decision |
|---|---|---|---|---|---|---|---|---|---|
| EDDM | 13,976 / 20,872 | 0.556 | 0.947 | 18,675 | 19,698 | 1.05 | [15,410, 24,690] | ✓ ✓ ✓ | **SHIP** |
| EDDF | 18,403 / 15,019 | 0.659 | 0.626 | 14,092 | 16,121 | 1.14 | [13,545, 18,803] | ✓ ✓ ✓ | **SHIP** |
| LEBL | 22,049 / 22,463 | 0.617 | 0.844 | 11,305 | 14,136 | 1.25 | [12,041, 16,403] | ✓ ✓ ✓ | **SHIP** |
| LSZH | 15,011 / 17,022 | 0.640 | 0.805 | 14,310 | 15,599 | 1.09 | [13,224, 18,593] | ✓ ✓ ✓ | **SHIP** |

Reported, not acted on (G_w / G): EHAM 1.16; EGLL 1.45 (ESS 0.45); LEMD 0.67 (ESS 0.46); LFPG 1.39 (n25 1,982).

**Against the shapes:** every disputed airport matches the TRUE shape. G_w lands within +5% to +25% of G, inside the ±30%
band, and every interval is far above zero. The direction is the predicted one: the 2026 mix leans toward the stand-side
first samples where ADN gained most in 2025. None is FALSE or uninformative.

**Verdict:** the 2026 feature shift is benign for ADN at EDDM, EDDF, LEBL and LSZH, as far as the three key features and
the coverage code can show.
- The weights see 3 of the stacker's 29 inputs.
- Classifier uncertainty is not in the interval.
- Each gain is a 2025 fold number, not a board price.

**Final gate, by the owner's pre-agreed mapping:** EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LSZH, LTFM; LIRF keeps F on coverage.
- Handed to prc-challenge-53 to register as SHIP-ADN.4 ("D, per RWC") before building.
- SHIP-ADN's P formula for this gate = 5,680 (computed 16:1x from the gate entries; 53 to confirm).

## Amendment RWC.3 · 2026-09-11 17:49:46 EDT: LIRF, owner-directed, before any LIRF number

The owner asked for top 5 "with v12". RLD alone (−519 if the rows are fills) leaves v12 about 40 MSE short of 5th.
LIRF was kept on F by SHIP-ADN's 0.7× COVERAGE rule (0.448 → 0.199), not by the KS rule, so it was outside D. Its question
is the same one RWC asks: does ADN's 2025 gain survive the 2026 mix of joined rows at LIRF? A coverage collapse can
change the kind of flight that is still joined.

**Scope extension:** D' = {LIRF}. The machinery, clauses and thresholds are unchanged (V1 ESS ≥ 0.20 n, C1 lower bound > 0,
C2 G_w ≥ 0.5 G; the RWC.2 classifier). The harness is K0 at LIRF (the same rule) plus the already-PASSED pooled K+.
**Honest label:** this extension is chosen after RWC's other results were seen, and it is owner-directed.
**Decision:** SHIP at LIRF iff V1 ∧ C1 ∧ C2 and K0 PASS at LIRF. Otherwise LIRF keeps F, and v12 is RLD only.
A SHIP goes to prc-challenge-53 as a SHIP-ADN amendment (the owner's decision, "with v12"), gated before any build.
The projection uses SHIP-ADN's P formula with k = 1: P_LIRF = 189.

# RESULT RWC.3 (LIRF) · 2026-09-11 17:51:21 EDT (`reports/rwc_lirf.json`, `reports/rwc_lirf.console.log`)

- **Harness PASS.** K0 at LIRF: AUC 0.521, ESS 0.979, G_w inside the interval. K+: as before, 90% of the plant recovered.
- **LIRF:** n25 11,552 / n26 5,276; AUC 0.758; ESS/n 0.339; G 12,324; G_w 9,471 (0.77×); 95% day-block interval [3,062, 16,565].
  V1 ✓ C1 ✓ C2 ✓, so **SHIP**.
- **Against the shapes:** it passes every clause but sits at the edge of the TRUE shape. The ratio of 0.77 is inside ±30%.
  ESS 0.34 n is above the 0.20 bar but not "well above". The interval is wide.
  A weaker pass than D's four: the 2026 LIRF joined mix differs from 2025's more than any other airport's (AUC 0.76).
- **Projection:** P_LIRF 189 at k = 1; about 146 if scaled by the reweighted ratio.
- (The script's last print line referenced a key that exists only in the default mode. The crash came after the JSON was
  written, and the fix is to that print line only; tests 15 passed.)
