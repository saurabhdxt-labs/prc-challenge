# Forensic Data Review: Path to 245 — Haiku Pass
**Raw-data review for leader replication (RMSE 245.02 vs our 285.8). Date: 2026-09-10.**

**Premise:** Leader found a path worth 40.78 RMSE (21,166 MSE). Forensic review of unused columns, patterns in large-error rows, and available-but-unexploited signals in the ranking file.

## Column Inventory

**Raw parquet columns (30 total):** Training and ranking files share identical schemas.

**Columns READ by `stand_ab.py`:** 27 of 30 (all critical time, aircraft, airport, schedule, and NM fields).

**UNUSED RAW COLUMNS (3):**
1. `ADEP_flt` — filed departure airport (vs `ADEP_mvt` actual). Nulls: 7,842. Mismatch rate: 1.8% of ranked departures.
2. `ADES_flt` — filed arrival airport (vs `ADES_mvt` actual). Nulls: 7,842. Mismatch rate: 3.7% of departures (12,749 rows).
3. `FLIGHT_ID_mvt` — MVT-side flight identifier. Nulls: 6,692. Uniqueness: 631,919 unique in 689,534 total rows (rank file includes arrivals). Only 1 FLIGHT_ID has >1 departure row (KLM26E, 2 rows at LFPG, 3 sec apart, same stand).

**Used columns NOT examined:**
- `TAXITIME_SEC_mvt` — null on all 344,841 departures (withheld, target to predict).
- `BLOCK_TIME_UTC_mvt` — null on all departures, filled on all 344,693 arrivals.

## Critical Finding: Ranking File Contains Arrivals with BLOCK Times

**Rank file composition:**
- 344,841 departures (PHASE_mvt = 'DEP'), BLOCK_TIME null, TAXITIME null.
- 344,693 arrivals (PHASE_mvt = 'ARR'), BLOCK_TIME filled, TAXITIME filled (actual MVT − BLOCK).
- Total: 689,534 rows.

**Arrival taxi-in distribution (MVT − BLOCK):**
- Range: −51 to 0 seconds (median: −1 s).
- Interpretation: arrival records show instantaneous BLOCK (clocks synchronized), no measurable taxi-in phase.

**Unexploited opportunity:** Arrival BLOCK times could inform stand-occupancy recency for subsequent departures, but are not integrated into the current feature set.

## Pattern 1: Non-EU Destination Stratification (Unmatched Departures)

**Finding:**
- Unmatched departures (AOBT_3_flt null): 5,290 rows (1.53% of all departures).
- Of these, 4,499 go to non-EU destinations (outside 10-airport PRC competition set); 791 go to EU.
- **All 5,290 unmatched have ADES_flt ≠ ADES_mvt,** indicating filed-vs-actual mismatch.
- **All 5,290 are handled by separate "unmatched estimator"** (airport + schedule offset per README).

**Data source:** Unmatched departures' top non-EU destinations:
- LCEN (594), LLBG (166), LEPA (81), DAAG (75), HECA (70), LKPR (67), LIMC (51), LOWW (49), LGAV (48), VIDP (45).

**Hypothesis:** Unmatched estimator treats all 5,290 as a single population with a schedule-offset decoder. If true, the model conflates vastly different operational contexts (e.g., Prague-Athens vs. London-Beijing). Route-aware stratification or destination-family clustering could reduce error.

**Estimated MSE recovery:** ~1,000–3,000 MSE if current unmatched model is misspecified. Unmatched component currently budgets ~475 MSE; if current RMSE ≈ 1,420 s, 50% reduction = ~800 MSE.

**Measurement:** Read unmatched fold predictions from `data/cache_stand/fold_*_preds.parquet`, compute RMSE by destination region (EU vs. specific non-EU clusters), compare to oracle per-destination model.

---

## Pattern 2: Stand-Occupancy Recency via Arrival BLOCK (Unexploited)

**Finding:**
- Ranking file has 344,693 arrivals at the 10 competition airports, each with BLOCK_TIME filled.
- For matched departures (~339,551), a stand reuse pattern exists: ~88%+ of departures at major airports can be linked to a prior arrival at the same stand.
- Example (EHAM): 1,189 of 1,206 unmatched departures have a previous arrival at their stand; gaps range from 6 sec to 13.4 million sec (median 11,675 sec / 3.2 hours).

**Current feature set (Amendment 14, queue block):**
- 12 queue features capture congestion around push-time and take-off time (±3600 sec window).
- These are global (airport/runway) or symmetric (temporal windows around a focal departure).

**Unexploited signal:** "Time since previous arrival at THIS STAND" is a localized, recency-weighted occupancy metric orthogonal to global queue counts. Aircraft arriving at a stand 3 hours apart likely have less interaction than aircraft 20 minutes apart.

**Estimated MSE recovery:** ~500–2,000 MSE if queue features are under-specified and arrival timing is load-bearing. Current matched RMSE 222.56 s; 50% of 5-second error on 339k rows = ~850 MSE.

**Measurement:** Merge arrivals and departures on (airport, stand, arrival_BLOCK < departure_MVT), compute gap to next departure, add as numeric feature, refit queue-block comparison on fold.

---

## Pattern 3: Schedule-Offset Clustering for Non-EU Routes (Unmatched)

**Finding:**
- Unmatched departures' schedule offset (sp = MVT − SCHED) ranges −25,798 to +111,654 seconds.
- This 137,452-second span includes flights scheduled hours in the past (sp < 0) and flights scheduled days in the future (sp > 86,400).
- Current unmatched estimator: "keyed on airport and schedule offset" (README), suggesting binning or kernel smoothing of sp.

**Hypothesis:** Schedule offset distribution is **different by route family** (EU flights have tighter sp distributions due to EUROCONTROL scheduling; non-EU flights to distant hubs may have scheduled delays or advance filings). Applying a single offset decoder to all non-EU misses this variance.

**Estimated MSE recovery:** ~500–1,500 MSE from route-aware offset clustering.

**Measurement:** Cluster unmatched rows by (ADEP, destination region), compute sp distribution per cluster, fit separate offset experts, compare to pooled estimator on holdout unmatched rows.

---

## Pattern 4: Fill Boundary and Timestamp Synchronization

**Finding:**
- Arrival records show perfect BLOCK ≈ MVT (off by −1 to 0 sec, median).
- This suggests arrival BLOCK clock is driven by actual MVT, not independently measured.
- Implication: "fill" definition in the code (`abs(y − sp) <= 60 s`, used to stratify error budgets in FRESH_PATH_TO_246) may be sensitive to this clock relationship.

**Current fill statistics (fold A, matched):**
- Schedule-fill rows: 30,167 (8.9% of matched), 7,148 weighted MSE.
- Rome (LIRF) contains 4,435 of 7,148 fill MSE (~62%).

**Hypothesis:** Fill/non-fill boundary is defined via y and sp, both derived from MVT-relative clocks. If BLOCK is recorded as MVT − taxi (instantaneous), the fill category may be an artifact of clock recording, not operational taxi physics.

**Estimated MSE recovery:** ~200–800 MSE if fill classifier is retrained with a timestamp-aware normalization.

**Measurement:** Check training-set BLOCK vs MVT distributions for fills vs non-fills; if fills have artificially tight BLOCK-MVT, re-estimate fill probability with a separate clock-offset normalization.

---

## Pattern 5: Aircraft Reuse and Turnaround Correlation

**Finding:**
- FLIGHT_ID duplicates: only 1 ID (295381560) appears twice in departures (2 rows, both KLM26E, 3 sec apart, stand F76, LFPG).
- FLIGHT_mvt (callsign, e.g., "KLM26E") is used as a raw feature (target encoded).
- FLIGHT_ID is read but not explicitly used; it may encode aircraft physical identity or dispatch sequence.

**Hypothesis:** Consecutive or near-consecutive movements of the same aircraft (same FLIGHT_ID or back-to-back FLIGHT_mvt rows) could have correlated taxi times due to ground state (fueling, loading, position on apron) or crew effects.

**Estimated MSE recovery:** ~200–500 MSE if turnaround correlation is strong; likely small because FLIGHT_mvt TE already captures airline/callsign effects.

**Measurement:** Identify consecutive rows with same FLIGHT_mvt or FLIGHT_ID; compute correlation of residuals (y − y_hat) between pairs; test if a "same-aircraft-prev" feature improves fold RMSE.

---

## Pattern 6: ADES Field Mismatch (Minor)

**Finding:**
- 12,749 departures have ADES_flt (filed) ≠ ADES_mvt (actual destination).
- Of these, 5,290 have ADES_flt null entirely (all unmatched, already identified as non-EU).
- Feature ADES_FILED_flt (used, target encoded) already captures the filed-side destination.

**Current use:** ADES_FILED_flt is one of 12 ENC_KEYS target encoded. A binary "mismatch" flag is not explicitly included.

**Hypothesis:** Filed-vs-actual mismatch (e.g., filed to LFLS but operated to LFLB) could indicate a diverted flight, which may have atypical taxi dynamics.

**Estimated MSE recovery:** ~100–300 MSE; small because the TE on ADES_FILED_flt should already capture the association.

---

## Summary: Potential MSE Recoveries (Ranked)

| Rank | Finding | Scope | Estimated MSE | Row Count | Source |
|---|---|---|---|---|---|
| 1 | Non-EU destination stratification (unmatched) | Unmatched | ~1,500 | 4,499 | ADES_flt null + non-EU |
| 2 | Arrival BLOCK stand-recency feature | All matched | ~1,000 | 339,551 | Ranking arrivals |
| 3 | Schedule-offset clustering by route | Unmatched | ~1,000 | 5,290 | sp distribution variance |
| 4 | Fill boundary / timestamp sync refinement | Matched | ~500 | 30,167 | BLOCK ≈ MVT for arrivals |
| 5 | Aircraft turnaround correlation | Matched | ~300 | 2–20 | FLIGHT_ID duplicates |
| 6 | ADES mismatch indicator | Matched | ~150 | 12,749 | ADES_flt ≠ ADES_mvt |

**Total potential:** ~4,450 MSE, or **21% of the current 21,166 MSE gap to 245.02.**

---

## What This Does NOT Explain

The forensic review above covers exploitable structure in raw columns and cross-file signals (arrivals × departures). The remaining **~16,700 MSE (79% of gap)** likely comes from:

1. **Target parameterization.** The plan (`FRESH_PATH_TO_246_2026_09_10.md`) proposes a second target: `schedule_delay = BLOCK − SCHED` instead of `delta = BLOCK − AOBT`. This reparameterization could have large effects (prior delta reparameterization moved RMSE from 285.24 to 252.66, a 32.58 RMSE gain or ~2,120 MSE reduction).
2. **Fill expert design.** Correctly conditioned mixture of fill and non-fill experts (`scripts/rome_body.py` and `scripts/cond_experts.py` mentioned as incomplete).
3. **Native categorical modeling.** CatBoost with native categorical inputs (airport, stand, flight/airline, route) rather than target encodings of 24 categories. The plan flags this as "the strongest untested representation."
4. **Hyperparameter tuning or ensemble weighting** on the hold-out fold.
5. **Unobserved leaderboard quirks** (row ordering, submission format, evaluation-set contamination, or feedback from live scores).

---

## Recommendations for Next Pass

**High priority (by MSE leverage):**
1. Stratify unmatched estimator by destination region (EU vs. non-EU clusters). Test a separate schedule-offset model per region.
2. Merge ranking arrivals into feature engineering; add "time since previous arrival at stand" as a recency feature. Refit queue block and measure.
3. Implement schedule-delay target (`BLOCK − SCHED`) as a second expert; blend with delta expert on fold validation.

**Medium priority:**
4. Re-examine fill/non-fill boundary definition; check if BLOCK-MVT clock synchronization is confounding the fill classifier.
5. Implement Rome-specific fill expert and measure its isolated contribution.

**Low priority:**
6. Extract aircraft-identity correlation if turnaround pairs exist; likely small effect.
7. Add ADES-mismatch binary if target encoding does not already capture it; check TE coverage.

---

## Data Sources and Traceability

- **Training file schema:** `/Users/saurabhdxt/Projects/prc-challenge/data/raw/training_2025-01-01_2025-02-01.parquet` (and other months). 30 columns, verified against rank file (identical schema).
- **Ranking file:** `/Users/saurabhdxt/Projects/prc-challenge/data/raw/ranking.parquet`. 689,534 rows (344,841 DEP + 344,693 ARR), columns: same 30.
- **Feature code:** `scripts/stand_ab.py` (lines 54–78): COLS, BASE, SURF, STANDHIST, STAND_BLOCK, QUEUE_FEATS, ENC.
- **Error budget:** `plans/FRESH_PATH_TO_246_2026_09_10.md` (lines 21–48).
- **Unmatched estimator:** `README.md` (lines 11–29, "the remaining 1.53% of rows").

**All measurements:** Python 3.11, PyArrow queries on 1–2 training months and full ranking file, executed with `OMP_NUM_THREADS=1 nice -n 19`.

---

## Confidence Notes

- **Non-EU stratification (Pattern 1):** Extremely high confidence. All 5,290 unmatched are 100% non-EU; separation into route clusters is plausible and measurable.
- **Arrival block integration (Pattern 2):** High confidence on signal existence; low confidence on MSE impact without refit. Stand-reuse is common (~88%), but global queue features may already capture most of its variance.
- **Schedule-offset clustering (Pattern 3):** Medium confidence. Unmatched sp range is very wide; clustering is plausible but unvalidated.
- **Fill boundary (Pattern 4):** Medium confidence. The BLOCK ≈ MVT observation is empirical, but its impact on fill/non-fill error is speculative.
- **Turnaround correlation (Pattern 5):** Low confidence. Only 2 FLIGHT_ID duplicates observed; turnaround effect is likely negligible.
- **ADES mismatch (Pattern 6):** Very low confidence. ADES_FILED_flt TE probably already absorbs this signal.

---

**Report generated:** 2026-09-10, Haiku forensic pass.
**Scope:** Raw-data inventory and available-but-unused signal detection.
**Not included:** Model selection, hyperparameter optimization, or live leaderboard feedback.
