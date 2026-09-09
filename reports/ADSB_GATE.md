# ADS-B as a second measurement of the target — the stand-geometry gate

**Status: CLOSED — NO-GO, 2026-09-08. The ten-airport census (Amendment 6) returned 2 of 10
airports clearing the bars against a pre-registered threshold of 6.** Worth **2,243 global MSE**
(8.8% of the remaining gap), and applying it at serve time would require a ~190 GiB / ~20 h 2026
ingest plus unresolved eligibility. **The 2026 ingest is not justified and is not being run.**
The mechanism is real and the earlier per-airport numbers stand; the lane fails on receiver
coverage, not on the idea. Census detail in section "The census" below.

## Why the previous closure was wrong

The evening handoff closed ADS-B as NOT WORKING on two numbers: raw adsb.lol traces gave taxi
RMSE 575 against the model's 237, and OPDI's accurate `exit-parking_position` event covered only
0.7% of rows. Both are real measurements. Neither supports the closure.

**The 0.7% is OpenStreetMap's coverage, not the sensor's.** OPDI derives that event by matching
state vectors against OSM `aeroway=parking_position` polygons. Coverage is governed by how
completely European aprons are mapped in OSM, not by whether the aircraft was visible. Measured
here: the ADS-B archive joins to **97–99%** of departures at all three pilot airports.

**The raw-trace number was contaminated by a join defect.** `FLIGHT_mvt` at EHAM is the IATA
commercial number (`KL0689`); ADS-B transmits the ICAO radio callsign (`KLM836`). Measured on
2025-01-09:

| airport | departures | `FLIGHT_mvt` matches | `CALLSIGN_flt` matches |
|---|---|---|---|
| EDDM | 351 | 100.0% | 100.0% |
| EGLL | 634 | 99.8% | 99.8% |
| **EHAM** | 602 | **2.2%** | **99.5%** |

`CALLSIGN_flt` is already in the file and is the correct key; no IATA→ICAO mapping is needed.
Whether the original 575 figure used `FLIGHT_mvt` cannot be confirmed — that code was in a wiped
scratchpad — but any pooled ADS-B measurement keyed on `FLIGHT_mvt` had a third of its rows joined
to nothing. **Fifth instance of this project's "pooling hides the truth" landmine.**

## The measurement that matters

Estimating off-block from the time the transponder first appears is useless under squared loss —
the core is tight but the tail is enormous, because the transponder often comes on away from the
stand:

| estimator | EDDM RMSE | EHAM RMSE | EGLL RMSE |
|---|---|---|---|
| first sample | 922 | 766 | 952 |
| first on-ground sample | 916 | 765 | 941 |
| first sample with ground speed > 3 kt | 551 | 617 | 878 |
| last stationary sample before motion | 392 | 858 | 614 |

A **stand-geometry gate** changes the picture completely. Stand centroids are learned from the
data — the position of each aircraft at its own recorded off-block, aggregated per (airport,
stand), **leave-one-flight-out** so a flight never contributes to its own gate. The estimate is the
last ADS-B sample inside radius R of *this flight's own stand*. No OpenStreetMap, so coverage is
not capped by OSM completeness. RMSE below is bias-corrected, as a calibrated feature would be.

| R (m) | apt | coverage | bias | **RMSE** | robust sd | within ±120 s | model RMSE |
|---|---|---|---|---|---|---|---|
| 100 | EDDM | 72.9% | 225 | 269 | 162 | 53.5% | 174 |
| 100 | **EHAM** | 65.8% | 246 | **127** | 98 | 68.4% | **173** |
| 100 | EGLL | 22.9% | 391 | 235 | 205 | 48.3% | 249 |
| 200 | EDDM | 76.6% | 319 | 256 | 75 | 80.3% | 174 |
| 200 | **EHAM** | 75.9% | 288 | **150** | 96 | 78.8% | **173** |
| 200 | EGLL | 24.4% | 460 | 238 | 157 | 57.4% | 249 |

**At EHAM the gated sensor beats our model on two-thirds of rows** (127 against 173). That is the
first quantity in this project that improves on the model using information independent of it.

230 of 312 stands had ≥2 off-block fixes on a single day, which is what the leave-one-out centroid
needs. A full year of labelled departures would sharpen every centroid by orders of magnitude.

## Honest projection

Inverse-variance blending the sensor with the model on covered rows, applying the measured gains at
the three pilot airports at their real scored weights and assuming **zero gain at the other seven**:
**≈ 2,700 global MSE**. If the remaining seven behave like these three: **≈ 8,000**.

That is a lower bound and an extrapolation respectively, not a measurement. What is measured is the
per-airport table above.

## What would falsify it

- Channel quality is **heterogeneous by a factor of two** (EHAM 127, EDDM 269) and coverage by a
  factor of three (EGLL 22.9%, EDDM 76.6%). Seven airports are unmeasured and could be EGLL-like.
- The blend assumes sensor and model errors are **independent**. They may share a common cause
  (irregular operations degrade both), in which case the projection overstates.
- One day, one season. Winter de-icing at EDDM and summer ATFM at LIRF are exactly the regimes
  where a stand gate could behave differently, and the scored months are January and July.
- Bias is large (225–460 s) and varies by airport and radius; it is correctable per stand only if
  enough labelled fixes exist per stand.

## Eligibility

**Not yet closed; do not build a submission dependency before it is.** ADSB.lol publishes under
ODbL 1.0 and the challenge permits documented open datasets, which is the safe path. OpenSky's own
historical database carries Terms of Use restricting it to non-profit research and forbidding
redistribution, which conflicts directly with the prize condition to open-source added data under
GPLv3 — it must not be used without written permission. OPDI names no dataset licence at all.
These licence readings come from a review agent and are **not independently verified here**.

## Reproduction

`scripts/analysis/` does not yet contain this analysis; it was run from session scratchpad scripts
against `data/adsb/{pilot_truth,adsb_2025-01-09}.parquet`. Promoting it is part of the census.
Landmine: `.astype("int64")` on these stamps yields MICROseconds — this bug produced a clean-looking
0% coverage on the first run. Use `(t - epoch).dt.total_seconds()` with a `tz="UTC"` epoch and
assert the result is in the 1.7e9 range.


## The census — Amendment 6, 2025-01-09, ten airports

4,886 labelled matched departures, 705,380 in-box ADS-B samples, callsign join 85.6%.

| apt | dep | join | R=50 cov/RMSE | R=100 cov/RMSE | R=200 cov/RMSE | verdict |
|---|---|---|---|---|---|---|
| EDDF | 505 | 100% | 7% / 603 | 10% / 531 | 11% / 494 | fail |
| EDDM | 350 | 100% | 65% / 278 | 73% / 270 | 76% / 257 | fail (RMSE) |
| EGLL | 630 | 100% | 19% / 243 | 21% / 239 | 23% / 242 | fail (coverage) |
| **EHAM** | 592 | 99% | 59% / 183 | **67% / 127** | 77% / 150 | **PASS, beats model 173** |
| LEBL | 372 | 99% | 25% / 792 | 30% / 749 | 33% / 687 | fail |
| LEMD | 518 | 98% | 0% | 0% | 0% | fail |
| LFPG | 589 | 99% | 0% | 0% | 0% | fail |
| LIRF | 358 | 99% | 0% | 0% | 0% | fail |
| **LSZH** | 294 | 100% | 45% / 289 | 51% / 165 | **55% / 157** | **PASS, beats model 188** |
| LTFM | 678 | 1% | 0% | 0% | 0% | fail (no join) |

**2 of 10 clear coverage ≥ 50% and bias-corrected RMSE ≤ 250 s. Threshold was 6. NO-GO.**

### Why it fails, and why that is not a code defect

Ground-level ADS-B reception is a property of the local community receiver network, not of the
aircraft. Diagnosed on the same day:

| apt | median samples/dep | median \|t − off-block\| | within 300 s of off-block | **on-ground share** |
|---|---|---|---|---|
| EDDM | 163 | 0.8 min | 89% | 51% |
| EHAM | 161 | 2.1 min | 81% | 50% |
| LSZH | 145 | 4.8 min | 53% | 31% |
| LEBL | 140 | 6.3 min | 41% | 43% |
| EDDF | 97 | 7.9 min | 23% | 16% |
| EGLL | 73 | 14.0 min | 22% | 10% |
| LEMD | 42 | 15.8 min | 0% | **0%** |
| LFPG | 42 | 15.4 min | 0% | **0%** |
| LIRF | 40 | 16.6 min | 0% | **0%** |
| LTFM | 2 | 19.1 min | 0% | **0%** |

**Madrid, Paris, Rome and Istanbul contain no on-ground samples at all** — the aircraft first
appear once airborne, a median 15–17 minutes after off-block. There is nothing to gate. Istanbul
additionally joins at 1%. This is a sensor-coverage limitation and no amount of gating, radius
tuning or better centroids can create samples that were never received.

### What it is worth, honestly

Only EHAM and LSZH give a usable channel. Inverse-variance blending the sensor with the model on
covered rows, at fold-A per-airport weights:

| apt | n | model | sensor | coverage | blended | SSE saved |
|---|---|---|---|---|---|---|
| EHAM | 40,141 | 173.2 | 127 | 67% | 130.1 | 524,683,095 |
| LSZH | 21,811 | 187.7 | 157 | 55% | 154.4 | 248,663,168 |

**GLOBAL MSE REMOVED: 2,243 — 8.8% of the 25,629 needed.** Against that: a ~190 GiB, ~20-hour
2026 ingest on a volume that is 93% full, plus an unresolved licence question with the organisers.
**The cost-benefit is not close. Per Amendment 6 §5 the lane is closed.**

The projection previously recorded here (~8,000 MSE if the other seven airports behaved like the
first three) is **withdrawn**: the three pilot airports were EHAM, EDDM and EGLL, two of which are
among the best-covered in the network. That was a biased sample and the extrapolation from it was
wrong.
