# PREREG — PRC Data Challenge 2026, taxi-out time · frozen 2026-09-08

**Frozen before any data was read and before any score was computed.** Amendments are appended
below with a reason and a timestamp. Nothing above an amendment line is ever edited. Team
`merry-quicksand`. Challenge window 2026-09-01 → 2026-10-11 23:59:59 CET.

Design inherited from the 2026-09-01 assessment made before access was granted. That assessment
is the reason this file exists before the data does.

## What is being predicted

Taxi-out time for departures at eleven major European airports: the interval from actual
off-block to actual take-off. Scored by RMSE on January and July 2026; trained on 2025.

## Why the label comes first

Taxi-out is a difference of two recorded timestamps, so every defect in either timestamp lands
directly in the target. RMSE is dominated by its tail. A small number of very long values, from
towing, returns to stand, de-icing holds, or clock errors, will move the score more than any
feature. Two consequences are pre-registered here:

1. **The censoring policy is fixed before any cross-validation number is seen.** Otherwise the
   policy gets tuned to the score, which is fitting the evaluation rather than the problem.
2. **Every row removed is counted and reported.** A censoring policy that quietly removes a
   large share of the data is a different model of the world, not a cleaning step.

### Censoring rules, pre-registered in shape

The exact column names are unknown until the data lands; an amendment will bind these rules to
real columns, and that amendment is the only place binding may happen.

| # | rule | disposition |
|---|---|---|
| C1 | taxi-out is negative or zero | DROP. Physically impossible; indicates a timestamp pair error. |
| C2 | either timestamp is missing | DROP. Cannot form the label. |
| C3 | taxi-out exceeds an absolute ceiling, provisionally 180 minutes | FLAG and report both with and without. Not silently dropped: long taxi-outs are real at congested airports, and the ceiling is a judgement that must be visible. |
| C4 | the movement is marked as a return to stand, towing, or a cancelled departure, where such a marker exists | DROP, and report how many such markers exist at all. |
| C5 | duplicate movement keys | DROP all but one, chosen by an explicit deterministic rule, and report the count. |

**Reporting requirement:** the label section of every result carries the kept-row count, the
dropped count per rule, and the retained distribution's five-number summary plus the 99th and
99.9th percentiles. A result without those numbers is incomplete.

## The free answers — the denominator for everything

No model result is reported in isolation. Every result is reported as reduction in RMSE against
the best of this ladder, computed on the same rows with the same folds:

1. global mean taxi-out
2. per-airport mean
3. per-airport, per-hour-of-day
4. per-airport, per-hour-of-day, per-month
5. per-airport, per-runway (where inferable), per-stand-zone (where inferable)

Rungs 3 to 5 are computed strictly as-of, from training months only, with a documented fallback
to the next coarser rung when a cell is unseen or thin.

**Pre-registered abort:** if the best free answer is already at or near the leaderboard's
published range, the honest finding is that there is little room, and that is reported rather
than papered over with a model.

## Folds

Flights in one departure bank share queue state, runway configuration and weather, so rows
within an airport-day are not independent.

* Group by **airport-day**.
* Hold out **whole months**.
* At least one **winter** and one **summer** month held out, because the evaluation set is
  January and July and de-icing makes those opposite regimes.
* Report the **spread across held-out months**, never only the mean. A model that is good in
  July and bad in January is not a good model for this evaluation.

## The as-of-T rule

For a departure with off-block time T, every feature is computed only from events **completed at
or before T**. The dangerous construction is named in advance: **departure queue depth**, because
the natural way to write it needs other flights' take-off times, which are in the future
relative to T for exactly the aircraft still in the queue.

Enforcement is by construction and by test, not by inspection:

* every rolling or aggregate feature filters to completions with timestamp ≤ T;
* every such feature has a unit test that plants a future value which must turn the test red;
* the feature module exposes one function that applies the filter, so the guard lives at the
  boundary that consumes the timestamps rather than at each call site.

That last point is not stylistic. A guard placed at call sites instead of at the consuming
boundary is the bug class that crashed a stage-5 run in a sibling project on 2026-09-03, after
an hour and fifty-eight minutes of completed work, because one of three call sites lacked it.

## Leak adjudication, and why one specific tool is not trusted here

An information-cliff leak detector, rule E, exists in a sibling project. **It is not the gate
here, and this is pre-registered so that a clean verdict from it cannot later be read as
safety.** Measured on 2026-08-31: it silently passes five of six leaked rolling windows, and an
empty finding list means "nothing was adjudicated" as often as it means "nothing leaked".
Rolling windows are precisely this problem's feature family, so the tool is weakest exactly
where it would be needed.

What is used instead:

* **Construction and unit tests**, as above.
* **A planted-leak harness.** A known future-reaching feature is injected. Cross-validation
  must visibly inflate while the leaderboard does not move.
  **If cross-validation does not inflate, the harness cannot see leaks, and every number
  downstream of it is void.** That is an abort condition, not a caveat.

This inverts the usual dependency: the competition tests the leak gate, rather than the leak
gate certifying the competition. That inversion is the intended research contribution and is
publishable independently of placement.

## Model ladder

Start at the bottom and climb only while the next rung beats the one below on held-out months
by more than the spread across those months. The tier is a ceiling, not a target.

1. best free answer
2. linear model on a small feature set
3. `HistGradientBoostingRegressor`, squared error (LightGBM is not installed and is not required)

## What this pre-registration does not claim

Nothing about placement. Nothing about the other teams. Nothing transferred from the GNSS
interference work beyond infrastructure and method: that project's features are integrity-field
based and have no relevance here, and its metrics are ranking metrics, which are meaningless
for RMSE.

## Honest prior, recorded before seeing anything

Most of the achievable reduction will come from the label policy and from per-airport,
per-hour-of-day structure, not from a clever model. I expect the gradient-boosted rung to beat
the best free answer by a modest margin, and I expect January and July to disagree. I expect at
least one feature to fail the planted-future test on first write.

---

## Amendment log

*(empty at freeze)*

### Amendment 1 · 2026-09-08 · binding the design to the real schema

**Written after reading the schema and label distribution of four downloaded files, and before
any cross-validation number, any model, and any submission exists.** The four files were
`ranking.parquet` and training months 2025-01, 2025-03, 2025-04; the remaining ten were still
downloading. Nothing above this line has been edited.

**1. Ten airports, not eleven.** Measured, not assumed: `EDDF, EDDM, EGLL, EHAM, LEBL, LEMD,
LFPG, LIRF, LSZH, LTFM`. Identical set in training and ranking. The earlier "eleven" came from
the 2026-09-01 assessment written before access and is wrong.

**2. The evaluation months are exactly 2026-01 and 2026-07**, confirmed by the months present in
`ranking.parquet`. Winter and summer, as the fold design assumed.

**3. The label is SUPPLIED, not computed, and it is internally exact.** `TAXITIME_SEC_mvt`
(int32, zero nulls in training) equals `MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt` on
**100.0000%** of the 153,706 departures in 2025-01, maximum absolute difference zero seconds.
This retires the concern that drove the original label section: there is no timestamp-pair
disagreement to adjudicate. **The censoring question does not disappear, it narrows** to which
supplied values are physically meaningful.

**4. The prediction target, precisely.** `ranking.parquet` holds 689,534 rows, of which the
**344,841 `PHASE_mvt == 'DEP'` rows have `TAXITIME_SEC_mvt` null** and are what must be
predicted. The 344,693 `ARR` rows have their taxi times **populated**. Arrivals are therefore
context, not targets, and they are a legitimate feature source for arrival pressure — subject
without exception to the as-of-T rule, since an arrival that lands after a departure's off-block
time is future information.

**5. Rows are movements, both phases.** Every file carries DEP and ARR at a roughly 1:1 ratio.
For a DEP row the airport is `ADEP_mvt`; for an ARR row it is `ADES_mvt`. The 900-plus distinct
`ADEP_mvt` values in the raw file are the origins of arriving flights and must not be mistaken
for the airport set.

**6. Censoring rules, rebound to real columns.** C2 (missing timestamp) and C5 (duplicate keys,
`MVT_ID_mvt`) stand as written. C1 and C3 are now measurable and are restated:

| rule | measured on 2025-01 departures | disposition |
|---|---|---|
| C1 taxi-out ≤ 0 | 48 rows, minimum −12 seconds | DROP. Negative taxi time is not physical. |
| C3 long tail | p99 = 2,589 s (43 min), p99.9 = 4,267 s (71 min), **max = 87,177 s (24.2 hours)**, 10 rows above 3 hours | FLAG, report with and without, ceiling unchanged at 180 min |

**C3 is now the load-bearing rule, and the reason is arithmetic.** A single retained 87,177-second
value, against a median prediction of 906 seconds, contributes on its own roughly 220 seconds to
the RMSE of a 153,706-row month. Ten such rows can outweigh every feature in the model. This is
pre-registered as the expected dominant effect, so that it cannot later be presented as a
discovery.

**7. `RUNWAY_mvt` and `STAND_mvt` are 0.0% null.** The deepest rung of the free-answer ladder,
airport by runway by stand, is therefore computable rather than aspirational, and no
stand-zone inference from H3 geometry is required. That removes a planned component.

**8. Airport alone is already a strong baseline.** Median taxi-out by airport ranges from 11.8
minutes at LSZH to 21.0 minutes at EGLL. Recorded now so that a model beating the global mean
cannot later be reported as skill.

**Unchanged by this amendment:** the fold design, the as-of-T rule, the refusal to treat rule E
as the leak gate, the planted-leak abort condition, and the model ladder.

**Still unknown at the time of writing:** the contents of `submitting.parquet` (1.6 MiB, not yet
downloaded) and whether it is the submission template or a scored subset. A second amendment
will bind it.

### Amendment 2 · 2026-09-08 · the submission contract, measured

**Written after fetching all fourteen objects over the S3 API with MD5-versus-ETag verification,
and still before any cross-validation number, any model and any submission.** Amendment 1 left
`submitting.parquet` unread because the console would not deliver it; it is now on disk and
verified.

**1. The submission format.** `submitting.parquet` is a template, not data: **344,841 rows and
exactly two columns**, `MVT_ID_mvt` (float64, integral, unique, no duplicates) and
`TAXITIME_SEC_mvt`, entirely null. A submission is that file with the second column filled in,
saved as `merry-quicksand_vN.parquet` and put in `prc-2026-merry-quicksand`.

**2. The key is sound, and this was checked rather than assumed.** The 344,841 identifiers in
the template are an **exact set match** with the 344,841 null-taxi rows in `ranking.parquet`:
zero on either side only. Every one of those rows is `PHASE_mvt == 'DEP'`. `MVT_ID_mvt` is
unique across the whole 689,534-row ranking file, so it is a safe join key with no
disambiguation rule required.

**3. The evaluation set, by month and airport.** Two months, ten airports, twenty cells:

| month | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-01 | 15,667 | 10,953 | 19,283 | 16,232 | 12,201 | 16,802 | 18,129 | 11,061 | 9,995 | 22,396 |
| 2026-07 | 20,650 | 15,013 | 20,557 | 21,950 | 17,879 | 20,152 | 21,743 | 15,838 | 13,155 | 25,185 |

July is larger than January in every airport, between 6% and 47% more departures. **The score is
therefore weighted toward summer**, and a model tuned on the January half of the fold design
would be optimising the smaller share. Recorded before any fold is run so it cannot be
rediscovered as a finding.

**4. Consequences for the fold design, which is otherwise unchanged.** Reporting stays per
held-out month, but the headline comparison against the free answer must additionally be
reported **weighted by the evaluation cell counts above**, since that is what the leaderboard
actually computes. An unweighted mean over months would flatter a January-strong model.

**5. Provenance now recorded per file.** Every object was verified by MD5 against the bucket's
ETag at fetch time and the result is in `data/raw/FETCH_MANIFEST.json`. This is the check to
re-run before every submission, because the two evaluation-side objects were regenerated on
2026-09-04, three days after the challenge opened, and can be regenerated again.

**Nothing else in this pre-registration changes.** The label policy, the free-answer ladder, the
as-of-T rule, the refusal to trust rule E, the planted-leak abort condition and the model ladder
all stand as written.

---

## RESULT 1 · 2026-09-08 · the censoring policy applied to all twelve training months

**This confirms a prediction made in the frozen text above, it does not discover it.** The
pre-registration stated that RMSE would be dominated by the tail and that C3 would be the
load-bearing rule. It is, and the magnitude is now measured.

Policy applied by `prc/labels.py` (8 tests, 5 mutations rehearsed) to 4,167,797 movement rows.

| outcome | rows | share |
|---|---|---|
| input | 4,167,797 | |
| dropped: not a departure | 2,082,750 | 49.972% |
| dropped: C2 missing label or timestamp | **0** | 0.000% |
| dropped: C1 taxi-out ≤ 0 | 388 | 0.009% |
| dropped: C5 duplicate movement id | **0** | 0.000% |
| **kept** | **2,084,659** | |
| flagged: C3 above three hours | 202 | 0.0097% of kept |

The data is cleaner than the policy anticipated: no missing labels, no duplicate identifiers.
C1 and C3 are the only rules that fire at all.

**Distribution of the kept label, seconds:** min 1, p1 293, median 912, mean 991, p99 2,339,
p99.9 4,501, **max 131,167 (36.4 hours)**.

**The finding.** RMSE of predicting the global mean:

| rows scored | RMSE |
|---|---|
| all 2,084,659 | **546.2 s** |
| excluding the 202 C3-flagged rows | **429.8 s** |

**202 rows in 2.08 million — one in ten thousand — carry roughly 38% of the total squared
error.** That is the arithmetic the pre-registration warned about, at a larger magnitude than
was guessed.

**The consequence, recorded before any model exists.** Rows cannot be dropped from the
evaluation set: the leaderboard scores all 344,841 departures, tail included, and a 36-hour
taxi-out is not predictable from anything in this data. A substantial and *shared* part of every
team's score is therefore irreducible noise. Two things follow, and neither is a post-hoc
rationalisation because both are written here before the first fit:

1. **Leaderboard gaps will be compressed** by that shared noise floor, so a small RMSE
   difference between teams may not be a real difference in skill. Any claim we make about
   placement must account for it.
2. **The competition is over the other ~62%.** Effort belongs in the body of the distribution,
   and the tail is a robustness question — whether to train on it, weight it down, or model it
   separately — not a prediction question.

**C3 remains flag-not-drop.** The excluded-tail number above is reported as a diagnostic of how
much noise the tail carries, never as our score.

---

## RESULT 2 · 2026-09-08 · the free-answer ladder, and an amendment to its shape

**Amendment to the ladder as frozen.** The pre-registered ladder listed
`airport × hour-of-day` as rung 3 and `airport × runway × stand` as rung 5. Implementing it
showed those are not a nesting: falling back from `airport × runway` to `airport × hour` moves a
thin cell sideways into a different partition rather than into a coarser description of itself.
The ladder is now a **strict chain**, each rung adding exactly one key, with hour at the fine end
because time of day refines a stand-and-runway pairing rather than generalising it. Caught by a
test, not by a score. The rungs are otherwise unchanged.

**Protocol.** Fitted on the ten months of 2025 that are not January or July (1,740,325 rows);
evaluated on held-out January and July 2025 (344,334 rows), mirroring the real evaluation's
winter-and-summer split. Cells with fewer than 30 observations fall back one rung.

| rung | RMSE, seconds |
|---|---|
| global mean | 686.6 |
| airport | 660.0 |
| airport × runway | 653.1 |
| airport × runway × stand | 629.1 |
| **airport × runway × stand × hour** | **627.7** |

**The best free answer is 627.7 seconds, 8.6% better than the global mean. That is the
denominator. No model result is reported except as reduction against it.**

Two observations, both recorded before any model exists.

**July is the harder month and also the larger one.** January 553.9 seconds on 153,656 rows;
July 681.4 seconds on 190,678 rows. The naive expectation was the reverse, on the assumption
that de-icing would make winter volatile. It does not hold: summer peak traffic produces more
congestion variance than winter weather does. Combined with Amendment 2's finding that July has
more departures at every airport, **the score is dominated by the harder season twice over**,
and any tuning that favours January is optimising the wrong half.

**Most of the geometry is already spent.** Stand and runway together buy 4.4% off the global
mean, and hour adds only a further 0.2%. The remaining error is not about where an aircraft
starts and which runway it uses. It is about what else was happening on the surface at the time,
which is precisely the congestion term the queuing decomposition names and which no rung of this
ladder can see.

---

### Amendment 3 · 2026-09-08 · retracting the leaderboard clause, re-anchoring to take-off

**Written before any model is fitted and before any submission exists.** This amendment retracts
one pre-registered procedure, corrects the anchor that half of this document is written against,
and restates the denominator. Nothing above this line has been edited; the earlier text stands as
written, including the parts this amendment overrules.

**1. The leaderboard-observation clause is retracted, unconditionally.**

The planted-leak harness above requires that "cross-validation must visibly inflate **while the
leaderboard does not move**." That second half is withdrawn. It instructs us to spend submissions
probing how the ranking process responds to a deliberately corrupted model, which is exactly the
behaviour the organisers state they monitor for, and 2026 has **no phase two** — the ranking
score is the final score, so there is no separate set on which such probing could be harmless.

The harness is now **local only**: a planted future-reaching feature must inflate held-out error
across whole-month folds. If it does not inflate, the harness is blind and every number
downstream of it is void — that abort condition is unchanged and is not weakened by this
amendment. Its trigger is now measured entirely on data we hold.

Restated as a standing rule for the remainder of the competition: **no threshold, hyper-parameter,
feature, model or fold design may be chosen by comparing leaderboard scores.** Submissions may be
used only to confirm that a locally-computed number and the remote number agree, and any
disagreement is investigated locally.

**2. The as-of-T rule is re-anchored from off-block to take-off.**

The rule above reads "for a departure with off-block time T". Measured on the 344,841 scored rows
of `ranking.parquet`: `BLOCK_TIME_UTC_mvt` is **100.000% null**, necessarily, since the target is
take-off minus off-block. `MVT_TIME_UTC_mvt` is **0.000% null**. The pre-registered anchor does
not exist for a single row we must predict.

**T is redefined as take-off time, `MVT_TIME_UTC_mvt`.** Every feature is computed only from
events completed at or before take-off. Off-block becomes a *predicted* quantity, never an input.
The enforcement clauses are unchanged and now bind against the new anchor: the boundary-level
filter, the planted-future test per aggregate feature, and the single consuming function.

This is a strict weakening of what the model may see in one respect and a strict widening in
another, and both are deliberate: the taxi-out interval itself is now inside the observable
window, so any feature that counts events in `[T − w, T]` may include the subject flight's own
taxi. **Every windowed feature must therefore exclude the subject row itself**, and that
exclusion gets its own test.

Also recorded, since it follows from the same measurement: for the 1.534% of scored rows with no
Network Manager record, **every `*_flt` column is null together** — `AOBT_3_flt`, `EOBT_1_flt`,
`LOBT_flt`, `IOBT_flt`, `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `WK_TBL_CAT_flt` and
`ARVT_3_flt` all at exactly 1.534%. Any model for that stratum is restricted to the `_mvt`
columns plus the schedule, and the operator survives only through the callsign prefix of
`FLIGHT_mvt`.

**3. The denominator is 629.1, not 627.7.**

RESULT 2 reports the deepest rung, `airport × runway × stand × hour`, at 627.7 seconds. That rung
keys on `_hour`, which `prc/baselines.py` derives from `BLOCK_TIME_UTC_mvt`. On the evaluation
file that column is 100% null, so the rung never fires and `Ladder.predict` returns exactly the
`airport × runway × stand` prediction. **The free answer actually available on the submission set
is 629.1 seconds.** RESULT 2's table is not withdrawn — it is a correct measurement of a
quantity that is not deliverable.

Two consequences. The rung is re-anchored to take-off hour rather than deleted, since take-off
hour is present on every scored row. And the reported best rung was selected on the held-out set
itself, so **629.1 is mildly optimistic** and is treated as an upper bound on the free answer, not
a point estimate.

**4. A pre-registered stationarity concern, with its test named before it is run.**

The stratum that decides the score is not stationary between the training year and the evaluation
year, and this is recorded now so that a later failure cannot be read as a surprise.

| | 2025 Jan | 2026 Jan | 2025 Jul | 2026 Jul |
|---|---|---|---|---|
| unmatched rate | 0.92% | **1.61%** | 2.05% | **1.48%** |

The pooled rate barely moves — 1.545% against 1.534% — and that stability is an artefact of two
opposite shifts cancelling. `test_stratum_prevalence_is_stationary` is written to fire **red**
against this, per month and not pooled.

A second, heavier non-stationarity is recorded at the same time, measured on `MVT_TIME − SCHED_TIME`,
which is computable on the evaluation file and is the only 2026-side signal available without
spending a submission. On unmatched rows:

| | n | p99 | p99.9 | max | share above 3h |
|---|---|---|---|---|---|
| 2025 Jan+Jul | 5,321 | 20,973 | 72,951 | 98,691 | 7.20% |
| **2026 evaluation** | 5,290 | **28,377** | **78,917** | **111,654** | **12.63%** |

The 2026 unmatched stratum is **1.75× heavier in the tail** that dominates squared error. Every
stratum estimate measured on 2025 is therefore treated as **optimistic for 2026**, and no
stratum-level pass/fail threshold may be set from a 2025 measurement alone.

**5. Threshold discipline, restated.** No pass/fail threshold may be narrower than the 95%
bootstrap interval of the quantity it tests. Measured on the unmatched stratum, held-out Jan+Jul
2025, n = 5,321: RMSE 2,276 with a 95% interval of **[1,576, 2,981]**, a width of ±31%. Model
selection on that stratum by held-out RMSE alone is therefore **not possible**, and any claimed
improvement inside that band is not a claim.

---

### Amendment 4 · 2026-09-08 · a bounded, pre-registered ADS-B experiment

**Written before a single byte of ADS-B has been downloaded and before any ADS-B-derived number
exists.** Nothing above this line has been edited. The day selection and the kill conditions below
are frozen here precisely so that neither can be chosen after seeing a result.

**1. Why this is different from everything else tried.**

Every lever tested on 2026-09-08 — weather, queue and surface counts, kNN neighbourhoods,
mixture-of-experts, departure-sequence topology — attempts to *infer* the withheld off-block time
from the supplied columns. Two purpose-built diagnostics (neighbourhood purity, topology) both
found the model **unbiased in every stratum**, with bias² ≤ 3.3% of MSE everywhere and 0.013%
overall. The residual is conditional dispersion, not learner error. Ground ADS-B is categorically
different: it can **observe the physical event** — the aircraft beginning to move — rather than
infer it. That is the only remaining hypothesis that supplies new information about the hidden
quantity itself.

**2. The hypothesis, stated so it can fail.**

> For departures with usable ground ADS-B coverage, a movement-derived estimate of physical
> off-block reduces held-out RMSE on the challenge target beyond the current model.

**TRUE shape:** on scoring days never used for calibration, adding the ADS-B estimate reduces
taxi-out RMSE on covered rows, and the coverage-adjusted projection to the full evaluation set
exceeds 5 seconds. **FALSE shape:** coverage-adjusted projection below 2 seconds, or the estimate
degrades RMSE, or ground coverage on matched challenge flights is too sparse to matter.

**3. Days, fixed now, chosen by seeded RNG (`numpy default_rng(20260908)`), not by inspection.**

| role | days |
|---|---|
| **calibration** | 2025-01-09, 2025-01-21, 2025-07-07, 2025-07-12 |
| **scoring** | 2025-01-26, 2025-01-29, 2025-07-13, 2025-07-20 |

2025 is used because it is the only period with **both** ADS-B archives and challenge labels;
without labels the movement-to-`BLOCK_TIME` relationship cannot be calibrated at all. Airports are
restricted initially to **EHAM, EGLL, EDDM**, the three where a local probe on 2026-07-15 measured
real ground visibility (456, 321 and 252 ground reports in a 220-snapshot slice). LFPG, LIRF, LSZH
and LTFM showed zero and are out of scope until coverage is demonstrated.

**4. Off-block definitions — all evaluated, none selected on the scoring days.**

First sustained groundspeed above a low threshold; first sustained displacement from the
stationary stand centroid; end of the last stationary period before take-off; first exit from a
stand-radius; and variants requiring persistence across consecutive polls rather than a single
noisy report. **The definition is chosen on calibration days only.** Choosing it on the scoring
days would be selecting a threshold on the evaluation set, which this pre-registration forbids
elsewhere and forbids here.

**5. Matching.** Candidate traces are joined by callsign (`FLIGHT_mvt` / `CALLSIGN_flt`) plus
departure airport plus a window around the known exact take-off time. Match rate is reported as a
first-class number, not assumed.

**6. The primary metric is the challenge metric.** Not correlation, not AUC, not tail detection
rate: `RMSE(model + ADS-B) − RMSE(model)` on the real target, on scoring days, with a paired
bootstrap. Reported alongside: coverage fraction, the distribution of
`first_motion − BLOCK_TIME` by airport, and performance specifically on the large-negative-delta
rows that carry the error.

**7. Kill conditions, fixed before the experiment.**

| coverage-adjusted projection to the full evaluation set | action |
|---|---|
| **< 2 s** | **STOP.** Do not download further data. Record NOT WORKING. |
| 2–5 s | interesting, insufficient — do not authorise a large ingest |
| > 5 s | scale to more days and more airports |
| > 10 s | authorise the full ingest |

**8. Cost ceiling for this experiment.** Eight daily archives (~2 GiB each in January, ~3–3.7 GiB
in July) ≈ **15–25 GiB of transfer**. Each day is streamed, reduced to the three airport areas,
written as a small derived Parquet, and **the source archive is deleted before the next day is
fetched**. Only ~4 GiB of disk is in use at any instant, against 32 GiB free. No persistent raw
storage. The full 2025 archive is ~2,115 GiB and is explicitly **not** authorised by this
amendment.

**9. Provenance and licence, recorded before use.** The archives are published by **ADSB.lol**
under **ODbL 1.0**. ADSB.lol is **not** the competition's data host — the challenge is supported by
OpenSky; an earlier claim otherwise in this project's notes was wrong and is corrected here. The
eligibility rules permit additional openly accessible datasets provided they are documented, so
every release tag consumed, the extraction procedure, and the ODbL attribution are recorded in the
repository if any ADS-B-derived feature reaches a submission.

**10. This amendment does not authorise a submission, a model change, or the large ingest.** It
authorises one bounded measurement whose result is reported against the table in §7 whatever it
says.

---

# AMENDMENT 5 — as-of-T scope, and the stand-occupancy experiment
*Appended 2026-09-08 (night). Nothing above this line has been edited.*

**1. What this amends.** Amendment 3 re-anchored the as-of-T rule from off-block to **take-off**:
a feature may read only information available at or before the flight's own take-off. That rule is
too narrow for one specific class of information, and this amendment scopes it.

**2. The relaxation, precisely.** Timestamps belonging to **other movements** and supplied in
`ranking.parquet` are admissible even when they fall after the subject flight's take-off. The
subject flight's own `BLOCK_TIME_UTC_mvt` remains **off-limits** and is the target; nothing derived
from it may enter a feature.

**3. Why this is legitimate and not leaderboard exploitation.** The organisers blank
`BLOCK_TIME_UTC_mvt` on departure rows only. Arrival rows retain it, populated on 100% of the
344,693 ARR rows, together with `STAND_mvt`. Withholding the departure clock while supplying the
arrival clock is a deliberate construction choice, and using a supplied column is categorically
different from the prohibited activity, which is learning from the ranking process. No threshold,
feature or hyper-parameter is chosen by comparing leaderboard scores; that prohibition (Amendment
3) stands unchanged.

**4. What it costs.** The relaxation makes any resulting feature unusable in a real-time
deployment, because the next aircraft has not yet arrived when the subject flight departs. The
competition scores a static file, so this cost is not paid here — but it is recorded so no future
reader mistakes the method for an operational one.

**5. The pre-registered experiment.** Hypothesis: *for a departure whose airport off-block is far
earlier than NM's `AOBT_3`, another aircraft in-blocking at the same stand is direct evidence of
the mechanism, and supplying it to the matched-row model removes global MSE.*

Definitions: `T_c` = in-block time of the last arrival at the same (airport, stand) at or before an
anchor; witness = `T_c > BLOCK_dep` (truth, diagnostic only); slack `s = T_c − BLOCK_dep`; the
identity `y = (MVT − T_c) + s` holds exactly.

Design: 70-feature E1-equivalent baseline versus the identical model plus the stand block — same
rows, same whole-month folds, same seed, same hyper-parameters. Paired bootstrap. Decomposed by
global / matched / witness / non-witness / `delta < −600` / each airport separately.

**TRUE shape:** paired global MSE removed **> 1,000** with a bootstrap interval excluding zero,
replicated on a second fold.
**FALSE shape:** interval spans zero, or the gain is confined to the fold-A months.

**Kill thresholds, fixed in advance:** > 4,000 MSE removed = major; 1,000–4,000 = useful and
carried into the pipeline; **< 1,000 = closed**, and the NN/slack elaboration (steps 4–6 of the
stand plan) is not attempted. The NN elaboration additionally requires that neighbourhood
conditioning reduce slack dispersion from ~453 s to **≤ 300 s**; at ≥ 400 s it is not attempted.

**6. Already measured under this amendment, reported whatever it says.** The mechanism gate has
passed and the incremental-value gate is **open** — see `reports/STAND_OCCUPANCY.md`.
