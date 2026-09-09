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

---

# AMENDMENT 6 — the ten-airport ADS-B gate census
*Appended 2026-09-08 (night). Nothing above this line has been edited.*

**1. Why this exists.** Amendment 4 closed ADS-B as NOT WORKING. That closure is now known to
rest on two defects (`reports/ADSB_GATE.md`): OPDI's 0.7% is OpenStreetMap polygon coverage rather
than sensor coverage — the archive joins to 97–99% of departures — and the raw-trace comparison was
keyed on `FLIGHT_mvt`, which at EHAM is the IATA commercial number and matches 2.2% of ADS-B
callsigns against 99.5% for `CALLSIGN_flt`. **Amendment 4's ADS-B verdict is retracted.** A
stand-geometry gate with centroids learned from the data reaches RMSE 127 at EHAM against the
model's 173, on 65.8% of rows. That is one day and three airports.

**2. The hypothesis.** *A stand-geometry gate on ground ADS-B yields a pushback estimate whose
bias-corrected RMSE is competitive with the per-airport model, at usable coverage, across all ten
scored airports and in both scored seasons.*

**3. Design, fixed before any 2025 archive is opened.** Two daily archives — one January, one July,
both 2025 so labels exist — extracted to the ten airport areas. Stand centroids learned per
(airport, stand) from aircraft positions at their own recorded off-block, **leave-one-flight-out**.
Estimate = last ADS-B sample inside radius R of the flight's own stand. R ∈ {50, 100, 200} m,
reported for each; the operating R is chosen per airport on the JANUARY day only and applied
unchanged to July, so the July number is out-of-sample in season.

**4. TRUE shape.** Median per-airport gated RMSE ≤ 250 s at coverage ≥ 0.5, holding on **at least
6 of the 10 airports**, on both days.
**FALSE shape.** Fewer than 6 airports clear it, or July degrades by more than 50% against January
at the January-chosen radius.

**5. Kill threshold, fixed in advance.** **GO** at ≥ 6 of 10 airports: the 2026 serving ingest
(~190 GiB, ~20 h, and its own amendment) becomes justified. **NO-GO** below 6: the ADS-B lane is
closed for a second and final time, the projection in `reports/ADSB_GATE.md` is withdrawn, and the
remaining days go to consolidation, the LIRF classification term and LightGBM at a realistic ~285.

**6. What this amendment does NOT authorise.** It does not authorise the 2026 ingest, any
submission, or any model change. It authorises two daily archives — inside Amendment 4 §8's
15–25 GiB budget — streamed one at a time, each deleted before the next is fetched, ≤4 GiB of disk
at any instant, on a volume that is currently 93% full.

**7. Licence.** ADSB.lol publishes under ODbL 1.0 and Amendment 4 §9 already records the
attribution obligation. **OpenSky's own historical database is NOT used** — its Terms of Use
restrict it to non-profit research and forbid redistribution, which conflicts with the prize
condition to open-source added data under GPLv3. The eligibility question for any ADS-B-derived
feature reaching a submission remains open with the organisers and is not settled by this
amendment.

**8. Reported whatever it says.** The per-airport table goes into `reports/ADSB_GATE.md` with the
verdict against §5, including a NO-GO.

---

# AMENDMENT 7 — is the LEARNER the bottleneck, not the data?
*Appended 2026-09-08 (night). Nothing above this line has been edited.*

**1. Why this exists.** Every experiment in this project has used one learner:
`HistGradientBoostingRegressor(max_iter=400, lr=0.05, max_leaf_nodes=127, random_state=0)` —
untuned, single-seed, single-family. The published solutions of the two prior editions of this
competition did not. `PRC-Data-Challenge-2024/team_likable_jelly` used **LightGBM with 50,000
trees, averaged over several random seeds** (single model 1612 RMSE, ten-model average 1564 —
a 3.0% gain from averaging alone). `team_tiny_rainbow` ensembled LightGBM + XGBoost + CatBoost +
neural networks. `PRC-Data-Challenge-2025/zestful-mango` split into two gradient-boosting models
by trajectory data quality.

**We have spent this project testing whether the DATA is exhausted, using a baseline learner whose
capacity has never been established.** That is the untested assumption.

**2. Hypothesis.** *A properly-sized LightGBM, with native categorical handling for high-cardinality
columns, materially reduces matched-row RMSE against the 400-iteration HGB control on the identical
fold.*

**3. Design.** Fold A (Jan+Jul 2025 held out), identical cached rows, identical `delta` target,
identical scoring code, identical leakage guards, identical feature set. Three arms:

| arm | learner | features |
|---|---|---|
| control | HGB, 400 iters, lr 0.05, 127 leaves, seed 0 | the 68-feature encoded matrix |
| `lgb_enc` | LightGBM, up to 50,000 trees, lr 0.01, early stopping | the same 68-feature matrix |
| `lgb_native` | LightGBM, same config | raw categoricals natively, encodings dropped |

The third arm separates **boosting capacity** from **information recovered by not target-encoding
`STAND_mvt`** (1,862 levels; sklearn caps native categoricals at 255). LightGBM's own guidance is
that high-cardinality categoricals need `cat_smooth` / `min_data_per_group` regularisation, so both
are set rather than left at defaults.

**Early stopping uses two months carved out of the TRAINING fold, never the held-out fold.** Using
Jan+Jul for early stopping would leak the evaluation set into model selection.

**4. Recorded regardless of outcome.** Best iteration reached; matched RMSE; projected total with
the stratum held fixed at 1527.2; per-airport RMSE; `delta < -600` RMSE; **residual correlation
with the HGB control**; and global MSE removed against the ledger. The residual correlation matters
independently of the headline: a learner that is only slightly better but fails *differently* is
worth more in an ensemble than one that is better and fails the same way.

**5. Decision thresholds, fixed before the run.**
- **≥10% matched RMSE reduction** (230.4 → ≤207): pivot the project. Tuning, multi-seed averaging,
  CatBoost/XGBoost and ensembling become the main programme, and the stand A/B is **re-run on
  LightGBM** — an HGB-based A/B may have judged a feature useless only because HGB cannot represent
  what LightGBM handles natively.
- **5–10%**: major result, carried into the pipeline.
- **2–5%**: useful ensemble material, not the path to 248.
- **<2%**: learner capacity is NOT the missing mechanism; return to the stand/slack specialists
  with that question closed.

**6. Seeds.** Establish first that the learner wins on a single seed. Only then average seeds 0,1,2,
and expand to 5–10 only if the ensemble moves materially. The 2024 evidence is a **3%** gain from
1→10 models — it is not evidence for a large effect and must not be quoted as one.

**7. Machine tradeoff, named.** LightGBM runs with `n_jobs=4` of 18 cores rather than the standing
`OMP_NUM_THREADS=1`, because 50,000 trees single-threaded is hours rather than minutes. Swap is
2.7 GB of 4.1 GB with ~5.4 GB reclaimable; the run aborts if swap passes 3.5 GB. Three
`com.phantom.*` jobs are running and are not touched.

---

# AMENDMENT 8 — the first board score, and two lanes probed WITHOUT pre-registration

**2026-09-09.** Nothing above this line has been edited.

## 8.0 An honesty note about this amendment's status

Amendments 5-7 each pre-registered a threshold BEFORE measuring. **The two probes recorded in
§8.2 and §8.3 did not** — they were exploratory diagnostics run during a build wait, and they are
being written down after the fact. That is a deviation from this document's own discipline and is
named rather than hidden.

Mitigating, and the reason they are recorded at all rather than discarded: both returned
NOT WORKING / negative, and a post-hoc write-up of a negative result cannot have been
goal-post-moved in our favour. Neither is used to support any claim we make. **Neither may be
cited as a pre-registered closure**, and §8.3 in particular is explicitly left OPEN.

## 8.1 RESULT — the first real submission, and the fold's absolute bias

`merry-quicksand_v2.parquet`, the shipped pipeline with the consolidated stratum, scored
**301.7019** on 344,841 rows at 2026-09-09 02:10:51Z — **rank 26 of 80 teams**. (`v1`, at
689.6901, was the constant-predictor probe of 09-08; the version number was already taken and the
build had to be issued as v2.)

The same artifact's `--validate` fold total is 330.22. **The fold is +28.52 s (+18,021 MSE)
PESSIMISTIC as an absolute instrument at this model class.** Probable mechanism: `--validate`
holds out Jan+Jul 2025, denying the validation model precisely the seasonality the scored months
carry, while the submitted artifact fits all twelve months.

**Binding consequence for every later amendment:** `--validate` remains the correct instrument for
A/B comparisons and is NOT a predictor of board position. No fold total may be quoted as an
expected score. The offset is one measurement at one model class and must not be assumed constant.

## 8.2 PROBE (not pre-registered) — is there a better off-block clock in the unread columns?

Motivation: on matched rows the target reduces to the exact identity
`y = MVT_TIME - BLOCK_TIME`, so the entire matched problem is the disagreement between two
off-block clocks, and `build_submission.py` reads only 22 of the file's 30 columns.

**Measured:** the schema is 30 columns; the eight unread are `FLIGHT_ID_mvt`, `CALLSIGN_flt`,
`ADEP_flt`, `ADES_flt`, `ADES_FILED_flt`, `AIRCRAFT_TYPE_flt`, `ARVT_1_flt`, `ARVT_3_flt`. There is
no `AOBT_1` or `AOBT_2`. **Verdict: NOT WORKING — no second off-block clock exists.** This is a
structural read of the schema, not a threshold comparison, so the lack of pre-registration costs
nothing here.

## 8.3 PROBE (not pre-registered, and LEFT OPEN) — is the stratum a hidden fill-classification?

Motivation: `fit_unmatched` predicts a mixture `p*sp + (1-p)*nf` because it cannot tell which
unmatched rows had the scheduled push stamped into the off-block field.

**Measured on the fold's 5,321 unmatched rows:** shipped mixture 1867.9; ORACLE fill-flag paired
with a CONSTANT non-fill predictor 3080.5; oracle restricted to fill rows 24.5; constant on
non-fill rows 3156.5; fill share 4.8%.

**Verdict: the CHEAP version is refuted — a perfect classifier alone loses badly to the mixture,
so work that improves only the classifier is not worth doing. The LANE IS NOT CLOSED.** The oracle
arm conflates the classifier with the non-fill regressor by pairing a perfect flag with the weakest
possible regressor. The arm that actually bounds the lane — **oracle flag + a fitted non-fill
model** — has not been run. A first write-up of this said "lane closed"; a review the same night
judged that beyond the evidence, and it is withdrawn. That arm is cheap and should be
pre-registered properly before any further stratum work.

## 8.4 NAMED, NOT MEASURED — the arrival stream

`ranking.parquet` is 689,534 rows: the 344,841 scored departures **plus 344,693 arrivals**, and on
those arrival rows `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are fully populated (taxi-in, mean
537 s). Every scored departure's take-off time is also known. `derive()` currently builds no count
or congestion feature of any kind. This is not one of the four closed lanes, it is unmeasured, and
it requires a proper pre-registration before it is probed.

---

# AMENDMENT 9 — superseding 8.1: the fold/board gap is a two-row draw, not seasonality

**2026-09-09, same night as Amendment 8. Nothing above this line has been edited; 8.1 stands as
written and is superseded here.**

## 9.1 What 8.1 claimed, and why it is wrong

Amendment 8.1 recorded the fold at 330.22 against a board of 301.70 and attributed the gap to
`--validate` holding out Jan+Jul 2025, denying the validation model the seasonality of the scored
months. Two independent reviews challenged it and the measurement settles it.

**Corrections of fact:**
- The instrument prints **330.81**, not 330.22. The latter was a re-weighting of the fold's
  components by the 2026 scored-file weights, done in analysis and quoted as instrument output.
  The gap is 29.11 s / 18,412 MSE.
- **The gap is carried by two rows.** The fold's top two unmatched rows — LFPG, January,
  y = 84,240 s and 58,206 s, `sp` 1,740 and 2,043, predicted 1,090 and 1,216 — carry **29,511 MSE:
  54.7% of the fold's stratum SSE and 27.0% of its total.** The worst single row is 18.3% of the
  fold. 29,511 exceeds the 18,412 the seasonality story had to explain.
- Adjusting those rows to an expected Family-B load (13-17k MSE) puts the fold at 304.8-311.3
  against a board of 301.70: a residual of +3 to +10 s, not 29.
- `reports/FAMILY_B.md` §8 had **already** recorded that this fold is ~25 s unluckier than average
  on this family. It was read and not subtracted, and a mechanism was invented for the residue.

**Verdict on 8.1's causal claim: NOT WORKING.** Seasonality is not needed to explain the gap and no
evidence was offered for it beyond the gap itself.

## 9.2 The replacement rule, which binds harder than 8.1's

8.1 said "`--validate` is a biased absolute instrument, offset +28.52 s, do not assume the offset is
constant." That is too weak and invites correcting by the offset anyway.

**The fold TOTAL has no stable absolute value.** It is set by a two-row draw from a heavy tail, so
its absolute uncertainty is tens of seconds and it cannot be calibrated by any number of
submissions. Accordingly:

1. **Never carry a fold→board offset.** Not across model classes, not across runs, not at all.
2. **Never quote a fold TOTAL as a predicted board score.** The board is the only instrument for
   absolute position.
3. **`--validate` remains valid for A/B on MATCHED rows**, where n = 339,015 and no comparable
   concentration exists. Matched-side A/B conclusions in this document are unaffected.
4. **Any stratum or TOTAL comparison must report the monster rows separately** — as an ex-monster
   figure plus an explicit expected-load term — because a stratum RMSE computed over 5,321 rows
   whose SSE is 55% concentrated in two of them is not a measurement of a model.

## 9.3 Consequence for the target

Amendment 8's framing of the gap to the leader is unaffected in direction but its precision was
overstated. The board gap of 30,158 MSE stands, since both terms are board numbers. What does NOT
stand is any inference from fold totals about how far a change moves us on the board.

Note for anyone reading 8.3 and 8.4: those verdicts rest on stratum comparisons, and per rule 4
above they must be re-expressed ex-monster before they are cited. 8.3 was already left OPEN. 8.4 is
unmeasured and unaffected.

---

# AMENDMENT 10 — two screens, thresholds locked BEFORE measurement

**2026-09-09. Nothing above this line has been edited. Written and committed BEFORE either screen
is run** — 8.2 and 8.3 were measured before being registered, Amendment 9 exists partly because of
that slack, and this amendment does not repeat it.

Both screens are read-only diagnostics under 5 minutes. Neither ships anything. Their only purpose
is to decide whether a full experiment (~45 min, and we have ~40 left before the wall) is warranted.

## 10.1 SCREEN A — is `AOBT_3` synthetic on the matched tail?

**Motivation.** On matched rows the pipeline predicts `delta = BLOCK_TIME - AOBT_3`. The error is
extremely heavy-tailed: 1.47% of matched rows carry 50.4% of matched squared error. If the tail is
where the Network Manager's `AOBT_3` is not a measurement but a *fallback* — copied from a filed
estimate — then those rows carry a different data-generating process, and crucially the copy would
be **observable at serve time**, because `LOBT_flt`, `EOBT_1_flt` and `IOBT_flt` are populated on
scored rows while `BLOCK_TIME` is not.

**Hypothesis H-A.** Rows in the `delta` tail are enriched for an observable signature that
`AOBT_3` was derived from a filed time rather than measured.

Candidate signatures, all computable at serve time: `AOBT_3` equal to `EOBT_1` / `LOBT` / `IOBT`
exactly or within 60 s; `AOBT_3` landing exactly on a whole minute; `AOBT_3` on a 5-minute grid.

**TRUE shape.** At least one signature shows enrichment (rate in tail / rate in body) of **>= 3.0x
at >= 4 of the 10 airports**, AND the flagged rows account for **>= 20% of matched squared error**.
-> the lane earns a full experiment.

**FALSE shape.** No signature reaches **2.0x at any airport**, or flagged rows carry **< 5%** of
matched squared error. -> H-A NOT WORKING, lane closed, do not revisit without new information.

**Ambiguous** (between the two) -> report as inconclusive and do NOT spend a full experiment on it
this week; it goes behind consolidation in the queue.

Tail is defined before looking: `|delta - median(delta)| > 900 s`, body is the complement, both on
the fold's matched rows only, and enrichment is computed **per airport** (pooling has hidden the
truth five times in this project).

## 10.2 SCREEN B — the arm that actually bounds the stratum

**Motivation.** Amendment 8.3 refuted the cheap version of the fill-classification lane but did not
bound it, because its oracle arm paired a perfect classifier with a CONSTANT non-fill predictor.
The arm that separates the classifier from the regressor has never been run.

**Arms**, all fitted on the ten non-holdout months and scored on the fold's unmatched rows:
- **B0** shipped mixture `p_hat*sp + (1-p_hat)*nf_cells` — the incumbent, expected 1867.9.
- **B1** oracle fill-flag x `nf_cells` — classifier headroom given today's regressor.
- **B2** `p_hat` x `nf_fit`, where `nf_fit` is a small fitted model on non-fill rows — the
  **shippable** arm.
- **B3** oracle fill-flag x `nf_fit` — the bound.

**Per Amendment 9 rule 4, every arm is reported BOTH pooled and ex-monster**, because a stratum
RMSE over ~5,300 rows whose SSE is 55% concentrated in two of them is not a measurement of a model.
**The ex-monster figures are the decisional ones.** Monsters are defined before looking, as
unmatched fold rows with `y > 3h`.

**TRUE shape / thresholds, locked:**
- Regressor lane **ESTABLISHED** iff `B0 - B2 >= 20 s` on ex-monster stratum RMSE. -> wire `nf_fit`.
- Classifier lane **ALIVE** iff `B3 - B2 >= 100 s` ex-monster; **DEAD** iff `< 60 s`; between the
  two, inconclusive and deprioritised.
- If `B2 >= B0` ex-monster (the fitted regressor does not beat the cell estimator), the whole
  stratum-regressor lane is **NOT WORKING** and closes.

**What no result here licenses.** No board projection. Per Amendment 9, stratum and total fold
numbers do not translate to the board; these thresholds are stated in fold ex-monster RMSE and are
decisional only for whether to spend a full experiment.

---

# RESULT 3 · 2026-09-09 · both Amendment 10 screens, run and verdicted

Thresholds were locked in Amendment 10 and committed (ba1b132) BEFORE either screen ran. Neither
threshold was moved. Implemented by a delegated agent; both decisional numbers were then reproduced
independently before being written here.

## R3.1 SCREEN A — H-A NOT WORKING. Lane closed.

Tail = matched fold rows with `|delta - median(delta)| > 900 s`: 10,272 of 339,015 (3.03%),
carrying 62.5% of a model-free proxy for matched SSE (squared deviation of `delta` from its
per-airport median — a proxy, not model SSE).

| signature | rate in tail | rate in body | enrichment |
|---|---|---|---|
| `AOBT_3` == `EOBT_1` exact | 0.94% | 4.90% | **0.19x** |
| `AOBT_3` == `LOBT` exact | 0.72% | 4.13% | 0.17x |
| `AOBT_3` == `IOBT` exact | 0.71% | 4.14% | 0.17x |
| within 60 s of `EOBT_1` | 3.24% | 14.41% | 0.22x |
| within 60 s of `LOBT` / `IOBT` | 2.35% / 2.27% | 12.26% / 12.27% | 0.19x / 0.18x |
| whole minute / 5-min grid | 95.8% / 19.7% | 97.7% / 19.9% | 0.98x / 0.99x |

Per-airport, best signature: EHAM 1.17x, then 0.21x down to 0.02x. **Zero airports at 3.0x, zero at
2.0x.**

**Threshold was >= 3.0x at >= 4 of 10 airports AND >= 20% of matched SSE; kill at < 2.0x
everywhere. Measured max 1.17x, zero airports at 2.0x. VERDICT: FAIL — H-A NOT WORKING.**

The finding is the opposite of the hypothesis: every filed-time signature is **depleted** in the
tail at all ten airports. The tail is where `AOBT_3` *disagrees* with the filed times, not where it
copies them. So the tail is not a fallback-provenance artefact and carries no serve-time-visible
signature of that kind. **Lane closed; do not revisit without genuinely new information.**

**Strategic consequence, stated plainly: this was the only named candidate of the magnitude
required to reach 248** (its ceiling, had it passed, was ~16,500 MSE). With it dead, no identified
mechanism closes the remaining gap.

## R3.2 SCREEN B — regressor lane clears the amendment's bar; classifier headroom is LIRF-only

Arms fitted on the ten non-holdout months, scored on the fold's 5,321 unmatched rows. The
reconstruction of `p_hat x nf_cells` reproduces `bs.fit_unmatched` to atol 1e-9. Monsters
(`y > 3h`) = 56, of which 52 are LIRF.

| arm | pooled | **ex-monster (n=5,265, decisional)** |
|---|---|---|
| B0 `p_hat x nf_cells` (incumbent) | 1867.90 | **995.60** |
| B1 oracle x `nf_cells` | 1601.26 | 747.29 |
| B2 `p_hat x nf_fit` (shippable) | 2190.81 | **927.57** |
| B3 oracle x `nf_fit` (the bound) | 3054.33 | **633.10** |

- **Regressor:** threshold `B0 - B2 >= 20 s` ex-monster; measured **+68.0 s**, row-bootstrap 95%
  [18.6, 169.9]. **Clears the Amendment 10 bar.** B2 beats B0 at 9 of 10 airports.
- **Classifier:** measured headroom **294.5 s** [217, 377], threshold >= 100 s. **ALIVE** — but see
  the qualifications below, which matter more than the verdict.

### Three qualifications, none of which the verdict alone conveys

**(a) A sign error in Amendment 10.2, mine.** It reads "classifier lane ALIVE iff `B3 - B2 >= 100 s`".
B3 is the oracle and therefore has the LOWER RMSE, so `B3 - B2` is negative and the literal reading
returns DEAD. The intent is unambiguous from the same amendment ("B3 ... the bound", "classifier
headroom"), and the delegated agent flagged the contradiction and asked rather than silently taking
the favourable reading. The verdict above uses the headroom reading `B2 - B3`. **The threshold was
written wrong; it is corrected here rather than quietly reinterpreted.**

**(b) "Clears the bar" is NOT "ESTABLISHED" by this project's standing rule.** The global rule
requires a paired bootstrap over >= 3 seeds with the gain exceeding 2x the measured seed sd. This
was a single fit with a row bootstrap and no seed replication, and the interval's lower bound
(18.6 s) sits *below* the 20 s threshold the point estimate clears. **Status: promising, not
established.** Seed replication is required before any promotion.

**(c) The pooled numbers run the other way, and this blocks shipping as-is.** B2 pooled is **+323 s
WORSE** than B0. The cause is mechanical: winsorising `nf_fit`'s target at 3,000 s caps its output
near 2,952 s, while `nf_cells` reaches 80,989 s on LIRF's fine cells, so on non-fill monsters the
fitted arm abandons tail load the cell estimator was carrying. Per Amendment 9 rule 4 the
ex-monster figure is decisional for *whether the mechanism works* — but **the monsters exist on the
board too**. Wiring `nf_fit` as-is would trade a 68 s ex-monster gain for a heavier monster load.
A full experiment must test an un-winsorised or hybrid variant (body from `nf_fit`, tail load from
`nf_cells`) before anything ships.

**(d) The classifier headroom is a LIRF problem, not a general one.** B2 -> B3 moves LIRF from 2,863
to 1,097 and moves every other airport by under 10 s. LIRF is the airport whose schedule-fill rate
is 48.5% against 0.2-9.4% elsewhere. Any work here is LIRF work and should be scoped and named as
such.

## R3.3 Where this leaves the target

Screen A closed the only named mechanism of the right size. Screen B's surviving gains are real but
small and lane-local: an ex-monster stratum improvement worth roughly 1,000 MSE at the fold's
weights, against a board gap to the leader of 30,158 MSE. **The pre-committed stop condition from
the 09-09 planning review therefore fires: with both screens' large-magnitude candidates dead, the
honest target is a top-10 finish, not 248.** Consolidating the already-measured but never-assembled
matched components remains the largest single available gain and is unaffected by either screen.

---

# AMENDMENT 11 — the LIRF fill regime is observable in the arrival stream; thresholds locked

**2026-09-09. Written and committed BEFORE the experiment runs. Nothing above this line has been
edited.**

## 11.0 Why this lane, and why now

Three measurements this session, all on `data/raw/`, reframe the competition:

1. **The board is quantized.** Team trajectories plateau at ~284, ~277, ~265 and 248. `youthful-giraffe`
   sat at 263-266 for ten submissions before a single 15 s step; `quick-boat` at 265-269 for fifteen.
   Plateaus that many teams cannot tune past are discrete modelling decisions, each ~5-9k MSE.
2. **The stake at LIRF is 228,371 MSE — 2.5x our entire score.** The 383 unmatched LIRF rows in the
   scored file carry `sp` up to 111,654 s; on 2025 data a LIRF fill has `|y - sp|` p50 = 3 s, p90 = 5 s,
   and a non-fill has y p50 = 1,070 s. Every one of these rows is a bet on P(fill); the 12h+ band alone
   is 127k MSE on 11 rows. The leader's 7,290-MSE step at v13 is the size of one such bet flipping.
3. **P(fill) at LIRF is non-stationary and a 2026 observable tracks it.** The unmatched-departure fill
   rate swings from 28.3% (Jul 2025) to 77.0% (Dec 2025) by month and from 8% to 100% by airline. The
   L-e logistic (STRATUM_MONSTERS §3b) is calibrated within 2025 and its own verdict names as NOT
   COVERED "any change in LIRF's handling-agent mix in 2026". The LIRF **arrival** stream — 160,504
   rows/yr whose `BLOCK_TIME_UTC_mvt` is populated in `ranking.parquet` — shows the same
   `BLOCK == SCHED` stamping, and its rate tracks the departure-unmatched fill rate at **corr 0.79
   across months and 0.72 across airlines** (day-level corr 0.15: it is a regime signal, not a daily
   one). In 2026 it reads Jan 10.8% / Jul 12.3%, against 2025's Jan 11.0% / Jul 10.5%.

Screen A closed the tail-provenance lane; Screen B found the classifier headroom is LIRF-only. This
amendment is what "LIRF-only" turns out to mean.

## 11.1 Hypothesis H-11, precisely

Adding regime features computed from the LIRF **arrival** rows of the same file — the fraction of
arrivals with `|BLOCK_TIME - SCHED_TIME| <= 60 s`, by calendar month, by airline (3-letter prefix)
shrunk toward the month rate with k = 20, and by airline x month shrunk toward the airline rate —
to the L-e logistic's design matrix improves the calibration of P(fill) on LIRF unmatched departures
under month-to-month regime shift, and reduces stratum RMSE.

Serve-time legitimacy, verified: every feature reads only ARR rows' `BLOCK_TIME` (100% populated on
`ranking.parquet` ARR rows) and `SCHED_TIME`; no DEP `BLOCK_TIME` or `TAXITIME` is touched. The
existing `test_serve_time_contract` guards the latter.

## 11.2 Harness — identical in shape to STRATUM_MONSTERS §3b so the result is comparable

12-fold leave-one-month-out over 2025. For held-out month m: fit on the other eleven; the
arrival-stamping features for month m's rows are computed from **month m's arrivals only** — this
mirrors serve time exactly, where `ranking.parquet` supplies Jan+Jul 2026 arrivals and nothing else
from 2026. Training rows' features are computed from their own month's arrivals.

Control arm = L-e exactly as shipped in `build_submission.py::_lirf_fill_model`. Treatment arm =
L-e + the three regime features. Both inside the shipped mixture; cells elsewhere unchanged. Only
LIRF rows change, so the stratum comparison is over all ten airports' unmatched rows.

Report, for both arms: stratum RMSE pooled and **ex-monster (y > 3h)**; AUC and reliability deciles on
LIRF unmatched; and the **regime-tracking error** — for each of the 12 held-out months, the absolute
gap between the mean predicted P(fill) and the realised fill rate on LIRF unmatched rows, averaged
over months. Paired month-block bootstrap (12 blocks, 2,000 draws) on the stratum RMSE difference.

Negative control: the same treatment arm with the three regime features permuted across months.
The gain must vanish; if it does not, the gain is not the mechanism.

## 11.3 TRUE / FALSE shapes, locked

- **ESTABLISHED** iff the month-block paired interval on stratum RMSE (treatment minus control)
  excludes zero on **ex-monster** rows, AND the regime-tracking error falls by **>= 30%** relative
  to L-e, AND the permuted control's gain is inside its own noise. All three.
- **NOT WORKING** iff the ex-monster interval includes zero, OR the regime-tracking error does not
  fall by >= 15%, OR the permuted control retains >= half the gain.
- Between: **inconclusive**; do not ship; report and stop spending on it.

Additionally reported, not decisional: the pooled-with-monsters interval (Amendment 9 rule 4 makes
ex-monster decisional, but the monsters are where the money is on the board, so the pooled number is
stated alongside it in every table).

## 11.4 What a WORKING result licenses, and what it does not

Licenses: wiring the three features into `_lirf_fill_model`, a `--validate` A/B, and a v3
submission that differs from v2 ONLY in this. That makes v3 a clean board measurement of the lane.

Does not license: any board projection from the fold (Amendment 9), any claim about the 2026 fill
rate itself (the 2026 arrival stamping is an input, not a verdict), or any change outside LIRF.
