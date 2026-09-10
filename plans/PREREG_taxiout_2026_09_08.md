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

---

# RESULT 4 · 2026-09-09 · Amendment 11 — NOT WORKING

12-fold LOMO exactly as specified in 11.2; control reproduces `bs.fit_unmatched` bit-exact on every
fold; wall-clock 4 s, peak RSS 2.86 GB. Script `amend11_lomo.py` and `amend11_result.json` in the
session scratchpad; numbers reproduced below from the run log.

| arm | stratum RMSE pooled | **ex-monster** | AUC LIRF | regime-tracking error |
|---|---|---|---|---|
| control (L-e as shipped) | 1610.68 | 958.96 | 0.877 | 0.0709 |
| treatment (+3 arrival-regime features) | 1600.09 | **950.54** | 0.885 | 0.0512 |
| permuted (features shuffled across months) | 1606.84 | **952.98** | 0.884 | 0.0589 |

Paired month-block bootstrap, treatment − control, ex-monster: **−8.42 [−13.27, −3.49]**.
Permuted − control, ex-monster: −5.98 [−13.51, +3.33].

**Clauses (11.3):**
- (a) ex-monster interval excludes zero in the improving direction — **TRUE**.
- (b) regime-tracking error falls 27.8% — between the 30% ESTABLISHED bar and the 15% kill bar;
  **inconclusive** on its own.
- (c) the permuted arm retains **71% of the gain** (5.98 of 8.42) — **>= half, which is a NOT
  WORKING trigger.**

**VERDICT: NOT WORKING.** The hypothesis was that the arrival stream's *month-level* stamping
regime calibrates P(fill) under shift. It does not: shuffling the month assignment leaves most of
the gain intact, so what the features carry is the *airline-level* stamping rate, which is one more
encoding of airline identity — a signal L-e already has. The residual attributable to the actual
mechanism is ~2.4 s of ex-monster stratum RMSE, and the whole treatment is worth ~250 MSE at fold
weights. Not shippable, not worth a submission.

**What this closes and what it does not.** Closed: aggregate (month, airline, airline×month)
arrival-stamping rates as regime features. Not addressed: a *row-level* link — whether the inbound
arrival of the same turnaround (same stand, immediately prior) carries the stamp flag. That is a
different hypothesis with a different mechanism (a per-record capture failure shared by both legs)
and is screened separately before any amendment is written for it.

Also on record: the per-airline 2026 arrival-stamping drift table computed the same night (RYR
15→22%, LAV 56→26%, 60% of the LIRF stake on airlines that moved >3pp) was a preview of THIS
mechanism and is therefore **not evidence of anything** now that the mechanism is refuted.

### R4 addendum · same night · a post-hoc null, and why the verdict stands anyway

After the pre-registered run, the delegated agent ran a **post-hoc, non-decisional** null: 200
distinct month-derangements (seed 1) instead of the single seed-0 derangement the amendment locked.
The seed-0 derangement turns out to preserve the month-level correlation between arrival stamping
and realised fill (+0.647, the 100th percentile of the null), i.e. it happened to map months onto
months with similar stamping rates, so the "permuted" control still carried the signal. Against
the 200-derangement null: permuted gain mean +0.08 s [−7.77, +5.24]; only 6% of derangements
retain half the treatment gain; the treatment's +8.42 s sits at the 100th percentile; the
treatment's 27.8% tracking drop also sits at the 100th percentile (3% of derangements reach 15%).

**Read honestly:** the registered control was under-designed — one derangement is a weak null —
and under a proper null the *mechanism* looks real. **The verdict is not flipped**, for two reasons
that are stronger than the rule: (1) flipping a pre-registered verdict on a test chosen after
seeing the data is the thing this document exists to forbid; (2) it does not matter, because the
effect is **8.4 s of ex-monster stratum RMSE ≈ 250 MSE at fold weights**, against a 228,371 MSE
LIRF stake and a 30,158 MSE gap. Whether or not the month regime is readable from arrivals, it
does not move the bets. **NOT WORKING stands, on magnitude.**

Lesson for future amendments: a negative control must be a *distribution* (>= 100 permutations),
not one draw. Written into the next amendment that uses one.

### R4 screen · turnaround linkage · weak, no amendment

Shape stated before running: worth an amendment if P(fill | inbound same-stand arrival stamped)
− P(fill | inbound not stamped) >= 40pp at LIRF unmatched with >= 60% linkable; dead if < 15pp.
Measured (12 months, 1,488 LIRF unmatched departures, 94.9% linkable within 24h):

| link set | n | P(fill \| inbound stamped) | P(fill \| inbound not) | diff |
|---|---|---|---|---|
| all linkable | 1,412 | 56.7% (n=194) | 45.2% (n=1,218) | **+11.5pp** |
| same airline both legs | 885 | 46.0% (n=124) | 28.4% (n=761) | +17.6pp |
| gap < 6h | 1,189 | 52.5% (n=141) | 42.3% (n=1,048) | +10.2pp |

On LIRF MATCHED rows (150,259 same-airline links): 36.6% vs 18.5%, +18.1pp. The stamp is loosely
shared across a rotation — consistent with an airline/handler effect, not a per-record capture
failure. **Below the amendment bar; no per-row fill identifier here.** At most a minor feature for
a consolidation bundle, and it must not be described as anything more.

### Where the LIRF stratum stands after tonight

Three probes into the 228k stake — aggregate arrival regime (R4), turnaround linkage (this
screen), and the oracle-vs-fitted bound (R3.2) — all return the same shape: **P(fill) at LIRF is
calibrated as well as the available observables allow, and no observable found so far identifies
fills per row.** The stratum's remaining error is bet variance, not model error. Further LIRF
classifier work is closed absent a genuinely new observable.


---

# RESULT 5 · 2026-09-09 · v3 on the board: 291.63, rank 20 of 87 — matched fold gains transfer

v3 differs from v2 ONLY on the 339,551 matched rows (asserted: 5,290 unmatched rows byte-identical,
id order identical). Board: **291.6317**, usedPairs 344,841, processed 14:19:14Z. Against v2's
301.7019: **−10.07 s, −5,975 MSE**. The projection from the fold's matched A/B (237.46 → 226.24,
`lgbm_ab_full.log`) was −8.6 s / −5,123 MSE; the realised gain is ~117% of it.

**Verdict on the standing question "do fold gains transfer": matched-side relative gains DO, at
>= 1:1 (n = 339,551, no row concentration). Fold TOTALS still do not (Amendment 9).** Both are now
measured and they are consistent. Planning rule from here: a matched-side gain measured on the fold
with a paired interval excluding zero is bankable on the board; a stratum or total number is not.

Operational note for the next fit: the first launch ran lightgbm's predict single-threaded under
the OMP_NUM_THREADS=1 cap for 40 min before being killed; fixed at the origin (`predict_delta`
forwards `num_threads`, commit a570407). Second launch: 56 min end to end.

---

# AMENDMENT 12 — v4: seed averaging and the per-airport blend on the LightGBM path; thresholds locked

**2026-09-09, written BEFORE the fold measurement runs. Nothing above this line has been edited.**

## 12.1 Hypotheses

- **H-12a (seeds):** averaging `delta_hat` over three LightGBM refits (seeds 0, 1, 2; identical
  params and tree count) reduces matched RMSE on the fold relative to the single seed-0 model.
  Prior evidence: 2024-competition analogue (single 1612 → ten-model 1564, ~3%); unmeasured here.
- **H-12b (per-airport blend):** blending the pooled prediction 0.5/0.5 with a per-airport LightGBM
  (airports with >= 20,000 training matched rows; others keep pooled) reduces matched RMSE further.
  Prior evidence: +3.08 s per-airport and +2.46 s blend, measured on HGB (`build_submission.py`
  docstring); never tried on LightGBM.

## 12.2 Harness (`scripts/lgbm_fold.py`), fold A exactly as `lgbm_ab.py`

Holdout months (1, 7); ES months (3, 9) held out of the training fold to find best_iter; refit on
the ten training months at n_ref; in-fold encodings; matched rows only; y_hat = max(proxy −
delta_hat, 1). Arms: **A0** seed 0 alone (must reproduce `lgb_refit` 226.24 within 0.5 s, else the
harness is wrong and nothing below counts); **A1** seeds 1 and 2 alone; **A2** mean of seeds 0, 1, 2;
**A3** A2 + per-airport blend. Paired row bootstrap, 2,000 draws, seed 0. Seed sd := sd of the three
single-seed matched RMSEs. Everything reported pooled AND per airport.

## 12.3 Thresholds, locked

- **H-12a ESTABLISHED** iff the paired interval of A2 − A0 excludes zero (improving) AND the point
  gain exceeds **2 × seed sd** (the repo's standing ESTABLISHED rule). NOT WORKING iff the interval
  includes zero or the gain is below 1 × seed sd. Between: inconclusive — ship A2 anyway ONLY if the
  interval excludes zero (averaging cannot hurt in expectation), but do not call it established.
- **H-12b ESTABLISHED** iff the paired interval of A3 − A2 excludes zero (improving) AND >= 6 of 10
  airports individually improve. NOT WORKING iff the interval includes zero. If A3 − A2 is
  established but fewer than 6 airports improve, ship per-airport ONLY at the airports that improve
  (named in advance as the fallback rule, not decided after seeing which).
- **What ships as v4:** the best arm among {A2, A3} that clears its rule; v4 differs from v3 only in
  that. Expected board effect, matched-side transfer >= 1:1 per RESULT 5: −2 to −4 s. **No board
  projection beyond that range is to be quoted.**

## 12.4 What no result here licenses

Nothing on the stratum. Nothing about seeds beyond three (a fourth seed is a new measurement). The
per-airport tree-count rule the implementer chooses is recorded in the fold log and is part of the
tested configuration; changing it after the fold run is a new amendment.


---

# AMENDMENT 13 — binding constraint: no leaderboard probing, no reading labels off the scorer

**2026-09-09, owner decision, verbatim: "we are not cheating and in any case don't cheat anything
anywhere."** Nothing above this line has been edited.

Submissions exist to be scored, not to be read. No file is ever uploaded whose purpose is to learn
the hidden taxi-out of any row from the score it returns — not one row at a time, not in groups,
not by bisection, not by any encoding. This holds regardless of what the competition rules permit.
Any future session that derives this idea (it is the only mechanism of the right size for the last
~15k MSE, and the leader's flat-then-step trajectory looks like it) stops here.

Consequence, recorded so it is not re-litigated: **the honest target for this competition is the
engineered ceiling, ~275–280, top-10. 248 is not the target.** Every submission from v4 on differs
from its predecessor only by a modelling change measured on the fold with a paired interval.


---

# AMENDMENT 14 — v6: push-anchored queue features on the matched rows; thresholds locked

**2026-09-09, written BEFORE the feature block is finished and BEFORE any fold measurement.
Nothing above this line has been edited.**

## 14.1 Hypothesis H-14

On matched rows, features anchored on the row's OWN push (`AOBT_3`) — the number of aircraft
already pushed and not yet airborne when I push, at the airport and on my runway; take-offs that
occur during my taxi; pushes in the 20 minutes before mine; symmetric take-off density around my
own take-off; arrivals taxiing at my push and their taxi-in level; the runway's ambient `proxy`;
the gap to the next arrival at my stand — reduce matched RMSE on the fold beyond the best v4 arm.
Mechanism named in advance: the 12,658-MSE "pushed back early, then held" tail (non-fill,
|delta| > 20 min, 78% with y longer than proxy) is a queueing phenomenon, and the incumbent SURF
block is take-off-anchored and backward-looking, so it cannot see queue state at the moment of push.

Serve-time legitimacy: every feature reads other departures' `MVT_TIME`/`AOBT_3`/`SCHED` and
arrivals' `MVT_TIME`/`BLOCK_TIME`/`TAXITIME`, all populated on `ranking.parquet`; none reads a
departure's own `BLOCK_TIME`/`TAXITIME`. Tested by the hidden-clock test the block must ship with.

## 14.2 Harness

`scripts/lgbm_fold.py` fold A, exactly as Amendment 12: holdout (1, 7), ES (3, 9), in-fold
encodings, matched rows only, paired row bootstrap 2,000 draws seed 0. Baseline arm = the v4 arm
that shipped (A2 or A3 per Amendment 12's rule). Treatment arm = baseline + `QUEUE_FEATS` appended
to the design matrix, everything else identical (same seeds, same tree-count procedure — best_iter
re-found with the extra columns, since capacity may shift). Reported pooled and per airport.

## 14.3 Thresholds, locked

- **ESTABLISHED** iff the paired interval of (treatment − baseline) matched RMSE excludes zero in
  the improving direction AND the point gain >= **+2.0 s** AND the gain exceeds 2 × the seed sd
  measured in Amendment 12 AND >= **4 of the 10 airports** improve individually with the four
  named in advance as the likely ones — LEBL, LTFM, EGLL, LSZH — not required to be exactly those.
- **NOT WORKING** iff the interval includes zero, or the point gain < +1.0 s.
- Between: inconclusive; do not ship; queue behind v5.
- Additionally reported, not decisional: the RMSE on the pre-defined tail subset (non-fill,
  |delta| > 1,200 s) for both arms, because that is where the mechanism says the gain must live. If
  the pooled gain is established but the tail does NOT improve, the gain is not the named mechanism
  and the amendment says so in the result rather than claiming it.
- **What ships as v6:** baseline + QUEUE_FEATS iff ESTABLISHED; v6 differs from v5 (or v4) only in
  that. Expected board effect: −2 to −4 s, per the transfer rule of RESULT 5. No larger projection
  is to be quoted.

## 14.4 Negative control

The treatment arm re-run with QUEUE_FEATS shuffled across rows WITHIN airport (100 permutations,
per the Amendment 9/R4 lesson that one draw is not a null); the treatment's gain must exceed the
95th percentile of the permuted gains. If it does not, NOT WORKING regardless of 14.3.


---

# AMENDMENT 15 — the fifth lane: a second learner family and a capacity sweep; thresholds locked

**2026-09-09, written BEFORE any measurement. Nothing above this line has been edited.**

## 15.0 Why a fifth lane

The 10th place on the board is a moving target: 326.6 → 301.7 → 294.9 → 288.9 → 281.5 → 278.2 over
09-04..09-09, decelerating; extrapolated to the 4 Oct freeze it sits near 270–274. Planning bar:
**<= 272**, i.e. ~11,000 MSE below v3's 85,049. Amendments 12 and 14 plus the matched-fill head and
the stratum hybrid sum to 6–10k on their fold ranges. A fifth lane is therefore required, not
optional, and it is pre-registered here before it is built so the bar is not lowered later.

## 15.1 H-15a — second learner family (CatBoost), blended

CatBoost at matched capacity (depth 8, lr 0.03, ~5k–10k iterations found by the same ES months,
seed 0; run ONLY from the isolated venv `~/.venvs/prc-catboost`, never installed globally — it
needs numpy<2 and the live fleet's site-packages must not be touched) on the identical design
matrix and target, blended 0.5/0.5 with the best LightGBM arm. Mechanism: residual correlation
between families is below 1, so the blend's variance is below either alone (HGB/LGBM residual corr
was 0.954 in `lgbm_ab`; the gain is bounded by that).
- **ESTABLISHED** iff paired interval of (blend − LGBM arm) excludes zero AND gain >= +1.5 s AND
  the blend weight curve (0.3/0.5/0.7) is not monotone toward 0 (i.e. CatBoost is contributing,
  not merely diluting).
- **NOT WORKING** iff interval includes zero or gain < +0.7 s.

## 15.2 H-15b — capacity sweep, screened then confirmed

The `lgb_refit` parameters were taken from lgbm_ab's single setting, never swept. Sweep
`num_leaves` in {127, 255, 511} x `min_data_in_leaf` in {20, 40, 100} x `feature_fraction` in
{0.6, 0.8}: 18 settings. **Screening tier:** rank all 18 on a 4-month subfold (train Feb–May,
stop on Jun; matched rows) at lr 0.05 — ranking only, NO magnitude from this tier may be quoted
(this project has twice been burned quoting subsample magnitudes). **Confirmation tier:** the top
2 by screen rank re-run on the full fold A exactly as Amendment 12, against the incumbent setting.
- **ESTABLISHED** iff the confirmed best beats the incumbent with a paired interval excluding zero
  AND gain > 2 x seed sd.
- **NOT WORKING** iff neither confirmed setting beats the incumbent's interval.
- A setting that screens first but fails confirmation is recorded as such — the screen's ranking
  power is itself a measurement of whether the fast harness is trustworthy (RESULT to be written).

## 15.3 What ships

Whichever of 15.1 / 15.2 is ESTABLISHED ships as its own submission, differing from its
predecessor only in that change. Combined effects are measured jointly on the fold before a
combined submission — no summing of separately measured gains into a claimed total.


---

# RESULT 6 · 2026-09-09 · screen: flight-number fill propensity at LIRF — DEAD

Owner set the target to first place, which reopens the LIRF stratum search. Shape stated before
running: a per-flight-number fill history (mangling stripped, LOMO target encoding, shrunk to the
airline) is amendment-worthy if LOMO AUC on LIRF unmatched fill >= 0.92 against the L-e's 0.877;
dead below 0.89.

| key | history source | LOMO AUC |
|---|---|---|
| airline (L-e's key) | unmatched rows | 0.826 |
| flight number | unmatched rows | 0.817 |
| flight number | MATCHED-row copy history (159k rows) | 0.810 |
| flight number x hour-band | matched rows | 0.797 |

**DEAD (< 0.89).** A rotation's identity carries no information beyond its airline. Fourth
legitimate probe into the 228k LIRF stake (aggregate arrival regime, turnaround stamp, oracle bound,
flight-number history) returning the same shape: P(fill) is as calibrated as the observables allow.
Note for the record: the 12h+ band in 2025 (n = 29) shows a 62% fill rate with AUC ~0.6, i.e. the
biggest bets are the least discriminable — yet the board score is only consistent with those bets
having been won in 2026, so the 2025 sample there is small and not to be planned against.

**Standing rule (Amendment 13): no leaderboard probing.** The stratum's remaining error is bet
variance under a calibrated model; first place therefore has to come from the matched side and from
the stratum's REGRESSOR (Screen B's fitted non-fill model), not from fill classification.


### 12.5 · decision recorded BEFORE the fold run · per-airport tree count = `--pa-trees es`

The implementer built two rules and refused to pick: `share` (each airport gets n_ref scaled by
its row fraction, floor 200 — 1,229–2,700 trees on the fold) and `es` (each airport early-stops on
its own fit/ES rows, same months, seed 0, then refits at its own n_ref). The v3 stopping curve loses
~4.8 s between 2,000 trees and its optimum, so `share` would under-train the per-airport models and
H-12b could read NOT WORKING because of the rule rather than the blend. **Decision: `es`.** Cost
+~30 min per run. Recorded here, before launch, per 12.4; the fold log carries the rule string.


### 14.4 amended · 2026-09-09 11:20 local · BEFORE any queue-arm run

100 full retrains for the within-airport permutation control is infeasible (~25 min each). The
control is implemented as **20 permutations (seed 0), each a full retrain at QUARTER capacity
(best_iter / 4), paired against the unpermuted treatment retrained at the SAME quarter capacity**,
so the comparison is like-for-like. The treatment's quarter-capacity gain must exceed the 95th
percentile of the 20 permuted quarter-capacity gains. This replaces "100 permutations" in 14.4;
the decisional interval in 14.3 is still the full-capacity paired interval. Recorded before the
arm exists in runnable form.

---

# AMENDMENT 16 — v5: a schedule-fill mixture head on MATCHED rows; thresholds locked

**2026-09-09, written before the head is built and before any measurement. Nothing above this line
has been edited.**

## 16.1 Hypothesis H-16

On matched rows, 8.9% have `BLOCK_TIME == SCHED_TIME` (the airport stamped the schedule), so
`y == sp` exactly for them and the proxy identity `y = proxy − delta` is the wrong model. They carry
13.5% of matched SSE; an oracle that predicts `sp` on them and the incumbent elsewhere scores
221.15 against 237.46 (7,365 global MSE at fold weights; LIRF 4,452 of it). The tree already leans
65% of the way toward `sp` on them (implicit p median 0.65). H-16: a fitted classifier
`p = P(fill | x)`, trained on all matched training rows (label visible on all 2M), used as an
MSE-optimal mixture `y_hat = p * sp + (1 − p) * (proxy − delta_hat_base)`, reduces matched RMSE
beyond the best v4/v6 arm.

Classifier: LightGBM binary on the same 68-feature design (+ `nmdelay = proxy − sp` explicitly),
early-stopped on the ES months, seed 0; reliability deciles reported. Serve-time: every input is
already in the serve-mode cache.

## 16.2 Harness

`lgbm_fold.py --fillhead`, fold A as Amendment 12; baseline = the best shipped arm at the time;
treatment = baseline + mixture head. Paired row bootstrap 2,000 draws seed 0. Reported pooled, per
airport, and on the two pre-defined subsets: **fills** (`|y − sp| <= 60`) and **non-fills**.

## 16.3 Thresholds, locked

- **ESTABLISHED** iff the paired interval of (treatment − baseline) excludes zero (improving) AND
  point gain >= **+1.5 s** AND the gain on the FILL subset >= **+5 s** (the mechanism must show up
  where it is claimed) AND the NON-FILL subset is not worse by more than **0.5 s** (the head must not
  poison the majority) AND the gain exceeds 2 x the Amendment 12 seed sd.
- **NOT WORKING** iff the interval includes zero, or the fill-subset gain < +2 s, or the non-fill
  subset worsens by > 1.0 s.
- Between: inconclusive; do not ship.
- Reported, not decisional: LIRF alone (60% of the ceiling); reliability deciles of `p`.
- **What ships as v5:** baseline + head iff ESTABLISHED, differing from its predecessor only in
  that. Expected board effect −2 to −4 s (transfer rule of RESULT 5). No larger projection.


### 14.4 amended (ii) · 2026-09-09 12:05 local · BEFORE any queue-arm run

Building the control surfaced a confound: at reduced capacity, every within-airport permuted run
beat a 68-column reduced baseline by the same ~8 s, because LightGBM's `feature_fraction` sampling
stream depends on the column count — a baseline with fewer columns is not like-for-like with an
80-column treatment even at equal tree counts. The control is therefore stated WITHOUT a baseline:
**the treatment's quarter-capacity RMSE must be below the 5th percentile of the 20 permuted
quarter-capacity RMSEs**, all runs at the same 80 columns, same seed, same tree count. This is the
"gain exceeds the 95th percentile of permuted gains" form with the (cancelling) baseline removed.
The JSON reports both forms; the verdict uses this one. The decisional interval in 14.3 is
unchanged (full-capacity paired interval against the shipped v4 arm's own predictions).


---

# RESULT 7 · 2026-09-09 13:28 local · Amendment 12 fold — seeds ESTABLISHED (small), per-airport NOT WORKING

`scripts/lgbm_fold.py --pa-trees es`, fold A, matched rows n = 339,015; 2 h 18 min; peak RSS ~4.2 GB.
Log `reports/lgbm_fold_v4.console.log`, numbers `reports/lgbm_fold_v4.json`, per-row predictions
`data/cache_stand/fold_v4_preds.parquet`. **A0 reproduces `lgbm_ab`'s `lgb_refit`: 226.309 vs
226.245 (+0.064 s, tolerance 0.5); best_iter 17,525 vs 17,557; n_ref 21,908 vs 21,948.** The harness
measures the procedure that shipped as v3.

| arm | matched RMSE |
|---|---|
| A0 seed 0 | 226.309 |
| A1a seed 1 / A1b seed 2 | 226.489 / 226.202 |
| **A2 mean of 3 seeds** | **225.825** |
| A3 = A2 + per-airport blend (0.5, `es` trees, 10/10 airports fitted) | 225.388 |

Seed sd (ddof=1) over the three singles: **0.145 s**; 2× = 0.291 s.

| comparison | point | paired 95% (2,000 draws) | excludes 0 | gain / seed sd | airports improving |
|---|---|---|---|---|---|
| A2 − A0 | **+0.484** | [+0.321, +0.652] | yes | 3.33 | 10 / 10 |
| A3 − A2 | +0.437 | [−0.157, +1.045] | **no** | 3.01 | 8 / 10 |
| A3 − A0 | +0.921 | [+0.304, +1.544] | yes | 6.33 | 8 / 10 |

**H-12a (seeds): ESTABLISHED** by 12.3 — interval excludes zero, gain 3.3× seed sd, 10/10 airports.
Honest magnitude: 0.48 s of matched RMSE ≈ **~215 MSE at fold weights, ~0.5 s on the board**. The
"2024 analogue, ~3%" prior was wrong by ~15×; averaging three seeds of a 22k-tree model with
bagging already inside it buys almost nothing. Ships as v4 because it is real and because the three
seed boosters are the prerequisite for v5 (Amendment 16) — not because it moves the rank.

**H-12b (per-airport blend): NOT WORKING** by 12.3 — the pooled interval includes zero. Per
airport: EDDF +1.41*, EDDM +1.33, EGLL +0.31, EHAM +1.09, LEBL +0.29, LEMD +1.89*, LFPG +2.16,
**LIRF −2.70***, LSZH −1.76, LTFM +0.73* (* = the airport's own interval excludes zero). Eight of
ten improve; LIRF — the largest RMSE — gets significantly worse and dominates the pooled variance.
The 12.3 fallback ("ship per-airport only where it improves") applies only when A3 − A2 is
established; it is not. **Nothing per-airport ships.** Choosing the airport list after seeing this
table would be a goalpost move; instead the pattern is registered below as a new hypothesis for a
DIFFERENT fold.

Per-airport tree rule used: `es` (12.5); per-airport n_ref recorded in the JSON.

---

# AMENDMENT 17 — per-airport blend EXCLUDING LIRF and LSZH; a fold-B hypothesis, not a v4 change

**2026-09-09 13:35 local. Generated from RESULT 7's per-airport table, therefore NOT testable on
fold A. Nothing above this line has been edited.**

H-17: A2 blended 0.5/0.5 with the per-airport model at the eight airports where RESULT 7 showed
improvement (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LTFM), A2 at LIRF and LSZH, beats A2 on
**fold B** (holdout months (2, 8); ES months (4, 10)) with a paired interval excluding zero and
>= 6 of the 8 selected airports improving individually. NOT WORKING iff the fold-B interval
includes zero. Fold A may not be used for this verdict. Expected value ~0.5–0.9 s matched; low
priority against Amendments 14–16; run only when compute is idle.


---

# RESULT 8 · 2026-09-09 13:46 local · Amendment 16 (fill head) — NOT WORKING

`lgbm_fold.py --fillhead --baseline A2 --pa-trees es`, fold A, 339,015 matched holdout rows, 11 min,
peak 3.4 GB. Head: LightGBM binary, 69 features, best_iter 4,578 → n_ref 5,723; holdout AUC 0.896;
mean p 0.0855 vs fill share 0.0890 (calibrated in the mean); mean p on fills 0.297, on non-fills
0.065. Log `reports/lgbm_fold_fillhead.console.log`, JSON `reports/lgbm_fold_fillhead.json`,
predictions `data/cache_stand/fold_preds_fillhead.parquet`.

| subset | rows | baseline A2 | treatment | gain | paired 95% |
|---|---|---|---|---|---|
| pooled | 339,015 | 225.825 | 225.683 | +0.14 | **[−0.48, +0.72]** |
| fills (|y−sp| ≤ 60) | 30,167 (8.9%) | 288.72 | 251.05 | **+37.68** | [+34.26, +41.51] |
| non-fills | 308,848 (91.1%) | 218.71 | 223.05 | **−4.34** | [−4.76, −3.97] |
| LIRF alone | 26,131 | 383.32 | 379.33 | +3.99 | [−0.33, +8.65] |
| LIRF fills | 4,947 | 560.61 | 462.76 | +97.85 | [+86.29, +109.43] |
| LIRF non-fills | 21,184 | 328.41 | 357.05 | −28.64 | [−32.45, −25.26] |

Per airport: 9 of 10 worse, each with its own interval excluding zero (EDDF −0.44, EDDM −0.55,
EGLL −0.46, EHAM −0.39, LEBL −0.48, LEMD −0.90, LFPG −0.55, LSZH −0.29, LTFM −0.15); LIRF +3.99.

**Clauses (16.3):** pooled interval includes zero → NOT WORKING; fill-subset gain +37.7 ≥ +5 ✓;
**non-fill subset worse by 4.34 s > 1.0 s → NOT WORKING on its own.** Verdict: **NOT WORKING.**

**Why, mechanically.** The head is calibrated and ranks well (AUC 0.90), but a mixture pays the
residual p on every non-fill: with mean p 0.065 and typical |sp − y_hat| in the thousands of seconds,
the leak onto 91% of rows costs 4.3 s while the fills gain 38 s on 9%. Net zero. This is the same
structure Screen B found in the stratum: **a calibrated hedge cannot beat per-row identification,
and no observable identifies fills per row** (four stratum probes + this). The 7,365-MSE oracle
ceiling was real; none of it is capturable by a classifier of this quality. **Lane closed** unless
a head with AUC well above 0.95 appears, which nothing in the data suggests.

Not tried, on record: a mixture applied only where p is extreme (p > 0.9), i.e. a thresholded
head rather than a soft one. It is a different hypothesis (selective override), would need its own
amendment, and on these numbers its upside is bounded by the fill subset's 37 s on the high-p
fraction of 8.9% of rows — small. Deprioritised behind Amendments 14 and 15.

**Standing after RESULT 7 and 8:** v4 = A2 (+0.48 s). Per-airport blend and fill head both dead.
Remaining pre-registered lanes: 14 (queue features), 15.1 (CatBoost blend), 15.2 (capacity sweep),
17 (fold-B per-airport), the stratum regressor hybrid. The queue fold runs next as the largest.


---

# AMENDMENT 18 — v7: a fitted non-fill regressor for the stratum BODY with the cell estimator's
tail load preserved; thresholds locked

**2026-09-09 13:55 local, written before the code exists and before any measurement. Nothing above
this line has been edited.**

## 18.1 Hypothesis H-18

Screen B (RESULT 3.2) established, on the fold, that a fitted non-fill regressor `nf_fit` beats the
hierarchical cell estimator `nf_cells` on EX-MONSTER unmatched rows (B0 995.6 → B2 927.6, +68 s,
9/10 airports) while losing badly POOLED (+323 s worse) because its 3,000-s target winsorisation
abandons the tail load that `nf_cells` carries to 80,989 s on LIRF's fine cells. H-18: a HYBRID
non-fill predictor — `nf_fit` where the row's finest populated cell has `nf_cells < T_tail`, and
`nf_cells` otherwise — keeps the body gain without the tail loss, so the mixture
`p_hat * sp + (1 − p_hat) * nf_hybrid` beats the incumbent `fit_unmatched` on BOTH the ex-monster
and the pooled stratum RMSE. `T_tail` is fixed in advance at **3,000 s** (the winsorisation point,
not tuned). `p_hat` is the incumbent's (L-e at LIRF, cells elsewhere) — unchanged.

`nf_fit`: HistGradientBoostingRegressor, max_leaf_nodes 15, max_iter 200, min_samples_leaf 100,
target winsorised at 3,000 s for fitting only, features sp / dayoff / hr / airport / runway / stand
prefix / airline / operator target-encodings fitted on training rows only — exactly Screen B's B2
regressor, so the ex-monster number is comparable to R3.2.

## 18.2 Harness

12-fold LOMO over 2025 on the stratum (all ten airports' unmatched rows), the STRATUM_MONSTERS §3b
design, plus fold A reported alongside for comparability with R3.2. Arms: **S0** incumbent
`fit_unmatched`; **S1** hybrid. Three seeds of `nf_fit` (random_state 0/1/2) → seed sd; report S1
as the mean over seeds AND per seed. Paired month-block bootstrap (12 blocks, 2,000 draws, seed 0)
on pooled and ex-monster stratum RMSE; per airport; per month.

## 18.3 Thresholds, locked

- **ESTABLISHED** iff (a) ex-monster paired interval of (S1 − S0) excludes zero improving AND
  gain >= 20 s AND > 2 × seed sd; AND (b) pooled paired interval does NOT show S1 worse (upper
  bound of S1 − S0 loss < 20 s, i.e. the tail is preserved); AND (c) >= 6 of 10 airports improve
  ex-monster.
- **NOT WORKING** iff the ex-monster interval includes zero, OR the pooled interval shows S1 worse
  by more than 20 s (the tail was not preserved).
- Between: inconclusive; do not ship.
- Reported, not decisional: the LIRF-only cut; the number of rows routed to `nf_fit` vs `nf_cells`.
- **Ships as v7** (its own version number, differing from its predecessor only on the 5,290
  unmatched rows). Expected board effect: unknowable from the fold's stratum (Amendment 9); the
  board is the instrument. Fold weight of the ex-monster gain ~1,000 MSE.


---

# AMENDMENT 19 — where the error actually is, and two arms aimed at it; thresholds locked

**2026-09-09 17:00 local. Written after the diagnosis below and BEFORE either arm is run. Nothing
above this line has been edited. Owner directive 16:45: "work towards it and find why it is not
improving."**

## 19.0 The diagnosis (measured on the v4 fold record, matched rows, A2)

| |delta| band | rows | RMSE | share of matched SSE |
|---|---|---|---|
| < 2 min | 38.6% | 137.0 | 14.2% |
| 2–10 min | 53.7% | 189.8 | 38.0% |
| 10–20 min | 6.0% | 384.9 | 17.4% |
| > 20 min | 1.7% | 968.8 | 30.5% |

**The 7.63% of matched rows with |delta| > 10 min carry 23,979 global MSE — 47.8% of matched
error.** Perfect knowledge of those rows alone would take matched RMSE from 225.8 to 163.1; the
gap to the leader is ~30,000 MSE. On the 39% of rows where the clocks agree within 2 min, RMSE is
137 s — larger than the band — the signature of a model predicting the conditional MEAN of a
bimodal distribution. That is why nothing today moved the number: every lane attacked the 92% we
already predict well; the error lives in a mode no feature identifies. `FLIGHT_ID_mvt` (the one
unread column) is not the identifier: AUC 0.55 for mode rows.

**The mode clusters on DAYS.** Per airport-day, the day's mode rate (p10 1.4%, p50 5.6%, p90 16.5%,
max 60%) correlates with the day's own departure-`proxy` p90 at 0.79 (LFPG, 0.90 on p50), 0.76
(EGLL), 0.73 (EDDM), 0.65–0.72 (LTFM), 0.68 (EDDF). A top-decile day has 2.3–3.7× the mode rate of
a bottom-decile day at 8 of 10 airports (LFPG 3.68×, EDDM 3.15×, LSZH 3.11×, EGLL 2.83×, LTFM 2.79×;
LEMD 1.0×). Within a day the signal reaches only AUC 0.55–0.63 — regime, not identity. All of this is
computable transductively from `ranking.parquet` (other departures' `proxy`, arrivals' taxi-in), so
external weather data would largely duplicate it.

## 19.1 Arm D — airport-day regime block (`DAY_FEATS`)

Per (airport, calendar day), from the same file the row lives in: departure `proxy` p50 and p90,
share with `proxy <= 0`, arrival taxi-in p50 and p90, share of arrival taxi-in > 1,200 s, departure
and arrival counts; plus each row's `proxy` minus the day's p50. Appended to the best current design
(A2 + QUEUE_FEATS if Amendment 14 ships, else A2). Serve-time: all inputs populated on the scored
file (verified in 14.1 / v6 tests); no departure's own BLOCK/TAXITIME is read.

## 19.2 Arm Y — formulation: predict `y` directly, and blend

Same design, same LightGBM config and seeds, target `y` instead of `delta` (no proxy anchor);
`y_hat_Y = max(prediction, 1)`. Reported alone and as the 0.5/0.5 blend with the delta-target arm.
Mechanism: at LTFM and EDDM `corr(proxy, y)` is 0.08 and 0.16 — the anchor carries almost nothing —
and the one comparison that chose `delta` (252.66 vs 285.24) was on the 26-feature HGB, never on
this learner or these features. Reported pooled, per airport, and on the |delta| bands of 19.0.

## 19.3 Thresholds, locked (same harness as Amendments 12/14; paired row bootstrap 2,000 draws)

- **Arm D ESTABLISHED** iff the paired interval excludes zero, point gain >= **+1.0 s**, gain > 2 ×
  seed sd, >= 6 of 10 airports improve, AND the |delta| > 10 min bands improve by >= +5 s (the block
  must move the mode rows, not the body). NOT WORKING iff the interval includes zero.
- **Arm Y** (blend vs delta-target) ESTABLISHED iff the paired interval excludes zero AND gain >=
  +1.0 s AND > 2 × seed sd. Y alone reported; per-airport Y-vs-delta reported for LTFM/EDDM.
  NOT WORKING iff the blend's interval includes zero.
- Point-gain bars are compute-economy thresholds, as before; the owner decides on real gains below
  them (RESULT 9's precedent).

## 19.4 The honest budget

Regime shifts a day's base rate; it does not identify rows. Expect arms D and Y to be worth low
single-digit seconds each if they work. The 23,979-MSE mode budget is only capturable by a per-row
identifier, and none has been found in this data. **That budget is the leader's edge, and the
search for the identifier continues under Amendment 13's rule: never by reading the scorer.**


---

# AMENDMENT 20 — the standard approach, end to end: one model, all rows, taxi time as the target

**2026-09-09 17:35 local. Owner, verbatim: "people are continuously moving towards 245 and you are
saying this is the ceiling ... this is seriously nonsense." Correct. Five teams are under 270 and
one is at 246 on the same data. Amendment 19.4's "mode budget" was the ceiling of THIS pipeline's
decomposition, not of the problem. Written before the arm runs. Nothing above this line has been
edited.**

## 20.1 What has never been tested

Every structural decision in this pipeline — predict the clock difference `delta` rather than taxi
time; split matched from unmatched; a hand-built mixture for the unmatched rows — was made on the
26-feature HGB months ago (delta 252.66 vs y 285.24; "a booster in place of the cell estimator
scored worse, 1793.6 vs 1648.4") and never re-tested at capacity with the 80-feature design. A
strong first submission by a competitor is, with high probability, the standard approach: **one
gradient-boosted model over all rows, taxi time as the target, every column in, missing anchors
left missing, tuned, seed-averaged, blended.** H-20: that approach, on our features, beats our
two-part pipeline on the fold.

## 20.2 Arm U (unified)

One LightGBM regressor, target `y`, trained on ALL training rows (matched and unmatched), design =
BASELINE_FEATS + QUEUE_FEATS (+ DAY_FEATS when Arm D exists), every `AOBT_3`-derived column NaN on
unmatched rows, plus `is_unmatched`; prediction `max(pred, 1)` on every row, no proxy subtraction,
no mixture. Same lr/leaves/ES procedure, seeds 0,1,2. Unmatched sample weight W in {1, 10}.
Baseline = the current pipeline's per-row predictions (matched from the v4/queue record; unmatched
from `fit_unmatched` as shipped).

## 20.3 Thresholds, locked

Reported three ways, each with a paired row bootstrap (2,000 draws, seed 0), pooled and per airport:
- **(i) matched rows:** Arm U ADOPTED for matched rows iff its interval vs the pipeline excludes
  zero in U's favour; the 0.5 blend likewise reported and adopted iff its interval excludes zero
  and it beats U alone.
- **(ii) unmatched rows:** decisional on EX-MONSTER RMSE (Amendment 9 rule 4); pooled reported with
  the monster rows named. Arm U ADOPTED for unmatched rows iff ex-monster interval excludes zero in
  U's favour AND pooled is not worse by more than 20 s.
- **(iii) fold TOTAL at the 2026 weights** — reported for scale only, never as a board prediction.
- The adoption decisions are made per subset, so the shipped pipeline may end up unified on
  matched rows and mixture on unmatched, or any combination the intervals support.
- **Point-gain bars: none.** Any paired gain with an interval excluding zero on matched rows is
  bankable by the transfer rule of RESULT 5. Below-bar decisions are the owner's, not the agent's.

## 20.4 What follows a positive result

If U beats the pipeline on matched rows, the delta formulation is retired for that subset and every
later arm (sweep, CatBoost, D) is re-run on U's design. The sweep (15.2) and the CatBoost blend
(15.1) run regardless — they are the rest of "the standard approach" and have never been run here.


---

# RESULT 9 · 2026-09-09 18:19 local · Amendment 14 (queue block) — real, below its own bar; SHIPS by owner decision

`lgbm_fold.py --queue --baseline A2 --pa-trees es`, fold A, 339,015 matched holdout rows, 4 h 31 min
(the 20-permutation control at quarter capacity took 3 h of it), peak ~5 GB. Log
`reports/lgbm_fold_queue.console.log`, JSON `reports/lgbm_fold_queue.json`, per-row predictions
`data/cache_stand/fold_preds_queue.parquet`. Treatment best_iter 21,316 → n_ref 26,647 (80 features).

| arm | matched RMSE |
|---|---|
| baseline A2 (v4 record, 3 seeds) | 225.825 |
| treatment = A2 + 12 QUEUE_FEATS (3 seeds) | **224.258** |

| clause (14.3 / 14.4 ii) | measured | passes |
|---|---|---|
| paired interval excludes zero | **+1.568 [+1.316, +1.812]** | yes |
| point gain >= +2.0 s | +1.568 | **no** |
| gain > 2 × seed sd (0.145) | 10.8× | yes |
| >= 4 of 10 airports improve | **9 / 10** (LIRF −0.74) | yes |
| tail subset (non-fill, |delta| > 20 min) improves >= +5 s | +0.37 [−2.83, +3.54] | **no — not the named mechanism** |
| control: treatment vs 20 within-airport permutations at equal columns, quarter capacity | treatment gain +1.372 vs p95 of permuted −0.190 (max −0.177; 0 at or above) | **yes, decisively** |

**Verdict by the letter: INCONCLUSIVE** — the +2.0 s point bar fails; the mechanism clause fails and
is reported as 14.3 requires: **the gain is in the body (+1.75 s), not the holding tail (+0.37 s) the
block was named for.** Queue state helps the model generally; it does not identify the mode rows.

**Owner decision (16:45 local, "work towards it"):** the +2.0 s bar was a compute-economy threshold
written before the experiment was spent; the gain is replicated on three seeds, its interval excludes
zero by a wide margin, and the permutation control is unambiguous. **The block ships in v4 together
with the established seed averaging (RESULT 7), one submission differing from v3 by those two
changes** (seeds' +0.48 s is established and small; attribution of the board delta is not materially
confounded). Expected board effect by the RESULT 5 transfer rule: about −2 s.

Per airport (treatment − baseline, * = own interval excludes zero): EDDF +3.09*, EDDM +1.95*,
EGLL +1.00*, EHAM +1.33*, LEBL +1.19*, LEMD +1.50*, LFPG +1.81*, LIRF −0.74, LSZH +2.48*, LTFM +3.28*.


---

# RESULT 10 · 2026-09-09 18:26 local · Amendment 18 (stratum hybrid) — ESTABLISHED, all three clauses

`scripts/stratum_fold.py`, 12-fold LOMO over 2025 on the 22,219 unmatched rows (+ fold A alongside),
3 seeds of `nf_fit`, month-block paired bootstrap 2,000 draws; 11 s of fitting after the 3 GB load;
peak RSS 3.0 GB. Log `reports/stratum_fold_v7.console.log`, JSON `reports/stratum_fold_v7.json`,
per-row predictions `data/cache_stand/stratum_fold_v7_preds.parquet`. Fold-A S0 reproduced R3.2.

| (12-fold LOMO) | S0 incumbent | S1 hybrid (3-seed mean) | gain | month-block 95% |
|---|---|---|---|---|
| **ex-monster stratum RMSE** (decisional) | — | — | **+33.17 s** | **[+27.70, +40.22]** |
| pooled stratum RMSE | — | — | **+16.20 s** | [+9.73, +24.53] |

Seed sd (ex-monster, 3 seeds): 0.729 s → gain is 45 × seed sd. Per seed: +32.0 / +32.8 / +33.5.

**Clauses (18.3):** (a) ex-monster interval excludes zero, gain >= 20 s, > 2 × seed sd — **holds**;
(b) pooled not worse (it improves by 16 s; the tail load is preserved, the hybrid routing works) —
**holds**; (c) airports improving ex-monster: **10 of 10** — holds. Per airport ex-monster gain:
EDDF +42.7, EDDM +15.6, EGLL +33.5, EHAM +63.8, LEBL +39.8, LEMD +22.3, LFPG +51.5, LIRF +9.6,
LSZH +70.4, LTFM +62.1. Per month: 12 of 12 improve on both metrics.

**VERDICT: ESTABLISHED.** Ships as its own version, differing from its predecessor only on the
5,290 unmatched rows (`build_submission.py --hybrid --base <prev> --version N`). Board effect is not
projected (Amendment 9: stratum totals do not transfer; the board is the instrument). The LOMO
pooled gain is ~16 s of stratum RMSE ≈ ~800 MSE at the 2026 weight if it transferred 1:1 — stated
for scale only.

Sequencing decision: this ships FIRST as v4 (5 minutes, no matched refit), and RESULT 9's
seeds + queue block ships as v5 on top of it (~2 h fit), so both lanes are on the board tonight
and each board delta attributes to one lane.


### R10 board · 2026-09-09 18:35 local · v4 = 289.6732, rank 19 of 92

v4 (v3 + hybrid stratum; matched rows byte-identical to v3) scored **289.6732**: **−1.96 s / −1,138
MSE vs v3's 291.6317**, from the 5,290 unmatched rows alone. The LOMO pooled stratum gain of +16.2 s
would be ~800 MSE at a 1:1 transfer; realised 1,138 — the stratum lane transferred at ~1.4×, the
opposite sign of Amendment 9's warning about fold TOTALS (which remain untrustworthy; this is a
paired relative gain, like RESULT 5's). First board improvement from the stratum in this project.


### R9 board · 2026-09-09 20:45 local · v5 = 288.1406, rank 19 of 94

v5 (v4 + RESULT 7's seed averaging + RESULT 9's queue block on the 339,551 matched rows; the 5,290
unmatched rows byte-identical to v4, verified before upload) scored **288.1406**: **-1.53 s /
-886 MSE vs v4's 289.6732.** Fold projection at a 1:1 transfer of the matched-row MSE gains
(queue 705 + seeds 219 matched MSE, × w_m 0.98466) = -910 MSE;
realised -886 → transfer **0.97×**. Third consecutive matched-side fold gain to land
on the board at ~1:1 in MSE (RESULT 5: 1.17×; this: 0.97×). Board at 20:45: 94 teams, leader 245.29,
10th **276.51** (278.21 at 13:55, 277.90 at 20:33 — the cut is falling ~1.5 s per day today).
Fit: `lgbm_submit.py --version 5 --seeds 0,1,2 --queue --base v4`, 2 h 15 min, peak RSS 5.07 GB,
best_iter 24,517 → n_ref 29,439 per seed, 80 features; log `reports/lgbm_submit_v5.log`, meta
`submissions/merry-quicksand_v5.meta.json`. Matched-row shift vs v4: mean +0.6 s, RMS 39.0 s.

Standing after v5: **288.14 − 276.51 = 11.6 s = ~6,400 MSE to the current 10th place**, more by the
freeze. Amendments 19 (D, Y) and 20 (U) are the open arms; the sweep (15.2) and CatBoost (15.1) run
regardless. Next free version: v6.


### 20.5 · 2026-09-09 21:05 local · BEFORE any Arm U fold result exists · the all-rows arm early-stops on MATCHED rows

Observed from the running fold's log at 20:55 (no result had been produced): `U_w1`'s early-stopping
run stopped at best_iter **1,354** (n_ref 1,692) where the delta arm on the same months stops at
24,517; its stopping-set RMSE was 287.36 over **348,353 rows of which 3,513 are unmatched**. The
stopping metric is plain RMSE on y over all stopping rows (20.2: "same ES procedure"), and the
unmatched rows' squared errors — monsters of up to 24 h, bet variance by RESULTS 3–10 — are of the
order of the matched rows' entire MSE. The stop is therefore decided by noise on ~1% of the rows,
and the unified model is left with ~1/15 of the delta model's capacity on the 99% that carry the
board. As registered, Arm U measures the stopping rule's failure, not the formulation.

**Amendment.** In the all-rows arm the early-stopping metric is evaluated on the **matched rows of
the stopping months only** (`es & ~is_unmatched`); the fit rows, the unmatched sample weight W, the
refit rule `n_ref = best_iter × n_train / n_fit`, lr, leaves, patience and seeds are unchanged. This
is the competition's own weighting to within 1.5% and is exactly what a competitor doing the
standard approach obtains by holding out matched rows for validation. Everything in 20.3 (clauses,
subsets, bootstrap, no point-gain bars) is unchanged.

**Reporting.** The run already in flight (ES on all rows, the 20.2 reading) is allowed to finish
and is reported as "U-20.2" for the record; the decisional run is the re-run under this amendment,
"U-20.5". If U-20.2 nevertheless beats the pipeline on matched rows, that stands on its own
interval. The harness change ships with tests that (i) the stopping Dataset in the all-rows arm has
exactly the matched stopping rows and (ii) the default (matched-only designs) path is unchanged.


---

# AMENDMENT 21 — the mode rows' identifier: is AOBT_3's derivation fingerprinted in the row?

**2026-09-09 21:10 local, written BEFORE the screen runs.** Owner: "anywhere near top 10" — not with
the measured lanes (~281–283 at the freeze against a cut heading for ~270). The one mechanism that
squares with a 245 on this data is a per-row identifier of the mode rows (19.0: |BLOCK − AOBT_3| >
10 min, 7.6% of matched rows, 47.8% of matched error; perfect knowledge → matched 163 → total ≈ 244.6).

## 21.1 Hypothesis H-21

`AOBT_3_flt` is the Network Manager's actual off-block time. When no airport message carries an
actual, NM derives one arithmetically (the take-off time minus a standard taxi time, or a flight-plan
off-block time carried forward). Such a derived `AOBT_3` disagrees with the airport's `BLOCK_TIME`
by construction, and it leaves an exact fingerprint in the row it came from: `MVT_TIME − AOBT_3` a
whole number of minutes (or one of a few standard values per airport/runway), or `AOBT_3` equal to
`EOBT_1`, `LOBT`, `IOBT` or `SCHED` to the second. H-21: **a fingerprint computable at serve time from
the scored row alone (MVT_TIME, AOBT_3, EOBT_1, LOBT, IOBT, SCHED are all present on matched scored
rows) identifies the mode rows.**

## 21.2 Screen (ranking only; no bankable number comes from it)

Raw training months 1 and 7 (the fold's holdout months), departures at the ten airports with
`AOBT_3` and `TAXITIME` present; `delta = BLOCK − AOBT_3`; mode := |delta| > 600 s. Indicators:
(a) `(MVT − AOBT_3) mod 60 == 0`; (b) `MVT − AOBT_3` equal to one of the five most frequent exact
values at the airport; (c) `AOBT_3 == EOBT_1`; (d) `AOBT_3 == LOBT`; (e) `AOBT_3 == IOBT`;
(f) `AOBT_3 == SCHED`; (g) `AOBT_3.second == 0` (already `aobt_sec` in the design; the control);
(h) `MVT.second == 0`. For each: prevalence, mode rate inside vs outside, share of mode rows
covered, and the AUC of the best single indicator and of a 2-fold logistic on all of them, per
airport and pooled.

## 21.3 TRUE / FALSE shapes, locked

- **TRUE:** some indicator, or the logistic, reaches mode-row AUC **>= 0.75 pooled** with the
  indicator's mode rate >= 3× the base rate and covering >= 40% of mode rows. Then the fingerprint
  block enters the design as Arm F (fold A, paired interval, same harness as Amendment 14, with the
  permutation control) — the first mode-row lane with a mechanism.
- **FALSE:** AUC < 0.65 (the day-regime level of 19.0) or coverage < 20%. Then H-21 is NOT WORKING
  and the fingerprints are not the identifier; the search moves to the arrivals' same-day clock
  disagreement (the second candidate named to the owner at 21:08).
- Between 0.65 and 0.75: INCONCLUSIVE, reported, and Arm F still runs on the fold because the cost is
  one fold run.


---

# RESULT 11 · 2026-09-09 21:12 local · Amendment 21 (AOBT_3 derivation fingerprints) — NOT WORKING

Raw months 1 + 7, 339,046 matched departures, 25,898 mode rows (7.64%). Every fingerprint is at
chance: (a) whole-minute proxy AUC 0.507 (prevalence 5.2%, lift 1.3×, covers 6.4% of mode rows);
(b) top-5 proxy value at the airport 0.494; (c) AOBT_3 == EOBT_1 0.483 (lift 0.3×: rows with a
carried-forward off-block are LESS often mode rows); (d)(e) == LOBT / IOBT 0.486; (f) == SCHED
0.488; (g) AOBT_3 second == 0 — 97.6% of ALL rows, so AOBT_3 is a minute-precision stamp
everywhere, not on the mode rows; (h) MVT second == 0 0.507; (j) whole-5-minute proxy 0.507
(lift 1.9×, covers 3.0%). Two-fold logistic on all indicators: **0.532**; with |proxy| bins 0.619 —
the day-regime level of 19.0 again. Per airport 0.50–0.59 (EDDM highest). 21.3 FALSE shape met.

**Verdict: NOT WORKING.** The mode rows are not an arithmetically derived AOBT_3; the row's own
timestamps carry no per-row identifier of which clock BLOCK_TIME follows. Mode rates by airport
(EDDF 1.9%, LEMD 2.9%, LSZH 3.1%, LEBL 5.2%, EHAM 6.1%, LFPG 7.7%, EGLL 7.9%, EDDM 8.4%, LTFM
15.5%, LIRF 15.7%) say the disagreement is an airport-integration property, not a row property.
Screen log: scratchpad `amend21_fingerprints.log`; script `amend21_fingerprints.py`.

### 21.4 · second candidate, written BEFORE its screen · the arrival side and the same stand

Arrivals in the scored file carry BLOCK_TIME (on-block) and MVT_TIME (landing) from the airport and
ARVT_3 from NM — two sources for one event, both visible at serve time. Two readings:
- **21.4a day regime:** per airport-day, the arrivals' disagreement rate |ARVT_3 − MVT| > 600 s vs
  the departures' mode rate. Reported as the day-level correlation and the within-day AUC of the
  day rate. TRUE: day corr >= 0.85 AND within-day AUC >= 0.70 (better than 19.0's proxy p90).
  FALSE: within-day AUC <= 0.63 — regime again.
- **21.4b per row:** the previous arrival on the SAME STAND within 6 h before the departure's
  AOBT_3 — its |ARVT_3 − MVT| disagreement, its |BLOCK − ARVT_3| gap, and its fill flag (BLOCK ==
  SCHED) — as per-row features of the departure. TRUE: AUC >= 0.70 pooled, or >= 0.75 at LIRF/LTFM.
  FALSE: AUC < 0.63.
Ranking screen on raw months 1 + 7 as in 21.2; nothing bankable comes from it.


### RESULT 11.4 · 2026-09-09 21:15 local · the arrival side — NOT WORKING

Raw months 1 + 7; 343,999 arrivals (99.2% with ARVT_3), 339,046 matched departures. **There is no
clock disagreement to read on the arrival side:** |ARVT_3 − MVT| is 16 s at the median, 64 s at p90,
> 600 s on 0.10% of arrivals; arrival fills (BLOCK == SCHED) 0.18%. (a) Day level: corr(arrival
disagreement rate, departure mode rate) −0.18..+0.43 by airport (LIRF +0.43 highest) against the
reference proxy-p90 correlation of +0.10..+0.79; within-day AUC of the day's arrival disagreement
rate 0.560 pooled, of the day's arrival taxi-in p90 0.620 (≈ the reference 0.591 — regime again).
(b) Per row, the previous arrival on the same stand within 6 h (81.1% of departures joined): its
disagreement AUC 0.436, its taxi-in 0.573, its fill flag 0.501; a previous arrival disagreeing by
> 600 s (n = 240) lifts the mode rate 1.66× and covers 0.1% of mode rows. Two-fold logistic on all
of it plus the day rates: **0.613** vs 0.609 for the day rates alone. Both 21.4 FALSE shapes met.

**Standing after Amendment 21:** the mode rows have no per-row identifier in the row's own
timestamps, in the arrival side, or in the stand's previous occupant. Together with 19.0 (day regime,
AUC ≤ 0.63 within day), RESULT 8 (fill head), RESULT 4 (arrival regime), the turnaround and
flight-number screens and FLIGHT_ID (0.55): every observable in this data that could name the mode
has been screened and none does. The mode rates by airport (1.9%–15.7%) are an airport-integration
property. What remains is formulation (Arm U-20.5, in flight), capacity (sweep), a second learner
(CatBoost), regime (D, Y) — worth ~2–4k MSE together on the measured record.
Screen: scratchpad `amend21_4_arrivals.py` / `.log`.


---

# RESULT 12 · 2026-09-09 21:16 local · Arm U as registered (U-20.2, stopping set = all rows) — NOT WORKING; superseded by 20.5

`lgbm_fold.py --ytarget --all-rows --unmatched-weight 1,10 --queue --baseline A2 --pa-trees es` at
commit 4f376ca (stopping metric on all 348,353 stopping rows), fold A, 23 min, peak 4.06 GB; outputs
renamed `reports/lgbm_fold_queue_allrows_20_2.{json,log,console.log}`,
`data/cache_stand/fold_preds_queue_allrows_20_2.parquet`. Early stop: w1 best_iter 1,354 → n_ref
1,692; w10 1,047 → 1,309 (the delta arm: 21,316 → 26,647). Baseline = the queue record (224.258).

| subset | pipeline | U_w1 | Ublend_w1 | U_w10 | Ublend_w10 |
|---|---|---|---|---|---|
| matched (n 339,015) | 224.258 | 268.576 (−44.3 [−108.4, −5.4]) | 235.804 (−11.5 [−31.8, −0.5]) | 283.864 | 240.485 |
| unmatched ex-monster (5,265) | 995.60 | 1223.66 (−228 [−448, −29]) | 1017.63 (−22 [−120, +68]) | 1157.18 | 990.83 (+4.8 [−75, +77]) |
| unmatched pooled (5,321) | 1867.9 | 2867.2 (−999) | 2126.9 (−259) | 2900.2 | 2142.5 |

No clause of 20.3 holds for any arm: 0 of 10 airports improve on matched rows for U alone; LIRF
384 → 634 (the under-fitted y-model carries the unmatched monsters into matched LIRF rows; the
2–10 min band's interval [−190, −5] is a handful of monster-sized predictions). **NOT WORKING as
registered, and not decisional** — the stop was decided by the unmatched rows' noise (20.5). The
decisional run, U-20.5 (stopping metric on the matched stopping rows), launches next on 427d82c and
takes the canonical output names.


### 21.5 · third candidate, written BEFORE its screen · record-ordering fingerprints

`FLIGHT_ID_mvt` is a perfectly time-ordered key (Spearman 1.000 with MVT_TIME at every airport) —
no ordering signal can live in it. `MVT_ID_mvt` is a sequential key at EDDF/EGLL/LIRF (Spearman
0.92–0.96 with MVT_TIME), partly at LTFM (0.45), not at LFPG (0.00): where it is sequential, a record
inserted out of time order (a late or manual entry, a batch import) sits off the id-vs-time line.
H-21.5: the deviation of MVT_ID from its airport's id-vs-time line identifies (a) the matched mode
rows and/or (b) the unmatched fills (BLOCK == SCHED; the stratum's 228k-MSE stake). Screen on raw
months 1 + 7: per airport, rows sorted by MVT_TIME, deviation = id minus the rolling median id of the
±250 neighbouring rows, normalised by that window's id spread; AUC of |deviation| and of the signed
deviation for (a) and (b), per airport and pooled; the same for FLIGHT_ID's deviation as a control.
TRUE / FALSE shapes as 21.3 (AUC >= 0.75 with lift >= 3× and coverage >= 40% → Arm F; < 0.65 →
NOT WORKING). Computable at serve time: MVT_ID and MVT_TIME are on every scored row.


### RESULT 11.5 · 2026-09-09 21:32 local · record-ordering fingerprints — NOT WORKING

Raw months 1 + 7. (a) Mode rows: MVT_ID's deviation from its airport-month id-vs-time line, AUC
|dev| **0.532** (per airport 0.50–0.57; the 419 rows with |dev| > 2 have a 30% mode rate, 4× the
base, covering 0.5% of mode rows); FLIGHT_ID's deviation **0.587** (EDDF 0.68, EGLL 0.66, EHAM 0.64,
LEMD 0.63, the rest 0.53–0.58) — a weak signal at four airports, below the 0.65 FALSE line pooled and
below the 0.75 TRUE line everywhere; signed deviations 0.45–0.49. (b) Unmatched fills (fill :=
|BLOCK − SCHED| < 60 s; 253 of 5,373 unmatched departures, LIRF 34%, LTFM 9%, LSZH 5%, EHAM 3%):
MVT_ID deviation AUC **0.436** pooled, LIRF 0.483, the 383 monsters (sp > 3 h, fill rate 12%) 0.454.
The first run of (b) used exact equality (22 fills) and is void; this is the corrected measurement.
FLIGHT_ID is null on the unmatched rows (they have no NM record), so it cannot fingerprint the stratum.

**Verdict: NOT WORKING.** Screens: scratchpad `amend21_5_ordering.py/.log`, `amend21_5b_fills.log`.

**Standing after 21.5:** timestamps' arithmetic (21), the arrival side and the stand's previous
occupant (21.4), record ordering (21.5) — all at chance or at the regime level. FLIGHT_ID deviation at
0.63–0.68 on four airports is the only per-row signal above 0.6 found tonight; it is recorded for a
possible Arm F on those airports if nothing better appears, not as an identifier.


---

# AMENDMENT 22 — Arm F: the record-ordering block (FLIGHT_ID / MVT_ID off the airport's id-vs-time line)

**2026-09-09 22:20 local, written BEFORE the block is built or run.** RESULT 11.5's one per-row
signal above 0.6: FLIGHT_ID's deviation from its airport-month id-vs-time line separates the mode
rows at AUC 0.68 (EDDF), 0.66 (EGLL), 0.64 (EHAM), 0.63 (LEMD), 0.53–0.58 elsewhere; MVT_ID's
deviation 0.50–0.57 with a 4× lift on its 419 largest deviations. Weak, per-row, computable at serve
time from the scored file alone (MVT_ID, FLIGHT_ID, MVT_TIME are on every scored row; the line is
transductive within the file, as Amendment 19.1 permits for the day block). Built under the owner's
"build everything measurable"; expected worth: low single-digit seconds at most.

## 22.1 The block (`ORDER_FEATS`, `data/cache_order/`)

Per raw file and airport, over EVERY departure at the airport with a finite MVT_TIME (matched or
not, labelled or not — the same reference stream in both modes), rows sorted by MVT_TIME then
MVT_ID: `o_dev_mvt` = (MVT_ID − rolling median over the ±250 neighbouring rows) / max(rolling p90 −
p10, 1); `o_dev_flt` the same for FLIGHT_ID, NaN where FLIGHT_ID is null; `o_n_line` the window's
row count (50–501). Each feature row (the stand cache's departure stream, take-off order) takes its
own three values. Serve mode: per calendar month of the scored file, as build_day_ranking.

## 22.2 Harness

`lgbm_fold.py --orderfeats --queue --baseline A2 --pa-trees es` (mode `queue_order`: design = A2 +
QUEUE_FEATS + ORDER_FEATS, 83 columns), baseline = the queue record; `lgbm_submit.py --orderfeats`
for the submission. Same lr/leaves/ES/seeds; paired row bootstrap 2,000 draws; the |delta| bands
of 19.0 and the per-airport table reported.

## 22.3 Thresholds, locked

ESTABLISHED iff (i) the paired interval vs the queue record excludes zero in F's favour, (ii) point
gain >= +1.0 s, (iii) gain > 2 × seed sd, (iv) the over-10-min band's own interval excludes zero
(the mechanism: the block must move the mode rows, not the body). NOT WORKING iff (i) fails.
(ii)–(iv) failing with (i) holding: INCONCLUSIVE, the owner decides (RESULT 9's precedent).


---

# AMENDMENT 23 — the lane that decides it: the 5,290 unmatched rows carry 40% of the board score

**2026-09-09 22:45 local, written BEFORE the screen.** Owner: "the top candidate is doing something
no one is even thinking about — it is now 245."

## 23.0 The arithmetic that reframes the problem (measured, not assumed)

At the 2026 weights (w_m 0.984660 / w_u 0.015340) and v5's board 288.1406 = 83,025 MSE, with the
shipped matched RMSE of 224.258 (the queue fold's treatment):

| | contribution | share |
|---|---|---|
| 339,551 matched rows | 49,520 MSE | 59.6% |
| **5,290 unmatched rows** | **33,505 MSE** | **40.4%** (implied RMSE 1,478 s) |

Holding matched fixed, **10th place (276.51) needs unmatched RMSE 1,478 → 1,325 s** (−11% on that
lane); the leader's 245.29 needs 833 s. Holding unmatched fixed, matched would have to reach
**164.6 s** — exactly Amendment 19.0's perfect-mode-knowledge bound, i.e. unreachable without an
oracle. **One 24-hour row predicted at 1,000 s costs 40 s of board RMSE.** The lane that decides
this competition is the 1.5% of rows the field treats as noise, not the 98.5% everyone tunes.

Fold record (`stratum_fold_v7_preds.parquet`, 5,321 fold-A unmatched rows, scaled to 5,290 at the
2026 weight): S0 53,524 MSE → S1 (shipped) 52,830 → **ORACLE fill decision 39,333**. The fill
decision alone is **13,497 MSE of headroom, 26% of the lane**; 5,809 of it sits in 158 fills hedged
below p = 0.5 and 6,030 in 43 non-fills hedged above it. Two LFPG rows (y 84,240 s and 58,206 s at
sp 1,740 / 2,043 s — the airport's off-block stamp is ~23 h stale) carry 29,502 MSE and are not
fills: they are the bet-variance term, unreachable without identification.

## 23.1 Hypothesis H-23 — the schedule-only record fingerprint

A row that is BOTH unmatched (no NM `AOBT_3`) AND filled (`BLOCK == SCHED`) has no actual off-block
observation anywhere: its movement record was created from schedule data. H-23: **such a record is
incomplete in OTHER observable fields too** — a missing or generic `STAND_mvt` / `RUNWAY_mvt`, a
null `FLIGHT_ID`, a missing `EOBT_1`/`LOBT`/`IOBT`, an `ADES_FILED` differing from `ADES` — and that
completeness pattern identifies fills per row at serve time, where sp bands and the L-e logistic
(AUC 0.877 at LIRF) currently leave 13,497 MSE on the table.

## 23.2 Screen (ranking only, raw months 1 + 7, unmatched departures at the ten airports)

Per candidate indicator: prevalence, fill rate inside vs outside, lift, coverage, AUC — pooled, at
LIRF, and on the sp > 3 h subset where the decision is worth hours. Then a 2-fold logistic on the
completeness block alone and on completeness + the shipped model's inputs (log sp, dayoff, airport),
reported as AUC and as the MSE the mixture would carry with that p on the fold-A rows.

## 23.3 TRUE / FALSE shapes, locked

- **TRUE:** the completeness block reaches **fill AUC >= 0.80 pooled** (against the shipped cell/L-e
  model's own AUC on the same rows, computed alongside as the reference) OR cuts the fold-A
  mixture's MSE by **>= 3,000 MSE** at the 2026 weight with p from a 2-fold fit. Then it becomes
  Amendment 24's arm: the block enters `_lirf_fill_model` and the hierarchical cells, measured on
  `stratum_fold.py`'s 12-fold LOMO with a month-block paired interval, exactly as Amendment 18 was.
- **FALSE:** AUC below the shipped model's on the same rows AND fold-A MSE gain < 1,000. Then
  record-completeness is NOT the identifier and the lane's remaining headroom is bet variance, to be
  stated as such.
- Between: INCONCLUSIVE, reported with the numbers, and the arm still runs (one fold run is cheap
  against a 13,497-MSE headroom).
- No point-gain bar: any paired gain on this lane is bankable by RESULT 10's precedent (the stratum
  transferred at ~1.4x).


---

# RESULT 13 · 2026-09-09 22:55 local · Amendment 23 — H-23 NOT WORKING; the unmatched lane is priced and closed

**H-23 (the schedule-only record fingerprint): NOT WORKING, and it cannot work.** An unmatched row
has no flight-plan record at all: `FLIGHT_ID` is null on **99.87%** of unmatched departures and every
`_flt` column with it, so the completeness pattern is constant and has nothing to separate. Every
indicator returned AUC 0.49-0.50; the completeness block as a 2-fold logistic scores **0.489**
against the fill label. 23.3's FALSE shape is met.

**The shipped `p` is better than every alternative fitted against it** (fold-A rows, out of fold on
both sides): cells + the L-e logistic **0.944** pooled, **0.984** on sp > 3 h, 0.859 at LIRF, against
a continuous logistic on (log sp, dayoff, airport) at 0.913 / 0.917 / 0.420 and a per-airport-slope
version at 0.925 / 0.914 / 0.459. The cell discretization is not leaving the headroom on the table;
the mixture's MSE with the shipped p is 53,546 against the logistic's 76,840. Reliability in the
sp > 3 h top decile: predicted 0.66 vs realised 0.60.

**Where the 13,497-MSE oracle headroom actually is:** ALL of it at LIRF (17,690 → 3,520 MSE);
ex-LIRF the oracle fill decision saves **42 MSE** of 35,856. LIRF's fill identification is the lane
four probes have already closed (Amendment 11, the turnaround screen, the oracle bound, RESULT 8),
with the shipped model already at AUC 0.859 there.

**The rest of the lane is four rows.** Unmatched departures with y > 3 h that are NOT fills and whose
sp is under 3 h - the stale-`BLOCK` artifacts - number **4 in two months**, on 4 distinct
airport-days, at 3 airports (LFPG ×2, EHAM, EGLL), each the only such row on its day, with no burst,
no day-level signature and no shared offset (BLOCK precedes SCHED by 0.17 / 3.17 / 15.6 / 22.9 h).
**Zero of the 339,046 matched rows show the shape** - an NM anchor precludes it. Two of them (LFPG,
y 84,240 s and 58,206 s) carry **29,502 MSE = 56% of the fold's whole unmatched lane**.

**The insurance arithmetic, priced before anyone proposes it.** Raising a scored row's prediction
from 1,000 s to 40,000 s costs (39,000)²/344,841 = **4,411 MSE if the row is ordinary** and saves
**14,400 MSE if it is a 23-hour monster**: the bet pays only when P(monster) > **23%**. The measured
base rate on unmatched rows is 4/5,373 = **0.07%** and no feature moves it. Deliberate over-prediction
of the tail is a losing bet by a factor of ~300 and is refused.

**Verdict.** The unmatched lane holds 40% of the board score, 26% of it is oracle-reachable, and
every reachable part is LIRF fill identification - closed. The remainder is a draw of one-in-a-million
rows. **The 23,000 MSE that separates us from the leader is not in this lane**, which leaves the
matched mode rows (Amendment 19.0) and, by 23.0's arithmetic, a leader whose matched RMSE is near
165 s. Screens: scratchpad `amend23_unmatched_anatomy.log`, `amend23_screen.log`,
`amend23_p_quality.log`, `amend23_monsters.log`.


---

# RESULT 14 · 2026-09-09 23:10 local · the airport-clock anchor and the irreducible bounds

## 14a · CORRECTION to 23.0's split — it rests on an assumption, and I stated it as measured

23.0 divided the board score 59.6% matched / 40.4% unmatched **by assuming the scored file's
matched rows behave like the fold's (224.26 s)**. That assumption is not measurable from the board:
only the TOTAL is observed. Across a plausible matched range the split moves:

| assumed scored matched RMSE | matched share | implied unmatched RMSE |
|---|---|---|
| 200 s | 47.4% | 1,686 s |
| 224.26 s (the fold) | 59.6% | 1,478 s |
| 250 s | 74.1% | 1,183 s |

What survives: the unmatched rows are a **large minority of the score** (26-53%) at 1.5% of the
rows, and RESULT 13's oracle decomposition — which was measured on the fold record, not inferred
from the board — is unaffected. The precise "40%" is withdrawn; the lane's ranking is not.

## 14b · The turnaround anchor (airport clock at both ends) — NOT WORKING

The scored file carries **344,693 ARR rows with `BLOCK_TIME` 100% populated**, so the airport's own
on-block stamp is observable at serve time for the aircraft that occupied each stand. That admits an
anchor the field does not use: instead of `y = proxy - delta` (anchored on NM's `AOBT_3`, the very
clock that disagrees on the mode rows), anchor on the airport's previous stamp at the same stand —
`y = MVT_dep - (BLOCK_arr + turn)` — where a systematic clock offset cancels because both stamps
come from one system. 97.6% of matched departures link to a previous arrival at their stand
within 24 h (330,843 rows).

**It is decisively worse.** `turn = BLOCK_dep - BLOCK_arr` has sd **13,503 s** against `delta`'s
**427 s**; within (airport, stand, runway, hour) the floors are **9,060 s** vs **380 s**, and on the
mode rows 8,497 s vs 939 s. Ground time is hours-scale and multimodal (aircraft sit overnight), so
it swamps any clock offset it removes. The NM anchor is not merely conventional, it is the right
one. **NOT WORKING; closed.**

## 14c · The irreducible bounds, measured (pooled within-group spread, dof per group)

A model keyed only on the named columns cannot beat the within-group spread of `delta`
(= the spread of `y`, since `proxy` is known exactly):

| key | within-group RMSE(delta) |
|---|---|
| airport | 408.8 s |
| airport + stand + runway | 392.2 s |
| + hour | 380.5 s |
| airport + hour + proxy band + delay band | 311.1 s |
| airport + stand + runway + hour + proxy + delay bands | 250.2 s |
| **body only** (excl. mode rows), airport+stand+runway+hour | **197.3 s** |
| **mode rows only**, same key | **938.7 s** |

Our shipped 224.3 s already beats every coarse-key bound (the 80-feature design and the proxy carry
more than any of these keys), so these are floors for coarser models, not for ours. The one that
binds: **mode rows have a 939 s within-group spread even at stand+runway+hour resolution.** With the
mode rows predicted perfectly and the body at its own coarse bound, matched lands at 166.0 s —
within 1.4 s of the leader's implied matched RMSE under 23.0's assumption. Screens:
scratchpad `turn_anchor_screen.log`, `irreducible_delta_bound.log`.


---

# AMENDMENT 24 — weather at the airport-hour, aimed at the mode rows

**2026-09-09 23:20 local. Written BEFORE any weather observation has been downloaded for a
training month and before any weather-derived number exists.** Nothing above this line is edited.

## 24.0 Why now, when weather already returned +0.49 s

The step-6 weather test (README, "negative results") measured **+0.49 s pooled on taxi time**,
on the 26-feature HGB, at low capacity, before the queue block existed and before Amendment 19.0
identified where the error lives. Three things have changed:

1. **The target is now named.** 19.0: 7.63% of matched rows carry 47.8% of matched error, and
   **78% of them are "pushed back early then held"** — the airport stamps off-block at pushback,
   NM stamps `AOBT_3` at the real movement. The two physical causes of a hold are **de-icing**
   (winter) and **ATFM/queue** (summer). January is half the scored file.
2. **The queue half is now measured** (RESULT 9, +1.57 s, established mechanism in the body not
   the tail). Weather is the other half of the same mechanism and has never been tested beside it.
3. **Everything else is closed.** RESULTS 11, 11.4, 11.5, 13, 14 exhaust the file's own columns;
   ADS-B is closed on sensor coverage (`reports/ADSB_GATE.md`: 4 of 10 airports have zero
   on-ground samples, whole-lane value 2,243 MSE). Weather is the last unexhausted observable, and
   unlike ADS-B it covers **all ten airports in both scored months**.

## 24.1 Hypothesis H-24

> Surface weather at the airport in the hour of a departure's pushback — freezing conditions,
> precipitation, low visibility, thunderstorms, wind — reduces held-out matched RMSE beyond the
> queue design, **and does so on the |delta| > 10 min rows**, because it names the physical cause
> of the hold that separates the two clocks.

## 24.2 The data, and its provenance recorded before use

Iowa Environmental Mesonet ASOS/METAR archive (`mesonet.agron.iastate.edu`), **public domain**,
routine + special reports, for the ten scored airports, for every month the caches use (2025-01 to
2025-12) and both scored months (2026-01, 2026-07). Archived to `data/weather/` **before use** and
never re-fetched for a run: the 2024 winner was bitten by upstream back-correction. No licence
restriction applies and none is claimed; the fetch script and the exact query go in the repo.

## 24.3 The block (`WEATHER_FEATS`, `data/cache_weather/`)

Per row, from the observation valid **at or before the row's own pushback anchor** (`AOBT_3` for
matched rows; `MVT_TIME` minus the airport's median proxy for rows without one), never after it:
temperature °C, dewpoint spread, visibility, wind speed, wind gust, one-hour precipitation, and
the derived flags `w_freezing` (temp <= 3 °C and precipitation > 0, or a wx code containing FZ/SN/
PL/GS), `w_lowvis` (visibility < 1,500 m), `w_thunder` (TS in the code), plus the observation's age
in seconds. Missing observations are NaN, never imputed, and the NaN rate is asserted against the
twelve-month envelope exactly as the unmatched block's is.

## 24.4 Harness and thresholds, locked

`lgbm_fold.py --weatherfeats --queue --baseline A2 --pa-trees es`, baseline = the queue record,
paired row bootstrap 2,000 draws seed 0, per airport and on 19.0's |delta| bands.

- **ESTABLISHED** iff (i) the paired interval excludes zero in the block's favour, (ii) point gain
  >= **+1.0 s**, (iii) gain > 2 x seed sd, AND (iv) **the over-10-min band's own interval excludes
  zero** — the mechanism clause: weather must move the held rows, not the body.
- **NOT WORKING** iff (i) fails.
- (i) holding with (iv) failing is reported as INCONCLUSIVE with both numbers, exactly as RESULT 9
  was, and the ship decision is the owner's.
- Reported additionally: January vs July separately (the de-icing season against the convective
  one), because a pooled number would hide a mechanism that only exists in one of them.


---

# RESULT 15 · 2026-09-10 00:16 local · Arm U under 20.5 — NOT WORKING; the standard approach loses

`lgbm_fold.py --ytarget --all-rows --unmatched-weight 1,10 --queue --baseline A2 --pa-trees es` at
commit 427d82c/8d6d2ca with the 20.5 stopping rule (the metric on the MATCHED stopping rows only),
fold A, 2 h 58 min, peak 4.95 GB. Records `reports/lgbm_fold_queue_allrows.{json,log,console.log}`,
predictions `data/cache_stand/fold_preds_queue_allrows.parquet`. Baseline = the queue record
(matched 224.258; unmatched S0 of the stratum record). **The amendment worked as intended:** w1
stopped at best_iter 7,375 → n_ref 9,220 and w10 at 23,621 → 29,532, against U-20.2's 1,354 and
1,047, and every arm improved against U-20.2 (matched 268.6 → 263.5 at w1, 283.9 → 273.5 at w10).

| subset | pipeline | U_w1 | Ublend_w1 | U_w10 | Ublend_w10 |
|---|---|---|---|---|---|
| matched (n 339,015) | **224.258** | 263.500 | 234.154 | 273.550 | 237.036 |
| unmatched ex-monster (5,265) | **995.60** | 1324.32 | 1048.66 | 1221.81 | 1010.88 |
| unmatched pooled (5,321) | **1867.9** | 2965.0 | 2161.9 | 2946.9 | 2159.6 |

**No clause of 20.3 holds for any arm.** Matched: no interval excludes zero in the arm's favour
(U_w1 −39.24 [−101.65, −1.38]; the 0.5 blend −9.90 [−29.43, +0.66]); 0–5 of 10 airports improve.
Unmatched: ex-monster and pooled both decisively worse. LIRF is again the discriminator —
384.07 → 622.84 (U_w1) and 665.94 (U_w10) — the unified model carries the stratum's monsters into
matched LIRF rows.

**VERDICT: NOT WORKING.** H-20 is refuted: the standard approach — one gradient-boosted model over
all rows with taxi time as the target — does NOT beat this project's two-part pipeline on this
design, at capacity, with the stopping rule corrected in its favour. **The delta formulation and the
matched/unmatched split are both retained**, and 20.4's "re-run every later arm on U's design" is
NOT triggered. The sweep (15.2) and the CatBoost blend (15.1) still run: they are the rest of
"the standard approach" and are independent of this result.

### 15a · the one signal inside a refuted arm, recorded for Amendment 25

The blend's per-band table inverts with |delta|, and the two intervals that exclude zero do so in
**opposite directions**:

| band | n | share of SSE | Ublend_w1 gain vs pipeline |
|---|---|---|---|
| < 2 min | 127,466 | 13.7% | +0.21 [−0.03, +0.44] |
| 2–10 min | 185,229 | 37.8% | **−22.23 [−60.33, −0.21]** |
| 10–20 min | 20,667 | 17.4% | +0.45 [−0.20, +1.11] |
| **> 20 min** | 5,653 | 31.1% | **+6.98 [+2.10, +11.53]** |
| **over10 (10–20 + >20)** | 26,320 | 48.5% | **+2.83 [+1.00, +4.51]** |

The y-target model is **better than the pipeline on the rows where the clocks disagree most** and
worse on the body — mechanically what Amendment 19.2 predicted, because the `proxy` anchor is
exactly what fails on a held row. A blend applied everywhere pays for the tail gain out of the body.
A blend applied only where the pipeline's OWN predicted |delta_hat| is large would not — and
`delta_hat` is observable at serve time. That is Amendment 25's hypothesis; it costs no new fit,
only a rule over two prediction columns that both records already contain.


---

# AMENDMENT 25 — the gated blend: use the y-target model only where the anchor is failing

**2026-09-10 00:25 local. Written BEFORE the rule is evaluated on any row.** Nothing above this
line is edited. RESULT 15 refuted Arm U pooled but left one measured, mechanism-shaped fact: the
0.5 blend of the y-target model with the pipeline **gains +6.98 s [+2.10, +11.53] on the |delta| >
20 min band and +2.83 s [+1.00, +4.51] on the over-10-min bands**, while losing −22.23 s on the
2–10 min band. The `proxy` anchor is what fails on a held row, so a model that does not use it is
better exactly there — and a blend applied everywhere pays for that gain out of the body.

## 25.1 Hypothesis H-25

> Applying the 0.5 blend ONLY on rows where the pipeline's own predicted |delta_hat| exceeds a
> threshold — a quantity observable at serve time — captures the tail gain without paying the body
> cost, and improves matched RMSE against the pipeline.

## 25.2 The rule, fixed

`y_hat = pipeline` where `|delta_hat| <= T`; `y_hat = w * U_w1 + (1 - w) * pipeline` where
`|delta_hat| > T`. `delta_hat` is the queue arm's own three-seed mean delta (`baseline` in
`data/cache_stand/fold_preds_queue_allrows.parquet`); `U_w1` is the unified arm's taxi time from the
same record and the same fold, so every comparison is paired on identical rows.

**Primary setting, chosen before measuring and not tunable afterwards: T = 600 s and w = 0.5.**
T = 600 s is Amendment 19.0's over-10-min band edge, defined on 2026-09-09 before this arm existed;
w = 0.5 is Amendment 19.2's registered blend weight, unchanged. On the fold, |delta_hat| > 600 s
selects **4.64%** of matched rows (the true |delta| > 600 s rate is 7.63%: the mean-regressing model
under-selects, which is the conservative direction).

## 25.3 Thresholds, locked

Paired row bootstrap, 2,000 draws, seed 0, on the 339,015 matched holdout rows, against the queue
pipeline:
- **ESTABLISHED** iff (i) the paired interval at the primary setting excludes zero in the rule's
  favour, (ii) point gain >= **+1.0 s**, (iii) gain > 2 x seed sd (0.145 s), AND (iv) the negative
  control below is cleared.
- **NOT WORKING** iff (i) fails.
- (i)–(iii) holding with (iv) failing is **NOT WORKING**: the gate would not be doing the work.

**(iv) Negative control, specified before running.** Apply the identical blend to a RANDOM subset of
matched rows of the same size, 1,000 draws, and compare the real rule's gain against that null
distribution. The rule must exceed the **95th percentile** of the permuted gains. This is what
separates "the gate finds the rows where the anchor fails" from "blending anything helps".

**Sensitivity, reported but NOT decisional:** T in {300, 900, 1200} s and w in {0.25, 0.75}. A
better number at a non-primary setting is reported as sensitivity and does NOT change the verdict —
choosing T after seeing the grid is the exact failure this pre-registration exists to prevent.

## 25.4 What a positive result licenses, and what it does not

It licenses ONE thing: fitting the y-target model on all twelve training months and predicting the
scored rows (~2.5 h), so the rule can be applied to a submission. It does NOT license shipping on
this evidence alone — the fold record's `U_w1` is a fold-A fit, and Amendment 9's rule stands: fold
TOTALS do not transfer, only paired relative gains on matched rows do (RESULT 5, 0.97–1.17x).


---

# RESULT 16 · 2026-09-10 00:30 local · Amendment 25 (the gated blend) — NOT WORKING as registered

Evaluated exactly as 25.2/25.3 fixed it, on the 339,015 matched holdout rows of
`data/cache_stand/fold_preds_queue_allrows.parquet` (no new fit; both prediction columns are in the
record). Primary setting T = 600 s, w = 0.5: the gate selects **15,743 rows (4.64%)**, of which
**77.3% truly have |delta| > 600 s** — the gate does identify held rows.

| clause | measured | passes |
|---|---|---|
| (i) interval excludes zero in the rule's favour | **−9.844 [−29.366, +0.696]** | no |
| (ii) point gain >= +1.0 s | −9.844 | no |
| (iii) gain > 2 x seed sd | −9.844 vs 0.290 | no |
| (iv) gain > p95 of 1,000 random-subset blends | −9.844 vs +0.079 | no |

**VERDICT: NOT WORKING.** Sensitivity (reported, not decisional): every T in {300, 600, 900, 1200}
and w in {0.25, 0.5, 0.75} is negative pooled, and smaller w is uniformly better — the ordering of a
blend weight that should be zero.

**Where it fails is one airport.** Per-airport gain at the primary setting: EGLL **+2.295**,
EDDF +0.736, LEBL +0.640, LTFM +0.511, LFPG +0.372, EDDM +0.215, LEMD +0.042, LSZH +0.004,
EHAM −0.840, **LIRF −72.754** (gate 11.58% of its rows). Nine of ten airports improve; LIRF alone
moves the pooled number by more than the other nine combined, because the unified model is
catastrophically worse there (RESULT 15: LIRF 384.07 → 622.84).

---

# AMENDMENT 26 — the gated blend, ex-LIRF

**2026-09-10 00:32 local. Written BEFORE the ex-LIRF number is computed.** RESULT 16's per-airport
table is the evidence: the rule is positive at nine of ten airports and LIRF alone sinks it.
Amendment 12.3 and Amendment 17 set the precedent — LIRF is where per-airport decisions have
repeatedly diverged from the pool, because its error is dominated by fills and monsters rather than
by taxi dynamics.

## 26.1 Hypothesis H-26

> The Amendment 25 rule, applied at every airport EXCEPT LIRF, improves matched RMSE against the
> pipeline on those nine airports and on the full matched set.

## 26.2 The rule

Amendment 25.2 unchanged (T = 600 s, w = 0.5, `delta_hat` from the queue arm), with the gate forced
to False on LIRF rows. LIRF keeps the pipeline exactly as it ships today.

## 26.3 Thresholds, locked — DELIBERATELY STRICTER THAN 25.3

This is the fourth variant evaluated on one fold record (U pooled, the 0.5 blend, the gate, now the
ex-LIRF gate). Testing repeatedly on the same rows inflates the chance of a false positive, so the
bar rises rather than falls:
- **ESTABLISHED** iff (i) the paired interval on the FULL matched set (LIRF included, scored with
  the pipeline there) excludes zero in the rule's favour, (ii) point gain >= **+1.5 s** (above
  25.3's +1.0 s), (iii) gain > **3 x** seed sd (above 2x), (iv) the gain exceeds the **99th**
  percentile (above the 95th) of 1,000 random-subset blends of the same size drawn from the nine
  airports, AND (v) **at least 7 of the 9 airports improve individually**.
- **NOT WORKING** iff (i) fails.
- Anything between is INCONCLUSIVE and is reported with every number, not shipped.

## 26.4 What it licenses

A positive result licenses only the twelve-month y-target fit needed to apply the rule to a
submission (~2.5 h), never a ship on fold evidence alone (Amendment 9).


---

# RESULT 17 · 2026-09-10 00:35 local · Amendment 26 (the gated blend, ex-LIRF) — INCONCLUSIVE: real, controlled, and small

Same record, same paired bootstrap, LIRF forced to the pipeline. The gate selects **12,718 rows
(3.75% of matched)**, of which **79.8% truly have |delta| > 600 s**.

| clause (26.3) | measured | passes |
|---|---|---|
| (i) interval excludes zero in the rule's favour | **+0.454 [+0.132, +0.758]** | **yes** |
| (ii) point gain >= +1.5 s | +0.454 | **no** |
| (iii) gain > 3 x seed sd (0.435) | +0.454 | yes |
| (iv) gain > p99 of 1,000 random 9-airport subsets | **+0.454 vs p99 +0.101** (null mean +0.017, max +0.133) | **yes** |
| (v) >= 7 of the 9 airports improve | **8 / 9** | yes |

**VERDICT: INCONCLUSIVE.** Four clauses of five hold, including the negative control decisively —
the real rule's gain is **4.5x the 99th percentile** of blending the same number of randomly chosen
rows, and above the null's maximum over 1,000 draws. **The gate is doing the work, not the
blending.** The point bar fails under BOTH the strict 26.3 bar (+1.5 s) and 25.3's original
(+1.0 s), so there is no reading of the pre-registration under which this clears.

Per airport: EGLL +2.295, EDDF +0.736, LEBL +0.640, LTFM +0.511, LFPG +0.372, EDDM +0.215,
LEMD +0.042, LSZH +0.004, EHAM −0.840, LIRF untouched. Nine-airport pooled 205.359 → 204.822.

**Honest size and cost.** +0.454 s of matched RMSE is **~200 board MSE** at the 2026 weight
(224.2576² − 223.8037² = 203.4 matched MSE x w_m) — smaller than the seed-averaging lane's ~215 and
**3% of the 6,884 MSE that separates us from tenth place**. Shipping it requires the twelve-month
y-target fit (~2.5 h) that 26.4 licenses, for that ~200 MSE. Below-bar but real: the ship decision
is the owner's, on RESULT 9's precedent, and it is not step-sized either way.

### 17a · a stopping note on multiplicity, recorded against myself

Four variants have now been evaluated on this one fold record (U pooled, the flat 0.5 blend, the
gate, the ex-LIRF gate). Each additional cut of the same rows raises the chance that a surviving
interval is noise, which is why 26.3 raised its bars rather than lowering them. **No further
variant of this rule will be evaluated on this record.** A fifth cut would be fishing, and the
honest next step for this idea is not another threshold but the twelve-month fit and a fresh fold.


---

# RESULT 18 · 2026-09-10 02:38 local · Amendment 19.1 arm D (the airport-day regime block) — NOT WORKING

`lgbm_fold.py --dayfeats --queue --baseline A2 --pa-trees es`, fold A, 2 h 04 min, peak 4.86 GB.
Records `reports/lgbm_fold_queue_day.{json,log,console.log}`, predictions
`data/cache_stand/fold_preds_queue_day.parquet`. Baseline = the queue record (224.258); treatment =
the same design plus the nine DAY_FEATS (89 columns), best_iter re-found → n_ref 27,634.

| clause (19.3) | measured | passes |
|---|---|---|
| (i) paired interval excludes zero in the block's favour | **−0.054 [−0.311, +0.210]** | no |
| (ii) point gain >= +1.0 s | −0.054 | no |
| (iii) gain > 2 x seed sd (0.291) | −0.054 | no |
| (iv) >= 6 of 10 airports improve | 5 / 10 | no |
| (v) the >10 min bands improve by >= +5 s | −0.907 [−2.079, +0.260] | no |

**VERDICT: NOT WORKING.** The block is flat: the interval is centred on zero and the point estimate
is negative. By band it gains a little on the rows that agree (+0.391 [+0.075, +0.685] under two
minutes) and loses on the ones that matter (−2.837 [−5.793, +0.128] beyond twenty minutes) — the
exact inverse of the mechanism 19.1 claimed for it.

**This is what Amendment 19.0 predicted and the arm confirms it.** The mode clusters on days — a
top-decile day carries 2.3–3.7x the mode rate of a bottom-decile day — but *within* a day the
signal reaches only AUC 0.63. Regime shifts a base rate; it does not identify a row. A day-level
block hands the model information it already reconstructs from the 80 columns it has, and the
holding tail is untouched. **Day-level regime is closed as a source of matched-row gain.**


---

# RESULT 19 · 2026-09-10 04:26 local · Amendment 19.2 arm Y (taxi time as the target, blended) — NOT WORKING pooled; the band inversion is stronger than Arm U's

`lgbm_fold.py --ytarget --queue --baseline A2 --pa-trees es`, fold A, 1 h 47 min, peak 4.71 GB.
Records `reports/lgbm_fold_queue_ytarget.{json,log,console.log}`. Baseline = the queue record
(224.258). Y alone 260.651; the registered 0.5 blend 233.341.

| clause (19.3) | measured | passes |
|---|---|---|
| (i) the blend's paired interval excludes zero | **−9.083 [−27.996, +1.094]** | no |
| (ii) gain >= +1.0 s | −9.083 | no |
| (iii) gain > 2 x seed sd (0.291) | −9.083 | no |

**VERDICT: NOT WORKING** on the registered pooled clause.

### 19a · what the bands say, which 19.2 required be reported

| band | share of matched SSE | blend vs baseline | Y alone vs baseline |
|---|---|---|---|
| < 2 min | 13.7% | −0.011 [−0.193, +0.161] | −1.010 [−1.397, −0.642]* |
| 2–10 min | 37.8% | **−21.083** [−58.016, +0.141] | −73.096 [−176.747, −0.867]* |
| 10–20 min | 17.4% | **+1.420 [+0.807, +2.009]*** | +0.058 [−1.225, +1.258] |
| **> 20 min** | 31.1% | **+9.992 [+5.628, +14.348]*** | +8.585 [+0.107, +16.658]* |
| **over10** | 48.5% | **+4.459 [+2.835, +6.058]*** | +3.212 [+0.043, +6.202]* |

Three of the blend's band intervals exclude zero in its favour, and they are **the three that carry
48.5% of matched error**. This is the same inversion RESULT 15a found on the unified arm, and it is
**larger here**: over10 +4.459 against the unified arm's +2.83, gt20 +9.99 against +6.98. A
matched-only y-target model is a better tail model than the all-rows one, which is what Amendment
19.2 predicted and Amendment 20 muddied by mixing the strata.

**Per airport, the blend:** EDDF 177.9→176.8, EDDM 166.5→165.8, EGLL 229.1→226.4, LEBL 220.3→219.6,
LEMD 174.7→174.4, LFPG 244.5→243.4, LSZH 174.2→173.6, LTFM 243.6→242.6 — eight improve, EHAM
163.1→163.8 does not, and **LIRF 384.1→453.9 costs more than the other nine gain**. The third arm
in a row where Rome alone reverses the pooled sign.


---

# AMENDMENT 27 — the gated blend on arm Y's predictions: ONE test, no new knobs

**2026-09-10 04:32 local. Written BEFORE the rule touches arm Y's record.**

RESULT 17 measured the gated blend at +0.454 s using the UNIFIED arm's predictions. RESULT 19 then
showed the matched-only y-target model is a **better tail model than the unified one** (over10
+4.459 [+2.835, +6.058] against +2.83; gt20 +9.99 against +6.98) — and it fails pooled for the same
two reasons: the 2–10 min band and LIRF. The gate and the LIRF exclusion address exactly those two.

## 27.1 Hypothesis H-27

> Amendment 26's rule, unchanged, applied to arm Y's predictions instead of the unified arm's,
> improves matched RMSE against the pipeline.

## 27.2 No parameter is chosen here

`T = 600 s`, `w = 0.5`, LIRF excluded — **every one inherited from Amendment 26**, which inherited T
from 19.0's band edge (fixed 2026-09-09) and w from 19.2. `delta_hat` is arm Y's own three-seed
baseline column; the pipeline is `max(proxy − delta_hat, 1)` from the same record, so the comparison
is paired on identical rows. **No grid is searched and no sensitivity sweep is run** — a sweep here
would be the fishing this amendment exists to prevent.

## 27.3 Thresholds, locked — corrected for the family of tests

This is the **sixth** test in the blend family on the same fold-A holdout (U pooled, U's flat blend,
the gate, the gate ex-LIRF, arm Y's flat blend, this). A nominal 95% interval no longer carries 95%
across a family that size, so:
- **ESTABLISHED** iff (i) the paired **99%** interval excludes zero in the rule's favour — a
  Bonferroni-style correction for six tests, not the 95% every earlier clause used — AND (ii) point
  gain >= **+1.5 s**, (iii) gain > 3 x seed sd, (iv) the gain exceeds the **99th** percentile of
  1,000 random-subset blends of the same size drawn from the nine airports, (v) **>= 7 of the 9
  airports improve**.
- **NOT WORKING** iff (i) fails.
- Anything between is INCONCLUSIVE, reported with every number, and shipped only on an explicit
  owner decision (RESULT 9's precedent).
- Both the 95% and the 99% intervals are reported, so the effect of the correction is visible rather
  than hidden.

## 27.4 The family stops here

Whatever this returns, **no seventh test of this family runs on fold A.** If it clears, the next step
is the twelve-month fit and the board; if it does not, the blend family is closed.


---

# RESULT 20 · 2026-09-10 04:45 local · Amendment 27 — INCONCLUSIVE, and a scale error caught by its own control

### 20a · the error, recorded first

The first run of this test reported a gain of **−109.0 s** and is **VOID**. Cause: the two fold
records use different conventions and I assumed one. `fold_preds_queue_allrows.parquet` stores arm
columns as **taxi times** (its unmatched rows have no proxy, so it must);
`fold_preds_queue_ytarget.parquet` stores them on the **delta scale**, recovered as
`max(proxy − col, 1)` — the convention its own harness tests use. Reading the second as if it were
the first turned every prediction into nonsense.

**What caught it was the negative control, not the result.** A random-subset blend scored −23.9 s
where it must score ~0 by construction, because a random subset of a *correct* blend cannot be
systematically harmful. A wrong number that large in the control is unmissable; the same bug in a
lane without a control would have been reported as a finding. Both records are now verified against
their own JSON (`baseline` 224.2576, `Y` 260.6511, `blend` 233.3405 all reproduce exactly), and
**RESULTS 16 and 17 are unaffected** — they used the all-rows record under its correct convention,
re-verified here.

### 20b · the corrected result

Amendment 26's rule, parameters unchanged, on arm Y's predictions. The gate selects **12,718 rows
(3.75%)**, of which **79.8% are truly held**.

| clause (27.3) | measured | passes |
|---|---|---|
| (i) **99%** interval excludes zero (corrected for six tests) | **+0.527 [+0.114, +0.915]** | **yes** |
| (ii) point gain >= +1.5 s | +0.527 | **no** |
| (iii) gain > 3 x seed sd (0.436) | +0.527 | yes |
| (iv) gain > p99 of 1,000 random 9-airport subsets | **+0.527 vs +0.115** (mean +0.032, max +0.150) | **yes** |
| (v) >= 7 of the 9 airports improve | **8 / 9** | yes |

95% interval for comparison: [+0.234, +0.820]. **VERDICT: INCONCLUSIVE** — the same shape as
RESULT 17 and slightly better (+0.527 against +0.454), surviving a Bonferroni-corrected interval and
a control it beats 4.6x, but failing the point bar under every registered reading.

Per airport: EGLL +2.394, EDDF +0.779, LTFM +0.666, LEBL +0.579, EDDM +0.461, LSZH +0.241,
LFPG +0.202, LEMD +0.194, EHAM −0.586, LIRF untouched. On the gated rows the pipeline scores 418.79,
the y-model 416.49 and the blend **411.21** — the blend beats both parents there, which is what a
real variance reduction looks like.

**Size: ~232 board MSE, 3.4% of the 6,884 needed for tenth**, and it costs a twelve-month y-target
fit (~2.5 h) to ship. Per 27.4 **the blend family is now closed on fold A**: six tests, one
consistent picture — the y-target model is the better tail model, the gate finds the tail, Rome
must be excluded, and the whole thing is worth a few hundred MSE. It is not the step.


---

# RESULT 21 · 2026-09-10 06:20 local · Amendment 22 arm F (the record-ordering block) — ESTABLISHED, and the first arm to pass a mechanism clause

`lgbm_fold.py --orderfeats --queue --baseline A2 --pa-trees es`, fold A, 1 h 53 min, peak 4.84 GB.
Records `reports/lgbm_fold_queue_order.{json,log,console.log}`. Baseline = the queue record
(224.258); treatment = the same design plus the three ORDER_FEATS (83 columns), n_ref 27,155.

| clause (22.3) | measured | passes |
|---|---|---|
| (i) paired interval excludes zero in the block's favour | **+1.694 [+1.407, +1.993]** | yes |
| (ii) point gain >= +1.0 s | +1.694 | yes |
| (iii) gain > 2 x seed sd (0.291) | 11.6x | yes |
| (iv) **the over-10-min band's own interval excludes zero** | **+12.156 [+10.823, +13.466]** | **yes** |

**VERDICT: ESTABLISHED**, all four clauses.

**It is the first arm in this project to pass its mechanism clause.** Every other lane either moved
the body and not the tail (the queue block, RESULT 9) or moved nothing (arm D, RESULT 18). This one
moves the rows the diagnosis named:

| band | share of matched SSE | gain |
|---|---|---|
| < 2 min | 13.7% | −2.059 [−2.387, −1.748] |
| 2–10 min | 37.8% | −0.381 [−0.594, −0.154] |
| 10–20 min | 17.4% | **+7.114 [+6.290, +7.970]** |
| **> 20 min** | 31.1% | **+22.584 [+19.103, +25.882]** |
| **over10** | 48.5% | **+12.156 [+10.823, +13.466]** |

It is a genuine trade — small, real losses on the body against large gains on the tail — and the net
is positive by a wide margin. Per airport: **LIRF +8.17** (the airport that has reversed the sign of
four other lanes), EDDM +2.08, LTFM +1.50, EDDF +1.41, LEBL +1.37, LEMD +0.77, LSZH +0.11,
EGLL +0.06; LFPG −0.14 and EHAM −0.10. **8 of 10 improve.**

### 21a · the transfer check, and a correction to my first reading of it

A feature whose serve-time distribution differs from training wins on the fold and loses on the
board, so the block's distribution was compared before anything was shipped. **The correct reference
is the FIT months** — what the model actually saw — not one arbitrary training month:

| | \|o_dev_flt\| p90 | ratio to fit |
|---|---|---|
| fit months (2025, excluding 1 and 7) | 0.0248 | — |
| fold holdout (2025, months 1 and 7) | 0.0277 | 1.12 |
| **scored file (2026, months 1 and 7)** | **0.0256** | **1.03** |

The fold measured this gain **through a wider shift than the scored file presents**, so the transfer
rule applies without a discount. My first pass compared the scored file against March alone and
reported ratios of 1.2–1.9x; that reference was a single noisy month and the reading is withdrawn.
Per airport the scored/fit ratio is 0.94–1.09 everywhere except LSZH at 1.41, whose gain (+0.11) is
immaterial either way. `o_dev_mvt` p90: fit 0.5054, holdout 0.5241, scored 0.5008.

**Board scale:** +1.694 s of matched RMSE is **~745 MSE** at the 2026 weight — the largest matched
lane since the queue block, 11% of the 6,884 to tenth place. **It ships as v6** under the
established-lane rule, after arm W clears the machine.


---

# RESULT 22 · 2026-09-10 08:02 local · Amendment 24 arm W (weather) — INCONCLUSIVE, and it refutes the mechanism I argued for it

`lgbm_fold.py --weatherfeats --queue --baseline A2 --pa-trees es`, fold A, 1 h 42 min, peak 5.00 GB.
Records `reports/lgbm_fold_queue_weather.{json,log,console.log}`. Baseline = the queue record
(224.258); treatment = the same design plus the ten WEATHER_FEATS (90 columns), n_ref 24,438.

| clause (24.4) | measured | passes |
|---|---|---|
| (i) paired interval excludes zero in the block's favour | **+0.725 [+0.495, +0.947]** | yes |
| (ii) point gain >= +1.0 s | +0.725 | no |
| (iii) gain > 2 x seed sd (0.291) | 5.0x | yes |
| (iv) **the over-10-min band's own interval excludes zero** | **−0.544 [−1.578, +0.439]** | **no** |

**VERDICT: INCONCLUSIVE**, and the mechanism clause fails in an instructive direction.

| band | share of matched SSE | gain |
|---|---|---|
| < 2 min | 13.7% | **+1.220 [+0.957, +1.487]** |
| 2–10 min | 37.8% | **+1.221 [+1.027, +1.423]** |
| 10–20 min | 17.4% | +0.846 [+0.079, +1.584] |
| **> 20 min** | 31.1% | **−2.673 [−5.178, −0.095]** |
| over10 | 48.5% | −0.544 [−1.578, +0.439] |

**Weather improves ordinary taxi time and does not touch the holds.** It gains on the body, where
visibility and wind plausibly change how fast an aircraft actually taxis, and it is *worse* on the
rows beyond twenty minutes with an interval that excludes zero. The seasonal split 24.4 required
makes it plainer: **January +1.739 s pooled** (the de-icing season, and the block's best case) but
only **+0.932 s on January's held rows**; **July +0.045 s pooled and −1.357 s on July's held rows.**

**This refutes the reasoning I used to justify the lane.** I argued weather was the remaining lane
"with a physical mechanism behind it" because 78% of held rows are pushed-back-then-held and holds
are caused by de-icing and flow control. The block does carry the de-icing signal — January is where
all of its gain is — but **naming the weather does not name the held row**. Freezing conditions cover
1.92% of scored rows while held rows are 7.6%: the mechanism is real, too rare, and evidently not
what separates the two clocks. The earlier +0.49 s result was not an artifact of low capacity after
all; at capacity with 80 columns beside it the block is worth +0.725 s, and the difference is body,
not tail.

**Board scale: ~320 MSE, 5% of the gap.** Below its point bar and failing its mechanism clause, so
it does not ship on the established-lane rule; per RESULT 9's precedent that is the owner's call.
**Its gains are complementary to arm F's** — W moves the body (+1.22 / +1.22), F moves the tail
(+7.11 / +22.58) — which makes a combined design the obvious next fold arm rather than a second
submission.


---

# NOTE · 2026-09-10 16:20 local · RESULT 13 — SUPERSEDED IN PART by the date-slip prereg

RESULT 13 described the remainder of the unmatched lane as **"4 stale-BLOCK rows in two months on four
distinct airport-days with no shared signature."** The last clause is **wrong**, and this note records
that rather than editing the block above it (the file is append-only).

The 24-hour tail rows DO share a signature, found by a parallel session and pre-registered in
`plans/PREREG_rome_dateslip_2026_09_10.md` (H-DS): the airport's actual off-block **clock time stamped
under the schedule's date**, so the label is the taxi time plus exactly 86,400 s. On 2025 LIRF unmatched
rows with `dayoff = 1`, above `sp` 56,000 s there are 10 date-slips against 5 fills, and `y − 86,400`
on the date-slips has p10/50/90 of 771 / 1,020 / 1,679 s — an ordinary taxi time. I looked at the same
rows and read four unrelated stale stamps; the structure was the date, not the time. Its verdict and its
disclosed boundary-fragility belong to that file and to `reports/GAP_LOCATED_2026_09_10.md`, not here.

What of RESULT 13 still stands: the oracle fill decision is worth 13,497 MSE and all of it is at LIRF;
the shipped `p` scores AUC 0.944 pooled; tail insurance is priced and refused (break-even P = 23%).

# NOTE · 2026-09-10 16:20 local · a validation defect in this harness, recorded where its results live

The overnight audit (`plans/TOP_PATH_2026_09_10.md`, Priority 0; `reports/bug_classes.md` BC-1) found
that `lgbm_fold.py::load_fold` builds target encodings over `tr`, which INCLUDES the inner early-stopping
months, so early-stopping rows' encodings were fitted on their own labels. **Every `best_iter` in every
RESULT above — and every shipped version — was selected against that biased signal.** The outer
January/July holdout was never contaminated, so every held-out RMSE and every paired interval recorded
in this file remains a valid measurement of the model that was fitted; what is not established is that
those models were tuned as well as they could have been. The repair is `prc/encoding.py`, built and
tested by the auditing session and deliberately not wired while this file's arms were running.
