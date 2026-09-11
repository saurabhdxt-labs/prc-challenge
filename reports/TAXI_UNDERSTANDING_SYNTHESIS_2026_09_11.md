# What moves taxi-out time, and what our model does with it — synthesis of three studies

Written 2026-09-11 after: `reports/TAXI_FACTOR_ATLAS_2026_09_10.md` (physical factors in seconds, SHAP on a held-out
validated model, 2,035,114 ordinary departures), `reports/FEATURE_IMPORTANCE_2026_09_10.md` (drop-group refits with
day-block intervals on arm F's design), `reports/RESEARCH_TAXI_FEATURES_2026_09_10.md` (literature, EUROCONTROL docs,
aircraft data, 10-airport table). Conditional effects are model attributions, not causal claims.

## 1. The physics, in seconds (ordinary taxis; typical taxi ≈ 909 s)

| factor | effect | where |
|---|---|---|
| airport | medians 710 s (LSZH) … 1,318 s (EGLL) | |
| runway within airport | up to +326 s (EHAM 36L Polderbaan vs 24), +320 s (LTFM 17R vs 36) | EGLL runways within 30 s |
| stand area within airport | 162 s (EHAM) … 481 s (EDDF) range | |
| **freezing weather (de-icing)** | **+398 s** (LFPG +507, LSZH +414, EHAM +368, EDDM +359); snow at EDDM +671 s | tail driver at EHAM/LSZH/EDDM/LTFM (lift 11–28×) |
| visibility < 1.5 km | +257 s | |
| runway queue at pushback | ≈ +20 s per extra aircraft (EDDM 11.5 … LFPG 23.5) | most of EGLL's long median |
| recent taxi times on my runway | ≈ +12 s per extra minute | |
| aircraft size | Heavy +64–98 s vs Medium; A380 +149 s | |
| airline (beyond stand, type) | UAL +99, BAW +85, IBS −114 | LIRF US carriers: 39–42% over 30 min |
| night (22–02 h) | +22 … +110 s | daytime clock ±10 s |
| holiday, weekday, rain, arrivals taxiing | < ~15 s each | holidays leave hourly traffic unchanged |

Variance explained (held-out R²): airport 0.167 → + runway/stand 0.342 → + time/season 0.389 → + congestion 0.484 → +
aircraft/airline 0.512 → + delay/turnaround 0.581 → + weather 0.593 → **+ NM clock proxy 0.695**. Physical factors alone
explain ≈ 59%; the proxy adds 10 points, largely re-measuring congestion and layout.

## 2. What the shipped model relies on (drop-group value, ΔRMSE s, day-block 95%)

Clock gaps +36.6 [31.5, 42.3] ≫ stand identity +3.9, stand history +3.3, airline/segment +3.3, queue +2.6, ordering +1.9,
destination +1.1, aircraft type + wake +0.95, recent taxi +0.64, airport +0.51, season/time +0.30 [−0.43, +0.97], record
flags −0.13 (dead). Constant, unused: `sched_sec`, `actype_null`. Unmatched body: `AIRCRAFT_OPERATOR_flt` is 100% null.

## 3. Structural findings that change the engineering

1. **`AOBT_3` is a PLANNED value at LEBL, LEMD, EDDM, LTFM** (EUROCONTROL DPI guide: NM derives off-block as take-off
   minus a per stand×runway taxi-time parameter). Verified here: within stand×runway, corr(proxy, taxi) = −0.13 / −0.10 /
   −0.07 / +0.03 there vs +0.45 / +0.49 at EGLL / EHAM; the proxy sits on its modal minute on 50–81% of rows vs 17–20% for
   the true taxi. At those four airports the model's strongest input carries almost no per-flight information.
2. **Weather was never built properly.** Arm W's precipitation column is 0.0 on every row (European METARs have no `p01i`
   group), so its freezing flag fired only on weather codes. De-icing is the biggest physical tail driver found (+398 s,
   lift up to 28×) and January 2026 contained the EHAM storm.
3. **"Delay" is mostly a recording artefact**: for 18–22% of late flights the airport stamps off-block > 5 min before NM;
   on clock-agreeing rows a 30–60 min delay adds ≈ +20 s, not +102 s.
4. The unmatched lane (no NM record, no proxy) is where physical factors should matter most — and its body sees only
   schedule offset, hour, the congestion witness and four live encodings (no queue, weather, aircraft type, stand history).
   Arm U (with the queue block) beat S1 by ≈ 527–619 MSE on non-Rome unmatched rows in fold A — a measured floor, not a cap.

## 4. Feature-engineering candidates, ordered by evidence × size (each to be registered and measured in the pipeline)

| # | candidate | lane | evidence | risk |
|---|---|---|---|---|
| 1 | **Weather done right**: METAR weather-code parsing (snow, freezing precip / fog, ice pellets), temp/dew-point, visibility, a de-icing-airport × freezing flag; hour-level storm state | both (unmatched first) | atlas +398 s, tail lift 11–28×; arm W's precip bug | METAR is Iowa Mesonet (already used, not label-derived) |
| 2 | **Physical-factor unmatched body**: queue counts, recent runway taxi, stand history, weather, aircraft type, airline × airport — the atlas's physics for rows without a proxy | unmatched | physics explains 59% without the proxy; arm U floor +527–619 | fills / date-slips must stay in their own lanes |
| 3 | **NM taxi-parameter block** at the four planned-value airports: modal proxy minute per stand×runway, deviation, at-mode flag | matched | DPI guide + the correlations above | none (serve-time) |
| 4 | airport-scoped aircraft type / airline encodings | both | single-key screen −24 s; atlas airline effects | BC-1 (fit on training mask only) |
| 5 | runway queue over an estimated taxi window (Idris R² 0.59) | matched | literature | needs out-of-fold first stage |
| 6 | remove dead inputs (flags, constants, null operator) | both | drop-group ≈ 0 | none |

Not worth building: holidays (≤ ~15 s, traffic unchanged), season beyond weather, aircraft physical dimensions
(ceiling ≤ 32 MSE; size already carried by type), airport size / runway count (absorbed by airport identity).

**Correction (2026-09-11 06:41:41), after independent review:** "physical factors alone explain ≈ 59% without the proxy" includes the
delay / turnaround step (+0.069 R²), which is DERIVED from AOBT_3 and does not exist on unmatched rows. Without any
NM-derived input the ladder stands at ≈ 0.52 (0.512 after aircraft / airline, + ≈ 0.012 weather). The same caveat applies
to the E5 prereg's "Why" paragraph.
