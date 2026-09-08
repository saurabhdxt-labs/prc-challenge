# Train/serve audit — does the 2025 model meet the same world in 2026?

The model fits 2025 and scores January and July 2026. Nobody had checked whether the features it
leans on behave the same there. Peak RSS 1.13 GB; all cuts per airport, because pooling has hidden
the truth four separate times in this project.

## 1. There is no bonus labelled 2026 data

`ranking.parquet` holds 689,534 rows = 344,841 DEP + 344,693 ARR. Every 2026 departure is scored;
none carries a label. The hypothesis that labelled 2026 departures were sitting in the file is dead.

## 2. Encoding vocabulary transfers cleanly — not the defect

Share of 2026 scored rows whose key was seen in the 2025 training vocabulary:

```
STAND_mvt              99.92%      ADES_mvt               99.95%
RUNWAY_mvt            100.00%      AIRCRAFT_TYPE_mvt      99.88%
airport|runway|stand   99.28%      AIRCRAFT_OPERATOR_flt  98.41%
airline (FLIGHT[:3])   99.72%
```

Lowest per airport is EDDM at 95.4% for `STAND_mvt`. Target encodings are not silently falling back
to the prior for any material block of rows.

## 3. Feature distributions are stable, with named exceptions

`proxy` (`MVT - AOBT_3`) is stable everywhere; the largest median shift is EHAM at +56 s. Shifts
worth knowing:

- **LFPG `aobt_eobt` median 480 -> 180 s (-300 s).** This is the second most important feature in
  the E1 model (permutation +93.5). LFPG is 11.6% of scored rows and 13.4% of matched SSE.
- `sp` shifts: EGLL -118, LFPG -173, EHAM -80, LSZH +75, LTFM +69.
- `nmdelay` shifts: EHAM -120, LFPG -120, LTFM +120.

The *conditional* relationship cannot be checked without 2026 labels; only the marginals are
verifiable. The LFPG shift is the one to watch.

## 4. The validation design interpolates in time; the task extrapolates

Fold A holds out Jan+Jul 2025 and trains on the ten months **surrounding** them. The competition
trains on 2025 and predicts forward into 2026. Median `delta = BLOCK - AOBT_3` moves month to month
at every airport — spread over the twelve months: EDDM 112 s, LTFM 126 s, LIRF 174 s, LEBL 61 s,
LEMD 60 s, EGLL 55 s, LFPG 56 s. Forward generalisation has never been measured, so no bootstrap
interval reported in this project reflects the actual extrapolation risk. Forward folds
(train Jan–Jun -> test Jul, etc.) are outstanding.

## 5. The residual has two named mechanisms and they run in opposite directions

`P(delta < -600)` — the "pushed back, then held" tail that carries 30.5% of matched SSE — ranks the
airports exactly as their RMSE does, and its seasonality has opposite signs:

```
        Jan    Apr    Jul    Dec        RMSE
EHAM   0.07   0.01   0.15   0.06        168     no pathology
LEMD   2.08   2.28   3.41   2.21        179
EDDM  11.61   5.01   5.88  14.49        174     WINTER peak -> de-icing
LSZH   2.93   2.08   2.91   2.70        188
LEBL   1.55   3.08   7.13   3.29        225     summer peak
EGLL   1.77   1.62   5.21   2.14        249     summer peak
LFPG   4.48   3.04   5.01   3.78        249     summer peak
LTFM   2.13   3.03   5.71   2.80        255
LIRF   8.35  13.83  19.87   9.55        391     SUMMER peak -> ATFM slot holds
```

**The scored months are January and July** — precisely where the two mechanisms sit at opposite
maxima. Weather was tested pooled and returned +0.49 s; the mechanism is per-airport and per-season,
and de-icing at EDDM/LSZH/EDDF in January is directly observable from METAR (temperature, dewpoint,
precipitation type). This is the largest untested lever on the matched side.

## 6. Closed cheaply

`BLOCK_TIME` is **not** minute-rounded — whole-minute share is 1.7–8.4% by airport against 1.67%
under uniform seconds, while `AOBT_3` is 83.8–100% whole-minute. So the label's seconds-within-minute
is not recoverable from `MVT`, and there is no free lattice structure to exploit. Dead, cheaply.

## 7. Model-free floor — inconclusive, recorded so it is not re-run

A repeat-pair estimator (within-group sd of `delta` for rows matching on progressively finer keys)
gives 362.8 at (airport, month) down to 216.5 at the finest key. **These are not a floor on our
model**: the fine keys retain only the 10% of rows that have twins, which are the routine ones, and
our model uses far more than the key columns. Selection-biased and not evidence of headroom either
way. The sanity check passed — (airport, month) reproduces the constant-per-airport 364.3.
