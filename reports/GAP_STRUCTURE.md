# Where the gap to the leader actually is

> **AMENDED 2026-09-09 02:15Z, after the first real score.** `merry-quicksand_v2` scored
> **301.7019** on the board (rank 26 of 80) against a `--validate` fold number of 330.22 for the
> same artifact. The fold is **+28.52 s pessimistic** at this model class, most likely because
> `--validate` holds out Jan+Jul 2025 and so denies the validation model the very seasonality the
> evaluation months carry. Consequences for what follows:
>
> * The true gap to the leader is **30,158 MSE**, not 43,000.
> * **Section 2's claim that "perfect matched scores 231.35, so the leader is unreachable through
>   the matched lane" is RETRACTED.** It is fold-derived. Board-anchored, under the fold's
>   matched/stratum split ratio -- an assumption, not a measurement, since the board reports only
>   a total -- perfect matched implies 211.37 and would beat the leader.
> * What survives: the stratum is roughly half of our squared error, a 1% relative gain is worth
>   about the same in either lane (Section 3), and Sections 1, 5, 6 and 7 are direct measurements
>   on the data and are unaffected.
> * Every fold total in this file is an absolute number and inherits the offset. The A/B
>   comparisons do not.

Measured 2026-09-09. Every number below came from one consistent fold (held-out Jan+Jul 2025,
n=344,336) or from a direct read of `data/raw/`. Nothing here is a subsample magnitude, and no
total mixes components from different runs. Scripts ran `OMP_NUM_THREADS=1 nice -n 19 python3.11 -B`.

## 0. The one-paragraph version

The scored file splits into 339,551 **matched** rows (a Network Manager off-block exists) and 5,290
**unmatched** rows (1.53%). Those 5,290 rows carry **51.5% of our squared error**. A *perfect*
matched model — RMSE exactly 0 — would still score **231.35**, only 15 s better than the leader's
246.71. Conversely, holding matched at its LightGBM value of 226.24, the stratum alone reaching 826
lands exactly on 246.71. **The 43,000 MSE gap and our stratum deficit are the same number.** All
four lanes closed in previous sessions (stand occupancy, 24h date-slip, ADS-B, learner capacity)
were matched-side or general-capacity work.

## 1. The exact identity that defines the problem

`y = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt` holds **exactly** — max|difference| = 0.0 over all
344,419 DEP rows of Jan+Jul 2025, zero nulls. Taxi-out is one subtraction; `BLOCK_TIME_UTC_mvt` is
the single hidden quantity, and it is 100% null on the scored rows (`ranking.parquet`).

On matched rows the pipeline models `delta = BLOCK_TIME - AOBT_3` and recovers `y = proxy - delta`.
So the matched problem is **entirely** the disagreement between two off-block clocks:

| quantity | value |
|---|---|
| sd(`delta`), matched fold rows (n=339,012) | 426.86 |
| RMSE predicting `delta` = 0 (i.e. `proxy` as-is) | 426.95 |
| RMSE predicting `delta` = mean | 426.85 |
| shipped matched model | **237.46** |
| LightGBM matched (`lgb_refit`) | **226.24** |

The matched model is **not** weak: it explains 69% of `delta`'s variance. The error is heavy-tailed —
1.47% of matched rows carry 50.4% of the squared error, 0.29% carry 29.7%.

**There is no second off-block clock to find.** The raw schema is 30 columns (the pipeline reads 22);
the nine unread are `FLIGHT_ID_mvt`, `CALLSIGN_flt`, `ADEP_flt`, `ADES_flt`, `ADES_FILED_flt`,
`AIRCRAFT_TYPE_flt`, `ARVT_1_flt`, `ARVT_3_flt`. No `AOBT_1`/`AOBT_2` exists. Hypothesis closed.

## 2. The budget, on one consistent fold

Weights are exact from `ranking.parquet`: w_matched = 0.984660, w_unmatched = 0.015340.

| position | matched | stratum | total | matched share | stratum share |
|---|---|---|---|---|---|
| shipped artifact | 237.46 | 1946.23 | 337.09 | 48.9% | 51.1% |
| + consolidated stratum | 237.46 | 1867.90 | 330.22 | 50.9% | 49.1% |
| + LightGBM matched | 226.24 | 1867.90 | **322.37** | 48.5% | **51.5%** |

**Perfect matched (m = 0) scores 231.35.** The leader's 246.71 is unreachable through the matched
lane alone, at any level of skill.

## 3. Marginal value — the number that should drive scheduling

From matched 226.24 / stratum 1867.90:

| | per SECOND of RMSE | per PERCENT of RMSE |
|---|---|---|
| matched | 445.5 MSE | 1,008 MSE |
| stratum | 57.3 MSE | 1,070 MSE |

A second of matched is worth 7.8x a second of stratum — but the stratum has 8x more room, so **a 1%
relative gain is worth the same in either lane (0.94x)**. Neither lane dominates. Work whichever is
more tractable, and expect the winning line to use both.

## 4. What the leader's score actually requires

| board score | if stratum = our 1867.90, matched must be | if matched = our 226.24, stratum must be |
|---|---|---|
| 292.08 (true board median) | 174.6 | 1508.6 |
| 265.76 (`youthful-giraffe` FIRST submission) | 131.8 | 1148.3 |
| 246.71 (leader, current) | 86.4 | 826.0 |

Matched 86.4 means explaining 96% of `delta`'s variance; stratum 826 means 95.7% of the unmatched
variance on rows whose p99.9 is 21.6 hours. **Neither single-lane story is plausible.** A joint
reading is: matched 200 + stratum 1183 gives 246.69 — a 12% and a 37% relative gain. That is the
shape to aim at, and it is reachable.

## 5. The stratum is a mixture, and the obvious lever is REFUTED

`fit_unmatched` predicts `p*sp + (1-p)*nf`, where a "fill" row is one whose airport stamped the
scheduled push into the off-block field. The tempting hypothesis was that this is a hidden binary
classification problem: identify the fills and the error collapses. **Measured on the fold, it is not.**

| predictor on the 5,321 fold unmatched rows | RMSE |
|---|---|
| shipped mixture | **1867.9** |
| ORACLE fill-flag + constant for non-fills | 3080.5 |
| ORACLE, restricted to fill rows only | 24.5 |
| constant, restricted to non-fill rows | 3156.5 |

A perfect fill classifier makes things **worse**, because the mixture's hedging is doing real work
and the non-fill rows are the monsters: sd(y) = 3975, p95 = 2,528 s, p99.9 = 77,903 s, max 88,132 s
(24.5 hours). Fills are only 4.8% of the stratum. The shipped mixture already explains 78% of the
unmatched variance. **Lane closed: "identify the schedule-fills" is not the stratum lever.**

## 6. The fold understates the 2026 stratum — a caution on every projection

The stratum's *size* transfers almost exactly; its *difficulty* does not.

| | 2026 scored | 2025 fold | ratio |
|---|---|---|---|
| unmatched rows | 5,290 (1.534%) | 5,321 (1.545%) | 1.00 |
| `sp` p90 / p95 / p99 | 12,004 / 15,616 / 28,377 | 9,742 / 12,061 / 20,973 | 1.23 / 1.29 / 1.35 |
| `sp` sd | 6,375 | 5,407 | 1.18 |
| share with `sp` > 3h | **12.63%** | **7.20%** | 1.75 |

The airport mix also moved hard: EHAM 1.49x, LSZH 1.38x, LTFM 1.10x against EGLL 0.53x, EDDF 0.70x,
LEMD 0.77x. Since `p` is strongly airport-specific (LIRF 48.5% against 0.2-9.4% elsewhere), the mix
shift matters on its own; LIRF itself is stable at 0.96.

`sp` is the fill branch's entire prediction, so a heavier `sp` tail makes the 2026 stratum **harder**
than the fold. **Our ~322 fold position is optimistic on the stratum side, not pessimistic.** This
is the same warning the deliberate `test_stationarity` failures carry, now quantified on `sp`
rather than on prevalence.

## 7. Unexploited information, named

`ranking.parquet` is 689,534 rows: our 344,841 scored departures **plus 344,693 arrivals**. On the
arrival rows `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are **fully populated** (taxi-in, mean
537 s). Every scored departure's `MVT_TIME` (take-off) is also known. So the realised departure
sequence and the contemporaneous arrival stream are both observable inside the evaluation window,
and `derive()` currently builds **no count or congestion feature of any kind** — it is per-row time
differences only. This is not any of the four closed lanes. It is unmeasured, and it is the
first thing to screen.
