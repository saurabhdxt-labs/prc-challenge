# Where the gap to the leader actually is

> **AMENDED 2026-09-09 02:15Z, after the first real score.** `merry-quicksand_v2` scored
> **301.7019** on the board (rank 26 of 80) against a `--validate` fold number of 330.22 for the
> same artifact. The fold is **+28.52 s pessimistic** at this model class, most likely because
> `--validate` holds out Jan+Jul 2025 and so denies the validation model the very seasonality the
> evaluation months carry. Consequences for what follows:
>
> * **Gap to the leader, like-for-like:** the SHIPPED artifact's gap is 48,179 MSE on the fold and
>   **30,158 MSE on the board**. The ~43,000 quoted elsewhere is the LightGBM-inclusive FOLD gap
>   (43,056) and must not be compared against 30,158 -- different configuration AND different
>   instrument. A first version of this amendment made exactly that comparison; it is corrected in
>   `reports/MSE_LEDGER.md`. **The LightGBM configuration has no board number at all.**
> * **The claim "perfect matched scores 231.35, so the leader is unreachable through the matched
>   lane" is RETRACTED** wherever it appears -- Section 0 and Section 2 both state it. It is
>   fold-derived. Board-anchoring it requires assuming the offset splits proportionally across the
>   two lanes; under that ASSUMPTION perfect matched implies 211.37 and would beat the leader.
>   The assumption is untestable from the board, which reports only a total, and three defensible
>   ways of applying the offset (RMSE-additive, MSE-additive, proportional) give 202.8, 188.4 and
>   211.37. **No board-anchored value for perfect-matched should be quoted.** The claim is simply
>   withdrawn, not replaced.
> * **What survives:** the stratum is roughly half of our squared error (49.1% for the artifact
>   that scored), and Sections 1, 5, 6 and 7 rest on direct reads of the data rather than on fold
>   totals. Section 3's "1% is worth the same in either lane" survives only as a FOLD statement --
>   carrying it to the board needs the same proportional-split assumption, so it is not established
>   board-side.
> * **Section 6 is directionally contradicted by this result and is NOT unaffected.** It argued
>   from a heavier 2026 `sp` tail that the fold would prove optimistic; the board came in 28.52 s
>   BETTER. The `sp` measurements in Section 6 are correct as measurements; the inference drawn
>   from them about which way the fold would err was wrong, and the seasonality mechanism above
>   evidently dominates. Section 6 is retained for its data and its conclusion is withdrawn.
> * **Section 5's "lane closed" is downgraded to "the cheap version is refuted"** -- see the
>   status paragraph in that section.
> * Every fold total in this file is an absolute number and inherits the offset. The A/B
>   comparisons do not.

Measured 2026-09-09. Every number below came from one consistent fold (held-out Jan+Jul 2025,
n=344,336) or from a direct read of `data/raw/`. Nothing here is a subsample magnitude. On mixing: the
Section 2 table applies 2026 weights to 2025 fold RMSEs (quantified in that section), and a first
version of the amendment below compared a LightGBM-inclusive fold gap against a shipped-artifact
board gap. Both are corrected in place rather than silently rewritten. Scripts ran `OMP_NUM_THREADS=1 nice -n 19 python3.11 -B`.

## 0. The one-paragraph version

The scored file splits into 339,551 **matched** rows (a Network Manager off-block exists) and 5,290
**unmatched** rows (1.53%). Those 5,290 rows carry **49.1% of the squared error of the
artifact that actually scored** (51.5% for the not-yet-submitted LightGBM configuration).
~~A *perfect* matched model — RMSE exactly 0 — would still score 231.35, only 15 s better than the
leader's 246.71, and the 43,000 MSE gap and our stratum deficit are the same number.~~
**RETRACTED — both sentences are fold-derived and do not survive board anchoring; see the
amendment above.** What survives: holding matched fixed, the stratum is the single largest block of
our squared error, at roughly half. All
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
the eight unread (30 - 22) are `FLIGHT_ID_mvt`, `CALLSIGN_flt`, `ADEP_flt`, `ADES_flt`, `ADES_FILED_flt`,
`AIRCRAFT_TYPE_flt`, `ARVT_1_flt`, `ARVT_3_flt`. No `AOBT_1`/`AOBT_2` exists. Hypothesis closed.

## 2. The budget, on one consistent fold

Weights are exact from `ranking.parquet`: w_matched = 0.984660, w_unmatched = 0.015340. **These are
the 2026 scored-file weights applied to 2025 fold RMSEs**; the fold's own shares are 0.984547 /
0.015453. The difference is third-decimal and moves no conclusion, but the header's claim that "no
total mixes components from different runs" is not literally true of this table, so it is corrected
here rather than left standing. Row counts likewise differ by 3 between the header's n=344,336 and
Section 2 + Section 5 (339,012 + 5,321 = 344,333), because Section 1's matched cut applies a plain
0 <= y <= 3h filter rather than the pipeline's `admissible()`.

| position | matched | stratum | total | matched share | stratum share |
|---|---|---|---|---|---|
| shipped artifact | 237.46 | 1946.23 | 337.09 | 48.9% | 51.1% |
| + consolidated stratum | 237.46 | 1867.90 | 330.22 | 50.9% | 49.1% |
| + LightGBM matched | 226.24 | 1867.90 | **322.37** | 48.5% | **51.5%** |

~~**Perfect matched (m = 0) scores 231.35**, so the leader's 246.71 is unreachable through the
matched lane alone.~~ **RETRACTED, fold-derived — see the amendment.**

**Which row actually scored:** v2 is the middle row (matched 237.46, stratum 1867.90). Its board
total is 301.70; its fold total is 330.22.

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
| 292.08 (median of the 200 most RECENT submissions — NOT the board median, which is 329.6) | 174.6 | 1508.6 |
| 265.76 (`youthful-giraffe` FIRST submission) | 131.8 | 1148.3 |
| 246.71 (leader, current) | 86.4 | 826.0 |

Matched 86.4 means explaining 96% of `delta`'s variance; stratum 826 means 95.7% of the unmatched
variance on rows whose p99.9 is 21.6 hours. **Neither single-lane story is plausible.** A joint
reading is: matched 200 + stratum 1183 gives 246.69 — a 12% and a 37% relative gain. That is the
shape to aim at, and it is reachable.

**Caveat added 09-09:** every number in this section is fold-framed, and the fold is now known to be
a biased absolute instrument. Board-anchoring this table requires assuming the +28.52 s offset
splits proportionally across the two lanes, which is unevidenced — the board reports only a total,
never a split. Treat this section as showing the SHAPE of a winning configuration, not its
coordinates.

## 5. The stratum is a mixture, and the obvious lever is WEAKER THAN IT LOOKS

`fit_unmatched` predicts `p*sp + (1-p)*nf`, where a "fill" row is one whose airport stamped the
scheduled push into the off-block field. The tempting hypothesis was that this is a hidden binary
classification problem: identify the fills and the error collapses. **Measured on the fold, it is not.**

| predictor on the 5,321 fold unmatched rows | RMSE |
|---|---|
| shipped mixture | **1867.9** |
| ORACLE fill-flag + constant for non-fills | 3080.5 |
| ORACLE, restricted to fill rows only | 24.5 |
| constant, restricted to non-fill rows | 3156.5 |

A perfect fill classifier paired with a CONSTANT non-fill predictor makes things **worse**, because
the mixture's hedging is doing real work and the non-fill rows are the monsters: sd(y) = 3975,
p95 = 2,528 s, p99.9 = 77,903 s, max 88,132 s (24.5 hours). Fills are only 4.8% of the stratum. The
shipped mixture already explains 78% of the unmatched variance.

**What this does and does not establish.** It establishes that *classification alone is not the
lever*: even a perfect flag loses to the mixture when the non-fill branch is a constant, so any
work that improves only the classifier is not worth doing. It does **not** establish that
classification cannot help at all — the oracle arm pairs a perfect classifier with the weakest
possible non-fill regressor, so it conflates the two. The untested arm is *oracle flag + a fitted
non-fill model*, and that arm bounds the whole lane. A first version of this section said "lane
closed"; a review on 09-09 correctly judged that beyond the evidence. **Status: the cheap version
of this lane is refuted; the lane itself is NOT closed until the oracle-flag + fitted-non-fill arm
is measured.** That measurement is cheap and should precede any further stratum work.

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

`sp` is the fill branch's entire prediction, so a heavier `sp` tail should make the 2026 stratum
harder than the fold. On that basis this section originally concluded: *"our ~322 fold position is
optimistic on the stratum side, not pessimistic."*

**That conclusion is WITHDRAWN — the board refuted it within the hour.** v2 scored 301.70 against a
330.22 fold, i.e. 28.52 s BETTER, not worse. The `sp` measurements above are correct and still
worth knowing; the directional inference from them was wrong. Whatever the heavier `sp` tail costs
is evidently smaller than what the fold's holdout of Jan+Jul 2025 costs the validation model, and
the two effects work against each other. The honest position: **the fold and the board differ for
at least two identified reasons pulling in opposite directions, the net is +28.52 s in our favour
at this model class, and neither component is separately calibrated.** This is a stronger version
of the warning the deliberate `test_stationarity` failures carry — the fold and the evaluation
differ in composition, and the sign of the net effect is not predictable a priori.

## 7. Unexploited information, named

`ranking.parquet` is 689,534 rows: our 344,841 scored departures **plus 344,693 arrivals**. On the
arrival rows `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are **fully populated** (taxi-in, mean
537 s). Every scored departure's `MVT_TIME` (take-off) is also known. So the realised departure
sequence and the contemporaneous arrival stream are both observable inside the evaluation window,
and `derive()` currently builds **no count or congestion feature of any kind** — it is per-row time
differences only. This is not any of the four closed lanes. It is unmeasured, and it is the
first thing to screen.
