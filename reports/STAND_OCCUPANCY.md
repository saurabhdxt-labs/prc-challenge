# Stand occupancy as evidence for the negative-delta tail

**Status: the mechanism gate has PASSED. The incremental-value gate is OPEN and is the only thing
that decides whether this is worth anything.** Pre-registered in Amendment 5 with kill thresholds.

## The idea

`ranking.parquet` blanks `BLOCK_TIME_UTC_mvt` on **departure** rows only. Arrival rows keep it,
populated on 100% of 344,693 ARR rows, alongside `STAND_mvt`. So for January and July 2026 we know
when aircraft physically **entered** each stand, measured by the same airport system whose
departure off-block clock is the target.

If a stand is an exclusive resource, an arrival in-blocking there is proof the previous occupant
had gone:

```
BLOCK_dep  <  T_c          T_c = in-block time of the next arrival at that stand
```

and when `T_c << AOBT_3` that is direct evidence of the catastrophic mechanism `RED_TEAM.md` §4.3
named but could not identify: the aircraft left the stand, held somewhere, and NM recorded the
off-block much later. The exact identity is

```
y = (MVT - T_c) + s          s = T_c - BLOCK_dep      ("stand-vacancy slack")
```

so predicting slack **is** predicting the target — a reparameterisation, in the same family as the
`delta` reparameterisation that was worth 32 s, not a new source of information. Its value depends
entirely on whether `s` is easier to predict than `y`.

## Method

Twelve months of 2025, month by month to cap RSS (peak 1.13 GB). Arrivals keyed by
`(ADES_mvt, STAND_mvt)`, departures by `(ADEP_mvt, STAND_mvt)`; per-stand sorted arrival times,
counts by `searchsorted`. Epoch seconds via `(t - EPOCH).dt.total_seconds()` with a tz-aware
epoch. **Stage 1 uses the hidden `BLOCK_dep` to establish whether the physical event occurs; it is
a diagnostic and never a feature. Stage 2 and 3 read only supplied columns.**

> **Landmine.** The composite key was first built with a `\x00` separator. Pandas' Arrow-backed
> string path treats NUL as a terminator: 2,277 distinct (airport, stand) keys collapsed to **10**,
> one per airport, and every count returned zero — an output indistinguishable from a clean
> NOT WORKING verdict. Use `"|"`. The code now asserts `len(idx) > 500` and that the arrival and
> departure key sets join, so this cannot recur silently.

## Stage 1 — the event happens, with 39x enrichment

`reocc` = at least one arrival in-blocked at this exact stand between our off-block and `AOBT_3`.
Tail = `delta < -600` (3.77% of rows; 30.5% of matched SSE per `RED_TEAM.md` §4.3).

| apt | n | tail n | P(reocc\|tail) | P(reocc\|ctrl) | lift | P(tail\|reocc) |
|---|---|---|---|---|---|---|
| EDDF | 228,185 | 3,523 | 9.74% | 0.16% | 61.3x | 49.00% |
| EDDM | 166,324 | 11,886 | 10.45% | 0.55% | 19.0x | 59.43% |
| EGLL | 238,128 | 6,160 | **36.44%** | 0.28% | 128.1x | **77.28%** |
| EHAM | 243,865 | 151 | 34.44% | 0.11% | 306.3x | 15.95% |
| LEBL | 177,838 | 6,748 | 5.26% | 0.24% | 22.2x | 46.71% |
| LEMD | 211,295 | 5,909 | 3.72% | 0.07% | 54.6x | 61.11% |
| LFPG | 235,719 | 9,634 | 7.04% | 0.29% | 24.7x | 51.25% |
| LIRF | 159,214 | 20,674 | 1.67% | 0.02% | 88.9x | 92.99% |
| LSZH | 132,540 | 3,437 | 8.12% | 0.46% | 17.7x | 32.07% |
| LTFM | 269,320 | 9,665 | 8.09% | 0.11% | 73.2x | 73.15% |
| **ALL** | **2,062,428** | **77,787** | **8.41%** | **0.21%** | **39.4x** | **60.71%** |

## Stage 2 — two witness pools, and they are not interchangeable

Anchoring the witness at **take-off** instead of `AOBT_3` enlarges the pool twentyfold, but the
enlarged pool is far less tail-enriched. Both are detectable from supplied columns at high
precision. **Precision below is precision for detecting the witness event, not for detecting the
tail** — the two must not be conflated.

| pool | share of rows | P(tail \| witness) | best serve-time detector | precision | recall |
|---|---|---|---|---|---|
| `AOBT_3`-anchored | 0.53% | **60.7%** | `gap <= 600 & ncnt >= 1` | 93.14% | 71.22% |
| take-off-anchored | 9.82% | 6.6% | `gapm <= 1200` | 99.28% | 94.09% |

## Stage 3 — the bound is too loose to commit to

True slack `s = T_c - BLOCK_dep` on take-off-anchored witnesses: p10 **270 s**, p50 **654 s**,
p90 **1,204 s**. The stand sits empty for about eleven minutes before the next aircraft takes it,
so committing to `y = MVT - T_c` under-predicts systematically.

| apt | n | bound `MVT - T_c` | bound + median slack | per-airport constant |
|---|---|---|---|---|
| EDDF | 25,506 | 675.0 | **301.6** | 350.1 |
| EDDM | 14,821 | 596.6 | **338.9** | 369.6 |
| EGLL | 62,317 | 915.6 | **411.2** | 501.9 |
| EHAM | 18,542 | 635.7 | **347.5** | 399.9 |
| LEBL | 10,608 | 799.9 | 366.1 | **364.0** |
| LEMD | 8,225 | 867.9 | 362.1 | **341.2** |
| LFPG | 16,659 | 839.8 | **435.3** | 569.6 |
| LIRF | 6,518 | 1658.9 | **1201.2** | 1380.2 |
| LSZH | 10,629 | 610.9 | **304.8** | 356.7 |
| LTFM | 27,354 | 835.7 | **421.8** | 516.1 |
| **ALL** | **201,179** | **841.2** | **453.3** | **566.7** |

**Verdict on the vehicle: a hard constraint `min(BLOCK_gbm, T_c - g)` is the wrong form.** It
would fire on 9.8% of rows and bias them by roughly eleven minutes; the bound alone (841) is worse
than a per-airport constant (567). The right form is a **derived feature with a learned offset** —
and `MVT - T_c` is exactly the class of quantity a tree cannot construct from `proxy` and a stand
gap held separately, which is the class that produced this project's largest measured gain
(`MATCHED_MODEL.md` §2, +8.17 s: *"trees cannot form the difference of two inputs"*).

Note also that two airports, LEBL and LEMD, show **no gain over a constant** even with the offset.
There is no single global stand relationship, so a pooled null result must be decomposed by airport
before the idea is declared dead.

## What is NOT established

**Whether any of this removes global MSE.** The comparisons above are against a per-airport
constant, which is not our model. `RED_TEAM.md` §4.3's objection — that `prev_arr_gap` is already
in the E1 feature set and "the model already fits those rows" — is not refuted, and cannot be
settled from records: the E1 feature code lived in a session scratchpad that has been wiped, so
whether its `prev_arr_gap` was anchored at take-off (in which case it *is* this quantity) or at
`AOBT_3` (in which case it is not) is unknown. The shipped `build_submission.py` has no stand
features at all.

Settling it requires the paired A/B of Amendment 5: 70-feature E1-equivalent baseline versus the
identical model plus the stand block, same rows, folds, seed and hyper-parameters.

**Early gate before any NN elaboration.** Unconditional dispersion falls from `sd(y) = 567` to
`sd(s) = 453` on witness rows — a 20% variance reduction, real but modest. The NN/slack work is
only attempted if neighbourhood conditioning takes slack dispersion to <= 300 s.

## Reproduction

`scripts/` does not yet contain this analysis; it was run from session scratchpad scripts
`stand.py`, `stand2.py`, `stand3.py`, whose logic is described above in full and which read only
`data/raw/training_2025-*.parquet`. Promoting them into `scripts/` is part of the stand phase.
