# The MSE ledger — what stands between the measured baseline and first place

Every score in this competition decomposes exactly, because the two row classes are disjoint and
their sizes are known from `ranking.parquet` (344,841 scored departures, 5,290 without a Network
Manager off-block):

```
RMSE^2  =  0.9846596 * matched^2  +  0.0153404 * unmatched^2
```

This file is the running ledger. Nothing enters it as a gain without a paired bootstrap interval
that excludes zero. Seconds are not the unit of account — **MSE removed** is, because seconds are
not additive and MSE is.

## MEASURED 2026-09-09 — the first real score, and the fold is PESSIMISTIC

**`merry-quicksand_v2.parquet` scored 301.7019 on 344,841 rows at 2026-09-09 02:10:51Z — rank 26
of 80 teams.** (Our only prior submission, `v1` at 689.6901, was the constant-predictor probe.)
The artifact is the shipped pipeline including the consolidated stratum: no LightGBM, no new
features. It is the model whose `--validate` fold number is 330.22.

**The fold is +28.52 s pessimistic at this model class (+18,021 MSE).** This is the first
model-level calibration point the project has; `HARNESS_CALIBRATION.md` established fold/board
agreement to within +/-5 s at the NAIVE-predictor level only, and that agreement does not carry
to a fitted model.

The probable cause is structural rather than luck: `--validate` holds out Jan+Jul 2025, which
removes from the training fold exactly the two months whose seasonality the evaluation is scored
on. The submitted artifact fits all twelve months and has seen January and July. **Treat
`--validate` as a sound RELATIVE instrument for A/B work and a biased ABSOLUTE one.** Do not
quote a fold total as a predicted board score without this offset, and do not assume the offset
is constant across model classes -- it is one measurement.

**The gap to the leader is therefore 30,158 MSE, not the ~43,000 projected below.**
Board 301.70 (MSE 91,024) against `youthful-giraffe` 246.71 (MSE 60,866).

A claim made earlier on 09-09 and now RETRACTED: "a perfect matched model still scores 231.35, so
the leader is unreachable through the matched lane alone." That was fold-derived. Board-anchored,
under the fold's matched/stratum split ratio (an assumption, not a measurement), perfect matched
implies 211.37, which would beat the leader. The robust surviving claim is only that the stratum
is roughly half of our squared error and remains a large lever. See `reports/GAP_STRUCTURE.md`.

## CORRECTION 2026-09-09 — every "best measured total" in this file was fold-mixed

**Read this before any number below.** The totals quoted as 296.7 and 293.57 paired a **fold-A
matched** number with a **twelve-month LOMO stratum** number (1527.2). Those come from different
validation schemes over different row sets, and the error is not small.

It matters because **the scored months are January and July**, and those are the stratum's two
worst months — per-fold 2,904 (Jan) and 1,317 (Jul) against a 1,527 twelve-month average. For a
Jan+Jul evaluation the fold-A stratum is the correct estimate and the LOMO average is not.

Measured on one consistent fold (`scripts/build_submission.py --validate`, held-out Jan+Jul 2025,
n=344,336):

```
shipped artifact                matched 237.46   stratum 1946.23   TOTAL 337.71
+ consolidated stratum          matched 237.46   stratum 1867.90   TOTAL 330.81
+ LightGBM matched (arithmetic) matched 226.24   stratum 1867.90   TOTAL ~323
```

**Honest position is ~323 on the fold, not ~294.** Gap to the leader's 246.71 is roughly
**43,000 MSE, not 24,431.** Board median is 292.7, so we are currently BELOW median.

Two things support the pessimistic reading: the 2026 stratum's expected MSE is 42,257 against
the fold's 42,289 (ratio 1.00, so the fold is representative in composition), and the
constant-predictor anchor is 686.07 on our fold against 689.69 observed on the board. One thing
pulls the other way: `FAMILY_B.md` §8 puts our fold ~25 s unluckier than average on the Family B
draw. **Honest range for what we would score today: ~315-325.**

**Rule added:** no total is quoted unless every component came from the same fold and the same
run. This is the third instance tonight of pairing numbers from different settings (the ADS-B
projection, the LightGBM smoke, and this), and it is the error class that most misleads.

## Positions

| position | matched | unmatched | MSE | RMSE |
|---|---|---|---|---|
| shipped artifact (`build_submission.py --validate`) | 237.46 | 1946.23 | 113,614 | **337.71** |
| best measured components, not yet consolidated | 230.41 | 1527.2 | 88,036 | **296.7** |
| leader `youthful-giraffe` v13, 2026-09-08 17:21Z | — | — | 61,752 | **248.48** |

**Consolidation is worth 25,578 MSE and is already paid for.** The shipped pipeline carries a
26-feature matched model and the coarse `SP_EDGES` bucket mixture; the L-e logistic
(`STRATUM_MONSTERS.md` §3b) and the ambient residual correction (`RED_TEAM.md` §3) are both
established with intervals excluding zero and neither is wired in. Of that gap, 22,330 MSE is the
stratum alone (1946 → 1527) and 3,248 is the matched side (237.5 → 230.4).

## The budget at the consolidated position (88,036 MSE)

| block | MSE | share | status |
|---|---|---|---|
| matched rows (339,551) | 52,274 | 59.4% | mechanism named, membership unidentified |
| stratum, ordinary rows | 14,256 | 16.2% | ambient correction established |
| 24h date-slip rows | 8,281 | 9.4% | **closed** — see below |
| Family B | 13,225 | 15.0% | irreducible, and **common to every competitor** |

**Family B does not belong in the competitive gap.** Those rows are either present in the 2026
file or they are not; nobody can predict them (`FAMILY_B.md` §10, closed five ways); the leader's
248.48 already contains exactly the same term ours will. It cancels in any comparison between us
and them, so effort spent buying it back is effort spent on a constant. Prior sessions computed
"expected total under the monster load", which is right for forecasting our own score and wrong for
deciding where to spend compute.

## The ledger

**Gap from the consolidated position to 248.48: 26,284 MSE.**

| # | lever | MSE removed | verdict |
|---|---|---|---|
| 1 | mangled-designator / 24h-slip repair | **+305** | **CLOSED — not established** |
| 2 | stand-occupancy / witness block (marginal to E1) | **+350** | **CLOSED — single-seed, label unearned; see §2** |
| 3 | ADS-B stand gate (ten-airport census, Amendment 6) | **+2,243** | **CLOSED — NO-GO, 2 of 10 airports; ingest not justified** |
| 4 | LightGBM at proper capacity (Amendment 7) | **+4,251** | **BANKED — USEFUL band (+4.0%), not the pivot** |
| 5 | stratum consolidation (hierarchical cells + LIRF logistic) | **+4,613** | **BANKED — measured on the fold, 337.71 -> 330.81** |
| | | | see the CORRECTION above: honest fold position ~323, gap ~43,000 |

*(Levers 1-3 are closed and bank nothing; only lever 4 moves the position. Measured total:
296.7 -> 293.57.)*

### 1. Mangled-designator separator — CLOSED (2026-09-08)

`FAMILY_B.md` §9.1 observed that LIRF unmatched rows with a doubled-prefix designator
(`ITYTY680`, `RYRR90BN`) are 91% fill-or-slip against 64%, and carry 9 of the 12 24h-slip rows. It
was never validated out of sample. Under 12-fold leave-one-month-out, with the separator derived
only from `FLIGHT_mvt` (a serve-time column) and added as a cell dimension at LIRF:

```
incumbent (airport x coarse sp bucket, hierarchical k=5)   stratum RMSE = 1644.76
+ mangled dimension at LIRF                                stratum RMSE = 1638.71
paired difference                                                       =   +6.05 s
paired row bootstrap, 2000 resamples, 95% CI                = [-12.87, +26.58]
GLOBAL MSE REMOVED                                                      =     +305
```

The descriptive finding reproduces exactly (166 of 1,488 LIRF unmatched rows mangled, 11.16%;
median `y` 9,090 against 1,800). Only LIRF moves, by +32.2 s on a group RMSE of 4,622.

This is not a tuning gap. LIRF unmatched carries **53% of all stratum SSE**, so the ceiling there
is large — perfect prediction of every LIRF unmatched row would remove 21,942 MSE — but clearing
the 4,000 MSE band requires a **−437 s** improvement at LIRF and the separator delivers −32 s. The
cause is sample size and it is structural: the 24h-slip block rests on **12 rows** in 2025 and an
expected **3.09** in 2026. No separator estimated from twelve examples yields an established gain,
however clean it looks descriptively. **Do not reopen without new information, not a new estimator.**

### 2. Stand-occupancy / witness block — CLOSED (2026-09-08)

Paired A/B, pre-registered in Amendment 5. Fold A, 1,723,425 training rows, 339,015 held-out
matched rows, identical folds/seed/hyper-parameters, differing only by the ten-feature stand block.

```
baseline (68 feats)   235.59      <- E1 reference 235.95, reproduced to 0.36 s
+ stand block (78)    234.84
paired gain            +0.76 s    95% CI [+0.37, +1.14]   <- LABEL WITHDRAWN
GLOBAL MSE REMOVED       +350     of 25,979 needed (1.3%)   -> band: CLOSED (<1,000)
```

**CORRECTED 2026-09-08 (Protocol B).** This entry originally cited a witness-cut decomposition
(`gapa <= 600`: 352.7 → 362.0, "the witness rows got worse") as the mechanism. **That evidence was
noise and is withdrawn.** It carried no interval, in violation of this file's own Rule 1, and
across three fold configurations the same cut gives +1.02 [−9.75, +11.13], −9.25 [−25.51, +5.18]
and −15.84 [−29.35, −2.47] — sign-unstable on 0.5% of held-out rows.

The evidence that *does* demonstrate redundancy is an ablation that was never run at the time —
a fourth arm with no stand information at all:

| arm | features | RMSE | vs A0 | global MSE | 95% CI |
|---|---|---|---|---|---|
| A0 no stand information | 64 | 249.73 | — | — | — |
| A3 stand block, no `prev_arr_*` | 74 | 247.76 | +1.97 s | **+964** | [+1.44, +2.47] |
| A1 `prev_arr_*` only (the baseline) | 68 | 247.13 | +2.60 s | **+1,271** | [+2.19, +3.02] |
| A2 both (the variant) | 78 | 246.89 | +2.84 s | **+1,390** | [+2.30, +3.38] |

**The total stand information is +1,390 MSE, of which the E1 incumbent already banks +1,271 —
69% measured redundancy.** `RED_TEAM.md` §4.3 was right and is now demonstrated rather than
asserted. Note the baseline dependence: against the *shipped* artifact, whose `NUMERIC` list
contains no stand feature at all, the family is worth ~1,271 MSE, not 350.

Note the sample-size lesson in reverse: at two training months the same block was worth +89.9 s on
`gapa <= 600` and +809 MSE overall. With ten months the baseline learns the same thing and the
advantage evaporates. **A gain measured on a weak baseline is not a gain.**

Per Amendment 5's fixed threshold, steps 4–6 of the stand plan (slack GBM, NN slack, leaf-similarity
neighbours) are **not attempted**. The mechanism is real — 39x enrichment, 93–99% witness detection —
and it is already priced into the incumbent.

## The budget was partitioned wrongly — corrected framing

The block table above splits the stratum by ROW FAMILY (ordinary / slips / Family B). That shears a
single mechanism across three lines and books two of them as closed, which is why it never appeared
as a lever. The correct decomposition for the mixture estimator is exact:

```
E[(y - yhat)^2]  =  p(1-p)(sp - m)^2   +   (1-p)*Var(non-fill)
                    ^classification         ^regression
```

Measured on the **actual 5,290 scored 2026 rows**, with cells fitted on 2025:

| estimator | classification | regression | total |
|---|---|---|---|
| shipped (airport × coarse `sp`) | **21,846** | 20,149 | 41,996 |
| airport × fine `sp` | 18,875 | 20,079 | 38,954 |
| + LIRF logistic (L-e style) | 18,439 | 20,208 | 38,647 |

**21,846 MSE — 83% of the whole gap to the leader — is classification variance**, 97.2% of it at
LIRF's 383 rows, 65% in 74 rows, 37% in ten rows.

**But that is a ceiling, not a lever.** It vanishes only under a perfect discriminator; measured
estimators recover **~3,000–3,400 MSE**, and even that is optimistic because the closed form treats
fitted `p` as the true probability, which flatters sharper cells. `STRATUM_MONSTERS.md` §3a rejected
fine buckets under held-out LOMO. **That tension is unresolved** and settling it needs a LOMO test
scored by expected MSE on the 2026 composition, which has not been run.

### 4. Learner capacity — BANKED at +4.0%, and it is NOT the missing mechanism

Every measurement in this project until tonight used one learner: sklearn
HistGradientBoosting, 400 iterations, single seed, untuned. The published solutions of both
prior editions did not — `team_likable_jelly` (2024) used LightGBM with 50,000 trees averaged
over seeds; `team_tiny_rainbow` ensembled four families. Those 50 prior-year repositories had
been an open to-do since `plans/RESEARCH_2026_09_08.md` line 166 and had never been opened.

Fold A, identical rows, target, scoring code and leakage guards:

```
hgb_control   235.59            best_iter    —      resid corr 1.000
lgb_enc       227.67   +3.4%    best_iter 17,557    0.956
lgb_native    233.73   +0.8%    best_iter  2,175    0.939
lgb_refit     226.24   +4.0%    21,948 trees        0.954    +4,251 MSE
```

**Capacity WAS binding** — LightGBM wanted 17,557 trees where HGB used 400–900 — but it is
worth 4%, not the >=10% that would have pivoted the project. Per Amendment 7 section 5 this is
the USEFUL band, so the stand A/B is NOT re-run on LightGBM.

**Native categoricals LOSE to target encoding.** `lgb_native` gained only 0.8% and early-stopped
at 2,175 trees. `STAND_mvt`'s 1,884 levels were never the problem, and sklearn's 255-level cap
was never costing us anything. That hypothesis is closed.

`lgb_refit` at 226.24 beats the previous best matched configuration (230.41, itself a two-model
blend) by 4.2 s. Every airport improves; the `delta < -600` tail moves 733.3 -> 693.2.

**A methodological failure worth recording.** The 3-month smoke of this same experiment showed
+9.1% and was reported as indicative. The full 10-month run gives +4.0%. The smoke's HGB control
was data-starved. This file already contained that exact warning from lever 2 — "a gain measured
against a weak baseline is not a gain" — and it was repeated anyway, on a preliminary number that
the owner then planned around. **No smoke or subsample result is to be quoted as a magnitude
again; only as evidence that the code runs.**

### What the leader's trajectory says, and what it rules out

`youthful-giraffe` is public: v1 265.76 on day one, then two days of 0.2 s increments to 262.74,
then **v13 at 248.48 — a single 14.26 s step (7,290 MSE)** — then polish to 246.71.

Two separable facts. Their OPENING submission was 265.76 while our best measured position after
all of this is 293.57: **~28 s of base-model advantage that learner capacity explains only ~9 s
of.** And there exists a discrete ~7,290 MSE object that at least one team found on 2026-09-08.

Four candidate explanations for that edge are now closed with measurement, not opinion:
stand occupancy (+350), the date-slip block (+305), ADS-B (+2,243, NO-GO), and learner capacity
(+4,251, useful but not it). **Whatever their advantage is, it is none of these.**

## Open, with a mechanism and enough coverage to matter

**ADS-B — CLOSED NO-GO.** The ten-airport census returned 2 of 10 clearing the bars against a
pre-registered threshold of 6. Madrid, Paris, Rome and Istanbul have **zero on-ground ADS-B
samples**; the aircraft are first seen 15-17 minutes after off-block. Worth **2,243 MSE** from EHAM
and LSZH alone, against a ~190 GiB / ~20 h ingest and an open licence question. See
`reports/ADSB_GATE.md`. The earlier ~8,000 projection is withdrawn as a biased extrapolation from
three airports, two of which are the best covered in the network.

**The fold is optimistic by +4.26 s.** Forward (train Jan–Jun, test Jul) against straddle at equal
training volume: A1 273.65 vs 269.39. Every component number in this ledger was measured on an
interleaved fold that is measurably easier than the real task, which extrapolates into 2026.
The forward-fold replication is outstanding and the ledger currently sums numbers from the easier
setting.

**The matched-row tail** — 30.5% of matched SSE sits in 0.61% of rows, and its two mechanisms are
now named and separated by airport and season (`reports/TRAIN_SERVE_AUDIT.md` §3): summer ATFM slot
holds at LIRF/LEBL/EGLL, winter de-icing at EDDM. Weather was tested pooled and returned +0.49 s;
the scored months are January and July, which are precisely where the two mechanisms are at
opposite maxima.

### The seed defect — why "+0.76 ESTABLISHED" was not established

`cmd_fit` hard-coded `random_state=0`. Across three seeds, on identical rows, features and data,
the A1→A2 increment measured **+0.245 / −0.212 / −0.253 — sd 0.28 s, and the sign flips.** The
published half-width was 0.39 s. A single-seed paired row bootstrap therefore certified *fitting
variance* as an established gain, and this file's Rule 1 permitted it. The verdict is unchanged —
every value is far below the 1,000 MSE band — but the label was unearned. `cmd_fit` now averages
three seeds and refuses the ESTABLISHED label unless the gain exceeds twice the measured seed sd.

### Why the stratum intervals are so wide

The stratum's Kish effective sample size on squared errors is **22.9, not 22,219**. The single
worst row is 12.5% of stratum SSE; the top five are 41.1%. A 2,000-resample bootstrap over 22,219
rows is arithmetically a bootstrap over ~23 observations. The mangle interval
[−12.87, +26.58] is honest about that, which is why the closure holds.

## Rules for this ledger

- A gain enters only with a paired bootstrap interval excluding zero, on the same rows,
  **averaged over at least three seeds, and exceeding twice the measured seed sd.**
- Any decomposition offered as *evidence* carries an interval too, not only the headline gain.
- Gains are recorded as MSE, never as seconds.
- A lever whose measured value is below 1,000 MSE is closed, and the reason is recorded here so it
  is not reopened by a later session with the same estimator.
- Family B is never counted as a lever.
