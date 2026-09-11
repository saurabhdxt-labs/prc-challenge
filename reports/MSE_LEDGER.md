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

## MEASURED 2026-09-09 22:33Z — v4: 289.67, rank 19 of 92 — the stratum hybrid (Amendment 18 / RESULT 10)

v4 = v3 with only the 5,290 unmatched rows re-predicted by the hybrid non-fill regressor (fitted body,
cell-estimator tail preserved). **289.6732, −1.96 s / −1,138 MSE.** LOMO pooled stratum gain +16.2 s
(~800 MSE at 1:1); realised 1,138. v5 = seeds + queue block on top of v4 is fitting (RESULT 7 + 9).

## MEASURED 2026-09-09 14:19Z — v3: 291.63, rank 20 of 87, and matched-side fold gains TRANSFER

`merry-quicksand_v3.parquet` = v2 with the 339,551 matched rows re-predicted by the config measured
in `reports/lgbm_ab_full.log` (single pooled LightGBM, seed 0, `lgb_refit`, the 68-feature
`stand_ab` cache; 5,290 unmatched rows byte-identical to v2). Fit: best_iter 25,689 on months (3, 9),
refit 30,846 trees on all twelve months, 56 min, peak RSS 4.51 GB; `reports/lgbm_submit_v3.log`.

| | board RMSE | board MSE | Δ vs v2 |
|---|---|---|---|
| v2 (shipped HGB, 26 features) | 301.70 | 91,024 | — |
| **v3** | **291.63** | **85,049** | **−10.07 s / −5,975 MSE** |
| projection from the fold's 237.46 → 226.24 matched gain | ~293.1 | ~85,901 | −8.6 s / −5,123 MSE |

**The fold's matched-side gain transferred at ~117%.** This is the second calibration point and it
is the one that matters for planning: Amendment 9 says fold TOTALS have no stable absolute value
(a two-row draw sets them); this says matched-side RELATIVE gains, measured on 339k rows, carry to
the board at one-to-one or better. Every remaining lane in the plan is matched-side.

Board-anchored position after v3, holding the implied stratum at 35,502 MSE: matched contributes
~49,547 → implied matched RMSE **224.3** (fold 226.24 — consistent).

## MEASURED 2026-09-09 — the first real score, and the fold is PESSIMISTIC

**`merry-quicksand_v2.parquet` scored 301.7019 on 344,841 rows at 2026-09-09 02:10:51Z — rank 26
of 80 teams.** (Our only prior submission, `v1` at 689.6901, was the constant-predictor probe.)
The artifact is the shipped pipeline including the consolidated stratum: no LightGBM, no new
features. It is the model whose `--validate` fold number is **330.81** (reproduced 2026-09-09).

**SUPERSEDED the same night: the 29.11 s fold-vs-board gap is NOT seasonality. It is the fold's
monster draw, and TWO ROWS carry it.**

The first version of this block attributed the gap to `--validate` holding out Jan+Jul 2025 and so
denying the validation model the evaluation months' seasonality. Measured, that story is neither
needed nor supported:

| | MSE |
|---|---|
| gap the seasonality story had to explain | 18,412 |
| carried by the fold's top TWO unmatched rows | **29,511** |

Those two rows are LFPG, January, y = 84,240 s (23.4 h) and 58,206 s (16.2 h) — both with small
`sp` (1,740 / 2,043), so nothing flags them; the model predicts 1,090 and 1,216. **They are 54.7%
of the fold's entire stratum SSE and 27.0% of its total MSE**, and the single worst row is 18.3% of
the fold by itself. Replacing them with an expected Family-B load of 13-17k MSE puts the fold at
304.8-311.3 against the board's 301.70 — a residual of +3 to +10 s, against the 29 s that had been
attributed to a mechanism.

`reports/FAMILY_B.md` §8 already recorded that this fold is roughly 25 s unluckier than average on
this family. It was read and not subtracted. That is the actual failure here: a known correction
sitting in the repo went unapplied, and a mechanism was invented to explain the residue.

**The operative consequence is STRONGER than the one first written.** It is not that the fold
carries a +28.52 s offset to correct for. It is that **the fold's TOTAL has no stable absolute
value**: it is set by a two-row draw from a heavy tail, so its absolute uncertainty is tens of
seconds. Carry no offset, across model classes or anything else. `--validate` remains sound for A/B
on MATCHED rows, where n = 339,015 and no such concentration exists; its TOTAL should not be quoted
as a predicted board score at all.

**A number this file introduced and now retracts: 330.22.** The instrument prints **330.81**.
330.22 was a re-weighting of the fold's components by the 2026 scored-file weights, performed in
analysis and then quoted as though it were the instrument's output.

**The gap to the leader, measured like-for-like.** A review on 09-09 caught that the first
version of this block compared 30,158 against ~43,000 — which is itself a configuration mix, the
very error this file exists to correct. The ~43,000 is a **LightGBM-inclusive fold** number
(322.37^2 - 246.71^2 = 43,056); the 30,158 is the **shipped, non-LightGBM board** artifact. Stated
without mixing:

| artifact | instrument | total | gap to 246.71 |
|---|---|---|---|
| shipped + consolidated stratum | fold | 330.81 | 48,594 MSE |
| shipped + consolidated stratum | **BOARD (measured)** | **301.70** | **30,158 MSE** |
| + LightGBM matched | fold | ~322 (arithmetic, never run end-to-end) | ~43,000 MSE |
| + LightGBM matched | board | **not measured** | — |

The like-for-like statement is the first two rows: **for one artifact, the fold overstates the gap
by 18,021 MSE.** Any board figure for the LightGBM configuration is a projection until submitted,
and projecting it requires assuming the offset carries across model classes, which is unevidenced.

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


## Board entries · 2026-09-09 evening (the board is the instrument; fold numbers are relative gains only)

| version | board RMSE | board MSE | Δ MSE vs prev | lane | fold projection (matched MSE × w_m) | transfer |
|---|---|---|---|---|---|---|
| v3 | 291.6317 | 85,049 | −5,975 vs v2 | LightGBM on the E1 cache, matched rows | −5,123 | 1.17× |
| v4 | 289.6732 | 83,911 | −1,138 | stratum hybrid (Amendment 18), unmatched rows only | not projected (stratum) | — |
| **v5** | **288.1406** | **83,025** | **−886** | seeds (RESULT 7) + queue block (RESULT 9), matched rows only | −910 | **0.97×** |

Rule confirmed a third time: a matched-side paired fold gain, in MSE, lands on the board at ~1:1.
Stratum gains are not projected (Amendment 9). The 10th place at 20:45 was 276.51 (= 76,458 MSE):
**6,567 MSE below v5**, and the cut is still falling. Every remaining matched-side gain must be
measured as a paired interval on the fold and then banked here in MSE, never in seconds.

---

## 2026-09-10 01:45 — three findings that change what is open, and one that changes what is trusted

Added after the owner's audit (`reports/top_path_audit.json`, `plans/TOP_PATH_2026_09_10.md`)
established that several lanes this ledger records as closed were closed on evidence narrower than
the conclusion drawn. Nothing here is a banked gain; two are defects and one is a re-pricing.

**1. A validation defect, in this ledger's own instrument.** Target encodings for the
early-stopping rows were fitted on data including those rows' own labels
(`lgbm_fold.py:1464`, and `fold_masks` at :657 puts ES inside `tr`). Measured on 861,290 real rows:
**24 of 24 encoding columns on ES rows move when ES labels move.** Outer-holdout numbers and every
paired interval in this file remain valid — the holdout was never contaminated — but **`best_iter`
was selected against a biased signal for every arm ever run, and for every shipped version**
(`lgbm_submit.py:1452`). Full sibling list and scope: `reports/bug_classes.md` BC-1. Repair built
and tested in `prc/encoding.py`; NOT wired, because the arms queue is live against those files.

**2. The fill lane was mispriced. The fill branch is worth +1,782 MSE with a broken blend on top.**
RESULT 8's arm was `p·sp + (1−p)·m(X)` where `m` is the all-rows model — already `E[y|X]`, itself
the mixture — so it applies fill behaviour twice. Recomputed per subset from its own record:

| subset | rows | baseline | treatment | MSE |
|---|---|---|---|---|
| fill (`F=1`) | 30,167 | 288.725 | **251.047** | **+1,782** |
| non-fill | 308,848 | 218.714 | 223.051 | **−1,719** |
| pooled — what got reported | 339,015 | 225.825 | 225.683 | +63 |

*(Both arms carry the shipped `max(proxy − pred, 1)` floor. A first version of this block floored
only the treatment and read +1,789 / −1,534 / +256; corrected 2026-09-10 02:05.)*

The classifier is sound (holdout AUC **0.8957**, decile fill rates monotone 0.000 → 0.416). A
pre-registered gate test (`plans/PREREG_fill_gate_2026_09_10.md`, RESULT G) confirmed the mechanism
on all three clauses — the real rule's gain is **12× the p95** of the same-size random-subset null —
and recovered **+322 MSE**, only 18% of the prize, because `p` is mis-calibrated (mean 0.297 on true
fills against an 8.90% base rate, so a 0.5 threshold has 10.4% recall). Harness for the fitted
design: `scripts/cond_experts.py`. Detail: `reports/PRIORITY2_FILL_LANE_PRICED.md`.

**3. The encoding keys are not airport-scoped.** Stand `210` exists at six of the ten airports;
**55.3% of rows carry a shared `STAND_mvt` name, 44.6% a shared `RUNWAY_mvt` name**, and ~~`ars` is
`STAND|RUNWAY` with no airport in it (`stand_ab.py:340`)~~ **— false, corrected 2026-09-10 06:40 in
`reports/PRIORITY1_IDENTITY_SCREEN.md` §2 and carried here only now: `ars` is `<airport>|<stand>|<runway>`
on 100% of cached rows.** Single-key out-of-fold, scoped against
unscoped: aircraft type **−24.1 s**, destination **−19.9 s**, stand −6.9, runway −2.3. Single-key
magnitudes do not transfer to a 68-feature model that already carries `ADEP_mvt`; what they
establish is that the statistic is corrupted on a majority of rows. Cheapest open bet — no new data,
no new cache. Harness: `scripts/enc_ab.py`. Detail: `reports/PRIORITY1_IDENTITY_SCREEN.md`.

**A retraction, recorded against this file's own framing.** `reports/WHERE_WE_STAND_2026_09_10.md`
first claimed a ~250 ceiling for any approach not identifying extreme rows. **False.** Perfect
prediction on the 25,870 matched rows with `|delta| > 600 s` removes **23,901 MSE** and would score
**243.15** — beating the leader, with no extreme row involved (max `y` in that set is 15,059 s). The
error was pricing two convenient oracles and generalising their sum into a bound. Retraction and
root cause are at the top of that file.


---

## 2026-09-10 late morning — the gap located, and Rome rebuilt (nothing banked until the board says so)

Detail: `reports/GAP_LOCATED_2026_09_10.md`, preregs `plans/PREREG_rome_{fill,dateslip,matched_fill,body}_2026_09_10.md`.

- **The gap to the 271–274 tier is our unmatched lane at Rome.** On July 2025, against a public GPLv3
  system (elegant-alligator, board 274 → 271) that also held July out: the unmatched-lane SSE gap to
  our SHIPPED S1 is 1,145 M = 6,006 MSE in July's frame — the whole July all-rows gap — and LIRF alone is
  1,210 M (the other nine airports are 65 M better than theirs).
- **The leader's one-submission jump (262.7 → 248.5, 09-08) is one ~50,000 s row;** the LIRF 24 h
  date-slip row that public arithmetic puts at ≈ 84,200 s is already ≈ right in v5 (84,904).
- **Arm R** (stake-weighted LIRF fill classifier): NOT WORKING as registered (month interval spans
  zero; fold A −27.1%). **Arm R2** (R + the 24 h date-slip class): **all registered clauses pass**
  (LOMO −15.2%, fold A −24.1%), July Rome 3,925.6 → 3,459.5 vs the public 3,437.9; the date-slip part is
  boundary-fragile (post-hoc sensitivity). Projected ≈ 4,150 board MSE — **projection, not banked.**
- **Arm M** (fill blend over arm F on LIRF matched rows): NOT WORKING / INCONCLUSIVE — the body pays
  (RESULT 8's double counting, caught by its registered clause). Arm B (a conditioned body refit) is
  registered and queued behind the v6 build.
- **Arm FW**: code built and tested (`--orderfeats --weatherfeats --queue`), NOT run — ≤ ~300 MSE, not
  worth the heavy slot against the Rome work.

## MEASURED 2026-09-10 13:13–13:14 — v6 and v7 on the board

| version | board RMSE | board MSE | Δ MSE vs prev | lane | fold projection | transfer |
|---|---|---|---|---|---|---|
| **v6** | **286.9017** | 82,313 | **−712** vs v5 | arm F (record ordering), matched rows only, base v5 | −745 | **0.96×** |
| **v7** | **285.8013** | 81,682 | **−630** vs v6 | Rome R2 (stake-weighted fill classifier + 24 h date-slip), 382 LIRF unmatched rows only | ≈ −4,150 (fold A) | **≈ 0.15×** |

- The matched-lane rule holds a fourth time (0.96–1.17×).
- **The Rome unmatched projection did NOT transfer.** R2 helped (−630 MSE, the right sign) but delivered
  ~15% of its fold-A projection. The projection rested on a few extreme 2025 rows each worth thousands
  of MSE; 2026's Rome rows evidently differ (a public team reports the same about LIRF's tail). Rule
  from today: **an unmatched/extreme-row fold projection is not a board forecast** — price such lanes
  at a steep discount until a board reading calibrates them.
- Rank 21 of 103 unchanged; 10th 275.09, 5th 266.40. **274 now needs −6,606 MSE.**

## 2026-09-10 14:46–14:50 — arm B (Rome) scored; E1 registered and v8 built (nothing banked)

- **RESULT B** (`plans/PREREG_rome_body_2026_09_10.md`, `reports/rome_body.json`): B +7.76 s [3.59, 12.18],
  B_hyb +6.63 s [3.44, 10.08] on LIRF matched fold-A rows — both **INCONCLUSIVE**: clause 4 fails
  (non-fill rows +3.49 / +4.83 s vs the 1.0 s bound). Projected +438 / +375 board MSE. The conditioned
  body ALONE beats F on LIRF non-fill rows by 21.2 s (298.55 vs 319.70); the body loss is the mixture
  weight `p` on non-fill rows, not the body. B.1 / B.2 (all airports) pending on AC power.
- **E1 registered** (`plans/PREREG_rome_bandfloor_2026_09_10.md`, from `reports/PATH_TO_245_FABLE_2026_09_10.md`,
  re-verified): 2025 has no LIRF unmatched row with `sp ≥ 24,000` and `y < sp − 60` (56 rows). v8 = v7 +
  `max(v7, rint(sp))` on LIRF unmatched `24,000 ≤ sp < 86,400`: 13 rows move, two carry 5,612 of 5,627 MSE.
  **A two-row bet:** −5,627 if both fills, +7,012 if both ordinary. Priced on the 2026 file's own
  composition, never on a fold draw. Not uploaded.

## 2026-09-10 14:50 — arm B (conditioned body): a Rome-only effect, ≈ 400 MSE, not shortlisted
Gated all-airport mixture +379, continuous +314 weighted fold MSE (clause 4 fails for all four arms; none
reaches the 1,000 shortlist rule). Nearly all of it is LIRF, where a body fitted without fill rows beats F on
non-fill rows by 21 s; elsewhere ±1.4 s. Detail: `plans/PREREG_rome_body_2026_09_10.md` RESULT B.

## 2026-09-10 15:43 — E3C (congestion-aware unmatched body): WORKING on 12-month LOMO (not yet shipped)

`plans/PREREG_unm_congestion_2026_09_10.md` RESULT E3C, `reports/unm_congestion.json`. S1's unmatched body
had no congestion input; adding the same-file airport-hour median of matched `MVT − AOBT_3` passes all
seven clauses (12/12 months, 7/9 airports, calm control, ex-monster, seeds; S1 reproduced bit-exact).
**2026-priced +3,960 board MSE (registered); post-hoc sensitivities +1,664 to +2,299** because the 2026
stake is EHAM's January disruption and the 2025 evidence for the hottest bin is mostly LTFM Feb 2025.
Priced on the 2026 file's own composition; the board will say. Ship path not yet built.
- **15:49 — arm B B.1 / B.2 (all airports): both INCONCLUSIVE, +379 / +314 projected, NOT shortlisted**
  (`reports/rome_body_all.json`). Gated form fires almost only at LIRF; the continuous form loses at six
  airports through `p` on body rows. Arm B closed at this scale; the body expert survives as a component.
- **15:42–~15:59 — an uncoordinated `catboost_native.py fit --target delta` (launched from another session,
  not this one; no memory probe, no gate here) ran to iteration 2,000 and exited; no prediction written.
  NOT a result — no number from its log is quoted anywhere. Owner: PRC heavy jobs only from prc-challenge-25
  for now. The C_delta / C_sched prereg stands; rerun deliberately later.**

## MEASURED 2026-09-10 16:08 — v9 on the board: 282.6790 (rank 15 of 106)

| version | board RMSE | board MSE | Δ MSE vs v7 | lane | fold projection | transfer |
|---|---|---|---|---|---|---|
| **v9** | **282.6790** | 79,907 | **−1,775** | E3C congestion-aware unmatched body, 4,828 non-LIRF unmatched rows, base v7 | +3,960 registered / +1,664–2,299 event-excluded / +800 unbiased floor | **0.45× registered, ≈ 1.0× event-excluded** |

An unmatched-lane mechanism (not a row draw) transferred. v8 (E1) is built and NOT yet uploaded. 274 now
needs −4,831 MSE; 5th (266.40) −10,935.

## MEASURED 2026-09-10 16:13 — v10 on the board: 281.5182 (rank 15 of 106) — E1 PARTIAL

| version | board RMSE | board MSE | Δ MSE vs v9 | lane | pre-written shapes | verdict |
|---|---|---|---|---|---|---|
| **v10** | **281.5182** | 79,252 | **−655** | E1 LIRF schedule floor (13 rows), base v9 (rows disjoint from E3C) | −5,627 both fills / −680 one / +7,012 none | **PARTIAL** |

Today: v7 285.8013 → v9 282.6790 (E3C −1,775) → **v10 281.5182** (E1 −655). Total −2,430 MSE this afternoon.
274 now needs −4,176 MSE; 5th (266.40) −10,280. The E1 reading is used for nothing else (pledge).
- **16:50 — E3R (offset-to-witness unmatched body): NOT WORKING** (`reports/unm_congestion_resid.json`; C1 interval
  spans zero, 6/12 months, 5/9 airports, calm hours lose). Pooled +1.6 s; the ≥ 2,400 bin gains, everything else
  pays. Not shipped. The E3C winsorised body stays the shipped unmatched body.
- **16:48 — matched-lane congestion bias checked (research lead, re-verified here):** arm F's mean residual by
  airport-hour witness bin is −1.5 / −8.6 / +1.8 / −18.5 / −12.8 / −5.1 s (and −120 s on 133 rows ≥ 2,400) — the
  matched model already carries congestion; the E3C analogue on matched rows is not a lever.
- **Correction (16:57) to the 15:42 line above:** the stopped C_delta run was registered-and-gated by the earlier PRC
  session (prc-challenge-90: real-data probe 14:50 peak 3.67 GB, an `assemble_output` name-collision bug fixed with a
  test, full input 14:53, gate recorded in `plans/PREREG_catboost_native_2026_09_10.md`); it was uncoordinated with this
  session, not ungated. Its log is kept as `reports/catboost_native_delta_STOPPED_1542.console.log`; not a result.
  C_delta relaunched 16:57 from this session.
- **18:28 — E4 (ADS-B EHAM, AOBT_3-referenced) cannot return WORKING by design (C3 failed on the 2025 census; recorded
  pre-ingest); E4b (label-referenced, validated on unseen 2025-12-27/28): NOT WORKING** — V1 matched RMSE 487.6 s vs a
  200 s bar, driven by a handful of gross errors (without the top 20 of 914 rows: 116 s); V2 on 30 unmatched rows
  119 s vs S1C 397 s. No 2026 ingest. A guarded successor would need fresh validation days.
- **18:49 — CatBoost native C_delta screen: NOT SHORTLISTED.** Alone −1,422 (225.78 vs F 222.56, worse at 9/10 airports);
  fixed 0.5 blend with F **+533** [+0.83, +1.61 s], gain carried by the |delta| > 10 min tail (+5.2 s) and LIRF (+4.9 s).
  Below the 1,000 compute bar; blend-on-inner-OOF is the only path left for this learner.
- **19:20 — RESULT E4b is VOID** (instrument defect, found by an independent review and re-verified): the matched join
  picks the same-callsign INBOUND leg on some rows; 7 such rows = 91.8% of V1's SSE. Not a verdict on the sensor.
  A fresh E4c (take-off-anchored join, regression test, fresh validation days) is required before any ADS-B claim.
  The same review: matched-lane paired bootstraps are row-iid (intervals ≈ 4.6× too narrow) — no past verdict flips,
  but new matched arms use a day-block interval.
- **Correction (19:40) to the 19:20 line:** "row bootstrap ≈ 4.6× too narrow" is the independent review's figure
  (project memory `project_prc_full_review_2026_09_10.md`, arm F's interval; its raw JSON was in a wiped scratchpad) and
  was NOT independently verified here. TOP_PATH's own date-block check on the queue block gave ≈ 1.5× ([+1.186, +1.954]
  vs row [+1.316, +1.812]) — the factor is likely lane-specific. phantom-forecaster-22 is re-intervalling its arm
  records with a day-block bootstrap from the stored parquets; the per-arm ratios will be recorded when they exist.
- **19:55 — day-block re-interval of the matched arms (phantom-forecaster-22, RESULT 23 in its prereg; `tools/dayblock_reinterval.py`;
  62 dates, Jan / Jul resampled separately, 2,000 draws; reproduces TOP_PATH's independent queue-block interval):**
  width ratio day-block / row — queue 1.59×, **arm F 4.60×** ([+0.633, +3.327] s), W 1.58×, D 1.47×, Y blend 1.02×, A27 0.98×.
  The review's 4.6× is therefore CORRECT for arm F and lane-specific (F's gain lives on held-row tail days). F still passes
  every clause under date blocks; its board value is real but wide (≈ 280–1,500 MSE, not a tight ≈ 745). Weakened: W's gt20
  "worse" → no better, not worse; Y-alone's tail gains do not survive (Y-blend's do). Supersedes the 19:40 correction's
  "unverified". **Rule from here: any tail-targeted or matched arm is intervalled by day blocks** (as CAP's C1 already is).
- **Timestamp correction (2026-09-10 20:47:51, from `date`):** stamps in this session's appended blocks today (e.g. "19:20", "19:40",
  "19:55", and prereg amendment stamps) were typed estimates, not clock reads. Where a log fixes the order it is noted in
  the prereg (CAP.1 ≤ 19:18:11, before the fit). From now on every stamp is taken from `date`.
- **2026-09-10 21:15:57 — CAP (127,20,0.6 on arm F's design): ESTABLISHED, net +253 weighted fold MSE, NOT shipped** (ship bar +500).
  Day-block [+0.32, +0.82] s; all three seeds beat F's; fill and tail improve. A real, small, spread gain — worth ≈ +250 on the
  board at the matched lane's 1:1 transfer if a ship is separately registered.
