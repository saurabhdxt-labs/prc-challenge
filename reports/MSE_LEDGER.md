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
| | | | **remaining: 25,629** |

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

## Open, with a mechanism and enough coverage to matter

**ADS-B with a learned stand gate — see `reports/ADSB_GATE.md`.** The strongest lever measured:
the gated sensor reaches RMSE 127 at EHAM against the model's 173, on 65.8% of rows. Projection
≈2,700 global MSE from three airports measured, ≈8,000 if the other seven behave alike. One day,
three airports; the ten-airport census is the next step.

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
