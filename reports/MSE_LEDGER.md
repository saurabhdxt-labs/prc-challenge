# The MSE ledger — what stands between the measured baseline and first place

Every score in this competition decomposes exactly, because the two row classes are disjoint and
their sizes are known from `ranking.parquet` (344,841 scored departures, 5,290 without a Network
Manager off-block):

```
RMSE^2  =  0.98466 * matched^2  +  0.015337 * unmatched^2
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
| | | | **remaining: 25,979** |

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

## Open, with a mechanism and enough coverage to matter

**Stand-occupancy / witness reparameterisation** — mechanism gate PASSED, incremental-value gate
OPEN. See `reports/STAND_OCCUPANCY.md`. Pre-registered with kill thresholds in Amendment 5.

**The matched-row tail** — 30.5% of matched SSE sits in 0.61% of rows, and its two mechanisms are
now named and separated by airport and season (`reports/TRAIN_SERVE_AUDIT.md` §3): summer ATFM slot
holds at LIRF/LEBL/EGLL, winter de-icing at EDDM. Weather was tested pooled and returned +0.49 s;
the scored months are January and July, which are precisely where the two mechanisms are at
opposite maxima.

## Rules for this ledger

- A gain enters only with a paired bootstrap interval excluding zero, on the same rows.
- Gains are recorded as MSE, never as seconds.
- A lever whose measured value is below 1,000 MSE is closed, and the reason is recorded here so it
  is not reopened by a later session with the same estimator.
- Family B is never counted as a lever.
