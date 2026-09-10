# Where we stand, what we tried, and what it would take to beat 246

> ## RETRACTION — 2026-09-10 00:55 local, before this file was acted on
>
> **§2's central claim was wrong and is withdrawn.** It said: *"the best conceivable score without
> capturing a single extreme row is ~250"*, and concluded that beating 246 requires row-level
> identification of the extreme rows. **Both statements are false.**
>
> The error was a category mistake in the experiment, not in the arithmetic. I priced two *narrow*
> oracles — perfect knowledge of the `<2 min` band, and perfect ordinary-unmatched prediction — and
> then treated their sum as an upper bound on all non-monster work. It is not an upper bound on
> anything. It is the value of two specific interventions.
>
> The refutation is one line, and it contains **no monster rows at all**:
>
> | oracle, matched lane only | rows | MSE removed | resulting RMSE |
> |---|---|---|---|
> | perfect on `|delta| > 600 s` (max y = 15,059 s — ordinary rows) | 25,870 (7.63%) | **23,901** | **243.15 — beats the leader** |
> | perfect on all matched rows | 339,015 | 49,520 | 183.04 |
>
> A second, independent error of the same kind: I argued that because our model beats a perfect
> `|delta|`-band-mean oracle (224.26 vs 392.72), there is "no hidden class left in the matched rows."
> That oracle **replaces** the model's inputs with band membership instead of **adding** membership
> to them. The correct comparison is `model(X)` against `model(X + oracle membership)`. Beating a
> band-mean predictor says the coarse partition alone is uninformative; it says nothing about the
> ceiling, and nothing about whether membership information helps a model that keeps X.
>
> **What survives §2:** the measured decomposition (§1), the concentration figures, the monster
> census (§3), and the closure that hedging cannot recover extreme rows (H-F). **What does not
> survive:** every sentence containing "ceiling", "not reachable", "arithmetically closed", and the
> scenario table's framing as a bound. The corrected position is in §6, and the work that follows
> from it is in `plans/TOP_PATH_2026_09_10.md`.
>
> Root cause, recorded against myself: I ran oracles that *substitute* for the model rather than
> *augment* it, and I did not sweep the obvious subsets (`|delta| > 600`, per-airport, all-matched)
> before publishing a bound. A bound claim requires the maximal oracle, not a convenient one.

**Written 2026-09-10 00:45 local.** Every number in this file is either measured in this repository
or arithmetic over measured numbers, and each is labelled with its instrument (BOARD or FOLD).
Where a number is fold-derived it inherits Amendment 9's warning: **fold TOTALS have no stable
absolute value** — they are set by a two-row draw from a heavy tail. Fold *shares* and fold *paired
relative gains* do transfer; fold totals do not. Nothing below is a board projection.

---

## 1. The position, in numbers

| | RMSE | MSE | source |
|---|---|---|---|
| our best, `merry-quicksand_v5` | **288.1406** | **83,025** | BOARD, rank 19 of 95 |
| 10th place | 275.94 | 76,143 | BOARD, 2026-09-10 00:07 |
| leader | 245.29 | 60,167 | BOARD, 2026-09-10 00:07 |

- **Gap to tenth: 6,882 MSE.** It widens ~0.5–1.5 s/day as the cut drifts down.
- **Gap to first: 22,858 MSE.** That is 3.3× the gap to tenth.

Our score decomposes exactly, because the two row classes are disjoint and their sizes are known:

```
RMSE² = 0.9846596 · matched² + 0.0153404 · unmatched²
```

Anchoring the matched side at its measured fold value of 224.258 (justified: RESULT 5/9/10 show
matched-side paired gains land on the board at 0.97–1.17×, three calibration points):

| lane | rows | share of rows | board MSE | share of our error |
|---|---|---|---|---|
| matched (a Network Manager off-block exists) | 339,551 | 98.47% | ~49,520 | 59.6% |
| **unmatched** (no NM record at all) | **5,290** | **1.53%** | **~33,505** | **40.4%** |

The implied board unmatched RMSE is ~1,478, against the fold's 1,868. **The 2026 file's unmatched
lane is materially lighter than our fold's** — about 20,000 MSE lighter in the same frame. That is
consistent with the known fact that our fold drew two exceptional rows.

---

## 2. Oracle diagnostics — MEASUREMENTS VALID, CONCLUSIONS RETRACTED

**Read the retraction at the top of this file first.** The numbers in this section are correct and
reproducible. Every *inference* drawn from them about a ceiling is withdrawn. They locate where the
error lives; they bound nothing.

I computed, on the fold record, what the **most optimistic oracles available** are worth. Not
models — oracles. Perfect knowledge, free.

**Matched lane.** Our model already beats a perfect `|delta|` band oracle:

| oracle | matched RMSE | vs our 224.26 |
|---|---|---|
| perfect `|delta|` band (4 groups), predict the band mean | 392.72 | far worse |
| perfect airport × band, predict the cell mean | 326.90 | far worse |
| predict `delta` = its global mean | 427.98 | far worse |

~~This kills the "there is a hidden class in the matched rows" story.~~ **RETRACTED.** This oracle
*replaces* the model's inputs with band membership rather than *adding* it. All it shows is that a
4-group partition is a poor predictor on its own — which is unsurprising and says nothing about
whether membership information helps a model that keeps X. The correct test is `model(X)` vs
`model(X + membership)`, and it has not been run. What the table does show is where we over-pay:

| `|delta|` band | rows | our RMSE | within-band sd(`delta`) | share of matched SSE |
|---|---|---|---|---|
| **< 2 min** | 130,929 | **135.79** | **69.49** | 14.2% |
| 2–10 min | 182,216 | 187.50 | 295.76 | 37.6% |
| 10–20 min | 20,257 | 381.63 | 798.56 | 17.3% |
| > 20 min | 5,613 | 969.77 | 2015.34 | 31.0% |

On the `< 2 min` band we score 135.79 where the irreducible is 69.49 — we pay ~2× because the model
must hedge against rows it cannot distinguish. **Recovering all of it is worth 5,176 board MSE**
(matched 224.26 → 212.21),
and only under an oracle that tells you which rows are in the band. RESULT 11 measured every
available indicator of that membership: AUC 0.48–0.51. On every other band we already beat the
within-band spread, so there is no headroom by that measure.

**Unmatched lane.** In the fold's frame (53,524 MSE), here is what perfect prediction of the worst
rows is worth:

| perfect on the worst … | unmatched RMSE | MSE removed |
|---|---|---|
| 1 row | 1,479.8 | **19,933** |
| 2 rows | 1,256.7 | **29,297** |
| 5 rows | 1,125.7 | 34,084 |
| 20 rows | 926.3 | 40,361 |
| 56 rows (all monsters) | 758.7 | 44,694 |

**One row is worth 87% of the gap to the leader. Two rows are worth more than all of it.** For
scale: every measured lane in this project combined has moved 8,000 MSE.

### A scenario table — RETRACTED as a bound, retained as arithmetic

Anchoring matched at 224.258 and taking our board unmatched contribution as the residual (33,505
MSE), here is what each oracle is worth **on the board's own scale**:

| scenario | matched MSE | unmatched MSE | total | RMSE | vs leader |
|---|---|---|---|---|---|
| **v5 today** | 49,520 | 33,505 | 83,025 | **288.14** | +42.85 s |
| + perfect `<2 min` matched body | 44,344 | 33,505 | 77,849 | 279.01 | +33.72 s |
| + perfect **ordinary** unmatched rows | 49,520 | 18,299 | 67,819 | 260.42 | +15.13 s |
| **+ both, no monster capture** | 44,344 | 18,299 | 62,643 | **250.29** | **+5.00 s** |
| + perfect **monsters** only | 49,520 | 15,206 | 64,726 | 254.41 | +9.12 s |
| + the whole unmatched lane | 49,520 | 0 | 49,520 | 222.53 | **beats 245.29** |

*(The 18,299 / 15,206 split comes from asking what the unmatched lane would contribute if our
ordinary-row performance on the 2026 file matched the fold's ex-monster RMSE of 995.6. That gives
15,206 MSE of ordinary error and leaves **~18,300 MSE of monster residue** inside our v5 score. The
split is sensitive to that assumption: at an ordinary RMSE of 800 the monster residue is 23,700, at
1,200 it is 11,400. **The honest band is 11,000–24,000 MSE, and it is the largest single object in
our error under every value in it.**)*

Two independent framings — the fold's own oracle arithmetic and this board-anchored one — both land
in the same place:

> ~~The best conceivable score without capturing a single extreme row is ~250.~~ **RETRACTED — this
> is the false claim.** It generalises two chosen oracles into a bound over all of them.

~~**Beating 246 therefore runs through the extreme rows.**~~ **RETRACTED — false.** Perfect
prediction on the 25,870 matched rows with `|delta| > 600 s` removes 23,901 MSE with no extreme row
involved. The table above prices two chosen interventions, not the set of all available ones.

### What the leader's 245.29 could be — A SCENARIO, NOT AN INFERENCE

**One total score is one equation in two unknowns.** Nothing below is evidence about their method.
The 213.7 figure was presented as an inference in the first draft of this file; that framing is
withdrawn.

Inverting the same identity — what matched RMSE would the leader need, given each assumption about
their unmatched lane?

| their unmatched contribution | required matched MSE | required matched RMSE | plausible? |
|---|---|---|---|
| same as ours (33,505) | 26,662 | **164.6** | no — we already beat a perfect band oracle at 224.3; 164.6 is below every airport we have except EHAM/EDDM |
| **ordinary-only, i.e. they capture the monsters (15,206)** | 44,962 | **213.7** | **yes — a 4.7% relative gain on matched, the size of one good feature block or a second learner family** |
| zero | 60,167 | 247.2 | trivially, but implies a perfect stratum |

**The parsimonious explanation of the leader's score is: they identify the extreme unmatched rows,
and their matched model is ~4.7% better than ours.** That single hypothesis reproduces 245.29 to
within noise, and neither half of it is exotic. It also fixes the priority order: **the monster lane
is ~80% of the gap, the matched lane ~20%.**

---

## 3. What the monster rows are, and why we cannot see them

Census over all twelve training months (2,085,047 departure rows):

| threshold | rows | rate | share unmatched | where |
|---|---|---|---|---|
| y > 2 h | 584 | 0.028% | 82.4% | LIRF 480, EGLL 37, LFPG 37, LTFM 21 |
| y > 3 h | 202 | 0.0097% | 92.1% | **LIRF 188**, LFPG 6, EGLL 4 |
| y > 10 h | 49 | 0.0024% | 98.0% | **LIRF 46**, LFPG 2, LSZH 1 |

Projected onto the 2026 scored file: **~60 rows with y > 3 h.** If they were predicted at a normal
1,000 s they would cost **252,197 MSE — three times our entire board score.** We do not pay that,
which means the pipeline already captures most of them. It captures them through `sp`
(take-off minus schedule): at LIRF the extreme rows have `sp ≈ y`, so the schedule offset carries
the signal and the mixture estimator follows it.

The ones we miss have no such signature. The fold's two worst rows:

| airport | month | y | sp | our prediction | residual |
|---|---|---|---|---|---|
| LFPG | 1 | 84,240 s (23.4 h) | 1,740 | 1,090 | **−83,150** |
| LFPG | 1 | 58,206 s (16.2 h) | 2,043 | 1,216 | **−56,990** |

The schedule says these are ordinary flights running half an hour late. The label says off-block was
a day earlier than take-off — operationally impossible, and therefore a corrupt `BLOCK_TIME`, which
is the target itself. Those two rows are 54.7% of the fold's stratum SSE and 27.0% of its total.

**Hedging cannot recover them.** The arithmetic is closed: `MSE(c) = Var(y) + (E[y] − c)²`, so the
most a shift of any group's prediction can ever buy is the square of that group's bias. Measured:
our pooled unmatched bias is **0.0 s**, and a per-airport *oracle* shift is worth **416 MSE**. You
cannot buy the tail by predicting a little higher everywhere; you have to name the row.

---

## 4. What we tried — the full record

Every entry is measured with a paired interval on the same rows unless noted. "CLOSED" means the
measured value is below the 1,000 MSE band and the reason is recorded so it is not reopened with the
same estimator.

### 4a. What worked

| # | lever | value | evidence |
|---|---|---|---|
| 1 | **The `delta` reparameterisation** — model `BLOCK − AOBT_3`, recover `y = proxy − delta` | matched **285.24 → 252.66** on identical fold/features/seed | the single largest modelling decision in the repo |
| 2 | LightGBM at proper capacity vs sklearn HGB | +4.0% matched, +4,251 MSE fold; **board v3 −5,975 MSE** | capacity *was* binding (17,557 trees vs 400) |
| 3 | Stratum hybrid — fitted body, cell-estimator tail | ESTABLISHED, all three clauses; **board v4 −1,138 MSE** | RESULT 10 |
| 4 | Queue / surface-congestion block | real, below its own bar; shipped by owner decision | RESULT 9 |
| 5 | Seed averaging (3 seeds) | ESTABLISHED but small, ~215 MSE | RESULT 7 |
| — | **v5 = seeds + queue** | **board −886 MSE, transfer 0.97×** | third 1:1 calibration point |

### 4b. What did not work — matched lane

| lever | measured | why it failed |
|---|---|---|
| Native categoricals instead of target encoding | +0.8% vs +3.4% | `STAND_mvt`'s 1,884 levels were never the problem; sklearn's 255-level cap cost nothing |
| Per-airport tree counts | NOT WORKING | RESULT 7 |
| Stand occupancy / witness block | +350 MSE marginal | **69% measured redundancy** with the incumbent `prev_arr_*`; the mechanism is real (39× enrichment) and already priced in |
| Weather, pooled | +0.49 s | tested at the wrong granularity; Arm W (airport-hour) is queued |
| Out-of-fold target encoding | −0.96 s | hurt |
| Winsorised training targets | −30.3 s | hurt badly — censoring the tail destroys the conditional mean |
| Absolute-error loss | −14.9 s | hurt; the metric is RMSE, the conditional mean is what it wants |
| Single undivided model | −144 s | hurt catastrophically |
| **Arm U — one y-target model over all rows** | **NOT WORKING twice** (RESULT 12, RESULT 15) | the standard approach loses to the two-part pipeline. LIRF matched 384 → 623. Refuted at capacity, with the stopping rule corrected in its favour |
| Gated blend, `|delta_hat| > 600 s` (Amdt 25) | NOT WORKING, −9.844 s | one airport (LIRF −72.75) sinks a rule that helps at nine |
| Gated blend ex-LIRF (Amdt 26) | INCONCLUSIVE, **+0.454 s ≈ 200 MSE** | 4 of 5 clauses hold, negative control passes at 4.5× p99 — the gate really does find the failing rows — but the point bar fails under both the strict and the original threshold |
| AOBT_3 derivation fingerprints | **AUC 0.48–0.51** | there is no detectable signature of how the NM produced the second clock |
| Record ordering (`MVT_ID`, `FLIGHT_ID`) | AUC 0.53 / 0.59 | not an identifier; Arm F tests it as a feature block anyway |
| Arrival-side clocks (NM vs airport landing) | agree to **16 s**; >10 min on 0.10% | no second clock to exploit there |
| Same-stand previous arrival as a witness | AUC ≤ 0.58 | RESULT 11.4 |
| Airport-clock turnaround anchor | **sd(turn) 13,503 s vs sd(delta) 427 s** | the anchor is 30× noisier than the thing it would anchor. RESULT 14b |
| ADS-B ground surveillance | **NO-GO on a measured ten-airport census** | only 2 of 10 clear the gate; **LFPG, LIRF, LEMD, LTFM have ZERO on-ground samples** — aircraft first seen 15–17 min after off-block. Worth 2,243 MSE from EHAM+LSZH alone against a ~190 GiB / ~20 h ingest |

### 4c. What did not work — unmatched lane

| lever | measured | why it failed |
|---|---|---|
| Fill classification, oracle flag + constant non-fill | **1,868 → 3,081 (worse)** | the mixture's hedging is doing real work; the non-fill rows are the monsters (sd 3,975, p99.9 77,903) |
| Fine `sp` buckets | rejected under held-out LOMO | overfits the cells |
| Mangled-designator separator at LIRF | **+305 MSE**, CI [−12.87, +26.58] | the descriptive finding reproduces exactly (91% vs 64%) but rests on **12 rows in 2025, ~3 expected in 2026**. Structural, not tuning |
| Flight-number fill propensity at LIRF | DEAD | RESULT 6 |
| Fill head (Amdt 16) | NOT WORKING | RESULT 8 |
| Amendment 11 lane | NOT WORKING | RESULT 4 |
| **Oracle fill decision, priced** | **13,497 MSE — and all of it at LIRF** (ex-LIRF it saves 42 of 35,856) | the shipped `p` already scores **AUC 0.944 pooled / 0.984 on the tail** and beats every alternative fitted against it. The remainder is **4 stale-BLOCK rows in two months**, on four distinct airport-days, with no shared signature |
| Tail insurance (raise predictions to hedge the tail) | **REFUSED on price** | pays above P = 23%; measured rate 0.07% |

### 4d. Method failures worth keeping

Recorded because they cost real time and will recur:

- **A 3-month smoke of the LightGBM experiment showed +9.1%; the full 10-month run gave +4.0%.** The
  smoke's control was data-starved. Standing rule since: no smoke or subsample magnitude is ever
  quoted as a result, only as evidence the code runs.
- **A gain measured against a weak baseline is not a gain.** The stand block was worth +809 MSE at
  two training months and +350 at ten — the baseline learned the same thing.
- **A single-seed paired bootstrap certified fitting variance as an established gain.** Across three
  seeds the same increment measured +0.245 / −0.212 / −0.253 — the sign flips. ESTABLISHED now
  requires ≥3 seeds and a gain > 2× the seed sd.
- **Fold totals were built from components of different runs three separate times.** Two LFPG rows
  were 27% of one fold total. Rule: no total is quoted unless every component came from the same
  fold and the same run.
- **A 29 s fold-vs-board gap was explained with an invented seasonality mechanism** when a correction
  already sitting in `FAMILY_B.md` §8 accounted for it. The failure was not the wrong theory; it was
  not reading the repo's own record before theorising.

---

## 5. What is still in flight

| arm | status | prior |
|---|---|---|
| Hyperparameter sweep (Amdt 15.2, screening) | **done 00:35** — incumbent `255,40,0.8` ranks **10 of 18**; top two are `127,20,0.6` and `127,20,0.8` | ranking only; confirmation tier not yet run. Plausibly ~600 MSE |
| Arm D — airport-day regime block | **running now** (started 00:36) | unmeasured |
| Arm Y — y-target formulation + blend | queued | RESULT 15 already refutes the pooled version |
| Arm F — record-ordering block | queued | the identifier hypothesis is dead; the feature block is not |
| Arm W — airport-hour weather | queued | the seasonal mechanism is present in the cache (freezing hours 1.92% scored vs 0.00% training July) |
| `--confirm` (sweep top two on the full fold) | not started — needs the sweep ranking | |
| `--catboost` blend (Amdt 15.1) | not started, timing unmeasured | second learner family; the one untested item with a plausible four-figure prior |

Prior from every measured lane so far: **low single-digit seconds each.** Nothing queued has yet
returned a verdict.

---

## 6. The corrected position on beating 246

**REWRITTEN 2026-09-10 00:55 after the retraction above. The previous verdict — "not reachable,
reachable at all only through the extreme unmatched rows" — is withdrawn.**

The honest position is that **no ceiling has been demonstrated.** What has been demonstrated is
that a specific set of *implemented* approaches did not clear their pre-registered bars. Several of
those experiments tested something narrower than the conclusion drawn from them:

| the conclusion drawn | what was actually tested | still open |
|---|---|---|
| "a second learner family doesn't help" | CatBoost on the **same float32 matrix with no `cat_features`** — verified in `scripts/catboost_fold_worker.py:136`, no categorical argument anywhere | CatBoost's actual categorical representation is untested |
| "the fill mixture doesn't work" | `p·sp + (1−p)·m(X)` where `m` is the all-rows model, i.e. `E[y|X]`, **not** `E[y|X,F=0]` — verified at `scripts/lgbm_fold.py:100` | properly conditioned experts (`mu_body` fit on non-fill rows only) are untested |
| "identity features don't help" | 12 keys collapsed to target/delta means; **full `FLIGHT_mvt` and `CALLSIGN_flt` are not model inputs** (only the 3-char airline prefix) | full identity, airport-scoped stand/runway, and operator×route interactions are untested |
| "LIRF reverses every pooled gain" | pooled methods rejected *because* LIRF reversed them, four times | a Rome expert with an inner-validated blend weight has never been fitted |

And one defect that undermines model selection across every arm measured so far:

**Early-stopping features contain their own stopping labels.** In `scripts/lgbm_fold.py::load_fold`,
`S.infold_encodings(d, y, dlt, tr)` is fitted on `tr`, and `fold_masks` puts the ES months *inside*
`tr` ("ES months come out of the TRAINING fold", line 657). So the 24 target/delta encodings on the
early-stopping rows were fitted on data including those rows' own labels. **The outer Jan+Jul
holdout is clean** — every reported RMSE and paired interval remains a valid measurement — but
`best_iter` was selected against an optimistically biased signal, so tree counts and cross-arm model
selection are distorted. The audit's synthetic dependency check confirms it: an inner stopping label
changes fit and stopping features; an outer holdout label does not.

This is a repair, not a banked gain. Its performance effect is unmeasured.

### What the numbers actually say now

- The matched lane holds ~49,520 board MSE. **Perfect prediction on the 7.63% of matched rows with
  `|delta| > 600 s` removes 23,901 MSE and would score 243.15** — beating the leader, using no
  extreme rows (max y in that set is 15,059 s). This is a diagnostic ceiling, never an achievable
  forecast, but it establishes that the matched lane is *large enough* to hold the gap.
- Perfect prediction on all current schedule-fill rows removes 7,324 MSE. LIRF alone, 11,195.
  These subsets overlap and none is achievable; they bound where the error lives, not what a model
  will recover.
- The unmatched lane holds ~33,505 board MSE. Same-flight arrival records recover a bounded
  ~1,355 MSE — a real secondary lane, not the main thesis.
- **The leader's matched RMSE remains unknown.** One total score is one equation in two unknowns.
  §2's earlier "they must be at 213.7" was a scenario presented as an inference; it is neither
  evidence about their method nor a constraint on ours.

### The order of work, and why

Priorities, deliverables and decision criteria are in **`plans/TOP_PATH_2026_09_10.md`**; this
section records only why the ordering changed.

| # | lane | why it is not a closed lane |
|---|---|---|
| **0** | **Correct the inner encoding/stopping separation** | a verified dependency in the code, not another feature hypothesis. Every comparison that follows depends on it. No measured gain |
| **1** | Rich categorical model on the `delta` target | full flight identity, airport-scoped stand/runway and interactions have never been inputs; CatBoost has never seen a categorical column |
| **2** | Conditional fill / non-fill experts | the old arm reused an unconditional regressor as its non-fill expert. `mu_body = proxy - E[delta|X,F=0]` fitted on non-fill rows only is untested |
| **3** | Joint context with a Rome-specific treatment | blocks were tested in isolation, never jointly; Rome's 22.6% of matched SSE has never had its own expert with an inner-validated blend weight |
| **4** | Clock recovery from same-flight arrival records | bounded at ~1,355 MSE on the fold — a real secondary lane, capped at one day of work |
| **5** | One regularized ensemble, then independent confirmation | separate gains overlap and cannot be added |

**Posture.** Finish the running queue under its original recorded rules and change nothing beneath
it. Then repair validation, then spend the research budget on priorities 1 and 2. Ship whatever
clears its bar on confirmation — including small maintenance gains — but do not describe a
collection of 200-MSE improvements as a route to first.

**What is honest to say about the target.** Top ten is 6,884 MSE and the cut drifts ~0.5-1.5 s/day.
First is 22,858. Neither is forecast here. The diagnostic ceilings show the matched lane is large
enough to contain the gap; whether any model reaches it is unmeasured, and unmeasured is the
correct word. If the surviving candidates recover 1,000-3,000 MSE, that is what gets reported.

---

## 7. Open hypotheses, stated so they are not carried forward as assumptions

| # | hypothesis | status | what would close it |
|---|---|---|---|
| H-A | Our v5 board score contains ~18,300 MSE (band 11,000–24,000) of extreme-row residue in the unmatched lane | **OPEN, and the single most load-bearing claim in this file.** It rests on assuming our ordinary-unmatched performance on the 2026 file equals the fold's 995.6 | not closeable from the board (it reports only a total) and not by probing (forbidden). Partially closeable by a LOMO estimate of ordinary-unmatched RMSE under the 2026 airport mix — **cheap, not yet run, and it would tighten the 11,000–24,000 band that every priority in §6 depends on** |
| H-A2 | The leader's 245.29 requires them to capture extreme rows | **CLOSED — WORKING**, conditional on H-A's band. Every non-monster oracle together stops at 250.3 | measured in §2, this file |
| H-B | The leader's edge is an external data source | **OPEN** | their published solution after the freeze |
| H-C | Arms D / Y / F / W are each worth < 1,000 MSE | **OPEN — verdicts by ~10:30** | the queue, against each amendment's own pre-registered clauses |
| H-D | A second learner family (CatBoost) blended is worth > 1,000 MSE | **OPEN, untested** | Amendment 15.1 |
| H-E | The matched lane's remaining error is continuous rather than a hidden class | **CLOSED — WORKING.** Our model beats a perfect band oracle (392.72) and a perfect airport×band oracle (326.90) at 224.26 | measured in §2, this file |
| H-F | Hedging can recover the extreme rows | **CLOSED — NOT WORKING.** Max gain from any group shift is bias²; measured pooled bias 0.0 s, per-airport oracle shift 416 MSE | measured in §3, this file |

---

## 8. Reproducing every number in §2 and §3

```bash
cd ~/Projects/prc-challenge
OMP_NUM_THREADS=1 nice -n 19 python3.11 -B - <<'PY'
import pandas as pd, numpy as np
d = pd.read_parquet('data/cache_stand/fold_preds_queue_allrows.parquet',
                    columns=['ap','y','delta','sp','is_unmatched','pipeline'])
m = d[d.is_unmatched == 0].copy(); m['r'] = m.pipeline - m.y
m['band'] = pd.cut(m.delta.abs(), [-1, 120, 600, 1200, 1e9])
# oracle over any grouping = within-group sum of squares of `delta`
g = m.groupby('band', observed=True).delta
print(np.sqrt((m.delta - g.transform('mean')).pow(2).sum() / len(m)))   # 392.72
PY
```

The twelve-month monster census reads `data/raw/training_*.parquet` directly with
`PHASE_mvt == 'DEP'`, `ADEP_mvt` in the ten airports, and `BLOCK_TIME` / `TAXITIME` non-null.

Source records: `reports/MSE_LEDGER.md` (board table, per-version transfer),
`reports/GAP_STRUCTURE.md` (lane budget), `reports/FAMILY_B.md` (the extreme rows, closed five
ways), `reports/ADSB_GATE.md` (the coverage census), `plans/PREREG_taxiout_2026_09_08.md`
(RESULT 1–17 and every threshold, each registered before its run).
