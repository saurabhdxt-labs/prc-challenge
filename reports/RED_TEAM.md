# RED TEAM — what has been missed, measured · 2026-09-08 (17:20–18:05 UTC)

Brief: assume the owner is right that 260.93 is an engineering problem, hunt for the overlooked lever, measure it.
Everything below was run under `OMP_NUM_THREADS=1 nice -n 19`, whole-month folds, encodings fitted inside the
training fold, row-bootstrap intervals, paired bootstraps for every A-vs-B. Scripts and raw outputs are under
`.../scratchpad/rt/` (`*.py` + `*.out`). Nothing under the repository was modified except this file.
Peak RSS of any run: 2.39 GB (one run; the rest ≤ 1.1 GB). Three runs were killed by my own memory guard
because the machine's swap reached 13.8 GB of 14.3 GB (see §7) — those are reported as UNMEASURED, not as results.

## 0. Bottom line

1. **Leaderboard moved while you worked: the leader is now 248.48** (`youthful-giraffe` v13, 17:21Z, a single-step
   drop from 262.7). 260.93 (`enthusiastic-daisy` v568) is second. 74 teams scored; team-best quantiles
   p10/p25/p50 = 280 / 295 / 328. Your expected 2026 score with today's best components is **≈ 300** (§2).
2. **The single established thing you have not used: the matched neighbours' own proxy taxi-out as a feature for
   the unmatched stratum.** For a row with no `AOBT_3`, the median `MVT − AOBT_3` of the matched departures that
   took off in the previous hour (same airport, same runway) is observable on the ranking file and predicts the
   non-fill taxi. As a robust residual correction on top of your cell estimator and your LIRF `p`:
   stratum **1574.5 → 1527.2, −47.3 [−62.8, −36.8]**; non-monster **963 → 882 (−81 [−89, −73])**; non-LIRF
   **1170 → 1103 (−67 [−116, −48])**; better in **12/12 months**; seed-stable. In total-score terms: **≈ −3.7 s**
   (195.0 → 189.2 s in quadrature). Established, modest, and it does not touch the monsters.
3. **Everything else I tried is a negative result** (§4): no ID/row-order leak, the ARVT "shortcut" is worth
   0.4 s, no stand-chain explanation of the matched tail, no arrival-side tell for LIRF fills, flight-number
   encodings hurt, a tree in place of your cell means hurts.
4. **Verdict on 260.93 with as-of-take-off features alone: NOT WORKING under an expected monster load.**
   With the best components measured today (matched 233.6–235.9, stratum 1527) the expected total is 299–301.
   260.9 needs matched RMSE **181** at the expected monster load, or **233** under the luckiest possible draw
   (zero Family-B and zero 24h-slip rows in the 2026 set, probability ≈ 0.09 × 0.05). 248.5 needs 162 / 219.
   The leader, if carrying the expected load, has an everything-else RMSE of **201** — 35 s below your matched
   model. I could not find where 35 s live. §5 says what that leaves open.

## 1. What was run (all fold A = Jan+Jul 2025 held out, unless LOMO is stated)

| script | question | rows / peak RSS |
|---|---|---|
| `order_leak.py` | does file order, `MVT_ID`, or `FLIGHT_ID` leak `BLOCK_TIME`? | 3 files, 0.80 GB |
| `flightid.py` | which event orders `FLIGHT_ID`; can it be inverted into a BLOCK estimate? | Jan 2025, 0.29 GB |
| `decomp.py`, `tail.py`, `tail2.py`, `standchain.py` | where the matched error sits under your E0/E1 predictions | 339k, 0.42 GB |
| `arvt.py` | effect size of `ARVT_3 − AOBT_3 − route median airborne` | 339k, 0.65 GB |
| `strat_amb.py`, `strat_amb2.py` | ambient/stand-chain features for the unmatched stratum, 12-fold LOMO, paired vs your `predE` | 22,219, 1.04 GB |
| `lirf_arr.py`, `lirf_turn.py`, `lirf_turn2.py` | arrival-side schedule-copy state as a serve-time tell for LIRF fills / matched contamination | 160k, 0.46 GB |
| `lean.py` (0.12 subsample of the training months, full test fold) | flight-number encodings; ARVT ablation; direct-y / log-y / blend | 207k train, 1.8–2.0 GB; blend KILLED |

The subsampled harness (`lean.py`) has reference RMSE 252.70 at 12% of the training rows (your full-data E1 is
235.95); its A-vs-B gains are paired on identical rows and are indicative, not absolute.

## 2. Reachability arithmetic, from measured components

Shares from the ranking file: 344,841 scored rows, 5,290 unmatched (1.534%). Monster loads from your
`STRATUM_MONSTERS.md` §4 (Family B λ = 2.38 rows × 1.93e9; 24h-slips λ = 3.09 rows × 29,763²).

| matched model | stratum | expected 2026 total | luckiest draw (0 Family B, 0 slips) | 5 Family-B rows |
|---|---|---|---|---|
| E0 244.1 | your predE 1574.5 | 311.0 | 274.7 | 333.7 |
| E1 235.9 | predE | 304.7 | 267.6 | 327.9 |
| E1 235.9 | ambient 1527.2 | **301.0** | 263.3 | 324.4 |
| E1+E3resid 234.6 | ambient | 299.9 | 262.1 | 323.4 |
| E4 mixture 233.6 | ambient | 299.2 | 261.2 | 322.7 |

Expected monster load in total-MSE terms: Family B 13,320 (115 s in quadrature), slips 7,940 (89 s). Together
they are 21,260 of the ≈ 90,600 expected total MSE — **23%**, and they are not predictable from anything in
the file (your finding; I found nothing new either, §4.6).

| target | matched RMSE needed, expected load, ambient stratum | matched RMSE needed, luckiest draw |
|---|---|---|
| 248.5 (leader) | 162 | 219 |
| 260.9 (2nd) | 181 | 233 |
| 266.8 (3rd) | 190 | 240 |
| 270.2 (5th) | 195 | 244 |

Reading: **fifth place is reachable with today's components and an average-to-lucky draw; second place needs
either the luckiest draw plus two seconds, or a matched model 50 s better than anything measured.** The
leader's single-submission jump 262.7 → 248.5 is a 7,300-MSE step. One LIRF scored row (`NOS`, `sp` = 111,654 s,
your report §5) moved from a ≈1,000 s prediction to ≈50,000 s is worth exactly 7,480 MSE *if it is a fill*
(and −7,000 if it is not). I cannot tell whether that is what happened; I note that the step size matches a
one-row decision better than it matches a modelling improvement.

## 3. The one established gain — ambient proxy for the unmatched stratum

**Hypothesis.** For an unmatched departure, the taxi-out of the *matched* departures taking off around it is an
unbiased, serve-time-observable estimate of the queue state that its own missing `AOBT_3` would have given.
TRUE shape: a paired, in-fold gain on the non-fill component with the LIRF `p` held fixed; FALSE shape: gain
inside the interval, or a gain that vanishes when only backward-looking windows are used.

**Features** (all computable on the ranking file; `T` = own take-off): median and p90 of `MVT − AOBT_3` over
matched departures at the airport with take-off in `[T−1h, T]` and `[T−3h, T]`; the same on the same runway;
count in the window; median `AOBT_3 − SCHED` (ambient NM delay) in `[T−1h, T]`; arrivals on-blocking in the
previous 30 min and their mean taxi-in; previous/next arrival on-block at the same stand. Spearman with the
non-fill target: runway-ambient proxy **+0.41**, airport-ambient proxy +0.37, ambient delay +0.19, previous
arrival's taxi-in +0.28, stand-chain gaps ≈ 0.

**Estimator that works.** Keep your hierarchical cells (`[ap]→[ap,cb]→[ap,fb]→[ap,fb,dayoff]`, k = 5) and your
LIRF logistic `p` (I read `pE.npy`, so `p` is *identical* to your L-e run); fit a small HGB (15 leaves, 200
iters, min-leaf 100) on the **inner-LOMO out-of-fold residual** `y − nf` of the non-fill training rows,
**winsorised at ±3,000 s for fitting only**; predict `p·sp + (1−p)·(nf + g(x))`.

| variant (12-fold LOMO, paired vs your predE) | stratum | monster | non-monster | non-LIRF |
|---|---|---|---|---|
| your predE (L-e) | 1574.5 | 13,647 | 963.1 | 1170.0 |
| sanity: same recipe, no ambient features | 1573.0 (−1.5 [−4.4, +1.0]) | 13,624 | 963.5 | 1170.8 |
| **HGB residual, W = 3000, backward windows only** | **1527.2 (−47.3 [−62.8, −36.8])** | 13,655 (n.s.) | **882.0 (−81.1 [−89.4, −73.2])** | **1103.1 (−66.9 [−115.5, −48.0])** |
| same, W = 2000 / 6000 | 1529.4 / 1527.7 | | 884 / 885 | |
| same, all windows incl. symmetric ±30 min | 1525.0 (−49.5 [−65.8, −38.7]) | | 878.0 | 1100.5 |
| proxy-family features only | 1534.5 (−39.9 [−53.7, −30.6]) | | 893.2 | 1113.1 |
| ridge instead of HGB | 1552.9 (−21.5 [−29.9, −15.7]) | | 924.2 | 1138.8 |
| seed 1 | 1526.0 (−48.5 [−64.6, −37.7]) | | 879.2 | 1100.6 |

Per month (predE → ambient): Jan 2904.6 → 2887.5, Feb 2132 → 1727, Mar 1070 → 1022, Apr 1012 → 977, May 2406 →
2393, Jun 2002 → 1988, **Jul 1317 → 1289**, Aug 1176 → 1151, Sep 980 → 944, Oct 863 → 855, Nov 1121 → 1043,
Dec 941 → 884. The symmetric window adds 2 s and is not worth the as-of-take-off argument; ship the backward
version. Coverage: gain is on non-fill rows at every airport; monsters untouched (as expected — nothing here
changes `p`). Not covered: 2026 application (features computed, model not refitted on 12 months for the 5,290
rows — that is a 10-second job with `strat_amb.py`'s feature block plus `strat_amb2.py`'s fit).

**A negative inside the same experiment, worth recording:** replacing your cell means by a plain HGB on `y`
(non-fill rows, 31 leaves) is **+408 s worse** on the stratum (+96 on non-monster) even with the ambient
features — the heavy tail makes leaves that a k = 5 shrinkage does not. Your estimator choice for the stratum
is right; only the residual correction adds.

## 4. Negative results (do not repeat)

### 4.1 No identifier or row-order leak of `BLOCK_TIME`
- Files are 92% sorted by `MVT_TIME`, not by BLOCK (BLOCK non-decreasing on 57% of consecutive rows = chance).
  Within airport-day, row index and `MVT_ID` correlate identically with MVT, BLOCK, SCHED and AOBT (0.018 / 0.39).
- `FLIGHT_ID_mvt` is time-ordered (within airport-day Spearman 0.999) — **by LOBT**: where LOBT order and SCHED
  order disagree, the ID follows LOBT 1.000 / 0.000; a leave-one-out ID-neighbour estimate of LOBT has |err|
  p50 0 s, p90 60 s, while the same estimate of BLOCK has p50 419 s (worse than own `AOBT_3`, p50 176 s).
  It carries nothing beyond `LOBT`, which is already a feature, and it is **null on 99.8% of unmatched rows**.
- `FLIGHT_ID` is shared between the DEP and ARR legs for 16% of departures (destination in the ten airports);
  the ARR leg's on-block is the destination's, not ours.

### 4.2 The ARVT "shortcut" is worth 0.4 s
`g2g − route-median airborne` (route medians from the training months): Pearson with y **+0.148** vs the proxy's
+0.556; the part orthogonal to the proxy (`airborne3 − route median`) has Pearson −0.04 with y and −0.03 with
delta. OLS `y ~ proxy` 396.25 → `+ ab_res` 396.29. Regressing your E1 residual on it *on the test fold*
(optimistic) moves 235.95 → 235.94. Ablating the three ARVT features from the full set: **−0.38 s [−0.71,
−0.04]** (subsampled harness). Consistent with your permutation numbers (+0.47, +0.56). Nobody is winning with
this. The decision it informs: not worth the conduct risk for 0.4 s.

### 4.3 The matched tail is not the stand chain, not schedule copies outside LIRF, and is a summer phenomenon
Under E1 on fold A: |residual| > 1,000 s is 0.61% of rows and **30.5% of matched SSE** (> 600: 2.1% / 45.8%).
Of those 2,068 rows, **83% have `BLOCK − AOBT_3 < −600`** (airport off-block 20–35 min *before* NM's), 89% have
`y > proxy`, median NM delay 34–53 min (`AOBT_3 − SCHED`), median stand gap to the previous arrival 7,000 s
(ordinary turnarounds). Only LIRF's tail rows are schedule copies (52%); at LTFM/LFPG/EGLL 4–7% are. July
carries 73% of tail SSE; P(delta < −600) is 3.2% in Jan vs 5.4% in Jul, and rises monotonically with NM delay
(0.1% at on-time, 11–13% above 30 min). Your E1 has the conditional mean right in every delay bucket (residual
means −5 to +13 s) — this is **variance of a ~10% mixture member on delayed flights** whose membership no
serve-time column marks. Checked and dead: stand-occupancy bound (`prev_arr_gap` < 1,500 s covers 3.5% of the
tail; the model already fits those rows), schedule/LOBT/EOBT copies, hour of day (flat 10–15h).
Reading: BLOCK looks like an airline OUT-type event (brakes released, waiting for TSAT/CTOT) on slot-held
flights, while `AOBT_3` is the movement. If any column identified slot-held flights it would be worth up to
~25 s on matched RMSE (197.3 without the tail rows) — none does.

### 4.4 Error decomposition for reference (E1, fold A)
SSE share by airport: LIRF 21.8% (7.7% of rows, RMSE 396), LTFM 16.4% (260), LFPG 13.4% (256), EGLL 12.6% (245),
the other six 8% or less (173–226). LIRF schedule-contaminated matched rows are 0.66% of rows and **8.8% of
SSE** (RMSE 859 on them; 225.4 if their error were zero) — your exp4 oracle (+12.9) and mixture (+2.4) bracket
exactly this.

### 4.5 LIRF fills: no arrival-side tell, no rotation tell
- LIRF arrivals also carry schedule copies (on-block = SCHED on 12.1%), but the daily arrival copy rate does not
  track the daily departure fill rate (r = −0.22 / +0.21), and on the rows that decide the score (unmatched,
  `sp` > 3h) the same-airline-same-day arrival copy rate has **AUC 0.498**.
- The inbound rotation's on-block being a copy raises matched contamination from 7.7% to 20.9% (AUC 0.57
  alone) but adds **+0.003 AUC** to a LOMO logistic that already has the airline propensity (0.744 → 0.747;
  Brier −0.0004, established but worthless in y-terms: implied mixture MSE 80,422 → 80,547).

### 4.6 Nothing new on Family B / 24h slips
I re-read your anatomy and did not find a serve-time column you missed. The only new observation is §2's
one-row arithmetic: the 11 LIRF scored rows with `sp` > 12 h are each worth 6,000–35,000 MSE depending on
fill/slip/ordinary, and your mixture already places them at the conditional mean given 2025's cell counts
(8 rows in the 18h+ cell: 6 slips, 2 fills). That is the correct decision under squared loss; it is also the
biggest single source of variance in anyone's final score.

### 4.7 Flight-number target encodings hurt as implemented
Adding in-fold smoothed encodings (K = 50) keyed by `FLIGHT_mvt`: **−9.1 s [−10.5, −7.6]**; keyed by
`(FLIGHT_mvt, STAND)`: −100 s. Cause: self-inclusion — with ~10 training rows per key the row's own target
dominates its encoding, the trees trust it, and it is noise on the test months. Your coarse keys (hundreds of
rows per level) do not suffer this. A leave-one-out / out-of-fold encoding was not tested; I would expect it to
be neutral, not the 10-second lever.

### 4.8 Direct-y target
At 12% of the training rows, HGB on `y` directly: 290.9 [249.9, 356.8] vs 252.7 for the delta
reparameterisation on identical rows and features — same conclusion as your 285 vs 253 at full data. The
interval width says the direct model's tail is uncontrolled; a blend can only help if it is small-weight.

## 5. Answers to the five questions

1. **Framing.** Delta reparameterisation is right (re-confirmed, §4.8); trees with in-fold encodings are the
   right estimator for delta (your capacity and per-airport results; the ridge variant of §3 recovers less than
   half of the HGB gain on the stratum); cell means with shrinkage are the right estimator for the stratum
   (§3's negative). A per-(stand, runway) GAM is subsumed by the `de_ars`/`te_ars` encodings. Log-taxi: not
   measured (killed, §7); theory and §4.8's tail say it would not be the lever.
2. **Unused information.** The ambient proxy for the stratum (§3, established). Same-aircraft chains: no
   registration exists; `FLIGHT_ID` links legs of the same flight, not rotations; stand-chain features are in
   your set and do not explain the tail. Nothing else in the 30 columns is unread.
3. **Leak.** None in identifiers or order (§4.1). ARVT: 0.4 s (§4.2). 260.93 is not reachable *with* it either.
4. **Ensembling / transforms.** Direct-y: measured, worse (§4.8). Blend and log-y: **UNMEASURED** — the runs
   were killed by memory pressure twice; `rt/lean.py 0.12 direct` finishes them in 90 s when swap is below
   ~10 GB (§7). Seeds: your exp6 is the right measurement; I did not duplicate it.
5. **Else.** The leader's step size (§2). The stratum's non-monster component has more room than "nearly
   maxed": 963 → 882 here, and the 12-month per-fold table says the non-LIRF delay pile at every airport moves.

## 6. Verdict

- Stratum ambient correction: **WORKING**, +3.7 s on the total, interval excludes zero on all three cuts,
  12/12 months, two seeds; not covered: the 2026 refit itself (mechanical).
- "260.93 is reachable with as-of-take-off features alone": **NOT WORKING** at the expected monster load
  (needs matched 181); **open only under the luckiest draw** (needs matched 233 — E1 + E3resid + E4 mixture is
  233.6, so second place is possible if 2026 contains no Family-B and no 24h-slip row, p ≈ 0.005, or a few of
  either and a matched model 5–10 s better than measured). 248.5: **NOT WORKING** under any draw with the
  components I can see (needs 219 even when lucky).
- Honest statement of what I could not find: where 35 s of everything-else RMSE separate the leader from E1.
  The candidates I can name and did not measure: an out-of-fold flight-number/rotation encoding (§4.7), a
  small-weight direct/delta blend (§4.8), and a serve-time indicator for slot-held flights (§4.3) that I could
  not construct from the columns.

## 7. Machine note — read this before running anything else

At 17:47Z the machine had **swap 13.8 GB used of 14.3 GB** and free pages fell to 3,700 (60 MB) several
times. Concurrent consumers: your `mm/chain.sh` runs (2.4–3.1 GB each), `phantom-spoofing/scripts/
aggregate_spoof_cell_hour.py` (1.7 GB), a Virtualization.framework VM (1.2 GB). My guard (`rt/guard.sh`,
kills at swap-free < 700 MB or free pages < 20k) fired three times on my own runs. Your chain's `exp3_A_wr`
also died once with rc = 143. Nothing under `com.phantom.*` was touched. Do not run two ≥2 GB fits at once
until the spoofing aggregation and the VM are gone.
