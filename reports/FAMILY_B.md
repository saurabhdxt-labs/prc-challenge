# Family B — is it predictable at serve time? (PRC Data Challenge 2026)

Generated 2026-09-08. Every number below is copied from the outputs of the scripts under the session
scratchpad `.../scratchpad/famb/` (`extract.py → enum.py → sig.py → mech2.py → cluster.py → mech3.py →
inbound.py → value.py → final.py`, each with a `.out`). Runs used `OMP_NUM_THREADS=1 nice -n 19` with
the Phantom venv python 3.11 / pandas 3.0.2. Nothing under the repository was modified except this
file. **Two runs exceeded the 2.5 GB RSS cap** (`extract.py` 2.75 GB — the one-off concat of all
4.86 M rows with string columns; `mech2.py` 2.61 GB — an `astype(str)` over the full frame). Later
scripts avoided both patterns (peaks 0.49–2.43 GB). No `com.phantom.*` job was touched.

The question, unbroadened: **does the family of monster rows whose `sp` looks ordinary carry any
signature observable on a scored row?** The answer is **no — NOT PREDICTABLE**, with the cost stated
in §7. Every candidate, positive or negative, is in §4. A reinterpretation of the leaderboard
arithmetic that motivated the question is in §8: the leader's 248.48 does **not** require predicting
these rows.

## 1. Definitions and the exact count

| symbol | meaning |
|---|---|
| `y` | `TAXITIME_SEC_mvt` = `MVT_TIME − BLOCK_TIME` (verified exact on 2,084,659 admissible 2025 departures and on 2,427,442 arrivals as `BLOCK − MVT`) |
| `sp` | `MVT_TIME − SCHED_TIME`, computable on every scored row |
| `d` | `BLOCK_TIME − SCHED_TIME`, training only; `y = sp − d` identically |
| admissible | PHASE=DEP, TAXITIME/BLOCK/MVT non-null, TAXITIME>0, first of duplicated MVT_ID → **2,084,659** rows |
| unmatched | `AOBT_3_flt` null → 22,219 admissible 2025 rows (1.07%); 5,290 of the 344,841 scored 2026 rows |
| monster | `y > 10,800` → **202** rows (186 unmatched, 16 matched) |
| fill | `|d| ≤ 60` (then `y == sp` by construction) |
| 24h slip | not fill, `y − 86,400 ∈ [0, 10,800]` |
| **Family B (definitional)** | `y > 10,800` **and** `sp < 10,800` → **10 rows**, all unmatched, 0 matched |
| **Family B (mechanism)** | definitional **and** `d ≤ −10,000` (BLOCK stamped ≥ 2.8 h before SCHED) → **9 rows** |

The 10th definitional row (EGLL `SHT14L`, 2025-07-30, `sp` 10,563, `d` −604, `y` 11,167) is a genuine
2.9 h delay + 10 min taxi that the `sp < 3h` cut happens to catch; it belongs with the 9 matched
"genuine long wait" monsters (§1b) and is excluded from the mechanism set below.

All 202 monsters classified (enum.out):

| | fill | Family B | 24h slip | genuine long wait (`d` ≈ 0, `sp` 11–21k) |
|---|---|---|---|---|
| unmatched (186) | 164 | 10 | 12 | 0 |
| matched (16) | 7 | **0** | 0 | 9 |

### 1a. The nine rows (every column that is not constant or null)

| MVT_ID | month | ADEP | ADES | FLIGHT | type | rwy | stand | rule | SCHED | BLOCK | MVT | sp | d | **y** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 182040319 | Jan | EHAM | EHAM | NCG01A | DH8A | 22 | HG01 | V | 01-06 08:45 | 01-06 05:35:04 | 01-06 08:38:56 | −364 | −11,396 | 11,032 |
| 182378789 | Jan | LFPG | LFMN | EJU983W | A319 | 08L | C03 | I | 01-19 18:25 | **01-18** 19:30:00 | 01-19 18:54:00 | 1,740 | −82,500 | 84,240 |
| 182377662 | Jan | LFPG | GMMX | EJU42AY | A320 | 26R | D06 | I | 01-23 14:35 | **01-22** 22:58:57 | 01-23 15:09:03 | 2,043 | −56,163 | 58,206 |
| 183662617 | Feb | LSZH | LSZF | HBCYH | C172 | 28 | MFG | V | 02-08 15:30 | 02-08 10:51:47 | 02-08 15:58:36 | 1,716 | −16,693 | 18,409 |
| 183907367 | Feb | LIRF | LFPG | ITY324 | null | 25 | 404 | I | 02-27 14:10 | 02-27 10:59:54 | 02-27 14:36:05 | 1,565 | −11,406 | 12,971 |
| 189109908 | May | LIRF | LROP | WMT7TL | null | 25 | 310 | I | 05-02 07:40 | **05-01** 23:23:56 | 05-02 08:05:04 | 1,504 | −29,764 | 31,268 |
| 188253302 | May | LSZH | LFBE | NJE389D | C56X | 28 | 103 | I | 05-14 14:40 | **05-13** 14:39:58 | 05-14 14:55:39 | 939 | −86,402 | 87,341 |
| 190205968 | Jun | LIRF | KEWR | ETH508 | null | 25 | 903 | I | 06-18 03:25 | 06-18 00:31:59 | 06-18 03:49:01 | 1,441 | −10,381 | 11,822 |
| 195000946 | Aug | LIRF | LGIR | WMT8PV | null | 25 | 309 | I | 08-02 10:15 | 08-02 03:46:01 | 08-02 10:42:59 | 1,679 | −23,339 | 25,018 |

`FLIGHT_ID`, every `*_flt` column, and `AIRCRAFT_TYPE` at LIRF are null on all nine — as on every
unmatched row. Hours 3–18, all seven weekdays, four airports, eight airlines, nine stands, four runways.
Three rows fall in the held-out months (all January): EHAM 11,032; LFPG 84,240; LFPG 58,206.

### 1b. What the matched side says about the mechanism

The mechanism "BLOCK hours before SCHED while take-off is ordinary" **occurs only on unmatched
rows** (9/9). Matched rows with `d < −10,800` exist (139, EHAM 113) but every one has `MVT ≪ SCHED`
— a wrong SCHED, ordinary `y`. Matched rows where BLOCK disagrees with NM's AOBT by more than 3 h:
12 in the year, none with ordinary `sp`. So there is no matched-side twin of this family from which
to learn a signature with `*_flt` columns present.

## 2. Is Family B the tail of a larger, learnable population?

`d` for rows with ordinary `sp ∈ (−600, 10,800)` (mech3.out):

| `d` bucket | matched | unmatched |
|---|---|---|
| < −80,000 | 0 | 2 |
| −80,000 … −43,200 | 0 | 1 |
| −43,200 … −21,600 | 0 | 2 |
| −21,600 … −10,800 | 0 | 3 |
| −10,800 … −7,200 | 0 | 6 |
| −7,200 … −3,600 | 3 | 29 |
| −3,600 … −1,800 | 93 | 5 |
| −1,800 … −900 | 6,543 | 77 |

Below −3,600 the population is 53 unmatched rows (LIRF 46, EHAM 2, LFPG 2, LSZH 2, EDDM 1), i.e. a
continuum in which the nine are the tail. Its LIRF core is a distinct, *regular* sub-family (§9,
side finding 2): three Brussels Airlines rotations plus AEZ/LZB/AEA with `y ≈ 7,800–8,300` almost
independent of `sp` and `d` clustered at **−7,200 ± 6 s** — "SCHED minus exactly two hours". That
sub-family does not extend to the nine (their offsets are −10,381 … −86,402 with no repeats except
two rows at −11,400 ± 6 s, one at −86,400 + 2 s), and its airlines (BEL, AEZ, LZB) contain no Family B
row. Signature hunting on the 53 (prefix, hour, weekday, type, runway, rule; final.out / mech3.out)
returns only the LIRF/runway-25 concentration, which is the airport itself (1,314 of LIRF's 1,488
unmatched rows depart from 25).

## 3. What the BLOCK stamps are — and are not

Tested mechanisms for where the wrong BLOCK value comes from (sig.out, mech2.out, inbound.out,
final.out):

| hypothesis | test | result |
|---|---|---|
| BLOCK = previous arrival's on-block at the same stand | merge_asof on (airport, stand) ±5 s / ±60 s, ARR rows | **0/9**; fills 0/171, slips 1/12, ordinary unmatched 1/3,000 — dead |
| BLOCK = some other record's MVT/BLOCK/SCHED at the airport (±3 s, ±3 days) | exhaustive scan | only round-minute coincidences (`19:30:00` = three other flights' SCHED) — dead |
| BLOCK = an inbound arrival's landing or on-block (any stand, ±5 s) | scan over all 53 BLOCK-early rows vs chance on 500 ordinary rows | 26/53 hits vs chance **172/500 = 34%**; same-airline 2/53 vs 41/500 — at chance. WMT7TL ← `WMTMT4BS` landing 23:23:56 is one of those two and cannot be told from coincidence |
| BLOCK is a minute-granular (planned) value, not a clock reading | share within ±6 s of a whole minute | Family B 8/9; but real matched off-blocks (`|BLOCK − AOBT3| ≤ 60`, n = 433,571) are **75.5%** — the airport clocks are coarse; binomial p = 0.31. Not evidence |
| discrete "fill-with-offset" family | `d` within ±15 s of round offsets, all unmatched rows | −7,200: 4 (LIRF BEL/AEZ); −11,400: 3 (EHAM 2, LIRF 1 — two of them Family B); −86,400: 1 (Family B); −10,800: 1 (EHAM, `y` 1,085) — too sparse to be a family |

Verdict on mechanism: **unidentified**. The stamps are orphan values that match nothing else in the
file. Two of the nine (LFPG C03 `19:30:00` the day before; LSZH `SCHED − 86,402`) look like a planned
time from a different day; the rest do not resolve to anything.

## 4. Serve-time signatures — every candidate, with its test

All tests use only columns present and non-null on scored rows. "base" = the other 22,210 unmatched
2025 rows unless stated. n = 9 throughout, so anything short of a large effect is noise by
construction; nothing here reached the stage of an out-of-sample rule.

| # | candidate | Family B | base | test / verdict |
|---|---|---|---|---|
| 1 | airport | LIRF 4, LFPG 2, LSZH 2, EHAM 1 | LIRF 6.7% of unmatched | LIRF over-represented (4/9 vs 0.067; Fisher-type p ≈ 0.001) **but** this is the airport cut the caller warned about: rate 4/1,488 at LIRF vs 2/3,762 LFPG, 2/2,054 LSZH, 1/4,083 EHAM — 0.02–0.27% everywhere; useless as a rule |
| 2 | `AIRCRAFT_TYPE` null | 4/9 | 6.8% | Fisher p = 0.002 — **the airport trap**: the 4 are exactly the 4 LIRF rows (LIRF unmatched rows are 100% type-null). Within airport: nothing |
| 3 | hour of day | 8,18,14,15,14,7,14,3,10 | — | a post-hoc pick "hour ∈ {7,8,14,15}" gives 6/9 vs 0.251, Fisher p = 0.010 — chosen after seeing the rows; garden of forking paths, rejected |
| 4 | weekday | 0,6,3,5,3,4,2,2,5 | flat | nothing |
| 5 | `FLIGHT_RULE` = V | 2/9 | 12.1% | Fisher p = 0.30 |
| 6 | `ADES == ADEP` | 1/9 | 7.3% | p = 0.50 |
| 7 | runway / stand | LIRF all rwy 25; nine different stands | rwy 25 = 88% of LIRF unmatched | nothing |
| 8 | destination prefix | LF, GM, LS, LF, LR, LF, KE, LG, EH | — | nothing shared |
| 9 | "unexpectedly unmatched" — designator's out-of-month matched rate | median 0.99, 6/9 ≥ 0.9 | ordinary unmatched median 0.97, **65%** ≥ 0.9 | no discrimination; candidates ≥ 0.9 number 14,239 and contain 6 Family B rows (0.04%) |
| 10 | airline-prefix out-of-month matched rate | median 0.986 | 0.985 | nothing |
| 11 | MVT_ID out of time order (displacement from ±10 id-neighbours' median MVT) | LFPG rows 6.0 d and 1.5 d; LIRF rows 0–5 min | LFPG unmatched: 75% > 6 h, **44% > 5 d**; LIRF 3.8% > 6 h | airport property (LFPG/EHAM ids are batch-loaded per airline); the LIRF rows are perfectly in order |
| 12 | SCHED seconds-of-minute; MVT seconds | SCHED all :00; MVT 7/9 near whole minute | SCHED 100% :00; MVT 66% | nothing |
| 13 | designator did not operate the previous day | 6/9 overall; regular designators (≥200 rows/yr) 1/3 | regular unmatched: 8.5% | the 6 include a helicopter, a C172, a bizjet and two easyJet rotations that are not daily; on regular designators 1/3 — nothing |
| 14 | SCHED time-of-day deviates from the designator's usual STD | median 10 min | median 0, p90 205 | nothing (ITY324, WMT7TL, WMT8PV depart at their usual STD) |
| 15 | same designator / same stand / same callsign had another movement nearby | see §5 | — | nothing beyond the ordinary rotation |
| 16 | gap to previous / next arrival on-block at the stand | 0.5–36 h, matches the stand's normal use (stand histories in sig.out) | — | nothing |

Pooled-vs-within-airport check (the trap that caught the prior work three times): every apparent
signal above (#1, #2, #7, #11) evaporates within airport. There is no within-airport candidate with
an effect size, let alone one to test on the held-out months.

## 5. Clustering in time (a recording outage would burst)

Per row, same airport (cluster.out): departures within ±1 h / ±3 h with the number that are
unmatched, BLOCK-early (`d < −3,600`), or `y > 3,600`:

| row | ±1 h (dep, unm, early, y>1h) | ±3 h | gap to nearest other BLOCK-early row | gap to nearest other unmatched row |
|---|---|---|---|---|
| EHAM NCG01A | 89, 1, 0, 0 | 233, 4, 0, 0 | 7,784 h | 55 min |
| LFPG EJU983W | 61, 0, 0, 0 | 193, 1, 0, 0 | 92 h | 176 min |
| LFPG EJU42AY | 71, 2, 0, 1 | 174, 2, 0, 1 | 92 h | 10 min |
| LSZH HBCYH | 36, 1, 0, 0 | 98, 7, 0, 0 | 2,279 h | 30 min |
| LIRF ITY324 | 40, 0, 0, 0 | 143, 1, 0, 1 | 94 h | 108 min |
| LIRF WMT7TL | 63, 0, 0, 1 | 155, 2, 0, 1 | 73 h | 148 min |
| LSZH NJE389D | 48, 0, 1, 0 | 132, 1, 1, 0 | 2,279 h | 68 min |
| LIRF ETH508 | 24, 0, 0, 0 | 64, 0, 0, 1 | 180 h | 223 min |
| LIRF WMT8PV | 55, 0, 0, 0 | 179, 2, 0, 1 | 70 h | 116 min |

Base gap from an unmatched row to its nearest unmatched neighbour: median 23–66 min (EHAM 23, LFPG
24, LSZH 24, LIRF 66). The nine are isolated singletons: no other BLOCK-early row within three days
of any of them, no burst of unmatched rows, no second monster nearby. MVT_ID neighbours (±4) are all
ordinary matched rows with normal BLOCK. Only one airport-day in the year has two BLOCK-early rows
(LIRF 2025-03-03, both `y` ≈ 4,500, neither a monster). **No temporal clustering.**

## 6. Is the airport-day anomalous without BLOCK?

Percentile of the row's airport-day among all days of that airport (cluster.out), for
departure count, arrival count, unmatched share, median and p90 of `sp`, type-null share,
FLIGHT_ID-null share, mangled-designator share:

| row | n_dep pct | n_arr pct | unm share pct | sp med pct | sp p90 pct | fid-null pct |
|---|---|---|---|---|---|---|
| EHAM 01-06 | 0.18 | 0.17 | 0.91 | 0.92 | 0.95 | 0.88 |
| LFPG 01-19 | 0.04 | 0.05 | 0.27 | 0.49 | 0.49 | 0.31 |
| LFPG 01-23 | 0.08 | 0.12 | 0.35 | 0.09 | 0.13 | 0.39 |
| LSZH 02-08 | 0.06 | 0.08 | 0.88 | 0.35 | 0.40 | 0.91 |
| LIRF 02-27 | 0.24 | 0.23 | 0.69 | 0.25 | 0.13 | 0.66 |
| LIRF 05-02 | 0.76 | 0.76 | 0.71 | 0.27 | 0.42 | 0.67 |
| LSZH 05-14 | 0.62 | 0.62 | 0.76 | 0.25 | 0.34 | 0.72 |
| LIRF 06-18 | 0.68 | 0.77 | 0.72 | 0.72 | 0.61 | 0.67 |
| LIRF 08-02 | 0.76 | 0.76 | 0.93 | 0.92 | 0.97 | 0.92 |

Two days (EHAM 01-06, LIRF 08-02) are high-delay, high-unmatched days; LFPG 01-23 is a low-delay
day; the rest are unremarkable. The unmatched-share percentiles lean high (median 0.72) but that is
the selection effect of the row itself being unmatched on a low-count airport-day. **No consistent
airport-day anomaly.**

## 7. What it is worth, and what a false positive costs

Held-out Jan+Jul 2025 fold (344,336 rows, total RMSE 323.1 → SSE 3.595 × 10¹⁰), the three mechanism
rows (value.out):

| prediction on the three | their SSE | share of fold SSE | fold RMSE with them at zero error |
|---|---|---|---|
| 1,200 | 1.024 × 10¹⁰ | 28.5% | 273.2 |
| 1,500 | 1.015 × 10¹⁰ | 28.2% | **273.7** |
| 2,500 | 9.857 × 10⁹ | 27.4% | 275.3 |

Gain as a function of the precision of a hypothetical flag (flagged ordinary rows taken as
`y` = 1,200 ± 900, the SSE-optimal constant predicted on the flagged set — this already prices the
false positives):

| precision | rows flagged (K) | optimal constant | fold RMSE |
|---|---|---|---|
| 100% | 3 | 51,159 | 287.9 |
| 50% | 6 | 26,180 | 306.2 |
| 20% | 15 | 11,192 | 316.7 |
| 10% | 30 | 6,196 | 320.1 |
| 5% | 60 | 3,698 | 321.8 |
| 2% | 150 | 2,199 | 322.8 |
| 1% | 300 | 1,700 | 323.0 |

A 20%-precision rule would be worth ≈ 6 s; below 5% precision nothing is left. No candidate in §4
has any precision above the 0.04% base rate.

False-positive cost of predicting a monster value `V` on an ordinary row (`y` ≈ 1,200): `(V − 1,200)²`
per row. Per 1,000 such rows in the 344,841-row score: `V` = 5,000 → +59 s on 323.1 (+73 s on 248.5);
`V` = 10,000 → +251 s; `V` = 30,000 → +1,261 s; `V` = 60,000 → +2,860 s. One wrong 60,000 s
prediction alone adds 4.3 s to a 248.5 score.

**Irreducible cost on the 2026 score.** Expected count: pooled 9/22,219 × 5,290 = **2.14**;
per-airport rates × 2026 per-airport unmatched counts (EHAM 1,206, LFPG 734, LSZH 659, LIRF 383) =
**2.36**. Mean squared error of a mechanism row at the best available prediction (1,500):
**2.165 × 10⁹** (per-row errors 9,532 … 85,841). In quadrature on 344,841 rows: **116 s (λ = 2.14) /
122 s (λ = 2.36)**; Poisson 10th–90th percentile of the count is 1–4 rows → **79–158 s**; P(0 rows) =
12%, P(≤ 1) = 37%. This replaces the prior report's 115–119 s with the same conclusion and a slightly
wider band.

## 8. The leaderboard does not imply anyone predicts these rows

The caller's decomposition `248.48² = 0.98454·matched² + 0.01546·stratum²` treats the stratum RMSE
as a stable quantity. It is not: with λ ≈ 2.1 and a per-row MSE of 2.2 × 10⁹, Family B alone puts
**935 s** on the expected 2026 stratum RMSE (`√(2.14 × 2.165e9 / 5,290)`), **640 s** in a 1-row draw,
**0** in a 0-row draw (12% chance), and the realised size of the one or two rows swings it by hundreds
more (the two Jan-2025 LFPG rows are 84k and 58k; the median of the nine is 18k). A "stratum 790" is
what a 1-row draw of typical size looks like with everything else ordinary; it is not evidence of
prediction.

The fair comparison is on the rest. Our fold with the three rows removed is **273.7**; adding back
Family B *at its expectation* (119 s in quadrature) gives **298.5**, not 323.1 — the fold is ≈ 25 s
unluckier than average on this family alone. For the leader to sit at 248.48 with a 1-row Family B
draw, their non-Family-B score must be ≈ 236; with two rows ≈ 222; with zero, 248.5. **The gap to
the leader is 25–50 s on the matched side and the ordinary stratum, not on Family B.** The
matched-side "exhausted" conclusion in the handoff should be re-examined in that light; this report
does not do so (out of its scope).

## 9. Side findings (outside the question; reported because they fall out of the same tables)

1. **Mangled designators separate slips from fills at LIRF.** 166 of LIRF's 1,488 unmatched rows
   carry a doubled-prefix designator (`ITYTY680`, `RYRR90BN`, `BAWW561D`, `EXSXS1GT`, `CHHHH438`,
   `WMTMT306`, regex `^[A-Z]{3}[A-Z]{1,3}[0-9]`, length ≥ 7); the share is ≤ 0.4% at every other
   airport. Their `y` median is **9,090** vs 1,800 for other LIRF unmatched rows. Among LIRF unmatched
   rows with `sp > 3h` (243): mangled → 59 fill, **9 slip**, 7 ordinary; non-mangled → 105 fill, 3
   slip, 60 ordinary. So 9 of the 12 slip rows are mangled, and mangled rows are 91% fill-or-slip vs
   64%. This is a serve-time column (`FLIGHT_mvt`) and it is exactly the fill/slip separator the prior
   report lacked; 40 LIRF unmatched scored rows in 2026 are mangled. **Not validated out-of-sample
   here** — it needs the LOMO harness before it is believed. Family B rows: 0/9 mangled.
2. **A "SCHED minus 2 h" fill family at LIRF.** 15 of 23 unmatched Brussels Airlines rows
   (`BEL2TH` 08:10, `BEL3SG` 16:10, `BEL5LC` 19:00) plus AEZ/LZB/AEA rows have `y` ≈ 7,800–8,300
   regardless of `sp` (`d` clustered at −7,200 ± 6 s and −4,000 … −7,400); when `sp > 6,000` the same
   rotations are ordinary (650–780) or plain fills. Matched BEL rows at LIRF are ordinary (`y` median
   842). 2026 has 5 BEL and 12 AEZ unmatched LIRF rows. Worth ≈ 2 s on 323.1 if perfectly handled —
   small, but it is a rule, not a guess.
3. **Matched rows with a wrong SCHED.** 139 matched rows have `BLOCK` ≥ 3 h before `SCHED` because
   `MVT` is hours before `SCHED` (EHAM 113); `y` is ordinary. Any matched-side feature built on `sp`
   must not be surprised by `sp` ≈ −80,000.

## 10. Verdict

**Family B is NOT PREDICTABLE at serve time.** Nine rows in 2,084,659 (0.04% of unmatched rows,
0.0004% overall), isolated in time, at four airports and eight airlines, on unremarkable airport-days,
with BLOCK stamps that match no other record and no serve-time column whose within-airport
distribution differs from the rest of the unmatched stratum. Every apparent signal was the airport
cut or a post-hoc choice; nothing reached an out-of-sample test because nothing had an effect size to
test. The prior verdict stands and is now closed against the matched side, the broader BLOCK-early
population, timestamp provenance, temporal clustering and airport-day statistics as well.

Irreducible cost: **≈ 116–122 s in quadrature on the 2026 score in expectation (79–158 s across the
Poisson 10–90% band)**; the correct treatment of a scored row is the cell mean, never a monster value.
Stop here on Family B. The action items that come out of this report are elsewhere: the mangled
designator column for the slip/fill split (§9.1), and the re-examination of the matched side that §8
implies.
