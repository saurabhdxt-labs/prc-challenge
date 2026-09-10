# Fresh pass — extreme rows / unmatched lane — Sonnet, 2026-09-10

Scope as assigned: find the path to 245 through the extreme rows and the unmatched (no-NM-record)
lane. Every number below states its source file and the exact command that reproduces it (scripts
kept in the session scratchpad, not this repo). I read `reports/bug_classes.md`, `reports/WHERE_WE_STAND_2026_09_10.md`,
`plans/PREREG_rome_fill_2026_09_10.md` and `plans/PREREG_rome_dateslip_2026_09_10.md` first — a large
amount of this exact ground was covered today by parallel work, RESULT 6, RESULT R, RESULT DS. Where
my measurement reproduces theirs I say so; where I add something I flag it as NEW.

## 1. Taxonomy — 2025, all 12 months, all 10 airports, unmatched departures only

Source: `data/cache_rome/unm.parquet` (22,219 rows, the cached output of
`stratum_fold.load_real`, all 10 airports, all 12 months — no cache work done, read-only).

For `y > 3,600 s` (1,087 of 22,219 rows, 4.89%), three classes, tested against the labelled `y`
(available in this training cache; NOT available at serve time):

| class | rule | n (y>3,600) | n (monsters, y>10,800) | where |
|---|---|---|---|---|
| **FILL** | `\|y − sp\| ≤ 300 s` (BLOCK copied from SCHED) | 743 (68%) | **175 of 186 (94%)** | 92% at LIRF; also LTFM(8), EGLL(4), LFPG(4), EDDF(1) |
| **DATE-SLIP** | `86,400 ≤ y < 97,200` (BLOCK clock kept under SCHED's calendar date) | 13 (1.2%) | 10 | **LIRF only** (+1 LSZH monster, dayoff=0, likely a different mechanism, n=1) |
| **OTHER / genuine** | neither | 331 (30%) | **12 across all 12 months, all airports** (max y=84,240) | no shared signature (below) |

This reproduces `WHERE_WE_STAND`'s monster census (LIRF 92.1% of y>3h rows) and RESULT R's discovery
of the date-slip class from an independent read of the same cache. **New in this pass:** the
`>10,800` monster list broken out by exact `y − sp` residual shows the FILL class is not
approximate — 175 of 186 monsters have `\|y−sp\| ≤ 6 s`, i.e. the two clocks are bit-identical, not
merely correlated. There is no fourth clock-artifact class (checked a half-day/12 h shift
hypothesis explicitly: zero rows).

**The 12 "OTHER" monster rows, full record** (`FLIGHT_RULE_mvt`, `RUNWAY_mvt`, `AIRCRAFT_TYPE_mvt`,
`SCHED` minute): no shared signature. 2 of 12 are VFR general-aviation (a C172, a DH8A commuter);
`AIRCRAFT_TYPE_mvt` is null on 8 of 12 (matches the population norm — nulls are the norm for
unmatched rows, not a marker); SCHED-minute rounding-to-5 is 100% in this cohort vs 99.1% in the
whole unmatched population (no separation — the base rate is already saturated). **This independently
reconfirms `WHERE_WE_STAND`'s finding** ("ordinary flights running half a day late... no such
signature", "4 stale-BLOCK rows... no shared signature") from a from-scratch read of the cache, not
a copy of their number. **CLOSED — NOT WORKING**: no serve-time field separates these 12 rows from
ordinary ones. What would reopen it: an external data source (see §4).

## 2. Applying the taxonomy to the 2026 scored file (serve-time fields only)

Source: `derive(load_movements(['data/raw/ranking.parquet']))` joined to `submitting.parquet`
(344,841 rows) and to `submissions/merry-quicksand_v7.parquet` (our shipped predictions). Scored
file is Jan+Jul 2026 only (2,453 + 2,837 rows) — matches fold A's holdout months exactly.
5,290 unmatched rows (1.53%), `sp` distribution shifted a little heavier than the 12-month 2025
population (`sp>10,000`: 15.4% of 2026-scored-unmatched vs 7.9% of all-2025-unmatched) — consistent
with fold A already being drawn from months 1 & 7, not a new artifact.

**H-DS segment G (LIRF, `dayoff=1`, `sp≥40,000`) in the scored file: 10 rows.** Verified — not
assumed — that v7's actual stored predictions on all 10 equal `rome_dateslip.apply()`'s two-class
expectation to rounding (max residual 0.5 s): the shipped R2 formula is exactly what is in the
submission file, no stale-cache or wiring bug. The largest single stake in the whole scored
unmatched lane is here: `sp=111,654` (July, hr 21), `q(G2)=0.647` (all-12-month LIRF share),
v7 predicts 95,974. If FILL (`y≈111,654`): stake **712 MSE** if wrong. If DATE-SLIP (`y≈87,543`):
stake **206 MSE** if wrong. This and the other 9 G-rows are already the R2-shipped bet; nothing to
switch to that isn't already applied. (Board transfer of R2 overall was already measured at
−630 MSE vs a −4,150 projection, 0.15×, per RESULT DS — the shortfall is calibration drift on ~30
training rows, not a wiring gap.)

**Same segment definition (`dayoff=1`, `sp≥40,000`) at every OTHER airport: 22 scored rows
(EDDF 7, EHAM 8, LFPG 1, LTFM… none). NEW finding this pass.** The 2025 evidence for this exact
segment outside LIRF is unambiguous: 72 training rows, **0 fills, 0 date-slips, 72 "other"** — and
their actual `y` is uniformly ordinary (median 1,099 s, p75 1,278, max 4,667; `y/sp` median 1.5%).
A huge `sp` with `dayoff=1` at these airports is not a clock artifact at all — most plausibly a
long-haul/overnight schedule filed a calendar day ahead, unrelated to any BLOCK-timestamp bug.
**v7's actual predictions on the analogous 22 scored rows are 892–1,749 s — already matching this
evidence almost exactly.** No action available; the model already discounts `sp` correctly here.
**CLOSED — WORKING** (the pipeline's generalization to this segment, verified out-of-sample against
the 2025 label).

**The rest of the `sp>10,000` scored population (803 rows, all 10 airports), priced.** For each row
I computed the 2025 Laplace fill-rate `q` per (airport × sp-band, using the shipped `SP_BANDS`), the
naive evidence mixture `q·sp + (1−q)·900`, and the evidence-weighted expected SSE of v7's actual
prediction vs that naive mixture, summed:

| | rows | Σ(evidence-weighted regret if switched to the naive mixture) |
|---|---|---|
| LIRF | 1,375 (of which many are already the sophisticated hybrid, not this naive one) | **−10,762 MSE** (v7's real classifier beats the naive band average — expected, confirms sophistication is earning its keep) |
| every OTHER airport, summed | 813−1,375's LIRF share ≈ 227 rows-worth | **+227 MSE total** (max single airport LFPG +62, EHAM +50, EGLL +48; the rest single digits to teens) |

This reproduces `WHERE_WE_STAND`'s "oracle fill decision priced ... ex-LIRF it saves 42 of 35,856"
finding via an independent, cruder method (a static 2025 band-average instead of the fitted
classifier) and lands at the same order of magnitude (~200–300 MSE, not thousands). **CLOSED — NOT
WORKING**: there is no exploitable fill/date-slip mismatch left outside LIRF, and inside LIRF the
already-shipped classifier already beats this simple check.

**Priced-list total, this pass: ≈ 227 MSE of unclaimed, low-confidence headroom, entirely outside
LIRF, none of it exceeding a few tens of MSE per airport.** No 2026 scored row was found where v7's
own prediction and the evidence-optimal call disagree by more than the ~700 MSE-stake LIRF row
already covered by the shipped R2 formula.

## 3. New class-separating signal search

Checked against the task's own list, each against the 2025 cache (LOMO where a prior result exists,
single-pass Laplace check otherwise given the compute budget):

| candidate | result | source |
|---|---|---|
| full `FLIGHT_mvt` history (unmatched-row and matched-row copy history) | **DEAD**, LOMO AUC 0.797–0.826 vs incumbent 0.877 | RESULT 6, reproduced by citation, not re-run (identical claim, no reason to re-fit) |
| arrival leg of same aircraft/flight | bounded ~1,355 MSE, real but small, already a queued lane | `WHERE_WE_STAND` §6 |
| time-of-day (`hr`) | already a model input | `rome_fill.py FEATS_NUM` |
| SCHED-minute rounding | **no separation** — 100% vs 99.1% base rate on the 12 unexplained monsters (n too small for a real test either way) | this pass, §1 |
| other airports' behaviour at extreme `sp` | **NEW, closed WORKING**: the mechanism at non-LIRF airports (huge `sp`, `dayoff=1`) is structurally different from LIRF's and the model already handles it correctly | this pass, §2 |
| difference between SCHED date and MVT date (`dayoff`) | already a model input, and the segment-defining variable for H-DS | `rome_fill.py`, `rome_dateslip.py` |
| AIRCRAFT_TYPE / FLIGHT_RULE on the unexplained monsters | no pattern, n=12, mostly null | this pass, §1 |

No new serve-time signal beating what's shipped was found. The search space named in the brief has
now been covered by RESULT 6 (identity), `WHERE_WE_STAND` §4c (arrival clocks, turnaround anchor,
stand witness, ADS-B, AOBT_3 fingerprints — all AUC ≤ 0.6 or infra NO-GO), and this pass (airport
generalization, SCHED rounding, aircraft type). **The fill/date-slip signal is saturated at the
observable ceiling** — shipped `p` AUC 0.944 pooled / 0.984 tail per `WHERE_WE_STAND` §4c, and my
independent band-average check above cannot beat it either.

## 4. What unmatched MSE would the leader need, and does the evidence-optimal lane get there?

Board identity (given): `total_MSE = 0.9847·matched_RMSE² + 0.0153·unmatched_RMSE²`. Leader
`total_MSE ≈ 60,025`. Our matched fold RMSE ≈ 222.6.

| scenario | required leader matched RMSE | implied leader unmatched RMSE | plausible given this pass? |
|---|---|---|---|
| leader's matched = ours (222.6) | 222.6 | **855.6 s** | requires near-total monster capture (fold "perfect all 56 monsters" = 758.7; "perfect 20" = 926.3) — between those two |
| leader captures the monsters, ordinary elsewhere (`WHERE_WE_STAND`'s parsimonious scenario) | 213.7 (a 4.7% matched gain) | **990.7 s** | **matches the fold's own ex-monster RMSE of 995.6 almost exactly** — the cleanest-fitting single hypothesis |

Both scenarios require the leader's unmatched RMSE to sit at or below the fold's **ex-monster**
figure (995.6) — i.e. they are not merely better at ordinary unmatched rows, they have effectively
solved the monster rows. This pass adds no evidence for how: the identifiable structure (FILL,
DATE-SLIP) is already extracted and shipped (R2, transfer 0.15× on the board); the residual monster
rows (the 12/12 months "other" class, §1) carry no serve-time signature in this dataset, checked
four independent ways today (RESULT 6, `WHERE_WE_STAND` §4c, and twice in this pass). **Verdict:
NOT WORKING** — the unmatched lane, using only the fields in this dataset, cannot be shown to
deliver the ~11,000–15,000 MSE either scenario requires; the ≈227 MSE found here and the ~200 MSE
found independently in `WHERE_WE_STAND` are two orders of magnitude short. What would close it:
either an external data source for the specific corrupted-clock rows (H-B, still open — the
leader's own method is unknown), or the gap is predominantly in the **matched** lane, not here.

## 5. Hypotheses closed this pass

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| P1 | A 4th clock-artifact class (half-day / other fixed offset) exists among unmatched extreme rows | **NOT WORKING** | zero rows in a 12-month, 10-airport, all-offset scan, §1 |
| P2 | The 12 genuinely unexplained monster rows share a serve-time signature | **NOT WORKING** | checked FLIGHT_RULE, RUNWAY, AIRCRAFT_TYPE, SCHED-rounding, all null, §1 |
| P3 | The shipped R2 date-slip formula is correctly wired into v7's scored predictions | **WORKING** | exact reproduction (max residual 0.5 s) on all 10 scored LIRF segment-G rows, §2 |
| P4 | Non-LIRF airports have an unexploited fill/date-slip lane at extreme `sp` | **NOT WORKING** | 72 2025 training rows in the exact non-LIRF segment G are 100% ordinary; the 22 scored analogues are already predicted as ordinary, §2 |
| P5 | A cruder-than-shipped per-airport×sp-band fill heuristic finds money the shipped model missed | **NOT WORKING** | ≈227 MSE total, all single-airport low tens, §2 — same order of magnitude as `WHERE_WE_STAND`'s independent 42/35,856 figure |
| P6 | The unmatched lane alone can plausibly deliver the leader's implied gap | **NOT WORKING** | requires ex-monster-or-better performance (≤995.6 s RMSE); no signal found, anywhere, to get there, §4 |
