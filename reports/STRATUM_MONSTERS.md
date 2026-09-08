# The unmatched stratum and the monster rows — PRC Data Challenge 2026

Generated 2026-09-08. Every number below is copied from command output saved under the session
scratchpad (`.../scratchpad/strat/{extract,extract2,explore1,explore2,explore3,run1,run2,run3,run4}.py`
and their `.out` files; `harness.py` and `common.py` hold the shared code). Runs used
`OMP_NUM_THREADS=1 nice -n 19 python3.11`; peak RSS of the heaviest run was **0.36 GB** (`/usr/bin/time -l`).
Re-running `run2.py` reproduced its output byte-for-byte. Nothing under the repository was modified
except this file.

## 0. Scope, data, definitions

| item | value |
|---|---|
| stratum | `AOBT_3_flt` null, PHASE=DEP, after `prc.labels.clean` — **22,219** rows in 2025 (reproduced exactly), **5,290** scored rows in 2026 (reproduced exactly) |
| monster | `y > 10,800 s` — **186** rows (LIRF 180, LFPG 2, LSZH 2, EGLL 1, EHAM 1) |
| `sp` | `MVT_TIME − SCHED_TIME` (s), computable on every scored row |
| `d` | `BLOCK_TIME − SCHED_TIME` (s), training only |
| fill | `|d| ≤ 60`; on a fill row `y == sp` by construction |
| `dayoff` | `date(MVT) − date(SCHED)` in days, clipped to {0, 1} — computable on every scored row |
| coarse buckets | `[<15m, 15–30m, 30–60m, 1–2h, 2–3h, 3–6h, 6–12h, 12h+]` (the caller's) |
| fine buckets | coarse below 3 h, then `3–4h, 4–5h, 5–6h, 6–8h, 8–12h, 12–18h, 18h+` |
| validation | 12-fold leave-one-month-out over all 22,219 rows; every cell statistic, target encoding, classifier and calibrator fitted inside the training months only; calibrators fitted on *inner* LOMO out-of-fold probabilities within the training months |
| intervals | paired row bootstrap of the RMSE difference (2,000 resamples, seed 0); `*` = interval excludes zero. For the headline claims a paired **month-block** bootstrap (12 months resampled with replacement, 4,000 resamples) is reported as well |
| monster RMSE | RMSE over the 186 monster rows only, always reported next to the stratum RMSE |

Epoch seconds were taken as `ts.astype("int64") // 1_000_000` throughout (parquet timestamps are `timestamp[us]`).

## 1. Findings at a glance

1. **The baseline reproduces**: per-(airport, coarse bucket) mixture, LOMO stratum **1,655.4**, monster **13,518** (caller: 1,648.4 / 13,522; the residual is fallback handling on empty cells).
2. **The monsters are three families, not two, and only one of them is a fill.**
   - **Fill monsters (164 rows)**: `y = sp`, all LIRF. 10.0% of stratum squared error under the baseline.
   - **24-hour date-slip monsters (12 rows)**: all LIRF, *not* fills, `y = 86,400 + ordinary taxi` (y ∈ 87,168–88,392; y−86,400 ∈ 768–1,992 s). Mechanism verified row by row: `BLOCK` carries **SCHED's date with the real time-of-day** (11/12 have `date(BLOCK) = date(SCHED)` while `date(MVT) = date(SCHED)+1`; the 12th is a SCHED just after midnight, `date(BLOCK) = date(SCHED)−1`). 14.2% of stratum squared error. **They live in the same airlines, stands and destinations as fills; only `sp` magnitude and `dayoff` separate them.**
   - **Family B (10 rows)**: `sp` ordinary (−364 … 2,043 s), `BLOCK` recorded 3–24 h before SCHED, y 11,032–87,341. Six are not at LIRF. Nothing observable at serve time distinguishes them. **31.7% of stratum squared error, 56% of monster squared error.**
3. **The calibration-limited hypothesis is NOT WORKING.** The rich LIRF classifier is already calibrated on held-out folds (decile means vs realised fill rate on LIRF `sp > 3h`: 0.22/0.16, 0.50/0.63, 0.72/0.71, 0.90/0.90, 0.99/0.98); isotonic recalibration moves the stratum by −6.8 [−18.3, +2.4] and monsters by −57 [−217, +55] — not established. Platt is a significant *loss* (+1.3 [+0.5, +2.5]).
4. **Why a better classifier does not move the monsters**: the classifier only acts on fill monsters (10% of stratum SSE) and it *hurts* the 24h-slip rows, which look like fills (slip RMSE 26,810 → 29,763 under the rich model) while it helps the fills (6,080 → 5,362). Family B (56% of monster SSE) is untouched by anything. Net monster change +129 [−414, +658] — noise. **Monster RMSE is floor-limited, not classification-limited and not calibration-limited.**
5. **The pooled AUC is an artifact of the airport cut.** The caller's cell-rate `p` has pooled AUC 0.918 but **within LIRF 0.559** and within LIRF `sp > 3h` **0.449**. All useful classification is within LIRF; the "0.688 → 0.935" story is mostly airport separation that the per-airport cell already had.
6. **What is established** (paired intervals exclude zero, row and month-block): a LIRF-only logistic fill model with in-fold target-encoded **airline prefix**, the **airline's schedule-copy propensity measured on LIRF's 159,216 matched rows** (`|d| ≤ 60 & |BLOCK − AOBT_3| > 300`), stand first character and destination prefix, inside the same mixture: stratum **1,574.5** (−80.9 [−133, −35] row; [−152, −20] month-block). The gain is on the non-monster LIRF rows (1,105 → 963), i.e. the 1–3 h fill/ordinary separation. Monster RMSE 13,647 (unchanged within noise). A permutation control (airline/stand/destination shuffled inside each training fold) gives 1,654.0 — the baseline — so the gain is signal, not leakage.
7. **Best stratum number: 1,551.7** (sp²-weighted logistic + inner-LOMO isotonic), monster **13,398**. Versus the logistic reference: stratum −22.8 [−46.3, −4.4] (row), [−45.9, −0.5] (month-block); monster −250 [−610, −12.5] (row), [−595, +19] (month-block). **Borderline — the intervals graze zero and this was one of eight shaping variants tried; treat as suggestive, not established.**
8. **Precise floors (2026 scored set, 344,841 rows)**: Family B alone is an expected **2.4–2.5 rows** with mean squared error 1.93e9 → **115–119 s of total RMSE in quadrature** (Poisson 10th–90th percentile of the count 1–5 rows → 75–167 s). The 24h-slip rows add ≈ 80–91 s under every model tried. Together ≈ 145 s of the 312.4 is unreachable by this stratum's features.
9. **Total-score translation**: the established gain (1,655 → 1,575) is worth ≈ **−6.5 s** on the total (312.4 → 305.9); the borderline variant ≈ −8.3 s (→ 304.1). The caller's "eliminating monster error takes us to 270.1" counts the ≈ 145 s of irreducible floor; the reachable monster gain is far smaller.

## 2. The anatomy that explains everything below

### 2a. Stratum squared error by group, baseline mixture (LOMO)

| group | n | RMSE in group | share of stratum SSE |
|---|---|---|---|
| fill monsters (LIRF, `y = sp`) | 164 | 6,080 | 0.100 |
| 24h-slip monsters (LIRF, `y ≈ 86,400 + taxi`) | 12 | 26,810 | 0.142 |
| Family B monsters (`sp` ordinary, BLOCK hours early) | 10 | 43,936 | **0.317** |
| ordinary rows with `sp > 3h` (false-positive cost) | 1,174 | 2,490 | 0.120 |
| LIRF non-monster, `sp ≤ 3h` (1–3 h fill/ordinary mixing) | 1,241 | 2,991 | 0.182 |
| non-LIRF non-monster, `sp ≤ 3h` | 19,618 | 659 | 0.140 |

Stratum RMSE with one group's error set to zero: fill monsters → 1,570.9; slips → 1,533.7; Family B → **1,368.1**; ordinary `sp > 3h` → 1,553.3.

### 2b. Oracle bounds (in-fold cell means, oracle indicators)

| oracle | stratum RMSE | monster RMSE |
|---|---|---|
| none (baseline) | 1,655.4 | 13,518 |
| perfect fill indicator | 1,303.7 | 11,891 |
| perfect fill + perfect slip indicator | 1,170.0 | 10,095 |
| Family B alone, everything else perfect | 923.6 | **10,095** |

So even a perfect classifier leaves monster RMSE at ≈ 10,100; the whole reachable monster range is 13,518 → 10,095, and the reachable *stratum* range from classification is 1,655 → 1,304.

### 2c. LIRF `sp > 3h` (243 rows): what separates the three groups

| cut | fill | ordinary | slip |
|---|---|---|---|
| `dayoff = 0` (189) | 123 | 65 | 1 |
| `dayoff = 1` (54) | 41 | **2** | **11** |
| `sp` 3–4 h (116) | 63 | 52 | 1 |
| 4–5 h (41) | 30 | 11 | 0 |
| 5–6 h (23) | 20 | 3 | 0 |
| 6–8 h (13) | 12 | 1 | 0 |
| 8–12 h (21) | 21 | **0** | 0 |
| 12–18 h (22) | 17 | 0 | 5 |
| 18 h+ (7) | 1 | 0 | **6** |

- Ordinary taxi-outs with `sp > 8h` do not occur at LIRF in 2025 (0/51). Above 12 h the only question is fill vs slip.
- The 24h-slip rows are 11/12 `dayoff = 1`; slips at `dayoff = 0` are 1/189.
- Airline prefix inside this cell: ISR 13/13, AEZ 7/7, TAR 6/6, LAV 6/6, TWB 4/4, ARG 4/4, ETH 4/4, TKJ 3/3 fill; BAW 0/4, CYF 1/4, UAL 1/4; RYR 7/20, ITY 7/19, WMT 15/22. Destination prefix: SB 8/8, DT 8/8, LR 7/7 fill; KJ 0/4. Stand first character: `4` 0/9, `2` 17/19.
- Airline schedule-copy propensity on LIRF **matched** rows (n ≥ 200): ISR 0.88, TKJ 0.87, TWB 0.63, KMM 0.51, UPS 0.50, EXS 0.28, AEZ 0.25 … RYR 0.07, VLG 0.07 — the same ordering as the unmatched fill rates, from a 159k-row source.
- Ordinary rows here have y p50 1,090 / p90 2,071 / p99 6,478 and `d` p50 11,217: genuine late pushbacks with ordinary taxi.
- Non-LIRF `sp > 3h` rows (1,107): y mean 1,279 at `dayoff = 0`, 2,265 at `dayoff = 1` (n = 95), max 10,309; no fills to speak of.

## 3. Experiments — all LOMO, all paired against the baseline

Format: stratum RMSE / monster RMSE; paired difference [95% row bootstrap].

### 3a. Cell refinement (mechanism-guided cells, hierarchical shrinkage with k pseudo-counts)

| estimator | stratum | monster | vs baseline (stratum) | vs baseline (monster) |
|---|---|---|---|---|
| E0 baseline: (airport × coarse), k = 0 | 1,655.4 | 13,518 | — | — |
| E0 shrunk k = 2 | 1,651.1 | 13,502 | −4.3 [−27, +20] | −16 [−396, +353] |
| E0 shrunk k = 5 | 1,675.8 | 13,896 | +20 [−25, +70] | +378 [−299, +1,197] |
| E0 shrunk k = 10 | 1,731.4 | 14,725 | +76 [+6, +159] * | +1,206 [+238, +2,550] * |
| E1 + fine buckets, k = 2 / 5 / 10 | 1,657.1 / 1,666.2 / 1,697.0 | 13,491 / 13,652 / 14,136 | +1.7 / +10.8 / +41.6, none established | none established |
| E2 + fine + `dayoff`, k = 2 / 5 / 10 | 1,661.0 / 1,658.7 / 1,677.7 | 13,636 / 13,639 / 13,939 | +5.6 / +3.3 / +22.3, none established | none established |
| E2 vs E1 (k = 5) | | | −7.5 [−20, +4] | −13 [−174, +161] |

**Verdict: NOT WORKING.** Finer `sp` cells and the `dayoff` split are mechanistically right (§2c) but redundant with the coarse bucket (`12h+` is already almost all `dayoff = 1`) and the extra cells are too thin: what they gain in sharpness they lose in sampling noise. Shrinking `p` toward the parent cell is *harmful* to the monsters — the `12h+` cell's own rate is what the monsters need.

### 3b. LIRF logistic fill model inside the mixture (cells elsewhere; nonfill mean from E2 cells, k = 5)

| features | stratum | monster | AUC LIRF / LIRF `sp>3h` | vs baseline (stratum) | vs baseline (monster) |
|---|---|---|---|---|---|
| L-a fine bucket + dayoff | 1,647.1 | 13,537 | 0.573 / 0.624 | −8 [−41, +27] | +19 [−483, +551] |
| L-b + airline prefix (in-fold, LOO-smoothed) | 1,600.3 | 13,749 | 0.854 / 0.841 | −55 [−105, −11] * | +231 [−290, +741] |
| L-c + matched-row copy propensity | 1,598.8 | 13,828 | 0.861 / 0.853 | −57 [−104, −12] * | +309 [−186, +856] |
| L-d + stand first char | 1,597.8 | 13,846 | 0.865 / 0.856 | −58 [−107, −11] * | +328 [−195, +870] |
| **L-e + destination prefix** | **1,574.5** | 13,647 | 0.885 / 0.860 | **−81 [−133, −35] *** | +129 [−414, +658] |
| L-f + hour of day (sin/cos) | 1,579.4 | 13,738 | 0.887 / 0.865 | −76 [−130, −28] * | +220 [−335, +781] |
| L-g matched propensity only (+fb, dayoff) | 1,623.0 | 14,121 | 0.838 / 0.816 | −32 [−79, +15] | +603 [+46, +1,362] * (worse) |
| L-e vs L-c (is the destination prefix established?) | | | | −24 [−54, +2] — **not established** | −180 [−636, +131] |
| L-e month-block bootstrap vs E0 | | | | −81 [−152, −20] * | +129 [−738, +782] |

Regularisation (L-e): C = 0.1 → 1,622 (+48 [+21, +85], worse); C = 0.3 → 1,595 (+20 [+4, +40], worse); C = 3 → 1,569 (−6 [−20, +9]); C = 10 → 1,570 (−4 [−27, +23]). C = 1 kept.

Per-fold stratum RMSE, E0 → L-e: Jan 2,920 → 2,905; Feb 2,197 → 2,132; Mar 1,023 → 1,070; Apr 1,016 → 1,012; May 2,464 → 2,406; Jun 1,951 → 2,002; Jul 1,449 → 1,317; Aug 1,504 → 1,176; Sep 1,217 → 980; Oct 962 → 863; Nov 1,146 → 1,121; Dec 1,054 → 941. Ten of twelve months improve; March and June worsen by ≈ 50.

**Negative control**: L-e with airline / stand / destination permuted inside each training fold → 1,654.0, AUC LIRF 0.588 — back to the baseline. The gain is carried by the real encodings.

**Verdict: WORKING (established) for the stratum**, on the mechanism "airline identity predicts which LIRF records are schedule copies". Coverage: all 12 months, all ten airports (the model only alters LIRF). Not covered: airline prefixes unseen in 2025 (13 of 383 LIRF scored rows fall back to the prior); any change in LIRF's handling-agent mix in 2026. **Monster RMSE: no established movement.**

### 3c. Calibration and squared-error shaping of `p` (all vs L-e)

| variant | stratum | monster | vs L-e (stratum) | vs L-e (monster) |
|---|---|---|---|---|
| isotonic, fitted on inner-LOMO OOF `p` | 1,567.7 | 13,590 | −6.8 [−18, +2] | −57 [−217, +55] |
| Platt, same | 1,575.8 | 13,673 | +1.3 [+0.5, +2.5] * (worse) | +26 [+14, +47] * (worse) |
| shrink 0.25 toward cell `p` | 1,573.0 | 13,514 | −1.5 [−14, +11] | −133 [−314, +42] |
| shrink 0.5 toward cell `p` | 1,586.8 | 13,468 | +12 [−13, +41] | −179 [−518, +185] |
| cap `p` ∈ [0.02, 0.98] | 1,574.6 | 13,648 | +0.1 [0, +0.1] * | +0.8 [+0.4, +1.5] * |
| cap `p` ∈ [0.05, 0.90] | 1,578.2 | 13,689 | +3.7 [+1.6, +5.9] * (worse) | +41 [+13, +81] * (worse) |
| sp²-weighted logistic (weight ∝ (sp − nonfill mean)²) | 1,558.0 | 13,397 | −16.5 [−39, +2] | −251 [−622, +3] |
| **sp²-weighted + isotonic (W)** | **1,551.7** | **13,398** | −22.8 [−46, −4] * | −250 [−610, −13] * |
| W month-block bootstrap vs L-e | | | −23 [−46, −0.5] | −250 [−595, +19] |
| W month-block bootstrap vs E0 | | | −104 [−182, −34] * | −121 [−1,064, +577] |

Reliability of L-e `p` on held-out folds, LIRF all rows, deciles (mean `p` / realised): 0.05/0.06, 0.12/0.09, 0.21/0.22, 0.27/0.18, 0.41/0.51, 0.51/0.53, 0.64/0.63, 0.77/0.76, 0.91/0.89, 0.99/0.98. LIRF `sp > 3h`, quintiles: 0.22/0.16, 0.50/0.63, 0.72/0.71, 0.90/0.90, 0.99/0.98. The caller's cell `p` on the same rows: 0.49/0.71, 0.57/0.63, 0.67/0.39, 0.76/0.77, 0.94/0.88 — the *cell* rate is the miscalibrated one.

**Verdicts.** Calibration-limited: **NOT WORKING** — the logistic `p` is already calibrated; isotonic is inside noise, Platt is a significant loss, capping and shrinking do nothing or hurt. Squared-error weighting: **borderline** — W beats L-e with intervals that graze zero on both metrics and it is one of eight variants; it is the best number on the table but not an established gain over L-e. It *is* established over the baseline (month-block −104 [−182, −34]).

### 3d. Other estimators

| estimator | stratum | monster | comparison |
|---|---|---|---|
| per-cell argmin of in-fold SSE over {mixture, `sp`, nonfill mean} (E2 cells) | 1,681.2 | 14,014 | vs E2: +22 [−2, +57]; monster +375 [+28, +906] * (worse) |
| 3-class multinomial at LIRF (fill / 24h-slip / ordinary), `pred = p_f·sp + p_s·87,500 + p_o·ordinary mean` | 1,628.3 | 14,224 | vs L-e: +54 [−6, +167]; slip rows 34,869 vs 29,763 (worse), fill monsters 4,814 vs 5,362 (better) |
| 3-class + hard rule `sp > 18h → 87,500` | 1,633.0 | 14,289 | vs L-e: +59 [−21, +179]; the one 131,163 s fill takes a 43.7k error |
| HistGradientBoosting classifier, same features (depth 3, 150 iter) | 1,698.5 | 14,922 | vs L-e: +124 [+66, +199] * (worse); AUC LIRF 0.807 |
| HGB + isotonic | 1,659.5 | 14,610 | +85 [+43, +139] * (worse) |
| HGB sp²-weighted + isotonic | 1,665.1 | 14,511 | +91 [+50, +144] * (worse) |

**Verdicts.** Per-cell hard selection: NOT WORKING (the mixture is the in-cell squared-error optimum; selection only adds variance). Explicit slip modelling: NOT WORKING — with 12 slip rows the multinomial spreads slip mass onto the wrong rows; the mixture's per-cell nonfill mean (87.5k in `12h+`) already carries the slip expectation. Tree classifier: NOT WORKING at this sample size (consistent with the caller's GBM two-stage result).

### 3e. Group decomposition under E0 / L-e / W (RMSE in group; share of stratum SSE)

| group | n | E0 | L-e | W |
|---|---|---|---|---|
| fill monsters | 164 | 6,080 (0.100) | 5,362 (0.086) | 4,243 (0.055) |
| 24h-slip monsters | 12 | 26,810 (0.142) | 29,763 (0.193) | 30,459 (0.208) |
| Family B monsters | 10 | 43,936 (0.317) | 43,930 (0.350) | 43,931 (0.361) |
| ordinary rows `sp > 3h` | 1,174 | 2,490 (0.120) | 2,130 (0.097) | 2,121 (0.099) |
| LIRF non-monster `sp ≤ 3h` | 1,241 | 2,991 (0.182) | 2,334 (0.123) | 2,285 (0.121) |
| non-LIRF non-monster `sp ≤ 3h` | 19,618 | 659 (0.140) | 653 (0.152) | 653 (0.156) |

This is the answer to the central puzzle in one table: every fill-classification gain (fill monsters 6,080 → 4,243, LIRF 1–3 h 2,991 → 2,285, ordinary large-`sp` 2,490 → 2,121) is paid back on the 12 slip rows (26,810 → 30,459), and Family B does not move. After W, **Family B + slips are 57% of stratum SSE**.

## 4. The irreducible floor, precisely

Family B: 10 rows in 22,219 (EGLL 1, EHAM 1, LFPG 2, LIRF 4, LSZH 2). Expected 2026 count: pooled rate × 5,290 = **2.38**; per-airport rates × 2026 per-airport counts = **2.53**. Mean squared error per row at the best available prediction (the cell mean): **1.930e9** (per-row errors 3,702 … 86,525; RMSE within group 43,930). Floor on the 344,841-row score:

- λ = 2.38 → **115.4 s** in quadrature; Poisson 10th–90th percentile of the count 1–4 rows → 75–150 s.
- λ = 2.53 → **119.0 s**; count 1–5 rows → 75–167 s.

The 24h-slip rows (12 of 1,488 LIRF unmatched → expected **3.09** in 2026) add **80 s** (E0), **89 s** (L-e), **91 s** (W) in quadrature under the respective models; the oracle-cell treatment of §2b would bring them to ≈ 0 only with an indicator this data does not contain.

Stratum in total-score terms: E0 contributes 205.0 s in quadrature (monster rows alone 153.2); L-e 195.0 (154.7); W 192.2 (151.8). At a total of 312.4, E0 → L-e gives **305.9**, E0 → W gives **304.1**. With Family B at zero and everything else as is, 312.4 would be 290.3 — that is the part of the caller's "270.1" that cannot be reached from this stratum's features.

## 5. Recommended stratum recipe (what to ship for the 5,290 rows)

Cells for every airport: `p` = in-fold fill rate and `nf` = nonfill mean, hierarchical `[ap] → [ap, coarse] → [ap, fine] → [ap, fine, dayoff]`, k = 5 (E2; equivalent to the baseline within noise, kept because the nonfill means feed the LIRF model). At LIRF replace `p` with the logistic model of §3b (L-e): fine-bucket one-hot, `dayoff`, in-fold leave-one-out-smoothed log-odds of the unmatched fill rate by airline prefix (k = 5), log-odds of the matched-row schedule-copy rate by airline prefix (k = 20, from the LIRF matched rows of the training months, `|d| ≤ 60 & |BLOCK − AOBT_3| > 300`), stand first character and destination two-letter prefix encoded the same way; C = 1. Prediction `p·sp + (1 − p)·nf`. Optionally the W variant (sample weight ∝ (sp − nf)² floored at 1e4, isotonic on inner-LOMO OOF `p`) — best number, not an established improvement over L-e.

Fitted on all twelve months and applied to the 2026 scored unmatched rows (`scratchpad/strat/pred_2026_unmatched.parquet`, columns `MVT_ID_mvt, ap, sp, p_Le, pred_E0, pred_Le, pred_W`): LIRF 383 rows, 74 with `sp > 3h`, mean prediction 7,156 (E0) / 7,256 (W); every other airport predicts below 4,525 everywhere, and EHAM's 297 disruption rows with `sp > 3h` receive p50 852 / max 1,671 — the airport-conditional `p` is protective as intended. **One scored row deserves a name**: LIRF, `NOS`, `sp = 111,654` (31 h), `dayoff = 1`, predicted 102,513 (L-e) / 105,228 (W). In 2025 the only `sp > 100k` LIRF row was a fill (131,163) and the 18 h+ cell is 6 slips + 2 fills; if this row is ordinary (~1,000 s) the prediction costs ≈ 44 s of total RMSE on its own, and predicting 1,000 s would cost the same if it is a fill. The data supports the prediction (0 ordinary rows above 8 h in 2025); it is the single largest bet in the submission and should be known as such.

## 6. What was looked for and not found

| check | result |
|---|---|
| miscalibration of the rich `p` as the reason monsters did not improve | no — held-out reliability is flat within noise on LIRF and on LIRF `sp > 3h`; isotonic/Platt/cap/shrink do not help |
| a serve-time indicator for the 24h-slip rows beyond `sp` and `dayoff` | none: same airlines (ITY 4/19, BAW 1/4, NOS 1/8, EXS 1/5 …), same stands, same destinations as fills |
| a serve-time indicator for Family B | none: `sp` −364 … 2,043 s, six airports, four airlines |
| an ordinary taxi-out at LIRF with `sp > 8h` | none in 2025 (0/51) — `p` may be taken as 1 minus the slip probability there |
| any airport other than LIRF with schedule-fill monsters | none (EGLL's one monster is a genuine 3.1 h taxi, `d = −604`) |
| a gain from hour-of-day, from tree classifiers, from explicit 3-class modelling, from hard per-cell selection | none — all inside noise or established losses |
| a change in the LIRF feed between 2025 and 2026 | none: `AIRCRAFT_TYPE_mvt` null on 100% of LIRF scored unmatched rows; `sp > 3h` share 19.3% vs 20.9%; `dayoff = 1` share among `sp > 3h` 27% (20/74) vs 22% (54/243) |
