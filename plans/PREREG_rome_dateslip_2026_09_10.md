# Pre-registration — H-DS: Rome's 24-hour date-slip class

**Written 2026-09-10 10:03 local, BEFORE the script exists and before
any of its numbers exists.** Standalone file. Tag: **`DS`**. Follows RESULT R
(`plans/PREREG_rome_fill_2026_09_10.md`), which located the class; this is a NEW hypothesis, not a
rescue of R, and R's failed clauses stand.

## Hypothesis

> On LIRF unmatched rows whose take-off is on the calendar day after the schedule (`dayoff = 1`)
> and whose `sp ≥ 40,000 s`, the record is either a FILL (y = sp) or a DATE-SLIP (the actual
> off-block clock time under the schedule's date: y = taxi + 86,400), almost never ordinary; the
> date-slip share rises with `sp`. Predicting the two-class expectation beats the incumbent
> mixture on those rows.

Evidence before the test (2025, LIRF unmatched, `dayoff = 1`): above `sp` 56,000 s, 10 date-slips vs
5 fills; `y − 86,400` on date-slips p10/50/90 = 771 / 1,020 / 1,679 s.

## Arms (from `data/cache_stand/rome_fill_preds.parquet` — no refit)

Segment **G** = LIRF unmatched, `dayoff = 1`, `sp ≥ 40,000`. Two sub-bands: G1 = [40,000, 56,000),
G2 = [56,000, ∞). For each fold's TRAINING months only (LOMO: the other eleven; fold A: the ten
excluding 1 and 7), on LIRF unmatched training rows in each sub-band:
`q = (n_dateslip + 1) / (n_band + 2)` (Laplace), `m = median(y − 86,400)` over ALL training
date-slip rows (any band; fallback 1,000 s if none), date-slip := `86,400 ≤ y < 90,000`.

Inside G: `pred = q · (86,400 + m) + (1 − q) · sp`. Outside G: the base arm's prediction, unchanged.

| arm | outside G | inside G |
|---|---|---|
| `S1` | S1 | S1 (control) |
| **`D2`** | S1 | the two-class expectation — **the clean test of H-DS** |
| `R2` | R | the two-class expectation — the combined candidate |

## Thresholds, locked (the five of arm R, each on the whole Rome unmatched set)

For `D2` vs `S1`, and separately for `R2` vs `S1`:
1. month-block bootstrap (12 months, 2,000 draws, seed 0) interval on LOMO SSE(S1) − SSE(arm)
   excludes zero; 2. LOMO reduction ≥ 15% (**D2: ≥ 3%** — it touches ~30 rows a year, registered now);
3. fold-A reduction ≥ 15% (**D2: ≥ 3%**); 4. no seed sd exists for D2 (no fitting) — clause 4 is
   evaluated for R2 only, on R's seed sd; 5. ≥ 8 of 12 months not worse (D2: months with no G rows
   count as not worse).
NOT WORKING iff (1) fails. Reported: rows in G per fold, SSE inside G, q per band per fold.

## Limitation named now

~30 training rows per fold in G; the whole effect rides on a handful of rows. A pass is evidence
about 2025's mix; the 2026 file holds 5 rows in G2 — the board outcome is a bet on those.

---

# RESULT DS · 2026-09-10 10:07 local

`scripts/rome_dateslip.py` → `reports/rome_dateslip.json`, `data/cache_stand/rome_dateslip_preds.parquet`.
Segment G: 30 LOMO rows, 12 fold-A rows. Fold-A shares: q(G1) 0.083, q(G2) 0.600, m 1,143 s.

| arm | LOMO red | fold-A red | months not worse | month-block interval (LOMO SSE) | registered verdict |
|---|---|---|---|---|---|
| D2 = S1 + DS | 5.3% | 17.1% | 9/12 | [−1,326, +4,144] M | **NOT WORKING** (clause 1) |
| R2 = R + DS | 15.2% | 24.1% | 10/12 | **[+640, +8,638] M** | **all clauses pass** (clause 4 on R's seed sd: gain 352 s vs 2×sd 95) |

Fold-A JULY Rome RMSE: S1 3,925.6 · R 3,538.8 · **R2 3,459.5** · public anchor 3,437.9.
Projected board (fold A × 383/397 / 344,841): R2 ≈ 4,154 · R ≈ 4,657 · D2 ≈ 2,941 MSE.

**Post-hoc robustness — NOT registered, reported because the boundaries were read off the same 2025
labels the LOMO scores.** R2 across segment boundaries (G_LO, split):

| (30k, 50k) | (35k, 56k) | (40k, 56k) reg. | (40k, 50k) | (40k, 60k) | (45k, 60k) | (50k, 65k) | (40k, one band) |
|---|---|---|---|---|---|---|---|
| +311 M | +704 | +597 | +195 | **−426** | **−4,447** | **−5,320** | **−1,094** |

(lower bound of the month-block interval, 1,000 draws). **The pass holds in a neighbourhood of the
registered boundaries and fails beyond a 60k split** — the date-slip component rides on 2–3 rows in
sp 56–60k. And on fold A, R alone beats R2 (4,484 M vs 4,664 M SSE).

**Honest classification.** R2 clears its registered bar, and I do not overturn a registered verdict
after the fact. But under the hypothesis-closure rule the DATE-SLIP COMPONENT is **not robust**: its
contribution depends on where a line was drawn in-sample. The classifier (R) carries the robust part
of the Rome gain (fold A −27.1% on its own); its month-to-month instability is what failed R's bar.
Any ship decision on Rome is a bet on a handful of 2026 rows, and is the owner's.

## BOARD · 2026-09-10 13:14 — shipped as v7 (owner's decision): −630 MSE against ≈ −4,150 projected (≈ 0.15×)
The sign transferred; the magnitude did not. See reports/MSE_LEDGER.md, the 13:13 block.
