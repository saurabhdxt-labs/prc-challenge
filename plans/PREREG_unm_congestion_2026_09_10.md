# Pre-registration — arm E3′: a congestion-aware unmatched body (non-Rome)

**Written 2026-09-10 15:25 local, BEFORE the harness exists and before any LOMO number of it exists.**
Tag `E3C`. Owner decision 2026-09-10: "congestion fix first". Source: `reports/PATH_TO_245_SYNTHESIS_2026_09_10.md`
lane 1, from `reports/PATH_245_PASS_FABLE_2026_09_10.md` §3 and `reports/PATH_245_PASS_OPUS_2026_09_10.md`.

**Disclosure of prior looks (so this is read honestly).** Two exploratory looks exist and chose the idea,
not these thresholds: (a) Fable-2 on fold A (months 1, 7; ex-LIRF): a hand gate `S1 + 0.75·(m_h − S1)`
earned +509 MSE [210, 896], swept over 3 thresholds × 3 weights; reproduced in the main session;
(b) Opus: 12-month LOMO **bias** of S1 by hourly `MVT − AOBT_3` median bin (+404 / +1,947 / +3,140 s at
1.5–1.8k / 1.8–2.4k / > 2.4k; positive in 19/20 airport-months). Neither fitted the arm below; neither
measured its gain. The registered test is the fitted arm, all twelve months, clauses fixed here.

## Hypothesis

**H-E3C:** the shipped unmatched body (S1's `NonFillRegressor`, `build_submission.py:66–67`: `sp, dayoff, hr`
+ five target encodings) is congestion-blind; adding an airport-hour congestion witness taken from the
SAME file's matched departures reduces non-Rome unmatched error out of month, with the gain concentrated
in congested hours and no material loss in calm ones.

## The witness (serve-time; never reads BLOCK or a label)

For every airport × UTC clock hour of `MVT_TIME_UTC_mvt` (floor to the hour), over the MATCHED admissible
departures of that airport-hour (`AOBT_3_flt` present):
- `hprox_med` = median of `MVT − AOBT_3` in seconds; `hprox_n` = their count.
- `hprox_med` is NaN where `hprox_n < 5` (the regressor's native missing handling).
Computed from the same file the row is scored in (2025 training months for LOMO; the 2026 ranking file at
serve time). The unmatched row itself never contributes (it has no `AOBT_3`).

## Arms (12-month LOMO over 2025, the stratum harness's folds)

- **S1** — the incumbent, exactly `stratum_fold.stratum_arms` (hybrid, seeds 0, 1, 2). **Guard:** the new
  harness's S1 must reproduce `data/cache_stand/stratum_fold_v7_preds.parquet` (`fold == "lomo"`, column
  `S1`) on every row to ≤ 1e-6 s, or nothing is scored.
- **S1C** — identical `p` and `nf_cells` (taken from the same `fit_unmatched` parts); the body
  regressor is S1's `NonFillRegressor` with `NF_NUMERIC + ["hprox_med", "hprox_n"]` and nothing else
  changed (same params, same winsorisation at 3,000 s, same encodings, same seeds, same non-fill training
  rows); `nf = nf_hybrid(nf_cells, mean_seeds(nf_fit_C))`; `pred = mixture(p, sp, nf)`.
- Scored on **non-LIRF** unmatched rows (LIRF keeps R2 / E1; its S1C numbers are reported, not decisional).

## Clauses, locked (on the LOMO predictions of all non-LIRF unmatched 2025 rows)

Let `g = SE(S1) − SE(S1C)` per row (raw labels, never winsorised).
1. **C1** — date-block bootstrap (blocks = calendar dates, 2,000 draws, seed 0) of Σg: 95% interval
   lower bound > 0.
2. **C2** — **2026-priced gain ≥ +500 board MSE.** Pricing: bins of `hprox_med` — NaN, < 900, 900–1,100,
   1,100–1,300, 1,300–1,500, 1,500–2,000, 2,000–2,400, ≥ 2,400. Per-row mean `g` per bin (2025 LOMO) ×
   the count of 2026 scored non-LIRF unmatched rows in that bin ÷ 344,841. A bin with < 30 2025 rows
   takes the per-row gain of the next LOWER bin (no upward extrapolation).
3. **C3** — Σg > 0 in ≥ 8 of 12 months.
4. **C4** — Σg > 0 at ≥ 6 of the 9 non-LIRF airports.
5. **C5 (calm control)** — Σg over rows with `hprox_med < 1,100` ≥ −10% of the pooled Σg.
6. **C6 (not the monsters)** — Σg over rows with `y ≤ 10,800` > 0.
7. **C7 (seeds)** — pooled gain > 2 × sd of the three single-seed gains (S1C_seed s vs S1_seed s).

**Verdict:** WORKING iff all seven pass; NOT WORKING iff C1 fails; else INCONCLUSIVE. Never revised.
**Ship rule:** only if WORKING — v_next = the best board version with the non-LIRF unmatched rows replaced
by S1C (trained on all twelve 2025 months, 2026 witness), behind a guard that the shipped S1 path
rebuilds the base on every non-LIRF unmatched row exactly (the `rome_ship.py` pattern).

## Predicted shapes

- **TRUE:** Σg > 0 driven by the ≥ 1,300 bins; calm bins ≈ 0; most months and airports positive; 2026
  pricing ≈ +2.5k to +7k (2026 has ≈ 3× 2025's hot-hour exposure, EHAM 3–7 Jan dominant).
- **FALSE:** Σg ≈ 0 or negative, or positive only in one or two months / airports (a date artefact), or
  bought by losses in calm hours.

## Limits (named now)

- The fit target stays winsorised at 3,000 s (as S1); a congestion effect above that is capped — this
  arm tests the witness, not the winsorisation.
- 2026 EHAM 3–7 Jan reaches hourly medians beyond EHAM's 2025 range; trees do not extrapolate, so the
  2026 prediction there is the regressor's highest-congestion leaf — conservative by construction. The
  transfer of that regime is the main risk (downside if storm-day unmatched rows are diversions or
  mis-recorded cancellations); ADS-B for those days is the independent check, not part of this arm.
- Fold-A (months 1, 7) was seen by the exploratory look; it is inside the twelve LOMO months and is
  reported separately as well.

## AMENDMENT E3C.1 · 15:35 — implementation judgement calls, BEFORE the real LOMO run (no real number exists)

Harness `scripts/unm_congestion.py` (tests `tests/test_unm_congestion.py`, 22 passed, 28 mutations rehearsed
RED; shipped `build_submission.py` / `stratum_fold.py` byte-identical to HEAD). Clarifications, none moving a
threshold:
1. C2's "next LOWER bin" has no lower bin for the NaN bin and `< 900`: a thin one there is priced at **0**
   and flagged (no extrapolation in any direction).
2. An airport-hour with no matched departure has `hprox_n = 0`, `hprox_med = NaN`.
3. C1's blocks are the UTC calendar dates of `MVT_TIME`; the witness hour is the UTC floor of `MVT_TIME`.
4. The 2026 witness is computed over every matched DEP row of `ranking.parquet` (not only scored rows).
5. C5 is implemented literally (bound = −10% × pooled Σg); if pooled Σg < 0, C1 decides anyway.
6. **Train/serve skew, named:** the 2025 witness is built from matched rows AFTER `admissible()` (which needs
   labels); the 2026 witness from all matched DEP rows of the serve file (labels are null there). A median
   over near-identical row sets; not expected to matter; not corrected, because correcting it would need labels
   at serve time.

## RESULT E3C · 2026-09-10 15:43 — **WORKING** (all seven clauses pass)

Source: `reports/unm_congestion.json` (`unm_congestion.py score`, exit 0), predictions
`data/cache_stand/unm_congestion_lomo.parquet` (`lomo`, exit 0, 23 s, peak 3.16 GB). **Reproduction guard:
S1 reproduced the stored v7 LOMO S1 on all 22,219 rows, max |diff| 0.0 s.** Non-LIRF LOMO rows: 20,731.

| clause | value | bar | pass |
|---|---|---|---|
| C1 date-block bootstrap (365 dates) | Σg +1.712e9 s², 95% [+4.43e8, +3.45e9] | lower > 0 | ✓ |
| C2 2026-priced | **+3,960 board MSE** | ≥ 500 | ✓ |
| C3 months | 12 of 12 positive | ≥ 8 | ✓ |
| C4 airports | 7 of 9 (EDDF −3.9e6, LSZH −0.7e6) | ≥ 6 | ✓ |
| C5 calm (< 1,100, n 13,729) | +1.57e7 | ≥ −1.71e8 | ✓ |
| C6 ex-monster | +1.73e9 | > 0 | ✓ |
| C7 seeds | pooled +1.71e9 vs 2 × sd 5.3e7 | > | ✓ |

RMSE pooled S1 1,141.50 → S1C 1,104.75 (+36.76 s); ex-monster 649.31 → 581.53 (+67.78 s). Fold-A months
(seen by the exploratory look) +1.10e8; the other ten months +1.60e9. LIRF (report-only): +3.45 s.

**Reported, not decisional — concentration, stated plainly.** LTFM February 2025 (a six-date disruption,
173 rows at `hprox_med ≥ 2,000`, y mean 5,182 s, S1 1,268, S1C 2,272) carries 77% of pooled Σg; LTFM 80%.
Without LTFM-Feb, Σg is still **+4.01e8 on 20,367 rows** (RMSE 1,070.08 → 1,060.83), and every other
hot-hour event moves the right way (LFPG Nov, EGLL Mar, LFPG Jan, EHAM Jan, LTFM Dec / Oct). The 2026
price is 79% the `≥ 2,400` bin (185 rows: **EHAM Jan 149**, LFPG Jan 30), priced from 98 2025 rows of
which 65 are LTFM. Post-hoc sensitivities of C2: top-bin gain from non-LTFM rows **+2,299**; from EHAM's
own 10 rows **+1,664**; every bin re-priced without LTFM-Feb **+1,793**. **Honest board expectation
≈ +1.7k to +4.0k.** The verdict is the registered one and is not revised by these.

**Residual, named:** even S1C under-predicts the hottest hours by ~half (LTFM-Feb 2,272 vs 5,182) —
the fit target is winsorised at 3,000 s (a registered limit). A follow-up that lifts that cap in hot
hours is a separate hypothesis and needs its own registration.

## BUILD · 2026-09-10 16:05 — v9 = v7 + S1C on non-LIRF unmatched rows (owner: base v7, for clean attribution)

`scripts/unm_congestion_ship.py --base submissions/merry-quicksand_v7.parquet --version 9` (exit 0, 6.3 s;
tests `tests/test_unm_congestion_ship.py` 11 passed, 15 mutations rehearsed RED). Rebuild guard: the shipped S1
path reproduced v7 on all 4,907 scored non-LIRF unmatched rows (LIRF's 383 not compared, not changed). 2026
witness from 339,551 matched serve rows; 2.8% of scored unmatched rows NaN. **4,828 values differ**, all
non-LIRF unmatched (independent diff); RMS shift 237 s; Σ shift² / 344,841 = **800** (= the expected gain if
S1C is unbiased in 2026; the registered +3,960 assumes 2026's hot rows sit as far above S1C as 2025's did).

| 2026 bin | rows | witness mean | v7 mean | v9 mean |
|---|---:|---:|---:|---:|
| < 900 | 1,692 | 739 | 858 | 837 |
| 1,300–1,500 | 443 | 1,445 | 1,207 | 1,246 |
| 1,500–2,000 | 279 | 1,666 | 1,235 | 1,621 |
| 2,000–2,400 | 47 | 2,148 | 1,143 | 2,076 |
| ≥ 2,400 | 185 | 3,321 | 1,018 | 1,691 |

EHAM rows at witness ≥ 2,000 (the January storm): 161 rows, witness mean 3,262 s, v7 896 → v9 1,446 (max
2,962) — S1C moves them up but stays well below the witness (winsorised target, tree support).
`submissions/merry-quicksand_v9.parquet` sha256 `95b35326…573264e2`. **Board verdict rule for v9 (fixed now,
before upload):** the reading is Δ vs v7; the expectation band is +800 (unbiased S1C) to +3,960 (registered);
a Δ ≥ 0 (no gain) is recorded as NOT TRANSFERRED and v9 is not carried forward.

## BOARD · 2026-09-10 16:08 — v9 = **282.6790** (v7 285.8013): **−1,775.0 board MSE — TRANSFERRED**

Uploaded 16:07:21 (HTTP 200) after an on-disk re-verification (sha `95b35326…`, template, id order == v7,
339,551 matched + 383 LIRF unmatched rows byte-identical to v7, 4,828 non-LIRF unmatched rows changed).
Scorer `result.json`: Succeeded, used_pairs 344,841, score 282.679. Rank 15 of 106 (from 21).
Inside the pre-written band (+800 unbiased-S1C floor … +3,960 registered): **0.45× the registered price,
≈ 1.0× the post-hoc sensitivities that do not lean on LTFM-Feb (+1,664 … +2,299).** The lesson for pricing:
a bin priced from one dominant 2025 event transferred at the conservative (event-excluded) price.
