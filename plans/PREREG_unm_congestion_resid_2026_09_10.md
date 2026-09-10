# Pre-registration — arm E3R: the congestion body learns the offset to the witness, not the level

**Written 2026-09-10 16:40 local, BEFORE the harness exists and before any number of this arm exists.**
Tag `E3R`. Owner decision 2026-09-10 ("yes start it"). Follows RESULT E3C (`plans/PREREG_unm_congestion_2026_09_10.md`,
WORKING; v9 = 282.6790, −1,775 board MSE; v10 = 281.5182 is the best entry).

## Why

S1C (shipped in v9/v10) is S1's `NonFillRegressor` with the witness (`hprox_med`, `hprox_n`) added, but its
fit target is still **y winsorised at ±3,000 s** (`build_submission.WINSOR_S`). In the hottest hours the label
sits far above the cap: 416 non-LIRF LOMO rows have y > 3,000, **201 of them in hours with `hprox_med ≥ 2,000`**;
LTFM Feb 2025 hot rows average y 5,182 vs S1C 2,272; EHAM's 2026 storm rows move only 896 → 1,446 s against
a witness of 3,262 s. A level model whose target is capped at 3,000 cannot follow a witness above it, and a
tree does not extrapolate. The structural fix is to model the **offset** from the witness, so the level comes
from the witness itself and the cap applies to the offset.

## Hypothesis

**H-E3R:** for non-fill unmatched rows with a witness, predicting `y − hprox_med` (winsorised at ±3,000 s)
and adding `hprox_med` back beats S1C out of month on non-LIRF rows, with the gain in the hot bins and no
material loss in calm ones.

## Arms (12-month LOMO over 2025, the E3C harness's folds, non-LIRF rows scored)

- **S1C** — control: exactly `unm_congestion.congestion_arms`' S1C (what v9/v10 ship). **Guard:** the harness's
  S1C must reproduce `data/cache_stand/unm_congestion_lomo.parquet`'s `S1C` column on every row to ≤ 1e-6 s,
  or nothing is scored.
- **S1R** — identical `p`, `nf_cells`, seeds, params, encodings and non-fill training rows as S1C. The body:
  on training rows WITH a witness (`hprox_med` not NaN), target `r = clip(y − hprox_med, −3,000, +3,000)`;
  features = S1C's features exactly (so `hprox_med`, `hprox_n` remain inputs); prediction
  `nf_fit_R = hprox_med + r_hat`. Rows WITHOUT a witness (train and test) use S1C's `nf_fit` unchanged.
  Target encodings are computed on `r` (the fitted target), smoothed as S1C's. `nf = nf_hybrid(nf_cells,
  mean_seeds(nf_fit_R))`; `pred = mixture(p, sp, nf)`.

## Clauses, locked (non-LIRF LOMO rows; `g = SE(S1C) − SE(S1R)`, raw labels)

1. **C1** — date-block bootstrap (calendar dates, 2,000 draws, seed 0) of Σg: 95% lower bound > 0.
2. **C2 — the deciding price is event-excluded** (the lesson of v9): drop the ONE airport-month with the largest
   |Σg|, recompute the per-bin per-row gains (E3C's bins and < 30-rows-borrows-next-lower rule), price on the
   2026 scored non-LIRF unmatched bin counts ÷ 344,841: **≥ +300 board MSE**. The full (non-excluded) price is
   reported, not decisional.
3. **C3** — Σg > 0 in ≥ 8 of 12 months.
4. **C4** — Σg > 0 at ≥ 6 of 9 non-LIRF airports.
5. **C5 (calm control)** — Σg over `hprox_med < 1,100` ≥ −10% of pooled Σg.
6. **C6 (not the monsters)** — Σg over `y ≤ 10,800` > 0.
7. **C7 (seeds)** — pooled gain > 2 × sd of the three single-seed gains.

**Verdict:** WORKING iff all seven pass; NOT WORKING iff C1 fails; else INCONCLUSIVE. Never revised.
**Ship rule:** only if WORKING — v_next = v10 with the non-LIRF unmatched rows replaced by S1R (trained on all
twelve 2025 months, 2026 witness), behind a guard that the S1C path rebuilds v10's non-LIRF unmatched rows
exactly. The board reading is Δ vs v10; the expectation reported to the owner is the **event-excluded**
price, with the full price as the optimistic end — never the other way round.

## Predicted shapes

- **TRUE:** Σg > 0 carried by the ≥ 2,000 bins, several airport-events positive (not only LTFM Feb), calm
  bins ≈ 0; 2026 EHAM storm predictions move from ≈ 1,450 toward the witness (≈ 3,000+).
- **FALSE:** calm-hour losses (the witness is a noisier level than S1C's encodings where congestion is low) or
  hot-bin gains confined to one event; or the unmatched rows in hot hours sit well below the matched
  witness (then adding `hprox_med` back overshoots).

## Limits (named now)

- The offset is still capped at ±3,000 s; a storm hour where unmatched taxi exceeds witness + 3,000 is not
  followed. Fills and the LIRF lane are untouched.
- 2026 EHAM (witness up to ~3,300) is within 2025's pooled witness range (LTFM Feb ≥ 2,400) but beyond EHAM's
  own 2025 range; the transfer of that regime remains the main risk. ADS-B for 3–7 Jan is the independent check.
- Today's board used 5 of 5 slots (UTC); a shipped v11 is read tomorrow.

## AMENDMENT E3R.1 · 16:55 — implementation judgement calls, BEFORE the real LOMO run (no real number exists)

Harness `scripts/unm_congestion_resid.py` + `scripts/unm_congestion_resid_ship.py` (tests 34 new, 98 with the
neighbours; 40 mutations rehearsed, 40 RED). None moves a threshold:
1. The offset cap is `build_submission.WINSOR_S` (asserted == 3,000); the parent's winsorisation is then the
   identity on `r`.
2. The reproduction guard holds S1C (registered) and also S1 + the per-seed S1C arms to 1e-6 s against
   E3C's parquet, read through its convention stamp.
3. The excluded airport-month is the largest |Σg| over (ADEP, month); ties broken by airport then month.
4. After exclusion the table is re-binned, then E3C's < 30-rows rule and downward borrowing apply to the
   rows that remain. **Pre-read, stated now:** if LTFM-Feb is the dropped event, the ≥ 2,400 bin (98 rows,
   65 LTFM) turns thin and borrows the 2,000–2,400 per-row gain — the registered rule, and it means the
   decisional price for the 185 hottest 2026 rows rests on the 2,000–2,400 gain.
5. The 2026 bin counts are recomputed from `ranking.parquet` and must equal E3C's stored witness bins.
6. The three-seed `nf_fit_R` is the mean of `(hprox_med + r_hat_s)` (equal to `hprox_med + mean r_hat` to
   ≤ 5e-13 s).
7. Ship guard: `rint(S1C with the 2026 witness)` must equal v10 on every non-LIRF unmatched row; rows
   without a witness must not change.

## RESULT E3R · 2026-09-10 16:50 — **NOT WORKING** (C1 fails; C3, C4, C5 fail too)

Source: `reports/unm_congestion_resid.json` (`score`, exit 0), predictions `data/cache_stand/unm_congestion_resid_lomo.parquet`
(`lomo`, exit 0, 23 s, peak 3.26 GB). Reproduction guard: S1C, S1 and the per-seed S1C arms reproduced E3C's
record on all 22,219 rows, max |diff| 0.0 s. Non-LIRF rows 20,731. RMSE S1C 1,104.75 → S1R 1,103.15 (+1.60 s).

| clause | value | bar | pass |
|---|---|---|---|
| C1 date-block bootstrap | Σg +7.32e7, 95% [−1.50e8, +3.52e8] | lower > 0 | ✗ |
| C2 event-excluded 2026 price (dropped EGLL-03, +6.95e7 on 91 rows) | +600 | ≥ 300 | ✓ |
| C3 months | 6 of 12 | ≥ 8 | ✗ |
| C4 airports | 5 of 9 | ≥ 6 | ✗ |
| C5 calm (< 1,100) | −2.18e7 | ≥ −7.3e6 | ✗ |
| C6 ex-monster | +7.32e7 | > 0 | ✓ |
| C7 seeds | +7.32e7 vs 2 × sd 3.46e7 | > | ✓ |

Full (non-decisional) price +957. **Not shipped; no v11 from this arm.** Reported, not decisional: the gain is
entirely the ≥ 2,400 bin (+1.89e8 on 98 rows; y 4,512, witness 2,816, S1C 2,321 → S1R 3,041), while calm
bins lose (< 900: −2.54e7 on 8,843 rows) and the 2,000–2,400 bin loses (−8.52e7; y 3,780 vs witness 2,219 —
the witness under-states the level there, so anchoring to it does not help). The offset formulation trades
a small, real hot-hour gain for broad calm-hour noise. Any gated variant is post-hoc from this reading and
would need its own registration with that disclosed.
