# Pre-registration — arm SEL: a learned model selector over the stored experts (SCREEN, zero compute)

**Written 2026-09-11 10:44:45 (from `date`), session prc-challenge-c4, BEFORE the harness exists and before any number of
this arm exists.** Tag `SEL`. Owner, 2026-09-11: "let's work on the multi model architecture — using a classifier to decide
which model / which tree would work better; do we need different models per location, maybe every region behaves
differently, but that might already be included".

## Why

Several models already predict the SAME 339,015 fold-A holdout rows (Jan + Jul 2025), each out of sample; fixed 50/50
blends of pairs have gained (C_delta + F +533; + CAP +304) and a regime mixture gained +633 against its own control. A
selector learns, per row, which model to trust. Per-airport LightGBM models blended at a FIXED 0.5 were NOT WORKING
(RESULT 7: +0.44 s, interval spans zero); a LEARNED per-airport weight was never tested.

## Experts (stored, row-aligned with arm F; each recovered to taxi seconds, floored at 1, guard = its recorded RMSE ± 5e-4)

F (queue_order `treatment`, 222.5632) · CAP (`cap`, 221.9850) · CB (C_delta, 225.7836) · Y (ytarget `Y`, 260.6511) ·
W (queue_weather `treatment`, 223.533) · D (queue_day `treatment`, 224.3119) · PA (v4 `A3`, 225.388, the per-airport blend) ·
REG (the regime mixture, taxi_time, 224.981) and its experts `mu_agree / mu_early / mu_late / mu_fill`.
Meta-features for the selectors: airport, hour of take-off, proxy, sp, REG's gate probabilities `p_*`.

## Selectors — CROSS-FITTED BY MONTH (fit on January's rows, predict July's, and the reverse; no row ever scored by a
selector that saw it)

- **S0** CAP alone (the best single expert; the control).
- **S1** global linear stack: non-negative weights summing to 1, least squares on the other month.
- **S2** per-airport linear stack: the same, one weight vector per airport (the "every region behaves differently" test).
- **S3** LightGBM stacker: y from all expert predictions + meta-features (num_leaves 31, lr 0.05, early stopping on a
  day-split of the fitting month).
- **S4** classifier selector (the owner's framing): LightGBM multiclass predicting which expert is closest to y; prediction
  = Σ p_k · expert_k.
- Reported, never a result: the per-row ORACLE (the closest expert on every row) — the diversity ceiling.

## Decision (locked; weighted fold MSE = 0.9846596 × ΔSSE / 339,015; paired date-block interval, 62 dates, 2,000 draws)

- **BUILD** the best selector into the pipeline (as a Stack plug-in, then a confirmation on inner out-of-fold predictions of
  all twelve months) iff its gain over S0 ≥ **+500** with the date-block lower bound > 0.
- **Per-location answer:** per-airport selection matters iff S2 − S1 ≥ **+200** with lower bound > 0; otherwise one
  global selector (airport already an input) is the design.
- Otherwise: NOT WORTH BUILDING at this expert set (record why).

## Predicted shapes (written now)

- **TRUE:** the oracle is several thousand above S0 (the experts disagree a lot); S3 or S4 recovers +500–2,000 over CAP,
  carried by LIRF and the |δ| > 600 tail (where CB / Y / REG differ from F); S2 ≈ S1 (airport is already an input).
- **FALSE:** S1–S4 all within ±300 of CAP — the stored experts are too similar (all LightGBM-family on one design) for a
  selector to exploit; a useful selector needs genuinely different experts (other features / targets / learners).

## Limits (named now)

Cross-month fitting pays any January ↔ July seasonal change (conservative); a ship fits on inner out-of-fold predictions of
all twelve months, which this screen does not have. REG / BASE use a different encoder and learning rate. One fold.

## RESULT SEL · 2026-09-11 10:49:06 (from `date`) — **NOT WORTH BUILDING at this expert set; per-airport selection does not matter**

Source `reports/model_select.json` (`scripts/model_select.py`, 62 s, 0.99 GB; tests `tests/test_model_select.py` 5, 5 mutants
killed). All expert guards reproduced their records (F 222.5632, Y 260.6511, W 223.533, D 224.3119, PA 225.388); every record
row-aligned with arm F.

| selector (cross-fitted Jan ↔ Jul) | vs S0 = CAP | date-block 95% | LIRF | |δ|>600 tail | clock-agree |
|---|---|---|---|---|---|
| S1 global linear stack | +224.8 | [−275.7, +650.6] | −179 | −562 | +786 |
| **S2 per-airport linear stack** | **+253.2** | [−15.9, +529.0] | +226 | −384 | +637 |
| S3 LightGBM stacker | **−24,475** | [−65,035, −3,762] | — | — | — |
| S4 classifier selector | −1,987 | [−2,606, −1,464] | — | — | — |
| S2 − S1 (per-airport vs global) | +28.4 | [−289.8, +420.8] | | | |

Decision (locked): best S2 +253 < +500 and its lower bound < 0 → **not BUILD**; S2 − S1 < +200 → **one global selector** is the
design (airport identity is already an input). Oracle (closest of 12 experts per row, never a result): +35,633 — mostly
chance agreement of wild regime-conditioned experts (mu_fill RMSE 2,459), not usable diversity.

**Against the pre-written shapes:** the FALSE shape's core appeared — S1 / S2 within ±300 of CAP (the stored experts are all
LightGBM-family on one design). The learned selectors (S3, S4) went further than FALSE predicted: they LOST. S4 learns which
expert happened to be closest on each row — mostly chance; S3 predicted y from scratch on half a month of rows instead of
correcting the best expert (a design weakness named now; a residual-target stacker would be a new registration, not a
rescue of this one). The linear stacks gain on clock-agree rows and give some back in the tail.

**What it says for the multi-model architecture:** selection pays only when the experts carry DIFFERENT information; these do
not. The architecture stays (gate / experts / stack plug-ins in the pipeline), but its next experts must differ in inputs
or targets, not in learner settings.
