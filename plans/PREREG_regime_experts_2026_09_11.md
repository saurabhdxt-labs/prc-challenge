# Pre-registration — arm REG: a regime-gated multi-model matched lane (SCREEN tier)

**Written 2026-09-11 08:55:49 (from `date`), session prc-challenge-c4, BEFORE the harness exists and before any number
of this arm exists.** Tag `REG`. Owner, 2026-09-11: "think about the multi model architecture", "we have to get to 245",
"yes go ahead".

## Why

Arm F's matched error (board MSE 48,774) by clock regime, from its stored fold-A record (delta = BLOCK − AOBT_3):

| regime | rows | board MSE | share | F's RMSE | F's mean error |
|---|---|---|---|---|---|
| clocks agree (|delta| ≤ 600, not fill) | 84.3% | 24,418 | 50% | 172 | −21 s |
| fill (|y − sp| ≤ 60) | 8.9% | 7,148 | 15% | 286 | +66 s |
| early off-block (delta < −600, not fill) | 3.8% | 14,064 | 29% | 616 | **+343 s** |
| late off-block (delta > +600, not fill) | 3.0% | 3,144 | 6% | 327 | −211 s |

One global model averages four data-generating situations; on early-off-block rows it misses by +343 s on average.
`scripts/cond_experts.py` (never run on full data) built the two-regime version (fill / body) correctly; this arm
generalises it to four regimes.

## Hypothesis

**H-REG:** a calibrated four-class regime gate with one expert per regime, combined as the probability-weighted mean,
lowers fold-A matched error against a same-settings single model — and the gain comes from the tail regimes without
the clock-agree rows paying for it.

## Design (fixed now)

- **Rows, folds, features:** fold A exactly as arm F (`lgbm_fold.fold_masks`: holdout months 1 and 7, early stopping on
  3 and 9 carved from training); arm F's 83-column design (`FEATS_QUEUE_ORDER`: stand-cache + queue + order blocks) with
  the 24 target encodings replaced by `prc.encoding`'s SEPARATED stages (BC-1-clean: stopping on fit-months encodings,
  refit on cross-fitted ones — `enc_ab.encoding_columns("separated")`, the path `cond_experts.py` uses).
- **Regime label** (training rows only; fill takes precedence): fill `|y − sp| ≤ 60`; early `delta < −600`; late
  `delta > +600`; agree otherwise.
- **Gate:** LightGBM multiclass (4), `L.P` with learning rate 0.05, early stopping on the ES rows (multi_logloss),
  refit on all training rows at `n_refit`; holdout probabilities from the refit. No calibration map (the holdout is
  never used to fit; the reliability table is reported).
- **Experts** (each a delta model with `L.P` at lr 0.05, early-stopped on its own ES rows, refit on its own training
  rows, seed 0): `μ_agree` on agree rows, `μ_early` on early rows, `μ_late` on late rows; taxi = `max(proxy − δ̂, 1)`.
  `μ_fill = sp + shrunk per-airport mean(y − sp | fill)` (`cond_experts.fill_correction`). **No expert is trained on
  all rows** (RESULT 8's double counting).
- **Arms on the holdout:** `BASE` = one delta model on ALL training rows, same encodings / params / seed (the paired
  control — any gain against it is the architecture, not the settings); `REG = max(Σ_k p_k μ_k, 1)`.
- One seed: **screening**, not confirmation.

## Clauses, locked (REG vs BASE, 339,015 holdout rows; weighted fold MSE = 0.9846596 × ΔSSE / 339,015)

1. **C1** — paired calendar-date block bootstrap (take-off dates, Jan / Jul resampled separately, 2,000 draws, seed 0):
   95% lower bound > 0.
2. **C2** — net ≥ **+1,000** weighted fold MSE (the ledger's compute bar).
3. **C3 (the body does not pay)** — the clock-agree rows' net ≥ **−300** weighted fold MSE (RESULT 8 lost 1,719 there).
4. **C4** — ≥ 7 of 10 airports improve.
**Verdict:** SHORTLIST iff all four; otherwise NOT SHORTLISTED (one seed, one fold — not a refutation). A shortlisted arm
goes to a three-seed confirmation on arm F's settings inside the pipeline under its own registration.

**Reported, not decisional:** gate one-vs-rest AUC per regime (early especially) and reliability deciles; an
ORACLE-GATE arm (true regime one-hot × the same experts — the experts' ceiling, never a result); per-regime, per-airport
and per-month cuts; REG and BASE against arm F's stored record (F 222.5632; F differs in lr, seeds and the leaky
encoder, so this comparison is context only); walls and peak memory.

**Guards:** the loader's holdout rows equal arm F's record row for row (y, proxy); regime counts on the holdout equal
the table above (agree 285,942 / fill 30,167 / early 12,754 / late 10,152); a `--smoke` run on three months passes
before the full run.

## Predicted shapes (written now)

- **TRUE:** early-vs-rest AUC ≥ 0.80; REG − BASE ≥ +1,000, at least half of it on early-off-block rows; the agree rows
  within ±300; the oracle-gate arm far above REG (the gate, not the experts, is the limit).
- **FALSE:** early-vs-rest AUC < 0.70 and REG within ±300 of BASE (the regimes are not predictable from today's
  features — the tail needs new information, e.g. ADS-B seeing the aircraft leave the stand); or REG gains on the tail
  and loses as much on the agree rows (the mixture weights leak).

## Limits (named now)

One seed, one fold; January and July are inspected development months. lr 0.05 (speed) not F's 0.01. The gate is
uncalibrated. The separated encoder makes BASE differ from F; the comparison that decides is REG vs BASE on identical
rows. Heavy: one run, ≈ 1 h, ≈ 5 GB peak — quality gate, AC power in its own step, the only ≥ 2 GB job.

## RESULT REG · 2026-09-11 09:57:27 (from `date`) — **NOT SHORTLISTED** (C2 fails: +633 < +1,000; C1, C3, C4 pass)

Source `reports/regime_experts_full.json`, predictions `data/cache_stand/regime_experts_preds.parquet` (convention
`taxi_time`, stamped); log `reports/regime_experts_full.console.log` (exit 0, 1,558 s, peak RSS 5.95 GB). Guards passed:
holdout rows equal arm F's record row for row; holdout regimes equal the registered counts; F 222.5632 reproduced.

| clause | value | bar | pass |
|---|---|---|---|
| C1 date-block (62 dates) | +632.8 [+251.3, +1,006.4] | lower > 0 | ✓ |
| C2 net | **+632.8** | ≥ +1,000 | **✗** |
| C3 clock-agree rows | +264 | ≥ −300 | ✓ |
| C4 airports | 7 of 10 (EHAM −68, EDDF −32, EDDM −21) | ≥ 7 | ✓ |

RMSE: BASE 226.40 → REG 224.98 (F, context: 222.56 — BASE is lr 0.05, one seed, separated encoder; REG vs F −1,066, BASE vs
F −1,698). By regime: agree +264, fill +468, **early −221**, late +123. By month Jan +258, Jul +374. LIRF +495 carries most.

**Reported, not decisional:**
- Gate one-vs-rest AUC: agree 0.874, fill 0.894, **early 0.961**, late 0.976; mean p early 0.038 (true share 3.76%).
  Early reliability by decile: the top decile's mean p 0.327 against an observed rate 0.312 — calibrated, and never sure:
  **no decile of the gate isolates early-off-block rows** (precision ≤ 31%).
- **ORACLE gate (true regime one-hot × the same experts): +22,959** vs BASE — early +7,751, fill +7,263, agree +5,605, late
  +2,340. The experts carry more than the whole gap to 245 (19,217); the gate is the entire limit.

**Against the pre-written shapes — neither appeared as written:** TRUE's gate condition held (early AUC 0.961 ≥ 0.80) and
the agree rows held (+264), but REG − BASE is +633 (< +1,000) and the early rows LOST (−221, not "at least half the
gain"); FALSE's first branch (AUC < 0.70) is refuted and REG is not within ±300 of BASE. What happened: a high AUC on a rare
class (3.8%) is ranking, not isolation — calibrated probabilities that never exceed ≈ 0.33 keep the mixture near the
agree expert, so the regime's value stays locked. The oracle says the value is there; the gate's inputs do not carry it.

**What this closes and opens.** Closed: "a four-regime mixture on TODAY's features is worth ≥ 1,000 at screen level" —
not under this registration. Open, and now priced: regime identification is worth up to ≈ 23,000 (oracle) — the lever
is INFORMATION that tells the airport's off-block from NM's (ADS-B seeing the aircraft move; airport procedure signals),
not a different learner. The fill regime (+468 of an oracle +7,263) and LIRF (+495) are where today's features already help.

*Correction to RESULT REG's reported (non-decisional) description · 2026-09-11 10:03:03 (from `date`), after the GID diagnostic; the verdict
NOT SHORTLISTED is unchanged:* "calibrated probabilities that never exceed ≈ 0.33 … no decile isolates early rows
(precision ≤ 31%)" is WRONG — a decile holds 10% of rows and the class is 3.8%, so no decile can show more than ≈ 38%.
The gate gives p_early ≥ 0.5 on 7,824 rows at 74.8% precision, covering 46% of early rows. The correct mechanism
(`reports/GATE_INFORMATION_2026_09_11.md` Part 2): where the gate is confident the single model already prices the regime
(BASE error +112 s; oracle +696 left); the value sits in early rows the gate cannot see (BASE error +532 s; oracle
+7,054). "The gate's inputs are the limit" stands; "the gate never commits" does not.
