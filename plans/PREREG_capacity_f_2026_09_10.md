# Pre-registration — arm CAP: the capacity sweep's top setting, confirmed on arm F's design

**Written 2026-09-10 19:05 local, BEFORE the harness exists and before any number of this arm exists.** Tag `CAP`.
Owner decision 2026-09-10 ("yes" to the overnight capacity confirmation). Source: `reports/PIPELINE_TO_245_2026_09_10.md`
Stage 3(b); the Amendment 15.2 sweep (`reports/lgbm_fold_sweep.json`, phantom-forecaster-22's arm, committed in
d2280cd) ranked `127,20,0.6` first of 18 at lr 0.05 on a Feb–May subfold ("RANKING ONLY") and the incumbent
`255,40,0.8` 10th; the confirmation tier was never run. **The existing `lgbm_fold.py --confirm` fits the 68-feature
v4 design (`L.FEATS`), not the shipped arm F (83 features)** — so this confirmation is registered on arm F's
design, where a gain would actually ship. This registration does not touch `plans/PREREG_taxiout_2026_09_08.md`.

## Hypothesis

**H-CAP:** on arm F's exact design (FEATS + QUEUE + ORDER, 83 columns, A2 = no per-airport models, seeds 0, 1, 2,
learning rate 0.01, early stopping on months 3 and 9, refit per seed at arm F's n_ref rule), replacing
`num_leaves, min_data_in_leaf, feature_fraction = 255, 40, 0.8` by **`127, 20, 0.6`** lowers fold-A matched RMSE.

## Arms (fold A, 339,015 matched holdout rows)

- **F** — control: arm F's stored record `data/cache_stand/fold_preds_queue_order.parquet` (`treatment`, seed mean;
  `delta_hat_seed{0,1,2}`). **Guard:** it must recover 222.5632 (`max(proxy − treatment, 1)`) or nothing is scored.
- **CAP** — `lgbm_fold.fit_arm` on `load_fold(..., FEATS_QUEUE_ORDER, qcache, ocache)` (arm F's loader exactly) with
  `setting_params(L.P, (127, 20, 0.6))`; everything else as arm F. Rows joined to F on `row`; labels must match.

## Clauses, locked (the ledger's confirmation rules)

1. **C1** — paired row bootstrap (2,000, seed 0) of RMSE(F) − RMSE(CAP) excludes zero in CAP's favour.
2. **C2** — gain > 2 × seed sd (sd of CAP's three single-seed RMSEs; F's reported beside it).
3. **C3** — both holdout months (1 and 7) improve.
4. **C4** — ≥ 7 of 10 airports improve.
5. **C5** — no cut loses materially: fill rows (|y − sp| ≤ 60) and the |delta| > 600 s tail each lose ≤ 1.0 s.

**Verdict:** ESTABLISHED iff all five pass; NOT WORKING iff C1 fails; else INCONCLUSIVE. Never revised.
**Reported:** net weighted fold MSE = 0.98466 × ΔSSE / 339,015; per-airport, per-band, fill / non-fill cuts; best_iter,
n_ref, walls, peak memory; a date-block bootstrap (calendar dates) beside the row bootstrap.
**Ship rule (a separate build, not tonight):** only if ESTABLISHED **and** net ≥ +500 weighted fold MSE — the
submission path (`lgbm_submit.py`) gains a tested setting override, v_next = best board version with matched rows
refitted at `127,20,0.6` (matched-lane transfer has been 0.96–1.17× on four board readings).

## Predicted shapes

- **TRUE:** gain of roughly +0.5 to +2 s (≈ +200 to +900 weighted MSE), most airports positive — the screen's
  subfold advantage (224.93 vs the incumbent's rank-10 value, ranking only) carrying to the full design.
- **FALSE:** gain ≈ 0 or negative: the sweep's subfold (4 training months, lr 0.05, 68 features) does not transfer
  to the full 10-month, lr 0.01, 83-feature fit — smaller leaves may simply need more trees to reach the same place.

## Limits (named now)

- BC-1 (early-stopping encodings fitted on their own rows) is present in BOTH arms exactly as in arm F — same footing,
  unrepaired; `best_iter` is biased identically. One fold (A); months 1 and 7 are inspected development months.
- No forward-time check tonight (the pipeline's confirmation protocol asks for one before a ship ≥ 1,000).

## AMENDMENT CAP.1 · 19:22 — BEFORE the fit has started (no CAP number exists): C1 becomes the day-block interval

An independent review (session 80a473c1, 18:30–19:10) verified that the matched lane's paired ROW bootstrap treats
rows as independent and understates the interval ≈ 4.6× (rows within a day share weather / congestion). **C1 is
therefore decided by the paired calendar-date block bootstrap** (dates of take-off, 2,000 draws, seed 0; lower bound
> 0 in CAP's favour). The row bootstrap is reported beside it, not decisional. Nothing else changes.
*Note to Amendment CAP.1 (19:40, before any CAP number exists; no clause changes):* the "≈ 4.6×" figure is the review's,
not verified here; a queue-block date-block check gave ≈ 1.5×. The amendment stands on the estimator's merit — rows
within a day share weather and congestion — not on the size of that factor.

*Timestamp correction (2026-09-10 20:47:51, from `date`):* the stamps "19:22" on Amendment CAP.1 and "19:40" on its note were typed,
not read from the clock. The fit's own log starts at **19:18:11** (`reports/capacity_f_fit.console.log`), and CAP.1 was
written before that launch, so its true time is ≤ 19:18:11. The ORDER (amendment before any CAP number) is unchanged and
verifiable from the log. All stamps written by this session from now on come from `date`.

## RESULT CAP · 2026-09-10 21:15:57 (from `date`) — **ESTABLISHED** (all five clauses) — **ship rule NOT met** (net +253 < +500)

Source: `reports/capacity_f.json` (`score`, exit 0), fit `reports/capacity_f_fit.json` (best_iter 33,352, n_ref 41,694,
seeds 0,1,2, wall 7,015 s, peak RSS 4.13 GB; launched 19:18:11). 339,015 rows; F 222.5632 reproduced.

| clause | value | bar | pass |
|---|---|---|---|
| C1 day-block interval (Amendment CAP.1) | gain **+0.578 s**, [+0.320, +0.825] (row, reported: [+0.309, +0.854]) | lower > 0 | ✓ |
| C2 seeds | 0.578 > 2 × 0.116 | > | ✓ |
| C3 months | Jan +0.13, Jul +0.89 | both > 0 | ✓ |
| C4 airports | 7 of 10 (EHAM −0.95, LSZH −0.25, EDDF −0.13) | ≥ 7 | ✓ |
| C5 cuts | fill −1.33 s, tail −2.35 s (both IMPROVE) | loss ≤ 1.0 s | ✓ |

Every single seed beats F's same seed (222.41 / 222.64 / 222.52 vs 223.20 / 223.09 / 222.94). CAP 221.985 vs F 222.563.
**Net weighted fold MSE +253** — below the registered ship bar (+500), so **no ship under this registration**. Reported,
not decisional: here the day-block and row intervals are nearly equal (unlike arm F's 4.6×) — the gain is spread, not
clustered; LIRF +2.84 s and EGLL +1.28 s carry the most. Matched-lane gains have transferred 0.96–1.17×, so a shipped
CAP would be worth ≈ +250 board MSE (≈ −0.45 RMSE) — the owner's call whether a new, separately registered ship is worth
the ≈ 2.5 h refit plus a tested setting override in `lgbm_submit.py`.
