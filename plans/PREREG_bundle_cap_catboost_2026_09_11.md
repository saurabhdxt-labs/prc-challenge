# Pre-registration — arm BND: the consolidated matched build (CAP + a CatBoost blend), staged by cost

**Written 2026-09-11 07:47:58 (from `date`), session prc-challenge-c4, BEFORE any number of this arm exists.** Tag `BND`.
Owner goal (2026-09-11 handoff §6 step 4): "register one consolidated build — CAP setting, CatBoost blend weight fitted on
inner OOF only, E5 body — measured with day blocks; ship only if the registered bar clears." Inputs, all already on disk
and all measured against arm F on fold A (Jan + Jul 2025, 339,015 matched rows):

| component | record | its own verdict |
|---|---|---|
| F (arm F, shipped since v6) | `data/cache_stand/fold_preds_queue_order.parquet` (DELTA; seeds 0–2) | 222.5632 |
| CAP (`127, 20, 0.6` on F's design) | `data/cache_stand/capacity_f_preds.parquet` (`cap_seed{0,1,2}`, `cap`) | ESTABLISHED, 221.985, +253 net |
| C_delta (native CatBoost, 1 seed) | `data/cache_stand/catboost_native_delta.npy` (+ `.json`) | alone −1,422; `0.5·C + 0.5·F` 221.345, +533 (screen) |

## Why staged

The full inner-OOF build is expensive: a CAP fit and a CatBoost fit on an inner split (≈ 2 h + ≈ 2 h), two more
CatBoost seeds on fold A (≈ 4 h, 6.4 GB each), then the ship refits (CAP ≈ 2.5 h; CatBoost × 3 on twelve months). None
of it is worth spending if CatBoost's complementarity to F (the screen's +533, carried by the |delta| > 10 min tail and
LIRF) is already bought by CAP — CAP itself improved the fill (−1.33 s) and tail (−2.35 s) cuts. Stage 0 answers that
from stored predictions at zero compute; Stage 1 is registered now but runs only if Stage 0 passes.

## Hypothesis

**H-BND:** on fold A, a blend of C_delta with CAP lowers matched error beyond CAP alone, by enough to be worth the
compute; and with its weight fitted on inner out-of-fold predictions only, the blend beats F and CAP by the ship bar.

## Stage 0 (zero compute; decides only whether Stage 1's compute is spent — never a ship)

- Guards first (BC-2): F → 222.5632 as `max(proxy − treatment, 1)`; CAP → 221.985 (the convention is read from the
  harness that wrote it and must reproduce this number, or nothing is scored); C_delta → 225.784 as `max(proxy − pred, 1)`;
  `0.5·C_delta + 0.5·F` (taxi-time scale, both floored) → 221.345. Row alignment of all three to F's `row` asserted
  (labels equal row for row). If any guard misses by > 5e-4 s, Stage 0 stops with the defect named.
- **Arm B0 = 0.5·C_delta + 0.5·CAP** (taxi-time scale, both arms floored at 1 before blending; the 0.5 is the weight
  the screen fixed before it ran — not fitted here).
- **Gate G0 (locked):** vs CAP, net weighted fold MSE (0.9846596 × ΔSSE / 339,015) ≥ **+300** AND the paired
  calendar-date block bootstrap (take-off dates, Jan / Jul resampled separately, 2,000 draws, seed 0) lower bound > 0.
  Pass → Stage 1 is proposed to the owner with its compute cost. Fail → **the CatBoost blend is closed at this
  learner / this design** and the bundle reduces to CAP alone (whose own ship bar, +500, it does not meet).
- Reported (not decisional): B0 vs F; per-airport, month, fill (|y − sp| ≤ 60), |delta| > 600 s tail cuts; the
  blend weight that WOULD minimise the holdout MSE — printed as a diagnostic of how far 0.5 is from optimal, never used.
- **Named limit:** January and July 2025 are inspected development months — CAP's and the screen's numbers were read on
  them. B0 on the same rows is a development reading, not a transfer test; Stage 0 can only close the path, not open a
  ship.

## Stage 1 (heavy; registered now, runs only after G0 passes and the owner accepts its cost)

- Inner split inside fold A's training months: inner holdout = months **3 and 9**; early stopping on months **5 and
  11**; inner training = the remaining six months. BC-1 applies (target encodings must be fitted on the inner training
  mask only, never on stopping or inner-holdout rows) — the repaired encoder `prc/encoding.py` is wired for these fits,
  or Stage 1 does not run.
- Fit CAP (3 seeds) and C_delta (seed 0) on the inner split; predict months 3 and 9. Fit ONE scalar weight
  `w ∈ [0, 1]` minimising inner-holdout MSE of `w·C + (1 − w)·CAP` (taxi-time, floored). Record `w` before any fold-A
  holdout row is scored with it.
- Fit C_delta seeds 1 and 2 on fold A (the screen's design exactly) so the CatBoost side has three seeds.
- **Arm B1 = w·mean_seeds(C_delta) + (1 − w)·CAP** on fold A's holdout.
- **Clauses (locked; CAP's five, on B1 vs F):** C1 date-block lower bound > 0; C2 gain > 2 × seed sd (three paired
  single-seed blends); C3 both months improve; C4 ≥ 7 of 10 airports; C5 fill and tail cuts each lose ≤ 1.0 s.
  **Verdict:** ESTABLISHED iff all five; NOT WORKING iff C1 fails; else INCONCLUSIVE. Never revised.
- **Ship rule (matched lane):** ESTABLISHED **and** net vs F ≥ **+800** weighted fold MSE (CAP's +500 bar plus the
  extra learner's compute and maintenance) → the pipeline's matched lane becomes the blend (CatBoost as a plug-in over
  `scripts/catboost_native_worker.py` in its own venv; never lightgbm and torch in one process, BC-5), refitted on all
  twelve months, A1-style guard: the pipeline with `w = 0` must reproduce the CAP-only build exactly.

## E5 (the unmatched body) — NOT in the decisional bundle

E5's own registration returned INCONCLUSIVE (C2: event-excluded 2026 price +208 < 300; gain concentrated in LTFM-02 41%
and the NaN-witness bin 30%; the predicted weather shape did not appear). Folding it into a bundle would ship it under a
bar other than the one it failed — a goal-post move. It is therefore priced beside the bundle at its conservative +208
and enters a ship **only on the owner's explicit, separate acceptance**, recorded in the ledger with that reason.

## Predicted shapes (written now)

- **TRUE (G0 passes):** B0 beats CAP by ≈ +300–500 weighted MSE, carried like the screen's blend by the |delta| > 10 min
  tail and LIRF, with EHAM / LSZH slightly negative; the holdout-optimal weight near 0.3–0.5.
- **FALSE (G0 fails):** B0 − CAP ≈ 0–200 — CAP's smaller leaves already capture what CatBoost's categorical handling
  added in the tail; the holdout-optimal weight near 0–0.2.

## What this build is worth against the target — stated before measuring

Even at the optimistic end (CAP +253, blend ≈ +500, E5 +208), the bundle is ≈ +1,000 board MSE: v10 281.52 → ≈ 279.7.
Position 1 (245.02) needs −19,217 MSE; 10th (≈ 275.1) needs ≈ −3,570. **This build is polish on the matched lane; it
cannot reach the owner's target by itself.** Where the rest of the gap lives is the subject of the next registration,
not this one.

## RESULT BND Stage 0 · 2026-09-11 07:52:35 (from `date`) — **G0 PASSES, at the margin** (+303.7 vs a +300 bar)

Source `reports/bundle_stage0.json` (`scripts/bundle_stage0.py`, exit 0, 1.1 s, 0.35 GB; tests `tests/test_bundle_stage0.py`,
7 targeted mutants killed). All four guards reproduced their own records (F 222.5632, CAP 221.9850, C_delta 225.7836,
0.5·C + 0.5·F 221.3446); the three records are the same 339,015 rows with equal labels and proxies.

| arm | RMSE | vs CAP (weighted fold MSE, date-block 95%, 62 dates) | vs F |
|---|---|---|---|
| CAP | 221.985 | — | +253 (its own record) |
| **B0 = 0.5·C_delta + 0.5·CAP** | **221.289** | **+303.7 [+97.5, +510.1]** | +556.8 [+319.9, +793.1] |

G0: net ≥ +300 ✓ (by 3.7) and lower bound > 0 ✓ → **pass**: Stage 1 is proposed to the owner with its cost (below).

**Against the pre-written shapes: the TRUE shape appeared, at its low edge.** B0 − CAP +304 (TRUE predicted +300–500);
LIRF +180 of it (59%), LTFM +82, EGLL +39, EHAM −14 and LSZH −14 (TRUE predicted "carried by the tail and LIRF, EHAM /
LSZH slightly negative"); |delta| > 600 s tail +3.32 s, fill +2.82 s; the holdout-optimal weight 0.317 (TRUE predicted
0.3–0.5; FALSE 0–0.2). Months: Jan +116, Jul +187. CAP bought ≈ 43% of the screen's CatBoost complementarity (+533 → +304).

**Post-hoc, for compute allocation only (not a verdict; `posthoc_compute_allocation_NOT_DECISIONAL` in the JSON):** the
best ANY scalar weight reaches on this development holdout with one CatBoost seed is **+710 vs F** (w = 0.317) — below
Stage 1's registered ship bar of **+800**. Stage 1 would add two CatBoost seeds (CAP's own seed averaging bought +0.54 s)
but fits its weight on inner OOF, which cannot beat the holdout-optimal weight on these rows. **Stage 1 (≈ 8 h of heavy
compute) is therefore unlikely to clear its own bar, and the bar is not lowered after seeing this.** Recommendation to
the owner: do not spend Stage 1's compute under this registration. If a smaller matched ship is wanted, it needs its own
registration with its bar set on grounds other than this reading (e.g. the ledger's matched-lane rule), and costs the
CAP refit (≈ 2.5 h, ≈ 5 GB) plus a twelve-month CatBoost refit (≈ 2 h, 6.4 GB) for an expected ≈ +550 board MSE
(≈ −1.0 RMSE; 281.52 → ≈ 280.5).
