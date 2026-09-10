# Pre-registration — arm M: a stake-weighted fill mixture on Rome's MATCHED rows, over arm F

**Written 2026-09-10 10:16 local, BEFORE the script exists and before any of its numbers exists.** Tag `M`.

**Why.** Arm F's fold record (`fold_preds_queue_order.parquet`, reproduces 222.563): LIRF matched
fill rows (`|y − sp| ≤ 60`) are 18.9% of LIRF's 26,131 holdout rows and carry **41.4% of its SSE**
(fill-row RMSE 555.6 vs 319.7); perfect handling = **≈ 4,435 board MSE**. Arm R (unmatched) showed a
stake-weighted classifier captures roughly a third of such an oracle.

## Arms (LIRF matched holdout rows of fold A only; every other row keeps F's prediction)

`F` = arm F's treatment, `max(proxy − treatment, 1)` (DELTA convention, BC-2). `p` = mean over
seeds 0,1,2 of a LightGBM binary fill classifier (`scripts/rome_fill.py`'s params and stake weight,
unchanged) trained on ALL LIRF departures of the ten fold-A training months (matched + unmatched),
label `|BLOCK − SCHED| ≤ 60 s`, features = rome_fill's plus the NM-clock columns present on matched
rows: `proxy, nmdelay, eobt_p, lobt_p, iobt_p, aobt_eobt, lobt_sched, eobt_sched` (NaN on unmatched).

| arm | LIRF matched rows |
|---|---|
| `F` | F (control) |
| **`M`** | `max(p·sp + (1−p)·F, 1)` |
| `M_g` | M where `p ≥ 0.5`, F elsewhere (the gate, registered now) |
| `M_perm` | M with within-month permuted training labels (negative control) |

## Thresholds, locked — for M and for M_g separately, on LIRF matched fold-A rows

1. paired row bootstrap (2,000 draws, seed 0) on RMSE(F) − RMSE(arm) excludes zero in the arm's favour;
2. point gain ≥ **+5.0 s** of LIRF matched RMSE (~ +490 board MSE at LIRF's weight);
3. **both** holdout months (1 and 7) improve on their own;
4. **the body is bounded:** on non-fill rows the arm's RMSE may not exceed F's by more than **1.0 s**;
5. gain > 2 × seed sd (sd over the three single-seed versions of the arm).
**NOT WORKING** iff (1) fails; otherwise INCONCLUSIVE unless all five hold. Control: `M_perm` must
NOT pass clause 1. Reported: fill/body subset RMSEs; projected board MSE = 0.98466 × ΔSSE / 339,015 (the 2026 matched weight over fold A's matched rows).

## Named limitations

Fold A only (arm F's record exists only there); no LOMO. F is an all-rows model, so M double-counts
fill mass where p is intermediate — clause 4 exists to catch exactly that (RESULT 8's failure mode).

---

# RESULT M · 2026-09-10 10:21 local

`scripts/rome_matched.py` → `reports/rome_matched.json`. 26,131 LIRF matched fold-A rows (all mapped to
MVT_ID through the order cache, labels equal); arm F reproduces 222.5632. Control `M_perm` −283.2 s
[−344.2, −239.0] — does not pass clause 1.

| arm | RMSE | gain [95%] | month 1 / 7 | body loss | fill-row RMSE | clauses (1–5) | verdict |
|---|---|---|---|---|---|---|---|
| F | 375.90 | — | — | — | 555.6 | — | control |
| M | 376.08 | −0.18 [−5.25, +5.13] | −11.8 / +5.6 | **+38.8 s** | 443.7 | ✗ ✗ ✗ ✗ ✗ | **NOT WORKING** |
| M_g (p ≥ 0.5) | 369.72 | **+6.18 [+2.65, +9.79]** | +1.5 / +8.6 | **+7.1 s** | 514.4 | ✓ ✓ ✓ **✗** ✓ | **INCONCLUSIVE** |

Projected board: M ≈ −10, M_g ≈ +350 MSE. **The fill branch is real (fill rows 555.6 → 443.7) and
the body pays for it** — RESULT 8's double counting, caught by the clause registered for it. Blending
over an all-rows model is closed as a route to Rome's matched fills; the route that remains is a
conditioned body model (E[delta | X, F = 0] on LIRF's non-fill rows), which is a refit.
