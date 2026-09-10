# Pre-registration — arm FW: the record-ordering block and the weather block, fitted together

**Written 2026-09-10 before 08:57 local (the first edit to `scripts/lgbm_fold.py` for this arm is
stamped 08:57), BEFORE the combined run exists, before its code exists, and before any of its
numbers exists.** *(Timestamps corrected 09:12 from file-system mtimes; the first version of this
file carried guessed times that were later than the true ones.)* Standalone file: the concurrent session owns
`plans/PREREG_taxiout_2026_09_08.md`. This file does not claim an amendment number in that
sequence; the arm's tag is **`FW`**, and the record carries `"amendment": "FW"`.

**Standing of this run under the owner's decision of 2026-09-10 ("245 only, no shipping").** This arm
is INFORMATION, not a submission candidate. Whatever it returns, nothing ships. It answers one
question the fold records cannot: whether two blocks that each moved a different part of the error
distribution stack when fitted together, or trade against each other.

---

## 0. What exists, measured, before this run

| arm | record | vs the queue baseline 224.258 | over-10-min band | body (< 2 min / 2–10 min) |
|---|---|---|---|---|
| F (RESULT 21) | `reports/lgbm_fold_queue_order.json` | **+1.694 [+1.407, +1.993]** | **+12.156 [+10.823, +13.466]** | −2.059 / −0.381 |
| W (RESULT 22) | `reports/lgbm_fold_queue_weather.json` | +0.725 [+0.495, +0.947] | −0.544 [−1.578, +0.439] | **+1.220 / +1.221** |

**Verified 2026-09-10 09:05, the precondition for pairing across runs:** the two prediction records
have 339,015 rows, and `row`, `month`, `ap`, `y`, `proxy`, `delta`, `sp`, `baseline_seed0..2`
and `baseline` are **bit-identical** between them. The baseline reproduces 224.2576 from both.
So a third record built the same way can be paired row-for-row against F's `treatment`, provided the
same identity check passes on it. If that check fails, H-FW2 is VOID, not re-run with a looser
check.

## 1. Arms

`lgbm_fold.py --orderfeats --weatherfeats --queue --baseline A2 --pa-trees es`, fold A, seeds 0,1,2.
Baseline = the queue record (224.258, 80 features), read exactly as arms F and W read it.
Treatment = baseline + the three ORDER_FEATS + the ten WEATHER_FEATS (**93 features**), `best_iter`
re-found. Mode `queue_order_weather`, records `reports/lgbm_fold_queue_order_weather.{json,log}` and
`data/cache_stand/fold_preds_queue_order_weather.parquet`, which never overwrite arm F's or arm W's.

## 2. Hypotheses and locked thresholds

### H-FW1 — the combined block against the queue baseline (reported, on F's footing)

The same four clauses as Amendment 22.3, so FW sits beside F in one table:
(i) paired interval excludes zero in the block's favour; (ii) point gain >= +1.0 s; (iii) gain >
2 × seed sd; (iv) the over10 band's own interval excludes zero. `tools/block_arm_verdict.py` applies
them. **This is not the decisional question**: F alone already passes all four, so FW passing them
says nothing about whether W added anything.

### H-FW2 — DECISIONAL: does fitting W with F beat F alone?

> On fold A's 339,015 matched holdout rows, the FW treatment has lower RMSE than arm F's treatment,
> and the improvement comes from the body bands without giving back F's tail.

Paired on identical rows (FW `treatment` against F `treatment`), both recovered with
`max(proxy − col, 1)` — the DELTA convention, per `reports/bug_classes.md` BC-2. Row bootstrap,
2,000 draws, seed 0.

**Seed sd for clause (iii):** the LARGEST of F's treatment seed sd, FW's treatment seed sd, and the
0.291 s recorded for F and W. The conservative choice is taken now so it cannot be chosen later.

> **CORRECTION 2026-09-10 08:58 (file mtime), before the run started and before any FW number exists.** The
> "0.291" in RESULTs 21/22 is **2 × seed sd**, not the seed sd: `lgbm_fold_queue_order.json`
> `seed_sd.sd` = **0.14538** (source: the queue json); F's own treatment sd is 0.12769. Read
> literally, the sentence above uses 0.291 as the sd and so sets the bar at **2 × 0.291 = 0.582 s**.
> **That stricter bar is KEPT as registered.** Taking the corrected reading (bar 0.291 s) after
> noticing the slip would be choosing a threshold in the looser direction; keeping it costs only
> strictness. Implemented as `tools/pair_arms.py --sd-floor 0.291`: the sd used is
> max(both arms' recorded sds, both treatment sds, 0.291).

- **COMBINES (ESTABLISHED)** iff all four:
  - (i) the paired interval (FW − F) excludes zero in FW's favour;
  - (ii) point gain > 2 × that seed sd;
  - (iii) **the tail is kept:** the over10 band's FW-vs-F interval does NOT exclude zero in F's
    favour;
  - (iv) **the mechanism:** the `< 2 min` and `2–10 min` bands' FW-vs-F intervals BOTH exclude zero
    in FW's favour.
- **NOT WORKING** iff (i) fails.
- **INCONCLUSIVE** otherwise, with every number reported.

### Predicted shapes, written before the run

- **TRUE shape:** FW beats F by roughly +0.3 to +0.7 s; both body bands gain on the order of +1 s
  (W's own body gain, largely surviving); the over10 band within about ±1 s of F.
- **FALSE shape:** FW − F interval straddles zero or favours F. The likely causes are that W's body
  gain overlaps information the order block already supplies, or that the trees spend W's columns
  on the tail and lose F's gain there (the over10 band then favours F with an interval excluding
  zero, failing (iii)).

### Additivity — the quantity being TESTED, never a claim

Reported as one line: `gain(FW vs baseline)` against `gain(F vs baseline) + gain(W vs baseline)` =
+2.419 s. **The sum is the hypothesis under test, not a figure anyone may quote.** The repo rule
against adding gains from different runs stands; this line exists to show by how much the sum is
wrong.

## 3. Controls, required before any number is read

1. **Reproduce known numbers first (BC-2 tripwire 1).** From the FW record, the baseline must read
   224.2576; from F's record, F's treatment must read 222.5632. Either miss → stop.
2. **Row identity.** `row`, `month`, `ap`, `y`, `proxy`, `baseline` bit-identical between the FW and
   F records. Any mismatch → H-FW2 VOID.
3. **Negative control, must score ~0 by construction.** Randomly swap the FW and F predictions on each
   row (200 draws, seed 0) and compute the gain. The real gain is reported as a multiple of the
   control's p99 of |gain|. This is the control that caught both BC-2 instances.

## 4. Named limitations, registered now

- Row-iid bootstrap, as for F and W. `plans/TOP_PATH_2026_09_10.md:259` asks for date-block intervals
  cut per airport and month; the prediction records carry `month` but no date, so a date-block
  interval is not computable from them. Per-airport and per-month cuts ARE reported.
- BC-1 site 1 (the early-stopping encoding leak) is present in this run exactly as in F and W — the
  three are on the same footing, and `best_iter` is biased in all three by an unmeasured amount.
- One fold. Fold totals do not transfer (Amendment 9); only the paired gain is read.

## 5. What this licenses

Nothing ships (owner's decision). A COMBINES verdict establishes that blocks aimed at different bands
stack at roughly their standalone values, which bears on how the tail hunt's findings would be
combined with the incumbent. A NOT WORKING verdict establishes the opposite and is equally useful.
