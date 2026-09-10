# The gap to the pack is located: the unmatched lane, and mostly Rome

**2026-09-10, late morning.** Research requested by the owner ("something is seriously off"). Every
number below is measured in this repo or read from a public source that is linked. No model was
fitted; no submission was made; the board was not probed (only public API data was used).

## 1. The board says we are behind on ORDINARY modelling, not on the 245 trick

- Public leaderboard (`datacomp.opensky-network.org/.../leaderboard`, 731 submissions, 103 teams):
  #1 youthful-giraffe 245.02; #2–#6 at 260.9–268.5; us 288.14 (rank 21).
- The leader sat at 262.7–266 with the pack, then dropped **262.74 → 248.48 in ONE submission
  (2026-09-08)** ≈ 2.5e9 squared-seconds ≈ one row corrected by ~50,000 s.
- Teams scoring **281–283 on their first day** (quick-boat, upbeat-goblin) beat our best.

## 2. The date-slip row: we already have it

kind-mango's public repo documents one LIRF matched row with MVT − SCHED = 94,560 s; moving it
5,465 → 77,604 moved their board 370.63 → 345.70 (−6.16e9 SSE). Solving
`(T − 5,465)² − (T − 77,604)² = 6.16e9` gives **truth ≈ 84,200 s. v5 predicts 84,904.** A team
predicting an ordinary value there pays ~18,000 MSE on that row alone — so on every other row the
pack is **≥ 11,000 MSE** (if they have the row) to **~30,000 MSE** (if they do not) better than us.

> **CORRECTION, same day, before anything was built on §3.** §3's "ours" column is **S0** — the
> pre-v4 unmatched model (`fold_preds_queue_allrows.parquet`'s `pipeline` equals S0 exactly on all
> 5,321 rows). v4/v5 ship **S1**, the stratum hybrid. Against S1 (`stratum_fold_v7_preds.parquet`,
> fold A, July):
>
> | July 2025 unmatched | S1 RMSE | theirs | S1 SSE | their SSE |
> |---|---|---|---|---|
> | **LIRF** (337) | **3,925.6** | 3,437.9 | **5,193 M** | 3,983 M |
> | LTFM (425) | 704.4 | 690.0 | 211 M | 202 M |
> | EHAM (515) | 245.2 | 329.6 | 31 M | 56 M |
> | EDDF (435) | 387.8 | 401.2 | 65 M | 70 M |
> | all ten | | | 6,447 M | 5,302 M |
>
> The gap is **1,145 M = 6,006 MSE** in July's frame (not 7,154), and it is **entirely Rome**:
> LIRF alone is 1,210 M; the other nine airports are, together, 65 M *better* than theirs. S1's July
> all-rows RMSE is ≈ 301.6, so the all-rows gap to their 291.5 is ≈ 5,990 MSE — still fully
> accounted for by the unmatched lane. The conclusion stands; the size is ~6,000, and the target is
> narrower: **LIRF's unmatched rows**.

## 3. Located on 2025 labels, against a public system with a known board score

elegant-alligator (board 274.34 → 271.04, GPLv3, `github.com/javidmardanov/PRC-Data-Challenge-2026`)
publishes July-2025 validation numbers from models that held July out of training.

| July 2025 | ours (`fold_preds_queue_allrows.parquet`, `pipeline`) | theirs |
|---|---|---|
| all rows | **303.50** | 291.50 (v3), 294.34 (v2) |
| unmatched lane, SSE | **6,666 M** | 5,302 M (v1 missing model) |
| of which LIRF (337 rows) | **3,928 RMSE / 5,200 M** | 3,438 RMSE / 3,983 M |
| of which LTFM (425 rows) | 863 / 317 M | 690 / 202 M |
| other eight airports | within a few % | |

The unmatched-lane SSE difference, **1,364 M over July's 190,678 rows = 7,154 MSE**, is essentially
the ENTIRE July all-rows gap (303.50² − 291.50² = 7,140 MSE). Scaled to both scored months it is
~7,800 MSE, against our real board gap to them of 7,764 MSE. **89% of it is Rome.**

Reproduce ours: `fold_preds_queue_allrows.parquet` stores TAXI TIMES (BC-2 map); matched
`pipeline` reproduces 224.258 before anything else is read.

## 4. What their Rome model does that ours does not (read from `docs/missing_v2.json`)

1. **A separate normal expert trained on non-identity rows only** (`target < 6000 and nonidentity`,
   CatBoost 600×d6, native categoricals, window counts, stand/flight prefixes) — i.e. `E[y|X,F=0]`,
   the conditioned body expert `reports/PRIORITY2_FILL_LANE_PRICED.md` named and never fitted.
2. **A Rome-only fill classifier weighted by the MSE stake**:
   `sample_weight × clip(1 + (max(sp,0)/3600)², 1, 25)`, target `|y − sp| ≤ 6 s`.
3. Mixture `p·sp + (1−p)·normal`, `p = 0` when `sp ≤ 0`, blended into the pipeline with a validated
   Rome alpha (0.42). Rows with `sp ≥ 24,000` at Rome keep their earlier pipeline.

## 5. Hypotheses closed today (with evidence)

| hypothesis | verdict | evidence |
|---|---|---|
| BLOCK_TIME is an exact copy of another serve-time timestamp | NOT WORKING | ≤ 0.7% exact equality on any candidate; collisions ~3% in every subset alike |
| EHAM 2026 long-`sp` surge is mispredicted by v5 | NOT WORKING | v5 implied fill weight ≈ 0 at EHAM, matching 2025's 0% fill rate |
| v5 mispredicts the 2026 long-proxy regime | NOT WORKING | isolated long proxies: 2025 truth 0.21×proxy, v5 0.21–0.33×; disruption days 0.78–0.81 vs 0.85–0.91 |
| Stand occupancy separates LIRF fills from genuine delays | NOT WORKING | positive control 99.6% / 87.5%; zero reuse of the stand in 1,665 delay windows, fill or not — fills are recording artifacts |
| "~630 rows with BLOCK ≈ AOBT − 1 h" (Discord) is a separable class | NOT WORKING | no sharp spike at ±3,600 s in 1-min bins; LTFM +3,600 bump is broad and already handled by the long-proxy regime |
| The gap to 271–274 teams is in the matched lane | NOT WORKING | on July 2025 the unmatched lane alone accounts for 100% of it |

## 6. What this does and does not license

It licenses rebuilding the unmatched lane — Rome first — as a conditioned normal expert plus a
stake-weighted fill classifier, measured on fold A per airport against the current pipeline and
against the public July anchor (LIRF 3,438). Matching that anchor is worth ~7,000 board MSE, ~10×
arm F. **It does not license a claim about 245**: it closes the gap to the 271–274 tier; the
leader's remaining edge is not explained by anything found here. Their code is GPLv3 and read for
ideas only; our implementation is our own.
