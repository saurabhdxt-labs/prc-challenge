# Where the unmatched lane's error is, and how much of it is learnable — diagnostic `UMD`

Session prc-challenge-c4. Owner: "yeah" (run it), "how close can we go to 245".

## Part 1 — written BEFORE any residual was examined · 2026-09-11 08:23:50 (from `date`)

**Why.** v10 = 79,252 board MSE. The matched lane is ≈ 48,800 of it (fold 222.56² × 0.98466 at the ≈ 1:1 transfer seen
four times), so the unmatched lane (5,290 rows, 1.5% of rows) is ≈ 30,000 — inferred, not measured (the board gives one
number). Position 1 needs −19,217. Every matched lever measured is worth hundreds. The question here is whether the
unmatched lane's error is **learnable** (a mechanism over many rows) or a **row draw** (a few extreme rows).

**Hypothesis H-UMD:** on the 2026 file's own composition, ≥ 3,000 board MSE of v10's expected unmatched error sits in
row classes with a serve-time mechanism and enough 2025 support to learn from (ordinary taxis; LIRF fills with
sp < 10,800), rather than in Family B, 24 h date-slips or large-sp fill monsters.

### Instrument (fixed now)

- **v10's unmatched composition on 2025, out of month** (12-month LOMO, the harnesses' own folds, `fold == "lomo"`):
  non-LIRF rows = `S1C` from `data/cache_stand/unm_congestion_lomo.parquet` (taxi-time convention; guard: RMSE equals
  E3C's JSON on its reported cuts); LIRF rows = `R2` from `data/cache_stand/rome_dateslip_preds.parquet` (guard: LOMO
  RMSE 4,098.644), then E1 as shipped: `max(rint(R2), rint(sp))` on `24,000 ≤ sp < 86,400`. **E1 is IN-SAMPLE on 2025**
  (its rule was read off 2025: no such row has `y < sp − 60`), so the LIRF large-sp rows are flattered — named, not fixed.
  Row sets must agree (ids equal between the two records on LIRF rows).
- **Row classes (label-defined, mutually exclusive, in this order):** `fill` |sp − y| ≤ 60; `dateslip` not fill and
  y > 80,000; `familyB` not fill, y > 10,800 and sp < 10,800; `monster_other` not fill, y > 10,800; `ordinary` the rest.
- **Serve-observable strata:** airport × sp band, bands `< 0`, `0–1,800`, `1,800–3,600`, `3,600–10,800`,
  `10,800–24,000`, `24,000–86,400`, `≥ 86,400` (s).
- **2026 expectation:** per stratum, 2025 mean SE (and its class split) × the 2026 scored file's unmatched row count in
  that stratum ÷ 344,841 = expected board MSE. A stratum with no 2025 row borrows the airport's mean (counted, reported).
  The 2026 counts come from the pipeline's own ingest (`prc.pipeline.ingest.ingest_scored`), never from a submission.
- **Calibration check (reported):** the 2026 expectation's total vs the inferred ≈ 30,000.
- **Concentration:** per class, the share of its 2025 SSE in its top 10 rows; the Kish effective sample size of SE.
- **Learnable** = `ordinary` + `fill` rows with sp < 10,800 (LIRF's 1–3 h separation, where `STRATUM_MONSTERS.md` §3b
  established a gain); **not learnable** = `familyB` (closed five ways, `FAMILY_B.md` §10), `dateslip` (12 rows),
  large-sp fill monsters (sp ≥ 10,800: the E1 / R2 territory already mined). This split is fixed now.

### Pre-written shapes and decision (locked)

- **TRUE:** learnable classes ≥ **3,000** expected board MSE on 2026, spread (each learnable class's top-10 share < 30%).
  → register the next arm on the largest learnable block (plans/PREREG_…, before any fit).
- **FALSE:** learnable < **1,500**, or the expectation is carried by few rows (top-20 rows of the whole lane > 50% of its
  SSE). → record "the unmatched lane's remaining error is mostly a row draw"; no arm; the owner decides strategy with
  that on the table.
- **MARGINAL:** 1,500–3,000 → one arm may be registered only with a bar that prices its 2026 share conservatively.

**Limits named now.** 2025 LOMO is not 2026: the January 2026 EHAM storm (the E3C witness's hottest bin) has little
2025 support, and Rome's 2025 → 2026 transfer has been 0.15× (R2). The class split of a 2026 stratum is assumed equal to
2025's (the class is label-defined and unknown on 2026). Expected-MSE arithmetic on a heavy tail (Kish ESS ≈ 23 on the
stratum before) is itself noisy; the concentration numbers are there to say how noisy.

## Part 2 — RESULT UMD · 2026-09-11 08:27:20 (from `date`) — **AMBIGUOUS by the rule's own construction → no arm (conservative default)**

Source `reports/unm_diag.json` (`scripts/unm_diag.py`, exit 0, 1.4 s, 0.63 GB; tests `tests/test_unm_diag.py` 6, 10 targeted
mutants killed). Guards reproduced E3C's S1C (1,566.5011 / 914.4572) and R2's LOMO RMSE (4,098.6442); LIRF ids, labels
and sp equal across the two records; no 2026 stratum needed borrowing. 22,219 rows in 2025; v10's composition out of month
RMSE 1,471.7 (E1 in-sample).

**Calibration:** 2026 expectation total **32,038** board MSE vs ≈ 30,478 inferred from the board (+5%).

| class (2025 label) | 2025 rows | share of 2025 SSE | top-10 share | Kish ESS | 2026 expected board MSE |
|---|---|---|---|---|---|
| 24 h date-slip | 14 (LIRF 12, LFPG 1, LSZH 1) | **48.0%** | 99.8% | 3.4 | 12,971 |
| ordinary | 20,801 | 26.9% | 16.4% | 217 | 8,578 |
| fill | 1,396 (LIRF 722, LTFM 344, EHAM 165) | 14.8% | 42.1% | 33 | 7,825 |
| Family B | 8 | 10.3% | 100% | 2.1 | 2,665 |
| **learnable = ordinary + fills with sp < 10,800** | 22,033 | 33.6% | **13.1%** (fills 18.8%, ordinary 16.4%) | — | **10,809** (LIRF 6,731; other nine 4,078) |

By airport (2026 expected): LIRF 20,724 (65%), LSZH 4,482, LFPG 3,753, LTFM 1,431, the other six ≤ 577 each.

**Against the locked rules.** TRUE's two conditions pass (learnable 10,809 ≥ 3,000; each learnable class's top-10 < 30%).
FALSE's second disjunct ALSO fires (the lane's top-20 rows hold **64.5%** of its SSE > 50%) — and it fires on exactly the
22 rows Part 1 classed as NOT learnable. **Defect in Part 1, named:** the FALSE concentration clause measured the whole
lane where it should have measured the learnable part, so the rules contradict each other on this outcome. Under the
hypothesis-closure default (ambiguous → the conservative branch) **no arm is registered from this diagnostic alone**;
the decision is the owner's, with the facts below. Neither branch is chosen in hindsight.

**Against the pre-written shapes.** Both looks appeared, each in its own part of the lane: the TRUE look (a large,
spread learnable pool) in the ordinary / small-sp fill rows; the FALSE look (a few rows carry the expectation) in the
date-slip and Family B rows — 22 rows, 58.3% of 2025 SSE, Kish ESS 2–3.

**What the numbers do and do not say.**
- "Learnable" is the error currently SITTING in classes where learning is possible — the pool, not the prize. For scale:
  E5, a real physics block on the non-LIRF body, recovered ≈ 5–8% of that part (+208–308 of ≈ 4,078).
- **A bound on the owner's question.** Even a perfect model of every learnable row (10,809) plus the matched ship
  (≈ +557) reaches ≈ **260.6**, not 245. Recovering 10 / 20 / 30% of the learnable pool (+ matched ship + E5) gives ≈
  278.2 / 276.3 / 274.3. **245 is not reachable through learnable unmatched rows**: the remainder is in the date-slip /
  Family B rows (≈ 15,600 expected on 2026, from 22 rows of 2025 — too few to learn from; the leader's one-submission
  jump was one ≈ 50,000 s row, class not known to us) or in a matched lane much better than ours.
- The 2026 date-slip figure (12,971) rests on 12 LIRF rows of 2025 (Kish ESS 3.4) mapped onto 2026's large-sp LIRF rows;
  the E1 board reading (PARTIAL) and R2's 0.15× transfer say 2026's large-sp rows behave differently. Treat it as the
  size of a lottery, not as a forecast.
- Largest learnable block: **LIRF's ordinary + small-sp fill rows (6,731 expected; 1,308 rows in 2025: 750 ordinary,
  558 fills)** — where
  `STRATUM_MONSTERS.md` §3b's fill logistic already established a gain (now inside R2). Next: the non-LIRF body (4,078;
  E3C and E5 already mined it).

*Corrections, 2026-09-11 08:29 (before anything used them):* three figures in Part 2 were first written from memory — the
LIRF learnable count (1,472 → **1,308**), E5's share of the non-LIRF pool (≈ 3% → **5–8%**) and the class of the leader's
≈ 50,000 s row (asserted → **unknown**). Corrected above from `reports/unm_diag.json` / the records.
