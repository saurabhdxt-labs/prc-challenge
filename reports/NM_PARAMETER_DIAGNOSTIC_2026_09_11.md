# The four-airport lead (LEBL / LEMD / EDDM / LTFM) — a light diagnostic before any arm

Session prc-challenge-c4. Handoff §6 step 2(a). Tag `NMD`.

## Part 1 — written BEFORE any residual was examined · 2026-09-11 07:10:39 (from `date`)

Already known before this was written (reproductions, not measurements of this diagnostic):
`fold_preds_queue_order.parquet` (arm F, fold A = Jan + Jul 2025, 339,015 matched rows, DELTA convention)
reproduces RMSE **222.5632** as `max(proxy − treatment, 1)`; the four airports carry **16,965 board-MSE (34.8%)** of
its error (LTFM 7,814 · LEBL 3,966 · LEMD 3,077 · EDDM 2,109). The record's `row` indexes the concatenation of the
twelve `data/cache_stand/training_2025-*.parquet` months in month order — verified: y, proxy, delta equal row for row
(max |diff| 0.0), airport and month 100%. The proxy is second-precise (`proxy % 60 == 0` on 1.6–8.3% of rows), so the
NM table value shows up as a MINUTE: `pm = floor(proxy / 60)`.

### Hypothesis

**H-NMD:** at the four airports arm F systematically mishandles the rows whose `AOBT_3` is the NM planned value, and a
correction built only from the NM taxi-parameter axes (serve-time: proxies, never labels) removes a material part of
their error out of month.

### Instrument (fixed now)

- **NM block, serve-time.** Pair = `ars` (`<airport>|<stand>|<runway>`). From the TEN non-holdout months' rows
  (proxies only): `m*(pair)` = modal `pm` (ties → smallest), pair support `n_pair`, mode share `s_pair`. On holdout rows:
  `dev = pm − m*` (integer minutes), `at_mode = (dev == 0)`, pair unseen → its own class.
- **Residual** `r = y − ŷ_F`, `ŷ_F = max(proxy − treatment, 1)`.
- **D1 (descriptive, all ten airports; the six others are the control group — their `AOBT_3` is an actual value):**
  at-mode share; board-MSE and mean `r` for at-mode vs off-mode rows; the within-pair slopes of `y` and of `ŷ_F` on
  `dev` (pair-demeaned, off-mode rows with |dev| ≤ 15 included) — if F follows the proxy's within-pair variation more
  than `y` does, F over-trusts a planned value.
- **D2 (decisional for "room") — cross-fitted residual corrections, four airports' rows:**
  - `K1`: cell = (airport, dev class) with dev classes {unseen, ≤ −6, −5…−2, −1, 0, +1, +2…+5, +6…+15, > +15}; value =
    mean `r` of the cell.
  - `K2`: cell = (pair, {at mode, off mode, unseen}); value = mean `r` shrunk toward the (airport, same class) mean with
    weight `n / (n + 50)`.
  - Fitted on January's holdout residuals and applied to July, and vice versa (no row ever corrected by its own month).
    `ŷ' = max(ŷ_F + c, 1)`. Gain `Σg = Σ [SE(F) − SE(F')]`; board price = `Σg / 339,015 × 0.9846596`.
  - Interval: date-block bootstrap of Σg (take-off dates, January and July resampled separately, 2,000 draws, seed 0).
  - **Negative control NC:** K2 with `ars` permuted within (airport, month) before fitting and applying, and K1 with
    the dev class permuted within (airport, month). Must be ≈ 0 or negative by construction.
  - **Specificity control:** the same K1 / K2 on the six other airports' rows; per-row gain compared.

### Pre-written shapes and thresholds (locked)

- **TRUE (room):** best of K1 / K2 on the four airports ≥ **+300 board MSE**, date-block 95% lower bound > 0, its NC
  ≤ 25% of it, and its per-row gain ≥ 2× the same correction's per-row gain on the six control airports. Expected look:
  F's within-pair slope on `dev` exceeds `y`'s at the four airports (and not at EGLL / EHAM), and the at-mode rows
  carry a mean bias of the same sign across months. → register the arm (`plans/PREREG_nm_parameter_block_2026_09_11.md`)
  and run step 2(b) as the single heavy job.
- **FALSE (no room):** best gain < **+100** or its lower bound ≤ 0. Expected look: F's within-pair slope on `dev` ≈
  `y`'s (F already ignores the planned proxy there and predicts from the other inputs); the at-mode / off-mode residuals
  have no cross-month-stable bias; the error at these airports is high because no per-flight off-block clock exists
  there, which no reshaping of the NM table value can supply. → record "no room" in the ledger.
- **MARGINAL:** anything between (+100 … +300, or specificity fails) → step 2(b) (a LightGBM restricted to the four
  airports, with and without the NM block, and a direct-y target, lr 0.05, day-block interval) decides; run only as the
  single heavy job.

**What this instrument cannot see.** A residual correction prices what F misses ALONG these axes additively; a refit
with the block could also reweight other inputs on at-mode rows (interactions), so a FALSE here does not strictly bound
a refit. It is the cheapest test that can go red, not the last word; that limit is why MARGINAL routes to 2(b). A cell
fitted on one month and applied to the other is also penalised by any Jan ↔ Jul seasonal change — conservative.

### Amendment NMD.1 · 2026-09-11 07:14:52 (from `date`) — BEFORE the real run (no real number of this diagnostic exists)

Harness `scripts/nm_param_diag.py`, tests `tests/test_nm_param_diag.py` (19; 10 targeted mutants rehearsed, all killed —
two survivors on the first pass, the within-pair demeaning and the within-month date resampling, were answered by
strengthening their tests). **A correction to Part 1's description of the negative control, found by the planted-signal
test:** permuting the dev class (or the pair) within airport-month keeps each airport-month's mean residual, so NC is
**not** "≈ 0 by construction" — it prices what a plain airport-mean correction would get (pinned by
`test_negative_control_recovers_an_airport_level_mean_by_design`). The registered clause "NC ≤ 25% of the gain" stands
unchanged and now reads as what it tests: the gain must be carried by the NM axes, not by the airport's mean. No
threshold moves. Interpreter: `/opt/homebrew/bin/python3.11` (pandas 3.0.1); the shell's first `python3` is the CatBoost
venv (pandas 2.3.3), not the project's.

## Part 2 — RESULT NMD · 2026-09-11 07:16:06 (from `date`) — **FALSE: no room along the NM-parameter axes**

Source `reports/nm_param_diag.json` (exit 0, 2.3 s, peak RSS 0.94 GB; console `reports/nm_param_diag.console.log`).
Guards passed: arm F reproduced 222.5632; positional join exact on y / proxy / delta / airport / month; `ars` airport-scoped
on every holdout row; holdout dates aligned to the record row for row. NM table: 11,629 pairs from the ten non-holdout
months.

**D2 (decisional), board MSE, date-block 95% (62 dates):**

| rows | arm | board MSE | 95% | note |
|---|---|---|---|---|
| four airports (136,244) | K1 (airport × dev class) | **−626** | [−734, −516] | LTFM −532 of it |
| | K2 (pair × at/off, shrunk) | **−586** | [−682, −489] | LTFM −514 |
| | NC_K1 / NC_K2 | −585 / −620 | [−679, −490] / [−714, −526] | the airport-month mean alone does the same harm |
| six controls (202,771) | K1 / K2 | −1,574 / −119 | [−2,458, −840] / [−166, −66] | LSZH −1,404 in K1 (sparse far-off classes) |

Best gain on the four airports −586 < +100 and its whole interval is below zero → **FALSE (no room)** under the locked
rule. Every correction pays because the airport-month mean residual reverses between the two months (LTFM −41 s in
January, +27 / +40 s in July; the at-mode and off-mode classes move together), so nothing along these axes transfers
across months.

**Comparison with the pre-written shapes.** The FALSE shape's expected look is what appeared, on each count:

| pre-written FALSE expectation | measured |
|---|---|
| F's within-pair slope on `dev` ≈ `y`'s (F already ignores the planned proxy) | LEBL ŷ −27.65 vs y −28.21 s/min · LEMD +1.53 vs −2.68 · EDDM +1.40 vs −0.85 · LTFM 14.57 vs 18.50 (controls: EGLL 18.35 vs 17.88, EHAM 23.27 vs 21.63, LSZH 70.69 vs 69.48) |
| at-mode / off-mode residuals have no cross-month-stable bias | mean r at-mode Jan / Jul: LEBL −12 / +6, LEMD −21 / −10, EDDM −19 / +11, LTFM −41 / +27; off-mode moves the same way |
| the error is high because no per-flight off-block clock exists | at-mode rows are NOT F's worst rows: RMSE at-mode vs off-mode LEBL 212 vs 263, LEMD 164 vs 228, EDDM 137 vs 181, LTFM 226 vs 252 |

The TRUE shape's look (ŷ slope above y's; a stable at-mode bias) did not appear anywhere. What the four airports' 16,965
board MSE is: ordinary per-flight uncertainty with no clock to anchor it (at LEBL / LEMD 87–88% of holdout rows sit on the
table minute; at EDDM / LTFM ~40%), which F already handles as well as its other inputs allow. **A re-shaping of the NM
table value cannot supply information that is not in it.** Step 2(b) is NOT run (the rule routes only MARGINAL there).

Side observations, recorded and not claimed: (1) at LSZH 81% of holdout rows sit on the table minute too, yet its
off-mode deviations carry real clock information (y slope 69.5 s/min) — LSZH's AOBT_3 is a mixture, not "planned"; (2)
LTFM's fold residual has a month-level level shift of ≈ ±35–40 s between January and July — the fold model trains on
neither month; the shipped model trains on all twelve, so this says nothing about v10's 2026 bias.

### Post-hoc sensitivity (NON-decisional) — definition written 2026-09-11 07:16:06, BEFORE it was computed

Because the cross-month instrument is dominated by the month-level shift, one optimistic bound is added: the IN-SAMPLE,
within-airport-month between-class sum of squares, `Σ_cells n_c (mean_c − mean_airport-month)²`, for K1's classes and for
K2's pair × at/off cells (unshrunk — maximally overfit). It is what an oracle additive correction along the NM axes could
remove after the airport-month mean is taken out, with no out-of-month transfer penalty. It cannot change the verdict. If
this ceiling is itself below +300 board MSE on the four airports, the lead is empty along these axes even in-sample.

**Computed 2026-09-11 07:17:41** (`reports/nm_param_diag.json` → `posthoc_nondecisional`). Four airports, board MSE:

| in-sample oracle | real | permutation null (r shuffled within airport-month, 20 draws: mean / p95) | above noise |
|---|---|---|---|
| K1 dev classes beyond the airport-month mean | **91** | (≈ 72 cells; negligible overfit) | ≤ 91 |
| pair identity alone (no NM axis) | 1,229 | 807 / 852 | ≈ 422 — not NM-specific |
| pair × at/off (K2 cells, unshrunk) | 1,880 | 1,322 / 1,393 | ≈ 558 |
| **at/off split beyond the pair** | 651 | 515 | **≈ 136** |

Even as an in-sample oracle, the NM-specific structure (the at-mode flag beyond the pair, and the deviation classes) is
worth ≈ 90–140 board MSE on the four airports, below the +300 bar. The ≈ 422 of pair-level structure is within-month
stand×runway residual that did not transfer across months (K2 cross-fitted lost 586) and exists equally at the control
airports (in-sample pair ceilings: LFPG 1,046, LIRF 801 before any noise correction). **The sensitivity agrees with the
verdict: no room along the NM-parameter axes.**

The LSZH control figure (K1 −1,404) was checked, not assumed: it is one July row (mean r −2,104) forming the `>+15` class,
applied as a correction to 119 January rows (−1,360) — K1 is unshrunk as registered, which is why K2 is the arm to read.

### What this closes and what stays open

- **Closed (FALSE):** "at LEBL / LEMD / EDDM / LTFM F mishandles the planned-value rows; an NM taxi-parameter block
  recovers thousands." F already uses the planned proxy correctly there (slopes match y's), and the NM axes carry ≤ ~140
  board MSE even as an in-sample oracle. The four airports' 16,965 board MSE is missing per-flight information, not
  model error on the proxy.
- **Not closed by this:** whether ANY other serve-time source carries per-flight off-block information at those airports.
  `reports/ADSB_GATE.md`: zero on-ground ADS-B at LEMD and LTFM; LEBL 25–33% coverage with an off-block channel RMSE of
  687–792 s; **EDDM 65–77% coverage, channel RMSE 257–278 s (failed the census bar)** — EDDM is the only one of the four
  where a per-flight clock could exist outside NM, and it is prc-challenge-6e's lane (EDDM = 2,109 of the 16,965). The
  other per-flight signals are the physical ones already in F (queue, stand, weather, type, airline).
