# Pre-registration: BNU — is the CAP + CatBoost blend worth building for the rows ADS-B cannot reach?

**Written 2026-09-11 18:30:44 EDT by prc-challenge-70, BEFORE the restricted number is computed.** Zero compute: it
reads the three stored fold-A records (arm F, CAP, C_delta) that BND Stage 0 already validated, and the ADS-B coverage column.

## Why

v12 (266.3738, 5th) corrects 166,288 matched rows with ADS-B. The blend B0 = 0.5·CAP + 0.5·C_delta was measured at
**+556.8 weighted fold MSE vs F [320, 793]** over all 339,015 rows, and shelved as below its ship bar. On rows the ADS-B
stage moves, B0's gain and ADN's overlap (both correct the same F). On rows it does NOT move, they cannot overlap:
ADN's gain there is exactly 0 by construction.

## Hypothesis BNU

> On fold A's rows with ADS-B coverage < 2 — the rows the stage leaves at F — the blend B0 keeps enough of its gain over F
> to be worth a registered ship step.

## Statistic

Per row g_i = (taxi_F − y)² − (taxi_B0 − y)², restricted to coverage < 2; net = 0.9846596 · Σg / 339,015 (the ledger's
convention, so the number is directly comparable with every other lever); 95% interval by the calendar-date block bootstrap
(2,000 draws, months resampled separately), as in BND Stage 0.

## Shapes, fixed now

- **TRUE (worth registering a ship step):** net ≥ +400 weighted fold MSE on the uncovered rows AND the interval's lower bound > 0.
- **FALSE (not worth the refits):** net < +400, or the interval includes 0. The lane closes for this build; it can return only
  with a different design.
- Reported, not decisional: the same on covered rows; the split by airport; and a crude 2026 scaling, since the uncovered
  share falls from 61.8% (2025, 209,575 of 339,015) to 50.8% (2026, 172,552 of 339,551).

## If TRUE

It is NOT a ship on this evidence: CAP needs a ≈2.5 h refit and C_delta a CatBoost plug-in in its own venv (BC-5), and the
ship step needs its own registration with a bar, a build and an A1-style guard. This measurement only decides whether that
work is worth starting.

# RESULT BNU · 2026-09-11 18:31:45 EDT — TRUE, at the bar

Zero compute; the three stored records through BND Stage 0's own loader and guards, split by the 2025 ADS-B coverage column.

| rows | net vs F (weighted fold MSE) | 95% day-block |
|---|---|---|
| all 339,015 | +556.8 | [+319.9, +793.1] |
| **uncovered (coverage < 2), 209,575 = 61.8%** | **+403.5** | **[+229.0, +580.2]** |
| covered (ADN acts), 129,440 | +153.3 | [+2.1, +303.4] |

**Against the shapes: TRUE** (net ≥ +400 and the lower bound > 0) — by 3.5 units, at the bar, not comfortably past it.

Uncovered net by airport: LIRF +140, EGLL +99, LTFM +99, LFPG +54, LEMD +30, EDDF +11, LEBL +10, EDDM −5, EHAM −7, LSZH −28.

**Reported, not decisional:**
- Crude 2026 scaling ×0.82 (uncovered share 61.8% → 50.8%) gives ≈ 331.
- Composition matters more than the crude factor: EGLL is 77% covered in 2026, so most of its +99 disappears, while LTFM (uncovered) and LIRF (80% uncovered) stay.
- The covered-row +153 is NOT additive with ADN: both correct the same F.

**What this decides:** only that a ship step is worth registering. It is not a ship. CAP needs a ≈2.5 h refit, C_delta a CatBoost
plug-in in its own venv (BC-5), and the step needs its own bar, build and A1-style guard. Expected value on the board ≈ 330,
which is about a fifth of the 1,709 to 4th, and not a route to 260 on its own.

**Correction (2026-09-11 18:4x, from prc-challenge-53):** I had told 53 that ADX would move rows out of the uncovered set and
so reduce BNU's value. That is wrong. ADX adds features to the stacker; the rows the stage moves stay exactly "coverage ≥ 2 at a
gated airport", and ADX changes neither coverage nor the gate. BNU's +403.5 on uncovered rows is unaffected by ADX, and the two
lanes stay close to independent. Only a coverage or gate change — a separate registration — could move the split.
