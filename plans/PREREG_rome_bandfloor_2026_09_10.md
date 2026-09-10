# Pre-registration — arm E1: the LIRF unmatched schedule floor (v8)

**Written 2026-09-10 14:45 local, BEFORE v8 is built and before any board reading of it exists.**
Tag `E1`. Owner decision 2026-09-10 14:4x: preregister, build v8 = v7 + E1 alone, confirm before upload.
Source: `reports/PATH_TO_245_FABLE_2026_09_10.md` §3–§5 (outside review), re-verified independently
in this session from `data/cache_rome/unm.parquet` and `submissions/merry-quicksand_v7.rome_rows.parquet`.

## Hypothesis

**H-E1:** a 2026 LIRF unmatched departure whose schedule offset `sp` lies in [24,000, 86,400) s has a
taxi-out `y ≥ sp − 60` (a schedule fill, `|y − sp| ≤ 60`, or a 24 h date-slip, `y ≥ 86,400`), as every
such row of 2025 did.

**Evidence, 2025 (all twelve months, LIRF unmatched, 1,488 rows):** 56 rows have `sp ≥ 24,000`; **every
one** has `y ≥ sp − 60` (fills and date-slips only; zero rows with `y` below `sp`). The largest `sp` with
`y < sp − 60` is **23,039** (August, y = 1,078). In [24,000, 56,000): 40 fills + 1 date-slip of 41,
across 24 airlines, both `dayoff` values, 17 schedule hours, 10 months. No other airport has a fill
with `sp ≥ 24,000` (0 of 109) — the rule is LIRF-only.

**Why v7 is below `sp` on two such rows.** R2's stake-weighted classifier (`p_r`) applies airline /
hour effects learned mostly at 2–4 h, where 2025 has real ordinary rows; out of sample (LOMO) it puts
`p < 0.5` on ~10% of high-sp fills (Fable §5.1). The two rows are VOE87GA (`sp` 39,365, `p_r` 0.089,
v7 4,524; VOE has 5 LIRF unmatched rows in all of 2025) and RYRR388G (`sp` 33,057, `p_r` 0.158, v7 6,196).

## The rule, fixed now (no fitted parameter)

For every scored row with `ADEP = LIRF`, unmatched, `24,000 ≤ sp < 86,400`:
`pred_v8 = max(pred_v7, rint(sp))`. Every other row is copied from v7 byte-for-byte.

**Dominance argument:** for both classes 2025 contains in this band (fill: `y ≈ sp`; date-slip:
`86,400 ≤ y < 90,000 > sp`), raising a prediction that is below `sp` up to `sp` cannot increase its
error by more than 60 s. The upper bound is where the argument ends (above 86,400 a date-slip lies
below `sp`); v7 has no row below `sp` in [56,000, 86,400), so the bound changes nothing today. The
lower bound sits just above the largest 2025 ordinary row (23,039) — a choice made on 2025 data,
tested only on 2026.

**What changes (computed before the build, from `v7.rome_rows.parquet`):** 13 rows. Two carry 5,612.5
of the 5,627.0 MSE at stake; the other eleven move by ≤ 1,779 s and carry 14.5 if fills.

## Predicted board shapes (Δ vs v7's 81,682.38 MSE; the board is the only instrument)

| outcome for the two rows | Δ board MSE | board RMSE |
|---|---:|---:|
| **TRUE — both fills** (2025 says 40/41) | **−5,627** | **275.78** |
| VOE fill, RYR ordinary (y ≈ 1,300) | −680 | 284.61 |
| RYR fill, VOE ordinary | +2,065 | 289.39 |
| **FALSE — both ordinary** (y ≈ 1,300) | **+7,012** | **297.82** |
| either a date-slip (y ≈ 87,000) | more negative than TRUE | — |

Expected under 2025's rate (40/41 non-ordinary, Wilson-95 lower bound ≈ 0.87 per row): ≈ −5,000.
Break-even per-row non-ordinary probability ≈ 0.56.

## Verdict rule, locked

- **WORKING** iff the board moves by ≤ −5,000 MSE vs v7 (both rows non-ordinary).
- **NOT WORKING** iff the board moves by ≥ +1,000 MSE (at least one row ordinary and the net is a loss).
- anything between: **PARTIAL** (exactly one of the two rows is non-ordinary), recorded as such.
If NOT WORKING or PARTIAL, v7 remains the best entry (the board ranks the lowest RMSE; nothing is lost
but a submission slot).

*Amendment E1.1 · 14:50, before upload and before any reading:* the parenthetical above contradicts
the MSE bands for the "RYR fill, VOE ordinary" outcome (+2,065). The verdict is decided by the board Δ
alone: **WORKING** ≤ −5,000; **NOT WORKING** > 0 (a net loss vs v7); **PARTIAL** otherwise. The
parentheticals are descriptions, not the rule.

## Pledge — against leaderboard probing

A v8 that changes two rows materially reads those two rows' classes from the board. That reading is
recorded here and **used for nothing else**: no other row's prediction is tuned on it, no band edge is
moved because of it, and no follow-up submission is built to read further rows. The rule above was
fixed from 2025 data before the reading; if it fails, the verdict stands and the rule is withdrawn,
not re-tuned.

## Limits

One board reading on effectively two rows: this closes H-E1 for those rows only. It says nothing
about the 2–6.7 h bands (where 2025 has real ordinary rows and the classifier's skill matters) and
nothing about EHAM's January 2026 disruption (Fable §3, E3 — a separate hypothesis).

## AMENDMENT E1.2 · 16:15 — ship on v9 as v10 (v8 never uploaded); before any reading of E1 exists

v8 was built but never uploaded. v9 (= v7 + E3C, board 282.6790) changed only non-LIRF unmatched rows; E1
changes only LIRF unmatched rows — **disjoint rows, so board MSE deltas add exactly**: Δ(v10 vs v9) in MSE is
identical to what Δ(v8 vs v7) would have been. E1 ships as **v10 = v9 + the same rule**, and every band above
is read as **Δ board MSE of v10 vs v9's 79,907.4** (TRUE ≈ −5,627 → ≈ 272.54; FALSE ≈ +7,012 → ≈ 294.84).
Verdict rule (E1.1), the rule itself and the pledge are unchanged. Guard 1 is run against v9 with v7's Rome
detail file (v9's 383 LIRF unmatched values are byte-identical to v7's, verified at v9's upload).

## BUILD · 2026-09-10 14:48 — v8 written, not uploaded

`scripts/rome_bandfloor.py --base submissions/merry-quicksand_v7.parquet --version 8` (exit 0;
tests `tests/test_rome_bandfloor.py` 6 passed, seven mutations rehearsed RED). Guard 1 passed (v7's
383 LIRF unmatched values equal `v7.rome_rows.after`; sp equals derive()'s). 23 rule rows, **13 changed,
all upward**, exactly the 13 precomputed above; if all are fills the board gain is **5,627.0 MSE**.
Independent diff (separate code): same ids/order/dtype as v7, 13 values differ, no nulls, min 1.
`submissions/merry-quicksand_v8.parquet` sha256 `7b1f6d83…a5176965`. Upload waits for the owner.

## BUILD · 2026-09-10 16:17 — v10 = v9 + E1 (Amendment E1.2), not yet uploaded

`scripts/rome_bandfloor.py --base submissions/merry-quicksand_v9.parquet --version 10` (exit 0). Guard 1 passed
against v9 with v7's Rome detail file. 23 rule rows, 13 changed (the same 13 as v8), fill gain if both rows are
fills 5,627.0 MSE. Independent check: v10 == v9 with v8's 13 E1 rows spliced in, bit-for-bit; E1 ∩ E3C rows = 0.
`submissions/merry-quicksand_v10.parquet` sha256 `18c911de…4e41b348`.

## BOARD · 2026-09-10 16:13 — v10 = **281.5182** (v9 282.6790): **−654.9 board MSE → PARTIAL** (Amendment E1.1/E1.2)

Uploaded 16:12:39 (HTTP 200) after on-disk re-verification (sha `18c911de…`, template, id order == v9, exactly
13 LIRF band rows raised, all else byte-identical to v9). Scorer: Succeeded, used_pairs 344,841. Rank 15 of 106.
**Verdict: PARTIAL** — Δ is between −5,000 and 0. It lies next to the pre-written "exactly one of the two rows
non-ordinary" shape (−680); H-E1 as a universal statement ("every 2026 LIRF unmatched row with sp in
[24,000, 86,400) has y ≥ sp − 60") is therefore **refuted for 2026** (at least one row violates it), while the
rule still netted a gain. v10 is the best entry (the board ranks the lowest RMSE).
**Pledge honoured:** this reading is recorded and used for nothing else — no row is tuned on it, no band edge
moves, no follow-up submission is built to read further rows; the rule is not re-tuned.
