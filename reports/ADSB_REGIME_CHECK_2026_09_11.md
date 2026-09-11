# Can ADS-B see the early-off-block regime the gate cannot? — read-only check of prc-challenge-6e's table

## Part 1 — written BEFORE any number of this check · 2026-09-11 10:55:31 (from `date`), session prc-challenge-c4

Source (read-only, 6e's lane): `data/adsb/v2/features_2025janjul.parquet` (339,015 fold-A holdout rows keyed by `row`;
adsb.lol ODbL; coverage code 0–3, 3 = seen from the stand). Joined to arm REG's record (regime, gate p_early, ORACLE).
GID said the regime's value sits in 6,899 early rows (δ < −600) the gate cannot see (BASE +532 s, oracle +7,054).

Questions: (1) what share of those rows ADS-B covers (code ≥ 2, code 3); (2) among covered rows, how well the timing
features (first_ground_rel_aobt3, dwell_end_rel_aobt3, first_move_rel_mvt, dwell_end_rel_mvt) separate early from not
(AUC); (3) how much of the +7,054 oracle sits on covered rows (the ceiling ADS-B could unlock on 2025's coverage).
**Expected before looking:** coverage follows the airport table (EHAM / EDDM / LEBL / LSZH high; LIRF / LTFM / LFPG ≈ 0), so
most of the +7,054 — which sits at LIRF / LTFM / LFPG — is NOT covered; where covered, first_ground_rel_aobt3 < 0 should flag
early rows strongly (AUC > 0.85).

## Part 2 — result · 2026-09-11 10:56:13 (from `date`)

Joined row for row (339,015; y equal). Coverage codes over all rows: 1 = 209,575, 2 = 68,454, 3 = 60,986.

| early-off-block rows | n | ADS-B code ≥ 2 | code 3 | oracle (board MSE) | on code ≥ 2 | on code 3 |
|---|---|---|---|---|---|---|
| gate unsure (p_early < 0.5) | 6,899 | 36% | 15% | +7,054 | **+2,339** | +937 |
| gate confident | 5,855 | 44% | 26% | +696 | +282 | +115 |

Gate-unsure early rows by airport (n / code ≥ 2 / oracle): LTFM 1,063 / **0%** / +1,526 · LFPG 1,339 / **6%** / +1,389 · LIRF 1,512 /
52% / +940 · LEBL 671 / 75% / +820 · EDDF 433 / 57% / +612 · LEMD 512 / 29% / +530 · EGLL 440 / 33% / +479 · LSZH 496 / 70% / +370 ·
EDDM 412 / 55% / +273 · EHAM 21 / 57% / +114.

AUC early vs rest on covered rows (all | gate-unsure): `sensor_rel_aobt3` **0.947 | 0.899** (NaN on 68% of covered rows);
`first_ground_rel_aobt3` 0.761 | 0.695; `dwell_end_rel_aobt3` 0.625 | 0.522 (NaN 91%); `first_move_rel_mvt` 0.246 | 0.275 (i.e. 0.75
reversed). As a direct taxi measurement, stand-dwell-end → take-off is off by a median −1,684 s (MAD 464) on code-3 rows.

**Against the expectation:** as expected, most of the value sits where ADS-B is blind (LTFM, LFPG: +2,915 of +7,054 with ≤ 6%
coverage); LIRF is better covered than the census (52% mid-taxi, not 0%). `first_ground_rel_aobt3` flags early rows less
well than expected (0.76, not > 0.85); `sensor_rel_aobt3` flags them very well but exists on a third of covered rows.
**Ceiling for an ADS-B-fed regime gate on 2025 coverage: ≈ +2,300 board MSE (oracle on covered rows); realistic a fraction —
a few hundred to ≈ 1,000 — and only with the 2026 ADS-B fetch (≈ 4 h, prc-challenge-6e's lane).**

*Correction · 2026-09-11 11:22:17 (from `date`):* the closing estimate "realistic a fraction — a few hundred to ≈ 1,000" is SUPERSEDED. It priced only
the regime-gate use on early rows. prc-challenge-6e's ADN stacker, re-priced independently here, gains +5,146 [+4,720, +5,618] weighted fold
MSE over arm F across all covered rows (+1,431 of it on early-off-block rows) — see the ledger line of 2026-09-11 11:22:17.
