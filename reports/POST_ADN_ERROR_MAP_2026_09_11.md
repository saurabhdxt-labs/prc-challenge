# Where the matched-lane error sits after ADN (descriptive; 2026-09-11 13:50:33 EDT, prc-challenge-70)

**What this is:** a decomposition of an EXISTING record, prc-challenge-6e's `data/adsb/v2/adn_fold_preds.parquet` (arm B, the
ADS-B blend, and arm F), on fold A's scored matched rows (334,132; the spent day 2025-01-09 excluded). No model was fitted and
nothing is claimed as an improvement, so there is no prereg. It exists so the next registration is chosen against where the
error actually is, not against where levers have been easy.

**Units:** contribution to the weighted fold MSE, 0.9846596 × SSE / 339,015, the convention of `reports/MSE_LEDGER.md`. Regimes
use the true delta δ = proxy − y:
- **early:** δ < −600 s. Off-block ≥ 10 min before `AOBT_3`.
- **late:** δ > +600 s.
- **agree:** everything else.

"Movable" = ADS-B coverage ≥ 2 (joined), the rows the stage can change. The coverage mix is 2025's; 2026's is not measured yet.

## 1. Totals

Arm F: 48,116. After ADN (B): **42,969** (gain 5,146, as RESULT ADN).

| regime | rows | after ADN | share | ADN's gain there |
|---|---|---|---|---|
| agree | 308,672 | 22,260 | 52% | 3,242 |
| **early** | **14,789 (4.4%)** | **18,057** | **42%** | 1,431 |
| late | 10,671 | 2,653 | 6% | 474 |

| | agree | early | late | total |
|---|---|---|---|---|
| NOT movable (ADS-B can't act) | 16,285 | 11,223 | 1,970 | **29,478 (69%)** |
| movable | 5,975 | 6,834 | 682 | 13,491 |

## 2. The largest cells after ADN

| airport · coverage | rows | after ADN | ADN gain | note |
|---|---|---|---|---|
| LTFM · not movable | 45,207 | 7,687 | 0 | 2025 coverage 0.00; early 2,738 (1,857 rows, RMSE 712 s) |
| LFPG · not movable | 36,142 | 6,255 | 0 | 2025 coverage 0.05; early 2,675 (1,705 rows, RMSE 735 s) |
| LIRF · movable | 11,552 | 6,139 | 414 | early 4,283 on 2,161 rows (RMSE 826 s): ADN barely moves them |
| EGLL · not movable | 27,512 | 4,174 | 0 | early 1,289 |
| LIRF · not movable | 14,222 | 4,110 | 0 | early 2,256 |
| LEMD · not movable | 29,879 | 2,525 | 0 | 2025 coverage 0.13 |

**Early rows by airport after ADN:** LIRF 6,539 (3,876 rows), LFPG 2,827, LTFM 2,738, LEBL 1,704, EGLL 1,606. The other five total 2,643.

## 3. What the early cells are (only what is already measured)

- **LIRF early rows:** 30% are schedule fills (|y − sp| ≤ 60 s). At LEBL it is 26%; at LFPG, LTFM and EGLL it is 4–8%.
  - The matched-lane fill class is CLOSED on current inputs. RESULT 8's fill head was NOT WORKING: AUC 0.90, but "no observable identifies fills per row". RESULT G's p-gated variant came to about +322.
  - ADS-B sees the aircraft, not the recording process. 53/6e's ADU note says the same.
- **LIRF's covered early rows keep RMSE 826 s after ADN.** Consistent with LIRF's coverage being mostly code 2 (joined mid-taxi, 0.44 movable vs 0.07 seen from the stand): the pushback itself is rarely observed there. This is a reading, not a measurement of cause.
- **LTFM / LFPG early rows** have no ADS-B in 2025. NMD (2026-09-11) closed the NM-table axes ("only new per-flight info can move it"), and GID found the early rows invisible to a regime gate built on the NM features.

## 4. What follows for the target (top 7 needs −7,229 from v10; ADN projects ≈ −5,300 at 2025 coverage)

1. **The next ~2–3k is not in another model on the same inputs.** 69% of the post-ADN error sits on rows no current observable reaches:
   - uncovered LTFM / LFPG / EGLL / LEMD;
   - the LIRF fill and pushback-unseen rows.
   Everything tried on those inputs is closed: fill head, p-gate, REG, SEL, NMD, local dayoff.
2. **The decision point is 2026 ADS-B coverage at LTFM and LFPG** (SHIP-ADN's gate report, ~15:20–15:30). 6e expected coverage growth at LEMD / LFPG / LIRF.
   - Newly covered airports ship under ADN.7 (transfer k = 0.595 in the projection).
   - LTFM + LFPG's uncovered rows hold 13,942 of the post-ADN error, their uncovered early rows alone 5,413. A covered share at 2025-EHAM levels would make them the largest remaining lever.
   - The gate report is the evidence; this map is not.
3. **LIRF's early rows (6,539) are the other large cell.** No registered arm reaches them. The parked candidate ("seen taxiing at all" as a fill-classifier input, LIRF unmatched coverage 0.63) applies here too.
   - Before any registration it needs a written TRUE/FALSE shape and must clear RESULT 8's bar: per-row identification, AUC well above 0.95.
   - On current evidence (AUC 0.90 without ADS-B) that bar is unlikely to be met. Named, not opened.

**Not covered here:** the unmatched lane (its own records; `reports/UNMATCHED_DIAGNOSTIC_2026_09_11.md`), 2026 coverage, and board
transfer. **Correction (2026-09-11 ~17:00): matched-lane fold gains have transferred at 0.96–1.17×** (v3 1.17, v5 0.97, v6 0.96;
`reports/MSE_LEDGER.md` board tables). The "0.4×" is only SHIP-ADN's conservative pass bar, not a measured ratio; 0.15× was the
Rome extreme-row lane. prc-challenge-70 misquoted 0.4× as history in chat on 2026-09-11.

## 5. Outcome of the decision point (added 2026-09-11 15:31 EDT, from SHIP-ADN's gate in `data/adsb/v2/models/MANIFEST.json`)

2026 joined ADS-B coverage (2025 → 2026):

| airport | 2025 → 2026 | gate |
|---|---|---|
| LTFM | 0.000 → 0.0001 (≈ 5 rows) | allowed, but nothing to act on |
| LFPG | 0.05 → 0.12 | allowed |
| LEMD | 0.13 → 0.65 | allowed |
| EGLL | 0.30 → 0.77 | allowed |
| LIRF | 0.45 → 0.20 | kept on F |
| EDDF, LEBL, LSZH | coverage holds | kept on F by the KS hold rule |

**The LTFM / LFPG lever of §4.2 is closed for ADS-B:** LTFM is unseen in 2026 and LFPG barely seen. Their 13,942 units of
uncovered fold error stay unreachable by any registered source.

53 has since found a design error in the KS hold: the same rule fires Jan-vs-Jul inside 2025, on the data the stacker validated on.
That rule is what keeps EDDF, LEBL and LSZH on F, three of 2025's strongest ADN airports. It is an owner decision, recorded in SHIP-ADN.

## 6. Where the reachable error sits, by HOW the flight was seen (added 2026-09-11 18:21 EDT; descriptive, fold A)

The owner's target moved to 260 (2nd is 260.93; −3,930 board MSE from v11). Splitting the ADS-B-covered rows after ADN:

| regime · coverage code | rows | after ADN | ADN's gain | RMSE after |
|---|---|---|---|---|
| agree · 2 (joined mid-taxi) | 63,345 | 4,086 | 1,355 | 149 s |
| agree · 3 (seen from the stand) | 55,550 | 1,889 | 1,887 | 108 s |
| **early · 2** | **3,186** | **4,967** | 542 | **733 s** |
| early · 3 | 2,941 | 1,867 | 888 | 468 s |
| late · 2 / 3 | 1,094 / 1,620 | 364 / 318 | 191 / 283 | 338 / 260 s |

**Code 2 carries 9,417 of the 13,491 units still reachable (70%), on 67,625 rows.**

**Mechanism, from the features themselves:** on early rows,
- code 3: first seen on the ground 803 s BEFORE AOBT_3, stand dwell ends +24 s, and ADN predicts −880 of a −962 s true delta (91%).
- code 2: first seen +162 s AFTER AOBT_3, no dwell at all, and ADN predicts −671 of −926 (72%).
The pushback is simply not observed on code-2 rows; the stacker is extrapolating from a mid-taxi sighting.

**Headroom (an upper bound, not a forecast):** bringing code-2 rows to code-3 accuracy would be worth ≈ 4,900 fold units
(early ≈ 2,930, agree ≈ 1,940). Parity is unlikely — the observation itself is missing — but a third of it is ≈ 1,600,
and 260 needs ≈ 3,300 from v12.

**The idea this suggests** (prc-challenge-53's lane, needs its own registration): reconstruct the unobserved start of taxi on
code-2 rows by working back from the first sighting — the stand's position, `path_m`, `taxi_med_gs`, `dwell_to_centroid_m` —
instead of leaving the stacker to extrapolate. Nothing here measures that; it is where the remaining reachable error is.
