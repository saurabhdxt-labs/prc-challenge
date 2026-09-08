# PRC Data Challenge 2026 - DATA DICTIONARY (30 columns)

Breadth sweep, facts only. No model was fitted; the only 'predictor' quantities are level-mean lookups and rank correlations, reported as measurements of column-target association.

## 0. Provenance

| item | value |
|---|---|
| generated | 2026-09-08, `s7_report.py` from `s0.json … s6.json` (scripts in Appendix A; every number below is read from those files) |
| interpreter | `/Users/saurabhdxt/Projects/Phantom/.venv/bin/python` 3.11.15, pyarrow 23.0.1, pandas 3.0.2, numpy 2.4.4 |
| run prefix | every script: `OMP_NUM_THREADS=1 nice -n 19 <python> <script>` |
| peak RSS per script (MB) | s0 804, s1 1006, s2 900, s3 892, s4 1445, s5 488, s6 788 (cap 4,096) |
| training input | 12 × `data/raw/training_2025-*.parquet`, 4,167,797 rows, PHASE DEP 2,085,047 / ARR 2,082,750 |
| admissibility filter | PHASE_mvt=="DEP" (−2,082,750) → TAXITIME/BLOCK/MVT non-null (−0) → TAXITIME>0 (−388) → drop dup MVT_ID keep first (−0) = **2,084,659 admissible departures** ("train_adm") |
| ranking.parquet | 689,534 rows (DEP 344,841, ARR 344,693) |
| scored rows | 344,841 = ranking rows whose MVT_ID is in submitting.parquet; identical to `PHASE==DEP & TAXITIME null` (True); submitting ids unique True, all present in ranking True, overlap with 2025 admissible ids 0 |
| scored by month | {'2026-01': 152719, '2026-07': 192122} ; MVT_TIME span 2026-01-01 00:01:55+00:00 → 2026-07-31 23:56:19+00:00 |
| unmatched stratum | `AOBT_3_flt.isna()` on scored rows = 5,290 rows = 1.534%; identical to 'all 16 _flt columns null' (True). 14 further rows are null ONLY in FLIGHT_RULE_flt+WK_TBL_CAT_flt (ADEP {'LSZH': 8, 'EHAM': 4, 'LEBL': 2}); 74 such rows in train_adm |
| drift comparison sets | A = train_adm 2025-01 + 2025-07 = 344,336 rows; B = 2026 scored = 344,841 rows |
| target-relationship set | train_adm, all 12 months, n = 2,084,659; held-out lookup = fit level means on Feb–Jun+Aug–Dec, score on Jan+Jul 2025 (n test = 344,336, the same Jan+Jul 2025 admissible set as drift set A); global-mean RMSE on that hold-out = 686.53 s |

**Drift flag rule (fixed before reading results):** categorical → FLAG if total-variation distance A↔B ≥ 0.05 or ≥ 1% of 2026 rows carry a level never seen in all of 2025; timestamp → FLAG if any of p10/p50/p90 of (column − MVT_TIME) moved by > 10% relative; ids → flagged by construction.

## 1. One-line summary of all 30 columns

| # | column | dtype | null% train raw | null% train_adm | null% scored | null% unmatched | card. adm / scored | USABLE AT SERVE TIME | 2025→2026 drift | association with target (train_adm) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `MVT_ID_mvt` | double | 0.000 | 0.000 | 0.000 | 0.00 | 2,084,659 / 344,841 | yes as a KEY only (value carries no signal: abs(Spearman) ≤ 0.015) | **FLAG** id range disjoint from 2025 (by construction; useless as a value) | Spearman +0.0109 |
| 2 | `FLIGHT_ID_mvt` | double | 0.742 | 1.064 | 1.532 | 99.87 | 2,062,483 / 339,557 | yes as a KEY only (value carries no signal: abs(Spearman) ≤ 0.015) | **FLAG** id range disjoint from 2025 (by construction; useless as a value) | Spearman +0.0145 |
| 3 | `FLIGHT_mvt` | string | 0.010 | 0.009 | 0.017 | 0.43 | 45,656 / 20,329 | yes (0.0168% null) | **FLAG** TVD 0.454; 21.1% of 2026 rows unseen in 2025 | held-out lookup gain +56.6 s RMSE (η² 0.374, 45,656 levels) |
| 4 | `FLIGHT_RULE_mvt` | string | 0.017 | 0.010 | 0.006 | 0.15 | 2 / 2 | yes (0.0058% null) | no (TVD 0.000) | held-out lookup gain +0.3 s RMSE (η² 0.002, 2 levels) |
| 5 | `ADEP_mvt` | string | 0.002 | 0.000 | 0.000 | 0.00 | 10 / 10 | yes (0.0000% null) | no (TVD 0.015) | held-out lookup gain +26.6 s RMSE (η² 0.112, 10 levels) |
| 6 | `ADES_mvt` | string | 0.003 | 0.006 | 0.004 | 0.11 | 1,560 / 1,105 | yes (0.0038% null) | no (TVD 0.036) | held-out lookup gain +13.6 s RMSE (η² 0.062, 1,560 levels) |
| 7 | `PHASE_mvt` | string | 0.000 | 0.000 | 0.000 | 0.00 | 1 / 1 | yes (0.0000% null) | no (constant) | held-out lookup gain +0.0 s RMSE (η² 0.000, 1 levels) |
| 8 | `MVT_TIME_UTC_mvt` | timestamp[us, tz=UTC] | 0.000 | 0.000 | 0.000 | 0.00 | 1,932,919 / 320,158 | yes (0.0000% null) | no (hour-of-day TVD 0.011) | hour-of-day lookup gain +2.0 s |
| 9 | `BLOCK_TIME_UTC_mvt` | timestamp[us, tz=UTC] | 0.000 | 0.000 | 100.000 | 100.00 | 1,848,933 / 0 | **NO** - 100% null on scored rows | n/a (null on 2026) | defines the target (MVT-BLOCK) |
| 10 | `SCHED_TIME_UTC_mvt` | timestamp[us, tz=UTC] | 0.000 | 0.000 | 0.000 | 0.00 | 99,108 / 16,987 | yes (0.0000% null) | no (offset-to-MVT p10/p50/p90 within 10%) | as offset MVT−SCHED: Spearman +0.387 |
| 11 | `AIRCRAFT_TYPE_mvt` | string | 0.040 | 0.072 | 0.113 | 7.33 | 268 / 199 | yes (0.1125% null) | **FLAG** TVD 0.066 | held-out lookup gain +15.6 s RMSE (η² 0.067, 268 levels) |
| 12 | `RUNWAY_mvt` | string | 0.000 | 0.000 | 0.000 | 0.00 | 53 / 50 | yes (0.0000% null) | **FLAG** TVD 0.102 | held-out lookup gain +30.4 s RMSE (η² 0.133, 53 levels) |
| 13 | `STAND_mvt` | string | 0.002 | 0.001 | 0.001 | 0.00 | 1,898 / 1,697 | yes (0.0012% null) | **FLAG** TVD 0.079 | held-out lookup gain +33.6 s RMSE (η² 0.142, 1,898 levels) |
| 14 | `TAXITIME_SEC_mvt` | int32 | 0.000 | 0.000 | 100.000 | 100.00 | 5,163 / 0 | **NO** - 100% null on scored rows | n/a (null on 2026) | is the target |
| 15 | `LOBT_flt` | timestamp[us, tz=UTC] | 0.910 | 1.066 | 1.534 | 100.00 | 186,888 / 32,779 | yes, except the unmatched stratum (1.534% null) | no (offset-to-MVT p10/p50/p90 within 10%) | as offset MVT−LOBT: Spearman +0.509 |
| 16 | `CALLSIGN_flt` | string | 0.910 | 1.066 | 1.534 | 100.00 | 44,807 / 20,375 | yes, except the unmatched stratum (1.534% null) | **FLAG** TVD 0.532; 24.8% of 2026 rows unseen in 2025 | held-out lookup gain +45.6 s RMSE (η² 0.214, 44,807 levels) |
| 17 | `ADEP_flt` | string | 0.910 | 1.066 | 1.534 | 100.00 | 10 / 10 | yes, except the unmatched stratum (1.534% null) | no (TVD 0.015) | held-out lookup gain +24.1 s RMSE (η² 0.105, 10 levels) |
| 18 | `ADES_flt` | string | 0.910 | 1.066 | 1.534 | 100.00 | 1,437 / 1,033 | yes, except the unmatched stratum (1.534% null) | no (TVD 0.036) | held-out lookup gain +12.9 s RMSE (η² 0.057, 1,437 levels) |
| 19 | `ADES_FILED_flt` | string | 0.910 | 1.066 | 1.534 | 100.00 | 1,433 / 1,031 | yes, except the unmatched stratum (1.534% null) | no (TVD 0.036) | held-out lookup gain +12.9 s RMSE (η² 0.057, 1,433 levels) |
| 20 | `MARKET_SEGMENT_flt` | string | 0.910 | 1.066 | 1.534 | 100.00 | 8 / 8 | yes, except the unmatched stratum (1.534% null) | no (TVD 0.015) | held-out lookup gain +5.9 s RMSE (η² 0.024, 8 levels) |
| 21 | `IOBT_flt` | timestamp[us, tz=UTC] | 0.910 | 1.066 | 1.534 | 100.00 | 178,155 / 31,284 | yes, except the unmatched stratum (1.534% null) | no (offset-to-MVT p10/p50/p90 within 10%) | as offset MVT−IOBT: Spearman +0.511 |
| 22 | `FLIGHT_RULE_flt` | string | 0.913 | 1.069 | 1.538 | 100.00 | 4 / 3 | yes, except the unmatched stratum (1.538% null) | no (TVD 0.000) | held-out lookup gain -0.1 s RMSE (η² 0.000, 4 levels) |
| 23 | `FLIGHT_TYPE_flt` | string | 0.910 | 1.066 | 1.534 | 100.00 | 5 / 5 | yes, except the unmatched stratum (1.534% null) | no (TVD 0.001) | held-out lookup gain +0.5 s RMSE (η² 0.002, 5 levels) |
| 24 | `AIRCRAFT_TYPE_flt` | string | 0.910 | 1.066 | 1.534 | 100.00 | 239 / 180 | yes, except the unmatched stratum (1.534% null) | **FLAG** TVD 0.065 | held-out lookup gain +15.1 s RMSE (η² 0.065, 239 levels) |
| 25 | `WK_TBL_CAT_flt` | string | 0.913 | 1.069 | 1.538 | 100.00 | 4 / 4 | yes, except the unmatched stratum (1.538% null) | no (TVD 0.004) | held-out lookup gain +8.4 s RMSE (η² 0.036, 4 levels) |
| 26 | `AIRCRAFT_OPERATOR_flt` | string | 0.910 | 1.066 | 1.534 | 100.00 | 676 / 579 | yes, except the unmatched stratum (1.534% null) | no (TVD 0.038) | held-out lookup gain +27.1 s RMSE (η² 0.115, 676 levels) |
| 27 | `EOBT_1_flt` | timestamp[us, tz=UTC] | 0.910 | 1.066 | 1.534 | 100.00 | 295,502 / 50,720 | yes, except the unmatched stratum (1.534% null) | no (offset-to-MVT p10/p50/p90 within 10%) | as offset MVT−EOBT: Spearman +0.568 |
| 28 | `ARVT_1_flt` | timestamp[us, tz=UTC] | 0.910 | 1.066 | 1.534 | 100.00 | 1,981,417 / 326,574 | yes, except the unmatched stratum (1.534% null) | no (offset-to-MVT p10/p50/p90 within 10%) | as offset MVT−ARVT: Spearman -0.258 |
| 29 | `AOBT_3_flt` | timestamp[us, tz=UTC] | 0.910 | 1.066 | 1.534 | 100.00 | 486,924 / 83,435 | yes, except the unmatched stratum (1.534% null) | no (offset-to-MVT p10/p50/p90 within 10%) | as offset MVT−AOBT: Spearman +0.615 |
| 30 | `ARVT_3_flt` | timestamp[us, tz=UTC] | 0.910 | 1.066 | 1.534 | 100.00 | 1,726,790 / 279,144 | yes, except the unmatched stratum (1.534% null) | no (offset-to-MVT p10/p50/p90 within 10%) | as offset MVT−ARVT: Spearman -0.297 |

Reading the association column: 'held-out lookup gain' = global-mean RMSE (686.53 s) minus the RMSE of predicting each row by its level's training mean (unseen level → global mean). It is a measurement of association in RMSE units, not a model; high-cardinality columns are inflated by memorisation and by low 2026 vocabulary overlap (see §3).

## 2. Per-column detail

### 2.1 `MVT_ID_mvt`

*Interpretation (from name and measured relations; no codebook was available):* movement identifier (airport movement record); submission key

| property | value |
|---|---|
| dtype | `double` |
| null % train raw / train_adm / scored / unmatched | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| cardinality train_adm / scored | 2,084,659 / 344,841 |
| usable at serve time | yes as a KEY only (value carries no signal: abs(Spearman) ≤ 0.015) |
| drift 2025 Jan+Jul → 2026 | **FLAG** id range disjoint from 2025 (by construction; useless as a value) |

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| train_adm | 2084659 | 181,979,889 | 182,023,925 | 183,420,890 | 191,612,277 | 199,843,671 | 201,977,042 | 202,048,515 | 202,050,610 | 191,693,758 |
| scored | 344841 | 202,180,234 | 202,183,779 | 202,321,853 | 212,248,241 | 213,727,531 | 213,805,554 | 213,808,707 | 213,809,059 | 208,476,233 |

Relationship to target: Pearson +0.0048, Spearman +0.0109. Decile means of target: 1002, 976, 978, 1007, 982, 974, 1008, 984, 1008, 995 (flat).

Checks: unique in train_adm True; unique on scored True; Spearman(MVT_ID, MVT_TIME) = 0.9929 (ids are assigned nearly in time order). 2026 ids start at 202,180,234 > 2025 max 202,050,610.

### 2.2 `FLIGHT_ID_mvt`

*Interpretation (from name and measured relations; no codebook was available):* flight identifier linking the DEP row at ADEP to the ARR row at ADES

| property | value |
|---|---|
| dtype | `double` |
| null % train raw / train_adm / scored / unmatched | 0.7421 / 1.0636 / 1.5320 / 99.8677 |
| cardinality train_adm / scored | 2,062,483 / 339,557 |
| usable at serve time | yes as a KEY only (value carries no signal: abs(Spearman) ≤ 0.015) |
| drift 2025 Jan+Jul → 2026 | **FLAG** id range disjoint from 2025 (by construction; useless as a value) |

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| train_adm | 2062486 | 280,351,437 | 280,510,452 | 281,841,508 | 287,172,378 | 293,219,147 | 294,431,357 | 294,558,376 | 294,567,590 | 287,461,006 |
| scored | 339558 | 294,567,549 | 294,597,057 | 294,831,598 | 301,570,958 | 302,672,205 | 302,938,484 | 302,955,372 | 302,958,437 | 299,045,274 |

Relationship to target: Pearson +0.0090, Spearman +0.0145. Decile means of target: 991, 980, 972, 981, 971, 1013, 988, 992, 989, 992 (flat).

Checks: duplicated FLIGHT_ID among admissible departures = 0.0001% of rows (a FLIGHT_ID pairs a DEP with an ARR row, not two DEPs). In the unmatched stratum it is 99.87% null, i.e. it is essentially a 17th NM-linked column.

### 2.3 `FLIGHT_mvt`

*Interpretation (from name and measured relations; no codebook was available):* flight designator as recorded by the airport (mixed IATA/ICAO forms, e.g. LH760, EK150, TAY9LA)

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.0098 / 0.0090 / 0.0168 / 0.4348 |
| cardinality train_adm / scored | 45,656 / 20,329 |
| usable at serve time | yes (0.0168% null) |
| drift 2025 Jan+Jul → 2026 | **FLAG** TVD 0.454; 21.1% of 2026 rows unseen in 2025 |
| string length min–max (train) / blank strings | [3, 8] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 72,744 (21.09%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | TAY9LA | 0.023 | LH760 | 0.018 |
| 2 | TAY3VQ | 0.018 | AIC2018 | 0.018 |
| 3 | EVA088 | 0.018 | CES554 | 0.018 |
| 4 | SIA335 | 0.018 | DAH1001 | 0.018 |
| 5 | ETH735 | 0.018 | DAH1013 | 0.018 |
| 6 | UAE65W | 0.018 | LH584 | 0.018 |
| 7 | ITY1733 | 0.018 | TK1592 | 0.018 |
| 8 | THY1VR | 0.018 | LH860 | 0.018 |
| 9 | GF016 | 0.018 | LH454 | 0.018 |
| 10 | UAE5T | 0.018 | SQ025 | 0.018 |

Top-5 within the unmatched stratum (scored): PGT1984 1.177%, PGT1982 1.177%, NCG01A 0.854%, ZXP24A 0.684%, ZXP26A 0.589%

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.4539; levels in B unseen in A: 39.195% of rows; unseen in all of 2025: 21.098% of rows (7373 levels); null% A/B 0.007/0.017.

| level | A % | B % | Δ pp |
|---|---|---|---|
| LH756 | 0.018 | 0.018 | -0.000 |
| SIA335 | 0.018 | 0.018 | -0.000 |
| SVA110 | 0.018 | 0.018 | -0.000 |
| QR068 | 0.018 | 0.018 | -0.000 |
| DAL83 | 0.018 | 0.018 | -0.000 |
| AIC2018 | 0.018 | 0.018 | +0.000 |
| LH760 | 0.018 | 0.018 | +0.000 |
| CES554 | 0.018 | 0.018 | +0.000 |
| WJA9 | 0.018 | 0.018 | -0.001 |
| ITY1625 | 0.018 | 0.018 | -0.000 |
| SEH671 | 0.018 | 0.018 | -0.000 |
| QTR096 | 0.018 | 0.018 | -0.000 |

Relationship to target (train_adm, 45,656 levels): η² (in-sample) = 0.3738; held-out level-mean lookup RMSE = 629.95 s (gain +56.58 s vs global mean 686.53); held-out rows with unseen level = 1.492%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| TAY9LA | 470 | 1164.6 | 1026 | 3215 |
| TAY3VQ | 367 | 1084.9 | 1029 | 2235 |
| EVA088 | 366 | 1351.7 | 1208 | 3141 |
| ITY1733 | 366 | 957.6 | 904 | 1939 |
| ETH735 | 366 | 1162.4 | 1085 | 2107 |
| THY1VR | 366 | 1083.3 | 972 | 2553 |
| SIA335 | 366 | 1334.8 | 1252 | 3024 |
| UAE65W | 366 | 1277.4 | 1209 | 2423 |
| SVA256 | 365 | 1228.2 | 1147 | 2407 |
| THY3PC | 365 | 1032.4 | 971 | 1882 |
| BAW81 | 365 | 1547.7 | 1494 | 2953 |
| THY350 | 365 | 1041.1 | 959 | 2644 |
| UAE6RM | 365 | 1291.0 | 1194 | 2806 |
| THY7 | 365 | 1195.3 | 1083 | 3071 |
| EK150 | 365 | 1365.6 | 1181 | 3358 |

Vocabulary turnover: 21.1% of 2026 scored rows carry a FLIGHT_mvt never seen in any 2025 admissible departure (matched stratum 78.9% seen, unmatched stratum 78.3% seen). FLIGHT_mvt equals CALLSIGN_flt on only 74.0% of matched scored rows (IATA-style vs ICAO-style designators).

### 2.4 `FLIGHT_RULE_mvt`

*Interpretation (from name and measured relations; no codebook was available):* flight rules per airport record (I/V)

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.0165 / 0.0097 / 0.0058 / 0.1512 |
| cardinality train_adm / scored | 2 / 2 |
| usable at serve time | yes (0.0058% null) |
| drift 2025 Jan+Jul → 2026 | no (TVD 0.000) |
| string length min–max (train) / blank strings | [1, 1] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 0 (0.00%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | I | 99.871 | I | 99.878 |
| 2 | V | 0.129 | V | 0.122 |

Top-5 within the unmatched stratum (scored): I 92.067%, V 7.933%

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0001; levels in B unseen in A: 0.000% of rows; unseen in all of 2025: 0.000% of rows (0 levels); null% A/B 0.029/0.006.

| level | A % | B % | Δ pp |
|---|---|---|---|
| I | 99.871 | 99.878 | +0.007 |
| V | 0.129 | 0.122 | -0.007 |

Relationship to target (train_adm, 2 levels): η² (in-sample) = 0.0015; held-out level-mean lookup RMSE = 686.18 s (gain +0.35 s vs global mean 686.53); held-out rows with unseen level = 0.029%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| I | 2,081,760 | 992.2 | 912 | 2340 |
| V | 2,696 | 398.2 | 244 | 1562 |

### 2.5 `ADEP_mvt`

*Interpretation (from name and measured relations; no codebook was available):* departure airport (ICAO); the ten challenge airports

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.0020 / 0.0000 / 0.0000 / 0.0000 |
| cardinality train_adm / scored | 10 / 10 |
| usable at serve time | yes (0.0000% null) |
| drift 2025 Jan+Jul → 2026 | no (TVD 0.015) |
| string length min–max (train) / blank strings | [4, 4] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 0 (0.00%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | LTFM | 13.094 | LTFM | 13.798 |
| 2 | EHAM | 11.894 | LFPG | 11.562 |
| 3 | EGLL | 11.491 | EGLL | 11.553 |
| 4 | LFPG | 11.488 | EHAM | 11.072 |
| 5 | EDDF | 11.04 | LEMD | 10.716 |
| 6 | LEMD | 10.181 | EDDF | 10.532 |
| 7 | LEBL | 8.62 | LEBL | 8.723 |
| 8 | EDDM | 8.027 | LIRF | 7.8 |
| 9 | LIRF | 7.709 | EDDM | 7.53 |
| 10 | LSZH | 6.456 | LSZH | 6.713 |

Top-5 within the unmatched stratum (scored): EHAM 22.798%, LFPG 13.875%, LTFM 13.629%, LSZH 12.457%, LEBL 8.355%

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0145; levels in B unseen in A: 0.000% of rows; unseen in all of 2025: 0.000% of rows (0 levels); null% A/B 0.000/0.000.

| level | A % | B % | Δ pp |
|---|---|---|---|
| LTFM | 13.516 | 13.798 | +0.282 |
| EGLL | 11.678 | 11.553 | -0.124 |
| LFPG | 11.497 | 11.562 | +0.065 |
| EHAM | 11.892 | 11.072 | -0.820 |
| EDDF | 10.696 | 10.532 | -0.164 |
| LEMD | 10.260 | 10.716 | +0.456 |
| LEBL | 8.418 | 8.723 | +0.305 |
| LIRF | 7.704 | 7.800 | +0.096 |
| EDDM | 7.868 | 7.530 | -0.338 |
| LSZH | 6.472 | 6.713 | +0.241 |

Relationship to target (train_adm, 10 levels): η² (in-sample) = 0.1119; held-out level-mean lookup RMSE = 659.91 s (gain +26.62 s vs global mean 686.53); held-out rows with unseen level = 0.000%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| LTFM | 272,963 | 1053.0 | 963 | 2587 |
| EHAM | 247,948 | 784.6 | 742 | 1775 |
| EGLL | 239,546 | 1364.2 | 1319 | 2701 |
| LFPG | 239,487 | 1019.8 | 954 | 2409 |
| EDDF | 230,141 | 863.5 | 837 | 1819 |
| LEMD | 212,242 | 1014.7 | 985 | 1932 |
| LEBL | 179,700 | 956.6 | 906 | 1962 |
| EDDM | 167,334 | 811.9 | 774 | 1921 |
| LIRF | 160,704 | 1194.6 | 1025 | 4019 |
| LSZH | 134,594 | 742.2 | 712 | 1708 |

Per-airport unmatched-stratum rate (A = 2025 Jan+Jul, B = 2026 scored):

| airport | rows A | rows B | unmatched % A | unmatched % B |
|---|---|---|---|---|
| EDDF | 36,830 | 36,317 | 1.626 | 1.159 |
| EDDM | 27,091 | 25,966 | 0.875 | 0.917 |
| EGLL | 40,210 | 39,840 | 1.159 | 0.622 |
| EHAM | 40,949 | 38,182 | 1.973 | 3.159 |
| LEBL | 28,985 | 30,080 | 1.711 | 1.469 |
| LEMD | 35,328 | 36,954 | 0.875 | 0.644 |
| LFPG | 39,589 | 39,872 | 2.213 | 1.841 |
| LIRF | 26,528 | 26,899 | 1.497 | 1.424 |
| LSZH | 22,287 | 23,150 | 2.136 | 2.847 |
| LTFM | 46,539 | 47,581 | 1.412 | 1.515 |

### 2.6 `ADES_mvt`

*Interpretation (from name and measured relations; no codebook was available):* destination airport (ICAO) per airport record

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.0029 / 0.0058 / 0.0038 / 0.1134 |
| cardinality train_adm / scored | 1,560 / 1,105 |
| usable at serve time | yes (0.0038% null) |
| drift 2025 Jan+Jul → 2026 | no (TVD 0.036) |
| string length min–max (train) / blank strings | [4, 4] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 176 (0.05%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | EGLL | 1.884 | EGLL | 1.863 |
| 2 | EHAM | 1.788 | EDDF | 1.682 |
| 3 | EDDF | 1.69 | LEMD | 1.645 |
| 4 | LFPG | 1.659 | LFPG | 1.615 |
| 5 | LEMD | 1.648 | EHAM | 1.59 |
| 6 | LEBL | 1.568 | LEBL | 1.497 |
| 7 | EDDM | 1.45 | EDDM | 1.403 |
| 8 | LPPT | 1.429 | LIRF | 1.383 |
| 9 | LIRF | 1.376 | LPPT | 1.381 |
| 10 | LEPA | 1.277 | LEPA | 1.297 |

Top-5 within the unmatched stratum (scored): LCEN 11.241%, EHAM 5.659%, LLBG 3.142%, LEPA 1.533%, DAAG 1.419%

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0363; levels in B unseen in A: 0.444% of rows; unseen in all of 2025: 0.051% of rows (64 levels); null% A/B 0.004/0.004.

| level | A % | B % | Δ pp |
|---|---|---|---|
| EGLL | 1.896 | 1.863 | -0.034 |
| EHAM | 1.770 | 1.590 | -0.179 |
| LEMD | 1.640 | 1.645 | +0.005 |
| EDDF | 1.599 | 1.682 | +0.082 |
| LFPG | 1.616 | 1.615 | -0.001 |
| LEBL | 1.537 | 1.497 | -0.040 |
| LPPT | 1.421 | 1.381 | -0.039 |
| EDDM | 1.395 | 1.403 | +0.008 |
| LIRF | 1.344 | 1.383 | +0.039 |
| LEPA | 1.290 | 1.297 | +0.007 |
| LSZH | 1.216 | 1.277 | +0.061 |
| EIDW | 1.189 | 1.205 | +0.016 |

Relationship to target (train_adm, 1,560 levels): η² (in-sample) = 0.0625; held-out level-mean lookup RMSE = 672.89 s (gain +13.63 s vs global mean 686.53); held-out rows with unseen level = 0.026%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| EGLL | 39,267 | 940.4 | 851 | 2392 |
| EHAM | 37,269 | 926.1 | 864 | 2215 |
| EDDF | 35,239 | 882.4 | 801 | 2029 |
| LFPG | 34,585 | 962.1 | 893 | 2178 |
| LEMD | 34,355 | 984.2 | 901 | 2281 |
| LEBL | 32,677 | 973.7 | 895 | 2329 |
| EDDM | 30,232 | 903.2 | 841 | 1999 |
| LPPT | 29,778 | 944.0 | 885 | 2098 |
| LIRF | 28,679 | 958.3 | 900 | 2050 |
| LEPA | 26,629 | 890.5 | 841 | 1862 |
| LSZH | 25,862 | 933.4 | 859 | 2154 |
| LOWW | 25,735 | 903.3 | 836 | 2042 |
| EIDW | 24,775 | 1016.3 | 960 | 2221 |
| EDDB | 24,509 | 833.8 | 770 | 1920 |
| LGAV | 23,738 | 989.8 | 903 | 2643 |

### 2.7 `PHASE_mvt`

*Interpretation (from name and measured relations; no codebook was available):* DEP or ARR; scored rows are all DEP

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| cardinality train_adm / scored | 1 / 1 |
| usable at serve time | yes (0.0000% null) |
| drift 2025 Jan+Jul → 2026 | no (constant) |
| string length min–max (train) / blank strings | [3, 3] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 0 (0.00%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | DEP | 100.0 | DEP | 100.0 |

Top-5 within the unmatched stratum (scored): DEP 100.0%

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0000; levels in B unseen in A: 0.000% of rows; unseen in all of 2025: 0.000% of rows (0 levels); null% A/B 0.000/0.000.

| level | A % | B % | Δ pp |
|---|---|---|---|
| DEP | 100.000 | 100.000 | +0.000 |

Relationship to target (train_adm, 1 levels): η² (in-sample) = 0.0000; held-out level-mean lookup RMSE = 686.53 s (gain +0.00 s vs global mean 686.53); held-out rows with unseen level = 0.000%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| DEP | 2,084,659 | 991.4 | 912 | 2339 |

### 2.8 `MVT_TIME_UTC_mvt`

*Interpretation (from name and measured relations; no codebook was available):* take-off time (DEP) - the minuend of the target; GIVEN on scored rows

| property | value |
|---|---|
| dtype | `timestamp[us, tz=UTC]` |
| null % train raw / train_adm / scored / unmatched | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| cardinality train_adm / scored | 1,932,919 / 320,158 |
| usable at serve time | yes (0.0000% null) |
| drift 2025 Jan+Jul → 2026 | no (hour-of-day TVD 0.011) |

Range and grid alignment (fraction of non-null values whose epoch-seconds are exact multiples of 60 / 300 / 3600):

| set | n | min | max | % whole-second | % on 60 s grid | % on 300 s grid | % on 3600 s grid |
|---|---|---|---|---|---|---|---|
| train_adm | 2,084,659 | 2025-01-01 00:05:56+00:00 | 2025-12-31 23:57:36+00:00 | 100.0 | 5.131 | 1.032 | 0.084 |
| scored | 344,841 | 2026-01-01 00:01:55+00:00 | 2026-07-31 23:56:19+00:00 | 100.0 | 5.148 | 1.042 | 0.083 |

Drift as calendar shares (A vs B): hour-of-day TVD = 0.0115; day-of-week A {'0': 13.113, '1': 14.293, '2': 15.932, '3': 16.274, '4': 14.897, '5': 12.436, '6': 13.055} vs B {'0': 12.99, '1': 12.419, '2': 14.355, '3': 16.371, '4': 16.874, '5': 13.801, '6': 13.19}; month split A {'1': 44.624, '7': 55.376} vs B {'1': 44.287, '7': 55.713}; rows/day A 5553.8 vs B 5562.0.

| hour UTC | A % | B % |
|---|---|---|
| 00 | 0.272 | 0.306 |
| 01 | 0.242 | 0.242 |
| 02 | 0.374 | 0.325 |
| 03 | 0.920 | 0.912 |
| 04 | 2.228 | 2.332 |
| 05 | 4.375 | 4.178 |
| 06 | 4.817 | 4.775 |
| 07 | 4.890 | 4.869 |
| 08 | 5.914 | 5.964 |
| 09 | 6.561 | 6.413 |
| 10 | 6.679 | 6.687 |
| 11 | 6.800 | 6.760 |
| 12 | 6.395 | 6.266 |
| 13 | 5.809 | 6.030 |
| 14 | 6.247 | 6.344 |
| 15 | 5.921 | 6.063 |
| 16 | 5.401 | 5.242 |
| 17 | 4.756 | 4.671 |
| 18 | 4.872 | 5.077 |
| 19 | 5.361 | 5.529 |
| 20 | 5.560 | 5.296 |
| 21 | 3.052 | 3.041 |
| 22 | 1.578 | 1.595 |
| 23 | 0.975 | 1.083 |

Target by take-off hour UTC (train_adm): η² = 0.0087, held-out lookup RMSE 684.55 (gain +1.98 s).

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| 0 | 5,008 | 1120.2 | 896 | 4263 |
| 1 | 4,505 | 1033.1 | 903 | 3775 |
| 2 | 7,424 | 1089.0 | 1025 | 2341 |
| 3 | 18,583 | 928.2 | 845 | 2213 |
| 4 | 49,774 | 894.1 | 842 | 2021 |
| 5 | 95,262 | 911.9 | 852 | 2092 |
| 6 | 98,322 | 904.5 | 840 | 2277 |
| 7 | 104,823 | 936.8 | 870 | 2386 |
| 8 | 125,445 | 992.3 | 925 | 2405 |
| 9 | 137,573 | 1052.7 | 971 | 2517 |
| 10 | 141,085 | 1021.0 | 954 | 2545 |
| 11 | 141,376 | 1029.0 | 964 | 2454 |
| 12 | 130,147 | 1040.5 | 969 | 2408 |
| 13 | 123,909 | 1014.7 | 957 | 2279 |
| 14 | 130,581 | 1003.4 | 923 | 2340 |
| 15 | 124,898 | 956.9 | 889 | 2177 |
| 16 | 112,045 | 985.4 | 904 | 2214 |
| 17 | 98,875 | 973.6 | 899 | 2169 |
| 18 | 103,244 | 961.3 | 899 | 2110 |
| 19 | 114,867 | 963.7 | 900 | 2111 |
| 20 | 111,610 | 983.7 | 912 | 2217 |
| 21 | 57,575 | 1076.3 | 1008 | 2627 |
| 22 | 28,994 | 1079.3 | 968 | 2717 |
| 23 | 18,734 | 1197.4 | 1072 | 2934 |

Target by day of week (0=Mon) (train_adm): η² = 0.0003, held-out lookup RMSE 686.45 (gain +0.08 s).

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| 0 | 300,836 | 995.1 | 919 | 2342 |
| 1 | 290,224 | 974.6 | 905 | 2219 |
| 2 | 299,235 | 985.8 | 909 | 2286 |
| 3 | 299,397 | 990.5 | 912 | 2287 |
| 4 | 307,732 | 1004.1 | 934 | 2398 |
| 5 | 285,728 | 983.6 | 907 | 2339 |
| 6 | 301,507 | 1004.7 | 924 | 2508 |

Target by month (train_adm): η² = 0.0007, held-out lookup RMSE 686.53 (gain +0.00 s).

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| 1 | 153,658 | 995.6 | 906 | 2589 |
| 2 | 143,694 | 1002.6 | 904 | 2891 |
| 3 | 164,421 | 972.2 | 905 | 2157 |
| 4 | 175,256 | 971.6 | 908 | 2098 |
| 5 | 185,181 | 986.7 | 925 | 2168 |
| 6 | 183,080 | 973.6 | 902 | 2223 |
| 7 | 190,678 | 1025.9 | 948 | 2635 |
| 8 | 191,137 | 994.0 | 929 | 2279 |
| 9 | 183,932 | 989.9 | 914 | 2225 |
| 10 | 185,654 | 992.1 | 925 | 2265 |
| 11 | 162,314 | 997.1 | 927 | 2320 |
| 12 | 165,654 | 995.2 | 911 | 2344 |

### 2.9 `BLOCK_TIME_UTC_mvt`

*Interpretation (from name and measured relations; no codebook was available):* off-block time (DEP) - the subtrahend of the target; HIDDEN on scored rows

| property | value |
|---|---|
| dtype | `timestamp[us, tz=UTC]` |
| null % train raw / train_adm / scored / unmatched | 0.0000 / 0.0000 / 100.0000 / 100.0000 |
| cardinality train_adm / scored | 1,848,933 / 0 |
| usable at serve time | **NO** - 100% null on scored rows |
| drift 2025 Jan+Jul → 2026 | n/a (null on 2026) |

Range and grid alignment (fraction of non-null values whose epoch-seconds are exact multiples of 60 / 300 / 3600):

| set | n | min | max | % whole-second | % on 60 s grid | % on 300 s grid | % on 3600 s grid |
|---|---|---|---|---|---|---|---|
| train_adm | 2,084,659 | 2024-12-31 23:38:00+00:00 | 2025-12-31 23:41:59+00:00 | 100.0 | 6.397 | 1.438 | 0.135 |
| scored | 0 | - | - | - | - | - | - |

Drift as offset to take-off, `BLOCK_TIME_UTC_mvt − MVT_TIME_UTC_mvt` in seconds (A vs B); % within ±60 s of zero: A 0.045 / B None:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 344336 | -88,132 | -2,629 | -1,505 | -923 | -569 | -293 | -124 | -1 | -1,012 |
| B | - | - | - | - | - | - | - | - | - | - |

Drift as offset to schedule, `BLOCK_TIME_UTC_mvt − SCHED_TIME_UTC_mvt` (s); % within ±60 s of zero: A 8.834 / B None:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 344336 | -82,500 | -723 | -237 | 481 | 2,833 | 9,183 | 22,203 | 269,095 | 1,072 |
| B | - | - | - | - | - | - | - | - | - | - |

Relationship to target via `MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt` (train_adm): Pearson +1.0000, Spearman +1.0000; equals the target exactly on 100.000% of rows; within ±60 s of zero on 0.035%. Decile table in §4.

### 2.10 `SCHED_TIME_UTC_mvt`

*Interpretation (from name and measured relations; no codebook was available):* scheduled off-block time per airport record (5-minute grid)

| property | value |
|---|---|
| dtype | `timestamp[us, tz=UTC]` |
| null % train raw / train_adm / scored / unmatched | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| cardinality train_adm / scored | 99,108 / 16,987 |
| usable at serve time | yes (0.0000% null) |
| drift 2025 Jan+Jul → 2026 | no (offset-to-MVT p10/p50/p90 within 10%) |

Range and grid alignment (fraction of non-null values whose epoch-seconds are exact multiples of 60 / 300 / 3600):

| set | n | min | max | % whole-second | % on 60 s grid | % on 300 s grid | % on 3600 s grid |
|---|---|---|---|---|---|---|---|
| train_adm | 2,084,659 | 2024-12-31 22:40:00+00:00 | 2025-12-31 23:40:00+00:00 | 100.0 | 100.000 | 99.975 | 12.667 |
| scored | 344,841 | 2025-12-31 09:35:00+00:00 | 2026-07-31 23:00:00+00:00 | 100.0 | 100.000 | 99.957 | 12.827 |

Drift as offset to take-off, `SCHED_TIME_UTC_mvt − MVT_TIME_UTC_mvt` in seconds (A vs B); % within ±60 s of zero: A 0.432 / B 0.343:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 344336 | -270,508 | -10,379 | -4,014 | -1,503 | -605 | -58 | 1,004 | 66,358 | -2,084 |
| B | 344841 | -174,412 | -11,344 | -4,035 | -1,501 | -653 | -101 | 848 | 85,810 | -2,125 |

Target by scheduled hour UTC (train_adm): η² = 0.0062, held-out lookup gain +1.29 s. Full per-hour table omitted (shape matches take-off hour, §2.8).

Relationship to target via `MVT_TIME_UTC_mvt − SCHED_TIME_UTC_mvt` (train_adm): Pearson +0.2320, Spearman +0.3866; equals the target exactly on 0.390% of rows; within ±60 s of zero on 0.443%. Decile table in §4.

### 2.11 `AIRCRAFT_TYPE_mvt`

*Interpretation (from name and measured relations; no codebook was available):* ICAO aircraft type per airport record

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.0396 / 0.0721 / 0.1125 / 7.3346 |
| cardinality train_adm / scored | 268 / 199 |
| usable at serve time | yes (0.1125% null) |
| drift 2025 Jan+Jul → 2026 | **FLAG** TVD 0.066 |
| string length min–max (train) / blank strings | [3, 4] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 18 (0.01%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | A320 | 16.464 | A320 | 15.394 |
| 2 | B738 | 10.488 | B738 | 9.702 |
| 3 | A20N | 8.15 | A21N | 9.254 |
| 4 | A321 | 7.788 | A20N | 8.81 |
| 5 | A21N | 7.164 | A321 | 7.026 |
| 6 | A319 | 6.465 | A319 | 5.494 |
| 7 | E190 | 3.785 | BCS3 | 4.628 |
| 8 | BCS3 | 3.711 | B789 | 3.417 |
| 9 | B77W | 3.226 | E190 | 3.293 |
| 10 | B789 | 2.944 | B77W | 3.187 |

Top-5 within the unmatched stratum (scored): B738 12.424%, A320 12.036%, A20N 7.364%, A21N 6.65%, E190 5.059%

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0661; levels in B unseen in A: 0.017% of rows; unseen in all of 2025: 0.005% of rows (8 levels); null% A/B 0.117/0.113.

| level | A % | B % | Δ pp |
|---|---|---|---|
| A320 | 16.530 | 15.394 | -1.136 |
| B738 | 10.631 | 9.702 | -0.930 |
| A20N | 7.913 | 8.810 | +0.897 |
| A21N | 7.016 | 9.254 | +2.237 |
| A321 | 7.725 | 7.026 | -0.699 |
| A319 | 6.332 | 5.494 | -0.838 |
| BCS3 | 3.447 | 4.628 | +1.181 |
| E190 | 3.720 | 3.293 | -0.427 |
| B77W | 3.394 | 3.187 | -0.207 |
| B789 | 2.992 | 3.417 | +0.425 |
| A359 | 2.787 | 3.040 | +0.252 |
| B38M | 2.238 | 2.714 | +0.477 |

Relationship to target (train_adm, 268 levels): η² (in-sample) = 0.0673; held-out level-mean lookup RMSE = 670.91 s (gain +15.62 s vs global mean 686.53); held-out rows with unseen level = 0.117%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| A320 | 342,965 | 932.9 | 844 | 2225 |
| B738 | 218,487 | 976.3 | 943 | 2218 |
| A20N | 169,767 | 1016.2 | 948 | 2272 |
| A321 | 162,231 | 913.3 | 844 | 2163 |
| A21N | 149,244 | 1065.4 | 1004 | 2577 |
| A319 | 134,675 | 939.3 | 850 | 2173 |
| E190 | 78,855 | 805.8 | 759 | 1863 |
| BCS3 | 77,316 | 942.5 | 896 | 2093 |
| B77W | 67,203 | 1199.2 | 1133 | 2592 |
| B789 | 61,322 | 1219.7 | 1152 | 2526 |
| A359 | 56,390 | 1129.0 | 1030 | 2576 |
| A333 | 47,159 | 1106.9 | 1018 | 2698 |
| B38M | 45,246 | 1143.7 | 1099 | 2343 |
| B772 | 33,999 | 1268.9 | 1208 | 2703 |
| A332 | 31,976 | 1094.1 | 1016 | 2643 |

### 2.12 `RUNWAY_mvt`

*Interpretation (from name and measured relations; no codebook was available):* departure runway

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| cardinality train_adm / scored | 53 / 50 |
| usable at serve time | yes (0.0000% null) |
| drift 2025 Jan+Jul → 2026 | **FLAG** TVD 0.102 |
| string length min–max (train) / blank strings | [1, 3] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 0 (0.00%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | 18 | 8.236 | 18 | 8.79 |
| 2 | 25 | 6.943 | 36L | 7.137 |
| 3 | 36L | 6.85 | 25 | 7.018 |
| 4 | 36 | 6.052 | 09R | 6.919 |
| 5 | 24L | 5.973 | 27L | 6.226 |
| 6 | 09R | 5.739 | 24L | 5.722 |
| 7 | 26R | 5.329 | 26R | 5.562 |
| 8 | 08L | 5.2 | 36 | 5.438 |
| 9 | 27L | 4.878 | 08L | 4.341 |
| 10 | 27R | 4.727 | 28 | 4.306 |

Top-5 within the unmatched stratum (scored): 24 8.941%, 18 7.448%, 28 6.805%, 25 6.465%, 36 5.917%

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.1020; levels in B unseen in A: 0.006% of rows; unseen in all of 2025: 0.000% of rows (0 levels); null% A/B 0.000/0.000.

| level | A % | B % | Δ pp |
|---|---|---|---|
| 18 | 8.645 | 8.790 | +0.145 |
| 36L | 6.560 | 7.137 | +0.577 |
| 25 | 6.610 | 7.018 | +0.408 |
| 27L | 7.086 | 6.226 | -0.860 |
| 26R | 6.955 | 5.562 | -1.393 |
| 24L | 6.473 | 5.722 | -0.752 |
| 36 | 6.004 | 5.438 | -0.566 |
| 09R | 3.227 | 6.919 | +3.693 |
| 27R | 6.007 | 3.264 | -2.743 |
| 28 | 3.848 | 4.306 | +0.459 |
| 35L | 4.086 | 3.987 | -0.100 |
| 36R | 3.715 | 3.826 | +0.111 |

Relationship to target (train_adm, 53 levels): η² (in-sample) = 0.1328; held-out level-mean lookup RMSE = 656.15 s (gain +30.38 s vs global mean 686.53); held-out rows with unseen level = 0.000%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| 18 | 171,683 | 931.4 | 914 | 1871 |
| 25 | 144,747 | 1159.2 | 1013 | 3967 |
| 36L | 142,793 | 999.1 | 972 | 1905 |
| 36 | 126,155 | 928.2 | 891 | 1929 |
| 24L | 124,512 | 960.3 | 914 | 1885 |
| 09R | 119,638 | 1268.5 | 1209 | 2577 |
| 26R | 111,099 | 935.1 | 845 | 2230 |
| 08L | 108,412 | 925.2 | 844 | 2226 |
| 27L | 101,681 | 1261.5 | 1203 | 2711 |
| 27R | 98,542 | 1335.5 | 1269 | 2751 |
| 24 | 85,236 | 666.4 | 627 | 1487 |
| 36R | 84,705 | 1078.4 | 1050 | 1954 |
| 28 | 84,692 | 691.3 | 673 | 1572 |
| 35L | 79,873 | 1091.4 | 1011 | 2706 |
| 26L | 58,033 | 763.1 | 720 | 1913 |

Sentinel: the literal string `"NA"` occurs as a runway value (not a null) on 36 train_adm rows {'LIRF': 24, 'EDDF': 8, 'EGLL': 3, 'EHAM': 1} and 4 scored rows {'EDDF': 2, 'LFPG': 2}. Per-airport runway drift in §3.2.

### 2.13 `STAND_mvt`

*Interpretation (from name and measured relations; no codebook was available):* stand / gate label (labels are NOT globally unique - e.g. '224' exists at several airports; key on (ADEP, STAND))

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.0015 / 0.0010 / 0.0012 / 0.0000 |
| cardinality train_adm / scored | 1,898 / 1,697 |
| usable at serve time | yes (0.0012% null) |
| drift 2025 Jan+Jul → 2026 | **FLAG** TVD 0.079 |
| string length min–max (train) / blank strings | [1, 7] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 284 (0.08%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | 224 | 0.428 | 224 | 0.441 |
| 2 | 311 | 0.41 | 302 | 0.435 |
| 3 | 302 | 0.406 | 312 | 0.435 |
| 4 | 309 | 0.405 | 309 | 0.434 |
| 5 | 312 | 0.403 | 311 | 0.431 |
| 6 | 301 | 0.381 | 301 | 0.421 |
| 7 | 310 | 0.367 | 310 | 0.395 |
| 8 | A11 | 0.354 | 226 | 0.358 |
| 9 | 313 | 0.343 | 217 | 0.357 |
| 10 | 308 | 0.337 | 308 | 0.349 |

Top-5 within the unmatched stratum (scored): HG01 2.798%, MFG 1.777%, HPW 1.096%, HG03 1.078%, K76 0.548%

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0790; levels in B unseen in A: 0.472% of rows; unseen in all of 2025: 0.082% of rows (30 levels); null% A/B 0.002/0.001.

| level | A % | B % | Δ pp |
|---|---|---|---|
| 224 | 0.425 | 0.441 | +0.016 |
| 302 | 0.405 | 0.435 | +0.031 |
| 309 | 0.405 | 0.434 | +0.028 |
| 312 | 0.396 | 0.435 | +0.039 |
| 311 | 0.399 | 0.431 | +0.032 |
| 301 | 0.361 | 0.421 | +0.060 |
| 310 | 0.356 | 0.395 | +0.039 |
| 217 | 0.333 | 0.357 | +0.024 |
| A11 | 0.354 | 0.333 | -0.021 |
| 226 | 0.329 | 0.358 | +0.029 |
| 313 | 0.339 | 0.341 | +0.003 |
| 308 | 0.321 | 0.349 | +0.027 |

Relationship to target (train_adm, 1,898 levels): η² (in-sample) = 0.1418; held-out level-mean lookup RMSE = 652.94 s (gain +33.59 s vs global mean 686.53); held-out rows with unseen level = 0.036%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| 224 | 8,913 | 950.5 | 857 | 2100 |
| 311 | 8,541 | 1084.7 | 966 | 2944 |
| 302 | 8,469 | 960.5 | 891 | 2285 |
| 309 | 8,448 | 1067.8 | 963 | 2638 |
| 312 | 8,391 | 980.4 | 904 | 2414 |
| 301 | 7,937 | 1049.8 | 959 | 2335 |
| 310 | 7,652 | 954.8 | 855 | 2514 |
| A11 | 7,390 | 839.8 | 763 | 1989 |
| 313 | 7,140 | 1076.1 | 968 | 2454 |
| 308 | 7,032 | 987.5 | 893 | 2461 |
| 217 | 6,928 | 1033.8 | 983 | 2041 |
| 226 | 6,850 | 988.0 | 922 | 1930 |
| 501 | 6,759 | 1223.1 | 1134 | 2944 |
| D12 | 6,727 | 931.2 | 891 | 2163 |
| 502 | 6,423 | 1221.4 | 1145 | 2933 |

Per-airport stand vocabulary coverage (labels are reused across airports, so the global 'unseen' figure above understates the real gap):

| airport | stands 2025 Jan+Jul | stands all 2025 | stands 2026 | 2026 rows unseen vs Jan+Jul % | 2026 rows unseen vs all 2025 % |
|---|---|---|---|---|---|
| EDDF | 223 | 246 | 238 | 4.645 | 0.826 |
| EDDM | 176 | 203 | 186 | 4.922 | 4.552 |
| EGLL | 212 | 231 | 207 | 0.018 | 0.000 |
| EHAM | 211 | 261 | 202 | 0.037 | 0.029 |
| LEBL | 218 | 232 | 215 | 0.013 | 0.003 |
| LEMD | 351 | 374 | 369 | 0.165 | 0.038 |
| LFPG | 374 | 408 | 361 | 0.068 | 0.020 |
| LIRF | 133 | 141 | 133 | 0.349 | 0.033 |
| LSZH | 178 | 207 | 184 | 5.028 | 0.251 |
| LTFM | 352 | 416 | 357 | 0.502 | 0.013 |

### 2.14 `TAXITIME_SEC_mvt`

*Interpretation (from name and measured relations; no codebook was available):* TARGET = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt in seconds (verified equal on 100% of admissible rows, see s4 pair 1)

| property | value |
|---|---|
| dtype | `int32` |
| null % train raw / train_adm / scored / unmatched | 0.0000 / 0.0000 / 100.0000 / 100.0000 |
| cardinality train_adm / scored | 5,163 / 0 |
| usable at serve time | **NO** - 100% null on scored rows |
| drift 2025 Jan+Jul → 2026 | n/a (null on 2026) |

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| train_adm | 2084659 | 1 | 293 | 571 | 912 | 1,471 | 2,339 | 4,501 | 131,167 | 991 |
| scored | - | - | - | - | - | - | - | - | - | - |

Target facts: 2,084,659 admissible rows, mean 991.4, median 912, p99 2339, p99.9 4501, max 131167 s (= 36.4 h). Rows > 3600 s: matched stratum 0.147% vs unmatched stratum 4.892%. The unmatched stratum (22,219 rows = 1.07% of train_adm) carries **42.2% of the total squared error around the global mean**.

### 2.15 `LOBT_flt`

*Interpretation (from name and measured relations; no codebook was available):* Network-Manager flight-plan field; name suggests 'last off-block time'. Equals IOBT_flt on 98.4% of rows

| property | value |
|---|---|
| dtype | `timestamp[us, tz=UTC]` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 186,888 / 32,779 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | no (offset-to-MVT p10/p50/p90 within 10%) |

Range and grid alignment (fraction of non-null values whose epoch-seconds are exact multiples of 60 / 300 / 3600):

| set | n | min | max | % whole-second | % on 60 s grid | % on 300 s grid | % on 3600 s grid |
|---|---|---|---|---|---|---|---|
| train_adm | 2,062,440 | 2025-01-01 00:20:00+00:00 | 2025-12-31 23:40:00+00:00 | 100.0 | 99.692 | 94.949 | 11.950 |
| scored | 339,551 | 2026-01-01 00:04:00+00:00 | 2026-07-31 23:30:00+00:00 | 100.0 | 99.770 | 94.348 | 12.114 |

Drift as offset to take-off, `LOBT_flt − MVT_TIME_UTC_mvt` in seconds (A vs B); % within ±60 s of zero: A 0.487 / B 0.432:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 339015 | -87,001 | -4,139 | -2,465 | -1,276 | -590 | -50 | 621 | 3,382 | -1,431 |
| B | 339551 | -86,820 | -4,138 | -2,404 | -1,263 | -597 | -62 | 530 | 3,173 | -1,414 |

Drift as offset to schedule, `LOBT_flt − SCHED_TIME_UTC_mvt` (s); % within ±60 s of zero: A 70.996 / B 71.43:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 339015 | -66,300 | -600 | 0 | 0 | 1,800 | 8,100 | 21,000 | 268,800 | 591 |
| B | 339551 | -86,400 | -600 | 0 | 0 | 1,860 | 8,700 | 23,100 | 172,800 | 640 |

Relationship to target via `MVT_TIME_UTC_mvt − LOBT_flt` (train_adm): Pearson +0.5141, Spearman +0.5086; equals the target exactly on 0.435% of rows; within ±60 s of zero on 0.485%. Decile table in §4.

### 2.16 `CALLSIGN_flt`

*Interpretation (from name and measured relations; no codebook was available):* ICAO callsign from NM flight plan (DLH584 vs FLIGHT_mvt LH584); equals FLIGHT_mvt on only 74.0% of matched scored rows

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 44,807 / 20,375 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | **FLAG** TVD 0.532; 24.8% of 2026 rows unseen in 2025 |
| string length min–max (train) / blank strings | [3, 7] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 84,232 (24.43%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | TAY9LA | 0.025 | SWR154 | 0.019 |
| 2 | TAY3VQ | 0.018 | DLH584 | 0.018 |
| 3 | THY1VR | 0.018 | FIN1LP | 0.018 |
| 4 | AEA6037 | 0.018 | BAW911U | 0.018 |
| 5 | ANE20JV | 0.018 | DLH2HW | 0.018 |
| 6 | IBB40MW | 0.018 | DLH2RX | 0.018 |
| 7 | AFR356 | 0.018 | CTN413 | 0.018 |
| 8 | UAE96 | 0.018 | TAM8071 | 0.018 |
| 9 | ITY1783 | 0.018 | DLH756 | 0.018 |
| 10 | ITY2130 | 0.018 | UAE13U | 0.018 |

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.5323; levels in B unseen in A: 46.909% of rows; unseen in all of 2025: 24.807% of rows (7808 levels); null% A/B 1.545/1.534.

| level | A % | B % | Δ pp |
|---|---|---|---|
| TAY9LA | 0.027 | 0.012 | -0.015 |
| SWR154 | 0.018 | 0.019 | +0.000 |
| ITY1603 | 0.018 | 0.018 | -0.000 |
| MSR792 | 0.018 | 0.018 | -0.000 |
| THY6NP | 0.018 | 0.018 | -0.000 |
| ITY1473 | 0.018 | 0.018 | -0.000 |
| ITY1467 | 0.018 | 0.018 | -0.000 |
| ETD51M | 0.018 | 0.018 | -0.000 |
| ITY1431 | 0.018 | 0.018 | -0.000 |
| ETH552 | 0.018 | 0.018 | -0.000 |
| EVA068 | 0.018 | 0.018 | -0.000 |
| ANE70DJ | 0.018 | 0.018 | -0.000 |

Relationship to target (train_adm, 44,807 levels): η² (in-sample) = 0.2145; held-out level-mean lookup RMSE = 640.88 s (gain +45.65 s vs global mean 686.53); held-out rows with unseen level = 2.961%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| TAY9LA | 511 | 1146.6 | 1020 | 3175 |
| TAY3VQ | 366 | 1085.2 | 1029 | 2236 |
| THY1VR | 366 | 1083.3 | 972 | 2553 |
| IBB40MW | 365 | 1119.0 | 1089 | 1882 |
| THY3PC | 365 | 1032.4 | 971 | 1882 |
| AEA6037 | 365 | 770.1 | 739 | 1462 |
| UAE96 | 365 | 1189.7 | 1138 | 1947 |
| AFR356 | 365 | 1254.2 | 1251 | 2518 |
| THY59E | 365 | 1045.1 | 970 | 2214 |
| ANE20JV | 365 | 976.1 | 995 | 1761 |
| ITY1571 | 365 | 941.1 | 900 | 2005 |
| THY370 | 365 | 991.4 | 912 | 2365 |
| ITY1589 | 365 | 972.9 | 912 | 2054 |
| EDW402 | 365 | 741.2 | 676 | 1458 |
| ITY1785 | 365 | 1129.2 | 1029 | 2443 |

### 2.17 `ADEP_flt`

*Interpretation (from name and measured relations; no codebook was available):* departure airport from NM flight plan; equals ADEP_mvt on 100% of matched scored rows

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 10 / 10 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | no (TVD 0.015) |
| string length min–max (train) / blank strings | [4, 4] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 0 (0.00%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | LTFM | 13.058 | LTFM | 13.801 |
| 2 | EHAM | 11.824 | EGLL | 11.66 |
| 3 | EGLL | 11.546 | LFPG | 11.526 |
| 4 | LFPG | 11.429 | EHAM | 10.89 |
| 5 | EDDF | 11.064 | LEMD | 10.813 |
| 6 | LEMD | 10.245 | EDDF | 10.572 |
| 7 | LEBL | 8.623 | LEBL | 8.729 |
| 8 | EDDM | 8.064 | LIRF | 7.809 |
| 9 | LIRF | 7.72 | EDDM | 7.577 |
| 10 | LSZH | 6.426 | LSZH | 6.624 |

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0147; levels in B unseen in A: 0.000% of rows; unseen in all of 2025: 0.000% of rows (0 levels); null% A/B 1.545/1.534.

| level | A % | B % | Δ pp |
|---|---|---|---|
| LTFM | 13.534 | 13.801 | +0.267 |
| EGLL | 11.723 | 11.660 | -0.063 |
| LFPG | 11.419 | 11.526 | +0.107 |
| EHAM | 11.840 | 10.890 | -0.951 |
| EDDF | 10.687 | 10.572 | -0.116 |
| LEMD | 10.330 | 10.813 | +0.483 |
| LEBL | 8.403 | 8.729 | +0.325 |
| LIRF | 7.708 | 7.809 | +0.101 |
| EDDM | 7.921 | 7.577 | -0.344 |
| LSZH | 6.434 | 6.624 | +0.190 |

Relationship to target (train_adm, 10 levels): η² (in-sample) = 0.1055; held-out level-mean lookup RMSE = 662.48 s (gain +24.05 s vs global mean 686.53); held-out rows with unseen level = 1.545%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| LTFM | 269,320 | 1049.6 | 963 | 2568 |
| EHAM | 243,865 | 787.7 | 744 | 1767 |
| EGLL | 238,132 | 1363.0 | 1319 | 2697 |
| LFPG | 235,725 | 1017.1 | 954 | 2402 |
| EDDF | 228,185 | 862.5 | 836 | 1816 |
| LEMD | 211,295 | 1014.3 | 984 | 1932 |
| LEBL | 177,838 | 955.7 | 905 | 1957 |
| EDDM | 166,324 | 810.6 | 774 | 1916 |
| LIRF | 159,216 | 1144.7 | 1024 | 3299 |
| LSZH | 132,540 | 742.0 | 712 | 1705 |

### 2.18 `ADES_flt`

*Interpretation (from name and measured relations; no codebook was available):* destination from NM flight plan; equals ADES_mvt on 98.6% of matched scored rows

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 1,437 / 1,033 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | no (TVD 0.036) |
| string length min–max (train) / blank strings | [4, 4] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 154 (0.04%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | EGLL | 1.892 | EGLL | 1.877 |
| 2 | EHAM | 1.722 | EDDF | 1.717 |
| 3 | EDDF | 1.72 | LEMD | 1.67 |
| 4 | LEMD | 1.674 | LFPG | 1.624 |
| 5 | LFPG | 1.668 | EHAM | 1.531 |
| 6 | LEBL | 1.567 | LEBL | 1.501 |
| 7 | EDDM | 1.467 | EDDM | 1.417 |
| 8 | LPPT | 1.442 | LPPT | 1.405 |
| 9 | LIRF | 1.395 | LIRF | 1.402 |
| 10 | EDDB | 1.309 | LEPA | 1.294 |

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0361; levels in B unseen in A: 0.444% of rows; unseen in all of 2025: 0.045% of rows (52 levels); null% A/B 1.545/1.534.

| level | A % | B % | Δ pp |
|---|---|---|---|
| EGLL | 1.899 | 1.877 | -0.022 |
| EDDF | 1.622 | 1.717 | +0.095 |
| LEMD | 1.669 | 1.670 | +0.001 |
| LFPG | 1.632 | 1.624 | -0.007 |
| EHAM | 1.699 | 1.531 | -0.168 |
| LEBL | 1.541 | 1.501 | -0.040 |
| LPPT | 1.439 | 1.405 | -0.033 |
| EDDM | 1.417 | 1.417 | +0.000 |
| LIRF | 1.359 | 1.402 | +0.043 |
| LEPA | 1.283 | 1.294 | +0.011 |
| LSZH | 1.213 | 1.276 | +0.062 |
| EIDW | 1.194 | 1.223 | +0.029 |

Relationship to target (train_adm, 1,437 levels): η² (in-sample) = 0.0571; held-out level-mean lookup RMSE = 673.62 s (gain +12.90 s vs global mean 686.53); held-out rows with unseen level = 1.562%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| EGLL | 39,024 | 937.0 | 851 | 2352 |
| EHAM | 35,521 | 951.8 | 890 | 2213 |
| EDDF | 35,481 | 882.8 | 807 | 1992 |
| LEMD | 34,527 | 985.1 | 902 | 2273 |
| LFPG | 34,410 | 958.2 | 893 | 2164 |
| LEBL | 32,316 | 972.3 | 896 | 2283 |
| EDDM | 30,249 | 903.9 | 841 | 1992 |
| LPPT | 29,733 | 942.9 | 884 | 2097 |
| LIRF | 28,762 | 960.2 | 902 | 2049 |
| EDDB | 26,999 | 878.9 | 792 | 2044 |
| LEPA | 26,349 | 889.1 | 840 | 1855 |
| LOWW | 25,542 | 898.7 | 836 | 2019 |
| LSZH | 25,481 | 936.2 | 863 | 2154 |
| EIDW | 24,590 | 1011.5 | 960 | 2212 |
| LGAV | 23,804 | 990.3 | 904 | 2637 |

### 2.19 `ADES_FILED_flt`

*Interpretation (from name and measured relations; no codebook was available):* destination as filed; equals ADES_flt on 99.9% of matched scored rows

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 1,433 / 1,031 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | no (TVD 0.036) |
| string length min–max (train) / blank strings | [4, 4] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 155 (0.04%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | EGLL | 1.892 | EGLL | 1.877 |
| 2 | EDDF | 1.719 | EDDF | 1.715 |
| 3 | EHAM | 1.716 | LEMD | 1.666 |
| 4 | LEMD | 1.673 | LFPG | 1.625 |
| 5 | LFPG | 1.668 | EHAM | 1.525 |
| 6 | LEBL | 1.566 | LEBL | 1.501 |
| 7 | EDDM | 1.464 | EDDM | 1.414 |
| 8 | LPPT | 1.442 | LPPT | 1.406 |
| 9 | LIRF | 1.393 | LIRF | 1.398 |
| 10 | EDDB | 1.309 | LEPA | 1.294 |

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0360; levels in B unseen in A: 0.447% of rows; unseen in all of 2025: 0.046% of rows (52 levels); null% A/B 1.545/1.534.

| level | A % | B % | Δ pp |
|---|---|---|---|
| EGLL | 1.900 | 1.877 | -0.023 |
| EDDF | 1.620 | 1.715 | +0.095 |
| LEMD | 1.667 | 1.666 | -0.001 |
| LFPG | 1.631 | 1.625 | -0.006 |
| EHAM | 1.693 | 1.525 | -0.168 |
| LEBL | 1.541 | 1.501 | -0.040 |
| LPPT | 1.440 | 1.406 | -0.035 |
| EDDM | 1.414 | 1.414 | -0.000 |
| LIRF | 1.360 | 1.398 | +0.038 |
| LEPA | 1.282 | 1.294 | +0.012 |
| LSZH | 1.212 | 1.273 | +0.061 |
| EDDB | 1.249 | 1.168 | -0.081 |

Relationship to target (train_adm, 1,433 levels): η² (in-sample) = 0.0572; held-out level-mean lookup RMSE = 673.61 s (gain +12.92 s vs global mean 686.53); held-out rows with unseen level = 1.563%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| EGLL | 39,028 | 937.0 | 851 | 2352 |
| EDDF | 35,450 | 882.8 | 807 | 1991 |
| EHAM | 35,389 | 952.5 | 891 | 2213 |
| LEMD | 34,498 | 984.8 | 902 | 2269 |
| LFPG | 34,408 | 958.0 | 893 | 2164 |
| LEBL | 32,307 | 972.5 | 896 | 2282 |
| EDDM | 30,187 | 904.4 | 842 | 1992 |
| LPPT | 29,749 | 943.1 | 884 | 2098 |
| LIRF | 28,738 | 959.8 | 902 | 2049 |
| EDDB | 27,005 | 878.9 | 792 | 2045 |
| LEPA | 26,342 | 889.1 | 840 | 1854 |
| LOWW | 25,532 | 898.6 | 836 | 2019 |
| LSZH | 25,454 | 936.4 | 863 | 2154 |
| EIDW | 24,573 | 1011.2 | 959 | 2211 |
| LGAV | 23,782 | 990.2 | 904 | 2634 |

### 2.20 `MARKET_SEGMENT_flt`

*Interpretation (from name and measured relations; no codebook was available):* NM market segment (Mainline/Lowcost/Regional/Cargo/Business/Non-Scheduled/Other/Military)

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 8 / 8 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | no (TVD 0.015) |
| string length min–max (train) / blank strings | [5, 13] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 0 (0.00%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | Mainline | 64.278 | Mainline | 65.844 |
| 2 | Lowcost | 19.369 | Lowcost | 19.227 |
| 3 | Regional | 10.224 | Regional | 8.714 |
| 4 | Cargo | 2.91 | Cargo | 2.918 |
| 5 | Business | 2.155 | Business | 2.242 |
| 6 | Non-Scheduled | 0.707 | Non-Scheduled | 0.634 |
| 7 | Other | 0.323 | Other | 0.359 |
| 8 | Military | 0.035 | Military | 0.062 |

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0147; levels in B unseen in A: 0.000% of rows; unseen in all of 2025: 0.000% of rows (0 levels); null% A/B 1.545/1.534.

| level | A % | B % | Δ pp |
|---|---|---|---|
| Mainline | 64.994 | 65.844 | +0.850 |
| Lowcost | 18.724 | 19.227 | +0.503 |
| Regional | 10.121 | 8.714 | -1.407 |
| Cargo | 2.924 | 2.918 | -0.006 |
| Business | 2.199 | 2.242 | +0.043 |
| Non-Scheduled | 0.695 | 0.634 | -0.061 |
| Other | 0.314 | 0.359 | +0.045 |
| Military | 0.029 | 0.062 | +0.033 |

Relationship to target (train_adm, 8 levels): η² (in-sample) = 0.0239; held-out level-mean lookup RMSE = 680.63 s (gain +5.90 s vs global mean 686.53); held-out rows with unseen level = 1.545%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| Mainline | 1,325,700 | 1032.8 | 957 | 2399 |
| Lowcost | 399,475 | 969.2 | 911 | 2156 |
| Regional | 210,854 | 770.4 | 724 | 1796 |
| Cargo | 60,014 | 1018.4 | 953 | 2348 |
| Business | 44,446 | 780.3 | 698 | 1977 |
| Non-Scheduled | 14,573 | 995.8 | 905 | 3202 |
| Other | 6,665 | 890.9 | 832 | 2347 |
| Military | 713 | 835.0 | 723 | 2626 |

### 2.21 `IOBT_flt`

*Interpretation (from name and measured relations; no codebook was available):* NM field; name suggests 'initial off-block time'; on a 1-minute grid, 95% on 5-minute grid

| property | value |
|---|---|
| dtype | `timestamp[us, tz=UTC]` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 178,155 / 31,284 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | no (offset-to-MVT p10/p50/p90 within 10%) |

Range and grid alignment (fraction of non-null values whose epoch-seconds are exact multiples of 60 / 300 / 3600):

| set | n | min | max | % whole-second | % on 60 s grid | % on 300 s grid | % on 3600 s grid |
|---|---|---|---|---|---|---|---|
| train_adm | 2,062,440 | 2025-01-01 00:20:00+00:00 | 2025-12-31 23:40:00+00:00 | 100.0 | 100.000 | 95.418 | 12.010 |
| scored | 339,551 | 2026-01-01 00:04:00+00:00 | 2026-07-31 23:30:00+00:00 | 100.0 | 100.000 | 94.870 | 12.182 |

Drift as offset to take-off, `IOBT_flt − MVT_TIME_UTC_mvt` in seconds (A vs B); % within ±60 s of zero: A 0.488 / B 0.432:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 339015 | -87,001 | -4,140 | -2,465 | -1,277 | -594 | -50 | 621 | 3,382 | -1,432 |
| B | 339551 | -86,820 | -4,139 | -2,404 | -1,263 | -598 | -61 | 530 | 3,173 | -1,415 |

Drift as offset to schedule, `IOBT_flt − SCHED_TIME_UTC_mvt` (s); % within ±60 s of zero: A 71.289 / B 71.833:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 339015 | -66,300 | -600 | 0 | 0 | 1,800 | 8,100 | 21,000 | 268,800 | 590 |
| B | 339551 | -86,400 | -600 | 0 | 0 | 1,860 | 8,700 | 23,100 | 172,800 | 639 |

Relationship to target via `MVT_TIME_UTC_mvt − IOBT_flt` (train_adm): Pearson +0.5155, Spearman +0.5106; equals the target exactly on 0.436% of rows; within ±60 s of zero on 0.486%. Decile table in §4.

### 2.22 `FLIGHT_RULE_flt`

*Interpretation (from name and measured relations; no codebook was available):* NM flight rules (I/Y/Z/V)

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.9132 / 1.0694 / 1.5381 / 100.0000 |
| cardinality train_adm / scored | 4 / 3 |
| usable at serve time | yes, except the unmatched stratum (1.538% null) |
| drift 2025 Jan+Jul → 2026 | no (TVD 0.000) |
| string length min–max (train) / blank strings | [1, 1] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 0 (0.00%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | I | 99.931 | I | 99.926 |
| 2 | Y | 0.069 | Y | 0.073 |
| 3 | Z | 0.0 | Z | 0.001 |
| 4 | V | 0.0 |  |  |

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0000; levels in B unseen in A: 0.001% of rows; unseen in all of 2025: 0.000% of rows (0 levels); null% A/B 1.547/1.538.

| level | A % | B % | Δ pp |
|---|---|---|---|
| I | 99.927 | 99.926 | -0.001 |
| Y | 0.072 | 0.073 | +0.000 |
| Z | 0.000 | 0.001 | +0.001 |
| V | 0.000 | 0.000 | -0.000 |

Relationship to target (train_adm, 4 levels): η² (in-sample) = 0.0002; held-out level-mean lookup RMSE = 686.59 s (gain -0.06 s vs global mean 686.53); held-out rows with unseen level = 1.547%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| I | 2,060,943 | 987.2 | 912 | 2306 |
| Y | 1,414 | 746.3 | 654 | 1945 |
| Z | 5 | 500.0 | 563 | 637 |
| V | 4 | 419.0 | 412 | 592 |

### 2.23 `FLIGHT_TYPE_flt`

*Interpretation (from name and measured relations; no codebook was available):* NM flight type (S scheduled / N non-scheduled / X / G general / M military)

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 5 / 5 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | no (TVD 0.001) |
| string length min–max (train) / blank strings | [1, 1] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 0 (0.00%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | S | 96.327 | S | 96.349 |
| 2 | N | 2.632 | N | 2.519 |
| 3 | X | 0.536 | X | 0.582 |
| 4 | G | 0.478 | G | 0.511 |
| 5 | M | 0.028 | M | 0.039 |

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0010; levels in B unseen in A: 0.000% of rows; unseen in all of 2025: 0.000% of rows (0 levels); null% A/B 1.545/1.534.

| level | A % | B % | Δ pp |
|---|---|---|---|
| S | 96.320 | 96.349 | +0.029 |
| N | 2.611 | 2.519 | -0.092 |
| X | 0.524 | 0.582 | +0.058 |
| G | 0.520 | 0.511 | -0.009 |
| M | 0.024 | 0.039 | +0.015 |

Relationship to target (train_adm, 5 levels): η² (in-sample) = 0.0022; held-out level-mean lookup RMSE = 686.03 s (gain +0.50 s vs global mean 686.53); held-out rows with unseen level = 1.545%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| S | 1,986,683 | 991.9 | 917 | 2308 |
| N | 54,277 | 865.0 | 819 | 2224 |
| X | 11,046 | 852.2 | 780 | 2825 |
| G | 9,853 | 825.3 | 728 | 2132 |
| M | 581 | 815.8 | 682 | 2563 |

### 2.24 `AIRCRAFT_TYPE_flt`

*Interpretation (from name and measured relations; no codebook was available):* ICAO aircraft type from NM flight plan; equals AIRCRAFT_TYPE_mvt on 99.8% of matched scored rows

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 239 / 180 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | **FLAG** TVD 0.065 |
| string length min–max (train) / blank strings | [3, 4] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 9 (0.00%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | A320 | 16.502 | A320 | 15.442 |
| 2 | B738 | 10.489 | B738 | 9.662 |
| 3 | A20N | 8.181 | A21N | 9.291 |
| 4 | A321 | 7.796 | A20N | 8.833 |
| 5 | A21N | 7.181 | A321 | 7.068 |
| 6 | A319 | 6.487 | A319 | 5.523 |
| 7 | E190 | 3.774 | BCS3 | 4.648 |
| 8 | BCS3 | 3.725 | B789 | 3.428 |
| 9 | B77W | 3.23 | E190 | 3.268 |
| 10 | B789 | 2.944 | B77W | 3.187 |

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0645; levels in B unseen in A: 0.014% of rows; unseen in all of 2025: 0.003% of rows (6 levels); null% A/B 1.545/1.534.

| level | A % | B % | Δ pp |
|---|---|---|---|
| A320 | 16.567 | 15.442 | -1.125 |
| B738 | 10.637 | 9.662 | -0.975 |
| A20N | 7.945 | 8.833 | +0.888 |
| A21N | 7.030 | 9.291 | +2.262 |
| A321 | 7.743 | 7.068 | -0.675 |
| A319 | 6.356 | 5.523 | -0.833 |
| BCS3 | 3.461 | 4.648 | +1.187 |
| E190 | 3.702 | 3.268 | -0.434 |
| B77W | 3.394 | 3.187 | -0.207 |
| B789 | 2.988 | 3.428 | +0.439 |
| A359 | 2.793 | 3.049 | +0.256 |
| B38M | 2.246 | 2.724 | +0.478 |

Relationship to target (train_adm, 239 levels): η² (in-sample) = 0.0649; held-out level-mean lookup RMSE = 671.47 s (gain +15.06 s vs global mean 686.53); held-out rows with unseen level = 1.548%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| A320 | 340,335 | 931.7 | 844 | 2222 |
| B738 | 216,325 | 975.8 | 943 | 2216 |
| A20N | 168,722 | 1016.2 | 948 | 2270 |
| A321 | 160,790 | 911.5 | 843 | 2154 |
| A21N | 148,100 | 1064.5 | 1004 | 2570 |
| A319 | 133,781 | 937.7 | 849 | 2169 |
| E190 | 77,843 | 803.7 | 757 | 1843 |
| BCS3 | 76,830 | 942.1 | 896 | 2092 |
| B77W | 66,620 | 1197.0 | 1133 | 2584 |
| B789 | 60,728 | 1218.1 | 1152 | 2519 |
| A359 | 55,882 | 1126.8 | 1029 | 2522 |
| A333 | 46,773 | 1104.1 | 1018 | 2645 |
| B38M | 44,961 | 1142.6 | 1098 | 2339 |
| A332 | 31,609 | 1092.3 | 1016 | 2635 |
| B772 | 30,874 | 1285.1 | 1248 | 2708 |

### 2.25 `WK_TBL_CAT_flt`

*Interpretation (from name and measured relations; no codebook was available):* ICAO wake-turbulence category (L/M/H/J)

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.9132 / 1.0694 / 1.5381 / 100.0000 |
| cardinality train_adm / scored | 4 / 4 |
| usable at serve time | yes, except the unmatched stratum (1.538% null) |
| drift 2025 Jan+Jul → 2026 | no (TVD 0.004) |
| string length min–max (train) / blank strings | [1, 1] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 0 (0.00%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | M | 78.386 | M | 77.557 |
| 2 | H | 20.407 | H | 21.25 |
| 3 | J | 0.713 | J | 0.704 |
| 4 | L | 0.494 | L | 0.489 |

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0039; levels in B unseen in A: 0.000% of rows; unseen in all of 2025: 0.000% of rows (0 levels); null% A/B 1.547/1.538.

| level | A % | B % | Δ pp |
|---|---|---|---|
| M | 77.926 | 77.557 | -0.369 |
| H | 20.887 | 21.250 | +0.363 |
| J | 0.721 | 0.704 | -0.017 |
| L | 0.467 | 0.489 | +0.023 |

Relationship to target (train_adm, 4 levels): η² (in-sample) = 0.0357; held-out level-mean lookup RMSE = 678.15 s (gain +8.38 s vs global mean 686.53); held-out rows with unseen level = 1.547%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| M | 1,616,604 | 936.6 | 878 | 2176 |
| H | 420,863 | 1172.6 | 1090 | 2585 |
| J | 14,712 | 1416.1 | 1327 | 2950 |
| L | 10,187 | 701.3 | 603 | 1909 |

### 2.26 `AIRCRAFT_OPERATOR_flt`

*Interpretation (from name and measured relations; no codebook was available):* operator, hashed (64-hex sha256-like); 676 levels in 2025

| property | value |
|---|---|
| dtype | `string` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 676 / 579 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | no (TVD 0.038) |
| string length min–max (train) / blank strings | [64, 64] / 0 |
| scored rows whose value never occurs in train_adm (global vocabulary) | 177 (0.05%) |

Top-10 values (share of non-null rows):

| rank | train_adm value | share % | scored value | share % |
|---|---|---|---|---|
| 1 | 11dca3faf550… | 10.807 | 95828822709b… | 11.452 |
| 2 | 95828822709b… | 10.791 | 11dca3faf550… | 9.558 |
| 3 | 6251e0eec78f… | 6.86 | aa14d43dcaf7… | 6.861 |
| 4 | aa14d43dcaf7… | 6.806 | 6251e0eec78f… | 6.326 |
| 5 | 8e92b7d7ac51… | 5.855 | 8e92b7d7ac51… | 5.832 |
| 6 | 4c1c009e6126… | 4.203 | 4c1c009e6126… | 4.058 |
| 7 | 813bafd420eb… | 3.783 | 813bafd420eb… | 3.6 |
| 8 | 3ba01c3b2192… | 3.392 | 3ba01c3b2192… | 3.206 |
| 9 | 4ac51f359070… | 3.023 | 4ac51f359070… | 3.089 |
| 10 | 65b92f7acecc… | 2.54 | 65b92f7acecc… | 2.58 |

Drift (A = 2025 Jan+Jul train_adm, B = 2026 scored): TVD = 0.0384; levels in B unseen in A: 0.236% of rows; unseen in all of 2025: 0.052% of rows (30 levels); null% A/B 1.545/1.534.

| level | A % | B % | Δ pp |
|---|---|---|---|
| 95828822709b… | 11.141 | 11.452 | +0.311 |
| 11dca3faf550… | 10.617 | 9.558 | -1.058 |
| aa14d43dcaf7… | 6.834 | 6.861 | +0.027 |
| 6251e0eec78f… | 6.894 | 6.326 | -0.568 |
| 8e92b7d7ac51… | 5.922 | 5.832 | -0.090 |
| 4c1c009e6126… | 4.202 | 4.058 | -0.144 |
| 813bafd420eb… | 3.738 | 3.600 | -0.138 |
| 3ba01c3b2192… | 3.319 | 3.206 | -0.113 |
| 4ac51f359070… | 3.092 | 3.089 | -0.003 |
| 65b92f7acecc… | 2.666 | 2.580 | -0.085 |
| 32d3f7447452… | 1.807 | 1.933 | +0.126 |
| 896f2dd89b8e… | 1.337 | 1.513 | +0.176 |

Relationship to target (train_adm, 676 levels): η² (in-sample) = 0.1152; held-out level-mean lookup RMSE = 659.42 s (gain +27.11 s vs global mean 686.53); held-out rows with unseen level = 1.551%.

| level | n | target mean | median | p99 |
|---|---|---|---|---|
| 11dca3faf550… | 222,892 | 791.7 | 746 | 1797 |
| 95828822709b… | 222,550 | 1060.6 | 967 | 2527 |
| 6251e0eec78f… | 141,484 | 764.5 | 718 | 1686 |
| aa14d43dcaf7… | 140,374 | 991.2 | 907 | 2393 |
| 8e92b7d7ac51… | 120,747 | 1383.8 | 1323 | 3119 |
| 4c1c009e6126… | 86,684 | 833.5 | 775 | 1923 |
| 813bafd420eb… | 78,031 | 794.4 | 756 | 1829 |
| 3ba01c3b2192… | 69,958 | 1139.0 | 1115 | 2137 |
| 4ac51f359070… | 62,340 | 952.2 | 909 | 1925 |
| 65b92f7acecc… | 52,395 | 1035.1 | 958 | 2351 |
| 1901074fe2ac… | 37,210 | 932.6 | 868 | 1997 |
| 32d3f7447452… | 36,090 | 997.8 | 971 | 1857 |
| 896f2dd89b8e… | 28,168 | 825.6 | 805 | 1695 |
| af8b67cf63bb… | 26,519 | 839.8 | 835 | 1742 |
| f3270be89c31… | 22,602 | 881.5 | 836 | 2163 |

### 2.27 `EOBT_1_flt`

*Interpretation (from name and measured relations; no codebook was available):* NM field; name suggests 'estimated off-block time, message 1'; 1-minute grid, 85% on 5-minute grid

| property | value |
|---|---|
| dtype | `timestamp[us, tz=UTC]` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 295,502 / 50,720 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | no (offset-to-MVT p10/p50/p90 within 10%) |

Range and grid alignment (fraction of non-null values whose epoch-seconds are exact multiples of 60 / 300 / 3600):

| set | n | min | max | % whole-second | % on 60 s grid | % on 300 s grid | % on 3600 s grid |
|---|---|---|---|---|---|---|---|
| train_adm | 2,062,440 | 2025-01-01 00:20:00+00:00 | 2025-12-31 23:45:00+00:00 | 100.0 | 99.697 | 85.077 | 10.007 |
| scored | 339,551 | 2026-01-01 00:04:00+00:00 | 2026-07-31 23:30:00+00:00 | 100.0 | 99.776 | 84.017 | 10.060 |

Drift as offset to take-off, `EOBT_1_flt − MVT_TIME_UTC_mvt` in seconds (A vs B); % within ±60 s of zero: A 0.145 / B 0.11:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 339015 | -87,001 | -3,918 | -2,348 | -1,279 | -668 | -278 | 299 | 9,390 | -1,428 |
| B | 339551 | -86,280 | -3,902 | -2,281 | -1,257 | -666 | -299 | 216 | 10,314 | -1,397 |

Drift as offset to schedule, `EOBT_1_flt − SCHED_TIME_UTC_mvt` (s); % within ±60 s of zero: A 53.686 / B 51.954:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 339015 | -66,300 | -720 | -300 | 0 | 1,980 | 8,160 | 21,300 | 268,800 | 593 |
| B | 339551 | -86,400 | -720 | -300 | 0 | 2,100 | 8,700 | 23,508 | 172,800 | 657 |

Relationship to target via `MVT_TIME_UTC_mvt − EOBT_1_flt` (train_adm): Pearson +0.5564, Spearman +0.5681; equals the target exactly on 0.471% of rows; within ±60 s of zero on 0.144%. Decile table in §4.

### 2.28 `ARVT_1_flt`

*Interpretation (from name and measured relations; no codebook was available):* NM field; name suggests 'arrival time, message 1'; lies AFTER take-off on >99.9% of rows (p0.1 of ARVT_1-MVT = -562 s in Jan+Jul 2025, p1 = +876 s)

| property | value |
|---|---|
| dtype | `timestamp[us, tz=UTC]` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 1,981,417 / 326,574 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | no (offset-to-MVT p10/p50/p90 within 10%) |

Range and grid alignment (fraction of non-null values whose epoch-seconds are exact multiples of 60 / 300 / 3600):

| set | n | min | max | % whole-second | % on 60 s grid | % on 300 s grid | % on 3600 s grid |
|---|---|---|---|---|---|---|---|
| train_adm | 2,062,440 | 2025-01-01 02:47:21+00:00 | 2026-01-01 12:53:09+00:00 | 100.0 | 1.701 | 0.335 | 0.029 |
| scored | 339,551 | 2026-01-01 02:44:06+00:00 | 2026-08-01 12:33:03+00:00 | 100.0 | 1.777 | 0.340 | 0.025 |

Drift as offset to take-off, `ARVT_1_flt − MVT_TIME_UTC_mvt` in seconds (A vs B); % within ±60 s of zero: A 0.04 / B 0.037:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 339015 | -83,542 | 876 | 2,376 | 5,800 | 30,448 | 43,458 | 48,485 | 59,757 | 10,380 |
| B | 339551 | -82,162 | 984 | 2,480 | 6,000 | 30,985 | 43,316 | 48,641 | 61,971 | 10,669 |

Relationship to target via `MVT_TIME_UTC_mvt − ARVT_1_flt` (train_adm): Pearson -0.2363, Spearman -0.2580; equals the target exactly on 0.000% of rows; within ±60 s of zero on 0.036%. Decile table in §4.

### 2.29 `AOBT_3_flt`

*Interpretation (from name and measured relations; no codebook was available):* NM field; name suggests 'actual off-block time, message 3'; BLOCK_TIME - AOBT_3 has median +51 s, p10 -310 s, p90 +363 s on training

| property | value |
|---|---|
| dtype | `timestamp[us, tz=UTC]` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 486,924 / 83,435 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | no (offset-to-MVT p10/p50/p90 within 10%) |

Range and grid alignment (fraction of non-null values whose epoch-seconds are exact multiples of 60 / 300 / 3600):

| set | n | min | max | % whole-second | % on 60 s grid | % on 300 s grid | % on 3600 s grid |
|---|---|---|---|---|---|---|---|
| train_adm | 2,062,440 | 2025-01-01 00:15:00+00:00 | 2025-12-31 23:39:00+00:00 | 100.0 | 97.700 | 19.881 | 1.711 |
| scored | 339,551 | 2026-01-01 00:02:00+00:00 | 2026-07-31 23:42:00+00:00 | 100.0 | 97.429 | 19.776 | 1.667 |

Drift as offset to take-off, `AOBT_3_flt − MVT_TIME_UTC_mvt` in seconds (A vs B); % within ±60 s of zero: A 0.116 / B 0.151:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 339015 | -87,181 | -2,278 | -1,495 | -960 | -625 | -351 | -15 | 11,269 | -1,014 |
| B | 339551 | -86,340 | -2,341 | -1,498 | -978 | -629 | -305 | -19 | 11,519 | -1,036 |

Drift as offset to schedule, `AOBT_3_flt − SCHED_TIME_UTC_mvt` (s); % within ±60 s of zero: A 9.278 / B 10.001:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 339015 | -67,860 | -1,020 | -360 | 480 | 2,760 | 9,000 | 21,719 | 269,280 | 1,007 |
| B | 339551 | -86,400 | -960 | -360 | 480 | 2,760 | 9,480 | 23,700 | 173,460 | 1,018 |

Relationship to target via `MVT_TIME_UTC_mvt − AOBT_3_flt` (train_adm): Pearson +0.5358, Spearman +0.6150; equals the target exactly on 0.646% of rows; within ±60 s of zero on 0.125%. Decile table in §4.

### 2.30 `ARVT_3_flt`

*Interpretation (from name and measured relations; no codebook was available):* NM field; name suggests 'arrival time, message 3'; lies AFTER take-off on >99.9% of rows (p0.1 of ARVT_3-MVT = +1192 s)

| property | value |
|---|---|
| dtype | `timestamp[us, tz=UTC]` |
| null % train raw / train_adm / scored / unmatched | 0.9098 / 1.0658 / 1.5340 / 100.0000 |
| cardinality train_adm / scored | 1,726,790 / 279,144 |
| usable at serve time | yes, except the unmatched stratum (1.534% null) |
| drift 2025 Jan+Jul → 2026 | no (offset-to-MVT p10/p50/p90 within 10%) |

Range and grid alignment (fraction of non-null values whose epoch-seconds are exact multiples of 60 / 300 / 3600):

| set | n | min | max | % whole-second | % on 60 s grid | % on 300 s grid | % on 3600 s grid |
|---|---|---|---|---|---|---|---|
| train_adm | 2,062,440 | 2025-01-01 02:35:24+00:00 | 2026-01-01 13:18:09+00:00 | 100.0 | 29.946 | 6.024 | 0.504 |
| scored | 339,551 | 2026-01-01 02:52:52+00:00 | 2026-08-01 12:54:00+00:00 | 100.0 | 32.500 | 6.473 | 0.517 |

Drift as offset to take-off, `ARVT_3_flt − MVT_TIME_UTC_mvt` in seconds (A vs B); % within ±60 s of zero: A 0.0 / B 0.001:

| set | n | min | p1 | p10 | p50 | p90 | p99 | p99.9 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 339015 | -84,061 | 1,669 | 2,756 | 6,219 | 30,966 | 44,068 | 48,823 | 59,888 | 10,796 |
| B | 339551 | -81,600 | 1,681 | 2,766 | 6,304 | 31,512 | 43,812 | 49,132 | 61,560 | 11,007 |

Relationship to target via `MVT_TIME_UTC_mvt − ARVT_3_flt` (train_adm): Pearson -0.2543, Spearman -0.2971; equals the target exactly on 0.000% of rows; within ±60 s of zero on 0.000%. Decile table in §4.

## 3. Drift summary, 2025 Jan+Jul (A) → 2026 scored (B)

### 3.1 Columns that moved

| column | evidence | verdict |
|---|---|---|
| `MVT_ID_mvt` | id range disjoint from 2025 (by construction; useless as a value) | FLAG |
| `FLIGHT_ID_mvt` | id range disjoint from 2025 (by construction; useless as a value) | FLAG |
| `FLIGHT_mvt` | TVD 0.454; 21.1% of 2026 rows unseen in 2025 | FLAG |
| `AIRCRAFT_TYPE_mvt` | TVD 0.066 | FLAG |
| `RUNWAY_mvt` | TVD 0.102 | FLAG |
| `STAND_mvt` | TVD 0.079 | FLAG |
| `CALLSIGN_flt` | TVD 0.532; 24.8% of 2026 rows unseen in 2025 | FLAG |
| `AIRCRAFT_TYPE_flt` | TVD 0.065 | FLAG |
| unmatched-stratum rate | by month: 2025-01 0.916% → 2026-01 1.606%; 2025-07 2.052% → 2026-07 1.477% (pooled 1.545% → 1.534%). By airport: EGLL 1.16→0.62, EHAM 1.97→3.16, LSZH 2.14→2.85 | FLAG (regime, not pooled rate) |
| day-of-week mix | A {'0': 13.113, '1': 14.293, '2': 15.932, '3': 16.274, '4': 14.897, '5': 12.436, '6': 13.055} vs B {'0': 12.99, '1': 12.419, '2': 14.355, '3': 16.371, '4': 16.874, '5': 13.801, '6': 13.19} (calendar alignment of the two months; Tue −1.9 pp, Fri +2.0 pp) | minor |

All other categorical columns: TVD < 0.05 and < 1% unseen rows (table in §1). All timestamp offsets to take-off: p10/p50/p90 within 10% (tables in §2).

### 3.2 RUNWAY_mvt drift is airport-specific (share of that airport's departures, %)

| airport | n A | n B | TVD | top runways A % → B % | runways in B unseen in A |
|---|---|---|---|---|---|
| EDDF | 36,830 | 36,317 | 0.087 | 18 64.2→63.0; 25C 26.8→19.3; 07C 8.7→17.2; 25L 0.2→0.4; 07R 0.1→0.1 | ['NA'] |
| EDDM | 27,091 | 25,966 | 0.039 | 26L 42.2→41.2; 26R 30.3→27.3; 08R 14.7→15.2; 08L 12.8→16.2 | - |
| EGLL | 40,210 | 39,840 | 0.269 | 27L 41.6→29.3; 27R 42.7→28.2; 09R 15.6→42.2; 09L 0.1→0.3; NA 0.0→0.0 | - |
| EHAM | 40,949 | 38,182 | 0.159 | 24 37.5→26.7; 36L 22.0→28.4; 18L 25.5→20.9; 36C 11.2→15.0; 09 0.6→3.4 | ['06', 'HEL'] |
| LEBL | 28,985 | 30,080 | 0.113 | 24L 76.9→65.6; 06R 21.3→32.3; 24R 1.2→1.3; 06L 0.6→0.8 | - |
| LEMD | 35,328 | 36,954 | 0.018 | 36L 38.4→37.2; 36R 36.2→35.7; 14L 15.1→14.9; 14R 10.3→12.2; 32L 0.0→0.0 | ['32L', '32R'] |
| LFPG | 39,589 | 39,872 | 0.188 | 26R 39.8→30.3; 08L 19.0→27.0; 27L 19.4→24.6; 09R 12.2→17.7; 27R 8.8→0.1 | ['NA'] |
| LIRF | 26,528 | 26,899 | 0.044 | 25 85.8→90.0; 16R 9.2→8.8; 34L 4.6→0.8; 16L 0.2→0.4; 34R 0.2→0.0 | - |
| LSZH | 22,287 | 23,150 | 0.055 | 28 59.5→64.2; 32 29.2→23.8; 16 9.4→10.1; 34 1.2→1.1; 10 0.5→0.5 | - |
| LTFM | 46,539 | 47,581 | 0.070 | 36 44.4→39.4; 35L 30.2→28.9; 18 13.1→15.6; 17R 8.6→10.2; 34L 2.5→4.3 | - |

### 3.3 Unmatched-stratum rate, train_adm by month (2025)

| 2025-01 | 2025-02 | 2025-03 | 2025-04 | 2025-05 | 2025-06 | 2025-07 | 2025-08 | 2025-09 | 2025-10 | 2025-11 | 2025-12 | all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.916 | 0.753 | 0.645 | 0.607 | 0.859 | 1.349 | 2.052 | 1.201 | 1.334 | 1.024 | 0.830 | 0.985 | 1.0658 |

## 4. Every derivable timestamp difference (36 pairs)

Nine timestamps → 36 unordered pairs, reported as X − Y in seconds, X earlier in the list {MVT, BLOCK, SCHED, LOBT, IOBT, EOBT_1, AOBT_3, ARVT_1, ARVT_3}. 'Computable on scored' = both operands non-null on at least one scored row; the null % is the union of operand nulls. Correlations and deciles are on train_adm.

| pair X − Y | computable on scored | null % scored | null % train_adm | train p1 | p10 | p50 | p90 | p99 | scored p10 | p50 | p90 | % \|d\|≤60 s (train) | % d == target | Pearson | Spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MVT - BLOCK | **NO** | 100.000 | 0.000 | 293 | 571 | 912 | 1,471 | 2,339 | - | - | - | 0.04 | 100.00 | 1.0000 | 1.0000 |
| MVT - SCHED | yes | 0.000 | 0.000 | 56 | 599 | 1,403 | 3,484 | 9,305 | 653 | 1,501 | 4,035 | 0.44 | 0.39 | 0.2320 | 0.3866 |
| MVT - LOBT | yes | 1.534 | 1.066 | 53 | 564 | 1,237 | 2,280 | 3,840 | 597 | 1,263 | 2,404 | 0.48 | 0.43 | 0.5141 | 0.5086 |
| MVT - IOBT | yes | 1.534 | 1.066 | 52 | 571 | 1,238 | 2,280 | 3,841 | 598 | 1,263 | 2,404 | 0.49 | 0.44 | 0.5155 | 0.5106 |
| MVT - EOBT_1 | yes | 1.534 | 1.066 | 269 | 663 | 1,255 | 2,215 | 3,603 | 666 | 1,257 | 2,281 | 0.14 | 0.47 | 0.5564 | 0.5681 |
| MVT - AOBT_3 | yes | 1.534 | 1.066 | 363 | 626 | 958 | 1,444 | 2,105 | 629 | 978 | 1,498 | 0.12 | 0.65 | 0.5358 | 0.6150 |
| MVT - ARVT_1 | yes | 1.534 | 1.066 | -43,361 | -30,173 | -5,813 | -2,411 | -962 | -30,985 | -6,000 | -2,480 | 0.04 | 0.00 | -0.2363 | -0.2580 |
| MVT - ARVT_3 | yes | 1.534 | 1.066 | -43,910 | -30,639 | -6,154 | -2,755 | -1,674 | -31,512 | -6,304 | -2,766 | 0.00 | 0.00 | -0.2543 | -0.2971 |
| BLOCK - SCHED | **NO** | 100.000 | 0.000 | -759 | -242 | 415 | 2,396 | 8,163 | - | - | - | 9.57 | 0.05 | -0.0057 | -0.0148 |
| BLOCK - LOBT | **NO** | 100.000 | 1.066 | -838 | -272 | 251 | 1,136 | 2,622 | - | - | - | 11.14 | 0.05 | -0.0502 | -0.0392 |
| BLOCK - IOBT | **NO** | 100.000 | 1.066 | -836 | -250 | 251 | 1,136 | 2,626 | - | - | - | 11.18 | 0.05 | -0.0489 | -0.0385 |
| BLOCK - EOBT_1 | **NO** | 100.000 | 1.066 | -603 | -127 | 283 | 1,036 | 2,336 | - | - | - | 12.52 | 0.05 | -0.0754 | -0.0535 |
| BLOCK - AOBT_3 | **NO** | 100.000 | 1.066 | -1,194 | -310 | 51 | 363 | 799 | - | - | - | 20.80 | 0.02 | -0.5602 | -0.4756 |
| BLOCK - ARVT_1 | **NO** | 100.000 | 1.066 | -44,580 | -31,394 | -6,766 | -3,282 | -1,825 | - | - | - | 0.01 | 0.00 | -0.2721 | -0.3387 |
| BLOCK - ARVT_3 | **NO** | 100.000 | 1.066 | -45,147 | -31,850 | -7,112 | -3,597 | -2,370 | - | - | - | 0.00 | 0.00 | -0.2895 | -0.3777 |
| SCHED - LOBT | yes | 1.534 | 1.066 | -7,200 | -1,500 | 0 | 0 | 600 | -1,860 | 0 | 0 | 73.51 | 0.00 | -0.0331 | -0.0139 |
| SCHED - IOBT | yes | 1.534 | 1.066 | -7,200 | -1,500 | 0 | 0 | 600 | -1,860 | 0 | 0 | 73.86 | 0.00 | -0.0327 | -0.0134 |
| SCHED - EOBT_1 | yes | 1.534 | 1.066 | -7,200 | -1,620 | 0 | 300 | 720 | -2,100 | 0 | 300 | 55.96 | 0.01 | -0.0375 | -0.0002 |
| SCHED - AOBT_3 | yes | 1.534 | 1.066 | -7,920 | -2,340 | -420 | 360 | 1,080 | -2,760 | -480 | 360 | 10.32 | 0.02 | -0.1138 | -0.1531 |
| SCHED - ARVT_1 | yes | 1.534 | 1.066 | -46,013 | -32,514 | -7,571 | -3,756 | -2,471 | -33,623 | -7,943 | -3,894 | 0.00 | 0.00 | -0.2680 | -0.3323 |
| SCHED - ARVT_3 | yes | 1.534 | 1.066 | -46,533 | -33,012 | -7,950 | -3,940 | -2,520 | -34,113 | -8,280 | -4,020 | 0.00 | 0.00 | -0.2843 | -0.3599 |
| LOBT - IOBT | yes | 1.534 | 1.066 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 98.45 | 0.00 | 0.0163 | 0.0054 |
| LOBT - EOBT_1 | yes | 1.534 | 1.066 | -1,260 | -240 | 0 | 300 | 900 | -300 | 0 | 300 | 70.65 | 0.01 | -0.0287 | 0.0068 |
| LOBT - AOBT_3 | yes | 1.534 | 1.066 | -2,760 | -1,200 | -240 | 420 | 1,080 | -1,260 | -240 | 360 | 12.71 | 0.02 | -0.2512 | -0.1976 |
| LOBT - ARVT_1 | yes | 1.534 | 1.066 | -44,980 | -31,798 | -7,120 | -3,610 | -2,376 | -32,714 | -7,363 | -3,711 | 0.00 | 0.00 | -0.2689 | -0.3349 |
| LOBT - ARVT_3 | yes | 1.534 | 1.066 | -45,568 | -32,291 | -7,500 | -3,798 | -2,475 | -33,237 | -7,716 | -3,840 | 0.00 | 0.00 | -0.2856 | -0.3657 |
| IOBT - EOBT_1 | yes | 1.534 | 1.066 | -1,260 | -300 | 0 | 300 | 900 | -300 | 0 | 300 | 70.18 | 0.01 | -0.0308 | 0.0059 |
| IOBT - AOBT_3 | yes | 1.534 | 1.066 | -2,760 | -1,200 | -240 | 420 | 1,080 | -1,320 | -240 | 360 | 12.72 | 0.02 | -0.2518 | -0.1981 |
| IOBT - ARVT_1 | yes | 1.534 | 1.066 | -44,983 | -31,799 | -7,122 | -3,611 | -2,377 | -32,717 | -7,366 | -3,711 | 0.00 | 0.00 | -0.2690 | -0.3351 |
| IOBT - ARVT_3 | yes | 1.534 | 1.066 | -45,569 | -32,293 | -7,500 | -3,799 | -2,476 | -33,239 | -7,720 | -3,840 | 0.00 | 0.00 | -0.2857 | -0.3658 |
| EOBT_1 - AOBT_3 | yes | 1.534 | 1.066 | -2,520 | -1,140 | -240 | 300 | 960 | -1,140 | -240 | 240 | 14.76 | 0.01 | -0.2675 | -0.2181 |
| EOBT_1 - ARVT_1 | yes | 1.534 | 1.066 | -44,975 | -31,824 | -7,109 | -3,640 | -2,505 | -32,690 | -7,333 | -3,720 | 0.00 | 0.00 | -0.2679 | -0.3348 |
| EOBT_1 - ARVT_3 | yes | 1.534 | 1.066 | -45,568 | -32,324 | -7,497 | -3,840 | -2,580 | -33,222 | -7,680 | -3,865 | 0.00 | 0.00 | -0.2846 | -0.3659 |
| AOBT_3 - ARVT_1 | yes | 1.534 | 1.066 | -44,526 | -31,337 | -6,787 | -3,286 | -1,788 | -32,188 | -6,991 | -3,386 | 0.01 | 0.00 | -0.2528 | -0.3014 |
| AOBT_3 - ARVT_3 | yes | 1.534 | 1.066 | -45,060 | -31,808 | -7,115 | -3,633 | -2,417 | -32,713 | -7,303 | -3,664 | 0.00 | 0.00 | -0.2706 | -0.3409 |
| ARVT_1 - ARVT_3 | yes | 1.534 | 1.066 | -2,733 | -1,273 | -280 | 467 | 1,158 | -1,275 | -246 | 497 | 7.38 | 0.01 | -0.2664 | -0.2391 |

Decile table of target mean (train_adm) for every pair computable on scored rows; bins are feature deciles, entries are target mean in seconds (bin edges in `s4.json`). Rows with fewer than 10 entries are offsets that are tied at one value on a large share of rows (e.g. `SCHED − LOBT` is exactly 0 on 73% of rows), so `pd.qcut(..., duplicates='drop')` yields fewer bins:

| pair X − Y | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 |
|---|---|---|---|---|---|---|---|---|---|---|
| MVT - SCHED | 695 | 778 | 860 | 926 | 986 | 1036 | 1092 | 1135 | 1172 | 1236 |
| MVT - LOBT | 708 | 757 | 819 | 885 | 930 | 997 | 1039 | 1116 | 1222 | 1398 |
| MVT - IOBT | 703 | 759 | 819 | 886 | 930 | 998 | 1039 | 1116 | 1223 | 1398 |
| MVT - EOBT_1 | 662 | 738 | 816 | 873 | 934 | 1008 | 1056 | 1130 | 1224 | 1434 |
| MVT - AOBT_3 | 697 | 728 | 788 | 870 | 918 | 1018 | 1094 | 1082 | 1249 | 1432 |
| MVT - ARVT_1 | 1191 | 1175 | 1023 | 981 | 958 | 927 | 894 | 881 | 880 | 959 |
| MVT - ARVT_3 | 1199 | 1176 | 1042 | 991 | 977 | 942 | 902 | 893 | 899 | 850 |
| SCHED - LOBT | 1043 | 978 | 978 | 1026 | | | | | | |
| SCHED - IOBT | 1043 | 977 | 979 | 1021 | | | | | | |
| SCHED - EOBT_1 | 1058 | 1000 | 967 | 1030 | 1003 | | | | | |
| SCHED - AOBT_3 | 1135 | 1084 | 1038 | 1012 | 976 | 952 | 942 | 929 | 911 | 883 |
| SCHED - ARVT_1 | 1202 | 1178 | 1063 | 1003 | 999 | 950 | 919 | 886 | 867 | 802 |
| SCHED - ARVT_3 | 1210 | 1182 | 1080 | 1023 | 999 | 954 | 924 | 892 | 858 | 747 |
| LOBT - IOBT | 986 | 1197 | | | | | | | | |
| LOBT - EOBT_1 | 1013 | 979 | 1023 | 986 | | | | | | |
| LOBT - AOBT_3 | 1226 | 1079 | 1028 | 981 | 953 | 937 | 932 | 912 | 903 | 895 |
| LOBT - ARVT_1 | 1205 | 1174 | 1059 | 1012 | 991 | 953 | 908 | 894 | 871 | 802 |
| LOBT - ARVT_3 | 1212 | 1179 | 1077 | 1029 | 999 | 952 | 923 | 899 | 855 | 743 |
| IOBT - EOBT_1 | 1017 | 978 | 1026 | 987 | | | | | | |
| IOBT - AOBT_3 | 1226 | 1079 | 1027 | 980 | 953 | 937 | 932 | 912 | 903 | 895 |
| IOBT - ARVT_1 | 1205 | 1174 | 1059 | 1012 | 991 | 953 | 908 | 894 | 871 | 802 |
| IOBT - ARVT_3 | 1212 | 1179 | 1078 | 1028 | 1000 | 952 | 923 | 899 | 855 | 743 |
| EOBT_1 - AOBT_3 | 1244 | 1085 | 1025 | 980 | 954 | 934 | 923 | 909 | 883 | 891 |
| EOBT_1 - ARVT_1 | 1205 | 1175 | 1057 | 1012 | 990 | 956 | 906 | 894 | 877 | 800 |
| EOBT_1 - ARVT_3 | 1212 | 1180 | 1075 | 1029 | 999 | 954 | 922 | 900 | 860 | 738 |
| AOBT_3 - ARVT_1 | 1198 | 1173 | 1038 | 1002 | 976 | 943 | 903 | 887 | 869 | 880 |
| AOBT_3 - ARVT_3 | 1206 | 1175 | 1056 | 1014 | 992 | 954 | 919 | 904 | 868 | 780 |
| ARVT_1 - ARVT_3 | 1245 | 1103 | 1043 | 1000 | 971 | 944 | 922 | 901 | 881 | 858 |

Observations that are facts of the pairs table:

- `MVT − BLOCK` equals the target on 100% of rows (definition check). The eight pairs containing BLOCK_TIME are **not computable on any scored row**.
- `BLOCK − AOBT_3`: median 51 s, p10 -310, p90 363, p1 -1,194, p99 799; within ±60 s on 20.8% of rows. So AOBT_3 is close to, but not equal to, the hidden off-block time; `MVT − AOBT_3` equals the target exactly on only 0.646% of rows.
- `LOBT − IOBT` is exactly zero on 98.4% of rows: the two columns are near-duplicates.
- `SCHED − LOBT` is exactly zero on 73.2% of rows; `SCHED − EOBT_1` on 54.9%.
- `ARVT_1` and `ARVT_3` lie after take-off on essentially all rows (train p99 of MVT − ARVT_3 = -1,674 s; p99.9 = -1,201). They are present on scored rows, but they are post-departure quantities; their target association (Spearman ≈ −0.3) is via flight length, i.e. destination distance.
- Among pairs computable on scored rows, the strongest rank associations with the target are `MVT − AOBT_3` (Spearman +0.615), `MVT − EOBT_1` (+0.568), `MVT − IOBT` (+0.511), `MVT − LOBT` (+0.509), `MVT − SCHED` (+0.387). Only `MVT − SCHED` is computable inside the unmatched stratum.

## 5. The unmatched stratum (`AOBT_3_flt.isna()`)

| quantity | matched (train_adm) | unmatched (train_adm) | unmatched (2026 scored) |
|---|---|---|---|
| rows | 2,062,440 | 22,219 (1.066%) | 5,290 (1.534%) |
| target mean / median / p99 / p99.9 / max | 987 / 911 / 2304 / 3960 / 87,002 | 1400 / 953 / 10,254 / 52,911 / 131,167 | hidden |
| target p1 / p10 | 299 / 573 | 11 / 365 | hidden |
| target > 3600 s / > 10800 s | 0.147% / 0.001% | 4.892% / 0.837% | hidden |
| share of total squared error around global mean | 57.8% | **42.2%** | - |
| MVT − SCHED p10 / p50 / p90 / p99 | 599 / 1,389 / 3,386 / 8,944 | 637 / 5,313 / 9,297 / 20,808 | 843 / 5,830 / 12,004 / 28,377 |
| Spearman(MVT − SCHED, target) | +0.3883 | +0.4135 | - |
| BLOCK − SCHED p10 / p50 / p90 | -242 / 409 / 2,283 | -115 / 4,097 / 7,799 | hidden |
| BLOCK within ±60 s of SCHED | 9.61% | 6.28% | hidden |
| AIRCRAFT_TYPE_mvt null | 0.113% (all scored) | - | 7.33% |
| FLIGHT_RULE_mvt = V | 0.122% (all scored) | - | 7.93% |

Decile table of target mean by `MVT − SCHED` inside the unmatched stratum (train_adm):

| decile | MVT − SCHED range (s) | n | target mean | median | p99 |
|---|---|---|---|---|---|
| 1 | -37,860 … 637 | 2,223 | 491 | 445 | 1,310 |
| 2 | 638 … 1,316 | 2,229 | 835 | 839 | 1,671 |
| 3 | 1,317 … 4,021 | 2,215 | 1171 | 1010 | 4,174 |
| 4 | 4,022 … 4,861 | 2,226 | 903 | 781 | 4,688 |
| 5 | 4,862 … 5,313 | 2,217 | 1096 | 1009 | 5,052 |
| 6 | 5,315 … 5,818 | 2,224 | 1220 | 1066 | 5,684 |
| 7 | 5,819 … 6,478 | 2,224 | 1348 | 1088 | 6,241 |
| 8 | 6,479 … 7,445 | 2,217 | 1544 | 1129 | 7,311 |
| 9 | 7,446 … 9,297 | 2,222 | 1682 | 1091 | 9,039 |
| 10 | 9,298 … 151,639 | 2,222 | 3713 | 1143 | 47,524 |

Fact: in the unmatched stratum the median `MVT − SCHED` is 5,313 s (train) / 5,830 s (2026), versus 1,389 s in the matched stratum - these departures take off ~1.5 h after their scheduled off-block, yet their median taxi-out (953 s) is ordinary. Whether that is a schedule-filled BLOCK_TIME or a genuine late push is not decidable from these columns; the depth dive owns that question.

## 6. Data-quality facts

| fact | numbers |
|---|---|
| `"NA"` literal string in RUNWAY_mvt | 36 train_adm rows, 4 scored rows; no other string column carries NA/NaN/NULL/blank sentinels (s6) |
| partially-null NM rows | 14 scored rows (train_adm 74) have AOBT_3 present but FLIGHT_RULE_flt and WK_TBL_CAT_flt both null |
| non-positive taxi in raw training DEP rows | 388 (dropped by the admissibility filter) |
| duplicate MVT_ID across the 12 files | 0 |
| take-off before scheduled off-block | train_adm p1 of MVT − SCHED = 56 s, p0.1 = -1,061 s, min -113,526 s; scored p0.1 = -848 s, min -85,810 s |
| SCHED_TIME earlier than the file window | train_adm min 2024-12-31 22:40:00+00:00; scored min 2025-12-31 09:35:00+00:00 (rows scheduled on 2025-12-31 that took off in 2026-01) |
| ARVT before MVT (impossible arrivals) | train_adm min of ARVT_3 − MVT = -84,061 s (Jan+Jul); fraction is below p0.1 |
| cross-column agreement on matched scored rows | ADEP 100.0%, ADES 98.56%, ADES_flt vs FILED 99.88%, AIRCRAFT_TYPE 99.82%, FLIGHT vs CALLSIGN 74.0%, FLIGHT_RULE 99.92% |
| grid alignment | SCHED 100% on 1-min, 99.97% on 5-min; IOBT 100% / 95.4%; LOBT 99.7% / 94.9%; EOBT_1 99.7% / 85.1%; AOBT_3 97.7% / 19.9%; MVT_TIME 5.1% (second-resolution); BLOCK_TIME 6.4% (second-resolution); ARVT_1 1.7%; ARVT_3 29.9% |

## Appendix A - scripts (verbatim) and commands

All scripts live in the session scratchpad `…/scratchpad/dd/`; the admissible-departure cache they build (`adm_2025-MM.parquet`, `scored.parquet`) is derived data outside the repository. Commands, in order:

```
OMP_NUM_THREADS=1 nice -n 19 /Users/saurabhdxt/Projects/Phantom/.venv/bin/python s0_build_cache.py
OMP_NUM_THREADS=1 nice -n 19 /Users/saurabhdxt/Projects/Phantom/.venv/bin/python s1_profile.py
OMP_NUM_THREADS=1 nice -n 19 /Users/saurabhdxt/Projects/Phantom/.venv/bin/python s2_drift.py
OMP_NUM_THREADS=1 nice -n 19 /Users/saurabhdxt/Projects/Phantom/.venv/bin/python s3_target.py
OMP_NUM_THREADS=1 nice -n 19 /Users/saurabhdxt/Projects/Phantom/.venv/bin/python s4_tsdiff.py
OMP_NUM_THREADS=1 nice -n 19 /Users/saurabhdxt/Projects/Phantom/.venv/bin/python s5_extras.py
OMP_NUM_THREADS=1 nice -n 19 /Users/saurabhdxt/Projects/Phantom/.venv/bin/python s6_sentinels.py
OMP_NUM_THREADS=1 nice -n 19 /Users/saurabhdxt/Projects/Phantom/.venv/bin/python s7_report.py
```

### dd_common.py

```python
import glob, os, resource
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.path.dirname(os.path.abspath(__file__))
COLS = pq.read_schema(f"{D}/scored.parquet").names
STR = [c for c in COLS if pa.types.is_string(pq.read_schema(f"{D}/scored.parquet").field(c).type)]
TS = [c for c in COLS if pa.types.is_timestamp(pq.read_schema(f"{D}/scored.parquet").field(c).type)]
NUM = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt"]
MONTHS = sorted(os.path.basename(f)[4:11] for f in glob.glob(f"{D}/adm_*.parquet"))
FLT = [c for c in COLS if c.endswith("_flt")]

def _fix(t):
    return t.unify_dictionaries() if any(pa.types.is_dictionary(f.type) for f in t.schema) else t

def load_adm(cols, months=None, with_month=True):
    """Admissible training departures (cache from s0), only `cols`, plus 'month' string."""
    months = months or MONTHS
    tabs = []
    for m in months:
        t = pq.read_table(f"{D}/adm_{m}.parquet", columns=cols, read_dictionary=[c for c in cols if c in STR])
        if with_month:
            t = t.append_column("month", pa.array([m] * t.num_rows, pa.string()))
        tabs.append(t)
    t = _fix(pa.concat_tables(tabs))
    return t.to_pandas()

def load_scored(cols):
    t = pq.read_table(f"{D}/scored.parquet", columns=cols, read_dictionary=[c for c in cols if c in STR])
    return _fix(t).to_pandas()

Q = [0.001, 0.01, 0.10, 0.50, 0.90, 0.99, 0.999]
QN = ["p0.1", "p1", "p10", "p50", "p90", "p99", "p99.9"]
def quant(x):
    x = pd.Series(x).dropna().astype("float64").to_numpy()
    if len(x) == 0:
        return {"n": 0}
    q = np.quantile(x, Q)
    d = {"n": int(len(x)), "min": float(x.min())}
    d.update({k: float(v) for k, v in zip(QN, q)})
    d.update({"max": float(x.max()), "mean": float(x.mean())})
    return d

def ts_seconds(s):
    """timestamp Series -> float seconds since epoch (NaN for null)."""
    naive = s.dt.tz_convert("UTC").dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
    return (naive.astype("datetime64[us]").astype("int64").astype("float64") / 1e6).where(s.notna())

def ts_us(s):
    """timestamp Series (no nulls) -> numpy int64 microseconds since epoch."""
    naive = s.dt.tz_convert("UTC").dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
    return naive.astype("datetime64[us]").astype("int64").to_numpy()

def rss():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)

def decile_table(feat, y, nbins=10):
    """Bins by feature quantile; returns per-bin count, feat range, target mean/median."""
    f = pd.Series(feat).astype("float64"); y = pd.Series(y).astype("float64")
    ok = f.notna() & y.notna(); f = f[ok]; y = y[ok]
    if len(f) == 0:
        return []
    try:
        b = pd.qcut(f, nbins, duplicates="drop")
    except ValueError:
        return []
    g = pd.DataFrame({"b": b, "f": f, "y": y}).groupby("b", observed=True)
    out = []
    for k, gg in g:
        out.append({"bin": str(k), "n": int(len(gg)), "f_min": float(gg.f.min()), "f_max": float(gg.f.max()),
                    "y_mean": round(float(gg.y.mean()), 1), "y_median": float(gg.y.median()), "y_p99": float(gg.y.quantile(0.99))})
    return out

def corr(feat, y):
    f = pd.Series(feat).astype("float64"); y = pd.Series(y).astype("float64")
    ok = f.notna() & y.notna(); f = f[ok]; y = y[ok]
    if len(f) < 3 or f.std() == 0:
        return {"n": int(len(f)), "pearson": None, "spearman": None}
    return {"n": int(len(f)), "pearson": round(float(f.corr(y)), 4), "spearman": round(float(f.corr(y, method="spearman")), 4)}
```

### s0_build_cache.py

```python
"""S0: per-month admissible-departure cache + raw-level null/row stats.
Admissible = PHASE_mvt=="DEP" & TAXITIME/BLOCK/MVT non-null & TAXITIME>0, then
drop duplicated MVT_ID_mvt keeping first (file order = chronological).
Also: scored mask on ranking.parquet (MVT_ID in submitting), unmatched stratum = AOBT_3_flt null.
"""
import json, glob, os, resource, sys
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

RAW = "/Users/saurabhdxt/Projects/prc-challenge/data/raw"
OUT = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(f"{RAW}/training_2025-*.parquet"))
assert len(files) == 12, files
cols = pq.read_schema(files[0]).names
assert len(cols) == 30, len(cols)

raw_rows = 0; raw_null = {c: 0 for c in cols}; phase = {}
adm_before_dedup = 0; seen = set(); dup_dropped = 0; adm_rows = 0
drop = {"not_DEP": 0, "null_taxi_block_mvt": 0, "taxi_le_0": 0}
per_month = {}
for f in files:
    t = pq.read_table(f)
    n = t.num_rows; raw_rows += n
    for c in cols:
        raw_null[c] += t[c].null_count
    vc = pc.value_counts(t["PHASE_mvt"])
    for r in vc.to_pylist():
        phase[str(r["values"])] = phase.get(str(r["values"]), 0) + r["counts"]
    is_dep = pc.equal(t["PHASE_mvt"], "DEP")
    is_dep = pc.fill_null(is_dep, False)
    drop["not_DEP"] += n - pc.sum(is_dep).as_py()
    nn = pc.and_(pc.and_(pc.is_valid(t["TAXITIME_SEC_mvt"]), pc.is_valid(t["BLOCK_TIME_UTC_mvt"])), pc.is_valid(t["MVT_TIME_UTC_mvt"]))
    m1 = pc.and_(is_dep, nn)
    drop["null_taxi_block_mvt"] += pc.sum(is_dep).as_py() - pc.sum(m1).as_py()
    pos = pc.fill_null(pc.greater(t["TAXITIME_SEC_mvt"], 0), False)
    m2 = pc.and_(m1, pos)
    drop["taxi_le_0"] += pc.sum(m1).as_py() - pc.sum(m2).as_py()
    a = t.filter(m2)
    adm_before_dedup += a.num_rows
    ids = a["MVT_ID_mvt"].to_pylist()
    keep = []
    for i, k in enumerate(ids):
        if k in seen:
            dup_dropped += 1; keep.append(False)
        else:
            seen.add(k); keep.append(True)
    a = a.filter(pa.array(keep))
    adm_rows += a.num_rows
    mon = os.path.basename(f)[9:16]
    per_month[mon] = {"raw": n, "admissible": a.num_rows}
    pq.write_table(a, f"{OUT}/adm_{mon}.parquet")
    del t, a
    print(mon, n, a_rows := per_month[mon]["admissible"], "rss_MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, file=sys.stderr)

# ranking / submitting
r = pq.read_table(f"{RAW}/ranking.parquet")
s = pq.read_table(f"{RAW}/submitting.parquet")
sub_ids = set(s["MVT_ID_mvt"].to_pylist())
scored = pa.array([k in sub_ids for k in r["MVT_ID_mvt"].to_pylist()])
rs = r.filter(scored)
assert rs.num_rows == 344841, rs.num_rows
# cross-check: scored == DEP & TAXITIME null?
alt = pc.and_(pc.fill_null(pc.equal(r["PHASE_mvt"], "DEP"), False), pc.is_null(r["TAXITIME_SEC_mvt"]))
rank_stats = {
    "ranking_rows": r.num_rows, "submitting_rows": s.num_rows, "scored_rows": rs.num_rows,
    "ranking_DEP_null_taxi_rows": pc.sum(alt).as_py(),
    "scored_equals_DEP_null_taxi": pc.all(pc.equal(scored, alt)).as_py(),
    "submitting_ids_unique": len(sub_ids) == s.num_rows,
    "submitting_ids_all_in_ranking": len(sub_ids & set(r["MVT_ID_mvt"].to_pylist())) == len(sub_ids),
    "submitting_TAXITIME_null_count": s["TAXITIME_SEC_mvt"].null_count,
    "scored_ids_in_training_admissible": len(sub_ids & seen),
    "ranking_phase_counts": {str(x["values"]): x["counts"] for x in pc.value_counts(r["PHASE_mvt"]).to_pylist()},
    "ranking_null_pct_all_rows": {c: round(100*r[c].null_count/r.num_rows, 4) for c in cols},
}
pq.write_table(rs, f"{OUT}/scored.parquet")
unm = pc.is_null(rs["AOBT_3_flt"])
rsu = rs.filter(unm)
flt = [c for c in cols if c.endswith("_flt")]
all_flt_null = None
for c in flt:
    x = pc.is_null(rs[c]); all_flt_null = x if all_flt_null is None else pc.and_(all_flt_null, x)
any_flt_null = None
for c in flt:
    x = pc.is_null(rs[c]); any_flt_null = x if any_flt_null is None else pc.or_(any_flt_null, x)
rank_stats.update({
    "unmatched_rows": rsu.num_rows, "unmatched_pct": round(100*rsu.num_rows/rs.num_rows, 4),
    "rows_all_flt_null": pc.sum(all_flt_null).as_py(),
    "rows_any_flt_null": pc.sum(any_flt_null).as_py(),
    "unmatched_equals_all_flt_null": pc.all(pc.equal(unm, all_flt_null)).as_py(),
    "scored_null_pct": {c: round(100*rs[c].null_count/rs.num_rows, 4) for c in cols},
    "unmatched_null_pct": {c: round(100*rsu[c].null_count/rsu.num_rows, 4) for c in cols},
})
out = {"columns": cols, "raw_rows": raw_rows, "raw_null_pct": {c: round(100*raw_null[c]/raw_rows, 4) for c in cols},
       "raw_phase_counts": phase, "drop": drop, "adm_before_dedup": adm_before_dedup, "dup_dropped": dup_dropped,
       "adm_rows": adm_rows, "per_month": per_month, "ranking": rank_stats,
       "peak_rss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6}
json.dump(out, open(f"{OUT}/s0.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("raw_null_pct",)}, indent=1)[:6000])
```

### s1_profile.py

```python
"""S1: per-column profile on (a) admissible training, all 12 months, (b) scored rows, (c) unmatched stratum."""
import json, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from dd_common import *

s0 = json.load(open(f"{D}/s0.json"))
schema = pq.read_schema(f"{D}/scored.parquet")
out = {}
sc_all = load_scored(COLS)
unm_mask = sc_all["AOBT_3_flt"].isna()
n_sc = len(sc_all); n_un = int(unm_mask.sum())
N_ADM = s0["adm_rows"]

def top(s, k=10):
    vc = s.value_counts(dropna=True).head(k)
    n = s.notna().sum()
    return [{"value": str(i), "count": int(c), "share_pct": round(100 * c / n, 3)} for i, c in vc.items()]

for c in COLS:
    r = {"dtype": str(schema.field(c).type), "null_pct_train_raw": s0["raw_null_pct"][c],
         "null_pct_scored": s0["ranking"]["scored_null_pct"][c], "null_pct_unmatched": s0["ranking"]["unmatched_null_pct"][c]}
    a = load_adm([c], with_month=False)[c]
    r["null_pct_train_adm"] = round(100 * a.isna().mean(), 4)
    r["n_unique_train_adm"] = int(a.nunique(dropna=True)); r["n_unique_scored"] = int(sc_all[c].nunique(dropna=True))
    if c in STR:
        r["top10_train_adm"] = top(a); r["top10_scored"] = top(sc_all[c]); r["top5_unmatched"] = top(sc_all.loc[unm_mask, c], 5)
        ls = a.dropna().astype(str).str.len()
        r["len_min_max_train"] = [int(ls.min()), int(ls.max())] if len(ls) else None
        r["empty_or_blank_train"] = int((a.dropna().astype(str).str.strip() == "").sum())
        r["n_levels_scored_unseen_in_train_adm"] = int((~sc_all[c].dropna().astype(str).isin(set(a.dropna().astype(str).unique()))).sum())
    elif c in TS:
        for tag, s in (("train_adm", a), ("scored", sc_all[c])):
            v = s.dropna()
            if len(v) == 0:
                r[tag] = {"n": 0}; continue
            us = ts_us(v)
            r[tag] = {"n": int(len(v)), "min": str(v.min()), "max": str(v.max()),
                      "pct_whole_second": round(100 * float((us % 1_000_000 == 0).mean()), 3),
                      "pct_on_60s_grid": round(100 * float((us % 60_000_000 == 0).mean()), 3),
                      "pct_on_300s_grid": round(100 * float((us % 300_000_000 == 0).mean()), 3),
                      "pct_on_3600s_grid": round(100 * float((us % 3_600_000_000 == 0).mean()), 3)}
    else:
        for tag, s in (("train_adm", a), ("scored", sc_all[c])):
            v = s.dropna()
            q = quant(v)
            if len(v):
                q["pct_integer_valued"] = round(100 * float((v.astype("float64") % 1 == 0).mean()), 3)
            r[tag] = q
    out[c] = r
    print(c, "rss", rss(), file=sys.stderr)
    del a

# id/link checks
a = load_adm(["MVT_ID_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt"])
out["_checks"] = {
    "MVT_ID_unique_in_adm": bool(a.MVT_ID_mvt.is_unique),
    "MVT_ID_monotone_with_MVT_TIME_spearman": round(float(a.MVT_ID_mvt.corr(ts_seconds(a.MVT_TIME_UTC_mvt), method="spearman")), 4),
    "FLIGHT_ID_dup_rows_in_adm_pct": round(100 * float(a.FLIGHT_ID_mvt.dropna().duplicated().mean()), 4),
    "scored_MVT_ID_unique": bool(sc_all.MVT_ID_mvt.is_unique),
    "scored_MVT_TIME_min": str(sc_all.MVT_TIME_UTC_mvt.min()), "scored_MVT_TIME_max": str(sc_all.MVT_TIME_UTC_mvt.max()),
    "scored_rows_by_month": {str(k): int(v) for k, v in sc_all.MVT_TIME_UTC_mvt.dt.to_period("M").value_counts().sort_index().items()},
    "unmatched_by_month_scored": {str(k): int(v) for k, v in sc_all.loc[unm_mask, "MVT_TIME_UTC_mvt"].dt.to_period("M").value_counts().sort_index().items()},
    "scored_ADEP_by_unmatched": {str(k): int(v) for k, v in sc_all.loc[unm_mask, "ADEP_mvt"].value_counts().items()},
    "scored_ADEP_counts": {str(k): int(v) for k, v in sc_all["ADEP_mvt"].value_counts().items()},
    "cross_col_consistency_scored_pct_equal": {
        "ADEP_mvt==ADEP_flt": round(100 * float((sc_all.ADEP_mvt.astype(str) == sc_all.ADEP_flt.astype(str))[~unm_mask].mean()), 3),
        "ADES_mvt==ADES_flt": round(100 * float((sc_all.ADES_mvt.astype(str) == sc_all.ADES_flt.astype(str))[~unm_mask].mean()), 3),
        "ADES_flt==ADES_FILED_flt": round(100 * float((sc_all.ADES_flt.astype(str) == sc_all.ADES_FILED_flt.astype(str))[~unm_mask].mean()), 3),
        "AIRCRAFT_TYPE_mvt==AIRCRAFT_TYPE_flt": round(100 * float((sc_all.AIRCRAFT_TYPE_mvt.astype(str) == sc_all.AIRCRAFT_TYPE_flt.astype(str))[~unm_mask].mean()), 3),
        "FLIGHT_mvt==CALLSIGN_flt": round(100 * float((sc_all.FLIGHT_mvt.astype(str) == sc_all.CALLSIGN_flt.astype(str))[~unm_mask].mean()), 3),
        "FLIGHT_RULE_mvt==FLIGHT_RULE_flt": round(100 * float((sc_all.FLIGHT_RULE_mvt.astype(str) == sc_all.FLIGHT_RULE_flt.astype(str))[~unm_mask].mean()), 3),
    },
    "peak_rss_MB": rss(),
}
# training unmatched rate per month (admissible)
u = load_adm(["AOBT_3_flt"])
out["_checks"]["train_adm_unmatched_pct_by_month"] = {m: round(100 * float(v), 3) for m, v in u.groupby("month")["AOBT_3_flt"].apply(lambda s: s.isna().mean()).items()}
out["_checks"]["train_adm_unmatched_pct_all"] = round(100 * float(u.AOBT_3_flt.isna().mean()), 4)
json.dump(out, open(f"{D}/s1.json", "w"), indent=1, default=str)
print("S1 done rss", rss())
```

### s2_drift.py

```python
"""S2: drift — 2025 Jan+Jul admissible departures (A) vs 2026 scored rows (B)."""
import json, sys
import numpy as np, pandas as pd
from dd_common import *

A = load_adm(COLS, months=["2025-01", "2025-07"])
B = load_scored(COLS)
vocab_all = {}
out = {"nA": len(A), "nB": len(B), "categorical": {}, "numeric": {}, "timestamp_offsets": {}, "calendar": {}}

def shares(s):
    return s.value_counts(dropna=True, normalize=True)

for c in STR:
    if c == "PHASE_mvt":
        pass
    sa, sb = shares(A[c].astype("str")), shares(B[c].astype("str"))
    idx = sa.index.union(sb.index)
    pa_, pb_ = sa.reindex(idx, fill_value=0), sb.reindex(idx, fill_value=0)
    tvd = 0.5 * float((pa_ - pb_).abs().sum())
    comb = (pa_ + pb_).sort_values(ascending=False)
    top = comb.index[:15]
    vocab = set(load_adm([c], with_month=False)[c].dropna().astype(str).unique())
    bnn = B[c].dropna().astype(str)
    out["categorical"][c] = {
        "null_pct_A": round(100 * A[c].isna().mean(), 4), "null_pct_B": round(100 * B[c].isna().mean(), 4),
        "n_unique_A": int(A[c].nunique()), "n_unique_B": int(B[c].nunique()), "tvd": round(tvd, 4),
        "B_rows_unseen_in_A_pct": round(100 * float((~bnn.isin(set(sa.index))).mean()), 3),
        "B_rows_unseen_in_all2025_pct": round(100 * float((~bnn.isin(vocab)).mean()), 3),
        "B_levels_unseen_in_all2025": int((~pd.Series(bnn.unique()).isin(vocab)).sum()),
        "levels": [{"value": str(v), "A_pct": round(100 * float(pa_[v]), 3), "B_pct": round(100 * float(pb_[v]), 3),
                    "delta_pp": round(100 * float(pb_[v] - pa_[v]), 3)} for v in top],
    }
    print(c, "tvd", round(tvd, 4), "rss", rss(), file=sys.stderr)

for c in ["MVT_ID_mvt", "FLIGHT_ID_mvt"]:
    out["numeric"][c] = {"A": quant(A[c]), "B": quant(B[c])}

ref = ts_seconds(A["MVT_TIME_UTC_mvt"]); refB = ts_seconds(B["MVT_TIME_UTC_mvt"])
for c in TS:
    if c == "MVT_TIME_UTC_mvt":
        continue
    da = ts_seconds(A[c]) - ref; db = ts_seconds(B[c]) - refB
    out["timestamp_offsets"][f"{c} - MVT_TIME_UTC_mvt (s)"] = {"A": quant(da), "B": quant(db),
        "A_pct_within_60s_of_zero": round(100 * float((da.abs() <= 60).mean()), 3) if da.notna().any() else None,
        "B_pct_within_60s_of_zero": round(100 * float((db.abs() <= 60).mean()), 3) if db.notna().any() else None}
# also offsets relative to SCHED (pure serve-time)
refS = ts_seconds(A["SCHED_TIME_UTC_mvt"]); refSB = ts_seconds(B["SCHED_TIME_UTC_mvt"])
for c in ["LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt"]:
    da = ts_seconds(A[c]) - refS; db = ts_seconds(B[c]) - refSB
    out["timestamp_offsets"][f"{c} - SCHED_TIME_UTC_mvt (s)"] = {"A": quant(da), "B": quant(db),
        "A_pct_within_60s_of_zero": round(100 * float((da.abs() <= 60).mean()), 3) if da.notna().any() else None,
        "B_pct_within_60s_of_zero": round(100 * float((db.abs() <= 60).mean()), 3) if db.notna().any() else None}

def cal(df):
    t = df["MVT_TIME_UTC_mvt"]
    return {"hour_pct": {int(k): round(100 * float(v), 3) for k, v in t.dt.hour.value_counts(normalize=True).sort_index().items()},
            "dow_pct": {int(k): round(100 * float(v), 3) for k, v in t.dt.dayofweek.value_counts(normalize=True).sort_index().items()},
            "month_pct": {int(k): round(100 * float(v), 3) for k, v in t.dt.month.value_counts(normalize=True).sort_index().items()},
            "rows_per_day_mean": round(float(t.dt.floor("D").value_counts().mean()), 1),
            "unmatched_pct": round(100 * float(df["AOBT_3_flt"].isna().mean()), 3),
            "unmatched_pct_by_month": {int(k): round(100 * float(v), 3) for k, v in df["AOBT_3_flt"].isna().groupby(t.dt.month).mean().items()},
            "unmatched_pct_by_airport": {str(k): round(100 * float(v), 3) for k, v in df["AOBT_3_flt"].isna().groupby(df["ADEP_mvt"].astype(str)).mean().items()}}
out["calendar"] = {"A": cal(A), "B": cal(B)}
ha = pd.Series(out["calendar"]["A"]["hour_pct"]); hb = pd.Series(out["calendar"]["B"]["hour_pct"])
out["calendar"]["hour_tvd"] = round(0.5 * float((ha.reindex(range(24), fill_value=0) - hb.reindex(range(24), fill_value=0)).abs().sum()) / 100, 4)
# per-airport row share and per-airport x month
out["calendar"]["airport_month_rows"] = {
    "A": {f"{a}|{m}": int(n) for (a, m), n in A.groupby([A["ADEP_mvt"].astype(str), A["MVT_TIME_UTC_mvt"].dt.month]).size().items()},
    "B": {f"{a}|{m}": int(n) for (a, m), n in B.groupby([B["ADEP_mvt"].astype(str), B["MVT_TIME_UTC_mvt"].dt.month]).size().items()}}
out["peak_rss_MB"] = rss()
json.dump(out, open(f"{D}/s2.json", "w"), indent=1, default=str)
print("S2 done rss", rss())
```

### s3_target.py

```python
"""S3: relationship to target on admissible training departures (all 12 months).
Categoricals: top-15 levels count/mean/median/p99 + eta^2 + held-out level-mean lookup RMSE
(fit Feb-Jun+Aug-Dec, evaluate Jan+Jul, unseen level -> train global mean). Diagnostic only, not a model.
Numerics: corr + decile table. Calendar of MVT_TIME: hour/dow/month tables."""
import json, sys
import numpy as np, pandas as pd
from dd_common import *

y_all = load_adm(["TAXITIME_SEC_mvt"])
Y = y_all["TAXITIME_SEC_mvt"].astype("float64"); MON = y_all["month"]
test = MON.isin(["2025-01", "2025-07"]).to_numpy()
gm = float(Y[~test].mean())
base_rmse = float(np.sqrt(((Y[test] - gm) ** 2).mean()))
out = {"n": int(len(Y)), "target": quant(Y), "heldout_global_mean_rmse": round(base_rmse, 2), "categorical": {}, "numeric": {}, "calendar": {}}

def cat_stats(s, name, k=15):
    s = s.astype("str")
    df = pd.DataFrame({"x": s, "y": Y})
    g = df.groupby("x", observed=True)["y"]
    agg = g.agg(["size", "mean", "median"]); agg["p99"] = g.quantile(0.99)
    agg = agg.sort_values("size", ascending=False)
    n_lv = len(agg)
    # eta^2 (in-sample)
    tot = float(((Y - Y.mean()) ** 2).sum()); btw = float((agg["size"] * (agg["mean"] - Y.mean()) ** 2).sum())
    # held-out level mean lookup
    tr = df[~test]; te = df[test]
    lm = tr.groupby("x", observed=True)["y"].mean()
    pred = te["x"].map(lm).astype("float64").fillna(gm)
    ho = float(np.sqrt(((te["y"] - pred) ** 2).mean()))
    unseen = float(te["x"].map(lm).isna().mean())
    r = {"n_levels": int(n_lv), "null_pct": round(100 * float(s.isna().mean()), 4), "eta2_in_sample": round(btw / tot, 4),
         "heldout_lookup_rmse": round(ho, 2), "heldout_rmse_gain_vs_global": round(base_rmse - ho, 2),
         "heldout_unseen_level_row_pct": round(100 * unseen, 3),
         "top15": [{"value": str(i), "n": int(r_["size"]), "mean": round(float(r_["mean"]), 1), "median": float(r_["median"]), "p99": float(r_["p99"])}
                   for i, r_ in agg.head(k).iterrows()]}
    return r

for c in STR:
    s = load_adm([c], with_month=False)[c]
    out["categorical"][c] = cat_stats(s, c)
    print(c, out["categorical"][c]["heldout_lookup_rmse"], "rss", rss(), file=sys.stderr)
    del s

for c in ["MVT_ID_mvt", "FLIGHT_ID_mvt"]:
    s = load_adm([c], with_month=False)[c].astype("float64")
    out["numeric"][c] = {"corr": corr(s, Y), "deciles": decile_table(s, Y)}

t = load_adm(["MVT_TIME_UTC_mvt"], with_month=False)["MVT_TIME_UTC_mvt"]
for nm, s in (("hour_utc", t.dt.hour), ("dow", t.dt.dayofweek), ("month", t.dt.month), ("minute_of_hour_bucket5", (t.dt.minute // 5) * 5)):
    r = cat_stats(s.astype("int64").astype("str"), nm, k=31)
    out["calendar"][nm] = r
# also SCHED hour
ts_ = load_adm(["SCHED_TIME_UTC_mvt"], with_month=False)["SCHED_TIME_UTC_mvt"]
out["calendar"]["sched_hour_utc"] = cat_stats(ts_.dt.hour.astype("int64").astype("str"), "sched_hour", k=24)
out["peak_rss_MB"] = rss()
json.dump(out, open(f"{D}/s3.json", "w"), indent=1, default=str)
print("S3 done rss", rss())
```

### s4_tsdiff.py

```python
"""S4: every pairwise timestamp difference among the 9 timestamp columns.
For each unordered pair X-Y (X later in list order, reported as X - Y):
  computable on scored (both non-null there), null% scored, null% train_adm, quantiles train_adm & scored,
  pct |diff|<=60s (train), corr with target, decile table (train_adm). Plus the unmatched-stratum view of SCHED-MVT."""
import json, sys, itertools
import numpy as np, pandas as pd
from dd_common import *

TSC = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_1_flt", "ARVT_3_flt"]
A = load_adm(TSC + ["TAXITIME_SEC_mvt"])
B = load_scored(TSC)
Y = A["TAXITIME_SEC_mvt"].astype("float64")
unmA = A["AOBT_3_flt"].isna(); unmB = B["AOBT_3_flt"].isna()
SA = {c: ts_seconds(A[c]) for c in TSC}; SB = {c: ts_seconds(B[c]) for c in TSC}
out = {"n_train_adm": len(A), "n_scored": len(B), "pairs": {}, "stratum": {}}
for x, y in itertools.combinations(TSC, 2):
    da = SA[x] - SA[y]; db = SB[x] - SB[y]
    name = f"{x} - {y}"
    r = {"computable_on_scored": bool(db.notna().any()), "null_pct_scored": round(100 * float(db.isna().mean()), 3),
         "null_pct_train_adm": round(100 * float(da.isna().mean()), 3),
         "train_adm": quant(da), "scored": quant(db),
         "train_pct_abs_le_60s": round(100 * float((da.abs() <= 60).mean()), 3),
         "train_pct_exact_zero": round(100 * float((da == 0).mean()), 3),
         "train_pct_equals_target": round(100 * float((da == Y).mean()), 3),
         "corr_target": corr(da, Y)}
    if r["computable_on_scored"]:
        r["deciles"] = decile_table(da, Y)
        r["scored_pct_abs_le_60s"] = round(100 * float((db.abs() <= 60).mean()), 3)
    out["pairs"][name] = r
    print(name, r["corr_target"], "rss", rss(), file=sys.stderr)

# stratum view: SCHED - MVT is the only *_flt-free offset; show its relation to target inside/outside unmatched
d = SA["MVT_TIME_UTC_mvt"] - SA["SCHED_TIME_UTC_mvt"]
bs = SA["BLOCK_TIME_UTC_mvt"] - SA["SCHED_TIME_UTC_mvt"]
for tag, m in (("unmatched", unmA), ("matched", ~unmA)):
    out["stratum"][tag] = {"n": int(m.sum()), "target": quant(Y[m]),
                           "MVT-SCHED": quant(d[m]), "corr_MVT-SCHED_vs_target": corr(d[m], Y[m]),
                           "deciles_MVT-SCHED": decile_table(d[m], Y[m]),
                           "BLOCK-SCHED": quant(bs[m]), "pct_BLOCK_within_60s_of_SCHED": round(100 * float((bs[m].abs() <= 60).mean()), 3),
                           "pct_BLOCK_exactly_SCHED": round(100 * float((bs[m] == 0).mean()), 3)}
    # tail share
    out["stratum"][tag]["pct_target_gt_3600"] = round(100 * float((Y[m] > 3600).mean()), 3)
    out["stratum"][tag]["pct_target_gt_10800"] = round(100 * float((Y[m] > 10800).mean()), 3)
out["stratum"]["scored"] = {"unmatched_n": int(unmB.sum()), "MVT-SCHED_unmatched": quant((SB["MVT_TIME_UTC_mvt"] - SB["SCHED_TIME_UTC_mvt"])[unmB]),
                            "MVT-SCHED_matched": quant((SB["MVT_TIME_UTC_mvt"] - SB["SCHED_TIME_UTC_mvt"])[~unmB])}
# what share of the target's variance / squared error sits in the unmatched stratum
gm = float(Y.mean())
se = (Y - gm) ** 2
out["stratum"]["share_of_total_squared_error_from_global_mean_in_unmatched_pct"] = round(100 * float(se[unmA].sum() / se.sum()), 2)
out["peak_rss_MB"] = rss()
json.dump(out, open(f"{D}/s4.json", "w"), indent=1, default=str)
print("S4 done rss", rss())
```

### s5_extras.py

```python
"""S5: per-airport runway share drift (2025 Jan+Jul adm vs 2026 scored), per-airport stand vocabulary coverage,
the 14 partially-null _flt rows, and per-airport target stats for RUNWAY x ADEP top levels."""
import json, sys
import numpy as np, pandas as pd
from dd_common import *
A = load_adm(["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt"], months=["2025-01", "2025-07"])
B = load_scored(["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AOBT_3_flt", "FLIGHT_RULE_flt", "WK_TBL_CAT_flt", "LOBT_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt"])
out = {"runway_by_airport": {}, "stand_by_airport": {}}
for ap in sorted(A.ADEP_mvt.astype(str).unique()):
    a = A[A.ADEP_mvt.astype(str) == ap]; b = B[B.ADEP_mvt.astype(str) == ap]
    sa = a.RUNWAY_mvt.astype(str).value_counts(normalize=True); sb = b.RUNWAY_mvt.astype(str).value_counts(normalize=True)
    idx = sa.index.union(sb.index); pa_ = sa.reindex(idx, fill_value=0); pb_ = sb.reindex(idx, fill_value=0)
    tvd = 0.5 * float((pa_ - pb_).abs().sum())
    lv = (pa_ + pb_).sort_values(ascending=False).index[:8]
    out["runway_by_airport"][ap] = {"nA": len(a), "nB": len(b), "tvd": round(tvd, 4),
        "levels": [(str(v), round(100 * float(pa_[v]), 2), round(100 * float(pb_[v]), 2)) for v in lv],
        "B_runways_unseen_in_A": sorted(set(sb.index) - set(sa.index))}
    va = set(a.STAND_mvt.dropna().astype(str)); vb = b.STAND_mvt.dropna().astype(str)
    out["stand_by_airport"][ap] = {"n_stands_A": len(va), "n_stands_B": int(vb.nunique()), "B_rows_unseen_in_A_pct": round(100 * float((~vb.isin(va)).mean()), 3)}
# full-2025 stand vocabulary per airport
Aall = load_adm(["ADEP_mvt", "STAND_mvt"], with_month=False)
for ap in out["stand_by_airport"]:
    va = set(Aall[Aall.ADEP_mvt.astype(str) == ap].STAND_mvt.dropna().astype(str))
    vb = B[B.ADEP_mvt.astype(str) == ap].STAND_mvt.dropna().astype(str)
    out["stand_by_airport"][ap]["n_stands_all2025"] = len(va)
    out["stand_by_airport"][ap]["B_rows_unseen_in_all2025_pct"] = round(100 * float((~vb.isin(va)).mean()), 3)
# the 14 partial rows
p = B[B.AOBT_3_flt.notna() & (B.FLIGHT_RULE_flt.isna() | B.WK_TBL_CAT_flt.isna())]
out["partial_flt_null_rows"] = {"n": len(p), "ADEP": p.ADEP_mvt.astype(str).value_counts().to_dict(),
                                "both_null": int((p.FLIGHT_RULE_flt.isna() & p.WK_TBL_CAT_flt.isna()).sum())}
# same in training
T = load_adm(["AOBT_3_flt", "FLIGHT_RULE_flt", "WK_TBL_CAT_flt"], with_month=False)
out["partial_flt_null_rows"]["train_adm_n"] = int((T.AOBT_3_flt.notna() & (T.FLIGHT_RULE_flt.isna() | T.WK_TBL_CAT_flt.isna())).sum())
# unmatched scored rows: is the FLIGHT_mvt callsign seen in 2025 training at all (could support a lookup)?
F = load_adm(["FLIGHT_mvt"], with_month=False)["FLIGHT_mvt"].dropna().astype(str)
vf = set(F)
u = B[B.AOBT_3_flt.isna()]
out["unmatched_scored_FLIGHT_mvt_seen_in_2025_pct"] = round(100 * float(u.FLIGHT_mvt.dropna().astype(str).isin(vf).mean()), 2)
out["matched_scored_FLIGHT_mvt_seen_in_2025_pct"] = round(100 * float(B[B.AOBT_3_flt.notna()].FLIGHT_mvt.dropna().astype(str).isin(vf).mean()), 2)
out["peak_rss_MB"] = rss()
json.dump(out, open(f"{D}/s5.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=0))
```

### s6_sentinels.py

```python
"""S6: literal sentinel strings ('NA','NaN','nan','NULL','', whitespace) in every string column, train_adm vs scored."""
import json
from dd_common import *
SENT = {"NA", "NaN", "nan", "NULL", "null", "None", "", "-"}
out = {}
B = load_scored(STR)
for c in STR:
    a = load_adm([c], with_month=False)[c].dropna().astype(str)
    b = B[c].dropna().astype(str)
    ca = a[a.str.strip().isin(SENT)].value_counts().to_dict(); cb = b[b.str.strip().isin(SENT)].value_counts().to_dict()
    if ca or cb:
        out[c] = {"train_adm": {k: int(v) for k, v in ca.items()}, "scored": {k: int(v) for k, v in cb.items()}}
# which airports / target for 'NA' runway in training
a = load_adm(["RUNWAY_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt"], with_month=False)
m = a.RUNWAY_mvt.astype(str) == "NA"
out["_NA_runway_train"] = {"n": int(m.sum()), "by_airport": a[m].ADEP_mvt.astype(str).value_counts().to_dict(), "target": quant(a[m].TAXITIME_SEC_mvt)}
bb = load_scored(["RUNWAY_mvt", "ADEP_mvt"]); mb = bb.RUNWAY_mvt.astype(str) == "NA"
out["_NA_runway_scored"] = {"n": int(mb.sum()), "by_airport": bb[mb].ADEP_mvt.astype(str).value_counts().to_dict()}
out["peak_rss_MB"] = rss()
json.dump(out, open(f"{D}/s6.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
```
