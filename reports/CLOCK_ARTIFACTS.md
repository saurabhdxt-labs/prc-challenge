# Off-block clock artifacts, per airport — PRC Data Challenge 2026

Generated 2026-09-08. Every number below is copied from a command output saved under the session
scratchpad (`.../scratchpad/clock/{load,a_dist,b_coincide,c_2026,d_followup,e_events,f_loose}.py`
and their `.out` files). Runs used `OMP_NUM_THREADS=1 nice -n 19 python3.11`; peak RSS of any
run was 0.97 GB. Nothing under the repository was modified except this file.

## 0. Data and definitions

| item | value |
|---|---|
| training rows in (12 files, PHASE=DEP) | 2,085,047 |
| admissible (label/off-block/take-off non-null, taxi>0, dedup MVT_ID) | **2,084,659** |
| SCHED_TIME_UTC_mvt null | 0 |
| target == MVT−BLOCK exactly | 100.0% |
| unmatched (AOBT_3_flt null) | 22,219 (1.1%) |
| every `*_flt` column on unmatched rows | 100.0% null (LOBT, IOBT, EOBT_1, AOBT_3, ARVT_1, ARVT_3) |
| every `*_flt` column on matched rows | 0.0% null |
| sub-second precision in any timestamp, 2025 or 2026 | none (all values whole seconds) — so "±1s" ≡ exact |
| 2026 scored rows (ranking ∩ submitting) | 344,841; BLOCK 100% null; unmatched 5,290 (1.5%); months 2026-01: 152,719, 2026-07: 192,122 |

Notation: `d` = BLOCK − SCHED (s); `ms` = MVT − SCHED (s); `proxy` = MVT − AOBT_3 (NM-clock taxi-out,
matched rows only). "copy" = |d| ≤ 60.

## 1. Findings at a glance

| # | airport | what | rule | fill rate | monthly stability 2025 | visible in 2026? | confidence |
|---|---|---|---|---|---|---|---|
| A1 | LIRF | schedule copied into off-block (known, extended) | unmatched: BLOCK = SCHED + jitter in [−6,+6] s (99.9% of the 722 copies) | 48.5% of unmatched | 32.3%–76.1% by month (never below 32%) | stratum present; 2026 unmatched LIRF `ms` p90/p99 15,793/57,264 vs 2025 15,035/60,291 — same shape; share >3h 19.3% vs 20.9% | solid |
| A1b | LIRF | same rule on **matched** rows | \|d\| ≤ 60 AND \|BLOCK − AOBT_3\| > 300 | 9.1% of LIRF matched (14,423 rows); 17.5% at a 60 s deviation threshold | 8.2%–9.9% every month | cannot be observed (BLOCK null) but rate is flat, so expect ~9% of LIRF matched 2026 targets to be schedule-anchored | solid |
| A2 | LSZH | heliport rows with near-zero target | unmatched, RUNWAY_mvt = `H` (stand `HPW`): target ≤ 15 s on 86.8%; VFR helicopters; BLOCK is written at lift-off | 190 rows/yr = 9.3% of LSZH unmatched; 11.3% of LSZH unmatched have target < 60 | 6–34 rows per month, every month | yes: 78 scored rows have RUNWAY=`H` (39 Jan, 39 Jul); the identifier is populated | solid |
| A3 | EHAM | heliport stands with very short target | unmatched, STAND `HG01`/`HG03` (A139 / DH8A, VFR): target p50 176 / 248 s | 1,403 rows/yr = 34% of EHAM unmatched | 20.9%–52.5% of EHAM unmatched per month, present all 12 months | yes: 148 + 57 scored rows on those stands | solid (operational, not a clock error) |
| M1 | all | mechanism of the unmatched stratum | matched ⇔ \|BLOCK − LOBT\| ≤ 3,606 s (min −3,606, max 3,606 at every airport, 0.0% beyond) | — | — | by construction identical in 2026 | solid |
| M2 | 7 airports | seconds-within-minute fingerprint | EDDM EGLL LEBL LEMD LFPG LIRF LTFM: BLOCK seconds ∈ {0–6, 54–59} only (13 values); EDDF EHAM LSZH uniform over 60 | — | identical in all 12 months | MVT has the same fingerprint in 2026 at every airport (LEBL/LEMD MVT uniform in both years) | solid |
| M3 | LTFM | off-block clock sits ~5 min after NM's AOBT | BLOCK − AOBT_3 p50 +296 s, only 8.7% within 60 s (others 19–26%) | — | 6.4%–10.3% within-60 by month | not observable | solid, a definition offset not a fill |
| — | all others | further fallback-fill rules (BLOCK = LOBT/IOBT/EOBT_1/ARVT) | none: exact-equality rates 0.1–1.5%, identical to SCHED's, and MVT (control) is 0.0% | — | — | — | negative result, solid |
| — | all | round-number / discrete spikes in `d` or in the target | none beyond A1–A3; top exact values ≤ 0.9% share; target 60 s-grid share equals the fingerprint expectation | — | — | — | negative result, solid |
| 2026 | EHAM | 297 unmatched rows with `ms` > 3h | 278 are January; 244 are on 2026-01-02..06; matched rows the same days show NM-clock proxy taxi p50 963–2,352 s, p90 up to 5,411 s, **0.0% > 3h** | — | 2025 disruption days at EHAM: unmatched >3h rows had median target 662 s, 0.0% > 3h, 0.0% copies | it is a disruption, not a fill | solid |

## 2. Q1 — distribution of BLOCK − SCHED, per airport × stratum (2025, all months)

| airport | stratum | n | exact 0 | \|d\|≤60 | \|d\|≤300 | p1 | p10 | p50 | p90 | p99 | d<−300 | d>3h | top-5 exact values : share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | matched | 228185 | 0.1% | 7.0% | 31.2% | -718 | -118 | 545 | 2283 | 7490 | 4.4% | 0.5% | 407:0.1%; 171:0.1%; 96:0.1%; 106:0.1%; 105:0.1% |
| EDDF | unmatched | 1956 | 0.0% | 0.7% | 3.0% | -905 | 2624 | 4734 | 8786 | 33357 | 1.7% | 5.4% | 3914:0.2%; 3689:0.2%; 4557:0.2%; 4639:0.2%; 3706:0.2% |
| EDDM | matched | 166324 | 0.4% | 10.4% | 42.2% | -658 | -184 | 354 | 1914 | 5701 | 6.0% | 0.2% | 55:0.5%; 65:0.5%; 117:0.5%; 62:0.5%; 123:0.5% |
| EDDM | unmatched | 1010 | 0.0% | 0.2% | 1.4% | -965 | 3717 | 4684 | 7390 | 14986 | 2.0% | 2.1% | 3781:0.9%; 3958:0.7%; 3842:0.6%; 3963:0.6%; 4925:0.6% |
| EGLL | matched | 238132 | 0.4% | 9.9% | 40.7% | -844 | -419 | 181 | 1924 | 6844 | 15.4% | 0.3% | -2:0.4%; 0:0.4%; 2:0.4%; -59:0.4%; -60:0.4% |
| EGLL | unmatched | 1414 | 0.1% | 0.4% | 1.0% | 26 | 3725 | 4864 | 8800 | 17500 | 0.3% | 5.4% | 3838:0.7%; 3958:0.6%; 3655:0.5%; 4085:0.5%; 4078:0.5% |
| EHAM | matched | 243865 | 0.1% | 7.0% | 30.3% | -759 | -100 | 568 | 2609 | 8571 | 4.5% | 0.6% | 152:0.1%; 91:0.1%; 131:0.1%; 62:0.1%; 202:0.1% |
| EHAM | unmatched | 4083 | 0.2% | 4.0% | 14.3% | -2155 | -122 | 3747 | 7771 | 19339 | 7.0% | 4.4% | 301:0.3%; -1:0.3%; 5:0.3%; 601:0.3%; 298:0.3% |
| LEBL | matched | 177838 | 0.5% | 10.0% | 38.9% | -657 | -241 | 360 | 2516 | 8218 | 7.7% | 0.4% | 3:0.5%; 1:0.5%; -1:0.5%; 0:0.5%; 2:0.5% |
| LEBL | unmatched | 1862 | 0.1% | 2.0% | 3.1% | -3 | 3657 | 4914 | 9170 | 18145 | 0.3% | 6.3% | 3958:0.6%; 3724:0.5%; 3781:0.5%; 3904:0.4%; 4138:0.4% |
| LEMD | matched | 211295 | 0.4% | 8.6% | 35.5% | -656 | -178 | 475 | 2397 | 7915 | 5.3% | 0.5% | -4:0.4%; 4:0.4%; 5:0.4%; -3:0.4%; 2:0.4% |
| LEMD | unmatched | 947 | 0.0% | 0.3% | 0.5% | 774 | 3779 | 4984 | 8580 | 14261 | 0.1% | 3.8% | 3725:0.7%; 4444:0.6%; 4984:0.6%; 3781:0.6%; 4021:0.5% |
| LFPG | matched | 235725 | 0.4% | 8.3% | 33.6% | -604 | -177 | 543 | 2641 | 8584 | 4.5% | 0.6% | 0:0.4%; 55:0.4%; -2:0.4%; 1:0.4%; 61:0.4% |
| LFPG | unmatched | 3762 | 0.1% | 0.6% | 3.0% | -119 | 2348 | 4675 | 8217 | 19664 | 0.6% | 4.0% | 3665:0.5%; 3901:0.4%; 3776:0.4%; 3714:0.4%; 4078:0.3% |
| LIRF | matched | 159216 | 1.5% | 20.5% | 41.4% | -843 | -356 | 238 | 2341 | 7555 | 12.0% | 0.4% | 2:1.5%; -5:1.5%; 4:1.5%; 0:1.5%; -3:1.5% |
| LIRF | unmatched | 1488 | 4.6% | 48.5% | 48.9% | -10514 | -5 | 5 | 7096 | 13407 | 5.0% | 2.6% | 0:4.6%; -5:4.6%; 5:4.4%; -1:4.2%; 2:4.2% |
| LSZH | matched | 132540 | 0.0% | 5.8% | 27.8% | -616 | -47 | 603 | 2249 | 5988 | 3.5% | 0.2% | 182:0.4%; 185:0.3%; 180:0.3%; 175:0.3%; 178:0.3% |
| LSZH | unmatched | 2054 | 0.0% | 3.8% | 15.4% | -1627 | -548 | 3354 | 6622 | 11541 | 15.6% | 1.3% | -3:0.3%; -1:0.2%; -4:0.2%; -598:0.2%; -458:0.2% |
| LTFM | matched | 269320 | 0.4% | 10.6% | 45.2% | -840 | -415 | 177 | 1914 | 10077 | 13.7% | 0.9% | 2:0.4%; 4:0.4%; 3:0.4%; 59:0.4%; 5:0.4% |
| LTFM | unmatched | 3643 | 0.4% | 9.4% | 40.7% | -786 | -475 | 175 | 4677 | 11915 | 16.4% | 1.3% | -238:0.7%; -179:0.6%; -123:0.6%; 119:0.5%; -120:0.5% |

Reading: the only exact-value spike is LIRF's 0 (and its ±1..±6 neighbours, which are the fingerprint
jitter, §4). The matched-row values at EGLL/LEBL/LEMD/LIRF/LTFM near 0 and ±55..60 are the same
jitter around whole minutes, each ≤ 1.5%. No round-number spikes (300, 600, 900 …) anywhere.

### 2b. Binned mass of BLOCK − SCHED (share of stratum)

| stratum | airport | n | <-3h | -3h..-30m | -30m..-10m | -10m..-5m | -5m..-1m | -60..-1 | 0 | 1..60 | 1m..5m | 5m..10m | 10m..30m | 30m..1h | 1h..3h | >3h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unmatched | EDDF | 1956 | 0.1% | 0.7% | 0.4% | 0.5% | 1.1% | 0.3% | 0.0% | 0.5% | 1.2% | 1.4% | 2.4% | 3.4% | 82.7% | 5.4% |
| unmatched | EDDM | 1010 | 0.0% | 0.6% | 1.0% | 0.4% | 0.4% | 0.2% | 0.0% | 0.0% | 0.8% | 0.4% | 0.9% | 0.8% | 92.5% | 2.1% |
| unmatched | EGLL | 1414 | 0.0% | 0.1% | 0.1% | 0.1% | 0.4% | 0.1% | 0.1% | 0.1% | 0.2% | 0.4% | 1.2% | 0.8% | 90.9% | 5.4% |
| unmatched | EHAM | 4083 | 0.1% | 1.0% | 2.8% | 3.1% | 4.2% | 1.8% | 0.2% | 2.0% | 6.0% | 7.3% | 12.9% | 4.3% | 49.8% | 4.4% |
| unmatched | LEBL | 1862 | 0.0% | 0.1% | 0.1% | 0.2% | 0.3% | 0.9% | 0.1% | 1.0% | 0.8% | 1.8% | 3.1% | 1.0% | 84.5% | 6.3% |
| unmatched | LEMD | 947 | 0.0% | 0.1% | 0.0% | 0.0% | 0.1% | 0.1% | 0.0% | 0.2% | 0.1% | 0.2% | 0.8% | 0.1% | 94.4% | 3.8% |
| unmatched | LFPG | 3762 | 0.1% | 0.1% | 0.2% | 0.3% | 0.6% | 0.2% | 0.1% | 0.4% | 1.8% | 1.5% | 4.1% | 2.3% | 84.4% | 4.0% |
| unmatched | LIRF | 1488 | 1.0% | 2.6% | 1.1% | 0.3% | 0.4% | 22.9% | 4.6% | 21.0% | 0.0% | 0.0% | 0.4% | 0.7% | 42.5% | 2.6% |
| unmatched | LSZH | 2054 | 0.1% | 0.6% | 8.0% | 6.8% | 5.3% | 2.2% | 0.0% | 1.6% | 6.2% | 6.2% | 10.1% | 3.3% | 48.1% | 1.3% |
| unmatched | LTFM | 3643 | 0.0% | 0.2% | 4.6% | 12.0% | 18.4% | 3.8% | 0.4% | 4.9% | 12.9% | 8.3% | 12.5% | 5.5% | 15.2% | 1.3% |
| matched | EDDF | 228185 | 0.0% | 0.2% | 1.4% | 2.9% | 8.2% | 3.2% | 0.1% | 3.7% | 16.0% | 17.3% | 32.3% | 10.9% | 3.5% | 0.5% |
| matched | EDDM | 166324 | 0.0% | 0.1% | 1.3% | 4.8% | 13.1% | 4.5% | 0.4% | 5.1% | 18.9% | 15.9% | 25.1% | 8.0% | 2.6% | 0.2% |
| matched | EGLL | 238132 | 0.0% | 0.0% | 4.3% | 11.4% | 17.2% | 4.5% | 0.4% | 4.6% | 13.7% | 12.0% | 20.8% | 7.5% | 3.2% | 0.3% |
| matched | EHAM | 243865 | 0.0% | 0.3% | 1.3% | 2.9% | 7.3% | 3.1% | 0.1% | 3.7% | 16.1% | 16.8% | 31.3% | 11.5% | 5.1% | 0.6% |
| matched | LEBL | 177838 | 0.0% | 0.0% | 1.5% | 6.4% | 13.0% | 4.4% | 0.5% | 4.9% | 16.0% | 14.2% | 23.7% | 10.1% | 4.9% | 0.4% |
| matched | LEMD | 211295 | 0.0% | 0.0% | 1.3% | 4.1% | 10.5% | 3.7% | 0.4% | 4.3% | 16.6% | 16.5% | 27.9% | 9.9% | 4.5% | 0.5% |
| matched | LFPG | 235725 | 0.0% | 0.0% | 1.1% | 3.5% | 11.1% | 3.5% | 0.4% | 4.1% | 14.4% | 13.8% | 30.3% | 12.2% | 5.1% | 0.6% |
| matched | LIRF | 159216 | 0.0% | 0.1% | 3.5% | 8.7% | 11.8% | 10.2% | 1.5% | 8.5% | 9.2% | 10.6% | 21.8% | 9.6% | 4.3% | 0.4% |
| matched | LSZH | 132540 | 0.0% | 0.1% | 1.0% | 2.4% | 6.0% | 2.6% | 0.0% | 3.2% | 15.9% | 18.6% | 35.4% | 11.2% | 3.4% | 0.2% |
| matched | LTFM | 269320 | 0.0% | 0.1% | 3.9% | 10.0% | 17.9% | 4.6% | 0.4% | 5.1% | 16.9% | 12.5% | 18.1% | 6.2% | 3.5% | 0.9% |

LIRF unmatched is bimodal: a copy mode (48.5%) and a delayed-flight mode (1h..3h, 42.5%); there is
nothing in 1m..1h (1.1% total). The other airports' unmatched rows are a delay pile (§7).

## 3. Q2 — timestamp granularity (chance for a continuous clock: 1.67% on a 60 s grid, 0.33% on 300 s)

| airport | stratum | n | BLOCK_60 | BLOCK_300 | MVT_60 | MVT_300 | SCHED_60 | SCHED_300 | AOBT_3_60 | AOBT_3_300 | EOBT_1_60 | EOBT_1_300 | LOBT_60 | LOBT_300 | BLOCK_60 among copies | BLOCK_60 among non-copies |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | matched | 228185 | 1.7% | 0.3% | 1.8% | 0.4% | 100.0% | 100.0% | 99.0% | 20.2% | 99.6% | 88.4% | 99.6% | 98.4% | 2.5% | 1.6% |
| EDDF | unmatched | 1956 | 2.1% | 0.3% | 1.4% | 0.3% | 100.0% | 100.0% | – | – | – | – | – | – | 0.0% | 2.2% |
| EDDM | matched | 166324 | 8.3% | 1.6% | 8.3% | 1.7% | 100.0% | 100.0% | 99.9% | 21.4% | 99.9% | 85.5% | 99.9% | 97.3% | 11.5% | 8.0% |
| EDDM | unmatched | 1010 | 7.7% | 1.5% | 6.4% | 1.5% | 100.0% | 100.0% | – | – | – | – | – | – | 0.0% | 7.7% |
| EGLL | matched | 238132 | 8.4% | 1.8% | 8.3% | 1.6% | 100.0% | 100.0% | 100.0% | 21.2% | 99.9% | 84.6% | 99.9% | 94.1% | 12.3% | 8.0% |
| EGLL | unmatched | 1414 | 7.8% | 1.6% | 8.8% | 1.9% | 100.0% | 100.0% | – | – | – | – | – | – | 20.0% | 7.7% |
| EHAM | matched | 243865 | 1.7% | 0.3% | 1.7% | 0.3% | 100.0% | 99.9% | 83.8% | 16.9% | 100.0% | 81.6% | 100.0% | 94.7% | 2.7% | 1.6% |
| EHAM | unmatched | 4083 | 3.4% | 1.3% | 4.2% | 0.7% | 100.0% | 99.7% | – | – | – | – | – | – | 9.1% | 3.2% |
| LEBL | matched | 177838 | 8.3% | 1.9% | 1.7% | 0.3% | 100.0% | 100.0% | 99.6% | 20.0% | 99.1% | 75.8% | 99.1% | 92.5% | 11.4% | 8.0% |
| LEBL | unmatched | 1862 | 7.5% | 1.6% | 1.6% | 0.3% | 100.0% | 99.9% | – | – | – | – | – | – | 5.3% | 7.6% |
| LEMD | matched | 211295 | 8.4% | 1.8% | 1.6% | 0.3% | 100.0% | 100.0% | 99.7% | 20.0% | 99.5% | 90.0% | 99.5% | 95.1% | 11.7% | 8.0% |
| LEMD | unmatched | 947 | 9.4% | 1.6% | 2.0% | 0.6% | 100.0% | 100.0% | – | – | – | – | – | – | 0.0% | 9.4% |
| LFPG | matched | 235725 | 8.3% | 1.7% | 8.3% | 1.7% | 100.0% | 100.0% | 99.7% | 20.0% | 99.9% | 87.1% | 99.9% | 94.7% | 12.0% | 7.9% |
| LFPG | unmatched | 3762 | 8.8% | 1.6% | 8.1% | 1.6% | 100.0% | 100.0% | – | – | – | – | – | – | 16.7% | 8.7% |
| LIRF | matched | 159216 | 8.3% | 3.3% | 8.4% | 1.7% | 100.0% | 100.0% | 99.6% | 19.9% | 100.0% | 64.0% | 100.0% | 91.1% | 9.3% | 8.1% |
| LIRF | unmatched | 1488 | 9.2% | 5.4% | 8.1% | 1.1% | 100.0% | 99.5% | – | – | – | – | – | – | 9.6% | 8.9% |
| LSZH | matched | 132540 | 2.0% | 0.3% | 1.7% | 0.3% | 100.0% | 100.0% | 100.0% | 19.9% | 100.0% | 94.3% | 100.0% | 98.2% | 2.6% | 1.9% |
| LSZH | unmatched | 2054 | 2.3% | 0.5% | 2.2% | 0.4% | 100.0% | 91.7% | – | – | – | – | – | – | 1.3% | 2.3% |
| LTFM | matched | 269320 | 8.3% | 1.7% | 8.3% | 1.7% | 100.0% | 99.9% | 98.9% | 19.9% | 99.2% | 93.9% | 99.2% | 93.9% | 11.7% | 7.9% |
| LTFM | unmatched | 3643 | 8.9% | 1.6% | 8.5% | 1.7% | 100.0% | 99.9% | – | – | – | – | – | – | 11.9% | 8.6% |

Reading: **grid inheritance is not a usable tell.** LIRF's copies are on the 60 s grid only 9.6% of
the time — the copy passes through the same seconds jitter as every other LIRF timestamp (§3b, §9).
No airport's unmatched BLOCK differs in granularity from its matched BLOCK.

### 3b. Seconds-within-minute fingerprint (values with share > 0.5%)

| airport | BLOCK 2025 | MVT 2025 | MVT 2026 |
|---|---|---|---|
| EDDF | uniform (60 values) | uniform (60) | uniform (60) |
| EDDM | 13: {0,1,2,3,4,5,6,54,55,56,57,58,59} | same 13 | same 13 |
| EGLL | 13 | 13 | 13 |
| EHAM | uniform (60) | uniform (60) | uniform (60) |
| LEBL | 13 | uniform (60) | uniform (60) |
| LEMD | 13 | uniform (60) | uniform (60) |
| LFPG | 13 | 13 | 13 |
| LIRF | 13 | 13 | 13 |
| LSZH | uniform (60) | uniform (60) | uniform (60) |
| LTFM | 13 | 13 | 13 |

The 13-value set is identical in every month of 2025 for BLOCK and MVT, and identical for MVT in
2026-01 and 2026-07 at all ten airports (D1b/D1c tables in `d_followup.out`). Consequence: at the 13-value
airports the target sits on a 60 s multiple 4.1–4.4% of the time and on a 10 s multiple 12.1–12.6%,
versus 1.6–1.7% / 9.4–10.0% at the uniform airports — that is arithmetic of the fingerprint, not a
rounding artifact in the target.

## 4. Q3 — does BLOCK coincide with any other clock? (matched rows; all `*_flt` are null on unmatched rows)

| airport | n | SCHED eq / ≤60 / ≤300 | LOBT eq / ≤60 / ≤300 | IOBT eq / ≤60 / ≤300 | EOBT_1 eq / ≤60 / ≤300 | AOBT_3 eq / ≤60 / ≤300 | ARVT_1 eq / ≤60 | ARVT_3 eq / ≤60 |
|---|---|---|---|---|---|---|---|---|
| EDDF | 228185 | 0.1 / 7.0 / 31.2 | 0.1 / 8.4 / 38.1 | 0.1 / 8.4 / 38.2 | 0.1 / 9.4 / 42.4 | 0.2 / 21.0 / 85.4 | 0.0 / 0.0 | 0.0 / 0.0 |
| EDDM | 166324 | 0.4 / 10.4 / 42.2 | 0.5 / 12.1 / 49.5 | 0.5 / 12.2 / 49.6 | 0.5 / 14.3 / 58.8 | 1.1 / 25.7 / 82.6 | 0.0 / 0.0 | 0.0 / 0.0 |
| EGLL | 238132 | 0.4 / 9.9 / 40.7 | 0.5 / 11.1 / 45.8 | 0.5 / 11.2 / 45.9 | 0.6 / 13.8 / 53.6 | 0.8 / 19.4 / 73.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| EHAM | 243865 | 0.1 / 7.0 / 30.3 | 0.1 / 8.9 / 39.1 | 0.1 / 8.9 / 39.1 | 0.1 / 9.5 / 41.2 | 0.2 / 23.7 / 81.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| LEBL | 177838 | 0.5 / 10.0 / 38.9 | 0.6 / 12.1 / 46.5 | 0.6 / 12.2 / 47.1 | 0.7 / 14.1 / 52.1 | 1.1 / 26.4 / 81.7 | 0.0 / 0.0 | 0.0 / 0.0 |
| LEMD | 211295 | 0.4 / 8.6 / 35.5 | 0.5 / 10.8 / 44.1 | 0.5 / 10.9 / 44.5 | 0.5 / 13.0 / 51.8 | 1.1 / 26.2 / 86.5 | 0.0 / 0.0 | 0.0 / 0.0 |
| LFPG | 235725 | 0.4 / 8.3 / 33.6 | 0.5 / 11.0 / 44.5 | 0.5 / 11.0 / 44.5 | 0.5 / 11.9 / 45.2 | 0.9 / 21.9 / 73.9 | 0.0 / 0.0 | 0.0 / 0.0 |
| LIRF | 159216 | 1.5 / 20.5 / 41.4 | 1.5 / 21.4 / 48.7 | 1.5 / 21.4 / 48.8 | 1.5 / 24.0 / 62.4 | 0.8 / 18.6 / 65.8 | 0.0 / 0.0 | 0.0 / 0.0 |
| LSZH | 132540 | 0.0 / 5.8 / 27.8 | 0.1 / 7.2 / 34.9 | 0.1 / 7.2 / 34.9 | 0.1 / 7.7 / 35.6 | 0.2 / 24.2 / 82.6 | 0.0 / 0.0 | 0.0 / 0.0 |
| LTFM | 269320 | 0.4 / 10.6 / 45.2 | 0.4 / 11.5 / 48.9 | 0.4 / 11.6 / 49.2 | 0.4 / 11.5 / 48.9 | 0.3 / 8.7 / 44.4 | 0.0 / 0.0 | 0.0 / 0.0 |

Control (take-off vs the same clocks): MVT exact-equality is 0.0% against every clock at every
airport, and ≤ 0.8% within 60 s. The NM clocks are near-copies of each other (SCHED = LOBT exactly on
74.0%, = EOBT_1 on 55.5%, LOBT = IOBT on 99.5% of matched rows), so LOBT/IOBT/EOBT_1 coincidence
rates track SCHED's; nothing exceeds the SCHED rate by more than what the LIRF schedule copy already
explains. **No fallback-fill rule other than LIRF's schedule copy exists.**

### 4b. Where BLOCK deviates from AOBT_3 by > 300 s, what does it equal instead?

| airport | n dev | share of matched | ≤60 of SCHED | ≤60 of EOBT_1 | ≤60 of LOBT | ≤60 of IOBT | taxi>3h | BLOCK on 60 s grid |
|---|---|---|---|---|---|---|---|---|
| EDDF | 33278 | 14.6% | 6.0% | 7.9% | 7.1% | 7.4% | 0.0% | 1.2% |
| EDDM | 28977 | 17.4% | 8.5% | 12.2% | 10.4% | 10.5% | 0.0% | 7.1% |
| EGLL | 64203 | 27.0% | 7.0% | 8.4% | 7.9% | 8.0% | 0.0% | 7.2% |
| EHAM | 46442 | 19.0% | 3.3% | 3.6% | 4.4% | 4.4% | 0.0% | 1.5% |
| LEBL | 32550 | 18.3% | 12.7% | 15.1% | 15.0% | 15.8% | 0.0% | 7.1% |
| LEMD | 28537 | 13.5% | 8.4% | 12.0% | 10.1% | 10.8% | 0.0% | 6.7% |
| LFPG | 61607 | 26.1% | 8.3% | 11.1% | 10.6% | 10.6% | 0.0% | 7.3% |
| LIRF | 54378 | 34.2% | 26.5% | 23.5% | 24.8% | 24.9% | 0.0% | 7.4% |
| LSZH | 23106 | 17.4% | 4.9% | 6.3% | 6.2% | 6.2% | 0.0% | 1.6% |
| LTFM | 149729 | 55.6% | 10.5% | 11.6% | 11.6% | 11.6% | 0.0% | 7.7% |

### 4c. Clock-definition offsets: BLOCK − AOBT_3 on matched rows (s)

| airport | n | p1 | p10 | p25 | p50 | p75 | p90 | p99 | \|x\|≤60 | \|x\|≤300 | x>1800 | x<−1800 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 228185 | -734 | -207 | -44 | 87 | 197 | 288 | 446 | 21.0% | 85.4% | 0.0% | 0.1% |
| EDDM | 166324 | -1326 | -424 | -177 | -2 | 117 | 184 | 415 | 25.7% | 82.6% | 0.0% | 0.4% |
| EGLL | 238132 | -1437 | -302 | -124 | 54 | 234 | 417 | 900 | 19.4% | 73.0% | 0.1% | 0.7% |
| EHAM | 243865 | -273 | -101 | -3 | 107 | 241 | 431 | 1236 | 23.7% | 81.0% | 0.3% | 0.0% |
| LEBL | 177838 | -1203 | -359 | -179 | -54 | 65 | 234 | 535 | 26.4% | 81.7% | 0.0% | 0.3% |
| LEMD | 211295 | -1077 | -302 | -122 | 2 | 120 | 234 | 366 | 26.2% | 86.5% | 0.0% | 0.3% |
| LFPG | 235725 | -1022 | -366 | -182 | -4 | 123 | 354 | 781 | 21.9% | 73.9% | 0.0% | 0.3% |
| LIRF | 159216 | -2280 | -717 | -360 | -118 | 61 | 185 | 481 | 18.6% | 65.8% | 0.0% | 1.6% |
| LSZH | 132540 | -785 | -356 | -183 | -33 | 96 | 225 | 374 | 24.2% | 82.6% | 0.0% | 0.1% |
| LTFM | 269320 | -1144 | -240 | 57 | 296 | 479 | 606 | 1020 | 8.7% | 44.4% | 0.6% | 0.3% |

LTFM's off-block runs a median 296 s after NM's AOBT (a different event definition, stable by month:
BLOCK within 60 s of AOBT_3 is 6.4%–10.3% in every month of 2025). LIRF's p1/p10 (−2280/−717) are the
schedule copies sitting before the true push.

## 5. Q4 — monthly stability through 2025

### 5a. Unmatched rows — n per month

| airport | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 164 | 74 | 84 | 72 | 157 | 289 | 435 | 193 | 173 | 127 | 92 | 96 |
| EDDM | 65 | 27 | 31 | 25 | 39 | 159 | 172 | 136 | 115 | 76 | 35 | 130 |
| EGLL | 128 | 38 | 91 | 36 | 68 | 127 | 338 | 112 | 204 | 91 | 54 | 127 |
| EHAM | 293 | 252 | 262 | 236 | 284 | 415 | 515 | 366 | 537 | 378 | 230 | 315 |
| LEBL | 66 | 45 | 67 | 86 | 132 | 213 | 430 | 226 | 247 | 194 | 67 | 89 |
| LEMD | 32 | 17 | 35 | 55 | 65 | 103 | 277 | 76 | 95 | 85 | 47 | 60 |
| LFPG | 215 | 123 | 106 | 150 | 265 | 452 | 661 | 400 | 389 | 316 | 372 | 313 |
| LIRF | 60 | 58 | 59 | 70 | 99 | 185 | 337 | 197 | 168 | 115 | 52 | 88 |
| LSZH | 153 | 85 | 130 | 107 | 213 | 242 | 323 | 271 | 205 | 154 | 88 | 83 |
| LTFM | 232 | 363 | 196 | 227 | 269 | 285 | 425 | 319 | 321 | 366 | 310 | 330 |

### 5b. Unmatched rows — share with |BLOCK − SCHED| ≤ 60 (the schedule-copy rate)

| airport | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 1.8% | 4.1% | 0.0% | 2.8% | 0.0% | 0.3% | 0.5% | 0.0% | 0.0% | 0.8% | 1.1% | 1.0% |
| EDDM | 0.0% | 0.0% | 3.2% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.3% | 0.0% | 0.0% |
| EGLL | 0.0% | 0.0% | 3.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2.2% | 0.0% | 0.0% |
| EHAM | 1.0% | 4.4% | 5.7% | 5.1% | 4.2% | 6.5% | 3.5% | 5.5% | 3.0% | 3.2% | 4.8% | 2.5% |
| LEBL | 1.5% | 2.2% | 1.5% | 3.5% | 4.5% | 3.3% | 1.2% | 0.4% | 1.2% | 1.0% | 9.0% | 2.2% |
| LEMD | 0.0% | 0.0% | 0.0% | 0.0% | 1.5% | 1.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.7% |
| LFPG | 0.9% | 0.0% | 3.8% | 2.0% | 0.8% | 0.7% | 0.5% | 0.2% | 0.5% | 0.9% | 0.3% | 0.0% |
| LIRF | 41.7% | 75.9% | 74.6% | 67.1% | 46.5% | 48.6% | 32.3% | 45.2% | 43.5% | 48.7% | 61.5% | 76.1% |
| LSZH | 4.6% | 8.2% | 6.9% | 5.6% | 5.2% | 3.3% | 3.4% | 3.7% | 1.5% | 0.6% | 4.5% | 2.4% |
| LTFM | 10.8% | 6.3% | 11.2% | 13.7% | 11.9% | 9.5% | 9.2% | 9.4% | 10.3% | 7.7% | 9.0% | 7.9% |

LIRF's rate is anti-correlated with volume (winter 62–76%, July 32%): the count of copies is roughly
flat while genuinely delayed flights swell in summer. LTFM's 6–14% equals its matched-row rate
(9.2%–11.5%) — ordinary on-time pushes, not fills. EHAM/LSZH 3–7% are the VFR/helicopter rows (§6).

### 5c. Unmatched rows — share with BLOCK − SCHED exactly 0

| airport | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LIRF | 6.7% | 10.3% | 5.1% | 12.9% | 4.0% | 2.7% | 3.9% | 3.6% | 2.4% | 3.5% | 3.8% | 9.1% |
| all others | ≤ 1.2% in every month (EGLL 10: 1.1%, LEBL 04: 1.2%, LTFM 08: 1.6%; otherwise 0.0–0.9%) | | | | | | | | | | | |

### 5d. Unmatched rows — share with target > 3 h

| airport | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LIRF | 11.7% | 17.2% | 10.2% | 14.3% | 15.2% | 9.2% | 13.4% | 12.2% | 11.9% | 7.8% | 11.5% | 12.5% |
| all others | 0.0% in 103 of the 108 other airport-months; the exceptions are single rows (EGLL 07: 0.3%, EHAM 01: 0.3%, LFPG 01: 0.9%, LSZH 02: 1.2%, LSZH 05: 0.5%) | | | | | | | | | | | |

### 5e. Matched rows — share with |BLOCK − SCHED| ≤ 60

| airport | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 9.1% | 9.2% | 8.4% | 8.3% | 6.8% | 6.0% | 4.9% | 5.4% | 5.1% | 7.0% | 8.8% | 7.0% |
| EDDM | 11.6% | 13.5% | 12.3% | 12.4% | 10.7% | 9.2% | 7.8% | 9.0% | 8.9% | 9.3% | 12.0% | 10.0% |
| EGLL | 8.6% | 10.3% | 9.7% | 11.2% | 11.0% | 9.9% | 8.8% | 10.4% | 8.7% | 9.7% | 10.5% | 9.8% |
| EHAM | 7.0% | 8.1% | 9.0% | 9.3% | 8.1% | 6.6% | 4.9% | 5.8% | 4.5% | 5.3% | 8.2% | 7.5% |
| LEBL | 10.7% | 11.1% | 9.6% | 10.0% | 9.4% | 8.2% | 10.2% | 11.9% | 9.5% | 9.2% | 10.7% | 10.3% |
| LEMD | 9.8% | 11.0% | 9.4% | 9.7% | 9.4% | 7.7% | 6.6% | 7.7% | 8.1% | 8.3% | 8.3% | 8.4% |
| LFPG | 8.3% | 10.1% | 11.4% | 9.7% | 8.6% | 7.3% | 6.1% | 7.0% | 7.1% | 8.1% | 8.8% | 7.9% |
| LIRF | 22.3% | 23.3% | 23.8% | 20.1% | 20.4% | 18.8% | 16.4% | 19.2% | 18.2% | 20.2% | 22.6% | 23.0% |
| LSZH | 7.7% | 8.0% | 8.2% | 5.9% | 6.0% | 4.7% | 3.8% | 4.6% | 4.2% | 4.8% | 7.4% | 6.3% |
| LTFM | 10.5% | 11.1% | 10.3% | 11.1% | 11.3% | 10.9% | 10.2% | 9.2% | 9.9% | 10.0% | 11.5% | 11.0% |

LIRF matched-row contamination (|d| ≤ 60 and |BLOCK − AOBT_3| > 300) by month: 8.2, 9.0, 9.9, 8.8,
8.9, 9.4, 8.9, 9.6, 8.5, 9.6, 8.8, 8.8%. The BLOCK 60 s-grid share and BLOCK = AOBT_3 exact rate are
flat within ±0.3 pp per airport across all months (`b_coincide.out` Q4b) — no source-system change
during 2025.

## 6. Q5 — discrete components in the target

| airport | stratum | n | on 60 s grid | on 10 s grid | mode | mode count | neighbour mean (±30 s) | spike ratio | p50 | p99 | >3h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | matched | 228185 | 1.7% | 10.0% | 798 | 351 | 307 | 1 | 836 | 1816 | 0.0% |
| EDDF | unmatched | 1956 | 1.6% | 9.8% | 1071 | 9 | 2 | 4 | 958 | 2122 | 0.0% |
| EDDM | matched | 166324 | 4.2% | 12.6% | 719 | 820 | 298 | 3 | 774 | 1916 | 0.0% |
| EDDM | unmatched | 1010 | 4.1% | 14.0% | 663 | 9 | 1 | 8 | 891 | 3655 | 0.0% |
| EGLL | matched | 238132 | 4.1% | 12.4% | 1319 | 767 | 268 | 3 | 1319 | 2697 | 0.0% |
| EGLL | unmatched | 1414 | 4.3% | 12.1% | 1314 | 9 | 2 | 6 | 1384 | 6302 | 0.1% |
| EHAM | matched | 243865 | 1.7% | 10.0% | 722 | 439 | 370 | 1 | 744 | 1767 | 0.0% |
| EHAM | unmatched | 4083 | 2.4% | 10.8% | 187 | 23 | 7 | 4 | 539 | 2266 | 0.0% |
| LEBL | matched | 177838 | 1.7% | 10.0% | 724 | 296 | 253 | 1 | 905 | 1957 | 0.0% |
| LEBL | unmatched | 1862 | 1.3% | 11.3% | 1230 | 7 | 2 | 3 | 1038 | 2279 | 0.0% |
| LEMD | matched | 211295 | 1.6% | 9.9% | 956 | 346 | 310 | 1 | 984 | 1932 | 0.0% |
| LEMD | unmatched | 947 | 1.6% | 9.4% | 1195 | 5 | 1 | 4 | 1096 | 2161 | 0.0% |
| LFPG | matched | 235725 | 4.2% | 12.5% | 841 | 931 | 333 | 3 | 954 | 2402 | 0.0% |
| LFPG | unmatched | 3762 | 3.9% | 11.5% | 966 | 23 | 5 | 5 | 1023 | 3831 | 0.1% |
| LIRF | matched | 159216 | 4.2% | 12.6% | 892 | 548 | 190 | 3 | 1024 | 3299 | 0.0% |
| LIRF | unmatched | 1488 | 3.9% | 12.0% | 783 | 7 | 1 | 9 | 3922 | 60264 | 12.1% |
| LSZH | matched | 132540 | 1.7% | 9.9% | 670 | 291 | 248 | 1 | 712 | 1705 | 0.0% |
| LSZH | unmatched | 2054 | 1.6% | 10.2% | **2** | 27 | 3 | 8 | 719 | 1894 | 0.1% |
| LTFM | matched | 269320 | 4.2% | 12.5% | 904 | 1044 | 379 | 3 | 963 | 2568 | 0.0% |
| LTFM | unmatched | 3643 | 4.4% | 12.1% | 782 | 23 | 4 | 5 | 1009 | 2568 | 0.0% |

The "spike ratio 3" at 13-value airports is the fingerprint (target on a 60 s multiple 4.2% vs
1.7%); it is not a pile-up at a round duration. Unmatched spike ratios are small-n noise (mode counts
5–27), except LSZH whose mode is 2 seconds.

### 6a. Target = MVT − SCHED within 60 s (the schedule-fill signature), per airport × stratum

| airport | stratum | n | share | n | among them >3h | among them p50 |
|---|---|---|---|---|---|---|
| LIRF | unmatched | 1488 | **48.5%** | 722 | **22.7%** | 7534 |
| LIRF | matched | 159216 | 20.5% | 32584 | 0.0% | 1084 |
| LTFM | unmatched | 3643 | 9.4% | 344 | 0.0% | 966 |
| LTFM | matched | 269320 | 10.6% | 28427 | 0.0% | 960 |
| EHAM | unmatched | 4083 | 4.0% | 165 | 0.0% | 239 |
| LSZH | unmatched | 2054 | 3.8% | 79 | 0.0% | 487 |
| LEBL | unmatched | 1862 | 2.0% | 38 | 0.0% | 910 |
| EDDF/EDDM/EGLL/LEMD/LFPG | unmatched | | 0.2–0.7% | 2–24 | 0.0% | 759–4795 |
| all others | matched | | 5.8–10.4% | | 0.0% | 696–1316 |

Only at LIRF does the signature carry a >3h tail; everywhere else the rows that satisfy it are on-time
pushes with ordinary taxi-outs.

### 6b. Near-zero targets

| airport | stratum | n | target<60 | n<60 | target<120 | target<300 | among <60: d p50 | among <60: ms p50 |
|---|---|---|---|---|---|---|---|---|
| LSZH | unmatched | 2054 | **11.3%** | 233 | 11.4% | 14.8% | 257 | 262 |
| EHAM | unmatched | 4083 | 2.6% | 108 | 9.6% | **34.2%** | 432 | 480 |
| LEBL | unmatched | 1862 | 0.8% | 14 | 0.9% | 2.9% | 1765 | 1801 |
| LFPG | unmatched | 3762 | 0.3% | 10 | 0.3% | 0.5% | 2935 | 2945 |
| all other unmatched | | | ≤ 0.1% | 0–2 | ≤ 0.2% | ≤ 1.9% | | |
| LSZH | matched | 132540 | 0.1% | 128 | 0.1% | 4.8% | 879 | 889 |
| EHAM | matched | 243865 | 0.0% | 81 | 0.3% | 2.5% | 958 | 1000 |
| all other matched | | | 0.0% | 0–61 | ≤ 0.1% | ≤ 2.3% | | |

**LSZH unmatched < 60 s (233 rows):** FLIGHT_RULE V 226 / I 7; types AS50 63, A109 41, C172 22, B429 17,
P28A 15, EC30 14, DA40 12, EC20 11; RUNWAY `H` 169, `28` 60; STAND `HPW` 132, `MFG` 56, `REG` 36.
Target values: 2 s ×27, 4 ×26, 9 ×26, 5 ×25, 10 ×19, 7 ×19, 11 ×18, 8 ×17, 3 ×17, 6 ×11.
Monthly (n<60 / n unmatched): 34/153, 22/85, 19/130, 13/107, 12/213, 22/242, 31/323, 28/271, 17/205,
12/154, 9/88, 14/83. Among LSZH unmatched VFR rows 25.3% have target < 60 (IFR: 0.6%). The runway-28
sub-group is GA at stand `MFG` (C172 21, P28A 14, DA40 12). BLOCK − SCHED on these rows is ordinary
(p10/p50/p90 −689/257/1789; only 12.9% copies) — the off-block is written at the moment of
departure, not copied from anything.

**LSZH matched target < 120 s (132 rows):** all IFR business jets (C56X 13, E55P 12, PC12 9, CL60 9); NM
proxy taxi p10/p50/p90 489/686/963 s and BLOCK − AOBT_3 474/661/955 s — the airport's off-block is
late by one full taxi, a clock error on 0.1% of rows, not a rule.

**EHAM unmatched < 300 s (1,396 rows):** V 1232 / I 147; A139 839, DH8A 279, DA40 55, DA42 38;
runway `22` 844, `04` 441; stand `HG01` 845, `HG03` 276. A139 (n 966): target p10/p50/p90 62/176/308,
stand HG01 on 963. DH8A (n 434): 111/247/463, stand HG03 on 427. Monthly share of EHAM unmatched with
target < 300: 27.6, 47.2, 46.2, 52.5, 37.7, 37.6, 28.0, 36.9, 20.9, 26.2, 40.0, 33.7%.

### 6c. Heliport identifiers: 2025 profile and 2026 presence (unmatched rows)

| airport | key | n 2025 | per month | target p10 | p50 | p90 | <60 s | <300 s | n 2026 | 2026-01 | 2026-07 | n 2025 Jan+Jul |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LSZH | RUNWAY = H | 190 | 16 | 2 | 7 | 112 | 88.9% | 96.8% | 78 | 39 | 39 | 57 |
| LSZH | STAND = HPW | 136 | 11 | 2 | 7 | 12 | 97.1% | 97.8% | 58 | 25 | 33 | 40 |
| EHAM | STAND = HG01 | 974 | 81 | 62 | 176 | 308 | 8.6% | 86.8% | 148 | 67 | 81 | 149 |
| EHAM | STAND = HG03 | 429 | 36 | 112 | 248 | 464 | 3.7% | 64.3% | 57 | 18 | 39 | 60 |
| EHAM | RUNWAY = HEL | 29 | 2 | 122 | 229 | 367 | 3.4% | 75.9% | 11 | 11 | 0 | 0 |

LSZH RUNWAY = H monthly n: 34, 20, 16, 11, 9, 18, 23, 21, 11, 11, 6, 10; target ≤ 15 s on 86.8%.

## 7. Mechanism of the unmatched stratum (needed to read §8)

| evidence | value |
|---|---|
| BLOCK − LOBT on matched rows, all airports | min −3,606, max 3,606, p99.9 3,474; share > 3,600: 0.0% at every airport (per-airport max 3,602–3,606, min −3,485…−3,606) |
| BLOCK − EOBT_1 | max 8,705, > 3,600 on 0.0–0.1% |
| BLOCK − SCHED on matched rows | > 3,600 on 2.8%–5.7%; max 269,095 |
| ⇒ | the NM record was joined to the movement with \|BLOCK − LOBT\| ≤ 1 h. A movement is "unmatched" when its off-block is > 1 h from the last-filed off-block time, or when there is no NM flight plan at all. |

| airport | unmatched in 1h..3h of SCHED | matched in 1h..3h | unmatched > 3h | unmatched < 1h | VFR share of unmatched 2025 | VFR share 2026 | IFR & d ≤ 1h share 2025 |
|---|---|---|---|---|---|---|---|
| EDDF | 82.7% | 3.5% | 5.4% | 11.9% | 1.1% | 0.2% | 10.3% |
| EDDM | 92.5% | 2.6% | 2.1% | 5.4% | 2.7% | 2.1% | 3.0% |
| EGLL | 90.9% | 3.2% | 5.4% | 3.6% | 0.0% | 0.0% | 3.6% |
| EHAM | 49.8% | 5.1% | 4.4% | 45.8% | 39.6% | 19.6% | 7.4% |
| LEBL | 84.5% | 4.9% | 6.3% | 9.2% | 7.3% | 0.7% | 2.3% |
| LEMD | 94.4% | 4.5% | 3.8% | 1.8% | 0.2% | 0.0% | 1.6% |
| LFPG | 84.4% | 5.1% | 4.0% | 11.6% | 0.0% | 0.0% | 11.6% |
| LIRF | 42.5% | 4.3% | 2.6% | 55.0% | 0.0% | 0.0% | 55.0% |
| LSZH | 48.1% | 3.4% | 1.3% | 50.5% | 43.4% | 26.4% | 7.3% |
| LTFM | 15.2% | 3.5% | 1.3% | 83.5% | 0.0% | 0.0% | 83.5% |

Three sub-populations: (i) IFR flights that pushed > 1 h from their filed time (EDDF EDDM EGLL LEBL
LEMD LFPG ≈ 85–94%); (ii) VFR/helicopter/GA with no NM plan (EHAM 40%, LSZH 43%); (iii) LTFM (83.5%
on-time IFR, unmatched for a reason this data cannot show) and LIRF (55% "on time" = the schedule
copies). LIRF unmatched rows also have AIRCRAFT_TYPE_mvt null on 99.9% (2026: 100.0%) while every
other airport's unmatched rows and LIRF's matched rows have 0.0% null — the LIRF fill rows come from a
different record source, and that source is still feeding the 2026 file.

## 8. The 2026 side — MVT − SCHED on unmatched rows, 2025 Jan+Jul vs 2026 Jan+Jul

Counts of unmatched rows with `ms` > 3 h — 2026: 668 (EHAM 297, LFPG 88, LIRF 74, LEBL 51, EDDF 36,
LSZH 34, EGLL 31, LEMD 22, EDDM 19, LTFM 16). 2025 Jan+Jul: 383 (LIRF 83, EHAM 62, LFPG 51, LEBL 48,
EDDF 47, EGLL 41, LEMD 23, LSZH 10, LTFM 10, EDDM 8).

### 8a. Pooled Jan+Jul, unmatched rows

| airport | year | n unm | p50 | p90 | p99 | max | ms<0 | >3h | n>3h |
|---|---|---|---|---|---|---|---|---|---|
| EDDF | 2025 | 599 | 5967 | 9940 | 28869 | 72293 | 1.0% | 7.8% | 47 |
| EDDF | 2026 | 421 | 5936 | 10240 | 25755 | 96310 | 0.2% | 8.6% | 36 |
| EDDM | 2025 | 237 | 5757 | 8786 | 16094 | 24296 | 0.0% | 3.4% | 8 |
| EDDM | 2026 | 238 | 6304 | 9943 | 20112 | 42059 | 0.0% | 8.0% | 19 |
| EGLL | 2025 | 466 | 6658 | 10562 | 17176 | 30002 | 0.2% | 8.8% | 41 |
| EGLL | 2026 | 248 | 6182 | 11538 | 23131 | 35466 | 0.0% | 12.5% | 31 |
| EHAM | 2025 | 808 | 5052 | 9515 | 20971 | 98691 | 7.3% | 7.7% | 62 |
| EHAM | 2026 | 1206 | 6740 | 15564 | 34362 | 85535 | 3.5% | **24.6%** | 297 |
| LEBL | 2025 | 496 | 6190 | 10688 | 18741 | 45569 | 0.0% | 9.7% | 48 |
| LEBL | 2026 | 442 | 6448 | 11419 | 21595 | 35383 | 0.0% | 11.5% | 51 |
| LEMD | 2025 | 309 | 6444 | 10131 | 16781 | 22512 | 0.0% | 7.4% | 23 |
| LEMD | 2026 | 238 | 6244 | 10534 | 16899 | 18960 | 0.0% | 9.2% | 22 |
| LFPG | 2025 | 876 | 5885 | 9114 | 20325 | 88202 | 0.0% | 5.8% | 51 |
| LFPG | 2026 | 734 | 6124 | 11421 | 21220 | 69540 | 0.3% | 12.0% | 88 |
| LIRF | 2025 | 397 | 6781 | 15035 | 60291 | 93535 | 1.3% | 20.9% | 83 |
| LIRF | 2026 | 383 | 6955 | 15793 | 57264 | 111654 | 1.3% | 19.3% | 74 |
| LSZH | 2025 | 476 | 4560 | 7826 | 12361 | 18795 | 9.5% | 2.1% | 10 |
| LSZH | 2026 | 659 | 5094 | 9279 | 18237 | 36187 | 7.6% | 5.2% | 34 |
| LTFM | 2025 | 657 | 1443 | 6715 | 12325 | 15124 | 1.2% | 1.5% | 10 |
| LTFM | 2026 | 721 | 1196 | 5701 | 17052 | 28622 | 1.2% | 2.2% | 16 |

### 8b. By month, unmatched rows

| airport | month | n unm | unm share | p50 | p90 | p99 | max | ms<0 | >3h | n>3h |
|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 2025-01 | 164 | 1.1% | 5878 | 9931 | 48030 | 71513 | 1.8% | 8.5% | 14 |
| EDDF | 2025-07 | 435 | 2.0% | 6019 | 9931 | 22224 | 72293 | 0.7% | 7.6% | 33 |
| EDDF | 2026-01 | 212 | 1.4% | 6308 | 11231 | 41408 | 72574 | 0.0% | 11.8% | 25 |
| EDDF | 2026-07 | 209 | 1.0% | 5689 | 8618 | 23266 | 96310 | 0.5% | 5.3% | 11 |
| EDDM | 2025-01 | 65 | 0.6% | 5936 | 8446 | 13845 | 15425 | 0.0% | 3.1% | 2 |
| EDDM | 2025-07 | 172 | 1.1% | 5703 | 8819 | 16532 | 24296 | 0.0% | 3.5% | 6 |
| EDDM | 2026-01 | 119 | 1.1% | 6841 | 10459 | 17165 | 23823 | 0.0% | 10.1% | 12 |
| EDDM | 2026-07 | 119 | 0.8% | 5758 | 9634 | 20259 | 42059 | 0.0% | 5.9% | 7 |
| EGLL | 2025-01 | 128 | 0.7% | 6602 | 10286 | 17732 | 30002 | 0.8% | 8.6% | 11 |
| EGLL | 2025-07 | 338 | 1.6% | 6662 | 10596 | 16045 | 22497 | 0.0% | 8.9% | 30 |
| EGLL | 2026-01 | 112 | 0.6% | 6004 | 12441 | 21000 | 23640 | 0.0% | 13.4% | 15 |
| EGLL | 2026-07 | 136 | 0.7% | 6269 | 11222 | 24429 | 35466 | 0.0% | 11.8% | 16 |
| EHAM | 2025-01 | 293 | 1.6% | 5047 | 9730 | 22125 | 68408 | 6.5% | 8.2% | 24 |
| EHAM | 2025-07 | 515 | 2.3% | 5062 | 9420 | 19766 | 98691 | 7.8% | 7.4% | 38 |
| EHAM | 2026-01 | 805 | **5.0%** | 8631 | 17183 | 34369 | 85535 | 2.7% | **34.5%** | 278 |
| EHAM | 2026-07 | 401 | 1.8% | 4749 | 7929 | 26828 | 81993 | 5.0% | 4.7% | 19 |
| LEBL | 2025-01 | 66 | 0.5% | 6061 | 9515 | 17510 | 18744 | 0.0% | 6.1% | 4 |
| LEBL | 2025-07 | 430 | 2.6% | 6190 | 10930 | 18611 | 45569 | 0.0% | 10.2% | 44 |
| LEBL | 2026-01 | 77 | 0.6% | 6137 | 13159 | 21047 | 27892 | 0.0% | 13.0% | 10 |
| LEBL | 2026-07 | 365 | 2.0% | 6468 | 11322 | 21314 | 35383 | 0.0% | 11.2% | 41 |
| LEMD | 2025-01 | 32 | 0.2% | 5992 | 9679 | 11371 | 11803 | 0.0% | 3.1% | 1 |
| LEMD | 2025-07 | 277 | 1.5% | 6505 | 10132 | 17145 | 22512 | 0.0% | 7.9% | 22 |
| LEMD | 2026-01 | 63 | 0.4% | 6895 | 9957 | 16369 | 17354 | 0.0% | 6.3% | 4 |
| LEMD | 2026-07 | 175 | 0.9% | 6072 | 10750 | 16578 | 18960 | 0.0% | 10.3% | 18 |
| LFPG | 2025-01 | 215 | 1.2% | 6054 | 9239 | 19118 | 25984 | 0.0% | 6.5% | 14 |
| LFPG | 2025-07 | 661 | 3.0% | 5882 | 9061 | 20507 | 88202 | 0.0% | 5.6% | 37 |
| LFPG | 2026-01 | 325 | 1.8% | 6478 | 12935 | 18282 | 52555 | 0.0% | 16.9% | 55 |
| LFPG | 2026-07 | 409 | 1.9% | 5946 | 10195 | 21461 | 69540 | 0.5% | 8.1% | 33 |
| LIRF | 2025-01 | 60 | 0.5% | 6688 | 14332 | 58064 | 60118 | 3.3% | 13.3% | 8 |
| LIRF | 2025-07 | 337 | 2.2% | 6783 | 15035 | 62323 | 93535 | 0.9% | 22.3% | 75 |
| LIRF | 2026-01 | 107 | 1.0% | 7141 | 19592 | 53239 | 56520 | 2.8% | 19.6% | 21 |
| LIRF | 2026-07 | 276 | 1.7% | 6897 | 14461 | 60822 | 111654 | 0.7% | 19.2% | 53 |
| LSZH | 2025-01 | 153 | 1.6% | 1657 | 6498 | 12391 | 14729 | 13.7% | 3.3% | 5 |
| LSZH | 2025-07 | 323 | 2.6% | 4855 | 8026 | 12174 | 18795 | 7.4% | 1.5% | 5 |
| LSZH | 2026-01 | 300 | 3.0% | 5884 | 10189 | 18718 | 24926 | 7.0% | 8.0% | 24 |
| LSZH | 2026-07 | 359 | 2.7% | 4888 | 8025 | 14312 | 36187 | 8.1% | 2.8% | 10 |
| LTFM | 2025-01 | 232 | 1.1% | 1084 | 6046 | 11375 | 14822 | 1.7% | 1.3% | 3 |
| LTFM | 2025-07 | 425 | 1.7% | 1740 | 6837 | 12651 | 15124 | 0.9% | 1.6% | 7 |
| LTFM | 2026-01 | 333 | 1.5% | 1555 | 7422 | 18315 | 28622 | 2.1% | 3.6% | 12 |
| LTFM | 2026-07 | 388 | 1.5% | 1077 | 4154 | 10203 | 28325 | 0.5% | 1.0% | 4 |

### 8c. Control — matched rows, MVT − SCHED, pooled Jan+Jul

| airport | 2025 n / p50 / p90 / p99 / >3h | 2026 n / p50 / p90 / p99 / >3h |
|---|---|---|
| EDDF | 36231 / 1474 / 3618 / 9811 / 0.8% | 35896 / 1484 / 3682 / 9348 / 0.7% |
| EDDM | 26854 / 1205 / 3177 / 7142 / 0.3% | 25728 / 1263 / 3294 / 8275 / 0.5% |
| EGLL | 39744 / 1683 / 4018 / 9657 / 0.7% | 39592 / 1565 / 3661 / 8882 / 0.6% |
| EHAM | 40141 / 1525 / 4036 / 10177 / 0.9% | 36976 / 1445 / 3947 / 13880 / **1.7%** |
| LEBL | 28489 / 1399 / 3967 / 10895 / 1.0% | 29638 / 1380 / 3951 / 10139 / 0.8% |
| LEMD | 35019 / 1522 / 3646 / 9314 / 0.6% | 36716 / 1575 / 3937 / 10079 / 0.8% |
| LFPG | 38713 / 1797 / 4259 / 10262 / 0.9% | 39138 / 1624 / 4319 / 11762 / 1.3% |
| LIRF | 26131 / 1564 / 4318 / 10675 / 1.0% | 26516 / 1557 / 4322 / 10380 / 0.9% |
| LSZH | 21811 / 1447 / 3394 / 7489 / 0.4% | 22491 / 1522 / 3631 / 8460 / 0.5% |
| LTFM | 45882 / 1315 / 3302 / 12375 / 1.3% | 46860 / 1384 / 3604 / 12421 / 1.3% |

NM-clock proxy taxi (MVT − AOBT_3) on matched rows, 2025 → 2026 p50/p90/p99: EHAM 862/1414/2641 →
918/1512/3609; LFPG 899/1494/2517 → 904/1501/2461; LIRF 901/1204/1622 → 901/1256/1505; every airport
0.0% > 3h in both years.

### 8d. Structure of `ms` on unmatched rows — no date/timezone slips

| airport | year | n | within 2 min of a whole hour (≥1h) | near 24h | near 23h/25h | near +1h | near +2h | near −1h | >20h | 3h..20h |
|---|---|---|---|---|---|---|---|---|---|---|
| EHAM | 2025 | 808 | 3.7% | 0.0% | 0.0% | 0.6% | 2.1% | 0.0% | 0.2% | 7.4% |
| EHAM | 2026 | 1206 | 4.3% | 0.1% | 0.0% | 0.2% | 2.0% | 0.0% | 0.2% | 24.4% |
| LIRF | 2025 | 397 | 4.3% | 0.3% | 0.0% | 0.3% | 2.5% | 0.0% | 0.8% | 20.2% |
| LIRF | 2026 | 383 | 3.7% | 0.3% | 0.0% | 0.8% | 1.3% | 0.0% | 0.5% | 18.8% |
| all others | both | | 0.6–5.2% | 0.0–0.1% | 0.0% | 0.0–0.4% | 0.1–3.8% | 0.0–0.2% | 0.0–0.5% | 1.5–12.5% |

"near +2h" 1–4% everywhere is the delay pile's edge, equal in both years. Nothing sits at 24 h.

### 8e. The EHAM inversion — date clustering of unmatched rows with `ms` > 3h

| airport | set | n | n dates | top date : n | top-1 share | top-5 share | top-5 dates |
|---|---|---|---|---|---|---|---|
| EHAM | 2025 (12 mo) | 218 | 131 | 2025-09-15 : 12 | 5.5% | 20.2% | 09-15:12; 09-16:11; 10-04:8; 07-12:7; 05-30:6 |
| EHAM | 2026 | 297 | 32 | 2026-01-04 : 72 | 24.2% | **82.2%** | 01-04:72; 01-05:66; 01-03:48; 01-06:29; 01-02:29 |
| LFPG | 2025 | 256 | 137 | 2025-12-05 : 28 | 10.9% | 29.3% | 12-05:28; 11-22:25; 07-02:12; 04-05:5; 07-31:5 |
| LFPG | 2026 | 88 | 34 | 2026-01-07 : 23 | 26.1% | 52.3% | 01-07:23; 01-05:9; 01-04:6; 01-03:4; 07-10:4 |
| LSZH | 2025 | 43 | 38 | 2025-01-04 : 3 | 7.0% | 23.3% | 01-04:3; 05-26:2; 05-28:2; 06-23:2; 01-08:1 |
| LSZH | 2026 | 34 | 18 | 2026-01-10 : 7 | 20.6% | 61.8% | 01-10:7; 01-02:5; 01-08:5; 01-04:2; 01-22:2 |
| LIRF | 2025 | 243 | 139 | 2025-07-13 : 13 | 5.3% | 14.8% | 07-13:13; 07-21:7; 07-14:6; 07-02:5; 07-07:5 |
| LIRF | 2026 | 74 | 38 | 2026-07-02 : 9 | 12.2% | 35.1% | 07-02:9; 07-05:5; 01-03:4; 07-11:4; 07-20:4 |
| EDDF | 2026 | 36 | 18 | 2026-01-06 : 6 | 16.7% | 44.4% | 01-06:6; 01-02:3; 01-26:3; 01-04:2; 01-03:2 |
| LTFM | 2025 | 86 | 49 | 2025-02-23 : 29 | 33.7% | 44.2% | 02-23:29; 04-05:3; 02-24:2; 02-19:2; 07-19:2 |

### 8f. EHAM January 2026, day by day, with the independent NM clock on matched rows

| date | n | matched | matched ms>3h | proxy p50 | proxy p90 | proxy >3h | unm | unm share | unm ms>3h |
|---|---|---|---|---|---|---|---|---|---|
| 2026-01-01 | 618 | 609 | 1.8% | 754 | 1343 | 0.0% | 9 | 1.5% | 1 |
| 2026-01-02 | 469 | 375 | 18.7% | 963 | 2359 | 0.0% | 94 | 20.0% | 29 |
| 2026-01-03 | 391 | 283 | 30.7% | 1998 | 4766 | 0.0% | 108 | 27.6% | 48 |
| 2026-01-04 | 336 | 213 | 37.1% | 2352 | 5397 | 0.0% | 123 | 36.6% | 72 |
| 2026-01-05 | 208 | 124 | 42.7% | 2110 | 5411 | 0.0% | 84 | 40.4% | 66 |
| 2026-01-06 | 293 | 225 | 24.4% | 1142 | 4427 | 0.0% | 68 | 23.2% | 29 |
| 2026-01-07 | 194 | 176 | 17.6% | 1573 | 4251 | 0.0% | 18 | 9.3% | 6 |
| 2026-01-08 | 585 | 571 | 1.1% | 795 | 1241 | 0.0% | 14 | 2.4% | 1 |
| 2026-01-09 | 497 | 448 | 2.9% | 1185 | 3279 | 0.0% | 49 | 9.9% | 11 |
| 2026-01-29 | 588 | 519 | 1.7% | 923 | 2410 | 0.0% | 69 | 11.7% | 5 |
| every other January date | 507–623 | 500–614 | 0.0–2.0% | 727–1169 | 1042–2406 | 0.0% | 1–19 | 0.2–3.3% | 0–3 |

Daily departures fall from ~600 to 208 on 01-05, matched-row MVT − SCHED > 3h reaches 42.7%, and the
NM-clock taxi-out p50 triples (754 → 2,352 s) with p90 5,411 s but **no matched row above 3 h**. The
unmatched rows' `ms` on those days are 3–6 h on 86% (149 in 3–4h, 77 in 4–5h, 30 in 5–6h of 297),
SCHED hours peak 08–09Z, MVT hours peak 12–15Z. The same pattern is seen on 2025 disruption days:

| airport | 2025 days with ≥5 unmatched ms>3h rows | n rows | unmatched target p50 | p90 | >3h | copies | matched target p50 same days |
|---|---|---|---|---|---|---|---|
| EHAM | 5 | 44 | 662 | 939 | 0.0% | 0.0% | 697 |
| LFPG | 5 | 75 | 1793 | 3479 | 0.0% | 0.0% | 1015 |
| EGLL | 3 | 29 | 1790 | 2373 | 0.0% | 0.0% | 1438 |
| EDDF | 4 | 32 | 1046 | 1437 | 0.0% | 0.0% | 869 |
| LEBL | 7 | 54 | 1134 | 2058 | 0.0% | 0.0% | 968 |
| LEMD | 3 | 22 | 1170 | 2065 | 0.0% | 0.0% | 993 |
| LTFM | 1 (2025-02-23) | 29 | 6610 | 8786 | 0.0% | 0.0% | 2635 |
| **LIRF** | 7 | 46 | **11452** | **43026** | **52.2%** | **47.8%** | 1254 |

Across all of 2025, unmatched rows with `ms` > 3h at the nine non-LIRF airports: 0.0% copies, 0.0%
target > 3h, target p50 776–1,527 s. At LIRF: 67.5% copies, 72.4% target > 3h, p50 13,556 s.

**Verdict on EHAM 2026:** heavily delayed flights during a 2026-01-02..07 disruption (with LFPG, LSZH,
EDDF, EDDM, EGLL, LEMD, LTFM all showing early-January clusters). Not a fill artifact: the disruption
is visible on matched rows through an independent clock, the unmatched `ms` values are a smooth 3–6 h
tail with no discrete structure, and EHAM's 2025 behaviour on disruption days was ordinary taxi-outs.
Expect true taxi-out on those 297 rows to be elevated (the day-matched NM proxy p50 is 1,100–2,350 s,
p90 up to 5,400 s) but not 3 h+. The LIRF 2026 rows, by contrast, look exactly like 2025 LIRF.

## 9. Extension of the LIRF artifact

| evidence | value |
|---|---|
| unmatched copies (n 722): BLOCK − SCHED value counts | −59:1, −6:48, −5:68, −4:57, −3:53, −2:51, −1:63, 0:69, 1:61, 2:63, 3:58, 4:54, 5:66, 6:10 → **99.9% within ±6 s** |
| chance of \|d\| ≤ 6 on a continuous clock (EDDF matched) | 0.8% |
| LIRF matched, \|d\| ≤ 6 | 17.8% |
| LIRF matched contaminated rows (\|d\| ≤ 60 & \|BLOCK − AOBT_3\| > 300) | 14,423 = 9.1%; 91.5% of them within ±6 s, the rest at ±54..60 (one minute off) |
| contamination at deviation thresholds 60 / 300 / 600 / 1800 / 3600 s | 17.5% / 9.1% / 4.6% / 1.1% / 0.2% |
| contaminated rows: target p10/p50/p90/p99 | 901 / 1501 / 3067 / 5520 |
| same rows: NM proxy taxi p10/p50/p90/p99 | 605 / 901 / 1203 / 1505 |
| LIRF clean rows (\|BLOCK − AOBT_3\| ≤ 60): target p10/p50/p90/p99 | 716 / 893 / 1148 / 1493 |
| median (target − proxy) on contaminated rows; share with BLOCK earlier than AOBT_3; median AOBT_3 − SCHED | +597 s; 89.2%; 600 s |
| the same contamination measure at other airports | 0.6% (EHAM) – 2.3% (LEBL); LTFM 5.9% but with target − proxy median −474 s (its +296 s clock offset, §4c), not a copy |

So at LIRF the label is schedule-anchored on ~9% of matched rows too, inflating those targets by a
median 10 minutes, and the training label's LIRF distribution (p90 3,067 vs 1,148 clean) carries it.
Cannot be observed in 2026 (BLOCK null) but the monthly rate never left 8.2–9.9%.

## 10. What was looked for and not found

| check | result |
|---|---|
| BLOCK equal to LOBT / IOBT / EOBT_1 / ARVT_1 / ARVT_3 more than to SCHED | no, at any airport (§4) |
| BLOCK on a coarser grid on unmatched rows than on matched rows | no (§3): identical fingerprint, ±1 pp |
| a spike at any exact BLOCK − SCHED value other than LIRF's 0 ± 6 | no; largest other single value 0.9% (EDDM unmatched 3781, n 9) |
| a spike in the target at a round duration | no; largest exact-value share 0.6% outside LSZH's 2–11 s helicopter block |
| BLOCK = MVT − constant (a default taxi) | no; LSZH's ≤ 15 s rows are one stand/runway, VFR, with continuous BLOCK − SCHED |
| any airport where the unmatched-row target exceeds 3 h at more than a single-row rate | only LIRF (12.1%; all others ≤ 0.1%) |
| a source-system change inside 2025 or between 2025 and 2026 | none: seconds fingerprint, 60 s-grid share and BLOCK = AOBT_3 rate flat every month; MVT fingerprint identical in 2026 |
| 24 h / whole-hour slips in MVT − SCHED on unmatched rows | none in either year (§8d) |
