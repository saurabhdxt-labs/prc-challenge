# Taxi-time factor atlas — what moves taxi-out, and by how many seconds (2025 labelled data)

**2026-09-10/11. Descriptive research.** No verdicts, no ship claims. Every **conditional** number is a TreeSHAP
attribution on a LightGBM model validated on held-out months. It is **model-based, not causal**. Every **raw**
number is a group statistic, and raw numbers are confounded by construction.

Population: the shipped recipe's admissible departures (`build_submission.derive(admissible(load_movements(...)))`),
12 months of 2025. Record classes are separated first (§A). The physics sections use the **2,035,114 ordinary matched
rows** (the row has an NM off-block, and it is not a fill, date-slip or monster). The fold: train on 8 months,
early-stop on March and September, hold out **January and July (334,562 rows)**. The numbers come from
`reports/taxi_factor_atlas.json`. Models and tables are in `data/atlas/`, and the code is in `data/atlas/code/`.

Source tags used below: **[M]** raw marginal, all 12 months, n given; **[S]** SHAP on the physical model f, 100,000
held-out rows (seeded); **[G]** SHAP summed over the columns of one factor (§F.1b); **[C]** clocks-agree sensitivity
(§G.2); **[W]** METAR present-weather codes (§D).

---

## 1. Plain-language summary: factor → how many seconds, where

The typical ordinary taxi-out is **909 s median (15.2 min)**, with p10–p90 of 570–1,446 s (n = 2,035,114) [M]. The
table is ordered by how much each factor moves a single departure.

| factor | how many seconds | where / notes | source (n) |
|---|---|---|---|
| **Airport** | medians from **710 s (LSZH)** to **1,318 s (EGLL)**: a 608 s spread. EDDM 772, EHAM 744, EDDF 835, LEBL 897, LFPG 953, LTFM 962, LEMD 981, LIRF 1,011 | alone it explains R² 0.167 of held-out variance | [M] §B (131,684–267,221 per airport) |
| **Runway** (within airport) | EHAM **36L (Polderbaan) +326 s vs 24** (raw medians 1,049 vs 626). LTFM **17R +320 s vs 36** (raw 1,264 vs 891). LEMD 36R +128 vs 14R. LFPG 27R +134 vs 08L. EDDF 18 about +50 vs 25C/07C. EGLL's three runways sit within 30 s of each other | the largest single within-airport factor at EHAM and LTFM | [G] runway + stand SHAP by runway §F.1j (n 1,071–6,947 per runway); [M] §B.1 |
| **Stand area** (first 2 chars of STAND, within airport) | the SHAP range across stand areas runs from **162 s (EHAM)** to **481 s (EDDF: B1 −281, V3 +200)**. LEMD 472, LIRF 434, LSZH 424, LEBL 397, LTFM 362, LFPG 304, EGLL 253, EDDM 234. Raw medians are wider still (EHAM K3 316 s vs R7 1,220 s): the runway a stand is paired with co-varies with it, and the model credits the runway | stand + runway together are the biggest physical block (mean \|SHAP\| 136 s) | [S] §F.18 (areas with ≥100 sample rows); [M] §D |
| **Runway queue at push** (departures pushed before me and not yet airborne, same runway) | **about +20 s per extra queued aircraft** (whole congestion block, 0–15 queued). By airport: EDDM 11.5, LEMD 13.3, LSZH 13.2, LEBL 14.4, LTFM 15.1, EHAM 16.1, EGLL 16.2, EDDF 16.5, LIRF 17.2, **LFPG 23.5**. The block runs from **−101 s (empty)** to **+396 s (20 queued)** | at **EGLL** the typical queue at push is 12–20 (raw Δmedian vs EGLL's own median: 0–7 queued −185 to −241 s; 21+ queued +245 s). Most of EGLL's long median is runway queue | [G] §F.1d (n 94,840); [M] §D |
| **Recent runway taxi times** (median of the last 20 ordinary take-offs on my runway before my push) | **+0.20 s per s**, i.e. about 12 s for each extra minute the recent median runs long. SHAP runs from −54 s (recent < 10 min) to +160 s (recent ≥ 30 min) | the strongest raw tail marker: 30 min+ recent gives a 37% tail rate (lift 11.8) | [S] §F.8 / §F.19; [M] §H |
| **Runway throughput at my take-off** (take-offs on my runway within ±10 min) | **−126 s (none) to +59 s (> 12)** | measured at take-off, so it is a context variable, not something knowable at push | [S] §F.x (n 100,000) |
| **Freezing / de-icing** (T ≤ 3 °C with FZ/SN/PL/GS/GR/IC codes) | weather block **+398 s** when freezing (pooled). **LFPG +507, LSZH +414, EHAM +368, EDDM +359, EGLL +317, EDDF +267**. Temperature alone: −5 to 0 °C +169 s, below −5 °C +206 s, above 3 °C about 0 | raw, Nov–Mar freezing vs not: **LFPG +1,251 s median (n 849)**, EDDM +701 (n 2,297), LTFM +550 (n 2,589), LSZH +280, EDDF +213, EHAM +178. Freezing is the top tail driver at EDDM/EHAM/LSZH (lift 22–28) | [G] §F.1f–g (679 freezing rows in sample); [M] §D winter |
| **Snow code** in the METAR | raw: **EDDM +671 s** (n 1,264), LSZH +266, LTFM +246, EDDF +121, EHAM +75 | not in the model (see surprise 2) | [W] §D |
| **Low visibility / fog** | weather block **+257 s below 1.5 km**, +107 s at 1.5–5 km [G]. Raw fog code: **LFPG +478 s, EGLL +370 s**, others +38 to +245 | | [G] §F.1h (n 1,408); [W] |
| **Aircraft size** (wake class + type) | **Medium −14, Heavy +54, Super (A388) +149 s** (pooled, against the model mean). Heavy minus Medium is 64–98 s at nine airports; LTFM 26 s | by type: **A388 +146, B77L +82, B772 +73, A339 +72**; CRJ9 −79, A321 −31, A320 −25. Raw within-airport Δmedian: H +124, J +259, L −173, M −42 | [G] §F.1c; [S] type §F.x; [M] §C |
| **Airline** (flight-number prefix, beyond stand area and type) | mean \|SHAP\| **47 s**. **UAL +99, BAW +85, AAL +61, THY +52**; **IBS −114, ANE −72, LH1 −68, AEA −63, KL0 −61** | at LIRF, UAL/DAL departures run 39–42% over 30 min (tail lift 7–8) | [S] §F.x (top 25 by n) |
| **Departure delay at push** (AOBT_3 − SCHED) | model f: **early > 5 min −126 s, late 30–60 min +102 s**. On rows where the airport's off-block and NM's agree [C], the late side is only **+15 to +25 s**, while early stays about −70 s | **mostly a recording effect**: 18–22% of late (15 min+) flights carry an airport off-block more than 5 min before NM's push, against 1–5% on time or early. At LIRF the raw effect vanishes on clock-agreeing rows | [G] §F.1i; [C] §G.2 (n 30,000); [M] §D delay × clock |
| **Time of day** (local push hour) | daytime clock effect only **−10 to +11 s**. **Night 22:00–02:00 +22 to +110 s** (02:00 +110, n 486) | raw daytime swings (06–08 −62 s, 12–14 +43) are congestion, not the clock | [G] §F.1e; [M] §C |
| **Thunderstorm code** | raw **+192 s** within-airport median (n 6,160). LEBL tail lift 15 | | [M] §C |
| **Stand turnaround < 30 min** (AOBT_3 − last on-block at my stand) | **+177 s** | small n | [S] §F.x (n 917) |
| **Traffic per hour** (departures ±30 min of push) | raw within-airport Δmedian **−63 s (0–9/h) to +66 s (60+/h)**. Conditional on the queue it adds nothing (its own SHAP slope is slightly negative: credit sharing) | read through the queue | [M] §C; [S] §F.19 |
| **Rain** | raw −32 to +59 s | near nothing | [W] |
| **Wind** | about 0 below 30 kt; +58 s at 30 kt+ | n 530 | [M] |
| **Season / month** | raw DJF − JJA medians −71 to +23 s; winter lives in the tails (means EDDM +84, EHAM +46). The model's month effect on held-out Jan/Jul is ±2 s: those months are unseen, so winter shows up through temperature/freezing instead | | [M] §D winter; [S] §F.10 |
| **Day of week** | raw −9 to +7 s; SHAP 1.5 s | negligible | [M]; [S] |
| **Public holidays** | raw −49 (LFPG) to +49 (EHAM), elsewhere within ±13 s; **traffic per hour is unchanged on holidays** (e.g. 39.7 vs 41.1 at EDDF); SHAP 0.2 s | negligible at these hubs | [M] §D holidays (3,540–10,239 holiday rows per airport) |
| **Market segment** | raw Regional −109, Business −107 within airport; **conditional ≈ 0** (absorbed by type and airline) | | [M]; [S] |
| **Arrivals taxiing in at my push** | raw flat (−13 to +5); SHAP 2 s | negligible given the departure queue | [M]; [S] |

## 2. Variance decomposition (held-out January + July 2025, ordinary matched rows, n = 334,562)

Held-out sd(y) = 431.7 s; predicting the training mean gives RMSE 432.1 s. Each step is its own early-stopped model.
The order was fixed in advance, and increments depend on it.

| step | adds | held-out RMSE (s) | R² | incremental R² |
|---|---|---|---|---|
| (a) | airport | 394.1 | 0.167 | +0.167 |
| (b) | + runway, stand area | 350.3 | 0.342 | **+0.175** |
| (c) | + local hour, weekday, month, holiday | 337.5 | 0.389 | +0.047 |
| (d) | + traffic, queue at push, recent taxi times, arrivals | 310.3 | 0.484 | **+0.095** |
| (e) | + size class, aircraft type, airline, segment | 301.5 | 0.512 | +0.029 |
| (e2) | + delay at push, stand turnaround | 279.4 | 0.581 | +0.069 (mostly recording, §G.2) |
| (f) | + weather | **275.3** | **0.593** | +0.012 (concentrated in freezing/fog hours) |
| (g) | + NM clock proxy (MVT − AOBT_3) | **238.6** | **0.695** | +0.101 |

Within-airport R² of the physical model f ranges from 0.416 (EGLL) to 0.592 (LEMD, LEBL). Adding the proxy lifts it
to 0.515 (LIRF) through 0.757 (EHAM) (§E.1).

## 3. What drives the > 30 min ordinary taxis (§H)

Share of ordinary taxis over 30 min: **EGLL 10.4%**, LIRF 5.2%, LTFM 4.1%, LFPG 3.5%, LEMD 1.6%, LEBL 1.3%,
EDDM 1.3%, EDDF 1.0%, EHAM 0.9%, LSZH 0.7% (pooled 3.2%, n 64,278 of 2,035,114). Over-represented levels (lift =
share among the tail ÷ share among all):
- **Winter airports (EDDM, EHAM, LSZH):** freezing, lift **22–28** (19–28% of freezing departures exceed 30 min).
  Also below 0 °C and visibility under 1.5 km (lift 7–17).
- **EGLL:** congestion: recent runway taxi 30 min+ (2.7), queue 21+ (2.5), 12+ arrivals taxiing (2.4), delay 30 min
  to 2 h (2.1–2.2). No weather dominates, because EGLL's tail is its queue.
- **LFPG:** recent runway taxi 30 min+ (13.1; 45% tail rate), below 0 °C (9.7), visibility under 1.5 km (8.9), stand
  area I5 (4.6).
- **LTFM:** recent taxi 30 min+ (11.3), freezing (10.8; 45% tail rate), runways 16R (7.6) and 34L (4.9).
- **LIRF:** **US carriers**: UAL (8.0; 42% tail rate), DAL (7.5), plus A333/B772/A339 and stand area 90. This looks
  like a long-haul stand / security-hold pattern, possibly recording. **Unresolved.**
- **LEBL:** thunderstorm (15.1). **LEMD:** THY (7.4), below 0 °C (6.9), stand areas 70–74 (4.4–5.1).
- **Physical model f under-predicts the tail by 663 s on average** on the held-out > 30 min rows (n 13,664). The
  proxy model g still under-predicts by 442 s. Much of the tail is not visible in any physical factor.

## 4. The two models, held out

| model | inputs | held-out RMSE Jan+Jul | Jan | Jul | R² |
|---|---|---|---|---|---|
| physical (f) | 33 physical columns, no own-clock information | **275.33 s** | 281.9 | 269.8 | 0.593 |
| with proxy (g) | f + MVT − AOBT_3 | **238.59 s** | 220.3 | 252.6 | 0.695 |

These numbers target y, ordinary rows only. They are not comparable to the shipped delta models' fold-A scores
(which cover every matched row and predict delta).

**How much the proxy absorbs** (block mean |SHAP|, f → g): congestion **98.7 → 54.5 s**, layout 136.5 → 98.7,
aircraft 62.9 → 42.4, weather 24.7 → 13.3, time 10.1 → 8.8. Delay + turnaround is unchanged (72.3 → 81.5), and the
proxy itself takes 152.4 s. The proxy is mostly re-measured congestion and layout. By queue alone: q_apt_at_push
goes 24.6 → 4.7 s and q_rwy_at_push 17.7 → 3.6 s.

## 5. Surprises and cautions

1. **Delay is mostly bookkeeping.** "Late flights taxi longer" (+92 s raw within-airport median at 30–60 min late) is
   largely the airport stamping off-block before the real push. The share with BLOCK more than 5 min before NM's AOBT_3
   climbs from 1–5% (on time or early) to 18–22% (late 15 min+). On clock-agreeing rows, the model's late-side delay
   effect falls to about +15 to +25 s [C]. Raw, on rows where the clocks agree within 120 s (§D delay × clock), only
   **EDDF** keeps a large late effect (+70 to +150 s over its on-time rows). EHAM and LEMD keep +30 to +50 s. EDDM,
   LFPG, LSZH and LIRF are flat. The model's early-push discount (about −70 s on clock-agreeing rows) is not
   uniform raw: early pushes are shorter at EGLL, LFPG and LSZH but not at EHAM or LEMD.
2. **Arm W's precipitation column is dead.** `w_precip_mm` is **0.0 on every row** of `data/cache_weather/`, because
   European METARs carry no `p01i` group. Two consequences for arm W: its precipitation feature is constant, and its
   `w_freezing` flag fires only on present-weather codes, never on "T ≤ 3 °C with precipitation" as documented. I
   re-derived rain/snow/fog from the same observation's `wxcodes` for the raw tables [W]; no existing file was edited.
3. **Wake class barely registers conditionally** (1.1 s mean |SHAP|) because aircraft type carries size. Read size as
   wake + type [G] (Heavy − Medium 64–98 s at nine airports). Likewise the airport column gets about 0 SHAP: runway and stand-area
   codes already imply the airport, so the layout block carries the between-airport offsets.
4. **Single-column queue SHAP flips sign.** `q_rwy_push_pre20` (−3.9 s per push) and `dep_hr` (−1.0 s per
   departure/h) have negative conditional slopes. That is credit sharing among collinear counts, not "more traffic →
   faster". Use the block curve (+20 s per queued aircraft).
5. **Stand area and runway, not aircraft, set the baseline.** Within an airport, the stand-area SHAP range (162–481 s)
   and the runway gap (up to about 325 s at EHAM/LTFM) are 3–7× the Heavy − Medium gap.
6. **Holidays do nothing at these hubs.** Hourly traffic is the same on holidays.
7. **LTFM's clocks disagree on 55% of ordinary rows.** Only 44.8% have |BLOCK − AOBT_3| ≤ 300 s, against 72–87%
   elsewhere: the known +296 s offset (CLOCK_ARTIFACTS M3). The LTFM proxy therefore reads about 5 min long.
8. **Record classes** (§A): fills are 8.3% of LIRF (12,867 matched + 527 unmatched) and 0.04–1.7% elsewhere under the
   one-sided NM rule. The plain `copy60` rule would flag 5.8–10.5% at every airport other than LIRF (20.7%), mostly on-time pushes.
   Date-slips: 14 (13 LIRF). Monsters: 188 (175 LIRF).

**Named tradeoffs** (for the owner to accept or reject):
- The physics covers matched rows only; the unmatched lane (22,219 rows, 1.1%) gets only §A.
- `prev_arr_taxiin` and `dep_hr` carry at most a one-count or which-arrival dependence on the row's own taxi length.
  I judged it negligible and did not remove it.
- The holiday list is hardcoded from memory of the 2025 calendars. Regional/city entries are the least certain.
- Month cannot generalise to held-out Jan/Jul.
- TreeSHAP interaction values were not computed: LightGBM 4.5 has no `pred_interactions` and `shap` is not
  installed, so no dependency was added. Interactions are shown by stratified SHAP instead (§F.20–21).

---

## Method (written before any fit; deviations logged at the end of this section)

**Population.** `scripts/build_submission.py` `derive(admissible(load_movements(file)))` per 2025 training month — the
shipped recipe (labelled DEP rows, TAXITIME > 0, first MVT_ID kept). Label `y = TAXITIME_SEC_mvt` (off-block to
runway). Every admissible row gets a record class:

| class | rule | why |
|---|---|---|
| fill (matched row) | \|y − sp\| ≤ 60 s **and** AOBT_3 − BLOCK > 300 s | BLOCK within a minute of SCHED while the NM clock says the aircraft left the stand ≥ 5 min later: the schedule was stamped as the off-block (reports/CLOCK_ARTIFACTS.md A1b, made one-sided so LTFM's +296 s BLOCK-after-AOBT_3 definition offset, M3, is not swept in) |
| fill (unmatched row) | \|y − sp\| ≤ 60 s **and** sp > 1,800 s | no NM clock to cross-check; a "taxi" ≥ 30 min that equals MVT − SCHED |
| date-slip | 86,400 ≤ y < 90,000 | taxi + 24 h (scripts/rome_dateslip.py `is_dateslip`) |
| monster | y > 10,800 and not a date-slip | scripts/stratum_fold.py `monster_mask` |
| ordinary | everything else | |

The plain shipped rule |y − sp| ≤ 60 s ("copy60") is reported for comparison, not used: outside LIRF it is mostly
on-time pushes with ordinary taxi-outs (CLOCK_ARTIFACTS §6a).

**Physical analysis = ordinary MATCHED rows** (the row has an NM off-block AOBT_3; 98% of admissible rows). The
engineered context exists only there: `data/cache_stand` (row-identical to the recipe's matched rows — asserted per
month: same count, same MVT_ID order via `data/cache_queue`, y bit-identical), `data/cache_queue`,
`data/cache_weather` (arm W's block exists on disk: METAR at or before AOBT_3). Unmatched rows (no NM record) get their
class shares and taxi distribution in §A only. **Tradeoff named:** the physical conclusions do not cover the
unmatched lane (1.5–2% of rows, concentrated at LIRF / EHAM / LSZH / LTFM heliport and heavily-delayed flights).

**Factor definitions.**
- airport; runway (`RUNWAY_mvt`); stand area (`stand_pref` = first two characters of STAND, within airport);
  size class = `WK_TBL_CAT_flt` (L/M/H/J); aircraft type; airline = flight-number prefix (3 chars); market segment.
- time: local hour / weekday of the **push** (AOBT_3 converted to the airport's time zone), month, public holiday
  (local date of push, list below).
- delay = AOBT_3 − SCHED.
- traffic: `dep_hr` = departures (all DEP movements in the raw month file, labelled or not) taking off within ±30 min
  of my push; `arr_hr` = arrivals landing within ±30 min of my push.
- queue at push (cache_queue, Amendment 14 definitions): `q_apt_at_push` = departures pushed before me and not yet
  airborne at my push; `q_rwy_at_push` = same, my runway; `q_rwy_push_pre20` = pushes on my runway in the 20 min
  before me; `q_arr_taxiing_at_push` = arrivals landed but not on-block at my push; `q_rwy_tko_sym10` = take-offs on
  my runway within ±10 min of my take-off.
- recent taxi times (computed here, other departures only): `rec_rwy_y20` = median y of the last 20 ORDINARY
  departures on my runway that were airborne strictly before my push; `rec_apt_y30` = same, last 30 at the airport;
  `q_arr_taxiin_sym30_push` = mean arrival taxi-in around my push; `prev_arr_taxiin` = taxi-in of the last arrival at
  my stand.
- stand turnaround `gapa` = AOBT_3 − on-block of the last arrival at my stand.
- weather at push (cache_weather): temperature, dew-point spread, wind, gust, visibility, 1 h precipitation, freezing
  flag (T ≤ 3 °C with precipitation or FZ/SN/PL/GS/GR/IC), low-visibility flag (< 1.5 km), thunder flag.

**Excluded from the physical set because they contain the row's own take-off or NM taxi (they carry the answer):**
proxy = MVT − AOBT_3; sp and every MVT − (EOBT/LOBT/IOBT) offset; ARVT − MVT; `prev_arr_gap`, `prev_dep_gap`,
`n_aobt_mvt`; the cache's `dep_dur` / `rwy_dur` (rolling medians of proxy that include the row itself); and the two
queue counts over my own [AOBT_3, MVT] window (`q_pushed_after_me`, `q_rwy_tko_in_taxi`), which scale with the proxy.
Record-fingerprint features (seconds-of-minute, callsign/type mismatch flags, record ordering) are not physics and are
not used.

**Conditional model.** LightGBM regression on y (L2), native categoricals (levels with < 300 training rows — < 100 for
runways — pooled into "other" within airport), num_leaves 63, learning rate 0.05, min_data_in_leaf 200,
feature_fraction 0.9, bagging 0.8/1, lambda_l2 1, cat_smooth 20, num_threads 2, seed 0. Trained on months
2, 4, 5, 6, 8, 10, 11, 12; early-stopped (patience 100) on months 3 and 9; evaluated on months 1 and 7 (held out,
never seen — the evaluation months' 2025 twins, the shipped fold A). TreeSHAP contributions in seconds
(`booster.predict(X, pred_contrib=True)`) on a seeded 100,000-row sample of the held-out months. **Every conditional
number is a model attribution on a held-out-validated model, not a causal effect.** SHAP credit is shared between
correlated inputs (e.g. the queue counts and `dep_hr`); read blocks, not single columns, where inputs overlap.

**Variance ladder.** Blocks added in the order the brief fixed — (a) airport; (b) + runway, stand area; (c) + local
hour, weekday, month, holiday; (d) + traffic, queue, recent taxi times; (e) + size class, aircraft type, airline,
market segment; (e2) + delay and stand turnaround; (f) + weather (= the physical model); (g) + the NM clock proxy.
Each step is its own early-stopped model; held-out R² = 1 − SSE/SST. Incremental R² depends on the order.

**Public holidays 2025** (local date of push; national + the airport's region/city; hardcoded from the official 2025
calendars, no download):
- EDDF (Hesse): 1 Jan, 18 Apr, 21 Apr, 1 May, 29 May, 9 Jun, 19 Jun, 3 Oct, 25–26 Dec.
- EDDM (Bavaria/Munich): as Germany + 6 Jan, 19 Jun, 15 Aug, 1 Nov.
- EGLL (England): 1 Jan, 18 Apr, 21 Apr, 5 May, 26 May, 25 Aug, 25–26 Dec.
- EHAM (Netherlands): 1 Jan, 18 Apr, 20–21 Apr, 26 Apr (King's Day, Saturday because 27 Apr is a Sunday), 5 May
  (Liberation Day, a lustrum year), 29 May, 8–9 Jun, 25–26 Dec.
- LEBL (Spain + Catalonia + Barcelona): 1 Jan, 6 Jan, 18 Apr, 21 Apr, 1 May, 24 Jun, 15 Aug, 11 Sep, 24 Sep, 1 Nov,
  6 Dec, 8 Dec, 25–26 Dec.
- LEMD (Spain + Comunidad de Madrid + Madrid city): 1 Jan, 6 Jan, 17–18 Apr, 1–2 May, 15 May, 25 Jul, 15 Aug, 1 Nov,
  10 Nov, 6 Dec, 8 Dec, 25 Dec.
- LFPG (France): 1 Jan, 21 Apr, 1 May, 8 May, 29 May, 9 Jun, 14 Jul, 15 Aug, 1 Nov, 11 Nov, 25 Dec.
- LIRF (Italy + Rome): 1 Jan, 6 Jan, 20–21 Apr, 25 Apr, 1 May, 2 Jun, 29 Jun, 15 Aug, 1 Nov, 8 Dec, 25–26 Dec.
- LSZH (Canton Zurich): 1–2 Jan, 18 Apr, 21 Apr, 1 May, 29 May, 9 Jun, 1 Aug, 25–26 Dec.
- LTFM (Turkey): 1 Jan, 30 Mar–1 Apr (Ramazan Bayramı), 23 Apr, 1 May, 19 May, 6–9 Jun (Kurban Bayramı), 15 Jul,
  30 Aug, 29 Oct.
Regional and city entries are from memory of the 2025 calendars and are the least certain; half-day eves are not
counted. School holidays are not modelled.

**Deviations and additions after the method was written (each with its reason):**
1. *Precipitation.* `w_precip_mm` in `data/cache_weather/` is 0.0 on every row (European METARs have no `p01i`).
   It stays in the model as a dead column (SHAP 0.0). Rain / snow / freezing precipitation / fog / heavy flags were
   re-derived from the same observation's `wxcodes` (valid = AOBT_3 − `w_age_s`, 99.997% found) for raw marginals
   only (`wxcodes.py`). The models were not refitted with them.
2. *Delay × clock diagnostic and the clocks-agree sensitivity* (`sens.py`, §G.2). Added after the delay block (e2)
   took +0.069 R², to test whether the delay effect is the airport stamping off-block before the NM push.
3. *Grouped SHAP* (`post.py`, §F.1b–j). Added after single-column SHAP showed credit sharing (wake 1.1 s, negative
   queue-count slopes).
4. *Interactions.* LightGBM 4.5 has no `pred_interactions` and `shap` is not installed (no dependency added). The
   brief's interactions are given as stratified SHAP: size × airport (§F.1c), queue × runway (§F.20), freezing ×
   airport (§F.21).
5. *Scheduling.* All loads over 1 GB and all fits waited for the feature-importance study to finish (one heavy job at
   a time): about 2 h 25 min, from 21:45 to 00:10. Every fit ran on AC power (checked before each launch) with
   `num_threads=2`. The peak RSS of any step was 4.88 GB (the marginals load); the fits peaked at 2.40 GB.

**Reproduce.** Copy `data/atlas/code/*.py` into an empty directory and run `build_frame.py` (12 frames, ~0.7 GB each),
then `marginals.py`, `model.py ladder`, `model.py shap 100000`, `sens.py`, `post.py`, `wxcodes.py` and `render.py`.
Tests: `python -m pytest test_atlas.py` (6 tests: record-class rules, windows, recent-median strictness, holiday
dates, bands, stats). The record-class test was mutation-rehearsed: flipping the NM sign and widening the date-slip
bound each turn it red.

---

# Detail

## A. Record classes per airport (every admissible departure, 2025)

Source: `build_submission.derive(admissible(load_movements(12 training files)))`; rules in §Method. `copy60` = the plain shipped `schedule_fill` rule |y − sp| ≤ 60 s with no second condition — it is mostly on-time pushes at airports other than LIRF (reports/CLOCK_ARTIFACTS.md §6a), shown for comparison only.

| airport | n admissible | n unmatched | ordinary | fill (matched / unmatched n) | date-slip (n) | monster (n) | copy60 (plain rule) |
|---|---|---|---|---|---|---|---|
| EDDF | 230,141 | 1,956 | 99.64% | 0.36% (830 m / 2 u) | 0.000% (0) | 0.000% (0) | 7.0% |
| EDDM | 167,334 | 1,010 | 98.80% | 1.20% (2,008 m / 1 u) | 0.000% (0) | 0.000% (0) | 10.3% |
| EGLL | 239,546 | 1,414 | 99.27% | 0.73% (1,747 m / 4 u) | 0.000% (0) | 0.002% (4) | 9.8% |
| EHAM | 247,948 | 4,083 | 99.96% | 0.04% (107 m / 1 u) | 0.000% (0) | 0.001% (2) | 6.9% |
| LEBL | 179,700 | 1,862 | 98.28% | 1.72% (3,089 m / 2 u) | 0.000% (0) | 0.000% (0) | 9.9% |
| LEMD | 212,242 | 947 | 99.14% | 0.86% (1,817 m / 0 u) | 0.000% (0) | 0.000% (0) | 8.6% |
| LFPG | 239,487 | 3,762 | 99.21% | 0.79% (1,890 m / 4 u) | 0.000% (0) | 0.003% (6) | 8.2% |
| LIRF | 160,704 | 1,488 | 91.55% | 8.34% (12,867 m / 527 u) | 0.008% (13) | 0.109% (175) | 20.7% |
| LSZH | 134,594 | 2,054 | 99.36% | 0.64% (856 m / 0 u) | 0.001% (1) | 0.001% (1) | 5.8% |
| LTFM | 272,963 | 3,643 | 99.22% | 0.78% (2,099 m / 26 u) | 0.000% (0) | 0.000% (0) | 10.5% |
| ALL | 2,084,659 | 22,219 | 98.65% | 1.34% (27,310 m / 567 u) | 0.001% (14) | 0.009% (188) | 9.6% |

Unmatched ordinary rows (no NM off-block; not in the physical analysis below): taxi-time distribution.

| airport | n | mean s | median s | p10 | p90 |
|---|---|---|---|---|---|
| EDDF | 1,954 | 971 | 957 | 526 | 1,395 |
| EDDM | 1,009 | 1,033 | 891 | 551 | 1,570 |
| EGLL | 1,409 | 1,549 | 1,383 | 970 | 1,992 |
| EHAM | 4,081 | 597 | 538 | 121 | 1,132 |
| LEBL | 1,860 | 1,042 | 1,038 | 603 | 1,482 |
| LEMD | 947 | 1,116 | 1,096 | 717 | 1,549 |
| LFPG | 3,756 | 1,146 | 1,023 | 652 | 1,688 |
| LIRF | 781 | 1,433 | 1,027 | 655 | 1,912 |
| LSZH | 2,052 | 709 | 718 | 11 | 1,108 |
| LTFM | 3,617 | 1,289 | 972 | 667 | 1,746 |


## B. Taxi-out distribution, ordinary matched rows

| airport | n | mean s | median s | p10 s | p90 s | sd s |
|---|---|---|---|---|---|---|
| EDDF | 227,355 | 861 | 835 | 478 | 1,245 | 327 |
| EDDM | 164,316 | 806 | 772 | 532 | 1,128 | 294 |
| EGLL | 236,382 | 1,360 | 1,318 | 908 | 1,804 | 411 |
| EHAM | 243,757 | 787 | 744 | 464 | 1,157 | 313 |
| LEBL | 174,749 | 947 | 897 | 593 | 1,342 | 321 |
| LEMD | 209,478 | 1,010 | 981 | 655 | 1,392 | 308 |
| LFPG | 233,831 | 1,013 | 953 | 655 | 1,440 | 389 |
| LIRF | 146,341 | 1,076 | 1,011 | 666 | 1,561 | 422 |
| LSZH | 131,684 | 739 | 710 | 405 | 1,074 | 294 |
| LTFM | 267,221 | 1,043 | 962 | 661 | 1,492 | 401 |
| ALL | 2,035,114 | 977 | 909 | 570 | 1,446 | 395 |

src: marginals (ordinary matched rows, 12 months 2025).


### B.1 Per runway

| airport | runway | n | mean s | median s | p10 | p90 |
|---|---|---|---|---|---|---|
| EDDF | 18 | 138,208 | 900 | 886 | 490 | 1,274 |
| EDDF | 25C | 45,501 | 785 | 734 | 478 | 1,156 |
| EDDF | 07C | 42,803 | 814 | 774 | 422 | 1,209 |
| EDDF | 25L | 457 | 886 | 850 | 611 | 1,209 |
| EDDF | 07R | 381 | 911 | 882 | 460 | 1,244 |
| EDDM | 26L | 56,730 | 756 | 718 | 471 | 1,072 |
| EDDM | 26R | 38,516 | 808 | 772 | 544 | 1,086 |
| EDDM | 08R | 36,449 | 894 | 846 | 649 | 1,151 |
| EDDM | 08L | 32,621 | 790 | 719 | 533 | 1,137 |
| EGLL | 09R | 86,570 | 1,333 | 1,309 | 904 | 1,788 |
| EGLL | 27R | 75,921 | 1,394 | 1,331 | 959 | 1,850 |
| EGLL | 27L | 73,783 | 1,355 | 1,316 | 897 | 1,809 |
| EGLL | 09L | 105 | 1,353 | 1,142 | 891 | 2,302 |
| EHAM | 24 | 84,165 | 665 | 626 | 426 | 943 |
| EHAM | 36L | 58,107 | 1,082 | 1,049 | 798 | 1,396 |
| EHAM | 18L | 55,081 | 735 | 700 | 513 | 967 |
| EHAM | 36C | 30,634 | 728 | 710 | 495 | 978 |
| EHAM | 09 | 7,789 | 839 | 800 | 581 | 1,100 |
| EHAM | 22 | 3,627 | 291 | 251 | 142 | 489 |
| EHAM | 04 | 2,182 | 380 | 358 | 261 | 535 |
| EHAM | 18C | 1,985 | 882 | 847 | 637 | 1,135 |
| EHAM | 06 | 127 | 799 | 758 | 543 | 1,111 |
| EHAM | 27 | 60 | 799 | 811 | 562 | 1,011 |
| LEBL | 24L | 121,263 | 951 | 905 | 612 | 1,329 |
| LEBL | 06R | 50,894 | 930 | 868 | 554 | 1,363 |
| LEBL | 24R | 1,610 | 1,222 | 1,199 | 818 | 1,622 |
| LEBL | 06L | 978 | 934 | 866 | 322 | 1,570 |
| LEMD | 36R | 83,549 | 1,074 | 1,047 | 750 | 1,435 |
| LEMD | 36L | 83,185 | 936 | 900 | 592 | 1,321 |
| LEMD | 14L | 25,288 | 1,082 | 1,047 | 742 | 1,465 |
| LEMD | 14R | 17,448 | 952 | 908 | 629 | 1,331 |
| LFPG | 08L | 73,668 | 975 | 902 | 656 | 1,376 |
| LFPG | 26R | 69,903 | 992 | 904 | 608 | 1,445 |
| LFPG | 09R | 31,443 | 1,076 | 1,011 | 653 | 1,553 |
| LFPG | 27L | 26,170 | 982 | 907 | 658 | 1,370 |
| LFPG | 27R | 20,924 | 1,098 | 1,028 | 771 | 1,491 |
| LFPG | 09L | 11,311 | 1,121 | 1,083 | 726 | 1,507 |
| LFPG | 26L | 216 | 1,078 | 1,026 | 716 | 1,551 |
| LFPG | 08R | 196 | 992 | 957 | 716 | 1,374 |
| LIRF | 25 | 131,826 | 1,041 | 964 | 661 | 1,498 |
| LIRF | 16R | 10,350 | 1,426 | 1,319 | 952 | 2,032 |
| LIRF | 34L | 3,666 | 1,290 | 1,195 | 841 | 1,848 |
| LIRF | 16L | 351 | 1,706 | 1,621 | 1,258 | 2,228 |
| LIRF | 34R | 132 | 1,415 | 1,321 | 913 | 1,849 |
| LSZH | 28 | 83,002 | 687 | 671 | 351 | 990 |
| LSZH | 32 | 32,277 | 747 | 717 | 474 | 1,036 |
| LSZH | 16 | 12,519 | 1,022 | 967 | 695 | 1,389 |
| LSZH | 10 | 2,518 | 854 | 820 | 551 | 1,189 |
| LSZH | 34 | 1,367 | 912 | 874 | 514 | 1,306 |
| LTFM | 36 | 124,239 | 925 | 891 | 610 | 1,264 |
| LTFM | 35L | 77,877 | 1,080 | 1,009 | 671 | 1,555 |
| LTFM | 18 | 31,205 | 1,059 | 1,018 | 779 | 1,387 |
| LTFM | 17R | 20,374 | 1,334 | 1,264 | 888 | 1,857 |
| LTFM | 34L | 8,283 | 1,524 | 1,268 | 905 | 2,469 |
| LTFM | 35R | 3,154 | 1,129 | 1,025 | 669 | 1,679 |
| LTFM | 16R | 1,405 | 1,672 | 1,629 | 1,256 | 2,106 |
| LTFM | 17L | 392 | 1,300 | 1,204 | 793 | 1,800 |
| LTFM | 34R | 292 | 1,303 | 1,271 | 957 | 1,688 |


## C. Marginal effects (raw, confounded) — pooled over airports

`within-airport Δ` = mean of (y − that airport's mean) and median of (y − that airport's median): the marginal with the airport mix removed, nothing else. src: marginals (ordinary matched rows, 12 months 2025).


### Wake / size class

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| H | 415,479 | 1,165 | 1,088 | 729 | 1,671 | 134 | 124 |
| J | 14,452 | 1,412 | 1,326 | 970 | 1,917 | 285 | 259 |
| L | 10,101 | 695 | 601 | 245 | 1,285 | -134 | -173 |
| M | 1,595,008 | 926 | 868 | 543 | 1,372 | -37 | -42 |
| NA | 74 | 562 | 460 | 234 | 982 | -223 | -270 |


### Market segment

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| Business | 44,019 | 773 | 690 | 264 | 1,386 | -88 | -107 |
| Cargo | 59,094 | 1,008 | 949 | 593 | 1,503 | 47 | 28 |
| Lowcost | 390,350 | 951 | 906 | 596 | 1,336 | 6 | 14 |
| Mainline | 1,310,392 | 1,025 | 955 | 609 | 1,508 | 17 | 6 |
| Military | 703 | 816 | 717 | 287 | 1,406 | -53 | -109 |
| Non-Scheduled | 14,160 | 947 | 898 | 507 | 1,432 | 6 | 13 |
| Other | 6,584 | 878 | 829 | 424 | 1,383 | -75 | -84 |
| Regional | 209,812 | 767 | 723 | 417 | 1,148 | -109 | -109 |


### Top-20 aircraft types (by n)

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| A320 | 335,895 | 922 | 841 | 570 | 1,374 | -54 | -59 |
| B738 | 211,543 | 958 | 928 | 581 | 1,338 | 2 | 12 |
| A20N | 166,372 | 1,008 | 937 | 623 | 1,492 | -40 | -43 |
| A321 | 159,180 | 904 | 841 | 593 | 1,269 | -53 | -57 |
| A21N | 144,987 | 1,046 | 978 | 657 | 1,493 | 34 | 38 |
| A319 | 132,393 | 932 | 847 | 546 | 1,390 | -52 | -57 |
| E190 | 77,532 | 801 | 755 | 473 | 1,172 | -62 | -61 |
| BCS3 | 75,982 | 937 | 895 | 600 | 1,323 | -6 | 6 |
| B77W | 65,974 | 1,191 | 1,131 | 769 | 1,683 | 126 | 119 |
| B789 | 59,815 | 1,210 | 1,149 | 782 | 1,680 | 141 | 132 |
| A359 | 54,972 | 1,117 | 1,027 | 736 | 1,570 | 124 | 118 |
| A333 | 46,346 | 1,097 | 1,016 | 717 | 1,555 | 122 | 111 |
| B38M | 44,131 | 1,130 | 1,091 | 773 | 1,501 | 143 | 173 |
| B772 | 33,393 | 1,265 | 1,207 | 780 | 1,793 | 172 | 149 |
| A332 | 31,077 | 1,080 | 1,013 | 663 | 1,561 | 74 | 63 |
| CRJX | 29,630 | 810 | 797 | 426 | 1,168 | -164 | -154 |
| E195 | 28,562 | 775 | 727 | 413 | 1,167 | -56 | -55 |
| E295 | 26,827 | 782 | 740 | 470 | 1,142 | -61 | -46 |
| B788 | 23,024 | 1,155 | 1,085 | 738 | 1,630 | 110 | 113 |
| CRJ9 | 20,479 | 562 | 534 | 289 | 850 | -264 | -244 |


### Top-20 airlines (flight-number prefix, by n)

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| THY | 216,683 | 1,053 | 965 | 667 | 1,498 | 12 | 4 |
| AFR | 133,797 | 994 | 907 | 655 | 1,390 | -21 | -46 |
| BAW | 115,152 | 1,394 | 1,327 | 900 | 1,867 | 83 | 60 |
| DLH | 101,740 | 815 | 769 | 533 | 1,191 | -54 | -56 |
| KL1 | 98,416 | 718 | 672 | 457 | 1,031 | -71 | -74 |
| VLG | 79,518 | 826 | 773 | 563 | 1,135 | -138 | -130 |
| SWR | 73,852 | 789 | 750 | 417 | 1,180 | -4 | 3 |
| RYR | 66,339 | 1,129 | 1,113 | 807 | 1,446 | 126 | 165 |
| IBE | 60,837 | 952 | 908 | 663 | 1,270 | -65 | -64 |
| LH1 | 55,389 | 744 | 735 | 379 | 1,073 | -116 | -100 |
| ITY | 47,185 | 1,018 | 953 | 611 | 1,502 | -42 | -51 |
| EJU | 37,009 | 929 | 864 | 605 | 1,310 | 8 | 12 |
| AEA | 34,328 | 994 | 968 | 658 | 1,349 | -3 | 2 |
| KL0 | 29,242 | 843 | 813 | 481 | 1,213 | 56 | 69 |
| ANE | 27,042 | 810 | 792 | 446 | 1,152 | -191 | -178 |
| EZY | 19,871 | 996 | 955 | 665 | 1,362 | 80 | 79 |
| IBS | 18,328 | 809 | 781 | 559 | 1,090 | -201 | -200 |
| WMT | 18,322 | 1,139 | 1,128 | 786 | 1,481 | 99 | 134 |
| SHT | 18,010 | 1,322 | 1,264 | 900 | 1,789 | -37 | -54 |
| LH8 | 16,957 | 743 | 708 | 465 | 1,041 | -118 | -127 |


### Local push hour (AOBT_3, airport local time)

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| 0 | 10,974 | 976 | 903 | 638 | 1,374 | -44 | -60 |
| 1 | 16,401 | 1,001 | 954 | 663 | 1,383 | -35 | -8 |
| 2 | 9,649 | 1,265 | 1,138 | 719 | 1,992 | 229 | 177 |
| 3 | 6,303 | 1,013 | 951 | 601 | 1,445 | -10 | -5 |
| 4 | 10,900 | 1,039 | 976 | 600 | 1,511 | 42 | 59 |
| 5 | 18,597 | 857 | 829 | 533 | 1,194 | -112 | -101 |
| 6 | 73,865 | 859 | 829 | 531 | 1,214 | -126 | -113 |
| 7 | 117,703 | 930 | 881 | 549 | 1,330 | -43 | -52 |
| 8 | 106,454 | 946 | 889 | 529 | 1,435 | -39 | -54 |
| 9 | 123,514 | 979 | 906 | 541 | 1,493 | 8 | 2 |
| 10 | 130,417 | 1,042 | 973 | 598 | 1,551 | 76 | 66 |
| 11 | 128,212 | 1,016 | 953 | 591 | 1,505 | 49 | 38 |
| 12 | 139,679 | 981 | 922 | 570 | 1,447 | 23 | 18 |
| 13 | 128,466 | 1,025 | 970 | 608 | 1,493 | 50 | 61 |
| 14 | 116,600 | 1,019 | 957 | 591 | 1,511 | 37 | 38 |
| 15 | 128,267 | 1,006 | 947 | 593 | 1,505 | 23 | 15 |
| 16 | 128,382 | 980 | 908 | 552 | 1,494 | 2 | 3 |
| 17 | 112,380 | 931 | 868 | 537 | 1,392 | -43 | -45 |
| 18 | 98,471 | 950 | 896 | 587 | 1,385 | -48 | -48 |
| 19 | 97,062 | 981 | 907 | 590 | 1,450 | -23 | -25 |
| 20 | 114,202 | 950 | 899 | 579 | 1,382 | -33 | -31 |
| 21 | 124,017 | 929 | 887 | 552 | 1,337 | -20 | -10 |
| 22 | 73,413 | 1,007 | 954 | 591 | 1,492 | 28 | 13 |
| 23 | 21,186 | 1,025 | 956 | 612 | 1,491 | 33 | 7 |


### Local day of week

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| Fri | 300,261 | 991 | 923 | 577 | 1,488 | 16 | 7 |
| Mon | 293,736 | 980 | 911 | 575 | 1,446 | 3 | 2 |
| Sat | 279,013 | 973 | 905 | 567 | 1,442 | -5 | -4 |
| Sun | 293,630 | 987 | 911 | 574 | 1,456 | 10 | 4 |
| Thu | 291,762 | 975 | 908 | 568 | 1,443 | -2 | 0 |
| Tue | 284,263 | 962 | 902 | 561 | 1,434 | -16 | -9 |
| Wed | 292,449 | 972 | 906 | 567 | 1,442 | -5 | -2 |


### Month

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| 1 | 150,611 | 985 | 904 | 560 | 1,487 | -0 | -27 |
| 2 | 140,923 | 986 | 902 | 564 | 1,452 | 3 | -23 |
| 3 | 161,311 | 962 | 903 | 570 | 1,434 | -16 | -7 |
| 4 | 171,877 | 961 | 906 | 584 | 1,397 | -16 | -2 |
| 5 | 181,108 | 975 | 914 | 578 | 1,439 | 1 | 6 |
| 6 | 178,154 | 958 | 899 | 551 | 1,434 | -16 | -7 |
| 7 | 183,951 | 1,002 | 934 | 574 | 1,498 | 28 | 13 |
| 8 | 185,915 | 976 | 916 | 567 | 1,442 | 2 | 6 |
| 9 | 179,015 | 975 | 909 | 563 | 1,447 | 1 | 5 |
| 10 | 181,287 | 979 | 914 | 571 | 1,447 | 5 | 8 |
| 11 | 158,970 | 986 | 916 | 574 | 1,476 | 6 | 0 |
| 12 | 161,992 | 983 | 908 | 585 | 1,455 | 0 | -3 |


### Season

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| autumn SON | 519,272 | 980 | 912 | 569 | 1,449 | 4 | 4 |
| spring MAM | 514,296 | 966 | 907 | 577 | 1,432 | -10 | 0 |
| summer JJA | 548,020 | 979 | 910 | 564 | 1,448 | 5 | 4 |
| winter DJF | 453,526 | 984 | 905 | 570 | 1,464 | 1 | -13 |


### Public holiday (local date of push)

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| holiday | 64,615 | 960 | 903 | 583 | 1,401 | -10 | -2 |
| not | 1,970,499 | 978 | 909 | 570 | 1,446 | 0 | 0 |


### Departure delay at push (AOBT_3 − SCHED)

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| early 0-5m | 292,634 | 916 | 886 | 543 | 1,328 | -86 | -63 |
| early >5m | 243,326 | 887 | 845 | 548 | 1,265 | -154 | -121 |
| late 0-5m | 357,414 | 943 | 897 | 547 | 1,391 | -35 | -18 |
| late 1-2h | 73,911 | 1,127 | 1,010 | 603 | 1,690 | 153 | 77 |
| late 15-30m | 343,504 | 1,031 | 963 | 598 | 1,553 | 79 | 68 |
| late 30-60m | 213,559 | 1,096 | 1,010 | 609 | 1,672 | 132 | 92 |
| late 5-15m | 485,207 | 965 | 906 | 566 | 1,439 | 11 | 13 |
| late >2h | 25,559 | 1,105 | 991 | 603 | 1,629 | 122 | 65 |


### Departures in ±30 min of push (traffic/hour)

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| 0-9 | 40,891 | 909 | 840 | 534 | 1,325 | -56 | -63 |
| 10-19 | 172,816 | 870 | 799 | 532 | 1,260 | -57 | -61 |
| 20-29 | 429,501 | 913 | 847 | 544 | 1,322 | -28 | -43 |
| 30-39 | 557,849 | 991 | 922 | 590 | 1,448 | 2 | 0 |
| 40-49 | 534,288 | 1,064 | 1,014 | 605 | 1,569 | 16 | 12 |
| 50-59 | 233,906 | 961 | 917 | 553 | 1,390 | 43 | 54 |
| 60+ | 65,863 | 963 | 908 | 566 | 1,420 | 60 | 66 |


### Arrivals landing in ±30 min of push

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| 0-9 | 101,196 | 966 | 902 | 579 | 1,404 | -10 | -13 |
| 10-19 | 292,064 | 915 | 850 | 540 | 1,328 | 0 | -1 |
| 20-29 | 536,796 | 925 | 866 | 550 | 1,335 | -8 | -8 |
| 30-39 | 603,550 | 1,013 | 955 | 590 | 1,505 | 6 | 3 |
| 40-49 | 371,474 | 1,077 | 1,023 | 616 | 1,571 | 9 | 7 |
| 50+ | 130,034 | 889 | 846 | 524 | 1,288 | -11 | -4 |


### Queue at push, airport (pushed, not yet airborne)

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| 0 | 20,786 | 840 | 777 | 480 | 1,256 | -80 | -91 |
| 1-2 | 105,339 | 816 | 768 | 484 | 1,199 | -79 | -77 |
| 12-15 | 402,152 | 1,033 | 979 | 610 | 1,493 | 12 | 9 |
| 16-20 | 243,152 | 1,205 | 1,147 | 712 | 1,739 | 102 | 112 |
| 21+ | 66,653 | 1,268 | 1,196 | 768 | 1,853 | 188 | 181 |
| 3-4 | 215,307 | 844 | 786 | 508 | 1,224 | -60 | -57 |
| 5-7 | 446,958 | 895 | 843 | 542 | 1,292 | -30 | -36 |
| 8-11 | 534,767 | 955 | 904 | 582 | 1,376 | -12 | -5 |


### Queue at push, same runway

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| 0 | 72,489 | 861 | 787 | 441 | 1,330 | -41 | -58 |
| 1-2 | 271,288 | 846 | 784 | 500 | 1,250 | -58 | -60 |
| 12-15 | 151,677 | 1,221 | 1,189 | 790 | 1,669 | 49 | 57 |
| 16-20 | 94,119 | 1,466 | 1,435 | 1,025 | 1,909 | 134 | 126 |
| 21+ | 14,866 | 1,611 | 1,563 | 1,146 | 2,050 | 256 | 246 |
| 3-4 | 401,952 | 875 | 831 | 528 | 1,264 | -43 | -48 |
| 5-7 | 606,225 | 931 | 890 | 565 | 1,330 | -8 | -2 |
| 8-11 | 422,498 | 1,026 | 971 | 651 | 1,440 | 41 | 53 |


### Arrivals taxiing in at push

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| 0 | 106,818 | 898 | 843 | 541 | 1,308 | -15 | -13 |
| 1-2 | 477,118 | 913 | 853 | 547 | 1,323 | 0 | 1 |
| 12+ | 79,065 | 1,020 | 955 | 606 | 1,498 | 13 | 5 |
| 3-4 | 506,509 | 963 | 903 | 563 | 1,431 | -3 | 0 |
| 5-7 | 592,863 | 1,038 | 968 | 596 | 1,555 | -0 | 0 |
| 8-11 | 272,741 | 1,003 | 943 | 569 | 1,505 | 8 | 2 |


### Recent taxi: median of the last 20 ordinary take-offs on my runway before my push

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| 10-12.5m | 361,502 | 750 | 720 | 474 | 1,063 | -81 | -63 |
| 12.5-15m | 561,820 | 890 | 847 | 572 | 1,254 | -48 | -41 |
| 15-17.5m | 475,877 | 1,003 | 962 | 653 | 1,387 | 16 | 25 |
| 17.5-20m | 263,623 | 1,115 | 1,075 | 717 | 1,555 | 61 | 66 |
| 20-25m | 214,559 | 1,288 | 1,252 | 830 | 1,789 | 98 | 79 |
| 25-30m | 60,966 | 1,462 | 1,432 | 952 | 1,981 | 211 | 184 |
| 30m+ | 15,115 | 1,772 | 1,609 | 909 | 2,875 | 634 | 429 |
| <10m | 76,266 | 632 | 609 | 358 | 914 | -152 | -135 |
| NA | 5,386 | 1,053 | 952 | 549 | 1,690 | 80 | 48 |


### Temperature at push

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| 0-3C | 82,723 | 1,008 | 871 | 535 | 1,630 | 104 | 15 |
| 10-20C | 913,920 | 969 | 907 | 565 | 1,440 | -14 | -2 |
| 20-30C | 528,846 | 998 | 949 | 595 | 1,457 | 10 | 8 |
| 3-10C | 405,067 | 940 | 892 | 550 | 1,386 | -34 | -29 |
| 30C+ | 66,264 | 1,040 | 985 | 620 | 1,500 | 45 | 36 |
| <0C | 38,224 | 1,103 | 910 | 540 | 1,917 | 245 | 97 |
| NA | 70 | 738 | 720 | 488 | 972 | -98 | -105 |


### Precipitation (1 h)

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| dry | 2,035,112 | 977 | 909 | 570 | 1,446 | 0 | 0 |


### Visibility

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| 1.5-5km | 40,330 | 1,028 | 907 | 552 | 1,611 | 99 | 36 |
| 5-9.5km | 146,382 | 972 | 893 | 549 | 1,463 | 24 | 0 |
| <1.5km | 19,225 | 1,157 | 970 | 592 | 1,925 | 268 | 139 |
| >=9.5km | 1,829,157 | 975 | 909 | 572 | 1,442 | -7 | -1 |


### Wind

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| 0-9kt | 1,383,940 | 964 | 903 | 559 | 1,434 | -4 | -2 |
| 10-19kt | 604,484 | 1,005 | 949 | 590 | 1,500 | 7 | 4 |
| 20-29kt | 46,080 | 1,016 | 918 | 597 | 1,501 | 21 | 2 |
| 30kt+ | 530 | 1,005 | 896 | 555 | 1,561 | 98 | 58 |
| NA | 80 | 862 | 876 | 536 | 1,182 | -48 | -8 |


### Freezing conditions (T ≤ 3 °C with precipitation / FZ-SN-PL codes)

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| freezing | 8,948 | 1,607 | 1,323 | 633 | 2,994 | 707 | 488 |
| not | 2,026,164 | 974 | 908 | 570 | 1,443 | -3 | 0 |


### Thunderstorm code

| level | n | mean s | median s | p10 | p90 | within-airport Δmean s | within-airport Δmedian s |
|---|---|---|---|---|---|---|---|
| not | 2,028,952 | 976 | 908 | 570 | 1,445 | -1 | 0 |
| thunder | 6,160 | 1,259 | 1,089 | 634 | 2,095 | 313 | 192 |


## D. Marginal effects per airport (raw) — Δ vs the airport's own median

src: marginals (ordinary matched rows, 12 months 2025). Levels with n < 50 (stand areas: n < 200) are omitted.


### D. size class × airport (Δ median s, n)

| airport | H | J | L | M |
|---|---|---|---|---|
| EDDF | +188 (51,370) | +410 (856) | -93 (675) | -59 (174,452) |
| EDDM | +137 (15,203) | +313 (2,272) | -173 (1,817) | -44 (145,011) |
| EGLL | +67 (85,891) | +186 (7,219) | — | -61 (143,272) |
| EHAM | +187 (42,204) | +389 (718) | -423 (1,461) | -39 (199,361) |
| LEBL | +36 (12,418) | +218 (354) | +225 (1,236) | -7 (160,737) |
| LEMD | +72 (39,065) | +186 (355) | +194 (1,236) | -14 (168,819) |
| LFPG | +130 (63,265) | +300 (1,614) | — | -57 (168,905) |
| LIRF | +302 (19,129) | +246 (367) | — | -52 (126,843) |
| LSZH | +194 (15,997) | +317 (347) | -258 (3,529) | -20 (111,772) |
| LTFM | +51 (70,937) | +469 (350) | -334 (98) | -5 (195,836) |


### D. local push hour band × airport (Δ median s, n)

| airport | 00-05 | 06-08 | 09-11 | 12-14 | 15-17 | 18-20 | 21-23 |
|---|---|---|---|---|---|---|---|
| EDDF | +34 (3,129) | -101 (28,719) | +35 (49,633) | +57 (50,152) | -39 (39,002) | -2 (27,372) | +10 (29,348) |
| EDDM | -47 (1,057) | -51 (24,914) | +11 (36,790) | +60 (26,119) | -40 (32,818) | -49 (23,345) | -40 (19,273) |
| EGLL | -482 (522) | -187 (36,553) | +54 (45,522) | +57 (45,069) | +118 (45,802) | -10 (43,384) | -59 (19,530) |
| EHAM | -116 (2,637) | -65 (31,665) | +16 (52,163) | +9 (50,809) | -3 (45,772) | +3 (31,682) | +31 (29,029) |
| LEBL | -130 (4,874) | -67 (29,613) | +102 (30,833) | +44 (30,220) | -5 (30,263) | -24 (30,290) | +40 (18,656) |
| LEMD | -14 (8,365) | -35 (33,083) | +12 (37,046) | +51 (33,761) | +8 (46,404) | -35 (32,320) | +10 (18,499) |
| LFPG | +68 (12,119) | -110 (23,571) | +74 (50,088) | +56 (47,237) | -52 (37,234) | -109 (34,654) | -2 (28,928) |
| LIRF | -101 (4,740) | -104 (18,575) | +123 (29,103) | +64 (26,484) | -51 (24,695) | -61 (22,170) | -40 (20,574) |
| LSZH | -97 (590) | -18 (22,680) | -2 (24,194) | +74 (30,135) | -44 (22,645) | -22 (19,444) | +22 (11,996) |
| LTFM | -6 (34,791) | +6 (48,649) | -52 (26,771) | -2 (44,759) | +48 (44,394) | +2 (45,074) | -8 (22,783) |


### D. delay band × airport (Δ median s, n)

| airport | early 0-5m | early >5m | late 0-5m | late 1-2h | late 15-30m | late 30-60m | late 5-15m | late >2h |
|---|---|---|---|---|---|---|---|---|
| EDDF | -83 (30,077) | -75 (14,847) | -43 (41,461) | +118 (6,951) | +63 (43,843) | +105 (24,970) | -1 (62,693) | +152 (2,513) |
| EDDM | -112 (26,233) | -165 (11,146) | -48 (32,544) | +117 (4,566) | +76 (29,919) | +120 (16,544) | +11 (42,347) | +79 (1,017) |
| EGLL | -67 (46,389) | -171 (40,601) | +8 (48,378) | +127 (6,935) | +113 (27,169) | +125 (19,205) | +63 (45,523) | +73 (2,182) |
| EHAM | -24 (33,597) | -22 (17,960) | -23 (47,997) | +25 (9,474) | +16 (41,314) | +25 (24,016) | +4 (65,869) | +63 (3,530) |
| LEBL | -69 (23,823) | -62 (15,238) | -66 (31,140) | +138 (7,909) | +83 (29,630) | +136 (20,011) | -12 (44,405) | +92 (2,593) |
| LEMD | -56 (27,593) | -45 (12,455) | -40 (38,557) | +121 (8,490) | +49 (38,372) | +106 (23,286) | -9 (57,991) | +76 (2,734) |
| LFPG | -106 (28,914) | -117 (21,238) | -48 (34,553) | +60 (11,204) | +61 (48,157) | +70 (33,085) | +2 (52,842) | +59 (3,838) |
| LIRF | -115 (20,397) | -180 (15,600) | -21 (25,540) | +72 (6,729) | +81 (26,667) | +120 (18,976) | +14 (30,463) | +79 (1,969) |
| LSZH | -118 (10,041) | -189 (4,916) | -55 (21,500) | +81 (4,512) | +69 (31,445) | +79 (17,001) | +8 (41,423) | +56 (846) |
| LTFM | +1 (45,570) | -126 (89,325) | +110 (35,744) | +59 (7,141) | +183 (26,988) | +125 (16,465) | +171 (41,651) | -5 (4,337) |


### D. runway queue at push × airport (Δ median s, n)

| airport | 0 | 1-2 | 12-15 | 3-4 | 5-7 | 8-11 | 16-20 | 21+ |
|---|---|---|---|---|---|---|---|---|
| EDDF | -121 (6,404) | -112 (33,456) | +203 (4,582) | -57 (50,646) | +16 (78,563) | +110 (53,694) | — | — |
| EDDM | -58 (12,500) | -47 (57,661) | — | +3 (54,776) | +18 (34,164) | +74 (5,169) | — | — |
| EGLL | -237 (982) | -185 (2,172) | -53 (77,967) | -235 (3,291) | -241 (10,805) | -179 (39,006) | +123 (87,460) | +245 (14,699) |
| EHAM | -271 (9,201) | -138 (20,004) | +274 (15,809) | -90 (39,596) | -23 (84,990) | +84 (72,426) | +469 (1,696) | — |
| LEBL | +27 (4,599) | -83 (9,999) | +216 (1,328) | -80 (26,860) | -24 (75,569) | +74 (56,392) | — | — |
| LEMD | -53 (6,773) | -49 (34,519) | +153 (615) | -24 (58,257) | +13 (81,623) | +67 (27,691) | — | — |
| LFPG | -106 (8,717) | -104 (34,794) | +259 (6,525) | -59 (53,243) | -2 (80,782) | +118 (49,459) | +423 (310) | — |
| LIRF | +179 (5,971) | -45 (17,181) | +126 (2,565) | -99 (32,844) | -41 (56,408) | +68 (31,290) | +282 (82) | — |
| LSZH | +54 (10,943) | -29 (36,034) | — | -11 (46,227) | +28 (37,354) | +140 (1,124) | — | — |
| LTFM | -111 (6,399) | -64 (25,468) | +107 (42,238) | -12 (36,212) | -7 (65,967) | +1 (86,247) | +179 (4,559) | +526 (131) |


### D. freezing × airport (Δ median s, n)

| airport | freezing | not |
|---|---|---|
| EDDF | +205 (575) | -1 (226,780) |
| EDDM | +658 (2,338) | -1 (161,978) |
| EGLL | +11 (193) | +0 (236,189) |
| EHAM | +184 (1,381) | -1 (242,376) |
| LEBL | — | +0 (174,749) |
| LEMD | — | +0 (209,472) |
| LFPG | +1206 (849) | +0 (232,982) |
| LIRF | — | +0 (146,341) |
| LSZH | +273 (1,017) | -1 (130,667) |
| LTFM | +549 (2,589) | +0 (264,630) |


### D. holiday × airport (Δ median s, n)

| airport | holiday | not |
|---|---|---|
| EDDF | -13 (5,908) | +0 (221,447) |
| EDDM | -3 (5,610) | +0 (158,706) |
| EGLL | -3 (5,079) | +0 (231,303) |
| EHAM | +48 (7,381) | -1 (236,376) |
| LEBL | -9 (6,643) | +1 (168,106) |
| LEMD | -4 (7,939) | +1 (201,539) |
| LFPG | -49 (7,019) | +0 (226,812) |
| LIRF | -1 (5,257) | +0 (141,084) |
| LSZH | +2 (3,540) | +0 (128,144) |
| LTFM | -3 (10,239) | +0 (256,982) |


### D. market segment × airport (Δ median s, n)

| airport | Business | Cargo | Lowcost | Mainline | Military | Non-Scheduled | Other | Regional |
|---|---|---|---|---|---|---|---|---|
| EDDF | -58 (3,383) | +59 (11,231) | +88 (31,164) | -20 (145,154) | -322 (52) | +115 (967) | +19 (345) | -15 (35,059) |
| EDDM | -170 (6,170) | +19 (1,868) | +60 (21,552) | +10 (102,607) | -46 (199) | +18 (1,519) | +5 (417) | -161 (29,984) |
| EGLL | — | -166 (1,083) | -166 (5,702) | +3 (226,482) | — | +52 (283) | -3 (370) | -229 (2,459) |
| EHAM | -422 (5,886) | +275 (6,398) | +19 (51,940) | +34 (120,167) | -241 (103) | +11 (2,144) | -157 (681) | -83 (56,438) |
| LEBL | +200 (5,669) | -127 (2,922) | +33 (121,436) | -65 (40,919) | — | +216 (989) | -226 (234) | -226 (2,536) |
| LEMD | +295 (8,371) | +73 (5,646) | +63 (51,572) | -11 (114,582) | -59 (115) | +32 (2,441) | -218 (878) | -157 (25,873) |
| LFPG | -298 (387) | +123 (16,640) | -50 (27,367) | +5 (158,313) | +32 (56) | -44 (1,981) | -64 (1,027) | -115 (28,060) |
| LIRF | — | -171 (1,330) | -48 (59,248) | +14 (80,112) | — | +14 (749) | -116 (415) | -117 (4,452) |
| LSZH | -228 (11,925) | +337 (369) | -14 (12,401) | +41 (85,081) | -196 (97) | -6 (808) | -6 (992) | -117 (20,011) |
| LTFM | -414 (2,197) | -187 (11,607) | -5 (7,968) | +5 (236,975) | — | -58 (2,279) | -62 (1,225) | -55 (4,940) |


### D. Stand areas per airport (first two characters of STAND; n ≥ 200): the 5 shortest and 5 longest by median

| airport | end | stand area | n | median s | Δ median s | p90 s |
|---|---|---|---|---|---|---|
| EDDF | short | B1 | 2,295 | 567 | -268 | 831 |
| EDDF | short | V2 | 2,843 | 613 | -222 | 982 |
| EDDF | short | V1 | 70,079 | 697 | -138 | 1,166 |
| EDDF | short | C2 | 956 | 707 | -128 | 1,016 |
| EDDF | short | A4 | 1,813 | 721 | -114 | 993 |
| EDDF | long | K6 | 419 | 1,124 | +289 | 1,485 |
| EDDF | long | E6 | 1,921 | 1,145 | +310 | 1,495 |
| EDDF | long | E9 | 1,860 | 1,159 | +324 | 1,521 |
| EDDF | long | K4 | 412 | 1,201 | +366 | 1,589 |
| EDDF | long | V3 | 2,788 | 1,239 | +404 | 1,673 |
| EDDM | short | 34 | 10,042 | 540 | -232 | 839 |
| EDDM | short | G1 | 517 | 590 | -182 | 902 |
| EDDM | short | 58 | 2,599 | 597 | -175 | 908 |
| EDDM | short | 59 | 1,310 | 600 | -172 | 908 |
| EDDM | short | G2 | 541 | 605 | -167 | 900 |
| EDDM | long | 18 | 9,861 | 837 | +65 | 1,139 |
| EDDM | long | 16 | 3,176 | 842 | +70 | 1,146 |
| EDDM | long | 11 | 18,929 | 846 | +74 | 1,196 |
| EDDM | long | 35 | 4,877 | 848 | +76 | 1,207 |
| EDDM | long | 37 | 1,447 | 899 | +127 | 1,265 |
| EGLL | short | 25 | 1,745 | 1,085 | -233 | 1,569 |
| EGLL | short | 61 | 693 | 1,085 | -233 | 1,569 |
| EGLL | short | 22 | 19,491 | 1,129 | -189 | 1,553 |
| EGLL | short | 21 | 15,116 | 1,191 | -127 | 1,620 |
| EGLL | short | 42 | 4,479 | 1,249 | -69 | 1,681 |
| EGLL | long | 32 | 13,770 | 1,386 | +68 | 1,857 |
| EGLL | long | 34 | 2,738 | 1,390 | +72 | 1,808 |
| EGLL | long | 53 | 10,814 | 1,428 | +110 | 1,920 |
| EGLL | long | 56 | 8,199 | 1,439 | +121 | 1,928 |
| EGLL | long | 55 | 8,475 | 1,445 | +127 | 1,978 |
| EHAM | short | K3 | 2,694 | 316 | -428 | 637 |
| EHAM | short | K1 | 1,084 | 324 | -420 | 604 |
| EHAM | short | K2 | 1,901 | 326 | -418 | 649 |
| EHAM | short | K7 | 598 | 342 | -402 | 808 |
| EHAM | short | A7 | 7,500 | 650 | -94 | 975 |
| EHAM | long | S8 | 1,445 | 995 | +251 | 1,509 |
| EHAM | long | S7 | 1,481 | 997 | +253 | 1,546 |
| EHAM | long | S6 | 1,602 | 998 | +254 | 1,592 |
| EHAM | long | J8 | 376 | 1,009 | +265 | 1,344 |
| EHAM | long | R7 | 1,036 | 1,220 | +476 | 1,590 |
| LEBL | short | 42 | 1,047 | 491 | -406 | 878 |
| LEBL | short | 41 | 1,687 | 496 | -401 | 946 |
| LEBL | short | 33 | 1,640 | 561 | -336 | 937 |
| LEBL | short | 40 | 725 | 587 | -310 | 1,217 |
| LEBL | short | 32 | 1,302 | 604 | -293 | 1,048 |
| LEBL | long | 11 | 22,539 | 1,186 | +289 | 1,483 |
| LEBL | long | 12 | 9,478 | 1,194 | +297 | 1,500 |
| LEBL | long | 10 | 20,090 | 1,195 | +298 | 1,481 |
| LEBL | long | 95 | 1,096 | 1,264 | +366 | 1,562 |
| LEBL | long | 96 | 1,693 | 1,273 | +376 | 1,546 |
| LEMD | short | 60 | 2,523 | 541 | -440 | 956 |
| LEMD | short | 61 | 1,982 | 582 | -398 | 1,039 |
| LEMD | short | 62 | 1,116 | 608 | -374 | 1,054 |
| LEMD | short | 23 | 347 | 691 | -290 | 960 |
| LEMD | short | 40 | 5,147 | 703 | -278 | 1,011 |
| LEMD | long | 71 | 1,387 | 1,314 | +333 | 1,714 |
| LEMD | long | 74 | 1,082 | 1,326 | +346 | 1,729 |
| LEMD | long | 73 | 1,367 | 1,331 | +350 | 1,732 |
| LEMD | long | 72 | 1,406 | 1,332 | +351 | 1,746 |
| LEMD | long | 70 | 1,333 | 1,354 | +373 | 1,754 |
| LFPG | short | R1 | 374 | 543 | -410 | 964 |
| LFPG | short | R0 | 203 | 602 | -351 | 1,122 |
| LFPG | short | G2 | 283 | 609 | -344 | 1,031 |
| LFPG | short | X0 | 7,673 | 710 | -243 | 1,128 |
| LFPG | short | G3 | 454 | 774 | -180 | 1,251 |
| LFPG | long | H1 | 517 | 1,206 | +253 | 1,739 |
| LFPG | long | L3 | 1,442 | 1,210 | +257 | 1,738 |
| LFPG | long | N0 | 2,152 | 1,258 | +305 | 1,803 |
| LFPG | long | CF | 1,050 | 1,260 | +306 | 1,862 |
| LFPG | long | I5 | 1,293 | 1,270 | +317 | 1,981 |
| LIRF | short | 10 | 509 | 711 | -300 | 1,080 |
| LIRF | short | 23 | 8,528 | 775 | -236 | 1,254 |
| LIRF | short | 22 | 6,766 | 782 | -230 | 1,312 |
| LIRF | short | 21 | 279 | 900 | -111 | 1,490 |
| LIRF | short | 32 | 1,540 | 903 | -108 | 1,384 |
| LIRF | long | 82 | 3,061 | 1,211 | +200 | 1,807 |
| LIRF | long | 70 | 9,948 | 1,253 | +242 | 1,909 |
| LIRF | long | 81 | 357 | 1,253 | +242 | 1,916 |
| LIRF | long | 83 | 612 | 1,330 | +320 | 1,991 |
| LIRF | long | 90 | 1,754 | 1,380 | +369 | 2,150 |
| LSZH | short | RE | 704 | 266 | -444 | 513 |
| LSZH | short | F7 | 3,530 | 367 | -343 | 678 |
| LSZH | short | GA | 2,153 | 378 | -332 | 739 |
| LSZH | short | H1 | 5,978 | 388 | -322 | 685 |
| LSZH | short | 10 | 4,524 | 408 | -302 | 770 |
| LSZH | long | T6 | 307 | 903 | +193 | 1,320 |
| LSZH | long | T5 | 1,103 | 923 | +213 | 1,276 |
| LSZH | long | G0 | 2,717 | 936 | +226 | 1,267 |
| LSZH | long | G1 | 1,355 | 942 | +232 | 1,267 |
| LSZH | long | C5 | 1,128 | 1,004 | +294 | 1,367 |
| LTFM | short | 60 | 268 | 534 | -428 | 982 |
| LTFM | short | H4 | 1,889 | 549 | -413 | 1,030 |
| LTFM | short | K9 | 594 | 718 | -244 | 1,270 |
| LTFM | short | K1 | 6,991 | 724 | -238 | 1,255 |
| LTFM | short | K7 | 278 | 728 | -234 | 1,209 |
| LTFM | long | B2 | 2,121 | 1,136 | +174 | 1,615 |
| LTFM | long | A1 | 5,926 | 1,138 | +176 | 1,626 |
| LTFM | long | D7 | 1,659 | 1,143 | +181 | 1,684 |
| LTFM | long | D9 | 1,631 | 1,147 | +185 | 1,735 |
| LTFM | long | B4 | 1,998 | 1,196 | +234 | 1,791 |


### D. Winter and de-icing

| airport | de-icing group | DJF median | JJA median | DJF−JJA median | DJF−JJA mean | Nov–Mar freezing n | freezing median | not-freezing median | freezing − not | Nov–Mar T≤0 °C n | T≤0 − T≥5 median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | yes | 829 | 838 | -9 | +12 | 575 | 1,040 | 827 | +213 | 10,239 | +25 |
| EDDM | yes | 773 | 775 | -2 | +84 | 2,297 | 1,431 | 730 | +701 | 18,333 | +130 |
| EGLL | no | 1,272 | 1,327 | -55 | -46 | 193 | 1,329 | 1,311 | +18 | 2,658 | -107 |
| EHAM | yes | 744 | 721 | +23 | +46 | 1,381 | 928 | 750 | +178 | 6,736 | +127 |
| LEBL | no | 853 | 921 | -68 | -68 | 0 | — | 860 | — | 99 | +22 |
| LEMD | no | 957 | 998 | -41 | -39 | 6 | 1,110 | 968 | +142 | 3,233 | +95 |
| LFPG | no | 907 | 956 | -49 | +23 | 849 | 2,159 | 908 | +1251 | 7,704 | +305 |
| LIRF | no | 958 | 1,029 | -71 | -145 | 0 | — | 958 | — | 23 | -113 |
| LSZH | yes | 705 | 718 | -13 | +18 | 1,017 | 983 | 703 | +280 | 9,930 | +60 |
| LTFM | no | 959 | 960 | -1 | +37 | 2,589 | 1,511 | 961 | +550 | 1,285 | +1030 |


### D. Public holidays (list in §Method)

| airport | holiday days | n | holiday median | other days median | non-holiday Sunday median | holiday − other | dep/h holiday | dep/h other |
|---|---|---|---|---|---|---|---|---|
| EDDF | 10 | 5,908 | 822 | 835 | 841 | -13 | 39.7 | 41.1 |
| EDDM | 13 | 5,610 | 769 | 772 | 772 | -3 | 30.7 | 31.1 |
| EGLL | 8 | 5,079 | 1,315 | 1,318 | 1,370 | -3 | 39.1 | 39.5 |
| EHAM | 11 | 7,381 | 792 | 743 | 754 | +49 | 43.9 | 44.2 |
| LEBL | 14 | 6,643 | 888 | 898 | 912 | -10 | 28.3 | 28.4 |
| LEMD | 14 | 7,939 | 977 | 982 | 980 | -5 | 34.1 | 34.3 |
| LFPG | 11 | 7,019 | 904 | 953 | 950 | -49 | 39.3 | 38.1 |
| LIRF | 13 | 5,257 | 1,010 | 1,011 | 1,020 | -1 | 26.0 | 25.8 |
| LSZH | 10 | 3,540 | 712 | 710 | 707 | +2 | 24.4 | 24.5 |
| LTFM | 14 | 10,239 | 959 | 962 | 960 | -3 | 40.5 | 39.9 |


### D. Delay × clock agreement — is the delay effect taxi physics or recording?

delta = BLOCK − AOBT_3 = proxy − y. 'BLOCK before NM push' = delta < −300 s (the airport's off-block is >5 min earlier than NM's: y then includes time spent at the stand). 'clocks agree' = |delta| ≤ 120 s. `NM taxi` = median proxy (MVT − AOBT_3). src: marginals (ordinary matched rows, 12 months 2025).


#### ALL

| delay band | n | median y | median NM taxi | BLOCK before NM push | clocks agree | n agree | median y (agree) | within-apt Δmedian (all) | within-apt Δmedian (agree) |
|---|---|---|---|---|---|---|---|---|---|
| early 0-5m | 292,634 | 886 | 1,015 | 1.9% | 42.5% | 124,364 | 894 | -63 | -50 |
| early >5m | 243,326 | 845 | 1,198 | 0.8% | 22.8% | 55,363 | 948 | -121 | -55 |
| late 0-5m | 357,414 | 897 | 923 | 5.0% | 47.6% | 170,210 | 854 | -18 | -32 |
| late 1-2h | 73,911 | 1,010 | 940 | 21.3% | 35.8% | 26,465 | 894 | 77 | 0 |
| late 15-30m | 343,504 | 963 | 898 | 18.9% | 38.9% | 133,570 | 856 | 68 | -4 |
| late 30-60m | 213,559 | 1,010 | 903 | 22.2% | 36.5% | 78,054 | 889 | 92 | 0 |
| late 5-15m | 485,207 | 906 | 899 | 9.4% | 44.4% | 215,261 | 846 | 13 | -20 |
| late >2h | 25,559 | 991 | 1,017 | 18.3% | 34.1% | 8,713 | 945 | 65 | 15 |


#### EDDF

| delay band | n | median y | median NM taxi | BLOCK before NM push | clocks agree | n agree | median y (agree) | within-apt Δmedian (all) | within-apt Δmedian (agree) |
|---|---|---|---|---|---|---|---|---|---|
| early 0-5m | 30,077 | 752 | 881 | 0.8% | 37.1% | 11,155 | 835 | -83 | 0 |
| early >5m | 14,847 | 760 | 914 | 1.3% | 28.1% | 4,172 | 840 | -75 | 5 |
| late 0-5m | 41,461 | 792 | 874 | 1.6% | 45.2% | 18,729 | 855 | -43 | 20 |
| late 1-2h | 6,951 | 953 | 956 | 14.3% | 39.4% | 2,737 | 936 | 118 | 101 |
| late 15-30m | 43,843 | 898 | 919 | 9.9% | 42.4% | 18,577 | 905 | 63 | 70 |
| late 30-60m | 24,970 | 940 | 944 | 12.8% | 40.7% | 10,162 | 928 | 105 | 93 |
| late 5-15m | 62,693 | 834 | 885 | 4.5% | 45.1% | 28,270 | 867 | -1 | 32 |
| late >2h | 2,513 | 987 | 1,038 | 13.9% | 38.5% | 968 | 987 | 152 | 152 |


#### EDDM

| delay band | n | median y | median NM taxi | BLOCK before NM push | clocks agree | n agree | median y (agree) | within-apt Δmedian (all) | within-apt Δmedian (agree) |
|---|---|---|---|---|---|---|---|---|---|
| early 0-5m | 26,233 | 660 | 722 | 0.7% | 54.8% | 14,385 | 713 | -112 | -59 |
| early >5m | 11,146 | 607 | 724 | 0.7% | 41.5% | 4,625 | 714 | -165 | -58 |
| late 0-5m | 32,544 | 724 | 722 | 2.9% | 59.1% | 19,231 | 724 | -48 | -48 |
| late 1-2h | 4,566 | 889 | 718 | 31.2% | 37.8% | 1,724 | 769 | 117 | -3 |
| late 15-30m | 29,919 | 848 | 718 | 29.1% | 38.9% | 11,646 | 770 | 76 | -2 |
| late 30-60m | 16,544 | 892 | 718 | 32.1% | 36.9% | 6,110 | 769 | 120 | -3 |
| late 5-15m | 42,347 | 783 | 722 | 11.5% | 50.2% | 21,255 | 730 | 11 | -42 |
| late >2h | 1,017 | 851 | 720 | 30.1% | 37.9% | 385 | 731 | 79 | -41 |


#### EGLL

| delay band | n | median y | median NM taxi | BLOCK before NM push | clocks agree | n agree | median y (agree) | within-apt Δmedian (all) | within-apt Δmedian (agree) |
|---|---|---|---|---|---|---|---|---|---|
| early 0-5m | 46,389 | 1,251 | 1,318 | 3.9% | 42.6% | 19,755 | 1,209 | -67 | -109 |
| early >5m | 40,601 | 1,147 | 1,325 | 1.6% | 31.9% | 12,969 | 1,148 | -171 | -170 |
| late 0-5m | 48,378 | 1,326 | 1,320 | 10.5% | 38.6% | 18,692 | 1,319 | 8 | 1 |
| late 1-2h | 6,935 | 1,445 | 1,325 | 24.2% | 29.8% | 2,069 | 1,330 | 127 | 12 |
| late 15-30m | 27,169 | 1,431 | 1,377 | 17.2% | 33.8% | 9,180 | 1,330 | 113 | 12 |
| late 30-60m | 19,205 | 1,443 | 1,322 | 23.6% | 31.0% | 5,947 | 1,324 | 125 | 6 |
| late 5-15m | 45,523 | 1,381 | 1,379 | 9.7% | 38.4% | 17,458 | 1,368 | 63 | 50 |
| late >2h | 2,182 | 1,391 | 1,378 | 18.7% | 32.2% | 702 | 1,328 | 73 | 10 |


#### EHAM

| delay band | n | median y | median NM taxi | BLOCK before NM push | clocks agree | n agree | median y (agree) | within-apt Δmedian (all) | within-apt Δmedian (agree) |
|---|---|---|---|---|---|---|---|---|---|
| early 0-5m | 33,597 | 720 | 892 | 0.2% | 39.2% | 13,173 | 745 | -24 | 1 |
| early >5m | 17,960 | 722 | 968 | 0.1% | 28.8% | 5,178 | 790 | -22 | 46 |
| late 0-5m | 47,997 | 721 | 834 | 0.3% | 49.5% | 23,770 | 728 | -23 | -16 |
| late 1-2h | 9,474 | 769 | 885 | 1.4% | 44.1% | 4,176 | 777 | 25 | 33 |
| late 15-30m | 41,314 | 760 | 862 | 0.9% | 45.9% | 18,947 | 764 | 16 | 20 |
| late 30-60m | 24,016 | 769 | 879 | 1.1% | 44.4% | 10,665 | 774 | 25 | 30 |
| late 5-15m | 65,869 | 748 | 842 | 0.6% | 48.9% | 32,214 | 748 | 4 | 4 |
| late >2h | 3,530 | 807 | 918 | 3.2% | 40.7% | 1,436 | 831 | 63 | 87 |


#### LEBL

| delay band | n | median y | median NM taxi | BLOCK before NM push | clocks agree | n agree | median y (agree) | within-apt Δmedian (all) | within-apt Δmedian (agree) |
|---|---|---|---|---|---|---|---|---|---|
| early 0-5m | 23,823 | 828 | 817 | 1.7% | 58.6% | 13,967 | 756 | -69 | -141 |
| early >5m | 15,238 | 835 | 1,211 | 0.4% | 40.5% | 6,167 | 751 | -62 | -146 |
| late 0-5m | 31,140 | 831 | 740 | 4.8% | 57.1% | 17,798 | 749 | -66 | -148 |
| late 1-2h | 7,909 | 1,035 | 820 | 22.7% | 39.5% | 3,128 | 843 | 138 | -54 |
| late 15-30m | 29,630 | 980 | 752 | 20.6% | 43.1% | 12,782 | 796 | 83 | -101 |
| late 30-60m | 20,011 | 1,033 | 784 | 21.5% | 42.5% | 8,497 | 847 | 136 | -50 |
| late 5-15m | 44,405 | 885 | 730 | 11.8% | 47.9% | 21,259 | 761 | -12 | -136 |
| late >2h | 2,593 | 989 | 911 | 20.7% | 38.5% | 998 | 852 | 92 | -46 |


#### LEMD

| delay band | n | median y | median NM taxi | BLOCK before NM push | clocks agree | n agree | median y (agree) | within-apt Δmedian (all) | within-apt Δmedian (agree) |
|---|---|---|---|---|---|---|---|---|---|
| early 0-5m | 27,593 | 925 | 1,001 | 1.4% | 52.1% | 14,387 | 983 | -56 | 2 |
| early >5m | 12,455 | 936 | 1,089 | 1.1% | 39.5% | 4,924 | 1,030 | -45 | 49 |
| late 0-5m | 38,557 | 941 | 980 | 3.6% | 56.6% | 21,806 | 953 | -40 | -28 |
| late 1-2h | 8,490 | 1,102 | 1,028 | 19.5% | 41.8% | 3,546 | 1,028 | 121 | 47 |
| late 15-30m | 38,372 | 1,030 | 977 | 16.7% | 45.2% | 17,355 | 967 | 49 | -14 |
| late 30-60m | 23,286 | 1,087 | 998 | 20.4% | 42.0% | 9,775 | 1,006 | 106 | 25 |
| late 5-15m | 57,991 | 972 | 972 | 8.2% | 52.1% | 30,203 | 944 | -9 | -37 |
| late >2h | 2,734 | 1,056 | 1,013 | 17.0% | 40.9% | 1,119 | 1,012 | 76 | 31 |


#### LFPG

| delay band | n | median y | median NM taxi | BLOCK before NM push | clocks agree | n agree | median y (agree) | within-apt Δmedian (all) | within-apt Δmedian (agree) |
|---|---|---|---|---|---|---|---|---|---|
| early 0-5m | 28,914 | 847 | 959 | 1.3% | 45.2% | 13,083 | 840 | -106 | -113 |
| early >5m | 21,238 | 836 | 1,201 | 0.9% | 19.7% | 4,190 | 788 | -117 | -165 |
| late 0-5m | 34,553 | 905 | 900 | 5.0% | 50.1% | 17,311 | 851 | -48 | -102 |
| late 1-2h | 11,204 | 1,012 | 840 | 29.4% | 36.3% | 4,067 | 839 | 60 | -114 |
| late 15-30m | 48,157 | 1,014 | 894 | 22.8% | 39.1% | 18,832 | 889 | 61 | -64 |
| late 30-60m | 33,085 | 1,023 | 844 | 29.6% | 35.7% | 11,801 | 849 | 70 | -104 |
| late 5-15m | 52,842 | 955 | 897 | 10.8% | 45.7% | 24,152 | 851 | 2 | -102 |
| late >2h | 3,838 | 1,012 | 844 | 28.0% | 37.2% | 1,428 | 846 | 59 | -107 |


#### LIRF

| delay band | n | median y | median NM taxi | BLOCK before NM push | clocks agree | n agree | median y (agree) | within-apt Δmedian (all) | within-apt Δmedian (agree) |
|---|---|---|---|---|---|---|---|---|---|
| early 0-5m | 20,397 | 896 | 902 | 6.8% | 49.6% | 10,121 | 890 | -115 | -121 |
| early >5m | 15,600 | 831 | 956 | 3.2% | 39.1% | 6,101 | 890 | -180 | -121 |
| late 0-5m | 25,540 | 990 | 897 | 14.6% | 40.4% | 10,324 | 900 | -21 | -111 |
| late 1-2h | 6,729 | 1,083 | 899 | 36.5% | 31.3% | 2,107 | 894 | 72 | -117 |
| late 15-30m | 26,667 | 1,092 | 898 | 38.5% | 29.5% | 7,860 | 898 | 81 | -113 |
| late 30-60m | 18,976 | 1,131 | 898 | 39.4% | 28.7% | 5,447 | 897 | 120 | -114 |
| late 5-15m | 30,463 | 1,025 | 899 | 25.2% | 35.5% | 10,831 | 897 | 14 | -114 |
| late >2h | 1,969 | 1,090 | 954 | 34.4% | 32.3% | 636 | 910 | 79 | -102 |


#### LSZH

| delay band | n | median y | median NM taxi | BLOCK before NM push | clocks agree | n agree | median y (agree) | within-apt Δmedian (all) | within-apt Δmedian (agree) |
|---|---|---|---|---|---|---|---|---|---|
| early 0-5m | 10,041 | 592 | 663 | 1.5% | 51.4% | 5,160 | 640 | -118 | -70 |
| early >5m | 4,916 | 521 | 604 | 1.4% | 39.3% | 1,930 | 607 | -189 | -103 |
| late 0-5m | 21,500 | 655 | 669 | 3.1% | 56.2% | 12,079 | 674 | -55 | -36 |
| late 1-2h | 4,512 | 791 | 697 | 24.6% | 36.9% | 1,665 | 693 | 81 | -17 |
| late 15-30m | 31,445 | 779 | 683 | 20.6% | 40.5% | 12,736 | 687 | 69 | -23 |
| late 30-60m | 17,001 | 789 | 686 | 24.3% | 38.6% | 6,567 | 685 | 79 | -25 |
| late 5-15m | 41,423 | 718 | 678 | 10.1% | 45.8% | 18,981 | 679 | 8 | -31 |
| late >2h | 846 | 766 | 690 | 24.8% | 36.6% | 310 | 684 | 56 | -26 |


#### LTFM

| delay band | n | median y | median NM taxi | BLOCK before NM push | clocks agree | n agree | median y (agree) | within-apt Δmedian (all) | within-apt Δmedian (agree) |
|---|---|---|---|---|---|---|---|---|---|
| early 0-5m | 45,570 | 963 | 1,200 | 1.3% | 20.1% | 9,178 | 1,147 | 1 | 185 |
| early >5m | 89,325 | 836 | 1,203 | 0.2% | 5.7% | 5,107 | 1,142 | -126 | 180 |
| late 0-5m | 35,744 | 1,072 | 1,200 | 5.2% | 29.3% | 10,470 | 1,189 | 110 | 227 |
| late 1-2h | 7,141 | 1,021 | 1,200 | 17.2% | 17.4% | 1,246 | 1,170 | 59 | 208 |
| late 15-30m | 26,988 | 1,145 | 1,199 | 24.3% | 20.9% | 5,655 | 1,189 | 183 | 227 |
| late 30-60m | 16,465 | 1,087 | 1,200 | 22.5% | 18.7% | 3,083 | 1,189 | 125 | 227 |
| late 5-15m | 41,651 | 1,133 | 1,199 | 13.5% | 25.5% | 10,638 | 1,194 | 171 | 232 |
| late >2h | 4,337 | 957 | 1,200 | 12.5% | 16.9% | 731 | 1,141 | -5 | 179 |


### D. Precipitation and obscuration from METAR present-weather codes

arm W's `w_precip_mm` is identically 0.0 on all rows (European METARs carry no `p01i` group), so precipitation was re-derived from the SAME observation's `wxcodes` (valid = AOBT_3 − w_age_s; found for 99.997% of 2,035,114 ordinary rows). rain = RA/DZ; snow = SN/SG/PL/GS/GR (not blowing/drifting); freezing precip = FZRA/FZDZ; fog = FG/FZFG (not BC/MI/PR); heavy = a '+' intensity; VC (vicinity) ignored. Raw marginals: Δ = median(flag) − median(no flag) within the airport, n = rows with the flag. The fitted models do NOT see these flags.

| airport | rain | snow | freezing rain/drizzle | fog | heavy precip |
|---|---|---|---|---|---|
| EDDF | +33 (21,103) | +121 (491) | +517 (53) | +96 (1,136) | +60 (1,610) |
| EDDM | +7 (19,618) | +670 (1,264) | +1261 (63) | +186 (2,910) | +181 (620) |
| EGLL | +56 (19,559) | -55 (134) | — (0) | +370 (477) | +112 (320) |
| EHAM | -32 (32,703) | +75 (1,117) | — (0) | +90 (2,392) | -138 (72) |
| LEBL | +50 (8,193) | +176 (32) | — (0) | +106 (336) | +167 (271) |
| LEMD | +32 (11,572) | — | — (0) | +55 (348) | +140 (220) |
| LFPG | +7 (16,777) | +62 (195) | — | +478 (2,845) | +74 (60) |
| LIRF | +59 (7,024) | — (0) | — (0) | +245 (111) | +378 (83) |
| LSZH | -3 (15,376) | +266 (624) | +143 (41) | +38 (2,924) | +102 (65) |
| LTFM | -1 (25,126) | +246 (3,233) | — (0) | +115 (1,531) | — |
| ALL (within-airport Δ median) | +5 (177,051) | +248 (7,092) | +604 (173) | +156 (15,010) | +92 (3,343) |


### D. Raw marginal slopes (binned means/medians, OLS on bin centres)

| airport | factor | Δmean s per unit | Δmedian s per unit | range | n |
|---|---|---|---|---|---|
| EDDF | q_apt_at_push | 10.51 | 11.82 | 0–21 | 227,277 |
| EDDF | q_rwy_at_push | 25.82 | 28.83 | 0–14 | 227,268 |
| EDDF | q_arr_taxiing_at_push | -0.2 | -0.77 | 0–15 | 227,148 |
| EDDF | dep_hr | 2.47 | 3.18 | 1–65 | 227,231 |
| EDDM | q_apt_at_push | 3.43 | 6.5 | 0–16 | 164,255 |
| EDDM | q_rwy_at_push | 15.03 | 17.08 | 0–11 | 164,270 |
| EDDM | q_arr_taxiing_at_push | -0.3 | 4.54 | 0–9 | 164,227 |
| EDDM | dep_hr | 0.21 | 1.89 | 1–59 | 164,104 |
| EGLL | q_apt_at_push | 35.09 | 35.09 | 0–25 | 235,965 |
| EGLL | q_rwy_at_push | 30.3 | 31.59 | 0–20 | 221,683 |
| EGLL | q_arr_taxiing_at_push | 19.97 | 17.78 | 0–14 | 236,242 |
| EGLL | dep_hr | 9.31 | 11.56 | 4–53 | 236,178 |
| EHAM | q_apt_at_push | 13.28 | 13.86 | 0–25 | 242,980 |
| EHAM | q_rwy_at_push | 34.0 | 36.3 | 0–19 | 243,667 |
| EHAM | q_arr_taxiing_at_push | -17.24 | -18.16 | 0–15 | 243,656 |
| EHAM | dep_hr | 0.51 | 1.26 | 1–70 | 241,111 |
| LEBL | q_apt_at_push | 20.64 | 28.04 | 0–13 | 174,675 |
| LEBL | q_rwy_at_push | 14.66 | 22.01 | 0–13 | 174,714 |
| LEBL | q_arr_taxiing_at_push | 30.11 | 33.7 | 0–5 | 174,617 |
| LEBL | dep_hr | 6.44 | 7.42 | 1–43 | 174,601 |
| LEMD | q_apt_at_push | 1.62 | 2.02 | 0–20 | 209,404 |
| LEMD | q_rwy_at_push | 14.13 | 15.58 | 0–12 | 209,361 |
| LEMD | q_arr_taxiing_at_push | 11.84 | 10.63 | 0–13 | 209,397 |
| LEMD | dep_hr | 0.11 | 0.14 | 1–64 | 209,339 |
| LFPG | q_apt_at_push | 21.32 | 19.84 | 0–25 | 233,444 |
| LFPG | q_rwy_at_push | 30.23 | 29.07 | 0–16 | 233,708 |
| LFPG | q_arr_taxiing_at_push | 9.37 | 8.29 | 0–15 | 233,667 |
| LFPG | dep_hr | 4.09 | 5.21 | 1–65 | 233,624 |
| LIRF | q_apt_at_push | 32.47 | 30.87 | 0–16 | 146,258 |
| LIRF | q_rwy_at_push | 8.69 | 9.2 | 0–14 | 146,164 |
| LIRF | q_arr_taxiing_at_push | 20.48 | 17.12 | 0–11 | 146,228 |
| LIRF | dep_hr | 11.28 | 10.72 | 1–44 | 146,239 |
| LSZH | q_apt_at_push | 31.06 | 28.92 | 0–10 | 131,646 |
| LSZH | q_rwy_at_push | 2.33 | 7.34 | 0–9 | 131,613 |
| LSZH | q_arr_taxiing_at_push | -4.21 | -2.8 | 0–7 | 131,605 |
| LSZH | dep_hr | 5.39 | 6.21 | 2–41 | 131,534 |
| LTFM | q_apt_at_push | 14.62 | 13.97 | 0–25 | 254,389 |
| LTFM | q_rwy_at_push | 13.12 | 13.44 | 0–20 | 267,090 |
| LTFM | q_arr_taxiing_at_push | -1.01 | 0.22 | 0–15 | 253,500 |
| LTFM | dep_hr | 3.91 | 4.77 | 3–70 | 263,261 |


## E. Variance decomposition (held-out Jan + Jul 2025)

n held out = 334,562; train 1,360,226; early-stop 340,326. Held-out sd(y) = 431.71 s; predicting the training mean gives RMSE 432.13 s. Each step is a separately early-stopped LightGBM with all blocks up to that step. R² = 1 − SSE/SST on the held-out rows (SST about the held-out mean). The order is fixed in advance; incremental R² depends on it.

| step | adds | trees | held-out RMSE s | R² | incremental R² |
|---|---|---|---|---|---|
| a_airport | ap | 160 | 394.107 | 0.1666 | +0.1666 |
| b_layout | rwy, stand_area | 152 | 350.267 | 0.3417 | +0.1751 |
| c_time | hour_local, dow_local, month, holiday | 634 | 337.486 | 0.3889 | +0.0472 |
| d_congestion | dep_hr, arr_hr, q_apt_at_push, q_rwy_at_push, q_rwy_push_pre20, q_arr_taxiing_at_push, q_rwy_tko_sym10, rec_rwy_y20, rec_apt_y30, q_arr_taxiin_sym30_push, prev_arr_taxiin | 2,142 | 310.253 | 0.4835 | +0.0946 |
| e_aircraft | wake, actype, airline, segment | 1,868 | 301.499 | 0.5123 | +0.0287 |
| e2_delay_turn | delay, gapa | 2,109 | 279.402 | 0.5811 | +0.0689 |
| f_weather | w_temp_c, w_dewspread_c, w_wind_kt, w_gust_kt, w_vis_km, w_precip_mm, w_freezing, w_lowvis, w_thunder | 2,573 | 275.327 | 0.5933 | +0.0121 |
| g_proxy | proxy | 2,195 | 238.586 | 0.6946 | +0.1013 |


### E.1 Within-airport R² by step (each airport's own SST)

| airport | a_airport | b_layout | c_time | d_congestion | e_aircraft | e2_delay_turn | f_weather | g_proxy |
|---|---|---|---|---|---|---|---|---|
| EDDF | -0.001 | 0.212 | 0.262 | 0.340 | 0.427 | 0.472 | 0.488 | 0.693 |
| EDDM | -0.003 | 0.128 | 0.159 | 0.368 | 0.415 | 0.533 | 0.581 | 0.698 |
| EGLL | -0.003 | 0.122 | 0.185 | 0.275 | 0.288 | 0.409 | 0.416 | 0.627 |
| EHAM | -0.002 | 0.335 | 0.381 | 0.467 | 0.489 | 0.507 | 0.536 | 0.757 |
| LEBL | -0.000 | 0.399 | 0.454 | 0.510 | 0.532 | 0.587 | 0.592 | 0.602 |
| LEMD | -0.000 | 0.319 | 0.381 | 0.441 | 0.515 | 0.572 | 0.592 | 0.629 |
| LFPG | -0.004 | 0.169 | 0.209 | 0.406 | 0.431 | 0.495 | 0.503 | 0.651 |
| LIRF | -0.011 | 0.136 | 0.241 | 0.324 | 0.389 | 0.501 | 0.498 | 0.514 |
| LSZH | -0.003 | 0.286 | 0.320 | 0.428 | 0.441 | 0.483 | 0.531 | 0.661 |
| LTFM | -0.001 | 0.201 | 0.259 | 0.388 | 0.399 | 0.527 | 0.536 | 0.559 |


## F. Conditional effects — TreeSHAP on the physical-factor model (f)

Model f: held-out RMSE (full holdout) 275.327 s; on the 100k SHAP sample 276.09 s. Expected value 975.2 s; SHAP sums reproduce predictions to 0.0 s. **Model-based attributions, not causal effects.** src: TreeSHAP, physical model f, 100k held-out rows (Jan+Jul 2025).


### F.1 Mean |SHAP| per factor block (seconds)

| block | mean \|SHAP\| s | sd s | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_airport | 0.64 | 1.7 | 0.47 | 0.47 | 0.57 | 0.46 | 1.41 | 0.45 | 0.75 | 1.0 | 0.51 | 0.54 |
| b_layout | 136.47 | 170.14 | 122.24 | 92.34 | 150.51 | 191.08 | 159.99 | 144.13 | 96.62 | 134.67 | 153.5 | 119.74 |
| c_time | 10.13 | 16.45 | 8.01 | 9.03 | 9.49 | 6.35 | 9.63 | 8.14 | 11.15 | 18.87 | 6.29 | 14.55 |
| d_congestion | 98.74 | 133.15 | 70.66 | 111.79 | 217.98 | 74.42 | 69.96 | 58.7 | 90.24 | 85.57 | 115.9 | 86.83 |
| e_aircraft | 62.92 | 79.92 | 79.83 | 64.17 | 75.73 | 55.0 | 49.53 | 82.67 | 60.52 | 57.57 | 44.25 | 51.37 |
| e2_delay_turn | 72.32 | 98.97 | 59.06 | 74.9 | 81.18 | 44.51 | 60.38 | 63.8 | 79.75 | 91.45 | 64.87 | 99.29 |
| f_weather | 24.69 | 59.49 | 25.37 | 40.18 | 19.74 | 34.22 | 11.72 | 15.52 | 42.59 | 14.21 | 34.19 | 11.98 |


### F.1b Grouped factors: how many seconds each physical factor moves (SHAP summed over its columns)

Correlated columns share SHAP credit (aircraft type absorbs most of the wake class; the queue counts, take-offs around me and dep_hr split congestion), so single-column SHAP understates grouped factors and can flip a sign (`q_rwy_push_pre20` and `dep_hr` have negative conditional slopes). The grouped sum is the readable quantity. mean |SHAP| and the p10→p90 spread of the grouped SHAP, seconds. src: TreeSHAP, physical model f, 100k held-out rows (Jan+Jul 2025).

| factor | ALL | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| layout (runway + stand area) | 136 / 450 | 122 / 352 | 92 / 224 | 150 / 318 | 191 / 404 | 160 / 471 | 144 / 390 | 97 / 311 | 135 / 367 | 154 / 416 | 120 / 392 |
| congestion block | 99 / 309 | 71 / 176 | 112 / 126 | 218 / 354 | 74 / 178 | 70 / 165 | 59 / 156 | 90 / 251 | 86 / 261 | 116 / 123 | 87 / 261 |
| aircraft (wake + type) | 31 / 92 | 38 / 112 | 36 / 104 | 34 / 102 | 29 / 91 | 27 / 67 | 32 / 92 | 37 / 114 | 23 / 65 | 24 / 74 | 21 / 67 |
| airline + segment | 48 / 135 | 52 / 142 | 38 / 91 | 60 / 148 | 42 / 107 | 32 / 91 | 64 / 169 | 38 / 119 | 52 / 131 | 37 / 84 | 49 / 55 |
| delay + turnaround | 72 / 222 | 59 / 176 | 75 / 223 | 81 / 232 | 44 / 133 | 60 / 192 | 64 / 199 | 80 / 228 | 92 / 279 | 65 / 190 | 99 / 289 |
| time (hour, weekday, month, holiday) | 10 / 29 | 8 / 23 | 9 / 28 | 10 / 27 | 6 / 19 | 10 / 26 | 8 / 24 | 11 / 34 | 19 / 59 | 6 / 19 | 15 / 40 |
| weather | 25 / 55 | 25 / 82 | 40 / 134 | 20 / 42 | 34 / 114 | 12 / 16 | 16 / 25 | 43 / 140 | 14 / 28 | 34 / 103 | 12 / 24 |


### F.1c Aircraft size (wake + type SHAP) by wake class, s

| where | L | M | H | J |
|---|---|---|---|---|
| ALL | +13 | -14 | +54 | +149 |
| EDDF | — | -20 | +77 | — |
| EDDM | -14 | -24 | +74 | +174 |
| EGLL | — | -14 | +50 | +144 |
| EHAM | +2 | -12 | +69 | +179 |
| LEBL | +70 | -18 | +54 | — |
| LEMD | +55 | -16 | +50 | — |
| LFPG | — | -16 | +68 | +144 |
| LIRF | — | -12 | +55 | — |
| LSZH | -9 | -6 | +58 | — |
| LTFM | — | -7 | +20 | — |
| n (ALL) | 486 | 78,091 | 20,743 | 678 |


### F.1d Whole congestion block SHAP by runway queue at push (capped at 20), s

| where | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | -101 | -99 | -90 | -76 | -58 | -44 | -28 | -7 | +14 | +37 | +61 | +87 | +111 | +149 | +182 | +216 | +253 | +294 | +328 | +338 | +396 |
| EDDF | -109 | -107 | -98 | -86 | -67 | -56 | -38 | -18 | +1 | +24 | +40 | +59 | +75 | +91 | — | — | — | — | — | — | — |
| EDDM | -116 | -119 | -111 | -96 | -86 | -73 | -67 | -37 | -34 | +6 | — | — | — | — | — | — | — | — | — | — | — |
| EGLL | +120 | +126 | +88 | +86 | +45 | +57 | +40 | +41 | +42 | +59 | +78 | +107 | +134 | +168 | +199 | +233 | +266 | +300 | +330 | +339 | +396 |
| EHAM | -149 | -113 | -104 | -96 | -78 | -69 | -54 | -39 | -21 | -3 | +14 | +36 | +53 | +78 | +103 | +122 | +115 | — | — | — | — |
| LEBL | -109 | -111 | -100 | -93 | -75 | -65 | -51 | -30 | -13 | +2 | +15 | +40 | — | — | — | — | — | — | — | — | — |
| LEMD | -80 | -78 | -68 | -54 | -41 | -30 | -19 | -6 | +11 | +34 | +60 | +86 | — | — | — | — | — | — | — | — | — |
| LFPG | -92 | -91 | -78 | -63 | -43 | -26 | -4 | +17 | +48 | +91 | +131 | +156 | +193 | +223 | +249 | — | — | — | — | — | — |
| LIRF | -48 | -58 | -60 | -56 | -43 | -22 | +3 | +26 | +59 | +69 | +82 | +92 | +104 | +119 | — | — | — | — | — | — | — |
| LSZH | -137 | -145 | -136 | -123 | -100 | -88 | -78 | -63 | -19 | — | — | — | — | — | — | — | — | — | — | — | — |
| LTFM | -51 | -50 | -40 | -20 | -8 | +6 | +13 | +29 | +39 | +52 | +74 | +95 | +107 | +142 | +161 | +172 | +173 | +207 | — | — | — |
| n (ALL) | 3,812 | 6,027 | 8,183 | 9,693 | 10,727 | 10,894 | 9,952 | 8,404 | 6,872 | 5,380 | 4,070 | 3,199 | 2,416 | 2,066 | 1,610 | 1,535 | 1,345 | 1,109 | 894 | 683 | 1,129 |

| where | s per extra aircraft queued on my runway (0–15) | s per extra aircraft queued at the airport (0–25) | n |
|---|---|---|---|
| ALL | 19.97 | 12.38 | 94,840 |
| EDDF | 16.5 | 8.22 | 10,857 |
| EDDM | 11.5 | 5.25 | 7,979 |
| EGLL | 16.19 | 23.25 | 6,924 |
| EHAM | 16.12 | 7.84 | 11,848 |
| LEBL | 14.4 | 15.37 | 8,441 |
| LEMD | 13.34 | 6.56 | 10,455 |
| LFPG | 23.49 | 15.85 | 11,436 |
| LIRF | 17.16 | 20.46 | 7,094 |
| LSZH | 13.2 | 14.95 | 6,390 |
| LTFM | 15.05 | 10.17 | 13,416 |


### F.1e Time block SHAP by local push hour, s

| where | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | +34 | +40 | +110 | +19 | +17 | +3 | -5 | +2 | +4 | +9 | +11 | +5 | -4 | -4 | -8 | -10 | -8 | -8 | -7 | -1 | +1 | +1 | +22 | +38 |
| EDDF | — | — | — | — | +25 | +3 | -6 | +2 | -0 | +3 | +8 | +7 | -5 | -4 | -7 | -8 | -7 | -8 | -6 | +1 | +3 | +3 | +34 | — |
| EDDM | — | — | — | — | — | -4 | -4 | -1 | -0 | +7 | +6 | +1 | -6 | -5 | -8 | -12 | -9 | -8 | -5 | +5 | +10 | +11 | +30 | +45 |
| EGLL | — | — | — | — | — | — | -8 | -0 | +3 | +12 | +10 | -0 | -8 | -7 | -3 | -3 | -2 | -3 | -3 | +1 | +3 | +0 | +27 | — |
| EHAM | — | — | — | — | — | +1 | -7 | -3 | -2 | +3 | +6 | +2 | -4 | -4 | -7 | -9 | -8 | -8 | -8 | -0 | +2 | +2 | +18 | +28 |
| LEBL | +28 | — | — | — | — | -5 | -7 | +4 | +4 | +9 | +10 | +7 | +2 | -1 | -9 | -11 | -12 | -12 | -10 | -3 | +2 | +1 | +25 | +74 |
| LEMD | +36 | +23 | +49 | — | — | +3 | -6 | +3 | +2 | +8 | +9 | +3 | -1 | -1 | -6 | -10 | -8 | -8 | -7 | -3 | -1 | +1 | +19 | +32 |
| LFPG | +23 | +15 | — | +21 | +14 | +2 | -5 | +2 | +1 | +5 | +9 | +3 | -10 | -8 | -12 | -14 | -12 | -12 | -11 | -5 | +0 | +1 | +18 | +34 |
| LIRF | — | — | — | — | +14 | +5 | +1 | +9 | +4 | +27 | +39 | +27 | -5 | -3 | -22 | -24 | -23 | -22 | -21 | -16 | -14 | -17 | +20 | +28 |
| LSZH | — | — | — | — | — | — | -5 | +5 | +1 | +6 | +9 | +3 | -4 | +2 | -4 | -7 | -5 | -4 | -2 | -1 | +4 | +6 | +13 | +34 |
| LTFM | +36 | +45 | +121 | +16 | +22 | +10 | -5 | +2 | +13 | +16 | +12 | +7 | -3 | -2 | -3 | -5 | -3 | -4 | -2 | +5 | +2 | +0 | +7 | +25 |
| n (ALL) | 585 | 841 | 486 | 366 | 587 | 908 | 3,599 | 5,730 | 5,228 | 5,990 | 6,337 | 6,192 | 6,867 | 6,336 | 5,840 | 6,239 | 6,158 | 5,571 | 4,771 | 4,845 | 5,504 | 5,988 | 3,829 | 1,203 |


### F.1f Weather block SHAP by temperature, s

| where | <-5C | -5-0C | 0-3C | 3-10C | 10-20C | 20-30C | 30C+ |
|---|---|---|---|---|---|---|---|
| ALL | +206 | +169 | +45 | -2 | -6 | -13 | -11 |
| EDDF | +190 | +119 | +32 | -3 | -6 | -13 | -15 |
| EDDM | +198 | +137 | +50 | -5 | -6 | -13 | -16 |
| EGLL | — | +162 | +30 | -5 | -7 | -13 | -18 |
| EHAM | — | +244 | +67 | +8 | -7 | -12 | -14 |
| LEBL | — | — | +15 | -6 | -6 | -12 | -14 |
| LEMD | +167 | +128 | +29 | -4 | -9 | -9 | -6 |
| LFPG | — | +245 | +71 | +1 | -6 | -17 | -17 |
| LIRF | — | — | — | -7 | -4 | -15 | -17 |
| LSZH | +250 | +140 | +24 | -6 | -6 | -11 | -10 |
| LTFM | — | — | +40 | -1 | -5 | -13 | -15 |
| n (ALL) | 304 | 5,932 | 7,988 | 21,633 | 26,556 | 33,093 | 4,488 |


### F.1g Weather block SHAP, freezing vs not, s

| where | freezing | not |
|---|---|---|
| ALL | +398 | +5 |
| EDDF | +267 | +8 |
| EDDM | +359 | +19 |
| EGLL | +317 | +2 |
| EHAM | +368 | +16 |
| LEBL | — | -9 |
| LEMD | — | -1 |
| LFPG | +507 | +18 |
| LIRF | — | -10 |
| LSZH | +414 | +12 |
| LTFM | — | -8 |
| n (ALL) | 679 | 99,321 |


### F.1h Weather block SHAP by visibility, s

| where | <1.5km | 1.5-5km | 5-9.5km | >=9.5km |
|---|---|---|---|---|
| ALL | +257 | +107 | +36 | -2 |
| EDDF | +206 | +164 | +108 | +3 |
| EDDM | +322 | +190 | +99 | +11 |
| EGLL | — | +60 | +33 | -1 |
| EHAM | +229 | +121 | +38 | +1 |
| LEBL | — | — | +3 | -10 |
| LEMD | +91 | +21 | -4 | -2 |
| LFPG | +328 | +141 | +60 | -5 |
| LIRF | — | — | -0 | -12 |
| LSZH | +275 | +138 | +40 | +2 |
| LTFM | — | +18 | +6 | -10 |
| n (ALL) | 1,408 | 3,190 | 6,937 | 88,464 |


### F.1i Delay + turnaround SHAP by delay band, s (see G.2: most of the late-side rise is recording)

| where | early>5m | early0-5m | late0-5m | late5-15m | late15-30m | late30-60m | late1-2h | late>2h |
|---|---|---|---|---|---|---|---|---|
| ALL | -126 | -59 | -9 | +25 | +72 | +102 | +120 | +85 |
| EDDF | -95 | -52 | -9 | +26 | +71 | +92 | +100 | +89 |
| EDDM | -121 | -74 | -25 | +18 | +77 | +121 | +132 | +114 |
| EGLL | -125 | -52 | +4 | +29 | +72 | +133 | +236 | +141 |
| EHAM | -84 | -44 | -8 | +18 | +40 | +53 | +56 | +50 |
| LEBL | -117 | -61 | -14 | +17 | +50 | +75 | +89 | +64 |
| LEMD | -110 | -59 | -8 | +26 | +74 | +100 | +93 | +82 |
| LFPG | -114 | -57 | -8 | +27 | +80 | +113 | +142 | +116 |
| LIRF | -151 | -77 | -12 | +28 | +87 | +122 | +136 | +118 |
| LSZH | -111 | -68 | -22 | +23 | +74 | +96 | +102 | +78 |
| LTFM | -144 | -63 | +0 | +46 | +108 | +131 | +110 | +51 |
| n (ALL) | 13,517 | 14,475 | 15,687 | 21,709 | 16,721 | 11,846 | 4,419 | 1,626 |


### F.1j Layout (runway + stand area SHAP) by runway, s

| airport | runway | n | runway + stand SHAP s |
|---|---|---|---|
| EDDF | 07C | 968 | -106 |
| EDDF | 18 | 6,947 | -53 |
| EDDF | 25C | 2,921 | -102 |
| EDDM | 08L | 1,017 | -86 |
| EDDM | 08R | 1,180 | -4 |
| EDDM | 26L | 3,367 | -114 |
| EDDM | 26R | 2,415 | -26 |
| EGLL | 09R | 1,815 | +120 |
| EGLL | 27L | 4,921 | +124 |
| EGLL | 27R | 4,974 | +148 |
| EHAM | 04 | 85 | -354 |
| EHAM | 09 | 46 | -193 |
| EHAM | 18C | 42 | -92 |
| EHAM | 18L | 3,068 | -186 |
| EHAM | 22 | 176 | -416 |
| EHAM | 24 | 4,528 | -232 |
| EHAM | 36C | 1,371 | -194 |
| EHAM | 36L | 2,630 | +94 |
| LEBL | 06L | 46 | +29 |
| LEBL | 06R | 1,802 | +20 |
| LEBL | 24L | 6,504 | +29 |
| LEBL | 24R | 89 | +184 |
| LEMD | 14L | 1,587 | +108 |
| LEMD | 14R | 1,071 | +8 |
| LEMD | 36L | 3,972 | +32 |
| LEMD | 36R | 3,825 | +135 |
| LFPG | 08L | 2,234 | -31 |
| LFPG | 09L | 82 | +124 |
| LFPG | 09R | 1,428 | +67 |
| LFPG | 26R | 4,489 | -0 |
| LFPG | 27L | 2,221 | +10 |
| LFPG | 27R | 994 | +103 |
| LIRF | 16R | 678 | +226 |
| LIRF | 25 | 6,107 | +80 |
| LIRF | 34L | 303 | +180 |
| LSZH | 10 | 32 | -39 |
| LSZH | 16 | 619 | +43 |
| LSZH | 28 | 3,857 | -180 |
| LSZH | 32 | 1,806 | -111 |
| LSZH | 34 | 76 | -4 |
| LTFM | 17L | 56 | +152 |
| LTFM | 17R | 1,226 | +242 |
| LTFM | 18 | 1,854 | +30 |
| LTFM | 34L | 348 | +227 |
| LTFM | 35L | 3,990 | +77 |
| LTFM | 35R | 85 | +64 |
| LTFM | 36 | 6,064 | -77 |


### F.2 Mean |SHAP| per feature (seconds), pooled

| feature | mean \|SHAP\| s | sd s | with proxy (model g) mean \|SHAP\| s |
|---|---|---|---|
| stand_area | 93.45 | 122.6 | 67.28 |
| rwy | 65.32 | 81.65 | 46.57 |
| delay | 62.59 | 80.82 | 72.12 |
| airline | 47.44 | 57.58 | 35.65 |
| rec_rwy_y20 | 42.69 | 52.7 | 20.87 |
| q_rwy_tko_sym10 | 40.13 | 50.06 | 35.17 |
| gapa | 30.21 | 44.98 | 29.65 |
| actype | 29.65 | 39.45 | 17.09 |
| q_apt_at_push | 24.63 | 32.46 | 4.69 |
| rec_apt_y30 | 22.8 | 34.94 | 12.05 |
| q_rwy_at_push | 17.67 | 25.44 | 3.61 |
| w_temp_c | 16.17 | 40.08 | 7.61 |
| dep_hr | 13.8 | 18.72 | 4.52 |
| q_rwy_push_pre20 | 12.0 | 16.84 | 3.51 |
| hour_local | 9.41 | 15.53 | 8.06 |
| q_arr_taxiin_sym30_push | 6.98 | 10.22 | 4.73 |
| arr_hr | 6.94 | 10.31 | 5.92 |
| w_dewspread_c | 4.58 | 7.28 | 2.7 |
| prev_arr_taxiin | 4.24 | 6.26 | 2.58 |
| w_vis_km | 3.99 | 12.79 | 3.31 |
| month | 3.07 | 4.49 | 2.45 |
| w_wind_kt | 3.04 | 5.06 | 2.47 |
| w_freezing | 2.41 | 15.83 | 0.92 |
| q_arr_taxiing_at_push | 2.17 | 3.21 | 1.75 |
| dow_local | 1.51 | 2.29 | 1.83 |
| wake | 1.15 | 1.79 | 0.26 |
| w_thunder | 0.88 | 8.71 | 0.83 |
| ap | 0.64 | 1.7 | 0.8 |
| w_gust_kt | 0.5 | 3.42 | 0.38 |
| segment | 0.43 | 1.25 | 0.2 |
| holiday | 0.2 | 0.63 | 0.31 |
| w_lowvis | 0.08 | 0.57 | 0.09 |
| w_precip_mm | 0.0 | 0.0 | 0.0 |
| proxy | — | — | 152.42 |


### F.3 Size class (wake) — mean SHAP s

| where | M | H | J | L |
|---|---|---|---|---|
| ALL | -1 | +3 | +3 | -3 |
| EDDF | -0 | +3 | — | — |
| EDDM | -0 | +3 | +4 | -4 |
| EGLL | -1 | +4 | +4 | — |
| EHAM | -0 | +3 | +2 | -2 |
| LEBL | -0 | +3 | — | -0 |
| LEMD | -1 | +3 | — | -3 |
| LFPG | -1 | +4 | +3 | — |
| LIRF | -1 | +4 | — | — |
| LSZH | -0 | +3 | — | -5 |
| LTFM | -0 | +2 | — | — |
| n (ALL) | 78,091 | 20,743 | 678 | 486 |


### F.4 Local push hour — mean SHAP s

| where | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | +30 | +35 | +106 | +17 | +16 | +2 | -6 | +1 | +2 | +8 | +10 | +4 | -5 | -4 | -8 | -11 | -9 | -9 | -8 | -2 | +0 | +1 | +22 | +38 |
| EDDF | — | — | — | — | +24 | +2 | -7 | +1 | -2 | +2 | +7 | +5 | -6 | -6 | -8 | -10 | -8 | -8 | -7 | +0 | +3 | +3 | +34 | — |
| EDDM | — | — | — | — | — | -4 | -5 | -1 | -1 | +6 | +5 | +1 | -6 | -6 | -9 | -12 | -10 | -9 | -5 | +5 | +9 | +10 | +29 | +44 |
| EGLL | — | — | — | — | — | — | -10 | -1 | +2 | +10 | +9 | -2 | -9 | -8 | -4 | -4 | -3 | -4 | -4 | +1 | +2 | -1 | +26 | — |
| EHAM | — | — | — | — | — | +1 | -6 | -2 | -1 | +4 | +7 | +3 | -3 | -3 | -6 | -8 | -7 | -8 | -7 | +0 | +3 | +3 | +19 | +32 |
| LEBL | +25 | — | — | — | — | -6 | -9 | +2 | +3 | +8 | +8 | +5 | +0 | -3 | -10 | -13 | -13 | -14 | -12 | -4 | +0 | -0 | +23 | +73 |
| LEMD | +32 | +19 | +48 | — | — | +2 | -8 | +2 | +2 | +7 | +7 | +3 | -1 | -2 | -7 | -11 | -8 | -9 | -8 | -3 | -2 | +1 | +18 | +31 |
| LFPG | +26 | +18 | — | +19 | +12 | +3 | -4 | +4 | +3 | +7 | +11 | +5 | -8 | -6 | -10 | -12 | -10 | -10 | -9 | -2 | +1 | +3 | +21 | +40 |
| LIRF | — | — | — | — | +15 | +6 | +2 | +9 | +6 | +28 | +40 | +27 | -4 | -2 | -21 | -23 | -22 | -21 | -20 | -15 | -13 | -16 | +22 | +28 |
| LSZH | — | — | — | — | — | — | -6 | +3 | -1 | +3 | +7 | +1 | -6 | -0 | -5 | -8 | -6 | -5 | -3 | -2 | +2 | +4 | +12 | +33 |
| LTFM | +30 | +39 | +117 | +15 | +20 | +7 | -7 | -1 | +8 | +11 | +10 | +4 | -7 | -6 | -8 | -10 | -7 | -8 | -6 | +0 | -2 | -3 | +4 | +22 |
| n (ALL) | 585 | 841 | 486 | 366 | 587 | 908 | 3,599 | 5,730 | 5,228 | 5,990 | 6,337 | 6,192 | 6,867 | 6,336 | 5,840 | 6,239 | 6,158 | 5,571 | 4,771 | 4,845 | 5,504 | 5,988 | 3,829 | 1,203 |


### F.5 Runway queue at push — mean SHAP s

| where | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | >24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | -6 | -7 | -10 | -13 | -13 | -15 | -13 | -7 | +1 | +13 | +19 | +28 | +31 | +40 | +43 | +57 | +62 | +69 | +80 | +84 | +90 | +101 | +108 | +108 | +100 | +91 |
| EDDF | -7 | -9 | -12 | -14 | -12 | -13 | -11 | -5 | +3 | +14 | +19 | +30 | +32 | +40 | — | — | — | — | — | — | — | — | — | — | — | — |
| EDDM | -6 | -8 | -11 | -13 | -11 | -12 | -10 | -5 | +3 | +13 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| EGLL | -1 | -1 | -9 | -11 | -12 | -16 | -16 | -15 | -9 | +4 | +11 | +22 | +27 | +38 | +43 | +58 | +62 | +70 | +80 | +84 | +89 | +101 | +108 | +109 | +100 | +91 |
| EHAM | -12 | -11 | -13 | -16 | -14 | -15 | -11 | -3 | +6 | +19 | +26 | +36 | +38 | +46 | +47 | +58 | +63 | — | — | — | — | — | — | — | — | — |
| LEBL | -2 | -2 | -7 | -11 | -10 | -12 | -11 | -6 | +0 | +8 | +12 | +19 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| LEMD | -2 | -4 | -8 | -11 | -10 | -12 | -11 | -6 | -1 | +8 | +12 | +20 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| LFPG | -8 | -10 | -14 | -17 | -16 | -18 | -17 | -10 | +4 | +21 | +30 | +46 | +51 | +61 | +70 | — | — | — | — | — | — | — | — | — | — | — |
| LIRF | -2 | -3 | -8 | -12 | -12 | -14 | -15 | -10 | -2 | +8 | +16 | +26 | +28 | +33 | — | — | — | — | — | — | — | — | — | — | — | — |
| LSZH | -6 | -8 | -10 | -14 | -15 | -20 | -18 | -12 | +3 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| LTFM | -5 | -7 | -10 | -14 | -14 | -15 | -14 | -9 | -1 | +12 | +19 | +27 | +30 | +39 | +42 | +51 | +55 | +61 | — | — | — | — | — | — | — | — |
| n (ALL) | 3,812 | 6,027 | 8,183 | 9,693 | 10,727 | 10,894 | 9,952 | 8,404 | 6,872 | 5,380 | 4,070 | 3,199 | 2,416 | 2,066 | 1,610 | 1,535 | 1,345 | 1,109 | 894 | 683 | 473 | 289 | 175 | 92 | 52 | 48 |


### F.6 Airport queue at push — mean SHAP s

| where | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | >30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | -50 | -43 | -39 | -35 | -30 | -25 | -21 | -19 | -17 | -15 | -14 | -11 | -7 | -4 | +4 | +15 | +31 | +42 | +56 | +64 | +75 | +79 | +85 | +79 | +74 | +82 | +75 | +76 | +73 | +75 | +81 | +95 |
| EDDF | -42 | -35 | -33 | -30 | -26 | -22 | -18 | -15 | -12 | -9 | -8 | -5 | -2 | -0 | +4 | +10 | +17 | +22 | +29 | +32 | — | — | — | — | — | — | — | — | — | — | — | — |
| EDDM | -34 | -28 | -26 | -23 | -21 | -18 | -16 | -14 | -12 | -10 | -10 | -8 | -5 | -3 | +1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| EGLL | — | -96 | -100 | -86 | -76 | -66 | -60 | -54 | -50 | -46 | -45 | -38 | -27 | -19 | -0 | +22 | +47 | +64 | +83 | +92 | +109 | +117 | +132 | +132 | +130 | +146 | — | — | — | — | — | — |
| EHAM | -53 | -45 | -38 | -35 | -29 | -23 | -19 | -16 | -14 | -12 | -10 | -6 | -2 | +0 | +7 | +14 | +21 | +27 | +37 | +42 | +47 | +54 | +64 | +62 | +61 | — | — | — | — | — | — | — |
| LEBL | -42 | -33 | -28 | -25 | -22 | -20 | -19 | -17 | -17 | -16 | -16 | -16 | -11 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| LEMD | -40 | -32 | -27 | -25 | -24 | -21 | -19 | -16 | -14 | -12 | -10 | -8 | -5 | -3 | +1 | +6 | +14 | +17 | +24 | +25 | — | — | — | — | — | — | — | — | — | — | — | — |
| LFPG | -58 | -52 | -47 | -40 | -36 | -30 | -25 | -22 | -18 | -15 | -13 | -9 | -4 | -1 | +8 | +18 | +31 | +37 | +47 | +60 | +69 | +75 | +85 | — | — | — | — | — | — | — | — | — |
| LIRF | -56 | -53 | -51 | -48 | -42 | -36 | -29 | -26 | -21 | -17 | -15 | -10 | -4 | -1 | +8 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| LSZH | -52 | -49 | -47 | -43 | -32 | -22 | -15 | -14 | -12 | -10 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| LTFM | -55 | -52 | -50 | -45 | -38 | -33 | -28 | -21 | -17 | -14 | -12 | -8 | -3 | -0 | +6 | +14 | +25 | +31 | +38 | +44 | +51 | +54 | +63 | +60 | +61 | +69 | +68 | +71 | +71 | +74 | +79 | +94 |
| n (ALL) | 1,011 | 2,203 | 3,548 | 5,004 | 6,479 | 7,243 | 7,573 | 7,292 | 6,953 | 6,683 | 6,230 | 5,945 | 5,398 | 5,262 | 4,566 | 4,127 | 3,441 | 2,852 | 2,293 | 1,680 | 1,187 | 803 | 572 | 421 | 338 | 248 | 184 | 123 | 99 | 67 | 73 | 102 |


### F.7 Departures ±30 min — mean SHAP s

| where | [0,10) | [10,20) | [20,30) | [30,40) | [40,50) | [50,60) | [60,70) | [70,200) |
|---|---|---|---|---|---|---|---|---|
| ALL | +44 | +21 | +14 | +13 | -8 | -12 | -20 | -35 |
| EDDF | +42 | +22 | +14 | +7 | -4 | -11 | -22 | — |
| EDDM | +46 | +23 | +13 | +7 | -3 | -6 | — | — |
| EGLL | +72 | +47 | +37 | +32 | -16 | -78 | — | — |
| EHAM | +50 | +27 | +18 | +8 | -5 | -10 | -16 | -25 |
| LEBL | +32 | +18 | +11 | +9 | +4 | — | — | — |
| LEMD | +42 | +20 | +13 | +8 | -2 | -8 | -15 | — |
| LFPG | +54 | +33 | +21 | +10 | -7 | -16 | -42 | — |
| LIRF | +30 | +13 | +10 | +14 | +4 | — | — | — |
| LSZH | +43 | +18 | +11 | +10 | +1 | — | — | — |
| LTFM | +41 | +23 | +16 | +11 | -6 | -12 | -26 | -43 |
| n (ALL) | 2,069 | 10,185 | 21,830 | 27,287 | 25,842 | 10,082 | 2,431 | 274 |


### F.8 Recent runway taxi times — mean SHAP s

| where | [0,600) | [600,750) | [750,900) | [900,1050) | [1050,1200) | [1200,1500) | [1500,1800) | [1800,1e+09) | NaN |
|---|---|---|---|---|---|---|---|---|---|
| ALL | -54 | -46 | -32 | +2 | +49 | +79 | +117 | +160 | +53 |
| EDDF | -42 | -40 | -31 | -4 | +45 | +86 | — | — | — |
| EDDM | -49 | -47 | -38 | -4 | +54 | +91 | +138 | +170 | — |
| EGLL | — | — | -24 | +8 | +51 | +77 | +109 | +133 | — |
| EHAM | -59 | -53 | -43 | +12 | +63 | +90 | — | +160 | +57 |
| LEBL | — | -36 | -29 | -3 | +39 | +72 | +138 | — | — |
| LEMD | — | -38 | -28 | +3 | +43 | +67 | +116 | — | — |
| LFPG | — | -39 | -30 | +2 | +45 | +72 | +127 | +157 | +61 |
| LIRF | — | -43 | -32 | +1 | +50 | +91 | +138 | +223 | — |
| LSZH | -44 | -45 | -37 | -0 | +49 | +80 | — | — | — |
| LTFM | — | -42 | -31 | +6 | +54 | +82 | +121 | +156 | +62 |
| n (ALL) | 3,931 | 17,409 | 26,732 | 23,536 | 12,604 | 11,323 | 3,054 | 1,134 | 277 |


### F.9 Delay at push (s) — mean SHAP s

| where | [-1e+09,-300) | [-300,0) | [0,300) | [300,900) | [900,1800) | [1800,3600) | [3600,7200) | [7200,1e+09) |
|---|---|---|---|---|---|---|---|---|
| ALL | -132 | -68 | -17 | +20 | +66 | +95 | +115 | +79 |
| EDDF | -102 | -60 | -18 | +17 | +60 | +82 | +98 | +69 |
| EDDM | -115 | -75 | -25 | +17 | +72 | +111 | +127 | +105 |
| EGLL | -133 | -71 | -9 | +22 | +62 | +110 | +193 | +128 |
| EHAM | -86 | -48 | -10 | +19 | +44 | +57 | +68 | +54 |
| LEBL | -114 | -62 | -14 | +20 | +60 | +84 | +95 | +71 |
| LEMD | -117 | -68 | -20 | +18 | +66 | +92 | +93 | +69 |
| LFPG | -114 | -63 | -18 | +17 | +66 | +101 | +129 | +107 |
| LIRF | -148 | -85 | -18 | +29 | +90 | +124 | +136 | +105 |
| LSZH | -103 | -65 | -24 | +18 | +70 | +95 | +103 | +77 |
| LTFM | -155 | -80 | -18 | +30 | +91 | +116 | +106 | +51 |
| n (ALL) | 11,432 | 13,306 | 16,224 | 22,712 | 17,726 | 12,375 | 4,579 | 1,646 |


### F.10 Month — mean SHAP s (Jan, Jul are held out: the model never saw them)

| where | 1 | 7 |
|---|---|---|
| ALL | -1 | +2 |
| EDDF | +0 | +1 |
| EDDM | -2 | +2 |
| EGLL | -0 | +3 |
| EHAM | -3 | +1 |
| LEBL | +2 | +2 |
| LEMD | -1 | +2 |
| LFPG | -6 | +1 |
| LIRF | -3 | +0 |
| LSZH | +1 | +2 |
| LTFM | +4 | +4 |
| n (ALL) | 44,925 | 55,075 |


### F.11 Holiday — mean SHAP s

| where | 0 | 1 |
|---|---|---|
| ALL | -0 | +3 |
| EDDF | -0 | +2 |
| EDDM | -0 | +2 |
| EGLL | -0 | +5 |
| EHAM | -0 | +2 |
| LEBL | -0 | +2 |
| LEMD | -0 | +3 |
| LFPG | -0 | +3 |
| LIRF | -0 | +4 |
| LSZH | -0 | +1 |
| LTFM | -0 | +4 |
| n (ALL) | 97,303 | 2,697 |


### F.12 Temperature — mean SHAP s

| where | [-1e+09,-5) | [-5,0) | [0,3) | [3,10) | [10,20) | [20,30) | [30,1e+09) |
|---|---|---|---|---|---|---|---|
| ALL | +180 | +134 | +67 | -2 | -5 | -8 | -7 |
| EDDF | — | +113 | +50 | -0 | -4 | -7 | -8 |
| EDDM | +184 | +116 | +67 | -2 | -5 | -6 | -8 |
| EGLL | — | +155 | +54 | -1 | -4 | -6 | -11 |
| EHAM | — | +163 | +88 | -1 | -6 | -6 | -10 |
| LEBL | — | — | +34 | -3 | -3 | -7 | -8 |
| LEMD | — | +140 | +43 | -4 | -5 | -5 | -3 |
| LFPG | — | +178 | +95 | -2 | -7 | -8 | -10 |
| LIRF | — | — | — | -4 | -2 | -10 | -11 |
| LSZH | — | +113 | +44 | -2 | -5 | -6 | -7 |
| LTFM | — | — | +62 | -4 | -4 | -10 | -11 |
| n (ALL) | 129 | 3,924 | 7,177 | 22,331 | 24,895 | 35,376 | 6,162 |


### F.13 Freezing flag — mean SHAP s

| where | [-1e+09,0.5) | [0.5,1e+09) |
|---|---|---|
| ALL | -1 | +184 |
| EDDF | -1 | +197 |
| EDDM | -1 | +187 |
| EGLL | -2 | +186 |
| EHAM | -1 | +151 |
| LEBL | -1 | — |
| LEMD | -1 | — |
| LFPG | -1 | +206 |
| LIRF | -1 | — |
| LSZH | -1 | +197 |
| LTFM | -1 | — |
| n (ALL) | 99,320 | 679 |


### F.14 Visibility — mean SHAP s

| where | [-1e+09,0.8) | [0.8,1.5) | [1.5,3) | [3,5) | [5,9.5) | [9.5,1e+09) |
|---|---|---|---|---|---|---|
| ALL | +97 | +74 | +37 | +19 | +3 | -2 |
| EDDF | — | +58 | +47 | +23 | +4 | -2 |
| EDDM | +82 | +71 | +47 | +25 | +4 | -2 |
| EGLL | — | — | +33 | +18 | +4 | -2 |
| EHAM | +85 | +64 | +31 | +16 | +2 | -2 |
| LEBL | — | — | — | — | +2 | -2 |
| LEMD | +104 | +74 | +41 | +20 | +1 | -2 |
| LFPG | +126 | +96 | +42 | +24 | +3 | -2 |
| LIRF | — | — | — | — | +4 | -2 |
| LSZH | +80 | +62 | +38 | +14 | +2 | -2 |
| LTFM | — | — | +30 | +14 | +3 | -2 |
| n (ALL) | 736 | 672 | 1,533 | 1,657 | 6,937 | 88,464 |


### F.15 Wind — mean SHAP s

| where | [-1e+09,5) | [5,10) | [10,15) | [15,20) | [20,25) | [25,30) | [30,1e+09) |
|---|---|---|---|---|---|---|---|
| ALL | -3 | -1 | +1 | +8 | +10 | +32 | +24 |
| EDDF | -4 | -1 | +2 | +13 | +20 | — | — |
| EDDM | -3 | -1 | +1 | +7 | +8 | — | — |
| EGLL | -4 | -2 | +1 | +15 | +20 | — | — |
| EHAM | -3 | -1 | +1 | +7 | +11 | +28 | +22 |
| LEBL | -2 | -0 | +1 | +5 | +7 | — | — |
| LEMD | -3 | -1 | -1 | +6 | +10 | — | — |
| LFPG | -2 | -1 | +0 | +8 | +11 | — | — |
| LIRF | -5 | -1 | +2 | +13 | +14 | — | — |
| LSZH | -2 | -1 | +0 | +8 | — | — | — |
| LTFM | -3 | -0 | +2 | +7 | +8 | — | — |
| n (ALL) | 25,883 | 42,013 | 21,220 | 8,573 | 2,125 | 134 | 51 |


### F.16 Precipitation — mean SHAP s

| where | [-1e+09,0.01) |
|---|---|
| ALL | +0 |
| EDDF | +0 |
| EDDM | +0 |
| EGLL | +0 |
| EHAM | +0 |
| LEBL | +0 |
| LEMD | +0 |
| LFPG | +0 |
| LIRF | +0 |
| LSZH | +0 |
| LTFM | +0 |
| n (ALL) | 99,999 |


### F.x q_arr_taxiing_at_push — mean SHAP s

| where | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | >20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | -2 | -2 | -2 | -1 | -1 | -0 | -0 | +2 | +3 | +3 | +3 | +4 | +4 | +5 | +6 | +5 | +6 | +10 | +9 | +24 | +22 | +22 |
| n (ALL) | 5,454 | 10,859 | 13,009 | 12,615 | 12,087 | 11,273 | 9,648 | 7,518 | 5,301 | 3,766 | 2,554 | 1,791 | 1,247 | 875 | 650 | 484 | 340 | 236 | 141 | 73 | 35 | 44 |


### F.x q_rwy_push_pre20 — mean SHAP s

| where | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | >20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | +56 | +38 | +27 | +18 | +14 | +9 | +7 | +4 | +0 | -2 | -6 | -8 | -11 | -13 | -16 | -19 | -22 | -24 | -32 | -33 | -37 | -39 |
| n (ALL) | 2,460 | 3,514 | 4,854 | 6,190 | 7,555 | 8,530 | 9,110 | 9,370 | 9,106 | 8,411 | 7,456 | 6,548 | 5,450 | 4,210 | 2,918 | 1,812 | 1,162 | 710 | 373 | 161 | 60 | 40 |


### F.x q_rwy_tko_sym10 — mean SHAP s

| where | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | >12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | -126 | -103 | -85 | -70 | -54 | -38 | -24 | -13 | -0 | +12 | +29 | +43 | +54 | +59 |
| n (ALL) | 1,928 | 2,987 | 4,556 | 6,280 | 7,644 | 8,711 | 9,510 | 9,958 | 9,776 | 9,179 | 8,191 | 6,970 | 5,306 | 9,004 |


### F.x arr_hr — mean SHAP s

| where | [0,10) | [10,20) | [20,30) | [30,40) | [40,50) | [50,60) | [60,200) |
|---|---|---|---|---|---|---|---|
| ALL | -22 | -8 | -4 | +2 | +8 | +14 | +14 |
| n (ALL) | 5,193 | 16,148 | 26,903 | 29,226 | 16,815 | 4,837 | 878 |


### F.x rec_apt_y30 — mean SHAP s

| where | [0,600) | [600,750) | [750,900) | [900,1050) | [1050,1200) | [1200,1500) | [1500,1800) | [1800,1e+09) | NaN |
|---|---|---|---|---|---|---|---|---|---|
| ALL | -30 | -24 | -15 | -8 | +15 | +52 | +91 | +176 | -57 |
| n (ALL) | 2,609 | 16,161 | 29,412 | 25,684 | 11,923 | 10,496 | 2,648 | 1,011 | 56 |


### F.x q_arr_taxiin_sym30_push — mean SHAP s

| where | [0,240) | [240,360) | [360,480) | [480,600) | [600,900) | [900,1e+09) | NaN |
|---|---|---|---|---|---|---|---|
| ALL | -14 | -11 | -3 | -1 | +8 | +16 | -21 |
| n (ALL) | 5,962 | 11,685 | 16,808 | 26,136 | 33,161 | 6,156 | 92 |


### F.x prev_arr_taxiin — mean SHAP s

| where | [-1e+09,240) | [240,360) | [360,480) | [480,600) | [600,900) | [900,1e+09) | NaN |
|---|---|---|---|---|---|---|---|
| ALL | -6 | -4 | -1 | +2 | +4 | +5 | -13 |
| n (ALL) | 11,942 | 16,910 | 18,096 | 16,574 | 23,811 | 11,938 | 729 |


### F.x gapa — mean SHAP s

| where | [-1e+09,1800) | [1800,3600) | [3600,7200) | [7200,14400) | [14400,43200) | [43200,1e+12) | NaN |
|---|---|---|---|---|---|---|---|
| ALL | +177 | -56 | +0 | +26 | +26 | +32 | +47 |
| n (ALL) | 917 | 20,695 | 36,600 | 17,310 | 17,987 | 5,738 | 753 |


### F.x dow_local — mean SHAP s

| where | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| ALL | -0 | +1 | +1 | -0 | +0 | -2 | -0 |
| n (ALL) | 13,129 | 14,398 | 15,908 | 16,368 | 14,859 | 12,381 | 12,957 |


### F.x w_dewspread_c — mean SHAP s

| where | [-1e+09,1) | [1,3) | [3,6) | [6,10) | [10,1e+09) |
|---|---|---|---|---|---|
| ALL | +13 | +8 | -0 | -4 | -2 |
| n (ALL) | 4,588 | 21,267 | 28,493 | 22,625 | 23,021 |


### F.x w_gust_kt — mean SHAP s

| where | [-1e+09,0.5) | [0.5,25) | [25,35) | [35,1e+09) |
|---|---|---|---|---|
| ALL | -0 | +5 | +9 | +55 |
| n (ALL) | 97,339 | 1,505 | 961 | 194 |


### F.x w_lowvis — mean SHAP s

| where | [-1e+09,0.5) | [0.5,1e+09) |
|---|---|---|
| ALL | -0 | +4 |
| n (ALL) | 98,591 | 1,408 |


### F.x w_thunder — mean SHAP s

| where | [-1e+09,0.5) | [0.5,1e+09) |
|---|---|---|
| ALL | -0 | +135 |
| n (ALL) | 99,650 | 349 |


### F.x segment — mean SHAP s

| where | Mainline | Lowcost | Regional | Cargo | Business | Non-Scheduled | Other |
|---|---|---|---|---|---|---|---|
| ALL | +0 | -1 | -0 | +1 | -1 | -3 | -7 |
| n (ALL) | 64,807 | 18,576 | 10,388 | 2,940 | 2,247 | 689 | 331 |


### F.x actype — mean SHAP s

| where | A320 | B738 | A20N | A321 | A21N | A319 | E190 | BCS3 | B77W | B789 | A359 | A333 | B38M | B772 | A332 | CRJX | E195 | E295 | CRJ9 | B788 | B739 | E75L | B77L | A339 | A388 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | -25 | -22 | -4 | -30 | +3 | -24 | -13 | +21 | +38 | +42 | +51 | +38 | +40 | +73 | +51 | -21 | -25 | +15 | -78 | +42 | -12 | -20 | +82 | +72 | +146 |
| n (ALL) | 16,682 | 10,531 | 7,867 | 7,814 | 6,948 | 6,291 | 3,740 | 3,468 | 3,335 | 3,009 | 2,774 | 2,368 | 2,237 | 1,641 | 1,569 | 1,494 | 1,449 | 1,306 | 1,127 | 1,115 | 936 | 908 | 844 | 786 | 678 |


### F.x airline — mean SHAP s

| where | THY | AFR | BAW | DLH | KL1 | other | VLG | SWR | RYR | IBE | LH1 | ITY | AEA | KL0 | EJU | ANE | IBS | WMT | SHT | EZY | LH8 | EWG | UAL | EN8 | AAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | +52 | -20 | +85 | -41 | -42 | +27 | -20 | -43 | +35 | -34 | -68 | -28 | -63 | -60 | +14 | -72 | -114 | +33 | +6 | +41 | -41 | -28 | +99 | +20 | +61 |
| n (ALL) | 10,979 | 6,686 | 5,650 | 5,003 | 4,755 | 4,161 | 3,848 | 3,556 | 3,251 | 3,044 | 2,642 | 2,435 | 1,747 | 1,531 | 1,451 | 1,353 | 1,003 | 921 | 918 | 864 | 835 | 697 | 687 | 683 | 663 |


### F.17 Runway — mean SHAP s per runway (within airport)

| airport | runway | n | mean SHAP s |
|---|---|---|---|
| EDDF | 18 | 6,947 | +22 |
| EDDF | 25C | 2,921 | -60 |
| EDDF | 07C | 968 | -46 |
| EDDM | 26L | 3,367 | -92 |
| EDDM | 26R | 2,415 | +7 |
| EDDM | 08R | 1,180 | +26 |
| EDDM | 08L | 1,017 | -52 |
| EGLL | 27R | 4,974 | +52 |
| EGLL | 27L | 4,921 | +22 |
| EGLL | 09R | 1,815 | +32 |
| EHAM | 24 | 4,528 | -129 |
| EHAM | 18L | 3,068 | -85 |
| EHAM | 36L | 2,630 | +147 |
| EHAM | 36C | 1,371 | -85 |
| EHAM | 22 | 176 | -233 |
| EHAM | 04 | 85 | -180 |
| EHAM | 09 | 46 | -106 |
| EHAM | 18C | 42 | -12 |
| LEBL | 24L | 6,504 | -35 |
| LEBL | 06R | 1,802 | -52 |
| LEBL | 24R | 89 | +105 |
| LEBL | 06L | 46 | +16 |
| LEMD | 36L | 3,972 | -13 |
| LEMD | 36R | 3,825 | +96 |
| LEMD | 14L | 1,587 | +68 |
| LEMD | 14R | 1,071 | -24 |
| LFPG | 26R | 4,489 | +18 |
| LFPG | 08L | 2,234 | -22 |
| LFPG | 27L | 2,221 | +20 |
| LFPG | 09R | 1,428 | +80 |
| LFPG | 27R | 994 | +118 |
| LFPG | 09L | 82 | +152 |
| LIRF | 25 | 6,107 | +31 |
| LIRF | 16R | 678 | +124 |
| LIRF | 34L | 303 | +96 |
| LSZH | 28 | 3,857 | -95 |
| LSZH | 32 | 1,806 | -42 |
| LSZH | 16 | 619 | +79 |
| LSZH | 34 | 76 | +0 |
| LSZH | 10 | 32 | +22 |
| LTFM | 36 | 6,064 | -104 |
| LTFM | 35L | 3,990 | +46 |
| LTFM | 18 | 1,854 | -10 |
| LTFM | 17R | 1,226 | +186 |
| LTFM | 34L | 348 | +170 |
| LTFM | 35R | 85 | +18 |
| LTFM | 17L | 56 | +118 |


### F.18 Stand area — mean SHAP s (5 lowest / 5 highest per airport, n ≥ 30 in the sample)

| airport | stand area | n | mean SHAP s |
|---|---|---|---|
| EDDF | B1 | 107 | -281 |
| EDDF | C2 | 42 | -231 |
| EDDF | V2 | 139 | -225 |
| EDDF | V7 | 178 | -199 |
| EDDF | V1 | 3,253 | -166 |
| EDDF | E5 | 88 | +77 |
| EDDF | V9 | 381 | +107 |
| EDDF | E6 | 89 | +120 |
| EDDF | E9 | 89 | +178 |
| EDDF | V3 | 140 | +200 |
| EDDM | 58 | 141 | -190 |
| EDDM | 59 | 62 | -181 |
| EDDM | 34 | 515 | -175 |
| EDDM | 80 | 52 | -166 |
| EDDM | 23 | 198 | -55 |
| EDDM | 31 | 607 | +22 |
| EDDM | 12 | 180 | +28 |
| EDDM | 35 | 253 | +33 |
| EDDM | 37 | 55 | +43 |
| EDDM | 11 | 885 | +44 |
| EGLL | 25 | 85 | -83 |
| EGLL | 22 | 980 | -45 |
| EGLL | other | 33 | -19 |
| EGLL | 24 | 512 | +2 |
| EGLL | 21 | 724 | +5 |
| EGLL | 56 | 398 | +147 |
| EGLL | 53 | 530 | +148 |
| EGLL | 52 | 744 | +171 |
| EGLL | 51 | 1,127 | +174 |
| EGLL | 50 | 1,113 | +208 |
| EHAM | Y7 | 92 | -185 |
| EHAM | K1 | 44 | -185 |
| EHAM | K2 | 95 | -180 |
| EHAM | K3 | 131 | -179 |
| EHAM | G7 | 64 | -178 |
| EHAM | S8 | 69 | +3 |
| EHAM | S7 | 60 | +38 |
| EHAM | S9 | 80 | +45 |
| EHAM | S6 | 71 | +53 |
| EHAM | R7 | 43 | +177 |
| LEBL | 42 | 53 | -259 |
| LEBL | 41 | 93 | -242 |
| LEBL | 30 | 39 | -216 |
| LEBL | 33 | 83 | -208 |
| LEBL | 40 | 42 | -202 |
| LEBL | 12 | 460 | +271 |
| LEBL | 10 | 889 | +300 |
| LEBL | 11 | 1,138 | +306 |
| LEBL | 96 | 72 | +342 |
| LEBL | 95 | 36 | +342 |
| LEMD | 61 | 100 | -288 |
| LEMD | 60 | 111 | -287 |
| LEMD | 40 | 233 | -168 |
| LEMD | 41 | 262 | -164 |
| LEMD | 22 | 85 | -162 |
| LEMD | T6 | 72 | +214 |
| LEMD | T9 | 83 | +229 |
| LEMD | T4 | 73 | +244 |
| LEMD | 71 | 66 | +313 |
| LEMD | 73 | 81 | +334 |
| LFPG | other | 151 | -141 |
| LFPG | X0 | 382 | -130 |
| LFPG | Q0 | 140 | -94 |
| LFPG | E0 | 291 | -63 |
| LFPG | J1 | 516 | -53 |
| LFPG | I7 | 90 | +67 |
| LFPG | I4 | 129 | +76 |
| LFPG | K2 | 85 | +91 |
| LFPG | K0 | 99 | +106 |
| LFPG | N0 | 104 | +164 |
| LIRF | 22 | 339 | -166 |
| LIRF | 23 | 452 | -163 |
| LIRF | 20 | 131 | -63 |
| LIRF | 32 | 58 | -22 |
| LIRF | 30 | 1,368 | +32 |
| LIRF | 82 | 133 | +198 |
| LIRF | 70 | 508 | +203 |
| LIRF | 71 | 119 | +212 |
| LIRF | 90 | 93 | +236 |
| LIRF | 80 | 221 | +268 |
| LSZH | F7 | 151 | -329 |
| LSZH | H1 | 322 | -301 |
| LSZH | I0 | 332 | -294 |
| LSZH | GA | 113 | -271 |
| LSZH | RE | 42 | -262 |
| LSZH | B3 | 696 | +36 |
| LSZH | C5 | 45 | +75 |
| LSZH | G1 | 66 | +94 |
| LSZH | G0 | 126 | +94 |
| LSZH | T5 | 47 | +134 |
| LTFM | K1 | 330 | -220 |
| LTFM | K5 | 153 | -110 |
| LTFM | 13 | 143 | -41 |
| LTFM | G2 | 228 | -29 |
| LTFM | F3 | 239 | -20 |
| LTFM | F4 | 265 | +99 |
| LTFM | A2 | 143 | +102 |
| LTFM | B8 | 203 | +108 |
| LTFM | B6 | 186 | +136 |
| LTFM | A1 | 314 | +142 |


### F.19 Slopes: seconds of SHAP per unit (linear fit of SHAP on the feature, 1st pct to cap)

| feature | ALL | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q_apt_at_push | +4.88 | +3.22 | +2.20 | +13.49 | +4.25 | +1.75 | +2.65 | +5.12 | +4.81 | +6.46 | +5.37 |
| q_rwy_at_push | +3.98 | +3.29 | +0.34 | +6.82 | +5.11 | +2.00 | +0.97 | +4.41 | +2.48 | -1.48 | +4.57 |
| q_arr_taxiing_at_push | +0.65 | +0.62 | +0.61 | +0.07 | +0.60 | +0.41 | +0.87 | +0.78 | +1.26 | +0.28 | +0.63 |
| q_rwy_push_pre20 | -3.88 | -3.75 | -3.81 | -3.75 | -3.28 | -3.26 | -3.91 | -4.36 | -5.38 | -4.21 | -4.55 |
| dep_hr | -0.99 | -0.91 | -0.82 | -4.10 | -0.99 | -0.50 | -0.79 | -1.40 | -0.09 | -0.56 | -1.04 |
| arr_hr | +0.62 | +0.51 | +0.72 | +0.39 | +0.44 | +0.79 | +0.53 | +0.80 | +0.74 | +0.26 | +0.83 |
| rec_rwy_y20 | +0.20 | +0.18 | +0.19 | +0.14 | +0.20 | +0.20 | +0.23 | +0.21 | +0.24 | +0.16 | +0.22 |
| rec_apt_y30 | +0.12 | +0.09 | +0.15 | +0.15 | +0.07 | +0.09 | +0.12 | +0.15 | +0.12 | +0.10 | +0.16 |
| delay | +0.07 | +0.06 | +0.07 | +0.07 | +0.04 | +0.06 | +0.06 | +0.06 | +0.08 | +0.06 | +0.10 |


### F.20 Queue × runway: seconds per extra aircraft queued on the runway at push (runways with ≥ 1,000 sample rows)

| runway | n | mean queue | s per aircraft |
|---|---|---|---|
| EDDF\|18 | 6,947 | 6.25 | 3.854 |
| EDDF\|25C | 2,921 | 3.33 | 0.343 |
| EDDM\|08L | 1,017 | 2.57 | 0.141 |
| EDDM\|08R | 1,180 | 3.54 | 0.738 |
| EDDM\|26L | 3,367 | 3.3 | 0.646 |
| EDDM\|26R | 2,415 | 2.35 | -1.049 |
| EGLL\|09R | 1,815 | 14.37 | 8.32 |
| EGLL\|27L | 4,921 | 13.8 | 6.5 |
| EGLL\|27R | 4,974 | 14.36 | 7.249 |
| EHAM\|18L | 3,068 | 6.13 | 5.448 |
| EHAM\|24 | 4,528 | 5.62 | 5.261 |
| EHAM\|36C | 1,371 | 6.86 | 5.628 |
| EHAM\|36L | 2,630 | 8.23 | 5.336 |
| LEBL\|06R | 1,802 | 5.17 | 1.323 |
| LEBL\|24L | 6,504 | 6.14 | 2.689 |
| LEMD\|14L | 1,587 | 5.2 | 1.206 |
| LEMD\|14R | 1,071 | 4.0 | 0.036 |
| LEMD\|36L | 3,972 | 4.0 | 0.18 |
| LEMD\|36R | 3,825 | 5.13 | 1.295 |
| LFPG\|08L | 2,234 | 5.85 | 5.978 |
| LFPG\|09R | 1,428 | 4.76 | 4.109 |
| LFPG\|26R | 4,489 | 5.5 | 5.002 |
| LFPG\|27L | 2,221 | 4.5 | 3.094 |
| LIRF\|25 | 6,107 | 5.74 | 3.112 |
| LSZH\|28 | 3,857 | 3.53 | -1.492 |
| LSZH\|32 | 1,806 | 3.58 | -0.612 |
| LTFM\|17R | 1,226 | 6.14 | 4.99 |
| LTFM\|18 | 1,854 | 9.56 | 4.465 |
| LTFM\|35L | 3,990 | 5.56 | 4.087 |
| LTFM\|36 | 6,064 | 8.56 | 4.781 |


### F.21 Interaction proxies (stratified SHAP; LightGBM 4.5 has no pred_interactions and `shap` is not installed)

| airport | Heavy − Medium wake SHAP s | weather-block SHAP when freezing s (n) | weather-block SHAP otherwise s |
|---|---|---|---|
| EDDF | 3.5 | 267 (31) | 8.5 |
| EDDM | 4.0 | 359 (161) | 19.0 |
| EGLL | 4.9 | 317 (35) | 1.7 |
| EHAM | 3.2 | 368 (182) | 16.1 |
| LEBL | 3.4 | — (0) | -9.1 |
| LEMD | 3.7 | — (1) | -0.8 |
| LFPG | 4.1 | 507 (148) | 18.2 |
| LIRF | 4.9 | — (0) | -10.5 |
| LSZH | 3.2 | 414 (121) | 12.2 |
| LTFM | 2.4 | — (0) | -7.7 |


## G. The same model with the NM clock proxy (model g)

Held-out RMSE 238.586 s (R² 0.6946) vs 275.327 s (R² 0.5933) without it. Block mean |SHAP| (s):

| block | model f (physical) | model g (+ proxy) |
|---|---|---|
| a_airport | 0.64 | 0.8 |
| b_layout | 136.47 | 98.68 |
| c_time | 10.13 | 8.78 |
| d_congestion | 98.74 | 54.52 |
| e_aircraft | 62.92 | 42.36 |
| e2_delay_turn | 72.32 | 81.45 |
| f_weather | 24.69 | 13.29 |
| g_proxy | — | 152.42 |


### G.1 Proxy (MVT − AOBT_3) — mean SHAP s in model g

| where | [-1e+09,300) | [300,600) | [600,900) | [900,1200) | [1200,1500) | [1500,1800) | [1800,2400) | [2400,3600) | [3600,1e+09) |
|---|---|---|---|---|---|---|---|---|---|
| ALL | -19 | -224 | -162 | +14 | +142 | +280 | +553 | +994 | +934 |
| n (ALL) | 525 | 6,538 | 36,406 | 29,562 | 18,548 | 5,383 | 2,250 | 643 | 145 |


## G.2 Sensitivity: the physical model on rows where the two off-block clocks agree (|BLOCK − AOBT_3| ≤ 300 s)

Share of ordinary rows kept: 75.65% (per airport: EDDF 85.73%, EDDM 83.59%, EGLL 73.58%, EHAM 80.99%, LEBL 83.14%, LEMD 87.24%, LFPG 74.46%, LIRF 71.64%, LSZH 83.1%, LTFM 44.75%). Refit on them (same recipe/fold): held-out RMSE 186.293 s, R² 0.6862; the full model f on the same rows scores 204.079 s. Selection is on the clock disagreement, which is a function of y, so this is a sensitivity, not a population. SHAP on 30,000 of these held-out rows, both models.

| block | full model f mean \|SHAP\| s | clocks-agree model mean \|SHAP\| s |
|---|---|---|
| a_airport | 0.54 | 0.58 |
| b_layout | 134.24 | 148.7 |
| c_time | 9.18 | 6.89 |
| d_congestion | 94.29 | 66.47 |
| e_aircraft | 62.57 | 55.89 |
| e2_delay_turn | 61.03 | 31.44 |
| f_weather | 23.32 | 17.54 |

| model | where | delay [-1e+09,-300) s | delay [-300,0) s | delay [0,300) s | delay [300,900) s | delay [900,1800) s | delay [1800,3600) s | delay [3600,7200) s | delay [7200,1e+09) s |
|---|---|---|---|---|---|---|---|---|---|
| full f | ALL | -120 | -68 | -17 | +19 | +62 | +86 | +96 | +75 |
| full f | EDDF | -103 | -60 | -18 | +17 | +60 | +78 | +88 | +64 |
| full f | EDDM | -114 | -74 | -26 | +17 | +65 | +95 | +104 | — |
| full f | EGLL | -125 | -70 | -8 | +21 | +61 | +92 | +134 | +106 |
| full f | EHAM | -82 | -48 | -10 | +18 | +44 | +57 | +64 | +54 |
| full f | LEBL | -103 | -61 | -15 | +19 | +58 | +80 | +91 | +72 |
| full f | LEMD | -113 | -70 | -20 | +18 | +65 | +91 | +91 | +82 |
| full f | LFPG | -106 | -64 | -18 | +16 | +63 | +91 | +107 | +93 |
| full f | LIRF | -144 | -82 | -18 | +27 | +81 | +106 | +108 | +81 |
| full f | LSZH | -105 | -67 | -25 | +17 | +65 | +88 | +99 | — |
| full f | LTFM | -151 | -85 | -20 | +29 | +86 | +106 | +96 | +64 |
| clocks agree | ALL | -70 | -37 | +5 | +20 | +23 | +21 | +16 | +5 |
| clocks agree | EDDF | -63 | -36 | +1 | +19 | +24 | +22 | +17 | -1 |
| clocks agree | EDDM | -74 | -42 | +1 | +21 | +25 | +24 | +20 | — |
| clocks agree | EGLL | -78 | -38 | +14 | +24 | +24 | +20 | +13 | +5 |
| clocks agree | EHAM | -58 | -34 | +2 | +19 | +22 | +20 | +17 | +11 |
| clocks agree | LEBL | -65 | -35 | +4 | +19 | +21 | +20 | +16 | +8 |
| clocks agree | LEMD | -69 | -39 | +2 | +21 | +26 | +24 | +19 | +13 |
| clocks agree | LFPG | -62 | -30 | +8 | +18 | +19 | +17 | +12 | -1 |
| clocks agree | LIRF | -74 | -35 | +13 | +20 | +19 | +17 | +10 | -6 |
| clocks agree | LSZH | -68 | -41 | -2 | +21 | +27 | +26 | +22 | — |
| clocks agree | LTFM | -75 | -40 | +8 | +24 | +25 | +23 | +16 | +8 |


## H. Tails: ordinary taxis > 30 min — over-represented factors (lift)

lift = share of the level among >30 min rows ÷ share among all rows at that airport; levels with ≥ 20 tail rows and ≥ 0.5% of rows. Raw (confounded) co-occurrence. src: marginals (ordinary matched rows, 12 months 2025).

| airport | n | n > 30 min | rate |
|---|---|---|---|
| EDDF | 227,355 | 2,272 | 1.00% |
| EDDM | 164,316 | 2,076 | 1.26% |
| EGLL | 236,382 | 24,476 | 10.35% |
| EHAM | 243,757 | 2,227 | 0.91% |
| LEBL | 174,749 | 2,245 | 1.28% |
| LEMD | 209,478 | 3,369 | 1.61% |
| LFPG | 233,831 | 8,086 | 3.46% |
| LIRF | 146,341 | 7,559 | 5.17% |
| LSZH | 131,684 | 959 | 0.73% |
| LTFM | 267,221 | 11,009 | 4.12% |
| ALL | 2,035,114 | 64,278 | 3.16% |


### H. EDDF: top 12 lifts

| factor | level | n level | n tail | tail rate in level | lift |
|---|---|---|---|---|---|
| b_temp | <0C | 6,702 | 462 | 6.9% | 6.9 |
| b_vis | 1.5-5km | 2,014 | 128 | 6.4% | 6.36 |
| b_vis | <1.5km | 1,687 | 97 | 5.8% | 5.75 |
| b_rec_rwy | 20-25m | 4,691 | 261 | 5.6% | 5.57 |
| stand_pref | V3 | 2,788 | 152 | 5.5% | 5.46 |
| AIRCRAFT_TYPE_mvt | B744 | 2,223 | 102 | 4.6% | 4.59 |
| b_delay | late 1-2h | 6,951 | 297 | 4.3% | 4.28 |
| b_delay | late >2h | 2,513 | 96 | 3.8% | 3.82 |
| b_temp | 0-3C | 14,115 | 484 | 3.4% | 3.43 |
| b_delay | late 30-60m | 24,970 | 748 | 3.0% | 3.0 |
| month | 1 | 15,278 | 449 | 2.9% | 2.94 |
| b_arr_hr | 0-9 | 3,790 | 108 | 2.9% | 2.85 |


### H. EDDM: top 12 lifts

| factor | level | n level | n tail | tail rate in level | lift |
|---|---|---|---|---|---|
| b_freezing | freezing | 2,338 | 654 | 28.0% | 22.14 |
| b_rec_rwy | 25-30m | 1,059 | 281 | 26.5% | 21.0 |
| b_rec_rwy | 20-25m | 2,893 | 347 | 12.0% | 9.49 |
| b_vis | <1.5km | 3,942 | 393 | 10.0% | 7.89 |
| WK_TBL_CAT_flt | J | 2,272 | 195 | 8.6% | 6.79 |
| AIRCRAFT_TYPE_mvt | A388 | 2,272 | 195 | 8.6% | 6.79 |
| b_delay | late 1-2h | 4,566 | 381 | 8.3% | 6.6 |
| b_vis | 1.5-5km | 4,082 | 338 | 8.3% | 6.55 |
| b_temp | <0C | 12,917 | 954 | 7.4% | 5.85 |
| b_delay | late >2h | 1,017 | 68 | 6.7% | 5.29 |
| b_rec_rwy | 17.5-20m | 3,185 | 207 | 6.5% | 5.14 |
| b_delay | late 30-60m | 16,544 | 960 | 5.8% | 4.59 |


### H. EGLL: top 12 lifts

| factor | level | n level | n tail | tail rate in level | lift |
|---|---|---|---|---|---|
| b_rec_rwy | 30m+ | 5,523 | 1,549 | 28.1% | 2.71 |
| b_q_apt | 21+ | 15,316 | 3,965 | 25.9% | 2.5 |
| b_q_rwy | 21+ | 14,699 | 3,760 | 25.6% | 2.47 |
| b_arr_taxiing | 12+ | 1,186 | 292 | 24.6% | 2.38 |
| b_temp | <0C | 1,246 | 288 | 23.1% | 2.23 |
| b_delay | late 1-2h | 6,935 | 1,591 | 22.9% | 2.22 |
| b_delay | late 30-60m | 19,205 | 4,130 | 21.5% | 2.08 |
| WK_TBL_CAT_flt | J | 7,219 | 1,443 | 20.0% | 1.93 |
| AIRCRAFT_TYPE_mvt | A388 | 7,219 | 1,443 | 20.0% | 1.93 |
| stand_pref | 55 | 8,475 | 1,526 | 18.0% | 1.74 |
| b_temp | 0-3C | 7,023 | 1,257 | 17.9% | 1.73 |
| AIRCRAFT_TYPE_mvt | B78X | 3,702 | 646 | 17.4% | 1.69 |


### H. EHAM: top 12 lifts

| factor | level | n level | n tail | tail rate in level | lift |
|---|---|---|---|---|---|
| b_freezing | freezing | 1,381 | 359 | 26.0% | 28.45 |
| b_temp | <0C | 3,722 | 583 | 15.7% | 17.14 |
| b_q_rwy | 16-20 | 1,696 | 150 | 8.8% | 9.68 |
| b_vis | <1.5km | 2,990 | 248 | 8.3% | 9.08 |
| b_temp | 0-3C | 12,056 | 826 | 6.8% | 7.5 |
| AIRCRAFT_TYPE_mvt | B744 | 1,575 | 84 | 5.3% | 5.84 |
| stand_pref | S7 | 1,481 | 68 | 4.6% | 5.03 |
| airline | DL0 | 5,438 | 234 | 4.3% | 4.71 |
| MARKET_SEGMENT_flt | Cargo | 6,398 | 253 | 4.0% | 4.33 |
| b_dep_hr | 0-9 | 3,388 | 125 | 3.7% | 4.04 |
| stand_pref | S9 | 1,776 | 64 | 3.6% | 3.94 |
| month | 1 | 18,156 | 652 | 3.6% | 3.93 |


### H. LEBL: top 12 lifts

| factor | level | n level | n tail | tail rate in level | lift |
|---|---|---|---|---|---|
| b_thunder | thunder | 1,206 | 234 | 19.4% | 15.1 |
| b_rec_rwy | 25-30m | 941 | 149 | 15.8% | 12.33 |
| b_delay | late 1-2h | 7,909 | 500 | 6.3% | 4.92 |
| b_rec_rwy | 20-25m | 6,539 | 379 | 5.8% | 4.51 |
| b_delay | late >2h | 2,593 | 141 | 5.4% | 4.23 |
| RUNWAY_mvt | 06L | 978 | 53 | 5.4% | 4.22 |
| RUNWAY_mvt | 24R | 1,610 | 80 | 5.0% | 3.87 |
| AIRCRAFT_TYPE_mvt | A359 | 1,322 | 64 | 4.8% | 3.77 |
| b_delay | late 30-60m | 20,011 | 877 | 4.4% | 3.41 |
| stand_pref | 95 | 1,096 | 41 | 3.7% | 2.91 |
| b_q_rwy | 0 | 4,599 | 163 | 3.5% | 2.76 |
| AIRCRAFT_TYPE_mvt | A332 | 1,770 | 58 | 3.3% | 2.55 |


### H. LEMD: top 12 lifts

| factor | level | n level | n tail | tail rate in level | lift |
|---|---|---|---|---|---|
| airline | THY | 1,404 | 167 | 11.9% | 7.4 |
| b_temp | <0C | 2,333 | 259 | 11.1% | 6.9 |
| b_thunder | thunder | 1,337 | 120 | 9.0% | 5.58 |
| stand_pref | 72 | 1,406 | 116 | 8.2% | 5.13 |
| stand_pref | 70 | 1,333 | 107 | 8.0% | 4.99 |
| stand_pref | 74 | 1,082 | 81 | 7.5% | 4.65 |
| stand_pref | 73 | 1,367 | 96 | 7.0% | 4.37 |
| stand_pref | 11 | 1,575 | 106 | 6.7% | 4.18 |
| stand_pref | 71 | 1,387 | 89 | 6.4% | 3.99 |
| WK_TBL_CAT_flt | L | 1,236 | 79 | 6.4% | 3.97 |
| MARKET_SEGMENT_flt | Business | 8,371 | 531 | 6.3% | 3.94 |
| stand_pref | 10 | 1,591 | 100 | 6.3% | 3.91 |


### H. LFPG: top 12 lifts

| factor | level | n level | n tail | tail rate in level | lift |
|---|---|---|---|---|---|
| b_rec_rwy | 30m+ | 1,957 | 888 | 45.4% | 13.12 |
| b_temp | <0C | 4,051 | 1,356 | 33.5% | 9.68 |
| b_vis | <1.5km | 3,143 | 971 | 30.9% | 8.93 |
| b_rec_rwy | 25-30m | 3,647 | 778 | 21.3% | 6.17 |
| b_q_apt | 21+ | 4,189 | 678 | 16.2% | 4.68 |
| stand_pref | I5 | 1,293 | 204 | 15.8% | 4.56 |
| b_temp | 0-3C | 12,396 | 1,820 | 14.7% | 4.25 |
| WK_TBL_CAT_flt | J | 1,614 | 197 | 12.2% | 3.53 |
| AIRCRAFT_TYPE_mvt | A388 | 1,615 | 197 | 12.2% | 3.53 |
| AIRCRAFT_TYPE_mvt | B77L | 4,571 | 520 | 11.4% | 3.29 |
| airline | FDX | 4,440 | 489 | 11.0% | 3.18 |
| b_q_rwy | 12-15 | 6,525 | 692 | 10.6% | 3.07 |


### H. LIRF: top 12 lifts

| factor | level | n level | n tail | tail rate in level | lift |
|---|---|---|---|---|---|
| airline | UAL | 1,622 | 674 | 41.5% | 8.04 |
| airline | DAL | 1,840 | 711 | 38.6% | 7.48 |
| b_rec_rwy | 30m+ | 1,588 | 585 | 36.8% | 7.13 |
| AIRCRAFT_TYPE_mvt | A333 | 936 | 274 | 29.3% | 5.67 |
| AIRCRAFT_TYPE_mvt | B772 | 1,029 | 239 | 23.2% | 4.5 |
| AIRCRAFT_TYPE_mvt | A339 | 3,637 | 818 | 22.5% | 4.35 |
| stand_pref | 90 | 1,754 | 363 | 20.7% | 4.01 |
| airline | BAW | 2,599 | 524 | 20.2% | 3.9 |
| b_rec_rwy | 25-30m | 4,936 | 978 | 19.8% | 3.84 |
| WK_TBL_CAT_flt | H | 19,129 | 3,171 | 16.6% | 3.21 |
| RUNWAY_mvt | 16R | 10,350 | 1,676 | 16.2% | 3.13 |
| AIRCRAFT_TYPE_mvt | B788 | 1,340 | 202 | 15.1% | 2.92 |


### H. LSZH: top 12 lifts

| factor | level | n level | n tail | tail rate in level | lift |
|---|---|---|---|---|---|
| b_freezing | freezing | 1,017 | 194 | 19.1% | 26.19 |
| b_rec_rwy | 20-25m | 1,219 | 94 | 7.7% | 10.59 |
| b_thunder | thunder | 710 | 47 | 6.6% | 9.09 |
| b_vis | <1.5km | 3,118 | 176 | 5.6% | 7.75 |
| b_temp | <0C | 6,679 | 369 | 5.5% | 7.59 |
| b_q_rwy | 8-11 | 1,124 | 62 | 5.5% | 7.57 |
| AIRCRAFT_TYPE_mvt | A343 | 2,150 | 87 | 4.0% | 5.56 |
| airline | UAL | 1,113 | 40 | 3.6% | 4.93 |
| b_rec_rwy | 17.5-20m | 3,404 | 120 | 3.5% | 4.84 |
| AIRCRAFT_TYPE_mvt | B763 | 1,202 | 39 | 3.2% | 4.46 |
| b_q_apt | 8-11 | 4,338 | 124 | 2.9% | 3.93 |
| RUNWAY_mvt | 34 | 1,367 | 37 | 2.7% | 3.72 |


### H. LTFM: top 12 lifts

| factor | level | n level | n tail | tail rate in level | lift |
|---|---|---|---|---|---|
| b_rec_rwy | 30m+ | 4,369 | 2,027 | 46.4% | 11.26 |
| b_freezing | freezing | 2,589 | 1,155 | 44.6% | 10.83 |
| RUNWAY_mvt | 16R | 1,405 | 438 | 31.2% | 7.57 |
| b_temp | 0-3C | 4,285 | 1,084 | 25.3% | 6.14 |
| RUNWAY_mvt | 34L | 8,283 | 1,675 | 20.2% | 4.91 |
| b_rec_rwy | 25-30m | 6,873 | 1,373 | 20.0% | 4.85 |
| b_delay | late 30-60m | 16,465 | 2,511 | 15.2% | 3.7 |
| b_delay | late 15-30m | 26,988 | 3,793 | 14.1% | 3.41 |
| month | 2 | 17,781 | 2,179 | 12.2% | 2.97 |
| b_delay | late 1-2h | 7,141 | 859 | 12.0% | 2.92 |
| b_q_rwy | 16-20 | 4,559 | 546 | 12.0% | 2.91 |
| RUNWAY_mvt | 17R | 20,374 | 2,325 | 11.4% | 2.77 |


### H. ALL: top 12 lifts

| factor | level | n level | n tail | tail rate in level | lift |
|---|---|---|---|---|---|
| b_rec_rwy | 30m+ | 15,115 | 5,654 | 37.4% | 11.84 |
| b_q_rwy | 21+ | 14,866 | 3,805 | 25.6% | 8.1 |
| b_rec_rwy | 25-30m | 60,966 | 10,797 | 17.7% | 5.61 |
| b_q_rwy | 16-20 | 94,119 | 13,649 | 14.5% | 4.59 |
| WK_TBL_CAT_flt | J | 14,452 | 2,036 | 14.1% | 4.46 |
| AIRCRAFT_TYPE_mvt | A388 | 14,453 | 2,036 | 14.1% | 4.46 |
| airline | DAL | 11,627 | 1,524 | 13.1% | 4.15 |
| airline | BAW | 115,152 | 14,859 | 12.9% | 4.09 |
| b_vis | <1.5km | 19,225 | 2,402 | 12.5% | 3.96 |
| airline | UAL | 14,807 | 1,830 | 12.4% | 3.91 |
| b_temp | <0C | 38,224 | 4,671 | 12.2% | 3.87 |
| AIRCRAFT_TYPE_mvt | A35K | 10,336 | 1,222 | 11.8% | 3.74 |

---
**Note (2026-09-11 02:27):** the atlas's weather-code tables used its own parser (`data/atlas/code/wxcodes.py`, now marked RETIRED),
whose fog definition excludes MI/BC/PR. The project parser is `prc/weather.py` v2.0.1 (commit 151d7ae; data
`data/weather_obs/weather_obs_v2.0.1.parquet`, sha256 cb05439b…22a8), which counts fog of any kind at any visibility.
Fog/visibility figures in this atlas may therefore differ slightly from anything built on prc.weather; all new work
uses prc.weather v2.0.1.
