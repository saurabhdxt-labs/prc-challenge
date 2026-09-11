# Pre-registration — arm E5: a physics-informed unmatched body (weather done right + congestion + stand + aircraft)

**Written 2026-09-11 05:17:09 (from `date`), BEFORE the harness exists and before any number of
this arm exists.** Tag `E5`. Owner: "end to end", "keep going", target Position 1. Evidence base:
`reports/TAXI_UNDERSTANDING_SYNTHESIS_2026_09_11.md` (atlas + importance + literature).

## Why

The unmatched lane (no NM record, no clock proxy) is ≈ 28–30k of v10's ≈ 79k board MSE. Its body (S1C, shipped in v9/v10)
sees only `sp, dayoff, hr`, the airport-hour witness (`hprox_med, hprox_n`) and encodings of ADEP / RUNWAY / stand_pref /
airline (AIRCRAFT_OPERATOR is 100% null there — dead). The atlas shows physical factors explain ≈ 59% of ordinary taxi
variance WITHOUT the proxy: runway (up to +326 s), stand area (up to 481 s range), freezing / de-icing (+398 s; tail lift
11–28×), visibility < 1.5 km (+257 s), runway queue (≈ +20 s per aircraft), recent runway taxi (+12 s per minute),
aircraft size / type (Heavy +64–98 s, A380 +149 s), airline × airport. Weather was never available correctly before
(arm W's precipitation was a constant 0; fixed as `prc/weather.py` v2.0.1).

## Hypothesis

**H-E5:** adding take-off-anchored physical inputs to the unmatched body lowers non-Rome unmatched error out of month.

## Arms (12-month LOMO over 2025, E3C's harness folds; scored on non-LIRF unmatched rows; LIRF report-only)

- **S1C** — control, exactly E3C's (reproduction guard: equals `data/cache_stand/unm_congestion_lomo.parquet` `S1C` to 1e-6 s).
- **E5w (reported, attribution)** — S1C's body + WEATHER only.
- **E5 (decisional)** — S1C's body + WEATHER + PHYSICS, all take-off-anchored and serve-time:
  - WEATHER: as-of join to `data/weather_obs/weather_obs_v2.0.1.parquet` (station == ADEP_mvt), latest observation with
    `valid <= MVT_TIME_UTC`, NaN beyond 3 h; inputs `temp_c, dewspread_c, vis_km, wind_kt, gust_kt, wx_precip, wx_frozen,
    wx_sn, wx_fzra, wx_fzdz, wx_fzfg, wx_fg, wx_br, wx_intensity, deicing_condition`, and `deicing_condition ×
    de-icing-airport` where the de-icing airports are EHAM, EDDM, EDDF, LSZH (fixed now from the atlas / literature).
  - PHYSICS: the take-off-anchored queue block available for unmatched rows (`stand_ab.UNMATCHED_QUEUE_FEATS`:
    `q_dep_tko_sym15, q_rwy_tko_sym10, q_arr_taxiin_sym30_tko, q_rwy_ambient_proxy`); stand-history features anchored at
    take-off where they exist for unmatched rows (`prev_arr_gap, prev_arr_taxiin, prev_dep_gap` — only if computed
    without AOBT_3); aircraft type (`AIRCRAFT_TYPE_mvt`) and airline × airport as smoothed target encodings fitted on the
    training fold's non-fill rows only (BC-1-clean: never on the stopping rows), same smoothing as S1C.
  - Same `p`, `nf_cells`, seeds, params and winsorisation as S1C; `nf = nf_hybrid(nf_cells, mean_seeds(nf_fit_E5))`.
  - Any input the harness cannot build for unmatched rows is DROPPED (listed in the report), never imputed from matched rows.

## Clauses, locked (non-LIRF LOMO rows; `g = SE(S1C) − SE(E5)`, raw labels)

1. **C1** — date-block bootstrap (calendar dates, 2,000 draws, seed 0) of Σg: 95% lower bound > 0.
2. **C2** — event-excluded 2026 price ≥ **+300** board MSE (drop the one airport-month with the largest |Σg|; E3C's bins,
   borrowing rule and 2026 witness bin counts ÷ 344,841). Full price reported.
3. **C3** — Σg > 0 in ≥ 8 of 12 months.
4. **C4** — Σg > 0 at ≥ 6 of 9 non-LIRF airports.
5. **C5 (calm control)** — Σg over rows with `hprox_med < 1,100` ≥ −10% of pooled Σg.
6. **C6 (not the monsters)** — Σg over rows with `y ≤ 10,800` > 0.
7. **C7 (seeds)** — pooled gain > 2 × sd of the three single-seed gains.
**Verdict:** WORKING iff all seven pass; NOT WORKING iff C1 fails; else INCONCLUSIVE. Never revised.
**Registered sensitivities (reported, not decisional):** the fog clause at any visibility vs ≤ 1.5 km; `<= +3 °C` vs
`< +3 °C` in the de-icing rule; E5w vs E5 (how much is weather).
**Ship rule:** only if WORKING — v_next = v10 with non-LIRF unmatched rows replaced by E5 (all twelve 2025 months, 2026
features built identically), rebuild guard on v10, board Δ vs v10 read against the event-excluded price (conservative) and
the full price (optimistic).

## Predicted shapes

- **TRUE:** Σg > 0 broadly (most months and airports), a larger share from winter months / de-icing airports and from
  congested hours; E5w carries a visible part in months with freezing conditions.
- **FALSE:** gains confined to one event or one airport; calm-hour losses; or ≈ 0 because the unmatched rows' error is
  dominated by record classes (fills, date-slips, monsters) that physics cannot predict.

## Limits

- The unmatched rows' taxi is the AIRPORT's recorded taxi; physics that works on matched ordinary rows may not transfer to
  records without an NM plan (charter, general aviation, military mixes). This arm measures, it does not assume.
- 2026 January's EHAM storm is beyond 2025's support; the transfer risk remains, priced conservatively.

## AMENDMENT E5.1 · 2026-09-11 06:03:19 (from `date`) — judgement calls, BEFORE the real LOMO run (no real number exists)

Harness `scripts/unm_physics.py` (tests `tests/test_unm_physics.py` 38; 136 across the related suites; 72 mutations rehearsed,
the one survivor answered). **All 25 registered inputs were built for unmatched rows; none dropped.** None moves a threshold:
1. `prev_dep_gap` is kept: on an unmatched row it is take-off minus ANOTHER flight's NM off-block at the stand — it never
   reads the row's own AOBT_3. Negative values (no earlier departure from the stand that month) become NaN.
2. `deicing_condition × de-icing-airport` is a three-valued AND: 0 at non-de-icing airports even when weather is unknown.
3. Temperature thresholds compare at 1e-6 °C (a reported +3 °C is stored as 2.999999999999999; without this the
   `< +3 °C` sensitivity would equal `<= +3 °C`).
4. The two new encodings exclude the regressor's internal early-stopping rows (split copied from sklearn 1.8.0, version pinned).
5. `ship` refuses any base other than v10.
**Named before measuring:** (a) `prc/weather.py` v2.0.1 float edge — a 3 °C dew-point spread stored as 3.0000000000000036
turns frost off on 283 observations (4 rows in 2025, 20 scored 2026 rows); E5 uses the table as registered. (b) **Transfer
risk:** `deicing_condition` = 1 on **21% of 2026 scored rows vs 4.8% of 2025 rows** — the weather effect is learned on few
rows and applied to many; the conservative (event-excluded) price is the realistic expectation.

## RESULT E5 · 2026-09-11 06:05:50 (from `date`) — **INCONCLUSIVE** (C2 fails; C1, C3–C7 pass)

Source: `reports/unm_physics.json` (`score`, exit 0); predictions `data/cache_stand/unm_physics_lomo.parquet` (`lomo`, exit 0,
89 s, peak 3.06 GB footprint). Reproduction guard: S1C, S1, the per-seed S1C arms, p_hat, nf_cells and nf_fit_c reproduced
E3C's record on all 22,219 rows, max |diff| 0.0. **All 25 inputs built; none dropped.** Non-LIRF rows 20,731.

| clause | value | bar | pass |
|---|---|---|---|
| C1 date-block bootstrap (365 dates) | Σg +4.50e8, 95% [+2.10e8, +8.23e8] | lower > 0 | ✓ |
| C2 event-excluded 2026 price (dropped LTFM-02, +1.84e8 on 364 rows) | **+207.8** (full, reported: +307.7) | ≥ 300 | **✗** |
| C3 months | 12 of 12 | ≥ 8 | ✓ |
| C4 airports | 8 of 9 | ≥ 6 | ✓ |
| C5 calm (< 1,100) | +2.03e8 | ≥ −4.5e7 | ✓ |
| C6 ex-monster | +4.06e8 (RMSE 581.53 → 564.42, +17.1 s) | > 0 | ✓ |
| C7 seeds | +4.50e8 vs 2 × sd 1.41e8 | > | ✓ |

RMSE pooled S1C 1,104.75 → E5 1,094.87 (+9.87 s). **Not shipped under this registration** (ship rule needs WORKING).
Reported, not decisional: weather is only **8.6%** of the gain (E5w alone: Σg +3.86e7, C1 interval spans zero, event-excluded
−162 — would read NOT WORKING); the physics increment (queue, stand history, aircraft type, airline × airport) carries the
rest. Sensitivities: fog ≤ 1.5 km → event-excluded +159.8; `< +3 °C` → +183.9 (both would also read INCONCLUSIVE).
A real, broad, small gain — every month, 8 of 9 airports — whose 2026 value sits at +208 (conservative) to +308 (full).

*Correction to RESULT E5's reported (non-decisional) description · 2026-09-11 06:41:41 (from `date`), after an independent review; the
verdict INCONCLUSIVE is unchanged:* "a real, broad, small gain" overstated it. **One airport-month (LTFM-02, 364 rows)
carries 41% of Σg, and the NaN-witness bin (333 rows) 30%.** The event-excluded +208 includes +34 from a 33-row ≥ 2,400
bin that prices at −36 in the full table. **The predicted TRUE shape did NOT appear:** the gain is not carried by weather
/ de-icing (weather is 8.6%), and the RESULT did not compare against the pre-written shapes — it should have. Also: the
peak RSS was 3.40 GB (log), not the 3.06 GB footprint quoted; `prev_dep_gap` reads OTHER flights' AOBT_3 (kept by E5.1)
and no ablation isolates the physics block, so "the physics increment carries the rest" is subtraction, not measurement;
the record's git sha (151d7ae, dirty) does not contain the untracked harness that ran — provenance is the file, not the sha,
until the harness is committed. Weather 2.0.2 (BC-4) replaced the default table after this RESULT; E5 stays on 2.0.1.

*Test-base note · 2026-09-11 07:45:29 (from `date`), session prc-challenge-c4 — the verdict INCONCLUSIVE is unchanged and no
number moves.* The independent review's two weak E5 tests were strengthened; `scripts/unm_physics.py` is byte-identical to
the file that produced RESULT E5 (sha256 c469de86…92b5). (1) `test_holdout_labels_never_reach_any_arm` now perturbs every
label-bearing column of the holdout rows — `y`, `TAXITIME_SEC_mvt`, `BLOCK_TIME_UTC_mvt` (a third of the rows put on their
schedule so the fill flag flips) — separately and together (`delta` is NaN on every unmatched row); a fill-oracle mutation
reading holdout BLOCK goes RED under it and SURVIVED the old y-only test. (2) The planted de-icing test now plants on the
TRUE take-off weather from an independent test-side as-of join, asserts the harness's joined condition equals it on every
row and that the world tells a backward join from a forward one (> 5% of rows differ); an anticipating join (take-off + 1 h)
and a one-row shift of the joined weather go RED under it and SURVIVED the old test. Both pass on the unchanged harness,
so RESULT E5 was not produced by either defect.
