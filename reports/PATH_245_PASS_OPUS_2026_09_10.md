# Path to 245: an outside pass on rules, other teams and external data (2026-09-10, afternoon)

**Lens:** what other teams do and what the competition allows. I ran the external search first
and read `reports/MSE_LEDGER.md` and `reports/PATH_TO_245_FABLE_2026_09_10.md` afterwards.
Nothing was fitted or submitted, and no labels were learned from the board. The only leaderboard
data used is the public API listing. Local compute stayed read-only and ran under two minutes per
step at `nice -n 19`.
Scratch scripts are in
`/private/tmp/claude-501/-Users-saurabhdxt-Projects-prc-challenge/81b461ea-ab3b-47e3-926a-3198c439b0c1/scratchpad/opus/`
(`witness.py`, `witness2.py`, `witness_lomo.py`, `price2026.py`, `fetch_lb.py`, `lb_summ.py`).

**Bottom line.**
1. None of the top six teams has published code or a write-up. I could not find the leader's
   method.
2. The main finding comes from the data itself. The external search is what pointed me to it. **Our
   shipped unmatched estimator uses no information about the airport-hour it departs in.** In the
   same file, the matched departures' `MVT − AOBT_3` tells us how long taxi-outs were taking that
   hour. Out of sample on 2025, that signal shows the estimator is **biased low by +404 / +1,947 /
   +3,140 s** when the same-hour matched median is 1.5–1.8 / 1.8–2.4 / >2.4 ks.
3. The scored file contains the largest disruption in the whole dataset, the Schiphol snow and
   de-icing crisis of 3–7 Jan 2026 (confirmed by press). v7 predicts a median of 775 s on its 401
   unmatched EHAM rows. **Priced at 2,500–7,100 board MSE, central ≈ 4,000.** It needs no
   external data. Adding ADS-B at EHAM for those 5 days (adsb.lol, ODbL, archives exist) is the
   best-priced external source.

---

## 1. Rules that matter (every rule with its source)

| rule | text / fact | source |
|---|---|---|
| Scored rows | Every submission is scored on **all 344,841 departures** (`usedPairs = 344841` on all 765 submissions). Ranked on the **best** RMSE per team | [leaderboard API](https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard), [ranking page](https://prc-data-challenge-2026.netlify.app/ranking.html) |
| Private/final split | **None stated for 2026.** Precedent says one may come: 2024 kept "an additional 52190 [flights] … for the final prize ranking"; 2025 released a final set in phase 2 "to avoid the possibility of *leaning* from the leaderboard". The 2026 site: "We reserve the right to change the rules" | [2024](https://prc-data-challenge.netlify.app/), [2025 ranking](https://prc-data-challenge-2025.netlify.app/ranking.html), [2026 home](https://prc-data-challenge-2026.netlify.app/) |
| Submissions | "limit of 5 submissions per day (and a total max of 1GB per bucket)" | [ranking.qmd](https://github.com/euctrl-pru/prc_data_challenge_website_2026/blob/main/ranking.qmd) |
| Board use | "We will monitor submissions for attempts to learn from or exploit the ranking process" | same |
| External data | Prize requires "All used external datasets are openly accessible/usable and documented" and "additional datasets … openly available under an open source license" | [eligibility](https://prc-data-challenge-2026.netlify.app/eligibility.html) |
| Code | Source on GitHub under **GPLv3**, reproducible docs. Re-use is allowed only with the authors' rights **and** "significant modifications" | same |
| Deadline | 11 Oct 2026 23:59:59 CET | [home](https://prc-data-challenge-2026.netlify.app/) |
| Ranking data | Same columns as training. Only `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are blanked, and only for departures. `AOBT_3_flt`, `LOBT_flt` and **arrival labels stay in** | [data page](https://prc-data-challenge-2026.netlify.app/data.html) |

**Verified versus interpreted.** kind-mango does not use `AOBT_3_flt` and quotes "the brief" as
saying that using it "is probably exploiting the ranking process". That sentence comes from
**their own brief file**
([docs/PRC_Data_Challenge_2026_BRIEF.md:178](https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango/blob/main/docs/PRC_Data_Challenge_2026_BRIEF.md)).
The organizers never wrote it. No organizer text forbids the field. **Concern to raise before it
bites:** the whole matched lane depends on it. A one-line confirmation on the organizers' Discord
is cheap insurance, and so is asking whether a final set is coming.

**Rule risk in our own record.** `GAP_LOCATED` §2 solves kind-mango's board delta for a hidden
label (≈ 84,200 s). That is learning a label from the ranking process, only through another team's
numbers. v5 came before it, so nothing we shipped depends on it. It should not be repeated, and no
prediction should ever be set from such an inference.

## 2. Other teams (public board as of this pass: 105 teams, 765 submissions)

| rank · team · best | public material | key idea | do we have it? |
|---|---|---|---|
| 1 youthful-giraffe 245.02 (solo, Indonesia) | **none found** (GitHub, web). [Team page](https://github.com/euctrl-pru/prc_data_challenge_website_2026/blob/main/teams/youthful-giraffe.qmd) | 265.76 on day one. **262.74 → 248.48 in one submission, 09-08 17:21Z (−7,290 MSE)**, then −1,700 of polish | — |
| 2–6 enthusiastic-daisy 260.93, jovial-uniform 264.25, quick-boat 264.33 (A. Canedo, who wrote the board viewer [arquicanedo/prc-challenge-2026](https://github.com/arquicanedo/prc-challenge-2026)), upstanding-firefly 266.40 (states an OpenSky ADS-B background and promises GitHub + JOAS), zesty-puzzle 266.96 | none found | — | — |
| 7 elegant-alligator 271.04 | [javidmardanov/PRC-Data-Challenge-2026](https://github.com/javidmardanov/PRC-Data-Challenge-2026) (claims verified against the API) | TimesFM-3 **hourly** airport forecasts whose covariates include the **hourly mean `MVT−AOBT_3` proxy** and arrival taxi mean ([timesfm_notes](https://github.com/javidmardanov/PRC-Data-Challenge-2026/blob/main/docs/timesfm_notes.md)). Also a CatBoost tail, a missing-timestamp expert, symmetric proxy-window queue features, and IEM METAR | Rome ideas are read. **An hour-level proxy witness in the unmatched lane: no** |
| 20 zestful-fountain 285.75 | [Phoenix-Ops-LTD/prc2026-taxiout](https://github.com/Phoenix-Ops-LTD/prc2026-taxiout) | CatBoost, carrier/specialist blends, a LIRF specialist | nothing new |
| 22 likable-eagle 287.71 | [ChandanHegde07/OpenAir](https://github.com/ChandanHegde07/OpenAir) | two clocks, residual LGB, a "disruption state" (other flights' AOBT−EOBT over the last 30 min), LIRF `T = D − G` | mostly yes |
| 39 jolly-lobster 300.35 | [LeeMarshall1113/prc-taxiout-2026](https://github.com/LeeMarshall1113/prc-taxiout-2026) | — | — |
| 40 kind-mango 303.71 | [skylinkapi/…-kind-mango](https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango) | Leaves out `AOBT_3`. Uses `linear_tree` (+10.5 s on *their* model), OPDI "live taxi of other flights", Eurocontrol daily ATFM, OSM paths, METAR, and a one-row ITY340 rule | OPDI, ATFM, `linear_tree`: no. The transfer prior is low: their base lacks `AOBT_3` |

Also found, at non-competitive scores or unscored: ahmetabdullahgultekin, col3name/air-data,
fenil264 (518), victoralcadi, ChandanHegde07, satam2, georgejhanlon.

**Pattern.** The public teams that use context use it at the **hour level, from other departures'
measured durations**: elegant-alligator's hourly proxy, kind-mango's live OPDI taxi, likable-eagle's
disruption state. Our matched model has it (`SURF`, queue block). **Our unmatched lane does not.**
`scripts/build_submission.py:66` sets `NF_NUMERIC = ["sp","dayoff","hr"]`, plus encodings of
airport, runway, stand prefix, airline and operator.

## 3. The finding: the unmatched lane cannot see a disrupted hour

**Hypothesis.** The shipped unmatched estimator (S1) is biased low in airport-hours where the median
`MVT − AOBT_3` of matched departures (`hprox`, serve-time, from the same file) is high.
- **TRUE shape:** the out-of-sample mean residual rises with `hprox` and is positive in most
  airport-months.
- **FALSE shape:** the residual is flat near 0.

**Evidence** (2025 LOMO predictions in `data/cache_stand/stratum_fold_v7_preds.parquet`, ex-LIRF):

| `hprox` bucket | n | mean y | S1 mean | **bias** | bias² share of SSE |
|---|---:|---:|---:|---:|---:|
| ≤ 900 | 8,968 | 843 | 838 | +5 | 0% |
| 900–1,200 | 7,925 | 975 | 995 | −19 | 0% |
| 1,200–1,500 | 2,668 | 1,242 | 1,196 | +46 | 1% |
| 1,500–1,800 | 666 | 1,805 | 1,402 | **+404** | 11% |
| 1,800–2,400 | 338 | 3,210 | 1,263 | **+1,947** | 48% |
| > 2,400 | 99 | 4,434 | 1,293 | **+3,140** | 76% |

The bias is positive in **19 of 20** airport×month cells with n ≥ 10 and `hprox` > 1,500: EGLL
10/11, EHAM Jan/Dec, LFPG 5/5, LTFM Feb (+3,806) and Oct. **Verdict: WORKING** that the bias exists
in 2025. This is in-project data, but out of sample relative to the estimator. A crude pooled
residual slope is **not** uniformly good: LOMO gives +964 to +1,089 M SSE, carried by LTFM
February, while EGLL comes out negative. So the fix needs a proper form: features in the non-fill
body, or per-airport ratios, pre-registered.

**Why it matters for 2026 (serve-time only, no scored label read).**
- EHAM's matched-proxy daily median ran **1,998 / 2,352 / 2,110 s on 3 / 4 / 5 Jan 2026**, with p75
  up to 4,566. EHAM's worst 2025 day was **1,204**.
- **401 unmatched EHAM departures on 3–7 Jan** (26–37% of departures on those days) get a v7
  median of **759–788 s**.
- Press confirms a de-icing-fluid crisis with 3,200+ cancellations
  ([Flightradar24](https://www.flightradar24.com/blog/aviation-news/airport-news/snow-chaos-at-schiphol/),
  [NL Times](https://nltimes.nl/2026/01/06/de-icing-operations-threat-schiphol-airport-585-flights-cancelled-325-delayed)).
- 2026 ex-LIRF unmatched rows with `hprox` > 1,500: **542** (EHAM 257, LFPG 103, EGLL 71, LTFM 51).
  Bucket counts are 248 / 80 / 214, and v7's bucket means (1,233 / 1,164 / 1,000) sit where S1's did.

| scenario for the true conditional mean | board MSE at stake |
|---|---:|
| per-airport 2025 ratios y/`hprox` (EHAM 0.67, LFPG 0.98, LTFM 1.98, …), only raising | **2,571** |
| pooled y = `hprox` | 4,769 |
| 2025 LOMO bucket bias carries over to 2026 | **7,116** |

**Open:** whether it transfers. The 2026 EHAM episode lies outside EHAM's 2025 support. Fable's E3
found the same event through arrival taxi-in and priced it at 1–2k. The departure-side witness is
more direct and prices higher. The size is consistent with the leader's 7,290 step, but that is
**not evidence** about their method.

## 4. External data, priced

| source | status (probed lightly) | verdict |
|---|---|---|
| **adsb.lol globe_history 2026** (ODbL-1.0) | Daily archives exist for **2–9 Jan 2026**, about 3.0 GB each (`v2026.01.0x-planes-readsb-prod-0`). The project already has `scripts/adsb_extract.py` and the stand-gate method. The EHAM 2025 census gave 67–77% coverage and RMSE 127–150 (`reports/ADSB_GATE.md`). The unmatched rows have no callsign, but **runway + take-off second is unique on 99.86%** of EHAM runway-minutes | **Best option.** It measures pushback on about two-thirds of the 401 disrupted EHAM rows. EHAM has 0% fills, so the physics predicts the label directly. It independently closes the transfer question in §3, and it captures per-row spread (matched p25–p75 952–4,455 s on 4 Jan) on top of the mean shift. **Incremental +1,000–3,000 MSE over §3's mean fix.** Cost: ~15 GB streamed and ~2 h ingest (disk is 94% full, peak 3.5 GiB), plus a day of work. Licence: ODbL is documented and the share-alike terms apply to the derived table |
| OPDI flight_events v002 (PRC/OSN) | The `20251231_20260110` chunk is **40.7 MB and holds only 31 Dec and 1 Jan**, so **2–10 Jan 2026 is missing**. It has zero `exit-parking_position` events at EHAM. July 2026 chunks exist | useless for the event. Matches the project's 0.7% finding |
| OpenSky historical DB | restrictive terms (`ADSB_GATE.md`) | avoid |
| IEM METAR, Eurocontrol daily ATFM | public domain / open. Used by elegant-alligator and kind-mango | at most a few hundred MSE (Arm W's prior) |

## 5. What the ledger and the Fable review miss

1. **The unmatched lane has no hour context.** Both documents treat the matched lane as the only
   one with surface state. Fable's E3 used arrival taxi-in, the weaker witness.
2. **OPDI's January gap.** Any plan that relies on OPDI for 2026 January fails silently.
3. **The final-set precedent.** Row-level Rome bets such as E1, sized to two 2026 rows, pay nothing
   if a final set is added. Weight general mechanisms above row bets.
4. **The label-inference line in `GAP_LOCATED` §2** (see §1).

## 6. Ranked moves toward 245 (expected board MSE × feasibility before 11 Oct)

| # | move | expected MSE | feasibility | notes |
|---|---|---:|---|---|
| 1 | **Hour witness in the unmatched non-fill body**: same airport-hour (±30 min, same runway) matched-proxy median, p75, count, ratio to airport norm. Pre-register it, run LOMO and fold A, exclude LIRF | **2,500–7,000 (central ≈ 4,000)** | high: a stratum refit of about 0.5 h, no new data | largest evidence-backed lever found |
| 2 | Fable E1, the LIRF band floor | +5,600 (two-row bet) | high | final-set risk |
| 3 | ADS-B at EHAM, 3–7 Jan, stand gate joined on runway + take-off second | +1,000–3,000 on top of #1 | medium | independent closure of #1 |
| 4 | Fable E2, arm B continuous | +1,500–3,500 | medium | |
| 5 | Ask the organizers: is `AOBT_3` allowed? Will there be a final set? | protects everything | trivial | |

Even the optimistic sum (≈ 12–19k) leaves 245 short without further luck. Top 5 (266.4) becomes
realistic. **245: still unexplained.** No public source reveals the leader's method.

**Watch item for the owner:** the machine was running on battery (81%) during this pass. Project
memory says to check for AC power before long runs.
