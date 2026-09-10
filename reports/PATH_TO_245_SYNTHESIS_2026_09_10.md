# The path to 245 — synthesis of five outside passes (2026-09-10, afternoon)

Inputs (each an independent pass; light compute; nothing fitted or submitted):
`PATH_TO_245_FABLE_2026_09_10.md` (Fable, first pass) · `PATH_245_PASS_HAIKU_…` · `PATH_245_PASS_SONNET_…` ·
`PATH_245_PASS_OPUS_…` · `PATH_245_PASS_FABLE_…` (second pass). Every load-bearing claim below was re-checked
in the main session; the check is named.

## The answer in one paragraph

Nobody found a verified 21,657-MSE path, and no top-six team has published its method (Opus; the leader,
youthful-giraffe, jumped 262.74 → 248.48 in ONE submission on 09-08, −7,290 MSE). What the passes did find
is a **real, independently confirmed structural defect that sits in exactly the lane where a jump of that
size is possible**: the shipped unmatched body S1 has **no congestion input** (`build_submission.py:66–67`:
`sp, dayoff, hr` + five identity encodings), it runs **hundreds to thousands of seconds low in congested
airport-hours**, and January 2026 contains the **EHAM snow / de-icing disruption (3–7 Jan)** — hour medians of
matched `MVT − AOBT_3` of 2,000–2,350 s against EHAM's 2025 worst day of 1,204 — where v7 predicts a median
≈ 775 s on ~400 unmatched EHAM rows. Priced on the 2026 file's own composition, that lane is worth
**≈ +2.5 to +7k** (central ≈ 4k); ADS-B for those days could add **+1–3k** and independently test it. With E1
(v8) and the smaller lanes, the evidence-backed total is **≈ 12–15k → board ≈ 262–267 (top 5)**. 245 needs
the upper end of every lane plus something not yet found.

## Verified findings, ranked

| # | lane | evidence | verified here | expected board MSE | cost |
|---|---|---|---|---|---|
| 1 | **E3′ congestion-aware unmatched body (ex-LIRF)** | Fable-2: fold A, gated `S1 + 0.75·(m_h − S1)`, **+509 [210, 896]**, 8/9 airports, calm-hour control passes; arm U (has queue block) +619 independently. Opus: 12-month LOMO, S1 bias +404 / +1,947 / +3,140 s in hours with matched `MVT−AOBT_3` median 1.5–1.8k / 1.8–2.4k / >2.4k; positive in 19/20 airport-months. 2026 exposure ≈ 3× 2025's (1,242 non-Rome unmatched rows in hot hours; bias² stake 4,743–8,958) | **Yes** — bias table reproduced with an independent join (hour from raw MVT on both sides): S1 790→1,322 vs y 832→2,395 across the congestion bins; S1 feature list read from code | +2.5k to +7k (central ≈ 4k); P(transfers) ≈ 0.75 — 2026 EHAM is beyond 2025's support | ≈ 0.5 h refit + LOMO harness |
| 2 | **E1 / v8 — LIRF schedule floor** | 2025: 0 of 56 LIRF unmatched rows with `sp ≥ 24,000` had `y < sp − 60` | **Yes** (`plans/PREREG_rome_bandfloor_2026_09_10.md`) | +5,627 if both rows fills, −7,012 if both ordinary | built; owner uploading |
| 3 | ADS-B (adsb.lol, ODbL) for EHAM 3–7 Jan 2026 | archives exist (~3 GB/day); runway + take-off second unique on 99.86% of EHAM runway-minutes; the project has `scripts/adsb_extract.py` (EHAM 2025 census: 67–77% coverage, RMSE 127–150) | no (Opus probe only) | +1k to +3k on top of #1 | ~15 GB streamed, ~2 h ingest, ~1 day |
| 4 | Arm B all-airport mixture (B.1 / B.2) | RESULT B (Rome): INCONCLUSIVE, body alone −21.2 s on LIRF non-fill rows | registered | +0.3k to +3.5k | light scoring (AC power) + 2.5 h refit to ship |
| 5 | Sweep confirmation (Amendment 15.2 never confirmed), CatBoost native, R recalibration at mid-sp | Fable-2 §closures | no | 0.5k–2k each | 2–3 h each |

## Rejected or weakened by the check

- **Haiku: "all unmatched departures go to non-EU destinations" — FALSE.** 74.6% of 2025 unmatched ADES are
  L/E-prefixed (EHAM, LEBL, LFMN, LSZH, EGLL among the top ten); LCEN (Ercan, 2,771) is the one real
  non-EUROCONTROL cluster. Haiku's MSE figures were not measured. Stand-recency (time since last arrival at
  the stand) is an unpriced lead only.
- **Sonnet: the unmatched lane's fill / date-slip structure is saturated** (175 of 186 monster rows are fills,
  10 date-slips, 12 unexplained; ex-LIRF headroom ≈ 227). Correct for the class structure — but Sonnet did
  not test congestion, which is lane #1.
- The first Fable pass's "v7 costs 14k more than band rates on LIRF" is a band-rate expectation that penalises
  confident per-row calls; Sonnet's measurement on real 2025 labels (classifier beats band rates by 10.8k) is
  the better instrument. E1 is unaffected (both agree the ≥ 24,000 band is 100% non-ordinary in 2025).

## Risks the passes surfaced (for the owner)

1. **Final-set risk (Opus).** 2026 scores every submission on all 344,841 rows, but PRC 2024 and 2025 both
   added a separate final evaluation set, and the 2026 site reserves the right to change rules. Row bets
   (E1) pay nothing if that happens; mechanisms (E3′) do. **Ask the organizers.**
2. **`AOBT_3_flt` legitimacy.** One team (kind-mango) avoids it, calling its use "probably exploiting the
   ranking process" — their interpretation, not an organizer ruling; the data page lists it as present in the
   ranking file. Our entire matched lane depends on it. **A one-line confirmation is cheap insurance.**
3. **`GAP_LOCATED` §2 inferred a hidden label from another team's leaderboard jump.** Nothing shipped depends
   on it (v5 predates it). Do not repeat.
4. E3′'s biggest stake (EHAM 3–7 Jan) is beyond 2025's support; if storm-day unmatched rows are diversions or
   mis-recorded cancellations, the downside is ≈ −2.5k. ADS-B is the independent check.

## Reachability, consensus of the passes

| target | MSE | what gets there | estimate |
|---|---:|---|---|
| 274 | 6,606 | E1 lands, or E3′ + B | likely (> 60%) |
| 266.4 (top 5) | 10,713 | E1 + E3′ + one of #3–#5 | ≈ 35–45% |
| **245** | **21,657** | E1 + E3′ at its top (≈ 7k) + ADS-B (≈ 3k) + #4–#5 at their tops, and no reversals — or a lane not yet found | **≈ 5–10%** |
