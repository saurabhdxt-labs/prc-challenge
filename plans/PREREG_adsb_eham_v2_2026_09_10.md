# Pre-registration — arm E4b: EHAM ADS-B off-block, label-referenced, validated on unseen congested 2025 days

**Written 2026-09-10 18:15 local.** Tag `E4b`. Successor to E4 (`plans/PREREG_adsb_eham_storm_2026_09_10.md`), which
cannot return WORKING by design (its bias referenced `AOBT_3`, not the label; see its PRE-INGEST FINDING). **Nothing
of this arm has been measured. The validation days below have never been read by any ADS-B work in this project.**
Eligibility gate inherited unchanged: measurement may proceed; **no submission may depend on this arm until the
organisers confirm the adsb.lol (ODbL-1.0) licence in writing**, and the licence is documented in the repo.

## The procedure (identical on the validation days and on 2026)

- Stand centroids: position at `AOBT_3` of MATCHED EHAM departures on the same day(s), leave-one-flight-out, per
  `STAND_mvt`, ≥ 2 other flights (2026 has no `BLOCK`, so `AOBT_3` locates the stand — location only).
- Estimate: last ADS-B sample within 100 m of the row's own stand centroid before take-off.
- **Bias: ONE constant, referenced to the label** — the median of (sensor − `BLOCK`) on matched EHAM rows of the
  2025-01-09 census day (already seen; used ONLY as the calibration day, never as validation).
- `taxi_hat = MVT − (sensor − bias)`; floored at 1 s.
- Unmatched join: runway + take-off second, as in E4 (the harness's tested join).

## Validation (decisional) — EHAM, 2025-12-27 and 2025-12-28 (the two most congested 2025 EHAM days for unmatched
rows: 46 unmatched rows, 20 at `hprox_med ≥ 1,500`; several hundred matched rows)

- **V1 (the sensor, against the label, in congestion):** matched rows — RMSE(`sensor − bias − BLOCK`) ≤ **200 s**
  with coverage ≥ **50%**. If V1 fails: **NOT WORKING**; nothing further is ingested.
- **V2 (the lever, against the incumbent, on the rows it would replace):** unmatched rows where the gate fires and
  the join is unambiguous — RMSE(`taxi_hat − y`) **<** RMSE(S1C − y) on the same rows, S1C taken from
  `data/cache_stand/unm_congestion_lomo.parquet` (out-of-month). Requires ≥ **15** gated unmatched rows; fewer →
  INCONCLUSIVE (too little to decide), reported.
- **V3 (join):** ambiguous or failed unmatched joins ≤ 5% of rows with a candidate.
- **Verdict:** WORKING iff V1–V3 pass; NOT WORKING iff V1 fails; else INCONCLUSIVE. Never revised.

## Only if WORKING

The 8-day 2026 ingest (2026-01-02 … 09) runs under the same procedure; the 2026 matched storm rows then give an
in-file diagnostic (sensor vs `AOBT_3`, reported only — `AOBT_3` is ≈ 269 s from the true off-block, so it cannot
be a pass/fail bar). Ship (eligibility permitting): best board version with EHAM unmatched rows of the eight days
replaced where the gate fires; rebuild guard; **expected price reported at the conservative end (+0.8k), full
end +1.8k**; Δ ≥ 0 = NOT TRANSFERRED.

## Predicted shapes

- **TRUE:** V1 ≈ 120–200 s at 55–75% coverage; on gated unmatched rows the sensor sits far closer to `y` than S1C
  in the busy hours (S1C runs hundreds of seconds low there).
- **FALSE:** congestion or winter conditions push aircraft off the stand gate (coverage < 50% or RMSE ≫ 200), or
  the census-day bias does not transfer across days (a large day-to-day offset).

## Limits

Two validation days, 46 unmatched rows — a small, honest test; its power is limited and V2's minimum-count rule is
there so a handful of rows cannot decide it. Winter 2025-12 is congestion, not necessarily snow; the January 2026
storm's snow and de-icing remain the untested part of the transfer.

## AMENDMENT E4b.1 · 18:17 — implementation judgement calls and the calibration, BEFORE `validate` (no validation number exists)

Harness: `scripts/adsb_storm.py` (`calibrate`, `validate`, `gate --e4b`, `ship --e4b`), `tests/test_adsb_e4b.py` (21 new;
83 across the ADS-B / ship suites; 36 mutations, 36 RED). **Bias constant (calibration day 2025-01-09 only):
244.54 s = median(sensor − BLOCK), n = 371 of 592 matched (62.7%), IQR 126.5 s** (`reports/adsb_e4b_bias.json`).
Judgement calls, none moving a threshold:
1. Stand centroids pool BOTH validation days (the prereg's "same day(s)"; 2026 pools its eight days likewise).
2. V1 coverage = gated matched rows ÷ all matched rows of the validation days.
3. V2 floors `taxi_hat` at 1 s without rounding; S1C is the stamped file's three-seed `S1C`; the file's `y` must equal
   the label.
4. By the registered mapping, V2 failing on ≥ 15 rows gives INCONCLUSIVE (only V1 decides NOT WORKING).
5. A flight that left its stand the previous UTC day cannot fire the gate.
6. A full 2026 `extract` refuses unless the E4b verdict on disk is WORKING (the prereg's "only if WORKING").
Probe of 2025-12-28 parsed: 11,970 EHAM samples in 60 MiB, 99.5% with callsign.

## RESULT E4b · 2026-09-10 18:28 — **NOT WORKING** (V1 fails)

Source: `reports/adsb_e4b_validate.json`, rows `data/adsb/adsb_e4b_validation_rows.parquet` (validate, exit 0). Ingest:
2025-12-27 (424,911 samples, 231 s) and 2025-12-28 (411,253, 260 s), archives deleted.

| clause | value | bar | pass |
|---|---|---|---|
| V1 matched rows vs BLOCK (914 gated of 1,193; coverage 76.6%) | **RMSE 487.6 s** | ≤ 200 s at ≥ 50% | ✗ |
| V2 gated unmatched rows vs S1C (n 30) | 119.4 s vs 397.2 s | lower | ✓ |
| V3 join ambiguity (n with candidate 38) | 2.6% | ≤ 5% | ✓ |

Per-day V1: 404.2 s (12-27, coverage 73.7%), 553.0 s (12-28, 79.5%); median sensor − BLOCK 229 / 235 s vs the
calibrated 244.5 s (the constant transferred). **Verdict stands; no 2026 ingest under this registration.**
Reported, not decisional (post-hoc diagnostics on these days): median |error| 58.8 s, 68.8% within 120 s, 97.4%
within 300 s; **5 rows carry 76% of the SSE; without the top 20 of 914 the RMSE is 116 s**; 9 rows exceed 1,000 s
(7 with the sensor EARLIER than BLOCK — consistent with tows / leaving the stand area before the recorded
off-block). V2's 30 unmatched rows had max |taxi_hat − y| 257 s vs S1C 1,162 s. A guarded successor is plausible,
but its guard would be designed after seeing these two days: it needs its own registration and validation on
fresh labelled days.

## RESULT E4b is VOID · 2026-09-10 19:20 — the instrument was defective, not the sensor

Surfaced by an independent 11-slice review (session 80a473c1, 18:30–19:10; recorded in project memory) and
re-verified here: `adsb_storm.join_matched` (`scripts/adsb_storm.py` ≈ l.320–330) assigns each matched flight the
same-callsign track with the MOST samples in [MVT − window, MVT + slack] — which can be the flight's INBOUND leg
sitting at the stand. On the 7 worst V1 rows the "last sample at the stand" is 47–100 min before BLOCK and 1–1.9 h
before take-off (an arrival, not a pushback); **those 7 rows carry 91.8% of V1's SSE.** The reviewer reports V1 ≈ 141 s
with a take-off-anchored join (post-hoc). My own diagnostic above ("consistent with tows") was WRONG — it named a
physical story for what was a join defect. **The registered verdict is not revised to anything; it is VOID** (a
measurement from a defective harness decides nothing — cf. BC-2's VOID instances). The mutation rehearsal did not
catch it because no fixture had two tracks sharing a callsign. A successor (E4c) must be registered fresh: a
take-off-anchored matched join, a regression test with inbound + outbound tracks under one callsign (seen RED
against the current join first), and validation on fresh labelled days (2025-12-27/28 are now spent).
