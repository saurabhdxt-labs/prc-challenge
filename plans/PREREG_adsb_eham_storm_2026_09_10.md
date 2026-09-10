# Pre-registration — arm E4: an ADS-B off-block measurement for EHAM's January 2026 storm days

**Written 2026-09-10 17:00 local, BEFORE any 2026 ADS-B archive is fetched and before any number of this arm exists.**
Tag `E4`. Source: `reports/PIPELINE_TO_245_2026_09_10.md` Stage 2. Reopens `reports/ADSB_GATE.md` (closed NO-GO
2026-09-08 for a ten-airport, ~190 GiB ingest) **narrowly**: one airport that passed its census (EHAM: gated RMSE
127 s at 66% coverage vs model 173 s), eight days, ≈ 25 GB streamed.

**Eligibility gate (from ADSB_GATE.md, binding):** adsb.lol publishes under ODbL-1.0 and the challenge permits
documented open datasets, but the licence question with the organisers is NOT closed. **Measurement may proceed;
no submission may depend on this arm until the owner has the organisers' answer in writing.** The derived table's
licence (ODbL share-alike) is documented in the repo before any ship.

## Hypothesis

**H-E4:** for EHAM departures on 2026-01-02 … 2026-01-09, the last ADS-B sample inside the flight's own stand
gate (radius 100 m, stand centroids learned leave-one-flight-out from 2026 matched EHAM departures at their
`AOBT_3`) is an off-block measurement accurate enough to replace the model where it fires.

## Design (fixed now)

- Ingest: `scripts/adsb_extract.py --day D` for D in 2026-01-02 … 2026-01-09 (adsb.lol daily archive, streamed,
  EHAM box only, archive deleted after extraction). Probe first (`--probe`) to confirm the 2026 format.
- Stand centroids: position at `AOBT_3` of matched EHAM departures on the eight days, leave-one-flight-out,
  per `STAND_mvt`; a stand needs ≥ 2 other flights.
- Estimate: last sample within 100 m of the row's own stand centroid before take-off; bias-corrected by the
  median (sensor − `AOBT_3`) of matched rows on OTHER days (leave-one-day-out).
- Join: matched rows by `CALLSIGN_flt` (ADSB_GATE's key); unmatched rows (no callsign) by runway + take-off
  second (`MVT_TIME_UTC_mvt` ↔ first ADS-B sample with ground speed > 80 kt on the runway); ambiguity is a
  clause, not an assumption.
- Prediction where the gate fires: `taxi_hat = MVT − offblock_hat`; elsewhere unchanged.

## Clauses, locked

- **C0 (join)** — unmatched runway + take-off-second join: ambiguous or unmatched share ≤ 5% of rows where a
  candidate exists; ambiguity rate reported.
- **C1 (the closure oracle, in-file, on the storm):** on 2026 matched EHAM departures of the eight days,
  bias-corrected RMSE of `offblock_hat − AOBT_3` ≤ **250 s** with coverage ≥ **50%**. If C1 fails: NOT WORKING,
  no unmatched row is touched.
- **C2 (stability):** leave-one-day-out — the C1 RMSE on each held-out day ≤ 300 s on ≥ 6 of 8 days.
- **C3 (beats the incumbent where it fires):** on matched storm rows where the gate fires, the ADS-B taxi
  estimate against the true label (`TAXITIME` is known for 2025 only — so this clause uses 2025-01-09's census
  rows, `data/adsb/`, as the only labelled check, and must show gated RMSE < model RMSE as in ADSB_GATE).
- **Verdict:** WORKING iff C0–C3 pass; NOT WORKING iff C1 fails; else INCONCLUSIVE.
- **Ship rule (only if WORKING AND the eligibility answer is yes):** v_next = best board version with EHAM
  unmatched rows of 2026-01-02 … 09 replaced by the ADS-B estimate where the gate fires (and, as a separate
  registration, matched storm rows); rebuild guard on the base; board reading Δ vs base; **expected price
  reported as the event-excluded, conservative end (+0.8k), full end +1.8k (research lead's Stage 2)**; Δ ≥ 0
  = NOT TRANSFERRED.

## Predicted shapes

- **TRUE:** C1 RMSE ≈ 130–250 s at 55–75% coverage in the storm; storm-day unmatched estimates well above
  v10's ≈ 1,100–1,300 s means, near the matched witness level.
- **FALSE:** snow / de-icing degrades ground reception (coverage < 50%), or de-icing pads move the aircraft
  far from the stand after pushback so "last sample at the stand" misses the recorded off-block (RMSE ≫ 250).

## Limits

EHAM only; eight days; one labelled 2025 day for C3. Receiver coverage in snow is untested. A final evaluation
set, if added by the organisers, would contain other days — this arm is regime-specific by construction.
