# Question to the PRC Data Challenge 2026 organisers — SENT by the owner 2026-09-11 14:02 EDT by email

**Sent version:** email to enrico.spinielli@eurocontrol.int, cc challenge@opensky-network.org, subject "Team merry-quicksand:
which inputs are permitted for the ranking predictions?" — the same two questions as the draft below, in the owner's wording
(AOBT_3_flt on ranking rows; adsb.lol globe_history + OPDI flight events for the scored flights). Awaiting the answer; the v11
upload is held until it arrives. The Discord draft below is superseded.

## Original draft (Discord), kept for the record

**Where:** OSN Discord `#prc-data-competition` (the ranking page names it as the place for questions). **From:** team
merry-quicksand. Drafted 2026-09-11 by session prc-challenge-c4; the owner decides the wording and whether to send.

---

Hello — team merry-quicksand here, with two questions about what is permitted for the ranking predictions. We would rather
ask than guess.

**1. `AOBT_3_flt` on ranking rows.** The ranking file provides `AOBT_3_flt` (the NM actual off-block time) on departure rows,
while `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are blanked. May a model use `AOBT_3_flt` (e.g. `MVT_TIME_UTC_mvt − AOBT_3_flt`)
as an input for the ranking predictions? We have seen it read as "probably exploiting the ranking process" and could not find
that sentence on the challenge pages, so we would like the official position.

**2. Open surveillance data for the same flights.** May a model use open ADS-B surveillance of the scored flights themselves —
specifically (a) adsb.lol globe_history data (ODbL 1.0, attribution kept), and (b) the OPDI flight events published by PRC with
the OpenSky Network (we understand OPDI is CC-BY 4.0)? For example, the time an aircraft is first seen moving on the ground,
derived only from those public position reports, for a flight in January or July 2026. We already use no EUROCONTROL/PRU taxi-
time statistics for 2026 and no data derived from the hidden labels.

If either is not permitted, we will remove it from our submissions. Thank you.

---

*Why these two (internal note, not for posting):* our matched lane rests on `AOBT_3` (every submission since v2), and the best
measured lever of 2026-09-11 — prc-challenge-6e's ADS-B stacker, +5,146 weighted fold MSE over arm F — reads each scored flight's
own adsb.lol track. Both are the kind of input a strict reading of "exploit the ranking process" could exclude. Evidence:
`reports/MSE_LEDGER.md` (11:2x lines), memory `reference_prc_opdi_and_aobt3_rules.md`.
