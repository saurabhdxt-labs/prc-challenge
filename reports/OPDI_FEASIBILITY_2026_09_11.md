# OPDI feasibility — can taxi-out models use it? (descriptive; no arm, no fit)

Written 2026-09-11 10:36:23 (from `date`), session prc-challenge-c4. Owner: "see if training models can be done on this data".

**Data** (free, no registration; licence reported by team kind-mango as CC-BY 4.0 confirmed by the organiser): OPDI v0.0.2
`flight_events_20250105_20250115.parquet` (244.6 MB, 8,428,168 events, sha256 6e5947b7…c7cb) and `flight_list_202501.parquet`
(33.9 MB, 1,036,088 flights, sha256 ea1a7443…5de) in `data/opdi/` (gitignored). Coverage 2022-01-01 … 2026-07-31. Source OSN for every
event. Callsigns are space-padded (strip before joining).

**Join** to our 2025 departures (6–13 January, callsign + origin, the OPDI flight whose runway entry / take-off / first-seen is within
20 min of our take-off): 31,770 of 39,362 (81%); LTFM 50%, LEMD 60%, LIRF 73%, others 75–98%.

| airport | stand exit seen | its error vs y (bias / MAD) | first taxiway entry seen | its error (bias / MAD) | AOBT_3 median abs error |
|---|---|---|---|---|---|
| **LSZH** | **71%** | −95 / 90 s | **93%** | **−78 / 30 s** | 134 s |
| LEBL | 13% | −145 / 131 | 76% | −539 / 185 | 119 |
| EDDM | 1% | — | 13% | −474 / 119 | 125 |
| EHAM | 0% | — | 6% | −682 / 156 | 127 |
| EGLL, EDDF, LFPG, LEMD | 0% | — | ≤ 1% | — | 121–183 |
| **LIRF, LTFM** | **0%** | — | **0%** | — | 235 / 422 |

(Error = MVT − event time − taxi-out label; MAD = median absolute deviation around the bias. Runway entry is seen for 18–97% of
flights but sits seconds to a minute before take-off — the END of the taxi.)

**What this means.** OPDI measures the taxi of the SAME flight precisely only at LSZH (93% of flights, ≈ 30 s after a fixed offset),
partly at LEBL; nowhere near the airports carrying most of our error (LIRF, LTFM, LFPG, LEMD — the early-off-block regime's 14,500 of
22,900). Arm F's LSZH error is ≈ 1,919 board MSE, so a same-flight OPDI feature there is worth at most ≈ 1,500–1,700 (≈ −3 RMSE) —
**and using a scored flight's own surveillance events is close to measuring the hidden answer: organiser question, not ours to
decide** (kind-mango uses only OTHER completed flights' events, lagged 600 s). Other-flight OPDI features (runway sequence, recent
observed taxis) are the defensible use; they overlap our queue features, so likely a few hundred MSE. Not measured here.
