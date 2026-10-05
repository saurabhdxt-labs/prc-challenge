# FM1 — data sources and licences

Raw data never enters the repository (`.gitignore`: `data/`, `*.parquet`, `*.csv`); each source is re-fetched by the
script named. Licence statements quote the fetch scripts' own records; where the primary licence text was not
re-checked, the table says so.

| source | used by FM1 | fetched by | licence / terms | evidence |
|---|---|---|---|---|
| PRC Data Challenge 2026 movement files (training months, ranking/submitting) | yes — all rows | `scripts/fetch_data.py` (the organiser's bucket; each object checked against its ETag (the MD5 of a single-part upload) or, for a multipart upload, its size; recorded in `data/raw/FETCH_MANIFEST.json`) | the organiser's challenge terms; not redistributed | `.gitignore` header |
| adsb.lol globe_history | yes — the ADS-B tables (stamp 51348d3e1cb3; the stored stack's training holdout used the older 801f9bcac9a5 tables) and the stand-exit table of the optional `uad` stage | `scripts/adsb_fetch.py` | **ODbL 1.0**; attribution "ADSB.lol contributors" | `scripts/adsb_fetch.py:6,52` |
| adsb.fi opendata (the owner's 60 s poll archive) | v55 only: July-2026 serve table (`adsb.sources.adsb_fi`; off in the forecaster v50) | the owner's poll archive, extracted by a local step that is not shipped (the capture cannot be redistributed) | **non-commercial** terms; not redistributed; July 2026 capture only, cannot be re-fetched — see the eligibility note below | this table |
| OpenStreetMap (Overpass) | yes — the ORL route table | `scripts/analysis/orl_build.py fetch/build` | **ODbL 1.0**; "(c) OpenStreetMap contributors" | `scripts/analysis/orl_build.py:14,164` |
| X-Plane Scenery Gateway apt.dat | no (tried as a stand fallback, not used) | not shipped | **GPL v2 or later** | — |
| EUROCONTROL ansperformance monthly taxi-out CSV (`taxi_out_additional_time_<year>.csv`) | only the scored-period build (MTX; off in the forecaster) | `data/mtx/raw/MANIFEST.json` (sha256) | EUROCONTROL performance open data (published monthly aggregates; use as a model input confirmed by the organiser, below); the page's reuse terms are not quoted here | the organiser statement of 2026-09-27 (below) |
| METAR via Iowa Environmental Mesonet (ASOS/METAR archive) | yes — the weather block | `scripts/fetch_weather.py` | stated **public domain** by the fetch script | `scripts/fetch_weather.py:5,44` |
| Open-Meteo | no (tried beside METAR, not used) | not shipped | not an FM1 input | — |

Derived inputs (ADS-B tables, the ORL route table, the MTX CSVs) are rebuilt from the sources above by code in the
repository; the stored ADS-B stack models are an input the current code cannot regenerate byte for byte. REPRODUCE.md
lists each one with the command and the version the final model uses.

Refused sources (never read by any FM1 path): EUROCONTROL 2026 taxi-out aggregates as a label source outside the MTX
calibration; paid data; OpenSky beyond the competition files.

## Organiser statements this project relies on

Paraphrased from the competition's Discord channel as recorded in the development record (the verbatim messages are not
in this repository):

- 2026-09-16: the other fields of a scored row and the other supplied rows may be used as inputs.
- 2026-09-17: the named EUROCONTROL daily files may be used for the challenge; ADSB.lol subject to its licence.
- Around 2026-09-18: the model is meant for post-operations use, not tactical prediction, and extracting off-block times
  from open trajectories is accepted — the basis of FM1's ADS-B features, which measure each flight's own surface
  movement (gate dwell, first movement, reconstructed taxi start) from its public track.
- 2026-09-27: EUROCONTROL's published monthly taxi-out figures for January / July 2026 may be used as a model input — the
  basis of the MTX calibration in the scored-period build (v51, v55).
- 2026-10-02 / 10-03: open data that helps may be used; data without a licence is not open; the final ranking is the
  public board plus a review of the top teams' code and documentation.

## Eligibility note on adsb.fi

The challenge's eligibility page asks for external data that is openly accessible and usable, and for reproducible
results. The adsb.fi capture behind v55 was polled from adsb.fi's open API under its non-commercial terms, is never
redistributed and cannot be re-fetched, so only the machine that holds it can rebuild v55. Whether that qualifies is the
organiser's decision; v51 (public board 251.1335) is the same model without it, on the challenge files and public open data only.

