# FM1 — data sources and licences

Raw data never enters the repository (`.gitignore`: `data/`, `*.parquet`, `*.csv`); each source is re-fetched by the
script named. Licence statements quote the fetch scripts' own records; where the primary licence text has not been
re-checked this session it says so.

| source | used by FM1 | fetched by | licence / terms | evidence |
|---|---|---|---|---|
| PRC Data Challenge 2026 movement files (training months, ranking/submitting) | yes — all rows | `data/raw/FETCH_MANIFEST.json` (S3) | the organiser's challenge terms; not redistributed | `.gitignore` header |
| adsb.lol globe_history | yes — the ADS-B tables (R12, stamp 801f9bcac9a5) | `scripts/adsb_fetch.py` | **ODbL 1.0**; attribution "ADSB.lol contributors" | `scripts/adsb_fetch.py:6,52` |
| adsb.fi opendata (the owner's 60 s poll archive) | optional: July-2026 serve table only (`adsb.sources.adsb_fi`, off in the forecaster) | `scripts/adsbfi_extract.py` | **non-commercial** terms; not redistributed; July 2026 capture only, cannot be re-fetched | `scripts/adsbfi_extract.py:6-8,43` |
| OpenStreetMap (Overpass) | yes — the ORL route table | `scripts/analysis/orl_build.py fetch/build` | **ODbL 1.0**; "(c) OpenStreetMap contributors" | `scripts/analysis/orl_build.py:14,164` |
| X-Plane Scenery Gateway apt.dat | no (the APT stand fallback arm is in the ablation table) | `scripts/analysis/apt_build.py` | **GPL v2 or later** | `scripts/analysis/apt_build.py` header |
| EUROCONTROL ansperformance monthly taxi-out CSV (`taxi_out_additional_time_<year>.csv`) | only the scored-period build (MTX; off in the forecaster) | `data/mtx/raw/MANIFEST.json` (sha256) | EUROCONTROL performance open data; **TODO: quote the exact reuse terms** | `scripts/analysis/mtx_source.py:6`; owner ruling in the manifest |
| METAR via Iowa Environmental Mesonet (ASOS/METAR archive) | yes — the weather block | `scripts/fetch_weather.py` | stated **public domain** by the fetch script | `scripts/fetch_weather.py:5,44` |
| Open-Meteo | no (OMW closed NOT WORKING at G1) | `scripts/analysis/omw_build.py` | **TODO: quote the terms** (not an FM1 input) | memory OMW note |

Derived inputs (ADS-B tables, the stack models, the ORL route table, the MTX CSVs) are rebuilt from the sources above by
code in the repository; REPRODUCE.md lists each one with the command and the version the final model uses.

Refused sources (never read by any FM1 path): EUROCONTROL 2026 taxi-out aggregates as a label source outside the MTX
calibration ruling (memory `feedback_prc_refused_label_aggregate_sources`); paid data (owner ruling); OpenSky beyond the
competition files (ADN prereg).
