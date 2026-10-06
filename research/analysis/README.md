# Research diagnostics (early analysis)

Read-only diagnostics. Each reads `data/raw/*.parquet` and prints to stdout; none writes
anything, mutates the repository, or touches the network. Run from the repository root:

    OMP_NUM_THREADS=1 nice -n 19 python3.11 -B research/analysis/<name>.py

| script | produces | written up in (development record, not in this repository) |
|---|---|---|
| `shift.py`  | 2025 vs 2026 clock availability, feature distributions, encoding vocabulary coverage | train/serve audit §1–3 |
| `drift.py`  | `delta` drift by airport and month; `P(delta < -600)` seasonality | train/serve audit §4–5 |
| `floor.py`  | timestamp granularity; repeat-pair floor (inconclusive, see caveat) | train/serve audit §6–7 |
| `stand.py`  | Stage 1 — does stand reoccupation happen (39x lift) | stand-occupancy report |
| `stand2.py` | Stage 2 — oracle value and serve-time detectability | stand-occupancy report |
| `stand3.py` | Stage 3 — take-off anchored pool, bound vs offset vs constant | stand-occupancy report |
| `mangle.py` | LOMO validation of the mangled-designator separator (closed) | measurement ledger §1 |

Every script that builds a composite `(airport, stand)` key asserts the key did not collapse.
A `\x00` separator is silently treated as a string terminator by pandas' Arrow-backed string
path and reduces 2,277 keys to 10 — see the landmine note in stand-occupancy report.

`stand3.py` section 4 printed a mis-scaled "ceiling" figure in the original run and it is not
quoted anywhere; the value question is settled by the paired A/B of Amendment 5, not by that block.
