# Analysis scripts

Read-only diagnostics. Each reads `data/raw/*.parquet` and prints to stdout; none writes
anything, mutates the repository, or touches the network. Run from the repository root:

    OMP_NUM_THREADS=1 nice -n 19 python3.11 -B scripts/analysis/<name>.py

| script | produces | report |
|---|---|---|
| `shift.py`  | 2025 vs 2026 clock availability, feature distributions, encoding vocabulary coverage | `reports/TRAIN_SERVE_AUDIT.md` §1–3 |
| `drift.py`  | `delta` drift by airport and month; `P(delta < -600)` seasonality | `reports/TRAIN_SERVE_AUDIT.md` §4–5 |
| `floor.py`  | timestamp granularity; repeat-pair floor (inconclusive, see caveat) | `reports/TRAIN_SERVE_AUDIT.md` §6–7 |
| `stand.py`  | Stage 1 — does stand reoccupation happen (39x lift) | `reports/STAND_OCCUPANCY.md` |
| `stand2.py` | Stage 2 — oracle value and serve-time detectability | `reports/STAND_OCCUPANCY.md` |
| `stand3.py` | Stage 3 — take-off anchored pool, bound vs offset vs constant | `reports/STAND_OCCUPANCY.md` |
| `mangle.py` | LOMO validation of the mangled-designator separator (closed) | `reports/MSE_LEDGER.md` §1 |

Every script that builds a composite `(airport, stand)` key asserts the key did not collapse.
A `\x00` separator is silently treated as a string terminator by pandas' Arrow-backed string
path and reduces 2,277 keys to 10 — see the landmine note in `reports/STAND_OCCUPANCY.md`.

`stand3.py` section 4 printed a mis-scaled "ceiling" figure in the original run and it is not
quoted anywhere; the value question is settled by the paired A/B of Amendment 5, not by that block.
