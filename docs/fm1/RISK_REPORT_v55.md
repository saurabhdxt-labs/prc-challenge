# Risk report — board file v55 (2026-10-04)

Structure: claim, evidence, threats to validity, residual uncertainty, reproduction. Numbers come from the files named;
registrations and reports cited by name are the project's development record, which the rebuild does not need.

## 1. Claim

`merry-quicksand_v55.parquet` (sha256 `225ab0be205a0a254d0c27e726ec33d6007981b98f937f9b00739499e5ac537d`, 344,841 rows)
is FM1's scored-period build, produced by the code in this repository from the stored artefact
`artifacts/fm1/d811c69dc93e-141862a0` (config `configs/fm1_v55.yaml`, config hash `d811c69dc93e…`). It scores
**249.9168 s RMSE** on the public board (all 344,841 scored departures of January and July 2026). The forecaster, on the
challenge files and public open data only (v50, `configs/fm1_v50.yaml`), scores 251.4767 s; with the published monthly
calibration only (v51), 251.1335 s.

Not claimed: that v55 is better than v50 / v51 on data other than this scored set; that the adsb.fi capture helps a
future period; any accuracy figure with a confidence interval over seeds (none exists).

## 2. Evidence

| claim part | evidence |
|---|---|
| board scores | the scoring service's result files for v50, v51, v55: status Succeeded, `used_pairs` 344,841, scores 251.4767 / 251.1335 / 249.9168 |
| the code reproduces the file | REPRODUCE.md, "Reproducing the board file v55": the current code serving the stored artefact (rebuild check) |
| the config is the final one | `tests/fm1/test_fm1_final_configs.py`: `configs/fm1_v55.yaml` resolves to the artefact's config hash and differs from the defaults in exactly the recorded overrides |
| determinism | two fits of a three-month slice of the final configuration served the full scored file identically (344,841 × 30 cells, 0 different; METHOD §6) |
| component contributions | `decided_by` on every row (MODEL_CARD); the swap-study ablations (METHOD §5); one-knob whole-model variants on the board (METHOD §8: MTX +172.5, adsb.fi +609.6, CatBoost half-blend +166.1 MSE) |
| 2025 reference (reduced configuration) | protocol S, screen tier, stack off: January 327.07 s, July 266.21 s (METHOD §6) |
| no label in a design | input contract and design checks (`prc/pipeline/fm1/rows.py`; AT2 tests); 2026 departure labels are blank in the organiser's scored file |
| test suite | `tools/run_suite.py` (whole suite, two isolated passes); clone tier `tools/clone_check.py` |

## 3. Threats to validity

1. **Selection on the scored set.** Every step kept after 2026-09-10 was kept because the public board improved, and the
   board is the same 344,841 rows every time. The cumulative gain is therefore optimistic for unseen data; the final-model
   audit of 2026-09-28 put about 28 % of it in board-specific patches (METHOD §8).
2. **2025 → 2026 shift.** ADS-B coverage moved sharply by airport-month (LIMITATIONS 14; imposing July 2026's losses on
   July 2025 costs ~860 bmse); LFPG's EOBT field changed in July 2026 (LIMITATIONS 9). Validation on 2025 cannot see
   either.
3. **Scored-period information.** MTX uses EUROCONTROL's published 2026 monthly means (an aggregate of the target
   period; allowed by the organiser, DATA_SOURCES); adsb.fi is a July-2026 capture that cannot be re-fetched.
4. **Stack served on a differently made F** (LIMITATIONS 16) and **CatBoost half at its round cap** (LIMITATIONS 13).
5. **In-sample constants and encodings** (LIMITATIONS 17): validation overstates the unmatched lane's rules and LDI.
6. **One run.** v55 is one run of the three-seed ensemble (`run.seeds` 0, 1, 2); no run-to-run variance is measured; differences of a few hundred board MSE may be noise.
7. **adsb.fi joins.** Possible misjoins to a previous rotation among the July serve rows (LIMITATIONS 20).

## 4. Residual uncertainty

We cannot exclude that: part of v55's margin over v50 is specific to this scored set; the stack would serve better on
an F made the way it was trained; the July 2026 serve rows carry a few hundred adsb.fi misjoins; a different seed moves
the board score by more than the margins between v50, v51 and v55. We cannot say how v55 would rank on a hidden period.

## 5. Reproduction

| item | value |
|---|---|
| commands | REPRODUCE.md, "Reproducing the board file v55" (fit v50, derive v55 with `--reuse-from`, build) |
| code | this repository at the commit that adds this report; the artefacts record the commit they were built on (v50: 0a9b5e7, v55: d7b4f1c, recorded as b285bda and 33ee40a before the 2026-10-06 history rewrite; each with uncommitted changes since committed; LIMITATIONS 18) |
| environment | Python 3.11.15; `requirements.txt`, `requirements-dev.txt`, `requirements-catboost.txt`; check with `python tools/env_check.py` |
| hardware | the development machine: Apple silicon, 18 cores, 24 GiB (full fit ~10 h, ~20 GB); cross-machine bit-identity is not claimed |
| seeds | `run.seeds: [0, 1, 2]` (configs), LightGBM deterministic with fixed threads (12) |
| data | DATA_SOURCES.md; derived inputs and what a clone cannot rebuild: REPRODUCE.md |
