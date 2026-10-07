# research/

Research and analysis code from the development of FM1: pricing harnesses, label-free checks, and experiments run
against candidate levers (stand occupancy, unmatched-lane bodies, ADS-B storms, gate censuses, model-selection and
calibration sweeps, and the rest). None of it is used by the final model (`prc/`, `configs/fm1_v57_open.yaml`) —
each file's own docstring says what it did and, where applicable, what it found. Kept here, open, under the same
GPLv3 as the rest of the repository (`LICENSE`), because every produced file stays in the repo. The sub-structure
mirrors `scripts/`: top-level harnesses live in `research/`, and the checks/experiments that used to live in
`scripts/analysis/` live in `research/analysis/`; development-time tools that read the development record (arm pairing and verdicts,
the top-path audit, a historical upload verifier, the arms queue driver) live in `research/tools/`.
