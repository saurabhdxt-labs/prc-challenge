# Risk report — board file v57 (2026-10-05)

Structure: claim, evidence, threats to validity, residual uncertainty, reproduction. v57 is v55 with one stage added, so
this report covers what the stage adds; everything in RISK_REPORT_v55.md still applies. Numbers come from the files
named; registrations and reports cited by name are the project's development record, which the rebuild does not need.

## 1. Claim

`merry-quicksand_v57.parquet` (sha256 `4bf0f915b90ae7374aa97e435a254d9d2bf73292bda30f47cad55d97c154350b`, 344,841 rows)
is v55 with the stand-exit stage `uad` on, produced by the code in this repository from the artefact
`artifacts/fm1/e3c1d6b473f4-141862a0` (config `configs/fm1_v57.yaml`, config hash `e3c1d6b473f4…`). It scores
**247.3187 s RMSE** on the public board (all 344,841 scored departures of January and July 2026); v55 scores 249.9168 s,
a difference of +1,291.9 MSE.

Not claimed: that the stage gains this much on another period; which of the stage's rows carried the gain; that v57
differs from v55 where the ADS-B store has no ground coverage at the stand (there the two files agree to within the
1–2 s the calibration moves); any accuracy figure with a confidence interval over seeds (none exists).

## 2. Evidence

| claim part | evidence |
|---|---|
| board score | the scoring service's result file for v57: status Succeeded, `used_pairs` 344,841, score 247.3187 (v55: 249.9168) |
| the code produced the file | written by `python -m prc build-submission --config configs/fm1_v57.yaml --out <dir> --version 57` with the committed code and a clean tree; the manifest names the artefact, the config hash and the sha256 of the stage's two inputs; rebuilt on 2026-10-05 by later code from the stored artefact, the file has the same sha256 (REPRODUCE.md, "Reproducing the board file v57") |
| the config is the final one | `tests/fm1/test_fm1_final_configs.py`: `configs/fm1_v57.yaml` resolves to the artefact's config hash and differs from v55 in one knob (`modules.uad.enabled`) |
| the stage on its own, labelled 2025 | one registered test, read once: the 317 January + July 2025 rows with an estimate, RMSE 432.4 → 274.4 s; body +53.3 bmse [21.8, 85.3], positive in both months, without the ten largest rows and with EHAM removed (METHOD §7, UAD; CLAIMS 24) |
| the stage's part in the file | `decided_by` = `uad` on 501 rows (class A1 450, A2 51; 124 on storm airport-days; EHAM 328, LEBL 47, LSZH 44, EDDM 34, EGLL 28, LEMD 14, EDDF 6); `uad_cls`, `uad_est`, `uad_shift` per row in the provenance file; an independent review before the upload recomputed all 501 values from the registered rule |
| the join behind the estimates | the registered checks on the 2026 table at EHAM: aircraft type equal on 0.9949 of callsign joins, type agreement on 0.9969 of stand joins, a median absolute deviation of 144.0 s for the estimate minus the Network Manager's clock on matched flights, and on a 1-in-10 sample of matched flights with a known callsign the stand rule found the right take-off for 0.9973 and a wrong one for none |
| the stage's code | `tests/fm1/test_fm1_uad.py` (including a differential test against the rule that was priced), `tests/test_uad_build.py`, `tests/test_uad_price.py`; mutation rehearsals in `tests/fm1/MUTATION_FM1.md` |
| test suite | `tools/run_suite.py` (whole suite, two isolated passes); clone tier `tools/clone_check.py` |

## 3. Threats to validity

1. **Part of the stage has thin evidence.** Of v57's change against v55, 58 % sits on 146 rows that the 2025 test
   barely covered: 124 storm-day rows (the storm weight is the registered floor, 0.8, resting on 28 design-month rows;
   the storm-day hypothesis could not be tested) and 51 class-A2 rows (January + July 2025 held five).
2. **One board evaluation, and a gain larger than predicted.** Before the upload the part the 2025 test supported was
   put at +220 … +430 MSE and the whole file at −150 … +1,000. The board gave +1,291.9, which is 1.96 times the
   change between the files (on the 2025 test the ratio was 1.55). The 2025 test does not explain the size, and one
   number cannot say which rows carried it.
3. **Concentration.** 328 of the 501 rows are at EHAM, 230 of them in January 2026, a snow month. The gain may belong
   to that event.
4. **Coverage.** The stage reaches 503 of the 5,290 scored departures without a flight-plan record, none at LIRF, LFPG
   or LTFM. The 2026 table uses the adsb.lol day store only. ADS-B ground coverage moved sharply between 2025 and 2026
   (LIMITATIONS 14), so coverage in another period is not known.
5. **Constants fitted against a different baseline.** The offsets and weights were fitted on 2025 against screen-tier
   leave-one-month-out predictions, not against the final model the stage runs after; those baseline fits include
   January and July 2025, so a little of the two test months reaches the calm weight.
6. **The table check came after the build.** When v57 was built the stage accepted any table with its four columns,
   and the 2026 table's build report was read by hand. The stage now refuses a table whose build report fails the
   registered label-free checks, and v57 rebuilt with that check in place is the same file. For tables built from now
   on the report holds the sha256 of the table and the stand positions it was given; the table behind v57 is tied to
   its report only by sharing a directory (both hashes are in the rebuild's manifest).
7. **Selection on the scored set** (RISK_REPORT_v55 §3.1) applies to keeping v57: it is our best single board file
   because the board says so. Nothing inside the stage was set from the board.

## 4. Residual uncertainty

We cannot exclude that: the gain is specific to January 2026 at EHAM; the storm weight or the class-A2 factor is far from
what labels would choose; a few of the 501 rows are joined to the wrong take-off (the check found none among sampled
matched flights, and missed 0.27 % of them); the gain would differ with the adsb.fi capture in the table. We cannot say
how v57 would rank on a hidden period.

## 5. Reproduction

| item | value |
|---|---|
| commands | REPRODUCE.md, "Reproducing the board file v57" (derive from the v50 artefact with `--reuse-from`, build); the stand-exit table and constants are derived inputs in the same document |
| code | the commit that adds the `uad` stage built the artefact and the file with a clean tree (the artefact records it) |
| environment, hardware, seeds | as RISK_REPORT_v55 §5 |
| data | DATA_SOURCES.md; the stand-exit table is built from the ADS-B day store, which is not in the repository |
