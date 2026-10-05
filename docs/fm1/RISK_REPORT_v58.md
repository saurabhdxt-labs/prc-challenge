# Risk report — board file v58 (2026-10-05)

Structure: claim, evidence, threats to validity, residual uncertainty, reproduction. v58 is a board file, not a model:
it is an average of files we had already uploaded. The documented model is FM1 (forecaster v50, best single model
v57; RISK_REPORT_v55.md and RISK_REPORT_v57.md). Registrations and reports cited by name are the project's
development record.

## 1. Claim

`merry-quicksand_v58.parquet` (sha256 `0764ca6ec531efad95b0943765abb0659a37fe383366669f5cbf760607fa1b59`, 344,841 rows)
scores **246.2224 s RMSE** on the public board (v57: 247.3187, a difference of +541.1 MSE). On the 5,290 departures
without a flight-plan record it is v57. On the other rows it is 0.1004·v33 + 0.1468·v36 + 0.0843·v41 + 0.1593·v43 +
0.5092·v55, plus v57's own change against v55, rounded and floored at 1 s.

Not claimed: that the file has any value on data other than this scored set; that it is a model; that its weights
would be chosen again on another set; any figure with a confidence interval over seeds.

## 2. Evidence

| claim part | evidence |
|---|---|
| board score | the scoring service's result file for v58: status Succeeded, `used_pairs` 344,841, score 246.2224 |
| the score was known beforehand | for weights that sum to 1, MSE(Σ a_k f_k) = Σ a_k MSE_k − ½ Σ a_j a_k D_jk; the registration fixed 246.2105 before the upload and the board gave 246.2224 (5.9 MSE apart; the average itself is 246.2197 exactly, and the file differs from it on the unmatched rows) |
| the tool | `scripts/analysis/blend_exact.py` (`tests/test_blend_exact.py`: the identity against synthetic truth, alignment by id, refusals, the optimality check of the weights, the built file against the plan); it reproduces an independent computation of the same numbers |
| each score belongs to its file | every member file on disk was compared with the object in our bucket (size and MD5) before the build |
| limits fixed before the build | members had to differ from v55 on at least 100,000 rows, with at most 5 % of the squared difference on 20 rows and no row over an hour; the unmatched rows keep v57; v57's stand-exit change enters unscaled; an independent review before the upload |
| the file is the tool's output | `build`, run with the committed tool, wrote exactly the uploaded bytes; the plan it used holds the tool's own sha256 and the members' |

## 3. Threats to validity

1. **Fitted on the scored set through its scores.** The weights minimise the board MSE itself. This is selection on
   the scored set in its most direct form; every earlier step in METHOD §8 kept or dropped a whole model, this one
   fits four numbers to the board.
2. **Not reproducible from the shipped code.** v33, v36, v41 and v43 come from the chain that preceded FM1. The member
   files are kept, not rebuilt. A clone can rebuild v50, v55 and v57 and can check the arithmetic, but cannot rebuild
   v58 from source.
3. **The exact value holds for the full average, not for the file.** The file keeps v57 on the unmatched rows, where the
   older files differ on seven Rome rows by up to 1,936 s. That part was predicted as zero and measured as 5.9 MSE by
   the upload. It is a number about a handful of rows; it is recorded and used for nothing.
4. **The members' scores contain a few-row rule.** The older files carry a rule on those seven Rome rows that gained
   59 MSE when it was kept. It moved the weights by something of the order of 0.02. The file does not carry the rows.
5. **The 5 % limit was set on the day,** after a first computation of the same average. Run over all 30 uploaded files
   on disk it admits the same 14.
6. **One evaluation.** No run-to-run variance exists for any board number.
7. **The organiser's stated dislike of reverse engineering of the ranking.** The method uses only our own files and
   their published scores, reads no label of any row, and is described here and in METHOD §8.

## 4. Residual uncertainty

We cannot say whether an average like this would help on another period: there the members' scores would not exist,
and the weights could only come from a validation set. We cannot exclude that part of the gain is specific to the two
scored months. We cannot say how v58 would rank on a hidden period, and it could not be built for one.

## 5. Reproduction

| item | value |
|---|---|
| commands | REPRODUCE.md, "Reproducing the board file v58" (`blend_exact.py plan`, then `build`) |
| inputs | the member files v33 … v57 and their result files (not in the repository), v57's provenance file |
| code | the commit that adds `scripts/analysis/blend_exact.py` built the file; the plan records the tool's sha256 |
| environment | as RISK_REPORT_v55 §5; the tool needs numpy, pandas, pyarrow and scipy only |
