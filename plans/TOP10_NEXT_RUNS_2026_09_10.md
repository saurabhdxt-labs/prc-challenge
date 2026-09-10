# Top ten: decisions after the overnight results

This updates the scheduling recommendations in TOP_PATH_2026_09_10.md. It does not
change existing preregistrations, running jobs or submission artifacts. Scores below
refer to the user's September 10 snapshot; the live leaderboard could not be refreshed.

**Objective:** recover at least 6,884 MSE against v5 in a complete, validated candidate.
Use 9,000–10,000 MSE as a research target with margin, not a forecast. Under one-to-one
MSE transfer, 9,000 would put 288.1406 at 272.075; 10,000 at 270.231. A matched-only
6,884-MSE improvement corresponds to 224.258 → 208.088 on the existing matched fold.
That is the scale the research program needs to address.

**Scheduling call:** preserve F's nearly completed work, defer W behind the structural
runs, and defer the 2.5-hour refit for the 232-MSE blend. The last inspected F log had
completed the third seed's refit. This is a change from my earlier recommendation to
finish the whole queue. W is lower priority on opportunity cost; D's failure does not
refute weather, and W should remain an unmeasured hypothesis if deferred.

**What was remeasured this turn.**

Run `OMP_NUM_THREADS=1 python3.11 -B tools/top10_preflight.py`. It reproduces the old
fill baseline/treatment RMSEs using the stored delta scale and identical positivity
floors, then audits unchanged predictions and all thirteen stand caches. The output is
[reports/top10_preflight.json](../reports/top10_preflight.json).

| Finding | Evidence | Decision |
|---|---|---|
| The +1,782 is not a measured candidate gain | It is the old blend's improvement on true-fill rows, offset by ~1,719 on non-fills | Do not bank it or assume a new body expert preserves it. |
| The 12× control result belongs to the ~322-MSE gated rule | PREREG_fill_gate RESULT G | It supports the tested gate's predictive value, not recovery of +1,782 by a future model. |
| Rome carries 80.74% of the fill-branch improvement | Rome +1,438.8 of +1,781.9 weighted MSE; other airports together ~343.1 | Keep the Rome-inclusive conditional expert primary. The already-registered exclusion is a fallback, not the main top-ten thesis. |
| Pooled probability calibration is much closer than the narrative suggests | Top decile: mean p 0.41081, actual 0.41556; overall mean p 0.08551, prevalence 0.08898 | Calibration may help, especially within airports, but it is not a demonstrated large lever. |
| `ars` already contains the airport | 2,401,991 / 2,401,991 rows across all twelve training caches and ranking equal airport + stand + runway | Adding `ap_ars` renames the same groups; its two target encodings are redundant. |

`E[p | true fill] = 0.297` does not establish miscalibration. Calibration asks whether
`E[F | predicted p]` is approximately p. Even a perfectly calibrated predictor can
have low recall at p > 0.5. The pooled deciles do not prove conditional calibration
on expensive rows either. See the [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

**The next two substantive bets.**

1. **Conditional experts, including Rome.** Use the corrected encoding pipeline and
   the baseline fit already inside `cond_experts.py`. Obtain one development result
   on full training volume before paying for three seeds. Evaluate `cond` and the
   existing `cond_ex` together as registered. Read pooled MSE and Rome's contribution,
   not just improvements conditional on labels. Keep p uncalibrated in the primary
   run, as already registered; any calibration follow-up uses inner OOF predictions.
2. **Native-categorical CatBoost with numeric clocks and queue features.** This remains
   an untested representation and needs its own runner/input path; the existing
   float-matrix CatBoost worker is not this experiment. Retain coarse airline and
   airport keys alongside full flight/callsign, scoped stand/runway, operator and route
   interactions, with counts/fallbacks. A weak flight-number mean tested alone does
   not settle its marginal contribution in this model. Keep squared loss and the delta
   target. Evaluate the candidate and one blend selected on inner OOF predictions.

Do not place a long sequence of encoder/scoping fits in front of both bets. Establish
one corrected reference, reuse identical reference predictions with a data/feature/
split/parameter manifest, and run the candidates against it. The full-fit baseline in
`cond_experts.py` currently repeats work that an identical `enc_ab` reference may
already have done. Reuse is valid only when feature order and every fitting choice
match, not merely the feature count.

**Runner issues to repair before expensive fits.** These are findings from code review,
not failed model hypotheses. Do not silently change locked comparisons; append a short
implementation correction before the affected runs.

- `enc_ab.py --screen` returns after early stopping and writes no independent
  validation prediction/score. Its stopping metric is explicitly NOT_A_RESULT; the
  incumbent and repaired encoders also differ in whether stopping labels enter
  features. This cannot fairly select the best two as currently described. Either
  give it a separate screening validation month outside fit and stopping, or bypass
  it and run the necessary reference/candidate development fits directly.
- `enc_ab.py` emits paired comparisons only against `incumbent`. H-E2 needs
  **scoped versus separated**, including when those are the only two selected variants.
- Both new harnesses currently discard per-row predictions when writing JSON.
  Save stable row identities (or checked file/row indices), dates, true labels,
  per-seed final taxi-time predictions, component outputs and a manifest. Without
  these, date resampling, paired ensemble fitting and error-budget analysis require
  repeating hours of model fitting. Use explicit `pred_taxi_seconds` /
  `pred_delta_seconds` names and an artifact schema declaring target and floor rule.
- Provide a full-volume one-seed development mode. A 60-tree smoke is not a model
  screen; three-seed full fits should be reserved for promising candidates. Preserve
  the registered three-seed rule for confirmation, with per-seed *assembled mixtures*
  saved so candidate variability can actually be measured.
- Remove `ars` from new scoping additions, or explicitly label its duplicated columns
  as redundant. The other scoped keys remain legitimate candidates, but they have
  not demonstrated a gain inside the model. Changing the addition from ten to eight
  columns must be recorded before interpreting the experiment.

Also, the preregistered H-E1 rule “no significant loss” is not statistical evidence
of non-inferiority. For a new non-inferiority claim, define a tolerated loss margin
and require the confidence bound to stay within it. Keep existing literal verdicts
separate from that stronger claim. Fold A remains repeatedly inspected development
data; prospective registration of another variant does not make it a fresh holdout.

**Where a large enough gain could come from.**

These are budget targets, not expected outcomes or new oracle claims:

| Change on existing matched fold | Weighted MSE recovered if achieved |
|---|---:|
| Rome RMSE 384.07 → 350 | 1,898 |
| Rome → 325 | 3,179 |
| Rome → 300 | 4,365 |
| Rome → 280 | 5,245 |

A Rome improvement toward 300 plus roughly 2,500 MSE elsewhere is the scale of a
top-ten candidate under the transfer assumption. This is why Rome-specific
representation and conditioning are more relevant than automatically excluding Rome.
The rest can come from other matched airports, the unmatched estimator or a measured
complementary ensemble. Those gains must be scored jointly; overlapping gains do not add.

**A bounded 48-hour program once compute is available.** Timing is an allocation, not
a measured runtime guarantee:

| Window | Work | Required output |
|---|---|---|
| First 2 hours | Finish F readout; fix runner output/comparison issues; freeze development and confirmation splits | Usable saved predictions and an explicit run manifest; no broad test expansion. |
| Next available fit slots | One corrected reference and conditional-expert development run | Net MSE gain, Rome/non-Rome contributions, complete predictions. |
| Remaining first day / second day | Native-categorical candidate; shortlist one optional sweep setting or scoping addon | At most two promising candidates for confirmation. |
| Within 48 hours if fits allow | Three-seed confirmation and a joint ensemble on independent months | A measured gap-closure figure against v5, with date-block uncertainty. |

Build the categorical inputs/runner while the first experiment runs if the work stays
below the one-heavy-job resource constraint. Do not load another multi-GB model beside it.

For new development runs, prioritize candidates with at least 1,000 MSE improvement;
retain smaller results as maintenance opportunities. This is a compute allocation rule,
not a claim that a weaker candidate is disproven. Confirm on months not used to select
the candidate and add a forward-time stress test. Keep both January and July reported.

If the best complete candidate gains ~7,000 or more and replicates, it has evidence
at the scale of the snapshot top-ten gap. If it gains 1,000–3,000, it does not yet.
Do not spend another night refitting tiny variants to manufacture visible movement.
At that point direct the next research slot to the largest remaining reproducible
residual pattern, with a measured error budget and genuinely new input or model capacity.

The 232-MSE blend would change 288.14 to only about 287.74 **if** its gain transferred.
Even 1,782 would reach only about 285.03. Neither justifies saying the top-ten path is
already solved. The two larger bets above are the next tests that can establish one.
