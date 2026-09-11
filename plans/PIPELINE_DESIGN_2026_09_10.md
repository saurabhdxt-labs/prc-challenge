# One pipeline, raw data → submission — design (boundary first, then code)

**Owner mandate 2026-09-10:** one end-to-end pipeline; every airport is a config entry; "everything goes through the
pipeline"; model chosen by measurement — "if tree works better then tree it is, but we have to be able to handle it
properly". Research inputs arriving: `reports/RESEARCH_TAXI_FEATURES_2026_09_10.md`, `reports/FEATURE_IMPORTANCE_2026_09_10.md`.

## Why (the root cause this fixes)

v10 exists only as a chain of guarded splices — v5 (matched LightGBM) → v6 (+ order block) → v7 (+ Rome R2) → v9 (+ E3C
congestion body) → v10 (+ E1 floor) — each script reading the previous submission file. It cannot be regenerated in one
run; the review found every rebuild guard now fails on v10 itself; a new scored file (a possible final evaluation set,
as in PRC 2024/2025) would need six manual steps (`lgbm_submit --base` defaults to v2 — a landmine). Prize eligibility
needs reproducible GPLv3 code. A patch stack is exactly what the no-patching rule forbids.

## Architecture (package `prc/pipeline/`; scripts become thin CLIs over it)

```
config ─► ingest ─► derive ─► features ─► lanes ─► assemble ─► validate ─► write
  │         │         │          │          │          │           │          │
airports  schema    clocks    registry   models    one frame   contract   file +
registry  checks    (UTC/     of named   behind    per scored  checks     manifest
(yaml)    (fail     local)    blocks     ONE       row, every  (template, (git sha,
          loud)               (fold-     interface lane's       ranges,    config
                              safe)                 output +    per-lane   hash,
                                                    provenance  row counts) inputs)
```

1. **Config** — `configs/pipeline.yaml`: airports (code, IANA time zone, enabled lanes, airport rules such as LIRF's fill /
   date-slip / schedule-floor), data paths, fold definitions, model choice per lane, seeds. Validated at load (schema,
   types, allowed values) — config is code (Part 1 rule 11). A new airport = a new entry + its data; lanes it has no
   history for fall back to the pooled model and are flagged in the manifest.
2. **Ingest** — raw parquet → canonical frame; declarative schema per boundary (columns, dtypes, ranges, nullability);
   refuses unknown airports; never reads BLOCK / TAXITIME on scored rows (the serve-time contract, tested).
3. **Derive** — clock features (`build_submission.derive`); **`dayoff` computed in the airport's LOCAL day** (review
   finding D1: UTC mislabels ~9% of LIRF rows) — introduced only as a registered change AFTER reproduction.
4. **Features** — a registry of named blocks (clocks, queue, order, stand history, witness, encodings, ...), each a pure
   function with declared inputs, outputs and a leakage class (`serve-safe` / `fold-fitted`). Target-reading statistics
   are fitted strictly on the training mask, never on stopping rows (BC-1 repair from `prc/encoding.py`, wired here).
5. **Lanes** — each lane owns its rows: `matched` (NM record), `unmatched` (no record), airport rules (LIRF). Each lane
   holds a **model plug-in** behind one interface:
   `fit(X, y, masks, seeds) → state`, `predict(state, X) → ŷ`, `save/load(state)`, `describe() → dict`.
   Implementations: LightGBM (incumbent), CatBoost (venv worker), NN (tabular), Blend (weights fitted on inner OOF only).
6. **Assemble** — one frame keyed by MVT_ID with every lane's prediction, its provenance (lane, model, stage) and the
   final value; no splice of a previous submission file anywhere.
7. **Validate** — `check_submission` + range checks + per-lane row counts + "each airport has predictions" + manifest.
8. **Write** — submission parquet + manifest JSON (git sha, dirty flag, config hash, input file hashes, per-lane
   provenance counts, timing). Every verdict / report produced by the evaluation harness carries the same stamp.

**Evaluation harness** (same pipeline code, fold mode): fold A and 12-month LOMO; paired **day-block** intervals for
every comparison (lesson of 2026-09-10); per-airport, fill / tail / ordinary cuts; locked clauses read from a prereg
file; candidates (models, feature sets, blends) compared identically and selected before any board reading.

## Acceptance (in order; each gates the next)

- **A1 — reproduce v10.** Unmatched + LIRF lanes reproduce v10's values **exactly** (they are deterministic; the existing
  ship guards prove it per stage). The matched lane reproduces v10's matched values **exactly from the saved v6 boosters**,
  and within a stated tolerance on a full refit (LightGBM at 4 threads is not bit-deterministic — `lgbm_fold.py` docstring);
  if the tolerance is not met, the refit runs deterministic / single-thread. The assembled file equals v10 on every row.
- **A2 — one command** produces the submission from raw data; a second run produces the same manifest hash.
- **A3 — a synthetic 11th airport** (fixture) flows through end-to-end with fallback lanes and a manifest flag.
- **A4 — tests:** every module unit-tested (happy / bad / edge), mutation-rehearsed; the serve-time contract and the
  shape of the manifest tested; the existing 4 deliberate failures stay deliberate.
- Only after A1–A4: registered changes (known defects: local-day `dayoff`, p on sp < −60, CAP setting, BC-1 wiring),
  feature engineering from the research, and model candidates (CatBoost blend, NN) — each through the harness.

## Pre-mortem (what could break, and the guard)

| risk | guard |
|---|---|
| the rebuild silently differs from v10 on some lane | A1 compares every row per lane and fails loud with the differing ids |
| a feature reads a label at serve time | ingest/derive contract test on the scored file (existing `test_serve_time_contract` pattern) |
| BC-1 repair changes `best_iter` and so the predictions | wired only after A1, as its own registered change, measured |
| a new airport with no history crashes a lane | fallback path + fixture airport test (A3) |
| two sessions edit the same modules | new package, new files; ADS-B stays prc-challenge-6e's; announce file ownership |
| NN beats trees on the fold by chance | the same day-block clauses and a replication fold; blends on inner OOF only |
| heavy refits collide | one heavy job at a time, AC power, quality gate |

## Build order and ownership

P2a skeleton + config + model interface + stage wrappers around the existing tested functions (no behaviour change) →
P2b A1 reproduction (unmatched + LIRF exact; matched from boosters) → P2c full-refit reproduction (heavy) → P3 feature
engineering (registered, from the research) → P4 NN (after `/ml-preflight`) and blends. New files only; existing
scripts keep working until the pipeline supersedes them.

## A1 — MET · 2026-09-11 08:19:56 (from `date`), session prc-challenge-c4

`PRC_A1_FULL=1 pytest tests/pipeline/test_pipeline_real_files.py -k one_command`: `cli.run` on `configs/pipeline.yaml`
(matched lane from the three saved v6 boosters, provenance-checked per booster; stratum lanes refit from data; no
submission file among the inputs) → `data/pipeline_runs/a1_full_20260911T114553Z/` (manifest v2, digest 9aede4c4ae2a…,
23 code files hashed, 12 not yet in HEAD). Compared by the TEST with `submissions/merry-quicksand_v10.parquet` (sha256
18c911de…b348): **344,841 / 344,841 rows equal** — matched 339,551, unmatched 4,907, LIRF 383, max |diff| 0 s. Wall
1,414 s, peak RSS 4.06 GB, the only heavy job, AC power, quality gate YELLOW (standing waivers b, c, f).
Before the run: the four weak tests the review found were replaced and rehearsed (each old version SURVIVED its targeted
break, each new one goes RED), `check_prediction` guards both lane boundaries, booster provenance fields are required,
the manifest hashes the code that ran. **Open:** A2 (a second run → the same digest), A3 on real caches (the 11th-airport
test still passes only on hand-built caches), the P2b list, write-before-manifest ordering (`write.py`).

## P3 — the multi-model pipeline (owner decision 2026-09-11: "it should be a multi model pipeline … the only thing that can take us to 245")

**Why (measured, not assumed).** Arm F's matched error by clock regime: clocks agree 84% of rows → 50% of matched MSE;
fill 8.9% → 15%; **airport off-block > 10 min before NM's 3.8% → 29% (RMSE 616, mean error +343 s)**; after NM's 3.0% →
6%. The unmatched lane is already a two-regime mixture (fill gate × body) with its own row classes. One global model per
lane averages data-generating situations that need different predictions; every lever added to that one model has been
worth hundreds.

**Boundary first — the contracts (each a module with its own tests):**

1. **Regimes are config.** `configs/pipeline.yaml` gains `regimes:` per lane — name, a label rule on TRAINING rows only
   (matched: `fill |y − sp| ≤ 60`, `early δ < −600`, `late δ > +600`, `agree` otherwise; unmatched: fill / date-slip /
   body), and which expert plug-in serves each. A lane with one regime `all` IS today's lane (the degenerate case).
2. **Gate plug-in** (`ModelPlugin`, output `(n, K)` probabilities summing to 1): multiclass LightGBM first; calibrated
   per class on INNER out-of-fold predictions (isotonic, renormalised) — never on the holdout, never on its own stopping
   rows (the fill-gate lesson: an uncalibrated `p` recovered 18% of its prize).
3. **Expert plug-ins**: any `ModelPlugin`, fitted ONLY on its regime's training rows (RESULT 8: an all-rows expert inside a
   mixture double-counts), each with its natural target (delta / taxi / `sp` copy).
4. **Mixture lane**: `ŷ = max(Σ_k p_k μ_k, 1)`, `check_prediction` at every boundary, provenance per row = the regime
   probabilities and each expert's value (component attribution: which part carried each row).
5. **Stack plug-in**: combines candidate lane predictions (single model, mixture, CatBoost, NN) with weights fitted on
   inner OOF only; a candidate with weight 0 is dropped from the ship build.
6. **Fold-mode harness** (`prc/pipeline/evaluate.py`, the piece the design always named and never built): for a config,
   produce inner-OOF predictions on the training months (month-grouped inner folds) and outer fold-A / LOMO holdout
   predictions for every plug-in; records stamped with their convention (BC-2) and the code hashes (BC-7); scoring with
   day-block intervals and the locked clauses read from the prereg.
7. **Feature blocks** stay config: the ADS-B table (prc-challenge-6e's `data/adsb/v2/features_<period>.parquet`, keyed by
   MVT_ID, with a coverage flag) enters as a block — first as a GATE input, where it is worth most (an aircraft seen
   leaving the stand is the early-off-block regime observed directly).

**Acceptance, in order (each gates the next):**
- **M1** — the degenerate config (one regime, one expert = today's model) reproduces A1 exactly on all 344,841 rows.
- **M2** — fold mode reproduces arm F's stored fold-A record for the single-model config (222.5632) and E3C's LOMO
  record for the unmatched lane.
- **M3** — gate / expert / mixture / stack plug-ins unit-tested with planted-signal and null tests, mutation-rehearsed
  (`scripts/regime_experts.py` is the screen-tier prototype; its 8 tests and 11 killed mutants port with it).
- **M4** — every candidate is a registration with locked clauses, measured in fold mode; ship only through the pipeline.

**Build order.** evaluate.py (single-model, M2) → gate / expert / mixture plug-ins from the REG prototype (M3) → inner-OOF
calibration and the stack plug-in → the unmatched lane re-expressed as the same mixture → the ADS-B block when its table
lands → ship build under A1-style guards. The REG screen (running 09:30) decides only which matched config is first:
a SHORTLIST makes REG the first multi-model matched lane; a NOT SHORTLISTED says today's features cannot drive the gate
and the tail needs the new information (ADS-B) the architecture is built to take.

### P3 interface detail (written 2026-09-11 before the code)

- `Design(X: float32 ndarray, proxy: float64, sp: float64, row_ids)` — what a mixture-kind plug-in receives; matrix-kind
  plug-ins keep receiving `X`. `Labels` gains `regime: int64 | None` (training rows only; −1 elsewhere, refused if read).
- `regimes.MATCHED = (agree, fill, early, late)` with the label rule in ONE function (`regimes.label_matched`), shared by
  the harness and every test; config names which regimes a lane uses and which expert serves each.
- `check_probabilities(model, P, n, k)` at the gate boundary: shape `(n, k)`, finite, non-negative, rows sum to 1 (±1e-6).
- `MixtureModel(gate, experts: {regime: (plugin, scale)})`, scale ∈ {`delta`, `taxi`, `sp_copy`}; `fit` trains the gate on
  `Labels.regime` and each expert on `masks ∧ (regime == k)` only; `predict(state, Design)` = `max(Σ p_k μ_k, 1)` with
  every μ recovered to taxi time inside the plug-in and checked by `check_prediction`; `describe()` lists its parts.
- Every stage that can bind a floor reports how often it did (the late expert binds on short proxies by design).

### P3 status · 2026-09-11 10:10:37 (from `date`)

- **Built and tested (not yet committed):** `prc/pipeline/scoring.py` (paired weighted-MSE pricing, day-block interval —
  equal to the recorded diagnostics' — convention-stamped fold records; 7 tests, 7 mutants killed);
  `prc/pipeline/regimes.py` (regime rule, gate boundary check, `Design`, `GateModel`, `MixtureModel` with a named
  minimum-rows refusal; 11 tests, 9 mutants killed); `prc/pipeline/evaluate.py` (fold harness: strict candidate schema,
  single and mixture candidates, stamped records with candidate + fold + code hashes; 13 tests, 8 mutants killed);
  `Labels.regime`; `legacy.fold()`. Pipeline suite 182 passed, 2 gated skips.
- **M2 (unit form) MET:** `evaluate.predict_single` equals `lgbm_fold.fit_arm` bit for bit (per seed, pooled, best_iter,
  n_ref) on the same synthetic fold at one thread.
- **M2 (real data) NOT RUN:** reproducing arm F's stored record (222.5632) is lr 0.01 × 3 seeds ≈ 2 h, ≈ 5 GB — needs the
  owner's go for the heavy slot. **Open:** the separated (BC-1-clean) encoder in fold mode (REG used it via enc_ab); LOMO
  folds; inner out-of-fold predictions (for gate calibration and stacking); a CLI subcommand; M1 (degenerate mixture ==
  today's lane); the ADS-B block (awaiting prc-challenge-6e's coverage).

### P3 — the ADS-B stack stage (written 2026-09-11 11:26:59, before any code; interface to be confirmed with prc-challenge-6e)

Why: 6e's ADN stacker beats arm F by +5,146 weighted fold MSE [+4,720, +5,618] on fold A (re-priced independently; day-level
OOF folds verified). It is the first lever the size of the gap to 10th. Ownership: 6e owns the ADS-B table and the ADN model;
the pipeline owns running it, guarding it and writing the submission.

- **Feature block** `adsb` (config: `lanes.matched.feature_blocks`): the table `data/adsb/v2/features_<period>.parquet`, joined BY
  MVT_ID (never positionally), with a per-row `coverage` code; the reader refuses a table whose metadata lacks git sha, extractor /
  feature versions and source licence (ODbL attribution kept), and refuses a label column.
- **Stack stage** (a `ModelPlugin` of kind "stack", matched lane only): input = the matched lane's F prediction (delta), proxy,
  the ADS-B block; output = the final delta → taxi via the lane's floor. Rows with coverage < 2 pass F through UNCHANGED (ADN's gain
  on uncovered rows is exactly 0 by construction; asserted). Learners run in their own processes where required (torch never with
  lightgbm — BC-5). Artefacts loaded through save/load with provenance (sha256 sidecars, like the body plug-ins).
- **Guards:** (1) stage disabled → the pipeline equals v10 on all 344,841 rows (A1); (2) stage enabled on 2025 fold-A inputs →
  reproduces 6e's stored OOF record `adn_fold_preds.parquet` within a stated tolerance; (3) 2026 coverage shares per airport
  reported before any upload and compared with 2025's (the gain transfers only where coverage holds).
- **Ship registration** (before any 2026 prediction exists): bar, pricing (2025 per-airport gain × 2026 coverage), and the organiser
  question's status (`reports/ORGANISER_QUESTION_DRAFT_2026_09_11.md`) named as a precondition the owner signs off.

### ADS-B stack stage — status 2026-09-11 11:34:37 (from `date`)

Built and tested (not committed): `prc/pipeline/adsb_stage.py` (strict table + gate readers, stage frame, apply with the contract's
guards; 14 tests, 8 mutants killed); config `lanes.matched.adsb_stage: {table, models_dir}` (optional; absent = today's lane, config
hash unchanged; refused on non-matched lanes); `lanes.run_matched_lane(..., stage_predictor=None)` applies the stage after the base
prediction, marks moved rows `+adsb` in provenance, records per-airport counts, gate and model manifest in the lane info, and refuses
if its base differs from what the lane ships; `legacy.adsb_predictor()` imports 6e's `adsb_stack.predict_residual` read-only (not yet
present — raises NotImplementedError until 6e ships it). 3 wiring tests (every rehearsed break RED). Pipeline suite 199 passed, 2 gated.
Waiting on 6e's artefacts: `data/adsb/v2/features_2026janjul.parquet`, `data/adsb/v2/models/` + MANIFEST.json with the gate table.

### ADS-B stack stage — status 2026-09-11 11:53:08 (from `date`)

Adapted to 6e's live contract: `read_gate` accepts `{ap: {allowed, rule, c25, c26}}` (an EMPTY gate is refused — it means
`adsb_ship.py gate` has not run on the 2026 table); the lane calls `predict_residual(frame, models_dir, oof=False)` (the full bag;
oof=True is the 2025 guard only). `legacy.adsb_stack()` is the one read-only import of 6e's module. Guard (2) PASSED on the real fold
(ledger 11:53:08: max |Δresid| 1.592e-12 s, RMSE 210.41984768 = RESULT ADN arm B). Guard (3) is `adsb_stage.coverage_transfer`
(per-airport movable / from-the-stand shares, 2026 ÷ 2025 ratio); its 2025 half is in the ledger, the 2026 half waits for the table.
Tests: stage 16, lane wiring 4 (every rehearsed break RED), gated guard 1. The guard's own rehearsal (stage_frame fed hour + 1) went
RED at 61.8 s. Still to do: the 2026 coverage report, the ship prereg, owner sign-off.

### ADS-B stack stage — status 2026-09-11 13:47:42 (from `date`)

- **The dry run checks the stage** (`adsb_stage.check_stage_inputs`, called from the matched lane's input check). It covers:
  - the table: stamps, exact routed ids, coverage codes;
  - the gate: non-empty, every gated airport among the manifest's airports;
  - every seed × fold model file (G / N / stats);
  - the manifest's features, which must be in the table;
  - `ap` / `hr`, which must be in the ranking cache.
  Rehearsed: each check removed -> RED.
- `configs/pipeline_adn.yaml` (6e, SHIP-ADN) = `pipeline.yaml` + the stage block only (diff checked).
- 6e's `scripts/adsb_v11_check.py` is the independent guard set (a)–(e). It takes proxy / hour / ap from the raw ranking file, so its guard (c) also cross-checks the stage's stand-cache join on 2026.
- BC-8 (shared-tree race in the two-run digest tests) fixed test-side; pipeline tier 206 passed, 3 gated.
- Next: the real dry run on `pipeline_adn.yaml` once the 2026 table and the gate exist; then the v11 build (the only heavy job, AC power, quality gate), 6e's v11 check, and the owner's upload decision.

### Rome rule RLD in the pipeline · 2026-09-11 17:47:07 EDT

The LIRF rules lane gains a fourth, last rule, `local_day_schedule {lo_s: 24000, hi_s: 86400}`
(`plans/PREREG_rome_local_day_rule_2026_09_11.md`): rint(sp) on LIRF unmatched rows in band whose take-off is on the
schedule's Rome LOCAL day.
- **Implementation:** it calls prc-challenge-53's `scripts/rome_local_day.apply_rld` read-only (`legacy.rome_local_day()`).
  - `check_pins` re-reads its SP_LO / SP_HI / AIRPORT.
  - The rule refuses a lane whose sp is not MVT − SCHED.
  - Provenance marks moved rows `+RLD`.
- **Tests:** config parse, band pin and order; rule semantics on five crafted rows; the sp guard; the pins; wiring on the
  synthetic stratum world. 6 of 6 rehearsed mutations RED. Pipeline tier 211 passed.
- **Real check:** the rules lane alone under `configs/pipeline_v12.yaml` changes exactly the prereg's 3 rows to its values.
- **v12** = the one-command build on `configs/pipeline_v12.yaml` (= `pipeline_adn.yaml` + that line).
