# FM1 — limitations (first written 2026-09-28; updated 2026-10-04 after the line-by-line audit)

1. **Validation is a reference point, not a score.** The 2025 validation (METHOD §6) runs at the screen tier with the ADS-B stack
   off; the full-tier configuration has no clean 2025 fold. Its only 2026 measurements are the public board (v50 251.4767,
   v55 249.9168; METHOD §8), which is also where every board-kept step was selected. No number has a run-to-run variance
   estimate: each board file is one run of the three-seed ensemble (`run.seeds` 0, 1, 2) and the screen tier uses one
   seed, so board differences of a few hundred MSE may be within run-to-run noise.
2. **Unported components.** The storm-as-feature form (owner decision 2's LOEO re-validation) is declared and refused by
   `fit` when switched on. ARN (v21) is DROPPED: its 8 scored 2026 rows keep their Rome / E2 values. (adsb.fi was ported
   on 2026-10-03: `adsb.sources.adsb_fi`, serve table only; item 12.)
2b. **LDI's constants are not refitted.** FM1 applies LDI's registered shifts (8.0 / 35.4 / 122.5 s; owner decision
   2026-09-29) and computes the target rows itself. Re-deriving the shifts from raw data needs two extra held-out fits
   (February out; December + March out) and is not done. The severity cut (late share >= 0.5, min temperature <= 1 C) was
   chosen after looking at 2025 February (the registration's provenance warning); it fired on 4 January-2026 days.
   The day test reads every LTFM departure of a local day: predict must receive whole days (PartialDayError otherwise).
3. **The unmatched lane is the shipped chain, selected in-sample.** The v26 body, the Rome rules and ARC were chosen after
   reading 2025 labels of all months or of the Jan/Jul holdout (PREREG_urp_u_chain_replay's FIT / SELECTION table: 12 of 13
   steps in-sample); E2's coefficients and the two day-check p99 thresholds are frozen 2025 constants. A validation fold
   therefore overstates what an unseen month would show for these components. The provisional `hgb` body is kept only as a
   non-default alternative and has no replay.
3b. **ARC outside LIRF is unvalidated.** `modules.arc.airports` defaults to LIRF (the shipped scope); the FMA's all-airport
   ARC needs a LOMO outside Rome first.
3c. **Set-level unmatched rules.** The witness, the storm anomaly, ARC's ARR join and E2's per-day checks read other scored
   rows, and E2 reads LIRF arrival in-blocks from `data.raw_dir` itself (`ranking.parquet` for 2026; guarded): serve the whole
   scored file together. The lane's artefact stores its training frames and re-runs the shipped fit-and-predict estimators
   at predict time (seeded, bit-reproducible; ~20 s).
3d. **E2W needs two training months with LIRF rows** (its beta_W is fitted leave-one-month-out): single-month folds (the
   strict swap) must run with `modules.rome.e2w=false`.
4. **Stored stack, no per-fold retraining.** The ADS-B stack is served from the R12 stack models (trained on the Jan + Jul
   2025 records). Validation folds testing January or July must run with the stack off (the leakage guard refuses otherwise),
   so no fold number yet covers the stack.
5. **R12 base F cannot be replayed from models.** The R12 refit saved predictions only; AT14's stored tier feeds its F in.
   FM1's own artefacts save every model.
6. **Blind airports and the early-off-block family** (LTFM / LFPG ground coverage ~0; ADS-B coverage drift 2025→2026;
   EGLL-July ESS 0.6 %): the support gate is the stored ADN.7 gate until the label-free ESS rule is set on 2025 LOMO.
7. **LIRF recording regime.** Rome's labels follow a handler-recording regime (BLOCK = STD on 14 %); a hidden period can
   change it. The Rome module is documented as modelling that regime, not taxiing.
8. **MTX is hindsight information** about the target period (published after the month closes); it is off in the
   forecaster and on only for the scored-period build. It keys scored rows by the UTC month of MVT while the published
   cells are local months (as v42 did).
9. **2026 drift (label-free audit of the 133 inputs, 2026-10-02).** At **LFPG in July 2026 the EOBT_1 field changed how it is
   produced**: the share of matched departures with EOBT_1 later than LOBT is 3.8 – 7.1 % in every 2025 month and 6.1 % in
   January 2026, but 35.1 % in July 2026 (median AOBT_3 − EOBT 600 s → 120 s while delays fell). `aobt_eobt`, the base
   model's largest input (19.9 % of split gain), therefore means something else there. The cell's mean prediction is on
   the published monthly target (gap 2.7 s), so the damage is row-level. Also flagged: LTFM January (later off-blocks
   relative to schedule; operational, no action) and the schedule shift LTFM SCHED == IOBT 0.56 → 0.82. The audit's
   multivariate domain classifier did not work (row-wise folds memorise days) and was not used.
9b. **The EOBT clip is a repair priced on a simulation.** `modules.eobt_clip` (off by default) sets EOBT_1 := min(EOBT_1,
   LOBT) at serve time in an airport-month whose field carries revisions (share ≥ 0.15 on ≥ 1,000 departures; LFPG only).
   Its value was priced on a held-out 2025 July with the 2026 practice imposed: about +195 … +290 bmse, LightGBM only; the
   simulator matched the revised share or the feature distributions, never both, so the price is not established. It
   assumes the practice changed only the field (not the flights), does not price the CatBoost half, and uses an absolute
   threshold at a named airport; a detector relative to each airport's own training share would need a training
   statistic stored in the artefact (a refit). **Measured on the board it lost:** v54 (v51 + clip) 251.5593 vs v51
   251.1335, −214 MSE. The clip lifted LFPG-July's mean prediction from 983 s to 1014 s against a published monthly mean of
   986 s, which already said the unclipped model had that cell's mean right. So the 2026 revisions seem to carry information
   the model reads correctly; the field change stays an open limitation, the module stays off, and a refit that treats
   revised EOBTs as their own input is the untested remedy.
10. **Serve rows without an ADS-B table row** are served as "no ADS-B data" (coverage 0) up to 0.1 % of the scored matched
    rows (measured 2025 rate 0.014 %); above that the run refuses (`adsb.serve_missing_max_share`).
11. **Stack residual rounding** depends on the batch size at the 1e-7 s level (float32 MLP); a row within 4e-7 s of a .5
    boundary could move by 1 s.
12. **adsb.fi cannot be re-fetched** (the owner's live capture of July 2026; non-commercial terms, never redistributed),
    so v55 is reproducible only on the machine that holds the capture. The challenge's eligibility page asks for external
    data that is openly accessible and usable and for reproducible results; whether a non-commercial live capture meets
    that is the organiser's call. v51 (251.1335) is the same model without it. Every build also needs the stored ADS-B
    stack models, which the current code cannot regenerate byte for byte (REPRODUCE.md).
13. **The CatBoost half is under-trained.** In the final fit CatBoost reached its round cap (best iteration 29,993 of
    `matched.catboost.max_rounds` 30,000 at learning rate 0.02; early stopping never fired). A higher cap needs a new full
    fit (~10 h). The board still measured the half-blend as worth +166 MSE (v53 vs v50).
14. **ADS-B coverage moved between 2025 and 2026** (share of matched departures with ground coverage, stack training months →
    2026: EDDF 0.909 → 0.623, EGLL 0.384 → 0.902, LIRF 0.405 → 0.299; METHOD §6). The stack keeps its stored gate; the
    label-free `coverage_ratio` gate would switch EDDF off, and the board measured that as no better (v52, −10.6 MSE).
    The flips are large by airport-month (LIRF July 93.5 % → 0.2 %, EDDF July 89 % → 59 %; EGLL July 21 % → 98 %, LEMD
    January 0 % → 97 %). Imposing July 2026's losses on held-out July 2025 costs the model about 860 bmse (LIRF 585, EDDF
    278; the ACR registration in the development record). Training with ADS-B randomly hidden (`adsb.train_dropout.*`, off) recovers almost none of it
    and costs accuracy where ADS-B is present: the loss is information, not calibration, so only another track source can
    restore it.
15. **Full-fit memory.** The full fit holds ~15 GB in the main process plus ~5 GB in the CatBoost worker; the dry-run memory
    gate measures neither the grown models nor the worker. Plan for ~20 GB or run under a memory guard (REPRODUCE).
16. **The stored ADS-B stack is served on a differently made F.** The lgbcat stack models were trained on
    F = 0.5 × the v36 LightGBM holdout (older ADS-B tables, 125 features, learning rate 0.01) + 0.5 × a CatBoost holdout at
    learning rate 0.05 (the R13 registration's amended full tier). FM1 serves them on its own 0.5 × LightGBM (133 features)
    + 0.5 × CatBoost at learning rate 0.02 (which reached its round cap, item 13). F is an input of the stack, so its
    residual is applied to an F distribution it did not see in training. The only bound is a label-free check against v46
    (30.5 s RMS per airport-month after removing each cell's mean, registered limit 81.6 s). Retraining the stack on
    FM1's own out-of-fold F is the remedy; it changes predictions and was not done.
17. **In-sample choices.** The 24 target encodings (`te_*`, `de_*`) are smoothed label means fitted on every matched
    training row, including the early-stopping months and the rows the learner then fits (as the shipped pipeline
    did): each row's encoding contains its own label at weight 1/(n+50), which flatters early stopping on thin keys. The
    unmatched lane's frozen constants (E2 coefficients, the E1 / date-slip / local-day rules, ARC, the two day-check p99
    thresholds, LDI's shifts) were chosen on 2025 data that includes the months the validation folds hold out
    (`scripts/analysis/urp_replay.py` labels those steps IN-SAMPLE). Validation numbers for those components overstate
    what an unseen month would show (item 3).
18. **Provenance of the stored artefacts.** v55's artefact was derived (`fit --reuse-from`) at 02:39 EDT on 2026-10-03
    on 33ee40a plus uncommitted changes that were committed 22 minutes later (0709298); its donor (v50) was fitted at
    03:16 EDT on 2026-10-02 on b285bda plus uncommitted changes committed later that morning (32aadc1). The artefacts
    record the git sha and a dirty flag, not the uncommitted diff. The stored stack's MANIFEST records sha
    fdeb704 with "-dirty", a commit that predates its producer scripts. The rebuild check in REPRODUCE.md (the current
    code reproduces the v55 file from its artefact) is the evidence that the committed code serves what was uploaded.
19. **A training file with a one-row spill-over month.** FM1 splits each raw file by the UTC month of take-off; six of the
    twelve 2025 files hold one row stamped 23:59:5x of the previous month. The shipped feature builder asserts more than
    100 stand keys per chunk, so a matched departure in such a one-row chunk would stop fit or predict. The 2025 rows there
    are two unmatched departures and four arrivals, and the 2026 scored file has no such chunk; a new data drop could.
20. **adsb.fi join checks have two blind spots.** The AFI acceptance's join-quality gate measures take-off time
    agreement, which is zero by construction for MVT-anchored joins (1,809 of the 11,690 newly joined July rows), and
    it never examines rows that were joined before and joined differently after the merge (501 callsign changes, 666
    first-seen times more than 30 min earlier). Misjoins to a previous rotation may remain among the July serve rows.
21. **Partial batches are refused only by LDI.** LDI refuses a partial local day (PartialDayError); the unmatched lane's
    set-level rules (item 3c) have no such check, so serve the whole scored file.
22. **The stand-exit stage (`uad`) reaches few rows, and part of it is thinly supported.** It needs the aircraft's ADS-B
    track on the ground at a stand whose position is known, so it gives an estimate to 503 of the 5,290 scored departures
    without a flight-plan record (none at LFPG or LTFM, where the store has no ground coverage). Its evidence is one
    registered test on 317 January + July 2025 rows; those months hold no storm at an airport with ground coverage, so
    the storm weight is the registered floor (0.8) resting on 28 design-month rows, the storm-day hypothesis could not
    be tested on 2025 (230 of the 503 rows are EHAM in January 2026, a snow month).
    The offsets and weights were fitted on 2025 against screen-tier leave-one-month-out predictions, not against the
    final model. The registration was amended three times after the first build and before any label was read: the
    checks were enforced in code; a stop rule had missed because Schiphol writes the 777 freighter's type differently
    from ADS-B (fixed at the cause, the bar kept); and the join for these rows was changed from a ±30 s clock window to
    the take-off whose track left the row's stand. The table is a prediction-time input built from the ADS-B day store,
    which is not in the repository; a clone must rebuild it (REPRODUCE.md). On the board the stage has one evaluation:
    v57 (v55 with the stage on) scored 247.3187 against 249.9168, +1,291.9 MSE. That confirms the file as a whole on
    January + July 2026. It does not say which rows carried the gain, so the storm weight and the class-A2 factor
    still have no labelled test of their own, and the gain was larger than the range written down before the upload.
    The 2026 table is built from the adsb.lol day store only, without the adsb.fi capture of July 2026 that the
    matched lane of v55 uses; no code checks the table the stage is given against the registered stop rules.
