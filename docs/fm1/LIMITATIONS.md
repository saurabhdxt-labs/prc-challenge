# FM1 — limitations (known on 2026-09-28; the list grows with days 4–7)

1. **Not validated yet.** No FM1 score exists; no FM1 number counts until the AT14 replay gate passes (its stored tier on
   the matched AND unmatched strata; the refit tier is day 8).
2. **Unported components.** adsb.fi and the storm-as-feature form (owner decision 2's LOEO re-validation) are declared
   and refused by `fit` when switched on. ARN (v21) is DROPPED: its 8 scored 2026 rows keep their Rome / E2 values.
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
9. **2026 drift.** Schedule-field shifts (LTFM SCHED == IOBT 0.56 → 0.82), LFPG July-2026 EOBT drift, ADS-B coverage drift:
   only label-free drift checks can see them (TODO day 7 drift report).
10. **Serve rows without an ADS-B table row** are served as "no ADS-B data" (coverage 0) up to 0.1 % of the scored matched
    rows (measured 2025 rate 0.014 %); above that the run refuses (`adsb.serve_missing_max_share`).
11. **Stack residual rounding** depends on the batch size at the 1e-7 s level (float32 MLP); a row within 4e-7 s of a .5
    boundary could move by 1 s.
12. **adsb.fi cannot be re-fetched** (a live capture, July 2026 only; non-commercial terms).
