# FM1 manual mutation rehearsal (AT12) — 2026-09-28

testing.md Part 1 rule 22. Each mutant is ONE targeted edit to an FM1 source file, run against the named test with
`python -B -m pytest -x -p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1` (no stale bytecode, BC-16), then the file
restored byte-for-byte (verified afterwards: no mutant text remains in any source).

**Result: 48 / 48 killed (100 %).** Modules covered: knobs, refdoc, rows, learners, features, stages,
components, artifact, engine, protocols, tools/fm1_replay_gate.py.

| id | file | mutation (old -> new) | test | result |
|---|---|---|---|---|
| M01 | `knobs.py` | `if not (lo <= v <= hi):` -> `if not (lo < v <= hi):` | `test_every_default_validates_against_its_own_knob` | KILLED (0.4 s) |
| M02 | `knobs.py` | `if isinstance(v, bool) or not isinstance(v, int if integer` -> `if False or not isinstance(v, int if integer` | `test_wrong_types_fail_with_the_path` | KILLED (0.3 s) |
| M03 | `knobs.py` | `if p == "matched.learner" and v in CUT_LEARNERS:` -> `if False:` | `test_xgboost_is_refused_as_cut` | KILLED (0.3 s) |
| M04 | `knobs.py` | `v == "auto" and self.values["run.purpose"] == "scored_period"` -> `v == "auto" and self.values["run.purpose"] == "forecaster"` | `test_mtx_auto_follows_the_purpose` | KILLED (0.3 s) |
| M05 | `knobs.py` | `k.status == "not_ported" and k.activates(self.values[k.path])` -> `k.status == "not_ported" and False` | `test_not_ported_lists_every_enabled_unported_component` | KILLED (0.3 s) |
| M06 | `knobs.py` | `unknown = sorted(p for p in flat if p not in KNOBS)` -> `unknown = sorted(p for p in flat if p not in KNOBS)` | `test_unknown_key_names_the_dotted_path` | KILLED (0.3 s) |
| M07 | `knobs.py` | `if stray:` -> `if False:` | `test_es_months_must_be_training_months` | KILLED (0.2 s) |
| M08 | `knobs.py` | `if SUBMISSIONS_DIR in parts:` -> `if False:` | `test_a_path_into_submissions_is_a_splice` | KILLED (0.2 s) |
| M09 | `refdoc.py` | `for k in by_section[sec]:` -> `for k in [k for k in by_section[sec] if k.status != "not_ported"]:` | `test_generated_reference_has_one_row_per_knob` | KILLED (0.2 s) |
| M10 | `refdoc.py` | `return v if yaml.safe_load(v) == v and not any(c in v for c in "*:#{}[` -> `return v` | `test_rendered_yaml_parses_to_the_defaults` | KILLED (0.2 s) |
| M11 | `rows.py` | `FORBIDDEN_SUFFIXES = ("_sec",)` -> `FORBIDDEN_SUFFIXES = ("_s",)` | `test_forbidden_design_columns` | KILLED (0.6 s) |
| M12 | `rows.py` | `(np.isfinite(taxi) \| frame.BLOCK_TIME_UTC_mvt.notna().to_numpy())` -> `np.isfinite(taxi)` | `test_labelled_departures_are_refused_at_serve` | KILLED (1.3 s) |
| M13 | `rows.py` | `if not ids.is_unique:` -> `if False:` | `test_contract_violations_are_named` | KILLED (1.2 s) |
| M14 | `learners.py` | `return max(N_REF_FLOOR, int(best_iter * n_all / n_fit))` -> `return int(best_iter * n_all / n_fit)` | `test_lightgbm_refit_count_rule` | KILLED (0.5 s) |
| M15 | `learners.py` | `"deterministic": True,` -> `"deterministic": False,` | `test_lightgbm_is_bit_deterministic` | KILLED (2.2 s) |
| M16 | `learners.py` | `b.save_model(str(d / f"lgb_seed{s}.txt"))` -> `b.save_model(str(d / f"lgb_seed{s}.txt"), num_iteration=3)` | `test_lightgbm_save_load_is_bit_exact` | KILLED (1.8 s) |
| M17 | `learners.py` | `if (fit_mask & es_mask).any():` -> `if False:` | `test_lightgbm_refuses_bad_inputs` | KILLED (1.8 s) |
| M18 | `learners.py` | `rows.assert_design_clean(feature_names, "learner design")` -> `pass` | `test_lightgbm_refuses_a_forbidden_feature_name` | KILLED (1.6 s) |
| M19 | `features.py` | `return [c for c in cols if not c.endswith("_sec")]` -> `return list(cols)` | `test_base_columns_equal_nsc_fold_fw` | KILLED (1.8 s) |
| M20 | `features.py` | `enc = (g["sum"] + prior * smooth) / (g["count"] + smooth)` -> `enc = (g["sum"] + prior * smooth) / (g["count"] + 2 * smooth)` | `test_encodings_equal_stand_ab_infold_encodings_bit_for_bit` | KILLED (2.6 s) |
| M21 | `features.py` | `if tz_mode == "UTC":` -> `if True:` | `test_dayoff_utc_vs_local` | KILLED (0.5 s) |
| M22 | `features.py` | `out.loc[miss, "coverage"] = 0.0` -> `pass` | `test_serve_rows_without_a_table_row_are_no_data_up_to_the_share` | KILLED (1.1 s) |
| M23 | `features.py` | `if serve and share > max_missing_share:` -> `if False:` | `test_serve_rows_without_a_table_row_are_no_data_up_to_the_share` | KILLED (1.2 s) |
| M24 | `features.py` | `return table_features(table_dir) + ADSB_EXTRA` -> `return ADSB_EXTRA + table_features(table_dir)` | `test_matched_design_without_orl_equals_the_v36_r12_refit_design` | KILLED (1.7 s) |
| M25 | `stages.py` | `if self.calibrated_by is not None:` -> `if False:` | `test_ledger_refuses_a_value_write_after_calibration` | KILLED (0.5 s) |
| M26 | `stages.py` | `"""Refuse an unknown stage, stages out of the fixed order, and a value` -> `"""Refuse an unknown stage, stages out of the fixed order, and a value` | `test_check_order_refuses_a_value_stage_after_calibration` | KILLED (0.5 s) |
| M27 | `stages.py` | `self.decided_by[pos] = label` -> `self.decided_by[pos] = np.where(pd.isna(self.decided_by[pos]), label, ` | `test_ledger_records_the_last_value_writer` | KILLED (0.5 s) |
| M28 | `stages.py` | `"TAXITIME_SEC_mvt": np.maximum(np.rint(self.value), 1).astype("int32")` -> `"TAXITIME_SEC_mvt": np.rint(self.value).astype("int32")` | `test_finalise_refuses_unserved_rows_and_floors_at_one` | KILLED (0.5 s) |
| M29 | `components.py` | `return np.rint(w * taxi_from_delta(proxy, C) + (1.0 - w) * np.asarray(` -> `return np.rint(w * (np.asarray(proxy, dtype="float64") - np.asarray(C,` | `test_blend_is_cbv_ship_builds_formula` | KILLED (0.5 s) |
| M30 | `components.py` | `threshold <= s / t <= 1.0 / threshold` -> `threshold <= s / t` | `test_coverage_ratio_gate` | KILLED (0.5 s) |
| M31 | `components.py` | `return {a: bool(ok and ratio.get(a, False)) for a, ok in manifest.item` -> `return {a: bool(ratio.get(a, False)) for a, ok in manifest.items()}` | `test_effective_gate_never_widens_the_manifest` | KILLED (0.5 s) |
| M32 | `components.py` | `seen["r"] = np.where(eligible, r, 0.0)` -> `seen["r"] = r` | `test_stack_values_serve_only_covered_gated_rows` | KILLED (0.5 s) |
| M33 | `artifact.py` | `bad = [f for f, h in listed.items() if sha256_file(d / f) != h]` -> `bad = []` | `test_tampered_extra_or_missing_files_are_refused` | KILLED (6.2 s) |
| M34 | `artifact.py` | `if final.exists():` -> `if False:` | `test_artefact_is_never_overwritten` | KILLED (5.9 s) |
| M35 | `engine.py` | `w = cfg["matched.blend.catboost_weight"]` -> `w = 0.6` | `test_component_outputs_reproduce_the_served_value` | KILLED (23.6 s) |
| M36 | `engine.py` | `if n_raw > HEAVY_ROWS and not allow_heavy:` -> `if False:` | `test_fit_refuses_unported_components_and_heavy_runs` | KILLED (4.7 s) |
| M37 | `engine.py` | `if elig.any():` -> `if False:` | `test_decided_by_follows_the_lanes` | KILLED (22.0 s) |
| M38 | `protocols.py` | `if leak:` -> `if False:` | `test_leakage_guard_refuses_the_stack_on_its_training_months` | KILLED (51.3 s) |
| M39 | `protocols.py` | `G_, lo, hi = cal.gain_ci(y, pa, pb,` -> `G_, lo, hi = cal.gain_ci(y, pb, pa,` | `test_compare_equals_the_registered_bootstrap` | KILLED (0.6 s) |
| M40 | `protocols.py` | `_fold("F2", set(ALL) - {7}, (3, 9), (7,))` -> `_fold("F2", set(ALL) - {1}, (3, 9), (1,))` | `test_primary_protocol_is_lomo_jan_and_jul` | KILLED (0.5 s) |
| M41 | `fm1_replay_gate.py` | `fail = ~np.isfinite(pkg) \| (diff > tol + floor)` -> `fail = ~np.isfinite(pkg) \| (diff > tol)` | `test_verdict_tolerance_is_one_second_plus_the_floor` | KILLED (0.5 s) |
| M42 | `fm1_replay_gate.py` | `fail = ~np.isfinite(pkg) \| (diff > tol + floor)` -> `fail = ~np.isfinite(pkg) \| (diff >= tol + floor)` | `test_verdict_tolerance_is_one_second_plus_the_floor` | KILLED (0.5 s) |
| M43 | `fm1_replay_gate.py` | `for key, g in ref.groupby(["ap", "month", "source", "lane"], sort=True` -> `for key, g in ref.groupby(["ap", "month", "source"], sort=True):` | `test_sample_takes_every_stratum_and_caps_each` | KILLED (0.8 s) |
| M44 | `fm1_replay_gate.py` | `fail = ~np.isfinite(pkg) \| (diff` -> `fail = np.zeros(len(pkg), bool) & (diff` | `test_verdict_fails_rows_the_package_cannot_produce` | KILLED (0.8 s) |
| M45 | `fm1_replay_gate.py` | `if bad_r12.any() or bad_v34.any():` -> `if False:` | `test_reference_inconsistency_is_refused` | KILLED (1.8 s) |
| M46 | `engine.py` | unmatched write `np.rint(np.maximum(p, 1.0))` -> `np.rint(np.maximum(p, 1.0)) + 1` | `test_component_outputs_reproduce_the_served_value` | KILLED (30.4 s) |
| M47 | `protocols.py` | `out = pd.concat(recs, ignore_index=True)` -> `out = pd.concat(recs[:1], ignore_index=True)` | `test_compare_pools_several_records_per_side` | KILLED |
| M48 | `protocols.py` | blend `w * B.pred + (1 - w) * A.pred` -> `w * A.pred + (1 - w) * B.pred` | `test_blend_oof_is_the_cbv_formula_on_common_rows` | KILLED |

Not rehearsed (named, not hidden): `catboost_worker.py` internals (exercised only through the worker round-trip test),
`refdoc.fmt_*` string formatting (covered by the committed-reference equality test), the CLI wrappers in
`prc/__main__.py` (covered by tests/fm1/test_fm1_cli.py's exit-code tests), `api.dry_run` path listing.

## Unmatched-lane port (2026-09-28, PREREG_fm1 A3): 38/38 non-equivalent mutants killed (M83 equivalent, justified in logs/fm1_port_scratch/mutation_results.jsonl)

| id | file | mutation | killing test | outcome |
|---|---|---|---|---|
| M49 | `unmatched.py` | `    if body.any(): ⏎         new[body] = shipped.unmatched_deps().anchored_values(v[body], A[body])` -> `    if body.any(): ⏎         new[:] = shipped.unmatched_deps().anchored_values(v, A)` | `test_storm_moves_only_body_rows_by_the_shipped_anchor` | KILLED (1.1 s) |
| M50 | `unmatched.py` | `    if not (len(v) == len(A) == len(body)):` -> `    if False:` | `test_storm_moves_only_body_rows_by_the_shipped_anchor` | KILLED (1.1 s) |
| M51 | `unmatched.py` | `np.where(v10 != v7, "+E1", "")` -> `np.where(v10 != np.rint(mix), "+E1", "")` | `test_rome_tags_name_every_rule_that_moved_a_row` | KILLED (0.4 s) (first run: SURVIVED; test strengthened) |
| M52 | `unmatched.py` | `[dep & rows.AOBT_3_flt.isna().to_numpy()]` -> `[dep]` | `test_unmatched_ids_are_the_departures_without_aobt3` | KILLED (0.4 s) |
| M53 | `unmatched.py` | `            if f.resolve() != cfg.path("data.scored_file").resolve():` -> `            if False:` | `test_e2_context_must_be_the_files_e2_reads` | KILLED (0.4 s) |
| M54 | `unmatched.py` | `        elif not sorted(raw.glob(f"training_{y:04d}-{m:02d}-01_*.parquet")):` -> `        elif False:` | `test_e2_context_must_be_the_files_e2_reads` | KILLED (0.4 s) |
| M55 | `unmatched.py` | `"max_iter": cfg["unmatched.hgb.max_iter"]` -> `"max_iter": cfg["unmatched.hgb.n_iter_no_change"]` | `test_body_params_default_to_the_shipped_mjs_arm` | KILLED (1.1 s) |
| M56 | `unmatched.py` | `    if len(ids) == 0: ⏎         return comp, info, snap          # BC-30: the shipped string-key builders never ` -> `    pass` | `test_regression_zero_unmatched_rows_skip_the_lane` | KILLED (3.3 s) |
| M57 | `unmatched.py` | `            ledger.write(stages["storm"], ids[moved], new[moved])` -> `            body = ledger.decided_of(ids) == "unmatched_body" ⏎             ledger.write(stages["storm"], ids[bo` | `test_world_storm_anchor_moves_only_body_rows_and_labels_them` | KILLED (3.6 s) |
| M58 | `unmatched.py` | `unm=D.with_sp_bins(pd.read_parquet(d / "unm.parquet"))` -> `unm=pd.read_parquet(d / "unm.parquet")` | `test_world_lane_serves_every_unmatched_departure_and_round_trips` | KILLED (6.1 s) |
| M59 | `unmatched.py` | `storm_values(before, A, ledger.decided_of(ids) == "unmatched_body")` -> `storm_values(before, A, np.ones(len(ids), dtype=bool))` | `test_unmatched_lane_replays_every_shipped_step_on_all_5290_scored_rows` | KILLED (136.0 s) (first run: INVALID (old text occurs 0 times); test strengthened) |
| M60 | `stages.py` | `    StageSpec("rome", "value", "rome"), ⏎     StageSpec("arc", "value", "arc"),` -> `    StageSpec("arc", "value", "arc"), ⏎     StageSpec("rome", "value", "rome"),` | `test_stage_order_is_the_shipped_chain_order` | KILLED (0.4 s) |
| M61 | `stages.py` | `"rome_e2w": cfg["modules.rome.enabled"] and cfg["modules.rome.e2"] and cfg["modules.rome.e2w"],` -> `"rome_e2w": cfg["modules.rome.enabled"] and cfg["modules.rome.e2"],` | `test_plan_turns_the_ported_modules_on_and_off` | KILLED (0.4 s) |
| M62 | `stages.py` | `        return self.value[self._positions(ids)].copy()` -> `        self._positions(ids) ⏎         return self.value[:len(ids)].copy()` | `test_ledger_reads_values_and_writers_by_id` | KILLED (0.4 s) |
| M63 | `stages.py` | `"storm": cfg["modules.storm.mode"] == "anchor"` -> `"storm": cfg["modules.storm.mode"] != "anchor"` | `test_plan_refuses_unported_components` | KILLED (0.3 s) |
| M64 | `knobs.py` | `        if v["modules.rome.e2w"] and not v["modules.rome.e2"]:` -> `        if False:` | `test_unmatched_lane_cross_checks_are_loud` | KILLED (0.2 s) |
| M65 | `knobs.py` | `        stray = sorted(set(v["modules.arc.airports"]) - codes)` -> `        stray = []` | `test_unmatched_lane_cross_checks_are_loud` | KILLED (0.2 s) |
| M66 | `knobs.py` | `        if v["data.dayoff_tz"] != "UTC":` -> `        if False:` | `test_unmatched_lane_cross_checks_are_loud` | KILLED (0.2 s) |
| M67 | `knobs.py` | `        if "LIRF" not in codes:` -> `        if False:` | `test_unmatched_lane_cross_checks_are_loud` | KILLED (0.2 s) |
| M68 | `knobs.py` | `    _K("unmatched.learner", "choice", "v26",` -> `    _K("unmatched.learner", "choice", "hgb",` | `test_unmatched_lane_defaults_are_the_ported_v36_chain` | KILLED (0.2 s) (first run: SURVIVED; test strengthened) |
| M69 | `fm1_replay_gate.py` | `    fail = beyond & (exc == "") ⏎     rows = sample.assign(` -> `    fail = beyond ⏎     rows = sample.assign(` | `test_verdict_names_dropped_component_rows_without_failing_them` | KILLED (0.4 s) |
| M70 | `fm1_replay_gate.py` | `    fail = beyond & (exc == "") ⏎     rows = sample.assign(` -> `    fail = beyond & (exc == "x") ⏎     rows = sample.assign(` | `test_verdict_names_dropped_component_rows_without_failing_them` | KILLED (0.4 s) |
| M71 | `fm1_replay_gate.py` | `        value[uid] = lane.reindex(uid).to_numpy(float)` -> `        value[uid] = s.loc[uid, "v36"].to_numpy(float)` | `test_unmatched_rows_are_served_by_the_package_lane` | KILLED (2.8 s) |
| M72 | `fm1_replay_gate.py` | `DROPPED = {"ARN": ("v20", "v21")}` -> `DROPPED = {"ARN": ("v17", "v21")}` | `test_dropped_rows_are_the_shipped_arn_rows` | KILLED (0.4 s) |
| M73 | `fm1_replay_gate.py` | `    fail = beyond & (exc == "") ⏎     bad = u.index` -> `    fail = beyond ⏎     bad = u.index` | `test_population_check_fails_non_exception_rows_and_lists_the_dropped_ones` | KILLED (0.4 s) |
| M74 | `fm1_replay_gate.py` | `    floor = float(np.where(ok, V.max(axis=0) - V.min(axis=0), 0.0).max()) if V.size else 0.0` -> `    floor = 0.0` | `test_population_check_fails_non_exception_rows_and_lists_the_dropped_ones` | KILLED (0.4 s) |
| M75 | `fm1_replay_gate.py` | `                                  "modules.storm.mode": "anchor"}))` -> `}))` | `test_gate_config_replays_v36` | KILLED (0.4 s) |
| M76 | `shipped.py` | `    if got != E2_RULES_SHA256:` -> `    if False:` | `test_shipped_deps_bind_the_chain_and_refuse_a_changed_e2_rule_file` | KILLED (1.1 s) |
| M77 | `engine.py` | `        served \|= set(U.unmatched_ids(rows).tolist())` -> `        pass` | `test_world_lane_serves_every_unmatched_departure_and_round_trips` | KILLED (3.2 s) |
| M78 | `artifact.py` | `            art.unmatched_lane.save(tmp / "unmatched_lane")` -> `            pass` | `test_world_lane_serves_every_unmatched_departure_and_round_trips` | KILLED (2.8 s) |
| M79 | `features.py` | `        if cfg["lanes.unmatched"] and cfg["unmatched.learner"] == "hgb":` -> `        if cfg["lanes.unmatched"]:` | `test_v26_builds_no_provisional_unmatched_frame` | KILLED (1.9 s) |
| M80 | `unmatched.py` | `    ledger.write(stages["unmatched_body"], ids, np.rint(b["mix"]))` -> `    ledger.write(stages["unmatched_body"], ids, np.rint(b["mix"]) + 1)` | `test_world_lane_serves_every_unmatched_departure_and_round_trips` | KILLED (3.6 s) |
| M81 | `unmatched.py` | `    comp["storm_A"], comp["storm_shift"] = A, new - before` -> `    comp["storm_A"], comp["storm_shift"] = A, new` | `test_world_storm_anchor_moves_only_body_rows_and_labels_them` | KILLED (3.6 s) |
| M82 | `fm1_replay_gate.py` | `    ids = U.unmatched_ids(rows) ⏎     led = S.Ledger(ids)` -> `    ids = rows.MVT_ID_mvt[rows.PHASE_mvt == "DEP"].to_numpy(float) ⏎     led = S.Ledger(ids)` | `test_unmatched_child_serves_the_lane_end_to_end` | KILLED (2.2 s) |
| M83 | `unmatched.py` | `airport_hour_witness(f[~f.unmatched])` -> `airport_hour_witness(f)` | `test_lane_frames_equal_the_shipped_loader` | EQUIVALENT (airport_hour_witness keeps only AOBT_3-present rows itself; unmatched == AOBT_3 null); replaced by M83b (1.5 s) (first run: SURVIVED; test strengthened) |
| M83b | `unmatched.py` | `    wit = D.uc.airport_hour_witness(f[~f.unmatched])` -> `    wit = D.uc.airport_hour_witness(f[(~f.unmatched) & (f.ADEP_mvt == ROME)])` | `test_lane_frames_equal_the_shipped_loader` | KILLED (1.6 s) |
| M84 | `unmatched.py` | `    f = D.bs.derive(D.bs.admissible(D.bs.load_movements(` -> `    f = D.bs.derive((D.bs.load_movements(` | `test_lane_frames_equal_the_shipped_loader` | KILLED (1.6 s) |
| M85 | `components.py` | `        if thin: ⏎             raise E.SchemaError(f"MTX: leave-month-out` -> `        if False: ⏎             raise E.SchemaError(f"MTX: leave-month-out` | `test_regression_bug_31_mtx_refuses_a_single_training_month_by_name` | KILLED (0.4 s) |
| M86 | `fm1_replay_gate.py` | `    if bad: ⏎         raise AssertionError(f"{len(bad)} named exception` -> `    if False: ⏎         raise AssertionError(f"{len(bad)} named exception` | `test_regression_bug_32_exceptions_must_be_unmatched_rows` | KILLED (0.4 s) |

TOTAL 38 / 38 killed (1 equivalent, justified)

## LDI port (2026-09-29, prc/pipeline/fm1/ldi.py; tests/fm1/test_fm1_ldi.py)

Run by a scratch harness (one mutant at a time, fresh PYTHONPYCACHEPREFIX per run, sources restored and sha256-verified
after the run; raw results logs/cml_scratch/mutation_fm1_ldi_20260929.json). Killing test = the first red one (-x), read from pytest's FAILED line (a -rf re-run; all 22 matched).

| id | file | mutation | killing test | result |
|---|---|---|---|---|
| M1-LDI | `ldi.py` | target = (w_freezing == 1) & severe -> severe | `test_flags_equal_the_shipped_2026_frame` | KILLED (0.67s) |
| M2-LDI | `ldi.py` | day = local_day(t) (UTC + 3 h) -> the UTC date | `test_flags_equal_the_shipped_2026_frame` | KILLED (0.62s) |
| M3-LDI | `ldi.py` | hit = target & (delta > 0) -> target | `test_a_zero_delta_band_writes_nothing` | KILLED (0.64s) |
| M4-LDI | `ldi.py` | NaN pre-value check -> skipped | `test_serve_refuses_a_target_row_no_value_stage_served` | KILLED (0.65s) |
| M5-LDI | `ldi.py` | population(rows, scored) -> 'scored_file' | `test_a_partial_day_of_the_scored_file_is_refused_by_name` | KILLED (0.65s) |
| M6-LDI | `ldi.py` | short = nb[nb < ns] -> nb[nb > ns] | `test_a_partial_day_of_the_scored_file_is_refused_by_name` | KILLED (0.65s) |
| M7-LDI | `ldi.py` | mixed-batch check -> skipped | `test_a_batch_from_another_file_is_its_own_population_and_a_mixed_batch_is_refused` | KILLED (0.66s) |
| M8-LDI | `ldi.py` | DELTA_KNOBS <30 / 30-60 swapped | `test_knobs_default_to_the_registered_constants_and_ldi_is_ported` | KILLED (0.59s) |
| M9-LDI | `ldi.py` | _check_deltas(delta) -> removed | `test_serve_refuses_deltas_that_are_not_the_three_bands` | KILLED (0.67s) |
| M10-LDI | `ldi.py` | missing-raw-row RoutingError -> skipped | `test_serve_refuses_a_matched_frame_without_weather_or_times` | KILLED (0.76s) |
| M11-LDI | `ldi.py` | null MVT/SCHED check -> skipped | `test_serve_refuses_a_matched_frame_without_weather_or_times` | KILLED (0.65s) |
| M12-LDI | `ldi.py` | apply passes rows (not the scored file) as the population | `test_apply_reads_the_deltas_and_the_scored_file_from_the_config` | KILLED (0.70s) |
| M13-LDI | `ldi.py` | before + delta -> before - delta | `test_serve_adds_the_band_delta_on_severe_freezing_ltfm_rows_only` | KILLED (0.64s) |
| M14-LDI | `ldi.py` | severe_days(..., w_temp_c) -> (..., w_freezing) | `test_flags_equal_the_shipped_2026_frame` | KILLED (0.61s) |
| M15-LDI | `ldi.py` | LTFM departures by ADEP -> by ADES | `test_serve_adds_the_band_delta_on_severe_freezing_ltfm_rows_only` | KILLED (0.63s) |
| M16-LDI | `ldi.py` | sp = MVT - SCHED -> SCHED - MVT | `test_flags_equal_the_shipped_2026_frame` | KILLED (0.61s) |
| M17-LDI | `ldi.py` | frame(): ap == LTFM -> ap != LTFM | `test_flags_equal_the_shipped_2026_frame` | KILLED (0.62s) |
| M18-LDI | `knobs.py` | ldi needs features.weather check -> skipped | `test_ldi_needs_the_matched_lane_and_its_weather_block` | KILLED (0.59s) |
| M19-LDI | `knobs.py` | delta_le30_s default 8.003118629519271 -> 8.0 | `test_knobs_default_to_the_registered_constants_and_ldi_is_ported` | KILLED (0.61s) |
| M20-LDI | `engine.py` | LDI.apply(...) -> a no-op serve(None, ...) | `test_world_predict_serves_ldi_through_the_engine` | KILLED (80.53s (0:01:20)) |
| M21-LDI | `stages.py` | plan: 'ldi': modules.ldi.enabled -> False | `test_plan_places_ldi_after_the_unmatched_lane_and_before_mtx` | KILLED (0.59s) |
| M22-LDI | `ldi.py` | serve's matched-column check before the no-LTFM return -> removed | `test_serve_refuses_a_matched_frame_without_weather_or_times` | KILLED (0.72s) |

TOTAL LDI 22 / 22 killed

## LightGBM + CatBoost stack base, v46's blend (2026-09-29, `adsb.stack.f_model`; tests/fm1/test_fm1_lgbcat.py)

Same harness as the LDI rows (fresh PYTHONPYCACHEPREFIX per mutant, sources sha256-verified after; raw
logs/cml_scratch/mutation_fm1_lgbcat_20260929.json); killing test read from pytest's FAILED line.

| id | file | mutation | killing test | result |
|---|---|---|---|---|
| M1-LGBCAT | `components.py` | blend weight 0.6 | `test_stack_base_is_the_r13_blend_formula` | KILLED (0.73s) |
| M2-LGBCAT | `components.py` | lightgbm returns C | `test_stack_base_is_the_r13_blend_formula` | KILLED (0.57s) |
| M3-LGBCAT | `components.py` | no C check | `test_stack_base_is_the_r13_blend_formula` | KILLED (0.57s) |
| M4-LGBCAT | `components.py` | unknown base returns F | `test_stack_base_is_the_r13_blend_formula` | KILLED (0.57s) |
| M5-LGBCAT | `knobs.py` | no models_lgbcat check | `test_stack_base_knob_default_and_contract` | KILLED (0.57s) |
| M6-LGBCAT | `knobs.py` | no learner check | `test_stack_base_knob_default_and_contract` | KILLED (0.56s) |
| M7-LGBCAT | `knobs.py` | no lightgbm-vs-lgbcat-models check | `test_stack_base_knob_default_and_contract` | KILLED (0.56s) |
| M8-LGBCAT | `engine.py` | fit cat only for cbv | `test_world_lgbcat_fits_the_catboost_even_without_the_half_blend` | KILLED (84.70s (0:01:24)) |
| M9-LGBCAT | `engine.py` | predict ignores the blend | `test_world_lgbcat_serves_the_blend_on_stack_rows_only` | KILLED (82.49s (0:01:22)) |
| M10-LGBCAT | `engine.py` | stack_F not recorded | `test_world_lgbcat_serves_the_blend_on_stack_rows_only` | KILLED (82.84s (0:01:22)) |
| M11-LGBCAT | `engine.py` | proxy not recorded | `test_world_lgbcat_serves_the_blend_on_stack_rows_only` | KILLED (90.81s (0:01:30)) |
| M12-LGBCAT | `engine.py` | catboost_C not recorded for all rows | `test_world_lgbcat_serves_the_blend_on_stack_rows_only` | KILLED (91.66s (0:01:31)) |
| E-LGBCAT | `engine.py` | reuse `c_all[~elig]` for the half-blend -> predict X[~elig] again | — | EQUIVALENT (the same rows through the same CatBoost give the same values; not counted) |

TOTAL LGBCAT 12 / 12 killed (1 equivalent, justified)

## 2026-10-01 — memory refactor of the fit path (features.matched_design, learners.LightGBMLearner.fit, engine del)

Why: the final-fit dry run peaked at a 17.8 GB footprint (logs/fm1_final/mem_probe.log: a 2.1 GB frame copied twice inside
matched_design plus a column-major result, three row copies of X inside the learner, and the per-month parts kept beside
the concatenated frame). Guard tests written first and seen red against the 2026-09-28 code: design peak 2.61 x X.nbytes
(limit 2.0), learner peak 3.20 x (limit 1.8), column-major output. Pinned-prediction test (sha256 of predict(X), computed
with the old code before any edit) proves bit-identity at test scale; the 1-round full-scale artefact diff proves it at
full scale (logs/fm1_final/dry_fit_v2.log).

| id | file | mutation (old -> new) | test | result |
|---|---|---|---|---|
| M49 | `features.py` | float32 cast dropped (`np.asarray(v, dtype=np.float64)`) | `test_matched_design_equals_the_reference_formula_bit_for_bit…` | KILLED |
| M50 | `features.py` | `np.empty(..., order='F')` | same | KILLED |
| M51 | `features.py` | frame column preferred over the encoding | `…encoding_columns_take_precedence…` (added after it survived) | KILLED |
| M52 | `features.py` | per-column build via `frame.copy()` | `…allocates_at_most_twice_the_design…` | KILLED |
| M53 | `learners.py` | `tr.any()` for `tr.all()` | `…rows_in_neither_mask_are_excluded…` | KILLED |
| M54 | `learners.py` | refit on `X[fit_mask]` only | `…unchanged_by_the_memory_refactor_pinned` | KILLED |
| M55 | `learners.py` | `del es_model, des, dfit` removed | `…does_not_hold_three_row_copies…` | KILLED |
| M56 | `learners.py` | `free_raw_data=False` restored on dfit | same | KILLED |
| M57 | `learners.py` | refit Dataset built with seed-0 params for every seed | pinned | SURVIVED — equivalent: `lgb.train`'s params (seed s) override the Dataset's at construction |
| — | `engine.py` | `del m_parts` / `del u_parts` | no unit test (a release, no value path); covered by the full-scale probe re-run | measured |

## 2026-10-02 — derived fits (`fit(..., reuse_from=donor)`; knobs.REUSE_FREE_KNOBS / reuse_blockers, engine._reuse_learners)

13 tests written first and seen red 2026-10-01 (no ReuseError / REUSE_FREE_KNOBS / reuse_blockers / reuse_from), 15 after
the row-count and CLI tests were added. Rehearsal (targeted test per mutant, sources restored and verified): **15 / 15 killed**.

| id | file | mutation | test | result |
|---|---|---|---|---|
| M58 | `knobs.py` | `features.weather` added to the allow-list | `test_the_reuse_allow_list_is_exactly_the_registered_groups` | KILLED |
| M59 | `knobs.py` | blockers taken from the free knobs | `test_reuse_blockers_names_every_differing_knob…` | KILLED |
| M60 | `engine.py` | blockers check off | `test_reuse_refuses_a_knob_outside_the_allow_list` | KILLED |
| M61 | `engine.py` | split / tier check off | `test_reuse_refuses_validation_splits_and_tiers` | KILLED |
| M62 | `engine.py` | input-data check off | `test_reuse_refuses_different_input_data_encodings_or_design` | KILLED |
| M63 | `engine.py` | `adsb/` dropped from the learner inputs | same | KILLED |
| M64 | `engine.py` | every input key compared (free stages too) | same (the stack-key case must pass) | KILLED |
| M65 | `engine.py` | design-columns check off | same | KILLED |
| M66 | `engine.py` | encodings check off | same | KILLED |
| M67 | `engine.py` | early-stopping row count not compared | same | KILLED |
| M68 | `engine.py` | missing-CatBoost check off | `test_reuse_refuses_a_variant_that_needs_a_learner_the_donor_lacks` | KILLED |
| M69 | `engine.py` | donor's CatBoost kept when the variant has none | `test_a_derived_artefact_predicts_bit_identically…` | KILLED |
| M70 | `engine.py` | reuse path falls through to training | `test_a_derived_fit_never_trains_a_matched_learner` | KILLED |
| M71 | `engine.py` | `reused_learners_from` not recorded | `test_a_derived_artefact_predicts_bit_identically…` | KILLED |
| M72 | `__main__.py` | `--reuse-from` not forwarded | `test_reuse_from_flag_defaults_to_none_and_reaches_the_api` | KILLED |

## 2026-10-02 — serve-time EOBT clip (`eobt_clip.py`; the late-knob hash rule in `knobs.py`; stage + engine wiring)

27 tests in `tests/fm1/test_fm1_eobt_clip.py`, written first and seen red 2026-10-02 09:0x EDT (no `eobt_clip` module, no
late knobs); green before the rehearsal (25 fast + 2 slow). Rehearsal (one mutant at a time, targeted tests, a fresh
bytecode prefix per mutant, sources restored and sha-verified; each failure read at its assertion): **24 / 24 killed**.

| id | file | mutation | test | result |
|---|---|---|---|---|
| M-CLP1 | `knobs.py` | late knobs never left out of the hash | `test_every_config_with_the_late_knobs_at_default_hashes_as_before…` | KILLED |
| M-CLP2 | `knobs.py` | late knobs always left out | `test_any_late_knob_off_its_default_puts_every_late_knob_in_the_hash` | KILLED |
| M-CLP3 | `knobs.py` | each late knob left out at its own default (not all-or-none) | same | KILLED |
| M-CLP4 | `knobs.py` | `min_share` default 0.2 | `test_late_knobs_and_their_defaults_are_pinned` | KILLED |
| M-CLP5 | `knobs.py` | no ICAO check on the clip's airports | `test_clip_knobs_refuse_bad_values` | KILLED |
| M-CLP6 | `knobs.py` | no airports.codes cross-check | same | KILLED |
| M-CLP7 | `knobs.py` | no "late must be hashed" schema check | `test_check_schema_refuses_an_unhashed_late_knob` | KILLED |
| M-CLP8 | `stages.py` | plan reads `modules.ldi.enabled` | `test_plan_places_the_clip_before_the_features_and_only_when_on` | KILLED |
| M-CLP9 | `eobt_clip.py` | AOBT_3 dropped from eligibility | `test_eligible_rows_are_epx_selection_plus_a_movement_time` | KILLED |
| M-CLP10 | `eobt_clip.py` | late = EOBT >= LOBT | `test_cell_table_counts_strictly_later_eobt_per_airport_month` | KILLED |
| M-CLP11 | `eobt_clip.py` | fires at share > min_share | `test_decide_fires_at_or_above_the_share` | KILLED |
| M-CLP12 | `eobt_clip.py` | no row floor | `test_decide_refuses_a_cell_below_the_row_floor_by_name` | KILLED |
| M-CLP13 | `eobt_clip.py` | floor n <= min_rows | same | KILLED |
| M-CLP14 | `eobt_clip.py` | clip every eligible row of every cell (fired ignored) | `test_serve_clips_only_the_late_eligible_rows_of_fired_cells` | KILLED |
| M-CLP15 | `eobt_clip.py` | clip to LOBT + 60 s | same / `test_the_clip_equals_epx_priced_clip…` | KILLED |
| M-CLP16 | `eobt_clip.py` | population always the batch | `test_a_batch_of_the_scored_file_is_decided_on_the_scored_files_whole_cell` | KILLED |
| M-CLP17 | `eobt_clip.py` | mixed batch let through | `test_a_batch_from_another_file_is_its_own_population…` | KILLED |
| M-CLP18 | `eobt_clip.py` | apply ignores `min_share` | `test_apply_reads_the_knobs_and_the_scored_file` | KILLED |
| M-CLP19 | `engine.py` | features built from the unclipped rows | `test_world_predict_with_the_clip_equals_predict_on_clipped_rows…` (slow) | KILLED (assert_series_equal) |
| M-CLP20 | `engine.py` | rows_digest of the clipped rows | same (slow) | KILLED (digest assertion) |
| M-CLP21 | `eobt_clip.py` | clip written into the caller's frame | `test_serve_clips_only_the_late_eligible_rows_of_fired_cells` | KILLED |
| M-CLP22 | `eobt_clip.py` | cell keyed on the day | `test_cell_table_counts_strictly_later_eobt_per_airport_month` | KILLED |
| M-CLP23 | `knobs.py` | `modules.` dropped from the reuse allow-list | `test_the_clip_knobs_are_reuse_free…` | KILLED |
| M-CLP24 | `eobt_clip.py` | fires at 0.05 | `test_on_the_real_files_the_detector_fires_at_lfpg_july_2026_only` (slow, real files) | KILLED |

`scripts/analysis/clp_checks.py` (CLP acceptance 1-3): 11 tests in `tests/test_clp_checks.py`, seen red first (no
module); **6 / 6 killed**: M-CLC1 every cell counted as changed · M-CLC2 cells outside the allow-list ignored · M-CLC3 an
empty change passes · M-CLC4 no clipped-row range · M-CLC5 no population check · M-CLC6 a missing clip report defaults
to a passing one.

## 2026-10-02 — BC-38 (clean checkout): `knobs.check_schema` structural only; `check_evidence_files` in the development tree

`tests/fm1/test_fm1_clean_checkout.py` (3 slow tests: FM1 imports and parses `configs/fm1.yaml` in a tree holding only
`prc/`, the git-tracked `scripts/*.py`, `configs/fm1.yaml` and the E2 rules file; the CLI regenerates
CONFIG_REFERENCE.md there byte-identically; every zero-argument `shipped` / `legacy` loader runs there) and two tests in
`test_fm1_knobs.py`; the two (a) tests were seen red against the old code (ConfigError: evidence file … does not exist).
**4 / 4 killed:** M-CCO1 the exists() check back in check_schema (3 tests red) · M-CCO2 no exists() in
check_evidence_files · M-CCO3 a shipped loader reads `plans/…` at import · M-CCO4 the "evidence is missing" check dropped.
Coverage additions in `test_fm1_eobt_clip.py` (written after the code, so rehearsed): **2 / 2 killed** — M-CLP25 `_check`
dropped from `eligible` · M-CLP26 `cell_table` returns None on no eligible row.

## 2026-10-02 — training-time ADS-B dropout (`features.set_unheard`, `features.dropout_masks`, knobs `adsb.train_dropout.*`, engine fit hook)

`tests/fm1/test_fm1_adsb_dropout.py` (9 fast + 1 slow), written first and seen red (no functions, no knobs). **9 / 9 killed:**
M-ACR1 FL also blanks the airport counts · M-ACR2 coverage 0 instead of 1 · M-ACR3 NB keeps moving_mvt_m10 · M-ACR4 no
mask-length check · M-ACR5 FL over all rows, not only joined · M-ACR6 NB per row, not per (airport, day) · M-ACR7 unseeded ·
M-ACR8 no features.adsb cross-check · M-ACR9 fit skips the dropout (slow world test).
`scripts/analysis/acr_price.py` (tests written after the code — stated; they caught a tuple / dict defect in `stats`):
**7 / 8 killed**, M-ACP2 (Frankfurt's share drawn from all rows instead of joined rows) equivalent on the real tables —
unjoined rows already hold the unheard state, so re-applying it changes nothing.
