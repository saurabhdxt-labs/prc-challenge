
## `--orderfeats` with `--all-rows`: two guards, one refusal (2026-09-09, Amendment 22)

`parse_args` refuses `--all-rows --orderfeats` twice: the dedicated
`if args.orderfeats: ap.error(...)` (which names the reason — the unmatched cache carries no
order columns) and, immediately after, the composition rule
`if not set(on) <= {"ytarget", "queue", "day"}`. Removing the dedicated error leaves the
composition rule refusing the same argv with a less specific message, so the mutation is
SEMANTICALLY EQUIVALENT under any test that asserts the refusal.

Kept anyway: the dedicated error is the one a reader sees, and the composition set is a list of
what MAY compose — a future amendment that adds "order" to that set would silently enable an arm
whose cache does not exist without it. The equivalence is recorded rather than tested.

## RETIRED 2026-09-11 — `w_lowvis`: `<` vs `<=` at 1.5 km was an EQUIVALENT mutant (2026-09-09, Amendment 24)

**Retired.** The equivalence was an artifact of reading the archive's rounded miles naively: a METAR's
1500 m arrives as 0.93 mi = 1.4967 km, so the strict rule wrongly flagged it. Since `prc.weather`
2.0.2 every visibility is snapped to the FM 15 metre grid, 1500 m lands exactly on 1.5 km, and the
`<` is TESTED (`tests/test_weather_features.py::test_low_visibility_is_strict_at_the_reported_1500_metres`).
The original reasoning is kept below as the record of how an equivalent mutant hid a defect.

`stand_ab.load_weather` derives `w_vis_km` as `parse(decimal string) * 1.609344` and flags
`w_vis_km < WX_LOWVIS_KM` (1.5). Flipping that to `<=` survives the test suite, and it is not a
coverage gap: **no decimal the archive can hold maps to exactly 1.5 km through that path.** The
two visibilities that straddle the bound by one float step are 0.9320567883560009 mi ->
1.4999999999999998 km and 0.932056788356001 mi -> 1.5000000000000002 km; `1.5 / 1.609344`
multiplied back lands on the first, not on 1.5. With no reachable input at the bound, `<` and `<=`
compute the same function.

The test pins what is real — one step below flags, one step above does not — and asserts the pair
actually straddles, so a change to the conversion constant that moved them to the same side would
fail rather than quietly weaken the check. Recorded rather than tested.

Since 2026-09-11 the parse (`* MI_TO_KM`) lives in `prc.weather.parse_observations` and
`load_weather` reads `vis_km` from it; the flag and the equivalence are unchanged. The same holds
for the visibility clause of `prc.weather`'s `dc_moist_cold` (`le3(vis_km, 1.5)`); there the `<=`
is not left untested, because `le3` itself is pinned at an exactly representable bound (3.0 <= 3.0)
in `tests/test_weather_parser.py::test_three_valued_helpers`.

---

## UPC.1 — two empty-input guards are EQUIVALENT MUTANTS (2026-09-19; `scripts/analysis/upc1_build.py` and its test are research files outside the repository)

`scripts/analysis/upc1_build.py`:
- `recover_from_arrivals` — `if dep.empty or arr.empty: return out`
- `duplicate_matched_nearby` — `if u.empty or m.empty: return out`

`tests/test_upc1_build.py::test_recovery_on_an_empty_departure_frame` and
`::test_duplicate_on_empty_frames` pin the contract "empty input → empty / all-False output, no
exception". **Neither test can be killed by any mutation of my code**, and this was measured, not
assumed:

1. Removing each early guard alone — both tests stay **GREEN**.
2. Removing the early guard **and** the downstream `if j.empty: return out` together — both tests
   still stay **GREEN**.

The reason is that pandas already provides the contract end to end: an empty `merge` yields an empty
frame, `.abs()` and the boolean mask over it are empty, and the positional assignment writes nothing.
The behaviour these tests assert is **supplied by pandas, not by any line I wrote**, so no mutant of
mine is distinguishable.

**Kept rather than deleted, with the reason written here:** the guards short-circuit before building
the `__i` column and the merge, and the tests are forward-looking — a later refactor of either
function to numpy positional indexing (the exact shape that produced the two ESCAPED mutations in this
same file, see `tests/test_upc1_build.py`'s docstring) *would* break the contract, and these are the
only tests that would notice. They are recorded as exempt rather than counted as kills.

**Not exempt, and not to be confused with these:** every other UPC/UPC.1 test has a measured kill —
41 decision-logic mutations plus 15 targeted ones, all RED, covering all 53 tests bar these two.

---

## UPC.4 — the paired NaN-key filters are EQUIVALENT MUTANTS INDIVIDUALLY, load-bearing AS A PAIR (2026-09-19)

`scripts/analysis/upc1_build.py` filters **both** frames entering each merge through `_keys_present`
(BC-15). Removing **either side alone** leaves every UPC.4 null-key test GREEN, because a NaN-to-NaN
match needs a null on both sides, so the surviving filter already blocks it. Measured, not assumed:
four single-side mutants all ESCAPED.

Removing **both** sides of a merge goes **RED**:
- `recover_from_arrivals` → kills `test_recovery_refuses_a_NULL_to_NULL_origin_match`,
  `::_NULL_CALLSIGN_join`, `::_NULL_DESTINATION_join`
- `duplicate_matched_nearby` → kills `test_duplicate_refuses_a_NULL_CALLSIGN_twin`,
  `::_NULL_AIRPORT_twin`

**Both sides are kept deliberately.** The maintainable invariant is "every frame entering a merge has
its keys validated", not "one side happens to be enough" — the latter is a precondition held somewhere
else, which is the BC-12 class this fix exists to close. The single-side mutants are recorded as
exempt; the paired mutants are the real kills and they are RED.


---

## UPC.5 — `_origin_matches`'s `notna` pair is an EQUIVALENT MUTANT (2026-09-19)

`scripts/analysis/upc5_build.py::_origin_matches` ends `& arr_adep.notna() & dep_adep.notna()`.
Removing **both** leaves every UPC.5 test GREEN, and that is correct behaviour, not a test gap:
**measured on pandas 3.0.1, an element-wise Series `==` returns False for None vs None**, so the
comparison already excludes unknown origins.

I originally wrote the opposite in that docstring - citing BC-15's NaN-to-NaN merge behaviour, which
applies to JOIN KEYS and not to `==`. The docstring is corrected and BC-15's scope is tightened in
`reports/bug_classes.md`. The guards are kept only to make "both known" explicit where the origin is
decided; they are exempt, not counted as kills. The **paired** mutant that removes this guard AND
B0's `ADEP_mvt` clause does go RED (`test_b0_null_departure_key`,
`test_waterfall_refuses_a_NULL_to_NULL_origin_match`).

---

## CSF — the null-callsign guards are EQUIVALENT MUTANTS **on pandas 3.0.1 only** (2026-09-19)

`scripts/analysis/csf_build.py::callsign_features` guards null callsigns twice (pool-side `.notna()`
and row-side `known`/`ok`). **Measured: removing either, or BOTH together, leaves every CSF test
GREEN** — four mutants, all equivalent.

**The cause is a pandas version behaviour, and it is now pinned rather than assumed.** On pandas 3.0.1
`Series.astype(str)` on an object column containing `None` yields **NaN**, not the string `"None"`, so
the `astype(str) → value_counts → map` path can never make two null callsigns collide. On pandas 2.x
the same expression yields `"None"` and **every no-callsign flight would count every other one as the
same aircraft — silently inventing this arm's entire signal**, which is a callsign-identity count.

`tests/test_csf_build.py::test_pandas_astype_str_PROPAGATES_nan_which_is_what_makes_the_null_path_safe`
asserts the behaviour directly. **If it goes RED the guards stop being redundant and become
load-bearing** — its failure message says so.

Kept, not deleted, for that reason. Note this is a DIFFERENT mechanism from **BC-15**
(`pandas.merge` joining NaN to NaN, which IS live and verified) — CSF does no merge.
Separately pinned and NOT exempt:
`test_null_callsign_stays_zero_even_with_a_literal_nan_string_in_the_pool`, which the pandas behaviour
does not cover.

---

## TRF — the two bootstrap resampling sites are EQUIVALENT SINGLY, load-bearing AS A PAIR (2026-09-19)

`scripts/analysis/trf_build.py::bootstrap_ratio_ci` resamples calendar days on **both** the 2025 and
2026 sides. Mutating **one** side to resample rows leaves
`test_bootstrap_resamples_DAYS_not_rows` GREEN, because the *other* side still injects day-level
variance into the ratio and the interval stays wide. Mutating **both** goes **RED**: the day-CI width
collapses to **0.0348**, exactly the row-CI reference width.

The test compares against an explicit **row-bootstrap reference implementation**, not a magic width
threshold, after two earlier versions of it let the mutant escape — one used within-day *independent*
noise (where the two methods genuinely agree) and one used heavy lognormal tails (where tail variance
swamped the 40-days-vs-2,000-rows distinction). Both failures are recorded in the test's docstring,
because the lesson is that a dependence-structure test needs data whose dependence structure actually
bites.

---

## USH-JJ — the full-frame BC-23 guard in `stats` is EQUIVALENT given the subset guard (2026-09-22)

`scripts/analysis/ushjj_build.py::stats` calls `arscat_build.assert_day_blocks_are_real` twice: on the whole
leg's day vector, then on the LFPG-excluded subset's. Mutant **U22** (the full-frame call dropped) ESCAPED
`tests/test_ushjj_build.py::test_stats_refuses_a_degenerate_day_vector_bc23`: any degenerate day vector the
test can build is degenerate on the non-LFPG subset too, so the second call raises the same `ValueError`
("day blocks") and the observable is unchanged. Mutant **U32** (BOTH calls dropped) is KILLED by the same test,
and **U33** (the subset call dropped) is KILLED by its second case (LFPG rows keep 12 days, non-LFPG rows
collapse to one), so each guard that can be observed is observed. The full-frame call is kept as defence in
depth for a frame with no LFPG rows at all (then the subset IS the frame and the first call is the only one),
a shape this lane never produces. Rehearsal record: `logs/ushjj_scratch_tools/mutate_ushjj_report.json`
(35 mutants, 34 killed, 28/28 tests seen red; U34-U35 added with the end-to-end main() test after the registered run).

---

## CSF.2 — three equivalent mutants from the 2026-09-22 rehearsal (`logs/csf2_scratch_tools/mutate_csf2_report.json`)

**X51 / X52 — the month pre-filter on the MATCHED rows in `csf2_build.lomo_fit` / `forward_fit`** (`l_[l_.month != m]` →
`l_`; `l_[l_.month < m]` → `l_[l_.month != m]`) SURVIVED because `rome_fill.rome_rows(unm, lirf, months)` applies its own
`lirf.month.isin(months)` filter, so the pre-filter is redundant by construction. The contract "no fit sees its test
month / the forward leg sees only strictly earlier months" is pinned on the frames actually handed to the learner by
`test_lomo_and_forward_fits_never_see_the_test_month` (recorder in place of the fit) and, for the fold lists, by
`test_lomo_folds_are_leave_one_month_out` / `test_forward_folds_train_only_on_strictly_earlier_months`. The pre-filters
are kept for readability; if `rome_rows` ever stops filtering months they become load-bearing and the recorder test
goes RED.

**Y2 — `rome_fill.fit_predict_fill` not declaring `extra_cat` columns in `categorical_feature`** SURVIVED
`test_extra_cat_planted_identity_is_recovered...` because a 2-level planted categorical separates equally well as
numeric codes (one split). The declared handling matters for high-cardinality columns (`cat_smooth`,
`min_data_per_group`, `max_cat_to_onehot` regularise per-level statistics); on a ≤ 3-level fixture the tree finds the
same partition either way. Equivalent on the fixture, NOT in general: the declaration is kept, and the 2026-09-22
CSF.2 run's `actype_any` (dozens of ICAO types) is where it bites. No cheap observable proof exists without reaching
into the booster's dump; recorded rather than tested.

**Round 2 (`mutate_csf2_round2_report.json`, 17 mutants, 15 killed; X50 and X58 now die).** **Z17** — `forward_fit`
passing `earlier + [m]` as the months list — is EQUIVALENT: the unmatched and matched frames are pre-filtered
`month < m` before `rome_rows`, so the list cannot re-admit the test month (same redundancy as X51/X52, from the
other side). **Z10** (synthetic `arr25` with no ARR legs) survived because `(arr25.ADES_mvt == "LIRF").all()` is
vacuously true on an empty frame; `test_synthetic_inputs_have_the_real_shapes` now asserts `len(arr25) > 50`
(tightened 2026-09-23 and re-rehearsed in round 3, `mutate_csf2_round3_report.json`: KILLED, so Z10 is NOT an exemption; round 3 also saw the six guard-branch tests red, 8/8).

## scripts/analysis/ctxg1_v26_build.py (rehearsal 2026-09-23, logs/ctxg1_scratch_tools/mutate_build.py, real-copy mirror)
- `candidate_values`: `(cand >= 1)` -> `(cand >= 0)` is EQUIVALENT: `bs.mixture` floors at 1.0 and the `cand == rint(mixture)`
  check runs first, so a candidate below 1 can never reach the floor check. Kept as belt-and-braces against a future change
  of the mixture floor.

## DFC2 — `load_frame`'s explicit `+ list(D.INPUT_COLS)` is an EQUIVALENT MUTANT today (2026-09-23, `logs/dfc2_scratch_tools/mutate_dfc2_report.json` M37)

`scripts/analysis/dfc2_build.py` `load_frame` keeps `INPUT_COLS = [ADES_mvt, doy, hr, tmin, mvt_sec]` explicitly so the
DFC block can be built from the loaded frame. Removing that term escapes `test_load_frame_keeps_the_dfc_inputs_and_the_leg_columns`
because every one of the five is ALREADY kept: `doy, hr, tmin, mvt_sec` are members of `stand_ab.BASELINE_FEATS` and
`ADES_mvt` is an encoding key (`stand_ab.ENC_KEYS`), both of which the keep-list includes. The term is kept on purpose as
the builder's input contract — if a future recipe drops `hr`/`tmin`/`mvt_sec` from the baseline the block must still
build — and the test would go RED the moment that happens, which is the behaviour wanted. Equivalent by the current
feature list, not by construction.

## scripts/analysis/omw_build.py (OMW, 2026-09-23, rehearsal `logs/omw_scratch_tools/mutate_omw.py`, 36/37 killed)
- **M37 `assert (om_age_s >= 0) & (om_age_s < 3600)` removed — EQUIVALENT.** `om_age_s = anchor - floor(anchor/3600)*3600`
  lies in [0, 3600) by arithmetic for every finite anchor, and a non-finite anchor is refused two lines earlier; no input
  reaches the assert in a failing state, so no test can see it red. Kept as a documented invariant, not a guard.

## DFC2 stage-2 additions — two EQUIVALENT mutants (2026-09-23, `logs/dfc2_scratch_tools/mutate_dfc2_stage2_report.json`)

- **S3 `serve_frame_2026` without the `ADEP_mvt.isin(SERVE_APTS)` clause**: `data/raw/ranking.parquet` holds movements at
  the ten scored airports only, so the clause removes zero rows today (339,551 either way). Kept as the serve filter's
  literal transcription of `stand_ab.build_features(serve=True)`; equivalent by the data file, not by construction.
- **S14 `fold_a_frame` without `+ list(D.INPUT_COLS)`**: the same equivalence as M37 above (`doy, hr, tmin, mvt_sec` are
  `BASELINE_FEATS` members and `ADES_mvt` an encoding key).

## tools/mem_guard.py — `--pause-on-phantom` addition (2026-09-28, rehearsal `logs/cml_scratch/mutation_memguard.txt`, 8/8 killed)
- **`phantom_heavy_alive`'s `if len(parts) != 2: continue` guard, mutated to a permissive fallback — EQUIVALENT for
  every input the tests (or real `ps -A -o pid=,command=` output) can produce.** The line is always followed by
  `pid = int(pid_s)` inside a `try/except ValueError: continue`, which already discards any line whose first token
  isn't a bare integer pid; a genuinely single-token line either has no second token to serve as `command` (so no
  heavy-job name can match, guard or no guard) or its sole token isn't a numeric pid (so the `int()` guard catches
  it regardless). Kept as an explicit, readable precondition on the two-column `ps` format rather than relying on
  the `int()` parse failure as an implicit one; replaced in the 8-mutant kill count by a mutant on the `int()` guard
  itself (removing its `try/except` — real kill, `tests/test_mem_guard.py::test_phantom_heavy_alive_skips_malformed_and_blank_lines`).

## scripts/analysis/trf_build.py — H-E2 additions (2026-09-28, rehearsal `logs/cml_scratch/mutation_he2.txt`, 9/10 killed = 90%)
- **`paired_dayblock_ci`'s `len(day) == n` term inside the combined length guard, mutated out
  (`if not (len(err_b) == len(airport) == len(day) == n)` -> drops `== len(day)`) — EQUIVALENT for the failure
  mode the test can reach.** With `err_a`/`err_b`/`airport` all matching `n` but `day` a different length, the
  mutated guard no longer raises, but `mtx_calib.gain_ci` immediately builds
  `pd.DataFrame({"g": g_row, "day": day, "month": airport})` from arrays of unequal length, and pandas itself
  raises `ValueError: All arrays must be of the same length` — which also matches the test's
  `pytest.raises(ValueError, match="same length")`, so the mutant is invisible to that specific assertion.
  Kept as an explicit, fail-fast precondition (a clear message naming `paired_dayblock_ci`, not a caller having
  to decode a pandas internals error) rather than relying on the downstream pandas check, which is an
  implementation detail of `gain_ci`/pandas, not a contract `paired_dayblock_ci` owns.
