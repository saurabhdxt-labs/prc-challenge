
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
