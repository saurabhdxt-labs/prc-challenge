
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

## `w_lowvis`: `<` vs `<=` at 1.5 km is an EQUIVALENT mutant (2026-09-09, Amendment 24)

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
