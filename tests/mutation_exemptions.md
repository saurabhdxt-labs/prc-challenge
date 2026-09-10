
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
