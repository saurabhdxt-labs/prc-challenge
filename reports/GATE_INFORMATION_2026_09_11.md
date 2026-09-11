# Where can a regime gate commit? — diagnostic `GID` (after arm REG)

Session prc-challenge-c4. Owner: "yes" (run the gate-information diagnostic).

## Part 1 — written BEFORE any number of this diagnostic was read · 2026-09-11 10:00:59 (from `date`)

**Known going in** (RESULT REG): the four-regime gate ranks early-off-block rows well (AUC 0.961) but its calibrated
probability never exceeds ≈ 0.33 in any decile, so the mixture never commits; the oracle gate is worth +22,959. The
question is whether that pooled picture hides airports, days or features where the regime IS identifiable — the places a
multi-model lane can win on today's inputs — and what kind of information separates the regime where it can be seen.

**Instrument (fixed now; descriptive, no fitting).** Source: `data/cache_stand/regime_experts_preds.parquet` (read through
`prc.pipeline.scoring.read_record`, convention `taxi_time`, tag REG) and the holdout rows' stand-cache features (joined on
`row`, y equal row for row).
- **D1 per airport:** early share; the gate's early AUC; precision in the top 5% of p_early; the number of rows with
  p_early ≥ 0.5 and their precision; the oracle's early gain vs REG's early gain (weighted fold MSE).
- **D2 commitment:** across all airports, how many early rows sit in cells where the gate is ≥ 50% right.
- **D3 day clustering:** per airport, the share of its early rows on its 5 worst days (of 62); a clustered regime points
  to a day-level cause (disruption, regulation, de-icing) that a day-level serve-time witness could carry.
- **D4 information:** the univariate AUC (early vs rest) of every numeric stand-cache feature on holdout rows, pooled and
  at the two largest early-regime airports — which existing inputs carry the signal.

**Pre-written expectations (to be compared, not decisional — this routes the next registration).**
- **Separable somewhere:** at ≥ 1 airport, ≥ 20% of its early rows sit where the gate is ≥ 50% right → a per-airport gate /
  expert config is the next arm there.
- **Not separable anywhere:** no airport reaches that → today's inputs cannot drive the gate; the next arm must add
  information (ADS-B block, a day-level witness if D3 shows clustering).
- **Day clustering:** if the top-5 days hold ≥ 40% of an airport's early rows, the regime is episodic there.

## Part 2 — RESULT GID · 2026-09-11 10:03:03 (from `date`)

Source `reports/gate_info.json` (`scripts/gate_info.py`, 1.7 s; tests `tests/test_gate_info.py` 3; the AUC matches
sklearn's). Features joined on `row` with y equal row for row.

**D2 — the gate DOES commit.** p_early ≥ 0.5 on 7,824 holdout rows at **74.8% precision**, covering **46%** of all early
rows (5,855 of 12,754). **D1 per airport** (commit = p_early ≥ 0.5): EDDM covers 80% of its early rows at 83% precision,
EGLL 67% at 87%, LIRF 45% at 66%, LEMD 43% at 84%, LTFM 38% at 70%, LEBL 33% at 74%, EDDF 31% at 74%, LFPG 24% at 61%,
LSZH 18% at 60%, EHAM 50% at 91% (on 23 rows). The "separable somewhere" expectation is met at 9 of 10 airports (all but
LSZH).

**Yet REG's early rows gained nothing, and this is why** (split by the gate's confidence, weighted fold MSE vs BASE):

| early-off-block rows | rows | BASE mean error | REG gain | ORACLE gain |
|---|---|---|---|---|
| gate confident (p ≥ 0.5) | 5,855 | +112 s | +270 | +696 |
| **gate unsure (p < 0.5)** | **6,899** | **+532 s** | −491 | **+7,054** |
| not early, gate confident | 1,969 | −250 s | −159 | +1,726 |

Where the regime is visible in today's inputs, the single model already prices it (its strongest univariate signals,
D4: `aobt_eobt` AUC 0.856, `aobt_sched` 0.807, `iobt_p`, `eobt_p`, `arvt_diff`, `lobt_p`, `sp` — all NM-clock versus
plan). The value is in early rows that look ordinary to every input: 67% of the early regime's error, BASE off by +532 s,
off-blocks typically just over 10 min before NM's (median |δ| 779 s vs 1,144 s where the gate is confident).
**D3:** not episodic — the top 5 of 62 days hold 20–26% of an airport's early rows (EHAM 62% on a 0.1% share).

**Against the expectations:** "separable somewhere" appeared (9 of 10 airports) — but it does not route to a
per-airport gate arm, because the separable part is already inside the single model. The routing it gives instead:
the next arm needs information about the flights whose NM clocks look ordinary — their actual off-block (ADS-B ground
movement where it exists) or an operational signal not yet in the inputs. Not a day-level witness (D3).
