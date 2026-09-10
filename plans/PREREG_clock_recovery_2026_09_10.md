# Pre-registration — recovering the missing NM clock from the same flight's arrival record

**Written 2026-09-10 02:10 local, BEFORE any number below the line is computed.** Standalone file,
so it cannot collide with the concurrent session's edits to `plans/PREREG_taxiout_2026_09_08.md`.

## 0. Why this lane, and why now

`plans/TOP_PATH_2026_09_10.md` Priority 4. The 5,290 unmatched scored rows have **every** `*_flt`
column null — no Network Manager record — so the `proxy = MVT_TIME − AOBT_3` reparameterisation that
carries the other 98.47% has nothing to lean on. But `ranking.parquet` holds 344,693 **arrival**
rows as well as our departures, and the arrival leg of the *same outbound flight* at its destination
may carry that flight's NM record, `AOBT_3_flt` included.

The owner's audit found **227 candidates in `ranking.parquet`** and 273 on fold A, with candidate
agreement against the known flight id at 99.93% / 99.79% / 99.94% on the three files checked. It
also priced the ceiling: **~1,355 weighted fold MSE even with perfect predictions on every
candidate.** That is a secondary lane, not a first-place thesis, and the plan caps it at one day.

**This is chosen now because it is the only lane that is pure measurement.** The arms queue owns the
machine until ~08:00 and no heavy fit may start; this reads parquet month by month.

**The audit's precision figure is not the thing that matters, and this pre-registration does not
inherit it.** It was measured on pairs whose flight id is already known on both legs — i.e. on
matched rows. Unmatched rows are selected for being hard to match, so the number need not transfer.
Measuring transfer on genuinely unmatched rows is the whole point of the experiment.

## 1. Hypotheses

**H-C1 (coverage).** A same-flight arrival candidate exists for a non-trivial share of *genuinely
unmatched* departures in the twelve training months.

**H-C2 (correctness).** The `AOBT_3` recovered from that candidate is the departure's true off-block
clock, in the sense that the implied `delta = BLOCK_TIME − AOBT_3_recovered` on those rows has the
same shape as `delta` on ordinary matched rows.

**H-C3 (value).** Predicting those rows through the recovered clock beats the shipped unmatched
estimator on them.

## 2. The matching rule, fixed now

A departure row `d` (`PHASE_mvt == "DEP"`, `ADEP_mvt` in the ten airports, `AOBT_3_flt` null) is
matched to an arrival row `a` iff **all** of:

1. `a.PHASE_mvt == "ARR"`
2. `a.FLIGHT_mvt == d.FLIGHT_mvt` (exact, case-sensitive, nulls never match)
3. `a.ADEP_mvt == d.ADEP_mvt` and `a.ADES_mvt == d.ADES_mvt`
4. `0 < a.MVT_TIME_UTC_mvt − d.MVT_TIME_UTC_mvt <= 6 hours`
5. **exactly one** arrival satisfies 1–4; zero or two or more is NO MATCH, never a best guess
6. `a.AOBT_3_flt` is non-null

`AOBT_3_recovered := a.AOBT_3_flt`. No callsign aliasing, no fuzzy flight numbers, no widened
window — the plan requires new coverage evidence before any of those, and this run does not produce
it.

## 3. Thresholds, locked

Measured on the twelve 2025 training months, on **genuinely unmatched departures with a label**
(never on artificially masked matched rows — those are an easier problem and would flatter every
number here).

- **H-C1 SUPPORTED** iff coverage >= **2%** of genuinely unmatched labelled departures. Below that
  the lane cannot reach its own ceiling and is closed.
- **H-C2 SUPPORTED** iff, on the matched candidates, the median |`delta_recovered`| is **<= 600 s**
  AND at least **80%** of candidates have |`delta_recovered`| <= 3,600 s. A recovered clock that is
  really the same flight's off-block should disagree with the airport's block stamp by minutes, as
  it does on ordinary matched rows (median `delta` +53 s, p95 535 s). Hours means the wrong leg.
- **H-C3 SUPPORTED** iff, on the candidate rows, the paired squared-error comparison against the
  shipped unmatched estimator excludes zero in the recovery's favour (2,000-draw paired row
  bootstrap, seed 0) AND the weighted MSE removed exceeds **200**.
- Any clause failing is **NOT WORKING** for that hypothesis, stated separately per hypothesis.

**Reported but NOT decisional:** per airport (the repo rule — LIRF behaves unlike the other nine);
per month; the same statistics computed on artificially masked matched rows, purely to quantify how
much the easy version flatters the hard one.

## 4. What a positive result licenses

Building the recovery as a serve-time feature plus a confidence flag, and a controlled blend against
the current unmatched estimator. It licenses **no submission** on this evidence, and it does **not**
license equating the recovered NM clock with the hidden airport off-block — that clock difference is
the original problem, not a solved one.

## 5. Budget

**One day, per the plan's cap.** If H-C1 or H-C2 fails, the lane closes here and no widened matching
rule is attempted on this evidence.

---

# RESULT C · 2026-09-10 02:25 local · H-C1 SUPPORTED, H-C2 SUPPORTED, H-C3 NOT WORKING as registered

Measured on all twelve 2025 training months, on **genuinely unmatched labelled departures** — never
on artificially masked matched rows. Matching rule exactly as §2 fixed it. No model was fitted.

## C.1 Coverage — H-C1 SUPPORTED

| | measured |
|---|---|
| genuinely unmatched labelled departures, 12 months | 22,405 |
| exactly-one same-flight arrival candidate | **902 = 4.03%** |
| bar | >= 2% |

## C.2 Correctness — H-C2 SUPPORTED, and this is the finding worth keeping

`delta_recovered = BLOCK_TIME − AOBT_3_recovered` on the 902 candidates, against the shape of
`delta` on the ordinary MATCHED population:

| statistic | recovered, on unmatched rows | ordinary matched rows |
|---|---|---|
| median | −61 s | +53 s |
| **p95** | **535 s** | **535 s** |
| p05 / p25 / p75 | −2,084 / −299 / +116 | — |
| median absolute | **184 s** (bar <= 600) | — |
| share within 3,600 s | **95.6%** (bar >= 80%) | — |

The p95 agrees to the second. **The clock recovered from the arrival leg is the same flight's
off-block clock** — not a coincidental match, not the wrong leg. The evidence is independent of any
model: it is the agreement in distribution between a recovered quantity on 902 unmatched rows and
the same quantity measured on 339,015 matched rows that were never involved in the matching.

Per airport, median |delta_recovered|: EDDF 111, LEMD 122, LEBL 176, LSZH 172, EDDM 182, LFPG 182,
EGLL 240, EHAM 273, LIRF 361, LTFM 389.

## C.3 Value — H-C3 NOT WORKING

Paired against the shipped stratum estimator `S1` (the v4 hybrid) on the same rows, joined 1:1 by
`MVT_ID_mvt`. Reported both pooled and ex-monster, per Amendment 9 rule 4.

| subset | n | shipped S1 | recovery (`y = proxy_rec`) | mean sq-err reduction | 95% CI | clears |
|---|---|---|---|---|---|---|
| pooled | 902 | **3,057.0** | 3,367.4 | −1,993,876 | [−3,328,262, −916,393] | no |
| **ex-monster (decisional)** | 896 | **959.4** | 1,294.2 | −754,518 | [−1,192,397, −379,649] | **no** |

**VERDICT: NOT WORKING as registered.** The registered predictor loses decisively, on both subsets,
by intervals that exclude zero in the *shipped estimator's* favour.

## C.4 What this does and does NOT establish — read this before closing the lane

**It does not establish that the recovered clock is uninformative.** §3's H-C3 registered the
crudest possible use of it — `y = proxy_rec`, i.e. assuming `delta = 0` — against a *fitted* mixture
that exploits `sp` and cell structure. A naive predictor losing to a fitted one is not evidence
about the information content of its input. That is the substitution error this project has already
paid for twice in twenty-four hours (the `|delta|`-band oracle, and CatBoost-without-categoricals),
and it is not repeated here as a conclusion.

What the result does establish, precisely:

1. **The cheap use of the recovery is refuted.** Substituting `proxy_rec` for the shipped estimator
   on candidate rows makes things worse, and by a wide margin. No submission may do this.
2. **C.2 stands on its own.** The clock is genuinely recovered. Whether a *fitted* use of it —
   `proxy_rec` and a confidence flag as FEATURES in the unmatched estimator, rather than as a
   replacement for it — beats the incumbent is **untested and needs a fit**.
3. **The reason the naive version fails is visible and mechanical:** on unmatched rows `delta` is not
   small, and the shipped mixture's `p·sp` branch is what catches the extreme rows. Predicting
   `proxy_rec` throws that away. Pooled and ex-monster both degrade, so this is not only a monster
   effect.

Per airport, ex-monster (reported, NOT decisional — the repo rule forbids selecting airport
exclusions by inspecting a fold): LTFM +639,462 and LEMD +56,780 and EDDF +17,852 improve; EHAM
−1,615,883 and LIRF −5,817,204 collapse. **No airport-selected rule is proposed on this evidence.**

## C.5 Budget decision

The plan caps this lane at one day and prices its ceiling at ~1,355 weighted fold MSE even with
perfect predictions on every candidate. That ceiling is barely above the 1,000 MSE promotion bar,
the cheap use is now refuted, and the fitted use costs a fold run that the queue currently blocks.

**Recommendation: the lane stays open but does not get the fit until the two larger bets — the
conditional experts and the scoped keys — have been measured.** Its one durable output is C.2, which
is worth keeping regardless: a 4.03% subset of the unmatched stratum has a real, recoverable NM
clock, and that fact is now measured rather than assumed.

**No widened matching rule** (callsign aliases, fuzzy flight numbers, a longer window) is attempted,
per §5.
