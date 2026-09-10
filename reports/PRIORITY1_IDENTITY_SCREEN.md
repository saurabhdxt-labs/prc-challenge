# Priority 1 — identity and scoping, screened before any fit

**2026-09-10 01:20 local.** Follows `plans/TOP_PATH_2026_09_10.md` Priority 1. No model was fitted
and nothing beneath the running arms queue was touched. Every number is an out-of-fold
single-key encoding measured on `data/cache_stand`, fitted on March–May 2025 and scored on June
(521,143 train / 180,610 test rows, `sd(delta)` 390.74).

**Two row sets, and they are not the same.** §1 reads `data/raw` with a `|delta| < 7200` filter
(521,155 train / 180,625 test); §2 reads `data/cache_stand` with no such filter (521,143 / 180,610).
That is why §1's "ap x stand 353.228" and §2's "STAND_mvt scoped 361.256" sit 8 s apart while naming
the same construction: the §1 filter removes the heavy tail that §2 keeps. **Neither number may be
compared across the two sections**, and only §2's paired scoped-vs-unscoped differences carry a
conclusion. Flagged by an adversarial review 2026-09-10 02:20; the first version left it
unreconciled.

**Read the limitation first.** A single-key encoding scored alone is a much weaker instrument than
a fit. It cannot say what a key is worth *inside* a 68-feature model. It is used here for two
questions it can answer: whether a key has enough support to be estimated at all, and whether two
versions of the *same* key differ. Nothing below is a projected gain.

---

## 1. Full flight identity is WEAKER than the 3-char airline prefix

Out-of-fold RMSE on June, each key alone (lower is better; the airport-only baseline is 361.587):

| key | levels | levels per 1k rows | OOF RMSE | vs airport alone |
|---|---|---|---|---|
| ap × stand | 2,491 | 3.5 | **353.228** | −8.360 |
| ap × operator | 2,377 | 3.4 | 354.746 | −6.841 |
| ap × ADES | 4,525 | 6.4 | 355.189 | −6.398 |
| ap × airline (3-char) | 4,220 | 6.0 | 355.207 | −6.380 |
| **ap × full FLIGHT** | **27,688** | **39.5** | **359.118** | **−2.469** |
| **ap × CALLSIGN** | **28,358** | **40.4** | **359.087** | **−2.501** |
| ap × runway | 65 | 0.1 | 359.450 | −2.138 |
| operator × ADES | 10,247 | 14.6 | 363.492 | **+1.904** (worse) |

Full flight number and callsign carry **~25 rows per level**. At `SMOOTH = 50` almost every level is
shrunk back toward the prior, so the extra granularity buys nothing and loses the airline-level
pooling. Operator × destination is worse than predicting the airport mean.

**What this does and does not establish.** It establishes that these keys are **support-limited as
standalone smoothed target encodings**, and that hierarchical fallbacks are not optional for them.
It does **not** refute the Priority 1 hypothesis, which proposes handing them to CatBoost as
**native categorical features** — ordered target statistics shrink differently, and a learner can
combine an identity column with the other 68 features rather than replacing them. Testing a key
alone and concluding about a key inside a model is the substitution error this project has already
paid for once tonight. **The marginal test — `model(X)` vs `model(X + identity)` — is the decisive
one and has not been run.**

Practical consequence for the build: full identity should enter as a **native categorical with a
count column and an airline-level fallback**, not as another smoothed mean.

---

## 2. The encoding keys are not airport-scoped, and more than half the rows are affected

Measured on `data/cache_stand`, June 2025 (180,610 rows):

| key | distinct names | names used at >1 airport | **rows carrying a shared name** |
|---|---|---|---|
| `STAND_mvt` | 1,586 | 457 | **99,938 — 55.3%** |
| `RUNWAY_mvt` | 43 | 11 | **80,502 — 44.6%** |

Stand `210` exists at **six** of the ten airports. Within one shared name, the per-airport mean
`delta` differs by a **median of 79 s, p90 317 s, max 728 s** — these are physically different
stands being averaged into one number.

> **CORRECTED 2026-09-10 06:40.** This paragraph previously read *"`ars` does not repair it. It is
> built as `STAND | RUNWAY` with no airport in the key."* **That is false.** `d_stand` is already
> airport-prefixed, so `ars` is `<airport>|<stand>|<runway>` — verified on **100% of cached rows**
> (`LFPG|I18|27L`, `LEMD|248|36L`). I read `stand_ab.py:340` without resolving the variable on the
> left of the concatenation. Found by the owner's review.

**`ars` DOES carry the airport, and it is the finest-grained key in the set** (7,208 levels against
7,199 distinct stand-runway pairs). So the model already holds one clean airport-scoped composite,
and `ap_ars` is redundant — it has been removed from `SCOPED_DEFAULT`, which now scopes **four**
keys, not five. What remains unscoped, and genuinely collides, is the four coarser keys below.

### Two DIFFERENT mechanisms, and the first version of this file conflated them

**CORRECTED 2026-09-10 08:55 after an adversarial review.** The table below was presented under the
heading "the encoding keys are not airport-scoped", implying one mechanism — name collision — for
all four keys. **That is only true for two of them.**

| key | names appearing at >1 airport | is that a COLLISION? |
|---|---|---|
| `STAND_mvt` | 457 of 1,586 (55.3% of rows) | **yes** — stand `210` at six airports is six different physical stands |
| `RUNWAY_mvt` | 11 of 43 (44.6% of rows) | **yes** — `08L` is a different strip of tarmac at each |
| `AIRCRAFT_TYPE_mvt` | 138 of 168 (99.8% of rows) | **no** — a B738 at LIRF is the same aircraft type as a B738 at EHAM |
| `ADES_mvt` | 583 of 950 (96.5% of rows) | **no** — `EGLL` is the same destination wherever you depart from |

So scoping splits into two claims:

* **STAND and RUNWAY: a repair.** The pooled statistic averages physically different objects, and
  the per-airport mean `delta` within one shared name differs by a median of 79 s. Fixing it
  recovers a corrupted number.
* **AIRCRAFT_TYPE and ADES: an interaction term.** Nothing is corrupted; `ap|B738` simply lets the
  encoder hold a per-airport effect for a globally-unique category. That is a legitimate feature,
  but it is **not** a repair, and a model that already carries `ADEP_mvt` and 67 other columns can
  in principle construct it by splitting.

**This matters for what to expect.** The two largest numbers below — aircraft type −24.102 and
destination −19.910 — are the two whose mechanism is *interaction*, which is the mechanism most
likely to be absorbed by the incumbent model. The first version of this file presented them as
evidence of a corrupted statistic. They are not.

### Scoped against unscoped — the same key, two versions

A paired comparison: identical rows, identical smoothing, identical fold. Only the key changes.

| key | unscoped OOF RMSE | airport-scoped | difference |
|---|---|---|---|
| `AIRCRAFT_TYPE_mvt` | 387.862 | **363.760** | **−24.102** |
| `ADES_mvt` | 383.027 | **363.116** | **−19.910** |
| `STAND_mvt` | 368.140 | **361.256** | **−6.884** |
| `RUNWAY_mvt` | 369.643 | **367.311** | **−2.333** |

Aircraft type is the largest: a B738's mean clock disagreement at LIRF is not its mean at EHAM, and
the shipped encoding averages them.

**A confound the arm must carry, named now.** The `scoped` variant has **32 encoding columns against
the incumbent's 24** (four scoped keys x te/de; `ars` is excluded as already scoped). Eight of that difference is new information, but eight extra columns is also extra
capacity, and a gain could come from either. The A/B as built cannot separate them; a capacity
control — eight permuted-value columns of the same shape — would, and is the honest companion run if
`scoped` wins.

**These magnitudes will not transfer.** The model already carries `ADEP_mvt` and 67 other columns
and recovers part of the pooling by splitting on airport. What the table establishes is that **the
statistic itself is corrupted on a majority of rows**, which is a defect independent of how much of
it the model currently repairs. The repair is nearly free — more encoding columns, no new data, no
new cache — so the marginal A/B is cheap to run.

---

## 3. Built

`prc.encoding.add_airport_scoped(keys, airport="ADEP_mvt", scope=SCOPED_DEFAULT)` appends
`ap_<key>` composites for **four** keys: `STAND_mvt`, `RUNWAY_mvt`, `AIRCRAFT_TYPE_mvt`, `ADES_mvt`.
**`ars` is excluded** — it is already `<airport>|<stand>|<runway>` (§2's correction).

Guards, each pinned by a test: the separator may not be NUL (the Arrow string path treats it as a
terminator and silently collapses composite keys — a landmine already paid for in this repo); a key
absent from the frame raises with its own name rather than pandas' bare `KeyError`; a null key
becomes the literal `"NA"` so it cannot swallow the airport prefix; the caller's frame is not
mutated.

**Tests: 15 in `tests/test_encoding_separation.py`, 38 across the three new files, all passing.
32 mutations rehearsed RED across the session, no survivors.** Two initially survived and both were
answered by strengthening the test rather than exempting it: deleting the explicit missing-key guard
(pandas raises its own bare `KeyError`), and disabling the useless-classifier AUC guard.

---

## 4. What is NOT done

- No fit. Nothing here is a measured gain in either direction.
- The scoped keys are not wired into any harness; `scripts/stand_ab.py` and `scripts/lgbm_fold.py`
  are byte-unchanged while the queue runs.
- Full identity needs an **identity cache**: `FLIGHT_mvt` and `CALLSIGN_flt` are **absent from
  `data/cache_stand`** (68 columns, neither present). Building one is a twelve-month raw pass and a
  second heavy job — it waits for the queue, per the one-heavy-job rule.
- Every number here was produced under the leaky early-stopping recipe (Priority 0). The scoped-key
  A/B should run on the corrected baseline, or it will need re-running.

## 5. Order

1. Queue drains (arm D refitting; Y, F, W to follow).
2. Priority 0 wiring + corrected baseline re-run.
3. **Scoped keys A/B** — cheapest of the three bets: no new data, no new cache, one paired fold run.
4. Conditional experts (`reports/PRIORITY2_FILL_LANE_PRICED.md`, prize ~1,782 MSE).
5. Identity cache + native-categorical CatBoost.
