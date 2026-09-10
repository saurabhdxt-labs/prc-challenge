# Pre-registration — arm B: a CONDITIONED body model for Rome's matched fill mixture

**Written 2026-09-10 10:25 local, BEFORE the harness exists and before any of its numbers exists.** Tag `B`.

**Why.** RESULT M (`plans/PREREG_rome_matched_fill_2026_09_10.md`): blending the fill branch over arm F
improves Rome's matched fill rows (555.6 → 443.7) but damages the body (+38.8 s; gated +7.1 s)
because F, an all-rows model, already contains fill mass. The body expert the mixture needs is
`E[delta | X, F = 0]`. Oracle on LIRF matched fill rows ≈ 4,435 board MSE.

## Arms (fold A; LIRF matched holdout rows; every other row keeps F)

**Body model:** arm F's design exactly (FEATS + QUEUE + ORDER, 83 columns, `L.P` params, A2 = no
per-airport models, seeds 0,1,2, best_iter re-found) — **with the training, fit and early-stopping
rows restricted to NON-FILL rows** (`|y − sp| > 60 s`), pooled over all ten airports. The holdout is
untouched. `body = max(proxy − delta_body, 1)`.
**p:** arm M's classifier unchanged (stake-weighted, all LIRF rows of the ten training months,
rome_fill features + NM-clock columns, mean of seeds 0,1,2).

| arm | LIRF matched rows |
|---|---|
| `F` | control |
| **`B`** | `max(p·sp + (1−p)·body, 1)` |
| `B_hyb` | `B` where `p ≥ 0.5`, F elsewhere (F is the better body model where fills are unlikely) |

## Thresholds, locked — RESULT M's five clauses, for B and B_hyb separately

1. paired row bootstrap (2,000, seed 0) on RMSE(F) − RMSE(arm) excludes zero in the arm's favour;
2. point gain ≥ **+5.0 s** LIRF matched RMSE; 3. months 1 and 7 both improve; 4. **body bounded:**
non-fill-row RMSE may exceed F's by ≤ **1.0 s**; 5. gain > 2 × seed sd (three single-seed versions,
body seed s paired with classifier mean). NOT WORKING iff (1) fails; else INCONCLUSIVE unless all pass.
**Reported, not decisional:** the body model's own RMSE on LIRF non-fill rows vs F's (its quality),
and the same on every airport; projected board MSE = 0.98466 × ΔSSE / 339,015.

## Cost and limits

One heavy fold fit (~2 h, ~5 GB) — runs only after v6 frees the machine. Fold A only. The body
model trains on ~91% of rows and loses the fill rows' information about the body — if its body-row
RMSE is worse than F's by more than the fill branch gains, B fails clause 4 and B_hyb is the test.

## AMENDMENT B.1 · 13:15 — BEFORE the body fit has started; no number of it exists

After v7's board reading (Rome unmatched transferred at ≈ 0.15×; matched lanes at 0.96–1.17×), one
secondary arm is added to the evaluate step. It uses the same conditioned body (the fit is pooled over
all ten airports already) and changes nothing about B / B_hyb or their clauses.

**`B_all`**: for EVERY airport's matched fold-A rows, `max(p_ap·sp + (1−p_ap)·body, 1)` where
`p_ap ≥ 0.5`, F elsewhere (the B_hyb form), with `p_ap` from an airport-specific classifier of the
same design (rome_fill params + stake weight + NM columns, trained on that airport's matched rows of
the ten training months, mean of seeds 0,1,2). Scored on ALL 339,015 matched rows against F with the
same five clauses, thresholds rescaled to the whole matched set: (1) interval excludes zero; (2) gain
≥ **+0.5 s** matched RMSE (≈ +220 board MSE); (3) both months improve; (4) body-row RMSE may exceed F's
by ≤ **0.2 s**; (5) gain > 2 × seed sd. Per-airport gains reported. Oracle context: all airports'
matched fill rows ≈ 7,148 board MSE (LIRF 4,435).

## AMENDMENT B.2 · 13:25 — BEFORE any arm B number exists (the body fit is still in early stopping)

Following `plans/FRESH_PATH_TO_246_2026_09_10.md` §3 (continuous probabilities primary; a fixed gate is
a fallback; shortlist by net squared error): one more arm on the same evaluate-all step —
**`B_all_cont`** = `max(p_ap·sp + (1−p_ap)·body, 1)` on EVERY airport's matched rows (no gate), same
classifiers and body. Same five clauses at B.1's bars. **Shortlist rule (compute allocation, not a
verdict):** an arm is carried to confirmation if its NET weighted fold MSE gain (0.98466 × ΔSSE /
339,015 over all matched rows) is ≥ **1,000**, whatever clause 4 says; clause verdicts are still
reported as registered and never revised.

## RESULT B · 2026-09-10 14:46 — the registered five clauses on LIRF matched fold-A rows

Source: `reports/rome_body.json` (`rome_body.py evaluate`, 14:46, exit 0), body fit
`reports/rome_body_fit.json` (best_iter 17,328, n_ref 21,677, seeds 0,1,2, 1,555,403 non-fill training
rows, wall 5,032 s, peak footprint 6.37 GB). 26,131 rows, fill share 18.93%.

| arm | RMSE (F 375.8976) | gain | 95% CI | by month (1 / 7) | body loss | seed sd | c1 | c2 | c3 | c4 | c5 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B** | 368.1379 | **+7.76 s** | [+3.59, +12.18] | +0.51 / +11.55 | **+3.49 s** | 0.255 | ✓ | ✓ | ✓ | **✗** | ✓ | **INCONCLUSIVE** |
| **B_hyb** | 369.2642 | **+6.63 s** | [+3.44, +10.08] | +2.37 / +8.92 | **+4.83 s** | 0.002 | ✓ | ✓ | ✓ | **✗** | ✓ | **INCONCLUSIVE** |

Clause 4 (non-fill rows may lose ≤ 1.0 s) fails for both; every other clause passes. Projected board MSE
(0.98466 × ΔSSE / 339,015; matched lane, LIRF rows only): B **+438**, B_hyb **+375**. Fill rows: F 555.6 →
B 518.2.

**Reported, not decisional — the body model itself is strongly better:** on LIRF non-fill rows the
conditioned body alone scores **298.55 vs F's 319.70** (−21.2 s). So clause 4's loss is NOT the body
model: it is the mixture weight `p` pulling non-fill rows toward `sp` (the classifier's mass on body
rows). The body expert hypothesis (`E[delta | X, F = 0]` beats F where F = 0) holds at Rome; the
Rome mixture as registered does not clear its body bound. Verdicts stand as written.

RESULT B.1 / B.2 (`evaluate-all`, all airports) pending — it builds a ~2.4 GB frame and runs only on AC power.

---

# RESULT B / B.1 / B.2 · 2026-09-10 14:48 local

Body fit: `reports/rome_body_fit.json` — non-fill training rows 1,555,403 (168,022 fill rows dropped), best_iter
17,328, n_ref 21,677, three seeds, wall 5,032 s, peak 6.4 GB (`/usr/bin/time`). Records `reports/rome_body.json`,
`reports/rome_body_all.json`, `data/cache_stand/rome_body_preds.parquet`.

| arm | rows | gain vs F [95%] | months 1 / 7 | body loss | fill-row RMSE F → arm | clauses 1–5 | verdict | net board MSE | shortlist (≥1,000) |
|---|---|---|---|---|---|---|---|---|---|
| B (LIRF, continuous) | 26,131 | **+7.76 s [+3.59, +12.18]** | +0.51 / +11.55 | +3.49 s | 555.6 → 518.2 | ✓ ✓ ✓ ✗ ✓ | INCONCLUSIVE | +438 | — (Rome-only arm) |
| B_hyb (LIRF, gated) | 26,131 | +6.63 s [+3.44, +10.08] | +2.37 / +8.92 | +4.83 s | 555.6 → 518.9 | ✓ ✓ ✓ ✗ ✓ | INCONCLUSIVE | +375 | — |
| B_all (all airports, gated; B.1) | 339,015 | +0.87 s [+0.47, +1.27] | +0.13 / +1.38 | +0.63 s | 285.6 → 272.9 | ✓ ✓ ✓ ✗ ✓ | INCONCLUSIVE | **+379** | **no** |
| B_all_cont (all, continuous; B.2) | 339,015 | +0.72 s [+0.13, +1.31] | −0.16 / +1.32 | +1.18 s | 285.6 → 269.7 | ✓ ✓ ✗ ✗ ✓ | INCONCLUSIVE | **+314** | **no** |

Per airport, B_all gains ≈ 0 everywhere except **LIRF +6.74 s**; B_all_cont LOSES at EDDM (−1.80), EHAM (−0.98),
EGLL (−0.79). **The fill-mixture lane is a Rome-only effect worth ≈ 380–440 board MSE** — below the shortlist rule.

**Diagnostic (reported, not decisional): why.** The body model trained WITHOUT fill rows beats F on Rome's
NON-FILL rows by **+21.2 s (319.7 → 298.5)** — Rome's fill rows contaminate F's fit of the body — and is within
±1.4 s of F on every other airport's non-fill rows, where fills are benign (F's fill-row RMSE 124–190 there).
Any future Rome lane should use a body fitted without Rome's fill rows; nothing else transfers from this arm.

## RESULT B.1 / B.2 · 2026-09-10 15:49 — all airports, both INCONCLUSIVE, neither shortlisted

Source: `reports/rome_body_all.json` (`rome_body.py evaluate-all`, exit 0, 79 s, peak 1.45 GB; built
`data/cache_rome/matched_all.parquet`). 339,015 matched fold-A rows, fill share 8.90%; F 222.5632.

| arm | RMSE | gain | 95% CI | by month (1 / 7) | body loss | seed sd | c1 | c2 (≥0.5 s) | c3 | c4 (≤0.2 s) | c5 | verdict | projected board MSE | shortlist (≥1,000) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B_all** (B.1, gated) | 221.6959 | +0.867 s | [+0.472, +1.274] | +0.13 / +1.38 | +0.633 s | 0.002 | ✓ | ✓ | ✓ | ✗ | ✓ | **INCONCLUSIVE** | **+379** | **no** |
| **B_all_cont** (B.2, continuous) | 221.8452 | +0.718 s | [+0.129, +1.315] | −0.16 / +1.32 | +1.185 s | 0.128 | ✓ | ✓ | ✗ | ✗ | ✓ | **INCONCLUSIVE** | **+314** | **no** |

Per airport (s): B_all is ≈ 0 everywhere but LIRF (+6.74) — the gate fires almost only at Rome. B_all_cont
gains at LIRF (+8.09) and LTFM (+0.34) but LOSES at EDDM −1.80, EHAM −0.98, EGLL −0.79, LEBL −0.39, LSZH
−0.39, LFPG −0.30: the continuous weight `p` pulls body rows toward `sp` where fills are rare (the risk
named before measuring). **Arm B closes at this scale** (ledger rule: < 1,000 measured). What survives,
reported not decisional: the conditioned body itself (LIRF non-fill −21.2 s vs F, RESULT B); the loss is
the mixture weight. A future arm would need `p` selected on the assembled squared error (plan §3) under
its own registration.
