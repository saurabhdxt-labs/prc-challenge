# Mutation exemptions — surviving mutants judged semantically equivalent

Each entry records a mutation that survived the suite, why no test can kill it, and what would
change that. An entry here is a claim that the mutant is EQUIVALENT ON THE AVAILABLE FIXTURE —
never that the code is unimportant.

---

## ME-1 · deleting the positivity floor in `scripts/cond_experts.py` (the `yhat_cond` line)

**Mutation:** `yhat_cond = np.maximum(assemble(p_te, np.maximum(mu_fill, 1.0), mu_body), 1.0)`
→ `yhat_cond = assemble(p_te, mu_fill, mu_body)`.

**Survives.** Rehearsed 2026-09-10; the suite stays green.

**Why no test kills it.** The floor is `max(raw, 1)` and it only changes a value when a predicted
taxi time comes out below one second. On the synthetic smoke fixture that never happens — the arm
reports `floor_binds_rows = 0` — so removing the floor is observationally identical there. On the
REAL fold it binds on 23 of 339,015 baseline rows, which is exactly how the
floored-vs-unfloored error in `reports/bug_classes.md` BC-2 arose.

**What IS tested instead.** `tests/test_cond_experts.py::test_conditional_prediction_carries_the_
shipped_positivity_floor` calls `CE.assemble` directly with a mixture that must clamp, so the floor
SEMANTICS are pinned even though the call site cannot be. The arm also emits `floor_binds_rows` in
its record, so a real run makes the binding count visible rather than silent.

**What would kill it.** A fixture whose predictions cross zero — i.e. a synthetic month with a
proxy small enough that `proxy − delta_hat < 1`. Not built: it would require perturbing
`tests/synthetic_caches.py`, which several other test files assert planted magnitudes against
(`test_lgbm_fold.py`), and changing it would invalidate those.

**Do not remove the floor on the strength of this exemption.** The exemption says the fixture
cannot see the difference; the real fold can.
