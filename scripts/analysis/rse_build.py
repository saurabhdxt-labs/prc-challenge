"""RSE — candidate v31 = v30 with RRH's frozen E2 on the Rome mixture rows (plans/PREREG_rse_rome_stand_inbound_e2_2026_09_23.md).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/rse_build.py      # gates B1-B8, then writes the candidate (NO upload)

S = rows where E2 fires AND v30 == v13 (no later Rome rule, ARC/ARN/date-slip/floor, overwrote the mixture) AND sp < 24,000.
On S: rint(mixture(p_E2, sp, nf)) with the shipped `build_submission.mixture`; every other row is v30 byte for byte.
p_R (= shipped R's p_C) and nf come from the CTXG1 scope rows, verified to rebuild v13 exactly. Writes only under
submissions/ with open('x'). The upload key is `build_submission.submission_name(31)`; uploading is a separate step that
needs the owner's per-file go.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOARD_N = 344_841.0
PREREG = ROOT / "plans/PREREG_rse_rome_stand_inbound_e2_2026_09_23.md"
HASHLOG = ROOT / "reports/rse_prereg_hashes.log"
RULE = ROOT / "data/rome_hunt_2026_09_23/agent_E_rules.py"
RULE_SHA = "bd01f4ed927d473585b6a182192073c59ed75fa220101db34035343c7629b606"
V30 = ROOT / "submissions/merry-quicksand_v30_oru.parquet"
V30_SHA = "d5aadef1af88215d1a75710af82b47e2c0cdad30d981d854192fb2d1a8831112"
SCOPE = ROOT / "data/pipeline_runs/ctxg1_20260913T131202Z/ctxg1_scope_rows.parquet"
SCOPE_SHA = "4fb712a83a844c20e87950ac4a39aeb971fe2e29b0796c8047ea25bdc12fe077"
RANKING = ROOT / "data/raw/ranking.parquet"
RANKING_SHA = "9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86"
RRH_REPORT = ROOT / "reports/rrh_holdout_report.json"
OUT_CAND = ROOT / "submissions/merry-quicksand_v31_rse.parquet"
OUT_MAN = ROOT / "submissions/merry-quicksand_v31_rse.manifest.parquet"
OUT_META = ROOT / "submissions/merry-quicksand_v31_rse.meta.json"
SP_CAP = 24_000.0
N_S = 273
S_SHIFT = 831.45
K_LO, K_PT, K_HI = 0.538, 0.947, 1.412
#: RSE.1: a 2026 day passes iff its move energy per moved row <= this quantile of the 2025 day distribution
DAY_Q = 0.99


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_set(fired, v30, v13, sp) -> np.ndarray:
    """S: E2 fires, v30 still carries the mixture value v13, and no later band rule owns the row (sp < 24,000)."""
    return (np.asarray(fired, bool) & (np.asarray(v30, float) == np.asarray(v13, float))
            & (np.asarray(sp, float) < SP_CAP))


def candidate(v30, S, p, sp, nf, mixture) -> np.ndarray:
    """v30 with rint(mixture(p, sp, nf)) on S only."""
    v30, S, p = np.asarray(v30, float), np.asarray(S, bool), np.asarray(p, float)
    if np.isnan(p[S]).any():
        raise ValueError(f"{int(np.isnan(p[S]).sum())} NaN p_E2 on S rows")
    out = v30.copy()
    out[S] = np.rint(mixture(p[S], np.asarray(sp, float)[S], np.asarray(nf, float)[S]))
    return out


def move_energy(before, after) -> float:
    return float(np.sum((np.asarray(after, float) - np.asarray(before, float)) ** 2) / BOARD_N)


def g_range(s: float) -> tuple:
    """G = S (2k - 1) at the registered k low / point / high."""
    return tuple(s * (2 * k - 1) for k in (K_LO, K_PT, K_HI))


def day_energy(before, after, day) -> dict:
    """{day: move energy per moved row} over the days with at least one moved row (RSE.1 e_d)."""
    before, after, day = np.asarray(before, float), np.asarray(after, float), np.asarray(day).astype(str)
    moved = after != before
    out = {}
    for d in np.unique(day[moved]):
        m = (day == d) & moved
        out[str(d)] = float(np.sum((after[m] - before[m]) ** 2) / BOARD_N / m.sum())
    return out


def reference_p99(energies: dict) -> float:
    if not energies:
        raise ValueError("empty reference: no 2025 day has a moved row")
    return float(np.quantile(np.array(list(energies.values()), float), DAY_Q))


def apply_day_check(S, day, energies: dict, p99: float) -> np.ndarray:
    """RSE.1 B8': S rows on a day with e_d > p99 revert to the base (drop out of S)."""
    S, day = np.asarray(S, bool), np.asarray(day).astype(str)
    bad = {d for d, e in energies.items() if e > p99}
    return S & ~np.isin(day, sorted(bad))


def reference_days_2025(E, mixture) -> dict:
    """The 2025 reference: every LIRF unmatched row (all 12 months) where E2 fires with sp < 24,000, stored R LOMO p
    and nf (CSF.2), rint mixture; e_d per UTC day of MVT."""
    c = pd.read_parquet(ROOT / "reports/csf2/csf2_preds.parquet")[["MVT_ID_mvt", "sp", "nf", "p_R"]]
    raw = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "data/raw").glob("training_2025-*.parquet"))],
                    ignore_index=True)
    rows = c.merge(raw.drop(columns=["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]), on="MVT_ID_mvt", how="left",
                   validate="1:1")
    p = np.asarray(E.rule_E2(rows), float)
    sp = rows.sp.to_numpy(float)
    fire = np.isfinite(p) & (sp < SP_CAP)
    base = np.rint(mixture(rows.p_R.to_numpy(float), sp, rows.nf.to_numpy(float)))
    new = np.where(fire, np.rint(mixture(np.where(fire, p, 0.5), sp, rows.nf.to_numpy(float))), base)
    return day_energy(base, new, rows.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy())


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reproduce_holdout(E, mixture) -> float:
    """B2: the frozen rule's RRH verdict-row g, recomputed, must equal the hashed report's to 1e-6 relative."""
    R = _load("rrh_eval", ROOT / "scripts/analysis/rrh_eval.py")
    rows = R.load_holdout()
    V = R.verdict_rows(rows.sp, R.arc_rule_for(rows))
    serve = rows.drop(columns=["y", "cls", "nf", "day", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
    p = np.asarray(E.rule_E2(serve), float)
    r = R.score_rule(p[V], *(np.asarray(a)[V] for a in (rows.p_R, rows.sp, rows.nf, rows.y, rows.month, rows.day)),
                     mixture)
    want = json.loads(RRH_REPORT.read_text())["rules"]["agent_E_rules.rule_E2"]["verdict_rows"]["g"]
    if abs(r["g"] - want) > 1e-6 * abs(want):
        raise AssertionError(f"B2: holdout g {r['g']} != RRH {want}")
    return r["g"]


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_submission as bs
    reg = sha256(PREREG)
    if not any(f"(full {reg})" in ln for ln in HASHLOG.read_text().splitlines()):
        raise AssertionError("B1: registration sha not in its hash log")
    for path, want in ((RULE, RULE_SHA), (V30, V30_SHA), (SCOPE, SCOPE_SHA), (RANKING, RANKING_SHA)):
        if sha256(path) != want:
            raise AssertionError(f"B1: {path.name} sha differs from the registration")
    for p in (OUT_CAND, OUT_MAN, OUT_META):
        if p.exists():
            raise FileExistsError(f"{p} exists; no silent overwrite")
    E = _load("agent_E_rules", RULE)
    g_h = reproduce_holdout(E, bs.mixture)

    sc = pd.read_parquet(SCOPE)
    sc["MVT_ID_mvt"] = sc.MVT_ID_mvt.astype("int64")
    rk = pd.read_parquet(RANKING)
    rows = sc[["MVT_ID_mvt", "p_C"]].rename(columns={"p_C": "p_R"}).merge(
        rk.drop(columns=["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]), on="MVT_ID_mvt", how="left", validate="1:1")
    sp = (rows.MVT_TIME_UTC_mvt - rows.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    if not np.allclose(sp, sc.sp.to_numpy()):
        raise AssertionError("scope sp does not match the ranking file")
    if not np.array_equal(np.rint(bs.mixture(sc.p_C.to_numpy(), sp, sc.nf.to_numpy()))[sc.in_scope.to_numpy()],
                          sc.v13.to_numpy()[sc.in_scope.to_numpy()]):
        raise AssertionError("p_C / nf do not rebuild v13 on the scope rows")
    p = np.asarray(E.rule_E2(rows), float)

    v30 = pd.read_parquet(V30)
    col = v30.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    v30_rows = col.reindex(sc.MVT_ID_mvt).to_numpy(float)
    S = row_set(np.isfinite(p), v30_rows, sc.v13.to_numpy(float), sp)
    if int(S.sum()) != N_S:
        raise AssertionError(f"B3: |S| {int(S.sum())} != registered {N_S}")
    new_rows = candidate(v30_rows, S, p, sp, sc.nf.to_numpy(), bs.mixture)
    s_shift = move_energy(v30_rows, new_rows)
    if abs(s_shift - S_SHIFT) > 0.01:
        raise AssertionError(f"B6: move energy {s_shift:.4f} != registered {S_SHIFT} (before the RSE.1 day check)")
    day = rows.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy()
    ref = reference_days_2025(E, bs.mixture)
    p99 = reference_p99(ref)
    e26 = day_energy(v30_rows, new_rows, day)
    S = apply_day_check(S, day, e26, p99)                                                        # B8' (RSE.1)
    new_rows = candidate(v30_rows, S, p, sp, sc.nf.to_numpy(), bs.mixture)
    s_shift = move_energy(v30_rows, new_rows)
    failed = sorted(d for d, e in e26.items() if e > p99)
    day_check = {"p99": p99, "n_ref_days": len(ref), "n_2026_days": len(e26), "failed_days": failed,
                 "n_S_after": int(S.sum()), "move_energy_after": s_shift,
                 "per_month_after": {str(m): move_energy(v30_rows[sc.month.to_numpy() == m],
                                                         new_rows[sc.month.to_numpy() == m]) for m in (1, 7)}}
    print(json.dumps({"day_check": day_check}, indent=1, default=float), flush=True)

    cand = v30.copy()
    upd = pd.Series(new_rows, index=sc.MVT_ID_mvt)
    in_s = cand.MVT_ID_mvt.isin(sc.MVT_ID_mvt[S]).to_numpy()
    vals = cand.TAXITIME_SEC_mvt.to_numpy().copy()
    vals[in_s] = upd.reindex(cand.MVT_ID_mvt[in_s]).to_numpy()
    cand["TAXITIME_SEC_mvt"] = vals.astype(v30.TAXITIME_SEC_mvt.dtype)
    bs.check_submission(cand, v30)                                                                 # B5
    if not (np.array_equal(cand.MVT_ID_mvt.to_numpy(), v30.MVT_ID_mvt.to_numpy())
            and list(cand.dtypes) == list(v30.dtypes)):
        raise AssertionError("B5: ids / order / dtypes differ from v30")
    same = cand.TAXITIME_SEC_mvt.to_numpy() == v30.TAXITIME_SEC_mvt.to_numpy()
    if not same[~in_s].all():                                                                      # B4
        raise AssertionError("B4: a row outside S changed")
    man = pd.DataFrame({"MVT_ID_mvt": cand.MVT_ID_mvt, "in_S": in_s, "v30": v30.TAXITIME_SEC_mvt.to_numpy(),
                        "candidate": cand.TAXITIME_SEC_mvt.to_numpy(), "changed": ~same})
    with open(OUT_CAND, "xb") as fh:
        cand.to_parquet(fh, index=False)
    with open(OUT_MAN, "xb") as fh:
        man.to_parquet(fh, index=False)
    lo, pt, hi = g_range(s_shift)
    meta = {"arm": "RSE", "prereg_sha256": reg, "rule_sha256": RULE_SHA, "base_v30_sha256": V30_SHA,
            "candidate_sha256": sha256(OUT_CAND), "manifest_sha256": sha256(OUT_MAN),
            "upload_key": bs.submission_name(31), "n_S": int(S.sum()), "n_changed": int((~same).sum()),
            "holdout_g_reproduced": g_h, "move_energy": s_shift, "G_low_point_high": [lo, pt, hi],
            "k_low_point_high": [K_LO, K_PT, K_HI], "day_check_B8prime": day_check}
    with open(OUT_META, "x") as fh:
        json.dump(meta, fh, indent=1, default=float)
    print(json.dumps(meta, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
