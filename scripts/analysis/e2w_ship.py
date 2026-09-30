"""E2WS — candidate v33 = v31 with E2's frozen coefficients replaced by the W refit (stake-weighted Brier, fitted on all 12 months of
2025) on v31's 272 changed rows (plans/PREREG_e2w_ship_v33_2026_09_24.md). NO upload here.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/e2w_ship.py      # gates B1-B7, then writes the candidate

Reuses rse_build's tested sha256 / candidate / move_energy / day_energy / reference_p99 / apply_day_check and e2w_build's tested
design / predict / fit_brier_w / fit_logloss / lomo / arc_rule_all_months. 2026 DEP BLOCK/TAXITIME are dropped before any feature is
built. Writes only under submissions/ with open('x'); the final `merry-quicksand_v33.parquet` is a byte copy of the working candidate.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOARD_N = 344_841.0
BOARD_ROWS = 344_841


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("rse_build", ROOT / "scripts/analysis/rse_build.py")
W = _load("e2w_build", ROOT / "scripts/analysis/e2w_build.py")

PREREG = ROOT / "plans/PREREG_e2w_ship_v33_2026_09_24.md"
HASHLOG = ROOT / "reports/e2w_ship_prereg_hashes.log"
RULE, RULE_SHA = B.RULE, "bd01f4ed927d473585b6a182192073c59ed75fa220101db34035343c7629b606"
V31 = ROOT / "submissions/merry-quicksand_v31_rse.parquet"
V31_SHA = "c1b65a22907d0aa7cc066734beebdd39c8ea47f7ccc71f0fccf59dfd7295fac5"
V31_MAN = ROOT / "submissions/merry-quicksand_v31_rse.manifest.parquet"
V31_MAN_SHA = "d5a0e01268911a83d84d25b64634f6d0e0dae83b21cc0ca78d11f5639d609839"
REPORT = ROOT / "reports/e2w_report.json"
REPORT_SHA = "fca3ebeb0cf776222337a28c3e78f5e1df4ab5b1061cf325e3103f0d8d5e2993"
TEMPLATE = ROOT / "data/raw/submitting.parquet"
KEY = "merry-quicksand_v33.parquet"                       # == build_submission.submission_name(33), asserted in main
OUT_CAND = ROOT / "submissions/merry-quicksand_v33_e2w.parquet"
OUT_MAN = ROOT / "submissions/merry-quicksand_v33_e2w.manifest.parquet"
OUT_META = ROOT / "submissions/merry-quicksand_v33_e2w.meta.json"
OUT_FINAL = ROOT / "submissions" / KEY
N_S = 272
K_PT = 0.8453611792929924                                 # reports/e2w_report.json k_W_vs_L
K_HIST = {"RSE_realised": 1.150, "CXE_realised": -0.301}
N_BOOT, SEED = 2000, 0
SP_LO, SP_HI = 3600.0, 24000.0
BETA_TOL = 1e-6


def e2_form(p_R, k_in, turn, sp, beta) -> np.ndarray:
    """E2's form with coefficients `beta`; NaN where the frozen rule_E2 would not fire (3,600 <= sp <= 24,000, finite p_R, known
    stand i.e. k_in not NaN). With beta = E2's coefficients this equals rule_E2 (tested)."""
    p_R, k_in, turn, sp = (np.asarray(a, float) for a in (p_R, k_in, turn, sp))
    if not (len(p_R) == len(k_in) == len(turn) == len(sp)):
        raise ValueError(f"length mismatch: p_R {len(p_R)}, k_in {len(k_in)}, turn {len(turn)}, sp {len(sp)}")
    p = W.predict(W.design(p_R, k_in, turn), beta)
    ok = (sp >= SP_LO) & (sp <= SP_HI) & np.isfinite(p_R) & ~np.isnan(k_in)
    return np.where(ok, p, np.nan)


def k_ratio(D, e) -> float:
    """k = sum(D e) / sum(D^2): the share of the move that the truth follows (G = S (2k - 1))."""
    D, e = np.asarray(D, float), np.asarray(e, float)
    den = float(np.sum(D * D))
    if den == 0:
        raise ValueError("no move: sum(D^2) == 0, k undefined")
    return float(np.sum(D * e) / den)


def k_day_ci(D, e, day, n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    """Day-block bootstrap CI95 of k: whole days resampled with replacement, k = ratio of resampled sums."""
    D, e, day = np.asarray(D, float), np.asarray(e, float), np.asarray(day).astype(str)
    if not (len(D) == len(e) == len(day)):
        raise ValueError(f"length mismatch: D {len(D)}, e {len(e)}, day {len(day)}")
    num = pd.Series(D * e).groupby(day).sum().to_numpy()
    den = pd.Series(D * D).groupby(day).sum().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(num), size=(n_boot, len(num)))
    ks = num[idx].sum(1) / den[idx].sum(1)
    return float(np.quantile(ks, 0.025)), float(np.quantile(ks, 0.975))


def g_range(s: float, ks) -> tuple:
    return tuple(s * (2 * k - 1) for k in ks)


def row_set(scope_ids, changed_ids) -> np.ndarray:
    """Mask over the scope rows: the rows v31 changed. Every changed id must be a scope row."""
    scope_ids = np.asarray(scope_ids)
    missing = set(np.asarray(changed_ids).tolist()) - set(scope_ids.tolist())
    if missing:
        raise ValueError(f"{len(missing)} changed ids not in the scope rows, e.g. {sorted(missing)[:3]}")
    return np.isin(scope_ids, np.asarray(changed_ids))


def assemble(base: pd.DataFrame, ids, values) -> pd.DataFrame:
    """A copy of base with TAXITIME set to `values` on `ids` only; dtypes kept; values must be finite integers."""
    ids, values = np.asarray(ids), np.asarray(values, float)
    if not np.all(np.isfinite(values)) or not np.array_equal(values, np.rint(values)):
        raise ValueError("values must be finite and integral (rint before assembling)")
    pos = pd.Index(base.MVT_ID_mvt).get_indexer(ids)
    if (pos < 0).any():
        raise KeyError(f"{int((pos < 0).sum())} ids not in the base")
    out = base.copy()
    vals = out.TAXITIME_SEC_mvt.to_numpy().copy()
    vals[pos] = values.astype(vals.dtype)
    out["TAXITIME_SEC_mvt"] = vals
    return out


def check_rows(cand: pd.DataFrame, base: pd.DataFrame, dep_ids: set, n_rows: int = BOARD_ROWS) -> None:
    """B5: row count, id order and dtypes equal the base, and the ids are exactly the ranking DEP rows."""
    assert len(cand) == n_rows, f"B5: {len(cand)} rows != {n_rows}"
    assert np.array_equal(cand.MVT_ID_mvt.to_numpy(), base.MVT_ID_mvt.to_numpy()), "B5: id order differs from v31"
    assert list(cand.columns) == list(base.columns) and cand.dtypes.equals(base.dtypes), "B5: columns / dtypes differ from v31"
    assert set(cand.MVT_ID_mvt.tolist()) == set(dep_ids), "B5: ids are not the ranking DEP rows"


def check_beta(got, want, tol: float = BETA_TOL) -> None:
    got, want = np.asarray(got, float), np.asarray(want, float)
    assert got.shape == want.shape and np.max(np.abs(got - want)) <= tol, f"B2: beta {got.tolist()} != report {want.tolist()}"


def serve_rows_2026(scope: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    """Scope rows (p_C as p_R, nf) joined to ranking's context columns; DEP BLOCK/TAXITIME are dropped first (B6)."""
    rk = ranking.drop(columns=[c for c in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt") if c in ranking.columns])
    sc = scope[["MVT_ID_mvt", "p_C", "nf"]].rename(columns={"p_C": "p_R"})
    miss = ~sc.MVT_ID_mvt.isin(rk.MVT_ID_mvt)
    if miss.any():
        raise ValueError(f"{int(miss.sum())} scope ids missing from ranking")
    return sc.merge(rk, on="MVT_ID_mvt", how="left", validate="1:1")


def evidence_from_rows(X, fill, w, month, fit_mask, V, sp, nf, y, day, mixture, n_boot: int = N_BOOT,
                       ref_day=None) -> dict:
    """2025 evidence: beta_W on the fit rows (all months); W-vs-L 12-month LOMO g, k and day-block k CI on the verdict rows V;
    the RSE.1 reference (per-day move energy of rint mixture(W_all12) vs rint mixture(E2) on V); in-sample SSE shift."""
    X, w, sp, nf, y = (np.asarray(a, float) for a in (X, w, sp, nf, y))
    fill, fit_mask, V = np.asarray(fill, bool), np.asarray(fit_mask, bool), np.asarray(V, bool)
    day = np.asarray(day).astype(str)
    ref_day = day if ref_day is None else np.asarray(ref_day).astype(str)
    beta_W = W.fit_brier_w(X[fit_mask], fill[fit_mask], w[fit_mask])
    pL = W.lomo(X, fill, w, month, lambda X_, y_, w_: W.fit_logloss(X_, y_), fit_mask)
    pW = W.lomo(X, fill, w, month, W.fit_brier_w, fit_mask)
    mL, mW = mixture(pL, sp, nf), mixture(pW, sp, nf)
    g = float((((mL - y) ** 2 - (mW - y) ** 2)[V]).sum() / BOARD_N)
    D, e = (mW - mL)[V], (y - mL)[V]
    mE, mW12 = mixture(W.predict(X, W.E2_BETA), sp, nf), mixture(W.predict(X, beta_W), sp, nf)
    return {"beta_W": beta_W, "g_WL": g, "k": k_ratio(D, e), "k_ci": k_day_ci(D, e, day[V], n_boot=n_boot),
            "ref_day_energy": B.day_energy(np.rint(mE)[V], np.rint(mW12)[V], ref_day[V]),
            "sse_shift_insample": float((((mE - y) ** 2 - (mW12 - y) ** 2)[V]).sum() / BOARD_N)}


def largest_moves(ids, before, after, n: int = 10) -> list:
    ids, before, after = np.asarray(ids), np.asarray(before, float), np.asarray(after, float)
    d = after - before
    order = np.argsort(-np.abs(d), kind="mergesort")[:n]
    return [{"MVT_ID_mvt": float(ids[i]), "v31": float(before[i]), "v33": float(after[i]), "delta": float(d[i])} for i in order]


def load_2025(E, A):
    """The E2W rows exactly as e2w_build.main builds them (raw read filtered to the 1,488 ids; B2 checks equivalence)."""
    c = pd.read_parquet(ROOT / "reports/csf2/csf2_preds.parquet")[["MVT_ID_mvt", "month", "sp", "y", "day", "cls", "nf", "p_R"]]
    ids = c.MVT_ID_mvt.tolist()
    raw = pd.concat([pd.read_parquet(f, filters=[("MVT_ID_mvt", "in", ids)])
                     for f in sorted((ROOT / "data/raw").glob("training_2025-*.parquet"))], ignore_index=True)
    rows = c.merge(raw.drop(columns=["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]), on="MVT_ID_mvt", how="left",
                   validate="1:1").reset_index(drop=True)
    f = E.stand_inbound_features(rows)
    sp, nf = rows.sp.to_numpy(float), rows.nf.to_numpy(float)
    band = (sp >= SP_LO) & (sp <= SP_HI) & f.k_in.notna().to_numpy()
    R = _load("rrh_eval", ROOT / "scripts/analysis/rrh_eval.py")
    V = R.verdict_rows(sp, W.arc_rule_all_months(rows, A)) & band
    return dict(X=W.design(rows.p_R, f.k_in, f.turn), fill=(rows.cls == "fill").to_numpy(), w=(sp - nf) ** 2,
                month=rows.month.to_numpy(), fit_mask=band & (rows.cls != "slip").to_numpy(), V=V, sp=sp, nf=nf,
                y=rows.y.to_numpy(float), day=rows.day.to_numpy()), rows.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy()


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "scripts/analysis"))
    import build_submission as bs
    import arr_row_clocks as A
    assert bs.submission_name(33) == KEY, "key != build_submission.submission_name(33)"
    reg = B.sha256(PREREG)
    assert any(f"(full {reg})" in ln for ln in HASHLOG.read_text().splitlines()), "B1: registration sha not logged"
    for path, want in ((RULE, RULE_SHA), (V31, V31_SHA), (V31_MAN, V31_MAN_SHA), (REPORT, REPORT_SHA),
                       (B.SCOPE, B.SCOPE_SHA), (B.RANKING, B.RANKING_SHA)):
        assert B.sha256(path) == want, f"B1: {path.name} sha differs"
    for p in (OUT_CAND, OUT_MAN, OUT_META, OUT_FINAL):
        if p.exists():
            raise FileExistsError(f"{p} exists; no silent overwrite")
    E = _load("agent_E_rules", RULE)
    rep = json.loads(REPORT.read_text())

    # ---- 2025 evidence + B2
    d25, mvt_day25 = load_2025(E, A)
    ev = evidence_from_rows(**d25, mixture=bs.mixture, ref_day=mvt_day25)
    check_beta(ev["beta_W"], rep["beta_W_all12"])
    assert abs(ev["g_WL"] - rep["W_vs_L"]["g"]) <= 1e-6 * abs(rep["W_vs_L"]["g"]), f"B2: g {ev['g_WL']} != report"
    assert abs(ev["k"] - K_PT) <= 1e-6, f"B2: k {ev['k']} != report {K_PT}"
    k_lo, k_hi = ev["k_ci"]
    print(json.dumps({"beta_W": ev["beta_W"].tolist(), "g_WL": ev["g_WL"], "k": ev["k"], "k_ci": ev["k_ci"],
                      "sse_shift_insample_2025": ev["sse_shift_insample"]}, default=float), flush=True)

    # ---- 2026 rows
    sc = pd.read_parquet(B.SCOPE)
    sc["MVT_ID_mvt"] = sc.MVT_ID_mvt.astype("int64")
    rk = pd.read_parquet(B.RANKING)
    dep_ids = set(rk.MVT_ID_mvt[rk.PHASE_mvt == "DEP"].tolist())
    rows = serve_rows_2026(sc, rk)
    del rk
    f = E.stand_inbound_features(rows)
    sp = (rows.MVT_TIME_UTC_mvt - rows.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    nf = sc.nf.to_numpy(float)
    pE = e2_form(rows.p_R, f.k_in, f.turn, sp, W.E2_BETA)
    frozen = np.asarray(E.rule_E2(rows), float)
    assert np.array_equal(np.isnan(pE), np.isnan(frozen)) and np.allclose(pE[np.isfinite(pE)], frozen[np.isfinite(frozen)],
                                                                           rtol=1e-12, atol=0), "B3: e2_form != rule_E2"
    pW = e2_form(rows.p_R, f.k_in, f.turn, sp, ev["beta_W"])

    v31 = pd.read_parquet(V31)
    man = pd.read_parquet(V31_MAN)
    S = row_set(sc.MVT_ID_mvt.to_numpy(), man.MVT_ID_mvt[man.changed].astype("int64").to_numpy())
    assert int(S.sum()) == N_S, f"B3: row set {int(S.sum())} != {N_S}"
    v31_rows = v31.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(sc.MVT_ID_mvt.astype(float)).to_numpy(float)
    assert np.array_equal(v31_rows[S], np.rint(bs.mixture(pE[S], sp[S], nf[S]))), "B3: E2 does not rebuild v31 on the row set"
    new = B.candidate(v31_rows, S, pW, sp, nf, bs.mixture)
    day26 = rows.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy()
    p99 = B.reference_p99(ev["ref_day_energy"])
    e26 = B.day_energy(v31_rows, new, day26)
    S2 = B.apply_day_check(S, day26, e26, p99)                                                      # B7 (RSE.1)
    new = B.candidate(v31_rows, S2, pW, sp, nf, bs.mixture)

    cand = assemble(v31, sc.MVT_ID_mvt.to_numpy(float)[S2], new[S2])
    bs.check_submission(cand, pd.read_parquet(TEMPLATE))
    check_rows(cand, v31, dep_ids)
    same = cand.TAXITIME_SEC_mvt.to_numpy() == v31.TAXITIME_SEC_mvt.to_numpy()
    in_s = cand.MVT_ID_mvt.isin(sc.MVT_ID_mvt.to_numpy(float)[S2]).to_numpy()
    assert same[~in_s].all(), "B4: a row outside the row set changed"
    a31, a33 = v31.TAXITIME_SEC_mvt.to_numpy(float), cand.TAXITIME_SEC_mvt.to_numpy(float)
    s = B.move_energy(a31, a33)
    ks = (k_lo, K_PT, k_hi)
    mf = pd.DataFrame({"MVT_ID_mvt": cand.MVT_ID_mvt, "in_S": in_s, "v31": v31.TAXITIME_SEC_mvt.to_numpy(),
                       "candidate": cand.TAXITIME_SEC_mvt.to_numpy(), "changed": ~same})
    with open(OUT_CAND, "xb") as fh:
        cand.to_parquet(fh, index=False)
    with open(OUT_MAN, "xb") as fh:
        mf.to_parquet(fh, index=False)
    with open(OUT_FINAL, "xb") as fh:
        fh.write(OUT_CAND.read_bytes())
    assert B.sha256(OUT_FINAL) == B.sha256(OUT_CAND), "final copy differs from the candidate"
    month26 = sc.month.to_numpy()
    meta = {"arm": "E2WS (owner-ruled bet; E2W verdict NOT WORKING)", "prereg_sha256": reg, "rule_sha256": RULE_SHA,
            "base_v31_sha256": V31_SHA, "report_sha256": REPORT_SHA, "candidate_sha256": B.sha256(OUT_CAND),
            "manifest_sha256": B.sha256(OUT_MAN), "final_path": str(OUT_FINAL.relative_to(ROOT)),
            "final_sha256": B.sha256(OUT_FINAL), "upload_key": KEY,
            "beta_W": ev["beta_W"].tolist(), "beta_E2": W.E2_BETA.tolist(),
            "reproduced_2025": {"g_WL": ev["g_WL"], "k": ev["k"], "k_ci95_dayblock": [k_lo, k_hi],
                                "sse_shift_insample_W12_vs_E2": ev["sse_shift_insample"]},
            "n_S_before_day_check": int(S.sum()), "n_S": int(S2.sum()), "n_changed": int((~same).sum()),
            "sum_abs_delta_s": float(np.abs(a33 - a31).sum()), "move_energy": s,
            "move_energy_by_month": {str(m): B.move_energy(v31_rows[S2 & (month26 == m)], new[S2 & (month26 == m)])
                                     for m in (1, 7)},
            "k_low_point_high": list(ks), "G_low_point_high": list(g_range(s, ks)),
            "G_history": {k: s * (2 * v - 1) for k, v in K_HIST.items()},
            "largest_moves": largest_moves(cand.MVT_ID_mvt.to_numpy(), a31, a33),
            "day_check": {"p99": p99, "n_ref_days": len(ev["ref_day_energy"]), "n_2026_days": len(e26),
                          "failed_days": sorted(dd for dd, v in e26.items() if v > p99)}}
    with open(OUT_META, "x") as fh:
        json.dump(meta, fh, indent=1, default=float)
    print(json.dumps(meta, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
