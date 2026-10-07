"""INDEPENDENT verifier for submissions/merry-quicksand_v29_orl.parquet (arm ORL Stage 3,
plans/PREREG_orl_osm_route_length_2026_09_23.md). Dry run by default; only `--upload` uploads, and only after the
owner's explicit per-file "go" AND a Stage-3 RESULT appended + hashed in the registration.

    /opt/homebrew/bin/python3.11 -B research/tools/verify_upload_v29.py     # every gate, NO upload (default)
    /opt/homebrew/bin/python3.11 -B research/tools/verify_upload_v29.py --no-upload  # the same, explicit
    /opt/homebrew/bin/python3.11 -B research/tools/verify_upload_v29.py --upload     # gates, then upload (owner's go only)

Independence: this file does NOT import scripts/analysis/orl_stage3.py, fwp_build, cbv_ship_build or afj_ship_build.
It re-derives the splice set R from the raw artifacts with its own code — the v13 provenance stage scope
(`lightgbm:delta:mean3`, 130,091), minus the ADS-B-stage AFJ rows (v18 != v17), whose v26 value equals
rint(0.5*max(proxy - C, 1) + 0.5*FW_int) (proxy/C from cbv_ship_input's holdout rows in order + cbv_ship_C_2026.npy,
FW_int from data/fwp_build/fw) — asserts |R| = 115,180 = FWP's touched set, then checks every gate:
V1 registration sha logged (build-time and current); V2 Stage 2 WORKING, k2 = the report's composed.k; V3 shas (base
= live v26, candidate/manifest/predictions = meta); V4 check_submission, ids/order/dtypes = v26; V5 R re-derived; V6 the
value on every R row = rint(0.5*max(proxy - C, 1) + 0.5*F_orl_int); V7 every row outside R byte-identical; V8 the
changed set = the manifest's changed rows, inside R; V9 the AFJ/AUN rules (sq share |move| > 300 s <= 47.8 %, rows
|move| > 600 s <= 0.19 % of R); V10 S and G = S*(2k - 1) over k in [k2 - 0.5, k2] recomputed = meta; V11 the 2026 move
energy by (month, airport) inside the Stage-2 fold-A [min, max] (else the k range is void -> refused); if the
registration carries V11_RULING_MARK (hash-checked by V1: the owner ruling of 2026-09-23), only cells ABOVE the max
void, and the low-side cells are reported, never silently dropped. The build's two-sided verdict must still reproduce.
Only build_submission.check_submission and the upload-key rule (SUBMISSION_RE) are shared.
"""
from __future__ import annotations

import hashlib, json, pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]          # research/tools/ -> the repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import build_submission as bs  # noqa: E402  (check_submission and the key rule only)

#: the scorer reads only `<team>_v<N>.parquet` (build_submission.SUBMISSION_RE): `..._v29_orl.parquet` was accepted by the
#: bucket on 2026-09-23 and never scored. The local candidate keeps its _orl name; only the bucket key follows the rule.
KEY, BUCKET = bs.submission_name(29), "prc-2026-merry-quicksand"
UPLOAD_FLAG = "--upload"
RESULT_MARK = "RESULT — Stage 3"
V11_RULING_MARK = "OWNER RULING — V11 upper-side only"
SCOPE_STAGE = "lightgbm:delta:mean3"
BOARD_N = 344_841.0
CONC_BAR_PCT, CAP_GT600_PCT = 47.8, 0.19
K_WIDTH = 0.5


def _p(rel):
    return ROOT / rel


DEFAULT = {
    "base": _p("submissions/merry-quicksand_v26.parquet"),
    "base_sha": "b5b47f2db90570a3b55f70927486cc1304a3b727f009c2972d7ba8b6b7238fcd",
    "cand": _p("submissions/merry-quicksand_v29_orl.parquet"),
    "meta": _p("submissions/merry-quicksand_v29_orl.meta.json"),
    "manifest": _p("submissions/merry-quicksand_v29_orl.manifest.parquet"),
    "template": _p("data/raw/submitting.parquet"),
    "prov": _p("submissions/merry-quicksand_v13.provenance.parquet"),
    "ship": _p("data/cache_stand/cbv_ship_input.parquet"),
    "c_npy": _p("data/cache_stand/cbv_ship_C_2026.npy"),
    "fw": _p("data/fwp_build/fw/merry-quicksand_v6.parquet"),
    "afj": _p("data/pipeline_runs/afj_20260916T212844Z/afj_touched_rows.parquet"),
    "fwp_touched": _p("data/pipeline_runs/fwp_20260917T065003Z/fwp_touched_rows.parquet"),
    "preds": _p("data/orl_build/orl_2026_predictions.parquet"),
    "stage2": _p("reports/orl_stage2_report.json"),
    "prereg": _p("plans/PREREG_orl_osm_route_length_2026_09_23.md"),
    "hashlog": _p("reports/orl_prereg_hashes.log"),
    "n_scope": 130_091, "n_R": 115_180, "scope_stage": SCOPE_STAGE,
}


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def config_from(cfg: dict) -> dict:
    """A verifier config from any mapping carrying the DEFAULT keys (the smoke world's, or the real one)."""
    out = dict(DEFAULT)
    out.update({k: cfg[k] for k in DEFAULT if k in cfg})
    return out


def logged(sha256: str, name: str, hashlog) -> bool:
    return any(f"(full {sha256})" in line and name in line for line in pathlib.Path(hashlog).read_text().splitlines())


def taxi_c_by_id(ship: pd.DataFrame, C) -> pd.Series:
    """max(score_proxy - C, 1) on the holdout rows, C aligned to them in input order (own code, not cbv_ship_build)."""
    h = ship[ship.holdout.to_numpy(bool)].reset_index(drop=True)
    C = np.asarray(C, dtype="float64")
    assert len(C) == len(h), f"{len(C)} C values for {len(h)} holdout rows"
    assert np.isfinite(C).all(), "non-finite C"
    assert h.MVT_ID_mvt.is_unique, "duplicate holdout id"
    return pd.Series(np.maximum(h.score_proxy.to_numpy("float64") - C, 1.0), index=h.MVT_ID_mvt.to_numpy("float64"))


def derive_R(cfg: dict, base: pd.DataFrame) -> pd.DataFrame:
    """R re-derived from raw artifacts: [MVT_ID_mvt, taxi_C, base]."""
    prov = pd.read_parquet(cfg["prov"], columns=["MVT_ID_mvt", "stage"])
    scope = prov.MVT_ID_mvt[prov.stage == cfg["scope_stage"]].to_numpy("float64")
    assert len(scope) == cfg["n_scope"], f"scope {len(scope)} != {cfg['n_scope']}"
    tc = taxi_c_by_id(pd.read_parquet(cfg["ship"], columns=["MVT_ID_mvt", "score_proxy", "holdout"]), np.load(cfg["c_npy"]))
    tcv = tc.reindex(scope).to_numpy("float64")
    assert np.isfinite(tcv).all(), "a scope id has no taxi_C"
    fw = pd.read_parquet(cfg["fw"]).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(scope).to_numpy("float64")
    assert np.isfinite(fw).all(), "a scope id has no FW_int"
    bv = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(scope).to_numpy("float64")
    assert np.isfinite(bv).all(), "a scope id is missing from the base"
    afj = pd.read_parquet(cfg["afj"])
    adsb = afj.MVT_ID_mvt[afj.v18 != afj.v17].to_numpy("float64")
    keep = (np.rint(0.5 * tcv + 0.5 * fw) == bv) & ~np.isin(scope, adsb)
    r = pd.DataFrame({"MVT_ID_mvt": scope[keep], "taxi_C": tcv[keep], "base": bv[keep]})
    assert len(r) == cfg["n_R"], f"|R| = {len(r)} != registered {cfg['n_R']}"
    fwp = set(pd.read_parquet(cfg["fwp_touched"]).MVT_ID_mvt.astype(float))
    assert set(r.MVT_ID_mvt) == fwp, "re-derived R is not FWP's touched set"
    return r


def rules(move) -> dict:
    m = np.asarray(move, "float64"); a = np.abs(m); sq = m ** 2; tot = float(sq.sum())
    share300 = 100 * float(sq[a > 300].sum() / tot) if tot else 0.0
    pct600 = 100 * float((a > 600).mean()) if len(m) else 0.0
    return {"sq_share_gt300_pct": share300, "pct_rows_gt600": pct600,
            "pass": bool(share300 <= CONC_BAR_PCT and pct600 <= CAP_GT600_PCT)}


def energy_void(move, ap, month, ref: dict, upper_only: bool = False) -> tuple:
    e = pd.DataFrame({"k": [f"{int(mo)}:{a}" for mo, a in zip(month, ap)], "e": np.asarray(move, float) ** 2}).groupby("k").e.mean()
    vals = np.asarray(list(ref.values()), float)
    lo, hi = float(vals.min()), float(vals.max())
    out = sorted(k for k, v in e.items() if (v > hi if upper_only else not (lo <= v <= hi)))
    return bool(out), out


def check_key(key: str) -> None:
    assert bs.SUBMISSION_RE.match(key), f"upload key {key!r} breaks the scorer convention <team>_v<N>.parquet"


def verify(cfg: dict, upload: bool = False) -> dict:
    """Every gate; AssertionError on the first failure. Returns a summary with status VERIFIED."""
    meta = json.loads(pathlib.Path(cfg["meta"]).read_text())
    # V1 registration
    assert logged(meta["prereg_sha256"], pathlib.Path(cfg["prereg"]).name, cfg["hashlog"]), "build-time registration sha not logged"
    cur = sha(cfg["prereg"])
    assert logged(cur, pathlib.Path(cfg["prereg"]).name, cfg["hashlog"]), "registration changed after its last hash line"
    # V2 stage 2
    rep2 = json.loads(pathlib.Path(cfg["stage2"]).read_text())
    assert rep2["verdict"]["verdict"] == "WORKING", "Stage 2 is not WORKING"
    k2 = float(rep2["composed"]["k"])
    assert np.isfinite(k2) and k2 == float(meta["k2"]), "k2 != the Stage-2 report's composed.k"
    assert meta["stage2_report_sha256"] == sha(cfg["stage2"]), "Stage-2 report changed since the build"
    # V3 shas
    assert sha(cfg["base"]) == cfg["base_sha"] == meta["base_sha256"], "base is not the live v26"
    assert sha(cfg["cand"]) == meta["candidate_sha256"], "candidate sha != meta"
    assert sha(cfg["manifest"]) == meta["manifest_sha256"], "manifest sha != meta"
    assert sha(cfg["preds"]) == meta["predictions_sha256"], "2026 predictions sha != meta"
    # V4 file shape
    tpl = pd.read_parquet(cfg["template"])
    base, cand = pd.read_parquet(cfg["base"]), pd.read_parquet(cfg["cand"])
    bs.check_submission(cand, tpl)
    assert list(cand.columns) == list(base.columns), "columns differ from v26"
    assert np.array_equal(cand.MVT_ID_mvt.to_numpy(), base.MVT_ID_mvt.to_numpy()), "id order differs from v26"
    assert cand.dtypes.equals(base.dtypes), "dtypes differ from v26"
    # V5 R
    r = derive_R(cfg, base)
    # V6 values on R
    preds = pd.read_parquet(cfg["preds"])
    assert preds.MVT_ID_mvt.is_unique, "duplicate id in the 2026 predictions"
    p = preds.set_index("MVT_ID_mvt").reindex(r.MVT_ID_mvt.to_numpy("float64"))
    orl = p.orl_taxi_int.to_numpy("float64")
    assert np.isfinite(orl).all() and (orl >= 1).all() and (np.rint(orl) == orl).all(), "bad F_orl taxi values on R"
    want = np.rint(0.5 * r.taxi_C.to_numpy() + 0.5 * orl)
    cv = cand.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(r.MVT_ID_mvt.to_numpy("float64")).to_numpy("float64")
    assert np.array_equal(cv, want), f"{int((cv != want).sum())} R rows do not carry the registered value"
    # V7 outside R
    a_b, a_c = base.TAXITIME_SEC_mvt.to_numpy(), cand.TAXITIME_SEC_mvt.to_numpy()
    in_r = cand.MVT_ID_mvt.isin(r.MVT_ID_mvt).to_numpy()
    assert np.array_equal(a_b[~in_r], a_c[~in_r]), "a row outside R changed"
    # V8 manifest
    man = pd.read_parquet(cfg["manifest"])
    changed = set(cand.MVT_ID_mvt[a_b != a_c])
    assert changed == set(man.MVT_ID_mvt[man.changed]), "changed rows != the manifest's"
    assert changed <= set(r.MVT_ID_mvt), "a changed row lies outside R"
    # V9 rules
    move = want - r.base.to_numpy()
    ru = rules(move)
    assert ru["pass"], f"pre-upload rules fail: {ru}"
    # V10 projection
    s = float((move ** 2).sum() / BOARD_N)
    assert abs(s - meta["projection"]["S"]) < 1e-9, "S != the build's"
    g = [s * (2 * (k2 - K_WIDTH) - 1), s * (2 * k2 - 1)]
    assert np.allclose(g, meta["projection"]["G_range"], rtol=0, atol=1e-9), "G range != the build's"
    # V11 tripwire
    ref = rep2["tripwires"]["move_energy_2025_pop_by_month_airport"]
    void2, outside2 = energy_void(move, p.ap.to_numpy(), p.month.to_numpy(), ref)
    assert void2 == bool(meta["tripwires"]["k_range_void"]), "the build's tripwire verdict does not reproduce"
    upper_only = V11_RULING_MARK in pathlib.Path(cfg["prereg"]).read_text()
    void, outside = energy_void(move, p.ap.to_numpy(), p.month.to_numpy(), ref, upper_only=upper_only)
    assert not void, f"2026 move energy outside the 2025 range in {outside}: k range void, no priced upload"
    low_ruled = sorted(set(outside2) - set(outside))
    out = {"status": "VERIFIED", "n_rows": int(len(cand)), "n_R": int(len(r)), "n_changed": len(changed),
           "S": s, "G_range": g, "rules": ru, "k2": k2,
           "v11_upper_only": upper_only, "v11_low_side_ruled": low_ruled}
    if upload:
        text = pathlib.Path(cfg["prereg"]).read_text()
        assert RESULT_MARK in text, f"no '{RESULT_MARK}' appended to the registration"
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    upload = UPLOAD_FLAG in argv
    out = verify(DEFAULT, upload=upload)
    print(f"VERIFIED: rows {out['n_rows']:,} | R {out['n_R']:,} | changed {out['n_changed']:,} | S {out['S']:.2f} "
          f"(MAGNITUDE, not a gain) | G in [{out['G_range'][0]:+.1f}, {out['G_range'][1]:+.1f}] at k2 {out['k2']:.3f} | "
          f"rules {out['rules']} | V11 upper-only {out['v11_upper_only']} low-side ruled {out['v11_low_side_ruled']}")
    if not upload:
        print("dry run: not uploading (owner per-file go required; pass --upload after it)")
        return 0
    check_key(KEY)
    from prc.s3 import S3, load_credentials
    c = S3(load_credentials())
    assert KEY not in [o["key"] for o in c.list_objects(BUCKET)], "v29 already in bucket"
    st = c.upload(BUCKET, KEY, pathlib.Path(DEFAULT["cand"]))
    print(f"UPLOADED {KEY} -> HTTP {st}")
    assert KEY in [o["key"] for o in c.list_objects(BUCKET)], "upload returned but the object is not listed"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
