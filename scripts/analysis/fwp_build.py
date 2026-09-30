"""FWP — production build of arm FW spliced onto the CatBoost-blend (CBV) rows of v19
(plans/PREREG_fw_production_build_2026_09_16.md). NO model fit here and NO upload: this module only reads what
`lgbm_submit.py` already wrote and splices it into v19 under the registered formula.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/fwp_build.py repro-check --out data/fwp_build/repro
    /opt/homebrew/bin/python3.11 -B scripts/analysis/fwp_build.py splice --fw data/fwp_build/fw

`repro-check` re-derives G2: on the 130,091 rows whose v13 stage is exactly `lightgbm:delta:mean3` (the SAME scope
cbv_ship_build uses), the matched value `lgbm_submit.py --version 6 ... --out-dir <dir>` wrote must equal the v13
value exactly. It writes `<dir>/fwp_g2_verdict.json` and exits 0 iff identical, 1 otherwise.

`splice` implements registration steps 3-4: R = scope rows (same 130,091-row stage scope) whose CURRENT v19 value
equals CBV's own blend formula rint(0.5*max(proxy-C,1) + 0.5*v13) (C = data/cache_stand/cbv_ship_C_2026.npy, aligned
exactly as cbv_ship_build.holdout_predictions/scope_frame do); scope rows whose v19 value differs (a later lever,
e.g. AFJ/AUN, touched them) are excluded from R and counted. On R the new value is
rint(0.5*max(proxy-C,1) + 0.5*FW) with FW read from the FW-fit's matched output (same file/column contract as G2's
input); every row outside R stays byte-identical to v19. It writes data/pipeline_runs/fwp_<UTC>/ only (candidate
parquet + manifest + touched-row detail) and NEVER a file under submissions/.

Two design choices not spelled out literally in the CLI surface of the registration, both documented here rather
than silently assumed:
  * `splice` also requires a PASSING G2 verdict on record before it will write a candidate ("a failed guard writes
    no candidate" bundles G1-G6, including G2, in the registration's Guards section). Since the registration fixes
    the repro-check output directory itself (`data/fwp_build/repro`, "Build (fixed now)" step 1), that path is a
    module constant (REPRO_DIR) rather than a new CLI flag on `splice`, matching how every sibling *_ship_build.py
    script hard-codes its fixed upstream paths (V13_DIR, V17, ...) instead of taking them as arguments.
  * G1's "C npy" input sha is RECORDED in the manifest but not asserted against a stored expected value: no prior
    artifact in this repo records an expected sha256 for data/cache_stand/cbv_ship_C_2026.npy to check against
    (unlike v13, which every *_ship_build.py script already pins to a frozen constant, and v19, whose own manifest
    records its own output sha). Asserting equality against a fabricated number would be improvising past what the
    registration specifies; this is flagged again in the build report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import cbv_ship_build as CBV      # noqa: E402  (holdout_predictions / scope_frame / blend_values / taxi — public, reused)
import afj_ship_build as AFJ      # noqa: E402  (splice / move_split / BOARD_N / CONC_BAR_PCT / CAP_GT600_PCT — public, reused)
import build_submission as BS     # noqa: E402  (submission_name — public, reused)

PREREG = ROOT / "plans" / "PREREG_fw_production_build_2026_09_16.md"
V13_DIR = ROOT / "data" / "pipeline_runs" / "v13_20260912T113635Z"
V13_SUB = V13_DIR / "merry-quicksand_v13.parquet"
V13_PROV = V13_DIR / "merry-quicksand_v13.provenance.parquet"
V13_SHA = CBV.V13_SHA
C_PATH = ROOT / "data" / "cache_stand" / "cbv_ship_C_2026.npy"
SHIP_INPUT = ROOT / "data" / "cache_stand" / "cbv_ship_input.parquet"
V19 = ROOT / "submissions" / "merry-quicksand_v19.parquet"
V19_MANIFEST = ROOT / "submissions" / "merry-quicksand_v19.manifest.json"
COMP_JSON = ROOT / "reports" / "fw_composition_2026_09_16.json"
REPRO_DIR = ROOT / "data" / "fwp_build" / "repro"
RUNS = ROOT / "data" / "pipeline_runs"

FW_VERSION = 6
FW_SUBMISSION_NAME = BS.submission_name(FW_VERSION)     # merry-quicksand_v6.parquet
G2_VERDICT_NAME = "fwp_g2_verdict.json"

BOARD_N = AFJ.BOARD_N                    # 344,841.0
CONC_BAR_PCT, CAP_GT600_PCT = AFJ.CONC_BAR_PCT, AFJ.CAP_GT600_PCT      # 47.8, 0.19 (registered, AFJ/AUN)
STOP_P = 50.0
G4_MAX_ABS_DELTA_S = 3_600.0


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _relkey(p) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def fw_output_path(out_dir) -> Path:
    """Where `lgbm_submit.py --version 6 ... --out-dir <out_dir>` writes: <out_dir>/merry-quicksand_v6.parquet,
    columns [MVT_ID_mvt (float), TAXITIME_SEC_mvt (int32)] on every scored row, matched rows (339,551) predicted."""
    return Path(out_dir) / FW_SUBMISSION_NAME


# ---------------------------------------------------------------- G2 (repro-check)

def repro_check(repro_dir, v13_sub_path=V13_SUB, v13_prov_path=V13_PROV) -> dict:
    """G2: on the 130,091 rows whose v13 stage is exactly cbv_ship_build.SCOPE_STAGE, the value
    `<repro_dir>/merry-quicksand_v6.parquet` carries must equal the v13 value exactly. Returns a JSON-serialisable
    verdict; raises if the scope itself does not match the registered count or a scope id is missing anywhere."""
    v13 = pd.read_parquet(v13_sub_path)
    prov = pd.read_parquet(v13_prov_path, columns=["MVT_ID_mvt", "stage"])
    counts = prov.stage.value_counts().to_dict()
    want = CBV.STAGE_COUNTS[CBV.SCOPE_STAGE]
    have = counts.get(CBV.SCOPE_STAGE)
    if have != want:
        raise AssertionError(f"G2: stage {CBV.SCOPE_STAGE!r} has {have} rows in provenance, registered {want}")
    ids = prov.MVT_ID_mvt[prov.stage == CBV.SCOPE_STAGE].astype("float64").to_numpy()
    v13v = v13.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids)
    if v13v.isna().any():
        raise AssertionError(f"G2: {int(v13v.isna().sum())} scope ids are missing from v13")
    repro_path = fw_output_path(repro_dir)
    if not repro_path.exists():
        raise FileNotFoundError(f"G2: {repro_path} does not exist — run the repro job first")
    repro = pd.read_parquet(repro_path)
    if list(repro.columns) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]:
        raise AssertionError(f"G2: {repro_path} columns {list(repro.columns)} != ['MVT_ID_mvt', 'TAXITIME_SEC_mvt']")
    rv = repro.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids)
    if rv.isna().any():
        raise AssertionError(f"G2: {int(rv.isna().sum())} scope ids are missing from {repro_path}")
    diff = rv.to_numpy("float64") - v13v.to_numpy("float64")
    order = np.argsort(-np.abs(diff))
    top = [{"MVT_ID_mvt": float(ids[i]), "diff": float(diff[i])} for i in order[:20] if diff[i] != 0]
    return {"guard": "G2", "repro_path": _relkey(repro_path), "n_scope": int(len(ids)), "n_diff": int((diff != 0).sum()),
            "max_abs_diff": float(np.abs(diff).max()) if len(diff) else 0.0,
            "identical": bool((diff == 0).all()), "first20_diff_ids": top}


def cmd_repro_check(out_dir, v13_sub_path=V13_SUB, v13_prov_path=V13_PROV) -> int:
    v = repro_check(out_dir, v13_sub_path=v13_sub_path, v13_prov_path=v13_prov_path)
    (Path(out_dir) / G2_VERDICT_NAME).write_text(json.dumps(v, indent=1))
    print(json.dumps(v, indent=1))
    return 0 if v["identical"] else 1


def _load_g2_verdict(repro_dir) -> dict:
    p = Path(repro_dir) / G2_VERDICT_NAME
    if not p.exists():
        raise AssertionError(f"G2: no verdict at {p} — run `repro-check --out {repro_dir}` first")
    v = json.loads(p.read_text())
    if not v.get("identical"):
        raise AssertionError(f"G2: {p} recorded a NOT-identical repro ({v.get('n_diff')} differing rows) — refusing to splice")
    return v


# ---------------------------------------------------------------- G1

def _manifest_output_sha(manifest: dict, filename: str):
    for k, v in manifest.get("outputs", {}).items():
        if Path(k).name == filename:
            return v
    return None


def g1_checks(v13_sub_path, v19_path, v19_manifest_path, c_path) -> dict:
    """v13 sha pinned to the frozen V13_SHA every sibling ship build uses; v19 sha checked against its OWN
    manifest's recorded output sha (catches a v19 mutated or replaced since AUN wrote it); C's sha is recorded only
    (see the module docstring — no prior artifact in this repo records an expected sha for it)."""
    v13_sha = sha256(v13_sub_path)
    if v13_sha != V13_SHA:
        raise AssertionError(f"G1: v13 submission sha {v13_sha} differs from the frozen V13_SHA {V13_SHA}")
    man = json.loads(Path(v19_manifest_path).read_text())
    want = _manifest_output_sha(man, Path(v19_path).name)
    if want is None:
        raise AssertionError(f"G1: {v19_manifest_path} has no recorded output sha for {Path(v19_path).name}")
    v19_sha = sha256(v19_path)
    if v19_sha != want:
        raise AssertionError(f"G1: v19 sha {v19_sha} differs from its manifest's recorded {want}")
    return {"v13_sha256": v13_sha, "v19_sha256": v19_sha, "v19_manifest_recorded_sha256": want,
            "c_sha256_recorded_only": sha256(c_path)}


# ---------------------------------------------------------------- scope / R / value formula

def build_scope(ship_input_path, c_path, v13_sub_path, v13_prov_path) -> pd.DataFrame:
    """The 130,091-row v13-stage scope with proxy/C/taxi_C/v13/airport, built with cbv_ship_build's own public
    functions (same S1/S3/S4 checks it runs)."""
    ship = pd.read_parquet(ship_input_path, columns=["MVT_ID_mvt", "score_proxy", "holdout", "delta"])
    if np.isfinite(ship.delta.to_numpy()[ship.holdout.to_numpy(bool)]).any():
        raise AssertionError("a 2026 holdout row carries a label")
    hold = CBV.holdout_predictions(ship, np.load(c_path))
    v13 = pd.read_parquet(v13_sub_path)
    prov = pd.read_parquet(v13_prov_path, columns=["MVT_ID_mvt", "airport", "stage"])
    s = CBV.scope_frame(prov, v13, hold)
    s = s.merge(prov[["MVT_ID_mvt", "airport"]].astype({"MVT_ID_mvt": "float64"}), on="MVT_ID_mvt", how="left")
    return s


def compute_R(scope: pd.DataFrame, v19: pd.DataFrame) -> tuple:
    """R = scope rows whose v19 value equals CBV's own blend formula on v13 (rint(0.5*taxi_C + 0.5*v13)). A scope
    row whose v19 value differs was touched by a later lever (AFJ/AUN) and is excluded, not raised on. G6: the
    recomputed CBV value equals v19 on every R row, by construction — asserted explicitly so a broken equality
    test cannot silently pass rows through."""
    cbv_candidate = CBV.blend_values(scope.taxi_C.to_numpy(), scope.v13.to_numpy())
    v19v = v19.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(scope.MVT_ID_mvt.to_numpy())
    if v19v.isna().any():
        raise AssertionError(f"G6: {int(v19v.isna().sum())} scope ids are missing from v19")
    v19v = v19v.to_numpy("float64")
    in_r = cbv_candidate == v19v
    r = scope[in_r].reset_index(drop=True).copy()
    r["v19"] = v19v[in_r]
    r["cbv_candidate"] = cbv_candidate[in_r]
    if not np.array_equal(r.cbv_candidate.to_numpy(), r.v19.to_numpy()):
        raise AssertionError("G6: the recomputed CBV value does not equal v19 on an R row")
    counts = {"n_scope": int(len(scope)), "n_R": int(len(r)), "n_excluded_later_lever": int((~in_r).sum())}
    return r, counts


def fw_values_for(ids, fw_dir, v13_values) -> np.ndarray:
    """G4: the FW-fit's matched value on `ids`, finite, >= 1, and within 3,600 s of v13."""
    path = fw_output_path(fw_dir)
    if not path.exists():
        raise FileNotFoundError(f"G4: {path} does not exist — has the FW fit run?")
    fw = pd.read_parquet(path)
    if list(fw.columns) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]:
        raise AssertionError(f"G4: {path} columns {list(fw.columns)} != ['MVT_ID_mvt', 'TAXITIME_SEC_mvt']")
    ids = np.asarray(ids, dtype="float64")
    present = np.isin(ids, fw.MVT_ID_mvt.to_numpy(dtype="float64"))
    if not present.all():                                    # distinct from "value is NaN": a genuinely absent id
        raise AssertionError(f"G4: {int((~present).sum())} R ids are missing from {path}")
    v = fw.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids).to_numpy("float64")
    if not np.isfinite(v).all():
        raise AssertionError("G4: a non-finite FW value")
    if (v < 1.0).any():
        raise AssertionError(f"G4: {int((v < 1.0).sum())} FW value(s) below 1")
    d = np.abs(v - np.asarray(v13_values, dtype="float64"))
    if (d > G4_MAX_ABS_DELTA_S).any():
        raise AssertionError(f"G4: {int((d > G4_MAX_ABS_DELTA_S).sum())} FW value(s) differ from v13 by more than {G4_MAX_ABS_DELTA_S:.0f} s")
    return v


def fw_candidate(proxy, c, fw) -> np.ndarray:
    """Build step 4's value on R: rint(0.5*max(proxy-C,1) + 0.5*FW) — CBV's own taxi/blend formula with v13 swapped
    for FW. Thin composition of cbv_ship_build's tested taxi()/blend_values(); no logic duplicated."""
    return CBV.blend_values(CBV.taxi(proxy, c), fw)


def assert_outside_R_unchanged(v19: pd.DataFrame, cand: pd.DataFrame, r_ids) -> None:
    """G3 (a check independent of afj_ship_build.splice's own, so a regression in either is caught): every row whose
    id is not in R is byte-identical between v19 and the candidate."""
    mask_r = v19.MVT_ID_mvt.isin(np.asarray(r_ids, dtype="float64")).to_numpy()
    outside = ~mask_r
    if not np.array_equal(cand.TAXITIME_SEC_mvt.to_numpy()[outside], v19.TAXITIME_SEC_mvt.to_numpy()[outside]):
        raise AssertionError("G3: a row outside R changed")


# ---------------------------------------------------------------- P, pre-upload rules

def projection_P(comp: dict, n_r: int, board_n: float = BOARD_N) -> dict:
    """P per the registration: SSE gain = (rmse_inc^2 - rmse_cand^2) * n_rows (fold-A composed, main_mean3); per pop
    row = that / n_pop; P = per_pop_row * |R| / board_n."""
    m = comp["main_mean3"]
    n_rows, n_pop = comp["n_rows"], comp["n_pop"]
    sse_gain = (m["rmse_inc"] ** 2 - m["rmse_cand"] ** 2) * n_rows
    per_pop_row = sse_gain / n_pop
    p = per_pop_row * n_r / board_n
    return {"sse_gain_s2": float(sse_gain), "per_pop_row_s2": float(per_pop_row), "n_rows": int(n_rows),
            "n_pop": int(n_pop), "n_R": int(n_r), "board_n": float(board_n), "P": float(p)}


def move_report(move) -> dict:
    """The two registered pre-upload rules and the noise stake on R's move (candidate - prior v19 value), via
    afj_ship_build.move_split (same CONC_BAR_PCT/CAP_GT600_PCT). FWP replaces v13 wholesale rather than residual-
    adjusting it, so there is no separate stage-only decomposition; `stage_only` duplicates `total` by construction
    (move passed twice)."""
    m = np.asarray(move, dtype="float64")
    return AFJ.move_split(m, m)


def per_airport_mean_move(r: pd.DataFrame, move) -> dict:
    d = pd.DataFrame({"airport": r.airport.to_numpy(), "move": np.asarray(move, dtype="float64")})
    g = d.groupby("airport").move.agg(["size", "mean"])
    return {str(a): {"n": int(row["size"]), "mean_move_s": float(row["mean"])} for a, row in g.iterrows()}


def stop_reason(p: float) -> str | None:
    """None iff p >= STOP_P (the registered bar is a strict "<": P == STOP_P does NOT stop)."""
    return None if p >= STOP_P else f"STOP: P {p:.1f} < {STOP_P}; no candidate"


# ---------------------------------------------------------------- orchestration

def build_splice(fw_dir, *, ship_input_path=SHIP_INPUT, c_path=C_PATH, v13_sub_path=V13_SUB, v13_prov_path=V13_PROV,
                  v19_path=V19, v19_manifest_path=V19_MANIFEST, comp_json_path=COMP_JSON, repro_dir=REPRO_DIR,
                  runs_dir=RUNS, board_n: float = BOARD_N) -> dict:
    """Registration steps 3-4 end to end. Returns a JSON-serialisable dict; on P < STOP_P returns
    {"stopped": True, ...} and writes nothing (no directory, no file — a failed guard/stop writes no candidate).
    `board_n` overrides BOARD_N in the P projection only (test hook); every production call uses the registered
    344,841 default."""
    g1 = g1_checks(v13_sub_path, v19_path, v19_manifest_path, c_path)
    g2 = _load_g2_verdict(repro_dir)
    scope = build_scope(ship_input_path, c_path, v13_sub_path, v13_prov_path)
    v19 = pd.read_parquet(v19_path)
    r, r_counts = compute_R(scope, v19)
    fw = fw_values_for(r.MVT_ID_mvt.to_numpy(), fw_dir, r.v13.to_numpy())
    candidate = fw_candidate(r.proxy.to_numpy(), r.C.to_numpy(), fw)
    comp = json.loads(Path(comp_json_path).read_text())
    proj = projection_P(comp, len(r), board_n=board_n)
    reason = stop_reason(proj["P"])
    if reason is not None:
        return {"stopped": True, "reason": reason, "projection": proj, "scope": r_counts, "g1": g1, "g2": g2}
    move = candidate - r.v19.to_numpy()
    split = move_report(move)
    by_ap = per_airport_mean_move(r, move)
    cand, touched = AFJ.splice(v19, r.MVT_ID_mvt.to_numpy(), candidate)
    if not (np.array_equal(cand.MVT_ID_mvt.to_numpy(), v19.MVT_ID_mvt.to_numpy()) and cand.dtypes.equals(v19.dtypes)):
        raise AssertionError("G3: ids / order / dtype")
    assert_outside_R_unchanged(v19, cand, r.MVT_ID_mvt.to_numpy())
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(runs_dir) / f"fwp_{stamp}"
    out_dir.mkdir(parents=False, exist_ok=False)
    sub_path, det_path = out_dir / "merry-quicksand_fwp.parquet", out_dir / "fwp_touched_rows.parquet"
    cand.to_parquet(sub_path, index=False)
    detail = pd.DataFrame({"MVT_ID_mvt": r.MVT_ID_mvt.to_numpy(), "airport": r.airport.to_numpy(), "proxy": r.proxy.to_numpy(),
                           "C": r.C.to_numpy(), "v13": r.v13.to_numpy(), "FW": fw, "v19": r.v19.to_numpy(),
                           "candidate": candidate, "move": move})
    detail.to_parquet(det_path, index=False)
    back = pd.read_parquet(sub_path)
    if not back.equals(cand):
        raise AssertionError("G5: re-read differs")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    manifest = {"tag": "FWP", "prereg": _relkey(PREREG), "prereg_sha256": sha256(PREREG), "head": head,
                "created_utc": stamp, "guards": {"G1": True, "G2": True, "G3": True, "G4": True, "G5": True, "G6": True},
                "g1": g1, "g2": g2, "scope": r_counts, "projection": proj, "stop_P": STOP_P,
                "move_split": split, "by_airport": by_ap, "touched": int(touched.sum()),
                "outputs": {_relkey(sub_path): sha256(sub_path), _relkey(det_path): sha256(det_path)},
                "inputs": {_relkey(p): sha256(p) for p in (v13_sub_path, v13_prov_path, c_path, ship_input_path,
                                                           v19_path, comp_json_path)},
                "upload": "NOT AUTHORISED - separate owner decision (a v-number must be claimed in the upload lane's log first)"}
    (out_dir / "fwp_manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    return {"stopped": False, "out_dir": _relkey(out_dir), "projection": proj, "scope": r_counts, "move_split": split,
            "by_airport": by_ap, "g1": g1, "g2": g2, "touched": int(touched.sum())}


def cmd_splice(fw_dir) -> int:
    r = build_splice(fw_dir)
    print(json.dumps(r, indent=1, default=float))
    return 2 if r.get("stopped") else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("repro-check")
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("splice")
    p2.add_argument("--fw", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "repro-check":
        return cmd_repro_check(args.out)
    return cmd_splice(args.fw)


if __name__ == "__main__":
    raise SystemExit(main())
