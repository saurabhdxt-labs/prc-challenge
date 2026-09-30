"""SHIP-AFJ — v18 = v17 + the ADS-B stage on the adsb.fi fallback rows (plans/PREREG_adsbfi_july_fallback_2026_09_16.md, §4 and
Amendment AFJ.1). Route B: a guarded splice on v17. NO upload (a separate owner decision).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/afj_ship_build.py

AFJ rows = scored matched rows whose AFJ-table source is adsb.fi (July, shipped v4 coverage < 2), AFJ coverage >= 2, at an
airport models_v4's gate allows, whose v13 stage is `lightgbm:delta:mean3` (the stage never moved them) and whose v13 value
is above the 1 s floor. On them: F' = proxy - v13, r = predict_residual(models_v4) through the pipeline's own stage code
(stage_frame -> apply_stage, |resid| <= 3,600 s refused), v18 = rint(max(proxy - (F' + r), 1)). Every other row is byte-identical
to v17. Guards G1-G6 and the registered move-size split are written to the manifest; a failed guard writes nothing.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prc.pipeline import adsb_stage as AS  # noqa: E402
from prc.pipeline import legacy  # noqa: E402

PREREG = ROOT / "plans" / "PREREG_adsbfi_july_fallback_2026_09_16.md"
V17 = ROOT / "submissions" / "merry-quicksand_v17.parquet"
V17_SHA = "79af6c855cbd8174b125a5cc1b1b5fe5f648077fbf57d4041e6327e180e0be98"
V13_DIR = ROOT / "data" / "pipeline_runs" / "v13_20260912T113635Z"
V13, V13_PROV = V13_DIR / "merry-quicksand_v13.parquet", V13_DIR / "merry-quicksand_v13.provenance.parquet"
V13_SHA = "2ac39110624144257be9a1f20c380803f7f3cf7f3ba4b640f1caf31be65309e7"
ADSB = ROOT / "data" / "adsb" / "v2"
TABLE, ROW_SOURCE, MODELS = ADSB / "features_2026janjul_afj.parquet", ADSB / "features_2026janjul_afj.row_source.parquet", ADSB / "models_v4"
TABLE_REPORT = ROOT / "reports" / "afj_table_report.json"
RANKING = ROOT / "data" / "raw" / "ranking.parquet"
RUNS = ROOT / "data" / "pipeline_runs"
V18 = ROOT / "submissions" / "merry-quicksand_v18.parquet"
NO_STAGE = "lightgbm:delta:mean3"
BOARD_N = 344_841.0
CBV_GAIN_PER_ROW = 249.2 / 127_403            # AFJ.1: v14's board gain per blended row, forgone on every AFJ row
CONC_BAR_PCT, CAP_GT600_PCT = 47.8, 0.19      # §6: squared-move share > 300 s; rows moved > 600 s per moved row


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _allowed(gate: dict, ap: str) -> bool:
    """A gate entry is a bool or {allowed: bool, ...}; anything else is refused."""
    v = gate.get(ap, False)
    allowed = v.get("allowed") if isinstance(v, dict) else v
    if not isinstance(allowed, bool):
        raise TypeError(f"gate entry for {ap} must be a bool or carry a bool `allowed`, got {v!r}")
    return allowed


def afj_rows(row_source: pd.DataFrame, gate: dict, v13_stage: pd.Series, v13_value: pd.Series) -> tuple:
    """(ids of the AFJ rows, counts). row_source: MVT_ID_mvt, source, ap, coverage_v4, coverage_afj (the table builder's map).
    A row qualifies iff source == adsb.fi, coverage_afj >= 2, coverage_v4 < 2, gated airport, v13 stage == NO_STAGE, v13 > 1.
    A row at a gated airport with source adsb.fi and coverage >= 2 whose v13 stage is NOT `lightgbm:delta:mean3` is a
    contradiction (its v4 coverage was >= 2) and is refused."""
    s = row_source
    allowed = s.ap.map(lambda a: _allowed(gate, a)).to_numpy(bool)
    cand = (s.source.to_numpy() == "adsb.fi") & (s.coverage_afj.to_numpy() >= AS.MIN_COVERAGE) & (s.coverage_v4.to_numpy() < AS.MIN_COVERAGE)
    ids = s.MVT_ID_mvt.to_numpy(dtype="float64")
    stage = v13_stage.reindex(ids).to_numpy(dtype=object)
    val = v13_value.reindex(ids).to_numpy(dtype="float64")
    if np.isnan(val[cand & allowed]).any() or pd.isna(stage[cand & allowed]).any():
        raise ValueError("an AFJ candidate row is missing from v13 or its provenance")
    bad = cand & allowed & (stage != NO_STAGE)
    if bad.any():
        raise ValueError(f"{int(bad.sum()):,} AFJ candidate rows were already staged in v13 (stage != {NO_STAGE!r})")
    floor = cand & allowed & (val <= 1.0)
    take = cand & allowed & ~floor
    counts = {"n_candidates": int(cand.sum()), "n_candidates_gated": int((cand & allowed).sum()),
              "n_at_v13_floor_left_at_v17": int(floor.sum()), "n_afj_rows": int(take.sum())}
    return ids[take], counts


def base_from_v13(proxy, v13) -> np.ndarray:
    """F' = proxy - v13 on rows above the floor (AFJ.1's registered approximation, +-0.5 s)."""
    proxy, v13 = np.asarray(proxy, "float64"), np.asarray(v13, "float64")
    if (v13 <= 1.0).any():
        raise ValueError("base_from_v13: a row at the 1 s floor cannot be reconstructed")
    return proxy - v13


def splice(v17: pd.DataFrame, ids, values) -> tuple:
    """(candidate, touched mask): v17 with `values` on `ids`, dtype / order / every other row untouched."""
    ids = np.asarray(ids, dtype="float64")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("splice: duplicate ids")
    pos = pd.Index(v17.MVT_ID_mvt.to_numpy("float64")).get_indexer(ids)
    if (pos < 0).any():
        raise ValueError(f"splice: {int((pos < 0).sum())} ids not in v17")
    new = v17.TAXITIME_SEC_mvt.to_numpy().copy()
    vals = np.asarray(values, dtype="float64")
    if not (np.isfinite(vals).all() and (vals >= 1).all() and (np.rint(vals) == vals).all()):
        raise ValueError("splice: a value is non-finite, below 1, or not an integer")
    new[pos] = vals.astype(new.dtype)
    cand = v17.copy()
    cand["TAXITIME_SEC_mvt"] = new
    touched = new != v17.TAXITIME_SEC_mvt.to_numpy()
    outside = touched.copy()
    outside[pos] = False
    if outside.any():
        raise AssertionError("G3: a row outside the AFJ set changed")
    return cand, touched


def move_split(move, stage_shift) -> dict:
    """§6 on the TOTAL move (v18 - v17) and the stage-only shift (-r): squared-move shares by size, rows > 600 s, the two
    registered rules, and the stake if every move were pure noise."""
    out = {}
    for name, m in (("total", np.asarray(move, "float64")), ("stage_only", np.asarray(stage_shift, "float64"))):
        a = np.abs(m)
        sq = m ** 2
        tot = float(sq.sum()) if len(m) else 0.0
        out[name] = {"n": int(len(m)), "mean_abs_s": float(a.mean()) if len(m) else 0.0, "p50_abs_s": float(np.median(a)) if len(m) else 0.0,
                     "p90_abs_s": float(np.quantile(a, 0.9)) if len(m) else 0.0, "max_abs_s": float(a.max()) if len(m) else 0.0,
                     "sq_share_gt120_pct": 100 * float(sq[a > 120].sum() / tot) if tot else 0.0,
                     "sq_share_gt300_pct": 100 * float(sq[a > 300].sum() / tot) if tot else 0.0,
                     "sq_share_gt600_pct": 100 * float(sq[a > 600].sum() / tot) if tot else 0.0,
                     "n_gt300": int((a > 300).sum()), "n_gt600": int((a > 600).sum()),
                     "pct_rows_gt600": 100 * float((a > 600).mean()) if len(m) else 0.0,
                     "stake_if_pure_noise_board_mse": -tot / BOARD_N}
    t = out["total"]
    out["rules"] = {"concentration_gt300_le_47_8pct": bool(t["sq_share_gt300_pct"] <= CONC_BAR_PCT),
                    "cap_rows_gt600_le_0_19pct": bool(t["pct_rows_gt600"] <= CAP_GT600_PCT),
                    "recommendation": "do not upload" if not (t["sq_share_gt300_pct"] <= CONC_BAR_PCT and t["pct_rows_gt600"] <= CAP_GT600_PCT) else "owner decision"}
    return out


def main() -> int:
    for p, want, name in ((V17, V17_SHA, "v17"), (V13, V13_SHA, "v13")):
        if sha256(p) != want:
            raise AssertionError(f"G1: {name} sha differs")
    rep = json.loads(TABLE_REPORT.read_text())
    if sha256(TABLE) != rep["table_sha256"]:
        raise AssertionError("G1: the AFJ table's sha differs from its report")
    if V18.exists():
        raise FileExistsError(V18)
    v17 = pd.read_parquet(V17)
    v13 = pd.read_parquet(V13).set_index("MVT_ID_mvt")
    prov = pd.read_parquet(V13_PROV, columns=["MVT_ID_mvt", "stage"]).set_index("MVT_ID_mvt")
    src = pd.read_parquet(ROW_SOURCE)
    gate, man = AS.read_gate(MODELS)
    gate_raw = json.loads((MODELS / "MANIFEST.json").read_text())["gate"]
    ids, counts = afj_rows(src, gate_raw, prov.stage, v13.TAXITIME_SEC_mvt.astype("float64"))
    print(f"AFJ rows {counts}", flush=True)
    rk = pd.read_parquet(RANKING, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "PHASE_mvt"])
    rk = rk[rk.PHASE_mvt == "DEP"].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").reindex(ids)
    if rk.ADEP_mvt.isna().any() or rk.AOBT_3_flt.isna().any():
        raise AssertionError("G2: an AFJ row without a matched DEP ranking row")
    mvt, aobt = pd.to_datetime(rk.MVT_TIME_UTC_mvt, utc=True), pd.to_datetime(rk.AOBT_3_flt, utc=True)
    proxy = (mvt - aobt).dt.total_seconds().to_numpy("float64")
    v13v = v13.TAXITIME_SEC_mvt.reindex(ids).to_numpy("float64")
    F = base_from_v13(proxy, v13v)
    table, tmeta = AS.read_table(TABLE, src.MVT_ID_mvt.to_numpy("float64"))
    frame = AS.stage_frame(ids, rk.ADEP_mvt.to_numpy(), mvt.dt.hour.to_numpy("float64"), proxy, F, table)
    real = legacy.adsb_predictor()
    seen = {}

    def fn(f):
        seen["resid"] = real(f, MODELS, oof=False)
        return seen["resid"]
    value, info = AS.apply_stage(frame, fn, gate)                       # G4: the stage's own refusals (zeros, finite, <= 3,600 s)
    if info["n_eligible"] != len(ids):
        raise AssertionError(f"G4: {info['n_eligible']} eligible of {len(ids)} AFJ rows (gate / coverage disagree with the row map)")
    if not np.array_equal(np.rint(np.maximum(proxy - F, 1.0)), v13v):
        raise AssertionError("G2: the reconstructed base does not reproduce v13")
    cand, touched = splice(v17, ids, value)
    if not (np.array_equal(cand.MVT_ID_mvt.to_numpy(), v17.MVT_ID_mvt.to_numpy()) and cand.dtypes.equals(v17.dtypes)):
        raise AssertionError("G3: ids / order / dtype")
    v17v = v17.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids).to_numpy("float64")
    move = value - v17v
    split = move_split(move, -seen["resid"])
    n_moved = int(touched.sum())
    P = float(rep["projection_P_board_mse"])
    P_net = P - n_moved * CBV_GAIN_PER_ROW
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RUNS / f"afj_{stamp}"
    out.mkdir(parents=False, exist_ok=False)
    sub, det_p = out / "merry-quicksand_afj.parquet", out / "afj_touched_rows.parquet"
    cand.to_parquet(sub, index=False)
    det = pd.DataFrame({"MVT_ID_mvt": ids, "ap": rk.ADEP_mvt.to_numpy(), "coverage_afj": frame.coverage.to_numpy(), "proxy": proxy,
                        "v13": v13v, "v17": v17v, "resid": seen["resid"], "v18": value, "move": move})
    det.to_parquet(det_p, index=False)
    back = pd.read_parquet(sub)
    if not back.equals(cand):
        raise AssertionError("G5: re-read differs")
    shutil.copyfile(sub, V18)
    if sha256(V18) != sha256(sub):
        raise AssertionError("G5: copy differs")
    by_ap = det.assign(moved=(det.v18 != det.v17)).groupby("ap").agg(n=("moved", "size"), moved=("moved", "sum"),
                                                                     mean_move_s=("move", "mean"), mean_resid_s=("resid", "mean"))
    if counts["n_candidates_gated"] != rep["n_new_movable_gated"]:       # G6: the AFJ set is the table report's newly movable set
        raise AssertionError(f"G6: {counts['n_candidates_gated']:,} gated candidates, table report {rep['n_new_movable_gated']:,}")
    for a, r in by_ap.iterrows():
        if int(r.n) > rep["per_airport"][a]["n_new_movable"] or not rep["per_airport"][a]["gated"]:
            raise AssertionError(f"G6: {a}: {int(r.n)} AFJ rows vs the table report's {rep['per_airport'][a]['n_new_movable']} (gated {rep['per_airport'][a]['gated']})")
    manifest = dict(tag="SHIP-AFJ", version="v18", route="B (guarded splice on v17; AFJ.1)", prereg=str(PREREG.relative_to(ROOT)),
                    prereg_sha256=sha256(PREREG), created_utc=stamp, guards=dict(G1=True, G2=True, G3=True, G4=True, G5=True, G6=True),
                    counts=counts, touched=n_moved, stage_info={k: info[k] for k in ("n_rows", "n_eligible", "n_changed", "changed_by_airport",
                                                                                      "mean_abs_resid_eligible_s", "max_abs_resid_eligible_s")},
                    by_airport={a: {k: (int(v) if k in ("n", "moved") else float(v)) for k, v in r.items()} for a, r in by_ap.iterrows()},
                    move_split=split, projection=dict(P_board_mse=P, forgone_cbv_blend_board_mse=n_moved * CBV_GAIN_PER_ROW, P_net=P_net,
                                                      working_bar=0.4 * P_net, rule="WORKING iff dv >= 0.4 P_net; PARTIAL 0 < dv < 0.4 P_net; NOT WORKING dv <= 0"),
                    guards_in_code={"SPAN_MIN_S": 0.001, "Z_CLIP": 300.0, "MAX_ABS_RESID_S": AS.MAX_ABS_RESID_S},
                    table_meta=tmeta, models_manifest={k: man[k] for k in man if k != "gate"},
                    outputs={str(p.relative_to(ROOT)): sha256(p) for p in (sub, det_p, V18)},
                    inputs={str(p.relative_to(ROOT)): sha256(p) for p in (V17, V13, V13_PROV, TABLE, ROW_SOURCE, RANKING, MODELS / "MANIFEST.json")},
                    upload="NOT AUTHORISED - separate owner decision")
    (out / "afj_manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    shutil.copyfile(out / "afj_manifest.json", V18.with_suffix(".manifest.json"))
    print(json.dumps({k: manifest[k] for k in ("counts", "touched", "move_split", "projection", "outputs")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
