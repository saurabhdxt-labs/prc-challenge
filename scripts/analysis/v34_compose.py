"""v34 step 7 — compose v34 from v33 in the v33 chain's lanes (plans/PREREG_nsc_no_seconds_2026_09_24.md §7 step 7,
AMENDMENT V34.7). NO upload; the owner decides per file (tools/verify_upload_v34.py, dry run by default).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/v34_launch.py --step 7 --detach            # the build (needs steps 1-6 PASS)
    /opt/homebrew/bin/python3.11 -B scripts/analysis/v34_launch.py --step 7 --detach --smoke    # the REPLAY on real data only

The v33 lanes (established from the shipped files 2026-09-24, re-asserted here on every run):
  M  339,551 matched rows (v13 provenance row_class == matched); everything else (5,290 unmatched rows, which hold the whole
     Rome chain: rome_fill / date-slip / RLD / E1 / ARC / ARN / RSE / E2W) is copied from v33 BYTE-IDENTICAL (asserted).
  R  115,180 CBV rows = FWP's touched set (data/pipeline_runs/fwp_20260917T065003Z); U = v30's ORU rows (5,364, U within R).
     R\\U: rint(0.5*max(proxy - C, 1) + 0.5*F_orl_int)   (v29 ORL)      U: rint(0.5*max(proxy - C, 1) + 0.5*FW_int)   (v30 ORU)
  N  224,371 = M \\ R, the ADS-B-stage rows: 209,460 v13 stage rows (v4 table), 14,911 AFJ rows (afj table) and, over both,
     the 4,617 AUN rows (union table): rint(max(proxy - (F + resid), 1)), resid = the stack's full-bag residual through
     prc.pipeline.adsb_stage (stage_frame -> apply_stage: zeros, finiteness, |resid| <= 3,600 s re-asserted), proxy / ap / hr
     from the stand-cache ranking rows (the pipeline's matched-lane inputs).
  v33 == v19 on N, and no version after v19 touched a matched non-R row (asserted from the files).

REPLAY (the gate before anything is written): the same formulas on the OLD inputs (models_v4; F' = proxy - v10 from
v13_detail, else proxy - v13: the approximation AFJ.1 / AUN registered, because v13's float F was never stored) must
reproduce v33 EXACTLY on R (C = cbv_ship_C_2026.npy, F_orl_int = orl_2026_predictions, FW_int = fwp_build/fw), on the 14,883
AFJ-only rows and on the 4,617 AUN rows (by AUN.4's anchored formula, v18 - (resid_union - resid_afj)), and on the v4 lane
(v13 stage rows, made with the float F) exactly on >= 75 % and within 1 s on >= 99.5 % of rows after the v10-floor rows
(F' not reconstructable) are excluded and counted. The first replay smoke (2026-09-24) found R / AFJ exact, AUN off by the
anchoring (fixed here) and the v4 lane at 79.3 % / 99.59 % / max 32 s: F is a stack input, so a 0.5 s F' error moves the
residual as well; the v4-lane floors were set from that observation (disclosed in V34.7).
v34: F = step 3's delta_mean, resid = step 2's models_v4_nsc, C = step 6's npy (holdout ids asserted identical to CBV's),
F_orl_int = step 5's taxi_int, FW_int = step 4's taxi_int.
Label-free projection (registered before any compose): S = sum((v34 - v33)^2) / 344,841, G = S (2k - 1) at k = the NSC fold
result's day-block CI low / point / high (0.6194 / 0.7411 / 0.8677, recomputed from data/nsc_run/nsc_fold_preds.parquet).
Outputs (write-once): submissions/merry-quicksand_v34_nsc.{parquet,manifest.parquet,meta.json} and the byte copy
submissions/merry-quicksand_v34.parquet (= build_submission.submission_name(34), the upload key).
LightGBM (the stack's G half) in this process; the stack's MLP runs in scripts/adsb_nn_worker.py subprocesses (BC-5).
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import pathlib
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("v34_common", "scripts/analysis/v34_common.py")
BS = _load("build_submission", "scripts/build_submission.py")
from prc.pipeline import adsb_stage as AST  # noqa: E402

STEP = 7
KEY = BS.submission_name(34)
SUBS = ROOT / "submissions"
RUNS = ROOT / "data" / "pipeline_runs"
ADSB = ROOT / "data" / "adsb" / "v2"
V33, V33_SHA = SUBS / "merry-quicksand_v33.parquet", "0786a41af93ee610bba817e8b3ac10e22fefe24be63525d24512c0f861c02132"
V19, V19_SHA = SUBS / "merry-quicksand_v19.parquet", "5b5378b0f040deb5e0a1bc0314bc2674f959c52cf8dee94fc76c88f68bcb6c0d"
V18, V18_SHA = SUBS / "merry-quicksand_v18.parquet", "fd579f4c145ca0a024c22e95beee37439bec8ca611e6096894313a4b77a2da50"
V13 = RUNS / "v13_20260912T113635Z" / "merry-quicksand_v13.parquet"
V13_SHA = "2ac39110624144257be9a1f20c380803f7f3cf7f3ba4b640f1caf31be65309e7"
V13_PROV = RUNS / "v13_20260912T113635Z" / "merry-quicksand_v13.provenance.parquet"
V13_PROV_SHA = "5478d2b4562a7889cea1bcf9d4e0d62673b43892905301fd6b13ef0508dac5d5"
FWP_TOUCHED = RUNS / "fwp_20260917T065003Z" / "fwp_touched_rows.parquet"
FWP_TOUCHED_SHA = "5927b58293deb295235dac7fdcf3be212ec54a82216a45b21cca0282c0260b7c"
V30_MANIFEST = SUBS / "merry-quicksand_v30_oru.manifest.parquet"
V30_MANIFEST_SHA = "a598cbf2cfd4799fb5d536a81a1fee633e096389c7777f5c531b182c1b4f4155"
AFJ_TOUCHED = RUNS / "afj_20260916T212844Z" / "afj_touched_rows.parquet"
AFJ_TOUCHED_SHA = "7ba1e00e75e8faa786f96f6a9c31b98a09fd88f608a5fce96542761c61c5d1b4"
AUN_TOUCHED = RUNS / "aun_20260917T005910Z" / "aun_touched_rows.parquet"
AUN_TOUCHED_SHA = "6dbaa67c7770f112051153e7ce0cb885f3139c778b69e9a45e3174dfc289acaa"
TABLES = {"v4": (ADSB / "features_2026janjul_v4.parquet", "73e00767950b79d1848fe03a2dc9caefd588aab2879aa6eecea1c4bd2233aa37"),
          "afj": (ADSB / "features_2026janjul_afj.parquet", "c42dbce0564cdc6a78ba8ce0dd979db365bdfa596ad2e71100d2af201fec6842"),
          "aun": (ADSB / "features_2026janjul_aun.parquet", "e338bbb76d4a8411d23192c2a4aecb113416f9f4bdb2b15969876fe24ebeb141")}
RANK_STAND = ROOT / "data" / "cache_stand" / "ranking.parquet"
RANK_STAND_SHA = "1b1df5f988d9ee669f8bd70a74e8adb7a61db6f18bb8520473b6567264c28739"
V13_DETAIL = ADSB / "v13_detail.parquet"
V13_DETAIL_SHA = "c6aa8d6b489628363a9c28bb2cdc3ba214ff9c822e36dc7ce9b3e5a87521fa10"
CBV_INPUT = ROOT / "data" / "cache_stand" / "cbv_ship_input.parquet"
CBV_INPUT_SHA = "a210da02a6bbe4c0f91ae81d1b3512148f6d9f6c4b426dc18914e7a22b482916"
C_OLD, C_OLD_SHA = ROOT / "data" / "cache_stand" / "cbv_ship_C_2026.npy", "1d192deb581738b0ba5fdd081ae840def19e28a47f0a4570f30aed839dc92035"
FW_OLD, FW_OLD_SHA = ROOT / "data" / "fwp_build" / "fw" / "merry-quicksand_v6.parquet", "4dc2f6e89cba0a6f163ccd6a413c6f52cc0d820499b2674a48d7345fc230f9bd"
ORL_OLD, ORL_OLD_SHA = ROOT / "data" / "orl_build" / "orl_2026_predictions.parquet", "bde5c7f9e061ad4ca778d2b2cfc7ce0a6eefc760f38ab268ce52685a83f3279f"
MODELS_V4 = ADSB / "models_v4"
TEMPLATE = ROOT / "data" / "raw" / "submitting.parquet"
NSC_FOLD_PREDS = ROOT / "data" / "nsc_run" / "nsc_fold_preds.parquet"

N_M, N_R, N_U, N_N, N_AFJ_N, N_AUN = 339_551, 115_180, 5_364, 224_371, 14_911, 4_617
#: the NSC fold result (RESULT, 2026-09-25 00:24Z): k = sum(D e)/sum(D^2), D = taxi(NSC) - taxi(FW record), e = y - taxi(FW
#: record); day-block 95 % CI, 2,000 draws, default_rng(0), 62 days. Recomputed by fold_k (tested).
K_LO, K_POINT, K_HI = 0.6193854909496406, 0.7410649371959134, 0.8677143149372897
MDE_BMSE = 73.2                                  # the NSC RESULT's realised MDE (board MSE)
#: the v4-lane replay floor (set from the first replay smoke, 2026-09-24 21:02, which observed 79.3 % exact / 99.59 % within
#: 1 s / max 32 s after the 2 v10-floor rows; disclosed in AMENDMENT V34.7): far below either = a routing / model error
REPLAY_V4_EXACT_MIN, REPLAY_V4_WITHIN1_MIN = 0.75, 0.995
MANIFEST_COLS = ["MVT_ID_mvt", "lane", "v33", "candidate", "changed", "F", "resid", "C", "partner_int"]


# ---------------------------------------------------------------- registered statistics

def fold_k(preds: pd.DataFrame, n_boot: int = 2_000, seed: int = 0) -> dict:
    """k and its day-block CI from the NSC fold predictions (columns proxy, y, incumbent, nsc, month, doy)."""
    inc = C.taxi(preds.proxy, preds.incumbent)
    d = C.taxi(preds.proxy, preds.nsc) - inc
    e = preds.y.to_numpy(float) - inc
    day = preds.month.to_numpy().astype(int) * 1000 + preds.doy.to_numpy().astype(int)
    u, inv = np.unique(day, return_inverse=True)
    de, dd = np.bincount(inv, d * e), np.bincount(inv, d * d)
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(u), len(u))
        ks.append(de[idx].sum() / dd[idx].sum())
    lo, hi = np.percentile(ks, [2.5, 97.5])
    return {"k": float((d * e).sum() / (d * d).sum()), "ci95_day": [float(lo), float(hi)], "n_days": int(len(u))}


def projection(move, ks=(K_LO, K_POINT, K_HI)) -> dict:
    move = np.asarray(move, float)
    s = float((move ** 2).sum() / C.N_BOARD)
    return {"S": s, "k_low_point_high": list(ks), "G": [s * (2 * k - 1) for k in ks], "mde_bmse": MDE_BMSE,
            "clauses": {"WORKING": f"G >= -{MDE_BMSE} (board G = 254.1149^2 - score^2: v34 costs no more than the fold "
                                   "could resolve); a positive G inside [G_lo, G_hi] is recorded as in-range",
                        "REFUTED": f"G < -{MDE_BMSE}: 'removing the seconds costs nothing' is refuted on the board -> owner",
                        "note": "S is a magnitude, never a gain; G is read only from our own v34 result json"}}


# ---------------------------------------------------------------- lanes

def table_source(ids, afj_ids, aun_ids) -> np.ndarray:
    """Per N row, the ADS-B table its v33 value came from: AUN (union) over AFJ (adsb.fi fallback) over v4."""
    ids = np.asarray(ids, dtype="float64")
    aun = np.isin(ids, np.asarray(sorted(aun_ids), dtype="float64"))
    afj = np.isin(ids, np.asarray(sorted(afj_ids), dtype="float64"))
    return np.where(aun, "aun", np.where(afj, "afj", "v4")).astype(object)


def f_prime(ids, proxy, v10: pd.Series, v13: pd.Series) -> np.ndarray:
    """The replay's F: proxy - v10 where v13_detail holds the row (a v13 stage row), else proxy - v13 (AFJ.1 / AUN)."""
    ids = pd.Index(np.asarray(ids, dtype="float64"))
    base = np.where(ids.isin(v10.index), v10.reindex(ids).to_numpy(float), v13.reindex(ids).to_numpy(float))
    if not np.isfinite(base).all():
        raise AssertionError("a replay row has neither v10 nor v13")
    return np.asarray(proxy, float) - base


def floor_ids(ids, v10: pd.Series, v13: pd.Series) -> set:
    """Rows whose F' base (v10, else v13) sits at the 1 s floor: F cannot be reconstructed there (AFJ.1 / AUN refused
    them); the replay excludes and counts them."""
    ids = pd.Index(np.asarray(ids, dtype="float64"))
    base = np.where(ids.isin(v10.index), v10.reindex(ids).to_numpy(float), v13.reindex(ids).to_numpy(float))
    return set(ids[base <= 1.0].tolist())


def matched_values(rows: pd.DataFrame, F: pd.Series, tables: dict, predict, gate: dict, rrows: pd.DataFrame,
                   Cs: pd.Series, orl_int: pd.Series, fw_int: pd.Series) -> tuple:
    """(values by MVT_ID for N and R rows, per-row detail). `rows`: N rows (MVT_ID_mvt, ap, hr, proxy, table);
    `predict(frame, table_name)` -> resid; `rrows`: R rows (MVT_ID_mvt, proxy, in_U)."""
    vals, detail = [], []
    for name in ("v4", "afj", "aun"):
        g = rows[rows.table.to_numpy() == name]
        if g.empty:
            continue
        ids = g.MVT_ID_mvt.to_numpy("float64")
        f = F.reindex(ids).to_numpy(float)
        if not np.isfinite(f).all():
            raise AssertionError(f"{int((~np.isfinite(f)).sum())} N rows ({name}) have no F")
        frame = AST.stage_frame(ids, g.ap.to_numpy(), g.hr.to_numpy(float), g.proxy.to_numpy(float), f, tables[name])
        seen = {}

        def fn(fr, _name=name):
            seen["r"] = np.asarray(predict(fr, _name), dtype="float64")
            return seen["r"]
        value, _ = AST.apply_stage(frame, fn, gate)
        vals.append(pd.Series(value, index=ids))
        detail.append(pd.DataFrame({"MVT_ID_mvt": ids, "lane": f"N_{name}", "F": f, "resid": seen["r"],
                                    "C": np.nan, "partner_int": np.nan}))
    rid = rrows.MVT_ID_mvt.to_numpy("float64")
    c = Cs.reindex(rid).to_numpy(float)
    in_u = rrows.in_U.to_numpy(bool)
    partner = np.where(in_u, fw_int.reindex(rid).to_numpy(float), orl_int.reindex(rid).to_numpy(float))
    if not (np.isfinite(c).all() and np.isfinite(partner).all()):
        raise AssertionError("an R row lacks C or its partner integer taxi")
    v = np.rint(0.5 * C.taxi(rrows.proxy, c) + 0.5 * partner)            # cbv_ship_build.blend_values' formula
    if not (np.isfinite(v).all() and (v >= 1).all()):
        raise AssertionError("an R value is non-finite or below 1")
    vals.append(pd.Series(v, index=rid))
    detail.append(pd.DataFrame({"MVT_ID_mvt": rid, "lane": np.where(in_u, "U_fw", "R_orl"), "F": np.nan, "resid": np.nan,
                                "C": c, "partner_int": partner}))
    out = pd.concat(vals)
    if not out.index.is_unique:
        raise AssertionError("a row is in two lanes")
    return out, pd.concat(detail, ignore_index=True)


def splice(base: pd.DataFrame, values: pd.Series, outside_ids) -> pd.DataFrame:
    """`base` with `values` (by MVT_ID) written in; ids / order / dtypes unchanged; values integer >= 1; no value may land
    on an outside (unmatched / Rome-chain) id, and every other row stays byte-identical."""
    v = values.to_numpy(float)
    if not (np.isfinite(v).all() and (v >= 1).all() and (np.rint(v) == v).all()):
        raise ValueError("splice: a value is non-finite, below 1, or not an integer")
    if np.isin(values.index.to_numpy("float64"), np.asarray(outside_ids, dtype="float64")).any():
        raise AssertionError("splice: a value targets a row outside the matched lanes")
    pos = pd.Index(base.MVT_ID_mvt.to_numpy("float64")).get_indexer(values.index.to_numpy("float64"))
    if (pos < 0).any():
        raise AssertionError(f"splice: {int((pos < 0).sum())} ids not in the base")
    new = base.TAXITIME_SEC_mvt.to_numpy().copy()
    new[pos] = v.astype(new.dtype)
    cand = base.copy()
    cand["TAXITIME_SEC_mvt"] = new
    keep = np.ones(len(base), bool)
    keep[pos] = False
    if not np.array_equal(cand.TAXITIME_SEC_mvt.to_numpy()[keep], base.TAXITIME_SEC_mvt.to_numpy()[keep]):
        raise AssertionError("splice: a row outside the matched lanes changed")
    return cand


def anchored_value(v18, r_shipped, r_union) -> np.ndarray:
    """AUN.4 (scripts/analysis/aun_build.anchored_value, restated): v18 moved by the stage's own change on the union
    features. This is how v33's 4,617 AUN rows were made; the replay uses it, v34 does not need it (it has the float F)."""
    v18, rs, ru = (np.asarray(x, "float64") for x in (v18, r_shipped, r_union))
    return np.rint(np.maximum(v18 - (ru - rs), 1.0))


def replay_values(rows: pd.DataFrame, old: dict, tables: dict, predict, gate: dict, rrows: pd.DataFrame,
                  v18: pd.Series) -> tuple:
    """v33's matched values re-derived from the OLD inputs: matched_values, then the AUN rows by AUN.4 (the union residual
    minus the shipped-table (afj) residual, anchored on v18). Returns (values, detail)."""
    vals, det = matched_values(rows, old["F"], tables, predict, gate, rrows, old["C"], old["orl_int"], old["fw_int"])
    aun = rows[rows.table.to_numpy() == "aun"]
    if not aun.empty:
        ids = aun.MVT_ID_mvt.to_numpy("float64")
        ru = det.set_index("MVT_ID_mvt").resid.reindex(ids).to_numpy(float)
        frame = AST.stage_frame(ids, aun.ap.to_numpy(), aun.hr.to_numpy(float), aun.proxy.to_numpy(float),
                                old["F"].reindex(ids).to_numpy(float), tables["afj"])
        seen = {}

        def fn(fr):
            seen["r"] = np.asarray(predict(fr, "afj"), dtype="float64")
            return seen["r"]
        AST.apply_stage(frame, fn, gate)
        vals.loc[ids] = anchored_value(v18.reindex(ids).to_numpy(float), seen["r"], ru)
    return vals, det


def replay_verdict(replayed: pd.Series, v33: pd.Series, lanes: dict, floor_ids=()) -> dict:
    """Per lane: R, N_afj, N_aun EXACT (their v33 values were made from stored inputs); N_v4 (v13's float F was never
    stored; F' = proxy - v10 is +-0.5 s off and F is a stack INPUT, so the residual moves too): rows whose F' cannot be
    reconstructed (v10 at the 1 s floor) are excluded and counted, and the rest must be exact on >= REPLAY_V4_EXACT_MIN
    and within 1 s on >= REPLAY_V4_WITHIN1_MIN of rows -- a gross routing / table / model error drops both far below."""
    out = {}
    floor = np.asarray(sorted(floor_ids), dtype="float64")
    for name, ids in lanes.items():
        ids = np.asarray(ids, dtype="float64")
        n_floor = int(np.isin(ids, floor).sum())
        ids = ids[~np.isin(ids, floor)]
        dlt = replayed.reindex(ids).to_numpy(float) - v33.reindex(ids).to_numpy(float)
        if not np.isfinite(dlt).all():
            raise AssertionError(f"replay {name}: ids missing")
        a = np.abs(dlt)
        out[name] = {"n": int(len(a)), "n_floor_excluded": n_floor, "n_mismatch": int((a > 0).sum()),
                     "exact_share": float((a == 0).mean()) if len(a) else 1.0,
                     "within_1_share": float((a <= 1).mean()) if len(a) else 1.0,
                     "max_abs_s": float(a.max()) if len(a) else 0.0}
    exact = [k for k in out if k != "N_v4"]
    v4 = out.get("N_v4")
    out["pass"] = bool(all(out[k]["n_mismatch"] == 0 for k in exact)
                       and (v4 is None or (v4["exact_share"] >= REPLAY_V4_EXACT_MIN
                                           and v4["within_1_share"] >= REPLAY_V4_WITHIN1_MIN)))
    out["rule"] = (f"R, N_afj, N_aun exact; N_v4 (v10-floor rows excluded) exact >= {REPLAY_V4_EXACT_MIN:.0%} and "
                   f"within 1 s >= {REPLAY_V4_WITHIN1_MIN:.1%}")
    return out


def lane_ids(rows: pd.DataFrame, rrows: pd.DataFrame) -> dict:
    """{lane: ids} for the replay verdict: R (all CBV rows) and N by table."""
    out = {"R": rrows.MVT_ID_mvt.to_numpy("float64")}
    for t in ("v4", "afj", "aun"):
        out[f"N_{t}"] = rows.MVT_ID_mvt.to_numpy("float64")[rows.table.to_numpy() == t]
    return out


def _series(v33: pd.DataFrame) -> pd.Series:
    return pd.Series(v33.TAXITIME_SEC_mvt.to_numpy(float), index=v33.MVT_ID_mvt.to_numpy("float64"))


def build(v33, template, rows, rrows, tables, gate, old: dict, new: dict, predict_old, predict_new, outside_ids,
          paths: dict, extra_meta=None, allow_roots=(C.OUT_ROOT, C.SMOKE_ROOT), check_counts=True, log=print,
          gate_new=None) -> dict:
    """Replay gate -> v34 values -> splice -> checks -> S / G / rules -> write candidate, manifest, final copy, meta.
    `old` / `new`: dicts F, C, orl_int, fw_int. Nothing is written unless the replay passes."""
    targets = {k: C.check_output_path(p, allow_roots=allow_roots) for k, p in paths.items()}
    s33 = _series(v33)
    ids_n, ids_r = rows.MVT_ID_mvt.to_numpy("float64"), rrows.MVT_ID_mvt.to_numpy("float64")
    if check_counts and (len(ids_n), len(ids_r), int(rrows.in_U.sum())) != (N_N, N_R, N_U):
        raise AssertionError(f"lane sizes N {len(ids_n)} / R {len(ids_r)} / U {int(rrows.in_U.sum())}")
    t0 = time.time()
    rep_vals, _ = replay_values(rows, old, tables, predict_old, gate, rrows, old.get("v18"))
    replay = replay_verdict(rep_vals, s33, lane_ids(rows, rrows), old.get("floor_ids", ()))
    log(f"replay on the old inputs: {replay} ({time.time() - t0:.0f}s)")
    if not replay["pass"]:
        raise AssertionError(f"replay does not reproduce v33: {replay}")
    vals, detail = matched_values(rows, new["F"], tables, predict_new, gate if gate_new is None else gate_new, rrows,
                                  new["C"], new["orl_int"], new["fw_int"])
    cand = splice(v33, vals, outside_ids)
    out_mask = v33.MVT_ID_mvt.isin(np.asarray(outside_ids, dtype="float64")).to_numpy()
    outside_identical = bool(np.array_equal(cand.TAXITIME_SEC_mvt.to_numpy()[out_mask],
                                            v33.TAXITIME_SEC_mvt.to_numpy()[out_mask]))
    if not outside_identical:
        raise AssertionError("the unmatched lane / Rome chain is not byte-identical to v33")
    if template is not None:
        BS.check_submission(cand, template)
    move = cand.TAXITIME_SEC_mvt.to_numpy(float) - v33.TAXITIME_SEC_mvt.to_numpy(float)
    proj = projection(move)
    AFJ = _load("afj_ship_build", "scripts/analysis/afj_ship_build.py")
    m_matched = move[~out_mask]
    split = AFJ.move_split(m_matched, m_matched)
    rules = {"total": split["total"], "rules": split["rules"],
             "pass": bool(split["rules"]["concentration_gt300_le_47_8pct"] and split["rules"]["cap_rows_gt600_le_0_19pct"])}
    d = detail.set_index("MVT_ID_mvt").reindex(cand.MVT_ID_mvt.to_numpy("float64"))
    man = pd.DataFrame({"MVT_ID_mvt": cand.MVT_ID_mvt.to_numpy("float64"),
                        "lane": d.lane.fillna("outside_v33").to_numpy(object).astype(str),
                        "v33": v33.TAXITIME_SEC_mvt.to_numpy(), "candidate": cand.TAXITIME_SEC_mvt.to_numpy(),
                        "changed": move != 0, "F": d.F.to_numpy(float), "resid": d.resid.to_numpy(float),
                        "C": d.C.to_numpy(float), "partner_int": d.partner_int.to_numpy(float)})[MANIFEST_COLS]
    by_lane = {k: {"n": int((man.lane == k).sum()), "n_changed": int(man.changed[man.lane == k].sum()),
                   "S": float((move[(man.lane == k).to_numpy()] ** 2).sum() / C.N_BOARD)} for k in sorted(set(man.lane))}
    cand_sha = C.write_parquet_new(cand, targets["cand"], allow_roots=allow_roots)
    man_sha = C.write_parquet_new(man, targets["manifest"], allow_roots=allow_roots)
    C.check_output_path(targets["final"], allow_roots=allow_roots)
    shutil.copyfile(targets["cand"], targets["final"])
    final_sha = C.sha256(targets["final"])
    if final_sha != cand_sha:
        raise AssertionError("the final file is not a byte copy of the candidate")
    meta = C.jsonable({"tag": "V34.7 compose", "status": "BUILT", "upload_key": KEY, "replay": replay,
                       "counts": {"N": int(len(ids_n)), "R": int(len(ids_r)), "U": int(rrows.in_U.sum()),
                                  "outside": int(out_mask.sum())},
                       "by_lane": by_lane, "n_changed": int((move != 0).sum()), "outside_identical": outside_identical,
                       "projection": proj, "move_energy": proj["S"], "rules": rules,
                       "sum_abs_move_s": float(np.abs(move).sum()), "max_abs_move_s": float(np.abs(move).max()),
                       "candidate_sha256": cand_sha, "manifest_sha256": man_sha, "final_sha256": final_sha,
                       "paths": {k: C.rel(p) for k, p in targets.items()}, **(extra_meta or {}),
                       "upload": "NOT AUTHORISED - the owner decides per file; tools/verify_upload_v34.py (dry run)",
                       "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")})
    C.write_json_new(meta, targets["meta"], allow_roots=allow_roots)
    log(f"v34 built: changed {meta['n_changed']:,}; S {proj['S']:.2f}; G {[round(g, 1) for g in proj['G']]}; "
        f"rules pass {rules['pass']}; final {targets['final'].name} sha {final_sha[:12]}")
    return meta


# ---------------------------------------------------------------- real inputs

def check_sha(path, want) -> None:  # pragma: no cover - real files
    got = C.sha256(path)
    if got != want:
        raise AssertionError(f"{path} sha {got[:12]} is not the registered {want[:12]}")


def load_lanes() -> dict:  # pragma: no cover - real files (exercised by the replay smoke)
    """Every lane set from the shipped files, with the lineage assertions of the docstring."""
    for p, w in ((V33, V33_SHA), (V19, V19_SHA), (V13, V13_SHA), (V13_PROV, V13_PROV_SHA), (FWP_TOUCHED, FWP_TOUCHED_SHA),
                 (V30_MANIFEST, V30_MANIFEST_SHA), (AFJ_TOUCHED, AFJ_TOUCHED_SHA), (AUN_TOUCHED, AUN_TOUCHED_SHA),
                 (RANK_STAND, RANK_STAND_SHA), (V13_DETAIL, V13_DETAIL_SHA), *TABLES.values()):
        check_sha(p, w)
    v33 = pd.read_parquet(V33)
    prov = pd.read_parquet(V13_PROV, columns=["MVT_ID_mvt", "row_class"])
    M = prov.MVT_ID_mvt[prov.row_class == "matched"].to_numpy("float64")
    R = pd.read_parquet(FWP_TOUCHED, columns=["MVT_ID_mvt"]).MVT_ID_mvt.to_numpy("float64")
    u = pd.read_parquet(V30_MANIFEST, columns=["MVT_ID_mvt", "in_U"])
    U = u.MVT_ID_mvt[u.in_U].to_numpy("float64")
    Nn = np.setdiff1d(M, R)
    afj = set(pd.read_parquet(AFJ_TOUCHED, columns=["MVT_ID_mvt"]).MVT_ID_mvt.astype(float)) & set(Nn.tolist())
    aun = set(pd.read_parquet(AUN_TOUCHED, columns=["MVT_ID_mvt"]).MVT_ID_mvt.astype(float))
    if not (set(R) <= set(M) and set(U) <= set(R) and aun <= set(Nn.tolist())):
        raise AssertionError("lane nesting R in M, U in R, AUN in N does not hold")
    if (len(M), len(R), len(U), len(Nn), len(afj), len(aun)) != (N_M, N_R, N_U, N_N, N_AFJ_N, N_AUN):
        raise AssertionError(f"lane sizes {(len(M), len(R), len(U), len(Nn), len(afj), len(aun))}")
    s33, s19 = _series(v33), _series(pd.read_parquet(V19))
    if not np.array_equal(s33.reindex(Nn).to_numpy(), s19.reindex(Nn).to_numpy()):
        raise AssertionError("v33 != v19 on an N row: a later version touched the ADS-B-stage lane")
    rk = pd.read_parquet(RANK_STAND, columns=["MVT_ID_mvt", "ap", "hr", "proxy"])
    rk = rk.set_index(rk.MVT_ID_mvt.to_numpy("float64"))
    rows = pd.DataFrame({"MVT_ID_mvt": Nn, "ap": rk.ap.reindex(Nn).astype(str).to_numpy(),
                         "hr": rk.hr.reindex(Nn).to_numpy(float), "proxy": rk.proxy.reindex(Nn).to_numpy(float),
                         "table": table_source(Nn, afj, aun)})
    outside = np.setdiff1d(v33.MVT_ID_mvt.to_numpy("float64"), M)
    return {"v33": v33, "M": M, "R": R, "U": U, "N": Nn, "rows": rows, "outside": outside, "rank": rk}


def load_tables(M) -> dict:  # pragma: no cover
    return {k: AST.read_table(p, M)[0] for k, (p, _) in TABLES.items()}


def c_by_id(inp_path, npy) -> pd.Series:  # pragma: no cover
    """C aligned to its input's holdout rows (in input order), as cbv_ship_build.holdout_predictions does."""
    h = pd.read_parquet(inp_path, columns=["MVT_ID_mvt", "score_proxy", "holdout"])
    h = h[h.holdout.to_numpy(bool)].reset_index(drop=True)
    c = np.load(npy)
    if len(c) != len(h) or not np.isfinite(c).all():
        raise AssertionError(f"{npy}: {len(c)} predictions for {len(h)} holdout rows / non-finite")
    return pd.Series(c, index=h.MVT_ID_mvt.to_numpy("float64")), pd.Series(h.score_proxy.to_numpy(float),
                                                                           index=h.MVT_ID_mvt.to_numpy("float64"))


def old_inputs(L: dict) -> dict:  # pragma: no cover
    for p, w in ((CBV_INPUT, CBV_INPUT_SHA), (C_OLD, C_OLD_SHA), (FW_OLD, FW_OLD_SHA), (ORL_OLD, ORL_OLD_SHA)):
        check_sha(p, w)
    Cs, proxy_c = c_by_id(CBV_INPUT, C_OLD)
    v10 = pd.read_parquet(V13_DETAIL, columns=["MVT_ID_mvt", "v10"]).set_index("MVT_ID_mvt").v10.astype(float)
    v13 = _series(pd.read_parquet(V13))
    F = pd.Series(f_prime(L["N"], L["rows"].proxy.to_numpy(float), v10, v13), index=L["N"])
    fw = pd.read_parquet(FW_OLD).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.astype(float)
    orl = pd.read_parquet(ORL_OLD, columns=["MVT_ID_mvt", "orl_taxi_int"]).set_index("MVT_ID_mvt").orl_taxi_int.astype(float)
    check_sha(V18, V18_SHA)
    return {"F": F, "C": Cs, "orl_int": orl, "fw_int": fw, "proxy_c": proxy_c, "v18": _series(pd.read_parquet(V18)),
            "floor_ids": floor_ids(L["N"], v10, v13)}


def new_inputs(reports: dict, old: dict) -> dict:  # pragma: no cover
    """F from step 3, C from step 6 (its holdout ids must equal CBV's, in order), ORL from step 5, FW from step 4."""
    def preds(step):
        p = next(ROOT / k for k in reports[step]["outputs"] if k.endswith("predictions_2026.parquet"))
        return pd.read_parquet(p).set_index("MVT_ID_mvt")
    F = preds(3).delta_mean.astype(float)
    fw, orl = preds(4).taxi_int.astype(float), preds(5).taxi_int.astype(float)
    inp = next(ROOT / k for k in reports[6]["outputs"] if k.endswith("cbv_nsc_input.parquet"))
    npy = next(ROOT / k for k in reports[6]["outputs"] if k.endswith(".npy"))
    Cs, proxy_c = c_by_id(inp, npy)
    if not (np.array_equal(Cs.index.to_numpy(), old["C"].index.to_numpy())
            and np.array_equal(proxy_c.to_numpy(), old["proxy_c"].to_numpy())):
        raise AssertionError("C-NSC's holdout ids / proxies are not CBV's, in order")
    return {"F": F, "C": Cs, "orl_int": orl, "fw_int": fw, "proxy_c": proxy_c}


def r_rows(L: dict, proxy_c: pd.Series) -> pd.DataFrame:  # pragma: no cover
    R = L["R"]
    return pd.DataFrame({"MVT_ID_mvt": R, "proxy": proxy_c.reindex(R).to_numpy(float), "in_U": np.isin(R, L["U"])})


def predictor(models_dir):  # pragma: no cover - the stack's contract entry point, full bag
    from prc.pipeline import legacy
    real = legacy.adsb_predictor()
    return lambda frame, table: real(frame, models_dir, oof=False)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rss-limit-gib", type=float, required=True, help="in-process peak-RSS abort (the launcher passes it)")
    ap.add_argument("--smoke", action="store_true", help="the REPLAY only, on real data (no v34 input needed), scratch report")
    return ap.parse_args(argv)


def main(argv=None) -> int:  # pragma: no cover - real-data entry point; the replay half is exercised by --smoke
    import run_guards as G
    args = parse_args(argv)
    log = lambda m: print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)
    G.start_rss_guard(int(args.rss_limit_gib * 1024 ** 3))
    t0 = time.time()
    fh = None
    if not args.smoke:
        sha = C.require_registration(STEP)
        log(f"registration sha256 {sha} logged; AMENDMENT V34.7 present")
        reports = {s: C.read_step_report(s) for s in (1, 2, 3, 4, 5, 6)}
        for s in (1, 3, 4, 5):
            C.assert_no_seconds(reports[s]["features"], f"step {s} features")
        C.assert_no_seconds(reports[6]["numeric"], "step 6 numeric")
        C.assert_no_seconds(reports[2]["features"], "step 2 features")
        fh = G.acquire_lock()
    try:
        L = load_lanes()
        log(f"lanes: N {len(L['N']):,} R {len(L['R']):,} U {len(L['U']):,} outside {len(L['outside']):,}; v33 == v19 on N")
        tables = load_tables(L["M"])
        gate, _ = AST.read_gate(MODELS_V4)
        old = old_inputs(L)
        rrows = r_rows(L, old["proxy_c"])
        if args.smoke:
            vals, _ = replay_values(L["rows"], old, tables, predictor(MODELS_V4), gate, rrows, old["v18"])
            verdict = replay_verdict(vals, _series(L["v33"]), lane_ids(L["rows"], rrows), old["floor_ids"])
            out = C.SMOKE_ROOT / f"smoke_step7_{dt.datetime.now():%Y%m%dT%H%M%S}" / "replay_report.json"
            C.write_json_new({"step": STEP, "smoke": True, "status": "SMOKE", "replay": verdict,
                              "wall_s": round(time.time() - t0, 1), "peak_rss_gib": G.peak_rss() / 1024 ** 3}, out)
            log(f"REPLAY {'PASS' if verdict['pass'] else 'FAIL'}: {verdict}; wall {time.time() - t0:.0f}s; peak RSS "
                f"{G.peak_rss() / 1024 ** 3:.2f} GiB -> {out}")
            return 0 if verdict["pass"] else 3
        new = new_inputs(reports, old)
        models_nsc = ROOT / reports[2]["models_dir"]
        gate_nsc, _ = AST.read_gate(models_nsc)
        if gate_nsc != gate:
            raise AssertionError("the NSC stack's gate differs from models_v4's")
        preds = pd.read_parquet(NSC_FOLD_PREDS)
        fk = fold_k(preds)
        if abs(fk["k"] - K_POINT) > 1e-9 or abs(fk["ci95_day"][0] - K_LO) > 1e-9 or abs(fk["ci95_day"][1] - K_HI) > 1e-9:
            raise AssertionError(f"the fold k {fk} is not the registered one")
        paths = {"cand": C.V34_CANDIDATE, "manifest": C.V34_MANIFEST, "meta": C.V34_META, "final": C.V34_FINAL}
        extra = {"prereg_sha256": sha, "step_reports": {str(s): {"status": r["status"], "outputs": r["outputs"]}
                                                        for s, r in reports.items()},
                 "base_v33_sha256": V33_SHA, "fold_k": fk,
                 "no_seconds_certificate": {"F": reports[3]["n_features"], "FW": reports[4]["n_features"],
                                            "ORL": reports[5]["n_features"], "C_numeric": reports[6]["n_numeric"],
                                            "stack_feats": reports[2]["n_features"]}}
        meta = build(L["v33"], pd.read_parquet(TEMPLATE), L["rows"], rrows, tables, gate, old, new,
                     predict_old=predictor(MODELS_V4), predict_new=predictor(models_nsc), outside_ids=L["outside"],
                     paths=paths, extra_meta=extra, log=log, gate_new=gate_nsc)
        log(f"meta -> {C.V34_META}; wall {time.time() - t0:.0f}s; peak RSS {G.peak_rss() / 1024 ** 3:.2f} GiB")
        return 0 if meta["rules"]["pass"] else 3
    finally:
        if fh is not None:
            G.release_lock(fh)


if __name__ == "__main__":
    raise SystemExit(main())
