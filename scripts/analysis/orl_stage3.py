"""ORL Stage 3 — the candidate build, ONLY if Stage 2 is WORKING (plans/PREREG_orl_osm_route_length_2026_09_23.md,
"Stage 3 — candidate only"). Nothing here uploads; the owner decides per file.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/orl_stage3.py --smoke        # synthetic world, seconds, no lock
    /opt/homebrew/bin/python3.11 -B scripts/analysis/orl_stage3.py --refit        # real: 12-month refit (~3 h, heavy lock)
    /opt/homebrew/bin/python3.11 -B scripts/analysis/orl_stage3.py --splice-only  # real: splice from the saved 2026 preds

Real launch (a later step, ONLY after reports/orl_stage2_report.json reads verdict WORKING, the pre-flight of the
registration's "Cost and guards" and the quality-gate skill; detached, nothing under /tmp):

    cd <repo root>
    /opt/homebrew/bin/python3.11 -B scripts/analysis/orl_stage3.py --smoke
    /opt/homebrew/bin/python3.11 -c "import subprocess,sys,datetime; t=datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'); subprocess.Popen([sys.executable,'-B','scripts/analysis/orl_stage3.py','--refit'], stdout=open(f'logs/orl_stage3_run_{t}.log','w'), stderr=subprocess.STDOUT, start_new_session=True); print(f'logs/orl_stage3_run_{t}.log')"
    # after it exits 0:
    /opt/homebrew/bin/python3.11 -B research/tools/verify_upload_v29.py   # dry run (the default): every gate, NO upload

What the stages are, and where each piece is reused from (by import, tested where it lives):
(a) serve design — the 99-column matrix (orl_stage2.stage2_feats: the production 93 + the six ORL columns) built by
    `lgbm_submit.load_frames` (12 training months + the 339,551-row ranking cache, queue/order/weather blocks) with the
    ORL block attached by `orl_stage1.attach_to_cache_frame` -> `orl_build.attach` on fit AND serve rows (one path);
    the serve rows are cross-checked against `orl_build.attach` on the RAW ranking.parquet keys (the audit's serve
    frame) and the fit-vs-serve NaN-share gate (`orl_build.nan_share_gate`) is re-run on this exact frame.
(b) production refit — `lgbm_submit` functions: split_masks (ES months 3, 9), stand_ab.infold_encodings on every
    training row, design_matrix, early_stop (ES seed 0, shipped P), fit_seeds at n_ref for seeds 0, 1, 2, mean_delta,
    recover_taxi_time (int32, the rounding FW_int had). The FWP recipe (`lgbm_submit.py --version 6 --queue
    --orderfeats --weatherfeats --seeds 0,1,2`) with 99 columns instead of 93.
(c) splice on v26 (the current best, sha b5b47f2d...) in fwp_build's form: R = the v13-stage `lightgbm:delta:mean3`
    scope rows (130,091), MINUS the ADS-B-stage AFJ rows (v18 != v17, 14,911), whose v26 value equals
    rint(0.5*max(proxy - C, 1) + 0.5*FW_int); new value rint(0.5*max(proxy - C, 1) + 0.5*F_orl_int); every other row
    byte-identical; ids/order/dtypes identical; check_submission on the frame and on the re-read. R must be exactly
    FWP's touched set (115,180) — see "Reading of the registration" below.
(d) label-free 2026 tripwires (INV section 6.3) BEFORE the candidate is written: move energy mean(move^2) by
    (month, airport) on R vs the Stage-2 fold-A reference (`tripwires.move_energy_2025_pop_by_month_airport`) — any
    2026 cell outside the 2025 [min, max] voids the k range; the share of moved rows / move energy on (stand, runway)
    pairs and stands with zero 2025 training rows; the NaN gate (a).
(e) projection G = S*(2k - 1), S = sum(move^2)/344,841 on R, k range [k2 - 0.5, k2] with k2 = the Stage-2 report's
    `composed.k` (k on the pop rows, k = sum(D*e)/sum(D^2), the registration's Stage-2 bullet); G < 0 refutes. The
    matched-lane transfer history is stated in the meta (HISTORY).
(f) pre-upload rules (AFJ/AUN, carried by FWP): squared-move share in |move| > 300 s <= 47.8 %, rows with
    |move| > 600 s <= 0.19 % of R (`afj_ship_build.move_split`).

Reading of the registration (named, not silently resolved; see UNTESTED/the stage-3 RESULT):
  * "R = the CBV rows whose value equals rint(0.5*max(proxy - C, 1) + 0.5*FW_int)" read literally on v26 selects
    115,230 rows: 50 of them are ADS-B-stage AFJ rows whose value coincides with the formula. The registration also
    says the 224,371 ADS-B-stage rows are untouched and that the splice is fwp_build's; FWP's R was 115,180 and did not
    include them. This module excludes the AFJ rows and ASSERTS R == FWP's touched set.
  * "any airport-month outside the 2025 range": read as outside [min, max] over the fold-A (month, airport) cells;
    per-cell ratios are reported too.
  * The run's RSS abort is 10 GiB, not Stage 1/2's 6 GiB: the FW 12-month refit (this recipe at 93 columns) measured
    7.53 GB peak RSS (data/fwp_build/fw/merry-quicksand_v6.meta.json). The registration's "~5 GB" is an estimate.
"""
from __future__ import annotations

import argparse, datetime as dt, gc, hashlib, importlib.util, json, pathlib, subprocess, sys, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


S2 = _load("orl_stage2", "scripts/analysis/orl_stage2.py")
S1, O, R = S2.S1, S2.O, S2.R
FWP = _load("fwp_build", "scripts/analysis/fwp_build.py")
CBV, AFJ, BS = FWP.CBV, FWP.AFJ, FWP.BS
LS = _load("lgbm_submit", "scripts/lgbm_submit.py")
S = LS.S                                                       # stand_ab, as lgbm_submit loaded it

PREREG, HASHLOG, REPORT2 = S2.PREREG, S2.HASHLOG, S2.REPORT2
ROUTES_SHA = S2.ROUTES_SHA
SUBS = ROOT / "submissions"
BASE = SUBS / "merry-quicksand_v26.parquet"
BASE_SHA = "b5b47f2db90570a3b55f70927486cc1304a3b727f009c2972d7ba8b6b7238fcd"
CAND = SUBS / "merry-quicksand_v29_orl.parquet"
META = SUBS / "merry-quicksand_v29_orl.meta.json"
MANIFEST = SUBS / "merry-quicksand_v29_orl.manifest.parquet"
TEMPLATE = ROOT / "data" / "raw" / "submitting.parquet"
RANKING = ROOT / "data" / "raw" / "ranking.parquet"
FW_FILE = FWP.fw_output_path(ROOT / "data" / "fwp_build" / "fw")
FWP_TOUCHED = ROOT / "data" / "pipeline_runs" / "fwp_20260917T065003Z" / "fwp_touched_rows.parquet"
AFJ_TOUCHED = ROOT / "data" / "pipeline_runs" / "afj_20260916T212844Z" / "afj_touched_rows.parquet"
OUT_DIR = ROOT / "data" / "orl_build"
PRED_2026 = OUT_DIR / "orl_2026_predictions.parquet"
REFIT_INFO = OUT_DIR / "orl_refit_info.json"
SMOKE_DIR = ROOT / "logs" / "orl_scratch_tools" / "smoke_stage3"

SEEDS = (0, 1, 2)
N_TRAIN, N_RANK, N_SCOPE, N_R, N_AFJ = 2_062_440, 339_551, 130_091, 115_180, 14_911
N_FEATS = 99
K_WIDTH = 0.5
BOARD_N = 344_841.0
CONC_BAR_PCT, CAP_GT600_PCT = AFJ.CONC_BAR_PCT, AFJ.CAP_GT600_PCT
RSS_LIMIT = 10 * 1024 ** 3
HISTORY = {"v20_FWP_direct_F_refit_on_CBV_rows": "board +264.1 vs P +78.8 = 3.35x P (this arm's object class)",
           "v22_BLS_stage_residual": "realised k -0.07", "v27_AGS_stage_member": "realised k 0.204",
           "reading": "INV sections 5/8: residual/identity-encoded stage arms invert; a direct F refit is not exposed. "
                      "The k range [k2 - 0.5, k2] is the registration's; nothing here is a board number."}
PRED_COLS = ["MVT_ID_mvt", "month", "ap", "orl_taxi_int", "pair_unseen", "stand_unseen"]


# ---------------------------------------------------------------- gates on the inputs

def stage2_gate(rep: dict) -> dict:
    """Stage 3 exists only if Stage 2 is WORKING. Returns k2 (composed.k, k on the pop rows), S_2025 and the fold-A
    move-energy reference by (month, airport); refuses a missing/non-finite k or an empty reference."""
    v = rep.get("verdict", {}).get("verdict")
    if v != "WORKING":
        raise RuntimeError(f"Stage 2 verdict is {v!r}, not WORKING: no Stage-3 candidate (registration)")
    k = float(rep["composed"]["k"])
    if not np.isfinite(k):
        raise AssertionError(f"Stage-2 composed k is not finite: {k}")
    ref = rep["tripwires"]["move_energy_2025_pop_by_month_airport"]
    if not ref:
        raise AssertionError("Stage-2 report carries no 2025 move-energy reference")
    return {"k2": k, "S_2025": float(rep["composed"]["S"]), "move_energy_2025": {str(a): float(b) for a, b in ref.items()}}


def check_feats(feats) -> list:
    """The 99 columns, exactly orl_stage2's arm list (production 93 + the six ORL columns, leak gate by name)."""
    _, arm = S2.stage2_feats(S)
    feats = list(feats)
    if feats != arm or len(feats) != N_FEATS:
        raise AssertionError(f"feature list is not orl_stage2's {N_FEATS}-column arm ({len(feats)} given)")
    return feats


# ---------------------------------------------------------------- (a) serve design

def raw_serve_keys(ids, path=RANKING) -> pd.DataFrame:
    """The three ORL keys of the RAW ranking.parquet (the audit's serve frame), one row per id in `ids`' order."""
    r = pd.read_parquet(path, columns=["MVT_ID_mvt"] + O.KEY_COLS)
    if not r.MVT_ID_mvt.is_unique:
        raise AssertionError("raw ranking carries a duplicate MVT_ID_mvt")
    ids = np.asarray(ids, dtype="float64")
    k = r.set_index("MVT_ID_mvt").reindex(ids)
    miss = ~np.isin(ids, r.MVT_ID_mvt.to_numpy("float64"))
    if miss.any():
        raise AssertionError(f"{int(miss.sum())} serve ids are absent from the raw ranking file")
    return k.reset_index(drop=True)[O.KEY_COLS]


def serve_crosscheck(rank_orl: pd.DataFrame, raw_keys: pd.DataFrame, routes: pd.DataFrame) -> int:
    """The serve rows' ORL block (attached through the cache frame) must equal orl_build.attach on the raw keys,
    value for value and NaN for NaN. Returns the matched count."""
    ref = O.attach(raw_keys.reset_index(drop=True), routes)
    got = rank_orl[O.ORL_COLS + [O.ORL_FLAG]].reset_index(drop=True)
    if len(ref) != len(got):
        raise AssertionError(f"serve rows {len(got)} != raw keys {len(ref)}")
    for c in O.ORL_COLS + [O.ORL_FLAG]:
        a, b = got[c].to_numpy(float), ref[c].to_numpy(float)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        if not same.all():
            raise AssertionError(f"serve ORL column {c} differs from attach() on the raw ranking keys on {int((~same).sum())} rows")
    return int(got[O.ORL_FLAG].sum())


def nan_gate(orl: pd.DataFrame, ap, is_rank) -> dict:
    """orl_build.nan_share_gate on this exact frame: fit = the training rows, serve = the ranking rows."""
    is_rank = np.asarray(is_rank, bool)
    f = orl[O.ORL_COLS].reset_index(drop=True).assign(ADEP_mvt=np.asarray(ap).astype(str))
    return O.nan_share_gate(f[~is_rank], f[is_rank])


def unseen_pair_mask(fit_keys: pd.DataFrame, serve_keys: pd.DataFrame) -> tuple:
    """Per serve row: (pair_unseen, stand_unseen) — its normalised (airport, stand, runway) pair / (airport, stand) has
    ZERO fit rows (orl_build.unseen_key_share's keys, row-level)."""
    def keys(f):
        ap = f.ADEP_mvt.map(O.norm_ref); st = f.STAND_mvt.map(O.norm_ref)
        rw = f.RUNWAY_mvt.map(lambda v: O.designator_from_mvt(v) or "")
        return (ap + "|" + st + "|" + rw).to_numpy(), (ap + "|" + st).to_numpy()
    fp, fs = keys(fit_keys); sp, ss = keys(serve_keys)
    return ~np.isin(sp, np.unique(fp)), ~np.isin(ss, np.unique(fs))


# ---------------------------------------------------------------- (b) production refit

def production_refit(d: pd.DataFrame, is_rank, feats, params, nest, patience, seeds=SEEDS, files=None, smoke=False,
                     enc_keys=None) -> dict:
    """lgbm_submit's refit on the frame `d` (training rows + ranking rows, flag `is_rank`), with `feats` as the design:
    ES run (seed ES_SEED, months 3/9) -> n_ref, one refit per seed on every training row, mean delta, int32 taxi time.
    The frame is modified in place (encodings added, key columns dropped). Ranking rows must carry no label."""
    is_rank = np.asarray(is_rank, bool)
    if not (d.y.to_numpy(float)[is_rank].size and np.isnan(d.y.to_numpy(float)[is_rank]).all()
            and np.isnan(d.delta.to_numpy(float)[is_rank]).all()):
        raise AssertionError("ranking rows carry a label (or there are none): not a serve-mode build")
    train, fit, es = LS.split_masks(d.month.to_numpy(), is_rank)
    y, dlt, proxy = d.y.to_numpy(float), d.delta.to_numpy(float), d.proxy.to_numpy(float)
    keys = list(S.ENC_KEYS if enc_keys is None else enc_keys)
    for c, v in S.infold_encodings(d, y, dlt, train, enc_keys=keys).items():
        d[c] = v
    enc_cols = [f"{p}_{k}" for k in keys for p in ("te", "de")]
    bad = [c for c in enc_cols if np.isnan(d[c].to_numpy(float)[is_rank]).any()]
    if bad:
        raise AssertionError(f"NaN encodings on ranking rows: {bad}")
    missing = [c for c in feats if c not in d.columns]
    if missing:
        raise KeyError(f"design columns absent from the frame: {missing}")
    X = LS.design_matrix(d, feats)
    if X.shape[1] != len(feats):
        raise AssertionError("design width != feature count")
    info = LS.early_stop(X, dlt, y, proxy, train, fit, es, dict(params, seed=LS.ES_SEED), nest, patience, target="delta")
    label = LS.regression_label(dlt, y, "delta")
    preds = LS.fit_seeds(X, label, train, X[is_rank], params, info["n_ref"], tuple(seeds), files=files, info=info,
                         smoke=smoke, n_features=len(feats), target="delta")
    del X; gc.collect()
    mean = LS.mean_delta([preds[s] for s in seeds])
    raw = proxy[is_rank] - mean
    taxi = LS.recover_taxi_time(proxy[is_rank], mean)
    sd = np.stack([preds[s] for s in seeds]).std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(mean))
    return {"delta_by_seed": {s: np.asarray(preds[s], float) for s in seeds}, "delta_mean": mean, "taxi_int": taxi,
            "info": info, "floor_bind_rate": float((raw < 1.0).mean()), "seed_sd_mean_s": float(sd.mean()),
            "n_train": int(train.sum()), "n_rank": int(is_rank.sum())}


# ---------------------------------------------------------------- (c) R and the splice

def compute_R(scope: pd.DataFrame, base: pd.DataFrame, fw_int: pd.DataFrame, adsb_ids) -> tuple:
    """R = scope rows NOT in the ADS-B-stage set whose base value equals rint(0.5*taxi_C + 0.5*FW_int). Returns
    (r frame [MVT_ID_mvt, proxy, C, taxi_C, base, FW_int], counts incl. the literal-rule coincident ADS-B rows)."""
    ids = scope.MVT_ID_mvt.to_numpy("float64")
    bv = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids).to_numpy("float64")
    if np.isnan(bv).any():
        raise AssertionError(f"{int(np.isnan(bv).sum())} scope ids are missing from the base submission")
    fw = fw_int.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids).to_numpy("float64")
    if not np.isfinite(fw).all():
        raise AssertionError(f"{int((~np.isfinite(fw)).sum())} scope ids lack a finite FW_int")
    formula = CBV.blend_values(scope.taxi_C.to_numpy(float), fw)
    eq = formula == bv
    in_adsb = np.isin(ids, np.asarray(list(adsb_ids), dtype="float64"))
    keep = eq & ~in_adsb
    r = scope.loc[keep, ["MVT_ID_mvt", "proxy", "C", "taxi_C"]].reset_index(drop=True).copy()
    r["base"] = bv[keep]; r["FW_int"] = fw[keep]
    counts = {"n_scope": int(len(ids)), "n_adsb_stage_in_scope": int(in_adsb.sum()),
              "n_adsb_coincident_with_formula": int((eq & in_adsb).sum()),
              "n_formula_mismatch": int((~eq & ~in_adsb).sum()), "n_R": int(keep.sum()),
              "n_literal_rule": int(eq.sum())}
    return r, counts


def new_values(r: pd.DataFrame, orl_taxi: pd.Series) -> np.ndarray:
    """rint(0.5*taxi_C + 0.5*F_orl_int) on R; F_orl_int must exist for every R id and be an integer >= 1."""
    v = orl_taxi.reindex(r.MVT_ID_mvt.to_numpy("float64")).to_numpy("float64")
    if not np.isfinite(v).all():
        raise AssertionError(f"{int((~np.isfinite(v)).sum())} R ids lack an F_orl taxi time")
    if (v < 1).any() or (np.rint(v) != v).any():
        raise AssertionError("F_orl taxi time is below 1 s or not an integer")
    return CBV.blend_values(r.taxi_C.to_numpy(float), v)


def splice_candidate(base: pd.DataFrame, r_ids, values, template: pd.DataFrame) -> tuple:
    """afj_ship_build.splice (R rows only), then independent checks: ids/order/dtypes, every row outside R
    byte-identical, check_submission."""
    cand, touched = AFJ.splice(base, r_ids, values)
    if not (np.array_equal(cand.MVT_ID_mvt.to_numpy(), base.MVT_ID_mvt.to_numpy()) and cand.dtypes.equals(base.dtypes)
            and list(cand.columns) == list(base.columns)):
        raise AssertionError("ids / order / dtypes differ from the base")
    outside = ~base.MVT_ID_mvt.isin(np.asarray(r_ids, dtype="float64")).to_numpy()
    if not np.array_equal(cand.TAXITIME_SEC_mvt.to_numpy()[outside], base.TAXITIME_SEC_mvt.to_numpy()[outside]):
        raise AssertionError("a row outside R changed")
    BS.check_submission(cand, template)
    return cand, touched


def write_and_reread(cand: pd.DataFrame, path, template: pd.DataFrame) -> str:
    """Write once (an existing file is refused), re-read, equality + check_submission on what landed. Returns sha256."""
    path = pathlib.Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: no silent overwrite")
    cand.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    if not back.equals(cand):
        raise AssertionError("re-read differs from the written frame")
    BS.check_submission(back, template)
    return FWP.sha256(path)


# ---------------------------------------------------------------- (d) tripwires, (e) projection, (f) rules

def move_energy_tripwire(move, ap, month, ref2025: dict) -> dict:
    """mean(move^2) by (month, airport) on R vs the fold-A reference; a 2026 cell outside [min, max] of the 2025 cells
    (inclusive) voids the registered k range."""
    move = np.asarray(move, float)
    e26 = R.move_energy(np.zeros(len(move)), move, ap, month)
    vals = np.asarray(list(ref2025.values()), float)
    lo, hi = float(vals.min()), float(vals.max())
    outside = sorted(c for c, v in e26.items() if not (lo <= v <= hi))
    ratio = {c: (v / ref2025[c] if c in ref2025 and ref2025[c] > 0 else None) for c, v in e26.items()}
    return {"move_energy_2026": e26, "range_2025": [lo, hi], "cells_outside": outside,
            "ratio_to_same_cell_2025": ratio, "cells_without_2025_reference": sorted(c for c in e26 if c not in ref2025),
            "k_range_void": bool(outside)}


def zero_training_moves(move, pair_unseen, stand_unseen) -> dict:
    """Share of moved rows and of move energy on (stand, runway) pairs / stands with zero 2025 training rows."""
    move = np.asarray(move, float); moved = move != 0; e = move ** 2
    out = {"n_moved": int(moved.sum())}
    for name, m in (("pair", np.asarray(pair_unseen, bool)), ("stand", np.asarray(stand_unseen, bool))):
        out[f"moved_row_share_{name}_unseen"] = float((moved & m).sum() / moved.sum()) if moved.any() else 0.0
        out[f"move_energy_share_{name}_unseen"] = float(e[m].sum() / e.sum()) if e.sum() > 0 else 0.0
    return out


def projection(move, k2: float, void: bool) -> dict:
    """G = S*(2k - 1) at both ends of the registered k range [k2 - 0.5, k2]; S = sum(move^2)/344,841."""
    move = np.asarray(move, float)
    s = float((move ** 2).sum() / BOARD_N)
    k_lo, k_hi = k2 - K_WIDTH, k2
    return {"S": s, "k2": float(k2), "k_range": [k_lo, k_hi], "G_range": [s * (2 * k_lo - 1), s * (2 * k_hi - 1)],
            "quotable": not void, "refuted_if": "board G < 0",
            "note": "S is a MAGNITUDE, never a gain; void => the k range (and G) is not quotable (INV section 6.3)"}


def pre_upload_rules(move) -> dict:
    """AFJ/AUN rules via afj_ship_build.move_split on R's move (over every R row)."""
    m = np.asarray(move, float)
    sp = AFJ.move_split(m, m)
    return {"total": sp["total"], "rules": sp["rules"], "pass": bool(sp["rules"]["concentration_gt300_le_47_8pct"]
                                                                   and sp["rules"]["cap_rows_gt600_le_0_19pct"])}


def per_airport_mean_move(ap, move) -> dict:
    g = pd.DataFrame({"ap": np.asarray(ap), "m": np.asarray(move, float)}).groupby("ap").m.agg(["size", "mean"])
    return {str(a): {"n": int(r["size"]), "mean_move_s": float(r["mean"])} for a, r in g.iterrows()}


# ---------------------------------------------------------------- build (splice + tripwires + meta)

def real_config() -> dict:
    return {"base": BASE, "base_sha": BASE_SHA, "cand": CAND, "meta": META, "manifest": MANIFEST, "template": TEMPLATE,
            "fw": FW_FILE, "afj": AFJ_TOUCHED, "fwp_touched": FWP_TOUCHED, "preds": PRED_2026, "stage2": REPORT2,
            "prereg": PREREG, "hashlog": HASHLOG, "refit_info": REFIT_INFO, "scope_parquet": None,
            "n_scope": N_SCOPE, "n_R": N_R, "n_afj": N_AFJ}


def load_scope(cfg: dict) -> pd.DataFrame:
    if cfg.get("scope_parquet"):
        return pd.read_parquet(cfg["scope_parquet"])
    return FWP.build_scope(FWP.SHIP_INPUT, FWP.C_PATH, FWP.V13_SUB, FWP.V13_PROV)


def build_candidate(cfg: dict) -> dict:
    """Everything after the refit, from files: gates, R, new values, tripwires (before writing), splice, write,
    manifest + meta. Refuses if the candidate exists. The candidate is written even when the k range is void or a
    rule fails — both are recorded, and the independent verifier refuses such a file."""
    for k in ("cand", "meta", "manifest"):
        if pathlib.Path(cfg[k]).exists():
            raise FileExistsError(f"{cfg[k]} exists: no silent rebuild")
    rep2 = json.loads(pathlib.Path(cfg["stage2"]).read_text())
    g2 = stage2_gate(rep2)
    prereg = S2.prereg_sha(cfg["prereg"], cfg["hashlog"])
    base_sha = FWP.sha256(cfg["base"])
    if base_sha != cfg["base_sha"]:
        raise AssertionError(f"base sha {base_sha} is not the registered current best {cfg['base_sha']}")
    template = pd.read_parquet(cfg["template"])
    base = pd.read_parquet(cfg["base"])
    BS.check_submission(base, template)
    scope = load_scope(cfg)
    if len(scope) != cfg["n_scope"]:
        raise AssertionError(f"scope has {len(scope)} rows, registered {cfg['n_scope']}")
    afj = pd.read_parquet(cfg["afj"])
    adsb = set(afj.MVT_ID_mvt[afj.v18 != afj.v17].astype(float))
    if len(adsb) != cfg["n_afj"]:
        raise AssertionError(f"AFJ set has {len(adsb)} rows, expected {cfg['n_afj']}")
    r, counts = compute_R(scope, base, pd.read_parquet(cfg["fw"]), adsb)
    if counts["n_R"] != cfg["n_R"]:
        raise AssertionError(f"|R| = {counts['n_R']}, registered {cfg['n_R']}")
    if set(r.MVT_ID_mvt) != set(pd.read_parquet(cfg["fwp_touched"]).MVT_ID_mvt.astype(float)):
        raise AssertionError("R is not FWP's touched set")
    preds = pd.read_parquet(cfg["preds"])
    if not preds.MVT_ID_mvt.is_unique:
        raise AssertionError("duplicate id in the 2026 predictions")
    p = preds.set_index("MVT_ID_mvt").reindex(r.MVT_ID_mvt.to_numpy("float64"))
    vals = new_values(r, preds.set_index("MVT_ID_mvt").orl_taxi_int)
    move = vals - r.base.to_numpy(float)
    ap, month = p.ap.to_numpy().astype(str), p.month.to_numpy().astype(int)
    # (d) BEFORE the candidate is written
    trip = move_energy_tripwire(move, ap, month, g2["move_energy_2025"])
    trip["zero_training"] = zero_training_moves(move, p.pair_unseen.to_numpy(bool), p.stand_unseen.to_numpy(bool))
    proj = projection(move, g2["k2"], trip["k_range_void"])
    rules = pre_upload_rules(move)
    cand, touched = splice_candidate(base, r.MVT_ID_mvt.to_numpy(), vals, template)
    cand_sha = write_and_reread(cand, cfg["cand"], template)
    man = pd.DataFrame({"MVT_ID_mvt": r.MVT_ID_mvt.to_numpy(), "ap": ap, "month": month, "proxy": r.proxy.to_numpy(),
                        "C": r.C.to_numpy(), "FW_int": r.FW_int.to_numpy(), "orl_taxi_int": p.orl_taxi_int.to_numpy(float),
                        "base": r.base.to_numpy(), "candidate": vals, "move": move, "changed": move != 0})
    man.to_parquet(cfg["manifest"], index=False)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    refit = json.loads(pathlib.Path(cfg["refit_info"]).read_text()) if pathlib.Path(cfg["refit_info"]).exists() else None
    meta = {"tag": "ORL", "stage": 3, "prereg_sha256": prereg, "head": head,
            "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "stage2_report_sha256": FWP.sha256(cfg["stage2"]), "stage2_verdict": "WORKING", "k2": g2["k2"],
            "base": pathlib.Path(cfg["base"]).name, "base_sha256": base_sha, "candidate_sha256": cand_sha,
            "manifest_sha256": FWP.sha256(cfg["manifest"]), "predictions_sha256": FWP.sha256(cfg["preds"]),
            "fw_sha256": FWP.sha256(cfg["fw"]), "counts": counts, "n_changed": int(touched.sum()),
            "shift": {"S": proj["S"], "mean_s": float(move.mean()), "max_abs_s": float(np.abs(move).max())},
            "projection": proj, "tripwires": trip, "rules": rules, "by_airport": per_airport_mean_move(ap, move),
            "history": HISTORY, "refit": refit,
            "upload": "NOT AUTHORISED - the owner decides per file; verify with research/tools/verify_upload_v29.py (dry run)"}
    pathlib.Path(cfg["meta"]).write_text(json.dumps(meta, indent=1, default=float))
    return meta


# ---------------------------------------------------------------- real refit orchestration

def run_refit() -> dict:
    """The 12-month production refit with the 99 columns; writes PRED_2026 + REFIT_INFO (never submissions/)."""
    started = time.monotonic()
    if PRED_2026.exists():
        raise FileExistsError(f"{PRED_2026} exists: no silent refit")
    g2 = stage2_gate(json.loads(REPORT2.read_text()))                  # refuses unless WORKING, before the lock
    G = _load("run_guards", "scripts/run_guards.py")
    fh = G.acquire_lock(); G.start_rss_guard(RSS_LIMIT)
    sha = S2.prereg_sha()
    if G.sha256_file(O.ROUTES) != ROUTES_SHA:
        raise AssertionError("route table sha256 is not the registered one")
    feats = check_feats(S2.stage2_feats(S)[1])
    routes = pd.read_parquet(O.ROUTES)
    d, is_rank, rank_ids = LS.load_frames(LS.CACHE, queue=True, order=True, weather=True)
    if int((~is_rank).sum()) != N_TRAIN or int(is_rank.sum()) != N_RANK:
        raise AssertionError(f"frame rows train {int((~is_rank).sum())} / rank {int(is_rank.sum())}")
    d = S1.attach_to_cache_frame(d, routes)
    n_match = serve_crosscheck(d.loc[is_rank], raw_serve_keys(rank_ids), routes)
    gate = nan_gate(d[O.ORL_COLS], d.ap.to_numpy(), is_rank)
    keys = d[["ap", "STAND_mvt", "RUNWAY_mvt"]].rename(columns={"ap": "ADEP_mvt"})
    pair_u, stand_u = unseen_pair_mask(keys[~is_rank], keys[is_rank])
    unseen = O.unseen_key_share(keys[~is_rank], keys[is_rank])
    month_r, ap_r = d.month.to_numpy()[is_rank].astype(int), d.ap.to_numpy()[is_rank].astype(str)
    del keys; gc.collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [(OUT_DIR / f"lgbm_orl99_seed{s}.txt", OUT_DIR / f"lgbm_orl99_seed{s}.fit.json") for s in SEEDS]
    res = production_refit(d, is_rank, feats, dict(LS.P), LS.NEST, LS.PATIENCE, SEEDS, files=files)
    out = pd.DataFrame({"MVT_ID_mvt": rank_ids, "month": month_r, "ap": ap_r, "orl_taxi_int": res["taxi_int"],
                        "pair_unseen": pair_u, "stand_unseen": stand_u, "F_orl_delta": res["delta_mean"],
                        **{f"d_seed{s}": res["delta_by_seed"][s] for s in SEEDS}})
    out.to_parquet(PRED_2026, index=False)
    info = {"tag": "ORL", "stage": 3, "prereg_sha256": sha, "routes_sha256": ROUTES_SHA, "k2": g2["k2"],
            "es_info": res["info"], "floor_bind_rate": res["floor_bind_rate"], "seed_sd_mean_s": res["seed_sd_mean_s"],
            "n_train": res["n_train"], "n_rank": res["n_rank"], "n_serve_orl_matched": n_match, "nan_gate": gate,
            "unseen_key_share_2026": unseen, "seconds": time.monotonic() - started,
            "peak_rss_GiB": G.peak_rss() / 1024 ** 3, "boosters": [str(p.relative_to(ROOT)) for p, _ in files]}
    REFIT_INFO.write_text(json.dumps(info, indent=1, default=float))
    G.release_lock(fh)
    return info


# ---------------------------------------------------------------- synthetic world (smoke + tests)

def synthetic_routes() -> pd.DataFrame:
    rows = []
    for icao in ("EGLL", "LFPG"):
        for i, st in enumerate(("A1", "A2", "B1", "B2")):
            for j, rw in enumerate(("09L", "27R")):
                rows.append({"icao": icao, "stand": st, "rwy": rw, "route_m": 1000.0 + 400 * i + 900 * j,
                             "n_edges": 10 + i + j, "straight_m": 700.0 + 300 * i + 500 * j})
    return pd.DataFrame(rows)


def synthetic_frame(n_train=4_000, n_rank=600, seed=0) -> tuple:
    """A cache-shaped frame: 12 training months + ranking rows (months 1/7, labels NaN) with every stand_ab ENC_KEY,
    two numeric features, delta = f(f1, route) + noise. Returns (d, is_rank, rank_ids, feats)."""
    rng = np.random.default_rng(seed)
    n = n_train + n_rank
    is_rank = np.r_[np.zeros(n_train, bool), np.ones(n_rank, bool)]
    month = np.where(is_rank, rng.choice([1, 7], n), rng.integers(1, 13, n))
    ap = rng.choice(["EGLL", "LFPG"], n)
    stand = rng.choice(["A1", "A2", "B1", "B2", "Z9"], n)
    rwy = rng.choice(["09L", "27R"], n)
    d = pd.DataFrame({"ap": ap, "ADEP_mvt": ap, "STAND_mvt": stand, "RUNWAY_mvt": rwy, "month": month,
                      "f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    for k in S.ENC_KEYS:
        if k not in d.columns:
            d[k] = rng.choice(["a", "b", "c"], n)
    d = S1.attach_to_cache_frame(d, synthetic_routes())
    route = d.orl_route_m.fillna(1500.0).to_numpy()
    delta = 200 * d.f1.to_numpy() + 0.05 * route + rng.normal(0, 5, n)
    y = 600 + 50 * d.f2.to_numpy() + rng.normal(0, 5, n)
    d["proxy"] = y + delta
    d["y"] = np.where(is_rank, np.nan, y); d["delta"] = np.where(is_rank, np.nan, delta)
    feats = ["f1", "f2", "te_STAND_mvt", "de_RUNWAY_mvt"] + O.ORL_COLS + [O.ORL_FLAG]
    return d, is_rank, np.arange(1, n_rank + 1, dtype="float64") * 10.0, feats


def synthetic_world(out: pathlib.Path, rank_ids=None, orl_taxi=None, seed=0, k2=0.9, ref_hi=1e6) -> dict:
    """Every file build_candidate and the verifier read, on synthetic ids, consistent by construction. Returns cfg."""
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = 400
    ids = np.arange(1, n + 1, dtype="float64") * 10.0 if rank_ids is None else np.asarray(rank_ids, "float64")
    extra = np.arange(1, 31, dtype="float64") * 10.0 + 0.5                        # "unmatched" rows, never touched
    all_ids = np.r_[ids, extra]
    n_m = len(ids)
    stage = np.where(np.arange(n_m) < int(0.6 * n_m), CBV.SCOPE_STAGE, "lightgbm:delta:mean3+adsb")
    scope_ids = ids[stage == CBV.SCOPE_STAGE]
    proxy = rng.uniform(600, 1500, n_m); C = rng.normal(0, 150, n_m)
    fw = np.rint(rng.uniform(300, 1200, n_m)).astype("int32")
    taxi_c = np.maximum(proxy - C, 1)
    cbv_fw = CBV.blend_values(taxi_c, fw)
    base_v = np.rint(rng.uniform(300, 1200, n_m))
    ns = len(scope_ids)
    afj_pos = np.arange(ns)[: ns // 10]                                          # first 10 % of scope = AFJ rows
    later_pos = np.arange(ns)[ns // 10: ns // 10 + 5]                            # 5 rows touched by a later lever
    base_v[:ns] = cbv_fw[:ns]
    base_v[afj_pos[3:]] = cbv_fw[afj_pos[3:]] + 7                                # AFJ values; 3 coincide with the formula
    base_v[later_pos] = cbv_fw[later_pos] + 11
    base = pd.DataFrame({"MVT_ID_mvt": all_ids, "TAXITIME_SEC_mvt": np.r_[base_v, rng.integers(100, 900, len(extra))].astype("int32")})
    template = pd.DataFrame({"MVT_ID_mvt": all_ids[::-1].copy(), "TAXITIME_SEC_mvt": np.zeros(len(all_ids), "int32")})
    r_pos = np.setdiff1d(np.arange(ns), np.r_[afj_pos, later_pos])
    ap = np.where(np.arange(n_m) % 2 == 0, "EGLL", "LFPG"); month = np.where(np.arange(n_m) % 3 == 0, 1, 7)
    if orl_taxi is None:
        orl_taxi = np.rint(np.maximum(fw + rng.normal(0, 20, n_m), 1)).astype("int32")
    preds = pd.DataFrame({"MVT_ID_mvt": ids, "month": month, "ap": ap, "orl_taxi_int": np.asarray(orl_taxi),
                          "pair_unseen": np.arange(n_m) % 17 == 0, "stand_unseen": np.arange(n_m) % 29 == 0})
    ship = pd.DataFrame({"MVT_ID_mvt": np.r_[extra[:5], ids], "score_proxy": np.r_[np.full(5, 500.0), proxy],
                         "holdout": np.r_[np.zeros(5, bool), np.ones(n_m, bool)], "delta": np.r_[np.ones(5), np.full(n_m, np.nan)]})
    scope = pd.DataFrame({"MVT_ID_mvt": scope_ids, "proxy": proxy[:ns], "C": C[:ns], "taxi_C": taxi_c[:ns],
                          "v13": fw[:ns], "airport": ap[:ns]})
    cfg = {k: out / v for k, v in {"base": "base_v26.parquet", "cand": "cand_v29.parquet", "meta": "cand_v29.meta.json",
                                   "manifest": "cand_v29.manifest.parquet", "template": "submitting.parquet",
                                   "fw": "fw_v6.parquet", "afj": "afj_touched.parquet", "fwp_touched": "fwp_touched.parquet",
                                   "preds": "orl_2026_predictions.parquet", "stage2": "orl_stage2_report.json",
                                   "prereg": "PREREG_orl_smoke.md", "hashlog": "orl_prereg_hashes.log",
                                   "refit_info": "orl_refit_info.json", "scope_parquet": "scope.parquet",
                                   "prov": "v13.provenance.parquet", "ship": "cbv_ship_input.parquet",
                                   "c_npy": "cbv_ship_C_2026.npy"}.items()}
    base.to_parquet(cfg["base"], index=False); template.to_parquet(cfg["template"], index=False)
    pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": fw}).to_parquet(cfg["fw"], index=False)
    pd.DataFrame({"MVT_ID_mvt": np.r_[scope_ids[afj_pos], ids[-3:]], "v17": np.r_[np.zeros(len(afj_pos)), np.ones(3)],
                  "v18": np.r_[np.ones(len(afj_pos)), np.ones(3)]}).to_parquet(cfg["afj"], index=False)
    pd.DataFrame({"MVT_ID_mvt": scope_ids[r_pos]}).to_parquet(cfg["fwp_touched"], index=False)
    preds.to_parquet(cfg["preds"], index=False); scope.to_parquet(cfg["scope_parquet"], index=False)
    pd.DataFrame({"MVT_ID_mvt": ids, "stage": stage}).to_parquet(cfg["prov"], index=False)
    ship.to_parquet(cfg["ship"], index=False); np.save(cfg["c_npy"], C)
    ref = {"1:EGLL": 0.0, "7:LFPG": float(ref_hi)}
    rep2 = {"verdict": {"verdict": "WORKING"}, "composed": {"k": k2, "S": 50.0},
            "tripwires": {"move_energy_2025_pop_by_month_airport": ref}}
    cfg["stage2"].write_text(json.dumps(rep2))
    cfg["prereg"].write_text("# synthetic registration\n")
    h = hashlib.sha256(cfg["prereg"].read_bytes()).hexdigest()
    cfg["hashlog"].write_text(f"2026  REGISTERED  {cfg['prereg'].name}  (full {h})  -- synthetic\n")
    base_sha = hashlib.sha256(cfg["base"].read_bytes()).hexdigest()
    cfg.update({"base_sha": base_sha, "n_scope": int(ns), "n_R": int(len(r_pos)), "n_afj": int(len(afj_pos)),
                "scope_stage": CBV.SCOPE_STAGE})
    return cfg


def run_smoke(out=SMOKE_DIR) -> dict:
    """Synthetic end to end: refit wrapper (60 rounds) -> serve preds -> build_candidate -> the independent verifier.
    No lock (no real data, seconds), nothing under /tmp, nothing under submissions/."""
    out = pathlib.Path(out)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out / stamp; run_dir.mkdir(parents=True, exist_ok=False)
    d, is_rank, rank_ids, feats = synthetic_frame()
    p = dict(LS.P); p["learning_rate"] = LS.SMOKE["learning_rate"]
    res = production_refit(d, is_rank, feats, p, 60, 20, SEEDS, enc_keys=None)
    cfg = synthetic_world(run_dir, rank_ids=rank_ids, orl_taxi=res["taxi_int"])
    meta = build_candidate(cfg)
    V = _load("verify_upload_v29", "research/tools/verify_upload_v29.py")
    ver = V.verify(V.config_from(cfg), upload=False)
    print(json.dumps({"smoke": "SMOKE - code path only, NOT a result", "dir": str(run_dir), "n_R": meta["counts"]["n_R"],
                      "verifier": ver["status"], "best_iter": res["info"]["best_iter"]}), flush=True)
    return {"meta": meta, "verify": ver, "refit": res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--refit", action="store_true", help="real 12-month refit, then the splice")
    g.add_argument("--splice-only", action="store_true", help="real splice from the saved 2026 predictions")
    a = ap.parse_args(argv)
    if a.smoke:
        run_smoke(); return 0
    if a.refit:
        run_refit()
    meta = build_candidate(real_config())
    print(json.dumps({k: meta[k] for k in ("counts", "shift", "projection", "rules", "candidate_sha256")}, indent=1, default=float))
    print(json.dumps({"k_range_void": meta["tripwires"]["k_range_void"], "cells_outside": meta["tripwires"]["cells_outside"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
