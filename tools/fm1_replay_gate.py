#!/usr/bin/env python3.11
"""FM1 AT14 — the HARD end-to-end replay gate (plans/PREREG_fm1_2026_09_28.md Amendment A1). Stored tier.

    /opt/homebrew/bin/python3.11 -B tools/fm1_replay_gate.py --out <new dir> [--per-stratum 100] [--runs 2] [--seed 0]
                                                             [--min-rows 5000] [--smoke]

Runs the FM1 package on REAL 2026 scored rows — raw ranking rows -> prc.pipeline.fm1.features (stand/queue/order/weather/ORL
blocks, the R12 ADS-B join) -> the package design (+ encodings fitted by the package on the 2025 stand caches) -> learners /
the ADS-B stack stage / the blend formula -> the value ledger -> finalise — for a seeded sample of EVERY airport x month x
lane of v36, and diffs each row against v36 (submissions/merry-quicksand_v36.parquet). Exit 0 = PASS, 1 = FAIL, 2 = usage.

Reference per row (asserted consistent before anything is compared): v36 = R12 serve at v36's five R12 airports
(v36 meta "airports"), v34 elsewhere (the v34 NSC manifest's lane / F / resid / C / partner), v33 on the unmatched rows.

What the package RECOMPUTES (listed per stratum in the report) vs what it takes STORED (the model was never saved):
  every row      the raw -> feature path, proxy, hour, the value formula, the ledger, finalise                  recomputed
  R12 N_v4       the ADS-B join (tables_801f9bcac9a5) + the stack residual (models_r12 through the stage contract)  recomputed
                 base F (the R12 refit saved predictions only)                                                   stored
  R_orl / U_fw   the partner's seed-0 delta from the saved v34 ORL / FW boosters on the package design, asserted bit-equal
                 to the stored seed-0 prediction; seeds 1-2 stored; the CatBoost C                               stored
  v34 N_*        the seed-0 F from the saved v34 F80 booster (bit-equal check), seeds 1-2 stored; the v34 stack residual
                 (models_v4_nsc on the v4 / afj / aun tables, not ported)                                        stored
  unmatched      the package's v26 lane (Amendment A3), in a CHILD process: fit from the twelve raw 2025 files (the
                 shipped loaders, S1 cells, MJS body, R2 fill, E2W's beta_W by nested LOMO) and served on the raw 2026
                 scored file (witness, storm anomaly, ARC's ARR join, E2's stand-inbound context)        recomputed
                 E2's coefficients and the two day-check p99 thresholds are frozen constants (knobs)      constant
The gate replays v36, so the lane runs with the storm anchor ON (modules.storm.mode=anchor), Rome ON, ARC at LIRF.
DROPPED components: the rows the shipped ARN step (v21) set — the v20 -> v21 diff (its build's base), 8 scored rows — are a NAMED exception
(listed with their ids and diffs, never a FAIL); ARN is dropped from FM1 (FMA §4.2). Every other unmatched row must replay.
The lane serves all 5,290 unmatched rows, so the report also checks the FULL unmatched population (not only the sample); a
non-exception population row beyond 1 s + floor also FAILS the gate.
Noise floor: the package computation is run --runs N >= 2 times; floor = the largest per-row spread across runs. A row FAILS if
|package - v36| > 1 s + floor, or if the package cannot produce it. The LightGBM refit floor (two reference refits) is the
`refit` tier's (day 8), not this script's.

Each v34 booster (~0.7 GB text, ~2.4 GB resident while predicting) runs in its own child process, one at a time.
Reads data/r12, data/nsc_v34 and submissions/ READ-ONLY; writes only --out.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prc.pipeline.fm1 import components as C  # noqa: E402
from prc.pipeline.fm1 import engine as G  # noqa: E402
from prc.pipeline.fm1 import errors as E  # noqa: E402
from prc.pipeline.fm1 import features as F  # noqa: E402
from prc.pipeline.fm1 import knobs as K  # noqa: E402
from prc.pipeline.fm1 import stages as S  # noqa: E402
from prc.pipeline.fm1 import unmatched as U  # noqa: E402

PATHS = {
    "v36": ROOT / "submissions/merry-quicksand_v36.parquet",
    "v36_meta": ROOT / "submissions/merry-quicksand_v36.parquet.meta.json",
    "v34": ROOT / "submissions/merry-quicksand_v34.parquet",
    "v34_manifest": ROOT / "submissions/merry-quicksand_v34_nsc.manifest.parquet",
    "r12_detail": ROOT / "data/r12/serve_801f9bcac9a5_full/matched_2026_r12_detail.parquet",
    "r12_served": ROOT / "data/r12/serve_801f9bcac9a5_full/matched_2026_r12.parquet",
    "r12_tables": ROOT / "data/r12/tables_801f9bcac9a5",
    "r12_models": ROOT / "data/r12/stack_801f9bcac9a5_full/models_r12",
    "ranking": ROOT / "data/raw/ranking.parquet",
    "caches": ROOT / "data/cache_stand",
    "v34_steps": ROOT / "data/nsc_v34",
    "v20": ROOT / "submissions/merry-quicksand_v20.parquet",
    "v21": ROOT / "submissions/merry-quicksand_v21.parquet",
}
#: components the shipped chain has and FM1 drops (FMA §4.2): (the step's BASE file, its output). ARN's build read v20
#: (scripts/analysis/arn_ship_build.py:25 V20); v17 -> v21 would also span the matched-lane steps v18-v20 (BC-32)
DROPPED = {"ARN": ("v20", "v21")}
#: v34 refit step per partner / F role: (step dir, booster stem)
STEPS = {"F80": ("step3_f_refit", "lgbm_nsc_f80"), "FW90": ("step4_fw_refit", "lgbm_nsc_fw90"),
         "ORL96": ("step5_orl_refit", "lgbm_nsc_orl96")}
TOL_S = 1.0
CHILD_THREADS = 2


# =================================================================================================================
# reference
# =================================================================================================================

def load_reference(paths=PATHS) -> pd.DataFrame:
    """One row per v36 id: MVT_ID_mvt, ap, month, source (r12 | v34 | v33), lane, v36, F, resid, C, partner_int.
    Refuses a v36 row whose value is not its declared source's value (the reference must be internally consistent)."""
    v36 = pd.read_parquet(paths["v36"])
    meta = json.loads(pathlib.Path(paths["v36_meta"]).read_text())
    r12_aps = set(meta["airports"])
    man = pd.read_parquet(paths["v34_manifest"])
    det = pd.read_parquet(paths["r12_detail"]).set_index("MVT_ID_mvt")
    served = pd.read_parquet(paths["r12_served"]).set_index("MVT_ID_mvt").taxi
    v34 = pd.read_parquet(paths["v34"]).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    rk = pq.read_table(paths["ranking"], columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]).to_pandas()
    rk = rk[rk.PHASE_mvt == "DEP"].set_index("MVT_ID_mvt")
    ids = v36.MVT_ID_mvt.to_numpy(float)
    ref = pd.DataFrame({"MVT_ID_mvt": ids, "v36": v36.TAXITIME_SEC_mvt.to_numpy(float)})
    ref["ap"] = rk.ADEP_mvt.reindex(ids).astype(str).to_numpy()
    ref["month"] = rk.MVT_TIME_UTC_mvt.reindex(ids).dt.month.to_numpy()
    m = man.set_index("MVT_ID_mvt").reindex(ids)
    ref["lane"] = m.lane.to_numpy()
    is_r12 = np.isin(ref.ap.to_numpy(), list(r12_aps)) & (ref.lane.to_numpy() != "outside_v33")
    ref["source"] = np.where(ref.lane == "outside_v33", "v33", np.where(is_r12, "r12", "v34"))
    for c in ("F", "resid", "C", "partner_int"):
        ref[c] = np.where(is_r12, det[c].reindex(ids).to_numpy(float), m[c].to_numpy(float))
    ref.loc[is_r12, "lane"] = det.lane.reindex(ids[is_r12]).to_numpy()
    bad_r12 = is_r12 & (ref.v36.to_numpy() != served.reindex(ids).to_numpy(float))
    bad_v34 = ~is_r12 & (ref.v36.to_numpy() != v34.reindex(ids).to_numpy(float))
    if bad_r12.any() or bad_v34.any():
        raise AssertionError(f"reference inconsistent: {int(bad_r12.sum())} R12 rows differ from the R12 serve file, "
                             f"{int(bad_v34.sum())} other rows differ from v34")
    if ref[["ap", "month", "lane"]].isna().any().any():
        raise AssertionError("a v36 id has no airport / month / lane")
    return ref


def sample_strata(ref: pd.DataFrame, per_stratum: int, seed: int, min_rows: int) -> pd.DataFrame:
    """Every (ap, month, source, lane) stratum present, up to per_stratum rows each (all rows of a smaller stratum)."""
    rng = np.random.default_rng(seed)
    parts = []
    for key, g in ref.groupby(["ap", "month", "source", "lane"], sort=True):
        take = g if len(g) <= per_stratum else g.iloc[np.sort(rng.choice(len(g), per_stratum, replace=False))]
        parts.append(take)
    out = pd.concat(parts, ignore_index=True)
    if len(out) < min_rows:
        raise ValueError(f"sample has {len(out):,} rows < {min_rows:,}; raise --per-stratum")
    return out


# =================================================================================================================
# the package computation
# =================================================================================================================

def gate_config(paths=PATHS) -> K.FM1Config:
    """The package config the gate runs: every feature block on (the v34 designs need ORL, R12 needs the ADS-B block),
    the R12 tables and stack, all ten airports; the unmatched lane as v36 served it: the v26 body WITH its storm anchor
    (v16 inside v24-v26), the Rome module (R2 / E1 / RLD / E2 / E2W) and ARC at LIRF."""
    return K.parse_config(K.nest({"adsb.table_dir": str(paths["r12_tables"]), "adsb.stack.models_dir": str(paths["r12_models"]),
                                  "modules.storm.mode": "anchor"}))


def dropped_rows(paths=PATHS) -> pd.Series:
    """{MVT_ID_mvt: component} of the scored rows a DROPPED component set in the shipped chain: the rows whose value its
    step changed against its build's base (ARN: v20 -> v21)."""
    out = {}
    for name, (before, after) in DROPPED.items():
        a = pd.read_parquet(paths[before]).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
        b = pd.read_parquet(paths[after]).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(a.index)
        for i in a.index[a.to_numpy() != b.to_numpy()]:
            out[float(i)] = name
    return pd.Series(out, dtype=object)


def check_exceptions(ref: pd.DataFrame, exceptions: pd.Series) -> None:
    """A named exception can never FAIL, so the set is bounded structurally (BC-32): every id must be one of the
    reference's unmatched (v33) rows — the only rows a dropped unmatched-lane component ever set."""
    unm = set(ref.MVT_ID_mvt[ref.source == "v33"].astype(float))
    bad = [i for i in exceptions.index.astype(float) if i not in unm]
    if bad:
        raise AssertionError(f"{len(bad)} named exception row(s) are not unmatched rows (e.g. {sorted(bad)[:3]}): the "
                             "dropped component's diff was taken against the wrong base")


def unmatched_values(cfg: K.FM1Config, paths=PATHS, log=print) -> tuple:
    """(value per scored unmatched id, lane info) from the package's unmatched lane, in a CHILD process (the shipped loader
    peaks near 3.3 GB; the parent never holds it): engine.fit on the unmatched lane alone (twelve raw 2025 files), then
    unmatched.serve on the raw scored file through a ledger, finalised."""
    with tempfile.TemporaryDirectory(prefix="fm1_gate_unm_") as t:
        (pathlib.Path(t) / "config.json").write_text(json.dumps(K.nest(dict(cfg.values)), default=list))
        r = subprocess.run([sys.executable, "-B", str(pathlib.Path(__file__).resolve()), "_unmatched", t,
                            str(paths["ranking"])], capture_output=True, text=True, timeout=5400)
        for line in r.stdout.splitlines()[-20:]:
            log(f"  [unmatched child] {line}")
        if r.returncode != 0:
            raise E.WorkerError(f"unmatched child failed: {r.stderr[-3000:]}")
        v = pd.read_parquet(pathlib.Path(t) / "values.parquet")
        info = json.loads((pathlib.Path(t) / "info.json").read_text())
    return pd.Series(v.TAXITIME_SEC_mvt.to_numpy(float), index=v.MVT_ID_mvt.to_numpy(float)), info


def _child_unmatched(t, ranking) -> int:
    import resource
    t = pathlib.Path(t)
    cfg = K.parse_config(json.loads((t / "config.json").read_text()))
    ucfg = cfg.replace(**{"lanes.matched": False})
    art = G.fit(ucfg, save=False, allow_heavy=True)
    rows = pd.read_parquet(ranking)
    ids = U.unmatched_ids(rows)
    led = S.Ledger(ids)
    stages = {s.name: s for s in S.plan(ucfg)}
    _, info, _ = U.serve(art.unmatched_lane, rows, ucfg, led, stages)
    out = led.finalise()
    out[["MVT_ID_mvt", "TAXITIME_SEC_mvt"]].to_parquet(t / "values.parquet", index=False)
    info = {"serve": info, "fit": art.fit_report.get("unmatched"), "decided_by": led.counts(),
            "child_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3}
    (t / "info.json").write_text(json.dumps(info, indent=1, sort_keys=True, default=str))
    return 0


def airport_month_rows(paths, ap: str, month: int) -> pd.DataFrame:
    """The raw scored rows of one airport-month: its departures and the arrivals at it (every block is airport-local)."""
    f = ((ds.field("PHASE_mvt") == "DEP") & (ds.field("ADEP_mvt") == ap)) | \
        ((ds.field("PHASE_mvt") == "ARR") & (ds.field("ADES_mvt") == ap))
    t = ds.dataset(paths["ranking"]).to_table(filter=f).to_pandas()
    return t[t.MVT_TIME_UTC_mvt.dt.month == month].reset_index(drop=True)


def cache_encodings(paths, keys, smooth) -> F.EncodingTables:
    """The package's EncodingTables over the twelve 2025 stand caches' training rows (what the v34 refits read), one key
    at a time so the string columns never sit in memory together."""
    files = sorted(pathlib.Path(paths["caches"]).glob("training_2025-*.parquet"))
    if len(files) != 12:
        raise E.SchemaError(f"expected 12 stand caches, found {len(files)}")
    yd = pd.concat([pd.read_parquet(p, columns=["y", "delta"]) for p in files], ignore_index=True)
    tables, priors = {}, None
    for k in keys:
        kf = pd.concat([pd.read_parquet(p, columns=[k]) for p in files], ignore_index=True)
        enc = F.EncodingTables.fit(kf, {"te": yd.y.to_numpy(float), "de": yd.delta.to_numpy(float)}, [k], smooth)
        tables.update(enc.tables)
        priors = enc.priors
    return F.EncodingTables(smooth=float(smooth), priors=priors, tables=tables, targets={"te": "te", "de": "de"})


def booster_predict(booster: pathlib.Path, X: np.ndarray, threads: int = CHILD_THREADS) -> np.ndarray:
    """Predict with one saved booster in a CHILD process (0.7 GB file, ~2.4 GB resident): the parent never holds one."""
    with tempfile.TemporaryDirectory(prefix="fm1_gate_") as t:
        xi, xo = pathlib.Path(t) / "X.npy", pathlib.Path(t) / "p.npy"
        np.save(xi, X, allow_pickle=False)
        r = subprocess.run([sys.executable, "-B", str(pathlib.Path(__file__).resolve()), "_booster", str(booster), str(xi),
                            str(xo), str(threads)], capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise E.WorkerError(f"booster child failed: {r.stderr[-2000:]}")
        return np.load(xo, allow_pickle=False)


def _child_booster(booster, xi, xo, threads) -> int:
    import lightgbm as lgb
    b = lgb.Booster(model_file=str(booster))
    np.save(xo, b.predict(np.load(xi, allow_pickle=False), num_threads=int(threads)).astype("float64"), allow_pickle=False)
    return 0


def package_values(sample: pd.DataFrame, paths=PATHS, log=print, unmatched_fn=None) -> tuple:
    """(one package value per sample row, per-row notes, component checks). Rows the package cannot produce are NaN.
    `unmatched_fn(cfg, log) -> (values by id, info)` serves the unmatched rows (default: unmatched_values, the child)."""
    cfg = gate_config(paths)
    weather = F.stand_ab().load_weather(cfg.path("data.weather_dir"))
    orl = F.read_orl_routes(cfg.path("features.orl_routes_file"))
    frames = []
    for (ap, month), g in sample[sample.source != "v33"].groupby(["ap", "month"], sort=True):
        raw = airport_month_rows(paths, ap, int(month))
        fr = F.build_frames(raw, serve=True, cfg=cfg, weather=weather, orl_routes=orl)
        m = fr.matched
        frames.append(m[m.MVT_ID_mvt.isin(g.MVT_ID_mvt)])
        log(f"  features {ap}-{int(month):02d}: {len(m):,} matched rows built, {int(m.MVT_ID_mvt.isin(g.MVT_ID_mvt).sum())} sampled")
    fm = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"MVT_ID_mvt": [], "proxy": []}))
    fm = fm.set_index("MVT_ID_mvt", drop=False)
    ids = sample.MVT_ID_mvt.to_numpy(float)
    missing = np.setdiff1d(ids[sample.source.to_numpy() != "v33"], fm.index.to_numpy())
    if len(missing):
        raise E.RoutingError(f"{len(missing)} sampled matched ids not admitted by the package's matched lane")
    s = sample.set_index("MVT_ID_mvt", drop=False)
    value = pd.Series(np.nan, index=ids)
    note = pd.Series("", index=ids, dtype=object)
    proxy = fm.proxy.reindex(ids)
    checks = {}
    # ---- unmatched: the package's v26 lane on the whole scored file (fit from raw 2025 + serve), sampled rows picked
    uid = s.index[s.source == "v33"]
    if len(uid):
        fn = unmatched_fn or (lambda c, lg: unmatched_values(c, paths, lg))
        lane, linfo = fn(cfg, log)
        value[uid] = lane.reindex(uid).to_numpy(float)
        note[uid] = ("recomputed (fit from raw 2025 + serve): bs.derive rows, S1 cells + LIRF logistic, witness, MJS body "
                     "(MAU extras), storm anomaly + anchor, R2 fill + date-slip, E1, RLD, ARC, E2 (frozen coefficients), "
                     "E2W (beta_W nested LOMO); constants: E2 coefficients, the two day-check p99")
        checks["unmatched_lane"] = linfo
        checks["unmatched_population"] = lane
        log(f"  unmatched lane: {len(lane):,} rows served, {int(lane.reindex(uid).notna().sum())} of {len(uid)} sampled")
    # ---- partners and v34 F: seed 0 recomputed through the package design from the saved v34 boosters ----
    roles = {"ORL96": s.index[(s.lane == "R_orl")], "FW90": s.index[(s.lane == "U_fw")],
             "F80": s.index[(s.source == "v34") & s.lane.str.startswith("N_")]}
    delta_mean = {}
    enc = cache_encodings(paths, F.stand_ab().ENC_KEYS, F.stand_ab().SMOOTH) if any(len(r) for r in roles.values()) else None
    for role, rid in roles.items():
        if len(rid) == 0:
            continue
        step, stem = STEPS[role]
        rep = json.loads((paths["v34_steps"] / step / "report.json").read_text())
        X = F.matched_design(fm.loc[rid].reset_index(drop=True), enc, rep["features"])
        p0 = booster_predict(paths["v34_steps"] / step / f"{stem}_seed0.txt", X)
        st = pd.read_parquet(paths["v34_steps"] / step / "predictions_2026.parquet").set_index("MVT_ID_mvt").reindex(rid)
        exact = p0 == st.d_seed0.to_numpy(float)
        checks[role] = {"n": int(len(rid)), "seed0_bit_equal": int(exact.sum()),
                        "seed0_max_abs_diff": float(np.max(np.abs(p0 - st.d_seed0.to_numpy(float))))}
        delta_mean[role] = pd.Series((p0 + st.d_seed1.to_numpy(float) + st.d_seed2.to_numpy(float)) / 3.0, index=rid)
        log(f"  {role}: seed-0 bit-equal {int(exact.sum())}/{len(rid)}")
    # ---- R12 N_v4: the package stack stage on the package frame, stored R12 F ----
    r12n = s.index[(s.source == "r12") & (s.lane == "N_v4")]
    if len(r12n):
        f = fm.loc[r12n].reset_index(drop=True)
        gate = C.gate_manifest(paths["r12_models"])
        vals, elig, resid, _ = C.stack_values(f, s.loc[r12n, "F"].to_numpy(float), C.stack_feats(paths["r12_models"]),
                                              paths["r12_models"], gate, 2)
        value[r12n] = vals
        checks["R12_stack"] = {"n": int(len(r12n)), "eligible": int(elig.sum()),
                               "resid_max_abs_diff_vs_stored": float(np.max(np.abs(resid - s.loc[r12n, "resid"].to_numpy(float))))}
        note[r12n] = "recomputed: features, proxy, hour, ADS-B join, stack residual; stored: R12 F"
    # ---- R / U lanes (both sources): blend with the recomputed partner ----
    for role, lane in (("ORL96", "R_orl"), ("FW90", "U_fw")):
        rid = s.index[s.lane == lane]
        if len(rid):
            partner = np.rint(np.maximum(proxy[rid].to_numpy(float) - delta_mean[role][rid].to_numpy(float), 1.0))
            value[rid] = C.blend_value(proxy[rid].to_numpy(float), s.loc[rid, "C"].to_numpy(float), partner, 0.5)
            note[rid] = f"recomputed: features, proxy, encodings, design, {role} seed-0 delta; stored: seeds 1-2, CatBoost C"
    # ---- v34 N lanes: recomputed F, stored v34 residual ----
    vn = roles["F80"]
    if len(vn):
        value[vn] = C.staged_value(proxy[vn].to_numpy(float), delta_mean["F80"][vn].to_numpy(float),
                                   s.loc[vn, "resid"].to_numpy(float))
        note[vn] = "recomputed: features, proxy, encodings, design, F80 seed-0; stored: seeds 1-2, v34 stack residual"
    # ---- through the ledger + finalise, exactly as predict() serves ----
    # ---- matched rows through the ledger + finalise, exactly as predict() serves them (the unmatched rows come
    # finalised from the lane's own ledger)
    out = pd.Series(np.nan, index=ids)
    out[uid] = value[uid].to_numpy()
    mm = (~np.isnan(value.to_numpy())) & (s.source.to_numpy() != "v33")
    if mm.any():
        led = S.Ledger(ids[mm])
        stage = {x.name: x for x in S.STAGE_ORDER}
        led.write(stage["base_model"], ids[mm], value.to_numpy()[mm], label="base_lightgbm")
        fin = led.finalise().set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.astype(float)
        out[ids[mm]] = fin.reindex(ids[mm]).to_numpy()
    return out.to_numpy(), note.to_numpy(), checks


# =================================================================================================================
# verdict
# =================================================================================================================

def verdict(sample: pd.DataFrame, runs: list, notes, tol: float = TOL_S, exceptions: pd.Series | None = None) -> tuple:
    """(rows frame, report). floor = max over rows of the spread across runs (NaN rows excluded); a row FAILS when the
    package produced no value, or |run0 - v36| > tol + floor — unless it is a named DROPPED-component row
    (`exceptions`: id -> component), which is listed under report["exceptions"] instead."""
    R = np.vstack(runs)
    both = np.isfinite(R).all(axis=0)
    spread = np.where(both, R.max(axis=0) - R.min(axis=0), 0.0)
    floor = float(spread.max()) if len(spread) else 0.0
    pkg = R[0]
    diff = np.abs(pkg - sample.v36.to_numpy(float))
    exc = (exceptions if exceptions is not None else pd.Series(dtype=object)).reindex(sample.MVT_ID_mvt.to_numpy(float))
    exc = exc.fillna("").astype(str).to_numpy()
    beyond = ~np.isfinite(pkg) | (diff > tol + floor)
    fail = beyond & (exc == "")
    rows = sample.assign(package=pkg, abs_diff=diff, fail=fail, note=notes, exception=exc)
    exceptions_rep = {}
    for name in sorted(set(exc) - {""}):
        m = exc == name
        exceptions_rep[name] = {"n": int(m.sum()), "n_differ": int((beyond & m).sum()),
                                "ids": sorted(float(i) for i in sample.MVT_ID_mvt.to_numpy(float)[m]),
                                "max_abs_diff": float(np.nanmax(diff[m]))}
    strata = {}
    for key, g in rows.groupby(["ap", "month", "source", "lane"], sort=True):
        strata["|".join(str(k) for k in key)] = {
            "n": int(len(g)), "n_fail": int(g.fail.sum()),
            "max_abs_diff": float(np.nanmax(g.abs_diff)) if np.isfinite(g.abs_diff).any() else None,
            "exact": int((g.abs_diff == 0).sum()), "note": str(g.note.iloc[0])}
    rep = {"verdict": "PASS" if not fail.any() else "FAIL", "n_rows": int(len(rows)), "n_fail": int(fail.sum()),
           "tolerance_s": tol, "noise_floor_s": floor, "n_runs": len(runs), "n_strata": len(strata),
           "fail_by_source_lane": {f"{a}|{b}": int(v) for (a, b), v in rows.groupby(["source", "lane"]).fail.sum().items()},
           "strata": strata, "exceptions": exceptions_rep}
    return rows, rep


def population_check(ref: pd.DataFrame, lanes: list, exceptions: pd.Series, tol: float = TOL_S) -> dict:
    """The unmatched lane served every scored unmatched row: compare ALL of them with v36 (floor = the spread across the
    runs). Non-exception rows beyond tol + floor are FAILS; exception rows are listed."""
    u = ref[ref.source == "v33"].set_index("MVT_ID_mvt")
    V = np.vstack([ln.reindex(u.index).to_numpy(float) for ln in lanes])
    ok = np.isfinite(V).all(axis=0)
    floor = float(np.where(ok, V.max(axis=0) - V.min(axis=0), 0.0).max()) if V.size else 0.0
    diff = np.abs(V[0] - u.v36.to_numpy(float))
    exc = exceptions.reindex(u.index).fillna("").astype(str).to_numpy()
    beyond = ~np.isfinite(V[0]) | (diff > tol + floor)
    fail = beyond & (exc == "")
    bad = u.index.to_numpy(float)[fail]
    return {"n": int(len(u)), "n_exact": int((diff == 0).sum()), "n_within_tol": int((~beyond).sum()), "floor_s": floor,
            "n_fail": int(fail.sum()), "fail_ids": sorted(float(i) for i in bad)[:50],
            "by_airport": {a: {"n": int((u.ap == a).sum()), "n_fail": int((fail & (u.ap.to_numpy() == a)).sum()),
                               "max_abs_diff_non_exception": float(np.nanmax(np.where(exc == "", diff, 0)[u.ap.to_numpy() == a]))}
                           for a in sorted(u.ap.unique())},
            "exceptions": {n: {"n": int((exc == n).sum()), "ids": sorted(float(i) for i in u.index.to_numpy(float)[exc == n]),
                               "diffs": {str(float(i)): float(d) for i, d in zip(u.index.to_numpy(float)[exc == n],
                                                                                 diff[exc == n])}}
                           for n in sorted(set(exc) - {""})}}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_booster":
        return _child_booster(*argv[1:5])
    if argv and argv[0] == "_unmatched":
        return _child_unmatched(*argv[1:3])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-stratum", type=int, default=100)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rows", type=int, default=5000)
    ap.add_argument("--smoke", action="store_true", help="one airport-month (LSZH, January), R12 lanes only: the code path")
    a = ap.parse_args(argv)
    if a.runs < 2:
        ap.error("--runs must be >= 2 (the noise floor needs two runs)")
    out = pathlib.Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f"refused: {out} is not empty", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref = load_reference()
    # the exception set is bounded against the FULL reference (its unmatched rows), before any --smoke subset
    exceptions = dropped_rows()
    check_exceptions(ref, exceptions)
    print(f"dropped-component rows (named exceptions): {exceptions.value_counts().to_dict()}", flush=True)
    if a.smoke:
        ref = ref[(ref.ap == "LSZH") & (ref.month == 1) & (ref.source == "r12")]
    sample = sample_strata(ref, a.per_stratum, a.seed, 0 if a.smoke else a.min_rows)
    print(f"sample: {len(sample):,} rows in {sample.groupby(['ap', 'month', 'source', 'lane']).ngroups} strata", flush=True)
    runs, notes, checks, lanes = [], None, [], []
    for i in range(a.runs):
        v, notes, ck = package_values(sample, log=lambda m: print(m, flush=True))
        runs.append(v)
        if "unmatched_population" in ck:
            lanes.append(ck.pop("unmatched_population"))
        checks.append(ck)
        print(f"run {i + 1}/{a.runs} done ({time.time() - t0:.0f}s)", flush=True)
    rows, rep = verdict(sample, runs, notes, exceptions=exceptions)
    if lanes:
        pop = population_check(ref, lanes, exceptions)
        rep["unmatched_population"] = pop
        if pop["n_fail"]:
            rep["verdict"] = "FAIL"
    rep.update(component_checks=checks, wall_s=round(time.time() - t0, 1), smoke=a.smoke, seed=a.seed,
               per_stratum=a.per_stratum, reference={k: str(v) for k, v in PATHS.items()},
               dropped_components={k: list(v) for k, v in DROPPED.items()})
    rows.to_parquet(out / "rows.parquet", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: rep.get(k) for k in ("verdict", "n_rows", "n_fail", "noise_floor_s", "fail_by_source_lane",
                                              "exceptions")}, indent=1))
    if "unmatched_population" in rep:
        print(json.dumps({k: v for k, v in rep["unmatched_population"].items() if k != "exceptions"}, indent=1))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
