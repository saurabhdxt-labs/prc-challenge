"""FM1's engine: fit, predict, build_submission (the public surface is prc.api; the CLI is prc.__main__).

fit(cfg)      raw training months -> features -> encodings -> base learner (+ CatBoost partner) -> the unmatched lane
              (v26: the shipped chain's training frames + E2W's beta_W, prc/pipeline/fm1/unmatched.py; hgb: the provisional
              body) (+ MTX mapping when calibration is on) -> a ModelArtifact, saved under run.artifacts_dir by default.
predict(art, rows)  raw scored rows -> the stage plan, in the fixed order, through the value ledger -> one row per scored
              departure with TAXITIME_SEC_mvt, decided_by and every component's own output.
build_submission(art, scored file, template, out_dir, version) -> <team>_v<N>.parquet in the template's id order, plus the
              per-row provenance and a manifest (with the sha256 of the inputs the fit recorded, as they are at build
              time, and of the published MTX file; and the inputs changed since the fit). Never overwrites; never reads
              a previous submission.

Heavy guard: a fit over more than HEAVY_ROWS raw training rows needs allow_heavy=True (the pre-run checks first; one heavy
job at a time; AC power). The tests' worlds are a few thousand rows.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import artifact as A
from . import components as C
from . import eobt_clip as EC
from . import errors as E
from . import features as F
from . import knobs as K
from . import learners as L
from . import ldi as LDI
from . import rows as R
from . import shipped
from . import stages as S
from . import unmatched as U

HEAVY_ROWS = 150_000
TRAIN_FILE_RE = re.compile(r"^training_(\d{4})-(\d{2})-01_")


def _log(msg: str) -> None:
    print(f"[fm1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def as_config(config, overrides=()) -> K.FM1Config:
    if isinstance(config, K.FM1Config):
        return config if not overrides else K.parse_config(K.nest(config.values), root=config.root, overrides=overrides)
    return K.load_config(config, overrides=overrides)


# =================================================================================================================
# inputs
# =================================================================================================================

def files_by_month(cfg) -> dict:
    """{month: [paths]} of every training file in data.raw_dir matching data.training_glob."""
    d = cfg.path("data.raw_dir")
    found: dict = {}
    for p in sorted(d.glob(cfg["data.training_glob"])):
        m = TRAIN_FILE_RE.match(p.name)
        if not m:
            raise E.SchemaError(f"training file {p.name} does not follow training_YYYY-MM-01_<next>.parquet")
        found.setdefault(int(m.group(2)), []).append(p)
    return found


def month_file(cfg, month: int) -> pathlib.Path:
    files = files_by_month(cfg).get(month, [])
    if len(files) != 1:
        raise E.SchemaError(f"month {month} has {len(files)} training file(s) in {cfg.path('data.raw_dir')} "
                            f"({cfg['data.training_glob']})")
    return files[0]


def training_files(cfg) -> dict:
    """{month: path} of the configured training months; refuses a month without exactly one file."""
    return {m: month_file(cfg, m) for m in cfg["data.train_months"]}


def in_registry(raw: pd.DataFrame, cfg) -> np.ndarray:
    """Rows of the configured airports: departures from them, arrivals at them."""
    reg = set(cfg["airports.codes"])
    dep = raw.PHASE_mvt.astype(str).to_numpy() == "DEP"
    return np.where(dep, raw.ADEP_mvt.astype(str).isin(reg).to_numpy(), raw.ADES_mvt.astype(str).isin(reg).to_numpy())


def _dir_hashes(d: pathlib.Path, prefix: str) -> dict:
    return {f"{prefix}/{p.relative_to(d)}": A.sha256_file(p) for p in sorted(d.rglob("*")) if p.is_file()}


def fit_inputs(cfg, files: dict, hash_file=None) -> dict:
    """{input name: sha256} of everything a fit reads. `hash_file` (default artifact.sha256_file, which refuses a missing
    file) may return None for a file it does not find: that input is then left out."""
    h = hash_file or A.sha256_file
    out = {f"raw/{p.name}": h(p) for p in files.values()}
    if cfg["lanes.matched"]:
        if cfg["features.weather"]:
            out.update(_dir_hashes(cfg.path("data.weather_dir"), "weather"))
        if cfg["features.orl_route_length"]:
            out["orl_routes"] = h(cfg.path("features.orl_routes_file"))
        if cfg["features.adsb"]:
            for n in ("train.parquet", "serve.parquet"):
                out[f"adsb/{n}"] = h(cfg.path("adsb.table_dir") / n)
        if cfg["adsb.stack.enabled"]:
            out.update(_dir_hashes(cfg.path("adsb.stack.models_dir"), "stack"))
    if cfg.mtx_enabled:
        out["mtx/reference"] = h(cfg.path("calibration.mtx.reference_csv"))
    if cfg["lanes.unmatched"] and cfg["modules.rome.enabled"] and cfg["modules.rome.e2"]:
        out["e2_rules"] = h(shipped.E2_RULES)
    return {k: v for k, v in out.items() if v is not None}


def _sha256_if_present(p) -> str | None:
    return A.sha256_file(p) if pathlib.Path(p).is_file() else None


def _weather(cfg):
    return F.stand_ab().load_weather(cfg.path("data.weather_dir")) if (cfg["lanes.matched"] and cfg["features.weather"]) \
        else None


def _orl(cfg):
    return F.read_orl_routes(cfg.path("features.orl_routes_file")) if (cfg["lanes.matched"] and
                                                                       cfg["features.orl_route_length"]) else None


# =================================================================================================================
# fit
# =================================================================================================================

#: fit_inputs keys the matched learners were fitted on (the free stages' inputs — stack models, MTX csv, E2 rules — change
#: with the reuse allow-list's knobs and are not compared)
#: what a fit reads to train the matched learners; adsb/serve.parquet is not one (fit reads only train.parquet, AFI)
_LEARNER_INPUTS = ("raw/", "weather/", "orl_routes", "adsb/train.parquet")


def _reuse_learners(donor: A.ModelArtifact, cols, enc_m, n_rows: int, n_es: int, need_cat: bool) -> tuple:
    """The donor's (base, catboost) for a derived fit, after checking that this fit's recomputed design columns,
    encodings and matched row counts are the donor's (so its learners are what this fit would train)."""
    if list(cols) != list(donor.design_columns):
        raise E.ReuseError(f"design columns differ from the donor {donor.key}'s")
    if donor.enc_matched is None or enc_m.to_json() != donor.enc_matched.to_json():
        raise E.ReuseError(f"the recomputed encodings differ from the donor {donor.key}'s (not the donor's training rows)")
    dm = donor.fit_report.get("matched", {})
    if (dm.get("n_rows"), dm.get("n_es")) != (n_rows, n_es):
        raise E.ReuseError(f"matched rows differ from the donor {donor.key}'s: {n_rows:,} rows / {n_es:,} early-stopping "
                           f"vs {dm.get('n_rows')} / {dm.get('n_es')}")
    if donor.base is None:
        raise E.ReuseError(f"the donor {donor.key} has no base learner")
    if need_cat and donor.catboost is None:
        raise E.ReuseError("the variant needs the CatBoost learner; the donor has none")
    return donor.base, (donor.catboost if need_cat else None)


def fit(config, *, save: bool = True, allow_heavy: bool = False, tier_overrides: dict | None = None, log=_log,
        overrides=(), split: dict | None = None, reuse_from=None) -> A.ModelArtifact:
    """`split` (validation protocols only): {"train_months": [...], "es_months": [...]} or {"train_months": [...],
    "es_days_from": d} (early stopping on the training rows whose UTC day of month >= d: the strict swap folds).

    `reuse_from` (a ModelArtifact or its directory): a derived fit. The donor's fitted matched learners are taken as they
    are — no LightGBM / CatBoost training — when every knob outside knobs.REUSE_FREE_KNOBS equals the donor's and the
    learners' input files are the same; frames, encodings, the unmatched lane and MTX are recomputed, and the recomputed
    design columns / encodings / row counts must equal the donor's. Anything else raises ReuseError. A derived artefact
    predicts what a full fit of the same config predicts (tests/fm1/test_fm1_reuse.py)."""
    cfg = as_config(config, overrides)
    donor = None
    if reuse_from is not None:
        if split or tier_overrides:
            raise E.ReuseError("reuse_from cannot be combined with a split / tier (a protocol fold or a screen tier "
                               "trains different models)")
        donor = reuse_from if isinstance(reuse_from, A.ModelArtifact) else A.load(reuse_from)
        blockers = K.reuse_blockers(cfg, donor.config)
        if blockers:
            raise E.ReuseError(f"knob(s) outside the reuse allow-list differ from the donor {donor.key}: {blockers}")
    stages = S.plan(cfg)
    names = {s.name for s in stages}
    split = dict(split or {})
    train_months = split.get("train_months", list(cfg["data.train_months"]))
    files = {m: month_file(cfg, m) for m in train_months}
    n_raw = sum(pq.ParquetFile(p).metadata.num_rows for p in files.values())
    if n_raw > HEAVY_ROWS and not allow_heavy:
        raise E.PipelineError(f"fit over {n_raw:,} raw training rows (> {HEAVY_ROWS:,}) is a heavy job: run the pre-run checks, "
                              "then pass allow_heavy=True / --allow-heavy (one heavy job at a time, AC power)")
    inputs = fit_inputs(cfg, files)
    if save:                                            # refused now, not after hours of fitting at the save
        A.refuse_existing(A.artifact_dir(cfg, inputs))
    if cfg["features.adsb"]:
        F.check_table_sources(cfg.path("adsb.table_dir"), cfg["adsb.sources.adsb_fi"])
    if donor is not None:
        diff = sorted(k for k in set(inputs) | set(donor.inputs)
                      if k.startswith(_LEARNER_INPUTS) and inputs.get(k) != donor.inputs.get(k))
        if diff:
            raise E.ReuseError(f"input data differ from the donor {donor.key}'s: {diff[:5]}"
                               + (f" (+{len(diff) - 5} more)" if len(diff) > 5 else ""))
    t0 = time.time()
    tier = dict(tier_overrides or {})
    seeds = tuple(tier.pop("seeds", cfg["run.seeds"]))
    v26 = "unmatched_body" in names and cfg["unmatched.learner"] == "v26"
    # the v26 lane first: its shipped loader (unm_congestion.load_real, all training files at once) peaks near 3.3 GB and
    # releases it before the matched frames are built
    lane = U.fit(cfg, files, seeds, log) if v26 else None
    need_frames = "base_model" in names or ("unmatched_body" in names and not v26)
    weather, orl = (_weather(cfg), _orl(cfg)) if need_frames else (None, None)
    m_parts, u_parts, dep_parts = [], [], []
    for month, p in files.items():
        raw = pd.read_parquet(p)
        # the organiser's monthly files hold all ten airports: keep the configured ones FIRST (departures from them,
        # arrivals at them), then hold the rows to the contract (FM1 notes N5)
        raw = raw[in_registry(raw, cfg)]
        row_rep = R.validate_raw(raw, "train", cfg["airports.codes"], where=p.name)
        if row_rep["n_labels_beyond_a_day"]:
            log(f"{p.name}: {row_rep['n_labels_beyond_a_day']} label(s) beyond {R.DAY_S} s (date-slip; kept, not refused)")
        if cfg.mtx_enabled:
            dep_parts.append(raw.loc[raw.PHASE_mvt.astype(str) == "DEP", ["ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]])
        if not need_frames:
            continue
        fr = F.build_frames(raw, serve=False, cfg=cfg, weather=weather, orl_routes=orl)
        if fr.matched is not None:
            m_parts.append(fr.matched.assign(file_month=month))
        if fr.unmatched is not None:
            u_parts.append(fr.unmatched.assign(file_month=month))
        log(f"features {p.name}: matched {0 if fr.matched is None else len(fr.matched):,}, "
            f"unmatched {0 if fr.unmatched is None else len(fr.unmatched):,}")
    es_months = set(split.get("es_months", cfg["matched.es_months"]))
    es_days_from = split.get("es_days_from")

    def es_mask(frame):
        if es_days_from is not None:
            return frame.day.astype(str).str[8:10].astype(int).to_numpy() >= int(es_days_from)
        return frame.file_month.isin(es_months).to_numpy()
    report = {"n_raw_rows": int(n_raw), "stages": [s.name for s in stages], "seeds": list(seeds), "tier": tier,
              "train_months": list(train_months),
              "early_stopping": {"days_from": es_days_from} if es_days_from is not None else {"months": sorted(es_months)}}
    base = cat = unm = enc_m = enc_u = None
    cols, ucols, gate_man, cov_share, feats = [], [], {}, {}, []
    if "base_model" in names:
        m = pd.concat(m_parts, ignore_index=True)
        del m_parts                                      # the per-month frames are a full second copy of m (2.1 GB)
        # the data's coverage before any dropout, for the support gate; the column exists only with the ADS-B block (BC-42)
        coverage_recorded = m.coverage.to_numpy(copy=True) if cfg["features.adsb"] else None
        fs, ds = cfg["adsb.train_dropout.flight"], cfg["adsb.train_dropout.day"]
        if fs > 0 or ds > 0:                             # fit only: the model learns to predict without the flight's track
            fl, nb = F.dropout_masks(m, fs, ds, seed=seeds[0])
            acols = F.adsb_columns(cfg.path("adsb.table_dir"))
            F.set_unheard(m, acols, fl, network=False)
            F.set_unheard(m, acols, nb, network=True)
            report["adsb_dropout"] = {"flight_share": fs, "day_share": ds, "n_flight": int(fl.sum()),
                                      "n_network": int(nb.sum())}
            log(f"ADS-B dropout: {int(fl.sum()):,} unheard flights, {int(nb.sum()):,} network-blackout rows")
        es = es_mask(m)
        target = m.delta.to_numpy(float) if cfg["matched.target"] == "delta" else m.y.to_numpy(float)
        enc_m = F.EncodingTables.fit(m, {"te": m.y.to_numpy(float), "de": m.delta.to_numpy(float)}, F.stand_ab().ENC_KEYS,
                                     F.stand_ab().SMOOTH)
        cols = F.matched_design_columns(cfg)
        need_cat = "cbv_blend" in names or ("adsb_stack" in names and cfg["adsb.stack.f_model"] == "lgbcat")
        if donor is None:
            X = F.matched_design(m, enc_m, cols)
            base = L.make_learner(cfg["matched.learner"], cfg, **_lr(tier, cfg["matched.learner"]))
            base.fit(X, target, ~es, es, seeds, cols)
            if need_cat:
                cat = L.make_learner("catboost", cfg, **_lr(tier, "catboost"))
                cat.fit(X, m.delta.to_numpy(float), ~es, es, seeds, cols)
            del X
        else:
            base, cat = _reuse_learners(donor, cols, enc_m, int(len(m)), int(es.sum()), need_cat)
            report["reused_learners_from"] = donor.key
        report["matched"] = {"n_rows": int(len(m)), "n_es": int(es.sum()), "base": base.describe()}
        if cat is not None:
            report["matched"]["catboost"] = cat.describe()
        if "adsb_stack" in names:
            gate_man = C.gate_manifest(cfg.path("adsb.stack.models_dir"))
            feats = C.stack_feats(cfg.path("adsb.stack.models_dir"))
            absent = [c for c in feats if c not in cols]
            if absent:
                raise E.SchemaError(f"the stack's features {absent} are not in the ADS-B block of {cfg['adsb.table_dir']}")
            cov_share = C.covered_share(m.ap.to_numpy(), coverage_recorded, cfg["adsb.min_coverage"])
        log(f"matched: {len(m):,} rows, base best_iter {base.best_iter} n_ref {base.n_ref}"
            + (f" (learners reused from {donor.key})" if donor is not None else ""))
    if v26:
        report["unmatched"] = {"learner": "v26", **lane.report}
    elif "unmatched_body" in names:
        u = pd.concat(u_parts, ignore_index=True)
        del u_parts
        if len(u) == 0:
            raise E.SchemaError("no unmatched training rows")
        enc_u = F.EncodingTables.fit(u, {"te": u.y.to_numpy(float)}, F.UNM_KEYS, cfg["unmatched.encoding_smooth"])
        ucols = F.unmatched_design_columns()
        Xu = F.unmatched_design(u, enc_u)
        es_u = es_mask(u)
        unm = L.make_learner("hgb", cfg)
        unm.fit(Xu, u.y.to_numpy(float), ~es_u, es_u, seeds, ucols)
        report["unmatched"] = {"n_rows": int(len(u)), "learner": unm.describe()}
    mtx = None
    if "mtx" in names:
        mtx = C.MtxCalibration().fit(pd.concat(dep_parts, ignore_index=True), cfg.path("calibration.mtx.reference_csv"))
        report["mtx"] = {"airports": sorted(mtx.params)}
    report["wall_s"] = round(time.time() - t0, 1)
    art = A.ModelArtifact(config=cfg, design_columns=cols, unmatched_columns=ucols, enc_matched=enc_m, enc_unmatched=enc_u,
                          base=base, catboost=cat, unmatched=unm,
                          stack_dir=cfg.path("adsb.stack.models_dir") if "adsb_stack" in names else None,
                          stack_feats=feats, gate_manifest=gate_man, train_covered_share=cov_share, mtx=mtx,
                          fit_report=report, inputs=inputs, unmatched_lane=lane)
    if save:
        A.save(art)
        log(f"artefact {art.directory}")
    return art


def _lr(tier: dict, learner: str) -> dict:
    lr = tier.get("learning_rate")
    return {} if lr is None or learner == "hgb" else {"learning_rate": lr}


# =================================================================================================================
# predict
# =================================================================================================================

def rows_digest(rows: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(rows, index=False).to_numpy().tobytes()).hexdigest()


def predict(art: A.ModelArtifact, rows: pd.DataFrame, log=_log) -> pd.DataFrame:
    cfg = art.config
    stages = S.plan(cfg)
    names = {s.name for s in stages}
    R.validate_raw(rows, "serve", cfg["airports.codes"])
    if cfg["features.adsb"]:
        F.check_table_sources(cfg.path("adsb.table_dir"), cfg["adsb.sources.adsb_fi"])
    given = rows                                        # the caller's rows: their digest identifies the input
    clipped = clip_rep = None
    if "eobt_clip" in names:                            # before the feature path: every stage sees the repaired field
        rows, clipped, clip_rep = EC.apply(cfg, rows)
    fr = F.build_frames(rows, serve=True, cfg=cfg, weather=_weather(cfg), orl_routes=_orl(cfg))
    dep_ids = fr.dep_ids
    v26 = "unmatched_body" in names and cfg["unmatched.learner"] == "v26"
    served = set()
    if fr.matched is not None:
        served |= set(fr.matched.MVT_ID_mvt.tolist())
    if fr.unmatched is not None:
        served |= set(fr.unmatched.MVT_ID_mvt.tolist())
    if v26:
        served |= set(U.unmatched_ids(rows).tolist())
    lost = np.setdiff1d(dep_ids, np.array(sorted(served), dtype="float64"))
    if len(lost):
        raise E.RoutingError(f"{len(lost):,} scored departure(s) are admitted by no lane (e.g. {lost[:3].tolist()}); "
                             "a matched row needs MVT, SCHED and AOBT_3; check lanes.* and the rows")
    ledger = S.Ledger(dep_ids)
    comp = pd.DataFrame({"MVT_ID_mvt": dep_ids}).set_index("MVT_ID_mvt")
    for c in ("proxy", "F", "base_value", "stack_F", "stack_resid", "catboost_C", "unmatched_pred"):
        comp[c] = np.nan
    comp["stack_eligible"] = False
    comp["lane"] = "unmatched"
    info: dict = {"n_adsb_missing": int(fr.n_adsb_missing)}
    if clipped is not None:
        comp["eobt_clipped"] = comp.index.isin(clipped)
        info["eobt_clip"] = clip_rep
    stage = {s.name: s for s in stages}
    if "base_model" in names and fr.matched is not None and len(fr.matched):
        m = fr.matched
        ids = m.MVT_ID_mvt.to_numpy(float)
        X = F.matched_design(m, art.enc_matched, art.design_columns)
        Fd = art.base.predict(X)
        proxy = m.proxy.to_numpy(float)
        bval = C.base_value(proxy, Fd, cfg["matched.target"])
        ledger.write(stage["base_model"], ids, bval, label=f"base_{art.base.name}")
        comp.loc[ids, "F"], comp.loc[ids, "base_value"], comp.loc[ids, "lane"] = Fd, bval, "matched"
        comp.loc[ids, "proxy"] = proxy
        elig = np.zeros(len(m), bool)
        c_all = None                                   # the CatBoost's delta on every matched row, when the blend needs it
        if "adsb_stack" in names:
            serve_share = C.covered_share(m.ap.to_numpy(), m.coverage.to_numpy(), cfg["adsb.min_coverage"])
            gate = C.effective_gate(cfg["adsb.support_gate.rule"], art.gate_manifest, art.train_covered_share,
                                    serve_share, cfg["adsb.support_gate.threshold"])
            if cfg["matched.target"] != "delta":
                raise E.ConfigValueError("the ADS-B stack corrects a delta-target F; matched.target must be delta")
            if cfg["adsb.stack.f_model"] == "lgbcat":
                if art.catboost is None:
                    raise E.ArtifactError("adsb.stack.f_model lgbcat but the artefact has no matched CatBoost")
                c_all = art.catboost.predict(X)
                comp.loc[ids, "catboost_C"] = c_all
            Fs = C.stack_base(Fd, c_all, cfg["adsb.stack.f_model"])
            vals, elig, resid, sinfo = C.stack_values(m, Fs, art.stack_feats, art.stack_dir, gate, cfg["adsb.min_coverage"])
            if elig.any():
                ledger.write(stage["adsb_stack"], ids[elig], vals[elig])
            comp.loc[ids, "stack_F"] = Fs
            comp.loc[ids, "stack_resid"], comp.loc[ids, "stack_eligible"] = resid, elig
            info.update(gate=gate, serve_covered_share=serve_share, stack=sinfo, stack_f_model=cfg["adsb.stack.f_model"])
        if "cbv_blend" in names and (~elig).any():
            Cd = c_all[~elig] if c_all is not None else art.catboost.predict(X[~elig])
            w = cfg["matched.blend.catboost_weight"]
            ledger.write(stage["cbv_blend"], ids[~elig], C.blend_value(proxy[~elig], Cd, bval[~elig], w))
            comp.loc[ids[~elig], "catboost_C"] = Cd
    if v26:
        if art.unmatched_lane is None:
            raise E.ArtifactError("the config serves the v26 unmatched lane but the artefact has none (fitted with another "
                                  "unmatched.learner?)")
        lane_comp, info["unmatched"], _ = U.serve(art.unmatched_lane, rows, cfg, ledger, stage, log)
        lids = lane_comp.index.to_numpy(dtype="float64")
        for c in lane_comp.columns:
            if c not in comp.columns:
                comp[c] = np.nan if c in U.COMP_FLOAT else pd.Series("", index=comp.index, dtype=object)
            comp.loc[lids, c] = lane_comp[c].to_numpy()
    elif "unmatched_body" in names and fr.unmatched is not None and len(fr.unmatched):
        u = fr.unmatched
        p = art.unmatched.predict(F.unmatched_design(u, art.enc_unmatched))
        uids = u.MVT_ID_mvt.to_numpy(float)
        ledger.write(stage["unmatched_body"], uids, np.rint(np.maximum(p, 1.0)))
        comp.loc[uids, "unmatched_pred"] = p
    if "ldi" in names:
        lc, info["ldi"] = LDI.apply(cfg, fr.matched, rows, ledger, stage["ldi"])
        comp["ldi_delta"], comp["ldi_band"] = np.nan, ""
        if len(lc):
            comp.loc[lc.index, "ldi_delta"] = lc.ldi_delta.to_numpy(dtype="float64")
            comp.loc[lc.index, "ldi_band"] = lc.ldi_band.to_numpy()
    if "mtx" in names:
        t = rows.set_index(rows.MVT_ID_mvt.to_numpy(float)).MVT_TIME_UTC_mvt.reindex(dep_ids)
        ap = rows.set_index(rows.MVT_ID_mvt.to_numpy(float)).ADEP_mvt.reindex(dep_ids).astype(str).to_numpy()
        frame = pd.DataFrame({"ap": ap, "month": t.dt.month.to_numpy(), "year": t.dt.year.to_numpy(), "pred": ledger.value})
        new, rep = art.mtx.apply(frame, cfg.path("calibration.mtx.published_csv"), cfg["calibration.mtx.lambda_rule"],
                                 cfg["calibration.mtx.mode"])
        ledger.calibrate(stage["mtx"], dep_ids, new)
        info["mtx"] = rep
    out = ledger.finalise()
    out = out.join(comp, on="MVT_ID_mvt")
    info.update(decided_by=ledger.counts(), n_rows=int(len(out)), artifact=art.key, rows_digest=rows_digest(given),
                stages=[s.name for s in stages])
    out.attrs["fm1_report"] = info
    log(f"predicted {len(out):,} departures: {info['decided_by']}")
    return out


# =================================================================================================================
# build-submission
# =================================================================================================================

def build_inputs(cfg, art: A.ModelArtifact) -> dict:
    """The inputs the fit recorded (fit_inputs: training files, weather, routes, ADS-B tables, stack models, MTX
    reference, E2 rules), hashed as they are at build time, plus the published MTX file when calibration is on; and
    every recorded input whose hash differs from the artefact's fit-time record (empty = the same files). A build does
    not need the fit-only inputs: a training month that is missing (or ambiguous: two files) and a missing MTX reference
    file are reported as changed, not refused. Runs before predict, so a broken raw directory stops a build at once."""
    present = {m: ps[0] for m, ps in files_by_month(cfg).items() if m in cfg["data.train_months"] and len(ps) == 1}
    now = fit_inputs(cfg, present, hash_file=_sha256_if_present)
    changed = sorted(k for k in set(now) | set(art.inputs) if now.get(k) != art.inputs.get(k))
    if cfg.mtx_enabled:
        now["mtx/published"] = A.sha256_file(cfg.path("calibration.mtx.published_csv"))
    return {"inputs": now, "changed_since_fit": changed}


def build_submission(art: A.ModelArtifact, out_dir, version: int | None = None, scored_file=None, template_file=None,
                     log=_log) -> dict:
    cfg = art.config
    scored_file = pathlib.Path(scored_file) if scored_file else cfg.path("data.scored_file")
    template_file = pathlib.Path(template_file) if template_file else cfg.path("data.template_file")
    for p in (scored_file, template_file):
        if "submissions" in p.resolve().parts:
            raise E.SpliceRefusedError(f"{p}: FM1 never reads a previous submission")
    out_dir = pathlib.Path(out_dir)
    name = f"{cfg['run.team']}_v{version}.parquet" if version is not None else "submission.parquet"
    target = out_dir / name
    prov = out_dir / (target.stem + ".provenance.parquet")
    man = out_dir / (target.stem + ".manifest.json")
    for p in (target, prov, man):
        if p.exists():
            raise E.ArtifactError(f"{p} exists; FM1 never overwrites an output")
    rows = pd.read_parquet(scored_file)
    template = pd.read_parquet(template_file)
    built_on = build_inputs(cfg, art)                  # before predict: a broken input directory stops the build at once
    if built_on["changed_since_fit"]:
        log(f"WARNING: {len(built_on['changed_since_fit'])} input(s) differ from the artefact's fit-time hashes "
            f"(recorded in the manifest): {built_on['changed_since_fit'][:5]}")
    pred = predict(art, rows, log=log)
    tids = template.MVT_ID_mvt.to_numpy(float)
    if not pd.Index(tids).is_unique or set(tids) != set(pred.MVT_ID_mvt):
        raise E.ValidationError(f"template ids ({len(tids):,}) and predicted departures ({len(pred):,}) differ")
    pred = pred.set_index("MVT_ID_mvt").reindex(tids).reset_index()
    sub = pd.DataFrame({"MVT_ID_mvt": tids, "TAXITIME_SEC_mvt": pred.TAXITIME_SEC_mvt.to_numpy().astype("int32")})
    if not ((sub.TAXITIME_SEC_mvt >= 1).all() and sub.MVT_ID_mvt.dtype == template.MVT_ID_mvt.dtype):
        raise E.ValidationError("submission values < 1 or id dtype differs from the template")
    out_dir.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(target, index=False)
    pred.to_parquet(prov, index=False)
    report = {**pred.attrs.get("fm1_report", {}), "submission": target.name, "sha256": A.sha256_file(target),
              "scored_file": {"path": str(scored_file), "sha256": A.sha256_file(scored_file)},
              "template_file": {"path": str(template_file), "sha256": A.sha256_file(template_file)},
              "artifact_dir": str(art.directory) if art.directory else None, "config_hash": cfg.hash,
              "inputs_at_build": built_on["inputs"], "inputs_changed_since_fit": built_on["changed_since_fit"]}
    man.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    log(f"wrote {target} ({len(sub):,} rows), provenance {prov.name}, manifest {man.name}")
    return {"submission": target, "provenance": prov, "manifest": man, "report": report}
