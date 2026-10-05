"""FM1's unmatched lane: the SHIPPED v36 unmatched chain, ported through the shipped modules' public functions
(plans/PREREG_fm1_2026_09_28.md Amendment A3).

Every estimator and rule is the shipped one, reached through prc.pipeline.fm1.shipped: the public composition helpers of
scripts/analysis/urp_replay.py (the tested replay of the shipped chain: serve parity 5,290 / 5,290 rows at all 13 versions,
plans/PREREG_urp_u_chain_replay_2026_09_24.md RESULT) and, through its `deps` namespace, build_submission, unm_congestion,
unm_congestion_ship, rome_fill, rome_dateslip, rome_localday, rome_local_day, congestion_anchor_ship, arr_row_clocks,
arc_ship_build, mau_build, mxi_build, mxj_build, rse_build, e2w_build, e2w_ship and the frozen agent_E_rules. Nothing is
copied; the few lines here only call those functions in the shipped order.

FIT (the training months) -> UnmatchedModel, the frames the shipped estimators fit on:
    unm     unmatched admissible departures (derive() columns + the airport-hour witness)    training_frames (BC-31)
    lirf    LIRF's matched admissible departures (S1's LIRF logistic, R2's fill classifier)  training_frames (BC-31)
    extra   MAU's extra matched rows (1 in 20, non-fill, leave-one-out witness)             mau_build.extra_rows
    beta_w  E2W's coefficients: stake-weighted Brier on LIRF's band rows, their p_R / nf nested leave-one-month-out
            (urp_replay.nested_lirf_inputs -> e2w_training -> fit_e2w_beta; cls = csf2_build.labels)
    The shipped S1 cells (build_submission.fit_unmatched) and R2 (rome_fill.fit_predict_fill) fit AND predict in one call,
    so the artefact stores the frames and serve() runs every fit, seeded: bit-reproducible (AT3 / AT4).

SERVE (the scored rows) -> one ledger write per stage, in the fixed order (stages.STAGE_ORDER):
    unmatched_body  every unmatched departure: rint(mixture(p, sp, not_fill_value(nf_cells, MJS body, tail_cells)))
                    tail_cells = modules.tail_cells.enabled (true: nf_hybrid; false: MJS body) IN THIS MIXTURE;  v26 without its anchor
                    a LIRF row is always decided below by `rome`, from S1's own hybrid, never changed by this knob
    rome            LIRF: R2 p -> mixture(p, sp, S1's nf) -> 24 h date-slip -> rint -> E1 floor -> RLD        v7 -> v10 -> v12
    arc             modules.arc.airports: ord -> max(rint(MVT - AOBT_3), 1); fill -> max(rint(sp), 1)        v17
    rome_e2         LIRF rows still at their Rome value, 3,600 <= sp < 24,000: E2 logit, per-day check       v31
    rome_e2w        the rows E2 moved: E2's form with beta_W, per-day check                                  v33
    storm           rows the body still decides: rint(v + max(A - 300, 0))                                   v16 inside v26
ARN (v21) is DROPPED (FMA §4.2): its 8 scored rows keep their Rome / E2 values (measured: no other row moves).

Set-level rules: the witness (airport-hour), the storm anomaly (airport-day / airport-month), ARC's ARR join (the flight's
own arrival, at any registry airport) and E2's per-day checks read OTHER scored rows, so serve the whole scored file; E2 / E2W
also read LIRF arrival in-blocks from data.raw_dir themselves (check_e2_context guards which file that is).
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import errors as E
from . import shipped

ROME = "LIRF"
FRAMES = ("unm", "lirf", "extra")
#: derive()'s sp bins are categoricals; the artefact stores the frames without them and re-attaches them with the shipped
#: rome_fill.with_sp_bins on load, exactly as rome_fill.load_frames treats its data/cache_rome cache
SP_BINS = ("sp_band", "sp_fine")
#: DEP label columns never enter the lane's scored frame (urp_replay.replay_serve drops them the same way)
LABELS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "y", "delta")
COMP_FLOAT = ("unmatched_pred", "unm_p_fill", "unm_nf_cells", "unm_body", "rome_p_R", "rome_v7", "rome_v10", "rome_value",
              "arc_value", "e2_p", "e2w_p", "storm_A", "storm_shift")
COMP_TEXT = ("rome_rule", "arc_rule")


def deps(raw_dir) -> SimpleNamespace:
    """The shipped chain's namespace (shipped.unmatched_deps) with E2's two context readers bound to `raw_dir` (they read
    LIRF arrival in-blocks from disk themselves; the rule file's default is the repo's data/raw)."""
    D = shipped.unmatched_deps()
    rd = str(raw_dir)
    return SimpleNamespace(**{**vars(D), "rule_E2": partial(D.E.rule_E2, raw_dir=rd),
                              "stand_inbound": partial(D.E.stand_inbound_features, raw_dir=rd)})


def body_params(cfg) -> dict:
    """The MJS body's HistGradientBoosting parameters from the unmatched.hgb.* knobs (defaults = mjs_build.ARMS["arm"])."""
    return {"max_leaf_nodes": cfg["unmatched.hgb.max_leaf_nodes"], "max_iter": cfg["unmatched.hgb.max_iter"],
            "min_samples_leaf": cfg["unmatched.hgb.min_samples_leaf"],
            "n_iter_no_change": cfg["unmatched.hgb.n_iter_no_change"], "learning_rate": cfg["unmatched.hgb.learning_rate"]}


# =================================================================================================================
# the fitted lane
# =================================================================================================================

@dataclass
class UnmatchedModel:
    unm: pd.DataFrame
    lirf: pd.DataFrame
    extra: pd.DataFrame
    train_months: tuple
    seeds: tuple
    beta_w: np.ndarray | None
    report: dict = field(default_factory=dict)

    def save(self, directory) -> list:
        d = pathlib.Path(directory)
        d.mkdir(parents=True)
        names = []
        for name in FRAMES:
            f = getattr(self, name)
            f.drop(columns=[c for c in SP_BINS if c in f.columns]).to_parquet(d / f"{name}.parquet", index=False)
            names.append(f"{name}.parquet")
        (d / "lane.json").write_text(json.dumps(
            {"train_months": [int(m) for m in self.train_months], "seeds": [int(s) for s in self.seeds],
             "beta_w": None if self.beta_w is None else [float(x) for x in self.beta_w], "report": self.report},
            indent=1, sort_keys=True, default=str))
        return names + ["lane.json"]

    @classmethod
    def load(cls, directory) -> "UnmatchedModel":
        """The artefact loader has verified every file's sha256 before this reads them (artifact.load)."""
        D = shipped.unmatched_deps()
        d = pathlib.Path(directory)
        meta = json.loads((d / "lane.json").read_text())
        return cls(unm=D.with_sp_bins(pd.read_parquet(d / "unm.parquet")),
                   lirf=D.with_sp_bins(pd.read_parquet(d / "lirf.parquet")),
                   extra=pd.read_parquet(d / "extra.parquet"), train_months=tuple(meta["train_months"]),
                   seeds=tuple(meta["seeds"]), beta_w=None if meta["beta_w"] is None else np.asarray(meta["beta_w"], float),
                   report=meta["report"])


def training_frames(D, files: dict) -> tuple:
    """(unm, lirf, summary): unm_congestion.load_real's recipe WITHOUT its first-fold self-check (BC-31). The files are read
    together (build_submission.load_movements -> admissible: ONE sort and de-duplication over all of them, as every shipped
    body saw them) -> derive; y = TAXITIME, delta = BLOCK - AOBT_3; the airport-hour witness of the MATCHED admissible
    departures (unm_congestion.airport_hour_witness) attached to the unmatched rows; LIRF's matched rows kept for S1's LIRF
    logistic and R2. load_real additionally fits bs.fit_unmatched twice on "the months after the first" to show that LIRF's
    matched rows are the only matched rows it reads; with ONE training month that fold is empty and it raises, which refused
    every single-month fit (the strict swap folds). Equality with load_real's frames on two or more months is pinned by
    tests/fm1/test_fm1_unmatched.py::test_lane_frames_equal_the_shipped_loader and by the 12-month cache replay."""
    f = D.bs.derive(D.bs.admissible(D.bs.load_movements([str(files[m]) for m in sorted(files)])))
    f["y"] = f.TAXITIME_SEC_mvt.astype("float64")
    f["delta"] = (f.BLOCK_TIME_UTC_mvt - f.AOBT_3_flt).dt.total_seconds()
    wit = D.uc.airport_hour_witness(f[~f.unmatched])
    unm = D.uc.attach_witness(f[f.unmatched], wit).reset_index(drop=True)
    lirf = f[(~f.unmatched) & (f.ADEP_mvt == ROME)].reset_index(drop=True)
    summary = {"n_airport_hours": int(len(wit)), "n_matched_rows": int((~f.unmatched).sum()),
               "unmatched_nan_share": float(unm.hprox_med.isna().mean()) if len(unm) else None}
    return unm, lirf, summary


def fit(cfg, files: dict, seeds, log=print) -> UnmatchedModel:
    """files: {month: path} of the training months (one or more). The frames are training_frames' (the shipped loader's
    recipe), kept to the airport registry; beta_W only when E2W will serve."""
    t0 = time.time()
    D = deps(cfg.path("data.raw_dir"))
    months = tuple(sorted(int(m) for m in files))
    seeds = tuple(int(s) for s in seeds)
    codes = set(cfg["airports.codes"])
    unm, lirf, summary = training_frames(D, files)
    if not set(unm.ADEP_mvt.astype(str).unique()) <= codes:
        unm = unm[unm.ADEP_mvt.astype(str).isin(codes)].reset_index(drop=True)
    if ROME not in codes:
        lirf = lirf.iloc[:0]
    if len(unm) == 0:
        raise E.SchemaError("no unmatched training rows at the registry airports")
    extra = pd.concat([D.MB.extra_rows(pd.read_parquet(files[m], columns=D.MB.RAW_COLS)) for m in months],
                      ignore_index=True)
    if not set(extra.ADEP_mvt.astype(str).unique()) <= codes:
        extra = extra[extra.ADEP_mvt.astype(str).isin(codes)].reset_index(drop=True)
    if len(extra) == 0:
        raise E.SchemaError("no MAU extra matched rows (mau_build.extra_rows): the v26 body weights them 0.25 * n_fit / "
                            "n_extra")
    if not extra.MVT_ID_mvt.is_unique or extra.MVT_ID_mvt.isin(unm.MVT_ID_mvt).any():
        raise E.SchemaError("MAU extra rows are not unique or overlap the unmatched rows (mjs_build.rebuild's guard)")
    beta_w, e2w = None, {}
    if cfg["modules.rome.enabled"] and cfg["modules.rome.e2"] and cfg["modules.rome.e2w"]:
        beta_w, e2w = fit_beta_w(D, unm, lirf, months, seeds, log)
    report = {"n_unm": int(len(unm)), "n_lirf_matched": int(len(lirf)), "n_extra": int(len(extra)),
              "train_months": list(months), "seeds": list(seeds), "witness": summary, "e2w": e2w,
              "wall_s": round(time.time() - t0, 1)}
    log(f"unmatched lane: {len(unm):,} unmatched, {len(lirf):,} LIRF matched, {len(extra):,} extra rows; "
        f"beta_W {None if beta_w is None else [round(float(b), 6) for b in beta_w]} ({report['wall_s']} s)")
    return UnmatchedModel(unm=unm, lirf=lirf, extra=extra, train_months=months, seeds=seeds, beta_w=beta_w, report=report)


def fit_beta_w(D, unm: pd.DataFrame, lirf: pd.DataFrame, months: tuple, seeds: tuple, log=print) -> tuple:
    """(beta_W, report): e2w_build.fit_brier_w on the training months' LIRF unmatched rows, as e2w_ship built v33's
    coefficients from the CSF.2 table: p_R and the non-fill term of each training month from fits on the OTHER training
    months (urp_replay.nested_lirf_inputs), the fill / slip / ordinary class from the label (csf2_build.labels)."""
    months_with_rows = sorted(set(unm.month[unm.ADEP_mvt.astype(str) == ROME].astype(int)) & set(months))
    if len(months_with_rows) < 2:
        raise E.SchemaError(f"E2W: beta_W's nested leave-one-month-out needs LIRF unmatched rows in >= 2 training months, "
                            f"got {months_with_rows}; switch modules.rome.e2w off for this fit")
    t0 = time.time()
    U = shipped.urp_replay()
    rows = unm[(unm.ADEP_mvt.astype(str) == ROME) & unm.month.isin(months)].reset_index(drop=True)
    nest = U.nested_lirf_inputs(D, SimpleNamespace(unm=unm, lirf=lirf), months, (), seeds)
    nest = nest.set_index("MVT_ID_mvt").reindex(rows.MVT_ID_mvt.to_numpy(dtype="float64"))
    cls = shipped.csf2_build().labels(rows).cls.to_numpy()
    table = U.e2w_training(D, rows, nest.p_R.to_numpy(dtype="float64"), nest.nf.to_numpy(dtype="float64"), cls)
    beta = np.asarray(U.fit_e2w_beta(D, table, ()), dtype="float64")
    rep = {"n_rows": int(len(rows)), "n_fit_rows": int(table.fit.sum()), "classes": pd.Series(cls).value_counts().to_dict(),
           "beta_w": beta.tolist(), "wall_s": round(time.time() - t0, 1)}
    log(f"E2W beta_W from {rep['n_fit_rows']} nested-LOMO band rows ({rep['wall_s']} s)")
    return beta, rep


# =================================================================================================================
# pure helpers
# =================================================================================================================

def unmatched_ids(rows: pd.DataFrame) -> np.ndarray:
    """The departures the lane serves: DEP rows without AOBT_3, in row order."""
    dep = rows.PHASE_mvt.astype(str).to_numpy() == "DEP"
    return rows.MVT_ID_mvt.to_numpy(dtype="float64")[dep & rows.AOBT_3_flt.isna().to_numpy()]


def storm_values(value, anomaly, body_rows) -> tuple:
    """(new values, moved mask): congestion_anchor_ship.anchored_values = rint(v + max(A - 300, 0)) on the rows the
    unmatched body decided (the shipped scope: v26 = anchor(rint(body mixture)) on the rows no Rome / ARC rule owns);
    every other row unchanged; a NaN A shifts nothing."""
    v = np.asarray(value, dtype="float64")
    A = np.asarray(anomaly, dtype="float64")
    body = np.asarray(body_rows, dtype=bool)
    if not (len(v) == len(A) == len(body)):
        raise ValueError(f"{len(v)} values, {len(A)} anomalies, {len(body)} body flags")
    new = v.copy()
    if body.any():
        new[body] = shipped.unmatched_deps().anchored_values(v[body], A[body])
    return new, new != v


def rome_tags(mix, v7, v10, v12) -> np.ndarray:
    """Per LIRF row, the Rome rules that moved it: R2 always; +DS (date-slip moved rint(mixture)), +E1 (the floor moved
    v7), +RLD (the local-day rule moved v10)."""
    mix, v7, v10, v12 = (np.asarray(a, dtype="float64") for a in (mix, v7, v10, v12))
    tag = np.full(len(v7), "R2", dtype=object)
    return tag + np.where(v7 != np.rint(mix), "+DS", "") + np.where(v10 != v7, "+E1", "") + np.where(v12 != v10, "+RLD", "")


def check_e2_context(cfg, sched) -> None:
    """E2 / E2W read LIRF arrival in-blocks from data.raw_dir themselves (agent_E_rules._files_for: the training file of
    the row's scheduled month; for 2026 `ranking.parquet`). Refuse a configuration where that file is not the scored rows'
    context: missing, or (2026) not the configured data.scored_file."""
    raw = cfg.path("data.raw_dir")
    t = pd.to_datetime(pd.Series(sched), utc=True).dropna()
    for y, m in sorted(set(zip(t.dt.year.astype(int), t.dt.month.astype(int)))):
        if y >= 2026:
            f = raw / "ranking.parquet"
            if not f.exists():
                raise E.SchemaError(f"E2 context: {f} does not exist; agent_E_rules reads the {y}-{m:02d} LIRF arrivals "
                                    "from it")
            if f.resolve() != cfg.path("data.scored_file").resolve():
                raise E.SchemaError(f"E2 context: {f}, the file agent_E_rules reads for {y}-{m:02d}, is not the configured "
                                    f"data.scored_file {cfg.path('data.scored_file')}")
        elif not sorted(raw.glob(f"training_{y:04d}-{m:02d}-01_*.parquet")):
            raise E.SchemaError(f"E2 context: no training file for {y:04d}-{m:02d} in {raw} (agent_E_rules reads the "
                                "row month's LIRF arrivals from it)")


# =================================================================================================================
# serve
# =================================================================================================================

def _empty_comp(ids) -> pd.DataFrame:
    comp = pd.DataFrame(index=pd.Index(np.asarray(ids, dtype="float64"), name="MVT_ID_mvt"))
    for c in COMP_FLOAT:
        comp[c] = np.nan
    for c in COMP_TEXT:
        comp[c] = pd.Series("", index=comp.index, dtype=object)
    return comp


def not_fill_value(D, nf_cells, body, tail_cells: bool) -> np.ndarray:
    """The unmatched body's not-copy value IN THE BODY MIXTURE (modules.tail_cells.enabled;
    plans/PREREG_tc_tail_cell_routing_2026_10_05.md): `D.nf_hybrid(nf_cells, body)` -- the shipped hybrid, which
    keeps a row's own cell mean where it is at or above build_submission.T_TAIL_S -- when `tail_cells` is true
    (the shipped rule, default); `body`, as float64, when it is false. Either way this is only the body's own
    mixture input: a LIRF row is decided by the `rome` stage from S1's own hybrid (parts["nf"], never nf_cells)
    regardless of `tail_cells`, so this function's result never reaches a LIRF row's final value. The false
    branch runs the SAME validation nf_hybrid runs (equal shapes, all finite, the same error type and wording
    style) so turning the hybrid off never skips the check it would otherwise have run; nf_hybrid's own refusal
    fires unchanged when `tail_cells` is true."""
    if tail_cells:
        return D.nf_hybrid(nf_cells, body)
    cells = np.asarray(nf_cells, dtype="float64")
    b = np.asarray(body, dtype="float64")
    if cells.shape != b.shape:
        raise ValueError(f"shape mismatch: nf_cells {cells.shape} vs body {b.shape}")
    if not (np.isfinite(cells).all() and np.isfinite(b).all()):
        raise ValueError("nf_cells and body must be finite")
    return b


def body_values(D, model: UnmatchedModel, te_w: pd.DataFrame, parts: dict, params: dict, *,
                tail_cells: bool) -> dict:
    """The v26 body on `te_w` (scored rows with the serve witness): per seed, unm_congestion's congestion regressor on the
    non-fill training rows gives the design; mxj_build.fit_body fits the HistGradientBoosting body on its fit split plus
    MAU's extra rows at 25 % mass, early-stopping on mau_build.stopping_split's validation split; mean over seeds;
    mixture(p, sp, not_fill_value(cells, body, tail_cells)) (modules.tail_cells.enabled). The construction of
    mjs_build.rebuild:93-113 / urp_replay.other_bodies."""
    nonfill = model.unm[~D.schedule_fill(model.unm)]
    ex = model.extra
    by_seed, n_iter = {}, {}
    for s in model.seeds:
        ctrl = D.fit_congestion_regressor(nonfill, s)
        x = ctrl.design(nonfill)
        yy = np.clip(nonfill.y.to_numpy(dtype="float64"), -D.bs.WINSOR_S, D.bs.WINSOR_S)
        fi, va = D.stopping_split(len(nonfill), s)
        body = D.fit_body(x.iloc[fi], yy[fi], ctrl.design(ex), ex.y.to_numpy(dtype="float64"), s, x.iloc[va], yy[va],
                          params)
        by_seed[s] = body.predict(ctrl.design(te_w))
        n_iter[int(s)] = int(body.n_iter_)
    mean = D.mean_over_seeds(by_seed)
    mix = D.mixture(parts["p"], parts["sp"], not_fill_value(D, parts["nf_cells"], mean, tail_cells))
    return {"mix": np.asarray(mix, dtype="float64"), "body": np.asarray(mean, dtype="float64"), "n_iter": n_iter}


def serve(model: UnmatchedModel, rows: pd.DataFrame, cfg, ledger, stages: dict, log=print) -> tuple:
    """Run the enabled lane stages (`stages`: name -> StageSpec, the plan's) on the unmatched departures of `rows` (the
    raw scored rows, DEP + ARR), writing each into `ledger`. Returns (component frame indexed by MVT_ID_mvt, info,
    {stage: values of every lane row after that stage})."""
    t0 = time.time()
    ids = unmatched_ids(rows)
    comp, info, snap = _empty_comp(ids), {"n_rows": int(len(ids))}, {}
    if len(ids) == 0:
        return comp, info, snap          # BC-30: the shipped string-key builders never see a zero-row frame
    D, U = deps(cfg.path("data.raw_dir")), shipped.urp_replay()

    def snapshot(name):
        snap[name] = pd.Series(ledger.values_of(ids), index=ids)

    derived = D.bs.derive(rows.loc[rows.PHASE_mvt.astype(str).to_numpy() == "DEP", list(D.bs.COLUMNS)]
                          .reset_index(drop=True))
    te = derived[derived.unmatched.to_numpy()].drop(columns=[c for c in LABELS if c in derived.columns]) \
        .reset_index(drop=True)
    if not np.array_equal(te.MVT_ID_mvt.to_numpy(dtype="float64"), ids):
        raise E.RoutingError("the lane's derived rows are not the unmatched departures in row order")
    seeds = model.seeds
    _, parts = U.s1_parts(D, model.unm, model.lirf, te, seeds, ())
    # ---- unmatched_body (v26 without its storm anchor)
    te_w, wsum = D.ucs.attach_serve_witness(te, derived[list(D.ucs.WITNESS_COLUMNS)])
    tail_cells = cfg["modules.tail_cells.enabled"]
    b = body_values(D, model, te_w, parts, body_params(cfg), tail_cells=tail_cells)
    ledger.write(stages["unmatched_body"], ids, np.rint(b["mix"]))
    comp["unmatched_pred"], comp["unm_body"] = b["mix"], b["body"]
    comp["unm_p_fill"], comp["unm_nf_cells"] = np.asarray(parts["p"], float), np.asarray(parts["nf_cells"], float)
    tail_ids = ids[np.asarray(parts["nf_cells"], dtype="float64") >= D.bs.T_TAIL_S]
    n_tail_rows = int(D.bs.routing_counts(parts["nf_cells"])["n_nf_cells"])
    info.update(witness=wsum, body_n_iter=b["n_iter"],
               tail_cells={"enabled": bool(tail_cells), "n_tail_rows": n_tail_rows})
    snapshot("unmatched_body")
    # ---- the LIRF / Rome module, R2 -> date-slip -> E1 -> RLD
    L = (te.ADEP_mvt.astype(str) == ROME).to_numpy()
    te_l, ids_l = te[L].reset_index(drop=True), ids[L]
    nf_l = np.asarray(parts["nf"], dtype="float64")[L]
    sp_l = te_l.sp.to_numpy(dtype="float64")
    rome = "rome" in stages and L.any()
    if rome:
        p_r = U.fill_p(D, model.unm, model.lirf, te_l, model.train_months, seeds, ())
        lr = U.lirf_rules(D, model.unm, p_r, te_l, nf_l, ())
        tags = rome_tags(D.mixture(p_r, sp_l, nf_l), lr["v7"], lr["v10"], lr["v12"])
        ledger.write(stages["rome"], ids_l, lr["v12"])
        comp.loc[ids_l, "rome_p_R"], comp.loc[ids_l, "rome_v7"] = p_r, lr["v7"]
        comp.loc[ids_l, "rome_v10"], comp.loc[ids_l, "rome_value"] = lr["v10"], lr["v12"]
        comp.loc[ids_l, "rome_rule"] = tags
        info["rome"] = {"n_rows": int(L.sum()), "rules": pd.Series(tags).value_counts().to_dict(),
                        "dateslip_q": lr["q"], "dateslip_m": lr["m"]}
        snapshot("rome")
    # ---- ARC
    if "arc" in stages:
        at = np.isin(te.ADEP_mvt.astype(str).to_numpy(), list(cfg["modules.arc.airports"]))
        rule = np.array([], dtype=object)
        if at.any():
            rule, ai, av = U.arc_rows(D, rows[rows.PHASE_mvt.astype(str).to_numpy() == "DEP"], rows, ids[at])
            comp.loc[ids[at], "arc_rule"] = np.asarray(rule, dtype=object)
            if len(ai):
                ledger.write(stages["arc"], ai, av)
                comp.loc[ai, "arc_value"] = av
        info["arc"] = {k: int((np.asarray(rule) == k).sum()) for k in ("ord", "fill", "none", "")}
        snapshot("arc")
    # ---- E2 (RSE) and E2W, on the LIRF rows, after ARC
    if rome and "rome_e2" in stages:
        check_e2_context(cfg, te_l.SCHED_TIME_UTC_mvt)
        rows_e = te_l.assign(p_R=p_r)
        p_e2 = np.asarray(D.rule_E2(rows_e), dtype="float64")
        v_now = ledger.values_of(ids_l)
        day = pd.to_datetime(te_l.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").to_numpy()
        v31, S_e2 = U.e2_apply(D, lr["v12"], v_now, p_e2, sp_l, nf_l, day, cfg["modules.rome.e2_day_p99"])
        if S_e2.any():
            ledger.write(stages["rome_e2"], ids_l[S_e2], v31[S_e2])
        comp.loc[ids_l, "e2_p"] = p_e2
        info["rome_e2"] = {"n_fired": int(np.isfinite(p_e2).sum()), "n_rows": int(S_e2.sum())}
        snapshot("rome_e2")
        if "rome_e2w" in stages:
            if model.beta_w is None:
                raise E.ArtifactError("modules.rome.e2w is on but the artefact's unmatched lane was fitted without beta_W")
            f = D.stand_inbound(rows_e)
            p_w = np.asarray(D.e2_form(p_r, f.k_in, f.turn, sp_l, model.beta_w), dtype="float64")
            v33, S_w = U.e2w_apply(D, v_now, v31, p_w, sp_l, nf_l, day, cfg["modules.rome.e2w_day_p99"])
            if S_w.any():
                ledger.write(stages["rome_e2w"], ids_l[S_w], v33[S_w])
            comp.loc[ids_l, "e2w_p"] = p_w
            info["rome_e2w"] = {"n_rows": int(S_w.sum())}
            snapshot("rome_e2w")
    # ---- the storm anchor, on the rows the body still decides
    if "storm" in stages:
        A = np.asarray(U.attach_anomaly(D, te, [rows]), dtype="float64")
        before = ledger.values_of(ids)
        new, moved = storm_values(before, A, ledger.decided_of(ids) == "unmatched_body")
        if moved.any():
            ledger.write(stages["storm"], ids[moved], new[moved])
        comp["storm_A"], comp["storm_shift"] = A, new - before
        info["storm"] = {"n_moved": int(moved.sum()), "n_anomaly": int(np.isfinite(A).sum())}
        snapshot("storm")
    # how many of the tail-cell rows (info["tail_cells"]["n_tail_rows"]) are still decided unmatched_body/storm
    # once every later stage has run -- the rows this knob can actually change in the output (a LIRF row is
    # always decided by `rome` above instead, from S1's own hybrid, never by this knob; same decided-by accessor
    # prc/pipeline/fm1/uad.py's `serve` uses for its own SCOPE check).
    decided_tail = ledger.decided_of(tail_ids)
    info["tail_cells"]["n_tail_rows_decided"] = int(np.isin(decided_tail, np.array(("unmatched_body", "storm"),
                                                                                   dtype=object)).sum())
    info["wall_s"] = round(time.time() - t0, 1)
    log(f"unmatched lane: {len(ids):,} rows ({int(L.sum())} LIRF), stages {sorted(snap)} ({info['wall_s']} s)")
    return comp, info, snap
