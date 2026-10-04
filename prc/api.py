"""FM1's public API (plans/PREREG_fm1_2026_09_28.md §1). Everything a user or a script needs; nothing else is public.

    fit(config, *, save=True, allow_heavy=False, overrides=(), reuse_from=None) -> ModelArtifact
                                                  reuse_from: a donor artefact (or directory) whose fitted matched
                                                  learners are reused when only allow-listed knobs differ
    predict(config | artifact, rows) -> DataFrame[MVT_ID_mvt, TAXITIME_SEC_mvt, decided_by, lane, value_before_calibration,
                                                  calibration_shift, F, base_value, stack_resid, stack_eligible,
                                                  catboost_C, unmatched_pred]      (.attrs["fm1_report"]: the run report)
    validate(config, protocol, out_dir=None, allow_heavy=False, save_models=False) -> report
                                                  protocol: S | swap | forward | lomo | forward_monthly | dry-run
                                                  save_models: also save each fold's fit as an artefact under
                                                  out_dir/models/<fold name>/ (needs out_dir; never overwritten)
    build_submission(config | artifact, out_dir, version=None) -> {submission, provenance, manifest, report}
    compare(oof_a, oof_b) -> paired gain of b over a with a day-block bootstrap CI (each side: one record or a list)
    blend_oof(oof_a, oof_b, w) -> the offline blend record (selection arm A2)
    load_artifact(directory) -> ModelArtifact                                    (re-hashes every file first)

`config` is a path to a YAML (configs/fm1.yaml) or an FM1Config; `overrides` are `path=value` strings (--set).
"""
from __future__ import annotations

import pathlib

import pandas as pd

from prc.pipeline.fm1 import artifact as _A
from prc.pipeline.fm1 import engine as _G
from prc.pipeline.fm1 import errors
from prc.pipeline.fm1 import knobs as _K
from prc.pipeline.fm1 import protocols as _P
from prc.pipeline.fm1 import stages as _S
from prc.pipeline.fm1.artifact import ModelArtifact
from prc.pipeline.fm1.knobs import FM1Config

__all__ = ["fit", "predict", "validate", "build_submission", "compare", "blend_oof", "load_artifact", "load_config", "ModelArtifact",
           "FM1Config", "errors"]


def load_config(path, overrides=()) -> FM1Config:
    return _K.load_config(path, overrides=overrides)


def load_artifact(directory) -> ModelArtifact:
    return _A.load(directory)


def fit(config, *, save: bool = True, allow_heavy: bool = False, overrides=(), log=_G._log,
        reuse_from=None) -> ModelArtifact:
    """`reuse_from` (an artefact or its directory): a derived fit that takes the donor's fitted matched learners when only
    predict-time / light-fit knobs differ (knobs.REUSE_FREE_KNOBS); see engine.fit."""
    return _G.fit(config, save=save, allow_heavy=allow_heavy, overrides=overrides, log=log, reuse_from=reuse_from)


def find_artifact(cfg: FM1Config) -> ModelArtifact:
    """The one saved artefact of this resolved config under run.artifacts_dir (refuses none or several)."""
    parent = cfg.path("run.artifacts_dir")
    hits = sorted(parent.glob(f"{cfg.short_hash}-*")) if parent.exists() else []
    hits = [h for h in hits if h.is_dir() and not h.name.startswith(".")]
    if not hits:
        raise errors.ArtifactError(f"no artefact for config {cfg.short_hash} under {parent}; run `fit` first")
    if len(hits) > 1:
        raise errors.ArtifactError(f"{len(hits)} artefacts for config {cfg.short_hash} under {parent} (different data); "
                                   "pass the artefact directory explicitly")
    return _A.load(hits[0])


def _artifact(config_or_artifact, overrides=()) -> ModelArtifact:
    is_artifact = isinstance(config_or_artifact, ModelArtifact) or (
        isinstance(config_or_artifact, (str, pathlib.Path)) and pathlib.Path(config_or_artifact).is_dir())
    if is_artifact and list(overrides):
        raise errors.ConfigValueError(f"overrides do not apply to an artefact (it carries its own configuration): "
                                      f"{list(overrides)}; pass a config instead, or fit a derived artefact")
    if isinstance(config_or_artifact, ModelArtifact):
        return config_or_artifact
    if is_artifact:
        return _A.load(config_or_artifact)
    return find_artifact(_G.as_config(config_or_artifact, overrides))


def predict(config_or_artifact, rows: pd.DataFrame, overrides=(), log=_G._log) -> pd.DataFrame:
    return _G.predict(_artifact(config_or_artifact, overrides), rows, log=log)


def build_submission(config_or_artifact, out_dir, version: int | None = None, overrides=(), log=_G._log) -> dict:
    return _G.build_submission(_artifact(config_or_artifact, overrides), out_dir, version=version, log=log)


def validate(config, protocol: str, out_dir=None, allow_heavy: bool = False, overrides=(), log=_G._log,
             save_models: bool = False) -> dict:
    cfg = _G.as_config(config, overrides)
    if protocol == "dry-run":
        return dry_run(cfg)
    return _P.run(cfg, protocol, out_dir=out_dir, allow_heavy=allow_heavy, log=log, save_models=save_models)


def dry_run(cfg: FM1Config) -> dict:
    """Everything that can be checked without fitting: the config, the stage plan (or what blocks it), every input path."""
    report = {"config_hash": cfg.hash, "differs_from_defaults": _K.differs_from_defaults(cfg),
              "not_ported": cfg.not_ported(), "mtx_enabled": cfg.mtx_enabled}
    try:
        report["stages"] = [s.name for s in _S.plan(cfg)]
    except errors.ComponentNotPortedError as exc:
        report["stages"] = None
        report["plan_refused"] = str(exc)
    paths = {}
    for k in _K.SCHEMA:
        if k.kind == "path" and k.path != "run.artifacts_dir":
            paths[k.path] = {"path": str(cfg.path(k.path)), "exists": cfg.path(k.path).exists()}
    report["paths"] = paths
    try:
        report["training_months"] = {m: p.name for m, p in _G.training_files(cfg).items()}
    except errors.SchemaError as exc:
        report["training_months"] = f"REFUSED: {exc}"
    return report


def _records(x) -> pd.DataFrame:
    xs = x if isinstance(x, (list, tuple)) else [x]
    return _P.pool([f if isinstance(f, pd.DataFrame) else pd.read_parquet(f) for f in xs])


def compare(oof_a, oof_b, n_boot: int = 2000, seed: int = 20260928) -> dict:
    """Paired gain of b over a. Each side is one OOF record or a list of them (pooled: F1 + F2 is the primary pool)."""
    return _P.compare(_records(oof_a), _records(oof_b), n_boot=n_boot, seed=seed)


def blend_oof(oof_a, oof_b, w: float) -> pd.DataFrame:
    """Arm A2 offline: rint(w * pred_b + (1 - w) * pred_a) on the common rows."""
    return _P.blend(_records(oof_a), _records(oof_b), w)
