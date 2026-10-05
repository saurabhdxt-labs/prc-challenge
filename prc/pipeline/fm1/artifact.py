"""FM1 model artefacts: one directory per (resolved-config hash, input-data digest), written once, verified on load (AT4).

    <run.artifacts_dir>/<config_hash[:12]>-<data_digest[:8]>/
        manifest.json                 files -> sha256, config hash, data digest + input sha256s, git sha, library versions
        config.json                   the resolved config (every knob) and the overrides
        design.json                   matched design columns, stack feats, unmatched design columns
        encodings_matched.json  encodings_unmatched.json
        learners/base/  learners/catboost/  learners/unmatched/      (each: learner.json + one model file per seed)
        unmatched_lane/               the v26 lane's training frames (unm, lirf, extra .parquet) + lane.json (beta_W, seeds)
        stack/                        a byte copy of the stack models the config names (self-contained artefact)
        gate.json                     stored stack gate + the training covered share per airport
        mtx.json                      the fitted MTX mapping (only when calibration is on)
        fit_report.json

Written to a temporary sibling and renamed (never half-written); an existing directory is never overwritten
(ArtifactError). Loading re-hashes every file against the manifest and refuses a missing, extra or changed file BEFORE any
model file is deserialised.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

from . import components as C
from . import errors as E
from . import features as F
from . import knobs as K
from . import learners as L
from . import unmatched as U

MANIFEST = "manifest.json"
FORMAT_VERSION = 1


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def data_digest(inputs: dict) -> str:
    """sha256 over the sorted (name, sha256) pairs of the input files."""
    return hashlib.sha256(json.dumps(sorted(inputs.items())).encode()).hexdigest()


def git_state(root=K.ROOT) -> dict:
    """The commit and whether the code FM1 runs differs from it: prc/ (tracked or untracked) or a tracked scripts/ file
    (FM1 loads only tracked scripts, BC-38). Configs are not code: the config's values are stored in config.json."""
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):          # pragma: no cover - git missing is recorded, not fatal
            return None
    dirty = run("status", "--porcelain", "--", "prc") or run("status", "--porcelain", "--untracked-files=no", "--", "scripts")
    return {"sha": run("rev-parse", "HEAD"), "dirty_fm1": bool(dirty)}


def library_versions() -> dict:
    import lightgbm, numpy, pandas, pyarrow, sklearn   # noqa: E401
    return {"python": platform.python_version(), "lightgbm": lightgbm.__version__, "numpy": numpy.__version__,
            "pandas": pandas.__version__, "pyarrow": pyarrow.__version__, "sklearn": sklearn.__version__}


@dataclass
class ModelArtifact:
    config: K.FM1Config
    design_columns: list
    unmatched_columns: list
    enc_matched: F.EncodingTables | None
    enc_unmatched: F.EncodingTables | None
    base: object | None
    catboost: object | None
    unmatched: object | None
    stack_dir: pathlib.Path | None
    stack_feats: list
    gate_manifest: dict
    train_covered_share: dict
    mtx: C.MtxCalibration | None
    fit_report: dict
    inputs: dict = field(default_factory=dict)            # input name -> sha256
    directory: pathlib.Path | None = None
    unmatched_lane: U.UnmatchedModel | None = None        # the v26 lane (unmatched.learner v26)

    @property
    def data_digest(self) -> str:
        return data_digest(self.inputs)

    @property
    def key(self) -> str:
        return f"{self.config.short_hash}-{self.data_digest[:8]}"


def _write_json(p: pathlib.Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))


def artifact_dir(config, inputs: dict, parent=None) -> pathlib.Path:
    """Where the artefact of this config fitted on these inputs is saved: <parent>/<config hash[:12]>-<input digest[:8]>,
    known before any fitting (ModelArtifact.key is the same name)."""
    parent = pathlib.Path(parent) if parent is not None else config.path("run.artifacts_dir")
    return parent / f"{config.short_hash}-{data_digest(inputs)[:8]}"


def refuse_existing(final: pathlib.Path) -> None:
    if final.exists():
        raise E.ArtifactError(f"artefact {final} already exists; FM1 never overwrites an artefact (same config and data "
                              "= same model; remove it deliberately to refit)")


def save(art: ModelArtifact, parent=None) -> pathlib.Path:
    parent = pathlib.Path(parent) if parent is not None else art.config.path("run.artifacts_dir")
    final = parent / art.key
    refuse_existing(final)
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / f".{art.key}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)                                   # our own pid's leftover only
    tmp.mkdir()
    try:
        cfg = art.config
        _write_json(tmp / "config.json", {"values": K.nest(cfg.values), "hash": cfg.hash,
                                          "overrides": [list(o) for o in cfg.overrides]})
        _write_json(tmp / "design.json", {"matched": art.design_columns, "unmatched": art.unmatched_columns,
                                          "stack_feats": art.stack_feats})
        if art.enc_matched is not None:
            (tmp / "encodings_matched.json").write_text(art.enc_matched.to_json())
        if art.enc_unmatched is not None:
            (tmp / "encodings_unmatched.json").write_text(art.enc_unmatched.to_json())
        for name in ("base", "catboost", "unmatched"):
            m = getattr(art, name)
            if m is not None:
                m.save(tmp / "learners" / name)
        if art.unmatched_lane is not None:
            art.unmatched_lane.save(tmp / "unmatched_lane")
        if art.stack_dir is not None:
            # content only: a copy that kept the source's timestamps would be born days old (BC-44)
            shutil.copytree(art.stack_dir, tmp / "stack", copy_function=shutil.copyfile)
        _write_json(tmp / "gate.json", {"manifest": art.gate_manifest, "train_covered_share": art.train_covered_share})
        if art.mtx is not None:
            (tmp / "mtx.json").write_text(art.mtx.to_json())
        _write_json(tmp / "fit_report.json", art.fit_report)
        files = {str(p.relative_to(tmp)): sha256_file(p) for p in sorted(tmp.rglob("*")) if p.is_file()}
        _write_json(tmp / MANIFEST, {
            "format_version": FORMAT_VERSION, "key": art.key, "config_hash": cfg.hash, "data_digest": art.data_digest,
            "inputs": art.inputs, "files": files, "git": git_state(), "libraries": library_versions(),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    art.directory = final
    if art.stack_dir is not None:
        art.stack_dir = final / "stack"
    return final


def verify(directory) -> dict:
    d = pathlib.Path(directory)
    mp = d / MANIFEST
    if not mp.exists():
        raise E.ArtifactError(f"{d}: no {MANIFEST} (not an FM1 artefact, or a half-written one)")
    man = json.loads(mp.read_text())
    if man.get("format_version") != FORMAT_VERSION:
        raise E.ArtifactError(f"{d}: format_version {man.get('format_version')!r}, this loader reads {FORMAT_VERSION}")
    listed = man["files"]
    present = {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and p.name != MANIFEST}
    missing, extra = sorted(set(listed) - present), sorted(present - set(listed))
    if missing or extra:
        raise E.ArtifactError(f"{d}: files missing {missing}, unexpected {extra}")
    bad = [f for f, h in listed.items() if sha256_file(d / f) != h]
    if bad:
        raise E.ArtifactError(f"{d}: file(s) changed since the artefact was written: {bad}")
    return man


def load(directory) -> ModelArtifact:
    d = pathlib.Path(directory)
    man = verify(d)
    cj = json.loads((d / "config.json").read_text())
    cfg = K.parse_config(cj["values"])
    if cfg.hash != cj["hash"] or cfg.hash != man["config_hash"]:
        raise E.ArtifactError(f"{d}: the stored config re-validates to hash {cfg.hash[:12]}, the manifest says "
                              f"{man['config_hash'][:12]} (schema drift since the fit)")
    design = json.loads((d / "design.json").read_text())
    gate = json.loads((d / "gate.json").read_text())

    def enc(name):
        p = d / name
        return F.EncodingTables.from_json(p.read_text()) if p.exists() else None

    def learner(name):
        p = d / "learners" / name
        return L.load_learner(p) if p.exists() else None
    mtx = C.MtxCalibration.from_json((d / "mtx.json").read_text()) if (d / "mtx.json").exists() else None
    return ModelArtifact(config=cfg, design_columns=design["matched"], unmatched_columns=design["unmatched"],
                         enc_matched=enc("encodings_matched.json"), enc_unmatched=enc("encodings_unmatched.json"),
                         base=learner("base"), catboost=learner("catboost"), unmatched=learner("unmatched"),
                         stack_dir=(d / "stack") if (d / "stack").exists() else None, stack_feats=design["stack_feats"],
                         gate_manifest=gate["manifest"], train_covered_share=gate["train_covered_share"], mtx=mtx,
                         fit_report=json.loads((d / "fit_report.json").read_text()), inputs=man["inputs"], directory=d,
                         unmatched_lane=U.UnmatchedModel.load(d / "unmatched_lane") if (d / "unmatched_lane").exists()
                         else None)
