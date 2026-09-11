"""Write the submission, its per-row provenance and the manifest; refuse to overwrite anything.

The manifest stamps every output with what produced it: git sha + dirty flag, the config hash (and the
config file's sha256), the sha256 of every input file, per-lane / per-airport counts, fallback airports,
the validation report, library versions and stage timings. `digest` is the sha256 of the manifest's
DETERMINISTIC part (everything except wall-clock times and the output directory), so two runs on the same
inputs with the same code produce the same digest (acceptance A2).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import platform
import re
import subprocess
import sys
import types

import pyarrow.parquet as pq

from . import errors as E
from . import legacy

MANIFEST_VERSION = 2          # 2: the `code` block (source-file hashes; review finding 2026-09-11)
#: keys excluded from the digest: they differ between two identical runs
VOLATILE = ("created_utc", "timings_s", "digest", "out_dir")
REQUIRED_KEYS = ("manifest_version", "pipeline", "created_utc", "git", "code", "config", "inputs", "outputs", "lanes",
                 "airports", "fallback_airports", "routing", "validation", "environment", "no_splice", "timings_s",
                 "digest")
#: repo directories whose loaded modules are the code that ran
CODE_DIRS = ("prc", "scripts")


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_state(root: pathlib.Path) -> dict:
    """Read-only: HEAD sha and whether the working tree is dirty (None if git is unavailable)."""
    def run(*a):
        try:
            r = subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"sha": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def loaded_code_files(root: pathlib.Path) -> list:
    """Repo-relative paths of every source file under root/{prc,scripts} that this process has imported, plus every
    module of prc/pipeline (imported or not), sorted. The git sha alone cannot say which code ran when the tree is
    dirty or the code is untracked; these files can.

    `sys.modules` is not the whole record: the scripts load each other with importlib specs and keep the handle as a
    module global without registering it (lgbm_submit.S is stand_ab, stratum_fold.bs is build_submission). So the walk
    starts from sys.modules and follows every module object held in the globals of an in-repo module. A module loaded
    and then dropped by every holder would be missed; none of the scripts does that."""
    root = pathlib.Path(root).resolve()

    def rel_of(mod):
        f = getattr(mod, "__file__", None)
        if not f or not str(f).endswith(".py"):
            return None
        try:
            rel = pathlib.Path(f).resolve().relative_to(root)
        except ValueError:
            return None
        return rel if rel.parts and rel.parts[0] in CODE_DIRS else None

    out, seen, stack = set(), set(), list(sys.modules.values())
    while stack:
        mod = stack.pop()
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        rel = rel_of(mod)
        if rel is None:
            continue
        out.add(rel)
        stack.extend(v for v in list(vars(mod).values()) if isinstance(v, types.ModuleType))
    out.update(p.resolve().relative_to(root) for p in (root / "prc" / "pipeline").glob("*.py"))
    return sorted(out)


def porcelain_paths(text: str) -> list:
    """Paths named by `git status --porcelain` output (modified, added, untracked, renamed -> the new name)."""
    out = []
    for line in text.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        out.append(path.strip('"'))
    return sorted(out)


def code_state(root: pathlib.Path) -> dict:
    """{files: {path: sha256}, differs_from_head: [paths git HEAD does not hold as they are] | None without git}."""
    root = pathlib.Path(root).resolve()
    files = loaded_code_files(root)
    rec = {str(r): sha256_file(root / r) for r in files}
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *map(str, files)], cwd=root,
                           capture_output=True, text=True, timeout=60)
        differs = porcelain_paths(r.stdout) if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        differs = None
    return {"files": rec, "differs_from_head": differs}


def environment() -> dict:
    import numpy
    import pandas
    import pyarrow
    import sklearn
    env = {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
           "pyarrow": pyarrow.__version__, "sklearn": sklearn.__version__}
    try:
        import lightgbm
        env["lightgbm"] = lightgbm.__version__
    except ImportError:                          # pragma: no cover - lightgbm is a hard dependency of the matched lane
        env["lightgbm"] = None
    return env


def describe_inputs(named_paths: dict, root: pathlib.Path) -> dict:
    """{name: {path (repo-relative when inside the repo), sha256, bytes}} for every input file."""
    out = {}
    for name, p in sorted(named_paths.items()):
        p = pathlib.Path(p)
        if not p.exists():
            raise E.SchemaError(f"input {name}: {p} does not exist")
        try:
            rel = str(p.resolve().relative_to(pathlib.Path(root).resolve()))
        except ValueError:
            rel = str(p)
        out[name] = {"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def manifest_digest(manifest: dict) -> str:
    blob = {k: v for k, v in manifest.items() if k not in VOLATILE}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def build_manifest(cfg, inputs: dict, outputs: dict, lanes: dict, routing, validation: dict, timings: dict,
                   no_splice: list, out_dir=None) -> dict:
    airports = {}
    by_ap = routing.by_airport()
    for code, a in cfg.airports.items():
        airports[code] = {"tz": a.tz, "history": a.history, "fallback": a.fallback, "lanes": dict(a.lanes),
                          "rules": list(a.rules.order), "n_scored_rows": int(sum(by_ap.get(code, {}).values())),
                          "rows_by_lane": by_ap.get(code, {})}
    cfg_file = cfg.source_path
    m = {"manifest_version": MANIFEST_VERSION, "pipeline": "prc.pipeline (P2a: wrappers over the shipped scripts)",
         "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "git": git_state(cfg.paths.root),
         "code": code_state(cfg.paths.root),
         "config": {"hash": cfg.config_hash, "file": str(cfg_file) if cfg_file else None,
                    "file_sha256": sha256_file(cfg_file) if cfg_file else None, "seeds": list(cfg.seeds),
                    "folds": {"fold_a_holdout_months": list(cfg.folds.fold_a_holdout_months),
                              "early_stopping_months": list(cfg.folds.early_stopping_months),
                              "lomo_months": list(cfg.folds.lomo_months)}},
         "inputs": inputs, "outputs": outputs, "lanes": lanes, "airports": airports,
         "fallback_airports": list(cfg.fallback_airports()),
         "routing": {"counts": routing.counts(), "by_airport": by_ap,
                     "n_fallback_rows": int(routing.fallback.sum())},
         "validation": validation, "environment": environment(),
         "no_splice": {"asserted": True, "n_inputs_checked": len(no_splice),
                       "rule": "no input under submissions/; no config key names a base submission"},
         "timings_s": timings, "out_dir": str(out_dir) if out_dir else None}
    m["digest"] = manifest_digest(m)
    return m


#: a key that names a wall-clock measurement; allowed only under the volatile keys (bug class found 2026-09-11:
#: a lane's `predict_s` inside `lanes` made two identical runs' digests differ whenever the rounding did)
TIMING_KEY = re.compile(r"(^|_)(wall|design|predict|fit|load|elapsed|duration|time)_s$|^wall_s$")


def volatile_leaks(node, path: str = "") -> list:
    """Dotted paths of timing-named keys anywhere outside VOLATILE."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            if not path and k in VOLATILE:
                continue
            if isinstance(k, str) and TIMING_KEY.search(k):
                out.append(p)
            out += volatile_leaks(v, p)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += volatile_leaks(v, f"{path}[{i}]")
    return out


def check_manifest_shape(m: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in m]
    if missing:
        raise E.ValidationError(f"manifest lacks {missing}")
    leaks = volatile_leaks(m)
    if leaks:
        raise E.ValidationError(f"wall-clock value(s) {leaks} outside {VOLATILE}: the digest would differ between two "
                                "identical runs")
    if m["digest"] != manifest_digest(m):
        raise E.ValidationError("manifest digest does not match its content")
    code = m["code"]
    if not isinstance(code, dict) or not code.get("files") or "prc/pipeline/write.py" not in code["files"]:
        raise E.ValidationError("manifest `code` must hash the source files that ran (prc/pipeline/write.py at least)")
    bad = [p for p, h in code["files"].items() if not (isinstance(h, str) and len(h) == 64)]
    if bad:
        raise E.ValidationError(f"manifest code entries without a sha256: {bad[:5]}")
    for name, rec in m["inputs"].items():
        if set(rec) != {"path", "sha256", "bytes"} or len(rec["sha256"]) != 64:
            raise E.ValidationError(f"manifest input {name!r} is malformed: {rec}")


def output_names(team: str, version: int | None) -> tuple:
    """(submission, provenance, manifest) file names. Without a version the file is `submission.parquet`: a
    reproduction run cannot be mistaken for an upload (the scorer only accepts <team>_vN.parquet)."""
    if version is None:
        stem = "submission"
    else:
        stem = legacy.stratum().bs.submission_name(int(version), team=team)[: -len(".parquet")]
    return f"{stem}.parquet", f"{stem}.provenance.parquet", f"{stem}.manifest.json"


def write_submission(assembled, template, out_dir, team: str, version: int | None = None) -> dict:
    """Write the submission and its provenance; refuse to overwrite; re-read and check what landed."""
    bs = legacy.stratum().bs
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_name, prov_name, _ = output_names(team, version)
    dest, prov_dest = out_dir / sub_name, out_dir / prov_name
    for p in (dest, prov_dest):
        if p.exists():
            raise E.ValidationError(f"refusing to overwrite {p}")
    assembled.submission.to_parquet(dest, index=False)
    reread = pq.read_table(dest).to_pandas()
    bs.check_submission(reread, template)
    if not reread.equals(assembled.submission):
        raise E.ValidationError(f"{dest} re-read differs from the frame written")
    assembled.provenance.to_parquet(prov_dest, index=False)
    return {"submission": {"file": dest.name, "sha256": sha256_file(dest), "rows": int(len(reread))},
            "provenance": {"file": prov_dest.name, "sha256": sha256_file(prov_dest)}}


def write_manifest(manifest: dict, out_dir, team: str, version: int | None = None) -> pathlib.Path:
    check_manifest_shape(manifest)
    path = pathlib.Path(out_dir) / output_names(team, version)[2]
    if path.exists():
        raise E.ValidationError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str))
    return path
