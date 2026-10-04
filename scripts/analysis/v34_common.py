"""v34 no-seconds retrain — what every step shares (plans/PREREG_nsc_no_seconds_2026_09_24.md §7 and AMENDMENTS V34.1-V34.7).

Standard library + numpy/pandas/pyarrow only: no model family is imported here, so every step's runner (LightGBM, the ADS-B
stack, the CatBoost venv driver, the composer) can import it without mixing OpenMP runtimes (bug class BC-5).

What lives here:
  * the three label-seconds columns and `assert_no_seconds` (the negative invariant every step applies to its design);
  * the output discipline: every artefact goes under data/nsc_v34/<step dir>/ (real) or logs/nsc_v34_scratch/ (smoke),
    or is one of the three registered v34 submission files; nothing that exists is ever overwritten, and a path inside a
    shipped-artefact directory (data/cache_stand, data/adsb/v2, data/orl_build, data/fwp_build, data/nsc_run) is refused;
  * the registration check: the prereg's CURRENT sha must be logged (run_guards.assert_prereg_registered) AND the step's
    own amendment heading must be in the file, so no step runs before its amendment is appended and hashed;
  * the step-report contract: each step writes <step dir>/report.json with status PASS/FAIL and the sha of every output;
    the next step reads it and re-hashes what it consumes (chain of custody, no pinned-at-build shas that go stale).
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PREREG = ROOT / "plans" / "PREREG_nsc_no_seconds_2026_09_24.md"
HASHLOG = ROOT / "reports" / "nsc_prereg_hashes.log"
OUT_ROOT = ROOT / "data" / "nsc_v34"
SMOKE_ROOT = ROOT / "logs" / "nsc_v34_scratch"

#: the label-seconds channel (scripts/stand_ab.py:198-200); equal to nsc_fold.SECONDS_COLS (tested)
SECONDS_COLS = ("aobt_sec", "mvt_sec", "sched_sec")
N_BOARD = 344_841
N_MATCHED, N_UNMATCHED = 339_551, 5_290

STEP_DIRS = {1: "step1_f_fold", 2: "step2_adsb_stack", 3: "step3_f_refit", 4: "step4_fw_refit", 5: "step5_orl_refit",
             6: "step6_catboost", 7: "step7_compose"}
REPORT_NAME = "report.json"
#: the three submission files step 7 may create (never overwrite) under submissions/
V34_CANDIDATE = ROOT / "submissions" / "merry-quicksand_v34_nsc.parquet"
V34_MANIFEST = ROOT / "submissions" / "merry-quicksand_v34_nsc.manifest.parquet"
V34_META = ROOT / "submissions" / "merry-quicksand_v34_nsc.meta.json"
V34_FINAL = ROOT / "submissions" / "merry-quicksand_v34.parquet"
V34_SUBMISSION_FILES = (V34_CANDIDATE, V34_MANIFEST, V34_META, V34_FINAL)
#: shipped-artefact directories no v34 output may land in
PROTECTED_DIRS = tuple(ROOT / p for p in ("data/cache_stand", "data/adsb/v2", "data/orl_build", "data/fwp_build",
                                          "data/nsc_run", "data/pipeline_runs", "data/raw", "submissions"))


def amendment_heading(step: int) -> str:
    """The heading line each step's hashed amendment starts with (the runners refuse to start without it)."""
    if step not in STEP_DIRS:
        raise ValueError(f"no v34 step {step!r}")
    return f"## AMENDMENT V34.{step} "


# ---------------------------------------------------------------- the negative invariant

def assert_no_seconds(cols, what: str) -> list:
    """`cols` as a list; refuses any of the three label-seconds columns (by exact name, and any other *_sec name, so a
    renamed copy of the channel cannot slip in either)."""
    cols = list(cols)
    bad = [c for c in cols if isinstance(c, str) and (c in SECONDS_COLS or c.endswith("_sec"))]
    if bad:
        raise AssertionError(f"{what}: label-seconds column(s) {bad} in a v34 design")
    if len(set(cols)) != len(cols):
        raise AssertionError(f"{what}: duplicate column names")
    return cols


def taxi(proxy, delta) -> np.ndarray:
    """The matched convention: taxi = max(proxy - delta, 1) (float; rounding happens where a submission is written)."""
    return np.maximum(np.asarray(proxy, dtype="float64") - np.asarray(delta, dtype="float64"), 1.0)


# ---------------------------------------------------------------- output discipline

def _under(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def check_output_path(path, allow_roots=(OUT_ROOT, SMOKE_ROOT), allow_files=V34_SUBMISSION_FILES) -> pathlib.Path:
    """A v34 output path: under an allowed root (data/nsc_v34 or the smoke scratch) or one of the registered submission
    files, never inside a shipped-artefact directory, and not existing yet (no silent overwrite)."""
    p = pathlib.Path(path).resolve()
    roots = [pathlib.Path(r).resolve() for r in allow_roots]
    files = {pathlib.Path(f).resolve() for f in allow_files}
    if p not in files:
        if not any(_under(p, r) for r in roots):
            raise PermissionError(f"{p} is outside the v34 output roots {[str(r) for r in roots]}")
        if any(_under(p, d.resolve()) for d in PROTECTED_DIRS):
            raise PermissionError(f"{p} is inside a shipped-artefact directory")
    if p.exists():
        raise FileExistsError(f"{p} exists: a v34 step never overwrites")
    return p


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_parquet_new(df: pd.DataFrame, path, **kw) -> str:
    """Write once (check_output_path), re-read, require equality; returns the sha256."""
    p = check_output_path(path, **kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    back = pd.read_parquet(p)
    if not back.equals(df.reset_index(drop=True)):
        raise AssertionError(f"{p}: the re-read frame differs from the written one")
    return sha256(p)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        if not math.isfinite(f):
            raise ValueError("non-finite number in a v34 report")
        return f
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, pathlib.Path):
        return str(o)
    return o


def write_json_new(obj: dict, path, **kw) -> str:
    """JSON (allow_nan=False: a NaN in a report is a defect, not a value), written once; returns the sha256."""
    p = check_output_path(path, **kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jsonable(obj), indent=1, allow_nan=False))
    return sha256(p)


# ---------------------------------------------------------------- registration

def require_registration(step: int, prereg=PREREG, hashlog=HASHLOG) -> str:
    """The prereg's current sha is logged on a line naming it (run_guards' BC-25 idiom) and the step's amendment heading
    is in the file. Returns the sha."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_guards as G
    sha = G.assert_prereg_registered(prereg, hashlog)
    if amendment_heading(step) not in pathlib.Path(prereg).read_text():
        raise RuntimeError(f"{pathlib.Path(prereg).name} has no '{amendment_heading(step).strip()}' section: step {step} "
                           "is not registered")
    return sha


# ---------------------------------------------------------------- step reports (chain of custody)

def step_dir(step: int, root=OUT_ROOT) -> pathlib.Path:
    return pathlib.Path(root) / STEP_DIRS[step]


def read_step_report(step: int, root=OUT_ROOT) -> dict:
    """A finished step's report; refuses a missing report, a non-PASS status, a smoke report, or an output whose bytes no
    longer match the sha the step recorded."""
    p = step_dir(step, root) / REPORT_NAME
    if not p.exists():
        raise FileNotFoundError(f"step {step} has no report at {p}: run it first")
    rep = json.loads(p.read_text())
    if rep.get("status") != "PASS":
        raise RuntimeError(f"step {step} report status is {rep.get('status')!r}, not PASS")
    if rep.get("smoke"):
        raise RuntimeError(f"step {step} report at {p} is a smoke report")
    for rel, want in rep.get("outputs", {}).items():
        q = ROOT / rel
        if not q.exists():
            raise FileNotFoundError(f"step {step} output {rel} is missing")
        if sha256(q) != want:
            raise AssertionError(f"step {step} output {rel} changed since the step recorded it")
    return rep


def rel(path) -> str:
    """Repo-relative path string when the path is inside the repo (reports store these), else the absolute path (tests'
    tmp dirs); `ROOT / rel(p)` is `p` either way."""
    p = pathlib.Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)
