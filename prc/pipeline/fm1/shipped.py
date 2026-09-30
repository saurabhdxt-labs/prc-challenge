"""The shipped, tested builders FM1 wraps — imported through their public entry points, never copied and never edited.

scripts/ modules come through prc.pipeline.legacy (one instance per process); scripts/analysis/ modules are loaded by path
(they are not a package), once per process, under their own module name so every caller shares the instance. The
unmatched lane's modules come through urp_replay.load_deps() (`unmatched_deps`), which imports them by name the same way.
"""
from __future__ import annotations

import functools
import importlib.util
import pathlib
import sys

from .. import legacy

ROOT = pathlib.Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "scripts" / "analysis"


@functools.lru_cache(maxsize=None)
def analysis_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    if str(ANALYSIS) not in sys.path:                  # mtx_run imports mtx_calib / mtx_source by bare name
        sys.path.insert(0, str(ANALYSIS))
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stand_ab():
    return legacy.matched().S


def unm_congestion():
    return legacy.stratum().uc


def orl_build():
    return analysis_script("orl_build")


def mtx_run():
    return analysis_script("mtx_run")


def mtx_calib():
    return analysis_script("mtx_calib")


def mtx_source():
    return analysis_script("mtx_source")


def ldi_correction():
    """scripts/analysis/ldi_correction.py: LDI's registered definitions (band, local_day, severe_days, to_epoch_s;
    plans/PREREG_ldi_ltfm_deicing_2026_09_27.md), tested in tests/test_ldi_correction.py."""
    return analysis_script("ldi_correction")


def adsb_predictor():
    """adsb_stack.predict_residual(frame, model_dir, oof) — the MLP half runs in adsb_nn_worker.py (BC-5)."""
    return legacy.adsb_predictor()


# ---- the unmatched lane (plans/PREREG_fm1_2026_09_28.md Amendment A3) ------------------------------------------------

#: the frozen E2 rule file the v31 / v33 builds pinned (rse_build.RULE_SHA, e2w_build.RULE_SHA, urp_replay.FROZEN)
E2_RULES = ROOT / "data" / "rome_hunt_2026_09_23" / "agent_E_rules.py"
E2_RULES_SHA256 = "bd01f4ed927d473585b6a182192073c59ed75fa220101db34035343c7629b606"


def urp_replay():
    """scripts/analysis/urp_replay.py: the tested replay of the SHIPPED v33 unmatched chain (serve parity 5,290/5,290 at
    all 13 versions, plans/PREREG_urp_u_chain_replay_2026_09_24.md RESULT). FM1 uses its public composition helpers
    (s1_parts, fill_p, lirf_rules, arc_rows, e2_apply, e2w_apply, attach_anomaly, nested_lirf_inputs, e2w_training,
    fit_e2w_beta) and, through `unmatched_deps()`, the shipped modules it binds."""
    return analysis_script("urp_replay")


@functools.lru_cache(maxsize=None)
def unmatched_deps():
    """urp_replay.load_deps(): the shipped chain's public functions and modules in one namespace (build_submission,
    unm_congestion, unm_congestion_ship, rome_fill, rome_dateslip, rome_localday, rome_local_day, congestion_anchor_ship,
    arr_row_clocks, arc_ship_build, mau_build, mxi_build, mxj_build, rse_build, e2w_build, e2w_ship, agent_E_rules).
    Refuses an E2 rule file whose sha256 is not the one v31 / v33 shipped with."""
    import hashlib
    got = hashlib.sha256(E2_RULES.read_bytes()).hexdigest()
    if got != E2_RULES_SHA256:
        raise RuntimeError(f"{E2_RULES}: sha256 {got[:16]} is not the shipped rule file {E2_RULES_SHA256[:16]}")
    return urp_replay().load_deps()


def csf2_build():
    """scripts/analysis/csf2_build.py: `labels` (fill / slip / ordinary), the class definition of the table the shipped
    E2W coefficients were fitted on (reports/csf2/csf2_preds.parquet `cls`)."""
    if str(ROOT) not in sys.path:                      # it imports `scripts.analysis.csf_build` as a package path
        sys.path.insert(0, str(ROOT))
    return analysis_script("csf2_build")
