"""URP — fold replay of the SHIPPED v33 unmatched (U) chain on 2025 (plans/PREREG_urp_u_chain_replay_2026_09_24.md).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/urp_replay.py --smoke     # lomo, January only; <= 5 min; writes logs/urp_scratch/
    /opt/homebrew/bin/python3.11 -B scripts/analysis/urp_replay.py --run       # lomo acceptance -> serve parity -> fold_a -> artifact

The chain, in shipped order (every step's code is IMPORTED from the file that shipped it, never re-implemented):
  v4  S1 stratum hybrid                     build_submission.fit_unmatched(hybrid=True)          all U rows
  v7  R2 fill classifier + 24 h date-slip   rome_fill.fit_predict_fill, rome_dateslip.apply      LIRF
  v9  E3C congestion body (S1C)             unm_congestion.fit_congestion_regressor              other
  v10 E1 schedule floor                     rome_localday.e1_floor                               LIRF
  v12 RLD Rome local-day rule               rome_local_day.apply_rld                             LIRF
  v16 SCA storm anchor                      congestion_anchor_ship.anchored_values               other
  v17 ARC arrival-row clocks                arr_row_clocks.sub_rule, arc_ship_build.arc_values   LIRF
  v21 ARN (ARC-N / XJ)                      rome_arn.rule_rows                                   LIRF
  v24 MAU extra matched rows at 25 % mass   mxj_build.fit_body(body_params(200))                 other
  v25 MXS max_iter 1,000                    mxj_build.fit_body(body_params(1000))                other
  v26 MJS + patience 50                     mxj_build.fit_body(body_params(1000) + patience 50)  other
  v31 RSE frozen E2 + per-day p99           agent_E_rules.rule_E2, rse_build.row_set/candidate   LIRF
  v33 E2WS stake-weighted refit + p99       e2w_ship.e2_form, e2w_build.fit_brier_w              LIRF
(v29 ORL and v30 ORU touched no U row; verified in the registration.)

Every fit goes through `train_view`, which raises LeakError on a training row from a held-out month. The pure functions take their
dependencies as a `deps` namespace (`load_deps()` binds the shipped modules), so the orchestration is tested on synthetic frames.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import resource
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PREREG = ROOT / "plans" / "PREREG_urp_u_chain_replay_2026_09_24.md"
HASHLOG = ROOT / "reports" / "urp_prereg_hashes.log"
OUT = ROOT / "data" / "cache_stand" / "u_fold_current_v33.parquet"
SIDECAR = OUT.with_suffix(".json")
SCRATCH = ROOT / "logs" / "urp_scratch"

BOARD_N = 344_841.0
HOLDOUT = (1, 7)
MONTHS = tuple(range(1, 13))
SEEDS = (0, 1, 2)
LIRF = "LIRF"
SCHEMES = ("lomo", "fold_a", "serve")
CHAIN = ("v4", "v7", "v9", "v10", "v12", "v16", "v17", "v21", "v24", "v25", "v26", "v31", "v33")
AFTER = tuple(f"after_{v}" for v in CHAIN)
LIRF_STEPS = {"v7", "v10", "v12", "v17", "v21", "v31", "v33"}
OTHER_STEPS = {"v9", "v16", "v24", "v25", "v26"}
P99_RSE = 15.941366958491995           # submissions/merry-quicksand_v31_rse.meta.json day_check_B8prime.p99
P99_E2W = 9.042848041672684            # submissions/merry-quicksand_v33_e2w.meta.json day_check.p99
SP_LO, SP_HI = 3_600.0, 24_000.0       # E2's band (agent_E_rules.SP_LO_S / SP_HI_S; e2w_ship.SP_LO / SP_HI)
WINSOR = 3_000.0                       # the body's target cap (build_submission.WINSOR_S; mau_build clips at 3,000)
SPENT_DAY = "2025-01-09"               # cln_build.SPENT_DAY: the day cln_build drops from lane U
RSS_LIMIT_GIB = {"smoke": 5.0, "run": 8.0}
SMOKE_ALARM_S = 290
ANOMALY_COLS = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
OUTPUT_COLUMNS = ["MVT_ID_mvt", "y", "month", "doy", "ap", "cln_spent_day", *AFTER, "pred"]

#: the recorded 2025 figures and frozen files the acceptance and parity checks read (sha256, full)
FROZEN = {
    "data/cache_rome/unm.parquet": "83bb2f78356ed0f669b5424d8fef927839d462f12fd04028cb6a17c9b3b8d100",
    "data/cache_rome/lirf.parquet": "189c75ae0a08e094b794e3b946b4d764b539ac95c366dcf43444e1b72bfe7ba7",
    "data/cache_stand/unm_physics_lomo.parquet": "1ae5f1392a35a44d96462f0d8c2e81481fffd0d6270fa28fae31c36a68238aa5",
    "data/cache_stand/stratum_fold_v7_preds.parquet": "7023a0acf32aed4c7cb4af4849e5af786d009ba3240c678cf823c6c17883e015",
    "data/cache_stand/rome_dateslip_preds.parquet": "888e5715e68c864345089c1a588a14abc1a01e980babc39a409aaa7cfc5e3f86",
    "reports/csf2/csf2_preds.parquet": "a614a610247d4e565e69f3b22090a1f1ca8b66c9e04c73060d7f1fceb36825bc",
    "reports/mau_test_predictions.parquet": "54ad6dd2e40252450c3c5d0bd71c0409d2f9c3d0c554dde741e4bb29a8db7101",
    "reports/mxi_predictions.parquet": "2afb801b95f43555c4e8d0e46a498601cc98e245501943256d85557ef094a28f",
    "reports/mxj_predictions.parquet": "c9a088e9ddf3e811b538b26826ff35be58cac772b865f2f56276e11efd20b6bb",
    "reports/rrh_holdout_report.json": "baf8873a2e376604e1743e81645521ada32f1a11b1f1cb346bad1cb61ad391c2",
    "reports/e2w_report.json": "fca3ebeb0cf776222337a28c3e78f5e1df4ab5b1061cf325e3103f0d8d5e2993",
    "reports/rome_arn_2026_09_16.json": "5ce13dcb5d86499affb2c78f715013a61aaca74ec8572530b13583f71818e2eb",
    "data/rome_hunt_2026_09_23/agent_E_rules.py": "bd01f4ed927d473585b6a182192073c59ed75fa220101db34035343c7629b606",
    "data/raw/ranking.parquet": "9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86",
}
SUBMISSIONS = {"v4": "merry-quicksand_v4.parquet", "v7": "merry-quicksand_v7.parquet", "v9": "merry-quicksand_v9.parquet",
               "v10": "merry-quicksand_v10.parquet", "v12": "merry-quicksand_v12.parquet", "v16": "merry-quicksand_v16.parquet",
               "v17": "merry-quicksand_v17.parquet", "v21": "merry-quicksand_v21.parquet", "v24": "merry-quicksand_v24.parquet",
               "v25": "merry-quicksand_v25.parquet", "v26": "merry-quicksand_v26.parquet", "v31": "merry-quicksand_v31_rse.parquet",
               "v33": "merry-quicksand_v33.parquet"}


class LeakError(AssertionError):
    """A training frame holds a row from a held-out month."""


@dataclasses.dataclass(frozen=True)
class Step:
    version: str
    name: str
    lane: str            # "LIRF" | "other" | "all"
    fit: str             # "OOF" | "none"  -- does the fold_a replay fit on a held-out label (never)
    selection: str       # "OOS" | "IN"    -- were the shipped form/constants chosen after reading Jan/Jul 2025

    @property
    def label(self) -> str:
        if self.fit not in ("OOF", "none"):
            raise ValueError(f"step {self.version}: unknown fit tag {self.fit!r}")
        if self.selection not in ("OOS", "IN"):
            raise ValueError(f"step {self.version}: unknown selection tag {self.selection!r}")
        return "OUT-OF-SAMPLE" if self.selection == "OOS" else "IN-SAMPLE"


STEPS = (
    Step("v4", "S1 stratum hybrid", "all", "OOF", "IN"),
    Step("v7", "R2 fill classifier + date-slip", LIRF, "OOF", "IN"),
    Step("v9", "E3C congestion body", "other", "OOF", "IN"),
    Step("v10", "E1 schedule floor", LIRF, "none", "IN"),
    Step("v12", "RLD local-day rule", LIRF, "none", "IN"),
    Step("v16", "SCA storm anchor", "other", "none", "IN"),
    Step("v17", "ARC arrival-row clocks", LIRF, "none", "IN"),
    Step("v21", "ARN ARC-N / XJ", LIRF, "none", "OOS"),
    Step("v24", "MAU extra matched rows", "other", "OOF", "IN"),
    Step("v25", "MXS max_iter 1000", "other", "OOF", "IN"),
    Step("v26", "MJS patience 50", "other", "OOF", "IN"),
    Step("v31", "RSE frozen E2", LIRF, "none", "IN"),
    Step("v33", "E2WS stake-weighted refit", LIRF, "OOF", "IN"),
)


# =============================================================================================== pure
def folds(scheme: str) -> list:
    """[(test_months, train_months)] for a scheme. serve tests 2026 (no 2025 month is held out)."""
    if scheme == "lomo":
        return [((m,), tuple(x for x in MONTHS if x != m)) for m in HOLDOUT]
    if scheme == "fold_a":
        return [(HOLDOUT, tuple(x for x in MONTHS if x not in HOLDOUT))]
    if scheme == "serve":
        return [((), MONTHS)]
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")


def train_view(frame: pd.DataFrame, held, what: str) -> pd.DataFrame:
    """`frame` unchanged, or LeakError if any of its rows is from a held-out month. Every fitted object is fitted on a view."""
    if "month" not in frame.columns:
        raise KeyError(f"{what}: the training frame has no 'month' column, so the leak guard cannot check it")
    held = {int(m) for m in held}
    bad = sorted({int(m) for m in frame.month.to_numpy()} & held)
    if bad:
        raise LeakError(f"{what}: {int(frame.month.isin(bad).sum())} training rows from held-out months {bad}")
    return frame


def overwrite(base, ids, rule_ids, rule_values) -> tuple:
    """(values, changed mask): `base` (aligned with `ids`) with `rule_values` on `rule_ids`. Refuses duplicate or unknown ids."""
    base, ids = np.asarray(base, dtype="float64"), np.asarray(ids, dtype="float64")
    rule_ids, rule_values = np.asarray(rule_ids, dtype="float64"), np.asarray(rule_values, dtype="float64")
    if len(base) != len(ids):
        raise ValueError(f"base has {len(base)} values for {len(ids)} ids")
    if len(rule_ids) != len(rule_values):
        raise ValueError(f"{len(rule_ids)} rule ids vs {len(rule_values)} rule values")
    if len(np.unique(rule_ids)) != len(rule_ids):
        raise ValueError("duplicate rule ids")
    pos = pd.Index(ids).get_indexer(rule_ids)
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} rule ids are not among the rows")
    out = base.copy()
    out[pos] = rule_values
    return out, out != base


def e2_apply(deps, v13, v_now, p_e2, sp, nf, day, p99: float = P99_RSE) -> tuple:
    """RSE (v31): S = E2 fires AND the row still holds its v13 value AND sp < 24,000 (rse_build.row_set); rint mixture(p_E2) on
    S; then the RSE.1 per-day check reverts every S row of a day whose move energy per moved row exceeds `p99`."""
    fired = np.isfinite(np.asarray(p_e2, dtype="float64"))
    S = deps.row_set(fired, v_now, v13, sp)
    new = deps.candidate(v_now, S, p_e2, sp, nf, deps.mixture)
    S2 = deps.apply_day_check(S, day, deps.day_energy(v_now, new, day), p99)
    return deps.candidate(v_now, S2, p_e2, sp, nf, deps.mixture), S2


def e2w_apply(deps, v_before, v31, p_w, sp, nf, day, p99: float = P99_E2W) -> tuple:
    """E2WS (v33): on the rows v31 changed, rint mixture(p_W); then the per-day p99 check against `p99`."""
    v_before, v31, p_w = (np.asarray(a, dtype="float64") for a in (v_before, v31, p_w))
    S = v31 != v_before
    if np.isnan(p_w[S]).any():
        raise ValueError(f"{int(np.isnan(p_w[S]).sum())} rows changed by v31 have no p_W")
    new = deps.candidate(v31, S, p_w, sp, nf, deps.mixture)
    S2 = deps.apply_day_check(S, day, deps.day_energy(v31, new, day), p99)
    return deps.candidate(v31, S2, p_w, sp, nf, deps.mixture), S2


def s1_parts(deps, train_unm, train_lirf, test, seeds, held) -> tuple:
    """(S1, parts) from build_submission.fit_unmatched(hybrid=True) on training views."""
    parts: dict = {}
    s1 = deps.fit_unmatched(train_view(train_unm, held, "S1 unmatched rows"), test,
                            train_matched=train_view(train_lirf, held, "S1 LIRF matched rows"),
                            hybrid=True, seeds=tuple(seeds), parts=parts)
    return np.asarray(s1, dtype="float64"), parts


def fill_p(deps, train_unm, train_lirf, test_lirf, months, seeds, held) -> np.ndarray:
    """R's P(fill) on `test_lirf`: mean over seeds of rome_fill.fit_predict_fill on every LIRF row of `months`."""
    tr = train_view(deps.rome_rows(train_view(train_unm, held, "R2 unmatched rows"),
                                   train_view(train_lirf, held, "R2 matched rows"), months), held, "R2 fill training rows")
    return np.mean([deps.fit_predict_fill(tr, test_lirf, seed=s) for s in seeds], axis=0)


def lirf_rules(deps, train_unm, p_r, test_lirf, nf, held) -> dict:
    """v7 -> v10 -> v12 on the LIRF rows in the shipped order (prc/pipeline/lanes.py:257-286): mixture(p_R, sp, nf), date-slip
    with shares from the training LIRF unmatched rows, rint, E1 floor, RLD."""
    sp = test_lirf.sp.to_numpy(dtype="float64")
    v = deps.mixture(p_r, sp, nf)
    q, m = deps.band_shares(train_view(train_unm[train_unm.ADEP_mvt == LIRF], held, "date-slip shares"))
    v7 = np.rint(deps.ds_apply(test_lirf, v, q, m))
    v10 = np.asarray(deps.e1_floor(v7, sp), dtype="float64")
    fr = pd.DataFrame({"MVT_ID_mvt": test_lirf.MVT_ID_mvt.to_numpy(), "ADEP_mvt": LIRF, "unmatched": True, "base": v10,
                       "SCHED_TIME_UTC_mvt": test_lirf.SCHED_TIME_UTC_mvt.to_numpy(),
                       "MVT_TIME_UTC_mvt": test_lirf.MVT_TIME_UTC_mvt.to_numpy()})
    v12, _ = deps.apply_rld(fr)
    return {"v7": v7, "v10": v10, "v12": np.asarray(v12, dtype="float64"), "q": q, "m": m}


def body_arms(deps) -> dict:
    """The three shipped body parameter sets (MAU v24, MXS v25, MJS v26); S1C (v9) is the plain congestion regressor."""
    return {"MAU": deps.body_params(200), "MXI": deps.body_params(1000),
            "MJS": {**deps.body_params(1000), "n_iter_no_change": 50}}


def other_bodies(deps, train_unm_w, test_w, extra, seeds, held) -> dict:
    """{arm: body averaged over seeds} for S1C (v9), MAU (v24), MXI (v25), MJS (v26) — mxj_build._fit_both's construction."""
    nonfill = train_view(train_unm_w[~deps.schedule_fill(train_unm_w)], held, "congestion body rows")
    ex = train_view(extra, held, "MAU extra matched rows")
    arms = body_arms(deps)
    preds = {k: {} for k in ("S1C", *arms)}
    for s in (int(x) for x in seeds):
        ctrl = deps.fit_congestion_regressor(nonfill, s)
        preds["S1C"][s] = ctrl.predict(test_w)
        x = ctrl.design(nonfill)
        yy = np.clip(nonfill.y.to_numpy(dtype="float64"), -WINSOR, WINSOR)
        fi, va = deps.stopping_split(len(nonfill), s)
        dex, dte = ctrl.design(ex), ctrl.design(test_w)
        for name, params in arms.items():
            model = deps.fit_body(x.iloc[fi], yy[fi], dex, ex.y.to_numpy(dtype="float64"), s, x.iloc[va], yy[va], params)
            preds[name][s] = model.predict(dte)
    return {k: deps.mean_over_seeds(v) for k, v in preds.items()}


def compose_other(deps, p, sp, nf_cells, bodies: dict, anomaly) -> dict:
    """v9 = rint S1C; v16 = storm anchor on v9; v24 / v25 / v26 = anchor(rint mixture(nf_hybrid(cells, body)))."""
    def mix(body):
        return deps.mixture(p, sp, deps.nf_hybrid(nf_cells, body))
    v9 = np.rint(mix(bodies["S1C"]))
    out = {"S1C": mix(bodies["S1C"]), "v9": v9, "v16": np.asarray(deps.anchored_values(v9, anomaly), dtype="float64")}
    for ver, arm in (("v24", "MAU"), ("v25", "MXI"), ("v26", "MJS")):
        out[ver] = np.asarray(deps.anchored_values(np.rint(mix(bodies[arm])), anomaly), dtype="float64")
    return out


def arc_rows(deps, dep_raw: pd.DataFrame, arr_raw: pd.DataFrame, ids) -> tuple:
    """(rule per id, ruled ids, values): arc_ship_build's route (arr_row_clocks.dep_frame / join_arr / sub_rule, arc_values)."""
    A = deps.A
    dep = A.dep_frame(dep_raw, np.asarray(ids))
    j = A.join_arr(dep, A.arr_frame(arr_raw))
    sp = (dep.mvt - dep.sched).to_numpy(dtype="float64")
    rule = A.sub_rule(sp, dep.sched, j.arr_lobt, j.arr_aobt3)
    m = np.isin(rule, ["ord", "fill"])
    if not m.any():
        return rule, np.array([], dtype="float64"), np.array([], dtype="float64")
    vals = deps.arc_values(rule[m], dep.mvt.to_numpy()[m], j.arr_aobt3.to_numpy()[m], sp[m])
    return rule, dep.MVT_ID_mvt.to_numpy(dtype="float64")[m], np.asarray(vals, dtype="float64")


def lirf_tail(deps, ids, v12, arc, arn, p_e2, p_w, sp, nf, day) -> dict:
    """v12 -> ARC (v17) -> ARN on rows ARC did not rule (v21) -> E2 (v31) -> E2W (v33), all on the LIRF rows."""
    v17, arc_mask = overwrite(v12, ids, arc[0], arc[1])
    keep = ~np.isin(np.asarray(arn[0], dtype="float64"), np.asarray(arc[0], dtype="float64"))
    v21, arn_mask = overwrite(v17, ids, np.asarray(arn[0])[keep], np.asarray(arn[1])[keep])
    v31, S_e2 = e2_apply(deps, v12, v21, p_e2, sp, nf, day)
    v33, S_w = e2w_apply(deps, v21, v31, p_w, sp, nf, day)
    return {"v17": v17, "v21": v21, "v31": v31, "v33": v33, "n_arc": int(arc_mask.sum()), "n_arn": int(arn_mask.sum()),
            "n_arn_skipped_arc": int((~keep).sum()), "n_e2": int(S_e2.sum()), "n_e2w": int(S_w.sum())}


def assemble_chain(is_lirf, s1, lirf: dict, other: dict) -> dict:
    """{after_vN: value on every U row}: each version replaces only its own lane's rows of its predecessor."""
    L = np.asarray(is_lirf, dtype=bool)
    n = len(L)
    cur = np.rint(np.asarray(s1, dtype="float64"))
    if len(cur) != n:
        raise ValueError(f"S1 has {len(cur)} rows, the lane mask {n}")
    out = {"after_v4": cur.copy()}
    for ver in CHAIN[1:]:
        lane_rows, src = (L, lirf) if ver in LIRF_STEPS else (~L, other)
        vals = np.asarray(src[ver], dtype="float64")
        if len(vals) != int(lane_rows.sum()):
            raise ValueError(f"{ver}: {len(vals)} values for {int(lane_rows.sum())} lane rows")
        cur = cur.copy()
        cur[lane_rows] = vals
        out[f"after_{ver}"] = cur
    return out


def compare(got, want, what: str, tol: float = 0.0, ids=None) -> dict:
    """PASS iff max |got - want| <= tol (NaN on either side is a mismatch)."""
    got, want = np.asarray(got, dtype="float64"), np.asarray(want, dtype="float64")
    if got.shape != want.shape:
        return {"what": what, "verdict": "FAIL", "reason": f"shape {got.shape} vs {want.shape}"}
    d = np.abs(got - want)
    bad = ~(d <= tol)
    rec = {"what": what, "n": int(len(got)), "n_diff": int(bad.sum()), "tol": tol,
           "max_abs": float(np.nanmax(d)) if len(d) and np.isfinite(d).any() else 0.0,
           "verdict": "PASS" if not bad.any() else "FAIL"}
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        rec["first"] = {"row": i, "id": None if ids is None else float(np.asarray(ids)[i]),
                        "got": float(got[i]), "want": float(want[i])}
    return rec


def check_output(frame: pd.DataFrame) -> None:
    """The comparator's schema: exact columns, unique ids, finite chain values >= 1, integer-valued."""
    if list(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"columns {list(frame.columns)} != {OUTPUT_COLUMNS}")
    if not frame.MVT_ID_mvt.is_unique:
        raise ValueError("duplicate MVT_ID_mvt")
    vals = frame[list(AFTER) + ["pred"]].to_numpy(dtype="float64")
    if not (np.isfinite(vals).all() and (vals >= 1).all() and (vals == np.rint(vals)).all()):
        raise ValueError("a chain value is non-finite, below 1 s or not a whole second")
    if not np.array_equal(frame.pred.to_numpy(), frame.after_v33.to_numpy()):
        raise ValueError("pred must equal after_v33")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_frozen(root=ROOT, frozen=FROZEN) -> dict:
    """Every frozen input still has its registered full sha256; AssertionError naming the first that does not."""
    got = {}
    for rel, want in frozen.items():
        s = sha256(pathlib.Path(root) / rel)
        if s != want:
            raise AssertionError(f"{rel}: sha {s[:16]} != registered {want[:16]}")
        got[rel] = s
    return got


def check_registered(prereg=PREREG, hashlog=HASHLOG) -> str:
    """The registration's current sha must appear in the hash log (the run refuses an unregistered or edited plan)."""
    s = sha256(prereg)
    if not pathlib.Path(hashlog).exists() or f"(full {s})" not in pathlib.Path(hashlog).read_text():
        raise AssertionError(f"registration sha {s[:8]} is not in {pathlib.Path(hashlog).name}; hash it before any run")
    return s


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3        # macOS: bytes


def guard_rss(limit_gib: float, stage: str, log=print) -> float:
    peak = peak_rss_gib()
    log(f"[rss] {stage}: peak {peak:.2f} GiB (limit {limit_gib})")
    if peak > limit_gib:
        raise MemoryError(f"{stage}: peak RSS {peak:.2f} GiB > {limit_gib} GiB")
    return peak


# =============================================================================================== deps
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_deps() -> SimpleNamespace:
    """Bind the shipped modules. Imported, never edited; lightgbm loads lazily inside rome_fill.fit_predict_fill."""
    for p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "analysis"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import arc_ship_build as ACS
    import arr_row_clocks as A
    import congestion_anchor_ship as ca
    import mau_build as MB
    import mxi_build as XI
    import mxj_build as XJ
    import rome_arn as R
    importlib.import_module("rome_bandfloor")   # the shipped floor must import; e1_floor is its rounded form (parity run)
    import rome_dateslip as ds
    import rome_fill as rf
    import rome_local_day as RLD
    import rome_localday as LD
    import unm_congestion as uc
    import unm_congestion_ship as ucs
    bs = uc.bs
    B = _load("rse_build", ROOT / "scripts/analysis/rse_build.py")
    W = _load("e2w_build", ROOT / "scripts/analysis/e2w_build.py")
    WS = _load("e2w_ship", ROOT / "scripts/analysis/e2w_ship.py")
    E = _load("agent_E_rules", ROOT / "data/rome_hunt_2026_09_23/agent_E_rules.py")
    return SimpleNamespace(
        bs=bs, uc=uc, ucs=ucs, A=A, R=R, E=E, W=W, MB=MB,
        fit_unmatched=bs.fit_unmatched, mixture=bs.mixture, nf_hybrid=bs.nf_hybrid, mean_over_seeds=bs.mean_over_seeds,
        schedule_fill=bs.schedule_fill, with_sp_bins=rf.with_sp_bins, rome_rows=rf.rome_rows, fit_predict_fill=rf.fit_predict_fill,
        band_shares=ds.band_shares, ds_apply=ds.apply, e1_floor=LD.e1_floor, apply_rld=RLD.apply_rld,
        fit_congestion_regressor=uc.fit_congestion_regressor, stopping_split=MB.stopping_split, fit_body=XJ.fit_body,
        body_params=XI.body_params, anchored_values=ca.anchored_values, day_anomaly=ca.day_anomaly,
        arc_values=ACS.arc_values, arn_rule_rows=R.rule_rows, rule_E2=E.rule_E2, stand_inbound=E.stand_inbound_features,
        e2_form=WS.e2_form, design=W.design, predict=W.predict, fit_brier_w=W.fit_brier_w, E2_BETA=W.E2_BETA,
        row_set=B.row_set, candidate=B.candidate, day_energy=B.day_energy, apply_day_check=B.apply_day_check)


# =============================================================================================== E2 / E2W inputs
def e2w_training(deps, rows: pd.DataFrame, p_r, nf, cls) -> pd.DataFrame:
    """The E2W fit table (e2w_ship.load_2025's construction): X = design(p_R, k_in, turn), fill, w = (sp - nf)^2, fit = band AND
    not slip. `rows` carry MVT_ID_mvt, month, sp and the stand columns stand_inbound_features reads."""
    f = deps.stand_inbound(rows)
    sp, nf = rows.sp.to_numpy(dtype="float64"), np.asarray(nf, dtype="float64")
    X = deps.design(p_r, f.k_in, f.turn)
    band = (sp >= SP_LO) & (sp <= SP_HI) & f.k_in.notna().to_numpy()
    cls = np.asarray(cls).astype(str)
    out = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    out["MVT_ID_mvt"] = rows.MVT_ID_mvt.to_numpy(dtype="float64")
    out["month"] = rows.month.to_numpy()
    out["fill"] = cls == "fill"
    out["w"] = (sp - nf) ** 2
    out["fit"] = band & (cls != "slip")
    return out


def fit_e2w_beta(deps, table: pd.DataFrame, held) -> np.ndarray:
    """beta_W = fit_brier_w on the fit rows of a training view."""
    t = train_view(table, held, "E2W fit rows")
    t = t[t.fit.to_numpy(dtype=bool)]
    if len(t) == 0:
        raise ValueError("E2W: no fit rows in the training view")
    X = t[[c for c in t.columns if c.startswith("x")]].to_numpy(dtype="float64")
    return np.asarray(deps.fit_brier_w(X, t.fill.to_numpy(dtype=bool), t.w.to_numpy(dtype="float64")), dtype="float64")


def nested_lirf_inputs(deps, data, train_months, held, seeds) -> pd.DataFrame:
    """fold_a only: p_R and nf for the LIRF U rows of every TRAINING month t, each fitted on the other training months
    (held-out months plus t excluded), so beta_W never sees a held-out label, not even through its features."""
    parts = []
    for t in train_months:
        inner = tuple(m for m in train_months if m != t)
        held_t = tuple(held) + (t,)
        te = data.unm[(data.unm.month == t) & (data.unm.ADEP_mvt == LIRF)]
        if te.empty:
            continue
        tr_unm = data.unm[data.unm.month.isin(inner)]
        tr_lirf = data.lirf[data.lirf.month.isin(inner)]
        _, pp = s1_parts(deps, tr_unm, tr_lirf, te, seeds, held_t)
        p = fill_p(deps, tr_unm, tr_lirf, te, inner, seeds, held_t)
        parts.append(pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.to_numpy(dtype="float64"), "p_R": p,
                                   "nf": np.asarray(pp["nf"], dtype="float64")}))
    return pd.concat(parts, ignore_index=True)


# =============================================================================================== one 2025 fold
def attach_anomaly(deps, rows: pd.DataFrame, raw_months: list) -> np.ndarray:
    """A per row from the rows' own months' raw files (congestion_anchor_ship.day_anomaly; MVT / AOBT_3 only, no label)."""
    A = pd.concat([deps.day_anomaly(r) for r in raw_months], ignore_index=True).set_index(["ADEP_mvt", "date"]).A
    date = pd.to_datetime(rows.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    return A.reindex(pd.MultiIndex.from_arrays([rows.ADEP_mvt.astype(str), date])).to_numpy(dtype="float64")


def replay_fold(deps, data, test_months, train_months, seeds=SEEDS, scheme="fold_a", log=print) -> tuple:
    """One 2025 fold: (frame of test rows with every after_vN column, info). `data`: unm (22,219 rows, witness attached), lirf,
    extra (MAU rows with month), raw(month) -> raw frame, csf (csf2 rows: MVT_ID_mvt, month, cls, p_R, nf)."""
    held = tuple(test_months)
    unm = data.unm
    te = unm[unm.month.isin(held)].reset_index(drop=True)
    tr_unm = unm[unm.month.isin(train_months)]
    tr_lirf = data.lirf[data.lirf.month.isin(train_months)]
    L = (te.ADEP_mvt == LIRF).to_numpy()
    info: dict = {"test_months": list(held), "train_months": list(train_months), "n_test": int(len(te)), "n_lirf": int(L.sum())}

    s1, parts = s1_parts(deps, tr_unm, tr_lirf, te, seeds, held)
    te_l, te_o = te[L].reset_index(drop=True), te[~L].reset_index(drop=True)
    nf_l = np.asarray(parts["nf"], dtype="float64")[L]
    p_r = fill_p(deps, tr_unm, tr_lirf, te_l, train_months, seeds, held)
    lr = lirf_rules(deps, tr_unm, p_r, te_l, nf_l, held)
    log(f"[{scheme} {held}] S1 + R2 + E1 + RLD done")

    bodies = other_bodies(deps, tr_unm, te_o, data.extra[data.extra.month.isin(train_months)], seeds, held)
    raws = {m: data.raw(m) for m in held}
    anomaly = attach_anomaly(deps, te_o, [raws[m] for m in held])
    oth = compose_other(deps, np.asarray(parts["p"])[~L], np.asarray(parts["sp"])[~L],
                        np.asarray(parts["nf_cells"])[~L], bodies, anomaly)
    log(f"[{scheme} {held}] S1C + SCA + MAU + MXI + MJS done")

    ids_l = te_l.MVT_ID_mvt.to_numpy(dtype="float64")
    arc_ids, arc_vals, arn_ids, arn_vals, rules = [], [], [], [], np.empty(0, dtype=object)
    for m in held:
        mm = (te_l.month == m).to_numpy()
        rule, ai, av = arc_rows(deps, raws[m], raws[m], ids_l[mm])
        rules = np.concatenate([rules, rule.astype(object)])
        arc_ids.append(ai), arc_vals.append(av)
        rr = deps.arn_rule_rows(raws[m], m)
        arn_ids.append(rr.MVT_ID_mvt.to_numpy(dtype="float64")), arn_vals.append(rr.rule_value.to_numpy(dtype="float64"))
    arc = (np.concatenate(arc_ids), np.concatenate(arc_vals))
    arn_all = (np.concatenate(arn_ids), np.concatenate(arn_vals))
    in_rows = np.isin(arn_all[0], ids_l)
    arn = (arn_all[0][in_rows], arn_all[1][in_rows])
    info["arc_rule_counts"] = {k: int((rules == k).sum()) for k in ("", "none", "ord", "fill")}

    rows_e = te_l.drop(columns=[c for c in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "y") if c in te_l.columns]).assign(p_R=p_r)
    p_e2 = np.asarray(deps.rule_E2(rows_e), dtype="float64")
    table = e2w_inputs(deps, data, scheme, held, train_months, seeds)
    beta_w = fit_e2w_beta(deps, table, held)
    f = deps.stand_inbound(rows_e)
    sp_l = te_l.sp.to_numpy(dtype="float64")
    p_w = np.asarray(deps.e2_form(p_r, f.k_in, f.turn, sp_l, beta_w), dtype="float64")
    day = pd.to_datetime(te_l.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    tail = lirf_tail(deps, ids_l, lr["v12"], arc, arn, p_e2, p_w, sp_l, nf_l, day)
    log(f"[{scheme} {held}] ARC + ARN + E2 + E2W done")

    lirf = {"v7": lr["v7"], "v10": lr["v10"], "v12": lr["v12"], **{k: tail[k] for k in ("v17", "v21", "v31", "v33")}}
    chain = assemble_chain(L, s1, lirf, oth)
    out = pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.to_numpy(dtype="float64"), "y": te.y.to_numpy(dtype="float64"),
                        "month": te.month.to_numpy(dtype="int64"),
                        "doy": pd.to_datetime(te.MVT_TIME_UTC_mvt, utc=True).dt.dayofyear.to_numpy(dtype="int64"),
                        "ap": te.ADEP_mvt.astype(str).to_numpy(),
                        "cln_spent_day": pd.to_datetime(te.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").eq(SPENT_DAY).to_numpy(),
                        **chain})
    out["pred"] = out.after_v33
    diag = pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.to_numpy(dtype="float64"), "S1": s1,
                         "p_hat": np.asarray(parts["p"], dtype="float64"), "nf_cells": np.asarray(parts["nf_cells"], dtype="float64"),
                         "nf_fit": np.asarray(parts["nf_fit"], dtype="float64")})
    lirf_diag = pd.DataFrame({"MVT_ID_mvt": ids_l, "p_R": p_r, "nf": nf_l, "p_E2": p_e2, "p_W": p_w, "R2_mix": lr["v7"]})
    other_diag = pd.DataFrame({"MVT_ID_mvt": te_o.MVT_ID_mvt.to_numpy(dtype="float64"), "S1C": oth["S1C"]})
    info.update({k: tail[k] for k in ("n_arc", "n_arn", "n_arn_skipped_arc", "n_e2", "n_e2w")}, beta_w=beta_w.tolist(),
                dateslip_q=lr["q"], dateslip_m=lr["m"])
    return out, {"info": info, "diag": diag, "lirf": lirf_diag, "other": other_diag}


def e2w_inputs(deps, data, scheme, held, train_months, seeds) -> pd.DataFrame:
    """The E2W fit table for a fold. lomo / serve: the stored csf2 p_R / nf (e2w_build's own inputs, the recorded harness);
    fold_a: nested out-of-fold p_R / nf (`nested_lirf_inputs`)."""
    rows = data.unm[(data.unm.ADEP_mvt == LIRF) & data.unm.month.isin(train_months)].reset_index(drop=True)
    csf = data.csf.set_index("MVT_ID_mvt").reindex(rows.MVT_ID_mvt.to_numpy(dtype="float64"))
    if csf.cls.isna().any():
        raise ValueError(f"{int(csf.cls.isna().sum())} LIRF training rows have no csf2 label class")
    if scheme == "fold_a":
        nest = nested_lirf_inputs(deps, data, train_months, held, seeds).set_index("MVT_ID_mvt") \
            .reindex(rows.MVT_ID_mvt.to_numpy(dtype="float64"))
        p_r, nf = nest.p_R.to_numpy(dtype="float64"), nest.nf.to_numpy(dtype="float64")
    else:
        p_r, nf = csf.p_R.to_numpy(dtype="float64"), csf.nf.to_numpy(dtype="float64")
    if not (np.isfinite(p_r).all() and np.isfinite(nf).all()):
        raise ValueError("E2W training rows with a missing p_R or nf")
    return e2w_training(deps, rows, p_r, nf, csf.cls.to_numpy())


# =============================================================================================== real-data loaders
def load_2025(deps, log=print) -> SimpleNamespace:
    """unm (cache, witness from unm_physics_lomo as mau_build attaches it), lirf, extra (mau_build.extra_rows per month), raw(m),
    csf. No file is read beyond what the chain reads."""
    # derive()'s sp bins are dropped by the cache (rome_fill.load_frames) and re-attached here exactly as the shipped loaders do
    # (rome_fill.with_sp_bins, prc/pipeline/ingest.load_rome_frames); fit_unmatched keys its cells on them.
    unm = deps.with_sp_bins(pd.read_parquet(ROOT / "data/cache_rome/unm.parquet"))
    ref = pd.read_parquet(ROOT / "data/cache_stand/unm_physics_lomo.parquet").set_index("MVT_ID_mvt").reindex(unm.MVT_ID_mvt)
    for c in ("y", "sp", "month"):
        if not np.array_equal(unm[c].to_numpy(), ref[c].to_numpy()):
            raise AssertionError(f"unm.parquet and unm_physics_lomo.parquet disagree on {c}")
    for c in deps.uc.WITNESS:
        unm[c] = ref[c].to_numpy()
    lirf = deps.with_sp_bins(pd.read_parquet(ROOT / "data/cache_rome/lirf.parquet"))
    files = {m: _raw_file(m) for m in MONTHS}
    extra = pd.concat([deps.MB.extra_rows(pd.read_parquet(files[m], columns=deps.MB.RAW_COLS)) for m in MONTHS],
                      ignore_index=True)
    if not extra.MVT_ID_mvt.is_unique or extra.MVT_ID_mvt.isin(unm.MVT_ID_mvt).any():
        raise AssertionError("MAU extra rows are not unique or overlap the unmatched rows")
    csf = pd.read_parquet(ROOT / "reports/csf2/csf2_preds.parquet", columns=["MVT_ID_mvt", "month", "cls", "p_R", "nf"])
    cols = sorted(set(deps.R.NEED_COLS_2026) | set(ANOMALY_COLS))     # no TAXITIME / BLOCK: no step needs a label here
    cache: dict = {}

    def raw(m):
        if m not in cache:
            cache.clear()                                   # at most one month resident besides the caller's reference
            cache[m] = pd.read_parquet(files[m], columns=cols)
        return cache[m]
    log(f"2025 frames: unm {len(unm):,}, lirf {len(lirf):,}, extra {len(extra):,}, csf {len(csf):,}")
    return SimpleNamespace(unm=unm, lirf=lirf, extra=extra, raw=raw, csf=csf)


def _raw_file(m: int) -> pathlib.Path:
    hits = sorted(RAW.glob(f"training_2025-{m:02d}-01_*.parquet"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one 2025 training file for month {m}, found {len(hits)}")
    return hits[0]


# =============================================================================================== acceptance (lomo)
def acceptance(out: pd.DataFrame, extras: list, info: list) -> list:
    """A1-A10 on the lomo run (Jan and/or Jul). Each clause names its recorded source."""
    rec = []
    ids = out.MVT_ID_mvt.to_numpy()
    diag = pd.concat([e["diag"] for e in extras]).set_index("MVT_ID_mvt").reindex(ids)
    u = pd.read_parquet(ROOT / "data/cache_stand/unm_physics_lomo.parquet").set_index("MVT_ID_mvt").reindex(ids)
    for c, d in (("p_hat", "p_hat"), ("nf_cells", "nf_cells"), ("nf_fit", "nf_fit"), ("S1", "S1")):
        rec.append(compare(diag[d], u[c], f"A1 {c} vs unm_physics_lomo", 1e-6, ids))
    ld = pd.concat([e["lirf"] for e in extras]).set_index("MVT_ID_mvt")
    ds = pd.read_parquet(ROOT / "data/cache_stand/rome_dateslip_preds.parquet")
    ds = ds[ds.fold == "lomo"].set_index("MVT_ID_mvt").reindex(ld.index)
    rec.append(compare(ld.p_R, ds.p_R, "A3 R2 p vs rome_dateslip_preds lomo p_R", 1e-9, ld.index))
    rec.append(compare(ld.R2_mix, np.rint(ds.R2), "A3 rint R2 vs rome_dateslip_preds lomo R2", 1e-6, ld.index))
    od = pd.concat([e["other"] for e in extras]).set_index("MVT_ID_mvt")
    rec.append(compare(od.S1C, u.reindex(od.index).S1C, "A4 S1C vs unm_physics_lomo", 1e-6, od.index))
    mau = pd.read_parquet(ROOT / "reports/mau_test_predictions.parquet").set_index("MVT_ID_mvt")
    o = out.set_index("MVT_ID_mvt")
    mau = mau[mau.month.isin(out.month.unique())]
    rec.append(compare(o.after_v16.reindex(mau.index), mau.control, "A5 after_v16 vs mau control", 0.0, mau.index))
    rec.append(compare(o.after_v24.reindex(mau.index), mau.candidate, "A6 after_v24 vs mau candidate", 0.0, mau.index))
    mxi = pd.read_parquet(ROOT / "reports/mxi_predictions.parquet")
    mxi = mxi[(mxi.leg == "screen") & mxi.month.isin(out.month.unique())].set_index("MVT_ID_mvt")
    rec.append(compare(o.after_v25.reindex(mxi.index), mxi.it1000, "A6 after_v25 vs mxi it1000", 0.0, mxi.index))
    mxj = pd.read_parquet(ROOT / "reports/mxj_predictions.parquet")
    mxj = mxj[(mxj.leg == "screen") & mxj.month.isin(out.month.unique())].set_index("MVT_ID_mvt")
    rec.append(compare(o.after_v26.reindex(mxj.index), mxj.arm, "A6 after_v26 vs mxj arm", 0.0, mxj.index))
    csf = pd.read_parquet(ROOT / "reports/csf2/csf2_preds.parquet").set_index("MVT_ID_mvt").reindex(ld.index)
    rec.append(compare(ld.p_R, csf.p_R, "A9 replayed p_R vs csf2 p_R", 1e-9, ld.index))
    rec.append({"what": "A7/A8/A9-g/A10 need both months; see acceptance_full", "verdict": "INFO"})
    return rec


def a10_band_compare(got, ref, p_e2, ids, tol: float = 1e-9) -> dict:
    """A10 as registered: lomo p_W vs e2w_build.lomo on the rows where E2 fires (finite p_E2) only. Out-of-band rows are
    dropped, not NaN-masked: compare() counts NaN as a mismatch (BC-24 instance, 2026-09-24)."""
    got, ref, p_e2 = (np.asarray(a, dtype="float64") for a in (got, ref, p_e2))
    band = np.isfinite(p_e2)
    if not band.any():
        raise ValueError("A10: no rows where E2 fires")
    return compare(got[band], ref[band], "A10 lomo p_W (E2 band rows) vs e2w_build.lomo", tol, np.asarray(ids)[band])


def acceptance_full(deps, data, out: pd.DataFrame, extras: list, info: list) -> list:
    """The clauses that need both lomo months: A7 ARC counts (rrh pool: months 1, 2, 7, 8), A8 ARN rows, A9 E2 g, A10 E2W."""
    rec = []
    R = _load("rrh_eval", ROOT / "scripts/analysis/rrh_eval.py")
    rows = R.load_holdout()
    counts = pd.Series(R.arc_rule_for(rows)).value_counts().to_dict()
    want = json.loads((ROOT / "reports/rrh_holdout_report.json").read_text())
    rec.append({"what": "A7 ARC counts (rrh pool) vs rrh arc_counts", "got": {k: int(v) for k, v in counts.items()},
                "want": want["arc_counts"], "verdict": "PASS" if {k: int(counts.get(k, 0)) for k in want["arc_counts"]}
                == want["arc_counts"] else "FAIL"})
    rec.append({"what": "chain ARC counts (serve-faithful own-month pool), reported", "verdict": "INFO",
                "got": [i["arc_rule_counts"] for i in info]})
    arn = pd.DataFrame(json.loads((ROOT / "reports/rome_arn_2026_09_16.json").read_text())["all_months_2025"]["rows"])
    arn = arn[arn.month.isin(HOLDOUT)]
    got = set()
    for m in HOLDOUT:
        rr = deps.arn_rule_rows(data.raw(m), m)
        got |= set(zip(rr.MVT_ID_mvt.astype(float), rr.rule_value.astype(float)))
    want_arn = set(zip(arn.MVT_ID_mvt.astype(float), arn.rule_value.astype(float)))
    rec.append({"what": "A8 ARN Jan/Jul ruled rows vs rome_arn all_months_2025", "got": sorted(got), "want": sorted(want_arn),
                "verdict": "PASS" if got == want_arn else "FAIL"})
    ld = pd.concat([e["lirf"] for e in extras]).set_index("MVT_ID_mvt")
    V = R.verdict_rows(rows.sp, R.arc_rule_for(rows))
    serve = rows.drop(columns=["y", "cls", "nf", "day", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "p_R"])
    serve["p_R"] = ld.p_R.reindex(rows.MVT_ID_mvt.to_numpy(dtype="float64")).to_numpy()
    nf = ld.nf.reindex(rows.MVT_ID_mvt.to_numpy(dtype="float64")).to_numpy()
    p = np.asarray(deps.rule_E2(serve), dtype="float64")
    r = R.score_rule(p[V], *(np.asarray(a)[V] for a in (serve.p_R, rows.sp, nf, rows.y, rows.month, rows.day)), deps.mixture)
    g_want = want["rules"]["agent_E_rules.rule_E2"]["verdict_rows"]["g"]
    rec.append({"what": "A9 E2 verdict-row g vs rrh", "got": r["g"], "want": g_want,
                "verdict": "PASS" if abs(r["g"] - g_want) <= 1e-6 * abs(g_want) else "FAIL"})
    table = e2w_inputs(deps, data, "lomo", (), MONTHS, SEEDS)
    X = table[[c for c in table.columns if c.startswith("x")]].to_numpy(dtype="float64")
    ref = deps.W.lomo(X, table.fill.to_numpy(), table.w.to_numpy(), table.month.to_numpy(), deps.fit_brier_w,
                      table.fit.to_numpy())
    refs = pd.Series(ref, index=table.MVT_ID_mvt).reindex(ld.index)
    rec.append(a10_band_compare(ld.p_W.to_numpy(), refs.to_numpy(), ld.p_E2.to_numpy(), ld.index.to_numpy()))
    beta = fit_e2w_beta(deps, table, ())
    want_b = json.loads((ROOT / "reports/e2w_report.json").read_text())["beta_W_all12"]
    rec.append(compare(beta, want_b, "A10b beta_W (12 months) vs e2w_report beta_W_all12", 1e-6))
    return rec


# =============================================================================================== serve (parity)
def replay_serve(deps, data, seeds=SEEDS, log=print) -> tuple:
    """The chain on the 5,290 scored 2026 U rows, trained on all twelve 2025 months, no 2026 label read."""
    bs = deps.bs
    template = pd.read_parquet(RAW / "submitting.parquet")
    ranking = bs.derive(bs.load_movements([RAW / "ranking.parquet"]))
    serve_frame = ranking[deps.ucs.WITNESS_COLUMNS].copy()
    scored = ranking[ranking.MVT_ID_mvt.isin(set(template.MVT_ID_mvt))].reset_index(drop=True)
    te = scored[scored.unmatched.to_numpy()].drop(columns=["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "y", "delta"],
                                                  errors="ignore").reset_index(drop=True)
    del ranking
    held: tuple = ()
    s1, parts = s1_parts(deps, data.unm, data.lirf, te, seeds, held)
    L = (te.ADEP_mvt == LIRF).to_numpy()
    te_l, te_o = te[L].reset_index(drop=True), te[~L].reset_index(drop=True)
    nf_l = np.asarray(parts["nf"], dtype="float64")[L]
    p_r = fill_p(deps, data.unm, data.lirf, te_l, MONTHS, seeds, held)
    lr = lirf_rules(deps, data.unm, p_r, te_l, nf_l, held)
    te_w, _ = deps.ucs.attach_serve_witness(te_o, serve_frame)
    bodies = other_bodies(deps, data.unm, te_w, data.extra, seeds, held)
    rank = pd.read_parquet(RAW / "ranking.parquet", columns=sorted(set(deps.R.NEED_COLS_2026) | set(ANOMALY_COLS)))
    anomaly = attach_anomaly(deps, te_o, [rank])
    oth = compose_other(deps, np.asarray(parts["p"])[~L], np.asarray(parts["sp"])[~L],
                        np.asarray(parts["nf_cells"])[~L], bodies, anomaly)
    log("[serve] S1, R2 chain, bodies done")
    ids_l = te_l.MVT_ID_mvt.to_numpy(dtype="float64")
    _, ai, av = arc_rows(deps, rank[rank.PHASE_mvt == "DEP"], rank, ids_l)
    rr = deps.R.rows_2026()
    arn = (rr.MVT_ID_mvt.to_numpy(dtype="float64"), rr.rule_value.to_numpy(dtype="float64"))
    rows_e = te_l.assign(p_R=p_r)
    p_e2 = np.asarray(deps.rule_E2(rows_e), dtype="float64")
    beta_w = fit_e2w_beta(deps, e2w_inputs(deps, data, "serve", held, MONTHS, seeds), held)
    f = deps.stand_inbound(rows_e)
    sp_l = te_l.sp.to_numpy(dtype="float64")
    p_w = np.asarray(deps.e2_form(p_r, f.k_in, f.turn, sp_l, beta_w), dtype="float64")
    day = pd.to_datetime(te_l.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    tail = lirf_tail(deps, ids_l, lr["v12"], (ai, av), arn, p_e2, p_w, sp_l, nf_l, day)
    lirf = {"v7": lr["v7"], "v10": lr["v10"], "v12": lr["v12"], **{k: tail[k] for k in ("v17", "v21", "v31", "v33")}}
    chain = assemble_chain(L, s1, lirf, oth)
    out = pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.to_numpy(dtype="float64"), "ap": te.ADEP_mvt.astype(str).to_numpy(), **chain})
    info = {k: tail[k] for k in ("n_arc", "n_arn", "n_arn_skipped_arc", "n_e2", "n_e2w")}
    info["beta_w"] = beta_w.tolist()
    return out, info


def parity(out: pd.DataFrame) -> list:
    """after_vN == shipped vN on every scored U row, exactly."""
    rec = []
    ids = out.MVT_ID_mvt.to_numpy(dtype="float64")
    for ver in CHAIN:
        sub = pd.read_parquet(ROOT / "submissions" / SUBMISSIONS[ver]).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
        want = sub.reindex(ids).to_numpy(dtype="float64")
        rec.append(compare(out[f"after_{ver}"], want, f"parity {ver}", 0.0, ids))
    return rec


# =============================================================================================== CLI
def _log_to(path):
    fh = open(path, "a", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
    return log


def write_output(frame: pd.DataFrame, sidecar: dict, out=OUT, side=SIDECAR) -> None:
    check_output(frame)
    with open(out, "xb") as fh:
        frame.to_parquet(fh, index=False)
    back = pd.read_parquet(out)
    pd.testing.assert_frame_equal(back, frame)
    sidecar["output_sha256"] = sha256(out)
    with open(side, "x") as fh:
        json.dump(sidecar, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="lomo, January only, A1-A6 + A9 p_R; <= 5 min; writes logs/urp_scratch/")
    g.add_argument("--run", action="store_true", help="lomo acceptance, serve parity, fold_a, then the artifact")
    args = ap.parse_args(argv)
    mode = "smoke" if args.smoke else "run"
    limit = RSS_LIMIT_GIB[mode]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log = _log_to(SCRATCH / f"urp_{mode}_{stamp}.log")
    reg = check_registered()
    frozen = check_frozen()
    if mode == "run" and (OUT.exists() or SIDECAR.exists()):
        raise FileExistsError(f"{OUT.name} or its sidecar exists; no silent rerun")
    if mode == "smoke":
        signal.alarm(SMOKE_ALARM_S)
    t0 = time.monotonic()
    log(f"URP {mode}: registration {reg[:8]}, frozen inputs verified ({len(frozen)})")
    deps = load_deps()
    data = load_2025(deps, log)
    guard_rss(limit, "load", log)
    report: dict = {"prereg_sha256": reg, "mode": mode, "frozen": frozen,
                    "code_sha256": {p: sha256(ROOT / p) for p in CODE_FILES}, "steps": [dataclasses.asdict(s) | {"label": s.label}
                                                                                    for s in STEPS]}
    lomo = folds("lomo")[:1] if mode == "smoke" else folds("lomo")
    outs, extras, infos = [], [], []
    for test_m, train_m in lomo:
        o, e = replay_fold(deps, data, test_m, train_m, SEEDS, "lomo", log)
        outs.append(o), extras.append(e), infos.append(e["info"])
        guard_rss(limit, f"lomo {test_m}", log)
    lo = pd.concat(outs, ignore_index=True)
    acc = acceptance(lo, extras, infos)
    if mode == "run":
        acc += acceptance_full(deps, data, lo, extras, infos)
    report["acceptance"] = acc
    report["lomo_info"] = infos
    report["lomo_sse_by_step"] = {c: float(((lo.y - lo[c]) ** 2).sum() / BOARD_N) for c in AFTER}
    failed = [r["what"] for r in acc if r["verdict"] == "FAIL"]
    log(f"acceptance: {sum(r['verdict'] == 'PASS' for r in acc)} PASS, {len(failed)} FAIL {failed}")
    if mode == "smoke":
        report["seconds"], report["peak_rss_gib"] = time.monotonic() - t0, peak_rss_gib()
        (SCRATCH / f"urp_smoke_{stamp}.json").write_text(json.dumps(report, indent=1, default=float))
        signal.alarm(0)
        return 0 if not failed else 3
    if failed:
        report["status"] = f"ACCEPTANCE FAILED: {failed}"
        (SCRATCH / f"urp_run_{stamp}.json").write_text(json.dumps(report, indent=1, default=float))
        return 3
    so, sinfo = replay_serve(deps, data, SEEDS, log)
    guard_rss(limit, "serve", log)
    par = parity(so)
    par.append(compare(sinfo["beta_w"], json.loads((ROOT / "reports/e2w_report.json").read_text())["beta_W_all12"],
                       "serve beta_W vs e2w_report beta_W_all12", 1e-6))
    report["parity"], report["serve_info"] = par, sinfo
    pfail = [r["what"] for r in par if r["verdict"] == "FAIL"]
    log(f"parity: {len(par) - len(pfail)} PASS, {len(pfail)} FAIL {pfail}")
    (test_m, train_m), = folds("fold_a")
    fo, fe = replay_fold(deps, data, test_m, train_m, SEEDS, "fold_a", log)
    guard_rss(limit, "fold_a", log)
    st = pd.read_parquet(ROOT / "data/cache_stand/stratum_fold_v7_preds.parquet")
    st = st[st.fold == "A"].set_index("MVT_ID_mvt").reindex(fo.MVT_ID_mvt)
    report["acceptance"].append(compare(fe["diag"].S1.to_numpy(), st.S1.to_numpy(), "A2 fold_a S1 vs stratum_fold_v7 A",
                                        1e-6, fo.MVT_ID_mvt.to_numpy()))
    dsa = pd.read_parquet(ROOT / "data/cache_stand/rome_dateslip_preds.parquet")
    dsa = dsa[dsa.fold == "A"].set_index("MVT_ID_mvt").reindex(fe["lirf"].MVT_ID_mvt)
    report["acceptance"].append(compare(fe["lirf"].p_R.to_numpy(), dsa.p_R.to_numpy(), "A3 fold_a R2 p vs rome_dateslip A",
                                        1e-9, fe["lirf"].MVT_ID_mvt.to_numpy()))
    report["fold_a_info"] = fe["info"]
    report["fold_a_sse_by_step"] = {c: float(((fo.y - fo[c]) ** 2).sum() / BOARD_N) for c in AFTER}
    a2fail = [r["what"] for r in report["acceptance"][-2:] if r["verdict"] == "FAIL"]
    report["status"] = "OK" if not (pfail or a2fail) else f"PARITY/A2-A3 FAILED: {pfail + a2fail}"
    report["seconds"], report["peak_rss_gib"] = time.monotonic() - t0, peak_rss_gib()
    fo = fo[OUTPUT_COLUMNS]
    write_output(fo, report)
    log(f"wrote {OUT.name} ({len(fo):,} rows); status {report['status']}; {report['seconds']:.0f} s, peak {report['peak_rss_gib']:.2f} GiB")
    return 0 if report["status"] == "OK" else 4


CODE_FILES = ("scripts/analysis/urp_replay.py", "scripts/build_submission.py", "scripts/unm_congestion.py",
              "scripts/unm_congestion_ship.py", "scripts/rome_fill.py", "scripts/rome_dateslip.py", "scripts/rome_localday.py",
              "scripts/rome_local_day.py", "scripts/rome_bandfloor.py", "scripts/analysis/congestion_anchor_ship.py",
              "scripts/analysis/arr_row_clocks.py", "scripts/analysis/arc_ship_build.py", "scripts/analysis/rome_arn.py",
              "scripts/analysis/mau_build.py", "scripts/analysis/mxi_build.py", "scripts/analysis/mxj_build.py",
              "scripts/analysis/rse_build.py", "scripts/analysis/e2w_build.py", "scripts/analysis/e2w_ship.py",
              "scripts/analysis/rrh_eval.py")


if __name__ == "__main__":
    sys.exit(main())
