"""PBL — does the 0.5 CatBoost blend still gain when the base is the PRODUCTION incumbent?

Registration: plans/PREREG_pbl_production_anchored_blend_2026_09_20.md (sha256 0ae2edf6).

No model is fitted. This re-scores predictions already on disk against the production composed
incumbent built at scripts/analysis/opc_screen.py:124-129, instead of against `shipped_params`
(scripts/lgbm_submit.py:174-177), which is a LightGBM-only config and not the pipeline.

Units. Two live in this project and they differ by about 2x on these populations:
  * `board_mse_equiv` = SSE / 344,841  -- the unit of lever_sweep/compose_cat.json and of the
    +706.3 under test. THIS ARM'S VERDICT UNIT.
  * `gain_weighted_mse` = (339,551/344,841) * mean per-row gain -- the unit of BLD's +73.60 and
    OPC's +404.68, produced by opc_screen.bootstrap_stats.
Both are reported for every arm, and the two implementations cross-check each other (X1 below).

Mutation rehearsal 2026-09-20, BC-16-safe harness (bytecode purged between mutants), over
`taxi_from_delta`, `blend`, `board_gain`, `trim_mask`, `decide` and `join_predictions`. First pass
11/14 = 78.6 %: three escapes, all tests that passed for the WRONG reason -- the month-CI boundary was
being satisfied by the sign-disagreement clause, the sign clause by the month-CI clause, and the NaN-key
test by whichever of the two asserts fired first. Tests added to separate them; second pass 14/14.
Extended at the end-work gate with the four mutants the first set missed -- both `max(..., 1)` clamps,
squared-error -> absolute-error, and a disabled trim -- **final 18/18 = 100 %**.
`day_block_bootstrap` carries no unit mutant: it is guarded at runtime by X1, an independent second
implementation of the same statistic in the other unit (opc_screen.bootstrap_stats), asserted to 1e-6.

Run:  /opt/homebrew/bin/python3.11 scripts/analysis/pbl_build.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import resource
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/pbl_report.json"

BOARD_ROWS = 344841                    # the fixed normaliser; never the population size
WEIGHT_DENOM = 339551                  # opc_screen.WEIGHT's numerator: 339551/344841
WEIGHT = 0.5                           # registered, fixed, never searched
SEED_SPREAD_BAR = 49.0                 # measured seed spread (handoff section 4)
TRIM_FRAC = 0.01                       # the top-1 % tail rule
INC_RMSE = 208.31040447352203          # opc_screen.py:129
N_ROWS, N_POP = 334132, 161956         # opc_screen.py:127
N_SWEEP_HOLDOUT = 339015               # lever_sweep's eval set
CAT_NPY = ROOT / "reports/lever_sweep/preds/cat_d8.npy"
SHIP_NPY = ROOT / "reports/lever_sweep/preds/shipped_params.npy"
N_BOOT, BOOT_SEED = 2000, 0
CI_Q = (0.025, 0.975)               # the registered 95 % interval


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- pure arithmetic

def taxi_from_delta(proxy: np.ndarray, delta_hat: np.ndarray) -> np.ndarray:
    """taxi = max(proxy - delta_hat, 1). The saved .npy are delta_hat, NOT taxi."""
    return np.maximum(np.asarray(proxy, float) - np.asarray(delta_hat, float), 1.0)


def blend(base: np.ndarray, member: np.ndarray, w: float) -> np.ndarray:
    """max((1-w)*base + w*member, 1) -- the composition v14 shipped."""
    return np.maximum((1.0 - w) * np.asarray(base, float) + w * np.asarray(member, float), 1.0)


def board_gain(y: np.ndarray, base: np.ndarray, cand: np.ndarray) -> float:
    """(SSE_base - SSE_cand) / 344,841. Positive means the candidate is better.

    Divides by the fixed board row count, never by len(y): a shrinking population must not
    flatter a gain.
    """
    y = np.asarray(y, float)
    d = (y - np.asarray(base, float)) ** 2 - (y - np.asarray(cand, float)) ** 2
    return float(d.sum() / BOARD_ROWS)


def trim_mask(base: np.ndarray, cand: np.ndarray, frac: float = TRIM_FRAC) -> np.ndarray:
    """Keep-mask dropping the `frac` rows with the largest ABSOLUTE move; the tail rule."""
    move = np.abs(np.asarray(cand, float) - np.asarray(base, float))
    k = int(round(frac * len(move)))
    keep = np.ones(len(move), bool)
    if k > 0:
        keep[np.argpartition(move, -k)[-k:]] = False
    return keep


def day_block_bootstrap(y, base, cand, day, month) -> dict:
    """Day-blocked bootstrap of the board-MSE-equivalent gain.

    Calendar days are the dependence unit, so days -- not rows -- are resampled, within month,
    exactly as opc_screen.bootstrap_stats does. Each draw is rescaled to the true population size
    so that a resample with more rows than the population cannot inflate the statistic.
    """
    y = np.asarray(y, float)
    gain = (y - np.asarray(base, float)) ** 2 - (y - np.asarray(cand, float)) ** 2
    frame = pd.DataFrame({"gain": gain, "day": np.asarray(day), "month": np.asarray(month)})
    daily = frame.groupby(["month", "day"], sort=True).agg(
        gain=("gain", "sum"), n=("gain", "size")).reset_index()
    rng = np.random.default_rng(BOOT_SEED)
    n_pop = len(gain)
    vals = np.zeros(N_BOOT)
    ns = np.zeros(N_BOOT)
    monthly = {}
    for m in sorted(frame.month.unique()):
        a = daily[daily.month.eq(m)]
        idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
        sums = a.gain.to_numpy()[idx].sum(axis=1)
        counts = a.n.to_numpy()[idx].sum(axis=1)
        vals += sums
        ns += counts
        n_m = int(a.n.sum())
        sel = frame.month.to_numpy() == m
        monthly[int(m)] = {
            "gain": board_gain(y[sel], np.asarray(base, float)[sel], np.asarray(cand, float)[sel]),
            "ci95": (np.quantile(sums / counts, CI_Q) * n_m / BOARD_ROWS).tolist(),
            "n": n_m,
        }
    return {"gain": board_gain(y, base, cand),
            "ci95": (np.quantile(vals / ns, CI_Q) * n_pop / BOARD_ROWS).tolist(),
            "monthly": monthly, "n": int(n_pop)}


def _ci_phrase(ci: list[float]) -> str:
    """Describe WHY a CI fails the `lower bound > 0` bar. A CI entirely below zero is a different
    fact from one straddling zero -- reporting both as "includes 0" understates a wrong-signed
    result, which is how a negative becomes a "not significant" in a write-up.
    """
    kind = "is entirely negative" if ci[1] < 0 else "includes 0"
    return f"{kind}: [{ci[0]:.1f}, {ci[1]:.1f}]"


def _bar_phrase(gain: float) -> str:
    """BC-19 again, in the function that defined the class: "is inside the seed spread" is false of
    -1,964.9, which is not inside anything. The phrase must describe the observation."""
    return "is inside the seed spread" if gain > 0 else "is negative, so it cannot clear the seed spread"


def decide(stats: dict, trimmed_gain: float) -> tuple[str, list[str]]:
    """The registered TRUE shape: all five clauses. Ambiguous = NOT WORKING.

    Every failed clause is collected, not just the first -- a verdict that reported only the first
    failure would hide how far an arm is from its bar.
    """
    why: list[str] = []
    if stats["gain"] <= 0:
        why.append(f"mean gain {stats['gain']:.1f} <= 0")
    if stats["ci95"][0] <= 0:
        why.append(f"pooled CI {_ci_phrase(stats['ci95'])}")
    signs = []
    for m, d in stats["monthly"].items():
        signs.append(np.sign(d["gain"]))
        if d["ci95"][0] <= 0:
            why.append(f"month {m} CI {_ci_phrase(d['ci95'])}")
    if len(set(signs)) > 1:
        why.append("months disagree in sign")
    if stats["gain"] <= SEED_SPREAD_BAR:
        why.append(f"gain {stats['gain']:.1f} {_bar_phrase(stats['gain'])} ({SEED_SPREAD_BAR})")
    if trimmed_gain <= 0:
        why.append(f"gain does not survive the top-1 % trim ({trimmed_gain:.1f})")
    return ("WORKING" if not why else "NOT WORKING"), why


def join_predictions(s: pd.DataFrame, preds: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Exact inner join on `row`, REPORTING the shortfall instead of hiding it.

    BC-15: pandas.merge joins NaN to NaN, so a missing key must never reach the merge.
    """
    assert s.row.notna().all(), "NaN key in s.row (BC-15)"
    assert preds.row.notna().all(), "NaN key in preds.row (BC-15)"
    assert s.row.is_unique, "duplicate row in s"
    assert preds.row.is_unique, "duplicate row in preds"
    out = s.merge(preds, on="row", how="inner", validate="one_to_one")
    return out.reset_index(drop=True), int(len(s) - len(out))


# --------------------------------------------------------------------------- data

def load_incumbent(OPC):
    """opc_screen.load_data() lines 103-129 WITHOUT peer_features, which this arm does not use.

    The guards are what make this the same object: G1 row counts, G2 the asserted incumbent RMSE.
    """
    ID = OPC.ID
    s = OPC.read("data/adsb/v2/v13_matched_oof.parquet",
                 ["row", ID, "ap", "month", "day", "fold", "y", "proxy", "F", "v13_delta",
                  "taxi_v13_delta"])
    h = OPC.read("data/cache_stand/catboost_native_input.parquet",
                 ["row", ID, "holdout", "score_y", "score_proxy", "month", "ap", "proxy"],
                 filters=[("holdout", "=", True)])
    h["C"] = np.load(ROOT / "data/cache_stand/catboost_native_delta.npy")
    overlap = (set(h.columns) & set(s.columns)) - {"row"}
    h = h.rename(columns={c: f"native_{c}" for c in overlap})
    s = OPC.joined(s, h.drop(columns="holdout"), "row")
    for c in [ID, "ap", "month", "proxy"]:
        assert np.array_equal(s[c], s[f"native_{c}"])
    assert np.array_equal(s.y, s.score_y) and np.array_equal(s.proxy, s.score_proxy)
    fw = OPC.read("data/cache_stand/fold_preds_queue_order_weather.parquet",
                  ["row", "y", "proxy", "treatment"])
    fw = fw.rename(columns={"y": "fw_y", "proxy": "fw_proxy", "treatment": "FW"})
    s = OPC.joined(s, fw, "row")
    assert np.array_equal(s.y, s.fw_y) and np.array_equal(s.proxy, s.fw_proxy)
    s = s[np.isfinite(s.fold)].copy().reset_index(drop=True)
    pop = s.v13_delta.eq(s.F).to_numpy()
    assert len(s) == N_ROWS and pop.sum() == N_POP, \
        f"G1 FAILED: {len(s)} rows (want {N_ROWS}), pop {pop.sum()} (want {N_POP})"
    s["taxi_C"] = np.maximum(s.proxy - s.C, 1)
    s["taxi_FW"] = np.maximum(s.proxy - s.FW, 1)
    s["inc"] = np.where(pop, .5 * (s.taxi_C + s.taxi_FW), s.taxi_v13_delta)
    rmse = float(np.sqrt(np.mean((s.y - s.inc) ** 2)))
    assert abs(rmse - INC_RMSE) < 1e-8, f"G2 FAILED: incumbent RMSE {rmse!r} != {INC_RMSE!r}; VOID"
    return s, pop


def load_sweep_predictions() -> pd.DataFrame:
    """`row` ids for the lever_sweep holdout predictions.

    The sweep's 66-feature cache carries NO `row` column, so the ids come from
    catboost_native_input's holdout subset. That alignment is POSITIONAL and therefore guarded:
    y, proxy and month must match element-wise across all 339,015 rows, or nothing is returned.
    """
    import pyarrow.parquet as pq
    cache = ROOT / "data/cache_stand"
    d = pd.concat([pq.read_table(p, columns=["y", "proxy", "delta", "month"]).to_pandas()
                   for p in sorted(cache.glob("training_2025-*.parquet"))], ignore_index=True)
    d = d[d.y.notna() & d.proxy.notna() & d.delta.notna()].reset_index(drop=True)
    ev = d[d.month.isin((1, 7))].reset_index(drop=True)
    h = pq.read_table(cache / "catboost_native_input.parquet",
                      columns=["row", "holdout", "score_y", "score_proxy", "month"]).to_pandas()
    h = h[h.holdout].reset_index(drop=True)
    assert len(ev) == len(h) == N_SWEEP_HOLDOUT, f"G3 FAILED: {len(ev)} vs {len(h)}"
    assert np.array_equal(ev.y.to_numpy(), h.score_y.to_numpy()), "G3 FAILED: y mismatch"
    assert np.array_equal(ev.proxy.to_numpy(), h.score_proxy.to_numpy()), "G3 FAILED: proxy mismatch"
    assert np.array_equal(ev.month.to_numpy(), h.month.to_numpy()), "G3 FAILED: month mismatch"
    out = pd.DataFrame({"row": h.row.to_numpy()})
    for name, path in (("cat_delta", CAT_NPY), ("ship_delta", SHIP_NPY)):
        arr = np.load(path)
        assert len(arr) == N_SWEEP_HOLDOUT, f"{path.name}: {len(arr)} rows"
        out[name] = arr.astype(np.float64)
    # G4: reproduce cat_d8's and shipped_params' published board_mse_equiv from compose_cat.json
    y, px = ev.y.to_numpy(float), ev.proxy.to_numpy(float)
    out.attrs["g4"] = {
        "cat_d8": float(((y - taxi_from_delta(px, out.cat_delta)) ** 2).sum() / BOARD_ROWS),
        "shipped_params": float(((y - taxi_from_delta(px, out.ship_delta)) ** 2).sum() / BOARD_ROWS),
    }
    return out


# --------------------------------------------------------------------------- the arms

def score_arm(d: pd.DataFrame, base: np.ndarray, cand: np.ndarray, OPC) -> dict:
    y = d.y.to_numpy(float)
    stats = day_block_bootstrap(y, base, cand, d.day.to_numpy(), d.month.to_numpy())
    keep = trim_mask(base, cand)
    stats["trimmed_gain"] = board_gain(y[keep], base[keep], cand[keep])
    verdict, why = decide(stats, stats["trimmed_gain"])
    stats["verdict"], stats["why"] = verdict, why
    # X1: an independent second implementation of the same statistic, in the OTHER unit.
    opc = OPC.bootstrap_stats(d, base, cand)
    stats["gain_weighted_mse"] = opc["gain_weighted_mse"]
    stats["gain_rmse_s"] = opc["gain_rmse_s"]
    stats["squared_move_share_gt300"] = opc["squared_move_share_gt300"]
    stats["changed_rows"] = opc["changed_rows"]
    stats["by_airport_gain_weighted_mse"] = opc["by_airport_gain_weighted_mse"]
    # X1: cross-check the CI -- NOT the point estimate. `opc["gain_weighted_mse"] * n / 339551`
    # reduces algebraically to `gain.sum()/344841`, so comparing point estimates can never fail and
    # guards nothing (found at the end-work gate 2026-09-20). The CI bounds, by contrast, come from
    # an independent resampling implementation in opc_screen.bootstrap_stats -- its own groupby, its
    # own rng, its own accumulation -- so agreement there does test this module's bootstrap path.
    k = len(d) / WEIGHT_DENOM
    stats["x1_ci_cross_check_abs_diff"] = [abs(o * k - m) for o, m in zip(opc["ci95"], stats["ci95"])]
    assert max(stats["x1_ci_cross_check_abs_diff"]) < 1e-6, (
        f"X1 FAILED: opc CI {[o * k for o in opc['ci95']]} vs this module's {stats['ci95']} -- "
        "the two bootstrap implementations disagree")
    return stats


def main() -> dict:
    t0 = time.monotonic()
    OPC = _load("opc_screen", "scripts/analysis/opc_screen.py")
    s, pop = load_incumbent(OPC)
    preds = load_sweep_predictions()
    g4 = preds.attrs["g4"]
    s, missing = join_predictions(s, preds)
    pop = s.v13_delta.eq(s.F).to_numpy()          # re-derived after the join, never carried across it
    s["taxi_cat"] = taxi_from_delta(s.proxy.to_numpy(), s.cat_delta.to_numpy())
    s["taxi_ship"] = taxi_from_delta(s.proxy.to_numpy(), s.ship_delta.to_numpy())

    report = {
        "prereg": "plans/PREREG_pbl_production_anchored_blend_2026_09_20.md",
        "prereg_sha256": "0ae2edf66e7c27d3d22c2e6b81618f4c5069d416518dd8c3c7aecd79988f84a9",
        "guards": {"G1_rows": int(len(s)), "G1_pop": int(pop.sum()),
                   "G2_inc_rmse": INC_RMSE,
                   "G3_rows_in_s_without_a_prediction": missing,
                   "G4_cat_d8_board_mse_equiv": g4["cat_d8"],
                   "G4_cat_d8_published": 52550.79,
                   "G4_shipped_params_board_mse_equiv": g4["shipped_params"],
                   "G4_shipped_params_published": 52875.76},
        "unit": "board_mse_equiv = SSE / 344,841",
        "arms": {},
    }
    for gk, gv, pub in (("G4_cat_d8", g4["cat_d8"], 52550.79),
                        ("G4_shipped_params", g4["shipped_params"], 52875.76)):
        assert abs(gv - pub) < 1.0, f"{gk} FAILED: {gv} vs published {pub}"

    for name, mask, label in (("A", ~pop, "coverage>=2 -- production does NOT blend"),
                              ("B", pop, "coverage<2 -- production ALREADY blends")):
        d = s[mask].reset_index(drop=True)
        inc = d.inc.to_numpy(float)
        cand = blend(inc, d.taxi_cat.to_numpy(float), WEIGHT)
        arm = score_arm(d, inc, cand, OPC)
        arm["population"] = label
        arm["n"] = int(len(d))
        # diagnostics: both baselines on the record, and the anchor's contribution to the +706.3
        y = d.y.to_numpy(float)
        ship = d.taxi_ship.to_numpy(float)
        arm["diagnostics"] = {
            "production_inc_board_mse_equiv": float(((y - inc) ** 2).sum() / BOARD_ROWS),
            "shipped_params_board_mse_equiv": float(((y - ship) ** 2).sum() / BOARD_ROWS),
            "production_inc_rmse": float(np.sqrt(np.mean((y - inc) ** 2))),
            "shipped_params_rmse": float(np.sqrt(np.mean((y - ship) ** 2))),
            "cat_d8_standalone_rmse": float(np.sqrt(np.mean((y - d.taxi_cat.to_numpy(float)) ** 2))),
            "cat_d8_standalone_board_mse_equiv": float(
                ((y - d.taxi_cat.to_numpy(float)) ** 2).sum() / BOARD_ROWS),
            "gain_vs_shipped_params_anchor": board_gain(
                y, ship, blend(ship, d.taxi_cat.to_numpy(float), WEIGHT)),
        }
        report["arms"][name] = arm

    report["runtime_s"] = round(time.monotonic() - t0, 1)
    report["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    r = main()
    for k, a in r["arms"].items():
        print(f"\nARM {k}: {a['population']}  n={a['n']:,}")
        print(f"  production base      {a['diagnostics']['production_inc_board_mse_equiv']:>12,.1f}"
              f"   RMSE {a['diagnostics']['production_inc_rmse']:.3f}")
        print(f"  shipped_params base  {a['diagnostics']['shipped_params_board_mse_equiv']:>12,.1f}"
              f"   RMSE {a['diagnostics']['shipped_params_rmse']:.3f}")
        print(f"  GAIN vs production   {a['gain']:>12,.1f}   CI95 "
              f"[{a['ci95'][0]:,.1f}, {a['ci95'][1]:,.1f}]   trimmed {a['trimmed_gain']:,.1f}")
        for m, d in a["monthly"].items():
            print(f"    month {m}            {d['gain']:>12,.1f}   CI95 [{d['ci95'][0]:,.1f}, {d['ci95'][1]:,.1f}]")
        print(f"  gain vs shipped_params anchor (DIAGNOSTIC) "
              f"{a['diagnostics']['gain_vs_shipped_params_anchor']:,.1f}")
        print(f"  VERDICT {a['verdict']}" + ("" if a["verdict"] == "WORKING" else ": " + "; ".join(a["why"])))
    print(f"\nruntime {r['runtime_s']}s  peak RSS {r['peak_rss_gb']} GB  "
          f"rows without a prediction: {r['guards']['G3_rows_in_s_without_a_prediction']}")
