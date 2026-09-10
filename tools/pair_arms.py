"""Pair the `treatment` columns of TWO fold records on identical rows.

Usage:
    pair_arms.py NEW_JSON NEW_PARQUET REF_JSON REF_PARQUET [--n-boot 2000] [--n-control 200]
                 [--sd-floor 0.291] [--out reports/pair_<new>_vs_<ref>.json]

The decisional instrument of plans/PREREG_fw_combined_2026_09_10.md, H-FW2: arm FW's treatment
against arm F's treatment. It never invents a threshold - the four clauses below are the prereg's.

Three guards run BEFORE any number is computed, each one a BC-2 tripwire (reports/bug_classes.md):
  1. ROW IDENTITY - row, month, ap, y, proxy and the baseline column must be bit-identical between
     the two records, or the pairing is VOID (exit 3). Two runs of lgbm_fold.py's block runner read
     the same queue record, so their baselines are the same bytes; if they are not, the runs are not
     on the same footing and no looser check is substituted.
  2. REPRODUCE FROM OWN JSON - each record's baseline and treatment, recovered as max(proxy - col, 1)
     (the DELTA convention), must reproduce the RMSE its own JSON states (exit 2). This is also the
     scale-convention check: a taxi-time column read as a delta misses by hundreds of seconds.
  3. SIGN-SWAP NEGATIVE CONTROL - which arm is "new" is swapped on a random half of the rows; the
     gain must centre on zero by construction. The real gain is reported as a multiple of the
     control's p99 |gain|.

Intervals: lgbm_fold.paired_bootstrap (row iid, percentile, seed 0) on the pooled rows, each
|delta| band of lgbm_fold.delta_bands (cut on the TRUE delta), each airport and each month.
"""
import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lgbm_fold as lf  # noqa: E402

IDENTITY_COLS = ("row", "month", "ap", "y", "proxy", "baseline")
#: |recovered RMSE - JSON RMSE| allowed by the reproduce check. The JSON stores float64 at full
#: precision; summation order can differ in the last bits. A convention error is off by >100 s.
REPRODUCE_TOL_S = 1e-6
CONTROL_SEED = 0


class IdentityError(ValueError):
    """The two records are not on the same rows / labels / baseline: the pairing is VOID."""


class ReproduceError(ValueError):
    """A record's column does not reproduce its own JSON RMSE under the DELTA convention."""


def load_record(json_path, parquet_path) -> dict:
    return {"json": json.loads(pathlib.Path(json_path).read_text()), "preds": pd.read_parquet(parquet_path),
            "json_path": str(json_path), "parquet_path": str(parquet_path)}


def recover(preds: pd.DataFrame, col: str) -> np.ndarray:
    """The DELTA convention: y_hat = max(proxy - col, 1) (lgbm_fold.taxi_time's floor)."""
    return np.maximum(preds["proxy"].to_numpy("float64") - preds[col].to_numpy("float64"), 1.0)


def _n_differ(x: np.ndarray, y: np.ndarray) -> int:
    if x.dtype.kind == "f" and y.dtype.kind == "f":
        return int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())
    return int((x != y).sum())


def check_identity(a: pd.DataFrame, b: pd.DataFrame, cols=IDENTITY_COLS) -> None:
    if len(a) != len(b):
        raise IdentityError(f"row count differs: {len(a):,} vs {len(b):,}")
    for c in cols:
        if c not in a.columns or c not in b.columns:
            raise IdentityError(f"identity column {c!r} is missing from a record")
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype != y.dtype:
            raise IdentityError(f"identity column {c!r} has dtype {x.dtype} vs {y.dtype}")
        n = _n_differ(x, y)
        if n:
            raise IdentityError(f"identity column {c!r} differs on {n:,} rows: the pairing is VOID")


def check_reproduces(rec: dict, arms=("baseline", "treatment"), tol=REPRODUCE_TOL_S) -> None:
    p, y = rec["preds"], rec["preds"]["y"].to_numpy("float64")
    for arm in arms:
        want = float(rec["json"]["arms"][arm]["rmse"])
        got = float(np.sqrt(((recover(p, arm) - y) ** 2).mean()))
        if not abs(got - want) <= tol:
            raise ReproduceError(f"{rec['json_path']}: arm {arm!r} recovers to {got:.6f} under the DELTA convention "
                                 f"against its own JSON {want:.6f} - wrong convention, or a different record")


def rule_sd(a_json: dict, b_json: dict, floor=None) -> float:
    """max(both arms' seed_sd.sd, both treatment sds, floor) - plans/PREREG_fw_combined_2026_09_10.md."""
    vals = [v for j in (a_json, b_json) for v in (j.get("seed_sd", {}).get("sd"), j.get("seed_sd", {}).get("treatment_sd"))
            if v is not None]
    if floor is not None:
        vals.append(float(floor))
    if not vals:
        raise ValueError("no seed sd in either record and no floor: clause (ii) cannot be evaluated")
    return float(max(vals))


def sign_swap_control(e_new, e_ref, n_draws=200, seed=CONTROL_SEED) -> dict:
    """Swap the two arms' squared errors on a random half of the rows, n_draws times; the gain
    rmse(ref') - rmse(new') must centre on zero whatever the real difference is."""
    e_new, e_ref = np.asarray(e_new, dtype="float64"), np.asarray(e_ref, dtype="float64")
    rng = np.random.default_rng(seed)
    gains = np.empty(n_draws, dtype="float64")
    for i in range(n_draws):
        m = rng.random(len(e_new)) < 0.5
        a, b = np.where(m, e_ref, e_new), np.where(m, e_new, e_ref)
        gains[i] = np.sqrt(b.mean()) - np.sqrt(a.mean())
    return {"n_draws": int(n_draws), "seed": int(seed), "mean_gain": float(gains.mean()),
            "p99_abs_gain": float(np.percentile(np.abs(gains), 99))}


def _pair(se_new, se_ref, n_boot) -> dict:
    n = len(se_new)
    if n < 2:
        return {"n_rows": int(n), "gain_s": None, "ci95": None}
    p = lf.paired_bootstrap({"new": se_new, "ref": se_ref}, [("new", "ref")], n_boot, lf.BOOT_SEED)["new_vs_ref"]
    return {"n_rows": int(n), "rmse_new": float(np.sqrt(se_new.mean())), "rmse_ref": float(np.sqrt(se_ref.mean())),
            "gain_s": p["gain_s"], "ci95": p["ci95"], "excludes_zero": p["excludes_zero"]}


def compare(new: dict, ref: dict, n_boot=lf.N_BOOT, n_control=200, sd_floor=None) -> dict:
    check_identity(new["preds"], ref["preds"])
    check_reproduces(new)
    check_reproduces(ref)
    p = new["preds"]
    y = p["y"].to_numpy("float64")
    se_new = (recover(p, "treatment") - y) ** 2
    se_ref = (recover(ref["preds"], "treatment") - y) ** 2
    se_base = (recover(p, "baseline") - y) ** 2

    pooled = _pair(se_new, se_ref, n_boot)
    bands = {name: _pair(se_new[m], se_ref[m], n_boot) for name, m in lf.delta_bands(p["delta"]).items()}
    ap, month = p["ap"].to_numpy(), p["month"].to_numpy()
    per_airport = {str(a): _pair(se_new[ap == a], se_ref[ap == a], n_boot) for a in sorted(set(ap))}
    per_month = {str(int(mo)): _pair(se_new[month == mo], se_ref[month == mo], n_boot) for mo in sorted(set(month))}
    control = sign_swap_control(se_new, se_ref, n_control)
    control["real_over_p99"] = (float(pooled["gain_s"] / control["p99_abs_gain"])
                                if control["p99_abs_gain"] > 0 else None)
    sd = rule_sd(new["json"], ref["json"], sd_floor)

    lo = pooled["ci95"][0]
    clauses = {
        "i_interval_excludes_zero": bool(lo > 0),
        "ii_gain_over_2x_sd": bool(pooled["gain_s"] > 2.0 * sd),
        # (iii) the tail is KEPT: the over10 interval must not exclude zero in the REFERENCE's favour
        "iii_tail_kept": bool(bands["over10"]["ci95"] is not None and bands["over10"]["ci95"][1] >= 0),
        "iv_body_bands_gain": bool(all(bands[b]["ci95"] is not None and bands[b]["ci95"][0] > 0
                                       for b in ("lt2", "2to10"))),
    }
    verdict = ("COMBINES" if all(clauses.values()) else
               "NOT WORKING" if not clauses["i_interval_excludes_zero"] else "INCONCLUSIVE")
    rm_base = float(np.sqrt(se_base.mean()))
    return {
        "new": {"json": new["json_path"], "parquet": new["parquet_path"], "mode": new["json"].get("mode")},
        "ref": {"json": ref["json_path"], "parquet": ref["parquet_path"], "mode": ref["json"].get("mode")},
        "n_rows": int(len(p)), "n_boot": int(n_boot), "boot_seed": int(lf.BOOT_SEED),
        "baseline_rmse": rm_base,
        "vs_baseline": {"new_gain_s": rm_base - pooled["rmse_new"], "ref_gain_s": rm_base - pooled["rmse_ref"]},
        "pooled": pooled, "bands": bands, "per_airport": per_airport, "per_month": per_month,
        "control": control, "rule_sd": sd, "sd_floor": sd_floor, "clauses": clauses, "verdict": verdict,
    }


def _fmt(r: dict) -> str:
    if r["gain_s"] is None:
        return f"n {r['n_rows']:7,}  no interval"
    lo, hi = r["ci95"]
    return (f"n {r['n_rows']:7,}  new {r['rmse_new']:8.3f}  ref {r['rmse_ref']:8.3f}  "
            f"gain {r['gain_s']:+7.3f} [{lo:+7.3f}, {hi:+7.3f}]{'*' if r['excludes_zero'] else ''}")


def report(res: dict) -> None:
    print(f"new {res['new']['mode']}  vs  ref {res['ref']['mode']}   rows {res['n_rows']:,}   "
          f"baseline {res['baseline_rmse']:.4f}   vs baseline: new {res['vs_baseline']['new_gain_s']:+.3f}  "
          f"ref {res['vs_baseline']['ref_gain_s']:+.3f}")
    print(f"  pooled   {_fmt(res['pooled'])}")
    for k, v in res["bands"].items():
        print(f"  band {k:7s} {_fmt(v)}")
    for k, v in res["per_airport"].items():
        print(f"  ap {k:9s} {_fmt(v)}")
    for k, v in res["per_month"].items():
        print(f"  month {k:6s} {_fmt(v)}")
    c = res["control"]
    print(f"  sign-swap control: {c['n_draws']} draws, mean {c['mean_gain']:+.4f}, p99 |gain| {c['p99_abs_gain']:.4f}, "
          f"real / p99 = {c['real_over_p99'] if c['real_over_p99'] is None else round(c['real_over_p99'], 1)}")
    print(f"  rule sd {res['rule_sd']:.4f} (floor {res['sd_floor']}) -> bar {2 * res['rule_sd']:.3f} s")
    for k, v in res["clauses"].items():
        print(f"    {k:28s} {'PASS' if v else 'fail'}")
    print(f"  VERDICT (H-FW2): {res['verdict']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new_json"); ap.add_argument("new_parquet"); ap.add_argument("ref_json"); ap.add_argument("ref_parquet")
    ap.add_argument("--n-boot", type=int, default=lf.N_BOOT)
    ap.add_argument("--n-control", type=int, default=200)
    ap.add_argument("--sd-floor", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    new, ref = load_record(a.new_json, a.new_parquet), load_record(a.ref_json, a.ref_parquet)
    try:
        res = compare(new, ref, a.n_boot, a.n_control, a.sd_floor)
    except IdentityError as e:
        print(f"VOID: {e}", file=sys.stderr)
        return 3
    except ReproduceError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = None
    res["git_sha"], res["written"] = sha, dt.datetime.now().isoformat(timespec="seconds")
    out = pathlib.Path(a.out) if a.out else ROOT / "reports" / f"pair_{res['new']['mode']}_vs_{res['ref']['mode']}.json"
    out.write_text(json.dumps(res, indent=1))
    report(res)
    print(f"json -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
