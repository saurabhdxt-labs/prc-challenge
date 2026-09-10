"""`scripts/unm_congestion_ship.py` -- ship arm E3C into a base submission (owner decision 2026-09-10).

    python scripts/unm_congestion_ship.py --base submissions/merry-quicksand_vN.parquet --version M

E3C = S1C, the shipped unmatched mixture with a congestion-aware body: S1's `NonFillRegressor` with the
airport-hour witness (`hprox_med`, `hprox_n`, scripts/unm_congestion.py) appended to its numeric features,
nothing else changed. Registered in plans/PREREG_unm_congestion_2026_09_10.md; RESULT E3C (2026-09-10 15:43)
returned WORKING on all seven clauses. Ship rule, quoted: "v_next = the best board version with the non-LIRF
unmatched rows replaced by S1C (trained on all twelve 2025 months, 2026 witness), behind a guard that the
shipped S1 path rebuilds the base on every non-LIRF unmatched row exactly (the `rome_ship.py` pattern)."

Training: the twelve 2025 months through the shipped recipe with the 2025 witness attached
(unm_congestion.load_real); the body is fitted on the same non-fill rows `fit_unmatched` uses
(`train[~schedule_fill(train)]`), over the same seeds. Serving: the 2026 witness is computed from EVERY
matched DEP row of data/raw/ranking.parquet (Amendment E3C.1 item 4) and attached to the scored unmatched
rows inside `build()`, which refuses rows that arrive with a witness already attached (a 2025 witness can
never leak in) and refuses a serve witness that covers no scored airport-hour (the wrong file).

Guards, in order, each refusing the write:
  1. S1's refit parts (build_submission.fit_unmatched, hybrid, the shipped seeds) must rebuild the BASE
     submission's value on EVERY scored NON-LIRF unmatched row exactly -- those rows are v4's S1 carried
     through v5-v8 (v5/v6 touched matched rows, v7/v8 LIRF unmatched rows). LIRF rows are R2 / E1's and are
     not compared;
  2. only scored non-LIRF unmatched rows change, to rint(S1C) in the base dtype; LIRF unmatched rows and every
     matched row are copied byte for byte (asserted after the splice, not assumed);
  3. build_submission.check_submission on the frame and on the file re-read from disk; an existing file is
     never overwritten.

Outputs: submissions/<team>_vM.parquet, submissions/<team>_vM.meta.json (base, arm, rows replaced, values
differing, per-airport mean shift, RMS shift, 2026 witness bins of the changed rows, sse_shift_vs_base_board_mse)
and submissions/<team>_vM.unm_rows.parquet (MVT_ID, ADEP, month, sp, hprox_med, hprox_n, bin, p, before, after).
The before/after tables by witness bin, by airport and by airport x bin are printed for the owner's eyes.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import unm_congestion as uc  # noqa: E402

bs = uc.bs
ROME = uc.ROME
PREREG = uc.PREREG
ARM = f"E3C S1C on non-LIRF unmatched rows ({PREREG} RESULT E3C, WORKING 2026-09-10)"
DETAIL_COLUMNS = ["MVT_ID_mvt", "ADEP_mvt", "month", "sp", "hprox_med", "hprox_n", "bin", "p", "before", "after"]
WITNESS_COLUMNS = ["ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]


def log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================================
# guard 1: the shipped S1 path rebuilds the base on every non-LIRF unmatched row
# =============================================================================================

def non_lirf_mask(scored_unm: pd.DataFrame) -> np.ndarray:
    """The rows this arm may touch: every scored unmatched row whose airport is not LIRF."""
    return (scored_unm.ADEP_mvt.astype(str) != ROME).to_numpy()


def verify_parts_rebuild_non_lirf(parts: dict, scored_unm: pd.DataFrame, base: pd.DataFrame) -> dict:
    """rome_ship.verify_parts_rebuild restricted to the non-LIRF rows: rint(mixture(p, sp, nf)) from S1's
    parts must equal the base on every scored non-LIRF unmatched row. LIRF rows are R2 / E1's in v7 / v8 and
    are not compared. Raises AssertionError on any miss; returns the counts for the meta record."""
    if not parts.get("hybrid"):
        raise ValueError("the parts are not the hybrid path's (fit_unmatched(..., hybrid=True)); the base is v4's S1 and "
                         "cannot be rebuilt from the cell path")
    m = non_lirf_mask(scored_unm)
    rebuilt = np.rint(bs.mixture(parts["p"], parts["sp"], parts["nf"])).astype("int64")[m]
    ids = scored_unm.MVT_ID_mvt.to_numpy()[m]
    have = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids).to_numpy(dtype="float64")
    if np.isnan(have).any():
        raise AssertionError(f"{int(np.isnan(have).sum())} scored non-LIRF unmatched ids are missing from the base submission")
    bad = int((rebuilt != have.astype("int64")).sum())
    if bad:
        raise AssertionError(f"S1's refit parts do not rebuild the base on {bad} of {len(have)} scored non-LIRF unmatched rows; "
                             "nothing is written")
    return {"n_checked": int(m.sum()), "n_lirf_not_compared": int((~m).sum()), "seeds": list(parts["seeds"])}


# =============================================================================================
# the 2026 witness, attached inside build so it can only come from the serve file
# =============================================================================================

def attach_serve_witness(scored_unm: pd.DataFrame, serve_frame: pd.DataFrame) -> tuple:
    """(scored rows with hprox_med / hprox_n, witness summary). The witness is computed from `serve_frame`'s
    MATCHED departures (unm_congestion.airport_hour_witness) -- every DEP row of the serve file, not only the
    scored ones. Refuses scored rows that already carry witness columns (nothing upstream may attach one) and a
    witness that covers no scored airport-hour (a 2025 witness, or the wrong file)."""
    present = [c for c in uc.WITNESS if c in scored_unm.columns]
    if present:
        raise ValueError(f"the scored rows already carry {present}; build attaches the serve file's witness itself so that "
                         "no other witness can be used")
    missing = [c for c in WITNESS_COLUMNS if c not in serve_frame.columns]
    if missing:
        raise ValueError(f"the serve frame needs {missing} to compute the witness")
    wit = uc.airport_hour_witness(serve_frame)
    if len(wit) == 0:
        raise ValueError("the serve frame has no matched departure; the 2026 witness would be empty")
    rows = uc.attach_witness(scored_unm, wit)
    covered = int((rows.hprox_n.to_numpy() > 0).sum())
    if covered == 0:
        raise ValueError("the serve witness covers none of the scored airport-hours: it is not the scored file's witness")
    summary = {"n_airport_hours": int(len(wit)), "n_serve_matched_rows": int((serve_frame.AOBT_3_flt.notna()
                                                                              & serve_frame.MVT_TIME_UTC_mvt.notna()).sum()),
               "n_scored_covered": covered, "n_scored": int(len(rows)),
               "scored_nan_share": float(rows.hprox_med.isna().mean()), "min_n": uc.MIN_N}
    return rows, summary


# =============================================================================================
# S1C from S1's own parts
# =============================================================================================

def s1c_from_parts(train_unm: pd.DataFrame, scored_w: pd.DataFrame, parts: dict) -> tuple:
    """(S1C, nf_fit_c, by_seed) on the scored rows: mixture(p, sp, nf_hybrid(nf_cells, mean over seeds of the
    congestion body)). p, sp and nf_cells are S1's parts untouched; the seeds are the ones that rebuilt the
    base; the body is fitted on `train[~schedule_fill(train)]`, fit_unmatched's own non-fill rows."""
    for c in uc.WITNESS:
        if c not in train_unm.columns:
            raise ValueError(f"the training rows lack the witness column {c!r}; load them through unm_congestion.load_real")
    seeds = tuple(int(s) for s in parts["seeds"])
    nonfill = train_unm[~bs.schedule_fill(train_unm)]
    by_seed = {s: uc.fit_congestion_regressor(nonfill, s).predict(scored_w) for s in seeds}
    nf_fit_c = bs.mean_over_seeds(by_seed)
    s1c = bs.mixture(parts["p"], parts["sp"], bs.nf_hybrid(parts["nf_cells"], nf_fit_c))
    return s1c, nf_fit_c, by_seed, int(len(nonfill))


# =============================================================================================
# the build
# =============================================================================================

def build(train_unm: pd.DataFrame, scored_unm: pd.DataFrame, parts: dict, base: pd.DataFrame,
          serve_frame: pd.DataFrame) -> tuple:
    """(new submission frame, info). `scored_unm` and `parts` are row-aligned (fit_unmatched's order);
    `scored_unm` arrives WITHOUT witness columns; `serve_frame` is the serve file's DEP rows."""
    rebuild = verify_parts_rebuild_non_lirf(parts, scored_unm, base)                 # guard 1
    m = non_lirf_mask(scored_unm)
    scored_w, wsum = attach_serve_witness(scored_unm, serve_frame)
    s1c, nf_fit_c, by_seed, n_nonfill = s1c_from_parts(train_unm, scored_w, parts)
    te = scored_w[m]
    ids = te.MVT_ID_mvt.to_numpy()
    vals = np.rint(s1c[m])
    if not (np.isfinite(vals).all() and (vals >= 1.0).all()):
        raise AssertionError("S1C produced a non-finite or sub-floor value; nothing is written")
    out = bs.splice_unmatched(base, ids, vals)                                        # guard 2, part 1
    pos = pd.Index(base.MVT_ID_mvt).get_indexer(ids)
    if not np.array_equal(out.TAXITIME_SEC_mvt.to_numpy()[pos].astype("int64"), vals.astype("int64")):
        raise AssertionError("rint(S1C) does not fit the submission dtype")
    changed = np.flatnonzero(out.TAXITIME_SEC_mvt.to_numpy() != base.TAXITIME_SEC_mvt.to_numpy())
    if not set(changed.tolist()) <= set(pos.tolist()):                                 # guard 2, part 2
        raise AssertionError("a row outside the scored non-LIRF unmatched set changed; nothing is written")
    before = base.TAXITIME_SEC_mvt.to_numpy()[pos].astype("float64")
    after = vals.astype("float64")
    moved = after != before
    detail = pd.DataFrame({"MVT_ID_mvt": ids, "ADEP_mvt": te.ADEP_mvt.astype(str).to_numpy(),
                           "month": te.month.to_numpy().astype("int64"), "sp": te.sp.to_numpy(dtype="float64"),
                           "hprox_med": te.hprox_med.to_numpy(dtype="float64"), "hprox_n": te.hprox_n.to_numpy(dtype="int64"),
                           "bin": uc.bin_of(te.hprox_med), "p": np.asarray(parts["p"], dtype="float64")[m],
                           "before": before, "after": after})[DETAIL_COLUMNS]
    shift = after - before
    info = {"rebuild": rebuild, "witness_2026": wsum, "seeds": list(parts["seeds"]), "n_train_nonfill": n_nonfill,
            "n_rows_replaced": int(m.sum()), "n_values_differ": int(moved.sum()),
            "n_routed_to_body": int((np.asarray(parts["nf_cells"], dtype="float64")[m] < bs.T_TAIL_S).sum()),
            "rms_shift_vs_base_s": float(np.sqrt(np.mean(shift ** 2))) if len(shift) else 0.0,
            "mean_shift_s": float(shift.mean()) if len(shift) else 0.0,
            "per_airport_mean_shift_s": {str(k): float(v) for k, v in pd.Series(shift).groupby(detail.ADEP_mvt.to_numpy()).mean().items()},
            "bins_2026_changed_rows": uc.bin_counts(detail.bin.to_numpy()[moved]),
            "bins_2026_replaced_rows": uc.bin_counts(detail.bin.to_numpy()),
            "sse_shift_vs_base_board_mse": float((shift ** 2).sum() / len(base)),
            "nf_fit_c": nf_fit_c[m], "nf_fit_c_by_seed": {s: v[m] for s, v in by_seed.items()},
            "before": before, "after": after, "detail": detail}
    return out, info


# =============================================================================================
# the owner's tables
# =============================================================================================

def shift_tables(detail: pd.DataFrame) -> dict:
    """{'by_bin', 'by_airport', 'by_airport_bin'}: n, before mean, after mean, shift mean, n changed."""
    d = detail.assign(shift=detail.after - detail.before, changed=(detail.after != detail.before).astype("int64"))

    def agg(keys):
        g = d.groupby(keys, observed=True).agg(n=("before", "size"), before=("before", "mean"), after=("after", "mean"),
                                                shift=("shift", "mean"), n_changed=("changed", "sum"))
        if keys == ["bin"] or keys == ["ADEP_mvt", "bin"]:
            order = {b: i for i, b in enumerate(uc.BINS)}
            g = g.reset_index()
            g["_o"] = g["bin"].map(order)
            g = g.sort_values([k for k in keys if k != "bin"] + ["_o"]).drop(columns="_o").set_index(keys)
        return g

    return {"by_bin": agg(["bin"]), "by_airport": agg(["ADEP_mvt"]), "by_airport_bin": agg(["ADEP_mvt", "bin"])}


def render_tables(tables: dict, out=log) -> None:
    for name, title in (("by_bin", "before / after mean by 2026 witness bin"), ("by_airport", "before / after mean by airport"),
                        ("by_airport_bin", "before / after mean by airport x witness bin")):
        t = tables[name]
        out(f"--- {title} (non-LIRF unmatched scored rows) ---")
        out(f"  {'key':22s} {'n':>7s} {'before':>9s} {'after':>9s} {'shift':>8s} {'changed':>8s}")
        for key, r in t.iterrows():
            k = " ".join(str(v) for v in key) if isinstance(key, tuple) else str(key)
            out(f"  {k:22s} {int(r.n):7,} {r.before:9.1f} {r.after:9.1f} {r['shift']:+8.1f} {int(r.n_changed):8,}")


# =============================================================================================
# writing: refuse to overwrite, check the frame and the re-read file, the detail parquet and the meta json
# =============================================================================================

def _git() -> dict:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"git_sha": run("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def write_outputs(out: pd.DataFrame, info: dict, template: pd.DataFrame, version: int, base_name: str,
                  sub_dir: pathlib.Path, t0: float) -> tuple:
    """(dest, meta). check_submission on the frame, refuse an existing file, write, check the re-read file,
    then the detail parquet and the meta json beside it."""
    bs.check_submission(out, template)
    dest = sub_dir / bs.submission_name(version)
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    sub_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    reread = pq.read_table(dest).to_pandas()
    bs.check_submission(reread, template)
    if not np.array_equal(reread.TAXITIME_SEC_mvt.to_numpy(), out.TAXITIME_SEC_mvt.to_numpy()) or \
            not np.array_equal(reread.MVT_ID_mvt.to_numpy(), out.MVT_ID_mvt.to_numpy()):
        raise AssertionError(f"{dest} re-read differs from the frame that was written")
    info["detail"].to_parquet(sub_dir / f"{dest.stem}.unm_rows.parquet", index=False)
    meta = {"version": int(version), "file": dest.name, "base": base_name, "arm": ARM, "prereg": PREREG, **_git(),
            "seeds": info["seeds"], "n_train_nonfill": info["n_train_nonfill"], "rebuild_guard": info["rebuild"],
            "witness_2026": {**info["witness_2026"], "source": "data/raw/ranking.parquet, every matched DEP row (Amendment E3C.1 item 4)"},
            "n_rows_replaced": info["n_rows_replaced"], "n_values_differ": info["n_values_differ"],
            "n_routed_to_body": info["n_routed_to_body"], "mean_shift_s": info["mean_shift_s"],
            "rms_shift_vs_base_s": info["rms_shift_vs_base_s"], "per_airport_mean_shift_s": info["per_airport_mean_shift_s"],
            "bins_2026_changed_rows": info["bins_2026_changed_rows"], "bins_2026_replaced_rows": info["bins_2026_replaced_rows"],
            "sse_shift_vs_base_board_mse": info["sse_shift_vs_base_board_mse"],
            "detail_file": f"{dest.stem}.unm_rows.parquet", "wall_s": round(time.time() - t0, 1)}
    (sub_dir / f"{dest.stem}.meta.json").write_text(json.dumps(meta, indent=1))
    return dest, meta


# =============================================================================================
# main
# =============================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the board version to build on, e.g. submissions/merry-quicksand_v8.parquet")
    ap.add_argument("--version", type=int, required=True, help="the new version M; writes submissions/<team>_vM.parquet")
    a = ap.parse_args(argv)
    t0 = time.time()
    raw = ROOT / "data" / "raw"
    base_path = ROOT / a.base
    if not base_path.exists():
        raise SystemExit(f"--base {base_path} does not exist")
    base = pq.read_table(base_path).to_pandas()
    template = pq.read_table(raw / "submitting.parquet").to_pandas()
    bs.check_submission(base, template)
    log(f"base {base_path.name}: {len(base):,} rows, dtype {base.TAXITIME_SEC_mvt.dtype}; template {len(template):,}")

    files = sorted(glob.glob(str(raw / "training_2025-*.parquet")))
    if len(files) != 12:
        raise SystemExit(f"expected the 12 training months under {raw}, found {len(files)}")
    unm, lirf, wsum25 = uc.load_real(files, log)
    log(f"2025: {len(unm):,} unmatched rows with the witness ({wsum25['unmatched_bin_counts']}), {len(lirf):,} LIRF matched rows")

    ranking = bs.derive(bs.load_movements([raw / "ranking.parquet"]))
    serve_frame = ranking[WITNESS_COLUMNS].copy()                       # every DEP row of the serve file
    scored = ranking[ranking.MVT_ID_mvt.isin(set(template.MVT_ID_mvt))].reset_index(drop=True)
    if len(scored) != len(template):
        raise ValueError(f"joined {len(scored)} scored rows, template has {len(template)}")
    scored_unm = scored[scored.unmatched.to_numpy()]
    log(f"ranking: {len(ranking):,} DEP rows, {int((~ranking.unmatched).sum()):,} matched; scored unmatched {len(scored_unm):,}, "
        f"non-LIRF {int(non_lirf_mask(scored_unm).sum()):,}")
    del ranking
    gc.collect()

    parts: dict = {}
    bs.fit_unmatched(unm, scored_unm, train_matched=lirf, hybrid=True, seeds=bs.SEEDS, parts=parts)
    out, info = build(unm, scored_unm, parts, base, serve_frame)
    log(f"guard 1: S1 rebuilds {base_path.name} on all {info['rebuild']['n_checked']:,} scored non-LIRF unmatched rows "
        f"({info['rebuild']['n_lirf_not_compared']:,} LIRF rows not compared)")
    log(f"2026 witness: {info['witness_2026']}")
    log(f"S1C: replaced {info['n_rows_replaced']:,} rows, {info['n_values_differ']:,} values differ, routed to the body "
        f"{info['n_routed_to_body']:,}; mean shift {info['mean_shift_s']:+.1f} s, RMS shift {info['rms_shift_vs_base_s']:.1f} s, "
        f"sse shift {info['sse_shift_vs_base_board_mse']:.1f} board MSE; changed-row bins {info['bins_2026_changed_rows']}")
    render_tables(shift_tables(info["detail"]))
    dest, meta = write_outputs(out, info, template, a.version, base_path.name, ROOT / "submissions", t0)
    print(json.dumps(meta, indent=1))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
