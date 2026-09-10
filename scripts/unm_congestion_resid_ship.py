"""`scripts/unm_congestion_resid_ship.py` -- ship arm E3R into a base submission (only if RESULT E3R is WORKING).

    python scripts/unm_congestion_resid_ship.py --base submissions/merry-quicksand_v10.parquet --version 11

E3R = S1R: S1C's mixture with the body replaced by the offset model (scripts/unm_congestion_resid.py): on the
non-fill 2025 rows WITH a witness, r = clip(y - hprox_med, -3,000, +3,000) is fitted with S1C's features, params,
encodings (on r) and seeds; the served value is hprox_med + r_hat; a scored row without a witness keeps S1C's body.
Registered in plans/PREREG_unm_congestion_resid_2026_09_10.md. Ship rule, quoted: "only if WORKING -- v_next = v10
with the non-LIRF unmatched rows replaced by S1R (trained on all twelve 2025 months, 2026 witness), behind a guard
that the S1C path rebuilds v10's non-LIRF unmatched rows exactly."

Training: the twelve 2025 months through unm_congestion.load_real (the 2025 witness attached). Serving: the 2026
witness from EVERY matched DEP row of data/raw/ranking.parquet, attached inside build() by
unm_congestion_ship.attach_serve_witness (which refuses pre-attached witness columns and a witness covering no
scored hour). S1's parts come from the shipped fit_unmatched (hybrid, the shipped seeds).

Guards, in order, each refusing the write:
  1. the S1C path (unm_congestion_ship.s1c_from_parts on S1's parts with the serve witness) must rebuild the BASE's
     value on EVERY scored NON-LIRF unmatched row exactly -- those rows are v9's S1C carried through v10 (v10 touched
     LIRF rows only). LIRF rows are R2 / E1's and are not compared;
  2. only scored non-LIRF unmatched rows change, to rint(S1R) in the base dtype; LIRF unmatched rows and every
     matched row are copied byte for byte (asserted after the splice, not assumed);
  3. build_submission.check_submission on the frame and on the file re-read from disk; an existing file is never
     overwritten.

Outputs: submissions/<team>_vM.parquet, submissions/<team>_vM.meta.json and submissions/<team>_vM.unm_rows.parquet
(MVT_ID, ADEP, month, sp, hprox_med, hprox_n, bin, p, nf_cells, nf_fit_c, r_hat, before, after). The before / after
tables by witness bin, by airport and by airport x bin are printed, plus the hot-witness (>= 2,000 s) rows per
airport: the prereg's TRUE shape says the EHAM storm rows should move from ~1,450 toward the witness.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import unm_congestion_resid as ur  # noqa: E402
import unm_congestion_ship as us  # noqa: E402

uc = ur.uc
bs = ur.bs
ROME = ur.ROME
PREREG = ur.PREREG
ARM = f"E3R S1R on non-LIRF unmatched rows ({PREREG}; base rows rebuilt by the S1C path)"
DETAIL_COLUMNS = ["MVT_ID_mvt", "ADEP_mvt", "month", "sp", "hprox_med", "hprox_n", "bin", "p", "nf_cells", "nf_fit_c", "r_hat",
                  "before", "after"]
HOT_S = 2_000.0                 # the prereg's "storm rows": witness >= 2,000 s, reported per airport
WITNESS_COLUMNS = us.WITNESS_COLUMNS
non_lirf_mask = us.non_lirf_mask
log = us.log
_git = us._git


# =============================================================================================
# guard 1: the S1C path rebuilds the base on every non-LIRF unmatched row
# =============================================================================================

def verify_s1c_rebuilds_non_lirf(s1c, scored_unm: pd.DataFrame, base: pd.DataFrame) -> dict:
    """rint(S1C) -- S1's parts with the serve witness through unm_congestion_ship.s1c_from_parts -- must equal the
    base on every scored non-LIRF unmatched row. LIRF rows are R2 / E1's and are not compared. Raises AssertionError
    on any miss; returns the counts for the meta record."""
    s1c = np.asarray(s1c, dtype="float64")
    if len(s1c) != len(scored_unm):
        raise ValueError(f"S1C has {len(s1c)} values for {len(scored_unm)} scored rows")
    m = non_lirf_mask(scored_unm)
    rebuilt = np.rint(s1c).astype("int64")[m]
    ids = scored_unm.MVT_ID_mvt.to_numpy()[m]
    have = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids).to_numpy(dtype="float64")
    if np.isnan(have).any():
        raise AssertionError(f"{int(np.isnan(have).sum())} scored non-LIRF unmatched ids are missing from the base submission")
    bad = int((rebuilt != have.astype("int64")).sum())
    if bad:
        raise AssertionError(f"the S1C path does not rebuild the base on {bad} of {len(have)} scored non-LIRF unmatched rows; "
                             "nothing is written")
    return {"n_checked": int(m.sum()), "n_lirf_not_compared": int((~m).sum()), "path": "S1C (unm_congestion_ship.s1c_from_parts)"}


# =============================================================================================
# the build
# =============================================================================================

def build(train_unm: pd.DataFrame, scored_unm: pd.DataFrame, parts: dict, base: pd.DataFrame,
          serve_frame: pd.DataFrame) -> tuple:
    """(new submission frame, info). `scored_unm` and `parts` are row-aligned (fit_unmatched's order); `scored_unm`
    arrives WITHOUT witness columns; `serve_frame` is the serve file's DEP rows. The 2026 witness is attached first
    (the S1C guard needs it), then guard 1, then S1R from the same parts and the same per-seed S1C body."""
    if not parts.get("hybrid"):
        raise ValueError("the parts are not the hybrid path's (fit_unmatched(..., hybrid=True)); S1C cannot be rebuilt from the cell path")
    scored_w, wsum = us.attach_serve_witness(scored_unm, serve_frame)
    s1c, nf_fit_c, by_seed_c, n_nonfill = us.s1c_from_parts(train_unm, scored_w, parts)
    rebuild = verify_s1c_rebuilds_non_lirf(s1c, scored_unm, base)                      # guard 1
    body = ur.residual_body(train_unm, scored_w, by_seed_c, seeds=parts["seeds"])
    if body["n_train_nonfill"] != n_nonfill:
        raise AssertionError("the S1R body and the S1C body were fitted on different non-fill row counts")
    s1r = bs.mixture(parts["p"], parts["sp"], bs.nf_hybrid(parts["nf_cells"], body["nf_fit_r"]))
    m = non_lirf_mask(scored_unm)
    te = scored_w[m]
    ids = te.MVT_ID_mvt.to_numpy()
    vals = np.rint(s1r[m])
    if not (np.isfinite(vals).all() and (vals >= 1.0).all()):
        raise AssertionError("S1R produced a non-finite or sub-floor value; nothing is written")
    out = bs.splice_unmatched(base, ids, vals)                                          # guard 2, part 1
    pos = pd.Index(base.MVT_ID_mvt).get_indexer(ids)
    if not np.array_equal(out.TAXITIME_SEC_mvt.to_numpy()[pos].astype("int64"), vals.astype("int64")):
        raise AssertionError("rint(S1R) does not fit the submission dtype")
    changed = np.flatnonzero(out.TAXITIME_SEC_mvt.to_numpy() != base.TAXITIME_SEC_mvt.to_numpy())
    if not set(changed.tolist()) <= set(pos.tolist()):                                   # guard 2, part 2
        raise AssertionError("a row outside the scored non-LIRF unmatched set changed; nothing is written")
    before = base.TAXITIME_SEC_mvt.to_numpy()[pos].astype("float64")
    after = vals.astype("float64")
    moved = after != before
    h = te.hprox_med.to_numpy(dtype="float64")
    nan_w = np.isnan(h)
    if moved[nan_w].any():
        raise AssertionError("a scored row without a witness changed; S1R must equal S1C (the base) there")
    detail = pd.DataFrame({"MVT_ID_mvt": ids, "ADEP_mvt": te.ADEP_mvt.astype(str).to_numpy(),
                           "month": te.month.to_numpy().astype("int64"), "sp": te.sp.to_numpy(dtype="float64"),
                           "hprox_med": h, "hprox_n": te.hprox_n.to_numpy(dtype="int64"), "bin": uc.bin_of(h),
                           "p": np.asarray(parts["p"], dtype="float64")[m], "nf_cells": np.asarray(parts["nf_cells"], dtype="float64")[m],
                           "nf_fit_c": np.asarray(nf_fit_c, dtype="float64")[m], "r_hat": np.asarray(body["r_hat"], dtype="float64")[m],
                           "before": before, "after": after})[DETAIL_COLUMNS]
    shift = after - before
    info = {"rebuild": rebuild, "witness_2026": wsum, "seeds": list(parts["seeds"]), "n_train_nonfill": n_nonfill,
            "n_train_witness": body["n_train_witness"], "n_clipped_low": body["n_clipped_low"], "n_clipped_high": body["n_clipped_high"],
            "n_rows_replaced": int(m.sum()), "n_values_differ": int(moved.sum()), "n_nan_witness_rows": int(nan_w.sum()),
            "n_routed_to_body": int((np.asarray(parts["nf_cells"], dtype="float64")[m] < bs.T_TAIL_S).sum()),
            "rms_shift_vs_base_s": float(np.sqrt(np.mean(shift ** 2))) if len(shift) else 0.0,
            "mean_shift_s": float(shift.mean()) if len(shift) else 0.0,
            "per_airport_mean_shift_s": {str(k): float(v) for k, v in pd.Series(shift).groupby(detail.ADEP_mvt.to_numpy()).mean().items()},
            "bins_2026_changed_rows": uc.bin_counts(detail.bin.to_numpy()[moved]),
            "bins_2026_replaced_rows": uc.bin_counts(detail.bin.to_numpy()),
            "sse_shift_vs_base_board_mse": float((shift ** 2).sum() / len(base)),
            "hot_rows": hot_table(detail),
            "nf_fit_r": body["nf_fit_r"][m], "nf_fit_r_by_seed": {s: v[m] for s, v in body["nf_fit_r_by_seed"].items()},
            "before": before, "after": after, "detail": detail}
    return out, info


# =============================================================================================
# the owner's tables
# =============================================================================================

def hot_table(detail: pd.DataFrame, hot_s: float = HOT_S) -> dict:
    """Per airport, the rows with witness >= hot_s: n, witness mean, before mean, after mean, after max."""
    d = detail[detail.hprox_med.to_numpy(dtype="float64") >= hot_s]
    out = {}
    for ap, part in d.groupby("ADEP_mvt", observed=True):
        out[str(ap)] = {"n": int(len(part)), "witness_mean": float(part.hprox_med.mean()), "before_mean": float(part.before.mean()),
                        "after_mean": float(part.after.mean()), "after_max": float(part.after.max())}
    return out


def render_hot(hot: dict, out=log, hot_s: float = HOT_S) -> None:
    out(f"--- rows with witness >= {hot_s:.0f} s by airport (non-LIRF unmatched scored rows) ---")
    out(f"  {'airport':10s} {'n':>6s} {'witness':>9s} {'before':>9s} {'after':>9s} {'after max':>10s}")
    for ap, r in hot.items():
        out(f"  {ap:10s} {r['n']:6,} {r['witness_mean']:9.0f} {r['before_mean']:9.0f} {r['after_mean']:9.0f} {r['after_max']:10.0f}")


# =============================================================================================
# writing: refuse to overwrite, check the frame and the re-read file, the detail parquet and the meta json
# =============================================================================================

def write_outputs(out: pd.DataFrame, info: dict, template: pd.DataFrame, version: int, base_name: str,
                  sub_dir: pathlib.Path, t0: float) -> tuple:
    """(dest, meta). check_submission on the frame, refuse an existing file, write, check the re-read file, then the
    detail parquet and the meta json beside it (unm_congestion_ship.write_outputs' pattern with E3R's record)."""
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
    meta = {"version": int(version), "file": dest.name, "base": base_name, "arm": ARM, "prereg": PREREG, "parent_prereg": ur.PARENT_PREREG,
            **_git(), "seeds": info["seeds"], "n_train_nonfill": info["n_train_nonfill"], "n_train_witness": info["n_train_witness"],
            "n_clipped_low": info["n_clipped_low"], "n_clipped_high": info["n_clipped_high"], "clip_s": ur.CLIP_S,
            "rebuild_guard": info["rebuild"],
            "witness_2026": {**info["witness_2026"], "source": "data/raw/ranking.parquet, every matched DEP row (Amendment E3C.1 item 4)"},
            "n_rows_replaced": info["n_rows_replaced"], "n_values_differ": info["n_values_differ"],
            "n_nan_witness_rows": info["n_nan_witness_rows"], "n_routed_to_body": info["n_routed_to_body"],
            "mean_shift_s": info["mean_shift_s"], "rms_shift_vs_base_s": info["rms_shift_vs_base_s"],
            "per_airport_mean_shift_s": info["per_airport_mean_shift_s"], "bins_2026_changed_rows": info["bins_2026_changed_rows"],
            "bins_2026_replaced_rows": info["bins_2026_replaced_rows"], "sse_shift_vs_base_board_mse": info["sse_shift_vs_base_board_mse"],
            "hot_rows": info["hot_rows"], "detail_file": f"{dest.stem}.unm_rows.parquet", "wall_s": round(time.time() - t0, 1)}
    (sub_dir / f"{dest.stem}.meta.json").write_text(json.dumps(meta, indent=1))
    return dest, meta


# =============================================================================================
# main
# =============================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the board version to build on, e.g. submissions/merry-quicksand_v10.parquet")
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
    log(f"guard 1: the S1C path rebuilds {base_path.name} on all {info['rebuild']['n_checked']:,} scored non-LIRF unmatched rows "
        f"({info['rebuild']['n_lirf_not_compared']:,} LIRF rows not compared)")
    log(f"2026 witness: {info['witness_2026']}")
    log(f"S1R: body fitted on {info['n_train_witness']:,} of {info['n_train_nonfill']:,} non-fill rows (with a witness); offset clipped low "
        f"{info['n_clipped_low']:,} / high {info['n_clipped_high']:,}; replaced {info['n_rows_replaced']:,} rows, {info['n_values_differ']:,} "
        f"values differ, {info['n_nan_witness_rows']:,} without a witness (unchanged), routed to the body {info['n_routed_to_body']:,}; "
        f"mean shift {info['mean_shift_s']:+.1f} s, RMS shift {info['rms_shift_vs_base_s']:.1f} s, sse shift "
        f"{info['sse_shift_vs_base_board_mse']:.1f} board MSE; changed-row bins {info['bins_2026_changed_rows']}")
    us.render_tables(us.shift_tables(info["detail"]))
    render_hot(info["hot_rows"])
    dest, meta = write_outputs(out, info, template, a.version, base_path.name, ROOT / "submissions", t0)
    print(json.dumps(meta, indent=1))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
