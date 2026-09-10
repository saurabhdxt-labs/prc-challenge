"""`scripts/rome_ship.py` -- ship Rome arm R2 into a base submission (owner decision 2026-09-10).

    python scripts/rome_ship.py --base submissions/merry-quicksand_v6.parquet --version 7

R2 = arm R's stake-weighted LightGBM fill classifier (scripts/rome_fill.py, mean of seeds 0,1,2,
trained on ALL LIRF departures of the twelve 2025 months) inside S1's mixture, plus the date-slip
two-class expectation on segment G (scripts/rome_dateslip.py, shares from all twelve months).
Registered and measured in plans/PREREG_rome_fill_2026_09_10.md (RESULT R) and
plans/PREREG_rome_dateslip_2026_09_10.md (RESULT DS).

Guards, in order, each refusing the write:
  1. S1's refit parts (build_submission.fit_unmatched, hybrid, the shipped seeds) must rebuild the
     BASE submission's value on EVERY scored unmatched row exactly -- the base's unmatched rows are
     v4's S1, so nf and the other airports' predictions are the shipped ones;
  2. only LIRF unmatched rows change; every other row is copied;
  3. build_submission.check_submission on the frame and on the file re-read from disk.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rome_dateslip as ds  # noqa: E402
import rome_fill as rf  # noqa: E402

bs = rf.sf.bs
SEEDS = rf.SEEDS


def verify_parts_rebuild(parts: dict, scored_unm: pd.DataFrame, base: pd.DataFrame) -> None:
    """rint(mixture(p, sp, nf)) from S1's parts must equal the base on every scored unmatched row."""
    rebuilt = np.rint(bs.mixture(parts["p"], scored_unm.sp.to_numpy(dtype="float64"), parts["nf"])).astype("int64")
    have = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(scored_unm.MVT_ID_mvt.to_numpy()).to_numpy()
    if np.isnan(have.astype("float64")).any():
        raise AssertionError("scored unmatched ids missing from the base submission")
    bad = int((rebuilt != have.astype("int64")).sum())
    if bad:
        raise AssertionError(f"S1's refit parts do not rebuild the base on {bad} of {len(have)} scored unmatched rows; "
                             "nothing is written")


def build(train_unm, train_lirf, scored_unm, parts, base, seeds=SEEDS) -> tuple:
    """(new submission frame, info). `scored_unm` and `parts` are row-aligned (fit_unmatched's order)."""
    verify_parts_rebuild(parts, scored_unm, base)
    m = (scored_unm.ADEP_mvt == "LIRF").to_numpy()
    te = scored_unm[m]
    tr = rf.rome_rows(train_unm, train_lirf, sorted(set(train_unm.month) | set(train_lirf.month)))
    p_r = np.mean([rf.fit_predict_fill(tr, te, seed=s) for s in seeds], axis=0)
    q, mt = ds.band_shares(train_unm[train_unm.ADEP_mvt == "LIRF"])
    mix = rf.mixture(p_r, te.sp.to_numpy(dtype="float64"), np.asarray(parts["nf"])[m])
    vals = np.rint(ds.apply(te, mix, q, mt))
    out = bs.splice_unmatched(base, te.MVT_ID_mvt.to_numpy(), vals)
    before = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.loc[te.MVT_ID_mvt.to_numpy()].to_numpy()
    info = {"n_rows_replaced": int(m.sum()), "n_values_differ": int((before != vals).sum()),
            "n_in_segment_g": int(ds.segment_mask(te).sum()), "q": q, "m": mt, "p_r": p_r,
            "p_s1_lirf": np.asarray(parts["p"])[m], "before": before, "after": vals,
            "sse_shift_vs_base_board_mse": float(((vals - before) ** 2).sum() / len(base))}
    return out, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--version", type=int, required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    raw = ROOT / "data" / "raw"
    base = pq.read_table(ROOT / a.base).to_pandas()
    template = pq.read_table(raw / "submitting.parquet").to_pandas()
    bs.check_submission(base, template)
    unm, lirf = rf.load_frames(print)
    ranking = bs.derive(bs.load_movements([raw / "ranking.parquet"]))
    scored = ranking[ranking.MVT_ID_mvt.isin(set(template.MVT_ID_mvt))].reset_index(drop=True)
    if len(scored) != len(template):
        raise ValueError(f"joined {len(scored)} scored rows, template has {len(template)}")
    scored_unm = scored[scored.unmatched.to_numpy()]
    parts: dict = {}
    bs.fit_unmatched(unm, scored_unm, train_matched=lirf, hybrid=True, seeds=bs.SEEDS, parts=parts)
    out, info = build(unm, lirf, scored_unm, parts, base)
    bs.check_submission(out, template)
    dest = ROOT / "submissions" / bs.submission_name(a.version)
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    out.to_parquet(dest, index=False)
    bs.check_submission(pq.read_table(dest).to_pandas(), template)
    lir = scored_unm[(scored_unm.ADEP_mvt == "LIRF").to_numpy()]
    detail = pd.DataFrame({"MVT_ID_mvt": lir.MVT_ID_mvt.to_numpy(), "sp": lir.sp.to_numpy(), "dayoff": lir.dayoff.to_numpy(),
                           "p_s1": info["p_s1_lirf"], "p_r": info["p_r"], "before": info["before"], "after": info["after"]})
    detail.to_parquet(ROOT / "submissions" / f"{dest.stem}.rome_rows.parquet", index=False)
    meta = {"version": a.version, "file": dest.name, "base": pathlib.Path(a.base).name,
            "arm": "R2 (plans/PREREG_rome_fill_2026_09_10.md RESULT R + plans/PREREG_rome_dateslip_2026_09_10.md RESULT DS)",
            "seeds": list(SEEDS), "n_rows_replaced": info["n_rows_replaced"], "n_values_differ": info["n_values_differ"],
            "n_in_segment_g": info["n_in_segment_g"], "q": info["q"], "m": info["m"],
            "rms_shift_vs_base_s": float(np.sqrt(np.mean((info["after"] - info["before"]) ** 2))),
            "wall_s": round(time.time() - t0, 1)}
    (ROOT / "submissions" / f"{dest.stem}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
