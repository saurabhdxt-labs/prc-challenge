"""`scripts/rome_bandfloor.py` -- arm E1 of plans/PREREG_rome_bandfloor_2026_09_10.md (owner decision 2026-09-10).

    python scripts/rome_bandfloor.py --base submissions/merry-quicksand_v7.parquet --version 8

For every scored LIRF unmatched row with LO <= sp < HI: pred = max(pred_base, rint(sp)). Every other
row is copied. No fitted parameter: in 2025 every LIRF unmatched row with sp >= 24,000 had
y >= sp - 60 (fills and 24 h date-slips only), and below 86,400 both classes lie at or above sp.

Guards, in order, each refusing the write:
  1. the base must BE the version whose Rome detail file is given (default v7): its value on every
     LIRF unmatched scored row equals that file's `after`, and the detail file's sp equals derive()'s;
  2. only rows inside the rule's mask change, each to exactly rint(sp) and upward; every other row is
     copied byte-for-byte;
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

LO, HI = 24_000.0, 86_400.0
AIRPORT = "LIRF"


def rule_mask(sp, ap, unmatched) -> np.ndarray:
    """Rows the registered rule may touch: LIRF, unmatched, LO <= sp < HI."""
    sp = np.asarray(sp, dtype="float64")
    return (np.asarray(ap) == AIRPORT) & np.asarray(unmatched, dtype=bool) & (sp >= LO) & (sp < HI)


def verify_base(base: pd.DataFrame, detail: pd.DataFrame, lirf_unm: pd.DataFrame) -> None:
    """Guard 1: `base` is the submission `detail` describes, and `detail`'s sp is derive()'s sp."""
    d = detail.set_index("MVT_ID_mvt")
    ids = lirf_unm.MVT_ID_mvt.to_numpy()
    if set(ids) != set(d.index):
        raise AssertionError(f"the Rome detail file covers {len(d)} rows, the scored LIRF unmatched set {len(ids)}: "
                             "different row sets")
    if not np.array_equal(d.sp.reindex(ids).to_numpy(dtype="float64"), lirf_unm.sp.to_numpy(dtype="float64")):
        raise AssertionError("the Rome detail file's sp differs from derive()'s sp: wrong file or changed derivation")
    have = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids).to_numpy(dtype="float64")
    want = d.after.reindex(ids).to_numpy(dtype="float64")
    bad = int((np.isnan(have) | (have != want)).sum())
    if bad:
        raise AssertionError(f"the base differs from the Rome detail file's `after` on {bad} of {len(ids)} LIRF "
                             "unmatched rows: it is not the version it claims to be; nothing is written")


def build(base: pd.DataFrame, scored: pd.DataFrame) -> tuple:
    """(new frame, info). `scored` carries MVT_ID_mvt, ADEP_mvt, unmatched, sp for every scored row."""
    m = rule_mask(scored.sp, scored.ADEP_mvt, scored.unmatched)
    rows = scored[m]
    col = base.TAXITIME_SEC_mvt
    pos = pd.Index(base.MVT_ID_mvt).get_indexer(rows.MVT_ID_mvt.to_numpy())
    if (pos < 0).any():
        raise AssertionError("rule rows missing from the base submission")
    before = col.to_numpy()[pos].astype("int64")
    floor = np.rint(rows.sp.to_numpy(dtype="float64")).astype("int64")
    after = np.maximum(before, floor)
    vals = col.to_numpy().copy()
    vals[pos] = after.astype(vals.dtype)
    if not np.array_equal(vals[pos].astype("int64"), after):
        raise AssertionError("the floor does not fit the submission dtype")
    out = base.copy()
    out["TAXITIME_SEC_mvt"] = vals
    changed = np.flatnonzero(out.TAXITIME_SEC_mvt.to_numpy() != col.to_numpy())
    if not set(changed) <= set(pos) or (vals[changed] < col.to_numpy()[changed]).any():
        raise AssertionError("a row outside the rule changed, or a row moved down")
    moved = after != before
    info = {"n_rule_rows": int(m.sum()), "n_values_differ": int(moved.sum()),
            "ids_changed": rows.MVT_ID_mvt.to_numpy()[moved], "sp": rows.sp.to_numpy()[moved],
            "before": before[moved], "after": after[moved],
            "fill_gain_board_mse": float(((before - floor)[moved] ** 2).sum() / len(base))}
    return out, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--detail", default="submissions/merry-quicksand_v7.rome_rows.parquet",
                    help="the Rome detail file written with the base (guard 1)")
    a = ap.parse_args(argv)
    import stratum_fold as sf
    bs = sf.bs
    t0 = time.time()
    raw = ROOT / "data" / "raw"
    base = pq.read_table(ROOT / a.base).to_pandas()
    template = pq.read_table(raw / "submitting.parquet").to_pandas()
    bs.check_submission(base, template)
    ranking = bs.derive(bs.load_movements([raw / "ranking.parquet"]))
    scored = ranking[ranking.MVT_ID_mvt.isin(set(template.MVT_ID_mvt))].reset_index(drop=True)
    if len(scored) != len(template):
        raise ValueError(f"joined {len(scored)} scored rows, template has {len(template)}")
    lirf_unm = scored[(scored.ADEP_mvt == AIRPORT).to_numpy() & scored.unmatched.to_numpy()]
    verify_base(base, pd.read_parquet(ROOT / a.detail), lirf_unm)
    out, info = build(base, scored[["MVT_ID_mvt", "ADEP_mvt", "unmatched", "sp"]])
    bs.check_submission(out, template)
    dest = ROOT / "submissions" / bs.submission_name(a.version)
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    out.to_parquet(dest, index=False)
    bs.check_submission(pq.read_table(dest).to_pandas(), template)
    meta = {"version": a.version, "file": dest.name, "base": pathlib.Path(a.base).name,
            "arm": "E1 LIRF unmatched schedule floor (plans/PREREG_rome_bandfloor_2026_09_10.md)",
            "rule": f"max(base, rint(sp)) on LIRF unmatched rows with {LO:.0f} <= sp < {HI:.0f}",
            "n_rule_rows": info["n_rule_rows"], "n_values_differ": info["n_values_differ"],
            "changed": [{"MVT_ID_mvt": int(i), "sp": float(s), "before": int(b), "after": int(x)}
                        for i, s, b, x in zip(info["ids_changed"], info["sp"], info["before"], info["after"])],
            "fill_gain_board_mse_if_all_fills": info["fill_gain_board_mse"], "wall_s": round(time.time() - t0, 1)}
    (ROOT / "submissions" / f"{dest.stem}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
