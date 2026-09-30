"""SHIP-ARC — v17 = v16 with the ARC LIRF rules on 42 rows (plans/PREREG_arr_row_clocks_2026_09_16.md, RESULT + corrections).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/arc_ship_build.py

R_ord rows (30) → max(rint(MVT − AOBT_3 from the same flight's ARR row), 1); R_fill rows (12) → rint(MVT − SCHED). Every other row
is byte-identical to v16. The values are recomputed here from the raw ranking file with arr_row_clocks' own functions and must
equal the transfer diagnostic's recorded rows. Owner said "build it" on 2026-09-16. The UPLOAD is a separate owner decision.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import arr_row_clocks as A  # noqa: E402

V16 = ROOT / "submissions" / "merry-quicksand_v16.parquet"
V16_SHA = "8701d117686af9513263fed148b4a806d114c0fe8de37567e1ca0a9f07997a3e"
TRANSFER = ROOT / "reports" / "arr_row_clocks_transfer_2026_09_16.json"
PREREG = ROOT / "plans" / "PREREG_arr_row_clocks_2026_09_16.md"
RUNS = ROOT / "data" / "pipeline_runs"
V17 = ROOT / "submissions" / "merry-quicksand_v17.parquet"
EXPECT = {"ord": 30, "fill": 12}
BOARD_N = 344_841.0


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arc_values(rule, mvt, aobt3, sp) -> np.ndarray:
    """The shipped value per ruled row: ord → max(rint(mvt − aobt3), 1); fill → max(rint(sp), 1). Refuses any other rule."""
    rule = np.asarray(rule)
    mvt, aobt3, sp = (np.asarray(x, dtype="float64") for x in (mvt, aobt3, sp))
    if not np.isin(rule, ["ord", "fill"]).all():
        raise ValueError("arc_values: rows without an ord/fill rule")
    v = np.where(rule == "ord", np.rint(mvt - aobt3), np.rint(sp))
    if not np.isfinite(v).all():
        raise ValueError("arc_values: non-finite value")
    return np.maximum(v, 1.0)


def apply_rows(base: pd.DataFrame, ids, values) -> tuple[pd.DataFrame, np.ndarray]:
    """(candidate, touched mask). base: MVT_ID_mvt, TAXITIME_SEC_mvt. Every id must exist exactly once; dtype and order kept."""
    ids = np.asarray(ids, dtype="float64")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("apply_rows: duplicate ids")
    pos = pd.Index(base.MVT_ID_mvt.to_numpy("float64")).get_indexer(ids)
    if (pos < 0).any():
        raise ValueError(f"apply_rows: {int((pos < 0).sum())} ids not in the base submission")
    new = base.TAXITIME_SEC_mvt.to_numpy().copy()
    new[pos] = np.asarray(values, dtype="float64").astype(new.dtype)
    cand = base.copy()
    cand["TAXITIME_SEC_mvt"] = new
    touched = new != base.TAXITIME_SEC_mvt.to_numpy()
    return cand, touched


def main() -> int:
    if sha256(V16) != V16_SHA:
        raise AssertionError("G1: v16 sha differs")
    if V17.exists():
        raise FileExistsError(V17)
    v16 = pd.read_parquet(V16)
    r = pd.read_parquet(A.RANK, columns=sorted(set(A.DEP_COLS + A.ARR_COLS)))
    d26 = r[(r.PHASE_mvt == "DEP") & r.AOBT_3_flt.isna() & (r.ADEP_mvt == "LIRF")]
    dep = A.dep_frame(d26, d26.MVT_ID_mvt.to_numpy())
    j = A.join_arr(dep, A.arr_frame(r))
    sp = (dep.mvt - dep.sched).to_numpy()
    rule = A.sub_rule(sp, dep.sched, j.arr_lobt, j.arr_aobt3)
    m = np.isin(rule, ["ord", "fill"])
    counts = {k: int((rule == k).sum()) for k in EXPECT}
    if counts != EXPECT:
        raise AssertionError(f"G2: rule counts {counts}, registered {EXPECT}")
    vals = arc_values(rule[m], dep.mvt.to_numpy()[m], j.arr_aobt3.to_numpy()[m], sp[m])
    ids = dep.MVT_ID_mvt.to_numpy("float64")[m]
    rec = pd.DataFrame(json.loads(TRANSFER.read_text())["rows_2026"]).set_index("MVT_ID_mvt")
    want = np.where(rec.rule.reindex(ids) == "ord", rec.new.reindex(ids), np.maximum(np.rint(rec.sp.reindex(ids)), 1.0))
    if not np.array_equal(vals, want):
        raise AssertionError("G2: recomputed values differ from the transfer record")
    cand, touched = apply_rows(v16, ids, vals)
    base = v16.TAXITIME_SEC_mvt.to_numpy("float64")
    tpos = pd.Index(v16.MVT_ID_mvt.to_numpy("float64")).get_indexer(ids)
    outside = touched.copy()
    outside[tpos] = False
    if outside.any():
        raise AssertionError("G3: a row outside the 42 changed")
    if not (np.array_equal(cand.MVT_ID_mvt.to_numpy(), v16.MVT_ID_mvt.to_numpy()) and cand.dtypes.equals(v16.dtypes)):
        raise AssertionError("G3: ids / order / dtype")
    newv = cand.TAXITIME_SEC_mvt.to_numpy("float64")
    if not (np.isfinite(newv).all() and (newv >= 1).all()):
        raise AssertionError("G3: non-finite or sub-1 value")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RUNS / f"arc_{stamp}"
    out.mkdir(parents=False, exist_ok=False)
    sub, rows_p = out / "merry-quicksand_arc.parquet", out / "arc_touched_rows.parquet"
    cand.to_parquet(sub, index=False)
    tr = pd.DataFrame({"MVT_ID_mvt": ids, "rule": rule[m], "sp": sp[m], "v16": base[tpos], "v17": vals})
    tr.to_parquet(rows_p, index=False)
    back = pd.read_parquet(sub)
    if not (back.equals(cand)):
        raise AssertionError("G4: re-read differs")
    shutil.copyfile(sub, V17)
    if sha256(V17) != sha256(sub):
        raise AssertionError("G4: copy differs")
    mv = vals - base[tpos]
    manifest = dict(tag="SHIP-ARC", version="v17", prereg=str(PREREG.relative_to(ROOT)), prereg_sha256=sha256(PREREG),
                    created_utc=stamp, guards=dict(G1=True, G2=True, G3=True, G4=True),
                    touched=int(touched.sum()), rules=counts,
                    moves=dict(mean_abs_s=float(np.abs(mv).mean()), max_abs_s=float(np.abs(mv).max()),
                               n_gt300=int((np.abs(mv) > 300).sum()), n_gt1800=int((np.abs(mv) > 1800).sum())),
                    stake_board_mse=dict(if_every_move_were_pure_noise=-float((mv ** 2).sum() / BOARD_N)),
                    expected_board_mse="+300…+460 (transfer diagnostic, not a clause; C +174, D +125…+286)",
                    evidence_statement="LIRF C/D WORKING by locked clauses but confirmatory (thresholds set knowing research numbers); "
                                       "OTHER NOT WORKING (not applied); >80 % of gain in >1,800 s moves: a bet on 42 rows",
                    outputs={str(p.relative_to(ROOT)): sha256(p) for p in (sub, rows_p, V17)},
                    inputs={str(p.relative_to(ROOT)): sha256(p) for p in (V16, A.RANK, TRANSFER)},
                    upload="NOT AUTHORISED - separate owner decision")
    (out / "arc_manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    shutil.copyfile(out / "arc_manifest.json", V17.with_suffix(".manifest.json"))
    print(json.dumps({k: manifest[k] for k in ("touched", "rules", "moves", "stake_board_mse", "outputs")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
