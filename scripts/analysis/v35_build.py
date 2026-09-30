"""v35 = v34 with the R12 matched values (plans/PREREG_v35_r12_2026_09_26.md).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/v35_build.py --vals data/r12/<serve dir>/matched_2026_r12.parquet

Takes the shipped v34 upload file, replaces the TAXITIME of exactly the ids `r12_stack serve` produced (v34's matched
lanes recomposed with the R12 base F and re-trained stack), rint + 1 s floor, int32 as v34. Every other row stays
byte-identical to v34. Writes submissions/merry-quicksand_v35.parquet + .meta.json; uploads NOTHING.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOARD_ROWS = 344_841
V34 = ROOT / "submissions" / "merry-quicksand_v34.parquet"
PREREG_DEFAULT = "plans/PREREG_v35_r12_2026_09_26.md"


def check_prereg(rel: str, root=ROOT) -> str:
    """The registration the meta cites: an existing file under plans/ (relative to the repo root)."""
    if not str(rel).startswith("plans/"):
        raise ValueError(f"prereg must be a path under plans/, got {rel!r}")
    if not (pathlib.Path(root) / rel).is_file():
        raise FileNotFoundError(f"prereg {rel} does not exist")
    return str(rel)


def compose(base: pd.DataFrame, vals: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(new frame, report). `vals` = MVT_ID_mvt, taxi (seconds, float). Refuses ids absent from `base`, duplicate ids
    and non-finite values; never mutates `base`."""
    ids = vals.MVT_ID_mvt.to_numpy(float)
    taxi = vals.taxi.to_numpy(float)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("duplicate MVT_ID_mvt in vals")
    if not np.isfinite(taxi).all():
        raise ValueError(f"{int((~np.isfinite(taxi)).sum())} vals are not finite")
    pos = pd.Index(base.MVT_ID_mvt.to_numpy(float)).get_indexer(ids)
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} vals ids are not in the base file")
    out = base.copy()
    old = out.TAXITIME_SEC_mvt.to_numpy().astype(np.int64)
    new = old.copy()
    new[pos] = np.maximum(np.rint(taxi), 1).astype(np.int64)
    out["TAXITIME_SEC_mvt"] = new.astype(np.int32)
    d = (new - old).astype(float)
    rep = {"n_given": int(len(ids)), "n_changed": int((d != 0).sum()), "S": float((d ** 2).sum() / BOARD_ROWS),
           "mean_shift_s": float(d[pos].mean()) if len(pos) else 0.0}
    return out, rep


AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")


def filter_airports(vals: pd.DataFrame, ap: pd.Series, keep) -> pd.DataFrame:
    """The rows of `vals` whose departure airport (`ap`, indexed by MVT_ID_mvt) is in `keep`, order preserved.
    Refuses an id `ap` cannot place and an airport name outside the ten scored ones."""
    bad = set(keep) - set(AIRPORTS)
    if bad:
        raise ValueError(f"unknown airport(s) {sorted(bad)}")
    a = ap.reindex(vals.MVT_ID_mvt.to_numpy(float))
    if a.isna().any():
        raise ValueError(f"{int(a.isna().sum())} vals ids have no departure airport in the ranking file")
    return vals[np.isin(a.to_numpy(), list(keep))].reset_index(drop=True)


def exclude_airport_months(vals: pd.DataFrame, ap: pd.Series, month: pd.Series, excl) -> pd.DataFrame:
    """`vals` without the rows whose (departure airport, MVT month) is in `excl` (those rows keep the base file's
    value). `ap` / `month` are indexed by MVT_ID_mvt."""
    for a, m in excl:
        if a not in AIRPORTS:
            raise ValueError(f"unknown airport {a!r}")
        if not 1 <= int(m) <= 12:
            raise ValueError(f"month must be 1..12, got {m!r}")
    ids = vals.MVT_ID_mvt.to_numpy(float)
    key = list(zip(ap.reindex(ids).to_numpy(), month.reindex(ids).to_numpy()))
    drop = np.array([(a, int(m)) in {(x, int(y)) for x, y in excl} if a == a else False for a, m in key], dtype=bool)
    return vals[~drop].reset_index(drop=True)


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:  # pragma: no cover - thin I/O shell around compose(), exercised by the v35 run
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--airports", default=",".join(AIRPORTS), help="comma list; default all ten")
    ap.add_argument("--base", default=str(V34), help="the file whose values are kept outside vals (default v34)")
    ap.add_argument("--exclude", default="", help="airport:month pairs kept at the base, e.g. LSZH:7")
    ap.add_argument("--prereg", default=PREREG_DEFAULT, help="the registration the meta cites (under plans/)")
    a = ap.parse_args(argv)
    prereg = check_prereg(a.prereg)
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_submission as bs
    name = bs.submission_name(a.version)
    out_p = ROOT / "submissions" / name
    if out_p.exists():
        raise SystemExit(f"refusing to overwrite {out_p}")
    base_p = pathlib.Path(a.base) if pathlib.Path(a.base).is_absolute() else ROOT / a.base
    base = pd.read_parquet(base_p)
    keep = tuple(a.airports.split(","))
    rk = pd.read_parquet(ROOT / "data" / "raw" / "ranking.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    rk = rk.set_index("MVT_ID_mvt")
    vals = filter_airports(pd.read_parquet(a.vals), rk.ADEP_mvt, keep)
    excl = [(x.split(":")[0], int(x.split(":")[1])) for x in a.exclude.split(",") if x]
    if excl:
        vals = exclude_airport_months(vals, rk.ADEP_mvt, pd.to_datetime(rk.MVT_TIME_UTC_mvt).dt.month, excl)
    new, rep = compose(base, vals)
    rep["airports"] = list(keep)
    rep["exclude"] = excl
    bs.check_submission(new, base)
    new.to_parquet(out_p, index=False)
    meta = dict(rep, base=str(base_p.relative_to(ROOT)), base_sha=_sha(base_p), vals=a.vals,
                vals_sha=_sha(pathlib.Path(a.vals)), out=str(out_p.relative_to(ROOT)), out_sha=_sha(out_p),
                prereg=prereg)
    (out_p.parent / (name + ".meta.json")).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
