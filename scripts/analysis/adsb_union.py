"""The adsb.lol + adsb.fi point union of one day (rule AUN.1, plans/PREREG_adsb_union_upgrade_2026_09_16.md; used by FM1
through plans/PREREG_afi_adsbfi_into_fm1_2026_10_03.md).

Every adsb.lol sample is kept unchanged; an adsb.fi sample is added iff it is at least NEAR_S from every adsb.lol sample of
the same aircraft. Day files share the adsb.lol day-file schema (COLS), so scripts/adsb_features reads a union day exactly
as an adsb.lol day. `write_manifest` gives a union directory the MANIFEST.json scripts/analysis/r12_build.py reads, with
the sources it contains declared (`contains`), which the feature store and the tables carry forward.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pandas as pd

COLS = ["icao", "reg", "type", "callsign", "airport", "t", "lat", "lon", "on_ground", "alt_missing", "alt", "gs", "track", "src"]
NEAR_S = 1.0
RULE = "AUN.1: adsb.lol unchanged + adsb.fi samples >= 1 s from any adsb.lol sample of the same aircraft"
CONTAINS = ["adsb.lol", "adsb.fi"]


def check_cols(frame, name):
    if frame is not None and list(frame.columns) != COLS:
        raise ValueError(f"{name} columns {list(frame.columns)} differ from the day-file schema {COLS}")


def union_day(lol: pd.DataFrame | None, fi: pd.DataFrame | None) -> tuple:
    """(union frame sorted by (icao, t, adsb.lol first), counts). adsb.lol rows are kept unchanged; an adsb.fi row is kept
    iff |t_fi - t_lol| >= NEAR_S for every adsb.lol row of the same aircraft (icao compared case-insensitively; kept fi
    icao written lower-case)."""
    check_cols(lol, "adsb.lol")
    check_cols(fi, "adsb.fi")
    lol = lol if lol is not None else pd.DataFrame({c: pd.Series(dtype=(fi[c].dtype if fi is not None else "float64")) for c in COLS})
    fi = fi if fi is not None else lol.iloc[0:0]
    n_lol, n_fi = len(lol), len(fi)
    keep = np.ones(n_fi, dtype=bool)
    if n_lol and n_fi:
        lk = lol.icao.astype("string").str.lower().to_numpy(dtype=object)
        fk = fi.icao.astype("string").str.lower().to_numpy(dtype=object)
        codes, uniq = pd.factorize(np.concatenate([lk, fk]))
        lc, fc = codes[:n_lol].astype("float64"), codes[n_lol:].astype("float64")
        lt, ft = lol.t.to_numpy("float64"), fi.t.to_numpy("float64")
        t0 = float(min(lt.min(), ft.min()))
        span = float(max(lt.max(), ft.max())) - t0
        if span >= 1e6:
            raise ValueError(f"day spans {span:.0f} s; the composite key needs < 1e6 s")
        lkey = lc * 1e6 + (lt - t0)
        order = np.argsort(lkey, kind="mergesort")
        skey, sc, st = lkey[order], lc[order], lt[order]
        pos = np.searchsorted(skey, fc * 1e6 + (ft - t0))
        near = np.zeros(n_fi, dtype=bool)
        for off in (-1, 0):                      # the nearest same-aircraft lol sample is one of the two bracketing ones
            j = pos + off
            ok = (j >= 0) & (j < n_lol)
            jj = np.clip(j, 0, n_lol - 1)
            near |= ok & (sc[jj] == fc) & (np.abs(st[jj] - ft) < NEAR_S)
        keep = ~near
    kept = fi[keep].copy()
    kept["icao"] = kept.icao.astype("string").str.lower().astype(fi.icao.dtype if n_fi else kept.icao.dtype)
    parts = [x.assign(_o=o) for x, o in ((lol, 0), (kept, 1)) if len(x)]
    if parts:
        out = pd.concat(parts, ignore_index=True).sort_values(["icao", "t", "_o"], kind="mergesort")
        out = out.drop(columns="_o").reset_index(drop=True)[COLS]
    else:
        out = lol.iloc[0:0].reset_index(drop=True)[COLS]
    info = {"n_lol": int(n_lol), "n_fi": int(n_fi), "n_fi_kept": int(keep.sum()), "n_fi_dropped_near_lol": int((~keep).sum())}
    return out, info


def fi_rows(u: pd.DataFrame, lol: pd.DataFrame) -> np.ndarray:
    """Mask of the union rows that are NOT adsb.lol rows: the union is lol + kept fi, sorted with lol first on ties, so the lol
    rows are recovered by matching (icao, t) multiplicities in order."""
    key_u = pd.MultiIndex.from_arrays([u.icao.to_numpy(dtype=object), u.t.to_numpy()])
    key_l = pd.MultiIndex.from_arrays([lol.icao.to_numpy(dtype=object), lol.t.to_numpy()])
    cnt = pd.Series(1, index=key_l).groupby(level=[0, 1]).size()
    rank = pd.Series(1, index=key_u).groupby(level=[0, 1]).cumcount().to_numpy()
    have = cnt.reindex(key_u).fillna(0).to_numpy()
    return rank >= have


def atomic_parquet(frame: pd.DataFrame, path: pathlib.Path, meta: dict | None = None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    t = pa.Table.from_pandas(frame, preserve_index=False)
    if meta:
        m = dict(t.schema.metadata or {})
        m.update({k.encode(): (v if isinstance(v, str) else json.dumps(v)).encode() for k, v in meta.items()})
        t = t.replace_schema_metadata(m)
    pq.write_table(t, tmp)
    os.replace(tmp, path)


def _day_file(d: pathlib.Path, day: str):
    p = pathlib.Path(d) / f"adsb_{day}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def build_days(days, out_dir: pathlib.Path, fi_dir, lol_dir, meta: dict | None = None, fi_exclude_airports=()) -> dict:
    """Write the union file of each day (and nothing else); a day with neither source is skipped; `meta` is stored in each
    file's parquet metadata. G6: the adsb.lol subset of each union day equals the adsb.lol file. Returns counts per day."""
    out = {}
    for day in days:
        target = out_dir / f"adsb_{day}.parquet"
        lol, fi = _day_file(lol_dir, day), _day_file(fi_dir, day)
        if lol is None and fi is None:
            continue
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        n_excl = 0
        if fi is not None and fi_exclude_airports:                       # their adsb.lol samples stay (AFI.1)
            drop = fi.airport.astype("string").isin(list(fi_exclude_airports)).to_numpy(dtype=bool)
            n_excl, fi = int(drop.sum()), fi[~drop].reset_index(drop=True)
        u, info = union_day(lol, fi)
        info["n_fi_excluded_airport"] = n_excl
        if lol is not None:                                            # G6
            back = u.iloc[np.flatnonzero(~fi_rows(u, lol))].reset_index(drop=True)
            if not back.equals(lol.sort_values(["icao", "t"], kind="mergesort").reset_index(drop=True)):
                raise AssertionError(f"G6 {day}: the adsb.lol subset of the union differs from the adsb.lol file")
        if len(u) != info["n_lol"] + info["n_fi_kept"]:
            raise AssertionError(f"G6 {day}: row count")
        atomic_parquet(u, target, {"union_rule": RULE, **(meta or {}), **{k: str(v) for k, v in info.items()}})
        out[day] = info
        print(f"  {day}: lol {info['n_lol']:,} fi {info['n_fi']:,} kept {info['n_fi_kept']:,} dropped {info['n_fi_dropped_near_lol']:,}", flush=True)
    return out


def write_manifest(out_dir: pathlib.Path, info: dict, source: str, licence: str) -> pathlib.Path:
    """MANIFEST.json in r12_build's format (days with status ok) + the sources the directory contains."""
    p = pathlib.Path(out_dir) / "MANIFEST.json"
    if p.exists():
        raise FileExistsError(f"refusing to overwrite {p}")
    man = {"source": source, "licence": licence, "contains": list(CONTAINS), "rule": RULE,
            "days": [{"day": d, "status": "ok", "counts": {k: int(v) for k, v in c.items()}} for d, c in sorted(info.items())]}
    p.write_text(json.dumps(man, indent=1))
    return p
