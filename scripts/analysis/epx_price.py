"""EPX (plans/PREREG_epx_lfpg_eobt_pricing_2026_10_02.md) — price the LFPG EOBT practice change on held-out 2025 rows.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/epx_price.py run --artifact <July-held-out artefact dir> \
        --out data/fm1/sel/epx

In July 2026 the EOBT_1 field at LFPG is later than LOBT on 35 % of matched departures (6.6 % in July 2025). The 2026
pattern (share revised and the size of the revision, per bin of AOBT_3 - LOBT) is read from the scored file's clocks —
no label exists there — and imposed on July 2025 LFPG rows; a model that never saw July 2025 predicts the original and
the perturbed month; the difference in squared error is the price. The serve-time clip EOBT := min(EOBT, LOBT) is priced
in the same run. Assumption A (stated in the registration): only the EOBT field changed, not the flights or their labels.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
EDGES = (0.0, 300.0, 600.0, 900.0, 1800.0, 3600.0)        # right-closed bins of AOBT_3 - LOBT (s): <= 0, (0, 300], ... > 3600
N_BINS = len(EDGES) + 1
BOARD_N = 344_841.0
SEEDS = (0, 1, 2, 3, 4)
AIRPORT, MONTH = "LFPG", 7
SIM_DRIFT_MAX, SIM_SHARE_TOL = 0.07, 0.02                  # the registered simulator check


def bin_of(al) -> np.ndarray:
    """Bin index (0 .. N_BINS - 1) of AOBT_3 - LOBT in seconds; edges are right-closed."""
    al = np.asarray(al, dtype="float64")
    if not np.isfinite(al).all():
        raise ValueError("bin_of: non-finite AOBT_3 - LOBT")
    return np.searchsorted(np.asarray(EDGES), al, side="left")


def pattern(al, el) -> dict:
    """Per bin: n, p = share of rows with EOBT - LOBT > 0, pool = those positive revisions."""
    b, el = bin_of(al), np.asarray(el, dtype="float64")
    out = {}
    for k in range(N_BINS):
        e = el[b == k]
        out[k] = {"n": int(len(e)), "p": float((e > 0).mean()) if len(e) else 0.0, "pool": e[e > 0]}
    return out


def _q(pat_from: dict, pat_to: dict, k: int) -> float:
    pf, pt = pat_from[k]["p"], pat_to[k]["p"]
    return max(0.0, (pt - pf) / (1.0 - pf)) if pf < 1.0 else 0.0


def expected_share(al, el, pat_from: dict, pat_to: dict, scale: float) -> float:
    """The expected share of revised rows after impose(..., scale): those already revised plus, per bin, the
    not-yet-revised rows times min(1, scale * q_b)."""
    b, el = bin_of(al), np.asarray(el, dtype="float64")
    tot = float((el > 0).sum())
    for k in range(N_BINS):
        tot += float(((b == k) & (el <= 0)).sum()) * min(1.0, scale * _q(pat_from, pat_to, k))
    return tot / len(el)


def calibrate_scale(al, el, pat_from: dict, pat_to: dict, target_share: float) -> float:
    """EPX.2: the common factor on every q_b that brings the expected revised share to `target_share` (the per-bin
    pattern over-shoots when the month it is imposed on has more delayed flights than the month it was read from)."""
    el = np.asarray(el, dtype="float64")
    if target_share < float((el > 0).mean()) - 1e-12:
        raise ValueError(f"target share {target_share:.3f} is below the share already revised")
    qs = [_q(pat_from, pat_to, k) for k in range(N_BINS)]
    pos = [q for q in qs if q > 0]
    hi = 1.0 / min(pos) if pos else 1.0
    if expected_share(al, el, pat_from, pat_to, hi) < target_share - 1e-12:
        raise ValueError(f"cannot reach a revised share of {target_share:.3f} with this pattern")
    lo = 0.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if expected_share(al, el, pat_from, pat_to, mid) < target_share else (lo, mid)
    return (lo + hi) / 2.0


def impose(al, el, pat_from: dict, pat_to: dict, rng, scale: float = 1.0) -> tuple:
    """Impose `pat_to`'s revision practice on rows that follow `pat_from`: a not-yet-revised row (EOBT - LOBT <= 0) in bin
    b is revised with probability min(1, scale * q_b), q_b = max(0, (p_to - p_from) / (1 - p_from)), its EOBT - LOBT drawn
    from pat_to's pool for b. Returns (new EOBT - LOBT, revised mask). Already-revised rows are never touched."""
    b, el = bin_of(al), np.asarray(el, dtype="float64")
    new, revised = el.copy(), np.zeros(len(el), dtype=bool)
    for k in range(N_BINS):
        q = min(1.0, scale * _q(pat_from, pat_to, k))
        idx = np.flatnonzero((b == k) & (el <= 0))
        if q <= 0.0 or len(idx) == 0:
            continue
        pick = idx[rng.random(len(idx)) < q]
        if len(pick) == 0:
            continue
        pool = np.asarray(pat_to[k]["pool"], dtype="float64")
        if len(pool) == 0:
            raise ValueError(f"bin {k}: revisions to impose but an empty 2026 pool")
        new[pick] = rng.choice(pool, size=len(pick), replace=True)
        revised[pick] = True
    return new, revised


def clip_eobt(eobt, lobt) -> np.ndarray:
    """The serve-time repair: EOBT := min(EOBT, LOBT); a missing value on either side leaves EOBT as it is."""
    eobt, lobt = np.asarray(eobt, dtype="float64"), np.asarray(lobt, dtype="float64")
    return np.where(np.isfinite(eobt) & np.isfinite(lobt), np.minimum(eobt, lobt), eobt)


def day_block_ci(row_delta, day, *, n_boot: int = 2000, seed: int = 20260928) -> tuple:
    """(point, lo, hi) of sum(row_delta) / BOARD_N with a 95 % bootstrap over whole days."""
    per_day = pd.Series(np.asarray(row_delta, dtype="float64")).groupby(np.asarray(day)).sum().to_numpy()
    rng = np.random.default_rng(seed)
    boots = per_day[rng.integers(0, len(per_day), (n_boot, len(per_day)))].sum(axis=1) / BOARD_N
    return float(per_day.sum() / BOARD_N), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _sec(a, b) -> np.ndarray:  # pragma: no cover - thin datetime helper used by the CLI only
    return (pd.to_datetime(a, utc=True) - pd.to_datetime(b, utc=True)).dt.total_seconds().to_numpy(dtype="float64")


def main(argv=None) -> int:  # pragma: no cover - I/O shell around the tested functions (real files, the model)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--artifact", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--match-share", action="store_true",
                   help="EPX.2: scale the per-bin probabilities so the expected revised share equals the 2026 share")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from prc import api
    from prc.pipeline.fm1 import engine as G, features as F, protocols as P, rows as R
    spec = importlib.util.spec_from_file_location("drf_audit", ROOT / "scripts" / "analysis" / "drf_audit.py")
    DRF = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DRF)
    out = pathlib.Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to write into the non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    art = api.load_artifact(a.artifact)
    cfg = art.config
    if MONTH in art.fit_report["train_months"]:
        raise SystemExit(f"the artefact was trained on month {MONTH}: not a held-out model")
    quiet = lambda *_a, **_k: None  # noqa: E731

    def sel(d):
        return ((d.PHASE_mvt.astype(str) == "DEP") & (d.ADEP_mvt.astype(str) == AIRPORT) & d.AOBT_3_flt.notna()
                & d.EOBT_1_flt.notna() & d.LOBT_flt.notna()).to_numpy()

    raw = pd.read_parquet(G.month_file(cfg, MONTH))
    raw = raw[G.in_registry(raw, cfg)].reset_index(drop=True)
    sc = pd.read_parquet(cfg.path("data.scored_file"))
    sc = sc[G.in_registry(sc, cfg)]
    sc = sc[pd.to_datetime(sc.MVT_TIME_UTC_mvt, utc=True).dt.month == MONTH].reset_index(drop=True)
    m25, m26 = sel(raw), sel(sc)
    al25, el25 = _sec(raw.AOBT_3_flt[m25], raw.LOBT_flt[m25]), _sec(raw.EOBT_1_flt[m25], raw.LOBT_flt[m25])
    al26, el26 = _sec(sc.AOBT_3_flt[m26], sc.LOBT_flt[m26]), _sec(sc.EOBT_1_flt[m26], sc.LOBT_flt[m26])
    pat25, pat26 = pattern(al25, el25), pattern(al26, el26)
    scale = calibrate_scale(al25, el25, pat25, pat26, float((el26 > 0).mean())) if a.match_share else 1.0
    weather = F.stand_ab().load_weather(cfg.path("data.weather_dir")) if cfg["features.weather"] else None
    orl = F.read_orl_routes(cfg.path("features.orl_routes_file")) if cfg["features.orl_route_length"] else None

    def feats(d, serve):
        fr = F.build_frames(d, serve=serve, cfg=cfg, weather=weather, orl_routes=orl).matched
        return fr[fr.ap == AIRPORT]

    f26 = feats(sc, True)
    idx = raw.MVT_ID_mvt.to_numpy(float)
    day = pd.Series(pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").to_numpy(), index=idx)
    apt = pd.Series(raw.ADEP_mvt.astype(str).to_numpy(), index=idx)

    def record(d):
        served, labels = R.hide_labels(d)
        rec = P.score(api.predict(art, served, log=quiet), labels, "EPX", MONTH, day, apt)
        rec = rec[(rec.ap == AIRPORT) & (rec.lane == "matched")]
        return rec.set_index("MVT_ID_mvt").sort_index()

    def with_el(new_el):
        d = raw.copy()
        lobt = pd.to_datetime(d.loc[m25, "LOBT_flt"], utc=True)
        d.loc[m25, "EOBT_1_flt"] = (lobt + pd.to_timedelta(new_el, unit="s")).astype(d.EOBT_1_flt.dtype).to_numpy()
        return d

    def clipped(d):
        c = d.copy()
        e, lo = pd.to_datetime(c.loc[m25, "EOBT_1_flt"], utc=True), pd.to_datetime(c.loc[m25, "LOBT_flt"], utc=True)
        c.loc[m25, "EOBT_1_flt"] = e.where(e <= lo, lo).astype(c.EOBT_1_flt.dtype).to_numpy()
        return c

    base = record(raw)
    y = base.y.to_numpy(float)
    e0 = (y - base.pred.to_numpy(float)) ** 2
    clip0 = record(clipped(raw)).reindex(base.index)
    dC = (y - clip0.pred.to_numpy(float)) ** 2 - e0
    dD, dR, sim = [], [], []
    for s in SEEDS:
        new_el, revised = impose(al25, el25, pat25, pat26, np.random.default_rng(s), scale=scale)
        d = with_el(new_el)
        fp = feats(d, False)
        chk = DRF.drift(fp, f26, ["eobt_lobt", "aobt_eobt", "eobt_p"]).to_dict()
        chk["revised_share"] = float((new_el > 0).mean())
        chk["n_revised_new"] = int(revised.sum())
        sim.append(chk)
        rp = record(d).reindex(base.index)
        rc = record(clipped(d)).reindex(base.index)
        dD.append((y - rp.pred.to_numpy(float)) ** 2 - e0)
        dR.append((y - rc.pred.to_numpy(float)) ** 2 - e0)
        print(json.dumps({"seed": s, **{k: round(v, 4) if isinstance(v, float) else v for k, v in chk.items()},
                          "D": round(dD[-1].sum() / BOARD_N, 1), "R": round(dR[-1].sum() / BOARD_N, 1)}), flush=True)
    days = base.day.to_numpy()
    res = {"n_rows": int(len(base)), "base_rmse": float(np.sqrt(e0.mean())), "base_bmse": float(e0.sum() / BOARD_N),
           "target_share_2026": float((el26 > 0).mean()), "share_2025": float((el25 > 0).mean()),
           "simulator": sim,
           "simulator_ok": bool(all(max(c["eobt_lobt"], c["aobt_eobt"], c["eobt_p"]) <= SIM_DRIFT_MAX
                                    and abs(c["revised_share"] - float((el26 > 0).mean())) <= SIM_SHARE_TOL for c in sim)),
           "D": day_block_ci(np.mean(dD, axis=0), days), "D_by_seed": [float(x.sum() / BOARD_N) for x in dD],
           "R": day_block_ci(np.mean(dR, axis=0), days), "R_by_seed": [float(x.sum() / BOARD_N) for x in dR],
           "C": day_block_ci(dC, days), "artifact": str(a.artifact), "scale": float(scale),
           "n_revised_rows_2026": int((el26 > 0).sum()), "n_rows_2026": int(len(el26))}
    (out / "epx.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_rows", "base_rmse", "base_bmse", "simulator_ok", "D", "R", "C")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
