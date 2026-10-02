"""ACR (plans/PREREG_acr_adsb_coverage_regime_2026_10_02.md) — price the 2025 -> 2026 change in which flights have ADS-B, and
the training-time dropout repair, on the held-out July 2025 file.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/acr_price.py run --m0 <held-out artefact> --m1 <dropout artefact> \
        --out data/fm1/sel/acr

Scenario (the 2026 July coverage pattern, masking only): every LIRF July row becomes a network blackout; a seeded 33.7 % of
EDDF's joined July rows become unheard flights (features.set_unheard — the same definition the training dropout uses);
every other airport as recorded. Applied to a scratch copy of the ADS-B tables, predicted through FM1's own path with the
labels hidden, scored on matched rows. D0 = cost of the scenario to M0; G = M1's gain over M0 under it; K = M1's cost
where coverage is as recorded. Verdict = the registered rule (G >= 100 bmse with a CI above 0, and K <= G / 2).
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
MONTH = 7
SEEDS = (0, 1, 2, 3, 4)
BLACKOUT_AIRPORTS = ("LIRF",)                    # 2026 July joined share 0.2 % (2025: 93.5 %)
PARTIAL = {"EDDF": 1.0 - 0.591 / 0.892}          # 2026 July 59.1 % vs 2025 89.2 %: this share of joined rows unheard
DROPOUT_KNOBS = ("adsb.train_dropout.flight", "adsb.train_dropout.day")
NOT_COMPARED = (*DROPOUT_KNOBS, "run.artifacts_dir")
G_MIN = 100.0


def _epx():
    spec = importlib.util.spec_from_file_location("epx_price", ROOT / "scripts" / "analysis" / "epx_price.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pair(cfg0, cfg1, knobs) -> None:
    """M0 and M1 must differ in the dropout knobs and in nothing else that can change a prediction."""
    diff = sorted(p for p in knobs if p not in NOT_COMPARED and cfg0[p] != cfg1[p])
    if diff:
        raise ValueError(f"M0 and M1 differ outside the dropout knobs: {diff}")
    if not any(cfg1[p] > 0 for p in DROPOUT_KNOBS) or any(cfg0[p] > 0 for p in DROPOUT_KNOBS):
        raise ValueError("M1 must train with dropout and M0 without")


def scenario_table(table: pd.DataFrame, adsb_cols, seed: int, set_unheard, month: int = MONTH) -> pd.DataFrame:
    """A copy of an ADS-B table with the 2026 July coverage pattern imposed on its `month` rows (see the docstring)."""
    out = table.copy()
    in_month = (out.month == month).to_numpy()
    ap = out.ap.astype(str).to_numpy()
    blackout = in_month & np.isin(ap, BLACKOUT_AIRPORTS)
    set_unheard(out, adsb_cols, blackout, network=True)
    rng = np.random.default_rng([int(seed), 7])
    joined = out.joined.to_numpy(dtype="float64", na_value=np.nan) == 1
    draw = rng.random(len(out))
    partial = np.zeros(len(out), dtype=bool)
    for a, share in PARTIAL.items():
        partial |= in_month & (ap == a) & joined & (draw < share)
    set_unheard(out, adsb_cols, partial, network=False)
    return out


def stats(e0_asis, e0_scen, e1_asis, e1_scen, day, ci) -> dict:
    """D0, G, K (bmse, point + CI) from per-row squared errors; *_scen are lists over seeds."""
    e0_asis, e1_asis = np.asarray(e0_asis, float), np.asarray(e1_asis, float)
    s0, s1 = np.mean(np.asarray(e0_scen, float), axis=0), np.mean(np.asarray(e1_scen, float), axis=0)

    def named(delta):
        point, lo, hi = ci(delta, day)
        return {"point": point, "lo": lo, "hi": hi}
    return {"D0": named(s0 - e0_asis), "G": named(s0 - s1), "K": named(e1_asis - e0_asis)}


def verdict(res: dict) -> dict:
    g, k = res["G"]["point"], res["K"]["point"]
    ok_g = g >= G_MIN and res["G"]["lo"] > 0
    ok_k = k <= g / 2
    return {"H_ACR": "TRUE" if ok_g and ok_k else "FALSE", "G_clears": bool(ok_g), "K_within_half_of_G": bool(ok_k)}


def main(argv=None) -> int:  # pragma: no cover - real files, the two models (the parts above are tested)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--m0", required=True)
    r.add_argument("--m1", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from prc import api
    from prc.pipeline.fm1 import engine as G, features as F, knobs as K, protocols as P, rows as R
    EPX = _epx()
    out = pathlib.Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to write into the non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    m0, m1 = api.load_artifact(a.m0), api.load_artifact(a.m1)
    check_pair(m0.config, m1.config, K.KNOBS)
    for m in (m0, m1):
        if MONTH in m.fit_report["train_months"]:
            raise SystemExit(f"{m.key} was trained on month {MONTH}: not a held-out model")
    cfg = m0.config
    raw = pd.read_parquet(G.month_file(cfg, MONTH))
    raw = raw[G.in_registry(raw, cfg)].reset_index(drop=True)
    idx = raw.MVT_ID_mvt.to_numpy(float)
    dayof = pd.Series(pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").to_numpy(), index=idx)
    apt = pd.Series(raw.ADEP_mvt.astype(str).to_numpy(), index=idx)
    served, labels = R.hide_labels(raw)
    quiet = lambda *_a, **_k: None  # noqa: E731

    def record(art):
        rec = P.score(api.predict(art, served, log=quiet), labels, "ACR", MONTH, dayof, apt)
        return rec[rec.lane == "matched"].set_index("MVT_ID_mvt").sort_index()

    tdir = cfg.path("adsb.table_dir")
    cols = F.adsb_columns(tdir)
    table = pd.read_parquet(tdir / "train.parquet")
    base0 = record(m0)
    y = base0.y.to_numpy(float)
    err = lambda rec: (y - rec.reindex(base0.index).pred.to_numpy(float)) ** 2  # noqa: E731
    e0_asis, e1_asis = err(base0), err(record(m1))
    e0_scen, e1_scen = [], []
    for s in SEEDS:
        sdir = out / f"tables_seed{s}"
        sdir.mkdir()
        scenario_table(table, cols, s, F.set_unheard).to_parquet(sdir / "train.parquet", index=False)
        for n in ("serve.parquet", "report.json"):
            shutil.copy2(tdir / n, sdir / n)
        for m, acc in ((m0, e0_scen), (m1, e1_scen)):
            acc.append(err(record(dataclasses.replace(m, config=m.config.replace(**{"adsb.table_dir": str(sdir)})))))
        print(json.dumps({"seed": s, "D0": round(float((e0_scen[-1] - e0_asis).sum() / EPX.BOARD_N), 1),
                          "G": round(float((e0_scen[-1] - e1_scen[-1]).sum() / EPX.BOARD_N), 1)}), flush=True)
    days = base0.day.to_numpy()
    res = stats(e0_asis, e0_scen, e1_asis, e1_scen, days, EPX.day_block_ci)
    res["by_airport"] = {}
    aps = base0.ap.astype(str).to_numpy()
    for apc in sorted(set(aps)):
        sel = aps == apc
        res["by_airport"][apc] = {k: float(v[sel].sum() / EPX.BOARD_N) for k, v in (
            ("D0", np.mean(e0_scen, axis=0) - e0_asis), ("G", np.mean(e0_scen, axis=0) - np.mean(e1_scen, axis=0)),
            ("K", e1_asis - e0_asis))}
    res.update(verdict(res), n_rows=int(len(base0)), m0=m0.key, m1=m1.key, m1_dropout=m1.fit_report.get("adsb_dropout"),
               rmse_m0_asis=float(np.sqrt(e0_asis.mean())), rmse_m1_asis=float(np.sqrt(e1_asis.mean())))
    (out / "acr.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("D0", "G", "K", "H_ACR", "n_rows")}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
