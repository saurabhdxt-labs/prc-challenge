"""FM1 validation protocols and the paired comparison (plans/PREREG_fm1_2026_09_28.md §5 + Amendment A2).

    S         PRIMARY: F1 = leave-January-out, F2 = leave-July-out (the scored months; a top team reports 12-month CV misled
              them). Early stopping on months 3 + 9.
    swap      SECONDARY, strict: fit on January only -> test July; fit on July only -> test January; early stopping on the
              training month's own days >= 24.
    forward   TERTIARY (informational): F3 = fit Jan-Jun -> test Jul; F4 = fit Jan-Sep -> test Oct-Dec.
    lomo      every month held out once (12 folds); forward_monthly: fit <= m -> test m + 1, m = 6..11.

Per fold: fit on the training months (forecaster: MTX off), hide the test months' labels (rows.hide_labels), predict them
through the full stage plan, and score the departures with a valid label (y finite, > 0). Records (the OOF file) carry
MVT_ID_mvt, fold, month, day, ap, y, pred, decided_by, lane and every component output. Units: MSE per row, RMSE, and
board MSE (bmse = SSE / 344,841, the board's row count).

Leakage guard: with the ADS-B stack on, a fold whose test months trained the stored stack (adsb.stack.train_months) is
refused, so the day-4 base-learner study runs with the stack off unless the stack is retrained per fold (not in FM1 days 1-3).

compare(a, b): paired on common ids, gain = (SSE_a - SSE_b) / 344,841 (positive = b better), 95 % day-block bootstrap CI
(days resampled within month: mtx_calib.gain_ci, the project's registered bootstrap), plus the concentration exclusions
(the dominant airport dropped; the top-3 and top-10 rows by |row gain| dropped), the body gain (y <= 3,600 s: the share
that predicts the board, plans/PREREG_rul_ruler_validation_2026_10_04.md) with and without its top 10 rows, and the gain
and body gain per month. promotion_gate(report): promote only when the body CI is above zero, the body gain survives
dropping its top 10 rows, and it is positive in January and in July separately.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import artifact as A
from . import engine as G
from . import errors as E
from . import rows as R
from . import shipped

BOARD_N = 344_841.0
BODY_MAX_S = 3600.0          # the body: flights of at most one hour; its gain predicts the board, the tail's does not
                             # (plans/PREREG_rul_ruler_validation_2026_10_04.md)
SCORED_MONTHS = ("1", "7")
ALL = tuple(range(1, 13))


@dataclass(frozen=True)
class Fold:
    name: str
    train: tuple
    es: tuple
    test: tuple
    es_days_from: int | None = None          # strict swap folds: early stopping on the training month's days >= this


def _fold(name, train, es, test, es_days_from=None) -> Fold:
    return Fold(name, tuple(sorted(train)), tuple(sorted(es)), tuple(sorted(test)), es_days_from)


SWAP_ES_DAY = 24
PROTOCOLS = {
    "S": (_fold("F1", set(ALL) - {1}, (3, 9), (1,)), _fold("F2", set(ALL) - {7}, (3, 9), (7,))),
    "swap": (_fold("X1to7", (1,), (), (7,), SWAP_ES_DAY), _fold("X7to1", (7,), (), (1,), SWAP_ES_DAY)),
    "forward": (_fold("F3", range(1, 7), (3,), (7,)), _fold("F4", range(1, 10), (3, 9), (10, 11, 12))),
    "lomo": tuple(_fold(f"M{m:02d}", set(ALL) - {m}, tuple(e for e in (3, 9) if e != m), (m,)) for m in ALL),
    "forward_monthly": tuple(_fold(f"W{m + 1:02d}", range(1, m + 1), (3, 9) if m >= 9 else (3,), (m + 1,))
                             for m in range(6, 12)),
}


def folds(protocol: str, available) -> tuple:
    """The protocol's folds restricted to the months that have a training file; a fold left without training, early-
    stopping, fit or test months is skipped (reported by run())."""
    if protocol not in PROTOCOLS:
        raise E.ConfigValueError(f"unknown protocol {protocol!r}; protocols are {sorted(PROTOCOLS)} and dry-run")
    av = set(available)
    out = []
    for f in PROTOCOLS[protocol]:
        tr = [m for m in f.train if m in av]
        es = [m for m in f.es if m in av and m in f.train]
        te = [m for m in f.test if m in av]
        out.append(_fold(f.name, tr, es, te, f.es_days_from))
    return tuple(out)


def _usable(f: Fold) -> bool:
    if not (f.test and f.train):
        return False
    if f.es_days_from is not None:
        return True
    return bool(f.es and [m for m in f.train if m not in f.es])


def split_of(f: Fold) -> dict:
    if f.es_days_from is not None:
        return {"train_months": list(f.train), "es_days_from": f.es_days_from}
    return {"train_months": list(f.train), "es_months": list(f.es)}


def tier_overrides(cfg) -> dict:
    if cfg["validate.tier"] == "screen":
        return {"learning_rate": cfg["validate.screen_learning_rate"], "seeds": [cfg["run.seeds"][0]]}
    return {}


def score(pred: pd.DataFrame, labels: pd.Series, fold: str, month: int, day: pd.Series, ap: pd.Series) -> pd.DataFrame:
    y = labels.reindex(pred.MVT_ID_mvt.to_numpy(float)).to_numpy(float)
    rec = pred.assign(fold=fold, month=month, y=y, pred=pred.TAXITIME_SEC_mvt.astype("float64"),
                      day=day.reindex(pred.MVT_ID_mvt.to_numpy(float)).to_numpy(),
                      ap=ap.reindex(pred.MVT_ID_mvt.to_numpy(float)).to_numpy())
    return rec[np.isfinite(rec.y) & (rec.y > 0)].reset_index(drop=True)


def summary(rec: pd.DataFrame) -> dict:
    e2 = (rec.y - rec.pred) ** 2
    by_ap = {str(a): {"n": int(len(g)), "rmse": float(np.sqrt(((g.y - g.pred) ** 2).mean()))} for a, g in rec.groupby("ap")}
    return {"n": int(len(rec)), "mse": float(e2.mean()), "rmse": float(np.sqrt(e2.mean())), "sse": float(e2.sum()),
            "bmse": float(e2.sum() / BOARD_N), "by_airport": by_ap,
            "decided_by": {str(k): int(v) for k, v in rec.decided_by.value_counts().sort_index().items()}}


def run(cfg, protocol: str, out_dir=None, allow_heavy: bool = False, log=G._log, save_models: bool = False) -> dict:
    """`save_models` (plan v3, owner-adopted 2026-09-28: every fit saves its boosters, feature manifest, predictions,
    row order, config, seed and hashes): each fold's fit is additionally saved as a full FM1 artefact under
    `<out_dir>/models/<fold name>/` (the package's own artefact writer; never overwritten). Needs `out_dir`. The
    default (False) writes exactly what it always did."""
    if cfg.mtx_enabled:
        raise E.ConfigValueError("validate runs the forecaster: calibration.mtx.enabled must resolve off (MTX has its own "
                                 "validation; docs/fm1/METHOD.md)")
    if save_models and out_dir is None:
        raise E.ConfigValueError("save_models=True needs out_dir: fold artefacts are written under "
                                 "<out_dir>/models/<fold name>/")
    avail = sorted(G.files_by_month(cfg))
    plan = folds(protocol, avail)
    stack_months = set(cfg["adsb.stack.train_months"]) if cfg["adsb.stack.enabled"] and cfg["lanes.matched"] else set()
    report = {"protocol": protocol, "config_hash": cfg.hash, "tier": cfg["validate.tier"], "folds": {}, "skipped": {}}
    out_dir = pathlib.Path(out_dir) if out_dir is not None else None
    if out_dir is not None:
        if out_dir.exists() and any(out_dir.iterdir()):
            raise E.ArtifactError(f"{out_dir} is not empty; validation outputs are never overwritten")
        out_dir.mkdir(parents=True, exist_ok=True)
    for f in plan:
        if not _usable(f):
            report["skipped"][f.name] = f"no usable months in the data (train {list(f.train)}, es {list(f.es)}, test {list(f.test)})"
            continue
        leak = sorted(set(f.test) & stack_months)
        if leak:
            raise E.ConfigValueError(f"fold {f.name} tests month(s) {leak} that trained the stored ADS-B stack "
                                     "(adsb.stack.train_months): leakage. Run with --set adsb.stack.enabled=false, or "
                                     "retrain the stack per fold")
        art = G.fit(cfg, save=False, allow_heavy=allow_heavy, tier_overrides=tier_overrides(cfg), log=log, split=split_of(f))
        recs = []
        for m in f.test:
            raw = pd.read_parquet(G.month_file(cfg, m))
            raw = raw[G.in_registry(raw, cfg)]
            served, labels = R.hide_labels(raw)
            pred = G.predict(art, served, log=log)
            idx = raw.MVT_ID_mvt.to_numpy(float)
            day = pd.Series(raw.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy(), index=idx)
            ap = pd.Series(raw.ADEP_mvt.astype(str).to_numpy(), index=idx)
            recs.append(score(pred, labels, f.name, m, day, ap))
        rec = pd.concat(recs, ignore_index=True)
        art_dir = art_manifest_sha = None
        if save_models:
            art_dir = A.save(art, parent=out_dir / "models" / f.name)
            art_manifest_sha = A.sha256_file(art_dir / A.MANIFEST)
            rec = rec.assign(artifact_dir=str(art_dir), artifact_manifest_sha256=art_manifest_sha)
        report["folds"][f.name] = {"train": list(f.train), "es": list(f.es), "es_days_from": f.es_days_from,
                                   "test": list(f.test), **summary(rec),
                                   "fit": art.fit_report}
        if art_dir is not None:
            report["folds"][f.name]["artifact_dir"] = str(art_dir)
            report["folds"][f.name]["artifact_manifest_sha256"] = art_manifest_sha
        if out_dir is not None:
            rec.to_parquet(out_dir / f"oof_{f.name}.parquet", index=False)
        log(f"fold {f.name}: n {len(rec):,}, rmse {report['folds'][f.name]['rmse']:.2f}")
    if out_dir is not None:
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    return report


def compare(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Paired comparison of two OOF records (b vs a; gain > 0 = b better) on their common ids."""
    ids = np.intersect1d(a.MVT_ID_mvt.to_numpy(float), b.MVT_ID_mvt.to_numpy(float))
    if len(ids) == 0:
        raise E.ValidationError("the two records share no row")
    A = a.set_index("MVT_ID_mvt").loc[ids]
    B = b.set_index("MVT_ID_mvt").loc[ids]
    if not np.array_equal(A.y.to_numpy(float), B.y.to_numpy(float)):
        raise E.ValidationError("the two records disagree on a label")
    y, pa, pb = A.y.to_numpy(float), A.pred.to_numpy(float), B.pred.to_numpy(float)
    cal = shipped.mtx_calib()
    G_, lo, hi = cal.gain_ci(y, pa, pb, A.day.to_numpy(), A.month.to_numpy(), BOARD_N, n_boot=n_boot, seed=seed)
    row = (y - pa) ** 2 - (y - pb) ** 2
    ap = A.ap.to_numpy().astype(str)
    by_ap = pd.Series(row).groupby(ap).sum() / BOARD_N
    dom = str(by_ap.abs().idxmax())
    keep = ap != dom
    g_ex_ap = cal.gain_ci(y[keep], pa[keep], pb[keep], A.day.to_numpy()[keep], A.month.to_numpy()[keep], BOARD_N,
                          n_boot=n_boot, seed=seed) if keep.any() else (0.0, 0.0, 0.0)
    day, month = A.day.to_numpy(), A.month.to_numpy()

    def ci(m):                                       # the registered bootstrap on a row subset
        g, l, h = cal.gain_ci(y[m], pa[m], pb[m], day[m], month[m], BOARD_N, n_boot=n_boot, seed=seed)
        return {"gain_bmse": g, "ci95": [l, h]}

    def without_top(m, k):                           # drop the k rows of the subset with the largest |row gain|
        idx = np.flatnonzero(m)
        keep = m.copy()
        keep[idx[np.argsort(-np.abs(row[idx]), kind="stable")[:k]]] = False
        return ci(keep)

    everything = np.ones(len(row), bool)
    body = y <= BODY_MAX_S
    months = sorted({str(int(v)) for v in month})
    return {"n": int(len(ids)), "gain_bmse": G_, "ci95": [lo, hi], "by_airport_bmse": {k: float(v) for k, v in by_ap.items()},
            "excluding_dominant_airport": {"airport": dom, "gain_bmse": g_ex_ap[0], "ci95": list(g_ex_ap[1:])},
            "excluding_top3_rows": without_top(everything, 3), "excluding_top10_rows": without_top(everything, 10),
            "body": {**ci(body), "n": int(body.sum())}, "body_excluding_top10_rows": without_top(body, 10),
            "by_month": {mo: ci(month.astype(int).astype(str) == mo) for mo in months},
            "body_by_month": {mo: ci(body & (month.astype(int).astype(str) == mo)) for mo in months}}


def promotion_gate(rep: dict) -> dict:
    """The promotion rule for a candidate's compare report (ruler v2): the body gain's 95 % CI is above zero, the body
    gain stays positive without its 10 largest rows, and it is positive in January and in July separately. Refuses a
    report that holds neither scored month."""
    months = rep.get("body_by_month", {})
    if not any(m in months for m in SCORED_MONTHS):
        raise E.ValidationError(f"the report holds no January and July body gain (months {sorted(months)})")
    checks = {"body_ci_low_above_zero": bool(rep["body"]["ci95"][0] > 0),
              "body_without_top10_positive": bool(rep["body_excluding_top10_rows"]["gain_bmse"] > 0),
              "body_positive_each_scored_month": all(months[m]["gain_bmse"] > 0 for m in SCORED_MONTHS if m in months)}
    return {"promote": all(checks.values()), "checks": checks}


def pool(records) -> pd.DataFrame:
    """Several OOF records (e.g. F1 and F2) as one; an id may appear once only."""
    recs = [records] if isinstance(records, pd.DataFrame) else list(records)
    out = pd.concat(recs, ignore_index=True)
    if not out.MVT_ID_mvt.is_unique:
        raise E.ValidationError(f"pooled records hold {int(out.MVT_ID_mvt.duplicated().sum())} duplicate id(s)")
    return out


def blend(a: pd.DataFrame, b: pd.DataFrame, w: float) -> pd.DataFrame:
    """Selection arm A2 offline: on the ids both records hold, pred = rint(w * pred_b + (1 - w) * pred_a) (the CBV
    half-blend on the taxi scale when w = 0.5); labels must agree. decided_by = 'blend_oof'."""
    if not 0.0 <= float(w) <= 1.0:
        raise E.ConfigValueError(f"w must be in [0, 1], got {w}")
    ids = np.intersect1d(a.MVT_ID_mvt.to_numpy(float), b.MVT_ID_mvt.to_numpy(float))
    A = a.set_index("MVT_ID_mvt").loc[ids]
    B = b.set_index("MVT_ID_mvt").loc[ids]
    if not np.array_equal(A.y.to_numpy(float), B.y.to_numpy(float)):
        raise E.ValidationError("the two records disagree on a label")
    out = A.reset_index()[["MVT_ID_mvt", *[c for c in ("fold", "month", "day", "ap", "y") if c in A.columns]]]
    out["pred"] = np.rint(float(w) * B.pred.to_numpy(float) + (1.0 - float(w)) * A.pred.to_numpy(float))
    out["decided_by"] = "blend_oof"
    return out
