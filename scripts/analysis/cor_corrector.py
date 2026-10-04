"""COR — a corrector cross-fitted on FM1's 12-month out-of-fold base predictions.

Registration: plans/PREREG_cor_crossfit_corrector_2026_10_04.md (REG 3c75c5ea, AMEND-A1 6c62c14b). The base OOF comes
from `python -m prc validate --protocol lomo` (each 2025 month predicted by a fit on the other eleven). A LightGBM
corrector learns the matched-lane residual y - pred on months {2, 4, 5, 6, 8, 10, 11, 12}, early-stops on {3, 9}, and
corrects January and July, which it never sees; base vs corrected is scored by FM1's compare and promotion gate.

    python scripts/analysis/cor_corrector.py --oof data/fm1/lomo_1004/lomo_screen --config configs/fm1_v50.yaml \\
        --set adsb.stack.enabled=false --set matched.blend.catboost_weight=0 --out reports/cor_report.json

Features come from FM1's own feature path, raw columns only (no encoding fitted on every month); the secondary COR-E
adds encodings refitted on the training months only. Reads 2025 data; no board read, no upload.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prc.pipeline.fm1 import features as F  # noqa: E402
from prc.pipeline.fm1 import protocols as P  # noqa: E402
from prc.pipeline.fm1 import rows as R  # noqa: E402

TRAIN_MONTHS = (2, 4, 5, 6, 8, 10, 11, 12)
ES_MONTHS = (3, 9)
EVAL_MONTHS = (1, 7)
MIN_BODY_GAIN = 150.0        # bmse: below ~100 the board cannot rank changes (RUL); the registered TRUE bar
PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "seed": 0,
          "deterministic": True, "force_row_wise": True, "num_threads": 12, "verbosity": -1}
PARAMS_ROUNDS = 3000
PATIENCE = 200
OOF_COLS = ("MVT_ID_mvt", "month", "day", "ap", "y", "pred", "lane")
Y_TOL_S = 0.5
SEED = 20260928           # the registered bootstrap seed (prc.api.compare's default)


# ---------------------------------------------------------------------------------------------- inputs

def load_oof(oof_dir) -> pd.DataFrame:
    """Every oof_M*.parquet of a LOMO run: twelve months, unique ids, the columns compare needs."""
    files = sorted(pathlib.Path(oof_dir).glob("oof_M*.parquet"))
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    missing = [c for c in OOF_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"{oof_dir}: OOF records miss column(s) {missing}")
    months = sorted(int(m) for m in frame.month.unique())
    if months != list(range(1, 13)):
        raise ValueError(f"{oof_dir}: OOF months {months}, expected 1-12")
    if frame.MVT_ID_mvt.duplicated().any():
        raise ValueError(f"{oof_dir}: {int(frame.MVT_ID_mvt.duplicated().sum())} duplicate MVT_ID_mvt")
    return frame


def raw_feature_columns(design_columns) -> list:
    """The design without any target / delta encoding (fitted on every month, they would carry January and July
    labels); refuses a label-derived column."""
    R.assert_design_clean(list(design_columns), "corrector design")
    return [c for c in design_columns if not c.startswith(("te_", "de_"))]


def join(oof: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """The matched-lane OOF rows with their feature row; refuses a matched row without one or a label mismatch."""
    m = oof[oof.lane == "matched"]
    f = feats.drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    absent = ~m.MVT_ID_mvt.isin(f.index)
    if absent.any():
        raise ValueError(f"{int(absent.sum())} matched OOF rows have no feature row")
    f = f.reindex(m.MVT_ID_mvt.to_numpy())
    bad = np.abs(f.y.to_numpy(float) - m.y.to_numpy(float)) > Y_TOL_S
    if bad.any():
        raise ValueError(f"labels disagree on {int(bad.sum())} rows between the OOF records and the features")
    extra = f.drop(columns=["y"]).reset_index(drop=True)
    return pd.concat([m.reset_index(drop=True), extra], axis=1)


def split(j: pd.DataFrame) -> tuple:
    mo = j.month.astype(int)
    return j[mo.isin(TRAIN_MONTHS)], j[mo.isin(ES_MONTHS)], j[mo.isin(EVAL_MONTHS)]


# ---------------------------------------------------------------------------------------------- the corrector

def fit_corrector(train: pd.DataFrame, es: pd.DataFrame, cols):
    """LightGBM on the residual y - pred, early stopping on the ES months (registered parameters)."""
    import lightgbm as lgb

    dtr = lgb.Dataset(train[cols].to_numpy(float), (train.y - train.pred).to_numpy(float), free_raw_data=False)
    des = lgb.Dataset(es[cols].to_numpy(float), (es.y - es.pred).to_numpy(float), reference=dtr)
    return lgb.train(PARAMS, dtr, num_boost_round=PARAMS_ROUNDS, valid_sets=[des],
                     callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])


def apply_correction(oof: pd.DataFrame, corr: pd.Series) -> pd.DataFrame:
    """pred + correction on matched rows that have one (floored at 1 s); every other row unchanged."""
    out = oof.copy()
    m = (out.lane == "matched") & out.MVT_ID_mvt.isin(corr.index)
    out.loc[m, "pred"] = np.maximum(out.loc[m, "pred"].to_numpy(float) +
                                    corr.reindex(out.loc[m, "MVT_ID_mvt"].to_numpy()).to_numpy(float), 1.0)
    return out


def training_encodings(train: pd.DataFrame, keys, smooth: float):
    """COR-E: te (y), de (delta, when present) and re (y - pred) encodings per key, fitted on the training rows only
    (FM1's EncodingTables); returns frame -> {column: values}."""
    targets = {"te": train.y.to_numpy(float), "re": (train.y - train.pred).to_numpy(float)}
    if "delta" in train.columns:
        targets["de"] = train.delta.to_numpy(float)
    enc = F.EncodingTables.fit(train, targets, list(keys), smooth)
    return enc.transform


def verdict_of(promote: bool, body_gain: float) -> str:
    """H-COR: TRUE needs the promotion gate AND a body gain of at least MIN_BODY_GAIN bmse; anything else is FALSE."""
    return "TRUE" if promote and body_gain >= MIN_BODY_GAIN else "FALSE"


def _arm(j, oof_eval, cols, n_boot):
    tr, es, ev = split(j)
    booster = fit_corrector(tr, es, cols)
    corr = pd.Series(booster.predict(ev[cols].to_numpy(float)), index=ev.MVT_ID_mvt.to_numpy())
    rep = P.compare(oof_eval, apply_correction(oof_eval, corr), n_boot=n_boot, seed=SEED)
    return {"compare": rep, "gate": P.promotion_gate(rep), "best_iteration": int(booster.best_iteration),
            "n_train": int(len(tr)), "n_es": int(len(es)), "n_eval_matched": int(len(ev)), "columns": list(cols)}


def run(oof_dir, builder, feature_cols, adsb_cols, keys, n_boot: int = 2000, smooth: float = 50.0) -> dict:
    """The registered primary (raw features + pred) and the secondaries (no ADS-B; COR-E with training-month
    encodings). `builder(months)` returns the matched feature rows (MVT_ID_mvt, y, feature_cols, keys[, delta])."""
    oof = load_oof(oof_dir)
    j = join(oof, builder(tuple(range(1, 13))))
    oof_eval = oof[oof.month.astype(int).isin(EVAL_MONTHS)]
    cols = list(feature_cols) + ["pred"]
    primary = _arm(j, oof_eval, cols, n_boot)
    verdict = verdict_of(primary["gate"]["promote"], primary["compare"]["body"]["gain_bmse"])
    no_adsb = _arm(j, oof_eval, [c for c in cols if c not in set(adsb_cols)], n_boot)
    tr = j[j.month.astype(int).isin(TRAIN_MONTHS)]
    encode = training_encodings(tr, keys, smooth)
    je = pd.concat([j.reset_index(drop=True), pd.DataFrame(encode(j))], axis=1)
    cor_e = _arm(je, oof_eval, cols + list(encode(j.head(1)).keys()), n_boot)
    return {"registration": "plans/PREREG_cor_crossfit_corrector_2026_10_04.md", "verdict": verdict,
            "min_body_gain": MIN_BODY_GAIN, "primary": primary, "secondary": {"no_adsb": no_adsb, "cor_e": cor_e}}


# ---------------------------------------------------------------------------------------------- FM1 wiring

def fm1_builder(cfg, log=print):
    """builder(months) from FM1's feature path: each month's raw file, the registry rows, build_frames(serve=False)
    with weather and routes, ADS-B attached from the training table; keeps the ids, y, delta, raw design columns and
    the encoding keys, float32 for the numeric columns."""
    from prc.pipeline.fm1 import engine as G

    weather, orl = G._weather(cfg), G._orl(cfg)
    cols = raw_feature_columns(F.matched_design_columns(cfg))
    keys = list(F.stand_ab().ENC_KEYS)

    def builder(months):
        parts = []
        for mo in months:
            raw = pd.read_parquet(G.month_file(cfg, mo))
            raw = raw[G.in_registry(raw, cfg)]
            fr = F.build_frames(raw, serve=False, cfg=cfg, weather=weather, orl_routes=orl)
            m = fr.matched
            keep = ["MVT_ID_mvt", "y", "delta"] + [c for c in cols if c in m.columns] + [k for k in keys if k in m.columns]
            part = m[list(dict.fromkeys(keep))].copy()
            for c in cols:
                if c in part.columns and part[c].dtype.kind in "fi":
                    part[c] = part[c].astype("float32")
            parts.append(part)
            log(f"features month {mo}: {len(part):,} matched rows")
        return pd.concat(parts, ignore_index=True)
    return builder, cols, keys


def main(argv=None) -> int:
    from prc.pipeline.fm1 import knobs as K

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oof", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        raise SystemExit(f"{out} exists; a result is never overwritten")
    cfg = K.load_config(a.config, overrides=a.overrides)
    builder, cols, keys = fm1_builder(cfg)
    adsb = F.adsb_columns(cfg.path("adsb.table_dir")) if cfg["features.adsb"] else []
    rep = run(a.oof, builder, cols, adsb, keys, smooth=F.stand_ab().SMOOTH)
    out.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str))
    p = rep["primary"]
    print(json.dumps({"verdict": rep["verdict"], "body": p["compare"]["body"], "gate": p["gate"],
                      "gain_all": p["compare"]["gain_bmse"], "no_adsb_body": rep["secondary"]["no_adsb"]["compare"]["body"],
                      "cor_e_body": rep["secondary"]["cor_e"]["compare"]["body"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
