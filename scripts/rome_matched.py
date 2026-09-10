"""`scripts/rome_matched.py` -- arm M of plans/PREREG_rome_matched_fill_2026_09_10.md.

    python scripts/rome_matched.py

On Rome's MATCHED fold-A holdout rows, blend arm F's prediction with the schedule-fill branch:

    M   = max(p * sp + (1 - p) * F, 1)          M_g = M where p >= 0.5, F elsewhere

p = mean over seeds of scripts/rome_fill.fit_predict_fill (the registered params and stake
weight) trained on every LIRF departure of the ten fold-A training months, with the NM-clock
columns of matched rows added (NaN on unmatched rows). No matched model is refitted: F is read
from `fold_preds_queue_order.parquet` (DELTA convention, reproduces 222.563), mapped to MVT_ID
through the order cache's positional contract (verified: 26,131 of 26,131 LIRF rows, labels equal).
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rome_fill as rf  # noqa: E402

NM_FEATS = ["proxy", "nmdelay", "eobt_p", "lobt_p", "iobt_p", "aobt_eobt", "lobt_sched", "eobt_sched"]
FOLD_A = (1, 7)
GATE = 0.5
BAR_S = 5.0
BODY_TOL_S = 1.0
W_MATCHED = 0.9846596
F_RECORD = ROOT / "data" / "cache_stand" / "fold_preds_queue_order.parquet"


def row_to_mvt_id(stand_dir, order_dir) -> np.ndarray:
    """MVT_ID for every row of the concatenated monthly stand cache, via the order cache, which
    holds MVT_ID for exactly those rows in the same order (stand_ab.build_order's contract)."""
    cs = sorted(glob.glob(str(pathlib.Path(stand_dir) / "training_2025-*.parquet")))
    co = sorted(glob.glob(str(pathlib.Path(order_dir) / "training_2025-*.parquet")))
    assert [pathlib.Path(p).name for p in cs] == [pathlib.Path(p).name for p in co], "stand/order file lists differ"
    ids = []
    for a, b in zip(cs, co):
        n = pq.read_metadata(a).num_rows
        ob = pq.read_table(b, columns=["MVT_ID_mvt"]).to_pandas().MVT_ID_mvt.to_numpy()
        assert len(ob) == n, f"row count differs for {pathlib.Path(a).name}: stand {n}, order {len(ob)}"
        ids.append(ob)
    return np.concatenate(ids)


def nm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The NM-clock columns (all serve-time: AOBT_3 / EOBT / LOBT / IOBT and the schedule)."""
    return pd.DataFrame({c: frame[c].to_numpy(dtype="float64") for c in NM_FEATS}, index=frame.index)


def _with_nm(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c in NM_FEATS:
        if c not in d.columns:
            d[c] = np.nan
    return d


def classifier_p(unm: pd.DataFrame, lirf: pd.DataFrame, rec: pd.DataFrame, seeds=rf.SEEDS) -> tuple:
    """({seed: p}, p_permuted) on the LIRF matched rows of `rec` (MVT_ID order), from the stake-
    weighted classifier trained on every LIRF departure of the fold-A training months + NM columns.
    Refuses a record whose labels differ from the LIRF frame's (a wrong row mapping)."""
    te = lirf.set_index("MVT_ID_mvt", drop=False).loc[rec.MVT_ID_mvt.to_numpy()]
    if not np.array_equal(te.y.to_numpy(dtype="float64"), rec.y.to_numpy(dtype="float64")):
        raise AssertionError("labels differ between the record and the LIRF frame: wrong row mapping")
    tr = _with_nm(rf.rome_rows(unm, lirf, [m for m in sorted(set(lirf.month)) if m not in FOLD_A]))
    te = _with_nm(te)
    ps = {s: rf.fit_predict_fill(tr, te, seed=s, extra_num=NM_FEATS) for s in seeds}
    pp = rf.fit_predict_fill(tr, te, seed=seeds[0], permute=True, extra_num=NM_FEATS)
    return ps, pp


def score(rec: pd.DataFrame, arms: dict, candidates, seeds, n_boot=rf.N_BOOT, bar=BAR_S, body_tol=BODY_TOL_S) -> dict:
    """The five registered clauses for every arm in `candidates` against arms['F'], on `rec`'s rows.
    `arms` holds '<arm>_seed<s>' for each candidate (clause 5); an arm named '<x>_perm' is scored
    as a control (its clause-1 result is reported, never a verdict). `bar` / `body_tol` default to arm
    M's registered +5.0 s / 1.0 s; amendment B.1 passes its own (+0.5 s / 0.2 s on all matched rows).
    The clause KEY names stay those of arm M whatever the bar, so records stay comparable."""
    import lgbm_fold as lf
    y, sp = rec.y.to_numpy(dtype="float64"), rec.sp.to_numpy(dtype="float64")
    se = {a: (np.asarray(v, dtype="float64") - y) ** 2 for a, v in arms.items()}
    rmse = {a: float(np.sqrt(e.mean())) for a, e in se.items()}
    fill = np.abs(y - sp) <= rf.FILL_TOL_S
    month = rec.month.to_numpy()
    others = [a for a in arms if a != "F" and not any(a.startswith(c + "_seed") for c in candidates)]
    boot = lf.paired_bootstrap(se, [(a, "F") for a in others], n_boot, lf.BOOT_SEED)
    clauses, verdict, extra = {}, {}, {}
    for a in candidates:
        gain = rmse["F"] - rmse[a]
        seed_r = [rmse[f"{a}_seed{s}"] for s in seeds]
        sd = float(np.std(seed_r, ddof=1)) if len(seeds) >= 2 else None
        by_month = {int(m): float(np.sqrt(se["F"][month == m].mean()) - np.sqrt(se[a][month == m].mean()))
                    for m in FOLD_A if (month == m).any()}
        body = float(np.sqrt(se[a][~fill].mean()) - np.sqrt(se["F"][~fill].mean()))
        c = {"c1_interval": bool(boot[f"{a}_vs_F"]["ci95"][0] > 0),
             "c2_gain_ge_5s": bool(gain >= bar),
             "c3_both_months": bool(len(by_month) == len(FOLD_A) and all(v > 0 for v in by_month.values())),
             "c4_body_bounded": bool(body <= body_tol),
             "c5_gain_over_2x_seed_sd": bool(sd is not None and gain > 2 * sd)}
        clauses[a] = c
        verdict[a] = "ESTABLISHED" if all(c.values()) else ("NOT WORKING" if not c["c1_interval"] else "INCONCLUSIVE")
        extra[a] = {"gain_s": gain, "ci95": boot[f"{a}_vs_F"]["ci95"], "seed_sd": sd, "gain_by_month": by_month,
                    "body_loss_s": body,
                    "fill_rows_rmse": {"F": float(np.sqrt(se["F"][fill].mean())), a: float(np.sqrt(se[a][fill].mean()))},
                    "projected_board_mse": float(W_MATCHED * (se["F"].sum() - se[a].sum()) / 339_015)}
    return {"n_rows": int(len(rec)), "fill_share": float(fill.mean()), "rmse": rmse, "clauses": clauses,
            "verdict": verdict, "detail": extra, "boot": boot}


def run(unm: pd.DataFrame, lirf: pd.DataFrame, rec: pd.DataFrame, seeds=rf.SEEDS, n_boot=rf.N_BOOT) -> dict:
    """`rec`: LIRF matched fold-A rows with MVT_ID_mvt, month, y, sp and F (taxi-time scale)."""
    rec = rec.reset_index(drop=True)
    ps, pp = classifier_p(unm, lirf, rec, seeds)
    p = np.mean(list(ps.values()), axis=0)
    sp, F = rec.sp.to_numpy(dtype="float64"), rec.F.to_numpy(dtype="float64")
    mix = lambda q: rf.mixture(q, sp, F)
    arms = {"F": F, "M": mix(p), "M_g": np.where(p >= GATE, mix(p), F), "M_perm": mix(pp)}
    for s, q in ps.items():
        arms[f"M_seed{s}"] = mix(q)
        arms[f"M_g_seed{s}"] = np.where(q >= GATE, mix(q), F)
    res = score(rec, arms, ("M", "M_g"), seeds, n_boot)
    fill = np.abs(rec.y.to_numpy() - sp) <= rf.FILL_TOL_S
    res.update(control_perm_passes_c1=bool(res["boot"]["M_perm_vs_F"]["ci95"][0] > 0), perm=res["boot"]["M_perm_vs_F"],
               p_mean_fill=float(p[fill].mean()), p_mean_body=float(p[~fill].mean()))
    return res


def load_record() -> pd.DataFrame:
    f = pd.read_parquet(F_RECORD, columns=["row", "month", "ap", "y", "proxy", "sp", "treatment"])
    yh = np.maximum(f.proxy - f.treatment, 1.0)
    r = float(np.sqrt(((yh - f.y) ** 2).mean()))
    if abs(r - 222.5632) > 1e-3:
        raise AssertionError(f"arm F's record recovers to {r:.4f}, not 222.5632: wrong convention or record")
    f["F"] = yh
    f["MVT_ID_mvt"] = row_to_mvt_id(ROOT / "data" / "cache_stand", ROOT / "data" / "cache_order")[f.row.to_numpy()]
    return f[f.ap == "LIRF"][["MVT_ID_mvt", "month", "ap", "y", "sp", "F"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "rome_matched.json"))
    a = ap.parse_args(argv)
    unm, lirf = rf.load_frames(print)
    res = run(unm, lirf, load_record())
    print(f"LIRF matched fold-A rows {res['n_rows']:,}, fill share {res['fill_share']:.3f}; mean p fill {res['p_mean_fill']:.3f} body {res['p_mean_body']:.3f}")
    for k, v in res["rmse"].items():
        print(f"  {k:10s} RMSE {v:8.2f}")
    for arm in ("M", "M_g"):
        d = res["detail"][arm]
        print(f"{arm}: gain {d['gain_s']:+.2f} s [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]  seed sd {d['seed_sd']}  by month {d['gain_by_month']}  "
              f"body loss {d['body_loss_s']:+.2f} s  fill-row RMSE {d['fill_rows_rmse']}  projected board {d['projected_board_mse']:+.0f} MSE")
        print(f"   clauses {res['clauses'][arm]} -> {res['verdict'][arm]}")
    print(f"control M_perm: {res['perm']['gain_s']:+.2f} [{res['perm']['ci95'][0]:+.2f}, {res['perm']['ci95'][1]:+.2f}]  passes c1: {res['control_perm_passes_c1']}")
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"json -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
