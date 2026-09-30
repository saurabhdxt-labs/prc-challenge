"""ARC — the same flight's ARR-row NM clocks for unmatched departures (plans/PREREG_arr_row_clocks_2026_09_16.md; no upload).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/arr_row_clocks.py              # the registered measurement
    /opt/homebrew/bin/python3.11 -B scripts/analysis/arr_row_clocks.py --transfer   # post-hoc 2026 diagnostic (not a clause)

An unmatched departure has |BLOCK − LOBT| > 3,606 s. When the flight lands at one of the ten airports, its ARR row carries NM's
LOBT and AOBT_3 for that departure. R_ord (LOBT within 1 h of SCHED ⇒ BLOCK is not SCHED ⇒ not a fill) predicts MVT − AOBT_3;
R_fill (LOBT re-filed > 1 h and NM's push agrees with it ⇒ BLOCK is not the push ⇒ fill) predicts sp. The join and the rules
read no label. Scored against the v16-equivalent 2025 LOMO incumbents; verdicts per the locked clauses.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

PREREG = ROOT / "plans" / "PREREG_arr_row_clocks_2026_09_16.md"
REPORT = ROOT / "reports" / "arr_row_clocks_2026_09_16.json"
TRANSFER = ROOT / "reports" / "arr_row_clocks_transfer_2026_09_16.json"
RAW = ROOT / "data" / "raw"
RANK = RAW / "ranking.parquet"
UNM = ROOT / "data" / "cache_stand" / "unm_physics_lomo.parquet"
POOL_MAP = ROOT / "reports" / "pool_map_v16_2026_09_13.json"
LCM = ROOT / "reports" / "lirf_chain_map_2026_09_12.json"
V16 = ROOT / "submissions" / "merry-quicksand_v16.parquet"

BOARD_N = 344_841.0
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
WIN_LO, WIN_HI = 600.0, 8 * 3600.0
SP_LO, SP_HI = 3_600.0, 24_000.0
ORD_TOL, FILL_TOL = 3_600.0, 3_606.0
G1_TOL = 0.005
N_BOOT, SEED = 2_000, 0
MIN_ROWS = 15
DEP_COLS = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
ARR_COLS = ["PHASE_mvt", "ADES_mvt", "ADEP_flt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]


def secs(t) -> np.ndarray:
    """UTC timestamps → epoch seconds (float, NaN for NaT); µs-safe."""
    return (pd.to_datetime(pd.Series(t), utc=True) - EPOCH).dt.total_seconds().to_numpy(dtype="float64")


# --------------------------------------------------------------------------- pure (unit-tested)

def join_arr(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """One row per dep row (same order): arr_lobt, arr_aobt3 in epoch s (NaN if no join).
    dep: MVT_ID_mvt, ADEP_mvt, ADES_mvt, FLIGHT_mvt, mvt (s). arr: ADEP_flt, ADES_mvt, CALLSIGN_flt, mvt (s), lobt (s), aobt3 (s).
    Callsign-matched only; ARR mvt in (dep mvt + 600, dep mvt + 8 h]; lobt and aobt3 present; earliest ARR mvt wins."""
    if dep.MVT_ID_mvt.duplicated().any():
        raise ValueError("join_arr: duplicate MVT_ID_mvt in dep")
    a = arr[arr.lobt.notna() & arr.aobt3.notna() & arr.CALLSIGN_flt.notna()]
    a = a.rename(columns={"ADEP_flt": "ADEP_mvt", "CALLSIGN_flt": "FLIGHT_mvt", "mvt": "arr_mvt"})
    d = dep[dep.FLIGHT_mvt.notna()][["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "mvt"]]
    m = d.merge(a[["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "arr_mvt", "lobt", "aobt3"]],
                on=["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt"], how="inner")
    dt = m.arr_mvt - m.mvt
    m = m[(dt > WIN_LO) & (dt <= WIN_HI)].sort_values(["MVT_ID_mvt", "arr_mvt"], kind="mergesort")
    m = m.drop_duplicates("MVT_ID_mvt", keep="first").set_index("MVT_ID_mvt")
    out = pd.DataFrame({"MVT_ID_mvt": dep.MVT_ID_mvt.to_numpy()})
    out["arr_lobt"] = m.lobt.reindex(out.MVT_ID_mvt).to_numpy(dtype="float64")
    out["arr_aobt3"] = m.aobt3.reindex(out.MVT_ID_mvt).to_numpy(dtype="float64")
    return out


def sub_rule(sp, sched, lobt, aobt3) -> np.ndarray:
    """'ord' | 'fill' | 'none' (joined, in scope, neither rule) | '' (not joined or out of scope). All inputs epoch s / s."""
    sp, sched, lobt, aobt3 = (np.asarray(x, dtype="float64") for x in (sp, sched, lobt, aobt3))
    joined = np.isfinite(lobt) & np.isfinite(aobt3)
    scope = joined & np.isfinite(sp) & (sp >= SP_LO) & (sp < SP_HI)
    with np.errstate(invalid="ignore"):
        ord_ = scope & (np.abs(lobt - sched) <= ORD_TOL)
        fill = scope & ~ord_ & (np.abs(aobt3 - lobt) <= FILL_TOL)
    return np.where(ord_, "ord", np.where(fill, "fill", np.where(scope, "none", "")))


def arm_pred(inc, rule, arm: str, mvt, aobt3, sp) -> np.ndarray:
    """Incumbent with the arm's sub-rule(s) applied. arm ∈ {C, D, CD}."""
    if arm not in ("C", "D", "CD"):
        raise ValueError(f"unknown arm {arm!r}")
    inc, mvt, aobt3, sp = (np.asarray(x, dtype="float64") for x in (inc, mvt, aobt3, sp))
    rule = np.asarray(rule)
    out = inc.copy()
    if "C" in arm:
        m = rule == "ord"
        out[m] = np.maximum(np.rint(mvt[m] - aobt3[m]), 1.0)
    if "D" in arm:
        m = rule == "fill"
        out[m] = np.rint(sp[m])
    return out


def arm_mask(rule, arm: str) -> np.ndarray:
    rule = np.asarray(rule)
    return {"C": rule == "ord", "D": rule == "fill", "CD": np.isin(rule, ["ord", "fill"])}[arm]


def label_class(y, sp) -> np.ndarray:
    """ARC.1: fill |y − sp| ≤ 60; slip y ≥ 82,800; ordinary y < 10,800; other."""
    y, sp = np.asarray(y, dtype="float64"), np.asarray(sp, dtype="float64")
    return np.where(np.abs(y - sp) <= 60, "fill", np.where(y >= 82_800, "slip", np.where(y < 10_800, "ordinary", "other")))


def evaluate(y, inc, pred, month, block, n26: int, rng_seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """Metrics on the touched rows passed in (all arrays already restricted to them)."""
    y, inc, pred = (np.asarray(x, dtype="float64") for x in (y, inc, pred))
    n = len(y)
    g = (y - inc) ** 2 - (y - pred) ** 2
    scale = (n26 / n) if n else 0.0
    move = np.abs(pred - inc)
    by_m = pd.Series(g).groupby(np.asarray(month)).sum()
    top5 = np.sort(g)[::-1][:5].sum() if n else 0.0
    ci = [0.0, 0.0]
    if n:
        blocks = pd.Series(g).groupby(np.asarray(block)).sum().to_numpy()
        rng = np.random.default_rng(rng_seed)
        draws = blocks[rng.integers(0, len(blocks), size=(n_boot, len(blocks)))].sum(axis=1) / BOARD_N * scale
        ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    return dict(n_2025=int(n), n_2026=int(n26), gain_sse=float(g.sum()), price=float(g.sum() / BOARD_N * scale),
                ci95=ci, months_with_rows=int(len(by_m)), months_positive=int((by_m > 0).sum()),
                gain_minus_top5_sse=float(g.sum() - top5),
                gain_moves_gt300_sse=float(g[move > 300].sum()), gain_moves_gt1800_sse=float(g[move > 1800].sum()),
                share_gain_moves_gt1800=float(g[move > 1800].sum() / g.sum()) if g.sum() > 0 else None,
                rows_losing=int((g < 0).sum()), worst_row_board=float(g.min() / BOARD_N) if n else 0.0)


def verdict(arm: str, ev: dict, precision_fill=None, n_fill_calls: int = 0) -> str:
    if ev["n_2025"] < MIN_ROWS:
        return "UNDERPOWERED"
    if arm == "C":
        ok = (ev["price"] >= 100 and ev["months_positive"] >= 11 and ev["ci95"][0] > 0 and ev["worst_row_board"] >= -60)
    elif arm == "D":
        ok = (ev["price"] >= 300 and precision_fill is not None and n_fill_calls >= 15 and precision_fill >= 0.85
              and ev["months_positive"] >= 0.75 * ev["months_with_rows"] and ev["gain_minus_top5_sse"] > 0)
    else:
        return "REPORTED"
    return "WORKING" if ok else "NOT WORKING"


def transfer_expectation(v16, new, sp, e_ord, rule, p_fill: float) -> dict:
    """Post-hoc, label-free 2026 diagnostic (RESULT ARC; not a clause). Expected board MSE of replacing the shipped v16 value on
    2026 rows. R_ord rows: y = new + e with e drawn from the 2025 R_ord residuals (all ordinary in 2025). R_fill rows: y = rint(sp)
    with probability p_fill, else y = new + e."""
    v16, new, sp, e = (np.asarray(x, dtype="float64") for x in (v16, new, sp, e_ord))
    rule = np.asarray(rule)
    if len(e) == 0:
        raise ValueError("transfer_expectation: no 2025 R_ord residuals")
    o, f = rule == "ord", rule == "fill"
    g_ord = np.array([np.mean((new[i] + e - v16[i]) ** 2 - e ** 2) for i in np.flatnonzero(o)])
    fill_g = (sp[f] - v16[f]) ** 2
    wrong_g = np.array([np.mean((new[i] + e - v16[i]) ** 2 - (new[i] + e - sp[i]) ** 2) for i in np.flatnonzero(f)])
    move_o = np.abs(new[o] - v16[o])
    exp_f = p_fill * fill_g + (1 - p_fill) * wrong_g
    return dict(C=float(g_ord.sum() / BOARD_N), C_rows=int(o.sum()), C_rows_moved_gt300=int((move_o > 300).sum()),
                C_share_gain_moves_gt1800=float(g_ord[move_o > 1800].sum() / g_ord.sum()) if g_ord.sum() > 0 else None,
                D=float(exp_f.sum() / BOARD_N), D_rows=int(f.sum()), p_fill=float(p_fill),
                D_rows_moved_gt1800=int((np.abs(np.rint(sp[f]) - v16[f]) > 1800).sum()),
                D_worst_wrong_call=float(wrong_g.min() / BOARD_N) if f.any() else 0.0,
                D_best_right_call=float(fill_g.max() / BOARD_N) if f.any() else 0.0)


# --------------------------------------------------------------------------- loaders (run once)

def load_raw(cols, files) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def arr_frame(raw: pd.DataFrame) -> pd.DataFrame:
    a = raw[raw.PHASE_mvt == "ARR"]
    return pd.DataFrame({"ADEP_flt": a.ADEP_flt.to_numpy(), "ADES_mvt": a.ADES_mvt.to_numpy(),
                         "CALLSIGN_flt": a.CALLSIGN_flt.to_numpy(), "mvt": secs(a.MVT_TIME_UTC_mvt),
                         "lobt": secs(a.LOBT_flt), "aobt3": secs(a.AOBT_3_flt)})


def dep_frame(raw: pd.DataFrame, ids) -> pd.DataFrame:
    d = raw[raw.MVT_ID_mvt.isin(ids)].drop_duplicates("MVT_ID_mvt")
    d = d.set_index("MVT_ID_mvt").reindex(ids).reset_index()
    if d.ADEP_mvt.isna().any():
        raise AssertionError(f"{int(d.ADEP_mvt.isna().sum())} incumbent rows missing from the raw files")
    return pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt.to_numpy(), "ADEP_mvt": d.ADEP_mvt.to_numpy(),
                         "ADES_mvt": d.ADES_mvt.to_numpy(), "FLIGHT_mvt": d.FLIGHT_mvt.to_numpy(),
                         "mvt": secs(d.MVT_TIME_UTC_mvt), "sched": secs(d.SCHED_TIME_UTC_mvt),
                         "date": pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").to_numpy()})


def incumbents_2025(train_files) -> pd.DataFrame:
    """MVT_ID_mvt, scope (LIRF|OTHER), ap, month, y, sp, inc — v16-equivalent LOMO predictions, guarded (G1)."""
    import congestion_anchor_ship as CA
    import lirf_chain_map as LCMM
    import rome_local_day as RLD
    import rome_localday as LD
    # LIRF chain (v16 = v14 = v13 chain)
    m = LCMM.load_2025()
    e1 = LD.e1_floor(m.R2.to_numpy(float), m.sp.to_numpy(float))
    fr = pd.DataFrame({"MVT_ID_mvt": m.MVT_ID_mvt, "ADEP_mvt": "LIRF", "unmatched": True, "base": e1,
                       "SCHED_TIME_UTC_mvt": m.SCHED_TIME_UTC_mvt, "MVT_TIME_UTC_mvt": m.MVT_TIME_UTC_mvt})
    chain, _ = RLD.apply_rld(fr)
    lcm = json.loads(LCM.read_text())
    sse = float(((m.y.to_numpy(float) - chain.astype(float)) ** 2).sum())
    if abs(sse - lcm["sse_total"]) > G1_TOL * lcm["sse_total"]:
        raise AssertionError(f"G1 LIRF: chain SSE {sse:.0f} vs recorded {lcm['sse_total']:.0f}")
    lirf = pd.DataFrame({"MVT_ID_mvt": m.MVT_ID_mvt.to_numpy(float), "scope": "LIRF", "ap": "LIRF",
                         "month": m.month.to_numpy(int), "y": m.y.to_numpy(float), "sp": m.sp.to_numpy(float),
                         "inc": chain.astype(float)})
    # non-LIRF: S1C + max(A − 300, 0), A from all twelve training months
    u = pd.read_parquet(UNM, columns=["MVT_ID_mvt", "month", "date", "ADEP_mvt", "y", "sp", "S1C"])
    u = u[u.ADEP_mvt != "LIRF"].reset_index(drop=True)
    A = CA.day_anomaly(load_raw(["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"], train_files))
    u = u.merge(A, on=["ADEP_mvt", "date"], how="left", validate="many_to_one")
    u["inc"] = np.rint(u.S1C.to_numpy(float) + CA.shift(u.A.to_numpy()))
    pm = json.loads(POOL_MAP.read_text())
    want, got = 0.0, 0.0
    for p in pm["pools"]:
        if p["lane"] == "unmatched" and p["n_2026"] > 0:
            want += p["price"] * BOARD_N * p["n_2025"] / p["n_2026"]
            g = u[(u.ADEP_mvt == p["ap"]) & (u.month == p["month"])]
            got += float(((g.y - g.inc) ** 2).sum())
    if abs(got - want) > G1_TOL * want:
        raise AssertionError(f"G1 OTHER: v16 Jan+Jul SSE {got:.0f} vs pool map {want:.0f}")
    other = pd.DataFrame({"MVT_ID_mvt": u.MVT_ID_mvt.to_numpy(float), "scope": "OTHER", "ap": u.ADEP_mvt.astype(str),
                          "month": u.month.to_numpy(int), "y": u.y.to_numpy(float), "sp": u.sp.to_numpy(float),
                          "inc": u.inc.to_numpy(float)})
    return pd.concat([lirf, other], ignore_index=True), dict(lirf_sse=sse, other_janjul_sse=got, other_janjul_want=want)


def main_transfer() -> int:
    """RESULT ARC's post-hoc 2026 diagnostic, made reproducible (writes TRANSFER; LIRF scope only)."""
    if TRANSFER.exists():
        raise FileExistsError(TRANSFER)
    train = sorted(RAW.glob("training_2025-*.parquet"))
    inc, _ = incumbents_2025(train)
    raw = load_raw(sorted(set(DEP_COLS + ARR_COLS) - {"PHASE_mvt"}) + ["PHASE_mvt"], train)
    dep = dep_frame(raw, inc.MVT_ID_mvt.to_numpy(dtype="float64"))
    j = join_arr(dep, arr_frame(raw))
    rule = sub_rule(inc.sp, dep.sched, j.arr_lobt, j.arr_aobt3)
    lirf = (inc.scope == "LIRF").to_numpy()
    new25 = np.maximum(np.rint((dep.mvt - j.arr_aobt3).to_numpy()), 1.0)
    e_ord = (inc.y.to_numpy() - new25)[lirf & (rule == "ord")]
    cls = label_class(inc.y, inc.sp)
    fm = lirf & (rule == "fill")
    k, n = int((cls[fm] == "fill").sum()), int(fm.sum())
    r = pd.read_parquet(RANK, columns=sorted(set(DEP_COLS + ARR_COLS)))
    d26 = r[(r.PHASE_mvt == "DEP") & r.AOBT_3_flt.isna()]
    dep26 = dep_frame(d26, d26.MVT_ID_mvt.to_numpy())
    j26 = join_arr(dep26, arr_frame(r))
    sp26 = (dep26.mvt - dep26.sched).to_numpy()
    rule26 = sub_rule(sp26, dep26.sched, j26.arr_lobt, j26.arr_aobt3)
    rule26 = np.where(dep26.ADEP_mvt.to_numpy() == "LIRF", rule26, "")
    v16 = pd.read_parquet(V16).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(dep26.MVT_ID_mvt).to_numpy(float)
    new26 = np.maximum(np.rint((dep26.mvt - j26.arr_aobt3).to_numpy()), 1.0)
    ps = {"observed": k / n, "beta_mean": (k + 1) / (n + 2), "low": 0.76}
    out = dict(tag="ARC-transfer", NOT_A_CLAUSE=True, fill_hits_2025=[k, n], n_e_ord=int(len(e_ord)),
               by_p={name: transfer_expectation(v16, new26, sp26, e_ord, rule26, p) for name, p in ps.items()},
               rows_2026=[dict(MVT_ID_mvt=float(m), rule=str(ru), sp=float(a), v16=float(b), new=float(c))
                          for m, ru, a, b, c in zip(dep26.MVT_ID_mvt, rule26, sp26, v16, new26) if ru in ("ord", "fill")])
    TRANSFER.write_text(json.dumps(out, indent=1))
    for name, t in out["by_p"].items():
        print(name, {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in t.items()})
    return 0


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    train = sorted(RAW.glob("training_2025-*.parquet"))
    if len(train) != 12:
        raise AssertionError(f"expected 12 training files, got {len(train)}")
    inc, g1 = incumbents_2025(train)
    raw = load_raw(sorted(set(DEP_COLS + ARR_COLS) - {"PHASE_mvt"}) + ["PHASE_mvt"], train)
    dep = dep_frame(raw, inc.MVT_ID_mvt.to_numpy(dtype="float64"))
    if not np.allclose(dep.mvt - dep.sched, inc.sp, atol=1e-6):
        raise AssertionError("stored sp differs from raw MVT − SCHED")
    arr = arr_frame(raw)
    j1, j2 = join_arr(dep, arr), join_arr(dep, arr)
    if not j1.equals(j2):
        raise AssertionError("G2: join is not deterministic")
    rule = sub_rule(inc.sp, dep.sched, j1.arr_lobt, j1.arr_aobt3)
    cls = label_class(inc.y, inc.sp)
    # 2026 (label-free)
    r = pd.read_parquet(RANK, columns=sorted(set(DEP_COLS + ARR_COLS)))
    d26 = r[(r.PHASE_mvt == "DEP") & r.AOBT_3_flt.isna()]
    dep26 = dep_frame(d26, d26.MVT_ID_mvt.to_numpy())
    j26 = join_arr(dep26, arr_frame(r))
    sp26 = dep26.mvt - dep26.sched
    rule26 = sub_rule(sp26, dep26.sched, j26.arr_lobt, j26.arr_aobt3)
    v16 = pd.read_parquet(V16).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(dep26.MVT_ID_mvt).to_numpy(float)
    scope26 = np.where(dep26.ADEP_mvt.to_numpy() == "LIRF", "LIRF", "OTHER")
    res = dict(tag="ARC", prereg=str(PREREG.relative_to(ROOT)), prereg_sha256=hashlib.sha256(PREREG.read_bytes()).hexdigest(),
               NO_UPLOAD=True, g1=g1, scopes={})
    for scope in ("LIRF", "OTHER"):
        s = (inc.scope == scope).to_numpy()
        s26 = scope26 == scope
        fillm = s & (rule == "fill")
        prec = float((cls[fillm] == "fill").mean()) if fillm.any() else None
        f26 = s26 & (rule26 == "fill")
        fp_cost = float(np.median((np.rint(sp26[f26]) - v16[f26]) ** 2) / BOARD_N) if f26.any() else None
        out = dict(rules_2025={k: int(v) for k, v in pd.Series(rule[s]).value_counts().items()},
                   rules_2026={k: int(v) for k, v in pd.Series(rule26[s26]).value_counts().items()},
                   confusion_2025=pd.crosstab(rule[s], cls[s]).to_dict(), fill_precision_2025=prec,
                   n_fill_calls_2025=int(fillm.sum()), fp_cost_one_false_fill_2026=fp_cost, arms={})
        for arm in ("C", "D", "CD"):
            t = s & arm_mask(rule, arm)
            n26 = int((s26 & arm_mask(rule26, arm)).sum())
            pred = arm_pred(inc.inc, rule, arm, dep.mvt, j1.arr_aobt3, inc.sp)
            blk = (inc.ap + "|" + dep.date).to_numpy()
            ev = evaluate(inc.y[t], inc.inc[t], pred[t], inc.month[t], blk[t], n26)
            ev["by_airport_gain_sse"] = {k: float(v) for k, v in pd.Series(
                ((inc.y - inc.inc) ** 2 - (inc.y - pred) ** 2)[t].to_numpy()).groupby(inc.ap[t].to_numpy()).sum().items()}
            ev["verdict"] = verdict(arm, ev, prec, int(fillm.sum()))
            out["arms"][arm] = ev
        out["class_scaled_price_CD"] = out["arms"]["C"]["price"] + out["arms"]["D"]["price"]
        res["scopes"][scope] = out
    REPORT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    for scope, o in res["scopes"].items():
        print(scope, "rules25", o["rules_2025"], "rules26", o["rules_2026"], "prec", o["fill_precision_2025"],
              "fp_cost", o["fp_cost_one_false_fill_2026"])
        for arm, ev in o["arms"].items():
            print(f"  {arm}: n {ev['n_2025']}/{ev['n_2026']} price {ev['price']:+.1f} CI {ev['ci95'][0]:+.1f}..{ev['ci95'][1]:+.1f} "
                  f"months {ev['months_positive']}/{ev['months_with_rows']} top5-out {ev['gain_minus_top5_sse']/BOARD_N:+.1f} "
                  f">1800 share {ev['share_gain_moves_gt1800']} worst {ev['worst_row_board']:+.1f} -> {ev['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_transfer() if "--transfer" in sys.argv else main())
