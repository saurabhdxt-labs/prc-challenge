"""Arm ADN ship step on v10 (plans/PREREG_adsb_nn_2026_09_10.md, "Only if WORKING").

The matched rows of v10 are arm F's 12-month refit (v6's lane). This step adds the ADS-B stacker's residual on the
scored matched departures that have a joined ADS-B flight, at airports that pass the registered coverage gate, and
leaves every other row byte-identical. Upload only with the owner's confirmation.

Pure pieces here are unit-tested (tests/test_adsb_ship.py); the fitting reuses scripts/adsb_stack.py.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUBS = ROOT / "submissions"
BASE_NAME = "merry-quicksand_v10.parquet"
GATE_RATIO = 0.7


def coverage_gate(cov_2025: dict, cov_2026: dict) -> dict:
    """{airport: ships?} — ships iff its 2025 joined share is > 0 and its 2026 share >= GATE_RATIO x the 2025 share."""
    return {ap: bool(c25 > 0 and cov_2026.get(ap, 0.0) >= GATE_RATIO * c25) for ap, c25 in cov_2025.items()}


def delta_from_taxi(proxy, taxi) -> np.ndarray:
    """The shipped matched lane's delta prediction, recovered from its taxi time: proxy − taxi."""
    return np.asarray(proxy, float) - np.asarray(taxi, float)


def assemble(v10: pd.DataFrame, rows: pd.DataFrame, gate: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """v11 = v10 with taxi = rint(max(proxy − (F + resid), 1)) on rows that are joined (coverage >= 2) at a gated
    airport; everything else untouched. `rows`: MVT_ID_mvt, ap, proxy, F, resid, coverage. Returns (v11, detail of
    the rows whose value changed)."""
    out = v10.copy()
    r = rows.copy()
    r["ships"] = r.ap.map(lambda a: gate.get(a, False)).astype(bool) & (r.coverage.to_numpy() >= 2)
    r = r[r.ships]
    new = np.rint(np.maximum(r.proxy.to_numpy(float) - (r.F.to_numpy(float) + r.resid.to_numpy(float)), 1.0))
    pos = pd.Series(np.arange(len(out)), index=out.MVT_ID_mvt.to_numpy())
    idx = pos.loc[r.MVT_ID_mvt.to_numpy()].to_numpy()
    before = out.TAXITIME_SEC_mvt.to_numpy()[idx]
    vals = out.TAXITIME_SEC_mvt.to_numpy().copy()
    vals[idx] = new.astype(np.int32)
    out["TAXITIME_SEC_mvt"] = vals.astype(np.int32)
    detail = r.assign(before=before, after=new.astype(np.int32))
    detail = detail[detail.before != detail.after][["MVT_ID_mvt", "ap", "proxy", "F", "resid", "coverage", "before",
                                                    "after"]]
    return out, detail.reset_index(drop=True)


def check_paths(base: pathlib.Path, dest: pathlib.Path) -> None:
    if pathlib.Path(base).name != BASE_NAME:
        raise SystemExit(f"base must be {BASE_NAME} (got {pathlib.Path(base).name})")
    if pathlib.Path(dest).exists():
        raise SystemExit(f"refusing to overwrite {dest}")


def contract(sub: pd.DataFrame, base: pd.DataFrame) -> None:
    """The submission contract against its base: same rows in the same order, unique ids, finite ints >= 1."""
    if len(sub) != len(base) or not (sub.MVT_ID_mvt.to_numpy() == base.MVT_ID_mvt.to_numpy()).all():
        raise ValueError("rows differ from the base (count, ids or order)")
    if not sub.MVT_ID_mvt.is_unique:
        raise ValueError("duplicate MVT_ID_mvt")
    v = sub.TAXITIME_SEC_mvt.to_numpy()
    if not np.issubdtype(v.dtype, np.integer):
        raise ValueError(f"values must be integer seconds, got dtype {v.dtype}")
    if not np.isfinite(v.astype(float)).all() or (v < 1).any():
        raise ValueError("every value must be a finite integer >= 1")


NEWLY_LOW = 0.20            # ADN.7: a 2025 joined share below this ...
NEWLY_GROWTH = 2.0          # ... or 2026 >= this x 2025 marks an airport as newly covered


def gate_table(cov_2025: dict, cov_2026: dict, transfer_pass: bool, holds: dict | None = None,
               coverage_overrides: dict | None = None) -> dict:
    """{airport: {allowed, rule, c25, c26}}: newly covered airports (ADN.7) ship iff the transfer test passed; every
    other airport follows the registered 0.7x coverage gate and, under SHIP-ADN, keeps F when `holds` lists a
    within-code key-feature KS flag for it (plans/PREREG_adsb_ship_v11_2026_09_11.md). ADN.7 governs newly covered
    airports, so holds do not apply to them."""
    holds = holds or {}
    overrides = coverage_overrides or {}                # SHIP-ADN.5: {airport: reason}; lifts a failed 0.7x rule only
    out = {}
    for ap in sorted(set(cov_2025) | set(cov_2026)):
        c25, c26 = float(cov_2025.get(ap, 0.0)), float(cov_2026.get(ap, 0.0))
        newly = c25 < NEWLY_LOW or c26 >= NEWLY_GROWTH * c25
        if newly:
            allowed = bool(transfer_pass and c26 > 0)
            rule = f"ADN.7 newly covered (transfer {'PASS' if transfer_pass else 'FAIL'})"
        else:
            allowed = bool(c26 >= GATE_RATIO * c25)
            rule = f"0.7x gate ({c26:.3f} vs {GATE_RATIO * c25:.3f})"
            if not allowed and ap in overrides:            # a hold below still wins
                allowed = True
                rule += f" fails; coverage override: {overrides[ap]} (owner, SHIP-ADN.5)"
            if allowed and holds.get(ap):
                allowed = False
                rule += f"; KS hold ({', '.join(holds[ap])})"
        out[ap] = {"allowed": allowed, "rule": rule, "c25": c25, "c26": c26}
    return out


KS_FLAG = 0.10


def joined_share(table: pd.DataFrame) -> dict:
    """{airport: share of rows with coverage >= 2}."""
    return {ap: float((g.coverage >= 2).mean()) for ap, g in table.groupby("ap")}


KS_MIN_N = 30
#: SHIP-ADN key features: the ADS-B inputs with >= 5% of the summed gain importance of the 15 saved G boosters
#: (computed 2026-09-11 13:30 from the boosters alone, before any 2026 table existed).
KEY_FEATS = ("first_ground_rel_aobt3", "first_move_rel_mvt", "first_ground_gs")
STRATA = (2, 3)


def _ks(a, b):
    """Two-sample KS statistic on the finite values; None when either side has fewer than KS_MIN_N."""
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if len(a) < KS_MIN_N or len(b) < KS_MIN_N:
        return None
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a) - np.searchsorted(b, grid, "right") / len(b))))


def _ks_rows(a: pd.DataFrame, b: pd.DataFrame, feats: list) -> dict:
    out = {}
    for f in feats:
        k = _ks(a[f].to_numpy(float), b[f].to_numpy(float))
        out[f] = {"ks": k, "flag": bool(k is not None and k > KS_FLAG)}
    return out


def ks_contract(t25: pd.DataFrame, t26: pd.DataFrame, feats: list) -> dict:
    """Two-sample KS statistic per airport per feature on JOINED rows (coverage >= 2), pooled; flag above KS_FLAG."""
    out = {}
    for ap in sorted(set(t25.ap) & set(t26.ap)):
        out[ap] = _ks_rows(t25[(t25.ap == ap) & (t25.coverage >= 2)], t26[(t26.ap == ap) & (t26.coverage >= 2)], feats)
    return out


def ks_by_code(t25: pd.DataFrame, t26: pd.DataFrame, feats: list) -> dict:
    """{airport: {code: {feature: {ks, flag}}}} inside each joined coverage-code stratum (2 mid-taxi, 3 from the stand):
    a change in the code mix moves the pooled distribution without moving any code's own."""
    out = {}
    for ap in sorted(set(t25.ap) & set(t26.ap)):
        out[ap] = {c: _ks_rows(t25[(t25.ap == ap) & (t25.coverage == c)], t26[(t26.ap == ap) & (t26.coverage == c)], feats)
                   for c in STRATA}
    return out


def ks_holds(by_code: dict, key_feats=KEY_FEATS) -> dict:
    """{airport: ["code<c>:<feature>", ...]} for every flagged KEY feature; airports without one are absent."""
    out = {}
    for ap, strata in by_code.items():
        hits = [f"code{c}:{f}" for c, d in strata.items() for f, r in d.items() if f in key_feats and r["flag"]]
        if hits:
            out[ap] = hits
    return out


def season_holds(ks26: dict, season: dict, key_feats=KEY_FEATS) -> dict:
    """SHIP-ADN.3: {airport: ["code<c>:<feature> (KS26 > KS_season)", ...]}. Holds a cell iff both statistics are defined
    and KS26 (2026 vs 2025, both Jan+Jul) > KS_season (2025 Jan vs 2025 Jul), the within-2025 shift the stacker's day
    folds absorbed. Airports without a held cell are absent."""
    out = {}
    for ap, strata in ks26.items():
        if ap not in season:
            raise KeyError(f"no seasonal baseline for {ap}")
        hits = []
        for c, d in strata.items():
            for f, r in d.items():
                if c not in season[ap]:
                    raise KeyError(f"{ap}: no seasonal baseline for stratum {c}")
                b = season[ap][c].get(f, {}).get("ks")
                if f in key_feats and r["ks"] is not None and b is not None and r["ks"] > b:
                    hits.append(f"code{c}:{f} ({r['ks']:.3f} > {b:.3f})")
        if hits:
            out[ap] = hits
    return out


RWC_DISPUTED = ("EDDM", "EDDF", "LEBL", "LSZH")


def rwc_holds(rwc: dict) -> dict:
    """SHIP-ADN.4 ("D, per RWC", plans/PREREG_reweight_check_2026_09_11.md): hold a disputed airport iff its RWC decision is
    not SHIP; a harness that did not pass holds all of D. Airports outside D are never held from this file."""
    if rwc.get("harness") != "PASS":
        return {a: [f"RWC harness {rwc.get('harness')}"] for a in RWC_DISPUTED}
    out = {}
    for a in RWC_DISPUTED:
        if a not in rwc["results"]:
            raise KeyError(f"RWC has no result for disputed airport {a}")
        d = rwc["results"][a]["decision"]
        if d != "SHIP":
            out[a] = [f"RWC {d}"]
    return out


def rwc3_overrides(rwc_lirf: dict) -> dict:
    """SHIP-ADN.5: {"LIRF": "RWC.3 SHIP"} iff RWC.3's harness passed and LIRF's decision is SHIP; otherwise {}."""
    if rwc_lirf.get("harness") != "PASS":
        return {}
    if "LIRF" not in rwc_lirf["results"]:
        raise KeyError("RWC.3 has no result for LIRF")
    return {"LIRF": "RWC.3 SHIP"} if rwc_lirf["results"]["LIRF"]["decision"] == "SHIP" else {}


N_BOARD = 344_841
#: 2025 fold-A SSE gain of B vs F per joined row (RESULT ADN predictions, computed 13:30; SHIP-ADN projection)
G_PER_JOINED = {"EDDF": 14_091, "EDDM": 18_674, "EGLL": 14_561, "EHAM": 13_621, "LEBL": 11_304, "LIRF": 12_326,
                "LSZH": 14_310}
G_POOLED = 13_906
K_NEW = 0.595               # mean of the four ADN.7 transferred fractions


def projection(gate: dict, n26: dict, m26: dict) -> dict:
    """SHIP-ADN projected board MSE gain: sum over allowed airports of k * g * n26 * m26 / N_BOARD (newly covered:
    pooled g, k = K_NEW)."""
    by = {}
    for ap, v in gate.items():
        if not v["allowed"]:
            continue
        if v["rule"].startswith("ADN.7"):
            g, k = G_POOLED, K_NEW
        elif ap in G_PER_JOINED:
            g, k = G_PER_JOINED[ap], 1.0
        else:
            raise ValueError(f"{ap}: allowed within 2025 support but has no 2025 per-joined-row gain")
        by[ap] = k * g * n26[ap] * m26[ap] / N_BOARD
    return {"P": float(sum(by.values())), "by_airport": by}


def write_gate(model_dir, t25: pd.DataFrame, t26: pd.DataFrame, transfer: dict, holds: dict | None = None,
               ks_rule: str = "SHIP-ADN: within-code KS > KS_FLAG", coverage_overrides: dict | None = None) -> dict:
    """Compute the gate table from the two feature tables and the recorded ADN.7 transfer verdict; write it (and its
    evidence) into model_dir/MANIFEST.json. Refuses to overwrite a gate that is already set."""
    import json
    model_dir = pathlib.Path(model_dir)
    man = json.loads((model_dir / "MANIFEST.json").read_text())
    if man.get("gate"):
        raise SystemExit("gate already set in MANIFEST.json; refusing to overwrite")
    c25, c26 = joined_share(t25), joined_share(t26)
    tp = bool(transfer["rule"]["PASS"])
    man["gate"] = gate_table(c25, c26, transfer_pass=tp, holds=holds, coverage_overrides=coverage_overrides)
    man["gate_evidence"] = {"transfer_PASS": tp, "transfer_n_pass": transfer["rule"].get("n_pass"),
                            "gate_ratio": GATE_RATIO, "newly_low": NEWLY_LOW, "newly_growth": NEWLY_GROWTH,
                            "joined_share_2025": c25, "joined_share_2026": c26, "ks_holds": holds or {},
                            "key_feats": list(KEY_FEATS), "ks_flag": KS_FLAG, "ks_rule": ks_rule,
                            "coverage_overrides": coverage_overrides or {}}
    (model_dir / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man["gate"]


SPENT_DAYS = ("2025-01-09",)


def tables_with_airport(t25_path, record_path, raw_months, t26_path, ranking_path):
    """(2025 table with `ap`, spent day removed; 2026 table with `ap`). 2025's airport comes from the fold record
    (row-aligned), 2026's from the ranking file by MVT_ID."""
    t25 = pd.read_parquet(t25_path)
    rec = pd.read_parquet(record_path, columns=["row", "ap"])
    if not (rec.row.to_numpy() == t25.row.to_numpy()).all():
        raise ValueError("2025 table rows are not the record's rows")
    t25 = t25.assign(ap=rec.ap.to_numpy())
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]) for p in raw_months])
    when = pd.to_datetime(t25.MVT_ID_mvt.map(raw.drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
                                             .MVT_TIME_UTC_mvt), utc=True)
    t25 = t25.assign(month=when.dt.month.to_numpy())
    t25 = t25[~when.dt.strftime("%Y-%m-%d").isin(SPENT_DAYS).to_numpy()].reset_index(drop=True)
    t26 = pd.read_parquet(t26_path)
    rk = pd.read_parquet(ranking_path, columns=["MVT_ID_mvt", "ADEP_mvt"]).drop_duplicates("MVT_ID_mvt")
    ap = t26.MVT_ID_mvt.map(rk.set_index("MVT_ID_mvt").ADEP_mvt)
    if ap.isna().any():
        raise ValueError(f"{int(ap.isna().sum())} 2026 rows without an airport")
    return t25, t26.assign(ap=ap.to_numpy())


def main(argv=None) -> int:
    """gate: compute the gate table, the KS contract (pooled, reported; within-code key-feature holds, acted on) and the
    SHIP-ADN projection from the two feature tables; write the gate into the models' MANIFEST.json and the evidence to
    reports/adsb_gate.json."""
    import argparse
    import json
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("cmd", choices=["gate"])
    ap.add_argument("--t25", default=None, help="2025 feature table (default: the v2 table)")
    ap.add_argument("--t26", default=None, help="2026 feature table (default: the v2 table)")
    ap.add_argument("--models", default=None, help="artefact directory whose MANIFEST.json receives the gate")
    ap.add_argument("--report", default=None, help="gate evidence JSON (default: reports/adsb_gate.json)")
    ap.add_argument("--rule", required=True, choices=["season", "rwc", "rwc3"],
                    help="season: SHIP-ADN.3 (KS26 > KS_season); rwc: SHIP-ADN.4 (D, per reports/rwc.json); "
                         "rwc3: SHIP-ADN.5 (rwc + the RWC.3 LIRF coverage override, reports/rwc_lirf.json)")
    a = ap.parse_args(argv)
    v2 = ROOT / "data" / "adsb" / "v2"
    models_dir = pathlib.Path(a.models) if a.models else v2 / "models"
    t25, t26 = tables_with_airport(a.t25 or v2 / "features_2025janjul.parquet",
                                  ROOT / "data" / "cache_stand" / "fold_preds_queue_order.parquet",
                                  [ROOT / "data" / "raw" / "training_2025-01-01_2025-02-01.parquet",
                                   ROOT / "data" / "raw" / "training_2025-07-01_2025-08-01.parquet"],
                                  a.t26 or v2 / "features_2026janjul.parquet", ROOT / "data" / "raw" / "ranking.parquet")
    report_path = pathlib.Path(a.report) if a.report else ROOT / "reports" / "adsb_gate.json"
    if report_path.exists():
        raise SystemExit(f"{report_path} exists; keep it as evidence under another name first")
    transfer = json.loads((ROOT / "reports" / "adsb_transfer.json").read_text())
    feats = json.loads((models_dir / "MANIFEST.json").read_text())["feats"]
    ks = ks_contract(t25, t26, feats)
    by_code = ks_by_code(t25, t26, feats)
    old_holds = ks_holds(by_code)                                   # the SHIP-ADN rule, replaced; reported only
    season = ks_by_code(t25[t25.month == 1], t25[t25.month == 7], feats)
    overrides = {}
    if a.rule == "season":
        holds, rule = season_holds(by_code, season), "SHIP-ADN.3: KS26 > KS_season"
    else:
        holds, rule = rwc_holds(json.loads((ROOT / "reports" / "rwc.json").read_text())), "SHIP-ADN.4: D, per RWC"
        if a.rule == "rwc3":
            overrides = rwc3_overrides(json.loads((ROOT / "reports" / "rwc_lirf.json").read_text()))
            rule = "SHIP-ADN.5: D, per RWC + LIRF per RWC.3 (owner, with v12)"
    gate = write_gate(models_dir, t25, t26, transfer, holds=holds, ks_rule=rule, coverage_overrides=overrides)
    n26 = {a: int(n) for a, n in t26.groupby("ap").size().items()}
    proj = projection(gate, n26, joined_share(t26))
    report_path.write_text(json.dumps(
        {"prereg": "plans/PREREG_adsb_ship_v11_2026_09_11.md", "ks_rule": rule, "gate": gate, "holds": holds,
         "ks_holds_season_rule": season_holds(by_code, season), "coverage_overrides": overrides,
         "ks_holds_old_rule": old_holds, "projection": proj,
         "ks_season": {a: {str(c): d for c, d in v.items()} for a, v in season.items()},
         "n26": n26, "ks_contract_pooled": ks, "ks_by_code": {a: {str(c): d for c, d in v.items()} for a, v in by_code.items()}},
        indent=1))
    for k, v in gate.items():
        print(f"{k}: {'SHIP' if v['allowed'] else 'keep F'}  2025 {v['c25']:.3f}  2026 {v['c26']:.3f}  ({v['rule']})")
    flagged = {ap_: [f for f, r in d.items() if r["flag"]] for ap_, d in ks.items()}
    print("KS > 0.10 pooled (reported):", {k: v for k, v in flagged.items() if v})
    print("KS holds, old rule (replaced, reported):", old_holds)
    print(f"holds under {rule}:", holds)
    print(f"projection P = {proj['P']:.1f} board MSE  " + str({a: round(x, 1) for a, x in proj["by_airport"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
