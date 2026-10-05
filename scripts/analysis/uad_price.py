"""UAD — the aircraft's own ADS-B ground track as the off-block estimate for departures without a flight-plan
record, blended with the served prediction.

Registration: plans/PREREG_uad_unmatched_adsb_offblock_2026_10_04.md (REG 17a570a6). Fits the registered per-airport
offset and the calm/storm blend weights on the 2025 design months only, applies the registered blend to the
unmatched-body and storm population, and computes the two registered verdicts (H-UAD-1 on January + July against
the real promotion gate, H-UAD-2 on the design months' storm days by leave-one-event-out refit).

    python scripts/analysis/uad_price.py constants --lomo-dir D --table T --raw-dir R --out constants.json
    python scripts/analysis/uad_price.py verdict --lomo-dir D --table T --raw-dir R --constants constants.json \\
        --out report.json

The `constants` command drops every January and July row before fitting anything: its output holds no information
from the scored months. Reads 2025 data; no board read, no upload.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prc.pipeline.fm1 import protocols as P  # noqa: E402

import slc_slices as SLC  # noqa: E402  (the raw join and the scoring masks)

DESIGN_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
SCORED_MONTHS = (1, 7)
SCOPE = ("unmatched_body", "storm")
OFFSET_MAX_ABS_S = 900.0
OFFSET_MIN_ROWS = 200
STORM_A_S = 300.0
STORM_MIN_ROWS = 30
W1_FLOOR = 0.8
A2_WEIGHT_FACTOR = 0.5
H1_MIN_RMSE_FALL = 0.10
H2_MAX_RATIO = 0.6
H2_MIN_ROWS = 8
N_BOOT = 2000
SEED = 0
REGISTRATION = "17a570a6"
SCORE_COLS = ["MVT_ID_mvt", "y", "pred", "day", "month", "ap"]
LOMO_COLS = ["MVT_ID_mvt", "lane", "decided_by", "month", "day", "ap", "y", "pred", "unm_p_fill", "storm_A"]
ALL_MONTHS = tuple(range(1, 13))

# Stop rules for the table (prereg "Stop rules for the builder", as enforced by AMEND-A1 item 1 and read by AMEND-A2
# and AMEND-A3). TABLE_PARAMS_HASH is the builder's parameter hash after AMEND-A3.
TABLE_PARAMS_HASH = "d196b652a9a7"
STOP_AIRPORT = "EHAM"
STOP_MIN_CALLSIGN_TYPE_EQUAL = 0.97       # (a) exact type equality on callsign joins of matched rows
STOP_MIN_STAND_TYPE_AGREE = 0.95          # (b) type agreement (equal or an alias) on stand joins of unmatched rows
STOP_MAX_EXIT_MAD_S = 200.0               # (c) MAD of est - (MVT - AOBT_3) on matched rows with a class
STOP_MIN_STAND_CHECK_RIGHT = 0.95         # (d) the stand rule against the callsign join: right on at least this share
STOP_MAX_STAND_CHECK_WRONG = 0.005        #     and wrong on at most this share
STOP_CLASSED_RANGE = (508, 1000)          # sanity bracket: the stop airport's unmatched rows with a class (2025 table)
STOP_BARS = (("callsign_matched_type_equal_share", STOP_MIN_CALLSIGN_TYPE_EQUAL, "below"),
             ("stand_unmatched_type_agree_share", STOP_MIN_STAND_TYPE_AGREE, "below"),
             ("matched_exit_mad_s", STOP_MAX_EXIT_MAD_S, "above"),
             ("stand_check_right_share", STOP_MIN_STAND_CHECK_RIGHT, "below"),
             ("stand_check_wrong_share", STOP_MAX_STAND_CHECK_WRONG, "above"))


# ---------------------------------------------------------------------------------------------- stop rules

def check_stop_rules(report: dict) -> dict:
    """The builder's report.json against the registered stop rules: the params hash must be TABLE_PARAMS_HASH;
    report["stop_rules"][STOP_AIRPORT] must hold the five statistics of STOP_BARS, each a finite number on the right
    side of its bar (a "below" bar fails when the value is under it, an "above" bar when it is over it; the bar
    itself passes); and the stop airport's unmatched rows with a class (unmatched_A1 + unmatched_A2 summed over its
    months in report["by_airport_month"]) must number within STOP_CLASSED_RANGE, both ends included. Rule (a) reads
    the exact-equality share, never the alias-aware callsign_matched_type_agree_share; rule (b) reads the stand
    joins, never time_unmatched_type_agree_share (the report carries both). A missing key, None or NaN is a failure
    naming that statistic. Never raises on a malformed report: anything that is not the expected dict holds no
    statistic. Returns {"ok", "failed": [reasons], "values": {the statistics as found, "unmatched_classed",
    "params_hash"}}."""
    report = report if isinstance(report, dict) else {}
    stop_rules = report.get("stop_rules")
    stop_rules = stop_rules if isinstance(stop_rules, dict) else {}
    stats = stop_rules.get(STOP_AIRPORT)
    stats = stats if isinstance(stats, dict) else {}

    def finite(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
        return x if np.isfinite(x) else None

    failed = []
    values = {}
    for key, bar, word in STOP_BARS:
        raw = stats.get(key)
        values[key] = raw
        v = finite(raw)
        if v is None:
            failed.append(f"{key} missing or not finite")
        elif (v < bar) if word == "below" else (v > bar):
            failed.append(f"{key} {word} {bar}")

    by_month = report.get("by_airport_month")
    by_month = by_month if isinstance(by_month, dict) else {}
    cells = [v for k, v in by_month.items() if isinstance(k, str) and k.split("|")[0] == STOP_AIRPORT]
    try:
        classed = sum(int(c["unmatched_A1"]) + int(c["unmatched_A2"]) for c in cells) if cells else None
    except (KeyError, TypeError, ValueError):
        classed = None
    values["unmatched_classed"] = classed
    lo, hi = STOP_CLASSED_RANGE
    if classed is None:
        failed.append(f"{STOP_AIRPORT} unmatched rows with a class: count missing")
    elif not lo <= classed <= hi:
        failed.append(f"{STOP_AIRPORT} unmatched rows with a class: {classed} outside [{lo}, {hi}]")

    params_hash = report.get("params_hash")
    values["params_hash"] = params_hash
    if params_hash != TABLE_PARAMS_HASH:
        failed.append(f"params_hash mismatch: {params_hash!r} != {TABLE_PARAMS_HASH!r}")

    return {"ok": not failed, "failed": failed, "values": values}


def require_stop_rules(table_path) -> dict:
    """Reads report.json from the table file's own directory and runs check_stop_rules; raises ValueError naming
    the path when the file is missing, or listing every failed reason when the check does not pass."""
    report_path = pathlib.Path(table_path).resolve().parent / "report.json"
    if not report_path.exists():
        raise ValueError(f"require_stop_rules: missing report.json at {report_path}")
    report = json.loads(report_path.read_text())
    check = check_stop_rules(report)
    if not check["ok"]:
        raise ValueError("require_stop_rules: stop rule failure(s): " + "; ".join(check["failed"]))
    return check


# ---------------------------------------------------------------------------------------------- frame assembly

_RAW_MONTH_RE = re.compile(r"^training_2025-(\d{2})-")


def _raw_file_month(path) -> int | None:
    """The MM in a training_2025-MM-... raw file name, or None when the name does not match that convention."""
    m = _RAW_MONTH_RE.match(pathlib.Path(path).name)
    return int(m.group(1)) if m else None


def _select_raw_files(raw_files, months) -> list:
    """Only the raw files whose filename's MM is in `months`, in month order. Raises ValueError naming any
    requested month with no matching raw file."""
    by_month: dict = {}
    for f in raw_files:
        mm = _raw_file_month(f)
        if mm is not None:
            by_month.setdefault(mm, []).append(pathlib.Path(f))
    missing = sorted(m for m in months if not by_month.get(m))
    if missing:
        raise ValueError(f"load_frame: no raw file for month(s) {missing}")
    out = []
    for m in sorted(months):
        out.extend(by_month[m])
    return out


def load_frame(lomo_dir, table_path, raw_files, months=None) -> pd.DataFrame:
    """Every row of the requested LOMO month files (`months=None`: all 12, the historical default), joined to the
    UAD table and then to the raw columns (slc_slices.attach). With `months` given, ONLY those months' oof_MXX.parquet
    files are opened, and raw_files is filtered to only the raw file(s) of those months (by the MM in their
    training_2025-MM-... name); a requested month with no raw file raises. The missing-file and duplicate-id checks
    below apply only to the requested months.

    Refuses: a missing LOMO month file (of the requested months); a duplicated MVT_ID_mvt across those files; a
    duplicated table id; a baseline row (matched or unmatched) with no table row (AMEND-A1 item 6: every lane, not
    only unmatched); a row whose lane == "unmatched" disagrees with the table's unmatched flag.
    Adds v = pred, p = unm_p_fill, est = exit_rel_s, sp = seconds from SCHED_TIME_UTC_mvt to MVT_TIME_UTC_mvt (NaN
    when SCHED is null), scope = lane is unmatched and decided_by in SCOPE, storm = storm_A > STORM_A_S (NaN ->
    False). Matched rows keep p NaN (unm_p_fill is already null for them)."""
    lomo_dir = pathlib.Path(lomo_dir)
    if months is None:
        month_list = list(ALL_MONTHS)
        raw_selected = list(raw_files)
    else:
        month_list = sorted(int(m) for m in months)
        raw_selected = _select_raw_files(raw_files, month_list)

    month_files = [lomo_dir / f"oof_M{m:02d}.parquet" for m in month_list]
    missing = [f.name for f in month_files if not f.exists()]
    if missing:
        raise ValueError(f"load_frame: missing LOMO month file(s): {missing}")
    lomo = pd.concat([pd.read_parquet(f, columns=LOMO_COLS) for f in month_files], ignore_index=True)   # only what is used
    if lomo.MVT_ID_mvt.duplicated().any():
        raise ValueError(f"load_frame: {int(lomo.MVT_ID_mvt.duplicated().sum())} duplicate LOMO id(s)")

    table = pd.read_parquet(table_path)
    if table.MVT_ID_mvt.duplicated().any():
        raise ValueError(f"load_frame: {int(table.MVT_ID_mvt.duplicated().sum())} duplicate table id(s)")

    merged = lomo.merge(table[["MVT_ID_mvt", "unmatched", "usable", "exit_rel_s", "cls"]],
                        on="MVT_ID_mvt", how="left", indicator=True)
    is_unmatched_lomo = merged.lane.to_numpy() == "unmatched"
    present = (merged["_merge"] == "both").to_numpy()
    missing_table = ~present
    if missing_table.any():
        raise ValueError(f"load_frame: {int(missing_table.sum())} baseline row(s) have no table row")
    disagree = is_unmatched_lomo != merged["unmatched"].to_numpy(dtype=bool)
    if disagree.any():
        raise ValueError(f"load_frame: {int(disagree.sum())} row(s) disagree between lane and the table's "
                         f"unmatched flag")
    merged = merged.drop(columns="_merge")

    raw = SLC.load_raw(raw_selected)
    frame = SLC.attach(merged, raw)

    frame["v"] = frame["pred"].astype("float64")
    frame["p"] = frame["unm_p_fill"].astype("float64")
    frame["est"] = frame["exit_rel_s"].astype("float64")
    sched = pd.to_datetime(frame["SCHED_TIME_UTC_mvt"], utc=True)
    mvt = pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True)
    frame["sp"] = (mvt - sched).dt.total_seconds()
    frame["scope"] = (frame["lane"] == "unmatched") & frame["decided_by"].isin(SCOPE)
    frame["storm"] = frame["storm_A"].astype("float64") > STORM_A_S
    return frame


# ---------------------------------------------------------------------------------------------- the blend

def u_prime(p, sp, est, c) -> np.ndarray:
    """p*sp + (1-p)*(est+c). A row with no schedule or no fill probability (p or sp is NaN) has no schedule-copy
    branch: it falls back to est + c."""
    p = np.asarray(p, dtype="float64")
    sp = np.asarray(sp, dtype="float64")
    est = np.asarray(est, dtype="float64")
    c = np.asarray(c, dtype="float64")
    fallback = est + c
    bad = np.isnan(p) | np.isnan(sp)
    with np.errstate(invalid="ignore"):
        blended = p * sp + (1.0 - p) * fallback
    return np.where(bad, fallback, blended)


def _c_of(frame: pd.DataFrame, off: dict) -> np.ndarray:
    """The per-row airport offset: off["by_airport"][ap] if present, else the pooled value."""
    return frame.ap.map(off["by_airport"]).fillna(off["pooled"]).to_numpy(dtype="float64")


def offsets(frame: pd.DataFrame, months, exclude=None) -> dict:
    """Per-airport mean of (y - est) over ANY-lane class-A1 rows within `months`, bounded to |y - est| <=
    OFFSET_MAX_ABS_S; an airport with fewer than OFFSET_MIN_ROWS such rows takes the pooled mean instead.
    `exclude=(ap, day)` drops that airport-day's rows before fitting. Raises ValueError if nothing qualifies."""
    months_set = {int(m) for m in months}
    is_a1 = frame.cls.to_numpy() == "A1"
    in_months = frame.month.astype(int).isin(months_set).to_numpy()
    diff = frame.y.to_numpy(dtype="float64") - frame.est.to_numpy(dtype="float64")
    within_bound = np.abs(diff) <= OFFSET_MAX_ABS_S
    ok = is_a1 & in_months & within_bound
    if exclude is not None:
        ex_ap, ex_day = exclude
        ok = ok & ~((frame.ap.to_numpy() == ex_ap) & (frame.day.to_numpy() == ex_day))
    if not ok.any():
        raise ValueError("offsets: no qualifying row for the requested months")

    sub_diff = diff[ok]
    sub_ap = frame.ap.to_numpy()[ok]
    pooled = float(sub_diff.mean())
    by_airport: dict = {}
    n_by_airport: dict = {}
    pooled_used: list = []
    for ap in sorted(set(sub_ap)):
        m = sub_ap == ap
        n = int(m.sum())
        n_by_airport[ap] = n
        if n < OFFSET_MIN_ROWS:
            by_airport[ap] = pooled
            pooled_used.append(ap)
        else:
            by_airport[ap] = float(sub_diff[m].mean())
    return {"by_airport": by_airport, "pooled": pooled, "n_by_airport": n_by_airport, "pooled_used": pooled_used}


def _top10_share(terms: np.ndarray):
    """The sum of the 10 terms of largest absolute value divided by the sum of every term (AMEND-A1 item 5
    diagnostic); None when there are no terms or their total is 0. Fewer than 10 terms: all of them are "the 10
    largest"."""
    if terms.size == 0:
        return None
    total = float(terms.sum())
    if total == 0.0:
        return None
    order = np.argsort(-np.abs(terms))
    return float(terms[order[:10]].sum() / total)


def weights(frame: pd.DataFrame, off: dict, months, exclude=None) -> dict:
    """Separately for the calm (~storm) and storm population rows of class A1 within `months`: w = sum((u'-v)(y-v))
    / sum((u'-v)^2), clipped to [0, 1]. A zero denominator raises, naming the regime — except an empty storm
    regime, which the floor rule below handles instead. With fewer than STORM_MIN_ROWS storm rows,
    w1 = max(w0, W1_FLOOR). Also returns w0_top10_share / w1_top10_share (AMEND-A1 item 5): the share of each
    regime's sum(u'-v)(y-v) carried by its 10 largest-magnitude terms, a diagnostic only — it never feeds back into
    w0 or w1."""
    months_set = {int(m) for m in months}
    in_scope = frame.scope.to_numpy(dtype=bool)
    is_a1 = frame.cls.to_numpy() == "A1"
    in_months = frame.month.astype(int).isin(months_set).to_numpy()
    sel = in_scope & is_a1 & in_months
    if exclude is not None:
        ex_ap, ex_day = exclude
        sel = sel & ~((frame.ap.to_numpy() == ex_ap) & (frame.day.to_numpy() == ex_day))

    c = _c_of(frame, off)
    up = u_prime(frame.p.to_numpy(dtype="float64"), frame.sp.to_numpy(dtype="float64"),
                frame.est.to_numpy(dtype="float64"), c)
    v = frame.v.to_numpy(dtype="float64")
    y = frame.y.to_numpy(dtype="float64")
    storm = frame.storm.to_numpy(dtype=bool)
    terms = (up - v) * (y - v)

    def fit(mask: np.ndarray, label: str) -> float:
        num = float(((up[mask] - v[mask]) * (y[mask] - v[mask])).sum())
        den = float(((up[mask] - v[mask]) ** 2).sum())
        if den == 0.0:
            raise ValueError(f"weights: zero denominator fitting the {label} regime")
        return num / den

    calm_mask = sel & ~storm
    storm_mask = sel & storm
    n0 = int(calm_mask.sum())
    w0_raw = fit(calm_mask, "calm")
    w0 = float(np.clip(w0_raw, 0.0, 1.0))

    n1 = int(storm_mask.sum())
    w1_raw = fit(storm_mask, "storm") if n1 > 0 else None
    if n1 < STORM_MIN_ROWS:
        w1 = float(max(w0, W1_FLOOR))
        w1_rule = "floor"
    else:
        w1 = float(np.clip(w1_raw, 0.0, 1.0))
        w1_rule = "fit"
    return {"w0": w0, "w1": w1, "n0": n0, "n1": n1, "w0_raw": w0_raw, "w1_raw": w1_raw, "w1_rule": w1_rule,
            "w0_top10_share": _top10_share(terms[calm_mask]), "w1_top10_share": _top10_share(terms[storm_mask])}


def constants(frame: pd.DataFrame, months=DESIGN_MONTHS, exclude=None) -> dict:
    """offsets() + weights(), fitted using ONLY rows whose month is in `months`."""
    months_t = tuple(int(m) for m in months)
    off = offsets(frame, months_t, exclude=exclude)
    w = weights(frame, off, months_t, exclude=exclude)
    return {"offsets": off, "weights": w, "months": sorted(months_t), "registration": REGISTRATION}


def check_constants(const: dict, table_path, lomo_dir) -> None:
    """Provenance check (AMEND-A1 item 6): raises ValueError naming the first mismatch between a constants file (as
    produced by the `constants` CLI command, which bundles _input_hashes into its own keys) and its current inputs:
    a different registration; a `months` list other than the registered design months; a table file whose sha256 no
    longer matches `table_sha256`; or a `lomo_files` list that is not exactly the ten design-month files with their
    current sha256 (an entry naming a file that is now missing is a mismatch). Returns None (does not raise) when
    everything matches."""
    if const.get("registration") != REGISTRATION:
        raise ValueError(f"check_constants: registration {const.get('registration')!r} != {REGISTRATION!r}")
    if const.get("months") != sorted(DESIGN_MONTHS):
        raise ValueError(f"check_constants: months {const.get('months')!r} != {sorted(DESIGN_MONTHS)!r}")

    table_sha = _sha256(pathlib.Path(table_path))
    if const.get("table_sha256") != table_sha:
        raise ValueError(f"check_constants: table_sha256 mismatch for {table_path}")

    lomo_dir = pathlib.Path(lomo_dir)
    listed = const.get("lomo_files") or []
    expected_names = [f"oof_M{m:02d}.parquet" for m in sorted(DESIGN_MONTHS)]
    if sorted(e.get("file") for e in listed) != sorted(expected_names):
        raise ValueError(f"check_constants: lomo_files lists {sorted(e.get('file') for e in listed)!r}, "
                         f"expected exactly {sorted(expected_names)!r}")
    for entry in listed:
        name = entry.get("file")
        cur_path = lomo_dir / name
        if not cur_path.exists():
            raise ValueError(f"check_constants: lomo_files entry {name!r} is missing from {lomo_dir}")
        if entry.get("sha256") != _sha256(cur_path):
            raise ValueError(f"check_constants: lomo_files sha256 mismatch for {name!r}")


def apply(frame: pd.DataFrame, const: dict) -> np.ndarray:
    """new = v + w*(u'-v) for rows with scope and cls in {A1, A2} (w = w1 if storm else w0; A2 rows take
    A2_WEIGHT_FACTOR of that), rounded and floored at 1 s. Every other row returns v unchanged, with no rounding.
    An airport missing from the offsets uses the pooled value (_c_of)."""
    v = frame.v.to_numpy(dtype="float64")
    out = v.copy()
    cls = frame.cls.to_numpy()
    in_scope = frame.scope.to_numpy(dtype=bool)
    eligible_cls = np.isin(cls, ["A1", "A2"])
    sel = in_scope & eligible_cls
    if sel.any():
        off = const["offsets"]
        w0 = float(const["weights"]["w0"])
        w1 = float(const["weights"]["w1"])
        c = _c_of(frame, off)
        up = u_prime(frame.p.to_numpy(dtype="float64"), frame.sp.to_numpy(dtype="float64"),
                    frame.est.to_numpy(dtype="float64"), c)
        storm = frame.storm.to_numpy(dtype=bool)
        w = np.where(storm, w1, w0)
        factor = np.where(cls == "A2", A2_WEIGHT_FACTOR, 1.0)
        new = v + factor * w * (up - v)
        floored = np.maximum(new, 1.0)
        out = np.where(sel, np.rint(floored), out)
    return out


# ---------------------------------------------------------------------------------------------- scoring

def population(frame: pd.DataFrame, months) -> np.ndarray:
    """scope and cls in {A1, A2} and month in `months`."""
    months_set = {int(m) for m in months}
    in_scope = frame.scope.to_numpy(dtype=bool)
    eligible_cls = np.isin(frame.cls.to_numpy(), ["A1", "A2"])
    in_months = frame.month.astype(int).isin(months_set).to_numpy()
    return in_scope & eligible_cls & in_months


def records(frame: pd.DataFrame, new, months) -> tuple:
    """The two compare records (base: pred = v; candidate: pred = new) on the population rows."""
    pop = population(frame, months)
    new_arr = np.asarray(new, dtype="float64")[pop]
    sub = frame.loc[pop]
    base = sub[["MVT_ID_mvt", "y", "day", "month", "ap"]].copy()
    base["pred"] = sub["v"].to_numpy(dtype="float64")
    cand = base.copy()
    cand["pred"] = new_arr
    return base[SCORE_COLS].reset_index(drop=True), cand[SCORE_COLS].reset_index(drop=True)


def _rmse_ok(rmse_fall: float) -> bool:
    """The H1 RMSE bar: NaN (an empty population) never passes; otherwise rmse_fall >= H1_MIN_RMSE_FALL. Always a
    genuine Python bool (never numpy.bool_), since rmse_fall may be a numpy scalar."""
    if np.isnan(rmse_fall):
        return False
    return bool(rmse_fall >= H1_MIN_RMSE_FALL)


def h1(frame: pd.DataFrame, const: dict, n_boot: int = N_BOOT, seed: int = SEED, exclude_airports: tuple = ()) -> dict:
    """H-UAD-1: protocols.compare/promotion_gate on the SCORED_MONTHS population, plus a pooled-RMSE fall check and
    a both-scored-months-populated check. TRUE iff the gate promotes AND rmse_fall >= H1_MIN_RMSE_FALL AND both
    scored months hold at least one population row; FALSE (with a one-line reason) otherwise. `exclude_airports`
    (default (): no exclusion, the decisional H-UAD-1 behaviour) removes rows whose `ap` is in it from the compared
    population (base, candidate, the RMSEs and n_by_month) before anything else runs; used for the reported-only
    EHAM-excluded readout (AMEND-A1 item 5), never for the decisional verdict. If the population is empty after
    exclusion (or was already empty), returns FALSE directly without calling protocols.compare, which raises on a
    record that shares no row."""
    new = apply(frame, const)
    base, cand = records(frame, new, SCORED_MONTHS)
    if exclude_airports:
        keep = ~base.ap.isin(list(exclude_airports))
        base = base.loc[keep].reset_index(drop=True)
        cand = cand.loc[keep].reset_index(drop=True)

    counts = base.month.astype(int).value_counts().to_dict()
    n_by_month = {int(m): int(counts.get(int(m), 0)) for m in SCORED_MONTHS}
    months_ok = all(n_by_month[int(m)] > 0 for m in SCORED_MONTHS)

    if len(base) == 0:
        return {"compare": None, "gate": None, "rmse_base": float("nan"), "rmse_new": float("nan"),
                "rmse_fall": float("nan"), "n_by_month": n_by_month, "verdict": "FALSE",
                "reason": "a scored month has no population row"}

    y = base.y.to_numpy(dtype="float64")
    pb = base.pred.to_numpy(dtype="float64")
    pc = cand.pred.to_numpy(dtype="float64")
    rmse_base = float(np.sqrt(np.mean((y - pb) ** 2)))
    rmse_new = float(np.sqrt(np.mean((y - pc) ** 2)))
    rmse_fall = float("nan") if rmse_base == 0.0 else 1.0 - rmse_new / rmse_base

    rep = P.compare(base, cand, n_boot=n_boot, seed=seed)
    gate = P.promotion_gate(rep)
    gate_promotes = bool(gate["promote"])
    rmse_ok = _rmse_ok(rmse_fall)

    if gate_promotes and rmse_ok and months_ok:
        verdict, reason = "TRUE", "gate promotes, RMSE fell >= 10%, both scored months populated"
    else:
        bad = []
        if not gate_promotes:
            bad.append("gate does not promote")
        if not rmse_ok:
            bad.append("RMSE fall below 10%")
        if not months_ok:
            bad.append("a scored month has no population row")
        verdict, reason = "FALSE", "; ".join(bad)

    return {"compare": rep, "gate": gate, "rmse_base": rmse_base, "rmse_new": rmse_new, "rmse_fall": rmse_fall,
            "n_by_month": n_by_month, "verdict": verdict, "reason": reason}


def h2(frame: pd.DataFrame, months=DESIGN_MONTHS) -> dict:
    """H-UAD-2: for every (ap, day) of `months` with storm and >= H2_MIN_ROWS population rows, refit constants()
    excluding that event, apply it to the event's own population rows, and accumulate chain (v) vs blend (new) SSE.
    TRUE iff ratio = sqrt(SSE_blend / SSE_chain) <= H2_MAX_RATIO; NOT TESTABLE if no event qualifies."""
    months_t = tuple(int(m) for m in months)
    pop = population(frame, months_t)
    storm = frame.storm.to_numpy(dtype=bool)
    event_rows = pop & storm
    ap = frame.ap.to_numpy().astype(str)
    day = frame.day.to_numpy().astype(str)

    sub = pd.DataFrame({"ap": ap[event_rows], "day": day[event_rows]})
    counts = sub.groupby(["ap", "day"]).size()
    events = sorted(k for k, n in counts.items() if n >= H2_MIN_ROWS)
    if not events:
        return {"verdict": "NOT TESTABLE", "ratio": None, "sse_chain": 0.0, "sse_blend": 0.0, "events": []}

    y_all = frame.y.to_numpy(dtype="float64")
    v_all = frame.v.to_numpy(dtype="float64")
    sse_chain = 0.0
    sse_blend = 0.0
    table = []
    for ap_ev, day_ev in events:
        c = constants(frame, months_t, exclude=(ap_ev, day_ev))
        new = apply(frame, c)
        ev_mask = event_rows & (ap == ap_ev) & (day == day_ev)
        n = int(ev_mask.sum())
        sc = float(((y_all[ev_mask] - v_all[ev_mask]) ** 2).sum())
        sb = float(((y_all[ev_mask] - new[ev_mask]) ** 2).sum())
        sse_chain += sc
        sse_blend += sb
        storm_a_val = float(frame.loc[ev_mask, "storm_A"].iloc[0]) if n else float("nan")
        table.append({"ap": ap_ev, "day": day_ev, "n": n, "storm_A": storm_a_val,
                      "chain_rmse": float(np.sqrt(sc / n)) if n else float("nan"),
                      "blend_rmse": float(np.sqrt(sb / n)) if n else float("nan")})
    ratio = float(np.sqrt(sse_blend / sse_chain)) if sse_chain > 0 else float("nan")
    verdict = "TRUE" if (not np.isnan(ratio) and ratio <= H2_MAX_RATIO) else "FALSE"
    return {"verdict": verdict, "ratio": ratio, "sse_chain": sse_chain, "sse_blend": sse_blend, "events": table}


def breakdown(frame: pd.DataFrame, new, months) -> dict:
    """Reported, not decisional: gain in bmse (protocols.BOARD_N) by airport / class / regime, body (y <=
    protocols.BODY_MAX_S) vs tail, the non-lottery split (slc_slices.masks), the share of the body gain in its 10
    largest rows, and estimate coverage (population rows / scope rows) by airport."""
    months_t = tuple(int(m) for m in months)
    pop = population(frame, months_t)
    y = frame.y.to_numpy(dtype="float64")
    v = frame.v.to_numpy(dtype="float64")
    new_arr = np.asarray(new, dtype="float64")
    gain = ((y - v) ** 2 - (y - new_arr) ** 2) / P.BOARD_N
    ap = frame.ap.to_numpy().astype(str)
    cls = frame.cls.to_numpy().astype(str)
    storm = frame.storm.to_numpy(dtype=bool)
    body = y <= P.BODY_MAX_S

    def total(mask: np.ndarray) -> float:
        return float(gain[mask].sum())

    by_airport = {a: total(pop & (ap == a)) for a in sorted(set(ap[pop]))}
    by_class = {c: total(pop & (cls == c)) for c in sorted(set(cls[pop]))}
    by_regime = {"calm": total(pop & ~storm), "storm": total(pop & storm)}
    body_gain = total(pop & body)
    tail_gain = total(pop & ~body)
    total_gain = total(pop)

    lottery = SLC.masks(frame)["lottery"]
    non_lottery_gain = total(pop & ~lottery)

    body_idx = np.flatnonzero(pop & body)
    if body_idx.size:
        order = np.argsort(-np.abs(gain[body_idx]))
        top10 = body_idx[order[:10]]
        top10_share = float(gain[top10].sum() / body_gain) if body_gain != 0.0 else float("nan")
    else:
        top10_share = float("nan")

    scope_in_months = frame.scope.to_numpy(dtype=bool) & frame.month.astype(int).isin(months_t).to_numpy()
    coverage = {}
    for a in sorted(set(ap[scope_in_months])):
        n_scope = int((scope_in_months & (ap == a)).sum())
        n_pop = int((pop & (ap == a)).sum())
        coverage[a] = (n_pop / n_scope) if n_scope else float("nan")

    return {"by_airport": by_airport, "by_class": by_class, "by_regime": by_regime, "body_gain": body_gain,
            "tail_gain": tail_gain, "non_lottery_gain": non_lottery_gain, "total_gain": total_gain,
            "body_top10_share": top10_share, "coverage_by_airport": coverage}


# ---------------------------------------------------------------------------------------------- CLI

def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lomo_month_files(lomo_dir: pathlib.Path, months) -> list:
    return [lomo_dir / f"oof_M{m:02d}.parquet" for m in sorted(int(m) for m in months)]


def _input_hashes(table_path: pathlib.Path, lomo_dir: pathlib.Path, months) -> dict:
    """Label-free: file identities only (sha256), never a label or prediction value. `months` restricts lomo_files
    to those months' oof_MXX.parquet files (AMEND-A1 item 6: the constants command's output names only the ten
    design-month files it actually opened)."""
    return {"table_sha256": _sha256(table_path),
            "lomo_files": [{"file": f.name, "sha256": _sha256(f)} for f in _lomo_month_files(lomo_dir, months)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("constants", help="fit the registered constants on the 2025 design months only")
    pc.add_argument("--lomo-dir", required=True)
    pc.add_argument("--table", required=True)
    pc.add_argument("--raw-dir", required=True)
    pc.add_argument("--out", required=True)

    pv = sub.add_parser("verdict", help="apply the blend and compute H-UAD-1 / H-UAD-2")
    pv.add_argument("--lomo-dir", required=True)
    pv.add_argument("--table", required=True)
    pv.add_argument("--raw-dir", required=True)
    pv.add_argument("--constants", required=True)
    pv.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    lomo_dir = pathlib.Path(a.lomo_dir).resolve()
    table_path = pathlib.Path(a.table).resolve()
    raw_dir = pathlib.Path(a.raw_dir).resolve()
    out_path = pathlib.Path(a.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")
    stop_check = require_stop_rules(table_path)
    raw_files = sorted(raw_dir.glob("training_2025-*.parquet"))

    if a.command == "constants":
        frame = load_frame(lomo_dir, table_path, raw_files, months=DESIGN_MONTHS)
        frame = frame[~frame.month.astype(int).isin(SCORED_MONTHS)].reset_index(drop=True)   # second belt
        const = constants(frame, DESIGN_MONTHS)
        out = {**const, **_input_hashes(table_path, lomo_dir, DESIGN_MONTHS), "n_rows": int(len(frame)),
              "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "stop_rules": stop_check}
        out_path.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(json.dumps({"months": out["months"], "n_rows": out["n_rows"]}, indent=1))
        return 0

    const = json.loads(pathlib.Path(a.constants).resolve().read_text())
    check_constants(const, table_path, lomo_dir)
    frame = load_frame(lomo_dir, table_path, raw_files)
    new = apply(frame, const)
    h1_rep = h1(frame, const)
    h1_excl_rep = dict(h1(frame, const, exclude_airports=(STOP_AIRPORT,)))
    h1_excl_rep["readout"] = h1_excl_rep.pop("verdict")   # reported only: it decides nothing
    h2_rep = h2(frame, DESIGN_MONTHS)
    out = {"h1": h1_rep, "h1_excluding_EHAM": h1_excl_rep, "h2": h2_rep,
          "breakdown_scored": breakdown(frame, new, SCORED_MONTHS),
          "breakdown_design": breakdown(frame, new, DESIGN_MONTHS),
          "input_hashes": _input_hashes(table_path, lomo_dir, ALL_MONTHS), "stop_rules": stop_check}
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({"h1_verdict": h1_rep["verdict"], "h2_verdict": h2_rep["verdict"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
