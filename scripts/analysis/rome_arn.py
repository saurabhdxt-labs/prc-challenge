"""ARN — ARC with normalised designators (ARC-N) and cross-joined NM records (XJ), LIRF unmatched departures, held-out
test (plans/PREREG_rome_arcn_xj_2026_09_16.md; NO upload, NO submission file).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/rome_arn.py

Reuses `scripts/analysis/arr_row_clocks.py` (ARC) by import: `join_arr`, `sub_rule`, `label_class`, `secs`, `arr_frame`,
`dep_frame`. ARC-N is ARC's own join (`join_arr`) with the DEP-side designator normalised first (`norm`); its
sub-rule thresholds are ARC's own, unchanged (`sub_rule` is called directly, not reimplemented — the registration
says so and this module tests that `sub_rule`'s boundaries are what ARN relies on). XJ is a new DEP-row-to-DEP-row
cross-join: an unmatched LIRF departure X whose normalised designator equals another, matched LIRF departure H's own
`CALLSIGN_flt` borrows H's LOBT/AOBT_3, restricted to the same source file, ADES agreement, and a plausibility window
on X's take-off vs H's push-back.

Memory: a probe (this session, interactive) showed the concatenated-twelve-months load + ARC join that would be
needed to compute ARC's *exact* 2025 "reached" rows across month boundaries peaks at ~2.2 GB RSS — over this run's
1.5 GB budget (see reports/rome_arn_2026_09_16.json note). A single training file peaks at ~0.28 GB. The 2025 pass
below is therefore computed ONE FILE AT A TIME: ARC's raw-join exclusion, ARC-N's ARR pool, and XJ's host pool are
all scoped to that file only (XJ's own rule already requires "the same file"; this applies the same scoping to ARC's
exclusion and to ARC-N, for memory safety and internal consistency). A 2025 departure within ~8 h of a month
boundary could in principle have a true ARC/ARC-N match in the adjacent file that this per-file computation misses —
disclosed as a limitation of the 2025 diagnostic only (see module docstring RESULT / final report).

The 2026 pass is EXACT, not an approximation: ranking.parquet is already a single file, matching ARC's own real
2026 production computation (`arr_row_clocks.main()`'s `d26`/`arr_frame(r)`) exactly — no month-boundary lookahead is
needed and there is no memory pressure (689,534 rows, ~11 label-free columns).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import arr_row_clocks as ARC  # noqa: E402

PREREG = ROOT / "plans" / "PREREG_rome_arcn_xj_2026_09_16.md"
REPORT = ROOT / "reports" / "rome_arn_2026_09_16.json"
RAW = ROOT / "data" / "raw"
RANK = RAW / "ranking.parquet"
V19 = ROOT / "submissions" / "merry-quicksand_v19.parquet"

BOARD_N = 344_841.0
SP_LO, SP_HI = ARC.SP_LO, ARC.SP_HI            # 3,600 / 24,000 — registration: "unchanged" from ARC
XJ_WIN = 10_800.0                              # registration: 0 <= MVT_X - AOBT_3(H) <= 10,800 s
HELD_OUT_MONTHS = (1, 7)                       # January, July 2025 — registration's held-out months

NORM_RE = re.compile(r"^([A-Z]{3})([A-Z]{1,2})(\d.*)$")

NEED_COLS_2025 = sorted(set(ARC.DEP_COLS) | set(ARC.ARR_COLS) | {"ADES_flt", "TAXITIME_SEC_mvt"})
NEED_COLS_2026 = sorted(set(ARC.DEP_COLS) | set(ARC.ARR_COLS) | {"ADES_flt"})

RULED_DTYPES = {
    "MVT_ID_mvt": "float64", "FLIGHT_mvt": "object", "src": "int64", "sp": "float64", "sched": "float64",
    "mvt": "float64", "method": "object", "rule": "object", "rule_value": "float64", "resolved_lobt": "float64",
    "resolved_aobt3": "float64", "y": "float64",
}


# --------------------------------------------------------------------------- pure (unit-tested)

def norm(s) -> str | None:
    """Registered designator normalisation. Non-string -> None. If `s` matches ^([A-Z]{3})([A-Z]{1,2})(\\d.*)$ AND
    group 1 ends with group 2 -> group1 + group3 (RYRR88TM -> RYR88TM: group1 'RYR', group2 'R', group3 '88TM', 'RYR'
    ends with 'R'); otherwise `s` unchanged (AZA123, BAW12A: no A-Z run between the 3-letter prefix and the first
    digit, so the regex itself does not match; ABCXY123: matches structurally but 'ABC' does not end with 'XY')."""
    if not isinstance(s, str):
        return None
    m = NORM_RE.match(s)
    if m and m.group(1).endswith(m.group(2)):
        return m.group(1) + m.group(3)
    return s


def norm_arr(a) -> np.ndarray:
    """Elementwise `norm` over an array-like of designators. Object dtype (str or None)."""
    return np.array([norm(s) for s in a], dtype=object)


def in_scope(aobt3_own, sp) -> np.ndarray:
    """Registration's scope: AOBT_3_flt (own, unjoined) null AND 3,600 <= sp < 24,000 (ARC's own SP_LO/SP_HI,
    unchanged)."""
    aobt3_own = np.asarray(aobt3_own, dtype="float64")
    sp = np.asarray(sp, dtype="float64")
    return np.isnan(aobt3_own) & np.isfinite(sp) & (sp >= SP_LO) & (sp < SP_HI)


def arcn_join(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """ARC's own join (`arr_row_clocks.join_arr`), with the DEP-side designator normalised first — the registration's
    ARC-N. dep: MVT_ID_mvt (unique), ADEP_mvt, ADES_mvt, FLIGHT_mvt (RAW, unnormalised), mvt (s). arr: as
    `join_arr`'s `arr` (ADEP_flt, ADES_mvt, CALLSIGN_flt, mvt (s), lobt (s), aobt3 (s)). Inherits `join_arr`'s window
    (600 s exclusive, 8 h inclusive), earliest-MVT-wins tie-break, and the LOBT/AOBT_3-non-null requirement — only
    the join KEY changes (normalised FLIGHT_mvt vs CALLSIGN_flt, instead of raw vs raw)."""
    dep_n = dep.copy()
    dep_n["FLIGHT_mvt"] = norm_arr(dep.FLIGHT_mvt.to_numpy())
    return ARC.join_arr(dep_n, arr)


def xj_join(x: pd.DataFrame, host: pd.DataFrame) -> pd.DataFrame:
    """One row per X row (same order): xj_lobt, xj_aobt3 in epoch s (NaN/NaN if untouched — zero or >= 2 qualifying
    hosts, or X's own `fn` is None).
    x: MVT_ID_mvt (unique), fn (normalised FLIGHT_mvt of X), ADES_mvt, mvt (s), src.
    host: candidate LIRF DEP rows, one row per host CANDIDATE (the caller need not pre-filter "matched" — this
    function itself requires AOBT_3_flt and CALLSIGN_flt non-null): FLIGHT_mvt (host's own raw designator),
    CALLSIGN_flt, ADES_flt, LOBT_flt (s), AOBT_3_flt (s), src.
    A host H qualifies for an X row iff: norm(H.FLIGHT_mvt) != H.CALLSIGN_flt, H.CALLSIGN_flt == X.fn,
    H.ADES_flt == X.ADES_mvt, H.src == X.src, and 0 <= X.mvt - H.AOBT_3_flt <= 10,800 (inclusive both ends).
    Exactly one qualifying host -> that host's LOBT_flt / AOBT_3_flt; zero or >= 2 -> NaN/NaN (untouched)."""
    if x.MVT_ID_mvt.duplicated().any():
        raise ValueError("xj_join: duplicate MVT_ID_mvt in x")
    h = host[host.AOBT_3_flt.notna() & host.CALLSIGN_flt.notna()].copy()
    h["host_fn"] = norm_arr(h.FLIGHT_mvt.to_numpy())
    h = h[np.asarray(h.host_fn, dtype=object) != h.CALLSIGN_flt.to_numpy(dtype=object)]
    xd = x[x.fn.notna()][["MVT_ID_mvt", "fn", "ADES_mvt", "mvt", "src"]]
    m = xd.merge(h[["CALLSIGN_flt", "ADES_flt", "LOBT_flt", "AOBT_3_flt", "src"]],
                 left_on=["fn", "ADES_mvt", "src"], right_on=["CALLSIGN_flt", "ADES_flt", "src"])
    dt = m.mvt - m.AOBT_3_flt
    m = m[(dt >= 0.0) & (dt <= XJ_WIN)]
    cnt = m.groupby("MVT_ID_mvt").size()
    keep = m.MVT_ID_mvt.isin(cnt[cnt == 1].index)
    m1 = m[keep].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    out = pd.DataFrame({"MVT_ID_mvt": x.MVT_ID_mvt.to_numpy()})
    out["xj_lobt"] = m1.LOBT_flt.reindex(out.MVT_ID_mvt).to_numpy(dtype="float64")
    out["xj_aobt3"] = m1.AOBT_3_flt.reindex(out.MVT_ID_mvt).to_numpy(dtype="float64")
    return out


def rule_value(rule, mvt, aobt3, sp) -> np.ndarray:
    """ARC's own sub-rule values, unchanged (registration): 'ord' -> max(rint(MVT - AOBT_3), 1); 'fill' -> rint(sp);
    anything else (untouched / 'none' / out of scope) -> NaN."""
    rule = np.asarray(rule)
    mvt, aobt3, sp = (np.asarray(v, dtype="float64") for v in (mvt, aobt3, sp))
    out = np.full(len(rule), np.nan, dtype="float64")
    o = rule == "ord"
    out[o] = np.maximum(np.rint(mvt[o] - aobt3[o]), 1.0)
    f = rule == "fill"
    out[f] = np.rint(sp[f])
    return out


def call_correct(rule, cls) -> np.ndarray:
    """True iff the rule's call matches ARC.1's `label_class`: ord <-> ordinary, fill <-> fill. Rows whose rule is
    neither 'ord' nor 'fill' are not calls at all (returns False for them; callers restrict to ord/fill rows first)."""
    rule, cls = np.asarray(rule), np.asarray(cls)
    return np.where(rule == "ord", cls == "ordinary", np.where(rule == "fill", cls == "fill", False))


def verdict(n_held: int, n_wrong: int) -> str:
    """Registration: WORKING iff n_held >= 2 and every held-out call correct; NOT WORKING iff any held-out call is
    wrong (decisive regardless of n_held — a single wrong call already falsifies); INCONCLUSIVE iff n_held < 2 and no
    wrong call was seen (0 or 1 held-out rows, correct)."""
    if n_wrong > 0:
        return "NOT WORKING"
    if n_held >= 2:
        return "WORKING"
    return "INCONCLUSIVE"


def gain_sse(y, inc, pred) -> np.ndarray:
    """ARC-style per-row SSE gain: (y - inc)^2 - (y - pred)^2 (positive = pred closer to the assumed true value y
    than the incumbent inc was)."""
    y, inc, pred = (np.asarray(v, dtype="float64") for v in (y, inc, pred))
    return (y - inc) ** 2 - (y - pred) ** 2


def stake_alt(rule, mvt, aobt3, sp, v19) -> np.ndarray:
    """The other class's value (registration): for a 'fill' call, the ordinary alternative is MVT - AOBT_3 if
    finite, else v19; for an 'ord' call, the alternative is rint(sp). Anything else -> NaN."""
    rule = np.asarray(rule)
    mvt, aobt3, sp, v19 = (np.asarray(v, dtype="float64") for v in (mvt, aobt3, sp, v19))
    out = np.full(len(rule), np.nan, dtype="float64")
    f = rule == "fill"
    alt_ord = mvt[f] - aobt3[f]
    out[f] = np.where(np.isfinite(alt_ord), alt_ord, v19[f])
    o = rule == "ord"
    out[o] = np.rint(sp[o])
    return out


# --------------------------------------------------------------------------- loaders (I/O)

def lirf_frame(raw: pd.DataFrame, src: int) -> pd.DataFrame:
    """All LIRF DEP rows of `raw` (matched and unmatched): MVT_ID_mvt, FLIGHT_mvt, ADES_mvt, mvt (s), sched (s),
    aobt3_own (s; NaN = unmatched), CALLSIGN_flt, ADES_flt, LOBT_flt (s), src, y (TAXITIME_SEC_mvt, NaN if `raw`
    carries no such column — the 2026, label-free pass)."""
    d = raw[(raw.ADEP_mvt == "LIRF") & (raw.PHASE_mvt == "DEP")]
    y = d.TAXITIME_SEC_mvt.to_numpy(dtype="float64") if "TAXITIME_SEC_mvt" in d.columns else np.full(len(d), np.nan)
    return pd.DataFrame({
        "MVT_ID_mvt": d.MVT_ID_mvt.to_numpy(dtype="float64"),
        "FLIGHT_mvt": d.FLIGHT_mvt.to_numpy(),
        "ADES_mvt": d.ADES_mvt.to_numpy(),
        "mvt": ARC.secs(d.MVT_TIME_UTC_mvt),
        "sched": ARC.secs(d.SCHED_TIME_UTC_mvt),
        "aobt3_own": ARC.secs(d.AOBT_3_flt),
        "CALLSIGN_flt": d.CALLSIGN_flt.to_numpy(),
        "ADES_flt": d.ADES_flt.to_numpy(),
        "LOBT_flt": ARC.secs(d.LOBT_flt),
        "src": src,
        "y": y,
    })


def _empty_ruled() -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in RULED_DTYPES.items()})


def rule_rows(raw: pd.DataFrame, src: int) -> pd.DataFrame:
    """One row per ARC-N/XJ-ruled row (rule in {'ord', 'fill'}) among this single file's LIRF-unmatched, sp-scoped,
    ARC-unreached candidates. Columns: MVT_ID_mvt, FLIGHT_mvt, src, sp, sched, mvt, method ('ARC-N'|'XJ'), rule,
    rule_value, resolved_lobt, resolved_aobt3, y (NaN if `raw` carries no TAXITIME_SEC_mvt — the 2026 pass)."""
    L = lirf_frame(raw, src)
    sp = L.mvt.to_numpy() - L.sched.to_numpy()
    scope = in_scope(L.aobt3_own.to_numpy(), sp)
    cand = L[scope].reset_index(drop=True)
    cand_sp = sp[scope]
    if cand.empty:
        return _empty_ruled()

    dep_arc = ARC.dep_frame(raw, cand.MVT_ID_mvt.to_numpy())
    arr_pool = ARC.arr_frame(raw)
    j_arc = ARC.join_arr(dep_arc, arr_pool)
    not_reached = j_arc.arr_lobt.isna().to_numpy()
    cand, cand_sp = cand[not_reached].reset_index(drop=True), cand_sp[not_reached]
    if cand.empty:
        return _empty_ruled()

    dep_n = pd.DataFrame({"MVT_ID_mvt": cand.MVT_ID_mvt, "ADEP_mvt": "LIRF", "ADES_mvt": cand.ADES_mvt,
                          "FLIGHT_mvt": cand.FLIGHT_mvt, "mvt": cand.mvt})
    jN = arcn_join(dep_n, arr_pool)
    lobt = jN.arr_lobt.to_numpy(dtype="float64").copy()
    aobt3 = jN.arr_aobt3.to_numpy(dtype="float64").copy()
    resolved_n = np.isfinite(lobt) & np.isfinite(aobt3)
    method = np.where(resolved_n, "ARC-N", "")

    rest = ~resolved_n
    if rest.any():
        host = L[L.aobt3_own.notna()].rename(columns={"aobt3_own": "AOBT_3_flt"})
        xrows = cand[rest].copy()
        xrows["fn"] = norm_arr(xrows.FLIGHT_mvt.to_numpy())
        xrows["src"] = src
        xj = xj_join(xrows[["MVT_ID_mvt", "fn", "ADES_mvt", "mvt", "src"]],
                     host[["FLIGHT_mvt", "CALLSIGN_flt", "ADES_flt", "LOBT_flt", "AOBT_3_flt", "src"]])
        rest_idx = np.flatnonzero(rest)
        xl, xa = xj.xj_lobt.to_numpy(dtype="float64"), xj.xj_aobt3.to_numpy(dtype="float64")
        xj_ok = np.isfinite(xl) & np.isfinite(xa)
        lobt[rest_idx[xj_ok]] = xl[xj_ok]
        aobt3[rest_idx[xj_ok]] = xa[xj_ok]
        method_rest = np.where(xj_ok, "XJ", "")
        method = method.copy()
        method[rest_idx] = method_rest

    rule = ARC.sub_rule(cand_sp, cand.sched.to_numpy(), lobt, aobt3)
    ruled = np.isin(rule, ["ord", "fill"])
    if not ruled.any():
        return _empty_ruled()
    rv = rule_value(rule, cand.mvt.to_numpy(), aobt3, cand_sp)
    return pd.DataFrame({
        "MVT_ID_mvt": cand.MVT_ID_mvt.to_numpy()[ruled],
        "FLIGHT_mvt": cand.FLIGHT_mvt.to_numpy()[ruled],
        "src": src,
        "sp": cand_sp[ruled],
        "sched": cand.sched.to_numpy()[ruled],
        "mvt": cand.mvt.to_numpy()[ruled],
        "method": method[ruled],
        "rule": rule[ruled],
        "rule_value": rv[ruled],
        "resolved_lobt": lobt[ruled],
        "resolved_aobt3": aobt3[ruled],
        "y": cand.y.to_numpy()[ruled],
    })


def all_2025_rows() -> pd.DataFrame:
    """rule_rows() for each of the twelve 2025 training files in turn (src = calendar month 1..12), concatenated,
    with a `month` column equal to `src`. One file's raw DataFrame is loaded, used, and dropped before the next."""
    files = sorted(RAW.glob("training_2025-*.parquet"))
    if len(files) != 12:
        raise AssertionError(f"expected 12 training files, got {len(files)}")
    parts = []
    for i, f in enumerate(files, start=1):
        raw = pd.read_parquet(f, columns=NEED_COLS_2025)
        r = rule_rows(raw, src=i)
        r = r.assign(month=i)
        parts.append(r)
        del raw
    if not parts:
        return _empty_ruled().assign(month=pd.Series(dtype="int64"))
    return pd.concat(parts, ignore_index=True)


def rows_2026() -> pd.DataFrame:
    """rule_rows() on ranking.parquet alone (src=0) — label-free (NEED_COLS_2026 excludes TAXITIME_SEC_mvt), and
    matches ARC's own real 2026 production join exactly (single file, no month-boundary approximation)."""
    raw = pd.read_parquet(RANK, columns=NEED_COLS_2026)
    return rule_rows(raw, src=0)


# --------------------------------------------------------------------------- report (I/O)

def main() -> int:
    import hashlib
    import json

    if REPORT.exists():
        raise FileExistsError(REPORT)

    rows25 = all_2025_rows()
    cls25 = ARC.label_class(rows25.y.to_numpy(), rows25.sp.to_numpy())
    correct25 = call_correct(rows25.rule.to_numpy(), cls25)
    held_mask = rows25.month.isin(HELD_OUT_MONTHS).to_numpy()
    n_held, n_wrong = int(held_mask.sum()), int((held_mask & ~correct25).sum())
    v = verdict(n_held, n_wrong)
    n_disc, n_disc_correct = int((~held_mask).sum()), int((~held_mask & correct25).sum())

    by_rule = {}
    for ru in ("ord", "fill"):
        m = rows25.rule.to_numpy() == ru
        by_rule[ru] = dict(n=int(m.sum()), n_correct=int((m & correct25).sum()),
                            n_held=int((m & held_mask).sum()), n_held_correct=int((m & held_mask & correct25).sum()))

    r26 = rows_2026()
    v19_s = pd.read_parquet(V19).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    v19 = v19_s.reindex(r26.MVT_ID_mvt).to_numpy(dtype="float64")
    alt = stake_alt(r26.rule.to_numpy(), r26.mvt.to_numpy(), r26.resolved_aobt3.to_numpy(), r26.sp.to_numpy(), v19)
    expected = gain_sse(r26.rule_value.to_numpy(), v19, r26.rule_value.to_numpy())
    stake = gain_sse(alt, v19, r26.rule_value.to_numpy())

    res = dict(
        tag="ARN", prereg=str(PREREG.relative_to(ROOT)),
        prereg_sha256=hashlib.sha256(PREREG.read_bytes()).hexdigest(), NO_UPLOAD=True, BOARD_N=BOARD_N,
        limits=dict(twenty25_per_file_scoping=("ARC exclusion, ARC-N ARR pool and XJ host pool computed one "
                                                "training file at a time (memory budget); see module docstring")),
        all_months_2025=dict(
            n_ruled=int(len(rows25)), by_rule=by_rule,
            rows=[dict(MVT_ID_mvt=float(m), month=int(mo), src_held_out=bool(ho), FLIGHT_mvt=str(fl), method=str(me),
                       rule=str(ru), sp=float(sp), rule_value=float(rv), y=float(y), cls=str(cl), correct=bool(co))
                  for m, mo, ho, fl, me, ru, sp, rv, y, cl, co in zip(
                      rows25.MVT_ID_mvt, rows25.month, held_mask, rows25.FLIGHT_mvt, rows25.method, rows25.rule,
                      rows25.sp, rows25.rule_value, rows25.y, cls25, correct25)],
        ),
        held_out=dict(n_held=n_held, n_wrong=n_wrong, verdict=v),
        discovery_only=dict(n=n_disc, n_correct=n_disc_correct),
        pricing_2026=dict(
            n_ruled=int(len(r26)),
            rows=[dict(MVT_ID_mvt=float(m), FLIGHT_mvt=str(fl), rule=str(ru), method=str(me), rule_value=float(rv),
                       v19=float(vv), move=float(rv - vv), stake_alt=float(al), expected_row=float(ex),
                       stake_row=float(st))
                  for m, fl, ru, me, rv, vv, al, ex, st in zip(
                      r26.MVT_ID_mvt, r26.FLIGHT_mvt, r26.rule, r26.method, r26.rule_value, v19, alt, expected,
                      stake)],
            expected_value_total=float(expected.sum() / BOARD_N),
            stake_total=float(stake.sum() / BOARD_N),
        ),
    )
    REPORT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    print("all_months_2025 n_ruled", res["all_months_2025"]["n_ruled"], "by_rule", by_rule)
    print("held_out", res["held_out"], "discovery_only", res["discovery_only"])
    print("pricing_2026 n_ruled", res["pricing_2026"]["n_ruled"],
          "expected_value_total", round(res["pricing_2026"]["expected_value_total"], 2),
          "stake_total", round(res["pricing_2026"]["stake_total"], 2))
    for row in res["pricing_2026"]["rows"]:
        print(" ", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
