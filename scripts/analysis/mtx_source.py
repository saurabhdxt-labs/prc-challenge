"""Parsers for the PUBLISHED ansperformance.eu (EUROCONTROL AIU) monthly taxi-out figures — arm MTX
(plans/PREREG_mtx_monthly_calibration_2026_09_28.md). Owner ruling 2026-09-28 allows the published MONTHLY
airport / per-runway averages; per-flight gate times stay refused.

Sources (raw copies + sha256 in data/mtx/raw/MANIFEST.json):
  * taxi_out_additional_time_<year>.csv  (https://www.eurocontrol.int/performance/data/download/csv/...)
    one row per airport-month: TF (all departures), VALID_FL (after the data filters), NO_REF (valid flights
    whose stand-runway combo has no reference), TOTAL_REF_NB_FL = VALID_FL - NO_REF (the flights averaged),
    TOTAL_REF_TIME_MIN / TOTAL_ADD_TIME_MIN (sums, minutes). Mean actual taxi-out (AOBT->ATOT) over the averaged
    flights = (REF + ADD) / TOTAL_REF_NB_FL. Month = local time of take-off (methodology ATXOT, Mar 2023).
  * the AIU airport dashboard pages https://ansperformance.eu/dashboard/stakeholder/airport/db/<ICAO>.html:
    plotly widgets "monthly-taxi-out-time-source-apdf" (airport-month mean reference + additional, min/dep) and
    "monthly-taxi-out-time-per-runway-source-apdf" (the same per departure runway; no counts published).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam",
      "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LIRF": "Europe/Rome",
      "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}
_CSV_COLS = ("YEAR", "MONTH_NUM", "APT_ICAO", "TF", "VALID_FL", "NO_REF", "TOTAL_REF_NB_FL",
             "TOTAL_REF_TIME_MIN", "TOTAL_ADD_TIME_MIN")
_MONTHS = {m: i + 1 for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                                           "Nov", "Dec"))}


def load_monthly_csv(path, airports=AIRPORTS) -> pd.DataFrame:
    """One row per (ap, year, month) of `airports`: TF, VALID, NR (=NO_REF), P (=TOTAL_REF_NB_FL), INV (=TF-VALID),
    pub_mean_s (mean actual taxi-out over the P flights, seconds), ref_mean_s. Fails loud on a schema or count
    inconsistency."""
    raw = pd.read_csv(path)
    miss = [c for c in _CSV_COLS if c not in raw.columns]
    if miss:
        raise ValueError(f"{path}: missing columns {miss}")
    d = raw[raw.APT_ICAO.isin(airports)].copy()
    if d.duplicated(["APT_ICAO", "YEAR", "MONTH_NUM"]).any():
        raise ValueError(f"{path}: duplicate airport-month rows")
    if (d.TOTAL_REF_NB_FL <= 0).any():
        raise ValueError(f"{path}: TOTAL_REF_NB_FL <= 0 for {d[d.TOTAL_REF_NB_FL <= 0].APT_ICAO.tolist()}")
    if not np.array_equal(d.TOTAL_REF_NB_FL.to_numpy(float), (d.VALID_FL - d.NO_REF).to_numpy(float)):
        raise ValueError(f"{path}: TOTAL_REF_NB_FL != VALID_FL - NO_REF")
    out = pd.DataFrame({
        "ap": d.APT_ICAO.to_numpy(), "year": d.YEAR.astype(int).to_numpy(), "month": d.MONTH_NUM.astype(int).to_numpy(),
        "TF": d.TF.to_numpy(float), "VALID": d.VALID_FL.to_numpy(float), "NR": d.NO_REF.to_numpy(float),
        "P": d.TOTAL_REF_NB_FL.to_numpy(float), "INV": (d.TF - d.VALID_FL).to_numpy(float),
        "pub_mean_s": ((d.TOTAL_REF_TIME_MIN + d.TOTAL_ADD_TIME_MIN) / d.TOTAL_REF_NB_FL * 60.0).to_numpy(float),
        "ref_mean_s": (d.TOTAL_REF_TIME_MIN / d.TOTAL_REF_NB_FL * 60.0).to_numpy(float),
    })
    return out.sort_values(["ap", "year", "month"]).reset_index(drop=True)


def _widget_json(html: str, section: str) -> dict:
    i = html.find(f'id="{section}"')
    if i < 0:
        raise ValueError(f"dashboard section {section!r} not found")
    j = html.find('<script type="application/json"', i)
    if j < 0:
        raise ValueError(f"dashboard section {section!r} has no widget data")
    k = html.find(">", j) + 1
    return json.loads(html[k:html.find("</script>", k)])


def _pairs(traces):
    """Group (reference, additional) traces by subplot axis, in order; refuse anything unpaired."""
    by_axis: dict = {}
    for tr in traces:
        by_axis.setdefault(tr.get("xaxis", "x"), {}).setdefault(tr["name"], []).append(tr)
    for ax, d in by_axis.items():
        refs, adds = d.get("Avg. Reference time", []), d.get("Avg. Additional time", [])
        if len(refs) != len(adds):
            raise ValueError(f"axis {ax}: unpaired reference/additional traces")
        for r, a in zip(refs, adds):
            if list(r["x"]) != list(a["x"]) or list(r["customdata"]) != list(a["customdata"]):
                raise ValueError(f"axis {ax}: reference and additional traces are misaligned")
            yield ax, r, a


def parse_dashboard_runway(html: str, ap: str) -> pd.DataFrame:
    """Per departure runway monthly means: ap, year, month, runway, ref_mean_s, add_mean_s, pub_mean_s (seconds).
    The month comes from each subplot's axis title (Jan..Dec)."""
    w = _widget_json(html, "monthly-taxi-out-time-per-runway-source-apdf")["x"]
    rows = []
    for ax, r, a in _pairs(w["data"]):
        key = "xaxis" if ax == "x" else "xaxis" + ax[1:]
        title = w["layout"].get(key, {}).get("title")
        title = title.get("text") if isinstance(title, dict) else title
        if title not in _MONTHS:
            raise ValueError(f"axis {ax}: title {title!r} is not a month")
        for rw, yr, ref, add in zip(r["x"], r["customdata"], r["y"], a["y"]):
            rows.append((ap, int(yr), _MONTHS[title], str(rw), float(ref) * 60.0, float(add) * 60.0))
    d = pd.DataFrame(rows, columns=["ap", "year", "month", "runway", "ref_mean_s", "add_mean_s"])
    d["pub_mean_s"] = d.ref_mean_s + d.add_mean_s
    if d.duplicated(["year", "month", "runway"]).any():
        raise ValueError(f"{ap}: duplicate runway-month rows")
    return d


def parse_dashboard_monthly(html: str, ap: str) -> pd.DataFrame:
    """Airport-month means from the dashboard: ap, year, month, ref_mean_s, add_mean_s, pub_mean_s (seconds)."""
    w = _widget_json(html, "monthly-taxi-out-time-source-apdf")["x"]
    rows = []
    for _, r, a in _pairs(w["data"]):
        for mo, yr, ref, add in zip(r["x"], r["customdata"], r["y"], a["y"]):
            rows.append((ap, int(yr), int(mo), float(ref) * 60.0, float(add) * 60.0))
    d = pd.DataFrame(rows, columns=["ap", "year", "month", "ref_mean_s", "add_mean_s"])
    d["pub_mean_s"] = d.ref_mean_s + d.add_mean_s
    return d


def local_year_month(t_utc: pd.Series, ap: pd.Series):
    """(year, month) of each UTC timestamp in its departure airport's local time (the published month basis)."""
    ap = pd.Series(np.asarray(ap), index=t_utc.index)
    bad = sorted(set(ap.unique()) - set(TZ))
    if bad:
        raise ValueError(f"unknown airport(s) {bad}")
    t = pd.to_datetime(t_utc, utc=True)
    y = pd.Series(0, index=t.index, dtype=int)
    m = pd.Series(0, index=t.index, dtype=int)
    for a in ap.unique():
        sel = (ap == a).to_numpy()
        tl = t[sel].dt.tz_convert(TZ[a])
        y[sel] = tl.dt.year.to_numpy()
        m[sel] = tl.dt.month.to_numpy()
    return y, m
