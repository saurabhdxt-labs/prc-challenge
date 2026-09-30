"""Tests for scripts/analysis/mtx_source.py — parsers for the PUBLISHED ansperformance.eu monthly taxi-out figures
(arm MTX, plans/PREREG_mtx_monthly_calibration_2026_09_28.md). Written before the module (red first)."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("mtx_source", ROOT / "scripts" / "analysis" / "mtx_source.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mtx_source"] = mod
    spec.loader.exec_module(mod)
    return mod


ms = _load()

HEADER = ("YEAR,MONTH_NUM,MONTH_MON,APT_ICAO,APT_NAME,STATE_NAME,TF,VALID_FL,NO_REF,TOTAL_REF_NB_FL,"
          "TOTAL_REF_TIME_MIN,TOTAL_ADD_TIME_MIN,COMMENT\n")


def _csv(tmp_path, rows):
    p = tmp_path / "t.csv"
    p.write_text(HEADER + "".join(rows))
    return p


def test_load_monthly_csv_computes_the_published_mean_in_seconds_and_the_counts(tmp_path):
    """Kills: mean over VALID_FL instead of TOTAL_REF_NB_FL; minutes not converted; INV = TF-VALID wrong sign;
    other airports not filtered out."""
    p = _csv(tmp_path, ["2026,01,JAN,EDDF,Frankfurt,Germany,100,90,10,80,800.0,160.0,\n",
                        "2026,01,JAN,LOWW,Vienna,Austria,50,50,0,50,500.0,0.0,\n"])
    d = ms.load_monthly_csv(p)
    assert d.ap.tolist() == ["EDDF"]
    r = d.iloc[0]
    assert (r.year, r.month, r.TF, r.P, r.NR, r.INV) == (2026, 1, 100, 80, 10, 10)
    assert r.pub_mean_s == pytest.approx((800.0 + 160.0) / 80 * 60, abs=1e-9)
    assert r.ref_mean_s == pytest.approx(800.0 / 80 * 60, abs=1e-9)


def test_load_monthly_csv_refuses_inconsistent_counts(tmp_path):
    """TOTAL_REF_NB_FL must equal VALID_FL - NO_REF: a changed column meaning must fail loud, not shift targets."""
    p = _csv(tmp_path, ["2026,01,JAN,EDDF,Frankfurt,Germany,100,90,10,70,800.0,160.0,\n"])
    with pytest.raises(ValueError, match="TOTAL_REF_NB_FL != VALID_FL - NO_REF"):
        ms.load_monthly_csv(p)


def test_load_monthly_csv_refuses_duplicates_and_missing_columns(tmp_path):
    p = _csv(tmp_path, ["2026,01,JAN,EDDF,F,G,100,90,10,80,800.0,160.0,\n"] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        ms.load_monthly_csv(p)
    q = tmp_path / "bad.csv"
    q.write_text("YEAR,MONTH_NUM,APT_ICAO\n2026,1,EDDF\n")
    with pytest.raises(ValueError, match="missing columns"):
        ms.load_monthly_csv(q)


def test_load_monthly_csv_refuses_nonpositive_population(tmp_path):
    p = _csv(tmp_path, ["2026,01,JAN,EDDF,F,G,100,10,10,0,0.0,0.0,\n"])
    with pytest.raises(ValueError, match="TOTAL_REF_NB_FL <= 0"):
        ms.load_monthly_csv(p)


def _widget(section, data, layout):
    blob = json.dumps({"x": {"data": data, "layout": layout}})
    return (f'<div id="{section}" class="section level3"><h3>x</h3>'
            f'<div class="plotly html-widget"></div><script type="application/json" data-for="w">{blob}</script>')


def test_parse_dashboard_runway_reads_month_from_the_subplot_axis_title():
    """Kills: month taken from trace order instead of the axis title; ref/add swapped; minutes not converted."""
    data = [
        {"name": "Avg. Reference time", "xaxis": "x2", "x": ["09L", "09L"], "y": [10.0, 11.0], "customdata": ["2025", "2026"]},
        {"name": "Avg. Additional time", "xaxis": "x2", "x": ["09L", "09L"], "y": [2.0, 3.0], "customdata": ["2025", "2026"]},
        {"name": "Avg. Reference time", "xaxis": "x", "x": ["27R"], "y": [12.0], "customdata": ["2026"]},
        {"name": "Avg. Additional time", "xaxis": "x", "x": ["27R"], "y": [1.5], "customdata": ["2026"]},
    ]
    layout = {"xaxis": {"title": "Jul"}, "xaxis2": {"title": "Feb"}}
    html = "<html>" + _widget("monthly-taxi-out-time-per-runway-source-apdf", data, layout) + "</html>"
    d = ms.parse_dashboard_runway(html, "LFPG").sort_values(["year", "month", "runway"]).reset_index(drop=True)
    assert d[["year", "month", "runway"]].values.tolist() == [[2025, 2, "09L"], [2026, 2, "09L"], [2026, 7, "27R"]]
    assert d.ref_mean_s.tolist() == [600.0, 660.0, 720.0]
    assert d.pub_mean_s.tolist() == [720.0, 840.0, 810.0]
    assert (d.ap == "LFPG").all()


def test_parse_dashboard_runway_refuses_a_missing_section_or_misaligned_traces():
    with pytest.raises(ValueError, match="section"):
        ms.parse_dashboard_runway("<html></html>", "LFPG")
    data = [{"name": "Avg. Reference time", "xaxis": "x", "x": ["09L"], "y": [10.0], "customdata": ["2026"]},
            {"name": "Avg. Additional time", "xaxis": "x", "x": ["27R"], "y": [2.0], "customdata": ["2026"]}]
    html = _widget("monthly-taxi-out-time-per-runway-source-apdf", data, {"xaxis": {"title": "Jan"}})
    with pytest.raises(ValueError, match="misaligned"):
        ms.parse_dashboard_runway(html, "LFPG")


def test_parse_dashboard_monthly_reads_airport_month_means():
    data = [{"name": "Avg. Reference time", "xaxis": "x", "x": [1, 7], "y": [10.0, 11.0], "customdata": [2026, 2026]},
            {"name": "Avg. Additional time", "xaxis": "x", "x": [1, 7], "y": [2.0, 1.0], "customdata": [2026, 2026]}]
    html = _widget("monthly-taxi-out-time-source-apdf", data, {"xaxis": {"title": ""}})
    d = ms.parse_dashboard_monthly(html, "EDDF").sort_values("month")
    assert d.month.tolist() == [1, 7] and d.pub_mean_s.tolist() == [720.0, 720.0]


def test_local_month_uses_each_airports_timezone():
    """LTFM (UTC+3): 2025-12-31 22:30 UTC is January local; EGLL (UTC+0 in winter) stays December."""
    t = pd.Series(pd.to_datetime(["2025-12-31 22:30", "2025-12-31 22:30"], utc=True))
    ap = pd.Series(["LTFM", "EGLL"])
    y, m = ms.local_year_month(t, ap)
    assert y.tolist() == [2026, 2025] and m.tolist() == [1, 12]
    with pytest.raises(ValueError, match="unknown airport"):
        ms.local_year_month(t, pd.Series(["LOWW", "EGLL"]))


@pytest.mark.skipif(not (ROOT / "data/mtx/raw/taxi_out_additional_time_2025.csv").is_file(), reason="raw file absent")
def test_real_csv_and_dashboard_agree_on_the_airport_month_mean():
    """Two published products of the same source must agree (to 1e-6 min) for every one of our airport-months."""
    for yr in (2025, 2026):
        c = ms.load_monthly_csv(ROOT / f"data/mtx/raw/taxi_out_additional_time_{yr}.csv")
        for ap in ms.AIRPORTS:
            html = (ROOT / f"data/mtx/raw/dashboard/{ap}.html").read_text()
            d = ms.parse_dashboard_monthly(html, ap)
            j = c[c.ap == ap].merge(d[d.year == yr], on=["year", "month"], suffixes=("", "_db"))
            assert len(j) == (c.ap == ap).sum() > 0
            assert np.allclose(j.pub_mean_s, j.pub_mean_s_db, atol=1e-4)
