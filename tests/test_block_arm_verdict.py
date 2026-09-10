"""tools/block_arm_verdict.py: registered clause sets applied to a fold record.

Covers the arm FW clause set (plans/PREREG_fw_combined_2026_09_10.md H-FW1) and, because the tool
had no tests before it, the arm F set it copies: a record passing all four clauses reads
ESTABLISHED; one whose over10 interval straddles zero reads INCONCLUSIVE; one whose pooled interval
straddles zero reads NOT WORKING; an unknown amendment says so instead of borrowing a clause set.
"""
import json
import pathlib
import runpy
import sys

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "block_arm_verdict.py"


def _record(tmp_path, amendment, gain=1.7, ci=(1.4, 2.0), o10=(12.0, 10.8, 13.5), sd=0.145):
    band = {"n_rows": 26_000, "share_of_baseline_sse": 0.485,
            "pairs": {"treatment_vs_baseline": {"gain_s": o10[0], "ci95": [o10[1], o10[2]],
                                                 "excludes_zero": o10[1] > 0 or o10[2] < 0}}}
    d = {"mode": "queue_order_weather", "amendment": amendment, "git_sha": "abcdef0", "git_dirty": False,
         "wall_s": 7200, "peak_rss_gb": 5.1, "smoke": False,
         "arms": {"baseline": {"rmse": 224.258, "per_airport": {"LIRF": 384.0}},
                  "treatment": {"rmse": 224.258 - gain, "per_airport": {"LIRF": 380.0}}},
         "pairs": {"treatment_vs_baseline": {"gain_s": gain, "ci95": list(ci), "airports_improving": 8, "n_airports": 10}},
         "seed_sd": {"sd": sd}, "config": {"n_features": 93}, "treatment": {"n_ref": 25_000},
         "bands": {"over10": band}}
    p = tmp_path / f"r_{amendment}_{gain}.json"
    p.write_text(json.dumps(d))
    return p


def _run(monkeypatch, capsys, path) -> str:
    monkeypatch.setattr(sys, "argv", [str(TOOL), str(path)])
    runpy.run_path(str(TOOL), run_name="__main__")
    return capsys.readouterr().out


@pytest.mark.parametrize("amendment", ["22", "FW"])
def test_all_four_clauses_read_established(tmp_path, monkeypatch, capsys, amendment):
    out = _run(monkeypatch, capsys, _record(tmp_path, amendment))
    assert "VERDICT: ESTABLISHED" in out and out.count("PASS") == 4


def test_fw_is_labelled_not_decisional(tmp_path, monkeypatch, capsys):
    """Rehearsed 2026-09-10 RED: the FW entry removed from CLAUSES (the tool prints 'no registered
    clause set' and no verdict)."""
    out = _run(monkeypatch, capsys, _record(tmp_path, "FW"))
    assert "amendment FW" in out and "VERDICT:" in out


def test_over10_straddling_zero_is_inconclusive(tmp_path, monkeypatch, capsys):
    out = _run(monkeypatch, capsys, _record(tmp_path, "FW", o10=(-0.5, -1.6, 0.4)))
    assert "VERDICT: INCONCLUSIVE" in out


def test_pooled_interval_straddling_zero_is_not_working(tmp_path, monkeypatch, capsys):
    out = _run(monkeypatch, capsys, _record(tmp_path, "FW", gain=0.1, ci=(-0.2, 0.4)))
    assert "VERDICT: NOT WORKING" in out


def test_point_bar_and_seed_sd_clauses_are_separate(tmp_path, monkeypatch, capsys):
    """A +0.7 s gain with an interval excluding zero passes (i) and (iii) but fails (ii) the 1.0 s
    bar - arm W's exact shape in RESULT 22."""
    out = _run(monkeypatch, capsys, _record(tmp_path, "24", gain=0.725, ci=(0.495, 0.947), o10=(-0.5, -1.6, 0.4)))
    lines = [line for line in out.splitlines() if line.strip().startswith("(")]
    assert "PASS" in lines[0] and "fail" in lines[1] and "PASS" in lines[2] and "fail" in lines[3]
    assert "VERDICT: INCONCLUSIVE" in out


def test_an_unknown_amendment_borrows_no_clause_set(tmp_path, monkeypatch, capsys):
    out = _run(monkeypatch, capsys, _record(tmp_path, "99"))
    assert "no registered clause set for amendment '99'" in out and "VERDICT" not in out
