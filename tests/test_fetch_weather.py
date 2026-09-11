"""scripts/fetch_weather.py — the archiver of the frozen METAR copy. No test touches the network:
`fetch` is exercised against a patched `urlopen`, and `main` against a patched `fetch`.

The file had no tests until the 2026-09-11 post-mortem (bug class BC-3): its `check` validated the
shape of a month and passed the months whose p01i is a constant placeholder. The information
report added then is pinned here, with the rest of the script's contract.

Second rehearsal round, finished 2026-09-11 02:02 EDT (from `date`), all RED: a month on disk refetched
[main_writes_atomically]; the retry loop reduced to one attempt [fetch_retries]; the check skipped
before the write [main_leaves_no_file].
Third round, finished 2026-09-11 02:20 EDT (from `date`), RED: the information report moved back after the
rename [a_month_the_parser_cannot_read].
"""
from __future__ import annotations

import importlib.util
import pathlib
import urllib.error

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fetch_weather", ROOT / "scripts" / "fetch_weather.py")
fw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw)

HEAD = "station,valid,tmpf,dwpf,sknt,gust,vsby,p01i,wxcodes,metar"


def month_text(station="EDDM", n=400, p01i=lambda i: "0.00", codes=lambda i: "-RA" if i % 10 == 0 else "M") -> str:
    rows = [f"{station},2025-01-{1 + i // 48:02d} {(i % 48) // 2:02d}:{30 * (i % 2):02d},{30 + i % 7},{28 + i % 5},"
            f"{5 + i % 9},M,6.21,{p01i(i)},{codes(i)},x" for i in range(n)]
    return "\n".join([HEAD] + rows) + "\n"


def test_month_url_spans_exactly_one_month_and_asks_for_both_report_types():
    """December wraps to January 1 of the next year; routine and special reports are both asked for;
    the raw fields only. Fails if the end date were the last day (a day lost) or SPECI dropped."""
    u = fw.month_url("EGLL", 2025, 12)
    assert "station=EGLL" in u and "year1=2025&month1=12&day1=1" in u and "year2=2026&month2=1&day2=1" in u
    assert "report_type=3" in u and "report_type=4" in u and "tz=UTC" in u
    assert "data=tmpf%2Cdwpf%2Csknt%2Cgust%2Cvsby%2Cp01i%2Cwxcodes%2Cmetar" in u
    assert "year2=2025&month2=7&day2=1" in fw.month_url("EGLL", 2025, 6)


def test_check_accepts_a_complete_month_and_refuses_broken_ones():
    """A header, the station on every row and at least 336 observations; each failure names the
    month and the defect."""
    assert fw.check(month_text(), "EDDM", 2025, 1) == 400
    with pytest.raises(ValueError, match="EDDM 2025-01: no header"):
        fw.check("oops\n", "EDDM", 2025, 1)
    with pytest.raises(ValueError, match="no header, got '<empty>'"):
        fw.check("", "EDDM", 2025, 1)
    with pytest.raises(ValueError, match="a row is not this station"):
        fw.check(month_text(station="EDDF"), "EDDM", 2025, 1)
    with pytest.raises(ValueError, match="only 100 observations, expected >= 336"):
        fw.check(month_text(n=100), "EDDM", 2025, 1)


def test_information_report_names_a_constant_p01i_and_the_rain_it_contradicts():
    """THE fetch-side guard. A European month as the archive serves it: p01i "0.00" on all 400
    rows while 40 report rain -> one line naming p01i with that evidence (and the gust group, never
    reported). A month whose p01i varies is not named. Rehearsed 2026-09-11 between 01:21:39 and 01:48:18 EDT (two `date` reads): RED when the
    report returns [] (the pre-fix behaviour: check alone). Also rehearsed then, each RED: December
    not wrapping to the next year [month_url]; the 336-row floor dropped [check]."""
    lines = fw.information_report(month_text())
    p = [ln for ln in lines if ln.startswith("p01i:")]
    assert p == ["p01i: 1 distinct value(s) over 400 reported; the weather group reports precipitation on 40 rows"]
    assert any(ln.startswith("gust: 0 distinct") for ln in lines)
    varied = fw.information_report(month_text(p01i=lambda i: "0.00" if i % 3 else "0.02"))
    assert not any(ln.startswith("p01i:") for ln in varied)


def test_main_writes_atomically_reports_and_skips_months_on_disk(tmp_path, monkeypatch, capsys):
    """--smoke fetches one month into --out-dir through a .part rename (no .part left), prints the
    information report, skips the month on a second run, and fetches it again with --refetch."""
    calls = []
    monkeypatch.setattr(fw, "fetch", lambda url: calls.append(url) or month_text(station="EHAM"))
    assert fw.main(["--smoke", "--out-dir", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["EHAM_2025-01.csv"]
    out = capsys.readouterr().out
    assert "EHAM_2025-01.csv     400 observations" in out and "UNINFORMATIVE p01i: 1 distinct" in out
    assert fw.main(["--smoke", "--out-dir", str(tmp_path)]) == 0 and len(calls) == 1
    assert "0 fetched, 1 already on disk" in capsys.readouterr().out
    assert fw.main(["--smoke", "--out-dir", str(tmp_path), "--refetch"]) == 0 and len(calls) == 2


def test_main_leaves_no_file_when_a_month_fails_its_check(tmp_path, monkeypatch):
    """A truncated response raises before anything takes the real name — or the .part name."""
    monkeypatch.setattr(fw, "fetch", lambda url: month_text(station="EHAM", n=50))
    with pytest.raises(ValueError, match="only 50 observations"):
        fw.main(["--smoke", "--out-dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


class _Resp:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.body


def test_fetch_retries_transient_failures_then_gives_up(monkeypatch):
    """Two transient failures then a success returns the body; four failures raise naming the
    attempts and chaining the last error. No real sleep (BACKOFF_S = 0)."""
    monkeypatch.setattr(fw, "BACKOFF_S", 0.0)
    seq = [urllib.error.URLError("down"), TimeoutError("slow"), _Resp(b"station,valid\n")]

    def fake(url, timeout):
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(fw.urllib.request, "urlopen", fake)
    assert fw.fetch("http://x") == "station,valid\n"
    monkeypatch.setattr(fw.urllib.request, "urlopen", lambda url, timeout: (_ for _ in ()).throw(OSError("reset")))
    with pytest.raises(RuntimeError, match="fetch failed after 4 attempts: reset") as e:
        fw.fetch("http://x")
    assert isinstance(e.value.__cause__, OSError)


def test_an_interrupted_write_never_leaves_a_file_under_the_real_name(tmp_path, monkeypatch):
    """The atomicity claim, forced: the write dies halfway. Only the .part file may exist, so a later
    build cannot read a truncated month as complete. Rehearsed 2026-09-11: RED when the .part
    indirection is removed (the half-written file takes the real name)."""
    monkeypatch.setattr(fw, "fetch", lambda url: month_text(station="EHAM"))
    real_write = pathlib.Path.write_text

    def dies_halfway(self, data, *a, **k):
        real_write(self, data[: len(data) // 2], *a, **k)
        raise OSError("disk full")
    monkeypatch.setattr(pathlib.Path, "write_text", dies_halfway)
    with pytest.raises(OSError, match="disk full"):
        fw.main(["--smoke", "--out-dir", str(tmp_path)])
    assert not (tmp_path / "EHAM_2025-01.csv").exists()
    assert (tmp_path / "EHAM_2025-01.part").exists()


def test_a_month_the_parser_cannot_read_never_takes_its_name(tmp_path, monkeypatch):
    """The information report runs BEFORE the rename: a month with a token the grammar refuses
    raises and leaves nothing behind, so a re-run fetches it again instead of skipping a month it
    believes complete (review 2026-09-11: the report used to run after the rename)."""
    monkeypatch.setattr(fw, "fetch", lambda url: month_text(station="EHAM", codes=lambda i: "XX" if i == 5 else "M"))
    with pytest.raises(ValueError, match="unparseable present-weather token 'XX'"):
        fw.main(["--smoke", "--out-dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []
