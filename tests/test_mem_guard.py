"""tools/mem_guard.py — pause OUR long run on memory pressure, resume on recovery (owner ruling 2026-09-24).

Written before the module (seen RED: ModuleNotFoundError). Mutation rehearsal at write time (copies, then restored):
  M1 pause when swap_free >= threshold (flip <)           -> test_pauses_when_swap_free_drops_below_threshold RED
  M2 resume without the minimum pause time                -> test_resume_needs_min_pause_and_both_recovered RED
  M3 resume on swap alone (drop the pressure condition)   -> test_resume_needs_min_pause_and_both_recovered RED
  M4 accept a Phantom command as a target                 -> test_refuses_phantom_targets RED
  M5 parse 'free = 1,234.56M' as 1                        -> test_parse_swapusage_reads_free_mb RED

--pause-on-phantom (owner ruling 2026-09-28: "keep watching the memory when phantom starts just pause and resume").
Mutation rehearsal for that addition is logged separately in logs/cml_scratch/mutation_memguard.txt; RED output for
these new tests (seen before phantom_heavy_alive/decide_with_phantom/--pause-on-phantom existed) is in
logs/cml_scratch/red_log_memguard.txt.

BC-37 (2026-10-01, owner decision "1"): a phantom-caused pause resumes without the memory RESUME hysteresis. 6 tests seen
red first (HOLD on the real stuck case; next_cause / resume_label absent). Mutation rehearsal on copies (scratch dir, live
file untouched): 12 of 12 RED — phantom branch off, <= for < on swap, free-% condition dropped, hold-below-pause dropped,
default cause flipped, PAUSE cause constant, escalation dropped, cause not sticky, RESUME not clearing, cooldown off,
labels swapped, HOLD reason blank.
BC-37 part 2 (2026-10-01 22:1x, owner decision "1"): the memory resume rule is free RAM >= 30 % with swap not below the
pause line. 2 tests seen red first (HOLD at 684 MB / 65 %; constants). Rehearsal on copies: 8 of 8 RED — swap line back
at 1,000, RAM line 15 and 31, swap or RAM condition dropped, > for >= on swap, minimum pause dropped, cause ignored.
BC-43 (2026-10-04, tooling audit T1/T3): the loop is run() with an injected OS layer. With the loop extracted faithfully
(no release), the three regression tests were seen red for the real reason (last signal SIGSTOP; no SIGCONT after the
target ended) and the PermissionError test raised. Rehearsal on copies after the fix: 6 / 6 RED — MG1 no release in
finally, MG2 stopped pids not recorded at pause, MG3 not cleared on resume (double release), MG4 PermissionError read
as gone, MG5 the signal handler returns instead of exiting, MG6 --release signals nothing.
BC-43 part 2 (2026-10-04, Protocol B review): a resume also continues stopped processes that have left the tree;
signal_pids never signals a Phantom process and skips pids it may not signal (a recorded pid can be reused); a signal
exit is logged EXIT; --restart; the BC-43 tests run in both modes. Rehearsal on copies: 10 / 10 RED — MG7 no orphan
resume, MG8 no Phantom check, MG9 PermissionError not skipped, MG10 a signal exit logged as ERROR, MG11 --restart ignored,
MG12 the exit release walks the current tree, MG13 unsignalled pids not named, MG14 no handlers installed, MG15 the resume
does not clear the record, MG16 SIGHUP not handled.
BC-43 part 3 (Protocol B review 2026-10-04 02:50): the release holds further termination signals, a pid whose identity
`ps` cannot read still gets SIGCONT, and --release also continues members of process groups its tree leads.
Rehearsal 5 / 5 RED — MG17 no hold, MG18 SIGCONT withheld on an identity error, MG19 group members left out, MG20 group
members not identity-checked, MG21 --release back on the tree walk (survived until the CLI test held a reparented
member).
"""
from __future__ import annotations

import argparse
import pathlib
import signal
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import mem_guard as G  # noqa: E402


def test_parse_swapusage_reads_free_mb():
    s = "total = 14336.00M  used = 12857.62M  free = 1478.38M  (encrypted)"
    assert G.parse_swap_free_mb(s) == pytest.approx(1478.38)
    assert G.parse_swap_free_mb("total = 20,480.00M  used = 18,000.00M  free = 1,234.56M") == pytest.approx(1234.56)
    with pytest.raises(ValueError, match="cannot parse vm.swapusage"):
        G.parse_swap_free_mb("garbage")


def test_parse_memory_pressure_reads_free_percentage():
    out = "The system has 34359738368 (2097152 pages with a page size of 16384).\n...\nSystem-wide memory free percentage: 12%\n"
    assert G.parse_pressure_free_pct(out) == 12
    with pytest.raises(ValueError, match="cannot parse memory_pressure"):
        G.parse_pressure_free_pct("no percentage here")


def test_pauses_when_swap_free_drops_below_threshold():
    assert G.decide(paused=False, swap_free_mb=300, free_pct=40, paused_for_s=0) == "PAUSE"
    assert G.decide(paused=False, swap_free_mb=G.PAUSE_SWAP_MB, free_pct=40, paused_for_s=0) == "RUN"


def test_pauses_when_pressure_free_drops_below_threshold():
    assert G.decide(paused=False, swap_free_mb=5000, free_pct=G.PAUSE_FREE_PCT - 1, paused_for_s=0) == "PAUSE"
    assert G.decide(paused=False, swap_free_mb=5000, free_pct=G.PAUSE_FREE_PCT, paused_for_s=0) == "RUN"


def test_resume_needs_min_pause_and_both_recovered():
    ok_swap, ok_pct = G.RESUME_SWAP_MB, G.RESUME_FREE_PCT
    assert G.decide(paused=True, swap_free_mb=ok_swap, free_pct=ok_pct, paused_for_s=G.MIN_PAUSE_S) == "RESUME"
    assert G.decide(paused=True, swap_free_mb=ok_swap, free_pct=ok_pct, paused_for_s=G.MIN_PAUSE_S - 1) == "HOLD"
    assert G.decide(paused=True, swap_free_mb=ok_swap - 1, free_pct=ok_pct, paused_for_s=G.MIN_PAUSE_S) == "HOLD"
    assert G.decide(paused=True, swap_free_mb=ok_swap, free_pct=ok_pct - 1, paused_for_s=G.MIN_PAUSE_S) == "HOLD"


def test_hysteresis_is_on_free_ram_and_the_swap_resume_line_is_the_pause_line():
    """Owner decision 2026-10-01 22:1x ("1", BC-37 second part): free swap is not a recovery signal on macOS — a paused
    run's own pages sit in swap until it runs again, and the swap files are resized under it — so the resume line for
    swap is the pause line (350 MB) and the hysteresis lives on free RAM: pause < 8 %, resume >= 30 %."""
    assert G.RESUME_SWAP_MB == G.PAUSE_SWAP_MB == 350.0
    assert (G.PAUSE_FREE_PCT, G.RESUME_FREE_PCT) == (8, 30)


def test_regression_bug_37_a_memory_pause_resumes_on_free_ram_not_on_free_swap():
    """2026-10-01 21:59–22:1x: a memory pause (swap free 309 MB) outlived its cause: the other job ended, RAM free went
    to 65 %, but swap free sat at 684 MB (< the old 1,000 MB line) because the paused fit's own pages stayed in swap.
    Seen red: HOLD at these inputs. Mutation: put the swap resume line back at 1,000 MB -> red."""
    assert G.decide(paused=True, swap_free_mb=684, free_pct=65, paused_for_s=602) == "RESUME"
    assert G.decide_with_phantom(True, 684, 65, 602, phantom_names=[], phantom_quiet_for_s=9999,
                                 memory_caused=True) == ("RESUME", "")
    # still held: RAM not yet at the resume line, swap below the pause line, or the minimum pause not served
    assert G.decide(paused=True, swap_free_mb=5000, free_pct=29, paused_for_s=602) == "HOLD"
    assert G.decide(paused=True, swap_free_mb=5000, free_pct=30, paused_for_s=602) == "RESUME"
    assert G.decide(paused=True, swap_free_mb=349, free_pct=65, paused_for_s=602) == "HOLD"
    assert G.decide(paused=True, swap_free_mb=350, free_pct=65, paused_for_s=602) == "RESUME"
    assert G.decide(paused=True, swap_free_mb=684, free_pct=65, paused_for_s=G.MIN_PAUSE_S - 1) == "HOLD"


def test_refuses_phantom_targets():
    with pytest.raises(ValueError, match="refusing to signal a Phantom process"):
        G.check_target("/Users/x/Projects/Phantom/.venv/bin/python scripts/live_ingest/aggregate_cell_hour.py")
    with pytest.raises(ValueError, match="refusing to signal a Phantom process"):
        G.check_target("python r17_shadow_day_eval.py")
    G.check_target("/opt/homebrew/bin/python3.11 -B scripts/analysis/nsc_fold.py")  # ours: no raise


def test_tree_collects_descendants_from_a_parent_map():
    parents = {10: 1, 11: 10, 12: 11, 13: 10, 20: 1}
    assert G.tree(10, parents) == [10, 11, 13, 12] or sorted(G.tree(10, parents)) == [10, 11, 12, 13]
    assert 20 not in G.tree(10, parents)


# --- --pause-on-phantom -------------------------------------------------------------------------------------------

def test_phantom_heavy_alive_detects_a_heavy_job():
    ps = "111 /opt/homebrew/bin/python3.11 scripts/live_ingest/aggregate_cell_hour.py --day 2026-09-28\n"
    assert G.phantom_heavy_alive(ps) == ["aggregate_cell_hour"]


def test_phantom_heavy_alive_absent_returns_empty():
    ps = "111 /opt/homebrew/bin/python3.11 scripts/analysis/nsc_fold.py\n222 -zsh\n"
    assert G.phantom_heavy_alive(ps) == []


def test_phantom_heavy_alive_ignores_bash_lc_wrapper_line():
    ps = "111 /bin/bash -lc /Users/x/Projects/Phantom/run_track_record.sh\n"
    assert G.phantom_heavy_alive(ps) == []


def test_phantom_heavy_alive_ignores_our_own_pid():
    # BC-29: fixture updated 2026-09-28 -- the old fixture ("verify_cron_lookalike", no interpreter) only
    # matched via the naive substring scan this bug class removed; it no longer exercises a real job line.
    ps = "111 /opt/homebrew/bin/python3.11 scripts/live_ingest/verify_cron.py --pid 999\n"
    assert G.phantom_heavy_alive(ps, ignore_pids=frozenset({111})) == []
    assert G.phantom_heavy_alive(ps, ignore_pids=frozenset()) == ["verify_cron"]


def test_phantom_heavy_alive_skips_malformed_and_blank_lines():
    # BC-29: fixture updated 2026-09-28 -- a bare "refresh_recall.py" with no interpreter token is no longer
    # a valid job line under the stricter match (it must be run BY a Python interpreter); give it one.
    ps = "\n   \nnotapid onlyonefield\nabc some command refresh_recall\n333 python refresh_recall.py\n"
    assert G.phantom_heavy_alive(ps) == ["refresh_recall"]


def test_regression_bug_29_phantom_names_in_a_launcher_shell_are_not_phantom_jobs():
    """BC-29 (2026-09-28, live 14:20-16:28 EDT): a launcher shell whose OWN argv merely CONTAINS a Phantom
    job name as text -- a `pgrep -f "..."` wait-loop pattern, a `grep`/`tail` watching a job's log -- is not
    a Phantom heavy job. Only a process that actually EXECUTES the job's script (argv has a path ending in
    `<name>.py`, run by a Python interpreter) counts. Live: mem_guard (pid 33005) SIGSTOPped an unrelated run
    for 7,217s because its own `/bin/zsh -c ...` launcher's argv contained the pgrep pattern text below."""
    ps = "\n".join([
        '30001 /bin/zsh -c pgrep -f "aggregate_cell_hour|refresh_recall|verify_cron|track_record|retrain_gulf" ; sleep 5',
        '30002 grep aggregate_cell_hour /Users/x/Projects/prc-challenge/logs/cml_scratch/he2_doy_s0.log',
        '30003 tail -f /Users/x/Projects/prc-challenge/logs/cml_scratch/aggregate_cell_hour.log',
        '30004 /opt/homebrew/Cellar/python@3.11/3.11.15/Frameworks/Python.framework/Versions/3.11/Resources/'
        'Python.app/Contents/MacOS/Python scripts/live_ingest/aggregate_cell_hour.py --dates 2026-09-28 --force',
        '30005 /opt/homebrew/bin/python3.11 -B retrain_gulf_shadow_day_eval.py --day 2026-09-28',
    ]) + "\n"
    assert G.phantom_heavy_alive(ps) == ["aggregate_cell_hour", "retrain_gulf_shadow_day_eval"]


def test_phantom_heavy_alive_excludes_an_ancestor_pid_even_if_its_line_looks_like_a_real_job():
    """BC-29: `ignore_pids` also carries the guard's own ancestor pid chain (main() merges it in via
    `_ancestor_pids`), so a line that WOULD otherwise match a real Phantom job is still excluded when its pid
    is one of the guard's own ancestors."""
    ps = "40001 /opt/homebrew/bin/python3.11 scripts/live_ingest/track_record.py --day 2026-09-28\n"
    assert G.phantom_heavy_alive(ps, ignore_pids=frozenset({40001})) == []
    assert G.phantom_heavy_alive(ps, ignore_pids=frozenset()) == ["track_record"]


def test_phantom_heavy_alive_matches_a_phantom_path_process_even_without_python_in_the_executable_name():
    """The `/Projects/Phantom` fallback: a process whose executable name doesn't literally contain "python"
    (e.g. a venv symlink) still counts as long as it lives under Phantom's own tree AND actually runs the job
    script -- but a shell under that same tree still never matches (shells are always excluded)."""
    ps = "50001 /Users/x/Projects/Phantom/.venv/bin/py3 scripts/live_ingest/verify_cron.py --once\n"
    assert G.phantom_heavy_alive(ps) == ["verify_cron"]
    ps_shell = '50002 /bin/zsh -c "/Users/x/Projects/Phantom/scripts/live_ingest/verify_cron.py --once"\n'
    assert G.phantom_heavy_alive(ps_shell) == []


def test_phantom_heavy_alive_multiple_jobs_multiple_lines():
    ps = ("111 python aggregate_cell_hour.py\n"
          "222 python track_record.py\n")
    assert G.phantom_heavy_alive(ps) == ["aggregate_cell_hour", "track_record"]


def test_decide_with_phantom_pauses_on_phantom_alone():
    act, reason = G.decide_with_phantom(paused=False, swap_free_mb=5000, free_pct=40, paused_for_s=0,
                                         phantom_names=["track_record"], phantom_quiet_for_s=0)
    assert (act, reason) == ("PAUSE", "phantom:track_record")


def test_decide_with_phantom_memory_reason_takes_the_label_when_both_trigger():
    act, reason = G.decide_with_phantom(paused=False, swap_free_mb=100, free_pct=40, paused_for_s=0,
                                         phantom_names=["track_record"], phantom_quiet_for_s=0)
    assert (act, reason) == ("PAUSE", "memory")


def test_decide_with_phantom_runs_when_nothing_triggers():
    act, reason = G.decide_with_phantom(paused=False, swap_free_mb=5000, free_pct=40, paused_for_s=0,
                                         phantom_names=[], phantom_quiet_for_s=9999)
    assert (act, reason) == ("RUN", "")


def test_decide_with_phantom_stays_paused_while_phantom_alive_even_if_memory_fine():
    act, reason = G.decide_with_phantom(paused=True, swap_free_mb=G.RESUME_SWAP_MB, free_pct=G.RESUME_FREE_PCT,
                                         paused_for_s=G.MIN_PAUSE_S, phantom_names=["verify_cron"],
                                         phantom_quiet_for_s=0)
    assert (act, reason) == ("HOLD", "phantom:verify_cron")


def test_decide_with_phantom_holds_during_the_quiet_cooldown():
    act, reason = G.decide_with_phantom(paused=True, swap_free_mb=G.RESUME_SWAP_MB, free_pct=G.RESUME_FREE_PCT,
                                         paused_for_s=G.MIN_PAUSE_S, phantom_names=[],
                                         phantom_quiet_for_s=G.PHANTOM_QUIET_S - 1)
    assert (act, reason) == ("HOLD", "phantom-cooldown")


def test_decide_with_phantom_resumes_only_after_quiet_and_memory_recovered():
    act, reason = G.decide_with_phantom(paused=True, swap_free_mb=G.RESUME_SWAP_MB, free_pct=G.RESUME_FREE_PCT,
                                         paused_for_s=G.MIN_PAUSE_S, phantom_names=[],
                                         phantom_quiet_for_s=G.PHANTOM_QUIET_S)
    assert (act, reason) == ("RESUME", "")


def test_decide_with_phantom_memory_pressure_alone_still_pauses_like_decide():
    # Same inputs as test_pauses_when_swap_free_drops_below_threshold, with no phantom involved at all.
    act, reason = G.decide_with_phantom(paused=False, swap_free_mb=300, free_pct=40, paused_for_s=0,
                                         phantom_names=[], phantom_quiet_for_s=9999)
    assert (act, reason) == ("PAUSE", "memory")


def test_decide_with_phantom_matches_plain_decide_when_phantom_never_present():
    # Flag-off-equivalent path: with phantom_names always empty and quiet_for huge, decide_with_phantom's action
    # must equal plain decide()'s action for every state decide() itself is tested against.
    cases = [
        (False, 300, 40, 0),
        (False, G.PAUSE_SWAP_MB, 40, 0),
        (True, G.RESUME_SWAP_MB, G.RESUME_FREE_PCT, G.MIN_PAUSE_S),
        (True, G.RESUME_SWAP_MB, G.RESUME_FREE_PCT, G.MIN_PAUSE_S - 1),
    ]
    for paused, sw, pct, pf in cases:
        act, _ = G.decide_with_phantom(paused, sw, pct, pf, phantom_names=[], phantom_quiet_for_s=9999)
        assert act == G.decide(paused, sw, pct, pf)


def test_signal_target_list_never_contains_a_phantom_pid():
    """Negative invariant, extended: tree() enumerates candidate pids purely from a parent map (no OS calls);
    check_target -- the actual gate before any os.kill -- must still refuse every Phantom-marked command, so no
    Phantom pid can ever reach the signal step, flag on or off."""
    parents = {10: 1, 11: 10, 12: 11}
    commands = {10: "/opt/homebrew/bin/python3.11 scripts/analysis/nsc_fold.py",
                11: "/opt/homebrew/bin/python3.11 -B scripts/analysis/lgbm_fold.py",
                12: "python /Users/x/Projects/Phantom/scripts/live_ingest/aggregate_cell_hour.py"}
    pids = G.tree(10, parents)
    refused = []
    for p in pids:
        try:
            G.check_target(commands[p])
        except ValueError:
            refused.append(p)
    assert refused == [12]
    # simulate _signal_tree's own gate-before-kill ordering: it must raise before any kill is issued
    with pytest.raises(ValueError, match="refusing to signal a Phantom process"):
        for p in pids:
            G.check_target(commands[p])


def test_argparse_accepts_pause_on_phantom_flag():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--pause-on-phantom", action="store_true")
    a = ap.parse_args(["--pid", "123", "--log", "x.log", "--pause-on-phantom"])
    assert a.pause_on_phantom is True
    b = ap.parse_args(["--pid", "123", "--log", "x.log"])
    assert b.pause_on_phantom is False


def test_mem_guard_main_argparse_accepts_pause_on_phantom_flag(monkeypatch, tmp_path):
    """Exercises the real parser inside mem_guard.main(), not a re-declared copy, by stopping just after
    parsing (patching check_target to raise a sentinel) so no process loop or OS call runs."""
    sentinel = "STOP_AFTER_PARSE"

    def fake_check_target(command):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(G, "_command", lambda pid: "irrelevant")
    monkeypatch.setattr(G, "check_target", fake_check_target)
    log_path = tmp_path / "x.log"
    with pytest.raises(RuntimeError, match=sentinel):
        G.main(["--pid", "123", "--log", str(log_path), "--pause-on-phantom"])


# ------------------------------------------------------------------ BC-37 (2026-10-01): a phantom-caused pause and
# the memory hysteresis. Owner decision 2026-10-01 ("1"): after a pause that only a Phantom job caused, resume when
# Phantom is quiet and memory is not below the PAUSE thresholds; the 1,000 MB / 15 % resume hysteresis stays for
# pauses that memory caused (or that memory joined while paused).

def test_regression_bug_37_a_phantom_caused_pause_resumes_without_the_memory_hysteresis():
    """2026-10-01 17:44–18:09: the full fit sat paused 25 min after aggregate_cell_hour had gone, because swap free was
    761 MB (< the 1,000 MB RESUME line) with 57 % of RAM free; macOS sizes its swap files so that free swap hovers
    around 0.7–1.7 GB. Seen red: HOLD. Mutation: require RESUME_SWAP_MB for a phantom-caused pause -> red."""
    act, reason = G.decide_with_phantom(paused=True, swap_free_mb=761, free_pct=57, paused_for_s=1489,
                                         phantom_names=[], phantom_quiet_for_s=1400, memory_caused=False)
    assert (act, reason) == ("RESUME", "")


def test_a_memory_caused_pause_keeps_the_resume_hysteresis():
    """The two causes differ on free RAM: a phantom-only pause resumes above the PAUSE line (8 %), a memory-caused one
    needs the RESUME line (30 %). Mutation: ignore memory_caused -> red."""
    args = dict(paused=True, swap_free_mb=761, free_pct=20, paused_for_s=1489, phantom_names=[], phantom_quiet_for_s=9999)
    assert G.decide_with_phantom(**args, memory_caused=True) == ("HOLD", "")
    assert G.decide_with_phantom(**args, memory_caused=False) == ("RESUME", "")
    # and the default (no cause given) is the strict rule: existing callers lose no protection
    assert G.decide_with_phantom(**args)[0] == "HOLD"


def test_a_phantom_caused_pause_still_waits_for_phantom_and_its_cooldown():
    act, reason = G.decide_with_phantom(paused=True, swap_free_mb=5000, free_pct=60, paused_for_s=500,
                                         phantom_names=["aggregate_cell_hour"], phantom_quiet_for_s=0, memory_caused=False)
    assert (act, reason) == ("HOLD", "phantom:aggregate_cell_hour")
    act, reason = G.decide_with_phantom(paused=True, swap_free_mb=5000, free_pct=60, paused_for_s=500,
                                         phantom_names=[], phantom_quiet_for_s=G.PHANTOM_QUIET_S - 1, memory_caused=False)
    assert (act, reason) == ("HOLD", "phantom-cooldown")
    act, _ = G.decide_with_phantom(paused=True, swap_free_mb=5000, free_pct=60, paused_for_s=500,
                                    phantom_names=[], phantom_quiet_for_s=G.PHANTOM_QUIET_S, memory_caused=False)
    assert act == "RESUME"


def test_a_phantom_caused_pause_holds_while_memory_is_below_the_pause_thresholds():
    for sw, pct in ((G.PAUSE_SWAP_MB - 1, 60), (5000, G.PAUSE_FREE_PCT - 1)):
        act, reason = G.decide_with_phantom(paused=True, swap_free_mb=sw, free_pct=pct, paused_for_s=500,
                                             phantom_names=[], phantom_quiet_for_s=9999, memory_caused=False)
        assert (act, reason) == ("HOLD", "memory")
    # exactly at the pause thresholds is not below them (decide() pauses on < only)
    act, _ = G.decide_with_phantom(paused=True, swap_free_mb=G.PAUSE_SWAP_MB, free_pct=G.PAUSE_FREE_PCT,
                                    paused_for_s=500, phantom_names=[], phantom_quiet_for_s=9999, memory_caused=False)
    assert act == "RESUME"


def test_next_cause_tracks_who_caused_the_pause_and_escalates_to_memory():
    # a new pause takes its cause from the reason
    assert G.next_cause(False, False, "PAUSE", "memory", 100, 40) is True
    assert G.next_cause(False, True, "PAUSE", "phantom:track_record", 5000, 40) is False
    # running: no cause
    assert G.next_cause(False, True, "RUN", "", 5000, 40) is False
    # paused for Phantom, memory fine: stays phantom-caused
    assert G.next_cause(True, False, "HOLD", "phantom:track_record", 761, 57) is False
    # paused for Phantom, memory falls below a pause threshold: memory joins, and stays joined after it recovers a bit
    assert G.next_cause(True, False, "HOLD", "memory", G.PAUSE_SWAP_MB - 1, 57) is True
    assert G.next_cause(True, False, "HOLD", "phantom:track_record", 5000, G.PAUSE_FREE_PCT - 1) is True
    assert G.next_cause(True, True, "HOLD", "", 761, 57) is True
    # a resume clears it
    assert G.next_cause(True, True, "RESUME", "", 5000, 60) is False


def test_resume_label_names_the_cause_in_the_log():
    assert G.resume_label(True) == "memory" and G.resume_label(False) == "phantom-quiet"


# ------------------------------------------------------------------ BC-43 (2026-10-04, line-by-line tooling audit T1/T3):
# the guard must never exit — by an exception, a signal or the target ending — while processes it stopped stay stopped.
# The loop is `run()` with injected OS operations so these paths are testable; `main()` wires the real ones.

class FakeOps:
    """Scripted OS layer: `readings` is a list of (swap_free_mb, free_pct) per poll; `alive` a list of booleans per poll
    (or an exception to raise); `fail_at` raises at that poll's metric read; `trees` the process tree per poll (the last
    one repeats; default `tree` throughout). Records every signal sent."""

    def __init__(self, readings, alive=None, fail_at=None, tree=(100, 101, 102), trees=None):
        self.readings, self.alive_seq, self.fail_at = list(readings), list(alive or []), fail_at
        self.trees = [list(t) for t in trees] if trees else [list(tree)]
        self.poll, self.sent = -1, []

    @property
    def tree(self):
        return self.trees[min(max(self.poll, 0), len(self.trees) - 1)]

    def alive(self, pid):
        self.poll += 1
        a = self.alive_seq[self.poll] if self.poll < len(self.alive_seq) else True
        if isinstance(a, BaseException):
            raise a
        return a

    def swap_free_mb(self):
        if self.fail_at is not None and self.poll == self.fail_at:
            raise self.fail_exc if hasattr(self, "fail_exc") else ValueError("cannot parse memory_pressure output")
        return self.readings[min(self.poll, len(self.readings) - 1)][0]

    def free_pct(self):
        return self.readings[min(self.poll, len(self.readings) - 1)][1]

    def tree_pids(self, root):
        return list(self.tree)

    def signal_tree(self, root, sig):
        self.sent.append((sig, tuple(self.tree)))
        return list(self.tree)

    def signal_pids(self, pids, sig):
        self.sent.append((sig, tuple(pids)))
        return list(pids)

    def hold_signals(self):
        self.sent.append(("hold",))

    def phantom_names(self, ignore):
        return []


def _run(ops, pause_on_phantom=False, **kw):
    lines = []
    clock = iter(range(0, 10_000, 15))
    rc = G.run(100, lines.append, pause_on_phantom, ops, frozenset(), now=lambda: float(next(clock)),
               sleep=lambda s: None, **kw)
    return rc, lines


MODES = pytest.mark.parametrize("pause_on_phantom", [False, True], ids=["memory-only", "pause-on-phantom"])


@MODES
def test_regression_bug_43_a_crash_while_paused_resumes_every_stopped_process(pause_on_phantom):
    """T1: memory_pressure printed something unparseable while the run was paused; the guard died and the tree stayed
    SIGSTOPped for good (its stderr went to /dev/null). Now the exception propagates only after SIGCONT reaches every
    process the guard stopped, and the log says so."""
    ops = FakeOps(readings=[(100.0, 50)], fail_at=2)            # poll 0 pauses (swap < 350 MB); poll 2 cannot parse
    lines = []
    with pytest.raises(ValueError, match="cannot parse"):
        G.run(100, lines.append, pause_on_phantom, ops, frozenset(), now=lambda: 0.0, sleep=lambda s: None)
    assert ops.sent[0] == (G.signal.SIGSTOP, (100, 101, 102))
    assert ops.sent[-1] == (G.signal.SIGCONT, (100, 101, 102))
    assert any("MEMGUARD ERROR ValueError" in l for l in lines)
    assert any("MEMGUARD RELEASED pids=[100, 101, 102]" in l for l in lines)


@MODES
def test_regression_bug_43_a_signal_while_paused_resumes_every_stopped_process(pause_on_phantom):
    """A SIGTERM / SIGINT / SIGHUP is turned into SystemExit by main(); the release runs on that path too, and the log
    names it an EXIT (a requested stop), not an ERROR."""
    ops = FakeOps(readings=[(100.0, 50)], fail_at=1)
    ops.fail_exc = SystemExit(143)
    lines = []
    with pytest.raises(SystemExit):
        G.run(100, lines.append, pause_on_phantom, ops, frozenset(), now=lambda: 0.0, sleep=lambda s: None)
    assert ops.sent[-1] == (G.signal.SIGCONT, (100, 101, 102))
    assert "MEMGUARD EXIT code=143 (stopped by a signal)" in lines and not any("ERROR" in l for l in lines)


@MODES
def test_regression_bug_43_a_target_that_ends_while_paused_leaves_no_child_stopped(pause_on_phantom):
    """T3: the run's root process exited while the tree was paused; its children (a CatBoost or MLP worker) stayed
    SIGSTOPped as orphans holding their memory. The guard now resumes every process it stopped before it exits."""
    ops = FakeOps(readings=[(100.0, 50)], alive=[True, True, False])
    rc, lines = _run(ops, pause_on_phantom)
    assert rc == 0
    assert ops.sent == [(G.signal.SIGSTOP, (100, 101, 102)), ("hold",), (G.signal.SIGCONT, (100, 101, 102))]
    assert "MEMGUARD target gone while paused" in lines and "MEMGUARD END target exited" in lines
    assert lines[-1] == "MEMGUARD RELEASED pids=[100, 101, 102] (the guard is exiting; no process it stopped stays stopped)"


def test_a_normal_pause_and_resume_release_nothing_twice():
    """No regression of the decision path: pause, hold for the minimum time, resume on recovery — then exit with
    nothing left to release (the resume already continued the tree)."""
    readings = [(100.0, 50)] + [(2000.0, 60)] * 8
    ops = FakeOps(readings, alive=[True] * 9 + [False])
    rc, lines = _run(ops)
    assert rc == 0
    assert [s[0] for s in ops.sent] == [G.signal.SIGSTOP, G.signal.SIGCONT]
    assert not any("RELEASED" in l for l in lines)


def test_the_real_alive_check_reads_a_permission_error_as_alive(monkeypatch):
    """BC-39 sibling: os.kill(pid, 0) raising PermissionError means the process exists; the guard keeps guarding."""
    def deny(pid, sig):
        raise PermissionError(1, "Operation not permitted")
    monkeypatch.setattr(G.os, "kill", deny)
    assert G.OsOps().alive(1) is True


def test_release_mode_continues_a_tree_and_refuses_a_phantom_target(monkeypatch):
    sent = []
    monkeypatch.setattr(G, "_command", lambda pid: "/usr/bin/python3 fit.py")
    monkeypatch.setattr(G, "_parents", lambda: {101: 100, 102: 101, 7: 1, 300: 1})
    monkeypatch.setattr(G, "_groups", lambda: {100: 50, 101: 50, 102: 102, 7: 7, 300: 102})   # 300: reparented, 102's group
    monkeypatch.setattr(G.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert G.main(["--release", "100"]) == 0
    assert sent == [(100, G.signal.SIGCONT), (101, G.signal.SIGCONT), (102, G.signal.SIGCONT), (300, G.signal.SIGCONT)]
    monkeypatch.setattr(G, "_command", lambda pid: "python /Users/x/Projects/Phantom/scripts/track_record.py")
    with pytest.raises(ValueError, match="refusing to signal a Phantom process"):
        G.main(["--release", "100"])


def test_main_turns_termination_signals_into_a_clean_exit():
    with pytest.raises(SystemExit) as e:
        G._exit_on_signal(G.signal.SIGTERM, None)
    assert e.value.code == 128 + G.signal.SIGTERM


def test_regression_bug_43_the_exit_release_uses_the_pids_recorded_at_the_pause_not_the_current_tree():
    """T3 as it happens: once the root has exited, its stopped children are reparented and the tree walked from the
    root no longer holds them. The release must use the pids recorded at the pause."""
    ops = FakeOps(readings=[(100.0, 50)], alive=[True, True, False], trees=[(100, 101, 102), (100, 101, 102), (100,)])
    rc, lines = _run(ops)
    assert rc == 0
    assert ops.sent[-1] == (G.signal.SIGCONT, (100, 101, 102))


@MODES
def test_a_resume_also_continues_a_stopped_process_that_has_left_the_tree(pause_on_phantom):
    """102's parent ended during the pause, so 102 is no longer in the tree walked from the root; the resume continues it
    too, and the guard then has nothing left to release."""
    readings = [(100.0, 50)] + [(2000.0, 60)] * 8
    ops = FakeOps(readings, alive=[True] * 9 + [False], trees=[(100, 101, 102), (100, 101)])
    rc, lines = _run(ops, pause_on_phantom)
    assert rc == 0
    assert ops.sent == [(G.signal.SIGSTOP, (100, 101, 102)), (G.signal.SIGCONT, (100, 101)), (G.signal.SIGCONT, (102,))]
    assert any("MEMGUARD RESUMED" in l and "pids=[100, 101, 102]" in l for l in lines)
    assert not any("RELEASED" in l for l in lines)


def test_the_release_names_the_pids_it_could_not_signal():
    class Partial(FakeOps):
        def signal_pids(self, pids, sig):
            self.sent.append((sig, tuple(pids)))
            return [p for p in pids if p != 102]
    ops = Partial(readings=[(100.0, 50)], alive=[True, True, False])
    rc, lines = _run(ops)
    assert lines[-1].startswith("MEMGUARD RELEASED pids=[100, 101] ")
    assert lines[-1].endswith("; not signalled (ended, or the pid now belongs to another process): [102]")


def test_the_real_signal_pids_skips_a_phantom_a_gone_and_a_foreign_pid(monkeypatch):
    """A pid recorded at the pause may since belong to another process: never a Phantom one, and one we may not signal
    (another user's) or that has ended is skipped without stopping the rest."""
    commands = {1: "/usr/bin/python3 fit.py", 2: "python /x/Phantom/scripts/track_record.py", 3: "python w.py",
                4: "python other_user.py", 5: "python fit_worker.py"}
    sent = []

    def kill(pid, sig):
        if pid == 3:
            raise ProcessLookupError
        if pid == 4:
            raise PermissionError(1, "Operation not permitted")
        sent.append(pid)
    monkeypatch.setattr(G, "_command", lambda pid: commands[pid])
    monkeypatch.setattr(G.os, "kill", kill)
    assert G.OsOps().signal_pids([1, 2, 3, 4, 5], G.signal.SIGCONT) == [1, 5]
    assert sent == [1, 5]


@pytest.mark.parametrize("restart", [False, True])
def test_main_installs_the_exit_handlers_and_names_a_restart_in_the_log(monkeypatch, tmp_path, restart):
    """main() wires SIGTERM / SIGINT / SIGHUP to the clean exit (recorded here, so pytest's own handlers are untouched)
    and starts the log with START, or RESTART for a guard that replaces one that died during the same run."""
    installed, seen = {}, {}
    monkeypatch.setattr(G.signal, "signal", lambda sig, handler: installed.__setitem__(sig, handler))
    monkeypatch.setattr(G, "_command", lambda pid: "/usr/bin/python3 fit.py")
    monkeypatch.setattr(G, "_ancestor_pids", lambda pid: frozenset({7}))
    monkeypatch.setattr(G, "run", lambda pid, log, pop, ops, ignore: seen.update(pid=pid, pop=pop, ignore=ignore) or 0)
    log = tmp_path / "guard.log"
    assert G.main(["--pid", "123", "--log", str(log), *(["--restart"] if restart else [])]) == 0
    assert installed == {s: G._exit_on_signal for s in (G.signal.SIGTERM, G.signal.SIGINT, G.signal.SIGHUP)}
    assert seen["pid"] == 123 and seen["pop"] is False and {123, 7} <= seen["ignore"]
    first = log.read_text().splitlines()[0]
    assert (" MEMGUARD RESTART pid=123 " in first) is restart and (" MEMGUARD START pid=123 " in first) is not restart


@pytest.mark.slow
def test_a_terminated_guard_exits_143_and_logs_exit(tmp_path):
    """The real handler end to end: a guard watching a sleeping process of ours, sent SIGTERM, ends with 128 + 15 through
    the EXIT path; the process it watched is not left stopped (whatever the machine's memory made the guard do)."""
    target = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    log = tmp_path / "guard.log"
    try:
        g = subprocess.Popen([sys.executable, "-B", str(ROOT / "tools" / "mem_guard.py"), "--pid", str(target.pid),
                              "--log", str(log)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 30
        while "MEMGUARD START" not in (log.read_text() if log.exists() else ""):
            assert time.time() < deadline and g.poll() is None, "the guard never started"
            time.sleep(0.05)
        time.sleep(0.5)                          # START is logged just before the loop; be inside it
        g.send_signal(signal.SIGTERM)
        assert g.wait(timeout=30) == 128 + signal.SIGTERM
        assert "MEMGUARD EXIT code=143 (stopped by a signal)" in log.read_text()
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(target.pid)], capture_output=True, text=True).stdout
        assert state.strip() and not state.strip().startswith("T")
    finally:
        target.kill()
        target.wait()


def test_regression_the_release_ignores_further_termination_signals_before_it_starts():
    """Protocol B review 2026-10-04 (MEDIUM-1a): main's handlers stayed installed during run()'s release, so a second
    SIGTERM raised SystemExit in the middle of the release loop and the remaining pids stayed stopped. The release now
    holds those signals first."""
    ops = FakeOps(readings=[(100.0, 50)], fail_at=2)
    with pytest.raises(ValueError):
        G.run(100, [].append, False, ops, frozenset(), now=lambda: 0.0, sleep=lambda s: None)
    assert ops.sent[-2:] == [("hold",), (G.signal.SIGCONT, (100, 101, 102))]


@pytest.mark.slow
def test_the_real_hold_ignores_termination_signals():
    code = ("import os, signal, sys; sys.path.insert(0, %r); import mem_guard as G; G.OsOps().hold_signals(); "
            "[os.kill(os.getpid(), s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)]; print('still here')"
            % str(ROOT / "tools"))
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "still here" in r.stdout


def test_regression_a_pid_whose_identity_cannot_be_read_still_gets_sigcont(monkeypatch):
    """MEDIUM-1b: under memory pressure `ps` itself may fail to start (EAGAIN / ENOMEM) — exactly when the guard is
    busy. The release then aborted and left the rest stopped. Now SIGCONT (harmless) is still sent; any other signal
    is withheld from a process it cannot identify."""
    def command(pid):
        if pid == 3:
            raise BlockingIOError(35, "Resource temporarily unavailable")
        return "python fit.py"
    sent = []
    monkeypatch.setattr(G, "_command", command)
    monkeypatch.setattr(G.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert G.OsOps().signal_pids([1, 3, 5], G.signal.SIGCONT) == [1, 3, 5]
    assert G.OsOps().signal_pids([1, 3, 5], G.signal.SIGSTOP) == [1, 5]


def test_regression_release_also_continues_a_stopped_worker_reparented_out_of_the_tree():
    """MEDIUM-4: `--release` walked the parent tree only. A stopped worker whose parent died is reparented (ppid 1) but
    stays in the command's process group (the lock wrapper makes the command a group leader): it is continued too. A
    group whose leader is outside the tree is not touched."""
    parents = {101: 100, 102: 101, 300: 1, 400: 1}
    groups = {100: 9, 101: 9, 102: 101, 300: 101, 400: 400}          # 300 left the tree but is in 101's group
    assert G.release_set(100, parents, groups) == [100, 101, 102, 300]
    sent = []
    assert G.release(100, parents, groups, lambda pid: "python fit.py", kill=lambda p, s: sent.append(p)) == \
        [100, 101, 102, 300]
    assert sent == [100, 101, 102, 300]
    with pytest.raises(ValueError, match="refusing to signal a Phantom process"):
        G.release(100, parents, groups, lambda pid: "python /x/Phantom/r17_shadow.py" if pid == 300 else "python fit.py",
                  kill=lambda p, s: sent.append(p))
    def gone(p, s):
        if p == 102:
            raise ProcessLookupError
    assert G.release(100, parents, groups, lambda pid: "python fit.py", kill=gone) == [100, 101, 300]
