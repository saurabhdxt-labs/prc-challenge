#!/bin/zsh
# Serial driver for the remaining pre-registered arms. Each arm: --smoke first (real caches,
# smoke months); only on exit 0 does the full run start. One heavy job at a time, always.
# Outputs go to reports/ (never /tmp). Progress -> reports/ARMS_QUEUE.log
cd ~/Projects/prc-challenge || exit 1
Q=reports/ARMS_QUEUE.log
say() { print -r -- "[$(date +%H:%M:%S)] $*" | tee -a $Q; }
run() {                       # run <name> <console-log> <args...>
  local name=$1 console=$2; shift 2
  say "=== $name: smoke ==="
  if ! PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 python3.11 -B -u scripts/lgbm_fold.py "$@" --smoke \
        >> reports/${name}_smoke.log 2>&1; then
    say "!!! $name SMOKE FAILED (see reports/${name}_smoke.log) - skipping the full run"; return 1
  fi
  say "$name: smoke ok; starting the full run -> $console"
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 /usr/bin/time -l python3.11 -B -u scripts/lgbm_fold.py "$@" \
      > $console 2>&1
  local rc=$?
  say "$name: exit $rc  $(grep -E 'json ->' $console | tail -1 | cut -c1-120)"
  return $rc
}
say "### arms queue started; commit $(git rev-parse --short HEAD)"
run sweep         reports/lgbm_fold_sweep.console.log         --sweep
run queue_day     reports/lgbm_fold_queue_day.console.log     --dayfeats   --queue --baseline A2 --pa-trees es
run queue_ytarget reports/lgbm_fold_queue_ytarget.console.log --ytarget    --queue --baseline A2 --pa-trees es
run queue_order   reports/lgbm_fold_queue_order.console.log   --orderfeats --queue --baseline A2 --pa-trees es
run queue_weather reports/lgbm_fold_queue_weather.console.log --weatherfeats --queue --baseline A2 --pa-trees es
say "### the five unattended arms are done; --confirm and --catboost need the sweep's ranking and are left for the session"
