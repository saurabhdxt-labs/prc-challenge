#!/bin/zsh
# Freeze/thaw guard for ONE of my own heavy workers (never a fleet job).
# usage: freeze_guard.sh <pid> [label]
# FREEZE (SIGSTOP) when the kernel's free share < 8% AND swap free < 1200 MB  (wedge level, per the 09-03 lesson)
# THAW   (SIGCONT) when free share > 20% AND swap free > 2500 MB for two consecutive samples.
# Emits one line per transition plus a heartbeat every 15 min; exits when the pid is gone.
PID=$1; LABEL=${2:-worker}; frozen=0; ok=0; hb=$(date +%s)
read_state() {
  FREEPCT=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub(/%/,"",$2); print int($2)}')
  SW=$(sysctl -n vm.swapusage)   # "total = 7168.00M  used = 6147.75M  free = 1020.25M  (encrypted)"
  SWUSED=$(echo "$SW" | awk '{gsub(/M/,"",$6); print int($6)}')
  SWFREE=$(echo "$SW" | awk '{gsub(/M/,"",$9); print int($9)}')
  [ -z "$FREEPCT" ] && FREEPCT=100
}
while kill -0 "$PID" 2>/dev/null; do
  read_state
  if [ "$frozen" -eq 0 ] && [ "$FREEPCT" -lt 8 ] && [ "$SWFREE" -lt 1200 ]; then
    kill -STOP "$PID" && frozen=1 && ok=0 && echo "[guard] $(date +%H:%M:%S) FROZE $LABEL pid $PID: free ${FREEPCT}% swap used ${SWUSED}M free ${SWFREE}M"
  elif [ "$frozen" -eq 1 ]; then
    if [ "$FREEPCT" -gt 20 ] && [ "$SWFREE" -gt 2500 ]; then ok=$((ok+1)); else ok=0; fi
    if [ "$ok" -ge 2 ]; then kill -CONT "$PID" && frozen=0 && echo "[guard] $(date +%H:%M:%S) THAWED $LABEL pid $PID: free ${FREEPCT}% swap used ${SWUSED}M free ${SWFREE}M"; fi
  fi
  now=$(date +%s)
  if [ $((now-hb)) -ge 900 ]; then echo "[guard] $(date +%H:%M:%S) $LABEL $( [ $frozen -eq 1 ] && echo FROZEN || echo running ) free ${FREEPCT}% swap used ${SWUSED}M free ${SWFREE}M rss $(ps -o rss= -p $PID | awk '{printf "%.2fGB",$1/1048576}')"; hb=$now; fi
  sleep 20
done
echo "[guard] $(date +%H:%M:%S) $LABEL pid $PID exited (frozen=$frozen)"
