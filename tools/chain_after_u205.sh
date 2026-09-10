#!/bin/zsh
# Wait for the U-20.5 worker to exit, then start the serial arms queue. One heavy job at a time.
PID=21345
Q=~/Projects/prc-challenge/reports/ARMS_QUEUE.log
print -r -- "[$(date +%H:%M:%S)] chain: waiting for U-20.5 (pid $PID) to exit" | tee -a $Q
while kill -0 $PID 2>/dev/null; do sleep 60; done
print -r -- "[$(date +%H:%M:%S)] chain: U-20.5 exited; starting the arms queue" | tee -a $Q
exec $HOME/Projects/prc-challenge/tools/run_arms.sh
