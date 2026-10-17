#!/bin/bash
# every 5 min: append a status row to the PLAN.md change log (new done/FAILED lines from the queue logs + progress of running jobs)
cd "$(dirname "$0")"
declare -A seen
for q in queue_hstu.log queue_delphi.log; do seen[$q]=$(wc -l < $q); done
while true; do
  sleep 300
  new=""
  for q in queue_hstu.log queue_delphi.log; do
    n=$(wc -l < $q); [ "$n" -gt "${seen[$q]}" ] && new+="$(tail -n +$((seen[$q]+1)) $q | tr '\n' ';' ) "; seen[$q]=$n
  done
  run=""
  for pid in $(pgrep -f "^python (train|delphi_train).py"); do
    args=$(tr '\0' ' ' < /proc/$pid/cmdline); m=$(echo $args | sed -E 's/.*--model ([^ ]+).*/\1/'); s=$(echo $args | sed -E 's/.*--split ([^ ]+).*/\1/'); f=$(echo $args | sed -E 's/.*--fold ([^ ]+).*/\1/')
    last=$(tail -1 runs/$m/$s/$f/stdout.log 2>/dev/null | cut -c1-70)
    run+="$m $s $f: \`$last\`; "
  done
  [ -z "$run" ] && run="nothing running (GPU idle)"
  echo "| $(date -u +%H:%M) | [auto status] ${new:+Finished: $new}Running: $run |" >> PLAN.md
done
