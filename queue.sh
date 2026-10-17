#!/bin/bash
# usage: ./queue.sh <jobs file> <parallel>. Each line: <model> <split> <fold> [extra train.py flags]. Skips finished runs.
run() {
  set -- $1
  m=$1; s=$2; f=$3; shift 3
  [ -f runs/$m/$s/$f/predictions.parquet ] && { echo "skip $m $s $f"; return; }
  mkdir -p runs/$m/$s/$f
  script=train.py; case $m in Delphi*) script=delphi_train.py;; esac
  python $script --model $m --split $s --fold $f "$@" > runs/$m/$s/$f/stdout.log 2>&1 && echo "done $m $s $f" || echo "FAILED $m $s $f"
}
export -f run
grep -v '^#' "$1" | xargs -P "${2:-2}" -I{} bash -c 'run "{}"'
echo QUEUE_FINISHED
