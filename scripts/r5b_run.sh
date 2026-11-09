#!/usr/bin/env bash
# R5b. Every step uses the GPU and holds the RTX 4090 lock for its whole run. Logs go to
# data/logs/r5b/; each step's lock-held wall time is appended to data/logs/r5b/gpu_time.jsonl.
#
#   scripts/r5b_run.sh projection | profile | equivalence | cutoffs | quality | ladder | select | stages
#   scripts/r5b_run.sh e2e 200 50 20
set -euo pipefail
cd "$(dirname "$0")/.."
logs=data/logs/r5b
mkdir -p "$logs"

locked() {  # locked NAME ARGS...: one step under the lock, timed from the moment the lock is held
  local name=$1
  shift
  flock /var/tmp/rtx4090.lock bash -c '
    s=$(date +%s.%N)
    status=0
    uv run python -m sidrec.eval.r5b_beam "${@:2}" >>"'"$logs"'/$1.log" 2>&1 || status=$?
    e=$(date +%s.%N)
    echo "{\"step\": \"$1\", \"start\": $s, \"end\": $e, \"status\": $status}" >>"'"$logs"'/gpu_time.jsonl"
    exit $status' _ "$name" "$@"
  echo "done $name"
}

cmd=${1:-}
shift || true
case $cmd in
  projection | profile | equivalence | cutoffs | quality | ladder | select | stages)
    locked "$cmd" "$cmd"
    ;;
  e2e)
    for width in "$@"; do
      locked "e2e_w$width" e2e --width "$width"
    done
    ;;
  *)
    echo "usage: $0 projection | profile | equivalence | cutoffs | quality | ladder | select | stages | e2e width ..." >&2
    exit 2
    ;;
esac
