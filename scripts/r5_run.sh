#!/usr/bin/env bash
# R5. Every step uses the GPU and holds the RTX 4090 lock for its whole run. Logs go to
# data/logs/r5/; each step's lock-held wall time is appended to data/logs/r5/gpu_time.jsonl.
#
#   scripts/r5_run.sh projection | beam-valid | select | quality | ladder | stages
#   scripts/r5_run.sh e2e 200 50 20
set -euo pipefail
cd "$(dirname "$0")/.."
logs=data/logs/r5
mkdir -p "$logs"

locked() {  # locked NAME ARGS...: one step under the lock, timed from the moment the lock is held
  local name=$1
  shift
  flock /var/tmp/rtx4090.lock bash -c '
    s=$(date +%s.%N)
    status=0
    uv run python -m sidrec.eval.r5_serving "${@:2}" >>"'"$logs"'/$1.log" 2>&1 || status=$?
    e=$(date +%s.%N)
    echo "{\"step\": \"$1\", \"start\": $s, \"end\": $e, \"status\": $status}" >>"'"$logs"'/gpu_time.jsonl"
    exit $status' _ "$name" "$@"
  echo "done $name"
}

cmd=${1:-}
shift || true
case $cmd in
  projection | beam-valid | select | quality | ladder | stages)
    locked "$cmd" "$cmd"
    ;;
  e2e)
    for width in "$@"; do
      locked "e2e_w$width" e2e --width "$width"
    done
    ;;
  *)
    echo "usage: $0 projection | beam-valid | select | quality | ladder | stages | e2e width ..." >&2
    exit 2
    ;;
esac
