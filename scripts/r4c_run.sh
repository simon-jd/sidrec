#!/usr/bin/env bash
# R4c. GPU steps take the RTX 4090 lock, one process per seed; CPU steps (the HNSW index, the ANN
# searches, the selection and the projection) do not. Logs go to data/logs/r4c/.
#
#   scripts/r4c_run.sh ann-build | ann-select | select | projection | ann-test
#   scripts/r4c_run.sh beam-valid 0 1 2 | rerank-valid 0 1 2 | test 0 1 2
set -euo pipefail
cd "$(dirname "$0")/.."
logs=data/logs/r4c
mkdir -p "$logs"
cmd=${1:-}
shift || true
case $cmd in
  beam-valid | rerank-valid | test)
    for seed in "$@"; do
      flock /var/tmp/rtx4090.lock uv run python -m sidrec.eval.r4c_cascade "$cmd" --seed "$seed" \
        >>"$logs/${cmd}_seed$seed.log" 2>&1
      echo "done $cmd $seed"
    done
    ;;
  ann-build | ann-select | select | projection | ann-test)
    uv run python -m sidrec.eval.r4c_cascade "$cmd" >>"$logs/$cmd.log" 2>&1
    echo "done $cmd"
    ;;
  *)
    echo "usage: $0 ann-build | ann-select | select | projection | ann-test | beam-valid|rerank-valid|test seed ..." >&2
    exit 2
    ;;
esac
