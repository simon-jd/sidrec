#!/usr/bin/env bash
# R4b on the RTX 4090. Every process takes the GPU lock on its own. A full-split run trains for about an hour per
# process, saves its state and exits with code 75; the next process resumes it under a fresh lock. Finished runs
# are skipped, so the script can be restarted. Creating data/logs/r4b/STOP ends it after the current process.
#
#   scripts/r4b_run.sh tune-hstu-sid                   # HSTU-SID's 12 trials on the dev subset
#   scripts/r4b_run.sh final sasrec:0 hstu:0 tiger:0   # final runs, model:seed, in this order
#   scripts/r4b_run.sh eps 0 1 2                       # TIGER's validation ε curve, per seed
set -euo pipefail
cd "$(dirname "$0")/.."
logs=data/logs/r4b
mkdir -p "$logs"

job() {  # $1 the record the job writes when it is done, $2 its log, then the python arguments
  local out=$1 log=$2 rc
  shift 2
  while [ ! -f "$out" ]; do
    if [ -f "$logs/STOP" ]; then
      echo "stopped before $out"
      exit 0
    fi
    set +e
    flock /var/tmp/rtx4090.lock uv run python "$@" >>"$log" 2>&1
    rc=$?
    set -e
    if [ "$rc" -eq 0 ] && [ ! -f "$out" ]; then
      echo "no record after exit 0: $out" >&2
      exit 1
    fi
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 75 ]; then
      echo "failed with $rc: $*" >&2
      exit "$rc"
    fi
  done
  echo "done $out"
}

selected() {
  uv run python -c "from sidrec.train.scale_idrec import selected_trial; print(selected_trial('$1'))"
}

case $1 in
  tune-hstu-sid)
    for t in $(seq 1 12); do
      tag=trial$(printf %02d "$t")
      job "results/r4b/runs/hstu-sid/$tag.json" "$logs/tune_hstu-sid_$tag.log" \
        -m sidrec.train.scale_idrec --model hstu-sid --trial "$t"
    done
    ;;
  final)
    shift
    for spec in "$@"; do
      model=${spec%:*}
      seed=${spec#*:}
      trial=$(selected "$model")
      if [ "$model" = tiger ]; then
        cmd=(-m sidrec.train.scale_tiger)
      else
        cmd=(-m sidrec.train.scale_idrec --model "$model")
      fi
      job "results/r4b/runs/$model/seed$seed.json" "$logs/final_${model}_seed$seed.log" \
        "${cmd[@]}" --trial "$trial" --final --seed "$seed"
    done
    ;;
  eps)
    shift
    for seed in "$@"; do
      job "results/r4b/tiger_eps/seed$seed.json" "$logs/eps_seed$seed.log" -m sidrec.eval.r4b_tiger_eps --seed "$seed"
    done
    ;;
  *)
    echo "usage: $0 tune-hstu-sid | final model:seed ... | eps seed ..." >&2
    exit 2
    ;;
esac
