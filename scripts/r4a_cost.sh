#!/usr/bin/env bash
# R4a cost measurements on the RTX 4090: one training epoch and timed evaluations for each model, on the dev
# subset and the full train split, with the largest trial configuration (1: width 256, 4 layers) and a small one
# (4: width 64, 2 layers). Each process takes the GPU lock on its own. Finished measurements are skipped.
#
#   scripts/r4a_cost.sh
set -euo pipefail
cd "$(dirname "$0")/.."
logs=data/logs/r4a
mkdir -p "$logs"
for source in dev full; do
  for trial in 1 4; do
    tag=trial$(printf %02d "$trial")
    for model in sasrec hstu tiger; do
      [ -f "results/r4a/cost/${model}_${source}_${tag}.json" ] && continue
      if [ "$model" = tiger ]; then
        cmd=(-m sidrec.train.scale_tiger)
      else
        cmd=(-m sidrec.train.scale_idrec --model "$model")
      fi
      flock /var/tmp/rtx4090.lock uv run python "${cmd[@]}" --trial "$trial" --source "$source" --cost \
        >"$logs/cost_${model}_${source}_${tag}.log" 2>&1
    done
  done
done
