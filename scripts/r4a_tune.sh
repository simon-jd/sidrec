#!/usr/bin/env bash
# R4a tuning on the RTX 4090: the protocol's 12 shared trials of one model on the Electronics dev subset, seed 0,
# validation only. Each trial takes the GPU lock on its own, so other jobs can run between them. Finished trials
# are skipped, so the script can be restarted.
#
#   scripts/r4a_tune.sh hstu        # or sasrec, tiger
set -euo pipefail
model=$1
cd "$(dirname "$0")/.."
logs=data/logs/r4a
mkdir -p "$logs"
for t in $(seq 1 12); do
  tag=trial$(printf %02d "$t")
  [ -f "results/r4a/runs/$model/$tag.json" ] && continue
  if [ "$model" = tiger ]; then
    cmd=(-m sidrec.train.scale_tiger)
  else
    cmd=(-m sidrec.train.scale_idrec --model "$model")
  fi
  flock /var/tmp/rtx4090.lock uv run python "${cmd[@]}" --trial "$t" >"$logs/tune_${model}_$tag.log" 2>&1
done
