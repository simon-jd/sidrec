#!/usr/bin/env bash
# R2 for one item-ID model on Beauty: the 12 tuning trials, selection on validation NDCG@10, the selected
# trial's checkpoint on test (seed 0), then seeds 1 and 2. Each training process takes the GPU lock on its own,
# so other jobs can run between them. Finished runs are skipped, so the script can be restarted.
#
#   scripts/r2_idrec.sh sasrec
set -euo pipefail
model=$1
cd "$(dirname "$0")/.."
logs=data/logs/r2
mkdir -p "$logs"
run() {
  local log=$1
  shift
  flock /var/tmp/rtx4090.lock uv run python -m sidrec.train.idrec --model "$model" "$@" >"$logs/$log" 2>&1
}
for t in $(seq 1 12); do
  [ -f "results/r2/runs/$model/trial$(printf %02d "$t").json" ] || run "${model}_trial$t.log" --trial "$t"
done
best=$(uv run python -m sidrec.train.select "runs/$model")
tag=trial$(printf %02d "$best")
echo "$model: selected trial $best"
[ -f "results/r2/runs/$model/${tag}_seed0.json" ] || run "${model}_${tag}_test.log" --trial "$best" --test
for seed in 1 2; do
  [ -f "results/r2/runs/$model/${tag}_seed$seed.json" ] || run "${model}_${tag}_seed$seed.log" --trial "$best" --seed "$seed" --final
done
