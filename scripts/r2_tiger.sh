#!/usr/bin/env bash
# R2 for TIGER on Beauty: the 12 tuning trials on Sentence-T5 semantic IDs, selection on validation NDCG@10,
# the selected trial's checkpoint on test (seed 0) and seeds 1 and 2; then the same configuration with seeds
# 0, 1 and 2 on the Qwen3-Embedding-0.6B and -4B semantic IDs. The embeddings and RQ-VAE IDs must exist
# (sidrec.semantic_ids.embed and sidrec.semantic_ids.rqvae). Each training process takes the GPU lock on its
# own. Finished runs are skipped, so the script can be restarted.
set -euo pipefail
cd "$(dirname "$0")/.."
logs=data/logs/r2
mkdir -p "$logs"
run() {
  local log=$1
  shift
  flock /var/tmp/rtx4090.lock uv run python -m sidrec.train.tiger "$@" >"$logs/$log" 2>&1
}
enc=st5-xxl
for t in $(seq 1 12); do
  [ -f "results/r2/runs/tiger_$enc/trial$(printf %02d "$t").json" ] || run "tiger_${enc}_trial$t.log" --encoder $enc --trial "$t"
done
best=$(uv run python -m sidrec.train.select "runs/tiger_$enc")
tag=trial$(printf %02d "$best")
echo "tiger: selected trial $best"
[ -f "results/r2/runs/tiger_$enc/${tag}_seed0.json" ] || run "tiger_${enc}_${tag}_test.log" --encoder $enc --trial "$best" --test
for seed in 1 2; do
  [ -f "results/r2/runs/tiger_$enc/${tag}_seed$seed.json" ] ||
    run "tiger_${enc}_${tag}_seed$seed.log" --encoder $enc --trial "$best" --seed "$seed" --final
done
for enc in qwen3-emb-0.6b qwen3-emb-4b; do
  for seed in 0 1 2; do
    [ -f "results/r2/runs/tiger_$enc/${tag}_seed$seed.json" ] ||
      run "tiger_${enc}_${tag}_seed$seed.log" --encoder $enc --trial "$best" --seed "$seed" --final
  done
done
