#!/usr/bin/env bash
# Download the raw datasets into data/raw (git-ignored) and verify them against docs/SHA256SUMS.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/data/raw/beauty2014" "$root/data/raw/amazon2023/Electronics"
cd "$root/data/raw"

snap=https://snap.stanford.edu/data/amazon/productGraph/categoryFiles
mcauley=https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023
s3rec=e6c774da92dba847b00ac89e8bfc5c55cc44d378

get() {
  if [[ -f "$2" ]]; then echo "have $2"; return; fi
  curl -fsSL --retry 5 --retry-delay 5 -o "$2.part" "$1"
  mv "$2.part" "$2"
  echo "got  $2"
}

get "$snap/reviews_Beauty_5.json.gz" beauty2014/reviews_Beauty_5.json.gz &
get "$snap/meta_Beauty.json.gz" beauty2014/meta_Beauty.json.gz &
# The S3-Rec authors' processed Beauty sequences, used only to check our Beauty preprocessing.
get "https://raw.githubusercontent.com/RUCAIBox/CIKM2020-S3Rec/$s3rec/data/Beauty.txt" beauty2014/s3rec_Beauty.txt &
for s in train valid test; do
  get "$mcauley/benchmark/5core/timestamp/Electronics.$s.csv.gz" "amazon2023/Electronics/Electronics.$s.csv.gz" &
done
# The with-history variant of the eval splits, used only to check our history construction.
for s in valid test; do
  get "$mcauley/benchmark/5core/timestamp_w_his/Electronics.$s.csv.gz" "amazon2023/Electronics/w_his.Electronics.$s.csv.gz" &
done
get "$mcauley/raw/meta_categories/meta_Electronics.jsonl.gz" amazon2023/Electronics/meta_Electronics.jsonl.gz &
wait

sha256sum -c "$root/docs/SHA256SUMS"
