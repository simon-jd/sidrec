"""Beauty (Amazon 2014, 5-core): user sequences and the leave-one-out split of S3-Rec, P5 and TIGER.

    uv run python -m sidrec.data.beauty     # writes data/processed/beauty/sequences.parquet and the stats

Each user's reviews are sorted by unixReviewTime with a stable sort, so same-day ties keep file order, as in the
S3-Rec preprocessing script (data/data_process.py). The test target is the last item, the validation target the
second-to-last, and training uses the items before them.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "beauty2014"
OUT_DIR = ROOT / "data" / "processed" / "beauty"
REVIEWS = "reviews_Beauty_5.json.gz"
S3REC = "s3rec_Beauty.txt"


def read_reviews(path: Path) -> pl.DataFrame:
    schema = {"reviewerID": pl.String, "asin": pl.String, "overall": pl.Float64, "unixReviewTime": pl.Int64}
    with gzip.open(path) as f:
        df = pl.read_ndjson(f.read(), schema=schema)
    return df.rename(
        {"reviewerID": "user_id", "asin": "item_id", "overall": "rating", "unixReviewTime": "ts"}
    ).with_row_index("file_row")


def build_sequences(reviews: pl.DataFrame) -> pl.DataFrame:
    """One row per user, in order of the user's first review in the file, with the leave-one-out targets."""
    first = reviews.group_by("user_id").agg(first_row=pl.col("file_row").min())
    return (
        reviews.join(first, on="user_id")
        .sort("first_row", "ts", "file_row")
        .group_by("user_id", maintain_order=True)
        .agg(items=pl.col("item_id"), timestamps=pl.col("ts"), n=pl.len().cast(pl.Int64))
        .with_columns(
            train_items=pl.col("items").list.head(pl.col("n") - 2),
            valid_target=pl.col("items").list.get(-2),
            test_target=pl.col("items").list.get(-1),
        )
    )


def kcore_violations(reviews: pl.DataFrame, k: int = 5) -> dict[str, int]:
    users = reviews.group_by("user_id").len().filter(pl.col("len") < k).height
    items = reviews.group_by("item_id").len().filter(pl.col("len") < k).height
    return {"users_below_k": users, "items_below_k": items}


def stats(seq: pl.DataFrame) -> dict:
    n = seq["n"]
    return {
        "users": seq.height,
        "items": seq["items"].explode().n_unique(),
        "interactions": int(n.sum()),
        "seq_len_mean": round(float(n.mean()), 4),
        "seq_len_median": float(n.median()),
        "seq_len_min": int(n.min()),
        "seq_len_max": int(n.max()),
    }


def s3rec_lines(seq: pl.DataFrame) -> list[str]:
    """Renumber users and items the way S3-Rec's id_map does and render its Beauty.txt lines."""
    order = seq.select("items").explode("items")["items"]
    item_ids = {item: i + 1 for i, item in enumerate(order.unique(maintain_order=True).to_list())}
    return [
        f"{u} " + " ".join(str(item_ids[i]) for i in items)
        for u, items in enumerate(seq["items"].to_list(), start=1)
    ]


def compare_with_s3rec(seq: pl.DataFrame, path: Path) -> dict[str, int]:
    theirs = path.read_text().splitlines()
    ours = s3rec_lines(seq)
    return {
        "s3rec_lines": len(theirs),
        "our_lines": len(ours),
        "identical_lines": sum(a == b for a, b in zip(ours, theirs, strict=False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--stats", type=Path, default=ROOT / "results" / "r1" / "beauty_stats.json")
    args = parser.parse_args()
    reviews = read_reviews(args.raw / REVIEWS)
    seq = build_sequences(reviews)
    result = {**stats(seq), **kcore_violations(reviews)}
    if (args.raw / S3REC).exists():
        result["s3rec_check"] = compare_with_s3rec(seq, args.raw / S3REC)
    args.out.mkdir(parents=True, exist_ok=True)
    seq.write_parquet(args.out / "sequences.parquet")
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
