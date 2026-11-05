"""Polars reference of the Electronics SQL pipeline (sql/electronics/03_transform.sql).

Every function here mirrors one SQL statement; the parity check compares their outputs.

    uv run python -m sidrec.data.electronics            # writes data/reference/electronics/*.parquet
"""

from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import polars as pl

CATEGORY = "Electronics"
SPLITS = ("train", "valid", "test")
# The official absolute-timestamp cutoffs of the Amazon Reviews 2023 benchmark, in milliseconds.
T1 = 1628643414042
T2 = 1658002729837
# Item-document field limits in characters, as in REC_ITEM_DOCS.
LIMITS = {"title": 200, "brand": 80, "categories": 200, "features": 800, "description": 800}
LABELS = {
    "title": "Title: ",
    "brand": "Brand: ",
    "categories": "Categories: ",
    "features": "Features: ",
    "description": "Description: ",
}

CSV_SCHEMA = {"user_id": pl.String, "parent_asin": pl.String, "rating": pl.Float64, "timestamp": pl.Int64}
META_SCHEMA = {
    "parent_asin": pl.String,
    "title": pl.String,
    "store": pl.String,
    "categories": pl.List(pl.String),
    "features": pl.List(pl.String),
    "description": pl.List(pl.String),
    "details": pl.Struct({"Brand": pl.String}),
}

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "amazon2023" / CATEGORY
REF_DIR = ROOT / "data" / "reference" / "electronics"


def clean(expr: pl.Expr, n: int) -> pl.Expr:
    """REC_CLEAN: collapse space/tab/LF/CR runs to one space, trim, cut to n characters, trim the end."""
    return (
        expr.fill_null("")
        .str.replace_all(r"[ \t\n\r]+", " ")
        .str.strip_chars(" ")
        .str.head(n)
        .str.strip_chars_end(" ")
    )


def read_interactions(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """The three split files with their split name and 1-based row number (REC_RAW_INTERACTIONS)."""
    parts = []
    for split in SPLITS:
        df = pl.read_csv(raw_dir / f"{CATEGORY}.{split}.csv.gz", schema=CSV_SCHEMA)
        parts.append(df.with_row_index("file_row", offset=1).with_columns(split=pl.lit(split)))
    return pl.concat(parts).rename({"timestamp": "ts"})


def read_meta(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    with gzip.open(raw_dir / f"meta_{CATEGORY}.jsonl.gz") as f:
        return pl.read_ndjson(f.read(), schema=META_SCHEMA)


def build_interactions(raw: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    """REC_INTERACTIONS: drop items without metadata, number each user's timeline, flag cold rows."""
    r = raw.join(meta.select("parent_asin").unique(), on="parent_asin", how="semi")
    train_items = r.filter(pl.col("split") == "train").select("parent_asin").unique().with_columns(seen=True)
    split_rank = pl.col("split").replace_strict({"train": 0, "valid": 1, "test": 2}, return_dtype=pl.Int8)
    return (
        r.with_columns(split_rank=split_rank)
        .sort("user_id", "ts", "split_rank", "file_row")
        .with_columns(pos=pl.int_range(pl.len(), dtype=pl.Int64).over("user_id"))
        .join(train_items, on="parent_asin", how="left")
        .with_columns(is_cold=(pl.col("split") != "train") & pl.col("seen").is_null())
        .select("user_id", "parent_asin", "rating", "ts", "split", "pos", "is_cold")
    )


def build_train_seq(inter: pl.DataFrame) -> pl.DataFrame:
    """REC_TRAIN_SEQ: each user's train-split items and timestamps, in timeline order."""
    return (
        inter.filter(pl.col("split") == "train")
        .sort("user_id", "pos")
        .group_by("user_id", maintain_order=True)
        .agg(items=pl.col("parent_asin"), timestamps=pl.col("ts"), n=pl.len().cast(pl.Int64))
    )


def build_eval(inter: pl.DataFrame, split: str) -> pl.DataFrame:
    """REC_EVAL for one split: each row's history is every earlier item of the user, from any split."""
    timeline = (
        inter.sort("user_id", "pos").group_by("user_id", maintain_order=True).agg(all_items=pl.col("parent_asin"))
    )
    return (
        inter.filter(pl.col("split") == split)
        .join(timeline, on="user_id")
        .select(
            "user_id",
            target=pl.col("parent_asin"),
            rating=pl.col("rating"),
            ts=pl.col("ts"),
            hist_len=pl.col("pos"),
            is_cold=pl.col("is_cold"),
            history=pl.col("all_items").list.head(pl.col("pos")),
        )
    )


def build_item_docs(meta: pl.DataFrame, inter: pl.DataFrame) -> pl.DataFrame:
    """REC_ITEM_DOCS: cleaned fields and the labelled text document of every benchmark item."""
    brand_detail = clean(pl.col("details").struct.field("Brand"), LIMITS["brand"])
    fields = meta.join(inter.select("parent_asin").unique(), on="parent_asin", how="semi").select(
        "parent_asin",
        title=clean(pl.col("title"), LIMITS["title"]),
        brand=pl.when(brand_detail != "").then(brand_detail).otherwise(clean(pl.col("store"), LIMITS["brand"])),
        categories=clean(pl.col("categories").list.join(" > "), LIMITS["categories"]),
        features=clean(pl.col("features").list.join(" "), LIMITS["features"]),
        description=clean(pl.col("description").list.join(" "), LIMITS["description"]),
    )
    parts = [pl.when(pl.col(c) != "").then(pl.lit(label) + pl.col(c)) for c, label in LABELS.items()]
    return fields.with_columns(doc=pl.concat_str(parts, separator="\n", ignore_nulls=True).fill_null("")).with_columns(
        doc_chars=pl.col("doc").str.len_chars().cast(pl.Int64)
    )


def build_all(raw_dir: Path = RAW_DIR) -> dict[str, pl.DataFrame]:
    raw = read_interactions(raw_dir)
    meta = read_meta(raw_dir)
    inter = build_interactions(raw, meta)
    return {
        "interactions": inter,
        "train_seq": build_train_seq(inter),
        "eval_valid": build_eval(inter, "valid"),
        "eval_test": build_eval(inter, "test"),
        "item_docs": build_item_docs(meta, inter),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=REF_DIR)
    args = parser.parse_args()
    t0 = time.time()
    tables = build_all(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.write_parquet(args.out / f"{name}.parquet")
        print(f"{name}: {df.height:,} rows")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
