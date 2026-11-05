"""A small, deterministic user sample of the Electronics tables for development on one GPU.

    uv run python -m sidrec.data.dev_subset      # reads the Snowflake export, writes data/dev/electronics/

A user is kept when int(sha256(user_id)[:8], 16) % 16 == 0. All of a kept user's interactions are kept, in every
split, so positions and histories are unchanged. is_cold is recomputed against the subset's own train split,
because a model tuned on the subset only sees those training items. item_docs keeps the items the kept users touch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from sidrec.data import electronics, parity, tables

DEV_DIR = electronics.ROOT / "data" / "dev" / "electronics"
BUCKETS = 16
KEEP = 0


def keep_users(user_ids: pl.Series, buckets: int = BUCKETS, keep: int = KEEP) -> pl.DataFrame:
    bucket = [int(hashlib.sha256(u.encode()).hexdigest()[:8], 16) % buckets for u in user_ids.to_list()]
    return pl.DataFrame({"user_id": user_ids, "bucket": bucket}).filter(pl.col("bucket") == keep).select("user_id")


def make_dev(src: dict[str, pl.DataFrame], buckets: int = BUCKETS, keep: int = KEEP) -> dict[str, pl.DataFrame]:
    users = keep_users(src["interactions"]["user_id"].unique().sort(), buckets, keep)
    inter = src["interactions"].join(users, on="user_id", how="semi")
    train_items = inter.filter(pl.col("split") == "train")["parent_asin"].unique().implode()
    out = {
        "interactions": inter.with_columns(
            is_cold=(pl.col("split") != "train") & ~pl.col("parent_asin").is_in(train_items)
        ),
        "train_seq": src["train_seq"].join(users, on="user_id", how="semi"),
        "item_docs": src["item_docs"].join(inter.select("parent_asin").unique(), on="parent_asin", how="semi"),
    }
    for name in ("eval_valid", "eval_test"):
        out[name] = (
            src[name]
            .join(users, on="user_id", how="semi")
            .with_columns(is_cold=~pl.col("target").is_in(train_items))
        )
    return out


def describe(dev: dict[str, pl.DataFrame]) -> dict:
    inter = dev["interactions"]
    out = {
        "users": inter["user_id"].n_unique(),
        "items": inter["parent_asin"].n_unique(),
        "train_items": inter.filter(pl.col("split") == "train")["parent_asin"].n_unique(),
        "interactions": {s: inter.filter(pl.col("split") == s).height for s in electronics.SPLITS},
    }
    for name in ("eval_valid", "eval_test"):
        df = dev[name].filter(pl.col("hist_len") > 0)
        out[name] = {"eligible_rows": df.height, "cold_rows": int(df["is_cold"].sum())}
    out["item_docs"] = dev["item_docs"].height
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, default=parity.EXPORT_DIR)
    parser.add_argument("--out", type=Path, default=DEV_DIR)
    parser.add_argument("--stats", type=Path, default=parity.RESULTS / "dev_subset_stats.json")
    args = parser.parse_args()
    src = {t: tables.read_snowflake_export(args.export, t) for t in tables.TABLES}
    dev = make_dev(src)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in dev.items():
        df.write_parquet(args.out / f"{name}.parquet")
    stats = {"rule": f"int(sha256(user_id)[:8], 16) % {BUCKETS} == {KEEP}", **describe(dev)}
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
