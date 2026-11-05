"""The Electronics model-input tables: their schema, and reading them from either producer.

The Snowflake export (data/snowflake/electronics/<table>/*.parquet) and the polars reference
(data/reference/electronics/<table>.parquet) are normalized to the same column names and types here.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "interactions": {
        "user_id": pl.String(),
        "parent_asin": pl.String(),
        "rating": pl.Float64(),
        "ts": pl.Int64(),
        "split": pl.String(),
        "pos": pl.Int64(),
        "is_cold": pl.Boolean(),
    },
    "train_seq": {
        "user_id": pl.String(),
        "items": pl.List(pl.String()),
        "timestamps": pl.List(pl.Int64()),
        "n": pl.Int64(),
    },
    "eval_valid": {
        "user_id": pl.String(),
        "target": pl.String(),
        "rating": pl.Float64(),
        "ts": pl.Int64(),
        "hist_len": pl.Int64(),
        "is_cold": pl.Boolean(),
        "history": pl.List(pl.String()),
    },
    "item_docs": {
        "parent_asin": pl.String(),
        "title": pl.String(),
        "brand": pl.String(),
        "categories": pl.String(),
        "features": pl.String(),
        "description": pl.String(),
        "doc": pl.String(),
        "doc_chars": pl.Int64(),
    },
}
SCHEMAS["eval_test"] = SCHEMAS["eval_valid"]
TABLES = ("interactions", "train_seq", "eval_valid", "eval_test", "item_docs")


def normalize(df: pl.DataFrame, table: str) -> pl.DataFrame:
    """Lower-case the column names, parse JSON-text arrays, and cast to the table schema."""
    schema = SCHEMAS[table]
    df = df.rename({c: c.lower() for c in df.columns})
    exprs = []
    for name, dtype in schema.items():
        col = pl.col(name)
        if isinstance(dtype, pl.List) and df.schema[name] == pl.String:
            col = col.str.json_decode(dtype)
        exprs.append(col.cast(dtype))
    return df.select(exprs)


def read_snowflake_export(export_dir: Path, table: str) -> pl.DataFrame:
    files = sorted((export_dir / table).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files under {export_dir / table}")
    return normalize(pl.read_parquet(files), table)


def read_reference(ref_dir: Path, table: str) -> pl.DataFrame:
    return normalize(pl.read_parquet(ref_dir / f"{table}.parquet"), table)
