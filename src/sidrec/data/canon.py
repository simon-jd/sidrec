"""Order-independent content hashes of tables.

Each row is rendered as text with the columns in name order: lists are joined with spaces, booleans become 0/1,
nulls become \\N. The rows are sorted, and sha256 is taken over them. Two tables with the same rows in any order
get the same digest.
"""

from __future__ import annotations

import hashlib

import polars as pl


def canonical_rows(df: pl.DataFrame) -> pl.Series:
    parts = []
    for name in sorted(df.columns):
        dtype = df.schema[name]
        col = pl.col(name)
        if isinstance(dtype, pl.List):
            col = col.list.eval(pl.element().cast(pl.String)).list.join(" ")
        elif dtype == pl.Boolean:
            col = col.cast(pl.Int8).cast(pl.String)
        else:
            col = col.cast(pl.String)
        parts.append(col.fill_null(r"\N"))
    return df.select(pl.concat_str(parts, separator="\x1f").alias("row"))["row"]


def digest(df: pl.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(("\x1f".join(sorted(df.columns)) + "\n").encode())
    rows = canonical_rows(df).sort()
    for chunk in rows.to_frame().iter_slices(500_000):
        h.update(("\n".join(chunk["row"].to_list()) + "\n").encode())
    return h.hexdigest()
