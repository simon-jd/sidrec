"""The Electronics item catalog for R3: item documents in a fixed order, and the train and cold-start items.

Row i of every R3 array is the i-th item in ascending ``parent_asin`` order (protocol, Changes). The tables
come from the Snowflake export (data/snowflake/electronics/), read through ``sidrec.data.tables``. Polars is
imported inside the functions that need it, so the vLLM environment can use the paths and the JSON reader.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from sidrec.train import common

EXPORT = common.ROOT / "data" / "snowflake" / "electronics"
OUT = common.ROOT / "data" / "processed" / "electronics"
DOCS_JSON = OUT / "item_docs.json"  # {"parent_asin": [...], "doc": [...]} in item order
EMB_DIR = OUT / "emb"
SID_DIR = OUT / "sid"
RESULTS = common.ROOT / "results" / "r3"


def ids_sha256(ids: list[str]) -> str:
    """sha256 of the item IDs joined by line feeds: identifies the item order of an array."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def sample_rows(n: int, size: int, seed: int = 0) -> np.ndarray:
    """``size`` distinct row indices out of ``n``, drawn with a fixed seed and returned sorted."""
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


def load_docs_json(path: Path = DOCS_JSON) -> tuple[list[str], list[str]]:
    data = json.loads(path.read_text())
    return data["parent_asin"], data["doc"]


def item_docs(export: Path = EXPORT):
    """parent_asin, title and doc of every catalog item, sorted by parent_asin."""
    from sidrec.data import tables

    df = tables.read_snowflake_export(export, "item_docs").select("parent_asin", "title", "doc")
    return df.sort("parent_asin")


def train_items(export: Path = EXPORT) -> set[str]:
    """Items with at least one train-split interaction."""
    import polars as pl

    files = sorted((export / "interactions").glob("*.parquet"))
    df = pl.read_parquet(files, columns=["PARENT_ASIN", "SPLIT"])
    return set(df.filter(pl.col("SPLIT") == "train")["PARENT_ASIN"].unique().to_list())


def cold_test_targets(export: Path = EXPORT) -> list[str]:
    """Targets of the cold-start test rows the protocol evaluates, one per row: the row's user has an
    earlier interaction, and the target never appears in the train split."""
    import polars as pl

    from sidrec.data import tables

    df = tables.read_snowflake_export(export, "eval_test")
    return df.filter((pl.col("hist_len") >= 1) & pl.col("is_cold"))["target"].to_list()


def write_docs_json(ids: list[str], docs: list[str], path: Path = DOCS_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"parent_asin": ids, "doc": docs}))
