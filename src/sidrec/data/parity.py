"""Compare the Snowflake export with the polars reference, table by table and split by split.

    uv run python -m sidrec.data.parity          # writes results/r1/parity_electronics.json

For each table (and each split of the interactions table) it records the row count and the content digest
(sidrec.data.canon) of both producers, plus the count Snowflake itself reported in the check step. It also checks
the evaluation histories against the `history` column of the official timestamp_w_his files.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from sidrec.data import canon, electronics, tables

ROOT = electronics.ROOT
EXPORT_DIR = ROOT / "data" / "snowflake" / "electronics"
RESULTS = ROOT / "results" / "r1"
CHECK_KEYS = {
    "interactions/train": "interactions_train",
    "interactions/valid": "interactions_valid",
    "interactions/test": "interactions_test",
    "train_seq": "train_seq_users",
    "eval_valid": "eval_rows_valid",
    "eval_test": "eval_rows_test",
    "item_docs": "item_docs",
}


def parts(name: str, df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    if name == "interactions":
        return {f"interactions/{s}": df.filter(pl.col("split") == s) for s in electronics.SPLITS}
    return {name: df}


def official_history_match(eval_df: pl.DataFrame, raw_dir: Path, split: str) -> dict[str, int]:
    path = raw_dir / f"w_his.{electronics.CATEGORY}.{split}.csv.gz"
    official = pl.read_csv(path, schema={**electronics.CSV_SCHEMA, "history": pl.String})
    joined = eval_df.join(
        official.select(
            "user_id",
            target=pl.col("parent_asin"),
            ts=pl.col("timestamp"),
            official=pl.col("history").fill_null(""),
        ),
        on=["user_id", "target", "ts"],
        how="full",
        coalesce=True,
    )
    ours = pl.col("history").list.join(" ")
    return {
        "rows": eval_df.height,
        "official_rows": official.height,
        "unmatched_rows": joined.filter(pl.col("history").is_null() | pl.col("official").is_null()).height,
        "history_equal": joined.filter(ours == pl.col("official")).height,
    }


def run(export_dir: Path, ref_dir: Path, raw_dir: Path) -> dict:
    checks_path = RESULTS / "snowflake_checks.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    out: dict = {"tables": {}, "official_history": {}}
    for table in tables.TABLES:
        sf_df = tables.read_snowflake_export(export_dir, table)
        ref_df = tables.read_reference(ref_dir, table)
        sf_parts, ref_parts = parts(table, sf_df), parts(table, ref_df)
        for key in sf_parts:
            a, b = sf_parts[key], ref_parts[key]
            da, db = canon.digest(a), canon.digest(b)
            out["tables"][key] = {
                "rows_snowflake_export": a.height,
                "rows_reference": b.height,
                "rows_snowflake_count": checks.get(CHECK_KEYS[key]),
                "sha256_snowflake_export": da,
                "sha256_reference": db,
                "match": a.height == b.height and da == db,
            }
            print(f"{key:20s} {a.height:>12,} {'MATCH' if out['tables'][key]['match'] else 'DIFF'}", flush=True)
        if table.startswith("eval_"):
            split = table.removeprefix("eval_")
            out["official_history"][split] = official_history_match(sf_df, raw_dir, split)
    out["all_match"] = all(v["match"] for v in out["tables"].values())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, default=EXPORT_DIR)
    parser.add_argument("--reference", type=Path, default=electronics.REF_DIR)
    parser.add_argument("--raw", type=Path, default=electronics.RAW_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS / "parity_electronics.json")
    args = parser.parse_args()
    t0 = time.time()
    result = run(args.export, args.reference, args.raw)
    result["seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"all_match={result['all_match']} official_history={result['official_history']}")


if __name__ == "__main__":
    main()
