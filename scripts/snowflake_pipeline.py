"""Run the Electronics SQL pipeline on Snowflake and fetch its Parquet export.

It runs with the interpreter of the shared portfolio Snowflake environment, which provides
snowflake-snowpark-python and `sf.session()` (warehouse PORTFOLIO_XS, database PORTFOLIO_LAB, schema GPU):

    ~/.cache/portfolio-snowflake/.venv-mljobs/bin/python scripts/snowflake_pipeline.py all

Steps, in order: setup put load transform checks export get history. `all` runs every step.
Every statement is tagged with the query tag `sidrec-r1` and logged to results/r1/snowflake_queries.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql" / "electronics"
RAW_DIR = ROOT / "data" / "raw" / "amazon2023" / "Electronics"
EXPORT_DIR = ROOT / "data" / "snowflake" / "electronics"
RESULTS = ROOT / "results" / "r1"
SF_HOME = Path(os.environ.get("SIDREC_SF_HOME", "~/.cache/portfolio-snowflake")).expanduser()
QUERY_TAG = "sidrec-r1"
STAGE = "@REC_STAGE"
RAW_FILES = [
    "Electronics.train.csv.gz",
    "Electronics.valid.csv.gz",
    "Electronics.test.csv.gz",
    "meta_Electronics.jsonl.gz",
]
TABLES = ["interactions", "train_seq", "eval_valid", "eval_test", "item_docs"]
STEPS = ["setup", "put", "load", "transform", "checks", "export", "get", "history"]


def split_statements(text: str) -> list[str]:
    """Split a SQL file on semicolons that end a line, ignoring those inside $$ ... $$ bodies."""
    out, buf, in_body = [], [], False
    for line in text.splitlines():
        stripped = line.strip()
        if not buf and (not stripped or stripped.startswith("--")):
            continue
        buf.append(line)
        in_body ^= line.count("$$") % 2 == 1
        if not in_body and stripped.endswith(";"):
            out.append("\n".join(buf).rstrip().rstrip(";"))
            buf = []
    if "".join(buf).strip():
        out.append("\n".join(buf))
    return out


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Runner:
    def __init__(self) -> None:
        sys.path.insert(0, str(SF_HOME))
        import sf  # the shared portfolio settings module; reads the key itself

        self.session = sf.session()
        self.session.sql(f"ALTER SESSION SET QUERY_TAG = '{QUERY_TAG}'").collect()
        self.session.sql("USE SCHEMA PORTFOLIO_LAB.GPU").collect()
        RESULTS.mkdir(parents=True, exist_ok=True)
        self.log = (RESULTS / "snowflake_queries.jsonl").open("a")

    def sql(self, step: str, statement: str) -> list[dict]:
        t0 = time.time()
        with self.session.query_history() as history:
            rows = [r.as_dict() for r in self.session.sql(statement).collect()]
        entry = {
            "at": now(),
            "step": step,
            "query_id": history.queries[-1].query_id if history.queries else None,
            "seconds": round(time.time() - t0, 2),
            "rows": len(rows),
            "sql": " ".join(statement.split())[:160],
        }
        self.log.write(json.dumps(entry) + "\n")
        self.log.flush()
        print(f"[{step}] {entry['seconds']:7.1f}s {entry['sql'][:100]}", flush=True)
        return rows

    def run_file(self, step: str, name: str) -> list[list[dict]]:
        return [self.sql(step, s) for s in split_statements((SQL_DIR / name).read_text())]

    def setup(self) -> None:
        self.run_file("setup", "01_setup.sql")

    def put(self) -> None:
        for name in RAW_FILES:
            t0 = time.time()
            res = self.session.file.put(
                str(RAW_DIR / name),
                f"{STAGE}/raw/electronics/",
                auto_compress=False,
                source_compression="GZIP",
                overwrite=False,
                parallel=8,
            )
            status = [(r.source, r.status, r.target_size) for r in res]
            print(f"[put] {time.time() - t0:7.1f}s {status}", flush=True)
            self.log.write(json.dumps({"at": now(), "step": "put", "file": name, "result": status}) + "\n")

    def load(self) -> None:
        self.run_file("load", "02_load.sql")

    def transform(self) -> None:
        self.run_file("transform", "03_transform.sql")

    def checks(self) -> None:
        (rows,) = self.run_file("checks", "04_checks.sql")
        checks = {r["CHECK_NAME"]: int(r["VALUE"]) for r in rows}
        (RESULTS / "snowflake_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
        bad = {k: v for k, v in checks.items() if k.startswith("must_be_zero") and v != 0}
        if bad:
            raise SystemExit(f"failed checks: {bad}")
        print(json.dumps(checks, indent=1, sort_keys=True))

    def export(self) -> None:
        self.run_file("export", "05_export.sql")

    def get(self) -> None:
        for table in TABLES:
            target = EXPORT_DIR / table
            target.mkdir(parents=True, exist_ok=True)
            for old in target.glob("*.parquet"):
                old.unlink()
            res = self.session.file.get(f"{STAGE}/export/{table}/", str(target), parallel=8)
            print(f"[get] {table}: {len(res)} files, {sum(r.size for r in res) / 1e6:.1f} MB", flush=True)

    def history(self) -> None:
        """Save this pipeline's query history (execution time is what the warehouse bills for)."""
        # Other jobs share this user, so the window starts at this pipeline's first logged statement.
        with (RESULTS / "snowflake_queries.jsonl").open() as f:
            first = min(json.loads(line)["at"] for line in f if line.strip())
        rows = self.sql(
            "history",
            f"""SELECT query_id, query_type, warehouse_name, warehouse_size, start_time, end_time,
                       execution_time, total_elapsed_time, queued_overload_time, bytes_scanned,
                       credits_used_cloud_services
                FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_USER(
                       END_TIME_RANGE_START => DATEADD('minute', -5, '{first}'::TIMESTAMP_LTZ),
                       RESULT_LIMIT => 10000))
                WHERE query_tag = '{QUERY_TAG}'
                ORDER BY start_time""",
        )
        out = [{k.lower(): (str(v) if v is not None else None) for k, v in r.items()} for r in rows]
        (RESULTS / "snowflake_query_history.json").write_text(json.dumps(out, indent=1) + "\n")
        on_wh = [r for r in rows if r["WAREHOUSE_NAME"]]
        exec_ms = sum(int(r["EXECUTION_TIME"] or 0) for r in on_wh)
        print(f"[history] {len(rows)} queries, {len(on_wh)} on a warehouse, execution {exec_ms / 1000:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("steps", nargs="+", choices=STEPS + ["all"])
    args = parser.parse_args()
    steps = STEPS if "all" in args.steps else args.steps
    runner = Runner()
    try:
        for step in steps:
            getattr(runner, step)()
    finally:
        runner.session.close()


if __name__ == "__main__":
    main()
