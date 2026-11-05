import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sql_statement_splitter_keeps_function_bodies_whole():
    spec = importlib.util.spec_from_file_location("pipeline", ROOT / "scripts" / "snowflake_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stmts = module.split_statements((ROOT / "sql" / "electronics" / "01_setup.sql").read_text())
    assert len(stmts) == 5
    assert stmts[-1].startswith("CREATE OR REPLACE FUNCTION REC_CLEAN") and stmts[-1].rstrip().endswith("$$")
    for name, expected in [
        ("02_load.sql", 4),
        ("03_transform.sql", 5),
        ("04_checks.sql", 1),
        ("05_export.sql", 6),
    ]:
        assert len(module.split_statements((ROOT / "sql" / "electronics" / name).read_text())) == expected, (
            name
        )
