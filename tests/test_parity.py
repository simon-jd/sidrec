"""The Snowflake export and the polars reference must be identical.

Needs the downloaded data and both outputs; skipped otherwise (for example in CI).
"""

import json

import pytest

from sidrec.data import electronics, parity

needed = [
    parity.EXPORT_DIR / "item_docs",
    electronics.REF_DIR / "item_docs.parquet",
    electronics.RAW_DIR / "w_his.Electronics.test.csv.gz",
]
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not all(p.exists() for p in needed), reason="Electronics data or pipeline outputs missing"
    ),
]


def test_snowflake_export_matches_reference_and_committed_result():
    result = parity.run(parity.EXPORT_DIR, electronics.REF_DIR, electronics.RAW_DIR)
    assert result["all_match"], {k: v for k, v in result["tables"].items() if not v["match"]}
    for key, entry in result["tables"].items():
        if entry["rows_snowflake_count"] is not None:
            assert entry["rows_snowflake_count"] == entry["rows_snowflake_export"], key
    for split, h in result["official_history"].items():
        assert h["unmatched_rows"] == 0 and h["history_equal"] == h["rows"] == h["official_rows"], split
    committed = json.loads((parity.RESULTS / "parity_electronics.json").read_text())
    for key, entry in committed["tables"].items():
        assert entry["sha256_snowflake_export"] == result["tables"][key]["sha256_snowflake_export"], key
