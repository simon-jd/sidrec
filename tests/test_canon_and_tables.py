import polars as pl

from sidrec.data import canon, tables


def test_digest_ignores_row_order_and_sees_changes():
    a = pl.DataFrame({"u": ["x", "y"], "h": [["a", "b"], []], "c": [True, False], "r": [3.0, None]})
    assert canon.digest(a) == canon.digest(a.reverse())
    assert canon.digest(a) == canon.digest(a.select("r", "c", "h", "u"))
    b = a.with_columns(h=pl.Series([["b", "a"], []]))
    assert canon.digest(a) != canon.digest(b)


def test_normalize_parses_snowflake_json_arrays():
    raw = pl.DataFrame(
        {
            "USER_ID": ["u1"],
            "ITEMS": ['["I2","I1"]'],
            "TIMESTAMPS": ["[1344406083000,1408995743000]"],
            "N": pl.Series([2], dtype=pl.Decimal(18, 0)),
        }
    )
    df = tables.normalize(raw, "train_seq")
    assert df.schema == pl.Schema(tables.SCHEMAS["train_seq"])
    assert df.to_dicts() == [
        {"user_id": "u1", "items": ["I2", "I1"], "timestamps": [1344406083000, 1408995743000], "n": 2}
    ]
