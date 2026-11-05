import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from sidrec.data import electronics as el

SQL = Path(__file__).resolve().parents[1] / "sql" / "electronics" / "03_transform.sql"

HEADER = "user_id,parent_asin,rating,timestamp\n"
T0 = el.T1 - 1_000_000  # a train-period timestamp
SPLIT_ROWS = {
    # u1: a tie at T0 between I2 and I1 that must keep file order (I2 first).
    "train": ["u1,I2,5.0,{t0}", "u1,I1,4.0,{t0}", "u2,I1,3.0,{t1}", "u2,I9,3.0,{t2}"],
    "valid": ["u1,I3,5.0,{v}", "u3,I1,2.0,{v}"],
    "test": ["u1,I4,1.0,{t}", "u1,I1,2.0,{t2b}", "u3,I3,4.0,{t}"],
}
META = [
    {
        "parent_asin": "I1",
        "title": "  Cable\t\tUSB-C \n 2m ",
        "store": "Acme",
        "categories": ["Electronics", "Cables"],
        "features": ["fast", "long"],
        "description": [],
        "details": {"Brand": "AcmeBrand"},
    },
    {
        "parent_asin": "I2",
        "title": "x" * 250,
        "store": "Store2",
        "categories": [],
        "features": [],
        "description": ["a  b", "c"],
        "details": {},
    },
    {
        "parent_asin": "I3",
        "title": "",
        "store": None,
        "categories": [],
        "features": [],
        "description": [],
        "details": {"Brand": "  "},
    },
    {
        "parent_asin": "I4",
        "title": "Speaker",
        "store": "S4",
        "categories": ["Electronics"],
        "features": [],
        "description": ["d" * 900],
        "details": {"Brand": "B4", "Color": "red"},
    },
    {
        "parent_asin": "I8",
        "title": "not in the benchmark",
        "store": "",
        "categories": [],
        "features": [],
        "description": [],
        "details": {},
    },
]


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    ts = {
        "t0": el.T1 - 50,
        "t1": el.T1 - 40,
        "t2": el.T1 - 30,
        "v": el.T1 + 10,
        "t": el.T2 + 10,
        "t2b": el.T2 + 20,
    }
    for split, rows in SPLIT_ROWS.items():
        text = HEADER + "".join(r.format(**ts) + "\n" for r in rows)
        with gzip.open(tmp_path / f"Electronics.{split}.csv.gz", "wt") as f:
            f.write(text)
    with gzip.open(tmp_path / "meta_Electronics.jsonl.gz", "wt") as f:
        for m in META:
            f.write(json.dumps(m) + "\n")
    return tmp_path


def test_clean_collapses_trims_and_truncates():
    df = pl.DataFrame({"s": ["  a \t\n b  ", None, "abcdef", "ab   cd"]})
    out = df.select(el.clean(pl.col("s"), 3))["s"].to_list()
    assert out == ["a b", "", "abc", "ab"]


def test_interactions_order_split_and_cold(raw_dir: Path):
    t = el.build_all(raw_dir)
    inter = t["interactions"]
    # I9 has no metadata record, so its interaction is dropped.
    assert inter.height == 8
    u1 = inter.filter(pl.col("user_id") == "u1").sort("pos")
    assert u1["parent_asin"].to_list() == ["I2", "I1", "I3", "I4", "I1"]
    assert u1["pos"].to_list() == [0, 1, 2, 3, 4]
    cold = dict(zip(inter["parent_asin"] + "/" + inter["user_id"], inter["is_cold"], strict=True))
    assert cold["I3/u1"] is True  # I3 is never in train
    assert cold["I1/u3"] is False  # I1 is in train
    assert not inter.filter(pl.col("split") == "train")["is_cold"].any()


def test_train_seq_and_eval_history(raw_dir: Path):
    t = el.build_all(raw_dir)
    seq = {r["user_id"]: r["items"] for r in t["train_seq"].to_dicts()}
    assert seq == {"u1": ["I2", "I1"], "u2": ["I1"]}
    test = {(r["user_id"], r["target"]): r for r in t["eval_test"].to_dicts()}
    assert test[("u1", "I4")]["history"] == ["I2", "I1", "I3"]
    assert test[("u1", "I1")]["history"] == ["I2", "I1", "I3", "I4"]
    assert test[("u3", "I3")]["history"] == ["I1"]
    valid = {(r["user_id"], r["target"]): r for r in t["eval_valid"].to_dicts()}
    assert valid[("u3", "I1")]["hist_len"] == 0
    assert valid[("u3", "I1")]["history"] == []


def test_item_docs(raw_dir: Path):
    docs = {r["parent_asin"]: r for r in el.build_all(raw_dir)["item_docs"].to_dicts()}
    assert set(docs) == {"I1", "I2", "I3", "I4"}
    assert docs["I1"]["doc"] == (
        "Title: Cable USB-C 2m\nBrand: AcmeBrand\nCategories: Electronics > Cables\nFeatures: fast long"
    )
    assert docs["I2"]["brand"] == "Store2"  # no details.Brand: fall back to the store
    assert len(docs["I2"]["title"]) == 200
    assert docs["I2"]["description"] == "a b c"
    assert docs["I3"]["doc"] == "" and docs["I3"]["doc_chars"] == 0
    assert len(docs["I4"]["description"]) == 800
    assert docs["I4"]["doc_chars"] == len(docs["I4"]["doc"])


def test_sql_uses_the_same_field_limits():
    sql = SQL.read_text()
    assert f"v:title::STRING, {el.LIMITS['title']})" in sql
    assert f"features::ARRAY, ' '), {el.LIMITS['features']})" in sql
    assert f"description::ARRAY, ' '), {el.LIMITS['description']})" in sql
    assert f"categories::ARRAY, ' > '), {el.LIMITS['categories']})" in sql
    assert f"Brand::STRING, {el.LIMITS['brand']})" in sql
    assert f"store::STRING, {el.LIMITS['brand']})" in sql
