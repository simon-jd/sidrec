import polars as pl

from sidrec.data import dev_subset


def test_keep_users_is_deterministic_and_about_one_in_sixteen():
    users = pl.Series([f"user{i}" for i in range(16_000)])
    kept = dev_subset.keep_users(users)
    assert set(kept["user_id"]) == set(dev_subset.keep_users(users.reverse())["user_id"])
    assert 800 < kept.height < 1200


def test_make_dev_recomputes_cold_against_the_subset_train_split():
    users = pl.Series([f"user{i}" for i in range(256)])
    kept = dev_subset.keep_users(users)["user_id"].to_list()
    dropped = next(u for u in users if u not in kept)
    a = kept[0]
    inter = pl.DataFrame(
        {
            "user_id": [a, a, dropped, dropped],
            "parent_asin": ["I1", "I2", "I2", "I3"],
            "rating": [5.0] * 4,
            "ts": [1, 2, 1, 2],
            "split": ["train", "test", "train", "test"],
            "pos": [0, 1, 0, 1],
            "is_cold": [False, False, False, True],  # full-data flags: I2 is in the full train split
        }
    )
    ev = pl.DataFrame(
        {
            "user_id": [a, dropped],
            "target": ["I2", "I3"],
            "rating": [5.0, 5.0],
            "ts": [2, 2],
            "hist_len": [1, 1],
            "is_cold": [False, True],
            "history": [["I1"], ["I2"]],
        }
    )
    src = {
        "interactions": inter,
        "train_seq": pl.DataFrame({"user_id": [a, dropped], "items": [["I1"], ["I2"]]}),
        "eval_valid": ev.clear(),
        "eval_test": ev,
        "item_docs": pl.DataFrame({"parent_asin": ["I1", "I2", "I3"], "doc": ["d1", "d2", "d3"]}),
    }
    dev = dev_subset.make_dev(src)
    assert dev["interactions"]["user_id"].unique().to_list() == [a]
    # I2 is warm in the full data but never in the subset's train split, so it is cold here.
    assert dev["eval_test"]["is_cold"].to_list() == [True]
    assert dev["interactions"].filter(pl.col("split") == "test")["is_cold"].to_list() == [True]
    assert sorted(dev["item_docs"]["parent_asin"]) == ["I1", "I2"]
