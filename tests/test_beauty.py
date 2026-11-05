import gzip
import json
from pathlib import Path

import pytest

from sidrec.data import beauty

ROWS = [  # file order; u1 has a same-day tie between i2 and i1
    ("u2", "i3", 100),
    ("u1", "i2", 50),
    ("u1", "i1", 50),
    ("u1", "i3", 10),
    ("u2", "i1", 90),
    ("u1", "i4", 60),
]


@pytest.fixture()
def reviews(tmp_path: Path):
    path = tmp_path / "reviews.json.gz"
    with gzip.open(path, "wt") as f:
        for u, i, t in ROWS:
            f.write(json.dumps({"reviewerID": u, "asin": i, "overall": 5.0, "unixReviewTime": t}) + "\n")
    return beauty.read_reviews(path)


def test_stable_time_sort_and_leave_one_out(reviews):
    seq = beauty.build_sequences(reviews)
    rows = {r["user_id"]: r for r in seq.to_dicts()}
    assert seq["user_id"].to_list() == ["u2", "u1"]  # order of first appearance
    assert rows["u1"]["items"] == ["i3", "i2", "i1", "i4"]
    assert rows["u1"]["train_items"] == ["i3", "i2"]
    assert (rows["u1"]["valid_target"], rows["u1"]["test_target"]) == ("i1", "i4")
    assert rows["u2"]["items"] == ["i1", "i3"] and rows["u2"]["train_items"] == []


def test_s3rec_numbering(reviews):
    lines = beauty.s3rec_lines(beauty.build_sequences(reviews))
    assert lines == ["1 1 2", "2 2 3 1 4"]


@pytest.mark.data
@pytest.mark.skipif(not (beauty.RAW_DIR / beauty.S3REC).exists(), reason="Beauty data missing")
def test_real_beauty_matches_tiger_stats_and_s3rec_file():
    reviews = beauty.read_reviews(beauty.RAW_DIR / beauty.REVIEWS)
    seq = beauty.build_sequences(reviews)
    s = beauty.stats(seq)
    assert (s["users"], s["items"], s["interactions"]) == (22363, 12101, 198502)  # TIGER Table 6
    assert s["seq_len_median"] == 6 and round(s["seq_len_mean"], 2) == 8.88
    assert beauty.kcore_violations(reviews) == {"users_below_k": 0, "items_below_k": 0}
    check = beauty.compare_with_s3rec(seq, beauty.RAW_DIR / beauty.S3REC)
    assert check["identical_lines"] == check["s3rec_lines"] == 22363
