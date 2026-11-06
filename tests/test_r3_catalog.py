import json

import numpy as np
import pytest

from sidrec.semantic_ids import catalog
from sidrec.semantic_ids.embed_catalog import agreement, neighbour_overlap, padding, top_k
from sidrec.semantic_ids.rqvae_catalog import dedupe_train_first, neighbour_prefix_agreement, prefix_coverage
from sidrec.train import common


def test_smi_numbers_parses_a_state_line():
    state = (
        "NVIDIA GeForce RTX 4090, 595.99.02, P2, 2790 MHz, 10251 MHz, 3105 MHz, 55, 279.49 W, 450.00 W, "
        "97 %, 7746 MiB, 24564 MiB"
    )
    got = common.smi_numbers(state)
    assert got["pstate"] == "P2" and got["clocks.sm"] == 2790 and got["power.draw"] == pytest.approx(279.49)
    assert got["utilization.gpu"] == 97 and got["memory.used"] == 7746 and got["temperature.gpu"] == 55
    assert common.smi_numbers("unavailable: no nvidia-smi") == {"pstate": ""}


def test_sample_rows_is_fixed_distinct_and_sorted():
    a, b = catalog.sample_rows(1000, 50), catalog.sample_rows(1000, 50)
    assert np.array_equal(a, b) and len(set(a.tolist())) == 50 and np.all(np.diff(a) > 0)
    assert catalog.ids_sha256(["b", "a"]) != catalog.ids_sha256(["a", "b"])


def test_agreement_reports_cosine_and_largest_component_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(100, 16)).astype(np.float32)
    b = a.copy()
    b[7, 3] += 0.5
    got = agreement(a, b, chunk=32)
    assert got["max_abs_component_difference"] == pytest.approx(0.5, abs=1e-6)
    assert got["items_below_0.999"] >= 1 and got["cosine"]["p50"] == pytest.approx(1.0)
    assert agreement(a, a)["items_below_0.99"] == 0


def test_top_k_excludes_the_query_and_orders_by_score():
    emb = np.eye(4, dtype=np.float32)
    emb[1] = [0.9, 0.1, 0, 0]
    emb[2] = [0.5, 0.5, 0, 0]
    got = top_k(emb[[0]], emb, 2, np.array([0]))
    assert got.tolist() == [[1, 2]]
    assert neighbour_overlap(emb, emb, np.arange(4), k=2) == 1.0


def test_padding_counts_tokens_of_char_sorted_batches():
    texts = ["aaaa", "bb", "ccc", "d"]  # char order: aaaa, ccc, bb, d
    lengths = np.array([2, 5, 3, 1])
    got = padding(texts, lengths, batch_size=2)
    # batches (aaaa, ccc) -> 2 * 3 and (bb, d) -> 2 * 5; sorted by tokens: (bb, ccc) -> 10, (aaaa, d) -> 4
    assert got["real_tokens"] == 11 and got["padded_tokens"] == 16
    assert got["padded_tokens_if_sorted_by_tokens"] == 14 and got["padding_share"] == pytest.approx(
        1 - 11 / 16
    )


def test_fourth_code_numbers_train_items_first():
    codes3 = np.array([[1, 2, 3], [1, 2, 3], [0, 0, 0], [1, 2, 3], [1, 2, 4]])
    is_train = np.array([False, True, True, True, False])
    codes4 = dedupe_train_first(codes3, is_train)
    assert codes4[:, 3].tolist() == [2, 0, 0, 1, 0]
    assert len({tuple(c) for c in codes4}) == 5
    # Train items keep their IDs when the items outside train are dropped.
    alone = dedupe_train_first(codes3[is_train], np.ones(3, dtype=bool))
    assert np.array_equal(alone, codes4[is_train])


def test_prefix_coverage_and_neighbour_prefix_agreement():
    train = np.array([[1, 2, 3], [1, 5, 0], [4, 4, 4]])
    query = np.array([[1, 2, 9], [1, 9, 9], [7, 0, 0], [4, 4, 4]])
    assert prefix_coverage(train, query, 1) == pytest.approx(0.75)
    assert prefix_coverage(train, query, 2) == pytest.approx(0.5)
    assert prefix_coverage(train, query, 3) == pytest.approx(0.25)
    emb = np.array([[1, 0], [0.99, 0.14], [0, 1], [0.14, 0.99]], dtype=np.float32)
    codes3 = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]])
    got = neighbour_prefix_agreement(emb, codes3, np.arange(4), k=1)
    assert got["levels_1_to_1"]["top1_neighbours_sharing"] == 1.0
    assert got["levels_1_to_2"]["top1_neighbours_sharing"] == 0.5
    assert got["levels_1_to_1"]["random_pair_sharing"] == pytest.approx(0.5)


@pytest.mark.data
def test_catalog_docs_and_cold_items_match_r1_counts():
    if not catalog.DOCS_JSON.exists():
        pytest.skip("item documents not written")
    ids, docs = catalog.load_docs_json()
    assert len(ids) == len(docs) == 368_228 and ids == sorted(ids)
    record = json.loads((catalog.RESULTS / "item_docs.json").read_text())
    assert record["item_ids_sha256"] == catalog.ids_sha256(ids)
    assert record["train_items"] == 329_445
    assert (record["cold_test_rows"], record["cold_test_items"]) == (535_916, 36_881)


@pytest.mark.data
def test_semantic_ids_are_unique_and_match_their_checksum():
    path = catalog.SID_DIR / "qwen3-emb-0.6b.npy"
    if not path.exists():
        pytest.skip("semantic IDs not built")
    record = json.loads((catalog.RESULTS / "semantic_ids.json").read_text())
    assert catalog.file_sha256(path) == record["artifacts"]["codes"]["sha256"]
    codes = np.load(path)
    assert codes.shape == (368_228, 4) and len(np.unique(codes, axis=0)) == len(codes)
    assert int(codes[:, :3].max()) < 256


def test_distribution_keeps_the_mean_unrounded():
    from sidrec.semantic_ids.embed_catalog import distribution

    got = distribution(np.array([0.9991, 0.9999, 0.99995]))
    assert got["mean"] < 1.0 and got["min"] <= got["mean"] <= got["max"]
