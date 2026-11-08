import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from test_r4b import random_ids, sid_for
from test_scale import toy  # the two-user toy timelines

from sidrec.data import scale
from sidrec.eval import r4c_cascade as r4c
from sidrec.models import cascade, tiger_cold
from sidrec.models.tiger import Trie, build_t5, constrained_beam_search
from sidrec.train import scale_idrec
from sidrec.train.scale_idrec import build, catalog_ranks
from sidrec.train.scale_tiger import Inputs


def test_recency_query_weights_the_newest_item_most_and_skips_padding():
    vectors = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    history = torch.tensor([[0, 0, 1, 2], [3, 1, 2, 1]])
    q = cascade.recency_query(vectors, history, n=3, gamma=0.5)
    want0 = F.normalize(vectors[2] + 0.5 * vectors[1], dim=0)  # the padding slot adds nothing
    want1 = F.normalize(vectors[1] + 0.5 * vectors[2] + 0.25 * vectors[1], dim=0)
    torch.testing.assert_close(q, torch.stack([want0, want1]))
    torch.testing.assert_close(cascade.recency_query(vectors, history, 1, 0.5)[1], vectors[1])
    plain = cascade.recency_query(vectors, history, 50, 1.0)[1]  # n beyond the window: the plain mean
    torch.testing.assert_close(plain, F.normalize(vectors[[3, 1, 2, 1]].sum(0), dim=0))


def test_remove_history_keeps_the_search_order_and_pads():
    found = torch.tensor([[5, 3, 0, 7, 9, 4], [2, 6, 8, 1, 3, 0]])
    history = torch.tensor([[0, 3, 9], [6, 1, 0]])
    out = cascade.remove_history(found, history, 4)
    assert out.tolist() == [[5, 7, 4, -1], [2, 8, 3, -1]]


def test_tiger_count_rounds_halves_up():
    assert [cascade.tiger_count(k, 200) for k in r4c.K_TIGER_GRID] == list(r4c.K_TIGER_GRID)
    assert [cascade.tiger_count(k, 50) for k in (25, 50, 75, 150)] == [6, 13, 19, 38]  # 6.25 12.5 18.75 37.5
    assert cascade.tiger_count(75, 100) == 38 and cascade.tiger_count(100, 100) == 50


def test_union_takes_both_shares_removes_duplicates_and_backfills():
    tiger = torch.tensor([[1, 2, 3, 4, 5, 6]])
    ann = torch.tensor([[2, 7, 8, 9, 10, 11]])
    # Budget 4 with half for TIGER: 1 2 from TIGER, 2 7 from ANN; 2 is a duplicate, so one slot is left.
    assert cascade.union(tiger, ann, 100, 4, "ann").tolist() == [[1, 2, 7, 8]]
    assert cascade.union(tiger, ann, 100, 4, "tiger").tolist() == [[1, 2, 7, 3]]
    # Empty TIGER slots are skipped; when the backfill source runs out, the other source fills.
    short = torch.tensor([[1, 2, -1, -1, -1, -1]])
    assert cascade.union(short, ann, 100, 4, "tiger").tolist() == [[1, 2, 7, 8]]
    assert cascade.union(short, ann, 150, 4, "ann").tolist() == [[1, 2, 7, 8]]  # 3 TIGER slots, one empty
    both_short = torch.tensor([[1, 2, -1]]), torch.tensor([[2, 1, -1]])
    assert cascade.union(*both_short, 100, 4, "ann").tolist() == [[1, 2, -1, -1]]
    with pytest.raises(ValueError):
        cascade.union(tiger, ann, 100, 4, "both")


def test_union_on_chunks_equals_one_pass():
    g = torch.Generator().manual_seed(0)
    tiger = torch.stack([torch.randperm(40, generator=g)[:20] + 1 for _ in range(9)])
    ann = torch.stack([torch.randperm(40, generator=g)[:20] + 1 for _ in range(9)])
    cfg = {"k_tiger": 75, "backfill": "tiger"}
    whole = cascade.union(tiger, ann, 75, 16, "tiger")
    assert torch.equal(r4c.union_on(tiger, ann, cfg, 16, "cpu", chunk=2), whole)
    assert all(len(set(r.tolist())) == 16 for r in whole)  # 16 distinct items per row


def test_candidate_ranks_equal_the_full_ranking_when_every_other_item_is_a_candidate():
    g = torch.Generator().manual_seed(0)
    scores = torch.randn(16, 12, generator=g).round(decimals=1)  # ties on purpose
    history = torch.randint(0, 13, (16, 4), generator=g)
    target = torch.randint(1, 13, (16,), generator=g)
    perm = torch.stack([torch.randperm(12, generator=g) + 1 for _ in range(16)])  # any candidate order
    cands = perm.masked_fill((perm[:, :, None] == history[:, None, :]).any(-1), -1)
    got = cascade.candidate_ranks(scores.gather(1, (cands - 1).clamp(min=0)), cands, target, cands > 0)
    torch.testing.assert_close(got, catalog_ranks(scores, history, target))
    # Fewer candidates: never a worse rank for a target that stays, a miss for one that is cut.
    keep = cands.masked_fill(perm > 8, -1)
    fewer = cascade.candidate_ranks(scores.gather(1, (keep - 1).clamp(min=0)), keep, target, keep > 0)
    assert bool((fewer[target <= 8] <= got[target <= 8]).all())
    assert bool(torch.isinf(fewer[target > 8]).all())


def test_unscorable_candidates_rank_below_and_cannot_be_hit():
    scores = torch.tensor([[0.9, 0.5, 0.7, 0.1]])
    cands = torch.tensor([[4, 2, 7, 3]])
    scorable = torch.tensor([[False, True, True, True]])  # item 4 has no ID row
    assert cascade.candidate_ranks(scores, cands, torch.tensor([7]), scorable).tolist() == [1.0]
    assert cascade.candidate_ranks(scores, cands, torch.tensor([4]), scorable).tolist() == [math.inf]
    assert cascade.candidate_ranks(scores, cands, torch.tensor([5]), scorable).tolist() == [math.inf]


@pytest.mark.parametrize("name", ["hstu", "hstu-sid"])
def test_rerank_over_every_non_history_item_reproduces_the_r4b_ranks(name):
    d = toy()
    sid = sid_for(d)
    vocab_np, v = scale.id_vocab(d)
    torch.manual_seed(0)
    model = build(name, {"width": 8, "layers": 1, "dropout": 0.0}, v, sid if name == "hstu-sid" else None)
    model.eval()
    if name == "hstu-sid":
        with torch.no_grad():
            for table in model.item_emb.prefix:
                table.weight[1:].normal_()
    flat = scale.Flat(d, "cpu")
    rows, _, _, _ = scale.eval_rows(d, "valid", None, "cpu")
    hist = flat.id_inputs(rows, torch.as_tensor(vocab_np), scale_idrec.MAX_LEN, train=False)["history"]
    items = torch.arange(1, 11).expand(len(rows), -1)
    cands = items.masked_fill((items[:, :, None] == hist[:, None, :]).any(-1), -1)
    ranks, full = r4c.rerank(model, flat, rows, cands, torch.as_tensor(vocab_np), v, full=True)
    torch.testing.assert_close(ranks, full)
    if name == "hstu":  # rows 0 and 1 have targets outside train (9 and 7): misses by construction
        assert torch.isinf(ranks[:2]).all() and torch.isfinite(ranks[2])


def test_search_returns_the_top_items_without_the_history():
    import faiss

    g = np.random.default_rng(0)
    vecs = g.normal(size=(30, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(8)
    index.add(vecs)
    vectors = torch.cat([torch.zeros(1, 8), torch.from_numpy(vecs)])
    hist = torch.tensor([[0, 4, 9, 2], [7, 7, 1, 30]])
    out = r4c.search(index, vectors, hist, 3, 0.8)
    assert out.shape == (2, r4c.LIST)
    q = cascade.recency_query(vectors, hist, 3, 0.8)
    for row in range(2):
        order = [i + 1 for i in np.argsort(-(vecs @ q[row].numpy()), kind="stable")]
        want = [i for i in order if i not in hist[row].tolist()]
        got = out[row].tolist()
        assert got[: len(want)] == want and set(got[len(want) :]) == {-1}


def test_tiger_source_is_the_beam_and_the_rule_list(monkeypatch):
    monkeypatch.setattr(r4c, "BEAM", 8)
    monkeypatch.setattr(r4c, "TIGER_CHUNK", 4)  # one chunk: batch-independent numerics
    d = toy()
    codes, layout = random_ids(10)
    torch.manual_seed(0)
    model = build_t5(layout, width=16, layers=1, dropout=0.0).eval()
    inp = Inputs(d, layout.item_tokens(codes), layout, "cpu")
    trie = Trie(codes, layout.level_sizes)
    rows, users, _, _ = scale.eval_rows(d, "valid", None, "cpu")
    unseen = torch.as_tensor(~d["train_item"])
    unseen[0] = False
    plain, ruled, empty = r4c.tiger_source(model, inp, rows, users, trie, layout, unseen, True)
    x, history = inp.encoder(rows, users), inp.history(rows)
    last = {}
    items, _ = constrained_beam_search(model, x, x > 0, trie, layout, beam=8, exclude=history, last=last)
    assert torch.equal(plain.long(), items) and empty == int((items < 0).sum())
    cand, _ = tiger_cold.candidates(items, last, trie.tables[-1], unseen, history, 8)
    assert torch.equal(ruled.long(), tiger_cold.merge(items, unseen[items.clamp(min=0)], cand, r4c.EPS))
    for row, h in zip(plain.tolist(), history.tolist(), strict=True):  # no history item, no repeat
        real = [i for i in row if i > 0]
        assert len(real) == len(set(real)) and not set(real) & set(h) - {0}


def test_grids_and_tie_breaks_follow_the_protocol():
    grid = r4c.ann_grid()
    assert len(grid) == 25 and grid[0] == (1, 1.0) and grid[1] == (2, 1.0) and grid[-1] == (50, 0.4)
    ug = r4c.union_grid()
    assert len(ug) == 28 and r4c.key(ug[0]) == "off/25/ann" and r4c.key(ug[-1]) == "on/175/tiger"
    assert "off/50/ann" in {r4c.key(c) for c in ug}  # section 2's setting is a grid point
    assert r4c.pick([0.1, 0.3, 0.3, 0.2]) == 1


def test_overlap_counts_shared_items_over_the_list_length(monkeypatch):
    monkeypatch.setattr(r4c, "LIST", 4)
    a = np.array([[1, 2, 3, 4], [5, 6, -1, -1]])
    b = np.array([[4, 3, 9, 8], [6, 5, -1, -1]])
    assert r4c.overlap(a, b, chunk=1) == pytest.approx((2 + 2) / 8)
