import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sidrec.data import scale
from sidrec.eval.metrics import MISS, target_ranks
from sidrec.models.tiger import Layout, Trie, build_t5, constrained_beam_search
from sidrec.semantic_ids.rqvae import add_dedupe_code
from sidrec.train.fused_ce import fused_linear_cross_entropy
from sidrec.train.scale_idrec import vocab_ranks
from sidrec.train.scale_tiger import Inputs


def toy() -> dict[str, np.ndarray]:
    """Two users and 10 items. u0: train 5 3 8 1, valid 9, test 2. u1: train 4 6, valid 7 3, test 2."""
    items = np.array([5, 3, 8, 1, 9, 2, 4, 6, 7, 3, 2], dtype=np.int32)
    train_item = np.zeros(11, dtype=bool)
    train_item[[5, 3, 8, 1, 4, 6]] = True
    return {
        "n_items": np.int64(10),
        "offsets": np.array([0, 6, 11]),
        "items": items,
        "ts": np.arange(11, dtype=np.int64) * 1000 + 999,
        "split": np.array([0, 0, 0, 0, 1, 2, 0, 0, 1, 1, 2], dtype=np.int8),
        "train_len": np.array([4, 2], dtype=np.int32),
        "user_bucket": np.array([3, 7], dtype=np.int16),
        "train_item": train_item,
        "valid_user": np.array([0, 1, 1], dtype=np.int32),
        "valid_target": np.array([4, 8, 9]),
        "valid_cold": ~train_item[items[[4, 8, 9]]],
        "valid_order": np.array([2, 0, 1], dtype=np.int32),
    }


def test_item_id_windows_cut_from_the_timelines():
    d = toy()
    flat = scale.Flat(d, "cpu")
    vocab_np, v = scale.id_vocab(d)
    assert v == 6 and vocab_np.tolist() == [0, 1, 7, 2, 3, 4, 5, 7, 6, 7, 7]
    vocab = torch.as_tensor(vocab_np)
    tr = flat.id_inputs(scale.id_train_examples(d, "cpu"), vocab, 3, train=True)
    # u0 predicts 3 8 1 from 5 3 8; u1 has one target, 6, predicted from 4.
    assert tr["inputs"].tolist() == [[vocab_np[5], vocab_np[3], vocab_np[8]], [0, 0, vocab_np[4]]]
    assert tr["targets"].tolist() == [[vocab_np[3], vocab_np[8], vocab_np[1]], [0, 0, vocab_np[6]]]
    assert tr["query_ts"].tolist() == [[1, 2, 3], [0, 0, 7]]  # seconds of the predicted item
    # The sample is the first two rows in sha256 order (rows 2 and 0), kept in table order.
    rows, users, cold, order = scale.eval_rows(d, "valid", 2, "cpu")
    assert order.tolist() == [0, 2] and users.tolist() == [0, 1] and cold.tolist() == [True, False]
    ev = flat.id_inputs(rows, vocab, 3, train=False)
    # Row 0 (target 9): the last three of 5 3 8 1. Row 2 (target 3): 4 6 7, and 7 is outside train (OOV).
    assert ev["history"].tolist() == [[3, 8, 1], [4, 6, 7]] and ev["target"].tolist() == [9, 3]
    assert ev["inputs"].tolist() == [vocab_np[[3, 8, 1]].tolist(), [vocab_np[4], vocab_np[6], v + 1]]
    assert ev["query_ts"].tolist() == [[2, 3, 4], [7, 8, 9]]  # the last one is the target's time


def test_tiger_examples_use_every_train_position_oldest_first():
    d = toy()
    ex, users = scale.tiger_train_examples(d, "cpu")
    assert (
        ex.end.tolist() == [1, 2, 3, 7]
        and ex.start.tolist() == [0, 0, 0, 6]
        and users.tolist() == [0, 0, 0, 1]
    )
    codes = np.concatenate([np.zeros((1, 4), dtype=np.int64), np.arange(40).reshape(10, 4) % 4])
    layout = Layout((4, 4, 4, 4))
    inp = Inputs(d, layout.item_tokens(codes), layout, "cpu")
    x = inp.encoder(ex, users)
    assert x[:, 0].tolist() == [layout.user_offset + 3] * 3 + [layout.user_offset + 7]
    tok = layout.item_tokens(codes)
    assert x[2, 1:13].tolist() == tok[[5, 3, 8]].reshape(-1).tolist() and not x[2, 13:].any()
    assert inp.labels(ex)[0].tolist() == tok[3].tolist()
    rows, _, _, _ = scale.eval_rows(d, "valid", None, "cpu")
    assert inp.history(rows).tolist()[2][-3:] == [4, 6, 7] and inp.history(rows).tolist()[0][-4:] == [
        1,
        3,
        5,
        8,
    ]


def test_fused_cross_entropy_equals_the_full_softmax_loss_and_gradients():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(37, 8, generator=g, requires_grad=True)
    raw = torch.randn(50, 8, generator=g, requires_grad=True)
    y = torch.randint(0, 50, (37,), generator=g)
    for table in (lambda w: w, lambda w: F.normalize(w, dim=-1) / 0.05):  # SASRec and the HSTU-style model
        ours = fused_linear_cross_entropy(h, table(raw), y, chunk=5)
        ref = F.cross_entropy(h @ table(raw).T, y, reduction="sum")
        torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-5)
        got = torch.autograd.grad(ours / 37, (h, raw))
        want = torch.autograd.grad(ref / 37, (h, raw))
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_ranking_over_train_items_equals_the_full_catalog_ranking_with_unseen_items_last():
    g = torch.Generator().manual_seed(0)
    n = 30
    train_item = torch.zeros(n + 1, dtype=torch.bool)
    train_item[1:] = torch.rand(n, generator=g) < 0.6
    vocab = torch.full((n + 1,), int(train_item.sum()) + 1)
    v = int(train_item.sum())
    vocab[train_item] = torch.arange(1, v + 1)
    vocab[0] = 0
    scores = torch.randn(64, v, generator=g).round(decimals=1)  # rounding creates ties
    history = torch.randint(0, n + 1, (64, 6), generator=g)
    target = torch.randint(1, n + 1, (64,), generator=g)
    ours = vocab_ranks(scores, vocab[history], vocab[target], v)
    full = torch.full((64, n + 1), -math.inf)
    full[:, train_item] = scores
    exclude = torch.zeros(64, n + 1, dtype=torch.bool).scatter_(1, history, True)
    exclude[:, 0] = True
    ref = target_ranks(full, target, exclude)
    warm = train_item[target] & ~exclude.gather(1, target[:, None]).squeeze(1)
    assert torch.equal(ours[warm], ref[warm])
    assert bool((ours[~warm] == MISS).all()) and bool((ref[~warm] > v - 6).all())


def random_ids(n_items: int, sizes=(4, 4, 4), seed: int = 0):
    rng = np.random.default_rng(seed)
    codes4 = add_dedupe_code(rng.integers(0, sizes[0], size=(n_items, 3)))
    codes = np.concatenate([np.zeros((1, 4), dtype=np.int64), codes4])
    return codes, Layout((*sizes, int(codes4[:, 3].max()) + 1))


def test_beam_search_with_a_history_mask_ranks_every_other_item_exactly():
    codes, layout = random_ids(40)
    torch.manual_seed(0)
    model = build_t5(layout, width=16, layers=1, dropout=0.0).eval()
    trie = Trie(codes, layout.level_sizes)
    tokens = torch.as_tensor(layout.item_tokens(codes))
    inputs = torch.tensor([[layout.user_offset + 3, *tokens[5].tolist(), *tokens[9].tolist()]] * 2)
    exclude = torch.tensor([[0, 0, 5, 9, 17], [0, 0, 0, 0, 0]])
    items, scores = constrained_beam_search(model, inputs, inputs > 0, trie, layout, beam=64, exclude=exclude)
    free, masked = items[1][items[1] > 0], items[0][items[0] > 0]
    assert sorted(free.tolist()) == list(range(1, 41))
    assert masked.tolist() == [i for i in free.tolist() if i not in (5, 9, 17)]
    torch.testing.assert_close(
        scores[0][:37], scores[1][items[1] > 0][~torch.isin(free, torch.tensor([5, 9, 17]))]
    )


@pytest.mark.data
def test_electronics_caches_match_the_documented_counts():
    if not scale.cache_path("dev").exists() or not scale.cache_path("full").exists():
        pytest.skip("R4 input caches not built")
    dev, full = scale.describe(scale.load("dev")), scale.describe(scale.load("full"))
    assert full["interactions"] == {"train": 13_096_427, "valid": 1_161_644, "test": 1_215_465}
    assert (full["train_items"], full["valid_rows"], full["valid_cold_rows"]) == (329_445, 1_141_362, 185_482)
    assert (full["test_rows"], full["test_cold_rows"]) == (1_199_539, 535_916)
    assert (dev["users"], dev["train_items"], dev["valid_rows"], dev["valid_cold_rows"]) == (
        102_587,
        184_738,
        71_610,
        23_262,
    )


@pytest.mark.data
def test_semantic_ids_need_a_461_code_last_level():
    from sidrec.train.scale_tiger import SID, load_ids

    if not SID.exists():
        pytest.skip("R3 semantic IDs not present")
    codes, layout = load_ids()
    assert layout.level_sizes == (256, 256, 256, 461) and layout.vocab_size == 2 + 768 + 461 + 2000
    tokens = layout.item_tokens(codes)
    assert (
        tokens[1:].min() >= 2
        and tokens.max() < layout.user_offset
        and len({tuple(c) for c in codes[1:]}) == 368_228
    )
