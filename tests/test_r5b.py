import pytest
import torch
from test_tiger import random_ids

from sidrec.models.tiger import Trie, build_t5, constrained_beam_search
from sidrec.serve.beam import LADDER, FastBeam, GraphedBeam, options, serve
from sidrec.serve.requests import WINDOW, history_mask, tiger_input


def fixture(device="cpu", n_items=40, b=4, seed=0):
    """A two-layer TIGER over 40 random semantic IDs and four requests of 1, 5, 20 and 50 history items."""
    codes, layout = random_ids(n_items, seed=seed)
    torch.manual_seed(seed)
    model = build_t5(layout, width=16, layers=2, dropout=0.0).eval().to(device)
    trie = Trie(codes, layout.level_sizes, device)
    tokens = torch.as_tensor(layout.item_tokens(codes)).to(device)
    g = torch.Generator().manual_seed(seed)
    lengths = torch.tensor([1, 5, 20, 50][:b])
    items = torch.randint(1, n_items + 1, (b, WINDOW), generator=g)
    items = items.masked_fill(torch.arange(WINDOW)[None, :] < (WINDOW - lengths)[:, None], 0).to(device)
    user = (layout.user_offset + torch.randint(0, 100, (b,), generator=g)).to(device)
    return model, trie, layout, tokens, items, user


def reference(model, trie, layout, tokens, items, user, width):
    x = tiger_input(items, user, tokens)
    return constrained_beam_search(model, x, x > 0, trie, layout, beam=width, exclude=history_mask(items))


@pytest.mark.parametrize("width", [3, 10, 64])
@pytest.mark.parametrize("step", LADDER[1:4])
def test_each_step_returns_the_reference_lists_in_fp32(step, width):
    model, trie, layout, tokens, items, user = fixture()
    want_items, want_scores = reference(model, trie, layout, tokens, items, user, width)
    beam = FastBeam(model, trie, layout, width, torch.float32, **options(step))
    got_items, got_scores = serve(beam, items, user, tokens)
    assert torch.equal(got_items, want_items)  # the same items in the same order, empty slots included
    torch.testing.assert_close(got_scores, want_scores, rtol=1e-5, atol=1e-5)
    if width == 64:  # more slots than items left after the history: the rest are empty
        assert (got_items == -1).any() and torch.isinf(got_scores[got_items == -1]).all()


def test_each_option_needs_the_ones_before_it():
    assert options("cross-once") == {"cross_once": True, "kv_cache": False, "static": False}
    assert options("static") == {"cross_once": True, "kv_cache": True, "static": True}
    model, trie, layout, *_ = fixture()
    with pytest.raises(ValueError):
        FastBeam(model, trie, layout, 4, torch.float32, cross_once=False, kv_cache=True, static=False)
    with pytest.raises(ValueError):
        options("reference")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_graphs_pad_a_batch_and_give_the_eager_results():
    model, trie, layout, tokens, items, user = fixture("cuda")
    beam = FastBeam(model, trie, layout, 10, torch.float32)
    graphed = GraphedBeam(beam, tokens, items[:1], user[:1], max_batch=4)
    assert sorted(graphed.graphs) == [1, 2, 4]
    got = graphed(items[:3], user[:3])  # padded to 4
    want = reference(model, trie, layout, tokens, items[:3], user[:3], 10)
    assert torch.equal(got[0], want[0])
    torch.testing.assert_close(got[1], want[1], rtol=1e-5, atol=1e-5)
    lazy = GraphedBeam(beam, tokens, items[:1], user[:1], max_batch=4, lazy=True)
    assert not lazy.graphs
    assert torch.equal(lazy(items[1:2], user[1:2])[0], want[0][1:2]) and sorted(lazy.graphs) == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast as served needs a GPU")
def test_explicit_dtypes_compute_what_autocast_computes():
    model, trie, layout, tokens, items, user = fixture("cuda")
    auto = FastBeam(model, trie, layout, 10, torch.bfloat16, **options("kv-cache"))
    explicit = FastBeam(model, trie, layout, 10, torch.bfloat16, **options("static"))
    a, b = serve(auto, items, user, tokens), serve(explicit, items, user, tokens)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
