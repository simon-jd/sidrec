import numpy as np
import torch
import torch.nn.functional as F

from sidrec.data.loo import Beauty
from sidrec.models.t5 import T5, T5Config
from sidrec.models.tiger import USER_BUCKETS, Layout, Trie, build_t5, constrained_beam_search
from sidrec.semantic_ids.rqvae import add_dedupe_code
from sidrec.train.tiger import HISTORY, examples


def random_ids(n_items: int, sizes=(4, 4, 4), seed: int = 0):
    rng = np.random.default_rng(seed)
    codes4 = add_dedupe_code(rng.integers(0, sizes[0], size=(n_items, 3)))
    codes = np.concatenate([np.zeros((1, 4), dtype=np.int64), codes4])  # row 0 is padding
    return codes, Layout((*sizes, int(codes4[:, 3].max()) + 1))


def test_trie_tables_map_prefixes_to_items():
    codes, layout = random_ids(30)
    trie = Trie(codes, layout.level_sizes)
    for item in range(1, len(codes)):
        node = 0
        for level in range(4):
            node = int(trie.tables[level][node, codes[item, level]])
            assert node >= 0
        assert node == item
    assert int((trie.tables[3] >= 0).sum()) == 30


def tiny_model(layout: Layout) -> torch.nn.Module:
    torch.manual_seed(0)
    return build_t5(layout, width=16, layers=1, dropout=0.0).eval()


def sequence_logprob(model, inputs, item_tokens):
    """Teacher-forced log-probability of each candidate's four tokens, under a full-vocabulary softmax."""
    n = item_tokens.shape[0]
    dec = torch.cat([torch.zeros(n, 1, dtype=torch.long), item_tokens[:, :3]], dim=1)
    mask = (inputs > 0).expand(n, -1)
    logits = model.decode(dec, model.encode(inputs.expand(n, -1), mask), mask)
    return F.log_softmax(logits.float(), -1).gather(2, item_tokens[..., None]).squeeze(-1).sum(1)


def test_beam_search_is_exact_when_the_beam_covers_every_valid_id():
    codes, layout = random_ids(40)
    model = tiny_model(layout)
    trie = Trie(codes, layout.level_sizes)
    tokens = torch.as_tensor(layout.item_tokens(codes))
    inputs = torch.tensor([[layout.user_offset + 3, *tokens[5].tolist(), *tokens[9].tolist(), 0, 0, 0, 0]])
    items, scores = constrained_beam_search(model, inputs, (inputs > 0).long(), trie, layout, beam=64)
    valid = items[0][items[0] > 0]
    assert sorted(valid.tolist()) == list(range(1, 41))  # every item exactly once, nothing else
    exact = sequence_logprob(model, inputs, tokens[1:])
    torch.testing.assert_close(scores[0][: len(valid)], exact[valid - 1], atol=1e-4, rtol=1e-4)
    assert torch.all(scores[0][:-1][: len(valid) - 1] >= scores[0][1 : len(valid)])


def test_beam_search_never_emits_an_invalid_id_even_when_the_model_prefers_one():
    codes, layout = random_ids(60, sizes=(8, 8, 8), seed=1)
    model = tiny_model(layout)
    trie = Trie(codes, layout.level_sizes)
    valid = {tuple(c) for c in codes[1:]}
    # Push the model towards codes no item uses: a large bonus on every unused token of each level.
    bonus = torch.zeros(layout.vocab_size)
    for level, (offset, size) in enumerate(zip(layout.offsets, layout.level_sizes, strict=True)):
        used = set(codes[1:, level].tolist())
        bonus[[offset + c for c in range(size) if c not in used]] = 50.0
    decode = model.decode
    model.decode = lambda *args: decode(*args) + bonus
    g = torch.Generator().manual_seed(0)
    inputs = torch.randint(2, layout.user_offset, (32, 1 + 4 * 3), generator=g)
    items, scores = constrained_beam_search(model, inputs, torch.ones_like(inputs), trie, layout, beam=20)
    assert items.shape == (32, 20) and bool((items > 0).all()) and bool(torch.isfinite(scores).all())
    for row in items.tolist():
        assert len(set(row)) == 20
        assert all(tuple(codes[i]) in valid for i in row)


def test_training_examples_and_token_layout():
    seqs = [np.array([1, 2, 3, 4, 5, 6]), np.array([3, 1, 2, 5, 4])]
    data = Beauty(6, [f"i{i}" for i in range(1, 7)], ["u1", "u2"], seqs, [np.arange(6), np.arange(5)])
    codes, layout = random_ids(6)
    tokens = layout.item_tokens(codes)
    tr = examples(data, tokens, layout, "train")
    # u1 training part 1 2 3 4 -> targets 2 3 4; u2 training part 3 1 2 -> targets 1 2.
    assert tr["target"].tolist() == [2, 3, 4, 1, 2]
    assert tr["inputs"].shape == (5, 1 + 4 * HISTORY)
    first = tr["inputs"][2]
    assert layout.user_offset <= first[0] < layout.user_offset + USER_BUCKETS
    assert first[1:13].tolist() == tokens[[1, 2, 3]].reshape(-1).tolist() and not first[13:].any()
    te = examples(data, tokens, layout, "test")
    assert te["target"].tolist() == [6, 4]
    assert te["inputs"][1, 1:17].tolist() == tokens[[3, 1, 2, 5]].reshape(-1).tolist()
    assert layout.offsets == [2, 6, 10, 14] and tokens[0].tolist() == [0, 0, 0, 0]


def test_compact_t5_matches_hugging_face_t5_with_the_same_weights():
    from transformers import T5Config as HFConfig
    from transformers import T5ForConditionalGeneration

    hf_cfg = HFConfig(
        vocab_size=50,
        d_model=16,
        d_kv=8,
        d_ff=32,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=3,
        dropout_rate=0.0,
        feed_forward_proj="relu",
        decoder_start_token_id=0,
    )
    torch.manual_seed(0)
    hf = T5ForConditionalGeneration(hf_cfg).eval()
    ours = T5(T5Config(50, d_model=16, d_kv=8, d_ff=32, num_heads=3, num_layers=2, dropout=0.0)).eval()
    sd = hf.state_dict()
    names = {"shared.weight": "shared.weight"}
    for stack, n_sub in (("encoder", 2), ("decoder", 3)):
        names[f"{stack}.final_layer_norm.weight"] = f"{stack}.ln_final.weight"
        names[f"{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = (
            f"{stack}.rel_bias.weight"
        )
        for i in range(2):
            hf_b, b = f"{stack}.block.{i}.layer", f"{stack}.blocks.{i}"
            for w in "qkvo":
                names[f"{hf_b}.0.SelfAttention.{w}.weight"] = f"{b}.self_attn.{w}.weight"
                if stack == "decoder":
                    names[f"{hf_b}.1.EncDecAttention.{w}.weight"] = f"{b}.cross_attn.{w}.weight"
            names[f"{hf_b}.0.layer_norm.weight"] = f"{b}.ln_self.weight"
            if stack == "decoder":
                names[f"{hf_b}.1.layer_norm.weight"] = f"{b}.ln_cross.weight"
            names[f"{hf_b}.{n_sub - 1}.layer_norm.weight"] = f"{b}.ln_ff.weight"
            for w in ("wi", "wo"):
                names[f"{hf_b}.{n_sub - 1}.DenseReluDense.{w}.weight"] = f"{b}.ff.{w}.weight"
    ours.load_state_dict({ours_name: sd[hf_name] for hf_name, ours_name in names.items()})
    g = torch.Generator().manual_seed(1)
    inputs = torch.randint(2, 50, (4, 9), generator=g)
    inputs[1, 6:] = 0
    mask = inputs > 0
    dec = torch.randint(2, 50, (4, 5), generator=g)
    dec[:, 0] = 0
    with torch.no_grad():
        theirs = hf(input_ids=inputs, attention_mask=mask.long(), decoder_input_ids=dec).logits
        mine = ours.decode(dec, ours.encode(inputs, mask), mask)
    torch.testing.assert_close(mine, theirs, atol=1e-5, rtol=1e-5)
