import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from test_scale import toy  # the two-user toy timelines

from sidrec.data import scale
from sidrec.eval.metrics import target_ranks
from sidrec.models import tiger_cold
from sidrec.models.hstu_sid import prefix_rows
from sidrec.models.tiger import Layout, Trie, build_t5, constrained_beam_search
from sidrec.semantic_ids.rqvae import add_dedupe_code
from sidrec.train import scale_idrec
from sidrec.train.scale_idrec import build, catalog_ranks, output_table


def test_prefix_rows_index_the_train_items_prefixes_and_zero_the_rest():
    codes = np.array([[0, 0, 0], [1, 2, 3], [1, 2, 4], [1, 5, 3], [7, 2, 3], [1, 2, 3]])
    train = np.array([False, True, True, True, False, False])
    rows, sizes = prefix_rows(codes, train)
    assert sizes == [1, 2, 3]  # (1); (1,2) (1,5); (1,2,3) (1,2,4) (1,5,3)
    assert rows[1].tolist() == [1, 1, 1] and rows[2].tolist() == [1, 1, 2] and rows[3].tolist() == [1, 2, 3]
    assert rows[4].tolist() == [0, 0, 0]  # c1 = 7 belongs to no train item
    assert rows[5].tolist() == [1, 1, 1]  # an unseen item with a train item's whole prefix
    assert rows[0].tolist() == [0, 0, 0]


def sid_for(d: dict, seed: int = 0) -> dict:
    vocab, v = scale.id_vocab(d)
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 2, size=(len(vocab), 3))
    prefixes, sizes = prefix_rows(codes, d["train_item"])
    return {"vocab": vocab, "v": v, "prefixes": prefixes, "sizes": sizes}


def test_hstu_sid_starts_as_the_hstu_model_and_only_its_prefix_rows_differ():
    d = toy()
    sid = sid_for(d)
    v, vocab = sid["v"], torch.as_tensor(sid["vocab"])
    cfg = {"width": 8, "layers": 1, "dropout": 0.0}
    torch.manual_seed(3)
    hstu = build("hstu", cfg, v).eval()
    torch.manual_seed(3)
    model = build("hstu-sid", cfg, v, sid).eval()
    items = torch.tensor([[0, 5, 3, 7], [0, 0, 4, 9]])  # 7 and 9 are outside train
    ts = torch.tensor([[0, 1, 2, 3], [0, 0, 5, 6]])
    torch.testing.assert_close(
        model.encode(items, input_ts=ts, query_ts=ts + 1),
        hstu.encode(vocab[items], input_ts=ts, query_ts=ts + 1),
    )
    torch.testing.assert_close(output_table(model, v), output_table(hstu, v))
    with torch.no_grad():
        for table in model.item_emb.prefix:
            table.weight[1:].normal_()
    e = model.item_emb.weight
    for item in (7, 9):  # outside train: the sum of its prefix rows, which may be the zero row
        want = sum(t.weight[sid["prefixes"][item, level]] for level, t in enumerate(model.item_emb.prefix))
        torch.testing.assert_close(e[item], want)
    torch.testing.assert_close(e[0], model.item_emb.id.weight[0])  # padding: no prefix rows


def test_hstu_sid_loss_trains_train_prefixes_and_leaves_unseen_ones_at_zero():
    d = toy()
    sid = sid_for(d, seed=1)
    v = sid["v"]
    torch.manual_seed(0)
    model = build("hstu-sid", {"width": 8, "layers": 1, "dropout": 0.0}, v, sid)
    states = torch.randn(5, 8)
    loss = F.cross_entropy(states @ output_table(model, v).T, torch.tensor([0, 1, 2, 3, 4]))
    loss.backward()
    train_rows = [set(sid["prefixes"][d["train_item"], level].tolist()) for level in range(3)]
    for level, table in enumerate(model.item_emb.prefix):
        touched = {i for i in range(len(table.weight)) if table.weight.grad[i].abs().sum() > 0}
        assert touched and touched <= train_rows[level] and 0 not in touched
    assert model.item_emb.id.weight.grad[v + 1].abs().sum() == 0  # the out-of-vocabulary row


def test_catalog_ranks_rank_every_item_and_skip_the_history():
    g = torch.Generator().manual_seed(0)
    scores = torch.randn(16, 12, generator=g).round(decimals=1)
    history = torch.randint(0, 13, (16, 4), generator=g)
    target = torch.randint(1, 13, (16,), generator=g)
    full = torch.cat([torch.full((16, 1), -math.inf), scores], dim=1)
    exclude = torch.zeros(16, 13, dtype=torch.bool).scatter_(1, history, True)
    exclude[:, 0] = True
    torch.testing.assert_close(catalog_ranks(scores, history, target), target_ranks(full, target, exclude))


def test_merge_puts_candidates_where_the_epsilon_share_grows():
    items = torch.arange(1, 51)[None]
    cand = torch.tensor([[101, 102, 103, -1]])
    unseen = torch.zeros(1, 50, dtype=torch.bool)
    out = tiger_cold.merge(items, unseen, cand, 0.1)[0].tolist()
    assert [out[9], out[19], out[29]] == [101, 102, 103]  # positions 10, 20 and 30; then no candidate is left
    assert [x for x in out if x < 100] == list(range(1, 48))  # the beam's last three items drop out
    assert torch.equal(tiger_cold.merge(items, unseen, cand, 0)[0], items[0])
    # An unseen item the beam emitted itself (item 3, position 3) uses up the first slot of the share.
    unseen[0, 2] = True
    out = tiger_cold.merge(items, unseen, cand, 0.1)[0].tolist()
    assert out[9] == 10 and [out[19], out[29], out[39]] == [101, 102, 103]
    assert tiger_cold.quotas("0.3", 10) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]


def random_ids(n_items: int, sizes=(2, 2, 2), seed: int = 0):
    rng = np.random.default_rng(seed)
    codes4 = add_dedupe_code(rng.integers(0, sizes[0], size=(n_items, 3)))
    codes = np.concatenate([np.zeros((1, 4), dtype=np.int64), codes4])
    return codes, Layout((*sizes, int(codes4[:, 3].max()) + 1))


def test_cold_start_candidates_share_a_returned_prefix_and_carry_their_full_id_score():
    codes, layout = random_ids(60)
    torch.manual_seed(0)
    model = build_t5(layout, width=16, layers=1, dropout=0.0).eval()
    trie = Trie(codes, layout.level_sizes)
    tokens = torch.as_tensor(layout.item_tokens(codes))
    inputs = torch.tensor([[layout.user_offset + 3, *tokens[5].tolist(), *tokens[9].tolist()]])
    exclude = torch.tensor([[0, 5, 9]])
    unseen = torch.zeros(61, dtype=torch.bool)
    unseen[1::3] = True
    plain = constrained_beam_search(model, inputs, inputs > 0, trie, layout, beam=6, exclude=exclude)
    last = {}
    items, scores = constrained_beam_search(
        model, inputs, inputs > 0, trie, layout, beam=6, exclude=exclude, last=last
    )
    assert torch.equal(items, plain[0]) and torch.equal(scores, plain[1])  # ``last`` changes nothing
    cand, cand_scores = tiger_cold.candidates(items, last, trie.tables[-1], unseen, exclude, 60)
    got = [c for c in cand[0].tolist() if c > 0]
    prefixes = {tuple(codes[i, :3]) for i in items[0].tolist()}
    want = [
        i
        for i in range(1, 61)
        if unseen[i] and tuple(codes[i, :3]) in prefixes and i not in (5, 9) and i not in items[0].tolist()
    ]
    assert sorted(got) == sorted(want) and len(got) > 0
    # Each candidate's score is its whole ID's log-probability under the model, the beam's own score.
    n = len(got)
    dec = torch.cat([torch.zeros(n, 1, dtype=torch.long), tokens[got][:, :3]], dim=1)
    mask = (inputs > 0).expand(n, -1)
    logits = model.decode(dec, model.encode(inputs.expand(n, -1), mask), mask)
    exact = F.log_softmax(logits.float(), -1).gather(2, tokens[got][..., None]).squeeze(-1).sum(1)
    torch.testing.assert_close(cand_scores[0][:n], exact, atol=1e-4, rtol=1e-4)
    assert bool((cand_scores[0][: n - 1] >= cand_scores[0][1:n]).all())


def toy_train(monkeypatch, name: str, state_path, budget_s):
    d = toy()
    monkeypatch.setattr(scale_idrec, "sid_tables", lambda d: sid_for(d))
    cfg = {"width": 8, "layers": 1, "dropout": 0.1, "lr": 1e-2, "weight_decay": 1e-4}
    return scale_idrec.train(name, cfg, 0, d, "cpu", 6, None, state_path, budget_s)


@pytest.mark.parametrize("name", ["hstu", "hstu-sid"])
def test_a_run_resumed_after_every_epoch_equals_one_uninterrupted_run(monkeypatch, tmp_path, name):
    model_a, run_a, _, st_a, paused = toy_train(monkeypatch, name, None, None)
    assert not paused
    path, steps = tmp_path / "seed0.state.pt", 0
    while True:  # every call is a fresh model and optimizer that must pick up the saved state
        model_b, run_b, _, st_b, paused = toy_train(monkeypatch, name, path, 0.0)
        steps += 1
        if not paused:
            break
    assert steps == run_a["epochs_run"] >= 2
    strip = [{k: v for k, v in h.items() if not k.endswith("_s")} for h in run_a["history"]]
    assert strip == [{k: v for k, v in h.items() if not k.endswith("_s")} for h in run_b["history"]]
    for (k, a), b in zip(model_a.state_dict().items(), model_b.state_dict().values(), strict=True):
        assert torch.equal(a, b), k


def test_projection_from_the_r4a_cost_table(monkeypatch, tmp_path):
    from sidrec.eval import r4b_summary

    monkeypatch.setattr(r4b_summary, "RESULTS", tmp_path)  # no R4b record yet
    p = r4b_summary.projection()
    cost = r4b_summary.load(r4b_summary.R4A / "cost" / "sasrec_full_trial01.json")
    epoch = cost["history"][0]["train_s"] + cost["history"][0]["eval_s"]
    test = cost["full_valid_eval"]["seconds"] * 1_199_539 / 1_141_362
    sasrec = [r for r in p["runs"] if r["model"] == "sasrec"]
    # 11 epochs (the selected dev trial's), one test evaluation, two processes of at most an hour.
    assert [r["epochs"] for r in sasrec] == [11, 11, 11]
    assert sasrec[0]["hours"] == round((11 * epoch + test + 2 * 60) / 3600, 2)
    assert len(p["runs"]) == 13 and p["total_hours"] < p["bound_at_30_epochs_hours"]


def test_summary_tables_report_mean_and_sample_std_and_leave_empty_parts_empty():
    from sidrec.eval.r4b_summary import table

    def result(x: float, cold: bool = True) -> dict:
        part = {"recall@10": x, "recall@50": 2 * x, "ndcg@10": x / 2}
        return {"all": part, "warm": part, "cold": part if cold else None}

    t = table([result(0.1), result(0.2), result(0.3)])
    assert t["all"]["recall@50"]["mean"] == pytest.approx(0.4)
    assert t["all"]["recall@10"]["std"] == pytest.approx(0.1)  # sample standard deviation
    assert t["warm"]["ndcg@10"]["values"] == [0.05, 0.1, 0.15] and t["all"]["ndcg@10"]["n"] == 3
    no_cold = table([result(0.1, cold=False)])["cold"]
    assert no_cold == {"recall@10": None, "recall@50": None, "ndcg@10": None}
