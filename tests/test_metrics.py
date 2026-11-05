import math

import pytest
import torch

from sidrec.eval.metrics import history_exclusion, metrics_from_ranks, ranks_from_lists, target_ranks


def test_rank_counts_higher_scores_and_breaks_ties_by_lower_index():
    scores = torch.tensor(
        [
            [0.0, 0.1, 0.5, 0.3, 0.5],  # target 4 (0.5): index 2 is higher by tie-break -> rank 2
            [0.0, 0.1, 0.5, 0.3, 0.5],  # target 2 (0.5): index 4 ties but has a higher index -> rank 1
            [0.0, 0.9, 0.2, 0.2, 0.2],  # target 3: 0.9 is higher, index 2 ties before it -> rank 3
            [0.2, 0.2, 0.2, 0.2, 0.2],  # all equal, target 3 -> indices 0, 1, 2 rank above it -> rank 4
        ]
    )
    ranks = target_ranks(scores, torch.tensor([4, 2, 3, 3]))
    assert ranks.tolist() == [2.0, 1.0, 3.0, 4.0]


def test_excluded_items_do_not_count_and_an_excluded_target_is_a_miss():
    scores = torch.tensor([[9.0, 0.8, 0.9, 0.1], [9.0, 0.8, 0.9, 0.1]])
    histories = torch.tensor([[0, 2], [0, 1]])  # 0 is padding
    exclude = history_exclusion(histories, n_columns=4)
    assert exclude.tolist() == [[True, False, True, False], [True, True, False, False]]
    ranks = target_ranks(scores, torch.tensor([1, 1]), exclude)
    # row 0: the padding column (9.0) and item 2 (0.9) are excluded, so item 1 ranks first.
    assert ranks[0].item() == 1.0
    # row 1: the target itself is in the history.
    assert math.isinf(ranks[1].item())


def test_recall_and_ndcg_against_hand_computed_values():
    ranks = torch.tensor([1.0, 2.0, 5.0, 6.0, 10.0, 11.0, math.inf, 3.0])
    m = metrics_from_ranks(ranks, ks=(5, 10))
    # Recall@5: ranks 1, 2, 5, 3 -> 4/8; Recall@10 adds 6 and 10 -> 6/8.
    assert m["recall@5"] == pytest.approx(4 / 8)
    assert m["recall@10"] == pytest.approx(6 / 8)
    g = {r: 1 / math.log2(r + 1) for r in (1, 2, 3, 5, 6, 10)}
    assert m["ndcg@5"] == pytest.approx((g[1] + g[2] + g[5] + g[3]) / 8)
    assert m["ndcg@10"] == pytest.approx((g[1] + g[2] + g[5] + g[6] + g[10] + g[3]) / 8)
    assert g[1] == 1.0 and g[3] == pytest.approx(0.5)
    assert m["users"] == 8


def test_ranks_from_ranked_lists():
    lists = torch.tensor([[7, 3, 9], [5, 6, -1], [1, 2, 3]])
    ranks = ranks_from_lists(lists, torch.tensor([9, 4, 1]))
    assert ranks[0].item() == 3.0 and math.isinf(ranks[1].item()) and ranks[2].item() == 1.0


def test_score_and_list_paths_agree_on_a_random_case():
    g = torch.Generator().manual_seed(0)
    scores = torch.randn(64, 50, generator=g)
    targets = torch.randint(1, 50, (64,), generator=g)
    exclude = history_exclusion(torch.zeros(64, 1, dtype=torch.long), 50)
    ranks = target_ranks(scores, targets, exclude)
    masked = scores.masked_fill(exclude, -math.inf)
    order = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :10]
    top10 = torch.where(ranks <= 10, ranks, torch.full_like(ranks, math.inf))
    assert torch.equal(top10, ranks_from_lists(order, targets))
