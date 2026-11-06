"""Full-ranking metrics as docs/protocol.md defines them.

The target's rank among the candidates is

    rank = 1 + #(items with a strictly higher score) + #(items with an equal score and a lower item index)

so ties are broken in favour of the lower item index, never in favour of the target. Items excluded from the
ranking (the user's history, the padding index) take no part; an excluded target is a miss. With one target
per user, Recall@K = HR@K = 1[rank <= K] and NDCG@K = 1 / log2(rank + 1) if rank <= K, else 0, averaged over
users.
"""

from __future__ import annotations

import math

import torch

KS = (5, 10)
MISS = math.inf


def target_ranks(
    scores: torch.Tensor, targets: torch.Tensor, exclude: torch.Tensor | None = None
) -> torch.Tensor:
    """Rank of each row's target among all columns of ``scores``.

    scores: [B, N] float, one column per item index. targets: [B] long item indices.
    exclude: optional [B, N] bool, True for items removed from the ranking.
    Returns [B] float ranks, with ``inf`` where the target itself is excluded.
    """
    scores = scores.float()
    if exclude is not None:
        scores = scores.masked_fill(exclude, -math.inf)
    target_score = scores.gather(1, targets[:, None])
    index = torch.arange(scores.shape[1], device=scores.device)
    higher = (scores > target_score).sum(1)
    tied_before = ((scores == target_score) & (index[None, :] < targets[:, None])).sum(1)
    ranks = (1 + higher + tied_before).float()
    if exclude is not None:
        ranks = ranks.masked_fill(exclude.gather(1, targets[:, None]).squeeze(1), MISS)
    return ranks


def ranks_from_lists(ranked_items: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Rank of each target in a ranked list of distinct items (position + 1), ``inf`` when absent.

    ranked_items: [B, K] long, best first; -1 marks an empty slot. targets: [B] long.
    """
    hits = ranked_items == targets[:, None]
    position = torch.arange(1, ranked_items.shape[1] + 1, device=ranked_items.device).float()
    found = hits.any(1)
    first = (hits.float() * position[None, :]).sum(1)
    return torch.where(found, first, torch.full_like(first, MISS))


def metrics_from_ranks(ranks: torch.Tensor, ks: tuple[int, ...] = KS) -> dict[str, float]:
    """Recall@K and NDCG@K averaged over all rows of ``ranks``."""
    ranks = ranks.double()
    out: dict[str, float] = {}
    for k in ks:
        hit = ranks <= k
        gain = torch.where(hit, 1.0 / torch.log2(ranks.clamp(max=1e9) + 1.0), torch.zeros_like(ranks))
        out[f"recall@{k}"] = hit.double().mean().item()
        out[f"ndcg@{k}"] = gain.mean().item()
    out["users"] = int(ranks.numel())
    return out


def split_metrics(ranks: torch.Tensor, cold: torch.Tensor, ks: tuple[int, ...] = (10, 50)) -> dict:
    """Electronics metrics (protocol, section 2): all rows, warm rows (target seen in train) and cold rows."""
    out = {"all": metrics_from_ranks(ranks, ks)}
    for name, rows in (("warm", ~cold), ("cold", cold)):
        out[name] = metrics_from_ranks(ranks[rows], ks) if bool(rows.any()) else None
    return out


def history_exclusion(histories: torch.Tensor, n_columns: int, exclude_padding: bool = True) -> torch.Tensor:
    """[B, n_columns] bool mask of the items in each row of ``histories`` (0 is padding).

    Column 0 is the padding index and is excluded too when ``exclude_padding`` is set.
    """
    mask = torch.zeros(histories.shape[0], n_columns, dtype=torch.bool, device=histories.device)
    mask.scatter_(1, histories.long(), True)
    if exclude_padding:
        mask[:, 0] = True
    else:
        mask[:, 0] = False
    return mask
