"""The R4c two-stage cascade's list operations (docs/protocol.md, entry "2026-11-07: R4c two-stage cascade").

Lists hold catalog indices (1..N), best first, with -1 for an empty slot. Histories hold catalog indices with
0 for padding, newest item in the last column (the item-ID models' left-padded window).

- ``recency_query``: the ANN query, a recency-weighted mean of the last n history item vectors.
- ``remove_history``: an index search result without the history items, cut to a length.
- ``union``: TIGER's list and the ANN list merged at a candidate budget, by the protocol's rule.
- ``candidate_ranks``: the target's rank among a ranker's scores over the candidates.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from sidrec.eval.metrics import MISS

BACKFILL = ("ann", "tiger")


def recency_query(vectors: torch.Tensor, history: torch.Tensor, n: int, gamma: float) -> torch.Tensor:
    """[B, d] unit-length queries: sum over j < min(n, h) of gamma^j v(x_j), normalized.

    vectors: [N + 1, d], row 0 (padding) zero. history: [B, L] newest last, 0 padding. x_0 is the newest item.
    """
    h = history[:, -n:]
    power = torch.arange(h.shape[1] - 1, -1, -1, dtype=torch.float64)  # 0 for the last column
    w = (gamma**power).to(vectors.dtype)[None, :] * (h > 0)
    q = (w[..., None] * vectors[h]).sum(1)
    return F.normalize(q, dim=-1)


def remove_history(found: torch.Tensor, history: torch.Tensor, keep: int) -> torch.Tensor:
    """[B, keep]: the items of ``found`` [B, k] (catalog indices, <= 0 for no result) that are not in
    ``history`` [B, L], in their order, -1 after the last."""
    drop = (found <= 0) | (found[:, :, None] == history[:, None, :]).any(-1)
    order = torch.sort(drop.to(torch.uint8), dim=1, stable=True).indices[:, :keep]
    return found.gather(1, order).masked_fill(drop.gather(1, order), -1)


def first_unique(seq: torch.Tensor, n: int) -> torch.Tensor:
    """[B, n]: the first ``n`` distinct non-negative values of each row of ``seq``, in order, -1 after."""
    vals, perm = torch.sort(seq, dim=1, stable=True)  # equal values keep their order: the first comes first
    dup_sorted = torch.zeros_like(seq, dtype=torch.bool)
    dup_sorted[:, 1:] = vals[:, 1:] == vals[:, :-1]
    dup = torch.zeros_like(dup_sorted).scatter_(1, perm, dup_sorted)
    drop = dup | (seq < 0)
    order = torch.sort(drop.to(torch.uint8), dim=1, stable=True).indices[:, :n]
    out = seq.gather(1, order).masked_fill(drop.gather(1, order), -1)
    if out.shape[1] < n:
        out = torch.cat([out, out.new_full((len(out), n - out.shape[1]), -1)], dim=1)
    return out


def tiger_count(k_tiger: int, budget: int, full: int = 200) -> int:
    """round(s K) with s = k_tiger / full and halves rounded up, in integer arithmetic."""
    return (2 * k_tiger * budget + full) // (2 * full)


def union(tiger: torch.Tensor, ann: torch.Tensor, k_tiger: int, budget: int, backfill: str) -> torch.Tensor:
    """[B, budget] candidates: TIGER's first round(s K) items and ANN's first K - round(s K), without
    duplicates, filled up to K from the backfill source's next items, then the other source's.

    k_tiger: TIGER's items at the full budget of 200 (s = k_tiger / 200). Lists are [B, L], -1 for empty."""
    if backfill not in BACKFILL:
        raise ValueError(backfill)
    kt = tiger_count(k_tiger, budget)
    ka = budget - kt
    head = [tiger[:, :kt], ann[:, :ka]]
    rest = [ann[:, ka:], tiger[:, kt:]] if backfill == "ann" else [tiger[:, kt:], ann[:, ka:]]
    return first_unique(torch.cat(head + rest, dim=1), budget)


def hits(lists: torch.Tensor, target: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """[B] bool: whether the target is among the first ``k`` items (all if None)."""
    part = lists if k is None else lists[:, :k]
    return (part == target[:, None]).any(1)


def recall_split(hit: torch.Tensor, cold: torch.Tensor) -> dict:
    """The share of hits over all rows, warm rows and cold rows."""
    out = {"all": hit.double().mean().item()}
    for name, rows in (("warm", ~cold), ("cold", cold)):
        out[name] = hit[rows].double().mean().item() if bool(rows.any()) else None
    return out


def candidate_ranks(
    scores: torch.Tensor, cands: torch.Tensor, target: torch.Tensor, scorable: torch.Tensor
) -> torch.Tensor:
    """The target's rank among the scorable candidates; ``inf`` when it is not one of them.

    scores: [B, C] float; cands: [B, C] catalog indices; scorable: [B, C] bool, False for empty slots and
    for items the ranker cannot score (they rank below every scorable item). rank = 1 + #(higher score) +
    #(equal score and lower catalog index), the protocol's tie-break."""
    scores = scores.float().masked_fill(~scorable, -math.inf)
    present = (cands == target[:, None]) & scorable
    target_score = scores.masked_fill(~present, -math.inf).max(1).values[:, None]
    higher = (scores > target_score).sum(1)
    tied = ((scores == target_score) & scorable & (cands < target[:, None])).sum(1)
    ranks = (1 + higher + tied).float()
    return torch.where(present.any(1), ranks, torch.full_like(ranks, MISS))
