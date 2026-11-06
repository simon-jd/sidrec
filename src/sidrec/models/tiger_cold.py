"""TIGER's cold-start retrieval, as the R4b protocol entry fixes it.

The paper (Rajput et al., NeurIPS 2023; arXiv 2305.05065v3, section 4.3, "Cold-Start Recommendation"): a
semantic ID the model predicts retrieves the item with that ID; unseen items, those outside the train split,
that share its first three codes (c1, c2, c3) are added to the retrieved candidates; and a hyperparameter ε is
the largest share of unseen items in the top K. Its Fig. 5a uses ε = 0.1.

What the paper leaves open is fixed here:
- **Predicted IDs:** the items of plain TIGER's beam (width 50, constrained to the catalog, history removed).
- **Candidates:** unseen items whose three-code prefix is that of a beam item, that are neither in the user's
  history nor in the beam already. They are ordered by their own score, the sum of their four tokens'
  log-probabilities: the prefix's log-probability plus the last code's. The beam is ranked by the same score.
- **One ranked list for every K:** position j (1-based) takes the best remaining candidate while the list
  holds fewer than floor(ε j) unseen items, and the next beam item otherwise. Unseen items the beam emitted
  itself count towards the share. With ε = 0.1 the candidates take positions 10, 20, 30, 40 and 50, and the
  beam's last items drop out.
"""

from __future__ import annotations

import math
from fractions import Fraction

import torch


def _member(sorted_rows: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """[B, M] bool: whether each value occurs in its row of ``sorted_rows`` [B, E] (ascending)."""
    at = torch.searchsorted(sorted_rows, values).clamp(max=sorted_rows.shape[1] - 1)
    return sorted_rows.gather(1, at) == values


def candidates(
    items: torch.Tensor,
    last: dict,
    last_table: torch.Tensor,
    unseen: torch.Tensor,
    exclude: torch.Tensor | None,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The best ``k`` unseen candidates per row and their scores; -1 and -inf where there are fewer.

    items: [B, beam] the beam's items (-1 for an empty slot). last: what ``constrained_beam_search`` stored
    before its last level. last_table: the trie's last table, [prefix node, last code] -> item or -1.
    unseen: [N + 1] bool, items outside the train split (False for 0). exclude: [B, E] sorted history."""
    b, beam, size = last["logp"].shape
    returned = items >= 0
    # Prefixes the beam's items extend: column ``beam`` collects the empty slots and is dropped.
    used = torch.zeros(b, beam + 1, dtype=torch.bool, device=items.device)
    used.scatter_(1, last["src"].masked_fill(~returned, beam), True)
    used = used[:, :beam] & (last["node"] >= 0)
    cand = last_table[last["node"].clamp(min=0)]  # [b, beam, size]
    ok = (cand > 0) & used[..., None] & unseen[cand.clamp(min=0)]
    flat = cand.reshape(b, beam * size)
    ok = ok.reshape(b, beam * size)
    if exclude is not None:
        ok &= ~_member(exclude, flat)
    ok &= ~_member(items.sort(dim=1).values, flat)
    score = (last["score"][..., None] + last["logp"]).reshape(b, beam * size).masked_fill(~ok, -torch.inf)
    top, idx = score.topk(min(k, beam * size), dim=1)
    return flat.gather(1, idx).masked_fill(~torch.isfinite(top), -1), top


def quotas(eps, length: int) -> list[int]:
    """floor(ε j) for j = 1..length, in exact arithmetic (ε given as a decimal number or string)."""
    e = Fraction(str(eps))
    return [math.floor(e * j) for j in range(1, length + 1)]


def merge(items: torch.Tensor, item_unseen: torch.Tensor, cand: torch.Tensor, eps) -> torch.Tensor:
    """One ranked list of len(items[0]) items from the beam and the candidates, by the rule in the docstring.

    items: [B, L] the beam's items, best first; item_unseen: [B, L] whether each is outside the train split;
    cand: [B, M] candidates, best first, -1 after the last."""
    b, length = items.shape
    pad = torch.full((b, 1), -1, dtype=items.dtype, device=items.device)
    items_p = torch.cat([items, pad], dim=1)
    unseen_p = torch.cat([item_unseen, torch.zeros_like(pad, dtype=torch.bool)], dim=1)
    cand_p = torch.cat([cand, pad], dim=1)
    n_cand = (cand >= 0).sum(1)
    pb = torch.zeros(b, dtype=torch.long, device=items.device)
    pc, count = torch.zeros_like(pb), torch.zeros_like(pb)
    out = torch.empty_like(items)
    for j, quota in enumerate(quotas(eps, length)):
        take = (count < quota) & (pc < n_cand)
        from_beam = items_p.gather(1, pb[:, None]).squeeze(1)
        out[:, j] = torch.where(take, cand_p.gather(1, pc[:, None]).squeeze(1), from_beam)
        count += (take | unseen_p.gather(1, pb[:, None]).squeeze(1) & ~take).long()
        pb += (~take).long()
        pc += take.long()
    return out
