"""HSTU-SID: the HSTU-style model with semantic-ID prefix embeddings added to its item embedding (R4b).

Each catalog item i gets the embedding

    e(i) = E_id[id(i)] + E_1[(c1)] + E_2[(c1, c2)] + E_3[(c1, c2, c3)]

where id(i) is the item-ID row of a train item (the shared out-of-vocabulary row, kept at zero, for any other
item) and (c1), (c1, c2), (c1, c2, c3) are the prefixes of its semantic ID. Singh et al. (RecSys 2024; arXiv
2306.08121v2, section 3.3) represent an item by the sum of learned embeddings of pieces of its semantic
ID; the prefix pieces and the kept ID row follow the R4b protocol entry.

e(i) replaces the item table wherever the HSTU-style model uses it: as the input embedding and, L2-normalized
and divided by the temperature, as the scoring table. Every item therefore has an embedding, including items
outside the train split, so the model can rank them.

Each prefix table has one row per prefix that occurs among the train items, and a shared zero row (0) for
padding and for prefixes no train item has. The prefix tables start at zero, so with the same seed the model
starts as exactly the HSTU-style model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sidrec.models.hstu import HSTU

LEVELS = 3  # prefixes (c1), (c1, c2), (c1, c2, c3); the fourth code only separates collisions


def prefix_rows(
    codes: np.ndarray, train_item: np.ndarray, levels: int = LEVELS
) -> tuple[np.ndarray, list[int]]:
    """Row of each item's prefix of length 1..levels in that length's table, and the table sizes.

    codes: [N + 1, >= levels] semantic IDs, row 0 padding; codes below 256. train_item: [N + 1] bool.
    Rows 1..n of a table are the distinct prefixes of the train items in ascending code order; an item whose
    prefix no train item has, and the padding item 0, get row 0."""
    key = np.zeros(len(codes), dtype=np.int64)
    rows, sizes = np.zeros((len(codes), levels), dtype=np.int64), []
    for level in range(levels):
        key = key * 256 + codes[:, level]
        known = np.unique(key[train_item])
        at = np.searchsorted(known, key).clip(max=len(known) - 1)
        rows[:, level] = np.where(known[at] == key, at + 1, 0)
        sizes.append(len(known))
    rows[0] = 0
    return rows, sizes


class SemanticItemEmbedding(nn.Module):
    """e(i) for catalog indices i (0 is padding): the ID row plus the three prefix rows."""

    def __init__(self, vocab: torch.Tensor, prefixes: torch.Tensor, sizes: list[int], v: int, d: int):
        super().__init__()
        self.id = nn.Embedding(v + 2, d, padding_idx=0)  # 0 padding, 1..v train items, v + 1 the rest
        self.prefix = nn.ModuleList(nn.Embedding(n + 1, d, padding_idx=0) for n in sizes)
        for table in self.prefix:
            nn.init.zeros_(table.weight)
        self.register_buffer("vocab", vocab.long(), persistent=False)
        self.register_buffer("prefixes", prefixes.long(), persistent=False)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        out = self.id(self.vocab[items])
        for level, table in enumerate(self.prefix):
            out = out + table(self.prefixes[items, level])
        return out

    @property
    def weight(self) -> torch.Tensor:
        """[N + 1, d]: row j is catalog item j (row 0 is padding and is never read)."""
        return self(torch.arange(len(self.vocab), device=self.vocab.device))


class HSTUSID(HSTU):
    """The HSTU-style model over catalog indices with ``SemanticItemEmbedding`` as its item table.

    vocab: [N + 1] catalog index -> item-ID row (``scale.id_vocab``); prefixes, sizes: ``prefix_rows``."""

    def __init__(
        self,
        vocab: torch.Tensor,
        prefixes: torch.Tensor,
        sizes: list[int],
        v: int,
        d: int,
        layers: int,
        dropout: float,
        **kwargs,
    ):
        super().__init__(v + 1, d, layers, dropout, **kwargs)  # the HSTU-style model's random draws
        ids = self.item_emb
        self.item_emb = SemanticItemEmbedding(torch.as_tensor(vocab), torch.as_tensor(prefixes), sizes, v, d)
        with torch.no_grad():
            self.item_emb.id.weight.copy_(ids.weight)
            self.item_emb.id.weight[v + 1].zero_()  # items outside train: no ID row, never trained
        vocab = torch.as_tensor(vocab)
        train_items = torch.nonzero((vocab >= 1) & (vocab <= v)).squeeze(1)
        self.register_buffer("train_items", train_items, persistent=False)

    def train_table(self) -> torch.Tensor:
        """[v, d] scoring table of the train items; row j is item-ID row j + 1 (catalog order)."""
        return F.normalize(self.item_emb(self.train_items), dim=-1) / self.temperature

    def catalog_table(self) -> torch.Tensor:
        """[N, d] scoring table of every catalog item; row j is catalog item j + 1."""
        return F.normalize(self.item_emb.weight[1:], dim=-1) / self.temperature
