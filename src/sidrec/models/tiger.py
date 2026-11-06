"""TIGER (Rajput et al., NeurIPS 2023): a T5 encoder-decoder over semantic-ID tokens, decoded with beam search
constrained to the semantic IDs of real items.

Token layout: 0 is padding (and the decoder start), 1 is end-of-sequence, then one token range per code level,
then the 2,000 hashed user tokens. The encoder input is the user token followed by the four tokens of each
history item, oldest first; the decoder emits the next item's four tokens.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from sidrec.models.t5 import T5, T5Config

PAD, EOS = 0, 1
USER_BUCKETS = 2000


@dataclass
class Layout:
    level_sizes: tuple[int, ...]  # codes per level, e.g. (256, 256, 256, 9)

    @property
    def offsets(self) -> list[int]:
        return [2 + sum(self.level_sizes[:i]) for i in range(len(self.level_sizes))]

    @property
    def user_offset(self) -> int:
        return 2 + sum(self.level_sizes)

    @property
    def vocab_size(self) -> int:
        return self.user_offset + USER_BUCKETS

    def item_tokens(self, codes: np.ndarray) -> np.ndarray:
        """[N + 1, 4] codes (row 0 unused) -> [N + 1, 4] tokens, row 0 all padding."""
        tokens = codes + np.array(self.offsets)[None, :]
        tokens[0] = PAD
        return tokens

    def user_token(self, user_id: str) -> int:
        return self.user_offset + zlib.crc32(user_id.encode()) % USER_BUCKETS


def build_t5(layout: Layout, width: int, layers: int, dropout: float) -> T5:
    """TIGER's model: 6 heads of 64, feed-forward 1024, ReLU; width, layers and dropout from the trial."""
    config = T5Config(layout.vocab_size, d_model=width, num_layers=layers, dropout=dropout)
    return T5(config)


class Trie:
    """Valid semantic IDs as one dense transition table per level.

    tables[l][node, code] is the node reached by appending ``code`` at level l, or -1; at the last level it is
    the item index. Level-0 has the single root node 0.
    """

    def __init__(self, codes: np.ndarray, level_sizes: tuple[int, ...], device: str | torch.device = "cpu"):
        nodes: dict[tuple, int] = {(): 0}
        self.tables = []
        last = len(level_sizes) - 1
        for level, size in enumerate(level_sizes):
            table = np.full((len(nodes), size), -1, dtype=np.int64)
            children: dict[tuple, int] = {}
            for item in range(1, len(codes)):  # row 0 of codes is unused (padding)
                prefix, child = tuple(codes[item, :level]), tuple(codes[item, : level + 1])
                if level == last:
                    table[nodes[prefix], codes[item, level]] = item
                else:
                    table[nodes[prefix], codes[item, level]] = children.setdefault(child, len(children))
            self.tables.append(torch.from_numpy(table).to(device))
            nodes = children


@torch.no_grad()
def constrained_beam_search(
    model: T5,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    trie: Trie,
    layout: Layout,
    beam: int = 20,
    exclude: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``beam`` items per row, best first, and their log-probabilities. Only valid IDs can be emitted; a
    slot with no valid continuation left holds item -1 and score -inf.

    An item's score is the sum of its tokens' log-probabilities, each taken from the softmax over the whole
    token vocabulary. ``exclude`` [B, E], sorted ascending per row (0 is never an item), lists items that may
    not be emitted, such as the user's history: they are removed at the last level, so the beam fills with
    other items."""
    b = input_ids.shape[0]
    device = input_ids.device
    enc = model.encode(input_ids, attention_mask).repeat_interleave(beam, 0)
    mask = attention_mask.repeat_interleave(beam, 0)
    dec = torch.full((b * beam, 1), PAD, dtype=torch.long, device=device)
    scores = torch.full((b, beam), -torch.inf, device=device)
    scores[:, 0] = 0.0  # one live beam at the start
    node = torch.zeros(b, beam, dtype=torch.long, device=device)
    for level, (offset, size) in enumerate(zip(layout.offsets, layout.level_sizes, strict=True)):
        logits = model.decode(dec, enc, mask)[:, -1]
        logp = F.log_softmax(logits.float(), dim=-1)[:, offset : offset + size].view(b, beam, size)
        nxt = trie.tables[level][node.clamp(min=0)]  # [b, beam, size]
        nxt = nxt.masked_fill(~torch.isfinite(scores)[..., None], -1)
        if exclude is not None and level == len(layout.level_sizes) - 1:
            flat = nxt.reshape(b, beam * size)
            at = torch.searchsorted(exclude, flat).clamp(max=exclude.shape[1] - 1)
            nxt = nxt.masked_fill((exclude.gather(1, at) == flat).view(b, beam, size), -1)
        cand = (scores[..., None] + logp).masked_fill(nxt < 0, -torch.inf).view(b, beam * size)
        scores, idx = cand.topk(beam, dim=1)
        node = nxt.view(b, beam * size).gather(1, idx)
        src = idx // size
        tok = (idx % size) + offset
        prev = dec.view(b, beam, -1).gather(1, src[..., None].expand(-1, -1, dec.shape[1]))
        dec = torch.cat([prev, tok[..., None]], dim=2).view(b * beam, -1)
        node = node.masked_fill(~torch.isfinite(scores), -1)
    return node, scores
