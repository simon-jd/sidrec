"""SASRec (Kang & McAuley, ICDM 2018), in the form of the S3-Rec code behind the reference Beauty numbers.

Post-LN Transformer blocks with GELU feed-forward layers of width 4d, learned absolute positions over the
left-padded window, and the item embedding table shared between input and output. Training uses the
full-softmax cross-entropy over all items at every position instead of S3-Rec's one sampled negative.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d, eps=1e-12)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)
        self.ln2 = nn.LayerNorm(d, eps=1e-12)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        drop = self.dropout if self.training else 0.0
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        a = a.transpose(1, 2).reshape(b, n, d)
        x = self.ln1(x + F.dropout(self.out(a), drop, self.training))
        h = self.ff2(F.gelu(self.ff1(x)))
        return self.ln2(x + F.dropout(h, drop, self.training))


class SASRec(nn.Module):
    def __init__(self, n_items: int, d: int, layers: int, dropout: float, max_len: int = 50, heads: int = 2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        self.ln = nn.LayerNorm(d, eps=1e-12)
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        self.dropout = dropout
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear | nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def encode(self, inputs: torch.Tensor, **_) -> torch.Tensor:
        """[B, L] left-padded item indices -> [B, L, d] states; position i has seen inputs up to i."""
        b, n = inputs.shape
        pos = torch.arange(n, device=inputs.device)
        x = self.ln(self.item_emb(inputs) + self.pos_emb(pos)[None])
        x = F.dropout(x, self.dropout, self.training)
        valid = inputs > 0
        causal = torch.ones(n, n, dtype=torch.bool, device=inputs.device).tril()
        # Padding queries may attend to themselves so that no attention row is empty.
        mask = (causal[None] & valid[:, None, :]) | torch.eye(n, dtype=torch.bool, device=inputs.device)[None]
        mask = mask[:, None]
        for block in self.blocks:
            x = block(x, mask)
        return x

    def logits(self, states: torch.Tensor) -> torch.Tensor:
        """States [..., d] -> scores over item indices 0..N (column 0 is padding)."""
        return states @ self.item_emb.weight.T
