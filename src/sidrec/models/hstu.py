"""An HSTU-style sequential model (Zhai et al., "Actions Speak Louder than Words", ICML 2024).

Written from the paper's equations; no code is taken from Meta's generative-recommenders.

Each layer computes, on the layer-normalized input X,

    U, V, Q, K = split(SiLU(f1(X)))
    A = SiLU(Q K^T + rab(pos, time)) / n          pointwise aggregated attention, causal, no softmax
    Y = f2(dropout(LayerNorm(A V) * U))
    X <- X + Y

with n the window length and rab a relative attention bias shared across heads: one learned weight per
relative position plus one per log-bucket of the time gap |t_query - t_key|, where t_query is the timestamp of
the item the position predicts (the request time). Inputs are item embeddings scaled by sqrt(d) plus learned
positions; the output states and the item table are L2-normalized and scored by cosine similarity divided by
a temperature.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def time_buckets(dt: torch.Tensor, n_buckets: int) -> torch.Tensor:
    """log-scale bucket of |dt| seconds: floor(ln(max(|dt|, 1)) / 0.301), clamped to [0, n_buckets]."""
    return (torch.log(dt.abs().clamp(min=1).float()) / 0.301).long().clamp(0, n_buckets)


class RelativeBias(nn.Module):
    def __init__(self, max_len: int, n_buckets: int = 128, use_time: bool = True):
        super().__init__()
        self.max_len, self.n_buckets, self.use_time = max_len, n_buckets, use_time
        self.pos_w = nn.Parameter(torch.empty(2 * max_len - 1).normal_(std=0.02))
        self.time_w = nn.Parameter(torch.empty(n_buckets + 1).normal_(std=0.02))

    def forward(self, input_ts: torch.Tensor, query_ts: torch.Tensor) -> torch.Tensor:
        n = input_ts.shape[1]
        i = torch.arange(n, device=input_ts.device)
        pos = self.pos_w[(i[None, :] - i[:, None]) + self.max_len - 1]  # [n, n], key minus query
        if not self.use_time:
            return pos[None]
        dt = query_ts[:, :, None] - input_ts[:, None, :]  # [B, n, n]
        return pos[None] + self.time_w[time_buckets(dt, self.n_buckets)]


class HSTULayer(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.ln_in = nn.LayerNorm(d, eps=1e-6)
        self.f1 = nn.Linear(d, 4 * d)  # U, V, Q, K, each heads * dh = d
        self.ln_attn = nn.LayerNorm(d, eps=1e-6)
        self.f2 = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        u, v, q, k = F.silu(self.f1(self.ln_in(x))).split(d, dim=-1)
        q = q.view(b, n, self.heads, self.dh)
        k = k.view(b, n, self.heads, self.dh)
        v = v.view(b, n, self.heads, self.dh)
        scores = torch.einsum("bnhd,bmhd->bhnm", q, k) + bias[:, None]
        attn = F.silu(scores) / n * mask[:, None]
        av = torch.einsum("bhnm,bmhd->bnhd", attn, v).reshape(b, n, d)
        y = self.f2(F.dropout(self.ln_attn(av) * u, self.dropout, self.training))
        return x + y


class HSTU(nn.Module):
    def __init__(
        self,
        n_items: int,
        d: int,
        layers: int,
        dropout: float,
        max_len: int = 50,
        heads: int = 4,
        temperature: float = 0.05,
        n_buckets: int = 128,
        time_bias: bool = True,
    ):
        super().__init__()
        self.d, self.temperature, self.dropout = d, temperature, dropout
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        nn.init.trunc_normal_(self.item_emb.weight, std=0.02)
        self.pos_emb = nn.Embedding(max_len, d)
        nn.init.uniform_(self.pos_emb.weight, -math.sqrt(1.0 / d), math.sqrt(1.0 / d))
        self.rab = RelativeBias(max_len, n_buckets, use_time=time_bias)
        self.layers = nn.ModuleList(HSTULayer(d, heads, dropout) for _ in range(layers))

    def encode(self, inputs: torch.Tensor, input_ts: torch.Tensor, query_ts: torch.Tensor) -> torch.Tensor:
        b, n = inputs.shape
        valid = inputs > 0
        pos = torch.arange(n, device=inputs.device)
        x = self.item_emb(inputs) * math.sqrt(self.d) + self.pos_emb(pos)[None]
        x = F.dropout(x, self.dropout, self.training) * valid[..., None]
        causal = torch.ones(n, n, dtype=torch.bool, device=inputs.device).tril()
        mask = (causal[None] & valid[:, None, :]).to(x.dtype)
        bias = self.rab(input_ts, query_ts)
        for layer in self.layers:
            x = layer(x, bias, mask)
        return F.normalize(x, dim=-1)

    def logits(self, states: torch.Tensor) -> torch.Tensor:
        return states @ F.normalize(self.item_emb.weight, dim=-1).T / self.temperature
