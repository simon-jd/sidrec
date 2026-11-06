"""A compact T5 encoder-decoder (Raffel et al., JMLR 2020), the architecture TIGER uses.

Pre-norm blocks with RMS layer norm and no biases, ReLU feed-forward layers, unscaled dot-product attention
with T5's bucketed relative position bias (computed once per stack and shared by its layers, none in
cross-attention), and an output layer tied to the input embedding with the d_model^-0.5 rescaling of T5 v1.0.
It matches Hugging Face's T5ForConditionalGeneration with the same configuration to float precision
(tests/test_tiger.py loads the same weights into both). At TIGER's size it trains about 25% faster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class T5Config:
    vocab_size: int
    d_model: int = 128
    d_kv: int = 64
    d_ff: int = 1024
    num_heads: int = 6
    num_layers: int = 4
    dropout: float = 0.1
    buckets: int = 32
    max_distance: int = 128
    eps: float = 1e-6


def relative_bucket(rel: torch.Tensor, bidirectional: bool, buckets: int, max_distance: int) -> torch.Tensor:
    """T5's bucket of the offset key - query: exact for small offsets, log-spaced up to max_distance."""
    out = torch.zeros_like(rel)
    if bidirectional:
        buckets //= 2
        out = out + (rel > 0).long() * buckets
        n = rel.abs()
    else:
        n = (-rel).clamp(min=0)
    exact = buckets // 2
    large = (
        exact
        + (
            torch.log(n.float().clamp(min=1) / exact) / math.log(max_distance / exact) * (buckets - exact)
        ).long()
    )
    return out + torch.where(n < exact, n, large.clamp(max=buckets - 1))


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)


class Attention(nn.Module):
    def __init__(self, c: T5Config):
        super().__init__()
        inner = c.num_heads * c.d_kv
        self.q = nn.Linear(c.d_model, inner, bias=False)
        self.k = nn.Linear(c.d_model, inner, bias=False)
        self.v = nn.Linear(c.d_model, inner, bias=False)
        self.o = nn.Linear(inner, c.d_model, bias=False)
        self.h, self.dk, self.dropout = c.num_heads, c.d_kv, c.dropout

    def forward(self, x: torch.Tensor, kv: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, lq, _ = x.shape
        q = self.q(x).view(b, lq, self.h, self.dk).transpose(1, 2)
        k = self.k(kv).view(b, -1, self.h, self.dk).transpose(1, 2)
        v = self.v(kv).view(b, -1, self.h, self.dk).transpose(1, 2)
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), dropout_p=drop, scale=1.0)
        return self.o(out.transpose(1, 2).reshape(b, lq, -1))


class FeedForward(nn.Module):
    def __init__(self, c: T5Config):
        super().__init__()
        self.wi = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.wo = nn.Linear(c.d_ff, c.d_model, bias=False)
        self.dropout = c.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(F.dropout(F.relu(self.wi(x)), self.dropout, self.training))


class Block(nn.Module):
    def __init__(self, c: T5Config, cross: bool):
        super().__init__()
        self.ln_self, self.self_attn = RMSNorm(c.d_model, c.eps), Attention(c)
        self.cross = cross
        if cross:
            self.ln_cross, self.cross_attn = RMSNorm(c.d_model, c.eps), Attention(c)
        self.ln_ff, self.ff = RMSNorm(c.d_model, c.eps), FeedForward(c)
        self.dropout = c.dropout

    def forward(self, x, bias, enc=None, enc_bias=None):
        drop = lambda t: F.dropout(t, self.dropout, self.training)  # noqa: E731
        h = self.ln_self(x)
        x = x + drop(self.self_attn(h, h, bias))
        if self.cross:
            x = x + drop(self.cross_attn(self.ln_cross(x), enc, enc_bias))
        return x + drop(self.ff(self.ln_ff(x)))


class Stack(nn.Module):
    def __init__(self, c: T5Config, decoder: bool):
        super().__init__()
        self.blocks = nn.ModuleList(Block(c, cross=decoder) for _ in range(c.num_layers))
        self.rel_bias = nn.Embedding(c.buckets, c.num_heads)
        self.ln_final = RMSNorm(c.d_model, c.eps)
        self.c, self.decoder = c, decoder

    def position_bias(self, lq: int, lk: int, device) -> torch.Tensor:
        rel = torch.arange(lk, device=device)[None, :] - torch.arange(lq, device=device)[:, None]
        buckets = relative_bucket(rel, not self.decoder, self.c.buckets, self.c.max_distance)
        return self.rel_bias(buckets).permute(2, 0, 1)[None]  # [1, heads, lq, lk]

    def forward(self, x, key_mask, enc=None, enc_mask=None):
        n = x.shape[1]
        bias = self.position_bias(n, n, x.device)
        neg = torch.finfo(torch.float32).min
        if self.decoder:
            causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
            bias = bias.masked_fill(~causal, neg)
        else:
            bias = bias.masked_fill(~key_mask[:, None, None, :], neg)
        enc_bias = None
        if enc is not None:
            enc_bias = torch.zeros(enc_mask.shape, device=x.device).masked_fill(~enc_mask, neg)[
                :, None, None, :
            ]
        x = F.dropout(x, self.c.dropout, self.training)
        for block in self.blocks:
            x = block(x, bias, enc, enc_bias)
        return F.dropout(self.ln_final(x), self.c.dropout, self.training)


class T5(nn.Module):
    def __init__(self, c: T5Config):
        super().__init__()
        self.c = c
        self.shared = nn.Embedding(c.vocab_size, c.d_model)
        self.encoder = Stack(c, decoder=False)
        self.decoder = Stack(c, decoder=True)
        self._init()

    @torch.no_grad()
    def _init(self) -> None:
        """T5's initialisation (Hugging Face T5PreTrainedModel._init_weights with factor 1)."""
        c = self.c
        self.shared.weight.normal_(0.0, 1.0)
        for m in self.modules():
            if isinstance(m, Attention):
                m.q.weight.normal_(0.0, (c.d_model * c.d_kv) ** -0.5)
                m.k.weight.normal_(0.0, c.d_model**-0.5)
                m.v.weight.normal_(0.0, c.d_model**-0.5)
                m.o.weight.normal_(0.0, (c.num_heads * c.d_kv) ** -0.5)
            elif isinstance(m, FeedForward):
                m.wi.weight.normal_(0.0, c.d_model**-0.5)
                m.wo.weight.normal_(0.0, c.d_ff**-0.5)
            elif isinstance(m, Stack):
                m.rel_bias.weight.normal_(0.0, c.d_model**-0.5)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.shared(input_ids), attention_mask.bool())

    def decode(self, decoder_input_ids, enc, attention_mask) -> torch.Tensor:
        x = self.decoder(self.shared(decoder_input_ids), None, enc, attention_mask.bool())
        return (x * self.c.d_model**-0.5) @ self.shared.weight.T

    def forward(self, input_ids, attention_mask, labels):
        """Mean cross-entropy of ``labels`` [B, T], teacher-forced (decoder input: 0, labels[:, :-1])."""
        dec = F.pad(labels[:, :-1], (1, 0), value=0)
        logits = self.decode(dec, self.encode(input_ids, attention_mask), attention_mask)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
