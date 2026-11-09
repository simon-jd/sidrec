"""TIGER's constrained beam search as the R5b server runs it.

``FastBeam`` returns what ``sidrec.models.tiger.constrained_beam_search`` (R4c's code, the reference)
returns for one beam width, with less work per request. Its options are the R5b ladder; each keeps the ones
before it:

- ``cross_once``: the decoder's cross-attention keys and values are computed once per request from the
  encoder output, instead of once per beam at every step. The beams of a request are the query rows of one
  attention.
- ``kv_cache``: each step runs the decoder on the newest token only and attends to the cached self-attention
  keys and values of the earlier tokens, which follow their beams when the beams are reordered. The first step
  runs once per request, since every beam starts from the same token.
- ``static``: what does not depend on the request is computed at load: the encoder's position bias, each
  decoder step's position bias and the weights in the serving dtype. The model then runs with explicit dtypes
  instead of autocast: the same tensors cast to the same dtypes, with an input shared by several linear layers
  cast once.

``GraphedBeam`` captures the whole serving call, from the request's items to the lists, as one CUDA graph per
power-of-two batch size.

The search is the reference's: the same trie tables, history removal, scores, candidate tensor and top-k.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sidrec.models.t5 import T5
from sidrec.models.tiger import PAD, Layout, Trie
from sidrec.serve.ranker import bucket
from sidrec.serve.requests import history_mask, tiger_input

LADDER = ("reference", "cross-once", "kv-cache", "static", "graphs")


def options(step: str) -> dict:
    """The FastBeam options of a ladder step after the reference (``graphs`` adds GraphedBeam)."""
    i = LADDER.index(step)
    if i == 0:
        raise ValueError("the reference is constrained_beam_search itself")
    return {"cross_once": i >= 1, "kv_cache": i >= 2, "static": i >= 3}


class FastBeam:
    """The reference search for one width. ``dtype`` is the serving dtype: bf16 runs as the reference does
    under bf16 autocast; fp32 runs as it does without autocast."""

    def __init__(
        self,
        model: T5,
        trie: Trie,
        layout: Layout,
        width: int,
        dtype: torch.dtype = torch.bfloat16,
        cross_once: bool = True,
        kv_cache: bool = True,
        static: bool = True,
    ):
        if (kv_cache and not cross_once) or (static and not kv_cache):
            raise ValueError("each option needs the ones before it")
        self.model, self.trie, self.layout, self.beam = model.eval(), trie, layout, width
        self.dtype, self.cross_once, self.kv_cache, self.static = dtype, cross_once, kv_cache, static
        c = model.c
        self.h, self.dk, self.scale = c.num_heads, c.d_kv, c.d_model**-0.5
        self.neg = torch.finfo(torch.float32).min
        self.levels = len(layout.level_sizes)
        self._enc_bias: dict[int, torch.Tensor] = {}
        if static:
            device = model.shared.weight.device
            with torch.no_grad():
                self.w = {m: m.weight.to(dtype) for m in model.modules() if isinstance(m, nn.Linear)}
                self.w_out = model.shared.weight.to(dtype)
                full = model.decoder.position_bias(self.levels, self.levels, device)
                # contiguous: SDPA's result depends on the mask's memory layout
                self.step_bias = [
                    full[:, :, t : t + 1, : t + 1].to(dtype).contiguous() for t in range(self.levels)
                ]

    # ------------------------------------------------------------------------------------------- model pieces

    def lin(self, m: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(self.dtype), self.w[m]) if self.static else m(x)

    def cast(self, x: torch.Tensor) -> torch.Tensor:
        """A linear layer's input, cast once for the layers that share it (autocast casts per layer)."""
        return x.to(self.dtype) if self.static else x

    def ff(self, f, x: torch.Tensor) -> torch.Tensor:
        return self.lin(f.wo, F.relu(self.lin(f.wi, x)))

    def attend(self, a, x: torch.Tensor, kv: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """``Attention.forward`` in eval mode."""
        b, lq, _ = x.shape
        q = self.lin(a.q, x).view(b, lq, self.h, self.dk).transpose(1, 2)
        k = self.lin(a.k, kv).view(b, -1, self.h, self.dk).transpose(1, 2)
        v = self.lin(a.v, kv).view(b, -1, self.h, self.dk).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), scale=1.0)
        return self.lin(a.o, out.transpose(1, 2).reshape(b, lq, -1))

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = self.model
        if not self.static:
            return m.encode(ids, mask)
        n = ids.shape[1]
        if n not in self._enc_bias:
            self._enc_bias[n] = m.encoder.position_bias(n, n, ids.device)
        bias = self._enc_bias[n].masked_fill(~mask[:, None, None, :], self.neg).to(self.dtype)
        x = m.shared(ids)
        for blk in m.encoder.blocks:
            h = self.cast(blk.ln_self(x))
            x = x + self.attend(blk.self_attn, h, h, bias)
            x = x + self.ff(blk.ff, blk.ln_ff(x))
        return m.encoder.ln_final(x)

    def cross_kv(self, enc: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Each decoder layer's cross-attention keys and values [b, heads, n, d_kv], once per request."""
        b, n, _ = enc.shape
        enc, out = self.cast(enc), []
        for blk in self.model.decoder.blocks:
            a = blk.cross_attn
            k = self.lin(a.k, enc).view(b, n, self.h, self.dk).transpose(1, 2)
            v = self.lin(a.v, enc).view(b, n, self.h, self.dk).transpose(1, 2)
            out.append((k, v))
        return out

    def cross(self, a, x: torch.Tensor, kv, bias: torch.Tensor, b: int) -> torch.Tensor:
        """Cross-attention of x [b * g, lq, d] (g beams per request) against the request's keys and values:
        the g * lq rows of a request are the query rows of one attention."""
        r, lq, _ = x.shape
        q = self.lin(a.q, x).view(b, (r // b) * lq, self.h, self.dk).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=bias.to(q.dtype), scale=1.0)
        return self.lin(a.o, out.transpose(1, 2).reshape(r, lq, -1))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.static:
            return (x * self.scale).to(self.dtype) @ self.w_out.T
        return (x * self.scale) @ self.model.shared.weight.T

    def causal_bias(self, n: int, device) -> torch.Tensor:
        """The decoder's self-attention bias [1, heads, n, n], as ``Stack.forward`` builds it."""
        bias = self.model.decoder.position_bias(n, n, device)
        causal = torch.ones(n, n, dtype=torch.bool, device=device).tril()
        return bias.masked_fill(~causal, self.neg)

    def decode_full(self, dec: torch.Tensor, kv, enc_bias: torch.Tensor, b: int) -> torch.Tensor:
        """Logits [b * beam, vocab] of the last position, recomputing the whole prefix (cross_once only)."""
        d = self.model.decoder
        bias = self.causal_bias(dec.shape[1], dec.device)
        x = self.model.shared(dec)
        for blk, layer_kv in zip(d.blocks, kv, strict=True):
            h = blk.ln_self(x)
            x = x + blk.self_attn(h, h, bias)
            x = x + self.cross(blk.cross_attn, blk.ln_cross(x), layer_kv, enc_bias, b)
            x = x + blk.ff(blk.ln_ff(x))
        return self.logits(d.ln_final(x))[:, -1]

    def decode_step(self, tok: torch.Tensor, t: int, cache, kv, enc_bias: torch.Tensor, b: int):
        """Logits [r, vocab] of the token at position t, given the cached keys and values of positions < t,
        and the cache extended by position t."""
        d = self.model.decoder
        r = tok.shape[0]
        bias = (
            self.step_bias[t] if self.static else self.causal_bias(t + 1, tok.device)[:, :, t:].contiguous()
        )
        x = self.model.shared(tok)[:, None, :]
        new = []
        for i, (blk, layer_kv) in enumerate(zip(d.blocks, kv, strict=True)):
            a = blk.self_attn
            h = self.cast(blk.ln_self(x))
            q = self.lin(a.q, h).view(r, 1, self.h, self.dk).transpose(1, 2)
            k = self.lin(a.k, h).view(r, 1, self.h, self.dk).transpose(1, 2)
            v = self.lin(a.v, h).view(r, 1, self.h, self.dk).transpose(1, 2)
            if t > 0:
                k, v = torch.cat([cache[i][0], k], dim=2), torch.cat([cache[i][1], v], dim=2)
            new.append((k, v))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), scale=1.0)
            x = x + self.lin(a.o, out.transpose(1, 2).reshape(r, 1, -1))
            x = x + self.cross(blk.cross_attn, blk.ln_cross(x), layer_kv, enc_bias, b)
            x = x + self.ff(blk.ff, blk.ln_ff(x))
        return self.logits(d.ln_final(x))[:, -1], new

    # ------------------------------------------------------------------------------------------------ search

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, exclude: torch.Tensor | None = None):
        """Items [B, width] best first and their scores, as ``constrained_beam_search(model, input_ids,
        input_ids > 0, trie, layout, width, exclude)``."""
        auto = not self.static and self.dtype != torch.float32
        with torch.autocast(input_ids.device.type, dtype=self.dtype, enabled=auto):
            return self.search(input_ids, input_ids > 0, exclude)

    def search(self, input_ids, attention_mask, exclude):
        m, beam, trie = self.model, self.beam, self.trie
        b, device = input_ids.shape[0], input_ids.device
        mask = attention_mask.bool()
        enc = self.encode(input_ids, mask)
        if self.cross_once:
            kv = self.cross_kv(enc)
            enc_bias = torch.zeros(mask.shape, device=device).masked_fill(~mask, self.neg)[:, None, None, :]
            if self.static:
                enc_bias = enc_bias.to(self.dtype)
        else:
            enc, mask = enc.repeat_interleave(beam, 0), mask.repeat_interleave(beam, 0)
        scores = torch.full((b, beam), -torch.inf, device=device)
        scores[:, 0] = 0.0  # one live beam at the start
        node = torch.zeros(b, beam, dtype=torch.long, device=device)
        if self.kv_cache:
            tok, cache = torch.full((b,), PAD, dtype=torch.long, device=device), None
        else:
            dec = torch.full((b * beam, 1), PAD, dtype=torch.long, device=device)
        last = self.levels - 1
        for level, (offset, size) in enumerate(
            zip(self.layout.offsets, self.layout.level_sizes, strict=True)
        ):
            if self.kv_cache:
                logits, cache = self.decode_step(tok, level, cache, kv, enc_bias, b)
            elif self.cross_once:
                logits = self.decode_full(dec, kv, enc_bias, b)
            else:
                logits = m.decode(dec, enc, mask)[:, -1]
            g = logits.shape[0] // b  # 1 at the first step with the cache: every beam is the start token
            logp = F.log_softmax(logits.float(), dim=-1)[:, offset : offset + size].view(b, g, size)
            nxt = trie.tables[level][node.clamp(min=0)]  # [b, beam, size]
            nxt = nxt.masked_fill(~torch.isfinite(scores)[..., None], -1)
            if exclude is not None and level == last:
                flat = nxt.reshape(b, beam * size)
                at = torch.searchsorted(exclude, flat).clamp(max=exclude.shape[1] - 1)
                nxt = nxt.masked_fill((exclude.gather(1, at) == flat).view(b, beam, size), -1)
            cand = (scores[..., None] + logp).masked_fill(nxt < 0, -torch.inf).view(b, beam * size)
            scores, idx = cand.topk(beam, dim=1)
            node = nxt.view(b, beam * size).gather(1, idx)
            src = idx // size
            tok = (idx % size) + offset
            if level < last:
                if self.kv_cache:  # each beam takes its parent's cache rows
                    rows = (torch.arange(b, device=device)[:, None] * g + src.clamp(max=g - 1)).view(-1)
                    cache = [(k.index_select(0, rows), v.index_select(0, rows)) for k, v in cache]
                    tok = tok.view(-1)
                else:
                    prev = dec.view(b, beam, -1).gather(1, src[..., None].expand(-1, -1, dec.shape[1]))
                    dec = torch.cat([prev, tok[..., None]], dim=2).view(b * beam, -1)
            node = node.masked_fill(~torch.isfinite(scores), -1)
        return node, scores


def serve(beam: FastBeam, items: torch.Tensor, user_tok: torch.Tensor, tokens: torch.Tensor):
    """The serving call, as ``beam_lists``: TIGER's lists [B, width] and their scores for requests on the
    device, history removed."""
    return beam(tiger_input(items, user_tok, tokens), history_mask(items))


class GraphedBeam:
    """CUDA graphs of ``serve``, one per power-of-two batch size up to ``max_batch``. A batch is padded to
    the next size with copies of its first request. ``items`` [1, 50] and ``user_tok`` [1] are an example
    request on the device. All sizes are captured at once, the largest first into one memory pool; with
    ``lazy``, each size is captured when it is first called, into its own pool."""

    def __init__(
        self,
        beam: FastBeam,
        tokens: torch.Tensor,
        items,
        user_tok,
        max_batch: int,
        warmup: int = 3,
        lazy: bool = False,
    ):
        if not beam.static:
            raise ValueError("graphs need the static beam")
        self.beam, self.tokens, self.warmup = beam, tokens, warmup
        self.example = (items[:1].clone(), user_tok[:1].clone())
        self.static, self.graphs, self.outputs = {}, {}, {}
        self.max_batch = bucket(max_batch)
        if not lazy:
            pool = None
            for size in reversed([1 << i for i in range(self.max_batch.bit_length())]):
                pool = self.capture(size, pool)  # the largest first, so the smaller ones fit in its pool
        torch.cuda.synchronize()

    def capture(self, size: int, pool=None):
        st = (self.example[0].expand(size, -1).clone(), self.example[1].expand(size).clone())
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                serve(self.beam, *st, self.tokens)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            out = serve(self.beam, *st, self.tokens)
        self.static[size], self.graphs[size], self.outputs[size] = st, graph, out
        return graph.pool()

    def __call__(self, items: torch.Tensor, user_tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Lists [B, width] and scores, views of the graph's output (valid until the next call)."""
        b = len(items)
        size = bucket(b)
        if size > self.max_batch:
            raise ValueError(f"batch {b} above {self.max_batch}")
        if size not in self.graphs:
            self.capture(size)
        st_items, st_user = self.static[size]
        st_items[:b].copy_(items, non_blocking=True)
        st_user[:b].copy_(user_tok, non_blocking=True)
        if size > b:
            st_items[b:].copy_(st_items[:1].expand(size - b, -1))
            st_user[b:].copy_(st_user[:1].expand(size - b))
        self.graphs[size].replay()
        lists, scores = self.outputs[size]
        return lists[:b], scores[:b]
