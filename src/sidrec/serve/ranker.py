"""HSTU-SID as the R5 server runs it.

``RankerStep`` computes, for a batch of requests already on the device, the state ``HSTU.encode`` gives at the
newest column and its scores over the request's candidates, in fp32 or bf16:

- **no cache, or miss:** the full 50-row encoder. With the cache, it also writes, per layer, the keys and
  values of the 49 older rows of the window into the request's slot, with their times and padding.
- **hit:** the newest row only, attending to the cached keys and values.

The older rows' keys and values do not depend on the request time: the attention is causal and each older
row's time bias reads the time of the item after it. The newest row's bias reads the request time, so its
output, and from layer 2 on its keys and values, change with it; that row is always recomputed.

Scores are the fp32 dot product of the state with R4b's fp32 scoring table, computed once at load. ``Graphed``
replays a step from CUDA graphs, one per power-of-two batch size.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from sidrec.models.hstu import time_buckets
from sidrec.models.hstu_sid import HSTUSID
from sidrec.serve.requests import CANDS, WINDOW, query_times

TOP = 10
MODES = (None, "miss", "hit")


def bucket(b: int) -> int:
    """The smallest power of two at least ``b``."""
    return 1 << (b - 1).bit_length()


def _project(layer, x: torch.Tensor, d: int):
    return F.silu(layer.f1(layer.ln_in(x))).split(d, dim=-1)  # u, v, q, k


def _attend(layer, x, u, q, k, v, bias, mask, n: int) -> torch.Tensor:
    """``HSTULayer.forward`` (eval) for the query rows ``x`` [B, r, d] against keys and values of m rows.

    bias, mask: [B, r, m]; n: the window length, which divides the attention as in training."""
    b, r, d = x.shape
    q = q.reshape(b, r, layer.heads, layer.dh)
    k = k.reshape(b, -1, layer.heads, layer.dh)
    v = v.reshape(b, -1, layer.heads, layer.dh)
    scores = torch.einsum("bnhd,bmhd->bhnm", q, k) + bias[:, None]
    attn = F.silu(scores) / n * mask[:, None]
    av = torch.einsum("bhnm,bmhd->bnhd", attn, v).reshape(b, r, d)
    return x + layer.f2(layer.ln_attn(av) * u)


class HistoryCache:
    """Per slot: the keys and values [layers, 49, d] of the older rows, their times and their padding.

    Slot ``dummy`` (the last) takes the padded rows of a CUDA-graph batch."""

    def __init__(self, slots: int, layers: int, d: int, dtype, device):
        self.dummy = slots
        shape = (slots + 1, layers, WINDOW - 1, d)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.ts = torch.zeros(slots + 1, WINDOW - 1, dtype=torch.long, device=device)
        self.valid = torch.zeros(slots + 1, WINDOW - 1, dtype=torch.bool, device=device)


class RankerStep:
    def __init__(self, model: HSTUSID, dtype=torch.float32, mode: str | None = None, cache=None):
        if mode not in MODES or (mode is not None and cache is None):
            raise ValueError(f"mode {mode!r} needs a cache" if mode in MODES else f"mode {mode!r}")
        model = model.eval()
        with torch.no_grad():
            table = model.catalog_table().float()  # row j: catalog item j + 1
        self.table = torch.cat([table.new_zeros(1, table.shape[1]), table])  # row j: catalog item j
        self.model = copy.deepcopy(model).to(dtype).eval()
        self.dtype, self.mode, self.cache = dtype, mode, cache
        self.d, self.layers = model.d, len(model.layers)
        self.dummy = cache.dummy if cache is not None else None

    def with_mode(self, mode: str | None) -> RankerStep:
        """The same weights and cache in another mode."""
        other = copy.copy(self)
        other.mode = mode
        return other

    def spec(self) -> dict[str, tuple[tuple, torch.dtype]]:
        """Per-row shape and dtype of each input of ``__call__``."""
        long = torch.long
        if self.mode == "hit":
            names = {"slots": (), "item": (), "item_ts": (), "req_ts": (), "cands": (CANDS,)}
        else:
            names = {"items": (WINDOW,), "ts": (WINDOW,), "req_ts": (), "cands": (CANDS,)}
            if self.mode == "miss":
                names["slots"] = ()
        return {k: (shape, long) for k, shape in names.items()}

    def _embed(self, items: torch.Tensor, pos) -> torch.Tensor:
        m = self.model
        return m.item_emb(items) * math.sqrt(m.d) + m.pos_emb.weight[pos]

    def encode_full(self, items, input_ts, query_ts):
        """The newest column's state [B, d] and the older rows' keys and values [B, layers, n - 1, d]."""
        m, n = self.model, items.shape[1]
        valid = items > 0
        x = self._embed(items, slice(0, n)) * valid[..., None]
        causal = torch.ones(n, n, dtype=torch.bool, device=items.device).tril()
        mask = (causal[None] & valid[:, None, :]).to(x.dtype)
        bias = m.rab(input_ts, query_ts)
        ks, vs = [], []
        for i, layer in enumerate(m.layers):
            u, v, q, k = _project(layer, x, self.d)
            ks.append(k[:, :-1])
            vs.append(v[:, :-1])
            if i < self.layers - 1:
                x = _attend(layer, x, u, q, k, v, bias, mask, n)
            else:  # only the newest row is read after the last layer
                x = _attend(layer, x[:, -1:], u[:, -1:], q[:, -1:], k, v, bias[:, -1:], mask[:, -1:], n)
        return F.normalize(x[:, -1], dim=-1), torch.stack(ks, 1), torch.stack(vs, 1)

    def encode_hit(self, slots, item, item_ts, req_ts) -> torch.Tensor:
        """The newest column's state [B, d] from the cached older rows and the newest item."""
        m, c, n = self.model, self.cache, WINDOW
        rab = m.rab
        new = (item > 0)[:, None]
        x = self._embed(item[:, None], n - 1) * new[..., None]
        input_ts = torch.cat([c.ts[slots], item_ts[:, None]], dim=1)
        mask = torch.cat([c.valid[slots], new], dim=1)[:, None, :].to(x.dtype)
        j = torch.arange(n, device=item.device)
        bias = rab.pos_w[j - (n - 1) + rab.max_len - 1][None]  # key minus query, as RelativeBias
        if rab.use_time:
            bias = bias + rab.time_w[time_buckets(req_ts[:, None] - input_ts, rab.n_buckets)]
        bias = bias[:, None, :]
        ck, cv = c.k[slots], c.v[slots]
        for i, layer in enumerate(m.layers):
            u, v, q, k = _project(layer, x, self.d)
            keys, values = torch.cat([ck[:, i], k], dim=1), torch.cat([cv[:, i], v], dim=1)
            x = _attend(layer, x, u, q, keys, values, bias, mask, n)
        return F.normalize(x[:, -1], dim=-1)

    def score(self, state: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        """[B, C] fp32 scores; -inf for empty slots (-1)."""
        emb = self.table[cands.clamp(min=0)]
        s = torch.einsum("bcd,bd->bc", emb, state.float())
        return s.masked_fill(cands <= 0, -math.inf)

    @torch.no_grad()
    def __call__(self, **t) -> tuple[torch.Tensor, torch.Tensor]:
        """The top 10 candidates [B, 10] and every candidate's score [B, 200]."""
        if self.mode == "hit":
            state = self.encode_hit(t["slots"], t["item"], t["item_ts"], t["req_ts"])
        else:
            items, ts = t["items"], t["ts"]
            state, ks, vs = self.encode_full(items, ts, query_times(items, ts, t["req_ts"]))
            if self.mode == "miss":
                s, c = t["slots"], self.cache
                c.k.index_copy_(0, s, ks.to(c.k.dtype))
                c.v.index_copy_(0, s, vs.to(c.v.dtype))
                c.ts.index_copy_(0, s, ts[:, :-1])
                c.valid.index_copy_(0, s, items[:, :-1] > 0)
        scores = self.score(state, t["cands"])
        top = t["cands"].gather(1, scores.topk(TOP, dim=1).indices)
        return top, scores


class Graphed:
    """CUDA graphs of a ``RankerStep``, one per power-of-two batch size up to ``max_batch``.

    A batch is padded to the next size: padded rows copy its first row, and their cache slot is the dummy
    slot. ``example`` holds one valid request on the device (inputs of ``step.spec()``)."""

    def __init__(self, step: RankerStep, example: dict[str, torch.Tensor], max_batch: int, warmup: int = 3):
        self.step, self.static, self.graphs, self.outputs = step, {}, {}, {}
        sizes = [1 << i for i in range(bucket(max_batch).bit_length())]
        pool = None
        for size in reversed(sizes):  # the largest first, so the smaller ones fit in its memory pool
            st = {k: v[:1].expand(size, *v.shape[1:]).clone() for k, v in example.items()}
            if "slots" in st:
                st["slots"].fill_(step.dummy)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    step(**st)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out = step(**st)
            pool = graph.pool()
            self.static[size], self.graphs[size], self.outputs[size] = st, graph, out
        torch.cuda.synchronize()
        self.max_batch = sizes[-1]

    def __call__(self, **t) -> tuple[torch.Tensor, torch.Tensor]:
        b = len(t["cands"])
        size = bucket(b)
        st = self.static[size]
        for k, v in t.items():
            st[k][:b].copy_(v, non_blocking=True)
            if size > b:
                if k == "slots":
                    st[k][b:].fill_(self.step.dummy)
                else:
                    st[k][b:].copy_(st[k][:1].expand(size - b, *st[k].shape[1:]))
        self.graphs[size].replay()
        top, scores = self.outputs[size]
        return top[:b], scores[:b]
