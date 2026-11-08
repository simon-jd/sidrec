"""The R5 servers: the ranker alone and the whole cascade, each serving request numbers from a host pool.

Request number i is pool row i mod P, so a run longer than the pool cycles through it. Each server gathers its
batch's fields into pinned host buffers, copies them to the GPU, runs, and returns once the top 10 is in host
memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from sidrec.eval.r4c_cascade import search
from sidrec.models import cascade
from sidrec.models.tiger import Layout, Trie, constrained_beam_search
from sidrec.serve.ranker import Graphed, RankerStep, bucket
from sidrec.serve.requests import CANDS, history_mask, tiger_input

K_TIGER, BACKFILL = 175, "tiger"  # R4c's selected union


class HostPool:
    """Request fields in pinned host memory, gathered per batch into pinned staging buffers."""

    def __init__(self, fields: dict[str, torch.Tensor], max_batch: int):
        self.fields = {k: v.contiguous().pin_memory() for k, v in fields.items()}
        self.n = len(next(iter(self.fields.values())))
        cap = bucket(max_batch)
        self.stage = {k: torch.empty((cap, *v.shape[1:]), dtype=v.dtype).pin_memory() for k, v in self.fields.items()}

    def rows(self, requests: np.ndarray) -> np.ndarray:
        return requests % self.n

    def gather(self, rows: np.ndarray, names) -> dict[str, torch.Tensor]:
        idx = torch.from_numpy(rows)
        return {k: torch.index_select(self.fields[k], 0, idx, out=self.stage[k][: len(rows)]) for k in names}


class SlotMap:
    """The host side of the history cache: request key -> cache slot (here the key is the pool row)."""

    def __init__(self, max_batch: int):
        self.slot_of: dict[int, int] = {}
        self.buf = torch.empty(bucket(max_batch), dtype=torch.long).pin_memory()

    def assign(self, keys: np.ndarray) -> torch.Tensor:
        """Misses: each key gets its own slot, which the step then writes."""
        out = self.buf[: len(keys)]
        for j, key in enumerate(keys.tolist()):
            self.slot_of[key] = key
            out[j] = key
        return out

    def lookup(self, keys: np.ndarray) -> torch.Tensor:
        """Hits: the slots of keys that are in the cache (a KeyError otherwise)."""
        out = self.buf[: len(keys)]
        out.copy_(torch.as_tensor([self.slot_of[k] for k in keys.tolist()], dtype=torch.long))
        return out


class RankerServer:
    """HSTU-SID alone. The pool has ``cands`` (the request's 200 candidates) besides the request fields."""

    def __init__(self, step: RankerStep, pool: HostPool, graphs: bool, max_batch: int, device="cuda"):
        self.step, self.pool, self.max_batch, self.device = step, pool, max_batch, device
        self.names = list(step.spec())
        self.slots = SlotMap(max_batch) if step.mode else None
        if graphs:
            example = self._inputs(np.zeros(1, dtype=np.int64), fill=step.mode == "hit")
            example = {k: v.to(device) for k, v in example.items()}
            self.call = Graphed(step, example, max_batch)
        else:
            self.call = lambda **t: step(**{k: v.to(device, non_blocking=True) for k, v in t.items()})

    def _inputs(self, rows: np.ndarray, fill: bool = False) -> dict[str, torch.Tensor]:
        t = self.pool.gather(rows, [k for k in self.names if k != "slots"])
        if self.slots is not None:
            if self.step.mode == "hit" and not fill:
                t["slots"] = self.slots.lookup(rows)
            else:
                t["slots"] = self.slots.assign(rows) if self.step.mode == "miss" else self.slots.buf[:1].fill_(0)
        return t

    def run(self, requests: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Top 10 [B, 10] and scores [B, 200], on the host."""
        top, scores = self.call(**self._inputs(self.pool.rows(requests)))
        return top.cpu().numpy(), scores.cpu().numpy()

    def __call__(self, requests: np.ndarray) -> None:
        top, _ = self.call(**self._inputs(self.pool.rows(requests)))
        top.cpu()


def fill_cache(miss: RankerStep, slots: SlotMap, pool: HostPool, rows: np.ndarray, chunk: int = 1024) -> None:
    """Writes the cache of ``rows`` with the miss path, eagerly, and registers their keys (untimed)."""
    names = [k for k in miss.spec() if k != "slots"]
    device = miss.table.device
    for s in range(0, len(rows), chunk):
        r = rows[s : s + chunk]
        idx = torch.from_numpy(r)
        t = {k: pool.fields[k][idx].to(device) for k in names}
        slots.slot_of.update((key, key) for key in r.tolist())
        t["slots"] = idx.to(device)
        miss(**t)
    torch.cuda.synchronize()


@dataclass
class Tiger:
    model: torch.nn.Module
    trie: Trie
    layout: Layout
    tokens: torch.Tensor  # [N + 1, 4] item tokens on the GPU


def beam_lists(tiger: Tiger, items: torch.Tensor, user_tok: torch.Tensor, width: int) -> torch.Tensor:
    """TIGER's lists [B, width] for requests on the GPU: R4c's beam code, bf16 autocast, history removed."""
    x = tiger_input(items, user_tok, tiger.tokens)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lists, _ = constrained_beam_search(
            tiger.model, x, x > 0, tiger.trie, tiger.layout, beam=width, exclude=history_mask(items)
        )
    return lists


def ann_lists(index, vectors: torch.Tensor, items: torch.Tensor) -> np.ndarray:
    """R4c's ANN lists [B, 200] (int32) on the CPU: the newest item's vector, efSearch from the index."""
    return search(index, vectors, items, 1, 1.0)


class Cascade:
    """The whole cascade: TIGER's beam on the GPU while the HNSW search runs on the CPU, then the union and
    HSTU-SID (CUDA graphs, misses). Per batch it returns the stage times in milliseconds: GPU stages from CUDA
    events, the HNSW search from the host clock."""

    STAGES = ("beam_ms", "hnsw_ms", "union_ms", "ranker_ms", "batch_ms")

    def __init__(self, tiger: Tiger, index, vectors, rank: RankerStep, pool: HostPool, width: int, max_batch: int):
        self.tiger, self.index, self.vectors, self.pool, self.width = tiger, index, vectors, pool, width
        self.slots = SlotMap(max_batch)
        cap = bucket(max_batch)
        self.ann_buf = torch.empty((cap, CANDS), dtype=torch.long).pin_memory()
        example = {k: v[:1].cuda() for k, v in pool.gather(np.zeros(1, dtype=np.int64), ("items", "ts", "req_ts")).items()}
        example["cands"] = torch.arange(1, CANDS + 1, device="cuda")[None]
        example["slots"] = torch.zeros(1, dtype=torch.long, device="cuda")
        self.rank = Graphed(rank, example, max_batch)
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

    def run(self, requests: np.ndarray, keep: bool = False) -> tuple[np.ndarray, dict, dict]:
        ev = self.events
        rows = self.pool.rows(requests)
        b = len(rows)
        t0 = time.perf_counter()
        h = self.pool.gather(rows, ("items", "ts", "req_ts", "user_tok"))
        g = {k: v.to("cuda", non_blocking=True) for k, v in h.items()}
        ev[0].record()
        lists = beam_lists(self.tiger, g["items"], g["user_tok"], self.width)
        ev[1].record()
        t1 = time.perf_counter()
        ann = ann_lists(self.index, self.vectors, h["items"])
        t2 = time.perf_counter()
        buf = self.ann_buf[:b]
        buf.copy_(torch.from_numpy(ann))
        ann_g = buf.to("cuda", non_blocking=True)
        ev[2].record()
        cands = cascade.union(lists, ann_g, K_TIGER, CANDS, BACKFILL)
        ev[3].record()
        top, _ = self.rank(items=g["items"], ts=g["ts"], req_ts=g["req_ts"], cands=cands, slots=self.slots.assign(rows))
        ev[4].record()
        out = top.cpu().numpy()
        t3 = time.perf_counter()
        times = {
            "beam_ms": ev[0].elapsed_time(ev[1]),
            "hnsw_ms": (t2 - t1) * 1e3,
            "union_ms": ev[2].elapsed_time(ev[3]),
            "ranker_ms": ev[3].elapsed_time(ev[4]),
            "batch_ms": (t3 - t0) * 1e3,
        }
        extra = {"tiger": lists.cpu().numpy(), "ann": ann.copy(), "cands": cands.cpu().numpy()} if keep else {}
        return out, times, extra

    def __call__(self, requests: np.ndarray) -> dict:
        return self.run(requests)[1]
