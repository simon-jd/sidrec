"""R5 requests: user histories as the server receives them, and the model inputs it derives from them.

A request is one evaluation row seen from the serving side: the user's hashed token, the 50 most recent items
before the target with their timestamps in seconds, and the request time, which is the target's timestamp.
Items are catalog indices, left-padded with 0, newest in the last column. The target and its cold flag are kept
for the quality checks only; the server never reads them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sidrec.data import scale

WINDOW = 50  # the ranker's window, the history mask and the ANN query's window (R4c)
TIGER_WINDOW = 20  # TIGER's encoder history (sidrec.train.scale_tiger.HISTORY)
CANDS = 200


@dataclass
class Requests:
    rows: np.ndarray  # row numbers of the split's tables, in the order the requests are issued
    items: torch.Tensor  # [P, 50] catalog indices, newest last, 0 padding
    ts: torch.Tensor  # [P, 50] seconds, 0 for padding
    req_ts: torch.Tensor  # [P] the request time
    user_tok: torch.Tensor  # [P] TIGER's user token
    target: torch.Tensor  # [P] quality checks only
    cold: torch.Tensor  # [P] bool, quality checks only

    def __len__(self) -> int:
        return len(self.rows)

    def fields(self) -> dict[str, torch.Tensor]:
        """What the server may read, plus the newest item and its time (the cache-hit request)."""
        return {
            "items": self.items,
            "ts": self.ts,
            "req_ts": self.req_ts,
            "user_tok": self.user_tok,
            "item": self.items[:, -1].contiguous(),
            "item_ts": self.ts[:, -1].contiguous(),
        }


def build_requests(d: dict, split: str, rows: np.ndarray, user_offset: int) -> Requests:
    """The requests of ``rows`` (row numbers of the split's tables), in the given order."""
    rows = np.asarray(rows)
    users = d[f"{split}_user"][rows].astype(np.int64)
    end = torch.as_tensor(d[f"{split}_target"][rows].astype(np.int64))
    start = torch.as_tensor(d["offsets"][users].astype(np.int64))
    items_all = torch.as_tensor(d["items"].astype(np.int64))
    ts_all = torch.as_tensor(d["ts"].astype(np.int64) // 1000)  # seconds, as scale.Flat
    idx, valid = scale.window(end, start, WINDOW, len(items_all))
    return Requests(
        rows=rows,
        items=items_all[idx].masked_fill(~valid, 0),
        ts=ts_all[idx].masked_fill(~valid, 0),
        req_ts=ts_all[end],
        user_tok=user_offset + torch.as_tensor(d["user_bucket"][users].astype(np.int64)),
        target=items_all[end],
        cold=torch.as_tensor(np.asarray(d[f"{split}_cold"][rows], dtype=bool)),
    )


def query_times(items: torch.Tensor, ts: torch.Tensor, req_ts: torch.Tensor) -> torch.Tensor:
    """HSTU's query times: each column's is the time of the item after it, the newest column's the request time."""
    q = torch.cat([ts[:, 1:], req_ts[:, None]], dim=1)
    return q.masked_fill(items <= 0, 0)


def tiger_input(items: torch.Tensor, user_tok: torch.Tensor, item_tokens: torch.Tensor) -> torch.Tensor:
    """[B, 1 + 4 * 20] TIGER encoder input: the user token, then the four tokens of each of the 20 most recent
    items, oldest first, padding at the end (``scale_tiger.Inputs.encoder``)."""
    h = items[:, -TIGER_WINDOW:]
    n = (h > 0).sum(1)
    ar = torch.arange(TIGER_WINDOW, device=items.device)
    idx = (ar[None, :] + (TIGER_WINDOW - n)[:, None]).clamp(max=TIGER_WINDOW - 1)
    right = h.gather(1, idx).masked_fill(ar[None, :] >= n[:, None], 0)
    tok = item_tokens[right].reshape(len(h), -1)
    return torch.cat([user_tok[:, None], tok], dim=1)


def history_mask(items: torch.Tensor) -> torch.Tensor:
    """The 50-item history, sorted ascending with 0 padding first (``scale_tiger.Inputs.history``)."""
    return items.sort(dim=1).values
