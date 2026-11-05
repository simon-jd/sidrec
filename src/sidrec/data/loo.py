"""Beauty leave-one-out data as model inputs.

Items are numbered 1..N in order of first appearance, as S3-Rec's id_map does (0 is padding), so the numbering
matches the S3-Rec Beauty.txt that R1 checked line for line. Item order only matters for tie-breaking in the
ranking.

For a user's sequence s_0..s_{n-1} (n >= 5):
- training predicts s_1..s_{n-3} from the items before each, using one window per user (the last ``max_len``
  positions of the training part), as S3-Rec trains SASRec;
- validation predicts s_{n-2} from s_0..s_{n-3};
- test predicts s_{n-1} from s_0..s_{n-2}.
Windows are left-padded with 0. ``query_ts`` is the timestamp of the item each position predicts (the request
time), used only by the HSTU-style model's time bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
SEQUENCES = ROOT / "data" / "processed" / "beauty" / "sequences.parquet"


@dataclass
class Beauty:
    n_items: int
    item_ids: list[str]  # item index i is item_ids[i - 1]
    user_ids: list[str]
    seqs: list[np.ndarray]  # item indices, 1..n_items
    times: list[np.ndarray]  # unixReviewTime, seconds


def load(path: Path = SEQUENCES) -> Beauty:
    seq = pl.read_parquet(path)
    order = seq.select("items").explode("items")["items"].unique(maintain_order=True).to_list()
    index = {item: i + 1 for i, item in enumerate(order)}
    seqs = [np.array([index[i] for i in items], dtype=np.int64) for items in seq["items"].to_list()]
    times = [np.asarray(t, dtype=np.int64) for t in seq["timestamps"].to_list()]
    return Beauty(len(order), order, seq["user_id"].to_list(), seqs, times)


def _window(s: np.ndarray, t: np.ndarray, end: int, max_len: int):
    """Inputs s[start:end] and the items they predict, s[start+1:end+1]; start = max(0, end - max_len)."""
    start = max(0, end - max_len)
    return s[start:end], s[start + 1 : end + 1], t[start:end], t[start + 1 : end + 1]


def _left_pad(rows: list[np.ndarray], width: int) -> np.ndarray:
    out = np.zeros((len(rows), width), dtype=np.int64)
    for r, row in enumerate(rows):
        if len(row):
            out[r, width - len(row) :] = row
    return out


def id_model_arrays(data: Beauty, split: str, max_len: int = 50) -> dict[str, np.ndarray]:
    """Left-padded [U, max_len] arrays for one split.

    train: ``inputs``, ``targets`` (per position, 0 at padding), ``input_ts``, ``query_ts``.
    valid / test: ``inputs``, ``input_ts``, ``query_ts`` and ``target`` [U], the single item to rank.
    """
    offset = {"train": 3, "valid": 2, "test": 1}[split]
    inputs, targets, input_ts, query_ts = [], [], [], []
    for s, t in zip(data.seqs, data.times, strict=True):
        x, y, xt, yt = _window(s, t, len(s) - offset, max_len)
        inputs.append(x)
        targets.append(y)
        input_ts.append(xt)
        query_ts.append(yt)
    out = {
        "inputs": _left_pad(inputs, max_len),
        "input_ts": _left_pad(input_ts, max_len),
        "query_ts": _left_pad(query_ts, max_len),
    }
    if split == "train":
        out["targets"] = _left_pad(targets, max_len)
    else:
        out["target"] = np.array([y[-1] for y in targets], dtype=np.int64)
    return out
