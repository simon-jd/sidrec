"""Electronics model inputs for R4 (docs/protocol.md, section 2, and its R4a Changes entry).

Every user's interactions, from all three splits, form one timeline in the pipeline's ``pos`` order (timestamp,
ties in official file order). The split is by absolute time, so a user's train interactions are a prefix of
that timeline. Items are numbered 1..368,228 in ascending ``parent_asin`` order, the R3 item order; 0 is
padding. The dev subset uses the same catalog numbering.

``build`` reads the pipeline tables once and writes data/processed/electronics/r4/<source>.npz:
- ``offsets`` [U + 1], ``items`` [T] (catalog index), ``ts`` [T] (milliseconds), ``split`` [T] (0 train,
  1 validation, 2 test): the timelines, users in ascending ``user_id`` order;
- ``train_len`` [U]: the length of each user's train prefix;
- ``user_bucket`` [U]: crc32(user_id) % 2000, TIGER's hashed user token;
- ``train_item`` [N + 1]: items with a train interaction in this source (the dev subset has its own);
- per evaluation split: ``<s>_user`` and ``<s>_target`` (the target's flat index into the timeline arrays)
  for every row with at least one earlier interaction, in table order, ``<s>_cold``, and ``<s>_order``, the
  rows sorted by sha256 of ``user_id|parent_asin|timestamp`` (the protocol's sample order).

Model windows are cut from the flat arrays on the GPU by ``window``: an example is a pair (end, start) of flat
indices, and its input is the up to ``length`` items in [start, end).
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
EXPORT = ROOT / "data" / "snowflake" / "electronics"
DEV = ROOT / "data" / "dev" / "electronics"
CACHE = ROOT / "data" / "processed" / "electronics" / "r4"
SPLITS = ("train", "valid", "test")
EVAL_SPLITS = ("valid", "test")
USER_BUCKETS = 2000
R3_ITEM_IDS_SHA256 = "3dacae9d50fec59110d452cde587e91b17aa8812d9c86abce351d782cf863e9f"


def sample_key(user_id: str, item: str, ts: int) -> str:
    """The protocol's sample order: sha256 of ``user_id|parent_asin|timestamp`` (milliseconds)."""
    return hashlib.sha256(f"{user_id}|{item}|{ts}".encode()).hexdigest()


def _tables(source: str):
    import polars as pl

    from sidrec.data import tables

    if source == "full":
        return {
            t: tables.read_snowflake_export(EXPORT, t) for t in ("interactions", "eval_valid", "eval_test")
        }
    if source == "dev":
        return {
            t: tables.normalize(pl.read_parquet(DEV / f"{t}.parquet"), t)
            for t in ("interactions", "eval_valid", "eval_test")
        }
    raise ValueError(source)


def build(source: str) -> dict:
    """The arrays described in the module docstring, checked against the tables' own columns."""
    import polars as pl

    from sidrec.data import tables

    catalog = tables.read_snowflake_export(EXPORT, "interactions")["parent_asin"].unique().sort()
    ids_sha = hashlib.sha256("\n".join(catalog.to_list()).encode()).hexdigest()
    if ids_sha != R3_ITEM_IDS_SHA256:
        raise ValueError("catalog order differs from the R3 arrays")
    src = _tables(source)
    cat = pl.DataFrame({"parent_asin": catalog, "item": np.arange(1, len(catalog) + 1, dtype=np.int32)})
    inter = src["interactions"].join(cat, on="parent_asin", how="left").sort(["user_id", "pos"])
    if inter["item"].null_count():
        raise ValueError("an interaction's item is outside the catalog")
    users = inter.group_by("user_id", maintain_order=True).agg(
        n=pl.len(), n_train=(pl.col("split") == "train").sum()
    )
    counts = users["n"].to_numpy()
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    split = (
        inter["split"].replace_strict({s: i for i, s in enumerate(SPLITS)}, return_dtype=pl.Int8).to_numpy()
    )
    user_of_row = np.repeat(np.arange(len(counts)), counts)
    pos = inter["pos"].to_numpy()
    if not np.array_equal(pos, np.arange(len(pos)) - offsets[user_of_row]):
        raise ValueError("timeline positions are not 0..n-1 per user")
    same_user = user_of_row[1:] == user_of_row[:-1]
    if (np.diff(split.astype(np.int64))[same_user] < 0).any():
        raise ValueError("a user's train interactions are not a prefix of the timeline")
    items = inter["item"].to_numpy().astype(np.int32)
    out = {
        "n_items": np.int64(len(catalog)),
        "offsets": offsets,
        "items": items,
        "ts": inter["ts"].to_numpy().astype(np.int64),
        "split": split,
        "train_len": users["n_train"].to_numpy().astype(np.int32),
        "user_bucket": np.array(
            [zlib.crc32(u.encode()) % USER_BUCKETS for u in users["user_id"].to_list()], dtype=np.int16
        ),
    }
    train_item = np.zeros(len(catalog) + 1, dtype=bool)
    train_item[items[split == 0]] = True
    out["train_item"] = train_item
    user_index = users.select("user_id").with_row_index("u")
    for s in EVAL_SPLITS:
        ev = src[f"eval_{s}"].filter(pl.col("hist_len") > 0)
        ev = ev.join(user_index, on="user_id", how="left").join(
            inter.select("user_id", pl.col("parent_asin").alias("target"), "pos", "item"),
            on=["user_id", "target"],
            how="left",
        )
        u = ev["u"].to_numpy().astype(np.int64)
        flat = offsets[u] + ev["pos"].to_numpy()  # pos counts the user's earlier interactions
        if not (
            np.array_equal(items[flat], ev["item"].to_numpy())
            and np.array_equal(out["ts"][flat], ev["ts"].to_numpy())
            and np.array_equal(ev["hist_len"].to_numpy(), ev["pos"].to_numpy())
            and (split[flat] == SPLITS.index(s)).all()
        ):
            raise ValueError(f"{s}: evaluation rows disagree with the timelines")
        # The table's history column must be the timeline before the target (its last len(history) items).
        lengths = ev["history"].list.len().to_numpy().astype(np.int64)
        hist = (
            ev.select(pl.col("history").explode().alias("parent_asin"))
            .join(cat, on="parent_asin", how="left", maintain_order="left")["item"]
            .to_numpy()
        )
        row_start = np.repeat(flat - lengths, lengths)
        within = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        if not np.array_equal(items[row_start + within], hist):
            raise ValueError(f"{s}: history column differs from the timeline")
        cold = ev["is_cold"].to_numpy()
        if not np.array_equal(cold, ~train_item[items[flat]]):
            raise ValueError(f"{s}: is_cold differs from the train items of this source")
        keys = [
            sample_key(a, b, c)
            for a, b, c in zip(
                ev["user_id"].to_list(), ev["target"].to_list(), ev["ts"].to_list(), strict=True
            )
        ]
        out[f"{s}_user"] = u.astype(np.int32)
        out[f"{s}_target"] = flat
        out[f"{s}_cold"] = cold
        out[f"{s}_order"] = np.argsort(np.array(keys), kind="stable").astype(np.int32)
    return out


def cache_path(source: str) -> Path:
    return CACHE / f"{source}.npz"


def load(source: str, rebuild: bool = False) -> dict[str, np.ndarray]:
    path = cache_path(source)
    if rebuild or not path.exists():
        arrays = build(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def describe(d: dict[str, np.ndarray]) -> dict:
    """Counts that tie a cache file to docs/data.md."""
    split = d["split"]
    out = {
        "users": int(len(d["offsets"]) - 1),
        "interactions": {s: int((split == i).sum()) for i, s in enumerate(SPLITS)},
        "train_items": int(d["train_item"].sum()),
        "catalog_items": int(d["n_items"]),
    }
    for s in EVAL_SPLITS:
        out[f"{s}_rows"] = int(len(d[f"{s}_target"]))
        out[f"{s}_cold_rows"] = int(d[f"{s}_cold"].sum())
    return out


@dataclass
class Examples:
    """Examples as pairs of flat indices: the input is the items in [start, end), at most ``length`` of them."""

    end: torch.Tensor
    start: torch.Tensor

    def __len__(self) -> int:
        return int(self.end.shape[0])

    def take(self, idx: torch.Tensor) -> Examples:
        return Examples(self.end[idx], self.start[idx])


def window(end: torch.Tensor, start: torch.Tensor, length: int, size: int, left_pad: bool = True):
    """Flat indices [B, length] of the up to ``length`` items before ``end`` (not before ``start``), and a mask.

    left_pad: newest item in the last column, padding first (the item-ID models). Otherwise oldest first with
    padding at the end (TIGER's encoder input). Padding slots hold an index clamped into [0, size).
    """
    n = torch.minimum(end - start, torch.full_like(end, length))
    ar = torch.arange(length, device=end.device)
    if left_pad:
        idx = end[:, None] - length + ar[None, :]
        valid = ar[None, :] >= (length - n)[:, None]
    else:
        idx = (end - n)[:, None] + ar[None, :]
        valid = ar[None, :] < n[:, None]
    return idx.clamp(0, size - 1), valid


def id_train_examples(d: dict[str, np.ndarray], device) -> Examples:
    """One window per user with at least two train items: the inputs end before the last train item."""
    off = torch.as_tensor(d["offsets"][:-1], device=device)
    n = torch.as_tensor(d["train_len"].astype(np.int64), device=device)
    keep = n >= 2
    return Examples((off + n - 1)[keep], off[keep])


def tiger_train_examples(d: dict[str, np.ndarray], device) -> tuple[Examples, torch.Tensor]:
    """Every train position with at least one earlier item is a target; returns the examples and their users."""
    n = np.maximum(d["train_len"].astype(np.int64) - 1, 0)  # targets per user
    users = np.repeat(np.arange(len(n)), n)
    within = np.arange(len(users)) - np.repeat(np.cumsum(n) - n, n)
    end = d["offsets"][users] + 1 + within
    ex = Examples(torch.as_tensor(end, device=device), torch.as_tensor(d["offsets"][users], device=device))
    return ex, torch.as_tensor(users, device=device)


def eval_rows(d: dict[str, np.ndarray], split: str, sample: int | None, device):
    """Evaluation rows of one split: the first ``sample`` in the protocol's sha256 order (all if None or fewer).

    Returns (examples, users, cold flags, row numbers in table order)."""
    rows = d[f"{split}_order"]
    if sample is not None:
        rows = rows[:sample]
    rows = np.sort(rows)
    users = d[f"{split}_user"][rows].astype(np.int64)
    target = d[f"{split}_target"][rows]
    ex = Examples(torch.as_tensor(target, device=device), torch.as_tensor(d["offsets"][users], device=device))
    return (
        ex,
        torch.as_tensor(users, device=device),
        torch.as_tensor(d[f"{split}_cold"][rows], device=device),
        rows,
    )


def id_vocab(d: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    """Catalog index -> item-ID model index: train items 1..V in catalog order, every other item V + 1 (the
    shared out-of-vocabulary row), padding 0. Returns the map and V."""
    train = d["train_item"]
    v = int(train.sum())
    vocab = np.full(len(train), v + 1, dtype=np.int64)
    vocab[train] = np.arange(1, v + 1)
    vocab[0] = 0
    return vocab, v


class Flat:
    """The timeline arrays on the GPU: catalog items, and timestamps in seconds (the HSTU time bias's unit)."""

    def __init__(self, d: dict[str, np.ndarray], device):
        self.items = torch.as_tensor(d["items"].astype(np.int64), device=device)
        self.ts = torch.as_tensor(d["ts"] // 1000, device=device)

    def id_inputs(
        self, ex: Examples, vocab: torch.Tensor, length: int, train: bool
    ) -> dict[str, torch.Tensor]:
        """Left-padded item-ID model inputs; with ``train``, also the item each position predicts."""
        idx, valid = window(ex.end, ex.start, length, len(self.items))
        out = {
            "inputs": vocab[self.items[idx]].masked_fill(~valid, 0),
            "input_ts": self.ts[idx].masked_fill(~valid, 0),
            "query_ts": self.ts[idx + 1].masked_fill(~valid, 0),
        }
        if train:
            out["targets"] = vocab[self.items[idx + 1]].masked_fill(~valid, 0)
        else:
            out["history"] = self.items[idx].masked_fill(~valid, 0)  # catalog indices, for the ranking mask
            out["target"] = self.items[ex.end]
        return out
