"""Item text and item embeddings for Beauty.

    python -m sidrec.semantic_ids.embed --encoder st5-xxl

The text of an item is built from the fields TIGER names (title, price, brand, categories) in
``meta_Beauty.json.gz``, whose lines are Python literals. Embeddings are L2-normalized, written in item-index
order (row i is item index i + 1) to data/processed/beauty/emb/<encoder>.npy; the run record goes to
results/r2/embeddings/<encoder>.json. Model weights come from the Hugging Face cache, outside the repository.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import html
import re
import time
from pathlib import Path

import numpy as np
import torch

from sidrec.train import common

META = common.ROOT / "data" / "raw" / "beauty2014" / "meta_Beauty.json.gz"
EMB_DIR = common.ROOT / "data" / "processed" / "beauty" / "emb"

# name -> (Hugging Face repository, pinned revision)
ENCODERS = {
    "st5-xxl": ("sentence-transformers/sentence-t5-xxl", "97f38f2f69860f732f602829da6df76ee5c4c0f9"),
    "qwen3-emb-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    "qwen3-emb-4b": ("Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b"),
}


def _clean(value) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip()


def item_text(meta: dict) -> str:
    """'Title: ...; Brand: ...; Price: ...; Categories: ...', leaving out missing fields."""
    parts = []
    if meta.get("title"):
        parts.append(f"Title: {_clean(meta['title'])}")
    if meta.get("brand"):
        parts.append(f"Brand: {_clean(meta['brand'])}")
    if meta.get("price") is not None:
        parts.append(f"Price: {float(meta['price']):.2f}")
    cats = list(dict.fromkeys(_clean(c) for path in meta.get("categories", []) for c in path if _clean(c)))
    if cats:
        parts.append(f"Categories: {', '.join(cats)}")
    return "; ".join(parts)


def item_texts(item_ids: list[str], path: Path = META) -> list[str]:
    wanted = set(item_ids)
    texts = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            meta = ast.literal_eval(line)
            if meta["asin"] in wanted:
                texts[meta["asin"]] = item_text(meta)
    return [texts[i] for i in item_ids]


def encode(name: str, texts: list[str], batch_size: int) -> tuple[np.ndarray, dict]:
    from sentence_transformers import SentenceTransformer

    repo, revision = ENCODERS[name]
    model = SentenceTransformer(
        repo, revision=revision, device="cuda", model_kwargs={"dtype": torch.bfloat16}
    )
    tokens = sum(
        sum(model.tokenizer(texts[s : s + 10_000], return_length=True)["length"])
        for s in range(0, len(texts), 10_000)
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
    torch.cuda.synchronize()
    seconds = time.time() - t0
    info = {
        "repository": repo,
        "revision": revision,
        "dtype": "bfloat16",
        "parameters": sum(p.numel() for p in model.parameters()),
        "max_seq_length": model.max_seq_length,
        "pooling": str(model[1]) if len(model) > 1 else None,
        "tokens": tokens,
        "encode_seconds": round(seconds, 2),
        "items_per_s": round(len(texts) / seconds, 1),
        "tokens_per_s": round(tokens / seconds, 1),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    return emb.astype(np.float32), info


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    from sidrec.data import loo  # polars; imported here so the vLLM environment can use ENCODERS

    data = loo.load()
    texts = item_texts(data.item_ids)
    smi = common.nvidia_smi()
    t0 = time.time()
    emb, info = encode(args.encoder, texts, args.batch_size)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_DIR / f"{args.encoder}.npy", emb)
    norms = np.linalg.norm(emb, axis=1)
    record = {
        "encoder": args.encoder,
        **info,
        "items": len(texts),
        "dim": int(emb.shape[1]),
        "non_finite_values": int((~np.isfinite(emb)).sum()),
        "norm_min_max": [round(float(norms.min()), 4), round(float(norms.max()), 4)],
        "text_chars_mean": round(float(np.mean([len(t) for t in texts])), 1),
        "example_text": texts[0],
        "wall_time_s": round(time.time() - t0, 1),
        "gpu": {"nvidia_smi_start": smi, "nvidia_smi_end": common.nvidia_smi()},
        "environment": common.environment(),
        "git": common.git_state(),
    }
    common.write_json(common.RESULTS / "embeddings" / f"{args.encoder}.json", record)
    print({k: record[k] for k in ("encoder", "dim", "items_per_s", "peak_memory_gib", "non_finite_values")})


if __name__ == "__main__":
    main()
