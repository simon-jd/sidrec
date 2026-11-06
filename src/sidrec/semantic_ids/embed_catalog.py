"""Electronics item documents, the sentence-transformers embedding loop, and its agreement with vLLM.

    python -m sidrec.semantic_ids.embed_catalog docs
    python -m sidrec.semantic_ids.embed_catalog hf --batch-size 128 [--sample 20000]
    python -m sidrec.semantic_ids.embed_catalog compare

``docs`` writes the item documents in item order for both implementations, and records the token-length
distribution and the train and cold-start counts in results/r3/item_docs.json. ``hf`` runs R2's loop
(``sidrec.semantic_ids.embed.encode``); the full run writes data/processed/electronics/emb/hf.npy.
``compare`` checks the vLLM vectors against these on every item and records which vectors the semantic IDs
use (protocol, Changes): vLLM's, unless some item's two vectors have a cosine below 0.99.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from sidrec.semantic_ids import catalog
from sidrec.semantic_ids.embed import ENCODERS, encode
from sidrec.train import common

MIN_COSINE = 0.99
LENGTH_EDGES = [0, 64, 128, 256, 384, 512, 768, 1024, 2048]


def distribution(values: np.ndarray, percentiles=(1, 10, 50, 90, 99, 99.9)) -> dict:
    out = {"min": float(values.min()), "mean": round(float(values.mean()), 2), "max": float(values.max())}
    out.update({f"p{p:g}": float(np.percentile(values, p)) for p in percentiles})
    return out


def agreement(a: np.ndarray, b: np.ndarray, chunk: int = 65_536) -> dict:
    """Cosine of each row pair and the largest absolute difference of any component."""
    cos = np.empty(len(a), dtype=np.float64)
    max_abs = np.empty(len(a), dtype=np.float64)
    for s in range(0, len(a), chunk):
        x, y = a[s : s + chunk].astype(np.float64), b[s : s + chunk].astype(np.float64)
        cos[s : s + chunk] = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
        max_abs[s : s + chunk] = np.abs(x - y).max(1)
    return {
        "items": len(a),
        "cosine": distribution(cos, (0.01, 0.1, 1, 50)),
        "items_below_0.999": int((cos < 0.999).sum()),
        f"items_below_{MIN_COSINE}": int((cos < MIN_COSINE).sum()),
        "max_abs_component_difference": float(max_abs.max()),
        "per_item_max_abs_difference": distribution(max_abs, (50, 99, 99.9)),
    }


def top_k(queries: np.ndarray, emb: np.ndarray, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k rows of ``emb`` with the largest dot product with each query, best first.
    ``exclude[i]`` (a row index) is left out of query i's list."""
    scores = queries @ emb.T
    if exclude is not None:
        scores[np.arange(len(queries)), exclude] = -np.inf
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, 1), axis=1)
    return np.take_along_axis(part, order, 1)


def neighbour_overlap(a: np.ndarray, b: np.ndarray, rows: np.ndarray, k: int = 10) -> float:
    """Mean share of each sampled item's k nearest neighbours that the two embeddings have in common."""
    shared = []
    for s in range(0, len(rows), 256):
        r = rows[s : s + 256]
        na, nb = top_k(a[r], a, k, r), top_k(b[r], b, k, r)
        shared += [len(set(x) & set(y)) / k for x, y in zip(na, nb, strict=True)]
    return float(np.mean(shared))


def token_lengths(texts: list[str]) -> np.ndarray:
    """Tokens per text, end-of-text token included, with the encoder's tokenizer."""
    from transformers import AutoTokenizer

    repo, revision = ENCODERS["qwen3-emb-0.6b"]
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    return np.concatenate(
        [
            np.array(tok(texts[s : s + 20_000], return_length=True)["length"])
            for s in range(0, len(texts), 20_000)
        ]
    )


def padding(texts: list[str], lengths: np.ndarray, batch_size: int) -> dict:
    """Tokens the loop computes on. sentence-transformers sorts the texts by character count and pads each
    batch to its longest text; the counterfactual sorts by token count instead."""

    def padded(order: np.ndarray) -> int:
        return int(
            sum(
                lengths[order[s : s + batch_size]].max() * len(order[s : s + batch_size])
                for s in range(0, len(order), batch_size)
            )
        )

    by_chars = np.argsort([-len(t) for t in texts])  # the order SentenceTransformer.encode uses
    real = int(lengths.sum())
    out = {"real_tokens": real, "padded_tokens": padded(by_chars)}
    out["padding_share"] = round(1 - real / out["padded_tokens"], 4)
    out["padded_tokens_if_sorted_by_tokens"] = padded(np.argsort(-lengths, kind="stable"))
    return out


def cmd_docs() -> None:
    from transformers import AutoTokenizer

    t0 = time.time()
    df = catalog.item_docs()
    ids, docs = df["parent_asin"].to_list(), df["doc"].to_list()
    assert len(ids) == len(set(ids)) == 368_228
    catalog.write_docs_json(ids, docs)
    repo, revision = ENCODERS["qwen3-emb-0.6b"]
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    lengths = token_lengths(docs)
    train = catalog.train_items()
    cold_rows = catalog.cold_test_targets()
    record = {
        "items": len(ids),
        "item_ids_sha256": catalog.ids_sha256(ids),
        "docs_json": {
            "path": str(catalog.DOCS_JSON.relative_to(common.ROOT)),
            "sha256": catalog.file_sha256(catalog.DOCS_JSON),
        },
        "doc_chars": distribution(np.array([len(d) for d in docs])),
        "tokenizer": {
            "repository": repo,
            "revision": revision,
            "appended_token": tok.convert_ids_to_tokens(tok("a")["input_ids"][-1]),
        },
        "tokens_total": int(lengths.sum()),
        "tokens_per_item": distribution(lengths),
        "tokens_per_item_histogram": {
            "edges": LENGTH_EDGES,
            "counts": np.histogram(lengths, bins=LENGTH_EDGES)[0].tolist(),
        },
        "train_items": len(train),
        "non_train_items": len(ids) - len(train),
        "cold_test_rows": len(cold_rows),
        "cold_test_items": len(set(cold_rows)),
        "wall_time_s": round(time.time() - t0, 1),
        "git": common.git_state(),
    }
    common.write_json(catalog.RESULTS / "item_docs.json", record)
    print(
        {k: record[k] for k in ("items", "tokens_total", "train_items", "cold_test_rows", "cold_test_items")}
    )


def cmd_hf(batch_size: int, sample: int) -> None:
    ids, docs = catalog.load_docs_json()
    rows = catalog.sample_rows(len(docs), sample) if sample else np.arange(len(docs))
    texts = [docs[i] for i in rows]
    pad = padding(texts, token_lengths(texts), batch_size)
    smi_start = common.nvidia_smi()
    t0 = time.time()
    with common.SmiSampler() as sampler:
        emb, info = encode("qwen3-emb-0.6b", texts, batch_size)
    norms = np.linalg.norm(emb, axis=1)
    name = f"hf-bs{batch_size}" + (f"-sample{sample}" if sample else "")
    record = {
        "implementation": "sentence-transformers batch loop (R2's sidrec.semantic_ids.embed.encode)",
        "encoder": "qwen3-emb-0.6b",
        **info,
        "batch_size": batch_size,
        "padding": pad,
        "items": len(texts),
        "sample_seed": 0 if sample else None,
        "item_ids_sha256": catalog.ids_sha256([ids[i] for i in rows]),
        "norm_min_max": [round(float(norms.min()), 4), round(float(norms.max()), 4)],
        "non_finite_values": int((~np.isfinite(emb)).sum()),
        "wall_time_s": round(time.time() - t0, 1),
        "gpu": {
            "nvidia_smi_start": smi_start,
            "during_load_and_encode": sampler.summary(),
            "nvidia_smi_end": common.nvidia_smi(),
        },
        "environment": common.environment(),
        "git": common.git_state(),
    }
    if not sample:
        catalog.EMB_DIR.mkdir(parents=True, exist_ok=True)
        path = catalog.EMB_DIR / "hf.npy"
        np.save(path, emb)
        record["array"] = {"path": str(path.relative_to(common.ROOT)), "sha256": catalog.file_sha256(path)}
    common.write_json(catalog.RESULTS / "embeddings" / f"{name}.json", record)
    print({k: record[k] for k in ("items", "tokens", "encode_seconds", "items_per_s", "tokens_per_s")})


def cmd_compare() -> None:
    v = np.load(catalog.EMB_DIR / "vllm.npy")
    h = np.load(catalog.EMB_DIR / "hf.npy")
    assert v.shape == h.shape
    record = agreement(v, h)
    rows = catalog.sample_rows(len(v), 2_000)
    record["top10_neighbour_overlap"] = {
        "queries": len(rows),
        "sample_seed": 0,
        "mean_share_in_common": round(neighbour_overlap(v, h, rows), 4),
    }
    ok = record[f"items_below_{MIN_COSINE}"] == 0
    record["rule"] = f"the IDs use the vLLM vectors unless some item's cosine is below {MIN_COSINE}"
    record["ids_use"] = "vllm" if ok else "hf"
    record["arrays"] = {
        name: {
            "path": str((catalog.EMB_DIR / f"{name}.npy").relative_to(common.ROOT)),
            "sha256": catalog.file_sha256(catalog.EMB_DIR / f"{name}.npy"),
        }
        for name in ("vllm", "hf")
    }
    common.write_json(catalog.RESULTS / "embeddings" / "agreement.json", record)
    print({k: record[k] for k in ("cosine", "max_abs_component_difference", "ids_use")})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("docs")
    hf = sub.add_parser("hf")
    hf.add_argument("--batch-size", type=int, default=64)
    hf.add_argument("--sample", type=int, default=0, help="encode this many items (seed 0); 0 = all")
    sub.add_parser("compare")
    args = parser.parse_args()
    if args.cmd == "docs":
        cmd_docs()
    elif args.cmd == "hf":
        cmd_hf(args.batch_size, args.sample)
    else:
        cmd_compare()


if __name__ == "__main__":
    main()
