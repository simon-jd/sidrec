"""Semantic IDs for the Electronics catalog with Beauty's RQ-VAE (protocol, Changes, 2026-11-05).

    python -m sidrec.semantic_ids.rqvae_catalog

The RQ-VAE of ``sidrec.semantic_ids.rqvae`` trains on the 329,445 items with a train-split interaction for 746
epochs, which is Beauty's 240,000 steps. It then assigns three codes to all 368,228 items. The fourth code
numbers the items that share a three-code prefix, train items first, each group in item order. The embeddings
are the ones results/r3/embeddings/agreement.json selects.

Writes the codes to data/processed/electronics/sid/qwen3-emb-0.6b.npy ([368,228, 4], item order, no padding
row), the model to data/runs/r3/rqvae.pt, and the statistics, sanity examples and checksums to
results/r3/semantic_ids.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from sidrec.semantic_ids import catalog
from sidrec.semantic_ids.embed_catalog import top_k
from sidrec.semantic_ids.rqvae import add_dedupe_code, code_stats, train
from sidrec.train import common

EPOCHS = 746
RUNS = common.ROOT / "data" / "runs" / "r3"


def dedupe_train_first(codes3: np.ndarray, is_train: np.ndarray) -> np.ndarray:
    """[N, 3] -> [N, 4]. The fourth code numbers each three-code group's train items first, then the others,
    each in row order, so a train item's code does not depend on the items outside train."""
    order = np.concatenate([np.flatnonzero(is_train), np.flatnonzero(~is_train)])
    codes4 = np.empty((len(codes3), 4), dtype=np.int64)
    codes4[order] = add_dedupe_code(codes3[order])
    return codes4


def prefix_coverage(train_codes: np.ndarray, query_codes: np.ndarray, levels: int) -> float:
    """Share of the query rows whose first ``levels`` codes are some train row's first ``levels`` codes."""
    known = set(map(tuple, train_codes[:, :levels].tolist()))
    return float(np.mean([tuple(c) in known for c in query_codes[:, :levels].tolist()]))


def neighbour_prefix_agreement(emb: np.ndarray, codes3: np.ndarray, rows: np.ndarray, k: int = 10) -> dict:
    """For sampled items, the share of their k nearest embedding neighbours that share their first codes,
    against the chance that two random catalog items do."""
    nbrs = np.concatenate([top_k(emb[r], emb, k, r) for r in np.array_split(rows, max(1, len(rows) // 256))])
    out = {}
    for levels in (1, 2, 3):
        same = (codes3[nbrs][:, :, :levels] == codes3[rows][:, None, :levels]).all(-1).mean()
        counts = np.unique(codes3[:, :levels], axis=0, return_counts=True)[1]
        p = counts / counts.sum()
        out[f"levels_1_to_{levels}"] = {
            f"top{k}_neighbours_sharing": round(float(same), 4),
            "random_pair_sharing": round(float((p**2).sum()), 5),
        }
    return out


def sanity_examples(emb, codes4, titles, is_train, cold_items, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    def item(r: int, sim: float | None = None) -> dict:
        out = {"title": (titles[r] or "")[:100], "sid": codes4[r].tolist(), "train": bool(is_train[r])}
        if sim is not None:
            out["cosine"] = round(sim, 3)
        return out

    picks = [
        *rng.choice(np.flatnonzero(is_train), 3, replace=False),
        *rng.choice(cold_items, 2, replace=False),
    ]
    neighbours = []
    for r in picks:
        nb = top_k(emb[[r]], emb, 5, np.array([r]))[0]
        neighbours.append({"query": item(r), "neighbours": [item(n, float(emb[r] @ emb[n])) for n in nb]})
    prefixes, inverse, counts = np.unique(codes4[:, :2], axis=0, return_inverse=True, return_counts=True)
    groups = []
    for g in rng.choice(np.flatnonzero(counts >= 5), 3, replace=False):
        members = np.flatnonzero(inverse == g)
        shown = np.sort(rng.choice(members, 5, replace=False))
        groups.append(
            {
                "prefix": prefixes[g].tolist(),
                "items": len(members),
                "train_items": int(is_train[members].sum()),
                "examples": [item(r) for r in shown],
            }
        )
    full, inv3, counts3 = np.unique(codes4[:, :3], axis=0, return_inverse=True, return_counts=True)
    largest = np.flatnonzero(inv3 == counts3.argmax())
    return {
        "nearest_neighbours": neighbours,
        "same_two_code_prefix": groups,
        "largest_three_code_group": {
            "prefix": full[counts3.argmax()].tolist(),
            "items": len(largest),
            "examples": [item(r) for r in largest[:5]],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False  # exact nearest-code search, as in R2

    agree = json.loads((catalog.RESULTS / "embeddings" / "agreement.json").read_text())
    source = agree["ids_use"]
    df = catalog.item_docs()
    ids, titles = df["parent_asin"].to_list(), df["title"].to_list()
    emb_path = catalog.EMB_DIR / f"{source}.npy"
    assert catalog.file_sha256(emb_path) == agree["arrays"][source]["sha256"]
    emb_np = np.load(emb_path)
    train_set = catalog.train_items()
    is_train = np.array([i in train_set for i in ids])
    pos = {a: i for i, a in enumerate(ids)}
    cold_rows = np.array([pos[t] for t in catalog.cold_test_targets()])
    cold_items = np.unique(cold_rows)
    assert not is_train[cold_items].any()

    emb = torch.from_numpy(emb_np).cuda()
    train_idx = torch.from_numpy(np.flatnonzero(is_train)).cuda()
    smi_start = common.nvidia_smi()
    t0 = time.time()
    with common.SmiSampler(interval=30.0) as sampler:
        model, log = train(emb[train_idx], args.epochs, args.seed, log_every=50)
    train_s = time.time() - t0
    model.eval()
    with torch.no_grad():
        z_hat, codes3, _ = model.quantize(model.encoder(emb))
        x_hat = model.decoder(z_hat)

        def explained(rows) -> float:
            x = emb[rows]
            return round((1 - ((x - x_hat[rows]) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()).item(), 4)

        recon, q_loss, _ = model(emb)
        explained_all, explained_train = explained(slice(None)), explained(train_idx)
    codes3 = codes3.cpu().numpy()
    codes4 = dedupe_train_first(codes3, is_train)
    assert len(np.unique(codes4, axis=0)) == len(codes4)

    catalog.SID_DIR.mkdir(parents=True, exist_ok=True)
    codes_path = catalog.SID_DIR / "qwen3-emb-0.6b.npy"
    np.save(codes_path, codes4)
    RUNS.mkdir(parents=True, exist_ok=True)
    model_path = RUNS / "rqvae.pt"
    torch.save(model.state_dict(), model_path)

    train_codes = codes3[is_train]
    coverage = {}
    for name, rows in (
        ("cold_test_items", cold_items),
        ("cold_test_rows", cold_rows),
        ("non_train_items", np.flatnonzero(~is_train)),
    ):
        coverage[name] = {"count": len(rows)}
        coverage[name].update(
            {
                f"levels_1_to_{lv}": round(prefix_coverage(train_codes, codes3[rows], lv), 4)
                for lv in (1, 2, 3)
            }
        )
    sample = catalog.sample_rows(len(ids), 1_000)
    record = {
        "encoder": "qwen3-emb-0.6b",
        "embeddings": {
            "source": source,
            "path": str(emb_path.relative_to(common.ROOT)),
            "sha256": agree["arrays"][source]["sha256"],
        },
        "items": len(ids),
        "item_ids_sha256": catalog.ids_sha256(ids),
        "train_items": int(is_train.sum()),
        "settings": {
            "hidden": [512, 256, 128],
            "latent": 32,
            "levels": 3,
            "codes": 256,
            "beta": 0.25,
            "optimizer": "AdamW, weight decay 0",
            "lr": 1e-3,
            "batch": 1024,
            "epochs": args.epochs,
            "steps": args.epochs * math.ceil(int(is_train.sum()) / 1024),
            "seed": args.seed,
        },
        "final_loss_all_items": {"recon": round(recon.item(), 5), "quant": round(q_loss.item(), 5)},
        "reconstruction_explained_variance": {"all_items": explained_all, "train_items": explained_train},
        "all_items": code_stats(codes3),
        "train_items_only": code_stats(train_codes),
        "fourth_code_values": int(codes4[:, 3].max()) + 1,
        "prefix_coverage_by_train_items": coverage,
        "neighbour_prefix_agreement": {
            "queries": len(sample),
            "sample_seed": 0,
            **neighbour_prefix_agreement(emb_np, codes3, sample),
        },
        "sanity_examples": sanity_examples(emb_np, codes4, titles, is_train, cold_items),
        "artifacts": {
            "codes": {
                "path": str(codes_path.relative_to(common.ROOT)),
                "sha256": catalog.file_sha256(codes_path),
                "shape": list(codes4.shape),
            },
            "model": {
                "path": str(model_path.relative_to(common.ROOT)),
                "sha256": catalog.file_sha256(model_path),
            },
        },
        "loss_log": log,
        "train_seconds": round(train_s, 1),
        "wall_time_s": round(time.time() - t0, 1),
        "gpu": {
            "nvidia_smi_start": smi_start,
            "during_training": sampler.summary(),
            "nvidia_smi_end": common.nvidia_smi(),
        },
        "environment": common.environment(),
        "git": common.git_state(),
    }
    common.write_json(catalog.RESULTS / "semantic_ids.json", record)
    keys = ("reconstruction_explained_variance", "fourth_code_values", "prefix_coverage_by_train_items")
    print({k: record[k] for k in keys}, record["all_items"]["collision_rate"])


if __name__ == "__main__":
    main()
