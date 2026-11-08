"""R4c: TIGER's beam and an HNSW search over the item vectors retrieve 200 candidates; the R4b rankers re-rank
them (docs/protocol.md, entry "2026-11-07: R4c two-stage cascade").

    python -m sidrec.eval.r4c_cascade ann-build              # CPU: the HNSW index over all item vectors
    python -m sidrec.eval.r4c_cascade ann-select             # CPU: query grid on validation, recall vs exact
    python -m sidrec.eval.r4c_cascade beam-valid --seed 0    # GPU: TIGER's width-200 beam, the union grid
    python -m sidrec.eval.r4c_cascade select                 # CPU: the union chosen on validation
    python -m sidrec.eval.r4c_cascade rerank-valid --seed 0  # GPU: the rankers on the validation union
    python -m sidrec.eval.r4c_cascade projection             # CPU: GPU-hours projected from validation
    python -m sidrec.eval.r4c_cascade ann-test               # CPU: the ANN lists of the test rows
    python -m sidrec.eval.r4c_cascade test --seed 0          # GPU: the single test pass of one seed

Every step before ``ann-test`` reads only the 100,000-row validation sample. Records go to results/r4c/; the
index and the candidate lists, which are large, go to data/runs/r4c/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sidrec.data import scale
from sidrec.models import cascade, tiger_cold
from sidrec.models.hstu_sid import HSTUSID
from sidrec.models.tiger import Trie, build_t5, constrained_beam_search
from sidrec.semantic_ids.catalog import file_sha256
from sidrec.train import common, resume
from sidrec.train.scale_idrec import (
    CHECKPOINTS_R4B,
    MAX_LEN,
    RESULTS_R4B,
    SAMPLE,
    build,
    catalog_ranks,
    output_table,
    selected_trial,
    sid_tables,
    vocab_ranks,
)
from sidrec.train.scale_tiger import Inputs, cold_start_inputs, load_ids
from sidrec.tuning import shared_trials

RESULTS = common.ROOT / "results" / "r4c"
RUNS = common.ROOT / "data" / "runs" / "r4c"
INDEX = RUNS / "hnsw.faiss"
EMB = common.ROOT / "data" / "processed" / "electronics" / "emb" / "vllm.npy"
EMB_SHA256 = "fbb7496582c3a5dd66e63140a4cc280c2648e68d7c419a539ff51f32a1a9a8f9"  # results/r3/SHA256SUMS
PROTOCOL_VERSION = "2026-11-05 with the R2, R3, R4a, R4b and R4c clarifications"

HISTORY = 50  # the history mask and the ANN query's window
LIST = 200  # candidates per source and in the union
BUDGETS = (50, 100, 200)
HNSW_M, EF_CONSTRUCTION = 32, 200
EF_SEARCH = (512, 1024, 2048)  # the first, then the fallback values
SEARCH_K = 250  # LIST + HISTORY: enough left after the history is removed
EXACT_RECALL_MIN = 0.95
N_GRID = (1, 2, 3, 5, 10, 20, 50)
GAMMA_GRID = (1.0, 0.8, 0.6, 0.4)
K_TIGER_GRID = (25, 50, 75, 100, 125, 150, 175)
RULES = ("off", "on")
BEAM = 200
EPS = 0.1
TIGER_CHUNK = 128  # 128 rows x beam 200: R4b's 512 x 50 decoder sequences per step
THREADS = 24
RANKERS = ("hstu", "hstu-sid")
SEEDS = (0, 1, 2)


def ann_grid() -> list[tuple[int, float]]:
    """The 25 (n, gamma) in the tie-break order: smaller n first, then larger gamma; n = 1 once."""
    return [(n, g) for n in N_GRID for g in GAMMA_GRID if n > 1 or g == GAMMA_GRID[0]]


def union_grid() -> list[dict]:
    """The 28 union settings in the tie-break order: rule off first, smaller TIGER share, ANN backfill."""
    return [
        {"rule": r, "k_tiger": k, "backfill": b}
        for r in RULES
        for k in K_TIGER_GRID
        for b in cascade.BACKFILL
    ]


def key(cfg: dict) -> str:
    return f"{cfg['rule']}/{cfg['k_tiger']}/{cfg['backfill']}"


def pick(scores: list[float]) -> int:
    """Index of the highest score; exact ties go to the first."""
    return max(range(len(scores)), key=lambda i: (scores[i], -i))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def check_checkpoints(paths: list[Path]) -> dict:
    """sha256 of each checkpoint, which must match results/r4b/SHA256SUMS."""
    sums = {}
    for line in (RESULTS_R4B / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    out = {}
    for p in paths:
        rel = str(p.relative_to(common.ROOT))
        digest = file_sha256(p)
        if sums.get(rel) != digest:
            raise SystemExit(f"{rel}: sha256 differs from results/r4b/SHA256SUMS")
        out[rel] = digest
    return out


def history_window(d: dict, rows: scale.Examples) -> torch.Tensor:
    """[B, 50] catalog indices of each row's input, newest last, 0 padding (CPU)."""
    items = torch.as_tensor(d["items"].astype(np.int64))
    idx, valid = scale.window(rows.end.cpu(), rows.start.cpu(), HISTORY, len(items))
    return items[idx].masked_fill(~valid, 0)


def load_vectors() -> np.ndarray:
    if file_sha256(EMB) != EMB_SHA256:
        raise SystemExit("item vectors differ from the R3 file")
    return np.ascontiguousarray(np.load(EMB), dtype=np.float32)


def search(index, vectors: torch.Tensor, hist: torch.Tensor, n: int, gamma: float, chunk: int = 4096):
    """[B, 200] int32 lists: the index's top SEARCH_K for each row's query, history removed."""
    out = np.empty((len(hist), LIST), dtype=np.int32)
    for s in range(0, len(hist), chunk):
        h = hist[s : s + chunk]
        q = cascade.recency_query(vectors, h, n, gamma).numpy()
        _, ids = index.search(np.ascontiguousarray(q, dtype=np.float32), SEARCH_K)
        found = torch.as_tensor(np.where(ids >= 0, ids + 1, 0).astype(np.int64))
        out[s : s + chunk] = cascade.remove_history(found, h, LIST).numpy()
    return out


def retrieval_recalls(lists: np.ndarray | torch.Tensor, target: torch.Tensor, cold: torch.Tensor) -> dict:
    """recall@50, @100 and @200 of one ranked list, over all, warm and cold rows."""
    lists = torch.as_tensor(lists).long()
    return {str(k): cascade.recall_split(cascade.hits(lists, target, k), cold) for k in BUDGETS}


def overlap(a: np.ndarray, b: np.ndarray, chunk: int = 4096) -> float:
    """Mean over rows of |a ∩ b| / 200 (empty slots excluded)."""
    total = 0
    for s in range(0, len(a), chunk):
        x, y = torch.as_tensor(a[s : s + chunk]), torch.as_tensor(b[s : s + chunk])
        total += int(((x[:, :, None] == y[:, None, :]).any(-1) & (x >= 0)).sum())
    return total / (len(a) * LIST)


def sample_rows(d: dict, split: str, device="cpu"):
    """The protocol's 100,000-row validation sample, or every test row."""
    return scale.eval_rows(d, split, SAMPLE if split == "valid" else None, device)


def faiss_setup():
    import faiss

    faiss.omp_set_num_threads(THREADS)
    return faiss


def cmd_ann_build(_args) -> None:
    faiss = faiss_setup()
    vecs = load_vectors()
    t = time.time()
    index = faiss.IndexHNSWFlat(vecs.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.add(vecs)
    seconds = round(time.time() - t, 1)
    RUNS.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX))
    record = {
        "index": "IndexHNSWFlat, inner product",
        "M": HNSW_M,
        "efConstruction": EF_CONSTRUCTION,
        "items": int(index.ntotal),
        "dim": int(vecs.shape[1]),
        "vectors": {"path": str(EMB.relative_to(common.ROOT)), "sha256": EMB_SHA256},
        "path": str(INDEX.relative_to(common.ROOT)),
        "sha256": file_sha256(INDEX),
        "bytes": INDEX.stat().st_size,
        "build_seconds": seconds,
        "threads": THREADS,
        "faiss": faiss.__version__,
        "note": "multi-threaded build, not bit-reproducible; every R4c result uses this saved index",
        "git": resume.code_state(),
        "protocol": PROTOCOL_VERSION,
    }
    common.write_json(RESULTS / "ann" / "build.json", record)
    print(record["sha256"], seconds)


def open_index():
    faiss = faiss_setup()
    build_rec = load_json(RESULTS / "ann" / "build.json")
    if file_sha256(INDEX) != build_rec["sha256"]:
        raise SystemExit("the HNSW index differs from the one ann-build recorded")
    return faiss, faiss.read_index(str(INDEX)), build_rec["sha256"]


def cmd_ann_select(_args) -> None:
    faiss, index, index_sha = open_index()
    vecs = load_vectors()
    vectors = torch.cat([torch.zeros(1, vecs.shape[1]), torch.from_numpy(vecs)])
    d = scale.load("full")
    rows, _, cold, order = sample_rows(d, "valid")
    hist = history_window(d, rows)
    target = torch.as_tensor(d["items"][rows.end.numpy()].astype(np.int64))
    t0 = time.time()
    rounds = []
    for ef in EF_SEARCH:
        index.hnsw.efSearch = ef
        grid = []
        for n, g in ann_grid():
            t = time.time()
            lists = search(index, vectors, hist, n, g)
            grid.append(
                {
                    "n": n,
                    "gamma": g,
                    "recall": retrieval_recalls(lists, target, cold),
                    "seconds": round(time.time() - t, 1),
                }
            )
            print(ef, n, g, grid[-1]["recall"]["200"]["all"], grid[-1]["seconds"], flush=True)
        best = pick([r["recall"]["200"]["all"] for r in grid])
        n, g = grid[best]["n"], grid[best]["gamma"]
        ann = search(index, vectors, hist, n, g)
        exact_index = faiss.IndexFlatIP(vecs.shape[1])
        exact_index.add(vecs)
        t = time.time()
        exact = search(exact_index, vectors, hist, n, g)
        exact_s = round(time.time() - t, 1)
        del exact_index
        rec = {
            "efSearch": ef,
            "grid": grid,
            "selected": {"n": n, "gamma": g},
            "recall_vs_exact@200": overlap(ann, exact),
            "exact_seconds": exact_s,
            "target_recall": {
                "ann": retrieval_recalls(ann, target, cold),
                "exact": retrieval_recalls(exact, target, cold),
            },
        }
        rounds.append(rec)
        print(ef, rec["selected"], rec["recall_vs_exact@200"], flush=True)
        if rec["recall_vs_exact@200"] >= EXACT_RECALL_MIN:
            break
    final = rounds[-1]
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez(RUNS / "ann_valid.npz", rows=order, lists=ann)
    record = {
        "split": "valid",
        "rows": len(order),
        "cold_rows": int(cold.sum()),
        "index_sha256": index_sha,
        "search_k": SEARCH_K,
        "rounds": rounds,
        "selected": final["selected"] | {"efSearch": final["efSearch"]},
        "recall_vs_exact@200": final["recall_vs_exact@200"],
        "lists": {
            "path": str((RUNS / "ann_valid.npz").relative_to(common.ROOT)),
            "sha256": file_sha256(RUNS / "ann_valid.npz"),
        },
        "threads": THREADS,
        "wall_time_s": round(time.time() - t0, 1),
        "git": resume.code_state(),
        "protocol": PROTOCOL_VERSION,
    }
    common.write_json(RESULTS / "ann" / "valid.json", record)


@torch.no_grad()
def tiger_source(model, inp: Inputs, rows, users, trie, layout, unseen, with_rule: bool):
    """TIGER's width-200 lists, best first, as int32 on the CPU: the plain beam and, with ``with_rule``, the
    list with the cold-start rule at ε = 0.1. Also the number of empty beam slots."""
    model.eval()
    plain, ruled, empty = [], [], 0
    for s in range(0, len(rows), TIGER_CHUNK):
        ex, u = rows.take(slice(s, s + TIGER_CHUNK)), users[s : s + TIGER_CHUNK]
        x = inp.encoder(ex, u)
        history = inp.history(ex)
        last = {} if with_rule else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            items, _ = constrained_beam_search(
                model, x, x > 0, trie, layout, beam=BEAM, exclude=history, last=last
            )
        empty += int((items < 0).sum())
        plain.append(items.int().cpu())
        if with_rule:
            cand, _ = tiger_cold.candidates(items, last, trie.tables[-1], unseen, history, BEAM)
            ruled.append(tiger_cold.merge(items, unseen[items.clamp(min=0)], cand, EPS).int().cpu())
    return torch.cat(plain), (torch.cat(ruled) if with_rule else None), empty


def load_tiger(d: dict, seed: int, device):
    cfg = shared_trials()[selected_trial("tiger") - 1]
    ckpt = CHECKPOINTS_R4B / "tiger" / f"seed{seed}.pt"
    codes, layout = load_ids()
    inp = Inputs(d, layout.item_tokens(codes), layout, device)
    trie = Trie(codes, layout.level_sizes, device)
    model = build_t5(layout, cfg["width"], cfg["layers"], cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    unseen, _ = cold_start_inputs(d, codes, device)
    return model, inp, trie, layout, unseen, ckpt


def gpu_record(t0: float, smi_before: dict, sampler, git: dict) -> dict:
    return {
        "wall_time_s": round(time.time() - t0, 1),
        "gpu": {
            "nvidia_smi_start": smi_before,
            "nvidia_smi_end": common.nvidia_smi(),
            "sampled": sampler.summary(),
            "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        },
        "environment": common.environment(),
        "git": git,
        "protocol": PROTOCOL_VERSION,
    }


def load_lists(path: Path, rows: np.ndarray) -> torch.Tensor:
    with np.load(path) as f:
        if not np.array_equal(f["rows"], rows):
            raise SystemExit(f"{path}: rows differ from this split's rows")
        return torch.as_tensor(f["lists"]).long()


def cmd_beam_valid(args) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    t0 = time.time()
    smi_before, git = common.nvidia_smi(), resume.code_state()
    d = scale.load("full")
    with common.SmiSampler() as sampler:
        model, inp, trie, layout, unseen, ckpt = load_tiger(d, args.seed, device)
        sums = check_checkpoints([ckpt])
        rows, users, cold, order = sample_rows(d, "valid", device)
        torch.cuda.synchronize()
        t = time.time()
        plain, ruled, empty = tiger_source(model, inp, rows, users, trie, layout, unseen, True)
        beam_s = round(time.time() - t, 1)
    np.savez(RUNS / f"tiger_valid_seed{args.seed}.npz", rows=order, plain=plain.numpy(), ruled=ruled.numpy())
    ann = load_lists(RUNS / "ann_valid.npz", order)
    target = inp.items[rows.end].cpu()
    cold = cold.cpu()
    lists = {"off": plain.long(), "on": ruled.long()}
    grid = []
    for cfg in union_grid():
        recall = {}
        for k in BUDGETS:
            u = union_on(lists[cfg["rule"]], ann, cfg, k, device)
            recall[str(k)] = cascade.recall_split(cascade.hits(u, target), cold)
        grid.append(cfg | {"recall": recall})
    record = {
        "split": "valid",
        "seed": args.seed,
        "rows": len(order),
        "checkpoint": sums,
        "beam": BEAM,
        "eps": EPS,
        "empty_slots": empty,
        "tiger": {r: retrieval_recalls(lists[r], target, cold) for r in RULES},
        "ann": retrieval_recalls(ann, target, cold),
        "union_grid": grid,
        "beam_seconds": beam_s,
        "lists": str((RUNS / f"tiger_valid_seed{args.seed}.npz").relative_to(common.ROOT)),
    } | gpu_record(t0, smi_before, sampler, git)
    common.write_json(RESULTS / "valid" / f"tiger_seed{args.seed}.json", record)
    print(record["tiger"]["off"]["50"]["all"], beam_s, record["wall_time_s"])


def cmd_select(_args) -> None:
    recs = [load_json(RESULTS / "valid" / f"tiger_seed{s}.json") for s in SEEDS]
    ann = load_json(RESULTS / "ann" / "valid.json")
    grid = []
    for i, cfg in enumerate(union_grid()):
        per_seed = []
        for r in recs:
            g = r["union_grid"][i]
            if key(g) != key(cfg):
                raise SystemExit("union grids are in different orders")
            per_seed.append(g["recall"]["200"]["all"])
        grid.append(cfg | {"recall@200": per_seed, "mean": float(np.mean(per_seed))})
    best = pick([g["mean"] for g in grid])
    sel = {k: grid[best][k] for k in ("rule", "k_tiger", "backfill")}
    record = {
        "criterion": "union recall@200 over all rows of the validation sample, mean over TIGER seeds 0-2",
        "ann": ann["selected"],
        "union": sel,
        "union_grid": grid,
        "section_2_setting": next(g for g in grid if key(g) == "off/50/ann"),
        "protocol": PROTOCOL_VERSION,
    }
    common.write_json(RESULTS / "selection.json", record)
    print(record["ann"], sel, grid[best]["mean"])


def load_ranker(name: str, seed: int, d: dict, device):
    cfg = shared_trials()[selected_trial(name) - 1]
    vocab_np, v = scale.id_vocab(d)
    sid = sid_tables(d) if name == "hstu-sid" else None
    model = build(name, cfg, v, sid).to(device)
    ckpt = CHECKPOINTS_R4B / name / f"seed{seed}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.eval(), torch.as_tensor(vocab_np, device=device), v, ckpt


@torch.no_grad()
def rerank(model, flat: scale.Flat, rows, cands: torch.Tensor, vocab, v: int, full: bool, chunk: int = 1024):
    """Ranks of each row's target among its candidates [B, C] (CPU, catalog indices), from the ranker's R4b
    scores; with ``full``, also its R4b full-catalog ranks."""
    sid = isinstance(model, HSTUSID)
    table = model.catalog_table() if sid else output_table(model, v)
    ident = torch.arange(len(vocab), device=vocab.device)
    out, out_full = [], []
    for s in range(0, len(rows), chunk):
        ex = rows.take(slice(s, s + chunk))
        b = flat.id_inputs(ex, ident if sid else vocab, MAX_LEN, train=False)
        states = model.encode(b["inputs"], input_ts=b["input_ts"], query_ts=b["query_ts"])[:, -1]
        scores = states @ table.T  # sid: column j is catalog item j + 1; else train item j + 1 (vocab index)
        c = cands[s : s + chunk].to(vocab.device)
        if sid:
            col, scorable = (c - 1).clamp(min=0), c > 0
        else:
            m = vocab[c.clamp(min=0)]
            col, scorable = (m - 1).clamp(0, v - 1), (c > 0) & (m >= 1) & (m <= v)
        out.append(cascade.candidate_ranks(scores.gather(1, col), c, b["target"], scorable))
        if full:
            if sid:
                out_full.append(catalog_ranks(scores, b["history"], b["target"]))
            else:
                out_full.append(vocab_ranks(scores, vocab[b["history"]], vocab[b["target"]], v))
    return torch.cat(out), (torch.cat(out_full) if full else None)


def union_on(tiger: torch.Tensor, ann: torch.Tensor, cfg: dict, budget: int, device, chunk: int = 65536):
    """``cascade.union`` over row chunks on ``device``; the result is on the CPU."""
    out = []
    for s in range(0, len(tiger), chunk):
        t, a = tiger[s : s + chunk].to(device), ann[s : s + chunk].to(device)
        out.append(cascade.union(t, a, cfg["k_tiger"], budget, cfg["backfill"]).cpu())
    return torch.cat(out)


def selected_union(
    lists: dict, ann: torch.Tensor, sel: dict, budget: int = LIST, device="cpu"
) -> torch.Tensor:
    u = sel["union"]
    return union_on(lists[u["rule"]], ann, u, budget, device)


def union_stats(
    u: torch.Tensor, tiger: torch.Tensor, unseen: torch.Tensor, device, chunk: int = 8192
) -> dict:
    """Mean candidates per row, how many came from TIGER's list, and how many are outside the train split."""
    n = from_tiger = outside = short = 0
    unseen = unseen.to(device)
    for s in range(0, len(u), chunk):
        x, t = u[s : s + chunk].to(device), tiger[s : s + chunk].to(device)
        real = x >= 0
        n += int(real.sum())
        short += int((real.sum(1) < u.shape[1]).sum())
        from_tiger += int(((x[:, :, None] == t[:, None, :]).any(-1) & real).sum())
        outside += int((unseen[x.clamp(min=0)] & real).sum())
    rows = len(u)
    return {
        "candidates_per_row": n / rows,
        "from_tiger_list_per_row": from_tiger / rows,
        "outside_train_per_row": outside / rows,
        "rows_below_200": short,
    }


def cmd_rerank_valid(args) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    sel = load_json(RESULTS / "selection.json")
    t0 = time.time()
    smi_before, git = common.nvidia_smi(), resume.code_state()
    d = scale.load("full")
    rows, _, cold, order = sample_rows(d, "valid", device)
    ann = load_lists(RUNS / "ann_valid.npz", order)
    with np.load(RUNS / f"tiger_valid_seed{args.seed}.npz") as f:
        lists = {"off": torch.as_tensor(f["plain"]).long(), "on": torch.as_tensor(f["ruled"]).long()}
    u = selected_union(lists, ann, sel, device=device)
    unseen = torch.as_tensor(~d["train_item"])
    unseen[0] = False
    flat = scale.Flat(d, device)
    out = {"split": "valid", "seed": args.seed, "rows": len(order), "union": sel["union"], "ann": sel["ann"]}
    out["union_stats"] = union_stats(u, lists[sel["union"]["rule"]], unseen, device)
    out["rankers"], sums = {}, {}
    with common.SmiSampler() as sampler:
        for name in RANKERS:
            model, vocab, v, ckpt = load_ranker(name, args.seed, d, device)
            sums |= check_checkpoints([ckpt])
            torch.cuda.synchronize()
            t = time.time()
            ranks, full = rerank(model, flat, rows, u, vocab, v, full=True)
            seconds = round(time.time() - t, 1)
            r4b_valid = load_json(RESULTS_R4B / "runs" / name / f"seed{args.seed}.json")["valid"]
            in_union = torch.isfinite(ranks)
            out["rankers"][name] = {
                "cascade": common_metrics(ranks, cold),
                "full_catalog": common_metrics(full, cold),
                "r4b_record_valid_ndcg@10": r4b_valid["all"]["ndcg@10"],
                "full_catalog_ndcg@10_minus_record": common_metrics(full, cold)["all"]["ndcg@10"]
                - r4b_valid["all"]["ndcg@10"],
                "rows_cascade_rank_above_full_rank": int((ranks[in_union] > full[in_union]).sum()),
                "seconds": seconds,
            }
            del model
    out["checkpoint"] = sums
    out |= gpu_record(t0, smi_before, sampler, git)
    common.write_json(RESULTS / "valid" / f"rerank_seed{args.seed}.json", out)
    print(
        {
            k: (r["cascade"]["all"]["ndcg@10"], r["full_catalog"]["all"]["ndcg@10"])
            for k, r in out["rankers"].items()
        }
    )


def common_metrics(ranks: torch.Tensor, cold: torch.Tensor) -> dict:
    from sidrec.eval.metrics import split_metrics

    return split_metrics(ranks.cpu(), cold.cpu())


def cmd_projection(_args) -> None:
    """Test GPU-hours from the timed validation passes: the beam and the rankers scale with the rows."""
    test_rows = 1_199_539
    seeds = []
    for s in SEEDS:
        beam = load_json(RESULTS / "valid" / f"tiger_seed{s}.json")
        rr = load_json(RESULTS / "valid" / f"rerank_seed{s}.json")
        scale_f = test_rows / beam["rows"]
        rank_s = sum(r["seconds"] for r in rr["rankers"].values())
        test_s = (beam["wall_time_s"] - beam["beam_seconds"]) + beam["beam_seconds"] * scale_f
        test_s += (rr["wall_time_s"] - rank_s) + rank_s * scale_f
        seeds.append(
            {
                "seed": s,
                "valid_beam_s": beam["wall_time_s"],
                "valid_rerank_s": rr["wall_time_s"],
                "test_projected_s": round(test_s, 1),
            }
        )
    spent = sum(x["valid_beam_s"] + x["valid_rerank_s"] for x in seeds) / 3600
    test = sum(x["test_projected_s"] for x in seeds) / 3600
    record = {
        "seeds": seeds,
        "validation_gpu_hours": round(spent, 3),
        "test_projected_gpu_hours": round(test, 3),
        "total_projected_gpu_hours": round(spent + test, 3),
        "stop_at_gpu_hours": 8,
        "note": "the rerank's full-catalog check is not run on test, so the test rerank is overestimated",
    }
    common.write_json(RESULTS / "projection.json", record)
    print(record)


def cmd_ann_test(_args) -> None:
    sel = load_json(RESULTS / "selection.json")["ann"]
    faiss, index, index_sha = open_index()
    index.hnsw.efSearch = sel["efSearch"]
    vecs = load_vectors()
    vectors = torch.cat([torch.zeros(1, vecs.shape[1]), torch.from_numpy(vecs)])
    d = scale.load("full")
    rows, _, _, order = sample_rows(d, "test")
    hist = history_window(d, rows)
    t = time.time()
    lists = search(index, vectors, hist, sel["n"], sel["gamma"])
    seconds = round(time.time() - t, 1)
    np.savez(RUNS / "ann_test.npz", rows=order, lists=lists)
    record = {
        "split": "test",
        "rows": len(order),
        "selected": sel,
        "index_sha256": index_sha,
        "search_seconds": seconds,
        "threads": THREADS,
        "lists": {
            "path": str((RUNS / "ann_test.npz").relative_to(common.ROOT)),
            "sha256": file_sha256(RUNS / "ann_test.npz"),
        },
        "note": "lists only; no target is read here",
        "git": resume.code_state(),
        "protocol": PROTOCOL_VERSION,
    }
    common.write_json(RESULTS / "ann" / "test.json", record)
    print(seconds)


def cmd_test(args) -> None:
    out_path = RESULTS / "test" / f"seed{args.seed}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists: the test split is used once per seed")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    sel = load_json(RESULTS / "selection.json")
    rule_on = sel["union"]["rule"] == "on"
    t0 = time.time()
    smi_before, git = common.nvidia_smi(), resume.code_state()
    d = scale.load("full")
    with common.SmiSampler() as sampler:
        model, inp, trie, layout, unseen, ckpt = load_tiger(d, args.seed, device)
        sums = check_checkpoints([ckpt])
        rows, users, cold, order = sample_rows(d, "test", device)
        ann = load_lists(RUNS / "ann_test.npz", order)
        torch.cuda.synchronize()
        t = time.time()
        plain, ruled, empty = tiger_source(model, inp, rows, users, trie, layout, unseen, rule_on)
        beam_s = round(time.time() - t, 1)
        del model, inp, trie
        torch.cuda.empty_cache()
        lists = {"off": plain.long()} | ({"on": ruled.long()} if rule_on else {})
        tiger = lists[sel["union"]["rule"]]
        target = torch.as_tensor(d["items"][rows.end.cpu().numpy()].astype(np.int64))
        cold_c = cold.cpu()
        unions = {k: selected_union(lists, ann, sel, k, device) for k in BUDGETS}
        retrieval = {
            "tiger": retrieval_recalls(tiger, target, cold_c),
            "ann": retrieval_recalls(ann, target, cold_c),
            "union": {str(k): cascade.recall_split(cascade.hits(unions[k], target), cold_c) for k in BUDGETS},
        }
        if rule_on:
            retrieval["tiger_plain_beam"] = retrieval_recalls(plain, target, cold_c)
        stats = union_stats(unions[LIST], tiger, unseen, device)
        del unions[50], unions[100]
        flat = scale.Flat(d, device)
        rankers = {}
        for name in RANKERS:
            m, vocab, v, rckpt = load_ranker(name, args.seed, d, device)
            sums |= check_checkpoints([rckpt])
            torch.cuda.synchronize()
            t = time.time()
            ranks, _ = rerank(m, flat, rows, unions[LIST], vocab, v, full=False)
            rankers[name] = common_metrics(ranks, cold_c) | {"seconds": round(time.time() - t, 1)}
            del m
    record = {
        "split": "test",
        "seed": args.seed,
        "rows": len(order),
        "cold_rows": int(cold_c.sum()),
        "selection": {"ann": sel["ann"], "union": sel["union"]},
        "checkpoint": sums,
        "beam": BEAM,
        "eps": EPS if rule_on else None,
        "empty_slots": empty,
        "retrieval": retrieval,
        "union_stats": stats,
        "rerank": rankers,
        "beam_seconds": beam_s,
    } | gpu_record(t0, smi_before, sampler, git)
    common.write_json(out_path, record)
    print(retrieval["union"]["200"], {k: r["all"]["ndcg@10"] for k, r in rankers.items()})


COMMANDS = {
    "ann-build": cmd_ann_build,
    "ann-select": cmd_ann_select,
    "beam-valid": cmd_beam_valid,
    "select": cmd_select,
    "rerank-valid": cmd_rerank_valid,
    "projection": cmd_projection,
    "ann-test": cmd_ann_test,
    "test": cmd_test,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
