"""R5: how fast the R4c cascade serves on the RTX 4090 (docs/protocol.md, entry "2026-11-07: R5 serving
benchmark").

    python -m sidrec.eval.r5_serving projection        # short timed pass and the GPU-hour projection
    python -m sidrec.eval.r5_serving beam-valid        # TIGER at widths 20 and 50 on validation
    python -m sidrec.eval.r5_serving select            # B_max of each batched configuration (validation pool)
    python -m sidrec.eval.r5_serving quality           # each ladder step on validation; serving check
    python -m sidrec.eval.r5_serving ladder            # the ranker ladder (test pool)
    python -m sidrec.eval.r5_serving stages            # batch-1 stage latencies (test pool)
    python -m sidrec.eval.r5_serving e2e --width 200   # the cascade under load (test pool)

Every command uses the GPU. Only validation rows are scored; test rows are request histories for timing and no
test metric is computed. Records go to results/r5/, raw per-request arrays to data/runs/r5/raw/.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from sidrec.data import scale
from sidrec.eval import r4c_cascade as r4c
from sidrec.eval.metrics import split_metrics
from sidrec.models import cascade
from sidrec.models.tiger import constrained_beam_search
from sidrec.serve import loadgen
from sidrec.serve.machine import Background
from sidrec.serve.pipeline import (
    BACKFILL,
    K_TIGER,
    Cascade,
    HostPool,
    RankerServer,
    Tiger,
    ann_lists,
    beam_lists,
    fill_cache,
)
from sidrec.serve.ranker import HistoryCache, RankerStep
from sidrec.serve.requests import CANDS, build_requests
from sidrec.train import common, resume
from sidrec.train.scale_tiger import load_ids

RESULTS = common.ROOT / "results" / "r5"
RUNS = common.ROOT / "data" / "runs" / "r5"
LOGS = common.ROOT / "data" / "logs" / "r5"
PROTOCOL_VERSION = "2026-11-05 with the R2, R3, R4a, R4b, R4c and R5 clarifications"

SEED = 0
SLO = 0.050
WINDOW = 0.002
POOL = 50_000
LARGEST = 1024
WIDTHS = (20, 50, 200)
THREADS = 16
BATCH1_N, BATCH1_WARM = 2000, 200
WARM_BATCHES = 20
WARM_S, BISECT_S, CONFIRM_S, SAT_S = 2.0, 20.0, 30.0, 5.0
BISECT_STEPS, REPS, LOWER, TRIES = 7, 3, 0.95, 4
SWEEP_N, SWEEP_WARM = 50, 10
OVERLAP_MIN, NDCG_SHIFT_MAX = 0.95, 0.01
CHECK_ROWS = 2000
STOP_GPU_HOURS = 6.0
LADDER = ("fp32", "bf16", "graphs", "cache-miss", "cache-hit", "batching-miss", "batching-hit")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def ladder_plan(keeps: dict) -> dict:
    """Each step's configuration; a step that failed the quality check is left out of the steps after it."""
    dt = "bf16" if keeps.get("bf16", True) else "fp32"
    graphs = keeps.get("graphs", True)
    cache = keeps.get("cache-miss", True) and keeps.get("cache-hit", True)

    def cfg(name, dtype, g, mode, batching):
        return {"name": name, "dtype": dtype, "graphs": g, "cache": mode, "batching": batching}

    return {
        "fp32": cfg("fp32", "fp32", False, None, False),
        "bf16": cfg("bf16", "bf16", False, None, False),
        "graphs": cfg("graphs", dt, True, None, False),
        "cache-miss": cfg("cache-miss", dt, graphs, "miss", False),
        "cache-hit": cfg("cache-hit", dt, graphs, "hit", False),
        "batching-miss": cfg("batching-miss", dt, graphs, "miss" if cache else None, True),
        "batching-hit": cfg("batching-hit", dt, graphs, "hit", True) if cache else None,
    }


def final_ranker(keeps: dict) -> dict:
    """The ranker the cascade serves with: the last ladder step's configuration, with misses."""
    return ladder_plan(keeps)["batching-miss"]


def load_json(path):
    return json.loads(path.read_text())


def keeps_from_quality() -> dict:
    path = RESULTS / "valid" / "quality.json"
    return load_json(path)["keeps"] if path.exists() else {}


# ------------------------------------------------------------------------------------------------------ setup


def setup() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(THREADS)


def load_index():
    faiss, index, sha = r4c.open_index()
    faiss.omp_set_num_threads(THREADS)
    index.hnsw.efSearch = r4c.load_json(r4c.RESULTS / "selection.json")["ann"]["efSearch"]
    vecs = r4c.load_vectors()
    vectors = torch.cat([torch.zeros(1, vecs.shape[1]), torch.from_numpy(vecs)])
    return index, vectors, {"sha256": sha, "efSearch": index.hnsw.efSearch, "omp_threads": THREADS}


def load_tiger(d):
    model, inp, trie, layout, _unseen, ckpt = r4c.load_tiger(d, SEED, "cuda")
    return Tiger(model.eval(), trie, layout, inp.tokens), inp, r4c.check_checkpoints([ckpt])


def load_hstu_sid(d, with_vocab: bool = False):
    model, vocab, v, ckpt = r4c.load_ranker("hstu-sid", SEED, d, "cuda")
    sums = r4c.check_checkpoints([ckpt])
    return (model, sums, vocab, v) if with_vocab else (model, sums)


def user_offset() -> int:
    return load_ids()[1].user_offset


def ann_for(split: str, rows: np.ndarray) -> torch.Tensor:
    """R4c's saved ANN lists [len(rows), 200] for ``rows`` of the split, in their order."""
    with np.load(r4c.RUNS / f"ann_{split}.npz") as f:
        saved, lists = f["rows"], f["lists"]
    at = np.searchsorted(saved, rows)
    if not np.array_equal(saved[at.clip(max=len(saved) - 1)], rows):
        raise SystemExit(f"ann_{split}.npz lacks some requested rows")
    return torch.as_tensor(lists[at]).long()


def pool_rows(d: dict, split: str) -> np.ndarray:
    """The request pool: the split's first 50,000 rows in the protocol's sha256 order, in that order."""
    return np.asarray(d[f"{split}_order"][:POOL])


def make_ranker(model, cfg: dict, fields: dict, max_batch: int, slots: int) -> RankerServer:
    dtype = DTYPES[cfg["dtype"]]
    cache = HistoryCache(slots, len(model.layers), model.d, dtype, "cuda") if cfg["cache"] else None
    miss = RankerStep(model, dtype, "miss" if cache else None, cache)
    step = miss.with_mode(cfg["cache"]) if cache else miss
    pool = HostPool(fields, max_batch)
    server = RankerServer(step, pool, cfg["graphs"], max_batch)
    if cfg["cache"] == "hit":
        fill_cache(miss, server.slots, pool, np.arange(pool.n))
    return server


def make_cascade(
    tiger, index, vectors, model, cfg: dict, fields: dict, width: int, max_batch: int
) -> Cascade:
    dtype = DTYPES[cfg["dtype"]]
    pool = HostPool(fields, max_batch)
    cache = HistoryCache(pool.n, len(model.layers), model.d, dtype, "cuda")
    return Cascade(tiger, index, vectors, RankerStep(model, dtype, "miss", cache), pool, width, max_batch)


def warm(server, max_batch: int) -> None:
    if max_batch == 1:
        for i in range(BATCH1_WARM):
            server(np.array([i]))
    else:
        loadgen.batch_times(server, max_batch, 0, WARM_BATCHES)


def save_raw(tag: str, raw: dict) -> str:
    path = RUNS / "raw" / f"{tag}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **raw)
    return str(path.relative_to(common.ROOT))


def stage_summary(raw: dict) -> dict:
    """Per-batch stage times (ms) of a cascade run: quantiles and each stage's share of batch time."""
    if "batch_ms" not in raw or "beam_ms" not in raw:
        return {}
    total = float(raw["batch_ms"].sum())
    out = {}
    for k in Cascade.STAGES:
        v = raw[k].astype(np.float64)
        q = np.quantile(v, (0.5, 0.99))
        out[k] = {
            "p50": float(q[0]),
            "p99": float(q[1]),
            "mean": float(v.mean()),
            "share": float(v.sum() / total),
        }
    return out


def gpu_record(t0: float, bg: Background, git: dict) -> dict:
    return {
        "wall_time_s": round(time.time() - t0, 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "background": bg.summary(),
        "environment": common.environment(),
        "git": git,
        "protocol": PROTOCOL_VERSION,
    }


def measure(server, max_batch: int, tag: str, batch1: bool = True) -> dict:
    """The protocol's measurements of one configuration: batch-1 latency, saturation, the bisection for the
    QPS at the SLO and its confirmation runs. Raw arrays go to data/runs/r5/raw/<tag>/."""
    warm(server, max_batch)
    out: dict = {"max_batch": max_batch, "window_s": WINDOW if max_batch > 1 else 0.0}
    if batch1:
        b1, b1_raw = loadgen.sequential_ms(lambda i: server(np.array([i])), BATCH1_N, BATCH1_WARM)
        out["batch1"] = b1 | {
            "raw": save_raw(f"{tag}/batch1", {"latency_ms": (b1_raw * 1e3).astype(np.float32)})
        }
    out["saturation"] = loadgen.saturation(server, max_batch, SAT_S)
    window = WINDOW if max_batch > 1 else 0.0

    def run(rate: float, seconds: float, seed: int, name: str) -> dict:
        s, raw = loadgen.open_loop_run(server, rate, max_batch, window, SLO, WARM_S, seconds, seed)
        stages = stage_summary(raw)
        print(
            f"{tag} {name} rate {rate:.1f} p99 {s.get('p99_ms', float('nan')):.2f} pass {s['pass']}",
            flush=True,
        )
        return s | ({"stages": stages} if stages else {}) | {"raw": save_raw(f"{tag}/{name}", raw)}

    lo, bis = loadgen.bisect_rate(
        lambda rate, i: run(rate, BISECT_S, 1000 + i, f"bisect{i}"),
        1.05 * out["saturation"]["qps"],
        BISECT_STEPS,
    )
    out["bisection"] = bis
    qps, conf = None, []
    if lo > 0:
        qps, conf = loadgen.confirm_rate(
            lambda rate, a, j: run(rate, CONFIRM_S, 2000 + 10 * a + j, f"confirm{a}_{j}"),
            lo,
            REPS,
            LOWER,
            TRIES,
        )
    out["confirmation"] = conf
    out["qps_at_slo"] = qps
    out["candidates_per_s"] = CANDS * qps if qps else None
    if qps:
        runs = conf[-1]["runs"]
        out["at_slo"] = {
            "p50_ms": [r["p50_ms"] for r in runs],
            "p99_ms": [r["p99_ms"] for r in runs],
            "p50_ms_median": float(np.median([r["p50_ms"] for r in runs])),
            "p99_ms_median": float(np.median([r["p99_ms"] for r in runs])),
            "mean_batch": [r["mean_batch"] for r in runs],
        }
    return out


# ----------------------------------------------------------------------------------------------- the commands


def cmd_projection(_args) -> None:
    """Times a few batches of every kind of work, then projects each command's GPU time."""
    setup()
    t0, git = time.time(), resume.code_state()
    with Background(LOGS / "projection_background.log") as bg:
        t = time.time()
        d = scale.load("full")
        tiger, _inp, _sums = load_tiger(d)
        model, _ = load_hstu_sid(d)
        index, vectors, _ = load_index()
        load_s = time.time() - t
        rows = pool_rows(d, "test")[:400]
        req = build_requests(d, "test", rows, tiger.layout.user_offset)
        g_items, g_user = req.items.cuda(), req.user_tok.cuda()
        beam1, beam128 = {}, {}
        for w in WIDTHS:

            def f(i, w=w):
                beam_lists(tiger, g_items[i : i + 1], g_user[i : i + 1], w)
                torch.cuda.synchronize()

            beam1[w] = loadgen.sequential_ms(f, 20, 5)[0]["mean_ms"]
            beam_lists(tiger, g_items[:128], g_user[:128], w)
            torch.cuda.synchronize()
            s = time.perf_counter()
            for j in (128, 256):
                beam_lists(tiger, g_items[j : j + 128], g_user[j : j + 128], w)
            torch.cuda.synchronize()
            beam128[w] = 256 / (time.perf_counter() - s)
        fields = req.fields() | {"cands": ann_for("test", rows)}
        eager = make_ranker(model, ladder_plan({})["fp32"], fields, 1, len(rows))
        eager_ms = loadgen.sequential_ms(lambda i: eager(np.array([i])), 200, 20)[0]["mean_ms"]
        casc = make_cascade(tiger, index, vectors, model, final_ranker({}), req.fields(), 200, 1)
        e2e_ms = loadgen.sequential_ms(lambda i: casc(np.array([i])), 20, 5)[0]["mean_ms"]
    # Each open-loop configuration: saturation, 7 bisection runs, 3 confirmation runs (worst: 4 more tries).
    run_s = SAT_S + BISECT_STEPS * (WARM_S + BISECT_S) + REPS * (WARM_S + CONFIRM_S)
    retry_s = TRIES * REPS * (WARM_S + CONFIRM_S)
    b1_s = lambda ms: (BATCH1_N + BATCH1_WARM) * ms / 1e3  # noqa: E731
    valid_rows = r4c.SAMPLE
    plan = {
        "beam-valid": load_s + sum(valid_rows / beam128[w] for w in (20, 50)) + 120,
        "select": load_s
        + 2 * 11 * (SWEEP_N + SWEEP_WARM) * 0.005
        + 3 * 8 * (SWEEP_N + SWEEP_WARM) * SLO / 3
        + 60,
        "quality": load_s + 7 * valid_rows * eager_ms / 1e3 + 2 * 60 + CHECK_ROWS * e2e_ms / 1e3,
        "ladder": load_s + len(LADDER) * (run_s + b1_s(eager_ms) + 30),
        "stages": load_s
        + sum(b1_s(beam1[w]) for w in WIDTHS)
        + b1_s(5.0)
        + sum(b1_s(e2e_ms) for _ in WIDTHS),
        "e2e": len(WIDTHS) * (load_s + run_s + 60),
    }
    expected = sum(plan.values()) / 3600 + (time.time() - t0) / 3600
    worst = expected + (len(LADDER) + len(WIDTHS)) * retry_s / 3600
    record = {
        "measured": {
            "load_s": round(load_s, 1),
            "beam_batch1_ms": beam1,
            "beam_rows_per_s_chunk128": beam128,
            "ranker_fp32_eager_batch1_ms": eager_ms,
            "cascade_w200_batch1_ms": e2e_ms,
        },
        "plan_seconds": {k: round(v) for k, v in plan.items()},
        "projected_gpu_hours": round(expected, 2),
        "worst_case_gpu_hours": round(worst, 2),
        "note": "quality's graph steps are projected at the eager speed, an upper bound; worst case: every "
        "confirmation lowered 4 times",
        "stop_at": STOP_GPU_HOURS,
        "stop": expected > STOP_GPU_HOURS,
    } | gpu_record(t0, bg, git)
    common.write_json(RESULTS / "projection.json", record)
    print(record["projected_gpu_hours"], record["worst_case_gpu_hours"], record["measured"])


def shared_items(a: torch.Tensor, b: torch.Tensor, chunk: int = 8192) -> float:
    """Mean over rows of how many items of ``a`` are in ``b`` (empty slots excluded), on the GPU."""
    total = 0
    for s in range(0, len(a), chunk):
        x, y = a[s : s + chunk].cuda(), b[s : s + chunk].cuda()
        total += int(((x[:, :, None] == y[:, None, :]).any(-1) & (x >= 0)).sum())
    return total / len(a)


def cmd_beam_valid(_args) -> None:
    """TIGER seed 0 at widths 20 and 50 over the validation sample; union recall@200 and HSTU-SID's NDCG@10 on
    the union at widths 20, 50 and 200 (R4c's saved lists)."""
    setup()
    t0, git = time.time(), resume.code_state()
    d = scale.load("full")
    with Background(LOGS / "beam_valid_background.log") as bg:
        tiger, inp, sums = load_tiger(d)
        rows, users, cold, order = r4c.sample_rows(d, "valid", "cuda")
        lists, seconds = {}, {}
        for w in (20, 50):
            out = []
            torch.cuda.synchronize()
            t = time.time()
            for s in range(0, len(rows), r4c.TIGER_CHUNK):
                ex, u = rows.take(slice(s, s + r4c.TIGER_CHUNK)), users[s : s + r4c.TIGER_CHUNK]
                x = inp.encoder(ex, u)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    items, _ = constrained_beam_search(
                        tiger.model, x, x > 0, tiger.trie, tiger.layout, beam=w, exclude=inp.history(ex)
                    )
                out.append(items.int().cpu())
            seconds[w] = round(time.time() - t, 1)
            lists[w] = torch.cat(out).long()
            RUNS.mkdir(parents=True, exist_ok=True)
            np.savez(RUNS / f"tiger_valid_w{w}.npz", rows=order, lists=lists[w].int().numpy())
        with np.load(r4c.RUNS / f"tiger_valid_seed{SEED}.npz") as f:
            if not np.array_equal(f["rows"], order):
                raise SystemExit("R4c's TIGER lists cover other rows")
            lists[200] = torch.as_tensor(f["plain"]).long()
        del tiger
        torch.cuda.empty_cache()
        ann = r4c.load_lists(r4c.RUNS / "ann_valid.npz", order)
        target = inp.items[rows.end].cpu()
        cold_c = cold.cpu()
        model, rsums, vocab, v = load_hstu_sid(d, with_vocab=True)
        sums |= rsums
        flat = scale.Flat(d, "cuda")
        cfg = {"k_tiger": K_TIGER, "backfill": BACKFILL}
        widths = {}
        for w in WIDTHS:
            u = r4c.union_on(lists[w], ann, cfg, r4c.LIST, "cuda")
            ranks, _ = r4c.rerank(model, flat, rows, u, vocab, v, full=False)
            widths[str(w)] = {
                "tiger_alone_recall@w": cascade.recall_split(cascade.hits(lists[w], target), cold_c),
                "union_recall@200": cascade.recall_split(cascade.hits(u, target), cold_c),
                "tiger_items_in_union": shared_items(u, lists[w]),
                "hstu_sid": split_metrics(ranks.cpu(), cold_c),
                "beam_seconds_chunk128": seconds.get(w),
                "lists": str((RUNS / f"tiger_valid_w{w}.npz").relative_to(common.ROOT))
                if w != 200
                else str((r4c.RUNS / f"tiger_valid_seed{SEED}.npz").relative_to(common.ROOT)),
            }
    record = {
        "split": "valid",
        "rows": len(order),
        "seed": SEED,
        "union": cfg | {"budget": r4c.LIST},
        "checkpoint": sums,
        "widths": widths,
        "r4c_record_hstu_sid_ndcg@10": load_json(r4c.RESULTS / "valid" / f"rerank_seed{SEED}.json")[
            "rankers"
        ]["hstu-sid"]["cascade"]["all"]["ndcg@10"],
    } | gpu_record(t0, bg, git)
    common.write_json(RESULTS / "valid" / "beam_widths.json", record)
    print({w: (r["union_recall@200"]["all"], r["hstu_sid"]["all"]["ndcg@10"]) for w, r in widths.items()})


def cmd_select(_args) -> None:
    """B_max of the batched ranker configurations and of the cascade at each width, on the validation pool."""
    setup()
    t0, git = time.time(), resume.code_state()
    d = scale.load("full")
    keeps = keeps_from_quality()
    plan = ladder_plan(keeps)
    with Background(LOGS / "select_background.log") as bg:
        tiger, _inp, sums = load_tiger(d)
        model, rsums = load_hstu_sid(d)
        index, vectors, index_rec = load_index()
        rows = pool_rows(d, "valid")
        req = build_requests(d, "valid", rows, tiger.layout.user_offset)
        fields = req.fields() | {"cands": ann_for("valid", rows)}
        out = {"ranker": {}, "cascade": {}}

        def sweep(server):
            warm(server, 8)
            return loadgen.choose_max_batch(
                lambda s: loadgen.batch_times(server, s, SWEEP_N, SWEEP_WARM), SLO / 3, LARGEST
            )

        for name in ("batching-miss", "batching-hit"):
            if plan[name] is None:
                continue
            server = make_ranker(model, plan[name], fields, LARGEST, len(rows))
            best, sw = sweep(server)
            out["ranker"][name] = {"config": plan[name], "max_batch": best, "sweep": sw}
            print(name, best, flush=True)
            del server
            torch.cuda.empty_cache()
        for w in WIDTHS:
            casc = make_cascade(tiger, index, vectors, model, final_ranker(keeps), req.fields(), w, LARGEST)
            best, sw = sweep(casc)
            out["cascade"][f"w{w}"] = {"width": w, "max_batch": best, "sweep": sw}
            print(w, best, flush=True)
            del casc
            torch.cuda.empty_cache()
    record = (
        {
            "split": "valid",
            "pool": len(rows),
            "limit_ms": SLO / 3 * 1e3,
            "rule": "largest power of two up to 1024 whose p99 batch time over 50 batches (after 10)"
            " is at most SLO/3",
            "keeps": keeps,
            "checkpoint": sums | rsums,
            "index": index_rec,
        }
        | out
        | gpu_record(t0, bg, git)
    )
    common.write_json(RESULTS / "valid" / "select.json", record)


def top10(scores: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
    """The 10 best candidates by score, ties to the lower catalog index."""
    s = scores.masked_fill(cands <= 0, -float("inf"))
    by_item = torch.argsort(cands, dim=1, stable=True)
    order = torch.argsort(s.gather(1, by_item), dim=1, descending=True, stable=True)
    return cands.gather(1, by_item.gather(1, order[:, :10]))


def overlap10(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per row |a ∩ b| / 10 of two [B, 10] lists."""
    return (a[:, :, None] == b[:, None, :]).any(-1).sum(1).double() / 10


def check_features(d: dict, req, rows: np.ndarray, layout, codes) -> dict:
    """The serving inputs equal R4c's: the ranker window and times, TIGER's encoder input and history mask,
    the ANN window."""
    from sidrec.serve.requests import history_mask, query_times, tiger_input
    from sidrec.train.scale_idrec import MAX_LEN
    from sidrec.train.scale_tiger import Inputs

    ex, users, _cold, order = r4c.sample_rows(d, "valid", "cpu")
    if not np.array_equal(order, rows):
        raise SystemExit("rows differ from the validation sample")
    n = 4096
    ex, users = ex.take(slice(0, n)), users[:n]
    flat = scale.Flat(d, "cpu")
    ident = torch.arange(int(d["n_items"]) + 1)
    b = flat.id_inputs(ex, ident, MAX_LEN, train=False)
    inp = Inputs(d, layout.item_tokens(codes), layout, "cpu")
    items, ts, req_ts = req.items[:n], req.ts[:n], req.req_ts[:n]
    checks = {
        "ranker_items": torch.equal(items, b["inputs"]),
        "ranker_input_ts": torch.equal(ts, b["input_ts"]),
        "ranker_query_ts": torch.equal(query_times(items, ts, req_ts), b["query_ts"]),
        "target": torch.equal(req.target[:n], b["target"]),
        "tiger_encoder": torch.equal(
            tiger_input(items, req.user_tok[:n], inp.tokens), inp.encoder(ex, users)
        ),
        "history_mask": torch.equal(history_mask(items), inp.history(ex)),
        "ann_window": torch.equal(items, r4c.history_window(d, ex)),
    }
    if not all(checks.values()):
        raise SystemExit(f"serving inputs differ from R4c's: {checks}")
    return checks | {"rows_checked": n}


def score_all(server: RankerServer, n: int, batch: int) -> torch.Tensor:
    out = []
    for s in range(0, n, batch):
        _, scores = server.run(np.arange(s, min(s + batch, n), dtype=np.int64))
        out.append(torch.from_numpy(scores))
    return torch.cat(out)


def cmd_quality(_args) -> None:
    """Each ladder step's ranking on the validation sample against step 1 (fp32), and the serving-path
    check."""
    setup()
    t0, git = time.time(), resume.code_state()
    d = scale.load("full")
    select = load_json(RESULTS / "valid" / "select.json")
    with Background(LOGS / "quality_background.log") as bg:
        model, sums = load_hstu_sid(d)
        codes, layout = load_ids()
        rows = np.sort(np.asarray(d["valid_order"][: r4c.SAMPLE]))
        req = build_requests(d, "valid", rows, layout.user_offset)
        features = check_features(d, req, rows, layout, codes)
        sel = r4c.load_json(r4c.RESULTS / "selection.json")
        with np.load(r4c.RUNS / f"tiger_valid_seed{SEED}.npz") as f:
            if not np.array_equal(f["rows"], rows):
                raise SystemExit("R4c's TIGER lists cover other rows")
            plain = torch.as_tensor(f["plain"]).long()
        ann = r4c.load_lists(r4c.RUNS / "ann_valid.npz", rows)
        cands = r4c.selected_union({"off": plain}, ann, sel, r4c.LIST, "cuda")
        fields = req.fields() | {"cands": cands}
        keeps, steps, ref, tops = {}, {}, None, {}
        for name in LADDER:
            cfg = ladder_plan(keeps)[name]
            if cfg is None:
                continue
            batch = select["ranker"][name]["max_batch"] if cfg["batching"] else 1
            server = make_ranker(model, cfg, fields, batch, len(rows))
            t = time.time()
            scores = score_all(server, len(rows), batch)
            seconds = round(time.time() - t, 1)
            ranks = cascade.candidate_ranks(scores, cands, req.target, cands > 0)
            top = top10(scores, cands)
            tops[name] = top
            metrics = split_metrics(ranks, req.cold)
            if ref is None:
                ref = {"top": top, "ranks": ranks, "metrics": metrics}
            ndcg0 = ref["metrics"]["all"]["ndcg@10"]
            ov = float(overlap10(top, ref["top"]).mean())
            shift = metrics["all"]["ndcg@10"] - ndcg0
            keeps[name] = bool(ov >= OVERLAP_MIN and abs(shift) <= NDCG_SHIFT_MAX * ndcg0)
            steps[name] = {
                "config": cfg,
                "batch": batch,
                "top10_overlap": ov,
                "top10_identical_rows": float((top == ref["top"]).all(1).double().mean()),
                "ndcg@10_change": shift,
                "ndcg@10_change_relative": shift / ndcg0,
                "rows_target_rank_changed": int((ranks != ref["ranks"]).sum()),
                "keeps_quality": keeps[name],
                "metrics": metrics,
                "seconds": seconds,
            }
            print(name, ov, shift, flush=True)
            del server
            torch.cuda.empty_cache()
        r4c_ndcg = load_json(r4c.RESULTS / "valid" / f"rerank_seed{SEED}.json")["rankers"]["hstu-sid"][
            "cascade"
        ]["all"]["ndcg@10"]
        # The serving-path check: the cascade on the first validation rows at width 200.
        tiger, _inp, tsums = load_tiger(d)
        sums |= tsums
        index, vectors, index_rec = load_index()
        n = CHECK_ROWS
        batch = select["cascade"]["w200"]["max_batch"]
        sub = {k: v[:n] for k, v in req.fields().items()}
        casc = make_cascade(tiger, index, vectors, model, final_ranker(keeps), sub, 200, batch)
        tl, al, cl, tops_c = [], [], [], []
        for s in range(0, n, batch):
            top, _, extra = casc.run(np.arange(s, min(s + batch, n), dtype=np.int64), keep=True)
            tl.append(extra["tiger"])
            al.append(extra["ann"])
            cl.append(extra["cands"])
            tops_c.append(top)
        tl, al, cl = (torch.as_tensor(np.concatenate(x)).long() for x in (tl, al, cl))
        saved_t = plain[:n]
        same_t = (tl == saved_t).all(1)
        check = {
            "rows": n,
            "batch": batch,
            "ann_equal_to_r4c": bool(torch.equal(al, ann[:n])),
            "tiger_lists_identical_share": float(same_t.double().mean()),
            "tiger_lists_overlap": float(
                ((tl[:, :, None] == saved_t[:, None, :]).any(-1) & (tl >= 0)).sum() / (n * 200)
            ),
            "union_identical_share": float((cl == cands[:n].cpu()).all(1).double().mean()),
            "top10_overlap_with_ranker_alone": float(
                overlap10(
                    torch.as_tensor(np.concatenate(tops_c)), tops[final_ranker(keeps)["name"]][:n]
                ).mean()
            ),
        }
    record = {
        "split": "valid",
        "rows": len(rows),
        "candidates": "R4c's selected union for TIGER seed 0 (saved lists)",
        "checkpoint": sums,
        "index": index_rec,
        "features": features,
        "reference": "fp32",
        "keep_rule": {"top10_overlap_min": OVERLAP_MIN, "ndcg@10_change_max_relative": NDCG_SHIFT_MAX},
        "keeps": keeps,
        "steps": steps,
        "fp32_against_r4c_record": {
            "r4c": r4c_ndcg,
            "fp32": ref["metrics"]["all"]["ndcg@10"],
            "difference": ref["metrics"]["all"]["ndcg@10"] - r4c_ndcg,
        },
        "serving_check": check,
    } | gpu_record(t0, bg, git)
    common.write_json(RESULTS / "valid" / "quality.json", record)
    print(keeps, check)


def test_pool(d: dict, offset: int, cands: bool):
    rows = pool_rows(d, "test")
    req = build_requests(d, "test", rows, offset)
    fields = req.fields() | ({"cands": ann_for("test", rows)} if cands else {})
    return rows, req, fields


def cmd_ladder(args) -> None:
    setup()
    d = scale.load("full")
    keeps = load_json(RESULTS / "valid" / "quality.json")["keeps"]
    select = load_json(RESULTS / "valid" / "select.json")
    plan = ladder_plan(keeps)
    model, sums = load_hstu_sid(d)
    _rows, _req, fields = test_pool(d, user_offset(), cands=True)
    for name in args.steps or LADDER:
        cfg = plan[name]
        if cfg is None:
            continue
        t0, git = time.time(), resume.code_state()
        with Background(LOGS / f"ladder_{name}_background.log") as bg:
            batch = select["ranker"][name]["max_batch"] if cfg["batching"] else 1
            server = make_ranker(model, cfg, fields, batch, POOL)
            rec = {"step": name, "config": cfg, "split": "test", "pool": POOL, "checkpoint": sums}
            rec |= measure(server, batch, f"ladder/{name}")
            del server
            torch.cuda.empty_cache()
        rec |= gpu_record(t0, bg, git)
        common.write_json(RESULTS / "test" / "ladder" / f"{name}.json", rec)
        print(name, rec["qps_at_slo"], rec.get("batch1", {}).get("p50_ms"), flush=True)


def cmd_stages(_args) -> None:
    """Batch-1 latency of each stage on its own, and of the whole cascade at each width."""
    setup()
    t0, git = time.time(), resume.code_state()
    d = scale.load("full")
    keeps = load_json(RESULTS / "valid" / "quality.json")["keeps"]
    n = BATCH1_N + BATCH1_WARM
    with Background(LOGS / "stages_background.log") as bg:
        tiger, _inp, sums = load_tiger(d)
        model, rsums = load_hstu_sid(d)
        sums |= rsums
        index, vectors, index_rec = load_index()
        _rows, req, fields = test_pool(d, tiger.layout.user_offset, cands=False)
        g_items, g_user = req.items[:n].cuda(), req.user_tok[:n].cuda()
        out, raw = {}, {}
        kept = {}
        for w in WIDTHS:
            got = []

            def beam(i, w=w, got=got):
                got.append(beam_lists(tiger, g_items[i : i + 1], g_user[i : i + 1], w))
                torch.cuda.synchronize()

            out[f"beam_w{w}"], raw[f"beam_w{w}"] = loadgen.sequential_ms(beam, BATCH1_N, BATCH1_WARM)
            kept[w] = got
        got_ann = []

        def hnsw(i):
            got_ann.append(ann_lists(index, vectors, req.items[i : i + 1]))

        out["hnsw"], raw["hnsw"] = loadgen.sequential_ms(hnsw, BATCH1_N, BATCH1_WARM)
        ann_g = torch.as_tensor(np.concatenate(got_ann)).long().cuda()
        got_u = []

        def union(i):
            got_u.append(cascade.union(kept[200][i], ann_g[i : i + 1], K_TIGER, CANDS, BACKFILL))
            torch.cuda.synchronize()

        out["union"], raw["union"] = loadgen.sequential_ms(union, BATCH1_N, BATCH1_WARM)
        sub = {k: v[:n] for k, v in fields.items()} | {"cands": torch.cat(got_u).cpu()}
        server = make_ranker(model, final_ranker(keeps), sub, 1, n)
        out["ranker"], raw["ranker"] = loadgen.sequential_ms(
            lambda i: server(np.array([i])),  # noqa: F821 (called before the del below)
            BATCH1_N,
            BATCH1_WARM,
        )
        del server
        for w in WIDTHS:
            casc = make_cascade(tiger, index, vectors, model, final_ranker(keeps), fields, w, 1)
            times = []

            def e2e(i, casc=casc, times=times):
                times.append(casc(np.array([i])))

            out[f"cascade_w{w}"], raw[f"cascade_w{w}"] = loadgen.sequential_ms(e2e, BATCH1_N, BATCH1_WARM)
            parts = {k: np.array([t[k] for t in times[BATCH1_WARM:]]) for k in Cascade.STAGES}
            out[f"cascade_w{w}"]["stages_ms"] = {
                k: {"p50": float(np.median(v)), "p99": float(np.quantile(v, 0.99))} for k, v in parts.items()
            }
            del casc
            torch.cuda.empty_cache()
    paths = {k: save_raw(f"stages/{k}", {"latency_ms": (v * 1e3).astype(np.float32)}) for k, v in raw.items()}
    record = {
        "split": "test",
        "requests": n,
        "ranker": final_ranker(keeps),
        "checkpoint": sums,
        "index": index_rec,
        "batch1": out,
        "raw": paths,
    } | gpu_record(t0, bg, git)
    common.write_json(RESULTS / "test" / "stages_batch1.json", record)
    print({k: round(v["p50_ms"], 3) for k, v in out.items()})


def cmd_e2e(args) -> None:
    setup()
    t0, git = time.time(), resume.code_state()
    d = scale.load("full")
    keeps = load_json(RESULTS / "valid" / "quality.json")["keeps"]
    select = load_json(RESULTS / "valid" / "select.json")
    w = args.width
    batch = select["cascade"][f"w{w}"]["max_batch"]
    with Background(LOGS / f"e2e_w{w}_background.log") as bg:
        tiger, _inp, sums = load_tiger(d)
        model, rsums = load_hstu_sid(d)
        sums |= rsums
        index, vectors, index_rec = load_index()
        _rows, _req, fields = test_pool(d, tiger.layout.user_offset, cands=False)
        casc = make_cascade(tiger, index, vectors, model, final_ranker(keeps), fields, w, batch)
        rec = {
            "width": w,
            "split": "test",
            "pool": POOL,
            "ranker": final_ranker(keeps),
            "checkpoint": sums,
            "index": index_rec,
        }
        rec |= measure(casc, batch, f"e2e/w{w}", batch1=False)
    rec |= gpu_record(t0, bg, git)
    common.write_json(RESULTS / "test" / f"e2e_w{w}.json", rec)
    print(w, rec["qps_at_slo"], rec.get("at_slo"))


COMMANDS = {
    "projection": cmd_projection,
    "beam-valid": cmd_beam_valid,
    "select": cmd_select,
    "quality": cmd_quality,
    "ladder": cmd_ladder,
    "stages": cmd_stages,
    "e2e": cmd_e2e,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--width", type=int, default=200, choices=WIDTHS)
    parser.add_argument("--steps", nargs="*", choices=LADDER)
    args = parser.parse_args()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
