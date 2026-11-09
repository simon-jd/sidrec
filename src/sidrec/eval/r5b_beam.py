"""R5b: TIGER's beam made fast, and the cascade re-measured (docs/protocol.md, entry "2026-11-08: R5b fast
TIGER beam").

    python -m sidrec.eval.r5b_beam projection        # short timed pass and the GPU-hour projection
    python -m sidrec.eval.r5b_beam profile           # torch.profiler on the reference beam and the final one
    python -m sidrec.eval.r5b_beam equivalence       # fp32: each step's lists against the reference (valid)
    python -m sidrec.eval.r5b_beam cutoffs           # fp32: each mismatch traced through the four steps
    python -m sidrec.eval.r5b_beam quality           # bf16: union recall@200 and HSTU-SID NDCG@10 (valid)
    python -m sidrec.eval.r5b_beam ladder            # the beam alone at each step and width (test pool)
    python -m sidrec.eval.r5b_beam select            # B_max of the cascade at each width (validation pool)
    python -m sidrec.eval.r5b_beam stages            # batch-1 stage latencies (test pool)
    python -m sidrec.eval.r5b_beam e2e --width 200   # the cascade under load (test pool)

Everything not named in the entry follows R5: the constants, load generator, B_max rule, bisection and
confirmations are R5's (sidrec.eval.r5_serving). Records go to results/r5b/, raw arrays to data/runs/r5b/.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections import defaultdict

import numpy as np
import torch

from sidrec.data import scale
from sidrec.eval import r4c_cascade as r4c
from sidrec.eval import r5_serving as r5
from sidrec.eval.metrics import split_metrics
from sidrec.models import cascade
from sidrec.models.tiger import constrained_beam_search
from sidrec.serve import loadgen
from sidrec.serve.beam import LADDER, FastBeam, GraphedBeam, options, serve
from sidrec.serve.machine import Background
from sidrec.serve.pipeline import BACKFILL, K_TIGER, Cascade, HostPool, ann_lists
from sidrec.serve.ranker import HistoryCache, RankerStep
from sidrec.serve.requests import CANDS, build_requests, history_mask, tiger_input
from sidrec.train import common, resume

RESULTS = common.ROOT / "results" / "r5b"
RUNS = common.ROOT / "data" / "runs" / "r5b"
LOGS = common.ROOT / "data" / "logs" / "r5b"
PROTOCOL_VERSION = "2026-11-05 with the R2, R3, R4a, R4b, R4c, R5 and R5b clarifications"

WIDTHS = r5.WIDTHS
SERVE_DTYPE = torch.bfloat16
EQUIV_ROWS, EQUIV_CHUNK = 2048, 128
QUALITY_CHUNK = r4c.TIGER_CHUNK  # 128, as R4c and R5 ran their validation beams
LADDER_BATCH = 64
PROFILE_CALLS = 10
STOP_GPU_HOURS = 4.0


def load_json(path):
    return json.loads(path.read_text())


def r5_keeps() -> dict:
    return load_json(r5.RESULTS / "valid" / "quality.json")["keeps"]


def save_raw(tag: str, raw: dict) -> str:
    path = RUNS / "raw" / f"{tag}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **raw)
    return str(path.relative_to(common.ROOT))


def gpu_record(t0: float, bg: Background, git: dict) -> dict:
    return r5.gpu_record(t0, bg, git) | {"protocol": PROTOCOL_VERSION}


# ------------------------------------------------------------------------------------------------- the beams


def reference(tiger, width: int, dtype=SERVE_DTYPE):
    """R4c's beam code, as R5 served it in bf16 (``beam_lists``) or without autocast in fp32; returns
    (lists, scores)."""

    def call(items, user_tok):
        x = tiger_input(items, user_tok, tiger.tokens)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            return constrained_beam_search(
                tiger.model, x, x > 0, tiger.trie, tiger.layout, beam=width, exclude=history_mask(items)
            )

    return call


def make_beam(tiger, width: int, step: str, dtype=SERVE_DTYPE, max_batch: int = 1, example=None, lazy=False):
    """A ladder step's beam: a callable (items, user_tok) -> (lists, scores) on the GPU."""
    if step == "reference":
        return reference(tiger, width, dtype)
    fast = FastBeam(tiger.model, tiger.trie, tiger.layout, width, dtype, **options(step))
    if step != "graphs":
        return lambda items, user_tok: serve(fast, items, user_tok, tiger.tokens)
    return GraphedBeam(fast, tiger.tokens, *example, max_batch, lazy=lazy)


def example_request(items: torch.Tensor, user_tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return items[:1].cuda(), user_tok[:1].cuda()


def make_cascade(
    tiger, index, vectors, model, cfg: dict, fields: dict, width: int, max_batch: int, lazy=False
):
    """R5's cascade with the final beam (CUDA graphs, bf16) in place of R5's beam."""
    pool = HostPool(fields, max_batch)
    cache = HistoryCache(pool.n, len(model.layers), model.d, r5.DTYPES[cfg["dtype"]], "cuda")
    beam = make_beam(
        tiger,
        width,
        "graphs",
        SERVE_DTYPE,
        max_batch,
        example_request(pool.fields["items"], pool.fields["user_tok"]),
        lazy=lazy,
    )
    rank = RankerStep(model, r5.DTYPES[cfg["dtype"]], "miss", cache)
    return Cascade(tiger, index, vectors, rank, pool, width, max_batch, beam=lambda i, u: beam(i, u)[0])


def run_chunks(
    beam, items: torch.Tensor, user_tok: torch.Tensor, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lists and scores of every row, in chunks, on the host."""
    lists, scores = [], []
    for s in range(0, len(items), chunk):
        a, b = beam(items[s : s + chunk].cuda(), user_tok[s : s + chunk].cuda())
        lists.append(a.cpu())
        scores.append(b.float().cpu())
    return torch.cat(lists), torch.cat(scores)


# --------------------------------------------------------------------------------------------------- profile


@contextlib.contextmanager
def labelled(model):
    """Names the reference model's pieces for the profiler, without changing what they compute."""
    patched = []

    def wrap(obj, attr, name):
        orig = getattr(obj, attr)

        def f(*a, **k):
            with torch.profiler.record_function(name):
                return orig(*a, **k)

        setattr(obj, attr, f)
        patched.append((obj, attr))

    wrap(model, "encode", "encoder")
    wrap(model, "decode", "decoder")
    wrap(model.decoder, "forward", "decoder_layers")
    for blk in model.decoder.blocks:
        wrap(blk.self_attn, "forward", "dec_self_attn")
        wrap(blk.cross_attn, "forward", "dec_cross_attn")
        wrap(blk.cross_attn.k, "forward", "dec_cross_kv")
        wrap(blk.cross_attn.v, "forward", "dec_cross_kv")
        wrap(blk.ff, "forward", "dec_ff")
    try:
        yield
    finally:
        for obj, attr in patched:
            delattr(obj, attr)


LAUNCH = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx", "cudaGraphLaunch")
SYNC = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")
LABELS = ("encoder", "decoder", "decoder_layers", "dec_self_attn", "dec_cross_attn", "dec_cross_kv", "dec_ff")


def own_kernel_us(e) -> float:
    """Kernel time launched inside a CPU event, through its descendants (the profiler's annotations of nested
    labels are left out)."""
    total, stack = 0.0, list(e.cpu_children)
    while stack:
        c = stack.pop()
        if c.name not in LABELS:
            total += c.self_device_time_total
        stack.extend(c.cpu_children)
    return total


def profile_one(fn, calls: int, synced_wall_ms: float) -> dict:
    """One configuration under torch.profiler: per call, kernel time and count, launches, host syncs and the
    labelled pieces."""
    from torch.autograd import DeviceType

    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=acts) as prof:
        for _ in range(calls):
            fn()
            torch.cuda.synchronize()
    evs = prof.events()
    dev = [e for e in evs if e.device_type == DeviceType.CUDA and e.name not in LABELS]
    kernels = [e for e in dev if not e.name.startswith(("Memcpy", "Memset"))]
    copies = [e for e in dev if e.name.startswith(("Memcpy", "Memset"))]
    cpu = [e for e in evs if e.device_type == DeviceType.CPU]
    count = defaultdict(int)
    for e in cpu:
        count[e.name] += 1
    by_kernel = defaultdict(lambda: [0.0, 0])
    for e in kernels:
        by_kernel[e.name][0] += e.time_range.elapsed_us()
        by_kernel[e.name][1] += 1
    gpu_ms = sum(e.time_range.elapsed_us() for e in kernels) / calls / 1e3
    labels = defaultdict(lambda: {"cpu_ms": 0.0, "gpu_ms": 0.0, "calls": 0.0})
    for e in cpu:
        if e.name in LABELS:
            lab = labels[e.name]
            lab["cpu_ms"] += e.time_range.elapsed_us() / calls / 1e3
            lab["gpu_ms"] += own_kernel_us(e) / calls / 1e3
            lab["calls"] += 1 / calls
    top = sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:8]
    out = {
        "wall_ms": synced_wall_ms,
        "gpu_kernel_ms": gpu_ms,
        "gpu_busy_share": gpu_ms / synced_wall_ms,
        "kernels": len(kernels) / calls,
        "copies_and_memsets": len(copies) / calls,
        "copies_and_memsets_ms": sum(e.time_range.elapsed_us() for e in copies) / calls / 1e3,
        "launch_calls": sum(count[n] for n in LAUNCH) / calls,
        "host_syncs": {n: count[n] / calls for n in SYNC if count[n]},
        "item_calls": count["aten::_local_scalar_dense"] / calls,
        "nonzero_calls": count["aten::nonzero"] / calls,
        "labels": dict(labels),
        "top_kernels": [{"name": k[:120], "ms": v[0] / calls / 1e3, "count": v[1] / calls} for k, v in top],
    }
    if "encoder" in labels and "decoder" in labels:
        enc, dec = labels["encoder"]["gpu_ms"], labels["decoder"]["gpu_ms"]
        out["split_gpu_ms"] = {"encoder": enc, "decoder": dec, "search_ops": gpu_ms - enc - dec}
    return out


def cmd_profile(_args) -> None:
    """The reference beam at batch 1 and R5's B_max for each width, and at batch 64; the final beam alike."""
    r5.setup()
    t0, git = time.time(), resume.code_state()
    d = scale.load("full")
    r5sel = load_json(r5.RESULTS / "valid" / "select.json")["cascade"]
    with Background(LOGS / "profile_background.log") as bg:
        tiger, _inp, sums = r5.load_tiger(d)
        rows = r5.pool_rows(d, "test")[:LADDER_BATCH]
        req = build_requests(d, "test", rows, tiger.layout.user_offset)
        items, user = req.items.cuda(), req.user_tok.cuda()
        configs = []
        for w in WIDTHS:
            for b in sorted({1, r5sel[f"w{w}"]["max_batch"], LADDER_BATCH}):
                configs.append((w, b))
        out = {}
        for step in ("reference", "graphs"):
            for w, b in configs:
                beam = make_beam(tiger, w, step, SERVE_DTYPE, b, example_request(items, user))

                def fn(beam=beam, b=b):
                    beam(items[:b], user[:b])

                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                calls = PROFILE_CALLS if b < LADDER_BATCH or step == "graphs" else 3
                wall = []
                for _ in range(20):
                    s = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    wall.append(time.perf_counter() - s)
                ctx = labelled(tiger.model) if step == "reference" else contextlib.nullcontext()
                with ctx:
                    rec = profile_one(fn, calls, float(np.median(wall)) * 1e3)
                out[f"{step}_w{w}_b{b}"] = {
                    "step": step,
                    "width": w,
                    "batch": b,
                    "profiled_calls": calls,
                } | rec
                print(
                    step,
                    w,
                    b,
                    round(rec["wall_ms"], 3),
                    round(rec["gpu_kernel_ms"], 3),
                    rec["kernels"],
                    flush=True,
                )
                del beam
                torch.cuda.empty_cache()
    record = {
        "split": "test",
        "requests": "the first 64 rows of the test pool",
        "dtype": "bf16",
        "checkpoint": sums,
        "note": "wall_ms is the median of 20 synchronized calls without the profiler; the rest is per call "
        "under torch.profiler, which also counts the one synchronize per call. Label times are the kernels "
        "launched inside each piece of the reference (all four decoder steps together). split_gpu_ms: the "
        "encoder, the decoder, and the rest of the search (inputs, top-k, trie gathers, history removal).",
        "configs": out,
    } | gpu_record(t0, bg, git)
    common.write_json(RESULTS / "profile.json", record)


# ---------------------------------------------------------------------------------------- equivalence (fp32)


NEAR_TIE = 1e-4


def compare(ref: tuple, got: tuple, show: int = 20) -> dict:
    """Lists and scores of a beam against the reference's: identical rows and, for the others, where they
    part.

    At the first position where a row's lists part, the tie gap is the difference between the reference's
    score there and the beam's; a mismatch is a near-tie if it is at most NEAR_TIE."""
    (rl, rs), (gl, gs) = ref, got
    same = (rl == gl).all(1)
    both = torch.isfinite(rs) & torch.isfinite(gs) & (rl == gl)
    diff = (rs - gs).abs().masked_fill(~both, 0)
    bad = torch.nonzero(~same).flatten()
    pos = (rl[bad] != gl[bad]).int().argmax(1)
    gap = (rs[bad, pos] - gs[bad, pos]).abs().nan_to_num(nan=float("inf"))
    near = gap <= NEAR_TIE
    out = {
        "rows": len(rl),
        "identical_rows": int(same.sum()),
        "identical_share": float(same.double().mean()),
        "items_shared": float(
            ((gl[:, :, None] == rl[:, None, :]).any(-1) & (gl >= 0)).sum() / (rl >= 0).sum()
        ),
        "max_abs_score_diff_same_items": float(diff.max()),
        "mismatched_rows": len(bad),
        "near_tie_rows": int(near.sum()),
        "max_tie_gap": float(gap.max()) if len(bad) else None,
        "first_position_min": int(pos.min()) if len(bad) else None,
        "first_position_median": float(pos.float().median()) if len(bad) else None,
        "mismatches": [],
    }
    for j in range(min(show, len(bad))):
        r, p = int(bad[j]), int(pos[j])
        q = slice(max(p - 1, 0), p + 2)
        out["mismatches"].append(
            {
                "row": r,
                "first_position": p,
                "reference_items": rl[r, q].tolist(),
                "items": gl[r, q].tolist(),
                "reference_scores": rs[r, q].tolist(),
                "scores": gs[r, q].tolist(),
                "tie_gap": float(gap[j]),
            }
        )
    return out


def cmd_equivalence(_args) -> None:
    """fp32: each step against the reference in chunks of 128, the final step at batch 1 too; and the final
    step in bf16 against R5's bf16 beam at batch 1. The first 2,048 rows of the validation pool."""
    r5.setup()
    t0, git = time.time(), resume.code_state()
    d = scale.load("full")
    with Background(LOGS / "equivalence_background.log") as bg:
        tiger, _inp, sums = r5.load_tiger(d)
        rows = r5.pool_rows(d, "valid")[:EQUIV_ROWS]
        req = build_requests(d, "valid", rows, tiger.layout.user_offset)
        items, user = req.items, req.user_tok
        ex = example_request(items, user)
        out = {}
        for w in WIDTHS:
            res = {}
            for dtype, name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
                ref1 = run_chunks(reference(tiger, w, dtype), items, user, 1)
                g = make_beam(tiger, w, "graphs", dtype, EQUIV_CHUNK, ex)
                res[f"{name}_batch1_graphs"] = compare(ref1, run_chunks(g, items, user, 1))
                if name == "fp32":
                    ref = run_chunks(reference(tiger, w, dtype), items, user, EQUIV_CHUNK)
                    res["fp32_reference_chunk128_vs_batch1"] = compare(ref1, ref)
                    for step in LADDER[1:]:
                        beam = g if step == "graphs" else make_beam(tiger, w, step, dtype)
                        res[f"fp32_chunk128_{step}"] = compare(
                            ref, run_chunks(beam, items, user, EQUIV_CHUNK)
                        )
                del g
                torch.cuda.empty_cache()
            out[str(w)] = res
            print(w, {k: v["identical_share"] for k, v in res.items()}, flush=True)
    record = {
        "split": "valid",
        "rows": EQUIV_ROWS,
        "sample": "the first 2,048 rows of the validation pool (the protocol's sha256 order)",
        "chunk": EQUIV_CHUNK,
        "tf32": False,
        "checkpoint": sums,
        "widths": out,
    } | gpu_record(t0, bg, git)
    common.write_json(RESULTS / "valid" / "equivalence.json", record)


@contextlib.contextmanager
def traced_topk(log: list):
    """Records every top-k of the search (the candidates [B, beam * size] of a step): the kept values and
    indices, and the best value left out. The search's own top-k is unchanged."""
    orig = torch.Tensor.topk

    def topk(self, k, dim=-1, largest=True, sorted=True):  # noqa: A002
        out = orig(self, k, dim=dim, largest=largest, sorted=sorted)
        if self.dim() == 2 and dim == 1 and k < self.shape[1]:
            nxt = orig(self, k + 1, dim=1).values[:, k]
            log.append((out.values.float().cpu(), out.indices.cpu(), nxt.float().cpu()))
        return out

    torch.Tensor.topk = topk
    try:
        yield
    finally:
        torch.Tensor.topk = orig


def kept_prefixes(log: list, row: int, sizes) -> list[dict]:
    """Per step, the live prefixes a beam kept for one row of its batch, with their scores."""
    out, prev = [], None
    for (values, indices, nxt), size in zip(log, sizes, strict=True):
        v, i = values[row], indices[row]
        src, code = (i // size).tolist(), (i % size).tolist()
        pref = [(c,) if prev is None else prev[s] + (c,) for s, c in zip(src, code, strict=True)]
        live = torch.isfinite(v)
        out.append(
            {
                "prefixes": pref,
                "scores": {p: float(x) for p, x, ok in zip(pref, v, live, strict=True) if ok},
                "cutoff_gap": float(v[-1] - nxt[row]),
            }
        )
        prev = pref
    return out


def first_split(ref: list[dict], got: list[dict]) -> dict:
    """The first step at which two beams keep different prefixes, and the gap there between the weakest prefix
    only the reference kept and the strongest only the other beam kept."""
    for step, (a, b) in enumerate(zip(ref, got, strict=True)):
        only_ref, only_got = set(a["scores"]) - set(b["scores"]), set(b["scores"]) - set(a["scores"])
        if only_ref or only_got:
            lo = min((a["scores"][p] for p in only_ref), default=float("inf"))
            hi = max((b["scores"][p] for p in only_got), default=float("-inf"))
            return {
                "step": step + 1,
                "swapped": max(len(only_ref), len(only_got)),
                "gap": abs(lo - hi),
                "reference_cutoff_gap": a["cutoff_gap"],
            }
    return {"step": None}


def traced_run(beam, items, user, chunk: int) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Lists, scores and the per-chunk top-k logs of an eager beam."""
    lists, scores, logs = [], [], []
    for s in range(0, len(items), chunk):
        log = []
        with traced_topk(log):
            a, b = beam(items[s : s + chunk].cuda(), user[s : s + chunk].cuda())
        lists.append(a.cpu())
        scores.append(b.float().cpu())
        logs.append(log)
    return torch.cat(lists), torch.cat(scores), logs


def classify(ref, got, chunks: tuple[int, int], sizes) -> dict:
    """Each mismatched row of two traced runs (run in chunks of ``chunks``): an order-only near-tie, a cutoff
    near-tie, or a defect."""
    (rl, rs, rlog), (gl, gs, glog) = ref, got
    bad = torch.nonzero(~(rl == gl).all(1)).flatten().tolist()
    rows, counts = [], {"order_near_tie": 0, "cutoff_near_tie": 0, "defect": 0}
    for r in bad:
        (c, j), (cg, jg) = divmod(r, chunks[0]), divmod(r, chunks[1])
        split = first_split(kept_prefixes(rlog[c], j, sizes), kept_prefixes(glog[cg], jg, sizes))
        if split["step"] is None:
            p = int(torch.nonzero(rl[r] != gl[r])[0])
            split |= {"gap": float((rs[r, p] - gs[r, p]).abs()), "position": p}
        tie = "order_near_tie" if split["step"] is None else "cutoff_near_tie"
        kind = tie if split["gap"] <= NEAR_TIE else "defect"
        counts[kind] += 1
        rows.append({"row": r, "kind": kind} | split)
    gaps = [x["gap"] for x in rows]
    return {"mismatched_rows": len(bad)} | counts | {"max_gap": max(gaps) if gaps else None, "rows": rows}


def cmd_cutoffs(_args) -> None:
    """fp32: every mismatched row of the equivalence comparisons, traced through the four steps (the
    protocol's note "R5b near-ties at a beam cutoff"). The final beam is traced eagerly (step 4 without
    graphs) and its lists checked against the graphs'."""
    r5.setup()
    t0, git = time.time(), resume.code_state()
    d = scale.load("full")
    with Background(LOGS / "cutoffs_background.log") as bg:
        tiger, _inp, sums = r5.load_tiger(d)
        sizes = tiger.layout.level_sizes
        rows = r5.pool_rows(d, "valid")[:EQUIV_ROWS]
        req = build_requests(d, "valid", rows, tiger.layout.user_offset)
        items, user = req.items, req.user_tok
        ex = example_request(items, user)
        out = {}
        for w in WIDTHS:
            ref, eager = reference(tiger, w, torch.float32), make_beam(tiger, w, "static", torch.float32)
            graphs = make_beam(tiger, w, "graphs", torch.float32, EQUIV_CHUNK, ex)
            res = {}
            for chunk in (EQUIV_CHUNK, 1):
                r = traced_run(ref, items, user, chunk)
                g = traced_run(eager, items, user, chunk)
                same = torch.equal(g[0], run_chunks(graphs, items, user, chunk)[0])
                res[f"chunk{chunk}"] = {"eager_lists_equal_graphs": same} | classify(
                    r, g, (chunk, chunk), sizes
                )
                res[f"chunk{chunk}"]["reference"] = r
            ref1, ref128 = res["chunk1"].pop("reference"), res[f"chunk{EQUIV_CHUNK}"].pop("reference")
            res["reference_chunk128_vs_batch1"] = classify(ref1, ref128, (1, EQUIV_CHUNK), sizes)
            out[str(w)] = res
            del graphs
            torch.cuda.empty_cache()
            keys = ("mismatched_rows", "order_near_tie", "cutoff_near_tie", "defect", "max_gap")
            print(w, {k: {m: v[m] for m in keys} for k, v in res.items()}, flush=True)
    record = {
        "split": "valid",
        "rows": EQUIV_ROWS,
        "sample": "the equivalence sample: the first 2,048 rows of the validation pool",
        "near_tie": NEAR_TIE,
        "beam": "the static step in fp32, run eagerly so its top-k can be traced; eager_lists_equal_graphs "
        "checks that its lists are the graphs'",
        "checkpoint": sums,
        "widths": out,
    } | gpu_record(t0, bg, git)
    common.write_json(RESULTS / "valid" / "equivalence_cutoffs.json", record)


# -------------------------------------------------------------------------------------------- quality (bf16)


def cmd_quality(_args) -> None:
    """The final beam in bf16 over the 100,000-row validation sample in chunks of 128, at widths 200, 50, 20:
    union recall@200 and HSTU-SID's NDCG@10 on the union, against R5's beam."""
    r5.setup()
    t0, git = time.time(), resume.code_state()
    d = scale.load("full")
    r5bw = load_json(r5.RESULTS / "valid" / "beam_widths.json")
    with Background(LOGS / "quality_background.log") as bg:
        tiger, inp, sums = r5.load_tiger(d)
        rows, users, cold, order = r4c.sample_rows(d, "valid", "cuda")
        req = build_requests(d, "valid", order, tiger.layout.user_offset)
        ex = example_request(req.items, req.user_tok)
        lists, refs, seconds = {}, {}, {}
        for w in WIDTHS:
            beam = make_beam(tiger, w, "graphs", SERVE_DTYPE, QUALITY_CHUNK, ex)
            torch.cuda.synchronize()
            t = time.time()
            lists[w] = run_chunks(beam, req.items, req.user_tok, QUALITY_CHUNK)[0].long()
            seconds[w] = round(time.time() - t, 1)
            del beam
            torch.cuda.empty_cache()
            RUNS.mkdir(parents=True, exist_ok=True)
            np.savez(RUNS / f"tiger_valid_w{w}.npz", rows=order, lists=lists[w].int().numpy())
            path = (
                r4c.RUNS / f"tiger_valid_seed{r5.SEED}.npz" if w == 200 else r5.RUNS / f"tiger_valid_w{w}.npz"
            )
            with np.load(path) as f:
                if not np.array_equal(f["rows"], order):
                    raise SystemExit(f"{path} covers other rows")
                refs[w] = torch.as_tensor(f["plain"] if w == 200 else f["lists"]).long()
        del tiger
        torch.cuda.empty_cache()
        ann = r4c.load_lists(r4c.RUNS / "ann_valid.npz", order)
        target = inp.items[rows.end].cpu()
        cold_c = cold.cpu()
        model, rsums, vocab, v = r5.load_hstu_sid(d, with_vocab=True)
        sums |= rsums
        flat = scale.Flat(d, "cuda")
        cfg = {"k_tiger": K_TIGER, "backfill": BACKFILL}
        widths = {}
        for w in WIDTHS:
            u = r4c.union_on(lists[w], ann, cfg, r4c.LIST, "cuda")
            ranks, _ = r4c.rerank(model, flat, rows, u, vocab, v, full=False)
            before = r5bw["widths"][str(w)]
            rec = {
                "tiger_alone_recall@w": cascade.recall_split(cascade.hits(lists[w], target), cold_c),
                "union_recall@200": cascade.recall_split(cascade.hits(u, target), cold_c),
                "hstu_sid": split_metrics(ranks.cpu(), cold_c),
                "beam_seconds_chunk128": seconds[w],
                "lists_identical_to_r5_share": float((lists[w] == refs[w]).all(1).double().mean()),
                "items_shared_with_r5": r5.shared_items(lists[w], refs[w]) / w,
                "lists": str((RUNS / f"tiger_valid_w{w}.npz").relative_to(common.ROOT)),
            }
            r5_recall, r5_ndcg = before["union_recall@200"]["all"], before["hstu_sid"]["all"]["ndcg@10"]
            rec["against_r5"] = {
                "r5_union_recall@200": r5_recall,
                "union_recall@200_change": rec["union_recall@200"]["all"] - r5_recall,
                "union_recall@200_change_relative": (rec["union_recall@200"]["all"] - r5_recall) / r5_recall,
                "r5_hstu_sid_ndcg@10": r5_ndcg,
                "ndcg@10_change": rec["hstu_sid"]["all"]["ndcg@10"] - r5_ndcg,
                "ndcg@10_change_relative": (rec["hstu_sid"]["all"]["ndcg@10"] - r5_ndcg) / r5_ndcg,
                "r5_lists": before["lists"],
            }
            widths[str(w)] = rec
            print(w, rec["against_r5"], flush=True)
    record = {
        "split": "valid",
        "rows": len(order),
        "seed": r5.SEED,
        "dtype": "bf16",
        "chunk": QUALITY_CHUNK,
        "union": cfg | {"budget": r4c.LIST},
        "checkpoint": sums,
        "widths": widths,
    } | gpu_record(t0, bg, git)
    common.write_json(RESULTS / "valid" / "quality.json", record)


# ------------------------------------------------------------------------------------------------- the ladder


def cmd_ladder(args) -> None:
    """The beam alone at each step and width (test pool, bf16): batch-1 latency, and 50 back-to-back batches
    of 64 after 10 untimed."""
    r5.setup()
    d = scale.load("full")
    rows = r5.pool_rows(d, "test")
    offset = r5.user_offset()
    req = build_requests(d, "test", rows, offset)
    n = len(rows)
    for step in args.steps or LADDER:
        t0, git = time.time(), resume.code_state()
        with Background(LOGS / f"ladder_{step}_background.log") as bg:
            tiger, _inp, sums = r5.load_tiger(d)
            items, user = req.items.cuda(), req.user_tok.cuda()
            rec = {
                "step": step,
                "split": "test",
                "pool": n,
                "dtype": "bf16",
                "checkpoint": sums,
                "widths": {},
            }
            if step != "reference":
                rec["options"] = options(step) | {"graphs": step == "graphs"}
            for w in WIDTHS:
                beam = make_beam(tiger, w, step, SERVE_DTYPE, LADDER_BATCH, example_request(items, user))

                def one(i, beam=beam, items=items, user=user):
                    beam(items[i : i + 1], user[i : i + 1])
                    torch.cuda.synchronize()

                b1, b1_raw = loadgen.sequential_ms(one, r5.BATCH1_N, r5.BATCH1_WARM)

                def batch(req_nums, beam=beam, items=items, user=user):
                    idx = torch.from_numpy(req_nums % n).cuda()
                    beam(items[idx], user[idx])
                    torch.cuda.synchronize()

                bt = loadgen.batch_times(batch, LADDER_BATCH, r5.SWEEP_N, r5.SWEEP_WARM)
                rec["widths"][str(w)] = {
                    "batch1": b1
                    | {
                        "raw": save_raw(
                            f"ladder/{step}_w{w}_batch1", {"latency_ms": (b1_raw * 1e3).astype(np.float32)}
                        )
                    },
                    f"batch{LADDER_BATCH}": loadgen.quantiles_ms(bt)
                    | {"batches": len(bt), "requests_per_s": LADDER_BATCH * len(bt) / float(bt.sum())},
                }
                print(
                    step,
                    w,
                    round(b1["p50_ms"], 3),
                    round(LADDER_BATCH * len(bt) / float(bt.sum()), 1),
                    flush=True,
                )
                del beam
                torch.cuda.empty_cache()
            del tiger
        rec |= gpu_record(t0, bg, git)
        common.write_json(RESULTS / "test" / "ladder" / f"{step}.json", rec)


# ---------------------------------------------------------------------------------------------- the cascade


def cmd_select(_args) -> None:
    """B_max of the cascade with the final beam at each width (validation pool, R5's rule)."""
    r5.setup()
    t0, git = time.time(), resume.code_state()
    d = scale.load("full")
    keeps = r5_keeps()
    with Background(LOGS / "select_background.log") as bg:
        tiger, _inp, sums = r5.load_tiger(d)
        model, rsums = r5.load_hstu_sid(d)
        index, vectors, index_rec = r5.load_index()
        rows = r5.pool_rows(d, "valid")
        req = build_requests(d, "valid", rows, tiger.layout.user_offset)
        out = {}
        for w in WIDTHS:
            casc = make_cascade(
                tiger, index, vectors, model, r5.final_ranker(keeps), req.fields(), w, r5.LARGEST, lazy=True
            )
            r5.warm(casc, 8)
            best, sw = loadgen.choose_max_batch(
                lambda s, casc=casc: loadgen.batch_times(casc, s, r5.SWEEP_N, r5.SWEEP_WARM),
                r5.SLO / 3,
                r5.LARGEST,
            )
            out[f"w{w}"] = {"width": w, "max_batch": best, "sweep": sw}
            print(w, best, flush=True)
            del casc
            torch.cuda.empty_cache()
    record = {
        "split": "valid",
        "pool": len(rows),
        "limit_ms": r5.SLO / 3 * 1e3,
        "rule": "largest power of two up to 1024 whose p99 batch time over 50 batches (after 10) is at most "
        "SLO/3",
        "beam": "graphs (bf16); each batch size's CUDA graph is captured during its first warm-up batch",
        "ranker": r5.final_ranker(keeps),
        "checkpoint": sums | rsums,
        "index": index_rec,
        "cascade": out,
    } | gpu_record(t0, bg, git)
    common.write_json(RESULTS / "valid" / "select.json", record)


def cmd_stages(_args) -> None:
    """R5's batch-1 stage latencies with the final beam: the beam at each width, HNSW, union, ranker, and the
    whole cascade at each width."""
    r5.setup()
    t0, git = time.time(), resume.code_state()
    d = scale.load("full")
    keeps = r5_keeps()
    n = r5.BATCH1_N + r5.BATCH1_WARM
    with Background(LOGS / "stages_background.log") as bg:
        tiger, _inp, sums = r5.load_tiger(d)
        model, rsums = r5.load_hstu_sid(d)
        sums |= rsums
        index, vectors, index_rec = r5.load_index()
        _rows, req, fields = r5.test_pool(d, tiger.layout.user_offset, cands=False)
        g_items, g_user = req.items[:n].cuda(), req.user_tok[:n].cuda()
        out, raw, kept = {}, {}, {}
        for w in WIDTHS:
            beam = make_beam(tiger, w, "graphs", SERVE_DTYPE, 1, example_request(g_items, g_user))
            got = []

            def one(i, beam=beam, got=got):
                got.append(beam(g_items[i : i + 1], g_user[i : i + 1])[0].clone())
                torch.cuda.synchronize()

            out[f"beam_w{w}"], raw[f"beam_w{w}"] = loadgen.sequential_ms(one, r5.BATCH1_N, r5.BATCH1_WARM)
            kept[w] = got
            del beam
        got_ann = []

        def hnsw(i):
            got_ann.append(ann_lists(index, vectors, req.items[i : i + 1]))

        out["hnsw"], raw["hnsw"] = loadgen.sequential_ms(hnsw, r5.BATCH1_N, r5.BATCH1_WARM)
        ann_g = torch.as_tensor(np.concatenate(got_ann)).long().cuda()
        got_u = []

        def union(i):
            got_u.append(cascade.union(kept[200][i], ann_g[i : i + 1], K_TIGER, CANDS, BACKFILL))
            torch.cuda.synchronize()

        out["union"], raw["union"] = loadgen.sequential_ms(union, r5.BATCH1_N, r5.BATCH1_WARM)
        sub = {k: v[:n] for k, v in fields.items()} | {"cands": torch.cat(got_u).cpu()}
        server = r5.make_ranker(model, r5.final_ranker(keeps), sub, 1, n)

        def rank(i, server=server):
            server(np.array([i]))

        out["ranker"], raw["ranker"] = loadgen.sequential_ms(rank, r5.BATCH1_N, r5.BATCH1_WARM)
        del server
        for w in WIDTHS:
            casc = make_cascade(tiger, index, vectors, model, r5.final_ranker(keeps), fields, w, 1)
            times = []

            def e2e(i, casc=casc, times=times):
                times.append(casc(np.array([i])))

            out[f"cascade_w{w}"], raw[f"cascade_w{w}"] = loadgen.sequential_ms(
                e2e, r5.BATCH1_N, r5.BATCH1_WARM
            )
            parts = {k: np.array([t[k] for t in times[r5.BATCH1_WARM :]]) for k in Cascade.STAGES}
            out[f"cascade_w{w}"]["stages_ms"] = {
                k: {"p50": float(np.median(v)), "p99": float(np.quantile(v, 0.99))} for k, v in parts.items()
            }
            del casc
            torch.cuda.empty_cache()
    paths = {k: save_raw(f"stages/{k}", {"latency_ms": (v * 1e3).astype(np.float32)}) for k, v in raw.items()}
    record = {
        "split": "test",
        "requests": n,
        "beam": "graphs (bf16)",
        "ranker": r5.final_ranker(keeps),
        "checkpoint": sums,
        "index": index_rec,
        "batch1": out,
        "raw": paths,
    } | gpu_record(t0, bg, git)
    common.write_json(RESULTS / "test" / "stages_batch1.json", record)
    print({k: round(v["p50_ms"], 3) for k, v in out.items()})


def measure(server, max_batch: int, tag: str) -> dict:
    """R5's measurement of a configuration under load (``r5_serving.measure`` without batch 1), with the raw
    arrays under data/runs/r5b/raw/."""
    r5.warm(server, max_batch)
    out: dict = {"max_batch": max_batch, "window_s": r5.WINDOW if max_batch > 1 else 0.0}
    out["saturation"] = loadgen.saturation(server, max_batch, r5.SAT_S)
    window = r5.WINDOW if max_batch > 1 else 0.0

    def run(rate: float, seconds: float, seed: int, name: str) -> dict:
        s, raw = loadgen.open_loop_run(server, rate, max_batch, window, r5.SLO, r5.WARM_S, seconds, seed)
        stages = r5.stage_summary(raw)
        print(
            f"{tag} {name} rate {rate:.1f} p99 {s.get('p99_ms', float('nan')):.2f} pass {s['pass']}",
            flush=True,
        )
        return s | ({"stages": stages} if stages else {}) | {"raw": save_raw(f"{tag}/{name}", raw)}

    lo, bis = loadgen.bisect_rate(
        lambda rate, i: run(rate, r5.BISECT_S, 1000 + i, f"bisect{i}"),
        1.05 * out["saturation"]["qps"],
        r5.BISECT_STEPS,
    )
    out["bisection"] = bis
    qps, conf = None, []
    if lo > 0:
        qps, conf = loadgen.confirm_rate(
            lambda rate, a, j: run(rate, r5.CONFIRM_S, 2000 + 10 * a + j, f"confirm{a}_{j}"),
            lo,
            r5.REPS,
            r5.LOWER,
            r5.TRIES,
        )
    out["confirmation"] = conf
    out["qps_at_slo"] = qps
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


def cmd_e2e(args) -> None:
    r5.setup()
    t0, git = time.time(), resume.code_state()
    d = scale.load("full")
    keeps = r5_keeps()
    w = args.width
    batch = load_json(RESULTS / "valid" / "select.json")["cascade"][f"w{w}"]["max_batch"]
    with Background(LOGS / f"e2e_w{w}_background.log") as bg:
        tiger, _inp, sums = r5.load_tiger(d)
        model, rsums = r5.load_hstu_sid(d)
        sums |= rsums
        index, vectors, index_rec = r5.load_index()
        _rows, _req, fields = r5.test_pool(d, tiger.layout.user_offset, cands=False)
        casc = make_cascade(tiger, index, vectors, model, r5.final_ranker(keeps), fields, w, batch)
        rec = {
            "width": w,
            "split": "test",
            "pool": r5.POOL,
            "beam": "graphs (bf16)",
            "ranker": r5.final_ranker(keeps),
            "checkpoint": sums,
            "index": index_rec,
        }
        rec |= measure(casc, batch, f"e2e/w{w}")
    rec |= gpu_record(t0, bg, git)
    common.write_json(RESULTS / "test" / f"e2e_w{w}.json", rec)
    print(w, rec["qps_at_slo"], rec.get("at_slo"))


# ------------------------------------------------------------------------------------------------ projection


def cmd_projection(_args) -> None:
    """Times a few calls of each kind of work, then projects each command's GPU time."""
    r5.setup()
    t0, git = time.time(), resume.code_state()
    with Background(LOGS / "projection_background.log") as bg:
        t = time.time()
        d = scale.load("full")
        tiger, _inp, _sums = r5.load_tiger(d)
        model, _ = r5.load_hstu_sid(d)
        index, vectors, _ = r5.load_index()
        load_s = time.time() - t
        rows = r5.pool_rows(d, "test")[:400]
        req = build_requests(d, "test", rows, tiger.layout.user_offset)
        items, user = req.items.cuda(), req.user_tok.cuda()
        ex = example_request(items, user)
        ms = defaultdict(dict)
        for step in LADDER:
            for w in WIDTHS:
                t = time.time()
                beam = make_beam(tiger, w, step, SERVE_DTYPE, LADDER_BATCH, ex)
                build_s = time.time() - t

                def one(i, beam=beam):
                    beam(items[i : i + 1], user[i : i + 1])
                    torch.cuda.synchronize()

                def b64(i, beam=beam):
                    beam(items[:LADDER_BATCH], user[:LADDER_BATCH])
                    torch.cuda.synchronize()

                ms[step][w] = {
                    "batch1": loadgen.sequential_ms(one, 10, 3)[0]["mean_ms"],
                    "batch64": loadgen.sequential_ms(b64, 3, 1)[0]["mean_ms"],
                    "build_s": build_s,
                }
                del beam
                torch.cuda.empty_cache()
        fields = req.fields()
        casc = make_cascade(tiger, index, vectors, model, r5.final_ranker(r5_keeps()), fields, 200, 1)
        e2e_ms = loadgen.sequential_ms(lambda i: casc(np.array([i])), 50, 10)[0]["mean_ms"]
    run_s = r5.SAT_S + r5.BISECT_STEPS * (r5.WARM_S + r5.BISECT_S) + r5.REPS * (r5.WARM_S + r5.CONFIRM_S)
    retry_s = r5.TRIES * r5.REPS * (r5.WARM_S + r5.CONFIRM_S)
    b1_s = lambda ms_: (r5.BATCH1_N + r5.BATCH1_WARM) * ms_ / 1e3  # noqa: E731
    ref, fin = ms["reference"], ms["graphs"]
    eager_steps = [s for s in LADDER if s != "graphs"]
    plan = {
        "profile": load_s
        + sum(30 * (ref[w]["batch1"] + ref[w]["batch64"] + fin[w]["batch64"]) / 1e3 for w in WIDTHS)
        + 120,
        "equivalence": load_s
        + sum(
            2 * EQUIV_ROWS * ref[w]["batch1"] / 1e3
            + 2 * EQUIV_ROWS * fin[w]["batch1"] / 1e3
            + 5 * EQUIV_ROWS / LADDER_BATCH * ref[w]["batch64"] * 2 / 1e3
            for w in WIDTHS
        )
        + 120,
        "quality": load_s
        + sum(r4c.SAMPLE / LADDER_BATCH * fin[w]["batch64"] * 2 / 1e3 for w in WIDTHS)
        + 240,
        "ladder": len(LADDER) * load_s
        + sum(
            b1_s(ms[s][w]["batch1"]) + 60 * ms[s][w]["batch64"] / 1e3 + ms[s][w]["build_s"]
            for s in LADDER
            for w in WIDTHS
        ),
        "select": load_s + len(WIDTHS) * (11 * (r5.SWEEP_N + r5.SWEEP_WARM) * r5.SLO / 3 + 60),
        "stages": load_s
        + sum(b1_s(fin[w]["batch1"]) for w in WIDTHS)
        + b1_s(5.0)
        + len(WIDTHS) * b1_s(e2e_ms)
        + 120,
        "e2e": len(WIDTHS) * (load_s + run_s + 120),
    }
    expected = sum(plan.values()) / 3600 + (time.time() - t0) / 3600
    worst = expected + len(WIDTHS) * retry_s / 3600
    record = {
        "measured": {
            "load_s": round(load_s, 1),
            "beam_ms": ms,
            "cascade_w200_batch1_ms": e2e_ms,
            "eager_steps": eager_steps,
        },
        "plan_seconds": {k: round(v) for k, v in plan.items()},
        "projected_gpu_hours": round(expected, 2),
        "worst_case_gpu_hours": round(worst, 2),
        "note": "select assumes every size of the sweep runs at the SLO/3 limit; worst case: every e2e "
        "confirmation lowered 4 times",
        "stop_at": STOP_GPU_HOURS,
        "stop": worst > STOP_GPU_HOURS,
    } | gpu_record(t0, bg, git)
    common.write_json(RESULTS / "projection.json", record)
    print(record["projected_gpu_hours"], record["worst_case_gpu_hours"], json.dumps(ms))


COMMANDS = {
    "projection": cmd_projection,
    "profile": cmd_profile,
    "equivalence": cmd_equivalence,
    "cutoffs": cmd_cutoffs,
    "quality": cmd_quality,
    "ladder": cmd_ladder,
    "select": cmd_select,
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
