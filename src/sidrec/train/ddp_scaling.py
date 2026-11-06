"""DDP scaling of the HSTU-style model on the full Electronics train split: 1 or 2 GPUs, one global batch.

    python -m sidrec.train.ddp_scaling --trial 1 --global-batch 128 --out one.json
    torchrun --standalone --nproc_per_node=2 -m sidrec.train.ddp_scaling --trial 1 --out two.json

Every run draws the same sequence of global batches: a seed-0 permutation of the full split's 1,576,803
training windows. With W processes, rank r takes rows r, r + W, ... of each global batch. Each rank's loss is
its summed cross-entropy divided by the number of targets in the whole global batch, times W. DDP averages the
gradients over ranks, so an update equals the 1-GPU update on the same global batch, up to floating point and
the dropout draws. The model, loss, optimizer and warm-up are those of ``sidrec.train.scale_idrec``.

Timing: ``--warmup`` untimed steps, then ``--steps`` timed steps, each ended by a CUDA synchronize. Rank 0
samples nvidia-smi for every GPU about once a second during the timed steps. With 2 processes the run also
times NCCL all-reduces of the model's gradient size. Only numpy and torch are needed, so the module runs in
the Snowflake node's offline environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from sidrec.data import scale
from sidrec.train import common
from sidrec.train.fused_ce import fused_linear_cross_entropy
from sidrec.train.scale_idrec import BATCH, MAX_EPOCHS, MAX_LEN, WARMUP_SHARE, build, output_table
from sidrec.tuning import shared_trials

ARRAYS = ("n_items", "offsets", "items", "ts", "train_len", "train_item")


class Step(nn.Module):
    """One training step's loss as a module call, so that DDP sees the forward pass."""

    def __init__(self, model: nn.Module, v: int):
        super().__init__()
        self.model, self.v = model, v

    def forward(self, b: dict[str, torch.Tensor], scale_by: torch.Tensor) -> torch.Tensor:
        states = self.model.encode(b["inputs"], input_ts=b["input_ts"], query_ts=b["query_ts"])
        keep = b["targets"] > 0
        loss = fused_linear_cross_entropy(
            states[keep], output_table(self.model, self.v), b["targets"][keep] - 1
        )
        return loss * scale_by


class GpuSampler:
    """nvidia-smi utilization, SM clock, power and memory of every GPU, sampled in a background thread."""

    FIELDS = "index,utilization.gpu,clocks.sm,power.draw,memory.used,temperature.gpu"

    def __init__(self, interval: float = 1.0):
        self.interval, self.samples, self._stop = interval, [], threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                for line in out.strip().splitlines():
                    values = [v.strip() for v in line.split(",")]
                    self.samples.append(
                        {k: _num(v) for k, v in zip(self.FIELDS.split(","), values, strict=False)}
                    )
            except (OSError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> GpuSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        out = {}
        for gpu in sorted({s.get("index") for s in self.samples if s.get("index") is not None}):
            rows = [s for s in self.samples if s.get("index") == gpu]
            out[f"gpu{int(gpu)}"] = {"samples": len(rows)} | {
                k: float(np.median([r[k] for r in rows if isinstance(r.get(k), float)]))
                for k in self.FIELDS.split(",")[1:]
                if any(isinstance(r.get(k), float) for r in rows)
            }
        return out


def _num(v: str):
    try:
        return float(v)
    except ValueError:
        return v


def allreduce_bench(numel: int, device, reps: int = 10) -> dict:
    """Time of an NCCL all-reduce of ``numel`` float32 values, and its bus bandwidth (as in nccl-tests)."""
    x = torch.ones(numel, device=device)
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    sec = (time.perf_counter() - t) / reps
    w = dist.get_world_size()
    size = numel * 4
    return {
        "mb": round(size / 2**20, 1),
        "ms": round(sec * 1e3, 2),
        "algbw_gb_s": round(size / sec / 1e9, 2),
        "busbw_gb_s": round(size / sec / 1e9 * 2 * (w - 1) / w, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--global-batch", type=int, default=BATCH["hstu"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--data", type=Path, default=scale.cache_path("full"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.global_batch % world:
        raise SystemExit("the global batch must divide by the number of processes")
    t_load = time.time()
    with np.load(args.data) as f:
        d = {k: f[k] for k in ARRAYS}
    flat = scale.Flat(d, device)
    vocab_np, v = scale.id_vocab(d)
    vocab = torch.as_tensor(vocab_np, device=device)
    ex = scale.id_train_examples(d, device)
    targets_per_window = torch.minimum(ex.end - ex.start, torch.full_like(ex.end, MAX_LEN))
    load_s = time.time() - t_load
    cfg = shared_trials()[args.trial - 1]
    common.seed_all(args.seed)  # the same initial weights on every rank; DDP also broadcasts rank 0's
    model = build("hstu", cfg, v).to(device)
    step = Step(model, v)
    if distributed:
        step = nn.parallel.DistributedDataParallel(step, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(ex) / args.global_batch)
    warmup = max(1, int(WARMUP_SHARE * MAX_EPOCHS * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, common.warmup_schedule(warmup, "constant"))
    order = torch.randperm(len(ex), generator=torch.Generator().manual_seed(args.seed)).to(device)
    times, losses, sampler = [], [], GpuSampler() if rank == 0 else None
    total = args.warmup + args.steps
    for i in range(total):
        if i == args.warmup and sampler is not None:
            sampler.__enter__()
        torch.cuda.synchronize()
        t = time.perf_counter()
        rows = order[i * args.global_batch : (i + 1) * args.global_batch]
        n_global = targets_per_window[rows].sum()
        b = flat.id_inputs(ex.take(rows[rank::world]), vocab, MAX_LEN, train=True)
        loss = step(b, world / n_global)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        torch.cuda.synchronize()
        if i >= args.warmup:
            times.append(time.perf_counter() - t)
            losses.append(loss.detach() / world)
    if sampler is not None:
        sampler.__exit__()
    numel = sum(p.numel() for p in model.parameters())
    record = {
        "model": "hstu",
        "trial": args.trial,
        "config": cfg,
        "processes": world,
        "global_batch": args.global_batch,
        "per_gpu_batch": args.global_batch // world,
        "warmup_steps_untimed": args.warmup,
        "timed_steps": args.steps,
        "params": numel,
        "gradient_mb": round(numel * 4 / 2**20, 1),
        "load_s": round(load_s, 1),
        "rank0_step_ms": {
            "mean": round(1e3 * float(np.mean(times)), 2),
            "median": round(1e3 * float(np.median(times)), 2),
            "p90": round(1e3 * float(np.percentile(times, 90)), 2),
        },
        "windows_per_s": round(args.global_batch / float(np.mean(times)), 1),
        "mean_loss_timed": float(torch.stack(losses).sum().item() / len(losses)),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
        "gpus": sampler.summary() if sampler is not None else None,
        "environment": common.environment() | {"nccl": ".".join(map(str, torch.cuda.nccl.version()))},
        "source_commit": os.environ.get("SIDREC_COMMIT"),
    }
    if distributed:
        loss_sum = torch.stack(losses).sum()
        dist.all_reduce(
            loss_sum
        )  # the global mean loss: each rank's share was scaled by its share of targets
        record["mean_loss_timed"] = float(loss_sum.item() / len(losses))
        step_mean = torch.tensor([float(np.mean(times))], device=device)
        dist.all_reduce(step_mean, op=dist.ReduceOp.MAX)
        record["slowest_rank_step_ms"] = round(1e3 * float(step_mean.item()), 2)
        record["allreduce"] = allreduce_bench(numel, device)
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(
            json.dumps(
                {k: record[k] for k in ("processes", "global_batch", "rank0_step_ms", "windows_per_s")}
            )
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
