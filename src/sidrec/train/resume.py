"""The epoch loop with early stopping, resumable across processes (R4b).

A full-split run takes hours, and other jobs queue on the GPU lock. So a process trains while it has time left
in its budget (about an hour), saves the training state after every epoch and exits with code ``PAUSED`` when
one more epoch would pass the budget; the next process, under a fresh lock, resumes from the saved state.

The state holds the model, the optimizer and the schedule, every random generator the loop draws from (the
batch-order generator, and the CPU and CUDA default generators that dropout uses) and the early-stopping
bookkeeping. A resumed run therefore draws the same batches and dropout masks as one uninterrupted process.

Early stopping is the protocol's: an evaluation after every epoch, the best validation NDCG@10 kept, a stop
after ``patience`` epochs without improvement or at ``max_epochs``, and the best epoch's weights restored at
the end.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch

from sidrec.train import common

PAUSED = 75  # exit code of a process that saved its state and stopped before the run finished


def sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def new_state() -> dict:
    return {
        "epoch": 0,
        "best": -1.0,
        "best_epoch": 0,
        "best_state": None,
        "best_valid": None,
        "history": [],
        "done": False,
    }


def save(path: Path, model, opt, sched, gen: torch.Generator, st: dict) -> None:
    blob = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "gen": gen.get_state(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if gen.device.type == "cuda" else None,
        "st": st,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load(path: Path, model, opt, sched, gen: torch.Generator) -> dict:
    """Restore a saved state into the given objects; a fresh state if ``path`` is None or missing."""
    if path is None or not path.exists():
        return new_state()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    opt.load_state_dict(blob["opt"])
    sched.load_state_dict(blob["sched"])
    gen.set_state(blob["gen"])
    torch.set_rng_state(blob["cpu_rng"])
    if blob["cuda_rng"] is not None:
        torch.cuda.set_rng_state(blob["cuda_rng"])
    return blob["st"]


def run_epochs(
    model: torch.nn.Module,
    opt,
    sched,
    gen: torch.Generator,
    device,
    train_epoch: Callable[[], float],
    evaluate: Callable[[], dict],
    row: Callable[[dict], dict],
    max_epochs: int,
    patience: int,
    state_path: Path | None = None,
    budget_s: float | None = None,
) -> tuple[dict, bool]:
    """Train until early stopping ends the run or the time budget pauses it. Returns (state, paused).

    train_epoch runs one epoch and returns its mean loss; evaluate returns validation metrics with
    ["all"]["ndcg@10"]; row picks the history fields from them. When the run ends, the best epoch's weights
    are loaded into ``model``."""
    st = load(state_path, model, opt, sched, gen)
    st["first_epoch"] = st["epoch"] + 1  # this process's first epoch
    start = time.time()
    while not st["done"]:
        epoch = st["epoch"] + 1
        model.train()
        sync(device)
        t0 = time.time()
        loss = train_epoch()
        sync(device)
        train_s = time.time() - t0
        metrics = evaluate()
        eval_s = time.time() - t0 - train_s
        st["history"].append(
            {
                "epoch": epoch,
                "loss": round(loss, 5),
                **row(metrics),
                "train_s": round(train_s, 1),
                "eval_s": round(eval_s, 1),
            }
        )
        print(st["history"][-1], flush=True)
        if metrics["all"]["ndcg@10"] > st["best"]:
            st["best"], st["best_epoch"], st["best_valid"] = metrics["all"]["ndcg@10"], epoch, metrics
            st["best_state"] = copy.deepcopy(model.state_dict())
        st["epoch"] = epoch
        st["done"] = epoch >= max_epochs or epoch - st["best_epoch"] >= patience
        paused = not st["done"] and budget_s is not None and time.time() - start + train_s + eval_s > budget_s
        if state_path is not None:
            save(state_path, model, opt, sched, gen, st)
        if paused:
            return st, True
    model.load_state_dict(st["best_state"])
    return st, False


def code_state() -> dict:
    """The commit a process started from, the tree hash of src/ (equal hashes mean the same code) and whether
    src/ had uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=common.ROOT, capture_output=True, text=True).stdout.strip()

    return common.git_state() | {"src_tree": git("rev-parse", "HEAD:src")}


def processes_path(state_path: Path) -> Path:
    return state_path.with_name(state_path.name.replace(".state.pt", ".processes.json"))


def read_processes(state_path: Path) -> list[dict]:
    """The entries of the earlier processes of a run that is being resumed (none for a fresh run)."""
    path = processes_path(state_path)
    return json.loads(path.read_text()) if state_path.exists() and path.exists() else []


def write_processes(state_path: Path, entries: list[dict]) -> None:
    common.write_json(processes_path(state_path), entries)


def process_entry(st: dict, t0: float, smi_start: dict, sampler, git: dict) -> dict:
    """What one process did: its epochs, wall time, code and GPU state."""
    return {
        "epochs": [st["first_epoch"], st["epoch"]],
        "wall_time_s": round(time.time() - t0, 1),
        "git": git,
        "nvidia_smi_start": smi_start,
        "nvidia_smi_end": common.nvidia_smi(),
        "sampled": sampler.summary(),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available()
        else None,
    }
