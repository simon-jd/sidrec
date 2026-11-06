"""Shared pieces of the training scripts: seeding, schedules, run records."""

from __future__ import annotations

import json
import os
import random
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "r2"
CHECKPOINTS = ROOT / "data" / "runs" / "r2"
PROTOCOL_VERSION = "2026-11-05 with the R2 clarifications"

SMI_FIELDS = (
    "name,driver_version,pstate,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,power.limit,"
    "utilization.gpu,memory.used,memory.total"
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nvidia_smi() -> dict:
    """GPU state at this moment, as nvidia-smi reports it, with the other compute processes on the GPU."""

    def query(args: list[str]) -> str:
        try:
            return subprocess.run(
                ["nvidia-smi", *args], capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"unavailable: {e}"

    state = query([f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader"])
    apps = query(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "fields": SMI_FIELDS, "state": state, "apps": apps}


SAMPLED = (
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "memory.used",
)


def smi_numbers(state: str) -> dict:
    """The sampled fields of one nvidia-smi state line; numbers without units, pstate as text."""
    values = dict(zip(SMI_FIELDS.split(","), (v.strip() for v in state.split(",")), strict=False))
    out = {}
    for field in SAMPLED:
        raw = values.get(field, "")
        if field == "pstate":
            out[field] = raw
            continue
        try:
            out[field] = float(raw.split()[0])
        except (ValueError, IndexError):
            continue
    return out


class SmiSampler:
    """Samples the nvidia-smi state every ``interval`` seconds in a background thread while a run goes on.

    Clocks cannot be locked on this machine, so a run records what they were while it ran.
    """

    def __init__(self, interval: float = 10.0):
        self.interval, self.samples = interval, []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(smi_numbers(nvidia_smi()["state"]))
            self._stop.wait(self.interval)

    def __enter__(self) -> SmiSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        out: dict = {"samples": len(self.samples), "interval_s": self.interval}
        out["pstates"] = sorted({s["pstate"] for s in self.samples if s.get("pstate")})
        for field in SAMPLED[1:]:
            v = np.array([s[field] for s in self.samples if field in s])
            if len(v):
                out[field] = {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}
        return out


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    dirty = run("status", "--porcelain", "--untracked-files=no", "--", "src")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(dirty)}


def warmup_schedule(warmup: int, decay: str):
    """LR multiplier: linear warm-up over ``warmup`` steps, then constant or inverse-square-root decay."""

    def factor(step: int) -> float:
        s = step + 1
        if s <= warmup:
            return s / warmup
        return 1.0 if decay == "constant" else (warmup / s) ** 0.5

    return factor


def environment() -> dict:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "host": os.uname().nodename,
    }


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
