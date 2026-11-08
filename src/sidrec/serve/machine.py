"""The background load on the machine during an R5 command.

A sidecar shell process, outside the benchmark's Python process, writes ``/proc/loadavg`` every 2 s and the
GPU's state and compute processes from nvidia-smi every 10 s. The busiest processes (``top``, over half a
second) are listed when the command starts and when it ends.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import numpy as np

GPU_FIELDS = "clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,pstate"
SIDECAR = f"""
i=0
while true; do
  echo "L $(date +%s.%N) $(cat /proc/loadavg)"
  if [ $((i % 5)) -eq 0 ]; then
    echo "G $(date +%s.%N) $(nvidia-smi --query-gpu={GPU_FIELDS} --format=csv,noheader,nounits | head -1)"
    echo "A $(date +%s.%N) $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\\n' ';')"
  fi
  i=$((i + 1))
  sleep 2
done
"""


def top_processes(n: int = 10) -> list[dict]:
    """The ``n`` processes using the most CPU over half a second, from top's second sample."""
    try:
        out = subprocess.run(
            ["top", "-b", "-n", "2", "-d", "0.5", "-o", "%CPU", "-w", "200"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return [{"error": str(e)}]
    blocks = [b for b in out.split("\n\n") if b.strip()]
    lines = blocks[-1].splitlines() if blocks else []
    rows = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 12 or not parts[0].isdigit():
            continue
        rows.append({"pid": int(parts[0]), "cpu_pct": float(parts[8]), "mem_pct": float(parts[9]),
                     "command": " ".join(parts[11:])})
    return rows[:n]


def parse(lines: list[str], own_pid: int) -> dict:
    """Summary of the sidecar's lines: load averages, GPU fields and the other GPU compute processes."""
    load, gpu, apps = [], [], set()
    for ln in lines:
        parts = ln.split(maxsplit=2)
        if len(parts) < 3:
            continue
        kind, rest = parts[0], parts[2]
        if kind == "L":
            v = rest.split()
            load.append((float(v[0]), float(v[1]), float(v[2]), v[3]))
        elif kind == "G":
            v = [x.strip() for x in rest.split(",")]
            if len(v) == len(GPU_FIELDS.split(",")):
                gpu.append(v)
        elif kind == "A":
            for app in rest.split(";"):
                f = [x.strip() for x in app.split(",")]
                if len(f) == 3 and f[0].isdigit() and int(f[0]) != own_pid:
                    apps.add(f"{f[1]} ({f[2]})")
    out: dict = {"load_samples": len(load), "gpu_samples": len(gpu), "other_gpu_processes": sorted(apps)}
    if load:
        a = np.array([x[:3] for x in load])
        for i, name in enumerate(("load1", "load5", "load15")):
            out[name] = {"min": float(a[:, i].min()), "median": float(np.median(a[:, i])), "max": float(a[:, i].max())}
        out["running_tasks"] = {"max": max(int(x[3].split("/")[0]) for x in load)}
    names = GPU_FIELDS.split(",")
    for i, name in enumerate(names):
        if name == "pstate":
            out[name] = sorted({g[i] for g in gpu})
            continue
        vals = []
        for g in gpu:
            try:
                vals.append(float(g[i]))
            except ValueError:
                pass
        if vals:
            v = np.array(vals)
            out[name] = {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}
    return out


class Background:
    """Context manager: runs the sidecar for the duration of a command, writing its lines to ``path``."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> Background:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.top_start = top_processes()
        self._file = open(self.path, "w")  # noqa: SIM115
        self._proc = subprocess.Popen(
            ["bash", "-c", SIDECAR], stdout=self._file, stderr=subprocess.DEVNULL, start_new_session=True
        )
        return self

    def __exit__(self, *exc) -> None:
        os.killpg(self._proc.pid, signal.SIGTERM)
        self._proc.wait()
        self._file.close()
        self.top_end = top_processes()

    def summary(self) -> dict:
        lines = self.path.read_text().splitlines()
        return parse(lines, os.getpid()) | {
            "cpus": os.cpu_count(),
            "top_at_start": self.top_start,
            "top_at_end": self.top_end,
            "sidecar_log": str(self.path),
        }
