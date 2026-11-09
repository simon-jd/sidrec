"""Open-loop load for an in-process server (R5).

Arrival times are drawn in advance (Poisson: exponential gaps) and never wait for the server. One thread
serves: when it is free and requests wait, it takes the oldest ones as a batch, either at once (``max_batch``
1) or once ``max_batch`` wait or the oldest has waited ``window``. A request's latency runs from its scheduled
arrival to the end of its batch, so the time it spent queued is counted and a slow server cannot slow the
arrivals (no coordinated omission).

``step(requests)`` serves request numbers ``requests`` (an int64 array) and returns once their responses are
in host memory; it may return a dict of per-batch numbers, which are kept.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

QUANTILES = (0.5, 0.9, 0.99, 0.999)


def poisson_arrivals(rate: float, seconds: float, seed: int) -> np.ndarray:
    """Arrival times in [0, seconds) of a Poisson process of ``rate`` per second (numpy PCG64, ``seed``)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    mean = rate * seconds
    gaps = rng.exponential(1.0 / rate, int(mean + 10 * math.sqrt(mean) + 100))
    t = np.cumsum(gaps)
    while t[-1] < seconds:
        t = np.concatenate([t, t[-1] + np.cumsum(rng.exponential(1.0 / rate, len(gaps)))])
    return t[t < seconds]


def wait_until(t: float, clock: Callable[[], float] = time.perf_counter) -> None:
    """Sleep, then spin, until ``clock()`` reaches ``t``."""
    while True:
        left = t - clock()
        if left <= 0:
            return
        if left > 0.002:
            time.sleep(left - 0.001)


def serve(
    step: Callable,
    arrivals: np.ndarray,
    max_batch: int,
    window: float,
    abort_age: float = 1.0,
    clock: Callable[[], float] = time.perf_counter,
    wait: Callable[[float], None] | None = None,
) -> dict:
    """Serves requests 0..n-1 at ``arrivals`` (seconds from the start, ascending).

    Returns per request ``latency`` and ``start`` (when its batch began), in seconds (nan if never served),
    and per batch ``size``, ``begin``, ``end`` and ``info``. ``aborted`` is True if a waiting request grew
    older than ``abort_age``; the rest are then not served."""
    wait = wait or (lambda t: wait_until(t, clock))
    n = len(arrivals)
    latency, started = np.full(n, np.nan), np.full(n, np.nan)
    sizes, begins, ends, infos = [], [], [], []
    head, aborted = 0, False
    t0 = clock()
    while head < n:
        now = clock() - t0
        arrived = int(np.searchsorted(arrivals, now, side="right"))
        if arrived <= head:  # nothing waiting: idle until the next arrival
            wait(t0 + arrivals[head])
            continue
        if now - arrivals[head] > abort_age:
            aborted = True
            break
        waiting = arrived - head
        if waiting < max_batch:
            deadline = arrivals[head] + window
            if now < deadline:
                nxt = arrivals[arrived] if arrived < n else math.inf
                wait(t0 + min(deadline, nxt))
                continue
        end = head + min(waiting, max_batch)
        begin = clock() - t0
        info = step(np.arange(head, end, dtype=np.int64))
        done = clock() - t0
        latency[head:end] = done - arrivals[head:end]
        started[head:end] = begin
        sizes.append(end - head)
        begins.append(begin)
        ends.append(done)
        infos.append(info)
        head = end
    return {
        "latency": latency,
        "start": started,
        "size": np.array(sizes, dtype=np.int64),
        "begin": np.array(begins),
        "end": np.array(ends),
        "info": infos,
        "aborted": aborted,
    }


def quantiles_ms(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {}
    q = np.quantile(x, QUANTILES) * 1e3
    return {
        "p50_ms": float(q[0]),
        "p90_ms": float(q[1]),
        "p99_ms": float(q[2]),
        "p999_ms": float(q[3]),
        "mean_ms": float(x.mean() * 1e3),
        "max_ms": float(x.max() * 1e3),
    }


def histogram_ms(x: np.ndarray, lo: float = 0.01, hi: float = 10_000.0, bins: int = 60) -> dict:
    """Counts over log-spaced edges in milliseconds (below ``lo`` and above ``hi`` go to the end bins)."""
    edges = np.geomspace(lo, hi, bins + 1)
    counts, _ = np.histogram(np.clip(np.asarray(x) * 1e3, lo, hi), bins=edges)
    return {"edges_ms": [round(float(e), 5) for e in edges], "counts": counts.tolist()}


def open_loop_run(
    step: Callable,
    rate: float,
    max_batch: int,
    window: float,
    slo: float,
    warm_s: float,
    measure_s: float,
    seed: int,
    min_n: int = 3000,
    max_n: int = 1_000_000,
    abort_age: float = 1.0,
) -> tuple[dict, dict]:
    """One open-loop run: ``warm_s`` seconds of arrivals, not recorded, then the measured ones (``measure_s``
    seconds, stretched or cut to between ``min_n`` and ``max_n`` requests). Returns its summary and the raw
    arrays of the measured requests."""
    measure_s = min(max(measure_s, min_n / rate), max_n / rate)
    arrivals = poisson_arrivals(rate, warm_s + measure_s, seed)
    res = serve(step, arrivals, max_batch, window, abort_age)
    m = arrivals >= warm_s
    lat, wait = res["latency"][m], (res["start"] - arrivals)[m]
    served = np.isfinite(lat)
    in_window = (res["end"] > warm_s) if len(res["end"]) else np.zeros(0, dtype=bool)
    summary = {
        "rate": rate,
        "seed": seed,
        "measured_s": measure_s,
        "measured": int(m.sum()),
        "served": int(served.sum()),
        "aborted": bool(res["aborted"]),
        "max_batch": max_batch,
        "window_s": window,
    }
    ok = not res["aborted"] and bool(served.all())
    if ok:
        span = res["end"][-1] - arrivals[m][0]
        summary |= quantiles_ms(lat) | {
            "wait": quantiles_ms(wait),
            "throughput": float(m.sum() / span) if span > 0 else None,
            "batches": int(in_window.sum()),
            "mean_batch": float(res["size"][in_window].mean()) if in_window.any() else None,
            "busy_share": float((res["end"] - res["begin"])[in_window].sum() / (res["end"][-1] - warm_s)),
            "histogram": histogram_ms(lat),
        }
    summary["pass"] = ok and summary["p99_ms"] <= slo * 1e3
    raw = {
        "latency_ms": (lat * 1e3).astype(np.float32),
        "wait_ms": (wait * 1e3).astype(np.float32),
        "batch_size": res["size"][in_window],
        "service_ms": ((res["end"] - res["begin"])[in_window] * 1e3).astype(np.float32),
    }
    infos = [i for i, keep in zip(res["info"], in_window, strict=True) if keep and i]
    if infos:
        for k in infos[0]:
            raw[k] = np.array([i[k] for i in infos], dtype=np.float32)
    return summary, raw


def bisect_rate(run: Callable[[float, int], dict], hi: float, steps: int) -> tuple[float, list[dict]]:
    """The largest passing rate found by ``steps`` halvings of [0, hi]; 0 if none passed.

    ``run(rate, step_index)`` returns a summary with a ``pass`` flag."""
    lo, history = 0.0, []
    for i in range(steps):
        mid = (lo + hi) / 2
        r = run(mid, i)
        history.append(r)
        if r["pass"]:
            lo = mid
        else:
            hi = mid
    return lo, history


def confirm_rate(
    run: Callable[[float, int, int], dict], rate: float, reps: int, lower: float, tries: int
) -> tuple[float | None, list[dict]]:
    """``reps`` runs at ``rate``; the rate stands if all pass, otherwise it is multiplied by ``lower`` and the
    runs repeated, at most ``tries`` times. ``run(rate, attempt, rep)`` returns a summary. Returns the
    confirmed rate (None if none) and every attempt."""
    attempts = []
    for attempt in range(tries + 1):
        runs = [run(rate, attempt, j) for j in range(reps)]
        attempts.append({"rate": rate, "runs": runs})
        if all(r["pass"] for r in runs):
            return rate, attempts
        rate *= lower
    return None, attempts


def batch_times(step: Callable, size: int, n: int, warm: int, start: int = 0) -> np.ndarray:
    """Seconds of ``n`` back-to-back batches of ``size`` requests, after ``warm`` untimed ones."""
    out, i = [], start
    for j in range(warm + n):
        t = time.perf_counter()
        step(np.arange(i, i + size, dtype=np.int64))
        if j >= warm:
            out.append(time.perf_counter() - t)
        i += size
    return np.array(out)


def choose_max_batch(
    times: Callable[[int], np.ndarray], limit: float, largest: int = 1024
) -> tuple[int, list]:
    """The largest power of two up to ``largest`` whose p99 batch time is at most ``limit`` (1 if none); the
    sweep stops at the first size above it. ``times(size)`` returns batch times in seconds."""
    best, sweep, size = 1, [], 1
    while size <= largest:
        t = times(size)
        p99 = float(np.quantile(t, 0.99))
        sweep.append({"size": size} | quantiles_ms(t))
        if p99 > limit:
            break
        best = size
        size *= 2
    return best, sweep


def saturation(step: Callable, size: int, seconds: float) -> dict:
    """Requests per second with back-to-back batches of ``size`` for ``seconds``."""
    i, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        step(np.arange(i, i + size, dtype=np.int64))
        i += size
    elapsed = time.perf_counter() - t0
    return {"batch": size, "requests": i, "seconds": elapsed, "qps": i / elapsed}


def sequential_ms(fn: Callable[[int], None], n: int, warm: int) -> tuple[dict, np.ndarray]:
    """Latency of ``n`` calls ``fn(i)`` one after another, after ``warm`` untimed ones; ``fn`` must return
    only once its work is done."""
    for i in range(warm):
        fn(i)
    out = np.empty(n)
    for j in range(n):
        t = time.perf_counter()
        fn(warm + j)
        out[j] = time.perf_counter() - t
    return quantiles_ms(out) | {"n": n, "warm": warm}, out
