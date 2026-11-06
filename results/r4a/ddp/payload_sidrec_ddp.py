"""sidrec R4a: DDP scaling of the HSTU-style ranker, 1 GPU against 2 at one global batch, on PORTFOLIO_2XA10_POOL.

argv: <sidrec source tar> <data tar> <trial> <plan> [budget minutes]

plan "smoke": 1 and 2 processes at global batch 128, 10 untimed and 40 timed steps each.
plan "main":  1 and 2 processes at global batch 128 (500 timed steps), 1 process at 64 (the per-GPU share
              alone), then 1 and 2 processes at global batch 1024 (150 timed steps).

Fetches the assets from @PORTFOLIO_LAB.GPU.ASSETS (PF_LOCAL_ASSETS=<dir> for a dry run), checks them against
SHA256SUMS, unpacks the CUDA 13.2 compat driver, the torch wheels of the frozen vLLM 0.31.0 wheelhouse (torch
2.13.0+cu130), the sidrec source at one commit and the Electronics input arrays, installs a venv with
--no-index --no-deps, then runs ``python -m sidrec.train.ddp_scaling`` once per plan entry (torch.distributed.run
for 2 processes, NCCL_DEBUG=INFO to record the transport). Prints a RESULT line per entry and a summary.
An entry is skipped when it would overrun the budget, and 2-process entries are skipped with fewer than 2 GPUs.

Dry run: PF_LOCAL_ASSETS=<dir> PF_WORK=<dir on disk> PF_GPU_LOCK=/var/tmp/rtx4090.lock (each entry holds the
lock for its own duration). A dry run keeps the venv between invocations (PF_WORK/venv/.installed).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

WORK = os.environ.get("PF_WORK", "/tmp/pf-sidrec")
LOCAL = os.environ.get("PF_LOCAL_ASSETS")
GPU_LOCK = os.environ.get("PF_GPU_LOCK")
BASE = ["cuda-compat-13-2.tar", "wheelhouse-vllm-0.31.0.tar"]
VENV = os.path.join(WORK, "venv")
START = time.time()
TORCH_WHEELS = ("torch-", "triton-", "numpy-", "filelock-", "typing_extensions-", "sympy-", "mpmath-",
                "networkx-", "jinja2-", "markupsafe-", "fsspec-", "setuptools-", "nvidia_", "cuda_")
PLANS = {
    "smoke": [("1gpu_b128", 1, 128, 10, 40), ("2gpu_b128", 2, 128, 10, 40)],
    "main": [("1gpu_b128", 1, 128, 50, 500), ("2gpu_b128", 2, 128, 50, 500), ("1gpu_b64", 1, 64, 50, 500),
             ("1gpu_b1024", 1, 1024, 20, 150), ("2gpu_b1024", 2, 1024, 20, 150)],
}


def emit(step, payload):
    print("RESULT " + json.dumps({"step": step, "t_s": round(time.time() - START, 1), step: payload}), flush=True)


def sh(cmd, env=None, timeout=900, cwd=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        return {"rc": p.returncode, "out": p.stdout, "err": p.stderr}
    except Exception as e:
        return {"rc": None, "out": "", "err": repr(e)[:800]}


def text(cmd):
    r = sh(cmd, timeout=60)
    return (r["out"] or r["err"]).strip()[:3000]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def untar(path, dest):
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(path) as t:
        try:
            t.extractall(dest, filter="data")
        except TypeError:
            t.extractall(dest)


def node_info():
    return {
        "python": sys.version.split()[0], "cpus": os.cpu_count(),
        "cpu_model": text(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo"]),
        "gpus": text(["nvidia-smi", "-L"]), "topo": text(["nvidia-smi", "topo", "-m"]),
        "p2p_read": text(["nvidia-smi", "topo", "-p2p", "r"]),
        "nvlink": text(["nvidia-smi", "nvlink", "--status"]),
        "query": text(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,clocks.max.sm,"
                       "pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current",
                       "--format=csv,noheader"]),
    }


def bootstrap(names, out):
    os.makedirs(WORK, exist_ok=True)
    marker = os.path.join(VENV, ".installed")
    reuse = bool(LOCAL) and os.path.exists(marker)
    fetch = names if not reuse else names[2:]
    t = time.time()
    if LOCAL:
        for n in fetch + ["SHA256SUMS"]:
            shutil.copy(os.path.join(LOCAL, n), WORK)
    else:
        from snowflake.snowpark import Session
        s = Session.builder.getOrCreate()
        for n in fetch + ["SHA256SUMS"]:
            s.file.get(f"@PORTFOLIO_LAB.GPU.ASSETS/{n}", WORK)
    out["fetch_s"] = round(time.time() - t, 1)
    want = dict(reversed(line.split()) for line in open(os.path.join(WORK, "SHA256SUMS")) if line.strip())
    out["sha256_mismatch"] = [n for n in fetch if sha256(os.path.join(WORK, n)) != want.get(n)]
    if out["sha256_mismatch"]:
        raise RuntimeError(f"sha256 mismatch: {out['sha256_mismatch']}")
    out["sha256"] = {n: want.get(n) for n in names}
    compat, wheels, src, data = (os.path.join(WORK, d) for d in ("compat", "wheels", "src", "data"))
    for d in (src, data):
        shutil.rmtree(d, ignore_errors=True)
    t = time.time()
    for dest, n in zip((compat, wheels, src, data), names):
        if n in fetch:
            untar(os.path.join(WORK, n), dest)
            os.remove(os.path.join(WORK, n))
    out["untar_s"] = round(time.time() - t, 1)
    if not reuse:
        t = time.time()
        shutil.rmtree(VENV, ignore_errors=True)
        sh([sys.executable, "-m", "venv", VENV], timeout=300)
        whls = sorted(os.path.join(wheels, f) for f in os.listdir(wheels)
                      if f.endswith(".whl") and f.startswith(TORCH_WHEELS))
        r = sh([os.path.join(VENV, "bin", "pip"), "install", "-q", "--no-index", "--no-deps", *whls], timeout=1800)
        out["install"] = {"rc": r["rc"], "wheels": len(whls), "seconds": round(time.time() - t, 1),
                          "err_tail": (r.get("err") or "").strip().splitlines()[-4:]}
        if r["rc"] != 0:
            raise RuntimeError("venv install failed")
        shutil.rmtree(wheels, ignore_errors=True)
        open(marker, "w").close()
    out["venv_reused"] = reuse
    env = dict(os.environ, PYTHONPATH=os.path.join(src, "src"), OMP_NUM_THREADS="8",
               PATH=os.path.join(VENV, "bin") + ":" + os.environ.get("PATH", ""))
    if not LOCAL:  # the node's driver 570 is CUDA 12.8; torch 2.13.0+cu130 needs the compat driver
        env["LD_LIBRARY_PATH"] = compat + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    commit_file = os.path.join(src, "SOURCE_COMMIT")
    if os.path.exists(commit_file):
        env["SIDREC_COMMIT"] = out["source_commit"] = open(commit_file).read().strip()
    py = os.path.join(VENV, "bin", "python")
    check = sh([py, "-c", "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda': "
                "torch.version.cuda, 'gpus': torch.cuda.device_count(), 'nccl': torch.cuda.nccl.version()}))"],
               env=env, timeout=300)
    out["torch"] = json.loads(check["out"].strip().splitlines()[-1]) if check["rc"] == 0 else check["err"][-800:]
    return py, env, data


def main():
    src_tar, data_tar, trial, plan = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    budget = float(sys.argv[5]) * 60 if len(sys.argv) > 5 else 30 * 60
    out = {}
    try:
        emit("node", node_info())
        py, env, data = bootstrap(BASE + [src_tar, data_tar], out)
        emit("bootstrap", out)
        gpus = out["torch"]["gpus"] if isinstance(out["torch"], dict) else 0
        npz = os.path.join(data, "full.npz")
        runs, longest = {}, 0.0
        for name, procs, batch, warm, steps in PLANS[plan]:
            if procs > gpus:
                emit(name, {"skipped": f"{procs} processes, {gpus} GPU(s)"})
                continue
            left = budget - (time.time() - START)
            if left < 1.5 * longest + 60:
                emit(name, {"skipped": f"budget: {left:.0f} s left, longest entry {longest:.0f} s"})
                continue
            res = os.path.join(WORK, "out", f"{name}.json")
            args = ["-m", "sidrec.train.ddp_scaling", "--trial", trial, "--global-batch", str(batch),
                    "--warmup", str(warm), "--steps", str(steps), "--data", npz, "--out", res]
            cmd = [py, *args] if procs == 1 else [py, "-m", "torch.distributed.run", "--standalone",
                                                   f"--nproc_per_node={procs}", *args]
            if GPU_LOCK:
                cmd = ["flock", GPU_LOCK, *cmd]
            run_env = dict(env, NCCL_DEBUG="INFO") if procs > 1 else env
            t = time.time()
            r = sh(cmd, env=run_env, timeout=max(120, int(left - 30)))
            sec = time.time() - t
            longest = max(longest, sec)
            lines = [x.strip()[:220] for x in (r["out"] + "\n" + r["err"]).splitlines() if "NCCL INFO" in x]
            nccl = ([x for x in lines if " via " in x][:12]
                    + [x for x in lines if any(w in x for w in ("NCCL version", "P2P", "SHM", "NET/", "Connected all"))][:12])
            if r["rc"] == 0 and os.path.exists(res):
                runs[name] = json.load(open(res))
                emit(name, {"wall_s": round(sec, 1), "record": runs[name], "nccl": nccl})
            else:
                emit(name, {"rc": r["rc"], "wall_s": round(sec, 1), "err_tail": (r["err"] or "").splitlines()[-25:],
                            "nccl": nccl})
        summary = {}
        for batch in (128, 1024):
            one, two = runs.get(f"1gpu_b{batch}"), runs.get(f"2gpu_b{batch}")
            if one and two:
                t1, t2 = one["rank0_step_ms"]["mean"], two["rank0_step_ms"]["mean"]
                summary[f"b{batch}"] = {"step_ms_1gpu": t1, "step_ms_2gpu": t2, "speedup": round(t1 / t2, 3),
                                        "scaling_efficiency": round(t1 / (2 * t2), 3)}
        out_summary = {"summary": summary, "entries": sorted(runs), "total_s": round(time.time() - START, 1)}
        emit("summary", out_summary)
    except Exception as e:
        emit("error", {"error": repr(e)[:1500], **out})


if __name__ == "__main__":
    main()
