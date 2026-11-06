"""Electronics item embeddings with vLLM 0.31.0's pooling runner.

    PYTHONPATH=src .venv-vllm/bin/python -m sidrec.semantic_ids.embed_vllm [--sample 20000]

It runs in its own environment (.venv-vllm, built from the vLLM 0.31.0 wheelhouse), which has no polars, so
it reads the item documents that ``python -m sidrec.semantic_ids.embed_catalog docs`` wrote. The engine is
shut down before the process exits. Vectors are L2-normalized after pooling, as R2's loop does; the norms vLLM
returned are recorded. The full run writes data/processed/electronics/emb/vllm.npy ([368,228, 1,024], item
order); every run writes its record to results/r3/embeddings/.
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np

from sidrec.semantic_ids import catalog
from sidrec.semantic_ids.embed import ENCODERS
from sidrec.train import common

CHUNK = 50_000  # prompts per LLM.encode call; bounds the memory the queued requests hold


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--encoder", choices=["qwen3-emb-0.6b"], default="qwen3-emb-0.6b")
    parser.add_argument("--sample", type=int, default=0, help="encode this many items (seed 0); 0 = all")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.83)
    args = parser.parse_args()
    assert args.gpu_memory_utilization <= 0.83

    ids, docs = catalog.load_docs_json()
    rows = catalog.sample_rows(len(docs), args.sample) if args.sample else np.arange(len(docs))
    texts = [docs[i] for i in rows]
    repo, revision = ENCODERS[args.encoder]
    smi_start = common.nvidia_smi()
    t_start = time.time()

    import vllm
    from vllm import LLM

    llm = LLM(
        model=repo,
        revision=revision,
        runner="pooling",
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
    )
    ready_s = time.time() - t_start
    cfg = llm.llm_engine.vllm_config
    engine = {
        "max_model_len": cfg.model_config.max_model_len,
        "max_num_seqs": cfg.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
        "pooler_config": str(cfg.model_config.pooler_config),
        "enforce_eager": cfg.model_config.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    emb = np.empty((len(texts), 1024), dtype=np.float32)
    tokens = 0
    with common.SmiSampler() as sampler:
        t0 = time.time()
        for s in range(0, len(texts), CHUNK):
            outs = llm.encode(texts[s : s + CHUNK], pooling_task="embed", use_tqdm=False)
            for j, out in enumerate(outs):
                emb[s + j] = out.outputs.data.float().cpu().numpy()
                tokens += len(out.prompt_token_ids)
        seconds = time.time() - t0
    llm.llm_engine.engine_core.shutdown(timeout=60)
    del llm
    gc.collect()

    norms = np.linalg.norm(emb, axis=1)
    emb /= norms[:, None]
    full = not args.sample
    name = "vllm" if full else f"vllm-sample{args.sample}"
    record = {
        "implementation": "vllm",
        "vllm": vllm.__version__,
        "encoder": args.encoder,
        "repository": repo,
        "revision": revision,
        "dtype": "bfloat16",
        "engine": engine,
        "items": len(texts),
        "sample_seed": 0 if args.sample else None,
        "item_ids_sha256": catalog.ids_sha256([ids[i] for i in rows]),
        "tokens": tokens,
        "startup_s": round(ready_s, 1),
        "encode_seconds": round(seconds, 2),
        "items_per_s": round(len(texts) / seconds, 1),
        "tokens_per_s": round(tokens / seconds, 1),
        "returned_norm_min_max": [round(float(norms.min()), 4), round(float(norms.max()), 4)],
        "non_finite_values": int((~np.isfinite(emb)).sum()),
        "wall_time_s": round(time.time() - t_start, 1),
        "gpu": {
            "nvidia_smi_start": smi_start,
            "during_encode": sampler.summary(),
            "nvidia_smi_end": common.nvidia_smi(),
        },
        "environment": common.environment(),
        "git": common.git_state(),
    }
    if full:
        catalog.EMB_DIR.mkdir(parents=True, exist_ok=True)
        path = catalog.EMB_DIR / "vllm.npy"
        np.save(path, emb)
        record["array"] = {"path": str(path.relative_to(common.ROOT)), "sha256": catalog.file_sha256(path)}
    common.write_json(catalog.RESULTS / "embeddings" / f"{name}.json", record)
    print(
        {
            k: record[k]
            for k in ("items", "tokens", "startup_s", "encode_seconds", "items_per_s", "tokens_per_s")
        }
    )


if __name__ == "__main__":
    main()
