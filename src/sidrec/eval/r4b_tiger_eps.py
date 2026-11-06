"""Validation-only ε curve of TIGER's cold-start retrieval (R4b analysis; nothing is selected from it).

    python -m sidrec.eval.r4b_tiger_eps --seed 0

Loads the final TIGER checkpoint of one seed, runs the beam once over the protocol's 100,000-row validation
sample and scores the cold-start list at every ε in ``EPS`` (``sidrec.models.tiger_cold``). ε = 0 is plain
TIGER. The test split is not read. The record goes to results/r4b/tiger_eps/seed<S>.json.
"""

from __future__ import annotations

import argparse
import time

import torch

from sidrec.data import scale
from sidrec.models.tiger import Trie, build_t5
from sidrec.train import common, resume
from sidrec.train.scale_idrec import (
    CHECKPOINTS_R4B,
    PROTOCOL_VERSION_R4B,
    RESULTS_R4B,
    SAMPLE,
    selected_trial,
)
from sidrec.train.scale_tiger import Inputs, cold_start_inputs, evaluate, load_ids
from sidrec.tuning import shared_trials

EPS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    trial = selected_trial("tiger")
    cfg = shared_trials()[trial - 1]
    ckpt = CHECKPOINTS_R4B / "tiger" / f"seed{args.seed}.pt"
    d = scale.load("full")
    t0 = time.time()
    smi_before, git = common.nvidia_smi(), resume.code_state()
    with common.SmiSampler() as sampler:
        codes, layout = load_ids()
        inp = Inputs(d, layout.item_tokens(codes), layout, device)
        trie = Trie(codes, layout.level_sizes, device)
        model = build_t5(layout, cfg["width"], cfg["layers"], cfg["dropout"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        rows, users, cold, _ = scale.eval_rows(d, "valid", SAMPLE, device)
        unseen, key = cold_start_inputs(d, codes, device)
        t = time.time()
        metrics = evaluate(
            model, inp, rows, users, cold, trie, layout, eps=EPS, unseen=unseen, prefix_key=key
        )
        seconds = round(time.time() - t, 1)
    record = {
        "model": "tiger",
        "analysis": "cold-start retrieval, validation-only ε curve; nothing is selected from it",
        "dataset": "electronics-full",
        "split": "valid",
        "trial": trial,
        "seed": args.seed,
        "checkpoint": str(ckpt.relative_to(common.ROOT)),
        "eps": list(EPS),
        "metrics": metrics,
        "eval_seconds": seconds,
        "wall_time_s": round(time.time() - t0, 1),
        "gpu": {
            "nvidia_smi_start": smi_before,
            "nvidia_smi_end": common.nvidia_smi(),
            "sampled": sampler.summary(),
        },
        "environment": common.environment(),
        "git": git,
        "protocol": PROTOCOL_VERSION_R4B,
    }
    out = RESULTS_R4B / "tiger_eps" / f"seed{args.seed}.json"
    common.write_json(out, record)
    print(out, {e: metrics["cold_start"][str(e)]["cold"]["recall@50"] for e in EPS})


if __name__ == "__main__":
    main()
