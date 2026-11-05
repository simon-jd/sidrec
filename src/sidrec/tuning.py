"""The tuning budget of docs/protocol.md, section 3: 12 shared trials, trial 1 the published setting."""

from __future__ import annotations

import itertools
import random

GRID = {
    "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "weight_decay": [0.0, 1e-4, 1e-2],
    "width": [64, 128, 256],
    "layers": [2, 4],
}
SEED = 20261105
N_TRIALS = 12

# Trial 1 on Beauty: the configuration published for each model (see the protocol's Changes section).
PUBLISHED = {
    # S3-Rec run_finetune_full.py defaults: hidden 64, 2 layers, 2 heads, dropout 0.5, Adam lr 1e-3, no decay.
    "sasrec": {"lr": 1e-3, "dropout": 0.5, "weight_decay": 0.0, "width": 64, "layers": 2},
    # generative-recommenders configs/amzn-books/hstu-sampled-softmax-n512-final.gin: embedding 64, 4 blocks,
    # 4 heads of 16, dropout 0.5, lr 1e-3, weight decay 0.
    "hstu": {"lr": 1e-3, "dropout": 0.5, "weight_decay": 0.0, "width": 64, "layers": 4},
    # TIGER section 4: 4 encoder and 4 decoder layers, width 128, dropout 0.1, lr 0.01, inverse-sqrt decay.
    "tiger": {"lr": 1e-2, "dropout": 0.1, "weight_decay": 0.0, "width": 128, "layers": 4},
}


def shared_trials() -> list[dict]:
    grid = [dict(zip(GRID, values, strict=True)) for values in itertools.product(*GRID.values())]
    return random.Random(SEED).sample(grid, N_TRIALS)


def trials(model: str) -> list[dict]:
    """The 12 Beauty trials of ``model``, numbered from 1."""
    return [dict(PUBLISHED[model])] + shared_trials()[1:]
