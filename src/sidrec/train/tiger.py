"""Train and evaluate TIGER on Beauty, as docs/protocol.md fixes.

    python -m sidrec.train.tiger --encoder st5-xxl --trial 1          # tuning trial: seed 0, no test
    python -m sidrec.train.tiger --encoder st5-xxl --trial 1 --test   # its best checkpoint on test
    python -m sidrec.train.tiger --encoder qwen3-emb-4b --trial 1 --seed 2 --final   # train, then test

Training examples: every position of each user's training part, with up to 20 preceding items as input.
Evaluation: beam search of width 20 constrained to valid semantic IDs; the top 10 are ranked, with no history
masking. Selection: validation NDCG@10, every epoch; patience 10, at most 300 epochs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time

import numpy as np
import torch

from sidrec.data import loo
from sidrec.eval.metrics import metrics_from_ranks, ranks_from_lists
from sidrec.models.tiger import Layout, Trie, build_t5, constrained_beam_search
from sidrec.semantic_ids.embed import ENCODERS
from sidrec.semantic_ids.rqvae import SID_DIR
from sidrec.train import common
from sidrec.tuning import trials

HISTORY = 20
MAX_EPOCHS = 300
PATIENCE = 10
WARMUP_SHARE = 0.05
BATCH = 256
BEAM = 20
CLIP = 1.0
FIXED = {
    "heads": 6,
    "head_dim": 64,
    "d_ff": 1024,
    "activation": "relu",
    "user_buckets": 2000,
    "grad_clip": CLIP,
}


def examples(data: loo.Beauty, tokens: np.ndarray, layout: Layout, split: str) -> dict[str, np.ndarray]:
    """Encoder inputs [M, 1 + 4 * HISTORY] (right-padded) and target items [M] for one split."""
    users, hists, targets = [], [], []
    for u, s in enumerate(data.seqs):
        if split == "train":
            part = s[:-2]
            ends = range(1, len(part))
        else:
            part = s[:-2] if split == "valid" else s[:-1]
            ends = [len(part)]
        for end in ends:
            users.append(u)
            hists.append(part[max(0, end - HISTORY) : end])
            targets.append(s[end])
    hist = np.zeros((len(hists), HISTORY), dtype=np.int64)
    for r, h in enumerate(hists):
        hist[r, : len(h)] = h  # oldest first, padding at the end
    user_tok = np.array([layout.user_token(data.user_ids[u]) for u in users], dtype=np.int64)
    inputs = np.concatenate([user_tok[:, None], tokens[hist].reshape(len(hist), -1)], axis=1)
    return {"inputs": inputs, "target": np.array(targets, dtype=np.int64)}


@torch.no_grad()
def evaluate(model, split: dict[str, torch.Tensor], trie: Trie, layout: Layout, chunk: int = 1024) -> dict:
    model.eval()
    ranks, slots = [], 0
    for s in range(0, split["inputs"].shape[0], chunk):
        inputs = split["inputs"][s : s + chunk]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            items, _ = constrained_beam_search(model, inputs, inputs > 0, trie, layout, beam=BEAM)
        slots += int((items[:, :10] < 0).sum())
        ranks.append(ranks_from_lists(items[:, :10], split["target"][s : s + chunk]))
    out = metrics_from_ranks(torch.cat(ranks))
    out["empty_top10_slots"] = slots
    return out


def load_ids(encoder: str):
    codes = np.load(SID_DIR / f"{encoder}.npy")
    layout = Layout((256, 256, 256, int(codes[1:, 3].max()) + 1))
    return codes, layout


def train(cfg: dict, seed: int, data: loo.Beauty, encoder: str, device: str, max_epochs: int = MAX_EPOCHS):
    common.seed_all(seed)
    codes, layout = load_ids(encoder)
    tokens = layout.item_tokens(codes)
    tr = {k: torch.as_tensor(v, device=device) for k, v in examples(data, tokens, layout, "train").items()}
    tr["labels"] = torch.as_tensor(tokens, device=device)[tr["target"]]
    va = {k: torch.as_tensor(v, device=device) for k, v in examples(data, tokens, layout, "valid").items()}
    trie = Trie(codes, layout.level_sizes, device)
    model = build_t5(layout, cfg["width"], cfg["layers"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = tr["inputs"].shape[0]
    steps_per_epoch = math.ceil(n / BATCH)
    warmup = max(1, int(WARMUP_SHARE * MAX_EPOCHS * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, common.warmup_schedule(warmup, "inverse_sqrt"))
    gen = torch.Generator(device=device).manual_seed(seed)
    best, best_epoch, best_state, best_valid, history = -1.0, 0, None, None, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        t_epoch = time.time()
        order = torch.randperm(n, device=device, generator=gen)
        total = torch.zeros((), device=device)
        for s in range(0, n, BATCH):
            idx = order[s : s + BATCH]
            inputs = tr["inputs"][idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(inputs, inputs > 0, tr["labels"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            sched.step()
            total += loss.detach() * len(idx)
        train_s = time.time() - t_epoch
        metrics = evaluate(model, va, trie, layout)
        history.append(
            {
                "epoch": epoch,
                "loss": round(total.item() / n, 5),
                "valid_ndcg@10": metrics["ndcg@10"],
                "valid_recall@10": metrics["recall@10"],
                "train_s": round(train_s, 1),
                "eval_s": round(time.time() - t_epoch - train_s, 1),
            }
        )
        print(history[-1], flush=True)
        if metrics["ndcg@10"] > best:
            best, best_epoch, best_valid = metrics["ndcg@10"], epoch, metrics
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    run = {
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "train_examples": n,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup,
        "level_sizes": list(layout.level_sizes),
        "vocab_size": layout.vocab_size,
        "valid": best_valid,
        "history": history,
        "params": sum(p.numel() for p in model.parameters()),
    }
    return model, run, (data, tokens, layout, trie)


def test_metrics(model, data, tokens, layout, trie, device) -> dict:
    te = {k: torch.as_tensor(v, device=device) for k, v in examples(data, tokens, layout, "test").items()}
    return evaluate(model, te, trie, layout)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--trial", type=int, required=True, help="1..12, the protocol's trial list")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final", action="store_true", help="evaluate on test after training")
    parser.add_argument("--test", action="store_true", help="only evaluate the saved checkpoint on test")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="for smoke tests only")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    cfg = trials("tiger")[args.trial - 1]
    run_dir = common.RESULTS / "runs" / f"tiger_{args.encoder}"
    tag = f"trial{args.trial:02d}" + (f"_seed{args.seed}" if args.final else "")
    ckpt = common.CHECKPOINTS / f"tiger_{args.encoder}" / f"{tag}.pt"
    data = loo.load()
    t0 = time.time()
    smi_before = common.nvidia_smi()
    git = common.git_state()  # the code this run started with
    if args.test:  # the selected tuning trial becomes the seed-0 final run; test is computed only here
        record = json.loads((run_dir / f"{tag}.json").read_text())
        codes, layout = load_ids(args.encoder)
        model = build_t5(layout, cfg["width"], cfg["layers"], cfg["dropout"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        trie = Trie(codes, layout.level_sizes, device)
        record["test"] = test_metrics(model, data, layout.item_tokens(codes), layout, trie, device)
        record["test_eval"] = {"nvidia_smi": smi_before, "wall_time_s": round(time.time() - t0, 1)}
        record["phase"] = "final"
        common.write_json(run_dir / f"{tag}_seed0.json", record)
        print(record["test"])
        return
    model, run, (data, tokens, layout, trie) = train(
        cfg, args.seed, data, args.encoder, device, args.max_epochs
    )
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    record = {
        "model": "tiger",
        "encoder": args.encoder,
        "embedding_model": ENCODERS[args.encoder][0],
        "dataset": "beauty",
        "phase": "final" if args.final else "tune",
        "trial": args.trial,
        "seed": args.seed,
        "config": cfg,
        "fixed": {
            "batch": BATCH,
            "history": HISTORY,
            "beam": BEAM,
            "optimizer": "AdamW, inverse-sqrt",
            **FIXED,
        },
        **run,
        "test": test_metrics(model, data, tokens, layout, trie, device) if args.final else None,
    }
    record.update(
        wall_time_s=round(time.time() - t0, 1),
        gpu={"nvidia_smi_start": smi_before, "nvidia_smi_end": common.nvidia_smi()},
        environment=common.environment(),
        git=git,
        protocol=common.PROTOCOL_VERSION,
        checkpoint=str(ckpt.relative_to(common.ROOT)),
    )
    common.write_json(run_dir / f"{tag}.json", record)
    print(record["valid"], record["test"])


if __name__ == "__main__":
    main()
