"""Train and evaluate the item-ID models (SASRec, HSTU-style) on Beauty, as docs/protocol.md fixes.

    python -m sidrec.train.idrec --model sasrec --trial 3                   # tuning trial: seed 0, no test
    python -m sidrec.train.idrec --model sasrec --trial 3 --test            # its best checkpoint on test
    python -m sidrec.train.idrec --model sasrec --trial 3 --seed 1 --final  # final run: train, then test

Loss: full-softmax cross-entropy over all items at every non-padding position of each user's training window.
Selection: validation NDCG@10, evaluated every epoch; patience 10, at most 300 epochs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time

import torch
import torch.nn.functional as F

from sidrec.data import loo
from sidrec.eval.metrics import history_exclusion, metrics_from_ranks, target_ranks
from sidrec.models.hstu import HSTU
from sidrec.models.sasrec import SASRec
from sidrec.train import common
from sidrec.tuning import trials

MAX_LEN = 50
MAX_EPOCHS = 300
PATIENCE = 10
WARMUP_SHARE = 0.05
BATCH = {"sasrec": 256, "hstu": 128}  # S3-Rec; generative-recommenders amzn-books
FIXED = {
    "sasrec": {"heads": 2, "ffn": "4 x width, GELU", "norm": "post-LN", "scores": "dot product, tied table"},
    "hstu": {"heads": 4, "head_dim": "width / 4", "rab_time_buckets": 128, "temperature": 0.05},
}


def build(model: str, cfg: dict, n_items: int) -> torch.nn.Module:
    cls = {"sasrec": SASRec, "hstu": HSTU}[model]
    return cls(n_items, cfg["width"], cfg["layers"], cfg["dropout"], max_len=MAX_LEN)


def to_device(arrays: dict, device: str) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, split: dict[str, torch.Tensor], n_items: int, chunk: int = 2048) -> dict:
    model.eval()
    ranks = []
    for s in range(0, split["inputs"].shape[0], chunk):
        b = {k: v[s : s + chunk] for k, v in split.items()}
        states = model.encode(b["inputs"], input_ts=b["input_ts"], query_ts=b["query_ts"])[:, -1]
        scores = model.logits(states)
        exclude = history_exclusion(b["inputs"], n_items + 1)
        ranks.append(target_ranks(scores, b["target"], exclude))
    return metrics_from_ranks(torch.cat(ranks))


def train(model_name: str, cfg: dict, seed: int, data: loo.Beauty, device: str, max_epochs: int = MAX_EPOCHS):
    common.seed_all(seed)
    train_s = to_device(loo.id_model_arrays(data, "train", MAX_LEN), device)
    valid_s = to_device(loo.id_model_arrays(data, "valid", MAX_LEN), device)
    model = build(model_name, cfg, data.n_items).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    batch = BATCH[model_name]
    users = train_s["inputs"].shape[0]
    steps_per_epoch = math.ceil(users / batch)
    warmup = max(1, int(WARMUP_SHARE * MAX_EPOCHS * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, common.warmup_schedule(warmup, "constant"))
    best, best_epoch, best_state, history = -1.0, 0, None, []
    gen = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(users, device=device, generator=gen)
        total, count = 0.0, 0
        for s in range(0, users, batch):
            b = {k: v[order[s : s + batch]] for k, v in train_s.items()}
            states = model.encode(b["inputs"], input_ts=b["input_ts"], query_ts=b["query_ts"])
            keep = b["targets"] > 0
            logits = model.logits(states[keep])[:, 1:]  # column 0 is padding, not a class
            loss = F.cross_entropy(logits, b["targets"][keep] - 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * int(keep.sum())
            count += int(keep.sum())
        metrics = evaluate(model, valid_s, data.n_items)
        history.append({"epoch": epoch, "loss": round(total / count, 5), "valid_ndcg@10": metrics["ndcg@10"]})
        if metrics["ndcg@10"] > best:
            best, best_epoch, best_valid = metrics["ndcg@10"], epoch, metrics
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break
        print(f"epoch {epoch} loss {total / count:.4f} valid ndcg@10 {metrics['ndcg@10']:.4f}", flush=True)
    model.load_state_dict(best_state)
    run = {
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup,
        "valid": best_valid,
        "history": history,
        "params": sum(p.numel() for p in model.parameters()),
    }
    return model, run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=["sasrec", "hstu"], required=True)
    parser.add_argument("--trial", type=int, required=True, help="1..12, the protocol's trial list")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final", action="store_true", help="evaluate on test after training")
    parser.add_argument("--test", action="store_true", help="only evaluate the saved checkpoint on test")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="for smoke tests only")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    cfg = trials(args.model)[args.trial - 1]
    tag = f"trial{args.trial:02d}" + (f"_seed{args.seed}" if args.final else "")
    ckpt = common.CHECKPOINTS / args.model / f"{tag}.pt"
    data = loo.load()
    t0 = time.time()
    smi_before = common.nvidia_smi()
    if args.test:  # the selected tuning trial becomes the seed-0 final run; test is computed only here
        record = json.loads((common.RESULTS / "runs" / args.model / f"{tag}.json").read_text())
        model = build(args.model, cfg, data.n_items).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        record["test"] = evaluate(
            model, to_device(loo.id_model_arrays(data, "test", MAX_LEN), device), data.n_items
        )
        record["test_eval"] = {"nvidia_smi": smi_before, "wall_time_s": round(time.time() - t0, 1)}
        record["phase"] = "final"
        out = common.RESULTS / "runs" / args.model / f"{tag}_seed0.json"
        common.write_json(out, record)
        print(out, record["test"])
        return
    model, run = train(args.model, cfg, args.seed, data, device, args.max_epochs)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    record = {
        "model": args.model,
        "dataset": "beauty",
        "phase": "final" if args.final else "tune",
        "trial": args.trial,
        "seed": args.seed,
        "config": cfg,
        "fixed": {"batch": BATCH[args.model], "max_len": MAX_LEN, "optimizer": "AdamW", **FIXED[args.model]},
        **run,
        "test": None,
    }
    if args.final:
        record["test"] = evaluate(
            model, to_device(loo.id_model_arrays(data, "test", MAX_LEN), device), data.n_items
        )
    record.update(
        wall_time_s=round(time.time() - t0, 1),
        gpu={"nvidia_smi_start": smi_before, "nvidia_smi_end": common.nvidia_smi()},
        environment=common.environment(),
        git=common.git_state(),
        protocol=common.PROTOCOL_VERSION,
        checkpoint=str(ckpt.relative_to(common.ROOT)),
    )
    out = common.RESULTS / "runs" / args.model / f"{tag}.json"
    common.write_json(out, record)
    print(out, record["valid"], record["test"])


if __name__ == "__main__":
    main()
