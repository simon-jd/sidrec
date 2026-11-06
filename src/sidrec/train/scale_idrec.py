"""Train and evaluate the item-ID models (SASRec, HSTU-style) on Electronics, as docs/protocol.md fixes.

    python -m sidrec.train.scale_idrec --model hstu --trial 3                       # dev-subset tuning trial
    python -m sidrec.train.scale_idrec --model hstu --trial 1 --source full --cost  # cost measurement

Tuning trials (seed 0) train on the dev subset's train split and evaluate on its validation rows every epoch;
they compute no test metrics. ``--cost`` trains one epoch, evaluates the 100,000-row validation sample, then
times one evaluation of every eligible validation row; it writes a cost record and no checkpoint.

Loss: exact full-softmax cross-entropy over the V train items at every non-padding position of each user's
training window, computed in row chunks by ``sidrec.train.fused_ce``. Ranking: every catalog item; items
outside the train split rank below every train item, so a cold target is a miss.
"""

from __future__ import annotations

import argparse
import copy
import math
import time

import torch
import torch.nn.functional as F

from sidrec.data import scale
from sidrec.eval.metrics import MISS, split_metrics, target_ranks
from sidrec.models.hstu import HSTU
from sidrec.models.sasrec import SASRec
from sidrec.train import common
from sidrec.train.fused_ce import fused_linear_cross_entropy
from sidrec.tuning import shared_trials

MAX_LEN = 50
MAX_EPOCHS = 30
PATIENCE = 3
WARMUP_SHARE = 0.05
SAMPLE = 100_000
BATCH = {"sasrec": 256, "hstu": 128}  # the R2 values, on both the dev subset and the full train split
FIXED = {
    "sasrec": {"heads": 2, "ffn": "4 x width, GELU", "norm": "post-LN", "scores": "dot product, tied table"},
    "hstu": {"heads": 4, "head_dim": "width / 4", "rab_time_buckets": 128, "temperature": 0.05},
}
RESULTS = common.ROOT / "results" / "r4a"
CHECKPOINTS = common.ROOT / "data" / "runs" / "r4a"
PROTOCOL_VERSION = "2026-11-05 with the R2, R3 and R4a clarifications"


def build(model: str, cfg: dict, v: int) -> torch.nn.Module:
    """Item table rows: 0 padding, 1..v the train items, v + 1 the shared out-of-vocabulary row (zero)."""
    if model == "sasrec":
        m = SASRec(v + 1, cfg["width"], cfg["layers"], cfg["dropout"], max_len=MAX_LEN)
    else:
        m = HSTU(v + 1, cfg["width"], cfg["layers"], cfg["dropout"], max_len=MAX_LEN, time_bias=True)
    with torch.no_grad():
        m.item_emb.weight[v + 1].zero_()
    return m


def output_table(model: torch.nn.Module, v: int) -> torch.Tensor:
    """[v, d] scoring table of the train items: SASRec's tied item table; the HSTU-style model's
    L2-normalized table divided by its temperature."""
    w = model.item_emb.weight[1 : v + 1]
    if isinstance(model, HSTU):
        return F.normalize(w, dim=-1) / model.temperature
    return w


@torch.no_grad()
def evaluate(model, flat: scale.Flat, rows: scale.Examples, cold, vocab, v: int, chunk: int = 1024) -> dict:
    model.eval()
    table = output_table(model, v)
    ranks = []
    for s in range(0, len(rows), chunk):
        b = flat.id_inputs(rows.take(slice(s, s + chunk)), vocab, MAX_LEN, train=False)
        states = model.encode(b["inputs"], input_ts=b["input_ts"], query_ts=b["query_ts"])[:, -1]
        scores = states @ table.T  # column j is train item j + 1, in catalog order
        ranks.append(vocab_ranks(scores, vocab[b["history"]], vocab[b["target"]], v))
    return split_metrics(torch.cat(ranks), cold)


def vocab_ranks(scores: torch.Tensor, history: torch.Tensor, target: torch.Tensor, v: int) -> torch.Tensor:
    """Protocol ranks from scores over the train items ([B, v], column j is model index j + 1).

    history and target hold model indices (0 padding, v + 1 out of vocabulary). Items outside the train split
    rank below every train item, so a target outside it is a miss (rank > 50 in the full catalog); the
    tie-break by lower index is the catalog's, because train items keep catalog order."""
    exclude = torch.zeros(len(history), v + 2, dtype=torch.bool, device=history.device)
    exclude.scatter_(1, history, True)
    ranks = target_ranks(scores, (target - 1).clamp(0, v - 1), exclude[:, 1 : v + 1])
    return ranks.masked_fill(target > v, MISS)


def train(name: str, cfg: dict, seed: int, d: dict, device: str, max_epochs: int, sample: int | None):
    common.seed_all(seed)
    flat = scale.Flat(d, device)
    vocab_np, v = scale.id_vocab(d)
    vocab = torch.as_tensor(vocab_np, device=device)
    ex = scale.id_train_examples(d, device)
    va, _, va_cold, _ = scale.eval_rows(d, "valid", sample, device)
    model = build(name, cfg, v).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    batch, n = BATCH[name], len(ex)
    steps_per_epoch = math.ceil(n / batch)
    warmup = max(1, int(WARMUP_SHARE * MAX_EPOCHS * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, common.warmup_schedule(warmup, "constant"))
    gen = torch.Generator(device=device).manual_seed(seed)
    best, best_epoch, best_state, best_valid, history = -1.0, 0, None, None, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        torch.cuda.synchronize()
        t0 = time.time()
        order = torch.randperm(n, device=device, generator=gen)
        total = torch.zeros((), device=device)
        count = torch.zeros((), device=device, dtype=torch.long)
        for s in range(0, n, batch):
            b = flat.id_inputs(ex.take(order[s : s + batch]), vocab, MAX_LEN, train=True)
            states = model.encode(b["inputs"], input_ts=b["input_ts"], query_ts=b["query_ts"])
            keep = b["targets"] > 0
            loss_sum = fused_linear_cross_entropy(
                states[keep], output_table(model, v), b["targets"][keep] - 1
            )
            k = keep.sum()
            opt.zero_grad(set_to_none=True)
            (loss_sum / k).backward()
            opt.step()
            sched.step()
            total += loss_sum.detach()
            count += k
        torch.cuda.synchronize()
        train_s = time.time() - t0
        metrics = evaluate(model, flat, va, va_cold, vocab, v)
        eval_s = time.time() - t0 - train_s
        history.append(
            {
                "epoch": epoch,
                "loss": round(total.item() / count.item(), 5),
                "valid_ndcg@10": metrics["all"]["ndcg@10"],
                "valid_recall@10": metrics["all"]["recall@10"],
                "valid_recall@50": metrics["all"]["recall@50"],
                "train_s": round(train_s, 1),
                "eval_s": round(eval_s, 1),
            }
        )
        print(history[-1], flush=True)
        if metrics["all"]["ndcg@10"] > best:
            best, best_epoch, best_valid = metrics["all"]["ndcg@10"], epoch, metrics
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    run = {
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "train_windows": n,
        "train_targets": int(count.item()),
        "vocab_items": v,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup,
        "valid_rows": len(va),
        "valid": best_valid,
        "history": history,
        "params": sum(p.numel() for p in model.parameters()),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    return model, run, (flat, vocab, v)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(BATCH), required=True)
    parser.add_argument("--trial", type=int, required=True, help="1..12, the protocol's shared trial list")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", choices=["dev", "full"], default="dev")
    parser.add_argument("--cost", action="store_true", help="one epoch and timed evaluations, no checkpoint")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    cfg = shared_trials()[args.trial - 1]
    d = scale.load(args.source)
    t0 = time.time()
    smi_before = common.nvidia_smi()
    git = common.git_state()  # the code this run started with
    with common.SmiSampler() as sampler:
        model, run, (flat, vocab, v) = train(
            args.model, cfg, args.seed, d, device, 1 if args.cost else MAX_EPOCHS, SAMPLE
        )
        if args.cost:  # one evaluation over every eligible validation row, timed; nothing is selected from it
            rows, _, cold, _ = scale.eval_rows(d, "valid", None, device)
            torch.cuda.synchronize()
            t = time.time()
            full = evaluate(model, flat, rows, cold, vocab, v)
            run["full_valid_eval"] = {
                "rows": len(rows),
                "seconds": round(time.time() - t, 1),
                "metrics": full,
            }
    record = {
        "model": args.model,
        "dataset": f"electronics-{args.source}",
        "phase": "cost" if args.cost else "tune",
        "trial": args.trial,
        "seed": args.seed,
        "config": cfg,
        "fixed": {
            "batch": BATCH[args.model],
            "max_len": MAX_LEN,
            "optimizer": "AdamW, linear warm-up then constant",
            "loss": "exact full-softmax CE, chunked fused forward-backward",
            "precision": "float32 with TF32 matmuls",
            **FIXED[args.model],
        },
        "data": scale.describe(d),
        **run,
        "test": None,
    }
    tag = f"trial{args.trial:02d}"
    if args.cost:
        out = RESULTS / "cost" / f"{args.model}_{args.source}_{tag}.json"
    else:
        ckpt = CHECKPOINTS / args.model / f"{tag}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        record["checkpoint"] = str(ckpt.relative_to(common.ROOT))
        out = RESULTS / "runs" / args.model / f"{tag}.json"
    record.update(
        wall_time_s=round(time.time() - t0, 1),
        gpu={
            "nvidia_smi_start": smi_before,
            "nvidia_smi_end": common.nvidia_smi(),
            "sampled": sampler.summary(),
        },
        environment=common.environment(),
        git=git,
        protocol=PROTOCOL_VERSION,
    )
    common.write_json(out, record)
    print(out, record["valid"]["all"])


if __name__ == "__main__":
    main()
