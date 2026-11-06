"""Train and evaluate TIGER on Electronics with the R3 semantic IDs, as docs/protocol.md fixes.

    python -m sidrec.train.scale_tiger --trial 3                        # dev-subset tuning trial
    python -m sidrec.train.scale_tiger --trial 1 --source full --cost   # cost measurement

Semantic IDs: R3's Qwen3-Embedding-0.6B RQ-VAE codes for all 368,228 items, three levels of 256 codes and a
fourth code with as many values as the largest collision group (461). Training examples: every train position
with at least one earlier train item; the input is the user token and up to 20 preceding items, oldest first.
Evaluation: beam search of width 50 constrained to the semantic IDs of the whole catalog, with the 50 most
recent history items removed at the last level; the 50 items are ranked by their summed token
log-probabilities. ``--cost`` trains one epoch and times the evaluations like ``scale_idrec``.
"""

from __future__ import annotations

import argparse
import copy
import math
import time

import numpy as np
import torch

from sidrec.data import scale
from sidrec.eval.metrics import ranks_from_lists, split_metrics
from sidrec.models.tiger import Layout, Trie, build_t5, constrained_beam_search
from sidrec.semantic_ids.catalog import file_sha256
from sidrec.train import common
from sidrec.train.scale_idrec import MAX_EPOCHS, PATIENCE, PROTOCOL_VERSION, SAMPLE, WARMUP_SHARE
from sidrec.tuning import shared_trials

HISTORY = 20
MASK_HISTORY = 50
BATCH = 256
BEAM = 50
CLIP = 1.0
SID = common.ROOT / "data" / "processed" / "electronics" / "sid" / "qwen3-emb-0.6b.npy"
SID_SHA256 = "d870373c68ac2da8f04a45281259dd7a905969c35892cc94c61fa8b142bae01f"  # results/r3/SHA256SUMS
FIXED = {
    "heads": 6,
    "head_dim": 64,
    "d_ff": 1024,
    "activation": "relu",
    "user_buckets": 2000,
    "grad_clip": CLIP,
}
RESULTS = common.ROOT / "results" / "r4a"
CHECKPOINTS = common.ROOT / "data" / "runs" / "r4a"


def load_ids(path=SID) -> tuple[np.ndarray, Layout]:
    """[N + 1, 4] codes with row 0 for padding, and the token layout; the last level gets max + 1 codes."""
    codes = np.load(path)
    codes = np.concatenate([np.zeros((1, 4), dtype=codes.dtype), codes]).astype(np.int64)
    return codes, Layout((256, 256, 256, int(codes[1:, 3].max()) + 1))


class Inputs:
    """TIGER encoder inputs, labels and history masks cut from the flat timelines on the GPU."""

    def __init__(self, d: dict, tokens: np.ndarray, layout: Layout, device):
        self.items = torch.as_tensor(d["items"].astype(np.int64), device=device)
        self.tokens = torch.as_tensor(tokens, device=device)
        self.user_tok = layout.user_offset + torch.as_tensor(d["user_bucket"].astype(np.int64), device=device)

    def encoder(self, ex: scale.Examples, users: torch.Tensor) -> torch.Tensor:
        idx, valid = scale.window(ex.end, ex.start, HISTORY, len(self.items), left_pad=False)
        hist = self.items[idx].masked_fill(~valid, 0)
        tok = self.tokens[hist].reshape(len(hist), -1)  # padding item 0 has four padding tokens
        return torch.cat([self.user_tok[users][:, None], tok], dim=1)

    def labels(self, ex: scale.Examples) -> torch.Tensor:
        return self.tokens[self.items[ex.end]]

    def history(self, ex: scale.Examples) -> torch.Tensor:
        """The most recent 50 items before each target, sorted ascending (0 pads)."""
        idx, valid = scale.window(ex.end, ex.start, MASK_HISTORY, len(self.items))
        return self.items[idx].masked_fill(~valid, 0).sort(dim=1).values


@torch.no_grad()
def evaluate(model, inp: Inputs, rows, users, cold, trie: Trie, layout: Layout, chunk: int = 512) -> dict:
    model.eval()
    ranks, slots = [], 0
    for s in range(0, len(rows), chunk):
        ex, u = rows.take(slice(s, s + chunk)), users[s : s + chunk]
        x = inp.encoder(ex, u)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            items, _ = constrained_beam_search(
                model, x, x > 0, trie, layout, beam=BEAM, exclude=inp.history(ex)
            )
        slots += int((items < 0).sum())
        ranks.append(ranks_from_lists(items, inp.items[ex.end]))
    out = split_metrics(torch.cat(ranks), cold)
    out["empty_slots"] = slots
    return out


def train(cfg: dict, seed: int, d: dict, device: str, max_epochs: int, sample: int | None):
    common.seed_all(seed)
    codes, layout = load_ids()
    inp = Inputs(d, layout.item_tokens(codes), layout, device)
    trie = Trie(codes, layout.level_sizes, device)
    ex, ex_users = scale.tiger_train_examples(d, device)
    va, va_users, va_cold, _ = scale.eval_rows(d, "valid", sample, device)
    model = build_t5(layout, cfg["width"], cfg["layers"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = len(ex)
    steps_per_epoch = math.ceil(n / BATCH)
    warmup = max(1, int(WARMUP_SHARE * MAX_EPOCHS * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, common.warmup_schedule(warmup, "inverse_sqrt"))
    gen = torch.Generator(device=device).manual_seed(seed)
    best, best_epoch, best_state, best_valid, history = -1.0, 0, None, None, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        torch.cuda.synchronize()
        t0 = time.time()
        order = torch.randperm(n, device=device, generator=gen)
        total = torch.zeros((), device=device)
        for s in range(0, n, BATCH):
            idx = order[s : s + BATCH]
            b = ex.take(idx)
            x = inp.encoder(b, ex_users[idx])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, x > 0, inp.labels(b))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            sched.step()
            total += loss.detach() * len(idx)
        torch.cuda.synchronize()
        train_s = time.time() - t0
        metrics = evaluate(model, inp, va, va_users, va_cold, trie, layout)
        eval_s = time.time() - t0 - train_s
        history.append(
            {
                "epoch": epoch,
                "loss": round(total.item() / n, 5),
                "valid_ndcg@10": metrics["all"]["ndcg@10"],
                "valid_recall@10": metrics["all"]["recall@10"],
                "valid_recall@50": metrics["all"]["recall@50"],
                "valid_cold_recall@50": (metrics["cold"] or {}).get("recall@50"),
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
        "train_examples": n,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup,
        "level_sizes": list(layout.level_sizes),
        "vocab_size": layout.vocab_size,
        "valid_rows": len(va),
        "valid": best_valid,
        "history": history,
        "params": sum(p.numel() for p in model.parameters()),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    return model, run, (inp, trie, layout)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trial", type=int, required=True, help="1..12, the protocol's shared trial list")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", choices=["dev", "full"], default="dev")
    parser.add_argument("--cost", action="store_true", help="one epoch and timed evaluations, no checkpoint")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    cfg = shared_trials()[args.trial - 1]
    if file_sha256(SID) != SID_SHA256:
        raise SystemExit("semantic IDs differ from the R3 file")
    d = scale.load(args.source)
    t0 = time.time()
    smi_before = common.nvidia_smi()
    git = common.git_state()
    with common.SmiSampler() as sampler:
        model, run, (inp, trie, layout) = train(
            cfg, args.seed, d, device, 1 if args.cost else MAX_EPOCHS, SAMPLE
        )
        if args.cost:
            rows, users, cold, _ = scale.eval_rows(d, "valid", None, device)
            torch.cuda.synchronize()
            t = time.time()
            full = evaluate(model, inp, rows, users, cold, trie, layout)
            run["full_valid_eval"] = {
                "rows": len(rows),
                "seconds": round(time.time() - t, 1),
                "metrics": full,
            }
    record = {
        "model": "tiger",
        "encoder": "qwen3-emb-0.6b",
        "semantic_ids": {"path": str(SID.relative_to(common.ROOT)), "sha256": SID_SHA256},
        "dataset": f"electronics-{args.source}",
        "phase": "cost" if args.cost else "tune",
        "trial": args.trial,
        "seed": args.seed,
        "config": cfg,
        "fixed": {
            "batch": BATCH,
            "history": HISTORY,
            "beam": BEAM,
            "history_mask": MASK_HISTORY,
            "optimizer": "AdamW, linear warm-up then inverse-sqrt",
            "precision": "bfloat16 autocast",
            **FIXED,
        },
        "data": scale.describe(d),
        **run,
        "test": None,
    }
    tag = f"trial{args.trial:02d}"
    if args.cost:
        out = RESULTS / "cost" / f"tiger_{args.source}_{tag}.json"
    else:
        ckpt = CHECKPOINTS / "tiger" / f"{tag}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        record["checkpoint"] = str(ckpt.relative_to(common.ROOT))
        out = RESULTS / "runs" / "tiger" / f"{tag}.json"
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
