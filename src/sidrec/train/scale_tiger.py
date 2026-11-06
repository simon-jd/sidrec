"""Train and evaluate TIGER on Electronics with the R3 semantic IDs, as docs/protocol.md fixes.

    python -m sidrec.train.scale_tiger --trial 3                        # dev-subset tuning trial
    python -m sidrec.train.scale_tiger --trial 1 --source full --cost   # cost measurement

Semantic IDs: R3's Qwen3-Embedding-0.6B RQ-VAE codes for all 368,228 items, three levels of 256 codes and a
fourth code with as many values as the largest collision group (461). Training examples: every train position
with at least one earlier train item; the input is the user token and up to 20 preceding items, oldest first.
Evaluation: beam search of width 50 constrained to the semantic IDs of the whole catalog, with the 50 most
recent history items removed at the last level; the 50 items are ranked by their summed token
log-probabilities. ``--cost`` trains one epoch and times the evaluations like ``scale_idrec``.

    python -m sidrec.train.scale_tiger --trial 2 --final --seed 1      # R4b final run

``--final`` (R4b) trains the selected trial on the full train split with early stopping on the 100,000-row
validation sample, then evaluates the best epoch's weights once on all eligible test rows. That one beam pass
gives plain TIGER's metrics and those of TIGER with the paper's cold-start retrieval at ε = 0.1
(``sidrec.models.tiger_cold``). Long runs pause and resume as in ``scale_idrec``.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from sidrec.data import scale
from sidrec.eval.metrics import ranks_from_lists, split_metrics
from sidrec.models import tiger_cold
from sidrec.models.tiger import Layout, Trie, build_t5, constrained_beam_search
from sidrec.semantic_ids.catalog import file_sha256
from sidrec.train import common, resume
from sidrec.train.scale_idrec import (
    CHECKPOINTS_R4B,
    MAX_EPOCHS,
    PATIENCE,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_R4B,
    RESULTS_R4B,
    SAMPLE,
    WARMUP_SHARE,
    selected_trial,
)
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
COLD_START_EPS = 0.1  # TIGER's Fig. 5a setting, fixed by the R4b protocol entry


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
def evaluate(
    model,
    inp: Inputs,
    rows,
    users,
    cold,
    trie: Trie,
    layout: Layout,
    chunk: int = 512,
    eps: tuple = (),
    unseen: torch.Tensor | None = None,
    prefix_key: torch.Tensor | None = None,
) -> dict:
    """Plain TIGER's metrics; with ``eps``, also those of the cold-start retrieval at each ε, from the same
    beam.

    unseen: [N + 1] bool, items outside the train split; prefix_key: [N + 1] long, a key of (c1, c2, c3)."""
    model.eval()
    ranks, slots = [], 0
    cold_ranks = {e: [] for e in eps}
    stats = dict.fromkeys(
        ("rows_with_candidates", "candidates", "cold_target_prefix_in_beam", "cold_target_in_candidates"), 0
    )
    for s in range(0, len(rows), chunk):
        ex, u = rows.take(slice(s, s + chunk)), users[s : s + chunk]
        x = inp.encoder(ex, u)
        history = inp.history(ex)
        last = {} if eps else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            items, _ = constrained_beam_search(
                model, x, x > 0, trie, layout, beam=BEAM, exclude=history, last=last
            )
        slots += int((items < 0).sum())
        target = inp.items[ex.end]
        ranks.append(ranks_from_lists(items, target))
        if eps:
            cand, _ = tiger_cold.candidates(items, last, trie.tables[-1], unseen, history, BEAM)
            for e in eps:
                merged = tiger_cold.merge(items, unseen[items.clamp(min=0)], cand, e)
                cold_ranks[e].append(ranks_from_lists(merged, target))
            n = (cand >= 0).sum(1)
            c = cold[s : s + chunk]
            beam_keys = prefix_key[items.clamp(min=0)].masked_fill(items < 0, -1)
            stats["rows_with_candidates"] += int((n > 0).sum())
            stats["candidates"] += int(n.sum())
            prefix_hit = (beam_keys == prefix_key[target][:, None]).any(1)
            stats["cold_target_prefix_in_beam"] += int((prefix_hit & c).sum())
            stats["cold_target_in_candidates"] += int(((cand == target[:, None]).any(1) & c).sum())
    out = split_metrics(torch.cat(ranks), cold)
    out["empty_slots"] = slots
    if eps:
        out["cold_start"] = {str(e): split_metrics(torch.cat(cold_ranks[e]), cold) for e in eps}
        out["cold_start_stats"] = stats | {"rows": len(rows), "cold_rows": int(cold.sum())}
    return out


def cold_start_inputs(d: dict, codes: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    """unseen [N + 1]: items outside this source's train split (False for padding); prefix_key [N + 1]."""
    unseen = torch.as_tensor(~d["train_item"], device=device)
    unseen[0] = False
    key = torch.as_tensor((codes[:, 0] * 256 + codes[:, 1]) * 256 + codes[:, 2], device=device)
    return unseen, key


def train(
    cfg: dict,
    seed: int,
    d: dict,
    device: str,
    max_epochs: int,
    sample: int | None,
    state_path=None,
    budget_s: float | None = None,
):
    """Returns (model, run record fields, (inp, trie, layout, codes), state, paused)."""
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

    def train_epoch() -> float:
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
        return total.item() / n

    def row(m: dict) -> dict:
        return {
            "valid_ndcg@10": m["all"]["ndcg@10"],
            "valid_recall@10": m["all"]["recall@10"],
            "valid_recall@50": m["all"]["recall@50"],
            "valid_cold_recall@50": (m["cold"] or {}).get("recall@50"),
        }

    st, paused = resume.run_epochs(
        model,
        opt,
        sched,
        gen,
        device,
        train_epoch,
        lambda: evaluate(model, inp, va, va_users, va_cold, trie, layout),
        row,
        max_epochs,
        PATIENCE,
        state_path,
        budget_s,
    )
    history = st["history"]
    run = {
        "epochs_run": len(history),
        "best_epoch": st["best_epoch"],
        "train_examples": n,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup,
        "level_sizes": list(layout.level_sizes),
        "vocab_size": layout.vocab_size,
        "valid_rows": len(va),
        "valid": st["best_valid"],
        "history": history,
        "params": sum(p.numel() for p in model.parameters()),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None,
    }
    return model, run, (inp, trie, layout, codes), st, paused


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trial", type=int, required=True, help="1..12, the protocol's shared trial list")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", choices=["dev", "full"], default="dev")
    parser.add_argument("--cost", action="store_true", help="one epoch and timed evaluations, no checkpoint")
    parser.add_argument("--final", action="store_true", help="R4b: full split, one test evaluation")
    parser.add_argument("--budget-hours", type=float, default=1.0, help="pause after about this long")
    args = parser.parse_args()
    if args.final:
        args.source = "full"
        if args.trial != selected_trial("tiger"):
            raise SystemExit(f"--final needs the selected trial, {selected_trial('tiger')}")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    cfg = shared_trials()[args.trial - 1]
    if file_sha256(SID) != SID_SHA256:
        raise SystemExit("semantic IDs differ from the R3 file")
    tag = f"seed{args.seed}" if args.final else f"trial{args.trial:02d}"
    ckpt = (CHECKPOINTS_R4B if args.final else CHECKPOINTS) / "tiger" / f"{tag}.pt"
    state_path = ckpt.with_name(f"{tag}.state.pt") if args.final else None
    t0 = time.time()  # the GPU lock is held from here, data loading included
    d = scale.load(args.source)
    smi_before = common.nvidia_smi()
    git = resume.code_state()
    processes = resume.read_processes(state_path) if args.final else []
    with common.SmiSampler() as sampler:
        model, run, (inp, trie, layout, codes), st, paused = train(
            cfg,
            args.seed,
            d,
            device,
            1 if args.cost else MAX_EPOCHS,
            SAMPLE,
            state_path,
            args.budget_hours * 3600 if args.final else None,
        )
        test = None
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
        elif args.final and not paused:  # the test split, used once: plain TIGER and its cold-start retrieval
            rows, users, cold, _ = scale.eval_rows(d, "test", None, device)
            unseen, key = cold_start_inputs(d, codes, device)
            torch.cuda.synchronize()
            t = time.time()
            eps = (COLD_START_EPS,)
            test = evaluate(
                model, inp, rows, users, cold, trie, layout, eps=eps, unseen=unseen, prefix_key=key
            )
            test["seconds"] = round(time.time() - t, 1)
    if args.final:
        processes.append(resume.process_entry(st, t0, smi_before, sampler, git))
        resume.write_processes(state_path, processes)
        if paused:
            print(f"paused after epoch {st['epoch']}", flush=True)
            raise SystemExit(resume.PAUSED)
    record = {
        "model": "tiger",
        "encoder": "qwen3-emb-0.6b",
        "semantic_ids": {"path": str(SID.relative_to(common.ROOT)), "sha256": SID_SHA256},
        "dataset": f"electronics-{args.source}",
        "phase": "cost" if args.cost else ("final" if args.final else "tune"),
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
        }
        | ({"cold_start_eps": COLD_START_EPS, "cold_start_prefix": "c1, c2, c3"} if args.final else {}),
        "data": scale.describe(d),
        **run,
        "test": test,
    }
    if args.cost:
        out = RESULTS / "cost" / f"tiger_{args.source}_{tag}.json"
    else:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        record["checkpoint"] = str(ckpt.relative_to(common.ROOT))
        out = (RESULTS_R4B if args.final else RESULTS) / "runs" / "tiger" / f"{tag}.json"
    if args.final:
        record.update(
            wall_time_s=round(sum(p["wall_time_s"] for p in processes), 1),
            processes=processes,
            environment=common.environment(),
            protocol=PROTOCOL_VERSION_R4B,
        )
    else:
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
    if state_path is not None:
        state_path.unlink(missing_ok=True)
        resume.processes_path(state_path).unlink(missing_ok=True)
    print(out, record["valid"]["all"], (test or {}).get("all"))


if __name__ == "__main__":
    main()
