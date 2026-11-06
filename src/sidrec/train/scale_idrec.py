"""Train and evaluate the item-ID models (SASRec, HSTU-style) and HSTU-SID on Electronics, as docs/protocol.md
fixes.

    python -m sidrec.train.scale_idrec --model hstu --trial 3                       # dev-subset tuning trial
    python -m sidrec.train.scale_idrec --model hstu --trial 1 --source full --cost  # cost measurement
    python -m sidrec.train.scale_idrec --model hstu --trial 1 --final --seed 2      # R4b final run

Tuning trials (seed 0) train on the dev subset's train split and evaluate on its validation rows every epoch;
they compute no test metrics. ``--cost`` trains one epoch, evaluates the 100,000-row validation sample, then
times one evaluation of every eligible validation row; it writes a cost record and no checkpoint. ``--final``
trains the selected trial on the full train split with early stopping on the 100,000-row validation sample,
then evaluates the best epoch's weights once on all eligible test rows.

Models:
- ``sasrec``, ``hstu``: item IDs. Loss: exact full-softmax cross-entropy over the V train items at every
  non-padding position of each user's training window, computed in row chunks by ``sidrec.train.fused_ce``.
  Ranking: every catalog item; items outside the train split rank below every train item, so a cold target is
  a miss.
- ``hstu-notime`` (R4b): the HSTU-style model without the time part of its relative attention bias.
- ``hstu-sid`` (R4b): the HSTU-style model whose item embedding adds semantic-ID prefix embeddings
  (``sidrec.models.hstu_sid``). Same loss over the train items; it ranks every catalog item by its score.

A run longer than ``--budget-hours`` saves its state after every epoch and exits with code 75; running the
same command again resumes it (``sidrec.train.resume``).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from sidrec.data import scale
from sidrec.eval.metrics import MISS, split_metrics, target_ranks
from sidrec.models.hstu import HSTU
from sidrec.models.hstu_sid import HSTUSID, prefix_rows
from sidrec.models.sasrec import SASRec
from sidrec.train import common, resume
from sidrec.train.fused_ce import fused_linear_cross_entropy
from sidrec.tuning import shared_trials

MAX_LEN = 50
MAX_EPOCHS = 30
PATIENCE = 3
WARMUP_SHARE = 0.05
SAMPLE = 100_000
BATCH = {"sasrec": 256, "hstu": 128, "hstu-notime": 128, "hstu-sid": 128}  # R2 values, dev and full split
HSTU_FIXED = {"heads": 4, "head_dim": "width / 4", "rab_time_buckets": 128, "temperature": 0.05}
FIXED = {
    "sasrec": {"heads": 2, "ffn": "4 x width, GELU", "norm": "post-LN", "scores": "dot product, tied table"},
    "hstu": HSTU_FIXED,
    "hstu-notime": HSTU_FIXED | {"rab_time_buckets": 0, "time_bias": "none, position bias only"},
    "hstu-sid": HSTU_FIXED
    | {
        "item_embedding": "ID row (zero outside train) + embeddings of prefixes (c1), (c1,c2), (c1,c2,c3)",
        "prefix_tables": "one row per prefix of a train item, zero init; other prefixes share a zero row",
        "ranking": "every catalog item by its score",
    },
}
TUNED_IN = {"sasrec": "r4a", "hstu": "r4a", "hstu-notime": "r4a", "hstu-sid": "r4b", "tiger": "r4a"}
TUNED_AS = {"hstu-notime": "hstu"}  # the ablation reuses the HSTU-style model's selected trial
RESULTS = common.ROOT / "results" / "r4a"
CHECKPOINTS = common.ROOT / "data" / "runs" / "r4a"
RESULTS_R4B = common.ROOT / "results" / "r4b"
CHECKPOINTS_R4B = common.ROOT / "data" / "runs" / "r4b"
PROTOCOL_VERSION = "2026-11-05 with the R2, R3 and R4a clarifications"
PROTOCOL_VERSION_R4B = "2026-11-05 with the R2, R3, R4a and R4b clarifications"


def sid_tables(d: dict) -> dict:
    """HSTU-SID's lookup tables for one source: the item-ID map and the prefix rows of every catalog item."""
    from sidrec.train.scale_tiger import load_ids

    vocab, v = scale.id_vocab(d)
    codes, _ = load_ids()
    prefixes, sizes = prefix_rows(codes, d["train_item"])
    return {"vocab": vocab, "v": v, "prefixes": prefixes, "sizes": sizes}


def build(model: str, cfg: dict, v: int, sid: dict | None = None) -> torch.nn.Module:
    """Item table rows: 0 padding, 1..v the train items, v + 1 the shared out-of-vocabulary row (zero).

    HSTU-SID takes catalog indices instead and needs ``sid`` (``sid_tables``)."""
    if model == "sasrec":
        m = SASRec(v + 1, cfg["width"], cfg["layers"], cfg["dropout"], max_len=MAX_LEN)
    elif model == "hstu-sid":
        return HSTUSID(
            torch.as_tensor(sid["vocab"]),
            torch.as_tensor(sid["prefixes"]),
            sid["sizes"],
            v,
            cfg["width"],
            cfg["layers"],
            cfg["dropout"],
            max_len=MAX_LEN,
        )
    else:
        m = HSTU(
            v + 1, cfg["width"], cfg["layers"], cfg["dropout"], max_len=MAX_LEN, time_bias=model == "hstu"
        )
    with torch.no_grad():
        m.item_emb.weight[v + 1].zero_()
    return m


def output_table(model: torch.nn.Module, v: int) -> torch.Tensor:
    """[v, d] scoring table of the train items: SASRec's tied item table; the HSTU-style model's
    L2-normalized table divided by its temperature; HSTU-SID's summed embeddings, normalized the same way."""
    if isinstance(model, HSTUSID):
        return model.train_table()
    w = model.item_emb.weight[1 : v + 1]
    if isinstance(model, HSTU):
        return F.normalize(w, dim=-1) / model.temperature
    return w


@torch.no_grad()
def evaluate(model, flat: scale.Flat, rows: scale.Examples, cold, vocab, v: int, chunk: int = 1024) -> dict:
    model.eval()
    if isinstance(model, HSTUSID):  # catalog indices in, every catalog item scored
        table = model.catalog_table()
        ident = torch.arange(len(vocab), device=vocab.device)
    else:
        table = output_table(model, v)
    ranks = []
    for s in range(0, len(rows), chunk):
        ex = rows.take(slice(s, s + chunk))
        if isinstance(model, HSTUSID):
            b = flat.id_inputs(ex, ident, MAX_LEN, train=False)
            states = model.encode(b["inputs"], input_ts=b["input_ts"], query_ts=b["query_ts"])[:, -1]
            ranks.append(catalog_ranks(states @ table.T, b["history"], b["target"]))
        else:
            b = flat.id_inputs(ex, vocab, MAX_LEN, train=False)
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


def catalog_ranks(scores: torch.Tensor, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Protocol ranks from scores over every catalog item ([B, N], column j is catalog item j + 1); history
    and target hold catalog indices (0 padding)."""
    exclude = torch.zeros(len(history), scores.shape[1] + 1, dtype=torch.bool, device=history.device)
    exclude.scatter_(1, history, True)
    return target_ranks(scores, target - 1, exclude[:, 1:])


def train(
    name: str,
    cfg: dict,
    seed: int,
    d: dict,
    device: str,
    max_epochs: int,
    sample: int | None,
    state_path=None,
    budget_s: float | None = None,
):
    """Returns (model, run record fields, (flat, vocab, v), paused)."""
    common.seed_all(seed)
    flat = scale.Flat(d, device)
    vocab_np, v = scale.id_vocab(d)
    vocab = torch.as_tensor(vocab_np, device=device)
    ex = scale.id_train_examples(d, device)
    va, _, va_cold, _ = scale.eval_rows(d, "valid", sample, device)
    sid = sid_tables(d) if name == "hstu-sid" else None
    model = build(name, cfg, v, sid).to(device)
    # HSTU-SID reads catalog indices; its loss targets are still the train items' rows.
    inputs_map = torch.arange(len(vocab), device=device) if sid is not None else vocab
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    batch, n = BATCH[name], len(ex)
    steps_per_epoch = math.ceil(n / batch)
    warmup = max(1, int(WARMUP_SHARE * MAX_EPOCHS * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, common.warmup_schedule(warmup, "constant"))
    gen = torch.Generator(device=device).manual_seed(seed)

    def train_epoch() -> float:
        order = torch.randperm(n, device=device, generator=gen)
        total = torch.zeros((), device=device)
        count = torch.zeros((), device=device, dtype=torch.long)
        for s in range(0, n, batch):
            b = flat.id_inputs(ex.take(order[s : s + batch]), inputs_map, MAX_LEN, train=True)
            states = model.encode(b["inputs"], input_ts=b["input_ts"], query_ts=b["query_ts"])
            keep = b["targets"] > 0
            targets = b["targets"][keep] if sid is None else vocab[b["targets"][keep]]
            loss_sum = fused_linear_cross_entropy(states[keep], output_table(model, v), targets - 1)
            k = keep.sum()
            opt.zero_grad(set_to_none=True)
            (loss_sum / k).backward()
            opt.step()
            sched.step()
            total += loss_sum.detach()
            count += k
        return total.item() / count.item()

    def row(m: dict) -> dict:
        out = {
            "valid_ndcg@10": m["all"]["ndcg@10"],
            "valid_recall@10": m["all"]["recall@10"],
            "valid_recall@50": m["all"]["recall@50"],
        }
        if sid is not None:
            out["valid_cold_recall@50"] = (m["cold"] or {}).get("recall@50")
        return out

    st, paused = resume.run_epochs(
        model,
        opt,
        sched,
        gen,
        device,
        train_epoch,
        lambda: evaluate(model, flat, va, va_cold, vocab, v),
        row,
        max_epochs,
        PATIENCE,
        state_path,
        budget_s,
    )
    run = {
        "epochs_run": len(st["history"]),
        "best_epoch": st["best_epoch"],
        "train_windows": n,
        "train_targets": int((ex.end - ex.start).clamp(max=MAX_LEN).sum()),
        "vocab_items": v,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup,
        "valid_rows": len(va),
        "valid": st["best_valid"],
        "history": st["history"],
        "params": sum(p.numel() for p in model.parameters()),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None,
    }
    if sid is not None:
        run["prefix_table_rows"] = sid["sizes"]
    return model, run, (flat, vocab, v), st, paused


def selected_trial(model: str) -> int:
    """The tuning trial with the best validation NDCG@10 (protocol, section 3); ties go to the lower trial."""
    tuned = TUNED_AS.get(model, model)
    runs = []
    folder = common.ROOT / "results" / TUNED_IN[model] / "runs" / tuned
    for path in sorted(folder.glob("trial[0-9][0-9].json")):
        r = json.loads(path.read_text())
        runs.append((r["valid"]["all"]["ndcg@10"], -r["trial"], r["trial"]))
    if len(runs) != 12:
        raise SystemExit(f"{tuned}: {len(runs)} tuning trials found, expected 12")
    return max(runs)[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(BATCH), required=True)
    parser.add_argument("--trial", type=int, required=True, help="1..12, the protocol's shared trial list")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", choices=["dev", "full"], default="dev")
    parser.add_argument("--cost", action="store_true", help="one epoch and timed evaluations, no checkpoint")
    parser.add_argument("--final", action="store_true", help="R4b: full split, one test evaluation")
    parser.add_argument("--budget-hours", type=float, default=1.0, help="pause after about this long")
    args = parser.parse_args()
    if args.final:
        args.source = "full"
        if args.trial != selected_trial(args.model):
            raise SystemExit(f"--final needs the selected trial, {selected_trial(args.model)}")
    elif args.model == "hstu-notime" or (args.model == "hstu-sid" and (args.cost or args.source != "dev")):
        raise SystemExit(f"{args.model}: only --final runs (and, for hstu-sid, dev-subset tuning trials)")
    r4b = args.final or args.model == "hstu-sid"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    cfg = shared_trials()[args.trial - 1]
    tag = f"seed{args.seed}" if args.final else f"trial{args.trial:02d}"
    ckpt = (CHECKPOINTS_R4B if r4b else CHECKPOINTS) / args.model / f"{tag}.pt"
    state_path = ckpt.with_name(f"{tag}.state.pt") if r4b else None
    t0 = time.time()  # the GPU lock is held from here, data loading included
    d = scale.load(args.source)
    smi_before = common.nvidia_smi()
    git = resume.code_state()  # the code this process started with
    processes = resume.read_processes(state_path) if r4b else []
    with common.SmiSampler() as sampler:
        model, run, (flat, vocab, v), st, paused = train(
            args.model,
            cfg,
            args.seed,
            d,
            device,
            1 if args.cost else MAX_EPOCHS,
            SAMPLE,
            state_path,
            args.budget_hours * 3600 if r4b else None,
        )
        test = None
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
        elif args.final and not paused:  # the test split, used once, on the best epoch's weights
            rows, _, cold, _ = scale.eval_rows(d, "test", None, device)
            torch.cuda.synchronize()
            t = time.time()
            test = evaluate(model, flat, rows, cold, vocab, v)
            test["seconds"] = round(time.time() - t, 1)
    if r4b:
        processes.append(resume.process_entry(st, t0, smi_before, sampler, git))
        resume.write_processes(state_path, processes)
        if paused:
            print(f"paused after epoch {st['epoch']}", flush=True)
            raise SystemExit(resume.PAUSED)
    record = {
        "model": args.model,
        "dataset": f"electronics-{args.source}",
        "phase": "cost" if args.cost else ("final" if args.final else "tune"),
        "trial": args.trial,
        "seed": args.seed,
        "config": cfg,
        "fixed": {
            "batch": BATCH[args.model],
            "max_len": MAX_LEN,
            "optimizer": "AdamW, linear warm-up then constant",
            "loss": "exact full-softmax CE over the train items, chunked fused forward-backward",
            "precision": "float32 with TF32 matmuls",
            **FIXED[args.model],
        },
        "data": scale.describe(d),
        **run,
        "test": test,
    }
    if args.cost:
        out = RESULTS / "cost" / f"{args.model}_{args.source}_{tag}.json"
    else:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        record["checkpoint"] = str(ckpt.relative_to(common.ROOT))
        out = (RESULTS_R4B if r4b else RESULTS) / "runs" / args.model / f"{tag}.json"
    if r4b:
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
