"""How much of HSTU-SID's ranking items outside the train split take: a diagnostic, validation only.

    uv run python scripts/r4b_hstu_sid_flood.py --trial N      # a dev-subset tuning trial
    uv run python scripts/r4b_hstu_sid_flood.py --seed S       # a final run, on the full split

Loads an HSTU-SID checkpoint and scores the validation rows the run evaluated every epoch: all 71,610 eligible
dev rows for a trial, the 100,000-row sample for a final run. For each it reports
- the protocol's ranking over the whole catalog (what the run records);
- the share of the top 10 and top 50 slots that items outside the train split take, against their share of the
  catalog;
- the same model ranked among the train items only, the item-ID models' ranking, on which a cold target is a
  miss.
The test split is not read, and nothing is selected from this. The record goes to
results/r4b/hstu_sid_flood/<tag>.json.
"""

from __future__ import annotations

import argparse
import time

import torch

from sidrec.data import scale
from sidrec.eval.metrics import split_metrics
from sidrec.train import common, resume
from sidrec.train.scale_idrec import (
    CHECKPOINTS_R4B,
    MAX_LEN,
    RESULTS_R4B,
    SAMPLE,
    build,
    catalog_ranks,
    selected_trial,
    sid_tables,
    vocab_ranks,
)
from sidrec.tuning import shared_trials


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--trial", type=int)
    which.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    final = args.seed is not None
    trial = selected_trial("hstu-sid") if final else args.trial
    tag = f"seed{args.seed}" if final else f"trial{trial:02d}"
    source, sample = ("full", SAMPLE) if final else ("dev", None)
    t0 = time.time()
    d = scale.load(source)
    sid = sid_tables(d)
    v = sid["v"]
    model = build("hstu-sid", shared_trials()[trial - 1], v, sid).to(device)
    ckpt = CHECKPOINTS_R4B / "hstu-sid" / f"{tag}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    flat = scale.Flat(d, device)
    vocab = torch.as_tensor(sid["vocab"], device=device)
    ident = torch.arange(len(vocab), device=device)
    unseen = ~torch.as_tensor(d["train_item"], device=device)
    rows, _, cold, _ = scale.eval_rows(d, "valid", sample, device)
    train_table, catalog_table = model.train_table(), model.catalog_table()
    r_catalog, r_train, top10, top50 = [], [], [], []
    for s in range(0, len(rows), 1024):
        b = flat.id_inputs(rows.take(slice(s, s + 1024)), ident, MAX_LEN, train=False)
        states = model.encode(b["inputs"], input_ts=b["input_ts"], query_ts=b["query_ts"])[:, -1]
        scores = states @ catalog_table.T
        r_catalog.append(catalog_ranks(scores, b["history"], b["target"]))
        r_train.append(vocab_ranks(states @ train_table.T, vocab[b["history"]], vocab[b["target"]], v))
        history = torch.zeros(len(scores), scores.shape[1] + 1, dtype=torch.bool, device=device)
        history.scatter_(1, b["history"], True)
        top = scores.masked_fill(history[:, 1:], -torch.inf).topk(50, dim=1).indices + 1  # history not ranked
        top10.append(unseen[top[:, :10]].float().mean(1))
        top50.append(unseen[top].float().mean(1))
    out = {
        "model": "hstu-sid",
        "analysis": "share of the ranking taken by items outside the train split; validation only",
        "dataset": f"electronics-{source}",
        "trial": trial,
        "seed": args.seed if final else 0,
        "checkpoint": str(ckpt.relative_to(common.ROOT)),
        "valid_rows": len(rows),
        "catalog_share_outside_train": float(unseen[1:].float().mean()),
        "top10_share_outside_train": float(torch.cat(top10).mean()),
        "top50_share_outside_train": float(torch.cat(top50).mean()),
        "ranked_over_catalog": split_metrics(torch.cat(r_catalog), cold),
        "ranked_among_train_items": split_metrics(torch.cat(r_train), cold),
        "wall_time_s": round(time.time() - t0, 1),
        "git": resume.code_state(),
        "environment": common.environment(),
    }
    common.write_json(RESULTS_R4B / "hstu_sid_flood" / f"{tag}.json", out)
    print(tag, {k: round(out[k], 4) for k in ("top10_share_outside_train", "top50_share_outside_train")})


if __name__ == "__main__":
    main()
