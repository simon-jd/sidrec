"""How often TIGER's beam emits cold items: a diagnostic of the selected dev-subset trial (validation only).

    uv run python scripts/r4a_tiger_cold_share.py --trial N

Loads data/runs/r4a/tiger/trialNN.pt, runs the protocol's width-50 beam over the dev subset's 71,610
eligible validation rows, and counts the emitted items that are outside the dev subset's train split (the
items its cold rows target). Writes results/r4a/tiger_cold_share.json. Nothing is selected from it.
"""

from __future__ import annotations

import argparse
import time

import torch

from sidrec.data import scale
from sidrec.models.tiger import Trie, build_t5, constrained_beam_search
from sidrec.train import common
from sidrec.train.scale_tiger import BEAM, CHECKPOINTS, RESULTS, Inputs, load_ids
from sidrec.tuning import shared_trials


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()
    device = "cuda"
    cfg = shared_trials()[args.trial - 1]
    d = scale.load("dev")
    codes, layout = load_ids()
    inp = Inputs(d, layout.item_tokens(codes), layout, device)
    trie = Trie(codes, layout.level_sizes, device)
    model = build_t5(layout, cfg["width"], cfg["layers"], cfg["dropout"]).to(device).eval()
    ckpt = CHECKPOINTS / "tiger" / f"trial{args.trial:02d}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    rows, users, cold, _ = scale.eval_rows(d, "valid", None, device)
    train_item = torch.as_tensor(d["train_item"], device=device)
    emitted = cold_emitted = 0
    rows_with_cold = 0
    t = time.time()
    for s in range(0, len(rows), 512):
        ex, u = rows.take(slice(s, s + 512)), users[s : s + 512]
        x = inp.encoder(ex, u)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            items, _ = constrained_beam_search(
                model, x, x > 0, trie, layout, beam=BEAM, exclude=inp.history(ex)
            )
        valid = items > 0
        is_cold = valid & ~train_item[items.clamp(min=0)]
        emitted += int(valid.sum())
        cold_emitted += int(is_cold.sum())
        rows_with_cold += int(is_cold.any(1).sum())
    out = {
        "trial": args.trial,
        "checkpoint": str(ckpt.relative_to(common.ROOT)),
        "rows": len(rows),
        "cold_target_rows": int(cold.sum()),
        "emitted_items": emitted,
        "emitted_cold_items": cold_emitted,
        "cold_share_of_emitted": cold_emitted / emitted,
        "rows_with_any_cold_item": rows_with_cold,
        "catalog_items_outside_dev_train": int((~d["train_item"][1:]).sum()),
        "seconds": round(time.time() - t, 1),
        "git": common.git_state(),
        "environment": common.environment(),
    }
    out["catalog_share_outside_dev_train"] = out["catalog_items_outside_dev_train"] / int(d["n_items"])
    common.write_json(RESULTS / "tiger_cold_share.json", out)
    print(
        {
            k: out[k]
            for k in ("cold_share_of_emitted", "rows_with_any_cold_item", "catalog_share_outside_dev_train")
        }
    )


if __name__ == "__main__":
    main()
