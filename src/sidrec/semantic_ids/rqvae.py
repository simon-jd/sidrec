"""RQ-VAE semantic IDs as TIGER (Rajput et al., NeurIPS 2023), section 4, describes them.

    python -m sidrec.semantic_ids.rqvae --encoder st5-xxl

Encoder 512-256-128 (ReLU) to a 32-dim latent, 3 levels of residual quantization with 256 codes each, and a
mirrored decoder. The loss is the paper's,

    ||x - x_hat||^2 + sum_l (||sg[r_l] - e_l||^2 + beta ||r_l - sg[e_l]||^2),   beta = 0.25,

with a straight-through gradient; batch 1024, 20,000 epochs, each level's codebook initialised by k-means on
the first batch. The optimizer is Adam (AdamW without weight decay) at 1e-3, not the paper's Adagrad at 0.4:
every Adagrad variant we tried either diverged or left a decoder that outputs the mean embedding (protocol,
Changes). Items that share all three codes get a fourth
code 0, 1, 2, ... in item-index order; every other item gets 0.

It trains on the items that occur in some user's training part and assigns codes to every item. Codes go to
data/processed/beauty/sid/<encoder>.npy ([N + 1, 4], row 0 unused); the statistics to
results/r2/semantic_ids/<encoder>.json.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import torch
from torch import nn

from sidrec.data import loo
from sidrec.semantic_ids.embed import EMB_DIR, ENCODERS
from sidrec.train import common

SID_DIR = common.ROOT / "data" / "processed" / "beauty" / "sid"


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def kmeans(x: torch.Tensor, k: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=x.device).manual_seed(seed)
    centers = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centers).argmin(1)
        sums = torch.zeros_like(centers).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).float()[:, None]
        centers = torch.where(counts > 0, sums / counts.clamp(min=1), centers)
    return centers


class RQVAE(nn.Module):
    def __init__(
        self, in_dim: int, hidden=(512, 256, 128), latent: int = 32, levels: int = 3, codes: int = 256
    ):
        super().__init__()
        self.encoder = mlp([in_dim, *hidden, latent])
        self.decoder = mlp([latent, *reversed(hidden), in_dim])
        self.codebooks = nn.Parameter(torch.zeros(levels, codes, latent))
        self.levels, self.codes = levels, codes

    @torch.no_grad()
    def init_codebooks(self, x: torch.Tensor) -> None:
        r = self.encoder(x)
        for level in range(self.levels):
            self.codebooks[level] = kmeans(r, self.codes, seed=level)
            r = r - self.codebooks[level][torch.cdist(r, self.codebooks[level]).argmin(1)]

    def quantize(self, z: torch.Tensor):
        r, z_hat, codes, q_loss = z, torch.zeros_like(z), [], z.new_zeros(())
        for level in range(self.levels):
            book = self.codebooks[level]
            idx = torch.cdist(r.detach(), book.detach()).argmin(1)
            e = book[idx]
            q_loss = (
                q_loss + ((r.detach() - e) ** 2).sum(1).mean() + 0.25 * ((r - e.detach()) ** 2).sum(1).mean()
            )
            codes.append(idx)
            z_hat = z_hat + e
            r = r - e.detach()
        return z_hat, torch.stack(codes, 1), q_loss

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z_hat, codes, q_loss = self.quantize(z)
        x_hat = self.decoder(z + (z_hat - z).detach())
        recon = ((x - x_hat) ** 2).sum(1).mean()
        return recon, q_loss, codes


def add_dedupe_code(codes: np.ndarray) -> np.ndarray:
    """[N, 3] -> [N, 4]: the fourth code numbers the items sharing a three-code prefix, in row order."""
    seen: Counter = Counter()
    fourth = np.zeros(len(codes), dtype=np.int64)
    for i, prefix in enumerate(map(tuple, codes)):
        fourth[i] = seen[prefix]
        seen[prefix] += 1
    return np.concatenate([codes, fourth[:, None]], axis=1)


def code_stats(codes3: np.ndarray, codes: int = 256) -> dict:
    groups = Counter(map(tuple, codes3))
    n = len(codes3)
    sizes = np.array(list(groups.values()))
    per_level = []
    for level in range(codes3.shape[1]):
        counts = np.bincount(codes3[:, level], minlength=codes)
        p = counts[counts > 0] / n
        per_level.append(
            {
                "codes_used": int((counts > 0).sum()),
                "usage": round(float((counts > 0).mean()), 4),
                "perplexity": round(float(np.exp(-(p * np.log(p)).sum())), 1),
            }
        )
    return {
        "items": n,
        "distinct_3code_ids": len(groups),
        "collision_rate": round(1 - len(groups) / n, 4),  # 1 - distinct IDs / items
        "items_in_collisions": int(sizes[sizes > 1].sum()),
        "share_of_items_in_collisions": round(float(sizes[sizes > 1].sum() / n), 4),
        "largest_collision_group": int(sizes.max()),
        "per_level": per_level,
    }


def train(x_train: torch.Tensor, epochs: int, seed: int, batch: int = 1024, lr: float = 1e-3):
    torch.manual_seed(seed)
    model = RQVAE(x_train.shape[1]).to(x_train.device)
    g = torch.Generator(device=x_train.device).manual_seed(seed)
    order = torch.randperm(x_train.shape[0], generator=g, device=x_train.device)
    model.init_codebooks(x_train[order[:batch]])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    log = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(x_train.shape[0], generator=g, device=x_train.device)
        totals = torch.zeros(2, device=x_train.device)
        for s in range(0, x_train.shape[0], batch):
            recon, q_loss, _ = model(x_train[order[s : s + batch]])
            opt.zero_grad(set_to_none=True)
            (recon + q_loss).backward()
            opt.step()
            totals += torch.stack([recon.detach(), q_loss.detach()])
        if epoch % 1000 == 0 or epoch == epochs:
            steps = -(-x_train.shape[0] // batch)
            r, q = (totals / steps).tolist()
            with torch.no_grad():
                codes = model(x_train)[2]
            used = [int(codes[:, level].unique().numel()) for level in range(model.levels)]
            distinct = int(codes.unique(dim=0).shape[0])
            log.append({"epoch": epoch, "recon": r, "quant": q, "codes_used": used, "distinct_ids": distinct})
            print(log[-1], flush=True)
    return model, log


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False  # exact nearest-code search
    data = loo.load()
    emb = torch.from_numpy(np.load(EMB_DIR / f"{args.encoder}.npy")).cuda()
    train_items = sorted({int(i) for s in data.seqs for i in s[:-2]})
    smi = common.nvidia_smi()
    t0 = time.time()
    model, log = train(emb[torch.tensor(train_items) - 1], args.epochs, args.seed)
    model.eval()
    with torch.no_grad():
        z = model.encoder(emb)
        _, codes3, _ = model.quantize(z)
        recon, q_loss, _ = model(emb)
        x_hat = model.decoder(model.quantize(z)[0])
        explained = 1 - ((emb - x_hat) ** 2).sum() / ((emb - emb.mean(0)) ** 2).sum()
    codes3 = codes3.cpu().numpy()
    codes4 = add_dedupe_code(codes3)
    SID_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SID_DIR / f"{args.encoder}.npy", np.concatenate([np.zeros((1, 4), dtype=np.int64), codes4]))
    record = {
        "encoder": args.encoder,
        "embedding_dim": int(emb.shape[1]),
        "train_items": len(train_items),
        "settings": {
            "hidden": [512, 256, 128],
            "latent": 32,
            "levels": 3,
            "codes": 256,
            "beta": 0.25,
            "optimizer": "AdamW, weight decay 0",
            "lr": 1e-3,
            "batch": 1024,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "final_loss_all_items": {"recon": round(recon.item(), 5), "quant": round(q_loss.item(), 5)},
        "reconstruction_explained_variance": round(explained.item(), 4),
        **code_stats(codes3),
        "fourth_code_values": int(codes4[:, 3].max()) + 1,
        "loss_log": log,
        "wall_time_s": round(time.time() - t0, 1),
        "gpu": {"nvidia_smi_start": smi, "nvidia_smi_end": common.nvidia_smi()},
        "environment": common.environment(),
        "git": common.git_state(),
    }
    common.write_json(common.RESULTS / "semantic_ids" / f"{args.encoder}.json", record)
    keys = ("reconstruction_explained_variance", "collision_rate", "largest_collision_group", "per_level")
    print({k: record[k] for k in keys})


if __name__ == "__main__":
    main()
