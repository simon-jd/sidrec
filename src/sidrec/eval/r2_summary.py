"""Summarize the R2 Beauty runs against the published numbers of docs/protocol.md.

    python -m sidrec.eval.r2_summary       # writes results/r2/summary.json and prints the tables

Final runs are the selected trial with seeds 0, 1 and 2; numbers are mean and sample standard deviation over
them. The gap is (ours - published) / published.
"""

from __future__ import annotations

import json
import statistics

from sidrec.train.common import RESULTS

METRICS = ("recall@5", "ndcg@5", "recall@10", "ndcg@10")
# Published Beauty numbers (docs/protocol.md, section 1). None where the source reports no number.
PUBLISHED = {
    "tiger_table1": ("TIGER Table 1", (0.0454, 0.0321, 0.0648, 0.0384)),
    "tiger_table9": ("TIGER Table 9, 3-seed mean", (0.0441, 0.0309, 0.0642, 0.0374)),
    "sasrec_tiger_table1": ("TIGER Table 1, SASRec (BCE)", (0.0387, 0.0249, 0.0605, 0.0318)),
    "mender_tiger": ("Mender Table 2, TIGER reproduction", (0.0431, 0.0275, 0.0681, 0.0356)),
    "esasrec_sasrec_ss": ("eSASRec Table 4, SASRec+SS", (None, None, 0.092, 0.053)),
    "esasrec_hstu": ("eSASRec Table 4, HSTU", (None, None, 0.079, 0.044)),
    "actionpiece_hstu": ("ActionPiece Table 4, HSTU", (0.0469, 0.0314, 0.0704, 0.0389)),
}
# model run directory -> (label, primary references, secondary references)
MODELS = {
    "sasrec": ("SASRec, CE loss", ["sasrec_tiger_table1"], ["esasrec_sasrec_ss"]),
    "hstu": ("HSTU-style, CE loss", ["esasrec_hstu"], ["actionpiece_hstu"]),
    "hstu-notime": ("HSTU-style without the time bias (ablation)", ["esasrec_hstu"], ["actionpiece_hstu"]),
    "tiger_st5-xxl": ("TIGER, Sentence-T5-XXL IDs", ["tiger_table1", "tiger_table9"], ["mender_tiger"]),
    "tiger_qwen3-emb-0.6b": (
        "TIGER, Qwen3-Embedding-0.6B IDs",
        ["tiger_table1", "tiger_table9"],
        ["mender_tiger"],
    ),
    "tiger_qwen3-emb-4b": (
        "TIGER, Qwen3-Embedding-4B IDs",
        ["tiger_table1", "tiger_table9"],
        ["mender_tiger"],
    ),
    # Diagnostic outside the protocol: the paper's setting with the paper's optimizer (Adafactor), seeds 0-2.
    "tiger_st5-xxl_adafactor": (
        "TIGER, Sentence-T5-XXL IDs, Adafactor (diagnostic)",
        ["tiger_table1", "tiger_table9"],
        ["mender_tiger"],
    ),
}


def load_runs(run_dir: str) -> tuple[list[dict], list[dict]]:
    tune, final = [], []
    for path in sorted((RESULTS / "runs" / run_dir).glob("*.json")):
        record = json.loads(path.read_text())
        (final if record["phase"] in ("final", "diagnostic") else tune).append(record)
    return tune, final


def gpu_seconds(records: list[dict]) -> float:
    """Wall time of the runs. A test-only record counts only its evaluation; its trial counts the training."""
    return sum(r["test_eval"]["wall_time_s"] if "test_eval" in r else r["wall_time_s"] for r in records)


def summarize() -> dict:
    out: dict = {
        "published": {
            k: {"source": s, **dict(zip(METRICS, v, strict=True))} for k, (s, v) in PUBLISHED.items()
        }
    }
    for run_dir, (label, primary, secondary) in MODELS.items():
        tune, final = load_runs(run_dir)
        if len(final) != 3:
            out[run_dir] = {"label": label, "status": f"{len(final)} of 3 final runs"}
            continue
        stats = {}
        for m in METRICS:
            values = [r["test"][m] for r in final]
            stats[m] = {"mean": statistics.mean(values), "sd": statistics.stdev(values), "values": values}
        gaps = {}
        for ref in primary + secondary:
            for i, m in enumerate(METRICS):
                published = PUBLISHED[ref][1][i]
                if published is not None:
                    gaps.setdefault(ref, {})[m] = (stats[m]["mean"] - published) / published
        breach = [f"{ref} {m}" for ref in primary for m in ("recall@10", "ndcg@10") if gaps[ref][m] < -0.15]
        out[run_dir] = {
            "label": label,
            "trial": final[0]["trial"],
            "config": final[0]["config"],
            "seeds": [r["seed"] for r in final],
            "best_epochs": [r["best_epoch"] for r in final],
            "test": stats,
            "valid_ndcg@10": [r["valid"]["ndcg@10"] for r in final],
            "relative_gap": gaps,
            "primary": primary,
            "secondary": secondary,
            "fifteen_percent_rule": breach,
            "tuning_trials": len(tune),
            "gpu": sorted({r["environment"]["device"] for r in tune + final}),
        }
    # Every recorded GPU job: runs (including diagnostics and superseded ones), embeddings, RQ-VAEs.
    records = [json.loads(p.read_text()) for p in sorted(RESULTS.rglob("*.json")) if p.name != "summary.json"]
    out["gpu_hours_rtx4090"] = round(gpu_seconds(records) / 3600, 2)
    return out


def table(summary: dict) -> str:
    rows = [
        "| Model | R@10 (ours) | N@10 (ours) | Reference | R@10 | N@10 | Gap R@10 | Gap N@10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run_dir in MODELS:
        s = summary[run_dir]
        if "test" not in s:
            continue
        r10, n10 = s["test"]["recall@10"], s["test"]["ndcg@10"]
        for ref in s["primary"] + s["secondary"]:
            pub = summary["published"][ref]
            gap = s["relative_gap"][ref]
            rows.append(
                f"| {s['label']} | {r10['mean']:.4f} ± {r10['sd']:.4f} | {n10['mean']:.4f} ± {n10['sd']:.4f} "
                f"| {pub['source']} | {pub['recall@10']} | {pub['ndcg@10']} "
                f"| {gap['recall@10']:+.1%} | {gap['ndcg@10']:+.1%} |"
            )
    return "\n".join(rows)


def main() -> None:
    summary = summarize()
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(table(summary))
    print("GPU hours (RTX 4090):", summary["gpu_hours_rtx4090"])


if __name__ == "__main__":
    main()
