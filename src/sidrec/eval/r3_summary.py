"""R3 summary: embedding throughput, the agreement of the two implementations, and the semantic-ID statistics.

    python -m sidrec.eval.r3_summary

Reads the run records under results/r3/ and writes results/r3/summary.json and results/r3/SHA256SUMS, the
checksums of the large arrays that stay out of git (check them from the repository root with
``sha256sum -c results/r3/SHA256SUMS``). Prints the README tables.
"""

from __future__ import annotations

import json
from pathlib import Path

from sidrec.semantic_ids import catalog
from sidrec.train import common

# record name -> label
RUNS = {
    "vllm": "vLLM 0.31.0, full catalog",
    "hf-bs64": "R2 loop, batch 64, full catalog",
    "vllm-sample20000": "vLLM 0.31.0, 20,000-item sample",
    "hf-bs64-sample20000": "R2 loop, batch 64, sample",
    "hf-bs128-sample20000": "R2 loop, batch 128, sample",
    "hf-bs256-sample20000": "R2 loop, batch 256, sample",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def run_row(name: str, rec: dict) -> dict:
    gpu = rec["gpu"].get("during_encode") or rec["gpu"]["during_load_and_encode"]
    return {
        "run": name,
        "label": RUNS[name],
        "items": rec["items"],
        "tokens": rec["tokens"],
        "encode_s": rec["encode_seconds"],
        "items_per_s": rec["items_per_s"],
        "tokens_per_s": rec["tokens_per_s"],
        "outside_encode_s": round(rec["wall_time_s"] - rec["encode_seconds"], 1),
        "padding_share": rec.get("padding", {}).get("padding_share", 0.0),
        "sm_clock_mhz_median": gpu["clocks.sm"]["median"],
        "power_w_median": gpu["power.draw"]["median"],
        "utilization_median": gpu["utilization.gpu"]["median"],
        "memory_used_mib_max": gpu["memory.used"]["max"],
        "pstates": gpu["pstates"],
        "git": rec["git"],
    }


def summarize(results: Path = catalog.RESULTS) -> dict:
    runs = {n: load(results / "embeddings" / f"{n}.json") for n in RUNS}
    rows = [run_row(n, r) for n, r in runs.items()]
    agree = load(results / "embeddings" / "agreement.json")
    sid = load(results / "semantic_ids.json")
    docs = load(results / "item_docs.json")
    gpu_s = sum(r["wall_time_s"] for r in runs.values()) + sid["wall_time_s"]
    return {
        "item_docs": {k: docs[k] for k in ("items", "tokens_total", "tokens_per_item", "train_items")},
        "throughput": rows,
        "full_catalog_speedup_vllm_over_hf": round(
            runs["vllm"]["tokens_per_s"] / runs["hf-bs64"]["tokens_per_s"], 2
        ),
        "agreement": {k: v for k, v in agree.items() if k != "arrays"},
        "semantic_ids": {
            k: sid[k]
            for k in (
                "train_items",
                "settings",
                "reconstruction_explained_variance",
                "fourth_code_values",
                "prefix_coverage_by_train_items",
                "neighbour_prefix_agreement",
            )
        }
        | {
            "all_items": sid["all_items"],
            "train_items_only": sid["train_items_only"],
        },
        "gpu_hours_rtx4090_recorded": round(gpu_s / 3600, 2),
    }


def checksums(results: Path = catalog.RESULTS) -> list[str]:
    docs = load(results / "item_docs.json")["docs_json"]
    arrays = load(results / "embeddings" / "agreement.json")["arrays"]
    art = load(results / "semantic_ids.json")["artifacts"]
    entries = [docs, arrays["vllm"], arrays["hf"], art["codes"], art["model"]]
    return [f"{e['sha256']}  {e['path']}" for e in entries]


def main() -> None:
    summary = summarize()
    common.write_json(catalog.RESULTS / "summary.json", summary)
    (catalog.RESULTS / "SHA256SUMS").write_text("\n".join(checksums()) + "\n")
    print("| Run | Items/s | Tokens/s | Encode s | Padding | SM clock | Power | Util |")
    print("|---|---|---|---|---|---|---|---|")
    for r in summary["throughput"]:
        print(
            f"| {r['label']} | {r['items_per_s']:,.0f} | {r['tokens_per_s']:,.0f} | {r['encode_s']:,.0f} "
            f"| {r['padding_share']:.0%} | {r['sm_clock_mhz_median']:.0f} MHz | {r['power_w_median']:.0f} W "
            f"| {r['utilization_median']:.0f}% |"
        )
    print(
        "speedup",
        summary["full_catalog_speedup_vllm_over_hf"],
        "GPU h",
        summary["gpu_hours_rtx4090_recorded"],
    )


if __name__ == "__main__":
    main()
