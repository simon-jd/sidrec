"""R5b summary: results/r5b/summary.json and results/r5b/SHA256SUMS from the R5b records, set against R5's.

    python -m sidrec.eval.r5b_summary

The tables (profile, beam ladder, fp32 equivalence, bf16 quality, batch-1 stages, the cascade under load,
before and after, GPU time and background load) are derived here; every R5b record they come from is copied
under "records". R5's numbers are read from results/r5/summary.json.
"""

from __future__ import annotations

import json

from sidrec.eval import r5_summary
from sidrec.eval.r5b_beam import LADDER, LADDER_BATCH, LOGS, RESULTS, RUNS, WIDTHS
from sidrec.semantic_ids.catalog import file_sha256
from sidrec.train import common

R5 = common.ROOT / "results" / "r5" / "summary.json"


def load(path):
    return json.loads(path.read_text())


def profile_table(rec: dict) -> dict:
    keep = (
        "wall_ms",
        "gpu_kernel_ms",
        "gpu_busy_share",
        "kernels",
        "launch_calls",
        "host_syncs",
        "item_calls",
    )
    out = {}
    for name, c in rec["configs"].items():
        row = {k: c[k] for k in keep} | {"top_kernel": c["top_kernels"][0] if c["top_kernels"] else None}
        if "split_gpu_ms" in c:
            row["split_gpu_ms"] = c["split_gpu_ms"]
            row["labels_gpu_ms"] = {k: v["gpu_ms"] for k, v in c["labels"].items()}
        out[name] = row
    return out


def ladder_table(records: dict) -> dict:
    out = {}
    for step in LADDER:
        rec = records.get(f"ladder_{step}")
        if not rec:
            continue
        out[step] = {}
        for w in WIDTHS:
            r = rec["widths"][str(w)]
            b = r[f"batch{LADDER_BATCH}"]
            out[step][str(w)] = {
                "batch1_p50_ms": r["batch1"]["p50_ms"],
                "batch1_p99_ms": r["batch1"]["p99_ms"],
                f"batch{LADDER_BATCH}_p50_ms": b["p50_ms"],
                f"batch{LADDER_BATCH}_p99_ms": b["p99_ms"],
                f"batch{LADDER_BATCH}_requests_per_s": b["requests_per_s"],
            }
    if "reference" in out:
        for widths in out.values():
            for w, row in widths.items():
                ref = out["reference"][w]
                row["batch1_speedup"] = ref["batch1_p50_ms"] / row["batch1_p50_ms"]
                key = f"batch{LADDER_BATCH}_requests_per_s"
                row[f"batch{LADDER_BATCH}_speedup"] = row[key] / ref[key]
    return out


def equivalence_table(rec: dict) -> dict:
    keep = (
        "identical_share",
        "mismatched_rows",
        "near_tie_rows",
        "max_tie_gap",
        "max_abs_score_diff_same_items",
        "items_shared",
        "first_position_min",
        "first_position_median",
    )
    return {w: {name: {k: c[k] for k in keep} for name, c in res.items()} for w, res in rec["widths"].items()}


def quality_table(rec: dict) -> dict:
    out = {}
    for w, r in rec["widths"].items():
        out[w] = {
            "union_recall@200": r["union_recall@200"],
            "hstu_sid_ndcg@10": r["hstu_sid"]["all"]["ndcg@10"],
            "lists_identical_to_r5_share": r["lists_identical_to_r5_share"],
            "items_shared_with_r5": r["items_shared_with_r5"],
        } | r["against_r5"]
    return out


def gpu_time() -> dict:
    path = LOGS / "gpu_time.jsonl"
    steps = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for s in steps:
        s["hours"] = (s["end"] - s["start"]) / 3600
    recorded = [s for s in steps if s["step"] not in ("smoke", "tests", "debug")]
    return {
        "steps": steps,
        "recorded_gpu_hours": sum(s["hours"] for s in recorded),
        "development_gpu_hours": sum(s["hours"] for s in steps) - sum(s["hours"] for s in recorded),
        "total_gpu_hours": sum(s["hours"] for s in steps),
        "definition": "wall time holding the RTX 4090 lock: scripts/r5b_run.sh for the steps; smoke "
        "tests, test suite runs and debugging passes are listed as development",
    }


def checksums() -> list[str]:
    lines = [f"{file_sha256(p)}  {p.relative_to(common.ROOT)}" for p in sorted(RUNS.rglob("*.npz"))]
    (RESULTS / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return lines


def main() -> None:
    before = load(R5)
    records = {"projection": load(RESULTS / "projection.json"), "profile": load(RESULTS / "profile.json")}
    for name in ("equivalence", "quality", "select"):
        records[name] = load(RESULTS / "valid" / f"{name}.json")
    records["stages_batch1"] = load(RESULTS / "test" / "stages_batch1.json")
    for step in LADDER:
        path = RESULTS / "test" / "ladder" / f"{step}.json"
        if path.exists():
            records[f"ladder_{step}"] = load(path)
    for w in WIDTHS:
        records[f"e2e_w{w}"] = load(RESULTS / "test" / f"e2e_w{w}.json")
    stages = {
        k: {"p50_ms": v["p50_ms"], "p99_ms": v["p99_ms"]}
        | ({"stages_ms": v["stages_ms"]} if "stages_ms" in v else {})
        for k, v in records["stages_batch1"]["batch1"].items()
    }
    quality = quality_table(records["quality"])
    cascade, before_after = {}, {}
    for w in WIDTHS:
        rec = records[f"e2e_w{w}"]
        cascade[str(w)] = {
            "max_batch": rec["max_batch"],
            "saturation_qps": rec["saturation"]["qps"],
            "batch1_p50_ms": stages[f"cascade_w{w}"]["p50_ms"],
            "batch1_p99_ms": stages[f"cascade_w{w}"]["p99_ms"],
            "under_load": r5_summary.under_load(rec),
            "valid_union_recall@200": quality[str(w)]["union_recall@200"],
            "valid_hstu_sid_ndcg@10": quality[str(w)]["hstu_sid_ndcg@10"],
        } | r5_summary.at_slo(rec)
        b, a = before["cascade"][str(w)], cascade[str(w)]
        bl, al = b.get("under_load", {}), a.get("under_load", {})
        before_after[str(w)] = {
            "beam_batch1_p50_ms": [
                before["stages_batch1"][f"beam_w{w}"]["p50_ms"],
                stages[f"beam_w{w}"]["p50_ms"],
            ],
            "beam_batch1_p99_ms": [
                before["stages_batch1"][f"beam_w{w}"]["p99_ms"],
                stages[f"beam_w{w}"]["p99_ms"],
            ],
            "cascade_batch1_p50_ms": [b["batch1_p50_ms"], a["batch1_p50_ms"]],
            "cascade_batch1_p99_ms": [b["batch1_p99_ms"], a["batch1_p99_ms"]],
            "max_batch": [b["max_batch"], a["max_batch"]],
            "saturation_qps": [b["saturation_qps"], a["saturation_qps"]],
            "qps_at_slo": [b["qps_at_slo"], a["qps_at_slo"]],
            "p50_ms_at_slo": [b.get("p50_ms"), a.get("p50_ms")],
            "p99_ms_at_slo": [b.get("p99_ms"), a.get("p99_ms")],
            "mean_batch_at_slo": [b.get("mean_batch"), a.get("mean_batch")],
            "beam_share_under_load": [bl.get("beam_ms", {}).get("share"), al.get("beam_ms", {}).get("share")],
            "beam_p50_ms_per_batch_under_load": [
                bl.get("beam_ms", {}).get("p50"),
                al.get("beam_ms", {}).get("p50"),
            ],
            "beam_p99_ms_per_batch_under_load": [
                bl.get("beam_ms", {}).get("p99"),
                al.get("beam_ms", {}).get("p99"),
            ],
            "dominant_stage_under_load": [bl.get("dominant"), al.get("dominant")],
            "union_recall@200": [b["valid_union_recall@200"]["all"], a["valid_union_recall@200"]["all"]],
            "hstu_sid_ndcg@10": [b["valid_hstu_sid_ndcg@10"], a["valid_hstu_sid_ndcg@10"]],
        }
    summary = {
        "slo_p99_ms": 50.0,
        "protocol": records["quality"]["protocol"],
        "before": "results/r5/summary.json (R5's records, read only)",
        "profile": profile_table(records["profile"]),
        "ladder": ladder_table(records),
        "equivalence": equivalence_table(records["equivalence"]),
        "quality": quality,
        "stages_batch1": stages,
        "cascade": cascade,
        "before_after": before_after,
        "gpu_time": gpu_time(),
        "projection": {k: records["projection"][k] for k in ("projected_gpu_hours", "worst_case_gpu_hours")},
        "background": r5_summary.background(records),
        "checksums": checksums(),
        "records": records,
    }
    common.write_json(RESULTS / "summary.json", summary)
    for w, row in before_after.items():
        print(w, {k: v for k, v in row.items() if k in ("beam_batch1_p50_ms", "qps_at_slo", "max_batch")})
    print(
        round(summary["gpu_time"]["recorded_gpu_hours"], 3), round(summary["gpu_time"]["total_gpu_hours"], 3)
    )


if __name__ == "__main__":
    main()
