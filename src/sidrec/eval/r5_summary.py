"""R5 summary: results/r5/summary.json and results/r5/SHA256SUMS from the R5 records.

    python -m sidrec.eval.r5_summary

The tables (ladder, batch-1 stages, the cascade under load, what a narrower beam costs, GPU time and the
background load) are derived here; every record they come from is copied under "records".
"""

from __future__ import annotations

import json

import numpy as np

from sidrec.eval.r5_serving import CANDS, LADDER, LOGS, RESULTS, RUNS, WIDTHS
from sidrec.semantic_ids.catalog import file_sha256
from sidrec.train import common

STAGES = ("beam_ms", "hnsw_ms", "union_ms", "ranker_ms")


def load(path):
    return json.loads(path.read_text())


def at_slo(rec: dict) -> dict:
    """The confirmed rate and the median p50 and p99 of its three runs; None if the SLO was never met."""
    if not rec.get("qps_at_slo"):
        return {"qps_at_slo": None}
    a = rec["at_slo"]
    return {
        "qps_at_slo": rec["qps_at_slo"],
        "p50_ms": a["p50_ms_median"],
        "p99_ms": a["p99_ms_median"],
        "p99_ms_runs": a["p99_ms"],
        "mean_batch": float(np.mean(a["mean_batch"])),
        "confirmation_attempts": len(rec["confirmation"]),
    }


def ladder_table(quality: dict) -> list[dict]:
    rows = []
    for name in LADDER:
        path = RESULTS / "test" / "ladder" / f"{name}.json"
        if not path.exists():
            continue
        rec = load(path)
        q = quality["steps"].get(name, {})
        row = {
            "step": name,
            "config": rec["config"],
            "max_batch": rec["max_batch"],
            "batch1_p50_ms": rec["batch1"]["p50_ms"],
            "batch1_p99_ms": rec["batch1"]["p99_ms"],
            "saturation_qps": rec["saturation"]["qps"],
        } | at_slo(rec)
        row["candidates_per_s"] = CANDS * row["qps_at_slo"] if row["qps_at_slo"] else None
        row["quality"] = {
            "top10_overlap": q.get("top10_overlap"),
            "ndcg@10_change": q.get("ndcg@10_change"),
            "ndcg@10_change_relative": q.get("ndcg@10_change_relative"),
            "keeps_quality": q.get("keeps_quality"),
        }
        rows.append(row)
    return rows


def under_load(rec: dict) -> dict:
    """Per-batch stage times of the three runs that confirmed the rate: medians of their p50, p99, mean and
    share of batch time, and the dominant stage."""
    if not rec.get("qps_at_slo"):
        return {}
    runs = rec["confirmation"][-1]["runs"]
    out = {}
    for k in (*STAGES, "batch_ms"):
        out[k] = {m: float(np.median([r["stages"][k][m] for r in runs])) for m in ("p50", "p99", "mean", "share")}
    out["wait_p50_ms"] = float(np.median([r["wait"]["p50_ms"] for r in runs]))
    out["wait_p99_ms"] = float(np.median([r["wait"]["p99_ms"] for r in runs]))
    out["dominant"] = max(STAGES, key=lambda k: out[k]["share"])
    return out


def gpu_time() -> dict:
    path = LOGS / "gpu_time.jsonl"
    steps = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for s in steps:
        s["hours"] = (s["end"] - s["start"]) / 3600
    return {
        "steps": steps,
        "recorded_gpu_hours": sum(s["hours"] for s in steps),
        "definition": "wall time holding the RTX 4090 lock, from scripts/r5_run.sh",
    }


def background(records: dict) -> dict:
    out = {}
    for name, rec in records.items():
        bg = rec.get("background")
        if not bg:
            continue
        out[name] = {
            "load1_median": bg.get("load1", {}).get("median"),
            "load1_max": bg.get("load1", {}).get("max"),
            "sm_clock_mhz_median": bg.get("clocks.sm", {}).get("median"),
            "sm_clock_mhz_min": bg.get("clocks.sm", {}).get("min"),
            "gpu_util_median": bg.get("utilization.gpu", {}).get("median"),
            "temperature_max": bg.get("temperature.gpu", {}).get("max"),
            "other_gpu_processes": bg.get("other_gpu_processes"),
            "top_cpu_at_start": [f"{p.get('command')} {p.get('cpu_pct')}%" for p in bg.get("top_at_start", [])[:5]],
        }
    return out


def checksums() -> list[str]:
    lines = []
    for path in sorted(RUNS.rglob("*.npz")):
        lines.append(f"{file_sha256(path)}  {path.relative_to(common.ROOT)}")
    (RESULTS / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return lines


def main() -> None:
    records = {"projection": load(RESULTS / "projection.json")}
    for name in ("beam_widths", "select", "quality"):
        records[name] = load(RESULTS / "valid" / f"{name}.json")
    records["stages_batch1"] = load(RESULTS / "test" / "stages_batch1.json")
    for name in LADDER:
        path = RESULTS / "test" / "ladder" / f"{name}.json"
        if path.exists():
            records[f"ladder_{name}"] = load(path)
    for w in WIDTHS:
        records[f"e2e_w{w}"] = load(RESULTS / "test" / f"e2e_w{w}.json")
    quality = records["quality"]
    stages = {
        k: {"p50_ms": v["p50_ms"], "p99_ms": v["p99_ms"]} for k, v in records["stages_batch1"]["batch1"].items()
    }
    cascade = {}
    for w in WIDTHS:
        rec = records[f"e2e_w{w}"]
        bw = records["beam_widths"]["widths"][str(w)]
        cascade[str(w)] = {
            "max_batch": rec["max_batch"],
            "saturation_qps": rec["saturation"]["qps"],
            "batch1_p50_ms": stages[f"cascade_w{w}"]["p50_ms"],
            "batch1_p99_ms": stages[f"cascade_w{w}"]["p99_ms"],
            "under_load": under_load(rec),
            "valid_union_recall@200": bw["union_recall@200"],
            "valid_hstu_sid_ndcg@10": bw["hstu_sid"]["all"]["ndcg@10"],
        } | at_slo(rec)
    summary = {
        "slo_p99_ms": 50.0,
        "protocol": records["quality"]["protocol"],
        "quality_check": {
            "keeps": quality["keeps"],
            "fp32_against_r4c_record": quality["fp32_against_r4c_record"],
            "serving_check": quality["serving_check"],
        },
        "ladder": ladder_table(quality),
        "stages_batch1": stages,
        "cascade": cascade,
        "gpu_time": gpu_time(),
        "background": background(records),
        "checksums": checksums(),
        "records": records,
    }
    common.write_json(RESULTS / "summary.json", summary)
    for row in summary["ladder"]:
        print(row["step"], round(row["batch1_p50_ms"], 3), row["qps_at_slo"], row["quality"]["top10_overlap"])
    for w, c in cascade.items():
        print(w, c["qps_at_slo"], c.get("p99_ms"), c["under_load"].get("dominant"))
    print(round(summary["gpu_time"]["recorded_gpu_hours"], 2))


if __name__ == "__main__":
    main()
