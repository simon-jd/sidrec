"""Build results/r4b/summary.json from the R4b records, and the GPU-hour projection made before the long runs.

    python -m sidrec.eval.r4b_summary              # summary.json
    python -m sidrec.eval.r4b_summary --projection # projection.json

**Projection (RTX 4090).** From R4a's cost table (results/r4a/cost, full split, width 256 x 4 layers, the size
of every selected configuration):
- one epoch = the measured training epoch plus one evaluation of the 100,000-row validation sample;
- one test evaluation = the timed evaluation of all 1,141,362 validation rows, scaled to the 1,199,539
  test rows;
- epochs per run = what the selected dev-subset trial ran (its best epoch plus the patience), at most 30;
- HSTU-SID and the ablation = the HSTU-style model's cost and epochs; HSTU-SID's epoch is scaled by its dev
  epoch time over the HSTU-style model's, at trial 1;
- a run = epochs x epoch + one test evaluation, plus 1 minute of start-up for each process of at most an hour.
Each R4b record that exists replaces its projected run by its measured wall time, so the same function
re-projects the total as runs finish.

**Summary.** Test metrics of each final run, mean and sample standard deviation over seeds; TIGER's cold-start
retrieval at ε = 0.1 next to plain TIGER; the time ablation; HSTU-SID's tuning trials; the validation ε curve;
and the GPU time of every record.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

from sidrec.train import common
from sidrec.train.scale_idrec import MAX_EPOCHS

R4A = common.ROOT / "results" / "r4a"
RESULTS = common.ROOT / "results" / "r4b"
TEST_ROWS, VALID_ROWS = 1_199_539, 1_141_362
SEEDS = (0, 1, 2)
FINAL = {"sasrec": SEEDS, "hstu": SEEDS, "hstu-sid": SEEDS, "tiger": SEEDS, "hstu-notime": (0,)}
COST_AS = {"hstu-sid": "hstu", "hstu-notime": "hstu"}
PARTS, METRICS = ("all", "warm", "cold"), ("recall@10", "recall@50", "ndcg@10")
STARTUP_S = 60.0
LIMIT_H = 60.0


def load(path):
    return json.loads(path.read_text())


def record(model: str, seed: int) -> dict | None:
    path = RESULTS / "runs" / model / f"seed{seed}.json"
    return load(path) if path.exists() else None


def dev_trial(model: str, trial: int) -> dict:
    folder = RESULTS if model == "hstu-sid" else R4A
    return load(folder / "runs" / model / f"trial{trial:02d}.json")


def selected(model: str) -> dict:
    """The selected dev-subset trial's record (best validation NDCG@10, ties to the lower trial)."""
    folder = RESULTS if model == "hstu-sid" else R4A
    runs = [load(p) for p in sorted((folder / "runs" / model).glob("trial[0-9][0-9].json"))]
    if len(runs) != 12:
        raise ValueError(f"{model}: {len(runs)} tuning trials")
    return max(runs, key=lambda r: (r["valid"]["all"]["ndcg@10"], -r["trial"]))


def projection(cost_dir=R4A / "cost") -> dict:
    """Projected GPU-hours of every R4b run; finished runs count with their measured wall time."""
    cost = {}
    for path in cost_dir.glob("*_full_trial01.json"):
        r = load(path)
        h = r["history"][0]
        cost[r["model"]] = {
            "epoch_s": h["train_s"] + h["eval_s"],
            "sample_eval_s": h["eval_s"],
            "test_s": r["full_valid_eval"]["seconds"] * TEST_ROWS / VALID_ROWS,
        }
    tuned = bool(tuning_records())
    ratio = 1.0
    if tuned:

        def mean(r):
            return statistics.mean(x["train_s"] for x in r["history"])

        ratio = mean(dev_trial("hstu-sid", 1)) / mean(dev_trial("hstu", 1))
    runs, total, done = [], 0.0, 0.0
    for model, seeds in FINAL.items():
        base = cost[COST_AS.get(model, model)]
        like = model if model != "hstu-notime" and (model != "hstu-sid" or tuned) else "hstu"
        epochs = min(MAX_EPOCHS, selected(like)["epochs_run"])
        epoch_s = base["epoch_s"] * (ratio if model == "hstu-sid" else 1.0)
        train_s = epochs * epoch_s
        projected_s = train_s + base["test_s"] + STARTUP_S * math.ceil(train_s / 3600)
        for seed in seeds:
            rec = record(model, seed)
            hours = rec["wall_time_s"] / 3600 if rec else projected_s / 3600
            done += hours if rec else 0.0
            total += hours
            runs.append(
                {
                    "model": model,
                    "seed": seed,
                    "epochs": rec["epochs_run"] if rec else epochs,
                    "hours": round(hours, 2),
                    "measured": rec is not None,
                }
            )
    # HSTU-SID's 12 trials: measured once they exist, else the HSTU-style model's R4a tuning time.
    trials = tuning_records() or [dev_trial("hstu", t) for t in range(1, 13)]
    tuning_h = sum(r["wall_time_s"] for r in trials) / 3600
    eps_recs = [load(p) for p in (RESULTS / "tiger_eps").glob("seed*.json")]
    eps_h = (
        sum(r["wall_time_s"] for r in eps_recs) / 3600
        if eps_recs
        else len(SEEDS) * (cost["tiger"]["sample_eval_s"] + STARTUP_S) / 3600
    )
    total += tuning_h + eps_h
    return {
        "basis": "R4a full-split cost at width 256 x 4; epochs of the selected dev trial (module docstring)",
        "epoch_hours": {m: round(c["epoch_s"] / 3600, 3) for m, c in cost.items()},
        "test_eval_hours": {m: round(c["test_s"] / 3600, 3) for m, c in cost.items()},
        "hstu_sid_epoch_ratio": round(ratio, 3),
        "runs": runs,
        "hstu_sid_tuning_hours": round(tuning_h, 2),
        "tiger_eps_curve_hours": round(eps_h, 2),
        "measured_hours": round(done + (tuning_h if tuning_records() else 0.0), 2),
        "total_hours": round(total, 1),
        "limit_hours": LIMIT_H,
        "within_limit": total <= LIMIT_H,
        "bound_at_30_epochs_hours": round(bound_at_cap(cost, ratio) + tuning_h + eps_h, 1),
    }


def bound_at_cap(cost: dict, ratio: float) -> float:
    total = 0.0
    for model, seeds in FINAL.items():
        base = cost[COST_AS.get(model, model)]
        epoch_s = base["epoch_s"] * (ratio if model == "hstu-sid" else 1.0)
        total += len(seeds) * (MAX_EPOCHS * epoch_s + base["test_s"]) / 3600
    return total


def tuning_records() -> list[dict]:
    return [load(p) for p in sorted((RESULTS / "runs" / "hstu-sid").glob("trial[0-9][0-9].json"))]


def mean_std(values: list[float]) -> dict:
    out = {"mean": statistics.mean(values), "n": len(values), "values": values}
    out["std"] = statistics.stdev(values) if len(values) > 1 else None
    return out


def table(tests: list[dict]) -> dict:
    """Mean and sample standard deviation over seeds of every part and metric of the given test results."""
    return {
        part: {
            m: mean_std([t[part][m] for t in tests]) if all(t[part] for t in tests) else None for m in METRICS
        }
        for part in PARTS
    }


def finals() -> dict:
    out = {}
    for model, seeds in FINAL.items():
        recs = [r for r in (record(model, s) for s in seeds) if r]
        if not recs:
            continue
        out[model] = {
            "trial": recs[0]["trial"],
            "config": recs[0]["config"],
            "seeds": [r["seed"] for r in recs],
            "test": table([r["test"] for r in recs]),
            "best_epochs": [r["best_epoch"] for r in recs],
            "epochs_run": [r["epochs_run"] for r in recs],
            "valid_ndcg@10": [r["valid"]["all"]["ndcg@10"] for r in recs],
            "params": recs[0]["params"],
            "wall_time_s": [r["wall_time_s"] for r in recs],
            "processes": [len(r["processes"]) for r in recs],
            "src_trees": sorted({p["git"]["src_tree"] for r in recs for p in r["processes"]}),
            "dirty": any(p["git"]["dirty"] for r in recs for p in r["processes"]),
            "test_seconds": [r["test"]["seconds"] for r in recs],
        }
        if model == "tiger":
            eps = recs[0]["fixed"]["cold_start_eps"]
            out["tiger-cold-start"] = {
                "eps": eps,
                "prefix": recs[0]["fixed"]["cold_start_prefix"],
                "seeds": [r["seed"] for r in recs],
                "test": table([r["test"]["cold_start"][str(eps)] for r in recs]),
                "stats": [r["test"]["cold_start_stats"] for r in recs],
            }
            out[model]["empty_slots"] = [r["test"]["empty_slots"] for r in recs]
    return out


def ablation(final: dict) -> dict:
    a, h = record("hstu-notime", 0), record("hstu", 0)
    if not (a and h):
        return {}
    return {
        "hstu_notime_seed0": {p: a["test"][p] for p in PARTS},
        "hstu_seed0": {p: h["test"][p] for p in PARTS},
        "hstu_mean": final.get("hstu", {}).get("test"),
        "sasrec_mean": final.get("sasrec", {}).get("test"),
        "best_epochs": {"hstu-notime": a["best_epoch"], "hstu": h["best_epoch"]},
    }


def hstu_sid_tuning() -> dict:
    runs = tuning_records()
    if not runs:
        return {}
    trials = [
        {
            "trial": r["trial"],
            "config": r["config"],
            "valid_ndcg@10": r["valid"]["all"]["ndcg@10"],
            "valid_recall@50": r["valid"]["all"]["recall@50"],
            "valid_warm_ndcg@10": r["valid"]["warm"]["ndcg@10"],
            "valid_cold_recall@50": (r["valid"]["cold"] or {}).get("recall@50"),
            "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"],
            "wall_time_s": r["wall_time_s"],
            "prefix_table_rows": r["prefix_table_rows"],
        }
        for r in runs
    ]
    hstu = {t: dev_trial("hstu", t)["valid"]["all"]["ndcg@10"] for t in range(1, 13)}
    out = {
        "trials": trials,
        "hstu_valid_ndcg@10": hstu,
        "gpu_hours": round(sum(t["wall_time_s"] for t in trials) / 3600, 2),
    }
    if len(runs) == 12:
        best = selected("hstu-sid")
        out["selected"] = {k: best[k] for k in ("trial", "config", "best_epoch", "valid")}
    return out


def eps_curve() -> dict:
    recs = [load(p) for p in sorted((RESULTS / "tiger_eps").glob("seed*.json"))]
    if not recs:
        return {}
    eps = recs[0]["eps"]
    return {
        "split": "valid, the 100,000-row sample",
        "seeds": [r["seed"] for r in recs],
        "eps": eps,
        "metrics": {str(e): table([r["metrics"]["cold_start"][str(e)] for r in recs]) for e in eps},
        "stats": [r["metrics"]["cold_start_stats"] for r in recs],
    }


def gpu_hours() -> dict:
    groups = {
        "final_runs": [p for p in (RESULTS / "runs").glob("*/seed*.json")],
        "hstu_sid_tuning": [p for p in (RESULTS / "runs" / "hstu-sid").glob("trial*.json")],
        "tiger_eps_curve": [p for p in (RESULTS / "tiger_eps").glob("seed*.json")],
    }
    out = {k: round(sum(load(p)["wall_time_s"] for p in v) / 3600, 2) for k, v in groups.items()}
    out["total"] = round(sum(out.values()), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--projection", action="store_true")
    args = parser.parse_args()
    if args.projection:
        p = projection()
        common.write_json(RESULTS / "projection.json", p)
        print(p["total_hours"], p["bound_at_30_epochs_hours"], p["within_limit"])
        return
    final = finals()
    summary = {
        "final": final,
        "time_ablation": ablation(final),
        "hstu_sid_tuning": hstu_sid_tuning(),
        "tiger_eps_curve_valid": eps_curve(),
        "gpu_hours": gpu_hours(),
        "projection_now": projection(),
    }
    common.write_json(RESULTS / "summary.json", summary)
    print(json.dumps(summary["gpu_hours"]))


if __name__ == "__main__":
    main()
