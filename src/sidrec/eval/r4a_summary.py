"""Build results/r4a/summary.json from the R4a records: the cost table, the tuning trials and the DDP study.

    python -m sidrec.eval.r4a_summary

GPU-hour estimates (RTX 4090), from the cost records:
- one epoch = the measured training epoch plus one evaluation of the validation sample (100,000 rows on the
  full split, all 71,610 rows on the dev subset);
- one test evaluation = the timed evaluation of all 1,141,362 eligible validation rows, scaled to the
  1,199,539 test rows;
- R4a tuning = 12 trials x epochs x the dev-subset epoch, per model;
- R4b = 3 seeds x (epochs x the full-split epoch + one test evaluation) per model, plus the cascade: per seed,
  TIGER's beam over the test rows (generative retrieval), a SASRec scoring (standing in for ANN retrieval) and
  an HSTU-style scoring (standing in for the ranker), each costed as that model's test evaluation.
"""

from __future__ import annotations

import json

from sidrec.train import common
from sidrec.train.scale_idrec import MAX_EPOCHS

RESULTS = common.ROOT / "results" / "r4a"
MODELS = ("sasrec", "hstu", "tiger")
TEST_ROWS, VALID_ROWS = 1_199_539, 1_141_362


def load(path):
    return json.loads(path.read_text())


def cost_rows() -> list[dict]:
    rows = []
    for path in sorted((RESULTS / "cost").glob("*.json")):
        r = load(path)
        h = r["history"][0]
        rows.append(
            {
                "model": r["model"],
                "source": r["dataset"].removeprefix("electronics-"),
                "trial": r["trial"],
                "width": r["config"]["width"],
                "layers": r["config"]["layers"],
                "params": r["params"],
                "steps_per_epoch": r["steps_per_epoch"],
                "epoch_train_s": h["train_s"],
                "ms_per_step": round(1e3 * h["train_s"] / r["steps_per_epoch"], 1),
                "sample_eval_rows": r["valid_rows"],
                "sample_eval_s": h["eval_s"],
                "all_valid_eval_rows": r["full_valid_eval"]["rows"],
                "all_valid_eval_s": r["full_valid_eval"]["seconds"],
                "peak_memory_gb": r["peak_memory_gb"],
                "sm_clock_median_mhz": r["gpu"]["sampled"].get("clocks.sm", {}).get("median"),
                "commit": r["git"]["commit"][:7],
            }
        )
    return rows


def estimates(rows: list[dict]) -> dict:
    by = {(r["model"], r["source"], r["trial"]): r for r in rows}

    def epoch_h(model, source, trial):
        r = by[(model, source, trial)]
        return (r["epoch_train_s"] + r["sample_eval_s"]) / 3600

    def test_h(model, trial):
        return by[(model, "full", trial)]["all_valid_eval_s"] * TEST_ROWS / VALID_ROWS / 3600

    out = {"tuning": {}, "r4b": {}}
    trials = sorted({k[2] for k in by})
    for trial in trials:
        if all((m, "dev", trial) in by for m in MODELS):
            per = {m: round(12 * MAX_EPOCHS * epoch_h(m, "dev", trial), 2) for m in MODELS}
            out["tuning"][f"trial{trial}_config_30_epochs"] = per | {"total": round(sum(per.values()), 2)}
        if all((m, "full", trial) in by for m in MODELS):
            for epochs in (10, MAX_EPOCHS):
                per = {
                    m: round(3 * (epochs * epoch_h(m, "full", trial) + test_h(m, trial)), 2) for m in MODELS
                }
                per["cascade"] = round(
                    3 * (test_h("tiger", trial) + test_h("sasrec", trial) + test_h("hstu", trial)), 2
                )
                out["r4b"][f"trial{trial}_config_{epochs}_epochs"] = per | {
                    "total": round(sum(per.values()), 2)
                }
    return out


def tuning() -> dict:
    out = {}
    for model in MODELS:
        runs = [load(p) for p in sorted((RESULTS / "runs" / model).glob("trial[0-9][0-9].json"))]
        runs = [r for r in runs if r["phase"] == "tune"]
        if not runs:
            continue
        trials = [
            {
                "trial": r["trial"],
                "config": r["config"],
                "valid_ndcg@10": r["valid"]["all"]["ndcg@10"],
                "valid_recall@10": r["valid"]["all"]["recall@10"],
                "valid_recall@50": r["valid"]["all"]["recall@50"],
                "best_epoch": r["best_epoch"],
                "epochs_run": r["epochs_run"],
                "wall_time_s": r["wall_time_s"],
            }
            for r in runs
        ]
        entry = {"trials": trials, "gpu_hours": round(sum(t["wall_time_s"] for t in trials) / 3600, 2)}
        if len(runs) == 12:
            best = max(runs, key=lambda r: (r["valid"]["all"]["ndcg@10"], -r["trial"]))
            entry["selected"] = {
                "trial": best["trial"],
                "config": best["config"],
                "best_epoch": best["best_epoch"],
                "valid": best["valid"],
                "checkpoint": best.get("checkpoint"),
            }
        out[model] = entry
    return out


def ddp() -> dict:
    path = RESULTS / "ddp" / "summary.json"
    return load(path) if path.exists() else {}


def main() -> None:
    rows = cost_rows()
    summary = {"cost": {"rows": rows, "gpu_hours": estimates(rows)}, "tuning": tuning(), "ddp": ddp()}
    common.write_json(RESULTS / "summary.json", summary)
    print(json.dumps(summary["cost"]["gpu_hours"], indent=1))
    for model, t in summary["tuning"].items():
        print(model, t["gpu_hours"], t.get("selected", {}).get("trial"))


if __name__ == "__main__":
    main()
