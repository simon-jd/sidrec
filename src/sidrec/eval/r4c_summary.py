"""The R4c summary: results/r4c/summary.json from the records in results/r4c, and the README's tables.

    python -m sidrec.eval.r4c_summary            # writes summary.json and prints the tables

Test numbers are mean ± sample standard deviation over seeds 0, 1 and 2. The cascade's change against R4b's
full-catalog ranking is taken per seed (same ranker checkpoint) and then averaged.
"""

from __future__ import annotations

import json

from sidrec.eval.r4b_summary import METRICS, PARTS, mean_std, table
from sidrec.train import common

RESULTS = common.ROOT / "results" / "r4c"
R4B = common.ROOT / "results" / "r4b"
SEEDS = (0, 1, 2)
RANKERS = ("hstu", "hstu-sid")
BUDGETS = ("50", "100", "200")
NAMES = {"hstu": "HSTU-style", "hstu-sid": "HSTU-SID"}


def load(path) -> dict:
    return json.loads(path.read_text())


def records(kind: str) -> list[dict]:
    """The per-seed records of one kind: test/seedS, valid/tiger_seedS or valid/rerank_seedS."""
    folder, prefix = kind.split("/")
    out = [RESULTS / folder / f"{prefix}{s}.json" for s in SEEDS]
    return [load(p) for p in out if p.exists()]


def retrieval_table(recalls: list[dict]) -> dict:
    """{K: {part: mean_std}} over seeds of {K: {part: recall}}."""
    return {
        k: {
            p: mean_std([r[k][p] for r in recalls]) if all(r[k][p] is not None for r in recalls) else None
            for p in PARTS
        }
        for k in BUDGETS
    }


def r4b_tests(name: str) -> list[dict]:
    return [load(R4B / "runs" / name / f"seed{s}.json")["test"] for s in SEEDS]


def paired_change(cascade: list[dict], full: list[dict]) -> dict:
    """Per-seed cascade minus full-catalog, averaged; and the relative change of the means."""
    out = {}
    for part in PARTS:
        out[part] = {}
        for m in METRICS:
            if not all(c[part] for c in cascade) or not all(f[part] for f in full):
                out[part][m] = None
                continue
            diff = [c[part][m] - f[part][m] for c, f in zip(cascade, full, strict=True)]
            base = sum(f[part][m] for f in full) / len(full)
            out[part][m] = mean_std(diff) | {"relative": (sum(diff) / len(diff)) / base if base else None}
    return out


def gpu_hours() -> dict:
    """Wall time of every recorded GPU process, which held the RTX 4090 lock throughout."""
    parts = {
        "beam_valid": sum(r["wall_time_s"] for r in records("valid/tiger_seed")),
        "rerank_valid": sum(r["wall_time_s"] for r in records("valid/rerank_seed")),
        "test": sum(r["wall_time_s"] for r in records("test/seed")),
    }
    return {k: round(v / 3600, 3) for k, v in parts.items()} | {"total": round(sum(parts.values()) / 3600, 3)}


def cpu_steps() -> dict:
    build = load(RESULTS / "ann" / "build.json")
    out = {"index_build_s": build["build_seconds"], "threads": build["threads"]}
    if (RESULTS / "ann" / "valid.json").exists():
        out["ann_select_s"] = load(RESULTS / "ann" / "valid.json")["wall_time_s"]
    if (RESULTS / "ann" / "test.json").exists():
        out["ann_test_search_s"] = load(RESULTS / "ann" / "test.json")["search_seconds"]
    return out


def validation() -> dict:
    ann = load(RESULTS / "ann" / "valid.json")
    final = ann["rounds"][-1]
    out = {
        "ann": {
            "efSearch_rounds": [r["efSearch"] for r in ann["rounds"]],
            "selected": ann["selected"],
            "recall_vs_exact@200": ann["recall_vs_exact@200"],
            "target_recall": final["target_recall"],
            "grid": [
                {"n": g["n"], "gamma": g["gamma"], "recall@200": g["recall"]["200"]["all"]}
                for g in final["grid"]
            ],
        },
    }
    if (RESULTS / "selection.json").exists():
        sel = load(RESULTS / "selection.json")
        out["union"] = {
            "selected": sel["union"],
            "grid": [{k: g[k] for k in ("rule", "k_tiger", "backfill", "mean")} for g in sel["union_grid"]],
            "section_2_setting": sel["section_2_setting"]["mean"],
        }
    beams = records("valid/tiger_seed")
    if beams:
        out["tiger_beam200"] = {r: retrieval_table([b["tiger"][r] for b in beams]) for r in ("off", "on")}
    reranks = records("valid/rerank_seed")
    if reranks:
        out["rerank"] = {
            name: {
                "cascade": table([r["rankers"][name]["cascade"] for r in reranks]),
                "full_catalog": table([r["rankers"][name]["full_catalog"] for r in reranks]),
                "full_catalog_minus_r4b_record_ndcg@10": [
                    r["rankers"][name]["full_catalog_ndcg@10_minus_record"] for r in reranks
                ],
                "rows_cascade_rank_above_full_rank": [
                    r["rankers"][name]["rows_cascade_rank_above_full_rank"] for r in reranks
                ],
            }
            for name in RANKERS
        }
    return out


def test_part() -> dict | None:
    recs = records("test/seed")
    if len(recs) != len(SEEDS):
        return None
    sources = ["tiger", "ann", "union"] + (
        ["tiger_plain_beam"] if "tiger_plain_beam" in recs[0]["retrieval"] else []
    )
    out = {
        "selection": recs[0]["selection"],
        "retrieval": {s: retrieval_table([r["retrieval"][s] for r in recs]) for s in sources},
        "union_stats": {k: mean_std([r["union_stats"][k] for r in recs]) for k in recs[0]["union_stats"]},
        "rerank": {},
    }
    for name in RANKERS:
        cascade = [{p: r["rerank"][name][p] for p in PARTS} for r in recs]
        full = r4b_tests(name)
        out["rerank"][name] = {
            "cascade": table(cascade),
            "r4b_full_catalog": table(full),
            "change": paired_change(cascade, full),
        }
    return out


def summary() -> dict:
    return {
        "protocol": "docs/protocol.md, 2026-11-07: R4c two-stage cascade",
        "validation": validation(),
        "projection": load(RESULTS / "projection.json") if (RESULTS / "projection.json").exists() else None,
        "test": test_part(),
        "gpu_hours": gpu_hours(),
        "cpu_steps": cpu_steps(),
        "records": {
            "test": records("test/seed"),
            "valid_beam": records("valid/tiger_seed"),
            "valid_rerank": records("valid/rerank_seed"),
            "ann_valid": load(RESULTS / "ann" / "valid.json"),
            "ann_build": load(RESULTS / "ann" / "build.json"),
        },
    }


def fmt(x: dict | None, digits: int = 5) -> str:
    if x is None:
        return "n/a"
    if x.get("std") is None:
        return f"{x['mean']:.{digits}f}"
    return f"{x['mean']:.{digits}f} ± {x['std']:.{digits}f}"


def markdown(s: dict) -> str:
    t = s["test"]
    lines = []
    names = {
        "tiger": "TIGER alone",
        "tiger_plain_beam": "TIGER plain beam",
        "ann": "ANN alone",
        "union": "Union",
    }
    for part in PARTS:
        lines += [f"\nRetrieval recall, {part} rows", "| Source | @50 | @100 | @200 |", "|---|---|---|---|"]
        for src, tab in t["retrieval"].items():
            lines.append(f"| {names[src]} | " + " | ".join(fmt(tab[k][part]) for k in BUDGETS) + " |")
    for part in PARTS:
        lines += [
            f"\nRe-ranked, {part} rows",
            "| Ranker | Candidates | Recall@10 | Recall@50 | NDCG@10 |",
            "|---|---|---|---|---|",
        ]
        for name in RANKERS:
            r = t["rerank"][name]
            for label, tab in (("full catalog (R4b)", r["r4b_full_catalog"]), ("cascade, 200", r["cascade"])):
                lines.append(
                    f"| {NAMES[name]} | {label} | " + " | ".join(fmt(tab[part][m]) for m in METRICS) + " |"
                )
            ch = r["change"][part]
            lines.append(
                f"| {NAMES[name]} | change | "
                + " | ".join(
                    "n/a"
                    if ch[m] is None
                    else f"{fmt(ch[m])} ({100 * ch[m]['relative']:+.1f}%)"
                    if ch[m]["relative"] is not None
                    else fmt(ch[m])
                    for m in METRICS
                )
                + " |"
            )
    return "\n".join(lines)


def main() -> None:
    s = summary()
    common.write_json(RESULTS / "summary.json", s)
    if s["test"]:
        print(markdown(s))
    print(json.dumps({"gpu_hours": s["gpu_hours"], "cpu_steps": s["cpu_steps"]}))


if __name__ == "__main__":
    main()
