import json

import pytest

from sidrec.eval import r2_summary


def write(path, **record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def test_summary_means_gaps_and_the_fifteen_percent_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(r2_summary, "RESULTS", tmp_path)
    runs = tmp_path / "runs" / "sasrec"
    env = {"device": "GPU"}
    write(runs / "trial03.json", phase="tune", trial=3, wall_time_s=100.0, environment=env)
    for seed, r10 in enumerate((0.05, 0.06, 0.07)):
        test = {"recall@5": 0.03, "ndcg@5": 0.02, "recall@10": r10, "ndcg@10": 0.0318}
        record = dict(
            phase="final",
            trial=3,
            seed=seed,
            config={},
            best_epoch=5,
            test=test,
            valid={"ndcg@10": 0.1},
            wall_time_s=50.0,
            environment=env,
        )
        if seed == 0:
            record["test_eval"] = {"wall_time_s": 2.0}
        write(runs / f"trial03_seed{seed}.json", **record)
    s = r2_summary.summarize()["sasrec"]
    assert s["test"]["recall@10"]["mean"] == pytest.approx(0.06)
    assert s["test"]["recall@10"]["sd"] == pytest.approx(0.01)
    # TIGER Table 1 SASRec: R@10 0.0605, N@10 0.0318.
    assert s["relative_gap"]["sasrec_tiger_table1"]["recall@10"] == pytest.approx(0.06 / 0.0605 - 1)
    assert s["relative_gap"]["sasrec_tiger_table1"]["ndcg@10"] == pytest.approx(0.0)
    assert s["fifteen_percent_rule"] == []
    assert r2_summary.gpu_seconds([json.loads(p.read_text()) for p in runs.glob("*.json")]) == 202.0
    write(
        runs / "trial03_seed2.json",
        phase="final",
        trial=3,
        seed=2,
        config={},
        best_epoch=5,
        test={"recall@5": 0.0, "ndcg@5": 0.0, "recall@10": 0.0, "ndcg@10": 0.0},
        valid={"ndcg@10": 0.1},
        wall_time_s=50.0,
        environment=env,
    )
    assert r2_summary.summarize()["sasrec"]["fifteen_percent_rule"] == [
        "sasrec_tiger_table1 recall@10",
        "sasrec_tiger_table1 ndcg@10",
    ]
