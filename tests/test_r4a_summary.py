import pytest

from sidrec.eval.r4a_summary import MODELS, estimates


def row(model: str, source: str, trial: int, train_s: float, eval_s: float, all_eval_s: float) -> dict:
    return {
        "model": model,
        "source": source,
        "trial": trial,
        "epoch_train_s": train_s,
        "sample_eval_s": eval_s,
        "all_valid_eval_s": all_eval_s,
    }


def test_gpu_hour_estimates_follow_the_documented_formulas():
    rows = [row(m, "dev", 1, 30.0, 6.0, 6.0) for m in MODELS]
    rows += [row(m, "full", 1, 600.0, 60.0, 1_141_362 / 1_199_539 * 360.0) for m in MODELS]
    est = estimates(rows)
    # 12 trials x 30 epochs x 36 s = 3.6 h per model.
    assert est["tuning"]["trial1_config_30_epochs"] == {
        "sasrec": 3.6,
        "hstu": 3.6,
        "tiger": 3.6,
        "total": 10.8,
    }
    # 3 seeds x (10 epochs x 660 s + 360 s of test evaluation) = 5.8 h; the cascade is 3 x 3 x 360 s = 0.9 h.
    r4b = est["r4b"]["trial1_config_10_epochs"]
    assert r4b["sasrec"] == pytest.approx(5.8) and r4b["cascade"] == pytest.approx(0.9)
    assert r4b["total"] == pytest.approx(3 * 5.8 + 0.9)
