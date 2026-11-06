import numpy as np
import pytest
import torch

from sidrec.data.loo import Beauty, id_model_arrays
from sidrec.models.hstu import HSTU, time_buckets
from sidrec.models.sasrec import SASRec
from sidrec.tuning import PUBLISHED, shared_trials, trials


def tiny() -> Beauty:
    seqs = [np.array([1, 2, 3, 4, 5]), np.array([3, 1, 4, 1, 5, 9, 2])]
    times = [np.arange(5) * 10, np.arange(7) * 100]
    return Beauty(9, [f"i{i}" for i in range(1, 10)], ["u1", "u2"], seqs, times)


def test_leave_one_out_windows():
    data = tiny()
    tr = id_model_arrays(data, "train", max_len=3)
    # u1: training part 1 2 3, predicts 2 3 from 1 2; u2: training part 3 1 4 1 5, last window predicts 4 1 5.
    assert tr["inputs"].tolist() == [[0, 1, 2], [1, 4, 1]]
    assert tr["targets"].tolist() == [[0, 2, 3], [4, 1, 5]]
    assert tr["query_ts"].tolist() == [[0, 10, 20], [200, 300, 400]]
    va = id_model_arrays(data, "valid", max_len=3)
    assert va["inputs"].tolist() == [[1, 2, 3], [4, 1, 5]] and va["target"].tolist() == [4, 9]
    te = id_model_arrays(data, "test", max_len=3)
    assert te["inputs"].tolist() == [[2, 3, 4], [1, 5, 9]] and te["target"].tolist() == [5, 2]
    assert te["query_ts"][:, -1].tolist() == [40, 600]  # the target's timestamp


@pytest.mark.parametrize("cls", [SASRec, HSTU])
def test_states_are_causal(cls):
    torch.manual_seed(0)
    model = cls(20, 32, 2, 0.0, max_len=6).eval()
    inputs = torch.tensor([[0, 0, 3, 4, 5, 6]])
    ts = torch.tensor([[0, 0, 10, 20, 30, 40]])
    qts = torch.tensor([[0, 0, 20, 30, 40, 50]])
    base = model.encode(inputs, input_ts=ts, query_ts=qts)
    later = inputs.clone()
    later[0, 5] = 7
    changed = model.encode(later, input_ts=ts, query_ts=qts)
    assert torch.allclose(base[0, 2:5], changed[0, 2:5], atol=1e-6)
    assert not torch.allclose(base[0, 5], changed[0, 5])
    assert model.logits(base).shape == (1, 6, 21)


def test_hstu_time_buckets_are_log_scaled():
    dt = torch.tensor([0, 1, 2, 86400, -86400, 10**12])
    assert time_buckets(dt, 128).tolist() == [0, 0, 2, 37, 37, 91]


def test_trial_list_is_fixed_and_trial_one_is_the_published_setting():
    shared = shared_trials()
    assert len(shared) == 12 and len({tuple(t.values()) for t in shared}) == 12
    assert shared[1] == {"lr": 3e-4, "dropout": 0.0, "weight_decay": 0.0, "width": 256, "layers": 4}
    for model in ("sasrec", "hstu", "tiger"):
        t = trials(model)
        assert t[0] == PUBLISHED[model] and t[1:] == shared[1:]


def test_hstu_without_time_bias_ignores_timestamps():
    torch.manual_seed(0)
    model = HSTU(20, 32, 2, 0.0, max_len=4, time_bias=False).eval()
    inputs = torch.tensor([[0, 3, 4, 5]])
    a = model.encode(inputs, input_ts=torch.tensor([[0, 1, 2, 3]]), query_ts=torch.tensor([[1, 2, 3, 4]]))
    b = model.encode(
        inputs, input_ts=torch.tensor([[0, 9, 10**6, 10**8]]), query_ts=torch.tensor([[5, 7, 9, 10**9]])
    )
    assert torch.allclose(a, b)
