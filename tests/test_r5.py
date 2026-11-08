import numpy as np
import pytest
import torch
from test_r4b import random_ids, sid_for
from test_scale import toy  # the two-user toy timelines

from sidrec.data import scale
from sidrec.eval import r5_serving as r5
from sidrec.serve import loadgen
from sidrec.serve.machine import parse
from sidrec.serve.ranker import Graphed, HistoryCache, RankerStep, bucket
from sidrec.serve.requests import WINDOW, build_requests, history_mask, query_times, tiger_input
from sidrec.train.scale_idrec import build
from sidrec.train.scale_tiger import Inputs


def test_requests_give_the_inputs_r4c_cuts_from_the_timelines():
    d = toy()
    codes, layout = random_ids(10)
    rows = np.array([2, 0, 1])
    req = build_requests(d, "valid", rows, layout.user_offset)
    ex = scale.Examples(
        torch.as_tensor(d["valid_target"][rows]), torch.as_tensor(d["offsets"][d["valid_user"][rows]])
    )
    flat = scale.Flat(d, "cpu")
    b = flat.id_inputs(ex, torch.arange(11), WINDOW, train=False)
    assert torch.equal(req.items, b["inputs"]) and torch.equal(req.ts, b["input_ts"])
    assert torch.equal(query_times(req.items, req.ts, req.req_ts), b["query_ts"])
    assert torch.equal(req.target, b["target"])
    inp = Inputs(d, layout.item_tokens(codes), layout, "cpu")
    users = torch.as_tensor(d["valid_user"][rows].astype(np.int64))
    assert torch.equal(tiger_input(req.items, req.user_tok, inp.tokens), inp.encoder(ex, users))
    assert torch.equal(history_mask(req.items), inp.history(ex))
    assert req.cold.tolist() == d["valid_cold"][rows].tolist()


def random_model(width=8, layers=2, seed=0):
    d = toy()
    sid = sid_for(d)
    torch.manual_seed(seed)
    model = build("hstu-sid", {"width": width, "layers": layers, "dropout": 0.0}, sid["v"], sid).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
    return model


def random_batch(b=5, n_items=10, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.tensor([1, 3, 17, 49, 50, 50, 8, 2][:b])
    items = torch.randint(1, n_items + 1, (b, WINDOW), generator=g)
    ts = torch.cumsum(torch.randint(1, 100_000, (b, WINDOW), generator=g), dim=1)
    pad = torch.arange(WINDOW)[None, :] < (WINDOW - lengths)[:, None]
    items, ts = items.masked_fill(pad, 0), ts.masked_fill(pad, 0)
    req_ts = ts[:, -1] + torch.randint(1, 10**7, (b,), generator=g)
    cands = torch.randint(1, n_items + 1, (b, 200), generator=g)
    cands[0, 5] = -1  # an empty slot
    return {k: v.to(device) for k, v in dict(items=items, ts=ts, req_ts=req_ts, cands=cands).items()}


def test_full_encoder_matches_the_model_and_r4c_scores():
    model = random_model()
    t = random_batch()
    step = RankerStep(model)
    top, scores = step(**t)
    q = query_times(t["items"], t["ts"], t["req_ts"])
    state = model.encode(t["items"], input_ts=t["ts"], query_ts=q)[:, -1]
    table = model.catalog_table()  # R4c: column j is catalog item j + 1
    want = (state @ table.T).gather(1, (t["cands"] - 1).clamp(min=0))
    valid = t["cands"] > 0
    torch.testing.assert_close(scores[valid], want[valid], rtol=1e-5, atol=1e-5)
    assert scores[0, 5] == -float("inf")
    assert torch.equal(top, t["cands"].gather(1, scores.topk(10, dim=1).indices))


def test_a_cache_hit_gives_the_miss_path_state_and_ignores_the_window_rows_it_does_not_need():
    model = random_model()
    t = random_batch()
    cache = HistoryCache(8, 2, model.d, torch.float32, "cpu")
    miss = RankerStep(model, mode="miss", cache=cache)
    slots = torch.tensor([3, 0, 7, 1, 5])
    _, want = miss(**t, slots=slots)
    hit = miss.with_mode("hit")
    _, got = hit(slots=slots, item=t["items"][:, -1], item_ts=t["ts"][:, -1], req_ts=t["req_ts"], cands=t["cands"])
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    # The cached rows do not depend on the request time: a later request reuses them.
    later = t["req_ts"] + 86_400 * 30
    _, again = hit(slots=slots, item=t["items"][:, -1], item_ts=t["ts"][:, -1], req_ts=later, cands=t["cands"])
    _, fresh = RankerStep(model)(items=t["items"], ts=t["ts"], req_ts=later, cands=t["cands"])
    torch.testing.assert_close(again, fresh, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(again, got)


def test_bf16_stays_close_to_fp32():
    model = random_model()
    t = random_batch()
    _, s32 = RankerStep(model)(**t)
    _, s16 = RankerStep(model, torch.bfloat16)(**t)
    valid = t["cands"] > 0
    assert (s16[valid] - s32[valid]).abs().max() < 0.1 * s32[valid].abs().max()


def test_modes_need_a_cache():
    with pytest.raises(ValueError):
        RankerStep(random_model(), mode="hit")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_graphs_pad_a_batch_and_give_the_eager_results():
    model = random_model().cuda()
    t = random_batch(device="cuda")
    cache = HistoryCache(8, 2, model.d, torch.float32, "cuda")
    step = RankerStep(model, mode="miss", cache=cache)
    t["slots"] = torch.tensor([3, 0, 7, 1, 5], device="cuda")
    graphed = Graphed(step, {k: v[:1] for k, v in t.items()}, max_batch=8)
    assert sorted(graphed.graphs) == [1, 2, 4, 8]
    top, scores = graphed(**{k: v[:3] for k, v in t.items()})  # padded to 4
    want_top, want = RankerStep(model)(**{k: v[:3] for k, v in t.items() if k != "slots"})
    torch.testing.assert_close(scores, want, rtol=1e-5, atol=1e-5)
    assert torch.equal(top, want_top)
    assert not cache.valid[[2, 4, 6]].any()  # padded rows write only the dummy slot
    assert torch.equal(cache.valid[[3, 0, 7]], t["items"][:3, :-1] > 0)


def test_bucket_is_the_next_power_of_two():
    assert [bucket(b) for b in (1, 2, 3, 4, 5, 1000, 1024)] == [1, 2, 4, 4, 8, 1024, 1024]


class VirtualClock:
    """A clock that only moves when the server waits or a step runs."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def wait(self, t: float) -> None:
        self.t = max(self.t, t)


def test_open_loop_counts_queueing_from_the_scheduled_arrival():
    clock = VirtualClock()

    def step(req):
        clock.t += 0.010  # 10 ms per batch, whatever its size

    arrivals = np.array([0.0, 0.001, 0.002, 0.050])
    res = loadgen.serve(step, arrivals, max_batch=1, window=0.0, clock=clock, wait=clock.wait)
    # One at a time: the second waits for the first, the third for both.
    np.testing.assert_allclose(res["latency"], [0.010, 0.019, 0.028, 0.010])
    assert res["size"].tolist() == [1, 1, 1, 1]


def test_dynamic_batching_waits_the_window_or_until_the_batch_is_full():
    clock = VirtualClock()

    def step(req):
        clock.t += 0.004

    arrivals = np.array([0.0, 0.0005, 0.0010, 0.0011, 0.0012, 0.0100])
    res = loadgen.serve(step, arrivals, max_batch=4, window=0.002, clock=clock, wait=clock.wait)
    # The first four fill a batch at 1.1 ms; the fifth waits for the server, by then past its window; the last
    # waits its own window.
    assert res["size"].tolist() == [4, 1, 1]
    np.testing.assert_allclose(res["begin"], [0.0011, 0.0051, 0.012])
    np.testing.assert_allclose(res["latency"][[0, 4, 5]], [0.0051, 0.0079, 0.006])


def test_an_overloaded_run_aborts():
    clock = VirtualClock()

    def step(req):
        clock.t += 0.5

    res = loadgen.serve(step, np.linspace(0, 1, 50), max_batch=1, window=0.0, abort_age=1.0, clock=clock, wait=clock.wait)
    assert res["aborted"] and np.isnan(res["latency"]).any()


def test_poisson_arrivals_have_the_rate_and_repeat_with_the_seed():
    a = loadgen.poisson_arrivals(1000.0, 20.0, seed=3)
    assert np.all(np.diff(a) > 0) and a[-1] < 20.0
    assert abs(len(a) / 20.0 - 1000.0) < 30
    np.testing.assert_array_equal(a, loadgen.poisson_arrivals(1000.0, 20.0, seed=3))


def test_bisection_and_confirmation():
    lo, hist = loadgen.bisect_rate(lambda r, i: {"pass": r <= 37.3}, 128.0, 7)
    assert lo <= 37.3 < lo + 1.0 and len(hist) == 7
    lo, _ = loadgen.bisect_rate(lambda r, i: {"pass": False}, 128.0, 7)
    assert lo == 0.0
    rate, attempts = loadgen.confirm_rate(lambda r, a, j: {"pass": r <= 90.0}, 100.0, 3, 0.95, 4)
    assert rate == pytest.approx(100 * 0.95**3) and len(attempts) == 4
    rate, attempts = loadgen.confirm_rate(lambda r, a, j: {"pass": False}, 100.0, 3, 0.95, 4)
    assert rate is None and len(attempts) == 5


def test_max_batch_is_the_largest_size_within_the_limit():
    times = {1: 0.002, 2: 0.003, 4: 0.006, 8: 0.012, 16: 0.025, 32: 0.010}
    best, sweep = loadgen.choose_max_batch(lambda s: np.full(50, times[s]), 0.050 / 3)
    assert best == 8 and [s["size"] for s in sweep] == [1, 2, 4, 8, 16]  # stops at the first size above
    best, _ = loadgen.choose_max_batch(lambda s: np.full(50, 0.02), 0.050 / 3)
    assert best == 1


def test_open_loop_run_summarizes_the_measured_requests_only():
    clock_step = 0.0001

    def step(req):
        t = loadgen.time.perf_counter()
        while loadgen.time.perf_counter() - t < clock_step:
            pass
        return {"beam_ms": 1.0, "batch_ms": 2.0}

    s, raw = loadgen.open_loop_run(step, 2000.0, 8, 0.0005, 0.050, 0.2, 0.5, seed=1, min_n=100)
    assert s["pass"] and s["measured"] == len(raw["latency_ms"]) and s["p99_ms"] < 50
    assert s["measured"] < 2000.0 * 0.9  # the warm-up arrivals are left out
    assert set(raw) >= {"latency_ms", "wait_ms", "batch_size", "service_ms", "beam_ms", "batch_ms"}
    assert sum(s["histogram"]["counts"]) == s["measured"]


def test_top10_breaks_ties_by_the_lower_catalog_index():
    cands = torch.tensor([[9, 4, 7, 2, 5, 6, 8, 3, 11, 12, 13, 14]])
    scores = torch.tensor([[1.0, 1.0, 1.0, 0.5, 2.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.3, -1.0]])
    assert r5.top10(scores, cands).tolist() == [[5, 4, 7, 9, 2, 11, 12, 13, 3, 6]]
    assert float(r5.overlap10(r5.top10(scores, cands), torch.arange(1, 11)[None])) == 0.7


def test_a_failed_step_is_left_out_of_the_later_steps():
    plan = r5.ladder_plan({})
    assert plan["batching-hit"] == {
        "name": "batching-hit", "dtype": "bf16", "graphs": True, "cache": "hit", "batching": True
    }
    plan = r5.ladder_plan({"bf16": False, "cache-hit": False})
    assert plan["bf16"]["dtype"] == "bf16" and plan["graphs"]["dtype"] == "fp32"
    assert plan["batching-miss"]["cache"] is None and plan["batching-hit"] is None


def test_the_sidecar_lines_are_summarized():
    lines = [
        "L 1.0 6.10 5.00 3.00 4/900 123",
        "G 1.0 2520, 10501, 60.5, 45, 30, 4000, P2",
        "A 1.0 4242, python, 3000 MiB;777, other, 100 MiB;",
        "L 3.0 7.10 5.10 3.10 9/900 124",
    ]
    s = parse(lines, own_pid=4242)
    assert s["load1"]["max"] == 7.1 and s["running_tasks"]["max"] == 9
    assert s["clocks.sm"]["median"] == 2520 and s["pstate"] == ["P2"]
    assert s["other_gpu_processes"] == ["other (100 MiB)"]


def test_the_summary_reads_the_confirmed_runs():
    from sidrec.eval import r5_summary

    stage = {k: {"p50": 1.0, "p99": 2.0, "mean": 1.0, "share": s} for k, s in
             (("beam_ms", 0.7), ("hnsw_ms", 0.2), ("union_ms", 0.01), ("ranker_ms", 0.05), ("batch_ms", 1.0))}
    run = {"pass": True, "p50_ms": 9.0, "p99_ms": 30.0, "mean_batch": 2.0, "stages": stage,
           "wait": {"p50_ms": 1.0, "p99_ms": 12.0}}
    rec = {
        "qps_at_slo": 95.0,
        "confirmation": [{"rate": 100.0, "runs": [run | {"pass": False}] * 3}, {"rate": 95.0, "runs": [run] * 3}],
        "at_slo": {"p50_ms": [9.0] * 3, "p99_ms": [30.0, 31.0, 29.0], "p50_ms_median": 9.0, "p99_ms_median": 30.0,
                   "mean_batch": [2.0] * 3},
    }
    assert r5_summary.at_slo(rec)["confirmation_attempts"] == 2
    assert r5_summary.at_slo({"qps_at_slo": None}) == {"qps_at_slo": None}
    load = r5_summary.under_load(rec)
    assert load["dominant"] == "beam_ms" and load["wait_p99_ms"] == 12.0
