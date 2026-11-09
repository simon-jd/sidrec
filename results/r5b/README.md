# R5b: a fast TIGER beam, and the cascade re-measured

R5 found TIGER's beam to be most of the cascade's time. R5b makes the beam faster without changing what it
returns, then re-measures the cascade by R5's protocol.

Nothing is trained and no model changes. Test rows only supply request histories for timing, and no test
metric is computed. Every quality number below is on validation.

**Protocol:** [docs/protocol.md](../../docs/protocol.md), entries "2026-11-08: R5b fast TIGER beam" and
"2026-11-08: R5b near-ties at a beam cutoff".
- The entry was committed in fda5ee5, before any R5b measurement.
- The note was committed in f6019cb, after the fp32 equivalence run and before any timing.
- Everything not named in them follows R5's entry unchanged: the SLO, request pools, load generator, dynamic
  batching, B_max rule, bisection and confirmations.

**Numbers below** come from [summary.json](summary.json), which `python -m sidrec.eval.r5b_summary` builds from
the records in this folder. The "before" numbers are R5's committed records ([../r5/](../r5/README.md)), read
and not rerun.

## Headline
- **The beam is 4.2 to 5.0 times faster at batch 1:** 8.2 to 8.4 ms in R5, 1.7 to 2.0 ms now. It is 5 to 8
  times faster in batches of 64.
- **The cascade's QPS at the 50 ms p99 SLO rises 96 times at width 200,** from 19.1 to 1,825.
  - **Width 50:** from 45.5 to 1,877.
  - **Width 20:** from 209 to 1,956.
  - A single request now takes 2.0 to 2.3 ms (p50) through the whole cascade, against 10.1 to 10.3 ms.
- **The lists do not change beyond rounding.**
  - **In fp32:** 95.4% to 100% of lists are identical to the reference's, in the same order. Every mismatch is
    a near-tie of at most 1.2e-5. The reference's own lists change as much between batch sizes.
  - **In bf16, as served:** union recall@200 and HSTU-SID's NDCG@10 stay within 0.25% of R5's at every width.
- **The bottleneck has moved.** At widths 20 and 50, the HNSW search on the CPU now takes most of each batch's
  time under load. At width 200 the beam still does, at 90%.
- **The narrower beam no longer buys throughput.** All three widths now serve about 1,800 to 2,000 QPS, so width
  200 keeps its recall (union recall@200 0.1106, against 0.0606 at width 20) at almost no cost in QPS.

## Profile: where the reference beam's time goes
torch.profiler on R5's beam (bf16), on the first 64 test-pool requests. Wall time is the median of 20
synchronized calls without the profiler; the rest is per call under the profiler ([profile.json](profile.json)).

| Beam | Width | Batch | Wall ms | Kernel ms | GPU busy | Kernels | Launch calls |
|---|---|---|---|---|---|---|---|
| Reference | 20 | 1 | 8.04 | 2.50 | 31% | 1,267 | 1,267 |
| Reference | 20 | 4 (R5 B_max) | 10.74 | 3.15 | 29% | 1,260 | 1,260 |
| Reference | 50 | 1 | 10.69 | 2.66 | 25% | 1,278 | 1,278 |
| Reference | 200 | 1 | 11.30 | 3.96 | 35% | 1,331 | 1,331 |
| Reference | 20 | 64 | 21.0 | 19.2 | 91% | 1,258 | 1,258 |
| Reference | 50 | 64 | 46.1 | 44.0 | 96% | 1,312 | 1,312 |
| Reference | 200 | 64 | 176.7 | 174.9 | 99% | 1,312 | 1,312 |
| Final (graphs) | 20 | 1 | 1.70 | 1.63 | 96% | 1,046 | 1 |
| Final (graphs) | 200 | 1 | 2.03 | 1.93 | 95% | 1,094 | 1 |
| Final (graphs) | 200 | 64 | 21.8 | 21.5 | 99% | 1,099 | 1 |

- **At small batches the reference is launch-bound.**
  - The GPU is busy for only 25% to 35% of the wall time.
  - Each request launches about 1,300 kernels one by one from Python, 236 of them autocast's casts to bf16.
  - The batch-1 wall times of widths 50 and 200 here (10.7 and 11.3 ms) are higher than in the ladder below
    (8.3 and 8.4 ms). The profile ran while another user's load test held 3 to 5 cores.
- **There are no host syncs in the search:** no `.item()`, no `nonzero` and no synchronize inside it. The only
  synchronize is the benchmark's own, once per call.
- **The encoder is small.** It takes 0.3 ms of kernels at batch 1, against 1.9 to 3.3 ms for the four decoder
  steps and 0.2 to 0.3 ms for the rest of the search (top-k, trie gathers, history removal).
- **At larger batches the cross-attention dominates.** At width 200 and batch 64, it takes 134 of the decoder's
  169 ms. 101 ms of that is its keys and values, recomputed for each of the 200 beams at each of the four steps
  from the same encoder output.
- **The final beam is one graph launch per batch,** and the GPU is busy 95% to 99% of the time. It is now bound
  by the GPU running about 1,050 to 1,100 small kernels, at about 1.6 µs each at batch 1.

## Beam ladder (the beam alone, bf16, test pool)
Each step keeps the steps before it ([test/ladder/](test/ladder/)).

**Columns:**
- **Batch 1:** p50 / p99 of 2,000 requests one after another, after 200 untimed.
- **Batch 64:** requests per second over 50 back-to-back batches of 64, after 10 untimed.

| Step | Width 20: batch 1 ms | Batch 64 req/s | Width 50: batch 1 ms | Batch 64 req/s | Width 200: batch 1 ms | Batch 64 req/s |
|---|---|---|---|---|---|---|
| 1. Reference (R5's beam) | 8.31 / 9.24 | 3,106 | 8.26 / 8.91 | 1,415 | 8.44 / 9.80 | 363 |
| 2. Cross-attention once | 8.04 / 22.3 | 7,709 | 7.69 / 8.54 | 5,285 | 7.93 / 10.9 | 1,417 |
| 3. KV cache | 8.02 / 8.43 | 8,105 | 7.95 / 8.65 | 6,737 | 8.16 / 26.6 | 2,539 |
| 4. Static tensors | 7.04 / 19.5 | 8,963 | 6.95 / 7.42 | 7,194 | 6.90 / 7.29 | 2,684 |
| 5. CUDA graphs | **1.67 / 1.70** | **15,768** | **1.75 / 1.78** | **9,714** | **1.99 / 2.01** | **2,951** |

- **Steps 2 to 4 cut work, not launches,** so batch 1 barely moves. They matter in batches.
  - **Cross-attention once:** 2.5 to 3.9 times the throughput in batches of 64.
  - **KV cache:** each step then runs one token instead of the whole prefix. At width 200 that is 1.8 times more
    again.
  - **Static tensors:** autocast's casts and the position-bias kernels go. Batch 1 falls by about 1 ms.
- **CUDA graphs remove the launches:** 4.2 to 5.0 times faster at batch 1 than the reference, and 5.1 to 8.1
  times the throughput at batch 64.
- **The p99 values of 19 to 27 ms in steps 2 to 4 are single stalls.** Their p50 is unchanged. These eager steps
  ran while the other user's load test used 1 to 5 cores (below).
- **Step 1 matches R5:** 8.31 to 8.44 ms here, against 8.18 to 8.38 ms in R5's records.

## The lists: fp32 equivalence and bf16 quality
### fp32, against the reference (validation, the first 2,048 rows of the pool, TF32 off)
Identical lists, in the same order ([valid/equivalence.json](valid/equivalence.json)).

| Comparison | Width 20 | Width 50 | Width 200 |
|---|---|---|---|
| 2. Cross-attention once, chunks of 128 | 100% | 100% | 100% |
| 3. KV cache, chunks of 128 | 99.95% | 99.51% | 96.04% |
| 4. Static tensors, chunks of 128 | 100% | 99.66% | 95.36% |
| 5. CUDA graphs, chunks of 128 | 100% | 99.66% (7 rows) | 95.36% (95 rows) |
| 5. CUDA graphs, batch 1 | 100% | 99.41% (12 rows) | 96.14% (79 rows) |
| The reference itself, chunks of 128 against batch 1 | 100% | 99.51% (10 rows) | 94.92% (104 rows) |

- **Cross-attention once is exact:** the same lists and bitwise the same scores.
- **From the KV cache on, the arithmetic differs.** Matrix products of other shapes and attention over one
  query row round differently. Scores move by at most 3.1e-5 where the items agree.
- **Every mismatch is a near-tie** ([valid/equivalence_cutoffs.json](valid/equivalence_cutoffs.json)):
  - Most are two adjacent items with scores within 1e-5 that swap places: 6 of 7 rows at width 50, 93 of 95 at
    width 200, in chunks of 128.
  - The rest part through a near-tie at an earlier step's beam cutoff (steps 2 to 4). Each beam then keeps a
    different prefix, and the lists part where the final scores are far apart, by up to 0.062.
  - The cutoff trace finds the gap at the first step where the kept prefixes differ. The largest is 1.2e-5,
    under the 1e-4 bar. No row is a defect.
  - The entry's original rule looked only at the parting position. It would have counted 1 row at width 50 and
    1 to 2 rows at width 200 as defects. The note, written before any timing, added the trace.
- **For scale:** R4c's own beam changes as many lists between chunks of 128 and batch 1, with the same kinds of
  ties.

### bf16, as served: what the cascade's quality does (validation, 100,000 rows, chunks of 128)
From [valid/quality.json](valid/quality.json). R5's numbers are R4c's saved lists at width 200 and R5's lists
at widths 50 and 20.

| Width | Union recall@200, R5 → R5b | HSTU-SID NDCG@10, R5 → R5b | Lists identical to R5's | Items shared |
|---|---|---|---|---|
| 200 | 0.11049 → 0.11056 (+0.06%) | 0.014973 → 0.015009 (+0.24%) | 0% | 97.8% |
| 50 | 0.07733 → 0.07731 (-0.03%) | 0.014690 → 0.014687 (-0.02%) | 0% | 97.0% |
| 20 | 0.06073 → 0.06062 (-0.18%) | 0.014282 → 0.014269 (-0.09%) | 2.5% | 96.3% |

- **bf16 rounding reorders near-ties in almost every list,** as it did between R4c's and R5's batch sizes. R5's
  serving check found 0% identical and 97.4% shared at width 200.
- **Recall and NDCG move by at most 0.25%, in both directions.** Warm and cold recall move as little: cold
  recall@200 is 0.0184 at width 200 and 0.0407 at width 20, as in R5.
- **At batch 1 against R5's beam,** on the 2,048 rows: 0% to 2% of lists identical, 96.2% to 97.6% of items
  shared.

## Before and after: the cascade by R5's protocol (test requests)
| | Width 200: R5 | R5b | Width 50: R5 | R5b | Width 20: R5 | R5b |
|---|---|---|---|---|---|---|
| Beam, batch 1, p50 / p99 ms | 8.38 / 8.96 | **1.99 / 2.00** | 8.20 / 9.58 | **1.76 / 1.79** | 8.18 / 9.52 | **1.68 / 1.72** |
| Whole cascade, batch 1, p50 / p99 ms | 10.26 / 12.60 | **2.26 / 5.59** | 10.17 / 13.03 | **2.02 / 2.24** | 10.06 / 14.10 | **1.94 / 2.25** |
| B_max | 1 | 32 | 1 | 32 | 4 | 32 |
| Saturation QPS | 101 | 2,621 | 102 | 2,347 | 358 | 2,758 |
| **QPS at the SLO** | 19.1 | **1,825** | 45.5 | **1,877** | 209 | **1,956** |
| p50 / p99 ms at the SLO (median of 3 runs) | 11.3 / 43.9 | 16.0 / 45.4 | 13.2 / 47.9 | 10.8 / 24.3 | 16.6 / 34.1 | 11.4 / 31.4 |
| p99 of the 3 runs, ms | 31.3, 43.9, 44.8 | 45.4, 45.4, 42.4 | 47.9, 43.1, 48.4 | 23.4, 31.3, 24.3 | 34.1, 32.8, 34.9 | 31.4, 42.7, 25.5 |
| Mean batch at the SLO | 1.0 | 18.0 | 1.0 | 12.8 | 2.4 | 14.1 |
| Beam per batch under load, p50 / p99 ms | 8.8 / 24.3 | 11.2 / 11.3 | 8.0 / 23.5 | 3.1 / 4.3 | 8.2 / 12.9 | 2.4 / 3.3 |
| Beam's share of batch time | 83% | 90% | 83% | 48% | 78% | 36% |
| HNSW per batch under load, p50 / p99 ms | 1.4 / 2.7 | 6.9 / 14.7 | 1.3 / 2.0 | 5.4 / 12.0 | 1.7 / 4.0 | 5.5 / 13.3 |
| Dominant stage | beam | beam | beam | HNSW | beam | HNSW |
| Confirmation attempts | 1 | 4 | 3 | 3 | 2 | 2 |

- **B_max is now 32 at every width,** chosen on the validation pool by R5's rule ([valid/select.json](valid/select.json)).
  - A cascade batch of 32 had p99 13.8 to 14.4 ms, under the 16.7 ms limit; a batch of 64 had 25.2 to 25.6 ms.
  - At widths 20 and 50, a batch of 32 is bound by the HNSW search: 32 queries on 16 OpenMP threads.
- **The HNSW search overlaps the beam** (the CPU searches while the GPU decodes), so the shares add up to more
  than 100%.
- **Width 200 is still bound by its beam under load,** at 11.2 ms per batch. Batches of 17 to 32 requests all
  run the graph of 32, so a batch of 18, the mean, costs as much as a full one.
- **The p50 at the SLO rose at width 200** (11.3 to 16.0 ms). It serves 96 times the rate, and a request now
  waits for the 2 ms window and for the batch ahead of it: queue wait 5.1 ms p50, 27.9 ms p99.
- **The bisection steps are not monotone,** and the rate the bisection found failed its confirmation at every
  width.
  - **Width 200:** it failed at 2,150 QPS, passed at 2,107 and 2,129, then took three 5% reductions to confirm
    at 1,825.
  - **Width 50:** it passed at 2,080 and failed at 2,118 with p99 156 ms; it confirmed at 1,877.
  - The tail is set by stalls on the busy machine (below), so the QPS at the SLO is uncertain by about 10%.
  - Every step and run is in the records.
- **The batch-1 p99 of the cascade at width 200 (5.6 ms)** comes from the HNSW search, whose p99 inside that run
  was 4.3 ms against 1.2 ms at p50.

## Background load, and how it may affect p99
The sidecar recorded the machine for every command (`summary.json`, "background"). Nothing was stopped: the
other processes belong to others.

- **Busiest processes:**
  - `td-legacy-vm` (a QEMU VM): 15% to 300% of a core;
  - another user's load test, `kvload` and `kv_server`: up to 490% and 200%, during the projection, profile,
    equivalence, quality and the eager ladder steps;
  - `java`: up to 800%, at the start of the width-50 cascade run;
  - `terraform` and `airflow`: about 100% each at times;
  - a kube-apiserver, kubelet, etcd and desktop apps: a few percent each.
- **1-minute load average:** a median of 5 to 8 for most commands, and 19.5 to 20.2 during the three cascade runs.
  - R5's cascade runs had 2 to 18.
  - Part of the load is the benchmark's own: a spinning server thread and 16 OpenMP threads for the HNSW
    search, now busy at about 2,000 QPS.
- **GPU:** no other GPU process was seen. SM clocks were 2,520 to 2,790 MHz while busy, and the temperature
  reached at most 79 °C. The clocks are not locked.
- **How this may affect p99:**
  - At widths 20 and 50 the cascade's batch time is now set by the HNSW search, which runs on 16 CPU threads.
    On a machine with a load of 20 on 32 cores those threads get descheduled. The HNSW per-batch p99 is 2 to 2.5
    times its p50.
  - So the p99 under load, and the QPS at the SLO, likely understate what a quiet machine gives. The
    confirmation failures and the 20% spread of p99 between runs at the same rate (31 to 43 ms at width 20)
    show this noise.
  - The eager ladder steps are bound by CPU launches, so their batch-1 p99 is sensitive to it too (the single
    stalls above). The graph step is GPU-bound and its p99 is within 2% of its p50.

## Limits
- **One consumer GPU,** an RTX 4090, shared with other services as above.
- **The fp32 lists are equal only up to near-ties:** up to 4.6% of lists at width 200. That is the reference's
  own sensitivity to batch size, but not bitwise equality.
- **The beam is now GPU-bound on small kernels.** About 1,050 kernels run per call, and RMS norm alone is six
  kernels, run 13 times per decoder step. Fusing them (torch.compile or hand-written kernels) would cut batch-1
  latency further, but would change fp32 rounding. R5b did not try it.
- **Power-of-two buckets:** TIGER now runs at the padded batch size. A batch of 17 costs as much as 32.
- **The HNSW search is now the bottleneck** at widths 20 and 50. R5b did not change it.
- **The short timed pass ran the eager static step at about 20 ms per call** at every width
  (`projection.json`), during a burst of the other user's load test. The ladder's 6.9 to 7.0 ms p50 did not
  reproduce it, and a CPU profile of that step in between found nothing slow.
- **R5's other limits still hold:** an in-process server with no network layer, and the ranker as R5 left it.

## GPU time
- **Recorded:** 0.60 GPU-hours on the RTX 4090, the wall time holding its lock, from
  `data/logs/r5b/gpu_time.jsonl`.
  - projection: 0.004;
  - equivalence: 0.051, its cutoff trace 0.043;
  - quality: 0.017;
  - profile: 0.010;
  - ladder: 0.077;
  - select: 0.005;
  - stages: 0.010;
  - the cascade under load: 0.154 at width 200, 0.127 at width 50 and 0.100 at width 20.
- **Development:** 0.028 GPU-hours more, also in the log: smoke tests, test-suite runs under the lock and one
  CPU-profiling pass.
- **Projection,** before the runs (`projection.json`): 0.75 GPU-hours expected, 1.07 at worst. The 4-hour stop
  was not reached.

## Provenance
- **Code** (each record has its `git` state):
  - the projection, equivalence and quality ran at f1c33cc;
  - the cutoff trace ran at 81701ee;
  - the profile, ladder, B_max choice, stages and the cascade runs ran at 843596a, whose `src` tree differs from
    81701ee only in one docstring.
- **Checkpoints:** every record checks TIGER and HSTU-SID against `results/r4b/SHA256SUMS`, and the HNSW index
  against `results/r4c/ann/build.json`, through R5's loaders.
- **The beam:** `src/sidrec/serve/beam.py`. `tests/test_r5b.py` checks each step against the reference in fp32
  on a small model, the graphs' padding and lazy capture, and the explicit dtypes against autocast.

## Files
| Path | Contents |
|---|---|
| `projection.json` | the short timed pass and the GPU-hour projection |
| `profile.json` | torch.profiler on the reference beam and the final one |
| `valid/equivalence.json` | fp32: each step's lists against the reference; bf16 at batch 1 against R5's beam |
| `valid/equivalence_cutoffs.json` | fp32: every mismatched row traced through the four steps |
| `valid/quality.json` | bf16, 100,000 rows: union recall@200 and HSTU-SID NDCG@10 at each width, against R5 |
| `valid/select.json` | the cascade's batch-time sweeps and B_max per width |
| `test/ladder/<step>.json` | each ladder step at each width: batch 1 and batches of 64 |
| `test/stages_batch1.json` | the batch-1 latency of each stage and of the whole cascade |
| `test/e2e_w<w>.json` | the cascade under load at width w, with per-batch stage times |
| `summary.json` | the tables above, before and after, GPU time, the background load and every record |
| `SHA256SUMS` | checksums of the raw arrays in `data/runs/r5b/` (outside git) |

The raw per-request latencies, waits and per-batch stage times are in `data/runs/r5b/raw/`. The bf16 validation
lists are in `data/runs/r5b/`. Check them with `sha256sum -c results/r5b/SHA256SUMS`.
