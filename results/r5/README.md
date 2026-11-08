# R5: serving the cascade on one RTX 4090

R5 measures how fast the R4c cascade serves:
- TIGER seed 0's beam and the HNSW search retrieve 200 candidates;
- HSTU-SID seed 0 re-ranks them;
- the response is the top 10.

Nothing is trained or selected on test. Test rows only supply request histories for timing, and no test metric is
computed. Every quality number below is on validation.

All steps follow [docs/protocol.md](../../docs/protocol.md), entry "2026-11-07: R5 serving benchmark":
- the entry was committed in f836321, before any R5 measurement;
- the code was committed in 0aeab5a and 43330a1, before the runs.

**Numbers below** come from [summary.json](summary.json), which `python -m sidrec.eval.r5_summary` builds from
the records in this folder.

**SLO:** p99 latency of at most 50 ms. Latency runs from a request's scheduled arrival to its top 10 in host
memory.

**Load:** open-loop Poisson arrivals, so queueing delay counts and coordinated omission is avoided.

**QPS at the SLO** is found in two parts:
- a 7-step bisection of 20 s runs;
- three 30 s confirmation runs. If any of the three fails, the rate drops by 5%, at most four times.

## Headline
- **The cascade meets the SLO at width 200, but only up to 19 QPS.** At that rate p50 is 11.3 ms and p99 is 43.9 ms.
  - A single request takes 10.3 ms (p50), so the GPU could sustain about 100 QPS.
  - The p99 is set by slow batches, not by the average. Under load, the beam's per-batch p99 is 24 ms, against
    9 ms when requests run one after another.
- **TIGER's beam dominates.** It takes 8.4 of the 10.3 ms at batch 1, and 78% to 83% of batch time under load.
  The HNSW search takes about 1 ms on the CPU, while the beam runs. The union and the ranker take 0.1 to
  0.2 ms each.
- **The ranker ladder makes HSTU-SID 4.6 times faster at batch 1 and about 200 times faster in throughput,**
  without changing its ranking:
  - batch-1 latency falls from 0.72 to 0.16 ms;
  - QPS at the SLO rises from 1,219 to 252,794 with misses and 1,120,994 with cache hits;
  - top-10 overlap with fp32 stays at 0.994, and the NDCG@10 change is at most -0.03%.
- **A narrower beam buys throughput at a large recall cost:**
  - **Width 50:** 45.5 QPS, union recall@200 -30%, NDCG@10 -1.9%.
  - **Width 20:** 209 QPS, union recall@200 -45%, NDCG@10 -4.6%.
  - The beam's batch-1 latency hardly depends on its width, about 8.2 ms at 20, 50 and 200. It is launch-bound,
    so a narrower beam gains only through larger batches.

## Ranker ladder (HSTU-SID alone, test requests)
Each step keeps the steps before it. Each request carries 200 candidates: its R4c ANN list, since the ranker's
cost does not depend on which items it scores.

**Columns:**
- **Batch 1:** requests one after another, with no queueing.
- **At the SLO:** the confirmed rate, with the median of its three runs' p50 and p99.
- **Candidates/s:** 200 times the QPS at the SLO.
- **Quality:** on the 100,000-row validation sample with R4c's union, against step 1.

| Step | B_max | Batch-1 p50 / p99 ms | QPS at SLO | p50 / p99 ms at SLO | Candidates/s | Top-10 overlap | NDCG@10 change |
|---|---|---|---|---|---|---|---|
| 1. fp32, eager | 1 | 0.717 / 0.855 | 1,219 | 2.96 / 17.1 | 0.24 M | 1 (reference) | 0 |
| 2. bf16 | 1 | 0.733 / 0.797 | not met | - | - | 0.9937 | -4.5e-6 (-0.03%) |
| 3. CUDA graphs | 1 | 0.157 / 0.161 | 6,348 | 0.81 / 6.08 | 1.27 M | 0.9937 | -4.5e-6 (-0.03%) |
| 4. history cache, miss | 1 | 0.156 / 0.164 | 5,610 | 0.62 / 11.2 | 1.12 M | 0.9937 | -4.5e-6 (-0.03%) |
| 4. history cache, hit | 1 | 0.136 / 0.139 | 6,395 | 0.90 / 35.7 | 1.28 M | 0.9938 | -2.7e-6 (-0.02%) |
| 5. dynamic batching, miss | 1,024 | 0.155 / 0.246 | 252,794 | 2.34 / 13.5 | 50.6 M | 0.9938 | -2.4e-6 (-0.02%) |
| 5. dynamic batching, hit | 1,024 | 0.145 / 0.164 | 1,120,994 | 1.32 / 2.28 | 224 M | 0.9938 | -2.4e-6 (-0.02%) |

- **Every step keeps quality** by the protocol's rule: top-10 overlap at least 0.95, and NDCG@10 within 1%.
  - bf16 changes the target's rank on 2,406 of 100,000 rows.
  - Step 1's NDCG@10 (0.0149727) is within 2.0e-6 of R4c's record (0.0149747), which used TF32 and batches of
    1,024.
- **bf16 alone does not speed up batch 1** (0.733 against 0.717 ms). At this size (width 64, 2 layers) the
  step is bound by kernel launches, not arithmetic.
- **bf16 has no QPS at the SLO.**
  - Its bisection passed at 1,246 QPS.
  - Every confirmation then failed: the runs at 1,246, 1,184, 1,125 and 1,068 QPS were aborted, because a
    request waited over 1 s. At 1,015 QPS one run's p99 was 512 ms; the other two had 8.0 and 7.4 ms.
  - Its saturation, 1,356 QPS, is that of fp32. These are stalls on the machine (below), not a slower step.
- **CUDA graphs** replay the whole step at once instead of launching each kernel: 4.6 times faster at batch 1.
- **The history cache** saves little at batch 1: a hit takes 0.136 ms against a miss's 0.156 ms. It computes
  1 of 50 rows, but launches dominate.
  - Under batching the hit path is much cheaper. At batch 1,024 its p99 is 0.72 ms against 4.55 ms for a miss,
    on the validation sweep.
  - So its QPS at the SLO is 4.4 times higher.
- **Dynamic batching** (2 ms window) is where the throughput comes from. B_max is 1,024, the top of the grid,
  for both: a miss batch of 1,024 has p99 4.55 ms, under the 16.7 ms limit. At the confirmed rates the mean
  batch is 513 requests with misses and 1,024 with hits.

## Stage latencies
### Batch 1: 2,000 test requests one after another, after 200 untimed
| Stage | p50 ms | p99 ms |
|---|---|---|
| TIGER beam, width 20 | 8.18 | 9.52 |
| TIGER beam, width 50 | 8.20 | 9.58 |
| TIGER beam, width 200 | 8.38 | 8.96 |
| HNSW query (CPU: query, search, history removal) | 1.04 | 1.66 |
| Union (GPU) | 0.094 | 0.101 |
| Ranker (CUDA graphs, bf16, miss) | 0.156 | 0.163 |
| Whole cascade, width 20 | 10.06 | 14.10 |
| Whole cascade, width 50 | 10.17 | 13.03 |
| Whole cascade, width 200 | 10.27 | 12.60 |

- **Inside the whole cascade at width 200,** the stages take, at p50 / p99:
  - beam: 8.57 / 10.64 ms;
  - HNSW: 1.23 / 1.90 ms;
  - union: 0.16 / 0.21 ms;
  - ranker: 0.17 / 0.19 ms.
- **The HNSW search overlaps the beam,** so the stages add up to more than the total.
- **The rest of the total** is the host-to-device copies, the batch's inputs and the top-10 copy back.

### Under load: the cascade at its QPS at the SLO
| Width | B_max | Saturation QPS | QPS at SLO | p50 / p99 ms (3 runs: p99) | Mean batch | Beam: share, p50 / p99 per batch | HNSW share | Queue wait p50 / p99 |
|---|---|---|---|---|---|---|---|---|
| 200 | 1 | 101 | **19.1** | 11.3 / 43.9 (31.3, 43.9, 44.8) | 1.0 | 83%, 8.8 / 24.3 ms | 12% | 0.0 / 24.4 ms |
| 50 | 1 | 103 | 45.5 | 13.2 / 47.9 (47.9, 43.1, 48.4) | 1.0 | 83%, 8.0 / 23.5 ms | 12% | 0.1 / 36.8 ms |
| 20 | 4 | 358 | 209.2 | 16.6 / 34.1 (34.1, 32.8, 34.9) | 2.4 | 78%, 8.2 / 12.9 ms | 17% | 5.7 / 20.6 ms |

- **The beam dominates at every width.** The union and the ranker take about 1.5% of batch time each.
- **Batch sizes, chosen on validation requests:**
  - **Width 200:** B_max is 1, because a batch of 2 had p99 24.1 ms.
  - **Width 50:** B_max is 1 for a closer reason: a batch of 1 had p99 17.3 ms, just over the 16.7 ms limit.
    The rule then gives 1.
  - **Width 20:** batches of 4 fit, with p99 15.2 ms.
- **The p99 is the tail of the beam,** not its average.
  - At width 200 the server is about 20% busy at 19 QPS, yet the beam's per-batch p99 is 24 ms. The protocol's
    batch-1 measurement, which runs requests back to back, gives 9 ms.
  - A request that meets one slow batch, then queues behind another, passes 50 ms.
  - The cause is not identified. The GPU's clocks are not locked and can drop in the idle gaps between Poisson
    arrivals, and the beam is bound by CPU launches on a shared CPU.
- **Width 200 needed 1 confirmation attempt, width 50 three and width 20 two.** The bisection steps are not
  monotone: width 200 failed at 19.9 QPS and passed at 18.2 and 19.1. Every step is in the records.

## What a narrower beam costs (validation, 100,000 rows, TIGER seed 0)
The union rule is unchanged. With width w below 175, it takes all w TIGER items and fills the rest from ANN's list.

| Width | TIGER alone recall@w | Union recall@200: all / warm / cold | HSTU-SID NDCG@10 | Cascade QPS at SLO |
|---|---|---|---|---|
| 200 | 0.1053 | **0.1105** / **0.1284** / 0.0184 | **0.01497** | 19.1 |
| 50 | 0.0531 | 0.0773 (-30%) / 0.0852 / 0.0365 | 0.01469 (-1.9%) | 45.5 |
| 20 | 0.0321 | 0.0607 (-45%) / 0.0646 / **0.0407** | 0.01428 (-4.6%) | 209.2 |

- **A narrower beam gives TIGER's slots to ANN.** Cold recall rises (0.018 to 0.041) and warm recall falls
  (0.128 to 0.065), so the net is a loss.
- **The cost in ranking is smaller than the cost in recall.** NDCG@10 falls by 1.9% at width 50 and by 4.6% at
  width 20. Most targets that drop out of the narrower union were ranked low anyway.
- **Width 200's NDCG@10 (0.014973)** is R4c's validation number for this seed, recomputed with TF32 off; R4c's
  record is 0.014975.

## Serving-path check (validation, first 2,000 rows, width 200, batch 1)
- **The ANN lists equal R4c's saved lists.**
- **TIGER's lists do not.**
  - None of the 2,000 lists is identical to R4c's in order. 97.4% of their items are shared.
  - The union differs accordingly. The cascade's top 10 overlaps the ranker-alone path's at 0.985.
- **Follow-up check:** 32 s on the GPU, outside the protocol, not in the records. It reran R4c's beam on the
  same 2,000 rows:
  - **In chunks of 128, as R4c ran:** 96.4% of lists identical, 99.99% of items shared, top 50 identical.
  - **At batch 1:** 0% identical, 97.4% of items and 97.3% of the top 50 shared.
  - **Recall@200 on these rows:** 0.1110 at batch 1, 0.1105 at 128, 0.1100 saved.
  - So TIGER's lists depend on the batch size, through bf16 rounding in kernels chosen per shape. R4c's numbers
    hold for its batch size, and batch 1 shows no recall loss on these rows.

## Limits
- **Hardware:** one consumer GPU, an RTX 4090.
  - Its clocks are not locked, since there is no root access. It ran in P2, with SM clocks between 2,520 and
    2,805 MHz while busy and up to 72 °C.
  - nvidia-smi saw no other GPU compute process during any command.
- **Other services shared the machine,** and the sidecar recorded them for every command (`summary.json`,
  "background"):
  - **Busiest processes:** two VMs (`td-legacy-vm` at 15% to 143% of a core, `qemu-system-x86` at up to 300%),
    and at times `java` (564%), `terraform` (75%) and `airflow` (69%).
  - **1-minute load average:** a median of 2 to 4 for most commands, and 8 to 18 during the stages, batching
    and width-20 runs. Part of that is the benchmark's own spinning server thread and its 16 OpenMP threads.
- **Stalls:** sporadic stalls of 0.1 to 1 s hit some load runs. Examples are bf16's aborted confirmations, a
  cache-miss run with p99 968 ms, and batching-miss runs at 115 and 81 ms. They make the p99 and the QPS at the
  SLO noisy; this is why some bisection steps and confirmations disagree.
- **In-process server:**
  - one Python thread and one CUDA stream;
  - no network layer, no serialization;
  - requests pre-loaded in pinned host memory: a 50,000-request test pool, cycled by long runs.
- **TIGER's beam is R4c's code, unchanged.**
  - It has no KV cache: at every step it recomputes the decoder and the cross-attention keys and values of all
    200 beams.
  - It runs eagerly, with hundreds of kernel launches per request.
  - The ladder sped up only the ranker, which was already less than 2% of the time.
- **The ladder's QPS gives the ranker the whole 50 ms budget,** so they are upper bounds for a ranker inside
  the cascade.
- **The batch-1 latency of the batching steps leaves out the 2 ms window,** which a lone request waits.
- **Hits and misses are timed in separate runs,** all hits and all misses. No hit rate is measured.

## GPU time
- **Recorded:** 1.48 GPU-hours on the RTX 4090, the wall time holding its lock, from `data/logs/r5/gpu_time.jsonl`.
  - projection: 0.003;
  - beam widths: 0.03;
  - select: 0.004;
  - quality: 0.06;
  - ladder: 0.50;
  - stages: 0.04;
  - the cascade under load: 0.43 at width 200, 0.30 at width 50 and 0.10 at width 20.
- **Projection,** before the runs (`projection.json`): 1.14 GPU-hours expected, 2.21 at worst. The 6-hour stop
  was not reached.
- **Not recorded:** about 0.15 GPU-hours more.
  - Smoke tests of the code before the projection, which wrote nothing.
  - The 32 s follow-up check of TIGER's lists.

## Provenance
- **Code state:**
  - every record but `projection.json` ran at the `src` tree of 43330a1;
  - the projection ran at 0aeab5a, which lacks only the summary module and a GPU version of one statistic in
    the beam-width record;
  - each record has its `git` state.
- **Checkpoints:** every record checks its checkpoints against `results/r4b/SHA256SUMS`, and the HNSW index
  against `results/r4c/ann/build.json`.
- **Inputs:** the serving inputs equal R4c's on 4,096 validation rows: the ranker window and times, TIGER's
  encoder input, the history mask and the ANN window (`valid/quality.json`, "features").

## Files
| Path | Contents |
|---|---|
| `projection.json` | the short timed pass and the GPU-hour projection |
| `valid/beam_widths.json` | TIGER at widths 20 and 50: union recall@200 and HSTU-SID NDCG@10, with width 200 |
| `valid/select.json` | the batch-time sweeps and B_max of each batched configuration |
| `valid/quality.json` | each ladder step against fp32, the input check and the serving-path check |
| `test/ladder/<step>.json` | each ladder step: batch-1 latency, saturation, bisection and confirmation runs |
| `test/stages_batch1.json` | the batch-1 latency of each stage and of the whole cascade |
| `test/e2e_w<w>.json` | the cascade under load at width w, with per-batch stage times |
| `summary.json` | the tables above, GPU time, the background load and every record |
| `SHA256SUMS` | checksums of the raw arrays in `data/runs/r5/` (outside git) |

The raw per-request latencies, waits and per-batch stage times of every run are in `data/runs/r5/raw/`. TIGER's
width-20 and width-50 validation lists are in `data/runs/r5/`. Check them with `sha256sum -c results/r5/SHA256SUMS`.
