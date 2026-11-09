# sidrec

Semantic-ID generative retrieval and an HSTU-style ranker, run as a two-stage cascade on Amazon Reviews 2023
Electronics: 368,228 items, 13.1M training interactions and the dataset's official split by absolute time.

```
user history (50 most recent items and their times)
  |
  +--> TIGER: T5 encoder-decoder, beam search over semantic IDs ----> first 175 items --+
  |      item text -> Qwen3-Embedding-0.6B -> RQ-VAE -> 4-token semantic ID             |--> 200 candidates
  +--> HNSW (FAISS) search with the most recent item's text embedding --> first 25 ----+       |
                                                                                               v
                                    HSTU-SID ranker (item IDs + semantic-ID prefix embeddings) --> top 10
```

Every choice was made on validation data, and each test split was evaluated once. Each phase's rules were
committed to [docs/protocol.md](docs/protocol.md) before that phase ran. The losses are listed below with the
wins.

## Results in brief
- **Best ranker: HSTU-SID.** It is an HSTU-style ranker whose item embedding adds semantic-ID prefix
  embeddings. Over the full catalog it reaches NDCG@10 0.00827 ± 0.00010. The HSTU-style ranker reaches
  0.00796, SASRec 0.00723 and TIGER 0.00686. HSTU-SID is also the only ranker that scores cold items.
- **The cascade helps.** TIGER's beam plus HNSW reach recall@200 0.0737. Re-ranking those 200 candidates lifts
  HSTU-SID to NDCG@10 0.00866, +4.8% over its full-catalog ranking.
- **Serving on one RTX 4090.** A faster TIGER beam computes cross-attention keys and values once, keeps a KV
  cache, precomputes static tensors and replays CUDA graphs. With it, the cascade's QPS at a 50 ms p99 rises
  from 19.1 to 1,825 at beam width 200.
- **Beauty reproduction.** SASRec and the HSTU-style model beat their published numbers. TIGER lands 17%
  below its paper on NDCG@10, and part of that gap is unexplained.

## Headline numbers
### Quality on the Electronics test split
- **Rows:** 1,199,539 test rows. 44.7% of them have a cold target, an item that never appears in the train
  split. On warm rows the target was seen in training.
- **Values:** mean ± sample standard deviation over seeds 0, 1 and 2.
- **Full catalog:** the model ranks all 368,228 items. TIGER instead decodes a width-50 beam over the catalog's
  semantic IDs.
- **Cascade:** the ranker orders the 200 union candidates. A target outside them counts as a miss.

| Model | Ranks | Recall@10 | Recall@50 | NDCG@10 | Warm NDCG@10 | Cold Recall@50 |
|---|---|---|---|---|---|---|
| SASRec | full catalog | 0.01336 ± 0.00003 | 0.03323 ± 0.00019 | 0.00723 ± 0.00002 | 0.01308 ± 0.00004 | 0 by construction |
| HSTU-style | full catalog | 0.01431 ± 0.00007 | 0.03513 ± 0.00018 | 0.00796 ± 0.00009 | 0.01438 ± 0.00016 | 0 by construction |
| HSTU-style, no time bias (seed 0) | full catalog | 0.01363 | 0.03380 | 0.00745 | 0.01347 | 0 by construction |
| TIGER | beam | 0.01296 ± 0.00016 | 0.03108 ± 0.00053 | 0.00686 ± 0.00005 | 0.01240 ± 0.00008 | 0.0000012 ± 0.0000022 |
| TIGER + cold-start rule, ε = 0.1 | beam | 0.01238 ± 0.00017 | 0.03084 ± 0.00059 | 0.00669 ± 0.00005 | 0.01197 ± 0.00008 | 0.00214 ± 0.00026 |
| HSTU-SID | full catalog | 0.01512 ± 0.00020 | 0.03704 ± 0.00014 | 0.00827 ± 0.00010 | 0.01470 ± 0.00019 | **0.00550 ± 0.00025** |
| HSTU-style | cascade | 0.01489 ± 0.00012 | 0.03737 ± 0.00039 | 0.00819 ± 0.00012 | 0.01481 ± 0.00021 | 0 by construction |
| HSTU-SID | cascade | **0.01609 ± 0.00005** | **0.04084 ± 0.00034** | **0.00866 ± 0.00006** | **0.01547 ± 0.00010** | 0.00359 ± 0.00034 |

- **"0 by construction":** the item-ID models rank only items they saw in training.
- **Bold:** the best value in its column.
- **Sources:** [results/r4b](results/r4b/README.md) and [results/r4c](results/r4c/README.md).

### Retrieval recall of the cascade's first stage (test)
Recall@K is the share of rows whose target is among the K retrieved items. HNSW alone does not depend on a
seed.

| Source | Recall@50 | Recall@100 | Recall@200 | Warm recall@200 | Cold recall@200 |
|---|---|---|---|---|---|
| TIGER alone, beam width 200 | 0.03322 ± 0.00046 | 0.04789 ± 0.00068 | 0.06501 ± 0.00087 | 0.11751 ± 0.00157 | 0.0000093 ± 0.0000067 |
| HNSW alone, most recent item | 0.02039 | 0.02754 | 0.03691 | 0.03548 | **0.03868** |
| Union: TIGER's 175 + HNSW's 25 | **0.03737 ± 0.00038** | **0.05358 ± 0.00062** | **0.07371 ± 0.00082** | **0.12058 ± 0.00148** | 0.01567 ± 0.000005 |

The union is the cascade's candidate set.

### Serving on one RTX 4090: before and after the fast beam
- **Before** is R5; **after** is R5b. The cascade is TIGER seed 0, HNSW and HSTU-SID seed 0, returning a top 10.
- **SLO:** p99 latency of at most 50 ms under open-loop Poisson arrivals. Latency runs from a request's
  scheduled arrival to its top 10 in host memory.
- **Requests:** test rows supply the request histories for timing. Recall is measured on the 100,000-row
  validation sample.

| Beam width | Union recall@200 (validation) | Beam alone, batch 1, p50 ms | Whole cascade, batch 1, p50 ms | Max batch | QPS at the SLO, before | QPS at the SLO, after |
|---|---|---|---|---|---|---|
| 200 | 0.1106 | 8.38 → 1.99 | 10.26 → 2.26 | 1 → 32 | 19.1 | **1,825** |
| 50 | 0.0773 | 8.20 → 1.76 | 10.17 → 2.02 | 1 → 32 | 45.5 | **1,877** |
| 20 | 0.0606 | 8.18 → 1.68 | 10.06 → 1.94 | 4 → 32 | 209.2 | **1,956** |

- **Narrower beams no longer buy throughput,** so width 200 keeps its recall at almost no cost in QPS.
- **The bottleneck moved.** At widths 20 and 50 the HNSW search on the CPU is now the slowest stage under load.
- **The ranker alone** (R5) went from 1,219 to 252,794 QPS at the SLO with bf16, CUDA graphs and dynamic
  batching, on history-cache misses.
- **Sources:** [results/r5](results/r5/README.md) and [results/r5b](results/r5b/README.md).

### Beauty reproduction (Amazon 2014 Beauty, all 22,363 test users)
| Model | Recall@10 | NDCG@10 | Primary published NDCG@10 | Gap on NDCG@10 |
|---|---|---|---|---|
| SASRec, full-softmax cross-entropy | 0.0897 ± 0.0002 | 0.0556 ± 0.0002 | 0.0318, TIGER paper, Table 1 | +74.7% |
| HSTU-style | 0.1023 ± 0.0019 | 0.0618 ± 0.0015 | 0.044, eSASRec, Table 4 | +40.4% |
| TIGER, Qwen3-Embedding-0.6B IDs | 0.0604 ± 0.0011 | 0.0318 ± 0.0002 | 0.0384, TIGER paper, Table 1 | -17.3% |
| TIGER, Sentence-T5-XXL IDs | 0.0551 ± 0.0014 | 0.0294 ± 0.0006 | 0.0384, TIGER paper, Table 1 | -23.5% |

- **SASRec's reference** is a BCE model with one negative. Against eSASRec's softmax-loss SASRec, the gap is
  +4.8%.
- **Source:** [results/r2](results/r2/README.md).

## What did not work, plainly
- **TIGER loses to SASRec on warm items.** Its warm NDCG@10 is 0.01240 against 0.01308 (-5.2%). Its warm
  Recall@50 is 0.05617 against 0.06006.
  - Plain TIGER almost never returns an item it never saw as a target: 2 cold hits at K = 50 in three seeds.
- **The cold-start rule gives up more warm hits than it gains cold ones.** That is TIGER's ε rule, which
  fills a share ε of the list with unseen items that share a beam item's semantic-ID prefix.
  - On test at ε = 0.1 it gives up about 1,430 warm hits at K = 50 to gain about 1,150 cold ones.
  - Overall Recall@50 falls 0.8% and NDCG@10 falls 2.4%.
  - On validation each cold hit costs 2.5 to 4.2 warm hits, and every ε above 0 lowers overall Recall@50.
- **HSTU-SID's tuned configuration differs from the HSTU-style model's.** HSTU-SID uses width 64, 2 layers and
  learning rate 3e-3. The HSTU-style model uses width 256, 4 layers and learning rate 1e-4.
  - So HSTU-SID's lead mixes the prefix embeddings with what each model's tuning chose.
  - No matched-configuration run separates the two.
- **The cascade costs HSTU-SID 35% of its cold recall.** Cold Recall@50 falls from 0.00550 to 0.00359,
  because the ranker only sees the cold items the union kept.
- **Two cascade choices sit at the edge of their grids.**
  - The HNSW query uses only the most recent item (n = 1, the shortest window).
  - TIGER takes 175 of the 200 slots, the largest share tried. Recall rose with every step, so the best share
    may lie above 175.
  - Both were reported as chosen. The ranker-only ladder's max batch of 1,024 is also the top of its grid.
- **bf16 had no confirmed rate.** In R5's ranker ladder, bf16 alone passed the bisection, but every
  confirmation run failed on machine stalls. Its saturation, 1,356 QPS, is close to fp32's 1,372, but there
  is no QPS at the SLO to report for that step.
- **Also:**
  - Most of the HSTU-style model's lead over SASRec, about 70% of it on NDCG@10, comes from its time bias, which
    sees the request time. SASRec does not get that input.
  - DDP on two A10 GPUs was 0.52× the speed of one at the protocol's batch size ([R4a](results/r4a/README.md)).
  - The fast beam's fp32 lists match the reference's only up to near-ties: 95.4% are identical at width 200.
    The reference itself gives 94.9% between two batch sizes.

## Limitations
- **Offline evaluation only.** These are next-item metrics on a historical split. There is no online A/B test,
  no interleaving and no measure of user value.
- **Public data under its own terms.** Electronics comes from Amazon Reviews 2023 (McAuley Lab), and Beauty from
  the 2014 Amazon review data.
  - Neither dataset page states a licence. The 2023 site reads "Copyright © 2024, McAuley Lab", and the review
    text and metadata belong to Amazon and the reviewers.
  - The authors ask users to cite Hou et al. (2024) for the 2023 data, and He & McAuley (WWW 2016) and
    McAuley et al. (SIGIR 2015) for the 2014 data.
  - This repository publishes code and derived metrics only. It does not redistribute reviews, metadata or
    interaction files. [docs/data.md](docs/data.md) has the sources and checksums.
- **One consumer GPU.** Everything ran on one RTX 4090, and its clocks were not locked (no root access). Each
  record samples the GPU's state instead.
- **A shared, busy machine during R5 and R5b.** Other users' VMs, a load test, Java and Airflow jobs shared the
  CPU, and none was stopped.
  - The median 1-minute load average was 2.0 to 10.8 during R5's cascade runs, and 19.5 to 20.2 during R5b's.
  - Sporadic stalls hit some load runs. The p99 latencies and the QPS at the SLO are therefore noisy, and
    probably understate what a quiet machine gives.
  - Bisection steps and confirmation runs disagree at some rates. Every run is in the records.
- **An in-process server.** One Python thread and one CUDA stream serve requests pre-loaded in pinned host
  memory. There is no network layer, no serialization and no multi-tenant isolation.
- **Checkpoints are not in git.**
  - Model weights, the HNSW index, embeddings, semantic IDs and raw latencies live under `data/`, which git
    ignores.
  - Their sha256 sums are in each phase's `SHA256SUMS`.
  - To use them, retrain with the commands below.
- **R1 needs a Snowflake account.** A polars reference produces identical tables (parity is checked). But the
  later phases read the Snowflake export's file layout, and there is no switch to the reference.

## Reproduce
**Hardware used:**
- one NVIDIA RTX 4090 (24 GB) on a 32-thread desktop CPU with 128 GB of RAM, Ubuntu, CUDA 13.2;
- an XSMALL Snowflake warehouse for R1;
- a Snowflake GPU pool with two A10s, for R4a's DDP study only.

Each phase README gives its GPU-hours.

**Software:** Python 3.12 with [uv](https://docs.astral.sh/uv/); the lock pins torch 2.15.0, FAISS 1.15.1 and
transformers 5.22.1. R3's vLLM run uses its own environment, `.venv-vllm`, with vLLM 0.31.0.

GPU jobs take a lock file (`flock /var/tmp/rtx4090.lock`) so that only one runs at a time. Run scripts skip
finished runs, so they can be restarted.

```bash
uv sync --locked
scripts/download_data.sh                       # about 1.8 GB into data/raw/, checked against docs/SHA256SUMS
uv run pytest                                  # tests that need data/ or CUDA skip themselves

# R1: data
uv run python -m sidrec.data.beauty
uv run python -m sidrec.data.electronics       # polars reference of the pipeline
SIDREC_SF_HOME=/path/to/sf-module python scripts/snowflake_pipeline.py all   # Snowflake SQL pipeline
uv run python -m sidrec.data.parity            # Snowflake export against the polars reference
uv run python -m sidrec.data.dev_subset        # 1/16 of users, for tuning

# R2: Beauty
scripts/r2_idrec.sh sasrec && scripts/r2_idrec.sh hstu
for s in 0 1 2; do uv run python -m sidrec.train.idrec --model hstu-notime --trial 1 --seed $s --final; done
for e in st5-xxl qwen3-emb-0.6b qwen3-emb-4b; do
  uv run python -m sidrec.semantic_ids.embed --encoder $e && uv run python -m sidrec.semantic_ids.rqvae --encoder $e
done
scripts/r2_tiger.sh

# R3: Electronics embeddings and semantic IDs
uv run python -m sidrec.semantic_ids.embed_catalog docs
PYTHONPATH=src .venv-vllm/bin/python -m sidrec.semantic_ids.embed_vllm
uv run python -m sidrec.semantic_ids.embed_catalog hf --batch-size 64
uv run python -m sidrec.semantic_ids.embed_catalog compare
uv run python -m sidrec.semantic_ids.rqvae_catalog

# R4a-R4c: cost, tuning on the dev subset, full-split seeds, the cascade
scripts/r4a_cost.sh
for m in sasrec hstu tiger; do scripts/r4a_tune.sh $m; done
scripts/r4b_run.sh tune-hstu-sid
scripts/r4b_run.sh final sasrec:0 sasrec:1 sasrec:2 hstu:0 hstu:1 hstu:2 tiger:0 tiger:1 tiger:2 \
  hstu-sid:0 hstu-sid:1 hstu-sid:2 hstu-notime:0
scripts/r4b_run.sh eps 0 1 2
scripts/r4c_run.sh ann-build && scripts/r4c_run.sh ann-select
scripts/r4c_run.sh beam-valid 0 1 2 && scripts/r4c_run.sh rerank-valid 0 1 2
scripts/r4c_run.sh select && scripts/r4c_run.sh projection && scripts/r4c_run.sh ann-test
scripts/r4c_run.sh test 0 1 2

# R5, R5b: serving
for s in projection beam-valid select quality ladder stages; do scripts/r5_run.sh $s; done
scripts/r5_run.sh e2e 200 50 20
for s in projection profile equivalence cutoffs quality ladder select stages; do scripts/r5b_run.sh $s; done
scripts/r5b_run.sh e2e 200 50 20

# Summaries: each writes results/<phase>/summary.json from the run records
for p in r2 r3 r4a r4b r4c r5 r5b; do uv run python -m sidrec.eval.${p}_summary; done
```

R4a's DDP study ran `results/r4a/ddp/payload_sidrec_ddp.py` as a Snowflake job. Its records are in that
folder, and it is not needed for anything else.

## Phases
| Phase | What it did | Write-up |
|---|---|---|
| R1 | Snowflake SQL pipeline for Electronics, a polars reference with a parity check, and the dev subset | [docs/data.md](docs/data.md) |
| R2 | Beauty reproduction of SASRec, the HSTU-style model and TIGER; the time-bias ablation; encoder comparison | [results/r2](results/r2/README.md) |
| R3 | Qwen3-Embedding item vectors with vLLM, checked against sentence-transformers; RQ-VAE semantic IDs | [results/r3](results/r3/README.md) |
| R4a | Cost at Electronics scale, 12 tuning trials per model, DDP scaling on two A10s | [results/r4a](results/r4a/README.md) |
| R4b | Full-split test runs over 3 seeds, TIGER's cold-start rule, HSTU-SID, the time ablation | [results/r4b](results/r4b/README.md) |
| R4c | The cascade: TIGER beam and HNSW union, re-ranked by the HSTU-style model and HSTU-SID | [results/r4c](results/r4c/README.md) |
| R5 | Serving benchmark on the RTX 4090: ranker ladder, stage latencies, QPS at a 50 ms p99 | [results/r5](results/r5/README.md) |
| R5b | The fast TIGER beam, its fp32 equivalence, and the cascade re-measured | [results/r5b](results/r5b/README.md) |

- **Design:** [docs/design.md](docs/design.md) covers the architecture, the reasons for each choice, the
  evaluation protocol, what was dropped and what comes next.
- **Protocol:** [docs/protocol.md](docs/protocol.md) is the evaluation protocol and its dated changes.

## Layout
```
docs/              data sources and terms, evaluation protocol, design notes
sql/electronics/   Snowflake SQL: setup, load, transform, checks, export
scripts/           data download, Snowflake runner, one run script per phase
src/sidrec/data/   Beauty preprocessing, polars reference, parity check, dev subset, model inputs
src/sidrec/models/ SASRec, HSTU-style, HSTU-SID, TIGER (compact T5), cold-start rule, cascade
src/sidrec/semantic_ids/  item text, embeddings (sentence-transformers, vLLM), RQ-VAE
src/sidrec/train/  training loops, chunked full-softmax loss, resumable runs, DDP scaling
src/sidrec/serve/  requests, served ranker, cascade server, fast beam, open-loop load generator
src/sidrec/eval/   metrics, per-phase runners and summaries
results/<phase>/   run records, summary.json, README, SHA256SUMS of artifacts outside git
tests/             unit tests
```

## Licence
The code is under the [MIT licence](LICENSE). The datasets are not part of this repository and keep their own
terms (see Limitations).
