# Design

This note explains how sidrec is put together, why each part was chosen, and how it was evaluated. It also
covers what was tried and dropped, and what would come next. The numbers come from the phase write-ups under
`results/`, which link to their records. The rules each phase followed are in [protocol.md](protocol.md).

## 1. Problem
Given a user's interaction history, recommend the next item from a catalog of 368,228 Electronics products.
- **Data:** Amazon Reviews 2023, the official 5-core benchmark with its split by absolute time. 44.7% of test
  rows have a target that never appears in the train split.
- **Two goals:** a cascade that handles warm and cold items, and that serves within a latency budget on one
  GPU.
- **Reproduction anchor:** Amazon 2014 Beauty, where published numbers exist for every model, comes first. It
  checks the implementations before anything new is measured.

## 2. Architecture and data flow

```
raw review and metadata files (docs/data.md)
  |  Snowflake SQL: stage, load, transform, checks, export to Parquet           (R1; polars reference + parity)
  v
user timelines, evaluation rows, item text documents
  |                                        |
  |                                        v
  |                        Qwen3-Embedding-0.6B via vLLM -> 1,024-d vectors (R3)
  |                                        |
  |                         RQ-VAE: 3 x 256 codes + a 4th collision code -> semantic IDs (R3)
  |                                        |
  v                                        v
item-ID models: SASRec, HSTU-style    TIGER: T5 over semantic-ID tokens; HSTU-SID: IDs + prefix embeddings
(R2 on Beauty, R4a tuning, R4b full-split seeds)
  |
  v
cascade (R4c): TIGER beam (175) + HNSW on the last item's vector (25) -> 200 candidates -> ranker -> top 10
  |
  v
serving (R5, R5b): in-process server, dynamic batching, CUDA graphs, fast beam, open-loop load at a p99 SLO
```

### Data (R1)
- **SQL pipeline:** `sql/electronics/` stages the raw files, loads them and builds the tables. That means user
  timelines, training sequences, evaluation rows with their histories, and one text document per item.
  - Every table is checked: 5-core, no repeated (user, item) pair, histories equal to the official
    `timestamp_w_his` files.
  - The tables are exported as Parquet.
- **Polars reference:** `sidrec.data.electronics` repeats every step in polars. `sidrec.data.parity` compares
  the two by an order-independent content hash, and all seven comparisons match.
- **Dev subset:** one user in 16, chosen by a hash of the user ID, with all of their interactions. It is the
  tuning set.
- **Model inputs:** `sidrec.data.scale` turns the tables into flat timeline arrays. Training windows are cut on
  the GPU from those arrays.

### Semantic IDs (R2, R3)
- **Item documents:** title, brand, categories, feature bullets and description, labelled and truncated per
  field.
- **Embeddings:** Qwen3-Embedding-0.6B with last-token pooling, run by vLLM's pooling runner in bf16. R3
  checked its vectors against a plain sentence-transformers loop: the minimum cosine over all items is 0.9991.
- **Quantization:** an RQ-VAE, as TIGER describes it, maps each vector to three codes of 256. A fourth code
  numbers the items that share all three.
  - On Electronics 32.7% of three-code IDs collide, and the largest group (461 warranty listings) sets the
    fourth code's range.
  - The ID vocabulary is 3 × 256 + 461 codes, plus 2,000 hashed user tokens.

### Models
- **SASRec:** a causal Transformer over item IDs, trained with the exact full-softmax cross-entropy over all
  329,445 train items.
  - `sidrec.train.fused_ce` computes the loss and its gradients in chunks of at most 2^28 logits. Nothing is
    sampled.
- **HSTU-style:** written from the HSTU paper's equations. Its attention is pointwise SiLU, with no softmax,
  plus a relative bias over positions and log-bucketed time gaps.
  - Its time bias sees the request time, the target's timestamp.
  - It uses the same full-softmax loss.
- **TIGER:** a compact T5 encoder-decoder, its own implementation, checked against Hugging Face's T5 logits.
  - The encoder reads a user token and the 20 most recent items as semantic-ID tokens.
  - The decoder emits the next item's four tokens by beam search, constrained to the catalog's IDs by a trie.
  - An item's score is the sum of its four token log-probabilities.
- **TIGER's cold-start rule:** the paper's ε rule, as an option at inference. Unseen items that share a beam
  item's three-code prefix fill at most a share ε of the list.
- **HSTU-SID:** the HSTU-style ranker with one change. An item's embedding is its ID row plus the embeddings of
  its prefixes (c1), (c1, c2) and (c1, c2, c3).
  - Items outside the train split get a zero ID row, so their embedding is the prefix sum alone. The model can
    therefore score cold items.
  - It still trains with the full softmax over the train items.

### Cascade (R4c)
- **Two sources:** TIGER's width-200 beam, and a FAISS HNSW search (inner product, M 32, efSearch 512) with the
  most recent item's vector as the query.
- **Union:** TIGER's first 175 items plus HNSW's first 25, without duplicates, filled from TIGER's list up to
  200.
- **Ranking:** the HSTU-style model or HSTU-SID scores the 200 candidates, and the top 10 is the response.

### Serving (R5, R5b)
- **The server:** one Python thread and one CUDA stream, in the benchmark's process.
  - Requests wait in a queue. Dynamic batching takes the oldest ones, up to B_max, within a 2 ms window.
  - B_max is the largest power of two whose p99 batch time on validation requests is at most a third of the
    SLO.
  - The HNSW search runs on the CPU while the GPU decodes the beam.
- **The ranker:** HSTU-SID in bf16, captured as a CUDA graph per power-of-two batch size. A history cache can
  reuse the keys and values of the 49 older window rows.
- **The fast TIGER beam (R5b):** the decoder's cross-attention keys and values are computed once per request,
  not once per beam at each step. The decoder keeps a KV cache.
  - Request-independent tensors (position biases, bf16 weights) are built at load.
  - The whole call is one CUDA graph per width and batch size.
  - The search itself is unchanged: the same trie, history removal, scores and top-k.
- **Load:** an open-loop Poisson generator with arrival times drawn in advance. Queueing delay counts, and a
  slow server cannot slow the arrivals.
  - QPS at the SLO is found by bisection, then confirmed by three runs.

## 3. Why each component
- **Electronics:** it has the smallest catalog of the categories with at least 5M interactions and 300K items
  with text. That keeps full-catalog softmax and full-catalog embedding affordable on one GPU. Every item has
  metadata.
- **The official time split:** it is published and reproducible, and it creates real cold-start items. A random
  or leave-one-out split hides how a model handles items that appear after training.
- **The Snowflake pipeline, with a polars twin:** the SQL is the warehouse-shaped version a team would run. The
  polars reference makes it checkable without trusting either side.
- **Semantic IDs:** an item-ID table cannot rank an item it never trained on, and 44.7% of test rows need one.
  Semantic IDs give every item, old or new, a position in a shared code space learned from its text.
- **Qwen3-Embedding-0.6B:** on Beauty its IDs beat Sentence-T5-XXL's by about 10% Recall@10 for TIGER. It tied
  the 4B model and encodes 5.5 times faster.
- **vLLM for the 368,228-item pass:** 4.2 times faster than the sentence-transformers loop, with vectors that
  agree to a cosine of at least 0.9991.
- **TIGER as the generative retriever:** it is the reference design for semantic-ID retrieval. Its beam is
  constrained to real IDs, so it never returns an invalid item.
- **The full softmax instead of sampled negatives:** it is the strongest baseline for the item-ID models, as
  eSASRec shows. A chunked fused loss made it fit in memory.
- **HSTU-SID:** the cheapest way to give the ranker cold-start reach. The prefix embeddings share strength
  between items of one cluster, and an unseen item inherits its cluster's embedding.
- **HNSW on the last item:** content-based retrieval reaches cold items, which TIGER's beam almost never
  returns. Validation chose the most recent item over every longer, weighted window.
- **CUDA graphs:** the beam and the small ranker were launch-bound. At batch 1 the GPU was busy for only 25% to
  35% of the reference beam's wall time. Graphs remove the launches without changing the arithmetic.
  - torch.compile could fuse kernels further, but it would change fp32 rounding, so it was left for later.

## 4. Evaluation protocol
- **Fixed before any training.** [protocol.md](protocol.md) was committed (2e1ee2b) before any model was
  trained. It fixes the splits, the cold-start definition, the metrics, the reference numbers, seeds and the
  tuning budget.
  - Each later phase added a dated entry before its first measurement. Those entries are R4a (0730bc6), R4b
    (f3e9db5), R4c (afb011d, with the correction b230f27), R5 (f836321) and R5b (fda5ee5).
  - R5b's near-tie note (f6019cb) came after its fp32 equivalence run and before any timing, and says so.
  - Changes forced by a finding, such as the RQ-VAE optimizer, are dated entries that give their reason.
- **Choose on validation, test once.**
  - Tuning uses 12 shared trials per model, drawn from one grid with a fixed seed, on the dev subset's
    validation rows. Selection is by validation NDCG@10.
  - The selected configuration trains on the full train split with seeds 0, 1 and 2. Each final checkpoint is
    evaluated on test once.
  - The cascade's union and the ANN query were chosen on a fixed 100,000-row validation sample. Serving
    choices, such as B_max and beam widths, were made on validation requests. Test requests only supply
    histories for timing.
- **Full-catalog ranking with history masking,** and no sampled negatives. Ties go to the lower item index.
  Metrics are Recall@10, Recall@50 and NDCG@10, overall and on warm and cold rows.
- **Reporting:** mean ± sample standard deviation over three seeds. A difference smaller than the larger
  standard deviation is not called an improvement.
- **The 15% rule on Beauty:** a model more than 15% below its primary published number must be investigated.
  TIGER was, and the unexplained remainder is reported as unexplained.
- **Provenance:** every run record carries its configuration, seed, commit, dirty flag, GPU state and wall
  time.
  - Checkpoints and large arrays stay out of git, with sha256 sums in `results/<phase>/SHA256SUMS`. Later
    phases verify their inputs against those sums.
  - Every summary is rebuilt from the records by a module.
- **GPU-hour stops:** each long phase projected its GPU time from a short timed pass, and would have stopped
  past a fixed limit. None did.

## 5. Tried and dropped
- **DDP on two A10 GPUs (R4a).** The two vGPUs sat in one Snowflake node, with no NVLink and no peer-to-peer.
  NCCL went through host shared memory.
  - At the protocol's batch of 128, two GPUs ran 0.52 times as fast as one (scaling efficiency 26%). At batch
    1,024 they ran 1.35 times as fast (68%).
  - **Why:** the full softmax makes the 329,447 × 256 item table's gradient dense, 327 MB per step. The
    all-reduce moved it at 2.2 GB/s (153 ms).
  - It cannot overlap with the backward pass, because the same table is the input embedding. Its gradient is
    complete only at the end.
  - One RTX 4090 was 2.0 times faster than one A10 at batch 128, so the full-split runs stayed on the 4090.
  - The protocol fixes the batch size, so larger per-GPU batches were not an option. bf16 gradient
    compression and sparse gradients were not tried.
- **The paper's RQ-VAE optimizer (Adagrad at 0.4).** Every variant diverged or learned only the mean embedding.
  Adam at 1e-3 replaced it.
  - A first fix produced IDs from an untrained encoder. Two TIGER trials ran on those IDs before that was
    caught; they are kept and labelled as superseded.
- **The paper's TIGER learning rate under AdamW.** At 0.01 it diverged. Adafactor, the paper's optimizer, ran
  as a diagnostic outside the protocol. It closed part of the gap, not all.
- **Sentence-T5-XXL item embeddings.** Its IDs used only 38 of 256 first-level codes on Beauty. Qwen3-Embedding
  IDs used 255 and gave TIGER better results.
- **The sentence-transformers embedding loop at Electronics scale.** It padded half of its tokens. vLLM
  replaced it.
- **TIGER's cold-start rule inside the cascade.** It lowered union recall@200 at every TIGER share on
  validation, because the HNSW part already brings in cold items. The cascade uses the plain beam.
- **Longer ANN queries.** Weighted means of the last n items, for n up to 50, all scored below the last item
  alone.
- **The protocol's original cascade setting:** 50 beam items, 150 ANN items, ANN backfill. Validation recall@200
  was 0.0794, against 0.1095 for the chosen union. The grid replaced it in the R4c entry, before any
  measurement.
- **bf16 alone for the ranker.** It did not speed up batch 1, because the step was launch-bound. CUDA graphs
  did, by 4.6 times.
- **Narrow beams for throughput (R5).** Width 20 served 209 QPS against 19 at width 200, for 45% less union
  recall. After the fast beam all widths serve about the same QPS, so the narrow beams lost their reason.

## 6. What would come next
- **Retrieval is the bottleneck.** 88% of warm test targets and 98% of cold ones are not among the 200
  candidates.
  - The fast beam makes wider beams cheap, so a width above 200 and a larger candidate set are the first
    things to test.
  - Both union choices sit at a grid edge. A new protocol entry should extend the grid: TIGER shares between
    175 and 200, and the HNSW query.
- **The HNSW search is now the slowest stage** at widths 20 and 50, and it competes for CPU threads.
  - Its query is a single item's vector, so its 25 neighbours can be precomputed per item, which turns the
    stage into a table lookup.
  - Alternatively, a GPU index would also do.
- **Separate HSTU-SID's prefix effect from its tuning.** Train HSTU-SID at the HSTU-style model's
  configuration, and the reverse, with three seeds each.
- **Better cold-start retrieval.** TIGER's decoder gives almost no probability to IDs it never saw as targets.
  Worth testing: ID-level dropout or held-out items during training, a prefix-level objective, or a learned
  mix of the beam and the content search in place of a fixed share.
- **Fuse the beam's small kernels.** About 1,050 kernels per call now bound the graph. torch.compile or
  hand-written RMSNorm and attention kernels would help, with the near-tie check repeated, since fp32 rounding
  would change.
- **A production-shaped server.** That means a network front end with serialization, several streams or
  processes, a measured cache hit rate, and p99 on a quiet machine with locked clocks.
- **Online evaluation.** Offline next-item metrics on a historical split cannot show user value. An A/B test
  or interleaving would be the real check.
- **Multi-GPU training that scales.** The full-softmax gradient needs a sharded item table, or a sampled loss
  whose gradient is sparse, before DDP pays off on PCIe-only hardware.
