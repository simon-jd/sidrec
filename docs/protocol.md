# Evaluation protocol

This file was committed on 2026-11-05, before any model in this repository was trained. Everything below is
fixed for every model. A change after that date is listed under "Changes" at the end, with its date and reason;
results produced before a change are kept and labelled with the protocol version they used.

There are two settings:
- **Beauty**, a small standard benchmark, used to reproduce published numbers.
- **Electronics**, a large category from Amazon Reviews 2023, used to run the same methods at scale and on
  cold-start items.

Dataset sources, counts and checksums are in [data.md](data.md).

## 1. Beauty (reproduction anchor)

### Data and split
- **Input:** `reviews_Beauty_5.json.gz` from the 2014 Amazon review data, the file the S3-Rec preprocessing
  script reads (`data/data_process.py` in the S3-Rec repository) and the source of the TIGER benchmark.
- **Ratings:** all are kept. The file is already 5-core, so no further filtering is applied.
- **Counts:** 22,363 users, 12,101 items and 198,502 interactions; the mean sequence length is 8.88 and the
  median is 6. These match TIGER's Table 6.
- **Order:** each user's sequence is sorted by `unixReviewTime` with a stable sort. Ties (the timestamps are
  per day) keep the order in which they appear in the file, as in the S3-Rec script.
- **Split (leave-one-out):**
  - test target: the last item;
  - validation target: the second-to-last item;
  - training: the items before those two.
  - At test time the input is every item before the test target, including the validation item.
- **History length:**
  - ID models (SASRec, HSTU-style) use the most recent 50 items, the S3-Rec default.
  - TIGER uses the most recent 20 items, as in the paper.

### Ranking and metrics
- **Candidates:** the full catalog of 12,101 items. No sampled negatives.
- **Score-based models (SASRec, HSTU-style):** items already in the user's input history are excluded from the
  ranking, as the S3-Rec evaluation code that produced the reference SASRec numbers does.
- **TIGER:** decodes with beam search (beam width 20), constrained to valid semantic IDs, and takes the top 10.
  There is no history masking, as in the paper.
- **Metrics:** Recall@5, Recall@10, NDCG@5 and NDCG@10, averaged over all 22,363 test users. With one target per
  user, Recall@K equals HR@K.
  - rank = 1 + (the number of items with a strictly higher score) + (the number of items with an equal score and
    a lower item index);
  - Recall@K = 1 if rank <= K;
  - NDCG@K = 1 / log2(rank + 1) if rank <= K, else 0.

### Published numbers R2 compares against
All of these use the same data and leave-one-out protocol, ranking over the full item set.

| Source | Model | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|---|
| TIGER (Rajput et al., NeurIPS 2023; arXiv 2305.05065v3), Table 1 | TIGER | 0.0454 | 0.0321 | 0.0648 | 0.0384 |
| TIGER, Table 9 (3 seeds, mean ± standard error) | TIGER | 0.0441 ± 0.00069 | 0.0309 ± 0.00062 | 0.0642 ± 0.00092 | 0.0374 ± 0.00061 |
| TIGER, Table 1 (taken from the S3-Rec public results) | SASRec (BCE loss, one negative) | 0.0387 | 0.0249 | 0.0605 | 0.0318 |
| Mender (Paischer et al., TMLR 2025; arXiv 2412.08604), Table 2, "TIGER (Ours)" | TIGER, independent open-source reproduction | 0.0431 | 0.0275 | 0.0681 | 0.0356 |
| eSASRec (Tikhonovich et al., RecSys 2025; arXiv 2508.06450), Table 4, "Our Experiments" | SASRec+SS (sampled softmax loss) | — | — | 0.092 | 0.053 |
| eSASRec, Table 4, "Our Experiments" | HSTU | — | — | 0.079 | 0.044 |
| ActionPiece (Hou et al., ICML 2025; arXiv 2502.13581), Table 4 | HSTU (their implementation, on tokenized item features) | 0.0469 | 0.0314 | 0.0704 | 0.0389 |

eSASRec reports only @10 metrics, rounded to three decimals.

#### Which number each R2 model is held to
| R2 model | Primary reference | Secondary reference |
|---|---|---|
| SASRec, full-softmax cross-entropy loss | TIGER Table 1, SASRec row | eSASRec SASRec+SS: the level a well-tuned softmax-loss SASRec reaches |
| HSTU-style ranker (item IDs) | eSASRec HSTU | ActionPiece HSTU |
| TIGER | TIGER Table 1 and the Table 9 mean | Mender's reproduction |

- **15% rule:** a model whose primary metric is more than 15% (relative) below its primary reference must be
  investigated before moving on. If the gap is still unexplained, it is reported as unexplained.
- **Secondary references:** the gap to them is always reported, but it does not trigger the 15% rule.

## 2. Electronics, Amazon Reviews 2023 (scale run)

### Data
- **Input:** the official 5-core benchmark with the absolute-timestamp split (`benchmark/5core/timestamp/`),
  plus the category's item metadata.
- **Split boundaries:** the published cutoffs t1 = 1628643414042 ms and t2 = 1658002729837 ms.
  - training: interactions before t1;
  - validation: from t1 up to t2;
  - test: at or after t2.
- The split is used as published. No interaction is added, removed or moved between splits.
- **Order:** a user's timeline is ordered by timestamp. Ties keep the order of the official files, which are
  grouped by user and sorted by time. This order reproduces the `history` column of the official
  `timestamp_w_his` files for 100% of validation and test rows.

### Training and evaluation inputs
- **Training data:** the train split only. Validation is used for early stopping and tuning. Test is used once
  per final run. Models are not retrained on train + validation.
- **Evaluation rows:** every validation or test interaction whose user has at least one earlier interaction.
  - The input is all of that user's earlier interactions, from any split, truncated to the most recent 50 items.
  - Rows with no earlier interaction are excluded: 20,282 in validation and 15,926 in test.
  - That leaves 1,141,362 validation rows and 1,199,539 test rows.
- **Cold-start rows:** test rows whose target item never appears in the train split. There are 535,916 of the
  1,199,539 eligible test rows (44.7%), covering 36,881 distinct items. Validation has 185,482 cold rows, defined
  the same way against train.
  - 228,918 of the cold test targets do appear in the validation split. They are still cold, because no model
    trains on validation.
- **Items outside the train split in ID-only models:**
  - In the input they map to one shared out-of-vocabulary embedding.
  - In the output ranking they are placed below every item seen in training, so their cold-start recall is 0 by
    construction. They are reported that way, not hidden.

### Ranking and metrics
- **Candidates:** the full catalog of 368,228 items, every item in the benchmark. Every item has a metadata
  record. No sampled negatives.
- **Masking:** items in the user's input history are excluded from the ranking. The benchmark has no repeated
  (user, item) pairs; this was checked.
- **Rank and NDCG:** defined as for Beauty.
- **Metrics reported on the test split:**
  - Recall@10, Recall@50 and NDCG@10 over all eligible test rows;
  - the same three metrics on warm rows (target seen in train) and on cold rows, with cold-start Recall@50 as the
    headline cold-start number;
  - for the cascade, the retrieval recall@K of each retriever alone and of their union, for K in {50, 100, 200}:
    the share of rows whose target is among the K retrieved items.
- **Cascade candidate counts:**
  - generative retrieval: beam width 50, giving 50 candidates;
  - ANN retrieval: 150 candidates;
  - their union is deduplicated and filled from the ANN list up to exactly 200;
  - the ranker scores those 200, and the final Recall@10/50 and NDCG@10 are computed on its order.
  - Items outside the 200 count as misses.
- **Evaluation sample (optional):** beam search over 1.2M test rows may be too expensive. In that case the
  cascade and the generative retriever are evaluated on a fixed sample instead:
  - the sample is the first 200,000 eligible test rows sorted by sha256 of `user_id|parent_asin|timestamp`;
  - every model is then also reported on that same sample, next to its full-test numbers.

### Seeds and reporting
- **Headline numbers:** 3 training seeds (0, 1, 2), reported as mean ± sample standard deviation.
- **Tuning runs:** seed 0 only.
- A difference smaller than the larger of the two standard deviations is not called an improvement.

## 3. Tuning budget, identical for every model
- **Budget:** 12 trials per model, in both settings.
- **Shared search space:**
  - learning rate: {1e-4, 3e-4, 1e-3, 3e-3};
  - dropout: {0.0, 0.1, 0.2, 0.3};
  - weight decay: {0, 1e-4, 1e-2};
  - model width (embedding / hidden size): {64, 128, 256};
  - number of Transformer layers: {2, 4}. For TIGER this is the count for the encoder and for the decoder each.
- **Shared trial list:** the trials are 12 draws without replacement from the 288-point grid, using
  `random.Random(20261105)` over the grid in the order listed above. Every model gets the same 12
  configurations.
- **Beauty:** trial 1 is replaced by the configuration published for that model. For TIGER that is the paper's
  setting: 4 layers, width 128, dropout 0.1 and learning rate 0.01 with inverse-square-root decay. For SASRec it
  is the S3-Rec default. Every model still gets 12 trials.
- **Fixed per model, not tuned:**
  - Batch size and optimizer: AdamW with linear warm-up over the first 5% of steps.
  - TIGER's semantic-ID tokenizer, trained once per dataset: RQ-VAE with 3 levels of 256 codes plus a fourth
    code to resolve collisions.
  - The beam widths given above.
- **Selection:** the trial with the best validation NDCG@10.
- **Training length:**
  - Beauty: an evaluation every epoch, early stopping with patience 10, at most 300 epochs.
  - Electronics: an evaluation every epoch on a fixed 100,000-row validation sample, patience 3, at most 30
    epochs.
- **Where tuning runs:**
  - Electronics tuning runs on the dev subset described in [data.md](data.md), on its validation rows.
  - The selected configuration is then trained on the full train split with the 3 seeds.

## Changes
None yet.
