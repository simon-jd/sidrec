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

### 2026-11-05: R2 clarifications, written before any Beauty model was trained
Nothing above changes. The sections above leave the points below open; R2 fixes them here, and every R2 result
uses them.

**Training schedule and selection (all models)**
- **Warm-up length:** "the first 5% of steps" means 5% of the maximum number of steps, 300 epochs times the
  steps per epoch. Early stopping makes the actual count unknown in advance.
- **After warm-up:** SASRec and the HSTU-style model keep the learning rate constant, as S3-Rec and
  generative-recommenders do. TIGER decays it with the inverse square root of the step in every trial, which
  is its paper's schedule.
- **AdamW:** weight decay applies to every parameter.
- **Batch sizes:** SASRec 256 (S3-Rec), HSTU-style 128 (the generative-recommenders Amazon Books config),
  TIGER 256 (its paper).
- **Test use:** tuning runs (seed 0) compute no test metrics. The selected trial's saved checkpoint is evaluated
  on test once and becomes the seed-0 final run. Seeds 1 and 2 are trained from scratch.

**Trial 1, the published configuration**
- **SASRec:** the S3-Rec defaults. Width 64, 2 layers, 2 heads, dropout 0.5 on hidden states and attention,
  learning rate 1e-3, no weight decay.
- **HSTU-style:** the generative-recommenders `configs/amzn-books/hstu-sampled-softmax-n512-final.gin`, the
  repository's Amazon setting. Width 64, 4 layers, 4 heads of 16, dropout 0.5, learning rate 1e-3, no weight
  decay.
- **TIGER:** as stated in section 3. The paper gives no weight decay, so it is 0.

**Item-ID models**
- **Loss:** full-softmax cross-entropy over all 12,101 items, at every position of the user's training window.
  For the HSTU-style model this replaces the sampled softmax with 512 negatives of the generative-recommenders
  config.
- **Training window:** one per user, the last 50 inputs of the training part, as S3-Rec trains SASRec.
- **SASRec, fixed:** 2 heads, post-LN blocks, GELU feed-forward of width 4 × width, and the item table shared
  between input and output.
- **HSTU-style, fixed:**
  - 4 heads of width / 4;
  - L2-normalized states and item table, with temperature 0.05;
  - a relative attention bias over positions and 128 log-scale time buckets;
  - attention divided by the window length, 50.
- **HSTU time bias:** the query time of a position is the timestamp of the item it predicts, the request time,
  as in generative-recommenders. At evaluation that is the target's review date, which is known at request
  time; the target item itself is not.

**TIGER**
- **Item text:** title, price, brand and categories from `meta_Beauty.json.gz`, the fields TIGER section 4
  lists. The paper gives no template, so ours is `Title: ... Brand: ... Price: ... Categories: ...`, with
  missing fields left out.
- **Sentence-T5 size:** the paper does not name one. We use `sentence-t5-xxl`, the largest. Its 4.9B-parameter
  encoder fits on the 24 GB RTX 4090 at 16-bit precision. It runs in bfloat16, not float16, because T5
  activations overflow in float16.
- **RQ-VAE:** as TIGER section 4 describes it.
  - Encoder 512-256-128 with ReLU, latent 32, 3 levels of 256 codes of size 32, beta 0.25.
  - Adagrad with learning rate 0.4, batch 1024, 20,000 epochs, k-means initialisation on the first batch.
  - It is trained once per encoder, with seed 0, on the 12,068 items that occur in some user's training part.
    It then assigns codes to all 12,101 items; 33 items occur only as validation or test targets.
  - The fourth code numbers the items that share a three-code prefix 0, 1, 2 and so on.
- **Training examples:** every position of each user's training part is a target. The input is up to 20
  preceding items (TIGER Appendix C).
- **Tokens:**
  - The input is one user token, the user ID hashed into 2,000 buckets, then the four tokens of each history
    item, oldest first.
  - The output is the target's four tokens.
  - Each code level has its own token range.
- **Model:** a T5 encoder-decoder (Hugging Face transformers) with 6 heads of 64, feed-forward width 1024 and
  ReLU. Width, layers and dropout come from the trial.
- **Gradient clipping:** global norm 1.0. TIGER trained with T5X, whose default Adafactor clips its updates;
  AdamW gets a gradient clip in its place.
- **Encoder comparison:** the 12 trials use the Sentence-T5 IDs. The selected configuration is then trained with
  seeds 0, 1 and 2 on IDs built the same way from Qwen3-Embedding-0.6B and Qwen3-Embedding-4B vectors. Those
  models use last-token pooling and L2 normalization, with no instruction prefix, as they are meant to be used
  for documents.

### 2026-11-05: RQ-VAE loss scale, found before any TIGER model was trained
The RQ-VAE as specified above did not train, so two details change. Both concern only the semantic-ID tokenizer.
No recommendation metric was computed while deciding them; the check was codebook use on the training items.

- **What failed:** with the reconstruction and quantization terms summed over dimensions, as the paper writes
  them, Adagrad at learning rate 0.4 diverged within 20 steps. The encoder output grew to a norm of about 10^4,
  and every item fell on a single code at all three levels.
- **Change 1, the Adagrad accumulator:** it starts at 0.1, the TensorFlow default, instead of PyTorch's 0. With
  that alone it still diverged.
- **Change 2, the reconstruction term:** it is averaged over the input dimensions instead of summed. The
  quantization terms stay squared norms over the 32 latent dimensions.
- **Result of the check:** over 6,000 steps on Sentence-T5 vectors, all three levels kept at least 232 of their
  256 codes, and about 99% of items had a distinct three-code ID.
- **Alternatives tried in the same check:**
  - averaging both terms: the first level collapsed to 2 codes;
  - summing both terms with a gradient clip of 1.0: it worked, but adds a mechanism the paper does not have;
  - AdamW at 1e-3: it collapsed with the averaged terms and kept only 74 first-level codes with the summed ones.
- **Averaging over the input dimensions also helps the encoder comparison:** the loss scale then no longer
  depends on the embedding width (768, 1,024 or 2,560).

### 2026-11-05: TIGER's T5 implementation
- **What changed:** the T5 encoder-decoder is our own compact implementation, `src/sidrec/models/t5.py`. It is
  not Hugging Face transformers' T5, as the first R2 clarification said.
- **Equivalence:** a test loads the same weights into both and checks that their logits agree to 1e-5.
- **Why:** it trains about 25% faster at TIGER's size (13.8 against 18.5 ms per step of 256 examples on the
  RTX 4090).

### 2026-11-05: RQ-VAE optimizer, replacing the RQ-VAE change above
The fix above was wrong. Its IDs used every code, but the RQ-VAE behind them had learned nothing.

- **What was wrong:** the decoder output the mean embedding for every item. It explained 0% of the embedding
  variance for all three encoders. The encoder output had shrunk until the quantization terms were about
  10^-6, so the codes came from an encoder that was barely trained. The check above looked only at code use
  and missed this.
- **How it was found:** two TIGER tuning trials (1 and 2) had run on those IDs, plus part of trial 3. Their
  validation NDCG@10 was low, which led to a check of the tokenizer's reconstruction. The new RQ-VAE was chosen
  on reconstruction and code use only, never on a recommendation metric.
  - The two finished trials are kept under `results/r2/superseded/`, labelled with the IDs they used, and are
    not part of the tuning budget.
  - TIGER's 12 trials are rerun from the start on the new IDs.
- **Variants checked:** 20,000 steps each on the Sentence-T5 vectors, scored by explained variance and code use.
  - Every Adagrad-at-0.4 variant either diverged or explained no variance: raw, centred or standardized
    inputs, and losses averaged or summed over dimensions.
  - AdamW at 1e-3 with averaged losses collapsed to one code.
- **New setting:**
  - the paper's loss, summed over dimensions as it is written, with the paper's batch, epochs and k-means
    initialisation;
  - Adam (AdamW with no weight decay) at learning rate 1e-3.
  - Over 40,000 steps it explained 57% of the Sentence-T5 variance and 55% of the Qwen3-Embedding-4B variance.
    About 95% of items had a distinct three-code ID.
- **What does not change:** the first level uses only part of its codebook, about 100 of 256 codes for
  Sentence-T5 and about 180 for Qwen3-Embedding-4B. The other two levels use all 256. The paper reports at
  least 80% use; this is reported, not tuned away.
- **Correction, after the full RQ-VAE runs (20,000 epochs):** first-level use fell further than the 40,000-step
  checks showed. The final runs, in `results/r2/semantic_ids/`, are:

  | Encoder | First-level codes used | Explained variance | Collision rate |
  |---|---|---|---|
  | Sentence-T5-XXL | 38 of 256 | 57% | 9.2% |
  | Qwen3-Embedding-0.6B | 255 of 256 | 56% | 9.3% |
  | Qwen3-Embedding-4B | 135 of 256 | 55% | 12.9% |

  The second and third levels use 247 to 256 codes. The IDs are used as they are.

### 2026-11-05: Electronics item embeddings and semantic IDs, written before any were computed
Section 2 does not say how Electronics items are embedded and tokenized. R3 fixes that here. It follows the
Beauty recipe and changes only what the larger catalog requires.

**Item embeddings**
- **Encoder:** Qwen3-Embedding-0.6B at the revision R2 pinned. R2's encoder comparison chose it
  (`results/r2/README.md`): its TIGER results tie the 4B model's within one standard deviation, and it encodes
  5.5 times faster. As on Beauty, it runs in bfloat16 with last-token pooling, L2 normalization and no
  instruction prefix.
- **Input:** each item's R1 document (the `doc` column of `REC_ITEM_DOCS`), unchanged and not truncated. The
  tokenizer appends the end-of-text token that last-token pooling reads.
- **Item order:** every R3 array has one row per catalog item, all 368,228 of them, in ascending `parent_asin`
  order.
- **Two implementations:** vLLM 0.31.0's pooling runner, and the sentence-transformers batch loop R2 used. Both
  encode the whole catalog on the RTX 4090.
- **Agreement:** measured on every item, as the cosine of its two vectors and the largest absolute difference
  of any component.
- **Which vectors the IDs use:** vLLM's. If any item's two vectors have a cosine below 0.99, the IDs use the
  sentence-transformers vectors instead, and that is reported.

**Semantic IDs**
- **Recipe:** Beauty's final RQ-VAE, unchanged: architecture, loss, Adam at 1e-3 without weight decay, batch
  1,024, k-means initialisation on the first batch, and seed 0.
- **Training items:** the 329,445 items with a train-split interaction. On Beauty the RQ-VAE trains on the items
  in some user's training part, and this is the same rule. Validation and test interactions never shape the
  tokenizer. Codes are then assigned to all 368,228 items.
- **Training length:** the number of optimizer steps is kept, not the number of epochs.
  - On Beauty, 20,000 epochs were 240,000 steps of 1,024 items.
  - On 329,445 items, 20,000 epochs would be 6.4 million steps: about 2.8 hours at the speed of R2's
    Qwen3-Embedding-0.6B run, 384 s for 240,000 steps.
  - R3 trains 746 epochs, which is 240,212 steps.
- **Fourth code:** items that share a three-code prefix are numbered 0, 1, 2 and so on. The train items come
  first, in item order, then the other items. A train item's ID therefore does not depend on which items occur
  only in validation or test. On Beauty the numbering followed the item index.
- **Reported, not tuned:**
  - reconstruction, as explained variance;
  - each level's code use;
  - the collision rate before the fourth code;
  - the share of cold-start test items whose first two codes occur among the train items' IDs.
- No recommendation metric is computed in R3.

### 2026-11-06: Electronics training and evaluation details, written before any Electronics model was trained
Sections 2 and 3 leave the points below open for the scale run. R4a fixes them here, and every Electronics run
uses them. Nothing above changes. R4a computes no test metric.

**Inputs**
- **Item numbering:** the R3 order. Catalog items are 1 to 368,228 in ascending `parent_asin` order, and 0 is
  padding. The dev subset uses the same numbering.
- **Training windows of SASRec and the HSTU-style model:** one per user, the last 50 inputs of the user's train
  interactions, as on Beauty. A user with one train interaction has no target.
  - On the full split this keeps 11,308,328 of the 11,491,609 train targets (98.4%).
  - The 183,281 left out are the earliest targets of the 7,005 users with more than 51 train interactions.
- **TIGER's training examples:** every train position with at least one earlier train item, as on Beauty. That
  is 11,491,609 examples on the full split. The input is the user token and the 20 most recent earlier items.
- **Evaluation input:** section 2's input, the 50 most recent earlier interactions from any split.
  - The item-ID models read all 50.
  - TIGER's encoder reads the most recent 20 of them, its Beauty history length. 13.3% of validation rows
    have more than 20.
  - The history mask always removes all 50, for every model.
- **Items outside the train split in the item-ID models:** the shared out-of-vocabulary row starts at zero. No
  training input is outside the train split, so the row never gets a gradient and stays zero.
- **Dev-subset vocabulary:** an item-ID model tuned on the dev subset has the subset's 184,738 train items, the
  same set its cold-start flags use (data.md).
- **HSTU time bias:** Electronics timestamps are in milliseconds. They are floored to seconds before bucketing,
  because the buckets are defined in seconds.

**Ranking**
- **Item-ID models:** the target is ranked among the train items. A target outside the train split is a miss.
  - For any K up to 50, this equals ranking the full catalog with every other item scored below all train
    items, as section 2 states. A test checks it.
  - Ties go to the lower catalog index, because the train items keep catalog order.
- **TIGER's beam width:** 50, the cascade's width, also when TIGER is evaluated on its own. Recall@50 needs 50
  items.
- **TIGER's valid IDs:** the beam is constrained to the semantic IDs of all 368,228 catalog items.
- **TIGER's history mask:** section 2's masking applies to every model on Electronics. On Beauty, TIGER had none,
  as in its paper. The 50 history items are removed at the fourth level, so the beam still returns 50 items.
- **How TIGER scores the beam:** an item's score is the sum of its four tokens' log-probabilities.
  - Each comes from the softmax over the whole token vocabulary, not renormalized over the valid codes.
  - The 50 items are ranked by that sum.
- **Fourth code:** its token range has 461 codes, the size of the largest collision group (R3). The vocabulary
  is 2 special tokens, 3 × 256 codes, 461 codes and 2,000 user tokens: 3,231 tokens.

**Training**
- **Loss of the item-ID models:** the exact full-softmax cross-entropy over the train items (329,445 on the full
  split).
  - It is computed in chunks of rows by a fused forward and backward pass (`sidrec.train.fused_ce`), so at most
    2^28 logits exist at once.
  - Logits and log-sum-exp are in float32, with TF32 matrix products as on Beauty. Nothing is sampled.
- **Batch sizes:** the R2 values, on the dev subset and on the full split: SASRec 256, HSTU-style 128, TIGER 256.
  - The learning rate is selected on the dev subset and reused on the full split. That transfer only holds at
    the same batch size.
  - Steps per epoch: 386, 771 and 2,796 on the dev subset; 6,160, 12,319 and 44,890 on the full split.
- **Warm-up:** 5% of the maximum number of steps, which is 30 epochs times the steps per epoch.
- **Everything else as on Beauty:**
  - AdamW, with weight decay on every parameter;
  - a constant learning rate after warm-up for the item-ID models, inverse-square-root decay for TIGER;
  - TIGER's gradient clip of 1.0;
  - float32 with TF32 for the item-ID models, bfloat16 autocast for TIGER.
- **Trials:** the 12 shared draws, trial 1 included. Section 3 replaces trial 1 with a published configuration
  only on Beauty.

**Validation and selection**
- **The 100,000-row validation sample:** the first 100,000 eligible validation rows sorted by sha256 of
  `user_id|parent_asin|timestamp`, with the timestamp in milliseconds. This is the test sample's rule from
  section 2.
- **Dev subset:** it has 71,610 eligible validation rows, fewer than 100,000, so every epoch evaluates all of
  them. Selection uses validation NDCG@10 over those rows. Cold rows count, and they are misses for the
  item-ID models.
- **Early stopping:** on validation NDCG@10, with patience 3 and at most 30 epochs. The best epoch's weights
  are kept.
- **Cost measurements:** they time one evaluation over all 1,141,362 eligible validation rows as a stand-in
  for the 1,199,539 test rows. Nothing is selected from them.

### 2026-11-06: R4b final runs, TIGER's cold-start retrieval, HSTU-SID and the time ablation, written before any R4b run
R4b trains the selected configurations on the full train split and reports test metrics. It also adds two
models that can rank cold-start items, and one ablation. The sections and entries above leave the points below
open; R4b fixes them here, and every R4b run uses them. Nothing above changes.

**Final runs**
- **Configurations:** the R4a selections on the dev subset ([results/r4a/README.md](../results/r4a/README.md)).
  - SASRec: trial 1.
  - The HSTU-style model: trial 1.
  - TIGER: trial 2.
  - HSTU-SID: selected the same way from its own 12 dev trials (below).
- **Seeds 0, 1 and 2, each trained from scratch on the full train split.** The selected dev trial is not reused
  as seed 0, because it trained on the dev subset. On Beauty, tuning and final runs shared one training set.
- **Training length:** section 3 and the R4a entry. Every epoch is evaluated on the 100,000-row validation
  sample, with patience 3 and at most 30 epochs, and the best epoch's weights are kept.
- **Test, used once per final run:** after training, the best epoch's weights are evaluated on all 1,199,539
  eligible test rows.
  - The metrics are Recall@10, Recall@50 and NDCG@10, over all rows, warm rows and cold rows.
  - Nothing is selected on test, and no final run is repeated because of its test result.
- **Runs that span processes:** each process holds the GPU lock for about an hour of training at most.
  - After every epoch it saves the training state: the model, the optimizer, the schedule, the batch-order
    generator, the CPU and CUDA random generators, and the early-stopping state.
  - The next process resumes from that state.
  - A test checks, on the CPU, that a run resumed after every epoch ends with the same history and weights as
    an uninterrupted one.

**TIGER's cold-start retrieval (inference only)**
- **Source:** Rajput et al., NeurIPS 2023 (arXiv 2305.05065v3), section 4.3 "New Capabilities", paragraph
  "Cold-Start Recommendation", and Fig. 5. The paper:
  - retrieves, for a semantic ID the model predicts, the seen item with that ID;
  - adds to the candidates the "unseen items having the same first three semantic tokens", (c1, c2, c3);
  - limits unseen items with a hyperparameter ε, the largest share of them among the top K;
  - uses ε = 0.1 in Fig. 5a, its main cold-start comparison, and varies ε in Fig. 5b.
- **Fixed from the paper:**
  - the prefix rule: the first three codes (c1, c2, c3), the RQ-VAE codes, not the fourth code that separates
    collisions;
  - ε = 0.1;
  - unseen items: items outside the train split, the protocol's cold items. That is 38,783 of the 368,228
    items on the full split.
- **What the paper leaves open, fixed here:**
  - **Predicted IDs:** the 50 items of plain TIGER's beam. That beam has width 50, is constrained to the
    catalog and has the user's history removed (R4a entry). There is no second decoding.
  - **Candidates:** unseen items whose (c1, c2, c3) equals that of one of the 50 beam items, and that are
    neither in the user's 50-item history nor in the beam already.
  - **Their order:** their own full-ID score, the sum of their four tokens' log-probabilities. That is the
    prefix's beam score plus the last code's log-probability, the same score that ranks the beam.
  - **One ranked list of 50 for every K:** position j (1-based) takes the best remaining candidate while the
    list holds fewer than floor(ε j) unseen items. Otherwise it takes the next beam item.
    - Unseen items that the beam emitted itself count towards the share.
    - With ε = 0.1, the candidates fill positions 10, 20, 30, 40 and 50, and the beam's last five items drop
      out.
    - The paper reports Recall@K only. The NDCG@10 of this list follows from this placement, which is ours.
- **Inference only,** on the three final TIGER checkpoints. The test pass that scores plain TIGER also scores
  this list at ε = 0.1, from the same beam. No other ε is computed on test.
- **Validation ε curve, as analysis:** ε in {0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1}, on the 100,000-row
  validation sample, one beam pass per checkpoint. Nothing is selected from it.

**HSTU-SID**
- **Definition:** the HSTU-style model with one change. The embedding of catalog item i is
  e(i) = E_id[i] + E_1[(c1)] + E_2[(c1, c2)] + E_3[(c1, c2, c3)], with R3's semantic IDs.
  - E_id is the HSTU-style model's item table. Items outside the train split get its zero row (R4a entry).
  - Each prefix table has one row per prefix that occurs among the source's train items. On the full split
    that is 256, 39,715 and 226,620 rows; on the dev subset 256, 32,962 and 138,876.
  - A prefix that no train item has shares a zero row with padding. On the full split, 1,355 of the 38,783
    items outside train lack a (c1, c2) row and 23,492 lack a (c1, c2, c3) row.
  - The prefix tables start at zero, so with the same seed HSTU-SID starts as exactly the HSTU-style model. A
    test checks this.
  - e(i) replaces the item table wherever the HSTU-style model uses it: as the input embedding (times the
    square root of the width), and, L2-normalized and divided by the temperature 0.05, as the scoring table.
- **Citation:** after Singh et al., RecSys 2024 (arXiv 2306.08121v2, section 3.3). They represent an item by
  the sum of learned embeddings of pieces of its semantic ID, in place of its random ID. Their pieces are
  non-overlapping N-grams. The prefix pieces and the kept ID row are this repository's design, as the plan
  states it.
- **Loss:** the HSTU-style model's, the exact full-softmax cross-entropy over the train items. Items outside
  the train split are never targets or negatives in training.
- **Ranking:** HSTU-SID scores every one of the 368,228 catalog items, on the dev subset too. The history is
  removed and ties go to the lower catalog index, as for every model. A cold target can therefore be ranked.
- **Tuning:** the 12 shared trials on the dev subset, seed 0, validation only.
  - Selection uses validation NDCG@10 over all 71,610 eligible dev rows, cold rows included.
  - The selected trial is then trained with seeds 0, 1 and 2 as above.
- **Everything else** is the HSTU-style model's: 4 heads, the relative attention bias with time, temperature
  0.05, batch 128, AdamW with a constant learning rate after warm-up, float32 with TF32.

**The HSTU-style model without the time bias**
- **Run:** the selected HSTU configuration (trial 1), seed 0, trained on the full train split and tested once.
- **The change:** the time part of the relative attention bias is removed, and the position part stays. As in
  R2's ablation on Beauty, this removes every timestamp input, including the target's time.
- **Comparison:** against HSTU seed 0 and the HSTU 3-seed mean and standard deviation. With one seed, a
  difference smaller than the HSTU standard deviation is not called a difference.

**GPU-hour stop**
- Before the long runs, the total is projected from R4a's cost table. It is projected again after each final
  run, with the measured times.
- R4b stops and reports options if the projection passes 60 GPU-hours.

### 2026-11-07: R4c two-stage cascade, written before any R4c measurement
R4c puts a retrieval stage in front of the R4b rankers: TIGER's beam and an approximate nearest-neighbour (ANN)
search over the item vectors retrieve 200 candidates, and a ranker re-ranks them. Nothing is trained. Section 2
fixed the cascade's counts (a width-50 beam, 150 ANN items, filled from the ANN list up to 200) but not the ANN
retriever. This entry replaces those counts with a split chosen on validation from a small grid, in which
section 2's counts are one point. Everything else in section 2 and the entries above stays.

**Inputs, all read-only**
- **Checkpoints:** the R4b final runs of TIGER, the HSTU-style model and HSTU-SID, seeds 0, 1 and 2, checked
  against `results/r4b/SHA256SUMS` before use.
- **Item vectors:** R3's vLLM Qwen3-Embedding-0.6B vectors (`data/processed/electronics/emb/vllm.npy`, checked
  against `results/r3/SHA256SUMS`). They are the vectors behind the semantic IDs. Every one of the 368,228
  catalog items has one, including the 38,783 items outside the train split, and all have unit length.
- **Rows:** the protocol's 100,000-row validation sample for every choice below; all 1,199,539 eligible test
  rows for the test pass. Section 2's optional 200,000-row test sample is not used.
- **History mask:** every source and ranker removes the 50 most recent earlier items, as in R4a and R4b.

**TIGER beam source**
- **Beam:** width 200, constrained to the catalog's semantic IDs, with the history removed at the fourth level,
  as in the R4a entry. It yields up to 200 items, ranked by the sum of their four tokens' log-probabilities.
  Empty slots are skipped wherever the list is used.
- **Why 200:** TIGER's recall@100 and @200 need that many items. The top 50 of a width-200 beam can differ from
  R4b's width-50 beam, so R4b's TIGER numbers are not reused for this source. They stay as R4b reported them.
- **Cold items inside the source,** chosen on validation with the union below, between two options:
  - off: the plain beam;
  - on: R4b's cold-start rule (paper section 4.3) over the 200-item list, at the paper's ε = 0.1. Unseen items
    sharing (c1, c2, c3) with any of the 200 beam items are the candidates, ordered by their own full-ID score.
    They are placed by R4b's rule, so when the beam itself returns no unseen item they take positions 10, 20,
    ..., 200. ε itself is not tuned.

**ANN source**
- **Index:** FAISS `IndexHNSWFlat` over all 368,228 item vectors, inner product, M = 32, efConstruction = 200.
  It is built once, saved under `data/runs/r4c/` and checksummed. The build is multi-threaded and therefore not
  bit-reproducible; the saved index is the one every result uses.
- **Search:** efSearch = 512 and k = 250. The history items are removed and the first 200 remaining items, in
  the index's order, are the ANN list.
- **Query:** q = sum over j < min(n, h) of γ^j v(x_j), normalized to unit length. Here x_0 is the most recent of
  the h items in the row's 50-item input, x_1 the one before, and v the item vector.
- **Grid for n and γ:** n in {1, 2, 3, 5, 10, 20, 50} and γ in {1.0, 0.8, 0.6, 0.4}, where γ = 1.0 is the plain
  mean. With n = 1 the weights do not matter, so it counts once: 25 queries.
- **Selection:** the highest ANN recall@200 over all rows of the validation sample. Ties go to the smaller n,
  then the larger γ.
- **Recall against exact search,** reported on the validation sample for the selected query:
  - exact search is the same query scored against every item by an exact inner product (FAISS
    `IndexFlatIP`), with the same history removal, top 200;
  - the recall is |ANN list ∩ exact list| / 200, averaged over rows. The target recall@50, @100 and @200 of
    both lists is reported too.
- **Fallback:** if that recall is below 0.95, efSearch becomes 1024, then 2048, and the selection is repeated at
  the new value. This happens before the union is selected.

**Union of the two sources**
- **Rule at a budget of K candidates,** with a TIGER share s and a backfill source b:
  - take TIGER's first round(s K) items (halves round up) and ANN's first K - round(s K) items;
  - remove duplicates;
  - fill up to K with the next items of source b that are not yet in the set, then with the other source's.
- **Grid,** at K = 200:
  - TIGER items 200 s in {25, 50, 75, 100, 125, 150, 175};
  - backfill from the ANN list or from TIGER's list;
  - TIGER's cold-start rule off or on.
  That is 28 points. Section 2's setting is (50, ANN, off).
- **Selection:** the highest union recall@200 over all rows of the validation sample, averaged over the three
  TIGER checkpoints. The ANN list is the same for all three. Exact ties go to the first point in this order:
  rule off before on, then the smaller TIGER share, then ANN backfill before TIGER backfill.
- **Budgets 50 and 100** use the selected s, b and rule. They are retrieval numbers only. The rankers always
  get the K = 200 set.

**Pairing**
- TIGER seed s goes with ranker seed s, for the HSTU-style model and for HSTU-SID.
- The ANN list does not depend on a seed.

**Re-ranking**
- **Scores:** each ranker reads its usual 50-item input and scores the union's candidates with its R4b scoring
  table:
  - HSTU-style: the L2-normalized item-ID table divided by the temperature;
  - HSTU-SID: the summed embedding of every catalog item, normalized the same way.
- **HSTU-style and items outside the train split:** they have no ID row. As in R4a and R4b, they rank below
  every train candidate, and a target outside the train split is a miss. Its cold metrics are therefore 0 by
  construction.
- **HSTU-SID** scores every candidate, cold candidates included.
- **Rank:** the target's rank among the candidates is 1 + the number of candidates scored higher + the number
  scored equal with a lower catalog index. A target outside the 200 candidates is a miss.
- **Metrics:** Recall@10, Recall@50 and NDCG@10 over all rows, warm rows and cold rows.
- **Comparison:** against the same model's R4b full-catalog test numbers. Those are read from the R4b records
  and not recomputed.

**What the test split is used for**
- For each seed, one test pass:
  - TIGER's width-200 beam;
  - the selected union, built from that beam and the ANN list;
  - both rankers on that union.
- **Reported on test:**
  - recall@50, @100 and @200 of TIGER alone (the selected source), ANN alone and the union, over all, warm and
    cold rows;
  - if the rule is on, the plain beam's recall from the same pass as well;
  - the re-ranked metrics above.
- No other grid point is evaluated on test, and nothing is selected on test.

**Checks and analysis on validation (nothing selected from them)**
- Each ranker checkpoint's full-catalog NDCG@10 on the validation sample is recomputed and compared with the
  best-epoch value in its R4b record.
- The rankers also re-rank the selected union on validation.

**Reporting and the GPU-hour stop**
- Numbers are mean ± sample standard deviation over seeds 0, 1 and 2. Section 2's rule for calling a difference
  applies.
- One validation pass is timed before the test passes, and the total is projected from it.
- R4c stops and reports if the projection passes 8 GPU-hours. A GPU-hour is an hour of wall time holding the
  RTX 4090 lock. CPU-only steps (the index build and the ANN searches) are reported separately.
- R4c changes no R4b record or checkpoint.

### 2026-11-07: R5 serving benchmark, written before any R5 measurement
R5 measures how fast the R4c cascade serves on the RTX 4090 workstation. Nothing is trained and no model
changes. The test split only supplies request histories for timing: no metric is computed on it and nothing is
chosen from it. Every choice below that is not fixed here is made on validation rows.

**What is served**
- **Models:** TIGER seed 0 and HSTU-SID seed 0, the R4b final runs, checked against `results/r4b/SHA256SUMS`;
  the R4c HNSW index, checked against `results/r4c/ann/build.json`.
- **Cascade:** R4c's selection. TIGER's beam (R4c's beam code unchanged: constrained, history removed, no
  cold-start rule, bf16 autocast) and the HNSW search (query: the newest item's vector; efSearch 512, k 250, history removed, 200 kept)
  feed the union (TIGER's first 175 items, ANN's first 25, duplicates removed, filled from TIGER's list, then
  from ANN's). HSTU-SID scores the 200 candidates.
- **Request:** one user history: the user's hashed token, the 50 most recent items before the target with
  their timestamps, and the request time, which is the target's timestamp (the time HSTU's bias reads). The
  server derives every model input from it on the GPU: the ranker's 50-item window and query times, TIGER's
  20-item encoder input, the 50-item history mask and the ANN query.
- **Response:** the top 10 of the re-ranked candidates, as catalog indices in host memory.

**Server**
- In-process: one Python thread, one CUDA stream, no network layer and no serialization.
- **One batch of the cascade:**
  1. the requests are gathered into pinned host buffers and copied to the GPU;
  2. TIGER's beam is launched;
  3. while the GPU runs it, the HNSW search runs on the CPU, and its lists are copied to the GPU;
  4. the union, then the ranker, run on the GPU;
  5. the top 10 is copied to the host.
- **CPU threads:** 16 OpenMP threads, shared by faiss and torch.
- **TF32** is off throughout R5.
- **Python's garbage collector** keeps its default settings.

**SLO:** p99 latency at most 50 ms. Latency runs from a request's scheduled arrival to its top 10 being in host
memory. The ladder applies the same SLO to the ranker alone, as if it had the whole budget, so its QPS are
upper bounds for a ranker that shares the budget with retrieval.

**Request samples**
- **Test pool:** the first 50,000 test rows in the protocol's sha256 order, issued in that order from the
  start in every run, cycling when a run needs more.
- **Validation pool,** for the batch-size choice only: the first 50,000 validation rows in the same order. All
  of them lie inside the protocol's 100,000-row validation sample.
- **Candidates for the ranker alone:** each request carries its row's 200-item R4c ANN list
  (`ann_test.npz` or `ann_valid.npz`). The ranker's cost does not depend on which items it scores.

**Load generator**
- Open-loop Poisson arrivals: exponential gaps drawn in advance from numpy's PCG64, seeded per run. Arrivals
  never wait for the server.
- Latency counts from the scheduled arrival, so queueing delay is included and coordinated omission is avoided.
- An idle server waits for the next scheduled arrival.
- A run is aborted and fails if a waiting request is more than 1 s old.

**Batching**
- Ladder steps 1 to 4 serve one request at a time, oldest first.
- **Dynamic batching:** when the server is free and requests are waiting, it dispatches once B_max requests are
  waiting or the oldest has waited W = 2 ms, whichever comes first. The batch is the oldest B_max waiting
  requests.
- The ranker's batches are padded to the next power of two for its CUDA graphs; TIGER runs at the exact batch
  size.
- **B_max,** per configuration, on the validation pool: the largest power of two up to 1,024 whose p99 batch
  service time (50 back-to-back batches after 10 warm-up batches) is at most SLO/3 (16.7 ms). Reason: a
  request can wait for the batch in service, the window and its own batch, 2 SLO/3 + W in all, which leaves
  the rest of the budget for Poisson bursts. If even a batch of 1 exceeds SLO/3, B_max is 1.

**QPS at the SLO**
- **Saturation X:** back-to-back batches of B_max test requests for 5 s, completed requests per second.
- **Bisection:** 7 steps on the rate, between 0 and 1.05 X.
  - Each step is one open-loop run: 2 s of arrivals as warm-up, not recorded, then 20 s measured, with at
    least 3,000 and at most 1,000,000 measured requests.
  - A step passes if its p99 is at most 50 ms and it was not aborted.
- **Confirmation:** three 30 s runs at the largest passing rate. The rate stands if all three pass. Otherwise it
  is lowered by 5% and the three runs are repeated, at most 4 times.
- **Reported:** the confirmed rate as the QPS at the SLO, each confirmation run's p50 and p99, and every
  bisection step. If no rate passes, the configuration is reported as missing the SLO.

**Warm-up and repetitions**
- Each configuration loads its models, captures its CUDA graphs, then serves 200 requests or 20 batches before
  any timed request.
- **Batch-1 latency:** 2,000 requests served one after another, each synchronized, after 200 untimed ones.
- **Under load:** the 2 s warm-up of each run, then the three confirmation runs above.

**Ranker ladder** (HSTU-SID seed 0, ranker alone; each step keeps the steps before it)
1. **fp32,** eager, batch 1, TF32 off. This is the reference.
2. **bf16:** encoder weights and activations in bf16. The candidate score stays the fp32 dot product of the
   state with the fp32 scoring table, R4b's table, computed once at load.
3. **CUDA graphs:** the whole ranker step (inputs, encoder, scores, top 10) is captured once per batch size.
   CUDA graphs are used instead of torch.compile.
4. **User-history cache:**
   - The cache holds, per layer, the keys and values of the 49 older items of the window. They do not depend
     on the request time.
   - The newest item's row reads the request time in its time bias, so its output, and from layer 2 on its
     keys and values, depend on it. That row is always recomputed.
   - **Hit:** read the cached keys and values and compute the newest row only.
   - **Miss:** run the full 50-row encoder and write the cache.
   - The cache is a GPU table; a host dictionary maps a request's key to its slot.
   - Hits and misses are timed in separate runs: all hits, with the cache filled beforehand, and all misses.
5. **Dynamic batching,** as above, with hits and with misses.

For each step: batch-1 p50 and p99, the QPS at the SLO, the p50 and p99 at that rate, and candidates scored
per second at that rate (200 times the QPS).

**Ranking quality of each step**
- **Sample:** the protocol's 100,000-row validation sample, with R4c's union for TIGER seed 0, built from the
  saved lists.
- Each step scores the candidates through its own serving path. Steps 1 to 4 run at batch 1; step 5 runs in
  batches of its B_max. Hits read a cache that the miss path filled.
- **Against step 1:**
  - top-10 overlap: the mean of |top 10 ∩ fp32 top 10| / 10, with the top 10 by score and ties to the lower
    catalog index;
  - the change in NDCG@10, with the protocol's rank and tie-break.
- **A step keeps quality** if its top-10 overlap is at least 0.95 and its NDCG@10 moves by at most 1% of step
  1's. A step that fails is reported, and the ladder continues without it.
- **Check:** step 1's NDCG@10 is compared with R4c's validation record for HSTU-SID seed 0, which was computed
  with TF32 in batches of 1,024.

**Stage latencies**
- **At batch 1,** as above, for each of:
  - TIGER's beam at widths 20, 50 and 200, from the encoder input on the GPU to the item lists;
  - the HNSW query on the CPU: the query, the search and the history removal;
  - the union on the GPU;
  - the ranker in its final ladder configuration, with misses;
  - the whole cascade at each width.
- **Under load:** the cascade at widths 200, 50 and 20, each at its own QPS at the SLO, with B_max chosen per
  width.
  - The ranker runs in its final ladder configuration with misses: each request is a new history.
  - Per batch, the time of each stage is recorded (CUDA events for the GPU stages, the host clock for the HNSW
    search), with the batch size and the queueing delay.
  - The dominant stage is the one with the largest share of batch time.
- **With widths 20 and 50,** the union rule is unchanged: it takes all w TIGER items and fills the other
  200 - w from ANN's list.

**What a narrower beam costs**
- A short validation pass runs TIGER seed 0 at widths 20 and 50 over the 100,000-row validation sample, with
  R4c's beam code.
- Reported for widths 20, 50 and 200: union recall@200 over all, warm and cold rows, and HSTU-SID seed 0's
  NDCG@10 on that union (fp32, R4c's re-ranking code). Width 200 uses R4c's saved lists.
- It is reported whether or not width 200 meets the SLO.

**Serving-path check:** the cascade serves the first 2,000 validation rows at width 200.
- Its ANN lists must equal R4c's saved lists.
- Its TIGER lists are compared with R4c's saved lists: the share that are identical and the mean overlap. The
  batch sizes differ from R4c's, so bf16 rounding can reorder near-ties.

**Limits and background load**
- **Hardware:** one consumer GPU. Its clocks are not locked, since there is no root access.
- **Other services** run on the machine. A sidecar process records `/proc/loadavg` every 2 s and nvidia-smi
  (clocks, power, temperature, utilization, memory and other GPU processes) every 10 s. The busiest CPU
  processes are listed at the start and end of every command. All of this is reported with the results.

**The GPU-hour stop:** a short timed pass comes first, and the total is projected from it. R5 stops and reports
if the projection passes 6 GPU-hours. A GPU-hour is an hour of wall time holding the RTX 4090 lock.

### 2026-11-08: R5b fast TIGER beam, written before any R5b measurement
R5 found TIGER's beam to be most of the cascade's time: 8.2 to 8.4 ms at batch 1 at every width, with no KV cache
and hundreds of kernel launches per request. R5b makes the beam faster without changing what it returns, then
re-measures the cascade. Nothing is trained and no model changes. Test rows only supply request histories for
timing. **Everything not named here follows the R5 entry above, unchanged.**

**Profile:** torch.profiler on R5's beam (R4c's beam code, bf16 autocast), on the first 64 requests of the test
pool, at widths 20, 50 and 200, at batch 1, at R5's B_max of each width and at batch 64. Per call: kernel time
and count, launch calls, host syncs, and the kernel time inside the encoder, inside the decoder (with its
self-attention, cross-attention, the cross-attention keys and values, and feed-forward) and in the rest of the
search. The final beam is profiled at the same settings. Wall time is the median of 20 synchronized calls
without the profiler.

**Beam ladder** (each step keeps the steps before it)
1. **Reference:** R5's beam.
2. **Cross-attention once:** each decoder layer's cross-attention keys and values are computed once per request
   from the encoder output, not once per beam at every step. The beams of a request are the query rows of one
   attention.
3. **KV cache:** each step runs the decoder on the newest token only and attends to the cached self-attention
   keys and values of the earlier tokens. The cache follows the beams when they are reordered. The first step
   runs once per request, since every beam starts from the same token.
4. **Static tensors:** what does not depend on the request is computed at load: the encoder's position bias, each
   decoder step's position bias, and the weights in bf16. The model runs with explicit dtypes instead of
   autocast, with the same casts. Nothing in the loop waits for the host.
5. **CUDA graphs:** the whole serving call, from the request's items to the lists, is captured as one CUDA graph
   per width and power-of-two batch size. A batch is padded with copies of its first request, so TIGER now runs
   at the padded size (R5: the exact size). CUDA graphs are used instead of torch.compile.

The search itself is R4c's: the same trie tables, history removal, scores, candidate tensor and top-k.

**Ladder measurements:** the beam alone, bf16, test pool, per step and width:
- batch-1 latency, as in R5: 2,000 requests one after another, each synchronized, after 200 untimed;
- 50 back-to-back batches of 64 requests after 10 untimed: p50 and p99 batch time and requests per second.

**Equivalence in fp32** (TF32 off): the first 2,048 rows of the validation pool, at widths 20, 50 and 200.
- Steps 2 to 5 run in fp32 against the reference in fp32, both in chunks of 128. Step 5 also runs at batch 1
  against the reference at batch 1.
- **The bar:** the same item lists in the same order. Reported per step and width: the share of identical lists
  and, for each mismatch, where the lists part, with both beams' items and scores there. Also the largest
  score difference where the items agree, and, for scale, how much the reference's own lists change between
  chunks of 128 and batch 1.
- A mismatch is a near-tie if the reference's two scores at the parting position differ by at most 1e-4.
  Any other mismatch is a defect: the beam is fixed and this check repeated before any timing.
- Step 5 in bf16 is also compared with R5's bf16 beam at batch 1. This is reported, not a bar.

**Quality at the serving dtype** (bf16): step 5 runs over the protocol's 100,000-row validation sample in chunks
of 128, as R4c and R5 ran, at widths 200, 50 and 20.
- Reported: union recall@200 over all, warm and cold rows, and HSTU-SID seed 0's NDCG@10 on that union (fp32,
  R4c's re-ranking code), each against R5's number. Width 200 compares with R4c's saved lists, widths 50 and
  20 with R5's lists. Also the share of lists identical to those.
- No threshold: the changes are reported as they are.

**Re-measurement:** the cascade with step 5 in bf16 and R5's final ranker (bf16, CUDA graphs, misses), by R5's
protocol: the same SLO, request pools, load generator, dynamic batching, B_max rule, saturation, bisection and
confirmations.
- **B_max:** chosen again per width on the validation pool by R5's rule. Each batch size's CUDA graph is
  captured during its first warm-up batch, which is not timed.
- **Batch 1:** R5's stage latencies (the beam at each width, the HNSW query, the union, the ranker, the whole
  cascade at each width).
- **Under load:** the cascade at widths 200, 50 and 20, with per-batch stage times.
- **Before:** R5's committed records, which R5b reads and does not rerun or change. The ladder's step 1
  re-measures R5's beam alone in R5b's sessions.

**Background load:** R5's sidecar runs for every command. Known before the runs: a QEMU VM (`td-legacy-vm`, about
40% of a core), a kube-apiserver and desktop apps. They belong to others and are not stopped.

**The GPU-hour stop:** a short timed pass comes first, and the total is projected from it. R5b stops and reports
if the projection passes 4 GPU-hours.
