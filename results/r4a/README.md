# R4a: the three models at Electronics scale, their cost, tuning and DDP scaling

R4a brings SASRec, the HSTU-style model and TIGER to the full Electronics catalog. It measures what they cost,
runs the protocol's 12 tuning trials per model on the dev subset, and measures DDP scaling on two A10 vGPUs.
The full-split seeds, the cascade and the cold-start results are R4b.

All runs follow [docs/protocol.md](../../docs/protocol.md). The entry "2026-11-06: Electronics training and
evaluation details" fixes what sections 2 and 3 left open. It was committed (0730bc6) before any Electronics
run. R4a computes no test metric.

**Numbers below:** they come from [summary.json](summary.json), which `python -m sidrec.eval.r4a_summary` builds
from the run records in this folder.

## How the models run at this scale
- **Ranking:** every model ranks the full catalog of 368,228 items, with the user's 50 most recent history
  items removed.
  - The item-ID models score the train items (329,445 on the full split, 184,738 on the dev subset). Any
    other target is a miss.
  - TIGER's beam is constrained to the semantic IDs of all 368,228 items, so it can return cold-start items.
- **Metrics:** Recall@10, Recall@50 and NDCG@10 over all rows, and separately over warm and cold rows.
- **Exact full-softmax loss:** the item-ID models' loss is the plain softmax cross-entropy over every train
  item, at every position.
  - Over 329,445 items, one batch's logits would take several GB, and autograd would keep them for the
    backward pass.
  - `sidrec.train.fused_ce` walks over the rows in chunks of at most 2^28 logits (1 GiB in float32). For each
    chunk it computes the logits, their log-sum-exp and, because the loss is the last node, the gradients
    right away: softmax - onehot gives dL/dh = G W and dL/dW += G^T h.
  - Only one chunk of logits exists at a time. The backward pass only scales the stored gradients.
  - Nothing is sampled or approximated. A test checks the loss and both gradients against
    `F.cross_entropy` on the full logits, for SASRec's table and for the HSTU-style model's normalized one.
- **TIGER's fourth code:** its token range has 461 codes, the size of R3's largest collision group. The
  vocabulary is 3,231 tokens.
- **Inputs:** the windows are cut on the GPU from the pipeline's user timelines (`sidrec.data.scale`). The
  input cache is checked against the tables when it is built:
  - every evaluation row's history column equals the timeline before its target;
  - the cold-start flags equal "target not in this source's train split";
  - the counts equal those in [data.md](../../docs/data.md).

## Cost, measured before tuning
Each model trained one epoch on the dev subset and on the full train split, on the RTX 4090. Two
configurations were measured:
- trial 1, the largest in the trial list: width 256, 4 layers;
- trial 4: width 64, 2 layers.

After the epoch, each run evaluated the per-epoch validation sample, then timed one evaluation of every
eligible validation row. That second evaluation stands in for the test split, which R4a does not touch.
Records: `cost/<model>_<source>_trialNN.json`.

| Model | Data | Config | Params | Steps / epoch | Train epoch | ms / step | Per-epoch eval | All 1,141,362 validation rows | Peak memory |
|---|---|---|---|---|---|---|---|---|---|
| SASRec | dev | 256 × 4 | 50.5M | 386 | 14.7 s | 38.1 | 1.8 s | — | 4.0 GB |
| SASRec | dev | 64 × 2 | 11.9M | 386 | 9.3 s | 24.1 | 1.2 s | — | 3.4 GB |
| SASRec | full | 256 × 4 | 87.5M | 6,160 | 378 s | 61.3 | 3.7 s | 42 s | 7.3 GB |
| SASRec | full | 64 × 2 | 21.2M | 6,160 | 257 s | 41.7 | 2.8 s | 31 s | 6.3 GB |
| HSTU-style | dev | 256 × 4 | 48.6M | 771 | 33.1 s | 42.9 | 1.9 s | — | 4.2 GB |
| HSTU-style | dev | 64 × 2 | 11.9M | 771 | 23.8 s | 30.9 | 1.2 s | — | 3.4 GB |
| HSTU-style | full | 256 × 4 | 85.7M | 12,319 | 768 s | 62.3 | 3.8 s | 43 s | 7.6 GB |
| HSTU-style | full | 64 × 2 | 21.1M | 12,319 | 508 s | 41.2 | 2.8 s | 32 s | 6.3 GB |
| TIGER | dev | 256 × 4 | 9.7M | 2,796 | 56.4 s | 20.2 | 51.6 s | — | 8.5 GB |
| TIGER | dev | 64 × 2 | 1.3M | 2,796 | 20.9 s | 7.5 | 16.0 s | — | 5.8 GB |
| TIGER | full | 256 × 4 | 9.7M | 44,890 | 908 s | 20.2 | 72.1 s | 822 s | 8.9 GB |
| TIGER | full | 64 × 2 | 1.3M | 44,890 | 322 s | 7.2 | 22.2 s | 253 s | 6.2 GB |

- **Per-epoch evaluation:** all 71,610 eligible validation rows on the dev subset, the 100,000-row sample on
  the full split.
- **The full-softmax loss dominates the item-ID models.** Going from width 256 × 4 layers to 64 × 2 cuts the
  step time by only a third.
- **TIGER's beam search dominates its evaluation.** The width-50 beam ranks about 1,400 rows per second at
  256 × 4. The item-ID models rank about 27,000 rows per second over all 329,445 train items.
- **Precision:** the item-ID models run in float32 with TF32 matrix products, TIGER in bfloat16 autocast, as on
  Beauty.

### GPU-hour estimates (RTX 4090)
The formulas are in `sidrec/eval/r4a_summary.py`:
- one epoch is a training epoch plus one per-epoch evaluation;
- a test evaluation is the timed all-validation evaluation, scaled to 1,199,539 rows;
- the cascade is costed per seed as TIGER's beam over the test rows plus one SASRec and one HSTU-style
  scoring of them.

| | Config of every run | Epochs per run | SASRec | HSTU-style | TIGER | Cascade | Total |
|---|---|---|---|---|---|---|---|
| R4a tuning, 12 trials per model | 256 × 4 | 30 (the cap) | 1.7 | 3.5 | 10.8 | — | **16.0** |
| | 64 × 2 | 30 | 1.1 | 2.5 | 3.7 | — | 7.2 |
| R4b, 3 seeds per model and the cascade | 256 × 4 | 10 | 3.2 | 6.5 | 8.9 | 0.8 | 19.4 |
| | 256 × 4 | 30 | 9.6 | 19.3 | 25.2 | 0.8 | **54.9** |
| | 64 × 2 | 10 | 2.2 | 4.3 | 3.1 | 0.3 | 9.8 |
| | 64 × 2 | 30 | 6.5 | 12.8 | 8.8 | 0.3 | 28.4 |

- **R4a tuning:** at most 16 GPU-hours, even if every trial had the largest configuration and ran all 30
  epochs. That is under the 30-hour stop line, so tuning went ahead as the protocol states.
- **R4b:** 10 to 55 GPU-hours on the 4090, depending on the selected configurations and on how many epochs
  early stopping allows. The selected configurations are below.
