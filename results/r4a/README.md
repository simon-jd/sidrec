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
- **Inputs:** the windows are cut on the GPU from the pipeline's user timelines (`sidrec.data.scale`).
  - When the input cache is built, it checks that the catalog order is R3's.
  - It also checks that every evaluation row's history column equals the timeline before its target.
  - And it checks that the cold-start flags mean "target not in this source's train split".
  - A test checks the cache's counts against [data.md](../../docs/data.md).

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
  early stopping allows.
  - All three selected configurations turned out to be 256 × 4 (next section), so the 256 × 4 rows apply.
  - That is 19 GPU-hours if the full-split runs stop near 10 epochs, and 55 at the 30-epoch cap.
  - On the dev subset the selected trials peaked at epochs 6 to 13.

## Tuning on the dev subset
Each model got the protocol's 12 shared trials on the dev subset, all on the RTX 4090, with seed 0.
- Every epoch evaluates all 71,610 eligible validation rows. Each trial early-stops with patience 3, at most 30
  epochs.
- Selection uses validation NDCG@10 over those rows.
- No test metric was computed.

Records: `runs/<model>/trialNN.json`. Checkpoints are in `data/runs/r4a/` and listed in [SHA256SUMS](SHA256SUMS).

| Model | Selected trial | Configuration | Best epoch | NDCG@10 | Recall@10 | Recall@50 | Warm NDCG@10 | Cold Recall@50 |
|---|---|---|---|---|---|---|---|---|
| SASRec | 1 | lr 1e-4, dropout 0.3, weight decay 1e-4, width 256, 4 layers | 8 | **0.00747** | 0.01448 | 0.03745 | 0.01106 | 0 |
| HSTU-style | 1 | lr 1e-4, dropout 0.3, weight decay 1e-4, width 256, 4 layers | 13 | **0.00676** | 0.01254 | 0.03106 | 0.01002 | 0 |
| TIGER | 2 | lr 3e-4, dropout 0, weight decay 0, width 256, 4 + 4 layers | 6 | **0.00599** | 0.01137 | 0.02857 | 0.00887 | 0 |

These are validation numbers on the dev subset. The dev subset has 102,587 users. 23,262 of its 71,610
validation rows (32%) have a cold target, so every model's overall numbers are lower than on warm rows.

Validation NDCG@10 of all 12 trials, with the best epoch after the @:

| Trial | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SASRec | **.0075@8** | .0072@3 | .0074@13 | .0070@3 | .0071@4 | .0067@3 | .0072@3 | .0071@5 | .0030@9 | .0070@2 | .0073@4 | .0063@14 |
| HSTU-style | **.0068@13** | .0046@3 | .0054@8 | .0059@3 | .0041@3 | .0058@3 | .0061@2 | .0032@3 | .0064@3 | .0060@3 | .0058@3 | .0030@3 |
| TIGER | .0037@16 | **.0060@6** | .0033@10 | .0030@13 | .0054@8 | .0051@6 | .0054@6 | .0058@7 | .0044@6 | .0046@6 | .0045@9 | .0042@9 |

The configurations are listed in the protocol (section 3) and in each record.

- **The item-ID models peak early,** mostly at epoch 2 to 5. Most of the later peaks are at learning rate
  1e-4. TIGER peaks later, at epochs 6 to 16.
- **On the dev subset, SASRec leads.** The HSTU-style model is about 10% below it on NDCG@10, and TIGER is
  about 20% below. On Beauty the HSTU-style model led (R2).
  - These are single-seed dev-subset numbers, so the protocol does not call them a ranking.
  - The full-split seeds in R4b decide it.
- **GPU time:** 0.39 hours for SASRec, 0.72 for the HSTU-style model and 2.43 for TIGER, 3.5 in all. That is
  well under the 16-hour bound in the cost section, because early stopping cut most trials short.

### TIGER almost never emits a cold-start item
Every model's cold-start Recall@50 is 0 or close to it.
- **The item-ID models:** 0 by construction (protocol).
- **TIGER:** its beam may return any of the 368,228 items, but the selected trial found none of the 23,262 cold
  targets.
- **A diagnostic** (`scripts/r4a_tiger_cold_share.py`, record `tiger_cold_share.json`) counted what the
  selected checkpoint emits on the dev validation rows:
  - 756 of its 3,580,500 emitted items (0.02%) are outside the dev subset's train split;
  - they come from 734 of the 71,610 rows;
  - yet 49.8% of the catalog is outside the dev train split.
- **Reading:** the decoder gives nearly all its probability to semantic IDs it saw as targets.
  - After one epoch, in the cost runs, cold Recall@50 was 0.0018 to 0.0028 on the dev subset and 0.00001 to
    0.00002 over all full-split validation rows.
- **Why this matters for R4b:** the full split's cold rows are 44.7% of test, and the plan's cold-start story
  needs TIGER to rank them. TIGER's paper evaluates cold-start retrieval with an extra step: a share ε of the
  top-K is reserved for unseen items that match the generated semantic-ID prefixes.
  - The protocol has no such step, and R4a did not add one.
  - Whether to add one is a protocol decision for R4b, to be made before any test run.

## DDP scaling on two A10 vGPUs
The study compares 1 against 2 GPUs at the same global batch. The user approved Snowflake credits for it.

**Hardware:** "two A10-24Q vGPUs in one Snowflake node", on PORTFOLIO_2XA10_POOL. The node runs driver 570 with
the CUDA 13.2 forward-compatibility driver. The software is torch 2.13.0+cu130 with NCCL 2.29.7, from the frozen
vLLM 0.31.0 wheelhouse.

**Interconnect:** there is no NVLink.
- `nvidia-smi topo -m` shows SYS between the two GPUs: PCIe plus the link between CPU sockets. GPU 0 is on NUMA
  node 2 and GPU 1 on NUMA node 0.
- `nvidia-smi topo -p2p r` reports peer-to-peer reads as not supported.
- NCCL connected both channels "via SHM/direct/direct", that is, through host shared memory.

**Model:** the HSTU-style ranker in its selected configuration, trial 1: width 256, 4 layers, learning rate
1e-4, dropout 0.3, weight decay 1e-4. Its tuning had finished, so this is the selected configuration, not a
fallback; it is also trial 1. It trains on the full Electronics train split: 1,576,803 windows and a full
softmax over 329,445 items.

**Method** (`sidrec.train.ddp_scaling`, run by `ddp/payload_sidrec_ddp.py`):
- Every run draws the same global batches, a seed-0 permutation of the training windows.
- With 2 processes (torchrun on the one node), each takes half of every global batch. Each scales its loss so
  that DDP's gradient average equals the 1-GPU gradient on the whole batch. A test checks this on the CPU.
- 50 untimed and 500 timed steps at global batch 128, the protocol's HSTU batch. 20 untimed and 150 timed
  steps at global batch 1024, as a second point.
- Each step ends with a CUDA synchronize. nvidia-smi is sampled every second during the timed steps.
- The mean training losses of the two settings agree: 13.6160 against 13.6165 at batch 128, and 13.6179
  against 13.6188 at batch 1024.

| Global batch | GPUs | Batch per GPU | Step time (mean) | Windows/s | Speedup | Scaling efficiency | GPU utilization (median) |
|---|---|---|---|---|---|---|---|
| 128 | 1 | 128 | 127.7 ms | 1,002 | | | 99% |
| 128 | 2 | 64 | 243.8 ms | 525 | 0.52× | **26%** | 98%, 98% |
| 64 | 1 | 64 | 85.9 ms | 746 | | | 97% |
| 1024 | 1 | 1024 | 770.3 ms | 1,329 | | | 99% |
| 1024 | 2 | 512 | 569.6 ms | 1,798 | 1.35× | **68%** | 99%, 99% |
| RTX 4090 reference, 128 | 1 | 128 | 63.0 ms | 2,030 | | | 98% |
| RTX 4090 reference, 64 | 1 | 64 | 44.1 ms | 1,450 | | | 96% |
| RTX 4090 reference, 1024 | 1 | 1024 | 343.3 ms | 2,983 | | | 100% |

- **Scaling efficiency** is the 1-GPU step time divided by twice the 2-GPU step time.
- **RTX 4090 rows:** the same payload, the same torch build and the same steps, run on the RTX 4090 workstation.

**Why two GPUs are slower at batch 128:** the all-reduce is large and slow.
- **Gradient size:** 327 MB per step. Most of it is the 329,447 × 256 item table, and the full softmax makes its
  gradient dense.
- **All-reduce speed:** an NCCL all-reduce of 327 MB took 153 ms on this node, a bus bandwidth of 2.2 GB/s.
- **No overlap:** the item table's gradient is complete only at the end of the backward pass, because the table
  is also the input embedding. DDP therefore cannot overlap its all-reduce with computation.
- **The sum matches:** one A10 on the 64-window share took 85.9 ms, and 85.9 + 153 ms is close to the
  measured 243.8 ms.
- **At batch 1024:** each GPU computes for much longer, the all-reduce stays about the same, and two GPUs come
  out 1.35 times faster than one.

**GPU utilization does not show the cost.** nvidia-smi reports 98–99% on both GPUs even when communication
dominates. It counts the time any kernel runs, and NCCL's copy kernels are kernels too. The step times show the
cost.

**What this means for R4b:**
- At the protocol's batch of 128, two A10s are 1.9 times slower than one.
- One RTX 4090 is 2.0 times faster than one A10 at batch 128. At batch 1024 it is 1.7 times faster than both
  A10s together.
- The full-split HSTU runs are therefore cheapest on the 4090.
- DDP on this node would need a much larger batch per GPU, which the protocol fixes, or a smaller all-reduce,
  such as bfloat16 gradient compression. Neither was tried.

**Credits:** 1.248 by the ledger's wall-time estimates (wall time × 3.50 credits per hour). The ledger went
from 29.739 to 30.987, under R4a's limit of 4.
- **Smoke test** (`20261106T102831Z-sidrec-smoke`): 0.424. It ran 40 timed steps per setting at batch 128 and
  confirmed the path before the main job.
- **Main job** (`20261106T121948Z-sidrec-ddp`): 0.824. It took 848 s, of which about 3 minutes were pool
  start-up and about 1.5 minutes fetching and installing the environment.

## Provenance and what went wrong
- **Order:**
  - The protocol entry (0730bc6) and the training code (e96df35) were committed before the first recorded run.
  - The dev-subset cost rows and the tuning estimate (7dbe095) were committed before tuning started.
  - The full-split rows (495a3af) followed while tuning ran.
- **Smoke tests** of the three scripts ran from the uncommitted tree before those commits. Their records were
  deleted and are not counted.
  - The input-cache build failed once, on the pipeline's 0-based `pos` column, before any run.
- **SASRec trial 9:** it started while an uncommitted edit to `sidrec/eval/r4a_summary.py` was in the tree.
  Training does not import that module, but the record was marked dirty. It was rerun from a clean tree, giving
  validation NDCG@10 0.00301 as before. The first record is in the history (cd49823).
- **Commits recorded by the runs:** every run record names its starting commit. After e96df35, no commit
  changed the training code: the input module, the loss, the models or the two training scripts. No record in
  this folder is marked dirty.
- **DDP smoke job:** its payload kept the first 16 NCCL lines, and those were initialization lines, so the
  transport went unrecorded. The main job's payload, the version in `ddp/`, keeps the "via" lines first.
  Nothing else changed between the two.
- **Snowflake:** no other pool and no warehouse query was used.

## GPU time on the RTX 4090
- **Recorded:** about 4.9 hours. The cost runs took 1.33, the tuning trials 3.54, plus the SASRec trial 9
  rerun and the TIGER diagnostic.
- **Not recorded:** about 0.2 hours more. That is the payload dry runs, which gave the 4090 reference rows,
  and the smoke tests.

## Artifacts outside git
Check them from the repository root with `sha256sum -c results/r4a/SHA256SUMS`.

| Path | Contents |
|---|---|
| `data/processed/electronics/r4/full.npz`, `dev.npz` | the input caches built by `sidrec.data.scale` |
| `data/runs/r4a/<model>/trialNN.pt` | the best-epoch weights of every tuning trial, 36 files, 3.1 GB |

The Snowflake job used two files uploaded to `@PORTFOLIO_LAB.GPU.ASSETS`:
- `sidrec-src-c73d0d2.tar`, sha256 `078638f86901f60b003f0684605008397e0c3686f4a9926ddf9c45cee2c3a7c1`:
  `git archive` of `src/sidrec` at c73d0d2, the DDP code as committed;
- `sidrec-electronics-r4-full.tar`, sha256 `b02bec02a8a12ec733b6764c76bdcf3241fe53b4f77459ad58b8a3c8b03a8f4f`:
  `full.npz` above.

## Files
| Path | Contents |
|---|---|
| `cost/` | the 12 cost records: one epoch and timed evaluations |
| `runs/<model>/trialNN.json` | the 36 tuning trials: configuration, per-epoch history, validation metrics, GPU state |
| `tiger_cold_share.json` | how many cold items the selected TIGER trial emits |
| `ddp/node-main/`, `ddp/node-smoke/` | the Snowflake jobs' `report.json` (status trace, RESULT lines, credits) and instance logs |
| `ddp/rtx4090/payload.log` | the same payload run on the RTX 4090, for the reference rows |
| `ddp/payload_sidrec_ddp.py` | the job payload |
| `summary.json` | everything above in one file, built by `python -m sidrec.eval.r4a_summary` |
| `SHA256SUMS` | checksums of the artifacts outside git |
