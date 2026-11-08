# R4b: full-split test results, TIGER's cold-start retrieval, HSTU-SID and the time ablation

R4b trains the configurations R4a selected on the full Electronics train split, with seeds 0, 1 and 2, and
evaluates each final run once on the test split. It adds three things:
- TIGER's own cold-start retrieval, at inference only;
- HSTU-SID, an HSTU-style ranker whose item embedding includes semantic-ID prefixes;
- the HSTU-style model without its time bias.

The cascade is R4c and is not here.

All runs follow [docs/protocol.md](../../docs/protocol.md). The entry "2026-11-06: R4b final runs, TIGER's
cold-start retrieval, HSTU-SID and the time ablation" was committed (f3e9db5) before any R4b run. The code was
committed in 3ab2c4c, also before any run.

**Where it ran:** every process ran on the RTX 4090 workstation, under the GPU lock. No Snowflake compute was
used (0 credits).

**Numbers below:** they come from [summary.json](summary.json), which `python -m sidrec.eval.r4b_summary`
builds from the records in this folder. They are test metrics over all 1,199,539 eligible rows, given as mean ±
sample standard deviation over seeds 0, 1 and 2. The ablation has seed 0 only.

## Test results
The configurations are in `summary.json` and in each record:
- **SASRec:** trial 1, width 256, 4 layers.
- **HSTU-style:** trial 1, width 256, 4 layers.
- **TIGER:** trial 2, width 256, 4 + 4 layers.
- **HSTU-SID:** trial 6, width 64, 2 layers; its tuning is below.
- **TIGER + cold-start:** the same three TIGER checkpoints and the same beam, with the paper's retrieval at
  ε = 0.1.

### All rows (1,199,539)
| Model | Recall@10 | Recall@50 | NDCG@10 |
|---|---|---|---|
| SASRec | 0.01336 ± 0.00003 | 0.03323 ± 0.00019 | 0.00723 ± 0.00002 |
| HSTU-style | 0.01431 ± 0.00007 | 0.03513 ± 0.00018 | 0.00796 ± 0.00009 |
| HSTU-SID | **0.01512 ± 0.00020** | **0.03704 ± 0.00014** | **0.00827 ± 0.00010** |
| TIGER | 0.01296 ± 0.00016 | 0.03108 ± 0.00053 | 0.00686 ± 0.00005 |
| TIGER + cold-start, ε = 0.1 | 0.01238 ± 0.00017 | 0.03084 ± 0.00059 | 0.00669 ± 0.00005 |
| HSTU-style without time bias (seed 0) | 0.01363 | 0.03380 | 0.00745 |

### Warm rows (663,623; target seen in train)
| Model | Recall@10 | Recall@50 | NDCG@10 |
|---|---|---|---|
| SASRec | 0.02414 ± 0.00005 | 0.06006 ± 0.00033 | 0.01308 ± 0.00004 |
| HSTU-style | 0.02587 ± 0.00013 | **0.06350 ± 0.00032** | 0.01438 ± 0.00016 |
| HSTU-SID | **0.02670 ± 0.00034** | 0.06251 ± 0.00036 | **0.01470 ± 0.00019** |
| TIGER | 0.02342 ± 0.00028 | 0.05617 ± 0.00095 | 0.01240 ± 0.00008 |
| TIGER + cold-start, ε = 0.1 | 0.02193 ± 0.00027 | 0.05402 ± 0.00093 | 0.01197 ± 0.00008 |
| HSTU-style without time bias (seed 0) | 0.02463 | 0.06110 | 0.01347 |

### Cold rows (535,916; target never in train)
| Model | Recall@10 | Recall@50 | NDCG@10 |
|---|---|---|---|
| SASRec | 0, by construction | 0, by construction | 0, by construction |
| HSTU-style | 0, by construction | 0, by construction | 0, by construction |
| HSTU-SID | 0.00078 ± 0.00009 | **0.00550 ± 0.00025** | 0.00030 ± 0.00003 |
| TIGER | 0 | 0 | 0 |
| TIGER + cold-start, ε = 0.1 | 0.00057 ± 0.00010 | 0.00214 ± 0.00026 | 0.00016 ± 0.00003 |
| HSTU-style without time bias (seed 0) | 0, by construction | 0, by construction | 0, by construction |

- **"By construction":** the item-ID models rank only train items (protocol, section 2), so a cold target is
  always a miss.
- **Plain TIGER's 0 is measured, not built in.** Its beam may return any catalog item, but in no seed did a cold
  target reach its top 50. That is R4a's finding at full scale: the decoder puts almost no probability on IDs it
  never saw as targets.
- **Bold** marks the best value in a column where the gap to the next model is larger than both standard
  deviations, the protocol's rule for calling a difference.
- **HSTU-SID leads overall** on all three metrics, by more than the standard deviations.
  - On warm rows it leads on Recall@10 and NDCG@10. The HSTU-style model leads on Recall@50.
  - On cold rows it is the only model above 0.005 Recall@50, 2.6 times TIGER with cold-start retrieval.
  - **Caveat:** its selected configuration differs from the HSTU-style model's (width 64 against 256, learning
    rate 3e-3 against 1e-4). The gap therefore mixes the prefix embeddings with what each model's tuning chose.
    At trial 6 the HSTU-style model reached dev NDCG@10 0.0058, below its own selected trial.
- **Cold rows are 44.7% of test,** so every model's overall numbers sit far below its warm numbers.

## TIGER's cold-start retrieval
- **Rule:** the paper's (Rajput et al. 2023, section 4.3, Fig. 5), fixed in the protocol entry before any run.
  - Unseen items that share the first three codes (c1, c2, c3) of a beam item are candidates, ordered by their
    own full-ID score.
  - They fill at most a share ε = 0.1 of the list, at positions 10, 20, 30, 40 and 50.
  - It ran on the same test beam pass as plain TIGER, so test was used once per checkpoint.
- **What it gives on test:**
  - cold Recall@50 rises from 0 to 0.00214 ± 0.00026, and cold Recall@10 to 0.00057;
  - warm Recall@50 falls by 3.8%, from 0.05617 to 0.05402;
  - overall Recall@50 falls by 0.8% and overall NDCG@10 by 2.5%.
- **Why it stays small: few cold targets share a full three-code prefix with the beam.**
  - In seeds 0, 1 and 2, 10,215, 9,819 and 9,920 of the 535,916 cold rows (about 1.9%) had a target whose
    (c1, c2, c3) matched one of the 50 beam items.
  - 8,716, 8,339 and 8,542 of those targets were among the 50 best candidates.
  - With ε = 0.1 only five candidates are placed, so at most about 1.6% of cold rows could be hit at K = 50.
  - R3 found that 40.1% of cold test rows share a three-code prefix with some train item. The beam's 50 items
    cover only a small part of those prefixes.
- **Records:** `runs/tiger/seedS.json`, under `test.cold_start` and `test.cold_start_stats`.

### Validation-only ε curve (analysis; nothing selected)
These are means over the three checkpoints on the protocol's 100,000-row validation sample. One beam pass per
checkpoint gives every ε. Records: `tiger_eps/seedS.json`.

| ε | Recall@10 | NDCG@10 | Recall@50 | Warm Recall@50 | Cold Recall@50 |
|---|---|---|---|---|---|
| 0 (plain TIGER) | 0.02257 | 0.01214 | 0.05227 | 0.06241 | 0 |
| 0.02 | 0.02257 | 0.01214 | 0.05204 | 0.06197 | 0.00090 |
| 0.05 | 0.02257 | 0.01214 | 0.05186 | 0.06160 | 0.00170 |
| 0.1 (the paper's) | 0.02133 | 0.01178 | 0.05094 | 0.06009 | 0.00377 |
| 0.2 | 0.02011 | 0.01127 | 0.04907 | 0.05729 | 0.00673 |
| 0.3 | 0.01858 | 0.01067 | 0.04695 | 0.05408 | 0.01025 |
| 0.5 | 0.01517 | 0.00871 | 0.04249 | 0.04694 | 0.01960 |
| 1 | 0.00162 | 0.00071 | 0.03288 | 0.03208 | 0.03701 |

- **The trade is close to one for one.** Each step up in ε moves Recall@50 from warm rows to cold rows, and
  overall Recall@50 falls at every step.
- **Below ε = 0.1 the top 10 is untouched,** because floor(10 ε) is 0. That is why Recall@10 and NDCG@10 are
  the same for ε = 0 to 0.05.
- **The validation sample's cold rows differ from test's.** Their targets include items that are new in
  validation, and the cold share is lower (16%). The ε = 0.1 cold Recall@50 is 0.00377 here, against 0.00214
  on test.

## HSTU-SID
- **The model:** each item's embedding is its ID row plus the embeddings of its semantic-ID prefixes (c1),
  (c1, c2) and (c1, c2, c3). Items outside train get a zero ID row.
  - The model ranks all 368,228 catalog items.
  - Its loss is the full softmax over the train items.
  - Prefix rows exist only for prefixes of train items. On the full split that is 256, 39,715 and 226,620
    rows.

### Tuning on the dev subset
The tuning used the 12 shared trials, seed 0, on all 71,610 eligible dev validation rows. Records:
`runs/hstu-sid/trialNN.json`. Checkpoints: `data/runs/r4b/hstu-sid/`.

| Trial | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Valid NDCG@10 | .0013 | .0012 | .0010 | .0058 | .0008 | **.0071** | .0050 | .0009 | .0039 | .0031 | .0012 | .0000 |
| Cold Recall@50 | .0029 | .0037 | .0025 | .0035 | .0030 | .0023 | .0080 | .0025 | .0072 | .0052 | .0037 | .0007 |
| Top-50 share outside train | 90% | 93% | 92% | 70% | 92% | 41% | 72% | 85% | 84% | 86% | 96% | 100% |

- **Selected:** trial 6 (learning rate 3e-3, no dropout or weight decay, width 64, 2 layers). Its validation
  NDCG@10 is 0.00709, against 0.00676 for the HSTU-style model's selected trial (R4a).
- **The dev subset exposes a weakness of this design.** Items outside the train split take most of the ranking.
  - On the dev subset those items are 49.8% of the catalog. In most trials they still filled 70% to 100% of the
    top 50.
  - Such an item's embedding is its prefix sum alone. That sum is the shared part of a cluster of train items,
    and after L2 normalization it often scores above each train item, which also carries its own ID row.
  - These items are never negatives in training, so nothing pushes them down.
  - The check is `scripts/r4b_hstu_sid_flood.py`, run on validation only (records in `hstu_sid_flood/`).
    Ranked among the train items only, trial 1 reaches warm NDCG@10 0.0073. Over the catalog it reaches 0.0018.
- **The design was kept as the protocol fixed it.** Selection on NDCG@10 over all rows favoured the trial that
  floods least.
- **On the full split it floods much less.** There, items outside train are 10.5% of the catalog. In the three
  final runs they take 7% to 8% of the top 10 and 17% to 21% of the top 50 (100,000-row validation sample).
  - Warm NDCG@10 is 0.0172 over the catalog and 0.0179 when only train items are ranked.
  - Scoring cold items therefore costs about 4% of warm NDCG@10, and it buys cold Recall@50 0.0055 on test.

## The HSTU-style model without its time bias
The run is seed 0 of the selected configuration, with the time part of the relative attention bias removed.
That removes every timestamp input, the target's time included, as R2's ablation did on Beauty.

| Model | Recall@10 | Recall@50 | NDCG@10 |
|---|---|---|---|
| HSTU-style, seed 0 | 0.01440 | 0.03513 | 0.00804 |
| HSTU-style, 3 seeds | 0.01431 ± 0.00007 | 0.03513 ± 0.00018 | 0.00796 ± 0.00009 |
| HSTU-style without time bias, seed 0 | 0.01363 | 0.03380 | 0.00745 |
| SASRec, 3 seeds | 0.01336 ± 0.00003 | 0.03323 ± 0.00019 | 0.00723 ± 0.00002 |

- **Removing the time bias costs 6% of NDCG@10** against the 3-seed mean (7% against seed 0). The gap of
  0.00051 is several of the HSTU-style model's standard deviations.
- **That accounts for about 70% of the HSTU-style model's lead over SASRec** on NDCG@10:
  (0.00796 - 0.00745) / (0.00796 - 0.00723).
- **Without the time bias it is still above SASRec,** by 0.00022 NDCG@10, but with one seed that is a single
  run.
- **Beauty showed the same pattern (R2),** where the time bias carried most of the lead.
- **The request time is legitimate context at serving time,** but SASRec does not get it. Both comparisons are
  reported.

## Training lengths
| Model | Epochs run (seeds 0, 1, 2) | Best epoch | Dev estimate (R4a) | GPU-hours |
|---|---|---|---|---|
| SASRec | 12, 11, 12 | 9, 8, 9 | 11 | 3.76 |
| HSTU-style | 12, 12, 11 | 9, 9, 8 | 16 | 7.49 |
| TIGER | 20, 17, 17 | 17, 14, 14 | 9 | 13.72 |
| HSTU-SID | 5, 5, 5 | 2, 2, 2 | 5 | 2.35 |
| HSTU-style without time bias | 12 | 9 | (16) | 1.93 |

- **TIGER trained about twice as long as its dev trial.** Its validation NDCG@10 was still rising slowly when
  early stopping ended it.
- **Long runs span several processes,** of at most about an hour each: one for each HSTU-SID run, two for
  SASRec, three for the HSTU-style model and five for TIGER. Each process saves the
  training state after every epoch, and the next one resumes from it (`sidrec.train.resume`). Each record lists
  its processes, with their commits, `nvidia-smi` state and sampled clocks.

## GPU time and the 60-hour stop
- **Recorded:** 30.54 GPU-hours on the RTX 4090. That is 29.26 for the 13 final runs, 1.18 for HSTU-SID's 12
  trials, 0.07 for the ε curve and 0.03 for the HSTU-SID checks.
- **Not recorded:** about 0.2 hours more. That covers smoke tests of HSTU-SID and the cold-start code from the
  uncommitted tree, and a first ad hoc HSTU-SID check before the script existed. None of them wrote a record.
- **Projection** (`projection.json`, built by `python -m sidrec.eval.r4b_summary --projection`):
  - Before the long runs it was 29.5 GPU-hours, from R4a's cost table and the dev trials' epoch counts (commit
    37cdeb8).
  - It was refreshed after each final run with the measured times. Its highest value was 35.6, when TIGER ran
    past its dev estimate.
  - It never reached the 60-hour stop line.

## Provenance and what went wrong
- **HSTU-SID trials 4 to 12 are marked dirty.** While they ran, an uncommitted edit to
  `src/sidrec/eval/r4b_summary.py` was in the tree. Training does not import that module. They were not rerun.
- **The final runs record two `src` tree hashes,** 9a5438a and e510523. The only change between them is
  1092a9b, which edits `src/sidrec/eval/r4b_summary.py`. No training code changed after 3ab2c4c, and no final run
  is marked dirty.
- **The first launch of the tuning script failed.** The script had been committed without its executable bit
  (fixed in e1aff22). Nothing ran.
- **The tuning was halted after HSTU-SID's first trial**, to look into its low validation NDCG@10. That
  investigation is the check described above. Trials 2 to 12 then ran unchanged.

## Artifacts outside git
Check them from the repository root with `sha256sum -c results/r4b/SHA256SUMS`.

| Path | Contents |
|---|---|
| `data/runs/r4b/<model>/seedS.pt` | best-epoch weights of the 13 final runs |
| `data/runs/r4b/hstu-sid/trialNN.pt` | best-epoch weights of HSTU-SID's 12 tuning trials |

There are 25 files, 5.5 GB in all. The resumable training states were deleted when each run finished.

## Files
| Path | Contents |
|---|---|
| `runs/<model>/seedS.json` | final runs: configuration, per-epoch history, validation and test metrics, processes |
| `runs/hstu-sid/trialNN.json` | HSTU-SID's 12 tuning trials (dev subset, validation only) |
| `tiger_eps/seedS.json` | the validation-only ε curve of each TIGER checkpoint |
| `hstu_sid_flood/` | how much of HSTU-SID's ranking items outside train take (validation only) |
| `projection.json` | the GPU-hour projection, as last refreshed |
| `summary.json` | everything above in one file |
| `SHA256SUMS` | checksums of the checkpoints |
