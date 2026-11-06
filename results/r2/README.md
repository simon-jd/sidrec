# R2: Beauty reproduction anchor

R2 trains three models on Amazon 2014 Beauty and compares them with published numbers:
- SASRec with full-softmax cross-entropy;
- an HSTU-style model;
- TIGER, with semantic IDs from three different text encoders.

All runs follow [docs/protocol.md](../../docs/protocol.md), including its dated "Changes" section.

**Where it ran:** every run is on the RTX 4090 workstation, under the GPU lock. No Snowflake compute was
used (0 credits).

**What each run records:** each JSON under `runs/` holds the configuration, seed, validation and test metrics,
per-epoch history, wall time, the commit it started from, and the `nvidia-smi` state at its start and end.

**Numbers below:** test metrics on all 22,363 users, as mean ± sample standard deviation over seeds 0, 1 and 2.
They come from [summary.json](summary.json), which `python -m sidrec.eval.r2_summary` builds from the run files.

## Ours against the published numbers

The gap is (ours - published) / published. The 15% rule applies to the primary reference only.

| Model | Recall@10 | NDCG@10 | Reference (primary first) | Its R@10 / N@10 | Gap R@10 | Gap N@10 |
|---|---|---|---|---|---|---|
| SASRec, CE loss | 0.0897 ± 0.0002 | 0.0556 ± 0.0002 | TIGER Table 1, SASRec | 0.0605 / 0.0318 | +48.3% | +74.7% |
| | | | eSASRec Table 4, SASRec+SS | 0.092 / 0.053 | -2.5% | +4.8% |
| HSTU-style, CE loss | 0.1023 ± 0.0019 | 0.0618 ± 0.0015 | eSASRec Table 4, HSTU | 0.079 / 0.044 | +29.5% | +40.4% |
| | | | ActionPiece Table 4, HSTU | 0.0704 / 0.0389 | +45.3% | +58.8% |
| TIGER, Sentence-T5-XXL IDs | 0.0551 ± 0.0014 | 0.0294 ± 0.0006 | TIGER Table 1 | 0.0648 / 0.0384 | -15.0% | **-23.5%** |
| | | | TIGER Table 9 (3-seed mean) | 0.0642 / 0.0374 | -14.2% | **-21.5%** |
| | | | Mender, TIGER reproduction | 0.0681 / 0.0356 | -19.1% | -17.5% |
| TIGER, Qwen3-Embedding-0.6B IDs | 0.0604 ± 0.0011 | 0.0318 ± 0.0002 | TIGER Table 1 | 0.0648 / 0.0384 | -6.8% | **-17.3%** |
| TIGER, Qwen3-Embedding-4B IDs | 0.0597 ± 0.0013 | 0.0316 ± 0.0012 | TIGER Table 1 | 0.0648 / 0.0384 | -7.9% | **-17.8%** |

Bold marks a gap past the 15% rule. The Recall@5 and NDCG@5 values, and the Table 9 and Mender gaps of the
Qwen rows, are in `summary.json`.

### Which configuration each model used
Each model got the protocol's 12 trials, with seed 0, selected on validation NDCG@10.

| Model | Selected trial | Configuration | Best epochs (seeds 0, 1, 2) |
|---|---|---|---|
| SASRec | 11 | lr 3e-4, dropout 0.1, weight decay 0, width 256, 2 layers | 18, 17, 17 |
| HSTU-style | 1, the published config | lr 1e-3, dropout 0.5, weight decay 0, width 64, 4 layers | 84, 59, 75 |
| TIGER | 11 | lr 3e-4, dropout 0.1, weight decay 0, width 256, 2 + 2 layers | 51, 56, 57 |

Validation NDCG@10 of all 12 trials, with the best epoch after the @:

| Trial | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SASRec | .0704@38 | .0668@14 | .0719@77 | .0700@26 | .0655@22 | .0669@9 | .0691@12 | .0647@19 | .0682@5 | .0705@9 | **.0720@18** | .0575@90 |
| HSTU-style | **.0804@84** | .0357@17 | .0613@61 | .0700@47 | .0314@16 | .0408@10 | .0392@12 | .0251@25 | .0532@9 | .0495@9 | .0593@20 | .0120@32 |
| TIGER (ST5) | .0165@8 | .0341@12 | .0362@220 | .0266@70 | .0314@20 | .0382@19 | .0344@19 | .0334@19 | .0215@4 | .0362@19 | **.0425@51** | .0286@38 |

## TIGER: encoder comparison for R3

The setup is identical for all three rows:
- the same RQ-VAE settings (seed 0), with one tokenizer trained per encoder;
- the same TIGER configuration (trial 11, chosen on the Sentence-T5 IDs);
- the same seeds.

| Encoder | Dim | Params | Encode (items/s, bf16) | RQ-VAE explained variance | First-level codes used | Collision rate | Largest group | TIGER R@10 | TIGER N@10 |
|---|---|---|---|---|---|---|---|---|---|
| sentence-t5-xxl | 768 | 4.87B | 197 | 57.1% | 38 / 256 | 9.2% | 8 | 0.0551 ± 0.0014 | 0.0294 ± 0.0006 |
| Qwen3-Embedding-0.6B | 1,024 | 0.60B | 1,404 | 55.7% | 255 / 256 | 9.3% | 18 | 0.0604 ± 0.0011 | 0.0318 ± 0.0002 |
| Qwen3-Embedding-4B | 2,560 | 4.02B | 257 | 54.7% | 135 / 256 | 12.9% | 29 | 0.0597 ± 0.0013 | 0.0316 ± 0.0012 |

- **The Qwen3 encoders beat Sentence-T5-XXL.** The gap is about 10% relative on Recall@10 and 8% on NDCG@10,
  several standard deviations.
- **Qwen3-Embedding-0.6B and -4B tie.** Their difference is smaller than their standard deviations, so the
  protocol does not call either one better.
- **Speed:** the 0.6B model encodes 5.5 times faster than the 4B, at 1,404 against 257 items per second on the
  4090. Recall@10 is then the same within noise.
- **Code use differs most at the first level:** 38, 255 and 135 of 256 codes. The second and third levels use
  247 to 256 codes for every encoder.
- **Collision rate** is 1 - (distinct three-code IDs / items). The fourth code makes every ID unique; its range is
  the size of the largest group.
- **No invalid IDs:** constrained beam search never emitted an ID outside the catalog. Every run records
  `empty_top10_slots: 0`, and `tests/test_tiger.py` checks this with a model pushed towards invalid codes.

Files: `embeddings/<encoder>.json` (throughput, peak memory, norms, example text) and
`semantic_ids/<encoder>.json` (per-level use and perplexity, collisions, reconstruction, loss log).

## The 15% rule: TIGER is past it on NDCG@10

With the protocol's selection, TIGER on Sentence-T5 IDs lands 15.0% below Table 1 on Recall@10. On NDCG@10 it
lands 23.5% below. With Qwen3 IDs the Recall@10 gap shrinks to 7-8%, but NDCG@10 stays 17-18% below. We checked
each suspect below. The best variants still sit 17-19% below Table 1 on NDCG@10: Qwen3 IDs with AdamW, and
Sentence-T5 IDs with Adafactor. That remainder is unexplained.

- **Preprocessing:** not the cause.
  - The sequences match the S3-Rec file line for line (R1), and the counts match TIGER's Table 6.
  - SASRec and the HSTU-style model use the same data and metric code, and land above their references.
- **The metric:** not the cause.
  - Ranks come from the top 10 of a width-20 beam, with no history masking, as in the paper.
  - The rank and NDCG code has hand-computed tests. Beam search matches exhaustive scoring when the beam covers
    every ID.
- **The optimizer:** part of the cause.
  - The protocol fixes AdamW. At the paper's learning rate of 0.01, AdamW diverged (trial 1: validation
    NDCG@10 0.0165).
  - TIGER trained with T5X's Adafactor. We ran the paper's exact setting with Adafactor as a diagnostic outside
    the protocol (`runs/tiger_st5-xxl_adafactor/`, 3 seeds). Its configuration was fixed in advance and nothing
    was selected from it.
  - It reaches validation NDCG@10 0.0456 against 0.0431 for the selected AdamW trial. On test it gets Recall@10
    0.0578 ± 0.0026 and NDCG@10 0.0310 ± 0.0013, which is -10.8% and -19.2% against Table 1.
- **Training length:** not the cause.
  - One Adafactor run had no early stopping and ran the full 300 epochs, about 154K steps against the paper's
    200K (`runs/tiger_st5-xxl_adafactor_patience300/`).
  - Validation NDCG@10 peaks at epoch 48 (0.0468), then falls steadily to 0.031 at epoch 300. Longer training
    overfits; patience 10 is not cutting runs short.
- **The loss:** not the cause, as far as we can tell. It is the standard token-level cross-entropy over the four
  semantic-ID tokens.
- **The tokenizer:** likely part of the cause.
  - The paper's RQ-VAE recipe (Adagrad at 0.4) did not train here (protocol, Changes), so ours uses Adam.
  - Its Sentence-T5 first level uses 38 of 256 codes; the paper reports at least 80% use.
  - IDs with fuller first levels (the Qwen3 rows) are clearly better.
- **Unexplained remainder:** the paper says its model has about 13M parameters. The configuration it states (4 + 4
  layers, width 128, 6 heads of 64, MLP 1024) comes to 4.8M in our T5, and trial 11 has 5.2M. We do not know what
  accounts for the difference.

## HSTU-style: where its lead comes from

The HSTU-style model beats SASRec by 14% on Recall@10. Most of that lead comes from the time part of its relative
attention bias, which uses the request time (the target's review date). It does not come from the attention
form itself.

The ablation reran the selected HSTU configuration with the same seeds and no time bias (`runs/hstu-notime/`).

| Model | Recall@10 | NDCG@10 |
|---|---|---|
| HSTU-style | 0.1023 ± 0.0019 | 0.0618 ± 0.0015 |
| HSTU-style without the time bias | 0.0920 ± 0.0019 | 0.0561 ± 0.0008 |
| SASRec | 0.0897 ± 0.0002 | 0.0556 ± 0.0002 |

Without the time bias, it is within about one standard deviation of SASRec on Recall@10 and level on NDCG@10.
The request time is legitimate context at serving time, but SASRec does not get it. The comparison is reported
both ways.

## What differs from the plan
- **TIGER encoders:** the plan named Qwen3-Embedding-0.6B. The brief asked for Sentence-T5 first, then both
  Qwen3 sizes; all three ran.
  - The paper does not name a Sentence-T5 size. We used the largest, sentence-t5-xxl; it fits on the 4090.
  - It ran in bfloat16 rather than float16, because T5 overflows in float16. Its peak memory was 13.2 GB.
- **RQ-VAE optimizer:** Adam at 1e-3 instead of the paper's Adagrad at 0.4. Every Adagrad variant diverged or
  learned only the mean embedding.
  - The first fix we committed (Adagrad with an averaged loss) produced IDs from an untrained encoder.
  - Two TIGER tuning trials ran on those IDs before we caught it. They are kept in
    `superseded/tiger_st5-xxl_adagrad-ids/`, with their tokenizers in `superseded/semantic_ids_adagrad/`.
  - All 12 TIGER trials were then rerun on the new IDs.
- **TIGER's T5:** our own compact T5 (`src/sidrec/models/t5.py`). A test checks that its logits match Hugging
  Face's T5 on the same weights. It trains 25% faster.
- **HSTU-style model:** our own implementation, written from the paper's equations. No code comes from Meta's
  generative-recommenders, so there is no licence header to carry. Trial 1 takes the published configuration
  from that repository's Amazon Books config.
- **Diagnostics added beyond the plan:** the HSTU time-bias ablation, the Adafactor TIGER runs and the
  no-early-stopping run.
- **Seed 0 of each final configuration** is the selected tuning run, evaluated once on test. Seeds 1 and 2 were
  trained fresh. Tuning runs computed no test metrics.

## Provenance notes
- **HSTU-style seed 1:** rerun. Its first record had been written while an uncommitted, unrelated edit was in
  the tree, so it was replaced by a clean run.
- **`embeddings/qwen3-emb-0.6b.json`:** it shows `dirty: true`. The flag is read at the end of the run, and an
  uncommitted edit to the HSTU files was in the tree then. The embedding code was as committed in 8b3f9d1. The
  vectors were not regenerated, because the semantic IDs are built from them.

## Compute
- **Recorded GPU jobs:** 6.1 hours of wall time on the RTX 4090. The sum covers every JSON here, including the
  diagnostics and the superseded runs; `summary.json` gives the total.
- **Not recorded:** smoke tests, benchmarks and the CPU and GPU checks of RQ-VAE variants, about another half
  hour.
- **Snowflake:** 0 credits.

## Files
| Path | Contents |
|---|---|
| `runs/<model>/trialNN.json` | tuning runs (seed 0, validation only) |
| `runs/<model>/trialNN_seedS.json` | final runs with test metrics |
| `runs/hstu-notime/` | HSTU-style ablation without the time bias |
| `runs/tiger_st5-xxl_adafactor*/` | TIGER diagnostics with the paper's optimizer (outside the protocol) |
| `embeddings/`, `semantic_ids/` | encoder runs and RQ-VAE tokenizers |
| `superseded/` | runs on the first, broken RQ-VAE IDs, kept and labelled |
| `summary.json` | means, standard deviations, gaps, the 15% check and GPU hours |
