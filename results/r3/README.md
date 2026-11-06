# R3: Electronics item embeddings and semantic IDs

R3 embeds all 368,228 Electronics items with Qwen3-Embedding-0.6B. It runs two implementations and checks that
they agree. It then turns the vectors into semantic IDs with the R2 RQ-VAE.

The rules R3 follows are in [docs/protocol.md](../../docs/protocol.md), under "Changes, 2026-11-05: Electronics
item embeddings and semantic IDs". That entry was committed before any of these runs.

**Where it ran:** every GPU process ran on the RTX 4090 workstation, each under the GPU lock. No Snowflake
compute was used (0 credits). The plan put this job on two A10s; the 2026-11-06 budget amendment moved it to
the 4090, so there is no 1-GPU against 2-GPU comparison.

**Numbers below:** they come from [summary.json](summary.json), which `python -m sidrec.eval.r3_summary` builds
from the run records in this folder.

## Inputs
- **Documents:** R1's item documents (`REC_ITEM_DOCS`), one per item, in `parent_asin` order.
- **Tokens:** 94,219,784 with the Qwen3 tokenizer, which appends the `<|endoftext|>` token that last-token
  pooling reads. Nothing is truncated.
- **Tokens per item:** min 4, 10th percentile 71, median 253, mean 256, 90th percentile 429, 99th percentile
  704, max 1,651. The histogram is in [item_docs.json](item_docs.json).

## Throughput: vLLM against R2's loop

| Run | Items/s | Tokens/s | Encode time | Padding | SM clock (median) | Power (median) | GPU util (median) |
|---|---|---|---|---|---|---|---|
| vLLM 0.31.0, full catalog | 480 | 122,691 | 768 s | 0% | 2,685 MHz | 445 W | 92% |
| R2 loop, batch 64, full catalog | 113 | 28,987 | 3,250 s | 50% | 2,760 MHz | 399 W | 100% |
| vLLM 0.31.0, 20,000-item sample | 486 | 123,735 | 41 s | 0% | 2,655 MHz | 446 W | 95% |
| R2 loop, batch 64, sample | 114 | 28,977 | 176 s | 50% | 2,760 MHz | 398 W | 100% |
| R2 loop, batch 128, sample | 102 | 25,987 | 196 s | 55% | 2,760 MHz | 397 W | 100% |
| R2 loop, batch 256, sample | 94 | 23,812 | 214 s | 58% | 2,760 MHz | 401 W | 100% |

- **vLLM:** the pooling runner in bfloat16 with its default engine settings, apart from a GPU memory
  utilization of 0.83. Those defaults are FlashAttention, chunked prefill with 8,192 batched tokens, up to 256
  sequences, torch.compile and piecewise CUDA graphs. Engine start-up (36 s on the full run) is not in the
  encode time.
- **R2 loop:** `SentenceTransformer.encode`, as R2 ran it: bfloat16, SDPA attention and eager PyTorch.
  - Batch 64 was the fastest of 64, 128 and 256 on the sample, so the full run used it.
  - Model loading and token counting (20 s) are not in the encode time.
- **Result:** vLLM is 4.2 times faster on the full catalog, 12.8 minutes against 54.2.
- **Padding accounts for a factor of about 2 of that 4.2.**
  - sentence-transformers sorts texts by character count, which tracks token count loosely on these
    documents. Its batches computed on 188.8 million padded tokens for 94.2 million real ones.
  - Sorting by token count would have cut that to 94.3 million.
  - The loop processed 58,000 padded tokens per second, so without padding vLLM would still be about 2.1 times
    faster. That is an estimate from these counts, not a measured run.
  - Larger batches pad more and run slower.
- **GPU state:** clocks cannot be locked on this machine, so each record samples `nvidia-smi` every 10 s.
  - The GPU was in P2 while encoding. The loop's records also catch P3 and P8 while the model loads.
  - vLLM ran near the 450 W power limit, with a median of 445 W and a median SM clock of 2,685 MHz.
  - The loop drew about 400 W at 2,760 MHz.
  - Both full runs reached 84 °C.

## Agreement of the two implementations
Measured on all 368,228 items (`embeddings/agreement.json`):
- **Cosine of matching vectors:** min 0.99910, 0.01st percentile 0.99951, median 0.99984. No item is below
  0.999.
- **Largest absolute difference of any component:** 0.0063. The per-item maximum has a median of 0.0022. The
  vectors are unit length with 1,024 components.
- **Nearest neighbours:** for 2,000 sampled items, 96.0% of their 10 nearest neighbours in the full catalog are
  the same under both.
- **Tokens:** both implementations saw the same 94,219,784.
- **Decision:** no item falls below the protocol's 0.99 threshold, so the semantic IDs use the vLLM vectors.

## Semantic IDs
The R2 recipe: 3 levels of 256 codes plus a fourth code that separates collisions. It was trained on the
329,445 train-split items for 746 epochs, which is 240,212 steps and took 440 s. Codes were then assigned to
every item.

| | All items | Train items |
|---|---|---|
| Items | 368,228 | 329,445 |
| Reconstruction (explained variance) | 51.7% | 51.5% |
| Codes used, levels 1 / 2 / 3 | 256 / 256 / 256 | 256 / 256 / 256 |
| Code perplexity, levels 1 / 2 / 3 | 212 / 236 / 245 | 213 / 238 / 246 |
| Distinct three-code IDs | 247,708 | 226,620 |
| Collision rate before the fourth code | 32.7% | 31.2% |
| Items whose three-code ID is shared | 47.0% | 45.3% |
| Largest group with one three-code ID | 461 | 461 |

- **Against Beauty (R2):** reconstruction falls from 56% to 52%, and the collision rate rises from 9.3% to
  32.7%. Both use the same 16.7 million possible three-code IDs, but this catalog is 30 times larger.
- **Fourth code:** it takes 461 values, not 256, so R4's token vocabulary has to allow for that.
  - The largest group is 461 extended-warranty listings: 460 of the titles contain "Protection Plan", and
    422 are Allstate plans.
  - Their titles differ mainly in the product type and the price band they cover.
  - This is reported as found, not tuned away (protocol).

### Cold-start items: is their prefix known from training?
This is the share of items whose first codes occur among the train items' IDs.

| Set | Count | Level 1 | Levels 1–2 | Levels 1–3 |
|---|---|---|---|---|
| Cold-start test items (distinct targets) | 36,881 | 100% | 96.5% | 39.4% |
| Cold-start test rows | 535,916 | 100% | 96.7% | 40.1% |
| All items outside train | 38,783 | 100% | 96.5% | 39.4% |

- **Levels 1–2:** 96.5% of cold-start test items share a two-code prefix with some train item. A generative
  retriever trained on train items has seen those first two tokens together.
- **Levels 1–3:** 39.4% share all three codes with a train item. Only the fourth code tells them apart.

### Sanity checks
- **Neighbours against codes:** for 1,000 sampled items, compare each with its 10 nearest embedding
  neighbours.
  - 86.0% share its first code, against 0.56% for two random items.
  - 32.3% share its first two codes, against 0.018%.
  - 7.7% share all three, against 0.001%.
- **Examples** (titles shortened; all are in `semantic_ids.json`):
  - **A charger for the 18.4-inch Samsung Galaxy View tablet** (train, ID 252-41-187-0). Its 5 nearest
    neighbours are chargers for the same tablet, with cosine 0.86 to 0.91. All have first code 252, and two
    also share 41.
  - **A portable amplified HD TV antenna** (cold, 151-93-253-1). Its 5 nearest neighbours are amplified HDTV
    antennas, all with first code 151.
  - **A cold 8-pin GPU power adapter angled at 180 degrees** (180-83-253-2). Its two nearest neighbours are
    train items of the same kind with the same three codes. They are numbered 0 and 1 first, so it gets 2.
  - **Same two-code prefixes:**
    - 207-139 (101 items) holds camera batteries, such as Nikon EN-EL3, EN-EL14, EN-EL15B and EN-EL18, and
      Pentax D-Li109.
    - 164-213 (26 items) holds lens hoods.
    - 45-84 (22 items) holds headphones.

## GPU time
- **Recorded runs:** 1.44 GPU-hours on the RTX 4090. The embedding runs, including the sample sweeps, took
  1.31; the RQ-VAE took 0.12.
- **Not recorded:** about 0.2 hours more.
  - A first vLLM smoke test and batch sweep ran from the uncommitted tree. The runs above repeat them from the
    commit.
  - Two short profiles of the loop found its padding.
- **Total:** about 1.65 GPU-hours. Snowflake: 0 credits.

## Departures from the plan
- **Hardware:** the 4090 instead of two A10s, by the budget amendment, so there is no 1-GPU against 2-GPU
  row.
- **Encoder size:** fixed at 0.6B by R2's encoder comparison.
- **Protocol choices:** the RQ-VAE keeps Beauty's step count rather than its epoch count, and the fourth code
  numbers train items first. Both are in the protocol entry.

## Artifacts outside git
The arrays are large, so they stay out of the repository. Check them from the repository root with
`sha256sum -c results/r3/SHA256SUMS`.

| Path | Contents |
|---|---|
| `data/processed/electronics/item_docs.json` | item IDs and documents in item order |
| `data/processed/electronics/emb/vllm.npy` | vLLM vectors, float32 [368,228, 1,024], unit length; the IDs use these |
| `data/processed/electronics/emb/hf.npy` | the R2 loop's vectors, same layout |
| `data/processed/electronics/sid/qwen3-emb-0.6b.npy` | semantic IDs, int64 [368,228, 4], item order |
| `data/runs/r3/rqvae.pt` | the trained RQ-VAE's state dict |

- **Software, vLLM environment:** `.venv-vllm`, built offline from `wheelhouse-vllm-0.31.0.tar`. It has vLLM
  0.31.0, torch 2.13.0+cu130 and transformers 5.17.0.
- **Software, loop environment:** torch 2.15.0+cu132, transformers 5.22.1 and sentence-transformers 6.2.0.

## Files
| Path | Contents |
|---|---|
| `item_docs.json` | document and token-length statistics, train and cold-start counts |
| `embeddings/vllm*.json`, `embeddings/hf-*.json` | one record per embedding run, with its `nvidia-smi` samples |
| `embeddings/agreement.json` | the agreement check and which vectors the IDs use |
| `semantic_ids.json` | RQ-VAE settings, loss log, code statistics, prefix coverage, sanity examples |
| `summary.json` | the tables above and the recorded GPU hours |
| `SHA256SUMS` | checksums of the artifacts outside git |
