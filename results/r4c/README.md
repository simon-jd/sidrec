# R4c: the two-stage cascade

R4c puts a retrieval stage in front of the R4b rankers:
- TIGER's beam and an HNSW search over the item vectors retrieve 200 candidates per row;
- the HSTU-style model or HSTU-SID re-ranks them.

Nothing is trained. Every checkpoint is an R4b final run, checked against `results/r4b/SHA256SUMS` before use.

All steps follow [docs/protocol.md](../../docs/protocol.md), entry "2026-11-07: R4c two-stage cascade":
- the entry was committed in afb011d, before any R4c measurement;
- b230f27 corrected its count of ANN queries from 22 to 25, also before any measurement;
- the code was committed in 3281e2f, before any run.

**Where it ran:** the GPU steps ran on the RTX 4090 workstation, under the GPU lock. The index and the ANN
searches ran on the CPU. No Snowflake compute was used.

**Numbers below:** they come from [summary.json](summary.json), which `python -m sidrec.eval.r4c_summary`
builds from the records in this folder. Test numbers cover all 1,199,539 eligible test rows and are given as
mean ± sample standard deviation over seeds 0, 1 and 2. TIGER seed s goes with ranker seed s.

## The cascade, as chosen on validation
Every choice used the protocol's 100,000-row validation sample. 16,256 of its rows are cold.

| Part | Setting |
|---|---|
| TIGER source | width-200 beam, constrained to the catalog, history removed; the plain beam, with no cold-start rule |
| ANN source | FAISS HNSW, inner product, M 32, efConstruction 200, efSearch 512; the query is the most recent item's vector |
| Union | TIGER's first 175 items + ANN's first 25, duplicates removed, filled from TIGER's list up to 200 |
| Rankers | the R4b HSTU-style model and HSTU-SID, each scoring the 200 candidates |

### ANN query (25 settings)
- **The most recent item alone (n = 1) won,** at recall@200 0.0362. Every longer window scored lower, and more so
  as the weights flattened. The best longer query was n = 2 with γ = 0.4, at 0.0355. The plain mean of the
  last 50 items reached 0.0155.
- **HNSW against exact search:** the HNSW lists share 99.91% of their 200 items with exact inner-product search
  over all 368,228 items. Target recall@200 is 0.03624 for HNSW and 0.03628 for exact search. The fallback to a
  larger efSearch was not needed.
- **On test the ANN source is the nearest-neighbour list of the last item.**

### Union (28 settings)
The criterion is union recall@200, the mean over the three TIGER checkpoints. The table shows TIGER's share
with backfill from TIGER's list. ANN backfill is lower by 0.00003 to 0.0015 at every share.

| TIGER items of 200 | 25 | 50 | 75 | 100 | 125 | 150 | 175 |
|---|---|---|---|---|---|---|---|
| Rule off | .0673 | .0809 | .0901 | .0974 | .1033 | .1073 | **.1095** |
| Rule on, ε = 0.1 | .0660 | .0785 | .0878 | .0948 | .1006 | .1044 | .1068 |

- **Selected: 175 TIGER items, backfill from TIGER, no cold-start rule.**
- **The choice is at the edge of the grid.** Recall rises with every step in TIGER's share, so the best share
  may lie between 175 and 200. The protocol fixed the grid, so 175 stands.
- **Backfill barely matters at 175.** The two backfills differ by 0.00003, less than the spread over seeds.
- **Section 2's original setting (50 TIGER items, ANN backfill, no rule) reaches 0.0794.**
- **TIGER's cold-start rule lowers union recall@200 at every share.** It gives up to 10% of TIGER's slots to
  unseen items with a matching prefix. Those slots find fewer targets than the beam items they replace, and the ANN
  part already brings in cold items.
- **TIGER alone at width 200** has validation recall@200 0.1043 (mean of three). The union adds about 0.005.

## Test: retrieval recall
**TIGER alone** is the selected source, the plain width-200 beam. Its top 50 is not R4b's width-50 beam: at
width 200, Recall@50 is 0.03322 against R4b's 0.03108.

**ANN alone** does not depend on a seed, so its standard deviation is 0. **Bold** marks the best source in a
row group, by more than both standard deviations.

| Rows | Source | @50 | @100 | @200 |
|---|---|---|---|---|
| All | TIGER alone | 0.03322 ± 0.00046 | 0.04789 ± 0.00068 | 0.06501 ± 0.00087 |
| All | ANN alone | 0.02039 | 0.02754 | 0.03691 |
| All | **Union** | **0.03737 ± 0.00038** | **0.05358 ± 0.00062** | **0.07371 ± 0.00082** |
| Warm | TIGER alone | 0.06005 ± 0.00084 | 0.08656 ± 0.00122 | 0.11751 ± 0.00157 |
| Warm | ANN alone | 0.01945 | 0.02638 | 0.03548 |
| Warm | **Union** | **0.06098 ± 0.00068** | **0.08782 ± 0.00112** | **0.12058 ± 0.00148** |
| Cold | TIGER alone | 0.0000012 ± 0.0000022 | 0.0000019 ± 0.0000032 | 0.0000093 ± 0.0000067 |
| Cold | **ANN alone** | **0.02156** | **0.02898** | **0.03868** |
| Cold | Union | 0.00812 | 0.01117 | 0.01567 |

The union at budgets 50 and 100 takes TIGER's first 44 and 88 items, by the protocol's rounding.

- **The union beats each source alone overall,** by 13% at @200 over TIGER alone (0.0737 against 0.0650).
- **The gain is mostly cold rows.**
  - Plain TIGER almost never retrieves a cold target: 0.0000093 at @200, about 5 of 535,916 cold rows per seed.
  - The union's 25 ANN slots reach 0.0157.
  - On warm rows the union adds 2.6% at @200.
- **ANN is the best cold retriever but the union keeps only its first 25 items,** so the union's cold recall is
  40% of ANN's at @200. Validation chose that trade.
- **Retrieval is the bottleneck.** 88% of warm targets and 98% of cold targets are not among the 200
  candidates, and no ranker can recover them.
- **The 200 candidates per row:** 176.9 come from TIGER's list. 7.3 lie outside the train split, and 7.32 of
  those come from ANN's first 25 items. No row had fewer than 200.

## Test: re-ranked, against R4b's full-catalog ranking
**Change** is the cascade minus R4b's full-catalog number, taken per seed with the same ranker checkpoint and
then averaged. The percentage is relative to R4b's mean.

| Rows | Ranker | Candidates | Recall@10 | Recall@50 | NDCG@10 |
|---|---|---|---|---|---|
| All | HSTU-style | full catalog (R4b) | 0.01431 ± 0.00007 | 0.03513 ± 0.00018 | 0.00796 ± 0.00009 |
| All | HSTU-style | cascade | 0.01489 ± 0.00012 | 0.03737 ± 0.00039 | 0.00819 ± 0.00012 |
| All | HSTU-style | change | +0.00058 (+4.1%) | +0.00224 (+6.4%) | +0.00024 (+3.0%) |
| All | HSTU-SID | full catalog (R4b) | 0.01512 ± 0.00020 | 0.03704 ± 0.00014 | 0.00827 ± 0.00010 |
| All | HSTU-SID | cascade | **0.01609 ± 0.00005** | **0.04084 ± 0.00034** | **0.00866 ± 0.00006** |
| All | HSTU-SID | change | +0.00097 (+6.4%) | +0.00380 (+10.3%) | +0.00039 (+4.8%) |
| Warm | HSTU-style | full catalog (R4b) | 0.02587 ± 0.00013 | 0.06350 ± 0.00032 | 0.01438 ± 0.00016 |
| Warm | HSTU-style | cascade | 0.02692 ± 0.00022 | 0.06755 ± 0.00071 | 0.01481 ± 0.00021 |
| Warm | HSTU-SID | full catalog (R4b) | 0.02670 ± 0.00034 | 0.06251 ± 0.00036 | 0.01470 ± 0.00019 |
| Warm | HSTU-SID | cascade | **0.02861 ± 0.00003** | **0.07093 ± 0.00058** | **0.01547 ± 0.00010** |
| Cold | HSTU-style | both | 0, by construction | 0, by construction | 0, by construction |
| Cold | HSTU-SID | full catalog (R4b) | **0.00078 ± 0.00009** | **0.00550 ± 0.00025** | **0.00030 ± 0.00003** |
| Cold | HSTU-SID | cascade | 0.00059 ± 0.00009 | 0.00359 ± 0.00034 | 0.00023 ± 0.00004 |
| Cold | HSTU-SID | change | -0.00019 (-24%) | -0.00191 (-35%) | -0.00007 (-23%) |

Standard deviations of the changes are in `summary.json`. On every all-row and warm-row metric, both rankers'
gains exceed both models' standard deviations. **Bold** marks the best value in a column and row group, by more
than both standard deviations.

### What the cascade costs and gains
- **Both rankers gain overall,** even though the candidates miss most targets.
  - A warm target inside the 200 ranks at least as high as in the full catalog, because only candidates
    compete with it.
  - The rows lost, whose target the ranker had in its full-catalog top 50 but the union missed, are fewer than
    the rows gained.
  - TIGER's beam acts as a filter on the ranker's false positives. On validation, no row ranked worse among
    the candidates than in the full catalog, which is the check that the scores are the same.
- **HSTU-style:** NDCG@10 +3.0% and Recall@50 +6.4%. It cannot score items outside the train split.
  - Its cold metrics stay 0 by construction.
  - The 7.3 cold candidates per row are slots it cannot use, so for this ranker 7 of the ANN part's 25 slots
    hold candidates it ranks last.
- **HSTU-SID:** NDCG@10 +4.8% and Recall@50 +10.3% overall, and +13.5% Recall@50 on warm rows.
  - It gains more than the HSTU-style model. In R4b's validation check, items outside train took 17% to 21%
    of its full-catalog top 50. The union admits only about 7 of them per row, so the flood that cost it warm accuracy is
    mostly gone.
- **HSTU-SID loses cold accuracy:** cold Recall@50 falls from 0.00550 to 0.00359 (-35%).
  - It can only rank the cold targets the union kept, 1.6% of cold rows at @200.
  - In the full catalog it could reach any of the 38,783 cold items.
  - The cascade trades cold rows for warm rows. Cold rows are 44.7% of test, and the net is still positive.
- **Against R4b's best full-catalog model:** HSTU-SID with the cascade leads every model R4b reported on all
  rows. The HSTU-style model with the cascade (NDCG@10 0.00819) is still below HSTU-SID in the full catalog
  (0.00827), by less than one standard deviation.

### Validation (analysis; nothing selected from it)
These are means over seeds on the 100,000-row sample, with the selected union.

| Ranker | Full-catalog NDCG@10 | Cascade NDCG@10 | Cascade cold Recall@50 | Full cold Recall@50 |
|---|---|---|---|---|
| HSTU-style | 0.01499 | 0.01510 | 0 | 0 |
| HSTU-SID | 0.01446 | 0.01494 | 0.00404 | 0.00646 |

- **Same direction as on test:** small gains for the HSTU-style model, larger gains and lower cold recall for
  HSTU-SID.
- **Checkpoint check:** each ranker's full-catalog validation NDCG@10, recomputed here, equals its R4b record
  to within 2e-18.

## GPU time and the 8-hour stop
- **Projection,** from the timed validation passes (`projection.json`): 3.28 GPU-hours in all, 3.02 of them
  for the test passes.
- **Recorded:** 3.12 GPU-hours on the RTX 4090.
  - TIGER's validation beams: 0.25.
  - Rankers on validation: 0.01.
  - Test passes: 2.86, about 57 minutes per seed, almost all of it TIGER's width-200 beam.
- **Not recorded:** about 0.05 GPU-hours more. Two smoke tests of the width-200 beam, run before any record,
  sized its chunk. They wrote nothing.
- **CPU only:**
  - index build: 64 s;
  - validation query grid and exact search: 28 min;
  - test ANN search: 8 min.
  All ran on 24 threads.

## Provenance and odd results
- **Code state:** every record has a clean `src`.
  - The validation steps ran at 3281e2f and ed6f1e9, which have the same `src` tree.
  - The test passes ran at a3d873b, which adds only the summary module to that tree.
  - The test ANN search started at f17cb54 and recorded a3d873b, the state at its end.
- **The protocol entry's ANN query count was wrong (22),** and was corrected to 25 before anything ran. The
  grid itself never changed.
- **Two choices sit at or near the edge of their grids:** n = 1 for the ANN query, and TIGER's largest share in
  the union. Both are reported as chosen.
- **The HNSW index was built with 24 threads,** so a rebuild is not bit-identical. Every result uses the saved
  index, whose sha256 is in `ann/build.json` and `SHA256SUMS`.
- **After the test passes, the workstation went offline and rebooted.** Nothing was running then, and every record
  had already been written.

## Artifacts outside git
Check them from the repository root with `sha256sum -c results/r4c/SHA256SUMS`.

| Path | Contents |
|---|---|
| `data/runs/r4c/hnsw.faiss` | the HNSW index over all 368,228 item vectors |
| `data/runs/r4c/ann_valid.npz`, `ann_test.npz` | the ANN lists of the validation sample and of every test row |
| `data/runs/r4c/tiger_valid_seedS.npz` | TIGER's width-200 validation lists, plain and with the rule |

## Files
| Path | Contents |
|---|---|
| `ann/build.json` | the index: parameters, build time, checksum |
| `ann/valid.json` | the 25-query grid, the selection and the recall against exact search |
| `ann/test.json` | the test ANN search: time and list checksum, no metrics |
| `valid/tiger_seedS.json` | TIGER's validation beam and the 28-point union grid |
| `selection.json` | the union chosen on validation |
| `valid/rerank_seedS.json` | the rankers on the validation union, and the checkpoint check |
| `projection.json` | the GPU-hour projection made before the test passes |
| `test/seedS.json` | the test pass of each seed: retrieval recall, union statistics, re-ranked metrics |
| `summary.json` | everything above, with the raw per-seed records |
| `SHA256SUMS` | checksums of the artifacts outside git |
