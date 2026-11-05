# Data

`scripts/download_data.sh` fetches the raw files listed below into `data/raw/`, which git ignores, and checks
them against [SHA256SUMS](SHA256SUMS). They total about 1.8 GB. Nothing under `data/` is committed or
redistributed.

The evaluation protocol that uses these files is in [protocol.md](protocol.md).

## Sources

| File | Source | Size (bytes) |
|---|---|---|
| `beauty2014/reviews_Beauty_5.json.gz` | `https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz` | 44,819,276 |
| `beauty2014/meta_Beauty.json.gz` | `https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz` | 99,156,148 |
| `beauty2014/s3rec_Beauty.txt` | `https://raw.githubusercontent.com/RUCAIBox/CIKM2020-S3Rec/e6c774da92dba847b00ac89e8bfc5c55cc44d378/data/Beauty.txt` | 1,090,094 |
| `amazon2023/Electronics/Electronics.{train,valid,test}.csv.gz` | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/timestamp/Electronics.{split}.csv.gz` | 219,579,387 / 27,677,512 / 26,805,493 |
| `amazon2023/Electronics/w_his.Electronics.{valid,test}.csv.gz` | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/timestamp_w_his/Electronics.{split}.csv.gz` | 60,518,111 / 59,170,875 |
| `amazon2023/Electronics/meta_Electronics.jsonl.gz` | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz` | 1,312,900,427 |

All files were downloaded on 2026-11-05. The sha256 of each is in [SHA256SUMS](SHA256SUMS).

The two `w_his` files and the S3-Rec `Beauty.txt` are not model inputs. They are used only to check our
preprocessing against the official histories and the standard Beauty sequences.

## Licence terms
None of the three sources forbids research or portfolio use. This repository publishes code, derived metrics and
small summaries. It does not publish review text, metadata or interaction files.

- **Amazon review data, 2014 version (McAuley lab, hosted on SNAP).**
  - The dataset page (jmcauley.ucsd.edu/data/amazon/) gives no licence or usage restriction.
  - It asks users to cite He & McAuley (WWW 2016) and McAuley et al. (SIGIR 2015) if they use the data "in any
    way".
- **Amazon Reviews 2023 (McAuley lab).**
  - Neither the dataset site (amazon-reviews-2023.github.io) nor the Hugging Face card
    (McAuley-Lab/Amazon-Reviews-2023) states a licence or usage restriction. The site's footer reads
    "Copyright © 2024, McAuley Lab".
  - The accompanying code repository, hyp1231/AmazonReviews2023, is MIT-licensed.
  - Users are asked to cite Hou et al., "Bridging Language and Items for Retrieval and Recommendation" (2024).
  - The review text and product metadata come from Amazon, and the underlying content belongs to its authors.
- **S3-Rec `Beauty.txt`.**
  - It comes from the S3-Rec repository (Zhou et al., CIKM 2020), which has no licence file.
  - It is used only as a local check and is not redistributed.

## Beauty (reproduction anchor)
Built by `sidrec.data.beauty` and written to `data/processed/beauty/sequences.parquet`. Its statistics are in
`results/r1/beauty_stats.json`.

- **Counts:** 22,363 users, 12,101 items and 198,502 interactions.
  - Sequence length: mean 8.876, median 6, minimum 5, maximum 204.
  - These are the figures in TIGER's Table 6.
- **5-core check:** no user or item has fewer than 5 interactions, so S3-Rec's 5-core filter removes nothing.
- **Check against S3-Rec:** we renumber users and items the way S3-Rec's `id_map` does. All 22,363 resulting
  sequences are identical, line for line, to the S3-Rec authors' `Beauty.txt`. That file is the dataset behind
  the S3-Rec SASRec numbers that TIGER compares against.
- **Item metadata:** `meta_Beauty.json.gz` is downloaded for R2's item text; R1 does not use it. Its lines are
  Python literals, not JSON.

## Electronics, Amazon Reviews 2023 (scale run)

### Why Electronics
The plan asks for a category with at least 5M interactions and at least 300K items with text metadata. The
official 5-core statistics (amazon-reviews-2023.github.io/data_processing/5core.html) give five qualifying
categories:

| Category | Users | Items | Ratings |
|---|---|---|---|
| **Electronics** | 1.6M | 368.2K | 15.5M |
| Kindle_Store | 892.2K | 466.6K | 16.1M |
| Books | 776.4K | 495.1K | 9.5M |
| Clothing_Shoes_and_Jewelry | 2.5M | 715.0K | 23.1M |
| Home_and_Kitchen | 2.9M | 763.6K | 28.2M |

"Unknown" is left out because it is not one category.

Electronics was chosen for these reasons:
- It clears both bars with a margin. Every one of its 368,228 items has a metadata record, and 368,211 have a
  non-empty title.
- It has the smallest catalog of the five. That keeps the full-catalog softmax baselines and the R3 embedding pass
  over every item cheapest.
- It has 42 interactions per item, against 19 for Books and 34 for Kindle_Store, so the item IDs are better
  trained.
- Its metadata is rich: 84% of items have feature bullets and 56% have a description. It is also the e-commerce
  setting that the target roles describe.

### Counts (verified by the pipeline)

| | Rows | Users | Items |
|---|---|---|---|
| All splits | 15,473,536 | 1,641,026 | 368,228 |
| Train (ts < t1) | 13,096,427 | 1,604,818 | 329,445 |
| Validation (t1 <= ts < t2) | 1,161,644 | 556,471 | 132,496 |
| Test (ts >= t2) | 1,215,465 | 469,838 | 124,121 |

- **Invariants:** no repeated (user, item) pair; every user and item has at least 5 interactions; every row's
  split agrees with the published cutoffs t1 = 1628643414042 and t2 = 1658002729837.
- **Evaluation rows with at least one earlier interaction:**
  - validation: 1,141,362, of which 185,482 are cold;
  - test: 1,199,539, of which 535,916 are cold (44.7%), covering 36,881 distinct cold items.
  - Cold means the target never appears in the train split.
  - Rows with no earlier interaction (20,282 in validation, 15,926 in test) are excluded by the protocol.
- **Official histories:** every evaluation row's history equals the `history` column of the official
  `timestamp_w_his` files, for 1,161,644 of 1,161,644 validation rows and 1,215,465 of 1,215,465 test rows.
  - The order inside a history is by timestamp, with ties in official file order.
  - Breaking ties by item ID instead would differ on 81 validation rows and 50 test rows.

### Item documents
`REC_ITEM_DOCS` builds one text document per benchmark item for the LLM embedding model. It has one line per
non-empty field, in this order.

| Field | Source in the metadata | Character limit |
|---|---|---|
| Title | `title` | 200 |
| Brand | `details.Brand`, or `store` when Brand is missing or blank | 80 |
| Categories | `categories`, joined with " > " | 200 |
| Features | `features` (the bullet points), joined with spaces | 800 |
| Description | `description`, joined with spaces | 800 |

- **Cleaning:** each field has runs of spaces, tabs, line feeds and carriage returns collapsed to one space. It is
  then trimmed, cut to its limit and trimmed again.
- **Labels:** each line starts with its label, for example "Title: ".
- **Features:** these are not in the plan's field list. They are included because 44% of Electronics items have
  no description, and the feature bullets are their main text.
- **Resulting lengths:** the documents are 11 to 2,028 characters long, with a mean of 990. No item has an empty
  document.
- **Field coverage:** 368,211 items have a title, 309,574 have features and 206,264 have a description.

## Snowflake pipeline
- **SQL:** `sql/electronics/01_setup.sql` to `05_export.sql`, run by `scripts/snowflake_pipeline.py`.
- **Snowflake setup:**
  - warehouse `PORTFOLIO_XS` (XSMALL, auto-suspend 60 s);
  - database `PORTFOLIO_LAB`;
  - session from the shared portfolio settings (`sf.session()`).
- **Objects:** every object is in schema `PORTFOLIO_LAB.GPU` with the `REC_` prefix, because the pipeline role has
  no CREATE SCHEMA privilege on the database:
  - stage `REC_STAGE`;
  - file formats `REC_CSV_GZ`, `REC_JSONL_GZ` and `REC_PARQUET`;
  - SQL function `REC_CLEAN`;
  - tables `REC_RAW_INTERACTIONS`, `REC_RAW_META`, `REC_INTERACTIONS`, `REC_USER_TIMELINE`, `REC_TRAIN_SEQ`,
    `REC_EVAL` and `REC_ITEM_DOCS`.

### Steps
1. **Stage:** PUT the three split files and the metadata file, unchanged, to `@REC_STAGE/raw/electronics/`.
   This took 187 s, mostly for the 1.3 GB metadata upload.
2. **Load:** COPY INTO `REC_RAW_INTERACTIONS` (16 s) and into `REC_RAW_META` as one VARIANT per line (67 s).
   - The split name comes from the file name.
   - `METADATA$FILE_ROW_NUMBER` keeps the official order of tied timestamps.
3. **Transform (33 s):**
   - **Filter:** keep interactions whose item has metadata. It drops 0 rows.
   - **Interactions:** each row gets its timeline position from `ROW_NUMBER()` and a cold-start flag.
   - **Sequences:** `ARRAY_AGG ... WITHIN GROUP (ORDER BY pos)` builds the training sequences and each user's
     timeline. Evaluation histories are `ARRAY_SLICE` of that timeline.
   - **Item documents:** built as described above.
   - Iterative k-core filtering is not needed. The benchmark is already 5-core, and the check step confirms it.
4. **Check:** `04_checks.sql` writes `results/r1/snowflake_checks.json`. Every `must_be_zero_*` check is 0.
5. **Export:** COPY INTO Parquet on `@REC_STAGE/export/` (7 s), then GET to `data/snowflake/electronics/<table>/`.
   - The export holds 1.2 GB in 8 files per table.
   - ARRAY columns arrive as JSON text, which `sidrec.data.tables` parses.

### Parity with the polars reference
- **Reference:** `sidrec.data.electronics` repeats every transformation in polars, from the same raw files, and
  writes `data/reference/electronics/` in 22 s.
- **Comparison:** `sidrec.data.parity` compares each table, and each split of the interactions table. It checks
  the row count and an order-independent sha256 of the content (`sidrec.data.canon`).
- **Result:** all seven comparisons match, and the row counts also equal Snowflake's own `COUNT(*)`. The result is
  in `results/r1/parity_electronics.json`.
- **Test:** `tests/test_parity.py` reruns the comparison when the data is present.

| Table | Rows | sha256 (first 16 hex) |
|---|---|---|
| interactions, train | 13,096,427 | a38d947da3082779 |
| interactions, validation | 1,161,644 | 24d78ff0124e9d15 |
| interactions, test | 1,215,465 | 30b75c7e6c6e8512 |
| train_seq | 1,604,818 | ba616b8de421b89b |
| eval_valid | 1,161,644 | 5023abde211deb93 |
| eval_test | 1,215,465 | 5b3f0961d4aab7c5 |
| item_docs | 368,228 | 7eadb672ae8e3303 |

### Warehouse credits
`scripts/warehouse_credits.py` writes `results/r1/warehouse_credits.json`.

- **Our queries:** 49 tagged queries executed for 130 s in total. They include a small probe before the run and
  the bookkeeping queries after it. That is 0.036 credits at 1 credit per hour for XSMALL.
- **Upper bound:** 0.14 credits. This assumes no other job kept the warehouse running: our busy intervals, each
  extended by the 60 s auto-suspend window.
- **Cloud services:** 0.005 credits.
- **Shared metering:** `PORTFOLIO_XS` is shared with other jobs, so its hourly `WAREHOUSE_METERING_HISTORY` rows
  include their use. At 22:54 UTC, the 22:00–23:00 UTC row read 0.216 credits for all jobs together. That
  snapshot may still lag.
- **Per-query attribution:** `QUERY_ATTRIBUTION_HISTORY` had no rows for our queries yet.

The R1 cap was 1.5 credits.

## Dev subset
- **Purpose:** small enough for tuning and debugging on one RTX 4090.
- **Builder:** `sidrec.data.dev_subset` builds it from the Snowflake export and writes
  `data/dev/electronics/<table>.parquet`. Its statistics are in `results/r1/dev_subset_stats.json`.
- **Users:** a user is kept when `int(sha256(user_id)[:8], 16) % 16 == 0`. Every interaction of a kept user is
  kept, so positions and histories are unchanged.
- **Size:** 102,587 users, 219,024 items (184,738 of them in train) and 965,008 interactions. By split that is
  816,106 train, 72,834 validation and 76,068 test.
- **Evaluation rows with history:** 71,610 in validation (23,262 cold) and 75,036 in test (41,983 cold).
- **Cold-start flags:** `is_cold` is recomputed against the subset's own train split, because a model tuned on
  the subset sees only those training items. This is why the subset's cold share is higher than in the full data.
- **Item documents:** `item_docs` holds the 219,024 items the kept users touch.
