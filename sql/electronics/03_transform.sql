-- From the raw tables to the four model inputs: interactions, training sequences, evaluation rows and item
-- documents. src/sidrec/data/electronics.py is the polars reference of every statement here.

-- 1. Interactions.
-- Filter: keep interactions whose item has a metadata record (every item does in this release; the check
-- step counts what is dropped). The split is the official one.
-- pos: the 0-based position in the user's whole timeline, ordered by timestamp, with ties in official file order.
-- is_cold: a validation or test interaction whose item never appears in the train split.
CREATE OR REPLACE TABLE REC_INTERACTIONS AS
WITH meta_items AS (
  SELECT DISTINCT v:parent_asin::STRING AS parent_asin FROM REC_RAW_META
),
r AS (
  SELECT i.*
  FROM REC_RAW_INTERACTIONS i
  JOIN meta_items m ON i.parent_asin = m.parent_asin
),
train_items AS (
  SELECT DISTINCT parent_asin FROM r WHERE split = 'train'
)
SELECT r.user_id,
       r.parent_asin,
       r.rating,
       r.ts,
       r.split,
       ROW_NUMBER() OVER (
         PARTITION BY r.user_id
         ORDER BY r.ts, DECODE(r.split, 'train', 0, 'valid', 1, 2), r.file_row
       ) - 1 AS pos,
       (r.split <> 'train' AND t.parent_asin IS NULL) AS is_cold
FROM r
LEFT JOIN train_items t ON r.parent_asin = t.parent_asin;

-- 2. Each user's whole timeline across all splits, in order (an intermediate for the evaluation rows).
CREATE OR REPLACE TABLE REC_USER_TIMELINE AS
SELECT user_id,
       ARRAY_AGG(parent_asin) WITHIN GROUP (ORDER BY pos) AS items,
       COUNT(*) AS n
FROM REC_INTERACTIONS
GROUP BY user_id;

-- 3. Training sequences: each user's train-split interactions, in order.
CREATE OR REPLACE TABLE REC_TRAIN_SEQ AS
SELECT user_id,
       ARRAY_AGG(parent_asin) WITHIN GROUP (ORDER BY pos) AS items,
       ARRAY_AGG(ts) WITHIN GROUP (ORDER BY pos) AS timestamps,
       COUNT(*) AS n
FROM REC_INTERACTIONS
WHERE split = 'train'
GROUP BY user_id;

-- 4. Evaluation rows: one per validation or test interaction. The history is every earlier interaction of the
-- user from any split (the convention of the official timestamp_w_his files); hist_len = 0 rows are kept here
-- and excluded at evaluation time (docs/protocol.md).
CREATE OR REPLACE TABLE REC_EVAL AS
SELECT i.split,
       i.user_id,
       i.parent_asin AS target,
       i.rating,
       i.ts,
       i.pos AS hist_len,
       i.is_cold,
       ARRAY_SLICE(t.items, 0, i.pos) AS history
FROM REC_INTERACTIONS i
JOIN REC_USER_TIMELINE t ON i.user_id = t.user_id
WHERE i.split IN ('valid', 'test');

-- 5. Item text documents for every item in the benchmark.
-- brand: details.Brand when present, otherwise the store name. Field limits in characters:
-- title 200, brand 80, categories 200, features 800, description 800.
CREATE OR REPLACE TABLE REC_ITEM_DOCS AS
WITH m AS (
  SELECT v:parent_asin::STRING AS parent_asin,
         REC_CLEAN(v:title::STRING, 200) AS title,
         COALESCE(NULLIF(REC_CLEAN(v:details:Brand::STRING, 80), ''), REC_CLEAN(v:store::STRING, 80)) AS brand,
         REC_CLEAN(ARRAY_TO_STRING(v:categories::ARRAY, ' > '), 200) AS categories,
         REC_CLEAN(ARRAY_TO_STRING(v:features::ARRAY, ' '), 800) AS features,
         REC_CLEAN(ARRAY_TO_STRING(v:description::ARRAY, ' '), 800) AS description
  FROM REC_RAW_META
  WHERE v:parent_asin::STRING IN (SELECT DISTINCT parent_asin FROM REC_INTERACTIONS)
),
d AS (
  SELECT m.*,
         ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(
           IFF(title <> '', 'Title: ' || title, NULL),
           IFF(brand <> '', 'Brand: ' || brand, NULL),
           IFF(categories <> '', 'Categories: ' || categories, NULL),
           IFF(features <> '', 'Features: ' || features, NULL),
           IFF(description <> '', 'Description: ' || description, NULL)
         ), CHR(10)) AS doc
  FROM m
)
SELECT d.*, LENGTH(doc) AS doc_chars
FROM d;
