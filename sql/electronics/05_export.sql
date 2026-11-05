-- Unload the model inputs as Parquet. ARRAY columns arrive as JSON text; src/sidrec/data/tables.py parses them.
REMOVE @REC_STAGE/export/;

COPY INTO @REC_STAGE/export/interactions/
FROM (SELECT user_id, parent_asin, rating, ts, split, pos, is_cold FROM REC_INTERACTIONS)
FILE_FORMAT = (FORMAT_NAME = REC_PARQUET) HEADER = TRUE MAX_FILE_SIZE = 268435456;

COPY INTO @REC_STAGE/export/train_seq/
FROM (SELECT user_id, items, timestamps, n FROM REC_TRAIN_SEQ)
FILE_FORMAT = (FORMAT_NAME = REC_PARQUET) HEADER = TRUE MAX_FILE_SIZE = 268435456;

COPY INTO @REC_STAGE/export/eval_valid/
FROM (SELECT user_id, target, rating, ts, hist_len, is_cold, history FROM REC_EVAL WHERE split = 'valid')
FILE_FORMAT = (FORMAT_NAME = REC_PARQUET) HEADER = TRUE MAX_FILE_SIZE = 268435456;

COPY INTO @REC_STAGE/export/eval_test/
FROM (SELECT user_id, target, rating, ts, hist_len, is_cold, history FROM REC_EVAL WHERE split = 'test')
FILE_FORMAT = (FORMAT_NAME = REC_PARQUET) HEADER = TRUE MAX_FILE_SIZE = 268435456;

COPY INTO @REC_STAGE/export/item_docs/
FROM (SELECT parent_asin, title, brand, categories, features, description, doc, doc_chars FROM REC_ITEM_DOCS)
FILE_FORMAT = (FORMAT_NAME = REC_PARQUET) HEADER = TRUE MAX_FILE_SIZE = 268435456;
