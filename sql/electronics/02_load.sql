-- Load the staged raw files as they were downloaded.

-- The official 5-core absolute-timestamp split: one file per split. The split name comes from the file name,
-- and the row number inside the file keeps the official order of tied timestamps.
CREATE OR REPLACE TABLE REC_RAW_INTERACTIONS (
  user_id     STRING        NOT NULL,
  parent_asin STRING        NOT NULL,
  rating      FLOAT,
  ts          NUMBER(18, 0) NOT NULL,
  split       STRING        NOT NULL,
  file_row    NUMBER        NOT NULL
);

COPY INTO REC_RAW_INTERACTIONS
FROM (
  SELECT $1, $2, $3, $4,
         REGEXP_SUBSTR(METADATA$FILENAME, '(train|valid|test)[.]csv[.]gz$', 1, 1, 'e', 1),
         METADATA$FILE_ROW_NUMBER
  FROM @REC_STAGE/raw/electronics/
)
PATTERN = '.*Electronics[.](train|valid|test)[.]csv[.]gz'
FILE_FORMAT = (FORMAT_NAME = REC_CSV_GZ)
ON_ERROR = ABORT_STATEMENT;

-- Item metadata, one JSON document per line, kept whole.
CREATE OR REPLACE TABLE REC_RAW_META (v VARIANT);

COPY INTO REC_RAW_META
FROM @REC_STAGE/raw/electronics/
PATTERN = '.*meta_Electronics[.]jsonl[.]gz'
FILE_FORMAT = (FORMAT_NAME = REC_JSONL_GZ)
ON_ERROR = ABORT_STATEMENT;
