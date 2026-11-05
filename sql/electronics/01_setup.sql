-- Objects for the Electronics pipeline.
-- The pipeline role cannot create schemas in PORTFOLIO_LAB, so every object lives in the schema it owns
-- (PORTFOLIO_LAB.GPU) under the REC_ prefix.

CREATE STAGE IF NOT EXISTS REC_STAGE
  COMMENT = 'sidrec: raw Amazon Reviews 2023 Electronics files and the Parquet export';

CREATE OR REPLACE FILE FORMAT REC_CSV_GZ
  TYPE = CSV SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"' COMPRESSION = GZIP;

CREATE OR REPLACE FILE FORMAT REC_JSONL_GZ
  TYPE = JSON COMPRESSION = GZIP;

CREATE OR REPLACE FILE FORMAT REC_PARQUET
  TYPE = PARQUET COMPRESSION = SNAPPY;

-- Text cleaning shared by every item-document field: collapse runs of space, tab, LF and CR into one space,
-- trim, cut to n characters, and trim the end again. The polars reference applies exactly this.
CREATE OR REPLACE FUNCTION REC_CLEAN(s STRING, n NUMBER)
  RETURNS STRING
  AS $$
    RTRIM(LEFT(TRIM(REGEXP_REPLACE(COALESCE(s, ''), '[ ' || CHR(9) || CHR(10) || CHR(13) || ']+', ' '), ' '), n), ' ')
  $$;
