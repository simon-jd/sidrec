-- Counts and invariants. The runner saves every row and fails if a check named must_be_zero_* is not 0.
SELECT 'raw_rows_' || split AS check_name, COUNT(*) AS value FROM REC_RAW_INTERACTIONS GROUP BY split
UNION ALL SELECT 'raw_meta_rows', COUNT(*) FROM REC_RAW_META
UNION ALL SELECT 'raw_meta_distinct_parent_asin', COUNT(DISTINCT v:parent_asin::STRING) FROM REC_RAW_META
UNION ALL SELECT 'interactions_' || split, COUNT(*) FROM REC_INTERACTIONS GROUP BY split
UNION ALL SELECT 'users', COUNT(DISTINCT user_id) FROM REC_INTERACTIONS
UNION ALL SELECT 'items', COUNT(DISTINCT parent_asin) FROM REC_INTERACTIONS
UNION ALL SELECT 'train_items', COUNT(DISTINCT parent_asin) FROM REC_INTERACTIONS WHERE split = 'train'
UNION ALL SELECT 'must_be_zero_dropped_without_meta',
  (SELECT COUNT(*) FROM REC_RAW_INTERACTIONS) - (SELECT COUNT(*) FROM REC_INTERACTIONS)
UNION ALL SELECT 'must_be_zero_split_disagrees_with_cutoffs', COUNT(*) FROM REC_INTERACTIONS
  WHERE split <> CASE WHEN ts < 1628643414042 THEN 'train' WHEN ts < 1658002729837 THEN 'valid' ELSE 'test' END
UNION ALL SELECT 'must_be_zero_duplicate_user_item', COUNT(*) - COUNT(DISTINCT user_id, parent_asin)
  FROM REC_INTERACTIONS
UNION ALL SELECT 'must_be_zero_users_below_5', COUNT(*) FROM (
  SELECT user_id FROM REC_INTERACTIONS GROUP BY user_id HAVING COUNT(*) < 5)
UNION ALL SELECT 'must_be_zero_items_below_5', COUNT(*) FROM (
  SELECT parent_asin FROM REC_INTERACTIONS GROUP BY parent_asin HAVING COUNT(*) < 5)
UNION ALL SELECT 'must_be_zero_pos_gaps', COUNT(*) FROM (
  SELECT user_id FROM REC_INTERACTIONS GROUP BY user_id HAVING MAX(pos) + 1 <> COUNT(*) OR MIN(pos) <> 0)
UNION ALL SELECT 'train_seq_users', COUNT(*) FROM REC_TRAIN_SEQ
UNION ALL SELECT 'eval_rows_' || split, COUNT(*) FROM REC_EVAL GROUP BY split
UNION ALL SELECT 'eval_rows_empty_history_' || split, COUNT_IF(hist_len = 0) FROM REC_EVAL GROUP BY split
UNION ALL SELECT 'eval_rows_eligible_cold_' || split, COUNT_IF(hist_len > 0 AND is_cold) FROM REC_EVAL GROUP BY split
UNION ALL SELECT 'must_be_zero_history_length_mismatch', COUNT_IF(ARRAY_SIZE(history) <> hist_len) FROM REC_EVAL
UNION ALL SELECT 'item_docs', COUNT(*) FROM REC_ITEM_DOCS
UNION ALL SELECT 'item_docs_with_title', COUNT_IF(title <> '') FROM REC_ITEM_DOCS
UNION ALL SELECT 'item_docs_with_description', COUNT_IF(description <> '') FROM REC_ITEM_DOCS
UNION ALL SELECT 'item_docs_with_features', COUNT_IF(features <> '') FROM REC_ITEM_DOCS
UNION ALL SELECT 'item_docs_empty', COUNT_IF(doc = '') FROM REC_ITEM_DOCS
UNION ALL SELECT 'item_docs_max_chars', MAX(doc_chars) FROM REC_ITEM_DOCS
UNION ALL SELECT 'must_be_zero_items_without_doc_row', COUNT(*) FROM (
  SELECT DISTINCT parent_asin FROM REC_INTERACTIONS
  MINUS SELECT parent_asin FROM REC_ITEM_DOCS)
ORDER BY check_name;
