# sidrec

A cascade recommender on public Amazon review data, built the way recent industrial systems are:

- item text embedded with an LLM embedding model;
- semantic IDs quantized from those embeddings;
- a generative retriever that decodes the next item's semantic ID;
- a sequential ranker;
- a cached-prefix serving path.

**Status:** work in progress. No model has been trained yet, and there are no model results.

## What exists so far
- **Data:** [docs/data.md](docs/data.md) lists the data sources, checksums and licence terms.
  - Amazon 2014 Beauty 5-core is the reproduction benchmark.
  - Amazon Reviews 2023 Electronics, with its official 5-core timestamp split, is the scale run.
- **Protocol:** [docs/protocol.md](docs/protocol.md) is the evaluation protocol, fixed before any training. It
  covers the splits, the cold-start definition, the metrics, the published numbers to compare against, the seeds
  and the tuning budget.
- **Pipeline:** a Snowflake SQL pipeline for Electronics, in `sql/electronics/` and run by
  `scripts/snowflake_pipeline.py`. It takes the raw files to user sequences, evaluation histories and item text
  documents, exported as Parquet.
  - A polars reference implementation is in `src/sidrec/data/electronics.py`.
  - A parity check, `src/sidrec/data/parity.py`, compares the two. The last run is in
    `results/r1/parity_electronics.json`.
- **Beauty preprocessing:** `src/sidrec/data/beauty.py` builds the leave-one-out sequences. They are checked
  against the S3-Rec authors' processed file.

## Layout
```
docs/                 data sources and licences, evaluation protocol, SHA256SUMS
sql/electronics/      Snowflake SQL: setup, load, transform, checks, export
scripts/              data download, Snowflake runner, warehouse credit estimate
src/sidrec/data/      Beauty preprocessing, the polars reference, parity check, dev subset
results/r1/           pipeline checks, query log, parity result, credits, dataset statistics
tests/                unit tests; the tests that need data skip themselves when data/ is missing
```

## Setup
```bash
uv sync
scripts/download_data.sh            # about 1.8 GB into data/raw (git-ignored), checked against docs/SHA256SUMS
uv run python -m sidrec.data.beauty
uv run python -m sidrec.data.electronics
uv run pytest
```

The Snowflake steps need the portfolio Snowflake environment on the development machine:
```bash
~/.cache/portfolio-snowflake/.venv-mljobs/bin/python scripts/snowflake_pipeline.py all
uv run python -m sidrec.data.parity
uv run python -m sidrec.data.dev_subset
```

The datasets are not redistributed here.
