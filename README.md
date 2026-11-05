# sidrec

A cascade recommender on public Amazon review data, built the way recent industrial systems are:

- item text embedded with an LLM embedding model;
- semantic IDs quantized from those embeddings;
- a generative retriever that decodes the next item's semantic ID;
- a sequential ranker;
- a cached-prefix serving path.

**Status:** work in progress. There are no model results yet.

## Setup
```bash
uv sync
scripts/download_data.sh     # about 1.8 GB into data/raw (git-ignored), checked against docs/SHA256SUMS
uv run pytest
```

The datasets are not redistributed here.
