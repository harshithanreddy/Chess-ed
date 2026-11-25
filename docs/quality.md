# Data quality gate

`python -m metalens validate 2016-01` checks an ingested month in two layers.

**Full-month SQL checks (DuckDB):** null/duplicate game ids, invalid results, score vs result mismatch,
Elo outside 100-4000, negative ply count (errors); dates outside the month, negative clocks,
unknown speed, zero-move games (warnings).

**Integrity checks:** Parquet row count must equal the ingest manifest; quarantine ratio must be under 1%.

**Pandera contract:** types, ranges and uniqueness on a 100k-row random sample.

Errors stop the pipeline. Reports are saved to `data/aggregated/quality/<month>.json` (committed),
so results remain visible after raw data is deleted.
