# MetaLens: game analytics at scale on the Lichess open database

A streaming, parallel data pipeline over Lichess monthly dumps (100M+ games for recent months),
producing partitioned Parquet plus game-analytics insights (meta shift, matchmaking fairness,
time-pressure behaviour) and ML models (win probability, retention).

> Status: Week 1 (ingestion). See the project proposal for the full roadmap.

## Quickstart

```bash
git clone <your-repo-url> && cd metalens
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest -q                                   # tests (uses the committed sample)
python -m metalens benchmark                # parser vs python-chess
python -m metalens download 2016-01         # -> data/raw/   (git-ignored)
python -m metalens ingest 2016-01           # -> data/processed/month=2016-01/
```

## Where data lives (everything is repo-relative)

| Folder | Contents | In git? | Safe to delete? |
|---|---|---|---|
| `data/raw/` | downloaded `.pgn.zst` dumps | no | yes, re-download anytime |
| `data/processed/` | full Parquet output | no | yes, re-run `ingest` |
| `data/sample/` | tiny sample for tests/benchmarks | yes | keep |
| `data/aggregated/` | small aggregates the dashboard reads | yes | keep |
| `benchmarks/` | benchmark results | yes | keep |

All paths come from `src/metalens/paths.py`, anchored to the repo root. A test
(`tests/test_no_absolute_paths.py`) fails CI if any machine-specific path is committed.

## Design notes

- **Streaming:** zstd stream is read in 16 MB blocks, cut at game boundaries; nothing is written decompressed.
- **Parallel:** workers parse and write Parquet themselves (no big result pickling); in-flight chunks are bounded, so memory stays flat.
- **Idempotent + atomic:** each month is written to a temp dir and renamed; `_SUCCESS` holds a manifest (rows, games/s). Finished months are skipped.
- **Quarantine:** malformed games are saved with a reason instead of crashing the run.
- **Parser trade-off:** it extracts headers/moves/clocks without replaying moves on a board, so it is much faster than
  python-chess but does not check move legality. The benchmark reports both `read_game` (full) and `headers_only`
  so the comparison is stated honestly.

## Benchmarks

Run `python -m metalens benchmark` and paste your numbers here (games/sec, machine, Python version).
{
  "games": 20000,
  "python": "3.11.9",
  "games_per_sec": {
    "metalens_parser_1proc": 39439.8,
    "python_chess_read_game": 5828.5,
    "python_chess_headers_only": 29471.4
  }
}

## Roadmap

- [x] Week 1: streaming parser, Parquet output, tests, CI
- [ ] Week 2: Prefect orchestration, Pandera checks, Dockerfile
- [ ] Week 3: DuckDB analytics (meta shift, matchmaking, time pressure)
- [ ] Week 4: ML models, Streamlit dashboard, deploy
