#   chess-ed: game analytics at scale on the Lichess open database

A streaming, parallel data pipeline over Lichess monthly dumps, producing partitioned
Parquet plus game-analytics insights (opening meta shift, matchmaking fairness,
time-pressure behaviour) and ML models (win/draw/loss probability, player retention
with SHAP), served from a themed Streamlit dashboard.

> Status: core pipeline, analytics, ML models, and dashboard are built and verified
> end to end on real data (4 ingested months, 2016-2024). See "Results" below for
> numbers from this project's own runs. Deployment is in progress.

## Quickstart

```bash
git clone <your-repo-url> && cd metalens
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

pytest -q                                   # tests (uses the committed sample)
python -m metalens benchmark                # parser vs python-chess
python -m metalens run 2016-01 --delete-raw # download + ingest + validate, one command
python -m metalens analyze --speed blitz    # -> data/aggregated/*.parquet (dashboard inputs)
streamlit run dashboard/app.py
```

## Results (from this project's own runs)

**Ingestion** (2016-01, 11 workers): 4,770,357 games, 0 quarantined, 64.25s,
**74,248 games/sec**. Data-quality gate passed with 0 errors; the only warnings were
expected ones (games spanning a UTC month boundary, games aborted before move 1).

**Parser benchmark** vs. `python-chess` (20,000 games, Python 3.11.9):

| Parser | games/sec | vs. MetaLens |
|---|---:|---:|
| MetaLens (this project) | 35,174.6 | -- |
| python-chess, full parse (`read_game`) | 6,214.2 | **5.7x slower** |
| python-chess, headers only (`skip_game`) | 31,256.5 | 1.1x slower |

The honest read: most of the speedup comes from skipping move-legality replay
entirely (MetaLens's parser and python-chess's `headers_only` mode both do this) --
not from out-parsing python-chess at an equivalent task. Against the directly
comparable "full legality-checked parse," MetaLens is **5.7x faster**.

**Scale**: across the 4 ingested months (2016-01, 2019-01, 2024-01, 2024-02),
2024-01 alone contains **~98.3M rated games** across all time controls (sum of the
per-speed counts in `data/aggregated/clock_coverage_by_month.parquet`).

**Win/draw/loss model** (LightGBM, time-based train/test split, 107.8M rated blitz
games: 86.2M train / 21.6M test):

| Model | Log loss |
|---|---:|
| LightGBM | 0.8250 |
| Logistic regression (one-hot ECO) | 0.8266 |
| Majority-class baseline | 0.8420 |

A single-month-only run (2024-01, 37.4M train / 9.4M test) gave log loss 0.8254 --
essentially identical to the 8-years-pooled result, suggesting the rating-to-outcome
relationship has been stable from 2016 to 2024 rather than drifting with the meta.

**Player retention model** (LightGBM + SHAP, 1.44M players, Jan 2024 features ->
Feb 2024 return label): **AUC 0.819**, average precision **0.939** against a 76.8%
base return rate. SHAP found `n_active_days` (days played per month) is by far the
strongest predictor of return (mean |SHAP| 1.57, ~8x the next-ranked feature) --
stronger than total games played or win rate.

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
  python-chess but does not check move legality (see the benchmark table above for the honest comparison).
- **Data integrity caught and fixed:** some openings share a display name across different ECO
  codes (e.g. several B1x lines can all render as plain "Caro-Kann Defense"). An early version of
  the dashboard's opening-meta chart grouped only by name, which left duplicate (month, name) rows
  that Altair drew as separate points at the same x position -- visually, a line plunging to
  near-zero and snapping back. Fixed by grouping on (month, opening) and summing `pick_rate` (correct,
  since it shares a denominator within a month) or taking a games-weighted average for `white_win_rate`
  (correct, since win rate is not additive). See `scripts/verify_opening_chart.py` for the
  verification tool this produced.

## Roadmap

- [x] Week 1: streaming parser, Parquet output, tests, CI
- [x] Week 2: Prefect orchestration, Pandera checks, Dockerfile
- [x] Week 3: DuckDB analytics (meta shift, matchmaking, time pressure)
- [x] Week 4: ML models (win probability, retention + SHAP), themed Streamlit dashboard
- [ ] Deploy to Streamlit Community Cloud

## ML models

```bash
python -m metalens train-win --speed blitz
python -m metalens train-retention 2024-01   # needs 2024-01 AND 2024-02 ingested
```

Both write metrics JSON + small Parquet/model files into `data/aggregated/models/`
(committed, used by the dashboard).

- **Win/draw/loss model**: LightGBM multiclass, native categorical handling for ECO,
  benchmarked against a one-hot logistic-regression baseline and a majority-class
  baseline, with a time-based train/test split (train on earlier games, test on later
  ones) to avoid leaking future rating information.
- **Retention model**: player-level features (games played, active days, rating
  trend, longest losing streak, rated-game share) predicting whether a player returns
  in the next calendar month, plus SHAP values for direction of effect (see Results
  above). Requires two consecutive ingested months.

## Dashboard

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

Reads only `data/aggregated/` -- no dependency on raw downloads or the full Parquet
output, so it works after you delete those, and deploys straight from the repo on
Streamlit Community Cloud. Theming lives in `dashboard/theme.py` (colors, fonts,
header banner) and `dashboard/charts.py` (themed Altair charts replacing Streamlit's
default palette); `.streamlit/config.toml` sets the base theme and must be committed
for a deployed app to pick it up.

**Live app:** _not yet deployed -- add the link here once deployment is complete._

## Diagnostics

`scripts/verify_opening_chart.py --band <rating_band>` checks the exact numbers
behind the Opening Meta chart against the raw aggregated Parquet, and flags any
same-root-name openings that collide within the current top-N for a given band
(informational -- not auto-merged, since most are genuinely different lines).
